#!/usr/bin/env python3
"""Exp E: can CTU-13 be turned into a backdoor by changing only which features are picked?

PRE-REGISTERED BEFORE THE RUN. Scripts 40/42/43/44 established that one recipe is
evasion on CTU-13 and a backdoor on UNSW-NB15, that trigger size does not explain it,
and that the trigger is aligned with CTU's clean decision and orthogonal to UNSW's.
That is a correlation across two datasets. This is the controlled version: hold the
dataset, constraints, magnitude, poison rows and seeds fixed, and vary ONLY which
sixteen features carry the watermark.

If a low-attribution trigger stops evading (the clean model is unmoved) but still
installs (the poisoned model flips attack rows), then alignment is what decides
backdoor-versus-evasion, demonstrated inside one dataset.

THE TRAP, and why the pool is filtered. Of 577 equality-eligible CTU features, 123
have sd == 0. Their SHAP attribution is exactly 0.0, so they tie at the very bottom
of the ranking and a naive "bottom 16" lands on all-constant columns with probability
1. build_trigger then substitutes sd=1.0, so the watermark is "mean plus six raw
units" -- and sklearn sets scale_=1.0 for those same columns, so the STANDARDISED
displacement is exactly +6.0 and a magnitude guard in sigma units passes at retention
1.000. The backdoor then installs trivially, because those weights see gradient only
from identically-patterned poisoned rows, which is a perfect separator. The run would
return CONVERTED on every seed with every guard green, having measured "an MLP can be
backdoored through dead input columns". So the pool excludes them, and A5 runs the
degenerate case deliberately as a reported negative control.

THE QUANTITY THE EARLIER DECOMPOSITION LACKS. Script 40 computes the backdoor
component against a CLEAN-model baseline. At high poison rate with an unaligned
trigger the poisoned model may degrade generally and call unstamped attack rows
benign, which would read as a backdoor. So the primary here is
B_marginal = poisoned_stamped - poisoned_PLAIN, and U = poisoned_plain - clean_plain
is reported as the contamination term. One extra forward pass. A paper arguing that a
one-forward-pass control belongs in every evaluation cannot omit it from its own.

DECISION RULE, fixed before the run.
  per arm, evasion:   EVASIVE if clean_stamped >= 0.50; NON-EVASIVE if <= 0.15
  per cell, backdoor: INSTALLED if B_marginal >= 0.50 AND poisoned_stamped >= 0.80
                      ABSENT    if B_marginal <= 0.15
  VOID a cell if U > 0.15, clean-accuracy drop > 0.02, constraint validity < 0.9999,
       or its retention differs from A1's by more than 0.15.
  CONVERTED requires A1 EVASIVE, A2 NON-EVASIVE, A2 INSTALLED at rate 0.10,
       A1 B_marginal <= 0.15 at 0.005, and A2 alignment dP <= 0.05.
  ALIGNMENT-REFUTED if A2 is EVASIVE or its dP >= 0.45.
  NOT-CONVERTED if A2 is NON-EVASIVE and ABSENT at every rate.
  Anything else INCONCLUSIVE, printed with its numbers, never rounded toward either.

Verdict cells are named in advance (0.005 and 0.10) so the clearing rate cannot be
chosen after the fact. Any headline clause additionally needs 5/5 seed unanimity.

Run:  python scripts/48_alignment_manipulation.py [--write-manifest]
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.data import apply_standardiser
from src.models import attack_success_rate, clean_accuracy, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.tb_vendor.constraints_numeric import check_constraints
from src.torch_utils import batched_apply
from src.trigger import (apply_trigger, build_trigger, equality_features,
                         raw_trigger_stats, shap_abs_matrix, shap_rank_features)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _m2_common import get_config, load_setup  # noqa: E402

COST, N_SIGMA, TARGET = 16, 6.0, config.ATTACK_TARGET
ALL_RATES = (0.005, 0.01, 0.02, 0.05, 0.10)
SPARSE_RATES = (0.005, 0.05, 0.10)
VERDICT_RATES = (0.005, 0.10)
EVASIVE_MIN, NON_EVASIVE_MAX = 0.50, 0.15
INSTALLED_MIN, ABSENT_MAX, ASR_MIN = 0.50, 0.15, 0.80
U_MAX, ACC_DROP_MAX, RETENTION_BAND = 0.15, 0.02, 0.15
DP_ORTHOGONAL_MAX, DP_MARGIN_MAX = 0.05, 0.45
MIN_HEADROOM_SIGMA, MIN_SINGLE_RETENTION = 6.0, 0.60
CKPT = config.RESULTS / "alignment_manipulation.checkpoint.json"


def fingerprint(cfg):
    return hashlib.sha256(json.dumps({
        "cost": COST, "sigma": N_SIGMA, "rates": list(ALL_RATES),
        "seeds": list(cfg["seeds"]), "epochs": cfg["mlp_epochs"]}, sort_keys=True
    ).encode()).hexdigest()[:16]


def load_ckpt(fp):
    if CKPT.exists():
        b = json.loads(CKPT.read_text())
        if b.get("fingerprint") == fp:
            return b.get("cells", {})
        print("checkpoint fingerprint differs; ignoring")
    return {}


def save_ckpt(fp, cells):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"fingerprint": fp, "cells": cells}))
    tmp.replace(CKPT)


def p_target(model, x_std, device):
    model = model.to(device).eval()
    x = torch.as_tensor(np.asarray(x_std), dtype=torch.float32)
    return float(np.concatenate(batched_apply(
        lambda xb: torch.softmax(model(xb), 1)[:, TARGET].detach().cpu().numpy(),
        x, device), axis=0).mean())


def displacement(stamped, idx, mu, sd):
    x = np.asarray(stamped, float)
    return float(np.mean([(x[:, j] - mu[j]) / (sd[j] if sd[j] > 0 else 1.0) for j in idx]))


def make_trigger(idx, mu, sd):
    return {"indices": list(idx),
            "values": [float(mu[i] + N_SIGMA * (sd[i] if sd[i] > 0 else 1.0)) for i in idx],
            "project": True}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config(smoke=False)
    fp = fingerprint(cfg)
    cells = load_ckpt(fp)
    S = load_setup(cfg)
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    feats, cons, bounds = S["features"], S["constraints"], S["bounds"]
    mu, sd, eq = raw_trigger_stats(x_tr_raw, cons)
    lo_b, hi_b = bounds
    x_bot_raw = S["x_te_raw"][S["y_te"] == 1]
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])

    # ---- pool construction (G1). Excludes dead columns and anything the box clips.
    eligible = [i for i in range(len(feats)) if str(feats[i]) not in eq]
    dead = [i for i in eligible if sd[i] <= 0]
    pool = [i for i in eligible if sd[i] > 0 and (hi_b[i] - mu[i]) / sd[i] >= MIN_HEADROOM_SIGMA]
    print(f"eligible {len(eligible)}   dead (sd==0) {len(dead)}   after headroom {len(pool)}")
    keep = []
    for i in pool:
        t1 = make_trigger([i], mu, sd)
        r = displacement(apply_trigger(x_bot_raw[:64], t1, cons, feats, bounds), [i], mu, sd) / N_SIGMA
        if r >= MIN_SINGLE_RETENTION:
            keep.append(i)
    pool = keep
    print(f"pool after single-feature retention >= {MIN_SINGLE_RETENTION}: {len(pool)}")

    ref_idx = {int(r["seed"]): list(r["trigger_indices"])
               for r in json.loads((config.RESULTS / "nc_masks.json").read_text())["tabular"]}

    t0 = time.time()
    for seed in cfg["seeds"]:
        x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
        clean = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean, x_tr_raw, S["scaler"], target=TARGET,
                                     kind="mlp", device=device)
        imp = shap_abs_matrix(clean, x_tr_raw, S["scaler"], target=TARGET,
                              kind="mlp", device=device).mean(axis=0)
        by_rank = [i for i in ranking if i in set(pool)]
        arms = {
            "A1_shap_top16":   list(build_trigger(ranking, COST, x_tr_raw, cons, feats,
                                                  bounds, stats=(mu, sd, eq))[0]["indices"]),
            "A2_shap_bottom16": by_rank[-COST:],
            "A3_shap_mid16":    by_rank[len(by_rank)//2 - COST//2: len(by_rank)//2 + COST//2],
            "A4_random16":      sorted(np.random.default_rng(seed).choice(
                                    [i for i in pool if i not in set(by_rank[:COST])],
                                    size=COST, replace=False).tolist()),
            "A5_degenerate":    dead[-COST:],
        }
        assert arms["A1_shap_top16"] == ref_idx[seed], f"seed {seed}: A1 differs from committed"  # G3

        a_clean_plain = attack_success_rate(
            clean, apply_standardiser(S["scaler"], x_bot_raw), TARGET, device)

        for arm, idx in arms.items():
            rates = ALL_RATES if arm in ("A1_shap_top16", "A2_shap_bottom16") else (
                SPARSE_RATES if arm != "A5_degenerate" else (0.005,))
            trig = make_trigger(idx, mu, sd)
            # G2: pre-projection displacement must be exactly six sigma
            pre = [(trig["values"][k] - mu[i]) / (sd[i] if sd[i] > 0 else 1.0)
                   for k, i in enumerate(idx)]
            assert max(abs(p - N_SIGMA) for p in pre) < 1e-6, f"{arm}: pre-projection sigma off"
            stamped = apply_trigger(x_bot_raw, trig, cons, feats, bounds)
            valid = float(check_constraints(stamped, cons, feats).mean())
            st_std = apply_standardiser(S["scaler"], stamped)
            a_clean_stamped = attack_success_rate(clean, st_std, TARGET, device)
            dp = p_target(clean, st_std, device) - p_target(
                clean, apply_standardiser(S["scaler"], x_bot_raw), device)
            ret = displacement(stamped, idx, mu, sd) / N_SIGMA
            share = float(imp[list(idx)].sum() / imp.sum())

            for rate in rates:
                key = f"{arm}|{rate}|{seed}"
                if key in cells:
                    continue
                x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
                    x_tr_raw, y_tr, trig, rate, S["scaler"], target=TARGET, seed=seed,
                    constraints=cons, feature_names=feats, bounds=bounds)
                pois = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)
                a_pois_plain = attack_success_rate(
                    pois, apply_standardiser(S["scaler"], x_bot_raw), TARGET, device)
                a_pois_stamped = attack_success_rate(pois, st_std, TARGET, device)
                cells[key] = {
                    "arm": arm, "rate": rate, "seed": seed,
                    "n_poison": int(len(poison_idx)),
                    "poison_idx_hash": hashlib.sha256(poison_idx.tobytes()).hexdigest()[:12],
                    "a_clean_plain": float(a_clean_plain),
                    "a_clean_stamped": float(a_clean_stamped),
                    "a_pois_plain": float(a_pois_plain),
                    "a_pois_stamped": float(a_pois_stamped),
                    "B_marginal": float(a_pois_stamped - a_pois_plain),
                    "B_legacy": float(a_pois_stamped - a_clean_stamped),
                    "U": float(a_pois_plain - a_clean_plain),
                    "clean_acc": float(clean_accuracy(pois, x_te_std, S["y_te"], device)),
                    "constraint_valid": valid, "retention": ret,
                    "delta_p": dp, "attribution_share": share,
                }
                save_ckpt(fp, cells)
            print(f"  seed {seed} {arm:18s} clean_stamped {a_clean_stamped:.4f} "
                  f"dP {dp:+.4f} ret {ret:.3f} share {share:.4f}  [{time.time()-t0:.0f}s]")

    # ---------- verdict
    def rows(arm, rate=None):
        return [c for c in cells.values() if c["arm"] == arm and (rate is None or abs(c["rate"] - rate) < 1e-9)]

    def mean(rs, k):
        return float(np.mean([r[k] for r in rs])) if rs else float("nan")

    a1_ret = mean(rows("A1_shap_top16", 0.005), "retention")
    summary = {}
    for arm in ("A1_shap_top16", "A2_shap_bottom16", "A3_shap_mid16", "A4_random16", "A5_degenerate"):
        rs = rows(arm)
        if not rs:
            continue
        cs = mean(rs, "a_clean_stamped")
        ev = "EVASIVE" if cs >= EVASIVE_MIN else "NON-EVASIVE" if cs <= NON_EVASIVE_MAX else "INCONCLUSIVE-EVASION"
        per_rate = {}
        for rate in sorted({r["rate"] for r in rs}):
            rr = rows(arm, rate)
            B, A, U = mean(rr, "B_marginal"), mean(rr, "a_pois_stamped"), mean(rr, "U")
            void = (U > U_MAX or mean(rr, "constraint_valid") < 0.9999
                    or abs(mean(rr, "retention") - a1_ret) > RETENTION_BAND)
            bd = ("VOID" if void else
                  "INSTALLED" if (B >= INSTALLED_MIN and A >= ASR_MIN) else
                  "ABSENT" if B <= ABSENT_MAX else "INCONCLUSIVE-BACKDOOR")
            per_rate[str(rate)] = {"B_marginal": B, "a_pois_stamped": A, "U": U, "verdict": bd}
        summary[arm] = {"clean_stamped": cs, "evasion_verdict": ev,
                        "delta_p": mean(rs, "delta_p"), "retention": mean(rs, "retention"),
                        "attribution_share": mean(rs, "attribution_share"), "by_rate": per_rate}

    a1, a2 = summary.get("A1_shap_top16", {}), summary.get("A2_shap_bottom16", {})
    a2_dp = a2.get("delta_p", 1.0)
    if a2.get("evasion_verdict") == "EVASIVE" or a2_dp >= DP_MARGIN_MAX:
        verdict = "ALIGNMENT-REFUTED"
    elif (a1.get("evasion_verdict") == "EVASIVE" and a2.get("evasion_verdict") == "NON-EVASIVE"
          and a2.get("by_rate", {}).get("0.1", {}).get("verdict") == "INSTALLED"
          and a1.get("by_rate", {}).get("0.005", {}).get("B_marginal", 1) <= ABSENT_MAX
          and a2_dp <= DP_ORTHOGONAL_MAX):
        verdict = "CONVERTED"
    elif (a2.get("evasion_verdict") == "NON-EVASIVE"
          and all(v["verdict"] == "ABSENT" for v in a2.get("by_rate", {}).values())):
        verdict = "NOT-CONVERTED"
    else:
        verdict = "INCONCLUSIVE"

    print("\n=== Exp E (rule fixed before the run)")
    for arm, s in summary.items():
        print(f"  {arm:18s} clean_stamped {s['clean_stamped']:.4f} [{s['evasion_verdict']}]  "
              f"dP {s['delta_p']:+.4f}  share {s['attribution_share']:.4f}  ret {s['retention']:.3f}")
        for rate, v in s["by_rate"].items():
            print(f"      rate {rate:<6} B_marginal {v['B_marginal']:+.4f}  "
                  f"pois_stamped {v['a_pois_stamped']:.4f}  U {v['U']:+.4f}  [{v['verdict']}]")
    print(f"  VERDICT: {verdict}")

    if args.write_manifest:
        p = config.RESULTS / "alignment_manipulation.json"
        p.write_text(json.dumps({
            "criterion": {"evasive_min": EVASIVE_MIN, "non_evasive_max": NON_EVASIVE_MAX,
                          "installed_min": INSTALLED_MIN, "absent_max": ABSENT_MAX,
                          "verdict_rates": list(VERDICT_RATES)},
            "pool_size": len(pool), "n_dead_excluded": len(dead),
            "verdict": verdict, "summary": summary, "cells": list(cells.values())}, indent=2) + "\n")
        print(f"\nwrote {p.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
