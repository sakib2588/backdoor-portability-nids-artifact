#!/usr/bin/env python3
"""Exp C: how much of ASR 1.0 is the backdoor, and how much is plain evasion?

PRE-REGISTERED BEFORE THE RUN. The paper reports attack-success rate on poisoned
victims and never asks what a CLEAN victim does with the same watermarked flows.
If a clean model also calls them benign, ASR is measuring test-time evasion that
needs no poisoning at all, and "clean-label backdoor" is the wrong description of
the result. This is the first question a competent reviewer asks and nothing on
disk answers it.

Closed three-way decomposition, per seed, one clean model, the same 407 botnet
test rows the grid uses:

    A_clean_plain     fraction of the UN-watermarked 407 the clean model calls
                      benign. Read from results/clean_mlp_test_recall.json
                      (1 - 0.9577 = 0.0423); NOT recomputed, so there is exactly
                      one such number in the project.
    A_clean_stamped   fraction of the WATERMARKED 407 the clean model calls
                      benign. This is the new measurement.
    A_poisoned_stamped   the committed grid value, results/detectors.json.

    E = A_clean_stamped - A_clean_plain      evasion component
    B = A_poisoned_stamped - A_clean_stamped backdoor component

A_clean_plain + E + B = A_poisoned_stamped by construction, so the split is
auditable rather than asserted.

DECISION RULE, fixed before any result was seen, taken at cost 16:
    CLEAN               A_clean_stamped <= 0.15
    CONTAMINATED        A_clean_stamped >= 0.50
    INCONCLUSIVE-MIXED  anything between, reported with its bootstrap CI and
                        never rounded toward either.

Swept over the full trigger-cost axis so the crossover -- the cost at which the
backdoor component B stops being zero -- is located rather than inferred from two
points. The cost-16 indices are checked against the committed reference; smaller
costs must be that list's prefix, since build_trigger takes eligible[:cost] from a
single ranking.

The unconditioned rate is primary. Restricting to rows the clean model already
classifies correctly sounds more principled and makes the number incomparable to
the committed ASR of 1.0, which is unconditioned; the conditioned variant is
emitted under its own key as a secondary and the verdict never reads it.

GUARDS, both of which abort the seed rather than warn:
  * Trigger identity. build_trigger consumes a SHAP ranking derived from that
    seed's clean MLP, so a different epoch count or device silently yields a
    different 16 features -- a different attack, compared against a committed
    number for another one. The cost-16 indices must equal, element-wise and in
    order, the trigger_indices stored per seed in results/nc_masks.json. The
    cost-8 indices must be that list's first eight, since build_trigger takes
    eligible[:cost] from one ranking.
  * Projection. apply_trigger without constraints/feature_names/bounds stamps the
    unprojected six-sigma watermark, a strictly larger perturbation that would
    manufacture CONTAMINATED out of nothing. Every stamped row is constraint-
    checked and the validity fraction is recorded.

Run:  python scripts/40_clean_model_stamped_asr.py [--write-manifest]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import apply_standardiser
from src.models import attack_success_rate, predict_labels, train_mlp
from src.tb_vendor.constraints_numeric import check_constraints
from src.trigger import apply_trigger, build_trigger, raw_trigger_stats, shap_rank_features

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _m2_common import get_config, load_setup  # noqa: E402

RATE = 0.005
COSTS = (1, 2, 4, 8, 16)
VERDICT_COST = 16
TARGET = config.ATTACK_TARGET
CLEAN_THRESH, CONTAMINATED_THRESH = 0.15, 0.50
CKPT = config.RESULTS / "clean_model_stamped_asr.checkpoint.json"


def fingerprint(cfg) -> str:
    payload = json.dumps({"rate": RATE, "costs": list(COSTS), "target": TARGET,
                          "epochs": cfg["mlp_epochs"], "seeds": list(cfg["seeds"])},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_ckpt(fp):
    if CKPT.exists():
        blob = json.loads(CKPT.read_text())
        if blob.get("fingerprint") == fp:
            return blob.get("seeds", {})
        print("checkpoint fingerprint differs from this config; ignoring it")
    return {}


def save_ckpt(fp, seeds):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"fingerprint": fp, "seeds": seeds}))
    tmp.replace(CKPT)


def reference_trigger_indices() -> dict[int, list[int]]:
    """Per-seed cost-16 trigger indices committed by scripts/10_nc_masks.py."""
    blob = json.loads((config.RESULTS / "nc_masks.json").read_text())
    if list(blob["cell"]) != [RATE, 16]:
        raise SystemExit(f"nc_masks.json is for cell {blob['cell']}, not [{RATE}, 16]")
    return {int(r["seed"]): list(r["trigger_indices"]) for r in blob["tabular"]}


def committed_poisoned_asr() -> dict[tuple[int, int], float]:
    grid = json.loads((config.RESULTS / "detectors.json").read_text())["main_grid"]
    return {(r["seed"], r["cost"]): r["asr"]
            for r in grid if abs(r["rate"] - RATE) < 1e-9}


def clean_plain_by_seed() -> dict[int, float]:
    """1 - botnet recall, from the one place this project already records it."""
    blob = json.loads((config.RESULTS / "clean_mlp_test_recall.json").read_text())
    return {int(r["seed"]): 1.0 - r["botnet_recall"] for r in blob["per_seed"]}


def run_seed(seed, S, cfg, device, ref_idx):
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    stats = raw_trigger_stats(x_tr_raw, constraints)

    x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
    clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
    ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET,
                                 kind="mlp", device=device)

    x_bot_raw = S["x_te_raw"][S["y_te"] == 1]
    out = {"seed": seed, "n_botnet_test": int(len(x_bot_raw)), "costs": {}}

    for cost in COSTS:
        realiz, _ = build_trigger(ranking, cost, x_tr_raw, constraints, features,
                                  bounds, stats=stats)
        got = list(realiz["indices"])
        want = ref_idx[seed][:cost]
        if got != want:
            raise SystemExit(
                f"seed {seed} cost {cost}: trigger indices differ from the committed run.\n"
                f"  got  {got}\n  want {want}\n"
                "This is a different attack; the comparison against the committed ASR "
                "would be meaningless. Check mlp_epochs and device against scripts/05.")

        x_bot_trig = apply_trigger(x_bot_raw, realiz, constraints, features, bounds)
        valid = check_constraints(x_bot_trig, constraints, features)
        stamped_std = apply_standardiser(S["scaler"], x_bot_trig)

        a_stamped = attack_success_rate(clean_mlp, stamped_std, TARGET, device)
        plain_pred = predict_labels(clean_mlp, apply_standardiser(S["scaler"], x_bot_raw), device)
        correct = plain_pred == 1
        stamped_pred = predict_labels(clean_mlp, stamped_std, device)
        conditioned = float((stamped_pred[correct] == TARGET).mean()) if correct.any() else float("nan")

        out["costs"][str(cost)] = {
            "a_clean_stamped": float(a_stamped),
            "a_clean_stamped_conditioned_secondary": conditioned,
            "constraint_valid_frac": float(valid.mean()),
        }
    return out


def bootstrap_ci(vals, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config(smoke=False)
    fp = fingerprint(cfg)
    done = load_ckpt(fp)
    ref_idx = reference_trigger_indices()
    pois = committed_poisoned_asr()
    plain = clean_plain_by_seed()

    print(f"device={device}  fingerprint={fp}  cached seeds: {sorted(done)}")
    S = load_setup(cfg)
    t0 = time.time()
    for seed in cfg["seeds"]:
        if str(seed) in done:
            print(f"--- seed {seed}: cached")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s)")
        done[str(seed)] = run_seed(seed, S, cfg, device, ref_idx)
        save_ckpt(fp, done)
        r = done[str(seed)]["costs"][str(VERDICT_COST)]
        print(f"    A_clean_stamped(cost {VERDICT_COST}) = {r['a_clean_stamped']:.4f}"
              f"   constraint-valid {r['constraint_valid_frac']:.4f}")

    rows = [done[str(s)] for s in cfg["seeds"]]
    summary = {}
    for cost in COSTS:
        stamped = [r["costs"][str(cost)]["a_clean_stamped"] for r in rows]
        pl = [plain[r["seed"]] for r in rows]
        po = [pois[(r["seed"], cost)] for r in rows]
        lo, hi = bootstrap_ci(stamped)
        summary[str(cost)] = {
            "a_clean_plain_mean": float(np.mean(pl)),
            "a_clean_stamped_mean": float(np.mean(stamped)),
            "a_clean_stamped_ci95": [lo, hi],
            "a_poisoned_stamped_mean": float(np.mean(po)),
            "evasion_component_E": float(np.mean(stamped) - np.mean(pl)),
            "backdoor_component_B": float(np.mean(po) - np.mean(stamped)),
            "per_seed_stamped": stamped,
        }

    v = summary[str(VERDICT_COST)]["a_clean_stamped_mean"]
    verdict = ("CLEAN" if v <= CLEAN_THRESH else
               "CONTAMINATED" if v >= CONTAMINATED_THRESH else "INCONCLUSIVE-MIXED")

    print("\n=== Exp C (decision rule fixed before the run, verdict at cost 16)")
    for cost in COSTS:
        s = summary[str(cost)]
        print(f"  cost {cost}: clean plain {s['a_clean_plain_mean']:.4f} | "
              f"clean stamped {s['a_clean_stamped_mean']:.4f} "
              f"CI [{s['a_clean_stamped_ci95'][0]:.4f}, {s['a_clean_stamped_ci95'][1]:.4f}] | "
              f"poisoned {s['a_poisoned_stamped_mean']:.4f}")
        print(f"           evasion E = {s['evasion_component_E']:.4f}   "
              f"backdoor B = {s['backdoor_component_B']:.4f}")
    print(f"  VERDICT: {verdict}")

    blob = {"criterion": {"clean_max": CLEAN_THRESH, "contaminated_min": CONTAMINATED_THRESH,
                          "verdict_cost": VERDICT_COST},
            "cell_rate": RATE, "fingerprint": fp, "verdict": verdict,
            "summary": summary, "per_seed": rows}
    if args.write_manifest:
        path = config.RESULTS / "clean_model_stamped_asr.json"
        path.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"\nwrote {path.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
