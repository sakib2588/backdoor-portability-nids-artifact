"""Task NC-11: does Wang et al.'s large-trigger ceiling transfer to tabular NIDS?

WHY. Wang et al. 2019, Section VII-B ("Larger Triggers"): as the reverse-engineered mask's L1 norm
grows, it blends into the range of uninfected labels and the anomaly index falls below threshold --
a SIZE ceiling, independent of the two-class degeneracy this project has already diagnosed.

This project's grid tops out at cost=16 (config.TRIGGER_COSTS), which is simultaneously "the edge of
what we swept" and "the presumed frontier" -- two different claims that have never been separated,
because nothing above 16 was ever run. build_trigger has no hard cap below ~755 equality-free
features, so probing past 16 needs no change to the attack machinery.

PUBLISHED BASELINE RECIPE ONLY. This tests the EXISTING rule's ceiling, not a repair, so it uses
reverse_engineer_tabular + nc_binary_flag at tau=2.0 against a per-seed clean reference -- not the
multistart/sparser/constraint_aware candidates, all of which NC-6 already rejected.

PRE-REGISTERED INTERPRETATION, locked before running. Three readings, and a monotonic story will NOT
be forced if the data does not show one:
  (a) detection RISES with cost   -> NC benefits from bigger, easier-to-localise triggers even under
                                     the binary degeneracy. Would somewhat undercut this project's
                                     NC null, and must be reported as such.
  (b) detection FALLS with cost   -> Wang et al.'s ceiling transfers to tabular, independently of the
                                     class-count issue.
  (c) FLAT / no signal            -> NC is already saturated by the class-count degeneracy and
                                     trigger size is a second-order effect. This is what the AUC
                                     0.389-0.444 class-asymmetry finding in nc_repair_calibration.json
                                     predicts.

SECOND, FREE FINDING. Cost=128 means satisfying 128 simultaneous feature constraints, a far harder
joint projection than cost=16. build_trigger returns both the realizable and the constraint-violating
trigger, so the ASR gap between them (this project's "H2 crossover") is recorded at every cost. An
ASR collapse for the realizable trigger at high cost is a reportable defence-by-constraints result in
its own right, not a failed run.

Run:  python scripts/24_nc_large_trigger_ceiling.py --smoke --seeds 42
      python scripts/24_nc_large_trigger_ceiling.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup, apply_overrides
from src import config
from src.data import apply_standardiser
from src.detectors.neural_cleanse import (
    balanced_inversion_sample,
    nc_binary_flag,
    reverse_engineer_tabular,
)
from src.models import attack_success_rate, clean_accuracy, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET
RATE = 0.005            # the strongest evasion-window rate, held fixed: this task sweeps COST
COSTS = (16, 32, 64, 128)   # 16 is the reference -- the current grid's edge
OUT_PATH = config.RESULTS / "nc_large_trigger_ceiling.json"


def _finite_or_none(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def load_checkpoint(path, key):
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} (seed, cost) rows already done")
    return blob.get("rows", {})


def save_checkpoint(path, key, rows):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(path)   # atomic: a crash mid-write never corrupts the checkpoint


def _config_key(cfg, seeds) -> dict:
    return dict(seeds=list(seeds), costs=list(COSTS), rate=RATE,
                nc_steps=cfg["nc_steps"], nc_sample=cfg["nc_sample"],
                mlp_epochs=cfg["mlp_epochs"], smoke=cfg["smoke"])


def main():
    t0 = time.time()
    argv = sys.argv
    cfg = apply_overrides(get_config(smoke="--smoke" in argv), argv)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} seeds={cfg['seeds']} costs={COSTS} rate={RATE}")

    ckpt_path = OUT_PATH.with_name(OUT_PATH.stem + ".checkpoint.json")
    S = load_setup(cfg)
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    stats = raw_trigger_stats(S["x_tr_raw"], S["constraints"])
    std_lo, std_hi = S["std_bounds"]
    bnd = (torch.as_tensor(std_lo, dtype=torch.float32),
           torch.as_tensor(std_hi, dtype=torch.float32))
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])
    x_bot_raw = S["x_te_raw"][S["y_te"] != TARGET]

    key = _config_key(cfg, cfg["seeds"])
    store = load_checkpoint(ckpt_path, key)
    rows = []

    for seed in cfg["seeds"]:
        if all(f"{seed}|{c}" in store for c in COSTS):
            rows.extend(store[f"{seed}|{c}"] for c in COSTS)
            continue

        print(f"--- seed {seed} ({time.time() - t0:.0f}s) ---")
        clean_mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean_mlp, S["x_tr_raw"], S["scaler"], target=TARGET,
                                     kind="mlp", device=device)
        # One clean reference per seed -- cost-invariant, so computed once, matching run_nc_boundary.
        sample_clean = balanced_inversion_sample(x_tr_std, S["y_tr"], cfg["nc_sample"], seed)
        cl_norms = reverse_engineer_tabular(clean_mlp, 2, sample_clean, device=device,
                                            steps=cfg["nc_steps"], bounds=bnd)
        _, cl_ratio = nc_binary_flag(cl_norms, tau=2.0)

        for cost in COSTS:
            ck = f"{seed}|{cost}"
            if ck in store:
                rows.append(store[ck])
                continue

            realiz, violating = build_trigger(ranking, cost, S["x_tr_raw"], S["constraints"],
                                              S["features"], S["bounds"], stats=stats)
            x_p_std, y_p, _ = poison_trainset_cleanlabel(
                S["x_tr_raw"], S["y_tr"], realiz, RATE, S["scaler"], target=TARGET, seed=seed,
                constraints=S["constraints"], feature_names=S["features"], bounds=S["bounds"])
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            # The H2 crossover at high cost: 128 simultaneous constraints is a much harder joint
            # projection than 16, so the realizable/violating ASR gap is the second finding here.
            # apply_trigger takes (constraints, feature_names, bounds) positionally and returns RAW
            # rows -- the realizable variant needs them to project onto the feasible set, and the
            # result must be standardised before the victim sees it. Same call shape as
            # scripts/04_poison_sweep.py, which is the reference for this pairing.
            x_trig_real = apply_trigger(x_bot_raw, realiz, S["constraints"], S["features"],
                                        S["bounds"])
            x_trig_viol = apply_trigger(x_bot_raw, violating)   # violating: stamped, unprojected
            asr_real = attack_success_rate(
                mlp, apply_standardiser(S["scaler"], x_trig_real), TARGET, device=device)
            asr_viol = attack_success_rate(
                mlp, apply_standardiser(S["scaler"], x_trig_viol), TARGET, device=device)
            acc = clean_accuracy(mlp, x_te_std, S["y_te"], device=device)

            sample = balanced_inversion_sample(x_p_std, y_p, cfg["nc_sample"], seed)
            norms = reverse_engineer_tabular(mlp, 2, sample, device=device,
                                             steps=cfg["nc_steps"], bounds=bnd)
            flagged_class, ratio = nc_binary_flag(norms, tau=2.0)
            flagged_target = bool(flagged_class == TARGET and ratio > cl_ratio)

            row = dict(
                seed=seed, rate=RATE, cost=cost,
                nc_flagged_target=flagged_target,
                nc_ratio=_finite_or_none(ratio),
                clean_reference_ratio=_finite_or_none(cl_ratio),
                target_mask_norm=_finite_or_none(norms[TARGET]),
                other_mask_norm=_finite_or_none(norms[1 - TARGET]),
                flagged_class=int(flagged_class),
                asr_realizable=_finite_or_none(asr_real),
                asr_violating=_finite_or_none(asr_viol),
                asr_realizability_gap=_finite_or_none(asr_viol - asr_real),
                clean_accuracy=_finite_or_none(acc),
                n_trigger_features=int(len(realiz["indices"])),
                inversion_converged=bool(np.all(np.isfinite(norms))),
            )
            store[ck] = row
            rows.append(row)
            save_checkpoint(ckpt_path, key, store)
            print(f"  cost={cost}: flagged={flagged_target} ratio={row['nc_ratio']} "
                  f"asr_real={row['asr_realizable']} asr_viol={row['asr_violating']} "
                  f"acc={row['clean_accuracy']}")

    by_cost = {}
    for cost in COSTS:
        sel = [r for r in rows if r["cost"] == cost]
        det = [r["nc_flagged_target"] for r in sel]
        ar = [r["asr_realizable"] for r in sel if r["asr_realizable"] is not None]
        av = [r["asr_violating"] for r in sel if r["asr_violating"] is not None]
        by_cost[str(cost)] = dict(
            n=len(sel),
            detection_rate=(float(np.mean(det)) if det else None),
            mean_asr_realizable=(float(np.mean(ar)) if ar else None),
            mean_asr_violating=(float(np.mean(av)) if av else None))

    out = dict(
        seeds=cfg["seeds"], costs=list(COSTS), rate=RATE,
        recipe=dict(nc_steps=cfg["nc_steps"], nc_sample=cfg["nc_sample"],
                    mlp_epochs=cfg["mlp_epochs"], smoke=cfg["smoke"], tau=2.0,
                    inversion="reverse_engineer_tabular (published baseline, NOT the NC-6 candidates)"),
        rows=rows, summary_by_cost=by_cost,
        note="NC-11 large-trigger ceiling. Tests whether Wang et al. Section VII-B's size ceiling "
             "transfers to tabular NIDS, using the PUBLISHED baseline recipe only -- this probes the "
             "existing rule's ceiling, not a repair. Rate is held at 0.005 so cost is the only axis. "
             "Three interpretations were locked before running (rises / falls / flat) and a monotonic "
             "story is not to be forced if absent. asr_realizability_gap is the second, independent "
             "finding: at cost=128 the realizable trigger must satisfy 128 simultaneous feature "
             "constraints, so an ASR collapse there is a defence-by-constraints result, not a bug.",
    )
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(config.ROOT)} ({len(rows)} rows)")
    for cost, v in by_cost.items():
        print(f"  cost={cost}: detection={v['detection_rate']} "
              f"asr_real={v['mean_asr_realizable']} asr_viol={v['mean_asr_violating']}")
    print(f"elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
