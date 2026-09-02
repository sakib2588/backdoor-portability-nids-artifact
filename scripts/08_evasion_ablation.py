"""M4 follow-up ablation: does an adaptive, budget-free threshold recover Spectral recall inside the
evasion window (results/analysis.json: evasion_window)?

Context (see notes/20260715-decision-evasion-window-reframe.md): at low poison rates (<1%) the clean-
label trigger succeeds (ASR>=0.8) but Spectral Signatures' fixed removal budget (`flag_by_scores`,
`ceil(1.5*expected_frac*n)` -- Tran et al. 2018's own convention, calibrated for vision-scale poison
rates) recovers <50% of the poison, even though the score distribution is near-perfectly separable
(AUC~0.98-0.99). This script retrains the SAME cells (same seeds/rate/cost/trigger construction as
scripts/05_run_detectors.py) and compares the ALREADY-REPORTED fixed-budget recall (as a determinism
self-check) against `flag_by_mad_threshold` -- a statistically-motivated, budget-free alternative --
at three PRE-REGISTERED z-thresholds {2.0, 2.5, 3.0} (no cherry-picking after seeing results).

This is NOT part of Tran et al.'s original algorithm; it is an explicit departure being tested, and
the outcome is reported honestly whichever way it goes (recovers recall, or doesn't). Scope note:
this rerun's determinism self-check compares fixed_budget_recall against detectors.json, the same
signal 05_run_detectors.py's own self-check already validates the retrained model against M2's
ASR/clean-accuracy; it does not re-check ASR/clean-accuracy itself, only the detector-relevant path.

Run:  python scripts/08_evasion_ablation.py --seeds 42       # single-seed check first
      python scripts/08_evasion_ablation.py                  # full 5-seed run
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup, apply_overrides
from src import config
from src.data import apply_standardiser
from src.detectors import poison_recall
from src.detectors.activation_clustering import detect as ac_detect
from src.detectors.spectral import flag_by_scores, flag_by_mad_threshold, spectral_scores
from src.models import mlp_penultimate_features, train_mlp
from src.trigger import shap_rank_features, build_trigger, raw_trigger_stats
from src.poison import poison_trainset_cleanlabel

TARGET = config.ATTACK_TARGET
SPECTRAL_K = 5   # must match scripts/04/05's SPECTRAL_K -- same detector, only the flag rule differs
AC_PCA_COMPONENTS = 10   # must match scripts/05_run_detectors.py's AC recipe -- same detector
# 05_run_detectors.py calls ac_detect(..., silhouette_sample=SILHOUETTE_SAMPLE) with SILHOUETTE_SAMPLE
# = 10000; matched here so the AC arm runs 05's exact recipe. The silhouette is a diagnostic only --
# it does not touch the cluster assignment, the poison mask, or the recall.
AC_SILHOUETTE_SAMPLE = 10000
Z_THRESHOLDS = (2.0, 2.5, 3.0)   # pre-registered before running; report all three, pick none after

# evasion-window cells (from results/analysis.json) + one matched-rate control where the fixed
# budget already recovers full recall (sanity check: the new rule must not break a working case)
TARGET_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16), (0.1, 16)]


def _load_original_index():
    """Index results/detectors.json's main_grid by (seed, rate, cost) ONCE, for the determinism
    self-check (this rerun must reproduce the already-reported fixed-budget recall before its
    adaptive-threshold number is trustworthy). Loaded once outside the cell loop, not per cell."""
    path = config.RESULTS / "detectors.json"
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    return {(r["seed"], r["rate"], r["cost"]): r for r in blob["main_grid"]}


def main():
    t0 = time.time()
    cfg = apply_overrides(get_config(smoke=False), sys.argv)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  seeds={cfg['seeds']}  target_cells={TARGET_CELLS}  "
          f"z_thresholds={Z_THRESHOLDS}")

    S = load_setup(cfg)
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    ref_index = _load_original_index()

    # seed-invariant across the whole run (x_tr_raw/constraints/scaler don't change per seed) --
    # raw_trigger_stats's own docstring (src/trigger.py) says callers sweeping many combinations
    # must compute this ONCE, not per seed/cell.
    x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
    stats = raw_trigger_stats(x_tr_raw, constraints)

    rows = []
    for seed in cfg["seeds"]:
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        clean_mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET,
                                     kind="mlp", device=device)

        for rate, cost in TARGET_CELLS:
            realiz, _ = build_trigger(ranking, cost, x_tr_raw, constraints, features, bounds, stats=stats)
            x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
                x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
                constraints=constraints, feature_names=features, bounds=bounds)
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            benign_pos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(benign_pos, poison_idx)
            has_poison = bool(is_poison.sum() > 0)
            n_poison, n_benign = int(is_poison.sum()), len(is_poison)
            feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
            scores = spectral_scores(feats, n_components=SPECTRAL_K)

            fixed_flag = flag_by_scores(scores, expected_frac=float(is_poison.mean()))
            fixed_recall = poison_recall(fixed_flag, is_poison) if has_poison else None
            auc = (float(roc_auc_score(is_poison, scores))
                  if has_poison and is_poison.sum() < len(is_poison) else None)

            adaptive = {}
            for z in Z_THRESHOLDS:
                flag = flag_by_mad_threshold(scores, z_thresh=z)
                n_true_pos = int(np.sum(flag & is_poison))   # exact count, not reconstructed later
                adaptive[str(z)] = dict(
                    recall=poison_recall(flag, is_poison) if has_poison else None,
                    n_flagged=int(flag.sum()), n_true_positive=n_true_pos,
                    n_false_positive=int(flag.sum()) - n_true_pos)

            # --- rule runtime: the adaptive rule's cost vs the fixed budget's, on identical scores.
            # Reported so a defender can price the substitution. Timed on the same `scores` array,
            # 10 repeats, median -- these are microsecond-scale and noisy at n=1.
            t_fixed, t_adaptive = [], []
            for _ in range(10):
                s = time.perf_counter()
                flag_by_scores(scores, expected_frac=float(is_poison.mean()))
                t_fixed.append(time.perf_counter() - s)
                s = time.perf_counter()
                flag_by_mad_threshold(scores, z_thresh=3.0)
                t_adaptive.append(time.perf_counter() - s)
            rule_runtime = dict(fixed_budget_s=float(np.median(t_fixed)),
                                adaptive_s=float(np.median(t_adaptive)),
                                n_scored=int(len(scores)), repeats=10)

            # --- Activation Clustering: Chen et al.'s hard 2-cluster rule, unchanged, rerun here
            # purely as a determinism self-check against results/detectors.json (ac_fixed_recall_ref).
            # No adaptive arm: AC has no removal budget to replace -- it flags the smaller cluster --
            # so the MAD substitution tested on Spectral has no premise here. An earlier version of
            # this script ran that substitution on a cluster-distance proxy score; the proxy proved to
            # be a near-monotone transform of Spectral's own score (Spearman rho=0.9967), so the arm
            # restated Spectral rather than measuring AC, and both the arm and the proxy were removed
            # (notes/20260716-decision-ac-proxy-not-independent.md). AC's actual failure mechanism
            # inside the evasion window is measured by scripts/11_ac_mechanism.py instead.
            ac_mask, _ = ac_detect(feats, n_components=AC_PCA_COMPONENTS, seed=seed,
                                   silhouette_sample=AC_SILHOUETTE_SAMPLE)
            ac_fixed_recall = poison_recall(ac_mask, is_poison) if has_poison else None

            ref = ref_index.get((seed, rate, cost))
            determinism_ok = None
            if ref is not None and ref["spectral"]["recall"] is not None and fixed_recall is not None:
                determinism_ok = bool(np.isclose(fixed_recall, ref["spectral"]["recall"], atol=1e-6))
                if not determinism_ok:
                    print(f"WARNING: rerun fixed-budget recall {fixed_recall:.6f} != "
                          f"detectors.json's {ref['spectral']['recall']:.6f} at seed={seed} "
                          f"rate={rate} cost={cost}")

            row = dict(seed=seed, rate=rate, cost=cost,
                       fixed_budget_recall=fixed_recall, fixed_budget_recall_ref=(
                           ref["spectral"]["recall"] if ref else None),
                       determinism_ok=determinism_ok,
                       spectral_auc=auc, n_poison=n_poison, n_benign=n_benign,
                       adaptive=adaptive,
                       ac_fixed_recall=ac_fixed_recall, ac_fixed_recall_ref=(
                           ref["ac"]["recall"] if ref else None),
                       rule_runtime=rule_runtime)
            rows.append(row)
            print(f"  rate={rate} cost={cost}: fixed={fixed_recall} "
                  f"(ref={ref['spectral']['recall'] if ref else None}) auc={auc} "
                  f"adaptive={ {z: v['recall'] for z, v in adaptive.items()} }")
            print(f"      AC: fixed={ac_fixed_recall} "
                  f"(ref={ref['ac']['recall'] if ref else None})")

    out = dict(z_thresholds=list(Z_THRESHOLDS), target_cells=TARGET_CELLS, seeds=cfg["seeds"], rows=rows,
               note="adaptive = flag_by_mad_threshold (budget-free); fixed_budget = flag_by_scores "
                    "(Tran et al. 2018's original convention, same as results/detectors.json). "
                    "fixed_budget_recall_ref/ac_fixed_recall_ref are the values already committed in "
                    "detectors.json for this exact cell -- determinism_ok checks this rerun "
                    "reproduces the Spectral one. n_true_positive/n_false_positive are exact counts "
                    "(flag & is_poison), not reconstructed from recall downstream. "
                    "ac_fixed_recall is Chen et al.'s hard 2-cluster rule, unchanged and rerun only "
                    "as a determinism self-check against ac_fixed_recall_ref; there is no AC "
                    "adaptive arm, because AC has no removal budget for a budget-free rule to "
                    "replace (see notes/20260716-decision-ac-proxy-not-independent.md, and "
                    "scripts/11_ac_mechanism.py for AC's actual failure mechanism). rule_runtime "
                    "prices the adaptive rule against the fixed budget on identical score arrays.")
    (config.RESULTS / "evasion_ablation.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/evasion_ablation.json ({len(rows)} rows)")
    elapsed = time.time() - t0
    print(f"elapsed={elapsed:.0f}s for {len(cfg['seeds'])} seed(s)")


if __name__ == "__main__":
    main()
