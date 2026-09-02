"""Task 7B follow-up: WHY does the centroid cutoff fail, and WHY did canonical AC flip on one seed?

Two diagnostics on one victim, because they share the expensive setup (clean MLP + SHAP + poisoned
victim). Neither selects anything: this script measures, it does not choose a cutoff. Any replacement
rule it motivates must be pre-registered before it is run on the evasion grid.

DIAGNOSTIC 1 -- the cutoff, not the score, is what fails.
`results/ac_centroid_repair_control.json` shows the centroid score ranking the loud poison at
AUC 0.981-0.989 on every seed, while the 99th-percentile clean cutoff returns recall 0.026-0.604 and
the MAD cutoff returns recall 1.0 at 7-12% FPR. A score that good with a recall that unstable means the
two distributions overlap in a specific, diagnosable way, so this dumps the actual shapes:

  - clean-calibration and poison score quantiles side by side
  - where each cutoff physically lands in those distributions
  - the ROC, and specifically the BEST ACHIEVABLE recall at FPR <= 0.01

That last number is the one that matters. If max achievable recall at the 1% budget is low despite a
high AUC, then no cutoff can satisfy the pre-registered gate and the honest finding is that the gate is
unreachable for this score -- not that a better estimator was waiting to be found. If instead the
achievable recall is high and only the q99 ESTIMATOR is missing it, that is a fixable instrument
problem, like the two decision-layer bugs already found in this round.

Hypothesis under test: the clean distance distribution is heavy-tailed, so the 99th percentile is set
by a handful of clean outliers that sit above most of the poison. That would explain a high AUC (most
poison outranks most clean) coexisting with a low TPR at the extreme-left of the ROC.

DIAGNOSTIC 2 -- canonical AC's seed-1337 collapse.
In the control run canonical AC scored 1.0 on four seeds and 0.00035 on seed 1337, at a loud
constraint-violating trigger where the committed `AcControlRecallMean` is 1.0 across all five seeds
(`results/macros.json`). The setups are not identical -- this arm holds 20k rows out of both the poison
pool and the suspect set -- so this is not a contradiction of the committed number. It IS evidence that
the 2-means split can flip to total failure under a modest perturbation of the clustered set. This
records what the split actually contained on each seed (minority size, purity, silhouette), which is
the same instrumentation `scripts/11_ac_mechanism.py` used to establish the original mechanism.

Run:  python scripts/18d_ac_cutoff_diagnostic.py --seeds 42,1337     # the two contrasting seeds
      python scripts/18d_ac_cutoff_diagnostic.py --seeds 42,123,456,789,1337
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup, apply_overrides
from src import config
from src.data import apply_standardiser
from src.detectors import poison_recall
from src.detectors.activation_clustering import (
    detect as ac_detect,
    fit_clean_reference_centroids,
    nearest_clean_centroid_scores,
)
from src.detectors.spectral import flag_by_mad_threshold, spectral_scores
from src.models import mlp_penultimate_features, train_mlp
from src.poison import carve_clean_reference, poison_trainset_cleanlabel
from src.trigger import build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET
CONTROL_RATE, CONTROL_COST = 0.05, 8       # the same loud cell the control stage used
SILHOUETTE_SAMPLE = 10000
QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]
FPR_BUDGETS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def _q(a, qs=QUANTILES):
    a = np.asarray(a, dtype=float)
    return {str(q): float(np.quantile(a, q)) for q in qs} if a.size else {}


def _tpr_at_fpr(is_poison, scores, budgets=FPR_BUDGETS) -> dict:
    """Best achievable recall at each FPR budget, read off the empirical ROC.

    This is the ceiling ANY cutoff on this score could reach. If it is low at 0.01, the pre-registered
    gate is unreachable for this score and no estimator fix can rescue it.
    """
    fpr, tpr, thr = roc_curve(is_poison.astype(int), scores)
    out = {}
    for b in budgets:
        ok = fpr <= b
        i = int(np.argmax(tpr * ok)) if ok.any() else 0
        out[str(b)] = dict(max_recall=float(tpr[i]), at_fpr=float(fpr[i]),
                           threshold=float(thr[i]) if np.isfinite(thr[i]) else None)
    return out


def run_seed(seed, S, cfg, device, stats) -> dict:
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]

    x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
    clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
    ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET, kind="mlp",
                                 device=device)
    _, violating = build_trigger(ranking, CONTROL_COST, x_tr_raw, constraints, features, bounds,
                                stats=stats)

    ref_idx, calib_idx, suspect_idx = carve_clean_reference(y_tr, target=TARGET, seed=seed)
    held_back = np.concatenate([ref_idx, calib_idx])
    x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
        x_tr_raw, y_tr, violating, CONTROL_RATE, S["scaler"], target=TARGET, seed=seed,
        constraints=constraints, feature_names=features, bounds=bounds, exclude_idx=held_back)
    mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

    is_poison = np.isin(suspect_idx, poison_idx)
    feats_ref = mlp_penultimate_features(mlp, x_p_std[ref_idx], device)
    feats_calib = mlp_penultimate_features(mlp, x_p_std[calib_idx], device)
    feats_suspect = mlp_penultimate_features(mlp, x_p_std[suspect_idx], device)

    reference = fit_clean_reference_centroids(
        feats_ref, feats_calib, n_components=config.AC_PCA_COMPONENTS, seed=seed,
        clean_quantile=config.CLEAN_CALIBRATION_QUANTILE, n_clusters=config.AC_N_CLUSTERS)

    cent = nearest_clean_centroid_scores(feats_suspect, reference)
    calib_scores = nearest_clean_centroid_scores(feats_calib, reference)
    spec = spectral_scores(feats_suspect, n_components=config.SPECTRAL_COMPONENTS)

    clean_scores = cent[~is_poison]
    poison_scores = cent[is_poison]

    # --- DIAGNOSTIC 1: where do the cutoffs land, and what is achievable at all? ---
    tau_q99 = float(reference.distance_threshold)
    mad_mask = flag_by_mad_threshold(cent, z_thresh=config.MAD_Z_PRIMARY)
    tau_mad = float(cent[mad_mask].min()) if mad_mask.any() else float("inf")

    cutoff = dict(
        q99_threshold=tau_q99,
        mad_threshold=(None if not np.isfinite(tau_mad) else tau_mad),
        q99_recall=float(poison_recall(cent > tau_q99, is_poison)),
        mad_recall=float(poison_recall(mad_mask, is_poison)),
        # what fraction of POISON falls below the clean 99th percentile
        poison_below_q99=float((poison_scores <= tau_q99).mean()),
        # how far into the clean tail the cutoff sits, in MADs -- a scale-free read on tail weight
        q99_in_clean_mads=float(
            (tau_q99 - np.median(clean_scores)) / max(np.median(np.abs(clean_scores - np.median(clean_scores))), 1e-12)),
        clean_tail_ratio_p999_over_p50=float(
            np.quantile(clean_scores, 0.999) / max(np.quantile(clean_scores, 0.5), 1e-12)),
        poison_median_over_clean_median=float(
            np.median(poison_scores) / max(np.median(clean_scores), 1e-12)),
    )

    # --- DIAGNOSTIC 2: what did canonical AC's 2-means split actually contain? ---
    ac_mask, ac_sil = ac_detect(feats_suspect, n_components=config.AC_PCA_COMPONENTS, seed=seed,
                                silhouette_sample=SILHOUETTE_SAMPLE)
    n_min = int(ac_mask.sum())
    ac = dict(
        recall=float(poison_recall(ac_mask, is_poison)),
        minority_size=n_min,
        minority_purity=(float((ac_mask & is_poison).sum() / n_min) if n_min else None),
        silhouette=(None if ac_sil is None else float(ac_sil)),
        n_poison=int(is_poison.sum()),
        n_suspect=int(len(suspect_idx)),
        poison_fraction=float(is_poison.mean()),
    )

    return dict(
        seed=seed, rate=CONTROL_RATE, cost=CONTROL_COST,
        auc=float(roc_auc_score(is_poison, cent)),
        spectral_auc=float(roc_auc_score(is_poison, spec)),
        clean_score_quantiles=_q(clean_scores),
        poison_score_quantiles=_q(poison_scores),
        calibration_score_quantiles=_q(calib_scores),
        cutoff=cutoff,
        achievable=_tpr_at_fpr(is_poison, cent),
        canonical_ac=ac,
    )


def main():
    t0 = time.time()
    cfg = apply_overrides(get_config(smoke="--smoke" in sys.argv), sys.argv)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} seeds={cfg['seeds']} cell=({CONTROL_RATE}, {CONTROL_COST}) violating")

    S = load_setup(cfg)
    stats = raw_trigger_stats(S["x_tr_raw"], S["constraints"])

    rows = []
    for seed in cfg["seeds"]:
        print(f"--- seed {seed} ({time.time() - t0:.0f}s) ---")
        r = run_seed(seed, S, cfg, device, stats)
        rows.append(r)
        c, a = r["cutoff"], r["canonical_ac"]
        print(f"  AUC={r['auc']:.4f}  q99_recall={c['q99_recall']:.4f}  mad_recall={c['mad_recall']:.4f}")
        print(f"  poison BELOW clean-q99: {c['poison_below_q99']:.4f}   "
              f"clean tail p999/p50: {c['clean_tail_ratio_p999_over_p50']:.1f}x   "
              f"q99 sits {c['q99_in_clean_mads']:.1f} MADs into clean")
        print(f"  MAX achievable recall @FPR<=0.01: {r['achievable']['0.01']['max_recall']:.4f}  "
              f"(@0.05: {r['achievable']['0.05']['max_recall']:.4f})")
        print(f"  canonical AC recall={a['recall']:.5f} |minority|={a['minority_size']} "
              f"purity={a['minority_purity']} silhouette={a['silhouette']}")

    out = dict(
        seeds=cfg["seeds"], cell=[CONTROL_RATE, CONTROL_COST], trigger_variant="violating",
        recipe=dict(pca_components=config.AC_PCA_COMPONENTS, n_clusters=config.AC_N_CLUSTERS,
                    spectral_k=config.SPECTRAL_COMPONENTS,
                    reference_size=config.AC_REFERENCE_SIZE,
                    calibration_size=config.AC_CALIBRATION_SIZE,
                    clean_quantile=config.CLEAN_CALIBRATION_QUANTILE,
                    mad_z=config.MAD_Z_PRIMARY, mlp_epochs=cfg["mlp_epochs"], smoke=cfg["smoke"]),
        rows=rows,
        note="Diagnostic ONLY -- selects nothing. Measures (1) why the centroid cutoff fails while the "
             "score ranks well, and (2) what canonical AC's 2-means split contained on each seed after "
             "the 20k-row holdout. `achievable` is the empirical ROC ceiling: max recall at each FPR "
             "budget for ANY cutoff on this score. If achievable recall at 0.01 is low despite a high "
             "AUC, the pre-registered FPR gate is unreachable for this score and no estimator fix "
             "changes that -- report it as a property of the score, not as a failed repair attempt. Any "
             "replacement cutoff this motivates MUST be pre-registered before it touches the evasion "
             "grid.")
    (config.RESULTS / "ac_cutoff_diagnostic.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/ac_cutoff_diagnostic.json ({len(rows)} rows)")
    print(f"elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
