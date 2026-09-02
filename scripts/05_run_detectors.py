"""M3 detector sweep: Spectral Signatures + Activation Clustering across the full realizable-trigger
MLP grid, Neural Cleanse on an explicit boundary subset, H3 (imbalance confound) at one operating
point, and H4 (LightGBM has no activations) recorded as a documented non-applicability.

Every MLP cell is RETRAINED here (M2 checkpointed metrics only, not model weights) using the exact
seed/trigger/poison construction 04_poison_sweep.py used, then self-checked against M2's stored ASR/
clean-acc for that cell (see D1 in docs/superpowers/plans/2026-07-15-m3-detector-sweep.md) --
disagreement beyond atol=1e-6 is a live determinism bug, not a modeling choice, and is surfaced as
`determinism_ok=False` on that row rather than silently absorbed.

Neural Cleanse runs on an EXPLICIT boundary subset (D2 in the plan), not the full grid -- gradient
inversion is the true compute bottleneck. H3 runs at one operating point (D3), matching the existing
tabular positive control's rate/cost, not a second full grid. See the plan doc for full rationale.

Run:  python scripts/05_run_detectors.py           # full 5-seed sweep
      python scripts/05_run_detectors.py --smoke    # fast validation slice
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup, apply_overrides
from src import config
from src.data import apply_standardiser, downsample_benign, fit_standardiser
from src.detectors import poison_recall
from src.detectors.activation_clustering import detect as ac_detect
from src.detectors.neural_cleanse import (
    balanced_inversion_sample, nc_binary_flag, reverse_engineer_tabular,
)
from src.detectors.spectral import flag_by_scores, spectral_scores
from src.models import (
    attack_success_rate, clean_accuracy, mlp_penultimate_features, train_mlp,
)
from src.trigger import shap_rank_features, build_trigger, raw_trigger_stats, apply_trigger

TARGET = config.ATTACK_TARGET
SPECTRAL_K = 5   # must match scripts/04_poison_sweep.py's SPECTRAL_K -- carried-forward M1 scope item
# Activation Clustering's silhouette is an O(n^2) DIAGNOSTIC; on the full ~140k-row benign NIDS
# subset it dominates wall-clock when run on every one of the 125 grid cells (it made M2's positive
# control tolerable only because that ran the detector 5 times, not 125). Cap the silhouette estimate
# to this many seeded-subsample points -- the clustering + poison_mask (the recall metric) stay on the
# full set, so only the reported silhouette becomes a subsample estimate. See ac_detect's docstring.
SILHOUETTE_SAMPLE = 10000

CKPT = "detector_sweep.checkpoint.json"


def _finite_or_none(x):
    """Map a non-finite float (inf from a fully-collapsed NC mask) to None + a saturation flag, so
    results/detectors.json stays strict RFC-8259 JSON (no `Infinity` token) for the M4 macros SSOT."""
    return x if (x is not None and math.isfinite(x)) else None


def cell_key(seed, rate, cost) -> str:
    return f"{seed}|{rate}|{cost}"


def _config_key(cfg: dict) -> dict:
    return dict(smoke=cfg["smoke"], rates=list(cfg["rates"]), costs=list(cfg["costs"]),
                mlp_epochs=cfg["mlp_epochs"], nc_steps=cfg["nc_steps"], subsample=cfg["subsample"],
                nc_boundary_rates=list(cfg["nc_boundary_rates"]),
                nc_boundary_costs=list(cfg["nc_boundary_costs"]),
                h3_rate=cfg["h3_rate"], h3_cost=cfg["h3_cost"], h3_target_ratio=cfg["h3_target_ratio"],
                nc_sample=cfg["nc_sample"])


def load_checkpoint(path, key):
    if not path.exists():
        return {}, {}, {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, {}, {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}, {}, {}
    print(f"resuming from checkpoint: {len(blob.get('main', {}))} main-grid cells, "
          f"{len(blob.get('nc', {}))} NC cells, {len(blob.get('h3', {}))} H3 rows already done")
    return blob.get("main", {}), blob.get("nc", {}), blob.get("h3", {})


def save_checkpoint(path, key, main, nc, h3):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, main=main, nc=nc, h3=h3)))
    tmp.replace(path)   # atomic: a crash mid-write never corrupts the checkpoint


def seed_setup_mlp(seed, S, cfg, device):
    """Per-seed clean MLP + SHAP ranking (MLP only -- H4 excludes LightGBM from this script)."""
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    clean_mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
    ranking = shap_rank_features(clean_mlp, S["x_tr_raw"], S["scaler"], target=TARGET,
                                 kind="mlp", device=device)
    return clean_mlp, ranking


def load_m2_reference() -> dict:
    """Index results/poison_sweep.json's per-cell realizable-MLP rows by (seed, rate, cost) for the
    determinism self-check (D1): a retrained cell's ASR/clean_acc MUST match what M2 already measured
    for that exact seed/rate/cost, or reproducibility (set_seed + cudnn.deterministic) is broken."""
    path = config.RESULTS / "poison_sweep.json"
    if not path.exists():
        print("WARNING: results/poison_sweep.json not found -- determinism self-check DISABLED "
              "(every row will have determinism_ok=None, not verified)")
        return {}
    blob = json.loads(path.read_text())
    return {
        (r["seed"], r["rate"], r["cost"]): (r["asr"], r["clean_acc"])
        for r in blob["per_cell"] if r["victim"] == "mlp" and r["variant"] == "realizable"
    }


def run_main_grid(seed, S, cfg, device, clean_mlp, ranking, m2_ref, main_cells, save_cb):
    """Full realizable-trigger MLP grid: retrain, self-check vs M2, run Spectral + AC."""
    # clean_mlp unused here; threaded through for run_nc_boundary's shared call signature
    from src.poison import poison_trainset_cleanlabel

    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])
    x_bot_raw = S["x_te_raw"][S["y_te"] == 1]
    stats = raw_trigger_stats(x_tr_raw, constraints)

    for rate in cfg["rates"]:
        for cost in cfg["costs"]:
            key = cell_key(seed, rate, cost)
            if key in main_cells:
                continue
            realiz, _ = build_trigger(ranking, cost, x_tr_raw, constraints, features, bounds, stats=stats)
            x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
                x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
                constraints=constraints, feature_names=features, bounds=bounds)
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            x_bot_trig = apply_trigger(x_bot_raw, realiz, constraints, features, bounds)
            asr = attack_success_rate(mlp, apply_standardiser(S["scaler"], x_bot_trig), TARGET, device)
            cacc = clean_accuracy(mlp, x_te_std, S["y_te"], device)

            ref = m2_ref.get((seed, rate, cost))
            determinism_ok = None
            if ref is not None:
                determinism_ok = bool(np.isclose(asr, ref[0], atol=1e-6)
                                      and np.isclose(cacc, ref[1], atol=1e-6))
                if not determinism_ok:
                    print(f"WARNING: determinism check FAILED at seed={seed} rate={rate} cost={cost} "
                          f"-- retrained asr={asr:.6f} vs M2 asr={ref[0]:.6f} "
                          f"(cacc={cacc:.6f} vs {ref[1]:.6f})")

            # detectors operate on the target-class-conditioned (benign) subset of the poisoned
            # TRAINING set -- the defender inspects training data for planted poison, same convention
            # as scripts/04_poison_sweep.py's positive_control.
            benign_pos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(benign_pos, poison_idx)
            has_poison = bool(is_poison.sum() > 0)
            feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)

            scores = spectral_scores(feats, n_components=SPECTRAL_K)
            spec_flag = flag_by_scores(scores, expected_frac=float(is_poison.mean()))
            spec_recall = poison_recall(spec_flag, is_poison) if has_poison else None
            spec_auc = (float(roc_auc_score(is_poison, scores))
                       if has_poison and is_poison.sum() < len(is_poison) else None)

            ac_mask, ac_sil = ac_detect(feats, seed=seed, silhouette_sample=SILHOUETTE_SAMPLE)
            ac_recall = poison_recall(ac_mask, is_poison) if has_poison else None

            main_cells[key] = dict(
                seed=seed, rate=rate, cost=cost, asr=asr, clean_acc=cacc,
                determinism_ok=determinism_ok,
                spectral=dict(recall=spec_recall, auc=spec_auc),
                ac=dict(recall=ac_recall, silhouette=ac_sil, auc=None,
                       auc_reason="Activation Clustering is a hard 2-cluster assignment, not a "
                                  "ranked score -- no principled AUC without an undisclosed proxy score"),
            )
            save_cb()


def run_nc_boundary(seed, S, cfg, device, clean_mlp, ranking, main_cells, nc_cells, save_cb):
    """Neural Cleanse on the explicit boundary subset only (D2 in the M3 plan) -- NC's gradient
    inversion is the compute bottleneck, so it runs on a small subset of the grid, not all of it.

    Retrains a fresh MLP per boundary cell (run_main_grid does not persist model weights, so there
    is nothing to reuse; the retrain is deterministically identical to that cell's main-grid model
    given the same seed/trigger/poison construction). Computes ONE clean-model reference mask-ratio
    per seed (rate/cost-invariant) and flags a cell iff the backdoored model's target-class mask is
    anomalously smaller than that clean reference.
    """
    # main_cells unused here: retained for signature symmetry with run_main_grid and so main()'s call
    # pattern is uniform across both sweep stages -- kept as a hook for a future consistency check
    # (e.g. asserting a boundary cell's (seed, rate, cost) was also swept in the main grid), not dead
    # weight from an accidental copy-paste.
    from src.poison import poison_trainset_cleanlabel

    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
    stats = raw_trigger_stats(x_tr_raw, constraints)

    # one clean reference per seed -- rate/cost-invariant, so computed once, not per boundary cell
    std_lo, std_hi = S["std_bounds"]
    bnd = (torch.tensor(std_lo, dtype=torch.float32), torch.tensor(std_hi, dtype=torch.float32))
    # CLASS-BALANCED inversion sample (~50/50), the binary analogue of the vision arm's all-class
    # sample. NC computes class yt's mask from rows NOT already in yt; the earlier target-class
    # conditioning left class TARGET's inversion with nothing to push, collapsing its mask to a
    # model-independent constant and reducing the ratio test to a class-1 measurement. See
    # balanced_inversion_sample and notes/20260717-decision-nc-inversion-sample-fix.md.
    # Both arms draw the same way, so the clean reference stays a like-for-like baseline.
    sample_clean = balanced_inversion_sample(x_tr_std, y_tr, cfg["nc_sample"], seed)
    cl_norms = reverse_engineer_tabular(clean_mlp, 2, sample_clean, device=device,
                                        steps=cfg["nc_steps"], bounds=bnd)
    _, cl_ratio = nc_binary_flag(cl_norms, tau=2.0)

    for rate in cfg["nc_boundary_rates"]:
        for cost in cfg["nc_boundary_costs"]:
            key = cell_key(seed, rate, cost)
            if key in nc_cells:
                continue
            realiz, _ = build_trigger(ranking, cost, x_tr_raw, constraints, features, bounds, stats=stats)
            x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
                x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
                constraints=constraints, feature_names=features, bounds=bounds)
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            sample = balanced_inversion_sample(x_p_std, y_p, cfg["nc_sample"], seed)
            bd_norms = reverse_engineer_tabular(mlp, 2, sample, device=device,
                                                steps=cfg["nc_steps"], bounds=bnd)
            nc_flag, nc_ratio = nc_binary_flag(bd_norms, tau=2.0)
            nc_flagged_target = bool(nc_flag == TARGET and nc_ratio > cl_ratio)

            nc_cells[key] = dict(seed=seed, rate=rate, cost=cost,
                                 nc_flagged_target=nc_flagged_target,
                                 nc_ratio=_finite_or_none(nc_ratio),
                                 nc_clean_ratio=_finite_or_none(cl_ratio),
                                 nc_ratio_saturated=bool(not math.isfinite(nc_ratio)),
                                 nc_clean_ratio_saturated=bool(not math.isfinite(cl_ratio)))
            save_cb()


def run_h3(seed, S, cfg, device, ranking, main_cells, save_cb):
    """H3: does down-sampling training-set benign toward 50/50 recover detector recall at the SAME
    (rate, cost) operating point already measured (imbalanced) in run_main_grid? Compares against
    that cell's already-computed recall -- no need to duplicate the imbalanced-training measurement.

    `ranking`/`save_cb` are accepted for a uniform call signature with the other runners; this
    function builds its OWN ranking on the rebalanced data (a clean MLP trained on the balanced set
    has different SHAP importances than the imbalanced one) and returns its row rather than mutating
    a shared dict, so `save_cb` is invoked by the caller after this returns."""
    from src.poison import poison_trainset_cleanlabel

    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    rate, cost = cfg["h3_rate"], cfg["h3_cost"]
    imbalanced_key = cell_key(seed, rate, cost)
    if imbalanced_key not in main_cells:
        raise RuntimeError(
            f"H3 needs the imbalanced result for seed={seed} rate={rate} cost={cost} first -- "
            "run_main_grid must cover the H3 operating point (it does: h3_rate/h3_cost default to "
            "the same values as control_rate/control_cost, which ARE in cfg['rates']/cfg['costs']).")

    x_bal_df, y_bal_ser = downsample_benign(
        pd.DataFrame(x_tr_raw, columns=features), pd.Series(y_tr),
        target_ratio=cfg["h3_target_ratio"], seed=seed)
    x_tr_raw_bal = x_bal_df.to_numpy()
    y_tr_bal = y_bal_ser.to_numpy()

    scaler_bal = fit_standardiser(pd.DataFrame(x_tr_raw_bal, columns=features))
    x_tr_std_bal = apply_standardiser(scaler_bal, x_tr_raw_bal)
    clean_mlp_bal = train_mlp(x_tr_std_bal, y_tr_bal, seed, epochs=cfg["mlp_epochs"], device=device)
    ranking_bal = shap_rank_features(clean_mlp_bal, x_tr_raw_bal, scaler_bal, target=TARGET,
                                     kind="mlp", device=device)

    realiz, _ = build_trigger(ranking_bal, cost, x_tr_raw_bal, constraints, features, bounds)
    x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
        x_tr_raw_bal, y_tr_bal, realiz, rate, scaler_bal, target=TARGET, seed=seed,
        constraints=constraints, feature_names=features, bounds=bounds)
    mlp_bal = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

    benign_pos = np.where(y_p == TARGET)[0]
    is_poison = np.isin(benign_pos, poison_idx)
    has_poison = bool(is_poison.sum() > 0)
    feats = mlp_penultimate_features(mlp_bal, x_p_std[benign_pos], device)

    scores = spectral_scores(feats, n_components=SPECTRAL_K)
    spec_recall_bal = (poison_recall(flag_by_scores(scores, expected_frac=float(is_poison.mean())), is_poison)
                       if has_poison else None)
    ac_mask, ac_sil = ac_detect(feats, seed=seed, silhouette_sample=SILHOUETTE_SAMPLE)
    ac_recall_bal = poison_recall(ac_mask, is_poison) if has_poison else None

    imbalanced = main_cells[imbalanced_key]
    return dict(seed=seed, rate=rate, cost=cost,
                balanced=dict(spectral_recall=spec_recall_bal, ac_recall=ac_recall_bal,
                              ac_silhouette=ac_sil,
                              train_balance=dict(n=len(y_tr_bal), botnet=int(y_tr_bal.sum()))),
                imbalanced=dict(spectral_recall=imbalanced["spectral"]["recall"],
                                ac_recall=imbalanced["ac"]["recall"]))


def main():
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    cfg = apply_overrides(get_config(smoke), sys.argv)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  smoke={smoke}  seeds={cfg['seeds']}  rates={cfg['rates']}  "
          f"costs={cfg['costs']}  nc_boundary_rates={cfg['nc_boundary_rates']}  "
          f"nc_boundary_costs={cfg['nc_boundary_costs']}  h3=({cfg['h3_rate']},{cfg['h3_cost']})")

    S = load_setup(cfg)
    m2_ref = load_m2_reference()
    key = _config_key(cfg)
    ckpt_path = config.RESULTS / CKPT
    main_cells, nc_cells, h3_rows = load_checkpoint(ckpt_path, key)
    save_cb = lambda: save_checkpoint(ckpt_path, key, main_cells, nc_cells, h3_rows)

    for seed in cfg["seeds"]:
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        clean_mlp, ranking = seed_setup_mlp(seed, S, cfg, device)
        run_main_grid(seed, S, cfg, device, clean_mlp, ranking, m2_ref, main_cells, save_cb)
        run_nc_boundary(seed, S, cfg, device, clean_mlp, ranking, main_cells, nc_cells, save_cb)
        if str(seed) not in h3_rows:
            h3_rows[str(seed)] = run_h3(seed, S, cfg, device, ranking, main_cells, save_cb)
            save_cb()

    n_determinism_checked = sum(1 for r in main_cells.values() if r["determinism_ok"] is not None)
    n_determinism_failed = sum(1 for r in main_cells.values() if r["determinism_ok"] is False)

    out = dict(
        config=dict(smoke=smoke, seeds=cfg["seeds"], rates=list(cfg["rates"]), costs=list(cfg["costs"]),
                   nc_boundary_rates=list(cfg["nc_boundary_rates"]),
                   nc_boundary_costs=list(cfg["nc_boundary_costs"]),
                   nc_coverage_note="Neural Cleanse ran on an explicit boundary subset "
                                    f"({len(cfg['nc_boundary_rates']) * len(cfg['nc_boundary_costs'])} "
                                    f"of {len(cfg['rates']) * len(cfg['costs'])} grid cells per seed) "
                                    "-- NOT the full grid. See D2 in the M3 plan for the rationale.",
                   h3_operating_point=dict(rate=cfg["h3_rate"], cost=cfg["h3_cost"]),
                   h4_lightgbm_no_activations=True,
                   h4_note="LightGBM has no penultimate-feature representation and no gradient path "
                           "-- Spectral/AC/NC cannot run on it. No row is reported for lgb; this is "
                           "the H4 finding itself, not missing data."),
        main_grid=list(main_cells.values()),
        nc_boundary=list(nc_cells.values()),
        h3=list(h3_rows.values()),
        summary=dict(
            n_determinism_checked=n_determinism_checked,
            n_determinism_failed=n_determinism_failed,
            determinism_all_ok=bool(n_determinism_failed == 0),
        ),
    )
    (config.RESULTS / "detectors.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    if n_determinism_failed:
        print(f"WARNING: {n_determinism_failed}/{n_determinism_checked} cells FAILED the "
              "determinism self-check -- do not treat this run's numbers as matching M2's until "
              "investigated (see D1 in the M3 plan).")

    elapsed = time.time() - t0
    n_seeds = len(cfg["seeds"])
    print(f"\nelapsed={elapsed:.0f}s for {n_seeds} seed(s) "
          f"(~{elapsed / max(n_seeds, 1):.0f}s/seed)")
    print(f"checkpoint kept at {ckpt_path}")


if __name__ == "__main__":
    main()
