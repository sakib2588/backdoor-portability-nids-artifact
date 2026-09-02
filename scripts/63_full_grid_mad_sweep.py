"""Round-2 objection #7 (harsh-review tracker): report MAD recall and FPR over the full 25-cell
CTU-13 grid, not just the 5 pre-registered window cells scripts/08_evasion_ablation.py covers.

Same detector recipe as scripts/08_evasion_ablation.py (flag_by_mad_threshold at the three
pre-registered z-thresholds {2.0, 2.5, 3.0}), extended from TARGET_CELLS to the full 5-rate x
5-cost grid used in scripts/05_run_detectors.py's main_grid, with a determinism self-check against
results/detectors.json's fixed-budget recall for the same reason 08_evasion_ablation.py has one.

Checkpointed per cell (not per seed): a timing probe measured ~109s/cell beyond the ~16s/seed
fixed cost of retraining the clean model and SHAP ranking, so 25 cells x 5 seeds is ~3.8 hours --
well over the project's ~10-minute checkpoint threshold. Follows scripts/05_run_detectors.py's
atomic-write pattern (tmp file + rename) so a kill mid-run loses at most the in-flight cell.

Run:  python scripts/63_full_grid_mad_sweep.py --seeds 42       # single-seed check first
      python scripts/63_full_grid_mad_sweep.py                  # full 5-seed run (resumable)
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
SPECTRAL_K = 5
AC_PCA_COMPONENTS = 10
AC_SILHOUETTE_SAMPLE = 10000
Z_THRESHOLDS = (2.0, 2.5, 3.0)

# full grid, matching scripts/05_run_detectors.py's main_grid (detectors.json's config)
RATES = [0.005, 0.01, 0.02, 0.05, 0.1]
COSTS = [1, 2, 4, 8, 16]

CKPT = "full_grid_mad_sweep.checkpoint.json"
OUT = "full_grid_mad_sweep.json"


def cell_key(seed, rate, cost) -> str:
    return f"{seed}|{rate}|{cost}"


def _config_key(cfg: dict) -> dict:
    return dict(rates=RATES, costs=COSTS, z_thresholds=list(Z_THRESHOLDS),
                mlp_epochs=cfg["mlp_epochs"])


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
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} cells already done")
    return blob.get("rows", {})


def save_checkpoint(path, key, rows):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(path)


def _load_original_index():
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
    print(f"device={device}  seeds={cfg['seeds']}  rates={RATES}  costs={COSTS}  "
          f"z_thresholds={Z_THRESHOLDS}  ({len(RATES) * len(COSTS)} cells/seed)")

    S = load_setup(cfg)
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    ref_index = _load_original_index()

    ckpt_path = config.RESULTS / CKPT
    key = _config_key(cfg)
    rows = load_checkpoint(ckpt_path, key)
    save_cb = lambda: save_checkpoint(ckpt_path, key, rows)

    x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
    stats = raw_trigger_stats(x_tr_raw, constraints)

    for seed in cfg["seeds"]:
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        clean_mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET,
                                     kind="mlp", device=device)

        for rate in RATES:
            for cost in COSTS:
                key_str = cell_key(seed, rate, cost)
                if key_str in rows:
                    continue

                realiz, _ = build_trigger(ranking, cost, x_tr_raw, constraints, features, bounds,
                                          stats=stats)
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
                    n_true_pos = int(np.sum(flag & is_poison))
                    n_flagged = int(flag.sum())
                    adaptive[str(z)] = dict(
                        recall=poison_recall(flag, is_poison) if has_poison else None,
                        n_flagged=n_flagged, n_true_positive=n_true_pos,
                        n_false_positive=n_flagged - n_true_pos,
                        fpr=(n_flagged - n_true_pos) / n_benign if n_benign else None)

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

                rows[key_str] = dict(seed=seed, rate=rate, cost=cost,
                           fixed_budget_recall=fixed_recall,
                           fixed_budget_recall_ref=(ref["spectral"]["recall"] if ref else None),
                           determinism_ok=determinism_ok,
                           spectral_auc=auc, n_poison=n_poison, n_benign=n_benign,
                           adaptive=adaptive,
                           ac_fixed_recall=ac_fixed_recall,
                           ac_fixed_recall_ref=(ref["ac"]["recall"] if ref else None))
                save_cb()
                print(f"  [{len(rows)}/{len(RATES)*len(COSTS)*len(cfg['seeds'])}] "
                      f"rate={rate} cost={cost} seed={seed}: fixed={fixed_recall} auc={auc} "
                      f"z3.0_recall={adaptive['3.0']['recall']} z3.0_fpr={adaptive['3.0']['fpr']:.4f} "
                      f"({time.time() - t0:.0f}s elapsed)")

    out = dict(z_thresholds=list(Z_THRESHOLDS), rates=RATES, costs=COSTS, seeds=cfg["seeds"],
               rows=list(rows.values()),
               note="Full 25-cell CTU-13 grid, MAD adaptive threshold (flag_by_mad_threshold) at "
                    "three pre-registered z-thresholds, plus fixed_budget recall/AUC as a "
                    "determinism self-check against results/detectors.json's main_grid. Extends "
                    "results/evasion_ablation.json's 5 pre-registered window cells to the full "
                    "grid, closing Round 2's harsh-review objection #7 (per-cost-mean MAD "
                    "recall+FPR over the full grid, not just the window). fpr is computed against "
                    "n_benign (the target-class pool), matching Table II's merged FPR convention.")
    (config.RESULTS / OUT).write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/{OUT} ({len(rows)} rows)")
    elapsed = time.time() - t0
    print(f"elapsed={elapsed:.0f}s for {len(cfg['seeds'])} seed(s)")
    print(f"checkpoint kept at {ckpt_path}")


if __name__ == "__main__":
    main()
