"""M4 follow-up: does the evasion window persist at Tran et al.'s original k=1, or is it an artefact
of this project's k=5 deviation (chosen on a vision-control diagnostic, see
notes/20260714-decision-spectral-ceiling.md)?

Retrains the MLP victim at the 4 evasion-window cells (Table I: rate in {0.005, 0.01}, cost in
{8, 16}) x 5 seeds, scores Spectral Signatures at BOTH n_components=1 (Tran et al.'s original) and
n_components=SPECTRAL_K=5 (this project's convention) on the identical trained model and poisoned
population, so the two arms differ ONLY in k, nothing else. No k=1 tabular run exists anywhere on
disk before this script -- every committed number used k=5.

Run:  python scripts/12_spectral_k1_ablation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup
from src import config
from src.data import apply_standardiser
from src.detectors import poison_recall
from src.detectors.spectral import flag_by_scores, spectral_scores
from src.models import mlp_penultimate_features, train_mlp
from src.trigger import shap_rank_features, build_trigger, raw_trigger_stats, apply_trigger
from src.poison import poison_trainset_cleanlabel

TARGET = config.ATTACK_TARGET
EVASION_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]  # Table I, tab:evasion
K_TRAN = 1
K_PROJECT = 5

OUT = config.RESULTS / "spectral_k1_ablation.json"


def main():
    t0 = time.time()
    cfg = get_config(smoke=False)
    device = "cpu"
    S = load_setup(cfg)
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    stats = raw_trigger_stats(x_tr_raw, constraints)

    rows = []
    for seed in cfg["seeds"]:
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
        clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET,
                                     kind="mlp", device=device)

        for rate, cost in EVASION_CELLS:
            realiz, _ = build_trigger(ranking, cost, x_tr_raw, constraints, features, bounds, stats=stats)
            x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
                x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
                constraints=constraints, feature_names=features, bounds=bounds)
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            benign_pos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(benign_pos, poison_idx)
            has_poison = bool(is_poison.sum() > 0)
            feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
            expected_frac = float(is_poison.mean())   # oracle: the TRUE poison fraction, same
                                                        # convention as scripts/05_run_detectors.py:170

            def score_at_k(k):
                scores = spectral_scores(feats, n_components=k)
                flag = flag_by_scores(scores, expected_frac=expected_frac)
                recall = poison_recall(flag, is_poison) if has_poison else None
                auc = (float(roc_auc_score(is_poison, scores))
                      if has_poison and is_poison.sum() < len(is_poison) else None)
                return recall, auc

            k1_recall, k1_auc = score_at_k(K_TRAN)
            k5_recall, k5_auc = score_at_k(K_PROJECT)

            row = dict(seed=seed, rate=rate, cost=cost,
                      n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
                      k1_recall=k1_recall, k1_auc=k1_auc,
                      k5_recall=k5_recall, k5_auc=k5_auc)
            rows.append(row)
            print(f"  rate={rate} cost={cost} k1_recall={k1_recall:.4f} k1_auc={k1_auc:.4f} "
                  f"k5_recall={k5_recall:.4f} k5_auc={k5_auc:.4f}")

    k1_recalls = [r["k1_recall"] for r in rows]
    k1_aucs = [r["k1_auc"] for r in rows]
    k5_recalls = [r["k5_recall"] for r in rows]
    k5_aucs = [r["k5_auc"] for r in rows]

    out = dict(
        evasion_cells=EVASION_CELLS, seeds=cfg["seeds"], k_tran=K_TRAN, k_project=K_PROJECT,
        rows=rows,
        summary=dict(
            k1_recall_mean=float(np.mean(k1_recalls)), k1_auc_mean=float(np.mean(k1_aucs)),
            k5_recall_mean=float(np.mean(k5_recalls)), k5_auc_mean=float(np.mean(k5_aucs)),
        ),
        note="k1_* uses Tran et al.'s (2018) original n_components=1; k5_* uses this project's "
             "n_components=5 (chosen on a vision-control diagnostic, see "
             "notes/20260714-decision-spectral-ceiling.md). Both are scored on the IDENTICAL "
             "trained model and poisoned population per row -- only k differs. expected_frac is the "
             "true poison fraction for that row (oracle), matching scripts/05_run_detectors.py:170, "
             "not a fixed guess.",
    )
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    print(f"\nwrote {OUT}  elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
