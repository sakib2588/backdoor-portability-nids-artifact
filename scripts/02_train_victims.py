"""M2 victims report: train the clean LightGBM + MLP victims across seeds and record their quality
(clean accuracy, botnet precision/recall, per-split balance). Written to results/victims.json.

Run:  python scripts/02_train_victims.py [--smoke]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup
from src import config
from src.data import apply_standardiser
from src.models import train_lightgbm, train_mlp, predict_labels


def _metrics(model, x_std, y, device):
    pred = predict_labels(model, x_std, device)
    y = np.asarray(y)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    n_pos = int((y == 1).sum())
    return dict(clean_acc=float((pred == y).mean()),
                botnet_precision=(tp / (tp + fp) if (tp + fp) else 0.0),
                botnet_recall=(tp / n_pos if n_pos else 0.0),
                majority_baseline=float(1.0 - y.mean()))


def main():
    smoke = "--smoke" in sys.argv
    cfg = get_config(smoke)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = load_setup(cfg)
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])

    rows = []
    for seed in cfg["seeds"]:
        lgb = train_lightgbm(x_tr_std, S["y_tr"], seed)
        mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
        rows.append(dict(seed=seed,
                         lgb=_metrics(lgb, x_te_std, S["y_te"], device),
                         mlp=_metrics(mlp, x_te_std, S["y_te"], device)))
        print(f"seed {seed}: lgb={rows[-1]['lgb']}  mlp={rows[-1]['mlp']}")

    out = dict(config=dict(smoke=smoke, seeds=cfg["seeds"],
                           train_balance=S["train_balance"], test_balance=S["test_balance"],
                           note="victims trained fresh on the modern stack; TabularBench pretrained "
                                "weights are on the uninstallable torch==1.12.1 pin"),
               per_seed=rows)
    (config.RESULTS / "victims.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {config.RESULTS / 'victims.json'}")


if __name__ == "__main__":
    main()
