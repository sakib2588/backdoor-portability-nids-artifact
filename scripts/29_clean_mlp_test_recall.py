"""Review-round check (D-9, 2026-08-13): the paper reports clean accuracy (0.9991 mean, MLP) but
never the MLP's clean botnet-class recall, and on a test block that is 0.74% positive (407/55082,
see results/tabular_positive_control.json's test_balance), accuracy alone is close to the
always-benign baseline of 0.9926. LightGBM's equivalent (19/407 missed -> recall 0.9533) is already
implicit in the paper's own false-negative-floor sentence; the MLP number never appears anywhere.

Trains the same clean (unpoisoned) MLP scripts/05_run_detectors.py trains per seed
(seed_setup_mlp's clean_mlp, before any poisoning), for the same 5 committed seeds, and reports its
botnet-class (class 1) recall on the held-out test set.

Run: python scripts/29_clean_mlp_test_recall.py
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

from _m2_common import get_config, load_setup
from src import config
from src.data import apply_standardiser
from src.models import predict_labels, train_mlp

OUT = config.RESULTS / "clean_mlp_test_recall.json"


def main():
    t0 = time.time()
    cfg = get_config(smoke=False)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  seeds={cfg['seeds']}")

    S = load_setup(cfg)
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])
    y_te = S["y_te"]
    botnet_mask = y_te == 1
    n_botnet = int(botnet_mask.sum())
    print(f"test set: n={len(y_te)}, botnet={n_botnet}")

    rows = []
    for seed in cfg["seeds"]:
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        clean_mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
        preds = predict_labels(clean_mlp, x_te_std, device)
        recall = float((preds[botnet_mask] == 1).mean())
        n_missed = int((preds[botnet_mask] != 1).sum())
        rows.append(dict(seed=seed, botnet_recall=recall, n_botnet=n_botnet, n_missed=n_missed))
        print(f"  botnet recall={recall:.4f} ({n_botnet - n_missed}/{n_botnet}, missed={n_missed})")

    mean_recall = float(np.mean([r["botnet_recall"] for r in rows]))
    out = dict(n_test=len(y_te), n_botnet=n_botnet, always_benign_accuracy=1 - n_botnet / len(y_te),
              per_seed=rows, mean_recall=mean_recall)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
