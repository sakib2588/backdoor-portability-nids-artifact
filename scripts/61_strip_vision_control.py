"""STRIP vision-control gate: prove the perturb-and-blend entropy test catches an MNIST patch
backdoor before any tabular STRIP number is interpreted.

Same standing rule as SPECTRE's gate (scripts/52_spectre_vision_control.py): a null without a passing
control is uninterpretable. STRIP clears the same MNIST BadNets control the other four detectors
cleared, at the same criterion: poison recall >= 0.90 on every seed. If this gate fails, STRIP does
not enter the paper. `scripts/01_vision_control.py` is left untouched for the same reason SPECTRE's
gate leaves it untouched: its cited numbers must not depend on a later detector being bolted on.

Blend pool is the held-out MNIST test set (clean, disjoint from the poisoned training set STRIP
scores) -- Gao et al. also draw STRIP's reference pool from clean held-out data, not the query set
itself.

Run:  .venv/bin/python scripts/61_strip_vision_control.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src import vision_control as vc
from src.detectors import poison_recall
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores
from src.detectors.strip import strip_scores

TARGET = 0
POISON_FRAC = 0.05
FEAT_DIM = 512
N_TRIALS = 64
ALPHA = 0.5

GATE_RECALL = 0.90
Z_THRESHOLDS = (2.0, 2.5, 3.0)

OUT = config.RESULTS / "strip_vision_control.json"


def run_seed(seed: int, device: str) -> dict:
    vc.set_seed(seed)
    x_tr, y_tr, x_te, y_te = vc.load_mnist()
    x_p, y_p, poison_idx = vc.poison_trainset(x_tr, y_tr, TARGET, POISON_FRAC, seed)
    model = vc.train_victim(x_p, y_p, seed, epochs=3, device=device, feat_dim=FEAT_DIM)

    target_mask = (y_p == TARGET).numpy()
    is_poison = np.zeros(len(y_p), dtype=bool)
    is_poison[poison_idx] = True
    is_poison_t = is_poison[target_mask]
    frac = float(is_poison_t.mean())

    x_query = x_p[target_mask].numpy()
    blend_pool = x_te.numpy()

    scores = strip_scores(model, x_query, blend_pool, n_trials=N_TRIALS, alpha=ALPHA,
                           device=device, seed=seed)

    def auc(sc):
        if is_poison_t.sum() == 0 or is_poison_t.sum() == len(is_poison_t):
            return None
        return float(roc_auc_score(is_poison_t, sc))

    return {
        "seed": seed,
        "target_poison_frac": frac,
        "strip_recall": poison_recall(flag_by_scores(scores, expected_frac=frac), is_poison_t),
        "strip_auc": auc(scores),
        "strip_mad_recall": {
            str(z): poison_recall(flag_by_mad_threshold(scores, z_thresh=z), is_poison_t)
            for z in Z_THRESHOLDS
        },
    }


def main() -> None:
    t0 = time.time()
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}  seeds = {list(config.SEEDS)}")

    rows = []
    for s in config.SEEDS:
        row = run_seed(s, device)
        rows.append(row)
        print(f"  seed {s}: strip_recall={row['strip_recall']:.4f} "
              f"strip_auc={row['strip_auc']:.4f}  ({time.time() - t0:.0f}s)")

    strip_pass = all(r["strip_recall"] >= GATE_RECALL for r in rows)
    summary = {
        "strip_recall_mean": float(np.mean([r["strip_recall"] for r in rows])),
        "strip_recall_min": float(np.min([r["strip_recall"] for r in rows])),
        "strip_auc_mean": float(np.mean([r["strip_auc"] for r in rows])),
        "strip_recall_all_seeds": strip_pass,
        "gate_recall": GATE_RECALL,
    }
    OUT.write_text(json.dumps(
        {"config": {"target": TARGET, "poison_frac": POISON_FRAC, "feat_dim": FEAT_DIM,
                    "n_trials": N_TRIALS, "alpha": ALPHA, "seeds": list(config.SEEDS),
                    "z_thresholds": list(Z_THRESHOLDS)},
         "per_seed": rows, "summary": summary}, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"STRIP gate (>={GATE_RECALL} recall, all seeds): {'PASS' if strip_pass else 'FAIL'}")
    print(f"wrote {OUT}  elapsed={time.time() - t0:.0f}s")
    if not strip_pass:
        raise SystemExit(
            "STRIP vision-control gate FAILED -- per project rule, STRIP does not enter the paper. "
            "Do not run the tabular script.")
    print("STRIP vision-control gate PASSED.")


if __name__ == "__main__":
    main()
