"""SPECTRE vision-control gate: prove the new scorer catches an MNIST patch backdoor before any
tabular SPECTRE number is interpreted.

This project's standing rule is that a null without a passing control is uninterpretable. SPECTRE is
a new detector in this codebase, so it clears the same MNIST BadNets control the other three cleared,
at the same criterion: poison recall >= 0.90 on every seed. If this gate fails, SPECTRE does not enter
the paper. See notes/20260826-decision-spectre-preregistration.md.

`scripts/01_vision_control.py` is deliberately NOT modified: its numbers (Spectral 0.9828, AC 0.9922,
NC 1.0) are cited in the manuscript, and re-running it with a fourth detector bolted on would put
those citations at risk for no gain. This script reuses the identical `src.vision_control` setup,
the same TARGET, POISON_FRAC, FEAT_DIM and seeds, so the two are directly comparable, and it scores
Spectral alongside SPECTRE on the SAME trained model per seed as an internal consistency check.

Run:  .venv/bin/python scripts/52_spectre_vision_control.py
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
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.detectors.spectre import spectre_scores

# Held identical to scripts/01_vision_control.py so the two gates are comparable.
TARGET = 0
POISON_FRAC = 0.05
FEAT_DIM = 512
SPECTRAL_K = 5

GATE_RECALL = 0.90
Z_THRESHOLDS = (2.0, 2.5, 3.0)

OUT = config.RESULTS / "spectre_vision_control.json"


def run_seed(seed: int, device: str) -> dict:
    vc.set_seed(seed)
    x_tr, y_tr, x_te, y_te = vc.load_mnist()
    x_p, y_p, poison_idx = vc.poison_trainset(x_tr, y_tr, TARGET, POISON_FRAC, seed)
    model = vc.train_victim(x_p, y_p, seed, epochs=3, device=device, feat_dim=FEAT_DIM)

    target_mask = (y_p == TARGET).numpy()
    is_poison = np.zeros(len(y_p), dtype=bool)
    is_poison[poison_idx] = True
    is_poison_t = is_poison[target_mask]
    feats_t = vc.penultimate_features(model, x_p[target_mask], device)
    frac = float(is_poison_t.mean())

    spec = spectral_scores(feats_t, n_components=SPECTRAL_K)
    spct = spectre_scores(feats_t, expected_frac=frac)

    def auc(scores):
        if is_poison_t.sum() == 0 or is_poison_t.sum() == len(is_poison_t):
            return None
        return float(roc_auc_score(is_poison_t, scores))

    return {
        "seed": seed,
        "target_poison_frac": frac,
        "spectral_recall": poison_recall(flag_by_scores(spec, expected_frac=frac), is_poison_t),
        "spectre_recall": poison_recall(flag_by_scores(spct, expected_frac=frac), is_poison_t),
        "spectral_auc": auc(spec),
        "spectre_auc": auc(spct),
        "spectre_mad_recall": {
            str(z): poison_recall(flag_by_mad_threshold(spct, z_thresh=z), is_poison_t)
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
        print(f"  seed {s}: spectre_recall={row['spectre_recall']:.4f} "
              f"spectral_recall={row['spectral_recall']:.4f} "
              f"spectre_auc={row['spectre_auc']:.4f}  ({time.time() - t0:.0f}s)")

    spectre_pass = all(r["spectre_recall"] >= GATE_RECALL for r in rows)
    summary = {
        "spectre_recall_mean": float(np.mean([r["spectre_recall"] for r in rows])),
        "spectre_recall_min": float(np.min([r["spectre_recall"] for r in rows])),
        "spectral_recall_mean": float(np.mean([r["spectral_recall"] for r in rows])),
        "spectre_auc_mean": float(np.mean([r["spectre_auc"] for r in rows])),
        "spectre_recall_all_seeds": spectre_pass,
        "gate_recall": GATE_RECALL,
    }
    OUT.write_text(json.dumps(
        {"config": {"target": TARGET, "poison_frac": POISON_FRAC, "feat_dim": FEAT_DIM,
                    "spectral_k": SPECTRAL_K, "seeds": list(config.SEEDS),
                    "z_thresholds": list(Z_THRESHOLDS)},
         "per_seed": rows, "summary": summary}, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"SPECTRE gate (>={GATE_RECALL} recall, all seeds): {'PASS' if spectre_pass else 'FAIL'}")
    print(f"wrote {OUT}  elapsed={time.time() - t0:.0f}s")
    if not spectre_pass:
        raise SystemExit(
            "SPECTRE vision-control gate FAILED -- per the pre-registration, SPECTRE does not enter "
            "the paper. Do not run the tabular script.")
    print("SPECTRE vision-control gate PASSED.")


if __name__ == "__main__":
    main()
