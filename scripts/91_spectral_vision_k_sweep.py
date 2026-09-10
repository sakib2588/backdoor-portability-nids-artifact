#!/usr/bin/env python3
"""Spectral Signatures' vision-control poison recall at k=1 against k=5.

Round-v6 review asked whether the paper's stated rule "no parameter is adjusted to
make it pass" survives the disclosed change of Spectral's top-k from the published
k=1 to k=5. That change was justified on a vision-control diagnostic whose result
was never stored: scripts/01_vision_control.py hard-codes SPECTRAL_K = 5, and
results/spectral_k1_ablation.json holds tabular cells only. This script measures
the missing number and nothing else, reusing 01's victim exactly (same seeds, same
MNIST poisoning, same feature width) and skipping Activation Clustering and the
Neural Cleanse inversion, which are what make 01 slow.

Checkpointed per seed to results/spectral_vision_k_sweep.checkpoint.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import vision_control as vc  # noqa: E402
from src.detectors.spectral import flag_by_scores, spectral_scores  # noqa: E402

TARGET = 0
POISON_FRAC = 0.05
FEAT_DIM = 512
SEEDS = [42, 123, 456, 789, 1337]
K_VALUES = [1, 3, 5, 10]
BAR = 0.90  # the vision-control bar stated in the manuscript

OUT = ROOT / "results" / "spectral_vision_k_sweep.json"
CKPT = ROOT / "results" / "spectral_vision_k_sweep.checkpoint.json"
FINGERPRINT = {
    "target": TARGET, "poison_frac": POISON_FRAC, "feat_dim": FEAT_DIM,
    "seeds": SEEDS, "k_values": K_VALUES, "epochs": 3,
}


def poison_recall(flagged: np.ndarray, is_poison: np.ndarray) -> float:
    if is_poison.sum() == 0:
        return float("nan")
    return float((flagged & is_poison).sum() / is_poison.sum())


def load_ckpt() -> dict:
    if CKPT.exists():
        d = json.loads(CKPT.read_text())
        if d.get("fingerprint") == FINGERPRINT:
            return d
        print("checkpoint fingerprint differs from this config, ignoring it")
    return {"fingerprint": FINGERPRINT, "per_seed": {}}


def save_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


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

    out = {"target_poison_frac": frac, "asr": vc.attack_success_rate(model, x_te, y_te, TARGET, device)}
    for k in K_VALUES:
        scores = spectral_scores(feats_t, n_components=k)
        flagged = flag_by_scores(scores, expected_frac=frac)
        out[f"recall_k{k}"] = poison_recall(flagged, is_poison_t)
    return out


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = load_ckpt()
    done = state["per_seed"]
    for seed in SEEDS:
        if str(seed) in done:
            print(f"seed {seed} already done, skipping")
            continue
        t0 = time.time()
        done[str(seed)] = run_seed(seed, device)
        done[str(seed)]["seed_seconds"] = round(time.time() - t0, 1)
        save_atomic(CKPT, state)
        r = done[str(seed)]
        print(f"seed {seed}: " + "  ".join(f"k={k} {r[f'recall_k{k}']:.4f}" for k in K_VALUES)
              + f"  ({r['seed_seconds']}s)")

    summary = {}
    for k in K_VALUES:
        vals = [done[str(s)][f"recall_k{k}"] for s in SEEDS]
        summary[f"k{k}"] = {
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "per_seed": [float(v) for v in vals],
            "clears_bar_every_seed": bool(np.min(vals) >= BAR),
        }
    result = {"config": {**FINGERPRINT, "bar": BAR, "device": device},
              "per_seed": done, "summary": summary}
    save_atomic(OUT, result)
    print("\n--- summary, bar %.2f on every seed ---" % BAR)
    for k in K_VALUES:
        s = summary[f"k{k}"]
        print(f"k={k:2d}  mean {s['mean']:.4f}  min {s['min']:.4f}  "
              f"clears={s['clears_bar_every_seed']}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
