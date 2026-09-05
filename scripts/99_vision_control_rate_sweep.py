#!/usr/bin/env python3
"""The vision control at MATCHED poison rates: does the fixed budget starve on MNIST too?

WHY THIS IS THE MISSING CONTROL. The committed vision control plants the patch at 5% of
non-target images, which lands at 31.3% of the target class. The tabular evasion window is
at 0.5% to 1% of the target class. The inherited fixed budget is 1.5 x expected_frac x n,
so on the vision control a defender may remove 47% of the class and on the window 0.75%.
Poison share is the one variable in the paper's own starvation derivation
(05_results.tex:82-90), and the control sits 30 to 60 times away from the experiment on it.
A control at that distance cannot fail to pass, and so it cannot control for starvation.

THE TWO OUTCOMES, BOTH INFORMATIVE, STATED BEFORE THE RUN.
  * Spectral's fixed-budget recall collapses on MNIST at 0.5% while its AUC holds: the
    starvation window is a low-rate phenomenon that exists on the vision substrate too, and
    the paper's finding generalizes beyond tabular. That is a stronger paper.
  * Spectral's fixed-budget recall holds on MNIST at 0.5%: the substrate genuinely matters,
    and the current framing stands with a control that finally tests it.

WHAT IS AND IS NOT MATCHED. Rate is matched. Label regime is not: this keeps the committed
control's dirty-label patch (labels flipped) against the tabular attack's clean-label
trigger. So this isolates rate and leaves the label confound for a later, harder variant.
Say that plainly in any write-up.

Everything else is the committed control unchanged: same SmallCNN at feat_dim 512, three
epochs, same TARGET 0, same Spectral top-k of 5, same rules. The only knob is the within-
target poison rate, solved per seed from the actual class counts so the requested rate is
what is planted rather than an approximation.

CPU and GPU cost: the committed control is ~3 epochs on 60k MNIST images per seed, well
under a minute on the 3060 Ti. Six rates x five seeds is a few minutes. Checkpointed per
(rate, seed) regardless.
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

TARGET = 0
FEAT_DIM = 512
SPECTRAL_K = 5
EPOCHS = 3
Z = 3.0

# Within-target-class poison rates. The last is the committed control's own rate, kept as
# the reproduction anchor: if this script does not reproduce ~0.98 recall there, it is not
# in the committed code path and nothing below it is interpretable.
TARGET_RATES = [0.005, 0.01, 0.02, 0.05, 0.10, 0.3134]

OUT = config.RESULTS / "vision_control_rate_sweep.json"
CKPT = config.RESULTS / "vision_control_rate_sweep.checkpoint.json"


def frac_for_rate(y: torch.Tensor, target: int, rate: float) -> float:
    """Non-target fraction f such that poisoned-within-target = rate, from actual counts.

    With n_t target rows and n_o non-target rows, planting f*n_o rows into the target class
    gives rate = f*n_o / (n_t + f*n_o), so f = rate*n_t / (n_o*(1 - rate)).
    """
    yn = y.numpy()
    n_t = int((yn == target).sum())
    n_o = int((yn != target).sum())
    return rate * n_t / (n_o * (1.0 - rate))


def load_ckpt(key):
    if not CKPT.exists():
        return {}
    try:
        blob = json.loads(CKPT.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != json.loads(json.dumps(key)):
        print("checkpoint config mismatch -- starting fresh")
        return {}
    print(f"resuming: {len(blob.get('rows', {}))} unit(s) done")
    return blob.get("rows", {})


def save_ckpt(key, rows):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(CKPT)


def run_cell(rate: float, seed: int, device: str) -> dict:
    vc.set_seed(seed)
    x_tr, y_tr, x_te, y_te = vc.load_mnist()
    frac = frac_for_rate(y_tr, TARGET, rate)
    x_p, y_p, poison_idx = vc.poison_trainset(x_tr, y_tr, TARGET, frac, seed)
    model = vc.train_victim(x_p, y_p, seed, epochs=EPOCHS, device=device, feat_dim=FEAT_DIM)

    clean_acc = vc.clean_accuracy(model, x_te, y_te, device)
    asr = vc.attack_success_rate(model, x_te, y_te, TARGET, device)

    target_mask = (y_p == TARGET).numpy()
    is_poison = np.zeros(len(y_p), dtype=bool)
    is_poison[poison_idx] = True
    is_poison_t = is_poison[target_mask]
    feats_t = vc.penultimate_features(model, x_p[target_mask], device)

    achieved = float(is_poison_t.mean())
    scores = spectral_scores(feats_t, n_components=SPECTRAL_K)
    fixed = poison_recall(flag_by_scores(scores, expected_frac=achieved), is_poison_t)
    mad_flag = flag_by_mad_threshold(scores, z_thresh=Z)
    n_clean = int((~is_poison_t).sum())
    return dict(
        requested_rate=rate, achieved_rate=achieved, non_target_frac=frac,
        n_poison=int(is_poison_t.sum()), n_target_class=int(target_mask.sum()),
        budget_rows=int(np.ceil(1.5 * achieved * target_mask.sum())),
        clean_acc=clean_acc, asr=asr,
        spectral_auc=float(roc_auc_score(is_poison_t, scores)) if 0 < is_poison_t.sum() < len(is_poison_t) else None,
        spectral_fixed_recall=fixed,
        spectral_mad_recall=poison_recall(mad_flag, is_poison_t),
        spectral_mad_fpr=float((mad_flag & ~is_poison_t).sum() / n_clean) if n_clean else None,
    )


def main() -> int:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.ensure_dirs()
    key = dict(rates=TARGET_RATES, seeds=list(config.SEEDS), target=TARGET, feat_dim=FEAT_DIM,
               spectral_k=SPECTRAL_K, epochs=EPOCHS, z=Z, label_regime="dirty")
    rows = load_ckpt(key)
    print(f"device={device} rates={TARGET_RATES} seeds={list(config.SEEDS)}")

    for rate in TARGET_RATES:
        for seed in config.SEEDS:
            k = f"{rate}|{seed}"
            if k in rows:
                continue
            print(f"--- rate {rate} seed {seed} ({time.time() - t0:.0f}s) ---")
            rows[k] = run_cell(rate, seed, device)
            save_ckpt(key, rows)
            r = rows[k]
            print(f"  achieved={r['achieved_rate']:.4f} n_poison={r['n_poison']} "
                  f"budget={r['budget_rows']} | auc={r['spectral_auc']:.4f} "
                  f"fixed={r['spectral_fixed_recall']:.4f} mad={r['spectral_mad_recall']:.4f} "
                  f"| asr={r['asr']:.3f}")

    table = []
    for rate in TARGET_RATES:
        cells = [rows[f"{rate}|{s}"] for s in config.SEEDS if f"{rate}|{s}" in rows]
        if not cells:
            continue
        def m(k):
            v = [c[k] for c in cells if c[k] is not None]
            return dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) if v else None
        table.append(dict(rate=rate, n=len(cells),
                          achieved_rate=m("achieved_rate"), n_poison=m("n_poison"),
                          asr=m("asr"), auc=m("spectral_auc"),
                          fixed_recall=m("spectral_fixed_recall"),
                          mad_recall=m("spectral_mad_recall"), mad_fpr=m("spectral_mad_fpr")))

    blob = dict(
        _purpose="vision control at matched within-target poison rates; label regime stays dirty",
        _reproduction_anchor="rate 0.3134 must reproduce the committed control's ~0.98 recall",
        device=device, label_regime="dirty-label BadNets patch, labels flipped",
        rows=rows, table=table, elapsed_s=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(blob, indent=2) + "\n")

    print(f"\n{'rate':>7} {'n_poison':>9} {'ASR':>6} {'AUC':>7} {'fixed':>7} {'MAD':>7}")
    for t in table:
        print(f"{t['rate']:>7} {t['n_poison']['mean']:>9.0f} {t['asr']['mean']:>6.3f} "
              f"{t['auc']['mean']:>7.4f} {t['fixed_recall']['mean']:>7.4f} {t['mad_recall']['mean']:>7.4f}")
    print(f"wrote {OUT}  ({blob['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
