"""M1 vision control: prove the three detectors catch an MNIST patch backdoor.

Exit gate (per seed, on the poisoned target class):
  - Spectral Signatures poison-recall  >= 0.90
  - Activation Clustering poison-recall >= 0.90
  - Neural Cleanse flags the target class with anomaly index > 2
A detector that fails here is mis-implemented; fix it before any NIDS run.
Numbers are written to results/vision_control.json; nothing is hardcoded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src import vision_control as vc
from src.detectors import poison_recall
from src.detectors.activation_clustering import detect as ac_detect
from src.detectors.neural_cleanse import anomaly_index, reverse_engineer
from src.detectors.spectral import flag_by_scores, spectral_scores

TARGET = 0
POISON_FRAC = 0.05
NC_SAMPLE = 200  # clean images used for trigger inversion
FEAT_DIM = 512  # widened from the original 128 -- see notes/20260714-decision-session-handover.md
SPECTRAL_K = 5  # top-k singular subspace for Spectral scoring (k=1 is vanilla Tran et al.;
#                 the backdoor tail hides on v2-v5, so k>1 is needed -- robust for any k in
#                 {3,5,10}, see notes/20260714-decision-spectral-ceiling.md. Applied to NIDS too.


def run_seed(seed: int, device: str) -> dict:
    vc.set_seed(seed)
    x_tr, y_tr, x_te, y_te = vc.load_mnist()
    x_p, y_p, poison_idx = vc.poison_trainset(x_tr, y_tr, TARGET, POISON_FRAC, seed)
    model = vc.train_victim(x_p, y_p, seed, epochs=3, device=device, feat_dim=FEAT_DIM)

    clean_acc = vc.clean_accuracy(model, x_te, y_te, device)
    asr = vc.attack_success_rate(model, x_te, y_te, TARGET, device)

    # class-conditioned activations for the TARGET class in the poisoned trainset
    target_mask = (y_p == TARGET).numpy()
    is_poison = np.zeros(len(y_p), dtype=bool)
    is_poison[poison_idx] = True
    is_poison_t = is_poison[target_mask]
    feats_t = vc.penultimate_features(model, x_p[target_mask], device)

    target_poison_frac = float(is_poison_t.mean())  # true poison rate WITHIN the target-conditioned subset
    scores = spectral_scores(feats_t, n_components=SPECTRAL_K)
    spec_flagged = flag_by_scores(scores, expected_frac=target_poison_frac)
    spec_recall = poison_recall(spec_flagged, is_poison_t)

    ac_mask, ac_sil = ac_detect(feats_t, n_components=10, seed=seed)
    ac_recall = poison_recall(ac_mask, is_poison_t)

    sample = x_te[:NC_SAMPLE]
    norms = reverse_engineer(model, 10, sample, device=device)
    nc_index, nc_flagged = anomaly_index(norms)

    return {
        "seed": seed,
        "clean_acc": clean_acc,
        "asr": asr,
        "spectral_recall": spec_recall,
        "target_poison_frac": target_poison_frac,
        "ac_recall": ac_recall,
        "ac_silhouette": ac_sil,
        "nc_anomaly_index": nc_index,
        "nc_flagged_class": nc_flagged,
        "nc_mask_norms": norms.tolist(),
    }


def main() -> None:
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    rows = [run_seed(s, device) for s in config.SEEDS]

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "clean_acc_mean": mean("clean_acc"),
        "asr_mean": mean("asr"),
        "spectral_recall_mean": mean("spectral_recall"),
        "ac_recall_mean": mean("ac_recall"),
        "nc_index_mean": mean("nc_anomaly_index"),
        "spectral_recall_all_seeds": all(r["spectral_recall"] >= 0.9 for r in rows),
        "ac_recall_all_seeds": all(r["ac_recall"] >= 0.9 for r in rows),
        "nc_flags_target_all_seeds": all(
            r["nc_flagged_class"] == TARGET and r["nc_anomaly_index"] > 2.0 for r in rows
        ),
    }
    out = {"config": {"target": TARGET, "poison_frac": POISON_FRAC, "seeds": list(config.SEEDS),
                      "feat_dim": FEAT_DIM, "spectral_k": SPECTRAL_K},
           "per_seed": rows, "summary": summary}
    path = config.RESULTS / "vision_control.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))

    spec_pass = summary["spectral_recall_all_seeds"]
    ac_pass = summary["ac_recall_all_seeds"]
    nc_pass = summary["nc_flags_target_all_seeds"]
    print(f"Spectral  gate (>=0.90 recall): {'PASS' if spec_pass else 'FAIL'}")
    print(f"AC        gate (>=0.90 recall): {'PASS' if ac_pass else 'FAIL'}")
    print(f"NeuralCln gate (flags target, index>2): {'PASS' if nc_pass else 'FAIL'}")
    print(f"results written to {path}")
    if not (spec_pass and ac_pass and nc_pass):
        raise SystemExit("M1 gate FAILED -- a detector is mis-implemented; fix before NIDS.")
    print("M1 vision-control gate PASSED.")


if __name__ == "__main__":
    main()
