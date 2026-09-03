"""Re-run the MNIST vision control's Activation Clustering arm with both reductions, inside ONE
code path, one victim per seed, one activation matrix per seed.

Why this script exists: the manuscript reduces Activation Clustering with PCA and states the AC
verdict "should be read against that substitution", because Chen et al.'s paper reports ICA
(FastICA) as their primary recipe. This project already ran FastICA once and removed it --
notes/20260714-bug-ac-fastica-instability.md records a per-seed poison-recall spread of
0.470-1.000 on this same MNIST vision control, failing the project's own gate (recall >= 0.90 on
EVERY seed), the same gate scripts/01_vision_control.py applies. Switching to PCA raised recall to
0.92-1.00 and passed; IBM ART's own `ActivationDefence` (Chen et al.'s canonical reference
implementation) itself defaults to reduce="PCA".

The two existing on-disk numbers for this question -- results/vision_control.json (PCA) and
results/vision_control.pre_ac_pca_fix.json.bak (FastICA) -- come from two DIFFERENT repo states
with an unrelated second fix applied in between (see the handover notes), so comparing them is a
cross-code-path comparison this project explicitly distrusts. This script redoes the comparison on
a single activation matrix per seed, computed once, then reduced both ways, so the only thing that
differs between the PCA arm and the ICA arm is the reduction itself.

Gate (same as scripts/01_vision_control.py): recall >= 0.90 on EVERY seed.

FastICA is a stochastic, non-convex fixed-point solver -- its `random_state` seeds only the solver
initialization/whitening, not the data. To measure solver instability directly (rather than
inferring it from two old files produced under different conditions), this script also runs ICA
five additional times per seed on the SAME feats_t matrix, varying only that solver seed
(seed + 1000*j for j in 1..5), and records the per-seed spread of recall across those 6 ICA runs
(the primary run plus 5 replicates).

Runs Activation Clustering only -- no Neural Cleanse, no Spectral Signatures -- both to keep this
script fast (skipping NC's gradient inversion is most of the runtime saved) and because this task
is scoped to the AC arm only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src import vision_control as vc
from src.detectors import poison_recall
from src.detectors.activation_clustering import detect as ac_detect

TARGET = 0
POISON_FRAC = 0.05
FEAT_DIM = 512
N_COMPONENTS = 10
EPOCHS = 3
N_ICA_REPLICATES = 5  # additional solver-restart runs beyond the primary ICA run, same feats_t

RESULTS_PATH = config.RESULTS / "ac_ica_vision_control.json"
CKPT_PATH = config.RESULTS / "ac_ica_vision_control.checkpoint.json"


def config_fingerprint() -> str:
    cfg = dict(target=TARGET, poison_frac=POISON_FRAC, feat_dim=FEAT_DIM,
               n_components=N_COMPONENTS, epochs=EPOCHS, n_ica_replicates=N_ICA_REPLICATES,
               seeds=list(config.SEEDS))
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def load_checkpoint(fp: str) -> dict:
    if not CKPT_PATH.exists():
        return {"fingerprint": fp, "rows": {}}
    try:
        blob = json.loads(CKPT_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {"fingerprint": fp, "rows": {}}
    if blob.get("fingerprint") != fp:
        print("checkpoint config fingerprint mismatch -- starting fresh (old checkpoint ignored)")
        return {"fingerprint": fp, "rows": {}}
    done = list(blob.get("rows", {}).keys())
    if done:
        print(f"resuming from checkpoint: seeds already done = {done}")
    return blob


def save_checkpoint(blob: dict) -> None:
    tmp = CKPT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob, indent=2))
    os.replace(tmp, CKPT_PATH)  # atomic: a crash mid-write never corrupts the checkpoint


def run_seed(seed: int, device: str) -> dict:
    """One victim, one activation matrix, both reductions applied to it."""
    vc.set_seed(seed)
    x_tr, y_tr, x_te, y_te = vc.load_mnist()
    x_p, y_p, poison_idx = vc.poison_trainset(x_tr, y_tr, TARGET, POISON_FRAC, seed)
    model = vc.train_victim(x_p, y_p, seed, epochs=EPOCHS, device=device, feat_dim=FEAT_DIM)

    target_mask = (y_p == TARGET).numpy()
    is_poison = np.zeros(len(y_p), dtype=bool)
    is_poison[poison_idx] = True
    is_poison_t = is_poison[target_mask]
    feats_t = vc.penultimate_features(model, x_p[target_mask], device)

    # PCA arm (must reproduce results/vision_control.json's ac_recall for this seed).
    pca_mask, pca_sil = ac_detect(feats_t, n_components=N_COMPONENTS, seed=seed, reduction="pca")
    pca_recall = poison_recall(pca_mask, is_poison_t)

    # ICA arm, primary run.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        ica_mask, ica_sil = ac_detect(feats_t, n_components=N_COMPONENTS, seed=seed,
                                       reduction="ica")
    ica_recall = poison_recall(ica_mask, is_poison_t)
    ica_warn_count = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))

    # ICA solver-restart replicates: same feats_t, only the solver seed varies.
    replicate_recalls = []
    replicate_warn_counts = []
    for j in range(1, N_ICA_REPLICATES + 1):
        rep_seed = seed + 1000 * j
        with warnings.catch_warnings(record=True) as caught_j:
            warnings.simplefilter("always", ConvergenceWarning)
            rep_mask, _ = ac_detect(feats_t, n_components=N_COMPONENTS, seed=rep_seed,
                                     reduction="ica")
        replicate_recalls.append(poison_recall(rep_mask, is_poison_t))
        replicate_warn_counts.append(
            sum(1 for w in caught_j if issubclass(w.category, ConvergenceWarning))
        )

    all_ica_recalls = [ica_recall] + replicate_recalls
    return {
        "seed": seed,
        "pca_recall": pca_recall,
        "pca_silhouette": pca_sil,
        "ica_recall": ica_recall,
        "ica_silhouette": ica_sil,
        "ica_convergence_warnings": ica_warn_count,
        "ica_replicate_recalls": replicate_recalls,
        "ica_replicate_convergence_warnings": replicate_warn_counts,
        "ica_all_recalls": all_ica_recalls,
        "ica_replicate_spread_max_minus_min": float(max(all_ica_recalls) - min(all_ica_recalls)),
        "target_poison_frac": float(is_poison_t.mean()),
    }


def main() -> None:
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    fp = config_fingerprint()
    ckpt = load_checkpoint(fp)
    rows_by_seed = ckpt["rows"]

    for seed in config.SEEDS:
        key = str(seed)
        if key in rows_by_seed:
            print(f"seed {seed}: skipping (checkpointed)")
            continue
        print(f"seed {seed}: running...")
        row = run_seed(seed, device)
        rows_by_seed[key] = row
        save_checkpoint({"fingerprint": fp, "rows": rows_by_seed})
        print(f"seed {seed}: pca_recall={row['pca_recall']:.4f} "
              f"ica_recall={row['ica_recall']:.4f} "
              f"ica_replicate_spread={row['ica_replicate_spread_max_minus_min']:.4f}")

    rows = [rows_by_seed[str(s)] for s in config.SEEDS]

    def stat(key: str, fn) -> float:
        return float(fn([r[key] for r in rows]))

    pca_recalls = [r["pca_recall"] for r in rows]
    ica_recalls = [r["ica_recall"] for r in rows]
    summary = {
        "pca": {
            "recall_mean": float(np.mean(pca_recalls)),
            "recall_min": float(np.min(pca_recalls)),
            "recall_max": float(np.max(pca_recalls)),
            "recall_all_seeds_ge_090": all(r >= 0.90 for r in pca_recalls),
        },
        "ica": {
            "recall_mean": float(np.mean(ica_recalls)),
            "recall_min": float(np.min(ica_recalls)),
            "recall_max": float(np.max(ica_recalls)),
            "recall_all_seeds_ge_090": all(r >= 0.90 for r in ica_recalls),
            "replicate_spread_max_across_seeds": max(
                r["ica_replicate_spread_max_minus_min"] for r in rows
            ),
            "total_convergence_warnings": sum(
                r["ica_convergence_warnings"] + sum(r["ica_replicate_convergence_warnings"])
                for r in rows
            ),
        },
    }

    out = {
        "config": {
            "target": TARGET, "poison_frac": POISON_FRAC, "seeds": list(config.SEEDS),
            "feat_dim": FEAT_DIM, "n_components": N_COMPONENTS, "epochs": EPOCHS,
            "n_ica_replicates": N_ICA_REPLICATES,
        },
        "per_seed": rows,
        "summary": summary,
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))

    print("\nper-seed table")
    print(f"{'seed':>6} {'pca_recall':>11} {'ica_recall':>11} {'pca_sil':>9} {'ica_sil':>9} "
          f"{'ica_replicates':>40} {'ica_spread':>11} {'warn(main+reps)':>16}")
    for r in rows:
        reps = ",".join(f"{x:.3f}" for x in r["ica_replicate_recalls"])
        warns = r["ica_convergence_warnings"] + sum(r["ica_replicate_convergence_warnings"])
        print(f"{r['seed']:>6} {r['pca_recall']:>11.4f} {r['ica_recall']:>11.4f} "
              f"{r['pca_silhouette']:>9.4f} {r['ica_silhouette']:>9.4f} {reps:>40} "
              f"{r['ica_replicate_spread_max_minus_min']:>11.4f} {warns:>16}")

    print(f"\nPCA gate (>=0.90 recall, every seed): "
          f"{'PASS' if summary['pca']['recall_all_seeds_ge_090'] else 'FAIL'}")
    print(f"ICA gate (>=0.90 recall, every seed): "
          f"{'PASS' if summary['ica']['recall_all_seeds_ge_090'] else 'FAIL'}")
    print(f"ICA replicate spread, max across seeds: "
          f"{summary['ica']['replicate_spread_max_across_seeds']:.4f}")
    print(f"results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
