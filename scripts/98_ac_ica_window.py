#!/usr/bin/env python3
"""Activation Clustering with FastICA reduction, on the four CTU-13 window cells.

READ THIS BEFORE QUOTING ANY NUMBER THIS SCRIPT PRODUCES.

The output of this script sits behind a FAILED positive control and is therefore
uninterpretable by this project's own stated standard. From CLAUDE.md:

    Validate detectors on the vision control FIRST. [...] A null without a passing control
    is uninterpretable.

FastICA does not pass that control. `notes/20260903-experiment-ac-ica-vision-gate.md`
records the measurement: on the matched MNIST vision control, PCA clears recall >= 0.90 on
5 of 5 seeds with a minimum of 0.9778, while ICA fails on 2 of 5 at 0.4996 and 0.5717.
Re-running ICA on the identical activation matrix, changing only the solver's random_state,
moved recall by up to 0.4922. The reduction is unstable at the solver level, and the
instability is local optima rather than non-convergence: zero of the thirty fits warned.

That is why the tabular ICA cells were deliberately never computed. This script exists
because a reviewer may ask for the numbers anyway, and "they do not exist" is a weaker
answer than "here they are, and here is why we do not build on them". Every number it
writes carries `interpretable: false` and the reason, in the output JSON, so the caveat
cannot be separated from the values by a later copy-paste.

WHAT WOULD MAKE THESE NUMBERS USABLE. Nothing available today. ICA would have to clear the
vision control first, which would mean either a stabilized solver configuration that itself
needs pre-registering, or a different control. Neither is a small change and neither should
be attempted by tuning until the gate passes.

The cell loop is copied from scripts/96 rather than re-derived. Reduction is the only thing
that differs between the two arms: both are scored on the same activations, from the same
victim, at the same cell and seed, so any difference is the reduction and not drift.
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
from src.detectors import poison_recall
from src.detectors.activation_clustering import cluster_and_reduce
from src.models import mlp_penultimate_features, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import build_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
SILHOUETTE_SAMPLE = 10000     # HANDOVER.md: cap it or AC falls into an O(n^2) score
WINDOW_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]

OUT = config.RESULTS / "ac_ica_window.json"
CKPT = config.RESULTS / "ac_ica_window.checkpoint.json"

UNINTERPRETABLE = (
    "FastICA fails the matched MNIST vision control on 2 of 5 seeds (0.4996, 0.5717) where "
    "PCA clears it on 5 of 5 (min 0.9778), and changing only the solver seed moves recall by "
    "up to 0.4922. A null behind a failed control is uninterpretable, so these values must "
    "not be quoted as a result about ICA on tabular NIDS. See "
    "notes/20260903-experiment-ac-ica-vision-gate.md."
)


def _smoke_paths():
    global OUT, CKPT
    OUT = config.RESULTS / "ac_ica_window_smoke.json"
    CKPT = config.RESULTS / "ac_ica_window_smoke.checkpoint.json"


def config_key(cfg) -> dict:
    return dict(seeds=list(cfg["seeds"]), mlp_epochs=cfg["mlp_epochs"],
                cells=[list(c) for c in WINDOW_CELLS], smoke=bool(cfg.get("smoke")),
                reductions=["pca", "ica"], silhouette_sample=SILHOUETTE_SAMPLE)


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


def main() -> int:
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    cfg = get_config(smoke)
    if smoke:
        cfg["seeds"] = [42]
        _smoke_paths()
    cfg["smoke"] = smoke
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = config_key(cfg)
    rows = load_ckpt(key)

    print("=" * 78)
    print("UNINTERPRETABLE BY CONSTRUCTION:")
    print(UNINTERPRETABLE)
    print("=" * 78)
    print(f"device={device} seeds={cfg['seeds']} cells={WINDOW_CELLS}")

    S = load_setup(cfg)
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    per_seed = {}

    def rank_for(seed):
        if seed not in per_seed:
            xs = apply_standardiser(S["scaler"], x_tr_raw)
            clean_mlp = train_mlp(xs, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
            per_seed[seed] = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"],
                                                target=TARGET, kind="mlp", device=device)
        return per_seed[seed]

    for rate, cost in WINDOW_CELLS:
        for seed in cfg["seeds"]:
            k = f"{seed}|{rate}|{cost}"
            if k in rows:
                print(f"--- {k}: cached ---")
                continue
            print(f"--- {k} ({time.time() - t0:.0f}s) ---")
            realiz, _ = build_trigger(rank_for(seed), cost, x_tr_raw, S["constraints"],
                                      S["features"], S["bounds"])
            x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
                x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
                constraints=S["constraints"], feature_names=S["features"], bounds=S["bounds"])
            mlp_bd = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)
            benign_pos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(benign_pos, poison_idx)
            feats = mlp_penultimate_features(mlp_bd, x_p_std[benign_pos], device)

            row = dict(seed=seed, rate=rate, cost=cost,
                       n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
                       interpretable=False, reason=UNINTERPRETABLE)
            # Both arms on the SAME activations. Reduction is the only difference.
            for red in ("pca", "ica"):
                labels, _reduced, sil = cluster_and_reduce(
                    feats, seed=seed, silhouette_sample=SILHOUETTE_SAMPLE, reduction=red)
                # AC's published rule flags the smaller cluster as poison.
                sizes = np.bincount(labels, minlength=2)
                small = int(np.argmin(sizes))
                row[red] = dict(recall=poison_recall(labels == small, is_poison),
                                silhouette=float(sil) if sil is not None else None,
                                smaller_cluster_frac=float(sizes[small] / sizes.sum()))
            rows[k] = row
            save_ckpt(key, rows)
            print(f"  pca recall={row['pca']['recall']:.4f}  ica recall={row['ica']['recall']:.4f}")

    cells = [rows[f"{s}|{rt}|{c}"] for rt, c in WINDOW_CELLS for s in cfg["seeds"]
             if f"{s}|{rt}|{c}" in rows]

    def agg(red):
        v = [r[red]["recall"] for r in cells]
        return dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                    n=len(v)) if v else None

    blob = dict(
        _WARNING=UNINTERPRETABLE, interpretable=False,
        vision_gate="notes/20260903-experiment-ac-ica-vision-gate.md: ICA fails 2 of 5 seeds",
        device=device, cells=cells,
        summary=dict(pca_recall=agg("pca"), ica_recall=agg("ica")),
        elapsed_s=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(blob, indent=2) + "\n")
    print(f"\npca {blob['summary']['pca_recall']}\nica {blob['summary']['ica_recall']}")
    print(f"wrote {OUT}  ({blob['elapsed_s']}s)  -- interpretable=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
