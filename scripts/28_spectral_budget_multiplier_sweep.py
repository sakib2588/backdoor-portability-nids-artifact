"""Review-round check (D-22, 2026-08-13): is the fixed-budget removal rule's constant (1.5) merely
miscalibrated, or does no fixed multiplier reach the recall the adaptive MAD rule reaches?

Retrains the anchor cell (rate=0.005, cost=16, MLP) exactly as scripts/05_run_detectors.py's
run_main_grid does, for the same 5 committed seeds, then re-flags the SAME computed Spectral scores
at multiple budget multipliers via src.detectors.spectral.flag_by_scores's existing `multiplier`
argument (no retraining needed to sweep multipliers once the scores exist -- only the flagging
threshold changes). Self-checks against results/detectors.json's committed multiplier=1.5 recall
before trusting anything else this script computes.

The cell is selectable so the recommended constant can be checked on cells that did NOT define the
budget-starvation window. The committed anchor (0.005, 16) keeps its original output paths, so a
rerun of the default reproduces results/spectral_budget_multiplier_sweep.json byte-for-byte; any
other cell writes to a cell-suffixed file and cannot overwrite the cited numbers.

Run: python scripts/28_spectral_budget_multiplier_sweep.py [--rate R --cost C]
"""
from __future__ import annotations

import argparse
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
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.models import mlp_penultimate_features, train_mlp
from src.trigger import build_trigger, raw_trigger_stats, shap_rank_features
from src.poison import poison_trainset_cleanlabel

TARGET = config.ATTACK_TARGET
SPECTRAL_K = 5  # must match scripts/05_run_detectors.py's SPECTRAL_K
ANCHOR_RATE = 0.005
ANCHOR_COST = 16
MULTIPLIERS = [1.5, 2, 3, 5, 7, 10, 14, 20]
# The budget-free rule is reported here too, so the question the multiplier sweep raises --
# whether the CONSTANT is regime-specific while the MAD rule is not -- can be answered at the
# same cells without a second retraining run.
Z_THRESHOLDS = (2.0, 2.5, 3.0)
CKPT = config.RESULTS / "spectral_budget_multiplier_sweep.checkpoint.json"
OUT = config.RESULTS / "spectral_budget_multiplier_sweep.json"


def committed_recall_at_1p5(seed: int) -> float | None:
    path = config.RESULTS / "detectors.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    for row in blob["main_grid"]:
        if row["seed"] == seed and row["rate"] == ANCHOR_RATE and row["cost"] == ANCHOR_COST:
            return row["spectral"]["recall"]
    return None


def load_checkpoint():
    if not CKPT.exists():
        return {}
    try:
        return json.loads(CKPT.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_checkpoint(rows: dict):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows))
    tmp.replace(CKPT)


def run_seed(seed: int, S: dict, cfg: dict, device: str) -> dict:
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]

    x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
    clean_mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
    ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET, kind="mlp",
                                 device=device)

    stats = raw_trigger_stats(x_tr_raw, constraints)
    realiz, _ = build_trigger(ranking, ANCHOR_COST, x_tr_raw, constraints, features, bounds,
                              stats=stats)
    x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
        x_tr_raw, y_tr, realiz, ANCHOR_RATE, S["scaler"], target=TARGET, seed=seed,
        constraints=constraints, feature_names=features, bounds=bounds)
    mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

    benign_pos = np.where(y_p == TARGET)[0]
    is_poison = np.isin(benign_pos, poison_idx)
    n_poison = int(is_poison.sum())
    n_total = int(len(is_poison))
    expected_frac = float(is_poison.mean())

    feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
    scores = spectral_scores(feats, n_components=SPECTRAL_K)

    committed = committed_recall_at_1p5(seed)
    fresh_1p5_flag = flag_by_scores(scores, expected_frac=expected_frac, multiplier=1.5)
    fresh_1p5_recall = float(poison_recall(fresh_1p5_flag, is_poison))
    determinism_ok = bool(committed is not None
                          and np.isclose(fresh_1p5_recall, committed, atol=1e-6))

    per_z = []
    n_clean_total = n_total - n_poison
    for z in Z_THRESHOLDS:
        zflag = flag_by_mad_threshold(scores, z_thresh=z)
        z_clean_flagged = int(zflag.sum() - (zflag & is_poison).sum())
        per_z.append(dict(z=z, recall=poison_recall(zflag, is_poison),
                          clean_flagged=z_clean_flagged,
                          clean_fpr=z_clean_flagged / n_clean_total if n_clean_total else None))

    per_multiplier = []
    for m in MULTIPLIERS:
        flag = flag_by_scores(scores, expected_frac=expected_frac, multiplier=m)
        recall = poison_recall(flag, is_poison)
        k_b = int(flag.sum())
        n_clean_flagged = int(k_b - (flag & is_poison).sum())
        n_clean = n_total - n_poison
        clean_fpr = n_clean_flagged / n_clean if n_clean else None
        per_multiplier.append(dict(multiplier=m, k_b=k_b, recall=recall,
                                   clean_flagged=n_clean_flagged, clean_fpr=clean_fpr))

    return dict(seed=seed, n_total=n_total, n_poison=n_poison, expected_frac=expected_frac,
               committed_recall_at_1p5=committed, fresh_recall_at_1p5=fresh_1p5_recall,
               determinism_ok=determinism_ok, per_multiplier=per_multiplier, per_z=per_z)


def _select_cell(rate: float, cost: int) -> None:
    """Point the module's cell and output paths at (rate, cost). The default anchor keeps the
    original filenames so the committed, cited result is reproduced rather than shadowed."""
    global ANCHOR_RATE, ANCHOR_COST, CKPT, OUT
    ANCHOR_RATE, ANCHOR_COST = rate, cost
    if (rate, cost) == (0.005, 16):
        return
    tag = f"_r{rate}_c{cost}"
    CKPT = config.RESULTS / f"spectral_budget_multiplier_sweep{tag}.checkpoint.json"
    OUT = config.RESULTS / f"spectral_budget_multiplier_sweep{tag}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=ANCHOR_RATE)
    ap.add_argument("--cost", type=int, default=ANCHOR_COST)
    args = ap.parse_args()
    _select_cell(args.rate, args.cost)

    t0 = time.time()
    cfg = get_config(smoke=False)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  cell=(rate={ANCHOR_RATE}, cost={ANCHOR_COST})  "
          f"multipliers={MULTIPLIERS}  seeds={cfg['seeds']}")

    S = load_setup(cfg)
    rows = load_checkpoint()

    for seed in cfg["seeds"]:
        if str(seed) in rows:
            print(f"--- seed {seed}: already complete, skipping ---")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        row = run_seed(seed, S, cfg, device)
        if not row["determinism_ok"]:
            print(f"WARNING: seed={seed} multiplier=1.5 recall {row['fresh_recall_at_1p5']:.6f} "
                  f"does not match committed {row['committed_recall_at_1p5']} -- do not trust this "
                  "seed's sweep until investigated.")
        rows[str(seed)] = row
        save_checkpoint(rows)

    all_ok = all(r["determinism_ok"] for r in rows.values())
    by_mult = {}
    for m in MULTIPLIERS:
        recalls = [next(pm["recall"] for pm in rows[s]["per_multiplier"] if pm["multiplier"] == m)
                  for s in rows]
        fprs = [next(pm["clean_fpr"] for pm in rows[s]["per_multiplier"] if pm["multiplier"] == m)
               for s in rows]
        by_mult[str(m)] = dict(recall_mean=float(np.mean(recalls)), recall_values=recalls,
                               clean_fpr_mean=float(np.mean(fprs)), clean_fpr_values=fprs)

    by_z = {}
    for z in Z_THRESHOLDS:
        rec = [next(q["recall"] for q in rows[s_]["per_z"] if q["z"] == z) for s_ in rows]
        fpr = [next(q["clean_fpr"] for q in rows[s_]["per_z"] if q["z"] == z) for s_ in rows]
        by_z[str(z)] = dict(recall_mean=float(np.mean(rec)), clean_fpr_mean=float(np.mean(fpr)))

    out = dict(anchor=dict(rate=ANCHOR_RATE, cost=ANCHOR_COST), multipliers=MULTIPLIERS,
              z_thresholds=list(Z_THRESHOLDS), by_z=by_z,
              determinism_all_ok=all_ok, per_seed=list(rows.values()), by_multiplier=by_mult)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(by_mult, indent=2))
    print(f"determinism_all_ok={all_ok}")
    print(f"elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
