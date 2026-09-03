#!/usr/bin/env python
"""Is SPECTRE a separate detector from Spectral Signatures, or Spectral in disguise?

Pre-registered in notes/20260903-decision-spectre-independence-preregistration.md BEFORE this ran.
The manuscript counts five vision-built detectors. Two of the four that emit a ranked score return
recall 1.0000 under the MAD rule on every window cell, and SPECTRE whitens the same penultimate
activations Spectral Signatures projects. This project's own bar for that question,
config.AC_SPECTRAL_INDEPENDENCE_MAX_RHO = 0.90 on the per-sample Spearman correlation, was applied
to the Activation Clustering score (0.3987, independent) and never to SPECTRE.

Recipe is scripts/53_spectre_window.py's, cell for cell: retrain the poisoned MLP, take the
target-class activations, score them with both detectors, and record the correlation of the two
score vectors, the overlap of their flagged sets, and both AUCs so each cell ties to the committed
numbers. Verdict: classify_centroid_independence on the median per-cell rho over the twenty window
cells. The matched-rate control (0.1, 16) is scored for contrast and is not in the median.

Checkpointed per cell, atomic write, config fingerprint; smoke and full runs write separate files.

Run:  .venv/bin/python scripts/84_spectre_spectral_independence.py --smoke
      .venv/bin/python scripts/84_spectre_spectral_independence.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup
from src import config
from src.data import apply_standardiser
from src.detectors import poison_recall
from src.detectors.activation_clustering import classify_centroid_independence
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.detectors.spectre import spectre_scores
from src.models import mlp_penultimate_features, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET
WINDOW_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]
CONTROL_CELL = (0.1, 16)
SPECTRAL_K = 5
Z_PRIMARY = config.MAD_Z_PRIMARY
BUDGET_MULTIPLIER = 1.5
MAX_RHO = config.AC_SPECTRAL_INDEPENDENCE_MAX_RHO


def paths(smoke: bool):
    tag = "_smoke" if smoke else ""
    return (config.RESULTS / f"spectre_spectral_independence{tag}.json",
            config.RESULTS / f"spectre_spectral_independence{tag}.checkpoint.json")


def cell_key(seed, rate, cost) -> str:
    return f"{seed}|{rate}|{cost}"


def config_key(cfg) -> dict:
    return dict(smoke=cfg["smoke"], mlp_epochs=cfg["mlp_epochs"], subsample=cfg["subsample"],
                cells=WINDOW_CELLS + [CONTROL_CELL], spectral_k=SPECTRAL_K, z=Z_PRIMARY,
                multiplier=BUDGET_MULTIPLIER, max_rho=MAX_RHO)


def load_ckpt(path, key) -> dict:
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh")
        return {}
    print(f"resuming: {len(blob.get('cells', {}))} cells done")
    return blob.get("cells", {})


def save_ckpt(path, key, cells) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, cells=cells)))
    tmp.replace(path)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 1.0


def rho(x, y) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def cell_row(spec, spct, is_poison, expected_frac) -> dict:
    mad_a = flag_by_mad_threshold(spec, z_thresh=Z_PRIMARY)
    mad_b = flag_by_mad_threshold(spct, z_thresh=Z_PRIMARY)
    bud_a = flag_by_scores(spec, expected_frac=expected_frac, multiplier=BUDGET_MULTIPLIER)
    bud_b = flag_by_scores(spct, expected_frac=expected_frac, multiplier=BUDGET_MULTIPLIER)
    auc = lambda s: (float(roc_auc_score(is_poison, s)) if 0 < is_poison.sum() < len(is_poison) else None)
    return dict(
        n_rows=int(len(spec)), n_poison=int(is_poison.sum()),
        rho_all=rho(spec, spct),
        rho_poison=rho(spec[is_poison], spct[is_poison]),
        rho_clean=rho(spec[~is_poison], spct[~is_poison]),
        jaccard_mad_flags=jaccard(mad_a, mad_b),
        jaccard_budget_flags=jaccard(bud_a, bud_b),
        spectral=dict(auc=auc(spec), mad_recall=poison_recall(mad_a, is_poison),
                      fixed_recall=poison_recall(bud_a, is_poison), n_mad_flagged=int(mad_a.sum())),
        spectre=dict(auc=auc(spct), mad_recall=poison_recall(mad_b, is_poison),
                     fixed_recall=poison_recall(bud_b, is_poison), n_mad_flagged=int(mad_b.sum())),
    )


def main() -> int:
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    cfg = get_config(smoke)
    if smoke:
        cfg["seeds"] = [42]
    cells_to_run = WINDOW_CELLS + [CONTROL_CELL]
    if smoke:
        cells_to_run = [WINDOW_CELLS[1]]     # one window cell is enough to prove the wiring
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path, ckpt_path = paths(smoke)
    key = config_key(cfg)
    done = load_ckpt(ckpt_path, key)
    print(f"device={device} smoke={smoke} seeds={cfg['seeds']} cells={cells_to_run} bar={MAX_RHO}")

    S = load_setup(cfg)
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    stats = raw_trigger_stats(x_tr_raw, constraints)

    for seed in cfg["seeds"]:
        pending = [c for c in cells_to_run if cell_key(seed, *c) not in done]
        if not pending:
            print(f"--- seed {seed}: cached ---")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s) ---")
        x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
        clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET,
                                     kind="mlp", device=device)
        for rate, cost in pending:
            t_c = time.time()
            realiz, _ = build_trigger(ranking, cost, x_tr_raw, constraints, features, bounds, stats=stats)
            x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
                x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
                constraints=constraints, feature_names=features, bounds=bounds)
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)
            benign_pos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(benign_pos, poison_idx)
            if is_poison.sum() == 0:
                print(f"  rate={rate} cost={cost}: no poison in target class, skipped")
                continue
            feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
            expected_frac = float(is_poison.mean())
            spec = np.asarray(spectral_scores(feats, n_components=SPECTRAL_K), dtype=float)
            spct = np.asarray(spectre_scores(feats, expected_frac=expected_frac), dtype=float)
            row = dict(seed=seed, rate=rate, cost=cost, is_window=(rate, cost) in WINDOW_CELLS,
                       **cell_row(spec, spct, is_poison, expected_frac),
                       elapsed_s=round(time.time() - t_c, 1))
            done[cell_key(seed, rate, cost)] = row
            save_ckpt(ckpt_path, key, done)
            print(f"  rate={rate} cost={cost}: rho_all={row['rho_all']:+.4f} rho_poison={row['rho_poison']:+.4f} "
                  f"rho_clean={row['rho_clean']:+.4f} jaccard_mad={row['jaccard_mad_flags']:.3f} "
                  f"auc {row['spectral']['auc']:.4f}/{row['spectre']['auc']:.4f}  {row['elapsed_s']}s")
            del mlp, feats, x_p_std
            if device == "cuda":
                torch.cuda.empty_cache()

    window = [r for r in done.values() if r["is_window"] and np.isfinite(r["rho_all"])]
    med = float(np.median([r["rho_all"] for r in window])) if window else float("nan")
    verdict = classify_centroid_independence(med, max_rho=MAX_RHO)
    summary = dict(
        n_window_cells=len(window), median_rho_all=med,
        min_rho_all=float(min(r["rho_all"] for r in window)) if window else None,
        max_rho_all=float(max(r["rho_all"] for r in window)) if window else None,
        median_rho_poison=float(np.median([r["rho_poison"] for r in window if np.isfinite(r["rho_poison"])])) if window else None,
        median_rho_clean=float(np.median([r["rho_clean"] for r in window if np.isfinite(r["rho_clean"])])) if window else None,
        median_jaccard_mad=float(np.median([r["jaccard_mad_flags"] for r in window])) if window else None,
        median_jaccard_budget=float(np.median([r["jaccard_budget_flags"] for r in window])) if window else None,
        bar=MAX_RHO, verdict=verdict,
        preregistration="notes/20260903-decision-spectre-independence-preregistration.md",
    )
    out_path.write_text(json.dumps(dict(config_key=key, cells=done, summary=summary,
                                        elapsed_s=round(time.time() - t0, 1)), indent=2))
    print(f"\nwindow cells {summary['n_window_cells']}: median rho_all {med:+.4f} "
          f"(range {summary['min_rho_all']} to {summary['max_rho_all']}), bar {MAX_RHO} -> "
          f"VERDICT {verdict}")
    print(f"wrote {out_path.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
