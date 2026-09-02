"""Does a stronger score repair the budget-starvation window, or does the inherited removal budget
starve SPECTRE too?

Pre-registered in notes/20260826-decision-spectre-preregistration.md BEFORE this ran. The paper's
diagnosis is that Tran et al.'s fixed removal budget, not Spectral Signatures' discriminative signal,
is what fails to port. SPECTRE (Hayase et al., ICML 2021) strengthens exactly the component that
diagnosis leaves untouched, so scoring it under the SAME two decision rules separates the two
explanations. If SPECTRE's fixed-budget recall is high, the score was the binding constraint and the
paper's central claim needs revision -- that outcome is pre-registered as falsifying.

SPECTRE cleared the MNIST vision-control gate first (recall 1.0 on all five seeds, AUC 1.0, against
Spectral's 0.9814): scripts/52_spectre_vision_control.py, results/spectre_vision_control.json. A null
below is therefore interpretable.

Both scorers run on the IDENTICAL trained model and poisoned population per row, so the arms differ
only in the scorer. Device is CUDA, matching scripts/05_run_detectors.py, so the Spectral arm
reproduces the committed grid rather than the CPU-trained variant that made
scripts/12_spectral_k1_ablation.py only within-arm comparable.

Checkpointed per cell-seed with an atomic temp-file-then-rename write and a config fingerprint, so a
kill mid-run costs at most one cell.

Run:  .venv/bin/python scripts/53_spectre_window.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup
from src import config
from src.data import apply_standardiser
from src.detectors import poison_recall
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.detectors.spectre import spectre_scores
from src.models import mlp_penultimate_features, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET
WINDOW_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]
SPECTRAL_K = 5
Z_THRESHOLDS = (2.0, 2.5, 3.0)
BUDGET_MULTIPLIER = 1.5

OUT = config.RESULTS / "spectre_window.json"
CKPT = config.RESULTS / "spectre_window.checkpoint.json"


def cell_key(seed, rate, cost) -> str:
    return f"{seed}|{rate}|{cost}"


def config_key(cfg) -> dict:
    return dict(cells=[list(c) for c in WINDOW_CELLS], seeds=list(cfg["seeds"]),
                mlp_epochs=cfg["mlp_epochs"], spectral_k=SPECTRAL_K,
                z=list(Z_THRESHOLDS), multiplier=BUDGET_MULTIPLIER, smoke=cfg["smoke"])


def load_ckpt(key):
    if not CKPT.exists():
        return {}
    try:
        blob = json.loads(CKPT.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh")
        return {}
    done = blob.get("rows", {})
    print(f"resuming from checkpoint: {len(done)} cell-seeds already done")
    return done


def save_ckpt(key, rows):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(CKPT)          # atomic: a crash mid-write never corrupts the checkpoint


def score_block(scores, is_poison, expected_frac):
    """Both decision rules against one score vector, plus the ranking metric and the MAD cost."""
    n_clean = int((~is_poison).sum())
    out = {}
    budget_flag = flag_by_scores(scores, expected_frac=expected_frac, multiplier=BUDGET_MULTIPLIER)
    out["fixed_recall"] = poison_recall(budget_flag, is_poison)
    out["auc"] = (float(roc_auc_score(is_poison, scores))
                  if 0 < is_poison.sum() < len(is_poison) else None)
    out["mad_recall"] = {}
    out["mad_fpr"] = {}
    for z in Z_THRESHOLDS:
        flag = flag_by_mad_threshold(scores, z_thresh=z)
        out["mad_recall"][str(z)] = poison_recall(flag, is_poison)
        out["mad_fpr"][str(z)] = (float((flag & ~is_poison).sum() / n_clean) if n_clean else None)
    return out


def main() -> int:
    t0 = time.time()
    cfg = get_config(smoke=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = config_key(cfg)
    done = load_ckpt(key)
    print(f"device={device}  cells={WINDOW_CELLS}  seeds={cfg['seeds']}")

    S = load_setup(cfg)
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    stats = raw_trigger_stats(x_tr_raw, constraints)

    for seed in cfg["seeds"]:
        pending = [c for c in WINDOW_CELLS if cell_key(seed, *c) not in done]
        if not pending:
            print(f"--- seed {seed}: all cells cached, skipping ---")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
        clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET,
                                     kind="mlp", device=device)

        for rate, cost in pending:
            realiz, _ = build_trigger(ranking, cost, x_tr_raw, constraints, features, bounds,
                                      stats=stats)
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
            expected_frac = float(is_poison.mean())   # oracle, same grant as the committed grid

            spec = spectral_scores(feats, n_components=SPECTRAL_K)
            spct = spectre_scores(feats, expected_frac=expected_frac)

            row = dict(seed=seed, rate=rate, cost=cost,
                       n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
                       expected_frac=expected_frac,
                       spectral=score_block(spec, is_poison, expected_frac),
                       spectre=score_block(spct, is_poison, expected_frac))
            done[cell_key(seed, rate, cost)] = row
            save_ckpt(key, done)
            print(f"  rate={rate} cost={cost}: "
                  f"spectral fixed={row['spectral']['fixed_recall']:.4f} auc={row['spectral']['auc']:.4f} | "
                  f"spectre fixed={row['spectre']['fixed_recall']:.4f} auc={row['spectre']['auc']:.4f} "
                  f"mad3={row['spectre']['mad_recall']['3.0']:.4f} "
                  f"fpr3={row['spectre']['mad_fpr']['3.0']:.4f}")

    rows = [done[cell_key(s, r, c)] for s in cfg["seeds"] for r, c in WINDOW_CELLS
            if cell_key(s, r, c) in done]

    def agg(det, field):
        return float(np.mean([r[det][field] for r in rows]))

    def agg_mad(det, field, z):
        return float(np.mean([r[det][field][str(z)] for r in rows]))

    summary = {
        "n_rows": len(rows),
        "spectral_fixed_recall_mean": agg("spectral", "fixed_recall"),
        "spectre_fixed_recall_mean": agg("spectre", "fixed_recall"),
        "spectral_auc_mean": agg("spectral", "auc"),
        "spectre_auc_mean": agg("spectre", "auc"),
        "spectre_mad_recall_mean": {str(z): agg_mad("spectre", "mad_recall", z) for z in Z_THRESHOLDS},
        "spectre_mad_fpr_mean": {str(z): agg_mad("spectre", "mad_fpr", z) for z in Z_THRESHOLDS},
        "spectral_mad_recall_mean": {str(z): agg_mad("spectral", "mad_recall", z) for z in Z_THRESHOLDS},
        "spectral_mad_fpr_mean": {str(z): agg_mad("spectral", "mad_fpr", z) for z in Z_THRESHOLDS},
    }
    verdict = ("B_score_was_binding" if summary["spectre_fixed_recall_mean"] >= 0.5
               else ("A_inherits_starvation" if summary["spectre_auc_mean"] >= 0.97
                     else "C_mixed"))
    out = dict(cells=WINDOW_CELLS, seeds=list(cfg["seeds"]), multiplier=BUDGET_MULTIPLIER,
               z_thresholds=list(Z_THRESHOLDS), spectral_k=SPECTRAL_K, device=device,
               preregistration="notes/20260826-decision-spectre-preregistration.md",
               vision_gate="results/spectre_vision_control.json",
               rows=rows, summary=summary, verdict=verdict)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"PRE-REGISTERED VERDICT: {verdict}")
    print(f"wrote {OUT}  elapsed={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
