"""H3 (imbalance confound) at the FOUR CTU-13 window cells, reporting the ranking AUC that the
original single-cell run never saved.

Why this exists. scripts/05_run_detectors.py's run_h3 measures the down-sampling confound at ONE
operating point, rate 0.05 / cost 8 (`.config.h3_operating_point` in results/detectors.json). That
cell is off-window: Spectral's fixed-budget recall there is already 0.88, so nothing has collapsed
and the confound is tested where there is no collapse to explain. The budget-starvation window the
paper's claim rests on lives at rates 0.005-0.01.

Worse, run_h3 saves recall only. `scores` is in scope and `roc_auc_score` is imported, but the
return dict carries no AUC on either arm. Without a balanced-condition AUC the paper cannot
distinguish "balancing starved the fixed budget through the class size n" from "balancing destroyed
the underlying score" -- and that is precisely the score-versus-rule distinction the whole paper
exists to draw. An IEEE Access review (2026-09-02) named this as a Major defect and it is correct.

What this script does differently:
  * runs the four window cells {0.005, 0.01} x {8, 16}, not the single off-window cell;
  * saves `spectral_auc` on BOTH arms, so the starved-budget and destroyed-score hypotheses separate;
  * takes the imbalanced arm from the committed results/detectors.json `.main_grid[]` rather than
    retraining it, so the comparison stays inside one code path and cannot drift from the published
    grid (this project has been bitten before by two scripts computing the same quantity two ways).

The balanced clean MLP, standardiser and SHAP ranking depend on the seed alone, so they are built
once per seed and reused across the four cells; the trigger depends on (seed, cost) and is cached.

Checkpointed per cell-seed, atomic temp-then-rename, config fingerprint -- a kill mid-run costs at
most one cell.

Run:  .venv/bin/python scripts/69_h3_window_auc.py
      .venv/bin/python scripts/69_h3_window_auc.py --smoke
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup
from src import config
from src.data import apply_standardiser, downsample_benign, fit_standardiser
from src.detectors import poison_recall
from src.detectors.activation_clustering import detect as ac_detect
from src.detectors.spectral import flag_by_scores, spectral_scores
from src.models import mlp_penultimate_features, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET
WINDOW_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]
SPECTRAL_K = 5            # matches scripts/04 and 05 -- carried-forward M1 scope item
SILHOUETTE_SAMPLE = 10000  # matches scripts/05; silhouette is an O(n^2) diagnostic
H3_TARGET_RATIO = 0.5

OUT = config.RESULTS / "h3_window_auc.json"
CKPT = config.RESULTS / "h3_window_auc.checkpoint.json"


def _smoke_paths():
    """A smoke run must never land on the real artefact path: a killed full run would otherwise
    leave a 1-cell smoke file sitting where the committed result belongs."""
    global OUT, CKPT
    OUT = config.RESULTS / "h3_window_auc_smoke.json"
    CKPT = config.RESULTS / "h3_window_auc_smoke.checkpoint.json"


def cell_key(seed, rate, cost) -> str:
    return f"{seed}|{rate}|{cost}"


def config_key(cfg) -> dict:
    return dict(cells=[list(c) for c in WINDOW_CELLS], seeds=list(cfg["seeds"]),
                mlp_epochs=cfg["mlp_epochs"], spectral_k=SPECTRAL_K,
                target_ratio=H3_TARGET_RATIO, smoke=cfg["smoke"])


def load_ckpt(key):
    if not CKPT.exists():
        return {}
    try:
        blob = json.loads(CKPT.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    # Compare against the JSON projection: cell lists hold tuples in memory and come back
    # from disk as lists, so a direct comparison never matches and the sweep silently restarts.
    if blob.get("config_key") != json.loads(json.dumps(key)):
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} row(s) already done")
    return blob.get("rows", {})


def save_ckpt(key, rows):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(CKPT)


def load_imbalanced_reference() -> dict:
    """Index the committed grid by (seed, rate, cost). The imbalanced arm is READ, never recomputed,
    so this script cannot drift from the published Table numbers."""
    path = config.RESULTS / "detectors.json"
    if not path.exists():
        raise SystemExit("results/detectors.json not found -- run scripts/05_run_detectors.py first")
    blob = json.loads(path.read_text())
    return {(r["seed"], r["rate"], r["cost"]): r for r in blob["main_grid"]}


def main() -> int:
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    cfg = get_config(smoke)
    global WINDOW_CELLS
    if smoke:
        cfg["seeds"] = [42]
        WINDOW_CELLS = WINDOW_CELLS[:1]
        _smoke_paths()
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = config_key(cfg)
    rows = load_ckpt(key)
    imbalanced_ref = load_imbalanced_reference()
    print(f"device={device}  cells={WINDOW_CELLS}  seeds={cfg['seeds']}")

    S = load_setup(cfg)
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]

    for seed in cfg["seeds"]:
        pending = [c for c in WINDOW_CELLS if cell_key(seed, *c) not in rows]
        if not pending:
            print(f"--- seed {seed}: all cells cached, skipping ---")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")

        # Balanced setup depends on the seed alone: build once, reuse across the four cells.
        # A clean MLP trained on the rebalanced set has different SHAP importances than the
        # imbalanced one, so the ranking and the trigger must both be rebuilt here.
        x_bal_df, y_bal_ser = downsample_benign(
            pd.DataFrame(x_tr_raw, columns=features), pd.Series(y_tr),
            target_ratio=H3_TARGET_RATIO, seed=seed)
        x_tr_raw_bal, y_tr_bal = x_bal_df.to_numpy(), y_bal_ser.to_numpy()
        scaler_bal = fit_standardiser(pd.DataFrame(x_tr_raw_bal, columns=features))
        x_tr_std_bal = apply_standardiser(scaler_bal, x_tr_raw_bal)
        clean_mlp_bal = train_mlp(x_tr_std_bal, y_tr_bal, seed, epochs=cfg["mlp_epochs"],
                                  device=device)
        ranking_bal = shap_rank_features(clean_mlp_bal, x_tr_raw_bal, scaler_bal, target=TARGET,
                                         kind="mlp", device=device)
        stats_bal = raw_trigger_stats(x_tr_raw_bal, constraints)
        print(f"  balanced train set: n={len(y_tr_bal)} botnet={int(y_tr_bal.sum())}")

        trig_cache = {}
        for rate, cost in pending:
            if cost not in trig_cache:
                trig_cache[cost], _ = build_trigger(ranking_bal, cost, x_tr_raw_bal, constraints,
                                                    features, bounds, stats=stats_bal)
            realiz = trig_cache[cost]

            x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
                x_tr_raw_bal, y_tr_bal, realiz, rate, scaler_bal, target=TARGET, seed=seed,
                constraints=constraints, feature_names=features, bounds=bounds)
            mlp_bal = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            benign_pos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(benign_pos, poison_idx)
            if is_poison.sum() == 0:
                print(f"  rate={rate} cost={cost}: no poison in target class, skipped")
                continue
            feats = mlp_penultimate_features(mlp_bal, x_p_std[benign_pos], device)

            scores = spectral_scores(feats, n_components=SPECTRAL_K)
            spec_recall_bal = poison_recall(
                flag_by_scores(scores, expected_frac=float(is_poison.mean())), is_poison)
            # THE point of this script: the ranking metric on the balanced arm. Without it a low
            # balanced recall is ambiguous between a starved budget and a destroyed score.
            spec_auc_bal = (float(roc_auc_score(is_poison, scores))
                            if 0 < is_poison.sum() < len(is_poison) else None)
            ac_mask, ac_sil = ac_detect(feats, seed=seed, silhouette_sample=SILHOUETTE_SAMPLE)
            ac_recall_bal = poison_recall(ac_mask, is_poison)

            ref = imbalanced_ref.get((seed, rate, cost))
            if ref is None:
                raise SystemExit(f"no committed grid row for seed={seed} rate={rate} cost={cost}")

            rows[cell_key(seed, rate, cost)] = dict(
                seed=seed, rate=rate, cost=cost,
                n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
                balanced=dict(spectral_recall=spec_recall_bal, spectral_auc=spec_auc_bal,
                              ac_recall=ac_recall_bal, ac_silhouette=ac_sil,
                              train_balance=dict(n=len(y_tr_bal), botnet=int(y_tr_bal.sum()))),
                imbalanced=dict(spectral_recall=ref["spectral"]["recall"],
                                spectral_auc=ref["spectral"]["auc"],
                                ac_recall=ref["ac"]["recall"]))
            save_ckpt(key, rows)
            r = rows[cell_key(seed, rate, cost)]
            print(f"  rate={rate} cost={cost}: balanced recall={spec_recall_bal:.4f} "
                  f"auc={spec_auc_bal:.4f} | imbalanced recall={r['imbalanced']['spectral_recall']:.4f} "
                  f"auc={r['imbalanced']['spectral_auc']:.4f}")

    ordered = [rows[cell_key(s, r, c)] for s in cfg["seeds"] for r, c in WINDOW_CELLS
               if cell_key(s, r, c) in rows]

    def agg(arm, field):
        vals = [r[arm][field] for r in ordered if r[arm][field] is not None]
        return dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                    n=len(vals)) if vals else None

    summary = dict(
        balanced=dict(spectral_recall=agg("balanced", "spectral_recall"),
                      spectral_auc=agg("balanced", "spectral_auc"),
                      ac_recall=agg("balanced", "ac_recall")),
        imbalanced=dict(spectral_recall=agg("imbalanced", "spectral_recall"),
                        spectral_auc=agg("imbalanced", "spectral_auc"),
                        ac_recall=agg("imbalanced", "ac_recall")))

    OUT.write_text(json.dumps(dict(config=key, summary=summary, rows=ordered), indent=2))
    print(f"\nwrote {OUT}  elapsed={time.time() - t0:.0f}s  n_rows={len(ordered)}")
    b, i = summary["balanced"], summary["imbalanced"]
    print(f"  balanced   : recall {b['spectral_recall']['mean']:.4f}  auc {b['spectral_auc']['mean']:.4f}")
    print(f"  imbalanced : recall {i['spectral_recall']['mean']:.4f}  auc {i['spectral_auc']['mean']:.4f}")
    print("  If the AUCs are close and only recall moves, balancing starved the budget.")
    print("  If the balanced AUC collapses too, balancing destroyed the score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
