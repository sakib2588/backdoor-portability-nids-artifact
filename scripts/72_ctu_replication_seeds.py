"""Out-of-selection replication of the CTU-13 budget-starvation window on a DISJOINT seed set.

Why this exists. The four CTU-13 window cells were selected post hoc from the 125-run grid, and
every headline CTU-13 number in the manuscript -- the 0.1425 window fixed recall, the recovered
recall of 1.0 under the budget-free rule, the 6.58% false-positive price -- is then scored on the
same five seeds {42, 123, 456, 789, 1337} that chose those cells. Section V-J discloses exactly this
problem for UNSW-NB15 ("the base column is scored on the seeds that chose it") and UNSW has a
disjoint replication arm to answer it. CTU-13 has none, yet Section VI-E claims "carrying a disjoint
replication seed set" among the mitigations. An IEEE Access review (2026-09-02) caught the
mismatch, and it is a fair hit.

This script supplies the missing arm. It re-measures the four window cells on the five seeds the
manuscript ALREADY declares as its disjoint replication set in the statistical protocol, a set that
until now was exercised only on UNSW-NB15. Reusing it rather than minting a new block keeps one
replication convention across both corpora and forecloses the objection that the replication seeds
were picked after seeing the CTU-13 result. Nothing downstream selects among them, and all five are
reported.

What is measured, and why each piece is here:
  * `asr` -- so the window CRITERION itself (mean attack success at or above 0.8) can be re-checked
    out of sample. If the criterion fails on disjoint seeds, the window is a seed artefact and the
    paper needs to say so.
  * Spectral fixed-budget recall under the published multiplier 1.5, which is the convention that
    reproduces the committed window mean of 0.1425.
  * Spectral ranking AUC and the budget-free MAD recall/FPR, so the score-versus-rule separation --
    the paper's central claim -- is replicated, not just the headline recall.
  * Activation Clustering recall, because Table 8's AC window value is itself under correction.

Checkpointed per cell-seed, atomic temp-then-rename, config fingerprint.

Run:  .venv/bin/python scripts/72_ctu_replication_seeds.py
      .venv/bin/python scripts/72_ctu_replication_seeds.py --smoke
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
from src.detectors.activation_clustering import detect as ac_detect
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.models import attack_success_rate, clean_accuracy, mlp_penultimate_features, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET

# The manuscript's OWN pre-registered disjoint replication set, declared in the statistical
# protocol (paper_access/sections/04_methods.tex) as the set used "where stated". Until now it was
# stated only for UNSW-NB15. Reusing it here rather than minting a fresh block keeps one
# replication convention across both corpora, and avoids the appearance of choosing new seeds
# after seeing the CTU-13 result. Disjoint from the five selecting seeds {42, 123, 456, 789, 1337}
# and from UNSW's 501-520 batch.
REPLICATION_SEEDS = (2026, 31415, 27182, 16180, 11235)

WINDOW_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]
SPECTRAL_K = 5             # matches scripts/04 and 05
SILHOUETTE_SAMPLE = 10000  # matches scripts/05
Z_THRESHOLDS = (2.0, 2.5, 3.0)
BUDGET_MULTIPLIER = 1.5    # flag_by_scores' published default; reproduces the 0.1425 window mean
WINDOW_ASR_CRITERION = 0.8

OUT = config.RESULTS / "ctu_replication_seeds.json"
CKPT = config.RESULTS / "ctu_replication_seeds.checkpoint.json"


def _smoke_paths():
    """A smoke run must never land on the real artefact path: a killed full run would otherwise
    leave a 1-cell smoke file sitting where the committed result belongs."""
    global OUT, CKPT
    OUT = config.RESULTS / "ctu_replication_seeds_smoke.json"
    CKPT = config.RESULTS / "ctu_replication_seeds_smoke.checkpoint.json"


def cell_key(seed, rate, cost) -> str:
    return f"{seed}|{rate}|{cost}"


def config_key(cfg) -> dict:
    return dict(seeds=list(REPLICATION_SEEDS), cells=[list(c) for c in WINDOW_CELLS],
                mlp_epochs=cfg["mlp_epochs"], spectral_k=SPECTRAL_K, z=list(Z_THRESHOLDS),
                multiplier=BUDGET_MULTIPLIER, smoke=cfg["smoke"])


def load_ckpt(key):
    if not CKPT.exists():
        return {}
    try:
        blob = json.loads(CKPT.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != json.loads(json.dumps(key)):
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} row(s) already done")
    return blob.get("rows", {})


def save_ckpt(key, rows):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(CKPT)


def main() -> int:
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    cfg = get_config(smoke)
    global WINDOW_CELLS
    seeds = list(REPLICATION_SEEDS)
    if smoke:
        seeds, WINDOW_CELLS = seeds[:1], WINDOW_CELLS[:1]
        _smoke_paths()
    cfg["seeds"] = seeds
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = config_key(cfg)
    rows = load_ckpt(key)
    print(f"device={device}  cells={WINDOW_CELLS}  DISJOINT replication seeds={seeds}")

    S = load_setup(cfg)
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])
    x_bot_raw = S["x_te_raw"][S["y_te"] == 1]
    stats = raw_trigger_stats(x_tr_raw, constraints)

    for seed in seeds:
        pending = [c for c in WINDOW_CELLS if cell_key(seed, *c) not in rows]
        if not pending:
            print(f"--- seed {seed}: all cells cached, skipping ---")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")

        x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
        clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET, kind="mlp",
                                     device=device)

        trig_cache = {}
        for rate, cost in pending:
            if cost not in trig_cache:
                trig_cache[cost], _ = build_trigger(ranking, cost, x_tr_raw, constraints, features,
                                                    bounds, stats=stats)
            realiz = trig_cache[cost]

            x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
                x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
                constraints=constraints, feature_names=features, bounds=bounds)
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            x_bot_trig = apply_trigger(x_bot_raw, realiz, constraints, features, bounds)
            asr = attack_success_rate(mlp, apply_standardiser(S["scaler"], x_bot_trig), TARGET,
                                      device)
            cacc = clean_accuracy(mlp, x_te_std, S["y_te"], device)

            benign_pos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(benign_pos, poison_idx)
            if is_poison.sum() == 0:
                print(f"  rate={rate} cost={cost}: no poison in target class, skipped")
                continue
            n_clean = int((~is_poison).sum())
            feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
            expected_frac = float(is_poison.mean())

            scores = spectral_scores(feats, n_components=SPECTRAL_K)
            fixed_recall = poison_recall(
                flag_by_scores(scores, expected_frac=expected_frac,
                               multiplier=BUDGET_MULTIPLIER), is_poison)
            auc = (float(roc_auc_score(is_poison, scores))
                   if 0 < is_poison.sum() < len(is_poison) else None)
            mad_recall, mad_fpr = {}, {}
            for z in Z_THRESHOLDS:
                flag = flag_by_mad_threshold(scores, z_thresh=z)
                mad_recall[str(z)] = poison_recall(flag, is_poison)
                mad_fpr[str(z)] = float((flag & ~is_poison).sum() / n_clean) if n_clean else None

            ac_mask, ac_sil = ac_detect(feats, seed=seed, silhouette_sample=SILHOUETTE_SAMPLE)
            ac_recall = poison_recall(ac_mask, is_poison)

            rows[cell_key(seed, rate, cost)] = dict(
                seed=seed, rate=rate, cost=cost, asr=asr, clean_acc=cacc,
                n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
                expected_frac=expected_frac,
                spectral=dict(fixed_recall=fixed_recall, auc=auc,
                              mad_recall=mad_recall, mad_fpr=mad_fpr),
                ac=dict(recall=ac_recall, silhouette=ac_sil))
            save_ckpt(key, rows)
            print(f"  rate={rate} cost={cost} asr={asr:.4f}: spectral fixed={fixed_recall:.4f} "
                  f"auc={auc:.4f} mad3={mad_recall['3.0']:.4f} fpr3={mad_fpr['3.0']:.4f} | "
                  f"ac={ac_recall:.4f}")

    ordered = [rows[cell_key(s, r, c)] for s in seeds for r, c in WINDOW_CELLS
               if cell_key(s, r, c) in rows]

    def agg(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return dict(mean=float(np.mean(vals)),
                    sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None, n=len(vals))

    summary = dict(
        asr=agg([r["asr"] for r in ordered]),
        spectral_fixed_recall=agg([r["spectral"]["fixed_recall"] for r in ordered]),
        spectral_auc=agg([r["spectral"]["auc"] for r in ordered]),
        spectral_mad_recall_3=agg([r["spectral"]["mad_recall"]["3.0"] for r in ordered]),
        spectral_mad_fpr_3=agg([r["spectral"]["mad_fpr"]["3.0"] for r in ordered]),
        ac_recall=agg([r["ac"]["recall"] for r in ordered]),
    )
    # Does the window criterion itself survive out of sample? Reported per cell, because a
    # criterion that holds on average but fails in one cell is a different finding from one
    # that holds everywhere.
    per_cell = {}
    for rate, cost in WINDOW_CELLS:
        cell_rows = [r for r in ordered if r["rate"] == rate and r["cost"] == cost]
        if not cell_rows:
            continue
        a = agg([r["asr"] for r in cell_rows])
        per_cell[f"{rate}|{cost}"] = dict(
            asr=a, meets_window_criterion=bool(a["mean"] >= WINDOW_ASR_CRITERION),
            spectral_fixed_recall=agg([r["spectral"]["fixed_recall"] for r in cell_rows]),
            spectral_auc=agg([r["spectral"]["auc"] for r in cell_rows]),
            spectral_mad_recall_3=agg([r["spectral"]["mad_recall"]["3.0"] for r in cell_rows]),
            ac_recall=agg([r["ac"]["recall"] for r in cell_rows]))
    summary["per_cell"] = per_cell
    summary["window_criterion_holds_all_cells"] = bool(
        per_cell and all(c["meets_window_criterion"] for c in per_cell.values()))

    OUT.write_text(json.dumps(
        dict(config=key, replication_seeds=list(REPLICATION_SEEDS),
             window_asr_criterion=WINDOW_ASR_CRITERION, summary=summary, rows=ordered), indent=2))
    print(f"\nwrote {OUT}  elapsed={time.time() - t0:.0f}s  n_rows={len(ordered)}")
    print(f"  window criterion holds in every cell: {summary['window_criterion_holds_all_cells']}")
    print(f"  spectral fixed recall : {summary['spectral_fixed_recall']['mean']:.4f} "
          f"(committed selecting-seed value 0.1425)")
    print(f"  spectral MAD recall z3: {summary['spectral_mad_recall_3']['mean']:.4f}")
    print(f"  AC recall             : {summary['ac_recall']['mean']:.4f} "
          f"(committed selecting-seed value 0.1005)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
