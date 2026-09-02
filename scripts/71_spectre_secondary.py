"""SPECTRE on the UNSW-NB15 window cells -- the one detector with no secondary-corpus result.

Why this exists. An IEEE Access review (2026-09-02) made this the single largest structural charge
against the manuscript: the detector verdicts are established on CTU-13, the corpus where the paper
itself concedes that poison removal is not the operative defence, while UNSW-NB15 -- where removal
does matter -- carries far less.

Checked against disk, the charge is mostly wrong but not entirely. results/secondary_detectors.json
already holds UNSW Activation Clustering (40 rows) and Neural Cleanse (20 rows), and
results/strip_detector.json holds UNSW STRIP; those results exist and were simply never reported.
SPECTRE is the genuine gap: results/spectre_window.json covers the four CTU-13 window cells only,
because scripts/53_spectre_window.py hardcodes the primary corpus.

This script is that script's secondary-corpus twin. It reuses the dual-corpus machinery already
proven in scripts/62_strip_detector.py (`load_secondary_setup`, `build_secondary_trigger`,
`poison_secondary_trainset`, `project_secondary_to_feasible`, `eligible_indices_for`) rather than
inventing a second UNSW path, and it scores Spectral alongside SPECTRE on the identical trained
model and poisoned population per row, so the two arms differ only in the scorer.

SPECTRE cleared the MNIST vision-control gate before any of this was trusted (recall 1.0 on all five
seeds, AUC 1.0): scripts/52_spectre_vision_control.py. A null here is therefore interpretable.

Cells match the UNSW window already reported for Spectral/AC/NC/STRIP, so this column slots into the
existing detector table rather than defining a new comparison.

Checkpointed per cell-seed, atomic temp-then-rename, config fingerprint.

Run:  .venv/bin/python scripts/71_spectre_secondary.py
      .venv/bin/python scripts/71_spectre_secondary.py --smoke
"""
from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config
from _m3_secondary_common import eligible_indices_for
from src import config
from src.constraints_secondary import project_secondary_to_feasible
from src.data import apply_standardiser
from src.data_secondary import load_secondary_setup
from src.detectors import poison_recall
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.detectors.spectre import spectre_scores
from src.models import attack_success_rate, clean_accuracy, mlp_penultimate_features, train_mlp
from src.poison import poison_secondary_trainset
from src.trigger import apply_secondary_trigger, build_secondary_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
# The UNSW window already reported for Spectral/AC/NC/STRIP -- see scripts/62_strip_detector.py:51.
UNSW_WINDOW_CELLS = [(0.005, 16), (0.01, 8), (0.01, 16)]
SPECTRAL_K = 5
Z_THRESHOLDS = (2.0, 2.5, 3.0)
BUDGET_MULTIPLIER = 1.5   # matches scripts/53_spectre_window.py so the corpora stay comparable

OUT = config.RESULTS / "spectre_secondary.json"
CKPT = config.RESULTS / "spectre_secondary.checkpoint.json"


def _smoke_paths():
    """A smoke run must never land on the real artefact path: a killed full run would otherwise
    leave a 1-cell smoke file sitting where the committed result belongs."""
    global OUT, CKPT
    OUT = config.RESULTS / "spectre_secondary_smoke.json"
    CKPT = config.RESULTS / "spectre_secondary_smoke.checkpoint.json"


def cell_key(seed, rate, cost) -> str:
    return f"{seed}|{rate}|{cost}"


def config_key(cfg) -> dict:
    return dict(dataset=config.SECONDARY_DATASET_ID,
                cells=[list(c) for c in UNSW_WINDOW_CELLS], seeds=list(cfg["seeds"]),
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


def score_block(scores, is_poison, expected_frac):
    """Both decision rules against one score vector, plus the ranking metric and the MAD cost.

    Identical to scripts/53_spectre_window.py's score_block, so the CTU-13 and UNSW-NB15 SPECTRE
    columns are produced by the same arithmetic and can be placed side by side.
    """
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
    smoke = "--smoke" in sys.argv
    cfg = get_config(smoke)
    global UNSW_WINDOW_CELLS
    if smoke:
        cfg["seeds"] = [42]
        UNSW_WINDOW_CELLS = UNSW_WINDOW_CELLS[:1]
        _smoke_paths()
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = config_key(cfg)
    rows = load_ckpt(key)
    print(f"device={device}  dataset={config.SECONDARY_DATASET_ID}  "
          f"cells={UNSW_WINDOW_CELLS}  seeds={cfg['seeds']}")

    S0 = load_secondary_setup(config.SECONDARY_DATASET_ID)
    eligible = eligible_indices_for(S0.feature_names)
    x_te_std = apply_standardiser(S0.scaler, S0.x_test_raw)
    x_te_attack_raw = S0.x_test_raw[S0.y_test != TARGET]

    for seed in cfg["seeds"]:
        pending = [c for c in UNSW_WINDOW_CELLS if cell_key(seed, *c) not in rows]
        if not pending:
            print(f"--- seed {seed}: all cells cached, skipping ---")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")

        x_tr_std = apply_standardiser(S0.scaler, S0.x_train_raw)
        clean_mlp = train_mlp(x_tr_std, S0.y_train, seed, epochs=cfg["mlp_epochs"], device=device,
                              log_every=5)
        ranking = shap_rank_features(clean_mlp, S0.x_train_raw, S0.scaler, target=TARGET, kind="mlp",
                                     device=device)

        trig_cache = {}
        for rate, cost in pending:
            if cost not in trig_cache:
                project_fn = partial(project_secondary_to_feasible, constraints=S0.constraints,
                                     bounds=S0.bounds)
                trig_cache[cost] = build_secondary_trigger(
                    ranking, cost, S0.x_train_raw, S0.feature_names, S0.bounds, project_fn, eligible)
            realiz = trig_cache[cost]

            x_p_std, y_p, poison_idx = poison_secondary_trainset(
                S0.x_train_raw, S0.y_train, realiz, rate, S0.scaler, target=TARGET, seed=seed)
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            x_te_attack_trig_raw = apply_secondary_trigger(x_te_attack_raw, realiz)
            asr = attack_success_rate(mlp, apply_standardiser(S0.scaler, x_te_attack_trig_raw),
                                      TARGET, device)
            cacc = clean_accuracy(mlp, x_te_std, S0.y_test, device)

            benign_pos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(benign_pos, poison_idx)
            if is_poison.sum() == 0:
                print(f"  rate={rate} cost={cost}: no poison in target class, skipped")
                continue
            feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
            expected_frac = float(is_poison.mean())   # oracle, same grant as the committed grid

            spec = spectral_scores(feats, n_components=SPECTRAL_K)
            spct = spectre_scores(feats, expected_frac=expected_frac)

            rows[cell_key(seed, rate, cost)] = dict(
                dataset=config.SECONDARY_DATASET_ID, seed=seed, rate=rate, cost=cost,
                asr=asr, clean_acc=cacc,
                n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
                expected_frac=expected_frac,
                spectral=score_block(spec, is_poison, expected_frac),
                spectre=score_block(spct, is_poison, expected_frac))
            save_ckpt(key, rows)
            r = rows[cell_key(seed, rate, cost)]
            print(f"  rate={rate} cost={cost} asr={asr:.4f}: "
                  f"spectral fixed={r['spectral']['fixed_recall']:.4f} auc={r['spectral']['auc']:.4f} | "
                  f"spectre fixed={r['spectre']['fixed_recall']:.4f} auc={r['spectre']['auc']:.4f} "
                  f"mad3={r['spectre']['mad_recall']['3.0']:.4f}")

    ordered = [rows[cell_key(s, r, c)] for s in cfg["seeds"] for r, c in UNSW_WINDOW_CELLS
               if cell_key(s, r, c) in rows]

    def agg(det, field, sub=None):
        vals = [(r[det][field][sub] if sub else r[det][field]) for r in ordered
                if (r[det][field][sub] if sub else r[det][field]) is not None]
        if not vals:
            return None
        return dict(mean=float(np.mean(vals)),
                    sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None, n=len(vals))

    summary = {
        det: dict(fixed_recall=agg(det, "fixed_recall"), auc=agg(det, "auc"),
                  mad_recall_3=agg(det, "mad_recall", "3.0"),
                  mad_fpr_3=agg(det, "mad_fpr", "3.0"))
        for det in ("spectral", "spectre")
    }
    summary["asr"] = (dict(mean=float(np.mean([r["asr"] for r in ordered])),
                           sd=float(np.std([r["asr"] for r in ordered], ddof=1))
                           if len(ordered) > 1 else None, n=len(ordered)) if ordered else None)

    OUT.write_text(json.dumps(dict(config=key, summary=summary, rows=ordered), indent=2))
    print(f"\nwrote {OUT}  elapsed={time.time() - t0:.0f}s  n_rows={len(ordered)}")
    for det in ("spectral", "spectre"):
        s = summary[det]
        if s["fixed_recall"]:
            print(f"  {det:9s}: fixed {s['fixed_recall']['mean']:.4f}  auc {s['auc']['mean']:.4f}  "
                  f"mad3 {s['mad_recall_3']['mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
