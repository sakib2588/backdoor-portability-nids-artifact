"""The active-paths detector on the CTU-13 window, behind a blocking positive control.

PRE-REGISTERED in notes/20260905-prereg-active-paths-detector.md. Bars, parameters and the score
definition were fixed there before this ran and are not swept here.

WHY. Every other detector in the paper is a vision port, so a null cannot separate "vision-built
detectors fail here" from "this attack defeats any detector". Hoyheim et al.'s active-paths method
is tabular-native and NIDS-native. It is the one untried detector family from the literature sweep,
and the manuscript already faults it for reporting no comparison against the canonical detectors
while supplying no such comparison itself.

STRUCTURE, copied from scripts/95_isolation_forest_window.py, which is the most recent
gate-then-window driver:

  1. The loud constraint-violating control runs on every seed FIRST.
  2. If its fixed-budget recall does not reach CONTROL_BAR on every seed, this writes its artifact
     with blocked=True and exits non-zero. The detector does not enter the paper and NO PARAMETER
     IS TUNED TO MAKE IT PASS. That is the user's decision of 2026-09-05 and the discipline that
     already ended density mitigation and the four gates of 2026-09-04.
  3. Only then are the four window cells scored, under the same three rules every other arm gets.

Spectral Signatures is scored on the identical block as a reference arm, so a null here can be read
against a detector known to rank this attack well on the same rows.

Run:  .venv/bin/python scripts/106_active_paths_window.py
      .venv/bin/python scripts/106_active_paths_window.py --smoke
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
from src.detectors.active_paths import (
    DEFAULT_COMPONENTS, DEFAULT_FIT_SUBSAMPLE, DEFAULT_MIN_CLUSTER_SIZE, DEFAULT_MIN_SAMPLES,
    active_path_scores, slope_matrix,
)
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.models import mlp_input_gradients, mlp_penultimate_features, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import build_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
SPECTRAL_K = 5
Z_THRESHOLDS = (2.0, 2.5, 3.0)

# Same bar the committed control applies to Spectral, scripts/04_poison_sweep.py:180.
CONTROL_BAR = 0.7

# Pre-registered primary bars, H-AP1.
PRIMARY_CONFIRMED = 0.80
PRIMARY_REFUTED = 0.65

WINDOW_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]

OUT = config.RESULTS / "active_paths_window.json"
CKPT = config.RESULTS / "active_paths_window.checkpoint.json"


def _smoke_paths():
    global OUT, CKPT
    OUT = config.RESULTS / "active_paths_window_smoke.json"
    CKPT = config.RESULTS / "active_paths_window_smoke.checkpoint.json"


def config_key(cfg) -> dict:
    return dict(seeds=list(cfg["seeds"]), mlp_epochs=cfg["mlp_epochs"],
                cells=[list(c) for c in WINDOW_CELLS], spectral_k=SPECTRAL_K,
                z=list(Z_THRESHOLDS), control_bar=CONTROL_BAR,
                components=DEFAULT_COMPONENTS, fit_subsample=DEFAULT_FIT_SUBSAMPLE,
                min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE, min_samples=DEFAULT_MIN_SAMPLES,
                control_rate=cfg["control_rate"], control_cost=cfg["control_cost"],
                smoke=cfg["smoke"])


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


def rule_block(scores, is_poison, expected_frac):
    """Both decision rules against one score vector, plus the ranking metric.

    Copied from scripts/95:106-125 so the new arm is produced by the same arithmetic as every
    other one and can be placed in the same table.
    """
    n_clean = int((~is_poison).sum())
    out = {
        "fixed_recall": poison_recall(flag_by_scores(scores, expected_frac=expected_frac), is_poison),
        "auc": (float(roc_auc_score(is_poison, scores))
                if 0 < is_poison.sum() < len(is_poison) and len(np.unique(scores)) > 1 else None),
        "mad_recall": {},
        "mad_fpr": {},
    }
    for z in Z_THRESHOLDS:
        flag = flag_by_mad_threshold(scores, z_thresh=z)
        out["mad_recall"][str(z)] = poison_recall(flag, is_poison)
        out["mad_fpr"][str(z)] = float((flag & ~is_poison).sum() / n_clean) if n_clean else None
    return out


def _score_one(x_tr_raw, y_tr, S, cfg, seed, device, rank_mlp, rate, cost, violating):
    """One poisoned block, scored by the active-paths method and by Spectral for reference."""
    realiz, violat = build_trigger(rank_mlp, cost, x_tr_raw, S["constraints"], S["features"],
                                   S["bounds"])
    if violating:
        x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
            x_tr_raw, y_tr, violat, rate, S["scaler"], target=TARGET, seed=seed)
    else:
        x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
            x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
            constraints=S["constraints"], feature_names=S["features"], bounds=S["bounds"])

    benign_pos = np.where(y_p == TARGET)[0]
    is_poison = np.isin(benign_pos, poison_idx)
    expected_frac = float(is_poison.mean())
    block = x_p_std[benign_pos]

    mlp_bd = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

    # The active path in input coordinates, then the published pipeline on top of it.
    slopes = mlp_input_gradients(mlp_bd, block, target=TARGET, device=device)
    diag = slope_matrix(slopes)
    ap = active_path_scores(slopes, seed=seed)

    feats = mlp_penultimate_features(mlp_bd, block, device)
    spec = spectral_scores(feats, n_components=SPECTRAL_K)

    del mlp_bd, feats, slopes
    return dict(
        seed=seed, rate=rate, cost=cost, violating=bool(violating),
        n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
        expected_frac=expected_frac,
        active_paths=rule_block(ap, is_poison, expected_frac),
        slope_diagnostics=diag,
        n_distinct_scores=int(len(np.unique(ap))),
        spectral_reference=rule_block(spec, is_poison, expected_frac),
    )


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
    global WINDOW_CELLS
    if smoke:
        WINDOW_CELLS = WINDOW_CELLS[:1]
    print(f"device={device}  seeds={cfg['seeds']}  cells={WINDOW_CELLS}  "
          f"control=(rate {cfg['control_rate']}, cost {cfg['control_cost']})")

    S = load_setup(cfg)
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    per_seed_rank = {}

    def rank_for(seed):
        if seed not in per_seed_rank:
            x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
            clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
            per_seed_rank[seed] = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"],
                                                     target=TARGET, kind="mlp", device=device)
        return per_seed_rank[seed]

    # ---- Gate: the loud control, every seed, before any window cell ----
    for seed in cfg["seeds"]:
        k = f"control|{seed}"
        if k in rows:
            print(f"--- control seed {seed}: cached ---")
            continue
        print(f"--- control seed {seed} ({time.time() - t0:.0f}s) ---")
        rows[k] = _score_one(x_tr_raw, y_tr, S, cfg, seed, device, rank_for(seed),
                             cfg["control_rate"], cfg["control_cost"], violating=True)
        save_ckpt(key, rows)
        r = rows[k]["active_paths"]
        print(f"  AP fixed={r['fixed_recall']:.4f} auc={r['auc']} "
              f"mad3={r['mad_recall']['3.0']:.4f} "
              f"degenerate={rows[k]['slope_diagnostics']['degenerate']} "
              f"clusters={rows[k]['n_distinct_scores']}")

    control_recalls = [rows[f"control|{s}"]["active_paths"]["fixed_recall"]
                       for s in cfg["seeds"] if f"control|{s}" in rows]
    passed = bool(control_recalls) and min(control_recalls) >= CONTROL_BAR

    if not passed:
        blob = dict(
            preregistration="notes/20260905-prereg-active-paths-detector.md",
            device=device, blocked=True, control_bar=CONTROL_BAR,
            control_recalls=control_recalls,
            controls=[rows[f"control|{s_}"] for s_ in cfg["seeds"] if f"control|{s_}" in rows],
            reason=("the loud control did not reach the bar on every seed; the detector does not "
                    "enter the paper and no parameter is tuned to pass"),
            elapsed_s=round(time.time() - t0, 1))
        OUT.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"GATE FAILED: min control recall "
              f"{min(control_recalls) if control_recalls else 'n/a'} < {CONTROL_BAR}. "
              f"Wrote {OUT} with blocked=True. Stopping.")
        return 1

    print(f"GATE PASSED: min control recall {min(control_recalls):.4f} >= {CONTROL_BAR}")

    # ---- The window cells ----
    for rate, cost in WINDOW_CELLS:
        for seed in cfg["seeds"]:
            k = f"{seed}|{rate}|{cost}"
            if k in rows:
                print(f"--- cell {k}: cached ---")
                continue
            print(f"--- cell {k} ({time.time() - t0:.0f}s) ---")
            rows[k] = _score_one(x_tr_raw, y_tr, S, cfg, seed, device, rank_for(seed),
                                 rate, cost, violating=False)
            save_ckpt(key, rows)
            r = rows[k]["active_paths"]
            print(f"  AP fixed={r['fixed_recall']:.4f} auc={r['auc']} "
                  f"mad3={r['mad_recall']['3.0']:.4f} | "
                  f"Spectral auc={rows[k]['spectral_reference']['auc']}")

    cells = [rows[f"{s}|{rt}|{c}"] for rt, c in WINDOW_CELLS for s in cfg["seeds"]
             if f"{s}|{rt}|{c}" in rows]

    def agg(fn):
        vals = [v for v in (fn(r) for r in cells) if v is not None]
        if not vals:
            return None
        return dict(mean=float(np.mean(vals)),
                    sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, n=len(vals))

    auc = agg(lambda r: r["active_paths"]["auc"])
    if auc is None:
        verdict = "NO DATA"
    elif auc["mean"] >= PRIMARY_CONFIRMED:
        verdict = "CONFIRMED"
    elif auc["mean"] < PRIMARY_REFUTED:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    blob = dict(
        preregistration="notes/20260905-prereg-active-paths-detector.md",
        device=device, blocked=False, control_bar=CONTROL_BAR,
        control_recalls=control_recalls,
        bars=dict(confirmed=PRIMARY_CONFIRMED, refuted=PRIMARY_REFUTED),
        verdict=verdict, cells=cells,
        controls=[rows[f"control|{s}"] for s in cfg["seeds"] if f"control|{s}" in rows],
        summary=dict(
            window_auc=auc,
            window_fixed_recall=agg(lambda r: r["active_paths"]["fixed_recall"]),
            window_mad3_recall=agg(lambda r: r["active_paths"]["mad_recall"]["3.0"]),
            window_mad3_fpr=agg(lambda r: r["active_paths"]["mad_fpr"]["3.0"]),
            spectral_reference_auc=agg(lambda r: r["spectral_reference"]["auc"]),
            any_degenerate=bool(any(r["slope_diagnostics"]["degenerate"] for r in cells)),
        ),
        elapsed_s=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(blob, indent=2) + "\n")

    s = blob["summary"]
    print(f"\nwindow AUC     {s['window_auc']}")
    print(f"window fixed   {s['window_fixed_recall']}")
    print(f"window MAD z=3 {s['window_mad3_recall']}")
    print(f"Spectral ref   {s['spectral_reference_auc']}")
    print(f"\nH-AP1: {verdict}  (confirmed >= {PRIMARY_CONFIRMED}, refuted < {PRIMARY_REFUTED})")
    print(f"wrote {OUT}  ({blob['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
