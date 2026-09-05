#!/usr/bin/env python3
"""Isolation Forest on the loud control and the four CTU-13 evasion-window cells.

WHY. Every detector in this study is vision-built, and the only non-vision comparator on
the board is a deliberately trivial univariate z-filter
(scripts/37_univariate_zfilter_baseline.py). Severi et al. 2021's own most-effective
mitigation against explanation-guided poisoning is an isolation-based outlier filter, and
this manuscript cites that work while never running its defense. PROJECT_STATUS.md lists
it as an unrun experiment a reviewer could name.

GATE FIRST, by construction of this driver and not by convention. The loud
constraint-violating trigger is scored before any window cell, at the same bar the
committed control uses for Spectral (fixed-budget recall >= 0.7,
scripts/04_poison_sweep.py:180). A null without a passing control is uninterpretable and
would be thrown away, so the window cells do not run unless the control clears.

The cell loop is copied from scripts/70_positive_control_strip_spectre.py rather than
re-derived, and the control construction matches scripts/04_poison_sweep.py:131-184
exactly: build_trigger's second return is the VIOLATING variant, and the poisoning call
deliberately omits constraints/bounds so no projection happens.

WHAT THIS IS NOT. Isolation Forest knows nothing about triggers, target classes or poison
fractions. It is an unsupervised outlier filter, included as a comparator native to
tabular data, which is the axis this paper is missing. Read the result that way.

One characteristic to keep in view when reading a weak result, pinned in
tests/detectors/test_isolation_forest.py: the forest splits on randomly chosen features,
so an anomaly confined to a small fraction of columns is diluted. This attack stamps 16
of 757 features. If the window numbers come back weak, dilution is the first explanation
to check, before anything is concluded about the attack.
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
from src.detectors.isolation_forest import isolation_diagnostics, isolation_scores
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.models import train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import build_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
SPECTRAL_K = 5
Z_THRESHOLDS = (2.0, 2.5, 3.0)

# Same bar the committed control applies to Spectral, scripts/04_poison_sweep.py:180.
# Fixed before the run. If it fails the window cells do not run and nothing is tuned.
CONTROL_BAR = 0.7

# The four pre-registered CTU-13 evasion-window cells, results/analysis.json's
# evasion_window block. Exactly these, no fifth.
WINDOW_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]

OUT = config.RESULTS / "isolation_forest_window.json"
CKPT = config.RESULTS / "isolation_forest_window.checkpoint.json"


def _smoke_paths():
    global OUT, CKPT
    OUT = config.RESULTS / "isolation_forest_window_smoke.json"
    CKPT = config.RESULTS / "isolation_forest_window_smoke.checkpoint.json"


def config_key(cfg) -> dict:
    return dict(seeds=list(cfg["seeds"]), mlp_epochs=cfg["mlp_epochs"],
                control_rate=cfg["control_rate"], control_cost=cfg["control_cost"],
                cells=[list(c) for c in WINDOW_CELLS], smoke=bool(cfg.get("smoke")),
                n_estimators=100, max_samples=256)


def load_ckpt(key):
    if not CKPT.exists():
        return {}
    try:
        blob = json.loads(CKPT.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    # Tuples round-trip as lists, so compare the round-tripped form. A naive compare here
    # never matches and silently redoes every cell -- scripts/62_strip_detector.py:186-190.
    if blob.get("config_key") != json.loads(json.dumps(key)):
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} unit(s) already done")
    return blob.get("rows", {})


def save_ckpt(key, rows):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(CKPT)


def rule_block(scores, is_poison, expected_frac):
    """Both decision rules against one score vector, plus the ranking metric.

    Copied from scripts/70:118-136. The two rules are reported separately throughout this
    paper because the central finding is that portability is decided by the RULE, not the
    score.
    """
    n_clean = int((~is_poison).sum())
    out = {
        "fixed_recall": poison_recall(flag_by_scores(scores, expected_frac=expected_frac), is_poison),
        "auc": (float(roc_auc_score(is_poison, scores))
                if 0 < is_poison.sum() < len(is_poison) else None),
        "mad_recall": {},
        "mad_fpr": {},
    }
    for z in Z_THRESHOLDS:
        flag = flag_by_mad_threshold(scores, z_thresh=z)
        out["mad_recall"][str(z)] = poison_recall(flag, is_poison)
        out["mad_fpr"][str(z)] = float((flag & ~is_poison).sum() / n_clean) if n_clean else None
    return out


def _score_one(x_tr_raw, y_tr, S, cfg, seed, device, rank_mlp, rate, cost, violating):
    """One poisoned block, scored by Isolation Forest and by Spectral for reference.

    `violating` selects build_trigger's second return, the constraint-violating variant,
    and suppresses projection -- the control construction from scripts/04:140-143.
    """
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

    # Isolation Forest scores the input block directly. It is model-free, so unlike the
    # activation-based detectors it needs no poisoned victim. The victim is still trained
    # for the Spectral reference arm, which is what makes this comparable to the rest.
    iso = isolation_scores(block, seed=seed)

    mlp_bd = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)
    from src.models import mlp_penultimate_features
    feats = mlp_penultimate_features(mlp_bd, block, device)
    spec = spectral_scores(feats, n_components=SPECTRAL_K)

    return dict(
        seed=seed, rate=rate, cost=cost, violating=bool(violating),
        n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
        expected_frac=expected_frac,
        isolation_forest=rule_block(iso, is_poison, expected_frac),
        isolation_diagnostics=isolation_diagnostics(iso, is_poison),
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
        r = rows[k]["isolation_forest"]
        print(f"  IF fixed={r['fixed_recall']:.4f} auc={r['auc']} "
              f"mad3={r['mad_recall']['3.0']:.4f}")

    control_recalls = [rows[f"control|{s}"]["isolation_forest"]["fixed_recall"]
                       for s in cfg["seeds"] if f"control|{s}" in rows]
    passed = bool(control_recalls) and min(control_recalls) >= CONTROL_BAR

    if not passed:
        blob = dict(preregistration="scripts/95 docstring; bar from scripts/04:180",
                    device=device, blocked=True, control_bar=CONTROL_BAR,
                    control_recalls=control_recalls,
                    controls=[rows[f"control|{s_}"] for s_ in cfg["seeds"]
                              if f"control|{s_}" in rows],
                    reason=("the loud control did not reach the bar on every seed; the "
                            "comparator does not enter the paper and no parameter is "
                            "tuned to pass"))
        OUT.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"GATE FAILED: min control recall {min(control_recalls) if control_recalls else 'n/a'} "
              f"< {CONTROL_BAR}. Wrote {OUT} with blocked=True. Stopping.")
        return 1

    print(f"GATE PASSED: min control recall {min(control_recalls):.4f} >= {CONTROL_BAR}")

    # ---- The four window cells ----
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
            r = rows[k]["isolation_forest"]
            print(f"  IF fixed={r['fixed_recall']:.4f} auc={r['auc']} "
                  f"mad3={r['mad_recall']['3.0']:.4f}")

    cells = [rows[f"{s}|{rt}|{c}"] for rt, c in WINDOW_CELLS for s in cfg["seeds"]
             if f"{s}|{rt}|{c}" in rows]

    def agg(fn):
        vals = [v for v in (fn(r) for r in cells) if v is not None]
        if not vals:
            return None
        return dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    n=len(vals))

    blob = dict(
        preregistration="scripts/95 docstring; control bar from scripts/04:180",
        device=device, blocked=False, control_bar=CONTROL_BAR,
        control_recalls=control_recalls, cells=cells,
        controls=[rows[f"control|{s}"] for s in cfg["seeds"] if f"control|{s}" in rows],
        summary=dict(
            window_auc=agg(lambda r: r["isolation_forest"]["auc"]),
            window_fixed_recall=agg(lambda r: r["isolation_forest"]["fixed_recall"]),
            window_mad3_recall=agg(lambda r: r["isolation_forest"]["mad_recall"]["3.0"]),
            window_mad3_fpr=agg(lambda r: r["isolation_forest"]["mad_fpr"]["3.0"]),
            spectral_reference_auc=agg(lambda r: r["spectral_reference"]["auc"]),
        ),
        elapsed_s=round(time.time() - t0, 1),
    )
    OUT.write_text(json.dumps(blob, indent=2) + "\n")
    s = blob["summary"]
    print(f"\nwindow: auc={s['window_auc']} fixed={s['window_fixed_recall']} "
          f"mad3={s['window_mad3_recall']}")
    print(f"wrote {OUT}  ({blob['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
