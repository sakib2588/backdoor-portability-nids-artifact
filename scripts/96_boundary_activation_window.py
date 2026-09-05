#!/usr/bin/env python3
"""Constraint-boundary-activation detector: falsifier, gate 0, then the window cells.

Pre-registered in notes/20260904-prereg-boundary-activation-detector.md. Read that file
before changing any constant here. The bars below are copied from it and are not to be
adjusted after seeing a result.

ORDER IS THE POINT, and it is enforced by this driver rather than by convention:

  1. FALSIFIER, on clean data, no model, seconds. If benign CTU-13 rows already sit on
     constraint boundaries at a high rate, there is no fingerprint to find and no
     threshold rescues it. That is a negative result about the detector's premise, and it
     is worth knowing before anything expensive runs.
  2. GATE 0, the loud constraint-violating trigger, ranking AUC >= 0.90 on at least 4 of
     5 seeds. A null without a passing control is uninterpretable. If the gate fails the
     window cells do not run and NO parameter is tuned to make it pass.
  3. The four window cells, only if 2 passed.

WHY THIS DETECTOR EXISTS. H4 (notes/20260904-experiment-h4-absorption-rank.md) measured
that the density-mitigation rule absorbs clean-label poison in the first quartile of its
own loss ordering, because the rule treats poison as what a surrogate finds surprising
and clean-label poison is correctly labeled and homogeneous, therefore easy. Any statistic
that is a function of model loss inherits that failure. Constraint slack is not: it is
computed from the record alone, with no forward pass.

TWO IMPLEMENTATION TRAPS, both live:

  * poison_trainset_cleanlabel stamps and projects in RAW space and then discards the raw
    matrix, returning only the standardized block. slack_matrix needs raw units, because
    the constraints are linear relations over raw features. The raw post-projection block
    is therefore rebuilt through the same apply_trigger call the poisoning path uses, as
    scripts/37_univariate_zfilter_baseline.py:130-136 does. Stamping the watermark by hand
    would measure a trigger this paper never deploys.
  * activation_count(relative=True) normalizes eps by a column max taken over whatever
    block it is handed. Scoring clean and poisoned rows in separate calls would use two
    different scales and manufacture a separation that is not there. The whole
    benign-conditioned block goes through in ONE call, and that is asserted.
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
from src.detectors.boundary_activation import activation_count, benign_activation_profile
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores
from src.models import train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, build_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
Z_THRESHOLDS = (2.0, 2.5, 3.0)

# From the pre-registration. Fixed before the run, not swept, not adjusted after.
GATE0_AUC_BAR = 0.90
GATE0_MIN_SEEDS = 4          # of 5
PRIMARY_CONFIRMED = 0.80
PRIMARY_REFUTED = 0.65
# If benign rows are active at or above this rate, the premise is dead and we say so.
FALSIFIER_CEILING = 0.50

WINDOW_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]

OUT = config.RESULTS / "boundary_activation_window.json"
CKPT = config.RESULTS / "boundary_activation_window.checkpoint.json"


def _smoke_paths():
    global OUT, CKPT
    OUT = config.RESULTS / "boundary_activation_window_smoke.json"
    CKPT = config.RESULTS / "boundary_activation_window_smoke.checkpoint.json"


def config_key(cfg) -> dict:
    return dict(seeds=list(cfg["seeds"]), mlp_epochs=cfg["mlp_epochs"],
                control_rate=cfg["control_rate"], control_cost=cfg["control_cost"],
                cells=[list(c) for c in WINDOW_CELLS], smoke=bool(cfg.get("smoke")),
                gate0_bar=GATE0_AUC_BAR, eps="config.CONSTRAINT_TOL", relative=True)


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
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} unit(s) already done")
    return blob.get("rows", {})


def save_ckpt(key, rows):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(CKPT)


def rule_block(scores, is_poison, expected_frac):
    """Both decision rules against one score vector, plus the ranking metric.

    Copied from scripts/70:118-136 so this detector lands on the same axes as
    tab:detectors.
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


def _score_one(S, cfg, seed, device, rank_mlp, rate, cost, violating):
    """One poisoned block scored by boundary activation, in raw post-projection space."""
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]

    realiz, violat = build_trigger(rank_mlp, cost, x_tr_raw, constraints, features, bounds)
    trig = violat if violating else realiz

    if violating:
        # The control's trigger VIOLATES the manifest, so no projection -- matches
        # scripts/04_poison_sweep.py:142-143 which omits constraints/bounds deliberately.
        _, y_p, poison_idx = poison_trainset_cleanlabel(
            x_tr_raw, y_tr, trig, rate, S["scaler"], target=TARGET, seed=seed)
    else:
        _, y_p, poison_idx = poison_trainset_cleanlabel(
            x_tr_raw, y_tr, trig, rate, S["scaler"], target=TARGET, seed=seed,
            constraints=constraints, feature_names=features, bounds=bounds)

    # Rebuild the poisoned block in RAW space, through the same apply_trigger call the
    # poisoning path uses, so the rows carry the projection. Stamping the raw watermark by
    # hand would measure a trigger this paper never deploys.
    # (scripts/37_univariate_zfilter_baseline.py:130-136)
    x_poisoned_raw = np.asarray(x_tr_raw, dtype=float).copy()
    x_poisoned_raw[poison_idx] = apply_trigger(
        x_poisoned_raw[poison_idx], trig, constraints, features, bounds)

    benign_pos = np.where(y_p == TARGET)[0]
    is_poison = np.isin(benign_pos, poison_idx)
    expected_frac = float(is_poison.mean())

    # ONE call over the whole block. Splitting clean and poison into two calls would use
    # two different eps scales -- see the module docstring.
    block = x_poisoned_raw[benign_pos]
    assert block.shape[0] == is_poison.shape[0], "block and label vector must align"
    scores = activation_count(block, constraints, features)

    return dict(
        seed=seed, rate=rate, cost=cost, violating=bool(violating),
        n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
        expected_frac=expected_frac,
        mean_active_poison=float(scores[is_poison].mean()) if is_poison.any() else None,
        mean_active_clean=float(scores[~is_poison].mean()) if (~is_poison).any() else None,
        boundary=rule_block(scores, is_poison, expected_frac),
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
    print(f"device={device}  seeds={cfg['seeds']}  cells={WINDOW_CELLS}")

    S = load_setup(cfg)
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]

    # ---- 1. Falsifier, on clean data, before anything expensive ----
    print(f"--- falsifier: benign activation profile ({time.time() - t0:.0f}s) ---")
    prof = benign_activation_profile(np.asarray(x_tr_raw, dtype=float),
                                     S["constraints"], S["features"])
    print(f"  mean_active={prof['mean_active']:.4f} median={prof['median_active']:.1f} "
          f"max={prof['max_active']:.0f} frac_any={prof['frac_rows_with_any_active']:.4f}")
    premise_dead = prof["frac_rows_with_any_active"] >= FALSIFIER_CEILING
    if premise_dead:
        print(f"  FALSIFIER: {prof['frac_rows_with_any_active']:.4f} of benign rows are already "
              f"active (ceiling {FALSIFIER_CEILING}). The premise is in doubt; continuing to "
              f"gate 0 so the control says whether any separation survives.")

    per_seed_rank = {}

    def rank_for(seed):
        if seed not in per_seed_rank:
            x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
            clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
            per_seed_rank[seed] = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"],
                                                     target=TARGET, kind="mlp", device=device)
        return per_seed_rank[seed]

    # ---- 2. Gate 0 ----
    for seed in cfg["seeds"]:
        k = f"control|{seed}"
        if k in rows:
            print(f"--- gate0 seed {seed}: cached ---")
            continue
        print(f"--- gate0 seed {seed} ({time.time() - t0:.0f}s) ---")
        rows[k] = _score_one(S, cfg, seed, device, rank_for(seed),
                             cfg["control_rate"], cfg["control_cost"], violating=True)
        save_ckpt(key, rows)
        b = rows[k]["boundary"]
        print(f"  auc={b['auc']} fixed={b['fixed_recall']:.4f} mad3={b['mad_recall']['3.0']:.4f} "
              f"active(poison/clean)={rows[k]['mean_active_poison']}/{rows[k]['mean_active_clean']}")

    aucs = [rows[f"control|{s}"]["boundary"]["auc"] for s in cfg["seeds"] if f"control|{s}" in rows]
    aucs = [a for a in aucs if a is not None]
    n_clear = sum(1 for a in aucs if a >= GATE0_AUC_BAR)
    need = 1 if smoke else GATE0_MIN_SEEDS
    passed = n_clear >= need

    base = dict(preregistration="notes/20260904-prereg-boundary-activation-detector.md",
                device=device, falsifier=prof, falsifier_ceiling=FALSIFIER_CEILING,
                premise_in_doubt=bool(premise_dead),
                gate0_bar=GATE0_AUC_BAR, gate0_min_seeds=need, gate0_aucs=aucs,
                gate0_seeds_clearing=n_clear,
                controls=[rows[f"control|{s}"] for s in cfg["seeds"] if f"control|{s}" in rows])

    if not passed:
        base.update(blocked=True,
                    reason=("gate 0 did not clear: the loud constraint-violating control did not "
                            "reach ranking AUC {:.2f} on at least {} seeds. The detector does not "
                            "enter the paper and no parameter is tuned to pass."
                            .format(GATE0_AUC_BAR, need)),
                    elapsed_s=round(time.time() - t0, 1))
        OUT.write_text(json.dumps(base, indent=2) + "\n")
        print(f"GATE 0 FAILED: {n_clear}/{len(aucs)} seeds >= {GATE0_AUC_BAR} (need {need}). "
              f"Wrote {OUT} with blocked=True. Stopping, tuning nothing.")
        return 1

    print(f"GATE 0 PASSED: {n_clear}/{len(aucs)} seeds >= {GATE0_AUC_BAR}")

    # ---- 3. The four window cells ----
    for rate, cost in WINDOW_CELLS:
        for seed in cfg["seeds"]:
            k = f"{seed}|{rate}|{cost}"
            if k in rows:
                print(f"--- cell {k}: cached ---")
                continue
            print(f"--- cell {k} ({time.time() - t0:.0f}s) ---")
            rows[k] = _score_one(S, cfg, seed, device, rank_for(seed), rate, cost,
                                 violating=False)
            save_ckpt(key, rows)
            b = rows[k]["boundary"]
            print(f"  auc={b['auc']} fixed={b['fixed_recall']:.4f} "
                  f"mad3={b['mad_recall']['3.0']:.4f}")

    cells = [rows[f"{s}|{rt}|{c}"] for rt, c in WINDOW_CELLS for s in cfg["seeds"]
             if f"{s}|{rt}|{c}" in rows]

    def agg(fn):
        vals = [v for v in (fn(r) for r in cells) if v is not None]
        if not vals:
            return None
        return dict(mean=float(np.mean(vals)),
                    sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, n=len(vals))

    window_auc = agg(lambda r: r["boundary"]["auc"])
    m = window_auc["mean"] if window_auc else None
    if m is None:
        verdict = "NO DATA"
    elif m >= PRIMARY_CONFIRMED:
        verdict = "CONFIRMED"
    elif m < PRIMARY_REFUTED:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    base.update(blocked=False, cells=cells,
                summary=dict(window_auc=window_auc,
                             window_fixed_recall=agg(lambda r: r["boundary"]["fixed_recall"]),
                             window_mad3_recall=agg(lambda r: r["boundary"]["mad_recall"]["3.0"]),
                             window_mad3_fpr=agg(lambda r: r["boundary"]["mad_fpr"]["3.0"])),
                primary_bars=dict(confirmed=PRIMARY_CONFIRMED, refuted=PRIMARY_REFUTED),
                verdict=verdict, elapsed_s=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(base, indent=2) + "\n")
    print(f"\nwindow AUC {window_auc} -> {verdict}")
    print(f"wrote {OUT}  ({base['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
