#!/usr/bin/env python3
"""Does a label-free statistic of the score vector predict which removal rule works?

PRE-REGISTERED in notes/20260904-prereg-tail-separation-rule-selection.md. Read that file
before changing any constant here. Bars, statistics and the discriminating control were all
fixed before this ran.

THE QUESTION. The paper's repair story is "the inherited fixed budget starves, use the
budget-free MAD rule". That is true for Spectral and SPECTRE and false for STRIP, at the same
poison rate on the same cells: STRIP's fixed recall is 0.2767 and its MAD recall is 0.0000.
So poison rate alone cannot tell a defender which rule to use. If nothing can, the repair is
a coin flip dressed as a method. If a label-free statistic of the scores can, that is a
diagnostic a defender can actually run, and it is the objective set in
notes/20260904-decision-objective-rate-aware-removal.md.

THE CIRCULARITY TRAP, and how both statistics avoid it. Any statistic shaped like "how much
mass lies beyond the MAD threshold" IS the MAD rule, and would confirm the hypothesis by
construction. S1 and S2 below use only distributional shape: moments, order statistics and
the interquartile range. Neither contains a threshold, a multiplier, or an expected fraction.

WHY SCORE VECTORS ARE RE-COMPUTED. No runner in this project persists a score vector; every
one computes scores, applies both rules, and discards the vector. The predictor has therefore
never been computable from disk. This script persists per-unit statistics (not the raw
vectors, which would be gigabytes) alongside both rules' recall.

Two of the five detectors, Isolation Forest and boundary departure, failed their own gates
today and do not enter the paper as working detectors. They are included here as scored arms
only, because a rule-selection claim needs cases where the two rules disagree and those two
are the clearest disagreements available. Any use of this result must say so.
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
from src.detectors.boundary_activation import departure_count
from src.detectors.isolation_forest import isolation_scores
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.detectors.spectre import spectre_scores
from src.detectors.strip import strip_scores
from src.models import mlp_penultimate_features, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, build_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
SPECTRAL_K = 5
Z = 3.0
N_TRIALS = 64          # STRIP, matches scripts/62
ALPHA = 0.5            # STRIP, matches scripts/62

# Widened from the four window cells to the full 25-cell grid on 2026-09-04. The narrow
# version could not separate rule preference from detector identity: among the arms that
# cleared their gates, every unit of an arm preferred the same rule (spectral 25/25 MAD,
# spectre 25/25 MAD, strip 0/25). On the full grid Spectral itself goes both ways --- the
# committed sweep in results/full_grid_mad_sweep.json prefers MAD on 16 cells and the fixed
# budget on 9 --- so arm identity and rule preference come apart within a single detector.
# The grid matches that sweep exactly, so the two are comparable cell for cell.
GRID_RATES = [0.005, 0.01, 0.02, 0.05, 0.1]
GRID_COSTS = [1, 2, 4, 8, 16]
WINDOW_CELLS = [(r, c) for r in GRID_RATES for c in GRID_COSTS]
CONTROL_CELL = (0.05, 8)

# Pre-registered bars.
PRIMARY_CONFIRMED = 0.80
PRIMARY_REFUTED = 0.65
DISCRIMINATION_MARGIN = 0.10

OUT = config.RESULTS / "rule_selection_statistic.json"
CKPT = config.RESULTS / "rule_selection_statistic.checkpoint.json"


# ---------------------------------------------------------------- statistics
def bimodality_coefficient(s: np.ndarray) -> float:
    """S1. (skew^2 + 1) / kurtosis, on the raw scores. Moments only, no threshold."""
    s = np.asarray(s, dtype=np.float64)
    n = s.size
    sd = s.std()
    if n < 4 or sd == 0:
        return float("nan")
    z = (s - s.mean()) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())          # non-excess
    if kurt <= 0:
        return float("nan")
    return (skew ** 2 + 1.0) / kurt


def normalized_top_gap(s: np.ndarray, top_frac: float = 0.10) -> float:
    """S2. Largest consecutive gap within the top decile, over the IQR. Order stats only."""
    s = np.sort(np.asarray(s, dtype=np.float64))
    n = s.size
    if n < 20:
        return float("nan")
    iqr = float(np.subtract(*np.percentile(s, [75, 25])))
    if iqr <= 0:
        return float("nan")
    k = max(2, int(round(top_frac * n)))
    tail = s[-k:]
    return float(np.max(np.diff(tail)) / iqr)


# ---------------------------------------------------------------- plumbing
def _key_core(key):
    """The part of the config key that changes a unit's value.

    `cells` is deliberately excluded. A unit is keyed by seed|rate|cost, which already fixes
    everything that determines its numbers; the cell list only decides which units get
    computed. Including it would make widening the grid discard 25 completed units that are
    still exactly correct. Every other field --- seeds, epochs, z, k, n_trials --- does change
    a unit's value and is still compared.
    """
    return {k: v for k, v in key.items() if k != "cells"}


def load_ckpt(key):
    if not CKPT.exists():
        return {}
    try:
        blob = json.loads(CKPT.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if _key_core(blob.get("config_key") or {}) != _key_core(json.loads(json.dumps(key))):
        print("checkpoint config mismatch -- starting fresh")
        return {}
    print(f"resuming: {len(blob.get('rows', {}))} unit(s) done")
    return blob.get("rows", {})


def save_ckpt(key, rows):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(CKPT)


def score_unit(scores, is_poison, expected_frac):
    """Both rules plus both label-free statistics, on one score vector."""
    scores = np.asarray(scores, dtype=np.float64)
    fixed = poison_recall(flag_by_scores(scores, expected_frac=expected_frac), is_poison)
    mad = poison_recall(flag_by_mad_threshold(scores, z_thresh=Z), is_poison)
    return dict(
        fixed_recall=float(fixed), mad_recall=float(mad),
        rule_margin=float(mad - fixed), mad_wins=bool(mad > fixed),
        S1_bimodality=bimodality_coefficient(scores),
        S2_top_gap=normalized_top_gap(scores),
        auc=(float(roc_auc_score(is_poison, scores))
             if 0 < is_poison.sum() < len(is_poison) else None),
    )


def main() -> int:
    t0 = time.time()
    cfg = get_config(False)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells = list(dict.fromkeys(WINDOW_CELLS + [CONTROL_CELL]))
    key = dict(cells=[list(c) for c in cells], seeds=list(cfg["seeds"]),
               epochs=cfg["mlp_epochs"], z=Z, k=SPECTRAL_K, n_trials=N_TRIALS)
    rows = load_ckpt(key)
    print(f"device={device} cells={cells} seeds={cfg['seeds']}")

    S = load_setup(cfg)
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    feats_n, cons, bounds = S["features"], S["constraints"], S["bounds"]
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])   # STRIP blend pool
    ranks = {}

    for rate, cost in cells:
        for seed in cfg["seeds"]:
            k = f"{seed}|{rate}|{cost}"
            if k in rows:
                print(f"--- {k}: cached ---")
                continue
            print(f"--- {k} ({time.time() - t0:.0f}s) ---")
            if seed not in ranks:
                xs = apply_standardiser(S["scaler"], x_tr_raw)
                cm = train_mlp(xs, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
                ranks[seed] = shap_rank_features(cm, x_tr_raw, S["scaler"], target=TARGET,
                                                 kind="mlp", device=device)
            realiz, _ = build_trigger(ranks[seed], cost, x_tr_raw, cons, feats_n, bounds)
            x_p_std, y_p, pidx = poison_trainset_cleanlabel(
                x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
                constraints=cons, feature_names=feats_n, bounds=bounds)
            mlp_bd = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            bpos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(bpos, pidx)
            ef = float(is_poison.mean())
            blk = x_p_std[bpos]
            act = mlp_penultimate_features(mlp_bd, blk, device)

            # raw post-projection block for the constraint statistic (scripts/37:130-136)
            x_raw_p = np.asarray(x_tr_raw, dtype=float).copy()
            x_raw_p[pidx] = apply_trigger(x_raw_p[pidx], realiz, cons, feats_n, bounds)

            unit = dict(seed=seed, rate=rate, cost=cost, expected_frac=ef,
                        n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
                        is_control=bool((rate, cost) == CONTROL_CELL))
            unit["detectors"] = {
                "spectral": score_unit(spectral_scores(act, n_components=SPECTRAL_K),
                                       is_poison, ef),
                "spectre": score_unit(spectre_scores(act, expected_frac=ef), is_poison, ef),
                "strip": score_unit(strip_scores(mlp_bd, blk, x_te_std, n_trials=N_TRIALS,
                                                 alpha=ALPHA, device=device, seed=seed),
                                    is_poison, ef),
                "isolation_forest": score_unit(isolation_scores(blk, seed=seed), is_poison, ef),
                "boundary_departure": score_unit(
                    departure_count(x_raw_p[bpos], cons, feats_n), is_poison, ef),
            }
            rows[k] = unit
            save_ckpt(key, rows)
            for name, d in unit["detectors"].items():
                print(f"    {name:20} fixed={d['fixed_recall']:.4f} mad={d['mad_recall']:.4f} "
                      f"S1={d['S1_bimodality']:.4f} S2={d['S2_top_gap']:.4f}")

    # ------------------------------------------------------------ analysis
    units = []
    for k, u in rows.items():
        for name, d in u["detectors"].items():
            units.append(dict(unit=k, detector=name, rate=u["rate"], cost=u["cost"],
                              expected_frac=u["expected_frac"], is_control=u["is_control"], **d))

    usable = [u for u in units if np.isfinite(u["S1_bimodality"]) and np.isfinite(u["S2_top_gap"])]
    y = np.array([u["mad_wins"] for u in usable], dtype=bool)

    falsifier = dict(n_units=len(usable), n_mad_wins=int(y.sum()),
                     n_budget_wins=int((~y).sum()),
                     degenerate=bool(y.all() or (~y).all()))

    def auc_of(vals):
        v = np.asarray(vals, dtype=np.float64)
        if falsifier["degenerate"] or not np.isfinite(v).all():
            return None
        a = float(roc_auc_score(y, v))
        return max(a, 1.0 - a)          # direction-free: the statistic may point either way

    preds = {
        "S1_bimodality": auc_of([u["S1_bimodality"] for u in usable]),
        "S2_top_gap": auc_of([u["S2_top_gap"] for u in usable]),
        "poison_rate": auc_of([u["rate"] for u in usable]),
        "expected_frac": auc_of([u["expected_frac"] for u in usable]),
    }
    # ---- secondary, within-detector. Specified in HANDOVER_PC.md Section 4 (E4) before this
    # run, for one reason: on the narrow four-cell grid every arm preferred one rule
    # uniformly, so a pooled AUC could not tell "the statistic predicts the rule" apart from
    # "the statistic recognises the detector". An arm that goes both ways answers that within
    # itself, with arm identity held constant. The pre-registered bars are NOT applied here;
    # this is a secondary breakdown and is reported as one.
    per_detector = {}
    for name in sorted({u["detector"] for u in usable}):
        sub = [u for u in usable if u["detector"] == name]
        ys = np.array([u["mad_wins"] for u in sub], dtype=bool)
        deg = bool(ys.all() or (~ys).all())

        def auc_sub(vals, ys=ys, deg=deg):
            v = np.asarray(vals, dtype=np.float64)
            if deg or not np.isfinite(v).all() or v.size != ys.size:
                return None
            a = float(roc_auc_score(ys, v))
            return max(a, 1.0 - a)

        per_detector[name] = dict(
            n_units=len(sub), n_mad_wins=int(ys.sum()), n_budget_wins=int((~ys).sum()),
            degenerate=deg,
            S1_bimodality=auc_sub([u["S1_bimodality"] for u in sub]),
            S2_top_gap=auc_sub([u["S2_top_gap"] for u in sub]),
            poison_rate=auc_sub([u["rate"] for u in sub]),
        )

    best_stat = max((v for k, v in preds.items() if k.startswith("S") and v is not None),
                    default=None)
    best_rate = max((v for k, v in preds.items() if not k.startswith("S") and v is not None),
                    default=None)

    if falsifier["degenerate"]:
        verdict = "VOID (rule_margin has one sign everywhere; nothing to predict)"
    elif best_stat is None:
        verdict = "NO DATA"
    elif best_stat >= PRIMARY_CONFIRMED:
        verdict = "CONFIRMED"
    elif best_stat < PRIMARY_REFUTED:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    margin = (best_stat - best_rate) if (best_stat is not None and best_rate is not None) else None
    blob = dict(
        _preregistration="notes/20260904-prereg-tail-separation-rule-selection.md",
        _bars=dict(confirmed=PRIMARY_CONFIRMED, refuted=PRIMARY_REFUTED,
                   discrimination_margin=DISCRIMINATION_MARGIN),
        _scope=("Isolation Forest and boundary departure failed their own gates today and are "
                "scored arms only, not working detectors. Activation Clustering and Neural "
                "Cleanse are excluded: no score to compute a statistic on."),
        device=device, falsifier=falsifier, predictors=preds,
        _secondary_note=("per_detector is a secondary, within-detector breakdown specified in "
                         "HANDOVER_PC.md Section 4 (E4) before the run. The pre-registered "
                         "bars apply to the pooled primary only."),
        per_detector=per_detector,
        best_statistic_auc=best_stat, best_rate_auc=best_rate,
        discrimination=dict(margin=margin, bar=DISCRIMINATION_MARGIN,
                            supported=bool(margin is not None and margin >= DISCRIMINATION_MARGIN)),
        verdict=verdict, units=units, elapsed_s=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(blob, indent=2) + "\n")

    print(f"\nfalsifier: {falsifier['n_mad_wins']} MAD-wins vs "
          f"{falsifier['n_budget_wins']} budget-wins of {falsifier['n_units']} units")
    for k, v in preds.items():
        print(f"  {k:18} AUC {v if v is None else round(v, 4)}")
    print("\nper detector (secondary, arm identity held constant):")
    for name, d in per_detector.items():
        print(f"  {name:20} {d['n_mad_wins']:3}MAD/{d['n_budget_wins']:3}BUDGET  "
              f"S1={d['S1_bimodality']} S2={d['S2_top_gap']} rate={d['poison_rate']}")
    print(f"\nbest statistic {best_stat} vs best rate {best_rate}  margin {margin}")
    print(f"PRIMARY: {verdict}")
    print(f"DISCRIMINATION: {'SUPPORTED' if blob['discrimination']['supported'] else 'NOT SUPPORTED'}")
    print(f"wrote {OUT}  ({blob['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
