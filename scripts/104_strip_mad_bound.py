"""Why the median-deviation rule flags no STRIP row: the cutoff sits above the score's own bound.

The manuscript reported STRIP's 0.0000 recall under the budget-free rule and left the mechanism
open, calling the rule "severely miscalibrated for this score rather than inert on it". This script
closes that, and it closes it with an inequality rather than a hypothesis.

STRIP scores are NEGATED mean entropy (`src/detectors/strip.py`), and entropy is non-negative, so
the score is bounded above by zero. The project's budget-free rule flags
`0.6745 * (score - median) / MAD > z`, which is the same as `score > median + z * MAD / 0.6745`
(`src/detectors/spectral.py`). When the clean bulk's median sits roughly one MAD below zero, that
threshold lands ABOVE zero, above every attainable score, and the rule cannot fire on any row at any
z. That is not a calibration that is merely too strict. It is a threshold outside the statistic's
range.

Run on a real window cell rather than the positive control, because the window is what the
manuscript claims about. Writes the median, the MAD, the attained maximum, and the threshold at each
z, so `scripts/78_derived_literal_check.py` can recompute every figure the prose prints.

Run:  .venv/bin/python scripts/104_strip_mad_bound.py
"""
from __future__ import annotations

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
from src.detectors.spectral import flag_by_mad_threshold
from src.detectors.strip import DEFAULT_ALPHA, DEFAULT_N_TRIALS, strip_scores
from src.models import train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import build_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
CELL_RATE, CELL_COST = 0.005, 16          # a CTU-13 window cell
Z_THRESHOLDS = (2.0, 2.5, 3.0)
MAD_SCALE = 0.6745                        # spectral.py's normal-consistency constant
OUT = config.RESULTS / "strip_mad_bound.json"


def main() -> int:
    t0 = time.time()
    cfg = get_config(False)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = load_setup(cfg)
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])

    rows = []
    for seed in cfg["seeds"]:
        xs = apply_standardiser(S["scaler"], x_tr_raw)
        clean = train_mlp(xs, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
        rank = shap_rank_features(clean, x_tr_raw, S["scaler"], target=TARGET, kind="mlp",
                                  device=device)
        realiz, _ = build_trigger(rank, CELL_COST, x_tr_raw, S["constraints"], S["features"],
                                  S["bounds"])
        x_p_std, y_p, pidx = poison_trainset_cleanlabel(
            x_tr_raw, y_tr, realiz, CELL_RATE, S["scaler"], target=TARGET, seed=seed,
            constraints=S["constraints"], feature_names=S["features"], bounds=S["bounds"])
        mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

        bpos = np.where(y_p == TARGET)[0]
        is_poison = np.isin(bpos, pidx)
        scores = np.asarray(strip_scores(mlp, x_p_std[bpos], x_te_std, n_trials=DEFAULT_N_TRIALS,
                                         alpha=DEFAULT_ALPHA, device=device, seed=seed),
                            dtype=np.float64)

        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med)))
        smax = float(scores.max())
        per_z = {}
        for z in Z_THRESHOLDS:
            thr = med + z * mad / MAD_SCALE
            flag = flag_by_mad_threshold(scores, z_thresh=z)
            per_z[str(z)] = dict(threshold=thr, above_attainable_max=bool(thr > smax),
                                 n_flagged=int(flag.sum()),
                                 recall=float(poison_recall(flag, is_poison)))
        rows.append(dict(seed=seed, n_scored=int(scores.size), n_poison=int(is_poison.sum()),
                         median=med, mad=mad, max_score=smax, per_z=per_z))
        print(f"seed {seed}: median={med:.6f} mad={mad:.6f} max={smax:.3e} "
              f"thr(z=3)={per_z['3.0']['threshold']:+.6f} flagged={per_z['3.0']['n_flagged']}")

    agg = dict(
        median=float(np.mean([r["median"] for r in rows])),
        mad=float(np.mean([r["mad"] for r in rows])),
        max_score=float(np.max([r["max_score"] for r in rows])),
        threshold={str(z): float(np.mean([r["per_z"][str(z)]["threshold"] for r in rows]))
                   for z in Z_THRESHOLDS},
        all_thresholds_above_max=bool(all(r["per_z"][str(z)]["above_attainable_max"]
                                          for r in rows for z in Z_THRESHOLDS)),
        total_flagged=int(sum(r["per_z"][str(z)]["n_flagged"] for r in rows for z in Z_THRESHOLDS)),
    )
    OUT.write_text(json.dumps(dict(
        _question="does the budget-free cutoff lie above STRIP's attainable maximum score?",
        cell=dict(rate=CELL_RATE, cost=CELL_COST), z_thresholds=list(Z_THRESHOLDS),
        mad_scale=MAD_SCALE, seeds=list(cfg["seeds"]), summary=agg, rows=rows,
        elapsed_s=round(time.time() - t0, 1)), indent=2) + "\n")
    print(f"\nmean median={agg['median']:.6f}  mean MAD={agg['mad']:.6f}  max score={agg['max_score']:.3e}")
    for z in Z_THRESHOLDS:
        print(f"  z={z}: mean threshold {agg['threshold'][str(z)]:+.6f}")
    print(f"every threshold above the attainable maximum: {agg['all_thresholds_above_max']}")
    print(f"rows flagged in total: {agg['total_flagged']}")
    print(f"wrote {OUT}  ({agg and round(time.time() - t0, 1)}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
