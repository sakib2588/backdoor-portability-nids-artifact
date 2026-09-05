"""Can a robust upper-tail rule fire on this score at all? A label-free pre-run check.

GENERALIZES the STRIP result. scripts/104 showed that the budget-free rule recovers 0.0000 from
STRIP not because the score is uninformative but because the cutoff sits above the largest value the
statistic can take. That is not a fact about STRIP. It is a fact about applying a threshold defined
in units of spread to a statistic whose range is bounded, and it is checkable in advance.

THE CRITERION. The project's budget-free rule flags `0.6745 * (s - median) / MAD > z`
(`src/detectors/spectral.py`), which is `s > median + z * MAD / 0.6745`. Define

    z_max = 0.6745 * (max_attainable - median) / MAD

as the largest z at which at least one row can still be flagged. If `z_max < z`, the rule is INERT
BY CONSTRUCTION on that score: no row can be flagged at that threshold regardless of how well the
score separates poison from clean. A defender can compute z_max from the scores alone, with no
labels and no poison count, before deciding which rule to run.

WHY THE ARMS DIFFER. Two of the five scored arms are bounded above by construction and three are
not:

  - STRIP returns negated mean entropy, and entropy is non-negative, so the score cannot exceed 0.
  - The constraint-departure count is a count of unpinned relations, so it cannot exceed the number
    of relations in the manifest.
  - Spectral Signatures sums squared projections, SPECTRE returns a whitened quadratic form, and the
    isolation score is a forest-depth statistic. None carries an analytic ceiling that the clean bulk
    sits near.

This script measures z_max for every arm on one window cell across the five seeds, so the prediction
is checked rather than asserted. It reports the empirical maximum, which is what a defender actually
has, alongside the analytic bound where one exists.

Writes results/rule_capability_taxonomy.json.

Run:  .venv/bin/python scripts/105_rule_capability_taxonomy.py
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
from src.detectors.boundary_activation import departure_count
from src.detectors.isolation_forest import isolation_scores
from src.detectors.spectral import flag_by_mad_threshold, spectral_scores
from src.detectors.spectre import spectre_scores
from src.detectors.strip import DEFAULT_ALPHA, DEFAULT_N_TRIALS, strip_scores
from src.models import mlp_penultimate_features, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, build_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
CELL_RATE, CELL_COST = 0.005, 16
SPECTRAL_K = 5
Z_THRESHOLDS = (2.0, 2.5, 3.0)
MAD_SCALE = 0.6745
OUT = config.RESULTS / "rule_capability_taxonomy.json"

# Analytic ceilings, where the statistic has one. None means unbounded above by construction.
ANALYTIC_BOUND = {
    "spectral": None,
    "spectre": None,
    "isolation_forest": None,
    "strip": 0.0,                 # negated entropy, entropy >= 0
    "boundary_departure": "n_relations",
}


def z_max_of(scores: np.ndarray) -> tuple[float, float, float, float]:
    s = np.asarray(scores, dtype=np.float64)
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    smax = float(s.max())
    zmax = float("inf") if mad == 0 else MAD_SCALE * (smax - med) / mad
    return med, mad, smax, zmax


def main() -> int:
    t0 = time.time()
    cfg = get_config(False)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = load_setup(cfg)
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    feats_n, cons, bounds = S["features"], S["constraints"], S["bounds"]
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])
    n_relations = len(cons)

    rows = []
    for seed in cfg["seeds"]:
        xs = apply_standardiser(S["scaler"], x_tr_raw)
        clean = train_mlp(xs, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
        rank = shap_rank_features(clean, x_tr_raw, S["scaler"], target=TARGET, kind="mlp",
                                  device=device)
        realiz, _ = build_trigger(rank, CELL_COST, x_tr_raw, cons, feats_n, bounds)
        x_p_std, y_p, pidx = poison_trainset_cleanlabel(
            x_tr_raw, y_tr, realiz, CELL_RATE, S["scaler"], target=TARGET, seed=seed,
            constraints=cons, feature_names=feats_n, bounds=bounds)
        mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

        bpos = np.where(y_p == TARGET)[0]
        is_poison = np.isin(bpos, pidx)
        ef = float(is_poison.mean())
        blk = x_p_std[bpos]
        act = mlp_penultimate_features(mlp, blk, device)

        x_raw_p = np.asarray(x_tr_raw, dtype=float).copy()
        x_raw_p[pidx] = apply_trigger(x_raw_p[pidx], realiz, cons, feats_n, bounds)

        arms = {
            "spectral": spectral_scores(act, n_components=SPECTRAL_K),
            "spectre": spectre_scores(act, expected_frac=ef),
            "strip": strip_scores(mlp, blk, x_te_std, n_trials=DEFAULT_N_TRIALS,
                                  alpha=DEFAULT_ALPHA, device=device, seed=seed),
            "isolation_forest": isolation_scores(blk, seed=seed),
            "boundary_departure": departure_count(x_raw_p[bpos], cons, feats_n),
        }

        per_arm = {}
        for name, sc in arms.items():
            med, mad, smax, zmax = z_max_of(sc)
            per_arm[name] = dict(
                median=med, mad=mad, empirical_max=smax, z_max=zmax,
                analytic_bound=(ANALYTIC_BOUND[name] if ANALYTIC_BOUND[name] != "n_relations"
                                else float(n_relations)),
                bounded_above=ANALYTIC_BOUND[name] is not None,
                fires={str(z): dict(predicted=bool(zmax >= z),
                                    observed_n_flagged=int(flag_by_mad_threshold(
                                        np.asarray(sc, float), z_thresh=z).sum()),
                                    recall=float(poison_recall(flag_by_mad_threshold(
                                        np.asarray(sc, float), z_thresh=z), is_poison)))
                       for z in Z_THRESHOLDS})
        rows.append(dict(seed=seed, n_scored=int(len(is_poison)), per_arm=per_arm))
        print(f"seed {seed}: " + "  ".join(
            f"{k}:z_max={v['z_max']:.2f}" for k, v in per_arm.items()))
        del mlp, act, blk

    # ---- does the criterion predict the observed firing, arm by arm?
    summary, checks, hits = {}, 0, 0
    for name in ANALYTIC_BOUND:
        zs = [r["per_arm"][name]["z_max"] for r in rows]
        summary[name] = dict(
            bounded_above=rows[0]["per_arm"][name]["bounded_above"],
            z_max_mean=float(np.mean(zs)), z_max_min=float(np.min(zs)),
            z_max_max=float(np.max(zs)),
            inert_at={str(z): bool(np.mean(zs) < z) for z in Z_THRESHOLDS})
        for r in rows:
            for z in Z_THRESHOLDS:
                f = r["per_arm"][name]["fires"][str(z)]
                checks += 1
                hits += int(f["predicted"] == (f["observed_n_flagged"] > 0))

    blob = dict(
        _criterion="z_max = 0.6745 * (max - median) / MAD; the rule is inert when z_max < z",
        cell=dict(rate=CELL_RATE, cost=CELL_COST), seeds=list(cfg["seeds"]),
        z_thresholds=list(Z_THRESHOLDS), n_relations=n_relations,
        prediction_agreement=dict(checks=checks, agreements=hits,
                                  rate=(hits / checks if checks else None)),
        summary=summary, rows=rows, elapsed_s=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(blob, indent=2) + "\n")

    print(f"\n{'arm':<20} {'bounded':>8} {'z_max mean':>11}   inert at z=2.0/2.5/3.0")
    for name, v in summary.items():
        i = v["inert_at"]
        print(f"{name:<20} {str(v['bounded_above']):>8} {v['z_max_mean']:11.2f}   "
              f"{i['2.0']}/{i['2.5']}/{i['3.0']}")
    print(f"\ncriterion predicts observed firing on {hits} of {checks} arm-seed-threshold checks")
    print(f"wrote {OUT}  ({blob['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
