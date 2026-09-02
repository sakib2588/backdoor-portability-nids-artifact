"""M2 trigger report: per victim, per cost, build the SHAP clean-label trigger and record whether the
realizable variant is 360-constraint-valid and how far the violating variant sits outside the feasible
set. Written to results/trigger.json.

Run:  python scripts/03_build_trigger.py [--smoke]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup
from src import config
from src.data import apply_standardiser
from src.models import train_lightgbm, train_mlp
from src.tb_vendor.constraints_numeric import check_constraints, violation_magnitude
from src.trigger import shap_rank_features, build_trigger, raw_trigger_stats, apply_trigger


def main():
    smoke = "--smoke" in sys.argv
    cfg = get_config(smoke)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = load_setup(cfg)
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw = S["x_tr_raw"]
    x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
    base = x_tr_raw[:500]

    seed = cfg["seeds"][0]     # trigger construction is a per-victim property; one seed suffices here
    lgb = train_lightgbm(x_tr_std, S["y_tr"], seed)
    mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
    ranking = {"lgb": shap_rank_features(lgb, x_tr_raw, S["scaler"], target=config.ATTACK_TARGET, kind="tree"),
               "mlp": shap_rank_features(mlp, x_tr_raw, S["scaler"], target=config.ATTACK_TARGET,
                                        kind="mlp", device=device)}

    stats = raw_trigger_stats(x_tr_raw, constraints)   # mu/sd/eq: same for every (kind, cost) here

    rows = []
    for kind in ("lgb", "mlp"):
        for cost in cfg["costs"]:
            realiz, violat = build_trigger(ranking[kind], cost, x_tr_raw, constraints, features, bounds,
                                           stats=stats)
            x_real = apply_trigger(base, realiz, constraints, features, bounds)
            x_viol = apply_trigger(base, violat)
            rows.append(dict(
                victim=kind, cost=cost, trigger_features=[features[i] for i in realiz["indices"]],
                realizable_valid=bool(check_constraints(x_real, constraints, features).all()),
                violating_feasible_rate=float(check_constraints(x_viol, constraints, features).mean()),
                violating_mean_violation=float(violation_magnitude(x_viol, constraints, features).mean()),
            ))
            print(f"{kind} cost={cost}: realizable_valid={rows[-1]['realizable_valid']}  "
                  f"violating_feasible={rows[-1]['violating_feasible_rate']:.3f}")

    out = dict(config=dict(smoke=smoke, seed=seed, costs=list(cfg["costs"])), per_trigger=rows)
    (config.RESULTS / "trigger.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {config.RESULTS / 'trigger.json'}")


if __name__ == "__main__":
    main()
