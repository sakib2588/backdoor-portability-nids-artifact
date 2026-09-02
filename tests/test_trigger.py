"""Task 4: SHAP-guided clean-label trigger (realizable vs violating), raw space."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.constraints_secondary import (
    audit_constraints,
    build_secondary_constraints,
    project_secondary_to_feasible,
)
from src.data import load_ctu, build_ctu_constraints, temporal_split, fit_standardiser
from src.models import train_lightgbm
from src.tb_vendor.constraints_numeric import check_constraints
from src.trigger import (
    shap_rank_features, build_trigger, apply_trigger, equality_features,
    build_secondary_trigger, apply_secondary_trigger,
)


@pytest.fixture(scope="module")
def env():
    x, y, features, metadata = load_ctu()
    constraints = build_ctu_constraints(features)
    # feasible subsample for speed: keep the first ~12k feasible rows in temporal order
    feasible = check_constraints(x.to_numpy(dtype=float), constraints, features)
    x_f = x[feasible].iloc[:12000].reset_index(drop=True)
    y_f = y[feasible].iloc[:12000].reset_index(drop=True)
    x_tr, y_tr, _, _, _, _ = temporal_split(x_f, y_f, train_rows=9000, val_frac=0.2, seed=1319)
    scaler = fit_standardiser(x_tr)
    lgb = train_lightgbm(scaler.transform(x_tr.to_numpy()), y_tr.to_numpy(), seed=42)
    meta_idx = metadata.set_index("feature")
    bounds = (meta_idx.loc[features, "min"].to_numpy(float), meta_idx.loc[features, "max"].to_numpy(float))
    return {"features": features, "constraints": constraints, "bounds": bounds, "scaler": scaler,
            "x_tr": x_tr, "lgb": lgb}


def test_ranking_and_cost_selection(env):
    ranked = shap_rank_features(env["lgb"], env["x_tr"], env["scaler"], kind="tree")
    assert len(ranked) == len(env["features"])
    assert sorted(ranked) == list(range(len(env["features"])))          # a permutation
    for cost in config.TRIGGER_COSTS:
        realiz, violat = build_trigger(ranked, cost, env["x_tr"].to_numpy(), env["constraints"],
                                       env["features"], env["bounds"])
        assert len(realiz["indices"]) == cost == len(violat["indices"])
        # strategy A: trigger features avoid the byte-conservation equalities
        eq = equality_features(env["constraints"])
        assert not any(env["features"][i] in eq for i in realiz["indices"])


def test_realizable_is_feasible_violating_can_break(env):
    ranked = shap_rank_features(env["lgb"], env["x_tr"], env["scaler"], kind="tree")
    base = env["x_tr"].to_numpy(dtype=float)[:300]
    any_violation = False
    for cost in config.TRIGGER_COSTS:
        realiz, violat = build_trigger(ranked, cost, env["x_tr"].to_numpy(), env["constraints"],
                                       env["features"], env["bounds"])
        x_real = apply_trigger(base, realiz, env["constraints"], env["features"], env["bounds"])
        assert check_constraints(x_real, env["constraints"], env["features"]).all(), \
            f"realizable trigger infeasible at cost={cost}"
        x_viol = apply_trigger(base, violat)
        if not check_constraints(x_viol, env["constraints"], env["features"]).all():
            any_violation = True
    assert any_violation, "violating variant never broke a constraint -- positive control would be vacuous"


def test_apply_trigger_stamps_values(env):
    ranked = shap_rank_features(env["lgb"], env["x_tr"], env["scaler"], kind="tree")
    realiz, violat = build_trigger(ranked, 4, env["x_tr"].to_numpy(), env["constraints"],
                                   env["features"], env["bounds"])
    base = env["x_tr"].to_numpy(dtype=float)[:10]
    x_viol = apply_trigger(base, violat)                                # unprojected: exact stamp
    for i, v in zip(violat["indices"], violat["values"]):
        assert np.allclose(x_viol[:, i], v)


# --- Task 4: secondary (UNSW-NB15) trigger, decoupled from CTU's constraint DSL ------------------
# Synthetic fixture data only (mirrors tests/test_constraints_secondary.py's fixture) -- no network
# access, no download, no dependency on the real UNSW-NB15 load.

SECONDARY_FEATURE_NAMES = (
    "dur", "tcprtt", "synack", "ackdat", "sttl", "dttl",
    "smeansz", "Spkts", "sbytes", "dmeansz", "Dpkts", "dbytes",
)

# tcprtt's own projectable repair overwrites it (tcprtt := synack + ackdat) -- excluded from trigger
# eligibility, the secondary-dataset analogue of CTU strategy A's equality-family exclusion, but
# supplied by the CALLER (not derived by build_secondary_trigger itself) per the Task 4 requirement
# that the function "must not assume CTU equality-feature exclusions".
_SECONDARY_EXCLUDED = {"tcprtt"}


def _secondary_synthetic(seed: int = 0, n: int = 4000):
    rng = np.random.default_rng(seed)
    col = {name: i for i, name in enumerate(SECONDARY_FEATURE_NAMES)}
    x = np.zeros((n, len(SECONDARY_FEATURE_NAMES)))
    x[:, col["dur"]] = rng.uniform(0.1, 5.0, n)
    x[:, col["synack"]] = rng.uniform(0.01, 0.1, n)
    x[:, col["ackdat"]] = rng.uniform(0.01, 0.1, n)
    x[:, col["tcprtt"]] = x[:, col["synack"]] + x[:, col["ackdat"]]
    x[:, col["sttl"]] = rng.uniform(30, 90, n)
    x[:, col["dttl"]] = rng.uniform(30, 90, n)
    spkts = rng.integers(1, 20, n).astype(float)
    dpkts = rng.integers(1, 20, n).astype(float)
    x[:, col["Spkts"]] = spkts
    x[:, col["Dpkts"]] = dpkts
    x[:, col["sbytes"]] = spkts * rng.uniform(40, 100, n)
    x[:, col["dbytes"]] = dpkts * rng.uniform(40, 100, n)
    x[:, col["smeansz"]] = x[:, col["sbytes"]] / spkts
    x[:, col["dmeansz"]] = x[:, col["dbytes"]] / dpkts
    y = (rng.random(n) < 0.05).astype(int)          # ~5% attack (label 1); target class 0 is benign
    return x, y


@pytest.fixture
def secondary_env():
    x, y = _secondary_synthetic()
    constraints = build_secondary_constraints(SECONDARY_FEATURE_NAMES)
    lo = np.minimum(x.min(axis=0), 0.0)
    hi = np.maximum(x.max(axis=0), 0.0) + 1e6            # generous: box-clip never fights the repairs
    bounds = (lo, hi)
    ranked = list(range(len(SECONDARY_FEATURE_NAMES)))[::-1]     # arbitrary but deterministic
    eligible_indices = [i for i, n in enumerate(SECONDARY_FEATURE_NAMES) if n not in _SECONDARY_EXCLUDED]
    scaler = fit_standardiser(pd.DataFrame(x, columns=SECONDARY_FEATURE_NAMES))
    return dict(x=x, y=y, constraints=constraints, bounds=bounds, ranked=ranked,
               eligible_indices=eligible_indices, features=SECONDARY_FEATURE_NAMES, scaler=scaler)


def _realizable_project_fn(env):
    return lambda rows: project_secondary_to_feasible(rows, env["constraints"], env["bounds"])


def test_secondary_trigger_records_selected_metadata(secondary_env):
    e = secondary_env
    cost = 4
    trig = build_secondary_trigger(e["ranked"], cost, e["x"], e["features"], e["bounds"],
                                   _realizable_project_fn(e), e["eligible_indices"], n_sigma=6.0)
    assert len(trig["indices"]) == cost == len(trig["values"]) == len(trig["feature_names"])
    assert trig["n_sigma"] == 6.0
    assert trig["n_candidate_features"] == len(e["eligible_indices"])
    assert "tcprtt" not in trig["feature_names"]                       # excluded feature never selected
    assert len(trig["post_projection_values"]) == cost
    assert len(trig["post_projection_deltas"]) == cost


def test_secondary_realizable_is_feasible_violating_can_break(secondary_env):
    e = secondary_env
    cost = 8
    # A large n_sigma so the watermark push on sttl/dttl (a simple [0, 255] box bound, index-4/5 in
    # this fixture's top-8 ranking) definitely clears the bound on its own. A smaller n_sigma=6 (the
    # production default) was observed NOT to break anything on this synthetic fixture: sbytes/Spkts
    # and dbytes/Dpkts are both in the top-8 simultaneously, so pushing both the MTU ratio's numerator
    # and denominator together can leave the ratio still under 1500, self-cancelling the violation --
    # a real finding about this fixture's feature correlations, not a bug in build_secondary_trigger.
    realiz = build_secondary_trigger(e["ranked"], cost, e["x"], e["features"], e["bounds"],
                                     _realizable_project_fn(e), e["eligible_indices"], n_sigma=50.0)
    violat = build_secondary_trigger(e["ranked"], cost, e["x"], e["features"], e["bounds"],
                                     lambda rows: rows, e["eligible_indices"], n_sigma=50.0)
    base = e["x"][:300]
    x_real = apply_secondary_trigger(base, realiz)
    assert audit_constraints(x_real, e["constraints"])["all_projectable_valid"]

    x_viol = apply_secondary_trigger(base, violat)
    assert not audit_constraints(x_viol, e["constraints"])["all_projectable_valid"], \
        "violating variant never broke a constraint -- positive control would be vacuous"


# Secondary-dataset POISONING tests (poison_secondary_trainset/poison_secondary_trainset_raw) live in
# tests/test_poison.py, alongside poison_trainset_cleanlabel's own tests -- moved there 2026-07-31
# (code-quality review) to match this codebase's established module boundary: poisoning logic belongs
# in src/poison.py (poison_secondary_trainset(_raw) moved there too), trigger construction/stamping
# stays in src/trigger.py and is what this file tests.
