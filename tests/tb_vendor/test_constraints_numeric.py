"""Task 2: raw-space numeric constraint checker + feasibility projection.

Uses the REAL CTU data (cached) so the 360-constraint set and the metadata box bounds are the
genuine ones. The load is cached by huggingface_hub, so these run in a couple of seconds.
"""
import numpy as np
import pytest

from src import config
from src.data import load_ctu, build_ctu_constraints
from src.tb_vendor.constraints_numeric import (
    check_constraints,
    violation_magnitude,
    project_to_feasible,
)
from src.tb_vendor.relation_constraint import LessEqualConstraint, SafeDivision


@pytest.fixture(scope="module")
def ctu():
    x, y, features, metadata = load_ctu()
    constraints = build_ctu_constraints(features)
    meta_idx = metadata.set_index("feature")
    lo = meta_idx.loc[features, "min"].to_numpy(dtype=float)
    hi = meta_idx.loc[features, "max"].to_numpy(dtype=float)
    return {
        "x": x.to_numpy(dtype=float),
        "features": features,
        "constraints": constraints,
        "bounds": (lo, hi),
    }


def _first_feature(features, prefix):
    for f in features:
        if f.startswith(prefix):
            return f, features.index(f)
    raise AssertionError(f"no feature with prefix {prefix!r}")


def test_evaluator_correct_on_real_ctu(ctu):
    # ORACLE. The evaluator is validated against the real data two ways:
    # (1) the 326 non-packet-size constraints (2 byte-conservation equalities + 324 min/max/sum
    #     orderings) hold on EVERY real row -- a non-trivial arithmetic identity over dozens of
    #     features that would not hold if the ManySum/Feature/ordering evaluation were wrong;
    # (2) the full 360-constraint feasibility rate is >= 99.9% -- the only failures are a handful of
    #     genuine packet-size outliers (rows with physically-impossible ~1e6 bytes/packet), which the
    #     evaluator correctly flags, not an evaluator bug.
    assert len(ctu["constraints"]) == config.CTU_N_CONSTRAINTS_EXPECTED
    cons = ctu["constraints"]
    x = ctu["x"]
    non_packet = [
        c for c in cons
        if not (isinstance(c, LessEqualConstraint) and isinstance(c.left_operand, SafeDivision))
    ]
    assert len(non_packet) == 326
    assert check_constraints(x, non_packet, ctu["features"]).all()          # equalities + orderings, all rows
    feasible_rate = check_constraints(x, cons, ctu["features"]).mean()
    assert feasible_rate > 0.999                                            # only rare genuine outliers


def test_detects_a_planted_violation(ctu):
    x = ctu["x"][:50].copy()
    # a duration_max feature: left of `duration_max <= duration_sum` -- raising it above the sum
    # is a genuine ordering violation.
    name, idx = _first_feature(ctu["features"], "duration_max_")
    x[:, idx] = 1e9
    ok = check_constraints(x, ctu["constraints"], ctu["features"])
    assert not ok.any()
    assert (violation_magnitude(x, ctu["constraints"], ctu["features"]) > 0).all()


def test_projection_restores_feasibility(ctu):
    features = ctu["features"]
    cons = ctu["constraints"]
    lo, hi = ctu["bounds"]
    # start from KNOWN-feasible base rows (strategy A only repairs what the trigger touched; it does
    # not fix the rare pre-existing packet-size outliers, which live on equality-adjacent features).
    feasible = check_constraints(ctu["x"], cons, features)
    x = ctu["x"][feasible][:200].copy()
    # perturb strategy-A-safe features (duration/min/max orderings, no equalities): push them to
    # large values that break the min<=max<=sum orderings.
    for pref in ("duration_max_", "duration_min_", "bytes_out_max_", "pkts_out_max_"):
        _, idx = _first_feature(features, pref)
        x[:, idx] = hi[idx] * 5.0 + 1.0
    assert not check_constraints(x, cons, features).all()               # now infeasible
    x_fixed = project_to_feasible(x, cons, features, ctu["bounds"])
    ok = check_constraints(x_fixed, cons, features)
    assert ok.all(), f"{(~ok).sum()} rows still infeasible after projection"


def test_projection_is_idempotent(ctu):
    x = ctu["x"][:100].copy()
    _, idx = _first_feature(ctu["features"], "duration_max_")
    x[:, idx] = 1e6
    once = project_to_feasible(x, ctu["constraints"], ctu["features"], ctu["bounds"])
    twice = project_to_feasible(once, ctu["constraints"], ctu["features"], ctu["bounds"])
    assert np.allclose(once, twice)
