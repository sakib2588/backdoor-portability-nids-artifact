"""Task 3: feasibility tests for the ndarray-based, projectable/audit_only-tagged UNSW-NB15
constraint layer (`src/constraints_secondary.py`).

Synthetic/fixture data only -- no network access, no download, no dependency on
`results/secondary_data_audit.json` existing on disk.
"""
import warnings
from typing import Tuple

import numpy as np
import pytest

from src.constraints_secondary import (
    CONSTRAINT_TOL_BOUND,
    MTU_BYTES,
    ProjectionIncompleteWarning,
    SecondaryConstraint,
    audit_constraints,
    build_secondary_constraints,
    project_secondary_to_feasible,
)

# The 11 raw columns build_secondary_constraints actually reads, in an arbitrary (non-alphabetical,
# non-dataset) order -- proves the constraint closures key off feature_names, not column position.
FEATURE_NAMES = (
    "dur", "tcprtt", "synack", "ackdat", "sttl", "dttl",
    "smeansz", "Spkts", "sbytes", "dmeansz", "Dpkts", "dbytes",
)


def _col(name: str) -> int:
    return FEATURE_NAMES.index(name)


def _clean_row() -> np.ndarray:
    """One row that satisfies every constraint in the manifest simultaneously."""
    row = np.zeros(len(FEATURE_NAMES))
    row[_col("dur")] = 1.0
    row[_col("tcprtt")] = 0.05
    row[_col("synack")] = 0.02
    row[_col("ackdat")] = 0.03
    row[_col("sttl")] = 64.0
    row[_col("dttl")] = 60.0
    row[_col("smeansz")] = 50.0
    row[_col("Spkts")] = 10.0
    row[_col("sbytes")] = 500.0
    row[_col("dmeansz")] = 60.0
    row[_col("Dpkts")] = 5.0
    row[_col("dbytes")] = 300.0
    return row


def _clean_rows(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = _clean_row()
    rows = np.tile(base, (n, 1))
    # Jitter dur only (not part of any constraint) so rows aren't byte-identical.
    rows[:, _col("dur")] = rng.uniform(0.1, 5.0, size=n)
    return rows


def _bounds_for(rows: np.ndarray, pad_hi: float = 1e6) -> Tuple[np.ndarray, np.ndarray]:
    """Wide-enough bounds that box-clipping never fights the repairs under test (a tight upper bound
    would make the MTU repair's raised Spkts/Dpkts immediately get clipped back down, which is a
    real interaction project_secondary_to_feasible must survive in production but is not what these
    idempotency/in-bounds/satisfies tests are targeting)."""
    lo = np.minimum(rows.min(axis=0), 0.0)
    hi = np.maximum(rows.max(axis=0), 0.0) + pad_hi
    return lo, hi


@pytest.fixture
def constraints() -> Tuple[SecondaryConstraint, ...]:
    return build_secondary_constraints(FEATURE_NAMES)


@pytest.fixture
def raw_rows() -> np.ndarray:
    """A batch that mixes clean rows with rows violating every PROJECTABLE constraint, so the
    projection tests exercise real repairs rather than a no-op on already-feasible data."""
    rows = _clean_rows(8, seed=1)
    # tcprtt violation
    rows[0, _col("tcprtt")] = 5.0
    # sttl / dttl out of [0, 255]
    rows[1, _col("sttl")] = 999.0
    rows[2, _col("dttl")] = -50.0
    # MTU violations (mean packet size > 1500)
    rows[3, _col("sbytes")] = 50_000.0
    rows[3, _col("Spkts")] = 1.0
    rows[4, _col("dbytes")] = 60_000.0
    rows[4, _col("Dpkts")] = 1.0
    return rows


@pytest.fixture
def bounds(raw_rows) -> Tuple[np.ndarray, np.ndarray]:
    return _bounds_for(raw_rows)


# --- Step 4's three named feasibility tests, verbatim intent -------------------------------------

def test_secondary_projection_is_idempotent(raw_rows, constraints, bounds):
    projected_once = project_secondary_to_feasible(raw_rows, constraints, bounds)
    projected_twice = project_secondary_to_feasible(projected_once, constraints, bounds)
    np.testing.assert_allclose(projected_once, projected_twice)


def test_secondary_projection_keeps_rows_in_bounds(raw_rows, constraints, bounds):
    projected = project_secondary_to_feasible(raw_rows, constraints, bounds)
    assert np.all(projected >= bounds[0])
    assert np.all(projected <= bounds[1])


def test_secondary_projection_satisfies_every_projectable_constraint(raw_rows, constraints, bounds):
    projected = project_secondary_to_feasible(raw_rows, constraints, bounds)
    assert all(c.check(projected).all() for c in constraints if c.kind == "projectable")


# --- kind taxonomy sanity --------------------------------------------------------------------------

def test_manifest_has_five_projectable_and_two_audit_only(constraints):
    kinds = [c.kind for c in constraints]
    assert kinds.count("projectable") == 5
    assert kinds.count("audit_only") == 2
    assert len(constraints) == 7


def test_manifest_omits_ltime_after_stime(constraints):
    """ltime_after_stime references Stime/Ltime, which are outside the classifier/trigger feature
    contract (raw_feature_columns excludes timestamps) -- see the constraints_secondary module
    docstring for the full reasoning. It must not silently reappear here."""
    names = {c.name for c in constraints}
    assert "ltime_after_stime" not in names


def test_audit_only_constraints_have_no_repair(constraints):
    for c in constraints:
        if c.kind == "audit_only":
            assert c.repair is None


def test_projectable_constraint_missing_repair_raises():
    with pytest.raises(ValueError, match="repair"):
        SecondaryConstraint(
            name="bad", source="test", tolerance=0.0,
            check=lambda x: np.ones(len(x), dtype=bool), kind="projectable", repair=None,
        )


def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        SecondaryConstraint(
            name="bad", source="test", tolerance=0.0,
            check=lambda x: np.ones(len(x), dtype=bool), kind="not_a_real_kind",
        )


# --- audit_constraints machinery -------------------------------------------------------------------

def test_audit_constraints_on_all_clean_rows_reports_zero_violations(constraints):
    rows = _clean_rows(20, seed=2)
    report = audit_constraints(rows, constraints)
    assert report["n_rows"] == 20
    assert report["n_constraints"] == len(constraints)
    assert all(v == 0 for v in report["violations"].values())
    assert report["valid_rate"] == pytest.approx(1.0)


def test_audit_constraints_counts_a_single_planted_violation(constraints):
    rows = _clean_rows(5, seed=3)
    rows[2, _col("sttl")] = -1.0  # violates sttl_byte_range only
    report = audit_constraints(rows, constraints)
    assert report["violations"]["sttl_byte_range"] == 1
    assert all(v == 0 for name, v in report["violations"].items() if name != "sttl_byte_range")


def test_audit_constraints_handles_zero_rows(constraints):
    empty = np.zeros((0, len(FEATURE_NAMES)))
    report = audit_constraints(empty, constraints)
    assert report["n_rows"] == 0
    assert report["valid_rate"] == 1.0


# --- per-constraint violation detection (Task 2 precedent: prove each check genuinely catches a
# synthetic violation, not just that the projection machinery runs) --------------------------------

_VIOLATIONS = {
    "tcprtt_equals_synack_plus_ackdat": {"tcprtt": 5.0},               # 5.0 != 0.02+0.03
    "sttl_byte_range": {"sttl": 300.0},                                  # > 255
    "dttl_byte_range": {"dttl": -10.0},                                  # < 0
    "smeansz_consistent_with_sbytes": {"sbytes": 50_000.0},              # smeansz*Spkts=500, far off
    "dmeansz_consistent_with_dbytes": {"dbytes": 30_000.0},              # dmeansz*Dpkts=300, far off
    "src_mean_packet_size_le_mtu": {"sbytes": 50_000.0, "Spkts": 1.0},   # mean pkt 50000 > 1500
    "dst_mean_packet_size_le_mtu": {"dbytes": 60_000.0, "Dpkts": 1.0},   # mean pkt 60000 > 1500
}


@pytest.mark.parametrize("name,overrides", sorted(_VIOLATIONS.items()))
def test_each_constraint_check_flags_a_deliberately_bad_row(name, overrides, constraints):
    row = _clean_row()
    for feat, val in overrides.items():
        row[_col(feat)] = val
    target = next(c for c in constraints if c.name == name)
    assert target.check(row[None, :])[0] == np.bool_(False)


def test_violation_table_covers_every_constraint(constraints):
    """Guards against silently forgetting a case above when a new constraint is added."""
    assert {c.name for c in constraints} == set(_VIOLATIONS.keys())


# --- MTU repair actually reaches the bound, not just "some change happened" ------------------------

def test_mtu_repair_raises_packet_count_to_satisfy_bound(constraints, bounds):
    row = _clean_row()
    row[_col("sbytes")] = 50_000.0
    row[_col("Spkts")] = 1.0  # mean packet size 50000, way over MTU_BYTES
    projected = project_secondary_to_feasible(row[None, :], constraints, bounds)
    mean_pkt = projected[0, _col("sbytes")] / projected[0, _col("Spkts")]
    assert mean_pkt <= MTU_BYTES + CONSTRAINT_TOL_BOUND
    # sbytes itself must be untouched -- the repair is only licensed to raise the divisor (Spkts).
    assert projected[0, _col("sbytes")] == pytest.approx(50_000.0)


def test_dst_mtu_repair_raises_packet_count_to_satisfy_bound(constraints, bounds):
    """Destination-side counterpart to test_mtu_repair_raises_packet_count_to_satisfy_bound above --
    same repair shape (raise the divisor, Dpkts), different feature pair. Without this, a bug that
    moved dbytes instead of Dpkts would still satisfy dst_mean_packet_size_le_mtu afterward (the
    ratio would still land <= MTU) and go undetected."""
    row = _clean_row()
    row[_col("dbytes")] = 60_000.0
    row[_col("Dpkts")] = 1.0  # mean packet size 60000, way over MTU_BYTES
    projected = project_secondary_to_feasible(row[None, :], constraints, bounds)
    mean_pkt = projected[0, _col("dbytes")] / projected[0, _col("Dpkts")]
    assert mean_pkt <= MTU_BYTES + CONSTRAINT_TOL_BOUND
    # dbytes itself must be untouched -- the repair is only licensed to raise the divisor (Dpkts).
    assert projected[0, _col("dbytes")] == pytest.approx(60_000.0)


def test_tcprtt_repair_only_touches_tcprtt(constraints, bounds):
    """tcprtt_equals_synack_plus_ackdat's repair sets tcprtt := synack + ackdat. Structurally
    load-bearing check: a bug that instead computed e.g. ackdat := tcprtt - synack would ALSO make
    the constraint pass afterward (the sum relation is symmetric in what it can absorb), so
    test_each_constraint_check_flags_a_deliberately_bad_row and
    test_secondary_projection_satisfies_every_projectable_constraint would both stay green even with
    the wrong feature moved. Only an explicit "the other two operands are untouched" assertion catches
    that bug class -- mirroring test_mtu_repair_raises_packet_count_to_satisfy_bound's
    untouched-operand pattern."""
    row = _clean_row()
    row[_col("tcprtt")] = 5.0  # synack=0.02, ackdat=0.03 (from _clean_row) -- 5.0 violates the sum
    projected = project_secondary_to_feasible(row[None, :], constraints, bounds)
    target = next(c for c in constraints if c.name == "tcprtt_equals_synack_plus_ackdat")
    assert target.check(projected)[0] == np.bool_(True)
    # synack/ackdat must be untouched -- the repair is only licensed to move tcprtt.
    assert projected[0, _col("synack")] == pytest.approx(0.02)
    assert projected[0, _col("ackdat")] == pytest.approx(0.03)


def test_build_secondary_constraints_rejects_incomplete_feature_names():
    with pytest.raises(ValueError, match="missing required column"):
        build_secondary_constraints(("dur", "sbytes"))


# --- regression: bounds tighter than a repair's target must not fail silently ----------------------
# Reviewer-reported gap (spec-compliance review, 2026-07-31): the trailing box-clip inside
# project_secondary_to_feasible's repair loop can silently undo a projectable repair when the fitted
# `hi` bound is below the value the repair needs to reach -- e.g. the MTU repair tries to raise
# `Spkts` to `sbytes / 1500`, but if `hi[Spkts]` is tighter than that target the clip pulls it straight
# back down, and the function used to return a stable, idempotent row that still failed the constraint
# it claimed to satisfy, with no signal. Exact scenario reproduced below: `hi[Spkts] = 10.0` while the
# MTU repair for this row needs `Spkts >= 50000 / 1500 ~= 33.33`.

def test_tight_bound_below_repair_target_emits_projection_incomplete_warning(constraints):
    row = _clean_row()
    row[_col("sbytes")] = 50_000.0
    row[_col("Spkts")] = 1.0  # needs Spkts >= 50000/1500 ~= 33.33 to satisfy the MTU constraint

    lo = np.zeros(len(FEATURE_NAMES))
    hi = np.full(len(FEATURE_NAMES), 1e6)
    hi[_col("Spkts")] = 10.0  # tight bound: below the repair's target of ~33.33
    bounds = (lo, hi)

    with pytest.warns(ProjectionIncompleteWarning, match="src_mean_packet_size_le_mtu"):
        projected = project_secondary_to_feasible(row[None, :], constraints, bounds)

    # The array is still returned, still clipped into bounds, and the clip -- not a bug in the
    # repair itself -- is exactly what left the constraint violated.
    assert projected[0, _col("Spkts")] == pytest.approx(10.0)
    mean_pkt = projected[0, _col("sbytes")] / projected[0, _col("Spkts")]
    assert mean_pkt > MTU_BYTES + CONSTRAINT_TOL_BOUND
    src_mtu = next(c for c in constraints if c.name == "src_mean_packet_size_le_mtu")
    assert src_mtu.check(projected)[0] == np.bool_(False)


def test_tight_bound_still_deterministic_and_idempotent_despite_incomplete_projection(constraints):
    """The warning signals an incomplete projection, but the guarantees the docstring keeps making
    (deterministic, idempotent, in-bounds) must still hold even in this failure mode -- a caller that
    ignores the warning should still get a stable, reproducible (if unrealizable) row, not garbage."""
    row = _clean_row()
    row[_col("sbytes")] = 50_000.0
    row[_col("Spkts")] = 1.0

    lo = np.zeros(len(FEATURE_NAMES))
    hi = np.full(len(FEATURE_NAMES), 1e6)
    hi[_col("Spkts")] = 10.0
    bounds = (lo, hi)

    with pytest.warns(ProjectionIncompleteWarning):
        once = project_secondary_to_feasible(row[None, :], constraints, bounds)
    with pytest.warns(ProjectionIncompleteWarning):
        twice = project_secondary_to_feasible(once, constraints, bounds)

    np.testing.assert_allclose(once, twice)
    assert np.all(once >= bounds[0])
    assert np.all(once <= bounds[1])


def test_audit_constraints_after_projection_reveals_tight_bound_failure(constraints):
    """The persistent, queryable signal callers (Task 4/5) should actually check before trusting a
    realizability claim: audit_constraints(..., bounds=bounds)['after_projection']['all_projectable_valid'].
    """
    row = _clean_row()
    row[_col("sbytes")] = 50_000.0
    row[_col("Spkts")] = 1.0

    lo = np.zeros(len(FEATURE_NAMES))
    hi = np.full(len(FEATURE_NAMES), 1e6)
    hi[_col("Spkts")] = 10.0
    bounds = (lo, hi)

    with pytest.warns(ProjectionIncompleteWarning):
        report = audit_constraints(row[None, :], constraints, bounds=bounds)

    assert report["after_projection"]["all_projectable_valid"] is False
    assert report["after_projection"]["violations"]["src_mean_packet_size_le_mtu"] == 1


def test_audit_constraints_after_projection_reports_true_when_bounds_are_wide_enough(constraints):
    """Sanity check that the new signal is not always-False: with the generous bounds the other
    projection tests use, after_projection.all_projectable_valid must be True and no warning fires."""
    row = _clean_row()
    row[_col("sbytes")] = 50_000.0
    row[_col("Spkts")] = 1.0
    row = row[None, :]
    wide_bounds = _bounds_for(row)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ProjectionIncompleteWarning)
        report = audit_constraints(row, constraints, bounds=wide_bounds)

    assert report["after_projection"]["all_projectable_valid"] is True
