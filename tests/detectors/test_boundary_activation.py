"""Constraint-boundary activation.

Shaped after tests/detectors/test_spectre.py. Every stage is tested where it should
succeed and where it should FAIL, per tests/test_density_mitigation.py's standing rule.

The must-fail case here is the detector's own stated falsifier: if benign rows sit on
boundaries too, there is no fingerprint and no threshold rescues it.
"""

import numpy as np
import pytest

from src import config
from src.detectors.boundary_activation import (
    activation_count,
    benign_activation_profile,
    slack_matrix,
)
from src.tb_vendor.constraints_numeric import LessEqualConstraint
from src.tb_vendor.relation_constraint import Constant, Feature


def _names(dim):
    return [f"f{i}" for i in range(dim)]


def _le(i, bound):
    """Constraint f_i <= bound."""
    return LessEqualConstraint(Feature(f"f{i}"), Constant(float(bound)))


def _constraints(dim, bound=1.0):
    return [_le(i, bound) for i in range(dim)]


def test_slack_is_positive_inside_and_zero_on_the_boundary():
    dim = 4
    cons, names = _constraints(dim, bound=1.0), _names(dim)
    interior = np.zeros((1, dim))          # every f_i = 0, bound 1 -> slack 1
    on_edge = np.ones((1, dim))            # every f_i = 1, bound 1 -> slack 0

    assert np.allclose(slack_matrix(interior, cons, names), 1.0)
    assert np.allclose(slack_matrix(on_edge, cons, names), 0.0)


def test_interior_rows_score_zero_and_boundary_rows_score_high():
    """The core claim: sitting on a constraint boundary is what the score counts."""
    dim = 6
    cons, names = _constraints(dim, bound=1.0), _names(dim)
    rng = np.random.default_rng(0)
    interior = rng.uniform(-0.5, 0.5, size=(50, dim))
    on_edge = np.ones((10, dim))

    block = np.vstack([interior, on_edge])
    a = activation_count(block, cons, names)

    assert a[:50].max() == 0, "interior rows must activate nothing"
    assert a[50:].min() == dim, "rows on every boundary must activate every constraint"


def test_scores_the_whole_block_on_one_scale():
    """relative=True normalizes by a column max over the block it is handed.

    Scoring subsets separately would use different scales. This pins that the documented
    behavior is real, so the runner's single-call requirement has a reason on record.
    """
    dim = 3
    cons, names = _constraints(dim, bound=1.0), _names(dim)
    tol = config.CONSTRAINT_TOL

    # Slack just above the absolute tolerance: inactive when this block is scored alone.
    near = np.full((5, dim), 1.0 - tol * 10)
    # A far-interior block drags the column max up, so eps * scale grows and the same rows
    # become active. Same rows, same constraints, different answer.
    far = np.full((5, dim), 1.0 - 1e6)

    alone = activation_count(near, cons, names)
    together = activation_count(np.vstack([near, far]), cons, names)[:5]

    assert alone.max() == 0, f"expected the near rows to be inactive alone, got {alone}"
    assert together.min() == dim, f"expected them active in the wider block, got {together}"
    assert not np.array_equal(alone, together), (
        "the relative scale must depend on the block, which is why the runner scores the "
        "whole benign-conditioned block in one call"
    )


def test_no_separation_when_benign_rows_sit_on_boundaries_too():
    """The falsifier, and the must-fail case.

    If clean traffic is already pinned to its bounds, poisoned rows carry no distinguishing
    boundary signature and the premise of the detector is dead.
    """
    dim = 5
    cons, names = _constraints(dim, bound=1.0), _names(dim)
    benign_on_edge = np.ones((60, dim))
    poison_on_edge = np.ones((10, dim))
    block = np.vstack([benign_on_edge, poison_on_edge])
    is_poison = np.zeros(len(block), dtype=bool)
    is_poison[60:] = True

    a = activation_count(block, cons, names)
    assert a[is_poison].mean() == a[~is_poison].mean(), (
        "identical geometry must produce identical scores; any difference here is an artifact"
    )


def test_benign_profile_reports_a_high_rate_when_the_premise_is_dead():
    dim = 4
    cons, names = _constraints(dim, bound=1.0), _names(dim)
    prof = benign_activation_profile(np.ones((100, dim)), cons, names)
    assert prof["frac_rows_with_any_active"] == 1.0
    assert prof["n_rows"] == 100

    rng = np.random.default_rng(1)
    quiet = benign_activation_profile(rng.uniform(-0.5, 0.5, size=(100, dim)), cons, names)
    assert quiet["frac_rows_with_any_active"] == 0.0


def test_is_deterministic():
    dim = 5
    cons, names = _constraints(dim), _names(dim)
    rng = np.random.default_rng(2)
    block = rng.uniform(-1, 1, size=(40, dim))
    assert np.array_equal(activation_count(block, cons, names),
                          activation_count(block, cons, names))


def test_rejects_a_constraint_set_with_no_inequalities():
    """slack is undefined without inequalities, and the module says so rather than guessing."""
    with pytest.raises(ValueError):
        slack_matrix(np.zeros((3, 2)), [], _names(2))


def test_accepts_a_single_row():
    dim = 3
    cons, names = _constraints(dim), _names(dim)
    a = activation_count(np.ones(dim), cons, names)
    assert a.shape == (1,)
