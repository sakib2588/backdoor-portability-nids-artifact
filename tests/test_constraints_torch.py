"""The torch constraint evaluator must agree with the numpy one, which is the source of truth.

src/constraints_torch.py exists only to put the CTU violation magnitude inside a gradient-based
inversion loss (Neural Cleanse's `constraint_aware` candidate, plan Task 7A). It is a second
implementation of a quantity every committed feasibility number already came from, so it is pinned
here rather than trusted.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tb_vendor.constraints_numeric import violation_magnitude
from src.tb_vendor.relation_constraint import (
    Constant,
    EqualConstraint,
    Feature,
    LessEqualConstraint,
    ManySum,
    SafeDivision,
)
from src.constraints_torch import (
    audit_constraints,
    make_constraint_penalty,
    split_constraints,
    violation_magnitude_torch,
)

FEATURES = ["a", "b", "c", "d"]


def _constraints():
    """One of each node shape the CTU builder actually emits."""
    return [
        LessEqualConstraint(Feature("a"), Feature("b")),                       # ordering
        LessEqualConstraint(                                                   # packet size <= 1500
            SafeDivision(Feature("c"), Feature("d"), Constant(0.0)), Constant(1500.0)),
        EqualConstraint(ManySum([Feature("a"), Feature("b")]), ManySum([Feature("c")])),
    ]


@pytest.mark.parametrize("rows", [
    np.array([[1.0, 2.0, 3.0, 1.0]]),                       # feasible ordering
    np.array([[5.0, 2.0, 7.0, 1.0]]),                       # a > b: violated
    np.array([[1.0, 2.0, 9000.0, 1.0]]),                    # division far over 1500
    np.array([[1.0, 2.0, 3.0, 0.0]]),                       # zero divisor -> fill branch
    np.array([[0.0, 0.0, 0.0, 0.0]]),                       # all-zero edge case
    np.array([[1.0, 2.0, 3.0, 1.0], [5.0, 2.0, 7.0, 0.0]]),  # batch, mixed
])
def test_torch_violation_matches_numpy(rows):
    cons = _constraints()
    expected = violation_magnitude(rows, cons, FEATURES)
    got = violation_magnitude_torch(torch.as_tensor(rows, dtype=torch.float64), cons, FEATURES)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-9, atol=1e-9)


def test_torch_violation_matches_numpy_on_random_rows():
    rng = np.random.default_rng(0)
    rows = rng.normal(0, 50, size=(64, len(FEATURES)))
    rows[::7, 3] = 0.0    # seed some zero divisors
    cons = _constraints()
    np.testing.assert_allclose(
        violation_magnitude_torch(torch.as_tensor(rows, dtype=torch.float64), cons, FEATURES).numpy(),
        violation_magnitude(rows, cons, FEATURES), rtol=1e-9, atol=1e-9)


def test_feasible_row_has_zero_violation():
    cons = [LessEqualConstraint(Feature("a"), Feature("b"))]
    got = violation_magnitude_torch(torch.tensor([[1.0, 2.0, 0.0, 0.0]]), cons, FEATURES)
    assert float(got.item()) == pytest.approx(0.0)


def test_violation_is_differentiable_and_pushes_toward_feasibility():
    """The whole point of this module: a gradient exists and points the right way."""
    cons = [LessEqualConstraint(Feature("a"), Feature("b"))]
    x = torch.tensor([[5.0, 2.0, 0.0, 0.0]], requires_grad=True)
    loss = violation_magnitude_torch(x, cons, FEATURES).sum()
    loss.backward()
    assert float(loss) > 0.0
    assert x.grad[0, 0] > 0   # decreasing 'a' reduces the violation
    assert x.grad[0, 1] < 0   # increasing 'b' reduces it


def test_1d_input_is_treated_as_a_single_row():
    cons = _constraints()
    row = np.array([5.0, 2.0, 7.0, 1.0])
    flat = violation_magnitude_torch(torch.as_tensor(row, dtype=torch.float64), cons, FEATURES)
    assert flat.shape == (1,)
    np.testing.assert_allclose(flat.numpy(), violation_magnitude(row, cons, FEATURES))


def test_unsupported_constraint_raises_rather_than_scoring_zero():
    class Weird:
        pass

    with pytest.raises(NotImplementedError):
        violation_magnitude_torch(torch.zeros(1, 4), [Weird()], FEATURES)


# --- the penalty wrapper --------------------------------------------------------------------------

class _Scaler:
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=float)
        self.scale_ = np.asarray(scale, dtype=float)


def test_penalty_destandardises_before_evaluating():
    """A standardised pattern that is feasible in RAW space must score 0, and vice versa."""
    cons = [LessEqualConstraint(Feature("a"), Feature("b"))]
    scaler = _Scaler(mean=[10.0, 10.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0, 1.0])
    penalty = make_constraint_penalty(cons, FEATURES, scaler)
    # std (-1, +1) -> raw (9, 11): a <= b, feasible
    assert float(penalty(torch.tensor([[-1.0, 1.0, 0.0, 0.0]]))) == pytest.approx(0.0)
    # std (+1, -1) -> raw (11, 9): violated by 2.0, normalised by 1 constraint
    assert float(penalty(torch.tensor([[1.0, -1.0, 0.0, 0.0]]))) == pytest.approx(2.0, abs=1e-5)


def test_penalty_is_normalised_by_the_constraint_count():
    """Same violation, more constraints -> the SAME per-constraint magnitude, so the weight transfers."""
    scaler = _Scaler(mean=[0.0] * 4, scale=[1.0] * 4)
    one = make_constraint_penalty([LessEqualConstraint(Feature("a"), Feature("b"))], FEATURES, scaler)
    two = make_constraint_penalty(
        [LessEqualConstraint(Feature("a"), Feature("b")),
         LessEqualConstraint(Feature("a"), Feature("b"))], FEATURES, scaler)
    x = torch.tensor([[3.0, 1.0, 0.0, 0.0]])
    assert float(one(x)) == pytest.approx(float(two(x)), abs=1e-6)


def test_audit_separates_projectable_from_equality_constraints():
    """`all_projectable_valid` must mean the INEQUALITY subset, as its name says.

    Folding the two byte-conservation equalities into one boolean would hide the fact that a
    gradient-inverted pattern can never satisfy them exactly (plan Task 7A, 2026-07-30 amendment).
    """
    cons = _constraints()   # 2 inequalities + 1 equality
    ineq, eq = split_constraints(cons)
    assert len(ineq) == 2 and len(eq) == 1

    # satisfies both inequalities, breaks the equality (a + b != c)
    row = np.array([[1.0, 2.0, 99.0, 1.0]])
    audit = audit_constraints(row, cons, FEATURES)
    assert audit["all_projectable_valid"] is True
    assert audit["all_valid"] is False
    assert audit["n_equality_violated_rows"] == 1
    assert audit["n_projectable"] == 2 and audit["n_equality"] == 1


def test_audit_reports_post_projection_validity_when_bounds_are_given():
    cons = [LessEqualConstraint(Feature("a"), Feature("b"))]
    bounds = (np.zeros(4), np.full(4, 100.0))
    row = np.array([[50.0, 10.0, 0.0, 0.0]])          # a > b: violated
    audit = audit_constraints(row, cons, FEATURES, bounds=bounds)
    assert audit["all_projectable_valid"] is False     # before projection
    assert audit["after_projection"]["all_projectable_valid"] is True   # projection repairs it


def test_audit_reports_constraint_level_valid_fraction():
    """CONSTRAINT-level (not row-level) validity: the share of individual inequality constraints that
    hold across every audited row. This resolves what the row-level `projectable_valid_fraction` cannot
    at small row counts (plan Task 7A extension, 2026-08-02 amendment,
    notes/20260802-decision-nc-track-extension-preregistration.md Section 1): a candidate that breaks
    1 of 3 constraints on every row reads 2/3 here, not a coarse row-count fraction.
    """
    always_ok_1 = LessEqualConstraint(Feature("a"), Constant(100.0))
    always_ok_2 = LessEqualConstraint(Feature("b"), Constant(100.0))
    always_violated = LessEqualConstraint(Feature("c"), Constant(1.0))
    cons = [always_ok_1, always_ok_2, always_violated]
    rows = np.array([
        [1.0, 2.0, 50.0, 0.0],   # c=50 > 1: violates always_violated on row 1
        [3.0, 4.0, 60.0, 0.0],   # c=60 > 1: violates always_violated on row 2 too
    ])
    audit = audit_constraints(rows, cons, FEATURES)
    assert audit["n_constraints_total"] == 3
    assert audit["n_constraints_satisfied"] == 2
    assert audit["constraint_valid_fraction"] == pytest.approx(2 / 3)


def test_audit_omits_the_projection_block_without_bounds():
    audit = audit_constraints(np.zeros((1, 4)), _constraints(), FEATURES)
    assert "after_projection" not in audit


def test_penalty_returns_a_scalar_with_a_gradient():
    scaler = _Scaler(mean=[0.0] * 4, scale=[1.0] * 4)
    penalty = make_constraint_penalty(
        [LessEqualConstraint(Feature("a"), Feature("b"))], FEATURES, scaler)
    x = torch.tensor([[3.0, 1.0, 0.0, 0.0]], requires_grad=True)
    out = penalty(x)
    assert out.dim() == 0
    out.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
