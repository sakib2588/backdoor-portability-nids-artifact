"""Regression test for the box-pinned SafeDivision case in `project_to_feasible`
(src/tb_vendor/constraints_numeric.py).

Standing instruction (notes/20260731-decision-7a-calibration-gates-not-assessable.md): do not amend
the constraint-validity gate again without first writing a test that reproduces this exact
configuration. Two previous amendments to that gate used a small toy constraint set that never hit
it, so both amendments landed on gate-2 numbers that could not actually be trusted.

Root cause: `project_to_feasible` repairs a violated `SafeDivision(a, b) <= K` by RAISING the divisor
`b` to `a / K`, and deliberately never reduces the dividend `a` (the dividend may be an
equality-bound byte sum -- see that function's own docstring). Its final step every iteration is
`x = np.clip(x, lo, hi)`. If the raised value `a / K` exceeds the divisor's own box maximum `hi[b]`,
the clip caps `b` back down below what the constraint needs, so the row is still violated -- and the
next iteration raises and clips it right back to the same place, exhausting `max_iters` still
violated. This is the "box-pinned" case: it only bites when `a / K > hi[b]`, which a disciplined
`build_trigger` perturbation never produces but an unconstrained pattern (e.g. a Neural Cleanse
inversion) can.

This test documents that this limitation still exists. It must NOT be "fixed" by changing
`project_to_feasible` -- a future change that makes this test fail is a signal the projector's
behaviour changed, which is exactly what a caller relying on `NC_MIN_CONSTRAINT_VALID_FRACTION`
(notes/20260802-decision-nc-track-extension-preregistration.md) needs to know about, not something to
paper over here.
"""
from __future__ import annotations

import numpy as np

from src.tb_vendor.constraints_numeric import check_constraints, project_to_feasible
from src.tb_vendor.relation_constraint import Constant, Feature, LessEqualConstraint, SafeDivision

# A single SafeDivision(a, b) <= K constraint, the exact node shape the CTU packet-size constraints
# use (LessEqualConstraint(SafeDivision(Feature, Feature, Constant), Constant), per
# src/tb_vendor/constraints_numeric.py's module docstring).
FEATURES = ["a", "b"]
K = 1500.0  # the MTU-style bound the real CTU packet-size constraints use


def _constraint():
    return LessEqualConstraint(SafeDivision(Feature("a"), Feature("b"), Constant(0.0)), Constant(K))


def test_box_pinned_safedivision_stays_violated_after_projection():
    """The trigger condition, spelled out: a / K > hi[b].

    a=200_000, K=1500 -> the repair needs b >= 133.33 to satisfy the constraint, but b's own box
    maximum is 100.0. The repair raises b toward 133.33, the trailing clip caps it back to 100.0,
    and that repeats every iteration -- the row is still violated when max_iters is exhausted.
    """
    cons = [_constraint()]
    lo = np.array([0.0, 0.0])
    hi = np.array([1_000_000.0, 100.0])  # b's box max (100.0) is BELOW a/K (133.33): the trigger
    assert 200_000.0 / K > hi[1], "test setup must actually hit the box-pinned case"

    # b starts in-box and already violates the constraint (200_000 / 50 = 4000 > 1500), matching how
    # an unconstrained inverted pattern can land anywhere in the box.
    x_raw = np.array([[200_000.0, 50.0]])
    assert not check_constraints(x_raw, cons, FEATURES).all(), "row must start violated"

    x_proj = project_to_feasible(x_raw, cons, FEATURES, (lo, hi))

    # The known-broken behaviour this test locks in: still violated after projection.
    assert not check_constraints(x_proj, cons, FEATURES).all()
    # The divisor is pinned exactly at its box maximum -- the signature of the raise-then-clip
    # oscillation, not some other violation path (e.g. the dividend being moved, which the projector
    # is not licensed to do for a SafeDivision left-operand).
    assert x_proj[0, 1] == 100.0
    assert x_proj[0, 0] == 200_000.0  # dividend untouched, exactly as the docstring says it must be
    assert x_proj[0, 0] / x_proj[0, 1] > K  # the ratio itself is still over the bound


def test_projection_still_repairs_the_non_box_pinned_case():
    """Sanity control: when the divisor's box maximum is wide enough to reach `a / K`, the same
    projector DOES repair the constraint. This isolates the box-pinning as the cause of the failure
    above, rather than some more general breakage in the SafeDivision repair path."""
    cons = [_constraint()]
    lo = np.array([0.0, 0.0])
    hi = np.array([1_000_000.0, 1_000.0])  # wide enough: a/K = 133.33 fits comfortably under 1000
    x_raw = np.array([[200_000.0, 50.0]])
    assert not check_constraints(x_raw, cons, FEATURES).all()

    x_proj = project_to_feasible(x_raw, cons, FEATURES, (lo, hi))

    assert check_constraints(x_proj, cons, FEATURES).all()
    assert x_proj[0, 0] == 200_000.0  # dividend still untouched
