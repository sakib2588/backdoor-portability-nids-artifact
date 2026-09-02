"""Differentiable (torch) mirror of the CTU constraint evaluator, for gradient-based inversion.

WHY THIS EXISTS. `src/tb_vendor/constraints_numeric.py` evaluates the 360 CTU constraints in numpy,
which is all the attack pipeline needs: triggers are projected once, then checked. Neural Cleanse's
`constraint_aware` repair candidate (plan Task 7A) needs the violation magnitude INSIDE the inversion
loss, so it has to be differentiable with respect to the reverse-engineered pattern.

This module is a node-for-node mirror of that evaluator in torch, over the same six node types the
CTU builder emits (Constant, Feature, ManySum, SafeDivision, LessEqualConstraint, EqualConstraint,
plus LessConstraint for completeness). It is NOT a second source of truth: it is pinned to the numpy
evaluator by tests/test_constraints_torch.py, which asserts the two agree on real CTU rows. If they
ever disagree, the numpy evaluator wins -- it is the one every committed feasibility number came from.

Gradient note: `SafeDivision`'s zero-divisor branch is a `torch.where`, so rows taking the fill branch
contribute no gradient through the divisor. That matches the numpy semantics and is the intended
behaviour; the penalty simply cannot push a row out of the fill branch.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from src import config
from src.tb_vendor.relation_constraint import (
    Constant,
    EqualConstraint,
    Feature,
    LessConstraint,
    LessEqualConstraint,
    ManySum,
    SafeDivision,
)


def _col_index(feature_names: Sequence[str]) -> Dict[str, int]:
    return {str(name): i for i, name in enumerate(feature_names)}


def _eval_torch(node, x: torch.Tensor, col: Dict[str, int]) -> torch.Tensor:
    """Evaluate a Value node to a per-row tensor. `x` is 2-D (n, D), RAW units, differentiable."""
    if isinstance(node, Constant):
        return torch.full((x.shape[0],), float(node.constant), dtype=x.dtype, device=x.device)
    if isinstance(node, Feature):
        return x[:, col[str(node.feature_id)]]
    if isinstance(node, ManySum):
        out = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for op in node.operands:
            out = out + _eval_torch(op, x, col)
        return out
    if isinstance(node, SafeDivision):
        d = _eval_torch(node.dividend, x, col)
        v = _eval_torch(node.divisor, x, col)
        f = _eval_torch(node.fill_value, x, col)
        zero = v == 0.0
        safe_v = torch.where(zero, torch.ones_like(v), v)
        return torch.where(zero, f, d / safe_v)
    raise NotImplementedError(f"unsupported value node: {type(node).__name__}")


def violation_magnitude_torch(
    x_raw: torch.Tensor, constraints: List, feature_names: Sequence[str],
    tol: float = config.CONSTRAINT_TOL,
) -> torch.Tensor:
    """Per-row summed constraint violation (0 = feasible), differentiable w.r.t. `x_raw`.

    Mirrors `src.tb_vendor.constraints_numeric.violation_magnitude` exactly: LessEqual/Less contribute
    `max(0, lhs - rhs - tol)`, Equal contributes `max(0, |lhs - rhs| - tol)`.
    """
    if x_raw.dim() == 1:
        x_raw = x_raw[None, :]
    col = _col_index(feature_names)
    total = torch.zeros(x_raw.shape[0], dtype=x_raw.dtype, device=x_raw.device)
    for c in constraints:
        if isinstance(c, (LessEqualConstraint, LessConstraint)):
            gap = _eval_torch(c.left_operand, x_raw, col) - _eval_torch(c.right_operand, x_raw, col)
            total = total + torch.clamp(gap - tol, min=0.0)
        elif isinstance(c, EqualConstraint):
            gap = (_eval_torch(c.left_operand, x_raw, col)
                   - _eval_torch(c.right_operand, x_raw, col)).abs()
            total = total + torch.clamp(gap - tol, min=0.0)
        else:
            raise NotImplementedError(f"unsupported constraint: {type(c).__name__}")
    return total


def split_constraints(constraints: List) -> tuple[List, List]:
    """Partition into (projectable inequalities, equalities).

    The distinction is load-bearing, not cosmetic. `project_to_feasible` repairs ONLY inequalities and
    documents that it assumes the two byte-conservation equalities already hold, because the attack
    pipeline perturbs non-equality features exclusively. A Neural-Cleanse-inverted pattern has no such
    discipline -- it moves all 757 features -- so its equalities cannot be repaired by projection and
    must be reported separately rather than folded into one pass/fail.
    """
    ineq = [c for c in constraints if isinstance(c, (LessEqualConstraint, LessConstraint))]
    eq = [c for c in constraints if isinstance(c, EqualConstraint)]
    return ineq, eq


def audit_constraints(
    x_raw, constraints: List, feature_names: Sequence[str],
    bounds: tuple = None, tol: float = config.CONSTRAINT_TOL,
) -> dict:
    """Feasibility audit of RAW rows, via the numpy evaluator (the source of truth).

    `all_projectable_valid` means every PROJECTABLE (inequality) constraint holds -- which is what the
    key has always been named for, and the only part projection can deliver. Whole-set validity is
    reported separately as `all_valid`, and equality violations get their own count, because an inverted
    pattern that moves equality-family features can never satisfy the two byte-conservation identities
    exactly and folding that into one boolean would hide it.

    Pass `bounds` to also report the audit AFTER projection onto the feasible set, which is the
    quantity the amended Task 7A validity gate turns on (see the 2026-07-30 amendment in
    notes/20260729-decision-extension-preregistration.md).

    `constraint_valid_fraction` (with `n_constraints_satisfied` / `n_constraints_total`) is the
    CONSTRAINT-level counterpart to `projectable_valid_fraction`'s ROW-level view: the share of
    projectable inequalities satisfied on every audited row. See
    notes/20260802-decision-nc-track-extension-preregistration.md Section 1.
    """
    from src.tb_vendor.constraints_numeric import (
        check_constraints, project_to_feasible, violation_magnitude,
    )

    x = np.asarray(x_raw, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    ineq, eq = split_constraints(constraints)

    def _audit(rows) -> dict:
        ok_all = check_constraints(rows, constraints, feature_names, tol=tol)
        ok_ineq = (check_constraints(rows, ineq, feature_names, tol=tol) if ineq
                   else np.ones(rows.shape[0], dtype=bool))
        ok_eq = (check_constraints(rows, eq, feature_names, tol=tol) if eq
                 else np.ones(rows.shape[0], dtype=bool))
        mag = violation_magnitude(rows, constraints, feature_names, tol=tol)
        mag_ineq = (violation_magnitude(rows, ineq, feature_names, tol=tol) if ineq
                    else np.zeros(rows.shape[0]))
        # CONSTRAINT-level (not row-level) validity: for each projectable inequality, is it satisfied
        # on every audited row? At the small row counts this audit runs at (n=2 for NC), the row-level
        # `projectable_valid_fraction` above is quantised to {0, 0.5, 1.0} and cannot distinguish "3
        # structurally SafeDivision-unrepairable constraints" from "something actually wrong" -- see
        # notes/20260802-decision-nc-track-extension-preregistration.md Section 1.
        per_constraint_ok = [bool(check_constraints(rows, [c], feature_names, tol=tol).all())
                             for c in ineq]
        n_constraints_satisfied = int(sum(per_constraint_ok))
        n_constraints_total = len(ineq)
        constraint_valid_fraction = (n_constraints_satisfied / n_constraints_total
                                     if ineq else 1.0)
        return dict(
            n_valid=int(ok_all.sum()),
            all_valid=bool(ok_all.all()),
            all_projectable_valid=bool(ok_ineq.all()),
            valid_fraction=float(ok_all.mean()),
            projectable_valid_fraction=float(ok_ineq.mean()),
            n_equality_violated_rows=int((~ok_eq).sum()),
            max_violation_magnitude=float(mag.max()) if mag.size else 0.0,
            mean_violation_magnitude=float(mag.mean()) if mag.size else 0.0,
            max_projectable_violation_magnitude=float(mag_ineq.max()) if mag_ineq.size else 0.0,
            n_constraints_satisfied=n_constraints_satisfied,
            n_constraints_total=n_constraints_total,
            constraint_valid_fraction=constraint_valid_fraction,
        )

    out = dict(n_rows=int(x.shape[0]), n_constraints=len(constraints),
               n_projectable=len(ineq), n_equality=len(eq))
    out.update(_audit(x))
    if bounds is not None:
        projected = project_to_feasible(x, constraints, feature_names, bounds, tol=tol)
        out["after_projection"] = _audit(projected)
    return out


def make_constraint_penalty(
    constraints: List, feature_names: Sequence[str], scaler, device: str = "cpu",
    tol: float = config.CONSTRAINT_TOL,
):
    """Build `penalty(pattern_std) -> scalar`, normalised by the constraint count.

    The inversion optimises in STANDARDISED space but the constraints live in RAW space, so the
    pattern is de-standardised (`raw = std * scale + mean`, differentiably) before evaluation.

    Normalising by `len(constraints)` is deliberate: without it the penalty's magnitude would scale
    with the constraint count, so the same `constraint_weight` would mean something different on a
    dataset with a different manifest and the hyperparameter would silently stop being comparable.
    """
    n = max(1, len(constraints))
    mean = torch.as_tensor(scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device)

    def penalty(pattern_std: torch.Tensor) -> torch.Tensor:
        raw = pattern_std * scale + mean
        return violation_magnitude_torch(raw, constraints, feature_names, tol=tol).sum() / n

    return penalty
