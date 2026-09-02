"""Numeric evaluation, checking, and feasibility projection for the CTU 360-constraint set.

The vendored `relation_constraint` DSL is an operator-overloaded AST with NO numeric backend. This
module supplies one, in RAW feature space (the space the constraints are defined in: SafeDivision
<= 1500 raw MTU bytes, byte-conservation over raw byte counts, box [min,max] from metadata). The
only node types present across all 360 CTU constraints are:
  EqualConstraint(ManySum, ManySum)                       x2   (byte conservation; tolerance None)
  LessEqualConstraint(Feature, Feature)                   x324 (min <= max <= sum orderings)
  LessEqualConstraint(SafeDivision(Feature,Feature,Const), Constant)  x34 (packet size <= 1500)
so the evaluator branches on ManySum / EqualConstraint / LessEqualConstraint / SafeDivision /
Feature / Constant only (NOT MathOperation, which the CTU builder never emits). Everything is keyed
by `feature_id` strings; DSL nodes are unhashable and their `==` returns a constraint, so they are
never hashed or compared with `==`.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

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


def _eval(node, x: np.ndarray, col: Dict[str, int]) -> np.ndarray:
    """Evaluate a Value node to a per-row array. `x` is 2-D (n, D), RAW units."""
    if isinstance(node, Constant):
        return np.full(x.shape[0], float(node.constant), dtype=np.float64)
    if isinstance(node, Feature):
        return x[:, col[str(node.feature_id)]].astype(np.float64)
    if isinstance(node, ManySum):
        out = np.zeros(x.shape[0], dtype=np.float64)
        for op in node.operands:
            out = out + _eval(op, x, col)
        return out
    if isinstance(node, SafeDivision):
        d = _eval(node.dividend, x, col)
        v = _eval(node.divisor, x, col)
        f = _eval(node.fill_value, x, col)
        safe_v = np.where(v == 0.0, 1.0, v)
        return np.where(v == 0.0, f, d / safe_v)
    raise NotImplementedError(f"unsupported value node: {type(node).__name__}")


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr[None, :] if arr.ndim == 1 else arr


def _satisfies(constraint, x: np.ndarray, col: Dict[str, int], tol: float) -> np.ndarray:
    if isinstance(constraint, LessEqualConstraint):
        return _eval(constraint.left_operand, x, col) <= _eval(constraint.right_operand, x, col) + tol
    if isinstance(constraint, LessConstraint):
        return _eval(constraint.left_operand, x, col) < _eval(constraint.right_operand, x, col)
    if isinstance(constraint, EqualConstraint):
        lhs = _eval(constraint.left_operand, x, col)
        rhs = _eval(constraint.right_operand, x, col)
        return np.abs(lhs - rhs) <= tol
    raise NotImplementedError(f"unsupported constraint: {type(constraint).__name__}")


def check_constraints(x_raw, constraints: List, feature_names: Sequence[str],
                      tol: float = config.CONSTRAINT_TOL) -> np.ndarray:
    """Per-row boolean: does each RAW row satisfy ALL constraints?"""
    x = _as_2d(x_raw)
    col = _col_index(feature_names)
    ok = np.ones(x.shape[0], dtype=bool)
    for c in constraints:
        ok &= _satisfies(c, x, col, tol)
    return ok


def violation_magnitude(x_raw, constraints: List, feature_names: Sequence[str],
                        tol: float = config.CONSTRAINT_TOL) -> np.ndarray:
    """Per-row summed violation magnitude (0 = feasible). LessEqual -> max(0, lhs-rhs);
    Equal -> |lhs-rhs| beyond tol. Used to report how far the violating variant sits outside."""
    x = _as_2d(x_raw)
    col = _col_index(feature_names)
    total = np.zeros(x.shape[0], dtype=np.float64)
    for c in constraints:
        if isinstance(c, (LessEqualConstraint, LessConstraint)):
            gap = _eval(c.left_operand, x, col) - _eval(c.right_operand, x, col)
            total += np.clip(gap - tol, 0.0, None)
        elif isinstance(c, EqualConstraint):
            gap = np.abs(_eval(c.left_operand, x, col) - _eval(c.right_operand, x, col))
            total += np.clip(gap - tol, 0.0, None)
    return total


def project_to_feasible(x_raw, constraints: List, feature_names: Sequence[str],
                        bounds: Tuple[np.ndarray, np.ndarray], tol: float = config.CONSTRAINT_TOL,
                        max_iters: int = 200) -> np.ndarray:
    """Repair RAW rows to satisfy all constraints (nearest-feasible, minimal moves).

    Assumes the caller (build_trigger, strategy A) perturbed only NON-equality features, so the two
    byte-conservation equalities hold untouched and are not repaired here. Inequalities are repaired
    monotonically: box-clip, then for each violated `Fe(a) <= Fe(b)` reduce the left feature `a` to
    `b`, and for each violated `SafeDivision(bytes, pkts) <= C` raise the divisor `pkts` (never the
    dividend, which may be an equality-bound byte sum). Repairs only reduce left-operands / raise
    divisors, bounded by the box, so the fixed-point iteration converges; `max_iters` guards it.
    """
    lo, hi = np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64)
    x = _as_2d(x_raw).copy()
    col = _col_index(feature_names)
    x = np.clip(x, lo, hi)

    ineqs = [c for c in constraints if isinstance(c, (LessEqualConstraint, LessConstraint))]
    for _ in range(max_iters):
        changed = False
        for c in ineqs:
            left = _eval(c.left_operand, x, col)
            right = _eval(c.right_operand, x, col)
            viol = left > right + tol
            if not viol.any():
                continue
            changed = True
            if isinstance(c.left_operand, Feature) and isinstance(c.right_operand, Feature):
                a = col[str(c.left_operand.feature_id)]
                x[viol, a] = right[viol]                      # reduce left feature to the bound
            elif isinstance(c.left_operand, SafeDivision):
                sd = c.left_operand
                di = col[str(sd.dividend.feature_id)]
                dv = col[str(sd.divisor.feature_id)]
                cval = right                                  # the Constant bound (e.g. 1500)
                need = x[viol, di] / np.maximum(cval[viol], 1e-9)
                x[viol, dv] = np.maximum(x[viol, dv], need)   # raise the divisor
            else:
                raise NotImplementedError("unexpected inequality shape in projection")
        x = np.clip(x, lo, hi)
        if not changed:
            break
    return x
