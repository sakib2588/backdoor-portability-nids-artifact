"""Constraint-boundary-activation detector.

Pre-registered in notes/20260904-prereg-boundary-activation-detector.md. Read that
file before changing any threshold here.

WHY THIS EXISTS, and why it is not another vision statistic. Every other detector
in this study scores a sample through the victim model, so every one of them is a
function of model loss or model confidence. H4 of
notes/20260904-hypothesis-calibration-not-representation.md shows why that is a
liability against this attack: the density-mitigation rule absorbs the clusters a
surrogate fits with lowest loss, and clean-label poison is definitionally the
easiest data in the space, being correctly labeled and sharing a stamped pattern.
A loss-shaped rule is therefore adverse to a clean-label attack by construction.

This detector never touches the model. It scores the record's position relative to
the feasible set.

THE STATISTIC. For an inequality constraint written as lhs <= rhs, a row's slack is
rhs - lhs, and the constraint is ACTIVE when that slack is within eps of zero,
meaning the row sits on the constraint's boundary. The per-row score is the number
of active constraints.

WHY IT SHOULD SEPARATE. project_to_feasible repairs an infeasible requested trigger
by moving it to the nearest feasible point, and that point lies ON the boundary of
whichever constraints were violated. Benign traffic has no reason to sit on
boundaries. Poisoned rows do, as a direct consequence of having been projected. The
realizability step that makes the attack plausible is what leaves the fingerprint.

WHAT WOULD KILL IT. If benign rows sit on boundaries at a comparable rate the
fingerprint does not exist, and no threshold rescues it. `benign_activation_profile`
exists to check that on clean data before any poisoned cell is scored.

Note the sign convention carefully. src/constraints_torch.py:62 measures VIOLATION,
clamped at zero, which projection drives to approximately nothing and which
therefore carries no post-projection signal. Slack is the complementary quantity and
is the one that survives.
"""

from typing import Dict, List, Sequence

import numpy as np

from src import config
from src.tb_vendor.constraints_numeric import (
    LessConstraint,
    LessEqualConstraint,
    _col_index,
    _eval,
)

__all__ = [
    "slack_matrix",
    "activation_count",
    "departure_count",
    "benign_activation_profile",
]


def slack_matrix(
    x_raw: np.ndarray,
    constraints: List,
    feature_names: Sequence[str],
) -> np.ndarray:
    """(n_rows, n_inequalities) signed slack. Non-negative means feasible.

    Equalities are excluded rather than folded in. An equality is active at every
    feasible row by definition, so counting it would add a constant to every score
    and dilute the statistic. src/constraints_torch.py:88 draws the same
    distinction for the same reason.
    """
    if x_raw.ndim == 1:
        x_raw = x_raw[None, :]
    col = _col_index(feature_names)
    ineq = [c for c in constraints if isinstance(c, (LessEqualConstraint, LessConstraint))]
    if not ineq:
        raise ValueError("no inequality constraints: slack is undefined")

    out = np.empty((x_raw.shape[0], len(ineq)), dtype=np.float64)
    for j, c in enumerate(ineq):
        lhs = _eval(c.left_operand, x_raw, col)
        rhs = _eval(c.right_operand, x_raw, col)
        out[:, j] = np.asarray(rhs, dtype=np.float64) - np.asarray(lhs, dtype=np.float64)
    return out


def activation_count(
    x_raw: np.ndarray,
    constraints: List,
    feature_names: Sequence[str],
    eps: float = config.CONSTRAINT_TOL,
    *,
    relative: bool = True,
) -> np.ndarray:
    """Per-row count of active constraints. Higher means closer to the boundary.

    `relative` scales eps by the constraint's own magnitude, so a constraint over
    byte counts in the millions and one over packet counts in the tens are judged
    on the same footing. An absolute eps would make the statistic a proxy for
    feature scale, which is exactly the confound that would produce a spurious
    separation between poisoned and benign rows.

    eps is pre-registered and is NOT to be swept. Sensitivity to it is a finding to
    report, not a knob.
    """
    s = slack_matrix(x_raw, constraints, feature_names)
    if relative:
        scale = np.maximum(np.abs(s).max(axis=0, keepdims=True), 1.0)
        active = np.abs(s) <= (eps * scale)
    else:
        active = np.abs(s) <= eps
    return active.sum(axis=1).astype(np.float64)


def benign_activation_profile(
    x_clean: np.ndarray,
    constraints: List,
    feature_names: Sequence[str],
    eps: float = config.CONSTRAINT_TOL,
) -> Dict:
    """Falsifier check, run on clean data before any poisoned cell is scored.

    If benign traffic already sits on boundaries at a high rate there is no
    fingerprint to find, and that is a negative result about the detector's premise
    rather than about the attack.
    """
    a = activation_count(x_clean, constraints, feature_names, eps)
    return {
        "n_rows": int(a.shape[0]),
        "mean_active": float(a.mean()),
        "median_active": float(np.median(a)),
        "max_active": float(a.max()),
        "frac_rows_with_any_active": float((a > 0).mean()),
        "quantiles": {q: float(np.quantile(a, q)) for q in (0.5, 0.9, 0.99, 1.0)},
    }


def departure_count(
    x_raw: np.ndarray,
    constraints: List,
    feature_names: Sequence[str],
    eps: float = config.CONSTRAINT_TOL,
) -> np.ndarray:
    """Per-row count of constraints the row is NOT pinned to. Higher means more suspicious.

    Pre-registered in notes/20260904-prereg-boundary-departure-inverted.md. Read that file
    before changing anything here.

    This is the complement of `activation_count`, and it exists because the proximity
    hypothesis was refuted by measurement rather than by argument. CTU-13's manifest is
    saturated: the median clean row sits at exact zero slack against 318.5 of 358
    inequality constraints, and 116 of them hold with exact equality on every row. When
    almost every row is pinned to almost every constraint, the anomaly is not touching a
    boundary, it is LEAVING one the rest of the corpus never leaves.

    The trigger stamps large values onto 16 features. Where a clean flow sat at exactly
    zero against a min/max relation, a stamped flow sits at a positive value with real
    slack, so stamping frees rows from constraints they were pinned to. Projection pulls
    them partly back but not to the benign level.

    Higher-means-more-suspicious is deliberate: it lets this score feed flag_by_scores and
    flag_by_mad_threshold unchanged, both of which read the upper tail.
    """
    # Build the slack matrix once. Calling slack_matrix for the column count and then
    # activation_count for the counts evaluated 358 constraints over the whole block twice,
    # per falsifier and per cell, to learn a constant.
    s = slack_matrix(x_raw, constraints, feature_names)
    scale = np.maximum(np.abs(s).max(axis=0, keepdims=True), 1.0)
    active = (np.abs(s) <= (eps * scale)).sum(axis=1)
    return (s.shape[1] - active).astype(np.float64)
