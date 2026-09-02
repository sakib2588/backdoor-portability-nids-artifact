"""Statistical primitives for M4 headline reporting.

Project protocol (>=5 seeds, no parametric tests at n<=30): inference is carried by bootstrap 95%
CIs and paired effect sizes (Cohen's d_z), never by t-tests/p-values. These functions are the single
home for that logic; `bootstrap_ci` is behaviourally identical to the inline one in
scripts/04_poison_sweep.py (same percentile convention and seeded generator) so headline numbers agree.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from src import config


def bootstrap_ci(vals: Sequence[float], seed: int = 0) -> Tuple[float, List[float]]:
    """Mean and percentile bootstrap 95% CI (config.BOOTSTRAP_RESAMPLES resamples, 2.5/97.5).

    Degenerate n<2 returns (mean, [min, max]) -- a CI is undefined for a single value, so the point
    is reported as its own bounds rather than fabricating a spread.
    """
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return float(v.mean()), [float(v.min()), float(v.max())]
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, len(v), replace=True).mean()
                      for _ in range(config.BOOTSTRAP_RESAMPLES)])
    lo, hi = np.percentile(means, [(1 - config.BOOTSTRAP_CI) / 2 * 100,
                                   (1 + config.BOOTSTRAP_CI) / 2 * 100])
    return float(v.mean()), [float(lo), float(hi)]


def cohens_dz(arm_a: Sequence[float], arm_b: Sequence[float]):
    """Paired Cohen's d_z = mean(a-b) / sd(a-b), sample sd (ddof=1). n>=3 required (project protocol).

    Returns None when sd(a-b) == 0 (the paired difference is constant, so a standardised effect size
    is undefined) -- callers store None + a note rather than emitting inf/NaN into the macros SSOT.
    """
    a = np.asarray(arm_a, dtype=float)
    b = np.asarray(arm_b, dtype=float)
    if a.shape != b.shape or a.size < 3:
        raise ValueError(f"cohens_dz needs paired arms of equal length >= 3, got {a.size} and {b.size}")
    diffs = a - b
    sd = diffs.std(ddof=1)
    if sd == 0:
        return None
    return float(diffs.mean() / sd)


def paired_diff_ci(arm_a: Sequence[float], arm_b: Sequence[float], seed: int = 0):
    """Mean paired difference (a-b) and its bootstrap 95% CI. Pairs by index, then bootstraps the
    per-seed differences -- the CI is on the raw recall-drop, the quantity the paper reports."""
    a = np.asarray(arm_a, dtype=float)
    b = np.asarray(arm_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired_diff_ci needs equal-length arms, got {a.size} and {b.size}")
    return bootstrap_ci((a - b).tolist(), seed=seed)
