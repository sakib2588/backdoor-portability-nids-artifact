"""Spectral Signatures (Tran et al., NeurIPS 2018).

Backdoored samples leave a signature along the top singular vector(s) of the
class-conditioned representation matrix. Score each sample by its squared
projection onto that subspace; the poisoned sub-population scores highest.

`n_components` controls how many leading singular vectors the score spans.
Tran et al. (2018) use the single top vector (`n_components=1`). The backdoor
direction is not guaranteed to be the leading principal component, though: when
it is spread across the first few components, single-vector scoring misses the
tail of poison that projects weakly onto v1 while natural intra-class variance
dominates v1. A mechanistic diagnostic on the MNIST vision control (poison
median score ~5x clean, but a ~13% tail of poison ranked low under top-1) showed
this is exactly the failure mode there: recall is <0.90 on 2/5 seeds under top-1
but robustly >=0.90 for every k in {3,5,10}. Scoring over the top-k subspace is
the multi-component reading also formalised by SPECTRE (Hayase et al., 2021).
See notes/20260714-decision-spectral-ceiling.md for the diagnostic and the
project-wide decision to score over the top-k subspace on BOTH vision and NIDS.
"""
from __future__ import annotations

import numpy as np


def spectral_scores(feats: np.ndarray, n_components: int = 1) -> np.ndarray:
    feats = np.asarray(feats, dtype=np.float64)
    centered = feats - feats.mean(axis=0, keepdims=True)
    # top-k right singular vectors via SVD of the centered matrix
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = max(1, min(n_components, vt.shape[0]))
    proj = centered @ vt[:k].T  # (n, k) projection onto the top-k subspace
    return (proj ** 2).sum(axis=1)


def flag_by_scores(
    scores: np.ndarray, expected_frac: float, multiplier: float = 1.5
) -> np.ndarray:
    scores = np.asarray(scores)
    n = len(scores)
    k = int(np.ceil(multiplier * expected_frac * n))
    k = max(1, min(k, n))
    thresh_idx = np.argsort(scores)[::-1][:k]
    flagged = np.zeros(n, dtype=bool)
    flagged[thresh_idx] = True
    return flagged


def flag_by_mad_threshold(scores: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    """Adaptive, budget-free outlier flag: MAD-based z-score threshold. NOT part of Tran et al.'s
    (2018) original algorithm, which fixes a removal budget proportional to an assumed poison count
    `expected_frac` (see `flag_by_scores`) -- a convention calibrated on vision-scale poison rates
    (Tran et al. use ~1-3%; this project's vision control uses 31%). At low poison rates (<1%) on a
    large heterogeneous tabular class, that fixed budget starves even when the score distribution is
    near-perfectly separable (AUC ~0.99): the budget is smaller than the number of clean points that
    outscore it. This threshold instead flags points whose score is a statistical outlier relative to
    the bulk of the (assumed mostly-clean) distribution, independent of any assumed poison count.

    Uses the 0.6745-scaled MAD (median absolute deviation), the same normal-consistency constant this
    project's Neural Cleanse binary rule already uses for its mask-ratio anomaly index
    (`src/detectors/neural_cleanse.py`): z = 0.6745 * (score - median) / MAD. Note this scaling is
    exactly calibrated only under a normal bulk distribution -- `spectral_scores` are sums of squared
    projections (chi-squared-like, right-skewed), so z_thresh is a robust, empirically-tuned cutoff on
    this specific score shape, not a calibrated tail probability transferable to a differently-shaped
    score distribution (e.g. a different `n_components` or a different detector's scores). Falls back
    to a plain std-based z-score when MAD == 0 (more than half the scores are identical, degenerate
    for MAD).
    """
    scores = np.asarray(scores, dtype=float)
    med = np.median(scores)
    mad = np.median(np.abs(scores - med))
    if mad > 0:
        z = 0.6745 * (scores - med) / mad
    else:
        std = scores.std()
        z = np.zeros_like(scores) if std == 0 else (scores - med) / std
    return z > z_thresh
