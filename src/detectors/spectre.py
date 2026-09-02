"""SPECTRE (Hayase et al., ICML 2021): robust-covariance whitening plus a QUE score.

Spectral Signatures scores a sample by its squared projection onto the top-k right singular vectors
of the raw centred representation matrix (`src/detectors/spectral.py`). SPECTRE's argument is that
this is the wrong geometry: natural intra-class variance is anisotropic, so a large projection onto
v1 mostly measures which direction the *clean* class happens to spread along, not which samples are
poisoned. SPECTRE first estimates the clean distribution's covariance robustly, whitens by it so the
clean bulk becomes roughly isotropic, and only then scores the residual anisotropy that poison
introduces, with a quantum-entropy (QUE) score that upweights the leading whitened directions.

What this module changes relative to `spectral_scores` is the SCORE ONLY. It deliberately returns a
per-sample score array with the same shape and orientation (higher = more poison-like), so both
decision rules in `src/detectors/spectral.py` apply to it unchanged:

    scores = spectre_scores(feats)
    flag_by_scores(scores, expected_frac=...)     # Tran et al.'s fixed removal budget
    flag_by_mad_threshold(scores, z_thresh=...)   # this project's budget-free rule

That separation is the point of the experiment. SPECTRE, like Spectral Signatures, still sizes its
removal from an assumed poison count, so running the stronger score under BOTH rules separates "the
score was too weak" from "the removal budget was too small" as explanations for the miss.

Faithfulness, stated plainly, in the same spirit as this project's PCA-for-ICA substitution in
Activation Clustering and its binary mask-ratio substitution in Neural Cleanse:

  - FAITHFUL: dimensionality reduction to a top-k subspace before estimation, robust estimation of
    the clean mean and covariance, whitening by that covariance, and the QUE score
    tau_i = x~_i' Q x~_i with Q = expm(alpha * (S - I) / ||S - I||_2) over the whitened covariance S.
  - SUBSTITUTED: Hayase et al. estimate the robust covariance with a filtering estimator carrying
    dimension-dependent guarantees. This module uses deterministic iterative trimming instead --
    estimate, rank by Mahalanobis distance, drop the top `trim_frac`, re-estimate, repeat `n_iter`
    times. It is a standard robust estimator, it is deterministic (this project's determinism
    self-checks re-score every committed run and would catch a stochastic estimator), and it needs no
    dimension-dependent tuning. It is NOT the estimator in the paper, and any SPECTRE result from this
    module must be read against that substitution.

No randomness anywhere in this module: same input gives the same scores.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

# QUE's temperature. alpha=4.0 is the value Hayase et al. report using throughout.
DEFAULT_ALPHA = 4.0
# Subspace dimension before estimation. The penultimate layer this project inspects is 128-wide, so
# 32 keeps the covariance well-conditioned at any class size the sweep produces.
DEFAULT_COMPONENTS = 32
# Fraction dropped per trimming round when no poison-rate hint is supplied.
DEFAULT_TRIM_FRAC = 0.05
DEFAULT_N_ITER = 3
# Ridge added to the covariance diagonal before inversion, scaled by its own trace, so a rank-deficient
# or near-singular estimate cannot produce an infinite whitened coordinate.
RIDGE = 1e-6


def _project_topk(feats: np.ndarray, n_components: int) -> np.ndarray:
    """Centre and project onto the top-k right singular vectors. Matches `spectral_scores`'s
    centring convention so the two detectors see the same starting geometry."""
    centered = feats - feats.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = max(1, min(n_components, vt.shape[0]))
    return centered @ vt[:k].T


def _robust_mean_cov(x: np.ndarray, trim_frac: float, n_iter: int):
    """Deterministic iterative-trimming estimate of the clean mean and covariance.

    Each round estimates mean/covariance on the currently-kept points, ranks every kept point by
    Mahalanobis distance under that estimate, and drops the furthest `trim_frac`. Poison, being the
    anisotropic minority, is what leaves first, so the estimate converges toward the clean bulk.
    """
    n = x.shape[0]
    keep = np.ones(n, dtype=bool)
    mu = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    for _ in range(max(0, n_iter)):
        sub = x[keep]
        if sub.shape[0] <= x.shape[1] + 1:
            break                      # too few points left to estimate a covariance
        mu = sub.mean(axis=0)
        cov = np.cov(sub, rowvar=False)
        cov = np.atleast_2d(cov)
        reg = cov + RIDGE * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        centred = x - mu
        try:
            d = np.einsum("ij,jk,ik->i", centred, np.linalg.inv(reg), centred)
        except np.linalg.LinAlgError:
            break
        n_drop = int(np.floor(trim_frac * n))
        if n_drop < 1:
            break
        # Drop the furthest `n_drop` of the points still kept, deterministically.
        kept_idx = np.flatnonzero(keep)
        order = kept_idx[np.argsort(d[kept_idx], kind="stable")]
        keep[order[-n_drop:]] = False
    return mu, np.atleast_2d(cov)


def _inv_sqrt(cov: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root via eigendecomposition, ridge-regularised."""
    cov = 0.5 * (cov + cov.T)                     # enforce exact symmetry before eigh
    ridge = RIDGE * max(np.trace(cov) / cov.shape[0], 1e-12)
    w, v = np.linalg.eigh(cov + ridge * np.eye(cov.shape[0]))
    w = np.clip(w, ridge, None)
    return v @ np.diag(w ** -0.5) @ v.T


def spectre_scores(
    feats: np.ndarray,
    n_components: int = DEFAULT_COMPONENTS,
    alpha: float = DEFAULT_ALPHA,
    trim_frac: float | None = None,
    n_iter: int = DEFAULT_N_ITER,
    expected_frac: float | None = None,
) -> np.ndarray:
    """Per-sample SPECTRE QUE score. Higher means more poison-like, matching `spectral_scores`.

    `expected_frac`, when given, sets the per-round trim to 1.5x the assumed poison fraction, which
    is the same oracle this project grants Spectral Signatures' fixed budget. It affects only the
    robust ESTIMATE, never the decision rule; the caller still chooses between `flag_by_scores` and
    `flag_by_mad_threshold`.
    """
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim != 2 or feats.shape[0] < 2:
        return np.zeros(feats.shape[0] if feats.ndim else 0, dtype=float)

    if trim_frac is None:
        trim_frac = (min(0.45, 1.5 * float(expected_frac))
                     if expected_frac is not None and expected_frac > 0
                     else DEFAULT_TRIM_FRAC)

    x = _project_topk(feats, n_components)
    mu, cov = _robust_mean_cov(x, trim_frac=trim_frac, n_iter=n_iter)
    whitened = (x - mu) @ _inv_sqrt(cov)

    # QUE: weight directions by how far the whitened covariance still departs from isotropy.
    s = np.cov(whitened, rowvar=False)
    s = np.atleast_2d(0.5 * (s + s.T))
    delta = s - np.eye(s.shape[0])
    nrm = np.linalg.norm(delta, ord=2)
    if not np.isfinite(nrm) or nrm <= 0:
        # Already isotropic: QUE degenerates to the plain whitened norm.
        return (whitened ** 2).sum(axis=1)
    q = expm(alpha * delta / nrm)
    tr = np.trace(q)
    scores = np.einsum("ij,jk,ik->i", whitened, q, whitened)
    return scores / tr if np.isfinite(tr) and tr > 0 else scores
