"""Active-paths detector (Hoyheim et al., ICMCIS 2026): local slopes, Kernel PCA, density clustering.

PRE-REGISTERED in notes/20260905-prereg-active-paths-detector.md. Read that file before changing
any constant here. The parameters below were fixed before the first cell ran and are not swept.

WHY THIS ARM EXISTS. Every other detector in this paper is a vision port, so a null cannot separate
"vision-built detectors fail here" from "this attack defeats any detector". Hoyheim et al.'s method
is tabular-native and NIDS-native, and it is the only untried detector *family* from the literature
sweep. It is scored here against the same clean-label, constraint-projected trigger as the rest.

THE PIPELINE, as they describe it: local feature contributions, then Kernel PCA, then density-based
clustering, then a comparison of mean feature contributions between clusters.

  1. Slopes. `src.models.mlp_input_gradients` gives d f_target / d x per row. For a ReLU victim this
     gradient is fixed by which units are active at that row, so it is the row's active path
     expressed in input coordinates.
  2. Kernel PCA with an RBF kernel, fitted on a subsample and applied to every row.
  3. HDBSCAN on the reduced coordinates. Rows HDBSCAN cannot place keep label -1 and are treated as
     one additional cluster, following `density_mitigation.py`.
  4. Each cluster's mean slope vector is compared with the size-weighted global mean slope vector,
     and a row inherits its cluster's L2 distance.

Step 4 is the project's addition, and it is why this arm gets a ranking AUC at all. The method as
published emits a cluster verdict, not a per-row score, so without an inherited score it would have
to carry `auc=None` like Activation Clustering. The construction follows the precedent in
`density_mitigation.py`, where a row inherits its cluster's absorption rank. Scores are oriented
higher-means-more-poison-like, matching every other detector here, so both existing decision rules
in `spectral.py` apply unchanged.

DECLARED DEVIATIONS, all recorded in the pre-registration rather than discovered later:

  - They derive slopes from pre-activation gradients. We use the gradient of the target-class logit
    with respect to the input. For a ReLU network the two carry the same active-path information,
    and the logit gradient is one vector per row rather than a `feat_dim x D` Jacobian per row,
    which is what makes the method runnable at 111,666 rows.
  - Kernel PCA is O(n^2) in memory and cannot be fitted on the full target-class block, so it is
    fitted on a uniform subsample and used to transform the rest.
  - Their victim is a 121-feature network on AIT-IDSv2. Ours is this paper's MLP. We are testing
    their method against our attack, not reproducing their experiment.

The method needs a differentiable victim, so like Neural Cleanse it does not run on the LightGBM
victim. That is the same structural limit already established for tree ensembles, not a new one.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import KernelPCA

# Pre-registered, not swept. See the table in the pre-registration note.
DEFAULT_COMPONENTS = 10
DEFAULT_FIT_SUBSAMPLE = 10_000
DEFAULT_MIN_CLUSTER_SIZE = 100
DEFAULT_MIN_SAMPLES = 10
DEGENERACY_TOL = 1e-12

# Rows per call to KernelPCA.transform. Not a model parameter and not pre-registered: transform is
# row-independent, so any chunk size gives bit-identical output. It exists only because the one-shot
# call allocates an (n_rows x fit_subsample) float64 kernel -- 8.9 GB at 111,666 rows against a
# 10,000-row fit set, and rbf_kernel materializes a pairwise-distance array of that same size before
# exponentiating it, so the true peak is twice that and does not fit in 15 GB of RAM.
TRANSFORM_CHUNK = 4096


def slope_matrix(slopes: np.ndarray) -> dict:
    """Describe the slope matrix before any clustering touches it.

    The pre-registration names one way this method can fail before it starts: if the slopes are
    near-constant across rows, there is no structure to cluster and any split the pipeline returns
    would be a statement about the clustering, not about the victim. This reports that up front so
    a degenerate case is visible in the artifact rather than showing up as a confident number.
    """
    s = np.asarray(slopes, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"expected a 2-D slope matrix (n, D), got shape {s.shape}")
    spread = float(s.std(axis=0).max()) if s.size else 0.0
    return dict(n_rows=int(s.shape[0]), n_features=int(s.shape[1]),
                max_feature_std=spread, degenerate=bool(spread <= DEGENERACY_TOL))


def active_path_scores(
    slopes: np.ndarray,
    seed: int = 0,
    n_components: int = DEFAULT_COMPONENTS,
    fit_subsample: int = DEFAULT_FIT_SUBSAMPLE,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> np.ndarray:
    """Per-row score (higher = more poison-like) from a slope matrix of shape (n, D).

    Returns zeros when the pipeline cannot run: too few rows to cluster, or a degenerate slope
    matrix. Returning a flat score is the honest outcome there, because a constant input cannot
    license a non-constant ranking, and it keeps the caller's recall and AUC arithmetic defined.
    """
    s = np.asarray(slopes, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"expected a 2-D slope matrix (n, D), got shape {s.shape}")
    n = s.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if slope_matrix(s)["degenerate"] or n < max(min_cluster_size, n_components + 1):
        return np.zeros(n, dtype=np.float64)

    rng = np.random.default_rng(seed)
    k = min(fit_subsample, n)
    fit_idx = np.sort(rng.choice(n, size=k, replace=False)) if k < n else np.arange(n)

    kpca = KernelPCA(n_components=min(n_components, k - 1), kernel="rbf", random_state=seed)
    kpca.fit(s[fit_idx])
    reduced = np.concatenate(
        [kpca.transform(s[i:i + TRANSFORM_CHUNK]) for i in range(0, n, TRANSFORM_CHUNK)], axis=0)
    if not np.isfinite(reduced).all():
        return np.zeros(n, dtype=np.float64)

    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                     n_jobs=-1).fit_predict(reduced)

    # Rows HDBSCAN cannot place keep label -1 and are treated as one further cluster, following
    # density_mitigation.py. Dropping them would silently exclude the rows most likely to be poison.
    global_mean = s.mean(axis=0)
    scores = np.zeros(n, dtype=np.float64)
    for lab in np.unique(labels):
        mask = labels == lab
        scores[mask] = float(np.linalg.norm(s[mask].mean(axis=0) - global_mean))
    return scores
