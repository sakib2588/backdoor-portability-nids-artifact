"""Activation Clustering (Chen et al., 2018).

Reduce the class-conditioned activations, split into two clusters. When the class
is poisoned, one cluster is the (smaller) poisoned sub-population; a high 2-cluster
silhouette indicates the split is real.

Reduction is PCA (deterministic SVD), matching the original paper's primary recipe
and IBM ART's `ActivationDefence` reference implementation (reduce="PCA", nb_dims=10,
cluster_analysis="smaller"). An earlier version of this detector used FastICA, which
is a stochastic, non-convex fixed-point solver: it produced a different reduced
subspace per run (confirmed by a ConvergenceWarning even on trivially-separable
20-D synthetic test data) and that instability -- not the clustering step itself --
was the source of the 0.47-1.00 per-seed poison-recall spread on the MNIST vision
control (see notes/20260714-bug-ac-fastica-instability.md).

`detect` -- canonical AC -- deliberately exposes NO ranked poison score. AC is a
hard 2-cluster assignment, which is why results/detectors.json reports `auc: null`
for AC. A cluster-distance proxy score was added and then deleted: on real activations it
correlates with Spectral Signatures' own score at Spearman rho=0.9967 (Pearson
r=0.99996 against sqrt(spectral)), because both are distances in the PCA/SVD
subspace of the same centred activation matrix, so any result computed on it
restates Spectral under an AC label rather than measuring AC. Its claim that
thresholding at the minority size reproduces `detect`'s mask also proved false on
real data (it held only on well-separated synthetic blobs). Relatedly, AC has no
removal budget to replace -- it flags the smaller cluster -- so a budget-free
threshold substitution has no premise here. See
notes/20260716-decision-ac-proxy-not-independent.md and scripts/11_ac_mechanism.py.

The clean-reference centroid repair at the foot of this module (plan Task 7B) DOES
expose a ranked score, but it is an AC-derived extension fitted on trusted-clean
activations, not Chen et al.'s rule, and it carries its own pre-registered
independence gate against Spectral. `detect`'s behaviour is untouched: every
committed AC number in results/ reproduces from it byte-for-byte.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from src import config


def cluster_and_reduce(
    feats: np.ndarray, n_components: int = 10, seed: int = 0,
    silhouette_sample: int | None = None,
    n_clusters: int = 2, algorithm: str = "kmeans",
    return_estimator: bool = False,
    reduction: str = "pca",
):
    """PCA-reduce then cluster. Returns (labels, reduced, silhouette).

    `reduction="ica"` uses FastICA instead, the original paper's recipe; default `"pca"` leaves
    every existing caller and every committed number unchanged.

    Extracted verbatim from `detect`, which used to own this block inline. The AC repair grid needs
    to reduce+cluster ONCE per (cell, k, algorithm) and then apply several different decision rules
    against that same clustering; without this split it would either re-run PCA and k-means per rule
    or grow a second, drifting copy of them.

    Defaults `n_clusters=2, algorithm="kmeans"` reproduce every committed AC number byte-for-byte.

    `return_estimator` appends the FITTED clusterer, making the return
    (labels, reduced, silhouette, estimator). It defaults to False so every existing caller keeps the
    three-tuple it unpacks today. scripts/66_ac_continuous_score_pilot.py needs it because a
    within-component depth score must come from the same fitted mixture that produced these labels;
    re-fitting a second GMM on the side would be a different object that could silently drift from
    the one the labels came from.
    """
    if algorithm not in ("kmeans", "gmm"):
        raise ValueError(f"algorithm must be 'kmeans' or 'gmm', got {algorithm!r}")
    feats = np.asarray(feats, dtype=np.float64)
    if reduction not in ("pca", "ica"):
        raise ValueError(f"reduction must be 'pca' or 'ica', got {reduction!r}")
    n_components = min(n_components, feats.shape[1])
    if reduction == "pca":
        reduced = PCA(n_components=n_components, random_state=seed).fit_transform(feats)
    else:
        # Chen et al.'s primary recipe. Opt-in only: every committed AC number was produced with
        # PCA and the default keeps it that way. whiten="unit-variance" is sklearn's current
        # default spelled out so a future default change cannot move a number silently.
        reduced = FastICA(n_components=n_components, random_state=seed,
                          whiten="unit-variance", max_iter=1000, tol=1e-3).fit_transform(feats)
    if algorithm == "kmeans":
        estimator = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        labels = estimator.fit_predict(reduced)
    else:
        estimator = GaussianMixture(n_components=n_clusters, random_state=seed).fit(reduced)
        labels = estimator.predict(reduced)

    # Silhouette is an O(n^2) DIAGNOSTIC (reported, not the decision metric). On the full ~140k-row
    # benign NIDS training subset it dominates wall-clock when the detector runs on every grid cell,
    # so `silhouette_sample` caps the points used for the silhouette estimate ONLY. The clustering,
    # the smaller-cluster choice, and the returned poison_mask (the recall metric) are always on the
    # full reduced set, so the scientific result is unchanged -- only the silhouette becomes a seeded
    # subsample estimate. Default None preserves the exact M1/M2 vision-control behaviour.
    if silhouette_sample is not None and reduced.shape[0] > silhouette_sample:
        try:
            sil = float(silhouette_score(reduced, labels, sample_size=silhouette_sample,
                                         random_state=seed))
        except ValueError:
            # The random subsample drew ZERO points from the minority cluster, so the silhouette is
            # undefined on it (sklearn needs >= 2 labels present). This happens exactly when the
            # minority is a handful of natural outliers among ~120k rows -- i.e. in the AC failure mode
            # this project already documents. The silhouette is a DIAGNOSTIC here, never the decision
            # metric: the clustering, the smaller-cluster choice and `poison_mask` are all computed on
            # the full set above and are unaffected. So degrade to None rather than abort a multi-hour
            # sweep over an unreportable diagnostic. Cells that previously computed a silhouette are
            # untouched, so no committed number moves.
            sil = None
    else:
        sil = float(silhouette_score(reduced, labels))
    if return_estimator:
        return labels, reduced, sil, estimator
    return labels, reduced, sil


def flag_by_relative_size(labels: np.ndarray) -> np.ndarray:
    """Chen et al.'s rule: the smallest cluster is the poisoned one.

    Uses argmin over the full bincount rather than comparing cluster 0 against cluster 1. The old
    two-label form (`0 if (labels==0).sum() <= (labels==1).sum() else 1`) silently ignored clusters
    2..k-1, so any k>2 result would have been scored against the wrong cluster while the k=2 pin
    kept passing.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    return labels == int(np.argmin(np.where(counts > 0, counts, counts.max() + 1)))


def flag_by_silhouette_gate(
    labels: np.ndarray, silhouette: float | None,
    threshold: float = config.AC_SILHOUETTE_GATE_THRESHOLD,
) -> np.ndarray:
    """Relative-size flagging, suppressed entirely when the split does not look real.

    Chen et al. treat a low silhouette as evidence the two-cluster structure is not genuine. This
    repo has already measured the silhouette ANTI-correlating with recall (Spearman -0.5549), so the
    gate is expected to misfire here -- it is pre-registered as a candidate to be TESTED, not assumed
    to help. A `None` silhouette (the subsample drew no minority points) counts as not passed: a gate
    that cannot read its own input has demonstrated nothing.
    """
    labels = np.asarray(labels)
    if silhouette is None or not np.isfinite(silhouette) or silhouette < threshold:
        return np.zeros(labels.shape[0], dtype=bool)
    return flag_by_relative_size(labels)


def detect(
    feats: np.ndarray, n_components: int = 10, seed: int = 0,
    silhouette_sample: int | None = None,
    n_clusters: int = 2, algorithm: str = "kmeans",
    reduction: str = "pca",
) -> tuple[np.ndarray, float]:
    """Canonical AC. Behaviour-preserving refactor over `cluster_and_reduce` +
    `flag_by_relative_size`; the new keyword-defaulted params leave all 16 existing call sites
    untouched and the k=2 kmeans path identical."""
    labels, _, sil = cluster_and_reduce(feats, n_components=n_components, seed=seed,
                                        silhouette_sample=silhouette_sample,
                                        n_clusters=n_clusters, algorithm=algorithm,
                                        reduction=reduction)
    return flag_by_relative_size(labels), sil


# --------------------------------------------------------------------------------------------------
# Exclusionary reclassification and the gap statistic (plan Task AC-3). CHEN ET AL.'S OWN RULES,
# never before run in this repo.
# --------------------------------------------------------------------------------------------------
# Both take already-computed inputs -- predictions, a reduced matrix -- rather than a model, so this
# module stays free of a torch/src.models import. The orchestration script owns retraining and
# prediction, mirroring how neural_cleanse.py's inversion functions take an instantiated model and
# never train one.


def exclusionary_reclassification_score(
    predictions: np.ndarray, own_label: int, other_label: int,
) -> dict:
    """Chen et al. Section 6.3. Exclude a cluster, retrain without it, reclassify the excluded rows.

    `l` counts excluded rows the retrained model sends back to their ORIGINAL label; `p` counts those
    it sends to the other class. A legitimate cluster reclassifies to its own label (high ratio); a
    poisoned one reclassifies to the source class it was really drawn from (low ratio).

    `max(p, 1)` is the zero-division guard and it is load-bearing, not cosmetic: p=0 is the ORDINARY
    case here, since the excluded minority is usually natural outliers that all come back as
    themselves. An unguarded l/p would emit inf and break strict JSON serialisation, matching the
    `_finite_or_none` discipline already used across this repo's result writers.
    """
    predictions = np.asarray(predictions)
    l = int((predictions == own_label).sum())
    p = int((predictions == other_label).sum())
    return dict(l=l, p=p, n_excluded=int(predictions.size), ratio=float(l / max(p, 1)))


def flag_by_exclusionary_reclassification(
    ratio: float, threshold: float = config.AC_EXRE_T_DEFAULT,
) -> bool:
    """True when the cluster looks POISONED, i.e. the ratio falls below T. Chen et al. publish T=1;
    it is retained as the uncalibrated comparison exactly as NC retains tau=2.0, while the AC track
    additionally calibrates T from a clean-model null."""
    if ratio is None or not np.isfinite(ratio):
        return False
    return bool(ratio < threshold)


def _within_cluster_dispersion(points: np.ndarray, labels: np.ndarray) -> float:
    """Pooled within-cluster sum of squares about each cluster centroid."""
    total = 0.0
    for lab in np.unique(labels):
        member = points[labels == lab]
        if member.shape[0] == 0:
            continue
        total += float(((member - member.mean(axis=0)) ** 2).sum())
    return total


def gap_statistic_k1_vs_k2(reduced: np.ndarray, seed: int, b: int = config.AC_GAP_STATISTIC_B) -> dict:
    """Tibshirani, Walther and Hastie (2001), restricted to the k=1-vs-k=2 question.

    Uniform reference draws inside the data's bounding box, `b` resamples, comparing log within-
    cluster dispersion against its reference expectation at k=1 and k=2.

    Scoped narrowly ON PURPOSE: Chen et al. use the gap statistic only for this question and report a
    NEGATIVE result for it. This is confirmatory, not a k-selection procedure, and it is pre-registered
    with the expectation that it fails.

    Dispersion is floored before the log. All-identical points give W=0 and an unguarded log(0) would
    put -inf into the artifact; the floor keeps both gaps finite and the verdict defined.
    """
    reduced = np.asarray(reduced, dtype=np.float64)
    rng = np.random.default_rng(seed)
    floor = 1e-12
    lo, hi = reduced.min(axis=0), reduced.max(axis=0)

    def _dispersions(points):
        w1 = _within_cluster_dispersion(points, np.zeros(points.shape[0], dtype=int))
        two = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(points)
        return w1, _within_cluster_dispersion(points, two)

    obs_w1, obs_w2 = _dispersions(reduced)
    ref_l1, ref_l2 = [], []
    for _ in range(max(1, b)):
        ref = rng.uniform(lo, hi, size=reduced.shape)
        r1, r2 = _dispersions(ref)
        ref_l1.append(np.log(max(r1, floor)))
        ref_l2.append(np.log(max(r2, floor)))

    gap_1 = float(np.mean(ref_l1) - np.log(max(obs_w1, floor)))
    gap_2 = float(np.mean(ref_l2) - np.log(max(obs_w2, floor)))
    return dict(gap_1=gap_1, gap_2=gap_2, b=int(max(1, b)),
                observed_w1=float(obs_w1), observed_w2=float(obs_w2),
                prefers_k2=bool(gap_2 > gap_1))


# --------------------------------------------------------------------------------------------------
# Clean-reference centroid repair (plan Task 7B). AC-DERIVED EXTENSION, NOT Chen et al.'s rule.
# --------------------------------------------------------------------------------------------------
# `detect` above fails inside the evasion window because 2-means splits off a few dozen natural
# outliers while the poison stays in the majority cluster (18 of 20 evasion rows, minority purity
# <=0.038; results/ac_mechanism.json). The poison never becomes the dominant secondary structure, so
# there is no ranking for any threshold to rescue -- which is why the MAD substitution that repairs
# Spectral has no premise here.
#
# This repair attacks that mechanism by changing WHERE the geometry comes from. PCA and the k-means
# centroids are fitted on a defender-held, trusted-clean reference subset carved out of the benign
# training rows BEFORE poisoning, so poison cannot define the coordinate system or the centroids. A
# suspect activation is then scored by its distance to the nearest clean-reference centroid, before
# any poisoned-data k-means split happens.
#
# The reference activations are still produced BY THE BACKDOORED VICTIM. That is deliberate: it keeps
# the activation coordinate system identical to the one the poison lives in, while denying poison any
# say in the centroids.
#
# Why this is not the deleted proxy: the rejected `cluster_distance_scores` was a distance in the
# PCA/SVD subspace of the same CONTAMINATED centred activation matrix that Spectral decomposes, so it
# correlated with Spectral at Spearman rho=0.9967 and merely restated Spectral under an AC label (see
# notes/20260716-decision-ac-proxy-not-independent.md). Here the subspace and the centring come from
# clean reference rows only, so the two scores are not algebraically tied. That is a hypothesis, not a
# guarantee: `classify_centroid_independence` measures it per cell and the pre-registered gate
# (config.AC_SPECTRAL_INDEPENDENCE_MAX_RHO) rejects the arm as Spectral-redundant at rho >= 0.90,
# however good its recall looks.


@dataclass(frozen=True)
class CleanReferenceCentroids:
    """A clean-fitted activation geometry plus its clean-calibrated decision cutoff."""
    pca: PCA
    centroids: np.ndarray
    distance_threshold: float


def assert_trusted_clean_reference(
    reference_indices: np.ndarray, poison_indices: np.ndarray
) -> None:
    """Refuse a reference subset that any poisoned row can reach.

    The whole premise of the repair is that poison did not define the geometry, so this is a hard
    precondition rather than a warning: a single poisoned reference row silently converts the
    detector back into a contaminated-geometry method.
    """
    reference_indices = np.asarray(reference_indices)
    poison_indices = np.asarray(poison_indices)
    overlap = np.intersect1d(reference_indices, poison_indices)
    if overlap.size:
        raise ValueError(
            f"reference subset is not trusted clean: {overlap.size} index/indices also poisoned "
            f"(e.g. {overlap[:5].tolist()})"
        )


def fit_clean_reference_centroids(
    reference_feats: np.ndarray,
    calibration_clean_feats: np.ndarray,
    n_components: int = 10,
    seed: int = 0,
    clean_quantile: float = 0.99,
    n_clusters: int = 2,
) -> CleanReferenceCentroids:
    """Fit PCA + k-means centroids on trusted-clean activations, then calibrate the cutoff on clean.

    `n_components=10` and `n_clusters=2` match canonical AC's reduction and cluster count, so the
    repair differs from `detect` in WHAT the geometry is fitted on, not in the recipe's shape.

    The cutoff is the `clean_quantile` of nearest-centroid distances over a held-back clean
    calibration set. It never sees a poison label, so it cannot be tuned to the answer.
    """
    reference_feats = np.asarray(reference_feats, dtype=np.float64)
    calibration_clean_feats = np.asarray(calibration_clean_feats, dtype=np.float64)
    k = min(n_components, reference_feats.shape[1], reference_feats.shape[0])
    pca = PCA(n_components=k, random_state=seed).fit(reference_feats)
    reduced = pca.transform(reference_feats)
    centroids = KMeans(
        n_clusters=n_clusters, n_init=10, random_state=seed
    ).fit(reduced).cluster_centers_

    # Build a provisional object so the calibration scores use the exact scoring path callers use.
    provisional = CleanReferenceCentroids(pca=pca, centroids=centroids, distance_threshold=float("inf"))
    calib_scores = nearest_clean_centroid_scores(calibration_clean_feats, provisional)
    threshold = float(np.quantile(calib_scores, clean_quantile))
    return CleanReferenceCentroids(pca=pca, centroids=centroids, distance_threshold=threshold)


def nearest_clean_centroid_scores(
    feats: np.ndarray, reference: CleanReferenceCentroids,
) -> np.ndarray:
    """Squared distance from each activation to its nearest clean-reference centroid.

    Higher is more anomalous. Unlike `detect` this IS a ranked score, so an AUC is defined for it --
    but it is an AC-derived score, never to be reported as Chen et al.'s AC.
    """
    reduced = reference.pca.transform(np.asarray(feats, dtype=np.float64))
    return ((reduced[:, None, :] - reference.centroids[None, :, :]) ** 2).sum(axis=2).min(axis=1)


def flag_by_clean_reference(
    feats: np.ndarray, reference: CleanReferenceCentroids
) -> np.ndarray:
    """Flag activations whose nearest-clean-centroid distance exceeds the clean-calibrated cutoff."""
    return nearest_clean_centroid_scores(feats, reference) > reference.distance_threshold


def classify_centroid_independence(
    spearman_rho: float, max_rho: float = 0.90
) -> str:
    """Verdict on whether this score is a genuinely new detector or Spectral in disguise.

    `max_rho` mirrors config.AC_SPECTRAL_INDEPENDENCE_MAX_RHO (pre-registered). The deleted proxy sat
    at 0.9967. Returns "spectral_redundant" or "independent"; a redundant score is not an AC result
    regardless of the recall it achieves.
    """
    if not np.isfinite(spearman_rho):
        return "undetermined"
    return "spectral_redundant" if abs(float(spearman_rho)) >= max_rho else "independent"
