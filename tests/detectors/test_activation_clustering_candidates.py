"""Regression tests for the AC repair candidates (plan Task AC-1).

Deliberately a SEPARATE file from `test_activation_clustering.py` so the k=2 default pin there stays
undisturbed and easy to diff.

Each test below reproduces an actual fragile case this repo's own evidence predicts will misfire, not
a toy that would pass either way. Mirrors NC-1's discipline: the box-pinned SafeDivision test earned
its keep by exercising the real configuration, not a synthetic stand-in.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.detectors.activation_clustering import (
    cluster_and_reduce,
    exclusionary_reclassification_score,
    flag_by_relative_size,
    flag_by_silhouette_gate,
    gap_statistic_k1_vs_k2,
)


def _separated_blobs(sizes, dim=6, spread=0.05, seed=0):
    """Well-separated blobs, one per entry in `sizes`, far enough apart that any sane clusterer
    recovers them."""
    rng = np.random.default_rng(seed)
    parts = []
    for i, n in enumerate(sizes):
        centre = np.zeros(dim)
        centre[i % dim] = 12.0 * (i + 1)
        parts.append(rng.normal(centre, spread, size=(n, dim)))
    return np.vstack(parts)


# --- 1. minority selection must generalise past two labels ------------------------------------

def test_minority_selection_generalises_beyond_two_labels():
    """The bug this guards: `minority_label = 0 if (labels==0).sum() <= (labels==1).sum() else 1`
    only ever compares clusters 0 and 1. At k=3 it silently ignores cluster 2, so a threading of
    n_clusters that touches only the KMeans call would mis-flag here and the existing k=2 pin would
    not catch it."""
    labels = np.array([0] * 5 + [1] * 40 + [2] * 200)
    flagged = flag_by_relative_size(labels)
    assert flagged.sum() == 5
    assert flagged[:5].all()
    assert not flagged[5:].any()


def test_minority_selection_handles_minority_not_at_index_zero():
    """Same guard, with the smallest cluster in the middle -- a version keyed to label order rather
    than to size would pass the test above by luck and fail this one."""
    labels = np.array([0] * 200 + [1] * 5 + [2] * 40)
    flagged = flag_by_relative_size(labels)
    assert flagged.sum() == 5
    assert flagged[200:205].all()


# --- 2. GMM must actually return k distinct components ----------------------------------------

def test_gmm_returns_the_requested_number_of_labels():
    """Degenerate-component collapse is a real sklearn failure mode at small effective per-cluster
    sample sizes. If GMM silently returns 3 labels when asked for 5, every k>2 GMM result would be
    quietly mislabelled."""
    feats = _separated_blobs([60, 60, 60, 60, 60], seed=1)
    labels, reduced, sil = cluster_and_reduce(feats, n_components=5, seed=0,
                                              n_clusters=5, algorithm="gmm")
    assert len(np.unique(labels)) == 5
    assert reduced.shape[0] == feats.shape[0]


def test_kmeans_and_gmm_both_recover_well_separated_blobs():
    feats = _separated_blobs([40, 120], seed=2)
    for algorithm in config.AC_CLUSTER_ALGORITHMS:
        labels, _, _ = cluster_and_reduce(feats, n_components=4, seed=0,
                                          n_clusters=2, algorithm=algorithm)
        flagged = flag_by_relative_size(labels)
        assert flagged.sum() == 40, f"{algorithm} did not recover the size-40 minority"


def test_unknown_algorithm_is_rejected():
    with pytest.raises(ValueError, match="algorithm"):
        cluster_and_reduce(_separated_blobs([10, 10]), n_clusters=2, algorithm="dbscan")


# --- 3. ExRe must not divide by zero ------------------------------------------------------------

def test_exre_ratio_is_finite_when_nothing_reclassifies_to_the_other_class():
    """p = 0 is the ordinary case when the excluded cluster is natural outliers rather than poison:
    every excluded point comes back as its own label. An unguarded l/p yields inf and breaks strict
    JSON serialisation, matching the `_finite_or_none` discipline already used elsewhere."""
    predictions = np.array([1, 1, 1, 1, 1])       # all reclassified to their own label
    out = exclusionary_reclassification_score(predictions, own_label=1, other_label=0)
    assert out["p"] == 0
    assert out["l"] == 5
    assert np.isfinite(out["ratio"])


def test_exre_ratio_matches_the_published_definition():
    predictions = np.array([1, 1, 1, 0, 0])       # l = 3 own, p = 2 other
    out = exclusionary_reclassification_score(predictions, own_label=1, other_label=0)
    assert out["l"] == 3 and out["p"] == 2
    assert out["ratio"] == pytest.approx(1.5)


def test_exre_poisoned_cluster_signature_is_a_low_ratio():
    """Chen et al.: a poisoned cluster's excluded points come back as the SOURCE class, so l is small
    and the ratio falls below T. Pinning the direction so a sign flip cannot pass silently."""
    predictions = np.array([0, 0, 0, 0, 1])       # l = 1, p = 4
    out = exclusionary_reclassification_score(predictions, own_label=1, other_label=0)
    assert out["ratio"] < config.AC_EXRE_T_DEFAULT


# --- 4. the silhouette gate is a DECISION INVERSION, not a numeric edge case --------------------

def test_silhouette_gate_returns_all_clean_below_threshold():
    """This repo has already measured the silhouette ANTI-correlating with recall
    (AcSilhouetteRecallSpearman = -0.5549), so a gate that suppresses flags on low silhouette is
    predicted to misfire. It must be exercised before it touches real data: relative-size alone would
    have flagged the size-5 cluster here, and the gate must override that."""
    labels = np.array([0] * 5 + [1] * 95)
    assert flag_by_relative_size(labels).sum() == 5
    gated = flag_by_silhouette_gate(labels, silhouette=0.05,
                                    threshold=config.AC_SILHOUETTE_GATE_THRESHOLD)
    assert gated.sum() == 0


def test_silhouette_gate_passes_through_above_threshold():
    labels = np.array([0] * 5 + [1] * 95)
    gated = flag_by_silhouette_gate(labels, silhouette=0.30,
                                    threshold=config.AC_SILHOUETTE_GATE_THRESHOLD)
    assert np.array_equal(gated, flag_by_relative_size(labels))


def test_silhouette_gate_treats_none_as_not_passed():
    """`detect` degrades the silhouette to None when the subsample draws no minority points. A gate
    that cannot read its own input has not demonstrated anything and must not flag."""
    labels = np.array([0] * 5 + [1] * 95)
    assert flag_by_silhouette_gate(labels, silhouette=None).sum() == 0


# --- 5. gap statistic must survive zero dispersion ----------------------------------------------

def test_gap_statistic_handles_identical_points_without_dividing_by_zero():
    """All points identical gives zero within-cluster dispersion, so a naive log(W) is log(0). The
    function must return a defined verdict rather than propagate -inf or NaN into the artifact."""
    reduced = np.zeros((40, 3))
    out = gap_statistic_k1_vs_k2(reduced, seed=0, b=5)
    assert out["prefers_k2"] in (True, False)
    assert np.isfinite(out["gap_1"]) and np.isfinite(out["gap_2"])


def test_gap_statistic_prefers_two_clusters_on_clearly_bimodal_data():
    reduced = _separated_blobs([60, 60], dim=3, seed=3)
    out = gap_statistic_k1_vs_k2(reduced, seed=0, b=config.AC_GAP_STATISTIC_B)
    assert out["prefers_k2"] is True


# --- 6. the k=2 default must remain byte-for-byte identical -------------------------------------

def test_defaults_reproduce_canonical_two_means():
    """Every committed AC number comes from the k=2 kmeans path. `cluster_and_reduce`'s defaults must
    not move it."""
    feats = _separated_blobs([30, 90], seed=4)
    labels, _, _ = cluster_and_reduce(feats, n_components=10, seed=0)
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    expected_reduced = PCA(n_components=min(10, feats.shape[1]),
                           random_state=0).fit_transform(np.asarray(feats, dtype=np.float64))
    expected = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(expected_reduced)
    assert np.array_equal(labels, expected)
