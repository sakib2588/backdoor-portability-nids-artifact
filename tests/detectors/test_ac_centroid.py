"""Tests for the clean-reference centroid repair of Activation Clustering (plan Task 7B).

These tests prove the repair is what it claims to be BEFORE any CTU number is interpreted:
clean-reference fitted, clean-calibrated, and non-redundant against Spectral. The independence gate
matters most -- the previous AC repair attempt was withdrawn because it correlated 0.9967 with
Spectral (notes/20260716-decision-ac-proxy-not-independent.md).
"""
from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.detectors.activation_clustering import (
    CleanReferenceCentroids,
    assert_trusted_clean_reference,
    classify_centroid_independence,
    detect,
    fit_clean_reference_centroids,
    flag_by_clean_reference,
    nearest_clean_centroid_scores,
)
from src.poison import carve_clean_reference


@pytest.fixture
def blobs():
    """Two clean clusters plus a distant poison blob, in 20-D."""
    rng = np.random.default_rng(0)
    clean_a = rng.normal(0.0, 0.5, size=(300, 20))
    clean_b = rng.normal(6.0, 0.5, size=(300, 20))
    poison = rng.normal(-14.0, 0.4, size=(40, 20))
    return clean_a, clean_b, poison


# --- trusted-clean precondition -------------------------------------------------------------------

def test_centroid_fit_rejects_poisoned_reference_indices():
    with pytest.raises(ValueError, match="trusted clean"):
        assert_trusted_clean_reference(reference_indices=np.array([1]), poison_indices=np.array([1]))


def test_disjoint_reference_and_poison_indices_are_accepted():
    assert_trusted_clean_reference(np.array([1, 2, 3]), np.array([7, 8]))  # must not raise


def test_carved_reference_is_disjoint_from_the_poison_draw():
    y = np.zeros(60_000, dtype=int)
    y[:1_000] = 1  # ~2% botnet, rest benign (target class 0)
    reference, calibration, suspect = carve_clean_reference(
        y, target=0, seed=42, n_reference=500, n_calibration=500)
    assert len(reference) == 500 and len(calibration) == 500
    assert np.intersect1d(reference, calibration).size == 0
    assert np.intersect1d(reference, suspect).size == 0
    assert np.intersect1d(calibration, suspect).size == 0
    # the three parts partition the benign pool exactly
    assert len(reference) + len(calibration) + len(suspect) == int((y == 0).sum())
    # and no botnet row leaked into any part
    assert set(np.concatenate([reference, calibration, suspect])).isdisjoint(set(np.where(y == 1)[0]))


def test_carve_refuses_to_consume_the_whole_benign_pool():
    y = np.zeros(100, dtype=int)
    with pytest.raises(ValueError, match="suspect set"):
        carve_clean_reference(y, target=0, seed=42, n_reference=60, n_calibration=40)


# --- clean calibration of the cutoff --------------------------------------------------------------

def test_centroid_threshold_comes_from_clean_calibration_distances(blobs):
    clean_a, clean_b, _ = blobs
    reference_feats = np.vstack([clean_a[:200], clean_b[:200]])
    calibration_clean_feats = np.vstack([clean_a[200:], clean_b[200:]])
    reference = fit_clean_reference_centroids(
        reference_feats, calibration_clean_feats, n_components=2, seed=42)
    assert reference.distance_threshold == pytest.approx(
        np.quantile(nearest_clean_centroid_scores(calibration_clean_feats, reference), 0.99))


def test_threshold_never_sees_a_poison_label(blobs):
    """Adding poison to the SUSPECT set must not move the cutoff."""
    clean_a, clean_b, poison = blobs
    reference_feats = np.vstack([clean_a[:200], clean_b[:200]])
    calibration_clean_feats = np.vstack([clean_a[200:], clean_b[200:]])
    ref = fit_clean_reference_centroids(reference_feats, calibration_clean_feats, n_components=2, seed=42)
    ref_again = fit_clean_reference_centroids(reference_feats, calibration_clean_feats, n_components=2, seed=42)
    assert ref.distance_threshold == pytest.approx(ref_again.distance_threshold)
    # scoring poison does not mutate the fitted object
    _ = nearest_clean_centroid_scores(poison, ref)
    assert ref.distance_threshold == pytest.approx(ref_again.distance_threshold)


def test_fit_is_deterministic_for_a_fixed_seed(blobs):
    clean_a, clean_b, _ = blobs
    ref_feats, calib = np.vstack([clean_a[:200], clean_b[:200]]), np.vstack([clean_a[200:], clean_b[200:]])
    a = fit_clean_reference_centroids(ref_feats, calib, n_components=2, seed=7)
    b = fit_clean_reference_centroids(ref_feats, calib, n_components=2, seed=7)
    np.testing.assert_allclose(
        np.sort(a.centroids, axis=0), np.sort(b.centroids, axis=0))
    assert a.distance_threshold == pytest.approx(b.distance_threshold)


# --- the score behaves like a detector on a case where it should work -----------------------------

def test_distant_poison_scores_above_the_clean_cutoff(blobs):
    """Positive-control shape: a far-off poison blob must be flagged at low clean FPR.

    This is the synthetic sanity check only. It does NOT stand in for the real MNIST patch and CTU
    loud-trigger controls the plan requires before any low-rate claim.
    """
    clean_a, clean_b, poison = blobs
    ref_feats = np.vstack([clean_a[:200], clean_b[:200]])
    calib = np.vstack([clean_a[200:], clean_b[200:]])
    ref = fit_clean_reference_centroids(ref_feats, calib, n_components=2, seed=42)

    assert flag_by_clean_reference(poison, ref).mean() == pytest.approx(1.0)
    assert flag_by_clean_reference(calib, ref).mean() <= 0.02  # ~99th percentile by construction


def test_scores_are_nonnegative_and_finite(blobs):
    clean_a, clean_b, poison = blobs
    ref_feats, calib = np.vstack([clean_a[:200], clean_b[:200]]), np.vstack([clean_a[200:], clean_b[200:]])
    ref = fit_clean_reference_centroids(ref_feats, calib, n_components=2, seed=42)
    scores = nearest_clean_centroid_scores(np.vstack([calib, poison]), ref)
    assert np.all(np.isfinite(scores)) and np.all(scores >= 0.0)


# --- independence gate against Spectral -----------------------------------------------------------

def test_redundancy_gate_rejects_spectral_equivalent_score():
    assert classify_centroid_independence(spearman_rho=0.9967) == "spectral_redundant"


def test_redundancy_gate_is_sign_blind_and_uses_the_preregistered_cutoff():
    assert classify_centroid_independence(-0.9967) == "spectral_redundant"
    assert classify_centroid_independence(0.90) == "spectral_redundant"     # boundary is inclusive
    assert classify_centroid_independence(0.8999) == "independent"
    assert classify_centroid_independence(float("nan")) == "undetermined"


def test_redundancy_gate_default_matches_config():
    # the gate constant and the function default must not drift apart
    assert classify_centroid_independence(config.AC_SPECTRAL_INDEPENDENCE_MAX_RHO) == "spectral_redundant"


# --- canonical AC is untouched --------------------------------------------------------------------

def test_canonical_detect_still_returns_a_mask_and_silhouette(blobs):
    """The repair must not perturb Chen et al.'s rule -- every committed AC number comes from it."""
    clean_a, clean_b, poison = blobs
    feats = np.vstack([clean_a, clean_b, poison])
    mask, sil = detect(feats, n_components=10, seed=42)
    assert mask.dtype == np.bool_ and mask.shape == (len(feats),)
    assert -1.0 <= sil <= 1.0
    assert mask.sum() <= (~mask).sum()  # it flags the SMALLER cluster


def test_canonical_detect_exposes_no_score_attribute():
    import src.detectors.activation_clustering as ac
    # the deleted proxy must not have crept back in under its old name
    assert not hasattr(ac, "cluster_distance_scores")


def test_silhouette_degrades_to_none_when_the_subsample_misses_the_minority():
    """A 10k subsample of ~120k rows can contain ZERO points from a ~30-point minority cluster.

    sklearn then raises "Number of labels is 1". The silhouette is a DIAGNOSTIC, never the decision
    metric -- the clustering and the returned mask are computed on the full set -- so it must degrade
    to None rather than abort a multi-hour sweep. Regression for the crash that killed the first 7B
    evasion-grid launch.
    """
    rng = np.random.default_rng(0)
    bulk = rng.normal(0.0, 1.0, size=(40_000, 12))
    tiny = rng.normal(60.0, 0.1, size=(15, 12))     # 15 extreme outliers among 40k
    feats = np.vstack([bulk, tiny])

    mask, sil = detect(feats, n_components=10, seed=0, silhouette_sample=2_000)
    assert mask.sum() <= (~mask).sum()              # still flags the smaller cluster
    assert sil is None or isinstance(sil, float)    # never raises


def test_silhouette_is_still_computed_when_both_clusters_are_sampled(blobs):
    """The degradation must not fire on healthy splits -- committed numbers must not move."""
    clean_a, clean_b, poison = blobs
    feats = np.vstack([clean_a, clean_b, poison])
    _, sil = detect(feats, n_components=10, seed=42, silhouette_sample=500)
    assert isinstance(sil, float) and -1.0 <= sil <= 1.0
