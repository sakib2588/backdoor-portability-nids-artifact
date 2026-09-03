import numpy as np
import pytest

from src.detectors import poison_recall
from src.detectors.activation_clustering import detect


def test_ac_separates_smaller_poison_cluster():
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 1, size=(180, 20))
    poison = rng.normal(0, 1, size=(20, 20)) + 10.0
    feats = np.vstack([clean, poison])
    is_poison = np.array([False] * 180 + [True] * 20)

    poison_mask, sil = detect(feats, n_components=10, seed=0)
    assert poison_recall(poison_mask, is_poison) >= 0.9
    assert sil > 0.1


from src.detectors.activation_clustering import cluster_and_reduce, flag_by_relative_size


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    clean = rng.normal(0, 1, size=(180, 20))
    poison = rng.normal(0, 1, size=(20, 20)) + 10.0
    return np.vstack([clean, poison]), np.array([False] * 180 + [True] * 20)


def test_ica_reduction_recovers_the_planted_cluster():
    feats, is_poison = _blobs()
    labels, reduced, sil = cluster_and_reduce(feats, n_components=10, seed=0, reduction="ica")
    assert reduced.shape == (200, 10)
    assert poison_recall(flag_by_relative_size(labels), is_poison) >= 0.9


def test_pca_stays_the_default_and_is_unchanged():
    feats, _ = _blobs()
    a = cluster_and_reduce(feats, n_components=10, seed=0)
    b = cluster_and_reduce(feats, n_components=10, seed=0, reduction="pca")
    assert (a[0] == b[0]).all()


def test_unknown_reduction_is_rejected():
    feats, _ = _blobs()
    with pytest.raises(ValueError):
        cluster_and_reduce(feats, reduction="umap")
