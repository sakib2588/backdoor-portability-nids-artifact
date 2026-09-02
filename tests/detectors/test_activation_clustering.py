import numpy as np

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
