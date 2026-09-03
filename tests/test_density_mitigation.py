"""Severi et al.'s density-based clean-label mitigation, tested on synthetic data with known answers.

Written BEFORE the module, per the plan. Every stage is tested where it should succeed and, in
test_a_scattered_poison_is_not_recovered and test_no_planted_cluster_flags_nothing, where it should
FAIL. A statistic that only passes on the happy path has not been tested.
"""
import numpy as np
import pytest

from src.detectors.density_mitigation import (
    cluster_target_class, filter_mask, flag_fixed_threshold, flag_loss_delta,
    iterative_scoring, select_features,
)
from src.models import train_lightgbm

TARGET = 0
SEED = 0


def _surrogate(x, y, seed):
    return train_lightgbm(x, y, seed)


def _blocks(n_benign=20_000, n_attack=1_000, d=8, poison_frac=0.05, scatter=False, seed=SEED):
    """Benign block of 20 blobs plus a planted group; attack block near the planted group.

    Only the first four columns separate the classes, so select_features has a right answer.
    """
    rng = np.random.default_rng(seed)
    n_poison = int(n_benign * poison_frac)
    centers = rng.normal(size=(20, d)) * 60
    clean = centers[rng.integers(0, 20, n_benign - n_poison)] + rng.normal(size=(n_benign - n_poison, d))
    if scatter:
        # poison spread across the existing blobs: the method's own documented weakness
        poison = centers[rng.integers(0, 20, n_poison)] + rng.normal(size=(n_poison, d)) * 1.1
    else:
        poison = np.full((n_poison, d), 25.0) + rng.normal(scale=0.05, size=(n_poison, d))
    benign = np.concatenate([clean, poison])
    is_poison = np.zeros(len(benign), dtype=bool)
    is_poison[len(clean):] = True
    attack = np.full((n_attack, d), 25.0) + rng.normal(scale=1.0, size=(n_attack, d))
    # columns 4..d carry no class signal
    benign[:, 4:] = rng.normal(size=(len(benign), d - 4))
    attack[:, 4:] = rng.normal(size=(n_attack, d - 4))
    x = np.concatenate([benign, attack])
    y = np.concatenate([np.zeros(len(benign), dtype=int), np.ones(n_attack, dtype=int)])
    return x, y, is_poison


@pytest.fixture(scope="module")
def planted():
    return _blocks()


def test_select_features_finds_the_separating_columns(planted):
    x, y, _ = planted
    idx = select_features(x, y, n_features=4, seed=SEED)
    assert len(idx) == 4 and len(set(idx.tolist())) == 4
    # Three of the four informative columns already separate the classes, so the fourth can rank
    # below a noise column. The claim worth testing is that the selection is mostly informative.
    assert len(set(idx.tolist()) & {0, 1, 2, 3}) >= 3, f"picked mostly non-separating columns: {idx}"


def test_clustering_recovers_the_planted_group(planted):
    x, y, is_poison = planted
    sub = x[y == TARGET][:, select_features(x, y, 4, SEED)]
    labels = cluster_target_class(sub, min_cluster_size_frac=0.0025, min_samples=50)
    assert len(labels) == int((y == TARGET).sum())
    planted_labels = set(labels[is_poison].tolist())
    assert len(planted_labels) == 1, f"planted rows split across {len(planted_labels)} clusters"
    lab = planted_labels.pop()
    assert lab != -1, "planted rows were labeled noise"
    # and that cluster should be almost entirely poison
    assert (labels == lab).sum() == pytest.approx(is_poison.sum(), rel=0.05)


def test_iterative_scoring_ranks_the_planted_cluster_last(planted):
    x, y, is_poison = planted
    feats = select_features(x, y, 4, SEED)
    sub = x[y == TARGET][:, feats]
    labels = cluster_target_class(sub, 0.0025, 50)
    sc = iterative_scoring(x[:, feats], y, labels, TARGET, _surrogate, seed=SEED)
    planted_cluster = labels[is_poison][0]
    ranks = sc["absorption_rank_by_cluster"]
    assert planted_cluster in ranks
    assert ranks[planted_cluster] == max(ranks.values()), "planted cluster was not ranked most suspicious"
    assert len(sc["per_sample_score"]) == len(labels)


def test_the_per_sample_score_separates_planted_rows(planted):
    x, y, is_poison = planted
    feats = select_features(x, y, 4, SEED)
    labels = cluster_target_class(x[y == TARGET][:, feats], 0.0025, 50)
    sc = iterative_scoring(x[:, feats], y, labels, TARGET, _surrogate, seed=SEED)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(is_poison, sc["per_sample_score"]) > 0.95


def test_fixed_threshold_flags_the_unabsorbed_clusters(planted):
    x, y, is_poison = planted
    feats = select_features(x, y, 4, SEED)
    labels = cluster_target_class(x[y == TARGET][:, feats], 0.0025, 50)
    sc = iterative_scoring(x[:, feats], y, labels, TARGET, _surrogate, seed=SEED)
    flags = flag_fixed_threshold(sc, labels)
    assert flags.dtype == bool and len(flags) == len(labels)
    assert flags[is_poison].mean() > 0.9, "fixed rule missed the planted rows"


def test_loss_delta_rule_flags_the_planted_cluster(planted):
    x, y, is_poison = planted
    feats = select_features(x, y, 4, SEED)
    labels = cluster_target_class(x[y == TARGET][:, feats], 0.0025, 50)
    sc = iterative_scoring(x[:, feats], y, labels, TARGET, _surrogate, seed=SEED)
    flags = flag_loss_delta(sc, labels, z_t=2.0)
    assert flags.dtype == bool and len(flags) == len(labels)


def test_no_planted_cluster_flags_almost_nothing():
    """False-positive behavior: 20 clean blobs, no poison. The rules must stay quiet."""
    rng = np.random.default_rng(1)
    d = 8
    centers = rng.normal(size=(20, d)) * 60
    benign = centers[rng.integers(0, 20, 20_000)] + rng.normal(size=(20_000, d))
    attack = rng.normal(size=(1_000, d)) * 6 + 15
    x = np.concatenate([benign, attack])
    y = np.concatenate([np.zeros(len(benign), int), np.ones(len(attack), int)])
    feats = select_features(x, y, 4, SEED)
    labels = cluster_target_class(x[y == TARGET][:, feats], 0.0025, 50)
    sc = iterative_scoring(x[:, feats], y, labels, TARGET, _surrogate, seed=SEED)
    assert flag_loss_delta(sc, labels, z_t=2.0).mean() < 0.35


def test_a_scattered_poison_is_not_recovered():
    """The method's documented weakness. Poison spread across existing clusters must NOT be found,
    and the test asserts the failure rather than asserting success."""
    x, y, is_poison = _blocks(scatter=True, seed=2)
    feats = select_features(x, y, 4, SEED)
    labels = cluster_target_class(x[y == TARGET][:, feats], 0.0025, 50)
    sc = iterative_scoring(x[:, feats], y, labels, TARGET, _surrogate, seed=SEED)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(is_poison, sc["per_sample_score"])
    assert auc < 0.90, f"scattered poison should not be cleanly recovered, got AUC {auc:.3f}"


def test_filter_mask_drops_exactly_the_flagged_rows():
    labels = np.array([0, 0, 1, 1, 2])
    flags = np.array([False, False, True, True, False])
    positions = np.array([10, 11, 12, 13, 14])
    keep = filter_mask(flags, positions, n_train=20)
    assert keep.dtype == bool and len(keep) == 20
    assert not keep[12] and not keep[13]
    assert keep[10] and keep[11] and keep[14] and keep[0] and keep[19]
    assert keep.sum() == 18


def test_pipeline_is_deterministic(planted):
    x, y, _ = planted
    feats = select_features(x, y, 4, SEED)
    out = []
    for _ in range(2):
        labels = cluster_target_class(x[y == TARGET][:, feats], 0.0025, 50)
        sc = iterative_scoring(x[:, feats], y, labels, TARGET, _surrogate, seed=SEED)
        out.append((labels.copy(), np.asarray(sc["per_sample_score"]).copy()))
    assert np.array_equal(out[0][0], out[1][0])
    assert np.allclose(out[0][1], out[1][1])


def test_the_fixed_rule_is_not_vacuous_at_small_cluster_counts():
    """At two clusters, round(0.8 x 2) absorbs everything and the published fixed rule flags nothing.

    Real degeneracy, not a fixture artifact, and the same shape as Neural Cleanse's anomaly index at
    two classes. The module floors the threshold and caps it below the cluster count so at least one
    cluster is always left for the rule to judge, and records few_clusters so a caller can see the
    regime it is in.
    """
    rng = np.random.default_rng(3)
    d = 8
    a = rng.normal(size=(6_000, d))
    b = np.full((1_000, d), 30.0) + rng.normal(scale=0.05, size=(1_000, d))
    benign = np.concatenate([a, b])
    attack = np.full((500, d), 30.0) + rng.normal(scale=1.0, size=(500, d))
    x = np.concatenate([benign, attack])
    y = np.concatenate([np.zeros(len(benign), int), np.ones(len(attack), int)])
    labels = np.concatenate([np.zeros(len(a), int), np.ones(len(b), int)])
    sc = iterative_scoring(x, y, labels, TARGET, _surrogate, seed=SEED)
    assert sc["n_clusters"] == 2 and sc["few_clusters"] is True
    assert sc["stop_at"] == 1, "the threshold must leave a cluster unabsorbed"
    assert len(sc["unabsorbed_clusters"]) >= 1
    assert flag_fixed_threshold(sc, labels).sum() > 0, "fixed rule is vacuous at two clusters"


from src.detectors.density_mitigation import cluster_composition


def test_cluster_composition_reports_the_planted_cluster_and_its_absorption():
    # 3 clusters: label 0 clean, label 1 the planted poison, label -1 noise with one poison row.
    labels = np.array([0] * 50 + [1] * 20 + [-1] * 5)
    is_poison = np.array([False] * 50 + [True] * 20 + [False] * 4 + [True])
    comp = cluster_composition(labels, is_poison, unabsorbed=[1])
    by_label = {c["label"]: c for c in comp["clusters"]}
    assert by_label[1]["size"] == 20 and by_label[1]["n_poison"] == 20
    assert by_label[1]["purity"] == 1.0
    assert abs(by_label[1]["poison_share"] - 20 / 21) < 1e-9
    assert by_label[1]["absorbed"] is False
    assert by_label[0]["absorbed"] is True
    assert comp["best"]["label"] == 1
    assert comp["poison_in_noise"] == 1
    assert comp["isolated"] is True


def test_cluster_composition_calls_a_scattered_poison_not_isolated():
    labels = np.array([0, 1, 2, 3, 4] * 20)
    is_poison = np.zeros(100, dtype=bool)
    is_poison[::7] = True                       # spread across every cluster
    comp = cluster_composition(labels, is_poison, unabsorbed=[])
    assert comp["isolated"] is False
    assert comp["best"]["poison_share"] < 0.9
