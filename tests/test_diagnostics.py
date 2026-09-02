"""Unit tests for src/diagnostics.py -- the attack-mechanism probes behind the LightGBM
FNR-floor result (notes/20260716-decision-lgb-fnr-floor.md)."""
import numpy as np
import pytest

from src.diagnostics import prediction_change_count, target_predicted_set, leaf_overlap_fraction


def test_prediction_change_count_counts_only_differences():
    assert prediction_change_count(np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1])) == 2


def test_prediction_change_count_rejects_length_mismatch():
    with pytest.raises(ValueError):
        prediction_change_count(np.array([0, 1]), np.array([0]))


def test_target_predicted_set_returns_row_indices_for_target():
    got = target_predicted_set(np.array([0, 1, 0, 1, 0]), target=0)
    assert got == frozenset({0, 2, 4})


def test_target_predicted_set_is_empty_when_target_absent():
    assert target_predicted_set(np.array([1, 1]), target=0) == frozenset()


def test_leaf_overlap_fraction_is_one_when_all_test_rows_share_a_poison_leaf():
    # 3 trees; poison rows occupy leaf 7 in every tree, and so do the test rows
    poison_leaves = np.array([[7, 7, 7], [7, 7, 7]])
    test_leaves = np.array([[7, 7, 7], [7, 7, 7], [7, 7, 7]])
    assert leaf_overlap_fraction(poison_leaves, test_leaves) == pytest.approx(1.0)


def test_leaf_overlap_fraction_is_zero_when_no_leaf_is_shared():
    poison_leaves = np.array([[1, 1], [1, 1]])
    test_leaves = np.array([[2, 2], [3, 3]])
    assert leaf_overlap_fraction(poison_leaves, test_leaves) == pytest.approx(0.0)


def test_leaf_overlap_fraction_averages_over_trees():
    # tree 0: both test rows land in the poison leaf (1.0); tree 1: neither does (0.0) -> 0.5
    poison_leaves = np.array([[5, 9]])
    test_leaves = np.array([[5, 4], [5, 4]])
    assert leaf_overlap_fraction(poison_leaves, test_leaves) == pytest.approx(0.5)


def test_leaf_overlap_fraction_rejects_tree_count_mismatch():
    with pytest.raises(ValueError):
        leaf_overlap_fraction(np.array([[1, 2]]), np.array([[1, 2, 3]]))
