"""Unit tests for the M4 statistical primitives (bootstrap CI, Cohen's d_z, paired-diff CI)."""
import numpy as np

from src.stats import bootstrap_ci, cohens_dz, paired_diff_ci


def test_bootstrap_ci_point_estimate_is_the_mean():
    vals = [0.90, 0.92, 0.95, 0.88, 0.91]
    mean, (lo, hi) = bootstrap_ci(vals, seed=0)
    assert abs(mean - float(np.mean(vals))) < 1e-9
    assert lo <= mean <= hi
    assert lo >= 0.0 and hi <= 1.0


def test_bootstrap_ci_is_deterministic_given_seed():
    vals = [0.1, 0.4, 0.5, 0.7, 0.9]
    a = bootstrap_ci(vals, seed=7)
    b = bootstrap_ci(vals, seed=7)
    assert a == b


def test_bootstrap_ci_degenerate_single_value():
    mean, (lo, hi) = bootstrap_ci([0.5], seed=0)
    assert mean == 0.5 and lo == 0.5 and hi == 0.5


def test_cohens_dz_known_direction_and_sign():
    vision = [0.98, 0.97, 0.99, 0.98, 0.97]
    nids = [0.10, 0.12, 0.08, 0.15, 0.11]
    dz = cohens_dz(vision, nids)
    assert dz > 0
    diffs = np.array(vision) - np.array(nids)
    expected = float(diffs.mean() / diffs.std(ddof=1))
    assert abs(dz - expected) < 1e-9


def test_cohens_dz_zero_variance_diff_returns_none():
    a = [0.5, 0.6, 0.7]
    b = [0.4, 0.5, 0.6]
    assert cohens_dz(a, b) is None


def test_paired_diff_ci_brackets_mean_difference():
    vision = [0.98, 0.97, 0.99, 0.98, 0.97]
    nids = [0.10, 0.12, 0.08, 0.15, 0.11]
    mean_diff, (lo, hi) = paired_diff_ci(vision, nids, seed=0)
    assert abs(mean_diff - float(np.mean(np.array(vision) - np.array(nids)))) < 1e-9
    assert lo <= mean_diff <= hi
    assert lo > 0
