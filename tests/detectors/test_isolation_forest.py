"""Isolation Forest as a tabular-native comparator.

Shaped after tests/detectors/test_spectre.py. Every stage is tested where it should
succeed and where it should fail, per tests/test_density_mitigation.py's standing rule:
a statistic that only passes on the happy path has not been tested.
"""

import numpy as np
import pytest

from src.detectors import poison_recall
from src.detectors.isolation_forest import isolation_diagnostics, isolation_scores
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores


def _planted(n_clean=400, n_poison=20, dim=24, shift=6.0, n_shifted=8, seed=0):
    """Clean rows from a standard normal, poison displaced along `n_shifted` axes.

    The displacement is large because this test is about wiring and direction, not about
    how faint a signal the forest can find. A subtle planted set would make a sign error
    and a weak detector indistinguishable.

    n_shifted matters more than it looks. Isolation Forest splits on randomly chosen
    features, so an anomaly confined to one of many dimensions is diluted by the ones
    that carry no signal. Eight of twenty-four is a fair test of the wiring;
    test_single_feature_anomaly_is_diluted covers the hard regime deliberately.
    """
    rng = np.random.default_rng(seed)
    clean = rng.standard_normal((n_clean, dim))
    poison = rng.standard_normal((n_poison, dim))
    poison[:, :n_shifted] += shift
    x = np.vstack([clean, poison])
    is_poison = np.zeros(len(x), dtype=bool)
    is_poison[n_clean:] = True
    return x, is_poison


def test_scores_are_higher_for_planted_outliers():
    """The sign test. sklearn's score_samples runs the other way and must be negated.

    This is the single most important assertion in the file. Without the negation in
    isolation_scores, every decision rule in this project reads the wrong tail and the
    detector becomes a perfect anti-detector whose recall would look like a finding.
    """
    x, is_poison = _planted()
    s = isolation_scores(x, seed=0)
    assert s.shape == (len(x),)
    assert np.isfinite(s).all()
    assert s[is_poison].mean() > s[~is_poison].mean()


def test_recovers_poison_under_both_decision_rules():
    """Both rules the project scores every detector on, against one score vector."""
    x, is_poison = _planted()
    s = isolation_scores(x, seed=0)

    fixed = poison_recall(flag_by_scores(s, expected_frac=is_poison.mean()), is_poison)
    mad = poison_recall(flag_by_mad_threshold(s, z_thresh=3.0), is_poison)

    assert fixed > 0.5, f"fixed-budget recall {fixed} on a 6-sigma planted set"
    assert mad > 0.5, f"MAD recall {mad} on a 6-sigma planted set"


def test_is_deterministic():
    x, _ = _planted()
    assert np.array_equal(isolation_scores(x, seed=7), isolation_scores(x, seed=7))


def test_different_seeds_do_not_change_the_ranking_direction():
    """Seed may move values; it must not flip which side the poison sits on."""
    x, is_poison = _planted()
    for seed in (0, 1, 42):
        s = isolation_scores(x, seed=seed)
        assert s[is_poison].mean() > s[~is_poison].mean(), f"direction flipped at seed {seed}"


def test_handles_degenerate_input():
    """Empty block, and a block with no variance at all."""
    assert isolation_scores(np.zeros((0, 5))).shape == (0,)

    identical = np.ones((50, 5))
    s = isolation_scores(identical, seed=0)
    assert s.shape == (50,)
    assert np.isfinite(s).all()
    # Nothing is an outlier when every row is the same. The spread should be ~0, and what
    # matters is that it does not produce NaN and does not raise.
    assert float(s.std()) < 1e-6

    with pytest.raises(ValueError):
        isolation_scores(np.ones(10))  # 1-D, not a block


def test_finds_nothing_when_there_is_nothing_to_find():
    """The must-fail case. With no planted structure, separation is near zero.

    A detector that reports separation on homogeneous data is measuring its own noise.
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal((400, 24))
    fake = np.zeros(len(x), dtype=bool)
    fake[:20] = True  # an arbitrary label with no corresponding structure
    d = isolation_diagnostics(isolation_scores(x, seed=0), fake)
    assert abs(d["separation"]) < 0.05, d


def test_diagnostics_guard_degenerate_labels():
    x, _ = _planted()
    s = isolation_scores(x, seed=0)
    assert isolation_diagnostics(s, np.zeros(len(x), dtype=bool))["separation"] is None
    assert isolation_diagnostics(s, np.ones(len(x), dtype=bool))["separation"] is None
    with pytest.raises(ValueError):
        isolation_diagnostics(s, np.zeros(3, dtype=bool))


def test_single_feature_anomaly_is_diluted():
    """A documented weakness, not a bug, and it is directly relevant to this paper.

    Isolation Forest chooses split features at random, so an anomaly confined to a small
    fraction of the columns is diluted by the columns carrying no signal. At one shifted
    feature in twenty-four, a 6-sigma displacement that the MAD rule still catches falls
    below half recall under the inherited fixed budget.

    This is worth pinning because the attack it will be run against stamps 16 of 757
    features, a far thinner fraction than this test uses. If the window-cell result comes
    back weak, this is the first thing to check before concluding anything about the
    attack.

    Note which way the two rules fall, because it is the opposite of this project's usual
    finding and the expectation written here first was wrong. Everywhere else, a starved
    fixed budget is the problem and the budget-free MAD rule recovers what it missed. Under
    dilution the order reverses: the fixed budget still takes its top-k and salvages
    something, while MAD collapses, because dilution flattens the upper tail that MAD keys
    on. A rule that is a repair in one regime is not a repair in every regime.
    """
    x, is_poison = _planted(n_shifted=1)
    s = isolation_scores(x, seed=0)
    fixed = poison_recall(flag_by_scores(s, expected_frac=is_poison.mean()), is_poison)
    mad = poison_recall(flag_by_mad_threshold(s, z_thresh=3.0), is_poison)
    assert fixed < 0.5, f"expected dilution under the fixed budget, got {fixed}"
    assert mad < fixed, (
        f"expected MAD to collapse harder than the fixed budget under dilution, "
        f"got mad={mad} fixed={fixed}"
    )
