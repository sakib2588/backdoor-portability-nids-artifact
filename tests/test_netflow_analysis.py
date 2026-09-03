"""The pre-registered NetFlow analysis: the heterogeneity statistic and the exact permutation test.

The heterogeneity statistic is tested against synthetic cases with known answers because two earlier
versions of it were wrong and both were caught by running them, not by reading them. Equal-width
binning measured outlier presence; equal-frequency binning was degenerate at about 1.0 for every
continuous feature. Both would have passed a reading.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "netflow_analysis", ROOT / "scripts" / "82_netflow_property_analysis.py")
na = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(na)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n, d = 20_000, 8
    iso = rng.normal(size=(n, d))
    return dict(
        iso=iso,
        rank1=rng.normal(size=(n, 1)) * np.arange(1, d + 1),
        outlier=np.concatenate([[np.concatenate([[1e6], iso[0, 1:]])], iso[1:]]),
        half_constant=np.concatenate([iso[:, :4], np.full((n, 4), 3.0)], axis=1),
    )


def test_isotropic_data_spans_every_dimension(data):
    assert na.benign_heterogeneity(data["iso"]) == pytest.approx(8.0, abs=0.05)


def test_rank_one_data_spans_exactly_one_dimension(data):
    assert na.benign_heterogeneity(data["rank1"]) == pytest.approx(1.0, abs=0.01)


def test_half_constant_features_halve_the_dimensionality(data):
    assert na.benign_heterogeneity(data["half_constant"]) == pytest.approx(4.0, abs=0.05)


def test_a_single_extreme_outlier_does_not_move_it(data):
    """The equal-width version's failure: one extreme value widened the range until nearly all mass
    sat in the first bin, so the statistic measured outlier presence instead of spread."""
    assert na.benign_heterogeneity(data["outlier"]) == pytest.approx(
        na.benign_heterogeneity(data["iso"]), abs=0.05)


def test_it_is_not_degenerate_across_continuous_data(data):
    """The equal-frequency version's failure: quantile bins are equally populated by construction, so
    normalized entropy returned about 1.0 for every continuous feature regardless of spread."""
    lo = na.benign_heterogeneity(data["rank1"])
    hi = na.benign_heterogeneity(data["iso"])
    assert hi - lo > 5.0


def test_all_constant_input_returns_zero():
    assert na.benign_heterogeneity(np.full((100, 5), 2.0)) == 0.0


def test_perfect_monotone_gradient_hits_the_minimum_attainable_p():
    r = na.rank_test(np.array([0.05, 0.20, 0.40, 0.60, 0.80, 0.96]),
                     np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    assert r["rho"] == pytest.approx(1.0)
    assert r["n_permutations"] == 720
    assert r["p_exact_two_sided"] == pytest.approx(2 / 720, abs=1e-9)
    assert r["min_attainable_p"] == pytest.approx(2 / 720)


def test_a_reversed_gradient_is_equally_significant_two_sided():
    r = na.rank_test(np.array([0.05, 0.20, 0.40, 0.60, 0.80, 0.96]),
                     np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))
    assert r["rho"] == pytest.approx(-1.0)
    assert r["p_exact_two_sided"] == pytest.approx(2 / 720, abs=1e-9)


def test_losing_levels_weakens_the_test_visibly():
    """Saturation removes levels. n falls and the minimum attainable p rises with it, and both are
    reported so a reader can see the test weaken rather than having to infer it."""
    full = na.rank_test(np.arange(6.0), np.arange(6.0))
    short = na.rank_test(np.arange(4.0), np.arange(4.0))
    assert short["n"] == 4 and short["n_permutations"] == 24
    assert short["min_attainable_p"] > full["min_attainable_p"]


def test_too_few_levels_refuses_to_report_a_rho():
    r = na.rank_test(np.array([0.1, 0.2]), np.array([1.0, 2.0]))
    assert r["rho"] is None and "fewer than three" in r["note"]


@pytest.mark.parametrize("rho_p,rho_r,expected", [
    (0.95, 0.80, "CONFIRMED"),
    (0.95, -0.80, "INCONCLUSIVE"),   # replication disagrees in sign
    (0.30, 0.90, "REFUTED"),
    (0.60, 0.60, "INCONCLUSIVE"),
    (None, 0.90, "INCONCLUSIVE"),
])
def test_verdict_follows_the_preregistered_rule(rho_p, rho_r, expected):
    assert na.verdict({"rho": rho_p}, {"rho": rho_r}) == expected
