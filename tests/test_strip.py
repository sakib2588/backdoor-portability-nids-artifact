"""STRIP's decision rules, and in particular the published FRR rule.

Why this file exists. STRIP shipped in this project with no test of any kind, scored only
under a top-k poison budget the defender is not entitled to assume, while
`strip_threshold_flag` --- Gao et al.'s actual decision rule --- sat in the module and was
never called from anywhere. This project's own history says a broken decision layer is
indistinguishable from an honest negative result, and STRIP's 0.0000 MAD recall currently
feeds two claims in the manuscript. So the rule is tested on synthetic scores with a known
answer, not on a model, before it is wired into any runner.

Orientation, which is the easiest thing here to get backwards: `strip_scores` returns
NEGATED mean entropy, so a high score means low entropy means poison-like. The cutoff
returned by `fit_strip_entropy_cutoff` is on the ENTROPY scale (positive), because that is
what `strip_threshold_flag` consumes and what Gao et al. state.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.detectors.strip import (
    DEFAULT_FRR,
    calibrate_strip_cutoff,
    fit_strip_entropy_cutoff,
    strip_scores,
    strip_threshold_flag,
)


def _clean_scores(n=10_000, seed=0):
    """Clean STRIP scores: entropy centred well above zero, so negated scores are negative."""
    rng = np.random.default_rng(seed)
    entropy = rng.normal(loc=1.0, scale=0.15, size=n)
    return -entropy


# --------------------------------------------------------------- the rule's defining property
def test_frr_flags_the_requested_fraction_of_clean_by_construction():
    """The whole point of the rule: at FRR 1%, 1% of clean is flagged. Not a proxy for it."""
    clean = _clean_scores()
    cut = fit_strip_entropy_cutoff(clean, frr=0.01, partition="clean_calibration")
    flagged = strip_threshold_flag(clean, cut)
    assert flagged.mean() == pytest.approx(0.01, abs=0.002)


@pytest.mark.parametrize("frr", [0.001, 0.01, 0.05, 0.10])
def test_frr_holds_across_rates(frr):
    clean = _clean_scores(seed=1)
    cut = fit_strip_entropy_cutoff(clean, frr=frr, partition="clean_calibration")
    assert strip_threshold_flag(clean, cut).mean() == pytest.approx(frr, abs=max(0.002, frr * 0.15))


def test_poison_below_the_clean_entropy_floor_is_recovered():
    """A triggered sample's entropy collapses. Planted below the clean floor, it must be caught
    at a cutoff fit only on clean rows -- and the clean rows must keep their 1% FRR."""
    clean = _clean_scores(seed=2)
    poison = -np.full(200, 0.05)                      # entropy 0.05, far below the clean mass
    cut = fit_strip_entropy_cutoff(clean, frr=0.01, partition="clean_calibration")

    assert strip_threshold_flag(poison, cut).all()
    assert strip_threshold_flag(clean, cut).mean() == pytest.approx(0.01, abs=0.002)


def test_poison_inside_the_clean_mass_is_not_magically_recovered():
    """A negative control. The rule is a threshold, not an oracle: poison drawn from the clean
    distribution must be recovered at about the FRR, not at 1.0. Without this, a test suite
    that only ever plants separable poison cannot tell a working rule from an always-flag."""
    clean = _clean_scores(seed=3)
    rng = np.random.default_rng(4)
    poison = -rng.normal(loc=1.0, scale=0.15, size=2000)
    cut = fit_strip_entropy_cutoff(clean, frr=0.01, partition="clean_calibration")
    assert strip_threshold_flag(poison, cut).mean() == pytest.approx(0.01, abs=0.01)


# --------------------------------------------------------------- orientation
def test_cutoff_is_on_the_entropy_scale_and_is_not_re_signed():
    """`strip_scores` is already negated (strip.py returns -mean_entropy). The cutoff must come
    back positive for a positive-entropy clean pool; a sign slip here silently inverts the
    detector and would read as STRIP failing."""
    clean = _clean_scores(seed=5)                     # entropy ~ 1.0, scores ~ -1.0
    cut = fit_strip_entropy_cutoff(clean, frr=0.01, partition="clean_calibration")
    assert cut > 0
    assert cut == pytest.approx(np.quantile(-clean, 0.01), rel=1e-12)
    assert cut < 1.0                                  # the 1st percentile sits below the mean


def test_higher_score_is_more_poison_like():
    cut = fit_strip_entropy_cutoff(_clean_scores(seed=6), frr=0.01, partition="clean_calibration")
    assert strip_threshold_flag(np.array([10.0]), cut)[0]      # entropy -10, extreme anomaly
    assert not strip_threshold_flag(np.array([-10.0]), cut)[0]  # entropy 10, extremely stable


# --------------------------------------------------------------- the leakage guard
def test_refuses_a_partition_that_is_not_clean_calibration():
    with pytest.raises(ValueError, match="clean_calibration"):
        fit_strip_entropy_cutoff(_clean_scores(), partition="test")


def test_refuses_a_pool_declared_to_contain_poison():
    """Distinct from the partition check: a pool correctly named but contaminated upstream."""
    with pytest.raises(ValueError, match="contains_poison"):
        fit_strip_entropy_cutoff(_clean_scores(), partition="clean_calibration",
                                 contains_poison=True)


def test_refuses_an_empty_pool():
    with pytest.raises(ValueError, match="empty"):
        fit_strip_entropy_cutoff(np.array([]), partition="clean_calibration")


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.01, 1.5, float("nan")])
def test_refuses_an_out_of_range_frr(bad):
    with pytest.raises(ValueError, match="frr"):
        fit_strip_entropy_cutoff(_clean_scores(), frr=bad, partition="clean_calibration")


def test_refuses_a_non_finite_pool():
    clean = _clean_scores(seed=7).copy()
    clean[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_strip_entropy_cutoff(clean, partition="clean_calibration")


def test_default_frr_is_gao_et_als_one_percent():
    assert DEFAULT_FRR == 0.01


# --------------------------------------------------------------- the shared calibration path
class _ConstantLogits(torch.nn.Module):
    """A model whose output ignores its input, so every blend has identical entropy."""

    def forward(self, x):
        return torch.zeros(x.shape[0], 2)


class _FirstFeatureModel(torch.nn.Module):
    """Logits driven by feature 0, so entropy varies with the input and blending matters."""

    def forward(self, x):
        z = x[:, :1] * 4.0
        return torch.cat([z, -z], dim=1)


def test_calibrate_returns_a_cutoff_that_flags_about_the_frr_on_held_out_clean_rows():
    """The integration property: a cutoff fit on one clean slice generalises to another. Fit and
    evaluation pools are disjoint draws, so this is not the by-construction identity above."""
    rng = np.random.default_rng(11)
    pool = rng.normal(size=(6000, 5)).astype(np.float32)
    held_out = rng.normal(size=(3000, 5)).astype(np.float32)
    model = _FirstFeatureModel()

    cut = calibrate_strip_cutoff(model, pool, n_trials=8, seed=3, n_calib=1500, frr=0.05)
    scores = strip_scores(model, held_out, pool, n_trials=8, seed=3)
    assert strip_threshold_flag(scores, cut).mean() == pytest.approx(0.05, abs=0.03)


def test_calibrate_never_blends_a_query_with_itself():
    """Guards the disjointness directly. With a constant-logit model every score is identical,
    so a self-blend cannot be detected by value -- instead assert the split itself is clean."""
    pool = np.arange(200 * 3, dtype=np.float32).reshape(200, 3)
    seen = {}
    real = strip_scores

    def spy(model, x_std, blend_pool_std, **kw):
        seen["q"] = {tuple(r) for r in np.asarray(x_std)}
        seen["r"] = {tuple(r) for r in np.asarray(blend_pool_std)}
        return real(model, x_std, blend_pool_std, **kw)

    import src.detectors.strip as strip_mod
    strip_mod.strip_scores = spy
    try:
        calibrate_strip_cutoff(_ConstantLogits(), pool, n_trials=4, seed=1, n_calib=50)
    finally:
        strip_mod.strip_scores = real

    assert len(seen["q"]) == 50
    assert seen["q"].isdisjoint(seen["r"])
    assert len(seen["q"]) + len(seen["r"]) == 200


def test_calibrate_refuses_a_pool_too_small_to_split():
    with pytest.raises(ValueError, match="too few"):
        calibrate_strip_cutoff(_ConstantLogits(), np.zeros((4, 3), dtype=np.float32))


def test_calibrate_is_deterministic_for_a_seed():
    pool = np.random.default_rng(12).normal(size=(2000, 4)).astype(np.float32)
    m = _FirstFeatureModel()
    a = calibrate_strip_cutoff(m, pool, n_trials=4, seed=7, n_calib=400)
    b = calibrate_strip_cutoff(m, pool, n_trials=4, seed=7, n_calib=400)
    assert a == b


# --------------------------------------------------------------- the saturation-floor tie
#
# Root cause of a false negative caught on 2026-09-04. STRIP's entropy is computed from a
# clamped softmax, so a fully saturated prediction lands on an exact numerical floor:
# 1e-12 * -log(1e-12) = 2.7631e-11. On a trained victim a large fraction of rows saturate --
# measured on the CTU-13 positive control, 2.88% of clean calibration rows, 3.59% of scored
# clean rows and 99.39% of poison. A 1% FRR target therefore falls INSIDE the tie, the fitted
# cutoff equals the floor, and a strict `entropy < cut` comparison can never fire. The rule
# reported recall 0.0000 at FPR 0.0000 on all five seeds while the same scores ranked at AUC
# 0.9839, which reads exactly like an honest negative result and is not one.
FLOOR = 1e-12 * -np.log(1e-12)


def _clean_with_floor_ties(n=10_000, tie_frac=0.03, seed=0):
    """Clean scores whose bottom `tie_frac` sit exactly on the saturation floor."""
    rng = np.random.default_rng(seed)
    entropy = np.abs(rng.normal(loc=0.05, scale=0.02, size=n)) + 1e-3
    entropy[: int(n * tie_frac)] = FLOOR
    return -entropy


def test_a_target_inside_the_tie_still_flags_the_tied_mass():
    """The regression. With 3% of clean tied at the floor and a 1% target, the old strict rule
    flagged nothing at all. The rule must instead reach the smallest achievable operating
    point, which is the tie itself."""
    clean = _clean_with_floor_ties(tie_frac=0.03)
    cut = fit_strip_entropy_cutoff(clean, frr=0.01, partition="clean_calibration")
    flagged = strip_threshold_flag(clean, cut)
    assert flagged.mean() > 0.0, "a 1% target inside a 3% tie must not flag zero rows"
    assert flagged.mean() == pytest.approx(0.03, abs=0.005)


def test_poison_sitting_on_the_floor_is_recovered():
    """The consequence that matters. Poison saturates the softmax, so it lands on the floor.
    Under the old strict rule none of it was recoverable at any FRR."""
    clean = _clean_with_floor_ties(tie_frac=0.03, seed=1)
    poison = -np.full(500, FLOOR)
    cut = fit_strip_entropy_cutoff(clean, frr=0.01, partition="clean_calibration")
    assert strip_threshold_flag(poison, cut).mean() == pytest.approx(1.0)


def test_degeneracy_is_reported_rather_than_silently_returning_a_dead_cutoff():
    """A cutoff that cannot hit its target must say so. Silence here is what made the false
    negative look like a finding."""
    clean = _clean_with_floor_ties(tie_frac=0.03, seed=2)
    cut, diag = fit_strip_entropy_cutoff(clean, frr=0.01, partition="clean_calibration",
                                         return_diagnostics=True)
    assert diag["degenerate"] is True
    assert diag["requested_frr"] == 0.01
    assert diag["realized_frr"] == pytest.approx(0.03, abs=0.005)
    assert diag["tie_mass_at_cutoff"] == pytest.approx(0.03, abs=0.005)


def test_no_degeneracy_is_reported_on_a_continuous_pool():
    clean = _clean_scores(seed=8)
    cut, diag = fit_strip_entropy_cutoff(clean, frr=0.01, partition="clean_calibration",
                                         return_diagnostics=True)
    assert diag["degenerate"] is False
    assert diag["realized_frr"] == pytest.approx(0.01, abs=0.002)
    assert diag["tie_mass_at_cutoff"] < 0.005


def test_diagnostics_are_opt_in_so_existing_callers_still_get_a_float():
    clean = _clean_scores(seed=9)
    assert isinstance(fit_strip_entropy_cutoff(clean, partition="clean_calibration"), float)
