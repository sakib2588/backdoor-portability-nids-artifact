"""Tests for the calibrated / multistart Neural Cleanse repair attempt (plan Task 7A).

The repair is only interpretable if start selection, tau calibration, convergence bookkeeping, and
raw-space feasibility all behave as specified BEFORE any CTU detection rate is read. In particular
`calibrate_binary_tau` must refuse held-out seeds: a leaked evaluation seed would make the reported
held-out detection rate partly self-selected, and nothing downstream would catch it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src import config
from src.constraints_torch import audit_constraints, make_constraint_penalty
from src.data import inverse_standardise
from src.detectors.neural_cleanse import (
    TabularInversionConfig,
    _converged,
    calibrate_binary_tau,
    nc_binary_flag,
    reverse_engineer_tabular,
    reverse_engineer_tabular_multistart,
    select_best_start,
)
from src.tb_vendor.relation_constraint import Feature, LessEqualConstraint

FEATURES = ["a", "b", "c", "d"]


class _TinyNet(nn.Module):
    """Deterministic 4-feature binary classifier, enough to run a real inversion loop."""

    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x):
        return self.net(x)


class _Scaler:
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=float)
        self.scale_ = np.asarray(scale, dtype=float)

    def inverse_transform(self, x):
        return np.asarray(x) * self.scale_ + self.mean_


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    return torch.as_tensor(rng.normal(0, 1, size=(32, 4)), dtype=torch.float32)


@pytest.fixture
def bounds():
    return (torch.full((4,), -3.0), torch.full((4,), 3.0))


# --- start selection ------------------------------------------------------------------------------

def test_multistart_selects_lowest_finite_converged_objective():
    ledger = [{"objective": 4.0, "converged": True}, {"objective": float("inf"), "converged": False},
              {"objective": 3.0, "converged": True}]
    assert select_best_start(ledger) == 2


def test_selection_ignores_an_unconverged_start_even_when_its_objective_is_lowest():
    ledger = [{"objective": 4.0, "converged": True}, {"objective": 0.1, "converged": False}]
    assert select_best_start(ledger) == 0


def test_selection_returns_none_when_no_start_converged():
    ledger = [{"objective": 1.0, "converged": False}, {"objective": float("nan"), "converged": False}]
    assert select_best_start(ledger) is None


def test_selection_rejects_a_converged_but_non_finite_objective():
    ledger = [{"objective": float("nan"), "converged": True}, {"objective": 5.0, "converged": True}]
    assert select_best_start(ledger) == 1


# --- convergence criterion ------------------------------------------------------------------------

def test_plateaued_loss_counts_as_converged():
    assert _converged([10.0] * 5 + [1.0] * 95) is True


def test_still_descending_loss_is_not_converged():
    assert _converged(list(np.linspace(100.0, 1.0, 100))) is False


def test_non_finite_loss_is_not_converged():
    assert _converged([5.0, float("nan"), 3.0]) is False
    assert _converged([]) is False


def test_loss_that_never_descended_is_not_converged():
    assert _converged([1.0] * 50) is False           # flat from the start: never improved
    assert _converged(list(np.linspace(1.0, 5.0, 50))) is False   # ascending


def _measured_trace(n_steps: int, first: float, at_tail_start: float, last: float) -> list[float]:
    """Reconstruct an inversion trace from measured anchors: descent over the first 90% of steps,
    then the observed tail movement over the last 10%."""
    tail = max(1, int(round(0.1 * n_steps)))
    head = np.linspace(first, at_tail_start, n_steps - tail)
    return list(head) + list(np.linspace(at_tail_start, last, tail))


def test_asymptoting_loss_counts_as_converged():
    """Regression: the real inversion trace must pass.

    Measuring tail movement against the LOSS VALUE rejected 6/6 starts at every step count, because
    cross-entropy + an L1 mask penalty asymptotes towards zero and shrinks the denominator. The anchors
    below are the observed 757-feature probe at steps=300: 1.0264 -> 0.042183, with the value 30 steps
    from the end at 0.042769. Total descent 0.9842, tail movement 0.000586, ratio 6.0e-4.
    """
    trace = _measured_trace(300, first=1.0264, at_tail_start=0.042769, last=0.042183)
    assert _converged(trace) is True


def test_a_start_still_making_real_progress_at_the_tail_is_not_converged():
    """Same total descent, but the tail is still moving a large share of it."""
    trace = _measured_trace(300, first=1.0264, at_tail_start=0.30, last=0.042183)
    assert _converged(trace) is False


# --- tau calibration -----------------------------------------------------------------------------

def test_clean_calibration_cannot_use_evaluation_seed_ratios():
    with pytest.raises(ValueError, match="evaluation seed"):
        calibrate_binary_tau(clean_ratios={42: 1.2, 789: 1.3},
                             calibration_seeds=(42,), evaluation_seeds=(789,))


def test_tau_is_the_requested_quantile_of_the_calibration_ratios():
    ratios = {42: 1.0, 123: 2.0, 456: 3.0}
    tau = calibrate_binary_tau(ratios, calibration_seeds=(42, 123, 456), evaluation_seeds=(789, 1337),
                               quantile=0.99)
    assert tau == pytest.approx(np.quantile([1.0, 2.0, 3.0], 0.99))


def test_tau_calibration_uses_the_project_seed_partition():
    ratios = {s: 1.0 + i for i, s in enumerate(config.ADAPTIVE_SURROGATE_SEEDS)}
    tau = calibrate_binary_tau(ratios, config.ADAPTIVE_SURROGATE_SEEDS, config.ADAPTIVE_EVAL_SEEDS,
                               quantile=config.CLEAN_CALIBRATION_QUANTILE)
    assert np.isfinite(tau)


def test_tau_calibration_ignores_a_non_calibration_seed_that_is_not_held_out():
    tau = calibrate_binary_tau({42: 1.0, 999: 99.0}, calibration_seeds=(42,), evaluation_seeds=(789,))
    assert tau == pytest.approx(1.0)   # 999 is neither calibration nor eval -> not used


def test_tau_calibration_refuses_when_no_finite_calibration_ratio_exists():
    with pytest.raises(ValueError, match="no finite clean ratios"):
        calibrate_binary_tau({42: float("inf")}, calibration_seeds=(42,), evaluation_seeds=(789,))


# --- the multistart inversion itself --------------------------------------------------------------

def test_single_start_zero_init_reproduces_the_published_baseline(sample, bounds):
    """`n_starts=1, constraint_weight=0` must BE the baseline, not a re-implementation of it."""
    model = _TinyNet(seed=1)
    baseline = reverse_engineer_tabular(
        _TinyNet(seed=1), num_classes=2, sample_x_std=sample, device="cpu",
        steps=40, lr=0.1, l1_weight=0.001, bounds=bounds)
    norms, masks, ledger = reverse_engineer_tabular_multistart(
        model, sample, bounds,
        TabularInversionConfig(steps=40, lr=0.1, l1_weight=0.001, n_starts=1, constraint_weight=0.0),
        seed=42, constraint_penalty=None)
    np.testing.assert_allclose(norms, baseline, rtol=1e-5, atol=1e-6)
    assert masks.shape == (2, 4)
    assert ledger["n_starts"] == 1
    assert all(e["init"] == "zeros" for e in ledger["starts"])


def test_ledger_keeps_every_start_for_every_class(sample, bounds):
    norms, masks, ledger = reverse_engineer_tabular_multistart(
        _TinyNet(seed=2), sample, bounds,
        TabularInversionConfig(steps=30, n_starts=3), seed=42, constraint_penalty=None)
    assert len(ledger["starts"]) == 2 * 3          # 2 classes x 3 starts, nothing dropped
    assert len(ledger["selection"]) == 2
    assert {e["start"] for e in ledger["starts"]} == {0, 1, 2}
    for e in ledger["starts"]:
        assert "converged" in e and "objective" in e and "failure_reason" in e
    assert norms.shape == (2,) and masks.shape == (2, 4)


def test_start_zero_is_zeros_init_and_later_starts_are_seeded(sample, bounds):
    _, _, ledger = reverse_engineer_tabular_multistart(
        _TinyNet(seed=3), sample, bounds,
        TabularInversionConfig(steps=20, n_starts=3), seed=42, constraint_penalty=None)
    inits = {(e["start"], e["init"]) for e in ledger["starts"]}
    assert (0, "zeros") in inits
    assert (1, "seeded_normal") in inits and (2, "seeded_normal") in inits


def test_multistart_is_deterministic_for_a_fixed_seed(sample, bounds):
    cfg = TabularInversionConfig(steps=25, n_starts=3)
    a = reverse_engineer_tabular_multistart(_TinyNet(seed=4), sample, bounds, cfg, seed=42,
                                           constraint_penalty=None)
    b = reverse_engineer_tabular_multistart(_TinyNet(seed=4), sample, bounds, cfg, seed=42,
                                           constraint_penalty=None)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-6, atol=1e-8)


def test_unconverged_starts_are_counted_not_hidden(sample, bounds):
    """One step cannot plateau, so every start must be recorded as a convergence failure."""
    _, _, ledger = reverse_engineer_tabular_multistart(
        _TinyNet(seed=5), sample, bounds,
        TabularInversionConfig(steps=1, n_starts=2), seed=42, constraint_penalty=None)
    assert ledger["n_convergence_failures"] == len(ledger["starts"])
    assert ledger["n_no_converged_start"] == 2      # both classes fell back
    assert all(s["selection"].startswith("no_converged_start") for s in ledger["selection"])


def test_constraint_penalty_changes_the_inversion(sample, bounds):
    """A weighted penalty must actually enter the objective -- otherwise the arm tests nothing."""
    cons = [LessEqualConstraint(Feature("a"), Feature("b"))]
    scaler = _Scaler(mean=[0.0] * 4, scale=[1.0] * 4)
    penalty = make_constraint_penalty(cons, FEATURES, scaler)
    cfg_off = TabularInversionConfig(steps=60, n_starts=1, constraint_weight=0.0)
    cfg_on = TabularInversionConfig(steps=60, n_starts=1, constraint_weight=50.0)
    off = reverse_engineer_tabular_multistart(_TinyNet(seed=6), sample, bounds, cfg_off, seed=42,
                                             constraint_penalty=penalty)[1]
    on = reverse_engineer_tabular_multistart(_TinyNet(seed=6), sample, bounds, cfg_on, seed=42,
                                            constraint_penalty=penalty)[1]
    assert not np.allclose(off, on), "constraint_weight had no effect on the inversion"


# --- raw-space feasibility audit ------------------------------------------------------------------

def test_constraint_aware_pattern_is_audited_in_raw_space():
    constraints = [LessEqualConstraint(Feature("a"), Feature("b"))]
    scaler = _Scaler(mean=[10.0, 10.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0, 1.0])
    pattern_std = np.array([-1.0, 1.0, 0.0, 0.0])          # raw (9, 11): a <= b, feasible
    raw_pattern = inverse_standardise(pattern_std, scaler)
    assert audit_constraints(raw_pattern, constraints, FEATURES)["all_projectable_valid"] is True


def test_audit_flags_an_infeasible_pattern():
    constraints = [LessEqualConstraint(Feature("a"), Feature("b"))]
    scaler = _Scaler(mean=[10.0, 10.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0, 1.0])
    raw_pattern = inverse_standardise(np.array([1.0, -1.0, 0.0, 0.0]), scaler)   # raw (11, 9)
    audit = audit_constraints(raw_pattern, constraints, FEATURES)
    assert audit["all_projectable_valid"] is False
    assert audit["max_violation_magnitude"] == pytest.approx(2.0, abs=1e-6)
    assert audit["n_valid"] == 0 and audit["n_rows"] == 1


def test_inverse_standardise_round_trips():
    scaler = _Scaler(mean=[1.0, 2.0, 3.0, 4.0], scale=[2.0, 2.0, 2.0, 2.0])
    std = np.array([[0.5, -0.5, 1.0, 0.0]])
    np.testing.assert_allclose(inverse_standardise(std, scaler), std * 2.0 + [1.0, 2.0, 3.0, 4.0])


# --- the decision rule the repair replaces --------------------------------------------------------

def test_baseline_binary_rule_is_unchanged():
    """tau=2.0 stays the NAMED BASELINE; calibration is a separate arm, not an edit to this rule."""
    flagged, ratio = nc_binary_flag(np.array([10.0, 2.0]), tau=2.0)
    assert flagged == 1 and ratio == pytest.approx(5.0)
    flagged, ratio = nc_binary_flag(np.array([0.0, 2.0]), tau=2.0)
    assert flagged == 0 and ratio == float("inf")
