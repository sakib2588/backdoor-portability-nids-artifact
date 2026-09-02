"""Tests for the expected-transferability (ET) decision rule (plan Task NC-13).

ET replaces the mask-L1 RATIO that NC-6 showed carries no signal (AUC 0.389-0.444 on 432 rows). Its
threshold is a DERIVED constant 1/2, so the thing most worth pinning is that the threshold is applied
as stated and never quietly calibrated, and that the transferable set unions over restarts rather
than taking the best start -- dropping either would turn ET into a different statistic wearing the
same name.

DELIBERATELY NOT TESTED: that a clean model yields ET <= 0.5. In the mask/pattern formulation an
optimiser can often overwrite whichever feature decides the class, giving a universal flip whether or
not a backdoor exists. Whether ET's null behaviour survives that on tabular NIDS is precisely the
open empirical question NC-13 was pre-registered to answer
(notes/20260803-decision-nc13-et-preregistration.md names it as the main risk). Asserting it here
would bake in the answer.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.detectors.neural_cleanse import (
    TabularInversionConfig,
    et_from_transfer_matrix,
    expected_transferability,
)


class _BackdoorNet(nn.Module):
    """Feature 0 controls the class: rows below 0.5 are class 1, driving it above 0.5 forces class 0.
    One shared shortcut, so a trigger found on any row transfers to all of them.

    The gate is LINEAR on purpose. A sigmoid gate is the natural way to write this, but a sharp one
    (`sigmoid((x0 - 0.5) * 20)`) puts rows drawn near -1.5 about forty units into saturation, where
    the gradient is numerically zero and the inversion cannot climb out at any step budget -- it
    drives the mask to 0.001 and never flips anything, giving ET 0.167 on a model that genuinely does
    have a universal backdoor. That would be a broken fixture masquerading as a failing statistic. A
    linear logit has a constant gradient and cannot saturate.
    """

    def forward(self, x):
        z = (x[:, 0] - 0.5) * 3.0
        return torch.stack([z, -z], dim=1)


class _AlwaysTargetNet(nn.Module):
    """Predicts class 0 for everything, so no row survives ET's f(x) = source filter."""

    def forward(self, x):
        return torch.stack([torch.full((x.shape[0],), 5.0), torch.full((x.shape[0],), -5.0)], dim=1)


# --- the pure statistic -----------------------------------------------------------------------

def test_all_transferable_gives_one():
    assert et_from_transfer_matrix(np.ones((4, 4), dtype=bool)) == pytest.approx(1.0)


def test_nothing_transferable_gives_zero():
    assert et_from_transfer_matrix(np.zeros((4, 4), dtype=bool)) == pytest.approx(0.0)


def test_diagonal_is_ignored():
    """A sample's trigger trivially flips that same sample. Alg. 1 scores each row out of N-1
    excluding m == n, so an identity matrix carries no transferability at all."""
    assert et_from_transfer_matrix(np.eye(5, dtype=bool)) == pytest.approx(0.0)


def test_known_partial_case():
    # row 0 transfers to 1 of 2 others, row 1 to 2 of 2, row 2 to 0 of 2 -> mean(0.5, 1.0, 0.0)
    t = np.array([[True, True, False],
                  [True, True, True],
                  [False, False, True]], dtype=bool)
    assert et_from_transfer_matrix(t) == pytest.approx((0.5 + 1.0 + 0.0) / 3)


def test_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        et_from_transfer_matrix(np.ones((2, 3), dtype=bool))


def test_rejects_fewer_than_two_samples():
    with pytest.raises(ValueError, match="fewer than 2"):
        et_from_transfer_matrix(np.ones((1, 1), dtype=bool))


# --- the driver -------------------------------------------------------------------------------

def _bounds(d):
    return (torch.full((d,), -3.0), torch.full((d,), 3.0))


def _source_rows(n=6, d=4, seed=0):
    """Rows comfortably below the gate, so the model predicts class 1 for all of them."""
    rng = np.random.default_rng(seed)
    x = rng.normal(-1.5, 0.2, size=(n, d))
    return torch.as_tensor(x, dtype=torch.float32)


def test_universal_backdoor_is_detected():
    """One shared shortcut exists, so a trigger inverted on any single row flips the rest."""
    out = expected_transferability(
        _BackdoorNet(), _source_rows(), _bounds(4),
        TabularInversionConfig(steps=60, n_starts=1), seed=42,
        putative_target=0, n_restarts=2)
    assert out["et"] > 0.5
    assert out["detected"] is True
    assert out["n_samples"] == 6


def test_threshold_is_the_derived_constant_and_is_reported():
    out = expected_transferability(
        _BackdoorNet(), _source_rows(), _bounds(4),
        TabularInversionConfig(steps=30, n_starts=1), seed=1,
        putative_target=0, n_restarts=1)
    assert out["threshold"] == 0.5
    assert "NOT calibrated" in out["threshold_origin"]
    assert out["detected"] == (out["et"] > 0.5)


def test_per_sample_rates_average_to_et():
    out = expected_transferability(
        _BackdoorNet(), _source_rows(), _bounds(4),
        TabularInversionConfig(steps=30, n_starts=1), seed=7,
        putative_target=0, n_restarts=1)
    assert np.mean(out["per_sample_transfer_rate"]) == pytest.approx(out["et"])


def test_rows_not_predicted_as_source_are_dropped():
    """Alg. 1 conditions on f(x) = i; a row already sent to the target evidences nothing."""
    out = expected_transferability(
        _AlwaysTargetNet(), _source_rows(), _bounds(4),
        TabularInversionConfig(steps=10, n_starts=1), seed=0,
        putative_target=0, n_restarts=1)
    assert out["et"] is None
    assert out["n_samples"] == 0
    assert "fewer than 2" in out["reason"]


def test_is_deterministic_for_a_fixed_seed():
    kw = dict(bounds=_bounds(4), config=TabularInversionConfig(steps=30, n_starts=1),
              seed=123, putative_target=0, n_restarts=2)
    a = expected_transferability(_BackdoorNet(), _source_rows(), **kw)
    b = expected_transferability(_BackdoorNet(), _source_rows(), **kw)
    assert a["et"] == b["et"]
    assert a["per_sample_transfer_rate"] == b["per_sample_transfer_rate"]


def test_rejects_non_binary_target():
    with pytest.raises(ValueError, match="must be 0 or 1"):
        expected_transferability(
            _BackdoorNet(), _source_rows(), _bounds(4),
            TabularInversionConfig(steps=5), seed=0, putative_target=2)
