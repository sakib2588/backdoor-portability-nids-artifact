"""The active-paths detector, and the input-gradient accessor it needs.

Pre-registered in notes/20260905-prereg-active-paths-detector.md. Read that before changing any
constant here.

Two things are tested separately because they fail differently. The gradient accessor has a known
closed form on a linear model and is checked against it exactly. The detector is a clustering
method, so it is checked on planted structure where the right answer is known by construction, plus
the degenerate cases that would otherwise return a confident number about nothing.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.detectors.active_paths import active_path_scores, slope_matrix
from src.models import mlp_input_gradients


class _Linear(torch.nn.Module):
    """f(x) = xW^T + b, so d f_c / d x is exactly W[c] everywhere."""

    def __init__(self, w: np.ndarray):
        super().__init__()
        self.lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor(w, dtype=torch.float32))

    def forward(self, x):
        return self.lin(x)


# --------------------------------------------------------------- the gradient accessor
def test_input_gradient_matches_the_closed_form_on_a_linear_model():
    w = np.array([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]])
    g = mlp_input_gradients(_Linear(w), np.random.default_rng(0).normal(size=(64, 3)), target=1)
    assert g.shape == (64, 3)
    assert np.allclose(g, w[1], atol=1e-5)


def test_input_gradient_selects_the_requested_target():
    w = np.array([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]])
    x = np.random.default_rng(1).normal(size=(16, 3))
    assert np.allclose(mlp_input_gradients(_Linear(w), x, target=0), w[0], atol=1e-5)
    assert np.allclose(mlp_input_gradients(_Linear(w), x, target=1), w[1], atol=1e-5)


def test_input_gradient_does_not_change_the_model():
    """The accessor builds a graph to differentiate the input. It must not move the weights."""
    w = np.random.default_rng(2).normal(size=(2, 4))
    m = _Linear(w)
    before = m.lin.weight.detach().clone()
    mlp_input_gradients(m, np.random.default_rng(3).normal(size=(32, 4)), target=0)
    assert torch.equal(before, m.lin.weight.detach())
    assert not m.training


def test_input_gradient_spans_batches():
    """More rows than the internal batch size, so the concatenation path is exercised."""
    w = np.array([[2.0, -1.0]])
    g = mlp_input_gradients(_Linear(np.vstack([w, -w])), np.zeros((2500, 2)), target=0)
    assert g.shape == (2500, 2)
    assert np.allclose(g, w[0], atol=1e-5)


def test_input_gradient_on_an_empty_block():
    assert mlp_input_gradients(_Linear(np.zeros((2, 3))), np.zeros((0, 3)), target=0).shape == (0, 3)


# --------------------------------------------------------------- the detector
def _planted(n=600, d=6, n_poison=120, seed=0):
    """Slopes with one distinct sub-population, the structure the detector must find."""
    rng = np.random.default_rng(seed)
    s = rng.normal(scale=0.1, size=(n, d))
    is_poison = np.zeros(n, dtype=bool)
    is_poison[:n_poison] = True
    s[:n_poison] += 4.0
    return s, is_poison


def test_a_distinct_slope_cluster_is_ranked_above_the_bulk():
    s, is_poison = _planted()
    sc = active_path_scores(s, seed=0, min_cluster_size=20)
    assert sc.shape == (s.shape[0],)
    assert sc[is_poison].mean() > sc[~is_poison].mean()


def test_scores_are_finite_and_non_negative():
    s, _ = _planted(seed=1)
    sc = active_path_scores(s, seed=0, min_cluster_size=20)
    assert np.isfinite(sc).all()
    assert (sc >= 0).all(), "the score is a distance, so it cannot be negative"


def test_rows_in_one_cluster_share_a_score():
    """A row inherits its cluster's distance, which is what makes the ranking well defined."""
    s, is_poison = _planted(seed=2)
    sc = active_path_scores(s, seed=0, min_cluster_size=20)
    assert len(np.unique(sc)) < len(sc), "every row scoring differently means no cluster inheritance"


def test_deterministic_for_a_seed():
    s, _ = _planted(seed=3)
    a = active_path_scores(s, seed=7, min_cluster_size=20)
    b = active_path_scores(s, seed=7, min_cluster_size=20)
    assert np.array_equal(a, b)


def test_a_constant_slope_matrix_degenerates_rather_than_inventing_structure():
    """The pre-registration names this: if slopes are near-constant the method degenerates before
    clustering, and that is a statement about the victim. It must not crash or fabricate a split."""
    sc = active_path_scores(np.ones((300, 5)), seed=0, min_cluster_size=20)
    assert np.isfinite(sc).all()
    assert np.allclose(sc, sc[0]), "a constant input cannot license a non-constant score"


def test_slope_matrix_reports_degeneracy():
    flat = slope_matrix(np.ones((200, 4)))
    varied, _ = _planted(seed=4)
    assert flat["degenerate"] is True
    assert slope_matrix(varied)["degenerate"] is False


def test_too_few_rows_to_cluster_returns_zeros_rather_than_raising():
    sc = active_path_scores(np.random.default_rng(5).normal(size=(5, 3)), seed=0,
                            min_cluster_size=20)
    assert sc.shape == (5,)
    assert np.isfinite(sc).all()
