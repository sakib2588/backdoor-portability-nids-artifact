import numpy as np
import torch

from src.detectors.neural_cleanse import (
    anomaly_index, balanced_inversion_sample, reverse_engineer, reverse_engineer_tabular,
)


def test_anomaly_index_flags_small_outlier():
    # one class has an anomalously small reverse-engineered trigger
    norms = np.array([10.0, 11.0, 9.5, 10.5, 1.0, 10.2, 9.8, 10.1, 9.9, 10.3])
    index, flagged = anomaly_index(norms)
    assert flagged == 4          # the small-norm class
    assert index > 2.0           # Neural Cleanse detection threshold


def test_anomaly_index_clean_has_low_index():
    norms = np.array([10.0, 10.1, 9.9, 10.2, 9.8, 10.05, 9.95, 10.1, 9.9, 10.0])
    index, _ = anomaly_index(norms)
    assert index < 2.0


def test_reverse_engineer_returns_masks_when_asked():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 3))
    x = torch.rand(8, 1, 4, 4)
    norms, masks = reverse_engineer(model, 3, x, steps=2, return_masks=True)
    assert norms.shape == (3,)
    assert masks.shape == (3, 4, 4)          # one (h, w) mask per candidate class
    assert ((masks >= 0) & (masks <= 1)).all()   # post-sigmoid


def test_reverse_engineer_norms_match_returned_mask_l1():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 3))
    x = torch.rand(8, 1, 4, 4)
    norms, masks = reverse_engineer(model, 3, x, steps=2, return_masks=True)
    np.testing.assert_allclose(norms, np.abs(masks).sum(axis=(1, 2)), rtol=1e-5)


def test_reverse_engineer_without_return_masks_is_unchanged():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 3))
    x = torch.rand(8, 1, 4, 4)
    norms = reverse_engineer(model, 3, x, steps=2)
    assert isinstance(norms, np.ndarray) and norms.shape == (3,)


def test_reverse_engineer_tabular_returns_masks_when_asked():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(6, 2))
    x = torch.randn(10, 6)
    norms, masks = reverse_engineer_tabular(model, 2, x, steps=2, return_masks=True)
    assert norms.shape == (2,)
    assert masks.shape == (2, 6)             # one per-feature mask per class
    assert ((masks >= 0) & (masks <= 1)).all()


# --- balanced_inversion_sample: the sample Neural Cleanse's inversion is run on ---------------
# See notes/20260717-decision-nc-inversion-sample-fix.md. Each class's inversion needs rows NOT
# already in that class, or its own objective is already satisfied and only the L1 term survives.

def _toy_xy(n_benign, n_botnet, d=4):
    """Rows carry their class in every feature, so a sample's composition is readable off the values."""
    x = np.concatenate([np.zeros((n_benign, d)), np.ones((n_botnet, d))]).astype(np.float32)
    y = np.concatenate([np.zeros(n_benign, dtype=int), np.ones(n_botnet, dtype=int)])
    return x, y


def test_balanced_inversion_sample_returns_requested_size():
    x, y = _toy_xy(500, 500)
    out = balanced_inversion_sample(x, y, 400, seed=42)
    assert out.shape == (400, 4)
    assert out.dtype == torch.float32


def test_balanced_inversion_sample_balances_a_98_2_imbalance():
    # the CTU-13 case: benign dominates ~98/2, yet class 1's inversion still needs class-0 rows
    x, y = _toy_xy(9800, 200)
    out = balanced_inversion_sample(x, y, 400, seed=42).numpy()
    n_botnet = int((out[:, 0] == 1).sum())
    assert n_botnet == 200            # n // n_classes of each, not the 98/2 population ratio
    assert len(out) - n_botnet == 200


def test_balanced_inversion_sample_is_deterministic_for_a_fixed_seed():
    x, y = _toy_xy(500, 500)
    a = balanced_inversion_sample(x, y, 400, seed=42)
    b = balanced_inversion_sample(x, y, 400, seed=42)
    torch.testing.assert_close(a, b)


def test_balanced_inversion_sample_takes_all_rows_of_a_short_class():
    # class 1 holds fewer than n // 2 rows -- take all of it; the sample is then as balanced as
    # the data permits, and is short of `n` rather than padded with the majority class
    x, y = _toy_xy(500, 30)
    out = balanced_inversion_sample(x, y, 400, seed=42).numpy()
    assert int((out[:, 0] == 1).sum()) == 30    # every botnet row available
    assert int((out[:, 0] == 0).sum()) == 200   # n // 2 benign
    assert len(out) == 230
