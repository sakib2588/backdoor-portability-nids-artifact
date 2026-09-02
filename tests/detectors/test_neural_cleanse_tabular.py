"""Task 0b control: the tabular Neural Cleanse objective + binary decision rule.

The MAD `anomaly_index` is degenerate on 2 classes (always 0.6745, never > 2), so binary
NIDS needs `nc_binary_flag` (a mask-L1 ratio test). These tests pin: (i) the ratio rule flags
the anomalously-small-mask class and separates backdoored from clean; (ii) `reverse_engineer_tabular`
on an actually-backdoored tiny MLP recovers a small mask for the backdoored target class, so an NC
null on the real trigger will be interpretable rather than an artefact of an ill-posed objective.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.detectors.neural_cleanse import nc_binary_flag, reverse_engineer_tabular
from src.vision_control import set_seed


def test_nc_binary_flag_ratio_rule():
    # backdoored: class 0 needs a much smaller trigger than class 1
    flagged, ratio = nc_binary_flag(np.array([1.0, 10.0]), tau=2.0)
    assert flagged == 0
    assert ratio >= 2.0
    # clean: both classes need a similar-size trigger -> ratio near 1, below tau
    _, clean_ratio = nc_binary_flag(np.array([10.0, 10.5]), tau=2.0)
    assert clean_ratio < 2.0


def _make_data(seed, with_trigger_poison):
    """2-class synthetic tabular data engineered to expose NC's small-mask signal.

    The genuine class boundary is BROAD and individually-weak: 16 features (0..15) each contribute a
    small margin, so forcing a class the natural way needs a large (many-feature) mask. The trigger
    is a compact 2-feature channel (features 16,17), whose clean spread is wide enough that its
    in-range trigger value is reverse-engineerable. If with_trigger_poison, samples with the trigger
    are labelled class 0 (target), so the model learns a small (2-feature) shortcut to class 0 ---
    exactly the small-trigger asymmetry Neural Cleanse detects."""
    rng = np.random.default_rng(seed)
    n, d = 500, 20
    x = rng.normal(0, 0.5, size=(n, d)).astype(np.float32)
    y = (rng.random(n) < 0.5).astype(np.int64)
    x[:, 0:16] += np.where(y == 1, 0.8, -0.8)[:, None]           # broad weak 16-feature signature
    x[:, 16:18] = rng.normal(0, 1.5, size=(n, 2)).astype(np.float32)  # wide 2-feature trigger channel
    if with_trigger_poison:
        idx = rng.choice(np.where(y == 1)[0], size=90, replace=False)
        x[idx, 16] = 4.0         # compact in-range trigger
        x[idx, 17] = 4.0
        y[idx] = 0               # trigger forces the target class 0
    return torch.from_numpy(x), torch.from_numpy(y)


def _train_tiny_mlp(x, y, seed):
    set_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], 32), nn.ReLU(), nn.Linear(32, 2))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(150):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    return model


def test_reverse_engineer_tabular_catches_backdoor_not_clean():
    seed = 42
    tau = 2.0
    x_bd, y_bd = _make_data(seed, with_trigger_poison=True)
    x_cl, y_cl = _make_data(seed, with_trigger_poison=False)

    bd_model = _train_tiny_mlp(x_bd, y_bd, seed)
    cl_model = _train_tiny_mlp(x_cl, y_cl, seed)

    # invert on clean (unpoisoned) inputs, mirroring the real pipeline
    sample = x_cl
    bd_norms = reverse_engineer_tabular(bd_model, 2, sample, steps=300, l1_weight=0.05)
    cl_norms = reverse_engineer_tabular(cl_model, 2, sample, steps=300, l1_weight=0.05)

    bd_flagged, bd_ratio = nc_binary_flag(bd_norms, tau=tau)
    _, cl_ratio = nc_binary_flag(cl_norms, tau=tau)

    assert bd_flagged == 0                 # the backdoored target class (smallest mask)
    assert bd_ratio >= tau                 # backdoor exceeds the detection threshold
    assert cl_ratio < tau                  # clean model does NOT trip the threshold
    assert bd_ratio > cl_ratio             # and the backdoor is clearly more asymmetric


# --- NC-10 rate-mechanism helpers (scripts/18_nc_rate_mechanism.py) ----------------------------
# The script owns these because they are reporting helpers, not detector logic. Loaded by path
# rather than imported as a package, since scripts/ is not one.

def _nc10():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "scripts" / "18_nc_rate_mechanism.py"
    spec = importlib.util.spec_from_file_location("_nc10", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trigger_share_chance_is_trigger_width_over_feature_count():
    """Pre-registered verbatim in the NC-10 task. The 0.0211 null this project reports for a
    width-16 trigger over 757 features is exactly this ratio, not a fitted constant."""
    assert _nc10().trigger_share_chance(trigger_width=16, n_features=757) == pytest.approx(16 / 757)


def test_mask_concentration_is_one_for_a_perfectly_compact_mask():
    m = np.zeros(757)
    m[:16] = 1.0
    assert _nc10().mask_concentration(m, top_k=16) == pytest.approx(1.0)


def test_mask_concentration_of_a_uniform_mask_equals_chance():
    """A mask with no structure puts exactly top_k/D of its mass on any top_k slice -- the same
    number trigger_share_chance returns, which is why a concentration at chance means the inversion
    resolved nothing."""
    m = np.ones(757)
    assert _nc10().mask_concentration(m, top_k=16) == pytest.approx(16 / 757)


def test_mask_cosine_is_one_when_the_mask_is_the_trigger_indicator():
    m = np.zeros(757)
    idx = [3, 11, 40]
    m[idx] = 1.0
    assert _nc10().mask_cosine_to_true_trigger(m, idx) == pytest.approx(1.0)


def test_mask_cosine_is_zero_when_the_mask_avoids_the_trigger():
    m = np.zeros(757)
    m[[100, 200]] = 1.0
    assert _nc10().mask_cosine_to_true_trigger(m, [3, 11]) == pytest.approx(0.0)
