"""FTTransformerVictim must be a drop-in for MLPVictim everywhere the NC pilot touches a victim:
same two methods, a live gradient path from the input, and permutation equivariance over features
(the property that justified choosing a transformer over a 1D-CNN)."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.models import FTTransformerVictim, MLPVictim, train_ft_transformer


def test_interface_parity_with_mlp_victim():
    d, n = 32, 8
    ft = FTTransformerVictim(in_dim=d, d_token=16, n_layers=2, n_heads=4)
    mlp = MLPVictim(in_dim=d)
    x = torch.randn(n, d)
    for m in (ft, mlp):
        assert hasattr(m, "features") and hasattr(m, "forward")
        assert m(x).shape == (n, 2)
        assert m.features(x).ndim == 2 and m.features(x).shape[0] == n
    assert ft.features(x).shape == (n, 16)          # penultimate width is d_token


def test_gradient_reaches_the_input():
    """Neural Cleanse optimises a mask through the victim. No input gradient, no inversion --
    this is exactly what disqualifies a tree ensemble (see notes/20260902-experiment-nc-gradient-
    barrier-trees.md), so the transformer must demonstrably NOT have that property."""
    ft = FTTransformerVictim(in_dim=32, d_token=16, n_layers=2, n_heads=4)
    x = torch.randn(4, 32, requires_grad=True)
    torch.nn.functional.cross_entropy(ft(x), torch.zeros(4, dtype=torch.long)).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad != 0).any()


def test_permutation_equivariance_over_features():
    """Permuting the input columns AND the per-feature tokenizer rows by the same permutation must
    leave the output unchanged. This is the property a 1D-CNN lacks, and the reason a null on this
    victim is attributable to architecture rather than to an injected locality bias."""
    d = 24
    ft = FTTransformerVictim(in_dim=d, d_token=16, n_layers=2, n_heads=4).eval()
    x = torch.randn(6, d)
    perm = torch.randperm(d, generator=torch.Generator().manual_seed(0))
    before = ft(x)
    with torch.no_grad():
        ft.w.copy_(ft.w[perm])
        ft.b.copy_(ft.b[perm])
    after = ft(x[:, perm])
    assert torch.allclose(before, after, atol=1e-5)


def test_training_is_seed_deterministic():
    x = np.random.RandomState(0).randn(256, 16).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    a = train_ft_transformer(x, y, seed=42, epochs=2, device="cpu", batch_size=64)
    b = train_ft_transformer(x, y, seed=42, epochs=2, device="cpu", batch_size=64)
    xa = torch.as_tensor(x)
    assert torch.allclose(a(xa), b(xa), atol=1e-6)


def test_parameters_stay_fp32_after_bf16_training():
    """bf16 is an autocast context, not a parameter dtype. The inversion runs OUTSIDE any autocast
    context and must therefore see fp32 weights -- this is what keeps the measured mask in the same
    precision as the committed MLP arm."""
    x = np.random.RandomState(1).randn(128, 16).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    m = train_ft_transformer(x, y, seed=7, epochs=1, device="cpu", batch_size=32)
    assert all(p.dtype == torch.float32 for p in m.parameters())


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8",
    reason="needs CUDA and CUBLAS_WORKSPACE_CONFIG=:4096:8 set before the process started")
def test_cuda_bf16_training_keeps_fp32_parameters_and_is_reproducible():
    """Exercises the branch the CPU test cannot reach, and checks both halves of the contract the
    experiment depends on: parameters stay fp32 (so the Neural Cleanse inversion runs at the same
    precision as the committed MLP arm) and two same-seed runs agree bitwise (so the 102-model
    null is reproducible)."""
    # 757 features, not a small width: this is CTU-13's real dimensionality, and it is also the
    # scale at which the bug this test guards actually appears. Measured with the determinism fix
    # disabled, two same-seed runs are bitwise identical at 24 and 128 features (so the test would
    # pass with the fix removed and prove nothing) but diverge by 2.8e-04 at 256 and 9.0e-04 at 757.
    # Do not shrink this for speed -- two trainings at this width cost about 0.4 s total, and a
    # narrower input silently converts this test back into a no-op.
    x = np.random.RandomState(2).randn(512, 757).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    a = train_ft_transformer(x, y, seed=11, epochs=2, device="cuda", batch_size=128)
    b = train_ft_transformer(x, y, seed=11, epochs=2, device="cuda", batch_size=128)
    assert all(p.dtype == torch.float32 for p in a.parameters())
    xa = torch.as_tensor(x, device="cuda")
    with torch.no_grad():
        assert torch.equal(a(xa), b(xa)), "same-seed CUDA runs must be bitwise identical"


def test_inversion_path_sees_fp32_activations():
    """The Neural Cleanse inversion must not run under autocast.

    Training may be bf16 -- it is only a means of producing the victim -- but the inverted mask IS
    the quantity the experiment reports, and it has to be computed at the same precision as the
    committed MLP arm or a difference in the result could be precision rather than architecture.
    Asserted on the dtype the model actually produces when called the way the pilot calls it."""
    m = FTTransformerVictim(in_dim=32, d_token=16, n_layers=2, n_heads=4).eval()
    x = torch.randn(8, 32)
    assert m(x).dtype == torch.float32
    assert m.features(x).dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to exercise bf16 autocast")
def test_autocast_would_have_changed_the_dtype_so_the_guard_is_meaningful():
    """A guard that cannot fail proves nothing. This shows the assertion above is load-bearing by
    demonstrating the dtype really does change under autocast, and reverts outside it."""
    m = FTTransformerVictim(in_dim=32, d_token=16, n_layers=2, n_heads=4).cuda().eval()
    x = torch.randn(8, 32, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert m(x).dtype == torch.bfloat16
    assert m(x).dtype == torch.float32
