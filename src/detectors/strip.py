"""STRIP (Gao et al., ACSAC 2019): perturb-and-blend entropy test.

A backdoored model's output on a triggered input is dominated by the trigger, not the input's own
content. STRIP tests this directly: superimpose each query sample onto many random references drawn
from a clean pool, run the model on each blend, and measure the entropy of the averaged prediction
distribution. A clean sample's prediction is unstable under blending (each blend looks like a mostly-
different, ambiguous input, so entropy stays high); a triggered sample keeps predicting the trigger's
target class regardless of what it is blended with, so entropy collapses.

This module returns per-sample scores oriented like every other detector in this codebase (higher =
more poison-like), so both existing decision rules in `src/detectors/spectral.py` apply unchanged:

    scores = strip_scores(model, x_std, blend_pool_std)
    flag_by_scores(scores, expected_frac=...)
    flag_by_mad_threshold(scores, z_thresh=...)

STRIP's own statistic is entropy, where LOW entropy is the anomaly; `strip_scores` therefore returns
negated mean entropy, so "higher score" still means "more poison-like" everywhere else in the codebase.

Faithfulness, stated the same way as this project's other detector substitutions:

  - FAITHFUL: superimpose-and-perturb, n_trials blends per query drawn from a clean reference pool,
    softmax the model's output, average the per-trial predicted-class distribution, score by its
    entropy (Gao et al. use Shannon entropy of the boundary-summed softmax; so does this module).
  - SUBSTITUTED: Gao et al. blend two IMAGES by pixel-wise superposition (`a*x + (1-a)*x_ref`, then
    renormalise so pixel values stay in [0, 1]). NetFlow features have no shared bounded range across
    columns, so this module blends in the model's STANDARDISED input space (`a*x + (1-a)*x_ref`, no
    renormalisation needed since standardised features are unbounded but centred) and does NOT project
    the blend back onto TabularBench's constraint set. That projection exists to keep an ATTACKER's
    trigger physically realizable on the wire; STRIP's blend is an internal defender-side computation
    that is never emitted as a flow, so realizability does not apply to it. Blend ratio `alpha` is
    fixed at 0.5 (Gao et al.'s default "half-half" superposition), not swept.

No randomness beyond the reference-pool draw, which is seeded by the caller for determinism.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_N_TRIALS = 64
DEFAULT_ALPHA = 0.5


@torch.no_grad()
def _softmax_logits(model, x: torch.Tensor, device: str) -> torch.Tensor:
    model = model.to(device).eval()
    return F.softmax(model(x.to(device)), dim=1).cpu()


def strip_scores(
    model,
    x_std: np.ndarray,
    blend_pool_std: np.ndarray,
    n_trials: int = DEFAULT_N_TRIALS,
    alpha: float = DEFAULT_ALPHA,
    device: str = "cpu",
    seed: int = 0,
    batch_size: int = 64,
) -> np.ndarray:
    """Per-sample STRIP score (higher = more poison-like). `model` must be a `torch.nn.Module`
    exposing `forward(x) -> logits`; GBDT victims have no differentiable/probabilistic output STRIP
    can use in the same way and are out of scope for this module (same H4 limit as Spectral/AC)."""
    x = np.asarray(x_std, dtype=np.float32)
    pool = np.asarray(blend_pool_std, dtype=np.float32)
    n = x.shape[0]
    sample_shape = x.shape[1:]                                    # works for (D,) or image (C,H,W)
    rng = np.random.default_rng(seed)
    ref_idx = rng.integers(0, pool.shape[0], size=(n, n_trials))

    x_t = torch.as_tensor(x)
    pool_t = torch.as_tensor(pool)

    mean_entropy = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bs = end - start
        base = x_t[start:end].unsqueeze(1)                        # (bs, 1, *sample_shape)
        refs = pool_t[torch.as_tensor(ref_idx[start:end])]        # (bs, n_trials, *sample_shape)
        blend = alpha * base + (1 - alpha) * refs                 # (bs, n_trials, *sample_shape)
        flat = blend.reshape(bs * n_trials, *sample_shape)
        probs = _softmax_logits(model, flat, device)
        probs = probs.reshape(bs, n_trials, -1).mean(dim=1)       # averaged prediction per query
        ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
        mean_entropy[start:end] = ent.numpy()

    return -mean_entropy


def strip_threshold_flag(scores: np.ndarray, entropy_cut: float) -> np.ndarray:
    """Gao et al.'s original decision rule: flag samples whose entropy falls below a threshold fit
    on a clean held-out set (their FRR target, typically the 1st percentile of clean entropy), rather
    than an assumed poison budget. `scores` are negated entropy, so the flag is `scores > -entropy_cut`."""
    return np.asarray(scores) > -entropy_cut
