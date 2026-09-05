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
DEFAULT_FRR = 0.01          # Gao et al.'s headline operating point: 1% false rejection


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


def fit_strip_entropy_cutoff(
    clean_scores: np.ndarray,
    frr: float = DEFAULT_FRR,
    *,
    partition: str,
    contains_poison: bool = False,
    return_diagnostics: bool = False,
):
    """Fit Gao et al.'s entropy threshold at a target false rejection rate, on clean rows only.

    STRIP's published rule is not a poison budget. The defender picks an FRR they can live
    with, takes that percentile of the entropy observed on a clean held-out pool, and flags
    anything below it. At frr=0.01 exactly 1% of clean traffic is rejected by construction,
    and the poison recall that follows is a measurement rather than a quantity the budget
    already fixed. This is the only rule STRIP publishes, and until now it was the one rule
    STRIP was never scored under in this project.

    `clean_scores` are `strip_scores` outputs, i.e. NEGATED mean entropy. The returned cutoff
    is on the entropy scale (positive for a normal clean pool), because that is what
    `strip_threshold_flag` consumes: entropy_cut = quantile(entropy, frr)
    = quantile(-clean_scores, frr) = -quantile(clean_scores, 1 - frr).

    The leakage guard follows `src.precision_triage.fit_precision_cutoff`. A threshold fit on
    the same rows it is later evaluated against, or on rows carrying poison, manufactures its
    own answer, and a mislabelled partition string and a genuinely contaminated pool are two
    distinct ways in --- so either one alone refuses, not merely both together.
    """
    if partition != "clean_calibration" or contains_poison:
        raise ValueError(
            f"fit_strip_entropy_cutoff refuses to fit on partition={partition!r} "
            f"contains_poison={contains_poison!r} -- the FRR cutoff may ONLY be fit on clean, "
            "held-out rows disjoint from the rows being scored (partition='clean_calibration', "
            "contains_poison=False). Fitting on the scored block, or on a pool that merely "
            "claims to be clean, leaks the evaluation into its own threshold."
        )
    arr = np.asarray(clean_scores, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(
            "fit_strip_entropy_cutoff: empty clean calibration pool -- cannot fit a percentile "
            "on nothing"
        )
    if not np.isfinite(arr).all():
        raise ValueError(
            "fit_strip_entropy_cutoff: clean calibration scores must all be finite; a NaN or "
            "inf here propagates silently into the cutoff and disables the detector"
        )
    if not (np.isfinite(frr) and 0.0 < frr < 1.0):
        raise ValueError(f"fit_strip_entropy_cutoff: frr must lie in (0, 1), got {frr!r}")

    cut = float(-np.quantile(arr, 1.0 - frr))
    if not return_diagnostics:
        return cut

    # A requested FRR is not always achievable. When the score has a point mass at the cutoff ---
    # which STRIP's clamped-softmax floor guarantees on a confident victim --- the achievable
    # rates jump over the request, and the caller needs to know the rate it actually got rather
    # than the one it asked for.
    entropy = -arr
    tie_mass = float(np.mean(entropy == cut))
    strict = float(np.mean(entropy < cut))
    realized = float(np.mean(entropy <= cut))
    diagnostics = dict(
        cutoff=cut,
        requested_frr=float(frr),
        realized_frr=realized,
        strict_frr=strict,
        tie_mass_at_cutoff=tie_mass,
        n_calibration=int(arr.size),
        degenerate=bool(tie_mass > 0.0 and strict < frr and realized > frr * 1.05),
    )
    return cut, diagnostics


def calibrate_strip_cutoff(
    model,
    blend_pool_std: np.ndarray,
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    alpha: float = DEFAULT_ALPHA,
    device: str = "cpu",
    seed: int = 0,
    n_calib: int = 2048,
    frr: float = DEFAULT_FRR,
    return_diagnostics: bool = False,
):
    """Fit the FRR cutoff from the clean pool a caller already holds, in ONE code path.

    Every runner in this project has a clean, held-out pool on hand --- the same one it hands
    STRIP as the blend reference. This function carves a calibration slice out of it, scores
    that slice against the REMAINING pool rows, and fits the cutoff. Two properties matter:

    - The calibration queries are disjoint from the blend references used to score them, so no
      query is ever blended with itself. Self-blending returns the query unchanged, collapses
      its entropy, and would drag the cutoff down --- weakening the detector in a way that
      looks like a property of STRIP.
    - The caller's own scoring path is untouched. It still passes the full pool to
      `strip_scores`, so the cutoff is the only thing this adds and committed numbers do not
      move.

    Every runner calls this rather than rolling its own slice, because two scripts computing
    the same quantity along two paths is how this project previously got 0.9828 and 0.9814 for
    one number.
    """
    pool = np.asarray(blend_pool_std, dtype=np.float32)
    n_pool = pool.shape[0]
    if n_pool < 8:
        raise ValueError(
            f"calibrate_strip_cutoff: clean pool has {n_pool} rows, too few to split into a "
            "calibration slice and a disjoint blend reference"
        )
    k = int(min(n_calib, n_pool // 2))
    if k < 1:
        raise ValueError(f"calibrate_strip_cutoff: n_calib={n_calib} resolves to {k} rows")

    order = np.random.default_rng(seed).permutation(n_pool)
    calib_idx, ref_idx = order[:k], order[k:]
    clean_scores = strip_scores(
        model, pool[calib_idx], pool[ref_idx],
        n_trials=n_trials, alpha=alpha, device=device, seed=seed,
    )
    return fit_strip_entropy_cutoff(
        clean_scores, frr=frr, partition="clean_calibration", contains_poison=False,
        return_diagnostics=return_diagnostics,
    )


def strip_threshold_flag(scores: np.ndarray, entropy_cut: float) -> np.ndarray:
    """Gao et al.'s original decision rule: flag samples whose entropy falls at or below a threshold
    fit on a clean held-out set (their FRR target), rather than an assumed poison budget.

    `scores` are negated entropy, so the flag is `scores >= -entropy_cut`.

    The comparison is NON-STRICT, and that is load-bearing rather than a detail. STRIP's entropy
    comes from a clamped softmax, so a fully saturated prediction lands on an exact numerical floor
    of `1e-12 * -log(1e-12)`, about 2.7631e-11. On a trained victim a large share of rows saturate:
    measured on the CTU-13 positive control, 2.88% of clean calibration rows, 3.59% of scored clean
    rows and 99.39% of poisoned rows sit exactly on that floor. A 1% false-rejection target
    therefore falls INSIDE the tie, the fitted cutoff IS the floor, and a strict `<` comparison can
    never fire because nothing lies below a floor. That returned recall 0.0000 at FPR 0.0000 while
    the same scores ranked poison at AUC 0.9839 --- a fabricated negative result, not a finding.

    Non-strict is also the correct reading of an empirical quantile in general. For a continuous
    score the two agree, since the probability of an exact tie is zero.
    """
    return np.asarray(scores) >= -entropy_cut
