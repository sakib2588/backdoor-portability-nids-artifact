"""Neural Cleanse (Wang et al., IEEE S&P 2019).

For each candidate target class, reverse-engineer the minimal trigger (mask +
pattern) that forces that class. A backdoored class needs an anomalously SMALL
trigger; a MAD-based anomaly index over per-class mask L1 norms above 2 flags it.

The MAD `anomaly_index` is well-posed only with several classes (vision: 10). On a
BINARY NIDS task it is mathematically degenerate --- with two classes the index is a
constant 0.6745 regardless of the data, so it can never exceed the >2 threshold. For
the binary tabular case use `reverse_engineer_tabular` (a mask/pattern objective over
standardised feature deltas) plus `nc_binary_flag` (a mask-L1 ratio test calibrated on
clean models), NOT `anomaly_index`. See notes/20260714-decision-m2-critical-review-resolutions.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _invert_trigger(
    model: torch.nn.Module,
    num_classes: int,
    sample_x: torch.Tensor,
    mask_shape: tuple,
    pattern_shape: tuple,
    pattern_fn,
    device: str,
    steps: int,
    lr: float,
    l1_weight: float,
    return_masks: bool = False,
) -> np.ndarray:
    """Shared Neural Cleanse inversion loop: per candidate target class, optimise a sigmoid mask +
    a (domain-specific) pattern to minimise cross-entropy-toward-target plus an L1 mask penalty.
    `pattern_fn` maps the free `pattern` parameter to the domain's valid range (sigmoid->[0,1] pixels
    for vision, an affine-bounded sigmoid for standardised tabular features) -- this is the only axis
    reverse_engineer (vision) and reverse_engineer_tabular (tabular) differ on, so it is the sole
    parameter threaded through; everything else (optimiser, loss, mask norm) is shared, one place to fix.

    `return_masks` additionally returns the converged per-class sigmoid masks (shape
    `(num_classes, *mask_shape)`), which the inversion otherwise discards -- retained so the
    reconstructed triggers can be inspected qualitatively (scripts/10_nc_masks.py), not just
    reduced to their L1 norm.
    """
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sample_x = sample_x.to(device)
    norms = np.zeros(num_classes, dtype=np.float64)
    masks = np.zeros((num_classes, *mask_shape), dtype=np.float64) if return_masks else None

    for target in range(num_classes):
        mask = torch.zeros(*mask_shape, device=device, requires_grad=True)
        pattern = torch.zeros(*pattern_shape, device=device, requires_grad=True)
        opt = torch.optim.Adam([mask, pattern], lr=lr)
        y_t = torch.full((sample_x.shape[0],), target, dtype=torch.long, device=device)
        for _ in range(steps):
            m = torch.sigmoid(mask)
            p = pattern_fn(pattern)
            blended = (1 - m) * sample_x + m * p
            opt.zero_grad()
            loss = F.cross_entropy(model(blended), y_t) + l1_weight * m.abs().sum()
            loss.backward()
            opt.step()
        final_mask = torch.sigmoid(mask).detach()
        norms[target] = float(final_mask.abs().sum().item())
        if return_masks:
            masks[target] = final_mask.cpu().numpy()
    return (norms, masks) if return_masks else norms


def reverse_engineer(
    model: torch.nn.Module,
    num_classes: int,
    sample_x: torch.Tensor,
    device: str = "cpu",
    steps: int = 200,
    lr: float = 0.1,
    l1_weight: float = 0.001,
    return_masks: bool = False,
):
    _, c, h, w = sample_x.shape
    out = _invert_trigger(
        model, num_classes, sample_x, mask_shape=(1, h, w), pattern_shape=(1, c, h, w),
        pattern_fn=torch.sigmoid,          # [0,1] per pixel
        device=device, steps=steps, lr=lr, l1_weight=l1_weight, return_masks=return_masks)
    if not return_masks:
        return out
    norms, masks = out
    return norms, masks.reshape(num_classes, h, w)   # drop the singleton channel axis


def anomaly_index(mask_norms: np.ndarray) -> tuple[float, int]:
    norms = np.asarray(mask_norms, dtype=np.float64)
    median = np.median(norms)
    mad = np.median(np.abs(norms - median))
    consistency = 1.4826  # MAD -> std for a normal distribution
    flagged = int(np.argmin(norms))
    if mad == 0:
        return 0.0, flagged
    index = float(abs(norms[flagged] - median) / (consistency * mad))
    return index, flagged


def reverse_engineer_tabular(
    model: torch.nn.Module,
    num_classes: int,
    sample_x_std: torch.Tensor,
    device: str = "cpu",
    steps: int = 300,
    lr: float = 0.1,
    l1_weight: float = 0.001,
    bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    return_masks: bool = False,
):
    """Neural Cleanse trigger inversion for a tabular victim (standardised space).

    Reverse-engineer, per candidate target class, the minimal feature-space trigger
    (mask `m in [0,1]^D` + free pattern `p in R^D`) that forces that class, applied as
    `x' = (1 - m) * x + m * p` in STANDARDISED feature space (features are z-scored, so
    the pattern is unbounded, unlike the [0,1] pixel case). Returns per-class `L1(m*)`.
    A backdoored target class yields an anomalously small mask. Decide with
    `nc_binary_flag` (a ratio test), NOT the MAD `anomaly_index`, when num_classes == 2.
    """
    sample_x_std = sample_x_std.to(device)
    d = sample_x_std.shape[1]
    # Bound the reverse-engineered pattern to each feature's valid range (the standardised analogue
    # of Neural Cleanse's [0,1] pixel bound). Without a bound, one unbounded feature can trivially
    # flip a near-linear tabular boundary, collapsing every class's mask to ~1 feature and destroying
    # the small-trigger signal. Default bound = the inversion sample's per-feature min/max; the real
    # pipeline should pass `bounds` derived from the constraint box (metadata min/max standardised),
    # which is wider than observed clean data and contains a realizable watermark trigger.
    # p = lo + (hi - lo) * sigmoid(pattern).
    if bounds is None:
        lo = sample_x_std.min(dim=0).values
        hi = sample_x_std.max(dim=0).values
    else:
        lo, hi = bounds[0].to(device), bounds[1].to(device)
    span = (hi - lo).clamp_min(1e-6)
    return _invert_trigger(
        model, num_classes, sample_x_std, mask_shape=(d,), pattern_shape=(d,),
        pattern_fn=lambda pattern: lo + span * torch.sigmoid(pattern),  # bounded to feature range
        device=device, steps=steps, lr=lr, l1_weight=l1_weight, return_masks=return_masks)


def balanced_inversion_sample(x_std, y, n: int, seed: int) -> torch.Tensor:
    """Draw a class-balanced inversion sample: `n // n_classes` seeded-random rows per class.

    WHY THIS EXISTS. Neural Cleanse reverse-engineers class `yt`'s trigger by pushing samples that
    are NOT already `yt` across the boundary into `yt`. Feed it a sample already conditioned on `yt`
    and `yt`'s own inversion is degenerate: cross-entropy toward `yt` starts at ~0, no gradient
    reaches the mask, and only the uniform L1 penalty acts on it, so the mask collapses evenly to a
    value fixed by (steps, lr, l1_weight) alone -- independent of the model, the poison, and the seed.
    That is exactly what the tabular arm measured before this fix: the flagged-class mask was the
    single value 0.002383 repeated across all 757 features, byte-identical on every one of the 5
    seeds (results/nc_masks.json). It is not a measurement of anything.

    The vision arm never hit this because it inverts from `x_te[:NC_SAMPLE]`, the whole 10-class test
    set, so ~90% of every class's sample is non-target and each inversion has a real objective. A
    BINARY task has no such free lunch and must construct the sample: at 2 classes, ~50% non-target
    is the closest analogue available, and it gives BOTH classes a well-posed inversion.

    Balanced, not population-proportional, on purpose. CTU-13 is ~98/2 benign/botnet, so a
    proportional draw would leave class 1's inversion with ~2% of rows to work from and reintroduce a
    weaker form of the same starvation. Falls back gracefully when a class holds fewer than
    `n // n_classes` rows: take all of them, leaving the sample as balanced as the data permits and
    short of `n` rather than padded with the majority class.

    See notes/20260717-decision-nc-inversion-sample-fix.md for the diagnosis and the before/after.
    """
    x_std = np.asarray(x_std)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, n // len(classes))

    picks = []
    for c in classes:
        pos = np.where(y == c)[0]
        take = min(per_class, len(pos))          # short class: take all of it, do not top up elsewhere
        picks.append(rng.choice(pos, size=take, replace=False))
    idx = np.sort(np.concatenate(picks))          # sorted: sample order is row order, not class order
    return torch.as_tensor(x_std[idx], dtype=torch.float32)


def nc_binary_flag(mask_norms: np.ndarray, tau: float) -> tuple[int, float]:
    """Binary Neural Cleanse decision rule (replaces the degenerate MAD index at n=2).

    `flagged_class = argmin(mask_norms)` (the class needing the smallest trigger);
    `ratio = max(mask_norms) / min(mask_norms)`. Flag iff `ratio >= tau`, where `tau`
    is calibrated on the null distribution of `ratio` from CLEAN-trained models (a
    backdoored class produces an anomalously small mask, hence a large ratio). Returns
    `(flagged_class, ratio)`; the caller compares `ratio` to `tau`.
    """
    norms = np.asarray(mask_norms, dtype=np.float64)
    flagged = int(np.argmin(norms))
    lo = float(norms.min())
    hi = float(norms.max())
    if lo <= 0.0:
        return flagged, float("inf")
    return flagged, hi / lo


# --------------------------------------------------------------------------------------------------
# Calibrated / multistart repair attempt (plan Task 7A)
# --------------------------------------------------------------------------------------------------
# The baseline binary rule -- single-start inversion from a zeros init, hardcoded tau=2.0 -- detects
# 11 of 30 tabular cells (0.3667) against 1.0 on vision, and its flagged masks recover the true
# trigger features at 0.0319 against a 0.0211 chance level (results/nc_masks.json). Three root causes
# could each produce that, and they are tested SEPARATELY because a combined "improved NC" run would
# be uninterpretable:
#
#   multistart        optimisation local minima smear an otherwise compact trigger
#   sparser           the L1 mask penalty under-regularises diffuse masks
#   constraint_aware  the inversion searches infeasible pattern space
#
# Orthogonally, tau=2.0 is a vision-era constant. `calibrate_binary_tau` fits it from CLEAN models on
# calibration seeds only, which is a change to the decision rule rather than the inversion.


@dataclass(frozen=True)
class TabularInversionConfig:
    """One frozen inversion recipe. `n_starts=1, constraint_weight=0.0` is the published baseline."""
    steps: int = 300
    lr: float = 0.1
    l1_weight: float = 0.001
    n_starts: int = 1
    constraint_weight: float = 0.0


def select_best_start(ledger: Sequence[dict]) -> int | None:
    """Index of the lowest-objective start among the finite, CONVERGED ones.

    Returns None when no start qualifies -- the caller must record that as an inversion failure rather
    than fall back to a diverged solution. Failed starts stay in the ledger with their reason; they
    are never dropped, because the pre-registered gate also checks that convergence failures did not
    increase relative to the baseline.
    """
    best, best_obj = None, float("inf")
    for i, entry in enumerate(ledger):
        obj = entry.get("objective", float("inf"))
        if not entry.get("converged", False):
            continue
        if obj is None or not np.isfinite(obj):
            continue
        if obj < best_obj:
            best, best_obj = i, float(obj)
    return best


def calibrate_binary_tau(
    clean_ratios: dict, calibration_seeds: Sequence[int], evaluation_seeds: Sequence[int],
    quantile: float = 0.99,
) -> float:
    """Fit `tau` as the `quantile` of CLEAN-model mask ratios, from calibration seeds only.

    Refuses outright if `clean_ratios` carries a held-out evaluation seed. Letting an evaluation seed
    into the null distribution is the exact failure the preregistration exists to prevent: it makes
    the reported held-out detection rate partly self-selected, and no downstream check would catch it.
    """
    def _seed_of(key):
        """Keys are either a seed, or a (seed, replica) pair when several clean models share a seed.

        Multiple clean models per seed exist so the false-alarm rate is estimable at all: with k clean
        models the FPR estimate is quantised to 1/k, so 3 seeds can only resolve 0.33.
        """
        return key[0] if isinstance(key, tuple) else key

    leaked = sorted({_seed_of(k) for k in clean_ratios} & set(evaluation_seeds))
    if leaked:
        raise ValueError(
            f"clean_ratios contains evaluation seed(s) {leaked}: tau must be calibrated on "
            f"calibration seeds {sorted(calibration_seeds)} only"
        )
    calib = set(calibration_seeds)
    usable = [float(v) for k, v in sorted(clean_ratios.items(), key=lambda kv: str(kv[0]))
              if _seed_of(k) in calib and np.isfinite(v)]
    if not usable:
        raise ValueError("no finite clean ratios on the calibration seeds; cannot calibrate tau")
    return float(np.quantile(usable, quantile))


def _converged(losses: Sequence[float], tol: float = 1e-2, tail_frac: float = 0.1) -> bool:
    """Plateau test: finite, descended, and the last `tail_frac` of steps moved it by < `tol` of the
    TOTAL descent.

    An explicit criterion rather than "it ran to completion": the ledger has to distinguish a start
    that settled from one still descending or oscillating, otherwise `select_best_start` would happily
    pick an unconverged low point.

    WHY THE TOTAL DESCENT IS THE REFERENCE, not the current loss. An earlier version of this function
    divided the tail movement by the loss VALUE, which rejected every start ever run -- 6 of 6 at 100,
    300 and 600 steps. The inversion loss is cross-entropy plus an L1 mask penalty, so it asymptotes
    towards zero while the penalty keeps shaving the mask: on a 757-feature probe it went 1.0264 ->
    0.0422 by step 300, a 96% descent, yet the tail-versus-value ratio was still 1.4e-2 purely because
    the denominator had become tiny. Measured against total descent the same trace gives 5.9e-4, three
    orders inside the gate, and 100 steps (2.7e-3) still reads as less settled than 600 (2.1e-4) --
    which is the ordering the flag exists to express.

    This is a fix to a broken instrument, made while looking only at loss traces on synthetic and
    smoke-config data. No detection rate, recall, or held-out outcome was consulted in choosing `tol`.
    """
    arr = np.asarray(losses, dtype=np.float64)
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        return False
    total_descent = float(arr[0] - arr[-1])
    if total_descent <= 0.0:
        return False        # never improved: a failed start, not a converged one
    tail = max(1, int(round(tail_frac * arr.size)))
    prev = float(arr[-tail - 1]) if arr.size > tail else float(arr[0])
    return abs(float(arr[-1]) - prev) <= tol * total_descent


def reverse_engineer_tabular_multistart(
    model: torch.nn.Module,
    sample_x_std: torch.Tensor,
    bounds: tuple[torch.Tensor, torch.Tensor],
    config: TabularInversionConfig,
    seed: int,
    constraint_penalty=None,
    num_classes: int = 2,
    device: str = "cpu",
    retain_start_masks: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Multistart tabular inversion. Returns (selected norms, selected masks, ledger).

    Start 0 always uses the zeros init, so `n_starts=1, constraint_weight=0.0` reproduces
    `reverse_engineer_tabular`'s trajectory exactly and the baseline candidate stays the published
    pipeline rather than a re-implementation of it. Starts 1.. draw their inits from
    `np.random.SeedSequence(seed).spawn(n_starts)`, which is deterministic for a fixed seed.

    The ledger records EVERY start per class -- objective, convergence, and failure reason -- so a
    diverged or non-finite start is visible in the artifact instead of silently vanishing.

    `retain_start_masks` additionally keeps each start's final mask vector under
    `ledger["start_masks"]`, as (target_class, start, mask) records. It defaults to False because the
    selected mask is all any existing caller wants and the per-start arrays are large (n_starts x
    num_classes x d floats per inversion); with the default the ledger is byte-identical to what every
    committed result was produced with. The restart-stability statistic in
    scripts/65_nc_mask_geometry_pilot.py is the one caller that needs them: how far the starts
    disagree is unanswerable from the selected mask alone.
    """
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sample_x_std = sample_x_std.to(device)
    d = sample_x_std.shape[1]
    lo, hi = bounds[0].to(device), bounds[1].to(device)
    span = (hi - lo).clamp_min(1e-6)

    child_seeds = np.random.SeedSequence(seed).spawn(max(1, config.n_starts))
    norms = np.zeros(num_classes, dtype=np.float64)
    masks = np.zeros((num_classes, d), dtype=np.float64)
    # Selected patterns ride in the ledger rather than the return tuple so the signature stays
    # (norms, masks, ledger). They are what the raw-space feasibility audit must inspect: the mask is
    # a [0,1] blend weight, so de-standardising a MASK and calling the result a pattern would audit
    # the wrong object. `ledger["selected_patterns_std"]` is in STANDARDISED space, as inverted.
    patterns = np.zeros((num_classes, d), dtype=np.float64)
    ledger: dict = {"starts": [], "selection": []}

    for target in range(num_classes):
        y_t = torch.full((sample_x_std.shape[0],), target, dtype=torch.long, device=device)
        per_start = []
        for s in range(max(1, config.n_starts)):
            if s == 0:
                mask_init = torch.zeros(d, device=device)
                pat_init = torch.zeros(d, device=device)
            else:
                rng = np.random.default_rng(child_seeds[s])
                mask_init = torch.as_tensor(rng.normal(0.0, 0.5, size=d), dtype=torch.float32,
                                            device=device)
                pat_init = torch.as_tensor(rng.normal(0.0, 0.5, size=d), dtype=torch.float32,
                                           device=device)
            mask = mask_init.clone().requires_grad_(True)
            pattern = pat_init.clone().requires_grad_(True)
            opt = torch.optim.Adam([mask, pattern], lr=config.lr)

            losses, failure = [], None
            for _ in range(config.steps):
                m = torch.sigmoid(mask)
                p = lo + span * torch.sigmoid(pattern)
                blended = (1 - m) * sample_x_std + m * p
                opt.zero_grad()
                loss = F.cross_entropy(model(blended), y_t) + config.l1_weight * m.abs().sum()
                if constraint_penalty is not None and config.constraint_weight:
                    loss = loss + config.constraint_weight * constraint_penalty(p)
                if not torch.isfinite(loss):
                    failure = "non_finite_loss"
                    break
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            final_mask = torch.sigmoid(mask).detach()
            final_pattern = (lo + span * torch.sigmoid(pattern)).detach()
            norm = float(final_mask.abs().sum().item())
            converged = failure is None and _converged(losses)
            if failure is None and not converged:
                failure = "did_not_plateau"
            per_start.append(dict(
                target_class=target, start=s,
                objective=(losses[-1] if losses else float("inf")),
                mask_l1=norm, converged=bool(converged), failure_reason=failure,
                n_steps_run=len(losses),
                init=("zeros" if s == 0 else "seeded_normal"),
                _mask=final_mask.cpu().numpy(),
                _pattern=final_pattern.cpu().numpy(),
            ))

        chosen = select_best_start(per_start)
        if chosen is None:
            # No start converged. Fall back to the zeros-init start's value so the run continues, and
            # say so loudly in the ledger -- the gate counts these.
            chosen = 0
            ledger["selection"].append(dict(target_class=target, selected_start=0,
                                            selection="no_converged_start_fallback_to_zeros_init"))
        else:
            ledger["selection"].append(dict(target_class=target, selected_start=int(chosen),
                                            selection="lowest_objective_converged"))
        norms[target] = per_start[chosen]["mask_l1"]
        masks[target] = per_start[chosen]["_mask"]
        patterns[target] = per_start[chosen]["_pattern"]
        ledger["starts"].extend(
            {k: v for k, v in e.items() if k not in ("_mask", "_pattern")} for e in per_start)
        if retain_start_masks:
            ledger.setdefault("start_masks", []).extend(
                dict(target_class=target, start=int(e["start"]),
                     mask=np.asarray(e["_mask"], dtype=np.float64))
                for e in per_start)

    ledger["selected_patterns_std"] = patterns
    ledger["n_starts"] = int(max(1, config.n_starts))
    ledger["n_convergence_failures"] = int(sum(1 for e in ledger["starts"] if not e["converged"]))
    ledger["n_no_converged_start"] = int(
        sum(1 for s in ledger["selection"] if s["selection"].startswith("no_converged_start")))
    return norms, masks, ledger


# ---------------------------------------------------------------------------------------------
# Expected transferability (ET). Xiang, Miller and Kesidis, ICLR 2022 (arXiv:2201.08474), Alg. 1.
#
# WHY A SECOND DECISION RULE AT ALL. `nc_binary_flag`'s mask-L1 RATIO compares the two classes
# against each other. On this task that comparison is dominated by class asymmetry rather than by
# backdoor presence: benign is the majority AND the attack target, so a perturbation toward benign is
# structurally cheap in EVERY model. Measured on results/nc_repair_calibration.json (432 rows), the
# ratio separates poisoned from clean models at AUC 0.389-0.444 -- at or below chance -- so no
# threshold on it can work, which is why recalibrating tau could not and did not help.
#
# ET never compares the two classes. It asks, per class independently, whether a trigger
# reverse-engineered from ONE sample also flips OTHER samples of that class. A real backdoor is a
# shared shortcut, so per-sample solutions coincide and transfer; absent a backdoor they are
# sample-specific and do not. The threshold is a derived constant 1/2, not a calibrated quantity,
# which is the second reason to prefer it here: it removes the calibration step entirely.
#
# Pre-registered in notes/20260803-decision-nc13-et-preregistration.md BEFORE any ET value was
# computed on this data. The 1/2 threshold is frozen and must not be tuned.
# ---------------------------------------------------------------------------------------------


def et_from_transfer_matrix(transfer) -> float:
    """ET from an (N, N) boolean transfer matrix. `transfer[n, m]` is True when a trigger
    reverse-engineered on sample n also flips sample m. The diagonal is ignored (Alg. 1 excludes
    m == n), so each row is scored out of N-1.

    Split out from the optimisation deliberately: this is the whole statistic, and keeping it pure
    means the combinatorics can be tested without running a single inversion.
    """
    arr = np.asarray(transfer, dtype=bool)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"transfer matrix must be square, got shape {arr.shape}")
    n = arr.shape[0]
    if n < 2:
        raise ValueError("ET is undefined for fewer than 2 samples: each row is scored out of N-1")
    off_diagonal = arr.copy()
    np.fill_diagonal(off_diagonal, False)
    per_sample = off_diagonal.sum(axis=1) / (n - 1)
    return float(per_sample.mean())


def _solutions_for_sample(
    model: torch.nn.Module,
    x_std: torch.Tensor,
    bounds: tuple[torch.Tensor, torch.Tensor],
    config: TabularInversionConfig,
    target: int,
    seed: int,
    n_restarts: int,
    device: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Every restart's (mask, pattern) for a SINGLE sample, not just the best one.

    Alg. 1 line 9 unions the transferable set over solution realisations, so a restart that finds a
    different-but-valid trigger must contribute too. Returning only the lowest-objective start would
    silently discard exactly the evidence ET is built on.

    The objective, blending and optimiser mirror `reverse_engineer_tabular_multistart` step for step;
    the only differences are the single-row batch and that no start is selected. That function is
    pinned by tests to reproduce the published baseline exactly, so it is reused by imitation rather
    than by modification.
    """
    lo, hi = bounds[0].to(device), bounds[1].to(device)
    span = (hi - lo).clamp_min(1e-6)
    d = x_std.shape[1]
    y_t = torch.full((1,), target, dtype=torch.long, device=device)
    child_seeds = np.random.SeedSequence(seed).spawn(max(1, n_restarts))

    solutions = []
    for s in range(max(1, n_restarts)):
        if s == 0:
            mask_init = torch.zeros(d, device=device)
            pat_init = torch.zeros(d, device=device)
        else:
            rng = np.random.default_rng(child_seeds[s])
            mask_init = torch.as_tensor(rng.normal(0.0, 0.5, size=d), dtype=torch.float32,
                                        device=device)
            pat_init = torch.as_tensor(rng.normal(0.0, 0.5, size=d), dtype=torch.float32,
                                       device=device)
        mask = mask_init.clone().requires_grad_(True)
        pattern = pat_init.clone().requires_grad_(True)
        opt = torch.optim.Adam([mask, pattern], lr=config.lr)

        for _ in range(config.steps):
            m = torch.sigmoid(mask)
            p = lo + span * torch.sigmoid(pattern)
            blended = (1 - m) * x_std + m * p
            opt.zero_grad()
            loss = F.cross_entropy(model(blended), y_t) + config.l1_weight * m.abs().sum()
            if not torch.isfinite(loss):
                break
            loss.backward()
            opt.step()

        solutions.append((torch.sigmoid(mask).detach(),
                          (lo + span * torch.sigmoid(pattern)).detach()))
    return solutions


def expected_transferability(
    model: torch.nn.Module,
    samples_std: torch.Tensor,
    bounds: tuple[torch.Tensor, torch.Tensor],
    config: TabularInversionConfig,
    seed: int,
    putative_target: int,
    n_restarts: int = 3,
    device: str = "cpu",
) -> dict:
    """ET for one putative backdoor target class, per Alg. 1 of Xiang et al. (ICLR 2022).

    `samples_std` are clean, standardised rows of the SOURCE class (1 - putative_target). Rows the
    model does not already predict as the source class are dropped: Alg. 1 conditions on f(x) = i,
    and a row the model already sends to the target cannot evidence transferability of anything.

    Returns the statistic plus the inputs needed to audit it. `detected` applies the DERIVED constant
    threshold 1/2 -- it is not calibrated, and per the pre-registration it must not be tuned.
    """
    if putative_target not in (0, 1):
        raise ValueError(f"binary task: putative_target must be 0 or 1, got {putative_target}")
    source = 1 - putative_target

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    X = samples_std.to(device)

    with torch.no_grad():
        keep = (model(X).argmax(dim=1) == source)
    X = X[keep]
    n = int(X.shape[0])
    if n < 2:
        return dict(et=None, detected=None, n_samples=n, putative_target=putative_target,
                    source_class=source, n_restarts=int(n_restarts),
                    reason="fewer than 2 source-class rows survive the f(x)=i filter")

    transfer = np.zeros((n, n), dtype=bool)
    child = np.random.SeedSequence(seed).spawn(n)
    for idx in range(n):
        sample_seed = int(np.random.default_rng(child[idx]).integers(0, 2**31 - 1))
        for mask, pattern in _solutions_for_sample(
                model, X[idx:idx + 1], bounds, config, putative_target, sample_seed,
                n_restarts, device):
            with torch.no_grad():
                blended = (1 - mask) * X + mask * pattern
                flipped = (model(blended).argmax(dim=1) == putative_target).cpu().numpy()
            transfer[idx] |= flipped          # union over restarts, per Alg. 1 line 9

    et = et_from_transfer_matrix(transfer)
    off = transfer.copy()
    np.fill_diagonal(off, False)
    return dict(
        et=et,
        detected=bool(et > 0.5),
        threshold=0.5,
        threshold_origin="derived constant from Xiang et al. 2022 Thm. 3.1, NOT calibrated",
        n_samples=n,
        n_dropped_not_source_class=int((~keep).sum().item()),
        putative_target=putative_target,
        source_class=source,
        n_restarts=int(n_restarts),
        per_sample_transfer_rate=[float(v) for v in (off.sum(axis=1) / (n - 1))],
    )
