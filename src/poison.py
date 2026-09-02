"""Clean-label poisoning and the poison-rate x cost sweep skeleton.

Clean-label (decision 4): poison genuinely-benign training rows (target class = benign), stamp the
trigger into their features in RAW space, leave labels UNCHANGED, then standardise for the victim.
Poison budget uses the whole-training-set denominator (decision 1): n_poison = round(rate * n_train),
all drawn from the benign class. Returns the poison index for downstream detector recall (M3).
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src import config
from src.data import apply_standardiser
from src.trigger import apply_secondary_trigger, apply_trigger


def poison_trainset_cleanlabel(
    x_tr_raw, y_tr, trigger: Dict, poison_rate: float, scaler,
    target: int = config.ATTACK_TARGET, seed: int = 0,
    constraints: List = None, feature_names: Sequence[str] = None,
    bounds: Tuple[np.ndarray, np.ndarray] = None,
    exclude_idx: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_poisoned_std, y_poisoned, poison_idx).

    n_poison = round(poison_rate * len(x_tr)) benign rows are stamped with the trigger (raw space,
    projected if realizable); labels are NOT flipped (clean-label). The poisoned matrix is returned
    standardised for victim training; `poison_idx` indexes the poisoned rows within x_tr.

    `exclude_idx` removes rows from the poison candidate pool. It exists for the clean-reference
    centroid repair (plan Task 7B), whose defender-held reference subset must be provably
    unpoisonable. Default None leaves the candidate pool and therefore the drawn `poison_idx`
    bit-identical to every committed result in results/ -- verified by
    tests/test_poison.py::test_exclude_none_reproduces_the_committed_draw.
    """
    x = np.array(x_tr_raw, dtype=np.float64, copy=True)
    y = np.asarray(y_tr).copy()
    n_poison = int(round(poison_rate * len(x)))

    rng = np.random.default_rng(seed)
    benign = np.where(y == target)[0]
    if exclude_idx is not None:
        benign = np.setdiff1d(benign, np.asarray(exclude_idx), assume_unique=False)
    if n_poison > len(benign):
        raise ValueError(f"n_poison={n_poison} exceeds benign pool {len(benign)}")
    poison_idx = np.sort(rng.choice(benign, size=n_poison, replace=False))

    if n_poison:
        x[poison_idx] = apply_trigger(x[poison_idx], trigger, constraints, feature_names, bounds)
    # labels unchanged (clean-label); standardise for the victim
    x_std = apply_standardiser(scaler, x)
    return x_std, y, poison_idx


def sweep_grid(poison_rates: Sequence[float] = config.POISON_RATES,
               costs: Sequence[int] = config.TRIGGER_COSTS) -> List[Tuple[float, int]]:
    """The poison-rate x trigger-cost grid the M3 sweep iterates."""
    return list(itertools.product(poison_rates, costs))


# ----------------------------------------------------------------------------------------------------
# Secondary-dataset (extension Task 4) clean-label poisoning. Same denominator/RNG/label-invariant
# contract as `poison_trainset_cleanlabel` above, but stamps via `apply_secondary_trigger` (which
# always runs the trigger's own `project_fn`) instead of `apply_trigger` (which needs
# constraints/feature_names/bounds threaded through explicitly and is wired to CTU's constraint DSL).
# This is what makes the secondary functions dataset-agnostic: any dataset whose trigger dict carries
# a `project_fn` -- built by `src.trigger.build_secondary_trigger` -- can poison with these, not just
# UNSW-NB15.
# ----------------------------------------------------------------------------------------------------


def poison_secondary_trainset_raw(
    x_tr_raw, y_tr, trigger: Dict, poison_rate: float,
    target: int = config.ATTACK_TARGET, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean-label poisoning in RAW space, mirroring `poison_trainset_cleanlabel`'s exact
    denominator/RNG/label-invariant contract above, but stamping via `apply_secondary_trigger` (so ANY
    dataset's `project_fn` applies, not just CTU's constraint DSL). Returns `(x_raw_poisoned, y,
    poison_idx)` -- the raw (unstandardised) matrix, for callers that need to audit constraints on the
    poisoned rows directly (`poison_secondary_trainset` below standardises on top of this).

    `n_poison = round(poison_rate * len(x_tr_raw))` (whole-training-set denominator, decision 1, same
    as the primary dataset), drawn without replacement from the `target`-class rows only, labels left
    UNCHANGED (clean-label).
    """
    x = np.array(x_tr_raw, dtype=np.float64, copy=True)
    y = np.asarray(y_tr).copy()
    n_poison = int(round(poison_rate * len(x)))

    rng = np.random.default_rng(seed)
    target_pool = np.where(y == target)[0]
    if n_poison > len(target_pool):
        raise ValueError(f"n_poison={n_poison} exceeds target-class pool {len(target_pool)}")
    poison_idx = np.sort(rng.choice(target_pool, size=n_poison, replace=False))

    if n_poison:
        x[poison_idx] = apply_secondary_trigger(x[poison_idx], trigger)
    return x, y, poison_idx


def poison_secondary_trainset(
    x_tr_raw, y_tr, trigger: Dict, poison_rate: float, scaler,
    target: int = config.ATTACK_TARGET, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`poison_secondary_trainset_raw` plus standardisation for the victim (space discipline: the
    trigger/constraints stay raw; only this last step moves to the standardised space the MLP sees)."""
    x_raw_poisoned, y, poison_idx = poison_secondary_trainset_raw(
        x_tr_raw, y_tr, trigger, poison_rate, target=target, seed=seed)
    x_std = apply_standardiser(scaler, x_raw_poisoned)
    return x_std, y, poison_idx


def carve_clean_reference(
    y_tr, target: int = config.ATTACK_TARGET, seed: int = 0,
    n_reference: int = config.AC_REFERENCE_SIZE,
    n_calibration: int = config.AC_CALIBRATION_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition the target-class rows into (reference, calibration, suspect) index arrays.

    Defender-side split for the clean-reference centroid repair (plan Task 7B), drawn BEFORE any
    poisoning so both held-back subsets can be passed to `poison_trainset_cleanlabel(exclude_idx=...)`
    and are then provably poison-free:

    - `reference`   fits the PCA basis and the k-means centroids (the geometry).
    - `calibration` fits the distance cutoff only (the 99th percentile of clean distances).
    - `suspect`     is what the detector is actually evaluated on; the poison lands here.

    Reference and calibration rows are held out of the suspect set as well as out of the poison pool.
    Scoring a row whose own activation helped define the centroids would depress its distance and
    flatter the false-positive rate, which is the number the pre-registered gate turns on.

    Uses a seed offset so this draw cannot shift the poison draw for the same seed.
    """
    y = np.asarray(y_tr)
    benign = np.where(y == target)[0]
    need = n_reference + n_calibration
    if need >= len(benign):
        raise ValueError(
            f"reference+calibration ({need}) must leave a suspect set: only {len(benign)} target-class rows"
        )
    rng = np.random.default_rng(seed + config.AC_REFERENCE_SEED_OFFSET)
    picked = rng.choice(benign, size=need, replace=False)
    reference = np.sort(picked[:n_reference])
    calibration = np.sort(picked[n_reference:])
    suspect = np.setdiff1d(benign, picked, assume_unique=False)
    return reference, calibration, suspect
