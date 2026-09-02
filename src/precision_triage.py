"""Extension Task 8: an exploratory second-stage SHAP-attribution-concentration filter on top of the
locked MAD z=3.0 (config.MAD_Z_PRIMARY) Spectral rule, fit ONLY on clean, MAD-flagged validation rows
(never poison, never test rows, never the actual trigger values) -- see
notes/2026-07-29-generalisation-adaptive-attacker-extension.md Task 8, whose only caller is
scripts/19_precision_triage.py.

This module holds the leakage guard (`fit_precision_cutoff`) and the recall/precision/fpr arithmetic
separate from the script so tests/test_analysis_extension.py -- which imports from src/, never from
scripts/ -- can exercise it directly. This mirrors src/analysis_extension.py's own docstring
convention: guards live in src/, never only inline in a script, so they can be unit-tested without
running the full pipeline.

Circularity warning (do not remove, and do not represent this pilot as a defence fix without reading
it first): the escalation feature set (`trigger_indices`) comes from a SHAP ranking on the same clean
surrogate model used to build the attacker's own trigger in the first place (src/trigger.py's
shap_rank_features/build_trigger). A precision gain here is evidence for THIS pilot's specific threat
model only, not a general defence result. Both the cutoff-fitting stage (Step 1) and the evaluation
stage (Step 2) explain rows through the SAME per-seed clean surrogate MLP -- never the poisoned
victim -- so the calibration and the evaluation are computed by the same explainer and so the
statistic is not trivially reconstructible from the poisoned model's own backdoor association (a
strictly worse circularity than the SHAP-ranking overlap the plan already flags). This is a design
decision made in scripts/19_precision_triage.py, not something this module enforces on its own.

This trades circularity-avoidance for deployment-realism: a real defender only ever has the
possibly-poisoned production model, not a clean counterfactual, so a held-out confirmation run (if one
is ever pre-registered off this pilot) should also test the poisoned-victim-explains-evaluation variant
as a robustness check against this choice.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from src import config
from src.detectors import poison_recall


def concentration_from_abs_shap(abs_shap_matrix: np.ndarray, trigger_indices: Sequence[int]) -> np.ndarray:
    """Per-row normalised attribution concentration over `trigger_indices` (plan Task 8, Step 1):

        concentration = abs_shap[trigger_indices].sum() / abs_shap.sum()

    One value per row of `abs_shap_matrix` (n, D). A row whose total |SHAP| is exactly zero (a
    degenerate all-zero attribution -- not expected in practice but not excluded by construction)
    gets concentration 0.0 rather than NaN/inf: json cannot represent either, and a zero-attribution
    row carries no leverage on the trigger features by definition.
    """
    m = np.asarray(abs_shap_matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"abs_shap_matrix must be 2-D (n, D), got shape {m.shape}")
    idx = np.asarray(list(trigger_indices), dtype=int)
    total = m.sum(axis=1)
    trig = m[:, idx].sum(axis=1)
    return np.divide(trig, total, out=np.zeros_like(trig), where=total > 0)


def fit_precision_cutoff(concentrations, partition: str, contains_poison: bool = False,
                         quantile: float = config.CLEAN_CALIBRATION_QUANTILE) -> float:
    """The escalation cutoff: the `quantile`-th percentile (config.CLEAN_CALIBRATION_QUANTILE = 0.99
    by default) of `concentrations`. Step 1's leakage guard: refuses to fit on anything but genuinely
    clean, held-out validation rows.

    `partition` must be the literal string "clean_validation" AND `contains_poison` must be False --
    either condition failing alone is a distinct leakage vector (a mislabelled partition string, or a
    partition correctly named "clean_validation" that was nonetheless contaminated with poisoned rows
    upstream) and either one alone must refuse, not just the conjunction of both being wrong.
    """
    if partition != "clean_validation" or contains_poison:
        raise ValueError(
            f"fit_precision_cutoff refuses to fit on partition={partition!r} "
            f"contains_poison={contains_poison!r} -- the escalation cutoff may ONLY be fit on clean "
            "validation rows (partition='clean_validation', contains_poison=False). Fitting on test "
            "rows, poisoned rows, or a partition that merely claims to be clean would leak exactly "
            "the rows this pilot is evaluated against into its own threshold."
        )
    arr = np.asarray(concentrations, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(
            "fit_precision_cutoff: no clean, MAD-flagged validation rows to fit from (empty "
            "concentration array) -- cannot fit a percentile on nothing"
        )
    return float(np.quantile(arr, quantile))


def two_stage_flag_from_concentration(mad_flag: np.ndarray, flagged_concentration: np.ndarray,
                                      cutoff: float) -> np.ndarray:
    """Combine the (locked) MAD flag with the second-stage concentration escalation: a row must be
    both MAD-flagged AND at/above the escalation cutoff to be flagged at the second stage.

    `flagged_concentration` holds exactly one concentration value per True position in `mad_flag`, in
    the same order as `np.where(mad_flag)[0]` -- callers explain only the MAD-flagged subset
    (concentration is only defined/needed there, and explaining the full pool would be wasted compute
    on rows that are already un-flagged), so this scatters the escalation decision back onto the
    MAD-flagged positions of a full-length boolean array.

    The result can only be a subset of `mad_flag` (never adds a flag MAD did not already raise), which
    is exactly why two_stage_recall/precision/fpr can only fall or hold relative to
    mad_recall/precision/fpr, never rise -- this is the mechanism behind Step 2's recall-drop rejection
    rule, not merely a documented expectation.
    """
    mad_flag = np.asarray(mad_flag, dtype=bool)
    flagged_concentration = np.asarray(flagged_concentration, dtype=np.float64)
    n_flagged = int(mad_flag.sum())
    if len(flagged_concentration) != n_flagged:
        raise ValueError(
            f"flagged_concentration has {len(flagged_concentration)} entries but mad_flag has "
            f"{n_flagged} True positions -- exactly one concentration value is required per "
            "MAD-flagged row, in the same order as np.where(mad_flag)[0]"
        )
    out = np.zeros_like(mad_flag)
    out[np.where(mad_flag)[0]] = flagged_concentration >= cutoff
    return out


def poison_precision(flagged: np.ndarray, is_poison: np.ndarray) -> Optional[float]:
    """True-positive-flagged / total-flagged. None (not 0.0, not 1.0, not NaN) when nothing is
    flagged -- this codebase's established None-for-undefined convention (e.g. scripts/16's
    `mad_fpr`, src/detectors/spectral.py's `auc_reason`), never a fabricated value and never a bare
    NaN (which `json.dumps` cannot serialise without `allow_nan=True`, which this project's strict-json
    requirement forbids relying on)."""
    flagged = np.asarray(flagged, dtype=bool)
    is_poison = np.asarray(is_poison, dtype=bool)
    n_flagged = int(flagged.sum())
    if n_flagged == 0:
        return None
    return float((flagged & is_poison).sum() / n_flagged)


def poison_fpr(flagged: np.ndarray, is_poison: np.ndarray) -> Optional[float]:
    """False-positive-flagged / total-clean (non-poison) rows in the pool. None when the pool has no
    clean rows (degenerate, not expected on any real cell here, but guarded rather than dividing by
    zero)."""
    flagged = np.asarray(flagged, dtype=bool)
    is_poison = np.asarray(is_poison, dtype=bool)
    n_clean = int((~is_poison).sum())
    if n_clean == 0:
        return None
    return float((flagged & ~is_poison).sum() / n_clean)


def poison_recall_or_none(flagged: np.ndarray, is_poison: np.ndarray) -> Optional[float]:
    """`src.detectors.poison_recall` returns NaN (not None) when nothing is poisoned -- fine for that
    module's own callers, but NaN is not valid strict JSON. This wrapper converts that one undefined
    case to None so every row this module produces round-trips through json.dumps/json.loads without a
    `NaN`/`Infinity` literal anywhere in the artifact."""
    r = poison_recall(flagged, is_poison)
    return None if math.isnan(r) else float(r)
