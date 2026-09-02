"""Gate logic for the Neural Cleanse repair (scripts/18a_nc_repair.py, plan Task 7A).

These guard the decision layer, which is where a wrong verdict does the most damage: a gate that
cannot be assessed must never be reported as a failed repair, and a gate must not reject a candidate
for missing a test of a hypothesis it is not testing.

Loaded via importlib because the gate lives in a numbered script rather than an importable package.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location("nc_repair", ROOT / "scripts" / "18a_nc_repair.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nc = _load()


def _summary(candidate, detection, fpr_loo=0.0, resolution=0.005, projected_valid=True,
             mean_violation=1e9, convergence_failures=0, constraint_valid_fraction=None):
    # `constraint_valid_fraction` is the new gate-2 quantity (NC-2). Default it FROM `projected_valid`
    # so every pre-existing call site -- which only ever set the old boolean -- keeps exercising the
    # same pass/fail scenario under the new fraction-based mechanism, not a stale one: True -> 1.0
    # (clears config.NC_MIN_CONSTRAINT_VALID_FRACTION), False -> 0.5 (well below it).
    if constraint_valid_fraction is None:
        constraint_valid_fraction = 1.0 if projected_valid else 0.5
    return dict(
        candidate=candidate, spec=nc.CANDIDATES[candidate], tau=1.5, tau_error=None, clean_ratios={},
        target_detection=detection, detection_at_baseline_tau_2=detection, clean_fpr_in_sample=1.0,
        clean_fpr_loo=dict(clean_fpr_loo=fpr_loo, n_clean_models=int(round(1 / resolution)),
                           resolution=resolution, estimable_at_gate=(resolution <= 0.01),
                           reason=None if resolution <= 0.01 else "resolution coarser than the gate"),
        mean_pattern_violation=mean_violation, max_pattern_violation=mean_violation * 2,
        mean_trigger_support_alignment=0.2, chance_alignment=0.02,
        all_patterns_valid=False,
        all_patterns_projectable_valid_after_projection=projected_valid,
        min_constraint_valid_fraction_after_projection=constraint_valid_fraction,
        n_convergence_failures=convergence_failures, n_no_converged_start=0,
        n_poisoned_cells=6, n_clean_models=int(round(1 / resolution)),
    )


# --- not-assessable must never masquerade as a negative result -------------------------------------

def test_coarse_fpr_resolution_is_not_assessable_rather_than_failed():
    """3 clean models resolve 0.33 at best; a 0.01 gate cannot be decided from that."""
    summaries = {"baseline": _summary("baseline", 0.3, resolution=1 / 3),
                 "multistart": _summary("multistart", 0.9, resolution=1 / 3)}
    out = nc.select_candidate(summaries)
    assert out["verdict"] == "gates_not_assessable"
    assert out["selected"] is None
    assert any("clean FPR gate not assessable" in r
               for r in out["rejected"][1]["not_assessable"])
    assert "do not report it as" in out["statement"].lower()


def test_ceiling_bound_baseline_is_not_assessable():
    """Baseline already at 1.0 leaves no headroom for any candidate to gain 0.10."""
    summaries = {"baseline": _summary("baseline", 1.0),
                 "multistart": _summary("multistart", 1.0)}
    out = nc.select_candidate(summaries)
    assert out["verdict"] == "gates_not_assessable"
    assert any("baseline detection is already 1.0" in r
               for r in out["rejected"][1]["not_assessable"])


# --- the violation-reduction gate is scoped to the candidate that tests that hypothesis ------------

def test_multistart_is_not_required_to_reduce_constraint_violation():
    """Regression: multistart runs constraint_weight=0.0, so its violation MATCHES baseline by
    construction. An earlier version applied the reduction gate universally and rejected a candidate
    with a +0.60 detection gain and a clean FPR."""
    summaries = {"baseline": _summary("baseline", 0.3, mean_violation=1e9),
                 "multistart": _summary("multistart", 0.9, mean_violation=1e9)}
    out = nc.select_candidate(summaries)
    assert out["verdict"] == "candidate_frozen"
    assert out["selected"] == "multistart"


def test_sparser_is_not_required_to_reduce_constraint_violation():
    summaries = {"baseline": _summary("baseline", 0.3, mean_violation=1e9),
                 "sparser": _summary("sparser", 0.9, mean_violation=1e9)}
    assert nc.select_candidate(summaries)["selected"] == "sparser"


def test_constraint_aware_must_reduce_violation():
    summaries = {"baseline": _summary("baseline", 0.3, mean_violation=1e9),
                 "constraint_aware": _summary("constraint_aware", 0.9, mean_violation=9e8)}
    out = nc.select_candidate(summaries)
    assert out["selected"] is None
    assert "violation reduction" in out["rejected"][1]["reason"]


def test_constraint_aware_passes_at_the_observed_reduction():
    """The magnitudes here are the ones measured on the smoke run: 3.07e9 -> 9.17e7, about 33x."""
    summaries = {"baseline": _summary("baseline", 0.3, mean_violation=3.07e9),
                 "constraint_aware": _summary("constraint_aware", 0.9, mean_violation=9.17e7)}
    out = nc.select_candidate(summaries)
    assert out["verdict"] == "candidate_frozen"
    assert out["selected"] == "constraint_aware"


# --- genuine failures still fail ------------------------------------------------------------------

def test_insufficient_detection_gain_is_a_genuine_failure():
    summaries = {"baseline": _summary("baseline", 0.3),
                 "sparser": _summary("sparser", 0.35)}
    out = nc.select_candidate(summaries)
    assert out["verdict"] == "no_nc_repair_candidate"
    assert "detection gain" in out["rejected"][1]["reason"]


def test_pattern_invalid_after_projection_is_a_genuine_failure():
    summaries = {"baseline": _summary("baseline", 0.3),
                 "multistart": _summary("multistart", 0.9, projected_valid=False)}
    out = nc.select_candidate(summaries)
    assert out["verdict"] == "no_nc_repair_candidate"
    assert "after projection" in out["rejected"][1]["reason"]


def test_gate_rejects_on_fraction_even_when_old_boolean_would_have_passed():
    """The actual point of NC-2. At n=2 audited rows the OLD row-level boolean is quantised to
    {0, 0.5, 1.0}: a candidate that repairs 355 of 358 constraints (fraction ~0.9916) still reads
    `all_patterns_projectable_valid_after_projection=False` there, indistinguishable from a candidate
    that repairs almost nothing. The NEW constraint-level fraction resolves that -- but only if the
    gate actually reads it. Here the two signals are made to DISAGREE the other way: the stale boolean
    says True (would have passed under the old gate) while the fraction (0.97) sits below
    config.NC_MIN_CONSTRAINT_VALID_FRACTION (~0.9916). If the gate were still keyed off the boolean,
    this candidate would be selected; it must not be."""
    assert 0.97 < nc.config.NC_MIN_CONSTRAINT_VALID_FRACTION
    summaries = {
        "baseline": _summary("baseline", 0.3),
        "multistart": _summary("multistart", 0.9, projected_valid=True,
                               constraint_valid_fraction=0.97),
    }
    out = nc.select_candidate(summaries)
    assert out["verdict"] == "no_nc_repair_candidate"
    assert out["selected"] is None
    assert "constraint-valid fraction" in out["rejected"][1]["reason"]


def test_high_clean_fpr_is_a_genuine_failure_when_resolvable():
    summaries = {"baseline": _summary("baseline", 0.3),
                 "multistart": _summary("multistart", 0.9, fpr_loo=0.5)}
    out = nc.select_candidate(summaries)
    assert out["verdict"] == "no_nc_repair_candidate"
    assert "clean FPR" in out["rejected"][1]["reason"]


def test_baseline_is_never_itself_a_candidate():
    summaries = {"baseline": _summary("baseline", 0.9)}
    out = nc.select_candidate(summaries)
    assert out["selected"] is None
    assert "baseline is the comparison" in out["rejected"][0]["reason"]


# --- tie-breaking is deterministic ----------------------------------------------------------------

def test_equal_candidates_break_ties_by_name_deterministically():
    summaries = {"baseline": _summary("baseline", 0.3),
                 "multistart": _summary("multistart", 0.9),
                 "sparser": _summary("sparser", 0.9)}
    first = nc.select_candidate(summaries)["selected"]
    assert first == nc.select_candidate(summaries)["selected"]
    assert first == "multistart"   # alphabetical last key in the ordered selection tuple


# --- the leave-one-out FPR estimator --------------------------------------------------------------

def _clean_row(seed, replica, ratio, flagged=0):
    return dict(seed=seed, replica=replica, candidate="baseline", flagged_class=flagged,
                ratio=ratio, ratio_saturated=False)


def test_loo_reports_resolution_and_refuses_the_gate_when_too_coarse():
    rows = [_clean_row(s, 0, 1.0 + i) for i, s in enumerate((42, 123, 456))]
    out = nc._clean_fpr_loo(rows, (42, 123, 456), quantile=0.99)
    assert out["n_clean_models"] == 3
    assert out["resolution"] == pytest.approx(1 / 3)
    assert out["estimable_at_gate"] is False
    assert "clean models" in out["reason"]


def test_loo_becomes_estimable_with_enough_replicas():
    rows = [_clean_row(s, r, 1.0 + 0.01 * r) for s in (42, 123, 456) for r in range(40)]
    out = nc._clean_fpr_loo(rows, (42, 123, 456), quantile=0.99)
    assert out["n_clean_models"] == 120
    assert out["resolution"] == pytest.approx(1 / 120)
    assert out["estimable_at_gate"] is True


def test_loo_needs_at_least_two_models():
    out = nc._clean_fpr_loo([_clean_row(42, 0, 1.0)], (42,), quantile=0.99)
    assert out["clean_fpr_loo"] is None and out["estimable_at_gate"] is False


def test_loo_excludes_models_from_non_calibration_seeds():
    rows = [_clean_row(42, 0, 1.0), _clean_row(123, 0, 1.0), _clean_row(789, 0, 99.0)]
    out = nc._clean_fpr_loo(rows, (42, 123), quantile=0.99)
    assert out["n_clean_models"] == 2   # the held-out 789 model is not used
