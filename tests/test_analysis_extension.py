"""Guards on the frozen extension preregistration, plus the aggregation guards `src/analysis_extension.py`
introduces (extension Task 5): a mixed-dataset row set, an incomplete seed x rate x cost grid, a
post-hoc MAD-threshold pick, or an un-overridden determinism failure must all fail loudly rather than
pass quietly. A later task may extend this module's guard set further (empty-arm, bootstrap, etc.).

These tests exist to make a post-hoc protocol change fail loudly rather than pass quietly.
Protocol: notes/20260729-decision-extension-preregistration.md (plan Task 1).
"""
from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.analysis_extension import (
    ADAPTIVE_ATTACKER_VERDICTS,
    classify_adaptive_attacker_outcome,
    classify_evasion_window,
    classify_replication,
    require_complete_grid,
    require_determinism_ok,
    select_primary_mad_result,
    summarise_cell,
    summarise_dataset_rows,
)
from src.precision_triage import (
    concentration_from_abs_shap,
    fit_precision_cutoff,
    poison_fpr,
    poison_precision,
    poison_recall_or_none,
    two_stage_flag_from_concentration,
)


def test_adaptive_surrogate_and_evaluation_seeds_are_disjoint():
    assert set(config.ADAPTIVE_SURROGATE_SEEDS).isdisjoint(config.ADAPTIVE_EVAL_SEEDS)
    assert set(config.ADAPTIVE_SURROGATE_SEEDS) | set(config.ADAPTIVE_EVAL_SEEDS) == set(config.SEEDS)


def test_extension_base_seeds_match_the_workspace_seed_protocol():
    # >=5 seeds for every headline result; the extension must not quietly shrink the seed set.
    assert config.EXTENSION_BASE_SEEDS == config.SEEDS
    assert len(config.EXTENSION_BASE_SEEDS) >= 5


def test_replication_seeds_do_not_overlap_the_base_seeds():
    # Replication seeds are additive for pre-specified headline cells, never a substitution.
    assert set(config.EXTENSION_REPLICATION_SEEDS).isdisjoint(config.SEEDS)


def test_locked_operating_points_hold_their_preregistered_values():
    assert config.MAD_Z_PRIMARY == 3.0
    assert config.SPECTRAL_COMPONENTS == 5
    assert config.ATTACK_EFFECTIVE_ASR == 0.8
    assert config.FIXED_BUDGET_MISS_RECALL == 0.5
    assert config.CLEAN_ACCURACY_DROP_MAX == pytest.approx(0.02)
    assert config.SECONDARY_RATES == (0.005, 0.01, 0.05, 0.10)
    assert config.SECONDARY_COSTS == (8, 16)


def test_repair_gates_hold_their_preregistered_values():
    assert config.AC_REPAIR_MIN_RECALL_GAIN == pytest.approx(0.20)
    assert config.NC_REPAIR_MIN_DETECTION_GAIN == pytest.approx(0.10)
    assert config.AC_REPAIR_MAX_CLEAN_FPR == pytest.approx(0.01)
    assert config.NC_REPAIR_MAX_CLEAN_FPR == pytest.approx(0.01)
    # The rejected post-split proxy correlated 0.9967 with Spectral; the gate must sit below that.
    assert config.AC_SPECTRAL_INDEPENDENCE_MAX_RHO == pytest.approx(0.90)
    assert config.AC_SPECTRAL_INDEPENDENCE_MAX_RHO < 0.9967
    assert config.CLEAN_CALIBRATION_QUANTILE == pytest.approx(0.99)


# --------------------------------------------------------------------------------------------------
# src/analysis_extension.py aggregation guards (extension Task 5, plan Task 5 Step 4)
# --------------------------------------------------------------------------------------------------


def complete_grid_rows():
    """Two-cell (seed x rate x cost) grid rows with the minimal fields `require_complete_grid` needs."""
    return [
        {"seed": 42, "rate": 0.005, "cost": 8},
        {"seed": 123, "rate": 0.005, "cost": 8},
    ]


def test_extension_analysis_rejects_mixed_dataset_rows():
    rows = [{"dataset": "ctu"}, {"dataset": "secondary"}]
    with pytest.raises(ValueError, match="single dataset"):
        summarise_dataset_rows(rows)


def test_extension_analysis_rejects_missing_seed_cell():
    rows = complete_grid_rows()[:-1]
    with pytest.raises(ValueError, match="missing seed"):
        require_complete_grid(rows, seeds=[42, 123], rates=[0.005], costs=[8])


def test_extension_analysis_accepts_a_genuinely_complete_grid():
    rows = complete_grid_rows()
    require_complete_grid(rows, seeds=[42, 123], rates=[0.005], costs=[8])  # must not raise


def test_extension_analysis_uses_predeclared_mad_cutoff_only():
    assert select_primary_mad_result({"3.0": {}}, primary_z=3.0) == {}


def test_extension_analysis_select_primary_mad_result_rejects_missing_key():
    with pytest.raises(ValueError, match="predeclared MAD"):
        select_primary_mad_result({"2.0": {}}, primary_z=3.0)


def test_extension_analysis_rejects_empty_row_set():
    with pytest.raises(ValueError, match="empty row set"):
        summarise_dataset_rows([])


def test_extension_analysis_summarise_computes_per_cell_means():
    rows = [
        {"dataset": "secondary", "rate": 0.005, "cost": 8, "asr": 0.9,
         "spectral": {"recall": 0.1}, "ac": {"recall": 0.8}},
        {"dataset": "secondary", "rate": 0.005, "cost": 8, "asr": 1.0,
         "spectral": {"recall": 0.3}, "ac": {"recall": 0.6}},
    ]
    out = summarise_dataset_rows(rows)
    assert out["dataset"] == "secondary"
    assert len(out["cells"]) == 1
    cell = out["cells"][0]
    assert cell["rate"] == 0.005 and cell["cost"] == 8 and cell["n_seeds"] == 2
    assert cell["mean_asr"] == pytest.approx(0.95)
    assert cell["mean_fixed_budget_recall"] == pytest.approx(0.2)
    assert cell["mean_ac_recall"] == pytest.approx(0.7)


def test_extension_analysis_classify_evasion_window_matches_step3_formula():
    assert classify_evasion_window(mean_asr=0.9, mean_fixed_budget_recall=0.1) is True
    assert classify_evasion_window(mean_asr=0.5, mean_fixed_budget_recall=0.1) is False  # attack fails
    assert classify_evasion_window(mean_asr=0.9, mean_fixed_budget_recall=0.6) is False  # not a miss


def test_extension_analysis_require_determinism_ok_raises_unless_overridden():
    rows = [{"seed": 42, "rate": 0.005, "cost": 8, "determinism_ok": False}]
    with pytest.raises(ValueError, match="determinism_ok is False"):
        require_determinism_ok(rows)
    require_determinism_ok(rows, override=True)  # must not raise


def test_extension_analysis_require_determinism_ok_ignores_none_and_true():
    rows = [
        {"seed": 42, "rate": 0.005, "cost": 8, "determinism_ok": None},
        {"seed": 123, "rate": 0.005, "cost": 8, "determinism_ok": True},
    ]
    require_determinism_ok(rows)  # must not raise


# --------------------------------------------------------------------------------------------------
# src/analysis_extension.py additions (extension Task 9, plan Task 9 Steps 1-3): per-cell bootstrap
# summaries, the replication classifier, and the adaptive-attacker verdict classifier.
# --------------------------------------------------------------------------------------------------


def test_summarise_cell_computes_mean_and_sorts_per_seed_by_seed():
    # Deliberately unsorted input -- per_seed must come back seed-ascending (seed 42 before 123)
    # regardless of row order, and the mean of {0.0, 1.0} is exactly 0.5 (hand-computable).
    rows = [{"seed": 123, "asr": 0.0}, {"seed": 42, "asr": 1.0}]
    out = summarise_cell(rows, "asr")
    assert out["n"] == 2
    assert out["mean"] == pytest.approx(0.5)
    assert out["per_seed"] == [1.0, 0.0]
    lo, hi = out["ci"]
    assert lo <= out["mean"] <= hi


def test_summarise_cell_rejects_empty_row_set():
    with pytest.raises(ValueError, match="empty row set"):
        summarise_cell([], "asr")


def test_summarise_cell_degenerate_single_row_returns_point_ci():
    # bootstrap_ci's own n<2 convention: CI collapses to [value, value], not a fabricated spread.
    out = summarise_cell([{"seed": 42, "mad_recall": 0.7}], "mad_recall")
    assert out["n"] == 1
    assert out["mean"] == pytest.approx(0.7)
    assert out["ci"] == pytest.approx([0.7, 0.7])


def test_classify_replication_attack_not_effective_short_circuits_before_touching_recall_or_mad():
    # ASR below the gate -- must return without ever needing "fixed_budget_recall" or "mad_recall" to
    # be present, so this cell dict deliberately omits both.
    cell = {"asr": {"mean": 0.5}}
    assert classify_replication(cell) == "attack_not_effective"


def test_classify_replication_no_fixed_budget_window_short_circuits_before_touching_mad():
    cell = {"asr": {"mean": 0.9}, "fixed_budget_recall": {"mean": 0.6}}
    assert classify_replication(cell) == "no_fixed_budget_window"


def test_classify_replication_mad_not_recovered():
    cell = {"asr": {"mean": 0.9}, "fixed_budget_recall": {"mean": 0.1}, "mad_recall": {"mean": 0.3}}
    assert classify_replication(cell) == "mad_not_recovered"


def test_classify_replication_window_and_recovery_replicated():
    cell = {"asr": {"mean": 0.9}, "fixed_budget_recall": {"mean": 0.1}, "mad_recall": {"mean": 0.9}}
    assert classify_replication(cell) == "window_and_recovery_replicated"


def test_classify_replication_raises_keyerror_when_mad_data_needed_but_missing():
    # Attack-effective AND fixed-budget-missed, but the caller never attached mad_recall -- this must
    # fail loudly (KeyError), never silently guess a classification.
    cell = {"asr": {"mean": 0.9}, "fixed_budget_recall": {"mean": 0.1}}
    with pytest.raises(KeyError):
        classify_replication(cell)


def _outcome(feasible=True, mad_evasion_ok=True, attack_effective=True, clean_accuracy_ok=True):
    """One held-out-seed `adaptive_outcomes` entry, real-shape (the four independent flags
    `src.adaptive_trigger.evaluate_candidate` actually writes, not the AND-ed `evasion_success`)."""
    return dict(feasible=feasible, mad_evasion_ok=mad_evasion_ok,
                attack_effective=attack_effective, clean_accuracy_ok=clean_accuracy_ok)


def _held_out_evasion_success(outcomes):
    """The exact conjunction scripts/17_adaptive_attacker.py:477 computes -- used here only to build
    test fixtures that are self-consistent with the real upstream formula (evasion_success = feasible
    AND mad_evasion_ok AND attack_effective AND clean_accuracy_ok, held_out_evasion_success = all(...)
    across seeds), so a test never hand-constructs an artifact shape the real pipeline could not emit.
    """
    return all(o["feasible"] and o["mad_evasion_ok"] and o["attack_effective"] and o["clean_accuracy_ok"]
               for o in outcomes)


def _adaptive_result(selection_verdict, outcomes=None):
    outcomes = outcomes or []
    return {
        "selection": {"verdict": selection_verdict},
        "evaluation": {
            "held_out_evasion_success": _held_out_evasion_success(outcomes) if outcomes else False,
            "adaptive_outcomes": outcomes,
        },
    }


def test_classify_adaptive_attacker_outcome_no_eligible_candidate():
    result = _adaptive_result("no_eligible_candidate")
    assert (classify_adaptive_attacker_outcome(result)
            == "no_eligible_adaptive_candidate_within_search_budget")


def test_classify_adaptive_attacker_outcome_incomplete_surrogate_coverage_also_counts_as_no_eligible():
    # Any selection.verdict other than the literal "selected" means nothing was evaluated -- not just
    # the specific "no_eligible_candidate" string.
    result = _adaptive_result("incomplete_surrogate_coverage")
    assert (classify_adaptive_attacker_outcome(result)
            == "no_eligible_adaptive_candidate_within_search_budget")


def test_classify_adaptive_attacker_outcome_failed_held_out_evaluation():
    # MAD still flags the trigger on one held-out seed (mad_evasion_ok=False) -- caught outright,
    # independent of the fact that the attack would otherwise have been effective there.
    outcomes = [_outcome(mad_evasion_ok=True), _outcome(mad_evasion_ok=False)]
    result = _adaptive_result("selected", outcomes=outcomes)
    assert result["evaluation"]["held_out_evasion_success"] is False  # sanity: fixture is self-consistent
    assert (classify_adaptive_attacker_outcome(result)
            == "adaptive_candidate_failed_held_out_evaluation")


def test_classify_adaptive_attacker_outcome_evaded_only_by_losing_attack_utility():
    # MAD is evaded (feasible + mad_evasion_ok True) on EVERY seed, but the second seed's attack
    # itself failed the effectiveness gate -- this is the branch the code-review flagged as
    # unreachable under the old (held_out_evasion_success-gated) formula; it is reachable now because
    # mad_evasion_ok is read independently of attack_effective/clean_accuracy_ok.
    outcomes = [
        _outcome(mad_evasion_ok=True, attack_effective=True, clean_accuracy_ok=True),
        _outcome(mad_evasion_ok=True, attack_effective=False, clean_accuracy_ok=True),
    ]
    result = _adaptive_result("selected", outcomes=outcomes)
    assert result["evaluation"]["held_out_evasion_success"] is False  # sanity: fixture is self-consistent
    assert (classify_adaptive_attacker_outcome(result)
            == "adaptive_candidate_evaded_only_by_losing_attack_utility")


def test_classify_adaptive_attacker_outcome_evaded_only_by_losing_attack_utility_via_clean_accuracy():
    # Same branch, triggered by clean_accuracy_ok instead of attack_effective -- both utility gates
    # must be checked independently, not just one of the two.
    outcomes = [
        _outcome(mad_evasion_ok=True, attack_effective=True, clean_accuracy_ok=True),
        _outcome(mad_evasion_ok=True, attack_effective=True, clean_accuracy_ok=False),
    ]
    result = _adaptive_result("selected", outcomes=outcomes)
    assert (classify_adaptive_attacker_outcome(result)
            == "adaptive_candidate_evaded_only_by_losing_attack_utility")


def test_classify_adaptive_attacker_outcome_confirmed_within_search_budget():
    outcomes = [_outcome(), _outcome()]  # feasible, MAD evaded, utility intact on every seed
    result = _adaptive_result("selected", outcomes=outcomes)
    assert result["evaluation"]["held_out_evasion_success"] is True  # sanity: fixture is self-consistent
    assert (classify_adaptive_attacker_outcome(result)
            == "adaptive_evasion_confirmed_within_search_budget")


def test_classify_adaptive_attacker_outcome_raises_on_empty_outcomes_when_selected():
    result = {"selection": {"verdict": "selected"},
              "evaluation": {"held_out_evasion_success": False, "adaptive_outcomes": []}}
    with pytest.raises(ValueError, match="adaptive_outcomes is empty"):
        classify_adaptive_attacker_outcome(result)


def test_classify_adaptive_attacker_outcome_raises_when_recorded_success_disagrees_with_outcomes():
    # The artifact claims held_out_evasion_success=True, but the per-seed flags it also carries say
    # MAD was NOT evaded on this seed -- an internally inconsistent artifact, must fail loudly rather
    # than silently pick one of the two contradictory signals.
    outcomes = [_outcome(mad_evasion_ok=False)]
    result = {"selection": {"verdict": "selected"},
              "evaluation": {"held_out_evasion_success": True, "adaptive_outcomes": outcomes}}
    with pytest.raises(ValueError, match="disagrees with the artifact's own recorded"):
        classify_adaptive_attacker_outcome(result)


def test_classify_adaptive_attacker_outcome_matches_the_real_task6_artifact_verdict():
    # Regression pin against the actual results/adaptive_attacker.json shape this project produced:
    # a candidate WAS selected (cost016_rate0.0050_sigma6.00_dirp1), attack utility held on both
    # held-out seeds (789, 1337), but MAD still flagged the trigger (mad_evasion_ok=False on both) --
    # this must land on "failed_held_out_evaluation", never "confirmed" and never "robust"/"secure".
    outcomes = [
        _outcome(feasible=True, mad_evasion_ok=False, attack_effective=True, clean_accuracy_ok=True),
        _outcome(feasible=True, mad_evasion_ok=False, attack_effective=True, clean_accuracy_ok=True),
    ]
    result = _adaptive_result("selected", outcomes=outcomes)
    assert (classify_adaptive_attacker_outcome(result)
            == "adaptive_candidate_failed_held_out_evaluation")


def test_adaptive_attacker_verdicts_never_claim_robust_or_secure():
    for verdict in ADAPTIVE_ATTACKER_VERDICTS:
        assert "robust" not in verdict
        assert "secure" not in verdict


# --------------------------------------------------------------------------------------------------
# src/precision_triage.py (extension Task 8, plan Task 8 Steps 1-3): the exploratory second-stage
# SHAP-attribution-concentration pilot's guarded arithmetic and leakage guard.
# --------------------------------------------------------------------------------------------------


def test_precision_cutoff_cannot_fit_on_test_or_poison_rows():
    # Step 3's leakage guard, verbatim from the plan.
    rows = [0.1, 0.2, 0.9]
    with pytest.raises(ValueError, match="clean validation"):
        fit_precision_cutoff(rows, partition="test", contains_poison=True)


def test_precision_cutoff_rejects_wrong_partition_even_without_poison():
    with pytest.raises(ValueError, match="clean validation"):
        fit_precision_cutoff([0.1, 0.2], partition="test", contains_poison=False)


def test_precision_cutoff_rejects_poison_even_with_the_right_partition_name():
    # A partition correctly NAMED "clean_validation" that is nonetheless contaminated with poisoned
    # rows must still be refused -- contains_poison is an independent leakage vector, not folded into
    # the partition-name check.
    with pytest.raises(ValueError, match="clean validation"):
        fit_precision_cutoff([0.1, 0.2], partition="clean_validation", contains_poison=True)


def test_precision_cutoff_accepts_the_clean_validation_partition():
    concentrations = np.linspace(0.0, 1.0, 101)   # 0.00, 0.01, ..., 1.00
    cutoff = fit_precision_cutoff(concentrations, partition="clean_validation", contains_poison=False)
    assert cutoff == pytest.approx(0.99, abs=1e-9)   # 99th percentile of a uniform 0..1 ladder


def test_precision_cutoff_rejects_empty_concentration_array():
    with pytest.raises(ValueError, match="empty concentration array"):
        fit_precision_cutoff([], partition="clean_validation", contains_poison=False)


def test_concentration_from_abs_shap_matches_hand_computation():
    # 3 rows, 4 features; trigger_indices = [1, 3].
    abs_shap = np.array([
        [1.0, 2.0, 1.0, 2.0],   # total=6, trig=2+2=4 -> 4/6
        [0.0, 0.0, 0.0, 0.0],   # total=0 -> concentration defined as 0.0, not NaN
        [4.0, 1.0, 4.0, 1.0],   # total=10, trig=1+1=2 -> 2/10
    ])
    conc = concentration_from_abs_shap(abs_shap, trigger_indices=[1, 3])
    assert conc == pytest.approx([4 / 6, 0.0, 2 / 10])


def test_concentration_from_abs_shap_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        concentration_from_abs_shap(np.zeros(5), trigger_indices=[0])


def test_two_stage_flag_is_a_subset_of_mad_flag():
    mad_flag = np.array([True, False, True, True, False])
    # concentrations for the 3 True positions (indices 0, 2, 3), in that order
    conc = np.array([0.9, 0.1, 0.5])
    two_stage = two_stage_flag_from_concentration(mad_flag, conc, cutoff=0.5)
    assert list(two_stage) == [True, False, False, True, False]
    assert np.all(two_stage <= mad_flag)   # never flags a row MAD did not already flag


def test_two_stage_flag_rejects_concentration_length_mismatch():
    mad_flag = np.array([True, False, True])
    with pytest.raises(ValueError, match="entries but mad_flag has"):
        two_stage_flag_from_concentration(mad_flag, flagged_concentration=[0.9], cutoff=0.5)


def test_poison_precision_recall_fpr_on_a_known_pool():
    # 5 rows: positions 0,1 are poison; flag catches 0 (TP) and 2 (FP); misses 1 (FN); 3,4 clean unflagged.
    is_poison = np.array([True, True, False, False, False])
    flagged = np.array([True, False, True, False, False])
    assert poison_recall_or_none(flagged, is_poison) == pytest.approx(0.5)     # 1/2 poison caught
    assert poison_precision(flagged, is_poison) == pytest.approx(0.5)          # 1/2 flagged are poison
    assert poison_fpr(flagged, is_poison) == pytest.approx(1 / 3)              # 1 FP / 3 clean rows


def test_poison_precision_and_fpr_return_none_when_denominator_is_zero():
    is_poison = np.array([True, False, False])
    nothing_flagged = np.array([False, False, False])
    assert poison_precision(nothing_flagged, is_poison) is None    # 0 flagged -> undefined precision
    all_poison = np.array([True])
    all_flagged = np.array([True])
    assert poison_fpr(all_flagged, all_poison) is None             # 0 clean rows -> undefined fpr


def test_poison_recall_or_none_converts_nan_to_none():
    # src.detectors.poison_recall returns NaN (not None) when nothing is poisoned in the pool.
    no_poison = np.array([False, False, False])
    flagged = np.array([True, False, False])
    assert poison_recall_or_none(flagged, no_poison) is None
