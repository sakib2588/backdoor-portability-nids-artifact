"""Aggregation guards for the cross-dataset extension analysis (extension Task 5, extended further by
later tasks -- see notes/20260729-decision-extension-preregistration.md for the frozen protocol these
guards enforce).

Every guard here exists to make a protocol violation fail loudly rather than pass quietly. Task 5
(scripts/15_secondary_detectors.py, scripts/16_secondary_adaptive_threshold.py) is the first place a
cross-dataset row set needs to be summarised and gated, so this module is introduced here rather than
later: a mixed-dataset row set, an incomplete seed x rate x cost grid, an un-overridden determinism
failure, or a post-hoc MAD-threshold choice would each silently corrupt the headline cross-dataset
replication claim if summarised or selected without being checked first. A later task (per the fuller
empty-arm/bootstrap guard set this module is expected to grow into) may extend this module further;
nothing here should be treated as the final word on its scope.

Task 9 (scripts/20_analyze_extension.py) is that later task. It adds three more functions below
`require_determinism_ok`:

- `summarise_cell` -- a per-(dataset, rate, cost) bootstrap-CI summary of one metric, built on the
  same `bootstrap_ci` (10,000 resamples, project protocol) already used everywhere else in this
  codebase. It is deliberately generic over `metric` (a flat key into each row) rather than hardcoded
  to ASR/recall, because Task 9 calls it against several different metrics per cell (asr,
  fixed_budget_recall, mad_recall, mad_fpr, clean_acc).
- `classify_replication` -- Task 9's Step 2 classifier. This is NOT `classify_evasion_window` above:
  that function answers a binary "is this cell inside the evasion window" question from two numbers;
  this one answers "what happened to the same operational pattern on THIS dataset" as one of four
  named outcomes, and it is applied per-dataset, per-cell -- never pooled across CTU and the secondary
  dataset, for the same reason `summarise_dataset_rows` refuses to pool rows.
- `classify_adaptive_attacker_outcome` -- Task 9's Step 3 classifier for `results/adaptive_attacker.json`
  (Task 6's score-aware adaptive attacker). It must land on exactly one of the four plan-mandated
  verdict strings and must never emit "robust" or "secure" -- those words assert a universal defensive
  guarantee this single search-budget-bounded, single-attacker-family experiment cannot support.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Sequence

import numpy as np

from src import config
from src.stats import bootstrap_ci


def _finite_or_none(x):
    """Map a non-finite float (e.g. a ratio with a zero denominator on a degenerate cell) to None, so
    downstream `json.dumps` output stays strict RFC-8259 JSON (no `NaN`/`Infinity` token) for
    `results/extension_analysis.json` / `results/extension_macros.json`. Same pattern as
    `scripts/05_run_detectors.py:_finite_or_none`, reimplemented here rather than imported because
    `scripts/` is not an importable package and this module must not depend on script-layer code.
    """
    return x if (x is not None and math.isfinite(x)) else None


def summarise_dataset_rows(rows: Sequence[dict]) -> dict:
    """Group detector-sweep rows by (rate, cost) and report per-cell mean ASR / fixed-budget Spectral
    recall / AC recall, refusing to summarise a row set that mixes more than one dataset.

    Every row must carry a "dataset" key. A summary spanning two datasets would silently average over
    incomparable pipelines (different feature counts, different constraint layers, a ~9x difference in
    training-set size) with no signal that it happened -- this is exactly the mistake that would make a
    "cross-dataset replication" claim meaningless, so it is refused outright rather than warned about.
    """
    if not rows:
        raise ValueError("summarise_dataset_rows: empty row set -- nothing to summarise")
    datasets = {r["dataset"] for r in rows}
    if len(datasets) > 1:
        raise ValueError(
            f"summarise_dataset_rows requires rows from a single dataset, got {sorted(datasets)} -- "
            "aggregating across datasets would silently average incomparable pipelines"
        )

    cell_asr = defaultdict(list)
    cell_fixed_recall = defaultdict(list)
    cell_ac_recall = defaultdict(list)
    for r in rows:
        key = (r["rate"], r["cost"])
        cell_asr[key].append(r["asr"])
        spec_recall = r.get("spectral", {}).get("recall")
        if spec_recall is not None:
            cell_fixed_recall[key].append(spec_recall)
        ac_recall = r.get("ac", {}).get("recall")
        if ac_recall is not None:
            cell_ac_recall[key].append(ac_recall)

    cells = []
    for key in sorted(cell_asr):
        rate, cost = key
        cells.append(dict(
            rate=rate, cost=cost,
            n_seeds=len(cell_asr[key]),
            mean_asr=float(np.mean(cell_asr[key])),
            mean_fixed_budget_recall=(float(np.mean(cell_fixed_recall[key]))
                                       if cell_fixed_recall[key] else None),
            mean_ac_recall=float(np.mean(cell_ac_recall[key])) if cell_ac_recall[key] else None,
        ))
    return dict(dataset=next(iter(datasets)), cells=cells)


def require_complete_grid(rows: Sequence[dict], seeds: Sequence, rates: Sequence[float],
                           costs: Sequence[int]) -> None:
    """Raise unless every (seed, rate, cost) cell in the declared grid is present in `rows`.

    Prevents a partial checkpoint, an early-stopped run, or a filtered `--cells` subset (see
    scripts/15_secondary_detectors.py's replication-seed mode) from being silently treated as a
    complete grid downstream -- the caller must know a cell is missing before averaging over it.
    """
    have = {(r["seed"], r["rate"], r["cost"]) for r in rows}
    missing = [(s, rt, c) for s in seeds for rt in rates for c in costs if (s, rt, c) not in have]
    if missing:
        shown = missing[:10]
        suffix = " ..." if len(missing) > 10 else ""
        raise ValueError(f"missing seed/rate/cost cell(s) in grid: {shown}{suffix}")


def select_primary_mad_result(mad_results: Dict[str, dict],
                               primary_z: float = config.MAD_Z_PRIMARY) -> dict:
    """Return the predeclared z-threshold's result only, guarding against a post-hoc pick among
    several tested thresholds. This project pre-registers z=3.0 (`config.MAD_Z_PRIMARY`) as the
    single headline MAD threshold; any other z value tested for context is exploratory and must never
    substitute for it in a reported claim.
    """
    key = str(primary_z)
    if key not in mad_results:
        raise ValueError(
            f"predeclared MAD z={primary_z} not present in mad_results "
            f"(available: {sorted(mad_results)}) -- cannot select the primary result"
        )
    return mad_results[key]


def classify_evasion_window(mean_asr: float, mean_fixed_budget_recall: float) -> bool:
    """Step 3's mechanical evasion-window rule (plan Task 5), applied identically on both datasets:
    the clean-label trigger is attack-effective (mean ASR >= config.ATTACK_EFFECTIVE_ASR) but
    Spectral's fixed removal budget recovers less than config.FIXED_BUDGET_MISS_RECALL of the poison.
    No cell is added or removed by hand after seeing whether the result looks favourable -- this
    function is the ONLY place that decision is made, and it takes only the two aggregate numbers.
    """
    return (mean_asr >= config.ATTACK_EFFECTIVE_ASR
            and mean_fixed_budget_recall < config.FIXED_BUDGET_MISS_RECALL)


def require_determinism_ok(rows: Sequence[dict], override: bool = False) -> None:
    """Refuse to proceed if any rerun cell's ASR/clean-accuracy determinism self-check
    (`determinism_ok`) came back False, unless explicitly overridden for debugging (Step 2).

    `determinism_ok is None` (no reference available to check against) is NOT a failure and does not
    raise -- only an explicit False does. A silent determinism failure means the cell this run's
    detector numbers are based on is NOT the same cell scripts/14_secondary_poison_sweep.py already
    validated (same seed/rate/cost, different ASR/clean-accuracy), so nothing downstream of it can be
    trusted without investigation.
    """
    failed = [(r.get("seed"), r.get("rate"), r.get("cost"))
              for r in rows if r.get("determinism_ok") is False]
    if failed and not override:
        shown = failed[:10]
        suffix = " ..." if len(failed) > 10 else ""
        raise ValueError(
            f"determinism_ok is False for {len(failed)} cell(s): {shown}{suffix} -- the rerun "
            "reproduced a DIFFERENT cell than the reference sweep recorded. Pass override=True only "
            "for debugging; do not trust these detector numbers otherwise."
        )


def summarise_cell(rows: Sequence[Dict[str, object]], metric: str) -> Dict[str, object]:
    """Bootstrap-CI summary of one metric across the seeds present in `rows` (plan Task 9 Step 1,
    verbatim signature).

    `rows` is expected to already be ONE (dataset, rate, cost) cell's per-seed rows -- this function
    does not group or filter by cell itself (that is the caller's job, mirroring how
    `summarise_dataset_rows` above groups before it means). It only sorts by seed (so `per_seed` has a
    stable, seed-ascending order regardless of input order) and defers the actual mean/CI arithmetic
    to `bootstrap_ci`, the same 10,000-resample percentile bootstrap used for every other headline
    number in this project -- there is no separate/weaker statistic for the extension.

    `mean`/`ci`/`per_seed` are passed through `_finite_or_none` so a degenerate metric (e.g. a ratio
    with a zero denominator slipping through from an upstream computation) cannot leak a `NaN`/
    `Infinity` token into the strict-JSON output, matching this project's established guard.
    """
    if not rows:
        raise ValueError(f"summarise_cell: empty row set for metric={metric!r} -- nothing to summarise")
    values = [row[metric] for row in sorted(rows, key=lambda row: row["seed"])]
    mean, ci = bootstrap_ci(values)
    return {
        "n": len(values),
        "mean": _finite_or_none(mean),
        "ci": [_finite_or_none(bound) for bound in ci],
        "per_seed": [_finite_or_none(float(v)) for v in values],
    }


def classify_replication(cell: Dict[str, object]) -> str:
    """Step 2's per-cell replication classification (plan Task 9, verbatim decision tree).

    This describes whether the SAME operational pattern (attack works, fixed-budget removal misses it,
    the budget-free MAD rule does or does not recover it) occurs on this one cell of this one dataset.
    It is deliberately NOT a pooled cross-dataset significance test -- CTU's 5 seeds and the secondary
    dataset's 5 seeds are not interchangeable paired replicates (different feature counts, different
    constraint layers, ~9x training-set size difference; the same reason `summarise_dataset_rows`
    refuses to average rows across datasets). Call this once per (dataset, rate, cost) cell; never on
    a value pooled across datasets.

    `cell` must carry `"asr"` and `"fixed_budget_recall"` keys, each a `summarise_cell`-shaped dict
    with a `"mean"`. `cell["mad_recall"]` is read ONLY when the first two branches do not already
    resolve the cell (i.e. only for a cell that is attack-effective AND fixed-budget-missed) --
    accessing a `"mad_recall"` key that the caller never attached is then a KeyError, which is the
    correct failure mode: it means this cell needed MAD data to classify but the caller did not supply
    it, and guessing instead of raising would silently corrupt the replication claim.
    """
    if cell["asr"]["mean"] < config.ATTACK_EFFECTIVE_ASR:
        return "attack_not_effective"
    if cell["fixed_budget_recall"]["mean"] >= config.FIXED_BUDGET_MISS_RECALL:
        return "no_fixed_budget_window"
    if cell["mad_recall"]["mean"] < config.FIXED_BUDGET_MISS_RECALL:
        return "mad_not_recovered"
    return "window_and_recovery_replicated"


ADAPTIVE_ATTACKER_VERDICTS = frozenset({
    "adaptive_evasion_confirmed_within_search_budget",
    "no_eligible_adaptive_candidate_within_search_budget",
    "adaptive_candidate_failed_held_out_evaluation",
    "adaptive_candidate_evaded_only_by_losing_attack_utility",
})


def classify_adaptive_attacker_outcome(result: Dict[str, object]) -> str:
    """Step 3's adaptive-attacker verdict classifier, applied to a `results/adaptive_attacker.json`-
    shaped dict (top-level keys `task, config, candidate_ledger, selection, evaluation` -- see
    `scripts/17_adaptive_attacker.py` / `src/adaptive_trigger.py`).

    Returns exactly one of `ADAPTIVE_ATTACKER_VERDICTS` above. Never "robust"/"secure" -- a single
    search-budget-bounded surrogate search against one MAD-thresholded detector on two held-out seeds
    cannot support a universal defensive-guarantee claim either way.

    Corrected against code review (2026-08-01): an earlier version of this function gated branch 2 on
    `evaluation.held_out_evasion_success` and then checked `attack_effective`/`clean_accuracy_ok` for
    branch 3. That made branch 3 mathematically unreachable, because
    `src.adaptive_trigger.evaluate_candidate` (the function that produces each `adaptive_outcomes`
    entry) already defines `evasion_success = feasible AND attack_effective AND clean_accuracy_ok AND
    mad_evasion_ok`, and `held_out_evasion_success = all(evasion_success across held-out seeds)`
    (`scripts/17_adaptive_attacker.py:477`). By the time `held_out_evasion_success` is False-checked
    and found True, `attack_effective`/`clean_accuracy_ok` are ALREADY guaranteed True on every seed by
    that same AND-gate -- there is no way to reach branch 3 through genuine artifact data that way.

    The fix: `evaluate_candidate` DOES expose an MAD-evasion signal independent of the attack-utility
    gates -- `mad_evasion_ok` (`mad_recall < config.FIXED_BUDGET_MISS_RECALL`), computed separately
    from `attack_effective`/`clean_accuracy_ok` before the four are AND-ed together into
    `evasion_success`. This function now reads that independent signal directly instead of reading the
    already-AND-ed `evasion_success`/`held_out_evasion_success` as its primary gate, which makes all
    four verdicts genuinely reachable:

    1. `selection.verdict != "selected"` (i.e. `"no_eligible_candidate"` or
       `"incomplete_surrogate_coverage"` -- see `scripts/17_adaptive_attacker.py`) means the surrogate
       search never found a candidate that jointly cleared feasibility, attack-effectiveness, and the
       clean-accuracy budget on all three surrogate seeds within `config.ADAPTIVE_SEARCH_MAX_CANDIDATES`
       candidates. Nothing was evaluated on the held-out seeds.
       -> "no_eligible_adaptive_candidate_within_search_budget"
    2. A candidate WAS selected and evaluated, but on at least one held-out seed the deployed trigger
       was infeasible OR MAD still flagged it (`not (feasible AND mad_evasion_ok)`) -- the detector
       caught it outright, independent of whether the attack would otherwise have been effective.
       -> "adaptive_candidate_failed_held_out_evaluation"
    3. MAD was evaded (feasible AND mad_evasion_ok) on EVERY held-out seed, but on at least one seed
       the attack failed the utility gates (`attack_effective` False, i.e. asr < config.
       ATTACK_EFFECTIVE_ASR, or `clean_accuracy_ok` False, i.e. clean_accuracy_drop exceeded config.
       CLEAN_ACCURACY_DROP_MAX) -- the trigger got past the detector only by no longer being a usable
       attack there.
       -> "adaptive_candidate_evaded_only_by_losing_attack_utility"
    4. MAD was evaded on every held-out seed AND attack utility held on every held-out seed.
       -> "adaptive_evasion_confirmed_within_search_budget"

    As a corruption guard, this function still cross-checks its own (mad-evasion AND utility) verdict
    against the artifact's own recorded `evaluation.held_out_evasion_success` -- the two must agree
    (`held_out_evasion_success` is exactly `mad_evaded_every_seed AND utility_intact_every_seed` by the
    AND-gate above), and a disagreement means the artifact was written by a different formula than the
    one this function assumes, which must fail loudly rather than emit a plausible-looking wrong verdict.
    """
    selection = result.get("selection", {})
    if selection.get("verdict") != "selected":
        return "no_eligible_adaptive_candidate_within_search_budget"

    evaluation = result.get("evaluation", {})
    outcomes = evaluation.get("adaptive_outcomes") or []
    if not outcomes:
        raise ValueError(
            "classify_adaptive_attacker_outcome: selection.verdict is 'selected' but "
            "evaluation.adaptive_outcomes is empty -- a candidate was chosen but nothing was recorded "
            "for the held-out evaluation. Treat as a corrupted or incomplete artifact, not a "
            "legitimate outcome."
        )

    mad_evaded_every_seed = all(
        bool(o.get("feasible")) and bool(o.get("mad_evasion_ok")) for o in outcomes
    )
    utility_intact_every_seed = all(
        bool(o.get("attack_effective")) and bool(o.get("clean_accuracy_ok")) for o in outcomes
    )

    computed_success = mad_evaded_every_seed and utility_intact_every_seed
    recorded_success = bool(evaluation.get("held_out_evasion_success"))
    if computed_success != recorded_success:
        raise ValueError(
            "classify_adaptive_attacker_outcome: computed held-out success "
            f"({computed_success}, from this seed's own feasible/mad_evasion_ok/attack_effective/"
            "clean_accuracy_ok flags) disagrees with the artifact's own recorded "
            f"evaluation.held_out_evasion_success ({recorded_success}) -- these must be the same "
            "quantity computed two ways (see scripts/17_adaptive_attacker.py's evasion_success "
            "AND-gate). A disagreement means this function's assumptions about the artifact's shape "
            "no longer hold; do not trust either value without investigating."
        )

    if not mad_evaded_every_seed:
        return "adaptive_candidate_failed_held_out_evaluation"
    if not utility_intact_every_seed:
        return "adaptive_candidate_evaded_only_by_losing_attack_utility"
    return "adaptive_evasion_confirmed_within_search_budget"
