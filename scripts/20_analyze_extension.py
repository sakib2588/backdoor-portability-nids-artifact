"""Extension Task 9: aggregate CTU-13 + secondary-dataset detector results and the Task 6 adaptive-
attacker outcome WITHOUT pooling seeds across datasets or overstating the adaptive-attacker result.

Reads (never writes) five existing artifacts:
    results/detectors.json                          -- CTU-13 full grid (5 seeds x 5 rates x 5 costs)
    results/evasion_ablation.json                    -- CTU-13 MAD z-sweep, 5 target cells
    results/secondary_detectors.json                 -- secondary full grid (5 seeds x 4 rates x 2 costs)
    results/secondary_adaptive_threshold.json        -- secondary MAD z-sweep, 7 target cells
    results/adaptive_attacker.json                   -- Task 6 surrogate search + held-out evaluation

Writes:
    results/extension_analysis.json  -- full detail: per-dataset, per-cell bootstrap-CI summaries
                                         (src.analysis_extension.summarise_cell), each cell's
                                         replication classification (classify_replication), and the
                                         adaptive-attacker verdict (classify_adaptive_attacker_outcome).
    results/extension_macros.json    -- flat name -> value SSOT, same convention as results/macros.json
                                         (round to 4dp via r3, PascalCase keys, strict JSON asserted).

CTU-13 predates the "dataset" tag (added in extension Task 3 for the secondary dataset) -- its rows
are tagged config.PRIMARY_DATASET_ID IN MEMORY ONLY by the two `_load_ctu_*_rows` helpers below; the
JSON files on disk (results/detectors.json, results/evasion_ablation.json) are never mutated.

Per-dataset, per-cell MAD (adaptive-threshold) data does not cover the full rate x cost grid on either
dataset (CTU-13: 5 of 25 cells, via results/evasion_ablation.json's TARGET_CELLS; secondary: 7 of 8
cells, via results/secondary_adaptive_threshold.json's target_cells) -- both were derived mechanically
upstream to cover exactly the cells `classify_replication` needs MAD data for (every attack-effective,
fixed-budget-missed cell), so `classify_replication`'s short-circuit design (it only reads
`cell["mad_recall"]` when the first two branches do not already resolve the cell) means every cell this
script classifies is classifiable with the MAD data actually available. If a future rerun changes which
cells are attack-effective and that invariant breaks, `classify_replication` raises `KeyError` on the
missing `"mad_recall"` key rather than silently guessing -- this script does not add a broader guard on
top of that because the KeyError IS the guard (see src/analysis_extension.py's classify_replication
docstring).

Run:  python scripts/20_analyze_extension.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.analysis_extension import (
    classify_adaptive_attacker_outcome,
    classify_replication,
    require_complete_grid,
    require_determinism_ok,
    select_primary_mad_result,
    summarise_cell,
)

CTU_MAD_TARGET_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16), (0.1, 16)]


def _load(name: str) -> dict:
    path = config.RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run the upstream milestone/task first")
    return json.loads(path.read_text())


def _finite_or_none(x):
    """Same convention as src.analysis_extension._finite_or_none / scripts/05_run_detectors.py's
    module-level helper of the same name -- guards a computed ratio (mad_fpr below) against a
    zero-denominator degenerate cell leaking a non-finite token into the strict-JSON output."""
    return x if (x is not None and math.isfinite(x)) else None


def _group_by_cell(rows: Sequence[dict]) -> Dict[tuple, List[dict]]:
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["rate"], r["cost"])].append(r)
    return buckets


# --------------------------------------------------------------------------------------------------
# CTU-13 (primary dataset) -- tag in memory, do not mutate the files on disk.
# --------------------------------------------------------------------------------------------------


def _load_ctu_main_rows() -> List[dict]:
    """results/detectors.json's main_grid, tagged dataset=config.PRIMARY_DATASET_ID and flattened to
    the flat metric keys summarise_cell/classify_replication need: asr, clean_acc, fixed_budget_recall
    (= main_grid's "spectral.recall", the fixed-removal-budget convention, NOT MAD)."""
    detectors = _load("detectors.json")
    rows = []
    for r in detectors["main_grid"]:
        rows.append(dict(
            dataset=config.PRIMARY_DATASET_ID,
            seed=r["seed"], rate=r["rate"], cost=r["cost"],
            asr=r["asr"], clean_acc=r["clean_acc"],
            fixed_budget_recall=r["spectral"]["recall"],
            determinism_ok=r["determinism_ok"],
        ))
    return rows


def _load_ctu_mad_rows() -> List[dict]:
    """results/evasion_ablation.json's rows (CTU_MAD_TARGET_CELLS only), tagged dataset=
    config.PRIMARY_DATASET_ID, with the predeclared z=3.0 MAD result (via select_primary_mad_result,
    guarding against a post-hoc z pick) flattened to mad_recall/mad_fpr.

    evasion_ablation.json's "adaptive" dict has the identical {str(z): {...}} shape
    select_primary_mad_result was built for (verified against scripts/16_secondary_adaptive_threshold.
    py's own usage on the analogous secondary-dataset file). Unlike the secondary file, these rows do
    not carry a precomputed "fpr" -- it is derived here as n_false_positive / n_benign, where
    evasion_ablation.json's "n_benign" is already the clean (non-poisoned) pool size for this cell
    (n_poison + n_benign together are the full pool the detector scored; confirmed by cross-checking
    the resulting FPR, ~0.066, against config.SPECTRAL_ACCEPTED_FPR_MEAN=0.0596 and its documented
    0.0292-0.0709 range from the same 25-row file).
    """
    ablation = _load("evasion_ablation.json")
    rows = []
    for r in ablation["rows"]:
        key = (r["rate"], r["cost"])
        if key not in CTU_MAD_TARGET_CELLS:
            continue
        mad = select_primary_mad_result(r["adaptive"], primary_z=config.MAD_Z_PRIMARY)
        n_benign = r["n_benign"]
        mad_fpr = _finite_or_none(mad["n_false_positive"] / n_benign) if n_benign else None
        rows.append(dict(
            dataset=config.PRIMARY_DATASET_ID,
            seed=r["seed"], rate=r["rate"], cost=r["cost"],
            mad_recall=mad["recall"], mad_fpr=mad_fpr,
            determinism_ok=r["determinism_ok"],
        ))
    return rows


# --------------------------------------------------------------------------------------------------
# Secondary dataset (UNSW-NB15) -- already tagged dataset=config.SECONDARY_DATASET_ID on disk.
# --------------------------------------------------------------------------------------------------


def _load_secondary_main_rows() -> List[dict]:
    """results/secondary_detectors.json's main_grid, flattened the same way as _load_ctu_main_rows.
    Deliberately ignores this file's own embedded `spectral.mad_recall` scalar (present on every row,
    a single z=3.0 point with no predeclared-z dict to gate through select_primary_mad_result) in
    favour of results/secondary_adaptive_threshold.json below, which IS shaped for that guard and is
    the file the plan's own context notes name as canonical for MAD numbers -- one MAD source per
    dataset, not two possibly-drifting ones.
    """
    detectors = _load("secondary_detectors.json")
    rows = []
    for r in detectors["main_grid"]:
        rows.append(dict(
            dataset=config.SECONDARY_DATASET_ID,
            seed=r["seed"], rate=r["rate"], cost=r["cost"],
            asr=r["asr"], clean_acc=r["clean_acc"],
            fixed_budget_recall=r["spectral"]["recall"],
            determinism_ok=r["determinism_ok"],
        ))
    return rows


def _load_secondary_mad_rows() -> List[dict]:
    """results/secondary_adaptive_threshold.json's base-5-seed rows (its own EXTENSION_BASE_SEEDS
    grid, NOT results/secondary_adaptive_threshold_replication.json's disjoint EXTENSION_REPLICATION_
    SEEDS -- conflating the two would silently mix two different seed sets into one summarise_cell
    call). MAD recall/fpr at the predeclared z via select_primary_mad_result on row["mad"], which this
    file's own scripts/16_secondary_adaptive_threshold.py already uses identically.
    """
    adaptive = _load("secondary_adaptive_threshold.json")
    rows = []
    for r in adaptive["rows"]:
        mad = select_primary_mad_result(r["mad"], primary_z=config.MAD_Z_PRIMARY)
        rows.append(dict(
            dataset=config.SECONDARY_DATASET_ID,
            seed=r["seed"], rate=r["rate"], cost=r["cost"],
            mad_recall=mad["recall"], mad_fpr=_finite_or_none(mad.get("fpr")),
            determinism_ok=r["determinism_ok"],
        ))
    return rows


# --------------------------------------------------------------------------------------------------
# Per-dataset cell assembly (Step 1-2)
# --------------------------------------------------------------------------------------------------


def build_dataset_cells(main_rows: List[dict], mad_rows: List[dict],
                         seeds: Sequence[int], rates: Sequence[float],
                         costs: Sequence[int]) -> List[dict]:
    """One dataset's per-(rate, cost) cells: bootstrap-CI summaries of asr/fixed_budget_recall/
    clean_acc from `main_rows` (the full grid), plus mad_recall/mad_fpr from `mad_rows` where
    available, plus each cell's Step 2 replication classification.

    `require_complete_grid` is checked against the FULL declared (seeds, rates, costs) grid for
    `main_rows` (must be complete -- that is the base sweep, not a target-cell subset) and, per cell,
    against just that one (rate, cost) for `mad_rows` (which only covers a subset of cells by design;
    checking the full cross-product there would wrongly demand cells the MAD sweep never intended to
    run).
    """
    require_complete_grid(main_rows, seeds=seeds, rates=rates, costs=costs)
    require_determinism_ok(main_rows)
    if mad_rows:
        require_determinism_ok(mad_rows)

    main_by_cell = _group_by_cell(main_rows)
    mad_by_cell = _group_by_cell(mad_rows)

    cells = []
    for rate in rates:
        for cost in costs:
            key = (rate, cost)
            cell_main_rows = main_by_cell.get(key)
            if not cell_main_rows:
                continue
            cell = dict(
                rate=rate, cost=cost,
                asr=summarise_cell(cell_main_rows, "asr"),
                fixed_budget_recall=summarise_cell(cell_main_rows, "fixed_budget_recall"),
                clean_acc=summarise_cell(cell_main_rows, "clean_acc"),
            )
            cell_mad_rows = mad_by_cell.get(key)
            if cell_mad_rows:
                require_complete_grid(cell_mad_rows, seeds=seeds, rates=[rate], costs=[cost])
                cell["mad_recall"] = summarise_cell(cell_mad_rows, "mad_recall")
                cell["mad_fpr"] = summarise_cell(cell_mad_rows, "mad_fpr")
            cell["replication_classification"] = classify_replication(cell)
            cells.append(cell)
    return cells


def _round4(x):
    """Round to 4dp, passing None through -- matches results/macros.json's own `r3` convention
    (scripts/06_analyze_stats.py), renamed here since it actually rounds to 4dp, not 3."""
    return None if x is None else round(float(x), 4)


def main() -> None:
    config.ensure_dirs()

    # --- CTU-13 ---
    ctu_main = _load_ctu_main_rows()
    ctu_mad = _load_ctu_mad_rows()
    ctu_cells = build_dataset_cells(
        ctu_main, ctu_mad, seeds=config.EXTENSION_BASE_SEEDS,
        rates=config.POISON_RATES, costs=config.TRIGGER_COSTS,
    )

    # --- secondary (UNSW-NB15) ---
    secondary_main = _load_secondary_main_rows()
    secondary_mad = _load_secondary_mad_rows()
    secondary_cells = build_dataset_cells(
        secondary_main, secondary_mad, seeds=config.EXTENSION_BASE_SEEDS,
        rates=config.SECONDARY_RATES, costs=config.SECONDARY_COSTS,
    )

    # --- adaptive attacker (Task 6) ---
    adaptive_attacker_result = _load("adaptive_attacker.json")
    adaptive_verdict = classify_adaptive_attacker_outcome(adaptive_attacker_result)
    selection = adaptive_attacker_result.get("selection", {})
    evaluation = adaptive_attacker_result.get("evaluation", {})
    adaptive_outcomes = evaluation.get("adaptive_outcomes") or []
    # Re-derive the two independent per-seed signals classify_adaptive_attacker_outcome's verdict is
    # actually built from (see that function's docstring, corrected 2026-08-01), so the artifact
    # states not just the verdict but WHICH gate produced it -- did MAD still catch the trigger on some
    # seed, or did MAD get evaded but only by losing attack utility there.
    mad_evaded_every_seed = (
        all(bool(o.get("feasible")) and bool(o.get("mad_evasion_ok")) for o in adaptive_outcomes)
        if adaptive_outcomes else None
    )
    utility_intact_every_seed = (
        all(bool(o.get("attack_effective")) and bool(o.get("clean_accuracy_ok")) for o in adaptive_outcomes)
        if adaptive_outcomes else None
    )
    adaptive_attacker_summary = dict(
        verdict=adaptive_verdict,
        scope=(
            f"selection.verdict={selection.get('verdict')!r} on surrogate seeds "
            f"{adaptive_attacker_result.get('config', {}).get('surrogate_seeds')}; "
            f"evaluation on held-out seeds {evaluation.get('seeds')}; "
            f"held_out_evasion_success={evaluation.get('held_out_evasion_success')} "
            "(conjunction across every held-out seed, not an average)."
        ),
        selection_verdict=selection.get("verdict"),
        winner_candidate_id=(selection.get("winner") or {}).get("candidate_id"),
        held_out_evasion_success=evaluation.get("held_out_evasion_success"),
        mad_evaded_every_held_out_seed=mad_evaded_every_seed,
        attack_utility_intact_every_held_out_seed=utility_intact_every_seed,
        per_seed_evasion_success=evaluation.get("per_seed_evasion_success"),
        held_out_mean_asr=_round4(
            sum(o["asr"] for o in adaptive_outcomes) / len(adaptive_outcomes)
        ) if adaptive_outcomes else None,
        held_out_mean_mad_recall=_round4(
            sum(o["mad_recall"] for o in adaptive_outcomes) / len(adaptive_outcomes)
        ) if adaptive_outcomes else None,
        held_out_mean_clean_accuracy_drop=_round4(
            sum(o["clean_accuracy_drop"] for o in adaptive_outcomes) / len(adaptive_outcomes)
        ) if adaptive_outcomes else None,
    )

    analysis = dict(
        config=dict(
            attack_effective_asr=config.ATTACK_EFFECTIVE_ASR,
            fixed_budget_miss_recall=config.FIXED_BUDGET_MISS_RECALL,
            mad_z_primary=config.MAD_Z_PRIMARY,
            extension_base_seeds=list(config.EXTENSION_BASE_SEEDS),
        ),
        datasets={
            config.PRIMARY_DATASET_ID: dict(
                seeds=list(config.EXTENSION_BASE_SEEDS),
                rates=list(config.POISON_RATES), costs=list(config.TRIGGER_COSTS),
                mad_coverage_note=(
                    "MAD recall/fpr available only for the evasion-window cells plus the (0.1, 16) "
                    "rate-matched control (results/evasion_ablation.json's TARGET_CELLS) -- absent on "
                    "the remaining cells, which do not need it to classify (see module docstring)."
                ),
                cells=ctu_cells,
            ),
            config.SECONDARY_DATASET_ID: dict(
                seeds=list(config.EXTENSION_BASE_SEEDS),
                rates=list(config.SECONDARY_RATES), costs=list(config.SECONDARY_COSTS),
                mad_coverage_note=(
                    "MAD recall/fpr available for 7 of 8 cells (results/secondary_adaptive_threshold."
                    "json's mechanically-derived target_cells: every attack-effective cell plus the "
                    "(0.1, 16) control) -- absent only on (0.005, 8), which classifies as "
                    "attack_not_effective without needing it."
                ),
                cells=secondary_cells,
            ),
        },
        adaptive_attacker=adaptive_attacker_summary,
    )

    raw = json.dumps(analysis, indent=2)
    assert "NaN" not in raw and "Infinity" not in raw, "non-strict JSON token in extension_analysis.json"
    (config.RESULTS / "extension_analysis.json").write_text(raw)

    def _count(cells, label):
        return sum(1 for c in cells if c["replication_classification"] == label)

    macros = {
        "CtuReplicationCellCount": len(ctu_cells),
        "CtuAttackNotEffectiveCellCount": _count(ctu_cells, "attack_not_effective"),
        "CtuNoFixedBudgetWindowCellCount": _count(ctu_cells, "no_fixed_budget_window"),
        "CtuMadNotRecoveredCellCount": _count(ctu_cells, "mad_not_recovered"),
        "CtuWindowAndRecoveryReplicatedCellCount": _count(ctu_cells, "window_and_recovery_replicated"),
        "SecondaryReplicationCellCount": len(secondary_cells),
        "SecondaryAttackNotEffectiveCellCount": _count(secondary_cells, "attack_not_effective"),
        "SecondaryNoFixedBudgetWindowCellCount": _count(secondary_cells, "no_fixed_budget_window"),
        "SecondaryMadNotRecoveredCellCount": _count(secondary_cells, "mad_not_recovered"),
        "SecondaryWindowAndRecoveryReplicatedCellCount":
            _count(secondary_cells, "window_and_recovery_replicated"),
        "AdaptiveAttackerVerdict": adaptive_verdict,
        "AdaptiveAttackerSelectionVerdict": adaptive_attacker_summary["selection_verdict"],
        "AdaptiveAttackerHeldOutEvasionSuccess": adaptive_attacker_summary["held_out_evasion_success"],
        "AdaptiveAttackerMadEvadedEveryHeldOutSeed": adaptive_attacker_summary["mad_evaded_every_held_out_seed"],
        "AdaptiveAttackerUtilityIntactEveryHeldOutSeed":
            adaptive_attacker_summary["attack_utility_intact_every_held_out_seed"],
        "AdaptiveAttackerHeldOutMeanAsr": adaptive_attacker_summary["held_out_mean_asr"],
        "AdaptiveAttackerHeldOutMeanMadRecall": adaptive_attacker_summary["held_out_mean_mad_recall"],
        "AdaptiveAttackerHeldOutMeanCleanAccuracyDrop":
            adaptive_attacker_summary["held_out_mean_clean_accuracy_drop"],
    }
    raw_macros = json.dumps(macros, indent=2)
    assert "NaN" not in raw_macros and "Infinity" not in raw_macros, \
        "non-strict JSON token in extension_macros.json"
    (config.RESULTS / "extension_macros.json").write_text(raw_macros)

    print(f"CTU-13: {len(ctu_cells)} cells classified "
          f"({_count(ctu_cells, 'window_and_recovery_replicated')} window_and_recovery_replicated)")
    print(f"Secondary: {len(secondary_cells)} cells classified "
          f"({_count(secondary_cells, 'window_and_recovery_replicated')} window_and_recovery_replicated)")
    print(f"Adaptive attacker verdict: {adaptive_verdict}")
    print("wrote results/extension_analysis.json and results/extension_macros.json")


if __name__ == "__main__":
    main()
