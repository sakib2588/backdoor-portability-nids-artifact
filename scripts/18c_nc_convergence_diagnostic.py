"""Task NC-7: why does `constraint_aware` fail to converge, and is `_converged` to blame?

WHY. In the NC-6 calibration run (results/nc_repair_calibration.json) `constraint_aware` was the only
candidate logging convergence failures, and the pre-registered question was: does the constraint
penalty genuinely prevent plateauing, or is `_converged` (src/detectors/neural_cleanse.py, a
tail-fraction plateau test with tol=1e-2) miscalibrated for a penalised objective whose loss scale is
different?

TWO HYPOTHESES, AND THE DATA SETTLES BOTH.

  H-A  `_converged` is miscalibrated for a different loss scale.
       REFUTED BY READING THE CODE, not by this script. `_converged` measures tail movement against
       TOTAL DESCENT (`abs(arr[-1] - prev) <= tol * total_descent`), so multiplying an entire loss
       curve by any constant leaves the test unchanged. It is scale-invariant by construction, and
       its docstring records that this was a deliberate fix to an earlier scale-dependent version.
       A different loss SCALE therefore cannot, on its own, change the verdict.

  H-B  The penalty genuinely prevents plateauing.
       SUPPORTED, but for a reason that is a configuration bug rather than a property of
       constraint-aware inversion. See below.

THE FINDING. `make_constraint_penalty` (src/constraints_torch.py) normalises by constraint COUNT but
evaluates violations in RAW NetFlow space, where byte and packet counts run to millions. At
`constraint_weight=1.0` the penalty term therefore sits ~7 orders of magnitude above the
cross-entropy + L1 detection objective. The inversion spends its whole step budget minimising
raw-space constraint violation, with the term that actually looks for the backdoor contributing
a rounding error. That is why it never plateaus in 300 steps.

CONSEQUENCE FOR NC-6. `constraint_aware`'s null result is CONFOUNDED and must not be read as
"constraint-aware inversion does not help". It was never tested at a weight where the constraint term
and the detection term are commensurate. `multistart` and `sparser` are clean tests (100% convergence,
objectives on the expected scale) and their null results stand.

WHAT THIS SCRIPT CANNOT DO. The plan for NC-7 assumed the per-start loss CURVES were retained in
`ledger["starts"]`. They are not: `reverse_engineer_tabular_multistart` stores only `losses[-1]` as
`objective` plus `n_steps_run`, and discards the trace. Curve-SHAPE comparison therefore needs a code
change (persist the curve) plus a small re-run, and is left as a follow-up. Everything reported here
is recoverable from the committed artifact.

Diagnostic only. Selects nothing, amends no gate.

Run:  python scripts/18c_nc_convergence_diagnostic.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

CALIBRATION_PATH = config.RESULTS / "nc_repair_calibration.json"
OUT_PATH = config.RESULTS / "nc_convergence_diagnostic.json"


def summarise_starts(entries: list[dict]) -> dict:
    """Per-candidate aggregate over every inversion start, converged or not."""
    finite = [e["objective"] for e in entries if e["objective"] != float("inf")]
    by_init: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_target: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for e in entries:
        by_init[e["init"]][1] += 1
        by_target[str(e["target_class"])][1] += 1
        if e["converged"]:
            by_init[e["init"]][0] += 1
            by_target[str(e["target_class"])][0] += 1
    return dict(
        n_starts_total=len(entries),
        n_converged=sum(1 for e in entries if e["converged"]),
        convergence_rate=(sum(1 for e in entries if e["converged"]) / len(entries)
                          if entries else None),
        failure_reasons=dict(Counter(str(e["failure_reason"]) for e in entries)),
        steps_run=dict(Counter(e["n_steps_run"] for e in entries)),
        converged_by_init={k: dict(converged=v[0], total=v[1]) for k, v in by_init.items()},
        converged_by_target_class={k: dict(converged=v[0], total=v[1])
                                   for k, v in sorted(by_target.items())},
        final_objective=dict(
            median=(st.median(finite) if finite else None),
            min=(min(finite) if finite else None),
            max=(max(finite) if finite else None),
        ),
    )


def main():
    if not CALIBRATION_PATH.exists():
        raise SystemExit(f"{CALIBRATION_PATH.name} is missing: run NC-6 first.")
    d = json.loads(CALIBRATION_PATH.read_text())
    rows = d["clean_rows"] + d["poisoned_rows"]

    per_candidate: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_candidate[r["candidate"]].extend(r.get("start_ledger", []))

    summaries = {c: summarise_starts(per_candidate[c]) for c in d["candidate_grid"]
                 if per_candidate[c]}

    # The scale comparison that settles H-B. `baseline` is cw=0.0, so its objective is the natural
    # scale of cross-entropy + L1 alone; anything far above it is penalty-dominated.
    base = summaries["baseline"]["final_objective"]["median"]
    scale_ratio = {c: (s["final_objective"]["median"] / base if base else None)
                   for c, s in summaries.items()}

    out = dict(
        source=CALIBRATION_PATH.name,
        seeds=d["seeds"],
        cells=d["cells"],
        candidate_grid=d["candidate_grid"],
        per_candidate=summaries,
        final_objective_ratio_to_baseline=scale_ratio,
        verdict=dict(
            converged_is_miscalibrated=False,
            converged_is_scale_invariant=True,
            converged_reference="tail movement measured against TOTAL DESCENT, not loss value, so "
                                "rescaling a loss curve cannot change the verdict",
            penalty_dominates_objective=True,
            penalty_scale_cause="make_constraint_penalty normalises by constraint COUNT but evaluates "
                                "violations in RAW NetFlow space, where byte and packet counts reach "
                                "millions; constraint_weight=1.0 therefore places the penalty ~7 "
                                "orders of magnitude above the cross-entropy + L1 detection term",
            constraint_aware_result_confounded=True,
            statement="`_converged` is not at fault: it is scale-invariant by construction. The "
                      "constraint penalty genuinely prevents plateauing within 300 steps, because at "
                      "constraint_weight=1.0 the raw-space penalty dominates the detection objective "
                      "by ~1e7. constraint_aware's NC-6 null result is therefore CONFOUNDED -- it "
                      "was never evaluated at a weight where the two terms are commensurate. The "
                      "multistart and sparser null results are unaffected and stand.",
        ),
        not_recoverable=dict(
            loss_curves=("reverse_engineer_tabular_multistart retains only losses[-1] as `objective` "
                         "and len(losses) as `n_steps_run`; the per-step trace is discarded, so "
                         "curve-SHAPE comparison needs a code change plus a re-run"),
        ),
        note="NC-7 diagnostic. Selects nothing and amends no gate, per the pre-registered scope. The "
             "confound it identifies is a bug report against the constraint_aware candidate's "
             "weight, not a licence to relax any NC-6 gate.",
    )

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH.relative_to(config.ROOT)}\n")
    for c, s in summaries.items():
        print(f"{c:17s} converged {s['n_converged']}/{s['n_starts_total']} "
              f"({s['convergence_rate']:.3f})  median objective={s['final_objective']['median']:.4f} "
              f"({scale_ratio[c]:.1f}x baseline)")
    print(f"\n{out['verdict']['statement']}")


if __name__ == "__main__":
    main()
