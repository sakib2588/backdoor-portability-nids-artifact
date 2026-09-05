#!/usr/bin/env python3
"""Anchor-cell budget-free numbers, with the estimator named for each one.

The manuscript prints three numbers together for the anchor cell (CTU-13, poison rate
0.005, trigger cost 16) under the budget-free MAD rule at z = 3.0: a flagged count, the
poison count recovered, and a precision. A reviewer who divides the first two does not get
the third, which reads as an arithmetic error and was logged as one in HANDOVER_LAPTOP.md.

It is not an error. The flagged count is the mean over seeds of the per-seed flagged counts,
while the precision is the mean over seeds of the per-seed precisions. Those are different
estimators, and the mean of a ratio is not the ratio of the means. This script computes both
so the manuscript can name which one it prints, and so the literal ledger can trace all
three to disk -- before this, none of the three existed in any results file.

Source: results/full_grid_mad_sweep.json, whose rows carry n_flagged, n_true_positive,
n_false_positive and fpr per seed at each of the three pre-registered z-thresholds.

ASSERT ONLY, NEVER GENERATE: reads a committed results file and writes a derived one.
It does not train, score, or touch the manuscript.

Run:  .venv/bin/python scripts/109_anchor_cell_budget_free.py
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import config

SRC = config.RESULTS / "full_grid_mad_sweep.json"
OUT = config.RESULTS / "anchor_cell_budget_free.json"
RATE, COST, Z = 0.005, 16, "3.0"


def main() -> int:
    rows = json.loads(SRC.read_text())["rows"]
    cells = [r for r in rows if r["rate"] == RATE and r["cost"] == COST]
    if not cells:
        raise SystemExit(f"no rows at rate={RATE} cost={COST} in {SRC.name}")

    per_seed = []
    for r in sorted(cells, key=lambda x: x["seed"]):
        a = r["adaptive"][Z]
        per_seed.append({
            "seed": r["seed"],
            "n_flagged": a["n_flagged"],
            "n_true_positive": a["n_true_positive"],
            "n_false_positive": a["n_false_positive"],
            "recall": a["recall"],
            "fpr": a["fpr"],
            "precision": a["n_true_positive"] / a["n_flagged"],
        })

    flagged = [s["n_flagged"] for s in per_seed]
    prec = [s["precision"] for s in per_seed]
    mean_flagged = st.mean(flagged)
    n_poison = per_seed[0]["n_true_positive"]

    out = {
        "_schema": "anchor_cell_budget_free/1",
        "_instrument": "scripts/109_anchor_cell_budget_free.py",
        "_source": SRC.name,
        "cell": {"corpus": "CTU-13", "rate": RATE, "cost": COST, "z": float(Z),
                 "n_seeds": len(per_seed), "n_benign_denominator": cells[0]["n_benign"]},
        "_estimators": (
            "mean_n_flagged is the mean of per-seed flagged counts. "
            "mean_of_per_seed_precision is the mean of per-seed precisions. "
            "precision_of_the_means is n_poison / mean_n_flagged. "
            "The last two differ because the mean of a ratio is not the ratio of the means; "
            "the manuscript prints mean_n_flagged and mean_of_per_seed_precision."
        ),
        "n_poison_recovered": n_poison,
        "recall_every_seed": min(s["recall"] for s in per_seed),
        "mean_n_flagged": mean_flagged,
        "mean_n_flagged_rounded": round(mean_flagged),
        "mean_of_per_seed_precision": st.mean(prec),
        "precision_of_the_means": n_poison / mean_flagged,
        "mean_fpr": st.mean(s["fpr"] for s in per_seed),
        "per_seed": per_seed,
    }
    OUT.write_text(json.dumps(out, indent=2))

    print(f"anchor cell: CTU-13 rate {RATE} cost {COST}, z={Z}, {len(per_seed)} seeds")
    print(f"  recall on every seed        {min(s['recall'] for s in per_seed)}")
    print(f"  poison recovered            {n_poison}")
    print(f"  mean n_flagged              {mean_flagged}  -> {round(mean_flagged)}")
    print(f"  mean of per-seed precision  {st.mean(prec):.6f}  -> {st.mean(prec):.4f}")
    print(f"  precision of the means      {n_poison/mean_flagged:.6f}  -> {n_poison/mean_flagged:.4f}")
    print(f"  mean fpr                    {st.mean(s['fpr'] for s in per_seed):.6f}")
    print(f"\nwrote {OUT.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
