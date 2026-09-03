#!/usr/bin/env python3
"""The displacement (sigma) axis of the trigger, from the adaptive-attacker sweep already on disk.

The manuscript varies trigger cost (feature COUNT) and never reports how attack success and the
Spectral detector's recall and ranking signal move with trigger DISPLACEMENT. The adaptive-attacker
surrogate stage already swept sigma in {1,...,6} at rates {0.005, 0.01} and costs {8, 16} on seeds
{42, 123, 456}, scoring asr, fixed-budget recall, MAD recall and Spectral AUC per candidate. This
reads that checkpoint and reports the axis. No compute.

Two limits belong with every number here. The rows are the SURROGATE seeds (the selection stage),
not the held-out evaluation seeds. And the ledger is not a full factorial above sigma 1: the
candidate cap left 12 rows per sigma at 2-6 against 24 at sigma 1, so some (sigma, cost, rate,
direction) cells are absent and are reported as absent rather than filled.

ASSERT ONLY, NEVER GENERATE: writes results/displacement_axis.json and prints a table.

Run:  .venv/bin/python scripts/92_displacement_axis_analysis.py
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

SRC = config.RESULTS / "adaptive_attacker_surrogate.checkpoint.json"
OUT = config.RESULTS / "displacement_axis.json"
METRICS = ("asr", "fixed_budget_recall", "mad_recall", "mad_fpr", "spectral_auc")


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean over seeds per (sigma, cost, rate, direction) of every metric; infeasible rows counted
    out. Direction and rate are kept as columns so nothing is pooled across them silently."""
    groups: dict[tuple, list[dict]] = {}
    infeasible: dict[tuple, int] = {}
    for r in rows:
        k = (float(r["n_sigma"]), int(r["cost"]), float(r["rate"]), int(r["direction"]))
        if not r.get("feasible", True):
            infeasible[k] = infeasible.get(k, 0) + 1
            continue
        groups.setdefault(k, []).append(r)
    table = []
    for k in sorted(set(groups) | set(infeasible)):
        g = groups.get(k, [])
        row = dict(n_sigma=k[0], cost=k[1], rate=k[2], direction=k[3],
                   n_seeds=len(g), n_infeasible=infeasible.get(k, 0))
        for m in METRICS:
            vals = [r[m] for r in g if r.get(m) is not None]
            row[f"mean_{m}"] = st.mean(vals) if vals else None
        table.append(row)
    return table


def monotonicity(table: list[dict], cost: int, rate: float = 0.005, direction: int = 1) -> dict:
    """Spearman rho of each metric against sigma, on one (cost, rate, direction) line."""
    line = sorted((t for t in table if t["cost"] == cost and t["rate"] == rate
                   and t["direction"] == direction and t["n_seeds"] > 0),
                  key=lambda t: t["n_sigma"])
    out = dict(cost=cost, rate=rate, direction=direction, n_sigma_levels=len(line))
    sig = [t["n_sigma"] for t in line]
    for m in METRICS:
        vals = [t[f"mean_{m}"] for t in line]
        if len(vals) >= 3 and None not in vals:
            out[f"{m}_rho"] = float(spearmanr(sig, vals).statistic)
        else:
            out[f"{m}_rho"] = None
    return out


def main() -> int:
    blob = json.loads(SRC.read_text())
    rows = list(blob["rows"].values())
    table = aggregate(rows)
    lines = [monotonicity(table, cost=c, rate=r, direction=d)
             for c in (8, 16) for r in (0.005, 0.01) for d in (1, -1)]
    OUT.write_text(json.dumps(dict(
        _schema="displacement_axis/1", _source=str(SRC.relative_to(config.ROOT)),
        _caveats=["surrogate seeds only (selection stage), not the held-out evaluation seeds",
                  "ledger is not a full factorial above sigma 1; absent cells are absent"],
        n_rows=len(rows), table=table, monotonicity=lines), indent=2))
    print(f"{'sigma':>5} {'cost':>4} {'rate':>6} {'dir':>3} {'n':>2} {'asr':>6} {'fixed':>6} "
          f"{'mad':>6} {'auc':>6}")
    for t in table:
        f = lambda v: "  --  " if v is None else f"{v:6.3f}"
        print(f"{t['n_sigma']:>5} {t['cost']:>4} {t['rate']:>6} {t['direction']:>3} {t['n_seeds']:>2} "
              f"{f(t['mean_asr'])} {f(t['mean_fixed_budget_recall'])} {f(t['mean_mad_recall'])} "
              f"{f(t['mean_spectral_auc'])}")
    print()
    for l in lines:
        if l["n_sigma_levels"] >= 3:
            print(f"cost {l['cost']} rate {l['rate']} dir {l['direction']}: "
                  f"rho vs sigma  asr={l['asr_rho']}  fixed={l['fixed_budget_recall_rho']}  "
                  f"auc={l['spectral_auc_rho']}  ({l['n_sigma_levels']} levels)")
    print(f"wrote {OUT.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
