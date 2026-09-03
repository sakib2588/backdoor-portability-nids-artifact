#!/usr/bin/env python3
"""Recompute the paper_access literals that the Tier 1 ledger cannot trace to a stored value.

WHY (2026-09-03). scripts/34 traces a literal by matching it against a scalar stored in
results/*.json. That catches an invented number, but it reports UNTRACED for any figure the paper
derives by aggregating -- a mean over a cost group, a standard deviation across seeds, a sum of two
counts. Running the ledger on paper_access/ for the first time produced 18 UNTRACED literals; four
were template placeholders and the remaining fourteen were all derived, not invented.

An allowlist would silence those and check nothing. This script instead RECOMPUTES each one from
the committed results and asserts it equals what the manuscript prints, so a future edit that drifts
a number fails here rather than reaching a reviewer.

Every expected value below is the literal as printed in paper_access/sections/*.tex, with the file
and line it appears on. Change one only after re-deriving it.

Run:  python scripts/78_derived_literal_check.py
Exit: 0 if every derived literal reproduces, 1 otherwise.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

CTU_WINDOW = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]
UNSW_WINDOW = [(0.005, 16), (0.01, 8), (0.01, 16)]
ANCHOR = (0.005, 16)


def check(label: str, computed: str, printed: str, where: str, out: list) -> None:
    ok = computed == printed
    out.append((ok, label, computed, printed, where))


def main() -> int:
    grid = json.load((RESULTS / "full_grid_mad_sweep.json").open())["rows"]
    sec = json.load((RESULTS / "secondary_detectors.json").open())["main_grid"]
    checks: list = []

    # --- Table "CTU-13 grid summarized by trigger cost" (05_results.tex:241-243).
    # Cost groups over all five rates at z=3.0. The FPR column divides by CLEAN rows only
    # (n_benign - n_poison); the stored `fpr` field uses all benign rows and must not be read.
    groups = {
        "<=4": ([r for r in grid if r["cost"] <= 4], ("0.0726", "0.0077", "7.01", "0.0603", "0.0545", "0.5023")),
        "8":   ([r for r in grid if r["cost"] == 8],  ("0.9998", "0.0004", "5.92", "0.4370", "0.4372", "0.9720")),
        "16":  ([r for r in grid if r["cost"] == 16], ("1.0000", "0.0000", "5.25", "0.6906", "0.3895", "0.9889")),
    }
    for name, (rs, exp) in groups.items():
        mad = [r["adaptive"]["3.0"]["recall"] for r in rs]
        fpr = [100 * r["adaptive"]["3.0"]["n_false_positive"] / (r["n_benign"] - r["n_poison"]) for r in rs]
        fx = [r["fixed_budget_recall"] for r in rs]
        auc = [r["spectral_auc"] for r in rs]
        for lbl, got, want in (
            ("MAD recall", f"{st.mean(mad):.4f}", exp[0]),
            ("MAD recall SD", f"{st.stdev(mad):.4f}", exp[1]),
            ("MAD FPR %", f"{st.mean(fpr):.2f}", exp[2]),
            ("fixed recall", f"{st.mean(fx):.4f}", exp[3]),
            ("fixed recall SD", f"{st.stdev(fx):.4f}", exp[4]),
            ("spectral AUC", f"{st.mean(auc):.4f}", exp[5]),
        ):
            check(f"cost {name}: {lbl}", got, want, "05_results.tex:241-243", checks)

    # --- Per-cost AUCs quoted in that table's caption (05_results.tex:232).
    for cost, want in ((1, "0.5016"), (2, "0.5024"), (4, "0.5029")):
        v = [r["spectral_auc"] for r in grid if r["cost"] == cost]
        check(f"cost {cost} mean spectral AUC", f"{st.mean(v):.4f}", want, "05_results.tex:232", checks)

    # --- Activation Clustering cost-16 mean over five rates (05_results.tex:142).
    ac16 = [r["ac_fixed_recall"] for r in grid if r["cost"] == 16]
    check("AC cost-16 fixed recall", f"{st.mean(ac16):.4f}", "0.6801", "05_results.tex:142", checks)

    # --- Budget-free rule on the CTU-13 window, and the anchor cell (06_discussion.tex:71-73).
    win = [r for r in grid if (r["rate"], r["cost"]) in CTU_WINDOW]
    anc = [r for r in grid if (r["rate"], r["cost"]) == ANCHOR]
    winfpr = [100 * r["adaptive"]["3.0"]["n_false_positive"] / (r["n_benign"] - r["n_poison"]) for r in win]
    check("window MAD FPR %", f"{st.mean(winfpr):.2f}", "6.58", "06_discussion.tex:71", checks)
    check("window mean precision",
          f"{st.mean(r['adaptive']['3.0']['n_true_positive'] / r['adaptive']['3.0']['n_flagged'] for r in win):.3f}",
          "0.105", "06_discussion.tex:71", checks)
    check("anchor mean precision",
          f"{st.mean(r['adaptive']['3.0']['n_true_positive'] / r['adaptive']['3.0']['n_flagged'] for r in anc):.4f}",
          "0.0714", "06_discussion.tex:71", checks)
    check("anchor flagged flows",
          f"{st.mean(r['adaptive']['3.0']['n_flagged'] for r in anc):,.0f}",
          "8,017", "06_discussion.tex:73", checks)

    # --- Activation Clustering on the UNSW-NB15 window (05_results.tex:359 and :410).
    acw = [r["ac"]["recall"] for r in sec if (r.get("rate"), r.get("cost")) in UNSW_WINDOW]
    check("AC UNSW window recall", f"{st.mean(acw):.4f}", "0.1443", "05_results.tex:359", checks)

    # --- Stated arithmetic identities, which have no disk value by construction.
    check("CTU-13 release total", f"{143046 + 55082:,}", "198,128", "04_methods.tex:49", checks)
    check("median poison rank", f"{1022 + 286:,}", "1,308", "05_results.tex:290", checks)

    width = max(len(c[1]) for c in checks)
    failed = 0
    for ok, label, got, want, where in checks:
        if not ok:
            failed += 1
        print(f"  {'ok ' if ok else 'FAIL'}  {label:<{width}}  computed {got:>9s}  printed {want:>9s}   {where}")
    print(f"\n{len(checks) - failed} of {len(checks)} derived literals reproduce from the committed results")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
