#!/usr/bin/env python3
"""Emit Table II's merged ten-seed UNSW-NB15 column to a results manifest.

The base arm (5 seeds) and the replication arm (5 further seeds) each have a
results JSON. The merged n=10 mean and spread the manuscript prints in
`tab:secondary` had no artefact of their own, so a future re-run of either arm
would change the printed values with nothing on disk to flag the drift.

Reads the raw per-seed rows of both arms and nothing else. Spread convention is
population SD (ddof=0), matching the table caption.

Run:  python scripts/36_secondary_merged_column.py [--write-manifest]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

Z = "3.0"                                   # the locked primary MAD threshold
CELLS = [(0.005, 16), (0.010, 8), (0.010, 16)]


def arm(path: Path, rate: float, cost: int) -> list[float]:
    rows = json.loads(path.read_text())["rows"]
    return [
        r["mad"][Z]["recall"]
        for r in rows
        if abs(r["rate"] - rate) < 1e-9 and r["cost"] == cost
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    base_path = RESULTS / "secondary_adaptive_threshold.json"
    repl_path = RESULTS / "secondary_adaptive_threshold_replication.json"

    out = {
        "z": float(Z),
        "spread_convention": "population SD (ddof=0), matching the table caption",
        "sources": [str(base_path.relative_to(ROOT)), str(repl_path.relative_to(ROOT))],
        "cells": [],
    }

    header = f"{'cell':<12}{'base':>18}{'replication':>18}{'merged n=10':>18}"
    print(header)
    for rate, cost in CELLS:
        base, repl = arm(base_path, rate, cost), arm(repl_path, rate, cost)
        merged = base + repl
        row = {"rate": rate, "cost": cost}
        for name, vals in (("base", base), ("replication", repl), ("merged", merged)):
            row[name] = {
                "n": len(vals),
                "mean": st.mean(vals),
                "sd_population": st.pstdev(vals),
                "sd_sample": st.stdev(vals) if len(vals) > 1 else None,
                "per_seed": vals,
            }
        out["cells"].append(row)
        print(f"{rate}/{cost:<7}"
              f"{row['base']['mean']:>10.4f} ({row['base']['sd_population']:.3f})"
              f"{row['replication']['mean']:>10.4f} ({row['replication']['sd_population']:.3f})"
              f"{row['merged']['mean']:>10.4f} ({row['merged']['sd_population']:.3f})")

    if args.write_manifest:
        path = RESULTS / "secondary_merged_column.json"
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
