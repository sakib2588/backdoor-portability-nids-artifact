#!/usr/bin/env python3
"""Emit Table II's n=30 merged UNSW-NB15 column (M5 fix, PC review 2026-08-28).

Extends scripts/36_secondary_merged_column.py's base+replication (n=10) merge with a third
20-seed batch (results/secondary_adaptive_threshold_batch3.json, seeds 501-520) on the same 3
window cells, reaching n=30 merged. Also merges Spectral ranking AUC across all three arms,
since the paper quotes a merged AUC figure (Results, "A second NIDS dataset") that the n=10
script did not compute.

Reads the raw per-seed rows of all three arms and nothing else. Spread convention is population
SD (ddof=0), matching the table caption.

Run:  python scripts/61_secondary_merged_column_n30.py --write-manifest
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

Z = "3.0"
CELLS = [(0.005, 16), (0.010, 8), (0.010, 16)]
ARMS = [
    ("base", RESULTS / "secondary_adaptive_threshold.json"),
    ("replication", RESULTS / "secondary_adaptive_threshold_replication.json"),
    ("batch3", RESULTS / "secondary_adaptive_threshold_batch3.json"),
]


def arm_rows(path: Path, rate: float, cost: int) -> list[dict]:
    rows = json.loads(path.read_text())["rows"]
    return [r for r in rows if abs(r["rate"] - rate) < 1e-9 and r["cost"] == cost]


def mad_recall(r: dict) -> float:
    return r["mad"][Z]["recall"]


def mad_fpr(r: dict) -> float:
    return r["mad"][Z]["fpr"]


def stat_block(vals: list[float]) -> dict:
    return {
        "n": len(vals),
        "mean": st.mean(vals),
        "sd_population": st.pstdev(vals),
        "sd_sample": st.stdev(vals) if len(vals) > 1 else None,
        "per_seed": vals,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    for name, path in ARMS:
        if not path.exists():
            raise SystemExit(f"{path} not found -- run its generating script first")

    out = {
        "z": float(Z),
        "spread_convention": "population SD (ddof=0), matching the table caption",
        "sources": [str(p.relative_to(ROOT)) for _, p in ARMS],
        "cells": [],
    }

    header = f"{'cell':<12}{'n=10 merged':>18}{'n=30 merged':>18}{'AUC n=30':>14}{'FPR n=30':>12}"
    print(header)
    for rate, cost in CELLS:
        rows_by_arm = {name: arm_rows(path, rate, cost) for name, path in ARMS}
        all_rows = [r for rows in rows_by_arm.values() for r in rows]

        recalls = {name: [mad_recall(r) for r in rows] for name, rows in rows_by_arm.items()}
        fprs = {name: [mad_fpr(r) for r in rows] for name, rows in rows_by_arm.items()}
        aucs_all = [r["spectral_auc"] for r in all_rows if r.get("spectral_auc") is not None]

        merged10 = recalls["base"] + recalls["replication"]
        merged30 = merged10 + recalls["batch3"]
        fpr30 = fprs["base"] + fprs["replication"] + fprs["batch3"]

        row = {
            "rate": rate, "cost": cost,
            "base": stat_block(recalls["base"]),
            "replication": stat_block(recalls["replication"]),
            "batch3": stat_block(recalls["batch3"]),
            "merged_n10": stat_block(merged10),
            "merged_n30": stat_block(merged30),
            "fpr_merged_n30": stat_block(fpr30),
            "spectral_auc_n30": {"n": len(aucs_all), "mean": st.mean(aucs_all)} if aucs_all else None,
        }
        out["cells"].append(row)
        print(f"{rate}/{cost:<7}"
              f"{row['merged_n10']['mean']:>10.4f} ({row['merged_n10']['sd_population']:.3f})"
              f"{row['merged_n30']['mean']:>10.4f} ({row['merged_n30']['sd_population']:.3f})"
              f"{row['spectral_auc_n30']['mean'] if row['spectral_auc_n30'] else float('nan'):>10.4f}"
              f"{row['fpr_merged_n30']['mean']*100:>10.2f}%")

    if args.write_manifest:
        path = RESULTS / "secondary_merged_column_n30.json"
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
