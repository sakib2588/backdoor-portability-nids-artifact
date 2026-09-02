#!/usr/bin/env python3
"""Mean Spectral recall over the attack-effective cells, selected without a recall condition.

The manuscript's headline 0.1425 is a mean over the four cells that satisfy BOTH
mean ASR >= 0.8 and fixed-budget recall < 0.5. A mean over a set chosen partly on
recall is bounded by that choice, so on its own it cannot show how much of the
number is the detector and how much is the selection. This computes the companion
figure: the same mean over every cell that clears the attack-side criterion alone.

Reads results/detectors.json raw per-seed rows only. No dependence on
analysis.json or macros.json, so it is an independent second path.

Run:  python scripts/35_unconditioned_effective_recall.py [--write-manifest]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

ASR_MIN = 0.8          # attack-side criterion, same constant as scripts/06_analyze_stats.py
RECALL_MAX = 0.5       # recall-side criterion, applied only for the conditioned figure


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    grid = json.loads((RESULTS / "detectors.json").read_text())["main_grid"]

    cells = defaultdict(list)
    for row in grid:
        cells[(row["rate"], row["cost"])].append(row)

    per_cell = []
    for (rate, cost), rows in sorted(cells.items()):
        asr = st.mean(r["asr"] for r in rows)
        recall = st.mean(r["spectral"]["recall"] for r in rows)
        auc_vals = [r["spectral"]["auc"] for r in rows if r["spectral"].get("auc") is not None]
        per_cell.append({
            "rate": rate,
            "cost": cost,
            "n_seeds": len(rows),
            "mean_asr": asr,
            "mean_spectral_recall": recall,
            "mean_spectral_auc": st.mean(auc_vals) if auc_vals else None,
            "attack_effective": asr >= ASR_MIN,
            "in_evasion_window": asr >= ASR_MIN and recall < RECALL_MAX,
        })

    effective = [c for c in per_cell if c["attack_effective"]]
    window = [c for c in per_cell if c["in_evasion_window"]]

    out = {
        "criterion": {"asr_min": ASR_MIN, "recall_max": RECALL_MAX},
        "source": "results/detectors.json:main_grid (raw per-seed rows)",
        "n_attack_effective_cells": len(effective),
        "n_evasion_window_cells": len(window),
        "unconditioned_mean_recall": st.mean(c["mean_spectral_recall"] for c in effective),
        "conditioned_mean_recall": st.mean(c["mean_spectral_recall"] for c in window),
        "conditioned_mean_auc": st.mean(c["mean_spectral_auc"] for c in window),
        "per_cell": per_cell,
    }

    print(f"attack-effective cells (mean ASR >= {ASR_MIN}): {out['n_attack_effective_cells']}")
    for c in effective:
        print(f"  rate {c['rate']:<6} cost {c['cost']:<3} ASR {c['mean_asr']:.3f}  "
              f"recall {c['mean_spectral_recall']:.4f}"
              f"{'   [evasion window]' if c['in_evasion_window'] else ''}")
    print(f"\nunconditioned mean recall over all {out['n_attack_effective_cells']}: "
          f"{out['unconditioned_mean_recall']:.4f}")
    print(f"conditioned mean recall over the {out['n_evasion_window_cells']} window cells: "
          f"{out['conditioned_mean_recall']:.4f}  (AUC {out['conditioned_mean_auc']:.4f})")

    if args.write_manifest:
        path = RESULTS / "unconditioned_effective_recall.json"
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
