"""Replacement headline figure: the budget-starvation window, showing recall AND the ranking
signal on the same axes.

The prior figure plotted poison recall alone, which is half of the window's definition -- the
window is recall below 0.5 while AUC stays high at rates where ASR is attacker-effective, and a
recall-only plot cannot show that the signal survives. Panel (a) puts Spectral Signatures' recall
and its ranking AUC on one poison-rate axis at the attacker-effective trigger cost, with the
window shaded. Panel (b) keeps the full-grid recall evidence.

Reads results/detectors.json only; the sole aggregation is a per-cell mean (and min/max spread)
over seeds, a plotting step. Writes figures/fig_window.pdf.

Run:  .venv/bin/python scripts/51_window_figure.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

# pdf.fonttype 42 embeds TrueType instead of matplotlib's default Type 3, which IEEE PDF eXpress
# rejects. Matches scripts/07_make_figures.py so both figures carry the same font class.
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

ANCHOR_COST = 16
ASR_EFFECTIVE = 0.8
BUDGET_CRITERION = 0.5


def _bucket(main_grid, cost=None):
    """(rate, cost) -> {metric: [per-seed values]}, skipping detectors that expose no value."""
    out = defaultdict(lambda: defaultdict(list))
    for row in main_grid:
        if cost is not None and row["cost"] != cost:
            continue
        key = (row["rate"], row["cost"])
        out[key]["asr"].append(row["asr"])
        out[key]["spectral_recall"].append(row["spectral"]["recall"])
        if row["spectral"]["auc"] is not None:
            out[key]["spectral_auc"].append(row["spectral"]["auc"])
        if row["ac"]["recall"] is not None:
            out[key]["ac_recall"].append(row["ac"]["recall"])
    return out


def main() -> int:
    grid = json.loads((config.RESULTS / "detectors.json").read_text())["main_grid"]

    rates = sorted({r["rate"] for r in grid})
    costs = sorted({r["cost"] for r in grid})
    slice_ = _bucket(grid, cost=ANCHOR_COST)

    x = np.arange(len(rates))
    spec_recall = np.array([np.mean(slice_[(r, ANCHOR_COST)]["spectral_recall"]) for r in rates])
    spec_lo = np.array([np.min(slice_[(r, ANCHOR_COST)]["spectral_recall"]) for r in rates])
    spec_hi = np.array([np.max(slice_[(r, ANCHOR_COST)]["spectral_recall"]) for r in rates])
    spec_auc = np.array([np.mean(slice_[(r, ANCHOR_COST)]["spectral_auc"]) for r in rates])
    ac_recall = np.array([np.mean(slice_[(r, ANCHOR_COST)]["ac_recall"]) for r in rates])
    asr = np.array([np.mean(slice_[(r, ANCHOR_COST)]["asr"]) for r in rates])

    window = (asr >= ASR_EFFECTIVE) & (spec_recall < BUDGET_CRITERION)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.6, 1.15), width_ratios=[1.3, 1.0])

    # -- panel (a): the window itself -------------------------------------------------
    if window.any():
        idx = np.flatnonzero(window)
        ax.axvspan(idx.min() - 0.45, idx.max() + 0.45, color="#d9534f", alpha=0.12, lw=0)
        ax.text(
            idx.mean(),
            1.10,
            "window",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#a03b38",
        )

    # Only Spectral Signatures is plotted here. It is the one detector carrying both a ranking
    # score and a removal budget, so it is the only one on which the window's defining
    # contrast -- signal intact, budget starved -- can be drawn at all. Activation Clustering
    # has no ranked score, and putting its near-coincident recall curve alongside would invite
    # exactly the conflation of the two detectors' results the paper is careful to avoid.
    ax.fill_between(x, spec_lo, spec_hi, color="#1f77b4", alpha=0.18, lw=0)
    ax.plot(x, spec_auc, "s--", color="#2ca02c", ms=2.6, lw=1.0, label="ranking AUC")
    ax.plot(x, spec_recall, "o-", color="#1f77b4", ms=2.6, lw=1.0, label="poison recall")
    ax.axhline(BUDGET_CRITERION, color="0.45", lw=0.7, ls=(0, (4, 3)))

    # Both curves sit at ~1.0 from rate 0.02 rightwards, so the lower-right quadrant is empty
    # and is the only place a legend does not cross a line.
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=6,
        handlelength=1.3,
        labelspacing=0.2,
        borderaxespad=0.1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:g}".lstrip("0") for r in rates])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylim(-0.06, 1.20)
    ax.set_xlim(-0.5, len(rates) - 0.5)
    ax.set_xlabel("poison rate")
    ax.set_ylabel("recall / AUC")
    ax.set_title(f"(a) cost {ANCHOR_COST}, ASR 1.0 at every rate", pad=7)

    # -- panel (b): full-grid Spectral recall -----------------------------------------
    full = _bucket(grid)
    heat = np.full((len(rates), len(costs)), np.nan)
    for i, r in enumerate(rates):
        for j, c in enumerate(costs):
            vals = full[(r, c)]["spectral_recall"]
            if vals:
                heat[i, j] = float(np.mean(vals))

    im = bx.imshow(heat, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    bx.set_xticks(range(len(costs)), [str(c) for c in costs])
    bx.set_yticks(range(len(rates)), [f"{r:g}" for r in rates])
    bx.set_xlabel("trigger cost")
    bx.set_ylabel("poison rate")
    bx.set_title("(b) Spectral recall")
    cb = fig.colorbar(im, ax=bx, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6)

    fig.tight_layout(pad=0.3, w_pad=1.1)
    out = config.FIGURES / "fig_window.pdf"
    fig.savefig(out)
    plt.close(fig)

    print(f"wrote {out}")
    print("rate | mean ASR | Spectral recall [min,max] | Spectral AUC | AC recall | in window")
    for i, r in enumerate(rates):
        print(
            f"{r:6g} | {asr[i]:.4f} | {spec_recall[i]:.4f} "
            f"[{spec_lo[i]:.4f},{spec_hi[i]:.4f}] | {spec_auc[i]:.4f} | "
            f"{ac_recall[i]:.4f} | {bool(window[i])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
