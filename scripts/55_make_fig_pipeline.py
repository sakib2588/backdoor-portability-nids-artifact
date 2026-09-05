#!/usr/bin/env python3
"""Figure 1: the study pipeline, drawn at full text width.

The previous version was six grey boxes carrying prose, which made it a
paragraph with a border rather than a diagram. This one draws the sweep: one
chip per fitted run, laid out as the grid actually ran, configurations across
and seeds down, so a reader sees that CTU-13 covers 25 rate-by-cost cells and
UNSW-NB15 covers 8, instead of reading two totals. The budget-starvation window
that used to be its own figure is absorbed as stage 5, so the page cost of the
wider figure is paid once rather than twice.

Every count is read from the committed artefacts. Nothing here is typed by hand:
  results/detectors.json                   primary grid, CTU-13 Neris
  results/secondary_detectors.json         secondary grid, UNSW-NB15
  results/secondary_threshold_margin.json  margin geometry, 10 seeds

Run with the repository virtualenv, which is where matplotlib lives:
  .venv/bin/python scripts/55_make_fig_pipeline.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pipeline_style as ps

OUT = ROOT / "paper/figures/fig_pipeline.pdf"
CTU, UNSW = "#4477AA", "#EE6677"

# Chip geometry sized so the widest grid, CTU-13's 25 configurations, fits
# inside a 44-unit panel. Both panels use the same chip so the two sweeps stay
# visually comparable; the smaller grid is meant to look smaller.
CW, CH, GX, GY = 1.2, 2.0, 0.36, 0.5


def counts():
    pri = json.load(open(ROOT / "results/detectors.json"))
    sec = json.load(open(ROOT / "results/secondary_detectors.json"))
    mar = json.load(open(ROOT / "results/secondary_threshold_margin.json"))
    ana = json.load(open(ROOT / "results/analysis.json"))
    # The removal rule's z grid is the PRE-REGISTERED one the manuscript states in Methods,
    # {2.0, 2.5, 3.0}. It is not the margin diagnostic's grid, which deliberately probes beyond the
    # pre-registered range to 3.5 and 4.0 and belongs to the margin panel next door. Reading the
    # margin grid here printed "z 2.0-4.0" on the MAD z-threshold box and contradicted the paper's
    # own pre-registration claim.
    z_pre = ana["adaptive_threshold"]["z_thresholds_tested"]
    return {
        "pri_rates": len(pri["config"]["rates"]),
        "pri_costs": len(pri["config"]["costs"]),
        "pri_seeds": len(pri["config"]["seeds"]),
        "pri_cells": len(pri["main_grid"]),
        "pri_nc": len(pri["nc_boundary"]),
        "sec_rates": len(sec["config"]["rates"]),
        "sec_costs": len(sec["config"]["costs"]),
        "sec_seeds": len(sec["config"]["seeds"]),
        "sec_cells": len(sec["main_grid"]),
        "sec_nc": len(sec["nc_boundary"]),
        "mar_rows": len(mar["rows"]),
        "mar_seeds": len(mar["config"]["seeds"]),
        "mar_cells": len(mar["config"]["cells"]),
        "z_lo": min(z_pre),
        "z_hi": max(z_pre),
        "mar_z_lo": min(mar["z_grid"]),
        "mar_z_hi": max(mar["z_grid"]),
        "det_ok": pri["summary"]["n_determinism_failed"] == 0
        and sec["summary"]["n_determinism_failed"] == 0,
    }


def centred(box_x, box_w, cols):
    """Left edge that centres a `cols`-wide chip grid inside a panel."""
    return box_x + (box_w - (cols * (CW + GX) - GX)) / 2


def sweep_panel(ax, x, w, y, h, title, subtitle, n_cfg, n_seeds, colour, footnote):
    """One poison sweep: a titled panel of configuration-by-seed chips.

    Title and grid specification go on separate lines. Run together they
    overflow the panel and collide with the neighbouring one.
    """
    ps.box(ax, x, y, w, h, "", None, "#FCFCFC", "#666666")
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="#111111")
    ax.text(x + w / 2, y + h - 7.0, subtitle, ha="center", va="center",
            fontsize=7.0, color="#444444")
    ps.chips(ax, centred(x, w, n_cfg), y + h - 12.0,
             [colour] * (n_cfg * n_seeds), cols=n_cfg,
             cw=CW, ch=CH, gx=GX, gy=GY)
    ax.text(x + w / 2, y + 3.0, footnote, ha="center", va="center",
            fontsize=7.0, color="#444444")


def main():
    c = counts()
    fig, ax = ps.canvas(width=10.4, height=3.70)

    # ------------------------------------------------------------- band 1
    # One unchanged recipe, applied to both corpora.
    ys, hs = 84, 12
    ps.box(ax, 4, ys, 20, hs, "Corpora", "CTU-13 Neris  |  UNSW-NB15")
    ps.box(ax, 28, ys, 20, hs, "Constraints", "TabularBench feasible set")
    ps.box(ax, 52, ys, 20, hs, "Trigger", "SHAP-ranked, clean-label")
    ps.box(ax, 76, ys, 20, hs, "Projection", "6-sigma watermark, projected")
    for x in (24, 48, 72):
        ps.arrow(ax, x, ys + hs / 2, x + 4, ys + hs / 2)
    ps.stage(ax, 5.4, ys + hs - 0.8, "1")
    ps.stage(ax, 53.4, ys + hs - 0.8, "2")

    # The one recipe splits here, one branch per corpus.
    ax.plot([86, 86], [ys, 80.5], color="#777777", lw=1.0, zorder=1)
    ax.plot([26, 86], [80.5, 80.5], color="#777777", lw=1.0, zorder=1)

    # ------------------------------------------------------------- band 2
    ym, hm = 48, 30
    sweep_panel(
        ax, 4, 44, ym, hm,
        f"Poison sweep, CTU-13 Neris   {c['pri_cells']} runs",
        f"{c['pri_rates']} rates x {c['pri_costs']} costs x {c['pri_seeds']} seeds",
        c["pri_rates"] * c["pri_costs"], c["pri_seeds"], CTU,
        f"Neural Cleanse on a boundary subset, {c['pri_nc']} runs")
    sweep_panel(
        ax, 52, 44, ym, hm,
        f"Poison sweep, UNSW-NB15   {c['sec_cells']} runs",
        f"{c['sec_rates']} rates x {c['sec_costs']} costs x {c['sec_seeds']} seeds",
        c["sec_rates"] * c["sec_costs"], c["sec_seeds"], UNSW,
        f"Neural Cleanse on a boundary subset, {c['sec_nc']} runs")
    ps.arrow(ax, 26, 80.5, 26, ym + hm)
    ps.arrow(ax, 74, 80.5, 74, ym + hm)
    ps.stage(ax, 5.4, ym + hm - 0.8, "3")

    # ------------------------------------------------------------- band 3
    # The vision control runs first, then the five vision-built detectors on
    # the identical poisoned models. Six boxes of width 13.5 with 2.2 gaps fill
    # x=4..96 exactly. The two long names carry an explicit line break because
    # ps.box centres a single line and does not wrap.
    yd, hd = 26, 16
    dw, dstep = 13.5, 15.7
    dx = [4 + i * dstep for i in range(6)]
    ps.box(ax, dx[0], yd, dw, hd, "Vision control", "MNIST BadNets",
           ps.FILL["c"], ps.EDGE["c"], tfs=8.0, sfs=7.0)
    ps.box(ax, dx[1], yd, dw, hd, "Spectral\nSignatures", "ranked score",
           ps.FILL["a"], ps.EDGE["a"], tfs=8.0, sfs=7.0)
    ps.box(ax, dx[2], yd, dw, hd, "SPECTRE", "ranked score",
           ps.FILL["a"], ps.EDGE["a"], tfs=8.0, sfs=7.0)
    ps.box(ax, dx[3], yd, dw, hd, "STRIP", "entropy score",
           ps.FILL["a"], ps.EDGE["a"], tfs=8.0, sfs=7.0)
    ps.box(ax, dx[4], yd, dw, hd, "Activation\nClustering", "no ranked score",
           ps.FILL["b"], ps.EDGE["b"], tfs=8.0, sfs=7.0)
    ps.box(ax, dx[5], yd, dw, hd, "Neural\nCleanse", "model-level",
           ps.FILL["b"], ps.EDGE["b"], tfs=8.0, sfs=7.0)
    # Both sweeps feed all five detectors, so they join a bus rather than
    # crossing ten arrows over each other.
    centres = [x + dw / 2 for x in dx]
    bus = yd + hd + 2.8
    for x in (26, 74):
        ax.plot([x, x], [ym, bus], color="#777777", lw=1.0, zorder=1)
    ax.plot([26, centres[-1]], [bus, bus], color="#777777", lw=1.0, zorder=1)
    for x in centres[1:]:
        ps.arrow(ax, x, bus, x, yd + hd)
    ps.arrow(ax, dx[0] + dw, yd + hd / 2, dx[1], yd + hd / 2)
    ps.stage(ax, 5.4, yd + hd - 0.8, "4")

    # ------------------------------------------------------------- band 4
    # What each score is turned into: a decomposition, two removal rules, and
    # the margin that separates recovery from collapse.
    yr, hr = 3, 17
    ps.box(ax, 4, yr, 22, hr, "ASR decomposition", "evasion vs backdoor",
           ps.FILL["d"], ps.EDGE["d"], tfs=8.0, sfs=7.0)
    ps.box(ax, 28, yr, 22, hr, "Fixed vision budget",
           "true poison fraction granted",
           ps.FILL["d"], ps.EDGE["d"], tfs=8.0, sfs=7.0)
    ps.box(ax, 52, yr, 22, hr, "MAD z-threshold",
           f"z {c['z_lo']:.1f}-{c['z_hi']:.1f}, scale 0.6745",
           ps.FILL["d"], ps.EDGE["d"], tfs=8.0, sfs=7.0)
    ps.box(ax, 76, yr, 20, hr, "Margin geometry",
           f"{c['mar_rows']} runs, {c['mar_seeds']} seeds",
           ps.FILL["c"], ps.EDGE["c"], tfs=8.0, sfs=7.0)
    ps.arrow(ax, 15, yd, 15, yr + hr)
    ps.arrow(ax, 39, yd, 39, yr + hr)
    ps.arrow(ax, 63, yd, 63, yr + hr)
    ps.arrow(ax, 86, yd, 86, yr + hr)
    ps.stage(ax, 5.4, yr + hr - 0.8, "5")

    fig.savefig(OUT)
    print("wrote", OUT)
    print(f"CTU-13 {c['pri_cells']} runs ({c['pri_rates']}x{c['pri_costs']} cfg "
          f"x {c['pri_seeds']} seeds), UNSW-NB15 {c['sec_cells']} runs "
          f"({c['sec_rates']}x{c['sec_costs']} cfg x {c['sec_seeds']} seeds), "
          f"margin {c['mar_rows']} rows, determinism_ok={c['det_ok']}")


if __name__ == "__main__":
    main()
