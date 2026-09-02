"""Shared journal-figure style for the data figures in paper_access/.

Why this exists. Each data-figure function in scripts/07 previously carried its own
`rcParams.update` and its own `figsize`, and they disagreed. Two figures were drawn at 6.4 and
7.0 inches and printed at 0.96\\columnwidth, which downscales them roughly 2x and 3.5x, so their
7pt type reached the page at about 3.5pt and 2pt. `scripts/07_make_figures.py` already documents
the correct discipline for fig1 -- draw at the width it prints at, so every font size in the
source is a PRINTED size -- and this module makes that discipline shared rather than
per-function.

`scripts/pipeline_style.py` is deliberately left alone. It is the schematic-drawing module for
the full-width pipeline figure, it carries box/arrow/chip primitives no data figure needs, and
its own docstring records that it is copied rather than imported across three independent paper
repositories. This module borrows its palette and its font settings and adds the axes, tick and
grid defaults that data figures need.

Palette is Paul Tol bright, colourblind-safe, matching pipeline_style so the schematic and the
data figures agree. Never use matplotlib's default C0/C1 prop cycle in a figure that ships: the
tab10 blue/orange pair is not safe for deuteranopia at thin line widths.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Column and text widths of the IEEEtran two-column A4 build, measured from the compiled
# document (252.002pt / 72.27 and 516.0pt / 72.27). A figure drawn at COL_IN and included at
# \columnwidth is scaled 1:1, so a 7pt label in the source is 7pt on the page.
COL_IN = 3.487
TEXT_IN = 7.14

# Printed point sizes. 6.5 was the previous floor and is genuinely hard to read on paper; 7 is
# the smallest size worth shipping and is what IEEE's own figure guidance treats as the minimum.
PT = 7.0
PT_SMALL = 6.5

# Paul Tol bright, colourblind-safe. Same hexes as pipeline_style.EDGE.
BLUE, RED, PURPLE, GREEN = "#4477AA", "#EE6677", "#AA3377", "#228833"
GREY, LIGHT_GREY = "#7f7f7f", "#BBBBBB"

RC = {
    "font.size": PT,
    "axes.titlesize": PT,
    "axes.labelsize": PT,
    "xtick.labelsize": PT_SMALL,
    "ytick.labelsize": PT_SMALL,
    "legend.fontsize": PT_SMALL,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # IEEE PDF eXpress rejects Type 3 fonts and every figure here is a matplotlib PDF that
    # lands in the submitted document, so this setting is a submission gate, not a preference.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.linewidth": 0.5,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "grid.color": LIGHT_GREY,
    "axes.axisbelow": True,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 1.8,
    "ytick.major.size": 1.8,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": LIGHT_GREY,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.4,
}


def apply():
    """Install the house style. Call once at module import in each figure script."""
    plt.rcParams.update(RC)


def thin_spines(ax, drop=("top", "right")):
    """Drop the top/right spines and thin the rest. Journal figures rarely want a full box."""
    for name in drop:
        ax.spines[name].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def label_small_bars(ax, bars, values, floor=0.02, fmt="{:.4f}", pad=0.012):
    """Print the value above any bar too short to read.

    A measured near-zero and a never-measured cell are otherwise pixel-identical. Activation
    Clustering's imbalanced recall is 0.00063 with a real confidence interval, and drawn on a
    0..1 axis it is invisible, so without this a reader cannot tell it from a missing bar. Only
    bars below `floor` are annotated, so the well-separated ones stay uncluttered.
    """
    for bar, v in zip(bars, values):
        if v is None:
            continue
        if v < floor:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad,
                    fmt.format(v), ha="center", va="bottom", fontsize=PT_SMALL - 0.5)


def mark_not_measured(ax, x, label="not measured", y=0.03):
    """Explicit marker for a cell that was never run, so it cannot read as a measured zero."""
    ax.text(x, y, label, ha="center", va="bottom", fontsize=PT_SMALL - 0.5,
            color=GREY, rotation=90, style="italic")
