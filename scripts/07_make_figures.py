"""M4 figures: the failure-boundary heatmap (headline) plus vision-vs-NIDS and H3 support figures.

Reads results/detectors.json (per-cell grid) + results/analysis.json (headline stats with CIs) and
writes vector PDFs to figures/. Plots only -- CIs/effect sizes come from analysis.json; the only
aggregation done here is a plain per-cell mean-over-seeds for the heatmap colour, a plotting step.

Run:  python scripts/07_make_figures.py
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
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

import figure_style as fs

# The house style (fonts, palette, grid, Type-42 embedding) now lives in one module so the six
# data figures cannot drift apart again. See scripts/figure_style.py for why each setting is
# there; pdf.fonttype 42 in particular is a submission gate, not a preference.
fs.apply()


def _load(name):
    return json.loads((config.RESULTS / name).read_text())


def _grid_mean(main_grid, detector_key, rates, costs):
    """Per (rate,cost) mean detector recall over seeds -> 2D array [rate, cost]. None cells -> nan."""
    buckets = defaultdict(list)
    for r in main_grid:
        val = r[detector_key]["recall"]
        if val is not None:
            buckets[(r["rate"], r["cost"])].append(val)
    grid = np.full((len(rates), len(costs)), np.nan)
    for i, rate in enumerate(rates):
        for j, cost in enumerate(costs):
            vals = buckets.get((rate, cost))
            if vals:
                grid[i, j] = float(np.mean(vals))
    return grid


def _heatmap(ax, grid, rates, costs, title):
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(costs)), costs)
    ax.set_yticks(range(len(rates)), rates)
    ax.set_xlabel("trigger cost (features)")
    ax.set_ylabel("poison rate")
    ax.set_title(title)
    for i in range(len(rates)):
        for j in range(len(costs)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=7)
    return im


# Column width of the IEEEtran two-column A4 build, measured from the compiled document
# (252.002pt / 72.27). fig1 is the only figure the manuscript includes, and it is drawn at
# exactly the width it prints at, so nothing is scaled down at \includegraphics time. The
# earlier version was drawn at 7.0in and printed at 1.68in, which put its smallest text at
# 2.275pt against the render audit's 6pt floor -- illegible on paper and a desk-reject risk.
# Every font size below is therefore a printed size, not a design size.
COL_IN = 3.487
PRINT_PT = 6.5


def fig1_failure_boundary(detectors):
    rates = detectors["config"]["rates"]
    costs = detectors["config"]["costs"]
    mg = detectors["main_grid"]
    fig, axes = plt.subplots(1, 2, figsize=(COL_IN, 1.12), sharey=True,
                             constrained_layout=True)
    for ax, (key, name) in zip(axes, [("spectral", "Spectral Signatures"),
                                      ("ac", "Activation Clustering")]):
        grid = _grid_mean(mg, key, rates, costs)
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                       vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(costs)), costs, fontsize=PRINT_PT)
        ax.set_yticks(range(len(rates)), rates, fontsize=PRINT_PT)
        ax.set_xlabel("trigger cost", fontsize=PRINT_PT, labelpad=1.5)
        ax.set_title(name, fontsize=PRINT_PT, pad=2.0)
        # sharey=True hides the non-first axis's y-tick labels by default (matplotlib's
        # automatic "label_outer"-style behavior for shared axes) -- force both panels to
        # print their own poison-rate labels, since a reader looking at the right panel
        # alone (e.g. a printed page) should not have to cross-reference the left one.
        ax.tick_params(length=1.5, width=0.4, pad=1.0, labelleft=True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
        for i in range(len(rates)):
            for j in range(len(costs)):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < 0.5 else "black",
                            fontsize=PRINT_PT)
    axes[0].set_ylabel("poison rate", fontsize=PRINT_PT, labelpad=1.5)
    cb = fig.colorbar(im, ax=axes, label="poison recall", shrink=0.95, pad=0.015)
    cb.ax.tick_params(labelsize=PRINT_PT, length=1.5, width=0.4, pad=1.0)
    cb.set_label("poison recall", fontsize=PRINT_PT, labelpad=1.5)
    cb.outline.set_linewidth(0.4)
    fig.savefig(config.FIGURES / "fig1_failure_boundary.pdf")
    plt.close(fig)


def fig2_portability(analysis):
    # Spell the detectors out. "AC" is an abbreviation this manuscript's prose never uses, so a
    # reader meeting it only on an axis has to guess.
    dets = [("Spectral\nSignatures", analysis["portability"]["spectral"]),
            ("Activation\nClustering", analysis["portability"]["ac"])]
    labels, vis, nids, vis_err, nids_err = [], [], [], [], []
    for name, p in dets:
        labels.append(name)
        vis.append(p["vision_recall_mean"]); nids.append(p["nids_recall_mean"])
        vis_err.append([p["vision_recall_mean"] - p["vision_recall_ci"][0],
                        p["vision_recall_ci"][1] - p["vision_recall_mean"]])
        nids_err.append([p["nids_recall_mean"] - p["nids_recall_ci"][0],
                         p["nids_recall_ci"][1] - p["nids_recall_mean"]])
    x = np.arange(len(labels)); w = 0.35
    # clip to >=0: percentile bootstrap CIs need not bracket the mean for small skewed arms, and
    # matplotlib rejects negative yerr. A clipped-to-0 whisker is honest (CI edge coincides with bar).
    vis_yerr = np.clip(np.array(vis_err).T, 0.0, None)
    nids_yerr = np.clip(np.array(nids_err).T, 0.0, None)
    fig, ax = plt.subplots(figsize=(fs.COL_IN, 2.0), constrained_layout=True)
    b1 = ax.bar(x - w/2, vis, w, yerr=vis_yerr, capsize=2, color=fs.BLUE,
                error_kw={"lw": 0.6}, label="vision control")
    b2 = ax.bar(x + w/2, nids, w, yerr=nids_yerr, capsize=2, color=fs.RED,
                error_kw={"lw": 0.6}, label="tabular NIDS")
    ax.set_xticks(x, labels); ax.set_ylabel("poison recall"); ax.set_ylim(0, 1.30)
    # The legend goes OUTSIDE the axes. Pinning it to "lower left" was no better than
    # matplotlib's "best": the tabular bars are near zero, so the lower left is exactly where
    # the figure's own subject sits, and the legend covered it.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncols=2, frameon=False)
    fs.label_small_bars(ax, list(b1) + list(b2), list(vis) + list(nids))
    fs.thin_spines(ax)
    fig.savefig(config.FIGURES / "fig2_portability.pdf")
    plt.close(fig)


def fig4_evasion_window(detectors):
    """Recall vs poison rate at cost=ANCHOR_COST (fixed at 16, matching the M4 analysis anchor),
    with AUC overlaid on a twin axis -- shows recall collapses at low poison rate while AUC (ranking
    quality) stays ~0.99 throughout, i.e. the miss is a fixed-budget removal artefact, not an absent
    signal. Recomputed here as a plain per-cell mean-over-seeds (a plotting aggregation), matching
    the CI-bearing anchor/evasion_window numbers already in analysis.json."""
    cost = 16
    mg = [r for r in detectors["main_grid"] if r["cost"] == cost]
    rates = sorted(set(r["rate"] for r in mg))
    recall_by_rate = defaultdict(list)
    auc_by_rate = defaultdict(list)
    asr_by_rate = defaultdict(list)
    for r in mg:
        recall_by_rate[r["rate"]].append(r["spectral"]["recall"])
        if r["spectral"].get("auc") is not None:
            auc_by_rate[r["rate"]].append(r["spectral"]["auc"])
        asr_by_rate[r["rate"]].append(r["asr"])
    recall = [float(np.mean(recall_by_rate[rt])) for rt in rates]
    auc = [float(np.mean(auc_by_rate[rt])) if auc_by_rate[rt] else np.nan for rt in rates]
    asr = [float(np.mean(asr_by_rate[rt])) for rt in rates]
    # Per-seed spread, previously discarded by the means above. The recall curve is the one the
    # caption makes a claim about, and at cost 16 its seed spread is wide enough to matter.
    recall_lo = [float(np.min(recall_by_rate[rt])) for rt in rates]
    recall_hi = [float(np.max(recall_by_rate[rt])) for rt in rates]

    fig, ax1 = plt.subplots(figsize=(fs.COL_IN, 2.1), constrained_layout=True)
    ax1.fill_between(rates, recall_lo, recall_hi, color=fs.BLUE, alpha=0.18, linewidth=0,
                     label="recall, seed range")
    ax1.plot(rates, recall, "o-", color=fs.BLUE, label="Spectral recall")
    ax1.plot(rates, asr, "s--", color=fs.GREEN, label="attack success")
    # The evasion window lives at 0.005-0.01. On a linear axis it collapses against the
    # left spine, so the figure hides the region its caption is about.
    ax1.set_xscale("log")
    ax1.set_xticks(rates)
    ax1.set_xticklabels([f"{r:g}" for r in rates])
    ax1.set_xlabel("poison rate")
    ax1.set_ylabel("recall / ASR")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(rates, auc, "^-", color=fs.PURPLE, label="Spectral AUC")
    ax2.set_ylabel("Spectral AUC")
    ax2.set_ylim(0.45, 1.02)
    ax2.grid(False)   # one grid only; twin axes drawing two sets of gridlines reads as noise
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Was loc="center right", which is exactly where the saturated recall and ASR curves sit at
    # the high-rate end. Lower centre is the one region all three curves have left empty.
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center", ncols=2)
    fs.thin_spines(ax1, drop=("top",))
    fs.thin_spines(ax2, drop=("top",))
    fig.savefig(config.FIGURES / "fig4_evasion_window.pdf")
    plt.close(fig)


def fig3_h3_confound(analysis):
    h3 = analysis["h3"]
    groups = [("Spectral", h3["spectral_imbalanced"], h3["spectral_balanced"]),
              ("AC", h3["ac_imbalanced"], h3["ac_balanced"])]
    labels = [g[0] for g in groups]
    # analysis.json stores each arm as [mean, [lo, hi]]. The intervals were previously discarded
    # here, so a figure about a confound showed no uncertainty at all.
    imb = [g[1][0] for g in groups]
    bal = [g[2][0] for g in groups]
    imb_err = np.clip(np.array([[g[1][0] - g[1][1][0] for g in groups],
                                [g[1][1][1] - g[1][0] for g in groups]]), 0.0, None)
    bal_err = np.clip(np.array([[g[2][0] - g[2][1][0] for g in groups],
                                [g[2][1][1] - g[2][0] for g in groups]]), 0.0, None)
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(fs.COL_IN, 2.0), constrained_layout=True)
    b1 = ax.bar(x - w/2, imb, w, yerr=imb_err, capsize=2, color=fs.BLUE,
                error_kw={"lw": 0.6}, label="imbalanced (~98/2)")
    b2 = ax.bar(x + w/2, bal, w, yerr=bal_err, capsize=2, color=fs.RED,
                error_kw={"lw": 0.6}, label="balanced (~50/50)")
    ax.set_xticks(x, labels); ax.set_ylabel("poison recall"); ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    # Activation Clustering's imbalanced arm is a MEASURED 0.00063 with its own interval. On a
    # 0..1 axis that bar has no visible height, so without the printed value it is
    # indistinguishable from a cell that was never run.
    fs.label_small_bars(ax, list(b1) + list(b2), list(imb) + list(bal))
    fs.thin_spines(ax)
    fig.savefig(config.FIGURES / "fig3_h3_confound.pdf")
    plt.close(fig)


def fig5_adaptive_threshold(adaptive_threshold, z_primary):
    """Fixed-budget vs adaptive-threshold recall per evasion-window cell (+ the matched-rate
    control), annotated with the adaptive threshold's false-positive rate. Reads the `per_cell`
    breakdown already computed by scripts/06_analyze_stats.py's adaptive_threshold block -- plots
    only, no statistic (recall/FPR mean) is recomputed here, matching this module's stated
    plot-only convention."""
    per_cell = adaptive_threshold["per_cell"]
    labels = [f"rate={c['rate']}\ncost={c['cost']}" for c in per_cell]
    fixed = [c["fixed_recall_mean"] for c in per_cell]
    adaptive = [c["adaptive_recall_mean"] for c in per_cell]
    fpr = [c["adaptive_fpr_mean"] for c in per_cell]

    x = np.arange(len(labels)); w = 0.35
    # Was 6.4in wide and printed at 0.96\columnwidth, a ~1.9x downscale that took its 7.5pt
    # labels to about 4pt on the page. Drawn at print width, these sizes are the printed sizes.
    fig, ax = plt.subplots(figsize=(fs.COL_IN, 2.2), constrained_layout=True)
    # The Tran et al. citation moved to the caption. A figure is not a place to cite.
    b1 = ax.bar(x - w/2, fixed, w, color=fs.BLUE, label="inherited fixed budget")
    b2 = ax.bar(x + w/2, adaptive, w, color=fs.RED,
                label=f"budget-free threshold (z={z_primary})")
    for i, f in enumerate(fpr):
        ax.text(i + w/2, min(adaptive[i] + 0.03, 1.02), f"{f:.1%}", ha="center",
                fontsize=fs.PT_SMALL - 0.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Spectral recall"); ax.set_ylim(0, 1.25)
    ax.legend(loc="upper left", ncols=1)
    fs.label_small_bars(ax, list(b1) + list(b2), list(fixed) + list(adaptive))
    fs.thin_spines(ax)
    fig.savefig(config.FIGURES / "fig5_adaptive_threshold.pdf")
    plt.close(fig)


def fig6_nc_masks():
    """Vision vs tabular Neural Cleanse reconstructions: what the inversion actually recovers.

    Left: the flagged class's reconstructed mask on the MNIST control -- the compact corner patch NC
    was designed to find. Right: the flagged class's per-feature mask on the tabular MLP at the
    anchor cell, sorted descending, with the true trigger features marked. Shows whether the tabular
    inversion concentrates on the real trigger or smears across 757 features.

    Plots ONE representative seed, chosen as the seed whose (vision_concentration,
    tabular_concentration, trigger_share) triple is closest (min Euclidean distance) to the 5-seed
    means in results/nc_masks.json's summary block -- NOT seed index 0. An earlier version plotted
    seed index 0 unconditionally; that seed (42) is an outlier where tabular concentration exceeds
    vision's, the opposite of every other seed and of the 5-seed mean the manuscript text reports,
    which reads as the figure contradicting the prose. The 5-seed mean is also printed directly in
    each panel's title so the plotted single seed can never be mistaken for the aggregate claim.
    """
    blob = json.loads((config.RESULTS / "nc_masks.json").read_text())
    summ = blob["summary"]
    mean_v = summ["vision_flagged_concentration_mean"]
    mean_t = summ["tabular_flagged_concentration_mean"]
    mean_share = summ["tabular_trigger_feature_mask_share_mean"]

    dists = [
        (blob["vision"][i]["flagged_concentration"] - mean_v) ** 2
        + (blob["tabular"][i]["flagged_concentration"] - mean_t) ** 2
        + (blob["tabular"][i]["trigger_feature_mask_share"] - mean_share) ** 2
        for i in range(len(blob["seeds"]))
    ]
    idx = int(np.argmin(dists))
    seed = blob["seeds"][idx]
    v = blob["vision"][idx]
    t = blob["tabular"][idx]

    # Was 7.0in wide, printed at 0.96\columnwidth: a ~3.5x downscale that took its type to
    # roughly 2pt on the page, the worst offender in the figure set.
    fig, (ax_v, ax_t) = plt.subplots(1, 2, figsize=(fs.COL_IN, 1.45),
                                     width_ratios=[1, 2.1], constrained_layout=True)

    vm = np.asarray(v["mask"])
    im = ax_v.imshow(vm, cmap="viridis", vmin=0, vmax=1)
    # The per-panel statistics that used to live in two multi-line set_title calls are caption
    # text, not axes furniture. Panels carry an identifier only.
    ax_v.set_title("vision (MNIST)")
    ax_v.set_xticks([]); ax_v.set_yticks([])
    ax_v.grid(False)
    cb = fig.colorbar(im, ax=ax_v, fraction=0.046)
    cb.ax.tick_params(labelsize=fs.PT_SMALL - 1, length=1.2, width=0.4, pad=0.8)
    cb.outline.set_linewidth(0.4)

    tm = np.abs(np.asarray(t["mask"]))
    order = np.argsort(tm)[::-1]
    trig = set(t["trigger_indices"])
    colours = [fs.RED if int(i) in trig else fs.GREY for i in order]
    ax_t.bar(range(len(tm)), tm[order], color=colours, width=1.0)
    ax_t.set_title("tabular NIDS")
    ax_t.set_xlabel("feature, sorted by mask weight")
    ax_t.set_ylabel("mask weight")
    ax_t.set_xlim(0, len(tm))
    ax_t.grid(False)

    handles = [Patch(facecolor=fs.RED, label="true trigger feature"),
               Patch(facecolor=fs.GREY, label="other feature")]
    # Was loc="upper right", directly on the peak of a descending-sorted bar series.
    ax_t.legend(handles=handles, loc="center right")
    fs.thin_spines(ax_t)
    fig.savefig(config.FIGURES / "fig6_nc_masks.pdf")
    plt.close(fig)


def _summary_yerr(summaries):
    """Asymmetric yerr array (2, N) from a list of summarise_cell-shaped {"mean", "ci": [lo, hi]}
    dicts, clipped to >=0 -- same convention as fig2_portability's vis_yerr/nids_yerr above (percentile
    bootstrap CIs need not bracket the mean for a small skewed n=5 arm, and matplotlib rejects negative
    yerr; a clipped-to-0 whisker is honest, the CI edge just coincides with the point)."""
    lo = np.array([s["mean"] - s["ci"][0] for s in summaries])
    hi = np.array([s["ci"][1] - s["mean"] for s in summaries])
    return np.clip(np.array([lo, hi]), 0.0, None)


def fig7_cross_dataset_boundary(extension_analysis):
    """Extension Task 9 Step 4: CTU-13 and secondary (UNSW-NB15) shown as SEPARATE panels -- never a
    pooled heatmap, since the two datasets' seeds are not interchangeable paired replicates (see
    src/analysis_extension.py's classify_replication docstring). Each dataset's panel plots ASR,
    fixed-budget recall, and (where available) MAD recall + MAD FPR at the locked z=3.0 cutoff against
    poison rate, one line per trigger cost -- the same "recall collapses, ASR/AUC stay high" framing
    as fig4_evasion_window, extended across both datasets and both removal rules.

    Every series carries its bootstrap 95% CI as a vertical whisker (via `_summary_yerr`, fig2's own
    convention) -- this is the paper's headline cross-dataset replication figure at n=5 seeds/cell, so
    the CIs `summarise_cell` already computed must be visible here, not just sitting unused in
    results/extension_analysis.json.

    Cells without MAD data (most of the CTU-13 grid, and (0.005, 8) on the secondary dataset -- see
    results/extension_analysis.json's per-dataset mad_coverage_note) simply have no MAD point plotted
    at that (rate, cost); this is not a gap the figure hides, it is the coverage the underlying MAD
    sweep actually ran.
    """
    datasets = [
        (config.PRIMARY_DATASET_ID, "CTU-13", extension_analysis["datasets"][config.PRIMARY_DATASET_ID]),
        (config.SECONDARY_DATASET_ID, "Secondary (UNSW-NB15)",
         extension_analysis["datasets"][config.SECONDARY_DATASET_ID]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    for ax, (_, label, blob) in zip(axes, datasets):
        cells = blob["cells"]
        by_cost = defaultdict(list)
        for c in cells:
            by_cost[c["cost"]].append(c)
        for i, cost in enumerate(sorted(by_cost)):
            ccells = sorted(by_cost[cost], key=lambda c: c["rate"])
            rates = [c["rate"] for c in ccells]
            asr_summaries = [c["asr"] for c in ccells]
            fbr_summaries = [c["fixed_budget_recall"] for c in ccells]
            mad_rate = [c["rate"] for c in ccells if "mad_recall" in c]
            mad_recall_summaries = [c["mad_recall"] for c in ccells if "mad_recall" in c]
            mad_fpr_summaries = [c["mad_fpr"] for c in ccells if "mad_fpr" in c]
            colour = f"C{i}"
            ax.errorbar(rates, [s["mean"] for s in asr_summaries], yerr=_summary_yerr(asr_summaries),
                       fmt="s--", color=colour, alpha=0.5, ms=3, capsize=2, elinewidth=0.8,
                       label=f"cost={cost} ASR" if ax is axes[0] else None)
            ax.errorbar(rates, [s["mean"] for s in fbr_summaries], yerr=_summary_yerr(fbr_summaries),
                       fmt="o-", color=colour, ms=3, capsize=2, elinewidth=0.8,
                       label=f"cost={cost} fixed-budget recall" if ax is axes[0] else None)
            if mad_recall_summaries:
                ax.errorbar(mad_rate, [s["mean"] for s in mad_recall_summaries],
                           yerr=_summary_yerr(mad_recall_summaries),
                           fmt="^:", color=colour, ms=5, capsize=2, elinewidth=0.8,
                           label=f"cost={cost} MAD (z=3.0) recall" if ax is axes[0] else None)
            if mad_fpr_summaries:
                ax.errorbar(mad_rate, [s["mean"] for s in mad_fpr_summaries],
                           yerr=_summary_yerr(mad_fpr_summaries),
                           fmt="x:", color=colour, ms=5, alpha=0.6, capsize=2, elinewidth=0.8,
                           label=f"cost={cost} MAD (z=3.0) FPR" if ax is axes[0] else None)
        ax.set_xlabel("poison rate")
        ax.set_title(label)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("ASR / recall / FPR")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=6.5,
              bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cross-dataset failure boundary at z=3.0 (separate panels, not pooled; whiskers = "
                "bootstrap 95% CI, n=5 seeds/cell)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.22, 1, 0.94))
    fig.savefig(config.FIGURES / "fig7_cross_dataset_boundary.pdf")
    plt.close(fig)


def fig8_adaptive_attacker(adaptive_attacker):
    """Extension Task 9 Step 4: baseline SHAP trigger vs the surrogate-selected adaptive trigger, on
    the held-out evaluation seeds ONLY (results/adaptive_attacker.json's evaluation.rows) -- feasibility,
    ASR, MAD recall, and clean-accuracy-drop side by side, per held-out seed. Surrogate-selection
    results (the candidate search on the surrogate seeds) are NOT plotted here as if they were held-out
    numbers; they are a different split of the same experiment and mixing them into this figure would
    misrepresent which numbers were used to pick the winner versus which numbers tested it.
    """
    rows = adaptive_attacker["evaluation"].get("rows", [])
    seeds = adaptive_attacker["evaluation"].get("seeds", [])
    conditions = ["baseline_shap_trigger", "selected_adaptive_trigger"]
    cond_labels = {"baseline_shap_trigger": "baseline SHAP trigger",
                   "selected_adaptive_trigger": "selected adaptive trigger"}
    by_seed_cond = {(r["seed"], r["condition"]): r for r in rows}

    metrics = [("asr", "ASR"), ("mad_recall", "MAD (z=3.0) recall"), ("clean_accuracy_drop", "clean-acc drop")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(9.5, 3.2))
    x = np.arange(len(seeds))
    w = 0.35
    for ax, (key, title) in zip(axes, metrics):
        for i, cond in enumerate(conditions):
            vals = [by_seed_cond[(s, cond)][key] for s in seeds]
            feasible = [by_seed_cond[(s, cond)]["feasible"] for s in seeds]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=cond_labels[cond])
            for bar, ok in zip(bars, feasible):
                if not ok:
                    bar.set_hatch("//")
        ax.set_xticks(x, [f"seed {s}\n(held-out)" for s in seeds], fontsize=7)
        ax.set_title(title, fontsize=8)
        if key != "clean_accuracy_drop":
            ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("Adaptive attacker: baseline vs selected trigger on HELD-OUT evaluation seeds only",
                fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(config.FIGURES / "fig8_adaptive_attacker.pdf")
    plt.close(fig)


def main():
    config.ensure_dirs()
    detectors = _load("detectors.json")
    analysis = _load("analysis.json")
    fig1_failure_boundary(detectors)
    fig2_portability(analysis)
    fig3_h3_confound(analysis)
    fig4_evasion_window(detectors)
    n = 4
    adaptive_threshold = analysis.get("adaptive_threshold")
    # gate on analysis.json's own populated block, not evasion_ablation.json's mere existence --
    # the two files are written by separate scripts (08 then 06) and can fall out of sync if 06
    # isn't rerun after 08; analysis.json is the single source fig5 reads, so gate on IT being ready.
    if adaptive_threshold is not None:
        fig5_adaptive_threshold(adaptive_threshold, adaptive_threshold["z_primary"])
        n = 5
    # nc_masks.json is written by a separate script (10) than the sweep this module's other figures
    # read, so gate on the file fig6 actually reads being present rather than assuming it was run.
    if (config.RESULTS / "nc_masks.json").exists():
        fig6_nc_masks()
        n += 1
    # extension_analysis.json (Task 9) and adaptive_attacker.json (Task 6) are each optional --
    # gate fig7/fig8 on their own required artifact rather than crashing the whole figure build if
    # one is absent (same convention as fig6's nc_masks.json gate above).
    if (config.RESULTS / "extension_analysis.json").exists():
        fig7_cross_dataset_boundary(_load("extension_analysis.json"))
        n += 1
    else:
        print("skipping fig7_cross_dataset_boundary: results/extension_analysis.json not found")
    if (config.RESULTS / "adaptive_attacker.json").exists():
        fig8_adaptive_attacker(_load("adaptive_attacker.json"))
        n += 1
    else:
        print("skipping fig8_adaptive_attacker: results/adaptive_attacker.json not found")
    print(f"wrote {n} figures to {config.FIGURES}")


if __name__ == "__main__":
    main()
