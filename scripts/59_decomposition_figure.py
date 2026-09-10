"""Decomposition figure across the full cost grid (PC review 2026-08-28, M8): Table I prints only
2 of 5 costs at the reported rate (0.005). This figure shows the evasion/backdoor split across all
5 costs on CTU-13, where the full grid exists (results/clean_model_stamped_asr.json). UNSW-NB15's
decomposition (results/secondary_clean_model_stamped_asr.json) was only ever measured at costs 8
and 16 -- no run exists for costs 1/2/4 there, so this figure shows exactly that, honestly, rather
than fabricating the missing bars. Extending UNSW's decomposition to the full grid is future work,
not done in this pass.

Reads results/clean_model_stamped_asr.json and results/secondary_clean_model_stamped_asr.json
only; the sole computation is reading the already-committed evasion/backdoor components per cost,
a plotting step. Writes figures/fig_decomposition.pdf.

Run:  .venv/bin/python scripts/59_decomposition_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
import figure_style as fs

fs.apply()

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

E_COLOR = "#1f77b4"
B_COLOR = "#ff7f0e"


def load_grid(path, costs):
    d = json.loads(Path(path).read_text())
    summary = d["summary"]
    out = {}
    for c in costs:
        key = str(c)
        if key not in summary:
            continue
        s = summary[key]
        out[c] = (s["evasion_component_E"], s["backdoor_component_B"])
    return out


def load_ctu_cost8_per_seed():
    """CTU-13's cost-8 decomposition, one entry per seed.

    The cost-8 MEAN is not plotted, because four of its five seeds sit at the clean-arm
    ceiling and the mean over four constants and one real value describes no trained model
    (sections/05_results.tex). results/cost_transition_sweep.json carries the already
    decomposed per-seed arms, so this is a read rather than a recomputation.
    """
    d = json.loads((config.RESULTS / "cost_transition_sweep.json").read_text())
    rows = d["rows"]
    out = []
    for seed in (42, 123, 456, 789, 1337):
        r = rows[f"{seed}|8"]
        out.append((seed, r["evasion_E"], r["backdoor_B"], bool(r["at_ceiling"])))
    return out


def main():
    ctu = load_grid(config.RESULTS / "clean_model_stamped_asr.json", [1, 2, 4, 8, 16])
    unsw = load_grid(config.RESULTS / "secondary_clean_model_stamped_asr.json", [1, 2, 4, 8, 16])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fs.COL_IN, 1.5), sharey=True,
                                   constrained_layout=True)

    # Both panels get the SAME cost axis. Previously each panel drew only the costs its own file
    # carried, so CTU showed five bars against UNSW's two on a different scale, and a reader
    # comparing them was comparing different x-axes. Costs UNSW never measured are now present
    # as labelled gaps rather than absent, which is the honest way to show a hole.
    all_costs = [1, 2, 4, 8, 16]
    ctu8 = load_ctu_cost8_per_seed()

    # Components are plotted SIGNED. An earlier version wrapped both in max(.., 0.0), which
    # silently hid CTU-13's per-seed backdoor components of -0.400 and -0.017 and UNSW-NB15's
    # evasion component of -0.000145. A component that measured below zero now draws below zero.
    for ax, data, title in ((ax1, ctu, "CTU-13"), (ax2, unsw, "UNSW-NB15")):
        x = list(range(len(all_costs)))
        per_seed_here = title == "CTU-13"
        # Cost 8 on CTU-13 is drawn per seed, so it is excluded from the aggregate bars.
        agg = [c for c in all_costs if c in data and not (per_seed_here and c == 8)]
        idx = [all_costs.index(c) for c in agg]
        evasion = [data[c][0] for c in agg]
        backdoor = [data[c][1] for c in agg]
        # Round-v8 review: these were stacked, backdoor on top of evasion. A stacked segment
        # cannot show sign -- a backdoor of -0.400 drew downward from an evasion of 0.948 and
        # was indistinguishable from a positive segment of the same length, in a figure whose
        # caption says "signed". Both components are now drawn from zero, side by side, so a
        # negative component falls below the axis line and reads as negative.
        bw = 0.30
        bars = ax.bar([i - bw / 2 for i in idx], evasion, color=E_COLOR,
                      label="evasion", width=bw)
        ax.bar([i + bw / 2 for i in idx], backdoor, color=B_COLOR,
               label="backdoor", width=bw)
        # UNSW-NB15's evasion components are -0.000145 and -0.000397 and CTU-13's low-cost
        # ones are 0.000 to 0.001. Printed beside each other at this width they collided, and
        # at 3 decimals the negatives read as "-0.000", a formatting error rather than a small
        # value. The magnitudes are given in the body; the axis only has to show that the bar
        # is a measured near-zero and not a missing one, which a tick mark does.
        bbars = [r for r in ax.containers[-1]]
        for series, col in ((list(zip(list(bars), evasion)), E_COLOR),
                               (list(zip(bbars, backdoor)), B_COLOR)):
            for bar, v in series:
                if v is not None and abs(v) < 0.02:
                    ax.plot(bar.get_x() + bar.get_width() / 2, 0.0, marker="_", ms=5,
                            color=col, mew=1.1, zorder=6)
        missing = [i for i, c in enumerate(all_costs) if c not in data]
        if missing:
            # Three separate horizontal markers overlapped each other. One centered label spans
            # the contiguous run of never-measured costs instead.
            fs.mark_not_measured(ax, sum(missing) / len(missing))

        if per_seed_here:
            # Cost 8 is the one cell whose backdoor component is bimodal across seeds, so the
            # 5 seeds are drawn individually. Five stacked pairs at this width were unreadable
            # at print size; the aggregate bar carries the evasion side and the per-seed
            # backdoor values are points, which is what the spread claim is about.
            slot = all_costs.index(8)
            e8 = sum(e for _, e, _, _ in ctu8) / len(ctu8)
            ax.bar(slot - bw / 2, e8, width=bw, color=E_COLOR)
            jitter = [(k - 2) * 0.055 for k in range(5)]
            for j, (seed, e, b, ceil) in zip(jitter, ctu8):
                # `ceil` (the clean-arm ceiling flag) previously selected a hatch pattern here;
                # the stacked-bar hatching was removed when the axis moved to grouped bars from
                # zero (round-v8 review), and no replacement encoding was added. Marked here so
                # the unused field is not silently dropped again without a decision either way.
                ax.plot(slot + bw / 2 + j, b, marker="o", ms=2.8, mfc=B_COLOR,
                        mec="white", mew=0.35, ls="none", zorder=5)
            ax.axhline(0.0, color="0.3", linewidth=0.5, zorder=1)

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in all_costs])
        ax.set_title(title)
        ax.set_xlabel("trigger cost")
        # The axis floor was -0.1, above CTU-13's most negative per-seed backdoor component of
        # -0.400, so the sign the caption promises had nowhere to be drawn.
        ax.set_ylim(-0.52, 1.15)
        ax.set_yticks([-0.5, 0.0, 0.5, 1.0])
        fs.thin_spines(ax)

    ax1.set_ylabel("attack success component")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.14))

    # Writes to paper_access/, the live manuscript. It used to write to paper/, whose caption
    # still describes the pre-2026-09-04 mean-based cost-8 bar; regenerating into that
    # directory left the older draft's caption contradicting its own figure.
    out_path = config.ROOT / "paper_access" / "figures" / "fig_decomposition.pdf"
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    print("CTU-13 costs present:", sorted(ctu.keys()))
    print("UNSW-NB15 costs present:", sorted(unsw.keys()),
          "(costs 1/2/4 were never measured for this dataset -- not fabricated here)")


if __name__ == "__main__":
    main()
