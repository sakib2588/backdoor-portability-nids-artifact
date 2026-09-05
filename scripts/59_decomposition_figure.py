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
        bars = ax.bar(idx, evasion, color=E_COLOR, label="evasion", width=0.6)
        ax.bar(idx, backdoor, bottom=evasion, color=B_COLOR, label="backdoor", width=0.6)
        fs.label_small_bars(ax, list(bars), evasion, floor=0.02, fmt="{:.3f}")
        for i, c in enumerate(all_costs):
            if c not in data:
                fs.mark_not_measured(ax, i)

        if per_seed_here:
            slot = all_costs.index(8)
            w = 0.15
            offs = [(k - 2) * w for k in range(5)]
            for off, (seed, e, b, ceil) in zip(offs, ctu8):
                ax.bar(slot + off, e, width=w * 0.9, color=E_COLOR,
                       hatch="///" if ceil else None, edgecolor="white", linewidth=0.3)
                ax.bar(slot + off, b, bottom=e, width=w * 0.9, color=B_COLOR,
                       hatch="///" if ceil else None, edgecolor="white", linewidth=0.3)
            ax.axhline(0.0, color="0.3", linewidth=0.5, zorder=1)

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in all_costs])
        ax.set_title(title)
        ax.set_xlabel("trigger cost")
        ax.set_ylim(-0.1, 1.15)
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
