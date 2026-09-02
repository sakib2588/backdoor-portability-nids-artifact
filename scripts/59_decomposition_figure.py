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
    for ax, data, title in ((ax1, ctu, "CTU-13"), (ax2, unsw, "UNSW-NB15")):
        x = list(range(len(all_costs)))
        present = [c for c in all_costs if c in data]
        evasion = [max(data[c][0], 0.0) if c in data else 0.0 for c in all_costs]
        backdoor = [max(data[c][1], 0.0) if c in data else 0.0 for c in all_costs]
        bars = ax.bar(x, evasion, color=E_COLOR, label="evasion", width=0.6)
        ax.bar(x, backdoor, bottom=evasion, color=B_COLOR, label="backdoor", width=0.6)
        # A measured component near zero (CTU at costs 1/2/4 is 0.000 to 0.001) and a cost that
        # was never run both draw nothing. Label the first, mark the second.
        fs.label_small_bars(ax, [b for b, c in zip(bars, all_costs) if c in data],
                            [v for v, c in zip(evasion, all_costs) if c in data],
                            floor=0.02, fmt="{:.3f}")
        for i, c in enumerate(all_costs):
            if c not in data:
                fs.mark_not_measured(ax, i)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in all_costs])
        ax.set_title(title)
        ax.set_xlabel("trigger cost")
        ax.set_ylim(0, 1.15)
        fs.thin_spines(ax)
        _ = present

    ax1.set_ylabel("attack success component")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.14))

    out_path = config.ROOT / "paper" / "figures" / "fig_decomposition.pdf"
    fig.savefig(out_path)
    print(f"wrote {out_path}")
    print("CTU-13 costs present:", sorted(ctu.keys()))
    print("UNSW-NB15 costs present:", sorted(unsw.keys()),
          "(costs 1/2/4 were never measured for this dataset -- not fabricated here)")


if __name__ == "__main__":
    main()
