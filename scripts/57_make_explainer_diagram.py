"""Explainer diagram: a project-history timeline for the backdoor-portability paper.

Renders a vector PDF into notes/figures/ for embedding in
notes/PROJECT_EXPLAINER.md. Documentation only -- the manuscript reads none of
this; it uses paper/figures/fig_pipeline.pdf and fig4_evasion_window.pdf directly.

    python scripts/57_make_explainer_diagram.py

The timeline auto-splits at phase boundaries so no panel exceeds one A4 text
block, and balances the split rather than filling greedily. Titles wrap at a
conservative width so a long title never gets clipped by the card edge.
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "notes" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

INK = "#1F1D1A"
MUTED = "#54504A"
SPINE = "#B4AEA1"

FIG_W = 7.30
MAX_H = 10.10  # A4 text height at 2.1 cm margins, less a caption

PHASES = {
    "setup":   dict(band="#F1EFE9", accent="#847E70", label="#6A6558"),
    "build":   dict(band="#E5F0E9", accent="#2E7D4F", label="#2E7D4F"),
    "find":    dict(band="#E3EDF7", accent="#2A5F8F", label="#2A5F8F"),
    "repair":  dict(band="#ECE7F6", accent="#5E35B1", label="#5E35B1"),
    "extend":  dict(band="#FBF1DD", accent="#C08018", label="#9C6A12"),
    "review":  dict(band="#FBE4E4", accent="#B33A3A", label="#8E2626"),
    "now":     dict(band="#E0F2F0", accent="#00796B", label="#00796B"),
}

STEPS = [
    ("setup", "THE QUESTION",
     "Does a high attack-success rate mean the model learned a backdoor?",
     "A poisoned intrusion detector that misclassifies triggered flows might have learned a"
     " backdoor, or the triggered flows might already fool the never-poisoned model. One"
     " reported attack-success rate cannot tell those apart, and nobody in this space measures"
     " which one a result is."),
    ("setup", None,
     "Pick a trigger that could actually be sent",
     "A NetFlow feature cannot be perturbed like a pixel. Port a clean-label, SHAP-guided"
     " trigger from malware classifiers, then project every poisoned flow onto a constraint-valid"
     " feasible set, so it stays a record a real sensor could produce."),

    ("build", "BUILD THE PIPELINE",
     "Two corpora, two victims, five seeds, 125 runs",
     "CTU-13 Neris (198,128 flows, 757 features) and UNSW-NB15 (1.58M rows, 38 features)."
     " An MLP victim whose activations the detectors can inspect, and a LightGBM victim that"
     " exposes none, as a built-in test of detector applicability by architecture."),
    ("build", None,
     "Validate every detector twice before trusting a single tabular number",
     "All five clear a matched vision control (BadNets on MNIST) at recall 0.98 to 1.0, and a"
     " loud, constraint-violating tabular trigger as well. Both gates are blocking, because a"
     " null from a detector that never demonstrated it can find anything is uninterpretable."),

    ("find", "THE REALIZATION SPLIT",
     "One recipe, two opposite phenomena",
     "At trigger cost 16, CTU-13's clean, never-poisoned model already calls every watermarked"
     " flow benign -- evasion component 0.9577, and a backdoor component bounded by construction"
     " rather than measured. UNSW-NB15's clean model stays at 0.0001 while the poisoned model"
     " reaches 0.9626, measured at both costs."),
    ("find", None,
     "The detector's signal survives; its removal rule does not",
     "At attacker-realistic poison rates, Spectral Signatures' inherited budget recovers only"
     " 0.1425 of the planted rows while its ranking AUC stays at 0.9825. At the anchor cell the"
     " budget is 858 rows and the median poison row sits at rank 1,308, so the budget is spent"
     " on false positives before it arrives."),

    ("repair", "REPAIR IT, AND PRICE THE REPAIR",
     "A budget-free threshold recovers full recall, and the same budget fails on images",
     "Replacing the fixed budget with a median-absolute-deviation threshold reaches recall 1.0"
     " at zero variance across the window, at 6.58% false positives. Re-run at matched poison"
     " rates, the inherited budget collapses on MNIST too, 0.9809 down to 0.1600 -- so starvation"
     " is not a property of tabular data."),
    ("repair", None,
     "Five detectors, five different fates",
     "SPECTRE is rescued at 3.6 times the false-positive cost. Activation Clustering has no"
     " ranked score until we construct one. Neural Cleanse resists repair across four families of"
     " decision statistic. STRIP's zero is a bound, not a shape: a label-free check predicts it"
     " and agrees with observed firing on 75 of 75 units."),

    ("extend", "TEST THE FRAMING, AND RETRACT IT",
     "A four-corpus family kills the dataset-dependent story",
     "Both outcomes occur inside every corpus able to fund the attack, at mean backdoor fractions"
     " of +0.4465, +0.5964 and +0.5478 across 240 cells of one shared NetFlow schema. Realization"
     " tracks the trigger, not the corpus. The title changed on this evidence, and the"
     " pre-registered benign-share explanation came back inconclusive."),
    ("extend", None,
     "The window recurs on a second corpus; the repair does not recur reliably",
     "UNSW-NB15 reproduces the window in three cells at AUC 0.9287 against fixed recall 0.0292."
     " But the recovery there is bimodal, not noisy: across 90 merged runs it recovers above 0.9"
     " in 62 and below 0.5 in 23, roughly one failure in four, with nothing signalling which run"
     " it is about to be."),
    ("extend", None,
     "Removing the poison is not the same as removing the backdoor",
     "Retraining after the repaired rule strips its flagged rows takes UNSW-NB15's attack success"
     " from 0.84, 0.96 and 0.99 down to 0.05, 0.43 and 0.62. Poison recall does not rank the three"
     " cells in the same order as the protection it buys. A restored detector is a partial"
     " defense, not a clean one."),

    ("review", "TWO HOSTILE REVIEW ROUNDS",
     "Reject, then Major revision from all five reviewers",
     "The 2026-09-04 round returned Reject with twelve confirmed defects, every one resolvable"
     " from data already on disk. The 2026-09-05 round put five quarantined reviewers on the"
     " rendered PDF alone, barred from asserting any number is wrong, and returned 61 raw findings"
     " and four blockers, all closed the same day."),
    ("review", None,
     "What the reviews actually cost was framing, not experiments",
     "No finding required a new experimental direction. What did need real work was claiming more"
     " than the evidence carried: a dataset-dependent conclusion the corpus family refuted, and a"
     " rule-selection statistic that turned out to be reading detector identity. Both retracted"
     " rather than defended."),

    ("now", "WHERE IT STANDS",
     "23 pages, five reported detectors, every number traced to a committed file",
     "Abstract at 249 words against a 250 cap. Seven supporting sections moved into the artifact"
     " repository with a 54-row provenance index that refuses to publish if a row names a file it"
     " does not contain. All 125 detector-sweep runs re-score to their committed values."),
    ("now", None,
     "Four things still open, and named as open",
     "Which property of the trigger decides realization. A deployable rule for the constructed"
     " Activation Clustering score. An adaptive attacker aimed at the repaired detectors. And why"
     " Neural Cleanse fails, which four independent families of decision statistic did not"
     " resolve."),
]

SPINE_X = 0.42
CARD_L, CARD_R = 0.92, 7.14
PAD_X = 0.23
WRAP = 92
TITLE_WRAP = 58

H_TOP = 0.105
H_PHASE = 0.225
H_TITLE = 0.225
H_TGAP = 0.050
H_BODY = 0.138
H_BOT = 0.140
BAND_PAD = 0.105
BAND_GAP = 0.145
HEADER = 0.88


def build_bands():
    bands, band = [], []
    for phase, label, title, body in STEPS:
        lines = textwrap.wrap(body, WRAP)
        title_lines = textwrap.wrap(title, TITLE_WRAP)
        h = H_TOP + H_TITLE * len(title_lines) + H_TGAP + len(lines) * H_BODY + H_BOT
        if label:
            h += H_PHASE
            if band:
                bands.append(band)
                band = []
        band.append(dict(phase=phase, label=label, title_lines=title_lines,
                         lines=lines, h=h))
    bands.append(band)
    return bands


def split_bands(bands):
    """Fewest panels under MAX_H, then the most balanced split at that count."""
    hs = [sum(s["h"] for s in b) + 2 * BAND_PAD + BAND_GAP for b in bands]
    n = len(bands)

    def pack(k):
        best = [None]

        def rec(i, parts, cur):
            if len(parts) + (1 if cur else 0) > k:
                return
            if i == n:
                if cur and len(parts) + 1 == k:
                    cand = parts + [cur]
                    tallest = max(HEADER + sum(hs[j] for j in p) for p in cand)
                    if tallest <= MAX_H and (best[0] is None or tallest < best[0][0]):
                        best[0] = (tallest, cand)
                return
            rec(i + 1, parts, cur + [i])
            if cur:
                rec(i + 1, parts + [cur], [i])

        rec(0, [], [])
        return best[0]

    for k in range(1, n + 1):
        got = pack(k)
        if got:
            return [[bands[j] for j in p] for p in got[1]]
    return [bands]


def draw_timeline():
    panels = split_bands(build_bands())
    first = 1
    for pi, panel in enumerate(panels, start=1):
        total = HEADER + sum(
            sum(s["h"] for s in b) + 2 * BAND_PAD + BAND_GAP for b in panel)
        fig = plt.figure(figsize=(FIG_W, total))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, FIG_W)
        ax.set_ylim(0, total)
        ax.axis("off")

        def Y(c):
            return total - c

        suffix = "" if len(panels) == 1 else f"   ({pi} of {len(panels)})"
        ax.text(0.06, Y(0.34), "How this project got here" + suffix,
                fontsize=18, fontweight="bold", family=SANS, color=INK, va="baseline")
        ax.text(0.06, Y(0.57),
                "Backdoor-detector portability to tabular NIDS -- one straight line"
                " through the whole project",
                fontsize=9.4, family=SANS, color=MUTED, style="italic", va="baseline")

        cursor = HEADER
        spine_top = cursor + 0.08
        n = first

        for band in panel:
            bh = sum(s["h"] for s in band) + 2 * BAND_PAD
            c = PHASES[band[0]["phase"]]
            ax.add_patch(FancyBboxPatch(
                (CARD_L, Y(cursor + bh)), CARD_R - CARD_L, bh,
                boxstyle="round,pad=0,rounding_size=0.085",
                facecolor=c["band"], edgecolor="none", zorder=1))

            y = cursor + BAND_PAD
            for step in band:
                ty = y + H_TOP
                if step["label"]:
                    ax.text(CARD_L + PAD_X, Y(ty + 0.140), step["label"],
                            fontsize=7.2, family=MONO, fontweight="bold",
                            color=c["label"], va="baseline", zorder=3)
                    ty += H_PHASE

                cy = Y(ty + 0.098)
                ax.plot([SPINE_X + 0.125, CARD_L - 0.04], [cy, cy],
                        color=SPINE, lw=1.4, solid_capstyle="round", zorder=2)
                ax.add_patch(Circle((SPINE_X, cy), 0.125, facecolor="white",
                                    edgecolor=c["accent"], lw=1.6, zorder=4))
                ax.text(SPINE_X, cy, str(n), fontsize=7.9, family=SANS,
                        fontweight="bold", color=c["accent"],
                        ha="center", va="center_baseline", zorder=5)
                n += 1

                for ti, tline in enumerate(step["title_lines"]):
                    ax.text(CARD_L + PAD_X, Y(ty + 0.158 + ti * H_TITLE), tline,
                            fontsize=10.9, family=SANS, fontweight="bold",
                            color=INK, va="baseline", zorder=3)
                ty += H_TITLE * len(step["title_lines"]) + H_TGAP

                for line in step["lines"]:
                    ty += H_BODY
                    ax.text(CARD_L + PAD_X, Y(ty - 0.034), line,
                            fontsize=8.4, family=SANS, color=MUTED,
                            va="baseline", zorder=3)
                y += step["h"]
            cursor += bh + BAND_GAP

        ax.plot([SPINE_X, SPINE_X], [Y(spine_top), Y(cursor - BAND_GAP + 0.02)],
                color=SPINE, lw=4.4, solid_capstyle="round", zorder=0)

        name = ("fig_explainer_timeline.pdf" if len(panels) == 1
                else f"fig_explainer_timeline_{pi}.pdf")
        fig.savefig(OUT / name, format="pdf")
        plt.close(fig)
        print(f"  {name}: {FIG_W:.2f} x {total:.2f} in, steps {first}-{n - 1}")
        first = n


if __name__ == "__main__":
    draw_timeline()
    print("done")
