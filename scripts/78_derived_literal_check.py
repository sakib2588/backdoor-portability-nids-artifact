#!/usr/bin/env python3
"""Recompute the paper_access literals that the Tier 1 ledger cannot trace to a stored value.

WHY (2026-09-03). scripts/34 traces a literal by matching it against a scalar stored in
results/*.json. That catches an invented number, but it reports UNTRACED for any figure the paper
derives by aggregating -- a mean over a cost group, a standard deviation across seeds, a sum of two
counts. Running the ledger on paper_access/ for the first time produced 18 UNTRACED literals; four
were template placeholders and the remaining fourteen were all derived, not invented.

An allowlist would silence those and check nothing. This script instead RECOMPUTES each one from
the committed results and asserts it equals what the manuscript prints, so a future edit that drifts
a number fails here rather than reaching a reviewer.

Every expected value below is the literal as printed in paper_access/sections/*.tex, with the file
and line it appears on. Change one only after re-deriving it.

Every check also reports WHERE its literal appears, resolved by grepping the section file at run
time rather than from a hand-written line pin. The pins rotted silently -- by 2026-09-06 all six
pointed at the wrong line, and nine `where` strings named the wrong section file outright, while
every check stayed green, because only the value decides the exit code.

Run:    python scripts/78_derived_literal_check.py
        python scripts/78_derived_literal_check.py --strict-location   # also fail on NOT FOUND
Exit:   0 if every derived literal reproduces, 1 otherwise.
"""
from __future__ import annotations

import json
import math
import re
import statistics as st
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

CTU_WINDOW = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]
UNSW_WINDOW = [(0.005, 16), (0.01, 8), (0.01, 16)]
ANCHOR = (0.005, 16)


def _finite(x) -> bool:
    return x is not None and math.isfinite(x)


def _auc(y, vals) -> float:
    """Direction-free AUC, matching scripts/101's own convention.

    The statistic may point either way, so 101 reports max(a, 1 - a) and this check must use the
    same convention or it would disagree with the run that produced the number.
    """
    a = float(roc_auc_score(list(y), [float(v) for v in vals]))
    return max(a, 1.0 - a)


# Default stays paper_access, the manuscript this instrument was written against.
# --paper <dir> retargets it; see the __main__ block. Without that flag this script
# silently audited paper_access no matter which manuscript was being worked on.
PAPER = ROOT / "paper_access" / "sections"
_SRC: dict[str, list[str]] = {}


def _source(name: str) -> list[str]:
    if name not in _SRC:
        # LaTeX writes thousands separators as 8{,}017 so the comma does not get spaced as
        # punctuation. Normalize them back, or every comma-bearing literal reads as absent.
        raw = (PAPER / name).read_text(encoding="utf-8")
        _SRC[name] = raw.replace("{,}", ",").splitlines()
    return _SRC[name]


def locate(where: str, printed: str) -> str:
    """Resolve `where` to the lines the literal actually occupies, at run time.

    WHY (2026-09-06). `where` used to carry a hand-written `file:line` pin. Nothing asserted it,
    because only the value comparison decides the exit code, so the pins rotted silently: by
    2026-09-06 all six pointed at the wrong line -- one at a blank line, one at a `\begin{table}`
    -- while every check stayed green. A green check that lies about its location is worse than a
    red one, because nobody re-reads it.

    Grepping the section file at run time cannot rot. It also upgrades the provenance note from a
    comment into a weak assertion: a literal that has left the manuscript entirely now reports
    NOT FOUND instead of a confident, stale line number.
    """
    name = where.split()[0].split(":")[0]
    tag = where.split(" ", 1)[1] if " " in where else ""
    if name == "DEFERRED":
        return tag or "deferred to the artifact"
    try:
        src = _source(name)
    except OSError:
        return f"{name}:UNREADABLE {tag}".strip()
    # Boundary-aware: a bare "0" or "75" must not match inside "0.0726" or "0.75".
    pat = re.compile(r"(?<![0-9.,])" + re.escape(printed) + r"(?![0-9,]*[0-9])")
    hits = [str(i) for i, line in enumerate(src, 1) if pat.search(line)]
    if not hits:
        loc = "NOT FOUND"
    elif len(hits) > 4:
        loc = ",".join(hits[:4]) + f",+{len(hits) - 4}"
    else:
        loc = ",".join(hits)
    return f"{name}:{loc} {tag}".strip()


def check(label: str, computed: str, printed: str, where: str, out: list,
          find: str | None = None) -> None:
    """`find` overrides what to grep for, when the check asserts a property rather than a literal.

    "thresholds all above max" prints True and "rows flagged" prints 0; neither string appears in
    the manuscript as such. Both are claims a sentence makes, so the sentence is what gets located.
    """
    ok = computed == printed
    out.append((ok, label, computed, printed, locate(where, find if find is not None else printed)))


def main(strict: bool = False) -> int:
    grid = json.load((RESULTS / "full_grid_mad_sweep.json").open())["rows"]
    sec = json.load((RESULTS / "secondary_detectors.json").open())["main_grid"]
    checks: list = []

    # --- Table "CTU-13 grid summarized by trigger cost" in 05_results.tex.
    # Cost groups over all five rates at z=3.0. The FPR column divides by CLEAN rows only
    # (n_benign - n_poison); the stored `fpr` field uses all benign rows and must not be read.
    groups = {
        "<=4": ([r for r in grid if r["cost"] <= 4], ("0.0726", "0.0077", "7.01", "0.0603", "0.0545", "0.5023", "5.67")),
        "8":   ([r for r in grid if r["cost"] == 8],  ("0.9998", "0.0004", "5.92", "0.4370", "0.4372", "0.9720", "2.79")),
        "16":  ([r for r in grid if r["cost"] == 16], ("1.0000", "0.0000", "5.25", "0.6906", "0.3895", "0.9889", "2.28")),
    }
    for name, (rs, exp) in groups.items():
        mad = [r["adaptive"]["3.0"]["recall"] for r in rs]
        fpr = [100 * r["adaptive"]["3.0"]["n_false_positive"] / (r["n_benign"] - r["n_poison"]) for r in rs]
        # The inherited budget's own false-positive rate, added so both sides of the cost
        # comparison in Section V-C are printed. Same clean-row denominator as the MAD column
        # above; the budget is ceil(1.5 * n_poison) and its true positives are recall * n_poison.
        fixfpr = [100 * (math.ceil(1.5 * r["n_poison"]) - r["fixed_budget_recall"] * r["n_poison"])
                  / (r["n_benign"] - r["n_poison"]) for r in rs]
        fx = [r["fixed_budget_recall"] for r in rs]
        auc = [r["spectral_auc"] for r in rs]
        for lbl, got, want in (
            ("MAD recall", f"{st.mean(mad):.4f}", exp[0]),
            ("MAD recall SD", f"{st.stdev(mad):.4f}", exp[1]),
            ("MAD FPR %", f"{st.mean(fpr):.2f}", exp[2]),
            ("fixed recall", f"{st.mean(fx):.4f}", exp[3]),
            ("fixed recall SD", f"{st.stdev(fx):.4f}", exp[4]),
            ("spectral AUC", f"{st.mean(auc):.4f}", exp[5]),
            ("fixed FPR %", f"{st.mean(fixfpr):.2f}", exp[6]),
        ):
            check(f"cost {name}: {lbl}", got, want, "05_results.tex (cost table)", checks)

    # --- Per-cost AUCs quoted in that table's caption in 05_results.tex.
    for cost, want in ((1, "0.5016"), (2, "0.5024"), (4, "0.5029")):
        v = [r["spectral_auc"] for r in grid if r["cost"] == cost]
        check(f"cost {cost} mean spectral AUC", f"{st.mean(v):.4f}", want, "05_results.tex (cost table caption)", checks)

    # --- Activation Clustering cost-16 mean over five rates, in 05_results.tex.
    ac16 = [r["ac_fixed_recall"] for r in grid if r["cost"] == 16]
    check("AC cost-16 fixed recall", f"{st.mean(ac16):.4f}", "0.6801", "05_results.tex (AC cost-16)", checks)

    # --- Budget-free rule on the CTU-13 window, and the anchor cell, in 06_discussion.tex.
    win = [r for r in grid if (r["rate"], r["cost"]) in CTU_WINDOW]
    anc = [r for r in grid if (r["rate"], r["cost"]) == ANCHOR]
    winfpr = [100 * r["adaptive"]["3.0"]["n_false_positive"] / (r["n_benign"] - r["n_poison"]) for r in win]
    check("window MAD FPR %", f"{st.mean(winfpr):.2f}", "6.58", "06_discussion.tex (budget-free rule)", checks)
    # The window's INHERITED-budget false-positive rate. Prose-only literal, and it collides
    # numerically with STRIP's UNSW delivered false-rejection rate of 1.03%; the two are
    # unrelated quantities that happen to round the same, so both are gated separately.
    winfx = [100 * (math.ceil(1.5 * r["n_poison"]) - r["fixed_budget_recall"] * r["n_poison"])
             / (r["n_benign"] - r["n_poison"]) for r in win]
    check("window fixed FPR %", f"{st.mean(winfx):.2f}", "1.03",
          "06_discussion.tex (window cost)", checks)
    check("window mean precision",
          f"{st.mean(r['adaptive']['3.0']['n_true_positive'] / r['adaptive']['3.0']['n_flagged'] for r in win):.3f}",
          "0.105", "05_results.tex (budget-free rule)", checks)
    check("anchor mean precision",
          f"{st.mean(r['adaptive']['3.0']['n_true_positive'] / r['adaptive']['3.0']['n_flagged'] for r in anc):.4f}",
          "0.0714", "05_results.tex (budget-free rule)", checks)
    check("anchor flagged flows",
          f"{st.mean(r['adaptive']['3.0']['n_flagged'] for r in anc):,.0f}",
          "8,017", "05_results.tex (budget-free rule)", checks)

    # Added 2026-09-08 after review round v12. The manuscript now compares the two rules
    # on MATCHED cells rather than against the window mean, so both fed cells' costs are
    # printed and must reproduce. Same false-positive denominator as the window figure
    # above: clean rows only, that is n_benign - n_poison.
    for (rate, cost), fx_want, md_want in [((0.05, 16), "2.70", "4.67"), ((0.1, 16), "5.71", "3.44")]:
        cell = [r for r in grid if (r["rate"], r["cost"]) == (rate, cost)]
        fx = [100 * (math.ceil(1.5 * r["n_poison"]) - r["fixed_budget_recall"] * r["n_poison"])
              / (r["n_benign"] - r["n_poison"]) for r in cell]
        md = [100 * r["adaptive"]["3.0"]["n_false_positive"] / (r["n_benign"] - r["n_poison"]) for r in cell]
        check(f"fed cell {rate}/{cost} fixed FPR %", f"{st.mean(fx):.2f}", fx_want,
              "05_results.tex (cheaper rule)", checks)
        check(f"fed cell {rate}/{cost} MAD FPR %", f"{st.mean(md):.2f}", md_want,
              "05_results.tex (cheaper rule)", checks)

    # Density-mitigation comparator ranking AUCs, printed in Appendix J from round v12 on.
    for fn, want in [("density_mitigation_ctu.json", "0.4991"),
                     ("density_mitigation_ctu.control_lgb.json", "0.4104"),
                     ("density_mitigation_ctu.control_selected.json", "0.5693"),
                     ("density_mitigation_ctu.control_bypass.json", "0.6000")]:
        ctrls = json.loads((RESULTS / fn).read_text())["controls"]
        check(f"density-mitigation AUC {fn.split('.')[1] if '.' in fn[:-5] else 'default'}",
              f"{st.mean(c['auc'] for c in ctrls):.4f}", want,
              "07_appendix.tex (density mitigation)", checks)

    # --- Activation Clustering on the UNSW-NB15 window, in 05_results.tex.
    acw = [r["ac"]["recall"] for r in sec if (r.get("rate"), r.get("cost")) in UNSW_WINDOW]
    check("AC UNSW window recall", f"{st.mean(acw):.4f}", "0.1443", "05_results.tex (AC UNSW window)", checks)

    # --- Stated arithmetic identities, which have no disk value by construction.
    check("CTU-13 release total", f"{143046 + 55082:,}", "198,128", "04_methods.tex (CTU-13 release)", checks)
    # The UNSW-NB15 mirror's pre-deduplication row count. Not a stored scalar: it is the
    # retained rows plus the exact duplicates dropped, both of which the audit stores.
    _sa = json.loads((RESULTS / "secondary_data_audit.json").read_text())["schema"]
    check("UNSW mirror rows before dedup",
          f"{_sa['n_rows'] + _sa['n_exact_duplicate_rows_dropped']:,}", "2,280,090",
          "07_appendix.tex (UNSW mirror)", checks)
    check("median poison rank", f"{1022 + 286:,}", "1,308", "06_discussion.tex (median poison rank)", checks)

    # --- Rule selection on the full 25-cell grid (05_results.tex, the moment-statistic
    # paragraphs). Added 2026-09-04 when the full-grid re-run refuted the narrow-grid claim.
    # Every figure below is recomputed here rather than read from the file's own summary
    # fields, so a rerun that changes the answer fails this check instead of silently
    # agreeing with a stale sentence.
    rsel = json.load((RESULTS / "rule_selection_statistic.json").open())
    units = [u for u in rsel["units"]
             if _finite(u["S1_bimodality"]) and _finite(u["S2_top_gap"])]
    check("rule-selection units", f"{len(units)}", "625",
          "DEFERRED SUPPLEMENTARY.md (moment statistic)", checks)

    y_all = [bool(u["mad_wins"]) for u in units]
    check("pooled S1 prediction AUC",
          f"{_auc(y_all, [u['S1_bimodality'] for u in units]):.4f}", "0.8196",
          "DEFERRED SUPPLEMENTARY.md (moment statistic)", checks)
    check("pooled rate prediction AUC",
          f"{_auc(y_all, [u['rate'] for u in units]):.4f}", "0.7138",
          "DEFERRED SUPPLEMENTARY.md (moment statistic)", checks)

    # Stratified by arm. These are the numbers that reverse the pooled ordering.
    for arm, n_mad, n_bud, s1_want, rate_want in (
            ("spectral", "80", "45", "0.5392", "0.9861"),
            ("isolation_forest", None, None, "0.5156", "0.9673"),
            ("boundary_departure", None, None, "0.6759", "0.8526")):
        sub = [u for u in units if u["detector"] == arm]
        ys = [bool(u["mad_wins"]) for u in sub]
        if n_mad is not None:
            check(f"{arm} MAD wins", f"{sum(ys)}", n_mad,
                  "DEFERRED SUPPLEMENTARY.md (moment statistic)", checks)
            check(f"{arm} budget wins", f"{len(ys) - sum(ys)}", n_bud,
                  "DEFERRED SUPPLEMENTARY.md (moment statistic)", checks)
        check(f"{arm} S1 AUC", f"{_auc(ys, [u['S1_bimodality'] for u in sub]):.4f}", s1_want,
              "DEFERRED SUPPLEMENTARY.md (moment statistic)", checks)
        check(f"{arm} rate AUC", f"{_auc(ys, [u['rate'] for u in sub]):.4f}", rate_want,
              "DEFERRED SUPPLEMENTARY.md (moment statistic)", checks)

    # The degenerate arms, quoted in the scope paragraph as "all 125".
    for arm, want in (("spectre", "125"), ("strip", "125")):
        sub = [u for u in units if u["detector"] == arm]
        check(f"{arm} units", f"{len(sub)}", want,
              "05_results.tex (scope paragraph)", checks)

    # Spectral's rate curve, quoted in the starvation paragraph.
    spec = [u for u in units if u["detector"] == "spectral"]
    for rate, fx_want, mad_want in ((0.005, "0.0219", "0.4415"), (0.1, "0.4937", "0.4434")):
        rs = [u for u in spec if u["rate"] == rate]
        check(f"spectral fixed recall at {rate}",
              f"{st.mean(u['fixed_recall'] for u in rs):.4f}", fx_want,
              "05_results.tex (starvation paragraph)", checks)
        check(f"spectral MAD recall at {rate}",
              f"{st.mean(u['mad_recall'] for u in rs):.4f}", mad_want,
              "05_results.tex (starvation paragraph)", checks)

    # --- Post-removal attack success. Reported in 05_results.tex; the `where` said
    # 01_introduction.tex until 2026-09-06.
    pr = json.load((RESULTS / "secondary_post_removal_asr.json").open())["per_cell"]
    post = {(c["rate"], c["cost"]): c["post_removal_asr"]["mean"] for c in pr}
    for cell, want in (((0.01, 8), "0.0546"), ((0.005, 16), "0.4310"), ((0.01, 16), "0.6154")):
        check(f"post-removal ASR {cell}", f"{post[cell]:.4f}", want,
              "05_results.tex (post-removal ASR)", checks)

    # --- Backdoor fractions across the corpus family, promoted to a contribution.
    q2 = json.load((RESULTS / "netflow_property_analysis.json").open())["q2_summary"]
    for corpus, want in (("NF-CSE-CIC-IDS2018-v2", "0.4465"),
                         ("NF-ToN-IoT-v2", "0.5964"),
                         ("NF-UNSW-NB15-v2", "0.5478")):
        check(f"backdoor fraction {corpus}",
              f"{q2[corpus]['mean_backdoor_fraction']:.4f}", want,
              "05_results.tex (corpus family)", checks)

    # --- STRIP under its own published rule (05_results.tex, the STRIP paragraph), added
    # 2026-09-04 when the FRR rule was wired in. Means over window cells and five seeds.
    strip = json.load((RESULTS / "strip_detector.json").open())["rows"]

    def strip_cells(ds, cost=None):
        out = []
        for k, v in strip.items():
            d, _seed, rate, c = k.split("|")
            if d != ds or (float(rate), int(c)) not in (
                    CTU_WINDOW if ds == "ctu13" else UNSW_WINDOW):
                continue
            if cost is not None and int(c) != cost:
                continue
            out.append(v["strip"])
        return out

    ctu = strip_cells("ctu13")
    ctu16 = strip_cells("ctu13", cost=16)
    unsw = strip_cells("unsw_nb15")
    check("STRIP CTU window FRR recall", f"{st.mean(r['frr_recall'] for r in ctu):.4f}", "0.6920",
          "05_results.tex (STRIP)", checks)
    check("STRIP CTU window FRR fpr", f"{st.mean(r['frr_fpr'] for r in ctu):.4f}", "0.0217",
          "05_results.tex (STRIP)", checks)
    check("STRIP CTU cost-16 FRR recall", f"{st.mean(r['frr_recall'] for r in ctu16):.4f}", "0.9441",
          "07_appendix.tex (STRIP)", checks)
    check("STRIP CTU cost-16 FRR fpr", f"{st.mean(r['frr_fpr'] for r in ctu16):.4f}", "0.0212",
          "05_results.tex (STRIP)", checks)
    check("STRIP CTU window budget", f"{st.mean(r['fixed_recall'] for r in ctu):.4f}", "0.2806",
          "05_results.tex (STRIP)", checks)
    check("STRIP CTU window MAD", f"{st.mean(r['mad']['3.0']['recall'] for r in ctu):.4f}", "0.0000",
          "05_results.tex (STRIP)", checks)
    check("STRIP UNSW FRR recall", f"{st.mean(r['frr_recall'] for r in unsw):.4f}", "0.0007",
          "07_appendix.tex (STRIP)", checks)
    check("STRIP UNSW ranking AUC", f"{st.mean(r['auc'] for r in unsw):.4f}", "0.6271",
          "05_results.tex (STRIP)", checks)
    check("STRIP CTU realized FRR",
          f"{st.mean(r['frr_calibration']['realized_frr'] for r in ctu):.4f}", "0.0214",
          "05_results.tex (STRIP)", checks)
    check("STRIP UNSW realized FRR",
          f"{st.mean(r['frr_calibration']['realized_frr'] for r in unsw):.4f}", "0.0103",
          "05_results.tex (STRIP)", checks)

    # --- E2, the multiplier sweep on UNSW-NB15 (05_results.tex, the recalibration paragraph).
    mult = json.load((RESULTS / "secondary_budget_multiplier_sweep.json").open())
    bm, bz = mult["by_multiplier"], mult["by_z"]
    for m, r_want, f_want in (("1.5", "0.0216", None), ("7", "0.7204", "0.0536"),
                              ("10", "0.9234", "0.0782")):
        check(f"UNSW multiplier {m} recall", f"{bm[m]['recall_mean']:.4f}", r_want,
              "05_results.tex (recalibration)", checks)
        if f_want:
            check(f"UNSW multiplier {m} FPR", f"{bm[m]['clean_fpr_mean']:.4f}", f_want,
                  "05_results.tex (recalibration)", checks)
    check("UNSW MAD z=3 recall", f"{bz['3.0']['recall_mean']:.4f}", "0.8861",
          "05_results.tex (recalibration)", checks)
    check("UNSW MAD z=3 FPR", f"{bz['3.0']['clean_fpr_mean']:.4f}", "0.0643",
          "05_results.tex (recalibration)", checks)

    # --- E3, imbalance on UNSW-NB15 (05_results.tex and 07_appendix.tex).
    h3u = json.load((RESULTS / "secondary_h3_window_auc.json").open())["summary"]
    for arm, field, want, where in (
            ("balanced", "spectral_recall", "0.8814", "05_results.tex (H3)"),
            ("balanced", "spectral_auc", "0.9924", "05_results.tex (H3)"),
            ("balanced", "ac_recall", "0.7229", "07_appendix.tex (H3)"),
            ("imbalanced", "spectral_recall", "0.0216", "05_results.tex (H3)"),
            ("imbalanced", "spectral_auc", "0.9581", "05_results.tex (H3)"),
            ("imbalanced", "ac_recall", "0.1443", "07_appendix.tex (H3)")):
        check(f"UNSW H3 {arm} {field}", f"{h3u[arm][field]['mean']:.4f}", want, where, checks)

    h3rows = json.load((RESULTS / "secondary_h3_window_auc.json").open())["rows"]
    tb = next(r["balanced"]["train_balance"] for r in h3rows if r.get("interpretable"))
    check("UNSW balanced train rows", f"{tb['n']:,}", "68,378", "05_results.tex (H3)", checks)
    check("UNSW balanced minority rows", f"{tb['minority']:,}", "34,189",
          "07_appendix.tex (H3)", checks)

    h3c = json.load((RESULTS / "h3_window_auc.json").open())
    check("CTU H3 balanced AUC", f"{h3c['summary']['balanced']['spectral_auc']['mean']:.4f}",
          "0.5559", "05_results.tex (H3)", checks)
    check("CTU balanced train rows",
          f"{next(r['balanced']['train_balance']['n'] for r in h3c['rows']):,}", "5,540",
          "05_results.tex (H3)", checks)

    # --- Why the budget-free cutoff cannot fire on STRIP. These 6 moved to 07_appendix.tex; the
    # `where` strings said 05_results.tex until 2026-09-06 and nothing noticed, because only the
    # value decided the exit code.
    bound = json.load((RESULTS / "strip_mad_bound.json").open())["summary"]
    check("STRIP score median", f"{bound['median']:.4f}", "-0.0558",
          "07_appendix.tex (STRIP)", checks)
    check("STRIP score MAD", f"{bound['mad']:.4f}", "0.0469", "07_appendix.tex (STRIP)", checks)
    check("STRIP cutoff at z=3.0", f"{bound['threshold']['3.0']:+.4f}", "+0.1530",
          "07_appendix.tex (STRIP)", checks)
    check("STRIP cutoff at z=2.0", f"{bound['threshold']['2.0']:+.4f}", "+0.0834",
          "07_appendix.tex (STRIP)", checks)
    check("STRIP thresholds all above max", f"{bound['all_thresholds_above_max']}", "True",
          "07_appendix.tex (STRIP)", checks, find="Both sit above the largest value")
    check("STRIP rows flagged by MAD", f"{bound['total_flagged']}", "0",
          "07_appendix.tex (STRIP)", checks, find="the rule flags no row on any seed")

    # --- The 2 rank counts at the anchor cell. They are different counts of the same cell, and a
    # blind reviewer read them as one quantity given 2 values. 05_results.tex counts CLEAN rows
    # above the typical poison sample; 06_discussion.tex adds the poison rows above their own
    # median, which is half of them by definition.
    anchor = [r for r in grid if r["rate"] == 0.005 and r["cost"] == 16]
    a_auc = st.mean(r["spectral_auc"] for r in anchor)
    a_clean = st.mean(r["n_benign"] - r["n_poison"] for r in anchor)
    a_pois = st.mean(r["n_poison"] for r in anchor)
    clean_above = round((1 - a_auc) * a_clean)
    check("anchor clean rows above poison", f"{clean_above:,}", "1,022",
          "05_results.tex (starvation)", checks)
    check("anchor rows of either kind above poison median",
          f"{round(clean_above + a_pois / 2):,}", "1,308",
          "06_discussion.tex (starvation)", checks)

    # --- What the INHERITED budget costs where it actually works. The grid stores no
    # false-positive field for the fixed rule, only for the budget-free one, so the count is derived
    # from the rule itself: flag_by_scores flags exactly ceil(1.5 * expected_frac * n) rows, and
    # expected_frac is n_poison/n, so the budget is ceil(1.5 * n_poison) regardless of n. True
    # positives are recall * n_poison. The derivation reproduces the anchor cell's printed 0.74%,
    # which is what licenses using it for the 2 cells where the budget reaches full recall.
    def _fixed_fp_rate(rate, cost):
        rs = [r for r in grid if r["rate"] == rate and r["cost"] == cost]
        fps = [math.ceil(1.5 * r["n_poison"]) - r["fixed_budget_recall"] * r["n_poison"] for r in rs]
        return st.mean(f / (r["n_benign"] - r["n_poison"]) for f, r in zip(fps, rs))

    for (rate, cost), want in (((0.005, 16), "0.74"), ((0.05, 16), "2.70"), ((0.1, 16), "5.71")):
        check(f"fixed budget FPR at rate {rate} cost {cost}",
              f"{100 * _fixed_fp_rate(rate, cost):.2f}", want,
              "05_results.tex (recommendation 1)", checks, find=f"{want}\\%")

    # --- The loud tabular control, re-run 2026-09-06 with STRIP's pool off the test partition.
    # Spectral and Activation Clustering do not read that pool and reproduce bit-identically; only
    # STRIP moves, and it stays above the 0.70 admission bar.
    pc = json.load((RESULTS / "positive_control_strip_spectre.json").open())
    pcs, pcr = pc["summary"], list(pc["per_seed"])
    # Table 5 prints the per-seed MINIMUM, not the mean, because the admission bar is stated per
    # seed. The means below still reproduce and are kept as pipeline checks; the minima are the
    # values the manuscript actually asserts.
    check("loud control spectral mean", f"{pcs['spectral_recall']['mean']:.4f}", "0.9995",
          "04_methods.tex (loud control)", checks)
    check("loud control AC mean", f"{pcs['ac_recall']['mean']:.4f}", "0.9998",
          "04_methods.tex (loud control)", checks)
    check("loud control STRIP budget mean", f"{pcs['strip_fixed_recall']['mean']:.4f}", "0.9357",
          "04_methods.tex (loud control)", checks)
    check("loud control spectral worst seed",
          f"{min(r['spectral_recall'] for r in pcr):.4f}", "0.9981",
          "04_methods.tex (loud control)", checks)
    check("loud control AC worst seed",
          f"{min(r['ac_recall'] for r in pcr):.4f}", "0.9990",
          "04_methods.tex (loud control)", checks)
    check("loud control STRIP budget worst seed",
          f"{min(r['strip']['fixed_recall'] for r in pcr):.4f}", "0.9250",
          "04_methods.tex (loud control)", checks)
    # Dual-rule columns added in the round-v7 follow-up: every arm scored under both rules at the
    # loud control, so the admission bar is not evaluated under the rule this paper calls broken.
    check("loud control spectral budget-free worst seed",
          f"{min(r['spectral']['mad_recall']['3.0'] for r in pcr):.4f}", "1.0000",
          "04_methods.tex (loud control)", checks)
    check("loud control STRIP budget-free worst seed",
          f"{min(r['strip']['mad_recall']['3.0'] for r in pcr):.4f}", "0.0000",
          "04_methods.tex (loud control)", checks)
    check("loud control SPECTRE budget-free worst seed",
          f"{min(r['spectre']['mad_recall']['3.0'] for r in pcr):.4f}", "0.5325",
          "04_methods.tex (loud control)", checks)
    check("loud control spectral AUC", f"{pcs['spectral_auc']['mean']:.4f}", "0.9826",
          "04_methods.tex (loud control)", checks)
    # Moved to Section IV-E with the bound argument it supports; Methods states the fact without
    # restating the number.
    check("loud control STRIP AUC", f"{pcs['strip_auc']['mean']:.4f}", "0.9787",
          "05_results.tex (bound account)", checks)

    # UNSW-NB15 loud control, same dual-rule treatment. Its Spectral published-rule worst seed is
    # the narrowest margin in Table 5, and the budget-free column is what shows that margin is a
    # starved budget rather than a weak control.
    sc = json.load((RESULTS / "secondary_positive_control.json").open())
    scs, scr = sc["summary"], list(sc["per_seed"])
    check("UNSW loud control spectral worst seed",
          f"{min(r['spectral_recall'] for r in scr):.4f}", "0.7009",
          "04_methods.tex (loud control)", checks)
    check("UNSW loud control spectral budget-free worst seed",
          f"{min(r['spectral']['mad_recall']['3.0'] for r in scr):.4f}", "0.9873",
          "04_methods.tex (loud control)", checks)
    check("UNSW loud control AC worst seed",
          f"{min(r['ac_recall'] for r in scr):.4f}", "0.9792",
          "04_methods.tex (loud control)", checks)
    check("UNSW loud control STRIP worst seed",
          f"{min(r['strip']['fixed_recall'] for r in scr):.4f}", "0.7212",
          "04_methods.tex (loud control)", checks)
    check("UNSW loud control STRIP budget-free worst seed",
          f"{min(r['strip']['mad_recall']['3.0'] for r in scr):.4f}", "0.0000",
          "04_methods.tex (loud control)", checks)
    check("UNSW loud control SPECTRE worst seed",
          f"{min(r['spectre']['fixed_recall'] for r in scr):.4f}", "0.0743",
          "04_methods.tex (loud control)", checks)
    check("UNSW loud control SPECTRE budget-free worst seed",
          f"{min(r['spectre']['mad_recall']['3.0'] for r in scr):.4f}", "0.4835",
          "04_methods.tex (loud control)", checks)
    check("UNSW loud control spectral AUC", f"{scs['spectral_auc']['mean']:.4f}", "0.9895",
          "04_methods.tex (loud control)", checks)
    check("UNSW loud control STRIP AUC", f"{scs['strip_auc']['mean']:.4f}", "0.9795",
          "04_methods.tex (loud control)", checks)
    # Neural Cleanse's clean-model calibration at the loud tabular control. Section III-J now
    # defines this rule and prints both ratio vectors; the "2 of 5" in Table 5 is the count of
    # seeds where the poisoned model's mask ratio exceeds its own clean model's.
    tpc = json.load((RESULTS / "tabular_positive_control.json").open())["per_seed"]
    check("NC loud control seeds exceeding clean calibration",
          str(sum(1 for r in tpc if r["nc_ratio"] > r["nc_clean_ratio"])), "2",
          "04_methods.tex (loud control)", checks)
    def _prose_list(vals):
        # The manuscript writes these as "a, b, c, d and e"; match that so a found value is not
        # reported as missing on a separator mismatch.
        v = [f"{x:.2f}" for x in vals]
        return ", ".join(v[:-1]) + " and " + v[-1]
    check("NC loud control poisoned ratios",
          _prose_list([r["nc_ratio"] for r in tpc]),
          "90.11, 103.31, 88.29, 107.51 and 76.66", "04_methods.tex (loud control)", checks)
    check("NC loud control clean ratios",
          _prose_list([r["nc_clean_ratio"] for r in tpc]),
          "97.96, 81.12, 81.33, 109.22 and 94.89", "04_methods.tex (loud control)", checks)

    # Round v8: literals added when the reviewers' findings were closed.
    ap = json.load((RESULTS / "active_paths_window.json").open())["controls"]
    check("active paths, budget-free worst seed",
          f"{min(c['active_paths']['mad_recall']['3.0'] for c in ap):.4f}", "0.0000",
          "07_appendix.tex (comparators)", checks)
    check("active paths, mean ranking AUC",
          f"{st.mean(c['active_paths']['auc'] for c in ap):.4f}", "0.7737",
          "07_appendix.tex (comparators)", checks)
    nd = json.load((RESULTS / "netflow_detectors.json").open())["cells"]
    na = json.load((RESULTS / "netflow_detector_analysis.json").open())
    wc = set(round(x, 10) for x in na["ac_modes"]["wrong_cluster_fpr"])
    wsil = [v["ac"]["silhouette"] for v in nd.values()
            if v.get("interpretable") and (v.get("ac") or {}).get("fpr") is not None
            and round(v["ac"]["fpr"], 10) in wc]
    isil = [v["ac"]["silhouette"] for v in nd.values()
            if v.get("interpretable") and (v.get("ac") or {}).get("recall", 0) >= 0.9]
    check("wrong-cluster silhouette mean", f"{st.mean(wsil):.4f}", "0.5394",
          "07_appendix.tex (netflow detectors)", checks)
    check("wrong-cluster silhouette worst", f"{min(wsil):.4f}", "0.4406",
          "07_appendix.tex (netflow detectors)", checks)
    check("isolating silhouette mean", f"{st.mean(isil):.4f}", "0.7740",
          "07_appendix.tex (netflow detectors)", checks)
    aa = json.load((RESULTS / "adaptive_attacker.json").open())
    check("adaptive attacker candidate budget", str(aa["config"]["max_candidates"]), "64",
          "04_methods.tex (statistical protocol)", checks)
    check("adaptive attacker aggregated candidates",
          str(len(aa["selection"]["aggregated_candidates"])), "28",
          "04_methods.tex (statistical protocol)", checks)
    sh = json.load((RESULTS / "surrogate_shap.json").open())["summary"]
    check("surrogate LGB evasion component", f"{sh['surrogate_lgb']['evasion_cost16']:.4f}",
          "0.3179", "04_methods.tex (threat model)", checks)

    check("CTU loud control SPECTRE worst seed",
          f"{min(r['spectre']['fixed_recall'] for r in pcr):.4f}", "0.0734",
          "04_methods.tex (loud control)", checks)
    check("loud control STRIP own rule",
          f"{st.mean(r['strip']['frr_recall'] for r in pcr):.4f}", "0.8872",
          "04_methods.tex (loud control)", checks)

    # --- The corpus-family control, now planted on each corpus rather than pooled on one.
    for corpus, det, field, want in (
            ("NF-CSE-CIC-IDS2018-v2", "spectral", "fixed_recall", "0.9971"),
            ("NF-ToN-IoT-v2",         "spectral", "fixed_recall", "1.0000"),
            ("NF-UNSW-NB15-v2",       "spectral", "fixed_recall", "0.9781"),
            ("NF-ToN-IoT-v2",         "ac",       "recall",       "0.5998"),
            ("NF-UNSW-NB15-v2",       "ac",       "recall",       "0.9984"),
            ("NF-ToN-IoT-v2",         "strip",    "fixed_recall", "0.9297"),
            ("NF-UNSW-NB15-v2",       "strip",    "fixed_recall", "0.7387"),
            ("NF-ToN-IoT-v2",         "spectre",  "fixed_recall", "0.1090"),
            ("NF-UNSW-NB15-v2",       "spectre",  "fixed_recall", "0.0830")):
        src = ("netflow_detectors.json" if corpus == "NF-CSE-CIC-IDS2018-v2"
               else f"netflow_control_{corpus}.json")
        ctrls = json.load((RESULTS / src).open())["controls"].values()
        got = st.mean(r[det][field] for r in ctrls)
        check(f"family control {corpus} {det}", f"{got:.4f}", want,
              "07_appendix.tex (family control)", checks)

    # --- Literals added 2026-09-06 answering review round 3. The window's own fixed-budget cost,
    # the adaptive attacker's numbers, the per-corpus mode counts behind "both outcomes in every
    # corpus", and the 2 gaps the post-hoc window criterion's thresholds sit inside.
    win = [r for r in grid if (r["rate"], r["cost"]) in {(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)}]
    win_fp = st.mean((math.ceil(1.5 * r["n_poison"]) - r["fixed_budget_recall"] * r["n_poison"])
                     / (r["n_benign"] - r["n_poison"]) for r in win)
    check("window fixed-budget FPR", f"{100 * win_fp:.2f}", "1.03",
          "06_discussion.tex (recommendation 1)", checks, find="1.03\\%")

    aa = json.load((RESULTS / "adaptive_attacker.json").open())["evaluation"]["adaptive_outcomes"]
    for row, want in zip(sorted(aa, key=lambda r: r["seed"]), ("6.47", "6.58")):
        check(f"adaptive attacker FPR seed {row['seed']}", f"{100 * row['mad_fpr']:.2f}", want,
              "05_results.tex (adaptive attacker)", checks, find=f"{want}\\%")

    fam = list(json.load((RESULTS / "netflow_poison_sweep.json").open())["cells"].values())
    for corpus, want_lo, want_hi, want_n in (("NF-CSE-CIC-IDS2018-v2", "7", "18", "46"),
                                             ("NF-ToN-IoT-v2", "3", "6", "21"),
                                             ("NF-UNSW-NB15-v2", "6", "6", "21")):
        bf = [r["backdoor_fraction"] for r in fam
              if r["corpus"] == corpus and r["status"] == "ok" and r.get("backdoor_fraction") is not None]
        check(f"{corpus} cells at or below 0.10", str(sum(1 for v in bf if v <= 0.10)), want_lo,
              "05_results.tex (family modes)", checks)
        check(f"{corpus} cells at or above 0.90", str(sum(1 for v in bf if v >= 0.90)), want_hi,
              "05_results.tex (family modes)", checks)
        check(f"{corpus} usable cells", str(len(bf)), want_n,
              "05_results.tex (family modes)", checks)

    prim = json.load((RESULTS / "detectors.json").open())["main_grid"]
    by_cell = {}
    for r in prim:
        by_cell.setdefault((r["rate"], r["cost"]), []).append(r["asr"])
    asr_means = sorted(st.mean(v) for v in by_cell.values())
    below = max(v for v in asr_means if v < 0.8)
    above = min(v for v in asr_means if v >= 0.8)
    check("ASR gap lower edge", f"{below:.3f}", "0.637", "07_appendix.tex (criterion gap)", checks)
    check("ASR gap upper edge", f"{above:.3f}", "0.870", "07_appendix.tex (criterion gap)", checks)
    fx = sorted(st.mean(r["fixed_budget_recall"] for r in grid
                        if r["rate"] == c[0] and r["cost"] == c[1]) for c in by_cell)
    fbelow = max(v for v in fx if v < 0.5)
    fabove = min(v for v in fx if v >= 0.5)
    check("fixed-recall gap lower edge", f"{fbelow:.3f}", "0.449", "07_appendix.tex (criterion gap)", checks)
    check("fixed-recall gap upper edge", f"{fabove:.3f}", "0.883", "07_appendix.tex (criterion gap)", checks)

    # --- The label-free capability check. Reported in 05_results.tex, not 06_discussion.tex.
    cap = json.load((RESULTS / "rule_capability_taxonomy.json").open())
    check("STRIP z_max", f"{cap['summary']['strip']['z_max_mean']:.2f}", "0.79",
          "05_results.tex (capability)", checks, find="anchor cell is 0.79")
    check("lowest firing arm z_max",
          f"{min(v['z_max_mean'] for k, v in cap['summary'].items() if k != 'strip' and math.isfinite(v['z_max_mean'])):.1f}",
          "38.7", "05_results.tex (capability)", checks)
    check("capability agreement",
          f"{cap['prediction_agreement']['agreements']} of {cap['prediction_agreement']['checks']}",
          "75 of 75", "05_results.tex (capability)", checks)

    # --- Activation Clustering's partition on the corpus family (07_appendix.tex). Added
    # 2026-09-05 after a reviewer's arithmetic claimed 41 should be 39. The data says 41, and
    # the real defect was the claim that every failing cell reports exactly 0.000.
    nfd = json.load((RESULTS / "netflow_detectors.json").open()).get("cells", {})
    acr = [v["ac"]["recall"] for v in nfd.values()
           if isinstance(v.get("ac"), dict) and v["ac"].get("recall") is not None]
    check("AC family interpretable cells", f"{len(acr)}", "75", "07_appendix.tex (AC)", checks)
    check("AC cells isolating poison", f"{sum(1 for r in acr if r >= 0.9)}", "41",
          "07_appendix.tex (AC)", checks)
    check("AC cells at exactly zero", f"{sum(1 for r in acr if r == 0.0)}", "32",
          "07_appendix.tex (AC)", checks)
    check("AC cells bimodal", f"{sum(1 for r in acr if r >= 0.9 or r == 0.0)}", "73",
          "07_appendix.tex (AC)", checks)

    width = max(len(c[1]) for c in checks)
    failed = 0
    unlocated = []
    for ok, label, got, want, where in checks:
        if not ok:
            failed += 1
        if "NOT FOUND" in where:
            unlocated.append((label, want, where))
        print(f"  {'ok ' if ok else 'FAIL'}  {label:<{width}}  computed {got:>9s}  printed {want:>9s}   {where}")
    print(f"\n{len(checks) - failed} of {len(checks)} derived literals reproduce from the committed results")

    print(f"{len(checks) - len(unlocated)} of {len(checks)} were located in paper_access/sections")
    if unlocated:
        print("\n  These values reproduce from results/ but do not appear in the manuscript. Each is\n"
              "  either deferred to the artifact repository or dropped from the text. Recomputing a\n"
              "  number the paper no longer prints checks the pipeline, not the paper:")
        for label, want, where in unlocated:
            print(f"    {label:<{width}}  {want:>9s}   {where}")
    if strict and unlocated:
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    _argv = sys.argv[1:]
    if "--paper" in _argv:
        PAPER = ROOT / _argv[_argv.index("--paper") + 1] / "sections"
    print(f"auditing {PAPER.parent.name}/\n")
    sys.exit(main(strict="--strict-location" in sys.argv))
