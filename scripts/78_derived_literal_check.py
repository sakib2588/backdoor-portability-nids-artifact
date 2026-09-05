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

Run:  python scripts/78_derived_literal_check.py
Exit: 0 if every derived literal reproduces, 1 otherwise.
"""
from __future__ import annotations

import json
import math
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


def check(label: str, computed: str, printed: str, where: str, out: list) -> None:
    ok = computed == printed
    out.append((ok, label, computed, printed, where))


def main() -> int:
    grid = json.load((RESULTS / "full_grid_mad_sweep.json").open())["rows"]
    sec = json.load((RESULTS / "secondary_detectors.json").open())["main_grid"]
    checks: list = []

    # --- Table "CTU-13 grid summarized by trigger cost" (05_results.tex:241-243).
    # Cost groups over all five rates at z=3.0. The FPR column divides by CLEAN rows only
    # (n_benign - n_poison); the stored `fpr` field uses all benign rows and must not be read.
    groups = {
        "<=4": ([r for r in grid if r["cost"] <= 4], ("0.0726", "0.0077", "7.01", "0.0603", "0.0545", "0.5023")),
        "8":   ([r for r in grid if r["cost"] == 8],  ("0.9998", "0.0004", "5.92", "0.4370", "0.4372", "0.9720")),
        "16":  ([r for r in grid if r["cost"] == 16], ("1.0000", "0.0000", "5.25", "0.6906", "0.3895", "0.9889")),
    }
    for name, (rs, exp) in groups.items():
        mad = [r["adaptive"]["3.0"]["recall"] for r in rs]
        fpr = [100 * r["adaptive"]["3.0"]["n_false_positive"] / (r["n_benign"] - r["n_poison"]) for r in rs]
        fx = [r["fixed_budget_recall"] for r in rs]
        auc = [r["spectral_auc"] for r in rs]
        for lbl, got, want in (
            ("MAD recall", f"{st.mean(mad):.4f}", exp[0]),
            ("MAD recall SD", f"{st.stdev(mad):.4f}", exp[1]),
            ("MAD FPR %", f"{st.mean(fpr):.2f}", exp[2]),
            ("fixed recall", f"{st.mean(fx):.4f}", exp[3]),
            ("fixed recall SD", f"{st.stdev(fx):.4f}", exp[4]),
            ("spectral AUC", f"{st.mean(auc):.4f}", exp[5]),
        ):
            check(f"cost {name}: {lbl}", got, want, "05_results.tex:241-243", checks)

    # --- Per-cost AUCs quoted in that table's caption (05_results.tex:232).
    for cost, want in ((1, "0.5016"), (2, "0.5024"), (4, "0.5029")):
        v = [r["spectral_auc"] for r in grid if r["cost"] == cost]
        check(f"cost {cost} mean spectral AUC", f"{st.mean(v):.4f}", want, "05_results.tex:232", checks)

    # --- Activation Clustering cost-16 mean over five rates (05_results.tex:142).
    ac16 = [r["ac_fixed_recall"] for r in grid if r["cost"] == 16]
    check("AC cost-16 fixed recall", f"{st.mean(ac16):.4f}", "0.6801", "05_results.tex:142", checks)

    # --- Budget-free rule on the CTU-13 window, and the anchor cell (06_discussion.tex:71-73).
    win = [r for r in grid if (r["rate"], r["cost"]) in CTU_WINDOW]
    anc = [r for r in grid if (r["rate"], r["cost"]) == ANCHOR]
    winfpr = [100 * r["adaptive"]["3.0"]["n_false_positive"] / (r["n_benign"] - r["n_poison"]) for r in win]
    check("window MAD FPR %", f"{st.mean(winfpr):.2f}", "6.58", "06_discussion.tex:71", checks)
    check("window mean precision",
          f"{st.mean(r['adaptive']['3.0']['n_true_positive'] / r['adaptive']['3.0']['n_flagged'] for r in win):.3f}",
          "0.105", "06_discussion.tex:71", checks)
    check("anchor mean precision",
          f"{st.mean(r['adaptive']['3.0']['n_true_positive'] / r['adaptive']['3.0']['n_flagged'] for r in anc):.4f}",
          "0.0714", "06_discussion.tex:71", checks)
    check("anchor flagged flows",
          f"{st.mean(r['adaptive']['3.0']['n_flagged'] for r in anc):,.0f}",
          "8,017", "06_discussion.tex:73", checks)

    # --- Activation Clustering on the UNSW-NB15 window (05_results.tex:359 and :410).
    acw = [r["ac"]["recall"] for r in sec if (r.get("rate"), r.get("cost")) in UNSW_WINDOW]
    check("AC UNSW window recall", f"{st.mean(acw):.4f}", "0.1443", "05_results.tex:359", checks)

    # --- Stated arithmetic identities, which have no disk value by construction.
    check("CTU-13 release total", f"{143046 + 55082:,}", "198,128", "04_methods.tex:49", checks)
    check("median poison rank", f"{1022 + 286:,}", "1,308", "05_results.tex:290", checks)

    # --- Rule selection on the full 25-cell grid (05_results.tex, the moment-statistic
    # paragraphs). Added 2026-09-04 when the full-grid re-run refuted the narrow-grid claim.
    # Every figure below is recomputed here rather than read from the file's own summary
    # fields, so a rerun that changes the answer fails this check instead of silently
    # agreeing with a stale sentence.
    rsel = json.load((RESULTS / "rule_selection_statistic.json").open())
    units = [u for u in rsel["units"]
             if _finite(u["S1_bimodality"]) and _finite(u["S2_top_gap"])]
    check("rule-selection units", f"{len(units)}", "625",
          "05_results.tex (moment statistic)", checks)

    y_all = [bool(u["mad_wins"]) for u in units]
    check("pooled S1 prediction AUC",
          f"{_auc(y_all, [u['S1_bimodality'] for u in units]):.4f}", "0.8196",
          "05_results.tex (moment statistic)", checks)
    check("pooled rate prediction AUC",
          f"{_auc(y_all, [u['rate'] for u in units]):.4f}", "0.7138",
          "05_results.tex (moment statistic)", checks)

    # Stratified by arm. These are the numbers that reverse the pooled ordering.
    for arm, n_mad, n_bud, s1_want, rate_want in (
            ("spectral", "80", "45", "0.5392", "0.9861"),
            ("isolation_forest", None, None, "0.5156", "0.9673"),
            ("boundary_departure", None, None, "0.6759", "0.8526")):
        sub = [u for u in units if u["detector"] == arm]
        ys = [bool(u["mad_wins"]) for u in sub]
        if n_mad is not None:
            check(f"{arm} MAD wins", f"{sum(ys)}", n_mad,
                  "05_results.tex (moment statistic)", checks)
            check(f"{arm} budget wins", f"{len(ys) - sum(ys)}", n_bud,
                  "05_results.tex (moment statistic)", checks)
        check(f"{arm} S1 AUC", f"{_auc(ys, [u['S1_bimodality'] for u in sub]):.4f}", s1_want,
              "05_results.tex (moment statistic)", checks)
        check(f"{arm} rate AUC", f"{_auc(ys, [u['rate'] for u in sub]):.4f}", rate_want,
              "05_results.tex (moment statistic)", checks)

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

    # --- Post-removal attack success, promoted to a contribution (01_introduction.tex).
    pr = json.load((RESULTS / "secondary_post_removal_asr.json").open())["per_cell"]
    post = {(c["rate"], c["cost"]): c["post_removal_asr"]["mean"] for c in pr}
    for cell, want in (((0.01, 8), "0.0546"), ((0.005, 16), "0.4310"), ((0.01, 16), "0.6154")):
        check(f"post-removal ASR {cell}", f"{post[cell]:.4f}", want,
              "01_introduction.tex (contributions)", checks)

    # --- Backdoor fractions across the corpus family, promoted to a contribution.
    q2 = json.load((RESULTS / "netflow_property_analysis.json").open())["q2_summary"]
    for corpus, want in (("NF-CSE-CIC-IDS2018-v2", "0.4465"),
                         ("NF-ToN-IoT-v2", "0.5964"),
                         ("NF-UNSW-NB15-v2", "0.5478")):
        check(f"backdoor fraction {corpus}",
              f"{q2[corpus]['mean_backdoor_fraction']:.4f}", want,
              "01_introduction.tex (contributions)", checks)

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
    check("STRIP CTU window FRR recall", f"{st.mean(r['frr_recall'] for r in ctu):.4f}", "0.8651",
          "05_results.tex (STRIP)", checks)
    check("STRIP CTU window FRR fpr", f"{st.mean(r['frr_fpr'] for r in ctu):.4f}", "0.0286",
          "05_results.tex (STRIP)", checks)
    check("STRIP CTU cost-16 FRR recall", f"{st.mean(r['frr_recall'] for r in ctu16):.4f}", "0.9989",
          "05_results.tex (STRIP)", checks)
    check("STRIP CTU cost-16 FRR fpr", f"{st.mean(r['frr_fpr'] for r in ctu16):.4f}", "0.0280",
          "05_results.tex (STRIP)", checks)
    check("STRIP CTU window budget", f"{st.mean(r['fixed_recall'] for r in ctu):.4f}", "0.2767",
          "05_results.tex (STRIP)", checks)
    check("STRIP CTU window MAD", f"{st.mean(r['mad']['3.0']['recall'] for r in ctu):.4f}", "0.0000",
          "05_results.tex (STRIP)", checks)
    check("STRIP UNSW FRR recall", f"{st.mean(r['frr_recall'] for r in unsw):.4f}", "0.0115",
          "05_results.tex (STRIP)", checks)
    check("STRIP UNSW ranking AUC", f"{st.mean(r['auc'] for r in unsw):.4f}", "0.6598",
          "05_results.tex (STRIP)", checks)
    check("STRIP CTU realized FRR",
          f"{st.mean(r['frr_calibration']['realized_frr'] for r in ctu):.4f}", "0.0230",
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

    # --- Why the budget-free cutoff cannot fire on STRIP (05_results.tex, the STRIP paragraph).
    bound = json.load((RESULTS / "strip_mad_bound.json").open())["summary"]
    check("STRIP score median", f"{bound['median']:.4f}", "-0.0558",
          "05_results.tex (STRIP)", checks)
    check("STRIP score MAD", f"{bound['mad']:.4f}", "0.0469", "05_results.tex (STRIP)", checks)
    check("STRIP cutoff at z=3.0", f"{bound['threshold']['3.0']:+.4f}", "+0.1530",
          "05_results.tex (STRIP)", checks)
    check("STRIP cutoff at z=2.0", f"{bound['threshold']['2.0']:+.4f}", "+0.0834",
          "05_results.tex (STRIP)", checks)
    check("STRIP thresholds all above max", f"{bound['all_thresholds_above_max']}", "True",
          "05_results.tex (STRIP)", checks)
    check("STRIP rows flagged by MAD", f"{bound['total_flagged']}", "0",
          "05_results.tex (STRIP)", checks)

    # --- The label-free capability check (06_discussion.tex, third recommendation).
    cap = json.load((RESULTS / "rule_capability_taxonomy.json").open())
    check("STRIP z_max", f"{cap['summary']['strip']['z_max_mean']:.2f}", "0.79",
          "06_discussion.tex (capability)", checks)
    check("lowest firing arm z_max",
          f"{min(v['z_max_mean'] for k, v in cap['summary'].items() if k != 'strip' and math.isfinite(v['z_max_mean'])):.1f}",
          "38.7", "06_discussion.tex (capability)", checks)
    check("capability agreement",
          f"{cap['prediction_agreement']['agreements']} of {cap['prediction_agreement']['checks']}",
          "75 of 75", "06_discussion.tex (capability)", checks)

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
    for ok, label, got, want, where in checks:
        if not ok:
            failed += 1
        print(f"  {'ok ' if ok else 'FAIL'}  {label:<{width}}  computed {got:>9s}  printed {want:>9s}   {where}")
    print(f"\n{len(checks) - failed} of {len(checks)} derived literals reproduce from the committed results")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
