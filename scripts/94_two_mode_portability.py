#!/usr/bin/env python3
"""Classify each detector's portability failure as calibration or structural.

Hypothesis H1 of notes/20260904-hypothesis-calibration-not-representation.md. The
manuscript currently reports "portability is a per-detector property", which is a
survey result. This tests a stronger and more useful claim: portability fails in
exactly two ways, and one measurement tells them apart.

  calibration failure  the statistic ported and a fitted constant did not.
                       Diagnostic: ranking AUC is high while the published rule's
                       recall is low. Remedy: refit the constant.
  structural failure   the rule's premise contradicts the substrate or the attack,
                       so no constant repairs it. Diagnostic: the statistic is
                       absent, undefined, or at chance. Remedy: another detector.

Every number is recomputed here from the committed result files rather than read
back from the manuscript, so this doubles as an independent check on the
tab:detectors literals.

Cells are the four pre-registered CTU-13 evasion-window cells, rate in
{0.005, 0.01} and cost in {8, 16}, matching results/analysis.json's evasion_window
block and scripts/19's list.
"""

import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import config

RESULTS = config.RESULTS
OUT = RESULTS / "two_mode_portability.json"

WINDOW = {(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)}
Z = "3.0"

# A statistic at or below this is not a ported signal, it is a coin flip.
CHANCE_CEILING = 0.70
# A published rule recovering less than this on a signal that ported is starved.
STARVED_CEILING = 0.50


def _mean(xs):
    return round(st.mean(xs), 4) if xs else None


def ctu_rows():
    d = json.load((RESULTS / "full_grid_mad_sweep.json").open())
    return [r for r in d["rows"] if (r["rate"], r["cost"]) in WINDOW]


def spectre_window_rows():
    """SPECTRE on the CTU-13 window, from its own pre-registered sweep."""
    d = json.load((RESULTS / "spectre_window.json").open())
    return [r for r in d["rows"] if (r["rate"], r["cost"]) in WINDOW]


def strip_window_rows():
    """STRIP on the CTU-13 window. Keyed by 'dataset|seed|rate|cost'."""
    d = json.load((RESULTS / "strip_detector.json").open())
    return [
        v for v in d["rows"].values()
        if v.get("dataset") == "ctu13" and (v["rate"], v["cost"]) in WINDOW
    ]


def classify(auc, rule_recall, *, statistic_exists, note, expects):
    """Two-mode verdict, plus a check that the hand-written note still fits the numbers.

    `note` is prose fixed when this script was written and `expects` is the mode it was
    written for. Numbers move on a re-run; prose does not. Returning them uncoupled would let
    a future run emit mode "structural" beside a note asserting the budget-free rule recovers
    the signal, which is how a stale sentence becomes a quoted claim. The mismatch is reported
    as data rather than resolved silently.
    """
    if not statistic_exists or auc is None:
        mode = "structural"
    elif auc <= CHANCE_CEILING:
        mode = "structural"
        note = f"statistic at or near chance ({auc}); {note}"
    elif rule_recall is not None and rule_recall < STARVED_CEILING:
        mode = "calibration"
    else:
        mode = "ported"
    return mode, note, (mode == expects)


def main() -> int:
    ctu = ctu_rows()
    detectors = []

    # --- Spectral Signatures, CTU-13 -------------------------------------
    auc = _mean([r["spectral_auc"] for r in ctu])
    fixed = _mean([r["fixed_budget_recall"] for r in ctu])
    mad = _mean([r["adaptive"][Z]["recall"] for r in ctu])
    mode, note, note_fits = classify(
        auc, fixed, statistic_exists=True, expects="calibration",
        note="budget-free MAD rule recovers the signal, which is what a "
             "calibration failure predicts",
    )
    detectors.append({
        "detector": "Spectral Signatures", "corpus": "CTU-13",
        "statistic": "top-k singular projection", "auc": auc,
        "published_rule_recall": fixed, "refit_rule_recall": mad,
        "mode": mode, "note": note, "note_fits_computed_mode": note_fits,
    })

    # --- Activation Clustering, CTU-13 -----------------------------------
    ac_fixed = _mean([r["ac_fixed_recall"] for r in ctu])
    mode, note, note_fits = classify(
        None, ac_fixed, statistic_exists=False, expects="structural",
        note="the published rule emits a partition, not a score, so there is no "
             "threshold to refit; a score has to be constructed first",
    )
    detectors.append({
        "detector": "Activation Clustering", "corpus": "CTU-13",
        "statistic": None, "auc": None,
        "published_rule_recall": ac_fixed, "refit_rule_recall": None,
        "mode": mode, "note": note, "note_fits_computed_mode": note_fits,
    })

    # --- SPECTRE, CTU-13 window ------------------------------------------
    sp = [r["spectre"] for r in spectre_window_rows()]
    auc = _mean([v["auc"] for v in sp])
    fixed = _mean([v["fixed_recall"] for v in sp])
    mad = _mean([v["mad_recall"][Z] for v in sp])
    mode, note, note_fits = classify(
        auc, fixed, statistic_exists=True, expects="calibration",
        note="budget-free rule recovers the signal, at a higher false-positive "
             "cost than Spectral pays",
    )
    detectors.append({
        "detector": "SPECTRE", "corpus": "CTU-13",
        "statistic": "robust-whitened top eigenvector", "auc": auc,
        "published_rule_recall": fixed, "refit_rule_recall": mad,
        "mode": mode, "note": note, "note_fits_computed_mode": note_fits,
    })

    # --- STRIP, CTU-13 window --------------------------------------------
    stv = [r["strip"] for r in strip_window_rows()]
    auc = _mean([v["auc"] for v in stv])
    fixed = _mean([v["fixed_recall"] for v in stv])
    mad = _mean([v["mad"][Z]["recall"] for v in stv])
    mode, note, note_fits = classify(
        auc, fixed, statistic_exists=True, expects="calibration",
        note="the statistic ports but the budget-free rule does not recover it "
             "either, flagging nothing at 0 false positives. A calibration "
             "failure that our own refit does not repair, which is the honest "
             "boundary of the repairable class.",
    )
    detectors.append({
        "detector": "STRIP", "corpus": "CTU-13",
        "statistic": "prediction entropy under superimposition", "auc": auc,
        "published_rule_recall": fixed, "refit_rule_recall": mad,
        "mode": mode, "note": note, "note_fits_computed_mode": note_fits,
    })

    # --- Neural Cleanse ---------------------------------------------------
    nc = json.load((RESULTS / "analysis.json").open()).get("neural_cleanse", {})
    detectors.append({
        "detector": "Neural Cleanse", "corpus": "CTU-13",
        "statistic": "MAD anomaly index over per-class mask norms",
        "auc": None, "published_rule_recall": None, "refit_rule_recall": None,
        "mode": "structural",
        "note": "the anomaly index is degenerate at two classes, a provable "
                "property of the median-absolute-deviation statistic over two "
                "values rather than a tuning failure",
        "source_keys": sorted(nc.keys())[:6],
    })

    # --- Density mitigation, HDBSCAN, on the loud control ----------------
    dm = json.load((RESULTS / "density_mitigation_ctu.control_bypass.composition.json").open())
    per_seed = []
    for c in dm["controls"]:
        comp = c["cluster_composition"]
        best = comp["best"]
        clusters = comp["clusters"]
        pure_benign_kept = sum(
            1 for cl in clusters
            if cl["label"] >= 0 and cl["n_poison"] == 0 and not cl["absorbed"]
        )
        per_seed.append({
            "seed": c["seed"],
            "auc": round(c["auc"], 4),
            "recall": c["fixed_recall"],
            "poison_cluster_size": best["size"],
            "poison_cluster_n_poison": best["n_poison"],
            "poison_cluster_purity": best["purity"],
            "poison_cluster_absorbed": best["absorbed"],
            "poison_in_noise": comp["poison_in_noise"],
            "pure_benign_clusters_kept": pure_benign_kept,
        })
    detectors.append({
        "detector": "density mitigation (HDBSCAN)", "corpus": "CTU-13 loud control",
        "statistic": "cluster membership",
        "auc": _mean([p["auc"] for p in per_seed]),
        "published_rule_recall": _mean([p["recall"] for p in per_seed]),
        "refit_rule_recall": None,
        "mode": "structural",
        "note": "the clustering separates the poison perfectly, so the failure is "
                "not the statistic. The rule absorbs the clusters a surrogate fits "
                "with lowest loss, and clean-label poison is by construction the "
                "easiest rows in the space, so the ordering is adverse and no "
                "constant repairs it. See H4.",
        "per_seed": per_seed,
    })

    modes = {}
    for d in detectors:
        modes.setdefault(d["mode"], []).append(d["detector"])

    payload = {
        "_schema": "two_mode_portability/1",
        "_instrument": "scripts/94_two_mode_portability.py",
        "_hypothesis": "notes/20260904-hypothesis-calibration-not-representation.md (H1)",
        "_window_cells": sorted(WINDOW),
        "_thresholds": {
            "chance_ceiling": CHANCE_CEILING,
            "starved_ceiling": STARVED_CEILING,
            "z": Z,
        },
        "_caveats": [
            "The density-mitigation arm failed its own pre-registered loud control "
            "and does not enter the paper as a defense. It is used here as evidence "
            "about decision rules, which is a different claim, and any manuscript "
            "text must keep that distinction explicit.",
            "The Activation Clustering score reaching AUC 0.9626 is our constructed "
            "gated statistic, not the published rule. Its presence is what makes AC "
            "a rule failure rather than a representation failure, but the published "
            "detector still has no score.",
            "All five window arms are the four CTU-13 cells at rate 0.005 and 0.01 "
            "by cost 8 and 16, five seeds each, but they come from four separate "
            "sweep files written by different scripts. Values are compared across "
            "code paths and a small disagreement between them is expected.",
        ],
        "by_mode": modes,
        "detectors": detectors,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'detector':30} {'corpus':22} {'AUC':>8} {'rule':>8} {'refit':>8}  mode")
    for d in detectors:
        f = lambda v: f"{v:.4f}" if isinstance(v, float) else "n/a"
        print(f"{d['detector']:30} {str(d['corpus']):22} {f(d['auc']):>8} "
              f"{f(d['published_rule_recall']):>8} {f(d['refit_rule_recall']):>8}  {d['mode']}")
    print()
    for m, names in sorted(modes.items()):
        print(f"{m:14} {len(names)}: {', '.join(names)}")
    print(f"\nwrote {OUT.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
