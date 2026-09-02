"""Read-only aggregator: every number the IEEE Access review response needs, derived from committed
results in one place, so each corrected literal in paper_access/ traces to a single artefact.

This manuscript has NO LaTeX macro layer -- every number is hand-typed into the .tex. That is exactly
how the defects this script exists to fix got in. Rather than introduce a macro layer mid-revision,
this produces results/review_response_aggregates.json as the one file a reviewer (or a future audit)
can diff the manuscript against.

Nothing here trains or re-runs anything. It only reads, recomputes, and cross-checks.

Sections produced:
  full_grid            -- the real 25-cell CTU-13 grid at per-(rate, cost) resolution, with attack
                          success, both decision rules, and the ranking metric. Table 6 currently
                          marginalises rate away and the manuscript then calls it "the full 25-cell
                          grid"; this is the grid that claim needs.
  window               -- the four CTU-13 window cells, aggregated the way the paper aggregates them,
                          including the Activation Clustering value the manuscript prints as 0.0000.
  ac_trials            -- the per-trial Activation Clustering recount behind the "12 of 20 trials"
                          sentence, whose arithmetic does not close as written.
  multiplier           -- the budget-multiplier sweep including the multiplier-2 row that Table 7
                          prints as a dash, and the half-recall crossing that row actually implies.
  strip                -- STRIP's rule-versus-score split on vision and on both tabular corpora.
  secondary            -- the UNSW-NB15 detector columns that exist on disk but appear nowhere in
                          the manuscript.
  fpr_convention       -- an audit of the false-positive-rate denominator, which the manuscript never
                          defines and which is NOT consistent across the codebase (see below).

FPR denominator, the thing to know. Two conventions exist in this repository:
  (a) false positives / (benign rows - poison rows)  -- clean rows only
  (b) false positives / benign rows                  -- clean rows plus poison
`scripts/63_full_grid_mad_sweep.py` STORES (b) in its `adaptive[z].fpr` field. The manuscript's
reported 6.58% window false-positive price is (a); (b) gives 6.53% on the same rows. The difference
is small but it is real, and anyone recomputing the paper's figure from the stored field gets a
different number. This script computes (a) -- matching the manuscript and matching the `n_clean`
convention already used in scripts/53 and scripts/62 -- and reports both so the gap is on the record
rather than rediscovered by a reviewer.

Run:  .venv/bin/python scripts/73_review_response_aggregates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

CTU_WINDOW_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]
WINDOW_ASR_CRITERION = 0.8
Z_MAIN = "3.0"

OUT = config.RESULTS / "review_response_aggregates.json"


def load(name):
    path = config.RESULTS / name
    if not path.exists():
        print(f"  (absent: {name})")
        return None
    return json.loads(path.read_text())


def stat(vals):
    """Mean, sample SD and n. The manuscript's stated convention is means with standard deviations;
    Tables 6, 7 and 8 currently print no dispersion at all, which is what hides the bimodality."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return dict(mean=float(np.mean(vals)),
                sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                min=float(np.min(vals)), max=float(np.max(vals)), n=len(vals))


def build_full_grid(detectors, mad_sweep):
    """The 25-cell grid at per-(rate, cost) resolution, joining attack success from the detector
    sweep to the two decision rules from the MAD sweep."""
    by_cell = {}
    for r in detectors["main_grid"]:
        by_cell.setdefault((r["rate"], r["cost"]), {}).setdefault("asr", []).append(r["asr"])
        by_cell[(r["rate"], r["cost"])].setdefault("spectral_recall", []).append(
            r["spectral"]["recall"])
        by_cell[(r["rate"], r["cost"])].setdefault("spectral_auc", []).append(r["spectral"]["auc"])
        by_cell[(r["rate"], r["cost"])].setdefault("ac_recall", []).append(r["ac"]["recall"])
        by_cell[(r["rate"], r["cost"])].setdefault("clean_acc", []).append(r["clean_acc"])

    mad_by_cell = {}
    if mad_sweep:
        for r in mad_sweep["rows"]:
            k = (r["rate"], r["cost"])
            d = mad_by_cell.setdefault(k, {})
            d.setdefault("fixed_budget_recall", []).append(r["fixed_budget_recall"])
            d.setdefault("ac_fixed_recall", []).append(r["ac_fixed_recall"])
            d.setdefault("spectral_auc", []).append(r["spectral_auc"])
            a = r["adaptive"][Z_MAIN]
            d.setdefault("mad_recall", []).append(a["recall"])
            n_clean = r["n_benign"] - r["n_poison"]
            d.setdefault("mad_fpr_clean_denom", []).append(a["n_false_positive"] / n_clean)
            d.setdefault("mad_fpr_stored", []).append(a["fpr"])

    cells = []
    for (rate, cost) in sorted(by_cell):
        v = by_cell[(rate, cost)]
        m = mad_by_cell.get((rate, cost), {})
        asr = stat(v["asr"])
        cells.append(dict(
            rate=rate, cost=cost,
            in_window=bool((rate, cost) in [tuple(c) for c in CTU_WINDOW_CELLS]),
            meets_asr_criterion=bool(asr["mean"] >= WINDOW_ASR_CRITERION),
            asr=asr, clean_acc=stat(v["clean_acc"]),
            spectral_recall=stat(v["spectral_recall"]),
            spectral_auc=stat(v["spectral_auc"]),
            ac_recall=stat(v["ac_recall"]),
            mad_fixed_budget_recall=stat(m.get("fixed_budget_recall", [])),
            mad_ac_fixed_recall=stat(m.get("ac_fixed_recall", [])),
            mad_recall_z3=stat(m.get("mad_recall", [])),
            mad_fpr_z3_clean_denom=stat(m.get("mad_fpr_clean_denom", [])),
            mad_fpr_z3_stored=stat(m.get("mad_fpr_stored", []))))
    return cells


def build_window(mad_sweep):
    """The four window cells under the paper's own aggregation. Reproduces Spectral's printed 0.1425
    exactly, which is what licenses correcting Activation Clustering's printed 0.0000 to the value
    the same convention gives."""
    if not mad_sweep:
        return None
    win = [r for r in mad_sweep["rows"]
           if (r["rate"], r["cost"]) in [tuple(c) for c in CTU_WINDOW_CELLS]]
    return dict(
        n_rows=len(win),
        spectral_fixed_recall=stat([r["fixed_budget_recall"] for r in win]),
        ac_fixed_recall=stat([r["ac_fixed_recall"] for r in win]),
        spectral_auc=stat([r["spectral_auc"] for r in win]),
        mad_recall_z3=stat([r["adaptive"][Z_MAIN]["recall"] for r in win]),
        mad_fpr_z3_clean_denom=stat(
            [r["adaptive"][Z_MAIN]["n_false_positive"] / (r["n_benign"] - r["n_poison"])
             for r in win]),
        mad_fpr_z3_stored=stat([r["adaptive"][Z_MAIN]["fpr"] for r in win]),
        per_cell={f"{r['rate']}|{r['cost']}|{r['seed']}": dict(
            spectral_fixed_recall=r["fixed_budget_recall"], ac_fixed_recall=r["ac_fixed_recall"])
            for r in win})


def build_ac_trials(detectors, mad_sweep):
    """Recount behind the manuscript's 'the minority cluster holds no poison in 12 of 20 trials,
    recovering it fully only in the two highest-poison cells'. As written that is 12 + 10 > 20."""
    out = {}
    for label, rows, getter in (
        ("detector_sweep_ac_recall", detectors["main_grid"], lambda r: r["ac"]["recall"]),
        ("mad_sweep_ac_fixed_recall",
         mad_sweep["rows"] if mad_sweep else [], lambda r: r["ac_fixed_recall"]),
    ):
        win = [r for r in rows if (r["rate"], r["cost"]) in [tuple(c) for c in CTU_WINDOW_CELLS]]
        if not win:
            continue
        vals = [getter(r) for r in win]
        zero = [r for r, v in zip(win, vals) if v == 0.0]
        full = [r for r, v in zip(win, vals) if v == 1.0]
        out[label] = dict(
            n_trials=len(win),
            n_zero=len(zero), n_full=len(full),
            n_partial=len(win) - len(zero) - len(full),
            zero_cells=sorted({f"{r['rate']}|{r['cost']}" for r in zero}),
            full_cells=sorted({f"{r['rate']}|{r['cost']}" for r in full}),
            mean=float(np.mean(vals)))
    # The full grid too: the manuscript's "two highest-poison cells" claim is checkable only there.
    if mad_sweep:
        full_grid_full = sorted({f"{r['rate']}|{r['cost']}" for r in mad_sweep["rows"]
                                 if r["ac_fixed_recall"] == 1.0})
        out["full_grid_cells_with_any_full_recovery"] = full_grid_full
    return out


def build_multiplier():
    """Table 7's dash at multiplier 2 is not missing data, and the half-recall crossing the
    manuscript predicts is contradicted by the sweep that is already on disk."""
    out = {}
    for label, fname in (("anchor_r0.005_c16", "spectral_budget_multiplier_sweep.json"),
                         ("r0.02_c16", "spectral_budget_multiplier_sweep_r0.02_c16.json"),
                         ("r0.05_c16", "spectral_budget_multiplier_sweep_r0.05_c16.json")):
        blob = load(fname)
        if not blob:
            continue
        rows = {}
        for m in blob["multipliers"]:
            per_seed = [ps["per_multiplier"][i]
                        for ps in blob["per_seed"]
                        for i, pm in enumerate(ps["per_multiplier"]) if pm["multiplier"] == m]
            rows[str(m)] = dict(
                budget=per_seed[0]["k_b"] if per_seed else None,
                recall=stat([p["recall"] for p in per_seed]))
        # Where does recall first cross 0.5? The manuscript asserts between multipliers 1.5 and 2.
        crossing = None
        ms = sorted(blob["multipliers"], key=float)
        for lo, hi in zip(ms, ms[1:]):
            r_lo, r_hi = rows[str(lo)]["recall"], rows[str(hi)]["recall"]
            if r_lo and r_hi and r_lo["mean"] < 0.5 <= r_hi["mean"]:
                crossing = dict(between_multipliers=[lo, hi],
                                between_budgets=[rows[str(lo)]["budget"], rows[str(hi)]["budget"]],
                                recall_below=r_lo["mean"], recall_above=r_hi["mean"])
                break
        out[label] = dict(by_multiplier=rows, half_recall_crossing=crossing)
    return out


def build_strip():
    """STRIP's null is rule-specific, and the vision control proves it is not a portability failure:
    on MNIST, where STRIP reaches recall 0.9996 at AUC 0.9984, the MAD rule still returns exactly
    0.0 at every threshold. A rule that returns zero where the detector provably works cannot be
    evidence about tabular data."""
    out = {}
    vis = load("strip_vision_control.json")
    if vis:
        out["vision_control"] = dict(
            recall=stat([r["strip_recall"] for r in vis["per_seed"]]),
            auc=stat([r["strip_auc"] for r in vis["per_seed"]]),
            mad_recall={z: stat([r["strip_mad_recall"][z] for r in vis["per_seed"]])
                        for z in ("2.0", "2.5", "3.0")})
    det = load("strip_detector.json")
    if det:
        for ds in ("ctu13", "unsw_nb15"):
            rows = [v for k, v in det["rows"].items() if k.startswith(f"{ds}|")]
            if not rows:
                continue
            out[ds] = dict(
                n_rows=len(rows),
                cells=sorted({f"{r['rate']}|{r['cost']}" for r in rows}),
                fixed_recall=stat([r["strip"]["fixed_recall"] for r in rows]),
                auc=stat([r["strip"]["auc"] for r in rows]),
                mad_recall={z: stat([r["strip"]["mad"][z]["recall"] for r in rows])
                            for z in ("2.0", "2.5", "3.0")},
                spectral_recall_same_rows=stat([r["spectral"]["recall"] for r in rows]))
    return out


def build_secondary():
    """UNSW-NB15 detector results that are on disk and absent from the manuscript. This is the
    material that answers the review's largest structural charge."""
    blob = load("secondary_detectors.json")
    if not blob:
        return None
    grid = blob["main_grid"]
    by_cell = {}
    for r in grid:
        by_cell.setdefault((r["rate"], r["cost"]), []).append(r)
    cells = []
    for (rate, cost) in sorted(by_cell):
        rows = by_cell[(rate, cost)]
        cells.append(dict(
            rate=rate, cost=cost, n_seeds=len(rows),
            asr=stat([r["asr"] for r in rows]),
            spectral_recall=stat([r["spectral"]["recall"] for r in rows]),
            spectral_auc=stat([r["spectral"]["auc"] for r in rows]),
            spectral_mad_recall=stat([r["spectral"]["mad_recall"] for r in rows]),
            ac_recall=stat([r["ac"]["recall"] for r in rows]),
            ac_silhouette=stat([r["ac"]["silhouette"] for r in rows]),
            ac_purity=stat([r["ac"]["purity"] for r in rows])))
    nc = blob.get("nc_boundary", [])
    return dict(
        overall=dict(
            spectral_recall=stat([r["spectral"]["recall"] for r in grid]),
            spectral_auc=stat([r["spectral"]["auc"] for r in grid]),
            spectral_mad_recall=stat([r["spectral"]["mad_recall"] for r in grid]),
            ac_recall=stat([r["ac"]["recall"] for r in grid]),
            n_rows=len(grid)),
        per_cell=cells,
        nc_boundary=dict(
            n_rows=len(nc),
            cells=sorted({f"{r['rate']}|{r['cost']}" for r in nc}),
            nc_ratio=stat([r["nc_ratio"] for r in nc if not r.get("nc_ratio_saturated")]),
            nc_clean_ratio=stat([r["nc_clean_ratio"] for r in nc
                                 if not r.get("nc_clean_ratio_saturated")]),
            n_flagged_target=sum(1 for r in nc if r["nc_flagged_target"]),
            n_discriminates=sum(1 for r in nc
                                if r["nc_flagged_target"]
                                and r["nc_ratio"] is not None
                                and r["nc_clean_ratio"] is not None
                                and r["nc_ratio"] > r["nc_clean_ratio"])) if nc else None)


def main() -> int:
    print("reading committed results (read-only)...")
    detectors = load("detectors.json")
    mad_sweep = load("full_grid_mad_sweep.json")
    if detectors is None:
        raise SystemExit("results/detectors.json is required")

    full_grid = build_full_grid(detectors, mad_sweep)
    window = build_window(mad_sweep)

    out = dict(
        provenance=dict(
            sources=["detectors.json", "full_grid_mad_sweep.json",
                     "spectral_budget_multiplier_sweep*.json", "strip_vision_control.json",
                     "strip_detector.json", "secondary_detectors.json"],
            ctu_window_cells=[list(c) for c in CTU_WINDOW_CELLS],
            window_asr_criterion=WINDOW_ASR_CRITERION,
            mad_z=Z_MAIN,
            fpr_denominator="false positives / (benign rows - poison rows); see module docstring"),
        full_grid=full_grid,
        window=window,
        ac_trials=build_ac_trials(detectors, mad_sweep),
        multiplier=build_multiplier(),
        strip=build_strip(),
        secondary=build_secondary(),
        asr_by_cost={
            str(c): stat([r["asr"] for r in detectors["main_grid"] if r["cost"] == c])
            for c in sorted({r["cost"] for r in detectors["main_grid"]})},
    )
    out["fpr_convention"] = dict(
        note="stored field uses benign rows; manuscript and scripts 53/62 use clean rows only",
        window_clean_denominator=window["mad_fpr_z3_clean_denom"]["mean"] if window else None,
        window_stored_denominator=window["mad_fpr_z3_stored"]["mean"] if window else None,
        cells_where_they_differ=sum(
            1 for c in full_grid
            if c["mad_fpr_z3_clean_denom"] and c["mad_fpr_z3_stored"]
            and abs(c["mad_fpr_z3_clean_denom"]["mean"] - c["mad_fpr_z3_stored"]["mean"]) > 1e-9))

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")

    w = out["window"]
    print("\n--- CTU-13 window (the four post-hoc cells) ---")
    print(f"  Spectral fixed recall : {w['spectral_fixed_recall']['mean']:.4f} "
          f"(SD {w['spectral_fixed_recall']['sd']:.4f})   manuscript prints 0.1425")
    print(f"  AC fixed recall       : {w['ac_fixed_recall']['mean']:.4f} "
          f"(SD {w['ac_fixed_recall']['sd']:.4f})   manuscript prints 0.0000  <-- CORRECTION")
    print(f"  MAD z3 recall         : {w['mad_recall_z3']['mean']:.4f}")
    print(f"  MAD z3 FPR (clean)    : {w['mad_fpr_z3_clean_denom']['mean']*100:.2f}%  "
          f"manuscript prints 6.58%")
    print(f"  MAD z3 FPR (stored)   : {w['mad_fpr_z3_stored']['mean']*100:.2f}%  "
          f"<-- different denominator, see docstring")

    print("\n--- attack success by trigger cost (all rates) ---")
    for c, s in out["asr_by_cost"].items():
        flag = "" if s["mean"] >= WINDOW_ASR_CRITERION else "   below the 0.8 window criterion"
        print(f"  cost {c:>2}: {s['mean']:.4f} (SD {s['sd']:.4f}){flag}")

    print("\n--- cells meeting the window attack-success criterion ---")
    meet = [f"{c['rate']}/{c['cost']}" for c in full_grid if c["meets_asr_criterion"]]
    print(f"  {len(meet)} of {len(full_grid)}: {', '.join(meet)}")

    mult = out["multiplier"].get("anchor_r0.005_c16")
    if mult:
        print("\n--- budget multiplier at the anchor cell ---")
        for m, r in mult["by_multiplier"].items():
            if r["recall"]:
                print(f"  multiplier {m:>4}  budget {r['budget']:>6}  "
                      f"recall {r['recall']['mean']:.4f}")
        cx = mult["half_recall_crossing"]
        if cx:
            print(f"  half-recall crossing lies between multipliers {cx['between_multipliers']} "
                  f"(budgets {cx['between_budgets']})")
            print("  the manuscript asserts between 1.5 and 2 (budgets 858 and 1144)")

    st = out["strip"]
    if "vision_control" in st:
        v = st["vision_control"]
        print("\n--- STRIP: the rule, not the corpus ---")
        print(f"  vision control recall {v['recall']['mean']:.4f} at AUC {v['auc']['mean']:.4f}, "
              f"but MAD z3 recall {v['mad_recall']['3.0']['mean']:.4f}")
    sec = out["secondary"]
    if sec:
        o = sec["overall"]
        print("\n--- UNSW-NB15 detector results present on disk, absent from the manuscript ---")
        print(f"  {o['n_rows']} rows: Spectral recall {o['spectral_recall']['mean']:.4f} "
              f"(AUC {o['spectral_auc']['mean']:.4f}, MAD {o['spectral_mad_recall']['mean']:.4f}), "
              f"AC recall {o['ac_recall']['mean']:.4f}")
        if sec["nc_boundary"]:
            n = sec["nc_boundary"]
            print(f"  NC boundary: {n['n_rows']} rows, flagged target {n['n_flagged_target']}, "
                  f"discriminates {n['n_discriminates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
