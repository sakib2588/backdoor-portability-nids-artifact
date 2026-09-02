#!/usr/bin/env python3
r"""Place this paper's experimental practice against the 50-paper topic corpus.

Aggregator only -- mirrors the reference NDSS-corpus instrument's own script 83: it does
NOT extract anything from the papers itself. It reads per-paper records from
results/experimental_survey/<paper_id>.json, one per corpus paper plus
OURS_backdoor_portability_nids.json, each hand/agent-filled against the paper's own text
(one general-purpose agent per corpus paper, this session, reading the marker .md and
writing a record; "ours" filled directly from this paper's own results/methods sections).

Two rules carried over from the reference instrument, both load-bearing:
  * Ours is NEVER a member of its own band.
  * A field a paper never states is null, and null is NOT zero or one. n_seeds in
    particular: many reference papers state no repetition count at all -- scoring silence
    as 1 would invent a number and drag the median toward a value nothing supports.

Contract: ASSERT ONLY, NEVER GENERATE. Exit 0 read cleanly, 1 a record failed its schema,
2 could not read the records at all. Exit 2 is never a pass.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
OURS_ID = "OURS_backdoor_portability_nids"

NUMERIC = [
    ("n_models", "models evaluated"),
    ("n_datasets", "datasets / corpora"),
    ("n_seeds", "seeds or repetitions"),
    ("n_eval_types", "distinct evaluation types"),
    ("n_baselines", "baselines compared"),
]
CATEGORICAL = [
    ("reports_variability", "reports any variability", lambda v: v not in (None, "none")),
    ("significance_test", "names a significance test", lambda v: v not in (None, "none")),
    ("per_seed_values_given", "gives per-seed values", bool),
    ("control_arm", "runs a control arm (any kind)", lambda v: v not in (None, "none")),
]
REQUIRED = {"paper_id", "subcorpus", "n_models", "n_datasets", "n_seeds", "n_eval_types",
            "n_baselines", "reports_variability", "significance_test",
            "per_seed_values_given", "control_arm", "evidence"}


def band(vals: list[float]) -> dict:
    s = sorted(vals)
    return {"n": len(s), "min": s[0], "median": st.median(s), "max": s[-1]}


def percentile_of(x: float, vals: list[float]) -> int:
    s = sorted(vals)
    return round(100 * sum(1 for v in s if v <= x) / len(s))


def load(d: pathlib.Path) -> tuple[dict, list[str]]:
    recs, bad = {}, []
    for f in sorted(d.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            bad.append(f"{f.name}: not valid JSON ({e})")
            continue
        missing = REQUIRED - set(r)
        if missing:
            bad.append(f"{f.name}: missing fields {sorted(missing)}")
            continue
        recs[r["paper_id"]] = r
    return recs, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/experimental_survey")
    ap.add_argument("--expect", type=int, default=51)
    ap.add_argument("--manifest", default="results/corpus_experimental_practice.json")
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    d = REPO / args.records
    if not d.exists():
        print(f"no records directory {d}")
        print("RESULT: UNRESOLVED")
        return 2

    recs, bad = load(d)
    if bad:
        for b in bad:
            print(f"  BAD RECORD  {b}")
        print("RESULT: UNRESOLVED")
        return 2
    if OURS_ID not in recs:
        print(f"our own record {OURS_ID}.json is absent; nothing to place")
        print("RESULT: UNRESOLVED")
        return 2

    ours = recs[OURS_ID]
    ref = {k: v for k, v in recs.items() if k != OURS_ID}

    print("=== EXPERIMENTAL-PRACTICE BANDS, 50-PAPER TOPIC CORPUS ===")
    print(f"  records: {len(recs)} ({len(ref)} reference + ours)")
    if len(recs) != args.expect:
        print(f"  WARNING: expected {args.expect} records, have {len(recs)} -- "
              f"the band is over whatever is present, which may not be the full corpus")
    print("  ours is excluded from its own band")
    print("  TOPIC-matched corpus, not venue-matched -- see scripts/30's caveat\n")

    print(f"  {'field':30}{'ours':>7}{'min':>6}{'med':>7}{'max':>6}{'pct':>6}   stated by")
    out_rows = {}
    for key, label in NUMERIC:
        vals = [r[key] for r in ref.values() if r.get(key) is not None]
        o = ours.get(key)
        stated = f"{len(vals)}/{len(ref)}"
        if not vals:
            print(f"  {label:30}{'--':>7}{'--':>6}{'--':>7}{'--':>6}{'--':>6}   {stated}")
            continue
        b = band(vals)
        pct = percentile_of(o, vals) if o is not None else None
        o_s = "null" if o is None else f"{o:g}"
        pct_s = "--" if pct is None else str(pct)
        print(f"  {label:30}{o_s:>7}{b['min']:>6g}{b['median']:>7g}{b['max']:>6g}"
              f"{pct_s:>6}   {stated}")
        out_rows[key] = {**b, "ours": o, "percentile": pct, "n_stated": len(vals)}

    print(f"\n  {'convention':32}{'ours':>7}   corpus")
    for key, label, test in CATEGORICAL:
        n = sum(1 for r in ref.values() if test(r.get(key)))
        o = "yes" if test(ours.get(key)) else "no"
        print(f"  {label:32}{o:>7}   {n}/{len(ref)}")
        out_rows[key] = {"ours": test(ours.get(key)), "n_corpus": n, "of": len(ref)}

    silent = sorted(k for k, r in ref.items() if r.get("n_seeds") is None)
    print(f"\n  -- {len(silent)} of {len(ref)} reference papers state NO repetition count --")
    print(f"     (never imputed as 1 in the band above)")

    surveys = sorted(k for k, r in ref.items()
                      if all(r.get(f) is None for f in ("n_models", "n_datasets", "n_eval_types")))
    if surveys:
        print(f"\n  -- {len(surveys)} pure-survey/position papers, no original experiments --")
        for s in surveys:
            print(f"     {s}")

    if args.write_manifest:
        p = REPO / args.manifest
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "_schema": "corpus_experimental_practice",
            "_generated_by": "scripts/32_experimental_practice_bands.py",
            "_corpus_caveat": (
                "50 papers, topic-matched, not venue-matched. Read as \"how this "
                "literature experiments\", never as \"the ICCIT norm\"."
            ),
            "_n_records": len(recs),
            "_n_reference": len(ref),
            "_silent_on_seeds": silent,
            "_pure_survey_papers": surveys,
            "ours": ours,
            "records": recs,
            "bands": out_rows,
        }, indent=1))
        print(f"\n  wrote {p}")
    else:
        print(f"\n  (manifest NOT written; pass --write-manifest to write {args.manifest})")

    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
