#!/usr/bin/env python3
"""Evaluate the arithmetic relations the prose asserts, and check float references resolve.

Two defect classes went into this build on 2026-08-21 and neither needed judgement to
catch:

  * a definitional sentence asserting that two components "sum to the reported rate",
    when they sum to the reported rate MINUS a third term. Every number in it was
    correct and the relation between them was not.
  * a dangling "Fig." left behind when a reference was deleted and the sentence was not
    re-read.

Prose review does not catch either. Both are mechanical. This checks them mechanically
rather than restating the discipline that failed to.

Part 1, REGISTERED RELATIONS. Relations the manuscript asserts, each written here as
code and evaluated against results/reframe_manifest.json. A registered relation that
stops holding is a hard failure. Adding a new claim to the paper means adding its
relation here.

Part 2, THE SWEEP. Every sentence carrying two or more numerals and an arithmetic
connective, listed for review. This cannot evaluate them all -- deciding what relation
an English sentence asserts is the judgement part -- but an unreviewed sentence of this
shape is precisely where the first defect lived, so the sweep makes the population
visible and countable rather than assumed.

Part 3, REFERENCES. Every \\ref resolves to a \\label that exists, and no float word is
left stranded without one.

Run:  python scripts/47_relation_and_reference_check.py [--show-sweep]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config

# Which manuscript to audit. This defaulted to paper/ and stayed there after
# paper_access/ became the live submission, so Parts 2 and 3 had never once run
# against the IEEE Access sections. Pass --paper paper_access for that manuscript.
# Part 1's registered relations read results/ and are manuscript-independent.
_PAPER = "paper"
for _i, _a in enumerate(sys.argv):
    if _a == "--paper" and _i + 1 < len(sys.argv):
        _PAPER = sys.argv[_i + 1]
SECTIONS = config.ROOT / _PAPER / "sections"
MAIN = config.ROOT / _PAPER / "main.tex"
TOL = 5e-4

CONNECTIVES = re.compile(
    r"\b(of|against|so|leaving|times|from|to|versus|vs|per|out of|sum to|"
    r"account for|rises?|falls?|drops?|recovers?|reaches?)\b", re.I)
NUMERAL = re.compile(r"(?<![\w.])-?\d+(?:[.,{}]\d+)*(?![\w])")


def registered_relations(man):
    d = man["dissociation_table_cost16"]
    m = man["mechanism"]
    out = []

    # The decomposition must close on both datasets and at both printed costs. This is the
    # relation the Methods sentence asserts, and the one that failed as originally written.
    # Table I gained the cost-8 column because cost 16 saturates both CTU-13 arms, which
    # makes its backdoor component of zero bounded by construction rather than measured; a
    # registered relation on the saturated column alone would never notice that.
    for cost, table in [(16, d), (8, man["dissociation_table_cost8"])]:
        for ds, p, s, q, e, b in [
            ("CTU-13", "ctu_clean_plain", "ctu_clean_stamped", "ctu_poisoned_stamped",
             "ctu_evasion_E", "ctu_backdoor_B"),
            ("UNSW-NB15", "unsw_clean_plain", "unsw_clean_stamped", "unsw_poisoned_stamped",
             "unsw_evasion_E", "unsw_backdoor_B"),
        ]:
            P, S, Q, E, B = table[p], table[s], table[q], table[e], table[b]
            out.append((f"{ds} c{cost}: evasion = clean_stamped - clean_plain", E, S - P))
            out.append((f"{ds} c{cost}: backdoor = poisoned_stamped - clean_stamped", B, Q - S))
            out.append((f"{ds} c{cost}: evasion + backdoor + clean_plain = reported rate",
                        E + B + P, Q))

    # The cost-8 column is the unsaturated one, and the prose says so. Assert that: neither
    # CTU-13 arm may sit at 1.0 there, or the sentence claiming both components are measured
    # is false and the paper is back to arguing from a ceiling.
    c8 = man["dissociation_table_cost8"]
    out.append(("CTU-13 c8: clean arm below ceiling", c8["ctu_clean_stamped"] < 1.0, True))
    out.append(("CTU-13 c8: poisoned arm below ceiling", c8["ctu_poisoned_stamped"] < 1.0, True))
    out.append(("CTU-13 c8: printed seed spread low", round(c8["ctu_clean_stamped_min"], 3), 0.113))
    out.append(("CTU-13 c8: printed seed spread high", c8["ctu_clean_stamped_max"], 1.0))

    # The uniform baselines the mechanism paragraph prints, and the multiples of them.
    out.append(("CTU-13: uniform baseline = 16/757", 16 / m["ctu_n_features"], 0.0211))
    out.append(("UNSW-NB15: uniform baseline = 16/38", 16 / m["unsw_n_features"], 0.4211))
    # The share-over-uniform multiples (26.8x and 2.1x) are NOT registered, because the
    # manuscript does not print them -- it prints each share against its own uniform
    # baseline and lets the reader see the contrast. They appear only in working notes.
    # A registry entry for a relation the paper does not assert tests the notes rather
    # than the artefact, and reports a failure the paper cannot cause.
    return out


def sentences(text):
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\(cite|ref|label|eqref)\{[^}]*\}", " CITE ", text)
    text = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", text, flags=re.S)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", text)
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = " ".join(s.split())
        if s:
            yield s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show-sweep", action="store_true")
    ap.add_argument("--paper", default="paper",
                    help="manuscript directory to audit (paper or paper_access)")
    args = ap.parse_args()

    man = json.loads((config.RESULTS / "reframe_manifest.json").read_text())
    failures = 0

    print("=== Part 1: registered relations")
    for label, got, want in registered_relations(man):
        ok = abs(got - want) <= max(TOL, abs(want) * 0.01)
        failures += 0 if ok else 1
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label:58s} {got:.4f} vs {want:.4f}")

    print("\n=== Part 2: sentences asserting a relation among their own numerals")
    flagged = []
    for f in sorted(SECTIONS.glob("*.tex")):
        for s in sentences(f.read_text()):
            nums = NUMERAL.findall(s)
            if len(nums) >= 2 and CONNECTIVES.search(s):
                flagged.append((f.name, s))
    print(f"  {len(flagged)} sentences carry two or more numerals and an arithmetic connective.")
    print("  These are the population where a correct-numbers/wrong-relation defect can hide.")
    if args.show_sweep:
        for name, s in flagged:
            print(f"    [{name[:14]:14s}] {s[:150]}")

    print("\n=== Part 3: float references")
    body = "".join(p.read_text() for p in list(SECTIONS.glob("*.tex")) + [MAIN])
    labels = set(re.findall(r"\\label\{([^}]*)\}", body))
    refs = set(re.findall(r"\\ref\{([^}]*)\}", body))
    dangling = sorted(refs - labels)
    unused = sorted(l for l in labels - refs if l.startswith(("fig:", "tab:")))
    # A float word not followed by a reference is the fingerprint of a deleted \ref.
    orphans = []
    for f in sorted(SECTIONS.glob("*.tex")):
        raw = f.read_text()
        for m in re.finditer(r"\b(Fig|Figure|Table|Tables)\.?~?(?!\\ref)(?!s?\b\s*\\ref)([^\s]{0,12})", raw):
            tail = m.group(2)
            if not tail.startswith("\\ref") and not tail.startswith("~\\ref"):
                ctx = " ".join(raw[max(0, m.start() - 60):m.start() + 60].split())
                orphans.append((f.name, ctx))
    print(f"  dangling refs (no matching label): {dangling or 'none'}")
    print(f"  labelled floats never referenced : {unused or 'none'}")
    print(f"  float words not followed by a reference: {len(orphans)}")
    for name, ctx in orphans[:6]:
        print(f"    [{name[:14]:14s}] ...{ctx}...")
    failures += len(dangling) + len(unused) + len(orphans)

    print(f"\n{'ALL CHECKS PASS' if failures == 0 else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
