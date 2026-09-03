#!/usr/bin/env python3
"""Per-section word budget of paper_access/ against the IEEE Access venue band.

scripts/75 measures the manuscript against 18 published IEEE Access NIDS papers at the
paper level (words, sentences, citations per 1k, pages, figures, tables). It says nothing
about whether any one SECTION is long or short for this venue, which is the question a
reviewer's "the related work is thin" or "the results section is bloated" actually asks.
This answers it the same way the sibling ML_Paper project's Section 2 table does: a
band per section role over the corpus papers that carry that role, ours excluded.

Corpus: references/venue_band_md/, the same 18-paper set scripts/75 uses. Sections are
recovered from marker's markdown by the IEEE numbering convention ("I. INTRODUCTION",
"II. RELATED WORK", ...). Marker emits those lines sometimes as `#` headings and
sometimes as bare bold lines, so both forms are accepted, and only a monotonically
increasing Roman-numeral sequence starting at I is trusted as the section skeleton --
anything out of sequence (a mis-rendered list item like "I. MODBUS/TCP") is discarded.

Role mapping is a keyword rule over the heading text, applied in a fixed priority order.
It is a reading judgement encoded as code, not a measurement, so every assignment is
written to the output and printed, and a paper whose skeleton has fewer than three
sections is listed as unmeasurable rather than scored.

Ours: each paper_access/sections/*.tex file is one role, stripped by src/tex_prose's
_strip_tex (floats, math, commands removed) so both sides count prose only. Two caveats
belong with any number this emits. First, ours merges Discussion and Conclusion into one
section, so it is compared against the corpus's discussion+conclusion(+limitations) tail
summed per paper, with the separate discussion and conclusion bands shown for context.
Second, _strip_tex drops the remainder of any line carrying a \\ref or \\label, so ours
is a slight undercount on lines that cite a table or figure mid-sentence; the paper-level
instrument has the same property, so the two are consistent with each other.

ASSERT ONLY, NEVER GENERATE: writes results/venue_band_sections.json and prints a table.
It does not touch the manuscript.

Run:  python3 scripts/89_venue_band_sections.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import config
from src.prose_metrics import CITE_TOKEN, band, percentile_of, words_in
from src.reference_corpus import _REFERENCES_HEAD_RE, _split_paragraphs, _strip_markup
from src.tex_prose import _strip_tex

CORPUS = config.ROOT / "references" / "venue_band_md"
PAPER = config.ROOT / "paper_access"
OUT = config.RESULTS / "venue_band_sections.json"

OURS = [
    ("introduction", "sections/01_introduction.tex"),
    ("related_work", "sections/02_related_work.tex"),
    ("threat_model", "sections/03_threat_model.tex"),
    ("methods", "sections/04_methods.tex"),
    ("results", "sections/05_results.tex"),
    ("discussion_conclusion", "sections/06_discussion.tex"),
]

ROLES = ["introduction", "related_work", "threat_model", "methods", "results",
         "discussion", "conclusion", "limitations"]

# A section heading line: optional markdown hashes, optional bold, Roman numeral, dot,
# title. Titles in this corpus are upper-case; the character class is permissive so a
# title-case heading is not silently lost.
_HEAD_RE = re.compile(
    r"^(?:#{1,6}\s*)?\**\s*([IVX]{1,6})\.\s*\**\s*([A-Za-z][^\n]{2,110}?)\s*\**\s*$",
    re.MULTILINE,
)
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
_ROMAN = {"I": 1, "V": 5, "X": 10}


def roman_to_int(s: str) -> int:
    total, prev = 0, 0
    for ch in reversed(s):
        v = _ROMAN.get(ch, 0)
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


def role_of(title: str) -> str:
    t = title.upper()
    if "CONCLUSION" in t or t.startswith("SUMMARY"):
        return "conclusion"
    if "LIMITATION" in t:
        return "limitations"
    if t.startswith("DISCUSSION"):
        return "discussion"
    if any(k in t for k in ("RESULT", "EXPERIMENT", "EVALUATION", "PERFORMANCE",
                            "BENCHMARK", "EFFICACY", "TESTING", "FINDINGS")):
        return "results"
    if "DISCUSSION" in t:
        return "discussion"
    if any(k in t for k in ("RELATED", "BACKGROUND", "LITERATURE", "PRELIMINAR",
                            "STATE OF THE ART", "REVIEW OF")):
        return "related_work"
    if "THREAT" in t or "ATTACK MODEL" in t or "ADVERSARY" in t:
        return "threat_model"
    if "INTRODUCTION" in t:
        return "introduction"
    return "methods"


def section_skeleton(body: str) -> list[tuple[int, str, int]]:
    """(numeral, title, char offset) for the trusted, increasing Roman-numeral sequence."""
    out, expect = [], 1
    for m in _HEAD_RE.finditer(body):
        n = roman_to_int(m.group(1))
        if n == expect:
            out.append((n, m.group(2).strip(), m.start()))
            expect += 1
    return out


def span_words(span: str) -> int:
    no_heads = _HEADING_LINE_RE.sub("", span)
    return sum(len(words_in(p.replace(CITE_TOKEN, ""))) for p in _split_paragraphs(no_heads))


def measure_corpus_paper(md: pathlib.Path) -> dict:
    cleaned = _strip_markup(md.read_text(encoding="utf-8", errors="replace"))
    refs_m = _REFERENCES_HEAD_RE.search(cleaned)
    body = cleaned[: refs_m.start()] if refs_m else cleaned
    skel = section_skeleton(body)
    if len(skel) < 3:
        raise ValueError(f"only {len(skel)} trusted section headings")
    roles: dict[str, int] = {}
    assignments = []
    for i, (n, title, start) in enumerate(skel):
        end = skel[i + 1][2] if i + 1 < len(skel) else len(body)
        w = span_words(body[start:end])
        r = role_of(title)
        roles[r] = roles.get(r, 0) + w
        assignments.append({"numeral": n, "title": title, "role": r, "words": w})
    tail = sum(roles.get(r, 0) for r in ("discussion", "conclusion", "limitations"))
    if tail:
        roles["discussion_conclusion"] = tail
    return {"roles": roles, "assignments": assignments}


def measure_ours() -> dict:
    out = {}
    for role, rel in OURS:
        tex = (PAPER / rel).read_text(encoding="utf-8")
        plain = _strip_tex(tex).replace(CITE_TOKEN, "")
        paras = [b for b in re.split(r"\n\s*\n", plain) if b.strip()]
        out[role] = sum(len(words_in(p)) for p in paras)
    return out


def main() -> int:
    per_paper, unmeasurable = {}, []
    for d in sorted(p for p in CORPUS.iterdir() if p.is_dir()):
        mds = list(d.glob("*.md"))
        if len(mds) != 1:
            continue
        try:
            per_paper[d.name] = measure_corpus_paper(mds[0])
        except ValueError as exc:
            unmeasurable.append({"paper": d.name, "reason": str(exc)})

    ours = measure_ours()
    table = []
    for role in ROLES + ["discussion_conclusion"]:
        vals = [p["roles"][role] for p in per_paper.values() if role in p["roles"]]
        if len(vals) < 3:
            table.append({"role": role, "n": len(vals), "ours": ours.get(role),
                          "band": None, "pct": None,
                          "verdict": "band undefined (fewer than 3 corpus papers carry it)"})
            continue
        b = band(vals)
        o = ours.get(role)
        pct = percentile_of(o, vals) if o is not None else None
        if o is None:
            verdict = "ours has no section of this role"
        elif o < b["min"]:
            verdict = f"below floor by {b['min'] - o}"
        elif o > b["max"]:
            verdict = f"above ceiling by {o - b['max']}"
        else:
            verdict = "in band"
        table.append({"role": role, "n": b["n"], "ours": o, "band": b, "pct": pct,
                      "verdict": verdict})

    OUT.write_text(json.dumps({
        "_schema": "venue_band_sections/1",
        "_instrument": "scripts/89_venue_band_sections.py",
        "_corpus": {"root": str(CORPUS.relative_to(config.ROOT)),
                    "n_measured": len(per_paper), "n_unmeasurable": len(unmeasurable),
                    "unmeasurable": unmeasurable},
        "_caveats": [
            "role mapping is a keyword rule over heading text; every assignment is listed",
            "ours merges discussion and conclusion; compared to the corpus tail summed per paper",
            "ours undercounts lines carrying \\ref/\\label (stripper drops the line tail)",
        ],
        "ours": ours,
        "table": table,
        "per_paper": per_paper,
    }, indent=2))

    print(f"corpus papers measured: {len(per_paper)}, unmeasurable: {len(unmeasurable)}")
    for u in unmeasurable:
        print(f"  unmeasurable: {u['paper'][:50]} -- {u['reason']}")
    print()
    print(f"{'role':24} {'n':>3} {'ours':>6} {'min':>6} {'median':>7} {'max':>6} {'pct':>4}  verdict")
    for r in table:
        b = r["band"]
        if b is None:
            print(f"{r['role']:24} {r['n']:>3} {str(r['ours']):>6} {'--':>6} {'--':>7} {'--':>6} {'--':>4}  {r['verdict']}")
        else:
            print(f"{r['role']:24} {b['n']:>3} {str(r['ours']):>6} {b['min']:>6} {b['median']:>7} "
                  f"{b['max']:>6} {str(r['pct']):>4}  {r['verdict']}")
    print()
    print("role assignments (audit these; they are a reading rule, not a measurement):")
    for name, p in per_paper.items():
        parts = ", ".join(f"{a['numeral']}.{a['title'][:28]}->{a['role']}({a['words']})"
                          for a in p["assignments"])
        print(f"  {name[:45]}: {parts}")
    print(f"\nwrote {OUT.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
