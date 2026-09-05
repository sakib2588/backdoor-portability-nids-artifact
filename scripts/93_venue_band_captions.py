#!/usr/bin/env python3
"""Per-caption word budget of paper_access/ against the IEEE Access venue band.

Why this exists. scripts/89 measures section prose, but src/tex_prose.py's _ENV_DROP_RE
deletes every table/figure/tabular environment before counting, so our caption text is
invisible to it. The corpus side of 89 does NOT drop captions -- a "**TABLE 1.** ..." line
survives _split_paragraphs as an ordinary paragraph -- so the section bands already compare
corpus-prose-plus-captions against our-prose-only. This script measures the axis that
asymmetry hides.

Both sides are read as rendered text rather than source. The corpus is marker-converted
markdown; ours is pdftotext over the built PDF. There is no brace-balanced \\caption{}
parser anywhere in this repo, and our captions carry \\textbf{}, $...$ and \\cite{}, so
parsing the LaTeX would need one written from scratch. ieeeaccess.cls sets captions
upper-case ("TABLE 5.") while body mentions are "Table 5" / "Fig. 5", which is what makes
the PDF route separable by case -- the same convention scripts/90 relies on.

Two normalizations are reported because they answer different questions. Per-paper TOTAL
is confounded by float count (we carry 19 floats against a corpus of 8-14), so per-caption
MEAN is the fairer comparison; totals are reported alongside so the reader can see both.

Not additive with results/venue_band_sections.json. See _caveats in the output.
"""

import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import config
from src.prose_metrics import band, percentile_of, words_in
from src.reference_corpus import _REFERENCES_HEAD_RE, _strip_markup

CORPUS = config.ROOT / "references" / "venue_band_md"
PDF = config.ROOT / "paper_access" / "main.pdf"
OUT = config.RESULTS / "venue_band_captions.json"

# Marker emits four caption conventions across the 18-paper corpus. Measured frequency:
#   **LABEL n.** text        330 lines
#   LABEL n. text (bare)      10
#   #### **LABEL n.** text     8
#   **LABEL n. text**          1   <- the form scripts/75:80-82 misses
# The fourth is 1 line in 349, so widening the pattern is a correctness fix, not a
# material change to the band. [ ~] is required: marker sometimes emits a non-breaking
# space between the label and its number.
_CORPUS_CAP_RE = re.compile(
    r"^#{0,6}\s*\*{0,2}(FIGURE|TABLE)[ ~]*(\d+)\.\*{0,2}\s*(.*)$", re.IGNORECASE
)

# Our side, from pdftotext. ieeeaccess.cls upper-cases the caption label.
_OURS_CAP_RE = re.compile(r"^(TABLE|FIGURE)\s+(\d+)\.\s*(.*)$")

# A numeric citation inside a caption is not a word.
_NUMERIC_CITE_RE = re.compile(r"\[\d+(?:\s*[,-]\s*\d+)*\]")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _caption_words(text: str) -> int:
    """Word count of one caption's body, citations and markdown furniture removed."""
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _NUMERIC_CITE_RE.sub("", text)
    text = re.sub(r"[*#_`]", "", text)
    return len(words_in(text))


def measure_corpus_paper(md: pathlib.Path) -> dict:
    """Caption word counts for one corpus paper, references tail excluded."""
    cleaned = _strip_markup(md.read_text(encoding="utf-8", errors="replace"))
    refs_m = _REFERENCES_HEAD_RE.search(cleaned)
    body = cleaned[: refs_m.start()] if refs_m else cleaned

    figs, tabs = [], []
    for line in body.splitlines():
        m = _CORPUS_CAP_RE.match(line.strip())
        if not m:
            continue
        w = _caption_words(m.group(3))
        if not w:
            continue
        (figs if m.group(1).upper() == "FIGURE" else tabs).append(w)

    if not (figs or tabs):
        raise ValueError("no captions matched")
    return {"figure": figs, "table": tabs}


def measure_ours() -> dict:
    """Caption word counts for the built manuscript, via pdftotext.

    A caption runs from its label to the first blank line. Deduplicated by (kind, number)
    because a float's label can recur in a running header or a continued table.
    """
    if not PDF.exists():
        raise SystemExit(f"missing {PDF} -- build the paper first")
    txt = subprocess.run(
        ["pdftotext", str(PDF), "-"], capture_output=True, text=True
    ).stdout

    seen: dict[tuple[str, int], int] = {}
    lines = txt.splitlines()
    for i, line in enumerate(lines):
        m = _OURS_CAP_RE.match(line.strip())
        if not m:
            continue
        key = (m.group(1), int(m.group(2)))
        if key in seen:
            continue
        body = [m.group(3)]
        for nxt in lines[i + 1 :]:
            if not nxt.strip():
                break
            body.append(nxt)
        seen[key] = _caption_words(" ".join(body))

    figs = [w for (k, _), w in seen.items() if k == "FIGURE" and w]
    tabs = [w for (k, _), w in seen.items() if k == "TABLE" and w]
    return {"figure": figs, "table": tabs}


def _verdict(ours: float, b: dict | None) -> str:
    if b is None:
        return "band undefined (fewer than 3 corpus papers carry it)"
    if ours < b["min"]:
        return f"below floor by {round(b['min'] - ours, 1)}"
    if ours > b["max"]:
        return f"above ceiling by {round(ours - b['max'], 1)}"
    return "in band"


def _row(label: str, ours_vals: list[int], corpus_vals: list[float], stat: str) -> dict:
    o = sum(ours_vals) if stat == "total" else (
        round(sum(ours_vals) / len(ours_vals), 2) if ours_vals else 0
    )
    b = band(corpus_vals) if len(corpus_vals) >= 3 else None
    return {
        "metric": label,
        "stat": stat,
        "ours": o,
        "n_ours_captions": len(ours_vals),
        "band": b,
        "pct": percentile_of(o, corpus_vals) if b else None,
        "verdict": _verdict(o, b),
    }


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
    ours_all = ours["figure"] + ours["table"]

    table = []
    for kind in ("figure", "table", "combined"):
        if kind == "combined":
            o = ours_all
            tot = [sum(v["figure"] + v["table"]) for v in per_paper.values()]
            mean = [
                sum(v["figure"] + v["table"]) / len(v["figure"] + v["table"])
                for v in per_paper.values()
                if v["figure"] + v["table"]
            ]
        else:
            o = ours[kind]
            tot = [sum(v[kind]) for v in per_paper.values() if v[kind]]
            mean = [sum(v[kind]) / len(v[kind]) for v in per_paper.values() if v[kind]]
        table.append(_row(kind, o, tot, "total"))
        table.append(_row(kind, o, mean, "mean_per_caption"))

    payload = {
        "_schema": "venue_band_captions/1",
        "_instrument": "scripts/93_venue_band_captions.py",
        "_corpus": {
            "root": str(CORPUS.relative_to(config.ROOT)),
            "n_measured": len(per_paper),
            "n_unmeasurable": len(unmeasurable),
            "unmeasurable": unmeasurable,
        },
        "_caveats": [
            "NOT additive with results/venue_band_sections.json: that file's corpus side "
            "counts caption words inside its section totals while its ours side drops "
            "them, so the two files measure overlapping text on the corpus side only.",
            "scripts/89:122 strips heading lines before paragraph splitting, so the 8 "
            "heading-form corpus captions are excluded from that file's section counts "
            "but are included here.",
            "Corpus captions are read from marker-converted markdown and ours from "
            "pdftotext; both are rendered text, but they are not the same converter.",
            "A caption here ends at the first blank line in the PDF text layer. A caption "
            "broken across a column boundary can therefore be undercounted.",
        ],
        "ours": {
            "figure": sorted(ours["figure"], reverse=True),
            "table": sorted(ours["table"], reverse=True),
            "total_words": sum(ours_all),
            "n_captions": len(ours_all),
        },
        "table": table,
        "per_paper": {
            k: {
                "figure_words": sum(v["figure"]),
                "table_words": sum(v["table"]),
                "total_words": sum(v["figure"] + v["table"]),
                "n_captions": len(v["figure"] + v["table"]),
            }
            for k, v in per_paper.items()
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"corpus: {len(per_paper)} papers measured, {len(unmeasurable)} unmeasurable")
    print(f"ours:   {len(ours_all)} captions, {sum(ours_all)} words\n")
    hdr = f"{'metric':10} {'stat':17} {'ours':>8} {'min':>8} {'median':>8} {'max':>8} {'pct':>4}  verdict"
    print(hdr)
    over = 0
    for r in table:
        b = r["band"] or {}
        print(
            f"{r['metric']:10} {r['stat']:17} {r['ours']:>8} "
            f"{b.get('min', '--'):>8} {b.get('median', '--'):>8} {b.get('max', '--'):>8} "
            f"{r['pct'] if r['pct'] is not None else '--':>4}  {r['verdict']}"
        )
        if r["verdict"].startswith("above"):
            over += 1
    print(f"\nwrote {OUT.relative_to(config.ROOT)}")
    return 1 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
