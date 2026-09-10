#!/usr/bin/env python3
"""Report how far every table and figure lands from its first in-text mention.

Why this exists. On 2026-09-03 the IEEE Access build had every body table landing
8 to 16 pages after its first reference, and nothing in the T0 gates said so. The
cause was one table 12.9pt taller than a page (main.log: "Float too large for page");
LaTeX can never place such a float, same-class floats keep their order, so all twelve
tables queued behind it and flushed at the \\FloatBarrier before the appendix. One
number in one table fixed all twelve. This makes the symptom visible so the next
such backlog is caught at build time rather than by a reader.

Method. pdftotext without -layout (so a right-column caption is not lost behind a
left-column line), pages split on form feeds. A float's landing page is the page of
its caption ("TABLE 5." / "FIGURE 3.", upper-case, as ieeeaccess.cls sets them). Its
first mention is the first page containing "Table 5" / "Fig. 5" in body case, which
cannot match a caption. Delta is landing minus first mention.

A delta of 2 or more is reported. It is not automatically a defect: a table placed in
an appendix and pointed at from the body, or a Results table forward-cited once from
Methods, will show a large delta by design. The check reports; the author decides.
Also reports any float with no body mention at all, which IS a defect.

Run:  python3 scripts/90_float_displacement_check.py [--pdf paper_access/main.pdf] [--tolerance 2]
Exit 1 if any float exceeds the tolerance or has no mention, so it can gate a build.
"""
from __future__ import annotations

import argparse
import re
import pathlib
import subprocess
import sys

_CAP_RE = re.compile(r"\b(TABLE|FIGURE) (\d+)\.")
_REF_RE = re.compile(r"\b(Table|Fig\.)\s*(\d+)\b")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="paper_access/main.pdf")
    ap.add_argument("--tolerance", type=int, default=2)
    args = ap.parse_args()

    # Without this the script exits 0 on a missing or unbuilt PDF: pdftotext writes nothing,
    # zero floats are found, nothing is flagged, and a green pass is reported for a file that
    # does not exist. The default target is paper_access, so a run against another manuscript
    # that forgot --pdf would have passed on the wrong paper.
    if not pathlib.Path(args.pdf).is_file():
        raise SystemExit(f"--pdf {args.pdf} does not exist; build it first")

    txt = subprocess.run(["pdftotext", args.pdf, "-"], capture_output=True, text=True).stdout
    if not txt.strip():
        raise SystemExit(f"--pdf {args.pdf} produced no text layer")
    pages = txt.split("\f")
    cap: dict[tuple[str, int], int] = {}
    ref: dict[tuple[str, int], int] = {}
    for i, p in enumerate(pages, 1):
        for m in _CAP_RE.finditer(p):
            cap.setdefault((m.group(1), int(m.group(2))), i)
        for m in _REF_RE.finditer(p):
            k = ("TABLE" if m.group(1) == "Table" else "FIGURE", int(m.group(2)))
            ref.setdefault(k, i)

    print(f"{args.pdf}: {len(pages) - 1} pages, {len(cap)} captioned floats")
    print(f"{'float':10} {'first ref':>9} {'lands':>6} {'delta':>6}")
    flagged = 0
    for k in sorted(cap):
        r, c = ref.get(k), cap[k]
        d = None if r is None else c - r
        if r is None:
            flag = "  <-- NO BODY MENTION"
        elif abs(d) >= args.tolerance:
            flag = f"  <-- |delta| >= {args.tolerance}"
        else:
            flag = ""
        flagged += bool(flag)
        print(f"{k[0]} {k[1]:<5} {str(r):>9} {c:>6} {str(d):>6}{flag}")
    print(f"flagged: {flagged}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
