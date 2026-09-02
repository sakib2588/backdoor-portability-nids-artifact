#!/usr/bin/env python3
"""How many body lines can this paper gain before it spills to a seventh page?

`available_points` in scripts/33_tier0_premises.py measures the leftover on the FINAL
page. On a document whose text reflows, that is 0.0 essentially always -- prose simply
re-fills the page -- so it cannot answer the only budget question that matters when
adding content. This project's own record says the same thing: estimating page cost
from .tex line counts was wrong by more than a factor of two in both directions, and
only recompiling settles it.

So this measures slack the only honest way: insert K lines of realistic filler prose,
compile, and binary-search the largest K that still fits in the target page count.
paper/ is never touched -- everything happens in a scratch copy.

Filler is real sentences, not \\vspace, because a rigid skip does not interact with
float placement and column breaking the way prose does, and float placement is what
actually decides this document's page count.

Run:  python scripts/46_page_slack_probe.py SCRATCH_DIR [--max-pages 6] [--hi 60]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config

# ~10.6 words per rendered line at IEEEtran 10/12pt two-column A4, measured from the
# committed build (5,737 PDF words over six pages).
FILLER = ("The measured quantity is reported here for completeness and carries no claim "
          "beyond the value stated in the preceding sentence of this paragraph. ")


def build(work: Path) -> int:
    for junk in ("main.pdf", "main.aux", "main.bbl", "main.log", "main.out",
                 "main.fdb_latexmk", "main.fls"):
        (work / junk).unlink(missing_ok=True)
    for cmd in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    pdf = work / "main.pdf"
    if not pdf.exists():
        return 10**6
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", info, re.M)
    return int(m.group(1)) if m else 10**6


def probe(src: Path, scratch: Path, lines: int, max_pages: int) -> bool:
    work = scratch / f"slack_{lines}"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(src, work)
    target = work / "sections" / "03_results.tex"
    text = target.read_text()
    # Append to the final paragraph of Results, the least float-entangled site.
    filler = "\n\n" + (FILLER * max(1, lines)).strip() + "\n" if lines else ""
    target.write_text(text + filler)
    pages = build(work)
    shutil.rmtree(work, ignore_errors=True)
    return pages <= max_pages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scratch")
    ap.add_argument("--max-pages", type=int, default=6)
    ap.add_argument("--hi", type=int, default=60)
    args = ap.parse_args()

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    src = config.ROOT / "paper"

    if not probe(src, scratch, 0, args.max_pages):
        print(f"BASELINE ALREADY EXCEEDS {args.max_pages} PAGES -- slack is negative")
        return 1

    lo, hi = 0, args.hi
    if probe(src, scratch, hi, args.max_pages):
        print(f"slack is at least {hi} lines (raise --hi to bound it)")
        return 0
    print(f"bracketing between {lo} and {hi} lines")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok = probe(src, scratch, mid, args.max_pages)
        print(f"  {mid:3d} filler lines -> {'fits' if ok else 'spills'}")
        if ok:
            lo = mid
        else:
            hi = mid
    print(f"\nMEASURED SLACK: {lo} body lines can be added before page {args.max_pages + 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
