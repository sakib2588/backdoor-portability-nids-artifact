#!/usr/bin/env python3
"""Rebuild the supervisor-facing corpus-tables PDF from its markdown source.

The PDF is a derived SUBSET of notes/20260902-report-venue-band-corpus-tables.md:
headings, tables, and bold-led note paragraphs only. Ordinary explanatory prose
is dropped.

Adapted from the sibling ML_Paper project's scripts/104_build_corpus_tables_pdf.py
(same extraction algorithm, same pandoc invocation) -- see that file's own docstring
for the drift-risk rationale this pattern exists to prevent: a supervisor-facing
PDF that silently falls behind its markdown source after an edit is worse than no
PDF at all. Run this after any edit to the source note, and diff the page count.

Usage:  python3 scripts/88_build_corpus_tables_pdf.py [--check]

  --check   exit 1 if the PDF is older than its source, build nothing
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

SRC = "notes/20260902-report-venue-band-corpus-tables.md"
OUT = "notes/20260902-report-venue-band-corpus-tables-only.pdf"
# Sections omitted from the supervisor-facing PDF. They stay in the markdown,
# which is the archive; this only controls what a supervisor is handed.
#   "8" -- "What this does not settle" is three prose paragraphs with no table
#          and no bold-led line anywhere in it (verified by reading the source
#          in full before writing this list). The line-level extraction below
#          keeps only headings, tables, and bold-led notes, so this section
#          would otherwise survive as an empty heading with nothing under it.
# Every other section carries unique tabular or bold-flagged content -- unlike
# the sibling project's Section 0, this note's Section 0 caveats are not
# restated anywhere else, so it is NOT in this list.
DROP_SECTIONS = {"8"}


def extract(md: str) -> str:
    """Keep headings, tables, and bold-led note paragraphs. Drop plain prose."""
    out, in_table, skipping = [], False, False
    for line in md.split("\n"):
        s = line.strip()

        if s.startswith("## "):
            num = s[3:].split(".")[0].strip()
            skipping = num in DROP_SECTIONS
            in_table = False
            if not skipping:
                out += ["", line, ""]
            continue

        if s.startswith("# "):          # document title
            out += [line, ""]
            continue

        if skipping:
            continue

        # Blank lines MUST survive: pandoc needs one to separate a paragraph from
        # a table that follows it.
        if not s:
            out.append("")
            in_table = False
            continue

        # Horizontal rules and bullets are dropped BEFORE the continuation rule
        # below, which would otherwise swallow a "---" into the running paragraph.
        if s.startswith("---") or s.startswith("- "):
            continue

        if s.startswith("|"):           # table row
            out.append(line)
            in_table = True
            continue

        # bold-led paragraphs are this document's convention for flagged notes
        # (the venue-band caveats, the BLOCKED biography status) -- those the
        # reader needs.
        if s.startswith("**"):
            out.append(line)
            continue

        # continuation of a bold-led note
        if out and out[-1].strip() and not out[-1].strip().startswith("|"):
            if out[-1].strip().startswith("**") or _is_note_body(out):
                out.append(line)
                continue

    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text)


def _is_note_body(out) -> bool:
    """True if we are inside a bold-led paragraph that has not ended yet."""
    for prev in reversed(out):
        p = prev.strip()
        if not p:
            return False
        if p.startswith("**"):
            return True
        if p.startswith("|") or p.startswith("#"):
            return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(SRC):
        print(f"FAIL: {SRC} not found", file=sys.stderr)
        return 2

    if args.check:
        if not os.path.exists(OUT):
            print(f"STALE: {OUT} does not exist")
            return 1
        if os.path.getmtime(OUT) < os.path.getmtime(SRC):
            print(f"STALE: {OUT} is older than {SRC} -- rebuild")
            return 1
        print("OK: PDF is at least as new as its source")
        return 0

    body = extract(open(SRC, encoding="utf-8").read())
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(body)
        tmp = fh.name

    cmd = [
        "pandoc", tmp, "-o", OUT,
        "--pdf-engine=pdflatex",
        "-V", "geometry:a4paper,landscape,margin=1.5cm",
        "-V", "fontsize=9pt",
        "-V", "colorlinks=true",
        "--toc", "--toc-depth=2",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(tmp)
    if r.returncode != 0:
        print(r.stdout[-3000:], file=sys.stderr)
        print(r.stderr[-3000:], file=sys.stderr)
        print("FAIL: pandoc build failed", file=sys.stderr)
        return 2

    print(f"built {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
