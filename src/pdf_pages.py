"""Page-count-only PDF measurement for the corpus-comparison study (Tier 4).

Scoped down from the reference NDSS-corpus instrument's src/pdf_pages.py: we don't need a
body/back-matter heading-boundary split here, because ICCIT's 6pp cap governs total pages
directly (references included, no separate appendix carve-out) -- unlike NDSS, where the cap
excludes back matter and that split is load-bearing. Just page counts.
"""
from __future__ import annotations

import pathlib
import subprocess


class PdfToolMissing(Exception):
    pass


def page_count(pdf: pathlib.Path) -> int | None:
    try:
        out = subprocess.run(
            ["pdfinfo", str(pdf)], capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError as e:
        raise PdfToolMissing("pdfinfo not found on PATH") from e
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None
