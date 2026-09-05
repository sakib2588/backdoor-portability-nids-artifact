#!/usr/bin/env python3
"""Render the markdown documents the manuscript defers to, into docs/artifact/.

The appendix used to carry a 54-row provenance table mapping every reported quantity to the
result file that produces it. At 55 rows of \\scriptsize it cost roughly a third of a page and
carried two numbers in total -- it is a file-name index, not data -- so on 2026-09-05 it moved
to PROVENANCE.md in the public artifact repository and the appendix now points at it.

Moving an index out of the paper is only safe if it cannot rot. This script therefore ASSERTS
that every results/<name>.json the index names actually exists, so a renamed or deleted result
file fails the export loudly instead of leaving the manuscript pointing at an index that
silently no longer resolves. That is the same class of failure as a LaTeX \\ref to a deleted
label, except LaTeX warns and a markdown index does not.

Source of truth is docs/provenance_index.json, converted verbatim from the LaTeX table body
rather than retyped.

scripts/77_build_artifact_repo.py copies docs/artifact/ into the published repository.

ASSERT ONLY, NEVER GENERATE: reads committed files and writes documentation.
It does not touch the manuscript and computes no experimental quantity.

Run:  .venv/bin/python scripts/110_export_docs.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import config

INDEX = config.ROOT / "docs" / "provenance_index.json"
OUT = config.ROOT / "docs" / "artifact"

# A cell may name several files, and may qualify one with a parenthetical such as
# "(H3 block)" or a JSON path. Only bare <name> tokens are checked for existence.
_TOKEN = re.compile(r"^[A-Za-z0-9_]+$")


def named_files(source: str) -> list[str]:
    out = []
    for part in source.split(","):
        tok = part.strip().split(" ")[0].strip()
        # continuation rows such as "_replication" name a suffix of the previous file
        if tok and _TOKEN.match(tok) and not tok.startswith("_"):
            out.append(tok)
    return out


def render_provenance() -> tuple[str, list[str]]:
    data = json.loads(INDEX.read_text())
    rows = data["rows"]
    missing: list[str] = []
    body = []
    for r in rows:
        for name in named_files(r["source"]):
            if not (config.RESULTS / f"{name}.json").is_file():
                missing.append(f"{r['quantity']} -> results/{name}.json")
        body.append(f"| {r['quantity']} | `{r['source']}` |")
    md = [
        "# Provenance of every reported quantity",
        "",
        "Each number in the manuscript is read from a committed result file rather than",
        "recomputed for the paper. This index maps each reported quantity to the file in",
        "`results/` that produces it. It stood as Table 13 of the appendix until it was moved",
        "here to reclaim page area; its content is unchanged.",
        "",
        "One class of exception is stated in the appendix itself: the 30-seed UNSW-NB15 figures",
        "merge three seed arms, and where a merged mean is quoted it is computed across the",
        "files named for that row rather than stored as a single field.",
        "",
        f"{len(rows)} rows.",
        "",
        "| Reported quantity | Source file (`results/`) |",
        "|---|---|",
        *body,
        "",
    ]
    return "\n".join(md), missing


NETFLOW_FAMILY = """# Activation Clustering on the corpus family: failure modes and scale control

Supporting detail for the appendix subsection "The detectors on the corpus family, against a
control on the same corpus". The appendix reports that Activation Clustering recovers 0.0063
of a blatant constraint-violating control on this family, and therefore that its cells there
are an unresolved control failure from which no portability verdict is drawn. This file
records what the detector does in those cells. None of it is evidence about whether the
detector ports.

## Two failure modes a mean recall cannot separate

The distinction decides whether an operator would notice. Across 75 interpretable cells the
detector isolates the poison in 41, collapses to flagging between 1 and 16 rows in 21, and
flags the wrong cluster in 13. The wrong-cluster cells take between 5.3% and 47.1% of clean
rows with them. Thirty-two of those 34 failing cells report a recall of exactly 0.000 and the
remaining two report 0.0635 and 0.0002, so per-cell recall is bimodal on 73 of the 75. That is
what a hard two-cluster assignment produces rather than a ranked score.

## A candidate mechanism for the wrong-cluster mode

Its silhouette sits near 0.5, far above the 0.125 at which the two-cluster structure would be
judged unreal, so the split is confident and genuine. It is not a split about poison. The
reading we cannot exclude, and do not test directly, is that the benign class on these corpora
carries dominant structure of its own and the two-cluster assumption lands on that structure.
The relative-size rule then flags the wrong half. The gate that exists to suppress a split that
is not real cannot fire, because this split is real.

## The subsample cap does not cause either failure

The target-class pool reaches 921,824 rows here against 111,666 on CTU-13, so detectors are
scored on a uniform subsample capped at 200,000 that preserves the poison fraction. Every
wrong-cluster cell was a subsampled one. We therefore re-ran both failure modes at full scale,
changing only the number of rows scored. The collapse still flags 49 to 52 rows at 921,600. The
wrong-cluster case still flags 349,767 to 356,472 at 768,000, between 45.5% and 46.4% of the
pool against 45.4% to 46.5% at the cap, and at the same silhouette. The cap does not cause
either failure.
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prov, missing = render_provenance()
    if missing:
        print("FAIL: the provenance index names result files that do not exist:")
        for m in missing:
            print(f"  {m}")
        return 1
    (OUT / "PROVENANCE.md").write_text(prov, encoding="utf-8")
    (OUT / "NETFLOW_FAMILY.md").write_text(NETFLOW_FAMILY, encoding="utf-8")
    n_rows = len(json.loads(INDEX.read_text())["rows"])
    print(f"PROVENANCE.md      {n_rows} rows, every named results file exists")
    print(f"NETFLOW_FAMILY.md  {len(NETFLOW_FAMILY.split())} words")
    print(f"\nwrote {OUT.relative_to(config.ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
