#!/usr/bin/env python3
"""Tier 1 numeric provenance ledger for the ICCIT review (cycle 2).

This manuscript has no LaTeX macro layer: every number is a hand-typed literal.
Prose review does not catch a stale or invented number, so this walks the other
way -- it takes each literal out of the .tex and tries to find it on disk.

Each literal is classified:
  TRACED          matches a value in some results/*.json at printed precision
  ARITHMETIC      appears in a sentence asserting a relation among numbers (T1-c)
  CONFIG          matches a constant in src/config.py
  CITATION_YEAR   a year inside a \\cite-adjacent context or a reference
  STRUCTURAL      LaTeX geometry (widths, skips), not a claim
  UNTRACED        nothing on disk produces it -- this is the finding

Deliberately conservative: a literal is only TRACED when a JSON value rounds to
it exactly at the precision the paper prints. "Close" is not traced.

Usage:
    python3 scripts/34_tier1_literal_ledger.py [--write-manifest] [--show untraced]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "paper" / "sections"
MAIN = ROOT / "paper" / "main.tex"
RESULTS = ROOT / "results"

# LaTeX we must remove before hunting for numbers, or citation years and float
# skips masquerade as claims.
STRIP_PATTERNS = [
    (re.compile(r"(?<!\\)%.*$", re.M), ""),                 # comments
    (re.compile(r"\\cite[tp]?\{[^}]*\}"), " CITE "),
    (re.compile(r"\\(ref|label|eqref|autoref)\{[^}]*\}"), " REF "),
    (re.compile(r"\\includegraphics\[[^\]]*\]\{[^}]*\}"), " GRAPHIC "),
    (re.compile(r"\\[a-zA-Z]+\s*=\s*-?[\d.]+\s*(pt|in|cm|em|ex)"), " LENGTH "),
    (re.compile(r"[\d.]+\\(linewidth|columnwidth|textwidth|baselineskip)"), " LENGTH "),
    (re.compile(r"\\[a-zA-Z]+"), " "),                       # remaining control words
]

NUMBER = re.compile(
    r"(?<![\w.])"
    r"(-?\d{1,3}(?:(?:\{,\}|,)\d{3})+(?:\.\d+)?"   # 143{,}046 / 8,017
    r"|-?\d*\.\d+"                                     # 0.9825
    r"|-?\d+)"                                           # 407
    r"(?![\w])"
)

# Sentences asserting a relation among their own numbers. These can be false
# with no JSON involved at all, so provenance tracing passes them regardless.
ARITHMETIC_CUES = re.compile(
    r"\b(\d[\d.,{}]*)\s+of\s+(\d[\d.,{}]*)\b"           # "8 of 20"
    r"|\bof\s+the\s+(\d[\d.,{}]*)\b"
    r"|\btimes\b|\bagainst\b|\bso\s+the\b|\bleaving\b"
    r"|\brises?\s+from\b|\bfalls?\s+from\b|\bdrops?\s+from\b"
    r"|\bfrom\s+[\d.]+\s+to\s+[\d.]+",
    re.I,
)


def clean_tex(text: str) -> str:
    for pat, repl in STRIP_PATTERNS:
        text = pat.sub(repl, text)
    return text


def normalise(tok: str) -> str:
    """LaTeX thousands separators: 8{,}017 and 8,017 are the same number."""
    return tok.replace("{,}", "").replace(",", "").replace("{", "").replace("}", "")


def iter_json_values(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_json_values(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_json_values(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield float(obj), path


def build_value_index() -> dict[str, list[str]]:
    """Map every printable rendering of every number on disk to where it came from.

    A value is indexed at every precision from 0 to 6 decimals, and also as a
    percentage, because the paper prints 0.0658 as "6.58%" in some places.
    """
    index: dict[str, list[str]] = defaultdict(list)
    files = sorted(RESULTS.rglob("*.json"))
    for jf in files:
        if ".checkpoint." in jf.name or jf.name.endswith(".bak"):
            continue
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        rel = str(jf.relative_to(ROOT))
        for val, path in iter_json_values(data):
            for dp in range(0, 7):
                for signed in (val, -val):
                    index[f"{signed:.{dp}f}".rstrip("0").rstrip(".") or "0"].append(f"{rel}:{path}")
                    index[f"{signed:.{dp}f}"].append(f"{rel}:{path}")
            pct = val * 100.0
            for dp in range(0, 4):
                index[f"{pct:.{dp}f}"].append(f"{rel}:{path}(as %)")
            if abs(val - round(val)) < 1e-9:
                index[str(int(round(val)))].append(f"{rel}:{path}")
    return index


def config_constants() -> set[str]:
    cfg = ROOT / "src" / "config.py"
    if not cfg.exists():
        return set()
    out = set()
    for m in re.finditer(r"=\s*([\d_]+\.?\d*)", cfg.read_text()):
        raw = m.group(1).replace("_", "")
        out.add(raw)
        try:
            f = float(raw)
            out.add(f"{f:g}")
            if abs(f - round(f)) < 1e-9:
                out.add(str(int(round(f))))
        except ValueError:
            pass
    return out


def sentences_of(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if p.strip()]


def classify(tok_raw: str, sentence: str, index, cfg) -> tuple[str, list[str]]:
    tok = normalise(tok_raw)
    hits = index.get(tok, [])
    if hits:
        # A short integer collides with something in some results file almost
        # every time, so reporting it as traced would be a check that cannot
        # fail. Only distinctive literals earn a full trace.
        digits = tok.lstrip("-").replace(".", "").replace("{,}", "").replace(",", "")
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        distinctive = decimals >= 2 or len(digits) >= 4
        return ("TRACED" if distinctive else "WEAK_TRACE"), sorted(set(hits))[:4]
    if ARITHMETIC_CUES.search(sentence):
        return "ARITHMETIC", []
    if tok in cfg:
        return "CONFIG", ["src/config.py"]
    if re.fullmatch(r"(19|20)\d\d", tok):
        return "CITATION_YEAR", []
    return "UNTRACED", []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--show", default=None, help="print every literal with this class")
    args = ap.parse_args()

    index = build_value_index()
    cfg = config_constants()

    ledger = []
    tex_files = sorted(SECTIONS.glob("*.tex")) + [MAIN]
    for tex in tex_files:
        if not tex.exists():
            continue
        rel = str(tex.relative_to(ROOT))
        for lineno, raw_line in enumerate(tex.read_text(errors="replace").splitlines(), 1):
            line = clean_tex(raw_line)
            for sent in sentences_of(line):
                for m in NUMBER.finditer(sent):
                    tok = m.group(1)
                    cls, prov = classify(tok, sent, index, cfg)
                    ledger.append(
                        {
                            "file": rel,
                            "line": lineno,
                            "literal": tok,
                            "normalised": normalise(tok),
                            "class": cls,
                            "provenance": prov,
                            "sentence": sent[:400],
                        }
                    )

    counts = defaultdict(int)
    for e in ledger:
        counts[e["class"]] += 1

    print(f"literal occurrences: {len(ledger)}   distinct: {len({e['normalised'] for e in ledger})}")
    for cls in ("TRACED", "WEAK_TRACE", "ARITHMETIC", "CONFIG", "CITATION_YEAR", "UNTRACED"):
        print(f"  {cls:14s} {counts[cls]:4d}")

    per_file = defaultdict(lambda: defaultdict(int))
    for e in ledger:
        per_file[e["file"]][e["class"]] += 1
    print("\nby file:")
    for f in sorted(per_file):
        row = per_file[f]
        print(f"  {Path(f).name:32s} untraced={row['UNTRACED']:3d}  traced={row['TRACED']:3d}  "
              f"arith={row['ARITHMETIC']:3d}  total={sum(row.values()):3d}")

    if args.show:
        want = args.show.upper()
        print(f"\n--- literals classified {want}")
        seen = set()
        for e in ledger:
            if e["class"] != want:
                continue
            key = (e["file"], e["line"], e["normalised"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  {Path(e['file']).name}:{e['line']}  {e['literal']}")
            print(f"      {e['sentence'][:170]}")

    if args.write_manifest:
        out = RESULTS / "tier1_literal_ledger.json"
        out.write_text(json.dumps(
            {"counts": dict(counts), "total": len(ledger), "ledger": ledger},
            indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}")

    return 1 if counts["UNTRACED"] else 0


if __name__ == "__main__":
    sys.exit(main())
