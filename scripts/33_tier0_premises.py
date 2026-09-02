#!/usr/bin/env python3
"""Tier 0 premise measurement for the ICCIT review (cycle 2).

Measures every fact both review tracks will argue about, so no finding is ever
debated against an estimate. ASSERT ONLY, NEVER GENERATE: this script measures
the artefact, it never edits it.

Checks:
  T0-a  build-freshness proof   -- fdb_latexmk recorded md5/size vs disk
  T0-b  page ledger             -- per-page last baseline, figure render scale
  T0-c  codepoint audit         -- every non-ASCII codepoint in PDF text + bib
  T0-d  render audit            -- figure rendered width and implied font size
  T0-e  venue compliance        -- geometry/class options readable from source

Writes results/tier0_premises.json when --write-manifest is passed; otherwise
measures and prints only. Exit 0 all green, 1 any check failed, 2 could not
measure (never a pass).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
PDF = PAPER / "main.pdf"
FDB = PAPER / "main.fdb_latexmk"
LOG = PAPER / "main.log"

# IEEEtran two-column: \columnwidth measured from the compiled document, not assumed.
# Fallback only if the bbox probe cannot run.
FALLBACK_COLUMN_WIDTH_PT = 252.0

# pdftotext emits PostScript big points; LaTeX points are 1/72.27 in.
BP_TO_PT = 72.27 / 72.0
# IEEEtran \normalsize is 10pt on a 12pt baseline (IEEEtran.cls), so one body line
# of vertical space is 12pt. Used to price float changes in lines rather than points.
BODY_BASELINE_PT = 12.0

# Codepoints allowed to appear outside ASCII, each with the reason it is there.
CODEPOINT_WHITELIST = {
    "\u2018": "left single quote (TeX ligature)",
    "\u2019": "right single quote / apostrophe (TeX ligature)",
    "\u201c": "left double quote (TeX ligature)",
    "\u201d": "right double quote (TeX ligature)",
    "\u2013": "en dash (numeric ranges)",
    "\u2014": "em dash",
    "\u2212": "minus sign (math)",
    "\u00d7": "multiplication sign (grid notation)",
    "\u2264": "less-than-or-equal (math)",
    "\u2265": "greater-than-or-equal (math)",
    "\u2260": "not-equal (math)",
    "\u2248": "approximately equal (math)",
    "\u00a0": "non-breaking space",
    "\ufb01": "fi ligature",
    "\ufb02": "fl ligature",
    "\u00e9": "e-acute (author names)",
    "\u00ed": "i-acute (author names)",
    "\u00e1": "a-acute (author names)",
    "\u00fc": "u-diaeresis (author names)",
    "\u00f6": "o-diaeresis (author names)",
    "\u00e7": "c-cedilla (author names)",
    "\u00b1": "plus-minus (mean +/- sd notation)",
    "\u00b7": "middle dot (multiplication in maths)",
    "\u03c1": "Greek rho (Spearman correlation)",
    "\u2208": "element-of (set membership in maths)",
    "\u2308": "left ceiling (removal-budget formula)",
    "\u2309": "right ceiling (removal-budget formula)",
    "\u2192": "rightwards arrow (Fig. 1 mechanism diagram, TikZ \\rightarrow)",
}

# Codepoints that are legal but worth reporting: a decomposed accent renders
# correctly yet breaks copy-paste and full-text search on the author name.
CODEPOINT_NOTE = {
    "\u0131": "dotless i -- half of a decomposed accent; check the name copy-pastes",
    "\u0301": "combining acute -- decomposed accent, see U+0131",
    "\u0302": "combining circumflex -- decomposed accent",
}

# Banned outright by the workspace rule, regardless of context.
BANNED = {
    "\u00a7": "section-mark",
    "\u00b6": "pilcrow",
    "\u2020": "dagger",
    "\u2021": "double-dagger",
}


class CannotMeasure(RuntimeError):
    """Raised when a check cannot be performed. Never silently downgraded to a pass."""


def _need(tool: str) -> None:
    if shutil.which(tool) is None:
        raise CannotMeasure(f"required tool not on PATH: {tool}")


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CannotMeasure(f"{cmd[0]} failed: {proc.stderr.strip()[:200]}")
    return proc.stdout


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- T0-a

FDB_INPUT = re.compile(r'^\s+"([^"]+)"\s+(\S+)\s+(\d+)\s+([0-9a-f]{32})\s+')


def check_build_freshness() -> dict:
    """Every input the PDF was built from must still hash to what latexmk recorded.

    Compares md5, not mtime: a touched-but-unchanged file is not a stale build,
    and an edited-but-backdated file is.
    """
    if not PDF.exists():
        raise CannotMeasure(f"missing {PDF}")
    if not FDB.exists():
        raise CannotMeasure(f"missing {FDB}")

    # A latexmk database only describes the run that wrote it. If the PDF was
    # produced later by some other route (a bare pdflatex chain, Overleaf), the
    # fdb is a stale witness and its size/md5 rows say nothing about this build.
    # Reporting that as "stale sources" is a false alarm -- it fired once and
    # cost an investigation, so the check now refuses to testify instead.
    fdb_mtime = FDB.stat().st_mtime
    pdf_mtime = PDF.stat().st_mtime
    if fdb_mtime < pdf_mtime - 1.0:
        raise CannotMeasure(
            f"fdb_latexmk ({fdb_mtime:.0f}) predates main.pdf ({pdf_mtime:.0f}) by "
            f"{(pdf_mtime - fdb_mtime) / 86400:.1f} days: it did not produce this PDF. "
            "Prove freshness by rebuilding into a scratch directory and diffing the "
            "text layer (see notes/20260820-review-cycle2-tier0.md)."
        )

    recorded: dict[str, tuple[int, str]] = {}
    for line in FDB.read_text(errors="replace").splitlines():
        m = FDB_INPUT.match(line)
        if not m:
            continue
        name, _mtime, size, digest = m.groups()
        recorded[name] = (int(size), digest)

    project_inputs = {
        name: v
        for name, v in recorded.items()
        if not name.startswith("/usr/share/") and not name.startswith("/var/lib/")
    }

    stale, missing, checked = [], [], 0
    for name, (size, digest) in sorted(project_inputs.items()):
        path = (PAPER / name).resolve()
        if not path.exists():
            missing.append(name)
            continue
        checked += 1
        actual_size = path.stat().st_size
        actual_md5 = md5_of(path)
        if actual_size != size or actual_md5 != digest:
            stale.append(
                {
                    "input": name,
                    "recorded_size": size,
                    "actual_size": actual_size,
                    "recorded_md5": digest,
                    "actual_md5": actual_md5,
                }
            )

    # Generated files (.aux/.bbl/.out) legitimately change during the run itself.
    generated_suffixes = (".aux", ".bbl", ".out", ".blg", ".log")
    stale_source = [s for s in stale if not s["input"].endswith(generated_suffixes)]

    return {
        "check": "T0-a build freshness",
        "recorded_inputs_total": len(recorded),
        "project_inputs_checked": checked,
        "missing_inputs": missing,
        "stale_inputs": stale,
        "stale_source_inputs": stale_source,
        "pdf_mtime": PDF.stat().st_mtime,
        "pass": not stale_source and not missing,
        "note": (
            "Generated files (.aux/.bbl/.out) are excluded from the pass criterion; "
            "they are rewritten by the run that produced the PDF."
        ),
    }


def check_rebuild_proof(scratch: Path) -> dict:
    """Decisive freshness proof: rebuild from current sources, diff the text layer.

    Never touches paper/main.pdf. The committed PDF is the artefact of record;
    this builds a throwaway copy elsewhere and compares what a reader would see.
    """
    for tool in ("pdflatex", "bibtex", "pdftotext"):
        _need(tool)

    work = scratch / "tier0_rebuild"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(PAPER, work)
    for junk in ("main.pdf", "main.aux", "main.bbl", "main.log", "main.out",
                 "main.fdb_latexmk", "main.fls"):
        (work / junk).unlink(missing_ok=True)

    for cmd in (
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
    ):
        subprocess.run(cmd, cwd=work, capture_output=True, text=True)

    fresh_pdf = work / "main.pdf"
    if not fresh_pdf.exists():
        raise CannotMeasure("rebuild produced no PDF")

    committed_text = _run(["pdftotext", str(PDF), "-"])
    fresh_text = _run(["pdftotext", str(fresh_pdf), "-"])

    def pages_of(path: Path) -> int:
        m = re.search(r"^Pages:\s+(\d+)", _run(["pdfinfo", str(path)]), re.M)
        return int(m.group(1)) if m else -1

    committed_pages, fresh_pages = pages_of(PDF), pages_of(fresh_pdf)
    identical = committed_text == fresh_text

    return {
        "check": "T0-a build freshness (rebuild proof)",
        "method": "rebuild current sources in a scratch copy, diff the PDF text layer",
        "committed_pages": committed_pages,
        "rebuilt_pages": fresh_pages,
        "committed_words": len(committed_text.split()),
        "rebuilt_words": len(fresh_text.split()),
        "text_layer_identical": identical,
        "scratch_dir": str(work),
        "pass": identical and committed_pages == fresh_pages,
        "note": (
            "paper/main.pdf is never overwritten by this check. A byte-identical "
            "text layer proves the committed PDF is the build of the sources on disk."
        ),
    }


# ---------------------------------------------------------------- T0-b / T0-d

BBOX_WORD = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)"')
PAGE_OPEN = re.compile(r'<page width="([\d.]+)" height="([\d.]+)">')
INCLUDE = re.compile(r"\\includegraphics\[(?P<opts>[^\]]*)\]\{(?P<path>[^}]+)\}")
WIDTH_OPT = re.compile(r"width\s*=\s*([\d.]+)\s*\\(linewidth|columnwidth|textwidth)")


def _pdf_page_boxes() -> list[dict]:
    _need("pdftotext")
    xml = _run(["pdftotext", "-bbox", str(PDF), "-"])
    pages: list[dict] = []
    current: dict | None = None
    for line in xml.splitlines():
        pm = PAGE_OPEN.search(line)
        if pm:
            if current:
                pages.append(current)
            current = {
                "width": float(pm.group(1)),
                "height": float(pm.group(2)),
                "x_min": None,
                "x_max": None,
                "y_max": None,
            }
            continue
        wm = BBOX_WORD.search(line)
        if wm and current is not None:
            x0, _y0, x1, y1 = (float(g) for g in wm.groups())
            current["x_min"] = x0 if current["x_min"] is None else min(current["x_min"], x0)
            current["x_max"] = x1 if current["x_max"] is None else max(current["x_max"], x1)
            current["y_max"] = y1 if current["y_max"] is None else max(current["y_max"], y1)
    if current:
        pages.append(current)
    if not pages:
        raise CannotMeasure("pdftotext -bbox produced no pages")
    return pages


def _text_block_width(pages: list[dict]) -> float:
    """Full text-block width, from the widest measured extent across pages."""
    xs = [p["x_max"] - p["x_min"] for p in pages if p["x_min"] is not None]
    if not xs:
        raise CannotMeasure("no word boxes found; cannot measure text block")
    return max(xs)


def check_page_ledger() -> dict:
    pages = _pdf_page_boxes()
    block_w_bp = _text_block_width(pages)
    # pdftotext reports PostScript big points (1/72 in); LaTeX points are 1/72.27 in.
    block_w = block_w_bp * BP_TO_PT
    # Two-column layout: block = 2*column + gutter. IEEEtran's \columnsep is 1pc = 12pt,
    # NOT the 0.2in (14.45pt) this check assumed until 2026-08-21. The old constant made
    # every derived width about 1% low and put the column at 249.814pt instead of 252.
    gutter_pt = 12.0
    column_w = (block_w - gutter_pt) / 2.0

    baselines = [round(p["y_max"] * BP_TO_PT, 3) for p in pages]
    deepest = max(baselines)
    last_page = baselines[-1]
    # Slack is how much further the LAST page could run before matching the
    # deepest full page. Identical values mean the page is full: zero slack.
    slack_pt = round(deepest - last_page, 3)

    return {
        "check": "T0-b page ledger",
        "pages": len(pages),
        "page_size_bp": [pages[0]["width"], pages[0]["height"]],
        "is_a4": abs(pages[0]["width"] - 595.276) < 1 and abs(pages[0]["height"] - 841.89) < 1,
        "text_block_width_pt": round(block_w, 3),
        "body_baseline_pt": BODY_BASELINE_PT,
        "column_width_pt": round(column_w, 3),
        "last_baseline_per_page": baselines,
        "deepest_baseline_pt": deepest,
        "last_page_baseline_pt": last_page,
        "available_points": slack_pt,
        "pass": True,
        "note": (
            "available_points is the vertical room left on the final page only. "
            "Zero means the last page is as full as the fullest page; any addition "
            "spills to a new page."
        ),
    }


def _figure_includes() -> list[dict]:
    out = []
    for tex in sorted((PAPER / "sections").glob("*.tex")) + [PAPER / "main.tex"]:
        if not tex.exists():
            continue
        for lineno, line in enumerate(tex.read_text(errors="replace").splitlines(), 1):
            m = INCLUDE.search(line)
            if not m:
                continue
            wm = WIDTH_OPT.search(m.group("opts"))
            out.append(
                {
                    "tex": str(tex.relative_to(ROOT)),
                    "line": lineno,
                    "graphic": m.group("path"),
                    "width_fraction": float(wm.group(1)) if wm else None,
                    "width_unit": wm.group(2) if wm else None,
                }
            )
    return out


def _min_font_size_pt(fig_pdf: Path) -> float | None:
    """Representative smallest text height in a figure's text layer, in native points.

    Uses the 10th percentile of word-box heights, not the raw minimum. A bare
    minimum reports the bounding box of whatever punctuation happens to be
    smallest -- a slash in a rotated axis label measured 1.07pt in one figure
    here, which is the height of a slash, not a font size. It was reported as a
    font size once; the percentile makes that failure mode impossible.
    """
    try:
        xml = _run(["pdftotext", "-bbox", str(fig_pdf), "-"])
    except CannotMeasure:
        return None
    heights = sorted(
        float(m.group(4)) - float(m.group(2)) for m in BBOX_WORD.finditer(xml)
    )
    heights = [h for h in heights if h > 0.5]
    if not heights:
        return None
    idx = max(0, int(0.10 * (len(heights) - 1)))
    return round(heights[idx], 3)


def check_render_audit(column_width_pt: float) -> dict:
    _need("pdfinfo")
    figs = _figure_includes()
    rows, failures = [], []
    for f in figs:
        gpath = (PAPER / f["graphic"]).resolve()
        if not gpath.exists():
            rows.append({**f, "error": "graphic file not found"})
            failures.append(f["graphic"])
            continue
        info = _run(["pdfinfo", str(gpath)])
        sm = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info)
        if not sm:
            rows.append({**f, "error": "could not read native page size"})
            failures.append(f["graphic"])
            continue
        native_w, native_h = float(sm.group(1)), float(sm.group(2))
        # Inside a single-column figure env, \linewidth == \columnwidth.
        base = column_width_pt
        rendered_w = (f["width_fraction"] or 1.0) * base
        scale = rendered_w / native_w
        native_min_font = _min_font_size_pt(gpath)
        effective_font = round(native_min_font * scale, 3) if native_min_font else None
        col_fraction = round(rendered_w / column_width_pt, 4)
        ok = col_fraction >= 0.9 and (effective_font is None or effective_font >= 6.0)
        if not ok:
            failures.append(f["graphic"])
        rows.append(
            {
                **f,
                "native_width_pt": native_w,
                "native_height_pt": native_h,
                "rendered_width_pt": round(rendered_w, 3),
                "rendered_width_in": round(rendered_w / 72.0, 3),
                "rendered_height_pt": round(rendered_w * native_h / native_w, 3),
                "scale": round(scale, 4),
                "column_fraction": col_fraction,
                "native_min_text_pt": native_min_font,
                "effective_min_text_pt": effective_font,
                "pass": ok,
            }
        )
    return {
        "check": "T0-d render audit",
        "criterion": "column_fraction >= 0.90 and effective_min_text_pt >= 6.0",
        "figures": rows,
        "failing": failures,
        "pass": not failures,
    }


def check_fonts() -> dict:
    """T0-f: is the PDF acceptable to the submission system, not merely readable?

    Added after a desk-reviewer pass found three Type 3 DejaVuSans instances that
    every other check in this tier walked straight past. IEEE PDF eXpress rejects
    Type 3 fonts; matplotlib emits them by default. The rest of Tier 0 asks whether
    the document renders. This asks whether it can be submitted at all.
    """
    _need("pdffonts")
    out = _run(["pdffonts", str(PDF)])
    rows = []
    for line in out.splitlines()[2:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        # name, then a possibly multi-word type, then emb/sub/uni flags
        name = parts[0]
        ftype = " ".join(parts[1:-5]) if len(parts) > 6 else parts[1]
        rows.append({
            "name": name,
            "type": ftype,
            "embedded": parts[-5] == "yes",
            "unicode_mapped": parts[-3] == "yes",
        })
    type3 = [r for r in rows if "Type 3" in r["type"]]
    not_embedded = [r for r in rows if not r["embedded"]]
    return {
        "check": "T0-f font submittability",
        "criterion": "no Type 3 fonts, every font embedded",
        "fonts": rows,
        "type3": type3,
        "not_embedded": not_embedded,
        "pass": not type3 and not not_embedded,
        "note": (
            "Type 3 usually arrives from matplotlib. Fix at the source with "
            "matplotlib.rcParams['pdf.fonttype'] = 42 and regenerate the figures; "
            "editing the .tex will not remove it."
        ),
    }


# ---------------------------------------------------------------- T0-c


def check_codepoints() -> dict:
    _need("pdftotext")
    sources = {
        "pdf_text_layer": _run(["pdftotext", str(PDF), "-"]),
    }
    bib = PAPER / "bib" / "refs.bib"
    if bib.exists():
        sources["refs.bib"] = bib.read_text(errors="replace")
    for tex in sorted((PAPER / "sections").glob("*.tex")) + [PAPER / "main.tex"]:
        if tex.exists():
            sources[str(tex.relative_to(ROOT))] = tex.read_text(errors="replace")

    found: dict[str, dict] = {}
    banned_hits: list[dict] = []
    for label, text in sources.items():
        for ch in set(text):
            if ord(ch) < 128:
                continue
            if ch in BANNED:
                banned_hits.append({"source": label, "codepoint": f"U+{ord(ch):04X}", "name": BANNED[ch]})
                continue
            key = f"U+{ord(ch):04X}"
            entry = found.setdefault(
                key,
                {
                    "char": ch,
                    "unicode_name": unicodedata.name(ch, "UNNAMED"),
                    "whitelisted": ch in CODEPOINT_WHITELIST or ch in CODEPOINT_NOTE,
                    "advisory": ch in CODEPOINT_NOTE,
                    "reason": CODEPOINT_WHITELIST.get(ch) or CODEPOINT_NOTE.get(ch),
                    "sources": [],
                },
            )
            if label not in entry["sources"]:
                entry["sources"].append(label)

    unlisted = {k: v for k, v in found.items() if not v["whitelisted"]}
    advisory = {k: v for k, v in found.items() if v.get("advisory")}
    return {
        "check": "T0-c codepoint audit",
        "distinct_non_ascii": len(found),
        "codepoints": dict(sorted(found.items())),
        "not_whitelisted": dict(sorted(unlisted.items())),
        "advisory": dict(sorted(advisory.items())),
        "banned_glyph_hits": banned_hits,
        "pass": not banned_hits and not unlisted,
    }


# ---------------------------------------------------------------- T0-e


def check_venue_source() -> dict:
    main = (PAPER / "main.tex").read_text(errors="replace")
    cls = re.search(r"\\documentclass\[([^\]]*)\]\{([^}]+)\}", main)
    logtext = LOG.read_text(errors="replace") if LOG.exists() else ""
    bib = PAPER / "bib" / "refs.bib"
    n_entries = len(re.findall(r"^@\w+\{", bib.read_text(errors="replace"), re.M)) if bib.exists() else None
    return {
        "check": "T0-e venue compliance (source side)",
        "documentclass_options": cls.group(1) if cls else None,
        "documentclass": cls.group(2) if cls else None,
        "a4paper_set": bool(cls and "a4paper" in cls.group(1)),
        "conference_mode": bool(cls and "conference" in cls.group(1)),
        "override_lockouts": "\\IEEEoverridecommandlockouts" in main,
        "blind_toggle_default": "\\blindtrue" in main,
        "undefined_references": logtext.lower().count("undefined"),
        "overfull_boxes": logtext.count("Overfull"),
        "bib_entries": n_entries,
        "pass": bool(cls and "a4paper" in cls.group(1) and "conference" in cls.group(1))
        and logtext.lower().count("undefined") == 0
        and logtext.count("Overfull") == 0,
        "note": "Live-CFP cross-check (page limit, anonymity wording) is a separate manual step.",
    }


# ---------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument(
        "--rebuild-proof",
        metavar="SCRATCH_DIR",
        help="prove build freshness by rebuilding into SCRATCH_DIR (paper/ is not touched)",
    )
    args = ap.parse_args()

    results: dict[str, dict] = {}
    could_not_measure: list[str] = []

    ledger = None
    checks = []
    if args.rebuild_proof:
        scratch = Path(args.rebuild_proof)
        checks.append(("build_freshness", lambda: check_rebuild_proof(scratch)))
    else:
        checks.append(("build_freshness", check_build_freshness))
    for name, fn in checks + [
        ("page_ledger", check_page_ledger),
        ("codepoints", check_codepoints),
        ("venue", check_venue_source),
        ("fonts", check_fonts),
    ]:
        try:
            results[name] = fn()
            if name == "page_ledger":
                ledger = results[name]
        except CannotMeasure as exc:
            results[name] = {"check": name, "error": str(exc), "pass": None}
            could_not_measure.append(name)

    try:
        col = ledger["column_width_pt"] if ledger else FALLBACK_COLUMN_WIDTH_PT
        results["render"] = check_render_audit(col)
    except CannotMeasure as exc:
        results["render"] = {"check": "T0-d render audit", "error": str(exc), "pass": None}
        could_not_measure.append("render")

    failed = [k for k, v in results.items() if v.get("pass") is False]

    for key, res in results.items():
        status = {True: "PASS", False: "FAIL", None: "CANNOT-MEASURE"}[res.get("pass")]
        print(f"[{status:14s}] {res.get('check', key)}")
        if res.get("error"):
            print(f"                 {res['error']}")

    if ledger:
        print()
        print(f"  pages={ledger['pages']} a4={ledger['is_a4']} "
              f"column_width={ledger['column_width_pt']}pt "
              f"available_points={ledger['available_points']}")
    if results.get("render", {}).get("figures"):
        print()
        for f in results["render"]["figures"]:
            if "rendered_width_in" not in f:
                continue
            print(f"  {f['graphic']:44s} {f['rendered_width_in']:.2f}in "
                  f"({f['column_fraction']*100:.0f}% col, scale {f['scale']:.2f}, "
                  f"min text {f['effective_min_text_pt']}pt) "
                  f"{'ok' if f['pass'] else 'FAIL'}")

    if args.write_manifest:
        out = ROOT / "results" / "tier0_premises.json"
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}")

    if could_not_measure:
        print(f"\nCANNOT MEASURE: {', '.join(could_not_measure)} -- exit 2, never a pass")
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
