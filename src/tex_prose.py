"""Reads OUR OWN paper directly from its LaTeX source, for the corpus-comparison study.

Unlike the reference NDSS-corpus instrument this project's design is modelled on, we hold
our own paper's source, so "ours" never needs a PDF or marker round-trip: this module strips
paper/sections/*.tex + paper/main.tex down to the same plain-prose-with-citation-tokens shape
src/reference_corpus.py produces for the marker-md corpus, so both sides can go through the
same src/prose_metrics.measure_prose()/detect_conventions().

ASSERT ONLY, NEVER GENERATE: this reads the live .tex files each call. If the section list
below stops matching paper/main.tex's actual \\input order, that is a bug to fix here, not a
number to patch in the report.
"""
from __future__ import annotations

import pathlib
import re

from src.prose_metrics import CITE_TOKEN, detect_conventions, measure_prose, words_in

REPO = pathlib.Path(__file__).resolve().parent.parent
PAPER_DIR = REPO / "paper"

# Confirmed against paper/main.tex's \input order this session (2026-08-16): Related Work
# runs BEFORE Methods. If main.tex's \input list changes, update this to match.
ABSTRACT_FILE = "sections/00_abstract.tex"
BODY_FILES = [
    "sections/01_introduction.tex",
    "sections/04_related_work.tex",
    "sections/02_methods.tex",
    "sections/03_results.tex",
    "sections/05_discussion_conclusion.tex",
]

_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_ENV_DROP_RE = re.compile(
    r"\\begin\{(table\*?|figure\*?|algorithm\*?|tabular\*?)\}.*?\\end\{\1\}", re.DOTALL
)
_MATH_DISPLAY_RE = re.compile(r"\\\[.*?\\\]", re.DOTALL)
_MATH_INLINE_RE = re.compile(r"\$[^$]*\$")
_CITE_CMD_RE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^}]*)\}")
_DROP_CMD_RE = re.compile(
    r"\\(ref|label|input|includegraphics|usepackage|documentclass|bibliography\w*|"
    r"IEEEauthorblock\w*|IEEEkeywords|maketitle|setlength|newcommand|renewcommand|"
    r"IEEEoverridecommandlockouts)\b[^\n]*"
)
_ARG_KEEP_CMD_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}")
_BARE_CMD_RE = re.compile(r"\\[a-zA-Z]+\*?")
_BRACES_RE = re.compile(r"[{}]")
_ITEMIZE_ITEM_RE = re.compile(r"\\item\b")
_BEGIN_ITEMIZE_RE = re.compile(r"\\begin\{(itemize|enumerate)\}")
_LIMITATIONS_HEAD_RE = re.compile(r"\\(sub)*section\*?\{[^}]*Limitations?[^}]*\}", re.IGNORECASE)


def _read(rel: str, paper_dir: pathlib.Path | None = None) -> str:
    return ((paper_dir or PAPER_DIR) / rel).read_text(encoding="utf-8")


def _mark_citations(text: str) -> str:
    def repl(m: re.Match) -> str:
        keys = [k for k in m.group(1).split(",") if k.strip()]
        return CITE_TOKEN * max(1, len(keys))

    return _CITE_CMD_RE.sub(repl, text)


def _strip_tex(text: str) -> str:
    text = _COMMENT_RE.sub("", text)
    text = _ENV_DROP_RE.sub("", text)
    text = _MATH_DISPLAY_RE.sub("", text)
    text = _MATH_INLINE_RE.sub("", text)
    text = _mark_citations(text)
    text = _DROP_CMD_RE.sub("", text)
    for _ in range(2):  # two passes: catches one level of nested wrapping commands
        text = _ARG_KEEP_CMD_RE.sub(r"\1", text)
    text = _BARE_CMD_RE.sub("", text)
    text = _BRACES_RE.sub("", text)
    return text


def _paragraphs(text: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def read_paper(paper_dir: pathlib.Path | None = None,
               body_files: list[str] | None = None,
               abstract_file: str | None = None) -> dict:
    """Measure a manuscript's prose.

    The three arguments default to the six-page ICCIT draft in paper/, which is what
    scripts/30 has always measured, so existing callers are unaffected. They exist because the
    IEEE Access manuscript in paper_access/ has a different section count and a different
    \\input order, and duplicating this module's TeX stripper to read it would give two
    implementations that could silently drift.
    """
    pd = paper_dir or PAPER_DIR
    bf = body_files or BODY_FILES
    af = abstract_file or ABSTRACT_FILE
    body_raw = "\n\n".join(_read(f, pd) for f in bf)
    abstract_raw = _read(af, pd)

    has_itemize = any(_BEGIN_ITEMIZE_RE.search(_read(f, pd)) for f in bf)
    has_limitations_heading = any(_LIMITATIONS_HEAD_RE.search(_read(f, pd)) for f in bf)

    body_marked = _strip_tex(body_raw)
    paragraphs = _paragraphs(body_marked)
    abstract_text = _strip_tex(abstract_raw).replace(CITE_TOKEN, "")

    return {
        "body_marked": body_marked,
        "paragraphs": paragraphs,
        "abstract_text": abstract_text,
        "has_early_bullet_list": has_itemize,
        "has_limitations_heading": has_limitations_heading,
    }


def measure_paper(paper_dir: pathlib.Path | None = None,
                  body_files: list[str] | None = None,
                  abstract_file: str | None = None) -> dict:
    parsed = read_paper(paper_dir, body_files, abstract_file)
    m = measure_prose(parsed["body_marked"], parsed["paragraphs"])
    conv = detect_conventions(parsed["body_marked"].replace(CITE_TOKEN, ""), parsed["abstract_text"])
    conv["has_limitations_heading"] = parsed["has_limitations_heading"]
    conv["has_early_bullet_list"] = parsed["has_early_bullet_list"]
    conv["abstract_words"] = len(words_in(parsed["abstract_text"]))
    conv["has_references_heading"] = True  # bibliography always present
    return {**m, **conv}
