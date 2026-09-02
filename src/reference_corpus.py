"""Reads the marker-converted reference corpus at references/external_marker_md/.

50 papers across two topic sub-corpora (27 backdoor/adversarial-ML, 23 NIDS/tabular-ML),
copied in from two other course projects for this corpus-comparison study. This is a
TOPIC-matched corpus, not a venue-matched one -- see notes/20260816-report-corpus-tables.md
for what that caveat means before quoting anything measured here.

ASSERT ONLY, NEVER GENERATE: a paper this module cannot parse raises CannotMeasure rather
than being silently skipped or scored as zero.
"""
from __future__ import annotations

import pathlib
import re

from src.prose_metrics import CITE_TOKEN, detect_conventions, measure_prose, words_in

REPO = pathlib.Path(__file__).resolve().parent.parent
CORPUS_ROOT = REPO / "references" / "external_marker_md"
SUBCORPORA = {
    "backdoor_ml_corpus": "backdoor / adversarial-ML corpus",
    "nids_ml_corpus": "NIDS / tabular-ML corpus",
}


class CannotMeasure(Exception):
    pass


_IMAGE_LINE_RE = re.compile(r"^!\[\][^\n]*$", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BOLD_ITALIC_RE = re.compile(r"\*{1,3}|_{1,3}")
_NUMERIC_CITE_RE = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")
_REFERENCES_HEAD_RE = re.compile(
    r"^#{0,6}\s*\**\s*(REFERENCES?|BIBLIOGRAPHY)\s*\**\s*$", re.IGNORECASE | re.MULTILINE
)
_ABSTRACT_HEAD_RE = re.compile(r"\*{0,2}ABSTRACT\*{0,2}", re.IGNORECASE)
_INDEX_TERMS_HEAD_RE = re.compile(
    r"\*{0,2}(INDEX TERMS|KEYWORDS|KEY WORDS)\*{0,2}", re.IGNORECASE
)
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
_BULLET_LINE_RE = re.compile(r"^\s*[-*•]\s+\S")
_BULLET_RUN_RE = re.compile(r"(?:^\s*[-*•]\s+.+$\n?){3,}", re.MULTILINE)
_LIMITATIONS_HEAD_RE = re.compile(
    r"^#{1,6}\s*\**\s*[IVXLC0-9.\s]*Limitations?\**\s*$", re.IGNORECASE | re.MULTILINE
)


def iter_corpus_papers():
    """Yield (paper_id, subcorpus_key, md_path) for every 1-md-file paper folder."""
    for sub in sorted(SUBCORPORA):
        d = CORPUS_ROOT / sub
        if not d.is_dir():
            continue
        for folder in sorted(p for p in d.iterdir() if p.is_dir()):
            mds = list(folder.glob("*.md"))
            if len(mds) != 1:
                continue
            yield folder.name, sub, mds[0]


def _strip_markup(text: str) -> str:
    text = _IMAGE_LINE_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    return text


def _mark_citations(text: str) -> str:
    return _NUMERIC_CITE_RE.sub(CITE_TOKEN, text)


def _split_paragraphs(body_no_headings: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", body_no_headings)
    out = []
    for b in blocks:
        b = b.strip()
        if not b or b.startswith("|") or _BULLET_LINE_RE.match(b):
            continue
        out.append(_BOLD_ITALIC_RE.sub("", b))
    return out


def read_paper(md_path: pathlib.Path) -> dict:
    raw = md_path.read_text(encoding="utf-8", errors="replace")
    if len(raw.strip()) < 500:
        raise CannotMeasure(f"{md_path}: suspiciously short ({len(raw.strip())} chars)")

    cleaned = _strip_markup(raw)
    refs_m = _REFERENCES_HEAD_RE.search(cleaned)
    body_raw = cleaned[: refs_m.start()] if refs_m else cleaned
    body_marked = _mark_citations(body_raw)
    body_no_headings = _HEADING_LINE_RE.sub("", body_marked)
    paragraphs = _split_paragraphs(body_no_headings)
    body_plain = _BOLD_ITALIC_RE.sub("", body_no_headings).replace(CITE_TOKEN, "")

    abs_m = _ABSTRACT_HEAD_RE.search(cleaned)
    abstract_text = None
    if abs_m:
        idx_m = _INDEX_TERMS_HEAD_RE.search(cleaned, abs_m.end())
        later_heads = list(_HEADING_LINE_RE.finditer(cleaned, abs_m.end()))
        end = idx_m.start() if idx_m else (later_heads[0].start() if later_heads else None)
        if end and end > abs_m.end():
            abstract_text = _BOLD_ITALIC_RE.sub("", cleaned[abs_m.end() : end]).strip()

    has_early_bullets = False
    early_window = body_raw[: max(1, int(len(body_raw) * 0.40))]
    if _BULLET_RUN_RE.search(early_window):
        has_early_bullets = True

    return {
        "path": str(md_path.relative_to(REPO)) if md_path.is_relative_to(REPO) else str(md_path),
        "raw_chars": len(raw),
        "has_references_heading": refs_m is not None,
        "body_marked": body_marked,
        "paragraphs": paragraphs,
        "abstract_text": abstract_text,
        "cleaned_full": cleaned,
        "body_plain": body_plain,
        "has_early_bullet_list": has_early_bullets,
    }


def measure_paper(parsed: dict) -> dict:
    m = measure_prose(parsed["body_marked"], parsed["paragraphs"])
    conv = detect_conventions(parsed["body_plain"], parsed["abstract_text"])
    conv["has_limitations_heading"] = bool(_LIMITATIONS_HEAD_RE.search(parsed["cleaned_full"]))
    conv["has_early_bullet_list"] = parsed["has_early_bullet_list"]
    conv["abstract_words"] = (
        len(words_in(parsed["abstract_text"])) if parsed["abstract_text"] else None
    )
    conv["has_references_heading"] = parsed["has_references_heading"]
    return {**m, **conv}
