"""Shared, format-agnostic prose metrics for the corpus-comparison study.

ASSERT ONLY, NEVER GENERATE: every function here measures text handed to it. Nothing in
this module estimates or imputes a value for text it cannot see.

Callers (src/reference_corpus.py for the marker-md corpus, src/tex_prose.py for our own
LaTeX source) are responsible for stripping their own markup down to plain prose text with
citation occurrences replaced by CITE_TOKEN, so this module never has to know whether a
citation looked like ``[19]`` or ``\\cite{Foo2020}``.

Passive voice is measured with a real dependency parse (`src/passive_voice.py`, spaCy), not
a regex proxy -- see that module's docstring for why the old regex was replaced.
"""
from __future__ import annotations

import re
import statistics as st

from src.passive_voice import count_passive_clauses

CITE_TOKEN = "‹CITE›"

_FIRST_PERSON_RE = re.compile(r"\b(we|our|us)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

_ABBREV = ("e.g.", "i.e.", "et al.", "Fig.", "Eq.", "Sec.", "cf.", "vs.",
           "approx.", "Dr.", "Prof.", "No.", "Ref.", "resp.", "etc.")


def _protect_abbrev(text: str) -> str:
    for a in _ABBREV:
        text = text.replace(a, a.replace(".", "․"))
    return text


def _restore_abbrev(text: str) -> str:
    return text.replace("․", ".")


def split_sentences(text: str) -> list[str]:
    protected = _protect_abbrev(text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(‹])", protected)
    return [_restore_abbrev(p).strip() for p in parts if _restore_abbrev(p).strip()]


def words_in(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def measure_prose(text_with_cite_tokens: str, paragraphs: list[str] | None = None) -> dict:
    """Core numeric metrics. `paragraphs`, if given, are pre-segmented blocks (still may
    carry CITE_TOKEN) that the caller judged to be real prose paragraphs, not captions,
    table rows, or bullet items."""
    n_cite = text_with_cite_tokens.count(CITE_TOKEN)
    plain = text_with_cite_tokens.replace(CITE_TOKEN, "")
    sentences = split_sentences(plain)
    swc = [len(words_in(s)) for s in sentences]
    words = words_in(plain)
    n_words = len(words)
    n_first_person = len(_FIRST_PERSON_RE.findall(plain))
    n_passive = count_passive_clauses(plain)

    out = {
        "words": n_words,
        "sentences": len(sentences),
        "mean_words_per_sentence": round(st.mean(swc), 2) if swc else 0.0,
        "median_words_per_sentence": (st.median(swc) if swc else 0),
        "pct_over_35w": round(100 * sum(1 for w in swc if w > 35) / len(swc), 1) if swc else 0.0,
        "max_words_per_sentence": max(swc) if swc else 0,
        "citations_per_1k": round(1000 * n_cite / n_words, 2) if n_words else 0.0,
        "first_person_per_1k": round(1000 * n_first_person / n_words, 2) if n_words else 0.0,
        "passive_per_1k_words": round(1000 * n_passive / n_words, 2) if n_words else 0.0,
    }
    if paragraphs is not None:
        pw = [len(words_in(p.replace(CITE_TOKEN, ""))) for p in paragraphs]
        pw = [w for w in pw if w >= 15]
        out["paragraphs"] = len(pw)
        out["mean_words_per_paragraph"] = round(st.mean(pw), 1) if pw else 0.0
    return out


# ---- structural/organisational conventions, operate on already-stripped plain text -------

_SECTION_OPENER_RE = re.compile(r"in this section,?\s+we\b", re.IGNORECASE)
_SIG_TEST_RE = re.compile(
    r"\b(t-test|wilcoxon|mann-whitney|anova|chi-square|paired t|p\s*[<=]\s*0\.0\d)",
    re.IGNORECASE,
)
_CI_MENTION_RE = re.compile(
    r"(\[\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\])|(\b95\s*%\s*CI\b)|(confidence interval)",
    re.IGNORECASE,
)
_ET_AL_RE = re.compile(r"[A-Z][a-zA-Z\-]+\s+et al\.")
_HOWEVER_RE = re.compile(r"\bhowever\b", re.IGNORECASE)
_IN_THIS_PAPER_RE = re.compile(r"\bin this paper\b", re.IGNORECASE)


def detect_conventions(plain_body_text: str, abstract_text: str | None) -> dict:
    ab = abstract_text or ""
    return {
        "n_section_opener_phrase": len(_SECTION_OPENER_RE.findall(plain_body_text)),
        "reports_ci_or_interval": bool(_CI_MENTION_RE.search(plain_body_text)),
        "names_significance_test": bool(_SIG_TEST_RE.search(plain_body_text)),
        "abstract_names_prior_author": bool(_ET_AL_RE.search(ab)),
        "abstract_has_however_gap": bool(_HOWEVER_RE.search(ab)),
        "abstract_has_in_this_paper_pivot": bool(_IN_THIS_PAPER_RE.search(ab)),
    }


def _r(x):
    return round(x, 2) if isinstance(x, float) else x


def band(vals: list[float]) -> dict:
    s = sorted(vals)
    return {"n": len(s), "min": _r(s[0]), "median": _r(st.median(s)), "max": _r(s[-1])}


def percentile_of(x: float, vals: list[float]) -> int:
    s = sorted(vals)
    return round(100 * sum(1 for v in s if v <= x) / len(s))
