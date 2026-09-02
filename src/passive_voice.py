"""Real, dependency-parse-based passive-voice detection.

Replaces the old `passive_per_1k_words_proxy` regex heuristic
(`\\b(?:is|are|was|were|be|been|being)\\s+\\w+ed\\b`), which both undercounted (irregular
past participles like "shown", "given", "taken", "written" don't end in `-ed`, so the regex
never saw them) and overcounted (some `be + -ed word` phrases are adjectival, not verbal,
and the regex couldn't tell the difference).

Uses spaCy's dependency parse instead: a finite passive clause is uniquely marked by a
`nsubjpass` dependency (or the newer Universal-Dependencies-style `nsubj:pass`, in case a
future model version relabels it) on its subject -- exactly one per passive clause, correctly
catching irregular participles and correctly distinguishing genuine passive constructions
from adjectival look-alikes, because the parser resolves that from full sentence structure,
not a fixed word-ending pattern.

Requires `spacy` + `en_core_web_sm` (installed this session via
`uv pip install --python .venv/bin/python spacy` and
`.venv/bin/python -m spacy download en_core_web_sm`; not yet added to requirements.txt).

ASSERT ONLY, NEVER GENERATE: `nlp()` is run only on text handed to it.
"""
from __future__ import annotations

import functools

_PASSIVE_SUBJ_DEPS = {"nsubjpass", "nsubj:pass", "csubjpass", "csubj:pass"}


@functools.lru_cache(maxsize=1)
def _nlp():
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    nlp.max_length = 2_000_000
    return nlp


def count_passive_clauses(text: str) -> int:
    """Number of finite passive clauses in `text` -- each has exactly one passive subject."""
    text = text.strip()
    if not text:
        return 0
    doc = _nlp()(text)
    return sum(1 for tok in doc if tok.dep_ in _PASSIVE_SUBJ_DEPS)
