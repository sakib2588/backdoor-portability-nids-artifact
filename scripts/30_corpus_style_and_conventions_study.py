#!/usr/bin/env python3
r"""Measure this paper's prose style and structural conventions against the 50-paper
topic-matched reference corpus (references/external_marker_md/).

Tier 1 (prose style, fully mechanical): word/sentence counts, mean/median words per
sentence, % sentences over 35 words, longest sentence, paragraphs, words/paragraph,
citations per 1k words, first-person per 1k, passive-voice regex proxy per 1k.

Tier 2 (structural conventions, regex heuristics): "In this section, we..." openers,
CI/interval reporting, named significance tests, prior-author naming in the abstract,
abstract's "however" gap / "in this paper" pivot, a named Limitations heading, and an
early bullet list (contributions-bullets proxy).

Ours is measured directly from paper/sections/*.tex + paper/main.tex (src/tex_prose.py) --
no PDF or marker round-trip needed for our own paper. Corpus papers are measured from the
already-converted marker .md files (src/reference_corpus.py).

Ours is NEVER a member of its own band. Every numeric row prints a percentile.

CORPUS CAVEAT, load-bearing: these 50 papers are TOPIC-matched (backdoor/adversarial-ML and
NIDS/tabular-ML), not venue-matched. They span NDSS, USENIX Security, IEEE S&P, NeurIPS,
ACSAC, ECML, ICLR, assorted journals, and un-peer-reviewed arXiv preprints, at very different
page-length conventions than ICCIT's 6pp conference format. Read every band below as "how
this literature writes", never as "how ICCIT papers write" or "the ICCIT norm".

Contract: ASSERT ONLY, NEVER GENERATE. A bare run measures and prints; only
--write-manifest writes results/corpus_style_and_conventions_study.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.prose_metrics import band, percentile_of  # noqa: E402
from src.reference_corpus import (  # noqa: E402
    SUBCORPORA,
    CannotMeasure,
    iter_corpus_papers,
    measure_paper,
    read_paper,
)
from src.tex_prose import measure_paper as measure_ours  # noqa: E402

NUMERIC_ROWS = [
    ("words", "body words"),
    ("sentences", "sentences"),
    ("mean_words_per_sentence", "mean words / sentence"),
    ("median_words_per_sentence", "median words / sentence"),
    ("pct_over_35w", "% sentences over 35w"),
    ("max_words_per_sentence", "longest sentence"),
    ("paragraphs", "paragraphs"),
    ("mean_words_per_paragraph", "mean words / paragraph"),
    ("citations_per_1k", "citations per 1k"),
    ("first_person_per_1k", "first person per 1k"),
    ("passive_per_1k_words", "passive per 1k"),
    ("abstract_words", "abstract words"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest", default="results/corpus_style_and_conventions_study.json"
    )
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    papers = list(iter_corpus_papers())
    if not papers:
        print(f"no corpus papers found under {REPO / 'references' / 'external_marker_md'}")
        print("RESULT: UNRESOLVED")
        return 2

    print("=== PROSE STYLE + STRUCTURAL CONVENTIONS, 50-PAPER TOPIC CORPUS ===")
    print(f"  corpus: {len(papers)} papers "
          f"({sum(1 for _, s, _ in papers if s == 'backdoor_ml_corpus')} backdoor/adv-ML, "
          f"{sum(1 for _, s, _ in papers if s == 'nids_ml_corpus')} NIDS/tabular-ML)")
    print("  TOPIC-matched, NOT venue-matched -- do not read any band below as \"the ICCIT")
    print("  norm\". Venues include NDSS, USENIX Security, IEEE S&P, NeurIPS, ACSAC, ECML,")
    print("  ICLR, assorted journals, and un-peer-reviewed arXiv preprints.")

    results, failed = {}, []
    for paper_id, sub, md_path in papers:
        try:
            results[paper_id] = {
                "subcorpus": sub,
                **measure_paper(read_paper(md_path)),
            }
        except CannotMeasure as e:
            failed.append((paper_id, str(e)))
    if failed:
        print("\n  -- COULD NOT MEASURE --")
        for n, e in failed:
            print(f"     {n}: {e}")

    ours = measure_ours()

    print(f"\n  measured {len(results)}/{len(papers)} corpus papers"
          + (f"  ({len(failed)} failed)" if failed else ""))

    print(f"\n  -- ours against the {len(results)}-paper band --")
    print(f"     {'metric':<38}{'ours':>10}{'min':>9}{'median':>9}{'max':>9}{'pct':>5}")
    bands_out = {}
    for key, label in NUMERIC_ROWS:
        vals = [v[key] for v in results.values() if v.get(key) is not None]
        o = ours.get(key)
        if not vals or o is None:
            print(f"     {label:<38}{'--':>10}  (insufficient data)")
            continue
        b = band(vals)
        pc = percentile_of(o, vals)
        bands_out[key] = {**b, "ours": o, "percentile": pc}
        print(f"     {label:<38}{o:>10}{b['min']:>9}{b['median']:>9}{b['max']:>9}{pc:>5}")

    print("\n  -- structural conventions --")
    conv_out = {}
    conv_rows = [
        ("n_section_opener_phrase", "\"In this section, we...\" openers", "count"),
        ("reports_ci_or_interval", "reports CI / bracketed interval", "bool"),
        ("names_significance_test", "names a significance test", "bool"),
        ("abstract_names_prior_author", "abstract names a prior author (\"X et al.\")", "bool"),
        ("abstract_has_however_gap", "abstract has a \"however\" gap sentence", "bool"),
        ("abstract_has_in_this_paper_pivot", "abstract has an \"in this paper\" pivot", "bool"),
        ("has_limitations_heading", "has a named Limitations heading", "bool"),
        ("has_early_bullet_list", "early bullet list  [contributions proxy]", "bool"),
    ]
    for key, label, kind in conv_rows:
        vals = [v[key] for v in results.values() if v.get(key) is not None]
        o = ours.get(key)
        if kind == "bool":
            n_yes = sum(1 for v in vals if v)
            o_s = "yes" if o else "no"
            print(f"     {label:<46}{o_s:>6}   {n_yes}/{len(vals)}")
            conv_out[key] = {"ours": bool(o), "n_yes": n_yes, "of": len(vals)}
        else:
            b = band(vals) if vals else None
            print(f"     {label:<46}{o!s:>6}   " +
                  (f"min {b['min']} median {b['median']} max {b['max']}" if b else "--"))
            conv_out[key] = {**(b or {}), "ours": o}

    manifest = {
        "_schema": "corpus_style_and_conventions_study",
        "_generated_by": "scripts/30_corpus_style_and_conventions_study.py",
        "_corpus_caveat": (
            "50 papers, topic-matched (backdoor/adversarial-ML + NIDS/tabular-ML), NOT "
            "venue-matched. Mixed venues (NDSS, USENIX Security, IEEE S&P, NeurIPS, ACSAC, "
            "ECML, ICLR, journals, arXiv preprints) at page lengths 6-20+. Never read as "
            "\"the ICCIT norm\"."
        ),
        "_passive_voice_method": (
            "passive_per_1k_words is a real dependency-parse count (spaCy en_core_web_sm, "
            "counts nsubjpass/nsubj:pass tokens -- one per finite passive clause), not a "
            "regex proxy. Replaces the earlier passive_per_1k_words_proxy field, which "
            "undercounted irregular past participles (\"shown\", \"given\") and overcounted "
            "adjectival be+-ed phrases the parser can distinguish but a fixed pattern cannot."
        ),
        "_citation_caveat": (
            "Corpus citation counting only matches numeric bracket style, e.g. [19] or "
            "[3, 7]. Papers using author-year style (Author, 2020) are undercounted -- a "
            "corpus min of 0.0 citations/1k most likely means an author-year paper, not a "
            "zero-citation one. Ours (\\cite{...}) is counted exactly, no proxy."
        ),
        "_n_corpus_papers": len(results),
        "_n_failed": len(failed),
        "_failed": failed,
        "_subcorpora": SUBCORPORA,
        "ours": ours,
        "papers": results,
        "bands": bands_out,
        "conventions": conv_out,
    }
    if args.write_manifest:
        p = REPO / args.manifest
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest, indent=1))
        print(f"\n  wrote {p}")
    else:
        print(f"\n  (manifest NOT written; pass --write-manifest to write {args.manifest})")

    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
