#!/usr/bin/env python
"""Stage 4 of the NetFlow benign-share study: the pre-registered analysis.

Reads the poison sweep and answers Q1 under the rule fixed in
notes/20260903-decision-netflow-multicorpus-preregistration.md before any number was seen:

  primary statistic  Spearman rho between benign share and mean backdoor fraction over the six
                     manipulated levels, at cost 16, on NF-CSE-CIC-IDS2018-v2
  null               exact permutation over all 720 orderings, never a parametric test
  CONFIRMED          |rho| >= 0.8 on the primary corpus AND the same sign on the replication corpus
  REFUTED            |rho| < 0.4 on the primary corpus
  otherwise          INCONCLUSIVE, reported as such

Saturated cells are excluded from the primary statistic and reported with their counts. A cell whose
clean model already accepts nearly every watermarked flow cannot show a positive backdoor fraction,
so including it would measure a ceiling rather than an effect.

Q2 is summarized and deliberately NOT given a rank test. Benign share co-varies with capture year,
topology and attack families across the four corpora, so a correlation there would invite the
confounded reading the design exists to avoid.

Run:  .venv/bin/python scripts/82_netflow_property_analysis.py
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

SWEEP = config.RESULTS / "netflow_poison_sweep.json"
CKPT = config.RESULTS / "netflow_poison_sweep.checkpoint.json"
GATE = config.RESULTS / "netflow_data_gate.json"
OUT = config.RESULTS / "netflow_property_analysis.json"

HETEROGENEITY_ROWS = 50_000   # benign rows sampled per level for the heterogeneity statistic
HETEROGENEITY_SEED = 20260903


def benign_heterogeneity(benign: np.ndarray) -> float:
    """Effective dimensionality of the benign class: the participation ratio of the eigenvalues of
    its rank-correlation matrix, (sum lambda)^2 / sum(lambda^2).

    Ranks are invariant to any monotone transform, so a single extreme value cannot dominate the
    spectrum the way it dominates a covariance. Reads as "how many independent directions does this
    corpus's benign traffic actually span", which is the quantity the hypothesis is about.

    Two earlier attempts at this statistic were wrong and both were caught by running them, not by
    reading them. Equal-width binning measured outlier presence, because one extreme value widens the
    range until nearly all mass sits in the first bin. Equal-frequency binning was degenerate:
    quantile bins are equally populated by construction, so normalized entropy returns about 1.0 for
    every continuous feature regardless of spread.

    Verified behavior at d=8 (2026-09-03): isotropic Gaussian 7.986; rank-1 data 1.000; isotropic
    plus one 1e6 outlier 7.990; isotropic with 5% of rows scaled by 1e4, 7.981; four of eight
    features held constant, 3.994.
    """
    x = np.asarray(benign, dtype=np.float64)
    keep = x.std(axis=0) > 0
    if not keep.any():
        return 0.0
    r = np.apply_along_axis(rankdata, 0, x[:, keep])
    r = (r - r.mean(axis=0)) / r.std(axis=0)
    ev = np.clip(np.linalg.eigvalsh(np.cov(r, rowvar=False)), 0.0, None)
    denom = float((ev ** 2).sum())
    return float(ev.sum() ** 2 / denom) if denom > 0 else 0.0


def boot_ci(values: np.ndarray, resamples: int = config.BOOTSTRAP_RESAMPLES,
            seed: int = 20260903) -> tuple:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    draws = rng.choice(v, size=(resamples, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def level_means(cells: list, corpus: str, cost: int) -> dict:
    """Mean backdoor fraction per benign-share level, with saturated cells excluded.

    A cell whose clean model already accepts every watermarked flow gives a backdoor fraction near
    zero for a reason that has nothing to do with the hypothesis. Excluded counts are reported, never
    silently dropped.
    """
    out = {}
    for share in config.NETFLOW_BENIGN_SHARES:
        rows = [c for c in cells if c["arm"] == "Q1" and c["corpus"] == corpus
                and c["benign_share"] == share and c["cost"] == cost]
        usable = [r for r in rows if r["status"] == "ok"]
        vals = np.array([r["backdoor_fraction"] for r in usable], dtype=float)
        clean = np.array([r["clean_model_stamped_asr"] for r in rows
                          if r["clean_model_stamped_asr"] is not None], dtype=float)
        out[share] = dict(
            n_cells=len(rows), n_usable=len(usable),
            n_saturated=sum(r["status"] == "saturated" for r in rows),
            n_ineffective=sum(r["status"] == "attack_ineffective" for r in rows),
            n_infeasible=sum(r["status"] == "attack_infeasible" for r in rows),
            mean=float(vals.mean()) if vals.size else None,
            sd=float(vals.std(ddof=1)) if vals.size > 1 else None,
            ci=boot_ci(vals) if vals.size > 1 else None,
            mean_clean_stamped_asr=float(clean.mean()) if clean.size else None,
            mean_poisoned_asr=float(np.mean([r["asr"] for r in rows
                                             if r["asr"] is not None])) if rows else None,
        )
    return out


def rank_test(prop: np.ndarray, outcome: np.ndarray) -> dict:
    """Spearman rho with an EXACT permutation null over every ordering.

    n=6 gives 720 orderings and a minimum attainable two-sided p of 0.00278. No parametric test is
    used (project protocol, n <= 30). If saturation or infeasibility removes levels, `n` here falls
    and `min_attainable_p` rises with it; both are reported so a reader can see the test weaken.
    """
    prop = np.asarray(prop, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if len(prop) < 3:
        return dict(rho=None, n=int(len(prop)), n_permutations=0,
                    p_exact_two_sided=None, min_attainable_p=None,
                    note="fewer than three usable levels; no rank test is defined")
    rho = float(spearmanr(prop, outcome).statistic)
    perms = list(permutations(range(len(prop))))
    null = np.array([spearmanr(prop, outcome[list(p)]).statistic for p in perms])
    return dict(rho=rho, n=int(len(prop)), n_permutations=len(perms),
                p_exact_two_sided=float((np.abs(null) >= abs(rho) - 1e-12).mean()),
                min_attainable_p=float(2 / len(perms)))


def verdict(primary: dict, replication: dict) -> str:
    """CONFIRMED needs the primary corpus over the bar AND the replication corpus agreeing in sign."""
    rp, rr = primary.get("rho"), replication.get("rho")
    if rp is None:
        return "INCONCLUSIVE"
    if abs(rp) >= config.NETFLOW_RHO_CONFIRM and rr is not None and np.sign(rp) == np.sign(rr):
        return "CONFIRMED"
    if abs(rp) < config.NETFLOW_RHO_REFUTE:
        return "REFUTED"
    return "INCONCLUSIVE"


def heterogeneity_by_level(gate: dict, corpora: list) -> dict:
    """Benign-class effective dimensionality per materialized sample.

    Reported alongside benign share because the manipulation moves the benign class's size and its
    sampled diversity together. A gradient result therefore identifies benign share as realized by
    subsampling, not heterogeneity in isolation, and this statistic is what lets a reader see how
    much the two move together in this design.
    """
    rng = np.random.default_rng(HETEROGENEITY_SEED)
    out = {}
    for tag, row in gate["samples"].items():
        if row.get("corpus") not in corpora or "parquet" not in row:
            continue
        df = pd.read_parquet(config.ROOT / row["parquet"])
        benign = df[df["Label"] == 0][row["feature_columns"]].to_numpy(dtype=np.float64)
        if len(benign) > HETEROGENEITY_ROWS:
            benign = benign[rng.choice(len(benign), HETEROGENEITY_ROWS, replace=False)]
        out[tag] = dict(corpus=row["corpus"], benign_share=row["benign_share"],
                        requested_benign_share=row.get("requested_benign_share"),
                        n_benign_scored=int(len(benign)),
                        effective_dimensionality=benign_heterogeneity(benign))
        del df, benign
    return out


def q2_summary(cells: list) -> dict:
    out = {}
    for corpus in config.NETFLOW_CORPORA:
        rows = [c for c in cells if c["arm"] == "Q2" and c["corpus"] == corpus]
        usable = [r for r in rows if r["status"] == "ok"]
        vals = np.array([r["backdoor_fraction"] for r in usable], dtype=float)
        clean = np.array([r["clean_model_stamped_asr"] for r in rows
                          if r["clean_model_stamped_asr"] is not None], dtype=float)
        asr = np.array([r["asr"] for r in rows if r["asr"] is not None], dtype=float)
        out[corpus] = dict(
            n_cells=len(rows), n_usable=len(usable),
            n_saturated=sum(r["status"] == "saturated" for r in rows),
            n_ineffective=sum(r["status"] == "attack_ineffective" for r in rows),
            n_infeasible=sum(r["status"] == "attack_infeasible" for r in rows),
            excluded_from_property_analysis=corpus in config.NETFLOW_PROPERTY_EXCLUDED,
            mean_backdoor_fraction=float(vals.mean()) if vals.size else None,
            backdoor_fraction_ci=boot_ci(vals) if vals.size > 1 else None,
            mean_clean_stamped_asr=float(clean.mean()) if clean.size else None,
            mean_poisoned_asr=float(asr.mean()) if asr.size else None,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-checkpoint", action="store_true",
                    help="analyze the checkpoint instead of the finished results file")
    ap.add_argument("--skip-heterogeneity", action="store_true",
                    help="skip the benign heterogeneity pass, which re-reads every sample parquet")
    args = ap.parse_args()

    src = CKPT if args.from_checkpoint else SWEEP
    if not src.exists():
        print(f"missing {src}; run scripts/80_netflow_poison_sweep.py first")
        return 1
    blob = json.loads(src.read_text())
    cells = list(blob["cells"].values())
    gate = json.loads(GATE.read_text())
    print(f"read {len(cells)} cells from {src.name}")

    manip = config.NETFLOW_MANIPULATION_CORPUS
    repl = config.NETFLOW_REPLICATION_CORPUS
    cost = config.NETFLOW_PRIMARY_COST

    per_corpus_levels, rank = {}, {}
    for corpus in (manip, repl):
        levels = level_means(cells, corpus, cost)
        per_corpus_levels[corpus] = levels
        usable = [(s, d["mean"]) for s, d in levels.items() if d["mean"] is not None]
        rank[corpus] = rank_test(np.array([s for s, _ in usable]),
                                 np.array([m for _, m in usable]))

    v = verdict(rank[manip], rank[repl])

    het = {} if args.skip_heterogeneity else heterogeneity_by_level(gate, [manip, repl])

    out = dict(
        preregistration="notes/20260903-decision-netflow-multicorpus-preregistration.md",
        source=src.name,
        n_cells=len(cells),
        primary_cost=cost,
        primary_corpus=manip,
        replication_corpus=repl,
        decision_rule=dict(confirm_abs_rho=config.NETFLOW_RHO_CONFIRM,
                           refute_abs_rho=config.NETFLOW_RHO_REFUTE,
                           saturation_asr=config.NETFLOW_SATURATION_ASR,
                           attack_effective_asr=config.ATTACK_EFFECTIVE_ASR),
        q1_levels=per_corpus_levels,
        q1_rank_test=rank,
        q1_verdict=v,
        benign_heterogeneity=het,
        q2_summary=q2_summary(cells),
        q2_rank_test="not computed by design",
    )
    OUT.write_text(json.dumps(out, indent=2, default=float))

    print(f"\nQ1, {manip}, cost {cost}")
    print(f"  {'share':>6} {'n_ok':>5} {'sat':>4} {'mean backdoor':>14}  {'95% CI':>22}  clean-stamped ASR")
    for share, d in per_corpus_levels[manip].items():
        ci = f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]" if d["ci"] else "n/a"
        mean = f"{d['mean']:+.4f}" if d["mean"] is not None else "n/a"
        cs = f"{d['mean_clean_stamped_asr']:.4f}" if d["mean_clean_stamped_asr"] is not None else "n/a"
        print(f"  {share:>6} {d['n_usable']:>5} {d['n_saturated']:>4} {mean:>14}  {ci:>22}  {cs}")
    for corpus in (manip, repl):
        r = rank[corpus]
        if r["rho"] is None:
            print(f"\n{corpus}: {r['note']}")
        else:
            print(f"\n{corpus}: rho = {r['rho']:+.4f} at n = {r['n']}, exact two-sided "
                  f"p = {r['p_exact_two_sided']:.5f} over {r['n_permutations']} orderings "
                  f"(minimum attainable {r['min_attainable_p']:.5f})")
    print(f"\nVERDICT against the pre-registered rule: {v}")
    print(f"wrote {OUT.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
