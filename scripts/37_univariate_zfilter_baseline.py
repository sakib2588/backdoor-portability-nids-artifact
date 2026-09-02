#!/usr/bin/env python3
"""P-003: does a trivial per-feature z-filter catch this trigger, with no detector at all?

PRE-REGISTERED BEFORE THE RUN. The watermark is each chosen feature's training-split
mean plus six standard deviations, applied jointly to up to sixteen features. Both
hostile reviewers independently asked the same question: constraint satisfaction is
not distributional plausibility, so a defender running a univariate outlier filter
on the raw training block may not need Spectral Signatures at all. If that is true,
the paper's subject has to be restated.

Two variants, and the distinction is the whole point:

  ORACLE     z computed against the CLEAN training block's mean and sd. This is an
             upper bound on the filter's power and is NOT available to a defender,
             who does not know which rows are clean.
  DEFENDER   z computed against the POISONED block's own mean and sd. This is what
             a real defender can actually compute, and the poison shifts the very
             statistics used to find it.

Score per row = max over features of |z|, used as a ranking score so it is directly
comparable to Spectral's ranking AUC, and thresholded at FPR budgets matched to the
rules the paper already reports.

DECISION RULE, fixed before any result was seen:
  UPHELD    poison recall >= 0.90 at FPR <= 0.0658 (the adaptive rule's cost)
  REFUTED   poison recall <  0.50 at that same FPR budget
  anything between the two is INCONCLUSIVE and is reported as such, not rounded
  toward either verdict.

Anchor cell (rate 0.005, cost 16), five seeds. Checkpointed per seed: a restart
skips completed seeds, and the key carries a config fingerprint so a changed input
invalidates the entry rather than silently reusing it.

Run:  python scripts/37_univariate_zfilter_baseline.py [--write-manifest]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import apply_standardiser
from src.models import train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import build_trigger, raw_trigger_stats, shap_rank_features

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _m2_common import get_config, load_setup  # noqa: E402

RATE, COST, TARGET = 0.005, 16, config.ATTACK_TARGET
MAD_FPR_BUDGET = 0.0658          # the adaptive rule's measured cost, from the manuscript
UPHELD_RECALL, REFUTED_RECALL = 0.90, 0.50
CKPT = config.RESULTS / "univariate_zfilter_baseline.checkpoint.json"


def fingerprint(cfg) -> str:
    payload = json.dumps({"rate": RATE, "cost": COST, "target": TARGET,
                          "epochs": cfg["mlp_epochs"], "seeds": list(cfg["seeds"]),
                          "n_sigma": 6.0, "score": "max_abs_z"}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_ckpt(fp):
    if CKPT.exists():
        blob = json.loads(CKPT.read_text())
        if blob.get("fingerprint") == fp:
            return blob.get("seeds", {})
        print("checkpoint fingerprint differs from this config; ignoring it")
    return {}


def save_ckpt(fp, seeds):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"fingerprint": fp, "seeds": seeds}))
    tmp.replace(CKPT)          # atomic: a crash mid-write never corrupts the checkpoint


def auc(scores, is_poison):
    """Rank-based AUC, ties averaged. No sklearn dependency for one number."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = np.asarray(scores)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n_pos = int(is_poison.sum()); n_neg = len(s) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[is_poison].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def recall_at_fpr(scores, is_poison, fpr_budget):
    """Flag the highest-scoring rows until the clean false-positive budget is spent."""
    clean = scores[~is_poison]
    n_allowed = int(np.floor(fpr_budget * len(clean)))
    if n_allowed <= 0:
        return 0.0, 0.0
    thresh = np.partition(clean, -n_allowed)[-n_allowed]
    flagged = scores >= thresh
    recall = float(flagged[is_poison].mean())
    fpr = float(flagged[~is_poison].mean())
    return recall, fpr


def run_seed(seed, S, cfg, device):
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    stats = raw_trigger_stats(x_tr_raw, constraints)

    x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
    clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
    ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET,
                                 kind="mlp", device=device)
    realiz, _ = build_trigger(ranking, COST, x_tr_raw, constraints, features, bounds, stats=stats)

    _, y_p, poison_idx = poison_trainset_cleanlabel(
        x_tr_raw, y_tr, realiz, RATE, S["scaler"], target=TARGET, seed=seed,
        constraints=constraints, feature_names=features, bounds=bounds)

    # Rebuild the poisoned block in RAW space, through the same apply_trigger call the
    # poisoning path uses, so the rows carry the projection. Stamping the raw watermark by
    # hand would measure a trigger this paper never deploys.
    from src.trigger import apply_trigger
    x_poisoned = np.asarray(x_tr_raw, dtype=float).copy()
    x_poisoned[poison_idx] = apply_trigger(
        x_poisoned[poison_idx], realiz, constraints, features, bounds)

    benign_pos = np.where(y_p == TARGET)[0]
    block = x_poisoned[benign_pos]
    is_poison = np.isin(benign_pos, poison_idx)

    clean_block = np.asarray(x_tr_raw, dtype=float)[benign_pos][~is_poison]

    out = {"seed": seed, "n_rows": int(len(block)), "n_poison": int(is_poison.sum())}
    for variant, ref in (("oracle", clean_block), ("defender", block)):
        mu, sd = ref.mean(axis=0), ref.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        scores = np.abs((block - mu) / sd).max(axis=1)
        rec, fpr = recall_at_fpr(scores, is_poison, MAD_FPR_BUDGET)
        out[variant] = {
            "auc": auc(scores, is_poison),
            "recall_at_mad_fpr": rec,
            "realised_fpr": fpr,
            "recall_at_hard_6sigma": float((scores[is_poison] > 6.0).mean()),
            "clean_flagged_frac_at_hard_6sigma": float((scores[~is_poison] > 6.0).mean()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config(smoke=False)

    fp = fingerprint(cfg)
    done = load_ckpt(fp)
    print(f"device={device}  fingerprint={fp}  already done: {sorted(done)}")

    S = load_setup(cfg)
    t0 = time.time()
    for seed in cfg["seeds"]:
        if str(seed) in done:
            print(f"--- seed {seed}: cached, skipping")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed)")
        done[str(seed)] = run_seed(seed, S, cfg, device)
        save_ckpt(fp, done)
        r = done[str(seed)]
        print(f"    oracle   AUC {r['oracle']['auc']:.4f}  recall@6.58%FPR {r['oracle']['recall_at_mad_fpr']:.4f}")
        print(f"    defender AUC {r['defender']['auc']:.4f}  recall@6.58%FPR {r['defender']['recall_at_mad_fpr']:.4f}")

    rows = [done[str(s)] for s in cfg["seeds"]]
    summary = {}
    for variant in ("oracle", "defender"):
        recs = [r[variant]["recall_at_mad_fpr"] for r in rows]
        summary[variant] = {
            "mean_auc": float(np.mean([r[variant]["auc"] for r in rows])),
            "mean_recall_at_mad_fpr": float(np.mean(recs)),
            "sd_recall_population": float(np.std(recs)),
            "mean_recall_at_hard_6sigma": float(np.mean([r[variant]["recall_at_hard_6sigma"] for r in rows])),
        }
    m = summary["defender"]["mean_recall_at_mad_fpr"]
    verdict = ("UPHELD" if m >= UPHELD_RECALL else
               "REFUTED" if m < REFUTED_RECALL else "INCONCLUSIVE")

    print("\n=== P-003 result (decision rule fixed before the run)")
    for v in ("oracle", "defender"):
        s = summary[v]
        print(f"  {v:9s} AUC {s['mean_auc']:.4f}   recall@6.58%FPR {s['mean_recall_at_mad_fpr']:.4f} "
              f"(SD {s['sd_recall_population']:.4f})   recall at a hard |z|>6 {s['mean_recall_at_hard_6sigma']:.4f}")
    print(f"  VERDICT (on the defender variant, which is the realistic one): {verdict}")

    blob = {"criterion": {"upheld_recall": UPHELD_RECALL, "refuted_recall": REFUTED_RECALL,
                          "fpr_budget": MAD_FPR_BUDGET},
            "cell": {"rate": RATE, "cost": COST}, "fingerprint": fp,
            "verdict": verdict, "summary": summary, "per_seed": rows}
    if args.write_manifest:
        path = config.RESULTS / "univariate_zfilter_baseline.json"
        path.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"\nwrote {path.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
