#!/usr/bin/env python
"""Summarize the NetFlow detector arm: does the vision toolbox port to the new corpora?

Reads results/netflow_detectors.json (or its checkpoint) and reports, per corpus and per benign-share
level, each detector under BOTH decision rules, with the loud control alongside so a null is always
read against a control on the same corpus.

Three things this deliberately does not do. It computes no rank correlation across corpora, because
corpus identity co-varies with everything. It never gives Activation Clustering or Neural Cleanse a
ranking AUC, because neither emits a ranked score. It never merges recall under the two rules.

Run:  .venv/bin/python scripts/86_netflow_detector_analysis.py [--from-checkpoint]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

FULL = config.RESULTS / "netflow_detectors.json"
CKPT = config.RESULTS / "netflow_detectors.checkpoint.json"
OUT = config.RESULTS / "netflow_detector_analysis.json"
Z = "3.0"


def boot_ci(v, resamples=config.BOOTSTRAP_RESAMPLES, seed=20260903):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    if v.size < 2:
        return None
    rng = np.random.default_rng(seed)
    d = rng.choice(v, size=(resamples, len(v)), replace=True).mean(axis=1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def agg(rows, path):
    vals = []
    for r in rows:
        cur = r
        for p in path:
            cur = cur.get(p) if isinstance(cur, dict) else None
            if cur is None:
                break
        if isinstance(cur, (int, float)) and not isinstance(cur, bool):
            vals.append(float(cur))
    if not vals:
        return dict(n=0, mean=None, ci=None)
    return dict(n=len(vals), mean=float(np.mean(vals)), sd=(float(np.std(vals, ddof=1)) if len(vals) > 1 else None),
                ci=boot_ci(vals))


def ac_mode(cell: dict) -> str:
    """Which of Activation Clustering's three outcomes this cell shows.

    Reporting a mean recall hides the distinction that matters operationally. Two cells can both
    read recall 0.000 while one flags nearly half of all benign traffic and the other flags a
    handful of rows. The first is a harmful detector, the second is merely a silent one, and a mean
    of zeros cannot tell them apart.
    """
    ac = cell.get("ac")
    if not isinstance(ac, dict) or ac.get("recall") is None:
        return "not scored"
    if ac["recall"] > 0.5:
        return "isolates the poison"
    n_poison = max(1, int(cell.get("n_poison_scored") or 1))
    return ("flags the wrong cluster" if ac.get("n_flagged", 0) > 5 * n_poison
            else "collapses to a fragment")


DETECTORS = [
    ("spectral", "Spectral Signatures", True),
    ("spectre", "SPECTRE", True),
    ("ac", "Activation Clustering", False),
    ("strip", "STRIP", True),
    ("nc", "Neural Cleanse", False),
]


def block(rows) -> dict:
    out = {}
    for key, name, scored in DETECTORS:
        if scored:
            out[key] = dict(
                name=name,
                fixed_recall=agg(rows, [key, "fixed_recall"]),
                mad_recall=agg(rows, [key, "mad_recall", Z]),
                mad_fpr=agg(rows, [key, "mad_fpr", Z]),
                auc=agg(rows, [key, "auc"]),
            )
        elif key == "ac":
            flagged = [r.get("ac", {}).get("n_flagged") for r in rows if isinstance(r.get("ac"), dict)]
            out[key] = dict(name=name, recall=agg(rows, ["ac", "recall"]),
                            fpr=agg(rows, ["ac", "fpr"]), purity=agg(rows, ["ac", "purity"]),
                            silhouette=agg(rows, ["ac", "silhouette"]),
                            median_n_flagged=(float(np.median([f for f in flagged if f is not None]))
                                              if any(f is not None for f in flagged) else None),
                            auc=None,
                            auc_reason="hard two-cluster assignment, not a ranked score")
        else:
            flags = [r.get("nc", {}).get("flags_target") for r in rows if isinstance(r.get("nc"), dict)]
            flags = [bool(f) for f in flags if f is not None]
            out[key] = dict(name=name, n=len(flags),
                            flags_target_rate=(float(np.mean(flags)) if flags else None),
                            auc=None, auc_reason="hard flag, not a ranked score")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-checkpoint", action="store_true")
    args = ap.parse_args()
    src = CKPT if args.from_checkpoint else FULL
    if not src.exists():
        print(f"missing {src}")
        return 1
    blob = json.loads(src.read_text())
    controls = list(blob.get("controls", {}).values())
    cells = list(blob.get("cells", {}).values())
    ok = [c for c in cells if c.get("interpretable")]
    print(f"read {len(cells)} cells ({len(ok)} interpretable) and {len(controls)} controls from {src.name}")

    from collections import Counter
    modes = Counter(ac_mode(c) for c in ok)
    wrong = [c for c in ok if ac_mode(c) == "flags the wrong cluster"]
    frag = [c for c in ok if ac_mode(c) == "collapses to a fragment"]
    ac_modes = dict(
        counts=dict(modes),
        wrong_cluster_fpr=(sorted(c["ac"]["fpr"] for c in wrong if c["ac"].get("fpr") is not None)
                           or None),
        fragment_n_flagged=(sorted(c["ac"]["n_flagged"] for c in frag) or None),
    )
    out = dict(source=src.name, n_cells=len(cells), n_interpretable=len(ok), ac_modes=ac_modes,
               n_infeasible=sum(c.get("status") == "attack_infeasible" for c in cells),
               n_ineffective=sum(c.get("status") == "attack_ineffective" for c in cells),
               mad_z=float(Z), control=block(controls), pooled=block(ok),
               by_corpus={}, by_level={})
    for corpus in config.NETFLOW_CORPORA:
        rows = [c for c in ok if c["corpus"] == corpus]
        if rows:
            out["by_corpus"][corpus] = dict(n=len(rows), **block(rows))
    for corpus in (config.NETFLOW_MANIPULATION_CORPUS, config.NETFLOW_REPLICATION_CORPUS):
        for share in config.NETFLOW_BENIGN_SHARES:
            rows = [c for c in ok if c["corpus"] == corpus and c.get("benign_share") == share]
            if rows:
                out["by_level"][f"{corpus}|{share}"] = dict(n=len(rows), **block(rows))
    OUT.write_text(json.dumps(out, indent=2))

    def line(tag, b, n):
        s, sp, st = b["spectral"], b["spectre"], b["strip"]
        ac, nc = b["ac"], b["nc"]
        f = lambda d, k="mean": ("  n/a " if d is None or d.get(k) is None else f"{d[k]:.4f}")
        print(f"  {tag:<30} n={n:>2}  Spec fix {f(s['fixed_recall'])} mad {f(s['mad_recall'])} auc {f(s['auc'])} | "
              f"SPECTRE auc {f(sp['auc'])} | STRIP auc {f(st['auc'])} | AC rec {f(ac['recall'])} | "
              f"NC tgt {f(nc, 'flags_target_rate')}")

    print(f"\nLOUD CONTROL (violating trigger) -- the gate every null below is read against")
    line("control", out["control"], len(controls))
    print(f"\nPOOLED over interpretable realizable cells")
    line("pooled", out["pooled"], len(ok))
    print(f"\nBY CORPUS")
    for c, b in out["by_corpus"].items():
        line(c, b, b["n"])
    print(f"\nBY BENIGN-SHARE LEVEL")
    for k, b in out["by_level"].items():
        line(k, b, b["n"])
    print("\nACTIVATION CLUSTERING, by outcome (a mean recall hides which failure occurred)")
    for k, n in modes.most_common():
        print(f"  {k:<26} {n:>3} of {len(ok)} cells")
    if ac_modes["wrong_cluster_fpr"]:
        f = ac_modes["wrong_cluster_fpr"]
        print(f"  when it flags the wrong cluster, false positives run {f[0]:.3f} to {f[-1]:.3f} "
              f"of clean rows")
    if ac_modes["fragment_n_flagged"]:
        g = ac_modes["fragment_n_flagged"]
        print(f"  when it collapses, it flags {g[0]} to {g[-1]} rows")
    print(f"\nwrote {OUT.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
