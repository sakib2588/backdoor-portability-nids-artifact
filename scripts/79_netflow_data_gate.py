#!/usr/bin/env python
"""Stage 1 of the NetFlow benign-share study: materialize every sample and admit the manifest.

Sixteen samples: four corpora at native balance for the Q2 replication arm, plus six
benign-share-controlled samples on each of the two corpora that can host the full gradient. This is
the only stage that touches the 76M-row source parquet; everything downstream reads the parquets
this writes.

For each sample it records the split sizes, the poison-budget feasibility at every configured rate,
and the fraction of clean rows satisfying each authored relation. The manifest is then admitted
twice: once scoped to the corpora the property analysis actually uses, and once over all four. A
corpus excluded from an arm by pre-registration must not veto relations for the corpora that are in
it, so the two scopings are reported separately and never merged.

Run:  .venv/bin/python scripts/79_netflow_data_gate.py --smoke   # 40k rows, wiring check
      .venv/bin/python scripts/79_netflow_data_gate.py           # the real 1.2M-row run
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
from src.constraints_netflow import (
    NETFLOW_RELATIONS, admitted_manifest, manifest_fingerprint, relation_satisfaction,
)
from src.data_netflow import (
    group_index, load_netflow_benign_share, load_netflow_corpus, split_netflow,
    validate_disjoint_split,
)

OUT_DIR = config.DATA_PROCESSED / "netflow"
CKPT = config.RESULTS / "netflow_data_gate.checkpoint.json"
RESULT = config.RESULTS / "netflow_data_gate.json"

# Feasibility is computed for the union of every rate any arm uses, so the sweep can look up any
# (sample, rate) pair without this stage having to know which arm reads it.
ALL_RATES = tuple(sorted({config.NETFLOW_RATE, *config.NETFLOW_NATIVE_RATES}))


def planned_samples() -> list:
    """Every sample this study needs, as (tag, corpus, benign_share|None)."""
    out = [(f"native__{c}", c, None) for c in config.NETFLOW_CORPORA]
    for corpus in (config.NETFLOW_MANIPULATION_CORPUS, config.NETFLOW_REPLICATION_CORPUS):
        for share in config.NETFLOW_BENIGN_SHARES:
            out.append((f"share{share:.2f}__{corpus}", corpus, share))
    return out


def feasibility(meta: dict, n_train: int, rates=ALL_RATES) -> dict:
    """For each poison rate, how much of the benign class it would consume.

    A clean-label attack plants the trigger in benign rows and keeps their true label, so the benign
    class is the attacker's entire budget. Where that budget binds, the number below is the finding.
    """
    n_benign_train = int(round(meta["benign_share"] * n_train))
    rows = {}
    for rate in rates:
        need = int(np.ceil(rate * n_train))
        rows[f"{rate}"] = dict(
            poison_rows_needed=need,
            benign_rows_available=n_benign_train,
            benign_fraction_required=(need / n_benign_train) if n_benign_train else None,
            feasible=bool(n_benign_train and need <= n_benign_train),
        )
    return dict(n_train=n_train, n_benign_train=n_benign_train, per_rate=rows,
                max_feasible_rate=(n_benign_train / n_train) if n_train else 0.0)


def config_key(args) -> dict:
    return dict(
        smoke=args.smoke,
        n_rows=args.rows,
        samples=[t for t, _, _ in planned_samples()],
        excluded_columns=list(config.NETFLOW_EXCLUDED_COLUMNS),
        train_frac=config.NETFLOW_TRAIN_FRAC,
        split_seed=config.NETFLOW_SPLIT_SEED,
        rates=list(ALL_RATES),
        admission_threshold=config.NETFLOW_RELATION_ADMISSION_FRAC,
        rate_cap=config.NETFLOW_RATE_CAP_BYTES_PER_S,
        rate_capped_columns=list(config.NETFLOW_RATE_CAPPED_COLUMNS),
        manifest=manifest_fingerprint(),
    )


def load_checkpoint(path: Path, key: dict) -> dict:
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('samples', {}))} samples already done")
    return blob.get("samples", {})


def save_checkpoint(path: Path, key: dict, samples: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, samples=samples)))
    tmp.replace(path)   # atomic: a crash mid-write never corrupts the checkpoint


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_sample(tag: str, corpus: str, share, n_rows: int) -> dict:
    t0 = time.time()
    if share is None:
        df, meta = load_netflow_corpus(corpus, n_rows=n_rows)
    else:
        df, meta = load_netflow_benign_share(corpus, benign_share=share, n_rows=n_rows)

    s = split_netflow(df, meta)
    validate_disjoint_split(s)

    # Relations are claims about what NetFlow can emit, so they are measured on clean rows only.
    clean = df[df["Label"] == 0]
    satisfaction = relation_satisfaction(clean)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet = OUT_DIR / f"{tag}.parquet"
    df.to_parquet(parquet, index=False, compression="zstd")

    elapsed = time.time() - t0
    return dict(
        tag=tag, corpus=corpus, requested_benign_share=share,
        n_rows=int(len(df)), n_train=int(len(s.y_tr)), n_test=int(len(s.y_te)),
        benign_share=meta["benign_share"],
        n_features=meta["n_features"], feature_columns=meta["feature_columns"],
        sampled_benign=meta["sampled_benign"], sampled_attack=meta["sampled_attack"],
        train_benign=int((s.y_tr == 0).sum()), test_benign=int((s.y_te == 0).sum()),
        nonfinite_values_zeroed=meta["nonfinite_values_zeroed"],
        rate_capped_values=meta["rate_capped_values"],
        row_groups_holding_corpus=meta["row_groups_holding_corpus"],
        row_groups_contributing=meta["row_groups_contributing"],
        sha256_feature_block=meta["sha256_feature_block"],
        n_clean_rows_measured=int(len(clean)),
        relation_satisfaction=satisfaction,
        feasibility=feasibility(meta, int(len(s.y_tr))),
        parquet=str(parquet.relative_to(config.ROOT)),
        parquet_sha256=sha256_file(parquet),
        parquet_bytes=parquet.stat().st_size,
        elapsed_s=round(elapsed, 2),
    )


def per_corpus_satisfaction(samples: dict) -> dict:
    """Worst measured satisfaction per corpus, over every sample drawn from it.

    A relation must hold on the data actually trained on, not merely on the native-balance draw, so
    the benign-share levels are folded in and the minimum is taken.
    """
    out: dict = {}
    for row in samples.values():
        bucket = out.setdefault(row["corpus"], {})
        for name, val in row["relation_satisfaction"].items():
            if val is None:
                bucket[name] = None
            elif bucket.get(name, np.inf) is not None:
                bucket[name] = min(bucket.get(name, np.inf), val)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="40k-row samples for a wiring check, written to the same layout")
    ap.add_argument("--rows", type=int, default=None,
                    help="override the sample size (default: 40000 with --smoke, else config)")
    args = ap.parse_args()
    args.rows = args.rows or (40_000 if args.smoke else config.NETFLOW_SAMPLE_ROWS)

    config.ensure_dirs()
    print(f"NetFlow data gate: {args.rows:,} rows per sample, "
          f"manifest {manifest_fingerprint()[:12]}")
    group_index()   # builds or validates the cached per-group class counts

    key = config_key(args)
    samples = load_checkpoint(CKPT, key)
    plan = planned_samples()

    for i, (tag, corpus, share) in enumerate(plan, 1):
        if tag in samples:
            print(f"[{i:2d}/{len(plan)}] {tag:<34} cached")
            continue
        try:
            row = build_sample(tag, corpus, share, args.rows)
        except ValueError as exc:
            # A share the corpus cannot fund is a measurement, not a crash. Record and continue.
            row = dict(tag=tag, corpus=corpus, requested_benign_share=share,
                       status="sample_infeasible", reason=str(exc))
            print(f"[{i:2d}/{len(plan)}] {tag:<34} INFEASIBLE: {exc}")
        else:
            print(f"[{i:2d}/{len(plan)}] {tag:<34} n={row['n_rows']:,} "
                  f"benign={row['benign_share']:.4f} train={row['n_train']:,} "
                  f"({row['elapsed_s']:.1f}s)")
        samples[tag] = row
        save_checkpoint(CKPT, key, samples)

    built = {t: r for t, r in samples.items() if "relation_satisfaction" in r}
    sat_by_corpus = per_corpus_satisfaction(built)

    property_corpora = [c for c in config.NETFLOW_CORPORA
                        if c not in config.NETFLOW_PROPERTY_EXCLUDED and c in sat_by_corpus]
    all_corpora = [c for c in config.NETFLOW_CORPORA if c in sat_by_corpus]
    kept_property, report_property = admitted_manifest(sat_by_corpus, corpora=property_corpora)
    kept_all, report_all = admitted_manifest(sat_by_corpus, corpora=all_corpora)

    blob = dict(
        config_key=key,
        n_rows_per_sample=args.rows,
        smoke=args.smoke,
        manifest_fingerprint=manifest_fingerprint(),
        samples=samples,
        per_corpus_satisfaction=sat_by_corpus,
        manifest_property_arm=report_property,
        manifest_all_corpora=report_all,
        total_elapsed_s=round(sum(r.get("elapsed_s", 0.0) for r in samples.values()), 2),
    )
    RESULT.write_text(json.dumps(blob, indent=2))

    print(f"\nmanifest, property arm ({', '.join(property_corpora)}): "
          f"authored {report_property['n_authored']}, "
          f"expressible {report_property['n_expressible']}, "
          f"admitted {report_property['n_admitted']}")
    print(f"manifest, all four corpora: authored {report_all['n_authored']}, "
          f"expressible {report_all['n_expressible']}, admitted {report_all['n_admitted']}")
    for name, frac in sorted(report_property["rejected"].items()):
        print(f"  rejected (property arm) {name:<34} worst satisfaction {frac:.6f}")

    print("\nfeasibility at each configured rate (train block):")
    print(f"  {'sample':<34} {'benign_train':>12}  " +
          "  ".join(f"r={r}" for r in ALL_RATES))
    for tag, row in samples.items():
        if "feasibility" not in row:
            continue
        flags = "  ".join(
            ("yes  " if row["feasibility"]["per_rate"][f"{r}"]["feasible"] else "NO   ")
            for r in ALL_RATES)
        print(f"  {tag:<34} {row['feasibility']['n_benign_train']:>12,}  {flags}")

    print(f"\nwrote {RESULT.relative_to(config.ROOT)} "
          f"(total {blob['total_elapsed_s']:.1f}s of sample construction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
