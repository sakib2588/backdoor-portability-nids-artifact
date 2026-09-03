"""Load one constituent corpus of NF-UQ-NIDS-v2 by its `Dataset` column value.

All four corpora share one NetFlow v2 schema, so one module serves all four and the feature space is
identical across them by construction. That is the point of the study: whatever differs between
corpora is a dataset property, not an encoding artifact.

Two entry points:
  load_netflow_corpus       -- native class balance, for the four-corpus replication arm
  load_netflow_benign_share -- a requested benign share, for the within-corpus manipulation

Both are stratified: the benign and attack counts are computed first and then filled by an exact
per-row-group allocation, so a requested share is delivered exactly rather than approached in
expectation. Measured 2026-09-03, the four corpora are interleaved uniformly across all 152 row
groups, so a proportional allocation spreads every draw over the whole file. There is no early
`break`: the allocation is computed before any data is read, and every group holding the corpus
contributes its quota.

The source parquet is 2.6 GB / 76M rows and lives in another project. It is read-only here.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, NamedTuple, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import config

# Columns never read. The four identifier columns are dropped at the parquet level because they are
# excluded features anyway and the two address columns are the file's heaviest strings.
_NEVER_READ = ("IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT")

_INDEX_PATH = config.DATA_PROCESSED / "netflow" / "_group_index.json"


class NetFlowSplit(NamedTuple):
    x_tr_raw: np.ndarray
    y_tr: np.ndarray
    x_te_raw: np.ndarray
    y_te: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    feature_columns: List[str]


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in config.NETFLOW_EXCLUDED_COLUMNS]


def _require(parquet_path: str) -> str:
    """Fail with the fix rather than a bare FileNotFoundError on a stat deep in the loader."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"NetFlow source parquet not found at {parquet_path!r}.\n"
            "config.NETFLOW_PARQUET resolves from the NETFLOW_PARQUET environment variable and "
            "otherwise defaults to a path inside this repository. On a machine that holds the "
            "corpus elsewhere, do either of:\n"
            "    export NETFLOW_PARQUET=/abs/path/to/NF-UQ-NIDS-v2.parquet\n"
            "    ln -s /abs/path/to/NF-UQ-NIDS-v2.parquet "
            "data/raw/nf_uq/NF-UQ-NIDS-v2.parquet"
        )
    return parquet_path


def _open(parquet_path: str) -> pq.ParquetFile:
    return pq.ParquetFile(_require(parquet_path))


def _read_columns(pf: pq.ParquetFile) -> List[str]:
    return [f.name for f in pf.schema_arrow if f.name not in _NEVER_READ]


def _source_stamp(parquet_path: str) -> Dict:
    st = os.stat(_require(parquet_path))
    return dict(path=str(parquet_path), size=st.st_size, mtime_ns=st.st_mtime_ns)


def group_index(parquet_path: str = config.NETFLOW_PARQUET, refresh: bool = False) -> Dict:
    """Per row group and per corpus, the benign and attack row counts.

    One narrow two-column pass over the file, cached to disk and keyed on the source's size and
    mtime, so a changed source rebuilds the index instead of silently reusing counts for other data.
    Every sampler needs these counts to allocate exact quotas before reading any feature data.
    """
    stamp = _source_stamp(parquet_path)
    if not refresh and _INDEX_PATH.exists():
        cached = json.loads(_INDEX_PATH.read_text())
        if cached.get("source") == stamp:
            return cached

    pf = _open(parquet_path)
    counts = {c: [] for c in config.NETFLOW_CORPORA}
    for i in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(i, columns=["Dataset", "Label"])
        ds = tbl.column("Dataset")
        lab = tbl.column("Label")
        for corpus in config.NETFLOW_CORPORA:
            hit = pc.equal(ds, corpus)
            n_benign = int(pc.sum(pc.and_(hit, pc.equal(lab, 0))).as_py() or 0)
            n_total = int(pc.sum(hit).as_py() or 0)
            counts[corpus].append([n_benign, n_total - n_benign])

    index = dict(source=stamp, n_row_groups=pf.metadata.num_row_groups, counts=counts)
    for corpus, (n_pop, b_pop) in config.NETFLOW_POPULATION.items():
        seen = np.asarray(counts[corpus], dtype=np.int64)
        if int(seen.sum()) != n_pop or int(seen[:, 0].sum()) != b_pop:
            raise RuntimeError(
                f"{corpus}: parquet holds {int(seen.sum()):,} rows / {int(seen[:, 0].sum()):,} "
                f"benign, config.NETFLOW_POPULATION says {n_pop:,} / {b_pop:,}. The source data "
                "changed; stop and re-verify before running anything downstream.")

    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(index))
    os.replace(tmp, _INDEX_PATH)
    return index


def allocate(avail: Sequence[int], need: int) -> np.ndarray:
    """Split `need` draws across row groups holding `avail` rows each, exactly and deterministically.

    Every non-empty group gets at least one draw where the budget allows, then the remainder goes out
    proportionally to remaining availability under a largest-remainder rule. Guaranteeing one per
    group is what keeps the draw spread over the whole file; the exact sum is what lets a requested
    benign share be delivered rather than approximated.
    """
    avail = np.asarray(avail, dtype=np.int64)
    if need < 0 or need > int(avail.sum()):
        raise ValueError(f"need={need} not fundable from {int(avail.sum())} available rows")
    alloc = np.zeros_like(avail)
    if need == 0:
        return alloc

    nonempty = np.flatnonzero(avail > 0)
    if need < len(nonempty):
        # Budget smaller than the number of groups: spend it on the largest groups.
        alloc[nonempty[np.argsort(-avail[nonempty], kind="stable")][:need]] = 1
        return alloc

    alloc[nonempty] = 1
    remaining = need - int(alloc.sum())
    if remaining:
        head = avail - alloc
        exact = head / head.sum() * remaining
        take = np.minimum(np.floor(exact).astype(np.int64), head)
        short = remaining - int(take.sum())
        if short:
            frac = exact - np.floor(exact)
            for i in np.argsort(-frac, kind="stable"):
                if short <= 0:
                    break
                add = min(int(head[i] - take[i]), short)
                take[i] += add
                short -= add
        if short:  # pathological caps; sweep whatever room is left
            for i in np.argsort(-(head - take), kind="stable"):
                if short <= 0:
                    break
                add = min(int(head[i] - take[i]), short)
                take[i] += add
                short -= add
        alloc = alloc + take

    if int(alloc.sum()) != need or bool((alloc > avail).any()):
        raise RuntimeError(f"allocation failed: sum={int(alloc.sum())} need={need}")
    return alloc


def _draw(pf: pq.ParquetFile, dataset: str, keep: List[str],
          alloc_b: np.ndarray, alloc_a: np.ndarray, rng: np.random.Generator
          ) -> Tuple[pa.Table, int]:
    """Read only the groups the allocation asks for and take the allotted rows of each class."""
    parts, groups_used = [], 0
    for i in np.flatnonzero(alloc_b + alloc_a):
        i = int(i)
        tbl = pf.read_row_group(i, columns=keep)
        tbl = tbl.filter(pc.equal(tbl.column("Dataset"), dataset)).drop_columns(["Dataset"])
        label = tbl.column("Label").to_numpy(zero_copy_only=False)
        picks = []
        for cls, alloc in ((0, alloc_b), (1, alloc_a)):
            quota = int(alloc[i])
            if quota == 0:
                continue
            pool = np.flatnonzero(label == cls)
            if len(pool) < quota:
                raise RuntimeError(
                    f"{dataset} group {i}: class {cls} holds {len(pool)} rows, allocation asked "
                    f"for {quota}. The group index is stale; rebuild it with refresh=True.")
            picks.append(rng.choice(pool, size=quota, replace=False))
        if picks:
            parts.append(tbl.take(np.sort(np.concatenate(picks))))
            groups_used += 1
    return pa.concat_tables(parts), groups_used


def _finalize(tbl: pa.Table, rng: np.random.Generator) -> Tuple[pd.DataFrame, int, int]:
    df = tbl.to_pandas()
    cols = feature_columns(df)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    block = df[cols].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    nonfinite = int((~np.isfinite(block.to_numpy())).sum())
    block = block.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Cap the two bytes-per-second columns at line rate. See config.NETFLOW_RATE_CAP_BYTES_PER_S:
    # the source data carries values up to 8.9e304 there, which overflow float64 when squared and
    # take the whole column to NaN inside the standardiser.
    capped = 0
    for name in config.NETFLOW_RATE_CAPPED_COLUMNS:
        if name in block.columns:
            over = block[name] > config.NETFLOW_RATE_CAP_BYTES_PER_S
            capped += int(over.sum())
            block.loc[over, name] = config.NETFLOW_RATE_CAP_BYTES_PER_S

    _assert_standardisable(block, cols)
    df[cols] = block
    df["Label"] = df["Label"].astype(np.int64)
    return df, nonfinite, capped


def _assert_standardisable(block: pd.DataFrame, cols: List[str]) -> None:
    """Fail loudly if any column would overflow float64 when its variance is computed.

    StandardScaler does not raise on overflow: it emits a RuntimeWarning, records a NaN variance and
    silently returns a column of NaN, which trains a victim that predicts one class for every row and
    scores exactly the benign share. That failure is invisible in every downstream number, so it is
    caught here at the point the data is built rather than inferred later from a suspicious accuracy.
    """
    arr = block.to_numpy()
    with np.errstate(over="ignore", invalid="ignore"):
        sq = np.square(arr).sum(axis=0)
    bad = [c for c, v in zip(cols, sq) if not np.isfinite(v)]
    if bad:
        raise RuntimeError(
            f"columns {bad} overflow float64 when squared, so any standardiser fit on them yields a "
            "NaN column and a degenerate victim. Add them to config.NETFLOW_RATE_CAPPED_COLUMNS with "
            "a physically motivated cap, or exclude them.")


def _meta(dataset: str, df: pd.DataFrame, seed: int, groups_total: int, groups_used: int,
          nonfinite: int, capped: int, extra: Dict) -> Dict:
    n_pop, b_pop = config.NETFLOW_POPULATION[dataset]
    cols = feature_columns(df)
    n_benign = int((df["Label"] == 0).sum())
    meta = dict(
        dataset=dataset,
        population_rows=n_pop,
        population_benign=b_pop,
        population_attack_rate=1.0 - b_pop / n_pop,   # from the measured table, never inferred
        sampled_rows=int(len(df)),
        sampled_benign=n_benign,
        sampled_attack=int(len(df) - n_benign),
        benign_share=n_benign / len(df),
        feature_columns=cols,
        n_features=len(cols),
        seed=seed,
        row_groups_holding_corpus=groups_total,
        row_groups_contributing=groups_used,
        nonfinite_values_zeroed=nonfinite,
        rate_capped_values=capped,
        rate_cap_bytes_per_s=config.NETFLOW_RATE_CAP_BYTES_PER_S,
        sha256_feature_block=hashlib.sha256(
            np.ascontiguousarray(df[cols].to_numpy())).hexdigest(),
    )
    meta.update(extra)
    return meta


def _load(dataset: str, need_b: int, need_a: int, seed: int, parquet_path: str,
          extra: Dict) -> Tuple[pd.DataFrame, Dict]:
    pf = _open(parquet_path)
    keep = _read_columns(pf)
    counts = np.asarray(group_index(parquet_path)["counts"][dataset], dtype=np.int64)
    rng = np.random.default_rng(seed)
    alloc_b = allocate(counts[:, 0], need_b)
    alloc_a = allocate(counts[:, 1], need_a)
    tbl, groups_used = _draw(pf, dataset, keep, alloc_b, alloc_a, rng)
    df, nonfinite, capped = _finalize(tbl, rng)
    groups_total = int((counts.sum(axis=1) > 0).sum())
    return df, _meta(dataset, df, seed, groups_total, groups_used, nonfinite, capped, extra)


def load_netflow_corpus(dataset: str,
                        n_rows: int = config.NETFLOW_SAMPLE_ROWS,
                        seed: int = config.NETFLOW_SPLIT_SEED,
                        parquet_path: str = config.NETFLOW_PARQUET,
                        ) -> Tuple[pd.DataFrame, Dict]:
    """Sample `n_rows` at the corpus's native class balance, stratified on the population ratio."""
    if dataset not in config.NETFLOW_POPULATION:
        raise ValueError(f"unknown corpus {dataset!r}; expected one of {config.NETFLOW_CORPORA}")
    n_pop, b_pop = config.NETFLOW_POPULATION[dataset]
    if n_rows > n_pop:
        raise ValueError(f"{dataset}: asked for {n_rows:,} rows but corpus has {n_pop:,}")
    need_b = int(round(n_rows * b_pop / n_pop))
    need_a = n_rows - need_b
    if need_a > n_pop - b_pop:
        need_a = n_pop - b_pop
        need_b = n_rows - need_a
    return _load(dataset, need_b, need_a, seed, parquet_path,
                 dict(mode="native", requested_benign_share=None))


def load_netflow_benign_share(dataset: str,
                              benign_share: float,
                              n_rows: int = config.NETFLOW_SAMPLE_ROWS,
                              seed: int = config.NETFLOW_SPLIT_SEED,
                              parquet_path: str = config.NETFLOW_PARQUET,
                              ) -> Tuple[pd.DataFrame, Dict]:
    """Sample `n_rows` with a *requested* benign share, for the within-corpus manipulation.

    Total rows are held fixed, so only the ratio moves. Raises rather than silently delivering a
    different share if the corpus lacks rows of either class.
    """
    if dataset not in config.NETFLOW_POPULATION:
        raise ValueError(f"unknown corpus {dataset!r}; expected one of {config.NETFLOW_CORPORA}")
    if not 0.0 < benign_share < 1.0:
        raise ValueError(f"benign_share must be in (0,1), got {benign_share}")
    n_pop, b_pop = config.NETFLOW_POPULATION[dataset]
    a_pop = n_pop - b_pop
    need_b = int(round(n_rows * benign_share))
    need_a = n_rows - need_b
    if need_b > b_pop:
        raise ValueError(
            f"{dataset}: benign share {benign_share} at n={n_rows:,} needs {need_b:,} benign rows, "
            f"corpus has {b_pop:,}")
    if need_a > a_pop:
        raise ValueError(
            f"{dataset}: benign share {benign_share} at n={n_rows:,} needs {need_a:,} attack rows, "
            f"corpus has {a_pop:,}")
    return _load(dataset, need_b, need_a, seed, parquet_path,
                 dict(mode="benign_share", requested_benign_share=benign_share))


def split_netflow(df: pd.DataFrame, meta: Dict,
                  seed: int = config.NETFLOW_SPLIT_SEED) -> NetFlowSplit:
    """Stratified random 80/20 split. NF-v2 carries no reliable wall-clock column, so the temporal
    split CTU-13 uses is unavailable here and a seeded stratified split is used instead. That
    difference is stated in Methods, not hidden."""
    rng = np.random.default_rng(seed)
    cols = meta["feature_columns"]
    y = df["Label"].to_numpy()
    train_parts, test_parts = [], []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        idx = idx[rng.permutation(len(idx))]
        cut = int(round(config.NETFLOW_TRAIN_FRAC * len(idx)))
        train_parts.append(idx[:cut])
        test_parts.append(idx[cut:])
    train_idx = np.sort(np.concatenate(train_parts))
    test_idx = np.sort(np.concatenate(test_parts))
    x = df[cols].to_numpy(dtype=np.float64)
    return NetFlowSplit(x[train_idx], y[train_idx], x[test_idx], y[test_idx],
                        train_idx, test_idx, cols)


def validate_disjoint_split(s: NetFlowSplit) -> None:
    overlap = np.intersect1d(s.train_idx, s.test_idx)
    if overlap.size:
        raise RuntimeError(f"train/test overlap on {overlap.size} rows")
