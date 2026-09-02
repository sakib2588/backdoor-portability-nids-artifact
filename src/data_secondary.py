"""UNSW-NB15 loading, label cleanup, temporal split, and bounds/scaler fitting for the
secondary-dataset generalisation extension (plan Task 3, migrating Task 2's data machinery).

This module OWNS the download/clean/dedupe/split/bounds/scaler pipeline for the secondary dataset
(plan Task 3, Step 1). It absorbs that portion of `src/secondary_data.py` (Task 2), which now
re-exports the names below for backward compatibility with `scripts/13_secondary_data_gate.py` and
the original `tests/test_secondary_data.py`, both committed under Task 2 and left unchanged here.
See `notes/20260731-decision-task3-constraint-layer-migration.md` for the full migration writeup and
`notes/20260729-decision-secondary-dataset-selection.md` for why UNSW-NB15 (raw superset, custom
temporal split) was selected over the two rejected candidates.

Source: the community HuggingFace mirror ``Mouwiya/UNSW-NB15`` of the official raw NUSW-NB15
49-feature CSV export (Moustafa, N. and Slay, J., UNSW Canberra Cyber -- see
https://research.unsw.edu.au/projects/unsw-nb15-dataset).

New in this module (Task 3, Step 1): `SecondarySetup`, a frozen dataclass bundling the split
matrices, the raw feature contract, a train-only-fitted `RawBounds` box, a train-only-fitted
`StandardScaler`, and the ndarray-based constraint set from `src.constraints_secondary`
(`build_secondary_constraints`) plus a `constraint_manifest` audit dict -- and `load_secondary_setup`,
the single narrow entry point that builds it. `load_secondary_setup` raises immediately if asked for
`dataset_id == "ctu_13_neris"` (the primary dataset owned by `src.data`, never this module) so the
extension cannot silently reuse the primary loader in place of a genuine secondary-dataset run.
"""
from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.constraints_secondary import SecondaryConstraint, audit_constraints, build_secondary_constraints

HF_REPO_ID = "Mouwiya/UNSW-NB15"
HF_SHARD_FILES = (
    "data/train-00000-of-00002.parquet",
    "data/train-00001-of-00002.parquet",
)
HF_FEATURES_DOC = "NUSW-NB15_features.csv"  # official per-feature description, same repo

SOURCE_URL_MIRROR = f"https://huggingface.co/datasets/{HF_REPO_ID}"
SOURCE_URL_OFFICIAL = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
LICENSE = (
    "Free use for academic research purposes granted in perpetuity by the original authors "
    "(Moustafa & Slay); commercial use requires the authors' agreement. The HuggingFace mirror "
    "re-hosts the same academic-licensed data and grants no separate rights."
)

TARGET_RAW = "attack_cat"
LABEL_COL = "label"

# Split protocol. 80/20 temporal cut on the deduplicated, time-sorted superset (see
# `temporal_split_unsw`); kept local to this module rather than added to src/config.py, which plan
# Task 1 locked for the primary-dataset preregistration constants.
SECONDARY_TRAIN_FRAC = 0.8

# Predeclared clean-data validity tolerance for the OLD (src.secondary_data, DataFrame-based,
# 8-entry) constraint manifest -- see that module's docstring for the measured coverage this was set
# from. Kept here (the manifest's evidentiary numbers were measured against the load/split pipeline
# this module now owns) and re-exported by src.secondary_data for the gate script.
SECONDARY_CLEAN_VALID_TOLERANCE = 0.995

# Columns excluded from the raw numeric feature contract, with reasons (see the decision note,
# "Raw feature contract" section, for the full justification of each).
ID_COLUMNS = ["srcip", "sport", "dstip", "dsport"]          # high-cardinality flow identifiers
TIME_COLUMNS = ["Stime", "Ltime"]                             # used only to build the temporal split
CATEGORICAL_EXCLUDED = ["proto", "service", "state"]          # free-text categorical, no numeric unit
EXCLUDED_COLUMNS = ID_COLUMNS + TIME_COLUMNS + CATEGORICAL_EXCLUDED + [TARGET_RAW, LABEL_COL]

# Columns that arrive with a blank-string / NaN encoding for "not applicable to this flow" rather
# than a true missing value (documented in NUSW-NB15_features.csv: ct_ftp_cmd, is_ftp_login, and
# ct_flw_http_mthd are all FTP/HTTP-specific counters that are 0/undefined on non-FTP/HTTP flows).
# Coerced to numeric with fill 0.0 rather than dropped -- verified empirically below.
_COERCE_FILL_ZERO = ["ct_ftp_cmd", "is_ftp_login", "ct_flw_http_mthd"]

# Whitespace and naming inconsistencies present in the raw `attack_cat` column, verified on disk
# 2026-07-31 (e.g. `' Fuzzers'`, `' Fuzzers '`, `'Backdoor'` alongside `'Backdoors'`). Normalising
# this is a label-integrity fix, not a semantic remapping: it maps distinct on-disk SPELLINGS of the
# same official MITRE-adjacent UNSW-NB15 attack category onto one canonical spelling before any
# binary collapse, so the 9-category taxonomy documented by the dataset's authors is unchanged.
_ATTACK_CAT_RENAME = {"Backdoor": "Backdoors"}


def download_unsw() -> List[str]:
    """Fetch both raw-data parquet shards (+ the official feature-description CSV) into data/raw.

    Cached by huggingface_hub after the first call. Expected one-time wall-clock: a few minutes for
    ~230MB total over a normal connection; every subsequent call is a local cache hit (seconds).
    """
    from huggingface_hub import hf_hub_download

    config.ensure_dirs()
    local_dir = str(config.DATA_RAW / "unsw_nb15")
    paths = [
        hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=f, local_dir=local_dir)
        for f in HF_SHARD_FILES
    ]
    hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=HF_FEATURES_DOC, local_dir=local_dir)
    return paths


def sha256_of_file(path: str) -> str:
    """Streamed SHA-256 (no full-file read into memory) -- the same file digest is stable across
    machines and is what the decision note and the audit artifact record as provenance evidence."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_attack_cat(s: pd.Series) -> pd.Series:
    """Strip whitespace, fold the `Backdoor`/`Backdoors` spelling, map benign NaN -> "Normal"."""
    cleaned = s.astype("string").str.strip()
    cleaned = cleaned.replace(_ATTACK_CAT_RENAME)
    cleaned = cleaned.fillna("Normal")
    return cleaned


def load_unsw_raw() -> pd.DataFrame:
    """Download, concatenate, clean, deduplicate, and temporally sort the full raw superset.

    Steps (each is a disk-verified necessity, not a convenience choice -- see the decision note):
      1. Concatenate both shards, sort by `Stime` (the row order in each individual shard is
         already time-ascending, but the two shards must be interleaved correctly).
      2. Drop EXACT full-row duplicates (`keep="first"`) -- 695,831 / 2,280,090 (30.5%) rows in this
         mirror are byte-for-byte duplicates of an earlier row, verified 2026-07-31. Left undropped,
         a duplicate can straddle the train/test cut and leak.
      3. Clean `attack_cat` (whitespace + spelling) and integrity-check it against `label`.
      4. Coerce the FTP/HTTP "not applicable" columns to numeric with fill 0.0.
    Returns the FULL cleaned frame (not yet split); callers use `temporal_split_unsw`.
    """
    paths = download_unsw()
    shards = [pd.read_parquet(p) for p in paths]
    full = pd.concat(shards, ignore_index=True)
    # OOM fix (2026-07-31, notes/20260731-bug-task5-oom-root-cause.md): `shards` (both raw parquet
    # shards, ~1.7GB together measured) has no further use once concatenated into `full`, but a plain
    # Python `del` alone leaves them reachable until the next unrelated allocation triggers pymalloc's
    # arena reuse; `gc.collect()` reclaims them immediately instead of leaving a multi-GB high-water
    # mark that this function's own later steps (sort_values/drop_duplicates, each producing a fresh
    # full-size copy) would otherwise stack on top of. Verified: this function's peak RSS drops from
    # ~5.7GB to ~2.4GB with this and the two del/gc.collect() calls below (see the decision note for
    # the measured trace) -- no output value changes, only intermediate copies are freed earlier.
    del shards
    gc.collect()
    full = full.sort_values("Stime", kind="mergesort").reset_index(drop=True)

    n_before = len(full)
    full = full.drop_duplicates(keep="first").reset_index(drop=True)
    n_after = len(full)
    full.attrs["n_exact_duplicate_rows_dropped"] = n_before - n_after
    gc.collect()  # drop_duplicates' pre-dedup copy (695,831/2,280,090 rows, ~30.5%) is now unreachable

    full["attack_cat"] = _clean_attack_cat(full["attack_cat"])
    label_from_cat = (full["attack_cat"] != "Normal").astype(int)
    if not (full["label"] == label_from_cat).all():
        mismatch = int((full["label"] != label_from_cat).sum())
        raise ValueError(
            f"attack_cat-derived binary label disagrees with the shipped `label` column on "
            f"{mismatch} rows after whitespace/spelling cleanup -- integrity check failed"
        )

    for col in _COERCE_FILL_ZERO:
        full[col] = pd.to_numeric(full[col], errors="coerce").fillna(0.0)

    return full


def raw_feature_columns(df: pd.DataFrame) -> List[str]:
    """The numeric raw feature contract: all columns except identifiers, timestamps, free-text
    categoricals, and the two label-bearing columns (see EXCLUDED_COLUMNS for reasons)."""
    return [c for c in df.columns if c not in EXCLUDED_COLUMNS]


def temporal_split_unsw(
    df: pd.DataFrame, train_frac: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leak-free temporal split: earliest `train_frac` of the (already time-sorted, deduplicated)
    rows form train, the strict remainder is test. Mirrors `src.data.temporal_split`'s CTU protocol
    (earliest rows = train, no future -> past leakage) rather than the official UNSW-NB15 partition's
    stratified-random protocol, which this dataset's evidence gate rejected for duplicate rows."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    cut = int(round(len(df) * train_frac))
    train = df.iloc[:cut].reset_index(drop=True)
    test = df.iloc[cut:].reset_index(drop=True)
    return train, test


def flow_row_ids(df: pd.DataFrame) -> np.ndarray:
    """A row-content fingerprint used ONLY for the leakage check (not modelling): the raw dataset's
    small, reused synthetic-testbed IP pool makes the (5-tuple + timestamp) key alone collide
    naturally within a single second even for genuinely distinct flows, so the disjointness check
    below instead hashes the full feature-plus-identifier row -- an exact match here means an exact
    duplicate RECORD crossing the train/test boundary, which is the leakage this gate cares about."""
    cols = [c for c in df.columns if c not in ("attack_cat",)]  # keep label; attack_cat is derived
    return pd.util.hash_pandas_object(df[cols], index=False).to_numpy()


class SecondarySplit(NamedTuple):
    """Minimal split container used by the leakage/bounds checks, on both the real pipeline data
    (`scripts/13_secondary_data_gate.py`) and synthetic data in the unit tests."""
    features: np.ndarray
    flow_ids: np.ndarray


def validate_disjoint_split(train: SecondarySplit, test: SecondarySplit) -> None:
    """Raise ValueError if any identifier is shared between train and test."""
    shared = set(np.asarray(train.flow_ids).tolist()) & set(np.asarray(test.flow_ids).tolist())
    if shared:
        preview = sorted(str(x) for x in shared)[:5]
        raise ValueError(
            f"train/test overlap detected: {len(shared)} shared flow identifier(s), e.g. {preview}"
        )


class RawBounds(NamedTuple):
    lower: np.ndarray
    upper: np.ndarray


def fit_raw_bounds(x) -> RawBounds:
    """Fit per-feature [min, max] bounds. Caller passes TRAIN features only -- this function has no
    way to enforce that itself, so `bounds_fit_partition: "train_only"` in the audit artifact and the
    `test_secondary_bounds_are_fit_on_train_only` test are what actually guard the invariant."""
    arr = np.asarray(x, dtype=np.float64)
    return RawBounds(lower=arr.min(axis=0), upper=arr.max(axis=0))


# --- Task 3, Step 1: the narrow SecondarySetup loader interface --------------------------------

@dataclass(frozen=True)
class SecondarySetup:
    """Everything downstream trigger-construction / detector-portability code needs for one
    secondary dataset, built by `load_secondary_setup()`. `bounds` and `scaler` are fit on
    `x_train_raw` ONLY (never `x_test_raw`) -- see `load_secondary_setup` for where that happens."""
    dataset_id: str
    x_train_raw: np.ndarray
    y_train: np.ndarray
    x_test_raw: np.ndarray
    y_test: np.ndarray
    feature_names: Tuple[str, ...]
    bounds: Tuple[np.ndarray, np.ndarray]
    scaler: StandardScaler
    constraints: Tuple[SecondaryConstraint, ...]
    constraint_manifest: Dict[str, object]


def load_secondary_setup(dataset_id: str = "unsw_nb15") -> SecondarySetup:
    """Build the full secondary-dataset setup: split, raw feature contract, train-only-fitted
    bounds and `StandardScaler`, the ndarray constraint set, and its clean-partition audit.

    Raises ValueError immediately -- before any download or split work -- if `dataset_id` is the
    PRIMARY dataset's id (`config.PRIMARY_DATASET_ID`, "ctu_13_neris"). The primary dataset is
    loaded exclusively by `src.data`; this function must never be a silent back door to it, per the
    plan Task 3 Step 1 requirement ("so the extension cannot silently reuse the primary loader").
    """
    if dataset_id == config.PRIMARY_DATASET_ID:
        raise ValueError(
            f"load_secondary_setup() must never load the primary dataset ({dataset_id!r}) -- use "
            "src.data's CTU-13 loaders instead. This guard exists so the cross-dataset extension "
            "cannot silently fall back to the primary dataset when the secondary one is intended."
        )
    if dataset_id != config.SECONDARY_DATASET_ID:
        raise ValueError(
            f"unsupported dataset_id {dataset_id!r}; only {config.SECONDARY_DATASET_ID!r} is "
            "implemented by src.data_secondary.load_secondary_setup()"
        )

    df = load_unsw_raw()
    feature_cols = tuple(raw_feature_columns(df))
    train_df, test_df = temporal_split_unsw(df, train_frac=SECONDARY_TRAIN_FRAC)

    x_train_raw = train_df[list(feature_cols)].to_numpy(dtype=np.float64)
    y_train = train_df[LABEL_COL].to_numpy(dtype=np.int64)
    x_test_raw = test_df[list(feature_cols)].to_numpy(dtype=np.float64)
    y_test = test_df[LABEL_COL].to_numpy(dtype=np.int64)

    # OOM fix (2026-07-31): `df`/`train_df`/`test_df` are dead weight from here on -- everything
    # downstream (bounds, scaler, constraints, audits) works off the numpy arrays already extracted
    # above. Measured: without this, the full cleaned/deduped DataFrame (~2.2GB) stays resident
    # alongside its train/test splits for the rest of this function and the caller's whole process
    # lifetime (nothing else in the pipeline ever reassigns/frees it). See
    # notes/20260731-bug-task5-oom-root-cause.md for the before/after RSS trace.
    del df, train_df, test_df
    gc.collect()

    bounds = fit_raw_bounds(x_train_raw)
    scaler = StandardScaler()
    scaler.fit(x_train_raw)  # train rows only -- never test

    _validate_feature_order_against_audit(feature_cols)

    constraints = build_secondary_constraints(feature_cols)

    train_benign_mask = y_train == 0
    test_benign_mask = y_test == 0
    train_audit = audit_constraints(x_train_raw[train_benign_mask], constraints)
    test_audit = audit_constraints(x_test_raw[test_benign_mask], constraints)
    n_train_benign = int(train_benign_mask.sum())
    n_test_benign = int(test_benign_mask.sum())
    constraint_manifest = {
        "n_constraints": len(constraints),
        "n_train_benign_rows": n_train_benign,
        "n_test_benign_rows": n_test_benign,
        "constraints": [
            {
                "constraint": c.name,
                "kind": c.kind,
                "source": c.source,
                "tolerance": c.tolerance,
                "clean_train_valid_rate": (
                    1.0 - train_audit["violations"][c.name] / n_train_benign if n_train_benign else 1.0
                ),
                "clean_test_valid_rate": (
                    1.0 - test_audit["violations"][c.name] / n_test_benign if n_test_benign else 1.0
                ),
            }
            for c in constraints
        ],
    }

    return SecondarySetup(
        dataset_id=dataset_id,
        x_train_raw=x_train_raw,
        y_train=y_train,
        x_test_raw=x_test_raw,
        y_test=y_test,
        feature_names=feature_cols,
        bounds=(bounds.lower, bounds.upper),
        scaler=scaler,
        constraints=constraints,
        constraint_manifest=constraint_manifest,
    )


def _validate_feature_order_against_audit(feature_cols: Sequence[str]) -> None:
    """If `results/secondary_data_audit.json` (Task 2's gate output) already exists, its recorded
    `schema.features` order must match this load exactly -- a silent reorder would misalign every
    downstream bounds/scaler/constraint index against the audit's recorded coverage numbers. No-op
    (nothing to compare against) when the audit artifact is absent, e.g. in a fresh checkout before
    the gate script has run."""
    audit_path = config.RESULTS / "secondary_data_audit.json"
    if not audit_path.exists():
        return
    try:
        audit = json.loads(audit_path.read_text())
    except json.JSONDecodeError:
        return  # partial/corrupt artifact from a prior failed gate run -- nothing to compare against
    audit_features = audit.get("schema", {}).get("features")
    if audit_features is None:
        return
    if list(audit_features) != list(feature_cols):
        raise ValueError(
            "feature order mismatch between load_secondary_setup() and the recorded "
            f"results/secondary_data_audit.json schema.\nexpected (from audit): {audit_features}\n"
            f"got (this load): {list(feature_cols)}"
        )
