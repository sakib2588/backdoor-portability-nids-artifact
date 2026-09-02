"""Task 2: read-only evidence gate for the secondary NIDS dataset (UNSW-NB15), run BEFORE any
model training on it. Downloads the raw data (cached after the first run), builds the documented
temporal split, fits raw-value bounds on TRAIN ONLY, evaluates the constraint manifest's coverage
on clean (label==0) rows, and writes results/secondary_data_audit.json.

Exit code is non-zero (and `passed: false` is written) if ANY of the following holds:
  - a required schema field is absent from the loaded data,
  - the downloaded file's SHA-256 differs from the hash recorded on a PRIOR run of this script
    (first run has nothing to compare against and records the hash instead),
  - train and test overlap (any duplicated row fingerprint spanning the partition boundary),
  - any constraint check produces a non-finite validity value,
  - clean-partition constraint validity falls below the predeclared tolerance
    (SECONDARY_CLEAN_VALID_TOLERANCE, see src/secondary_data.py -- set from evidence gathered BEFORE
    this script existed, per notes/20260729-decision-secondary-dataset-selection.md; it is not
    re-tuned here).

Run:  python scripts/13_secondary_data_gate.py
Expected wall-clock: first run ~2-5 minutes (one-time ~230MB download over a normal connection,
then vectorised pandas ops on ~2.3M rows, a few seconds); every subsequent run is a local cache hit,
well under a minute. No checkpointing is implemented: every step here is a single vectorised
pandas/numpy pass over in-memory data (not a per-row/per-constraint Python loop), so a full run is
far under the project's ~10-minute checkpoint threshold even on the first (download) pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.secondary_data import (
    LICENSE,
    SECONDARY_CLEAN_VALID_TOLERANCE,
    SECONDARY_TRAIN_FRAC,
    SOURCE_URL_MIRROR,
    SOURCE_URL_OFFICIAL,
    build_unsw_constraints,
    download_unsw,
    evaluate_constraint_coverage,
    fit_raw_bounds,
    flow_row_ids,
    load_unsw_raw,
    raw_feature_columns,
    sha256_of_file,
    temporal_split_unsw,
    validate_constraint_manifest,
    validate_disjoint_split,
    SecondarySplit,
)

OUT_PATH = config.RESULTS / "secondary_data_audit.json"

REQUIRED_SCHEMA_FIELDS = [
    "Stime", "Ltime", "attack_cat", "label",
    "sttl", "dttl", "sbytes", "dbytes", "Spkts", "Dpkts",
    "smeansz", "dmeansz", "tcprtt", "synack", "ackdat",
]


def _fail(reason: str, partial: dict) -> None:
    # No config.ensure_dirs() here: only ever called from within main(), after main() has already
    # ensured the output directories exist once, near the top.
    partial["passed"] = False
    partial["failure_reason"] = reason
    OUT_PATH.write_text(json.dumps(partial, indent=2, sort_keys=False))
    print(f"GATE FAILED: {reason}")
    print(f"wrote partial artifact to {OUT_PATH}")
    sys.exit(1)


def main() -> None:
    config.ensure_dirs()
    artifact: dict = {}

    # --- Step A: download + checksum -------------------------------------------------------
    print("Downloading (or reusing cached) UNSW-NB15 raw shards ...")
    paths = download_unsw()
    shard_hashes = {Path(p).name: sha256_of_file(p) for p in paths}
    primary_hash = shard_hashes[Path(paths[0]).name]
    print(f"shard SHA-256: {shard_hashes}")

    artifact["dataset"] = "UNSW-NB15 (raw NUSW-NB15 49-feature superset, deduplicated, temporal split)"
    artifact["source"] = {
        "url": SOURCE_URL_MIRROR,
        "official_reference_url": SOURCE_URL_OFFICIAL,
        "sha256": primary_hash,
        "shard_sha256": shard_hashes,
        "license": LICENSE,
    }

    # Checksum drift check: if a prior audit artifact recorded a hash for this shard, the current
    # download must match it (a changed hash means the mirror's content moved under us).
    if OUT_PATH.exists():
        try:
            prior = json.loads(OUT_PATH.read_text())
            prior_hashes = prior.get("source", {}).get("shard_sha256", {})
            for name, h in shard_hashes.items():
                if name in prior_hashes and prior_hashes[name] != h:
                    _fail(
                        f"checksum drift on {name}: recorded {prior_hashes[name]}, now {h}",
                        artifact,
                    )
        except (json.JSONDecodeError, KeyError):
            pass  # prior artifact unreadable/partial -- nothing to compare against, proceed

    # --- Step B: load, clean, schema check --------------------------------------------------
    print("Loading + cleaning + deduplicating (vectorised, a few seconds once downloaded) ...")
    df = load_unsw_raw()

    missing = [c for c in REQUIRED_SCHEMA_FIELDS if c not in df.columns]
    if missing:
        artifact["schema"] = {"n_rows": len(df), "n_features": 0, "features": []}
        _fail(f"required schema field(s) absent: {missing}", artifact)

    feature_cols = raw_feature_columns(df)
    artifact["schema"] = {
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "features": feature_cols,
        "excluded_columns": {
            "identifiers": ["srcip", "sport", "dstip", "dsport"],
            "timestamps_used_only_for_split": ["Stime", "Ltime"],
            "categorical_no_numeric_unit": ["proto", "service", "state"],
            "label_columns": ["attack_cat", "label"],
        },
        "n_exact_duplicate_rows_dropped": int(df.attrs.get("n_exact_duplicate_rows_dropped", -1)),
    }

    # --- Step C: labels -----------------------------------------------------------------------
    original_counts = df["attack_cat"].value_counts().to_dict()
    binary_mapping = {cat: (0 if cat == "Normal" else 1) for cat in original_counts}
    binary_counts = df["label"].value_counts().to_dict()
    artifact["labels"] = {
        "original_counts": {str(k): int(v) for k, v in original_counts.items()},
        "binary_mapping": binary_mapping,
        "binary_counts": {str(k): int(v) for k, v in binary_counts.items()},
    }

    # --- Step D: split + leakage check ---------------------------------------------------------
    train_df, test_df = temporal_split_unsw(df, train_frac=SECONDARY_TRAIN_FRAC)
    train_split = SecondarySplit(
        features=train_df[feature_cols].to_numpy(dtype=float), flow_ids=flow_row_ids(train_df)
    )
    test_split = SecondarySplit(
        features=test_df[feature_cols].to_numpy(dtype=float), flow_ids=flow_row_ids(test_df)
    )

    artifact["split"] = {
        "method": "temporal_80_20_on_deduplicated_stime_sorted_superset",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "leakage_checks": {
            "n_exact_duplicate_rows_dropped_before_split": int(
                df.attrs.get("n_exact_duplicate_rows_dropped", -1)
            ),
        },
    }
    try:
        validate_disjoint_split(train_split, test_split)
        artifact["split"]["leakage_checks"]["train_test_row_overlap"] = "none_detected"
    except ValueError as e:
        artifact["split"]["leakage_checks"]["train_test_row_overlap"] = str(e)
        _fail(str(e), artifact)

    # --- Step E: bounds fit on TRAIN ONLY -------------------------------------------------------
    bounds = fit_raw_bounds(train_split.features)
    artifact["bounds_fit_partition"] = "train_only"

    # --- Step F: constraint manifest + clean-data coverage ---------------------------------------
    constraints = build_unsw_constraints()
    try:
        validate_constraint_manifest(constraints)
    except ValueError as e:
        artifact["constraint_manifest"] = {"n_constraints": len(constraints), "relations": [],
                                            "train_valid_rate": 0.0, "test_valid_rate": 0.0}
        _fail(str(e), artifact)

    train_benign = train_df[train_df["label"] == 0]
    test_benign = test_df[test_df["label"] == 0]
    try:
        train_overall, train_per_rel = evaluate_constraint_coverage(train_benign, constraints)
        test_overall, test_per_rel = evaluate_constraint_coverage(test_benign, constraints)
    except ValueError as e:
        artifact["constraint_manifest"] = {"n_constraints": len(constraints), "relations": [],
                                            "train_valid_rate": 0.0, "test_valid_rate": 0.0}
        _fail(str(e), artifact)

    train_valid_rate = float(train_overall.mean())
    test_valid_rate = float(test_overall.mean())

    relations_summary = [
        {
            "name": c["name"], "kind": c["kind"], "source": c["source"],
            "description": c["description"], "features": c["features"], "tolerance": c["tolerance"],
            "train_benign_valid_rate": train_per_rel[c["name"]],
            "test_benign_valid_rate": test_per_rel[c["name"]],
        }
        for c in constraints
    ]
    artifact["constraint_manifest"] = {
        "n_constraints": len(constraints),
        "relations": relations_summary,
        "train_valid_rate": train_valid_rate,
        "test_valid_rate": test_valid_rate,
        "n_train_benign_rows": int(len(train_benign)),
        "n_test_benign_rows": int(len(test_benign)),
        "predeclared_tolerance": SECONDARY_CLEAN_VALID_TOLERANCE,
    }

    if train_valid_rate < SECONDARY_CLEAN_VALID_TOLERANCE:
        _fail(
            f"clean TRAIN constraint validity {train_valid_rate:.6f} below tolerance "
            f"{SECONDARY_CLEAN_VALID_TOLERANCE}", artifact,
        )
    if test_valid_rate < SECONDARY_CLEAN_VALID_TOLERANCE:
        _fail(
            f"clean TEST constraint validity {test_valid_rate:.6f} below tolerance "
            f"{SECONDARY_CLEAN_VALID_TOLERANCE}", artifact,
        )

    # --- pass ------------------------------------------------------------------------------------
    artifact["passed"] = True
    OUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=False))
    print(f"GATE PASSED. Wrote {OUT_PATH}")
    print(f"train_valid_rate={train_valid_rate:.6f} test_valid_rate={test_valid_rate:.6f} "
          f"(tolerance={SECONDARY_CLEAN_VALID_TOLERANCE})")


if __name__ == "__main__":
    main()
