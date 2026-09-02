"""Backward-compatibility shim for the UNSW-NB15 secondary-dataset module (plan Task 2), kept after
its Task 3 migration into `src/data_secondary.py` and `src/constraints_secondary.py`.

**Migration note (plan Task 3, 2026-07-31):** this module originally (Task 2) owned BOTH the raw-data
load/clean/dedupe/split/bounds machinery AND an 8-entry DataFrame-based constraint manifest
(`kind: "relation" | "bound"`). Task 3 asked for a narrower, ndarray-based constraint layer with a
different taxonomy (`kind: "projectable" | "audit_only"`) plus a `project_secondary_to_feasible`
projector that Task 2 never built. Rather than duplicate the download/clean/dedup/split code in a
third place, the load/split/bounds portion was MOVED to `src/data_secondary.py` (re-exported below
for backward compatibility), and the new ndarray constraint layer was built fresh in
`src/constraints_secondary.py` rather than adapted in place, because the two constraint
representations serve genuinely different consumers:

  - `build_unsw_constraints` (THIS module, unchanged below) operates on the full raw DataFrame
    (including `Stime`/`Ltime`, which are excluded from the classifier/trigger feature contract) and
    is what `scripts/13_secondary_data_gate.py` and the original `tests/test_secondary_data.py`
    (both committed under Task 2) import by name. Neither was rewritten by Task 3 -- the gate script
    needed no changes and its 8-entry manifest (including `ltime_after_stime`, which references the
    excluded timestamp columns) is still the right tool for a full-dataframe data-integrity gate.
  - `src.constraints_secondary.build_secondary_constraints` (NEW, Task 3) operates on the raw ndarray
    `x_train_raw`/`x_test_raw` a model or trigger actually sees (38 columns, no timestamps), uses the
    projectable/audit_only taxonomy, and is what `src.data_secondary.load_secondary_setup` and any
    downstream trigger-construction code should use going forward.

Everything below `SecondaryConstraint manifest (unchanged from Task 2)` is verbatim from the original
Task 2 module. Everything above it is a re-export of names now defined in `src/data_secondary.py`.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# --- tolerance constants: owned by src.constraints_secondary (the newer, ndarray-based module),
# imported here rather than redefined so the two constraint layers cannot silently drift apart. Both
# manifests were derived from the same evidentiary measurements (see the decision note), so sharing
# one set of numbers is correct, not coincidental.
from src.constraints_secondary import (
    CONSTRAINT_TOL_BOUND,
    CONSTRAINT_TOL_EQUALITY,
    CONSTRAINT_TOL_RELATIVE,
    MTU_BYTES,
)

# --- re-exported from src.data_secondary (moved there by Task 3; see the module docstring) --------
from src.data_secondary import (  # noqa: F401 -- re-exported for backward compatibility
    CATEGORICAL_EXCLUDED,
    EXCLUDED_COLUMNS,
    HF_FEATURES_DOC,
    HF_REPO_ID,
    HF_SHARD_FILES,
    ID_COLUMNS,
    LABEL_COL,
    LICENSE,
    RawBounds,
    SECONDARY_CLEAN_VALID_TOLERANCE,
    SECONDARY_TRAIN_FRAC,
    SOURCE_URL_MIRROR,
    SOURCE_URL_OFFICIAL,
    SecondarySplit,
    TARGET_RAW,
    TIME_COLUMNS,
    download_unsw,
    fit_raw_bounds,
    flow_row_ids,
    load_unsw_raw,
    raw_feature_columns,
    sha256_of_file,
    temporal_split_unsw,
    validate_disjoint_split,
)

# --- SecondaryConstraint manifest (unchanged from Task 2) ------------------------------------------
# Each relation is tagged with its evidentiary source:
#   "documented"  -- stated by the dataset's own NUSW-NB15_features.csv description or standard
#                    protocol semantics (TCP handshake timing, IP TTL byte width).
#   "empirical"   -- not asserted anywhere in the documentation, but verified to hold at coverage
#                    1.0 on the clean (label==0) training AND test partitions (measured 2026-07-31,
#                    see the decision note) before being trusted as a constraint.
# Tolerance constants (CONSTRAINT_TOL_EQUALITY, CONSTRAINT_TOL_RELATIVE, CONSTRAINT_TOL_BOUND,
# MTU_BYTES) are imported above from src.constraints_secondary, not redefined here.


def _rel_ltime_after_stime(df: pd.DataFrame) -> np.ndarray:
    return (df["Ltime"] >= df["Stime"]).to_numpy()


def _rel_tcprtt_handshake(df: pd.DataFrame) -> np.ndarray:
    ok = np.ones(len(df), dtype=bool)
    m = (df["tcprtt"] > 0).to_numpy()
    diff = (df["tcprtt"] - (df["synack"] + df["ackdat"])).abs().to_numpy()
    ok[m] = diff[m] <= CONSTRAINT_TOL_EQUALITY
    return ok


def _rel_sttl_bound(df: pd.DataFrame) -> np.ndarray:
    return df["sttl"].between(0, 255).to_numpy()


def _rel_dttl_bound(df: pd.DataFrame) -> np.ndarray:
    return df["dttl"].between(0, 255).to_numpy()


def _rel_smeansz_matches_sbytes(df: pd.DataFrame) -> np.ndarray:
    ok = np.ones(len(df), dtype=bool)
    m = (df["Spkts"] > 0).to_numpy()
    approx = (df["smeansz"] * df["Spkts"]).to_numpy()
    sbytes = df["sbytes"].to_numpy().astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        relerr = np.abs(approx[m] - sbytes[m]) / np.where(sbytes[m] == 0, np.nan, sbytes[m])
    ok[m] = np.nan_to_num(relerr, nan=0.0) <= CONSTRAINT_TOL_RELATIVE
    return ok


def _rel_dmeansz_matches_dbytes(df: pd.DataFrame) -> np.ndarray:
    ok = np.ones(len(df), dtype=bool)
    m = (df["Dpkts"] > 0).to_numpy()
    approx = (df["dmeansz"] * df["Dpkts"]).to_numpy()
    dbytes = df["dbytes"].to_numpy().astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        relerr = np.abs(approx[m] - dbytes[m]) / np.where(dbytes[m] == 0, np.nan, dbytes[m])
    ok[m] = np.nan_to_num(relerr, nan=0.0) <= CONSTRAINT_TOL_RELATIVE
    return ok


def _rel_src_mtu(df: pd.DataFrame) -> np.ndarray:
    ok = np.ones(len(df), dtype=bool)
    m = (df["Spkts"] > 0).to_numpy()
    mean_pkt = (df["sbytes"].to_numpy()[m]) / (df["Spkts"].to_numpy()[m])
    ok[m] = mean_pkt <= MTU_BYTES + CONSTRAINT_TOL_BOUND
    return ok


def _rel_dst_mtu(df: pd.DataFrame) -> np.ndarray:
    ok = np.ones(len(df), dtype=bool)
    m = (df["Dpkts"] > 0).to_numpy()
    mean_pkt = (df["dbytes"].to_numpy()[m]) / (df["Dpkts"].to_numpy()[m])
    ok[m] = mean_pkt <= MTU_BYTES + CONSTRAINT_TOL_BOUND
    return ok


def build_unsw_constraints() -> List[Dict]:
    """The UNSW-NB15 raw-feature constraint manifest: 8 entries -- 6 cross-feature relations plus 2
    single-feature box/clip bounds (sttl, dttl), each with a `check(df) -> bool array` evaluator plus
    its evidentiary source. `validate_constraint_manifest` counts only the 6 `kind == "relation"`
    entries toward the Feasibility gate's "non-trivial relation" requirement -- the 2 `kind == "bound"`
    entries are real, coverage-checked constraints (useful for `project_to_feasible`-style clipping
    later) but do not by themselves satisfy "at least two non-trivial ... relations" per the gate's
    explicit "Reject if: Only min/max clipping is possible" clause.

    Unchanged from Task 2. See `src.constraints_secondary.build_secondary_constraints` for the newer,
    ndarray-based, projectable/audit_only-tagged manifest used by Task 3 onward -- that manifest
    intentionally drops `ltime_after_stime` (Stime/Ltime are outside the classifier feature
    contract), so it has 7 entries rather than this function's 8."""
    return [
        {"name": "ltime_after_stime", "kind": "relation", "source": "documented",
         "description": "Flow end time is not before flow start time.",
         "features": ["Stime", "Ltime"], "tolerance": 0.0, "check": _rel_ltime_after_stime},
        {"name": "tcprtt_equals_synack_plus_ackdat", "kind": "relation", "source": "documented",
         "description": "TCP round-trip time decomposes into the SYN-ACK and ACK-data legs "
                         "(only meaningful where tcprtt > 0, i.e. a handshake was captured).",
         "features": ["tcprtt", "synack", "ackdat"], "tolerance": CONSTRAINT_TOL_EQUALITY,
         "check": _rel_tcprtt_handshake},
        {"name": "sttl_byte_range", "kind": "bound", "source": "documented",
         "description": "Source TTL is an IP header byte field, so it is bounded in [0, 255]. "
                         "Single-feature clip, not a cross-feature relation.",
         "features": ["sttl"], "tolerance": CONSTRAINT_TOL_BOUND, "check": _rel_sttl_bound},
        {"name": "dttl_byte_range", "kind": "bound", "source": "documented",
         "description": "Destination TTL is an IP header byte field, so it is bounded in [0, 255]. "
                         "Single-feature clip, not a cross-feature relation.",
         "features": ["dttl"], "tolerance": CONSTRAINT_TOL_BOUND, "check": _rel_dttl_bound},
        {"name": "smeansz_consistent_with_sbytes", "kind": "relation", "source": "empirical",
         "description": "Mean source packet size times source packet count reconstructs total "
                         "source bytes within 5% (smeansz is integer-rounded).",
         "features": ["smeansz", "Spkts", "sbytes"], "tolerance": CONSTRAINT_TOL_RELATIVE,
         "check": _rel_smeansz_matches_sbytes},
        {"name": "dmeansz_consistent_with_dbytes", "kind": "relation", "source": "empirical",
         "description": "Mean destination packet size times destination packet count reconstructs "
                         "total destination bytes within 5%.",
         "features": ["dmeansz", "Dpkts", "dbytes"], "tolerance": CONSTRAINT_TOL_RELATIVE,
         "check": _rel_dmeansz_matches_dbytes},
        {"name": "src_mean_packet_size_le_mtu", "kind": "relation", "source": "empirical",
         "description": "Mean source packet size (sbytes / Spkts) does not exceed the 1500-byte "
                         "Ethernet MTU (the same relation form as the CTU-13 packet-size constraint).",
         "features": ["sbytes", "Spkts"], "tolerance": CONSTRAINT_TOL_BOUND, "check": _rel_src_mtu},
        {"name": "dst_mean_packet_size_le_mtu", "kind": "relation", "source": "empirical",
         "description": "Mean destination packet size (dbytes / Dpkts) does not exceed the "
                         "1500-byte Ethernet MTU.",
         "features": ["dbytes", "Dpkts"], "tolerance": CONSTRAINT_TOL_BOUND, "check": _rel_dst_mtu},
    ]


def validate_constraint_manifest(relations: Sequence[Dict]) -> None:
    """Raise ValueError if the manifest has no non-trivial (cross-feature) relation constraint --
    a manifest containing only box/clip bounds does not support a defensible constraint-projected
    trigger (the Feasibility gate's explicit reject condition)."""
    if not relations:
        raise ValueError("constraint manifest has no non-trivial relation constraints")
    non_bound = [r for r in relations if r.get("kind") != "bound"]
    if not non_bound:
        raise ValueError(
            "constraint manifest has no non-trivial relation constraints (only box/clip bounds)"
        )


def evaluate_constraint_coverage(
    df: pd.DataFrame, constraints: Sequence[Dict]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Per-row AND across all constraints, plus a per-constraint coverage-rate dict."""
    overall = np.ones(len(df), dtype=bool)
    per_constraint: Dict[str, float] = {}
    for c in constraints:
        ok = c["check"](df)
        if not np.all(np.isfinite(np.asarray(ok, dtype=float))):
            raise ValueError(f"constraint {c['name']!r} produced a non-finite validity value")
        per_constraint[c["name"]] = float(np.mean(ok))
        overall &= ok
    return overall, per_constraint
