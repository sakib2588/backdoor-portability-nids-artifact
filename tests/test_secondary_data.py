"""Task 2: data-integrity guards for the secondary (UNSW-NB15) dataset gate.

These tests run on synthetic/toy data only -- no network access, no download. The real gate
(scripts/13_secondary_data_gate.py) exercises the same functions against the actual downloaded
dataset and is verified separately by running that script.

Task 3 addition (bottom of the file): guard tests for `src.data_secondary.load_secondary_setup`'s
"must never silently return CTU-13" invariant, and for the audit-JSON feature-order check. Both run
on synthetic data / a patched `config.RESULTS`, with no download -- the guard raises before any I/O.
"""
import json
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from src import config
from src.secondary_data import (
    SecondarySplit,
    build_unsw_constraints,
    evaluate_constraint_coverage,
    fit_raw_bounds,
    flow_row_ids,
    validate_constraint_manifest,
    validate_disjoint_split,
)


def toy_secondary_split(
    shared_id: bool = False, n_train: int = 20, n_test: int = 10, n_features: int = 3, seed: int = 0
) -> Tuple[SecondarySplit, SecondarySplit]:
    """Synthetic train/test pair for the data-integrity unit tests. Test-only scaffolding -- lives
    here rather than in src/secondary_data.py so it is not importable by real pipeline code. Train
    and test are drawn from disjoint numeric ranges so their per-feature maxima never coincide by
    chance, and `shared_id=True` injects one deliberately duplicated identifier to exercise the
    overlap-rejection path."""
    rng = np.random.default_rng(seed)
    train_features = rng.uniform(0.0, 10.0, size=(n_train, n_features))
    test_features = rng.uniform(20.0, 30.0, size=(n_test, n_features))
    # dtype=object (not numpy fixed-width '<U..') so a later item assignment cannot silently
    # truncate a longer string to the array's original max width.
    train_ids = np.array([f"train-{i}" for i in range(n_train)], dtype=object)
    test_ids = np.array([f"test-{i}" for i in range(n_test)], dtype=object)
    if shared_id:
        test_ids = test_ids.copy()
        test_ids[0] = train_ids[0]
    train = SecondarySplit(features=train_features, flow_ids=train_ids)
    test = SecondarySplit(features=test_features, flow_ids=test_ids)
    return train, test


def test_secondary_bounds_are_fit_on_train_only():
    train, test = toy_secondary_split()
    bounds = fit_raw_bounds(train.features)
    assert bounds.upper[0] == train.features[:, 0].max()
    assert bounds.upper[0] != test.features[:, 0].max()


def test_secondary_audit_rejects_train_test_overlap():
    train, test = toy_secondary_split(shared_id=True)
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_split(train, test)


def test_secondary_audit_rejects_constraint_manifest_with_no_relations():
    with pytest.raises(ValueError, match="non-trivial relation"):
        validate_constraint_manifest([])


def test_secondary_audit_accepts_disjoint_split():
    train, test = toy_secondary_split(shared_id=False)
    validate_disjoint_split(train, test)  # must not raise


def test_secondary_bounds_lower_matches_train_min():
    train, _ = toy_secondary_split()
    bounds = fit_raw_bounds(train.features)
    assert np.allclose(bounds.lower, train.features.min(axis=0))
    assert np.allclose(bounds.upper, train.features.max(axis=0))


def test_secondary_audit_rejects_manifest_with_only_bound_constraints():
    bound_only = [{"name": "x_in_range", "kind": "bound", "check": lambda df: np.ones(1, dtype=bool)}]
    with pytest.raises(ValueError, match="non-trivial relation"):
        validate_constraint_manifest(bound_only)


def test_secondary_constraint_manifest_has_at_least_two_relations():
    relations = build_unsw_constraints()
    non_bound = [r for r in relations if r.get("kind") != "bound"]
    assert len(non_bound) >= 2
    validate_constraint_manifest(relations)  # must not raise


def test_secondary_constraint_coverage_rejects_non_finite():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    bad_constraint = [{"name": "nan_check", "kind": "relation",
                        "check": lambda d: np.array([np.nan, 1.0])}]
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_constraint_coverage(df, bad_constraint)


def test_flow_row_ids_flags_exact_duplicate_rows_as_equal():
    df = pd.DataFrame({
        "Stime": [1, 1, 2],
        "sbytes": [100, 100, 200],
        "label": [0, 0, 1],
        "attack_cat": ["Normal", "Normal", "DoS"],
    })
    ids = flow_row_ids(df)
    assert ids[0] == ids[1]     # exact duplicate row (excluding attack_cat) -> same fingerprint
    assert ids[0] != ids[2]     # distinct row -> different fingerprint


# --- per-relation violation detection -----------------------------------------------------------
# 1.0 coverage on real clean data (verified by running scripts/13_secondary_data_gate.py) cannot
# distinguish "the relation genuinely holds" from "the check function is vacuously true due to a
# bug" (wrong column, flipped comparison sign, always-True mask, etc.). Each case below is a single
# synthetic row that satisfies every OTHER relation's baseline values but is deliberately engineered
# to violate exactly the named relation, proving the check function actually flags it.

def _baseline_row() -> dict:
    """Values that satisfy all 6 non-bound relations simultaneously (a "clean" row)."""
    return {
        "Stime": 100, "Ltime": 200,                 # Ltime >= Stime
        "tcprtt": 0.05, "synack": 0.02, "ackdat": 0.03,  # tcprtt == synack + ackdat, tcprtt > 0
        "smeansz": 50, "Spkts": 10, "sbytes": 500,   # smeansz * Spkts == sbytes
        "dmeansz": 60, "Dpkts": 5, "dbytes": 300,    # dmeansz * Dpkts == dbytes
        # sbytes/Spkts = 50 <= 1500 MTU; dbytes/Dpkts = 60 <= 1500 MTU (both baseline-valid too)
    }


_RELATION_VIOLATIONS = {
    "ltime_after_stime": {"Ltime": 50},                            # Ltime (50) < Stime (100)
    "tcprtt_equals_synack_plus_ackdat": {"tcprtt": 0.5},           # 0.5 != synack+ackdat (0.05)
    "smeansz_consistent_with_sbytes": {"sbytes": 5000},            # smeansz*Spkts=500, far off 5000
    "dmeansz_consistent_with_dbytes": {"dbytes": 3000},            # dmeansz*Dpkts=300, far off 3000
    "src_mean_packet_size_le_mtu": {"sbytes": 20000, "Spkts": 1},  # mean pkt size 20000 > 1500 MTU
    "dst_mean_packet_size_le_mtu": {"dbytes": 20000, "Dpkts": 1},  # mean pkt size 20000 > 1500 MTU
}


@pytest.mark.parametrize("relation_name,overrides", sorted(_RELATION_VIOLATIONS.items()))
def test_each_relation_check_flags_a_deliberately_bad_row(relation_name, overrides):
    row = _baseline_row()
    row.update(overrides)
    df = pd.DataFrame([row])

    relations = build_unsw_constraints()
    target = next(r for r in relations if r["name"] == relation_name)
    assert target["kind"] == "relation"   # sanity: this table covers non-trivial relations only

    _, per_constraint = evaluate_constraint_coverage(df, [target])
    assert per_constraint[relation_name] == 0.0


def test_relation_violation_table_covers_every_non_bound_relation():
    """Guards against silently forgetting to add a case above when a new relation is added."""
    relations = build_unsw_constraints()
    non_bound_names = {r["name"] for r in relations if r["kind"] != "bound"}
    assert non_bound_names == set(_RELATION_VIOLATIONS.keys())


# --- Task 3: load_secondary_setup()'s "must never silently return CTU-13" guard ------------------
# These import from src.data_secondary (Task 3's new module, not the Task 2 src.secondary_data
# shim tested above) and exercise only the guard clauses, which run BEFORE any download/load call --
# so no network access happens here, matching the project's "no network download inside unit tests"
# requirement.

def test_load_secondary_setup_rejects_ctu_13_neris_dataset_id():
    from src.data_secondary import load_secondary_setup

    with pytest.raises(ValueError, match="ctu_13_neris"):
        load_secondary_setup(config.PRIMARY_DATASET_ID)


def test_load_secondary_setup_rejects_unknown_dataset_id():
    from src.data_secondary import load_secondary_setup

    with pytest.raises(ValueError, match="unsupported dataset_id"):
        load_secondary_setup("some_other_dataset")


def test_config_dataset_id_constants_are_distinct():
    """The guard in load_secondary_setup only works if these two constants can never collide."""
    assert config.PRIMARY_DATASET_ID != config.SECONDARY_DATASET_ID
    assert config.SECONDARY_DATASET_ID == "unsw_nb15"
    assert config.PRIMARY_DATASET_ID == "ctu_13_neris"


# --- Task 3: audit-JSON feature-order validation --------------------------------------------------

def test_validate_feature_order_noop_when_audit_artifact_absent(tmp_path, monkeypatch):
    from src.data_secondary import _validate_feature_order_against_audit

    monkeypatch.setattr(config, "RESULTS", tmp_path)  # no secondary_data_audit.json in tmp_path
    _validate_feature_order_against_audit(["a", "b", "c"])  # must not raise


def test_validate_feature_order_accepts_matching_order(tmp_path, monkeypatch):
    from src.data_secondary import _validate_feature_order_against_audit

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    (tmp_path / "secondary_data_audit.json").write_text(
        json.dumps({"schema": {"features": ["a", "b", "c"]}})
    )
    _validate_feature_order_against_audit(["a", "b", "c"])  # must not raise


def test_validate_feature_order_rejects_mismatched_order(tmp_path, monkeypatch):
    from src.data_secondary import _validate_feature_order_against_audit

    monkeypatch.setattr(config, "RESULTS", tmp_path)
    (tmp_path / "secondary_data_audit.json").write_text(
        json.dumps({"schema": {"features": ["a", "b", "c"]}})
    )
    with pytest.raises(ValueError, match="feature order mismatch"):
        _validate_feature_order_against_audit(["a", "c", "b"])


# --- Task 3: SecondarySetup dataclass shape (constructed from synthetic components, no I/O) -------

def test_secondary_setup_bundles_train_only_fitted_bounds_and_scaler():
    from sklearn.preprocessing import StandardScaler

    from src.constraints_secondary import build_secondary_constraints
    from src.data_secondary import SecondarySetup

    feature_names = (
        "dur", "tcprtt", "synack", "ackdat", "sttl", "dttl",
        "smeansz", "Spkts", "sbytes", "dmeansz", "Dpkts", "dbytes",
    )
    rng = np.random.default_rng(0)
    x_train = rng.uniform(1.0, 100.0, size=(30, len(feature_names)))
    x_test = rng.uniform(1.0, 100.0, size=(10, len(feature_names)))
    scaler = StandardScaler().fit(x_train)
    constraints = build_secondary_constraints(feature_names)

    setup = SecondarySetup(
        dataset_id="unsw_nb15",
        x_train_raw=x_train,
        y_train=np.zeros(30, dtype=np.int64),
        x_test_raw=x_test,
        y_test=np.zeros(10, dtype=np.int64),
        feature_names=feature_names,
        bounds=(x_train.min(axis=0), x_train.max(axis=0)),
        scaler=scaler,
        constraints=constraints,
        constraint_manifest={"n_constraints": len(constraints)},
    )

    assert setup.dataset_id == "unsw_nb15"
    assert setup.x_train_raw.shape == (30, len(feature_names))
    assert len(setup.constraints) == 7
    # frozen dataclass -- reassignment must fail
    with pytest.raises(Exception):
        setup.dataset_id = "something_else"
