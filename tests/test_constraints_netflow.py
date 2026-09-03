import pandas as pd
import pytest

from src.constraints_netflow import (
    NETFLOW_RELATIONS, admitted_manifest, evaluate_relation, manifest_fingerprint,
    relation_satisfaction,
)


def _row(**kw):
    base = dict(
        IN_BYTES=1000.0, OUT_BYTES=800.0, IN_PKTS=10.0, OUT_PKTS=8.0,
        MIN_TTL=32.0, MAX_TTL=64.0, MIN_IP_PKT_LEN=40.0, MAX_IP_PKT_LEN=1500.0,
        SHORTEST_FLOW_PKT=40.0, LONGEST_FLOW_PKT=1500.0,
        RETRANSMITTED_IN_BYTES=0.0, RETRANSMITTED_OUT_BYTES=0.0,
        RETRANSMITTED_IN_PKTS=0.0, RETRANSMITTED_OUT_PKTS=0.0,
        NUM_PKTS_UP_TO_128_BYTES=5.0, NUM_PKTS_128_TO_256_BYTES=4.0,
        NUM_PKTS_256_TO_512_BYTES=3.0, NUM_PKTS_512_TO_1024_BYTES=3.0,
        NUM_PKTS_1024_TO_1514_BYTES=3.0,
        FLOW_DURATION_MILLISECONDS=5000.0, DURATION_IN=2.0, DURATION_OUT=1.0,
    )
    base.update(kw)
    return pd.DataFrame([base])


@pytest.mark.parametrize("name", [r["name"] for r in NETFLOW_RELATIONS])
def test_every_relation_holds_on_a_valid_row(name):
    rel = next(r for r in NETFLOW_RELATIONS if r["name"] == name)
    assert bool(evaluate_relation(rel, _row()).all())


@pytest.mark.parametrize("name,bad", [
    ("ttl_min_le_max", dict(MIN_TTL=200.0)),
    ("retrans_in_bytes_le_in_bytes", dict(RETRANSMITTED_IN_BYTES=5000.0)),
    ("retrans_out_bytes_le_out_bytes", dict(RETRANSMITTED_OUT_BYTES=5000.0)),
    ("retrans_in_pkts_le_in_pkts", dict(RETRANSMITTED_IN_PKTS=50.0)),
    ("retrans_out_pkts_le_out_pkts", dict(RETRANSMITTED_OUT_PKTS=50.0)),
    ("pkt_buckets_le_total_pkts", dict(NUM_PKTS_UP_TO_128_BYTES=500.0)),
    ("ip_pkt_len_min_le_max", dict(MIN_IP_PKT_LEN=9000.0)),
    ("flow_pkt_short_le_long", dict(SHORTEST_FLOW_PKT=9000.0)),
    ("longest_pkt_le_max_ip_pkt_len", dict(LONGEST_FLOW_PKT=9000.0)),
    ("in_bytes_ge_pkts_times_min_len", dict(IN_BYTES=10.0)),
    ("nonnegative_counters", dict(IN_BYTES=-1.0)),
])
def test_relation_catches_its_violation(name, bad):
    rel = next(r for r in NETFLOW_RELATIONS if r["name"] == name)
    assert not bool(evaluate_relation(rel, _row(**bad)).all())


def test_in_bytes_relation_is_vacuous_without_inbound_packets():
    rel = next(r for r in NETFLOW_RELATIONS if r["name"] == "in_bytes_ge_pkts_times_min_len")
    assert bool(evaluate_relation(rel, _row(IN_PKTS=0.0, IN_BYTES=0.0)).all())


def test_fingerprint_is_stable():
    assert manifest_fingerprint() == manifest_fingerprint()
    assert len(manifest_fingerprint()) == 64


def test_relation_satisfaction_reports_missing_columns_as_none():
    df = _row().drop(columns=["MIN_TTL"])
    sat = relation_satisfaction(df)
    assert sat["ttl_min_le_max"] is None
    assert sat["retrans_in_bytes_le_in_bytes"] == 1.0


def test_admission_is_per_corpus_and_a_degenerate_corpus_cannot_veto():
    """The review's I4 fix: a corpus excluded from an arm must not remove relations from it."""
    full = {r["name"]: 1.0 for r in NETFLOW_RELATIONS}
    per_corpus = {
        "good_a": dict(full),
        "good_b": {**full, "retrans_in_bytes_le_in_bytes": 0.9999},
        "degenerate": {**full, "ttl_min_le_max": 0.2, "retrans_in_bytes_le_in_bytes": 0.1},
    }
    kept, report = admitted_manifest(per_corpus, corpora=["good_a", "good_b"], threshold=0.999)
    names = [r["name"] for r in kept]
    assert "ttl_min_le_max" in names and "retrans_in_bytes_le_in_bytes" in names
    assert report["corpora_considered"] == ["good_a", "good_b"]
    assert report["n_authored"] == len(NETFLOW_RELATIONS)
    assert report["n_expressible"] == len(NETFLOW_RELATIONS)
    assert report["n_admitted"] == len(kept)

    kept_all, report_all = admitted_manifest(per_corpus, corpora=list(per_corpus), threshold=0.999)
    dropped = {r["name"] for r in kept} - {r["name"] for r in kept_all}
    assert dropped == {"ttl_min_le_max", "retrans_in_bytes_le_in_bytes"}
    assert report_all["rejected"]["ttl_min_le_max"] == 0.2


def test_admission_refuses_a_corpus_it_has_no_measurement_for():
    with pytest.raises(KeyError):
        admitted_manifest({"a": {}}, corpora=["a", "b"])


# --------------------------------------------------------------------------------------------
# Projection layer
# --------------------------------------------------------------------------------------------
import numpy as np

from src.constraints_netflow import (
    ProjectionIncompleteWarning, as_columns, audit_netflow_constraints, check_array,
    project_netflow_to_feasible, repair_fingerprint,
)

FEATS = [
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "MIN_TTL", "MAX_TTL",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SHORTEST_FLOW_PKT", "LONGEST_FLOW_PKT",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_PKTS",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
]

PROJECTABLE = [r for r in NETFLOW_RELATIONS if r["kind_tag"] == "projectable"]


def _matrix(**kw):
    row = dict(
        IN_BYTES=1000.0, OUT_BYTES=800.0, IN_PKTS=10.0, OUT_PKTS=8.0,
        MIN_TTL=32.0, MAX_TTL=64.0, MIN_IP_PKT_LEN=40.0, MAX_IP_PKT_LEN=1500.0,
        SHORTEST_FLOW_PKT=40.0, LONGEST_FLOW_PKT=1500.0,
        RETRANSMITTED_IN_BYTES=0.0, RETRANSMITTED_OUT_BYTES=0.0,
        RETRANSMITTED_IN_PKTS=0.0, RETRANSMITTED_OUT_PKTS=0.0,
        NUM_PKTS_UP_TO_128_BYTES=5.0, NUM_PKTS_128_TO_256_BYTES=4.0,
        NUM_PKTS_256_TO_512_BYTES=3.0, NUM_PKTS_512_TO_1024_BYTES=3.0,
        NUM_PKTS_1024_TO_1514_BYTES=3.0,
        FLOW_DURATION_MILLISECONDS=5000.0, DURATION_IN=2.0, DURATION_OUT=1.0,
    )
    row.update(kw)
    return np.array([[row[f] for f in FEATS]], dtype=np.float64)


def _wide_bounds():
    return (np.zeros(len(FEATS)), np.full(len(FEATS), 1e9))


def test_column_view_and_frame_agree_on_every_relation():
    """The array path must go through the same relation body as the frame path, not a second one."""
    df = _row()
    x = _matrix()
    for rel in NETFLOW_RELATIONS:
        assert bool(evaluate_relation(rel, df).all()) == bool(check_array(rel, x, FEATS).all())


@pytest.mark.parametrize("bad", [
    dict(MIN_TTL=200.0),
    dict(RETRANSMITTED_IN_BYTES=5_000.0),
    dict(SHORTEST_FLOW_PKT=9_000.0),
    dict(LONGEST_FLOW_PKT=9_000.0),
    dict(MIN_IP_PKT_LEN=9_000.0),
    dict(IN_BYTES=-5.0),
    dict(RETRANSMITTED_OUT_PKTS=99.0),
])
def test_projection_restores_every_projectable_relation(bad):
    x = _matrix(**bad)
    assert not all(bool(check_array(r, x, FEATS).all()) for r in PROJECTABLE)
    p = project_netflow_to_feasible(x, NETFLOW_RELATIONS, FEATS, _wide_bounds())
    for rel in PROJECTABLE:
        assert bool(check_array(rel, p, FEATS).all()), rel["name"]


def test_projection_is_idempotent_and_deterministic():
    x = _matrix(MIN_TTL=200.0, RETRANSMITTED_IN_BYTES=5_000.0)
    once = project_netflow_to_feasible(x, NETFLOW_RELATIONS, FEATS, _wide_bounds())
    twice = project_netflow_to_feasible(once, NETFLOW_RELATIONS, FEATS, _wide_bounds())
    again = project_netflow_to_feasible(x, NETFLOW_RELATIONS, FEATS, _wide_bounds())
    assert np.array_equal(once, twice)
    assert np.array_equal(once, again)


def test_projection_leaves_a_feasible_row_untouched():
    x = _matrix()
    p = project_netflow_to_feasible(x, NETFLOW_RELATIONS, FEATS, _wide_bounds())
    assert np.array_equal(x, p)


def test_projection_moves_only_the_licensed_feature():
    """A stamped MIN_TTL survives; the repair raises MAX_TTL instead of erasing the trigger."""
    x = _matrix(MIN_TTL=200.0)
    p = project_netflow_to_feasible(x, NETFLOW_RELATIONS, FEATS, _wide_bounds())
    assert p[0, FEATS.index("MIN_TTL")] == 200.0
    assert p[0, FEATS.index("MAX_TTL")] == 200.0


def test_a_tight_bound_that_undoes_a_repair_warns_and_is_reported_by_the_audit():
    lo = np.zeros(len(FEATS))
    hi = np.full(len(FEATS), 1e9)
    hi[FEATS.index("MAX_TTL")] = 64.0          # too tight for the repair to reach
    x = _matrix(MIN_TTL=200.0)
    with pytest.warns(ProjectionIncompleteWarning):
        project_netflow_to_feasible(x, NETFLOW_RELATIONS, FEATS, (lo, hi))
    audit = audit_netflow_constraints(x, NETFLOW_RELATIONS, FEATS, bounds=(lo, hi))
    assert audit["after_projection"]["all_projectable_valid"] is False
    assert audit["after_projection"]["violations"]["ttl_min_le_max"] == 1


def test_audit_reports_before_and_after_projection():
    x = _matrix(MIN_TTL=200.0)
    audit = audit_netflow_constraints(x, NETFLOW_RELATIONS, FEATS, bounds=_wide_bounds())
    assert audit["all_projectable_valid"] is False
    assert audit["after_projection"]["all_projectable_valid"] is True
    assert audit["n_relations"] == len(NETFLOW_RELATIONS)


def test_audit_only_relations_are_not_projected():
    """pkt_buckets_le_total_pkts has no unambiguous single-target repair and must stay untouched."""
    x = _matrix(NUM_PKTS_UP_TO_128_BYTES=500.0)
    p = project_netflow_to_feasible(x, NETFLOW_RELATIONS, FEATS, _wide_bounds())
    assert p[0, FEATS.index("NUM_PKTS_UP_TO_128_BYTES")] == 500.0
    rel = next(r for r in NETFLOW_RELATIONS if r["name"] == "pkt_buckets_le_total_pkts")
    assert not bool(check_array(rel, p, FEATS).all())


def test_repair_fingerprint_distinguishes_repair_targets():
    """Four relations share the _repair_raise factory, so the hash must read the closure values."""
    base = repair_fingerprint()
    swapped = [dict(r) for r in NETFLOW_RELATIONS]
    ttl = next(r for r in swapped if r["name"] == "ttl_min_le_max")
    from src.constraints_netflow import _repair_raise
    ttl["repair"] = _repair_raise("MAX_IP_PKT_LEN", "MIN_TTL")
    assert repair_fingerprint(swapped) != base
    assert repair_fingerprint() == base


def test_repair_fingerprint_is_independent_of_the_check_fingerprint():
    assert repair_fingerprint() != manifest_fingerprint()
