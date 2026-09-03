"""Feasible-set relations over the NetFlow v2 schema shared by all four NF-* corpora.

Authoring a relation is a claim about what NetFlow can emit. That claim is checked against clean
rows before the relation is used: `admitted_manifest` keeps only relations that hold on at least
`config.NETFLOW_RELATION_ADMISSION_FRAC` of clean rows in every corpus an arm analyzes. The three
counts (authored, expressible over the feature space actually present, admitted by measurement) are
reported separately, because conflating them is exactly the error the UNSW manifest made and had to
correct from 8 to "8 authored, 7 expressible, 5 enforced".

No relation compares DURATION_IN to FLOW_DURATION_MILLISECONDS: the units may differ and that has
not been confirmed.

Every relation here is an INEQUALITY. None recomputes a feature from others the way UNSW-NB15's
`tcprtt := synack + ackdat` equality does, so no feature has to be withheld from trigger eligibility
on this corpus family: a stamped value survives projection unless it actually violates a relation,
in which case it is clipped, and that clipping is the realizability cost the study measures. This is
the same policy as `scripts/_m3_secondary_common.EXCLUDED_TRIGGER_FEATURES`, which excludes exactly
the one UNSW feature a repair recomputes and nothing else.

A relation is `projectable` when a deterministic, minimal, single-target repair exists for it, and
`audit_only` otherwise. Whether a relation is USED is a separate question answered by measurement in
`admitted_manifest`; the projector runs over relations that are both admitted and projectable, so the
kind tag never depends on the measurement it would otherwise circularly justify.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config

TOL = 1e-6

_KIND_PROJECTABLE = "projectable"
_KIND_AUDIT_ONLY = "audit_only"
_VALID_KINDS = (_KIND_PROJECTABLE, _KIND_AUDIT_ONLY)


class _Column:
    """One column, exposing the `.to_numpy()` a relation body calls on a DataFrame column."""

    __slots__ = ("_a",)

    def __init__(self, a: np.ndarray):
        self._a = a

    def to_numpy(self) -> np.ndarray:
        return self._a


class _Columns:
    """Name-indexed view over a raw 2-D array, quacking like the DataFrame a relation expects.

    The relation bodies are the single source of truth for what each constraint means. Rather than
    write a second ndarray implementation of each one for the projector to use -- two code paths
    computing the same quantity is how this project once got 0.9828 and 0.9814 for one number -- the
    array side adapts itself to the frame API instead. Views only, no copies.
    """

    __slots__ = ("_x", "_idx")

    def __init__(self, x: np.ndarray, feature_names: Sequence[str]):
        self._x = x
        self._idx = {n: i for i, n in enumerate(feature_names)}

    def __getitem__(self, key):
        if isinstance(key, list):
            return _Column(self._x[:, [self._idx[n] for n in key]])
        return _Column(self._x[:, self._idx[key]])

    def __contains__(self, key) -> bool:
        return key in self._idx

    @property
    def columns(self):
        return list(self._idx)

    def index_of(self, name: str) -> int:
        return self._idx[name]


def as_columns(x_raw: np.ndarray, feature_names: Sequence[str]) -> _Columns:
    """Wrap a raw feature matrix so the relation bodies can read it by column name."""
    return _Columns(np.asarray(x_raw, dtype=np.float64), feature_names)


def _le(df, left: str, right: str) -> np.ndarray:
    return df[left].to_numpy() <= df[right].to_numpy() + TOL


def _rel_ttl_min_le_max(df: pd.DataFrame) -> np.ndarray:
    return _le(df, "MIN_TTL", "MAX_TTL")


def _rel_ip_pkt_len_min_le_max(df: pd.DataFrame) -> np.ndarray:
    return _le(df, "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN")


def _rel_flow_pkt_short_le_long(df: pd.DataFrame) -> np.ndarray:
    return _le(df, "SHORTEST_FLOW_PKT", "LONGEST_FLOW_PKT")


def _rel_retrans_in_bytes_le_in_bytes(df: pd.DataFrame) -> np.ndarray:
    return _le(df, "RETRANSMITTED_IN_BYTES", "IN_BYTES")


def _rel_retrans_out_bytes_le_out_bytes(df: pd.DataFrame) -> np.ndarray:
    return _le(df, "RETRANSMITTED_OUT_BYTES", "OUT_BYTES")


def _rel_retrans_in_pkts_le_in_pkts(df: pd.DataFrame) -> np.ndarray:
    return _le(df, "RETRANSMITTED_IN_PKTS", "IN_PKTS")


def _rel_retrans_out_pkts_le_out_pkts(df: pd.DataFrame) -> np.ndarray:
    return _le(df, "RETRANSMITTED_OUT_PKTS", "OUT_PKTS")


def _rel_pkt_buckets_le_total_pkts(df: pd.DataFrame) -> np.ndarray:
    buckets = (
        df["NUM_PKTS_UP_TO_128_BYTES"].to_numpy()
        + df["NUM_PKTS_128_TO_256_BYTES"].to_numpy()
        + df["NUM_PKTS_256_TO_512_BYTES"].to_numpy()
        + df["NUM_PKTS_512_TO_1024_BYTES"].to_numpy()
        + df["NUM_PKTS_1024_TO_1514_BYTES"].to_numpy()
    )
    return buckets <= df["IN_PKTS"].to_numpy() + df["OUT_PKTS"].to_numpy() + TOL


def _rel_in_bytes_ge_pkts_times_min_len(df: pd.DataFrame) -> np.ndarray:
    """A flow carrying k packets cannot carry fewer bytes than k times the smallest packet seen.
    Vacuously true when there are no inbound packets."""
    pkts = df["IN_PKTS"].to_numpy()
    lo = df["MIN_IP_PKT_LEN"].to_numpy()
    ok = df["IN_BYTES"].to_numpy() + TOL >= pkts * lo
    return np.where(pkts > 0, ok, True)


def _rel_longest_pkt_le_max_ip_pkt_len(df: pd.DataFrame) -> np.ndarray:
    return _le(df, "LONGEST_FLOW_PKT", "MAX_IP_PKT_LEN")


def _rel_nonnegative_counters(df: pd.DataFrame) -> np.ndarray:
    cols = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
            "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_OUT_BYTES",
            "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_PKTS",
            "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT"]
    present = [c for c in cols if c in df.columns]
    return (df[present].to_numpy() >= -TOL).all(axis=1)


def _repair_raise(target: str, source: str):
    """Restore `source <= target` by raising the target, the minimal move that keeps the source
    (which is what the attacker may have stamped) where it is."""
    def repair(x: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
        t, s = idx[target], idx[source]
        x[:, t] = np.maximum(x[:, t], x[:, s])
        return x
    return repair


def _repair_lower(target: str, source: str):
    """Restore `target <= source` by lowering the target. The target is always the derived counter
    (a retransmission subset), never the primary counter other relations read."""
    def repair(x: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
        t, s = idx[target], idx[source]
        x[:, t] = np.minimum(x[:, t], x[:, s])
        return x
    return repair


def _repair_in_bytes_floor(x: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
    """Raise IN_BYTES to the floor its own packet count and smallest packet imply."""
    pkts, lo, tgt = idx["IN_PKTS"], idx["MIN_IP_PKT_LEN"], idx["IN_BYTES"]
    floor = x[:, pkts] * x[:, lo]
    active = x[:, pkts] > 0
    x[:, tgt] = np.where(active, np.maximum(x[:, tgt], floor), x[:, tgt])
    return x


_NONNEGATIVE_COLUMNS = (
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
)


def _repair_nonnegative(x: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
    """Clamp counters at zero. Conditional, so a positive stamped value is never rewritten."""
    for name in _NONNEGATIVE_COLUMNS:
        if name in idx:
            j = idx[name]
            x[:, j] = np.maximum(x[:, j], 0.0)
    return x


NETFLOW_RELATIONS: List[Dict] = [
    dict(name="ttl_min_le_max", kind="inequality", fn=_rel_ttl_min_le_max,
         features=("MIN_TTL", "MAX_TTL"), kind_tag=_KIND_PROJECTABLE,
         repair=_repair_raise("MAX_TTL", "MIN_TTL"),
         doc="Observed minimum TTL cannot exceed observed maximum TTL."),
    dict(name="ip_pkt_len_min_le_max", kind="inequality", fn=_rel_ip_pkt_len_min_le_max,
         features=("MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN"), kind_tag=_KIND_PROJECTABLE,
         repair=_repair_raise("MAX_IP_PKT_LEN", "MIN_IP_PKT_LEN"),
         doc="Smallest IP packet length cannot exceed the largest."),
    dict(name="flow_pkt_short_le_long", kind="inequality", fn=_rel_flow_pkt_short_le_long,
         features=("SHORTEST_FLOW_PKT", "LONGEST_FLOW_PKT"), kind_tag=_KIND_PROJECTABLE,
         repair=_repair_raise("LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT"),
         doc="Shortest flow packet cannot exceed the longest."),
    dict(name="retrans_in_bytes_le_in_bytes", kind="inequality",
         fn=_rel_retrans_in_bytes_le_in_bytes,
         features=("RETRANSMITTED_IN_BYTES", "IN_BYTES"), kind_tag=_KIND_PROJECTABLE,
         repair=_repair_lower("RETRANSMITTED_IN_BYTES", "IN_BYTES"),
         doc="Retransmitted inbound bytes are a subset of inbound bytes."),
    dict(name="retrans_out_bytes_le_out_bytes", kind="inequality",
         fn=_rel_retrans_out_bytes_le_out_bytes,
         features=("RETRANSMITTED_OUT_BYTES", "OUT_BYTES"), kind_tag=_KIND_PROJECTABLE,
         repair=_repair_lower("RETRANSMITTED_OUT_BYTES", "OUT_BYTES"),
         doc="Retransmitted outbound bytes are a subset of outbound bytes."),
    dict(name="retrans_in_pkts_le_in_pkts", kind="inequality",
         fn=_rel_retrans_in_pkts_le_in_pkts,
         features=("RETRANSMITTED_IN_PKTS", "IN_PKTS"), kind_tag=_KIND_PROJECTABLE,
         repair=_repair_lower("RETRANSMITTED_IN_PKTS", "IN_PKTS"),
         doc="Retransmitted inbound packets are a subset of inbound packets."),
    dict(name="retrans_out_pkts_le_out_pkts", kind="inequality",
         fn=_rel_retrans_out_pkts_le_out_pkts,
         features=("RETRANSMITTED_OUT_PKTS", "OUT_PKTS"), kind_tag=_KIND_PROJECTABLE,
         repair=_repair_lower("RETRANSMITTED_OUT_PKTS", "OUT_PKTS"),
         doc="Retransmitted outbound packets are a subset of outbound packets."),
    dict(name="pkt_buckets_le_total_pkts", kind="inequality", fn=_rel_pkt_buckets_le_total_pkts,
         features=("NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
                   "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
                   "NUM_PKTS_1024_TO_1514_BYTES", "IN_PKTS", "OUT_PKTS"),
         kind_tag=_KIND_AUDIT_ONLY, repair=None,
         doc="Per-size packet buckets sum to no more than the total packet count."),
    dict(name="in_bytes_ge_pkts_times_min_len", kind="inequality",
         fn=_rel_in_bytes_ge_pkts_times_min_len,
         features=("IN_BYTES", "IN_PKTS", "MIN_IP_PKT_LEN"), kind_tag=_KIND_PROJECTABLE,
         repair=_repair_in_bytes_floor,
         doc="Inbound bytes are at least packets times the smallest packet length."),
    dict(name="longest_pkt_le_max_ip_pkt_len", kind="inequality",
         fn=_rel_longest_pkt_le_max_ip_pkt_len,
         features=("LONGEST_FLOW_PKT", "MAX_IP_PKT_LEN"), kind_tag=_KIND_PROJECTABLE,
         repair=_repair_raise("MAX_IP_PKT_LEN", "LONGEST_FLOW_PKT"),
         doc="Longest flow packet cannot exceed the largest IP packet length."),
    dict(name="nonnegative_counters", kind="inequality", fn=_rel_nonnegative_counters,
         features=("IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS"), kind_tag=_KIND_PROJECTABLE,
         repair=_repair_nonnegative,
         doc="Byte, packet and duration counters are non-negative."),
]


def evaluate_relation(rel: Dict, df: pd.DataFrame) -> np.ndarray:
    """Per-row boolean: does this row satisfy the relation."""
    return np.asarray(rel["fn"](df), dtype=bool)


def relation_satisfaction(df: pd.DataFrame,
                          relations: Sequence[Dict] = NETFLOW_RELATIONS) -> Dict[str, float]:
    """Fraction of rows satisfying each relation. A relation naming a column the frame does not
    carry is reported as None rather than silently skipped."""
    out: Dict[str, float] = {}
    for rel in relations:
        missing = [f for f in rel["features"] if f not in df.columns]
        out[rel["name"]] = None if missing else float(evaluate_relation(rel, df).mean())
    return out


def admitted_manifest(per_corpus: Dict[str, Dict[str, float]],
                      corpora: Sequence[str],
                      threshold: float = config.NETFLOW_RELATION_ADMISSION_FRAC,
                      ) -> Tuple[List[Dict], Dict]:
    """Admit relations that clear `threshold` on every corpus in `corpora`.

    Admission is scoped to the corpora an arm actually analyzes. A corpus excluded from an arm by
    pre-registration must not remove relations from it, or one degenerate synthetic corpus thins the
    feasible set for every other and weakens the realizability claim that depends on it.
    """
    missing = [c for c in corpora if c not in per_corpus]
    if missing:
        raise KeyError(f"no measured satisfaction for {missing}")
    scoped = {c: per_corpus[c] for c in corpora}
    names = [r["name"] for r in NETFLOW_RELATIONS]
    worst: Dict[str, float] = {}
    for name in names:
        vals = [scoped[c].get(name) for c in corpora]
        worst[name] = None if any(v is None for v in vals) else min(vals)

    expressible = [r for r in NETFLOW_RELATIONS if worst[r["name"]] is not None]
    kept = [r for r in expressible if worst[r["name"]] >= threshold]
    kept_names = {r["name"] for r in kept}
    report = dict(
        n_authored=len(NETFLOW_RELATIONS),
        n_expressible=len(expressible),
        n_admitted=len(kept),
        threshold=threshold,
        corpora_considered=list(corpora),
        admitted=[r["name"] for r in kept],
        rejected={r["name"]: worst[r["name"]]
                  for r in expressible if r["name"] not in kept_names},
        worst_case_satisfaction=worst,
        fingerprint=manifest_fingerprint(),
    )
    return kept, report


def manifest_fingerprint() -> str:
    """SHA-256 over the parsed source of every relation function with docstrings stripped, so a
    comment edit does not invalidate completed compute but a changed bound, tolerance or body does.
    Mirrors src.constraints_secondary.manifest_fingerprint's contract."""
    h = hashlib.sha256()
    for rel in NETFLOW_RELATIONS:
        src = inspect.getsource(rel["fn"])
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)
        h.update(rel["name"].encode())
        h.update(rel["kind"].encode())
        h.update(ast.dump(tree).encode())
    h.update(str(TOL).encode())
    return h.hexdigest()


class ProjectionIncompleteWarning(UserWarning):
    """Raised by `project_netflow_to_feasible` when the trailing box-clip undoes a repair.

    Fitted bounds come from real training rows, which never had to accommodate the value a repair
    needs to reach for an adversarially stamped row. When that happens the projector still returns a
    deterministic, idempotent fixed point, but not a fully feasible one. Python shows only the first
    occurrence of an identical warning tuple, so a caller that needs a persistent answer must read
    `audit_netflow_constraints(..., bounds=...)["after_projection"]["all_projectable_valid"]` rather
    than rely on seeing this fire.
    """


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr[None, :] if arr.ndim == 1 else arr


def check_array(rel: Dict, x_raw: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Evaluate a relation on a raw feature matrix, through the same body the frame path uses."""
    return evaluate_relation(rel, as_columns(x_raw, feature_names))


def project_netflow_to_feasible(x_raw,
                                relations: Sequence[Dict],
                                feature_names: Sequence[str],
                                bounds: Tuple[np.ndarray, np.ndarray],
                                max_iters: int = 50) -> np.ndarray:
    """Repair raw NetFlow rows onto every projectable relation in `relations`, minimally.

    Box-clip to `bounds`, then repair to a fixed point, clipping after each pass. Deterministic and
    idempotent: each repair writes only the one feature its own relation licenses, so a second call
    on an already-projected array is a single no-op pass. Relations tagged `audit_only` are left
    alone, and a caller must not describe the output as satisfying them.

    Mirrors `src.constraints_secondary.project_secondary_to_feasible`'s structure and its failure
    mode: the trailing clip can undo a repair when the fitted bounds are tighter than the value the
    repair needs, in which case a `ProjectionIncompleteWarning` names the still-violated relations.
    """
    lo = np.asarray(bounds[0], dtype=np.float64)
    hi = np.asarray(bounds[1], dtype=np.float64)
    idx = {n: i for i, n in enumerate(feature_names)}
    x = np.clip(_as_2d(x_raw).copy(), lo, hi)

    projectable = [r for r in relations
                   if r["kind_tag"] == _KIND_PROJECTABLE and r["repair"] is not None
                   and all(f in idx for f in r["features"])]
    for _ in range(max_iters):
        before = x.copy()
        for rel in projectable:
            x = np.clip(rel["repair"](x, idx), lo, hi)
        if np.array_equal(before, x):
            break

    still = [r["name"] for r in projectable if not check_array(r, x, feature_names).all()]
    if still:
        ok = np.logical_and.reduce(
            [check_array(r, x, feature_names) for r in projectable if r["name"] in still])
        warnings.warn(
            f"project_netflow_to_feasible: {still} still violated on {int((~ok).sum())} row(s) "
            f"after {max_iters} repair pass(es) -- the fitted bounds are likely tighter than the "
            "value the repair needed to reach, so the trailing box-clip undid it. Check "
            "audit_netflow_constraints(..., bounds=bounds)['after_projection']"
            "['all_projectable_valid'] before trusting a realizability claim on these rows.",
            ProjectionIncompleteWarning, stacklevel=2)
    return x


def audit_netflow_constraints(x_raw,
                              relations: Sequence[Dict],
                              feature_names: Sequence[str],
                              bounds: Tuple[np.ndarray, np.ndarray] = None) -> Dict[str, object]:
    """Per-relation violation counts plus an explicit `all_projectable_valid` signal.

    Pass `bounds` to also get `after_projection`: the same audit recomputed on the projector's
    output. That is the check that catches a repair silently undone by the box-clip, and it is what
    any realizability claim in the paper must be read from.
    """
    x = _as_2d(x_raw)
    usable = [r for r in relations if all(f in feature_names for f in r["features"])]

    def _audit(rows: np.ndarray) -> Dict[str, object]:
        violations = {r["name"]: int((~check_array(r, rows, feature_names)).sum()) for r in usable}
        denom = len(rows) * len(usable)
        valid_rate = 1.0 if denom == 0 else float(1 - sum(violations.values()) / denom)
        projectable = [r for r in usable if r["kind_tag"] == _KIND_PROJECTABLE]
        ok = np.ones(len(rows), dtype=bool)
        for r in projectable:
            ok &= check_array(r, rows, feature_names)
        return dict(n_rows=int(len(rows)), n_relations=len(usable), violations=violations,
                    valid_rate=valid_rate, all_projectable_valid=bool(ok.all()))

    out = _audit(x)
    if bounds is not None:
        out["after_projection"] = _audit(
            project_netflow_to_feasible(x, relations, feature_names, bounds))
    return out


def repair_fingerprint(relations: Sequence[Dict] = None) -> str:
    """SHA-256 over the projection layer: each relation's kind tag and its repair.

    Kept SEPARATE from `manifest_fingerprint`, which covers only the check bodies. Stage 1's data
    gate measures relation satisfaction and does not project anything, so a changed repair must not
    invalidate its samples; the poison sweep does project, so it folds both fingerprints into its
    checkpoint key. Closure cell values are hashed alongside the factory source, or the four
    `_repair_raise` closures would be indistinguishable from one another.
    """
    rels = NETFLOW_RELATIONS if relations is None else relations
    h = hashlib.sha256()
    for rel in rels:
        h.update(rel["name"].encode())
        h.update(rel["kind_tag"].encode())
        fn = rel["repair"]
        if fn is None:
            h.update(b"none")
            continue
        h.update(inspect.getsource(fn).encode())
        for cell in (fn.__closure__ or ()):
            h.update(repr(cell.cell_contents).encode())
    h.update(repr(_NONNEGATIVE_COLUMNS).encode())
    return h.hexdigest()
