"""Named, auditable raw-feature constraints for the UNSW-NB15 secondary dataset (plan Task 3).

This module owns the `SecondaryConstraint` dataclass, `audit_constraints`, the constraint
definitions themselves (re-derived from Task 2's 8-entry DataFrame-based manifest in
`src/secondary_data.py::build_unsw_constraints`, but re-tagged into a NEW `projectable` /
`audit_only` taxonomy -- NOT that manifest's `relation` / `bound` taxonomy), and
`project_secondary_to_feasible` (new for Task 3: nothing in Task 2 built a projector).

It deliberately does NOT import `src.tb_vendor` -- the CTU-13 constraint DSL and its projector
(`src.tb_vendor.constraints_numeric.project_to_feasible`) are specific to CTU-13 Neris's port-family
feature layout and its 360-constraint set. This module's own `project_secondary_to_feasible` follows
the SAME iterative repair pattern (box-clip, then repeatedly repair each violated projectable
constraint, re-clip, repeat to a fixed point) but is written from scratch against UNSW-NB15's own
38-column raw feature contract (`src.data_secondary.raw_feature_columns`), so this task's own
constraint checks stay independent of the CTU definitions as required.

Each `SecondaryConstraint.check` operates on a raw 2-D `(n_rows, n_features)` ndarray whose columns
are in the exact order of the `feature_names` tuple passed to `build_secondary_constraints` --
i.e. `src.data_secondary.SecondarySetup.feature_names` / `x_train_raw` / `x_test_raw`. Because that
feature contract EXCLUDES `Stime`/`Ltime` (timestamps, kept out of the classifier's/trigger's input
space by Task 2's decision -- see notes/20260729-decision-secondary-dataset-selection.md, "Raw
feature contract"), Task 2's `ltime_after_stime` relation has NO home here: it cannot be expressed as
a function of this feature space without adding two columns nothing downstream (model, SHAP, trigger)
is meant to see. It is intentionally NOT migrated into this module's manifest -- it remains exactly
where it always was, a full-dataframe integrity check inside `src.secondary_data.build_unsw_constraints`
/ `scripts/13_secondary_data_gate.py`, which still validates it against the raw (undropped) columns.
This module's manifest is therefore 7 entries (not 8): the other 7 of Task 2's 8 relations/bounds all
reference only feature-contract columns and are re-tagged below.

Projectable vs audit_only classification (the real judgment call this task asks for; each is also
reasoned at its definition below):
  - `ltime_after_stime`           -- NOT MIGRATED (see above; not expressible in this feature space).
  - `tcprtt_equals_synack_plus_ackdat` -- PROJECTABLE. Documented protocol semantics name `tcprtt` as
    the SUM of the `synack` and `ackdat` legs, so `tcprtt` is unambiguously the dependent quantity;
    repairing by setting `tcprtt := synack + ackdat` touches only the one feature the relation itself
    licenses, deterministically and idempotently.
  - `sttl_byte_range` / `dttl_byte_range` -- PROJECTABLE. Single-feature IP-header byte-width clips
    ([0, 255]); a box clip is the textbook projectable case (same as CTU's box-clip step).
  - `smeansz_consistent_with_sbytes` / `dmeansz_consistent_with_dbytes` -- AUDIT_ONLY. This is the
    "empirical rounding relationship" case: `smeansz`/`dmeansz` are documented as "mean packet size,
    integer-rounded by the extraction tool", and the relation only holds within a 5% tolerance BECAUSE
    of that undocumented rounding rule (we know it rounds; we do not know its exact rounding mode).
    Forcing exact consistency by recomputing e.g. `smeansz := sbytes / Spkts` would silently invent a
    value the real extraction tool might never emit (its rounding rule is unspecified), which is
    "changing an unspecified feature" in spirit even though the touched column IS one of the three --
    the AMBIGUITY is in which of the three to move and by how much, since all three are independently
    plausible attacker-controlled or donor-derived quantities and none is textually privileged as the
    dependent one the way `tcprtt` is. Kept audit-only rather than forced to a guessed value.
  - `src_mean_packet_size_le_mtu` / `dst_mean_packet_size_le_mtu` -- PROJECTABLE. Same `SafeDivision
    <= 1500` MTU shape as the CTU-13 constraint set's own 34 MTU constraints; the CTU projector's
    documented repair (raise the divisor -- here `Spkts`/`Dpkts` -- to the minimum value that brings
    the ratio back under the bound) is unambiguous and is reproduced here independently.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np


class ProjectionIncompleteWarning(UserWarning):
    """Raised by `project_secondary_to_feasible` when a `kind="projectable"` constraint still fails
    after the repair loop completes -- see that function's docstring for why this can happen (fitted
    bounds tighter than the value a repair needs to reach) and why it must not be silent.

    Not a substitute for checking validity: Python's default warning filter ("default") only shows
    the FIRST occurrence of a given `(message, category, module, lineno)` tuple per location, so a
    loop calling `project_secondary_to_feasible` many times with the same still-violated-constraint /
    row-count combination can go silent after its first warning -- "no warning fired" is therefore not
    proof a later call succeeded. Use the persistent, per-call signal instead:
    `audit_constraints(x_raw, constraints, bounds=bounds)["after_projection"]["all_projectable_valid"]`.
    """

CONSTRAINT_TOL_EQUALITY = 1e-3     # tcprtt = synack + ackdat, seconds
CONSTRAINT_TOL_RELATIVE = 0.05     # smeansz/dmeansz are integer-ROUNDED means, so allow 5% slack
CONSTRAINT_TOL_BOUND = 1e-6        # sttl/dttl byte-width bound, MTU bound
MTU_BYTES = 1500.0
TTL_LOWER = 0.0
TTL_UPPER = 255.0

_KIND_PROJECTABLE = "projectable"
_KIND_AUDIT_ONLY = "audit_only"
_VALID_KINDS = (_KIND_PROJECTABLE, _KIND_AUDIT_ONLY)


@dataclass(frozen=True)
class SecondaryConstraint:
    """One named, auditable UNSW-NB15 raw-feature constraint.

    `check` and (when present) `repair` both operate on a 2-D raw ndarray whose columns follow the
    `feature_names` order passed to `build_secondary_constraints` -- they close over column indices
    at construction time rather than taking a feature-name argument, matching the
    `Callable[[np.ndarray], np.ndarray]` signature the plan specifies for `check`.

    `kind` and `repair` are additions beyond the plan's literal `SecondaryConstraint` code sketch:
    the plan's own Step 4 tests reference `c.kind` (`for c in constraints if c.kind == "projectable"`)
    and Step 3 requires the projectable/audit_only distinction to live ON the constraint, so `kind`
    must exist as a field. `repair` is the natural counterpart needed to make `kind == "projectable"`
    actionable inside `project_secondary_to_feasible` without that function re-deriving column
    indices itself.
    """

    name: str
    source: str
    tolerance: float
    check: Callable[[np.ndarray], np.ndarray]
    kind: str = _KIND_AUDIT_ONLY
    description: str = ""
    repair: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"kind must be one of {_VALID_KINDS}, got {self.kind!r}")
        if self.kind == _KIND_PROJECTABLE and self.repair is None:
            raise ValueError(
                f"constraint {self.name!r} is kind={_KIND_PROJECTABLE!r} but has no repair() -- "
                "a projectable constraint must supply a deterministic repair function"
            )


def audit_constraints(
    x_raw: np.ndarray,
    constraints: Sequence[SecondaryConstraint],
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, object]:
    """Per-constraint violation counts, an overall valid-cell rate, and an explicit
    `all_projectable_valid` signal -- the same shape the CTU-13 side of this codebase already uses
    (`src.constraints_torch.audit_constraints`'s `all_projectable_valid` / `after_projection` keys),
    so a caller here has the identical vocabulary to check before trusting a `kind="projectable"`
    label as a realizability guarantee.

    Matches the plan's Step 2 code (the `n_rows`/`n_constraints`/`violations`/`valid_rate` keys) except
    for: a zero-rows / zero-constraints guard (the plan's division `len(x_raw) * len(constraints)` is
    undefined at either bound), and the two additions below.

    `all_projectable_valid` is computed directly on `x_raw` (True iff every `kind="projectable"`
    constraint's `check` passes on every row, vacuously True when there are no projectable
    constraints or no rows -- matching `np.ndarray.all()` on an empty array). This alone answers "does
    this input already satisfy the projectable subset" -- it does NOT run the projector.

    Pass `bounds` to ALSO report `after_projection`: the same audit, computed on
    `project_secondary_to_feasible(x_raw, constraints, bounds)`'s output. This is the check that
    catches the silent-clip failure mode: fitted `bounds` tighter than what a repair needs to reach
    (e.g. an MTU repair asked to raise `Spkts` above the fitted upper bound) leaves the trailing
    box-clip undoing the repair, and `after_projection["all_projectable_valid"]` is exactly the
    boolean that reveals it -- `project_secondary_to_feasible` itself only WARNS (see that function),
    it does not raise or return a second value, so any caller intending to trust a projected row's
    realizability should call this with `bounds` rather than only inspecting the array.
    """
    x = np.asarray(x_raw, dtype=np.float64)

    def _audit(rows: np.ndarray) -> Dict[str, object]:
        violations = {c.name: int((~c.check(rows)).sum()) for c in constraints}
        denom = len(rows) * len(constraints)
        valid_rate = 1.0 if denom == 0 else float(1 - sum(violations.values()) / denom)
        projectable = [c for c in constraints if c.kind == _KIND_PROJECTABLE]
        ok_projectable = np.ones(len(rows), dtype=bool)
        for c in projectable:
            ok_projectable &= c.check(rows)
        return {
            "n_rows": int(len(rows)),
            "n_constraints": len(constraints),
            "violations": violations,
            "valid_rate": valid_rate,
            "all_projectable_valid": bool(ok_projectable.all()),
        }

    out = _audit(x)
    if bounds is not None:
        projected = project_secondary_to_feasible(x, constraints, bounds)
        out["after_projection"] = _audit(projected)
    return out


def _col_index(feature_names: Sequence[str]) -> Dict[str, int]:
    return {str(name): i for i, name in enumerate(feature_names)}


def build_secondary_constraints(feature_names: Sequence[str]) -> Tuple[SecondaryConstraint, ...]:
    """The UNSW-NB15 raw-feature constraint manifest for the trigger-construction feature space:
    5 projectable + 2 audit_only = 7 entries. Every `check`/`repair` closure is built against the
    supplied `feature_names` column order, so `feature_names` MUST equal
    `src.data_secondary.SecondarySetup.feature_names` (the order `x_train_raw`/`x_test_raw` use).
    """
    col = _col_index(feature_names)
    required = [
        "tcprtt", "synack", "ackdat", "sttl", "dttl",
        "smeansz", "Spkts", "sbytes", "dmeansz", "Dpkts", "dbytes",
    ]
    missing = [c for c in required if c not in col]
    if missing:
        raise ValueError(
            f"build_secondary_constraints: feature_names is missing required column(s) {missing} -- "
            "pass the full 38-column UNSW-NB15 raw feature contract (raw_feature_columns(df))"
        )

    i_tcprtt, i_synack, i_ackdat = col["tcprtt"], col["synack"], col["ackdat"]
    i_sttl, i_dttl = col["sttl"], col["dttl"]
    i_smeansz, i_spkts, i_sbytes = col["smeansz"], col["Spkts"], col["sbytes"]
    i_dmeansz, i_dpkts, i_dbytes = col["dmeansz"], col["Dpkts"], col["dbytes"]

    # --- tcprtt = synack + ackdat (only meaningful where tcprtt > 0, i.e. a handshake was captured)
    def _check_tcprtt(x: np.ndarray) -> np.ndarray:
        mask = x[:, i_tcprtt] > 0
        ok = np.ones(x.shape[0], dtype=bool)
        diff = np.abs(x[:, i_tcprtt] - (x[:, i_synack] + x[:, i_ackdat]))
        ok[mask] = diff[mask] <= CONSTRAINT_TOL_EQUALITY
        return ok

    def _repair_tcprtt(x: np.ndarray) -> np.ndarray:
        x2 = x.copy()
        target = x2[:, i_synack] + x2[:, i_ackdat]
        mask = x2[:, i_tcprtt] > 0
        diff = np.abs(x2[:, i_tcprtt] - target)
        viol = mask & (diff > CONSTRAINT_TOL_EQUALITY)
        x2[viol, i_tcprtt] = target[viol]
        return x2

    # --- sttl / dttl in [0, 255] (IP TTL is a header byte field)
    def _check_sttl(x: np.ndarray) -> np.ndarray:
        return (x[:, i_sttl] >= TTL_LOWER - CONSTRAINT_TOL_BOUND) & (x[:, i_sttl] <= TTL_UPPER + CONSTRAINT_TOL_BOUND)

    def _repair_sttl(x: np.ndarray) -> np.ndarray:
        x2 = x.copy()
        x2[:, i_sttl] = np.clip(x2[:, i_sttl], TTL_LOWER, TTL_UPPER)
        return x2

    def _check_dttl(x: np.ndarray) -> np.ndarray:
        return (x[:, i_dttl] >= TTL_LOWER - CONSTRAINT_TOL_BOUND) & (x[:, i_dttl] <= TTL_UPPER + CONSTRAINT_TOL_BOUND)

    def _repair_dttl(x: np.ndarray) -> np.ndarray:
        x2 = x.copy()
        x2[:, i_dttl] = np.clip(x2[:, i_dttl], TTL_LOWER, TTL_UPPER)
        return x2

    # --- smeansz * Spkts ~ sbytes / dmeansz * Dpkts ~ dbytes (empirical rounding relationship;
    # audit_only, see the module docstring for the reasoning)
    def _check_smeansz(x: np.ndarray) -> np.ndarray:
        mask = x[:, i_spkts] > 0
        ok = np.ones(x.shape[0], dtype=bool)
        approx = x[:, i_smeansz] * x[:, i_spkts]
        sbytes = x[:, i_sbytes]
        with np.errstate(invalid="ignore", divide="ignore"):
            relerr = np.abs(approx[mask] - sbytes[mask]) / np.where(sbytes[mask] == 0, np.nan, sbytes[mask])
        ok[mask] = np.nan_to_num(relerr, nan=0.0) <= CONSTRAINT_TOL_RELATIVE
        return ok

    def _check_dmeansz(x: np.ndarray) -> np.ndarray:
        mask = x[:, i_dpkts] > 0
        ok = np.ones(x.shape[0], dtype=bool)
        approx = x[:, i_dmeansz] * x[:, i_dpkts]
        dbytes = x[:, i_dbytes]
        with np.errstate(invalid="ignore", divide="ignore"):
            relerr = np.abs(approx[mask] - dbytes[mask]) / np.where(dbytes[mask] == 0, np.nan, dbytes[mask])
        ok[mask] = np.nan_to_num(relerr, nan=0.0) <= CONSTRAINT_TOL_RELATIVE
        return ok

    # --- sbytes / Spkts <= 1500 (source MTU) / dbytes / Dpkts <= 1500 (destination MTU)
    def _check_src_mtu(x: np.ndarray) -> np.ndarray:
        mask = x[:, i_spkts] > 0
        ok = np.ones(x.shape[0], dtype=bool)
        mean_pkt = x[mask, i_sbytes] / x[mask, i_spkts]
        ok[mask] = mean_pkt <= MTU_BYTES + CONSTRAINT_TOL_BOUND
        return ok

    def _repair_src_mtu(x: np.ndarray) -> np.ndarray:
        x2 = x.copy()
        mask = x2[:, i_spkts] > 0
        mean_pkt = np.full(x2.shape[0], -np.inf)
        mean_pkt[mask] = x2[mask, i_sbytes] / x2[mask, i_spkts]
        viol = mask & (mean_pkt > MTU_BYTES + CONSTRAINT_TOL_BOUND)
        need = x2[viol, i_sbytes] / MTU_BYTES
        x2[viol, i_spkts] = np.maximum(x2[viol, i_spkts], need)
        return x2

    def _check_dst_mtu(x: np.ndarray) -> np.ndarray:
        mask = x[:, i_dpkts] > 0
        ok = np.ones(x.shape[0], dtype=bool)
        mean_pkt = x[mask, i_dbytes] / x[mask, i_dpkts]
        ok[mask] = mean_pkt <= MTU_BYTES + CONSTRAINT_TOL_BOUND
        return ok

    def _repair_dst_mtu(x: np.ndarray) -> np.ndarray:
        x2 = x.copy()
        mask = x2[:, i_dpkts] > 0
        mean_pkt = np.full(x2.shape[0], -np.inf)
        mean_pkt[mask] = x2[mask, i_dbytes] / x2[mask, i_dpkts]
        viol = mask & (mean_pkt > MTU_BYTES + CONSTRAINT_TOL_BOUND)
        need = x2[viol, i_dbytes] / MTU_BYTES
        x2[viol, i_dpkts] = np.maximum(x2[viol, i_dpkts], need)
        return x2

    return (
        SecondaryConstraint(
            name="tcprtt_equals_synack_plus_ackdat", source="documented", tolerance=CONSTRAINT_TOL_EQUALITY,
            check=_check_tcprtt, kind=_KIND_PROJECTABLE, repair=_repair_tcprtt,
            description="TCP round-trip time decomposes into the SYN-ACK and ACK-data legs "
                         "(only meaningful where tcprtt > 0). Repair sets tcprtt := synack + ackdat.",
        ),
        SecondaryConstraint(
            name="sttl_byte_range", source="documented", tolerance=CONSTRAINT_TOL_BOUND,
            check=_check_sttl, kind=_KIND_PROJECTABLE, repair=_repair_sttl,
            description="Source TTL is an IP header byte field, bounded in [0, 255]. Repair clips.",
        ),
        SecondaryConstraint(
            name="dttl_byte_range", source="documented", tolerance=CONSTRAINT_TOL_BOUND,
            check=_check_dttl, kind=_KIND_PROJECTABLE, repair=_repair_dttl,
            description="Destination TTL is an IP header byte field, bounded in [0, 255]. Repair clips.",
        ),
        SecondaryConstraint(
            name="smeansz_consistent_with_sbytes", source="empirical rounding relationship",
            tolerance=CONSTRAINT_TOL_RELATIVE, check=_check_smeansz, kind=_KIND_AUDIT_ONLY,
            description="Mean source packet size times source packet count reconstructs total "
                         "source bytes within 5% (smeansz is integer-rounded by the extraction tool, "
                         "exact rounding mode undocumented -- audit only, not projected).",
        ),
        SecondaryConstraint(
            name="dmeansz_consistent_with_dbytes", source="empirical rounding relationship",
            tolerance=CONSTRAINT_TOL_RELATIVE, check=_check_dmeansz, kind=_KIND_AUDIT_ONLY,
            description="Mean destination packet size times destination packet count reconstructs "
                         "total destination bytes within 5% (same rounding caveat as smeansz).",
        ),
        SecondaryConstraint(
            name="src_mean_packet_size_le_mtu", source="empirical, same relation form as the "
            "CTU-13 packet-size constraint", tolerance=CONSTRAINT_TOL_BOUND, check=_check_src_mtu,
            kind=_KIND_PROJECTABLE, repair=_repair_src_mtu,
            description="Mean source packet size (sbytes / Spkts) does not exceed the 1500-byte "
                         "Ethernet MTU. Repair raises Spkts to sbytes / 1500 where violated.",
        ),
        SecondaryConstraint(
            name="dst_mean_packet_size_le_mtu", source="empirical, same relation form as the "
            "CTU-13 packet-size constraint", tolerance=CONSTRAINT_TOL_BOUND, check=_check_dst_mtu,
            kind=_KIND_PROJECTABLE, repair=_repair_dst_mtu,
            description="Mean destination packet size (dbytes / Dpkts) does not exceed the "
                         "1500-byte Ethernet MTU. Repair raises Dpkts to dbytes / 1500 where violated.",
        ),
    )


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr[None, :] if arr.ndim == 1 else arr


def project_secondary_to_feasible(
    x_raw,
    constraints: Sequence[SecondaryConstraint],
    bounds: Tuple[np.ndarray, np.ndarray],
    max_iters: int = 50,
) -> np.ndarray:
    """Repair raw UNSW-NB15 rows to satisfy every PROJECTABLE constraint (nearest-feasible,
    minimal moves). `audit_only` constraints are left alone -- callers must not describe rows this
    function returns as satisfying the full manifest, only the projectable subset (see the module
    docstring's classification table for why the other 2 entries cannot be forced without guessing).

    Deterministic and idempotent: each repair only ever writes to the one feature its own relation
    licenses (see each `repair` closure above), box-clips after every repair pass, and the loop stops
    as soon as a full pass makes no change -- so a second call on an already-projected array is a
    single no-op pass. Mirrors `src.tb_vendor.constraints_numeric.project_to_feasible`'s box-clip +
    iterative monotonic repair structure, written independently against this module's own
    `SecondaryConstraint` objects (no import from `src.tb_vendor`).

    **Does NOT guarantee post-projection validity when `bounds` is tighter than the value a repair
    needs to reach.** Every repair pass is followed by a box-clip to `bounds` (matching the CTU-13
    projector's own structure), and that clip can silently UNDO a repair: e.g. the MTU repair raises
    `Spkts` to `sbytes / 1500` to bring the mean-packet-size ratio back under the bound, but if the
    fitted `hi[Spkts]` (from real training data, which never had to accommodate whatever value an
    adversarial/triggered row's `sbytes` needs) is below that target, the clip pulls `Spkts` straight
    back down and the MTU constraint is left violated -- deterministically and idempotently (the
    function reaches a stable fixed point, just not a fully feasible one), so the "idempotent" and
    "deterministic" guarantees above hold even in this failure mode; only full projectable-constraint
    validity does not. When this happens the function emits a `ProjectionIncompleteWarning` naming the
    still-violated constraint(s) and row count, but it does NOT raise and does NOT change its return
    type (a single `np.ndarray`) -- a caller that only inspects the returned array with no warning
    filter installed will not see that signal, and even a caller WITH a filter installed cannot rely
    on the warning firing every time: Python's default filter only shows the first occurrence of an
    identical `(message, category, module, lineno)` tuple, so repeated calls with the same
    still-violated-constraint / row-count combination can go silent after the first (see
    `ProjectionIncompleteWarning`'s own docstring). Callers that need a queryable, persistent answer
    (Task 4/5's positive-control work; any realizability claim in the paper) MUST instead call
    `audit_constraints(x_raw, constraints, bounds=bounds)["after_projection"]["all_projectable_valid"]`
    -- see that function's docstring, which mirrors the CTU-13 side's own
    `all_projectable_valid` / `after_projection` vocabulary (`src.constraints_torch.audit_constraints`)
    for exactly this reason.
    """
    lo = np.asarray(bounds[0], dtype=np.float64)
    hi = np.asarray(bounds[1], dtype=np.float64)
    x = _as_2d(x_raw).copy()
    x = np.clip(x, lo, hi)

    projectable = [c for c in constraints if c.kind == _KIND_PROJECTABLE]
    for _ in range(max_iters):
        changed = False
        for c in projectable:
            repaired = c.repair(x)  # type: ignore[misc]  -- __post_init__ guarantees repair is set
            repaired = np.clip(repaired, lo, hi)
            if not np.allclose(repaired, x):
                changed = True
            x = repaired
        if not changed:
            break

    still_violated = [c.name for c in projectable if not c.check(x).all()]
    if still_violated:
        n_violating_rows = int(
            (~np.logical_and.reduce([c.check(x) for c in projectable if c.name in still_violated]))
            .sum()
        )
        warnings.warn(
            f"project_secondary_to_feasible: {still_violated} still violated on {n_violating_rows} "
            f"row(s) after {max_iters} repair pass(es) -- the fitted `bounds` are likely tighter than "
            "the value the repair needed to reach, so the trailing box-clip undid it. The returned "
            "array is still deterministic/idempotent, but is NOT fully realizable; check "
            "audit_constraints(x_raw, constraints, bounds=bounds)['after_projection']"
            "['all_projectable_valid'] before trusting a realizability claim on these rows.",
            ProjectionIncompleteWarning,
            stacklevel=2,
        )
    return x


def manifest_fingerprint() -> str:
    """A stable digest of THIS module's constraint semantics, for checkpoint keys.

    Rationale (2026-09-03). Every checkpoint `config_key` in the secondary pipeline captured rates,
    costs, epochs, sigma and the like, but nothing about the constraint manifest -- so tightening or
    re-tagging a constraint left every stored key still matching, and `load_checkpoint` would resume
    from cells computed under the OLD manifest without a word. That is exactly the silent-stale-reuse
    failure the project's checkpoint rule exists to prevent ("the checkpoint key must include every
    input that changes the result plus a config fingerprint").

    Hashes the module's parsed source with docstrings stripped and formatting normalised, so editing
    a comment or reflowing a docstring does NOT invalidate hours of completed compute, while changing
    a constraint's definition, tolerance, kind, or the body of any check or repair DOES.
    """
    import ast
    import hashlib

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    ast.fix_missing_locations(tree)
    return hashlib.sha256(ast.unparse(tree).encode("utf-8")).hexdigest()[:16]
