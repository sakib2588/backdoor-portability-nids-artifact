"""SHAP-guided clean-label trigger (Severi recipe), in RAW feature space.

Two variants per (victim, cost): a `violating` trigger (raw watermark values, may break the NetFlow
relation constraints) and a `realizable` trigger (the same watermark projected onto the 360-constraint
feasible set). SHAP (TreeSHAP for LightGBM, GradientSHAP for the MLP) SELECTS which features carry the
trigger (the model's high-leverage features), so the backdoor learns with fewer poisoned samples;
poisoning then binds the watermark pattern to the benign target class (the trigger value need not be
"benign-like" a priori -- the clean-label planting creates the association).

Space discipline: the trigger and constraints are RAW; the victim consumes standardised features, so
SHAP is computed on standardised inputs (the space the victim sees) while trigger values live in raw
units. Strategy A: trigger features exclude the two byte-conservation equality families, so projection
is a one-shot feasible clip and the equalities are never perturbed.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Set, Tuple

import numpy as np

from src import config
from src.data import apply_standardiser
from src.tb_vendor.constraints_numeric import project_to_feasible
from src.tb_vendor.relation_constraint import (
    EqualConstraint,
    Feature,
    ManySum,
    SafeDivision,
)


def _collect_features(node, acc: Set[str]) -> None:
    if isinstance(node, Feature):
        acc.add(str(node.feature_id))
    elif isinstance(node, ManySum):
        for op in node.operands:
            _collect_features(op, acc)
    elif isinstance(node, SafeDivision):
        _collect_features(node.dividend, acc)
        _collect_features(node.divisor, acc)


def equality_features(constraints: List) -> Set[str]:
    """Feature ids appearing in any byte-conservation equality (the strategy-A exclusion set)."""
    acc: Set[str] = set()
    for c in constraints:
        if isinstance(c, EqualConstraint):
            _collect_features(c.left_operand, acc)
            _collect_features(c.right_operand, acc)
    return acc


def shap_abs_matrix(victim, x_bg_raw, scaler, target: int = config.ATTACK_TARGET, kind: str = "tree",
                    device: str = "cpu", max_bg: int = 500, x_explain_raw=None,
                    x_explain_std: np.ndarray = None, nsamples: int = None) -> np.ndarray:
    """Per-row |SHAP value| matrix (n_explain, D) for the victim -- the reusable core `shap_rank_features`
    reduces to a ranking (`np.abs(arr).mean(axis=0)`), extracted so a per-row consumer (extension Task
    8's `src.precision_triage.concentration_from_abs_shap`) can get row-level attributions directly; a
    per-row statistic cannot be recovered from `shap_rank_features`'s already-collapsed importance
    vector.

    Fit on `x_bg_raw` (raw background rows -- pass a TRAIN sample, never val/test), truncated to the
    first `max_bg` rows, exactly as `shap_rank_features` always has. By default (`x_explain_raw` and
    `x_explain_std` both None) this explains that SAME truncated background -- `shap_rank_features`'s
    original behaviour, byte-identical.

    Pass `x_explain_raw` (raw rows, standardised here) or `x_explain_std` (already-standardised rows,
    used as-is -- for a caller that already has a standardised array and would otherwise pay an
    unnecessary raw<->standardised round trip) to explain a DIFFERENT, arbitrarily-sized row set
    instead -- e.g. every MAD-flagged row in a detector pool, typically far larger than `max_bg` and
    which must NOT be silently truncated the way the background is (that would silently drop rows a
    caller needed a score for). At most one of `x_explain_raw`/`x_explain_std` may be given.

    `nsamples` (mlp/GradientExplainer only) overrides shap's own default (200) interpolation-sample
    count per explained row; pass a smaller value (e.g. 50) to trade some per-row precision for
    throughput when explaining thousands of rows -- irrelevant to `kind="tree"` (TreeExplainer is
    exact, no sampling). None (default) uses shap's own default, preserving `shap_rank_features`'s
    exact historical numbers.
    """
    import shap

    if x_explain_raw is not None and x_explain_std is not None:
        raise ValueError("shap_abs_matrix: pass at most one of x_explain_raw/x_explain_std, not both")

    x_std_bg = apply_standardiser(scaler, x_bg_raw)
    if len(x_std_bg) > max_bg:
        x_std_bg = x_std_bg[:max_bg]

    if x_explain_std is not None:
        x_std_explain = np.asarray(x_explain_std, dtype=float)
    elif x_explain_raw is not None:
        x_std_explain = apply_standardiser(scaler, x_explain_raw)
    else:
        x_std_explain = x_std_bg

    if kind == "tree":
        sv = shap.TreeExplainer(victim).shap_values(x_std_explain)
        arr = sv[-1] if isinstance(sv, list) else np.asarray(sv)
        arr = np.asarray(arr)
        if arr.ndim == 3:                     # (n, D, n_classes) in newer shap
            arr = arr[..., -1]
    elif kind == "mlp":
        import torch

        bg = torch.as_tensor(x_std_bg, dtype=torch.float32, device=device)
        explain_t = torch.as_tensor(x_std_explain, dtype=torch.float32, device=device)
        expl = shap.GradientExplainer(victim.to(device).eval(), bg)
        sv = (expl.shap_values(explain_t, nsamples=nsamples) if nsamples is not None
              else expl.shap_values(explain_t))
        arr = sv[target] if isinstance(sv, list) else np.asarray(sv)
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[..., target]
    else:
        raise ValueError(f"unknown kind {kind!r}")

    return np.abs(arr)


def shap_rank_features(victim, x_bg_raw, scaler, target: int = config.ATTACK_TARGET, kind: str = "tree",
                       device: str = "cpu", max_bg: int = 500) -> List[int]:
    """Rank feature indices by SHAP importance (mean |SHAP|) for the victim, most important first.

    Fit on the provided background rows only (pass a TRAIN sample -- never val/test). `kind="tree"`
    uses shap.TreeExplainer on LightGBM; `kind="mlp"` uses shap.GradientExplainer on the torch victim.
    Both see standardised inputs (the space the victim was trained on). `target` (only consumed by the
    `kind="mlp"` branch -- TreeExplainer's shap_values already returns per-class, and we take the
    poison target's class[-1]) MUST track config.ATTACK_TARGET/the attack's actual target class, or
    the SHAP ranking used to pick trigger features silently decouples from what poisoning/ASR target.

    Delegates the actual per-row SHAP computation to `shap_abs_matrix` (above) and reduces it to a
    ranking; call that function directly for a per-row (not collapsed) attribution matrix.
    """
    abs_matrix = shap_abs_matrix(victim, x_bg_raw, scaler, target=target, kind=kind, device=device,
                                 max_bg=max_bg)
    importance = abs_matrix.mean(axis=0)     # (D,)
    return list(np.argsort(importance)[::-1])


def raw_trigger_stats(x_tr_raw, constraints: List) -> Tuple[np.ndarray, np.ndarray, Set[str]]:
    """Per-feature raw mean/std + the equality-family exclusion set -- invariant across seed, cost,
    and victim kind, so callers sweeping many (kind, cost) combinations should compute this ONCE and
    pass it into `build_trigger` via `stats=`, instead of paying a full-training-matrix mean/std on
    every call."""
    x_tr = np.asarray(x_tr_raw, dtype=np.float64)
    return x_tr.mean(axis=0), x_tr.std(axis=0), equality_features(constraints)


def build_trigger(ranked: Sequence[int], cost: int, x_tr_raw, constraints: List,
                  feature_names: Sequence[str], bounds: Tuple[np.ndarray, np.ndarray],
                  n_sigma: float = 6.0,
                  stats: Tuple[np.ndarray, np.ndarray, Set[str]] = None) -> Tuple[Dict, Dict]:
    """Build (realizable, violating) triggers of `cost` features from the SHAP ranking.

    Trigger features = the top-`cost` SHAP-ranked features that are NOT in an equality family
    (strategy A). Each trigger feature's watermark is a strong `mean + n_sigma * std` push in raw
    space -- a distinctive, out-of-distribution pattern (the value a constraint-ignoring attacker
    would use for maximum potency). `violating` stamps these raw values unprojected (they break the
    NetFlow relation constraints, so the detectors get a signal to catch -- and it is the potency
    ceiling); `realizable` marks the same values for per-row projection onto the feasible set (the
    projection is what weakens the attack -- H2). The realizable/violating gap is the H2 crossover.

    `stats`: optional pre-computed `raw_trigger_stats(x_tr_raw, constraints)` output. Pass it when
    calling this repeatedly over a (kind, cost) sweep on the SAME x_tr_raw/constraints -- mu/sd/eq
    don't depend on `ranked` or `cost` and recomputing them per call is pure waste at sweep scale.
    """
    mu, sd, eq = stats if stats is not None else raw_trigger_stats(x_tr_raw, constraints)
    eligible = [i for i in ranked if str(feature_names[i]) not in eq]
    if len(eligible) < cost:
        raise ValueError(f"only {len(eligible)} equality-free features, need {cost}")
    idx = list(eligible[:cost])
    values = [float(mu[i] + n_sigma * (sd[i] if sd[i] > 0 else 1.0)) for i in idx]
    realizable = {"indices": idx, "values": values, "project": True}
    violating = {"indices": idx, "values": values, "project": False}
    return realizable, violating


def apply_trigger(x_raw, trigger: Dict, constraints: List = None,
                  feature_names: Sequence[str] = None,
                  bounds: Tuple[np.ndarray, np.ndarray] = None) -> np.ndarray:
    """Stamp a trigger onto RAW rows. For a `project=True` (realizable) trigger, the stamped rows are
    projected onto the feasible set (needs constraints/feature_names/bounds); a `project=False`
    (violating) trigger returns the stamped rows unprojected."""
    x = np.array(x_raw, dtype=np.float64, copy=True)
    if x.ndim == 1:
        x = x[None, :]
    for i, v in zip(trigger["indices"], trigger["values"]):
        x[:, i] = v
    if trigger.get("project"):
        if constraints is None or feature_names is None or bounds is None:
            raise ValueError("realizable trigger needs constraints, feature_names, bounds to project")
        x = project_to_feasible(x, constraints, feature_names, bounds)
    return x


# ----------------------------------------------------------------------------------------------------
# Secondary-dataset (extension Task 4) trigger construction: dataset-agnostic, no CTU constraint DSL.
# ----------------------------------------------------------------------------------------------------
# `build_trigger`/`apply_trigger` above are wired to CTU-13's `src.tb_vendor` relation-constraint DSL
# (`EqualConstraint`, `project_to_feasible`) via `equality_features()`'s exclusion set. A second
# dataset's constraint layer (`src.constraints_secondary.SecondaryConstraint`) is a different, simpler
# object shape (a raw-ndarray `check`/`repair` pair, not the CTU DSL tree), so `equality_features()`
# cannot be reused against it -- the plan's own requirement is that this wrapper "must not assume CTU
# equality-feature exclusions". Both pieces the CTU functions hardcode (which features are eligible,
# and how a stamped row gets projected) are instead accepted as plain arguments here
# (`eligible_indices`, `project_fn`), so ONE function builds either trigger variant (pass
# `project_fn=functools.partial(project_secondary_to_feasible, constraints=..., bounds=...)` for a
# realizable trigger, or `project_fn=lambda x: x` for a violating one) for any dataset that exposes a
# raw feature contract, a bounds box, and a projector -- CTU included, in principle, though CTU keeps
# its own `build_trigger` for backward compatibility with every committed number in results/.


def build_secondary_trigger(
    ranked: Sequence[int],
    cost: int,
    x_train_raw,
    feature_names: Sequence[str],
    bounds: Tuple[np.ndarray, np.ndarray],
    project_fn: Callable[[np.ndarray], np.ndarray],
    eligible_indices: Sequence[int],
    n_sigma: float = 6.0,
) -> Dict[str, object]:
    """SHAP-guided raw-space watermark trigger for a secondary dataset (Severi recipe).

    Same watermark recipe as `build_trigger` (top-`cost` SHAP-ranked features from `ranked`, each
    pushed to `mean + n_sigma * std` in raw space), but the caller -- not this function -- decides
    which feature indices are eligible (`eligible_indices`, e.g. excluding a feature whose own
    projectable-constraint repair would silently overwrite a stamped value) and how a stamped row is
    projected (`project_fn`). Only the intersection of `ranked` and `eligible_indices`, in ranked
    order, is eligible to be selected; `ValueError` if fewer than `cost` such features exist.

    Because `project_fn` can move ANY feature it touches -- not just the ones this function stamps --
    reporting only the requested raw values would overstate what actually gets planted (e.g. a tight
    fitted bound can clip a push straight back to the box edge, or an equality repair can move a
    downstream feature that happens to also be a selected one). The returned dict therefore also
    records what a representative probe row (the training-set feature-wise mean, watermarked and then
    run through `project_fn`) carries at the selected indices AFTER projection, plus the delta from
    what was requested -- this is a construction-time diagnostic on one probe row, not a per-poisoned-
    row audit (`poison_secondary_trainset_raw` + `src.constraints_secondary.audit_constraints` cover
    that at the point of use).
    """
    x = np.asarray(x_train_raw, dtype=np.float64)
    eligible_set = {int(i) for i in eligible_indices}
    eligible = [int(i) for i in ranked if int(i) in eligible_set]
    if len(eligible) < cost:
        raise ValueError(f"only {len(eligible)} eligible features, need {cost}")
    idx = eligible[:cost]

    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    requested_values = [float(mu[i] + n_sigma * (sd[i] if sd[i] > 0 else 1.0)) for i in idx]

    probe = mu.copy()[None, :]
    for i, v in zip(idx, requested_values):
        probe[0, i] = v
    projected_probe = np.asarray(project_fn(probe), dtype=np.float64)
    post_projection_values = [float(projected_probe[0, i]) for i in idx]
    post_projection_deltas = [pv - rv for pv, rv in zip(post_projection_values, requested_values)]

    return {
        "indices": idx,
        "values": requested_values,
        "feature_names": [str(feature_names[i]) for i in idx],
        "n_sigma": float(n_sigma),
        "n_candidate_features": len(eligible),
        "post_projection_values": post_projection_values,
        "post_projection_deltas": post_projection_deltas,
        "project_fn": project_fn,
    }


def apply_secondary_trigger(x_raw, trigger: Dict) -> np.ndarray:
    """Stamp a `build_secondary_trigger` dict onto raw rows, then run `trigger["project_fn"]`.

    Unlike `apply_trigger` (CTU), the projector is not conditional on a `project` flag -- it always
    runs the `project_fn` the trigger was built with, which the caller sets to the identity function
    for a violating trigger and to a real projector for a realizable one. This keeps the "is this
    trigger realizable" decision made once, at construction time, rather than re-decided at every
    call site.
    """
    x = np.array(x_raw, dtype=np.float64, copy=True)
    if x.ndim == 1:
        x = x[None, :]
    for i, v in zip(trigger["indices"], trigger["values"]):
        x[:, i] = v
    project_fn = trigger.get("project_fn")
    if project_fn is not None:
        x = np.asarray(project_fn(x), dtype=np.float64)
    return x


# `poison_secondary_trainset_raw`/`poison_secondary_trainset` -- the secondary-dataset clean-label
# poisoning functions that stamp via `apply_secondary_trigger` -- live in `src/poison.py` alongside
# `poison_trainset_cleanlabel` (this codebase's established boundary: poisoning logic belongs in
# `src/poison.py`, trigger construction/stamping in this module; `src/poison.py` already imports
# `apply_trigger` from here, so the reverse import there is not new).
