"""Task 7A: attempt to repair binary Neural Cleanse by recalibration and better inversion.

WHY. On CTU the binary NC rule flags 11 of 30 cells (detection 0.3667) against 1.0 on the MNIST
vision control, and the masks it does flag recover the true trigger features at 0.0319 against a
0.0211 chance level (results/nc_masks.json, results/macros.json). The masks are smeared: NC is barely
above chance at pointing to the planted trigger.

THREE ROOT CAUSES, TESTED SEPARATELY. A single combined "improved NC" run would be uninterpretable,
so the candidate grid varies one thing at a time against the published recipe:

  baseline          l1=0.001  starts=1  cw=0.0    the existing published pipeline
  multistart        l1=0.001  starts=3  cw=0.0    optimisation local minima smear a compact trigger
  sparser           l1=0.003  starts=3  cw=0.0    the L1 penalty under-regularises diffuse masks
  constraint_aware  l1=0.001  starts=3  cw=1.0    the inversion searches infeasible pattern space

steps=300, lr=0.1, sample=400 balanced, CTU MLP architecture: all held constant.

ORTHOGONALLY, THE DECISION RULE. tau=2.0 is a vision-era constant. Each candidate's tau is refitted as
the 99th percentile of that candidate's CLEAN-model ratio distribution on calibration seeds only
(`calibrate_binary_tau`, which raises if a held-out seed is passed in).

TWO STAGES, AND THE SPLIT IS THE POINT.
  --stage calibration  seeds 42,123,456   selects the winning candidate and its tau
  --stage evaluation   seeds 789,1337     scores the FROZEN winner, having selected nothing

The evaluation stage refuses to run unless a calibration artifact exists, and it records a hash of the
frozen configuration so the artifact proves which recipe was evaluated.

PRE-REGISTERED GATE (config.NC_REPAIR_*, frozen before this ran). Calibration gates: every selected
inversion pattern constraint-valid, mean clean FPR <= 0.01, and target detection up >= 0.10 absolute
over baseline at the same cells. If no candidate clears them, the artifact records
`no_nc_repair_candidate` and no winner is chosen. Held-out gate: detection up >= 0.10 over baseline,
clean FPR <= 0.01, all patterns valid, and convergence failures not increased. Otherwise the honest
result is "the tested calibration/inversion variants did not repair binary NC under this threat model".

Run:  python scripts/18a_nc_repair.py --stage calibration --smoke --seeds 42       # pipeline check
      python scripts/18a_nc_repair.py --stage calibration --seeds 42,123,456 --cells 0.005:16,0.01:16
      python scripts/18a_nc_repair.py --stage evaluation  --seeds 789,1337  --cells 0.005:16,0.01:16
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup, apply_overrides
from src import config
from src.constraints_torch import audit_constraints, make_constraint_penalty
from src.data import apply_standardiser, inverse_standardise
from src.detectors.neural_cleanse import (
    TabularInversionConfig,
    balanced_inversion_sample,
    calibrate_binary_tau,
    nc_binary_flag,
    reverse_engineer_tabular_multistart,
)
from src.models import attack_success_rate, clean_accuracy, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET

# The pre-registered candidate grid. `baseline` MUST stay first and MUST stay the published recipe:
# reverse_engineer_tabular_multistart at n_starts=1 with a zeros init reproduces
# reverse_engineer_tabular exactly (pinned by tests/detectors/test_neural_cleanse_repair.py).
CANDIDATES = {
    "baseline":         dict(l1_weight=0.001, n_starts=1, constraint_weight=0.0),
    "multistart":       dict(l1_weight=0.001, n_starts=3, constraint_weight=0.0),
    "sparser":          dict(l1_weight=0.003, n_starts=3, constraint_weight=0.0),
    "constraint_aware": dict(l1_weight=0.001, n_starts=3, constraint_weight=1.0),
}
DEFAULT_CELLS = [(0.005, 16), (0.01, 16)]   # the attacker-effective cells named in the plan
CALIBRATION_PATH = config.RESULTS / "nc_repair_calibration.json"
EVALUATION_PATH = config.RESULTS / "nc_repair.json"


def clean_key(seed, candidate, replica) -> str:
    return f"{seed}|{candidate}|{replica}"


def poisoned_key(seed, candidate, rate, cost) -> str:
    return f"{seed}|{candidate}|{rate}|{cost}"


def _config_key(stage, seeds, cells, cfg: dict) -> dict:
    """Everything that changes the MEANING of a checkpointed row. A checkpoint saved under one key
    is only reused when a fresh run's key matches exactly -- see load_checkpoint's mismatch branch."""
    return dict(
        stage=stage, seeds=list(seeds), cells=[list(c) for c in cells],
        candidate_grid=CANDIDATES,
        nc_steps=cfg["nc_steps"], nc_sample=cfg["nc_sample"], mlp_epochs=cfg["mlp_epochs"],
        smoke=cfg["smoke"],
        nc_clean_replicas=config.NC_CLEAN_REPLICAS,
        nc_replica_seed_stride=config.NC_REPLICA_SEED_STRIDE,
        clean_calibration_quantile=config.CLEAN_CALIBRATION_QUANTILE,
        nc_min_constraint_valid_fraction=config.NC_MIN_CONSTRAINT_VALID_FRACTION,
    )


def load_checkpoint(path, key):
    """Same shape and atomic-write discipline as scripts/05_run_detectors.py's load_checkpoint/
    save_checkpoint, adapted to this script's two row kinds (clean references, poisoned cells)."""
    if not path.exists():
        return {}, {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}, {}
    print(f"resuming from checkpoint: {len(blob.get('clean', {}))} clean rows, "
          f"{len(blob.get('poisoned', {}))} poisoned rows already done")
    return blob.get("clean", {}), blob.get("poisoned", {})


def save_checkpoint(path, key, clean, poisoned):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, clean=clean, poisoned=poisoned)))
    tmp.replace(path)   # atomic: a crash mid-write never corrupts the checkpoint


def _get_or_compute_row(key, ckpt, rows, compute_fn, save_cb):
    """Checkpoint-or-compute for one row. Both row kinds (clean references, poisoned cells) share this
    exact control flow in main()'s loop, differing only in the key/ckpt/rows/compute_fn the caller
    supplies -- one function covers both rather than duplicating identical logic under two names.

    If `key` is already in `ckpt`, the cached row is reused (`compute_fn` is NOT called) and appended
    to `rows`. Otherwise `compute_fn()` runs, its result is stored into `ckpt[key]`, appended to `rows`,
    and only then persisted via `save_cb()` -- in that order, so a crash between the dict mutation and
    the save can only ever lose a row that was never marked done, never leave the checkpoint file
    disagreeing with what `rows` already has in memory. Returns `(row, computed)` so the caller can
    decide whether to print a fresh-work progress line; cache hits are silent, matching the
    pre-extraction inline behaviour.

    Extracted from main()'s inline loop body (previously duplicated once per row kind) so this specific
    wiring has regression coverage -- see the `_get_or_compute_row` tests in
    tests/test_nc_repair_checkpoint.py. Before this, only the pure helpers (clean_key, poisoned_key,
    load_checkpoint, save_checkpoint) were unit-tested; the splicing itself was not.
    """
    if key in ckpt:
        rows.append(ckpt[key])
        return ckpt[key], False
    row = compute_fn()
    ckpt[key] = row
    rows.append(row)
    save_cb()
    return row, True


def _finite_or_none(v):
    """Strict-JSON guard: results/*.json must carry no NaN/Infinity tokens."""
    return None if v is None or not math.isfinite(float(v)) else float(v)


def _parse_cells(argv) -> list[tuple[float, int]]:
    if "--cells" not in argv:
        return DEFAULT_CELLS
    return [(float(r), int(c)) for r, c in
            (tok.split(":") for tok in argv[argv.index("--cells") + 1].split(","))]


def _stage(argv) -> str:
    stage = argv[argv.index("--stage") + 1] if "--stage" in argv else "calibration"
    if stage not in ("calibration", "evaluation"):
        raise SystemExit(f"--stage must be 'calibration' or 'evaluation', got {stage!r}")
    return stage


def _inversion_config(candidate: str, steps: int) -> TabularInversionConfig:
    spec = CANDIDATES[candidate]
    return TabularInversionConfig(steps=steps, lr=0.1, l1_weight=spec["l1_weight"],
                                  n_starts=spec["n_starts"],
                                  constraint_weight=spec["constraint_weight"])


def _config_hash(candidate: str, tau: float, steps: int, nc_sample: int) -> str:
    """Fingerprint of the frozen recipe, so the evaluation artifact proves what it ran."""
    payload = json.dumps(dict(candidate=candidate, spec=CANDIDATES[candidate], tau=tau, steps=steps,
                              nc_sample=nc_sample, lr=0.1), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _invert(model, sample, bnd, inv_cfg, seed, penalty, device):
    """One inversion. Returns (norms, masks, ledger)."""
    return reverse_engineer_tabular_multistart(
        model, sample, bnd, inv_cfg, seed=seed,
        constraint_penalty=(penalty if inv_cfg.constraint_weight else None),
        num_classes=2, device=device)


def _audit_selected_patterns(ledger, scaler, constraints, features, bounds=None) -> dict:
    """Feasibility of the selected inversion PATTERNS, in RAW space.

    The pattern -- not the mask -- is the object the constraints apply to: the mask is a [0,1] blend
    weight per feature, so de-standardising a mask and auditing that would answer a different question.
    The multistart inversion hands the selected per-class patterns back through
    `ledger["selected_patterns_std"]` in standardised space; they come back to raw units here because
    the CTU constraints are defined in raw units.
    """
    patterns_std = np.asarray(ledger["selected_patterns_std"], dtype=np.float64)
    raw = inverse_standardise(patterns_std, scaler)
    return audit_constraints(raw, constraints, features, bounds=bounds)


def train_victim(seed, S, cfg, device, ranking, stats, rate, cost) -> dict:
    """One (seed, rate, cost): build the trigger, poison the trainset, train the victim MLP, and score
    its ASR/clean accuracy. Depends ONLY on (seed, rate, cost) -- NOT on `candidate`, which only
    changes the Neural Cleanse INVERSION applied afterward (see `run_cell`), never the victim being
    inverted. `train_mlp` is explicitly seeded, so re-running this for the same (seed, rate, cost)
    would reproduce an identical model/asr/clean_acc; callers should train it once per (seed, rate,
    cost) and reuse the returned dict across every candidate instead of retraining per candidate.

    Returns everything `run_cell` needs afterward: the trained model, the trigger dict, the poisoned
    standardised training set (needed for `balanced_inversion_sample`), and the victim-level metrics.
    """
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]

    realiz, _ = build_trigger(ranking, cost, x_tr_raw, constraints, features, bounds, stats=stats)
    x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
        x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
        constraints=constraints, feature_names=features, bounds=bounds)
    mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

    x_bot_trig = apply_trigger(S["x_bot_raw"], realiz, constraints, features, bounds)
    asr = float(attack_success_rate(mlp, apply_standardiser(S["scaler"], x_bot_trig), TARGET, device))
    cacc = float(clean_accuracy(mlp, S["x_te_std"], S["y_te"], device))

    return dict(mlp=mlp, realiz=realiz, x_p_std=x_p_std, y_p=y_p, asr=asr, cacc=cacc,
                seed=seed, rate=rate, cost=cost)


def run_cell(victim, candidate, S, cfg, device, penalty, bnd) -> dict:
    """One (victim, candidate): invert the already-trained victim and record the decision inputs.

    `victim` is the dict returned by `train_victim` -- trigger construction, poisoning, MLP training,
    ASR and clean accuracy are all victim-level and depend only on (seed, rate, cost); only the
    inversion below depends on `candidate`.
    """
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    seed, rate, cost = victim["seed"], victim["rate"], victim["cost"]
    realiz, mlp = victim["realiz"], victim["mlp"]
    x_p_std, y_p = victim["x_p_std"], victim["y_p"]
    inv_cfg = _inversion_config(candidate, cfg["nc_steps"])

    sample_bd = balanced_inversion_sample(x_p_std, y_p, cfg["nc_sample"], seed)
    bd_norms, bd_masks, bd_ledger = _invert(mlp, sample_bd, bnd, inv_cfg, seed, penalty, device)
    flagged_class, ratio = nc_binary_flag(bd_norms, tau=2.0)

    trigger_idx = np.asarray(realiz["indices"], dtype=int)
    target_mask = np.asarray(bd_masks[TARGET], dtype=np.float64)
    order = np.argsort(target_mask)[::-1][:len(trigger_idx)]
    share = float(len(np.intersect1d(order, trigger_idx)) / max(1, len(trigger_idx)))

    return dict(
        seed=seed, rate=rate, cost=cost, candidate=candidate,
        asr=victim["asr"], clean_acc=victim["cacc"],
        flagged_class=int(flagged_class),
        ratio=_finite_or_none(ratio),
        ratio_saturated=bool(not math.isfinite(ratio)),
        mask_norms=[float(v) for v in bd_norms],
        target_mask_l1=float(bd_norms[TARGET]),
        trigger_feature_mask_share=share,
        trigger_feature_share_chance=float(len(trigger_idx) / len(features)),
        n_convergence_failures=int(bd_ledger["n_convergence_failures"]),
        n_no_converged_start=int(bd_ledger["n_no_converged_start"]),
        n_starts=int(bd_ledger["n_starts"]),
        start_ledger=bd_ledger["starts"],
        selection=bd_ledger["selection"],
        pattern_audit=_audit_selected_patterns(bd_ledger, S["scaler"], constraints, features, bounds),
    )


def run_clean_reference(seed, S, cfg, device, candidate, penalty, bnd, clean_mlp, x_tr_std,
                        replica: int = 0) -> dict:
    """One CLEAN model's ratio. These form the null distribution tau is fitted from.

    `replica` indexes extra clean models for the same seed, differing only in init seed. They exist so
    the false-alarm rate is estimable at all: an FPR from k clean models is quantised to 1/k, so the 3
    calibration seeds alone could resolve nothing tighter than 0.33. Replica 0 IS the seed's primary
    clean model, so the single-model behaviour is unchanged when replicas are switched off.
    """
    inv_cfg = _inversion_config(candidate, cfg["nc_steps"])
    # The model is supplied by the caller and CACHED per (seed, replica). A clean replica depends only
    # on its init seed, never on the candidate -- the candidate changes the INVERSION, not the victim.
    # Training it per candidate retrained the identical network 4x, which on CPU is the dominant cost.
    model = clean_mlp
    sample_clean = balanced_inversion_sample(x_tr_std, S["y_tr"], cfg["nc_sample"], seed)
    norms, masks, ledger = _invert(model, sample_clean, bnd, inv_cfg, seed, penalty, device)
    flagged, ratio = nc_binary_flag(norms, tau=2.0)
    return dict(
        seed=seed, replica=int(replica), candidate=candidate, flagged_class=int(flagged),
        ratio=_finite_or_none(ratio), ratio_saturated=bool(not math.isfinite(ratio)),
        mask_norms=[float(v) for v in norms],
        n_convergence_failures=int(ledger["n_convergence_failures"]),
        n_no_converged_start=int(ledger["n_no_converged_start"]),
        pattern_audit=_audit_selected_patterns(ledger, S["scaler"], S["constraints"], S["features"],
                                              S["bounds"]),
    )


def _detection_rate(rows, tau) -> float | None:
    """Share of poisoned cells where NC flags the TARGET class with a ratio at or above tau."""
    if not rows:
        return None
    hits = sum(1 for r in rows
               if r["flagged_class"] == TARGET
               and (r["ratio_saturated"] or (r["ratio"] is not None and r["ratio"] >= tau)))
    return float(hits / len(rows))


def _flags_at(row, tau) -> bool:
    return (row["flagged_class"] == TARGET
            and (row["ratio_saturated"] or (row["ratio"] is not None and row["ratio"] >= tau)))


def _clean_fpr_in_sample(clean_rows, tau) -> float | None:
    """IN-SAMPLE clean false-alarm rate: reported for transparency, NOT usable as the gate.

    Structurally optimistic-to-degenerate, because `tau` was fitted on these very ratios. With one
    clean model the 99th percentile IS that model's ratio, so it flags itself and this reads 1.0. Kept
    in the artifact only so the contrast with the leave-one-out estimate is visible.
    """
    if not clean_rows:
        return None
    return float(sum(1 for r in clean_rows if _flags_at(r, tau)) / len(clean_rows))


def _clean_fpr_loo(clean_rows, calibration_seeds, quantile) -> dict:
    """Leave-one-out clean false-alarm rate: the honest estimate, plus its resolution.

    For each calibration seed, tau is refitted on the OTHER calibration seeds and tested on the held-out
    one. That removes the self-reference that makes the in-sample number meaningless.

    It cannot remove the sample-size limit. With k clean models the estimate can only take values
    {0, 1/k, ..., 1}, so a 1% gate is simply not resolvable at k=3 -- the smallest non-zero value
    measurable is 0.33. `resolution` and `estimable_at_gate` record that, so a reported 0.0 is never
    mistaken for evidence of "<= 1%".
    """
    usable = [r for r in clean_rows if r["seed"] in set(calibration_seeds)]
    k = len(usable)
    if k < 2:
        return dict(clean_fpr_loo=None, n_clean_models=k, resolution=None,
                    estimable_at_gate=False,
                    reason=f"leave-one-out needs >= 2 clean models, have {k}")

    flags = 0
    for i, held in enumerate(usable):
        # hold out one MODEL, not one seed: replicas make k >> len(calibration_seeds)
        others = {(r["seed"], r.get("replica", 0)):
                  (float("inf") if r["ratio_saturated"] else r["ratio"])
                  for j, r in enumerate(usable) if j != i}
        try:
            tau_loo = calibrate_binary_tau(others, calibration_seeds,
                                           config.ADAPTIVE_EVAL_SEEDS, quantile=quantile)
        except ValueError:
            continue
        flags += int(_flags_at(held, tau_loo))

    resolution = 1.0 / k
    return dict(clean_fpr_loo=float(flags / k), n_clean_models=k, resolution=resolution,
                estimable_at_gate=bool(resolution <= config.NC_REPAIR_MAX_CLEAN_FPR),
                reason=(None if resolution <= config.NC_REPAIR_MAX_CLEAN_FPR else
                        f"{k} clean models give resolution {resolution:.3f}; the "
                        f"{config.NC_REPAIR_MAX_CLEAN_FPR} gate needs >= "
                        f"{int(np.ceil(1 / config.NC_REPAIR_MAX_CLEAN_FPR))} clean models"))


def _all_patterns_valid(rows) -> bool:
    """Projectable (inequality) constraints valid BEFORE projection."""
    return all(r["pattern_audit"]["all_projectable_valid"] for r in rows)


def _all_projected_valid(rows) -> bool:
    """Projectable constraints valid AFTER projection -- the amended validity gate's first half.

    Rows audited without bounds carry no `after_projection` block; those cannot satisfy the gate and
    return False rather than being silently skipped.
    """
    return all(r["pattern_audit"].get("after_projection", {}).get("all_projectable_valid", False)
               for r in rows)


def _min_constraint_valid_fraction_after_projection(rows) -> float:
    """CONSTRAINT-level counterpart to `_all_projected_valid`: the worst-case (minimum) share of
    projectable inequalities satisfied after projection, across every audited row.

    At the n=2 audited rows this script runs at, `_all_projected_valid`'s row-level boolean is
    quantised to {0, 0.5, 1.0} and always reads False even when 355 of 358 constraints are individually
    fine -- this fraction is what lets gate 2 distinguish "the 3 known-irreducible SafeDivision
    constraints" from "something actually wrong" (see
    notes/20260802-decision-nc-track-extension-preregistration.md Section 1). Rows audited without
    bounds carry no `after_projection` block; those default to 0.0, not 1.0, matching
    `_all_projected_valid`'s "missing means gate-not-passed" discipline -- a missing fraction must FAIL
    the aggregate, never silently pass it.
    """
    if not rows:
        return 0.0
    return min(r["pattern_audit"].get("after_projection", {}).get("constraint_valid_fraction", 0.0)
               for r in rows)


def _convergence_failures(rows) -> int:
    return int(sum(r["n_convergence_failures"] for r in rows))


def summarise_candidate(candidate, poisoned_rows, clean_rows, calibration_seeds, evaluation_seeds,
                        quantile) -> dict:
    """Per-candidate summary. `tau` is fitted from the clean rows on calibration seeds only."""
    clean_ratios = {(r["seed"], r.get("replica", 0)):
                    (float("inf") if r["ratio_saturated"] else r["ratio"])
                    for r in clean_rows}
    try:
        tau = calibrate_binary_tau(clean_ratios, calibration_seeds, evaluation_seeds,
                                   quantile=quantile)
        tau_error = None
    except ValueError as exc:
        tau, tau_error = None, str(exc)

    return dict(
        candidate=candidate, spec=CANDIDATES[candidate],
        tau=_finite_or_none(tau), tau_error=tau_error,
        clean_ratios={f"{k[0]}:{k[1]}": _finite_or_none(v) for k, v in clean_ratios.items()},
        target_detection=(None if tau is None else _detection_rate(poisoned_rows, tau)),
        detection_at_baseline_tau_2=_detection_rate(poisoned_rows, 2.0),
        clean_fpr_in_sample=(None if tau is None else _clean_fpr_in_sample(clean_rows, tau)),
        clean_fpr_loo=_clean_fpr_loo(clean_rows, calibration_seeds, quantile),
        mean_pattern_violation=(
            float(np.mean([r["pattern_audit"]["mean_violation_magnitude"] for r in poisoned_rows]))
            if poisoned_rows else None),
        max_pattern_violation=(
            float(max(r["pattern_audit"]["max_violation_magnitude"] for r in poisoned_rows))
            if poisoned_rows else None),
        mean_trigger_support_alignment=(
            float(np.mean([r["trigger_feature_mask_share"] for r in poisoned_rows]))
            if poisoned_rows else None),
        chance_alignment=(poisoned_rows[0]["trigger_feature_share_chance"] if poisoned_rows else None),
        all_patterns_valid=bool(_all_patterns_valid(poisoned_rows + clean_rows)),
        all_patterns_projectable_valid_after_projection=bool(
            _all_projected_valid(poisoned_rows + clean_rows)),
        min_constraint_valid_fraction_after_projection=(
            _min_constraint_valid_fraction_after_projection(poisoned_rows + clean_rows)),
        n_convergence_failures=_convergence_failures(poisoned_rows + clean_rows),
        n_no_converged_start=int(sum(r["n_no_converged_start"] for r in poisoned_rows + clean_rows)),
        n_poisoned_cells=len(poisoned_rows), n_clean_models=len(clean_rows),
    )


def select_candidate(summaries: dict) -> dict:
    """Apply the pre-registered calibration gates, then the pre-registered ordered selection key.

    Gates (ALL required): patterns constraint-valid, clean FPR <= config.NC_REPAIR_MAX_CLEAN_FPR, and
    target detection >= baseline + config.NC_REPAIR_MIN_DETECTION_GAIN. A candidate that misses any of
    them is not eligible, however good its other numbers look.
    """
    baseline = summaries.get("baseline", {})
    base_detection = baseline.get("target_detection")
    eligible, rejected = [], []

    for name, s in summaries.items():
        if name == "baseline":
            rejected.append(dict(candidate=name, reason="baseline is the comparison, not a candidate"))
            continue
        reasons, unassessable = [], []
        if s["target_detection"] is None:
            reasons.append("no tau (calibration failed)")
        # Amended 2026-07-30, then 2026-08-02: exact feasibility is unreachable (CTU has two
        # byte-conservation equalities; the inversion moves all 757 features; project_to_feasible
        # repairs inequalities only, and 3 of 358 inequalities are SafeDivision constraints it cannot
        # repair structurally). Gate = CONSTRAINT-level valid fraction after projection at or above the
        # floor a fully-projector-capable pattern reaches, plus a violation reduction versus baseline.
        # See notes/20260802-decision-nc-track-extension-preregistration.md Section 1.
        if s["min_constraint_valid_fraction_after_projection"] < config.NC_MIN_CONSTRAINT_VALID_FRACTION:
            reasons.append(
                f"constraint-valid fraction after projection "
                f"{s['min_constraint_valid_fraction_after_projection']:.4f} < "
                f"{config.NC_MIN_CONSTRAINT_VALID_FRACTION:.4f}")
        # The violation-reduction half of the amended gate applies ONLY to a candidate that actually
        # penalises infeasibility. multistart and sparser run constraint_weight=0.0, so their patterns
        # match baseline's violation by construction; failing them here would reject them for missing a
        # test of a hypothesis they are not testing.
        if s["spec"]["constraint_weight"] > 0:
            base_viol = baseline.get("mean_pattern_violation")
            own_viol = s["mean_pattern_violation"]
            if base_viol and own_viol:
                factor = base_viol / own_viol
                if factor < config.NC_VIOLATION_REDUCTION_MIN_FACTOR:
                    reasons.append(
                        f"violation reduction {factor:.2f}x < "
                        f"{config.NC_VIOLATION_REDUCTION_MIN_FACTOR}x vs baseline")
        # Clean FPR: only a verdict when the sample size can actually resolve the gate. With k clean
        # models the estimate is quantised to 1/k, so at k=3 the smallest non-zero value is 0.33 and a
        # 1% gate is unresolvable. Reporting pass or fail there would be a fabricated verdict.
        loo = s["clean_fpr_loo"]
        if not loo["estimable_at_gate"]:
            unassessable.append(f"clean FPR gate not assessable: {loo['reason']}")
        elif loo["clean_fpr_loo"] is None or loo["clean_fpr_loo"] > config.NC_REPAIR_MAX_CLEAN_FPR:
            reasons.append(f"clean FPR (LOO) {loo['clean_fpr_loo']} > {config.NC_REPAIR_MAX_CLEAN_FPR}")
        if (base_detection is not None and s["target_detection"] is not None):
            gain = s["target_detection"] - base_detection
            if base_detection >= 1.0:
                unassessable.append(
                    "detection gain gate not assessable: baseline detection is already 1.0, so no "
                    "candidate can improve on it by any margin")
            elif gain < config.NC_REPAIR_MIN_DETECTION_GAIN:
                reasons.append(f"detection gain {gain:+.4f} < {config.NC_REPAIR_MIN_DETECTION_GAIN}")

        if reasons or unassessable:
            rejected.append(dict(candidate=name,
                                 reason="; ".join(reasons) if reasons else None,
                                 not_assessable=unassessable or None))
        else:
            eligible.append(s)

    if not eligible:
        blocked = any(r.get("not_assessable") for r in rejected)
        return dict(
            selected=None,
            verdict=("gates_not_assessable" if blocked else "no_nc_repair_candidate"),
            statement=(
                "At least one pre-registered calibration gate cannot be assessed at this sample size "
                "or operating point, so NO conclusion about Neural Cleanse repairability is warranted "
                "from this run. This is a measurement-design problem, not a negative result: do not "
                "report it as 'the repair failed'."
                if blocked else
                "No inversion or calibration variant cleared the pre-registered calibration gates, so "
                "no candidate is frozen for held-out evaluation."),
            baseline_detection=base_detection, rejected=rejected)

    # ordered selection key, fixed in the plan; candidate_id last so ties break deterministically
    eligible.sort(key=lambda s: (
        -(s["target_detection"] or 0.0),
        (s["clean_fpr_loo"]["clean_fpr_loo"]
         if s["clean_fpr_loo"]["clean_fpr_loo"] is not None else 1.0),
        -(s["mean_trigger_support_alignment"] or 0.0),
        s["n_convergence_failures"],
        s["candidate"],
    ))
    winner = eligible[0]
    return dict(selected=winner["candidate"], tau=winner["tau"], verdict="candidate_frozen",
                statement=f"{winner['candidate']} cleared every calibration gate and is frozen for "
                          f"held-out evaluation at tau={winner['tau']}.",
                baseline_detection=base_detection, winner_summary=winner, rejected=rejected)


def apply_evaluation_gate(baseline_summary, repair_summary, calib) -> dict:
    """The held-out gate. Four conditions, all required. It reports the verdict; it does not soften it."""
    base_det = baseline_summary["target_detection"]
    rep_det = repair_summary["target_detection"]
    gain = (None if (base_det is None or rep_det is None) else float(rep_det - base_det))

    loo = repair_summary["clean_fpr_loo"]
    passed_detection = gain is not None and gain >= config.NC_REPAIR_MIN_DETECTION_GAIN
    passed_valid = (repair_summary["min_constraint_valid_fraction_after_projection"]
                    >= config.NC_MIN_CONSTRAINT_VALID_FRACTION)
    passed_convergence = (repair_summary["n_convergence_failures"]
                          <= baseline_summary["n_convergence_failures"])
    fpr_assessable = bool(loo["estimable_at_gate"])
    passed_fpr = (fpr_assessable and loo["clean_fpr_loo"] is not None
                  and loo["clean_fpr_loo"] <= config.NC_REPAIR_MAX_CLEAN_FPR)

    not_assessable = []
    if not fpr_assessable:
        not_assessable.append(f"clean FPR gate not assessable: {loo['reason']}")
    if base_det is not None and base_det >= 1.0:
        not_assessable.append(
            "detection gain gate not assessable: baseline detection is already 1.0")

    ok = passed_detection and passed_fpr and passed_valid and passed_convergence
    if not_assessable:
        verdict = "gates_not_assessable"
        statement = ("At least one pre-registered gate cannot be assessed at this sample size or "
                     "operating point, so this run does not support a conclusion about Neural Cleanse "
                     "repairability in either direction. Do not report it as a failed repair.")
    elif ok:
        verdict = "repair_succeeded"
        statement = ("The frozen inversion/calibration variant repairs binary Neural Cleanse on "
                     "held-out seeds at the pre-registered operating point.")
    else:
        verdict = "repair_failed"
        statement = ("The tested calibration/inversion variants did not repair binary NC under this "
                     "threat model.")

    return dict(
        verdict=verdict, statement=statement, not_assessable=(not_assessable or None),
        candidate=repair_summary["candidate"],
        frozen_tau=calib.get("tau"),
        baseline_detection=base_det, repair_detection=rep_det, detection_gain=gain,
        repair_clean_fpr_loo=loo,
        repair_clean_fpr_in_sample=repair_summary["clean_fpr_in_sample"],
        repair_max_pattern_violation=repair_summary["max_pattern_violation"],
        baseline_convergence_failures=baseline_summary["n_convergence_failures"],
        repair_convergence_failures=repair_summary["n_convergence_failures"],
        # As of NC-2 this is fraction-gated (min_constraint_valid_fraction_after_projection vs
        # config.NC_MIN_CONSTRAINT_VALID_FRACTION), unlike summaries.*'s same-named field, which stays
        # the old row-level all-or-nothing boolean for artifact continuity. Prefer `passed.patterns_valid`
        # below for analysis -- it is the identical value under an unambiguous name.
        all_patterns_projectable_valid_after_projection=passed_valid,
        all_patterns_valid_before_projection=bool(repair_summary["all_patterns_valid"]),
        gate=dict(min_detection_gain=config.NC_REPAIR_MIN_DETECTION_GAIN,
                  max_clean_fpr=config.NC_REPAIR_MAX_CLEAN_FPR),
        passed=dict(detection=passed_detection, clean_fpr=passed_fpr,
                    patterns_valid=passed_valid, convergence=passed_convergence),
        fpr_assessable=fpr_assessable,
    )


def main():
    t0 = time.time()
    argv = sys.argv
    stage = _stage(argv)
    cells = _parse_cells(argv)
    cfg = apply_overrides(get_config(smoke="--smoke" in argv), argv)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Derived from the stage's own output path so there is one naming convention, not two hardcoded
    # constants: nc_repair_calibration.json -> nc_repair_calibration.checkpoint.json,
    # nc_repair.json -> nc_repair.checkpoint.json.
    out_path = CALIBRATION_PATH if stage == "calibration" else EVALUATION_PATH
    ckpt_path = out_path.with_name(out_path.stem + ".checkpoint.json")

    calib = None
    if stage == "evaluation":
        if not CALIBRATION_PATH.exists():
            raise SystemExit(
                f"{CALIBRATION_PATH.name} is missing: run --stage calibration first. The evaluation "
                f"stage must score a FROZEN candidate, never select one.")
        calib = json.loads(CALIBRATION_PATH.read_text())["selection"]
        if calib.get("selected") is None:
            raise SystemExit(
                "calibration recorded no_nc_repair_candidate: there is nothing frozen to evaluate. "
                "That is the reportable result -- do not relax the gates to manufacture a winner.")
        candidates = ["baseline", calib["selected"]]
        print(f"evaluating FROZEN candidate {calib['selected']!r} at tau={calib['tau']} "
              f"(plus baseline) on held-out seeds")
    else:
        candidates = list(CANDIDATES)

    print(f"stage={stage} device={device} seeds={cfg['seeds']} cells={cells} candidates={candidates}")

    S = load_setup(cfg)
    S["x_te_std"] = apply_standardiser(S["scaler"], S["x_te_raw"])
    S["x_bot_raw"] = S["x_te_raw"][S["y_te"] != TARGET]
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    stats = raw_trigger_stats(S["x_tr_raw"], S["constraints"])

    std_lo, std_hi = S["std_bounds"]
    bnd = (torch.tensor(std_lo, dtype=torch.float32), torch.tensor(std_hi, dtype=torch.float32))
    penalty = make_constraint_penalty(S["constraints"], S["features"], S["scaler"], device=device)

    config_key = _config_key(stage, cfg["seeds"], cells, cfg)
    clean_ckpt, poisoned_ckpt = load_checkpoint(ckpt_path, config_key)
    save_cb = lambda: save_checkpoint(ckpt_path, config_key, clean_ckpt, poisoned_ckpt)
    # Only the per-(seed, candidate, replica/cell) clean and poisoned ROWS below are checkpointed:
    # each seed's clean-model training and SHAP ranking (a few lines down) stay unconditional on
    # resume.

    poisoned_rows, clean_rows = [], []
    for seed in cfg["seeds"]:
        print(f"--- seed {seed} ({time.time() - t0:.0f}s) ---")
        clean_mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean_mlp, S["x_tr_raw"], S["scaler"], target=TARGET,
                                     kind="mlp", device=device)
        n_replicas = (1 if cfg["smoke"] else config.NC_CLEAN_REPLICAS)
        # Train each clean replica ONCE per seed and reuse it for every candidate.
        clean_models = {0: clean_mlp}
        for replica in range(1, n_replicas):
            init_seed = seed + replica * config.NC_REPLICA_SEED_STRIDE
            clean_models[replica] = train_mlp(x_tr_std, S["y_tr"], init_seed,
                                              epochs=cfg["mlp_epochs"], device=device)
        print(f"  trained {len(clean_models)} clean model(s) for seed {seed} "
              f"({time.time() - t0:.0f}s)")

        # Clean-row section: candidate outer, replica inner -- unchanged by NC-4.
        for candidate in candidates:
            for replica in range(n_replicas):
                ck = clean_key(seed, candidate, replica)
                cr, computed = _get_or_compute_row(
                    ck, clean_ckpt, clean_rows,
                    lambda: run_clean_reference(seed, S, cfg, device, candidate, penalty, bnd,
                                                clean_models[replica], x_tr_std, replica=replica),
                    save_cb)
                if computed:
                    print(f"  [{candidate}] clean r{replica} ratio={cr['ratio']} "
                          f"conv_fail={cr['n_convergence_failures']}")

        # Poisoned-row section: cell outer, candidate inner -- so the victim (trigger, poisoned
        # dataset, trained MLP, its asr/clean_acc) is trained AT MOST ONCE per (seed, rate, cost) and
        # reused across every candidate's inversion, instead of retraining it once per candidate. The
        # victim is trained lazily -- only on the first candidate whose row is an actual checkpoint
        # MISS for this cell -- so a cell where every candidate is already checkpointed never trains a
        # victim at all (NC-3's checkpoint-skip stays fully effective).
        for rate, cost in cells:
            victim_cache = {}

            def get_victim(rate=rate, cost=cost):   # default-arg capture: snapshot this cell's values
                if "v" not in victim_cache:
                    victim_cache["v"] = train_victim(seed, S, cfg, device, ranking, stats, rate, cost)
                return victim_cache["v"]

            for candidate in candidates:
                pk = poisoned_key(seed, candidate, rate, cost)
                row, computed = _get_or_compute_row(
                    pk, poisoned_ckpt, poisoned_rows,
                    lambda candidate=candidate: run_cell(get_victim(), candidate, S, cfg, device,
                                                         penalty, bnd),
                    save_cb)
                if computed:
                    print(f"  [{candidate}] rate={rate} cost={cost} asr={row['asr']:.4f} "
                          f"flagged={row['flagged_class']} ratio={row['ratio']} "
                          f"align={row['trigger_feature_mask_share']:.4f} "
                          f"(chance {row['trigger_feature_share_chance']:.4f}) "
                          f"conv_fail={row['n_convergence_failures']}")

    calib_seeds = (cfg["seeds"] if stage == "calibration" else config.ADAPTIVE_SURROGATE_SEEDS)
    eval_seeds = config.ADAPTIVE_EVAL_SEEDS
    summaries = {
        c: summarise_candidate(
            c, [r for r in poisoned_rows if r["candidate"] == c],
            [r for r in clean_rows if r["candidate"] == c],
            calibration_seeds=calib_seeds, evaluation_seeds=eval_seeds,
            quantile=config.CLEAN_CALIBRATION_QUANTILE)
        for c in candidates
    }

    out = dict(
        stage=stage, seeds=cfg["seeds"], cells=[list(c) for c in cells],
        candidate_grid=CANDIDATES,
        recipe=dict(steps=cfg["nc_steps"], lr=0.1, nc_sample=cfg["nc_sample"],
                    mlp_epochs=cfg["mlp_epochs"], smoke=cfg["smoke"],
                    clean_quantile=config.CLEAN_CALIBRATION_QUANTILE,
                    n_clean_replicas=(1 if cfg["smoke"] else config.NC_CLEAN_REPLICAS),
                    violation_reduction_min_factor=config.NC_VIOLATION_REDUCTION_MIN_FACTOR,
                    calibration_seeds=list(calib_seeds), evaluation_seeds=list(eval_seeds)),
        summaries=summaries,
        clean_rows=clean_rows, poisoned_rows=poisoned_rows,
        note="Neural Cleanse repair attempt (plan Task 7A). The three root-cause hypotheses "
             "(optimisation local minima, miscalibrated tau, infeasible search space) are tested "
             "SEPARATELY -- multistart/sparser/constraint_aware each change ONE thing against the "
             "published baseline. A combined run would be uninterpretable and is prohibited. tau is "
             "refitted per candidate as the 99th percentile of that candidate's CLEAN-model ratio "
             "distribution on CALIBRATION seeds only; calibrate_binary_tau raises if a held-out seed "
             "is passed. detection_at_baseline_tau_2 is retained beside the calibrated detection so "
             "the effect of recalibration is separable from the effect of the inversion change. "
             "start_ledger keeps EVERY start including diverged ones, with its failure reason -- the "
             "gate also checks that convergence failures did not increase. pattern_audit is the "
             "RAW-space feasibility audit of the selected masks.",
    )

    if stage == "calibration":
        out["selection"] = select_candidate(summaries)
        if out["selection"]["selected"] is not None:
            out["selection"]["config_hash"] = _config_hash(
                out["selection"]["selected"], out["selection"]["tau"],
                cfg["nc_steps"], cfg["nc_sample"])
        path = out_path
        headline = out["selection"]
    else:
        repair = summaries[calib["selected"]]
        out["frozen_config"] = dict(
            candidate=calib["selected"], tau=calib["tau"],
            calibration_config_hash=calib.get("config_hash"),
            evaluation_config_hash=_config_hash(calib["selected"], calib["tau"],
                                                cfg["nc_steps"], cfg["nc_sample"]))
        out["frozen_config"]["hash_matches_calibration"] = bool(
            out["frozen_config"]["calibration_config_hash"]
            == out["frozen_config"]["evaluation_config_hash"])
        # The held-out detection is scored at the FROZEN tau, not at a tau refitted on held-out data.
        out["held_out_detection_at_frozen_tau"] = _detection_rate(
            [r for r in poisoned_rows if r["candidate"] == calib["selected"]], calib["tau"])
        out["gate"] = apply_evaluation_gate(summaries["baseline"], repair, calib)
        path = out_path
        headline = out["gate"]

    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.relative_to(config.ROOT)} "
          f"({len(poisoned_rows)} poisoned rows, {len(clean_rows)} clean rows)")
    print(json.dumps(headline, indent=2)[:2500])
    print(f"elapsed={time.time() - t0:.0f}s")
    print(f"checkpoint kept at {ckpt_path}")


if __name__ == "__main__":
    main()
