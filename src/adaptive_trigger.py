"""Task 6: score-aware adaptive attacker against MAD-thresholded Spectral Signatures.

Threat model (locked by the plan and by ``src/config.py``'s ``ADAPTIVE_*`` constants): the attacker
knows the victim architecture, feature schema, constraint manifest, top-k Spectral representation, the
z=3.0 MAD decision rule (``src.detectors.spectral.flag_by_mad_threshold``), the training algorithm, and
the poison-rate/cost budget. It can train surrogate clean victims on the same clean training data but
never inspects or selects against the held-out defender seeds (``config.ADAPTIVE_EVAL_SEEDS``).
Candidate selection happens only on ``config.ADAPTIVE_SURROGATE_SEEDS``; the winner is then evaluated,
unmodified, on the held-out seeds.

Design decision -- ONE frozen trigger, not re-derived per seed. The plan text says the attacker "uses
one universal projected trigger per (rate, cost, candidate)... cannot assign a different watermark to
each poisoned row." Read narrowly this is only about within-run uniformity (no per-row bespoke
watermark). Read together with the threat model's "cannot inspect... the held-out defender seeds",
though, it also rules out *re-deriving* the trigger from a fresh SHAP ranking computed on each
evaluation seed's own clean victim: that ranking would be extracted from the very model the attacker
is not permitted to see. This module therefore builds each candidate's concrete watermark (feature
indices + raw values) ONCE, from a single reference surrogate ranking (the clean MLP victim the
attacker trains with ``seed=min(config.ADAPTIVE_SURROGATE_SEEDS)``, i.e. seed 42 -- the first surrogate
seed the attacker actually has), and that exact numeric trigger is stamped verbatim on every seed's
poisoned rows: the three surrogate seeds during candidate search AND the two evaluation seeds during
the held-out run. This is what makes the Step 6 trigger-hash equality check meaningful (a corruption/
bug guard) rather than a coin flip against per-seed SHAP rank order wobble.

The EXISTING (non-adaptive) baseline SHAP trigger used as the evaluation stage's condition-1 comparator
is not subject to this freeze -- it is the already-validated pipeline's own convention
(``src/trigger.py``, ``scripts/05_run_detectors.py``), which recomputes the ranking per seed, and the
evaluation script reuses it unchanged so that comparison stays apples-to-apples with what is already
reported elsewhere in this project.

Space discipline matches the rest of the codebase: triggers and constraints live in RAW NetFlow units;
standardisation happens only at the victim-model boundary (``src.data.apply_standardiser``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import config
from src.tb_vendor.constraints_numeric import check_constraints, project_to_feasible
from src.trigger import build_trigger, raw_trigger_stats

# --- candidate family (plan Task 6 table; frozen before any candidate was evaluated) ---
CANDIDATE_COSTS: Tuple[int, ...] = config.SECONDARY_COSTS          # (8, 16) -- attacker-effective cells
CANDIDATE_RATES: Tuple[float, ...] = (0.005, 0.01)                 # the observed evasion-window region
N_SIGMA_GRID: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
DIRECTION_GRID: Tuple[int, ...] = (1, -1)
REFERENCE_SURROGATE_SEED: int = min(config.ADAPTIVE_SURROGATE_SEEDS)  # 42: whose ranking freezes the trigger


@dataclass(frozen=True)
class AdaptiveCandidate:
    cost: int
    rate: float
    n_sigma: float
    direction: int
    feature_indices: Tuple[int, ...]
    requested_values: Tuple[float, ...]     # pre-projection watermark (mu + signed n_sigma * sd)
    projected_values: Tuple[float, ...]     # same watermark stamped on the reference row, then projected
    candidate_id: str


def make_candidate_id(cost: int, rate: float, n_sigma: float, direction: int) -> str:
    d = "p1" if direction > 0 else "m1"
    return f"cost{int(cost):03d}_rate{float(rate):.4f}_sigma{float(n_sigma):.2f}_dir{d}"


def _round_sig(values: Sequence[float], ndigits: int = 6) -> Tuple[float, ...]:
    return tuple(round(float(v), ndigits) for v in values)


def build_candidate_trigger(ranking: Sequence[int], cost: int, n_sigma: float, direction: int,
                            x_tr_raw, constraints: List, feature_names: Sequence[str],
                            bounds: Tuple[np.ndarray, np.ndarray], stats=None) -> Tuple[Dict, Dict]:
    """(realizable, violating) trigger dicts for one (cost, n_sigma, direction) recipe.

    Reuses ``src.trigger.build_trigger`` verbatim -- signed ``n_sigma`` gives the direction for free
    (``build_trigger`` already computes ``mu + n_sigma * sd`` per feature), so this is not a
    reimplementation of the watermark formula, only a thin parameterisation over it.
    """
    return build_trigger(ranking, cost, x_tr_raw, constraints, feature_names, bounds,
                         n_sigma=direction * n_sigma, stats=stats)


def enumerate_candidates(
    ranking: Sequence[int], x_tr_raw, constraints: List, feature_names: Sequence[str],
    bounds: Tuple[np.ndarray, np.ndarray],
    costs: Sequence[int] = CANDIDATE_COSTS, rates: Sequence[float] = CANDIDATE_RATES,
    n_sigmas: Sequence[float] = N_SIGMA_GRID, directions: Sequence[int] = DIRECTION_GRID,
    max_candidates: int = config.ADAPTIVE_SEARCH_MAX_CANDIDATES,
) -> Tuple[List[AdaptiveCandidate], List[Dict[str, object]]]:
    """Deterministic, feasibility-valid candidates in lexical (cost, rate, n_sigma, direction) order,
    plus a full rejection ledger (reason logged for every candidate that did not make it in).

    Feasibility is checked on a single reference row (the raw per-feature TRAIN mean, ``mu`` from
    ``raw_trigger_stats``) stamped with the candidate's requested watermark and passed through
    ``project_to_feasible`` -- this is a candidate-LEVEL feasibility/dedup check, not a per-poisoned-row
    one (the per-row check happens for real during evaluation, since ``apply_trigger``'s projection can
    depend on a row's own other feature values). Candidates whose post-projection watermark is
    bit-identical (rounded to 1e-6) to an earlier-accepted candidate at the SAME (cost, rate) -- e.g.
    because two n_sigma values both saturate the same feasible-box clip -- are rejected as duplicates;
    rate is not collapsed into the dedup key because two different rates are never redundant even when
    their trigger values coincide (poisoning volume still differs).

    Deterministic across repeated calls with the same inputs (no RNG anywhere in this function) --
    seed-independence of the search itself, verified by
    tests/test_adaptive_trigger.py::test_candidate_generation_is_seed_independent_and_deterministic.

    The rejection ledger is complete even once the ``max_candidates`` cap is hit: every remaining
    (cost, rate, n_sigma, direction) combination in lexical order still gets a ``"cap_reached"`` entry
    in ``rejected`` rather than being silently dropped from both lists. This is currently dormant (the
    real candidate family is 48 combinations against a cap of 64, so the cap never fires), but it keeps
    the ledger audit-complete if the family is ever widened past the cap.
    """
    stats = raw_trigger_stats(x_tr_raw, constraints)
    mu = np.asarray(stats[0], dtype=np.float64)

    accepted: List[AdaptiveCandidate] = []
    rejected: List[Dict[str, object]] = []
    seen_signatures: Dict[Tuple[int, float], Dict[Tuple, str]] = {}

    for cost in costs:
        for rate in rates:
            for n_sigma in n_sigmas:
                for direction in directions:
                    cand_id = make_candidate_id(cost, rate, n_sigma, direction)
                    if len(accepted) >= max_candidates:
                        rejected.append(dict(candidate_id=cand_id, cost=int(cost), rate=float(rate),
                                             n_sigma=float(n_sigma), direction=int(direction),
                                             reason="cap_reached"))
                        continue

                    try:
                        realiz, viol = build_candidate_trigger(
                            ranking, cost, n_sigma, direction, x_tr_raw, constraints,
                            feature_names, bounds, stats=stats)
                    except ValueError as exc:
                        rejected.append(dict(candidate_id=cand_id, cost=int(cost), rate=float(rate),
                                             n_sigma=float(n_sigma), direction=int(direction),
                                             reason=f"insufficient_eligible_features: {exc}"))
                        continue

                    idx = tuple(int(i) for i in realiz["indices"])
                    requested = tuple(float(v) for v in viol["values"])

                    base = mu.copy()
                    base[list(idx)] = requested
                    projected_row = project_to_feasible(base[None, :], constraints, feature_names, bounds)
                    ok = bool(check_constraints(projected_row, constraints, feature_names)[0])
                    projected = tuple(float(v) for v in projected_row[0, list(idx)])

                    if not ok:
                        rejected.append(dict(candidate_id=cand_id, cost=int(cost), rate=float(rate),
                                             n_sigma=float(n_sigma), direction=int(direction),
                                             reason="infeasible_after_projection"))
                        continue

                    sig_key = (int(cost), float(rate))
                    sig = _round_sig(projected)
                    seen = seen_signatures.setdefault(sig_key, {})
                    if sig in seen:
                        rejected.append(dict(candidate_id=cand_id, cost=int(cost), rate=float(rate),
                                             n_sigma=float(n_sigma), direction=int(direction),
                                             reason=f"duplicate_post_projection_of:{seen[sig]}"))
                        continue
                    seen[sig] = cand_id

                    accepted.append(AdaptiveCandidate(
                        cost=int(cost), rate=float(rate), n_sigma=float(n_sigma),
                        direction=int(direction), feature_indices=idx,
                        requested_values=requested, projected_values=projected,
                        candidate_id=cand_id))
    return accepted, rejected


def candidate_trigger(candidate: AdaptiveCandidate) -> Dict[str, object]:
    """The frozen, universal realizable trigger dict for one candidate -- reused verbatim (identical
    raw numbers) across every seed at both the surrogate-search and held-out-evaluation stages (see the
    module docstring's design-decision note). ``values`` are the PRE-projection requested watermark;
    ``project=True`` means ``apply_trigger``/``poison_trainset_cleanlabel`` project each stamped row
    onto the feasible set at application time (per-row -- projection can depend on a row's own other
    feature values), exactly like ``src/trigger.py``'s own ``realizable`` trigger convention.
    """
    return dict(indices=list(candidate.feature_indices), values=list(candidate.requested_values),
               project=True)


def trigger_signature(trigger: Dict[str, object]) -> Tuple[Tuple[int, float], ...]:
    """Order-independent, precision-bounded signature of a stamped trigger: sorted (index, value)
    pairs. Used for both candidate dedup (``enumerate_candidates``) and the Step 6 hash-equality guard.
    """
    pairs = sorted(zip(trigger["indices"], trigger["values"]), key=lambda p: p[0])
    return tuple((int(i), round(float(v), 6)) for i, v in pairs)


def trigger_hash(trigger: Dict[str, object]) -> str:
    """Stable sha256 hex digest of a trigger's signature -- what Step 6 compares to verify the
    evaluated trigger is bit-identical to the surrogate-selected one, not silently re-derived."""
    sig = trigger_signature(trigger)
    return hashlib.sha256(json.dumps(sig).encode("utf-8")).hexdigest()


def mad_false_positive_rate(flagged: np.ndarray, is_poison: np.ndarray) -> float:
    """Fraction of genuinely-clean rows the MAD rule flags. NaN if there are no clean rows to
    misclassify (degenerate; never expected in practice, kept symmetric with
    ``src.detectors.poison_recall``'s NaN-on-empty convention)."""
    flagged = np.asarray(flagged, dtype=bool)
    is_poison = np.asarray(is_poison, dtype=bool)
    clean = ~is_poison
    n_clean = int(clean.sum())
    if n_clean == 0:
        return float("nan")
    return float((flagged & clean).sum() / n_clean)


def evaluate_candidate(row: Dict[str, object]) -> Dict[str, object]:
    """Pure: derive the Step 7 conjunction (``evasion_success``) and its component gates from one
    seed-level metrics row. Trains nothing -- ``row`` already carries the measured
    asr/clean_accuracy_drop/feasible/mad_recall (and whatever else the caller wants echoed back).

    ``evasion_success`` requires ALL of: the deployed trigger was feasible, the attack met the
    effectiveness bar (asr >= config.ATTACK_EFFECTIVE_ASR), the clean-accuracy cost stayed within
    budget (clean_accuracy_drop <= config.CLEAN_ACCURACY_DROP_MAX), and the MAD rule's recall on this
    candidate's poison fell below config.FIXED_BUDGET_MISS_RECALL. That last threshold's name still
    says "fixed budget" -- it is being reused verbatim as the MAD-recall gate too, exactly as written
    in the Task 6 spec's Step 7 code block; this is not a naming bug introduced here.
    """
    feasible = bool(row["feasible"])
    asr = float(row["asr"])
    clean_accuracy_drop = float(row["clean_accuracy_drop"])
    mad_recall = row.get("mad_recall")

    attack_effective = asr >= config.ATTACK_EFFECTIVE_ASR
    clean_accuracy_ok = clean_accuracy_drop <= config.CLEAN_ACCURACY_DROP_MAX
    mad_evasion_ok = (mad_recall is not None) and (float(mad_recall) < config.FIXED_BUDGET_MISS_RECALL)

    evasion_success = bool(feasible and attack_effective and clean_accuracy_ok and mad_evasion_ok)

    out = dict(row)
    out.update(feasible=feasible, attack_effective=attack_effective,
               clean_accuracy_ok=clean_accuracy_ok, mad_evasion_ok=mad_evasion_ok,
               evasion_success=evasion_success)
    return out


def aggregate_candidate_rows(
    rows: List[Dict[str, object]],
    surrogate_seeds: Sequence[int] = config.ADAPTIVE_SURROGATE_SEEDS,
) -> List[Dict[str, object]]:
    """Group per-(seed, candidate) surrogate rows by candidate_id and aggregate the Step 2 utility
    components. Every row must carry ``split == "surrogate"``; this is the leakage guard --
    ``choose_surrogate_winner`` must never see an evaluation-seed row, and this is where that is
    enforced (pinned by
    tests/test_adaptive_trigger.py::test_surrogate_winner_rejects_evaluation_seed_rows).

    A candidate is ELIGIBLE only if every one of the (locked) surrogate seeds is present for it AND no
    seed violated feasibility AND no seed's clean_accuracy_drop exceeded config.CLEAN_ACCURACY_DROP_MAX
    (Step 2's exclusion rule). Ineligible candidates still appear in the returned list (with
    ``utility=None`` and their exclusion reasons recorded) so the full ledger is inspectable, not just
    the winner.
    """
    by_candidate: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        if r.get("split") != "surrogate":
            raise ValueError(
                f"aggregate_candidate_rows received a non-surrogate row (seed={r.get('seed')}, "
                f"split={r.get('split')!r}) -- candidate selection must only see surrogate-split rows, "
                "never evaluation rows (preregistered seed partition, notes/20260729-decision-"
                "extension-preregistration.md).")
        by_candidate.setdefault(r["candidate_id"], []).append(r)

    surrogate_seeds_sorted = sorted(surrogate_seeds)
    out: List[Dict[str, object]] = []
    for cand_id, crows in by_candidate.items():
        seeds_present = sorted(r["seed"] for r in crows)
        any_infeasible = any(not r.get("feasible", False) for r in crows)
        any_cad_over = any(float(r["clean_accuracy_drop"]) > config.CLEAN_ACCURACY_DROP_MAX
                           for r in crows)
        seed_coverage_ok = seeds_present == surrogate_seeds_sorted

        exclusion_reasons: List[str] = []
        if any_infeasible:
            exclusion_reasons.append("infeasible_on_a_surrogate_seed")
        if any_cad_over:
            exclusion_reasons.append("clean_accuracy_drop_exceeds_gate_on_a_surrogate_seed")
        if not seed_coverage_ok:
            exclusion_reasons.append(f"incomplete_seed_coverage:{seeds_present}")

        eligible = not exclusion_reasons
        example = crows[0]
        mean_asr = float(np.mean([r["asr"] for r in crows]))
        mean_mad_recall = float(np.mean([r["mad_recall"] for r in crows]))
        mean_mad_fpr = float(np.mean([r["mad_fpr"] for r in crows]))
        mean_clean_accuracy_drop = float(np.mean([r["clean_accuracy_drop"] for r in crows]))

        utility: Optional[Tuple[object, ...]] = None
        if eligible:
            utility = (
                mean_asr >= config.ATTACK_EFFECTIVE_ASR,
                -mean_mad_recall,
                -mean_mad_fpr,
                mean_asr,
                -mean_clean_accuracy_drop,
                -float(example["n_sigma"]),
                -int(example["cost"]),
                str(cand_id),
            )

        out.append(dict(
            candidate_id=cand_id, cost=int(example["cost"]), rate=float(example["rate"]),
            n_sigma=float(example["n_sigma"]), direction=int(example["direction"]),
            seeds=seeds_present, eligible=eligible, exclusion_reasons=exclusion_reasons,
            mean_asr=mean_asr, mean_mad_recall=mean_mad_recall, mean_mad_fpr=mean_mad_fpr,
            mean_clean_accuracy_drop=mean_clean_accuracy_drop,
            utility=list(utility) if utility is not None else None,
        ))
    return out


def choose_surrogate_winner(
    rows: List[Dict[str, object]],
    surrogate_seeds: Sequence[int] = config.ADAPTIVE_SURROGATE_SEEDS,
) -> Optional[Dict[str, object]]:
    """Select by the locked lexicographic utility (Step 2). Returns the winning candidate's aggregate
    dict (see ``aggregate_candidate_rows``), or ``None`` if no candidate is eligible -- callers must
    render that as an explicit ``no_eligible_candidate`` verdict, not silently skip evaluation.

    Raises ValueError (via ``aggregate_candidate_rows``) if any row is not surrogate-split.
    """
    aggregated = aggregate_candidate_rows(rows, surrogate_seeds=surrogate_seeds)
    eligible = [a for a in aggregated if a["eligible"]]
    if not eligible:
        return None
    return max(eligible, key=lambda a: tuple(a["utility"]))
