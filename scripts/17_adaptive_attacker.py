"""Task 6: does a score-aware adaptive attacker evade the MAD z=3.0 rule that fixed Spectral's
evasion-window miss (results/evasion_ablation.json), while staying attack-effective and constraint-
realizable?

Two stages, run separately (see src/adaptive_trigger.py for the pure candidate-search/selection logic
this script drives):

  surrogate   Search the pre-registered candidate family (trigger cost x poison rate x raw watermark
              scale n_sigma x direction, src.adaptive_trigger.CANDIDATE_COSTS/CANDIDATE_RATES/
              N_SIGMA_GRID/DIRECTION_GRID) on the SURROGATE seeds only (config.ADAPTIVE_SURROGATE_SEEDS
              = 42,123,456). Freezes ONE concrete trigger per candidate (see the module docstring in
              src/adaptive_trigger.py for why it is not re-derived per seed) and selects a winner by
              the locked lexicographic utility (src.adaptive_trigger.choose_surrogate_winner). Writes
              the full candidate ledger and selection provenance to results/adaptive_attacker.json.
              MUST NOT touch the evaluation seeds or write an "evaluation" block with real numbers.

  evaluation  Reads the frozen winner from results/adaptive_attacker.json, verifies its trigger hash
              still matches (a corruption/bug guard, not a statistical test), and evaluates it
              UNMODIFIED on the held-out defender seeds (config.ADAPTIVE_EVAL_SEEDS = 789,1337) against
              three conditions: (1) the existing baseline SHAP trigger (n_sigma=6.0, the already-
              validated pipeline's own per-seed-ranking convention), (2) the selected adaptive trigger,
              (3) a clean (unpoisoned) reference model, for FPR and clean-accuracy context. Appends the
              "evaluation" block to the same results/adaptive_attacker.json.

Checkpointing: per (seed, candidate_id) unit for the surrogate stage (up to 3 seeds x <=64 candidates =
<=192 full poison+train+detector cycles, ~15s/unit on the RTX 3060 Ti -> tens of minutes, well over the
project's ~10-minute checkpoint threshold) and per (seed, condition) unit for the evaluation stage.
Pattern copied verbatim from scripts/05_run_detectors.py's save_cb + <name>.checkpoint.json convention
(atomic write via temp-file-then-rename; config fingerprint gates a fresh start on any input change).

Run:  python scripts/17_adaptive_attacker.py --stage surrogate --seeds 42 \
          --rates 0.005,0.01 --costs 8,16 --max-candidates 64          # single-seed smoke first
      python scripts/17_adaptive_attacker.py --stage surrogate --seeds 42,123,456 \
          --rates 0.005,0.01 --costs 8,16 --max-candidates 64          # full surrogate search
      python scripts/17_adaptive_attacker.py --stage evaluation --seeds 789,1337
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup
from src import config
from src.adaptive_trigger import (
    CANDIDATE_COSTS,
    CANDIDATE_RATES,
    DIRECTION_GRID,
    N_SIGMA_GRID,
    REFERENCE_SURROGATE_SEED,
    aggregate_candidate_rows,
    candidate_trigger,
    choose_surrogate_winner,
    enumerate_candidates,
    evaluate_candidate,
    mad_false_positive_rate,
    trigger_hash,
)
from src.data import apply_standardiser
from src.detectors import poison_recall
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.models import attack_success_rate, clean_accuracy, mlp_penultimate_features, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.tb_vendor.constraints_numeric import check_constraints
from src.trigger import apply_trigger, build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET
SPECTRAL_K = config.SPECTRAL_COMPONENTS       # 5, matches scripts/05_run_detectors.py's recipe
MAD_Z = config.MAD_Z_PRIMARY                  # 3.0, the pre-registered defended cutoff
RESULTS_PATH = config.RESULTS / "adaptive_attacker.json"
SURROGATE_CKPT = "adaptive_attacker_surrogate.checkpoint.json"
EVALUATION_CKPT = "adaptive_attacker_evaluation.checkpoint.json"


# --- tiny sys.argv CLI (project convention -- see scripts/18b_ac_centroid_repair.py's _stage/_parse_cells) ---

def _stage(argv) -> str:
    if "--stage" not in argv:
        raise SystemExit("--stage surrogate|evaluation is required")
    stage = argv[argv.index("--stage") + 1]
    if stage not in ("surrogate", "evaluation"):
        raise SystemExit(f"--stage must be 'surrogate' or 'evaluation', got {stage!r}")
    return stage


def _parse_ints(argv, flag, default):
    if flag not in argv:
        return list(default)
    return [int(s) for s in argv[argv.index(flag) + 1].split(",")]


def _parse_floats(argv, flag, default):
    if flag not in argv:
        return list(default)
    return [float(s) for s in argv[argv.index(flag) + 1].split(",")]


def _max_candidates(argv) -> int:
    if "--max-candidates" not in argv:
        return config.ADAPTIVE_SEARCH_MAX_CANDIDATES
    return int(argv[argv.index("--max-candidates") + 1])


# --- checkpoint helpers, copied verbatim from scripts/05_run_detectors.py's pattern ---

def load_checkpoint(path, key):
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != key:
        print(f"{path.name}: checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"{path.name}: resuming from checkpoint, {len(blob.get('rows', {}))} unit(s) already done")
    return blob.get("rows", {})


def save_checkpoint(path, key, rows):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(path)   # atomic: a crash mid-write never corrupts the checkpoint


def _preserve_existing_evaluation(out: dict) -> dict:
    """Read-before-write guard for run_surrogate. `run_evaluation` reads the existing
    results/adaptive_attacker.json before writing (it only ever adds/updates the "evaluation" key on
    top of what is already there); `run_surrogate` builds `out` from scratch and, without this, would
    silently drop any already-written "evaluation" block on re-run (e.g. re-running the surrogate
    stage later to extend the candidate grid, after Step 6 has already run). Call this immediately
    before every RESULTS_PATH.write_text(...) in run_surrogate."""
    if RESULTS_PATH.exists():
        try:
            existing = json.loads(RESULTS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
        if "evaluation" in existing:
            out["evaluation"] = existing["evaluation"]
    return out


# --- shared measurement: one full poison(candidate/rate/seed) -> train -> detector cycle ---

def run_one_cycle(seed, S, cfg, device, trigger, rate, clean_acc_ref):
    """Poison at `rate` with `trigger` (a realizable {indices, values, project=True} dict, either the
    candidate's frozen watermark or the existing baseline SHAP trigger), train an MLP with `seed`, and
    measure everything the Step 7 conjunction and the ledger need. Returns a metrics dict; does not
    decide anything (that is evaluate_candidate's job, applied by the caller)."""
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])
    x_bot_raw = S["x_te_raw"][S["y_te"] == 1]

    x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
        x_tr_raw, y_tr, trigger, rate, S["scaler"], target=TARGET, seed=seed,
        constraints=constraints, feature_names=features, bounds=bounds)
    mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

    # feasibility is MEASURED on the actual poisoned raw rows, not assumed from project=True.
    if len(poison_idx):
        x_check = apply_trigger(x_tr_raw[poison_idx], trigger, constraints, features, bounds)
        feasible = bool(check_constraints(x_check, constraints, features).all())
    else:
        feasible = True

    x_bot_trig = apply_trigger(x_bot_raw, trigger, constraints, features, bounds)
    asr = attack_success_rate(mlp, apply_standardiser(S["scaler"], x_bot_trig), TARGET, device)
    cacc = clean_accuracy(mlp, x_te_std, S["y_te"], device)
    clean_accuracy_drop = clean_acc_ref - cacc

    benign_pos = np.where(y_p == TARGET)[0]
    is_poison = np.isin(benign_pos, poison_idx)
    has_poison = bool(is_poison.sum() > 0)
    feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
    scores = spectral_scores(feats, n_components=SPECTRAL_K)

    fixed_flag = flag_by_scores(scores, expected_frac=float(is_poison.mean())) if has_poison \
        else np.zeros(len(scores), dtype=bool)
    fixed_budget_recall = poison_recall(fixed_flag, is_poison) if has_poison else None
    mad_flag = flag_by_mad_threshold(scores, z_thresh=MAD_Z)
    mad_recall = poison_recall(mad_flag, is_poison) if has_poison else None
    mad_fpr = mad_false_positive_rate(mad_flag, is_poison)
    spectral_auc = (float(roc_auc_score(is_poison, scores))
                    if has_poison and is_poison.sum() < len(is_poison) else None)

    return dict(
        feasible=feasible, asr=float(asr), clean_accuracy=float(cacc),
        clean_accuracy_ref=float(clean_acc_ref), clean_accuracy_drop=float(clean_accuracy_drop),
        fixed_budget_recall=fixed_budget_recall, mad_recall=mad_recall, mad_fpr=mad_fpr,
        spectral_auc=spectral_auc, n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
    )


def build_reference_ranking(S, cfg, device):
    """The ONE surrogate ranking (seed=REFERENCE_SURROGATE_SEED's clean MLP) that freezes every
    candidate's concrete trigger. See src/adaptive_trigger.py's module docstring for why this is not
    recomputed per seed."""
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    clean_mlp_ref = train_mlp(x_tr_std, S["y_tr"], REFERENCE_SURROGATE_SEED,
                              epochs=cfg["mlp_epochs"], device=device)
    ranking_ref = shap_rank_features(clean_mlp_ref, S["x_tr_raw"], S["scaler"], target=TARGET,
                                     kind="mlp", device=device)
    return clean_mlp_ref, ranking_ref


def run_surrogate(argv, t0):
    seeds = _parse_ints(argv, "--seeds", config.ADAPTIVE_SURROGATE_SEEDS)
    rates = _parse_floats(argv, "--rates", CANDIDATE_RATES)
    costs = _parse_ints(argv, "--costs", CANDIDATE_COSTS)
    max_candidates = _max_candidates(argv)
    non_canonical = [s for s in seeds if s not in config.ADAPTIVE_SURROGATE_SEEDS]
    if non_canonical:
        print(f"WARNING: seeds {non_canonical} are not in config.ADAPTIVE_SURROGATE_SEEDS "
              f"{config.ADAPTIVE_SURROGATE_SEEDS} -- rows will be produced but cannot contribute to a "
              "valid (complete-coverage) selection.")

    cfg = get_config(smoke=False)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[surrogate] device={device} seeds={seeds} rates={rates} costs={costs} "
          f"max_candidates={max_candidates}")

    S = load_setup(cfg)
    clean_mlp_ref, ranking_ref = build_reference_ranking(S, cfg, device)
    accepted, rejected = enumerate_candidates(
        ranking_ref, S["x_tr_raw"], S["constraints"], S["features"], S["bounds"],
        costs=costs, rates=rates, max_candidates=max_candidates)
    print(f"[surrogate] candidate ledger: {len(accepted)} accepted, {len(rejected)} rejected "
          f"({time.time() - t0:.0f}s elapsed)")
    by_id = {c.candidate_id: c for c in accepted}

    # NOTE (asymmetry vs run_evaluation's fingerprint): this key does not include a content-hash of the
    # accepted candidate ledger itself the way run_evaluation's fingerprint hashes the winner trigger
    # (trigger_hash) -- it relies on n_accepted plus the grid/ranking-recipe params to catch a changed
    # ledger. A change to build_trigger/build_candidate_trigger's watermark FORMULA that altered
    # concrete values without changing the grid parameters or the accepted count could in principle
    # serve stale cached rows on resume. Low real risk (the existing pure-function tests would likely
    # catch such a change first), not fixed here to keep this key comparison cheap and simple.
    key = dict(stage="surrogate", rates=rates, costs=costs,
              max_candidates=max_candidates, mlp_epochs=cfg["mlp_epochs"], spectral_k=SPECTRAL_K,
              mad_z=MAD_Z, reference_seed=REFERENCE_SURROGATE_SEED,
              n_sigma_grid=list(N_SIGMA_GRID), direction_grid=list(DIRECTION_GRID),
              n_accepted=len(accepted))
    ckpt_path = config.RESULTS / SURROGATE_CKPT
    rows = load_checkpoint(ckpt_path, key)
    save_cb = lambda: save_checkpoint(ckpt_path, key, rows)

    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])

    for seed in seeds:
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        clean_mlp_seed = clean_mlp_ref if seed == REFERENCE_SURROGATE_SEED else \
            train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
        clean_acc_ref_seed = clean_accuracy(clean_mlp_seed, x_te_std, S["y_te"], device)

        for cand in accepted:
            unit_key = f"{seed}|{cand.candidate_id}"
            if unit_key in rows:
                continue
            trig = candidate_trigger(cand)
            metrics = run_one_cycle(seed, S, cfg, device, trig, cand.rate, clean_acc_ref_seed)
            rows[unit_key] = dict(seed=seed, split="surrogate", candidate_id=cand.candidate_id,
                                  cost=cand.cost, rate=cand.rate, n_sigma=cand.n_sigma,
                                  direction=cand.direction, **metrics)
            save_cb()
        print(f"  seed {seed}: {sum(1 for k in rows if k.startswith(f'{seed}|'))} candidate(s) done "
              f"({time.time() - t0:.0f}s elapsed)")

    print(f"[surrogate] {len(rows)} (seed, candidate) row(s) on disk "
          f"({time.time() - t0:.0f}s elapsed)")

    # Only freeze a selection artifact once the FULL, canonical surrogate seed set is covered for
    # every accepted candidate -- a partial/smoke run (e.g. --seeds 42) must not write a selection.
    have_full_coverage = all(
        all(f"{s}|{c.candidate_id}" in rows for s in config.ADAPTIVE_SURROGATE_SEEDS)
        for c in accepted
    ) and bool(accepted)

    out = dict(
        task="task6_score_aware_adaptive_attacker",
        config=dict(surrogate_seeds=list(config.ADAPTIVE_SURROGATE_SEEDS),
                   eval_seeds=list(config.ADAPTIVE_EVAL_SEEDS), rates=rates, costs=costs,
                   n_sigma_grid=list(N_SIGMA_GRID), direction_grid=list(DIRECTION_GRID),
                   max_candidates=max_candidates, reference_surrogate_seed=REFERENCE_SURROGATE_SEED,
                   spectral_k=SPECTRAL_K, mad_z=MAD_Z,
                   attack_effective_asr=config.ATTACK_EFFECTIVE_ASR,
                   fixed_budget_miss_recall=config.FIXED_BUDGET_MISS_RECALL,
                   clean_accuracy_drop_max=config.CLEAN_ACCURACY_DROP_MAX,
                   trigger_freeze_note="each candidate's trigger is built once from the reference "
                                       "surrogate seed's SHAP ranking and reused verbatim across every "
                                       "seed -- see src/adaptive_trigger.py module docstring"),
        candidate_ledger=dict(
            accepted=[dict(candidate_id=c.candidate_id, cost=c.cost, rate=c.rate, n_sigma=c.n_sigma,
                          direction=c.direction, feature_indices=list(c.feature_indices),
                          requested_values=list(c.requested_values),
                          projected_values=list(c.projected_values)) for c in accepted],
            rejected=rejected,
        ),
    )

    if not have_full_coverage:
        seeds_seen = sorted({int(r["seed"]) for r in rows.values()})
        out["selection"] = dict(
            verdict="incomplete_surrogate_coverage",
            note=f"seeds requested this run: {seeds}; seeds with rows on disk: {seeds_seen}; full "
                 f"coverage needs all of {list(config.ADAPTIVE_SURROGATE_SEEDS)} for every accepted "
                 "candidate before a winner can be selected. No selection written; existing "
                 "results/adaptive_attacker.json (if any) left untouched.")
        if RESULTS_PATH.exists():
            print(f"[surrogate] partial run ({seeds}) -- NOT overwriting existing "
                  f"{RESULTS_PATH.name} selection. Ledger-only summary printed above.")
            return
        out = _preserve_existing_evaluation(out)
        RESULTS_PATH.write_text(json.dumps(out, indent=2))
        print(f"[surrogate] partial run -- wrote candidate ledger only (no selection yet) to "
              f"{RESULTS_PATH}")
        return

    surrogate_rows_all = [r for r in rows.values() if r["candidate_id"] in by_id]
    aggregated = aggregate_candidate_rows(surrogate_rows_all, surrogate_seeds=config.ADAPTIVE_SURROGATE_SEEDS)
    winner = choose_surrogate_winner(surrogate_rows_all, surrogate_seeds=config.ADAPTIVE_SURROGATE_SEEDS)
    # eligible candidates ranked best-first by their own utility tuple; ineligible ones trail, by id.
    eligible_sorted = sorted((a for a in aggregated if a["utility"] is not None),
                             key=lambda a: tuple(a["utility"]), reverse=True)
    ineligible_sorted = sorted((a for a in aggregated if a["utility"] is None),
                               key=lambda a: a["candidate_id"])
    aggregated_sorted = eligible_sorted + ineligible_sorted

    if winner is None:
        out["selection"] = dict(verdict="no_eligible_candidate", winner=None,
                                aggregated_candidates=aggregated_sorted)
        print("[surrogate] VERDICT: no_eligible_candidate -- no candidate cleared feasibility + "
              "clean-accuracy gates on all 3 surrogate seeds. This is a legitimate, reportable "
              "outcome, not a bug.")
    else:
        winner_cand = by_id[winner["candidate_id"]]
        winner_trig = candidate_trigger(winner_cand)
        winner_hash = trigger_hash(winner_trig)
        out["selection"] = dict(
            verdict="selected", winner=winner,
            winner_trigger=dict(indices=winner_trig["indices"], values=winner_trig["values"],
                               hash=winner_hash),
            aggregated_candidates=aggregated_sorted,
        )
        print(f"[surrogate] VERDICT: selected candidate_id={winner['candidate_id']} "
              f"(cost={winner['cost']} rate={winner['rate']} n_sigma={winner['n_sigma']} "
              f"direction={winner['direction']}) mean_asr={winner['mean_asr']:.4f} "
              f"mean_mad_recall={winner['mean_mad_recall']:.4f} mean_mad_fpr={winner['mean_mad_fpr']:.4f}")

    out = _preserve_existing_evaluation(out)
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"[surrogate] wrote selection provenance to {RESULTS_PATH} "
          f"({time.time() - t0:.0f}s elapsed)")


def run_evaluation(argv, t0):
    seeds = _parse_ints(argv, "--seeds", config.ADAPTIVE_EVAL_SEEDS)
    leaked = [s for s in seeds if s in config.ADAPTIVE_SURROGATE_SEEDS]
    if leaked:
        raise SystemExit(f"--stage evaluation was asked to run on surrogate seed(s) {leaked} -- "
                         "refusing (would contaminate the held-out partition).")

    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} does not exist -- run --stage surrogate to completion "
                         "(all 3 surrogate seeds) first.")
    blob = json.loads(RESULTS_PATH.read_text())
    selection = blob.get("selection")
    if selection is None or selection.get("verdict") not in ("selected", "no_eligible_candidate"):
        raise SystemExit(f"{RESULTS_PATH} has no completed surrogate selection "
                         f"(selection={selection!r}) -- run --stage surrogate on all 3 surrogate "
                         "seeds first.")

    if selection["verdict"] == "no_eligible_candidate":
        blob["evaluation"] = dict(
            verdict="skipped_no_eligible_candidate",
            note="the surrogate search selected no candidate (evasion + attack-effectiveness + "
                 "clean-accuracy gates were never jointly satisfied on all 3 surrogate seeds) -- "
                 "there is nothing to evaluate on the held-out seeds. This is a legitimate stop, not "
                 "a threshold loosened to force a result.")
        RESULTS_PATH.write_text(json.dumps(blob, indent=2))
        print("[evaluation] surrogate verdict is no_eligible_candidate -- nothing to evaluate. "
              f"Wrote a skip record to {RESULTS_PATH}. Stopping.")
        return

    winner = selection["winner"]
    winner_trig_blob = selection["winner_trigger"]
    winner_trigger = dict(indices=list(winner_trig_blob["indices"]),
                          values=list(winner_trig_blob["values"]), project=True)
    recomputed_hash = trigger_hash(winner_trigger)
    if recomputed_hash != winner_trig_blob["hash"]:
        raise SystemExit(
            f"trigger-hash verification FAILED: the winner_trigger read back from {RESULTS_PATH} "
            f"hashes to {recomputed_hash} but the artifact recorded {winner_trig_blob['hash']} at "
            "selection time. Refusing to evaluate a trigger that cannot be verified identical to the "
            "surrogate-selected one (Step 6 hard requirement).")
    print(f"[evaluation] trigger-hash verified: {recomputed_hash} == stored hash. Evaluating "
          f"candidate_id={winner['candidate_id']} (cost={winner['cost']} rate={winner['rate']} "
          f"n_sigma={winner['n_sigma']} direction={winner['direction']}) on seeds={seeds}.")

    cfg = get_config(smoke=False)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = load_setup(cfg)
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])
    stats = raw_trigger_stats(x_tr_raw, constraints)
    cost, rate = int(winner["cost"]), float(winner["rate"])

    key = dict(stage="evaluation", seeds=seeds, candidate_id=winner["candidate_id"], cost=cost,
              rate=rate, mlp_epochs=cfg["mlp_epochs"], spectral_k=SPECTRAL_K, mad_z=MAD_Z,
              winner_trigger_hash=recomputed_hash)
    ckpt_path = config.RESULTS / EVALUATION_CKPT
    rows = load_checkpoint(ckpt_path, key)
    save_cb = lambda: save_checkpoint(ckpt_path, key, rows)

    for seed in seeds:
        print(f"--- eval seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
        clean_acc_ref = clean_accuracy(clean_mlp, x_te_std, S["y_te"], device)
        ranking_seed = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET,
                                          kind="mlp", device=device)

        # condition 3: clean (unpoisoned) reference -- FPR and clean-accuracy context only.
        clean_key = f"{seed}|clean_reference"
        if clean_key not in rows:
            benign_all = np.where(y_tr == TARGET)[0]
            feats_clean = mlp_penultimate_features(clean_mlp, x_tr_std[benign_all], device)
            scores_clean = spectral_scores(feats_clean, n_components=SPECTRAL_K)
            mad_flag_clean = flag_by_mad_threshold(scores_clean, z_thresh=MAD_Z)
            rows[clean_key] = dict(
                seed=seed, split="evaluation", condition="clean_reference",
                candidate_id=None, cost=None, rate=None, feasible=True, asr=None,
                clean_accuracy=float(clean_acc_ref), clean_accuracy_ref=float(clean_acc_ref),
                clean_accuracy_drop=0.0, fixed_budget_recall=None, mad_recall=None,
                mad_fpr=float(mad_flag_clean.mean()), spectral_auc=None,
                n_poison=0, n_benign=int(len(benign_all)))
            save_cb()

        # condition 1: existing baseline SHAP trigger (n_sigma=6.0), this seed's OWN ranking --
        # matches the already-validated pipeline's convention exactly (src/trigger.py, scripts/05).
        base_key = f"{seed}|baseline_shap_trigger"
        if base_key not in rows:
            realiz, _ = build_trigger(ranking_seed, cost, x_tr_raw, constraints, features, bounds,
                                      n_sigma=6.0, stats=stats)
            metrics = run_one_cycle(seed, S, cfg, device, realiz, rate, clean_acc_ref)
            rows[base_key] = dict(seed=seed, split="evaluation", condition="baseline_shap_trigger",
                                  candidate_id=None, cost=cost, rate=rate, n_sigma=6.0, direction=1,
                                  **metrics)
            save_cb()

        # condition 2: the selected adaptive trigger -- frozen, verbatim, hash-verified above.
        adaptive_key = f"{seed}|selected_adaptive_trigger"
        if adaptive_key not in rows:
            metrics = run_one_cycle(seed, S, cfg, device, winner_trigger, rate, clean_acc_ref)
            rows[adaptive_key] = dict(seed=seed, split="evaluation",
                                      condition="selected_adaptive_trigger",
                                      candidate_id=winner["candidate_id"], cost=cost, rate=rate,
                                      n_sigma=winner["n_sigma"], direction=winner["direction"],
                                      **metrics)
            save_cb()

    adaptive_rows = [r for r in rows.values() if r["condition"] == "selected_adaptive_trigger"]
    adaptive_outcomes = [evaluate_candidate(r) for r in adaptive_rows]
    per_seed_evasion = {int(o["seed"]): o["evasion_success"] for o in adaptive_outcomes}
    held_out_evasion_success = bool(adaptive_outcomes) and all(o["evasion_success"] for o in adaptive_outcomes)

    blob["evaluation"] = dict(
        verdict="evaluated", seeds=seeds, winner_trigger_hash_verified=True,
        rows=list(rows.values()), adaptive_outcomes=adaptive_outcomes,
        per_seed_evasion_success=per_seed_evasion,
        held_out_evasion_success=held_out_evasion_success,
        note="held_out_evasion_success requires evasion_success True on EVERY held-out seed "
             "(conjunction across seeds, not an average) -- a single held-out failure means the "
             "attack did not transfer.",
    )
    RESULTS_PATH.write_text(json.dumps(blob, indent=2))
    print(f"[evaluation] held_out_evasion_success={held_out_evasion_success} "
          f"per_seed={per_seed_evasion} -- wrote {RESULTS_PATH} ({time.time() - t0:.0f}s elapsed)")


def main():
    t0 = time.time()
    argv = sys.argv
    stage = _stage(argv)
    if stage == "surrogate":
        run_surrogate(argv, t0)
    else:
        run_evaluation(argv, t0)
    print(f"\nelapsed={time.time() - t0:.0f}s (stage={stage})")


if __name__ == "__main__":
    main()
