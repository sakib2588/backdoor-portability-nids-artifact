"""Task 6: score-aware adaptive attacker against MAD-thresholded Spectral Signatures."""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest

from src import config
from src.adaptive_trigger import (
    aggregate_candidate_rows,
    candidate_trigger,
    choose_surrogate_winner,
    enumerate_candidates,
    evaluate_candidate,
    mad_false_positive_rate,
    trigger_hash,
    trigger_signature,
)
from src.data import build_ctu_constraints, fit_standardiser, load_ctu, temporal_split
from src.models import train_lightgbm
from src.tb_vendor.constraints_numeric import check_constraints
from src.trigger import shap_rank_features


@pytest.fixture(scope="module")
def env():
    """Real CTU constraints/bounds on a small feasible subsample -- mirrors tests/test_trigger.py's
    fixture. A LightGBM ranking (kind="tree") is used here purely as a fast, deterministic stand-in
    ranking to drive candidate generation; production candidate generation uses the MLP-GradientSHAP
    ranking (see scripts/17_adaptive_attacker.py), but enumerate_candidates itself is ranking-agnostic.
    """
    x, y, features, metadata = load_ctu()
    constraints = build_ctu_constraints(features)
    feasible = check_constraints(x.to_numpy(dtype=float), constraints, features)
    x_f = x[feasible].iloc[:12000].reset_index(drop=True)
    y_f = y[feasible].iloc[:12000].reset_index(drop=True)
    x_tr, y_tr, _, _, x_te, y_te = temporal_split(x_f, y_f, train_rows=9000, val_frac=0.2, seed=1319)
    scaler = fit_standardiser(x_tr)
    lgb = train_lightgbm(scaler.transform(x_tr.to_numpy()), y_tr.to_numpy(), seed=42)
    meta_idx = metadata.set_index("feature")
    bounds = (meta_idx.loc[features, "min"].to_numpy(float), meta_idx.loc[features, "max"].to_numpy(float))
    ranking = shap_rank_features(lgb, x_tr, scaler, kind="tree")
    x_tr_std = scaler.transform(x_tr.to_numpy())
    # extra fields (scaler/y_tr/x_te_raw/y_te/std_bounds) are only consumed by the run_surrogate
    # integration regression test below -- kept here so that test shares this module-scoped fixture
    # instead of paying for a second CTU load.
    return dict(features=features, constraints=constraints, bounds=bounds,
               std_bounds=(x_tr_std.min(axis=0), x_tr_std.max(axis=0)),
               x_tr_raw=x_tr.to_numpy(dtype=float), y_tr=y_tr.to_numpy(),
               x_te_raw=x_te.to_numpy(dtype=float), y_te=y_te.to_numpy(),
               scaler=scaler, ranking=ranking)


# --- row() test factory: a synthetic seed-level metrics row with sane defaults, overridable per test.
def row(**overrides):
    base = dict(
        seed=42, split="surrogate",
        candidate_id="cost008_rate0.0050_sigma3.00_dirp1",
        cost=8, rate=0.005, n_sigma=3.0, direction=1,
        feasible=True, asr=0.95, clean_accuracy_drop=0.0,
        mad_recall=0.1, mad_fpr=0.05, fixed_budget_recall=0.05, spectral_auc=0.95,
        n_poison=100, n_benign=10000,
    )
    base.update(overrides)
    return base


def rows_for_all_surrogate_seeds(**overrides):
    """Three rows, one per config.ADAPTIVE_SURROGATE_SEEDS, all for the same candidate."""
    return [row(seed=s, **overrides) for s in config.ADAPTIVE_SURROGATE_SEEDS]


# --- Step 1/enumerate_candidates ---

def test_candidate_generation_is_seed_independent_and_deterministic(env):
    first, first_rej = enumerate_candidates(
        env["ranking"], env["x_tr_raw"], env["constraints"], env["features"], env["bounds"],
        costs=(8,), rates=(0.005,))
    second, second_rej = enumerate_candidates(
        env["ranking"], env["x_tr_raw"], env["constraints"], env["features"], env["bounds"],
        costs=(8,), rates=(0.005,))
    assert first == second
    assert first_rej == second_rej
    assert len(first) > 0


def test_enumerate_candidates_respects_max_candidates_cap(env):
    accepted, _ = enumerate_candidates(
        env["ranking"], env["x_tr_raw"], env["constraints"], env["features"], env["bounds"],
        costs=(8, 16), rates=(0.005, 0.01), max_candidates=5)
    assert len(accepted) <= 5


def test_enumerate_candidates_logs_cap_reached_for_every_skipped_candidate():
    """Regression: candidates skipped once max_candidates is hit must still get a "cap_reached"
    ledger entry, not be silently dropped from both accepted and rejected (the docstring claims the
    rejection ledger is complete -- this is what makes that true once the cap actually fires)."""
    features = ["f0"]
    constraints = []
    x_tr_raw = np.array([[-1.0], [0.0], [1.0]])
    bounds = (np.array([-100.0]), np.array([100.0]))   # wide enough that nothing clips/dedups
    ranking = [0]

    accepted, rejected = enumerate_candidates(
        ranking, x_tr_raw, constraints, features, bounds,
        costs=(1,), rates=(0.005,), n_sigmas=(1.0, 2.0, 3.0), directions=(1, -1), max_candidates=2)

    assert len(accepted) == 2
    cap_reached = [r for r in rejected if r["reason"] == "cap_reached"]
    # 6 combinations total (3 n_sigma x 2 directions); 2 accepted -> the other 4 must be logged.
    assert len(cap_reached) == 4
    assert len(accepted) + len(rejected) == 6


def test_enumerate_candidates_rejects_duplicate_post_projection_within_same_rate():
    """Direct synthetic coverage of the "duplicate_post_projection_of" rejection path.

    This is the DOMINANT real-world rejection reason -- 20 of 48 candidates (42%) in the committed
    results/adaptive_attacker.json were excluded this way, which materially shaped which 28 candidates
    the surrogate winner was chosen from. Before this test, the only exposure was
    test_candidate_generation_is_seed_independent_and_deterministic, which only checks that repeated
    runs reject the SAME candidates -- a bug that made every run wrongly reject or wrongly accept a
    duplicate would pass that test unchanged, since it never checks the rejection is CORRECT.

    Two n_sigma values (5.0, 6.0) both push feature 0's raw watermark past a tight upper bound, so
    both clip to the SAME projected value under project_to_feasible's box-clip -- the exact mechanism
    behind the real rejections (constraints=[] here isolates the box-clip from the 360-constraint
    machinery; the real ledger's 20 duplicates are the same clip mechanism on real CTU bounds). Also
    directly checks the module's own stated dedup-key rationale (enumerate_candidates' docstring):
    dedup keys on (cost, rate) -- the SAME clipped values at a DIFFERENT rate must NOT be collapsed.
    """
    features = ["f0"]
    constraints = []                                    # isolates project_to_feasible's box-clip only
    x_tr_raw = np.array([[-1.0], [0.0], [1.0]])          # mean=0.0, population std=sqrt(2/3)
    bounds = (np.array([-100.0]), np.array([2.0]))       # tight hi: n_sigma=5 and 6 both saturate to it
    ranking = [0]

    accepted, rejected = enumerate_candidates(
        ranking, x_tr_raw, constraints, features, bounds,
        costs=(1,), rates=(0.005, 0.01), n_sigmas=(5.0, 6.0), directions=(1,))

    # exactly one accepted candidate per rate (n_sigma=5.0, first in lex order), both clipped to hi=2.0
    assert len(accepted) == 2
    assert {(c.rate, c.n_sigma) for c in accepted} == {(0.005, 5.0), (0.01, 5.0)}
    assert all(c.projected_values == (2.0,) for c in accepted)

    # the n_sigma=6.0 candidate at EACH rate is rejected as a duplicate of ITS OWN rate's accepted
    # candidate -- not collapsed across rates, per the dedup-key rationale.
    assert len(rejected) == 2
    by_rate = {r["rate"]: r for r in rejected}
    assert set(by_rate) == {0.005, 0.01}
    for rate, r in by_rate.items():
        assert r["n_sigma"] == 6.0
        assert r["reason"].startswith("duplicate_post_projection_of:")
        referenced_id = r["reason"].split(":", 1)[1]
        referenced = next(c for c in accepted if c.candidate_id == referenced_id)
        assert referenced.rate == rate, (
            "duplicate rejected against a candidate from a DIFFERENT rate -- rate must not be "
            "collapsed into the dedup key")


def test_enumerate_candidates_full_family_is_deterministic_lex_order(env):
    import itertools

    from src.adaptive_trigger import CANDIDATE_COSTS, CANDIDATE_RATES, DIRECTION_GRID, N_SIGMA_GRID

    accepted, _ = enumerate_candidates(
        env["ranking"], env["x_tr_raw"], env["constraints"], env["features"], env["bounds"])
    keys = [(c.cost, c.rate, c.n_sigma, c.direction) for c in accepted]
    # "lexical order" = the nested-loop enumeration order (cost, rate, n_sigma, direction), exactly as
    # listed in the plan's candidate-family table -- NOT numeric tuple sort (direction (1, -1) is not
    # numerically ascending).
    expected_order = [k for k in itertools.product(CANDIDATE_COSTS, CANDIDATE_RATES, N_SIGMA_GRID,
                                                    DIRECTION_GRID) if k in set(keys)]
    assert keys == expected_order
    assert len(accepted) <= config.ADAPTIVE_SEARCH_MAX_CANDIDATES
    # every accepted candidate is a genuinely distinct (cost, rate, n_sigma, direction) tuple
    assert len(set(keys)) == len(keys)


def test_realizable_candidate_trigger_is_feasible_on_real_rows(env):
    accepted, _ = enumerate_candidates(
        env["ranking"], env["x_tr_raw"], env["constraints"], env["features"], env["bounds"],
        costs=(8,), rates=(0.005,))
    assert accepted
    cand = accepted[0]
    trig = candidate_trigger(cand)
    from src.trigger import apply_trigger
    base = env["x_tr_raw"][:200]
    stamped = apply_trigger(base, trig, env["constraints"], env["features"], env["bounds"])
    assert check_constraints(stamped, env["constraints"], env["features"]).all()


# --- trigger hashing (Step 6 guard) ---

def test_trigger_hash_is_order_independent_and_stable():
    a = dict(indices=[3, 1, 2], values=[9.0, 7.0, 8.0], project=True)
    b = dict(indices=[1, 2, 3], values=[7.0, 8.0, 9.0], project=True)
    assert trigger_signature(a) == trigger_signature(b)
    assert trigger_hash(a) == trigger_hash(b)

    c = dict(indices=[1, 2, 3], values=[7.0, 8.0, 9.1], project=True)
    assert trigger_hash(a) != trigger_hash(c)


# --- Step 3: leakage + false-evasion guard tests (verbatim from the plan) ---

def test_surrogate_winner_rejects_evaluation_seed_rows():
    rows = [row(seed=789, split="evaluation")]
    with pytest.raises(ValueError, match="evaluation"):
        choose_surrogate_winner(rows)


def test_infeasible_candidate_cannot_win_even_with_low_mad_recall():
    rows = [row(candidate_id="bad", feasible=False, mad_recall=0.0, asr=1.0)]
    assert choose_surrogate_winner(rows) is None


def test_candidate_that_loses_asr_guardrail_cannot_claim_evasion():
    outcome = evaluate_candidate(row(asr=0.79, mad_recall=0.0, feasible=True))
    assert outcome["evasion_success"] is False


def test_candidate_generation_is_seed_independent_and_deterministic_pure_helper():
    """The evaluate_candidate/choose_surrogate_winner pure helpers have no hidden randomness either."""
    r = row(asr=0.95, mad_recall=0.1, feasible=True, clean_accuracy_drop=0.0)
    assert evaluate_candidate(r) == evaluate_candidate(dict(r))


# --- evaluate_candidate: the full Step 7 conjunction, each guardrail individually ---

def test_evasion_success_requires_all_four_gates():
    ok = row(feasible=True, asr=0.9, clean_accuracy_drop=0.0, mad_recall=0.1)
    assert evaluate_candidate(ok)["evasion_success"] is True

    infeasible = row(feasible=False, asr=0.9, clean_accuracy_drop=0.0, mad_recall=0.1)
    assert evaluate_candidate(infeasible)["evasion_success"] is False

    weak_asr = row(feasible=True, asr=0.5, clean_accuracy_drop=0.0, mad_recall=0.1)
    assert evaluate_candidate(weak_asr)["evasion_success"] is False

    costly = row(feasible=True, asr=0.9, clean_accuracy_drop=0.05, mad_recall=0.1)
    assert evaluate_candidate(costly)["evasion_success"] is False

    still_caught = row(feasible=True, asr=0.9, clean_accuracy_drop=0.0, mad_recall=0.9)
    assert evaluate_candidate(still_caught)["evasion_success"] is False


def test_mad_false_positive_rate_matches_hand_count():
    is_poison = np.array([True, True, False, False, False, False])
    flagged = np.array([True, False, True, True, False, False])
    # clean rows: indices 2,3,4,5 (4 total); flagged-and-clean: 2,3 -> 2/4
    assert mad_false_positive_rate(flagged, is_poison) == pytest.approx(0.5)


# --- Step 2: locked utility / tie-break, aggregated over the 3 surrogate seeds ---

def test_choose_surrogate_winner_prefers_lower_mad_recall_at_equal_asr():
    weak = rows_for_all_surrogate_seeds(candidate_id="weak", cost=8, n_sigma=1.0,
                                        asr=0.9, mad_recall=0.05, mad_fpr=0.05,
                                        clean_accuracy_drop=0.0, feasible=True)
    strong = rows_for_all_surrogate_seeds(candidate_id="strong", cost=8, n_sigma=6.0,
                                          asr=0.9, mad_recall=0.9, mad_fpr=0.05,
                                          clean_accuracy_drop=0.0, feasible=True)
    winner = choose_surrogate_winner(weak + strong)
    assert winner is not None
    assert winner["candidate_id"] == "weak"


def test_choose_surrogate_winner_requires_attack_effective_asr_first():
    """The leading Boolean term is non-negotiable: a candidate that evades MAD but fails the ASR bar
    must lose to one that clears the bar even with worse (higher) MAD recall."""
    evades_but_ineffective = rows_for_all_surrogate_seeds(
        candidate_id="ineffective", asr=0.5, mad_recall=0.0, mad_fpr=0.0,
        clean_accuracy_drop=0.0, feasible=True)
    effective_but_caught_more = rows_for_all_surrogate_seeds(
        candidate_id="effective", asr=0.85, mad_recall=0.3, mad_fpr=0.05,
        clean_accuracy_drop=0.0, feasible=True)
    winner = choose_surrogate_winner(evades_but_ineffective + effective_but_caught_more)
    assert winner is not None
    assert winner["candidate_id"] == "effective"


def test_aggregate_candidate_rows_excludes_incomplete_seed_coverage():
    only_one_seed = [row(candidate_id="partial", seed=config.ADAPTIVE_SURROGATE_SEEDS[0])]
    agg = aggregate_candidate_rows(only_one_seed)
    assert len(agg) == 1
    assert agg[0]["eligible"] is False
    assert any("incomplete_seed_coverage" in reason for reason in agg[0]["exclusion_reasons"])


def test_aggregate_candidate_rows_excludes_clean_accuracy_drop_over_gate():
    over_budget = rows_for_all_surrogate_seeds(candidate_id="costly", feasible=True,
                                               clean_accuracy_drop=config.CLEAN_ACCURACY_DROP_MAX + 0.01)
    agg = aggregate_candidate_rows(over_budget)
    assert agg[0]["eligible"] is False
    assert "clean_accuracy_drop_exceeds_gate_on_a_surrogate_seed" in agg[0]["exclusion_reasons"]


# --- regression: run_surrogate must never drop an existing "evaluation" block on rerun ---
#
# Reported live by a spec reviewer: run_surrogate built its results/adaptive_attacker.json output
# dict from scratch and wrote it directly, unlike run_evaluation (which reads the file first and only
# ever adds/updates its own "evaluation" key). Re-running the surrogate stage after Step 6 had already
# produced held-out numbers silently deleted the "evaluation" block from disk. Fixed by
# scripts/17_adaptive_attacker.py's _preserve_existing_evaluation, called at both RESULTS_PATH write
# sites in run_surrogate. This test exercises the REAL run_surrogate end-to-end (loaded via importlib,
# matching tests/test_nc_repair_gate.py's convention for numbered-script code) with a genuine no-op
# resume -- every (seed, candidate) unit is pre-checkpointed, so no poisoning/training happens, only
# the two heavy setup calls (data load, reference-ranking MLP) are monkeypatched to reuse the module's
# small `env` fixture instead of the real ~150k-row pipeline.

def _load_adaptive_attacker_script():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "adaptive_attacker_script", root / "scripts" / "17_adaptive_attacker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_surrogate_preserves_existing_evaluation_block_on_rerun(tmp_path, monkeypatch, env):
    mod = _load_adaptive_attacker_script()

    # isolate the script's file-based state: config.RESULTS drives the checkpoint path computed
    # inside run_surrogate; RESULTS_PATH is the script's own module-level constant for the main
    # results file and must be patched separately (it was already bound at import time).
    monkeypatch.setattr(mod.config, "RESULTS", tmp_path)
    monkeypatch.setattr(mod, "RESULTS_PATH", tmp_path / "adaptive_attacker.json")

    mlp_epochs = 1
    monkeypatch.setattr(mod, "get_config", lambda smoke: dict(mlp_epochs=mlp_epochs))
    monkeypatch.setattr(mod, "load_setup", lambda cfg: env)

    x_tr_std = mod.apply_standardiser(env["scaler"], env["x_tr_raw"])
    ref_model = mod.train_mlp(x_tr_std, env["y_tr"], mod.REFERENCE_SURROGATE_SEED,
                              epochs=mlp_epochs, device="cpu")
    ranking = list(range(len(env["features"])))
    monkeypatch.setattr(mod, "build_reference_ranking", lambda S, cfg, device: (ref_model, ranking))

    costs, rates = (8,), (0.005,)
    accepted, _ = mod.enumerate_candidates(ranking, env["x_tr_raw"], env["constraints"],
                                           env["features"], env["bounds"], costs=costs, rates=rates)
    assert accepted, "test setup produced no candidates -- fixture or grid is degenerate"

    # pre-fill a checkpoint with every (seed, candidate) unit already done, config_key matching
    # exactly what run_surrogate will itself compute -- a genuine no-op resume, not a partial one.
    key = dict(stage="surrogate", rates=list(rates), costs=list(costs),
              max_candidates=config.ADAPTIVE_SEARCH_MAX_CANDIDATES, mlp_epochs=mlp_epochs,
              spectral_k=mod.SPECTRAL_K, mad_z=mod.MAD_Z,
              reference_seed=mod.REFERENCE_SURROGATE_SEED,
              n_sigma_grid=list(mod.N_SIGMA_GRID), direction_grid=list(mod.DIRECTION_GRID),
              n_accepted=len(accepted))
    fabricated_metrics = dict(feasible=True, asr=1.0, clean_accuracy=1.0, clean_accuracy_ref=1.0,
                              clean_accuracy_drop=0.0, fixed_budget_recall=0.0, mad_recall=1.0,
                              mad_fpr=0.05, spectral_auc=0.99, n_poison=10, n_benign=1000)
    rows = {
        f"{seed}|{cand.candidate_id}": dict(seed=seed, split="surrogate",
                                            candidate_id=cand.candidate_id, cost=cand.cost,
                                            rate=cand.rate, n_sigma=cand.n_sigma,
                                            direction=cand.direction, **fabricated_metrics)
        for cand in accepted for seed in config.ADAPTIVE_SURROGATE_SEEDS
    }
    (tmp_path / mod.SURROGATE_CKPT).write_text(json.dumps(dict(config_key=key, rows=rows)))

    # seed an existing results file with a COMPLETED evaluation block, as if Step 6 had already run.
    existing_evaluation = dict(verdict="evaluated", held_out_evasion_success=False,
                               rows=[dict(seed=789, condition="clean_reference")])
    mod.RESULTS_PATH.write_text(json.dumps(dict(
        task="task6_score_aware_adaptive_attacker", config={}, candidate_ledger={},
        selection=dict(verdict="selected", winner=dict(candidate_id="placeholder")),
        evaluation=existing_evaluation)))

    mod.run_surrogate(
        ["--stage", "surrogate", "--seeds", "42,123,456", "--rates", "0.005", "--costs", "8"],
        time.time())

    out = json.loads(mod.RESULTS_PATH.read_text())
    assert out.get("evaluation") == existing_evaluation, (
        "run_surrogate dropped (or altered) the pre-existing evaluation block on a full-coverage "
        "rerun -- the read-before-write merge regressed")
    assert out["selection"]["verdict"] in ("selected", "no_eligible_candidate")
