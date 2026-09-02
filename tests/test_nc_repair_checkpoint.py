"""Checkpoint/resume behaviour for scripts/18a_nc_repair.py (NC-3).

The NC-6 run this script is headed for is expected to run roughly 3x longer than the 4822-second run
that already lost all its progress to an uncheckpointed crash (notes/20260731-decision-7a-calibration-
gates-not-assessable.md). These tests guard the retrofit that fixes that: an atomic checkpoint of the
per-(seed, candidate, replica) clean rows and per-(seed, candidate, rate, cost) poisoned rows, keyed so
a config change discards stale progress instead of silently mixing it into a new run.

Loaded via importlib because the pipeline lives in a numbered script rather than an importable package
-- same pattern as tests/test_nc_repair_gate.py's `_load()`.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "nc_repair_checkpoint", ROOT / "scripts" / "18a_nc_repair.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nc = _load()


# --- key formats -------------------------------------------------------------------------------

def test_clean_key_format():
    assert nc.clean_key(42, "baseline", 0) == "42|baseline|0"
    assert nc.clean_key(1337, "constraint_aware", 9) == "1337|constraint_aware|9"


def test_poisoned_key_format():
    assert nc.poisoned_key(42, "baseline", 0.005, 16) == "42|baseline|0.005|16"


# --- load_checkpoint / save_checkpoint round trip -----------------------------------------------

def _sample_key():
    return dict(stage="calibration", seeds=[42], cells=[[0.005, 16]], candidate_grid=nc.CANDIDATES,
                nc_steps=100, nc_sample=200, mlp_epochs=10, smoke=True,
                nc_clean_replicas=10, nc_replica_seed_stride=7919,
                clean_calibration_quantile=0.99, nc_min_constraint_valid_fraction=355 / 358)


def test_checkpoint_round_trip_preserves_rows(tmp_path):
    path = tmp_path / "nc_repair_calibration.checkpoint.json"
    key = _sample_key()
    clean = {nc.clean_key(42, "baseline", 0):
             dict(seed=42, candidate="baseline", replica=0, ratio=1.23,
                  flagged_class=1, n_convergence_failures=0)}
    poisoned = {nc.poisoned_key(42, "baseline", 0.005, 16):
                dict(seed=42, candidate="baseline", rate=0.005, cost=16, asr=0.91, clean_acc=0.98)}

    nc.save_checkpoint(path, key, clean, poisoned)
    loaded_clean, loaded_poisoned = nc.load_checkpoint(path, key)

    assert loaded_clean == clean
    assert loaded_poisoned == poisoned


def test_load_checkpoint_missing_file_returns_empty_dicts(tmp_path):
    path = tmp_path / "does_not_exist.checkpoint.json"
    clean, poisoned = nc.load_checkpoint(path, _sample_key())
    assert clean == {}
    assert poisoned == {}


def test_load_checkpoint_corrupt_json_returns_empty_dicts(tmp_path):
    path = tmp_path / "nc_repair_calibration.checkpoint.json"
    path.write_text("{not valid json")
    clean, poisoned = nc.load_checkpoint(path, _sample_key())
    assert clean == {}
    assert poisoned == {}


# --- config-key mismatch discards the checkpoint (never mixed in) -------------------------------

def test_config_key_mismatch_discards_checkpoint(tmp_path, capsys):
    path = tmp_path / "nc_repair_calibration.checkpoint.json"
    original_key = _sample_key()
    clean = {nc.clean_key(42, "baseline", 0): dict(seed=42, candidate="baseline", replica=0)}
    poisoned = {nc.poisoned_key(42, "baseline", 0.005, 16): dict(seed=42, candidate="baseline")}
    nc.save_checkpoint(path, original_key, clean, poisoned)

    mismatched_key = dict(original_key)
    mismatched_key["seeds"] = [42, 123]   # seeds list widened -- must invalidate the checkpoint

    loaded_clean, loaded_poisoned = nc.load_checkpoint(path, mismatched_key)

    assert loaded_clean == {}
    assert loaded_poisoned == {}
    assert "config mismatch" in capsys.readouterr().out


def test_config_key_changes_when_min_constraint_valid_fraction_changes(monkeypatch):
    """Gate-2's threshold (NC-2) is part of what a checkpointed row's meaning depends on: rows scored
    against the old floor must not be silently reused once the floor moves."""
    cfg = dict(nc_steps=300, nc_sample=400, mlp_epochs=20, smoke=False)
    key1 = nc._config_key("calibration", [42], [(0.005, 16)], cfg)
    monkeypatch.setattr(nc.config, "NC_MIN_CONSTRAINT_VALID_FRACTION", 0.5)
    key2 = nc._config_key("calibration", [42], [(0.005, 16)], cfg)
    assert key1 != key2


def test_config_key_changes_when_candidate_grid_changes(monkeypatch):
    """If someone edits CANDIDATES' specs (e.g. l1_weight), a stale checkpoint scored under the old
    specs must be discarded -- _config_key reads the module-level CANDIDATES at call time, not a
    value frozen at import."""
    cfg = dict(nc_steps=300, nc_sample=400, mlp_epochs=20, smoke=False)
    key1 = nc._config_key("calibration", [42], [(0.005, 16)], cfg)
    new_grid = dict(nc.CANDIDATES)
    new_grid["baseline"] = dict(l1_weight=0.999, n_starts=1, constraint_weight=0.0)
    monkeypatch.setattr(nc, "CANDIDATES", new_grid)
    key2 = nc._config_key("calibration", [42], [(0.005, 16)], cfg)
    assert key1 != key2


def test_config_key_stable_for_identical_inputs():
    cfg = dict(nc_steps=300, nc_sample=400, mlp_epochs=20, smoke=False)
    key1 = nc._config_key("calibration", [42, 123], [(0.005, 16), (0.01, 16)], cfg)
    key2 = nc._config_key("calibration", [42, 123], [(0.005, 16), (0.01, 16)], cfg)
    assert key1 == key2


def test_config_key_differs_between_stages():
    cfg = dict(nc_steps=300, nc_sample=400, mlp_epochs=20, smoke=False)
    calib_key = nc._config_key("calibration", [42], [(0.005, 16)], cfg)
    eval_key = nc._config_key("evaluation", [42], [(0.005, 16)], cfg)
    assert calib_key != eval_key


# --- atomic write --------------------------------------------------------------------------------

def test_save_checkpoint_leaves_no_tmp_file_after_success(tmp_path):
    path = tmp_path / "nc_repair_calibration.checkpoint.json"
    nc.save_checkpoint(path, _sample_key(), {}, {})
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()


def test_save_checkpoint_uses_tmp_then_replace_like_05_run_detectors():
    """Source-level regression: the atomic-write discipline must match
    scripts/05_run_detectors.py's save_checkpoint -- write to a `.tmp` sibling via `.with_suffix`,
    then `Path.replace()` onto the final path, so a crash mid-write never corrupts the checkpoint."""
    src = inspect.getsource(nc.save_checkpoint)
    assert ".with_suffix(" in src
    assert ".replace(" in src
    assert "write_text" in src


def test_checkpoint_path_naming_matches_output_path_stem():
    """--stage calibration must produce exactly results/nc_repair_calibration.checkpoint.json;
    --stage evaluation the same convention against nc_repair.json."""
    calib_ckpt = nc.CALIBRATION_PATH.with_name(nc.CALIBRATION_PATH.stem + ".checkpoint.json")
    assert calib_ckpt.name == "nc_repair_calibration.checkpoint.json"
    eval_ckpt = nc.EVALUATION_PATH.with_name(nc.EVALUATION_PATH.stem + ".checkpoint.json")
    assert eval_ckpt.name == "nc_repair.checkpoint.json"


# --- _get_or_compute_row: the loop-splicing logic itself, not just the pure helpers -------------
#
# The tests above exercise clean_key/poisoned_key/load_checkpoint/save_checkpoint in isolation, but
# none of them touch what main() actually does with those pieces: the "if key in ckpt: reuse ... else:
# compute, store, append, save" wiring around the two row loops. That logic now lives in
# `_get_or_compute_row`, extracted verbatim from main()'s inline loop body (one copy used for both the
# clean-row and poisoned-row loops) so it is unit-testable without running run_clean_reference/run_cell
# or any of main()'s data-loading/training setup.

def test_get_or_compute_row_cache_hit_skips_compute_but_still_appends_cached_row():
    """A key already in the checkpoint must reuse the stored row without recomputing it -- the
    compute_fn is never called -- but the row is still appended to the output list and save_cb is
    NOT re-invoked (nothing changed, nothing new to persist)."""
    cached_row = {"seed": 42, "ratio": 1.23}
    ckpt = {"k": cached_row}
    rows = []
    compute_calls = []
    save_calls = []

    def compute_fn():
        compute_calls.append(1)
        return {"seed": 999, "ratio": -1.0}   # would be a visibly wrong row if this ran

    row, computed = nc._get_or_compute_row("k", ckpt, rows, compute_fn, lambda: save_calls.append(1))

    assert compute_calls == []                 # compute_fn NOT called
    assert computed is False
    assert row is cached_row
    assert rows == [cached_row]                # cached row still appended to the output list
    assert save_calls == []                    # no redundant persistence of unchanged state
    assert ckpt == {"k": cached_row}            # checkpoint dict untouched


def test_get_or_compute_row_cache_miss_computes_stores_appends_then_saves():
    """A key absent from the checkpoint must call compute_fn exactly once, store its result under
    `key` in the checkpoint dict, append it to the output list, AND persist via save_cb -- with the
    checkpoint dict already holding the new row by the time save_cb runs, so a crash right after
    save_cb returns can never leave the persisted file missing a row that `rows` already has."""
    ckpt = {}
    rows = []
    compute_calls = []
    fresh_row = {"seed": 42, "ratio": 2.5}
    save_snapshots = []

    def compute_fn():
        compute_calls.append(1)
        return fresh_row

    def save_cb():
        # snapshot what save_cb SEES in ckpt at the moment it is called
        save_snapshots.append(dict(ckpt))

    row, computed = nc._get_or_compute_row("k", ckpt, rows, compute_fn, save_cb)

    assert compute_calls == [1]                # compute_fn called exactly once
    assert computed is True
    assert row == fresh_row
    assert ckpt == {"k": fresh_row}             # stored under the right key
    assert rows == [fresh_row]                  # appended to the output list
    assert len(save_snapshots) == 1             # save_cb invoked exactly once
    assert save_snapshots[0] == {"k": fresh_row}   # ckpt already updated when save_cb ran


def test_main_uses_get_or_compute_row_for_both_row_kinds():
    """Source-level regression: main()'s clean-row and poisoned-row loops must route through the
    tested `_get_or_compute_row` helper rather than reintroducing inline duplicated checkpoint-or-
    compute logic that no test would catch a regression in."""
    src = inspect.getsource(nc.main)
    assert src.count("_get_or_compute_row(") == 2


# --- NC-4: train_victim/run_cell split + lazy per-(seed, rate, cost) victim caching -------------
#
# train_victim/run_cell are too heavy to unit-test directly (real dataset, real MLP training), so
# these tests exercise the loop-restructuring PATTERN itself -- a single-slot lazy cache around a
# stand-in for train_victim, wired through the real `_get_or_compute_row` per candidate -- the same
# style the `_get_or_compute_row` tests above already use for a fake `compute_fn`. This is the exact
# pattern main()'s poisoned-row loop now uses (scripts/18a_nc_repair.py, inside `for rate, cost in
# cells:`): the victim is trained on the first candidate whose row is a checkpoint MISS, and every
# other candidate for that cell reuses it.

def _lazy_victim_loop(candidates, poisoned_ckpt, poisoned_rows, save_cb, train_calls):
    """Mirrors main()'s poisoned-row loop body for ONE (seed, rate, cost) cell."""
    seed, rate, cost = 42, 0.005, 16
    victim_cache = {}

    def get_victim():
        if "v" not in victim_cache:
            train_calls.append((seed, rate, cost))
            victim_cache["v"] = dict(seed=seed, rate=rate, cost=cost, asr=0.9)
        return victim_cache["v"]

    for candidate in candidates:
        pk = nc.poisoned_key(seed, candidate, rate, cost)
        nc._get_or_compute_row(
            pk, poisoned_ckpt, poisoned_rows,
            lambda candidate=candidate: dict(get_victim(), candidate=candidate),
            save_cb)
    return poisoned_rows


def test_lazy_victim_getter_trains_at_most_once_across_candidates_in_a_cell():
    """The point of NC-4: with 4 candidates sharing one (seed, rate, cost) cell, the victim-training
    call must happen exactly once, not once per candidate."""
    candidates = ["baseline", "multistart", "sparser", "constraint_aware"]
    train_calls = []
    poisoned_ckpt, poisoned_rows = {}, []
    _lazy_victim_loop(candidates, poisoned_ckpt, poisoned_rows, lambda: None, train_calls)

    assert len(train_calls) == 1
    assert len(poisoned_rows) == len(candidates)
    assert {r["candidate"] for r in poisoned_rows} == set(candidates)


def test_lazy_victim_getter_never_trains_when_every_candidate_row_is_already_checkpointed():
    """The other half of NC-4's point: a cell where every candidate row is already checkpointed must
    not train a victim at all -- otherwise the lazy caching would defeat NC-3's checkpoint-skip
    savings (training a victim for a cell that needs no new work)."""
    candidates = ["baseline", "multistart", "sparser", "constraint_aware"]
    seed, rate, cost = 42, 0.005, 16
    poisoned_ckpt = {nc.poisoned_key(seed, c, rate, cost):
                     dict(seed=seed, rate=rate, cost=cost, candidate=c, asr=0.9)
                     for c in candidates}
    poisoned_rows = []
    train_calls = []
    _lazy_victim_loop(candidates, poisoned_ckpt, poisoned_rows, lambda: None, train_calls)

    assert train_calls == []
    assert len(poisoned_rows) == len(candidates)


def test_main_trains_victim_lazily_with_cell_outer_candidate_inner():
    """Source-level regression: main()'s poisoned-row section must call `train_victim` from exactly
    one call site (the lazy getter), nested under `for rate, cost in cells:` with `for candidate in
    candidates:` INSIDE that -- candidate-outer would retrain the victim once per candidate, which is
    exactly the waste NC-4 removes."""
    src = inspect.getsource(nc.main)
    assert src.count("train_victim(") == 1
    poisoned_section = src[src.index("for rate, cost in cells:"):]
    assert poisoned_section.index("for rate, cost in cells:") < poisoned_section.index(
        "for candidate in candidates:")
