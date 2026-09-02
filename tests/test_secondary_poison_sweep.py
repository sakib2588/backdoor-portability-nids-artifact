"""Regression test: a smoke run must never touch the full-grid results/checkpoint artifact
(scripts/14_secondary_poison_sweep.py, extension Task 4).

Reproduced bug (spec review, 2026-07-31): `results_paths(smoke)` (and, before this fix, both
`main()`'s checkpoint path and `write_results`'s output path) used to resolve to the SAME fixed
filenames regardless of `smoke` -- so running the documented smoke command (Task 4 Step 3) AFTER the
full 40-cell grid was already committed silently overwrote it. This test writes a full-grid-shaped
results file, runs the actual write path a smoke invocation takes, and asserts the full-grid file is
byte-for-byte unchanged afterward -- the same class of bug and fix shape as a sibling task's
run_surrogate-preserves-evaluation regression test: a partial/exploratory run must never clobber a
more-complete committed one.

Loaded via importlib because the sweep lives in a numbered script rather than an importable package
(same pattern as tests/test_nc_repair_gate.py)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "secondary_poison_sweep", ROOT / "scripts" / "14_secondary_poison_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep = _load()


def test_smoke_and_full_paths_are_distinct_filenames():
    full_results, full_ckpt = sweep.results_paths(smoke=False)
    smoke_results, smoke_ckpt = sweep.results_paths(smoke=True)

    assert full_results != smoke_results
    assert full_ckpt != smoke_ckpt
    assert full_results.name == "secondary_poison_sweep.json"
    assert full_ckpt.name == "secondary_poison_sweep.checkpoint.json"
    assert smoke_results.name == "secondary_poison_sweep_smoke.json"
    assert smoke_ckpt.name == "secondary_poison_sweep_smoke.checkpoint.json"


def test_smoke_write_results_does_not_touch_full_grid_artifact(tmp_path, monkeypatch):
    """The exact bug the reviewer reproduced hands-on: write a full-grid-shaped results file, then
    call write_results with smoke=True (what --smoke actually does at the end of a run), and assert
    the full-grid file is byte-unchanged."""
    from src import config

    monkeypatch.setattr(config, "RESULTS", tmp_path)

    full_results_path, full_ckpt_path = sweep.results_paths(smoke=False)
    full_results_payload = json.dumps({
        "dataset": "unsw_nb15", "blocked": False,
        "per_cell": [{"seed": s, "rate": r, "cost": c} for s in (42, 123, 456, 789, 1337)
                    for r in (0.005, 0.01, 0.05, 0.1) for c in (8, 16)],
        "marker": "FULL_GRID_DO_NOT_TOUCH",
    })
    full_ckpt_payload = json.dumps({
        "config_key": {"smoke": False, "rates": [0.005, 0.01, 0.05, 0.1], "costs": [8, 16]},
        "cells": {"marker": "FULL_GRID_CHECKPOINT_DO_NOT_TOUCH"},
    })
    full_results_path.parent.mkdir(parents=True, exist_ok=True)
    full_results_path.write_text(full_results_payload)
    full_ckpt_path.write_text(full_ckpt_payload)
    assert len(json.loads(full_results_payload)["per_cell"]) == 40   # sanity: this IS full-grid-shaped

    smoke_cfg = sweep.get_config(smoke=True)
    sweep.write_results(
        smoke_cfg, True,
        clean_rows={"42": {"seed": 42, "clean_accuracy": 0.9}},
        control_rows={}, cells={},
    )

    # the full-grid results file must be untouched -- byte-for-byte
    assert full_results_path.read_text() == full_results_payload
    # write_results never touches the checkpoint at all, but confirm nothing else did either
    assert full_ckpt_path.read_text() == full_ckpt_payload

    smoke_results_path, _ = sweep.results_paths(smoke=True)
    assert smoke_results_path.exists()
    assert smoke_results_path != full_results_path
    smoke_out = json.loads(smoke_results_path.read_text())
    assert smoke_out["config"]["smoke"] is True


def test_smoke_checkpoint_path_is_distinct_from_full_checkpoint(tmp_path, monkeypatch):
    """The other half of the bug: `main()`'s checkpoint load/save must resolve to the smoke-scoped
    path under `--smoke`, never the full run's checkpoint filename."""
    from src import config

    monkeypatch.setattr(config, "RESULTS", tmp_path)

    full_results_path, full_ckpt_path = sweep.results_paths(smoke=False)
    full_ckpt_payload = json.dumps({"config_key": {"smoke": False}, "cells": {"real": "data"}})
    full_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    full_ckpt_path.write_text(full_ckpt_payload)

    smoke_results_path, smoke_ckpt_path = sweep.results_paths(smoke=True)
    sweep.save_checkpoint(smoke_ckpt_path, {"smoke": True}, clean={}, controls={}, cells={"fake": 1})

    assert full_ckpt_path.read_text() == full_ckpt_payload
    assert smoke_ckpt_path.exists()
    assert smoke_ckpt_path != full_ckpt_path
