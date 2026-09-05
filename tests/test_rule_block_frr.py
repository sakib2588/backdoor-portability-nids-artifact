"""Regression test: adding STRIP's FRR rule must not eat the MAD false-positive column.

Reproduced bug (2026-09-04, caught by diffing a rerun against the committed results file). The
`frr_recall` block was inserted into `rule_block` at four-space indent immediately before the
eight-space `out["mad_fpr"][str(z)] = ...` line. Python accepted it. The dedent silently ENDED the
`for z in Z_THRESHOLDS` loop one statement early, so `mad_fpr` was written once after the loop
using whatever `z` and `flag` were left over, and only when `entropy_cut` was not None. The
committed artifact came back with `mad_fpr == {}` for every arm scored without a cutoff and a
single leftover entry for the one arm scored with it, while every other value in the file was
bit-identical. Nothing raised.

That is the dangerous shape: a silently truncated results column looks exactly like a column that
was never requested. These tests assert the loop's completeness directly, per script, so the three
runners that share this function cannot drift apart or lose the column again.

The three `rule_block` implementations are deliberately near-identical across scripts 70, 74 and 81
so the corpora's columns are produced by the same arithmetic. They are tested as a group for the
same reason.

Loaded via importlib because the runners live in numbered scripts rather than an importable package
(same pattern as tests/test_secondary_poison_sweep.py and tests/test_nc_repair_gate.py).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

SCRIPTS = {
    "s70": "70_positive_control_strip_spectre.py",
    "s74": "74_secondary_positive_control.py",
    "s81": "81_netflow_detectors.py",
}


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module", params=sorted(SCRIPTS), ids=sorted(SCRIPTS))
def runner(request):
    return _load(request.param, SCRIPTS[request.param])


def _scores_and_labels(n=400, n_poison=40, seed=0):
    """Poison as a detached upper tail, so both rules have something to find."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=n)
    is_poison = np.zeros(n, dtype=bool)
    is_poison[:n_poison] = True
    scores[:n_poison] += 12.0
    return scores, is_poison


def test_mad_fpr_has_one_entry_per_threshold_without_a_cutoff(runner):
    """The bug's primary signature: an arm scored with no entropy cutoff lost the column entirely."""
    scores, is_poison = _scores_and_labels()
    out = runner.rule_block(scores, is_poison, float(is_poison.mean()))
    assert set(out["mad_fpr"]) == {str(z) for z in runner.Z_THRESHOLDS}
    assert set(out["mad_recall"]) == {str(z) for z in runner.Z_THRESHOLDS}


def test_mad_fpr_has_one_entry_per_threshold_with_a_cutoff(runner):
    """The bug's secondary signature: with a cutoff the column survived with one leftover entry."""
    scores, is_poison = _scores_and_labels(seed=1)
    out = runner.rule_block(scores, is_poison, float(is_poison.mean()), entropy_cut=1.0)
    assert set(out["mad_fpr"]) == {str(z) for z in runner.Z_THRESHOLDS}
    assert set(out["mad_recall"]) == {str(z) for z in runner.Z_THRESHOLDS}


def test_the_two_rules_are_unaffected_by_the_presence_of_a_cutoff(runner):
    """Adding STRIP's rule must be purely additive. Every pre-existing field keeps its value."""
    scores, is_poison = _scores_and_labels(seed=2)
    ef = float(is_poison.mean())
    without = runner.rule_block(scores, is_poison, ef)
    with_cut = runner.rule_block(scores, is_poison, ef, entropy_cut=1.0)
    for k in ("fixed_recall", "auc", "mad_recall", "mad_fpr"):
        assert without[k] == with_cut[k], f"{k} moved when the FRR rule was added"


def test_frr_fields_appear_only_when_a_cutoff_is_supplied(runner):
    scores, is_poison = _scores_and_labels(seed=3)
    ef = float(is_poison.mean())
    without = runner.rule_block(scores, is_poison, ef)
    for k in ("frr_recall", "frr_fpr", "entropy_cut", "frr_target"):
        assert k not in without
    with_cut = runner.rule_block(scores, is_poison, ef, entropy_cut=1.0)
    for k in ("frr_recall", "frr_fpr", "entropy_cut", "frr_target"):
        assert k in with_cut
    assert with_cut["frr_target"] == runner.STRIP_FRR


def test_frr_rule_recovers_a_detached_tail_and_reports_its_own_cost(runner):
    """A cutoff below the planted tail must catch it, and the false-positive cost must be real."""
    scores, is_poison = _scores_and_labels(seed=4)
    out = runner.rule_block(scores, is_poison, float(is_poison.mean()), entropy_cut=-6.0)
    assert out["frr_recall"] == pytest.approx(1.0)
    assert 0.0 <= out["frr_fpr"] <= 1.0
