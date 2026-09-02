"""Unit tests for the Rung-2 pilot statistics (scripts/64, 65, 66).

These cover the pure functions only -- the parts that turn an inverted mask or a fitted mixture into
a number. The runs themselves are GPU experiments with their own checkpointed artifacts; what is
worth pinning here is that the statistics mean what the pre-registration says they mean, because a
silently wrong statistic would produce a confident, reproducible, wrong verdict.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _load(script_name: str, module_name: str):
    """Import a numbered script by path -- `scripts/65_...` is not a legal module name."""
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / script_name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


mining = _load("64_nc_normpair_mining.py", "nc_normpair_mining")
geom = _load("65_nc_mask_geometry_pilot.py", "nc_mask_geometry_pilot")
acp = _load("66_ac_continuous_score_pilot.py", "ac_continuous_score_pilot")


# ---------------------------------------------------------------------------------------------
# AUC with midranks -- the tie convention the pre-registration locks
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("auc_fn", [mining.auc_midrank, acp.auc_midrank])
def test_auc_perfect_separation(auc_fn):
    scores = np.array([1.0, 2.0, 3.0, 10.0, 11.0])
    labels = np.array([0, 0, 0, 1, 1])
    assert auc_fn(scores, labels) == pytest.approx(1.0)


@pytest.mark.parametrize("auc_fn", [mining.auc_midrank, acp.auc_midrank])
def test_auc_all_ties_is_half_not_one(auc_fn):
    """The failure mode this convention exists to prevent: a saturating score where every value is
    identical must read 0.5, not 1.0. A naive '>' comparison would score it 0.0 or 1.0 by accident of
    sort order."""
    scores = np.ones(10)
    labels = np.array([0] * 7 + [1] * 3)
    assert auc_fn(scores, labels) == pytest.approx(0.5)


@pytest.mark.parametrize("auc_fn", [mining.auc_midrank, acp.auc_midrank])
def test_auc_partial_ties_matches_hand_computation(auc_fn):
    # Positives at 2.0 tie with one negative at 2.0: AUC = (1 full win + 0.5 tie) / 2 = 0.75
    scores = np.array([1.0, 2.0, 2.0])
    labels = np.array([0, 0, 1])
    assert auc_fn(scores, labels) == pytest.approx(0.75)


def test_two_sided_auc_is_direction_free():
    scores = np.array([10.0, 9.0, 8.0, 1.0, 0.0])
    labels = np.array([0, 0, 0, 1, 1])
    # Victims rank LOW: directed AUC is 0, but the two-sided reading is a perfect separation.
    assert mining.auc_midrank(scores, labels) == pytest.approx(0.0)
    assert mining.two_sided_auc(scores, labels) == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------
# N1 concentration
# ---------------------------------------------------------------------------------------------

def test_n1_concentrated_mask_beats_uniform_on_every_measure():
    d, k = 100, 16
    spike = np.zeros(d)
    spike[:k] = 1.0
    uniform = np.ones(d)

    s = geom.n1_concentration(spike, top_k=k)
    u = geom.n1_concentration(uniform, top_k=k)

    assert s["top_k_share"] == pytest.approx(1.0)
    assert u["top_k_share"] == pytest.approx(k / d)
    assert s["entropy"] < u["entropy"]      # concentrated == low entropy
    assert s["gini"] > u["gini"]            # concentrated == high inequality


def test_n1_is_scale_free():
    """scripts/64 showed raw mask SIZE carries nothing, so the shape statistics must not smuggle it
    back in: scaling a mask by 1000 must not change any of them."""
    rng = np.random.default_rng(0)
    mask = np.abs(rng.normal(size=200))
    a = geom.n1_concentration(mask)
    b = geom.n1_concentration(mask * 1000.0)
    for key in a:
        assert a[key] == pytest.approx(b[key])


def test_n1_all_zero_mask_does_not_divide_by_zero():
    out = geom.n1_concentration(np.zeros(50))
    assert out == {"top_k_share": 0.0, "entropy": 0.0, "gini": 0.0}


# ---------------------------------------------------------------------------------------------
# N2 SHAP alignment
# ---------------------------------------------------------------------------------------------

def test_n2_perfect_and_disjoint_overlap():
    d, k = 60, 16
    mask = np.zeros(d)
    mask[:k] = np.linspace(1.0, 2.0, k)

    same = list(range(k)) + list(range(k, d))
    assert geom.n2_shap_alignment(mask, same, top_k=k)["jaccard"] == pytest.approx(1.0)
    assert geom.n2_shap_alignment(mask, same, top_k=k)["overlap_count"] == k

    disjoint = list(range(k, d)) + list(range(k))
    out = geom.n2_shap_alignment(mask, disjoint, top_k=k)
    assert out["jaccard"] == pytest.approx(0.0)
    assert out["overlap_count"] == 0


def test_n2_mass_share_weights_by_magnitude_not_count():
    d, k = 40, 4
    mask = np.zeros(d)
    mask[0] = 100.0        # nearly all the mass on one SHAP-important feature
    mask[10:13] = 1.0
    ranking = [0] + list(range(20, d)) + list(range(1, 20))
    out = geom.n2_shap_alignment(mask, ranking, top_k=k)
    assert out["overlap_count"] == 1
    assert out["shap_mass_share"] > 0.95   # counted once, but carries the mass


# ---------------------------------------------------------------------------------------------
# N3 restart stability
# ---------------------------------------------------------------------------------------------

def _start(target, start, mask):
    return {"target_class": target, "start": start, "mask": np.asarray(mask, dtype=float)}


def test_n3_identical_starts_are_perfectly_stable():
    m = np.array([1.0, 2.0, 3.0])
    starts = [_start(0, 0, m), _start(0, 1, m), _start(0, 2, m)]
    conv = {(0, 0): True, (0, 1): True, (0, 2): True}
    out = geom.n3_restart_stability(starts, 0, conv)
    assert out["n_converged_starts"] == 3
    assert out["mask_l1_std"] == pytest.approx(0.0)
    assert out["mean_pairwise_cosine"] == pytest.approx(1.0)


def test_n3_orthogonal_starts_are_maximally_unstable():
    starts = [_start(0, 0, [1.0, 0.0]), _start(0, 1, [0.0, 1.0])]
    conv = {(0, 0): True, (0, 1): True}
    out = geom.n3_restart_stability(starts, 0, conv)
    assert out["mean_pairwise_cosine"] == pytest.approx(0.0)


def test_n3_undefined_below_two_converged_starts_rather_than_zero():
    """A single converged start must read undefined, NOT a stability of zero -- a zero would be
    indistinguishable from a genuinely perfectly-stable inversion in the aggregate."""
    starts = [_start(0, 0, [1.0, 2.0]), _start(0, 1, [5.0, 6.0])]
    conv = {(0, 0): True, (0, 1): False}     # second start diverged
    out = geom.n3_restart_stability(starts, 0, conv)
    assert out["n_converged_starts"] == 1
    assert out["mask_l1_std"] is None
    assert out["mean_pairwise_cosine"] is None


def test_n3_ignores_other_classes():
    starts = [_start(0, 0, [1.0, 1.0]), _start(1, 0, [9.0, 9.0]), _start(0, 1, [1.0, 1.0])]
    conv = {(0, 0): True, (0, 1): True, (1, 0): True}
    out = geom.n3_restart_stability(starts, 0, conv)
    assert out["n_converged_starts"] == 2


# ---------------------------------------------------------------------------------------------
# AC pilot: depth score and the matched-FPR sweep
# ---------------------------------------------------------------------------------------------

def test_depth_score_orders_within_the_minority_where_the_posterior_ties():
    """The reason the pre-registration picks depth over the GMM posterior. Inside a well-separated
    component every posterior saturates to 1.0, giving one tied value and no ordering to threshold;
    the component log-density keeps a distinct value per sample."""
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.0, 1.0, (200, 3)), rng.normal(8.0, 1.0, (40, 3))])
    gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    labels = gm.predict(X)
    is_poison = np.zeros(len(X), dtype=bool)
    is_poison[200:] = True

    scores, minority, sizes = acp.depth_score(X, gm, labels, is_poison)
    in_minority = labels == minority

    assert sizes[minority] == in_minority.sum()
    assert len(np.unique(np.round(gm.predict_proba(X)[in_minority, minority], 6))) == 1
    assert len(np.unique(np.round(scores[in_minority], 6))) == int(in_minority.sum())


def test_depth_score_picks_minority_without_consulting_labels():
    """The suspicious cluster must be chosen by SIZE, as canonical AC does. If `is_poison` could
    influence the choice the detector would be reading the answer key."""
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0.0, 1.0, (300, 2)), rng.normal(9.0, 1.0, (30, 2))])
    gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    labels = gm.predict(X)

    truthful = np.zeros(len(X), dtype=bool); truthful[300:] = True
    lying = ~truthful
    a, min_a, _ = acp.depth_score(X, gm, labels, truthful)
    b, min_b, _ = acp.depth_score(X, gm, labels, lying)
    assert min_a == min_b
    assert np.allclose(a, b)


def test_recall_at_fpr_budget_respects_the_budget():
    rng = np.random.default_rng(2)
    clean = rng.normal(0.0, 1.0, 1000)
    poison = rng.normal(4.0, 1.0, 50)
    scores = np.concatenate([clean, poison])
    is_poison = np.zeros(len(scores), dtype=bool)
    is_poison[1000:] = True

    recall = acp.recall_at_fpr_budget(scores, is_poison, 0.01)
    thresh = float(np.quantile(clean, 0.99))
    realised_fpr = float((clean > thresh).mean())
    assert realised_fpr <= 0.011
    assert 0.0 <= recall <= 1.0
    # A budget that admits more false alarms cannot lower recall.
    assert acp.recall_at_fpr_budget(scores, is_poison, 0.10) >= recall


def test_recall_at_fpr_budget_is_one_for_perfect_separation():
    scores = np.concatenate([np.zeros(100), np.full(10, 50.0)])
    is_poison = np.zeros(110, dtype=bool)
    is_poison[100:] = True
    assert acp.recall_at_fpr_budget(scores, is_poison, 0.01) == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------
# The descriptive rank gate (scripts/65)
# ---------------------------------------------------------------------------------------------

def test_rank_gate_reports_the_binomial_bound_it_cannot_exceed():
    """The gate's whole point: 3 of 3 victims flagged is NOT evidence of a 0.9 detection rate. The
    artifact must carry the bound that says so."""
    assert geom._binomial_lower_bound(3, 3) == pytest.approx(0.368, abs=0.01)
    assert geom._binomial_lower_bound(5, 5) == pytest.approx(0.549, abs=0.01)
    assert geom._binomial_lower_bound(0, 3) == pytest.approx(0.0)


def test_rank_gate_passes_only_when_every_victim_clears_the_bar():
    rng = np.random.default_rng(0)
    clean = list(rng.normal(0.0, 1.0, 100))

    separated = geom.evaluate_statistic("f", clean, [8.0, 9.0, 10.0], np.random.default_rng(0))
    assert separated["passes_rank_gate"] is True
    assert separated["verdict"] == "consistent_with_separation"
    assert separated["permutation_p"] < 0.05

    # One victim sitting in the bulk is enough to fail it.
    mixed = geom.evaluate_statistic("f", clean, [8.0, 9.0, 0.0], np.random.default_rng(0))
    assert mixed["passes_rank_gate"] is False
    assert mixed["verdict"] == "victims_inside_clean_null"


def test_rank_gate_detects_victims_that_rank_low():
    rng = np.random.default_rng(0)
    clean = list(rng.normal(0.0, 1.0, 100))
    out = geom.evaluate_statistic("f", clean, [-8.0, -9.0, -10.0], np.random.default_rng(0))
    assert out["direction"] == "below"
    assert out["passes_rank_gate"] is True
