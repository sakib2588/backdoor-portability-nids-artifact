import importlib.util
import pathlib

import pytest

spec = importlib.util.spec_from_file_location(
    "displacement", pathlib.Path("scripts/92_displacement_axis_analysis.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _row(seed, sigma, cost, asr, fixed, mad, auc, feasible=True):
    return dict(seed=seed, n_sigma=sigma, cost=cost, rate=0.005, direction=1, feasible=feasible,
                asr=asr, fixed_budget_recall=fixed, mad_recall=mad, mad_fpr=0.05,
                spectral_auc=auc, candidate_id=f"c{sigma}_{cost}")


def test_aggregate_groups_by_sigma_and_cost_and_averages_over_seeds():
    rows = [_row(42, 1.0, 8, 0.1, 0.0, 0.1, 0.3), _row(123, 1.0, 8, 0.3, 0.2, 0.3, 0.5),
            _row(42, 6.0, 8, 1.0, 0.9, 1.0, 0.99)]
    table = mod.aggregate(rows)
    by = {(t["n_sigma"], t["cost"]): t for t in table}
    assert by[(1.0, 8)]["n_seeds"] == 2
    assert abs(by[(1.0, 8)]["mean_asr"] - 0.2) < 1e-9
    assert abs(by[(1.0, 8)]["mean_spectral_auc"] - 0.4) < 1e-9
    assert by[(6.0, 8)]["n_seeds"] == 1


def test_infeasible_rows_are_excluded_and_counted():
    rows = [_row(42, 2.0, 16, 0.5, 0.1, 0.4, 0.6), _row(123, 2.0, 16, 0.5, 0.1, 0.4, 0.6, feasible=False)]
    table = mod.aggregate(rows)
    assert table[0]["n_seeds"] == 1 and table[0]["n_infeasible"] == 1


def test_monotonicity_summary_uses_spearman_over_sigma():
    rows = [_row(42, s, 8, asr, asr * 0.9, asr, 0.5 + asr / 2)
            for s, asr in [(1.0, 0.1), (2.0, 0.3), (3.0, 0.5), (4.0, 0.7), (5.0, 0.9), (6.0, 1.0)]]
    summ = mod.monotonicity(mod.aggregate(rows), cost=8)
    assert summ["asr_rho"] > 0.99 and summ["spectral_auc_rho"] > 0.99
