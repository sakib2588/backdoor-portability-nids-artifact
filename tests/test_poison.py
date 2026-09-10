"""Task 5: clean-label poisoning + sweep skeleton."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.constraints_secondary import (
    audit_constraints,
    build_secondary_constraints,
    project_secondary_to_feasible,
)
from src.data import fit_standardiser
from src.poison import (
    poison_secondary_trainset,
    poison_secondary_trainset_raw,
    poison_trainset_cleanlabel,
    sweep_grid,
)
from src.trigger import build_secondary_trigger


def _data(seed=0, n=5000, d=12, minority=0.02):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < minority).astype(int)
    x = rng.normal(0, 1, size=(n, d))
    return x, y


def _scaler(x):
    import pandas as pd
    return fit_standardiser(pd.DataFrame(x))


def _trigger(d):
    # a plain violating trigger (project=False) -- feasibility is not what this task tests
    return {"indices": [0, 1, 2], "values": [9.0, 9.0, 9.0], "project": False}


def test_exact_poison_count_whole_set_denominator():
    x, y = _data()
    for rate in config.POISON_RATES:
        _, _, idx = poison_trainset_cleanlabel(x, y, _trigger(x.shape[1]), rate, _scaler(x), seed=0)
        assert len(idx) == round(rate * len(x))          # whole-set denominator


def test_clean_label_invariant():
    x, y = _data()
    _, y_p, idx = poison_trainset_cleanlabel(x, y, _trigger(x.shape[1]), 0.05, _scaler(x), seed=0)
    assert (y[idx] == config.ATTACK_TARGET).all()        # only benign rows were chosen
    assert (y_p[idx] == config.ATTACK_TARGET).all()      # and their labels were NOT flipped
    assert np.array_equal(y, y_p)                        # no label changed anywhere


def test_poison_selection_is_reproducible():
    x, y = _data()
    _, _, a = poison_trainset_cleanlabel(x, y, _trigger(x.shape[1]), 0.05, _scaler(x), seed=7)
    _, _, b = poison_trainset_cleanlabel(x, y, _trigger(x.shape[1]), 0.05, _scaler(x), seed=7)
    assert np.array_equal(a, b)


def test_exclude_none_reproduces_the_committed_draw():
    """`exclude_idx=None` must leave the candidate pool -- and so the draw -- bit-identical.

    Guards every committed number in results/ against the Task 7B signature change.
    """
    x, y = _data()
    scaler = _scaler(x)
    _, _, without = poison_trainset_cleanlabel(x, y, _trigger(x.shape[1]), 0.05, scaler, seed=7)
    _, _, explicit_none = poison_trainset_cleanlabel(
        x, y, _trigger(x.shape[1]), 0.05, scaler, seed=7, exclude_idx=None)
    assert np.array_equal(without, explicit_none)


def test_excluded_rows_are_never_poisoned():
    x, y = _data()
    benign = np.where(y == config.ATTACK_TARGET)[0]
    excluded = benign[:1000]
    _, _, idx = poison_trainset_cleanlabel(
        x, y, _trigger(x.shape[1]), 0.05, _scaler(x), seed=7, exclude_idx=excluded)
    assert np.intersect1d(idx, excluded).size == 0
    assert len(idx) == round(0.05 * len(x))   # the budget is still met from the remaining pool


def test_exclusion_that_starves_the_pool_raises():
    x, y = _data()
    benign = np.where(y == config.ATTACK_TARGET)[0]
    with np.testing.assert_raises(ValueError):
        poison_trainset_cleanlabel(
            x, y, _trigger(x.shape[1]), 0.05, _scaler(x), seed=7, exclude_idx=benign[:-10])


def test_trigger_actually_stamped():
    x, y = _data()
    scaler = _scaler(x)
    trig = _trigger(x.shape[1])
    x_std, _, idx = poison_trainset_cleanlabel(x, y, trig, 0.05, scaler, seed=0)
    # de-standardise the poisoned rows and confirm the trigger features carry the watermark
    x_raw = scaler.inverse_transform(x_std)
    for i, v in zip(trig["indices"], trig["values"]):
        assert np.allclose(x_raw[idx, i], v)


def test_sweep_grid_size():
    grid = sweep_grid()
    assert len(grid) == len(config.POISON_RATES) * len(config.TRIGGER_COSTS)
    assert (0.05, 4) in grid


# --- Task 4: secondary (UNSW-NB15) poisoning -------------------------------------------------------
# Moved here from tests/test_trigger.py (2026-07-31, code-quality review) to match this file's own
# convention: poison_trainset_cleanlabel's tests live in test_poison.py, so
# poison_secondary_trainset(_raw)'s do too, now that both live in src/poison.py. Synthetic fixture
# data only (mirrors tests/test_constraints_secondary.py's fixture) -- no network access, no download,
# no dependency on the real UNSW-NB15 load.

SECONDARY_FEATURE_NAMES = (
    "dur", "tcprtt", "synack", "ackdat", "sttl", "dttl",
    "smeansz", "Spkts", "sbytes", "dmeansz", "Dpkts", "dbytes",
)

# tcprtt's own projectable repair overwrites it (tcprtt := synack + ackdat) -- excluded from trigger
# eligibility, the secondary-dataset analogue of CTU strategy A's equality-family exclusion, but
# supplied by the CALLER (not derived by build_secondary_trigger itself) per the Task 4 requirement
# that the function "must not assume CTU equality-feature exclusions".
_SECONDARY_EXCLUDED = {"tcprtt"}


def _secondary_synthetic(seed: int = 0, n: int = 4000):
    rng = np.random.default_rng(seed)
    col = {name: i for i, name in enumerate(SECONDARY_FEATURE_NAMES)}
    x = np.zeros((n, len(SECONDARY_FEATURE_NAMES)))
    x[:, col["dur"]] = rng.uniform(0.1, 5.0, n)
    x[:, col["synack"]] = rng.uniform(0.01, 0.1, n)
    x[:, col["ackdat"]] = rng.uniform(0.01, 0.1, n)
    x[:, col["tcprtt"]] = x[:, col["synack"]] + x[:, col["ackdat"]]
    x[:, col["sttl"]] = rng.uniform(30, 90, n)
    x[:, col["dttl"]] = rng.uniform(30, 90, n)
    spkts = rng.integers(1, 20, n).astype(float)
    dpkts = rng.integers(1, 20, n).astype(float)
    x[:, col["Spkts"]] = spkts
    x[:, col["Dpkts"]] = dpkts
    x[:, col["sbytes"]] = spkts * rng.uniform(40, 100, n)
    x[:, col["dbytes"]] = dpkts * rng.uniform(40, 100, n)
    x[:, col["smeansz"]] = x[:, col["sbytes"]] / spkts
    x[:, col["dmeansz"]] = x[:, col["dbytes"]] / dpkts
    y = (rng.random(n) < 0.05).astype(int)          # ~5% attack (label 1); target class 0 is benign
    return x, y


@pytest.fixture
def secondary_env():
    x, y = _secondary_synthetic()
    constraints = build_secondary_constraints(SECONDARY_FEATURE_NAMES)
    lo = np.minimum(x.min(axis=0), 0.0)
    hi = np.maximum(x.max(axis=0), 0.0) + 1e6            # generous: box-clip never fights the repairs
    bounds = (lo, hi)
    ranked = list(range(len(SECONDARY_FEATURE_NAMES)))[::-1]     # arbitrary but deterministic
    eligible_indices = [i for i, n in enumerate(SECONDARY_FEATURE_NAMES) if n not in _SECONDARY_EXCLUDED]
    scaler = fit_standardiser(pd.DataFrame(x, columns=SECONDARY_FEATURE_NAMES))
    return dict(x=x, y=y, constraints=constraints, bounds=bounds, ranked=ranked,
               eligible_indices=eligible_indices, features=SECONDARY_FEATURE_NAMES, scaler=scaler)


def _realizable_project_fn(env):
    return lambda rows: project_secondary_to_feasible(rows, env["constraints"], env["bounds"])


def test_secondary_poison_indices_are_only_target_class_rows(secondary_env):
    e = secondary_env
    trig = build_secondary_trigger(e["ranked"], 4, e["x"], e["features"], e["bounds"],
                                   _realizable_project_fn(e), e["eligible_indices"])
    _, y_poisoned, poison_idx = poison_secondary_trainset(
        e["x"], e["y"], trig, 0.05, e["scaler"], target=config.ATTACK_TARGET, seed=0)
    assert np.all(y_poisoned[poison_idx] == config.ATTACK_TARGET)


def test_secondary_realizable_poison_rows_pass_manifest(secondary_env):
    e = secondary_env
    trig = build_secondary_trigger(e["ranked"], 4, e["x"], e["features"], e["bounds"],
                                   _realizable_project_fn(e), e["eligible_indices"])
    x_poisoned_raw, _, poison_idx = poison_secondary_trainset_raw(
        e["x"], e["y"], trig, 0.05, target=config.ATTACK_TARGET, seed=0)
    audit = audit_constraints(x_poisoned_raw[poison_idx], e["constraints"])
    assert audit["all_projectable_valid"] is True


# --- STRIP's clean reference pool must be unpoisonable on the secondary dataset too ------------
# Added 2026-09-06 when STRIP's blend/calibration pool moved off the test partition. The primary
# path already had these guarantees; the secondary one did not.

def test_secondary_exclude_none_reproduces_the_committed_draw():
    """`exclude_idx=None` must leave the secondary candidate pool, and so the draw, bit-identical."""
    x, y = _data()
    a = poison_secondary_trainset_raw(x, y, _trigger(x.shape[1]), 0.05, seed=7)[2]
    b = poison_secondary_trainset_raw(
        x, y, _trigger(x.shape[1]), 0.05, seed=7, exclude_idx=None)[2]
    assert np.array_equal(a, b)


def test_secondary_excluded_rows_are_never_poisoned():
    x, y = _data()
    target_rows = np.where(y == config.ATTACK_TARGET)[0]
    excluded = target_rows[:1000]
    idx = poison_secondary_trainset_raw(
        x, y, _trigger(x.shape[1]), 0.05, seed=7, exclude_idx=excluded)[2]
    assert np.intersect1d(idx, excluded).size == 0
