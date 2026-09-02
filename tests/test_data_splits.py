"""Task 1: leak-free temporal split + train-only standardisation."""
import numpy as np
import pandas as pd
import pytest

from src.data import temporal_split, fit_standardiser, apply_standardiser


def _synthetic(n=1000, d=6, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(0, 1, size=(n, d)), columns=[f"f{i}" for i in range(d)])
    # imbalanced label like CTU (~2% minority), in temporal (row) order
    y = pd.Series((rng.random(n) < 0.02).astype(int), name="is_botnet")
    return x, y


def test_train_val_are_temporally_before_test():
    x, y = _synthetic()
    x_tr, y_tr, x_val, y_val, x_te, y_te = temporal_split(x, y, train_rows=800, val_frac=0.2, seed=1319)
    train_pool_idx = np.concatenate([x_tr.index.to_numpy(), x_val.index.to_numpy()])
    assert train_pool_idx.max() < x_te.index.to_numpy().min()   # no future -> past leakage
    assert len(x_te) == 200                                     # remainder is test
    assert len(x_tr) + len(x_val) == 800


def test_splits_are_disjoint_and_cover_all_rows():
    x, y = _synthetic()
    x_tr, _, x_val, _, x_te, _ = temporal_split(x, y, train_rows=800, val_frac=0.2, seed=1319)
    idx = np.concatenate([x_tr.index.to_numpy(), x_val.index.to_numpy(), x_te.index.to_numpy()])
    assert len(idx) == len(np.unique(idx)) == len(x)            # partition, no overlap


def test_val_is_stratified():
    x, y = _synthetic()
    _, y_tr, _, y_val, _, _ = temporal_split(x, y, train_rows=800, val_frac=0.2, seed=1319)
    pool_rate = (y_tr.sum() + y_val.sum()) / (len(y_tr) + len(y_val))
    val_rate = y_val.mean()
    assert abs(val_rate - pool_rate) < 0.02                     # class balance preserved in val


def test_standardiser_is_fit_on_train_only():
    x, y = _synthetic()
    x_tr, _, _, _, x_te, _ = temporal_split(x, y, train_rows=800, val_frac=0.2, seed=1319)
    scaler = fit_standardiser(x_tr)
    # scaler statistics come from the training rows only
    assert np.allclose(scaler.mean_, x_tr.to_numpy().mean(axis=0))
    # applying to train yields ~zero mean / unit std; test does NOT (it is transformed by train stats)
    z_tr = apply_standardiser(scaler, x_tr)
    assert np.allclose(z_tr.mean(axis=0), 0.0, atol=1e-6)
    z_te = apply_standardiser(scaler, x_te)
    assert z_te.shape == (len(x_te), x_te.shape[1])             # no refit, just transform


def test_downsample_benign_reaches_target_ratio():
    from src.data import downsample_benign

    rng = np.random.default_rng(0)
    n_benign, n_bot = 10_000, 200   # ~98/2, matching real CTU imbalance
    x = pd.DataFrame(rng.normal(size=(n_benign + n_bot, 4)), columns=list("abcd"))
    y = pd.Series([0] * n_benign + [1] * n_bot)

    x_ds, y_ds = downsample_benign(x, y, target_ratio=0.5, seed=42)

    n_pos = int((y_ds == 1).sum())
    n_neg = int((y_ds == 0).sum())
    assert n_pos == n_bot                          # every minority row kept
    assert abs(n_neg - n_pos) <= 1                  # ~50/50 (integer rounding only)
    assert len(x_ds) == len(y_ds)
    assert set(y_ds.unique()) == {0, 1}


def test_downsample_benign_is_deterministic_and_leaves_input_unmodified():
    from src.data import downsample_benign

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(1000, 3)), columns=list("abc"))
    y = pd.Series((rng.random(1000) < 0.05).astype(int))
    x_orig_shape, y_orig_sum = x.shape, int(y.sum())

    x1, y1 = downsample_benign(x, y, target_ratio=0.5, seed=7)
    x2, y2 = downsample_benign(x, y, target_ratio=0.5, seed=7)

    assert x.shape == x_orig_shape and int(y.sum()) == y_orig_sum   # caller's frames untouched
    assert list(y1) == list(y2)                                    # same seed -> same rows


@pytest.mark.parametrize("bad_ratio", [0.0, 1.0, 1.5])
def test_downsample_benign_rejects_invalid_target_ratio(bad_ratio):
    from src.data import downsample_benign

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(1000, 3)), columns=list("abc"))
    y = pd.Series((rng.random(1000) < 0.05).astype(int))

    with pytest.raises(ValueError):
        downsample_benign(x, y, target_ratio=bad_ratio, seed=42)


def test_downsample_benign_rejects_no_minority_rows():
    from src.data import downsample_benign

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
    y = pd.Series([0] * 500)   # all-zero labels -> no minority (label==1) rows

    with pytest.raises(ValueError):
        downsample_benign(x, y, target_ratio=0.5, seed=42)


def test_downsample_benign_caps_at_available_majority_rows():
    from src.data import downsample_benign

    # n_minority=100, majority pool only 50 rows; target_ratio=0.1 wants majority=900,
    # far more than exist -- the min(n_majority_target, len(majority_idx)) cap should
    # return ALL majority rows rather than raising or over-sampling.
    n_minority, n_majority = 100, 50
    x = pd.DataFrame(
        np.random.default_rng(0).normal(size=(n_minority + n_majority, 3)),
        columns=list("abc"),
    )
    y = pd.Series([1] * n_minority + [0] * n_majority)

    x_ds, y_ds = downsample_benign(x, y, target_ratio=0.1, seed=42)

    assert int((y_ds == 1).sum()) == n_minority
    assert int((y_ds == 0).sum()) == n_majority   # capped: all majority rows kept
    assert len(x_ds) == n_minority + n_majority
