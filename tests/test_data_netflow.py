import numpy as np
import pytest

from src import config
from src.data_netflow import (
    NetFlowSplit, allocate, group_index, load_netflow_corpus, load_netflow_benign_share,
    split_netflow, validate_disjoint_split,
)

SMALL = 40_000


@pytest.fixture(scope="module")
def unsw_v2():
    return load_netflow_corpus("NF-UNSW-NB15-v2", n_rows=SMALL, seed=42)


def test_group_index_matches_the_population_table():
    """group_index raises if the parquet disagrees with config.NETFLOW_POPULATION, so reaching this
    assertion at all is the check. Re-assert here so the failure names the corpus."""
    idx = group_index()
    for corpus, (n_pop, b_pop) in config.NETFLOW_POPULATION.items():
        seen = np.asarray(idx["counts"][corpus], dtype=np.int64)
        assert int(seen.sum()) == n_pop, corpus
        assert int(seen[:, 0].sum()) == b_pop, corpus


@pytest.mark.parametrize("need", [0, 1, 7, 150, 1_000, 40_000])
def test_allocation_is_exact_and_within_availability(need):
    rng = np.random.default_rng(0)
    avail = rng.integers(0, 500, size=152)
    avail[3] = 0
    need = min(need, int(avail.sum()))
    alloc = allocate(avail, need)
    assert int(alloc.sum()) == need
    assert bool((alloc <= avail).all())
    assert bool((alloc[avail == 0] == 0).all())


def test_allocation_spreads_across_every_nonempty_group():
    avail = np.full(152, 100, dtype=np.int64)
    alloc = allocate(avail, 200)
    assert int((alloc > 0).sum()) == 152


def test_allocation_refuses_an_unfundable_request():
    with pytest.raises(ValueError):
        allocate([1, 2, 3], 7)


def test_sample_size_and_feature_columns(unsw_v2):
    df, meta = unsw_v2
    assert len(df) == SMALL
    for col in config.NETFLOW_EXCLUDED_COLUMNS:
        assert col not in meta["feature_columns"]
    assert meta["n_features"] == len(meta["feature_columns"]) > 30


def test_native_class_ratio_preserved(unsw_v2):
    df, meta = unsw_v2
    n, b = config.NETFLOW_POPULATION["NF-UNSW-NB15-v2"]
    expected_attack = 1.0 - b / n
    assert meta["population_attack_rate"] == pytest.approx(expected_attack)
    assert abs(df["Label"].mean() - expected_attack) < 0.005


def test_sampling_is_not_front_loaded(unsw_v2):
    """Every row group that holds this corpus must contribute. An implementation that stops once the
    quota fills biases the sample toward early groups."""
    _, meta = unsw_v2
    assert meta["row_groups_contributing"] == meta["row_groups_holding_corpus"]


def test_deterministic_under_seed():
    a, ma = load_netflow_corpus("NF-UNSW-NB15-v2", n_rows=20_000, seed=7)
    b, mb = load_netflow_corpus("NF-UNSW-NB15-v2", n_rows=20_000, seed=7)
    assert a.equals(b)
    assert ma["sha256_feature_block"] == mb["sha256_feature_block"]


def test_a_different_seed_draws_a_different_sample():
    _, ma = load_netflow_corpus("NF-UNSW-NB15-v2", n_rows=20_000, seed=7)
    _, mb = load_netflow_corpus("NF-UNSW-NB15-v2", n_rows=20_000, seed=8)
    assert ma["sha256_feature_block"] != mb["sha256_feature_block"]


def test_split_is_disjoint_and_sized(unsw_v2):
    df, meta = unsw_v2
    s = split_netflow(df, meta, seed=42)
    assert isinstance(s, NetFlowSplit)
    assert len(s.x_tr_raw) + len(s.x_te_raw) == len(df)
    assert abs(len(s.x_tr_raw) / len(df) - config.NETFLOW_TRAIN_FRAC) < 0.01
    assert len(np.intersect1d(s.train_idx, s.test_idx)) == 0
    validate_disjoint_split(s)


def test_split_preserves_the_class_ratio(unsw_v2):
    df, meta = unsw_v2
    s = split_netflow(df, meta, seed=42)
    assert abs(s.y_tr.mean() - s.y_te.mean()) < 0.005


def test_feature_block_is_finite(unsw_v2):
    df, meta = unsw_v2
    assert np.isfinite(df[meta["feature_columns"]].to_numpy()).all()


@pytest.mark.parametrize("share", [0.05, 0.60, 0.96])
def test_requested_benign_share_is_delivered(share):
    df, meta = load_netflow_benign_share(
        "NF-CSE-CIC-IDS2018-v2", benign_share=share, n_rows=SMALL, seed=42)
    assert len(df) == SMALL
    assert (df["Label"] == 0).mean() == pytest.approx(share, abs=0.002)
    assert meta["requested_benign_share"] == share
    assert meta["mode"] == "benign_share"


def test_impossible_benign_share_raises():
    # NF-UNSW-NB15-v2 has 95,053 attack rows; a 0.05 benign share at 1.2M needs 1.14M of them.
    with pytest.raises(ValueError, match="attack"):
        load_netflow_benign_share("NF-UNSW-NB15-v2", benign_share=0.05,
                                  n_rows=config.NETFLOW_SAMPLE_ROWS, seed=42)


def test_impossible_benign_share_raises_on_the_benign_side():
    # NF-BoT-IoT-v2 has 135,037 benign rows; a 0.96 share at 1.2M needs 1.152M of them.
    with pytest.raises(ValueError, match="benign"):
        load_netflow_benign_share("NF-BoT-IoT-v2", benign_share=0.96,
                                  n_rows=config.NETFLOW_SAMPLE_ROWS, seed=42)


# --------------------------------------------------------------------------------------------
# Regression: the bytes-per-second overflow that silently produced degenerate victims
# --------------------------------------------------------------------------------------------
import pandas as pd

from src.data_netflow import _assert_standardisable
from src.data import fit_standardiser


def test_rate_columns_are_capped_at_line_rate(unsw_v2):
    df, meta = unsw_v2
    for name in config.NETFLOW_RATE_CAPPED_COLUMNS:
        assert df[name].max() <= config.NETFLOW_RATE_CAP_BYTES_PER_S, name
    assert meta["rate_cap_bytes_per_s"] == config.NETFLOW_RATE_CAP_BYTES_PER_S
    assert meta["rate_capped_values"] >= 0


def test_a_standardiser_fit_on_a_sample_has_finite_variance(unsw_v2):
    """The bug this pins: one column of 1e287 values overflows float64 when squared, StandardScaler
    records a NaN variance, every row of that column becomes NaN, and the victim collapses to a
    constant prediction scoring exactly the benign share. Nothing downstream reveals it."""
    df, meta = unsw_v2
    scaler = fit_standardiser(df[meta["feature_columns"]].to_numpy())
    assert np.isfinite(scaler.var_).all()
    assert np.isfinite(scaler.scale_).all()
    std = scaler.transform(df[meta["feature_columns"]].to_numpy())
    assert np.isfinite(std).all()


def test_overflow_guard_names_the_offending_column():
    block = pd.DataFrame({"ok": [1.0, 2.0], "explodes": [1e300, 2e300]})
    with pytest.raises(RuntimeError, match="explodes"):
        _assert_standardisable(block, ["ok", "explodes"])


def test_overflow_guard_passes_on_ordinary_magnitudes():
    block = pd.DataFrame({"a": [1.0, 2.0], "b": [1e10, 2e10]})
    _assert_standardisable(block, ["a", "b"])
