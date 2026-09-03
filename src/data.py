"""CTU-13 Neris loading and the 360-constraint set.

We do NOT depend on the TabularBench package (uninstallable torch pin). Instead we
pull the two CSVs it publishes on HuggingFace and rebuild the relation-constraint
set locally from the vendored constraint DSL. The `build_ctu_constraints` function
is a faithful port of TabularBench's `get_relation_constraints` for ctu_13_neris
(source commit bfb7541); the resulting count must be exactly 360.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from src import config
from src.tb_vendor.relation_constraint import (
    BaseRelationConstraint,
    Constant as Co,
    Feature as Fe,
    SafeDivision,
    Value,
)

HF_REPO_ID = "serval-uni-lu/tabularbench"
# NOTE: the TabularBench source hardcodes "ctu_13_neris/..." but the published
# HuggingFace repo stores these under "ctu_13/..." (verified 2026-07-13).
HF_DATA_FILE = "ctu_13/ctu_13_neris.csv"
HF_METADATA_FILE = "ctu_13/ctu_13_neris_metadata.csv"

# The classification target. It is listed among the metadata `feature` rows in the
# published CSV, so we name it explicitly rather than inferring it as a leftover.
TARGET = "is_botnet"

# Port families in TabularBench's ctu_13_neris constraint builder (order matters
# only for readability; the count does not depend on it).
_PORTS = ["1", "3", "8", "10", "21", "22", "25", "53", "80", "110", "123", "135",
          "138", "161", "443", "445", "993", "OTHER"]


def download_ctu() -> Tuple[str, str]:
    """Fetch the CTU data + metadata CSVs from HuggingFace into data/raw. Cached."""
    from huggingface_hub import hf_hub_download

    config.ensure_dirs()
    data_path = hf_hub_download(
        repo_id=HF_REPO_ID, repo_type="dataset", filename=HF_DATA_FILE,
        local_dir=str(config.DATA_RAW),
    )
    meta_path = hf_hub_download(
        repo_id=HF_REPO_ID, repo_type="dataset", filename=HF_METADATA_FILE,
        local_dir=str(config.DATA_RAW),
    )
    return data_path, meta_path


def load_ctu() -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """Return (X, y, feature_names, metadata).

    Features are exactly the rows of the metadata `feature` column; the target is
    the single data column not listed as a feature (TabularBench's convention).
    """
    data_path, meta_path = download_ctu()
    df = pd.read_csv(data_path)
    metadata = pd.read_csv(meta_path)

    if TARGET not in df.columns:
        raise ValueError(f"target {TARGET!r} not in data columns")
    features = [c for c in metadata["feature"].tolist() if c != TARGET]

    x = df[features].copy()
    y = df[TARGET].copy()
    return x, y, features, metadata


def temporal_split(
    x: pd.DataFrame,
    y: pd.Series,
    train_rows: int = config.TRAIN_ROWS,
    val_frac: float = config.VAL_FRAC,
    seed: int = config.VAL_SPLIT_SEED,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Leak-free split matching TabularBench's Ctu13Splitter.

    The earliest `train_rows` rows (temporal order = row order) form the train+val pool; the remainder
    is the test set (strictly later, so no future->past leakage). Validation is a stratified `val_frac`
    slice carved from the train pool (a random slice within the training PERIOD, which is acceptable;
    the leak-free guarantee is train+val < test in time). Returns raw frames/series; standardisation
    is fit later on train only (`fit_standardiser`).
    """
    from sklearn.model_selection import train_test_split

    pool_x = x.iloc[:train_rows]
    pool_y = y.iloc[:train_rows]
    test_x = x.iloc[train_rows:]
    test_y = y.iloc[train_rows:]

    tr_idx, val_idx = train_test_split(
        np.arange(len(pool_x)), test_size=val_frac, random_state=seed, stratify=pool_y.to_numpy()
    )
    tr_idx.sort()
    val_idx.sort()
    x_tr, y_tr = pool_x.iloc[tr_idx], pool_y.iloc[tr_idx]
    x_val, y_val = pool_x.iloc[val_idx], pool_y.iloc[val_idx]
    return x_tr, y_tr, x_val, y_val, test_x, test_y


def fit_standardiser(x_tr_raw):
    """Fit a StandardScaler on the TRAINING rows only (never val/test).

    Accepts a DataFrame or a raw ndarray, the same widening `apply_standardiser` below already has.
    The NetFlow arm splits to arrays rather than frames, and forcing it through a DataFrame purely to
    reach this function would copy a 300 MB block for nothing.
    """
    from sklearn.preprocessing import StandardScaler

    arr = x_tr_raw.to_numpy() if isinstance(x_tr_raw, (pd.DataFrame, pd.Series)) else np.asarray(x_tr_raw)
    return StandardScaler().fit(arr)


def apply_standardiser(scaler, x_raw) -> np.ndarray:
    """Transform raw features to standardised space with an already-fit scaler (no refit)."""
    arr = x_raw.to_numpy() if isinstance(x_raw, (pd.DataFrame, pd.Series)) else np.asarray(x_raw)
    return scaler.transform(arr)


def inverse_standardise(x_std, scaler) -> np.ndarray:
    """Standardised features back to RAW units. Inverse of `apply_standardiser`.

    The CTU constraints are defined in raw units (raw MTU bytes, raw byte counts) while Neural
    Cleanse's inversion optimises in standardised space, so a reverse-engineered pattern has to come
    back to raw space before its feasibility can be audited (plan Task 7A).
    """
    arr = np.asarray(x_std)
    return scaler.inverse_transform(arr[None, :] if arr.ndim == 1 else arr)


def downsample_benign(
    x_raw: pd.DataFrame, y: pd.Series, target_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """Down-sample the majority (benign, label 0) class toward `target_ratio` minority share.

    H3 (imbalance confound): tests whether detector power on the tabular MLP recovers once
    training-time class imbalance is removed, isolating "imbalance drives the miss" from
    "tabular modality drives the miss." Every minority (label 1) row is kept; majority rows
    are subsampled without replacement to hit the target ratio. Order is NOT preserved (this
    is a training-set rebalancing, not a temporal split) but the caller's frames are never
    mutated -- returns fresh, reset-index frames.
    """
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    y_arr = y.to_numpy()
    minority_idx = np.where(y_arr == 1)[0]
    majority_idx = np.where(y_arr == 0)[0]
    n_minority = len(minority_idx)
    if n_minority == 0:
        raise ValueError("downsample_benign: no minority (label==1) rows present")

    n_majority_target = int(round(n_minority * (1 - target_ratio) / target_ratio))
    n_majority_target = min(n_majority_target, len(majority_idx))

    rng = np.random.default_rng(seed)
    kept_majority = rng.choice(majority_idx, size=n_majority_target, replace=False)
    keep = np.sort(np.concatenate([minority_idx, kept_majority]))

    return x_raw.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


def build_ctu_constraints(features: List[str]) -> List[BaseRelationConstraint]:
    """Faithful port of TabularBench's ctu_13_neris `get_relation_constraints`.

    Must return exactly 360 constraints: 2 byte-conservation equalities +
    34 packet-size (17 ports x 2 directions) + 324 min/max/sum
    (3 fields x 18 ports x 2 directions x 3 orderings).
    """
    def family(prefix: str) -> List[str]:
        return [f for f in features if f.startswith(prefix)]

    def sum_list(fs: List[str]) -> Value:
        out: Value = Fe(fs[0])
        for el in fs[1:]:
            out = out + Fe(el)
        return out

    def sum_family(prefix: str) -> Value:
        return sum_list(family(prefix))

    # 2 byte-conservation equalities (source, then destination).
    g1 = (sum_family("icmp_sum_s_") + sum_family("udp_sum_s_")
          + sum_family("tcp_sum_s_")) == (sum_family("bytes_in_sum_s_")
                                          + sum_family("bytes_out_sum_s_"))
    g2 = (sum_family("icmp_sum_d_") + sum_family("udp_sum_d_")
          + sum_family("tcp_sum_d_")) == (sum_family("bytes_in_sum_d_")
                                          + sum_family("bytes_out_sum_d_"))

    # Packet-size: mean bytes/packet <= 1500 MTU, per port (drop trailing OTHER).
    g_packet_size: List[BaseRelationConstraint] = []
    for e in ["s", "d"]:
        bytes_outs = family(f"bytes_out_sum_{e}_")[:-1]
        pkts_outs = family(f"pkts_out_sum_{e}_")[:-1]
        if len(bytes_outs) != len(pkts_outs):
            raise ValueError("len(bytes_out) != len(pkts_out)")
        for byte_out, pkts_out in zip(bytes_outs, pkts_outs):
            g_packet_size.append(
                SafeDivision(Fe(byte_out), Fe(pkts_out), Co(0.0)) <= Co(1500)
            )

    # min <= sum, min <= max, max <= sum, per (field, port, direction).
    g_min_max_sum: List[BaseRelationConstraint] = []
    for e_1 in ["bytes_out", "pkts_out", "duration"]:
        for port in _PORTS:
            for e_2 in ["d", "s"]:
                g_min_max_sum.extend([
                    Fe(f"{e_1}_max_{e_2}_{port}") <= Fe(f"{e_1}_sum_{e_2}_{port}"),
                    Fe(f"{e_1}_min_{e_2}_{port}") <= Fe(f"{e_1}_sum_{e_2}_{port}"),
                    Fe(f"{e_1}_min_{e_2}_{port}") <= Fe(f"{e_1}_max_{e_2}_{port}"),
                ])

    return [g1, g2] + g_packet_size + g_min_max_sum


def _m0_exit_gate() -> None:
    """M0 exit gate: load CTU, print shape + balance, assert n_constraints == 360."""
    x, y, features, _ = load_ctu()
    constraints = build_ctu_constraints(features)
    n = len(constraints)
    pos = int((y != y.mode()[0]).sum())  # minority (botnet) count, label-agnostic
    print(f"CTU-13 Neris: X shape = {x.shape}, n_features = {len(features)}")
    print(f"class balance: {len(y) - pos} majority / {pos} minority "
          f"({100 * pos / len(y):.2f}% minority)")
    print(f"n_constraints = {n} (expected {config.CTU_N_CONSTRAINTS_EXPECTED})")
    assert n == config.CTU_N_CONSTRAINTS_EXPECTED, (
        f"constraint count {n} != {config.CTU_N_CONSTRAINTS_EXPECTED}"
    )
    print("M0 exit gate PASSED.")


if __name__ == "__main__":
    _m0_exit_gate()
