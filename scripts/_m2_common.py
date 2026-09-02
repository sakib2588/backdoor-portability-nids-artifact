"""Shared setup for the M2/M3 orchestration scripts (02/03/04/05).

A `--smoke` config runs a small, fast slice (1 seed, subsample, reduced grid) to validate the
pipeline end-to-end before the full multi-hour run. The full config is the real experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import (
    load_ctu, build_ctu_constraints, temporal_split, fit_standardiser, apply_standardiser,
)
from src.tb_vendor.constraints_numeric import check_constraints


def get_config(smoke: bool) -> dict:
    if smoke:
        return dict(smoke=True, seeds=[42], rates=(0.02, 0.05), costs=(2, 8),
                    mlp_epochs=10, nc_steps=100, subsample=15000, nc_sample=200,
                    control_rate=0.05, control_cost=8,
                    nc_boundary_rates=(0.02, 0.05), nc_boundary_costs=(2, 8),
                    h3_rate=0.05, h3_cost=8, h3_target_ratio=0.5)
    return dict(smoke=False, seeds=list(config.SEEDS), rates=config.POISON_RATES,
                costs=config.TRIGGER_COSTS, mlp_epochs=20, nc_steps=300, subsample=None,
                nc_sample=400, control_rate=0.05, control_cost=8,
                nc_boundary_rates=(0.005, 0.05), nc_boundary_costs=(4, 8, 16),
                h3_rate=0.05, h3_cost=8, h3_target_ratio=0.5)


def apply_overrides(cfg: dict, argv) -> dict:
    """CLI overrides. `--seeds 42,123` restricts to those seeds at the SAME (full or smoke) config --
    used for a full-scale single-seed calibration/bug-check run before the full sweep."""
    if "--seeds" in argv:
        cfg["seeds"] = [int(s) for s in argv[argv.index("--seeds") + 1].split(",")]
    return cfg


def load_setup(cfg: dict) -> dict:
    """Load CTU, take an optional feasible subsample (temporal order preserved), temporal-split, and
    fit the standardiser on train only. Returns raw train/test frames plus the scaler, constraints,
    and both raw and standardised per-feature bounds."""
    x, y, features, metadata = load_ctu()
    constraints = build_ctu_constraints(features)

    if cfg["subsample"]:
        # Stride-sample evenly across the whole timeline (NOT the first N rows -- CTU's early period
        # is nearly all benign, which would starve train of botnet). Order is preserved, so the split
        # stays leak-free, and the ~2% botnet rate is retained.
        feasible = check_constraints(x.to_numpy(dtype=float), constraints, features)
        fidx = np.where(feasible)[0]
        stride = max(1, len(fidx) // cfg["subsample"])
        keep = fidx[::stride][: cfg["subsample"]]
        x = x.iloc[keep].reset_index(drop=True)
        y = y.iloc[keep].reset_index(drop=True)
        train_rows = int(len(x) * (143046 / 198128))   # keep the canonical train fraction
    else:
        train_rows = config.TRAIN_ROWS

    x_tr, y_tr, x_val, y_val, x_te, y_te = temporal_split(
        x, y, train_rows=train_rows, val_frac=config.VAL_FRAC, seed=config.VAL_SPLIT_SEED)
    scaler = fit_standardiser(x_tr)

    meta_idx = metadata.set_index("feature")
    lo = meta_idx.loc[features, "min"].to_numpy(dtype=float)
    hi = meta_idx.loc[features, "max"].to_numpy(dtype=float)

    x_tr_std = apply_standardiser(scaler, x_tr)
    std_lo = x_tr_std.min(axis=0)
    std_hi = x_tr_std.max(axis=0)

    return dict(
        features=features, constraints=constraints, bounds=(lo, hi),
        std_bounds=(std_lo, std_hi),
        x_tr_raw=x_tr.to_numpy(dtype=float), y_tr=y_tr.to_numpy(),
        x_val_raw=x_val.to_numpy(dtype=float), y_val=y_val.to_numpy(),
        x_te_raw=x_te.to_numpy(dtype=float), y_te=y_te.to_numpy(),
        scaler=scaler,
        train_balance=dict(n=len(y_tr), botnet=int(y_tr.sum())),
        val_balance=dict(n=len(y_val), botnet=int(y_val.sum())),
        test_balance=dict(n=len(y_te), botnet=int(y_te.sum())),
    )
