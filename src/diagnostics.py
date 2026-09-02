"""Attack-mechanism probes: WHY an attack lands or does not, on a given victim.

Built for the LightGBM result (results/lgb_diagnostic.json): the sweep's pinned lgb ASR of 0.0467 is
19/407 botnet test flows -- the same row set in every configuration -- that the ensemble calls benign
whatever the trigger does, i.e. a false-negative floor with no attack effect in it. These probes are
what separate that reading from "the attack works a little". They are victim-agnostic: prediction
probes take label arrays, and only `leaf_overlap_fraction` assumes a tree ensemble's leaf-index
output (LightGBM's `predict(..., pred_leaf=True)`).
"""
from __future__ import annotations

import numpy as np


def prediction_change_count(pred_a: np.ndarray, pred_b: np.ndarray) -> int:
    """How many predictions differ between two label arrays over the same rows.

    The trigger's raw effect size on a victim: stamping that changes ~0 predictions is an inert
    trigger, whatever the resulting ASR happens to be.
    """
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    if a.shape != b.shape:
        raise ValueError(f"prediction arrays must align: {a.shape} vs {b.shape}")
    return int((a != b).sum())


def target_predicted_set(pred: np.ndarray, target: int) -> frozenset:
    """Row indices predicted as `target`, as a set for cross-configuration identity comparison.

    If this set is identical across seeds/rates/costs, the "successes" are a fixed property of the
    victim (its false-negative floor), not something the attack produced.
    """
    return frozenset(np.where(np.asarray(pred) == target)[0].tolist())


def leaf_overlap_fraction(poison_leaves: np.ndarray, test_leaves: np.ndarray) -> float:
    """Mean over trees of: fraction of triggered test rows landing in a leaf that some poisoned
    training row also reaches.

    A tree ensemble can only carry a backdoor where poisoned training rows and triggered test rows
    co-locate in the partition -- the poisoned rows' benign label only reaches a test row through a
    shared leaf. Low overlap means the trigger cannot transfer, however heavily the trees split on
    the trigger features. Both inputs are (n_rows, n_trees) leaf-index matrices from
    `booster.predict(x, pred_leaf=True)`.
    """
    p = np.asarray(poison_leaves)
    t = np.asarray(test_leaves)
    if p.ndim != 2 or t.ndim != 2:
        raise ValueError(f"leaf matrices must be 2-D (n_rows, n_trees): {p.shape}, {t.shape}")
    if p.shape[1] != t.shape[1]:
        raise ValueError(f"tree counts must match: {p.shape[1]} vs {t.shape[1]}")
    per_tree = np.empty(p.shape[1], dtype=np.float64)
    for tree in range(p.shape[1]):
        reached = set(np.unique(p[:, tree]).tolist())
        per_tree[tree] = np.mean([leaf in reached for leaf in t[:, tree]])
    return float(per_tree.mean())
