"""Severi et al.'s model-agnostic clean-label backdoor mitigation, ported for this study.

Reference: Severi, Boboila, Holodnak, Kratkiewicz, Izmailov, De Lucia and Oprea, "Model-Agnostic
Clean-Label Backdoor Mitigation in Cybersecurity Environments", MILCOM 2025, arXiv:2407.08159,
Algorithm 1. Pre-registered for this paper in
notes/20260903-decision-density-mitigation-preregistration.md before any of it was written.

Why it is here. Every detector in this manuscript is a defense built for image classifiers and
carried across. Without one defense built for tabular security data, run under the same attack, a
reader cannot separate "vision-built detectors fail here" from "this attack defeats any detector".
This is that comparator.

The pipeline, in the paper's own four stages:
  1. reduce to the |F| most important features, by an entropy-criterion tree
  2. cluster the target-class rows in that subspace, by density
  3. score clusters iteratively: start the clean set at the largest cluster plus every non-target
     row, train a surrogate, absorb the lowest-loss window of remaining clusters, retrain, repeat
     until `stop_frac` of clusters are absorbed, recording each cluster's loss delta on absorption
  4. flag, either by "never absorbed" or by an anomalous loss delta, then filter and retrain

Two deviations from the paper, both declared. Density clustering uses HDBSCAN rather than OPTICS,
because scikit-learn's OPTICS did not return within ten minutes on 140,000 rows in four dimensions
while HDBSCAN returned in 65 s and recovered a planted cluster exactly; both are variable-density
clusterers of the same lineage. And because the method flags clusters rather than rows, a per-sample
score is needed for the ranking AUC and the budget-free rule every other detector in this paper is
measured under: a row inherits its cluster's absorption rank. Ties inside a cluster are inherent to
a cluster-level method and are reported as such.

The learner is injected as `surrogate_fn(x, y, seed) -> model with predict_proba`, so this module
imports no estimator and the driver decides what the defender trains.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Sequence

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.tree import DecisionTreeClassifier

NOISE_LABEL = -1
_EPS = 1e-7


def select_features(x, y, n_features: int = 4, seed: int = 0) -> np.ndarray:
    """Indices of the `n_features` most important columns, by an entropy-criterion tree.

    Stage 1. Clean-label attacks in this domain place the trigger in features the model already
    relies on, so reducing to the most important ones is where the poison should still be visible
    and where distances are not diluted by hundreds of irrelevant dimensions.
    """
    x = np.asarray(x, dtype=np.float64)
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=seed)
    tree.fit(x, np.asarray(y))
    return np.sort(np.argsort(-tree.feature_importances_)[:n_features])


def cluster_target_class(x_sub, min_cluster_size_frac: float = 0.0025,
                         min_samples: int = 50) -> np.ndarray:
    """Density-cluster the target-class rows in the reduced subspace.

    `min_cluster_size_frac` is a fraction of the rows scored, fixed in the pre-registration rather
    than tuned. Points HDBSCAN cannot place keep label -1 and are treated downstream as one cluster,
    so every scored row carries a decision.
    """
    x_sub = np.asarray(x_sub, dtype=np.float64)
    mcs = max(2, int(round(min_cluster_size_frac * len(x_sub))))
    return HDBSCAN(min_cluster_size=mcs, min_samples=min_samples,
                   n_jobs=-1).fit_predict(x_sub)


def _mean_log_loss(model, x, y) -> float:
    p = np.clip(np.asarray(model.predict_proba(x))[:, 1], _EPS, 1.0 - _EPS)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def iterative_scoring(x, y, labels, target: int,
                      surrogate_fn: Callable, window_frac: float = 0.05,
                      stop_frac: float = 0.80, seed: int = 0) -> Dict:
    """Stage 3. Grow a clean set cluster by cluster, cheapest loss first, recording loss deltas.

    Returns the absorption rank per cluster (higher is more suspicious, with never-absorbed clusters
    ranked above every absorbed one), the loss delta each cluster produced when it entered, those
    deltas as z-scores, and the per-sample score every row inherits from its cluster.

    The intuition the paper rests on: a cluster of clean rows resembles the data the surrogate was
    trained on and scores a low loss, so it is absorbed early. A cluster carrying the trigger does
    not, so it is absorbed late or never, and its arrival moves the model's loss anomalously.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    labels = np.asarray(labels)
    target_pos = np.flatnonzero(y == target)
    if len(target_pos) != len(labels):
        raise ValueError(f"labels cover {len(labels)} rows but {len(target_pos)} are target-class")

    clusters = list(np.unique(labels))
    sizes = {c: int((labels == c).sum()) for c in clusters}
    largest = max(sizes, key=lambda c: sizes[c])

    non_target = np.flatnonzero(y != target)
    in_clean = {largest}
    remaining = [c for c in clusters if c != largest]
    # The largest cluster seeds the clean set and is never scored, so it carries rank 0: the least
    # suspicious value. Without this every row in it is missing from the per-sample score.
    absorbed_order_seed = {largest: 0}
    step = max(1, int(math.ceil(window_frac * len(clusters))))
    # Floor, not round, and never all of them. At 700 clusters the distinction is invisible; at 2 it
    # is the difference between a rule and a vacuous one, because round(0.8 x 2) = 2 absorbs
    # everything and leaves nothing for the fixed rule to flag. The paper's own setting is a
    # percentage of a large cluster count, so this preserves its intent at small counts rather than
    # silently returning an empty flag set.
    stop_at = min(len(clusters) - 1, max(1, int(stop_frac * len(clusters)))) if len(clusters) > 1 else 1

    def clean_rows(members: set) -> np.ndarray:
        sel = np.isin(labels, list(members))
        return np.concatenate([target_pos[sel], non_target])

    absorbed_order: Dict = dict(absorbed_order_seed)
    deltas: Dict = {}
    prev_loss: Dict = {}
    n_iter = 0
    rank = 0

    while remaining and len(in_clean) < stop_at:
        idx = clean_rows(in_clean)
        model = surrogate_fn(x[idx], y[idx], seed)
        losses = {}
        for c in remaining:
            rows = target_pos[labels == c]
            losses[c] = _mean_log_loss(model, x[rows], y[rows])
        take = sorted(remaining, key=lambda c: losses[c])[:step]
        for c in take:
            rank += 1
            absorbed_order[c] = rank
            # delta between the loss this cluster showed before joining and after the clean set grew
            deltas[c] = float(losses[c] - prev_loss.get(c, losses[c]))
            in_clean.add(c)
            remaining.remove(c)
        prev_loss = losses
        n_iter += 1

    # Never-absorbed clusters sit above every absorbed one. Among themselves they are ordered by
    # the last loss the surrogate assigned them, which is the same quantity the absorption order
    # used, so the ordering is the method's own rather than an invented tie-break. Leaving them tied
    # caps the ranking metric at whatever fraction of the unabsorbed set is actually poisoned.
    for c in sorted(remaining, key=lambda k: prev_loss.get(k, 0.0)):
        rank += 1
        absorbed_order[c] = rank

    d = np.array([deltas.get(c, 0.0) for c in clusters], dtype=float)
    mu, sd = float(d.mean()), float(d.std())
    z = {c: (0.0 if sd == 0 else float((deltas.get(c, 0.0) - mu) / sd)) for c in clusters}

    per_sample = np.array([absorbed_order[c] for c in labels], dtype=float)
    return dict(
        absorption_rank_by_cluster={int(k): int(v) for k, v in absorbed_order.items()},
        loss_delta_by_cluster={int(k): float(v) for k, v in deltas.items()},
        z_by_cluster={int(k): float(v) for k, v in z.items()},
        absorbed_clusters=[int(c) for c in clusters if c in absorbed_order and c not in remaining],
        unabsorbed_clusters=[int(c) for c in remaining],
        n_clusters=len(clusters), n_iterations=n_iter, stop_at=int(stop_at),
        few_clusters=bool(len(clusters) < 5),
        cluster_sizes={int(k): int(v) for k, v in sizes.items()},
        per_sample_score=per_sample,
    )


def flag_fixed_threshold(scoring: Dict, labels) -> np.ndarray:
    """The published fixed rule: clusters not absorbed by the stopping threshold are suspicious."""
    unabsorbed = set(scoring["unabsorbed_clusters"])
    return np.isin(np.asarray(labels), list(unabsorbed)) if unabsorbed \
        else np.zeros(len(labels), dtype=bool)


def flag_loss_delta(scoring: Dict, labels, z_t: float = 2.0) -> np.ndarray:
    """The published loss-delta rule: flag clusters whose absorption moved the loss anomalously.

    The paper cuts at z <= -z_t, the side on which a cluster's arrival drops the loss more than the
    spread of deltas explains.
    """
    z = scoring["z_by_cluster"]
    bad = [c for c, v in z.items() if v <= -abs(z_t)]
    return np.isin(np.asarray(labels), bad) if bad else np.zeros(len(labels), dtype=bool)


def filter_mask(flags_target_class, target_positions, n_train: int) -> np.ndarray:
    """Training-set keep mask that drops exactly the flagged target-class rows and nothing else."""
    keep = np.ones(int(n_train), dtype=bool)
    flagged = np.asarray(target_positions)[np.asarray(flags_target_class, dtype=bool)]
    keep[flagged] = False
    return keep


ISOLATION_SHARE = 0.9     # the planted cluster holds at least this share of all poison rows
ISOLATION_PURITY = 0.5    # and is at least half poison


def cluster_composition(labels, is_poison, unabsorbed) -> Dict:
    """Per-cluster poison composition, and whether the clusterer isolated the poison at all.

    This is the diagnostic the gate-failure note left unmeasured. `labels` are HDBSCAN labels
    over the target-class rows (noise is NOISE_LABEL), `is_poison` is aligned with them, and
    `unabsorbed` is `iterative_scoring(...)["unabsorbed_clusters"]`, the clusters the published
    fixed rule flags. A cluster is `absorbed` if the rule did NOT flag it.

    `isolated` is True when one cluster holds >= ISOLATION_SHARE of the poison at purity
    >= ISOLATION_PURITY. Read with `best.absorbed`: isolated and absorbed means the port is
    sound and the decision rule fails; not isolated means the clustering call never separated
    the poison and nothing downstream is interpretable.
    """
    labels = np.asarray(labels)
    is_poison = np.asarray(is_poison, dtype=bool)
    unabsorbed = set(int(c) for c in unabsorbed)
    total_poison = int(is_poison.sum())
    clusters = []
    for lab in sorted(set(int(v) for v in labels)):
        m = labels == lab
        size, n_p = int(m.sum()), int((m & is_poison).sum())
        clusters.append(dict(
            label=lab, size=size, n_poison=n_p,
            purity=(n_p / size) if size else 0.0,
            poison_share=(n_p / total_poison) if total_poison else 0.0,
            absorbed=(lab not in unabsorbed),
        ))
    best = max(clusters, key=lambda c: (c["poison_share"], c["purity"]))
    noise = next((c for c in clusters if c["label"] == NOISE_LABEL), None)
    return dict(
        clusters=clusters,
        best=best,
        poison_in_noise=int(noise["n_poison"]) if noise else 0,
        isolated=bool(best["poison_share"] >= ISOLATION_SHARE
                      and best["purity"] >= ISOLATION_PURITY),
        thresholds=dict(share=ISOLATION_SHARE, purity=ISOLATION_PURITY),
    )
