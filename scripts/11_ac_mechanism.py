"""M4 follow-up probe: WHY does Activation Clustering miss the poison inside the evasion window?

External review asked for the budget-free MAD threshold that rescues Spectral Signatures
(scripts/08_evasion_ablation.py) to be applied to Activation Clustering as well, the other detector
whose recall collapses at attacker-realistic poison rates. That request is declined on evidence, and
this script measures what actually happens instead.

The premise does not hold for AC. Chen et al.'s (2018) detector has NO removal budget to replace: it
splits the class-conditioned activations into two clusters and flags the smaller one, which is
already budget-free. Spectral's failure is a budget artefact (its ranking AUC stays ~0.98 while
`flag_by_scores` fixes removal at 1.5x an assumed poison fraction); AC's cannot be, because there is
no budget in the rule. A cluster-distance proxy score was built so the MAD rule could be tested on AC
anyway; it was deleted, because on real activations it correlates with Spectral's own score at
Spearman rho=0.9967 (Pearson r=0.99996 against sqrt(spectral)) -- both are distances in the PCA/SVD
subspace of the same centred activation matrix -- so any number computed on it restates Spectral
under an AC label. See notes/20260716-decision-ac-proxy-not-independent.md.

What this probe establishes: AC's evasion-window miss is a CLUSTERING-STRUCTURE failure. It records
what the 2-means split actually contains -- the minority cluster's size, and the share of it that is
truly poison (`minority_purity`) -- across the four evasion cells and the one rate-0.1 control cell
where AC works. It also records the silhouette, AC's own confidence diagnostic, so its behaviour on
splits that contain no poison can be compared against the split that works.

What this probe does NOT claim: it measures the clustering, not a ranked detector. There is no AUC,
no threshold, and no score here, because AC has none. Nothing here says whether some OTHER
AC-derived scoring rule could find the poison -- only that no threshold substitution addresses a
rule that has no budget and no ranking. Nor does it claim AC never finds the poison inside the
window: at n=5 the miss holds on 18 of 20 evasion rows, while the window's edge cell (rate 0.01,
cost 16) is seed-dependent, with 2 of 5 seeds reaching recall 1.0. A single-seed reading of this
probe will overstate the finding -- report the per-seed spread.

Run:  python scripts/11_ac_mechanism.py --seeds 42   # single-seed check first
      python scripts/11_ac_mechanism.py             # full 5-seed run
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup, apply_overrides
from src import config
from src.data import apply_standardiser
from src.detectors import poison_recall
from src.detectors.activation_clustering import detect as ac_detect
from src.models import mlp_penultimate_features, train_mlp
from src.trigger import shap_rank_features, build_trigger, raw_trigger_stats
from src.poison import poison_trainset_cleanlabel

TARGET = config.ATTACK_TARGET
AC_PCA_COMPONENTS = 10   # must match scripts/05_run_detectors.py's AC recipe -- same detector
# 05_run_detectors.py calls ac_detect(..., silhouette_sample=SILHOUETTE_SAMPLE) with SILHOUETTE_SAMPLE
# = 10000; matched here so this probe runs 05's exact recipe. The silhouette is a diagnostic only --
# it does not touch the cluster assignment, the minority choice, or the recall.
AC_SILHOUETTE_SAMPLE = 10000

# the four evasion-window cells (results/analysis.json) + the rate-0.1 control cell where AC's
# 2-cluster split does find the poison -- the contrast is the point of the probe
TARGET_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16), (0.1, 16)]
CONTROL_CELL = (0.1, 16)


def _mean_or_none(rows, key):
    """Mean over rows, skipping None. Returns None (never NaN) on an empty selection -- downstream
    results/macros.json is asserted to carry no NaN/Infinity tokens, so this file must not emit any."""
    vals = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(vals)) if vals else None


def _summarise(rows):
    return dict(n_rows=len(rows),
                mean_minority_size=_mean_or_none(rows, "minority_size"),
                mean_minority_purity=_mean_or_none(rows, "minority_purity"),
                mean_ac_recall=_mean_or_none(rows, "ac_recall"),
                mean_silhouette=_mean_or_none(rows, "silhouette"))


def main():
    t0 = time.time()
    cfg = apply_overrides(get_config(smoke=False), sys.argv)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  seeds={cfg['seeds']}  target_cells={TARGET_CELLS}  "
          f"control_cell={CONTROL_CELL}")

    S = load_setup(cfg)
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]

    # seed-invariant across the whole run (x_tr_raw/constraints/scaler don't change per seed) --
    # raw_trigger_stats's own docstring (src/trigger.py) says callers sweeping many combinations
    # must compute this ONCE, not per seed/cell.
    x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
    stats = raw_trigger_stats(x_tr_raw, constraints)

    rows = []
    for seed in cfg["seeds"]:
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET,
                                     kind="mlp", device=device)

        for rate, cost in TARGET_CELLS:
            realiz, _ = build_trigger(ranking, cost, x_tr_raw, constraints, features, bounds,
                                      stats=stats)
            x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
                x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
                constraints=constraints, feature_names=features, bounds=bounds)
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            benign_pos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(benign_pos, poison_idx)
            has_poison = bool(is_poison.sum() > 0)
            n_poison, n_benign = int(is_poison.sum()), len(is_poison)
            feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)

            mask, sil = ac_detect(feats, n_components=AC_PCA_COMPONENTS, seed=seed,
                                  silhouette_sample=AC_SILHOUETTE_SAMPLE)
            n_min = int(mask.sum())
            # purity = the share of what AC flags that is genuinely poison. None (not NaN) if the
            # minority cluster is empty, so this artefact stays strict RFC-8259 JSON for the macros SSOT.
            purity = float((mask & is_poison).sum() / n_min) if n_min else None
            recall = poison_recall(mask, is_poison) if has_poison else None

            rows.append(dict(seed=seed, rate=rate, cost=cost, n_poison=n_poison, n_benign=n_benign,
                             minority_size=n_min, minority_purity=purity, ac_recall=recall,
                             silhouette=float(sil), is_control_cell=bool((rate, cost) == CONTROL_CELL)))
            print(f"  rate={rate} cost={cost}: n_poison={n_poison} |minority|={n_min} "
                  f"purity={purity} recall={recall} silhouette={sil:.4f}")

    evasion_rows = [r for r in rows if not r["is_control_cell"]]
    control_rows = [r for r in rows if r["is_control_cell"]]

    out = dict(
        target_cells=TARGET_CELLS, control_cell=list(CONTROL_CELL), seeds=cfg["seeds"],
        ac_pca_components=AC_PCA_COMPONENTS, ac_silhouette_sample=AC_SILHOUETTE_SAMPLE,
        rows=rows,
        summary=dict(evasion_cells=_summarise(evasion_rows), control_cell=_summarise(control_rows)),
        note="Measures Activation Clustering's 2-means split itself, NOT a ranked detector: AC has "
             "no score and no removal budget (it flags the smaller cluster), so there is no AUC and "
             "no threshold here. minority_size is |smaller cluster|, minority_purity is the share of "
             "that cluster which is truly poison, ac_recall is Chen et al.'s unchanged rule (the same "
             "quantity results/detectors.json reports), silhouette is AC's own split-confidence "
             "diagnostic (a seeded subsample estimate at ac_silhouette_sample points; the clustering "
             "and the mask are always on the full set). The reviewer-suggested MAD-threshold "
             "substitution was NOT run on AC: it was tested on a cluster-distance proxy that turned "
             "out to correlate with Spectral's own score at Spearman rho=0.9967, and both proxy and "
             "arm were removed -- see notes/20260716-decision-ac-proxy-not-independent.md. Recipe "
             "(n_components, silhouette_sample) matches scripts/05_run_detectors.py exactly.")
    (config.RESULTS / "ac_mechanism.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/ac_mechanism.json ({len(rows)} rows)")
    print(json.dumps(out["summary"], indent=2))
    elapsed = time.time() - t0
    print(f"elapsed={elapsed:.0f}s for {len(cfg['seeds'])} seed(s)")


if __name__ == "__main__":
    main()
