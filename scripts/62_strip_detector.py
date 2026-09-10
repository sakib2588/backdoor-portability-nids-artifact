"""STRIP on the tabular NIDS window cells (both datasets), after the vision-control gate
(scripts/61_strip_vision_control.py) passed at recall 0.977-1.0, AUC 0.964-0.999 over 5 seeds.

Minimum bar per the plan: CTU-13's 4 window cells and UNSW-NB15's 3 window cells (the same cells
already reported for Spectral/AC/NC in Table I / Table II), 5 base seeds each. Each cell is an
independent retrain of the poisoned MLP (same construction as scripts/04/16), scored by both Spectral
(cross-check against the existing detectors.json/secondary_detectors.json numbers for that cell) and
STRIP. STRIP's blend and calibration pool is a clean slice of the TRAINING partition, never
the test block, so the rule that no transform is fitted on test holds for this arm too.

Run:  python scripts/62_strip_detector.py            # full window-cell sweep, both datasets
      python scripts/62_strip_detector.py --smoke     # 1 seed, 1 cell per dataset
"""
from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config as get_primary_config, load_setup as load_primary_setup
from src import config
from src.data import apply_standardiser
from src.data_secondary import load_secondary_setup
from src.detectors import poison_recall
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.detectors.strip import (calibrate_strip_cutoff, strip_scores,
                                 strip_threshold_flag)
from src.models import attack_success_rate, clean_accuracy, mlp_penultimate_features, train_mlp
from src.poison import poison_secondary_trainset, poison_trainset_cleanlabel
from src.tb_vendor.constraints_numeric import check_constraints
from src.trigger import (
    apply_secondary_trigger, apply_trigger, build_secondary_trigger, build_trigger,
    raw_trigger_stats, shap_rank_features,
)
from src.constraints_secondary import manifest_fingerprint, project_secondary_to_feasible
from _m3_secondary_common import eligible_indices_for

TARGET = config.ATTACK_TARGET
Z_THRESHOLDS = (2.0, 2.5, 3.0)
N_TRIALS = 64
ALPHA = 0.5
STRIP_FRR = 0.01     # Gao et al.'s published operating point, 1% false rejection
STRIP_RULE = "gao-frr-nonstrict"   # tie-inclusive; a strict cut cannot fire on the softmax floor

CTU_WINDOW_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]
UNSW_WINDOW_CELLS = [(0.005, 16), (0.01, 8), (0.01, 16)]

CKPT = "strip_detector.checkpoint.json"
OUT = "strip_detector.json"


def cell_key(dataset, seed, rate, cost) -> str:
    return f"{dataset}|{seed}|{rate}|{cost}"


def load_checkpoint(path, key):
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    # Compare against the JSON projection of the key, not the key itself: the cell
    # lists hold tuples in memory and come back from disk as lists, so a direct
    # comparison never matches and every run silently restarts the whole sweep.
    if blob.get("config_key") != json.loads(json.dumps(key)):
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} row(s) already done")
    return blob.get("rows", {})


def save_checkpoint(path, key, rows):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(path)


def score_cell(mlp, x_p_std, y_p, poison_idx, pool_std, device, seed):
    benign_pos = np.where(y_p == TARGET)[0]
    is_poison = np.isin(benign_pos, poison_idx)
    has_poison = bool(is_poison.sum() > 0)
    n_poison, n_benign = int(is_poison.sum()), len(is_poison)
    n_clean = n_benign - n_poison

    feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
    spec_scores = spectral_scores(feats, n_components=config.SPECTRAL_COMPONENTS)
    spec_fixed = flag_by_scores(spec_scores, expected_frac=float(is_poison.mean()))
    spec_recall = poison_recall(spec_fixed, is_poison) if has_poison else None
    spec_auc = (float(roc_auc_score(is_poison, spec_scores))
                if has_poison and is_poison.sum() < len(is_poison) else None)

    query = x_p_std[benign_pos]
    strip_sc = strip_scores(mlp, query, pool_std, n_trials=N_TRIALS, alpha=ALPHA,
                             device=device, seed=seed)
    strip_fixed = flag_by_scores(strip_sc, expected_frac=float(is_poison.mean()))
    strip_fixed_recall = poison_recall(strip_fixed, is_poison) if has_poison else None

    # STRIP's own published rule: an entropy cutoff fit at a 1% false rejection rate on clean
    # held-out rows, disjoint from the rows scored above. Until this was wired in, STRIP was the
    # only detector in this project scored solely under a top-k budget it does not publish, and
    # its recall under that budget feeds two claims in the manuscript.
    strip_cut, strip_diag = calibrate_strip_cutoff(mlp, pool_std, n_trials=N_TRIALS, alpha=ALPHA,
                                                   device=device, seed=seed, frr=STRIP_FRR,
                                                   return_diagnostics=True)
    strip_frr_flag = strip_threshold_flag(strip_sc, strip_cut)
    strip_frr_recall = poison_recall(strip_frr_flag, is_poison) if has_poison else None
    strip_frr_fpr = (float((strip_frr_flag & ~is_poison).sum() / n_clean) if n_clean > 0 else None)
    strip_auc = (float(roc_auc_score(is_poison, strip_sc))
                 if has_poison and is_poison.sum() < len(is_poison) else None)
    strip_mad = {}
    for z in Z_THRESHOLDS:
        mad_flag = flag_by_mad_threshold(strip_sc, z_thresh=z)
        n_tp = int(np.sum(mad_flag & is_poison))
        n_fp = int(mad_flag.sum()) - n_tp
        strip_mad[str(z)] = dict(
            recall=poison_recall(mad_flag, is_poison) if has_poison else None,
            fpr=(n_fp / n_clean) if n_clean > 0 else None)

    return dict(
        n_poison=n_poison, n_benign=n_benign,
        spectral=dict(recall=spec_recall, auc=spec_auc),
        strip=dict(fixed_recall=strip_fixed_recall, auc=strip_auc, mad=strip_mad,
                   frr_recall=strip_frr_recall, frr_fpr=strip_frr_fpr,
                   frr_target=STRIP_FRR, entropy_cut=float(strip_cut),
                   frr_calibration=dict(strip_diag)),
    )


def run_primary(cfg, rows, save_cb, device):
    S = load_primary_setup(cfg)
    constraints, bounds = S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])
    # STRIP's reference and calibration pool. The validation slice is carved out of the train pool
    # by temporal_split, so the victim never trains on it and no poison can reach it.
    strip_pool_std = apply_standardiser(S["scaler"], S["x_val_raw"])
    x_bot_raw = S["x_te_raw"][S["y_te"] == 1]
    stats = raw_trigger_stats(x_tr_raw, constraints)

    for seed in cfg["seeds"]:
        x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
        clean_mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET, kind="mlp",
                                      device=device)
        for rate, cost in CTU_WINDOW_CELLS:
            key = cell_key("ctu13", seed, rate, cost)
            if key in rows:
                continue
            realiz, _ = build_trigger(ranking, cost, x_tr_raw, constraints, S["features"], bounds,
                                       stats=stats)
            x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
                x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
                constraints=constraints, feature_names=S["features"], bounds=bounds)
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            x_bot_trig = apply_trigger(x_bot_raw, realiz, constraints, S["features"], bounds)
            asr = attack_success_rate(mlp, apply_standardiser(S["scaler"], x_bot_trig), TARGET, device)
            cacc = clean_accuracy(mlp, x_te_std, S["y_te"], device)

            row = score_cell(mlp, x_p_std, y_p, poison_idx, strip_pool_std, device, seed)
            row.update(dataset="ctu13", seed=seed, rate=rate, cost=cost, asr=asr, clean_acc=cacc)
            rows[key] = row
            save_cb()
            print(f"  ctu13 seed={seed} rate={rate} cost={cost} "
                  f"strip_recall={row['strip']['fixed_recall']} spec_recall={row['spectral']['recall']}")


def run_secondary(cfg, rows, save_cb, device):
    S0 = load_secondary_setup(config.SECONDARY_DATASET_ID)
    eligible = eligible_indices_for(S0.feature_names)

    # STRIP's reference and calibration pool for this dataset. UNSW-NB15 ships no validation split,
    # so one is drawn here from the TRAINING partition and excluded from the poison candidate pool,
    # which makes it provably clean. It differs from the primary dataset's pool in one respect worth
    # stating in the paper: the victim does train on these rows, whereas CTU-13's validation slice is
    # carved out of the train pool. Both satisfy the rule that no transform is fitted on the test
    # block, which is the property STRIP was violating.
    n_train = len(S0.x_train_raw)
    strip_pool_idx = np.sort(np.random.default_rng(config.VAL_SPLIT_SEED).choice(
        n_train, size=int(round(config.VAL_FRAC * n_train)), replace=False))

    for seed in cfg["seeds"]:
        x_tr_std = apply_standardiser(S0.scaler, S0.x_train_raw)
        clean_mlp = train_mlp(x_tr_std, S0.y_train, seed, epochs=cfg["mlp_epochs"], device=device,
                               log_every=5)
        ranking = shap_rank_features(clean_mlp, S0.x_train_raw, S0.scaler, target=TARGET, kind="mlp",
                                      device=device)
        x_te_std = apply_standardiser(S0.scaler, S0.x_test_raw)
        x_te_attack_raw = S0.x_test_raw[S0.y_test != TARGET]

        trig_cache = {}
        for rate, cost in UNSW_WINDOW_CELLS:
            key = cell_key("unsw_nb15", seed, rate, cost)
            if key in rows:
                continue
            if cost not in trig_cache:
                project_fn = partial(project_secondary_to_feasible, constraints=S0.constraints,
                                      bounds=S0.bounds)
                trig_cache[cost] = build_secondary_trigger(
                    ranking, cost, S0.x_train_raw, S0.feature_names, S0.bounds, project_fn, eligible)
            realiz = trig_cache[cost]

            x_p_std, y_p, poison_idx = poison_secondary_trainset(
                S0.x_train_raw, S0.y_train, realiz, rate, S0.scaler, target=TARGET, seed=seed,
                exclude_idx=strip_pool_idx)
            # excluded above, so these rows carry no trigger and the pool is clean by construction
            strip_pool_std = x_p_std[strip_pool_idx]
            assert not np.intersect1d(poison_idx, strip_pool_idx).size
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            x_te_attack_trig_raw = apply_secondary_trigger(x_te_attack_raw, realiz)
            asr = attack_success_rate(mlp, apply_standardiser(S0.scaler, x_te_attack_trig_raw),
                                       TARGET, device)
            cacc = clean_accuracy(mlp, x_te_std, S0.y_test, device)

            row = score_cell(mlp, x_p_std, y_p, poison_idx, strip_pool_std, device, seed)
            row.update(dataset="unsw_nb15", seed=seed, rate=rate, cost=cost, asr=asr, clean_acc=cacc)
            rows[key] = row
            save_cb()
            print(f"  unsw seed={seed} rate={rate} cost={cost} "
                  f"strip_recall={row['strip']['fixed_recall']} spec_recall={row['spectral']['recall']}")


def main():
    global CTU_WINDOW_CELLS, UNSW_WINDOW_CELLS
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    cfg = get_primary_config(smoke)
    if smoke:
        cfg["seeds"] = [42]
        CTU_WINDOW_CELLS, UNSW_WINDOW_CELLS = CTU_WINDOW_CELLS[:1], UNSW_WINDOW_CELLS[:1]
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} smoke={smoke} seeds={cfg['seeds']}")

    out_path = config.RESULTS / (OUT if not smoke else "strip_detector_smoke.json")
    ckpt_path = out_path.with_suffix(".checkpoint.json")
    ckpt_key = dict(smoke=smoke, strip_pool="train_holdout_v2", ctu_cells=CTU_WINDOW_CELLS,
                     unsw_cells=UNSW_WINDOW_CELLS,
                     mlp_epochs=cfg["mlp_epochs"], n_trials=N_TRIALS, alpha=ALPHA,
                     strip_frr=STRIP_FRR, strip_rule=STRIP_RULE,
                     constraints=manifest_fingerprint())
    rows = load_checkpoint(ckpt_path, ckpt_key)
    save_cb = lambda: save_checkpoint(ckpt_path, ckpt_key, rows)

    run_primary(cfg, rows, save_cb, device)
    run_secondary(cfg, rows, save_cb, device)

    out_path.write_text(json.dumps(dict(config=ckpt_key, rows=rows), indent=2))
    print(f"wrote {out_path}  elapsed={time.time() - t0:.0f}s  n_rows={len(rows)}")


if __name__ == "__main__":
    main()
