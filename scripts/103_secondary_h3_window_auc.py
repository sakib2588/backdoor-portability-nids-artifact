"""E3. The imbalance confound (H3) at the UNSW-NB15 evasion window, with the ranking AUC.

Same question and same instrument as scripts/69, asked on the second corpus. The manuscript's
imbalance rule-out is currently scoped to CTU-13, which is also the corpus where removal
provably does not matter -- so a reader can reasonably ask whether "the budget starves, the
score survives" is a claim about tabular NIDS or a claim about one capture. This closes that.

The distinction the AUC draws, and why recall alone cannot draw it. If balancing the training
set lifts recall, that could be either of two very different things: the fixed budget was
starved by the class size n and balancing fed it, OR balancing changed the representation and
built a better score. Recall alone cannot separate them. The AUC can: if the AUCs are close on
both arms and only recall moves, the budget was starved; if the balanced AUC moves too, the
score itself changed. That is the score-versus-rule distinction the whole paper is built on.

The imbalanced arm is READ from results/secondary_detectors.json, never recomputed. Two scripts
computing one quantity along two paths is how this project previously produced 0.9828 and
0.9814 for the same number, and it also makes this run far cheaper: UNSW's imbalanced arm is
the expensive one and it is already on disk.

COUNTER-INTUITIVE SIZING, worth stating because it inverts the obvious guess. The BALANCED arm
is the cheap one. `downsample_benign(target_ratio=0.5)` keeps all of the minority rows and
matches the majority to them, so the balanced training set is a small fraction of the
imbalanced one. The ~1.23M-row hazard that drove the 2026-07-31 OOM lives in the arm this
script does not retrain.

Even so the `del ...; gc.collect()` pairs stay. `spectral_scores` upcasts to float64 for its
SVD, that transient is deliberately unfixed to hold reproducibility at atol 1e-6, and a cell
that used to fit can stop fitting the moment anything else on the machine grows.

Window cells are the three that results/secondary_adaptive_threshold.json classifies as
`is_evasion_window == true`, the same list scripts/26 and scripts/62 hard-code. Across them the
fixed budget recovers 0.0003 to 0.0554 while the AUC stays 0.9533 to 0.9650. The list is
inherited rather than re-derived from the recall and AUC columns: re-deriving it adds
(0.005, 8), a cell the upstream classification does not carry, and would put this script on a
different cell set than every other UNSW result in the paper.

Run:  .venv/bin/python scripts/103_secondary_h3_window_auc.py
      .venv/bin/python scripts/103_secondary_h3_window_auc.py --seeds 42     # measure one
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m3_secondary_common import (
    EXCLUDED_TRIGGER_FEATURES, cell_key, eligible_indices_for, get_config,
)
from src import config
from src.constraints_secondary import manifest_fingerprint, project_secondary_to_feasible
from src.data import apply_standardiser, downsample_benign, fit_standardiser
from src.data_secondary import load_secondary_setup
from src.detectors import poison_recall
from src.detectors.activation_clustering import detect as ac_detect
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.models import mlp_penultimate_features, train_mlp
from src.poison import poison_secondary_trainset
from src.trigger import build_secondary_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
DATASET_ID = config.SECONDARY_DATASET_ID
SPECTRAL_K = config.SPECTRAL_COMPONENTS
SILHOUETTE_SAMPLE = 10_000        # AC's silhouette is an O(n^2) diagnostic; cap it, as scripts/15 does
H3_TARGET_RATIO = 0.5
MAD_Z = config.MAD_Z_PRIMARY
WINDOW_CELLS = [(0.005, 16), (0.01, 8), (0.01, 16)]

OUT = config.RESULTS / "secondary_h3_window_auc.json"
CKPT = config.RESULTS / "secondary_h3_window_auc.checkpoint.json"


def load_imbalanced_reference() -> dict:
    path = config.RESULTS / "secondary_detectors.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run scripts/15_secondary_detectors.py first; the imbalanced "
            "arm is read from that committed grid rather than recomputed here"
        )
    blob = json.loads(path.read_text())
    return {(r["seed"], r["rate"], r["cost"]): r for r in blob["main_grid"]}


def config_key(cfg, cells) -> dict:
    return dict(dataset=DATASET_ID, cells=[list(c) for c in cells], seeds=list(cfg["seeds"]),
                mlp_epochs=cfg["mlp_epochs"], spectral_k=SPECTRAL_K, mad_z=MAD_Z,
                target_ratio=H3_TARGET_RATIO, n_sigma=cfg["n_sigma"],
                silhouette_sample=SILHOUETTE_SAMPLE,
                excluded_trigger_features=sorted(EXCLUDED_TRIGGER_FEATURES),
                constraints=manifest_fingerprint(), smoke=cfg["smoke"])


def load_ckpt(key) -> dict:
    if not CKPT.exists():
        return {}
    try:
        blob = json.loads(CKPT.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != json.loads(json.dumps(key)):
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} cell-seed(s) already done")
    return blob.get("rows", {})


def save_ckpt(key, rows) -> None:
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(CKPT)


def balanced_setup(S, cfg, seed, device):
    """Rebalance, then rebuild everything that depends on the training distribution.

    A clean model trained on the rebalanced set has different SHAP importances than the
    imbalanced one, so the ranking and therefore the trigger must both be rebuilt here. Reusing
    the imbalanced ranking would measure a different attack, not a different balance.
    """
    cols = list(S.feature_names)
    x_bal_df, y_bal_ser = downsample_benign(
        pd.DataFrame(S.x_train_raw, columns=cols), pd.Series(S.y_train),
        target_ratio=H3_TARGET_RATIO, seed=seed)
    x_bal, y_bal = x_bal_df.to_numpy(), y_bal_ser.to_numpy()
    del x_bal_df, y_bal_ser
    gc.collect()

    scaler_bal = fit_standardiser(pd.DataFrame(x_bal, columns=cols))
    x_std_bal = apply_standardiser(scaler_bal, x_bal)
    clean_mlp = train_mlp(x_std_bal, y_bal, seed, epochs=cfg["mlp_epochs"], device=device)
    ranking = shap_rank_features(clean_mlp, x_bal, scaler_bal, target=TARGET, kind="mlp",
                                 device=device)
    del x_std_bal, clean_mlp
    gc.collect()
    return x_bal, y_bal, scaler_bal, ranking


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", default=None, help="comma-separated, e.g. 42 to measure one")
    args = ap.parse_args()

    t0 = time.time()
    cfg = get_config(args.smoke)
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    cells = list(WINDOW_CELLS)

    global OUT, CKPT
    if args.smoke:
        cfg["seeds"] = [42]
        cells = cells[:1]
        OUT = config.RESULTS / "secondary_h3_window_auc_smoke.json"
        CKPT = config.RESULTS / "secondary_h3_window_auc_smoke.checkpoint.json"

    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = config_key(cfg, cells)
    rows = load_ckpt(key)
    imbalanced_ref = load_imbalanced_reference()
    print(f"device={device}  dataset={DATASET_ID}  cells={cells}  seeds={cfg['seeds']}")

    S = load_secondary_setup(DATASET_ID)
    eligible = eligible_indices_for(S.feature_names)
    print(f"imbalanced train rows={len(S.y_train)}  minority={int((S.y_train == 1).sum())}")

    for seed in cfg["seeds"]:
        pending = [c for c in cells if cell_key(seed, *c) not in rows]
        if not pending:
            print(f"--- seed {seed}: all cells cached, skipping ---")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")

        x_bal, y_bal, scaler_bal, ranking = balanced_setup(S, cfg, seed, device)
        print(f"  balanced train set: n={len(y_bal)} minority={int((y_bal == 1).sum())}")

        trig_cache = {}
        for rate, cost in pending:
            c0 = time.time()
            if cost not in trig_cache:
                project_fn = partial(project_secondary_to_feasible, constraints=S.constraints,
                                     bounds=S.bounds)
                trig_cache[cost] = build_secondary_trigger(
                    ranking, cost, x_bal, S.feature_names, S.bounds, project_fn, eligible,
                    n_sigma=cfg["n_sigma"])
            realiz = trig_cache[cost]

            x_p_std, y_p, poison_idx = poison_secondary_trainset(
                x_bal, y_bal, realiz, rate, scaler_bal, target=TARGET, seed=seed)
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            benign_pos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(benign_pos, poison_idx)
            ref = imbalanced_ref.get((seed, rate, cost))
            if ref is None:
                raise SystemExit(
                    f"no committed secondary grid row for seed={seed} rate={rate} cost={cost}")
            if is_poison.sum() == 0:
                rows[cell_key(seed, rate, cost)] = dict(
                    seed=seed, rate=rate, cost=cost, interpretable=False,
                    reason="no poison rows landed in the target class of the balanced set")
                save_ckpt(key, rows)
                print(f"  rate={rate} cost={cost}: no poison in target class, skipped")
                continue

            feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
            del x_p_std
            gc.collect()

            scores = spectral_scores(feats, n_components=SPECTRAL_K)
            ef = float(is_poison.mean())
            n_clean = int((~is_poison).sum())
            fixed_flag = flag_by_scores(scores, expected_frac=ef)
            mad_flag = flag_by_mad_threshold(scores, z_thresh=MAD_Z)
            spec_auc_bal = (float(roc_auc_score(is_poison, scores))
                            if 0 < is_poison.sum() < len(is_poison) else None)
            ac_mask, ac_sil = ac_detect(feats, seed=seed, silhouette_sample=SILHOUETTE_SAMPLE)

            rows[cell_key(seed, rate, cost)] = dict(
                seed=seed, rate=rate, cost=cost, interpretable=True,
                n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
                expected_frac=ef,
                balanced=dict(
                    spectral_recall=poison_recall(fixed_flag, is_poison),
                    spectral_mad_recall=poison_recall(mad_flag, is_poison),
                    spectral_mad_fpr=(float((mad_flag & ~is_poison).sum() / n_clean)
                                      if n_clean else None),
                    spectral_auc=spec_auc_bal,
                    ac_recall=poison_recall(ac_mask, is_poison), ac_silhouette=ac_sil,
                    train_balance=dict(n=int(len(y_bal)), minority=int((y_bal == 1).sum()))),
                imbalanced=dict(
                    spectral_recall=ref["spectral"]["recall"],
                    spectral_mad_recall=ref["spectral"].get("mad_recall"),
                    spectral_auc=ref["spectral"]["auc"],
                    ac_recall=ref["ac"]["recall"]))
            save_ckpt(key, rows)

            r = rows[cell_key(seed, rate, cost)]
            b, i = r["balanced"], r["imbalanced"]
            print(f"  rate={rate} cost={cost} ({time.time() - c0:.0f}s): "
                  f"balanced recall={b['spectral_recall']:.4f} auc={b['spectral_auc']:.4f} | "
                  f"imbalanced recall={i['spectral_recall']:.4f} auc={i['spectral_auc']:.4f}")

            del mlp, feats, scores, fixed_flag, mad_flag, ac_mask
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        del x_bal, y_bal, scaler_bal, ranking, trig_cache
        gc.collect()

    ordered = [rows[cell_key(s, r, c)] for s in cfg["seeds"] for r, c in cells
               if cell_key(s, r, c) in rows]
    usable = [r for r in ordered if r.get("interpretable", False)]

    def agg(arm, field):
        vals = [r[arm][field] for r in usable if r[arm].get(field) is not None]
        if not vals:
            return None
        return dict(mean=float(np.mean(vals)),
                    sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None, n=len(vals))

    summary = dict(
        balanced={f: agg("balanced", f) for f in
                  ("spectral_recall", "spectral_mad_recall", "spectral_auc", "ac_recall")},
        imbalanced={f: agg("imbalanced", f) for f in
                    ("spectral_recall", "spectral_mad_recall", "spectral_auc", "ac_recall")})

    OUT.write_text(json.dumps(dict(dataset=DATASET_ID, config=key, summary=summary,
                                   rows=ordered), indent=2) + "\n")
    print(f"\nwrote {OUT}  elapsed={time.time() - t0:.0f}s  interpretable={len(usable)}")
    b, i = summary["balanced"], summary["imbalanced"]
    if b["spectral_recall"] and i["spectral_recall"]:
        print(f"  balanced   : recall {b['spectral_recall']['mean']:.4f}  "
              f"auc {b['spectral_auc']['mean']:.4f}")
        print(f"  imbalanced : recall {i['spectral_recall']['mean']:.4f}  "
              f"auc {i['spectral_auc']['mean']:.4f}")
        print("  Close AUCs with recall moving = balancing fed a starved budget.")
        print("  Balanced AUC moving too = balancing changed the score, not just the budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
