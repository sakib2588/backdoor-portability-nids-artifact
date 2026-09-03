#!/usr/bin/env python
"""Does Activation Clustering fail on NetFlow, or does our subsampling make it fail?

scripts/81 scores detectors on a uniform subsample of the target-class pool, capped at 200,000 rows,
because that pool reaches 921,824 here against CTU-13's 111,666. Every cell in which Activation
Clustering flagged the wrong cluster was a subsampled cell. Subsampling is also confounded with the
low poison fraction, since the cells needing a cap are exactly the high-benign-share ones, so the
grid cannot separate the two explanations.

This runs the SAME cell twice, changing only the number of rows scored, and reports Activation
Clustering both ways. If the outcome is unchanged, the cap is exonerated and the failure is the
detector's. If it flips, the failure is ours and scripts/81's Activation Clustering column must be
re-run uncapped or withdrawn.

Only Activation Clustering is recomputed. Spectral's float64 SVD has a large transient peak at this
scale and is not the detector under suspicion.

Run:  .venv/bin/python scripts/87_ac_subsampling_control.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.constraints_netflow import admitted_manifest, project_netflow_to_feasible
from src.data import apply_standardiser, fit_standardiser
from src.data_netflow import split_netflow
from src.detectors import poison_recall
from src.detectors.activation_clustering import detect as ac_detect
from src.models import attack_success_rate, mlp_penultimate_features, train_mlp
from src.poison import poison_secondary_trainset_raw
from src.trigger import apply_secondary_trigger, build_secondary_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
N_SIGMA, COST, RATE = 6.0, 16, 0.01
SILHOUETTE_SAMPLE = 10_000
CAPPED = 200_000
SEEDS = [42, 123, 456]
TAG = "share0.96__NF-CSE-CIC-IDS2018-v2"   # 921,600 benign rows, poison ~1.04% of them
OUT = config.RESULTS / "ac_subsampling_control.json"

if "--tag" in sys.argv:
    TAG = sys.argv[sys.argv.index("--tag") + 1]
    OUT = config.RESULTS / f"ac_subsampling_control_{TAG}.json"
if "--seeds" in sys.argv:
    SEEDS = [int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(",")]


def main() -> int:
    gate = json.loads((config.RESULTS / "netflow_data_gate.json").read_text())
    row = gate["samples"][TAG]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kept, _ = admitted_manifest(gate["per_corpus_satisfaction"],
                                corpora=[config.NETFLOW_MANIPULATION_CORPUS,
                                         config.NETFLOW_REPLICATION_CORPUS])
    df = pd.read_parquet(config.ROOT / row["parquet"])
    S = split_netflow(df, dict(feature_columns=row["feature_columns"]))
    del df
    gc.collect()
    cols = row["feature_columns"]
    scaler = fit_standardiser(S.x_tr_raw)
    bounds = (S.x_tr_raw.min(axis=0), S.x_tr_raw.max(axis=0))
    project = partial(project_netflow_to_feasible, relations=kept, feature_names=cols, bounds=bounds)

    out = []
    for seed in SEEDS:
        t0 = time.time()
        x_tr_std = apply_standardiser(scaler, S.x_tr_raw)
        clean = train_mlp(x_tr_std, S.y_tr, seed, epochs=20, device=device)
        ranking = shap_rank_features(clean, S.x_tr_raw, scaler, target=TARGET, kind="mlp",
                                     device=device)
        del x_tr_std, clean
        gc.collect()
        trig = build_secondary_trigger(ranking, COST, S.x_tr_raw, cols, bounds, project,
                                       list(range(len(cols))), n_sigma=N_SIGMA)
        x_p_raw, y_p, pidx = poison_secondary_trainset_raw(
            S.x_tr_raw, S.y_tr, trig, RATE, target=TARGET, seed=seed)
        x_p_std = apply_standardiser(scaler, x_p_raw)
        del x_p_raw
        gc.collect()
        mlp = train_mlp(x_p_std, y_p, seed, epochs=20, device=device)
        atk = apply_standardiser(scaler, apply_secondary_trigger(S.x_te_raw[S.y_te != TARGET], trig))
        asr = float(attack_success_rate(mlp, atk, TARGET, device=device))
        del atk

        benign_pos = np.where(y_p == TARGET)[0]
        is_poison_full = np.isin(benign_pos, pidx)
        rng = np.random.default_rng(20260903 + seed)   # same stream scripts/81 uses
        take = np.sort(rng.choice(len(benign_pos), size=CAPPED, replace=False))

        for label, pos, poi in (("capped", benign_pos[take], is_poison_full[take]),
                                ("full", benign_pos, is_poison_full)):
            feats = mlp_penultimate_features(mlp, x_p_std[pos], device)
            mask, sil = ac_detect(feats, seed=seed, silhouette_sample=SILHOUETTE_SAMPLE)
            n_clean = int((~poi).sum())
            rec = poison_recall(mask, poi)
            out.append(dict(seed=seed, scale=label, n_scored=int(len(poi)),
                            n_poison=int(poi.sum()),
                            poison_frac=float(poi.mean()),
                            n_flagged=int(mask.sum()), recall=float(rec),
                            fpr=float((mask & ~poi).sum() / n_clean) if n_clean else None,
                            silhouette=(float(sil) if sil is not None else None),
                            asr=asr))
            print(f"  seed {seed} {label:<7} n={len(poi):>7,} poison={int(poi.sum()):>6,} "
                  f"({poi.mean()*100:.2f}%) flagged={int(mask.sum()):>7,} recall={rec:.3f} "
                  f"sil={sil if sil is None else round(sil,3)}")
            del feats, mask
            gc.collect()
        del mlp, x_p_std
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  seed {seed} done in {time.time()-t0:.0f}s")

    same = all(
        (a["recall"] > 0.5) == (b["recall"] > 0.5)
        for a, b in zip([o for o in out if o["scale"] == "capped"],
                        [o for o in out if o["scale"] == "full"]))
    verdict = ("cap exonerated: the outcome is unchanged at full scale"
               if same else "CAP IMPLICATED: the outcome flips at full scale")
    OUT.write_text(json.dumps(dict(tag=TAG, cost=COST, rate=RATE, capped_at=CAPPED,
                                   seeds=SEEDS, rows=out, verdict=verdict), indent=2))
    print(f"\n{verdict}")
    print(f"wrote {OUT.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
