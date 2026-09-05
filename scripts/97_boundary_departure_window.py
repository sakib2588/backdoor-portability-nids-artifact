#!/usr/bin/env python3
"""Boundary departure: falsifier, a PROJECTED gate, the window cells, and two comparators.

Pre-registered in notes/20260904-prereg-boundary-departure-inverted.md, which discloses that
one number for this statistic was seen post-hoc before the file was written. Read it before
changing any constant here.

WHAT IS DIFFERENT FROM scripts/96, AND WHY. Its gate scored the loud constraint-VIOLATING
trigger, which is byte-identical to the realizable one except that projection is switched
off (src/trigger.py:173-174). This detector's signal is created by projection, so that gate
tested the one trigger type where the mechanism cannot exist. Here gate 0 uses the
REALIZABLE trigger at the control cell.

THE DISCRIMINATING CONTROL IS THE POINT. The dull alternative is that stamping 16 features
makes a row unusual in a generic way and any outlier detector would find it, in which case
the constraint manifest is doing no work and the framing is decoration. Boundary departure
is therefore scored beside the univariate z-filter and Isolation Forest on the same block,
and must beat the better of them by >= 0.05 mean AUC to count as supported.

Bars, all fixed before this run:
  gate 0     realizable trigger, inverted AUC >= 0.90 on >= 4 of 5 seeds. Blocking.
  primary    window-cell mean AUC >= 0.85 CONFIRMED, < 0.65 REFUTED, else INCONCLUSIVE.
             0.85 rather than 0.80 because 0.8135 was already observed at a different cell.
  H-BD2      departure - max(zfilter, isolation forest) >= 0.05 mean AUC.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup
from src import config
from src.data import apply_standardiser
from src.detectors import poison_recall
from src.detectors.boundary_activation import departure_count
from src.detectors.isolation_forest import isolation_scores
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores
from src.models import train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, build_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
Z_THRESHOLDS = (2.0, 2.5, 3.0)

GATE0_AUC_BAR = 0.90
GATE0_MIN_SEEDS = 4
PRIMARY_CONFIRMED = 0.85
PRIMARY_REFUTED = 0.65
DISCRIMINATION_MARGIN = 0.05

WINDOW_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]

OUT = config.RESULTS / "boundary_departure_window.json"
CKPT = config.RESULTS / "boundary_departure_window.checkpoint.json"


def _smoke_paths():
    global OUT, CKPT
    OUT = config.RESULTS / "boundary_departure_window_smoke.json"
    CKPT = config.RESULTS / "boundary_departure_window_smoke.checkpoint.json"


def config_key(cfg) -> dict:
    return dict(seeds=list(cfg["seeds"]), mlp_epochs=cfg["mlp_epochs"],
                control_rate=cfg["control_rate"], control_cost=cfg["control_cost"],
                cells=[list(c) for c in WINDOW_CELLS], smoke=bool(cfg.get("smoke")),
                gate0_bar=GATE0_AUC_BAR, primary=PRIMARY_CONFIRMED,
                statistic="departure_count", gate_trigger="realizable")


def load_ckpt(key):
    if not CKPT.exists():
        return {}
    try:
        blob = json.loads(CKPT.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != json.loads(json.dumps(key)):
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming: {len(blob.get('rows', {}))} unit(s) done")
    return blob.get("rows", {})


def save_ckpt(key, rows):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(CKPT)


def rule_block(scores, is_poison, expected_frac):
    """Both decision rules against one score vector, plus the ranking metric."""
    n_clean = int((~is_poison).sum())
    out = {
        "fixed_recall": poison_recall(flag_by_scores(scores, expected_frac=expected_frac), is_poison),
        "auc": (float(roc_auc_score(is_poison, scores))
                if 0 < is_poison.sum() < len(is_poison) else None),
        "mad_recall": {}, "mad_fpr": {},
    }
    for z in Z_THRESHOLDS:
        flag = flag_by_mad_threshold(scores, z_thresh=z)
        out["mad_recall"][str(z)] = poison_recall(flag, is_poison)
        out["mad_fpr"][str(z)] = float((flag & ~is_poison).sum() / n_clean) if n_clean else None
    return out


def zfilter_scores(block_std):
    """The univariate comparator: max |z| per row, scripts/37_univariate_zfilter_baseline.py:145."""
    mu, sd = block_std.mean(axis=0), block_std.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return np.abs((block_std - mu) / sd).max(axis=1)


def _score_one(S, cfg, seed, device, rank_mlp, rate, cost):
    """One cell. The trigger is ALWAYS the realizable (projected) one -- see the docstring."""
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]

    realiz, _ = build_trigger(rank_mlp, cost, x_tr_raw, constraints, features, bounds)
    x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
        x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
        constraints=constraints, feature_names=features, bounds=bounds)

    # Raw post-projection rebuild, scripts/37:130-136. Hand-stamping would measure a trigger
    # this paper never deploys.
    x_raw_p = np.asarray(x_tr_raw, dtype=float).copy()
    x_raw_p[poison_idx] = apply_trigger(x_raw_p[poison_idx], realiz, constraints, features,
                                        bounds)

    benign_pos = np.where(y_p == TARGET)[0]
    is_poison = np.isin(benign_pos, poison_idx)
    expected_frac = float(is_poison.mean())

    # One call over the whole block: the relative eps scale depends on the block handed in.
    dep = departure_count(x_raw_p[benign_pos], constraints, features)
    blk_std = x_p_std[benign_pos]

    return dict(
        seed=seed, rate=rate, cost=cost,
        n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
        expected_frac=expected_frac,
        # Guarded: an empty slice yields NaN, json.dumps writes a bare NaN token, and the
        # next run's json.loads fails and silently restarts every completed cell.
        mean_departure_poison=(float(dep[is_poison].mean()) if is_poison.any() else None),
        mean_departure_clean=(float(dep[~is_poison].mean()) if (~is_poison).any() else None),
        departure=rule_block(dep, is_poison, expected_frac),
        zfilter=rule_block(zfilter_scores(blk_std), is_poison, expected_frac),
        isolation_forest=rule_block(isolation_scores(blk_std, seed=seed), is_poison,
                                    expected_frac),
    )


def main() -> int:
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    cfg = get_config(smoke)
    if smoke:
        cfg["seeds"] = [42]
        _smoke_paths()
    cfg["smoke"] = smoke
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = config_key(cfg)
    rows = load_ckpt(key)
    print(f"device={device} seeds={cfg['seeds']} cells={WINDOW_CELLS} gate=REALIZABLE")

    S = load_setup(cfg)
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]

    # ---- Falsifier: is the clean departure distribution tight? ----
    print(f"--- falsifier ({time.time() - t0:.0f}s) ---")
    dep_clean = departure_count(np.asarray(x_tr_raw, dtype=float), S["constraints"],
                                S["features"])
    falsifier = dict(n_rows=int(dep_clean.shape[0]), mean=float(dep_clean.mean()),
                     sd=float(dep_clean.std()), median=float(np.median(dep_clean)),
                     q90=float(np.quantile(dep_clean, 0.9)),
                     q99=float(np.quantile(dep_clean, 0.99)),
                     max=float(dep_clean.max()))
    print(f"  clean departure: mean={falsifier['mean']:.2f} sd={falsifier['sd']:.2f} "
          f"median={falsifier['median']:.0f} q99={falsifier['q99']:.0f} max={falsifier['max']:.0f}")

    per_seed_rank = {}

    def rank_for(seed):
        if seed not in per_seed_rank:
            xs = apply_standardiser(S["scaler"], x_tr_raw)
            clean_mlp = train_mlp(xs, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
            per_seed_rank[seed] = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"],
                                                     target=TARGET, kind="mlp", device=device)
        return per_seed_rank[seed]

    # ---- Gate 0, on the REALIZABLE trigger at the control cell ----
    for seed in cfg["seeds"]:
        k = f"control|{seed}"
        if k in rows:
            print(f"--- gate0 seed {seed}: cached ---")
            continue
        print(f"--- gate0 seed {seed} ({time.time() - t0:.0f}s) ---")
        rows[k] = _score_one(S, cfg, seed, device, rank_for(seed),
                             cfg["control_rate"], cfg["control_cost"])
        save_ckpt(key, rows)
        d = rows[k]["departure"]
        print(f"  departure auc={d['auc']} fixed={d['fixed_recall']:.4f} "
              f"mad3={d['mad_recall']['3.0']:.4f}")

    aucs = [rows[f"control|{s}"]["departure"]["auc"] for s in cfg["seeds"]
            if f"control|{s}" in rows]
    aucs = [a for a in aucs if a is not None]
    need = 1 if smoke else GATE0_MIN_SEEDS
    n_clear = sum(1 for a in aucs if a >= GATE0_AUC_BAR)

    base = dict(preregistration="notes/20260904-prereg-boundary-departure-inverted.md",
                device=device, falsifier=falsifier,
                gate0_bar=GATE0_AUC_BAR, gate0_min_seeds=need, gate0_aucs=aucs,
                gate0_seeds_clearing=n_clear, gate0_trigger="realizable (projected)",
                controls=[rows[f"control|{s}"] for s in cfg["seeds"] if f"control|{s}" in rows])

    if n_clear < need:
        base.update(blocked=True,
                    reason=(f"gate 0 did not clear: the realizable control did not reach "
                            f"inverted AUC {GATE0_AUC_BAR} on at least {need} seeds. The "
                            f"statistic does not enter the paper and nothing is tuned."),
                    elapsed_s=round(time.time() - t0, 1))
        OUT.write_text(json.dumps(base, indent=2) + "\n")
        print(f"GATE 0 FAILED: {n_clear}/{len(aucs)} >= {GATE0_AUC_BAR} (need {need}). Stopping.")
        return 1

    print(f"GATE 0 PASSED: {n_clear}/{len(aucs)} >= {GATE0_AUC_BAR}")

    # ---- Window cells ----
    for rate, cost in WINDOW_CELLS:
        for seed in cfg["seeds"]:
            k = f"{seed}|{rate}|{cost}"
            if k in rows:
                print(f"--- cell {k}: cached ---")
                continue
            print(f"--- cell {k} ({time.time() - t0:.0f}s) ---")
            rows[k] = _score_one(S, cfg, seed, device, rank_for(seed), rate, cost)
            save_ckpt(key, rows)
            r = rows[k]
            print(f"  dep auc={r['departure']['auc']} | zfilt={r['zfilter']['auc']} "
                  f"| iso={r['isolation_forest']['auc']}")

    cells = [rows[f"{s}|{rt}|{c}"] for rt, c in WINDOW_CELLS for s in cfg["seeds"]
             if f"{s}|{rt}|{c}" in rows]

    def agg(fn):
        vals = [v for v in (fn(r) for r in cells) if v is not None]
        if not vals:
            return None
        return dict(mean=float(np.mean(vals)),
                    sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, n=len(vals))

    dep_auc = agg(lambda r: r["departure"]["auc"])
    zf_auc = agg(lambda r: r["zfilter"]["auc"])
    if_auc = agg(lambda r: r["isolation_forest"]["auc"])
    m = dep_auc["mean"] if dep_auc else None

    if m is None:
        verdict = "NO DATA"
    elif m >= PRIMARY_CONFIRMED:
        verdict = "CONFIRMED"
    elif m < PRIMARY_REFUTED:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    best_comp = max([a["mean"] for a in (zf_auc, if_auc) if a], default=None)
    margin = (m - best_comp) if (m is not None and best_comp is not None) else None
    discriminated = bool(margin is not None and margin >= DISCRIMINATION_MARGIN)

    base.update(blocked=False, cells=cells,
                summary=dict(departure_auc=dep_auc, zfilter_auc=zf_auc,
                             isolation_forest_auc=if_auc,
                             departure_fixed_recall=agg(lambda r: r["departure"]["fixed_recall"]),
                             departure_mad3_recall=agg(lambda r: r["departure"]["mad_recall"]["3.0"]),
                             departure_mad3_fpr=agg(lambda r: r["departure"]["mad_fpr"]["3.0"])),
                primary_bars=dict(confirmed=PRIMARY_CONFIRMED, refuted=PRIMARY_REFUTED),
                verdict=verdict,
                discrimination=dict(margin=margin, bar=DISCRIMINATION_MARGIN,
                                    best_comparator_auc=best_comp,
                                    supported=discriminated),
                elapsed_s=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(base, indent=2) + "\n")
    print(f"\nPRIMARY  departure AUC {dep_auc} -> {verdict}")
    print(f"H-BD2    zfilter {zf_auc}  isolation {if_auc}")
    print(f"         margin {margin} vs bar {DISCRIMINATION_MARGIN} -> "
          f"{'SUPPORTED' if discriminated else 'NOT SUPPORTED'}")
    print(f"wrote {OUT}  ({base['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
