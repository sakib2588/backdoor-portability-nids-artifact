"""Task NC-13 stage 1: does expected transferability separate poisoned from clean models here?

WHY. NC-6's re-analysis showed the mask-L1 ratio separates poisoned from clean models at AUC
0.389-0.444 -- at or below chance -- because under binary class imbalance it measures class asymmetry
rather than backdoor presence (all 432 rows flagged class 0). All three NC-6 candidates varied the
OPTIMISER while holding that statistic fixed, so none of them could have worked. ET changes the
statistic instead: it scores each class independently, needs no null distribution, and carries a
DERIVED threshold of 1/2 that is never calibrated.

STAGE 1 IS A PILOT WITH A FROZEN STOP RULE, set in
notes/20260803-decision-nc13-et-preregistration.md before any ET value existed on this data:

    proceed to the full run IFF  ET > 0.5 on >= 4 of 6 poisoned models
                            AND  ET <= 0.5 on >= 8 of 10 clean models

Anything else stops the track and is reported as "ET does not separate on tabular NIDS" -- a
publishable negative in exactly the way NC-6 is. The rule exists so the pilot cannot be re-read
favourably after the fact.

The inversion recipe is held at the published baseline (l1=0.001, constraint_weight=0.0), so ET is
the ONLY thing that changes against NC-6. Constraint-projected ET is explicitly out of scope,
especially given NC-7 showed constraint_weight=1.0 is itself broken.

THE RISK TO WATCH, named in the pre-registration: under this class imbalance, flipping a botnet row
to benign may be trivially easy for reasons unrelated to any backdoor, which could saturate ET toward
1 on CLEAN models too. That is why the clean arm is in the pilot rather than deferred -- if clean
models also exceed 0.5, ET is measuring the imbalance, not the backdoor.

Run:  python scripts/25_nc_et_pilot.py --smoke --seeds 42
      python scripts/25_nc_et_pilot.py
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
from src.detectors.neural_cleanse import TabularInversionConfig, expected_transferability
from src.models import train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET          # benign; the attack's target class
N_DETECTION_SAMPLES = 10               # N_i, frozen in the pre-registration
N_RESTARTS = 3                         # R, frozen
N_CLEAN_PILOT = 10                     # stage-1 clean arm
PILOT_CELLS = [(0.005, 16), (0.01, 16)]   # matches NC-6 so the comparison is like-for-like
OUT_PATH = config.RESULTS / "nc_et_pilot.json"


def _finite_or_none(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def load_checkpoint(path, key):
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} models already scored")
    return blob.get("rows", {})


def save_checkpoint(path, key, rows):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(path)


def _config_key(cfg, seeds) -> dict:
    return dict(seeds=list(seeds), cells=[list(c) for c in PILOT_CELLS],
                n_detection_samples=N_DETECTION_SAMPLES, n_restarts=N_RESTARTS,
                n_clean=N_CLEAN_PILOT, nc_steps=cfg["nc_steps"],
                mlp_epochs=cfg["mlp_epochs"], smoke=cfg["smoke"])


def main():
    t0 = time.time()
    argv = sys.argv
    cfg = apply_overrides(get_config(smoke="--smoke" in argv), argv)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} seeds={cfg['seeds']} cells={PILOT_CELLS} "
          f"N_i={N_DETECTION_SAMPLES} R={N_RESTARTS}")

    ckpt_path = OUT_PATH.with_name(OUT_PATH.stem + ".checkpoint.json")
    S = load_setup(cfg)
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    stats = raw_trigger_stats(S["x_tr_raw"], S["constraints"])
    std_lo, std_hi = S["std_bounds"]
    bounds = (torch.as_tensor(std_lo, dtype=torch.float32),
              torch.as_tensor(std_hi, dtype=torch.float32))
    inv_cfg = TabularInversionConfig(steps=cfg["nc_steps"], l1_weight=0.001, constraint_weight=0.0)

    # ET scores class i = 1 - putative_target, so with putative_target = TARGET (benign) the
    # detection rows are BOTNET rows. Drawn once, deterministically, and shared by every model so
    # differences are attributable to the model rather than to the sample.
    source = 1 - TARGET
    src_idx = np.where(S["y_tr"] == source)[0]
    picked = src_idx[np.linspace(0, len(src_idx) - 1, N_DETECTION_SAMPLES).astype(int)]
    samples = torch.as_tensor(x_tr_std[picked], dtype=torch.float32)
    print(f"detection sample: {len(picked)} class-{source} (botnet) rows\n")

    key = _config_key(cfg, cfg["seeds"])
    store = load_checkpoint(ckpt_path, key)
    rows = []

    n_clean = 1 if cfg["smoke"] else N_CLEAN_PILOT
    for replica in range(n_clean):
        rk = f"clean|{replica}"
        if rk in store:
            rows.append(store[rk]); continue
        init_seed = cfg["seeds"][0] + replica * config.NC_REPLICA_SEED_STRIDE
        mlp = train_mlp(x_tr_std, S["y_tr"], init_seed, epochs=cfg["mlp_epochs"], device=device)
        out = expected_transferability(mlp, samples, bounds, inv_cfg, seed=init_seed,
                                       putative_target=TARGET, n_restarts=N_RESTARTS, device=device)
        row = dict(kind="clean", replica=replica, init_seed=int(init_seed),
                   et=_finite_or_none(out["et"]), detected=out["detected"],
                   n_samples=out["n_samples"])
        store[rk] = row; rows.append(row); save_checkpoint(ckpt_path, key, store)
        print(f"  clean r{replica}: ET={row['et']} detected={row['detected']} "
              f"({time.time() - t0:.0f}s)")

    for seed in cfg["seeds"]:
        ranking = None
        for rate, cost in PILOT_CELLS:
            pk = f"poisoned|{seed}|{rate}|{cost}"
            if pk in store:
                rows.append(store[pk]); continue
            if ranking is None:
                clean_mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"],
                                      device=device)
                ranking = shap_rank_features(clean_mlp, S["x_tr_raw"], S["scaler"], target=TARGET,
                                             kind="mlp", device=device)
            realiz, _ = build_trigger(ranking, cost, S["x_tr_raw"], S["constraints"], S["features"],
                                      S["bounds"], stats=stats)
            x_p_std, y_p, _ = poison_trainset_cleanlabel(
                S["x_tr_raw"], S["y_tr"], realiz, rate, S["scaler"], target=TARGET, seed=seed,
                constraints=S["constraints"], feature_names=S["features"], bounds=S["bounds"])
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)
            out = expected_transferability(mlp, samples, bounds, inv_cfg, seed=seed,
                                           putative_target=TARGET, n_restarts=N_RESTARTS,
                                           device=device)
            row = dict(kind="poisoned", seed=seed, rate=rate, cost=cost,
                       et=_finite_or_none(out["et"]), detected=out["detected"],
                       n_samples=out["n_samples"])
            store[pk] = row; rows.append(row); save_checkpoint(ckpt_path, key, store)
            print(f"  poisoned seed={seed} rate={rate} cost={cost}: ET={row['et']} "
                  f"detected={row['detected']} ({time.time() - t0:.0f}s)")

    clean = [r for r in rows if r["kind"] == "clean"]
    pois = [r for r in rows if r["kind"] == "poisoned"]
    n_pois_hit = sum(1 for r in pois if r["detected"])
    n_clean_ok = sum(1 for r in clean if r["detected"] is False)
    # The stop rule, applied EXACTLY as frozen. Not re-derived, not softened.
    proceed = (not cfg["smoke"]) and n_pois_hit >= 4 and n_clean_ok >= 8

    out = dict(
        seeds=cfg["seeds"], cells=[list(c) for c in PILOT_CELLS],
        recipe=dict(n_detection_samples=N_DETECTION_SAMPLES, n_restarts=N_RESTARTS,
                    n_clean=n_clean, nc_steps=cfg["nc_steps"], mlp_epochs=cfg["mlp_epochs"],
                    l1_weight=0.001, constraint_weight=0.0, smoke=cfg["smoke"],
                    threshold=0.5, threshold_origin="derived constant, Xiang et al. 2022 Thm. 3.1"),
        rows=rows,
        stop_rule=dict(
            requirement="ET > 0.5 on >= 4/6 poisoned AND ET <= 0.5 on >= 8/10 clean",
            n_poisoned_detected=n_pois_hit, n_poisoned=len(pois),
            n_clean_below_threshold=n_clean_ok, n_clean=len(clean),
            proceed_to_stage_2=bool(proceed),
            verdict=("proceed to stage 2" if proceed else
                     "STOP -- ET does not separate on tabular NIDS at this operating point")),
        mean_et=dict(clean=_finite_or_none(np.mean([r["et"] for r in clean if r["et"] is not None])
                                           if clean else None),
                     poisoned=_finite_or_none(np.mean([r["et"] for r in pois if r["et"] is not None])
                                              if pois else None)),
        note="NC-13 stage-1 pilot. The 1/2 threshold is DERIVED (Xiang et al. 2022), never calibrated, "
             "and was frozen before any ET value existed on this data. The stop rule is applied "
             "exactly as pre-registered. A stop verdict is a reportable negative, not a failed run: "
             "it would mean ET's transferability signal does not survive a clean-label, "
             "constraint-realizable trigger under this class imbalance.",
    )
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(config.ROOT)}")
    print(f"  clean    mean ET={out['mean_et']['clean']}  ({n_clean_ok}/{len(clean)} at or below 0.5)")
    print(f"  poisoned mean ET={out['mean_et']['poisoned']}  ({n_pois_hit}/{len(pois)} above 0.5)")
    print(f"  VERDICT: {out['stop_rule']['verdict']}")
    print(f"elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
