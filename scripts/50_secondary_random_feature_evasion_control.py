#!/usr/bin/env python3
"""Round-4 review gap: the random-feature control (scripts/40b) is CTU-13 only, but the abstract's
"sixteen features chosen at random reproduce neither" claims both datasets. This is that missing
UNSW-NB15 arm.

On CTU-13, the SHAP trigger's clean-model rate is 1.0 (total evasion) while sixteen random features
give 0.0584 -- confirming the SHAP ranking, not sheer perturbation size, drives the effect. On
UNSW-NB15 the SHAP trigger's clean-model rate is already near zero (5.35e-5 at cost 16,
results/secondary_clean_model_stamped_asr.json), so there is no evasion phenomenon there for a random
control to explain away in the same direction. What this control instead resolves is an ambiguity a
reviewer raised: is UNSW-NB15's near-zero clean-model response to the SHAP trigger because the
attribution-guided direction specifically fails to align with this victim's decision surface (a random
direction might do better or worse), or because ANY 16-feature stamp is roughly powerless against this
victim regardless of how the features are chosen (a dimensionality effect -- UNSW-NB15's schema has
only 38 features, so 16 of them is 42% of the schema, against CTU-13's 16/757 = 2.1%)? A random draw
that also lands near zero supports "hard to move at all"; a random draw that lands far from zero (in
either direction) supports "alignment matters, and this direction happened not to have it."

Same protocol as scripts/40b_random_feature_evasion_control.py, ported onto the secondary-dataset
pipeline (scripts/42's infrastructure): five seeds, five independent random draws per seed, identical
+6-sigma magnitude and constraint projection, only the feature selection changes (uniform draw from the
eligible set instead of the top-16 SHAP ranking). No poisoning -- this measures a clean, never-trained-
on-poison model's response to a stamped attack-class test row, exactly as 40b does.

Run:  .venv/bin/python scripts/50_secondary_random_feature_evasion_control.py [--write-manifest]
Checkpointed per seed to results/secondary_random_feature_evasion_control.checkpoint.json
(atomic temp-then-rename, same pattern as scripts/05_run_detectors.py); a restart skips
seeds already done. Needs torch, so the venv interpreter, not system python.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.constraints_secondary import audit_constraints, project_secondary_to_feasible
from src.data import apply_standardiser
from src.data_secondary import load_secondary_setup
from src.models import attack_success_rate, train_mlp
from src.trigger import shap_rank_features

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _m3_secondary_common import eligible_indices_for, get_config  # noqa: E402

COST = 16
TARGET = config.ATTACK_TARGET
N_DRAWS = 5
CKPT = "secondary_random_feature_evasion_control.checkpoint.json"


def committed_shap_clean_asr() -> float:
    blob = json.loads((config.RESULTS / "secondary_clean_model_stamped_asr.json").read_text())
    return float(blob["summary"][str(COST)]["a_clean_stamped_mean"])


def _config_key(cfg: dict) -> dict:
    """Every input that changes a seed's result. Seed itself is the per-unit key, not part of this,
    so restricting the seed list still resumes cleanly from a fuller run's checkpoint."""
    return dict(dataset="unsw_nb15", cost=COST, n_draws=N_DRAWS, target=int(TARGET),
                n_sigma=cfg["n_sigma"], mlp_epochs=cfg["mlp_epochs"],
                constraints=manifest_fingerprint())


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
    done = blob.get("per_seed", {})
    print(f"resuming from checkpoint: {len(done)} seed(s) already done: {sorted(done)}")
    return done


def save_checkpoint(path, key, done):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, per_seed=done)))
    tmp.replace(path)   # atomic: a crash mid-write never corrupts the checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config(smoke=False)

    S0 = load_secondary_setup()
    S = SimpleNamespace(**{f: getattr(S0, f) for f in S0.__dataclass_fields__})
    eligible = eligible_indices_for(S.feature_names)
    project_fn = partial(project_secondary_to_feasible, constraints=S.constraints, bounds=S.bounds)

    x_te_attack_raw = S.x_test_raw[S.y_test != TARGET]
    mu = S.x_train_raw.mean(axis=0)
    sd = S.x_train_raw.std(axis=0)

    print(f"device={device}  eligible features={len(eligible)}/{len(S.feature_names)}  "
          f"cost={COST}  n_draws={N_DRAWS}")

    key = _config_key(cfg)
    ckpt_path = config.RESULTS / CKPT
    done = load_checkpoint(ckpt_path, key)

    t0 = time.time()
    for seed in cfg["seeds"]:
        if str(seed) in done:
            r = done[str(seed)]
            m_prev = float(np.mean([d["asr"] for d in r["random_draws"]]))
            print(f"--- seed {seed} cached   SHAP {r['shap_trigger_clean_asr']:.4f}   "
                  f"random-feature mean {m_prev:.4f}")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s)")
        x_tr_std = apply_standardiser(S.scaler, S.x_train_raw)
        clean_mlp = train_mlp(x_tr_std, S.y_train, seed, epochs=cfg["mlp_epochs"],
                              device=device, log_every=5)

        # SHAP trigger's own clean-model ASR, this seed -- for a per-seed comparison alongside the
        # already-committed cross-seed mean (committed_shap_clean_asr above uses the same recipe).
        ranking = shap_rank_features(clean_mlp, S.x_train_raw, S.scaler, target=TARGET,
                                     kind="mlp", device=device)
        shap_idx = [i for i in ranking if i in set(eligible)][:COST]
        shap_trig_values = [float(mu[i] + cfg["n_sigma"] * (sd[i] if sd[i] > 0 else 1.0))
                            for i in shap_idx]
        shap_stamped = x_te_attack_raw.copy()
        for i, v in zip(shap_idx, shap_trig_values):
            shap_stamped[:, i] = v
        shap_stamped = np.asarray(project_fn(shap_stamped), dtype=np.float64)
        shap_asr = attack_success_rate(
            clean_mlp, apply_standardiser(S.scaler, shap_stamped), TARGET, device)

        rng = np.random.default_rng(seed)
        draws = []
        for d in range(N_DRAWS):
            idx = sorted(rng.choice(eligible, size=COST, replace=False).tolist())
            values = [float(mu[i] + cfg["n_sigma"] * (sd[i] if sd[i] > 0 else 1.0)) for i in idx]
            stamped = x_te_attack_raw.copy()
            for i, v in zip(idx, values):
                stamped[:, i] = v
            stamped = np.asarray(project_fn(stamped), dtype=np.float64)
            audit = audit_constraints(stamped, S.constraints)
            draws.append({
                "draw": d,
                "asr": float(attack_success_rate(
                    clean_mlp, apply_standardiser(S.scaler, stamped), TARGET, device)),
                "constraint_valid": bool(audit["all_projectable_valid"]),
                "overlap_with_shap": len(set(idx) & set(shap_idx)),
            })
        done[str(seed)] = {"seed": seed, "shap_trigger_clean_asr": float(shap_asr),
                           "random_draws": draws}
        save_checkpoint(ckpt_path, key, done)
        m = float(np.mean([d["asr"] for d in draws]))
        print(f"    SHAP trigger {shap_asr:.4f}   random-feature mean {m:.4f}   "
              f"per-draw {[round(d['asr'], 4) for d in draws]}")

    rows = [done[str(seed)] for seed in cfg["seeds"]]
    rand_all = [d["asr"] for r in rows for d in r["random_draws"]]
    shap_all = [r["shap_trigger_clean_asr"] for r in rows]
    m = float(np.mean(rand_all))
    committed = committed_shap_clean_asr()

    print("\n=== UNSW-NB15 random-feature evasion control")
    print(f"  SHAP trigger, clean model (this run)      : mean {np.mean(shap_all):.4f}")
    print(f"  SHAP trigger, clean model (committed)     : {committed:.4f}")
    print(f"  random {COST} features, clean model        : mean {m:.4f}  SD {np.std(rand_all):.4f}  "
          f"n={len(rand_all)}")
    interpretation = ("hard to move at all (random also near zero)" if m <= 0.20 else
                      "alignment-dependent (random differs materially from the SHAP trigger)")
    print(f"  interpretation: {interpretation}")

    blob = {"dataset": "unsw_nb15", "cost": COST, "n_draws_per_seed": N_DRAWS,
            "n_sigma": cfg["n_sigma"],
            "shap_trigger_clean_asr_mean_this_run": float(np.mean(shap_all)),
            "shap_trigger_clean_asr_committed": committed,
            "random_trigger_clean_asr_mean": m,
            "random_trigger_clean_asr_sd": float(np.std(rand_all)),
            "interpretation": interpretation,
            "per_seed": rows}
    if args.write_manifest:
        path = config.RESULTS / "secondary_random_feature_evasion_control.json"
        path.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"\nwrote {path.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
