#!/usr/bin/env python3
"""Exp C control: is it the SHAP guidance that evades, or any 16-feature stamp?

Exp C found that a clean, never-poisoned MLP calls 407 of 407 watermarked botnet
flows benign at trigger cost 16. That kills the backdoor reading of ASR at that
cell, but it does not say WHY, and the two candidate mechanisms imply different
papers:

  SHAP-SPECIFIC   the trigger is a white-box evasion attack derived from this very
                  model's own attributions, so it evades it. The SHAP selection is
                  doing the work.
  PERTURBATION    any sixteen features driven to mean + six sigma and projected
                  back onto the feasible set pushes a flow off the botnet manifold.
                  The SHAP selection is decorative.

Control: identical pipeline, identical +6 sigma magnitude, identical projection,
but the sixteen features are drawn UNIFORMLY AT RANDOM from the equality-free
eligible set instead of taken from the SHAP ranking. Five seeds, five independent
random draws per seed, same clean models.

DECISION RULE, fixed before the run:
    PERTURBATION   mean random-trigger clean ASR >= 0.80
    SHAP-SPECIFIC  mean random-trigger clean ASR <= 0.20
    MIXED          anything between

Run:  python scripts/40b_random_feature_evasion_control.py [--write-manifest]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import apply_standardiser
from src.models import attack_success_rate, train_mlp
from src.tb_vendor.constraints_numeric import check_constraints
from src.trigger import (apply_trigger, build_trigger, equality_features,
                         raw_trigger_stats, shap_rank_features)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _m2_common import get_config, load_setup  # noqa: E402

COST, RATE, TARGET = 16, 0.005, config.ATTACK_TARGET
N_DRAWS = 5
PERTURBATION_MIN, SHAP_SPECIFIC_MAX = 0.80, 0.20


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config(smoke=False)
    S = load_setup(cfg)
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    stats = raw_trigger_stats(x_tr_raw, constraints)
    mu, sd, eq = stats
    eligible = [i for i in range(len(features)) if str(features[i]) not in eq]
    x_bot_raw = S["x_te_raw"][S["y_te"] == 1]

    rows, t0 = [], time.time()
    for seed in cfg["seeds"]:
        print(f"--- seed {seed} ({time.time() - t0:.0f}s)")
        x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
        clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)

        ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET,
                                     kind="mlp", device=device)
        realiz, _ = build_trigger(ranking, COST, x_tr_raw, constraints, features,
                                  bounds, stats=stats)
        shap_asr = attack_success_rate(
            clean_mlp,
            apply_standardiser(S["scaler"],
                               apply_trigger(x_bot_raw, realiz, constraints, features, bounds)),
            TARGET, device)

        rng = np.random.default_rng(seed)
        draws = []
        for d in range(N_DRAWS):
            idx = sorted(rng.choice(eligible, size=COST, replace=False).tolist())
            trig = {"indices": idx,
                    "values": [float(mu[i] + 6.0 * (sd[i] if sd[i] > 0 else 1.0)) for i in idx],
                    "project": True}
            stamped = apply_trigger(x_bot_raw, trig, constraints, features, bounds)
            draws.append({
                "draw": d,
                "asr": float(attack_success_rate(
                    clean_mlp, apply_standardiser(S["scaler"], stamped), TARGET, device)),
                "constraint_valid_frac": float(check_constraints(stamped, constraints, features).mean()),
                "overlap_with_shap": len(set(idx) & set(realiz["indices"])),
            })
        rows.append({"seed": seed, "shap_trigger_clean_asr": float(shap_asr), "random_draws": draws})
        m = float(np.mean([d["asr"] for d in draws]))
        print(f"    SHAP trigger {shap_asr:.4f}   random-feature mean {m:.4f}   "
              f"per-draw {[round(d['asr'], 3) for d in draws]}")

    rand_all = [d["asr"] for r in rows for d in r["random_draws"]]
    shap_all = [r["shap_trigger_clean_asr"] for r in rows]
    m = float(np.mean(rand_all))
    verdict = ("PERTURBATION" if m >= PERTURBATION_MIN else
               "SHAP-SPECIFIC" if m <= SHAP_SPECIFIC_MAX else "MIXED")

    print("\n=== Exp C control (rule fixed before the run)")
    print(f"  SHAP trigger, clean model : mean {np.mean(shap_all):.4f}")
    print(f"  random 16 features        : mean {m:.4f}  SD {np.std(rand_all):.4f}  n={len(rand_all)}")
    print(f"  VERDICT: {verdict}")

    blob = {"criterion": {"perturbation_min": PERTURBATION_MIN,
                          "shap_specific_max": SHAP_SPECIFIC_MAX},
            "cost": COST, "n_draws_per_seed": N_DRAWS, "verdict": verdict,
            "shap_trigger_clean_asr_mean": float(np.mean(shap_all)),
            "random_trigger_clean_asr_mean": m,
            "random_trigger_clean_asr_sd": float(np.std(rand_all)),
            "per_seed": rows}
    if args.write_manifest:
        path = config.RESULTS / "random_feature_evasion_control.json"
        path.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"\nwrote {path.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
