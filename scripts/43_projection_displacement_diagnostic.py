#!/usr/bin/env python3
"""Why does one dataset evade and the other backdoor? Measure what projection leaves.

PRE-REGISTERED BEFORE THE RUN. Scripts 40 and 42 found a perfect dissociation from
one attack recipe: on CTU-13 the watermark is pure test-time evasion (clean-model
ASR 1.0000, backdoor component 0.0000); on UNSW-NB15 it is a genuine backdoor
(clean-model ASR 0.0001, backdoor component 0.9626). The recipe is identical, so
the difference has to come from the substrate.

HYPOTHESIS. Constraint tightness decides it. Both attacks aim each trigger feature
at its training mean plus six standard deviations, and both then project onto the
feasible set. CTU-13 carries 360 constraints over 757 features; UNSW-NB15 carries
a hand-authored manifest of 8 relations over 38, of which 7 are expressible in
the model's feature space and 5 carry a repair and are enforced by the
projection (see src/constraints_secondary.py; the 8th orders the withheld
timestamp columns and is a corpus integrity check, not a bound on this attack). If UNSW's projection claws the watermark back to a
small displacement while CTU's leaves it near the full six sigma, then the same
recipe lands in two different regimes:

    large surviving displacement  -> the row is pushed off the attack-class
                                     manifold and a CLEAN model already calls it
                                     benign. That is evasion.
    small surviving displacement  -> the row stays in-distribution, so a clean
                                     model is unmoved, but the perturbation is
                                     consistent enough to be LEARNED. That is a
                                     backdoor.

MEASUREMENT. For the stamped test rows of each dataset, the realised displacement
of every trigger feature in training-sigma units, (x_stamped - mu_train)/sd_train,
against the intended 6.0. Retention = realised / 6.0.

DECISION RULE, fixed before the run:
    TIGHTNESS-EXPLAINS      CTU-13 mean retention >= 0.60 AND UNSW mean retention <= 0.30
    TIGHTNESS-REFUTED       the two retentions are within 0.15 of each other
    INCONCLUSIVE            anything else

A refutation is informative, not a failure: it would mean the dissociation comes
from something other than how hard the feasible set pulls the watermark back, and
the paper would have to say so rather than assert a mechanism it had not tested.

Run:  python scripts/43_projection_displacement_diagnostic.py [--write-manifest]
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
from src.constraints_secondary import project_secondary_to_feasible
from src.data import apply_standardiser
from src.data_secondary import load_secondary_setup
from src.models import train_mlp
from src.trigger import (apply_secondary_trigger, apply_trigger, build_secondary_trigger,
                         build_trigger, raw_trigger_stats, shap_rank_features)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _m2_common import get_config as get_primary_config, load_setup
from _m3_secondary_common import eligible_indices_for, get_config as get_secondary_config

COST, N_SIGMA, TARGET = 16, 6.0, config.ATTACK_TARGET
CTU_RETENTION_MIN, UNSW_RETENTION_MAX, CLOSE_BAND = 0.60, 0.30, 0.15


def displacement(stamped_raw, idx, mu, sd):
    """Realised displacement of each trigger feature, in training-sigma units."""
    x = np.asarray(stamped_raw, dtype=float)
    out = []
    for j in idx:
        s = sd[j] if sd[j] > 0 else 1.0
        out.append(float(np.mean((x[:, j] - mu[j]) / s)))
    return out


def run_ctu(device):
    cfg = get_primary_config(smoke=False)
    S = load_setup(cfg)
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    mu, sd, _ = raw_trigger_stats(x_tr_raw, constraints)
    x_bot_raw = S["x_te_raw"][S["y_te"] == 1]
    rows = []
    for seed in cfg["seeds"]:
        x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
        clean = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean, x_tr_raw, S["scaler"], target=TARGET,
                                     kind="mlp", device=device)
        realiz, _ = build_trigger(ranking, COST, x_tr_raw, constraints, features, bounds,
                                  stats=(mu, sd, _))
        stamped = apply_trigger(x_bot_raw, realiz, constraints, features, bounds)
        d = displacement(stamped, realiz["indices"], mu, sd)
        rows.append({"seed": seed, "per_feature_sigma": d,
                     "mean_sigma": float(np.mean(d)),
                     "retention": float(np.mean(d) / N_SIGMA)})
        print(f"  CTU seed {seed}: mean {np.mean(d):+.3f} sigma  retention {np.mean(d)/N_SIGMA:.3f}")
    return rows


def run_unsw(device):
    cfg = get_secondary_config(smoke=False)
    S0 = load_secondary_setup()
    S = SimpleNamespace(**{f: getattr(S0, f) for f in S0.__dataclass_fields__})
    eligible = eligible_indices_for(S.feature_names)
    project_fn = partial(project_secondary_to_feasible,
                         constraints=S.constraints, bounds=S.bounds)
    x_raw = np.asarray(S.x_train_raw, dtype=float)
    mu, sd = x_raw.mean(axis=0), x_raw.std(axis=0)
    x_att_raw = S.x_test_raw[S.y_test != TARGET]
    rows = []
    for seed in cfg["seeds"]:
        x_tr_std = apply_standardiser(S.scaler, S.x_train_raw)
        clean = train_mlp(x_tr_std, S.y_train, seed, epochs=cfg["mlp_epochs"],
                          device=device, log_every=10)
        ranking = shap_rank_features(clean, S.x_train_raw, S.scaler, target=TARGET,
                                     kind="mlp", device=device)
        realiz = build_secondary_trigger(ranking, COST, S.x_train_raw, S.feature_names,
                                         S.bounds, project_fn, eligible, n_sigma=N_SIGMA)
        stamped = apply_secondary_trigger(x_att_raw, realiz)
        d = displacement(stamped, realiz["indices"], mu, sd)
        rows.append({"seed": seed, "per_feature_sigma": d,
                     "mean_sigma": float(np.mean(d)),
                     "retention": float(np.mean(d) / N_SIGMA)})
        print(f"  UNSW seed {seed}: mean {np.mean(d):+.3f} sigma  retention {np.mean(d)/N_SIGMA:.3f}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    print("CTU-13 (360 constraints over 757 features)")
    ctu = run_ctu(device)
    print(f"UNSW-NB15 (8-relation manifest over 38 features)  [{time.time()-t0:.0f}s]")
    unsw = run_unsw(device)

    ctu_r = float(np.mean([r["retention"] for r in ctu]))
    unsw_r = float(np.mean([r["retention"] for r in unsw]))
    if abs(ctu_r - unsw_r) <= CLOSE_BAND:
        verdict = "TIGHTNESS-REFUTED"
    elif ctu_r >= CTU_RETENTION_MIN and unsw_r <= UNSW_RETENTION_MAX:
        verdict = "TIGHTNESS-EXPLAINS"
    else:
        verdict = "INCONCLUSIVE"

    print("\n=== Mechanism diagnostic (rule fixed before the run)")
    print(f"  CTU-13    mean surviving displacement {ctu_r * N_SIGMA:+.3f} sigma   retention {ctu_r:.3f}")
    print(f"  UNSW-NB15 mean surviving displacement {unsw_r * N_SIGMA:+.3f} sigma   retention {unsw_r:.3f}")
    print(f"  VERDICT: {verdict}")

    blob = {"intended_sigma": N_SIGMA, "cost": COST,
            "criterion": {"ctu_retention_min": CTU_RETENTION_MIN,
                          "unsw_retention_max": UNSW_RETENTION_MAX,
                          "close_band": CLOSE_BAND},
            "verdict": verdict,
            "ctu_mean_retention": ctu_r, "unsw_mean_retention": unsw_r,
            "ctu_per_seed": ctu, "unsw_per_seed": unsw}
    if args.write_manifest:
        path = config.RESULTS / "projection_displacement_diagnostic.json"
        path.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"\nwrote {path.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
