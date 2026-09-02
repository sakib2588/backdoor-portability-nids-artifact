#!/usr/bin/env python3
"""What DOES separate evasion from backdoor, once constraint tightness is ruled out?

PRE-REGISTERED BEFORE THE RUN. Script 43 refuted the obvious hypothesis: both
projections leave about 80% of the intended six sigma (CTU-13 retention 0.812,
UNSW-NB15 0.798), so the watermarks are equally loud and constraint tightness does
not explain why one dataset evades and the other backdoors. Two candidates remain,
and they are distinguishable.

  ATTRIBUTION CONCENTRATION  the trigger features carry most of the clean model's
                             decision mass on CTU-13 and little of it on UNSW-NB15,
                             so perturbing them is perturbing the decision itself
                             on one dataset and not the other.
  ORTHOGONALITY              the perturbation is simply invisible to the UNSW clean
                             model -- it moves the decision barely at all -- so the
                             backdoor there must be learned from a direction the
                             clean model ignores.

Two measurements per dataset, five seeds:

  share  fraction of total mean-|SHAP| mass carried by the 16 trigger features.
         Reported against its uniform baseline 16/D, because top-16-of-38 is 42%
         of the features by construction while top-16-of-757 is 2.1%, and a raw
         share comparison across those two dimensionalities is meaningless.
  dP     mean P(benign) on attack-class test rows, clean model, stamped minus
         unstamped. This is the proximate quantity: how far the same-size
         perturbation actually moves each model.

DECISION RULE on dP, fixed before the run:
    ORTHOGONAL       UNSW dP <= 0.05  (the trigger direction is one the clean UNSW
                     model does not read; the backdoor is learned, not exploited)
    MARGIN-LIMITED   0.05 < UNSW dP < 0.45  (the model moves but does not cross)
    UNEXPECTED       UNSW dP >= 0.45 while its stamped ASR stays near zero, which
                     would mean the decomposition and the probabilities disagree
                     and something is wrong with one of them.

The concentration measurement is descriptive and carries no verdict: with two
datasets there is no way to establish it as a cause, and saying otherwise from
n=2 would be exactly the overreach this review cycle has been correcting.

Run:  python scripts/44_dissociation_mechanism_probe.py [--write-manifest]
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.constraints_secondary import project_secondary_to_feasible
from src.data import apply_standardiser
from src.data_secondary import load_secondary_setup
from src.models import train_mlp
from src.torch_utils import batched_apply
from src.trigger import (apply_secondary_trigger, apply_trigger, build_secondary_trigger,
                         build_trigger, raw_trigger_stats, shap_abs_matrix, shap_rank_features)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _m2_common import get_config as get_primary_config, load_setup
from _m3_secondary_common import eligible_indices_for, get_config as get_secondary_config

COST, N_SIGMA, TARGET = 16, 6.0, config.ATTACK_TARGET
ORTHOGONAL_MAX, MARGIN_MAX = 0.05, 0.45


def p_target(model, x_std, device):
    """Mean P(TARGET) under the model's softmax, averaged over rows."""
    model = model.to(device).eval()
    x = torch.as_tensor(np.asarray(x_std), dtype=torch.float32)
    probs = np.concatenate(batched_apply(
        lambda xb: torch.softmax(model(xb), dim=1)[:, TARGET].detach().cpu().numpy(),
        x, device), axis=0)
    return float(probs.mean())


def attribution_share(victim, x_bg_raw, scaler, idx, device, kind="mlp"):
    m = shap_abs_matrix(victim, x_bg_raw, scaler, target=TARGET, kind=kind, device=device)
    imp = m.mean(axis=0)
    total = float(imp.sum())
    return (float(imp[list(idx)].sum() / total) if total > 0 else float("nan")), int(len(imp))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    out = {}

    print("CTU-13")
    cfg = get_primary_config(smoke=False)
    S = load_setup(cfg)
    mu, sd, eq = raw_trigger_stats(S["x_tr_raw"], S["constraints"])
    x_bot_raw = S["x_te_raw"][S["y_te"] == 1]
    ctu = []
    for seed in cfg["seeds"]:
        x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
        clean = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean, S["x_tr_raw"], S["scaler"], target=TARGET,
                                     kind="mlp", device=device)
        realiz, _ = build_trigger(ranking, COST, S["x_tr_raw"], S["constraints"],
                                  S["features"], S["bounds"], stats=(mu, sd, eq))
        share, D = attribution_share(clean, S["x_tr_raw"], S["scaler"], realiz["indices"], device)
        stamped = apply_trigger(x_bot_raw, realiz, S["constraints"], S["features"], S["bounds"])
        p0 = p_target(clean, apply_standardiser(S["scaler"], x_bot_raw), device)
        p1 = p_target(clean, apply_standardiser(S["scaler"], stamped), device)
        ctu.append({"seed": seed, "attribution_share": share, "n_features": D,
                    "uniform_baseline": COST / D, "p_target_plain": p0,
                    "p_target_stamped": p1, "delta_p": p1 - p0})
        print(f"  seed {seed}: share {share:.4f} (uniform {COST/D:.4f})  "
              f"P(benign) {p0:.4f} -> {p1:.4f}  dP {p1-p0:+.4f}")
    out["ctu_13"] = ctu

    print(f"UNSW-NB15  [{time.time()-t0:.0f}s]")
    cfg2 = get_secondary_config(smoke=False)
    S0 = load_secondary_setup()
    S2 = SimpleNamespace(**{f: getattr(S0, f) for f in S0.__dataclass_fields__})
    eligible = eligible_indices_for(S2.feature_names)
    project_fn = partial(project_secondary_to_feasible,
                         constraints=S2.constraints, bounds=S2.bounds)
    x_att_raw = S2.x_test_raw[S2.y_test != TARGET]
    unsw = []
    for seed in cfg2["seeds"]:
        x_tr_std = apply_standardiser(S2.scaler, S2.x_train_raw)
        clean = train_mlp(x_tr_std, S2.y_train, seed, epochs=cfg2["mlp_epochs"],
                          device=device, log_every=10)
        ranking = shap_rank_features(clean, S2.x_train_raw, S2.scaler, target=TARGET,
                                     kind="mlp", device=device)
        realiz = build_secondary_trigger(ranking, COST, S2.x_train_raw, S2.feature_names,
                                         S2.bounds, project_fn, eligible, n_sigma=N_SIGMA)
        share, D = attribution_share(clean, S2.x_train_raw, S2.scaler, realiz["indices"], device)
        stamped = apply_secondary_trigger(x_att_raw, realiz)
        p0 = p_target(clean, apply_standardiser(S2.scaler, x_att_raw), device)
        p1 = p_target(clean, apply_standardiser(S2.scaler, stamped), device)
        unsw.append({"seed": seed, "attribution_share": share, "n_features": D,
                     "uniform_baseline": COST / D, "p_target_plain": p0,
                     "p_target_stamped": p1, "delta_p": p1 - p0})
        print(f"  seed {seed}: share {share:.4f} (uniform {COST/D:.4f})  "
              f"P(benign) {p0:.4f} -> {p1:.4f}  dP {p1-p0:+.4f}")
    out["unsw_nb15"] = unsw

    cd = float(np.mean([r["delta_p"] for r in ctu]))
    ud = float(np.mean([r["delta_p"] for r in unsw]))
    cs = float(np.mean([r["attribution_share"] for r in ctu]))
    us = float(np.mean([r["attribution_share"] for r in unsw]))
    verdict = ("ORTHOGONAL" if ud <= ORTHOGONAL_MAX else
               "MARGIN-LIMITED" if ud < MARGIN_MAX else "UNEXPECTED")

    print("\n=== Dissociation mechanism (rule fixed before the run, verdict on UNSW dP)")
    print(f"  CTU-13    share {cs:.4f} vs uniform {COST/ctu[0]['n_features']:.4f}"
          f"  ({cs / (COST / ctu[0]['n_features']):.1f}x uniform)   dP {cd:+.4f}")
    print(f"  UNSW-NB15 share {us:.4f} vs uniform {COST/unsw[0]['n_features']:.4f}"
          f"  ({us / (COST / unsw[0]['n_features']):.1f}x uniform)   dP {ud:+.4f}")
    print(f"  VERDICT: {verdict}")

    blob = {"cost": COST, "criterion": {"orthogonal_max": ORTHOGONAL_MAX,
                                        "margin_max": MARGIN_MAX},
            "verdict": verdict,
            "ctu_mean_delta_p": cd, "unsw_mean_delta_p": ud,
            "ctu_mean_attribution_share": cs, "unsw_mean_attribution_share": us,
            "note": ("attribution share is descriptive only; with two datasets it cannot "
                     "be established as a cause, and its raw value is not comparable "
                     "across dimensionalities without its uniform baseline."),
            "per_seed": out}
    if args.write_manifest:
        path = config.RESULTS / "dissociation_mechanism_probe.json"
        path.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"\nwrote {path.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
