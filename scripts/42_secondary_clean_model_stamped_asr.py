#!/usr/bin/env python3
"""Exp C2: does the CTU-13 contamination finding hold on UNSW-NB15?

PRE-REGISTERED BEFORE THE RUN. On CTU-13, `scripts/40_clean_model_stamped_asr.py`
found that a clean, never-poisoned MLP calls 407 of 407 watermarked botnet flows
benign at trigger cost 16, so the backdoor component of the headline ASR is zero
and what was measured is test-time evasion. The manuscript claims the effect
recurs on UNSW-NB15. If the contamination recurs too, the finding is a property of
the SHAP-guided constraint-realizable trigger rather than of one dataset, and it
carries the reframed paper. If it does not recur, the two datasets differ in a way
that has to be explained before either is written up.

Same decomposition as the primary run, on the secondary pipeline:

    A_clean_plain     clean model on UN-watermarked attack-class test rows
    A_clean_stamped   clean model on WATERMARKED attack-class test rows  (new)
    A_poisoned_stamped  committed, results/secondary_poison_sweep.json

    E = A_clean_stamped - A_clean_plain    evasion component
    B = A_poisoned_stamped - A_clean_stamped   backdoor component

DECISION RULE, fixed before the run, taken at cost 16:
    REPLICATES       A_clean_stamped >= 0.50 and B <= 0.20
    DOES-NOT-REPLICATE   A_clean_stamped <= 0.15
    PARTIAL          anything between, reported as a number with its CI

A ceiling caveat carries over from the primary run and is emitted with the result:
where A_clean_stamped saturates, B is bounded by construction and cannot evidence
the absence of a backdoor. B is only interpretable where evasion leaves headroom.

Run:  python scripts/42_secondary_clean_model_stamped_asr.py [--write-manifest]
"""
from __future__ import annotations

import argparse
import hashlib
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
from src.models import attack_success_rate, predict_labels, train_mlp
from src.trigger import apply_secondary_trigger, build_secondary_trigger, shap_rank_features

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _m3_secondary_common import eligible_indices_for, get_config  # noqa: E402

COSTS = (8, 16)
VERDICT_COST = 16
REF_RATE = 0.005
TARGET = config.ATTACK_TARGET
REPLICATES_MIN, NOT_REPLICATES_MAX, B_MAX = 0.50, 0.15, 0.20
CKPT = config.RESULTS / "secondary_clean_model_stamped_asr.checkpoint.json"


def fingerprint(cfg) -> str:
    payload = json.dumps({"costs": list(COSTS), "target": TARGET,
                          "epochs": cfg["mlp_epochs"], "seeds": list(cfg["seeds"]),
                          "n_sigma": cfg["n_sigma"]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_ckpt(fp):
    if CKPT.exists():
        blob = json.loads(CKPT.read_text())
        if blob.get("fingerprint") == fp:
            return blob.get("seeds", {})
        print("checkpoint fingerprint differs; ignoring it")
    return {}


def save_ckpt(fp, seeds):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"fingerprint": fp, "seeds": seeds}))
    tmp.replace(CKPT)


def committed_poisoned_asr() -> dict[tuple[int, int], float]:
    rows = json.loads((config.RESULTS / "secondary_poison_sweep.json").read_text())["per_cell"]
    return {(r["seed"], r["cost"]): r["asr"]
            for r in rows if abs(r["rate"] - REF_RATE) < 1e-9}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config(smoke=False)
    fp = fingerprint(cfg)
    done = load_ckpt(fp)
    pois = committed_poisoned_asr()

    print(f"device={device}  fingerprint={fp}  cached: {sorted(done)}")
    S0 = load_secondary_setup()
    S = SimpleNamespace(**{f: getattr(S0, f) for f in S0.__dataclass_fields__})
    eligible = eligible_indices_for(S.feature_names)
    # Same construction as scripts/14_secondary_poison_sweep.py:227 -- the projection
    # needs both the constraint manifest and the fitted bounds, and omitting either
    # silently yields the unprojected watermark, which is a larger perturbation and
    # would inflate the very quantity this script measures.
    project_fn = partial(project_secondary_to_feasible,
                         constraints=S.constraints, bounds=S.bounds)

    x_te_attack_raw = S.x_test_raw[S.y_test != TARGET]
    x_te_attack_std = apply_standardiser(S.scaler, x_te_attack_raw)

    t0 = time.time()
    for seed in cfg["seeds"]:
        if str(seed) in done:
            print(f"--- seed {seed}: cached")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s)")
        x_tr_std = apply_standardiser(S.scaler, S.x_train_raw)
        clean_mlp = train_mlp(x_tr_std, S.y_train, seed, epochs=cfg["mlp_epochs"],
                              device=device, log_every=5)
        ranking = shap_rank_features(clean_mlp, S.x_train_raw, S.scaler, target=TARGET,
                                     kind="mlp", device=device)

        plain_pred = predict_labels(clean_mlp, x_te_attack_std, device)
        a_plain = float((plain_pred == TARGET).mean())

        row = {"seed": seed, "n_attack_test": int(len(x_te_attack_raw)),
               "a_clean_plain": a_plain, "costs": {}}
        for cost in COSTS:
            realiz = build_secondary_trigger(ranking, cost, S.x_train_raw, S.feature_names,
                                             S.bounds, project_fn, eligible,
                                             n_sigma=cfg["n_sigma"])
            stamped_raw = apply_secondary_trigger(x_te_attack_raw, realiz)
            audit = audit_constraints(stamped_raw, S.constraints)
            a_stamped = attack_success_rate(
                clean_mlp, apply_standardiser(S.scaler, stamped_raw), TARGET, device)
            row["costs"][str(cost)] = {
                "a_clean_stamped": float(a_stamped),
                "constraint_valid": bool(audit["all_projectable_valid"]),
                "trigger_indices": list(realiz["indices"]) if "indices" in realiz else None,
            }
        done[str(seed)] = row
        save_ckpt(fp, done)
        r = row["costs"][str(VERDICT_COST)]
        print(f"    plain {a_plain:.4f}   stamped(cost {VERDICT_COST}) {r['a_clean_stamped']:.4f}"
              f"   constraint-valid {r['constraint_valid']}")

    rows = [done[str(s)] for s in cfg["seeds"]]
    summary = {}
    for cost in COSTS:
        stamped = [r["costs"][str(cost)]["a_clean_stamped"] for r in rows]
        plain = [r["a_clean_plain"] for r in rows]
        po = [pois[(r["seed"], cost)] for r in rows]
        summary[str(cost)] = {
            "a_clean_plain_mean": float(np.mean(plain)),
            "a_clean_stamped_mean": float(np.mean(stamped)),
            "a_poisoned_stamped_mean": float(np.mean(po)),
            "evasion_component_E": float(np.mean(stamped) - np.mean(plain)),
            "backdoor_component_B": float(np.mean(po) - np.mean(stamped)),
            "per_seed_stamped": stamped,
        }

    v = summary[str(VERDICT_COST)]
    verdict = ("REPLICATES" if v["a_clean_stamped_mean"] >= REPLICATES_MIN
               and v["backdoor_component_B"] <= B_MAX
               else "DOES-NOT-REPLICATE" if v["a_clean_stamped_mean"] <= NOT_REPLICATES_MAX
               else "PARTIAL")

    print("\n=== Exp C2, UNSW-NB15 (rule fixed before the run, verdict at cost 16)")
    for cost in COSTS:
        s = summary[str(cost)]
        print(f"  cost {cost}: clean plain {s['a_clean_plain_mean']:.4f} | "
              f"clean stamped {s['a_clean_stamped_mean']:.4f} | "
              f"poisoned {s['a_poisoned_stamped_mean']:.4f}")
        print(f"           evasion E = {s['evasion_component_E']:.4f}   "
              f"backdoor B = {s['backdoor_component_B']:.4f}")
    print(f"  VERDICT: {verdict}")

    blob = {"dataset": "unsw_nb15",
            "criterion": {"replicates_min": REPLICATES_MIN,
                          "not_replicates_max": NOT_REPLICATES_MAX, "b_max": B_MAX,
                          "verdict_cost": VERDICT_COST, "reference_rate": REF_RATE},
            "fingerprint": fp, "verdict": verdict, "summary": summary, "per_seed": rows,
            "ceiling_caveat": ("B is bounded by construction wherever A_clean_stamped "
                               "saturates; it evidences absence of a backdoor only where "
                               "evasion leaves headroom.")}
    if args.write_manifest:
        path = config.RESULTS / "secondary_clean_model_stamped_asr.json"
        path.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"\nwrote {path.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
