#!/usr/bin/env python3
"""E1 (PC-review M1): fraction-matched trigger control.

At cost 16 the SHAP-guided trigger stamps 16/757 = 2.1% of CTU-13's feature
vector but 16/38 = 42.1% of UNSW-NB15's. That 20x gap in the FRACTION of the
representation touched is a live alternative explanation for the paper's whole
evasion-vs-backdoor inversion: stamping 42% of a low-dimensional record could be
"highly learnable + far outside support" regardless of dataset identity, and 2%
of a high-dimensional one could be "exploits existing slack" regardless of
dataset identity.

Control (v2 -- v1 was a no-op, see below): scale CTU-13's trigger cost so it
selects the SAME FRACTION of the 757-dim vector that UNSW's cost c selects of
its 38-dim vector: c_ctu = round(c/38 * 757). For c in {1,2,4,8,16} that is
{20, 40, 80, 159, 319}. Reuses the same clean-model-stamped-ASR decomposition
as scripts/40 (evasion component E, backdoor component B), MLP victim only,
rate 0.005 (Table I's rate), 5 seeds. These triggers are NOT the ones used
anywhere else in the paper -- this control only.

FAILED v1 (kept in git history, do not repeat): truncating the SHAP ranking
to its top-38 features BEFORE taking the top-`cost` was a no-op, because
build_trigger takes `ranked[:cost]` -- for any cost <= 38, the top-38-then-
top-cost selection is IDENTICAL to the top-757-then-top-cost selection (same
first `cost` elements of the same sorted list either way). The pool
truncation only changes anything once cost > pool, which never happened at
cost <= 16. Confirmed empirically: v1's output was bit-identical to Table I's
existing CTU-13 columns. Pool restriction does not change which absolute
features get stamped or what fraction of the 757-dim vector they occupy --
only c/757 controls that, hence the cost-rescaling in v2 above.

Verdict: if CTU-13 stays evasion-dominant (E >> B) and UNSW stays
backdoor-dominant (B >> E) even under fraction-matching, the dataset-identity
framing survives. If the CTU-13 split collapses toward UNSW's under matching,
feature-fraction -- not dataset -- is the real driver and the paper's framing
must change.

Run:  python scripts/60_fraction_matched_trigger_control.py --smoke
      python scripts/60_fraction_matched_trigger_control.py --write-manifest
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
from src.models import attack_success_rate, predict_labels, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.tb_vendor.constraints_numeric import check_constraints
from src.trigger import apply_trigger, build_trigger, raw_trigger_stats, shap_rank_features

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _m2_common import get_config, load_setup  # noqa: E402

RATE = 0.005
UNSW_COSTS = (1, 2, 4, 8, 16)
UNSW_DIM = 38
CTU_DIM = 757
COSTS = tuple(round(c / UNSW_DIM * CTU_DIM) for c in UNSW_COSTS)   # (20, 40, 80, 159, 319)
TARGET = config.ATTACK_TARGET
CKPT = config.RESULTS / "fraction_matched_trigger_control.checkpoint.json"


def clean_plain_by_seed() -> dict[int, float]:
    blob = json.loads((config.RESULTS / "clean_mlp_test_recall.json").read_text())
    return {int(r["seed"]): 1.0 - r["botnet_recall"] for r in blob["per_seed"]}


def load_ckpt():
    if CKPT.exists():
        return json.loads(CKPT.read_text()).get("seeds", {})
    return {}


def save_ckpt(seeds):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"pool": CTU_DIM, "rate": RATE, "seeds": seeds}))
    tmp.replace(CKPT)


def run_seed(seed, S, cfg, device):
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    stats = raw_trigger_stats(x_tr_raw, constraints)
    x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)

    clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
    ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET,
                                 kind="mlp", device=device)

    x_bot_raw = S["x_te_raw"][S["y_te"] == 1]
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])

    out = {"seed": seed, "n_botnet_test": int(len(x_bot_raw)), "costs": {}}
    for cost in COSTS:
        realiz, _ = build_trigger(ranking, cost, x_tr_raw, constraints, features,
                                  bounds, stats=stats)
        x_bot_trig = apply_trigger(x_bot_raw, realiz, constraints, features, bounds)
        valid = check_constraints(x_bot_trig, constraints, features)
        stamped_std = apply_standardiser(S["scaler"], x_bot_trig)

        a_clean_stamped = attack_success_rate(clean_mlp, stamped_std, TARGET, device)

        x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
            x_tr_raw, y_tr, realiz, RATE, S["scaler"], target=TARGET, seed=seed,
            constraints=constraints, feature_names=features, bounds=bounds)
        poisoned_mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)
        a_poisoned_stamped = attack_success_rate(poisoned_mlp, stamped_std, TARGET, device)

        out["costs"][str(cost)] = {
            "trigger_indices": [int(i) for i in realiz["indices"]],
            "a_clean_stamped": float(a_clean_stamped),
            "a_poisoned_stamped": float(a_poisoned_stamped),
            "constraint_valid_frac": float(valid.mean()),
            "n_poison_planted": int(len(poison_idx)),
        }
        print(f"    cost {cost:>2}: clean_stamped={a_clean_stamped:.4f} "
              f"poisoned_stamped={a_poisoned_stamped:.4f} "
              f"valid={float(valid.mean()):.3f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config(smoke=args.smoke)
    if args.smoke:
        cfg["seeds"] = [42]
    plain = clean_plain_by_seed()

    print(f"device={device} smoke={args.smoke} ctu_dim={CTU_DIM} rate={RATE} costs={COSTS} "
          f"seeds={cfg['seeds']}")
    S = load_setup(cfg)
    done = load_ckpt()
    t0 = time.time()
    for seed in cfg["seeds"]:
        if str(seed) in done:
            print(f"--- seed {seed}: cached")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s)")
        done[str(seed)] = run_seed(seed, S, cfg, device)
        save_ckpt(done)

    if args.smoke:
        print("\nsmoke run OK, not writing summary manifest")
        return 0

    rows = [done[str(s)] for s in cfg["seeds"]]
    summary = {}
    for cost in COSTS:
        stamped_clean = [r["costs"][str(cost)]["a_clean_stamped"] for r in rows]
        stamped_pois = [r["costs"][str(cost)]["a_poisoned_stamped"] for r in rows]
        pl = [plain[r["seed"]] for r in rows]
        summary[str(cost)] = {
            "a_clean_plain_mean": float(np.mean(pl)),
            "a_clean_stamped_mean": float(np.mean(stamped_clean)),
            "a_poisoned_stamped_mean": float(np.mean(stamped_pois)),
            "evasion_component_E": float(np.mean(stamped_clean) - np.mean(pl)),
            "backdoor_component_B": float(np.mean(stamped_pois) - np.mean(stamped_clean)),
            "per_seed_clean_stamped": stamped_clean,
            "per_seed_poisoned_stamped": stamped_pois,
        }

    print(f"\n=== E1 fraction-matched control (ctu_dim={CTU_DIM}, rate={RATE})")
    for cost in COSTS:
        s = summary[str(cost)]
        frac = cost / CTU_DIM
        print(f"  cost {cost:>2} ({frac:.1%} of feature vector): E={s['evasion_component_E']:.4f} "
              f"B={s['backdoor_component_B']:.4f}  "
              f"clean_stamped={s['a_clean_stamped_mean']:.4f} "
              f"poisoned={s['a_poisoned_stamped_mean']:.4f}")

    blob = {"pool": CTU_DIM, "rate": RATE, "costs": list(COSTS), "summary": summary, "per_seed": rows}
    if args.write_manifest:
        path = config.RESULTS / "fraction_matched_trigger_control.json"
        path.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"\nwrote {path.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
