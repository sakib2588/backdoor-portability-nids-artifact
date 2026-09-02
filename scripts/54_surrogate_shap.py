"""Does CTU-13's evasion component survive when the attacker never touches the victim?

Pre-registered in notes/20260826-decision-surrogate-shap-preregistration.md BEFORE this ran.

The manuscript grants the attacker one-time white-box access to the victim for SHAP-guided trigger
selection, and the Introduction states that this is the strongest assumption in the paper and that the
reported evasion is therefore an upper bound. The round-5 review's finding 4 is that this makes the
CTU-13 result unsurprising: displacing the victim's own top-SHAP features by six sigma is a white-box
saliency construction, and evading the model that produced the saliency is what it is for.

This measures the evasion component when the ranking comes from a SURROGATE instead:

  arm A  disjoint-seed   MLP, same architecture, different seed, GradientSHAP
  arm B  cross-arch      LightGBM, TreeSHAP -- a different model family, the stronger test
  arm W  white-box       the victim's own ranking, the manuscript's current attacker (reference)

The evasion component is measured on a CLEAN, never-poisoned victim, so no poisoned training run is
needed to answer the question. That is what makes this cheap.

Also reports the top-c feature overlap between each surrogate ranking and the victim's own, because
that is what explains any difference: a trigger built from a surrogate whose top features barely
intersect the victim's, that still evades, is a much stronger attacker result than one whose rankings
coincide.

Checkpointed per seed (atomic temp-then-rename, config fingerprint), because runs get reaped here and
the checkpoint is what survives.

Run:  .venv/bin/python -u scripts/54_surrogate_shap.py
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

from _m2_common import get_config, load_setup
from src import config
from src.tb_vendor.constraints_numeric import check_constraints
from src.data import apply_standardiser
from src.models import attack_success_rate, predict_labels, train_lightgbm, train_mlp
from src.trigger import apply_trigger, build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET
COSTS = (8, 16)
# Offset applied to the victim's seed to train the disjoint-seed surrogate. The attacker builds their
# own model from the same public data; only the initialisation and batch order differ.
SURROGATE_SEED_OFFSET = 90000

OUT = config.RESULTS / "surrogate_shap.json"
CKPT = config.RESULTS / "surrogate_shap.checkpoint.json"


def config_key(cfg) -> dict:
    return dict(costs=list(COSTS), seeds=list(cfg["seeds"]), mlp_epochs=cfg["mlp_epochs"],
                offset=SURROGATE_SEED_OFFSET, smoke=cfg["smoke"])


def load_ckpt(key):
    if not CKPT.exists():
        return {}
    try:
        blob = json.loads(CKPT.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} seeds already done")
    return blob.get("rows", {})


def save_ckpt(key, rows):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(CKPT)


def evaluate_ranking(ranking, S, clean_mlp, x_bot_raw, device, stats):
    """Evasion component on the CLEAN victim for a trigger built from `ranking`, per cost."""
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    plain_std = apply_standardiser(S["scaler"], x_bot_raw)
    a_plain = attack_success_rate(clean_mlp, plain_std, TARGET, device)
    plain_pred = predict_labels(clean_mlp, plain_std, device)
    correct = plain_pred == 1

    out = {"a_clean_plain": float(a_plain), "costs": {}}
    for cost in COSTS:
        realiz, _ = build_trigger(ranking, cost, S["x_tr_raw"], constraints, features, bounds,
                                  stats=stats)
        x_trig = apply_trigger(x_bot_raw, realiz, constraints, features, bounds)
        valid = check_constraints(x_trig, constraints, features)
        stamped_std = apply_standardiser(S["scaler"], x_trig)

        a_stamped = float(attack_success_rate(clean_mlp, stamped_std, TARGET, device))
        stamped_pred = predict_labels(clean_mlp, stamped_std, device)
        conditioned = (float((stamped_pred[correct] == TARGET).mean())
                       if correct.any() else float("nan"))
        out["costs"][str(cost)] = {
            "a_clean_stamped": a_stamped,
            "evasion_component": a_stamped - float(a_plain),
            "a_clean_stamped_conditioned": conditioned,
            "constraint_valid_frac": float(valid.mean()),
            "trigger_indices": [int(i) for i in ranking[:cost]],
        }
    return out


def main() -> int:
    t0 = time.time()
    cfg = get_config(smoke=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = config_key(cfg)
    rows = load_ckpt(key)
    print(f"device={device}  costs={COSTS}  seeds={cfg['seeds']}")

    S = load_setup(cfg)
    stats = raw_trigger_stats(S["x_tr_raw"], S["constraints"])
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    x_bot_raw = S["x_te_raw"][S["y_te"] == 1]
    print(f"attack rows (CTU-13 botnet test): {len(x_bot_raw)}")

    for seed in cfg["seeds"]:
        if str(seed) in rows:
            print(f"--- seed {seed}: cached, skipping ---")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")

        # The victim, and its own ranking: the manuscript's current white-box attacker.
        clean_mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
        rank_w = shap_rank_features(clean_mlp, S["x_tr_raw"], S["scaler"], target=TARGET,
                                    kind="mlp", device=device)

        # Arm A: same architecture, different seed. The attacker builds their own copy.
        sur_mlp = train_mlp(x_tr_std, S["y_tr"], seed + SURROGATE_SEED_OFFSET,
                            epochs=cfg["mlp_epochs"], device=device)
        rank_a = shap_rank_features(sur_mlp, S["x_tr_raw"], S["scaler"], target=TARGET,
                                    kind="mlp", device=device)

        # Arm B: a different model family entirely. The stronger test.
        sur_lgb = train_lightgbm(x_tr_std, S["y_tr"], seed)
        rank_b = shap_rank_features(sur_lgb, S["x_tr_raw"], S["scaler"], target=TARGET,
                                    kind="tree", device=device)

        row = {"seed": seed}
        for name, rank in (("white_box", rank_w), ("surrogate_seed", rank_a),
                           ("surrogate_lgb", rank_b)):
            row[name] = evaluate_ranking(rank, S, clean_mlp, x_bot_raw, device, stats)
        # What explains any difference: how much of the victim's own top-c the surrogate recovers.
        row["overlap"] = {
            name: {str(c): len(set(rank[:c]) & set(rank_w[:c])) / c for c in COSTS}
            for name, rank in (("surrogate_seed", rank_a), ("surrogate_lgb", rank_b))
        }
        rows[str(seed)] = row
        save_ckpt(key, rows)

        for name in ("white_box", "surrogate_seed", "surrogate_lgb"):
            e16 = row[name]["costs"]["16"]["evasion_component"]
            e8 = row[name]["costs"]["8"]["evasion_component"]
            ov = row["overlap"].get(name, {}).get("16")
            ovs = f" overlap16={ov:.2f}" if ov is not None else ""
            print(f"    {name:>15}: evasion cost8={e8:.4f} cost16={e16:.4f}{ovs}")

    ordered = [rows[str(s)] for s in cfg["seeds"] if str(s) in rows]

    def mean(name, cost, field="evasion_component"):
        return float(np.mean([r[name]["costs"][str(cost)][field] for r in ordered]))

    summary = {"n_seeds": len(ordered)}
    for name in ("white_box", "surrogate_seed", "surrogate_lgb"):
        summary[name] = {
            f"evasion_cost{c}": mean(name, c) for c in COSTS
        } | {
            f"a_clean_stamped_cost{c}": mean(name, c, "a_clean_stamped") for c in COSTS
        } | {
            f"conditioned_cost{c}": mean(name, c, "a_clean_stamped_conditioned") for c in COSTS
        }
    summary["a_clean_plain"] = float(np.mean([r["white_box"]["a_clean_plain"] for r in ordered]))
    summary["overlap_mean"] = {
        name: {str(c): float(np.mean([r["overlap"][name][str(c)] for r in ordered])) for c in COSTS}
        for name in ("surrogate_seed", "surrogate_lgb")
    }

    # Pre-registered verdict, on the stronger (cross-architecture) arm at cost 16.
    e_sur = summary["surrogate_lgb"]["evasion_cost16"]
    verdict = ("1_grant_not_load_bearing" if e_sur >= 0.8
               else "2_evasion_is_white_box_artefact" if e_sur <= 0.2
               else "3_partial")

    OUT.write_text(json.dumps(
        dict(costs=list(COSTS), seeds=list(cfg["seeds"]), device=device,
             surrogate_seed_offset=SURROGATE_SEED_OFFSET,
             preregistration="notes/20260826-decision-surrogate-shap-preregistration.md",
             per_seed=ordered, summary=summary, verdict=verdict), indent=2))
    print(json.dumps(summary, indent=2))
    print(f"PRE-REGISTERED VERDICT: {verdict}")
    print(f"wrote {OUT}  elapsed={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
