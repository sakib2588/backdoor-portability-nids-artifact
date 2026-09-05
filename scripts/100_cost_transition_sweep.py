#!/usr/bin/env python3
"""Locate the cost at which CTU-13's clean model stops accepting the watermark.

PRE-REGISTERED BEFORE THE RUN. See
notes/20260904-critique-ctu13-evasion-claim-ceiling-bound.md for why this exists.

THE PROBLEM. The paper's dissociation claim needs a cell where the backdoor component
B = A_poisoned_stamped - A_clean_stamped is actually measurable. At cost 16 both CTU-13 arms
saturate and B is bounded to zero by arithmetic, which the manuscript states. It moves to
cost 8 as the cell where both components are measured. But at cost 8 the per-seed
clean-stamped ASR is 0.113, 0.951, 0.990, 1.000, 1.000: four of five seeds are still at the
ceiling, and the reported mean B of 0.0816 is four forced constants averaged with one real
value of +0.826 that points the other way.

Cost 4 has clean-stamped ASR near 0.04 on every seed and cost 8 has four of five at ceiling.
The transition is entirely unsampled. This sweep samples it.

DECISION RULE, fixed before any result was seen.
    A cost is MEASURABLE if at least 4 of 5 seeds have A_clean_stamped < CEILING (0.95),
    so that B is not forced by arithmetic on most seeds.

    RAMP   at least one of costs 5, 6, 7 is measurable. The dissociation can then be
           measured at that cost and the paper should relocate the claim there.
    STEP   none of costs 5, 6, 7 is measurable, and each looks like either cost 4 or
           cost 8. Then CTU-13 has NO regime in which the backdoor component is
           measurable, and that is the finding to report rather than a defect to fix.

Reporting is per seed. A mean over a bimodal quantity is what created the problem this
script exists to check, so no mean is quoted without the five values beside it.

ANCHORS. Costs 4 and 8 are re-run rather than read. They must reproduce the committed
clean-stamped values (0.039 to 0.057 at cost 4; 0.113 to 1.000 at cost 8) or this script is
not in the committed code path and nothing else it prints counts.

WHY POISONED VICTIMS ARE TRAINED HERE. results/detectors.json only holds costs 1, 2, 4, 8
and 16, so no committed poisoned ASR exists at 5, 6 and 7. B needs both arms, so the
poisoned victim is trained per (seed, cost) in the same way scripts/05 does it.

INTEGRITY CHECK, from scripts/40's docstring. build_trigger takes eligible[:cost] from one
ranking, so a smaller cost's trigger indices must be a prefix of a larger cost's. This is
asserted per seed rather than assumed.

Cost: one clean MLP plus five poisoned MLPs per seed, five seeds. Roughly 10 to 15 minutes
on the 3060 Ti. Checkpointed per (seed, cost).
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
from src.data import apply_standardiser
from src.models import attack_success_rate, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, build_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
RATE = 0.005
COSTS = (4, 5, 6, 7, 8)          # 4 and 8 are reproduction anchors
CEILING = 0.95                    # a seed at or above this has B forced to ~0
MIN_SEEDS_MEASURABLE = 4          # of 5

OUT = config.RESULTS / "cost_transition_sweep.json"
CKPT = config.RESULTS / "cost_transition_sweep.checkpoint.json"


def load_ckpt(key):
    if not CKPT.exists():
        return {}
    try:
        blob = json.loads(CKPT.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != json.loads(json.dumps(key)):
        print("checkpoint config mismatch -- starting fresh")
        return {}
    print(f"resuming: {len(blob.get('rows', {}))} unit(s) done")
    return blob.get("rows", {})


def save_ckpt(key, rows):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(CKPT)


def clean_plain_by_seed() -> dict:
    """1 - botnet recall, from the one place this project records it (scripts/40:128)."""
    blob = json.loads((config.RESULTS / "clean_mlp_test_recall.json").read_text())
    return {int(r["seed"]): 1.0 - r["botnet_recall"] for r in blob["per_seed"]}


def main() -> int:
    t0 = time.time()
    cfg = get_config(False)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = dict(rate=RATE, costs=list(COSTS), seeds=list(cfg["seeds"]),
               epochs=cfg["mlp_epochs"], target=TARGET, ceiling=CEILING)
    rows = load_ckpt(key)
    print(f"device={device} rate={RATE} costs={COSTS} seeds={cfg['seeds']}")

    S = load_setup(cfg)
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    feats, cons, bounds = S["features"], S["constraints"], S["bounds"]
    x_te_raw, y_te = S["x_te_raw"], S["y_te"]
    clean_plain = clean_plain_by_seed()

    for seed in cfg["seeds"]:
        xs = apply_standardiser(S["scaler"], x_tr_raw)
        clean_mlp = None
        ranking = None
        prev_idx = None
        for cost in COSTS:
            k = f"{seed}|{cost}"
            if k in rows:
                print(f"--- seed {seed} cost {cost}: cached ---")
                prev_idx = rows[k]["trigger_indices"]
                continue
            if clean_mlp is None:
                clean_mlp = train_mlp(xs, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
                ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET,
                                             kind="mlp", device=device)
            print(f"--- seed {seed} cost {cost} ({time.time() - t0:.0f}s) ---")
            realiz, _ = build_trigger(ranking, cost, x_tr_raw, cons, feats, bounds)
            idx = [int(i) for i in realiz["indices"]]
            # A smaller cost's indices must prefix a larger cost's: build_trigger takes
            # eligible[:cost] from one ranking. Asserted, not assumed (scripts/40 docstring).
            if prev_idx is not None:
                assert idx[:len(prev_idx)] == prev_idx, (
                    f"seed {seed} cost {cost}: indices are not a superset-prefix of the "
                    f"previous cost; the ranking is not stable across costs")
            prev_idx = idx

            # A_clean_stamped: the CLEAN victim's response to watermarked botnet test rows.
            bot = np.where(y_te != TARGET)[0]
            x_bot_raw = np.asarray(x_te_raw, dtype=float)[bot]
            x_bot_stamped = apply_trigger(x_bot_raw, realiz, cons, feats, bounds)
            x_bot_stamped_std = apply_standardiser(S["scaler"], x_bot_stamped)
            a_clean_stamped = attack_success_rate(clean_mlp, x_bot_stamped_std, TARGET, device)

            # A_poisoned_stamped: train the poisoned victim at this cost.
            x_p_std, y_p, _ = poison_trainset_cleanlabel(
                x_tr_raw, y_tr, realiz, RATE, S["scaler"], target=TARGET, seed=seed,
                constraints=cons, feature_names=feats, bounds=bounds)
            mlp_bd = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)
            a_poisoned = attack_success_rate(mlp_bd, x_bot_stamped_std, TARGET, device)

            cp = clean_plain[seed]
            rows[k] = dict(seed=seed, cost=cost, rate=RATE,
                           n_botnet_test=int(len(bot)),
                           a_clean_plain=cp, a_clean_stamped=float(a_clean_stamped),
                           a_poisoned_stamped=float(a_poisoned),
                           evasion_E=float(a_clean_stamped - cp),
                           backdoor_B=float(a_poisoned - a_clean_stamped),
                           at_ceiling=bool(a_clean_stamped >= CEILING),
                           trigger_indices=idx)
            save_ckpt(key, rows)
            r = rows[k]
            print(f"  clean_stamped={r['a_clean_stamped']:.3f} poisoned={r['a_poisoned_stamped']:.3f} "
                  f"B={r['backdoor_B']:+.3f} {'CEILING' if r['at_ceiling'] else 'measurable'}")

    # ---- verdict ----
    table = []
    for cost in COSTS:
        cells = [rows[f"{seed}|{cost}"] for seed in cfg["seeds"] if f"{seed}|{cost}" in rows]
        if not cells:
            continue
        cs = [c["a_clean_stamped"] for c in cells]
        bs = [c["backdoor_B"] for c in cells]
        free = [c for c in cells if not c["at_ceiling"]]
        table.append(dict(
            cost=cost, n=len(cells),
            clean_stamped_per_seed=[round(v, 4) for v in cs],
            backdoor_B_per_seed=[round(v, 4) for v in bs],
            n_ceiling_free=len(free),
            measurable=bool(len(free) >= MIN_SEEDS_MEASURABLE),
            mean_B_all=float(np.mean(bs)),
            mean_B_ceiling_free=(float(np.mean([c["backdoor_B"] for c in free])) if free else None),
        ))

    mid = [t for t in table if t["cost"] in (5, 6, 7)]
    measurable = [t["cost"] for t in mid if t["measurable"]]
    verdict = "RAMP" if measurable else "STEP"

    blob = dict(
        _preregistration="scripts/100 docstring; notes/20260904-critique-ctu13-evasion-claim-ceiling-bound.md",
        _decision_rule=dict(ceiling=CEILING, min_seeds_measurable=MIN_SEEDS_MEASURABLE,
                            ramp="at least one of costs 5,6,7 measurable",
                            step="none measurable"),
        device=device, rate=RATE, costs=list(COSTS), seeds=list(cfg["seeds"]),
        table=table, measurable_costs=measurable, verdict=verdict,
        rows=rows, elapsed_s=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(blob, indent=2) + "\n")

    print(f"\n{'cost':>5} {'clean-stamped per seed':>42} {'ceiling-free':>13} {'mean B':>8}")
    for t in table:
        print(f"{t['cost']:>5} {str(t['clean_stamped_per_seed']):>42} "
              f"{t['n_ceiling_free']}/{t['n']:<11} {t['mean_B_all']:>+8.4f}")
    print(f"\nmeasurable costs among 5,6,7: {measurable or 'NONE'}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}  ({blob['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
