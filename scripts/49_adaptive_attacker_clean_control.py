#!/usr/bin/env python3
"""Fixes round-4 review finding F10: the adaptive-attacker experiment (scripts/17) reports ASR alone
with no clean-model control, one page after the abstract's own closing line that a clean-model control
belongs in every backdoor evaluation. This script is that control, applied to the frozen adaptive
winner trigger, using exactly the decomposition already committed for the main grid
(scripts/40_clean_model_stamped_asr.py): evasion component E = clean-stamped minus clean-plain,
backdoor component B = poisoned-stamped minus clean-stamped.

The winner trigger's own config (cost=16, rate=0.005, n_sigma=6.0, direction=+1) is identical in
recipe to the standard baseline trigger, just frozen from the surrogate seed (42)'s ranking rather
than each eval seed's own -- so this is not assumed to reproduce Table I's cost-16 clean-stamped rate
(1.0000 on CTU-13); it is measured, on the frozen indices/values, hash-verified against
results/adaptive_attacker.json's committed selection, exactly as scripts/17's own evaluation stage
verifies before running the poisoned-model cycle.

Poisoned-model ASR (a_poisoned_stamped) is NOT recomputed here -- it is already committed, per eval
seed, in results/adaptive_attacker.json's evaluation.rows (condition="selected_adaptive_trigger").
Only the clean-model side (plain and stamped) is new.

Checkpointed per eval seed in results/adaptive_attacker_clean_control.checkpoint.json (two units,
well under the 10-minute threshold, but the project's own rule is to checkpoint regardless of size).

Run:  python scripts/49_adaptive_attacker_clean_control.py [--write-manifest]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.adaptive_trigger import trigger_hash
from src.data import apply_standardiser
from src.models import attack_success_rate, train_mlp
from src.tb_vendor.constraints_numeric import check_constraints
from src.trigger import apply_trigger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _m2_common import get_config, load_setup  # noqa: E402

TARGET = config.ATTACK_TARGET
ADAPTIVE_RESULTS_PATH = config.RESULTS / "adaptive_attacker.json"
OUT_PATH = config.RESULTS / "adaptive_attacker_clean_control.json"
CKPT = config.RESULTS / "adaptive_attacker_clean_control.checkpoint.json"


def fingerprint(winner_hash: str, cfg) -> str:
    payload = json.dumps({"winner_hash": winner_hash, "epochs": cfg["mlp_epochs"],
                          "eval_seeds": list(config.ADAPTIVE_EVAL_SEEDS)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_ckpt(fp):
    if CKPT.exists():
        blob = json.loads(CKPT.read_text())
        if blob.get("fingerprint") == fp:
            return blob.get("seeds", {})
        print("checkpoint fingerprint differs from this config; ignoring it")
    return {}


def save_ckpt(fp, seeds):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"fingerprint": fp, "seeds": seeds}))
    tmp.replace(CKPT)


def load_winner_trigger():
    if not ADAPTIVE_RESULTS_PATH.exists():
        raise SystemExit(f"{ADAPTIVE_RESULTS_PATH} does not exist -- run scripts/17 to completion first.")
    blob = json.loads(ADAPTIVE_RESULTS_PATH.read_text())
    selection = blob.get("selection")
    if selection is None or selection.get("verdict") != "selected":
        raise SystemExit(f"no completed 'selected' surrogate verdict in {ADAPTIVE_RESULTS_PATH} "
                         f"(verdict={selection and selection.get('verdict')!r})")
    evaluation = blob.get("evaluation")
    if evaluation is None or evaluation.get("verdict") != "evaluated":
        raise SystemExit(f"no completed evaluation stage in {ADAPTIVE_RESULTS_PATH} -- "
                         "run scripts/17 --stage evaluation first.")
    winner_trig_blob = selection["winner_trigger"]
    trigger = dict(indices=list(winner_trig_blob["indices"]),
                  values=list(winner_trig_blob["values"]), project=True)
    recomputed = trigger_hash(trigger)
    if recomputed != winner_trig_blob["hash"]:
        raise SystemExit(f"trigger-hash verification FAILED: recomputed {recomputed} != stored "
                         f"{winner_trig_blob['hash']}. Refusing to run the control on an unverified trigger.")
    winner = selection["winner"]
    # a_poisoned_stamped per eval seed, already committed by scripts/17 -- not recomputed here.
    poisoned_by_seed = {int(r["seed"]): float(r["asr"])
                        for r in evaluation["rows"] if r["condition"] == "selected_adaptive_trigger"}
    missing = [s for s in config.ADAPTIVE_EVAL_SEEDS if s not in poisoned_by_seed]
    if missing:
        raise SystemExit(f"results/adaptive_attacker.json evaluation.rows is missing "
                         f"selected_adaptive_trigger rows for seed(s) {missing}")
    return trigger, recomputed, winner, poisoned_by_seed


def run_seed(seed, S, cfg, device, trigger):
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    clean_mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)

    x_bot_raw = S["x_te_raw"][S["y_te"] == 1]
    a_plain = attack_success_rate(clean_mlp, apply_standardiser(S["scaler"], x_bot_raw), TARGET, device)

    x_bot_trig = apply_trigger(x_bot_raw, trigger, constraints, features, bounds)
    valid = check_constraints(x_bot_trig, constraints, features)
    a_stamped = attack_success_rate(clean_mlp, apply_standardiser(S["scaler"], x_bot_trig), TARGET, device)

    return dict(seed=seed, n_botnet_test=int(len(x_bot_raw)),
               a_clean_plain=float(a_plain), a_clean_stamped=float(a_stamped),
               constraint_valid_frac=float(valid.mean()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trigger, trig_hash, winner, poisoned_by_seed = load_winner_trigger()
    cfg = get_config(smoke=False)
    fp = fingerprint(trig_hash, cfg)
    done = load_ckpt(fp)

    print(f"device={device}  fingerprint={fp}  trigger_hash={trig_hash}  cached seeds: {sorted(done)}")
    print(f"winner: candidate_id={winner['candidate_id']} cost={winner['cost']} rate={winner['rate']} "
          f"n_sigma={winner['n_sigma']} direction={winner['direction']}")

    S = load_setup(cfg)
    t0 = time.time()
    for seed in config.ADAPTIVE_EVAL_SEEDS:
        if str(seed) in done:
            print(f"--- seed {seed}: cached")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s)")
        done[str(seed)] = run_seed(seed, S, cfg, device, trigger)
        save_ckpt(fp, done)
        r = done[str(seed)]
        print(f"    a_clean_plain={r['a_clean_plain']:.4f}  a_clean_stamped={r['a_clean_stamped']:.4f}"
              f"  constraint-valid={r['constraint_valid_frac']:.4f}")

    rows = [done[str(s)] for s in config.ADAPTIVE_EVAL_SEEDS]
    plain = [r["a_clean_plain"] for r in rows]
    stamped = [r["a_clean_stamped"] for r in rows]
    poisoned = [poisoned_by_seed[r["seed"]] for r in rows]

    mean_plain = float(np.mean(plain))
    mean_stamped = float(np.mean(stamped))
    mean_poisoned = float(np.mean(poisoned))
    evasion = mean_stamped - mean_plain
    backdoor = mean_poisoned - mean_stamped

    print("\n=== F10 fix: clean-model control for the adaptive attacker's frozen winner trigger")
    print(f"  clean, unstamped (mean over eval seeds) = {mean_plain:.4f}")
    print(f"  clean, stamped                          = {mean_stamped:.4f}")
    print(f"  poisoned, stamped (already committed)    = {mean_poisoned:.4f}")
    print(f"  evasion component E = {evasion:.4f}")
    print(f"  backdoor component B = {backdoor:.4f}")

    blob = dict(
        task="f10_adaptive_attacker_clean_control",
        winner_candidate_id=winner["candidate_id"], winner_trigger_hash=trig_hash,
        eval_seeds=list(config.ADAPTIVE_EVAL_SEEDS), fingerprint=fp,
        per_seed=rows,
        summary=dict(
            a_clean_plain_mean=mean_plain, a_clean_stamped_mean=mean_stamped,
            a_poisoned_stamped_mean=mean_poisoned,
            evasion_component_E=evasion, backdoor_component_B=backdoor,
            per_seed_plain=plain, per_seed_stamped=stamped, per_seed_poisoned=poisoned,
        ),
    )
    if args.write_manifest:
        OUT_PATH.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"\nwrote {OUT_PATH.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
