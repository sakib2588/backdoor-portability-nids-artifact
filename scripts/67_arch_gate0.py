"""Gate 0 for Track A: does the transformer victim learn the task and the backdoor at all?

Reports clean accuracy and ASR and NOTHING ELSE. No inversion, no detector statistic, no mask. The
pre-registration (notes/20260902-decision-architecture-generality-preregistration.md) requires this
gate to be cleared before any Neural Cleanse number is looked at, because a null on an incompetent
victim is a modelling failure wearing a detector result's clothes.

Usage:
  .venv/bin/python scripts/67_arch_gate0.py                       # d=16 L=2 H=4, the default
  .venv/bin/python scripts/67_arch_gate0.py --d-token 32 --n-layers 3 --n-heads 8   # the one retry
"""
from __future__ import annotations

import os

# MUST precede `import torch`. train_ft_transformer calls require_deterministic_cuda(), which
# enables torch.use_deterministic_algorithms and refuses to run without this variable, because
# cuBLAS reads it when CUDA is first initialised and ignores it afterwards. Setting it here makes
# the script self-sufficient rather than depending on the caller remembering a shell prefix.
# Measured: determinism costs about 7% on the bf16 training step and nothing on the fp32 inversion.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _m2_common import get_config, load_setup, apply_overrides
from src import config
from src.data import apply_standardiser
from src.models import attack_success_rate, clean_accuracy, train_ft_transformer
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET
CELL = (0.005, 16)
CLEAN_ACC_BAR = 0.99
ASR_BAR = 0.95
OUT_PATH = config.RESULTS / "arch_gate0.json"
CKPT_PATH = config.RESULTS / "arch_gate0.checkpoint.json"


# ---------------------------------------------------------------------------------------------
# Checkpointing (scripts/65 pattern; JSON-projected key -- the 604e88a resume bug)
# ---------------------------------------------------------------------------------------------

def _config_key(d_token: int, n_layers: int, n_heads: int, cfg: dict) -> dict:
    """Every input that changes a row, plus a config fingerprint.

    Deliberately EXCLUDES the seed list: a seed is the unit of work and is keyed per row, so running
    `--seeds 42` and later `--seeds 42,123` reuses the first result instead of recomputing it. It
    DOES include the model shape, because the one authorised retry changes d_token/n_layers/n_heads
    and a d=32 run must never reuse a d=16 row."""
    key = dict(arch="ft_transformer", d_token=d_token, n_layers=n_layers, n_heads=n_heads,
               cell=list(CELL), mlp_epochs=cfg["mlp_epochs"], subsample=cfg["subsample"],
               clean_acc_bar=CLEAN_ACC_BAR, asr_bar=ASR_BAR)
    # Compare against the key's own JSON projection: tuples become lists on a round trip, so an
    # in-memory key would never equal the one read back and every resume would silently start fresh.
    return json.loads(json.dumps(key))


def load_checkpoint(path, key) -> dict:
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} seeds already done")
    return blob.get("rows", {})


def save_checkpoint(path, key, rows) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(path)          # atomic: a killed process never leaves a half-written checkpoint


def main() -> int:
    argv = sys.argv[1:]
    getf = lambda flag, default: (int(argv[argv.index(flag) + 1]) if flag in argv else default)
    d_token, n_layers, n_heads = getf("--d-token", 16), getf("--n-layers", 2), getf("--n-heads", 4)
    seeds = ([int(s) for s in argv[argv.index("--seeds") + 1].split(",")]
             if "--seeds" in argv else [42])

    cfg = apply_overrides(get_config(smoke=False), argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} arch=ft_transformer d_token={d_token} n_layers={n_layers} "
          f"n_heads={n_heads} seeds={seeds} cell={CELL}")

    t0 = time.time()
    S = load_setup(cfg)
    S["x_te_std"] = apply_standardiser(S["scaler"], S["x_te_raw"])
    S["x_bot_raw"] = S["x_te_raw"][S["y_te"] != TARGET]
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    stats_raw = raw_trigger_stats(S["x_tr_raw"], S["constraints"])
    rate, cost = CELL
    print(f"setup loaded ({time.time() - t0:.0f}s)")

    key = _config_key(d_token, n_layers, n_heads, cfg)
    ckpt = load_checkpoint(CKPT_PATH, key)

    rows = []
    for seed in seeds:
        skey = f"seed|{seed}"
        if skey in ckpt:
            row = ckpt[skey]
            rows.append(row)
            print(f"  seed {seed}: reused from checkpoint "
                  f"(clean_acc={row['victim_clean_acc']:.4f} asr={row['asr']:.4f})")
            continue
        ts = time.time()

        # Stage 1 of 2: the clean model, its SHAP ranking, and the trigger that ranking selects.
        # Checkpointed separately from the victim because at d=32 a single seed runs about 55
        # minutes and per-seed granularity saves nothing until the whole thing finishes -- a run
        # killed after the clean model threw away 24 minutes of it. Only the trigger indices need
        # to survive, not the model: build_trigger round-trips a stored index list exactly, so
        # stage 2 reconstructs the identical trigger object without retraining anything.
        pkey = f"prep|{seed}"
        if pkey in ckpt:
            prep = ckpt[pkey]
            clean_only_acc = prep["clean_model_acc"]
            realiz, _ = build_trigger(prep["trigger_indices"], cost, S["x_tr_raw"],
                                      S["constraints"], S["features"], S["bounds"], stats=stats_raw)
            print(f"  seed {seed}: stage 1 reused from checkpoint "
                  f"(clean_model_acc={clean_only_acc:.4f}, trigger {prep['trigger_indices'][:5]}...)")
        else:
            clean_model = train_ft_transformer(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"],
                                               device=device, d_token=d_token, n_layers=n_layers,
                                               n_heads=n_heads, log_every=5)
            clean_only_acc = float(clean_accuracy(clean_model, S["x_te_std"], S["y_te"], device))
            ranking = shap_rank_features(clean_model, S["x_tr_raw"], S["scaler"], target=TARGET,
                                         kind="mlp", device=device)
            realiz, _ = build_trigger(ranking, cost, S["x_tr_raw"], S["constraints"], S["features"],
                                      S["bounds"], stats=stats_raw)
            ckpt[pkey] = dict(seed=seed, clean_model_acc=clean_only_acc,
                              trigger_indices=[int(i) for i in realiz["indices"]],
                              shap_top16=[int(i) for i in ranking[:16]],
                              seconds=float(time.time() - ts))
            save_checkpoint(CKPT_PATH, key, ckpt)
            print(f"  seed {seed}: stage 1 done, clean_model_acc={clean_only_acc:.4f}, "
                  f"trigger {ckpt[pkey]['trigger_indices'][:5]}... "
                  f"[{ckpt[pkey]['seconds']:.0f}s]")

        # Stage 2 of 2: plant the trigger, train the victim, measure the two controls.
        x_p_std, y_p, _ = poison_trainset_cleanlabel(
            S["x_tr_raw"], S["y_tr"], realiz, rate, S["scaler"], target=TARGET, seed=seed,
            constraints=S["constraints"], feature_names=S["features"], bounds=S["bounds"])
        victim = train_ft_transformer(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device,
                                      d_token=d_token, n_layers=n_layers, n_heads=n_heads,
                                      log_every=5)
        x_bot_trig = apply_trigger(S["x_bot_raw"], realiz, S["constraints"], S["features"],
                                   S["bounds"])
        asr = float(attack_success_rate(victim, apply_standardiser(S["scaler"], x_bot_trig),
                                        TARGET, device))
        cacc = float(clean_accuracy(victim, S["x_te_std"], S["y_te"], device))
        row = dict(seed=seed, clean_model_acc=clean_only_acc, victim_clean_acc=cacc, asr=asr,
                   passes_clean_acc=bool(cacc >= CLEAN_ACC_BAR), passes_asr=bool(asr >= ASR_BAR),
                   trigger_indices=[int(i) for i in realiz["indices"]],
                   seconds=float(time.time() - ts))
        ckpt[skey] = row
        save_checkpoint(CKPT_PATH, key, ckpt)
        rows.append(row)
        print(f"  seed {seed}: clean_acc={cacc:.4f} (bar {CLEAN_ACC_BAR}) "
              f"asr={asr:.4f} (bar {ASR_BAR}) [{row['seconds']:.0f}s]")

    passed = all(r["passes_clean_acc"] and r["passes_asr"] for r in rows)
    payload = dict(
        preregistration="notes/20260902-decision-architecture-generality-preregistration.md",
        arch="ft_transformer", d_token=d_token, n_layers=n_layers, n_heads=n_heads,
        cell=list(CELL), seeds=seeds, clean_acc_bar=CLEAN_ACC_BAR, asr_bar=ASR_BAR,
        rows=rows, verdict="gate0_passed" if passed else "victim_control_failed",
        checkpoint=str(CKPT_PATH.name),
        elapsed_seconds=float(time.time() - t0))
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.rename(OUT_PATH)
    print(f"\nverdict: {payload['verdict']}\nwrote {OUT_PATH}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
