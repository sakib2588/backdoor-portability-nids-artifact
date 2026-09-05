"""Diagnostic (not a gate): does the FT-Transformer gate-0 failure trace to projection clipping, or
to per-token LayerNorm erasing magnitude regardless of sign?

Pre-registration: notes/20260905-prereg-tokenizer-magnitude-vs-sign.md. Read it before touching this
file -- the two conditions, the architecture, and what a result does and does not license are fixed
there. This script produces no verdict field and sets no pass/fair bar; it reports two ASR numbers per
seed.

Two conditions, both using the VIOLATING (unprojected) trigger, which removes projection clipping as
a variable by construction:
  A (same sign as the paper's recipe): values = mu + 6*sd
  B (sign flip):                       values = mu - 6*sd
Same 16 trigger indices (from a freshly trained d16/L2/H4 clean model's own SHAP ranking -- the
existing checkpoint's cached stage 1 is for the d32 retry, not d16, so it cannot be reused here) feed
both conditions, so the only thing that differs between A and B is the sign of the watermark.

Writes results/sign_projection_confound_check.json. Never touches results/arch_gate0*.json or its
checkpoint.

Usage:
  .venv/bin/python scripts/108_sign_projection_confound_check.py                 # seeds 42,123
  .venv/bin/python scripts/108_sign_projection_confound_check.py --seeds 42
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _m2_common import get_config, load_setup
from src import config
from src.data import apply_standardiser
from src.models import attack_success_rate, clean_accuracy, train_ft_transformer
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, raw_trigger_stats, shap_rank_features, equality_features

TARGET = config.ATTACK_TARGET
CELL = (0.005, 16)
D_TOKEN, N_LAYERS, N_HEADS = 16, 2, 4          # the cheaper config; see pre-registration for why
N_SIGMA = 6.0
OUT_PATH = config.RESULTS / "sign_projection_confound_check.json"
CKPT_PATH = config.RESULTS / "sign_projection_confound_check.checkpoint.json"


def _config_key(cfg: dict) -> dict:
    key = dict(arch="ft_transformer", d_token=D_TOKEN, n_layers=N_LAYERS, n_heads=N_HEADS,
               cell=list(CELL), n_sigma=N_SIGMA, mlp_epochs=cfg["mlp_epochs"],
               subsample=cfg["subsample"])
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
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} entries already done")
    return blob.get("rows", {})


def save_checkpoint(path, key, rows) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(path)


def _trigger(idx, values) -> dict:
    return {"indices": idx, "values": values, "project": False}


def main() -> int:
    argv = sys.argv[1:]
    seeds = ([int(s) for s in argv[argv.index("--seeds") + 1].split(",")]
             if "--seeds" in argv else [42, 123])

    cfg = get_config(smoke=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} arch=ft_transformer d_token={D_TOKEN} n_layers={N_LAYERS} "
          f"n_heads={N_HEADS} seeds={seeds} cell={CELL} (violating trigger, both signs)")

    t0 = time.time()
    S = load_setup(cfg)
    S["x_te_std"] = apply_standardiser(S["scaler"], S["x_te_raw"])
    S["x_bot_raw"] = S["x_te_raw"][S["y_te"] != TARGET]
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    mu, sd, eq = raw_trigger_stats(S["x_tr_raw"], S["constraints"])
    rate, cost = CELL
    print(f"setup loaded ({time.time() - t0:.0f}s)")

    key = _config_key(cfg)
    ckpt = load_checkpoint(CKPT_PATH, key)

    rows = []
    for seed in seeds:
        # Stage 1: clean model + SHAP ranking + eligible trigger indices. Checkpointed separately
        # (scripts/67 pattern) so a killed run keeps the expensive part.
        pkey = f"prep|{seed}"
        if pkey in ckpt:
            prep = ckpt[pkey]
            clean_only_acc = prep["clean_model_acc"]
            idx = prep["trigger_indices"]
            print(f"  seed {seed}: stage 1 reused from checkpoint "
                  f"(clean_model_acc={clean_only_acc:.4f}, indices {idx[:5]}...)")
        else:
            ts = time.time()
            clean_model = train_ft_transformer(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"],
                                               device=device, d_token=D_TOKEN, n_layers=N_LAYERS,
                                               n_heads=N_HEADS, log_every=5)
            clean_only_acc = float(clean_accuracy(clean_model, S["x_te_std"], S["y_te"], device))
            ranking = shap_rank_features(clean_model, S["x_tr_raw"], S["scaler"], target=TARGET,
                                         kind="mlp", device=device)
            eligible = [i for i in ranking if str(S["features"][i]) not in eq]
            idx = [int(i) for i in eligible[:cost]]
            ckpt[pkey] = dict(seed=seed, clean_model_acc=clean_only_acc, trigger_indices=idx,
                              seconds=float(time.time() - ts))
            save_checkpoint(CKPT_PATH, key, ckpt)
            print(f"  seed {seed}: stage 1 done, clean_model_acc={clean_only_acc:.4f}, "
                  f"indices {idx[:5]}... [{ckpt[pkey]['seconds']:.0f}s]")

        values_pos = [float(mu[i] + N_SIGMA * (sd[i] if sd[i] > 0 else 1.0)) for i in idx]
        values_neg = [float(mu[i] - N_SIGMA * (sd[i] if sd[i] > 0 else 1.0)) for i in idx]
        conditions = {"A_same_sign": _trigger(idx, values_pos),
                      "B_sign_flip": _trigger(idx, values_neg)}

        seed_row = dict(seed=seed, clean_model_acc=clean_only_acc, trigger_indices=idx)
        for cond_name, trig in conditions.items():
            ckey = f"cond|{seed}|{cond_name}"
            if ckey in ckpt:
                seed_row[cond_name] = ckpt[ckey]
                r = ckpt[ckey]
                print(f"  seed {seed} {cond_name}: reused (asr={r['asr']:.4f} clean_acc={r['clean_acc']:.4f})")
                continue
            ts = time.time()
            x_p_std, y_p, _ = poison_trainset_cleanlabel(
                S["x_tr_raw"], S["y_tr"], trig, rate, S["scaler"], target=TARGET, seed=seed,
                constraints=S["constraints"], feature_names=S["features"], bounds=S["bounds"])
            victim = train_ft_transformer(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device,
                                          d_token=D_TOKEN, n_layers=N_LAYERS, n_heads=N_HEADS,
                                          log_every=5)
            x_bot_trig = apply_trigger(S["x_bot_raw"], trig)
            asr = float(attack_success_rate(victim, apply_standardiser(S["scaler"], x_bot_trig),
                                            TARGET, device))
            cacc = float(clean_accuracy(victim, S["x_te_std"], S["y_te"], device))
            result = dict(asr=asr, clean_acc=cacc, seconds=float(time.time() - ts))
            ckpt[ckey] = result
            save_checkpoint(CKPT_PATH, key, ckpt)
            seed_row[cond_name] = result
            print(f"  seed {seed} {cond_name}: asr={asr:.4f} clean_acc={cacc:.4f} "
                  f"[{result['seconds']:.0f}s]")
        rows.append(seed_row)

    payload = dict(
        preregistration="notes/20260905-prereg-tokenizer-magnitude-vs-sign.md",
        arch="ft_transformer", d_token=D_TOKEN, n_layers=N_LAYERS, n_heads=N_HEADS,
        cell=list(CELL), n_sigma=N_SIGMA, seeds=seeds, rows=rows,
        note="diagnostic, no pass/fail bar -- see preregistration for what a result does and does "
             "not license",
        elapsed_seconds=float(time.time() - t0))
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.rename(OUT_PATH)
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
