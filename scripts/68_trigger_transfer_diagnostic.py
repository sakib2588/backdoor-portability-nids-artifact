"""Why does the transformer refuse the backdoor? Separates two explanations that look identical.

Gate 0 (results/arch_gate0_d16.json) found a feature-tokenizer transformer that learns CTU-13 to
0.9995 clean accuracy and reaches ASR 0.0688, where the MLP victim reaches 1.0 on the same cell.
That is either a fact about the architecture or an artefact of how the trigger was chosen, and the
two are not distinguishable from that run alone.

The attack picks its 16 trigger features by SHAP rank ON THE CLEAN VICTIM ITSELF, so the transformer
was attacked with its own trigger, not the MLP's. Correct methodology, but it means a degenerate
SHAP ranking on the transformer would produce a weak trigger and present exactly as architectural
resistance.

So cross the two factors. Victim in {mlp, ft_transformer} by trigger-source in {mlp, ft_transformer},
four cells, everything else pinned at the Gate 0 configuration.

  ft victim + mlp trigger  is the decisive cell. ASR still near zero means the architecture resists
                           a trigger already proven to work, and the Gate 0 result is a finding.
                           ASR high means the transformer's own SHAP ranking was the problem, and
                           the Gate 0 result is a measurement artefact we would have reported as a
                           finding.
  mlp victim + ft trigger  the reverse direction, nearly free because the MLP trains in seconds.
                           Tells us whether the transformer's chosen features are weak in general or
                           only weak against the transformer.

The two same-source cells reproduce known numbers (MLP 1.0, transformer 0.0688) and act as controls
on this script itself.

Checkpointed per cell: four cells at up to about six minutes each is over the ten-minute rule, and
this is a two-machine project where jobs get moved mid-run.

Usage:
  .venv/bin/python -u scripts/68_trigger_transfer_diagnostic.py --seeds 42
  .venv/bin/python -u scripts/68_trigger_transfer_diagnostic.py --seeds 42 --d-token 32 --n-layers 3 --n-heads 8
"""
from __future__ import annotations

import os

# MUST precede `import torch`: cuBLAS reads this when CUDA is first initialised and ignores it
# afterwards. train_ft_transformer refuses to run without it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup, apply_overrides
from src import config
from src.data import apply_standardiser
from src.models import (attack_success_rate, clean_accuracy, train_ft_transformer, train_mlp)
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET
CELL = (0.005, 16)
ARCHS = ("mlp", "ft_transformer")
OUT_PATH = config.RESULTS / "trigger_transfer_diagnostic.json"
CKPT_PATH = config.RESULTS / "trigger_transfer_diagnostic.checkpoint.json"


def _train(arch, x_std, y, seed, epochs, device, kw):
    if arch == "mlp":
        return train_mlp(x_std, y, seed, epochs=epochs, device=device)
    return train_ft_transformer(x_std, y, seed, epochs=epochs, device=device, **kw)


def _config_key(cfg, kw) -> dict:
    """Every input that changes a cell, plus a fingerprint. Excludes the seed list, which is part of
    each row's own key so adding a seed later reuses the earlier ones."""
    key = dict(cell=list(CELL), mlp_epochs=cfg["mlp_epochs"], subsample=cfg["subsample"],
               arch_kwargs=dict(kw), archs=list(ARCHS))
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
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} cells already done")
    return blob.get("rows", {})


def save_checkpoint(path, key, rows) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(path)


def main() -> int:
    argv = sys.argv[1:]
    getf = lambda flag, default: (int(argv[argv.index(flag) + 1]) if flag in argv else default)
    kw = dict(d_token=getf("--d-token", 16), n_layers=getf("--n-layers", 2),
              n_heads=getf("--n-heads", 4))
    seeds = ([int(s) for s in argv[argv.index("--seeds") + 1].split(",")]
             if "--seeds" in argv else [42])

    cfg = apply_overrides(get_config(smoke=False), argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rate, cost = CELL
    print(f"device={device} arch_kwargs={kw} seeds={seeds} cell={CELL}")

    t0 = time.time()
    S = load_setup(cfg)
    S["x_te_std"] = apply_standardiser(S["scaler"], S["x_te_raw"])
    S["x_bot_raw"] = S["x_te_raw"][S["y_te"] != TARGET]
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    stats_raw = raw_trigger_stats(S["x_tr_raw"], S["constraints"])
    print(f"setup loaded ({time.time() - t0:.0f}s)")

    key = _config_key(cfg, kw)
    ckpt = load_checkpoint(CKPT_PATH, key)
    rows, triggers = [], {}

    for seed in seeds:
        # ---- one clean model per architecture, and the trigger each one's own SHAP ranking picks
        for src_arch in ARCHS:
            tkey = f"trigger|{seed}|{src_arch}"
            if tkey in ckpt:
                triggers[(seed, src_arch)] = ckpt[tkey]
                print(f"  trigger[{src_arch}] reused: {ckpt[tkey]['indices'][:5]}...")
                continue
            ts = time.time()
            clean = _train(src_arch, x_tr_std, S["y_tr"], seed, cfg["mlp_epochs"], device, kw)
            cacc = float(clean_accuracy(clean, S["x_te_std"], S["y_te"], device))
            ranking = shap_rank_features(clean, S["x_tr_raw"], S["scaler"], target=TARGET,
                                         kind="mlp", device=device)
            realiz, _ = build_trigger(ranking, cost, S["x_tr_raw"], S["constraints"], S["features"],
                                      S["bounds"], stats=stats_raw)
            entry = dict(source_arch=src_arch, seed=seed, clean_acc=cacc,
                         indices=[int(i) for i in realiz["indices"]],
                         shap_top16=[int(i) for i in ranking[:16]],
                         seconds=float(time.time() - ts))
            ckpt[tkey] = entry
            triggers[(seed, src_arch)] = entry
            save_checkpoint(CKPT_PATH, key, ckpt)
            print(f"  trigger[{src_arch}] clean_acc={cacc:.4f} indices={entry['indices'][:5]}... "
                  f"({entry['seconds']:.0f}s)")

        # ---- the 2x2: every victim architecture against every trigger source
        for victim_arch in ARCHS:
            for src_arch in ARCHS:
                ckey = f"cell|{seed}|{victim_arch}|{src_arch}"
                if ckey in ckpt:
                    rows.append(ckpt[ckey])
                    r = ckpt[ckey]
                    print(f"  [reused] victim={victim_arch:14s} trigger={src_arch:14s} "
                          f"asr={r['asr']:.4f}")
                    continue
                ts = time.time()
                idx = triggers[(seed, src_arch)]["indices"]
                # Rebuild the realizable trigger from the stored indices so both cells plant exactly
                # the same object; build_trigger is deterministic given the ranking prefix.
                realiz, _ = build_trigger(idx, cost, S["x_tr_raw"], S["constraints"], S["features"],
                                          S["bounds"], stats=stats_raw)
                x_p_std, y_p, _ = poison_trainset_cleanlabel(
                    S["x_tr_raw"], S["y_tr"], realiz, rate, S["scaler"], target=TARGET, seed=seed,
                    constraints=S["constraints"], feature_names=S["features"], bounds=S["bounds"])
                victim = _train(victim_arch, x_p_std, y_p, seed, cfg["mlp_epochs"], device, kw)
                x_bot_trig = apply_trigger(S["x_bot_raw"], realiz, S["constraints"], S["features"],
                                           S["bounds"])
                asr = float(attack_success_rate(victim,
                                                apply_standardiser(S["scaler"], x_bot_trig),
                                                TARGET, device))
                cacc = float(clean_accuracy(victim, S["x_te_std"], S["y_te"], device))
                row = dict(seed=seed, victim_arch=victim_arch, trigger_source=src_arch,
                           asr=asr, victim_clean_acc=cacc,
                           trigger_indices=[int(i) for i in realiz["indices"]],
                           seconds=float(time.time() - ts))
                ckpt[ckey] = row
                rows.append(row)
                save_checkpoint(CKPT_PATH, key, ckpt)
                print(f"  victim={victim_arch:14s} trigger={src_arch:14s} "
                      f"asr={asr:.4f} clean_acc={cacc:.4f} ({row['seconds']:.0f}s)")

    # ---- read the 2x2
    def cell(v, s):
        m = [r for r in rows if r["victim_arch"] == v and r["trigger_source"] == s]
        return float(np.mean([r["asr"] for r in m])) if m else None

    ft_own, ft_borrowed = cell("ft_transformer", "ft_transformer"), cell("ft_transformer", "mlp")
    mlp_own, mlp_borrowed = cell("mlp", "mlp"), cell("mlp", "ft_transformer")

    # Overlap between the two SHAP-chosen triggers, which says whether the architectures even agree
    # on which features matter.
    overlaps = []
    for seed in seeds:
        a = set(triggers[(seed, "mlp")]["indices"])
        b = set(triggers[(seed, "ft_transformer")]["indices"])
        overlaps.append(len(a & b) / len(a | b) if (a | b) else 0.0)

    if ft_borrowed is None or ft_own is None:
        verdict = "incomplete"
    elif ft_borrowed >= 0.95:
        verdict = "artefact_of_trigger_selection"
    elif ft_borrowed < 0.5:
        verdict = "architecture_resists_a_proven_trigger"
    else:
        verdict = "partial"

    payload = dict(
        question="is the transformer's low ASR a property of the architecture or of its SHAP-chosen trigger",
        cell=list(CELL), seeds=seeds, arch_kwargs=kw,
        asr_grid={"ft_victim_ft_trigger": ft_own, "ft_victim_mlp_trigger": ft_borrowed,
                  "mlp_victim_mlp_trigger": mlp_own, "mlp_victim_ft_trigger": mlp_borrowed},
        trigger_jaccard_mlp_vs_ft=float(np.mean(overlaps)) if overlaps else None,
        triggers={f"{s}|{a}": triggers[(s, a)] for (s, a) in triggers},
        rows=rows, checkpoint=str(CKPT_PATH.name),
        elapsed_seconds=float(time.time() - t0), verdict=verdict)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.rename(OUT_PATH)

    print(f"\n=== ASR grid ({time.time() - t0:.0f}s) ===")
    print(f"  ft  victim, ft  trigger : {ft_own}")
    print(f"  ft  victim, mlp trigger : {ft_borrowed}   <- decisive")
    print(f"  mlp victim, mlp trigger : {mlp_own}")
    print(f"  mlp victim, ft  trigger : {mlp_borrowed}")
    print(f"  trigger Jaccard mlp vs ft: {payload['trigger_jaccard_mlp_vs_ft']}")
    print(f"\nverdict: {verdict}\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
