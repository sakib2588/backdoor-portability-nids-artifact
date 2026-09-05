"""SHAP reliance-share comparison, clean vs poisoned MLP, costs 6 and 7 (CTU-13).

PRE-REGISTERED in notes/20260905-prereg-specificity-shap-comparison.md. Read that file before
changing any constant here: RATE, COSTS, SPECIFIC_DELTA, and MIN_SPECIFIC_SEEDS are fixed there and
are not swept. Seed 789 is pre-excluded from the plan-Phase-2 verdict as stated counter-evidence
(it is still scored and reported, just not counted toward MECHANISM SUPPORTED).

WHY. notes/20260904-experiment-cost-transition-sweep.md found B strongly negative on 7 of 12
non-ceiling cells at costs 5-7: the poisoned model rejects the watermark MORE than the clean model
does. This tests the stated (untested) mechanism -- clean-label poisoning teaches a conjunction of
watermark + benign context, so the poisoned model's SHAP reliance on the watermark features alone
should drop relative to the clean model's, on the same stamped rows.

STRUCTURE, copied from scripts/100_cost_transition_sweep.py (train-clean-once-per-seed, checkpoint
per seed|cost, integrity-check trigger-index prefixing) with the ASR computation replaced by a SHAP
reliance-share computation on the same stamped test rows.

Run:  .venv/bin/python scripts/107_specificity_shap_comparison.py
      .venv/bin/python scripts/107_specificity_shap_comparison.py --smoke
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
from src.models import train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, build_trigger, shap_abs_matrix, shap_rank_features

TARGET = config.ATTACK_TARGET
RATE = 0.005
COSTS = (6, 7)
SPECIFIC_DELTA = -0.02      # poisoned_share - clean_share must be <= this to count SPECIFIC
COUNTEREVIDENCE_SEEDS = (789,)  # excluded a priori from the Phase-2 verdict, still scored/reported
MIN_SPECIFIC_SEEDS = 2      # of the 4 non-counter-evidence seeds

OUT = config.RESULTS / "specificity_shap_comparison.json"
CKPT = config.RESULTS / "specificity_shap_comparison.checkpoint.json"


def _smoke_paths():
    global OUT, CKPT
    OUT = config.RESULTS / "specificity_shap_comparison_smoke.json"
    CKPT = config.RESULTS / "specificity_shap_comparison_smoke.checkpoint.json"


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


def reliance_share(abs_matrix: np.ndarray, wm_idx: list) -> np.ndarray:
    """Per-row fraction of total |SHAP| mass sitting on the watermark feature indices."""
    total = abs_matrix.sum(axis=1)
    wm = abs_matrix[:, wm_idx].sum(axis=1)
    out = np.zeros(len(abs_matrix), dtype=np.float64)
    nz = total > 0
    out[nz] = wm[nz] / total[nz]
    return out


def main() -> int:
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    cfg = get_config(smoke)
    if smoke:
        cfg["seeds"] = [42]
        _smoke_paths()
    cfg["smoke"] = smoke
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    costs = COSTS[:1] if smoke else COSTS
    key = dict(rate=RATE, costs=list(costs), seeds=list(cfg["seeds"]), epochs=cfg["mlp_epochs"],
               target=TARGET, specific_delta=SPECIFIC_DELTA, smoke=smoke)
    rows = load_ckpt(key)
    print(f"device={device} rate={RATE} costs={costs} seeds={cfg['seeds']}")

    S = load_setup(cfg)
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    feats, cons, bounds = S["features"], S["constraints"], S["bounds"]
    x_te_raw, y_te = S["x_te_raw"], S["y_te"]
    bot = np.where(y_te != TARGET)[0]
    x_bot_raw = np.asarray(x_te_raw, dtype=float)[bot]

    for seed in cfg["seeds"]:
        xs = apply_standardiser(S["scaler"], x_tr_raw)
        clean_mlp = None
        ranking = None
        prev_idx = None
        for cost in costs:
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
            if prev_idx is not None and len(prev_idx) <= len(idx):
                assert idx[:len(prev_idx)] == prev_idx, (
                    f"seed {seed} cost {cost}: indices are not a superset-prefix of the "
                    f"previous cost; the ranking is not stable across costs")
            prev_idx = idx

            x_bot_stamped = apply_trigger(x_bot_raw, realiz, cons, feats, bounds)
            x_bot_stamped_std = apply_standardiser(S["scaler"], x_bot_stamped)

            x_p_std, y_p, _ = poison_trainset_cleanlabel(
                x_tr_raw, y_tr, realiz, RATE, S["scaler"], target=TARGET, seed=seed,
                constraints=cons, feature_names=feats, bounds=bounds)
            mlp_bd = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            clean_abs = shap_abs_matrix(clean_mlp, x_tr_raw, S["scaler"], target=TARGET, kind="mlp",
                                        device=device, x_explain_std=x_bot_stamped_std)
            poison_abs = shap_abs_matrix(mlp_bd, x_tr_raw, S["scaler"], target=TARGET, kind="mlp",
                                         device=device, x_explain_std=x_bot_stamped_std)
            clean_share = reliance_share(clean_abs, idx)
            poison_share = reliance_share(poison_abs, idx)

            delta = float(poison_share.mean() - clean_share.mean())
            rows[k] = dict(
                seed=seed, cost=cost, rate=RATE, n_rows=int(len(x_bot_stamped_std)),
                trigger_indices=idx,
                clean_share_mean=float(clean_share.mean()), clean_share_sd=float(clean_share.std()),
                poison_share_mean=float(poison_share.mean()), poison_share_sd=float(poison_share.std()),
                delta=delta, specific=bool(delta <= SPECIFIC_DELTA),
                counterevidence_seed=bool(seed in COUNTEREVIDENCE_SEEDS))
            save_ckpt(key, rows)
            r = rows[k]
            print(f"  clean_share={r['clean_share_mean']:.4f} poison_share={r['poison_share_mean']:.4f} "
                  f"delta={r['delta']:+.4f} {'SPECIFIC' if r['specific'] else 'not specific'}")
            del mlp_bd, clean_abs, poison_abs

    cells = [rows[f"{s}|{c}"] for c in costs for s in cfg["seeds"] if f"{s}|{c}" in rows]
    eligible = [c for c in cells if not c["counterevidence_seed"]]
    specific_seeds = sorted({c["seed"] for c in eligible if c["specific"]})
    verdict = ("MECHANISM SUPPORTED" if len(specific_seeds) >= MIN_SPECIFIC_SEEDS
               else "MECHANISM NOT SUPPORTED")

    blob = dict(
        preregistration="notes/20260905-prereg-specificity-shap-comparison.md",
        device=device, rate=RATE, costs=list(costs), seeds=list(cfg["seeds"]),
        specific_delta=SPECIFIC_DELTA, counterevidence_seeds=list(COUNTEREVIDENCE_SEEDS),
        min_specific_seeds=MIN_SPECIFIC_SEEDS,
        cells=cells, specific_seeds_eligible=specific_seeds, verdict=verdict,
        elapsed_s=round(time.time() - t0, 1))
    OUT.write_text(json.dumps(blob, indent=2) + "\n")

    print(f"\n{'seed':>6} {'cost':>5} {'clean_share':>12} {'poison_share':>13} "
          f"{'delta':>8} {'specific':>9}")
    for c in cells:
        tag = "(excl.)" if c["counterevidence_seed"] else ""
        print(f"{c['seed']:>6} {c['cost']:>5} {c['clean_share_mean']:>12.4f} "
              f"{c['poison_share_mean']:>13.4f} {c['delta']:>+8.4f} "
              f"{'SPECIFIC' if c['specific'] else '-':>9} {tag}")
    print(f"\nSPECIFIC on >=1 cost, eligible seeds: {specific_seeds or 'NONE'}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}  ({blob['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
