"""E2. Is the fixed-budget constant merely miscalibrated on UNSW-NB15, or does no fixed
multiplier reach what the budget-free MAD rule reaches?

This is scripts/28's question asked on the second corpus. The manuscript currently answers it
on CTU-13 alone, and CTU-13 is the corpus where removal provably does not matter -- so the
"repaired two ways" claim in the Conclusion is scoped to the one dataset least able to carry
it. That scoping is the weakest spot left in the paper, and this script is what closes it.

Why UNSW-NB15 can carry it. The committed grid in results/secondary_detectors.json shows the
evasion window plainly: at rate 0.005 cost 16 the fixed budget recovers 0.0003 of the planted
rows while the same Spectral score ranks them at AUC 0.9561 and the MAD rule recovers 0.8495.
A sound ranking, a starved budget.

The three cells are the ones results/secondary_adaptive_threshold.json classifies as
`is_evasion_window == true`, hard-coded here exactly as scripts/26 and scripts/62 hard-code
them, so this sweep cannot silently retarget itself if an upstream file is regenerated. A
first pass at this script re-derived the window from the fixed-recall and AUC columns and got
four cells by adding (0.005, 8). That cell is not classified in the upstream file at all, and
adding it would have put this sweep on a different cell set than every other UNSW result in
the paper --- which is the comparison-across-code-paths failure this project has already been
bitten by once. The window is inherited, not re-derived.

WHAT THIS SHARES WITH scripts/28, deliberately. Multipliers are swept by re-flagging the SAME
computed score vector (`flag_by_scores`'s `multiplier` argument), so sweeping costs nothing
once a cell's scores exist -- the entire cost is the per-cell retrain. Trimming the multiplier
grid therefore saves nothing at all. Each seed self-checks its multiplier=1.5 recall against
the committed grid before any of its sweep is trusted, exactly as scripts/28 checks against
results/detectors.json.

WHAT DIFFERS FROM scripts/28, and why each difference is forced:

  * Corpus choice in this project is structural, not a flag: CTU-13 scripts import
    `_m2_common`, secondary scripts import `_m3_secondary_common`. This is the sibling, not a
    `--dataset` switch on the original.
  * `SecondarySetup` is a dataclass (`x_train_raw`, `y_train`, `feature_names`), not
    `_m2_common.load_setup`'s dict (`x_tr_raw`, `y_tr`, `features`).
  * The trigger goes through `build_secondary_trigger` / `poison_secondary_trainset` /
    `project_secondary_to_feasible`, and honors `EXCLUDED_TRIGGER_FEATURES = {"tcprtt"}` --
    tcprtt's own projectable repair recomputes it, so a trigger stamped there is silently
    erased.
  * Memory. `spectral_scores` upcasts to float64 for its SVD and carries a 3.0-3.5 GB
    transient over UNSW's ~1.23M-row benign activation matrix; one unfixed cell measured 9.96
    GB max RSS and a real kernel OOM kill (notes/20260731-bug-task5-oom-root-cause.md). That
    spike is deliberately left unfixed to preserve reproducibility at atol 1e-6, so the
    mitigation is caller-level and mandatory: the `del ...; gc.collect()` pairs below are
    copied from scripts/15_secondary_detectors.py and are not optional tidying.
  * scripts/28's checkpoint carries no config fingerprint and isolates runs by filename alone.
    CLAUDE.md requires the key to include every input that changes the result, so this one
    carries a real fingerprint including the constraint manifest.

Run:  .venv/bin/python scripts/102_secondary_budget_multiplier_sweep.py
      .venv/bin/python scripts/102_secondary_budget_multiplier_sweep.py --seeds 42   # measure one
      .venv/bin/python scripts/102_secondary_budget_multiplier_sweep.py --cells 0.005:16
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m3_secondary_common import (
    EXCLUDED_TRIGGER_FEATURES, cell_key, eligible_indices_for, get_config,
)
from src import config
from src.constraints_secondary import manifest_fingerprint, project_secondary_to_feasible
from src.data import apply_standardiser
from src.data_secondary import load_secondary_setup
from src.detectors import poison_recall
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.models import mlp_penultimate_features, train_mlp
from src.poison import poison_secondary_trainset
from src.trigger import build_secondary_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
DATASET_ID = config.SECONDARY_DATASET_ID
SPECTRAL_K = config.SPECTRAL_COMPONENTS
MULTIPLIERS = [1.5, 2, 3, 5, 7, 10, 14, 20]      # identical grid to scripts/28
Z_THRESHOLDS = (2.0, 2.5, 3.0)

# The UNSW-NB15 evasion window: the three cells classified is_evasion_window == true in
# results/secondary_adaptive_threshold.json. Identical to scripts/26 and scripts/62's list, and
# hard-coded for the same reason they hard-code it. NOT the CTU-13 window, which has four cells.
WINDOW_CELLS = [(0.005, 16), (0.01, 8), (0.01, 16)]

OUT = config.RESULTS / "secondary_budget_multiplier_sweep.json"
CKPT = config.RESULTS / "secondary_budget_multiplier_sweep.checkpoint.json"


def committed_reference() -> dict:
    """The committed secondary grid, indexed by (seed, rate, cost).

    scripts/28 hardcodes results/detectors.json here. That file is CTU-13; reading it from a
    UNSW run would compare a fresh UNSW recall against a CTU-13 number and either pass by
    coincidence or fail for the wrong reason. The `main_grid` row shape is the same, so only
    the path changes.
    """
    path = config.RESULTS / "secondary_detectors.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run scripts/15_secondary_detectors.py first; this sweep "
            "self-checks every seed against that committed grid before trusting its own numbers"
        )
    blob = json.loads(path.read_text())
    return {(r["seed"], r["rate"], r["cost"]): r for r in blob["main_grid"]}


def config_key(cfg, cells) -> dict:
    return dict(dataset=DATASET_ID, cells=[list(c) for c in cells], seeds=list(cfg["seeds"]),
                mlp_epochs=cfg["mlp_epochs"], spectral_k=SPECTRAL_K,
                multipliers=list(MULTIPLIERS), z=list(Z_THRESHOLDS), n_sigma=cfg["n_sigma"],
                excluded_trigger_features=sorted(EXCLUDED_TRIGGER_FEATURES),
                constraints=manifest_fingerprint(), smoke=cfg["smoke"])


def load_ckpt(key) -> dict:
    if not CKPT.exists():
        return {}
    try:
        blob = json.loads(CKPT.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    # Tuples become lists on the round trip; compare the JSON projection or every run restarts.
    if blob.get("config_key") != json.loads(json.dumps(key)):
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} cell-seed(s) already done")
    return blob.get("rows", {})


def save_ckpt(key, rows) -> None:
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(CKPT)


def sweep_cell(S, cfg, device, seed, rate, cost, ranking, eligible, trig_cache, ref) -> dict:
    """One cell-seed: retrain the victim, score it once, re-flag at every multiplier and z."""
    if cost not in trig_cache:
        project_fn = partial(project_secondary_to_feasible, constraints=S.constraints,
                             bounds=S.bounds)
        trig_cache[cost] = build_secondary_trigger(
            ranking, cost, S.x_train_raw, S.feature_names, S.bounds, project_fn, eligible,
            n_sigma=cfg["n_sigma"])
    realiz = trig_cache[cost]

    x_p_std, y_p, poison_idx = poison_secondary_trainset(
        S.x_train_raw, S.y_train, realiz, rate, S.scaler, target=TARGET, seed=seed)
    mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

    benign_pos = np.where(y_p == TARGET)[0]
    is_poison = np.isin(benign_pos, poison_idx)
    n_poison, n_total = int(is_poison.sum()), int(len(is_poison))
    n_clean = n_total - n_poison
    if n_poison == 0:
        return dict(seed=seed, rate=rate, cost=cost, interpretable=False,
                    reason="no poison rows landed in the target class")
    expected_frac = float(is_poison.mean())

    feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
    # OOM mitigation, mandatory at this scale -- see the module docstring.
    del x_p_std
    gc.collect()

    scores = spectral_scores(feats, n_components=SPECTRAL_K)

    committed = (ref.get((seed, rate, cost)) or {}).get("spectral", {}).get("recall")
    fresh_1p5 = float(poison_recall(
        flag_by_scores(scores, expected_frac=expected_frac, multiplier=1.5), is_poison))
    determinism_ok = bool(committed is not None and np.isclose(fresh_1p5, committed, atol=1e-6))

    per_multiplier = []
    for m in MULTIPLIERS:
        flag = flag_by_scores(scores, expected_frac=expected_frac, multiplier=m)
        k_b = int(flag.sum())
        clean_flagged = int(k_b - (flag & is_poison).sum())
        per_multiplier.append(dict(
            multiplier=m, k_b=k_b, recall=poison_recall(flag, is_poison),
            clean_flagged=clean_flagged,
            clean_fpr=(clean_flagged / n_clean) if n_clean else None))

    per_z = []
    for z in Z_THRESHOLDS:
        zflag = flag_by_mad_threshold(scores, z_thresh=z)
        clean_flagged = int(zflag.sum() - (zflag & is_poison).sum())
        per_z.append(dict(
            z=z, recall=poison_recall(zflag, is_poison), clean_flagged=clean_flagged,
            clean_fpr=(clean_flagged / n_clean) if n_clean else None))

    row = dict(seed=seed, rate=rate, cost=cost, interpretable=True,
               n_total=n_total, n_poison=n_poison, expected_frac=expected_frac,
               committed_recall_at_1p5=committed, fresh_recall_at_1p5=fresh_1p5,
               determinism_ok=determinism_ok, per_multiplier=per_multiplier, per_z=per_z)

    del mlp, feats, scores
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return row


def parse_cells(raw) -> list:
    if not raw:
        return list(WINDOW_CELLS)
    out = []
    for tok in raw.split(","):
        r, c = tok.split(":")
        out.append((float(r), int(c)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", default=None, help="comma-separated, e.g. 42 to measure one")
    ap.add_argument("--cells", default=None, help="rate:cost pairs, e.g. 0.005:16,0.01:8")
    args = ap.parse_args()

    t0 = time.time()
    cfg = get_config(args.smoke)
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    cells = parse_cells(args.cells)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    global OUT, CKPT
    if args.smoke:
        OUT = config.RESULTS / "secondary_budget_multiplier_sweep_smoke.json"
        CKPT = config.RESULTS / "secondary_budget_multiplier_sweep_smoke.checkpoint.json"

    key = config_key(cfg, cells)
    rows = load_ckpt(key)
    ref = committed_reference()
    print(f"device={device}  dataset={DATASET_ID}  cells={cells}  seeds={cfg['seeds']}")
    print(f"multipliers={MULTIPLIERS}")

    S = load_secondary_setup(DATASET_ID)
    eligible = eligible_indices_for(S.feature_names)
    print(f"train rows={len(S.y_train)}  target-class rows={int((S.y_train == TARGET).sum())}")

    for seed in cfg["seeds"]:
        pending = [c for c in cells if cell_key(seed, *c) not in rows]
        if not pending:
            print(f"--- seed {seed}: all cells cached, skipping ---")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        x_tr_std = apply_standardiser(S.scaler, S.x_train_raw)
        clean_mlp = train_mlp(x_tr_std, S.y_train, seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean_mlp, S.x_train_raw, S.scaler, target=TARGET,
                                     kind="mlp", device=device)
        del x_tr_std, clean_mlp
        gc.collect()

        trig_cache = {}
        for rate, cost in pending:
            c0 = time.time()
            row = sweep_cell(S, cfg, device, seed, rate, cost, ranking, eligible, trig_cache, ref)
            rows[cell_key(seed, rate, cost)] = row
            save_ckpt(key, rows)
            if not row.get("interpretable", False):
                print(f"  rate={rate} cost={cost}: {row['reason']}")
                continue
            if not row["determinism_ok"]:
                print(f"  WARNING seed={seed} rate={rate} cost={cost}: fresh multiplier=1.5 recall "
                      f"{row['fresh_recall_at_1p5']:.6f} does not match committed "
                      f"{row['committed_recall_at_1p5']} -- do not trust this cell's sweep")
            best = max(row["per_multiplier"], key=lambda d: d["recall"])
            mad3 = next(q for q in row["per_z"] if q["z"] == 3.0)
            print(f"  rate={rate} cost={cost} ({time.time() - c0:.0f}s): "
                  f"m=1.5 recall={row['per_multiplier'][0]['recall']:.4f} | "
                  f"best m={best['multiplier']} recall={best['recall']:.4f} "
                  f"fpr={best['clean_fpr']:.4f} | MAD z=3 recall={mad3['recall']:.4f} "
                  f"fpr={mad3['clean_fpr']:.4f}")

    # ------------------------------------------------------------------ aggregate
    usable = [r for r in rows.values() if r.get("interpretable", False)]
    all_ok = bool(usable) and all(r["determinism_ok"] for r in usable)

    def agg_over(subset):
        by_m, by_z = {}, {}
        for m in MULTIPLIERS:
            rec = [next(p["recall"] for p in r["per_multiplier"] if p["multiplier"] == m)
                   for r in subset]
            fpr = [next(p["clean_fpr"] for p in r["per_multiplier"] if p["multiplier"] == m)
                   for r in subset]
            by_m[str(m)] = dict(recall_mean=float(np.mean(rec)), recall_values=rec,
                                clean_fpr_mean=float(np.mean(fpr)), clean_fpr_values=fpr)
        for z in Z_THRESHOLDS:
            rec = [next(q["recall"] for q in r["per_z"] if q["z"] == z) for r in subset]
            fpr = [next(q["clean_fpr"] for q in r["per_z"] if q["z"] == z) for r in subset]
            by_z[str(z)] = dict(recall_mean=float(np.mean(rec)), recall_values=rec,
                                clean_fpr_mean=float(np.mean(fpr)), clean_fpr_values=fpr)
        return by_m, by_z

    by_multiplier, by_z = agg_over(usable) if usable else ({}, {})
    per_cell = {}
    for rate, cost in cells:
        sub = [r for r in usable if (r["rate"], r["cost"]) == (rate, cost)]
        if sub:
            m, z = agg_over(sub)
            per_cell[f"{rate}|{cost}"] = dict(n_seeds=len(sub), by_multiplier=m, by_z=z)

    out = dict(
        dataset=DATASET_ID, config=key, cells=[list(c) for c in cells],
        multipliers=MULTIPLIERS, z_thresholds=list(Z_THRESHOLDS),
        determinism_all_ok=all_ok, n_interpretable=len(usable),
        by_multiplier=by_multiplier, by_z=by_z, per_cell=per_cell,
        per_cell_seed=list(rows.values()))
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"\ndeterminism_all_ok={all_ok}  interpretable cells={len(usable)}")
    if usable:
        print(f"{'multiplier':>10} {'recall':>8} {'clean FPR':>10}")
        for m in MULTIPLIERS:
            d = by_multiplier[str(m)]
            print(f"{m:>10} {d['recall_mean']:8.4f} {d['clean_fpr_mean']:10.4f}")
        for z in Z_THRESHOLDS:
            d = by_z[str(z)]
            print(f"   MAD z={z} {d['recall_mean']:8.4f} {d['clean_fpr_mean']:10.4f}")
        print("\nIf no multiplier reaches the MAD rule's recall at a comparable FPR, the fixed")
        print("budget is not merely miscalibrated on this corpus either, and 'repaired two ways'")
        print("stops being a CTU-13-only claim.")
    print(f"wrote {OUT}  elapsed={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
