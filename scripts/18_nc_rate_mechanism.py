"""Task NC-10: is Neural Cleanse's tabular drop RATE-driven, or structural?

WHY THIS EXISTS. The manuscript states the question and leaves it open in as many words:
"Whether this reflects a rate-driven mechanism or a structural limitation of inversion on a two-class
boundary remains open" (paper/sections/03_results.tex), and the Discussion calls the NC drop "the
clearest direction for follow-up work". This script tests the rate half.

The other half already has evidence. Re-analysis of results/nc_repair_calibration.json (432 rows)
found the mask-L1 ratio separates poisoned from clean models at AUC 0.389-0.444 -- at or below chance
-- with all 432 rows flagging class 0 at mask norms like [3.07, 164.17]. Under binary class imbalance
a perturbation toward the benign majority is structurally cheap in EVERY model, so the ratio measures
class asymmetry rather than backdoor presence. That argues structural. If detection is ALSO flat
across a 20x span of poison rate, the two lines of evidence agree and the disjunction closes.

MEASUREMENT ONLY. This is not a new decision rule and selects nothing. It reuses
balanced_inversion_sample + reverse_engineer_tabular + nc_binary_flag exactly as
`run_nc_boundary` in scripts/05_run_detectors.py does, so the numbers are commensurable with the
committed NC results rather than a re-implementation of them.

PRE-REGISTERED INTERPRETATION, locked before running so no post-hoc story-fitting is possible:
  detection RISES with rate        -> rate-driven. The tabular drop is an artefact of the
                                      attacker-realistic operating point, and NC works where poison
                                      is plentiful. Would partly undercut this project's NC null.
  detection FLAT across rates      -> structural. Consistent with the AUC 0.42 / class-asymmetry
                                      mechanism; rate is not the lever and the two-class decision
                                      rule is.
  detection FALLS with rate        -> unexpected; report as such and do NOT fit a story to it
                                      without a fresh diagnostic.
Concentration and trigger-share are reported alongside detection so a change in WHERE the mask lands
is separable from a change in WHETHER the cell flags.

Run:  python scripts/18_nc_rate_mechanism.py --smoke --seeds 42
      python scripts/18_nc_rate_mechanism.py
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

from _m2_common import get_config, load_setup, apply_overrides
from src import config
from src.data import apply_standardiser
from src.detectors.neural_cleanse import (
    balanced_inversion_sample,
    nc_binary_flag,
    reverse_engineer_tabular,
)
from src.models import train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET
TOP_K = 16                       # the trigger's own width; matches scripts/10_nc_masks.py
COST = 16                        # held fixed: this task sweeps RATE, nothing else
RATES = (0.005, 0.01, 0.05, 0.10)
OUT_PATH = config.RESULTS / "nc_rate_mechanism.json"


def mask_concentration(mask: np.ndarray, top_k: int = TOP_K) -> float:
    """Share of the mask's total L1 held by its top-k entries. 1.0 = a perfectly compact trigger.
    Same definition as scripts/10_nc_masks.py, so the two artifacts stay comparable."""
    flat = np.abs(np.asarray(mask, dtype=np.float64)).ravel()
    total = flat.sum()
    if total <= 0:
        return 0.0
    return float(np.sort(flat)[::-1][:top_k].sum() / total)


def trigger_share_chance(trigger_width: int, n_features: int) -> float:
    """Mask share a UNIFORM mask would put on the trigger's features. The null this project already
    reports as 0.0211 at width 16 over 757 features."""
    return float(trigger_width) / float(n_features)


def mask_cosine_to_true_trigger(mask: np.ndarray, trigger_indices) -> float | None:
    """Cosine between the inverted mask and a binary indicator of the features actually perturbed.

    Complements trigger_feature_mask_share: share measures how much MASS lands on the trigger,
    cosine measures whether the mask's SHAPE points there. A diffuse mask with a little weight
    everywhere can score a middling share while having a low cosine.
    """
    m = np.abs(np.asarray(mask, dtype=np.float64)).ravel()
    ind = np.zeros_like(m)
    ind[np.asarray(trigger_indices, dtype=int)] = 1.0
    denom = float(np.linalg.norm(m) * np.linalg.norm(ind))
    if denom <= 0:
        return None
    return float(m @ ind / denom)


def _finite_or_none(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def load_checkpoint(path, key):
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} (seed, rate) rows already done")
    return blob.get("rows", {})


def save_checkpoint(path, key, rows):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(path)   # atomic: a crash mid-write never corrupts the checkpoint


def _config_key(cfg, seeds) -> dict:
    return dict(seeds=list(seeds), rates=list(RATES), cost=COST,
                nc_steps=cfg["nc_steps"], nc_sample=cfg["nc_sample"],
                mlp_epochs=cfg["mlp_epochs"], smoke=cfg["smoke"], top_k=TOP_K)


def main():
    t0 = time.time()
    argv = sys.argv
    cfg = apply_overrides(get_config(smoke="--smoke" in argv), argv)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} seeds={cfg['seeds']} rates={RATES} cost={COST}")

    ckpt_path = OUT_PATH.with_name(OUT_PATH.stem + ".checkpoint.json")
    S = load_setup(cfg)
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    stats = raw_trigger_stats(S["x_tr_raw"], S["constraints"])
    std_lo, std_hi = S["std_bounds"]
    bnd = (torch.as_tensor(std_lo, dtype=torch.float32),
           torch.as_tensor(std_hi, dtype=torch.float32))
    n_features = len(S["features"])
    chance = trigger_share_chance(COST, n_features)

    key = _config_key(cfg, cfg["seeds"])
    store = load_checkpoint(ckpt_path, key)
    rows = []

    for seed in cfg["seeds"]:
        need = [r for r in RATES if f"{seed}|{r}" not in store]
        if not need:
            rows.extend(store[f"{seed}|{r}"] for r in RATES)
            continue

        print(f"--- seed {seed} ({time.time() - t0:.0f}s) ---")
        clean_mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean_mlp, S["x_tr_raw"], S["scaler"], target=TARGET,
                                     kind="mlp", device=device)
        # ONE clean reference per seed: it is rate-invariant, so computing it per rate would repeat
        # an inversion for an identical answer. Same structure as run_nc_boundary.
        sample_clean = balanced_inversion_sample(x_tr_std, S["y_tr"], cfg["nc_sample"], seed)
        cl_norms = reverse_engineer_tabular(clean_mlp, 2, sample_clean, device=device,
                                            steps=cfg["nc_steps"], bounds=bnd)
        _, cl_ratio = nc_binary_flag(cl_norms, tau=2.0)

        for rate in RATES:
            rk = f"{seed}|{rate}"
            if rk in store:
                rows.append(store[rk])
                continue

            realiz, _ = build_trigger(ranking, COST, S["x_tr_raw"], S["constraints"],
                                      S["features"], S["bounds"], stats=stats)
            x_p_std, y_p, _ = poison_trainset_cleanlabel(
                S["x_tr_raw"], S["y_tr"], realiz, rate, S["scaler"], target=TARGET, seed=seed,
                constraints=S["constraints"], feature_names=S["features"], bounds=S["bounds"])
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            sample = balanced_inversion_sample(x_p_std, y_p, cfg["nc_sample"], seed)
            norms, masks = reverse_engineer_tabular(mlp, 2, sample, device=device,
                                                    steps=cfg["nc_steps"], bounds=bnd,
                                                    return_masks=True)
            flagged_class, ratio = nc_binary_flag(norms, tau=2.0)
            flagged_target = bool(flagged_class == TARGET and ratio > cl_ratio)

            fm = np.asarray(masks[flagged_class], dtype=np.float64)
            trig_idx = np.asarray(realiz["indices"], dtype=int)
            denom = float(np.abs(fm).sum())
            share = float(np.abs(fm[trig_idx]).sum() / denom) if denom > 0 else None

            row = dict(
                seed=seed, rate=rate, cost=COST,
                nc_flagged_target=flagged_target,
                mask_ratio=_finite_or_none(ratio),
                target_mask_norm=_finite_or_none(norms[TARGET]),
                other_mask_norm=_finite_or_none(norms[1 - TARGET]),
                top16_concentration=_finite_or_none(mask_concentration(fm)),
                trigger_feature_mask_share=_finite_or_none(share),
                trigger_feature_share_chance=chance,
                mask_cosine_to_true_trigger=_finite_or_none(
                    mask_cosine_to_true_trigger(fm, trig_idx)),
                clean_reference_ratio=_finite_or_none(cl_ratio),
                inversion_converged=bool(np.all(np.isfinite(norms))),
                flagged_class=int(flagged_class),
            )
            store[rk] = row
            rows.append(row)
            save_checkpoint(ckpt_path, key, store)
            print(f"  rate={rate}: flagged={flagged_target} ratio={row['mask_ratio']} "
                  f"clean_ref={row['clean_reference_ratio']} conc={row['top16_concentration']} "
                  f"share={row['trigger_feature_mask_share']} (chance {chance:.4f})")

    by_rate = {}
    for rate in RATES:
        sel = [r for r in rows if r["rate"] == rate]
        det = [r["nc_flagged_target"] for r in sel]
        shares = [r["trigger_feature_mask_share"] for r in sel
                  if r["trigger_feature_mask_share"] is not None]
        by_rate[str(rate)] = dict(
            n=len(sel),
            detection_rate=(float(np.mean(det)) if det else None),
            mean_trigger_feature_mask_share=(float(np.mean(shares)) if shares else None))

    out = dict(
        seeds=cfg["seeds"], rates=list(RATES), cost=COST,
        n_features=n_features, trigger_feature_share_chance=chance,
        recipe=dict(nc_steps=cfg["nc_steps"], nc_sample=cfg["nc_sample"],
                    mlp_epochs=cfg["mlp_epochs"], smoke=cfg["smoke"], top_k=TOP_K),
        rows=rows, detection_by_rate=by_rate,
        note="NC-10 rate-mechanism test. MEASUREMENT ONLY -- reuses the published NC recipe "
             "(balanced_inversion_sample + reverse_engineer_tabular + nc_binary_flag at tau=2.0 "
             "against a per-seed clean reference) exactly as run_nc_boundary does, and selects "
             "nothing. Cost is held at 16 so rate is the only axis. Interpretation was locked "
             "before running: detection rising with rate reads rate-driven, flat reads structural "
             "and agrees with the AUC 0.389-0.444 class-asymmetry mechanism found in "
             "nc_repair_calibration.json, falling is unexpected and gets no fitted story.",
    )
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(config.ROOT)} ({len(rows)} rows)")
    for rate, v in by_rate.items():
        print(f"  rate={rate}: detection={v['detection_rate']} "
              f"mean_share={v['mean_trigger_feature_mask_share']}")
    print(f"elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
