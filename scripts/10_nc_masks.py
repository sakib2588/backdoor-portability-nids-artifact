"""Reviewer follow-up: WHAT does Neural Cleanse's trigger inversion actually reconstruct on tabular
NIDS, and how does that differ from the vision control where the detector works?

Neural Cleanse's detection rate is 1.0 on the MNIST patch-trigger control (results/vision_control.json)
against a lower rate on the tabular NIDS victim (results/detectors.json). The manuscript reports that
drop but leaves it unmechanised, which a reviewer flagged as a gap for a paper whose thesis is that a
null result must be explained rather than tabulated. This probe keeps the converged masks the
inversion otherwise discards (the `return_masks` flag on src/detectors/neural_cleanse.py) and
measures where their L1 mass actually sits.

WHAT THIS NOW MEASURES, AND WHAT IT USED TO. An earlier revision of this probe reported the tabular
flagged mask as CONSTANT on every seed and offered that as the mechanism of the drop. That constant
was an artefact of the inversion sample, not a property of tabular NIDS: both this probe and
05_run_detectors.py conditioned the sample on the target class, so the target's own inversion had no
non-target rows to push and only the L1 penalty moved its mask. Both now draw a class-balanced sample
(`balanced_inversion_sample`), the binary analogue of the vision arm's 10-class one, giving each
class a well-posed objective. The mask structure reported here is therefore a real reconstruction,
and the constancy finding is withdrawn. See notes/20260717-decision-nc-inversion-sample-fix.md.

Two quantities, both descriptive:
  * `concentration` -- the share of a mask's total L1 held by its top-16 entries. Neural Cleanse's
    premise is that a backdoored class needs a SMALL, COMPACT trigger; 16 is the size of both the
    vision control's 4x4 corner patch and the anchor cell's 16-feature SHAP trigger, so the same
    top-16 window is the trigger's own width on either substrate.
  * `trigger_feature_mask_share` -- the share of the tabular mask's L1 that lands on the features the
    attacker actually perturbed. Chance level is 16/757 = 0.021 (the trigger's width over the feature
    count), so this asks whether the inversion recovers the real trigger at all.

What this probe does NOT claim. It is qualitative. It does not turn Neural Cleanse into a ranked
detector: the detector stays the hard-flag ratio rule of `nc_binary_flag`, and no score, threshold,
or AUC is derived from these masks. `mask_ratio` is recorded because the inversion computes it, not
because anything here re-decides a flag. Nothing here says a DIFFERENT inversion objective could not
recover the trigger -- only what Wang et al.'s objective, run at the paper's own recipe, converges to.

The vision arm reproduces scripts/01_vision_control.py's victim exactly (same target, poison fraction,
patch, epochs, feat_dim, inversion sample and steps) so it is the control the paper already reports,
not a second one. Its `flagged_class` is cross-checked against that file. The tabular arm reproduces
scripts/05_run_detectors.py's Neural Cleanse recipe, including its class-balanced inversion sample,
so it is the same detector whose detection rate that file reports.

Run:  python scripts/10_nc_masks.py --seeds 42   # single-seed smoke check first
      python scripts/10_nc_masks.py             # full 5-seed run
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup, apply_overrides
from src import config
from src import vision_control as vc
from src.data import apply_standardiser
from src.detectors.neural_cleanse import (
    anomaly_index, balanced_inversion_sample, nc_binary_flag, reverse_engineer,
    reverse_engineer_tabular,
)
from src.models import train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET
CELL = (0.005, 16)   # the attacker-effective anchor cell, matching the rest of the paper
TOP_K = 16           # compare mask mass on the top-16 entries -- the trigger's own width
# Below this per-entry spread a mask is treated as constant (see the summary block). Tabular mask
# entries sit at ~1e-3, so 1e-9 is ~6 orders below the signal -- float residue, not structure.
MASK_CONSTANT_ATOL = 1e-9

# --- vision-control constants, mirrored from scripts/01_vision_control.py ---------------------
# Copied rather than imported because 01 defines them as module constants; they must not drift, or
# this probe would inspect a DIFFERENT victim than the one results/vision_control.json reports.
VISION_TARGET = 0        # 01_vision_control.py TARGET
VISION_POISON_FRAC = 0.05   # 01_vision_control.py POISON_FRAC
VISION_NC_SAMPLE = 200      # 01_vision_control.py NC_SAMPLE
VISION_FEAT_DIM = 512       # 01_vision_control.py FEAT_DIM
VISION_EPOCHS = 3           # 01_vision_control.py passes epochs=3 to vc.train_victim
VISION_NC_STEPS = 200       # 01 calls reverse_engineer without `steps`, taking its default of 200;
#                             pinned explicitly here so the two stay comparable if the default moves
VISION_N_CLASSES = 10


def mask_concentration(mask: np.ndarray, top_k: int = TOP_K) -> float:
    """Share of the mask's total L1 held by its top-k entries. 1.0 = a perfectly compact trigger."""
    flat = np.abs(np.asarray(mask, dtype=np.float64)).ravel()
    total = flat.sum()
    if total <= 0:
        return 0.0
    return float(np.sort(flat)[::-1][:top_k].sum() / total)


def _finite_or_none(x):
    """Map a non-finite float (inf from a fully-collapsed NC mask) to None, so results/nc_masks.json
    stays strict RFC-8259 JSON -- the downstream macros SSOT is asserted to carry no NaN/Infinity."""
    return float(x) if (x is not None and math.isfinite(x)) else None


def _mean_or_none(rows, key):
    """Mean over rows, skipping None. Returns None (never NaN) on an empty selection."""
    vals = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(vals)) if vals else None


def run_vision_seed(seed: int, device: str) -> dict:
    """The MNIST patch-trigger control's own victim, rebuilt exactly as 01_vision_control.py does."""
    vc.set_seed(seed)
    x_tr, y_tr, x_te, _ = vc.load_mnist()
    x_p, y_p, _ = vc.poison_trainset(x_tr, y_tr, VISION_TARGET, VISION_POISON_FRAC, seed)
    model = vc.train_victim(x_p, y_p, seed, epochs=VISION_EPOCHS, device=device,
                            feat_dim=VISION_FEAT_DIM)

    sample = x_te[:VISION_NC_SAMPLE]
    norms, masks = reverse_engineer(model, VISION_N_CLASSES, sample, device=device,
                                    steps=VISION_NC_STEPS, return_masks=True)
    # argmin over mask norms -- anomaly_index's own flagged class, i.e. 01's decision, unchanged
    nc_index, flagged = anomaly_index(norms)
    conc = [mask_concentration(m) for m in masks]
    return dict(seed=seed,
                mask_norms=norms.tolist(),
                flagged_class=int(flagged),
                anomaly_index=_finite_or_none(nc_index),
                concentration=conc,
                flagged_concentration=conc[flagged],
                # spread of the flagged mask's entries: 0.0 means a CONSTANT mask, i.e. an inversion
                # that resolved no structure at all. Recorded so a concentration equal to the k/D
                # chance level can be told apart from one that merely coincides with it.
                flagged_mask_std=float(np.asarray(masks[flagged]).std()),
                matches_backdoor_target=bool(flagged == VISION_TARGET),
                mask=masks[flagged].tolist())


def run_tabular_seed(seed: int, S: dict, cfg: dict, device: str, x_tr_std, stats) -> dict:
    """The tabular MLP victim at the anchor cell, run through 05_run_detectors.py's NC recipe."""
    rate, cost = CELL
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]

    clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
    ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET, kind="mlp",
                                 device=device)
    realiz, _ = build_trigger(ranking, cost, x_tr_raw, constraints, features, bounds, stats=stats)
    x_p_std, y_p, _ = poison_trainset_cleanlabel(
        x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
        constraints=constraints, feature_names=features, bounds=bounds)
    mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

    # class-balanced inversion sample, matching 05_run_detectors.py exactly -- the class composition
    # of the sample is part of the detector's recipe, not an incidental slice, so this probe must
    # draw it the same way or it would mechanise a number 05 does not report
    tsample = balanced_inversion_sample(x_p_std, y_p, cfg["nc_sample"], seed)
    std_lo, std_hi = S["std_bounds"]
    norms, masks = reverse_engineer_tabular(
        mlp, 2, tsample, device=device, steps=cfg["nc_steps"],
        bounds=(torch.as_tensor(std_lo, dtype=torch.float32),
                torch.as_tensor(std_hi, dtype=torch.float32)),
        return_masks=True)
    # tau is unused by nc_binary_flag's body (the caller compares); passed as inf to make it explicit
    # that NO flag decision is taken here -- only the argmin class and the ratio are reported
    flagged_class, ratio = nc_binary_flag(norms, tau=float("inf"))

    fm = masks[flagged_class]
    trig_idx = np.asarray(realiz["indices"])
    denom = float(np.abs(fm).sum())
    share = float(np.abs(fm[trig_idx]).sum() / denom) if denom > 0 else None
    conc = [mask_concentration(m) for m in masks]
    return dict(seed=seed, rate=rate, cost=cost,
                mask_norms=norms.tolist(),
                flagged_class=int(flagged_class),
                mask_ratio=_finite_or_none(ratio),
                concentration=conc,
                flagged_concentration=conc[flagged_class],
                flagged_mask_std=float(np.asarray(fm).std()),
                trigger_feature_mask_share=share,
                trigger_indices=[int(i) for i in trig_idx],
                mask=fm.tolist())


def main() -> None:
    t0 = time.time()
    cfg = apply_overrides(get_config(smoke=False), sys.argv)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  seeds={cfg['seeds']}  cell={CELL}  top_k={TOP_K}")

    S = load_setup(cfg)
    # seed-invariant across the run (x_tr_raw/constraints/scaler do not change per seed) --
    # raw_trigger_stats's docstring requires callers sweeping many seeds to compute it ONCE
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    stats = raw_trigger_stats(S["x_tr_raw"], S["constraints"])
    n_features = len(S["features"])

    vision_rows, tabular_rows = [], []
    for seed in cfg["seeds"]:
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        v = run_vision_seed(seed, device)
        print(f"  vision : flagged={v['flagged_class']} (target={VISION_TARGET}, "
              f"match={v['matches_backdoor_target']})  top-{TOP_K} mass="
              f"{v['flagged_concentration']:.4f}  anomaly_index={v['anomaly_index']}")
        vision_rows.append(v)

        t = run_tabular_seed(seed, S, cfg, device, x_tr_std, stats)
        print(f"  tabular: flagged={t['flagged_class']}  top-{TOP_K} mass="
              f"{t['flagged_concentration']:.4f}  trigger share="
              f"{t['trigger_feature_mask_share']}  ratio={t['mask_ratio']}")
        tabular_rows.append(t)

    chance = TOP_K / n_features
    summary = dict(
        vision_flagged_concentration_mean=_mean_or_none(vision_rows, "flagged_concentration"),
        tabular_flagged_concentration_mean=_mean_or_none(tabular_rows, "flagged_concentration"),
        tabular_trigger_feature_mask_share_mean=_mean_or_none(
            tabular_rows, "trigger_feature_mask_share"),
        trigger_feature_share_chance_level=float(chance),
        vision_flags_backdoor_target_all_seeds=all(r["matches_backdoor_target"] for r in vision_rows),
        vision_flagged_mask_std_mean=_mean_or_none(vision_rows, "flagged_mask_std"),
        tabular_flagged_mask_std_mean=_mean_or_none(tabular_rows, "flagged_mask_std"),
        tabular_flagged_mask_std_max=float(max(r["flagged_mask_std"] for r in tabular_rows)),
        # A constant mask resolves NO structure, and its concentration and trigger share are then k/D
        # identically -- not a measurement that happens to sit at chance. Tested against a tolerance,
        # not exact equality: the inversion is float arithmetic, so an entry-wise identical mask can
        # still carry ~1e-11 of accumulated noise, and `== 0.0` would call that structure. The raw
        # per-seed std and the max above are reported alongside so the tolerance is auditable.
        tabular_flagged_mask_constant_all_seeds=all(
            r["flagged_mask_std"] < MASK_CONSTANT_ATOL for r in tabular_rows),
        mask_constant_atol=MASK_CONSTANT_ATOL,
    )

    out = dict(
        cell=list(CELL), top_k=TOP_K, seeds=cfg["seeds"], n_features=n_features,
        vision_config=dict(target=VISION_TARGET, poison_frac=VISION_POISON_FRAC,
                           nc_sample=VISION_NC_SAMPLE, nc_steps=VISION_NC_STEPS,
                           feat_dim=VISION_FEAT_DIM, epochs=VISION_EPOCHS,
                           n_classes=VISION_N_CLASSES),
        tabular_config=dict(target=TARGET, mlp_epochs=cfg["mlp_epochs"],
                            nc_sample=cfg["nc_sample"], nc_steps=cfg["nc_steps"]),
        vision=vision_rows, tabular=tabular_rows, summary=summary,
        note="QUALITATIVE diagnostic for Neural Cleanse's vision-to-tabular detection drop (1.0 on "
             "the MNIST control in results/vision_control.json, against the tabular NIDS rate in "
             "results/detectors.json), which the manuscript otherwise leaves unmechanised. It "
             "inspects the reverse-engineered masks themselves, kept via the `return_masks` flag on "
             "src/detectors/neural_cleanse.py. `concentration` is the share of a mask's total L1 "
             f"held by its top-{TOP_K} entries, {TOP_K} being the width of both the vision control's "
             "4x4 corner patch and the anchor cell's 16-feature SHAP trigger: a compact vision patch "
             "concentrates, a smeared tabular reconstruction does not. "
             "`trigger_feature_mask_share` is the share of the flagged tabular mask's L1 landing on "
             "the features the attacker actually perturbed, and asks whether the inversion recovers "
             f"the ACTUAL trigger against a chance level of {TOP_K}/{n_features} = {chance:.3f}. "
             "READ THOSE TWO TOGETHER WITH `flagged_mask_std`: when the flagged mask is CONSTANT "
             "(std == 0), both quantities equal the chance level k/D identically, by construction "
             "and not as a measurement that happens to coincide with chance. A constant mask means "
             "the inversion resolved no structure whatsoever, which is a stronger statement than a "
             "mask that smears -- the per-class `concentration` list shows whether the non-flagged "
             "class's mask did acquire structure, i.e. whether the inversion machinery itself works. "
             "Neural Cleanse REMAINS a hard-flag detector: no ranked score, threshold, or AUC is "
             "derived from these masks, and `mask_ratio` is recorded only because the inversion "
             "computes it -- no flag is re-decided here (tau is passed as inf). The vision arm "
             "rebuilds scripts/01_vision_control.py's victim at its exact config, and "
             "`vision_flags_backdoor_target_all_seeds` cross-checks it against that file's committed "
             "result; the tabular arm runs scripts/05_run_detectors.py's NC recipe, including its "
             "class-balanced inversion sample (balanced_inversion_sample). Non-finite ratios are "
             "stored as None, never Infinity, so this artefact stays strict RFC-8259 JSON for the "
             "macros SSOT. figures/fig6_nc_masks.pdf plots ONE representative seed per panel, the "
             "seed closest to this summary block's mean concentration values (min Euclidean distance "
             "over vision/tabular concentration and trigger share), NOT a fixed seed index -- an "
             "earlier version hardcoded seed index 0, which happens to be an outlier that inverts the "
             "vision-vs-tabular concentration ordering relative to the 5-seed mean this file reports.")
    path = config.RESULTS / "nc_masks.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path} ({len(vision_rows)} vision rows, {len(tabular_rows)} tabular rows)")
    print(json.dumps(summary, indent=2))
    print(f"elapsed={time.time() - t0:.0f}s for {len(cfg['seeds'])} seed(s)")


if __name__ == "__main__":
    main()
