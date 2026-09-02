"""Extend the loud TABULAR positive control to STRIP and SPECTRE, the two detectors it never covered.

Why this exists. scripts/04_poison_sweep.py's `positive_control` plants a blatant VIOLATING trigger
that the detectors should catch, to separate "portability genuinely fails" from "the tabular wiring
is broken". Section IV-G of the manuscript then makes the strongest claim in the Methods section:
"No tabular null reported below can therefore be a wiring error."

results/tabular_positive_control.json `.per_seed[]` carries fields for Spectral, Activation
Clustering and Neural Cleanse only. There is NO STRIP field and NO SPECTRE field. An IEEE Access
review (2026-09-02) noticed that the two detectors whose nulls the paper actually reports -- STRIP's
MAD recall of exactly 0.0000, and SPECTRE -- are outside the control that is supposed to license
those nulls. The reviewer is right, and understated it: the gap is two detectors, not one.

This script closes the gap. It reproduces the EXACT control construction from 04 (same violating
trigger, same control rate/cost, same unconstrained clean-label poisoning call) and scores four
detectors on the identical trained model and poisoned population:

  * Spectral and Activation Clustering -- recomputed purely as a cross-check that this script sits
    in the same code path as the committed control. If these do not reproduce
    tabular_positive_control.json, nothing else here is trustworthy and the run says so.
  * STRIP and SPECTRE -- the new arms.

Neural Cleanse is deliberately NOT re-run: its inversion already has a control row on disk, and two
gradient inversions per seed dominate the wall-clock of this script by an order of magnitude.

STRIP is scored under BOTH decision rules, because the manuscript's STRIP null is rule-specific: a
ranking AUC around 0.96 sitting beside a MAD recall of exactly 0.0. If the loud control reproduces
that same shape -- high AUC, zero MAD recall -- then the MAD collapse is a property of the decision
rule rather than a wiring fault, which is the paper's own thesis.

Checkpointed per seed, atomic temp-then-rename, config fingerprint.

Run:  .venv/bin/python scripts/70_positive_control_strip_spectre.py
      .venv/bin/python scripts/70_positive_control_strip_spectre.py --smoke
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup
from src import config
from src.data import apply_standardiser
from src.detectors import poison_recall
from src.detectors.activation_clustering import detect as ac_detect
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.detectors.spectre import spectre_scores
from src.detectors.strip import strip_scores
from src.models import mlp_penultimate_features, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import build_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
SPECTRAL_K = 5
Z_THRESHOLDS = (2.0, 2.5, 3.0)
N_TRIALS = 64      # matches scripts/62_strip_detector.py
ALPHA = 0.5        # matches scripts/62_strip_detector.py
REPRODUCTION_ATOL = 1e-6

OUT = config.RESULTS / "positive_control_strip_spectre.json"
CKPT = config.RESULTS / "positive_control_strip_spectre.checkpoint.json"


def _smoke_paths():
    """A smoke run must never land on the real artefact path: a killed full run would otherwise
    leave a 1-seed smoke file sitting where the committed result belongs."""
    global OUT, CKPT
    OUT = config.RESULTS / "positive_control_strip_spectre_smoke.json"
    CKPT = config.RESULTS / "positive_control_strip_spectre_smoke.checkpoint.json"


def config_key(cfg) -> dict:
    return dict(seeds=list(cfg["seeds"]), mlp_epochs=cfg["mlp_epochs"],
                control_rate=cfg["control_rate"], control_cost=cfg["control_cost"],
                spectral_k=SPECTRAL_K, z=list(Z_THRESHOLDS), n_trials=N_TRIALS, alpha=ALPHA,
                smoke=cfg["smoke"])


def load_ckpt(key):
    if not CKPT.exists():
        return {}
    try:
        blob = json.loads(CKPT.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != json.loads(json.dumps(key)):
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} seed(s) already done")
    return blob.get("rows", {})


def save_ckpt(key, rows):
    tmp = CKPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(CKPT)


def load_committed_control() -> dict:
    """Index the committed control by seed, for the reproduction cross-check."""
    path = config.RESULTS / "tabular_positive_control.json"
    if not path.exists():
        print("WARNING: results/tabular_positive_control.json not found -- reproduction check DISABLED")
        return {}
    blob = json.loads(path.read_text())
    return {r["seed"]: r for r in blob.get("per_seed", [])}


def rule_block(scores, is_poison, expected_frac):
    """Both decision rules against one score vector, plus the ranking metric.

    The fixed-budget rule and the budget-free MAD rule are reported separately throughout this
    paper because the central finding is that portability is decided by the RULE, not the score.
    """
    n_clean = int((~is_poison).sum())
    out = {
        "fixed_recall": poison_recall(flag_by_scores(scores, expected_frac=expected_frac), is_poison),
        "auc": (float(roc_auc_score(is_poison, scores))
                if 0 < is_poison.sum() < len(is_poison) else None),
        "mad_recall": {},
        "mad_fpr": {},
    }
    for z in Z_THRESHOLDS:
        flag = flag_by_mad_threshold(scores, z_thresh=z)
        out["mad_recall"][str(z)] = poison_recall(flag, is_poison)
        out["mad_fpr"][str(z)] = float((flag & ~is_poison).sum() / n_clean) if n_clean else None
    return out


def main() -> int:
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    cfg = get_config(smoke)
    if smoke:
        cfg["seeds"] = [42]
        _smoke_paths()
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = config_key(cfg)
    rows = load_ckpt(key)
    committed = load_committed_control()
    print(f"device={device}  seeds={cfg['seeds']}  "
          f"control=(rate {cfg['control_rate']}, cost {cfg['control_cost']})")

    S = load_setup(cfg)
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])   # STRIP's blend pool

    for seed in cfg["seeds"]:
        if str(seed) in rows:
            print(f"--- seed {seed}: cached, skipping ---")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")

        x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
        clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
        rank_mlp = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET, kind="mlp",
                                      device=device)

        # The control's whole point is a trigger that VIOLATES the constraint manifest, so it is
        # blatant in feature space. build_trigger's second return is that violating variant, and
        # the poisoning call deliberately omits constraints/bounds -- both match 04 exactly.
        _, violat = build_trigger(rank_mlp, cfg["control_cost"], x_tr_raw, constraints, features,
                                  bounds)
        x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
            x_tr_raw, y_tr, violat, cfg["control_rate"], S["scaler"], target=TARGET, seed=seed)
        mlp_bd = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

        benign_pos = np.where(y_p == TARGET)[0]
        is_poison = np.isin(benign_pos, poison_idx)
        expected_frac = float(is_poison.mean())
        feats = mlp_penultimate_features(mlp_bd, x_p_std[benign_pos], device)

        spec_scores = spectral_scores(feats, n_components=SPECTRAL_K)
        spec_recall = poison_recall(flag_by_scores(spec_scores, expected_frac=expected_frac),
                                    is_poison)
        ac_mask, ac_sil = ac_detect(feats, seed=seed)
        ac_recall = poison_recall(ac_mask, is_poison)

        # --- the two new arms ---
        strip_sc = strip_scores(mlp_bd, x_p_std[benign_pos], x_te_std, n_trials=N_TRIALS,
                                alpha=ALPHA, device=device, seed=seed)
        spct_sc = spectre_scores(feats, expected_frac=expected_frac)

        row = dict(
            seed=seed, n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
            expected_frac=expected_frac,
            spectral_recall=spec_recall, ac_recall=ac_recall, ac_silhouette=ac_sil,
            strip=rule_block(strip_sc, is_poison, expected_frac),
            spectre=rule_block(spct_sc, is_poison, expected_frac),
        )

        # Reproduction cross-check against the committed control. A mismatch means this script is
        # not in the same code path as 04, and the new arms would be uninterpretable.
        # A smoke run uses a subsampled training set and fewer epochs than the committed control,
        # so it CANNOT reproduce it and a failure there means nothing. Only check on a full run.
        ref = None if smoke else committed.get(seed)
        if ref is not None:
            d_spec = abs(spec_recall - ref["spectral_recall"])
            d_ac = abs(ac_recall - ref["ac_recall"])
            row["reproduction"] = dict(
                spectral_committed=ref["spectral_recall"], spectral_delta=d_spec,
                ac_committed=ref["ac_recall"], ac_delta=d_ac,
                ok=bool(d_spec <= REPRODUCTION_ATOL and d_ac <= REPRODUCTION_ATOL))
            if not row["reproduction"]["ok"]:
                print(f"  WARNING: does not reproduce committed control "
                      f"(spectral delta {d_spec:.2e}, ac delta {d_ac:.2e})")
        else:
            row["reproduction"] = None

        rows[str(seed)] = row
        save_ckpt(key, rows)
        print(f"  spectral={spec_recall:.4f} ac={ac_recall:.4f} | "
              f"STRIP fixed={row['strip']['fixed_recall']:.4f} auc={row['strip']['auc']:.4f} "
              f"mad3={row['strip']['mad_recall']['3.0']:.4f} | "
              f"SPECTRE fixed={row['spectre']['fixed_recall']:.4f} auc={row['spectre']['auc']:.4f} "
              f"mad3={row['spectre']['mad_recall']['3.0']:.4f}")

    ordered = [rows[str(s)] for s in cfg["seeds"] if str(s) in rows]

    def agg(path_fn):
        vals = [v for v in (path_fn(r) for r in ordered) if v is not None]
        if not vals:
            return None
        return dict(mean=float(np.mean(vals)),
                    sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None, n=len(vals))

    summary = dict(
        spectral_recall=agg(lambda r: r["spectral_recall"]),
        ac_recall=agg(lambda r: r["ac_recall"]),
        strip_fixed_recall=agg(lambda r: r["strip"]["fixed_recall"]),
        strip_auc=agg(lambda r: r["strip"]["auc"]),
        strip_mad_recall_3=agg(lambda r: r["strip"]["mad_recall"]["3.0"]),
        spectre_fixed_recall=agg(lambda r: r["spectre"]["fixed_recall"]),
        spectre_auc=agg(lambda r: r["spectre"]["auc"]),
        spectre_mad_recall_3=agg(lambda r: r["spectre"]["mad_recall"]["3.0"]),
        # None, not True, when nothing was checked -- an unrun check must never read as a pass.
        reproduction_ok=(all(r["reproduction"]["ok"] for r in ordered
                             if r.get("reproduction") is not None)
                         if any(r.get("reproduction") is not None for r in ordered) else None),
    )
    # Same PASS bar the committed control uses for Spectral: the loud violating trigger must be
    # caught. Applied per detector, under the fixed-budget rule, so a detector that passes here
    # can no longer have its tabular null dismissed as a wiring fault.
    summary["strip_control_passes"] = bool(
        summary["strip_fixed_recall"] and summary["strip_fixed_recall"]["mean"] >= 0.7)
    summary["spectre_control_passes"] = bool(
        summary["spectre_fixed_recall"] and summary["spectre_fixed_recall"]["mean"] >= 0.7)

    OUT.write_text(json.dumps(dict(config=key, summary=summary, per_seed=ordered), indent=2))
    print(f"\nwrote {OUT}  elapsed={time.time() - t0:.0f}s  n_seeds={len(ordered)}")
    print(f"  reproduces committed control: {summary['reproduction_ok']}")
    print(f"  STRIP control passes  : {summary['strip_control_passes']}")
    print(f"  SPECTRE control passes: {summary['spectre_control_passes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
