"""The loud tabular positive control on UNSW-NB15, the corpus where removal is the operative defense.

Why this exists. `scripts/70_positive_control_strip_spectre.py` plants a blatant
constraint-violating trigger on CTU-13 and checks that each detector catches it, which is what
licenses reading any CTU-13 null as a property of the data or the decision rule rather than as an
implementation fault. UNSW-NB15 has no such control. An IEEE Access verification pass
(2026-09-02) named that gap as the single change that would move the methodological grade,
because the paper's own framing says CTU-13 removal cannot stop an attack the clean victim
already permits, so every security-relevant null lives on the corpus with no control behind it.

This is script 70's control construction wearing script 71's secondary-corpus plumbing.

The violating trigger on this corpus. `build_trigger` (CTU) returns a realizable and a violating
variant. `build_secondary_trigger` instead takes a `project_fn`, and `apply_secondary_trigger`'s
own docstring records the convention: the caller passes the identity function for a violating
trigger and a real projector for a realizable one. So the control builds its trigger with
`_identity` and the graded runs build theirs with `project_secondary_to_feasible`. Same recipe,
same features, no constraint repair, which is exactly what makes the control loud.

THE SILHOUETTE CAP IS LOAD-BEARING, NOT A TUNING CHOICE. `cluster_and_reduce` falls through to a
full `silhouette_score` when `silhouette_sample is None`, and script 70 calls `ac_detect` without
one. That is O(n^2). On CTU-13's 111,666-row benign class it is roughly 1.2e10 pairwise
distances and merely slow; on this corpus's 1,233,218 rows it is about 1.5e12, against a machine
whose RAM is shared with other jobs. We pass `silhouette_sample=10000`, the cap
`scripts/05_run_detectors.py` already applies on the main grid. It changes only the reported
diagnostic: the clustering, the smaller-cluster choice and the recall metric all stay on the
full set, per that module's own comment.

Neural Cleanse is deliberately not re-run. It already has a control row, and two gradient
inversions per seed would dominate this script's wall-clock by an order of magnitude.

Checkpointed per seed, atomic temp-then-rename, config fingerprint. A smoke run writes to a
separate path so a killed full run cannot leave a one-seed file where the committed artefact
belongs.

Run:  .venv/bin/python scripts/74_secondary_positive_control.py
      .venv/bin/python scripts/74_secondary_positive_control.py --smoke
"""
from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config
from _m3_secondary_common import eligible_indices_for
from src import config
from src.data import apply_standardiser
from src.data_secondary import load_secondary_setup
from src.detectors import poison_recall
from src.detectors.activation_clustering import detect as ac_detect
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.detectors.spectre import spectre_scores
from src.detectors.strip import strip_scores
from src.models import mlp_penultimate_features, train_mlp
from src.poison import poison_secondary_trainset
from src.trigger import build_secondary_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
SPECTRAL_K = 5
Z_THRESHOLDS = (2.0, 2.5, 3.0)
N_TRIALS = 64          # matches scripts/62 and 70
ALPHA = 0.5            # matches scripts/62 and 70
SILHOUETTE_SAMPLE = 10000   # see module docstring; NOT optional at this corpus size
CONTROL_PASS_BAR = 0.7      # the same bar script 70 and the committed CTU control apply

OUT = config.RESULTS / "secondary_positive_control.json"
CKPT = config.RESULTS / "secondary_positive_control.checkpoint.json"


def _smoke_paths():
    global OUT, CKPT
    OUT = config.RESULTS / "secondary_positive_control_smoke.json"
    CKPT = config.RESULTS / "secondary_positive_control_smoke.checkpoint.json"


def _identity(x):
    """The violating projector. Named rather than a lambda so the checkpoint config and any
    traceback say what it is."""
    return x


def config_key(cfg) -> dict:
    return dict(dataset=config.SECONDARY_DATASET_ID, seeds=list(cfg["seeds"]),
                mlp_epochs=cfg["mlp_epochs"], control_rate=cfg["control_rate"],
                control_cost=cfg["control_cost"], spectral_k=SPECTRAL_K,
                z=list(Z_THRESHOLDS), n_trials=N_TRIALS, alpha=ALPHA,
                silhouette_sample=SILHOUETTE_SAMPLE, trigger="violating_identity_projection",
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


def rule_block(scores, is_poison, expected_frac):
    """Both decision rules against one score vector, plus the ranking metric.

    Identical to scripts/70's rule_block so the two corpora's control columns are produced by the
    same arithmetic and can be placed side by side.
    """
    n_clean = int((~is_poison).sum())
    out = {
        "fixed_recall": poison_recall(flag_by_scores(scores, expected_frac=expected_frac),
                                      is_poison),
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
    print(f"device={device}  dataset={config.SECONDARY_DATASET_ID}  seeds={cfg['seeds']}  "
          f"control=(rate {cfg['control_rate']}, cost {cfg['control_cost']})  "
          f"silhouette_sample={SILHOUETTE_SAMPLE}")

    S0 = load_secondary_setup(config.SECONDARY_DATASET_ID)
    eligible = eligible_indices_for(S0.feature_names)
    x_te_std = apply_standardiser(S0.scaler, S0.x_test_raw)   # STRIP's blend pool

    for seed in cfg["seeds"]:
        if str(seed) in rows:
            print(f"--- seed {seed}: cached, skipping ---")
            continue
        t_seed = time.time()
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")

        x_tr_std = apply_standardiser(S0.scaler, S0.x_train_raw)
        clean_mlp = train_mlp(x_tr_std, S0.y_train, seed, epochs=cfg["mlp_epochs"],
                              device=device, log_every=5)
        ranking = shap_rank_features(clean_mlp, S0.x_train_raw, S0.scaler, target=TARGET,
                                     kind="mlp", device=device)

        # The control's whole point is a trigger that VIOLATES the feasibility manifest, so it is
        # blatant in feature space. Identity projection is this corpus's violating variant.
        violat = build_secondary_trigger(ranking, cfg["control_cost"], S0.x_train_raw,
                                         S0.feature_names, S0.bounds, _identity, eligible)
        x_p_std, y_p, poison_idx = poison_secondary_trainset(
            S0.x_train_raw, S0.y_train, violat, cfg["control_rate"], S0.scaler,
            target=TARGET, seed=seed)
        mlp_bd = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device,
                           log_every=5)

        benign_pos = np.where(y_p == TARGET)[0]
        is_poison = np.isin(benign_pos, poison_idx)
        if is_poison.sum() == 0:
            print(f"  seed {seed}: no poison in target class, skipped")
            continue
        expected_frac = float(is_poison.mean())
        feats = mlp_penultimate_features(mlp_bd, x_p_std[benign_pos], device)
        print(f"  scoring {len(benign_pos):,} benign rows, {int(is_poison.sum()):,} poisoned")

        spec_scores = spectral_scores(feats, n_components=SPECTRAL_K)
        spec_recall = poison_recall(flag_by_scores(spec_scores, expected_frac=expected_frac),
                                    is_poison)
        ac_mask, ac_sil = ac_detect(feats, seed=seed, silhouette_sample=SILHOUETTE_SAMPLE)
        ac_recall = poison_recall(ac_mask, is_poison)
        strip_sc = strip_scores(mlp_bd, x_p_std[benign_pos], x_te_std, n_trials=N_TRIALS,
                                alpha=ALPHA, device=device, seed=seed)
        spct_sc = spectre_scores(feats, expected_frac=expected_frac)

        rows[str(seed)] = dict(
            dataset=config.SECONDARY_DATASET_ID, seed=seed,
            n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
            expected_frac=expected_frac,
            spectral_recall=spec_recall, ac_recall=ac_recall, ac_silhouette=ac_sil,
            strip=rule_block(strip_sc, is_poison, expected_frac),
            spectre=rule_block(spct_sc, is_poison, expected_frac),
            seed_seconds=round(time.time() - t_seed, 1))
        save_ckpt(key, rows)
        r = rows[str(seed)]
        print(f"  spectral={spec_recall:.4f} ac={ac_recall:.4f} | "
              f"STRIP fixed={r['strip']['fixed_recall']:.4f} auc={r['strip']['auc']:.4f} "
              f"mad3={r['strip']['mad_recall']['3.0']:.4f} | "
              f"SPECTRE fixed={r['spectre']['fixed_recall']:.4f} "
              f"auc={r['spectre']['auc']:.4f} mad3={r['spectre']['mad_recall']['3.0']:.4f}")
        print(f"  seed wall-clock {r['seed_seconds']:.0f}s")

    ordered = [rows[str(s)] for s in cfg["seeds"] if str(s) in rows]

    def agg(fn):
        vals = [v for v in (fn(r) for r in ordered) if v is not None]
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
        mean_seed_seconds=agg(lambda r: r.get("seed_seconds")),
    )
    # Same PASS bar the committed CTU control applies, per detector, under the published rule.
    for det, field in (("spectral", "spectral_recall"), ("ac", "ac_recall"),
                       ("strip", "strip_fixed_recall"), ("spectre", "spectre_fixed_recall")):
        s = summary[field]
        summary[f"{det}_control_passes"] = bool(s and s["mean"] >= CONTROL_PASS_BAR)

    OUT.write_text(json.dumps(dict(config=key, summary=summary, per_seed=ordered), indent=2))
    print(f"\nwrote {OUT}  elapsed={time.time() - t0:.0f}s  n_seeds={len(ordered)}")
    for det in ("spectral", "ac", "strip", "spectre"):
        print(f"  {det:9s} control passes: {summary[f'{det}_control_passes']}")
    print("  A detector that passes here can no longer have its UNSW-NB15 null read as a")
    print("  wiring fault. One that fails bounds the claim instead of extending it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
