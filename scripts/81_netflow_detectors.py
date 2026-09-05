#!/usr/bin/env python
"""Stage 3 of the NetFlow study: do the five vision-built detectors catch this trigger on NetFlow?

The NetFlow sweep (scripts/80) established that the recipe realizes on four new corpora. It measured
attack realization only. Without this stage the manuscript's portability claim, which is its title
claim, still rests on two corpora, and the four new ones say nothing about detectors.

Runs Spectral Signatures, SPECTRE, Activation Clustering, STRIP and Neural Cleanse on the poisoned
training block of each cell, under BOTH decision rules the paper keeps separate: the inherited
fixed removal budget and the budget-free MAD rule. Reports the four axes it never blends -- ASR,
poison recall, ranking AUC, false-positive rate -- and never gives Activation Clustering a ranked
score it does not have.

**The loud control gates everything.** A detector null on a new corpus is uninterpretable without a
positive control on that corpus, which is this project's standing rule. Before any realizable cell
is scored, a blatant constraint-VIOLATING trigger is planted on the same corpus and the detectors
must catch it. If the control fails, this script stops and no realizable number is written.

Three scale decisions, all recorded on every row:
  - The target-class pool reaches 921,824 rows here against CTU-13's 111,666. Detectors are scored
    on a UNIFORM seeded subsample of it, which preserves the poison fraction and therefore the
    detection problem, rather than keeping all poison and thinning the clean rows, which would hand
    the fixed-budget rule a more generous denominator than the attack actually earns.
  - STRIP perturbs every scored row n_trials times, so it gets its own smaller cap.
  - Activation Clustering's silhouette is an O(n^2) diagnostic and is capped; the clustering and the
    recall stay on the full scored subsample.

LightGBM is excluded by the established H4 finding: no penultimate activations, no gradient path.
Re-measuring a structural limit costs compute and says nothing new.

Run:  .venv/bin/python scripts/81_netflow_detectors.py --smoke --limit-cells 2
      .venv/bin/python scripts/81_netflow_detectors.py --control-only
      .venv/bin/python scripts/81_netflow_detectors.py
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.constraints_netflow import (
    admitted_manifest, manifest_fingerprint, project_netflow_to_feasible, repair_fingerprint,
)
from src.data import apply_standardiser, fit_standardiser
from src.data_netflow import split_netflow
from src.detectors import poison_recall
from src.detectors.activation_clustering import detect as ac_detect
from src.detectors.neural_cleanse import (
    balanced_inversion_sample, nc_binary_flag, reverse_engineer_tabular,
)
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.detectors.spectre import spectre_scores
from src.detectors.strip import (calibrate_strip_cutoff, strip_scores,
                                 strip_threshold_flag)
from src.models import attack_success_rate, clean_accuracy, mlp_penultimate_features, train_mlp
from src.poison import poison_secondary_trainset_raw
from src.trigger import apply_secondary_trigger, build_secondary_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
N_SIGMA = 6.0
SPECTRAL_K = config.SPECTRAL_COMPONENTS
Z_THRESHOLDS = (2.0, 2.5, 3.0)
BUDGET_MULTIPLIER = 1.5
SILHOUETTE_SAMPLE = 10_000
STRIP_N_TRIALS = 64          # matches scripts/62 and 70
STRIP_ALPHA = 0.5
STRIP_FRR = 0.01             # Gao et al.'s published operating point
STRIP_RULE = "gao-frr-nonstrict"   # tie-inclusive; a strict cut cannot fire on the softmax floor
STRIP_MAX_SCORED = 20_000    # STRIP perturbs each row n_trials times
MAX_SCORED = 200_000         # uniform subsample of the target-class pool
SUBSAMPLE_SEED = 20260903

# The loud control's own numbers, mirroring the tabular control on the other two corpora.
CONTROL_RATE = 0.01
CONTROL_COST = 8
CONTROL_RECALL_BAR = 0.90    # Activation Clustering's bar on the existing loud control

GATE = config.RESULTS / "netflow_data_gate.json"
SWEEP = config.RESULTS / "netflow_poison_sweep.json"


def results_paths(smoke: bool):
    if smoke:
        return (config.RESULTS / "netflow_detectors_smoke.json",
                config.RESULTS / "netflow_detectors_smoke.checkpoint.json")
    return (config.RESULTS / "netflow_detectors.json",
            config.RESULTS / "netflow_detectors.checkpoint.json")


def cell_key(arm, corpus, share, seed, rate, cost) -> str:
    return f"{arm}|{corpus}|{share}|{seed}|{rate}|{cost}"


def config_key(args, gate) -> dict:
    return dict(smoke=args.smoke, mlp_epochs=args.mlp_epochs, cost=args.cost, rate=args.rate,
                n_sigma=N_SIGMA, spectral_k=SPECTRAL_K, z=list(Z_THRESHOLDS),
                multiplier=BUDGET_MULTIPLIER, max_scored=args.max_scored,
                strip_max_scored=STRIP_MAX_SCORED, strip_trials=STRIP_N_TRIALS,
                strip_frr=STRIP_FRR, strip_rule=STRIP_RULE,
                nc_sample=args.nc_sample, nc_steps=args.nc_steps,
                control=dict(rate=CONTROL_RATE, cost=CONTROL_COST, bar=CONTROL_RECALL_BAR),
                manifest=manifest_fingerprint(), repairs=repair_fingerprint(),
                gate=gate["config_key"])


def load_checkpoint(path: Path, key: dict):
    if not path.exists():
        return {}, {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh")
        return {}, {}
    print(f"resuming: {len(blob.get('controls', {}))} controls, {len(blob.get('cells', {}))} cells done")
    return blob.get("controls", {}), blob.get("cells", {})


def save_checkpoint(path: Path, key: dict, controls: dict, cells: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, controls=controls, cells=cells)))
    tmp.replace(path)


def load_sample(gate: dict, tag: str) -> SimpleNamespace:
    row = gate["samples"][tag]
    df = pd.read_parquet(config.ROOT / row["parquet"])
    S = split_netflow(df, dict(feature_columns=row["feature_columns"]))
    del df
    gc.collect()
    scaler = fit_standardiser(S.x_tr_raw)
    return SimpleNamespace(
        tag=tag, corpus=row["corpus"], feature_columns=row["feature_columns"],
        x_tr_raw=S.x_tr_raw, y_tr=S.y_tr, x_te_raw=S.x_te_raw, y_te=S.y_te, scaler=scaler,
        bounds=(S.x_tr_raw.min(axis=0), S.x_tr_raw.max(axis=0)),
        x_te_attack_raw=S.x_te_raw[S.y_te != TARGET], benign_share=row["benign_share"])


def rule_block(scores, is_poison, expected_frac, entropy_cut=None,
               cut_diagnostics=None) -> dict:
    """Both decision rules against one score vector, plus the ranking metric and each rule's cost.

    The paper's finding is that portability is decided by the RULE, not the score, so the two are
    never merged. The false-positive rate is recomputed here as false positives over the clean rows
    in the target class; a stored FPR field elsewhere in this project divides by all benign rows.
    """
    n_clean = int((~is_poison).sum())
    out = {"fixed_recall": poison_recall(
        flag_by_scores(scores, expected_frac=expected_frac, multiplier=BUDGET_MULTIPLIER), is_poison)}
    out["auc"] = (float(roc_auc_score(is_poison, scores))
                  if 0 < is_poison.sum() < len(is_poison) else None)
    out["mad_recall"], out["mad_fpr"] = {}, {}
    for z in Z_THRESHOLDS:
        flag = flag_by_mad_threshold(scores, z_thresh=z)
        out["mad_recall"][str(z)] = poison_recall(flag, is_poison)
        out["mad_fpr"][str(z)] = float((flag & ~is_poison).sum() / n_clean) if n_clean else None
    if entropy_cut is not None:
        # STRIP's own published rule (Gao et al., ACSAC 2019): flag below an entropy threshold
        # fit at a target false rejection rate on clean held-out rows. Reported alongside the
        # other two rather than replacing them, because the paper's claim is about which RULE a
        # score supports, and STRIP was previously the only detector here scored solely under a
        # rule it does not publish.
        frr_flag = strip_threshold_flag(scores, entropy_cut)
        out["frr_recall"] = poison_recall(frr_flag, is_poison)
        out["frr_fpr"] = (float((frr_flag & ~is_poison).sum() / n_clean) if n_clean else None)
        out["entropy_cut"] = float(entropy_cut)
        out["frr_target"] = STRIP_FRR
        # The rate the cutoff actually achieved on the calibration pool, which is not always the
        # rate requested: STRIP's clamped-softmax floor puts a point mass at the cutoff, and the
        # achievable rates jump over the target. Recorded so a degenerate fit is visible in the
        # artifact rather than showing up as a clean-looking zero.
        if cut_diagnostics is not None:
            out["frr_calibration"] = dict(cut_diagnostics)
    return out


def score_all_detectors(mlp, x_p_std, y_p, poison_idx, S, args, seed, device) -> dict:
    """Every detector on one poisoned model. Returns the per-cell detector block."""
    benign_pos = np.where(y_p == TARGET)[0]
    is_poison_full = np.isin(benign_pos, poison_idx)

    # Uniform subsample of the target-class pool: preserves the poison fraction, so both decision
    # rules see the detection problem the attack actually created.
    rng = np.random.default_rng(SUBSAMPLE_SEED + seed)
    if len(benign_pos) > args.max_scored:
        take = np.sort(rng.choice(len(benign_pos), size=args.max_scored, replace=False))
        benign_pos, is_poison = benign_pos[take], is_poison_full[take]
    else:
        is_poison = is_poison_full

    feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
    expected_frac = float(is_poison.mean())
    has_poison = bool(is_poison.sum() > 0)
    out = dict(n_scored=int(len(is_poison)), n_poison_scored=int(is_poison.sum()),
               n_target_class_total=int(len(is_poison_full)),
               poison_fraction_scored=expected_frac,
               subsampled=bool(len(is_poison) < len(is_poison_full)))
    if not has_poison:
        out["interpretable"] = False
        out["reason"] = "no poison rows in the scored target class"
        return out

    out["spectral"] = rule_block(np.asarray(spectral_scores(feats, n_components=SPECTRAL_K), float),
                                 is_poison, expected_frac)
    out["spectre"] = rule_block(np.asarray(spectre_scores(feats, expected_frac=expected_frac), float),
                                is_poison, expected_frac)

    ac_mask, ac_sil = ac_detect(feats, seed=seed, silhouette_sample=SILHOUETTE_SAMPLE)
    n_flagged = int(ac_mask.sum())
    out["ac"] = dict(
        recall=poison_recall(ac_mask, is_poison), n_flagged=n_flagged,
        purity=(float((ac_mask & is_poison).sum() / n_flagged) if n_flagged else None),
        fpr=float((ac_mask & ~is_poison).sum() / max(1, int((~is_poison).sum()))),
        silhouette=ac_sil, auc=None,
        auc_reason="Activation Clustering is a hard two-cluster assignment, not a ranked score")

    # STRIP gets a smaller cap: it perturbs every scored row STRIP_N_TRIALS times.
    s_idx = (np.sort(rng.choice(len(benign_pos), size=STRIP_MAX_SCORED, replace=False))
             if len(benign_pos) > STRIP_MAX_SCORED else np.arange(len(benign_pos)))
    s_poison = is_poison[s_idx]
    if s_poison.sum() > 0:
        blend = apply_standardiser(S.scaler, S.x_te_raw[S.y_te == TARGET][:5000])
        strip_sc = strip_scores(mlp, x_p_std[benign_pos[s_idx]], blend, n_trials=STRIP_N_TRIALS,
                                alpha=STRIP_ALPHA, device=device, seed=seed)
        strip_cut, strip_diag = calibrate_strip_cutoff(
            mlp, blend, n_trials=STRIP_N_TRIALS, alpha=STRIP_ALPHA, device=device, seed=seed,
            frr=STRIP_FRR, return_diagnostics=True)
        out["strip"] = rule_block(np.asarray(strip_sc, float), s_poison, float(s_poison.mean()),
                                  entropy_cut=strip_cut,
                                  cut_diagnostics=strip_diag)
        out["strip"]["n_scored"] = int(len(s_idx))
    else:
        out["strip"] = dict(interpretable=False,
                            reason="no poison rows survived STRIP's subsample")

    # Neural Cleanse: a hard flag, not a score. Two classes, so its anomaly index is degenerate by
    # construction and the binary rule is used, exactly as on the other two corpora.
    x_tr_std = apply_standardiser(S.scaler, S.x_tr_raw)
    std_lo, std_hi = x_tr_std.min(axis=0), x_tr_std.max(axis=0)
    bnd = (torch.tensor(std_lo, dtype=torch.float32), torch.tensor(std_hi, dtype=torch.float32))
    bd_norms = reverse_engineer_tabular(
        mlp, 2, balanced_inversion_sample(x_p_std, y_p, args.nc_sample, seed),
        device=device, steps=args.nc_steps, bounds=bnd)
    nc_flag, nc_ratio = nc_binary_flag(bd_norms, tau=2.0)
    out["nc"] = dict(flagged_class=int(nc_flag), flags_target=bool(int(nc_flag) == TARGET),
                     ratio=(float(nc_ratio) if np.isfinite(nc_ratio) else None),
                     auc=None, auc_reason="Neural Cleanse emits a hard flag, not a ranked score")
    del feats, x_tr_std
    gc.collect()
    out["interpretable"] = True
    return out


def build_and_poison(S, ranking, cost, rate, seed, relations, project, args, device):
    """Trigger, poison, victim. `project=None` gives the constraint-VIOLATING control trigger."""
    project_fn = (partial(project_netflow_to_feasible, relations=relations,
                          feature_names=S.feature_columns, bounds=S.bounds)
                  if project else (lambda rows: rows))
    trig = build_secondary_trigger(ranking, cost, S.x_tr_raw, S.feature_columns, S.bounds,
                                   project_fn, list(range(len(S.feature_columns))), n_sigma=N_SIGMA)
    x_p_raw, y_p, poison_idx = poison_secondary_trainset_raw(
        S.x_tr_raw, S.y_tr, trig, rate, target=TARGET, seed=seed)
    x_p_std = apply_standardiser(S.scaler, x_p_raw)
    mlp = train_mlp(x_p_std, y_p, seed, epochs=args.mlp_epochs, device=device)
    atk = apply_standardiser(S.scaler, apply_secondary_trigger(S.x_te_attack_raw, trig))
    asr = float(attack_success_rate(mlp, atk, TARGET, device=device))
    cacc = float(clean_accuracy(mlp, apply_standardiser(S.scaler, S.x_te_raw), S.y_te, device=device))
    del x_p_raw, atk
    gc.collect()
    return trig, x_p_std, y_p, poison_idx, mlp, asr, cacc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--control-only", action="store_true")
    ap.add_argument("--limit-cells", type=int, default=None)
    ap.add_argument("--mlp-epochs", type=int, default=None)
    ap.add_argument("--cost", type=int, default=config.NETFLOW_PRIMARY_COST)
    ap.add_argument("--rate", type=float, default=config.NETFLOW_RATE)
    ap.add_argument("--max-scored", type=int, default=MAX_SCORED)
    ap.add_argument("--nc-sample", type=int, default=400)
    ap.add_argument("--nc-steps", type=int, default=300)
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    args.mlp_epochs = args.mlp_epochs or (10 if args.smoke else 20)
    if args.smoke:
        args.max_scored = min(args.max_scored, 50_000)
        args.nc_steps = min(args.nc_steps, 100)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(config.SEEDS)

    config.ensure_dirs()
    for p in (GATE, SWEEP):
        if not p.exists():
            print(f"missing {p}; run the earlier stages first")
            return 1
    gate = json.loads(GATE.read_text())
    sweep = json.loads(SWEEP.read_text())
    if gate.get("smoke") and not args.smoke:
        print(f"REFUSING: {GATE.name} was built by a --smoke gate run")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q1 = [config.NETFLOW_MANIPULATION_CORPUS, config.NETFLOW_REPLICATION_CORPUS]
    kept_q1, _ = admitted_manifest(gate["per_corpus_satisfaction"], corpora=q1)
    kept_q2, _ = admitted_manifest(gate["per_corpus_satisfaction"], corpora=list(config.NETFLOW_CORPORA))
    key = config_key(args, gate)
    out_path, ckpt_path = results_paths(args.smoke)
    controls, cells = load_checkpoint(ckpt_path, key)
    t0 = time.time()
    print(f"device={device} smoke={args.smoke} epochs={args.mlp_epochs} cost={args.cost} "
          f"rate={args.rate} max_scored={args.max_scored:,} seeds={seeds}")

    # ---- the gate: loud constraint-violating control on the manipulation corpus ----
    ctag = f"native__{config.NETFLOW_MANIPULATION_CORPUS}"
    control_seeds = seeds[:1] if args.smoke else seeds
    if any(str(s) not in controls for s in control_seeds):
        print(f"\n=== loud control on {ctag}: violating trigger, rate {CONTROL_RATE}, "
              f"cost {CONTROL_COST}, bar recall >= {CONTROL_RECALL_BAR} ===")
        S = load_sample(gate, ctag)
        for seed in control_seeds:
            if str(seed) in controls:
                continue
            x_tr_std = apply_standardiser(S.scaler, S.x_tr_raw)
            clean_mlp = train_mlp(x_tr_std, S.y_tr, seed, epochs=args.mlp_epochs, device=device)
            ranking = shap_rank_features(clean_mlp, S.x_tr_raw, S.scaler, target=TARGET,
                                         kind="mlp", device=device)
            del x_tr_std, clean_mlp
            gc.collect()
            trig, x_p_std, y_p, pidx, mlp, asr, cacc = build_and_poison(
                S, ranking, CONTROL_COST, CONTROL_RATE, seed, None, False, args, device)
            det = score_all_detectors(mlp, x_p_std, y_p, pidx, S, args, seed, device)
            row = dict(seed=seed, corpus=S.corpus, tag=ctag, rate=CONTROL_RATE, cost=CONTROL_COST,
                       variant="violating", asr=asr, clean_acc=cacc, **det)
            sp = row.get("spectral", {}).get("fixed_recall")
            ac = row.get("ac", {}).get("recall")
            row["passes"] = bool(asr >= config.ATTACK_EFFECTIVE_ASR
                                 and (sp is not None and sp >= CONTROL_RECALL_BAR
                                      or ac is not None and ac >= CONTROL_RECALL_BAR))
            controls[str(seed)] = row
            save_checkpoint(ckpt_path, key, controls, cells)
            print(f"  seed {seed}: asr={asr:.4f} spectral_fixed={sp} ac_recall={ac} "
                  f"-> passes={row['passes']}")
            del mlp, x_p_std
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        del S
        gc.collect()

    failed = [s for s, r in controls.items() if not r.get("passes")]
    if failed:
        print(f"\nBLOCKING: loud control FAILED on seeds {failed}. A detector null on this corpus "
              "would be uninterpretable. Stopping before any realizable cell.")
        out_path.write_text(json.dumps(dict(config_key=key, controls=controls, cells=cells,
                                            blocked=True, blocking_seeds=failed), indent=2))
        return 1
    print(f"\nloud control PASSES on {len(controls)} seed(s); proceeding to realizable cells")
    if args.control_only:
        out_path.write_text(json.dumps(dict(config_key=key, controls=controls, cells={},
                                            control_only=True), indent=2))
        return 0

    # ---- realizable cells ----
    work = [("Q1", c, s, f"share{s:.2f}__{c}", args.rate, kept_q1)
            for c in q1 for s in config.NETFLOW_BENIGN_SHARES]
    work += [("Q2", c, None, f"native__{c}", args.rate, kept_q2) for c in config.NETFLOW_CORPORA]
    if args.smoke:
        work = work[:1]
    n_new = 0
    for arm, corpus, share, tag, rate, relations in work:
        todo = [s for s in seeds if cell_key(arm, corpus, share, s, rate, args.cost) not in cells]
        if not todo or (args.limit_cells is not None and n_new >= args.limit_cells):
            continue
        feas = gate["samples"][tag]["feasibility"]["per_rate"][f"{rate}"]
        if not feas["feasible"]:
            for s in todo:
                cells[cell_key(arm, corpus, share, s, rate, args.cost)] = dict(
                    arm=arm, corpus=corpus, benign_share=share, tag=tag, seed=s, rate=rate,
                    cost=args.cost, status="attack_infeasible", interpretable=False,
                    reason="no poison budget: the benign class cannot fund this rate")
            save_checkpoint(ckpt_path, key, controls, cells)
            print(f"--- {tag}: attack_infeasible, {len(todo)} cells recorded, not scored ---")
            continue
        print(f"\n--- {tag} [{arm}]: {len(todo)} cells ({time.time()-t0:.0f}s) ---")
        S = load_sample(gate, tag)
        for seed in todo:
            if args.limit_cells is not None and n_new >= args.limit_cells:
                break
            t_c = time.time()
            x_tr_std = apply_standardiser(S.scaler, S.x_tr_raw)
            clean_mlp = train_mlp(x_tr_std, S.y_tr, seed, epochs=args.mlp_epochs, device=device)
            ranking = shap_rank_features(clean_mlp, S.x_tr_raw, S.scaler, target=TARGET,
                                         kind="mlp", device=device)
            del x_tr_std, clean_mlp
            gc.collect()
            trig, x_p_std, y_p, pidx, mlp, asr, cacc = build_and_poison(
                S, ranking, args.cost, rate, seed, relations, True, args, device)
            ineffective = asr < config.ATTACK_EFFECTIVE_ASR
            base = dict(arm=arm, corpus=corpus, benign_share=share, tag=tag, seed=seed, rate=rate,
                        cost=args.cost, asr=asr, clean_acc=cacc,
                        trigger_features_clipped_by_projection=int(
                            (~np.isclose(np.asarray(trig["values"], float),
                                         np.asarray(trig["post_projection_values"], float))).sum()))
            if ineffective:
                row = dict(**base, status="attack_ineffective", interpretable=False,
                           reason=f"poisoned ASR {asr:.4f} below {config.ATTACK_EFFECTIVE_ASR}")
            else:
                row = dict(**base, status="ok",
                           **score_all_detectors(mlp, x_p_std, y_p, pidx, S, args, seed, device))
            row["elapsed_s"] = round(time.time() - t_c, 1)
            cells[cell_key(arm, corpus, share, seed, rate, args.cost)] = row
            save_checkpoint(ckpt_path, key, controls, cells)
            n_new += 1
            if row.get("interpretable"):
                print(f"  seed {seed}: asr={asr:.4f} | spectral fixed={row['spectral']['fixed_recall']:.4f} "
                      f"mad3={row['spectral']['mad_recall']['3.0']:.4f} auc={row['spectral']['auc']:.4f} | "
                      f"spectre auc={row['spectre']['auc']:.4f} | ac={row['ac']['recall']} | "
                      f"nc_target={row['nc']['flags_target']}  {row['elapsed_s']}s")
            else:
                print(f"  seed {seed}: {row['status']} ({row.get('reason')})")
            del mlp, x_p_std
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        del S
        gc.collect()

    out_path.write_text(json.dumps(dict(config_key=key, controls=controls, cells=cells,
                                        n_cells=len(cells), device=device,
                                        elapsed_s=round(time.time() - t0, 1)), indent=2))
    print(f"\n{len(cells)} cells on disk, {n_new} computed this invocation, {time.time()-t0:.0f}s")
    print(f"wrote {out_path.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
