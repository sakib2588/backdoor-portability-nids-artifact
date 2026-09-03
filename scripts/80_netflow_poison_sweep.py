#!/usr/bin/env python
"""Stage 2 of the NetFlow benign-share study: the poison sweep over both arms.

Q1, the within-corpus manipulation: two corpora x six benign-share levels x two costs x five seeds
at a fixed rate. Q2, the native-balance replication: four corpora x three rates x two costs x five
seeds. The recipe is the one the UNSW-NB15 arm already uses -- SHAP ranking on the clean victim, a
top-cost watermark at mean + 6 sigma in raw space, projected onto the admitted manifest, planted
clean-label into target-class rows -- so the NetFlow numbers are comparable to the existing arms
rather than a second recipe measured alongside them.

The primary outcome is the backdoor fraction: poisoned-model ASR minus the SAME trigger's ASR
against the CLEAN model. A corpus whose clean model already accepts watermarked flows realizes the
recipe as evasion; one that does not realizes it as a backdoor. Both pre-registered guards are
enforced here and recorded on the row: `saturated` when the clean model already accepts nearly
everything, `attack_ineffective` when the poisoning did not take.

Clean victims, SHAP rankings and triggers are cached in-process per (sample, seed) and (sample, seed,
cost). They do not depend on the poison rate, so recomputing them per cell would roughly double the
run for no change in any number.

Run:  .venv/bin/python scripts/80_netflow_poison_sweep.py --smoke --limit-cells 3
      .venv/bin/python scripts/80_netflow_poison_sweep.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.constraints_netflow import (
    NETFLOW_RELATIONS, admitted_manifest, audit_netflow_constraints, manifest_fingerprint,
    project_netflow_to_feasible, repair_fingerprint,
)
from src.data import apply_standardiser, fit_standardiser
from src.data_netflow import split_netflow
from src.models import attack_success_rate, clean_accuracy, train_mlp
from src.poison import poison_secondary_trainset_raw
from src.trigger import apply_secondary_trigger, build_secondary_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
N_SIGMA = 6.0        # the CTU-13 and UNSW-NB15 arms' value; changing it would break comparability

GATE = config.RESULTS / "netflow_data_gate.json"


def results_paths(smoke: bool):
    """Smoke and full runs write to SEPARATE files, so re-running the smoke command after the full
    grid is committed can never overwrite it. This project has already lost a committed results file
    to a probe that wrote to the real path."""
    if smoke:
        return (config.RESULTS / "netflow_poison_sweep_smoke.json",
                config.RESULTS / "netflow_poison_sweep_smoke.checkpoint.json")
    return (config.RESULTS / "netflow_poison_sweep.json",
            config.RESULTS / "netflow_poison_sweep.checkpoint.json")


def cell_key(arm, corpus, share, seed, rate, cost) -> str:
    return f"{arm}|{corpus}|{share}|{seed}|{rate}|{cost}"


def config_key(args, gate) -> dict:
    return dict(smoke=args.smoke, costs=list(config.NETFLOW_COSTS), rate=config.NETFLOW_RATE,
                native_rates=list(config.NETFLOW_NATIVE_RATES),
                shares=list(config.NETFLOW_BENIGN_SHARES),
                manip=config.NETFLOW_MANIPULATION_CORPUS,
                repl=config.NETFLOW_REPLICATION_CORPUS,
                mlp_epochs=args.mlp_epochs, n_sigma=N_SIGMA, target=TARGET,
                manifest=manifest_fingerprint(), repairs=repair_fingerprint(),
                gate=gate["config_key"])


def load_checkpoint(path: Path, key: dict) -> dict:
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming: {len(blob.get('cells', {}))} cells already done")
    return blob.get("cells", {})


def save_checkpoint(path: Path, key: dict, cells: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, cells=cells)))
    tmp.replace(path)   # atomic: a crash mid-write never corrupts the checkpoint


def arm_manifest(gate: dict, corpora: list) -> tuple:
    """The admitted manifest for the corpora this arm actually analyzes."""
    kept, report = admitted_manifest(gate["per_corpus_satisfaction"], corpora=corpora)
    return kept, report


def load_sample(gate: dict, tag: str) -> SimpleNamespace:
    """Materialized sample -> split, standardiser and fitted bounds, in one object."""
    row = gate["samples"][tag]
    df = pd.read_parquet(config.ROOT / row["parquet"])
    S = split_netflow(df, dict(feature_columns=row["feature_columns"]))
    del df
    gc.collect()
    scaler = fit_standardiser(S.x_tr_raw)
    bounds = (S.x_tr_raw.min(axis=0), S.x_tr_raw.max(axis=0))
    return SimpleNamespace(
        tag=tag, corpus=row["corpus"], feature_columns=row["feature_columns"],
        x_tr_raw=S.x_tr_raw, y_tr=S.y_tr, x_te_raw=S.x_te_raw, y_te=S.y_te,
        scaler=scaler, bounds=bounds,
        x_te_attack_raw=S.x_te_raw[S.y_te != TARGET],
        benign_share=row["benign_share"], n_train=int(len(S.y_tr)),
    )


def run_cell(arm, S, share, seed, rate, cost, relations, device, args, gate,
             clean_mlp, ranking, trigger, asr_clean) -> dict:
    """One grid cell. The caller checkpoints after every return."""
    base = dict(arm=arm, corpus=S.corpus, benign_share=share, seed=seed, rate=rate, cost=cost,
                tag=S.tag, sample_benign_share=S.benign_share)
    feas = gate["samples"][S.tag]["feasibility"]["per_rate"][f"{rate}"]
    if not feas["feasible"]:
        # Not a missing cell. A measurement: the benign class cannot fund this rate.
        return dict(**base, status="attack_infeasible",
                    benign_fraction_required=feas["benign_fraction_required"],
                    asr=None, clean_model_stamped_asr=None, backdoor_fraction=None, clean_acc=None)

    x_p_raw, y_p, poison_idx = poison_secondary_trainset_raw(
        S.x_tr_raw, S.y_tr, trigger, rate, target=TARGET, seed=seed)
    poisoned_mlp = train_mlp(apply_standardiser(S.scaler, x_p_raw), y_p, seed,
                             epochs=args.mlp_epochs, device=device)

    atk_std = apply_standardiser(S.scaler, apply_secondary_trigger(S.x_te_attack_raw, trigger))
    asr_poisoned = float(attack_success_rate(poisoned_mlp, atk_std, TARGET, device=device))
    cacc = float(clean_accuracy(poisoned_mlp, apply_standardiser(S.scaler, S.x_te_raw),
                                S.y_te, device=device))

    audit = (audit_netflow_constraints(x_p_raw[poison_idx], relations, S.feature_columns)
             if len(poison_idx) else dict(all_projectable_valid=True, violations={}))

    # Pre-registered guards. Both are recorded on the row; neither drops it.
    saturated = asr_clean >= config.NETFLOW_SATURATION_ASR
    ineffective = asr_poisoned < config.ATTACK_EFFECTIVE_ASR
    status = "saturated" if saturated else ("attack_ineffective" if ineffective else "ok")

    requested = np.asarray(trigger["values"], dtype=np.float64)
    post = np.asarray(trigger["post_projection_values"], dtype=np.float64)
    n_clipped = int((~np.isclose(requested, post)).sum())

    del x_p_raw, poisoned_mlp, atk_std
    gc.collect()

    return dict(**base, status=status, saturated=bool(saturated),
                attack_effective=bool(not ineffective),
                asr=asr_poisoned, clean_model_stamped_asr=asr_clean,
                backdoor_fraction=asr_poisoned - asr_clean, clean_acc=cacc,
                n_poisoned_rows=int(len(poison_idx)),
                trigger_features=list(trigger["feature_names"]),
                trigger_features_clipped_by_projection=n_clipped,
                trigger_cost=cost,
                constraint_audit=dict(
                    all_projectable_valid=bool(audit["all_projectable_valid"]),
                    violations=audit["violations"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit-cells", type=int, default=None,
                    help="compute at most this many NEW cells this invocation")
    ap.add_argument("--mlp-epochs", type=int, default=None)
    ap.add_argument("--arms", default="Q1,Q2")
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    args.mlp_epochs = args.mlp_epochs or (10 if args.smoke else 20)
    arms = args.arms.split(",")
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else list(config.SEEDS))

    config.ensure_dirs()
    if not GATE.exists():
        print(f"missing {GATE}; run scripts/79_netflow_data_gate.py first")
        return 1
    gate = json.loads(GATE.read_text())

    # A full sweep on 40k smoke samples would look like a real result and be worthless. Refuse it.
    if gate.get("smoke") and not args.smoke:
        print(f"REFUSING: {GATE.name} was built by a --smoke gate run "
              f"({gate['n_rows_per_sample']:,} rows per sample). Re-run the gate at full size "
              "before sweeping, or pass --smoke to sweep the smoke samples deliberately.")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q1_corpora = [config.NETFLOW_MANIPULATION_CORPUS, config.NETFLOW_REPLICATION_CORPUS]
    kept_q1, report_q1 = arm_manifest(gate, q1_corpora)
    kept_q2, report_q2 = arm_manifest(gate, list(config.NETFLOW_CORPORA))
    print(f"device={device}  smoke={args.smoke}  epochs={args.mlp_epochs}  seeds={seeds}")
    print(f"Q1 manifest admitted {report_q1['n_admitted']} of {report_q1['n_authored']}; "
          f"Q2 manifest admitted {report_q2['n_admitted']} of {report_q2['n_authored']}")
    if report_q1["admitted"] != report_q2["admitted"]:
        print(f"  NOTE: the two arms admit different relations. Q1-only: "
              f"{sorted(set(report_q1['admitted']) - set(report_q2['admitted']))}")

    key = config_key(args, gate)
    results_path, ckpt_path = results_paths(args.smoke)
    cells = load_checkpoint(ckpt_path, key)
    t0 = time.time()
    n_new = 0
    timings = []

    def budget_left() -> bool:
        return args.limit_cells is None or n_new < args.limit_cells

    # (arm, corpus, share, tag, rates, relations) work list, ordered so each sample loads once.
    work = []
    if "Q1" in arms:
        for corpus in q1_corpora:
            for share in config.NETFLOW_BENIGN_SHARES:
                work.append(("Q1", corpus, share, f"share{share:.2f}__{corpus}",
                             (config.NETFLOW_RATE,), kept_q1))
    if "Q2" in arms:
        for corpus in config.NETFLOW_CORPORA:
            work.append(("Q2", corpus, None, f"native__{corpus}",
                         tuple(config.NETFLOW_NATIVE_RATES), kept_q2))

    for arm, corpus, share, tag, rates, relations in work:
        wanted = [(seed, rate, cost) for seed in seeds for cost in config.NETFLOW_COSTS
                  for rate in rates]
        todo = [w for w in wanted if cell_key(arm, corpus, share, *w) not in cells]
        if not todo or not budget_left():
            if not todo:
                print(f"--- {tag} [{arm}]: complete, skipping (no reload) ---")
            continue

        print(f"--- {tag} [{arm}]: {len(todo)} cells to do "
              f"({time.time() - t0:.0f}s elapsed) ---")
        S = load_sample(gate, tag)
        project_fn = partial(project_netflow_to_feasible, relations=relations,
                             feature_names=S.feature_columns, bounds=S.bounds)
        eligible = list(range(len(S.feature_columns)))   # no exclusions: every relation is an
                                                         # inequality, so nothing erases a stamp
        clean_cache, trig_cache = {}, {}

        for seed in seeds:
            if not budget_left():
                break
            if not any(cell_key(arm, corpus, share, seed, r, c) not in cells
                       for c in config.NETFLOW_COSTS for r in rates):
                continue
            if seed not in clean_cache:
                t_c = time.time()
                x_tr_std = apply_standardiser(S.scaler, S.x_tr_raw)
                clean_mlp = train_mlp(x_tr_std, S.y_tr, seed, epochs=args.mlp_epochs, device=device)
                ranking = shap_rank_features(clean_mlp, S.x_tr_raw, S.scaler, target=TARGET,
                                             kind="mlp", device=device)
                clean_cache[seed] = (clean_mlp, ranking, time.time() - t_c)
                del x_tr_std
                gc.collect()
            clean_mlp, ranking, clean_s = clean_cache[seed]

            for cost in config.NETFLOW_COSTS:
                if not budget_left():
                    break
                if (seed, cost) not in trig_cache:
                    trigger = build_secondary_trigger(
                        ranking, cost, S.x_tr_raw, S.feature_columns, S.bounds, project_fn,
                        eligible, n_sigma=N_SIGMA)
                    stamped = apply_standardiser(
                        S.scaler, apply_secondary_trigger(S.x_te_attack_raw, trigger))
                    asr_clean = float(attack_success_rate(clean_mlp, stamped, TARGET, device=device))
                    trig_cache[(seed, cost)] = (trigger, asr_clean)
                    del stamped
                trigger, asr_clean = trig_cache[(seed, cost)]

                for rate in rates:
                    k = cell_key(arm, corpus, share, seed, rate, cost)
                    if k in cells or not budget_left():
                        continue
                    t_cell = time.time()
                    cells[k] = run_cell(arm, S, share, seed, rate, cost, relations, device,
                                        args, gate, clean_mlp, ranking, trigger, asr_clean)
                    save_checkpoint(ckpt_path, key, cells)
                    dt = time.time() - t_cell
                    timings.append(dt)
                    n_new += 1
                    r = cells[k]
                    if r["status"] == "attack_infeasible":
                        print(f"  {k}  attack_infeasible "
                              f"(needs {r['benign_fraction_required']:.2f} of the benign class)")
                    else:
                        print(f"  {k}  asr={r['asr']:.4f} clean_stamped={r['clean_model_stamped_asr']:.4f} "
                              f"backdoor={r['backdoor_fraction']:+.4f} acc={r['clean_acc']:.4f} "
                              f"clipped={r['trigger_features_clipped_by_projection']}/{cost} "
                              f"[{r['status']}]  {dt:.1f}s")

        del S, clean_cache, trig_cache
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    blob = dict(config_key=key, smoke=args.smoke, device=device, mlp_epochs=args.mlp_epochs,
                n_sigma=N_SIGMA, seeds=seeds,
                manifest_q1=report_q1, manifest_q2=report_q2,
                gate_rows_per_sample=gate["n_rows_per_sample"],
                cells=cells, n_cells=len(cells),
                elapsed_s=round(time.time() - t0, 1))
    results_path.write_text(json.dumps(blob, indent=2))

    print(f"\n{len(cells)} cells on disk, {n_new} computed this invocation, "
          f"{time.time() - t0:.0f}s elapsed")
    if len(timings) >= 2:
        marginal = float(np.mean(timings[1:]))
        print(f"marginal rate {marginal:.1f}s per cell (cells 2+, excluding the first which pays "
              f"CUDA and import warmup)")
    print(f"wrote {results_path.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
