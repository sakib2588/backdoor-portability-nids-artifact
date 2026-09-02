"""Extension Task 8: an EXPLORATORY second-stage SHAP-attribution-concentration filter on top of the
locked MAD z=3.0 (config.MAD_Z_PRIMARY) Spectral rule, run only on the four pre-registered CTU
evasion-window cells (results/analysis.json's `evasion_window` block: (0.005, 8), (0.005, 16),
(0.01, 8), (0.01, 16) -- the cells where the clean-label trigger is attack-effective but Spectral's
fixed removal budget misses most of the poison, and where the MAD rule already recovers full recall
at a measured FPR, per results/evasion_ablation.json).

The current MAD rule already reaches full recall inside the evasion window. A second-stage SHAP filter
might improve PRECISION on top of that, but it can easily become circular because the original attack
is SHAP-guided in the first place. This script is therefore an explicit pilot, not a defence fix: its
own output is rejected (`pilot_passes=False`) for any cell where two-stage recall drops more than 0.02
absolute from MAD recall (Step 2's rejection rule) -- precision gains without preserved recall do not
improve the paper's primary defence claim, and this script's result must not be added to the paper as
a defence improvement without a SEPARATE, pre-registered held-out confirmation run (not run here).

Method (see src/precision_triage.py for the guarded arithmetic):
  Step 1 (per seed, per cost -- shared across both rates at that cost, since trigger_indices depends
  only on cost): train the clean (unpoisoned) MLP, SHAP-rank its features, build that cost's trigger,
  then score+MAD-flag the CLEAN VALIDATION benign rows (x_val_raw, never poisoned, held out by
  `_m2_common.load_setup`) through the SAME clean MLP + Spectral + MAD machinery. Among those
  clean-and-flagged rows, compute concentration = abs_shap[trigger_indices].sum()/abs_shap.sum() (SHAP
  via the clean MLP) and fit the escalation cutoff as its 99th percentile
  (config.CLEAN_CALIBRATION_QUANTILE). `fit_precision_cutoff` refuses anything but this partition.
  Step 2 (per seed, rate, cost): retrain the poisoned MLP (identical recipe to scripts/05/08), score +
  MAD-flag the real poisoned pool, then compute concentration for every MAD-flagged row -- via the SAME
  clean MLP used in Step 1, not the poisoned one (see src/precision_triage.py's docstring for why: this
  keeps calibration and evaluation computed by the same explainer, and avoids the second-stage
  statistic being trivially reconstructible from the poisoned model's own backdoor association, which
  would be a strictly worse circularity than the SHAP-ranking overlap this pilot already accepts).
  two_stage_flag = mad_flag & (concentration >= cutoff): a strict subset of mad_flag by construction,
  so two_stage_recall/precision/fpr can only fall or hold relative to mad_recall/precision/fpr.

Checkpointed per (seed, rate, cost) row, atomic write, resumable -- mirrors
scripts/16_secondary_adaptive_threshold.py's checkpoint pattern exactly (this project's CLAUDE.md:
"Checkpoint every long run"). Each cell needs a full MLP retrain plus TWO SHAP explanation passes
(clean-validation cutoff fit, shared across a cost's two rates; poisoned-pool evaluation, per cell) on
thousands of MAD-flagged rows each -- expect this run to run well over 10 minutes; launch it in tmux.

Run:  python scripts/19_precision_triage.py --seeds 42 --cells 0.005:8   # single-seed/cell smoke check
      python scripts/19_precision_triage.py --seeds 42,123,456,789,1337 \\
          --cells 0.005:8,0.005:16,0.01:8,0.01:16                        # full pre-registered pilot
"""
from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup, apply_overrides
from src import config
from src.data import apply_standardiser
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.models import mlp_penultimate_features, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.precision_triage import (
    concentration_from_abs_shap,
    fit_precision_cutoff,
    poison_fpr,
    poison_precision,
    poison_recall_or_none,
    two_stage_flag_from_concentration,
)
from src.trigger import build_trigger, raw_trigger_stats, shap_abs_matrix, shap_rank_features

TARGET = config.ATTACK_TARGET

# The four pre-registered CTU evasion-window cells (plan Task 8, Step 4) -- exactly these, no 5th
# "control" cell (unlike scripts/08_evasion_ablation.py's TARGET_CELLS, which adds a (0.1, 16) rate-
# matched control; that control is explicitly out of scope for this pilot).
TARGET_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]

# GradientExplainer throughput knobs. Each evasion-window cell's MAD flag covers roughly 6-8k benign
# rows out of ~112k (measured: results/evasion_ablation.json, z=3.0), and every one of them needs a
# per-row SHAP explanation for the second-stage concentration statistic -- unlike shap_rank_features's
# existing callers, which only ever explain a <=500-row background sample once. At shap's own default
# (nsamples=200, unset here) this measured ~0.047s/row on this project's MLP/CTU shape (757 features,
# RTX 3060 Ti) -- for ~20 (seed, cell) combinations x ~8k rows that is >2 hours. nsamples=50 measured
# ~0.012s/row (~4x faster) for the same shape; the concentration statistic only feeds a 99th-percentile
# cutoff and a threshold comparison, both of which tolerate the added per-row sampling noise far better
# than the trigger-selection ranking (shap_rank_features) would, so the reduced sample count is not
# reused there.
SHAP_NSAMPLES = 50
SHAP_MAX_BG = 500

RESULTS_FILENAME = "precision_triage.json"


def row_key(seed, rate, cost) -> str:
    return f"{seed}|{rate}|{cost}"


def _config_key(target_cells, mlp_epochs: int) -> dict:
    return dict(target_cells=[list(c) for c in target_cells], mlp_epochs=mlp_epochs,
                z=config.MAD_Z_PRIMARY, quantile=config.CLEAN_CALIBRATION_QUANTILE,
                shap_nsamples=SHAP_NSAMPLES, shap_max_bg=SHAP_MAX_BG,
                spectral_components=config.SPECTRAL_COMPONENTS)


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
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} row(s) already done")
    return blob.get("rows", {})


def save_checkpoint(path: Path, key: dict, rows: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(path)   # atomic: a crash mid-write never corrupts the checkpoint


def parse_cell_list(argv):
    if "--cells" not in argv:
        return None
    spec = argv[argv.index("--cells") + 1]
    return [(float(r), int(c)) for r, c in (tok.split(":") for tok in spec.split(","))]


def _log_mem(label: str) -> None:
    """Print `free -h` with a label -- a timestamp-correlated host-memory reading for post-hoc
    correlation if this run is ever killed with no traceback (e.g. by the kernel OOM-killer, which can
    kill a process silently depending on cgroup/sandboxing). stdlib-only (subprocess), no new
    dependency; best-effort -- a missing `free` binary must never fail the run over a diagnostic."""
    try:
        out = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=5).stdout.strip()
        print(f"[mem] {label}:\n{out}")
    except Exception as exc:   # noqa: BLE001 -- diagnostic-only, must never abort the real run
        print(f"[mem] {label}: (free -h unavailable: {exc})")


def _load_original_index() -> dict:
    """Index results/detectors.json's main_grid by (seed, rate, cost) for the determinism self-check --
    same purpose as scripts/08_evasion_ablation.py's `_load_original_index`, since this script
    retrains the same seed/rate/cost cells that detectors.json's fixed-budget Spectral recall already
    validated."""
    path = config.RESULTS / "detectors.json"
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    return {(r["seed"], r["rate"], r["cost"]): r for r in blob["main_grid"]}


def main():
    t0 = time.time()
    cfg = apply_overrides(get_config(smoke=False), sys.argv)

    explicit_cells = parse_cell_list(sys.argv)
    if explicit_cells is not None and not set(explicit_cells) <= set(TARGET_CELLS):
        raise SystemExit(
            f"--cells {explicit_cells} includes a cell outside the four pre-registered evasion-window "
            f"cells {TARGET_CELLS} (plan Task 8, Step 4) -- this pilot may only run a subset of exactly "
            "these four cells (e.g. one cell for a smoke check), never a fifth control cell or an "
            "expanded grid. Omit --cells to use the full default four."
        )
    target_cells = explicit_cells if explicit_cells is not None else TARGET_CELLS
    # A --cells subset (e.g. a one-cell smoke check) writes to a SEPARATE filename -- the canonical
    # results/precision_triage.json must never contain a partial grid, even transiently between a smoke
    # run and the real one (this mirrors scripts/16_secondary_adaptive_threshold.py's RESULTS_FILENAME
    # vs RESULTS_FILENAME_SMOKE split).
    is_partial_run = set(target_cells) != set(TARGET_CELLS)
    out_filename = "precision_triage_smoke.json" if is_partial_run else RESULTS_FILENAME
    costs = sorted({c for _, c in target_cells})
    rates_for_cost: dict = defaultdict(list)
    for rate, cost in target_cells:
        rates_for_cost[cost].append(rate)

    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  seeds={cfg['seeds']}  target_cells={target_cells}  "
          f"z={config.MAD_Z_PRIMARY}  quantile={config.CLEAN_CALIBRATION_QUANTILE}  "
          f"shap_nsamples={SHAP_NSAMPLES}")
    _log_mem("run start")

    S = load_setup(cfg)
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    x_val_raw, y_val = S["x_val_raw"], S["y_val"]
    # Quick sanity check (not a full leak-freedom proof -- that's tests/test_data_splits.py's job for
    # temporal_split generically): catches a copy-paste bug that wired x_val_raw to x_tr_raw again.
    assert x_val_raw.shape[0] > 0 and not np.array_equal(x_val_raw[:5], x_tr_raw[:5]), (
        "x_val_raw looks identical to (or empty vs) x_tr_raw -- load_setup's val split appears "
        "mis-wired; refusing to fit a cutoff on what might be training rows"
    )
    ref_index = _load_original_index()

    x_tr_std = apply_standardiser(S["scaler"], x_tr_raw)
    x_val_std = apply_standardiser(S["scaler"], x_val_raw)
    stats = raw_trigger_stats(x_tr_raw, constraints)

    ckpt_key = _config_key(target_cells, cfg["mlp_epochs"])
    ckpt_path = (config.RESULTS / out_filename).with_suffix(".checkpoint.json")
    row_cells = load_checkpoint(ckpt_path, ckpt_key)
    save_cb = lambda: save_checkpoint(ckpt_path, ckpt_key, row_cells)

    for seed in cfg["seeds"]:
        seed_keys = [row_key(seed, rate, cost) for rate, cost in target_cells]
        if all(k in row_cells for k in seed_keys):
            print(f"--- seed {seed}: already complete, skipping (no retrain) ---")
            continue

        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        _log_mem(f"seed={seed} start")
        clean_mlp = train_mlp(x_tr_std, y_tr, seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean_mlp, x_tr_raw, S["scaler"], target=TARGET, kind="mlp",
                                     device=device)

        # --- Step 1 setup: score + MAD-flag the CLEAN validation benign pool through the clean MLP.
        val_benign_pos = np.where(y_val == TARGET)[0]
        feats_val = mlp_penultimate_features(clean_mlp, x_val_std[val_benign_pos], device)
        scores_val = spectral_scores(feats_val, n_components=config.SPECTRAL_COMPONENTS)
        mad_flag_val = flag_by_mad_threshold(scores_val, z_thresh=config.MAD_Z_PRIMARY)
        n_val_flagged = int(mad_flag_val.sum())
        print(f"  clean-validation MAD-flagged rows: {n_val_flagged}/{len(mad_flag_val)}")

        for cost in costs:
            cost_keys = [row_key(seed, rate, cost) for rate in rates_for_cost[cost]]
            if all(k in row_cells for k in cost_keys):
                continue

            realiz, _ = build_trigger(ranking, cost, x_tr_raw, constraints, features, bounds,
                                      stats=stats)
            trigger_indices = realiz["indices"]

            if n_val_flagged == 0:
                raise RuntimeError(
                    f"seed={seed} cost={cost}: zero clean-validation rows flagged by MAD at "
                    f"z={config.MAD_Z_PRIMARY} -- cannot fit an escalation cutoff on nothing. This is "
                    "a data/config problem that needs investigation, not a silent skip."
                )
            x_val_flagged_std = x_val_std[val_benign_pos][mad_flag_val]
            abs_shap_val = shap_abs_matrix(clean_mlp, x_tr_raw, S["scaler"], target=TARGET, kind="mlp",
                                           device=device, max_bg=SHAP_MAX_BG,
                                           x_explain_std=x_val_flagged_std, nsamples=SHAP_NSAMPLES)
            conc_val = concentration_from_abs_shap(abs_shap_val, trigger_indices)
            cutoff = fit_precision_cutoff(conc_val, partition="clean_validation",
                                          contains_poison=False)
            print(f"  cost={cost}: escalation_cutoff={cutoff:.6f} "
                 f"(fit from {n_val_flagged} clean-validation MAD-flagged rows, "
                 f"{time.time() - t0:.0f}s elapsed)")

            for rate in rates_for_cost[cost]:
                key = row_key(seed, rate, cost)
                if key in row_cells:
                    continue

                print(f"  seed={seed} rate={rate} cost={cost}: training poisoned MLP "
                     f"({time.time() - t0:.0f}s elapsed)")
                x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
                    x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
                    constraints=constraints, feature_names=features, bounds=bounds)
                mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

                benign_pos = np.where(y_p == TARGET)[0]
                is_poison = np.isin(benign_pos, poison_idx)
                n_poison, n_benign = int(is_poison.sum()), len(is_poison)
                n_clean = n_benign - n_poison

                feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
                scores = spectral_scores(feats, n_components=config.SPECTRAL_COMPONENTS)

                fixed_flag = flag_by_scores(scores, expected_frac=float(is_poison.mean()))
                fixed_recall = poison_recall_or_none(fixed_flag, is_poison)
                mad_flag = flag_by_mad_threshold(scores, z_thresh=config.MAD_Z_PRIMARY)

                mad_recall = poison_recall_or_none(mad_flag, is_poison)
                mad_precision = poison_precision(mad_flag, is_poison)
                mad_fpr_val = poison_fpr(mad_flag, is_poison)

                n_mad_flagged = int(mad_flag.sum())
                if n_mad_flagged > 0:
                    print(f"  seed={seed} rate={rate} cost={cost}: explaining {n_mad_flagged} "
                         f"MAD-flagged pool rows via SHAP (nsamples={SHAP_NSAMPLES}, "
                         f"{time.time() - t0:.0f}s elapsed)")
                    flagged_pos = np.where(mad_flag)[0]
                    x_pool_flagged_std = x_p_std[benign_pos][flagged_pos]
                    abs_shap_pool = shap_abs_matrix(clean_mlp, x_tr_raw, S["scaler"], target=TARGET,
                                                    kind="mlp", device=device, max_bg=SHAP_MAX_BG,
                                                    x_explain_std=x_pool_flagged_std,
                                                    nsamples=SHAP_NSAMPLES)
                    conc_pool = concentration_from_abs_shap(abs_shap_pool, trigger_indices)
                    two_stage_flag = two_stage_flag_from_concentration(mad_flag, conc_pool, cutoff)
                    del abs_shap_pool, conc_pool, x_pool_flagged_std
                else:
                    two_stage_flag = mad_flag.copy()

                two_stage_recall = poison_recall_or_none(two_stage_flag, is_poison)
                two_stage_precision = poison_precision(two_stage_flag, is_poison)
                two_stage_fpr_val = poison_fpr(two_stage_flag, is_poison)

                ref = ref_index.get((seed, rate, cost))
                determinism_ok = None
                if ref is not None and ref["spectral"]["recall"] is not None and fixed_recall is not None:
                    determinism_ok = bool(np.isclose(fixed_recall, ref["spectral"]["recall"],
                                                     atol=1e-6))
                    if not determinism_ok:
                        print(f"WARNING: rerun fixed-budget recall {fixed_recall} != "
                             f"detectors.json's {ref['spectral']['recall']} at seed={seed} "
                             f"rate={rate} cost={cost}")

                row = dict(
                    seed=seed, rate=rate, cost=cost,
                    mad_recall=mad_recall, mad_precision=mad_precision,
                    two_stage_recall=two_stage_recall, two_stage_precision=two_stage_precision,
                    mad_fpr=mad_fpr_val, two_stage_fpr=two_stage_fpr_val,
                    threshold_fit_partition="clean_validation_only",
                    n_poison=n_poison, n_clean=n_clean, n_benign=n_benign,
                    n_mad_flagged=n_mad_flagged,
                    escalation_cutoff=cutoff,
                    determinism_ok=determinism_ok,
                    fixed_budget_recall=fixed_recall,
                    fixed_budget_recall_ref=(ref["spectral"]["recall"] if ref else None),
                )
                row_cells[key] = row
                save_cb()
                print(f"  seed={seed} rate={rate} cost={cost}: mad_recall={mad_recall} "
                     f"two_stage_recall={two_stage_recall} mad_precision={mad_precision} "
                     f"two_stage_precision={two_stage_precision} mad_fpr={mad_fpr_val} "
                     f"two_stage_fpr={two_stage_fpr_val} ({time.time() - t0:.0f}s elapsed)")

                # OOM-fix discipline (mirrors scripts/16_secondary_adaptive_threshold.py's own
                # documented OOM fix): x_p_std/feats/scores/the flag arrays are all dead once the
                # scalar row above is computed -- free them before the next rate's cell builds its own
                # ~866MB (143046 x 757 x float64) poisoned raw/standardised matrix, rather than letting
                # them sit until the next loop iteration's reassignment implicitly frees them.
                del x_p_std, mlp, feats, scores, fixed_flag, mad_flag, two_stage_flag
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                _log_mem(f"seed={seed} rate={rate} cost={cost} done")

    rows = [row_cells[row_key(seed, rate, cost)]
            for seed in cfg["seeds"] for rate, cost in target_cells]

    # --- Step 2's rejection rule, per cell: reject (pilot_passes=False) if mean two-stage recall drops
    # more than 0.02 absolute from mean MAD recall. Computed here mechanically over whatever cells are
    # actually present in `rows` -- never hand-picked.
    cell_mad_recall, cell_two_stage_recall = defaultdict(list), defaultdict(list)
    for r in rows:
        key = (r["rate"], r["cost"])
        if r["mad_recall"] is not None:
            cell_mad_recall[key].append(r["mad_recall"])
        if r["two_stage_recall"] is not None:
            cell_two_stage_recall[key].append(r["two_stage_recall"])

    summary_cells = []
    for key in sorted(cell_mad_recall):
        mean_mad = float(np.mean(cell_mad_recall[key]))
        mean_two = float(np.mean(cell_two_stage_recall[key])) if cell_two_stage_recall[key] else None
        recall_drop = (mean_mad - mean_two) if mean_two is not None else None
        pilot_passes = bool(recall_drop is not None and recall_drop <= 0.02)
        summary_cells.append(dict(
            rate=key[0], cost=key[1], n_seeds=len(cell_mad_recall[key]),
            mean_mad_recall=mean_mad, mean_two_stage_recall=mean_two,
            recall_drop=recall_drop, pilot_passes=pilot_passes,
        ))

    n_determinism_checked = sum(1 for r in rows if r["determinism_ok"] is not None)
    n_determinism_failed = sum(1 for r in rows if r["determinism_ok"] is False)

    out = dict(
        exploratory=True,
        partial_run=is_partial_run,
        note=(
            "Extension Task 8 exploratory pilot ONLY: a second-stage SHAP-attribution-concentration "
            "filter on top of the locked MAD z=3.0 (config.MAD_Z_PRIMARY) Spectral rule, fit on clean "
            "MAD-flagged validation rows ONLY (threshold_fit_partition='clean_validation_only'). This "
            "is NOT a preregistered defence improvement -- it must not be added to the paper as one "
            "without a separate, pre-registered held-out confirmation run (plan Task 8's own "
            "instruction). pilot_passes is Step 2's rejection rule: mean_mad_recall minus "
            "mean_two_stage_recall must be <= 0.02 absolute; a False here means the pilot is rejected "
            "at that cell, reported honestly, not softened."
        ),
        config=dict(seeds=cfg["seeds"], target_cells=target_cells, mlp_epochs=cfg["mlp_epochs"],
                   z=config.MAD_Z_PRIMARY, quantile=config.CLEAN_CALIBRATION_QUANTILE,
                   shap_nsamples=SHAP_NSAMPLES, shap_max_bg=SHAP_MAX_BG,
                   spectral_components=config.SPECTRAL_COMPONENTS),
        rows=rows,
        summary=dict(
            cells=summary_cells,
            n_determinism_checked=n_determinism_checked,
            n_determinism_failed=n_determinism_failed,
            determinism_all_ok=bool(n_determinism_failed == 0),
            all_cells_pass=bool(all(c["pilot_passes"] for c in summary_cells)) if summary_cells else False,
        ),
    )
    out_path = config.RESULTS / out_filename
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({len(rows)} rows)")
    if n_determinism_failed:
        print(f"WARNING: {n_determinism_failed}/{n_determinism_checked} cells FAILED the "
             "determinism self-check against detectors.json.")
    elapsed = time.time() - t0
    print(f"elapsed={elapsed:.0f}s for {len(cfg['seeds'])} seed(s) x {len(target_cells)} cell(s)")


if __name__ == "__main__":
    main()
