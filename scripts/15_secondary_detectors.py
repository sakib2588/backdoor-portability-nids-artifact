"""Extension Task 5: the secondary (UNSW-NB15) analogue of scripts/05_run_detectors.py -- Spectral
Signatures + Activation Clustering across the full realizable-trigger grid Task 4 already proved
(results/secondary_poison_sweep.json), plus Neural Cleanse on an explicit boundary subset.

Every cell is RETRAINED here (Task 4 checkpointed metrics only, not model weights) using the exact
seed/trigger/poison construction scripts/14_secondary_poison_sweep.py used, then self-checked against
Task 4's own stored ASR/clean-accuracy for that cell -- disagreement beyond atol=1e-6 is a live
determinism bug, not a modeling choice, and is surfaced as `determinism_ok=False` on that row rather
than silently absorbed (mirrors 05_run_detectors.py's D1 self-check).

Spectral scoring/flagging is the exact call sequence plan Task 5 Step 1 specifies:

    scores = spectral_scores(benign_activations, n_components=config.SPECTRAL_COMPONENTS)
    fixed_flags = flag_by_scores(scores, expected_frac=float(is_poison.mean()))
    mad_flags = flag_by_mad_threshold(scores, z_thresh=config.MAD_Z_PRIMARY)

AC calls the existing `ac_detect()` unmodified and records minority-cluster size, purity, recall, and
silhouette -- no manufactured AC score (Step 1). NC stays bounded to an explicit, documented boundary
subset (Step 1's "predeclared boundary subset"): NC_BOUNDARY_RATES/NC_BOUNDARY_COSTS below are the two
lowest secondary poison rates x both secondary trigger costs -- the corner of the grid nearest where
CTU-13's own evasion window sits (rate<=0.01), chosen BEFORE this run, not selected after seeing which
cells turned out interesting. This keeps the same order of magnitude as the primary sweep's own NC
subset (6 cells/seed there; 4 cells/seed here, out of a smaller 8-cell full grid) -- gradient inversion,
not the fixed/AC arms, is the true compute bottleneck (see 05_run_detectors.py's own docstring).

Replication-seed mode (plan Task 5 Step 7): pass `--replication --cells rate:cost,rate:cost,...` to run
ONLY the explicitly-named headline cells (never a full-grid rerun) at whatever seeds `--seeds` gives,
writing to a SEPARATE file (secondary_detectors_replication.json) so an n=10 replication-seed row can
never be silently conflated with the committed n=5 base grid (same discipline as
scripts/14_secondary_poison_sweep.py's smoke/full file separation). NC does not run in replication mode
(it is not part of the Spectral/MAD headline narrative this replication targets).

Run:  python scripts/15_secondary_detectors.py --smoke
      python scripts/15_secondary_detectors.py --seeds 42,123,456,789,1337
      python scripts/15_secondary_detectors.py --replication --seeds 2026,31415,27182,16180,11235 \\
          --cells 0.005:8,0.005:16,0.01:16,0.1:16
"""
from __future__ import annotations

import gc
import itertools
import json
import math
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m3_secondary_common import (
    apply_overrides, cell_key, config_key, eligible_indices_for, get_config,
)
from src import config
from src.constraints_secondary import project_secondary_to_feasible
from src.data import apply_standardiser
from src.data_secondary import load_secondary_setup
from src.detectors import poison_recall
from src.detectors.activation_clustering import detect as ac_detect
from src.detectors.neural_cleanse import (
    balanced_inversion_sample, nc_binary_flag, reverse_engineer_tabular,
)
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.models import attack_success_rate, clean_accuracy, mlp_penultimate_features, train_mlp
from src.poison import poison_secondary_trainset
from src.trigger import apply_secondary_trigger, build_secondary_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
DATASET_ID = config.SECONDARY_DATASET_ID
SILHOUETTE_SAMPLE = 10_000   # AC silhouette is an O(n^2) DIAGNOSTIC only, matches scripts/14's recipe

# Neural Cleanse boundary subset (Step 1's "predeclared boundary subset"), documented above. Overridden
# to the smoke config's own (single) rate/cost so a smoke run never requests a cell outside its own
# reduced grid.
NC_BOUNDARY_RATES: tuple[float, ...] = (0.005, 0.01)
NC_BOUNDARY_COSTS: tuple[int, ...] = (8, 16)

CKPT = "secondary_detectors.checkpoint.json"
CKPT_REPLICATION = "secondary_detectors_replication.checkpoint.json"
CKPT_SMOKE = "secondary_detectors_smoke.checkpoint.json"
RESULTS_FILENAME = "secondary_detectors.json"
RESULTS_FILENAME_REPLICATION = "secondary_detectors_replication.json"
RESULTS_FILENAME_SMOKE = "secondary_detectors_smoke.json"


def _finite_or_none(x):
    """Map a non-finite float (inf from a fully-collapsed NC mask) to None + a saturation flag, so the
    results JSON stays strict RFC-8259 (no `Infinity` token). Mirrors scripts/05_run_detectors.py."""
    return x if (x is not None and math.isfinite(x)) else None


def results_paths(smoke: bool, replication: bool):
    """The (results_json_path, checkpoint_path) pair for this invocation, scoped to SEPARATE
    filenames per mode -- smoke, replication, and the full base-seed grid must never share a filename
    (same bug class 14_secondary_poison_sweep.py's results_paths was fixed for: a smoke or replication
    invocation run AFTER the full grid is committed must never silently overwrite it)."""
    if smoke:
        return config.RESULTS / RESULTS_FILENAME_SMOKE, config.RESULTS / CKPT_SMOKE
    if replication:
        return config.RESULTS / RESULTS_FILENAME_REPLICATION, config.RESULTS / CKPT_REPLICATION
    return config.RESULTS / RESULTS_FILENAME, config.RESULTS / CKPT


def parse_cell_list(argv) -> list[tuple[float, int]] | None:
    """`--cells 0.005:8,0.01:16` -> [(0.005, 8), (0.01, 16)]. None if `--cells` absent, meaning "the
    full cartesian product of cfg['rates'] x cfg['costs']" to the caller."""
    if "--cells" not in argv:
        return None
    spec = argv[argv.index("--cells") + 1]
    cells = []
    for tok in spec.split(","):
        r, c = tok.split(":")
        cells.append((float(r), int(c)))
    return cells


def _config_key(cfg: dict, nc_rates, nc_costs) -> dict:
    """Extends the shared `config_key()` (Task 4's checkpoint-compatibility convention) with the
    NC boundary subset this script alone runs -- reuses the base fields instead of re-listing them,
    so a field added to the shared key can't silently drift out of sync here."""
    key = config_key(cfg)
    key.pop("control_rate", None)
    key.pop("control_cost", None)
    key["nc_boundary_rates"] = list(nc_rates)
    key["nc_boundary_costs"] = list(nc_costs)
    return key


def load_checkpoint(path: Path, key: dict):
    if not path.exists():
        return {}, {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}, {}
    print(f"resuming from checkpoint: {len(blob.get('main', {}))} main-grid cells, "
          f"{len(blob.get('nc', {}))} NC cells already done")
    return blob.get("main", {}), blob.get("nc", {})


def save_checkpoint(path: Path, key: dict, main: dict, nc: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, main=main, nc=nc)))
    tmp.replace(path)   # atomic: a crash mid-write never corrupts the checkpoint


def load_secondary_reference() -> dict:
    """Index results/secondary_poison_sweep.json's per-cell rows by (seed, rate, cost) for the
    determinism self-check (Step 2): a retrained cell's ASR/clean-accuracy MUST match what Task 4
    already measured for that exact seed/rate/cost."""
    path = config.RESULTS / "secondary_poison_sweep.json"
    if not path.exists():
        print("WARNING: results/secondary_poison_sweep.json not found -- determinism self-check "
              "DISABLED (every row will have determinism_ok=None, not verified)")
        return {}
    blob = json.loads(path.read_text())
    return {(r["seed"], r["rate"], r["cost"]): (r["asr"], r["clean_accuracy"]) for r in blob["per_cell"]}


def seed_setup_mlp(seed, S, cfg, device):
    """Per-seed clean secondary MLP + SHAP ranking -- mirrors scripts/14's seed_setup, minus the
    clean-accuracy bookkeeping this script does not need."""
    x_tr_std = apply_standardiser(S.scaler, S.x_train_raw)
    clean_mlp = train_mlp(x_tr_std, S.y_train, seed, epochs=cfg["mlp_epochs"], device=device, log_every=5)
    ranking = shap_rank_features(clean_mlp, S.x_train_raw, S.scaler, target=TARGET, kind="mlp",
                                  device=device)
    return clean_mlp, ranking


def run_main_grid(seed, S, cfg, device, ranking, eligible, ref_index, main_cells, save_cb,
                   cells_to_run=None):
    """Retrain + self-check + Spectral/AC score every (rate, cost) cell. `cells_to_run` defaults to the
    full cartesian product of cfg['rates'] x cfg['costs']; pass an explicit list (replication mode) to
    restrict to pre-identified headline cells only."""
    x_te_std = apply_standardiser(S.scaler, S.x_test_raw)
    x_te_attack_raw = S.x_test_raw[S.y_test != TARGET]
    cells = cells_to_run if cells_to_run is not None else list(
        itertools.product(cfg["rates"], cfg["costs"]))

    trig_cache = {}
    for rate, cost in cells:
        key = cell_key(seed, rate, cost)
        if key in main_cells:
            continue
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

        x_te_attack_trig_raw = apply_secondary_trigger(x_te_attack_raw, realiz)
        asr = attack_success_rate(mlp, apply_standardiser(S.scaler, x_te_attack_trig_raw), TARGET, device)
        cacc = clean_accuracy(mlp, x_te_std, S.y_test, device)

        ref = ref_index.get((seed, rate, cost))
        determinism_ok = None
        if ref is not None:
            determinism_ok = bool(np.isclose(asr, ref[0], atol=1e-6)
                                   and np.isclose(cacc, ref[1], atol=1e-6))
            if not determinism_ok:
                print(f"WARNING: determinism check FAILED at seed={seed} rate={rate} cost={cost} "
                      f"-- retrained asr={asr:.6f} vs Task 4 asr={ref[0]:.6f} "
                      f"(cacc={cacc:.6f} vs {ref[1]:.6f})")

        # detectors operate on the target-class-conditioned (benign) subset of the poisoned TRAINING
        # set -- same convention as scripts/05_run_detectors.py and scripts/14's positive_control.
        benign_pos = np.where(y_p == TARGET)[0]
        is_poison = np.isin(benign_pos, poison_idx)
        has_poison = bool(is_poison.sum() > 0)
        feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
        # OOM fix (2026-07-31, notes/20260731-bug-task5-oom-root-cause.md): same fix as
        # scripts/16_secondary_adaptive_threshold.py's per-cell loop -- x_p_std has no further use
        # once `feats` is extracted, and spectral_scores' internal float64 SVD upcast has a large
        # transient peak of its own (~3.5GB measured on this dataset's scale).
        del x_p_std
        gc.collect()

        # --- Spectral: exact call sequence from plan Task 5 Step 1 ---
        scores = spectral_scores(feats, n_components=config.SPECTRAL_COMPONENTS)
        fixed_flags = flag_by_scores(scores, expected_frac=float(is_poison.mean()))
        mad_flags = flag_by_mad_threshold(scores, z_thresh=config.MAD_Z_PRIMARY)
        spec_recall = poison_recall(fixed_flags, is_poison) if has_poison else None
        spec_mad_recall = poison_recall(mad_flags, is_poison) if has_poison else None
        spec_auc = (float(roc_auc_score(is_poison, scores))
                    if has_poison and is_poison.sum() < len(is_poison) else None)

        # --- AC: existing ac_detect() unmodified, no manufactured score (Step 1) ---
        ac_mask, ac_sil = ac_detect(feats, seed=seed, silhouette_sample=SILHOUETTE_SAMPLE)
        ac_n_flagged = int(ac_mask.sum())
        ac_true_pos = int((ac_mask & is_poison).sum())
        ac_purity = (ac_true_pos / ac_n_flagged) if ac_n_flagged > 0 else None
        ac_recall = poison_recall(ac_mask, is_poison) if has_poison else None

        main_cells[key] = dict(
            dataset=DATASET_ID, seed=seed, rate=rate, cost=cost, asr=asr, clean_acc=cacc,
            determinism_ok=determinism_ok,
            n_poison=int(is_poison.sum()), n_benign=int(len(is_poison)),
            spectral=dict(recall=spec_recall, auc=spec_auc, mad_recall=spec_mad_recall,
                          mad_z=config.MAD_Z_PRIMARY),
            ac=dict(recall=ac_recall, minority_cluster_size=ac_n_flagged, purity=ac_purity,
                    silhouette=ac_sil),
        )
        save_cb()
        # OOM fix: free the trained model + activation matrix before the next cell's own peak.
        del mlp, feats, scores, fixed_flags, mad_flags, ac_mask
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def run_nc_boundary(seed, S, cfg, device, clean_mlp, ranking, eligible, nc_rates, nc_costs,
                     nc_cells, save_cb):
    """Neural Cleanse on the explicit, predeclared boundary subset only -- mirrors
    scripts/05_run_detectors.py's run_nc_boundary and scripts/14's positive_control NC block."""
    std_lo, std_hi = S.std_bounds
    bnd = (torch.tensor(std_lo, dtype=torch.float32), torch.tensor(std_hi, dtype=torch.float32))
    x_tr_std = apply_standardiser(S.scaler, S.x_train_raw)

    # one clean reference per seed -- rate/cost-invariant, computed once (class-balanced inversion
    # sample, same fix as the primary pipeline's; see src/detectors/neural_cleanse.py's docstring)
    sample_clean = balanced_inversion_sample(x_tr_std, S.y_train, cfg["nc_sample"], seed)
    cl_norms = reverse_engineer_tabular(clean_mlp, 2, sample_clean, device=device,
                                         steps=cfg["nc_steps"], bounds=bnd)
    _, cl_ratio = nc_binary_flag(cl_norms, tau=2.0)

    trig_cache = {}
    for cost in nc_costs:
        if cost not in trig_cache:
            project_fn = partial(project_secondary_to_feasible, constraints=S.constraints,
                                  bounds=S.bounds)
            trig_cache[cost] = build_secondary_trigger(
                ranking, cost, S.x_train_raw, S.feature_names, S.bounds, project_fn, eligible,
                n_sigma=cfg["n_sigma"])
        realiz = trig_cache[cost]
        for rate in nc_rates:
            key = cell_key(seed, rate, cost)
            if key in nc_cells:
                continue
            x_p_std, y_p, poison_idx = poison_secondary_trainset(
                S.x_train_raw, S.y_train, realiz, rate, S.scaler, target=TARGET, seed=seed)
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            sample = balanced_inversion_sample(x_p_std, y_p, cfg["nc_sample"], seed)
            bd_norms = reverse_engineer_tabular(mlp, 2, sample, device=device,
                                                 steps=cfg["nc_steps"], bounds=bnd)
            nc_flag, nc_ratio = nc_binary_flag(bd_norms, tau=2.0)
            nc_flagged_target = bool(nc_flag == TARGET and nc_ratio > cl_ratio)

            nc_cells[key] = dict(
                dataset=DATASET_ID, seed=seed, rate=rate, cost=cost,
                nc_flagged_target=nc_flagged_target,
                nc_ratio=_finite_or_none(nc_ratio), nc_clean_ratio=_finite_or_none(cl_ratio),
                nc_ratio_saturated=bool(not math.isfinite(nc_ratio)),
                nc_clean_ratio_saturated=bool(not math.isfinite(cl_ratio)),
                inversion_recipe=dict(nc_sample=cfg["nc_sample"], nc_steps=cfg["nc_steps"],
                                      class_balanced=True, tau=2.0),
            )
            save_cb()
            # OOM fix: same rationale as run_main_grid above.
            del x_p_std, mlp, sample, bd_norms
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()


def _load_secondary_positive_control_summary() -> dict | None:
    """Cite Task 4's already-validated positive-control gate rather than rerunning it (out of this
    script's scope, per Step 1's arm list). None if secondary_poison_sweep.json is missing."""
    path = config.RESULTS / "secondary_poison_sweep.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    return dict(source="secondary_poison_sweep.json",
                all_positive_controls_passed=blob["summary"]["all_positive_controls_passed"],
                n_positive_controls=blob["summary"]["n_positive_controls"],
                failed_control_seeds=blob["summary"]["failed_control_seeds"])


def main():
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    replication = "--replication" in sys.argv
    cells_to_run = parse_cell_list(sys.argv)
    if replication and cells_to_run is None:
        raise SystemExit(
            "--replication requires --cells rate:cost,rate:cost,... -- replication seeds are only "
            "ever spent on pre-identified headline cells (plan Task 5 Step 7), never a full-grid rerun")

    cfg = apply_overrides(get_config(smoke), sys.argv)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nc_rates = cfg["rates"] if smoke else NC_BOUNDARY_RATES
    nc_costs = cfg["costs"] if smoke else NC_BOUNDARY_COSTS

    print(f"dataset={DATASET_ID}  device={device}  smoke={smoke}  replication={replication}  "
          f"seeds={cfg['seeds']}  rates={cfg['rates']}  costs={cfg['costs']}  "
          f"cells_to_run={cells_to_run}  nc_boundary_rates={nc_rates}  nc_boundary_costs={nc_costs}")

    S0 = load_secondary_setup(DATASET_ID)
    x_tr_std = apply_standardiser(S0.scaler, S0.x_train_raw)
    std_bounds = (x_tr_std.min(axis=0), x_tr_std.max(axis=0))
    S = SimpleNamespace(**{f: getattr(S0, f) for f in S0.__dataclass_fields__}, std_bounds=std_bounds)
    eligible = eligible_indices_for(S0.feature_names)
    ref_index = load_secondary_reference()

    key = _config_key(cfg, nc_rates, nc_costs)
    results_path, ckpt_path = results_paths(smoke, replication)
    main_cells, nc_cells = load_checkpoint(ckpt_path, key)
    save_cb = lambda: save_checkpoint(ckpt_path, key, main_cells, nc_cells)

    for seed in cfg["seeds"]:
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        clean_mlp, ranking = seed_setup_mlp(seed, S, cfg, device)
        run_main_grid(seed, S, cfg, device, ranking, eligible, ref_index, main_cells, save_cb,
                      cells_to_run=cells_to_run)
        if not replication:
            run_nc_boundary(seed, S, cfg, device, clean_mlp, ranking, eligible, nc_rates, nc_costs,
                            nc_cells, save_cb)
        print(f"  seed {seed} done ({time.time() - t0:.0f}s elapsed)")

    n_determinism_checked = sum(1 for r in main_cells.values() if r["determinism_ok"] is not None)
    n_determinism_failed = sum(1 for r in main_cells.values() if r["determinism_ok"] is False)

    out = dict(
        dataset=DATASET_ID,
        config=dict(smoke=smoke, replication=replication, seeds=cfg["seeds"],
                   rates=list(cfg["rates"]), costs=list(cfg["costs"]),
                   cells_to_run=(cells_to_run if cells_to_run is not None else None),
                   nc_boundary_rates=(list(nc_rates) if not replication else []),
                   nc_boundary_costs=(list(nc_costs) if not replication else []),
                   nc_coverage_note=(
                       "Neural Cleanse ran on an explicit, predeclared boundary subset -- the two "
                       "lowest secondary poison rates x both secondary trigger costs, the grid corner "
                       "nearest CTU-13's own evasion window -- NOT the full grid. Chosen before this "
                       "run, not after seeing results. Not run at all in replication mode (out of "
                       "scope for the Spectral/MAD headline narrative that mode targets)."
                       if not replication else
                       "Replication-seed run: Neural Cleanse not run (see main-run config note)."),
                   mlp_epochs=cfg["mlp_epochs"], n_sigma=cfg["n_sigma"],
                   nc_sample=cfg["nc_sample"], nc_steps=cfg["nc_steps"]),
        positive_control_ref=_load_secondary_positive_control_summary(),
        main_grid=list(main_cells.values()),
        nc_boundary=list(nc_cells.values()),
        summary=dict(
            n_determinism_checked=n_determinism_checked,
            n_determinism_failed=n_determinism_failed,
            determinism_all_ok=bool(n_determinism_failed == 0),
        ),
    )
    results_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    if n_determinism_failed:
        print(f"WARNING: {n_determinism_failed}/{n_determinism_checked} cells FAILED the "
              "determinism self-check -- do not treat this run's numbers as matching Task 4's until "
              "investigated.")

    elapsed = time.time() - t0
    n_seeds = len(cfg["seeds"])
    print(f"\nelapsed={elapsed:.0f}s for {n_seeds} seed(s) (~{elapsed / max(n_seeds, 1):.0f}s/seed)")
    print(f"wrote {results_path}")
    print(f"checkpoint kept at {ckpt_path}")
    if smoke:
        print("SMOKE RUN: pipeline validation only -- do not cite these numbers in the manuscript.")


if __name__ == "__main__":
    main()
