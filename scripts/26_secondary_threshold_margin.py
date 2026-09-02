"""Why is the MAD threshold's recall unstable on UNSW-NB15? (H-margin)

Pre-registered in notes/20260803-decision-threshold-margin-preregistration.md. That note records
what was already eliminated from artefacts on disk -- nondeterminism, preprocessing, train/test
shift, victim-training instability, poison-budget drift, a defect in the threshold rule, and (the
falsified leading hypothesis) SHAP trigger-ranking instability. Seeds 42 and 1337 in cell
(0.01, 8) carry byte-identical triggers and still land at MAD recall 0.9991 vs 0.6438.

H-margin: the MAD cut sits close to the poison's score mass on UNSW-NB15, so small representation
differences move a large fraction of the poison across it. `flag_by_mad_threshold` flags when
`0.6745 * (score - median) / MAD > z`, i.e. `score > median + z * MAD / 0.6745`. Its own docstring
notes the 0.6745 scaling is normal-consistent while `spectral_scores` are right-skewed, so z=3.0 is
tuned to CTU-13's score shape rather than being a transferable tail probability.

This script re-runs the three UNSW-NB15 evasion-window cells across ten seeds, mirroring
scripts/16_secondary_adaptive_threshold.py's setup, seeding, trigger construction and poisoning
exactly, and additionally records the score-distribution geometry that H-margin predicts about:

  * median, MAD, and the implied cut for every z in the sweep
  * poison and clean score percentiles
  * the standardised margin (median(poison) - cut) / MAD
  * the fraction of poison above each cut
  * recall / FPR / flag counts at every z

CORRECTNESS GATE: at z=3.0 each row must reproduce the recall already committed in
results/secondary_adaptive_threshold.json (or its replication file) to within 1e-6. The run reports
`z3_matches_committed` per row rather than assuming it. A False there means this script has drifted
from the committed pipeline and none of its other numbers can be trusted.

Checkpointed per (seed, cell), config-fingerprinted, atomic temp-then-rename write, resumable --
the pattern from scripts/05_run_detectors.py and scripts/16, not a new scheme.

Usage:
  python scripts/26_secondary_threshold_margin.py                      # 3 cells x 10 seeds
  python scripts/26_secondary_threshold_margin.py --seeds 42,123       # pilot / calibration
  python scripts/26_secondary_threshold_margin.py --cells 0.01:8       # single cell
  python scripts/26_secondary_threshold_margin.py --smoke              # wiring check only
"""
from __future__ import annotations

import gc
import json
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

from _m3_secondary_common import apply_overrides, eligible_indices_for, get_config
from src import config
from src.constraints_secondary import manifest_fingerprint, project_secondary_to_feasible
from src.data import apply_standardiser
from src.data_secondary import load_secondary_setup
from src.detectors import poison_recall
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.models import attack_success_rate, clean_accuracy, mlp_penultimate_features, train_mlp
from src.poison import poison_secondary_trainset
from src.trigger import apply_secondary_trigger, build_secondary_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
DATASET_ID = config.SECONDARY_DATASET_ID

RESULTS_FILENAME = "secondary_threshold_margin.json"
RESULTS_FILENAME_SMOKE = "secondary_threshold_margin_smoke.json"

# The three UNSW-NB15 evasion-window cells, as classified in results/secondary_adaptive_threshold.json
# (is_evasion_window == true). Hard-coded rather than re-derived so this diagnostic cannot silently
# retarget itself if an upstream file is regenerated.
WINDOW_CELLS = [(0.005, 16), (0.01, 8), (0.01, 16)]

# Base five seeds plus the five replication seeds already spent on these cells.
DEFAULT_SEEDS = [42, 123, 456, 789, 1337, 2026, 31415, 27182, 16180, 11235]

Z_GRID = [2.0, 2.5, 3.0, 3.5, 4.0]

# flag_by_mad_threshold's normal-consistency constant. Kept as a named constant so the cut recorded
# here provably matches the rule being measured rather than re-deriving a slightly different one.
MAD_SCALE = 0.6745

COMMITTED_FILES = ["secondary_adaptive_threshold.json",
                   "secondary_adaptive_threshold_replication.json"]


def row_key(seed, rate, cost) -> str:
    return f"{seed}|{rate}|{cost}"


def _config_key(cells, seeds, z_grid, mlp_epochs, n_sigma) -> dict:
    """Every input that changes a result, so a stale checkpoint can never be silently reused."""
    return dict(cells=[list(c) for c in cells], seeds=list(seeds), z_grid=list(z_grid),
                mlp_epochs=mlp_epochs, n_sigma=n_sigma, spectral_components=config.SPECTRAL_COMPONENTS,
                mad_scale=MAD_SCALE, dataset=DATASET_ID, constraints=manifest_fingerprint())


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
        return list(WINDOW_CELLS)
    spec = argv[argv.index("--cells") + 1]
    return [(float(r), int(c)) for r, c in (tok.split(":") for tok in spec.split(","))]


def parse_output(argv, default_name: str) -> Path:
    if "--output" in argv:
        return config.RESULTS / argv[argv.index("--output") + 1]
    return config.RESULTS / default_name


def load_committed_z3() -> dict:
    """Index the already-committed z=3.0 MAD recalls by (seed, rate, cost) for the correctness gate."""
    index = {}
    for name in COMMITTED_FILES:
        path = config.RESULTS / name
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        for r in blob.get("rows", []):
            mad = r.get("mad", {}).get("3.0")
            if mad is not None:
                index[(r["seed"], r["rate"], r["cost"])] = mad.get("recall")
    return index


def score_geometry(scores: np.ndarray, is_poison: np.ndarray, z_grid) -> dict:
    """The distribution shape H-margin is about, recorded once per run.

    `cut(z)` inverts flag_by_mad_threshold's rule: it flags when 0.6745*(s-med)/MAD > z, so the
    implied score cut is med + z*MAD/0.6745. Recording it lets the margin be read directly rather
    than inferred from recall.
    """
    scores = np.asarray(scores, dtype=float)
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))
    poison_scores = scores[is_poison]
    clean_scores = scores[~is_poison]
    pct = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    geom = dict(
        median=med, mad=mad, mad_is_degenerate=bool(mad <= 0),
        score_min=float(scores.min()), score_max=float(scores.max()),
        poison_percentiles={str(p): float(np.percentile(poison_scores, p)) for p in pct},
        clean_percentiles={str(p): float(np.percentile(clean_scores, p)) for p in pct},
        per_z={},
    )
    for z in z_grid:
        if mad > 0:
            cut = med + z * mad / MAD_SCALE
            # Standardised margin: how many MADs the poison's median sits above the cut. Negative
            # means the median poison point is not flagged at all -- the collapse H-margin predicts.
            margin = (float(np.median(poison_scores)) - cut) / mad
        else:
            cut, margin = None, None
        geom["per_z"][str(z)] = dict(
            cut=cut,
            poison_median_margin_in_mads=margin,
            poison_frac_above_cut=(None if cut is None
                                   else float((poison_scores > cut).mean())),
            clean_frac_above_cut=(None if cut is None
                                  else float((clean_scores > cut).mean())),
        )
    return geom


def evaluate_z_grid(scores, is_poison, n_clean, z_grid) -> dict:
    """recall / FPR / counts at every z, so the steepness of recall(z) is measurable."""
    out = {}
    for z in z_grid:
        flag = flag_by_mad_threshold(scores, z_thresh=z)
        tp = int(np.sum(flag & is_poison))
        fp = int(flag.sum()) - tp
        out[str(z)] = dict(
            recall=poison_recall(flag, is_poison),
            n_flagged=int(flag.sum()), n_true_positive=tp, n_false_positive=fp,
            fpr=(fp / n_clean) if n_clean > 0 else None,
        )
    return out


def main() -> None:
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    out_path = parse_output(sys.argv, RESULTS_FILENAME_SMOKE if smoke else RESULTS_FILENAME)

    cfg = apply_overrides(get_config(smoke), sys.argv)
    if "--seeds" not in sys.argv and not smoke:
        cfg["seeds"] = list(DEFAULT_SEEDS)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target_cells = [(0.01, 16)] if smoke else parse_cell_list(sys.argv)
    committed_z3 = load_committed_z3()

    print(f"dataset={DATASET_ID}  device={device}  smoke={smoke}  seeds={cfg['seeds']}  "
          f"cells={target_cells}  z_grid={Z_GRID}  out={out_path}")
    print(f"correctness gate: {len(committed_z3)} committed z=3.0 recalls available for comparison")

    S0 = load_secondary_setup(DATASET_ID)
    eligible = eligible_indices_for(S0.feature_names)
    S = SimpleNamespace(**{f: getattr(S0, f) for f in S0.__dataclass_fields__})

    ckpt_key = _config_key(target_cells, cfg["seeds"], Z_GRID, cfg["mlp_epochs"], cfg["n_sigma"])
    ckpt_path = out_path.with_suffix(".checkpoint.json")
    rows = load_checkpoint(ckpt_path, ckpt_key)
    save_cb = lambda: save_checkpoint(ckpt_path, ckpt_key, rows)

    for seed in cfg["seeds"]:
        seed_keys = [row_key(seed, rate, cost) for rate, cost in target_cells]
        if all(k in rows for k in seed_keys):
            print(f"--- seed {seed}: already complete, skipping (no retrain) ---")
            continue

        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        x_tr_std = apply_standardiser(S.scaler, S.x_train_raw)
        clean_mlp = train_mlp(x_tr_std, S.y_train, seed, epochs=cfg["mlp_epochs"], device=device,
                              log_every=5)
        ranking = shap_rank_features(clean_mlp, S.x_train_raw, S.scaler, target=TARGET, kind="mlp",
                                     device=device)
        x_te_std = apply_standardiser(S.scaler, S.x_test_raw)
        x_te_attack_raw = S.x_test_raw[S.y_test != TARGET]

        trig_cache = {}
        for rate, cost in target_cells:
            key = row_key(seed, rate, cost)
            if key in rows:
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
            asr = attack_success_rate(mlp, apply_standardiser(S.scaler, x_te_attack_trig_raw),
                                      TARGET, device)
            cacc = clean_accuracy(mlp, x_te_std, S.y_test, device)

            benign_pos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(benign_pos, poison_idx)
            has_poison = bool(is_poison.sum() > 0)
            n_poison, n_benign = int(is_poison.sum()), len(is_poison)
            n_clean = n_benign - n_poison
            feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
            # Same OOM discipline as scripts/16: free the full poisoned matrix before
            # spectral_scores' float64 SVD upcast peaks. See notes/20260731-bug-task5-oom-root-cause.md
            del x_p_std
            gc.collect()

            scores = spectral_scores(feats, n_components=config.SPECTRAL_COMPONENTS)
            fixed_flag = flag_by_scores(scores, expected_frac=float(is_poison.mean()))
            fixed_recall = poison_recall(fixed_flag, is_poison) if has_poison else None
            auc = (float(roc_auc_score(is_poison, scores))
                   if has_poison and is_poison.sum() < len(is_poison) else None)

            z_results = evaluate_z_grid(scores, is_poison, n_clean, Z_GRID) if has_poison else {}
            geometry = score_geometry(scores, is_poison, Z_GRID) if has_poison else {}

            # Correctness gate: does z=3.0 reproduce the committed number?
            committed = committed_z3.get((seed, rate, cost))
            here = z_results.get("3.0", {}).get("recall")
            if committed is None or here is None:
                z3_match = None
            else:
                z3_match = bool(np.isclose(here, committed, atol=1e-6))
                if not z3_match:
                    print(f"WARNING: z=3.0 recall {here:.6f} != committed {committed:.6f} at "
                          f"seed={seed} rate={rate} cost={cost} -- pipeline has drifted")

            del mlp, feats, scores, fixed_flag
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            rows[key] = dict(
                dataset=DATASET_ID, seed=seed, rate=rate, cost=cost,
                asr=asr, clean_acc=cacc,
                n_poison=n_poison, n_benign=n_benign, n_clean=n_clean,
                fixed_budget_recall=fixed_recall, spectral_auc=auc,
                mad_by_z=z_results, geometry=geometry,
                committed_z3_recall=committed, z3_matches_committed=z3_match,
            )
            save_cb()   # checkpoint after every unit of work, not every seed
            m3 = geometry.get("per_z", {}).get("3.0", {})
            print(f"  rate={rate} cost={cost}: AUC={auc:.4f} "
                  f"recall@3.0={here if here is None else round(here, 4)} "
                  f"margin={m3.get('poison_median_margin_in_mads')} "
                  f"z3_gate={z3_match}")

        del clean_mlp
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    ordered = [rows[k] for k in sorted(rows, key=lambda s: (int(s.split("|")[0]), s))]
    gate = [r["z3_matches_committed"] for r in ordered if r["z3_matches_committed"] is not None]
    blob = dict(
        dataset=DATASET_ID, z_grid=Z_GRID, mad_scale=MAD_SCALE,
        config=dict(seeds=cfg["seeds"], cells=[list(c) for c in target_cells],
                    mlp_epochs=cfg["mlp_epochs"], n_sigma=cfg["n_sigma"],
                    spectral_components=config.SPECTRAL_COMPONENTS, smoke=smoke),
        prereg="notes/20260803-decision-threshold-margin-preregistration.md",
        rows=ordered,
        summary=dict(n_rows=len(ordered), n_z3_checked=len(gate),
                     n_z3_failed=int(sum(1 for g in gate if not g)),
                     z3_gate_all_ok=bool(gate) and all(gate)),
    )
    out_path.write_text(json.dumps(blob, indent=1))
    print(f"\nwrote {out_path}  ({len(ordered)} rows, {time.time() - t0:.0f}s)")
    print(f"z=3.0 correctness gate: {len(gate)} checked, "
          f"{sum(1 for g in gate if not g)} failed")


if __name__ == "__main__":
    main()
