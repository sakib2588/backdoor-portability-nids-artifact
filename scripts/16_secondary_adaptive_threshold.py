"""Extension Task 5, Step 6: the secondary (UNSW-NB15) analogue of scripts/08_evasion_ablation.py --
does the LOCKED z=3.0 MAD threshold (already chosen on the primary/CTU-13 dataset's own pre-registered
3-way ablation, config.MAD_Z_PRIMARY) recover Spectral recall on the secondary dataset's attack-
effective cells, where scripts/15_secondary_detectors.py's fixed-budget removal rule may have missed?

This is NOT a fresh threshold search: the CTU-13 ablation already tested {2.0, 2.5, 3.0} and locked 3.0
(results/analysis.json's adaptive_threshold block, z_primary=3.0). This script tests only whether that
ALREADY-LOCKED choice replicates here -- passing any other `--z` is refused unless an explicit
`--exploratory-output` path is given, so an exploratory run can never silently become (or overwrite) the
committed secondary_adaptive_threshold.json.

Target cells are derived MECHANICALLY, not hand-picked (plan Task 5 Step 3/6): every (rate, cost) cell
where results/secondary_detectors.json's 5-base-seed main grid shows mean ASR >= config.ATTACK_EFFECTIVE_ASR
("attack-effective"), plus the highest-rate cost=16 cell as a rate-matched sanity-check control (mirrors
scripts/08's own TARGET_CELLS, which always included the (0.1, 16) cell where the fixed budget already
worked, to prove the new rule does not break a working case). The MAD result is reported for EVERY
attack-effective cell, not only the ones that missed under the fixed budget (Step 3) -- this prevents a
favourable subset from hiding an FPR or recall failure elsewhere in the attack-effective set.

Each cell is RETRAINED here (an independent rerun, same discipline scripts/08 uses against
scripts/05's detectors.json) and self-checked against results/secondary_detectors.json's own
fixed-budget Spectral/AC recall for that seed/rate/cost.

Replication-seed mode (Step 7): pass `--cells rate:cost,...` to restrict to explicitly pre-identified
headline cells (never auto-derived) at whatever `--seeds` gives -- this is how the additional 5
replication seeds are spent, and only on cells already identified from the base 5-seed run.

Run:  python scripts/16_secondary_adaptive_threshold.py --smoke
      python scripts/16_secondary_adaptive_threshold.py --seeds 42,123,456,789,1337 --z 3.0
      python scripts/16_secondary_adaptive_threshold.py --seeds 2026,31415,27182,16180,11235 \\
          --cells 0.005:8,0.005:16,0.01:16,0.1:16 --output secondary_adaptive_threshold_replication.json
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
from src.analysis_extension import classify_evasion_window, select_primary_mad_result, summarise_dataset_rows
from src.constraints_secondary import manifest_fingerprint, project_secondary_to_feasible
from src.data import apply_standardiser
from src.data_secondary import load_secondary_setup
from src.detectors import poison_recall
from src.detectors.activation_clustering import detect as ac_detect
from src.detectors.spectral import flag_by_mad_threshold, flag_by_scores, spectral_scores
from src.models import attack_success_rate, clean_accuracy, mlp_penultimate_features, train_mlp
from src.poison import poison_secondary_trainset
from src.trigger import apply_secondary_trigger, build_secondary_trigger, shap_rank_features

TARGET = config.ATTACK_TARGET
DATASET_ID = config.SECONDARY_DATASET_ID
SILHOUETTE_SAMPLE = 10_000

RESULTS_FILENAME = "secondary_adaptive_threshold.json"
RESULTS_FILENAME_SMOKE = "secondary_adaptive_threshold_smoke.json"


def row_key(seed, rate, cost) -> str:
    return f"{seed}|{rate}|{cost}"


def _config_key(z: float, target_cells, mlp_epochs: int, n_sigma: float) -> dict:
    return dict(z=z, target_cells=[list(c) for c in target_cells], mlp_epochs=mlp_epochs,
                n_sigma=n_sigma, constraints=manifest_fingerprint())


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


def parse_cell_list(argv) -> list[tuple[float, int]] | None:
    if "--cells" not in argv:
        return None
    spec = argv[argv.index("--cells") + 1]
    return [(float(r), int(c)) for r, c in (tok.split(":") for tok in spec.split(","))]


def parse_z(argv) -> float:
    if "--z" not in argv:
        return config.MAD_Z_PRIMARY
    return float(argv[argv.index("--z") + 1])


def parse_output(argv, default_name: str) -> Path:
    if "--output" in argv:
        return config.RESULTS / argv[argv.index("--output") + 1]
    return config.RESULTS / default_name


def _load_detector_reference(smoke: bool) -> dict:
    """Index scripts/15's main_grid rows by (seed, rate, cost) -- both for the mechanical
    attack-effective-cell derivation and the per-cell determinism self-check."""
    name = "secondary_detectors_smoke.json" if smoke else "secondary_detectors.json"
    path = config.RESULTS / name
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/15_secondary_detectors.py first "
                          "(this script derives its target cells mechanically from its output)")
    blob = json.loads(path.read_text())
    return blob


def derive_target_cells(detector_blob: dict) -> list[tuple[float, int]]:
    """Mechanical target-cell derivation (Step 3/6): every (rate, cost) with mean ASR >=
    config.ATTACK_EFFECTIVE_ASR across the base grid's seeds, plus the highest-rate, cost=16 cell as a
    rate-matched sanity-check control (mirrors scripts/08's own inclusion of the CTU (0.1, 16) cell)."""
    rows = detector_blob["main_grid"]
    if not rows:
        raise SystemExit("secondary_detectors.json has an empty main_grid -- nothing to derive from")
    summary = summarise_dataset_rows(rows)
    attack_effective = {(c["rate"], c["cost"]) for c in summary["cells"]
                        if c["mean_asr"] >= config.ATTACK_EFFECTIVE_ASR}
    rates_present = {r for r, _ in ((c["rate"], c["cost"]) for c in summary["cells"])}
    control_cell = (max(rates_present), 16) if rates_present else None
    target = sorted(attack_effective | ({control_cell} if control_cell else set()))
    return target, attack_effective, control_cell


def main():
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    z = parse_z(sys.argv)
    exploratory_output = "--exploratory-output" in sys.argv
    if z != config.MAD_Z_PRIMARY and not exploratory_output:
        raise SystemExit(
            f"--z {z} rejected: only the predeclared config.MAD_Z_PRIMARY={config.MAD_Z_PRIMARY} may "
            f"write {RESULTS_FILENAME}. Pass --exploratory-output <path> to run at another z for "
            "debugging only -- it writes to that path, never the committed artifact.")

    out_path = (config.RESULTS / sys.argv[sys.argv.index("--exploratory-output") + 1]
                if exploratory_output else
                parse_output(sys.argv, RESULTS_FILENAME_SMOKE if smoke else RESULTS_FILENAME))

    cfg = apply_overrides(get_config(smoke), sys.argv)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detector_blob = _load_detector_reference(smoke)
    explicit_cells = parse_cell_list(sys.argv)
    if explicit_cells is not None:
        target_cells = explicit_cells
        attack_effective, control_cell = set(explicit_cells), None
    else:
        target_cells, attack_effective, control_cell = derive_target_cells(detector_blob)
    ref_index = {(r["seed"], r["rate"], r["cost"]): r for r in detector_blob["main_grid"]}

    print(f"dataset={DATASET_ID}  device={device}  smoke={smoke}  z={z}  seeds={cfg['seeds']}  "
          f"target_cells={target_cells}  attack_effective={sorted(attack_effective)}  "
          f"control_cell={control_cell}  out={out_path}")

    S0 = load_secondary_setup(DATASET_ID)
    eligible = eligible_indices_for(S0.feature_names)
    S = SimpleNamespace(**{f: getattr(S0, f) for f in S0.__dataclass_fields__})

    ckpt_key = _config_key(z, target_cells, cfg["mlp_epochs"], cfg["n_sigma"])
    ckpt_path = out_path.with_suffix(".checkpoint.json")
    row_cells = load_checkpoint(ckpt_path, ckpt_key)
    save_cb = lambda: save_checkpoint(ckpt_path, ckpt_key, row_cells)

    for seed in cfg["seeds"]:
        seed_keys = [row_key(seed, rate, cost) for rate, cost in target_cells]
        if all(k in row_cells for k in seed_keys):
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
            if key in row_cells:
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
            asr = attack_success_rate(mlp, apply_standardiser(S.scaler, x_te_attack_trig_raw), TARGET,
                                      device)
            cacc = clean_accuracy(mlp, x_te_std, S.y_test, device)

            benign_pos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(benign_pos, poison_idx)
            has_poison = bool(is_poison.sum() > 0)
            n_poison, n_benign = int(is_poison.sum()), len(is_poison)
            n_clean = n_benign - n_poison
            feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
            # OOM fix (2026-07-31, notes/20260731-bug-task5-oom-root-cause.md): x_p_std (the full
            # ~1.27M-row poisoned/standardised training matrix, ~0.39GB) has no further use once
            # `feats` is extracted -- mlp is already trained, and the detector calls below only need
            # `feats`. spectral_scores' internal float64 SVD upcast measured a ~3.5GB transient peak
            # on its own (see the decision note); freeing this array first stops that peak from
            # stacking on top of a still-resident poisoned-training-set copy.
            del x_p_std
            gc.collect()

            scores = spectral_scores(feats, n_components=config.SPECTRAL_COMPONENTS)
            fixed_flag = flag_by_scores(scores, expected_frac=float(is_poison.mean()))
            fixed_recall = poison_recall(fixed_flag, is_poison) if has_poison else None
            auc = (float(roc_auc_score(is_poison, scores))
                  if has_poison and is_poison.sum() < len(is_poison) else None)

            mad_flag = flag_by_mad_threshold(scores, z_thresh=z)
            n_true_pos = int(np.sum(mad_flag & is_poison))
            n_false_pos = int(mad_flag.sum()) - n_true_pos
            mad_recall = poison_recall(mad_flag, is_poison) if has_poison else None
            mad_fpr = (n_false_pos / n_clean) if n_clean > 0 else None
            mad_result = dict(recall=mad_recall, n_flagged=int(mad_flag.sum()),
                              n_true_positive=n_true_pos, n_false_positive=n_false_pos, fpr=mad_fpr)

            ac_mask, _ = ac_detect(feats, seed=seed, silhouette_sample=SILHOUETTE_SAMPLE)
            ac_fixed_recall = poison_recall(ac_mask, is_poison) if has_poison else None

            # OOM fix: mlp/feats/scores/fixed_flag/mad_flag/ac_mask are all dead once the scalar
            # results above are computed -- free them (and the CUDA allocator's cache for `mlp`, VRAM
            # not system RAM but cheap to release) before the next cell's own peak, rather than
            # letting them sit until the next loop iteration's reassignment implicitly frees them.
            del mlp, feats, scores, fixed_flag, mad_flag, ac_mask
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            ref = ref_index.get((seed, rate, cost))
            determinism_ok = None
            if ref is not None:
                ref_recall = ref["spectral"]["recall"]
                if ref_recall is not None and fixed_recall is not None:
                    determinism_ok = bool(np.isclose(fixed_recall, ref_recall, atol=1e-6))
                    if not determinism_ok:
                        print(f"WARNING: rerun fixed-budget recall {fixed_recall:.6f} != "
                             f"secondary_detectors.json's {ref_recall:.6f} at seed={seed} "
                             f"rate={rate} cost={cost}")

            row = dict(
                dataset=DATASET_ID, seed=seed, rate=rate, cost=cost,
                is_control_cell=bool(control_cell is not None and (rate, cost) == control_cell),
                is_attack_effective=bool((rate, cost) in attack_effective),
                asr=asr, clean_acc=cacc,
                n_poison=n_poison, n_benign=n_benign, n_clean=n_clean,
                fixed_budget_recall=fixed_recall,
                fixed_budget_recall_ref=(ref["spectral"]["recall"] if ref else None),
                spectral_auc=auc,
                determinism_ok=determinism_ok,
                mad={str(z): mad_result},
                ac_fixed_recall=ac_fixed_recall,
                ac_fixed_recall_ref=(ref["ac"]["recall"] if ref else None),
            )
            row_cells[key] = row
            save_cb()
            print(f"  rate={rate} cost={cost}: asr={asr:.4f} fixed_recall={fixed_recall} "
                 f"(ref={ref['spectral']['recall'] if ref else None}) auc={auc} "
                 f"mad(z={z})_recall={mad_recall} mad_fpr={mad_fpr} ac_fixed_recall={ac_fixed_recall}")

    rows = [row_cells[row_key(seed, rate, cost)]
            for seed in cfg["seeds"] for rate, cost in target_cells]

    # --- mechanical evasion-window classification per cell (Step 3), applied to the seed-aggregated
    # cells actually present in `rows` -- never hand-picked.
    from collections import defaultdict
    cell_asr, cell_fixed, cell_mad_recall, cell_mad_fpr = (defaultdict(list) for _ in range(4))
    for r in rows:
        key = (r["rate"], r["cost"])
        cell_asr[key].append(r["asr"])
        if r["fixed_budget_recall"] is not None:
            cell_fixed[key].append(r["fixed_budget_recall"])
        m = select_primary_mad_result(r["mad"], primary_z=z)
        if m["recall"] is not None:
            cell_mad_recall[key].append(m["recall"])
        if m["fpr"] is not None:
            cell_mad_fpr[key].append(m["fpr"])

    per_cell_summary = []
    for key in sorted(cell_asr):
        mean_asr = float(np.mean(cell_asr[key]))
        mean_fixed = float(np.mean(cell_fixed[key])) if cell_fixed[key] else None
        per_cell_summary.append(dict(
            rate=key[0], cost=key[1], n_seeds=len(cell_asr[key]), mean_asr=mean_asr,
            mean_fixed_budget_recall=mean_fixed,
            mean_mad_recall=(float(np.mean(cell_mad_recall[key])) if cell_mad_recall[key] else None),
            mean_mad_fpr=(float(np.mean(cell_mad_fpr[key])) if cell_mad_fpr[key] else None),
            is_evasion_window=(classify_evasion_window(mean_asr, mean_fixed)
                               if mean_fixed is not None else False),
        ))

    n_determinism_checked = sum(1 for r in rows if r["determinism_ok"] is not None)
    n_determinism_failed = sum(1 for r in rows if r["determinism_ok"] is False)

    out = dict(
        dataset=DATASET_ID, z=z, z_primary=config.MAD_Z_PRIMARY,
        exploratory=exploratory_output,
        config=dict(smoke=smoke, seeds=cfg["seeds"], target_cells=target_cells,
                   attack_effective_cells=sorted(attack_effective), control_cell=control_cell,
                   mlp_epochs=cfg["mlp_epochs"],
                   note="target_cells = every (rate, cost) with mean ASR >= "
                        f"config.ATTACK_EFFECTIVE_ASR ({config.ATTACK_EFFECTIVE_ASR}) in "
                        "secondary_detectors.json's base grid, plus the highest-rate cost=16 cell as "
                        "a rate-matched sanity-check control -- derived mechanically, not hand-picked "
                        "(unless --cells was given explicitly, e.g. replication-seed mode)."),
        rows=rows,
        per_cell=per_cell_summary,
        summary=dict(n_determinism_checked=n_determinism_checked,
                    n_determinism_failed=n_determinism_failed,
                    determinism_all_ok=bool(n_determinism_failed == 0)),
    )
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({len(rows)} rows)")
    if n_determinism_failed:
        print(f"WARNING: {n_determinism_failed}/{n_determinism_checked} cells FAILED the "
             "determinism self-check against secondary_detectors.json.")
    elapsed = time.time() - t0
    print(f"elapsed={elapsed:.0f}s for {len(cfg['seeds'])} seed(s) x {len(target_cells)} cell(s)")


if __name__ == "__main__":
    main()
