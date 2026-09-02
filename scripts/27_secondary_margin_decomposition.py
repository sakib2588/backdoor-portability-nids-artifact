"""Which seed-dependent input drives the UNSW-NB15 margin collapse: MLP initialisation, or which
benign rows carry the poison?

Follow-up to notes/20260803-experiment-threshold-margin-result.md, which closed the *what* (the MAD
cut's distance from the poison score mass separates all 30 runs perfectly, where AUC overlaps) and
left the *why* open. Three inputs vary with the seed in the committed pipeline:

  1. the SHAP ranking, via the clean MLP trained on that seed -> which features the trigger stamps
  2. which benign rows are sampled to carry the poison
  3. the poisoned MLP's weight initialisation and batch order

(1) is already falsified: seeds 42 and 1337 in cell (0.01, 8) carry byte-identical triggers and
still land at recall 0.9991 vs 0.6438. This script holds (1) fixed by construction -- one clean MLP,
one SHAP ranking, one trigger, reused for every row -- and then varies (2) and (3) one at a time:

  arm "vary_init"   : ranking=42, poison_rows=42, mlp_init=s   -> isolates initialisation
  arm "vary_poison" : ranking=42, poison_rows=s,  mlp_init=42  -> isolates poison-row sampling

Both arms include s=42, which is the same configuration in each and must produce identical rows --
a built-in internal consistency check reported as `arm_agreement_ok`.

Prediction registered before running: if initialisation drives it, "vary_init" reproduces the wide
margin spread seen in results/secondary_threshold_margin.json (-4.08 to +8.36 on this cell) while
"vary_poison" stays tight; if poison-row sampling drives it, the reverse; if both spread, the cause
is their interaction and neither alone is the lever.

Runs one cell only -- (0.01, 16), the cell with the widest observed margin range -- to keep this
cheap. Reuses scripts/26's geometry and z-grid measurements verbatim so the numbers are directly
comparable to that artefact.

Cost note: the clean MLP and SHAP ranking are computed ONCE, not per seed, because the ranking is
held fixed by design. That is what makes this ~20 rows at roughly one poisoned-MLP training each
rather than scripts/26's full per-seed cost.

Checkpointed per (arm, seed), config-fingerprinted, atomic temp-then-rename write, resumable.

Usage:
  python scripts/27_secondary_margin_decomposition.py
  python scripts/27_secondary_margin_decomposition.py --seeds 42,1337    # pilot
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
_m26 = import_module("26_secondary_threshold_margin")
score_geometry, evaluate_z_grid = _m26.score_geometry, _m26.evaluate_z_grid
Z_GRID, MAD_SCALE = _m26.Z_GRID, _m26.MAD_SCALE

TARGET = config.ATTACK_TARGET
DATASET_ID = config.SECONDARY_DATASET_ID
RESULTS_FILENAME = "secondary_margin_decomposition.json"

CELL = (0.01, 16)            # widest observed margin range in scripts/26's artefact
REFERENCE_SEED = 42          # the held-fixed value for whichever input an arm is not varying
DEFAULT_SEEDS = [42, 123, 456, 789, 1337, 2026, 31415, 27182, 16180, 11235]
ARMS = ["vary_init", "vary_poison"]


def row_key(arm, seed) -> str:
    return f"{arm}|{seed}"


def _config_key(seeds, mlp_epochs, n_sigma) -> dict:
    return dict(cell=list(CELL), seeds=list(seeds), arms=list(ARMS), z_grid=list(Z_GRID),
                reference_seed=REFERENCE_SEED, mlp_epochs=mlp_epochs, n_sigma=n_sigma,
                spectral_components=config.SPECTRAL_COMPONENTS, dataset=DATASET_ID,
                constraints=manifest_fingerprint())


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
    tmp.replace(path)


def main() -> None:
    t0 = time.time()
    cfg = apply_overrides(get_config(False), sys.argv)
    if "--seeds" not in sys.argv:
        cfg["seeds"] = list(DEFAULT_SEEDS)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = config.RESULTS / RESULTS_FILENAME
    rate, cost = CELL

    print(f"dataset={DATASET_ID}  device={device}  cell={CELL}  seeds={cfg['seeds']}  arms={ARMS}")
    print(f"trigger held FIXED at ranking from seed {REFERENCE_SEED} for every row")

    S0 = load_secondary_setup(DATASET_ID)
    eligible = eligible_indices_for(S0.feature_names)
    S = SimpleNamespace(**{f: getattr(S0, f) for f in S0.__dataclass_fields__})

    ckpt_key = _config_key(cfg["seeds"], cfg["mlp_epochs"], cfg["n_sigma"])
    ckpt_path = out_path.with_suffix(".checkpoint.json")
    rows = load_checkpoint(ckpt_path, ckpt_key)
    save_cb = lambda: save_checkpoint(ckpt_path, ckpt_key, rows)

    # --- the one clean MLP / SHAP ranking / trigger reused by every row -------------------------
    x_tr_std = apply_standardiser(S.scaler, S.x_train_raw)
    x_te_std = apply_standardiser(S.scaler, S.x_test_raw)
    x_te_attack_raw = S.x_test_raw[S.y_test != TARGET]
    print(f"--- building fixed trigger from seed {REFERENCE_SEED} ({time.time() - t0:.0f}s) ---")
    clean_mlp = train_mlp(x_tr_std, S.y_train, REFERENCE_SEED, epochs=cfg["mlp_epochs"],
                          device=device, log_every=5)
    ranking = shap_rank_features(clean_mlp, S.x_train_raw, S.scaler, target=TARGET, kind="mlp",
                                 device=device)
    project_fn = partial(project_secondary_to_feasible, constraints=S.constraints, bounds=S.bounds)
    trigger = build_secondary_trigger(ranking, cost, S.x_train_raw, S.feature_names, S.bounds,
                                      project_fn, eligible, n_sigma=cfg["n_sigma"])
    del clean_mlp
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    x_te_attack_trig_raw = apply_secondary_trigger(x_te_attack_raw, trigger)
    x_te_attack_trig_std = apply_standardiser(S.scaler, x_te_attack_trig_raw)
    print(f"fixed trigger features: {list(trigger['feature_names'])}")

    for arm in ARMS:
        for seed in cfg["seeds"]:
            key = row_key(arm, seed)
            if key in rows:
                continue
            poison_seed = REFERENCE_SEED if arm == "vary_init" else seed
            init_seed = seed if arm == "vary_init" else REFERENCE_SEED

            x_p_std, y_p, poison_idx = poison_secondary_trainset(
                S.x_train_raw, S.y_train, trigger, rate, S.scaler, target=TARGET, seed=poison_seed)
            mlp = train_mlp(x_p_std, y_p, init_seed, epochs=cfg["mlp_epochs"], device=device)

            asr = attack_success_rate(mlp, x_te_attack_trig_std, TARGET, device)
            cacc = clean_accuracy(mlp, x_te_std, S.y_test, device)

            benign_pos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(benign_pos, poison_idx)
            n_poison, n_benign = int(is_poison.sum()), len(is_poison)
            n_clean = n_benign - n_poison
            feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
            del x_p_std
            gc.collect()

            scores = spectral_scores(feats, n_components=config.SPECTRAL_COMPONENTS)
            fixed_recall = poison_recall(
                flag_by_scores(scores, expected_frac=float(is_poison.mean())), is_poison)
            auc = float(roc_auc_score(is_poison, scores))
            z_results = evaluate_z_grid(scores, is_poison, n_clean, Z_GRID)
            geometry = score_geometry(scores, is_poison, Z_GRID)

            del mlp, feats, scores
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            rows[key] = dict(
                dataset=DATASET_ID, arm=arm, seed=seed, rate=rate, cost=cost,
                poison_seed=poison_seed, init_seed=init_seed,
                asr=asr, clean_acc=cacc, n_poison=n_poison, n_clean=n_clean,
                fixed_budget_recall=fixed_recall, spectral_auc=auc,
                mad_by_z=z_results, geometry=geometry,
            )
            save_cb()
            m3 = geometry["per_z"]["3.0"]
            print(f"  {arm:<12} seed={seed:<6} poison={poison_seed:<6} init={init_seed:<6} "
                  f"AUC={auc:.4f} recall@3={z_results['3.0']['recall']:.4f} "
                  f"margin={m3['poison_median_margin_in_mads']:.2f}")

    # internal consistency: seed 42 is the identical configuration in both arms
    a = rows.get(row_key("vary_init", REFERENCE_SEED))
    b = rows.get(row_key("vary_poison", REFERENCE_SEED))
    agree = None
    if a and b:
        agree = bool(np.isclose(a["mad_by_z"]["3.0"]["recall"], b["mad_by_z"]["3.0"]["recall"],
                                atol=1e-9))
        if not agree:
            print("WARNING: the two arms disagree at the shared reference seed -- "
                  "the arms are not isolating what they claim to")

    ordered = [rows[k] for k in sorted(rows)]
    blob = dict(
        dataset=DATASET_ID, cell=list(CELL), reference_seed=REFERENCE_SEED, z_grid=Z_GRID,
        fixed_trigger_features=list(trigger["feature_names"]),
        config=dict(seeds=cfg["seeds"], arms=ARMS, mlp_epochs=cfg["mlp_epochs"],
                    n_sigma=cfg["n_sigma"], spectral_components=config.SPECTRAL_COMPONENTS),
        source_note="follow-up to notes/20260803-experiment-threshold-margin-result.md",
        rows=ordered,
        summary=dict(n_rows=len(ordered), arm_agreement_ok=agree),
    )
    out_path.write_text(json.dumps(blob, indent=1))
    print(f"\nwrote {out_path}  ({len(ordered)} rows, {time.time() - t0:.0f}s)  "
          f"arm_agreement_ok={agree}")


if __name__ == "__main__":
    main()
