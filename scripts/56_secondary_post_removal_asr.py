"""Does REMOVING the MAD-flagged rows actually kill the backdoor on UNSW-NB15?

`scripts/16_secondary_adaptive_threshold.py` shows that the budget-free MAD z=3.0 rule
(`config.MAD_Z_PRIMARY`) recovers Spectral Signatures' poison RECALL inside the secondary dataset's
three evasion-window cells, at a false-positive rate of roughly 6% of the benign class. Recall is a
detection statistic, not a defence outcome: it says the flagged set CONTAINS the poison, not that
deleting the flagged set leaves a model without a backdoor. Nothing committed in results/ closes that
gap on the secondary dataset -- `scripts/21_ac_repair_grid.py`'s exclude-and-retrain idiom exists only
inside the Activation-Clustering repair track, on the primary dataset.

This script closes it. For every (evasion-window cell, seed) it rebuilds the poisoned training block
exactly as scripts/16 does, scores it with Spectral Signatures, applies the SAME locked MAD rule,
drops the flagged rows, RETRAINS the MLP victim on what is left, and reports post-removal ASR and
clean accuracy beside the pre-removal pair.

Determinism gate. The pre-removal half of every row is a quantity scripts/16 already committed to
`results/secondary_adaptive_threshold.json`. Each row is therefore self-checked against that file
(ASR, clean accuracy, fixed-budget recall, MAD recall / flag count / FPR) and the run ABORTS on the
first mismatch rather than emitting post-removal numbers built on a training block that is not the
one the paper reports. Reproducing the committed pre-removal numbers is what licenses the new ones.

Trigger source. The trigger is NOT recomputed. scripts/16 derives it from a live GradientSHAP ranking
of a freshly trained clean victim, which needs `shap`; the per-(seed, cost) trigger it produced is
already committed in `results/secondary_poison_sweep.json` -> `per_cell[].trigger` (feature names plus
raw requested values, stable across poison rates -- the SHAP ranking depends on the seed and the
watermark on the training-set mean/std, neither of which the rate touches). This script reads that
cached trigger, re-derives the feature indices from the loaded feature contract, and re-runs the same
`project_secondary_to_feasible` projector at stamping time. As a guard, the construction-time probe
row is re-projected and checked against the cached `post_projection_values` before any training runs.

Target cells are derived MECHANICALLY, never hand-picked: the cells scripts/16 itself classified as
evasion windows (`per_cell[].is_evasion_window`), cross-checked against
`results/extension_analysis.json`'s independent `window_and_recovery_replicated` classification. The
two must agree or the run aborts.

Run:  python scripts/56_secondary_post_removal_asr.py --smoke
      python scripts/56_secondary_post_removal_asr.py
      python scripts/56_secondary_post_removal_asr.py --seeds 42 --cells 0.01:16
"""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

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
from src.stats import bootstrap_ci
from src.trigger import apply_secondary_trigger

TARGET = config.ATTACK_TARGET
DATASET_ID = config.SECONDARY_DATASET_ID

RESULTS_FILENAME = "secondary_post_removal_asr.json"
RESULTS_FILENAME_SMOKE = "secondary_post_removal_asr_smoke.json"

REFERENCE_FILE = "secondary_adaptive_threshold.json"
TRIGGER_FILE = "secondary_poison_sweep.json"
CLASSIFICATION_FILE = "extension_analysis.json"

# Determinism tolerances. The pipeline is seed-deterministic on a fixed stack, so the honest
# expectation is an exact match; these are float-comparison slack, not a "close enough" band.
ATOL_RATE = 1e-9        # ASR / accuracy / recall / FPR -- all in [0, 1]
TRIGGER_ATOL = 1e-9     # cached vs re-projected probe-row trigger values (raw units, rtol used too)


def row_key(seed, rate, cost) -> str:
    return f"{seed}|{rate}|{cost}"


def _config_key(z: float, target_cells, mlp_epochs: int, n_sigma: float) -> dict:
    return dict(z=z, target_cells=[list(c) for c in target_cells], mlp_epochs=mlp_epochs,
                n_sigma=n_sigma, spectral_components=config.SPECTRAL_COMPONENTS,
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
    tmp.replace(path)   # atomic: a crash mid-write never corrupts the checkpoint


def parse_cell_list(argv):
    if "--cells" not in argv:
        return None
    spec = argv[argv.index("--cells") + 1]
    return [(float(r), int(c)) for r, c in (tok.split(":") for tok in spec.split(","))]


def parse_output(argv, default_name: str) -> Path:
    if "--output" in argv:
        return config.RESULTS / argv[argv.index("--output") + 1]
    return config.RESULTS / default_name


def _load_json(name: str) -> dict:
    path = config.RESULTS / name
    if not path.exists():
        raise SystemExit(f"{path} not found -- this script reuses committed artifacts and cannot "
                         "regenerate them; run the upstream script first")
    return json.loads(path.read_text())


def derive_target_cells(ref_blob: dict, ext_blob: dict):
    """The evasion-window cells, derived from committed classifications rather than hand-picked.

    Primary source: scripts/16's own `per_cell[].is_evasion_window` (mean ASR >= ATTACK_EFFECTIVE_ASR
    and mean fixed-budget recall <= FIXED_BUDGET_MISS_RECALL, applied by
    `src.analysis_extension.classify_evasion_window`). Cross-checked against Task 9's independent
    per-cell `replication_classification == "window_and_recovery_replicated"` in extension_analysis.json.
    Disagreement means one of the two artifacts has drifted, and a hand-resolved subset is exactly what
    this derivation exists to prevent -- so it aborts.
    """
    from_ref = sorted((c["rate"], c["cost"]) for c in ref_blob["per_cell"] if c["is_evasion_window"])
    from_ext = sorted((c["rate"], c["cost"]) for c in ext_blob["datasets"][DATASET_ID]["cells"]
                      if c.get("replication_classification") == "window_and_recovery_replicated")
    if from_ref != from_ext:
        raise SystemExit(
            f"evasion-window cell sets disagree: {REFERENCE_FILE} says {from_ref}, "
            f"{CLASSIFICATION_FILE} says {from_ext} -- refusing to guess which is current")
    if not from_ref:
        raise SystemExit(f"{REFERENCE_FILE} classifies no cell as an evasion window -- nothing to run")
    return from_ref


def build_trigger_cache(sweep_blob: dict) -> dict:
    """(seed, cost) -> cached trigger dict from the committed poison sweep.

    Asserts the trigger is rate-invariant, which is what makes a per-(seed, cost) cache legitimate:
    `build_secondary_trigger` sees only the SHAP ranking (seed-dependent) and the training-set
    mean/std (fixed), never the poison rate.
    """
    cache = {}
    for cell in sweep_blob["per_cell"]:
        key = (cell["seed"], cell["cost"])
        trig = cell["trigger"]
        if key in cache and cache[key]["feature_names"] != trig["feature_names"]:
            raise SystemExit(f"cached trigger for seed={key[0]} cost={key[1]} differs across poison "
                             "rates -- the per-(seed, cost) cache assumption is broken")
        cache[key] = trig
    return cache


def realise_trigger(cached: dict, S, project_fn) -> dict:
    """Rebuild a `build_secondary_trigger`-shaped dict from the cached feature names/raw values."""
    name_to_idx = {str(n): i for i, n in enumerate(S.feature_names)}
    missing = [n for n in cached["feature_names"] if n not in name_to_idx]
    if missing:
        raise SystemExit(f"cached trigger names absent from the loaded feature contract: {missing}")
    return dict(indices=[name_to_idx[str(n)] for n in cached["feature_names"]],
                values=[float(v) for v in cached["requested_values"]],
                feature_names=[str(n) for n in cached["feature_names"]],
                n_sigma=None, project_fn=project_fn)


def check_trigger_projection(trigger: dict, cached: dict, x_train_raw, project_fn) -> None:
    """Re-run `build_secondary_trigger`'s own construction-time probe and compare to the cached
    `post_projection_values`. This is what proves the rebuilt trigger stamps the same thing the
    committed sweep stamped, without needing shap to re-derive the ranking."""
    mu = np.asarray(x_train_raw, dtype=np.float64).mean(axis=0)
    probe = mu.copy()[None, :]
    for i, v in zip(trigger["indices"], trigger["values"]):
        probe[0, i] = v
    projected = np.asarray(project_fn(probe), dtype=np.float64)
    got = np.array([projected[0, i] for i in trigger["indices"]], dtype=np.float64)
    want = np.asarray(cached["post_projection_values"], dtype=np.float64)
    if not np.allclose(got, want, rtol=1e-9, atol=TRIGGER_ATOL):
        raise SystemExit(
            "rebuilt trigger does not reproduce the committed post-projection values\n"
            f"  features: {trigger['feature_names']}\n  got:  {got.tolist()}\n  want: {want.tolist()}")


def determinism_deltas(row: dict, ref: dict, z: float) -> dict:
    """Absolute deltas between this rerun's pre-removal half and the committed scripts/16 row."""
    ref_mad = ref["mad"][str(z)]
    pairs = {
        "pre_removal_asr": (row["pre_removal_asr"], ref["asr"]),
        "pre_removal_clean_acc": (row["pre_removal_clean_acc"], ref["clean_acc"]),
        "fixed_budget_recall": (row["fixed_budget_recall"], ref["fixed_budget_recall"]),
        "mad_recall": (row["mad_recall"], ref_mad["recall"]),
        "mad_fpr": (row["mad_fpr"], ref_mad["fpr"]),
    }
    out = {k: (abs(float(a) - float(b)) if a is not None and b is not None else None)
           for k, (a, b) in pairs.items()}
    out["mad_n_flagged"] = abs(int(row["n_removed"]) - int(ref_mad["n_flagged"]))
    out["n_poison"] = abs(int(row["n_poison"]) - int(ref["n_poison"]))
    out["n_benign"] = abs(int(row["n_benign"]) - int(ref["n_benign"]))
    return out


def deltas_ok(deltas: dict) -> bool:
    for name, d in deltas.items():
        if d is None:
            continue
        limit = 0 if name.startswith("n_") or name == "mad_n_flagged" else ATOL_RATE
        if d > limit:
            return False
    return True


def summarise(rows, metric: str) -> dict:
    values = [r[metric] for r in sorted(rows, key=lambda r: r["seed"])]
    if any(v is None for v in values):
        return dict(n=len(values), mean=None, ci=[None, None], per_seed=values)
    mean, ci = bootstrap_ci([float(v) for v in values])
    return dict(n=len(values), mean=mean, ci=ci, per_seed=[float(v) for v in values])


def main() -> int:
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    allow_mismatch = "--allow-determinism-mismatch" in sys.argv
    z = config.MAD_Z_PRIMARY                     # locked; this script never searches for a threshold
    out_path = parse_output(sys.argv, RESULTS_FILENAME_SMOKE if smoke else RESULTS_FILENAME)

    cfg = apply_overrides(get_config(smoke), sys.argv)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ref_blob = _load_json(REFERENCE_FILE)
    ext_blob = _load_json(CLASSIFICATION_FILE)
    sweep_blob = _load_json(TRIGGER_FILE)

    explicit_cells = parse_cell_list(sys.argv)
    target_cells = explicit_cells if explicit_cells is not None else derive_target_cells(ref_blob, ext_blob)
    if explicit_cells is None and not smoke:
        cfg["seeds"] = list(ref_blob["config"]["seeds"])   # the seed set scripts/16 actually spent
    if ref_blob["z"] != z:
        raise SystemExit(f"{REFERENCE_FILE} was written at z={ref_blob['z']}, not the locked "
                         f"config.MAD_Z_PRIMARY={z} -- the determinism gate would compare "
                         "incomparable thresholds")

    ref_index = {(r["seed"], r["rate"], r["cost"]): r for r in ref_blob["rows"]}
    trig_cache = build_trigger_cache(sweep_blob)

    print(f"dataset={DATASET_ID}  device={device}  smoke={smoke}  z={z}  seeds={cfg['seeds']}  "
          f"target_cells={target_cells}  out={out_path}", flush=True)

    S = load_secondary_setup(DATASET_ID)
    eligible = set(eligible_indices_for(S.feature_names))
    project_fn = partial(project_secondary_to_feasible, constraints=S.constraints, bounds=S.bounds)
    x_te_std = apply_standardiser(S.scaler, S.x_test_raw)
    x_te_attack_raw = S.x_test_raw[S.y_test != TARGET]
    print(f"loaded setup ({time.time() - t0:.0f}s): x_train={S.x_train_raw.shape} "
          f"x_test={S.x_test_raw.shape}", flush=True)

    ckpt_key = _config_key(z, target_cells, cfg["mlp_epochs"], cfg["n_sigma"])
    ckpt_path = out_path.with_suffix(".checkpoint.json")
    row_cells = load_checkpoint(ckpt_path, ckpt_key)
    save_cb = lambda: save_checkpoint(ckpt_path, ckpt_key, row_cells)

    for seed in cfg["seeds"]:
        for rate, cost in target_cells:
            key = row_key(seed, rate, cost)
            if key in row_cells:
                continue
            if (seed, cost) not in trig_cache:
                raise SystemExit(f"no cached trigger for seed={seed} cost={cost} in {TRIGGER_FILE}")
            cached = trig_cache[(seed, cost)]
            realiz = realise_trigger(cached, S, project_fn)
            if not set(realiz["indices"]) <= eligible:
                raise SystemExit(f"cached trigger for seed={seed} cost={cost} touches a "
                                 "trigger-ineligible feature -- refusing to stamp it")
            check_trigger_projection(realiz, cached, S.x_train_raw, project_fn)

            print(f"--- seed={seed} rate={rate} cost={cost} ({time.time() - t0:.0f}s) ---", flush=True)

            # ---- pre-removal victim (must reproduce scripts/16 exactly) ----
            x_p_std, y_p, poison_idx = poison_secondary_trainset(
                S.x_train_raw, S.y_train, realiz, rate, S.scaler, target=TARGET, seed=seed)
            mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

            x_te_attack_trig_std = apply_standardiser(
                S.scaler, apply_secondary_trigger(x_te_attack_raw, realiz))
            pre_asr = attack_success_rate(mlp, x_te_attack_trig_std, TARGET, device)
            pre_cacc = clean_accuracy(mlp, x_te_std, S.y_test, device)

            benign_pos = np.where(y_p == TARGET)[0]
            is_poison = np.isin(benign_pos, poison_idx)
            n_poison, n_benign = int(is_poison.sum()), len(is_poison)
            n_clean = n_benign - n_poison
            feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)

            # Same OOM discipline as scripts/16 (notes/20260731-bug-task5-oom-root-cause.md): free the
            # full poisoned matrix before spectral_scores' float64 SVD transient, and rebuild it after
            # (poison_secondary_trainset is seeded and deterministic, ~1s, and the rebuild is asserted
            # identical below) rather than letting 0.4GB sit under a ~3.5GB peak.
            del x_p_std, mlp
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            scores = spectral_scores(feats, n_components=config.SPECTRAL_COMPONENTS)
            del feats
            gc.collect()

            fixed_flag = flag_by_scores(scores, expected_frac=float(is_poison.mean()))
            fixed_recall = poison_recall(fixed_flag, is_poison) if n_poison else None
            mad_flag = flag_by_mad_threshold(scores, z_thresh=z)
            del scores, fixed_flag
            gc.collect()

            n_true_pos = int(np.sum(mad_flag & is_poison))
            n_removed = int(mad_flag.sum())
            n_false_pos = n_removed - n_true_pos
            mad_recall = poison_recall(mad_flag, is_poison) if n_poison else None
            mad_fpr = (n_false_pos / n_clean) if n_clean > 0 else None

            # ---- exclude-and-retrain (scripts/21_ac_repair_grid.py's idiom) ----
            x_p_std, y_p2, poison_idx2 = poison_secondary_trainset(
                S.x_train_raw, S.y_train, realiz, rate, S.scaler, target=TARGET, seed=seed)
            if not np.array_equal(poison_idx, poison_idx2):
                raise SystemExit("poisoned-block rebuild drew a different poison index -- "
                                 "poison_secondary_trainset is not behaving deterministically")
            excluded_rows = benign_pos[mad_flag]
            keep = np.ones(x_p_std.shape[0], dtype=bool)
            keep[excluded_rows] = False
            retrained = train_mlp(x_p_std[keep], y_p2[keep], seed, epochs=cfg["mlp_epochs"],
                                  device=device)
            post_asr = attack_success_rate(retrained, x_te_attack_trig_std, TARGET, device)
            post_cacc = clean_accuracy(retrained, x_te_std, S.y_test, device)

            del x_p_std, retrained, mad_flag, excluded_rows, keep
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            row = dict(
                dataset=DATASET_ID, seed=seed, rate=rate, cost=cost, z=z,
                pre_removal_asr=pre_asr, post_removal_asr=post_asr,
                asr_drop=pre_asr - post_asr,
                pre_removal_clean_acc=pre_cacc, post_removal_clean_acc=post_cacc,
                clean_acc_drop=pre_cacc - post_cacc,
                n_poison=n_poison, n_benign=n_benign, n_clean=n_clean,
                n_removed=n_removed, n_poison_removed=n_true_pos, n_clean_removed=n_false_pos,
                n_poison_surviving=n_poison - n_true_pos,
                frac_benign_removed=n_removed / n_benign if n_benign else None,
                frac_train_removed=n_removed / len(S.y_train),
                mad_recall=mad_recall, mad_fpr=mad_fpr,
                fixed_budget_recall=fixed_recall,
                trigger_source=f"results/{TRIGGER_FILE}#per_cell[seed={seed},cost={cost}].trigger",
                trigger_features=realiz["feature_names"],
            )
            ref = ref_index.get((seed, rate, cost))
            if ref is None:
                raise SystemExit(f"no committed {REFERENCE_FILE} row for seed={seed} rate={rate} "
                                 f"cost={cost} -- the determinism gate cannot be satisfied")
            row["determinism_deltas"] = determinism_deltas(row, ref, z)
            row["determinism_ok"] = deltas_ok(row["determinism_deltas"])

            print(f"  pre_asr={pre_asr:.6f} post_asr={post_asr:.6f} "
                  f"pre_cacc={pre_cacc:.6f} post_cacc={post_cacc:.6f} "
                  f"removed={n_removed} (poison {n_true_pos}/{n_poison}, clean {n_false_pos}) "
                  f"determinism_ok={row['determinism_ok']}", flush=True)

            if not row["determinism_ok"]:
                print("DETERMINISM MISMATCH vs " + REFERENCE_FILE, flush=True)
                print(json.dumps(row["determinism_deltas"], indent=1), flush=True)
                if not allow_mismatch:
                    raise SystemExit(
                        "STOP: the rerun did not reproduce the committed pre-removal numbers for "
                        f"seed={seed} rate={rate} cost={cost}. Post-removal numbers from a training "
                        "block that is not the paper's are worthless; fix the environment (this "
                        "pipeline was committed on torch 2.4.1 / numpy 1.26.4 / pandas 2.2.2 / "
                        "scikit-learn 1.5.2) and rerun. Pass --allow-determinism-mismatch only to "
                        "debug, never to produce the committed artifact.")

            row_cells[key] = row
            save_cb()

    rows = [row_cells[row_key(seed, rate, cost)]
            for seed in cfg["seeds"] for rate, cost in target_cells]

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["rate"], r["cost"])].append(r)

    metrics = ["pre_removal_asr", "post_removal_asr", "asr_drop",
               "pre_removal_clean_acc", "post_removal_clean_acc", "clean_acc_drop",
               "mad_recall", "mad_fpr", "fixed_budget_recall",
               "n_removed", "n_poison_removed", "n_clean_removed", "n_poison_surviving",
               "frac_benign_removed", "frac_train_removed"]
    per_cell = []
    for cell in sorted(by_cell):
        cell_rows = by_cell[cell]
        per_cell.append(dict(rate=cell[0], cost=cell[1], n_seeds=len(cell_rows),
                             **{m: summarise(cell_rows, m) for m in metrics}))

    n_failed = sum(1 for r in rows if r["determinism_ok"] is False)
    out = dict(
        dataset=DATASET_ID, z=z, z_source="config.MAD_Z_PRIMARY (locked on CTU-13, not re-searched)",
        question="Does dropping the MAD-flagged rows and retraining remove the backdoor?",
        config=dict(smoke=smoke, seeds=cfg["seeds"], target_cells=[list(c) for c in target_cells],
                    mlp_epochs=cfg["mlp_epochs"], n_sigma=cfg["n_sigma"],
                    spectral_components=config.SPECTRAL_COMPONENTS, device=device,
                    target_cell_derivation=(
                        "explicit --cells" if explicit_cells is not None else
                        f"cells with is_evasion_window in results/{REFERENCE_FILE}, cross-checked "
                        f"against window_and_recovery_replicated in results/{CLASSIFICATION_FILE}"),
                    removal_rule=f"drop every target-class row with MAD z > {z} on the top-"
                                 f"{config.SPECTRAL_COMPONENTS} Spectral-Signatures score",
                    trigger_source=(
                        f"cached, results/{TRIGGER_FILE} -> per_cell[].trigger "
                        "(feature_names + requested_values, per (seed, cost)); re-projected at stamp "
                        "time with src.constraints_secondary.project_secondary_to_feasible. shap is "
                        "NOT invoked -- the committed SHAP ranking is reused via its trigger."),
                    determinism_reference=f"results/{REFERENCE_FILE}",
                    environment=dict(torch=torch.__version__, numpy=np.__version__,
                                     python=sys.version.split()[0])),
        rows=rows,
        per_cell=per_cell,
        summary=dict(n_rows=len(rows), n_determinism_checked=len(rows),
                     n_determinism_failed=n_failed, determinism_all_ok=bool(n_failed == 0)),
    )
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({len(rows)} rows)", flush=True)
    for c in per_cell:
        print(f"  rate={c['rate']} cost={c['cost']}: "
              f"ASR {c['pre_removal_asr']['mean']:.4f} -> {c['post_removal_asr']['mean']:.4f}  "
              f"clean-acc {c['pre_removal_clean_acc']['mean']:.4f} -> "
              f"{c['post_removal_clean_acc']['mean']:.4f}  "
              f"removed {c['n_removed']['mean']:.0f} "
              f"(poison {c['n_poison_removed']['mean']:.0f}, clean {c['n_clean_removed']['mean']:.0f})")
    print(f"elapsed={time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
