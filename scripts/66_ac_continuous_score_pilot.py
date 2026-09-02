"""A1/A2 of the Rung-2 repair pilots: give Activation Clustering a CONTINUOUS score, then judge it
with the same decision layer that repaired Spectral.

THE PROBLEM THIS ADDRESSES. Canonical AC ends in a hard binary flag: cluster the activations, pick a
cluster, call everything in it poison. On this project's evasion cells the k=2 GMM clustering finds
the poison perfectly -- recall 1.0 on all 20 cells -- inside a cluster holding about 20% of the data
(results/ac_repair_summary.json). So the detector is not blind; its output is simply the wrong SHAPE
to trade recall against false alarms. There is no knob. That is also why AC-0's pre-registered
matched-FPR gate was never implemented: with a binary flag there is nothing to sweep, a deviation the
independent audit flagged and this script closes.

WHAT IS NEW. Score every sample by WHERE it sits inside the suspicious component, rather than
whether it is inside at all. Then feed that score to `flag_by_mad_threshold` -- literally the
budget-free rule that repaired Spectral -- so AC and Spectral are finally compared inside one code
path instead of across two scripts that each computed their own version of the comparison.

DIRECTION, AND A RECORDED DEVIATION. The pilot was designed expecting poison to sit DEEP inside the
minority component. The seed-42 probe refuted that: directed AUC came back 0.0303, meaning the poison
is inside the cluster but in its least-dense TAIL -- peripheral, not core. The scored quantity is
therefore negated to ATYPICALITY (high = suspicious, matching Spectral's convention so the upper-tail
MAD cut reads the right side), every AUC is reported both directed and two-sided, and the flip is
recorded in the artifact under `deviation_from_preregistration`. The direction was chosen after
seeing probe data, which is exactly why both tails ship in every row rather than only the flattering
one.

WHY DEPTH AND NOT THE POSTERIOR. The obvious continuous score is the GMM's posterior probability of
belonging to the minority component. It is the wrong choice and the pre-registration says so in
advance: a posterior saturates near 1.0 for everything comfortably inside the component, so the
poison ties with the ~20% of clean rows that share the cluster. Under midrank tie handling that pins
AUC near 0.5 * (1 + 0.7974) = 0.899 -- the pre-registered 0.9 bar would be decided by floating-point
noise. Component log-likelihood keeps ordering inside the cluster, which is exactly the information
the binary flag throws away.

WHAT CANNOT BE CLAIMED, EVEN IF RECALL IS PERFECT. The nearest-clean-centroid distance score already
scored AUC about 0.98 here and was killed by this project's own independence rule: Spearman rho >=
0.90 against Spectral's score means the thing is Spectral wearing a different name
(results/ac_centroid_repair.json, verdict `spectral_redundant`). That gate is wired into this script's
verdict, not left to a reviewer to remember.

THE MISSING CONTROL. Every previous AC run here measured false alarms only inside poisoned training
sets; `CONTROL_COMBO = (0.1, 16)` in scripts/21_ac_repair_grid.py is a high-rate DETECTABILITY
control, not a clean one. This pilot adds rate=0 cells -- a model with no backdoor at all -- because
the deployment question is what the detector does on the many clean models it will see, and that was
never measured.

Run:  python scripts/66_ac_continuous_score_pilot.py --probe --seeds 42   # timing probe FIRST
      python scripts/66_ac_continuous_score_pilot.py                      # full 5-seed (resumable)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup, apply_overrides
from src import config
from src.data import apply_standardiser
from src.detectors.activation_clustering import classify_centroid_independence, cluster_and_reduce
from src.detectors.spectral import flag_by_mad_threshold, spectral_scores
from src.models import mlp_penultimate_features, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET

# Identical to scripts/21_ac_repair_grid.py's recipe so the clustering under test IS canonical AC's.
AC_PCA_COMPONENTS = 10
EVASION_COMBOS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]
CONTROL_COMBO = (0.1, 16)
CLEAN_COMBO = (0.0, 0)        # NEW: no poison at all -- the deployment false-alarm case
ALGORITHM = "gmm"             # pinned: the k-means minority excludes the poison (purity 0.041)
K = 2                         # pinned: k>2 is refuted (recall collapses to ~0 by k=10)

MAD_Z = 3.0                                  # the pre-registered z used by scripts/63
FPR_BUDGETS = [0.01, 0.02, 0.0658, 0.10]     # 6.58% is Spectral's repaired MAD operating point
AUC_GATE = 0.90
RECALL_GATE = 0.90
RECALL_GATE_MAX_FPR = 0.10
INDEPENDENCE_MAX_RHO = 0.90

OUT_PATH = config.RESULTS / "ac_continuous_score_pilot.json"
CKPT_PATH = config.RESULTS / "ac_continuous_score_pilot.checkpoint.json"


def _finite_or_none(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def auc_midrank(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC with midranks for ties -- the tie convention the pre-registration locks. Spelled out
    rather than imported so the handling is visible at the point the number is produced."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(scores, kind="mergesort")
    srt = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def depth_score(reduced: np.ndarray, estimator, labels: np.ndarray, is_poison: np.ndarray) -> tuple:
    """Depth inside the MINORITY component: that component's own log-density per sample.

    The minority component is chosen by size, exactly as canonical AC picks its suspicious cluster --
    never by looking at `is_poison`, which would leak the answer into the detector.
    """
    sizes = np.bincount(labels, minlength=K)
    minority = int(np.argmin(sizes))
    # Per-component log-density (log pi_k + log N(x | mu_k, Sigma_k)), the fitted mixture's own
    # decomposition -- not a re-derived Mahalanobis distance that could drift from it.
    per_component = estimator._estimate_weighted_log_prob(np.asarray(reduced, dtype=np.float64))
    scores = np.asarray(per_component[:, minority], dtype=np.float64)
    return scores, minority, sizes


def recall_fpr_at_threshold(scores, is_poison, thresh) -> tuple[float, float]:
    flagged = scores > thresh
    n_pos = int(is_poison.sum())
    n_neg = int((~is_poison).sum())
    recall = float(flagged[is_poison].sum() / n_pos) if n_pos else float("nan")
    fpr = float(flagged[~is_poison].sum() / n_neg) if n_neg else float("nan")
    return recall, fpr


def recall_at_fpr_budget(scores, is_poison, budget) -> float:
    """Highest recall achievable while holding FPR at or under `budget` -- the matched-FPR comparison
    AC-0 pre-registered and no AC run could implement until the score became continuous."""
    clean = np.asarray(scores)[~is_poison]
    if len(clean) == 0:
        return float("nan")
    # Threshold at the budget-th upper quantile of the CLEAN scores: by construction the clean
    # flag rate is <= budget.
    thresh = float(np.quantile(clean, 1.0 - budget))
    return float((np.asarray(scores)[is_poison] > thresh).sum() / max(1, int(is_poison.sum())))


def run_cell(seed, rate, cost, S, cfg, device, ranking, stats) -> dict:
    """One (seed, rate, cost). rate=0 means a clean model: no trigger, no poisoning, no positives."""
    t0 = time.time()
    is_clean_cell = (rate == 0.0)

    if is_clean_cell:
        x_p_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
        y_p = np.asarray(S["y_tr"])
        poison_idx = np.array([], dtype=int)
    else:
        realiz, _ = build_trigger(ranking, cost, S["x_tr_raw"], S["constraints"], S["features"],
                                  S["bounds"], stats=stats)
        x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
            S["x_tr_raw"], S["y_tr"], realiz, rate, S["scaler"], target=TARGET, seed=seed,
            constraints=S["constraints"], feature_names=S["features"], bounds=S["bounds"])

    mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)
    benign_pos = np.where(y_p == TARGET)[0]
    is_poison = np.isin(benign_pos, poison_idx)
    feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)

    labels, reduced, sil, estimator = cluster_and_reduce(
        feats, n_components=AC_PCA_COMPONENTS, seed=seed, n_clusters=K, algorithm=ALGORITHM,
        return_estimator=True)
    scores, minority, sizes = depth_score(reduced, estimator, labels, is_poison)

    # A2: the same score restricted to minority-cluster members (non-members pushed to -inf so they
    # can never be flagged). Reported alongside A1, not as a separate run.
    in_minority = (labels == minority)
    scores_a2 = np.where(in_minority, scores, -np.inf)

    spec = spectral_scores(feats)
    rho = float(spearmanr(scores, spec).statistic)

    row = dict(
        seed=seed, rate=rate, cost=cost, is_clean_cell=is_clean_cell,
        n_benign=int(len(benign_pos)), n_poison=int(is_poison.sum()),
        minority_cluster=minority, cluster_sizes=[int(v) for v in sizes],
        silhouette=_finite_or_none(sil),
        spearman_vs_spectral=_finite_or_none(rho),
        independence=classify_centroid_independence(rho, max_rho=INDEPENDENCE_MAX_RHO),
        seconds=float(time.time() - t0),
    )

    # MAD z-cut, mirroring scripts/63. Its docstring warns the constant is calibrated to Spectral's
    # chi-squared-like shape; a near-constant score gives MAD == 0 and silently falls back to std, so
    # that case is recorded as a shape failure rather than reported as a detection result.
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))
    row["mad_is_zero"] = bool(mad == 0.0)
    # DEVIATION from the pre-registration, recorded rather than silently applied. The pilot was
    # designed expecting poison to sit DEEP inside the minority component; the seed-42 probe showed
    # the opposite (AUC 0.0303, i.e. 0.9697 inverted) -- the poison is inside the cluster but at its
    # PERIPHERY, the least-dense tail. The scored quantity is therefore negated to atypicality, so
    # that (as in Spectral) a HIGH score means suspicious and the upper-tail MAD cut is the right
    # tail to read. Every AUC is additionally reported two-sided, matching scripts/64 and 65, which
    # were direction-free from the start; this script was the inconsistent one. Direction was chosen
    # on probe data, so both tails are reported and nothing rests on the choice.
    scores = -scores
    scores_a2 = np.where(in_minority, scores, -np.inf)
    for name, sc in (("a1", scores), ("a2", scores_a2)):
        finite = sc[np.isfinite(sc)]
        filled = np.where(np.isfinite(sc), sc, finite.min() if len(finite) else 0.0)
        # Per-arm shape check. A2's fill puts ~80% of samples at one identical value, so its MAD is 0
        # by construction and flag_by_mad_threshold silently switches to its std fallback -- legal,
        # but the artifact must say which rule actually produced the flags. a1's check above does not
        # cover this because it is computed on the unfilled scores.
        arm_med = float(np.median(filled))
        row[f"{name}_mad_is_zero"] = bool(float(np.median(np.abs(filled - arm_med))) == 0.0)
        flagged = flag_by_mad_threshold(filled, z_thresh=MAD_Z)
        n_pos = int(is_poison.sum())
        n_neg = int((~is_poison).sum())
        row[f"{name}_mad_recall"] = float(flagged[is_poison].sum() / n_pos) if n_pos else None
        row[f"{name}_mad_fpr"] = float(flagged[~is_poison].sum() / n_neg) if n_neg else None
        if n_pos:
            directed = auc_midrank(sc, is_poison.astype(int))
            row[f"{name}_auc_directed"] = _finite_or_none(directed)
            # Two-sided: a score that ranks poison consistently LOW is as usable as one that ranks it
            # high (flip the sign). The gate reads this, with the realised direction recorded.
            row[f"{name}_auc"] = _finite_or_none(max(directed, 1.0 - directed))
            row[f"{name}_direction"] = "poison_higher" if directed >= 0.5 else "poison_lower"
        else:
            row[f"{name}_auc_directed"] = row[f"{name}_auc"] = row[f"{name}_direction"] = None
        for b in FPR_BUDGETS:
            row[f"{name}_recall_at_fpr_{b}"] = (
                _finite_or_none(recall_at_fpr_budget(sc, is_poison, b)) if n_pos else None)
    return row


def _config_key(seeds, combos, cfg) -> dict:
    key = dict(seeds=list(seeds), combos=[list(c) for c in combos], algorithm=ALGORITHM, k=K,
               pca=AC_PCA_COMPONENTS, mad_z=MAD_Z, fpr_budgets=FPR_BUDGETS,
               mlp_epochs=cfg["mlp_epochs"], subsample=cfg["subsample"])
    # JSON projection: tuples become lists on a round trip, so an in-memory key never equals the one
    # read back and every resume would silently restart (the 604e88a bug).
    return json.loads(json.dumps(key))


def load_checkpoint(path, key) -> dict:
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}
    print(f"resuming from checkpoint: {len(blob.get('rows', {}))} cells already done")
    return blob.get("rows", {})


def save_checkpoint(path, key, rows) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(path)


def bootstrap_ci(values, n_boot=10_000, seed=42) -> dict:
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    boots = [float(rng.choice(vals, size=len(vals), replace=True).mean()) for _ in range(n_boot)]
    return {"mean": float(vals.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n": int(len(vals))}


def summarise(rows) -> dict:
    """Aggregation is PINNED by the pre-registration: seed-mean per evasion combo, then the MINIMUM
    across the 4 combos. A mean over combos would let one strong combo carry three weak ones."""
    out = {}
    for arm in ("a1", "a2"):
        per_combo = {}
        for combo in EVASION_COMBOS:
            sel = [r for r in rows if (r["rate"], r["cost"]) == combo]
            per_combo[f"{combo[0]}_{combo[1]}"] = {
                "auc": bootstrap_ci([r.get(f"{arm}_auc") for r in sel]),
                "mad_recall": bootstrap_ci([r.get(f"{arm}_mad_recall") for r in sel]),
                "mad_fpr": bootstrap_ci([r.get(f"{arm}_mad_fpr") for r in sel]),
                **{f"recall_at_fpr_{b}": bootstrap_ci([r.get(f"{arm}_recall_at_fpr_{b}") for r in sel])
                   for b in FPR_BUDGETS},
            }
        aucs = [v["auc"]["mean"] for v in per_combo.values() if v["auc"]["mean"] is not None]
        rec10 = [v[f"recall_at_fpr_{RECALL_GATE_MAX_FPR}"]["mean"] for v in per_combo.values()
                 if v[f"recall_at_fpr_{RECALL_GATE_MAX_FPR}"]["mean"] is not None]
        clean_rows = [r for r in rows if r["is_clean_cell"]]
        clean_flag = bootstrap_ci([r.get(f"{arm}_mad_fpr") for r in clean_rows])
        rhos = [abs(r["spearman_vs_spectral"]) for r in rows
                if not r["is_clean_cell"] and r["spearman_vs_spectral"] is not None]
        max_rho = float(max(rhos)) if rhos else None

        clauses = {
            "min_combo_auc_ge_0.90": bool(aucs and min(aucs) >= AUC_GATE),
            "min_combo_recall_ge_0.90_at_fpr_0.10": bool(rec10 and min(rec10) >= RECALL_GATE),
            "independent_of_spectral": bool(max_rho is not None and max_rho < INDEPENDENCE_MAX_RHO),
            "clean_control_within_budget": bool(clean_flag["mean"] is not None
                                                and clean_flag["mean"] <= RECALL_GATE_MAX_FPR),
        }
        if not clauses["independent_of_spectral"]:
            verdict = "spectral_redundant"
        elif all(clauses.values()):
            verdict = "pass"
        else:
            verdict = "informative_negative"
        out[arm] = {
            "per_combo": per_combo,
            "min_combo_auc": float(min(aucs)) if aucs else None,
            "min_combo_recall_at_10pct_fpr": float(min(rec10)) if rec10 else None,
            "max_abs_spearman_vs_spectral": max_rho,
            "clean_control_flag_rate": clean_flag,
            "gate_clauses": clauses,
            "verdict": verdict,
        }
    return out


def main() -> int:
    argv = sys.argv[1:]
    probe = "--probe" in argv
    cfg = apply_overrides(get_config(smoke=False), argv)
    seeds = cfg["seeds"] if not probe else cfg["seeds"][:1]

    combos = EVASION_COMBOS + [CONTROL_COMBO, CLEAN_COMBO]
    if probe:
        combos = [EVASION_COMBOS[0], CLEAN_COMBO]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} seeds={seeds} combos={combos} algorithm={ALGORITHM} k={K}")

    t0 = time.time()
    S = load_setup(cfg)
    stats = raw_trigger_stats(S["x_tr_raw"], S["constraints"])
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    print(f"setup loaded ({time.time() - t0:.0f}s)")

    key = _config_key(seeds, combos, cfg)
    ckpt = load_checkpoint(CKPT_PATH, key)
    save_cb = lambda: save_checkpoint(CKPT_PATH, key, ckpt)

    rows, timings = [], []
    for seed in seeds:
        print(f"--- seed {seed} ({time.time() - t0:.0f}s) ---")
        clean_mlp = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
        ranking = shap_rank_features(clean_mlp, S["x_tr_raw"], S["scaler"], target=TARGET,
                                     kind="mlp", device=device)
        for rate, cost in combos:
            ck = f"{seed}|{rate}|{cost}"
            if ck in ckpt:
                rows.append(ckpt[ck])
                continue
            row = run_cell(seed, rate, cost, S, cfg, device, ranking, stats)
            ckpt[ck] = row
            rows.append(row)
            save_cb()
            timings.append(row["seconds"])
            print(f"  r={rate} c={cost} auc={row['a1_auc']} mad_recall={row['a1_mad_recall']} "
                  f"mad_fpr={row['a1_mad_fpr']} rho={row['spearman_vs_spectral']:.3f} "
                  f"[{row['independence']}] ({row['seconds']:.0f}s)")

    elapsed = time.time() - t0
    marginal = float(np.mean(timings[1:])) if len(timings) > 2 else (timings[-1] if timings else None)
    summary = summarise(rows)

    payload = {
        "preregistration": "notes/20260901-decision-rung2-repair-pilots-preregistration.md",
        "probe": probe, "seeds": seeds, "combos": [list(c) for c in combos],
        "algorithm": ALGORITHM, "k": K, "pca_components": AC_PCA_COMPONENTS,
        "mad_z": MAD_Z, "fpr_budgets": FPR_BUDGETS,
        "score": ("NEGATED minority-component weighted log-density (atypicality within the "
                  "suspicious cluster), NOT the posterior; see the deviation note in run_cell"),
        "deviation_from_preregistration": (
            "Pre-registered as depth (high = deep inside the minority component). The seed-42 probe "
            "showed poison at the component's PERIPHERY (directed AUC 0.0303), so the score is "
            "negated and all AUCs are reported two-sided as in scripts/64 and 65. Direction was "
            "chosen on probe data; both the directed and two-sided values are in every row."),
        "tie_convention": "midrank",
        "elapsed_seconds": elapsed, "marginal_seconds_per_cell": marginal,
        "n_cells": len(rows), "rows": rows, "summary": summary,
        # A pass on either pre-registered arm is a pass; spectral-redundancy on both is that verdict.
        "verdict": ("pass" if any(summary[a]["verdict"] == "pass" for a in ("a1", "a2"))
                    else ("spectral_redundant"
                          if all(summary[a]["verdict"] == "spectral_redundant" for a in ("a1", "a2"))
                          else "informative_negative")),
    }
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.rename(OUT_PATH)

    print(f"\n=== {elapsed:.0f}s total"
          + (f", marginal {marginal:.1f}s/cell ===" if marginal else " ==="))
    for arm in ("a1", "a2"):
        s = summary[arm]
        print(f"  [{arm}] min-combo AUC={s['min_combo_auc']} "
              f"min-combo recall@10%FPR={s['min_combo_recall_at_10pct_fpr']} "
              f"max|rho|={s['max_abs_spearman_vs_spectral']} -> {s['verdict']}")
        for c, ok in s["gate_clauses"].items():
            print(f"        {'PASS' if ok else 'FAIL'}  {c}")
    print(f"\nverdict: {payload['verdict']}")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
