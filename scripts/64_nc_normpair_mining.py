"""N0 of the Rung-2 repair pilots: mine the scalar residue of the Neural Cleanse inversion for
any model-level statistic that separates poisoned victims from clean replicas.

NC-6 tested one statistic, the two-class mask ratio, and got detection gain +0.0000. That is a
result about the ratio, not about everything the inversion leaves behind. This script tests the
rest of the residue already sitting in results/nc_repair_calibration.json -- the raw mask-norm
pair, the convergence counters, and the numeric pattern_audit fields -- at zero compute cost,
before the expensive mask-geometry pilot (scripts/65) regenerates anything.

The trap this script exists to avoid: with 6 poisoned rows against 102 clean rows per candidate,
the null distribution of AUC is wide (standard error about 0.12), so the maximum over ~25
correlated mined fields lands around 0.75-0.85 by chance alone. Reading "well above 0.5" off the
best field is therefore the default outcome, not evidence. The pre-registered promotion bar
(notes/20260901-decision-rung2-repair-pilots-preregistration.md) is a permutation-based
max-statistic: a field is promoted only if its observed AUC beats the 95th percentile of the
max-AUC-over-all-fields distribution under label permutation.

Analysis only -- no training, no inversion, no GPU. Runs in seconds, so no checkpoint (the
project's ~10-minute checkpointing rule does not apply).

Run:  python scripts/64_nc_normpair_mining.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

CALIBRATION_PATH = config.RESULTS / "nc_repair_calibration.json"
OUT_PATH = config.RESULTS / "nc_normpair_mining.json"

N_PERMUTATIONS = 10_000
PERMUTATION_PERCENTILE = 95.0
PERMUTATION_SEED = 42


def flatten_numeric(prefix: str, obj: dict) -> dict:
    """Flatten one level of nested dicts into {name: float}, keeping bools as 0/1.

    pattern_audit carries an after_projection sub-dict; the pre-registration lists its numeric
    fields as mineable, so it is flattened rather than skipped.
    """
    out: dict[str, float] = {}
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, bool):
            out[name] = float(value)
        elif isinstance(value, (int, float)):
            out[name] = float(value)
        elif isinstance(value, dict):
            out.update(flatten_numeric(f"{name}.", value))
    return out


def row_features(row: dict) -> dict:
    """The fixed mined-field list from the pre-registration, for one calibration row."""
    norms = row["mask_norms"]
    flagged = int(row["flagged_class"])
    flagged_norm = float(norms[flagged])
    other_norm = float(norms[1 - flagged])

    feats = {
        "flagged_norm": flagged_norm,
        "other_norm": other_norm,
        "min_norm": min(flagged_norm, other_norm),
        "max_norm": max(flagged_norm, other_norm),
        "norm_difference": flagged_norm - other_norm,
        "norm_sum": flagged_norm + other_norm,
        # Does the flagged class carry the larger mask? NC assumes the target class inverts to a
        # SMALLER trigger, so the direction itself is a candidate signal.
        "asymmetry_direction": float(flagged_norm >= other_norm),
        "n_convergence_failures": float(row.get("n_convergence_failures", 0)),
        "n_no_converged_start": float(row.get("n_no_converged_start", 0)),
    }
    feats.update(flatten_numeric("pattern_audit.", row.get("pattern_audit", {}) or {}))
    return feats


def auc_midrank(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the rank-sum identity, with midranks for ties.

    Written out rather than imported so the tie convention is visible: the pre-registration
    locks midranks, and a saturating statistic makes ties the common case, not the edge case.
    """
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def two_sided_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Direction-free AUC: a field that ranks victims consistently LOW is as useful as one that
    ranks them high, so mining reports max(auc, 1 - auc) and records the direction separately."""
    auc = auc_midrank(scores, labels)
    if np.isnan(auc):
        return auc
    return max(auc, 1.0 - auc)


def mine_candidate(clean_rows: list, poisoned_rows: list, rng: np.random.Generator) -> dict:
    clean_feats = [row_features(r) for r in clean_rows]
    poisoned_feats = [row_features(r) for r in poisoned_rows]

    shared = sorted(set(clean_feats[0]).intersection(*[set(f) for f in clean_feats + poisoned_feats]))
    labels = np.array([0] * len(clean_feats) + [1] * len(poisoned_feats), dtype=int)

    mined: dict[str, dict] = {}
    dropped: list[str] = []
    score_matrix = []
    field_names = []

    for field in shared:
        values = np.array([f[field] for f in clean_feats] + [f[field] for f in poisoned_feats], dtype=float)
        if not np.all(np.isfinite(values)):
            dropped.append(f"{field} (non-finite)")
            continue
        if np.ptp(values) == 0.0:
            dropped.append(f"{field} (zero variance)")
            continue
        directed = auc_midrank(values, labels)
        mined[field] = {
            "auc_two_sided": two_sided_auc(values, labels),
            "auc_directed": directed,
            "direction": "poisoned_higher" if directed >= 0.5 else "poisoned_lower",
            "clean_mean": float(values[labels == 0].mean()),
            "poisoned_mean": float(values[labels == 1].mean()),
        }
        score_matrix.append(values)
        field_names.append(field)

    # Permutation null for the MAXIMUM two-sided AUC across all mined fields simultaneously.
    # Permuting the shared label vector (not each field independently) preserves the correlation
    # between fields, which is what makes the max-statistic the right guard here.
    max_null = np.empty(N_PERMUTATIONS, dtype=float)
    stacked = np.asarray(score_matrix)
    for p in range(N_PERMUTATIONS):
        permuted = rng.permutation(labels)
        max_null[p] = max(two_sided_auc(stacked[k], permuted) for k in range(len(field_names)))

    bar = float(np.percentile(max_null, PERMUTATION_PERCENTILE))
    for field, stats in mined.items():
        stats["exceeds_permutation_bar"] = bool(stats["auc_two_sided"] > bar)
        # Per-field permutation p against the same max-statistic null (family-wise by construction).
        stats["p_familywise"] = float((max_null >= stats["auc_two_sided"]).mean())

    promoted = sorted([f for f, s in mined.items() if s["exceeds_permutation_bar"]])
    ranked = sorted(mined.items(), key=lambda kv: kv[1]["auc_two_sided"], reverse=True)

    return {
        "n_clean": len(clean_feats),
        "n_poisoned": len(poisoned_feats),
        "n_fields_mined": len(field_names),
        "dropped_fields": dropped,
        "permutation_bar_auc": bar,
        "permutation_null_median": float(np.median(max_null)),
        "permutation_null_max": float(max_null.max()),
        "fields": mined,
        "top_five": [{"field": f, **s} for f, s in ranked[:5]],
        "promoted_fields": promoted,
        "verdict": "promoted" if promoted else "no_field_survives_multiplicity_guard",
    }


def main() -> int:
    if not CALIBRATION_PATH.exists():
        print(f"missing {CALIBRATION_PATH}", file=sys.stderr)
        return 1

    data = json.loads(CALIBRATION_PATH.read_text())
    candidates = sorted({r["candidate"] for r in data["clean_rows"]})
    rng = np.random.default_rng(PERMUTATION_SEED)

    per_candidate = {}
    for candidate in candidates:
        clean_rows = [r for r in data["clean_rows"] if r["candidate"] == candidate]
        poisoned_rows = [r for r in data["poisoned_rows"] if r["candidate"] == candidate]
        if not clean_rows or not poisoned_rows:
            continue
        per_candidate[candidate] = mine_candidate(clean_rows, poisoned_rows, rng)
        res = per_candidate[candidate]
        best = res["top_five"][0] if res["top_five"] else None
        print(
            f"{candidate:16s} n={res['n_clean']}c/{res['n_poisoned']}p "
            f"fields={res['n_fields_mined']:3d} bar={res['permutation_bar_auc']:.4f} "
            f"best={best['field'] if best else '-'}@{best['auc_two_sided']:.4f} "
            f"-> {res['verdict']}"
        )

    any_promoted = sorted({f for r in per_candidate.values() for f in r["promoted_fields"]})
    payload = {
        "source": str(CALIBRATION_PATH.name),
        "preregistration": "notes/20260901-decision-rung2-repair-pilots-preregistration.md",
        "n_permutations": N_PERMUTATIONS,
        "permutation_percentile": PERMUTATION_PERCENTILE,
        "permutation_seed": PERMUTATION_SEED,
        "tie_convention": "midrank",
        "auc_convention": "two-sided (max of auc, 1-auc); direction reported separately",
        "per_candidate": per_candidate,
        "promoted_any_candidate": any_promoted,
        "verdict": "promoted" if any_promoted else "no_field_survives_multiplicity_guard",
        "note": (
            "Zero-compute mining of the NC inversion's scalar residue. The multiplicity guard is a "
            "max-statistic permutation bar over all mined fields jointly, because with 6 poisoned "
            "rows the largest of ~25 correlated null AUCs lands near 0.8 by chance."
        ),
    }

    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.rename(OUT_PATH)
    print(f"\nverdict: {payload['verdict']}")
    print(f"promoted: {any_promoted or '(none)'}")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
