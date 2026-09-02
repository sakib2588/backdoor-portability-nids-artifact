"""Tasks AC-4/AC-6/AC-9: does ANY clustering choice repair Activation Clustering on tabular NIDS?

WHY. Canonical AC (2-means, smaller-cluster rule) fails inside the evasion window for a diagnosed
reason: the poison never becomes the dominant secondary structure. At rate 0.005-0.01 the median cell
has 858 poison rows but 2-means splits off a 58-row cluster of natural outliers instead, and 12 of 20
evasion rows carry ZERO minority purity (results/ac_mechanism.json). At the (0.1, 16) control, where
poison is ~10% of the class, the same detector recalls 1.0 at purity 0.8149. So AC's failure is about
the poison's rank among competing structures, not about the detector being wired wrong.

FOUR DECISION RULES AGAINST THE SAME CLUSTERING, so a difference is attributable to the rule and not
to a re-run of k-means:
  relative-size      Chen et al.'s published rule, at every k and both algorithms
  silhouette-gate    relative-size suppressed when the split looks unreal (expected to MISFIRE here:
                     this repo has already measured silhouette ANTI-correlating with recall, -0.5549)
  gap-statistic      k=1-vs-k=2 confirmatory only, per Chen et al.'s own narrow use AND their own
                     reported negative result
  ExRe               exclude the flagged minority, retrain, reclassify. k=2 only, minority only --
                     see the pre-registration for why arbitrating a 111k-row majority is not well posed

Everything above, plus the k-grid, both algorithms, the 25-cell list, gate direction and the
interpretation table, was frozen in notes/20260803-decision-ac-track-preregistration.md BEFORE this
script ran.

THE GATE IS MATCHED-FPR, NOT 1% ABSOLUTE. Task 7B already found a 1% absolute gate inconsistent with
Spectral's own accepted 0.0596 operating point; holding AC to a budget 6x stricter than its
comparator answers no question.

INDEPENDENCE IS SCREENING, NOT PROOF. rho >= 0.90 against Spectral marks a candidate as re-deriving
Spectral's signal. The threshold was calibrated for Task 7B's centroid construction and has NOT been
re-derived for these, so the raw rho is always recorded beside the verdict.

Run:  python scripts/21_ac_repair_grid.py --smoke --seeds 42     # pipeline check
      python scripts/21_ac_repair_grid.py                        # full 25-cell run
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
from src.detectors import poison_recall
from src.detectors.activation_clustering import (
    classify_centroid_independence,
    cluster_and_reduce,
    exclusionary_reclassification_score,
    flag_by_exclusionary_reclassification,
    flag_by_relative_size,
    flag_by_silhouette_gate,
    gap_statistic_k1_vs_k2,
)
from src.detectors.spectral import spectral_scores
from src.models import mlp_penultimate_features, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET
AC_PCA_COMPONENTS = 10        # must match 05_run_detectors.py's AC recipe -- same detector
AC_SILHOUETTE_SAMPLE = 10000  # silhouette is an O(n^2) DIAGNOSTIC; cap it, never the decision metric

# The 4 evasion (rate, cost) combinations from results/analysis.json's evasion_window (mean ASR >= 0.8
# AND mean Spectral recall <= 0.5), plus the (0.1, 16) control. Matches ac_mechanism.json's own
# target_cells exactly, so canonical AC's committed per-cell numbers are the direct baseline.
EVASION_COMBOS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]
CONTROL_COMBO = (0.1, 16)
GRID_PATH = config.RESULTS / "ac_repair_grid.json"
SUMMARY_PATH = config.RESULTS / "ac_repair_summary.json"


def build_cells(seeds) -> list[tuple[int, float, int]]:
    """A FLAT list of (seed, rate, cost) triples. `cell` means a triple throughout this script.

    An earlier plan draft wrote the loops as `for (seed, cell, k, algorithm)`, treating seed as a
    dimension on top of an already seed-inclusive 25-cell count -- read literally a 5x double-count.
    Enumerating once here removes the ambiguity structurally rather than by convention.
    """
    return [(s, r, c) for s in seeds for (r, c) in (EVASION_COMBOS + [CONTROL_COMBO])]


def _finite_or_none(x):
    """Strict RFC-8259 JSON: NaN/inf must never reach the artifact."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def load_checkpoint(path, key):
    if not path.exists():
        return {}, {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh (old checkpoint ignored)")
        return {}, {}
    print(f"resuming from checkpoint: {len(blob.get('cheap', {}))} cheap-grid units, "
          f"{len(blob.get('exre', {}))} ExRe units already done")
    return blob.get("cheap", {}), blob.get("exre", {})


def save_checkpoint(path, key, cheap, exre):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, cheap=cheap, exre=exre)))
    tmp.replace(path)   # atomic: a crash mid-write never corrupts the checkpoint


def _config_key(cells, cfg) -> dict:
    """Everything that changes the MEANING of a checkpointed row."""
    return dict(
        cells=[list(c) for c in cells],
        k_grid=list(config.AC_K_GRID),
        algorithms=list(config.AC_CLUSTER_ALGORITHMS),
        mlp_epochs=cfg["mlp_epochs"],
        pca_components=AC_PCA_COMPONENTS,
        silhouette_threshold=config.AC_SILHOUETTE_GATE_THRESHOLD,
        gap_b=config.AC_GAP_STATISTIC_B,
        exre_k=config.AC_EXRE_CANDIDATE_K,
        exre_algorithms=list(config.AC_EXRE_ALGORITHMS),
        exre_t_default=config.AC_EXRE_T_DEFAULT,
        exre_clean_replicas=config.AC_EXRE_CLEAN_REPLICAS,
        smoke=cfg["smoke"],
    )


def _get_or_compute(key, store, sink, compute, save_cb):
    """Checkpoint-aware row getter. Returns (row, was_computed)."""
    if key in store:
        sink.append(store[key])
        return store[key], False
    row = compute()
    store[key] = row
    sink.append(row)
    save_cb()
    return row, True


def build_victim(seed, rate, cost, S, cfg, device, ranking, stats):
    """Train the poisoned victim for one cell and return everything downstream needs.

    Trained AT MOST ONCE per (seed, rate, cost) and reused across every k, algorithm and decision
    rule -- the same victim-cache discipline NC-4 introduced after the NC track paid for retraining
    it per candidate.
    """
    realiz, _ = build_trigger(ranking, cost, S["x_tr_raw"], S["constraints"], S["features"],
                              S["bounds"], stats=stats)
    x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
        S["x_tr_raw"], S["y_tr"], realiz, rate, S["scaler"], target=TARGET, seed=seed,
        constraints=S["constraints"], feature_names=S["features"], bounds=S["bounds"])
    mlp = train_mlp(x_p_std, y_p, seed, epochs=cfg["mlp_epochs"], device=device)

    benign_pos = np.where(y_p == TARGET)[0]
    is_poison = np.isin(benign_pos, poison_idx)
    feats = mlp_penultimate_features(mlp, x_p_std[benign_pos], device)
    return dict(x_p_std=x_p_std, y_p=y_p, benign_pos=benign_pos, is_poison=is_poison,
                feats=feats, mlp=mlp)


def _independence(flag_vector, spec_scores):
    """Per-row independence: correlate the boolean flag against that cell's Spectral score.

    A constant flag vector (all-False) is the NORM at the evasion operating point, not an edge case:
    12 of 20 evasion rows already flag nothing. spearmanr returns NaN there, which
    classify_centroid_independence maps to 'undetermined'. That counts as gate-NOT-passed -- a
    candidate cannot claim independence it never demonstrated -- but is reported as a label distinct
    from 'redundant'. It is never the sole reason a real detector dies, since a candidate that flags
    nothing already fails the recall gate on zero recall.
    """
    if flag_vector.sum() == 0 or flag_vector.all():
        return None, "undetermined"
    rho = spearmanr(flag_vector.astype(float), spec_scores).statistic
    return _finite_or_none(rho), classify_centroid_independence(
        rho, max_rho=config.AC_SPECTRAL_INDEPENDENCE_MAX_RHO)


def run_cheap_grid(cells, S, cfg, device, store, sink, save_cb, victims):
    """25 cells x k-grid x algorithms. Clusters ONCE per (cell, k, algorithm), then applies three
    decision rules to that same clustering -- no re-clustering per rule, no duplicated flag logic."""
    for (seed, rate, cost) in cells:
        unit_keys = [f"{seed}|{rate}|{cost}|{k}|{a}"
                     for k in config.AC_K_GRID for a in config.AC_CLUSTER_ALGORITHMS]
        if all(u in store for u in unit_keys):
            sink.extend(store[u] for u in unit_keys)   # fully checkpointed cell: train nothing
            continue

        V = victims(seed, rate, cost)
        # ONCE per cell, not per (k, algorithm). spectral_scores runs an SVD over the full
        # ~114k x 64 activation matrix and depends only on the cell, so computing it inside the
        # inner loops repeated it 8x per cell -- wasted wall-clock and, on a 14 GiB box, enough
        # peak memory to get the process OOM-killed mid-run with no traceback.
        spec = spectral_scores(V["feats"])
        is_poison = V["is_poison"]

        for k in config.AC_K_GRID:
            for algorithm in config.AC_CLUSTER_ALGORITHMS:
                key = f"{seed}|{rate}|{cost}|{k}|{algorithm}"
                if key in store:
                    sink.append(store[key])
                    continue
                labels, reduced, sil = cluster_and_reduce(
                    V["feats"], n_components=AC_PCA_COMPONENTS, seed=seed,
                    silhouette_sample=AC_SILHOUETTE_SAMPLE, n_clusters=k, algorithm=algorithm)

                rel = flag_by_relative_size(labels)
                gated = flag_by_silhouette_gate(labels, sil)
                gap = (gap_statistic_k1_vs_k2(reduced, seed=seed, b=config.AC_GAP_STATISTIC_B)
                       if k == 2 else None)

                row = dict(seed=seed, rate=rate, cost=cost, k=k, algorithm=algorithm,
                           is_control_cell=bool((rate, cost) == CONTROL_COMBO),
                           n_poison=int(is_poison.sum()), n_benign=int(is_poison.size),
                           silhouette=_finite_or_none(sil),
                           cluster_sizes=np.bincount(labels, minlength=k).tolist())
                for rule_name, flag in (("relative_size", rel), ("silhouette_gate", gated)):
                    rho, verdict = _independence(flag, spec)
                    row[rule_name] = dict(
                        n_flagged=int(flag.sum()),
                        recall=_finite_or_none(poison_recall(flag, is_poison)
                                               if is_poison.sum() else None),
                        purity=_finite_or_none((flag & is_poison).sum() / flag.sum()
                                               if flag.sum() else None),
                        fpr=_finite_or_none((flag & ~is_poison).sum() / max((~is_poison).sum(), 1)),
                        spearman_vs_spectral=rho, independence=verdict)
                row["gap_statistic"] = gap
                store[key] = row
                sink.append(row)
                save_cb()
                r = row["relative_size"]
                print(f"  cell=({seed},{rate},{cost}) k={k} {algorithm}: "
                      f"relsize recall={r['recall']} purity={r['purity']} "
                      f"indep={r['independence']} sil={row['silhouette']}")


def run_exre_boundary(cells, S, cfg, device, store, sink, save_cb, victims):
    """ExRe at k=2 only, on the relative-size-flagged minority only.

    Chen et al. assume clusters of comparable retrainable size. This dataset splits 52-vs-111614 and
    30-vs-91636, so excluding the MAJORITY and retraining on ~100 rows would measure sample
    starvation, not backdoor removal. Scoped from the diagnosed mechanism, not fitted.
    """
    for (seed, rate, cost) in cells:
        for algorithm in config.AC_EXRE_ALGORITHMS:
            key = f"{seed}|{rate}|{cost}|{algorithm}"
            if key in store:
                sink.append(store[key])
                continue
            V = victims(seed, rate, cost)
            labels, _, _ = cluster_and_reduce(
                V["feats"], n_components=AC_PCA_COMPONENTS, seed=seed,
                silhouette_sample=AC_SILHOUETTE_SAMPLE,
                n_clusters=config.AC_EXRE_CANDIDATE_K, algorithm=algorithm)
            minority = flag_by_relative_size(labels)
            excluded_rows = V["benign_pos"][minority]

            keep = np.ones(V["x_p_std"].shape[0], dtype=bool)
            keep[excluded_rows] = False
            retrained = train_mlp(V["x_p_std"][keep], V["y_p"][keep], seed,
                                  epochs=cfg["mlp_epochs"], device=device)
            with torch.no_grad():
                xb = torch.as_tensor(V["x_p_std"][excluded_rows], dtype=torch.float32, device=device)
                preds = retrained(xb).argmax(dim=1).cpu().numpy()

            score = exclusionary_reclassification_score(
                preds, own_label=TARGET, other_label=1 - TARGET)
            row = dict(seed=seed, rate=rate, cost=cost, algorithm=algorithm,
                       k=config.AC_EXRE_CANDIDATE_K,
                       is_control_cell=bool((rate, cost) == CONTROL_COMBO),
                       n_excluded=int(minority.sum()),
                       minority_purity=_finite_or_none(
                           (minority & V["is_poison"]).sum() / minority.sum()
                           if minority.sum() else None),
                       flagged_poison_at_default_T=bool(
                           flag_by_exclusionary_reclassification(score["ratio"])),
                       **{f"exre_{k}": v for k, v in score.items()})
            store[key] = row
            sink.append(row)
            save_cb()
            print(f"  ExRe cell=({seed},{rate},{cost}) {algorithm}: n_excluded={row['n_excluded']} "
                  f"l={score['l']} p={score['p']} ratio={score['ratio']:.3f} "
                  f"purity={row['minority_purity']}")


def consolidate(cheap_rows, exre_rows) -> dict:
    """AC-9. One row per (k, algorithm, decision_rule), pooled over the evasion cells.

    Exists because CLAUDE.md is explicit that the contribution is a failure-boundary map, not a
    'we tested N detectors' table. Every verdict here traces to a spearman_vs_spectral value recorded
    in ac_repair_grid.json -- no summary row may report a verdict unbacked by a raw correlation.
    """
    out = []
    ev = [r for r in cheap_rows if not r["is_control_cell"]]
    for k in config.AC_K_GRID:
        for algorithm in config.AC_CLUSTER_ALGORITHMS:
            for rule in ("relative_size", "silhouette_gate"):
                sel = [r for r in ev if r["k"] == k and r["algorithm"] == algorithm]
                recalls = [r[rule]["recall"] for r in sel if r[rule]["recall"] is not None]
                fprs = [r[rule]["fpr"] for r in sel if r[rule]["fpr"] is not None]
                rhos = [r[rule]["spearman_vs_spectral"] for r in sel
                        if r[rule]["spearman_vs_spectral"] is not None]
                verdicts = [r[rule]["independence"] for r in sel]
                out.append(dict(
                    k=k, algorithm=algorithm, decision_rule=rule, n_cells=len(sel),
                    mean_recall=_finite_or_none(np.mean(recalls) if recalls else None),
                    mean_fpr=_finite_or_none(np.mean(fprs) if fprs else None),
                    mean_spearman_vs_spectral=_finite_or_none(np.mean(rhos) if rhos else None),
                    n_redundant=sum(1 for v in verdicts if v == "redundant"),
                    n_undetermined=sum(1 for v in verdicts if v == "undetermined"),
                    n_independent=sum(1 for v in verdicts if v == "independent")))
    ev_ex = [r for r in exre_rows if not r["is_control_cell"]]
    for algorithm in config.AC_EXRE_ALGORITHMS:
        sel = [r for r in ev_ex if r["algorithm"] == algorithm]
        ratios = [r["exre_ratio"] for r in sel if r["exre_ratio"] is not None]
        out.append(dict(k=config.AC_EXRE_CANDIDATE_K, algorithm=algorithm,
                        decision_rule="exclusionary_reclassification", n_cells=len(sel),
                        mean_exre_ratio=_finite_or_none(np.mean(ratios) if ratios else None),
                        n_flagged_poison_at_default_T=sum(
                            1 for r in sel if r["flagged_poison_at_default_T"])))
    return dict(rows=out,
                note="One row per (k, algorithm, decision_rule) pooled over the 20 evasion cells. "
                     "Independence counts are SCREENING labels: the 0.90 threshold was calibrated "
                     "for Task 7B's centroid construction and has not been re-derived for these, so "
                     "mean_spearman_vs_spectral is reported beside every verdict. 'undetermined' "
                     "means the flag vector was constant (the norm here, not an edge case) and "
                     "counts as gate-not-passed, never as independence demonstrated.")


def main():
    t0 = time.time()
    argv = sys.argv
    cfg = apply_overrides(get_config(smoke="--smoke" in argv), argv)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cells = build_cells(cfg["seeds"])
    n_cheap = len(cells) * len(config.AC_K_GRID) * len(config.AC_CLUSTER_ALGORITHMS)
    n_exre = len(cells) * len(config.AC_EXRE_ALGORITHMS)
    # Pre-flight, NOT a smoke observation: a --smoke --seeds 42 probe exercises 1 of 5 seeds and
    # would not surface a seed-dimension double-count, which only manifests once all 5 enumerate.
    if not cfg["smoke"]:
        assert len(cells) == 25, f"expected 25 cells, got {len(cells)}"
        assert n_cheap == 200, f"expected 200 cheap-grid units, got {n_cheap}"
        assert n_exre == 50, f"expected 50 ExRe units, got {n_exre}"

    print(f"device={device} seeds={cfg['seeds']} cells={len(cells)} "
          f"cheap_units={n_cheap} exre_units={n_exre}")

    ckpt_path = GRID_PATH.with_name(GRID_PATH.stem + ".checkpoint.json")
    S = load_setup(cfg)
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    stats = raw_trigger_stats(S["x_tr_raw"], S["constraints"])

    key = _config_key(cells, cfg)
    cheap_store, exre_store = load_checkpoint(ckpt_path, key)
    cheap_rows, exre_rows = [], []
    save_cb = lambda: save_checkpoint(ckpt_path, key, cheap_store, exre_store)

    # One victim per (seed, rate, cost), trained lazily and cached, so a cell whose units are all
    # checkpointed never trains anything (NC-3's skip stays effective under NC-4's cache).
    ranking_cache, victim_cache = {}, {}

    def victims(seed, rate, cost):
        if seed not in ranking_cache:
            clean = train_mlp(x_tr_std, S["y_tr"], seed, epochs=cfg["mlp_epochs"], device=device)
            ranking_cache[seed] = shap_rank_features(clean, S["x_tr_raw"], S["scaler"],
                                                     target=TARGET, kind="mlp", device=device)
            print(f"  [seed {seed}] clean model + SHAP ranking ({time.time() - t0:.0f}s)")
        ck = (seed, rate, cost)
        if ck not in victim_cache:
            victim_cache.clear()      # one victim in memory at a time: activations are ~114k x 64
            victim_cache[ck] = build_victim(seed, rate, cost, S, cfg, device,
                                            ranking_cache[seed], stats)
            print(f"  [victim {ck}] trained ({time.time() - t0:.0f}s)")
        return victim_cache[ck]

    print("\n=== cheap grid ===")
    run_cheap_grid(cells, S, cfg, device, cheap_store, cheap_rows, save_cb, victims)
    print("\n=== ExRe boundary ===")
    run_exre_boundary(cells, S, cfg, device, exre_store, exre_rows, save_cb, victims)

    out = dict(
        cells=[list(c) for c in cells], seeds=cfg["seeds"],
        evasion_combos=[list(c) for c in EVASION_COMBOS], control_combo=list(CONTROL_COMBO),
        k_grid=list(config.AC_K_GRID), algorithms=list(config.AC_CLUSTER_ALGORITHMS),
        recipe=dict(pca_components=AC_PCA_COMPONENTS, mlp_epochs=cfg["mlp_epochs"],
                    silhouette_sample=AC_SILHOUETTE_SAMPLE,
                    silhouette_gate_threshold=config.AC_SILHOUETTE_GATE_THRESHOLD,
                    gap_b=config.AC_GAP_STATISTIC_B, exre_k=config.AC_EXRE_CANDIDATE_K,
                    exre_t_default=config.AC_EXRE_T_DEFAULT, smoke=cfg["smoke"]),
        cheap_grid_rows=cheap_rows, exre_rows=exre_rows,
        note="AC repair grid (plan AC-6). Four decision rules applied to ONE clustering per "
             "(cell, k, algorithm) so any difference is attributable to the rule, not to a re-run "
             "of k-means. The silhouette gate is pre-registered as EXPECTED TO MISFIRE (this repo "
             "measured silhouette anti-correlating with recall at -0.5549) and the gap statistic as "
             "confirmatory only, per Chen et al.'s own negative finding. Independence verdicts are "
             "screening labels, not proof: the 0.90 threshold was calibrated for Task 7B's centroid "
             "construction and is not re-derived here, so the raw rho accompanies every verdict.",
    )
    GRID_PATH.write_text(json.dumps(out, indent=2))
    summary = consolidate(cheap_rows, exre_rows)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(f"\nwrote {GRID_PATH.relative_to(config.ROOT)} "
          f"({len(cheap_rows)} cheap rows, {len(exre_rows)} ExRe rows)")
    print(f"wrote {SUMMARY_PATH.relative_to(config.ROOT)} ({len(summary['rows'])} summary rows)")
    print(f"elapsed={time.time() - t0:.0f}s")
    print(f"checkpoint kept at {ckpt_path}")


if __name__ == "__main__":
    main()
