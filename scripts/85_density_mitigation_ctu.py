"""Does a defense built for tabular security data catch what the vision ports miss?

Every detector in this manuscript is a defense designed for image classifiers and carried across.
That leaves a reader unable to separate "vision-built detectors fail here" from "this attack defeats
any detector". Severi et al.'s density-based clean-label mitigation (MILCOM 2025, arXiv:2407.08159)
is the missing comparator: built for tabular security data, evaluated by its authors on CTU-13, and
model-agnostic, so it also runs on the LightGBM victim that no activation-based detector can touch.

Pre-registered in notes/20260903-decision-density-mitigation-preregistration.md BEFORE this ran. That
note fixes the method, the HDBSCAN-for-OPTICS deviation, the clustering parameters, the cells, both
victims, the loud-control gate at recall 0.90, the four reporting axes and the three-way decision
rule. Nothing here changes any of it, and no hyperparameter is chosen on an outcome.

The gate runs first, by construction of this driver and not by convention: --control-only scores the
constraint-VIOLATING trigger that the method is expected to catch loudly, and the full run refuses to
start until that control has passed on every seed. A null without a passing control is
uninterpretable and would be thrown away.

The cell loop is copied from scripts/53_spectre_window.py rather than re-derived, and the loud
control's construction is copied from scripts/04_poison_sweep.py:131 positive_control, so this arm
differs from the committed grid only in the detector.

Checkpointed per cell-seed-victim with an atomic temp-file-then-rename write and a config
fingerprint, so a kill mid-run costs at most one cell.

Run:  tmux new-session -d -s density '.venv/bin/python -u scripts/85_density_mitigation_ctu.py 2>&1 | tee logs/85_density_mitigation_ctu.log'
      .venv/bin/python scripts/85_density_mitigation_ctu.py --control-only
      .venv/bin/python scripts/85_density_mitigation_ctu.py --smoke
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup
from src import config
from src.data import apply_standardiser
from src.detectors import poison_recall
from src.detectors.density_mitigation import (
    cluster_composition, cluster_target_class, filter_mask, flag_fixed_threshold,
    flag_loss_delta, iterative_scoring, select_features,
)
from src.detectors.spectral import flag_by_mad_threshold
from src.models import attack_success_rate, clean_accuracy, train_lightgbm, train_mlp
from src.poison import poison_trainset_cleanlabel
from src.trigger import apply_trigger, build_trigger, raw_trigger_stats, shap_rank_features

TARGET = config.ATTACK_TARGET

# The four evasion-window cells the manuscript anchors on, plus the matched-rate control cell.
WINDOW_CELLS = [(0.005, 8), (0.005, 16), (0.01, 8), (0.01, 16)]
MATCHED_RATE_CELL = (0.10, 16)
CELLS = WINDOW_CELLS + [MATCHED_RATE_CELL]
VICTIMS = ("mlp", "lgb")

# Severi et al.'s own settings. |F| = 4 is their CTU-13 choice, w = 5% the absorption window, 80% the
# stopping threshold, z_t = 2 the loss-delta cut. The two the paper does not publish -- the density
# clusterer's own parameters -- are fixed in the pre-registration, with the reason, before any cell.
N_FEATURES = 4
MIN_CLUSTER_SIZE_FRAC = 0.0025
MIN_SAMPLES = 50
WINDOW_FRAC = 0.05
STOP_FRAC = 0.80
Z_T = 2.0

# The budget-free rule every other detector in this paper is measured under.
Z_THRESHOLDS = (2.0, 2.5, 3.0)

# The loud control, from the pre-registration: violating trigger, rate 0.05, cost 8, MLP, five seeds.
CONTROL_RATE = 0.05
CONTROL_COST = 8
CONTROL_BAR = 0.90

OUT = config.RESULTS / "density_mitigation_ctu.json"
CKPT = config.RESULTS / "density_mitigation_ctu.checkpoint.json"
SMOKE_OUT = config.RESULTS / "density_mitigation_ctu.smoke.json"
SMOKE_CKPT = config.RESULTS / "density_mitigation_ctu.smoke.checkpoint.json"
# The tree-ranked control is a follow-up probe, authorized after the pre-registered gate failed. It
# gets its own paths so it can never overwrite that gate's artifact.
LGB_OUT = config.RESULTS / "density_mitigation_ctu.control_lgb.json"
LGB_CKPT = config.RESULTS / "density_mitigation_ctu.control_lgb.checkpoint.json"
# The wiring probe: the trigger is placed in the features the defense's own stage 1 keeps, so a miss
# can no longer be explained by the reduction having discarded it.
SEL_OUT = config.RESULTS / "density_mitigation_ctu.control_selected.json"
SEL_CKPT = config.RESULTS / "density_mitigation_ctu.control_selected.checkpoint.json"
# Stage 1 bypassed entirely: stages 2 to 4 are handed the trigger's own features. This asks whether
# the clustering and the iterative scoring work at all on real data, with the reduction -- the stage
# every control so far has died in -- removed from the question.
BYPASS_OUT = config.RESULTS / "density_mitigation_ctu.control_bypass.json"
BYPASS_CKPT = config.RESULTS / "density_mitigation_ctu.control_bypass.checkpoint.json"
# --composition: the bypass control re-run with per-cluster poison composition recorded. Written
# to its own files so the pre-registered artifacts above are never overwritten by a post-hoc run.
COMP_OUT = config.RESULTS / "density_mitigation_ctu.control_bypass.composition.json"
COMP_CKPT = config.RESULTS / "density_mitigation_ctu.control_bypass.composition.checkpoint.json"

CONTROL_RANKINGS = ("mlp", "lgb", "selected")

PREREG = "notes/20260903-decision-density-mitigation-preregistration.md"


def cell_key(seed, rate, cost, victim) -> str:
    return f"{seed}|{rate}|{cost}|{victim}"


def control_key(seed) -> str:
    return f"control|{seed}"


def config_key(cfg) -> dict:
    """Every input that changes a result. Changing one invalidates completed compute rather than
    silently mixing two configurations."""
    return dict(cells=[list(c) for c in CELLS], seeds=list(cfg["seeds"]), victims=list(VICTIMS),
                n_features=N_FEATURES, min_cluster_size_frac=MIN_CLUSTER_SIZE_FRAC,
                min_samples=MIN_SAMPLES, window_frac=WINDOW_FRAC, stop_frac=STOP_FRAC, z_t=Z_T,
                z=list(Z_THRESHOLDS), mlp_epochs=cfg["mlp_epochs"], subsample=cfg["subsample"],
                control_rate=CONTROL_RATE, control_cost=CONTROL_COST,
                control_ranking=cfg["control_ranking"], bypass_stage1=cfg["bypass_stage1"],
                composition=cfg["composition"], smoke=cfg["smoke"])


def load_ckpt(key, path) -> dict:
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if blob.get("config_key") != key:
        print("checkpoint config mismatch -- starting fresh")
        return {}
    done = blob.get("rows", {})
    print(f"resuming from checkpoint: {len(done)} units already done")
    return done


def save_ckpt(key, rows, path) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, rows=rows)))
    tmp.replace(path)          # atomic: a crash mid-write never corrupts the checkpoint


def _finite_or_none(v):
    return float(v) if v is not None and np.isfinite(v) else None


def entropy_importance_ranking(x_std, y, seed: int) -> list:
    """Every feature ordered by the SAME entropy-tree importance stage 1 reduces with.

    This is the defender's own notion of importance, used here to aim the attacker's trigger INTO
    the subspace stage 1 keeps. `select_features` returns its picks sorted by index rather than by
    rank, so the ordering cannot be recovered from it and is recomputed; the caller asserts the two
    agree on the top `N_FEATURES`, which fails loudly rather than silently drifting if the module's
    tree specification ever changes.
    """
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=seed)
    tree.fit(np.asarray(x_std, dtype=np.float64), np.asarray(y))
    return [int(i) for i in np.argsort(-tree.feature_importances_)]


def train_victim(kind, x_std, y, seed, epochs, device):
    """The 04/05 recipe, unchanged."""
    return train_lightgbm(x_std, y, seed) if kind == "lgb" else train_mlp(
        x_std, y, seed, epochs=epochs, device=device)


def seed_setup(seed, S, cfg, device):
    """Clean victim and SHAP ranking per victim kind. Reused by every cell of this seed and by the
    clean-victim ASR baseline, so neither is refit per cell."""
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    clean = {k: train_victim(k, x_tr_std, S["y_tr"], seed, cfg["mlp_epochs"], device)
             for k in VICTIMS}
    ranking = {
        "lgb": shap_rank_features(clean["lgb"], S["x_tr_raw"], S["scaler"], target=TARGET,
                                  kind="tree"),
        "mlp": shap_rank_features(clean["mlp"], S["x_tr_raw"], S["scaler"], target=TARGET,
                                  kind="mlp", device=device),
    }
    ranking["selected"] = entropy_importance_ranking(x_tr_std, S["y_tr"], seed)
    # The two code paths must agree about what stage 1 keeps, or the wiring probe is testing
    # something other than the reduction it claims to target.
    top = sorted(ranking["selected"][:N_FEATURES])
    picked = sorted(int(f) for f in select_features(x_tr_std, S["y_tr"],
                                                    n_features=N_FEATURES, seed=seed))
    assert top == picked, f"ranking helper disagrees with select_features: {top} vs {picked}"
    return clean, ranking


def run_detector(x_p_std, y_p, poison_idx, seed, force_features=None):
    """The four stages on one poisoned training set, in standardized space.

    Standardized rather than raw because it is the space every other detector in this paper sees,
    and because both stages that consume the geometry here -- an entropy tree and a density
    clusterer -- are unaffected by the monotone rescaling for the tree and are given a common scale
    for the clusterer.
    """
    target_pos = np.flatnonzero(y_p == TARGET)
    is_poison = np.isin(target_pos, poison_idx)

    # `force_features` replaces stage 1 rather than informing it. Used only by the bypass probe, so
    # that a miss downstream cannot be blamed on the reduction.
    feats = (np.asarray(force_features, dtype=int) if force_features is not None
             else select_features(x_p_std, y_p, n_features=N_FEATURES, seed=seed))
    labels = cluster_target_class(x_p_std[target_pos][:, feats],
                                  min_cluster_size_frac=MIN_CLUSTER_SIZE_FRAC,
                                  min_samples=MIN_SAMPLES)
    scoring = iterative_scoring(x_p_std[:, feats], y_p, labels, TARGET, train_lightgbm,
                                window_frac=WINDOW_FRAC, stop_frac=STOP_FRAC, seed=seed)
    return feats, labels, scoring, target_pos, is_poison


def score_axes(scoring, labels, is_poison) -> dict:
    """The four axes the pre-registration fixes, never blended: recall under each published rule,
    recall under the budget-free MAD rule, ranking AUC, and FPR over clean target-class rows."""
    n_clean = int((~is_poison).sum())
    score = np.asarray(scoring["per_sample_score"], dtype=float)

    def fpr(flag):
        return float((flag & ~is_poison).sum() / n_clean) if n_clean else None

    fixed = flag_fixed_threshold(scoring, labels)
    delta = flag_loss_delta(scoring, labels, z_t=Z_T)

    out = dict(
        fixed_recall=_finite_or_none(poison_recall(fixed, is_poison)),
        fixed_fpr=fpr(fixed),
        fixed_flagged_frac=float(fixed.mean()),
        delta_recall=_finite_or_none(poison_recall(delta, is_poison)),
        delta_fpr=fpr(delta),
        delta_flagged_frac=float(delta.mean()),
        auc=(float(roc_auc_score(is_poison, score))
             if 0 < is_poison.sum() < len(is_poison) else None),
        mad_recall={}, mad_fpr={},
        n_clusters=scoring["n_clusters"], n_unabsorbed=len(scoring["unabsorbed_clusters"]),
        n_iterations=scoring["n_iterations"], few_clusters=scoring["few_clusters"],
    )
    for z in Z_THRESHOLDS:
        flag = flag_by_mad_threshold(score, z_thresh=z)
        out["mad_recall"][str(z)] = _finite_or_none(poison_recall(flag, is_poison))
        out["mad_fpr"][str(z)] = fpr(flag)
    return out


def run_control(seed, S, cfg, device, ranking, ranking_kind: str = "mlp",
                bypass_stage1: bool = False) -> dict:
    """The gate. A constraint-VIOLATING trigger the method should catch loudly, built exactly as
    scripts/04_poison_sweep.py:131 positive_control builds it: no constraint projection on the
    poisoning call, MLP victim, rate 0.05, cost 8.

    This is a tabular-native method, so the MNIST patch control that gates the vision ports is
    inapplicable by design and is not run.

    `ranking_kind` selects which model's SHAP picked the trigger features. The pre-registered gate
    uses "mlp". The follow-up probe uses "lgb", whose TreeSHAP ranking comes from a tree, as stage
    1's entropy importance does; everything else -- victim, rate, cost, the violating trigger -- is
    held fixed, so the only variable is the attacker's notion of an important feature.
    """
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]

    _, violat = build_trigger(ranking[ranking_kind], CONTROL_COST, x_tr_raw, constraints, features,
                              bounds)
    if ranking_kind == "selected":
        # The probe is only meaningful if the trigger really did land in the reduced space. Recorded
        # per seed below as n_trigger_features_selected; a zero there would void the probe.
        print(f"  wiring probe: tree-ranked trigger on features {list(violat['indices'])}")
    x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
        x_tr_raw, y_tr, violat, CONTROL_RATE, S["scaler"], target=TARGET, seed=seed)

    trig_idx = [int(i) for i in violat["indices"]]
    forced = trig_idx[:N_FEATURES] if bypass_stage1 else None
    if bypass_stage1:
        print(f"  stage 1 BYPASSED: clustering forced onto trigger features {forced}")
    feats, labels, scoring, _, is_poison = run_detector(x_p_std, y_p, poison_idx, seed,
                                                        force_features=forced)
    axes = score_axes(scoring, labels, is_poison)
    # The diagnostic the gate-failure note left unmeasured: does the clusterer isolate the poison
    # at all, and if it does, did the rule absorb that cluster? Cheap, and it separates "port
    # defective" from "decision rule fails". Authorized by the user 2026-09-03.
    axes["cluster_composition"] = cluster_composition(labels, is_poison,
                                                      scoring["unabsorbed_clusters"])
    return dict(seed=seed, rate=CONTROL_RATE, cost=CONTROL_COST, victim="mlp", variant="violating",
                ranking_kind=ranking_kind, stage1_bypassed=bool(bypass_stage1),
                n_poison=int(is_poison.sum()), n_target=int(len(is_poison)),
                selected_features=[int(f) for f in feats], trigger_features=trig_idx,
                n_trigger_features_selected=len(set(int(f) for f in feats) & set(trig_idx)),
                **axes)


def run_cell(seed, rate, cost, kind, S, cfg, device, clean, ranking, trig_cache, E) -> dict:
    """One (seed, rate, cost, victim) cell: attack, then detector, then filtering sanitization.

    `trig_cache` is scoped to a single seed by the caller: the trigger is built from that seed's own
    SHAP ranking, so a cache carried across seeds would stamp the first seed's trigger on all of them.
    """
    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    x_te_std, x_bot_raw, stats = E["x_te_std"], E["x_bot_raw"], E["stats"]

    if (kind, cost) not in trig_cache:
        trig_cache[(kind, cost)] = build_trigger(ranking[kind], cost, x_tr_raw, constraints,
                                                 features, bounds, stats=stats)
    realiz, _ = trig_cache[(kind, cost)]

    x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
        x_tr_raw, y_tr, realiz, rate, S["scaler"], target=TARGET, seed=seed,
        constraints=constraints, feature_names=features, bounds=bounds)
    victim = train_victim(kind, x_p_std, y_p, seed, cfg["mlp_epochs"], device)

    x_bot_trig_std = apply_standardiser(S["scaler"], apply_trigger(x_bot_raw, realiz, constraints,
                                                                   features, bounds))
    asr = attack_success_rate(victim, x_bot_trig_std, TARGET, device)
    cacc = clean_accuracy(victim, x_te_std, S["y_te"], device)
    # The same trigger against a victim trained on CLEAN data. Recorded so the attack-success gate
    # can be applied to the LightGBM arm from disk, without inventing a threshold here: on that
    # victim the attack is ineffective in the committed sweep, and that arm is reported as data
    # sanitization rather than as a detector null.
    asr_clean_victim = attack_success_rate(clean[kind], x_bot_trig_std, TARGET, device)

    feats, labels, scoring, target_pos, is_poison = run_detector(x_p_std, y_p, poison_idx, seed)
    if is_poison.sum() == 0:
        return dict(seed=seed, rate=rate, cost=cost, victim=kind, skipped="no poison in target class")
    axes = score_axes(scoring, labels, is_poison)

    # Filtering, under the published fixed rule -- the arm that yields an ASR-after-mitigation figure
    # comparable to this paper's removal experiments.
    keep = filter_mask(flag_fixed_threshold(scoring, labels), target_pos, len(y_p))
    filt = train_victim(kind, x_p_std[keep], y_p[keep], seed, cfg["mlp_epochs"], device)
    poison_left = int(np.isin(np.flatnonzero(keep), poison_idx).sum())

    return dict(seed=seed, rate=rate, cost=cost, victim=kind,
                n_poison=int(is_poison.sum()), n_target=int(len(is_poison)),
                selected_features=[int(f) for f in feats],
                trigger_features=[int(i) for i in realiz["indices"]],
                n_trigger_features_selected=len(set(int(f) for f in feats)
                                                & set(int(i) for i in realiz["indices"])),
                asr=asr, clean_acc=cacc, asr_clean_victim=asr_clean_victim,
                asr_after_filtering=attack_success_rate(filt, x_bot_trig_std, TARGET, device),
                clean_acc_after_filtering=clean_accuracy(filt, x_te_std, S["y_te"], device),
                n_dropped=int((~keep).sum()), n_poison_left=poison_left,
                **axes)


def summarize(rows, controls) -> tuple[dict, str]:
    """The pre-registered decision rule, applied to the window cells on the MLP victim."""
    def cell_mean(rate, cost, field):
        v = [r[field] for r in rows
             if r["victim"] == "mlp" and r["rate"] == rate and r["cost"] == cost
             and r.get(field) is not None]
        return float(np.mean(v)) if v else None

    per_cell = {f"{rate}|{cost}": dict(
        fixed_recall=cell_mean(rate, cost, "fixed_recall"),
        delta_recall=cell_mean(rate, cost, "delta_recall"),
        fixed_fpr=cell_mean(rate, cost, "fixed_fpr"),
        delta_fpr=cell_mean(rate, cost, "delta_fpr"),
        auc=cell_mean(rate, cost, "auc"),
    ) for rate, cost in CELLS}

    window = [per_cell[f"{r}|{c}"] for r, c in WINDOW_CELLS]
    have = [w for w in window if w["fixed_recall"] is not None and w["delta_recall"] is not None]

    verdict = "INCOMPLETE"
    if len(have) == len(WINDOW_CELLS):
        def a_rule_clears(w):
            """Some published rule reaches recall 0.70 at its OWN false-positive rate <= 0.10."""
            return any(w[f"{r}_recall"] >= 0.70 and (w[f"{r}_fpr"] or 0.0) <= 0.10
                       for r in ("fixed", "delta"))

        if all(a_rule_clears(w) for w in have):
            verdict = "PORTS-WHERE-VISION-FAILS"
        elif all(w["fixed_recall"] < 0.30 and w["delta_recall"] < 0.30 for w in have):
            verdict = "FAILS-LIKE-THE-PORTS"
        else:
            verdict = "MIXED"

    ctrl = [c["fixed_recall"] for c in controls if c.get("fixed_recall") is not None]
    summary = dict(n_rows=len(rows), per_cell_mlp=per_cell,
                   control_fixed_recall=ctrl,
                   control_min=float(np.min(ctrl)) if ctrl else None,
                   control_passed=bool(ctrl and min(ctrl) >= CONTROL_BAR))
    return summary, verdict


def main() -> int:
    t0 = time.time()
    argv = sys.argv[1:]
    smoke = "--smoke" in argv
    control_only = "--control-only" in argv
    cfg = get_config(smoke=smoke)
    cfg["control_ranking"] = (argv[argv.index("--control-ranking") + 1]
                              if "--control-ranking" in argv else "mlp")
    cfg["bypass_stage1"] = "--bypass-stage1" in argv
    cfg["composition"] = "--composition" in argv
    if cfg["composition"] and not cfg["bypass_stage1"]:
        print("--composition is only defined for the stage-1-bypass control; add --bypass-stage1")
        return 2
    if cfg["control_ranking"] not in CONTROL_RANKINGS:
        print(f"--control-ranking must be one of {CONTROL_RANKINGS}")
        return 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cells = [WINDOW_CELLS[0]] if smoke else CELLS
    out_path, ckpt_path = (SMOKE_OUT, SMOKE_CKPT) if smoke else (OUT, CKPT)
    if cfg["control_ranking"] == "lgb":
        out_path, ckpt_path = LGB_OUT, LGB_CKPT
    elif cfg["control_ranking"] == "selected":
        out_path, ckpt_path = SEL_OUT, SEL_CKPT
    if cfg["bypass_stage1"]:
        out_path, ckpt_path = BYPASS_OUT, BYPASS_CKPT
    if cfg["composition"]:
        out_path, ckpt_path = COMP_OUT, COMP_CKPT
    # The key does NOT record --control-only or the smoke cell subset: the smoke flag already
    # separates the two configurations, and keeping the gate run on the same key lets the full run
    # resume from the controls it already paid for.
    key = config_key(cfg)
    done = load_ckpt(key, ckpt_path)
    print(f"device={device}  cells={cells}  seeds={cfg['seeds']}  victims={VICTIMS}")

    S = load_setup(cfg)
    # Invariant across every cell: standardized test block, the botnet rows ASR is measured on, and
    # the raw trigger statistics. Computed once rather than per cell.
    E = dict(x_te_std=apply_standardiser(S["scaler"], S["x_te_raw"]),
             x_bot_raw=S["x_te_raw"][S["y_te"] == 1],
             stats=raw_trigger_stats(S["x_tr_raw"], S["constraints"]))

    # The gate, first. Every seed's control precedes any realizable cell.
    for seed in cfg["seeds"]:
        if control_key(seed) in done:
            continue
        print(f"--- control seed {seed} ({time.time() - t0:.0f}s) ---")
        _, ranking = seed_setup(seed, S, cfg, device)
        row = run_control(seed, S, cfg, device, ranking, cfg["control_ranking"],
                          bypass_stage1=cfg["bypass_stage1"])
        done[control_key(seed)] = row
        save_ckpt(key, done, ckpt_path)
        print(f"  control recall(80% rule)={row['fixed_recall']} auc={row['auc']} "
              f"fpr={row['fixed_fpr']} trigger_features_selected="
              f"{row['n_trigger_features_selected']}/{CONTROL_COST}")

    controls = [done[control_key(s)] for s in cfg["seeds"] if control_key(s) in done]
    ctrl_recalls = [c["fixed_recall"] for c in controls if c.get("fixed_recall") is not None]
    passed = bool(ctrl_recalls) and min(ctrl_recalls) >= CONTROL_BAR

    if smoke and not passed:
        print(f"SMOKE: control recall {min(ctrl_recalls) if ctrl_recalls else None} is below "
              f"{CONTROL_BAR}, but this is a subsampled wiring probe and NOT the pre-registered "
              f"gate. Continuing so the cell path is exercised. The gate is decided only by a "
              f"full-scale --control-only run.")
    elif not passed:
        blob = dict(preregistration=PREREG, device=device, blocked=True,
                    control_ranking=cfg["control_ranking"],
                    bypass_stage1=cfg["bypass_stage1"],
                    control_bar=CONTROL_BAR, controls=controls,
                    reason=("the loud control did not reach the pre-registered bar on every seed; "
                            "the method does not enter the paper and no parameter is tuned to pass"))
        out_path.write_text(json.dumps(blob, indent=2))
        print(f"GATE FAILED: min control recall {min(ctrl_recalls) if ctrl_recalls else None} "
              f"< {CONTROL_BAR}. Wrote {out_path} with blocked=True. Stopping.")
        return 1
    else:
        print(f"GATE PASSED: min control recall {min(ctrl_recalls):.4f} >= {CONTROL_BAR}")

    if control_only:
        blob = dict(preregistration=PREREG, device=device, blocked=False, control_only=True,
                    control_ranking=cfg["control_ranking"],
                    bypass_stage1=cfg["bypass_stage1"], control_bar=CONTROL_BAR,
                    controls=controls, smoke=smoke,
                    note=("gate passed for this control configuration; the pre-registered gate is "
                          "the control_ranking=mlp run in density_mitigation_ctu.json"))
        out_path.write_text(json.dumps(blob, indent=2))
        print(f"control-only run complete, wrote {out_path}, elapsed={time.time() - t0:.0f}s")
        return 0

    for seed in cfg["seeds"]:
        pending = [(r, c, k) for r, c in cells for k in VICTIMS
                   if cell_key(seed, r, c, k) not in done]
        if not pending:
            print(f"--- seed {seed}: all cells cached, skipping ---")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        clean, ranking = seed_setup(seed, S, cfg, device)
        trig_cache: dict = {}
        for rate, cost, kind in pending:
            t_cell = time.time()
            row = run_cell(seed, rate, cost, kind, S, cfg, device, clean, ranking, trig_cache, E)
            done[cell_key(seed, rate, cost, kind)] = row
            save_ckpt(key, done, ckpt_path)
            if "skipped" in row:
                print(f"  rate={rate} cost={cost} {kind}: {row['skipped']}")
                continue
            print(f"  rate={rate} cost={cost} {kind}: "
                  f"fixed={row['fixed_recall']:.4f} delta={row['delta_recall']:.4f} "
                  f"auc={row['auc']:.4f} fpr={row['fixed_fpr']:.4f} "
                  f"asr={row['asr']:.4f}->{row['asr_after_filtering']:.4f} "
                  f"clusters={row['n_clusters']} ({time.time() - t_cell:.0f}s)")

    rows = [done[cell_key(s, r, c, k)] for s in cfg["seeds"] for r, c in cells for k in VICTIMS
            if cell_key(s, r, c, k) in done and "skipped" not in done[cell_key(s, r, c, k)]]
    summary, verdict = summarize(rows, controls)
    blob = dict(cells=cells, seeds=list(cfg["seeds"]), victims=list(VICTIMS), device=device,
                preregistration=PREREG, blocked=False,
                params=dict(n_features=N_FEATURES, min_cluster_size_frac=MIN_CLUSTER_SIZE_FRAC,
                            min_samples=MIN_SAMPLES, window_frac=WINDOW_FRAC, stop_frac=STOP_FRAC,
                            z_t=Z_T, z_thresholds=list(Z_THRESHOLDS),
                            clusterer="HDBSCAN (declared substitution for OPTICS)",
                            surrogate="LightGBM"),
                control_bar=CONTROL_BAR, controls=controls, smoke=smoke,
                rows=rows, summary=summary, verdict=verdict)
    out_path.write_text(json.dumps(blob, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"PRE-REGISTERED VERDICT: {verdict}"
          if not smoke else "SMOKE RUN: no verdict, this is a subsampled wiring probe")
    print(f"wrote {out_path}  elapsed={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
