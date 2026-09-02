"""M2 poison-rate x trigger-cost sweep, attack-potency upper bound, tabular positive control, and the
M2 exit gate.

Formal gate (methodology's two conditions): realizable ASR >= 0.8 at poison <= 5% ON THE MLP across
seeds, AND every realizable trigger is 360-constraint-valid. The tabular positive control (detectors
catch the violating trigger on the MLP) is a SEPARATE reported prerequisite, not ANDed into the gate.
The violating trigger's ASR is the attack-potency upper bound the realizable attack is judged against.

GATE-DEFINITION CAVEAT (read before citing `asr_mlp_at_5pct_mean`): the gate's ASR statistic is the
BEST realizable-MLP ASR over every (cost, rate<=5%) cell per seed, not a single fixed operating point
-- it is an attacker-optimal ("best response within budget") framing, not a pre-registered cell. This
matters because the per-cell grid is highly cost-dependent (background ~0.03-0.05 at cost in {1,2,4},
rising at cost=8, saturating at cost=16 regardless of rate): the best-over-grid selection will report
1.0 as soon as ANY cell in the grid saturates, even if most cells don't. Report `asr_mlp_by_cost` (the
per-cost breakdown, mean over seeds) alongside the headline number so a reader sees the full shape, not
just the max. Do not add a cost value purely to inflate the headline; state honestly which cost band
the reported ASR came from.

Run:  python scripts/04_poison_sweep.py           # full 5-seed experiment (hours)
      python scripts/04_poison_sweep.py --smoke    # fast validation slice
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _m2_common import get_config, load_setup, apply_overrides
from src import config
from src.data import apply_standardiser
from src.detectors import poison_recall
from src.detectors.activation_clustering import detect as ac_detect
from src.detectors.neural_cleanse import nc_binary_flag, reverse_engineer_tabular
from src.detectors.spectral import flag_by_scores, spectral_scores
from src.models import (
    train_lightgbm, train_mlp, mlp_penultimate_features, clean_accuracy, attack_success_rate,
)
from src.tb_vendor.constraints_numeric import check_constraints
from src.trigger import shap_rank_features, build_trigger, raw_trigger_stats, apply_trigger

TARGET = config.ATTACK_TARGET
SPECTRAL_K = 5


def bootstrap_ci(vals, seed=0):
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return float(v.mean()), [float(v.min()), float(v.max())]
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, len(v), replace=True).mean()
                      for _ in range(config.BOOTSTRAP_RESAMPLES)])
    lo, hi = np.percentile(means, [(1 - config.BOOTSTRAP_CI) / 2 * 100,
                                   (1 + config.BOOTSTRAP_CI) / 2 * 100])
    return float(v.mean()), [float(lo), float(hi)]


def train_victim(kind, x_std, y, seed, epochs, device):
    return train_lightgbm(x_std, y, seed) if kind == "lgb" else train_mlp(
        x_std, y, seed, epochs=epochs, device=device)


def cell_key(seed, rate, cost, kind, variant):
    return f"{seed}|{rate}|{cost}|{kind}|{variant}"


def seed_setup(seed, S, cfg, device):
    """Per-seed setup: clean victims (also reused by the positive control) + per-victim SHAP ranking."""
    x_tr_std = apply_standardiser(S["scaler"], S["x_tr_raw"])
    clean = {"lgb": train_victim("lgb", x_tr_std, S["y_tr"], seed, cfg["mlp_epochs"], device),
             "mlp": train_victim("mlp", x_tr_std, S["y_tr"], seed, cfg["mlp_epochs"], device)}
    ranking = {
        "lgb": shap_rank_features(clean["lgb"], S["x_tr_raw"], S["scaler"], target=TARGET, kind="tree"),
        "mlp": shap_rank_features(clean["mlp"], S["x_tr_raw"], S["scaler"], target=TARGET,
                                  kind="mlp", device=device),
    }
    return clean, ranking


def sweep_seed(seed, S, cfg, device, ranking, cells, save_cb):
    """Resumable sweep for one seed: skips cells already in `cells`, saving after each new cell."""
    from src.poison import poison_trainset_cleanlabel

    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]
    x_te_std = apply_standardiser(S["scaler"], S["x_te_raw"])
    x_bot_raw = S["x_te_raw"][S["y_te"] == 1]
    stats = raw_trigger_stats(x_tr_raw, constraints)   # mu/sd/eq: same for every (kind, cost) this seed

    # (kind, cost) -> (realiz, violat, constraint_valid); constraint_valid is per-VARIANT (the
    # violating trigger is expected to be constraint-invalid by construction -- storing the
    # realizable trigger's validity on a violating row would be a category error).
    trig_cache = {}
    for rate in cfg["rates"]:
        for cost in cfg["costs"]:
            for kind in ("lgb", "mlp"):
                if (kind, cost) not in trig_cache:
                    realiz, violat = build_trigger(ranking[kind], cost, x_tr_raw, constraints, features, bounds,
                                                   stats=stats)
                    constraint_valid = {
                        "realizable": bool(check_constraints(
                            apply_trigger(x_bot_raw, realiz, constraints, features, bounds),
                            constraints, features).all()),
                        "violating": bool(check_constraints(
                            apply_trigger(x_bot_raw, violat), constraints, features).all()),
                    }
                    trig_cache[(kind, cost)] = (realiz, violat, constraint_valid)
                realiz, violat, constraint_valid = trig_cache[(kind, cost)]
                for variant, trig in (("realizable", realiz), ("violating", violat)):
                    key = cell_key(seed, rate, cost, kind, variant)
                    if key in cells:
                        continue
                    x_p_std, _, _ = poison_trainset_cleanlabel(
                        x_tr_raw, y_tr, trig, rate, S["scaler"], target=TARGET, seed=seed,
                        constraints=constraints, feature_names=features, bounds=bounds)
                    victim = train_victim(kind, x_p_std, y_tr, seed, cfg["mlp_epochs"], device)
                    x_bot_trig = apply_trigger(x_bot_raw, trig, constraints, features, bounds)
                    asr = attack_success_rate(victim, apply_standardiser(S["scaler"], x_bot_trig), TARGET, device)
                    cacc = clean_accuracy(victim, x_te_std, S["y_te"], device)
                    cells[key] = dict(seed=seed, rate=rate, cost=cost, victim=kind, variant=variant,
                                      asr=asr, clean_acc=cacc, realizable_valid=constraint_valid[variant])
                    save_cb()


def positive_control(seed, S, cfg, device, clean_mlp, rank_mlp):
    """Wiring/sanity control: a VIOLATING trigger on the MLP that Spectral/AC/NC SHOULD catch. Isolates
    'portability genuinely fails' from 'the tabular detector wiring is broken'. NOT P2 crossover evidence.
    Reuses the seed's already-trained clean MLP + ranking (no redundant training)."""
    from src.poison import poison_trainset_cleanlabel

    features, constraints, bounds = S["features"], S["constraints"], S["bounds"]
    x_tr_raw, y_tr = S["x_tr_raw"], S["y_tr"]

    _, violat = build_trigger(rank_mlp, cfg["control_cost"], x_tr_raw, constraints, features, bounds)

    x_p_std, y_p, poison_idx = poison_trainset_cleanlabel(
        x_tr_raw, y_tr, violat, cfg["control_rate"], S["scaler"], target=TARGET, seed=seed)
    mlp_bd = train_victim("mlp", x_p_std, y_p, seed, cfg["mlp_epochs"], device)

    # detectors operate on the target-class-conditioned (benign) subset; poison lives among benign rows
    benign_pos = np.where(y_p == TARGET)[0]
    is_poison = np.isin(benign_pos, poison_idx)
    feats = mlp_penultimate_features(mlp_bd, x_p_std[benign_pos], device)

    scores = spectral_scores(feats, n_components=SPECTRAL_K)
    spec_recall = poison_recall(flag_by_scores(scores, expected_frac=float(is_poison.mean())), is_poison)
    ac_mask, ac_sil = ac_detect(feats, seed=seed)
    ac_recall = poison_recall(ac_mask, is_poison)
    ac_small_frac = float(ac_mask.mean())   # size of AC's "poison" (smaller) cluster; diagnoses the
    #                                         outlier-dominated 2-means split (see notes 20260715)

    std_lo, std_hi = S["std_bounds"]
    bnd = (torch.tensor(std_lo, dtype=torch.float32), torch.tensor(std_hi, dtype=torch.float32))
    sample = torch.tensor(x_p_std[benign_pos][: cfg["nc_sample"]], dtype=torch.float32)
    bd_norms = reverse_engineer_tabular(mlp_bd, 2, sample, device=device, steps=cfg["nc_steps"], bounds=bnd)
    cl_norms = reverse_engineer_tabular(clean_mlp, 2, sample, device=device, steps=cfg["nc_steps"], bounds=bnd)
    nc_flag, nc_ratio = nc_binary_flag(bd_norms, tau=2.0)
    _, nc_clean_ratio = nc_binary_flag(cl_norms, tau=2.0)
    # NC "discriminates" iff the backdoored model's mask-ratio exceeds the clean-calibrated one. On
    # imbalanced binary NIDS the clean model already has a large class-mask asymmetry, so NC often
    # does NOT add discrimination -- an H1 finding, reported not gated. The control PASS validates the
    # ACTIVATION-based detectors (Spectral/AC), whose MLP-penultimate-feature wiring is what a tabular
    # null depends on; NC's inversion wiring is validated separately by tests/detectors/
    # test_neural_cleanse_tabular.py (Task 0b) on a controlled MLP.
    nc_discriminates = bool(nc_flag == TARGET and nc_ratio > nc_clean_ratio)
    # Control criterion (2026-07-15): PASS iff the activation-extraction WIRING is validated, i.e.
    # Spectral (top-k, same MLP penultimate features) catches the blatant violating trigger. AC and NC
    # failing even on the blatant trigger are DIAGNOSED H1 findings (AC: outlier-dominated 2-means, even
    # the faithful variant -- notes/20260715-finding-ac-fails-tabular-scale.md; NC: class-mask asymmetry
    # under imbalance), not wiring bugs -- M1 validated both implementations on vision. So they are
    # reported, not gated. This is a scientific-interpretation decision recorded in that note.
    ac_catches = bool(ac_recall >= 0.7)
    nc_catches = nc_discriminates
    passed = bool(spec_recall >= 0.7)   # activation-extraction wiring proven
    return dict(seed=seed, spectral_recall=spec_recall, ac_recall=ac_recall, ac_silhouette=ac_sil,
                ac_small_cluster_frac=ac_small_frac, ac_catches=ac_catches,
                nc_flagged=int(nc_flag), nc_ratio=nc_ratio, nc_clean_ratio=nc_clean_ratio,
                nc_discriminates=nc_discriminates, nc_catches=nc_catches, passed=passed)


CKPT = "poison_sweep.checkpoint.json"


def _config_key(cfg):
    """Checkpoint compatibility key -- deliberately EXCLUDES the seed list, so a single-seed
    calibration run's cells are reused by the full run (only rates/costs/scale/control must match)."""
    return dict(smoke=cfg["smoke"], rates=list(cfg["rates"]), costs=list(cfg["costs"]),
                mlp_epochs=cfg["mlp_epochs"], nc_steps=cfg["nc_steps"], subsample=cfg["subsample"],
                control_rate=cfg["control_rate"], control_cost=cfg["control_cost"])


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
    print(f"resuming from checkpoint: {len(blob.get('cells', {}))} cells, "
          f"{len(blob.get('controls', {}))} controls already done")
    return blob.get("cells", {}), blob.get("controls", {})


def save_checkpoint(path, key, cells, controls):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(config_key=key, cells=cells, controls=controls)))
    tmp.replace(path)   # atomic: a crash mid-write never corrupts the checkpoint


def main():
    t0 = time.time()
    smoke = "--smoke" in sys.argv
    cfg = apply_overrides(get_config(smoke), sys.argv)
    config.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  smoke={smoke}  seeds={cfg['seeds']}  rates={cfg['rates']}  costs={cfg['costs']}")

    S = load_setup(cfg)
    print(f"train balance={S['train_balance']}  test balance={S['test_balance']}")

    key = _config_key(cfg)
    ckpt_path = config.RESULTS / CKPT
    cells, controls = load_checkpoint(ckpt_path, key)
    save_cb = lambda: save_checkpoint(ckpt_path, key, cells, controls)

    for seed in cfg["seeds"]:
        seed_keys = [cell_key(seed, r, c, k, v)
                     for r in cfg["rates"] for c in cfg["costs"]
                     for k in ("lgb", "mlp") for v in ("realizable", "violating")]
        if all(sk in cells for sk in seed_keys) and str(seed) in controls:
            print(f"--- seed {seed}: already complete, skipping ---")
            continue
        print(f"--- seed {seed} ({time.time() - t0:.0f}s elapsed) ---")
        clean, ranking = seed_setup(seed, S, cfg, device)
        sweep_seed(seed, S, cfg, device, ranking, cells, save_cb)
        if str(seed) not in controls:
            controls[str(seed)] = positive_control(seed, S, cfg, device, clean["mlp"], ranking["mlp"])
            save_cb()

    sweep_rows = list(cells.values())
    control_rows = list(controls.values())

    # gate: realizable MLP ASR at rate<=0.05, BEST OVER COST per seed -> mean + bootstrap CI.
    # See the GATE-DEFINITION CAVEAT in the module docstring -- this is an attacker-optimal selection,
    # not a fixed operating point; asr_mlp_by_cost below reports the full per-cost shape so the
    # headline number can't be read as "the attack works at every cost."
    per_seed_best = []
    for seed in cfg["seeds"]:
        vals = [r["asr"] for r in sweep_rows if r["seed"] == seed and r["victim"] == "mlp"
                and r["variant"] == "realizable" and r["rate"] <= 0.05]
        if vals:
            per_seed_best.append(max(vals))
    asr_mean, asr_ci = bootstrap_ci(per_seed_best) if per_seed_best else (0.0, [0.0, 0.0])
    lgb_best = [r["asr"] for r in sweep_rows if r["victim"] == "lgb" and r["variant"] == "realizable"
                and r["rate"] <= 0.05]
    potency_mlp = [r["asr"] for r in sweep_rows if r["victim"] == "mlp" and r["variant"] == "violating"
                   and r["rate"] <= 0.05]
    asr_mlp_by_cost = {}
    for cost in cfg["costs"]:
        vals = [r["asr"] for r in sweep_rows if r["victim"] == "mlp" and r["variant"] == "realizable"
                and r["rate"] <= 0.05 and r["cost"] == cost]
        if vals:
            asr_mlp_by_cost[str(cost)] = float(np.mean(vals))

    asr_gate = bool(asr_mean >= 0.8)
    realizable_valid = bool(all(r["realizable_valid"] for r in sweep_rows if r["variant"] == "realizable"))
    control_passed = bool(all(c["passed"] for c in control_rows))

    summary = dict(
        asr_mlp_at_5pct_mean=asr_mean, asr_mlp_at_5pct_ci=asr_ci,
        asr_mlp_by_cost=asr_mlp_by_cost,
        asr_lgb_at_5pct_max=(max(lgb_best) if lgb_best else None),
        potency_upper_bound_mlp_at_5pct_max=(max(potency_mlp) if potency_mlp else None),
        asr_gate=asr_gate, realizable_valid=realizable_valid,
        formal_gate_pass=bool(asr_gate and realizable_valid),
        control_passed=control_passed,
    )
    out = dict(config=dict(smoke=smoke, seeds=cfg["seeds"], rates=list(cfg["rates"]),
                           costs=list(cfg["costs"]), target=TARGET, spectral_k=SPECTRAL_K,
                           train_balance=S["train_balance"], test_balance=S["test_balance"]),
               per_cell=sweep_rows, summary=summary)
    (config.RESULTS / "poison_sweep.json").write_text(json.dumps(out, indent=2))
    (config.RESULTS / "tabular_positive_control.json").write_text(
        json.dumps(dict(config=out["config"], per_seed=control_rows,
                        passed=control_passed), indent=2))

    print(json.dumps(summary, indent=2))
    print(f"FORMAL M2 GATE (ASR>=0.8 on MLP AND realizable 360-valid): "
          f"{'PASS' if summary['formal_gate_pass'] else 'FAIL'}")
    print(f"Tabular positive control (separate prerequisite): {'PASS' if control_passed else 'FAIL'}")
    if not summary["formal_gate_pass"]:
        print(f"NOTE: realizable ASR={asr_mean:.3f} vs potency ceiling="
              f"{summary['potency_upper_bound_mlp_at_5pct_max']} -- if the ceiling is high, "
              f"constraint realizability is the binding cost (H2). Record the crossover; do not fake it.")

    elapsed = time.time() - t0
    n_seeds = len(cfg["seeds"])
    print(f"\nelapsed={elapsed:.0f}s for {n_seeds} seed(s) "
          f"(~{elapsed / max(n_seeds, 1):.0f}s/seed -> full 5-seed run ~{elapsed / max(n_seeds, 1) * 5 / 60:.0f} min)")
    print(f"checkpoint kept at {ckpt_path} -- the full run reuses these seed(s) and only computes the rest")


if __name__ == "__main__":
    main()
