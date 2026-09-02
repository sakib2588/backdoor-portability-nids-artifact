"""M4 analysis: turn M3/M2/M1 results into headline statistics + the macros SSOT.

Reads results/detectors.json (M3), results/vision_control.json (M1), results/poison_sweep.json (M2).
Writes results/analysis.json (nested audit trail: per-seed arrays, per-cell grids) and
results/macros.json (flat name->value SSOT the manuscript cites). No parametric tests (project
protocol n<=30) -- inference is bootstrap CIs + Cohen's d_z effect sizes only. See the M4 plan doc.

Run:  python scripts/06_analyze_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.stats import bootstrap_ci, cohens_dz, paired_diff_ci

ANCHOR_COST = 16          # M4-D3: attacker-effective cell (ASR saturates here); rate = per-seed argmax<=0.05
RATE_CAP = 0.05
MATCHED_RATE = 0.1        # highest tested NIDS poison rate -- a second anchor where recall is not
                          # confounded by an extreme low-poison budget, contrasting with ANCHOR_COST/RATE_CAP
EVASION_ASR_MIN = 0.8     # attack is "effective" at this ASR or above
EVASION_RECALL_MAX = 0.5  # detector's removal budget recovers less than half the poison


ADAPTIVE_Z_PRIMARY = 3.0  # the most conservative of the pre-registered {2.0,2.5,3.0} sweep in
                          # scripts/08_evasion_ablation.py -- all three hit recall=1.0 on every
                          # evasion-window cell, so reporting the strictest (fewest false positives)
                          # is not cherry-picking a result, it is picking the least-cost threshold
                          # from a set that all achieve the same recall


def _load(name):
    path = config.RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run the upstream milestone first")
    return json.loads(path.read_text())


def _load_optional(name):
    """Like _load, but returns None instead of raising -- for follow-up artifacts (e.g. the M4
    evasion-ablation) that may not exist yet without blocking the rest of the analysis."""
    path = config.RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def _seed_sorted(rows, key="seed"):
    return sorted(rows, key=lambda r: r[key])


def main():
    detectors = _load("detectors.json")
    vision = _load("vision_control.json")

    seeds = list(config.SEEDS)
    main_grid = detectors["main_grid"]
    nc_boundary = detectors["nc_boundary"]
    h3_rows = detectors["h3"]

    # --- vision arm (per-seed, sorted) ---
    vis = _seed_sorted(vision["per_seed"])
    vis_spec = [r["spectral_recall"] for r in vis]
    vis_ac = [r["ac_recall"] for r in vis]
    vis_nc_detect = [1.0 if r["nc_anomaly_index"] > 2.0 else 0.0 for r in vis]

    # --- NIDS arm at the D3 anchor: cost=16, per-seed argmax-ASR cell within rate<=RATE_CAP ---
    # iterate sorted(seeds) so the NIDS arm is paired to the seed-sorted vision arm by index,
    # independent of config.SEEDS's declared order (d_z/paired-diff pair arm[i] to arm[i]).
    nids_spec, nids_ac, nids_spec_auc, anchor_cells = [], [], [], []
    for seed in sorted(seeds):
        cand = [r for r in main_grid if r["seed"] == seed and r["cost"] == ANCHOR_COST
                and r["rate"] <= RATE_CAP]
        if not cand:
            raise ValueError(f"no cost={ANCHOR_COST}, rate<={RATE_CAP} cell for seed {seed} in "
                             "detectors.json main_grid -- cannot anchor the D3 comparison")
        best = max(cand, key=lambda r: r["asr"])   # attacker-effective cell
        anchor_cells.append(dict(seed=seed, rate=best["rate"], cost=best["cost"], asr=best["asr"],
                                  spectral_auc=best["spectral"].get("auc")))
        nids_spec.append(best["spectral"]["recall"])
        nids_ac.append(best["ac"]["recall"])
        nids_spec_auc.append(best["spectral"].get("auc"))

    # --- MATCHED_RATE secondary anchor: cost=16, FIXED rate=MATCHED_RATE (not argmax-ASR) ---
    # Contrasts the D3 anchor (lowest tested rate, worst case for recall) against the highest tested
    # rate -- shows whether the "portability failure" is rate-driven, not a hard modality wall.
    matched_spec, matched_ac, matched_cells = [], [], []
    for seed in sorted(seeds):
        cand = [r for r in main_grid if r["seed"] == seed and r["cost"] == ANCHOR_COST
                and r["rate"] == MATCHED_RATE]
        if not cand:
            raise ValueError(f"no cost={ANCHOR_COST}, rate=={MATCHED_RATE} cell for seed {seed}")
        row = cand[0]
        matched_cells.append(dict(seed=seed, rate=row["rate"], cost=row["cost"], asr=row["asr"],
                                   spectral_auc=row["spectral"].get("auc")))
        matched_spec.append(row["spectral"]["recall"])
        matched_ac.append(row["ac"]["recall"])

    # --- paired portability effect (M4-D2): Spectral, AC ---
    def portability(detector, vis_arm, nids_arm):
        vmean, vci = bootstrap_ci(vis_arm)
        nmean, nci = bootstrap_ci(nids_arm)
        mean_diff, diff_ci = paired_diff_ci(vis_arm, nids_arm)
        dz = cohens_dz(vis_arm, nids_arm)
        return dict(detector=detector,
                    vision_recall_mean=vmean, vision_recall_ci=vci,
                    nids_recall_mean=nmean, nids_recall_ci=nci,
                    mean_recall_drop=mean_diff, recall_drop_ci=diff_ci,
                    cohens_dz=dz,
                    cohens_dz_note=(None if dz is not None else
                                    "sd of paired difference is 0 -- d_z undefined"))

    port_spec = portability("spectral", vis_spec, nids_spec)
    port_ac = portability("ac", vis_ac, nids_ac)
    matched_port_spec = portability("spectral", vis_spec, matched_spec)
    matched_port_ac = portability("ac", vis_ac, matched_ac)
    anchor_spec_auc_mean, anchor_spec_auc_ci = bootstrap_ci(nids_spec_auc)

    # --- evasion window (M4 follow-up): cells where the attack succeeds but the FIXED-BUDGET
    # removal rule recovers little poison, even though the ranking signal (AUC) is strong. This
    # is the honest reframe of the D3 anchor result -- recall collapses at low poison rate because
    # flag_by_scores's budget is proportional to expected_frac (src/detectors/spectral.py:40), not
    # because the tabular signal is absent. ---
    from collections import defaultdict
    cell_asr = defaultdict(list)
    cell_spec_recall = defaultdict(list)
    cell_spec_auc = defaultdict(list)
    for r in main_grid:
        key = (r["rate"], r["cost"])
        cell_asr[key].append(r["asr"])
        cell_spec_recall[key].append(r["spectral"]["recall"])
        if r["spectral"].get("auc") is not None:
            cell_spec_auc[key].append(r["spectral"]["auc"])

    evasion_cells = []
    for (rate, cost) in sorted(cell_asr):
        mean_asr = float(np.mean(cell_asr[(rate, cost)]))
        mean_recall = float(np.mean(cell_spec_recall[(rate, cost)]))
        if mean_asr >= EVASION_ASR_MIN and mean_recall < EVASION_RECALL_MAX:
            aucs = cell_spec_auc.get((rate, cost), [])
            evasion_cells.append(dict(rate=rate, cost=cost, mean_asr=mean_asr,
                                       mean_spectral_recall=mean_recall,
                                       mean_spectral_auc=float(np.mean(aucs)) if aucs else None))

    evasion_window = dict(
        asr_min=EVASION_ASR_MIN, recall_max=EVASION_RECALL_MAX, cells=evasion_cells,
        note="cells where the clean-label trigger succeeds (ASR>=asr_min) but Spectral's fixed-budget "
             "removal rule (top 1.5x*expected_frac(n), src/detectors/spectral.py:40) recovers less than "
             "recall_max of the poison -- while AUC stays high, showing the ranking signal is present "
             "and the miss is a removal-budget artefact at low poison rate, not an absent signal")

    # --- Neural Cleanse detection rates (M4-D4): rates, not d_z ---
    nc_nids_flags = [1.0 if r["nc_flagged_target"] else 0.0 for r in nc_boundary]
    nc_nids_rate, nc_nids_ci = bootstrap_ci(nc_nids_flags) if nc_nids_flags else (None, [None, None])
    nc_vis_rate, nc_vis_ci = bootstrap_ci(vis_nc_detect)

    # --- H3 imbalance confound (per-seed balanced vs imbalanced recall at the H3 op-point) ---
    h3 = _seed_sorted(h3_rows)
    def h3_arm(field, side):
        return [r[side][field] for r in h3 if r[side][field] is not None]
    h3_spec_imb = h3_arm("spectral_recall", "imbalanced")
    h3_spec_bal = h3_arm("spectral_recall", "balanced")
    h3_ac_imb = h3_arm("ac_recall", "imbalanced")
    h3_ac_bal = h3_arm("ac_recall", "balanced")

    # --- determinism self-check summary (D1 from M3) ---
    det_summary = detectors["summary"]

    # --- adaptive-threshold ablation (M4 follow-up, scripts/08_evasion_ablation.py): does a
    # budget-free MAD z-score threshold recover recall inside the evasion window? Optional -- only
    # present once 08_evasion_ablation.py has been run. Excludes the matched-rate control cell
    # (rate=0.1) from the aggregate: that cell already worked under the fixed budget, so it is not
    # part of the evasion-window claim, only a sanity check the new rule does not break a working case.
    evasion_ablation = _load_optional("evasion_ablation.json")
    adaptive_threshold = None
    if evasion_ablation is not None:
        ew_cells = {(c["rate"], c["cost"]) for c in evasion_window["cells"]}
        z_key = str(ADAPTIVE_Z_PRIMARY)
        ew_recalls, ew_fprs, control_recalls, control_fprs = [], [], [], []
        per_cell = {}
        for row in evasion_ablation["rows"]:
            n_poison, n_benign = row["n_poison"], row["n_benign"]
            n_clean = n_benign - n_poison
            a = row["adaptive"][z_key]
            recall, n_fp = a["recall"], a["n_false_positive"]   # exact count, no reconstruction
            fpr = (n_fp / n_clean) if (n_clean > 0 and recall is not None) else None
            cell = (row["rate"], row["cost"])
            target = (ew_recalls, ew_fprs) if cell in ew_cells else (control_recalls, control_fprs)
            if recall is not None and fpr is not None:   # keep bootstrap_ci's input finite-only
                target[0].append(recall)
                target[1].append(fpr)
            per_cell.setdefault(cell, dict(rate=cell[0], cost=cell[1], fixed_recalls=[],
                                           adaptive_recalls=[], adaptive_fprs=[]))
            per_cell[cell]["fixed_recalls"].append(row["fixed_budget_recall"])
            per_cell[cell]["adaptive_recalls"].append(recall)
            per_cell[cell]["adaptive_fprs"].append(fpr)
        recall_mean, recall_ci = bootstrap_ci(ew_recalls)
        fpr_mean, fpr_ci = bootstrap_ci(ew_fprs)
        control_recall_mean, _ = bootstrap_ci(control_recalls)
        control_fpr_mean, _ = bootstrap_ci(control_fprs)
        per_cell_summary = [
            dict(rate=v["rate"], cost=v["cost"],
                 fixed_recall_mean=float(np.mean(v["fixed_recalls"])),
                 adaptive_recall_mean=float(np.mean(v["adaptive_recalls"])),
                 adaptive_fpr_mean=float(np.mean(v["adaptive_fprs"])))
            for v in sorted(per_cell.values(), key=lambda d: (d["rate"], d["cost"]))]
        adaptive_threshold = dict(
            z_thresholds_tested=evasion_ablation["z_thresholds"], z_primary=ADAPTIVE_Z_PRIMARY,
            evasion_window_recall_mean=recall_mean, evasion_window_recall_ci=recall_ci,
            evasion_window_fpr_mean=fpr_mean, evasion_window_fpr_ci=fpr_ci,
            control_cell_recall_mean=control_recall_mean, control_cell_fpr_mean=control_fpr_mean,
            n_seeds=len(evasion_ablation["seeds"]), n_evasion_cells=len(ew_cells),
            per_cell=per_cell_summary,
            note="flag_by_mad_threshold (budget-free), NOT part of Tran et al. 2018's original "
                 "algorithm -- tests whether a statistically-motivated threshold recovers recall "
                 "where the fixed removal budget (flag_by_scores) misses it. Reported at "
                 f"z={ADAPTIVE_Z_PRIMARY} (the strictest of 3 pre-registered thresholds -- all three "
                 "achieved the same recall, so this is the least-false-positive choice, not "
                 "cherry-picking). fpr = false positives / clean population size, using the exact "
                 "n_false_positive count from evasion_ablation.json (not reconstructed from recall). "
                 "per_cell is the mean-over-seeds breakdown fig5_adaptive_threshold plots directly.")

    # --- rule_runtime aggregate (from evasion_ablation.json's per-row timing): how much faster the
    # adaptive MAD rule is than the fixed removal budget on identical score arrays ---
    rule_runtime_speedup_mean = None
    if evasion_ablation is not None:
        ratios = [row["rule_runtime"]["fixed_budget_s"] / row["rule_runtime"]["adaptive_s"]
                  for row in evasion_ablation["rows"] if "rule_runtime" in row]
        rule_runtime_speedup_mean = float(np.mean(ratios)) if ratios else None

    # --- LightGBM false-negative-floor diagnostic (M4 follow-up, scripts/09_lgb_diagnostic.py):
    # the LightGBM "ASR" of 0.0467 is n_target_predicted/n_botnet_test for a row set that is
    # IDENTICAL across every configuration -- the victim's pre-existing false-negative floor, not
    # attack success. See notes/20260716-decision-lgb-fnr-floor.md. ---
    lgb_diag = _load_optional("lgb_diagnostic.json")
    lgb_diagnostic = None
    if lgb_diag is not None:
        rows = lgb_diag["rows"]
        gain_shares = [r["trigger_gain_share"] for r in rows]
        leaf_overlaps = [r["leaf_overlap"] for r in rows]
        stamp_changed = [r["stamp_changed_predictions"] for r in rows]
        lgb_diagnostic = dict(
            n_target_predicted=rows[0]["n_target_predicted"],
            n_botnet_test=rows[0]["n_botnet_test"],
            fnr_floor_rate=rows[0]["n_target_predicted"] / rows[0]["n_botnet_test"],
            target_predicted_set_identical=lgb_diag["target_predicted_set_identical"],
            trigger_gain_share_mean=float(np.mean(gain_shares)),
            leaf_overlap_min=float(np.min(leaf_overlaps)),
            leaf_overlap_max=float(np.max(leaf_overlaps)),
            stamp_changed_predictions_mean=float(np.mean(stamp_changed)),
            mlp_contrast_stamp_changed=lgb_diag["mlp_contrast"]["stamp_changed_predictions"],
            mlp_contrast_asr=lgb_diag["mlp_contrast"]["asr"],
            mlp_contrast_n_botnet_test=lgb_diag["mlp_contrast"]["n_botnet_test"],
            note=lgb_diag["note"],
        )

    # --- Activation Clustering's real evasion mechanism (M4 follow-up,
    # scripts/11_ac_mechanism.py): AC has no ranked score or removal budget, so the reviewer's
    # adaptive-threshold suggestion does not apply to it. It fails a different way -- at low poison
    # rate its 2-means split never isolates the poison into the minority cluster. See
    # notes/20260716-decision-ac-proxy-not-independent.md. ---
    ac_mech = _load_optional("ac_mechanism.json")
    ac_mechanism = None
    if ac_mech is not None:
        rows = ac_mech["rows"]
        ev_rows = [r for r in rows if not r["is_control_cell"]]
        n_zero = sum(1 for r in ev_rows if r["minority_purity"] == 0.0)
        n_full = sum(1 for r in ev_rows if r["ac_recall"] == 1.0)
        n_near_zero = len(ev_rows) - n_zero - n_full
        sil = [r["silhouette"] for r in rows]
        rec = [r["ac_recall"] for r in rows]
        rho, pval = spearmanr(sil, rec)
        ac_mechanism = dict(
            n_evasion_cells=len(ev_rows),
            n_zero_purity_cells=n_zero,
            n_near_zero_purity_cells=n_near_zero,
            n_full_recovery_cells=n_full,
            evasion_minority_purity_mean=ac_mech["summary"]["evasion_cells"]["mean_minority_purity"],
            evasion_recall_mean=ac_mech["summary"]["evasion_cells"]["mean_ac_recall"],
            evasion_silhouette_mean=ac_mech["summary"]["evasion_cells"]["mean_silhouette"],
            control_minority_purity_mean=ac_mech["summary"]["control_cell"]["mean_minority_purity"],
            control_recall_mean=ac_mech["summary"]["control_cell"]["mean_ac_recall"],
            silhouette_recall_spearman=float(rho), silhouette_recall_spearman_pvalue=float(pval),
            note=ac_mech["note"],
        )

    # --- Neural Cleanse reconstructed-mask diagnostic (M4 follow-up, scripts/10_nc_masks.py):
    # qualitative mechanism check on the tabular detection drop -- NOT a causal explanation, the
    # drop itself remains unresolved (see analysis["neural_cleanse"] above). ---
    nc_masks = _load_optional("nc_masks.json")
    nc_mask_diagnostic = None
    if nc_masks is not None:
        nc_mask_diagnostic = dict(nc_masks["summary"], note=nc_masks["note"])

    analysis = dict(
        config=dict(seeds=seeds, anchor_cost=ANCHOR_COST, rate_cap=RATE_CAP, matched_rate=MATCHED_RATE,
                    anchor_cells=anchor_cells, matched_rate_cells=matched_cells,
                    bootstrap_resamples=config.BOOTSTRAP_RESAMPLES, bootstrap_ci=config.BOOTSTRAP_CI,
                    df=config.BOOTSTRAP_DF, no_parametric_tests_note=
                    "project protocol: n<=30, effect sizes + bootstrap CIs only, no t/Wilcoxon/ANOVA"),
        portability=dict(spectral=port_spec, ac=port_ac),
        portability_at_matched_rate=dict(
            spectral=matched_port_spec, ac=matched_port_ac,
            note=f"same paired comparison as portability, but the NIDS arm is the FIXED rate="
                 f"{MATCHED_RATE} (highest tested), cost={ANCHOR_COST} cell instead of the argmax-ASR "
                 "cell within rate<=rate_cap. Contrasts against the D3 anchor to show whether the "
                 "'portability failure' at the low-rate anchor is rate-driven, not a hard modality wall"),
        anchor_spectral_auc=dict(mean=anchor_spec_auc_mean, ci=anchor_spec_auc_ci,
            note="Spectral AUC at the D3 anchor cell (rate<=rate_cap, argmax-ASR, cost=anchor_cost). "
                 "AUC measures ranking quality independent of the recall metric's fixed removal budget "
                 "-- high AUC alongside low recall means the signal is present but the budget misses it"),
        evasion_window=evasion_window,
        adaptive_threshold=adaptive_threshold,
        neural_cleanse=dict(
            nids_detection_rate=nc_nids_rate, nids_detection_ci=nc_nids_ci,
            vision_detection_rate=nc_vis_rate, vision_detection_ci=nc_vis_ci,
            note="NC arms use different decision rules (MAD anomaly index at 10 vision classes vs "
                 "binary mask-ratio at 2 NIDS classes) -- reported as two rates, NOT a paired d_z"),
        h3=dict(
            spectral_imbalanced=bootstrap_ci(h3_spec_imb) if h3_spec_imb else (None, [None, None]),
            spectral_balanced=bootstrap_ci(h3_spec_bal) if h3_spec_bal else (None, [None, None]),
            ac_imbalanced=bootstrap_ci(h3_ac_imb) if h3_ac_imb else (None, [None, None]),
            ac_balanced=bootstrap_ci(h3_ac_bal) if h3_ac_bal else (None, [None, None]),
            note="does rebalancing training data rescue detector recall? if not, modality not "
                 "imbalance drives the miss"),
        h4=dict(lightgbm_no_activations=detectors["config"].get("h4_lightgbm_no_activations", True),
                note=detectors["config"].get("h4_note", "")),
        determinism=det_summary,
        rule_runtime_speedup_mean=rule_runtime_speedup_mean,
        lgb_diagnostic=lgb_diagnostic,
        ac_mechanism=ac_mechanism,
        nc_mask_diagnostic=nc_mask_diagnostic,
    )

    (config.RESULTS / "analysis.json").write_text(json.dumps(analysis, indent=2))

    # --- flat macros SSOT (M4-D6) ---
    def r3(x):
        return None if x is None else round(float(x), 4)

    macros = {
        "VisionSpectralRecallMean": r3(port_spec["vision_recall_mean"]),
        "VisionAcRecallMean": r3(port_ac["vision_recall_mean"]),
        "NidsSpectralRecallAtCost16Mean": r3(port_spec["nids_recall_mean"]),
        "NidsAcRecallAtCost16Mean": r3(port_ac["nids_recall_mean"]),
        "SpectralRecallDropMean": r3(port_spec["mean_recall_drop"]),
        "SpectralRecallDropCiLo": r3(port_spec["recall_drop_ci"][0]),
        "SpectralRecallDropCiHi": r3(port_spec["recall_drop_ci"][1]),
        "SpectralCohensDz": r3(port_spec["cohens_dz"]),
        "AcRecallDropMean": r3(port_ac["mean_recall_drop"]),
        "AcRecallDropCiLo": r3(port_ac["recall_drop_ci"][0]),
        "AcRecallDropCiHi": r3(port_ac["recall_drop_ci"][1]),
        "AcCohensDz": r3(port_ac["cohens_dz"]),
        "AnchorSpectralAucMean": r3(anchor_spec_auc_mean),
        "AnchorSpectralAucCiLo": r3(anchor_spec_auc_ci[0]),
        "AnchorSpectralAucCiHi": r3(anchor_spec_auc_ci[1]),
        "NidsSpectralRecallAtMatchedRateMean": r3(matched_port_spec["nids_recall_mean"]),
        "NidsAcRecallAtMatchedRateMean": r3(matched_port_ac["nids_recall_mean"]),
        "SpectralCohensDzAtMatchedRate": r3(matched_port_spec["cohens_dz"]),
        "AcCohensDzAtMatchedRate": r3(matched_port_ac["cohens_dz"]),
        "EvasionWindowCellCount": len(evasion_cells),
        "EvasionWindowSpectralRecallMean": r3(np.mean([c["mean_spectral_recall"] for c in evasion_cells])
                                              if evasion_cells else None),
        "EvasionWindowSpectralAucMean": r3(np.mean([c["mean_spectral_auc"] for c in evasion_cells
                                                     if c["mean_spectral_auc"] is not None])
                                           if any(c["mean_spectral_auc"] is not None for c in evasion_cells)
                                           else None),
        "AdaptiveEvasionWindowRecallMean": r3(adaptive_threshold["evasion_window_recall_mean"])
                                          if adaptive_threshold else None,
        "AdaptiveEvasionWindowFprMean": r3(adaptive_threshold["evasion_window_fpr_mean"])
                                       if adaptive_threshold else None,
        "AdaptiveEvasionWindowFprCiLo": r3(adaptive_threshold["evasion_window_fpr_ci"][0])
                                       if adaptive_threshold else None,
        "AdaptiveEvasionWindowFprCiHi": r3(adaptive_threshold["evasion_window_fpr_ci"][1])
                                       if adaptive_threshold else None,
        "AdaptiveControlCellRecallMean": r3(adaptive_threshold["control_cell_recall_mean"])
                                        if adaptive_threshold else None,
        "AdaptiveControlCellFprMean": r3(adaptive_threshold["control_cell_fpr_mean"])
                                     if adaptive_threshold else None,
        "NcNidsDetectionRate": r3(nc_nids_rate),
        "NcNidsDetectionCountFlagged": sum(1 for f in nc_nids_flags if f == 1.0),
        "NcNidsDetectionCountTotal": len(nc_nids_flags),
        "NcVisionDetectionRate": r3(nc_vis_rate),
        "RuleRuntimeSpeedupMean": r3(rule_runtime_speedup_mean),
        "LgbFnrFloorRate": r3(lgb_diagnostic["fnr_floor_rate"]) if lgb_diagnostic else None,
        "LgbNTargetPredicted": lgb_diagnostic["n_target_predicted"] if lgb_diagnostic else None,
        "LgbNBotnetTest": lgb_diagnostic["n_botnet_test"] if lgb_diagnostic else None,
        "LgbTargetPredictedSetIdentical": bool(lgb_diagnostic["target_predicted_set_identical"])
                                         if lgb_diagnostic else None,
        "LgbTriggerGainShareMean": r3(lgb_diagnostic["trigger_gain_share_mean"]) if lgb_diagnostic else None,
        "LgbLeafOverlapMin": r3(lgb_diagnostic["leaf_overlap_min"]) if lgb_diagnostic else None,
        "LgbLeafOverlapMax": r3(lgb_diagnostic["leaf_overlap_max"]) if lgb_diagnostic else None,
        "LgbMlpContrastStampChanged": lgb_diagnostic["mlp_contrast_stamp_changed"] if lgb_diagnostic else None,
        "LgbMlpContrastAsr": r3(lgb_diagnostic["mlp_contrast_asr"]) if lgb_diagnostic else None,
        "AcEvasionCellCount": ac_mechanism["n_evasion_cells"] if ac_mechanism else None,
        "AcEvasionZeroPurityCellCount": ac_mechanism["n_zero_purity_cells"] if ac_mechanism else None,
        "AcEvasionNearZeroPurityCellCount": ac_mechanism["n_near_zero_purity_cells"] if ac_mechanism else None,
        "AcEvasionFullRecoveryCellCount": ac_mechanism["n_full_recovery_cells"] if ac_mechanism else None,
        "AcEvasionMinorityPurityMean": r3(ac_mechanism["evasion_minority_purity_mean"]) if ac_mechanism else None,
        "AcEvasionRecallMean": r3(ac_mechanism["evasion_recall_mean"]) if ac_mechanism else None,
        "AcEvasionSilhouetteMean": r3(ac_mechanism["evasion_silhouette_mean"]) if ac_mechanism else None,
        "AcControlMinorityPurityMean": r3(ac_mechanism["control_minority_purity_mean"]) if ac_mechanism else None,
        "AcControlRecallMean": r3(ac_mechanism["control_recall_mean"]) if ac_mechanism else None,
        "AcSilhouetteRecallSpearman": r3(ac_mechanism["silhouette_recall_spearman"]) if ac_mechanism else None,
        "NcVisionFlaggedConcentrationMean": r3(nc_mask_diagnostic["vision_flagged_concentration_mean"])
                                           if nc_mask_diagnostic else None,
        "NcTabularFlaggedConcentrationMean": r3(nc_mask_diagnostic["tabular_flagged_concentration_mean"])
                                            if nc_mask_diagnostic else None,
        "NcTabularTriggerFeatureMaskShareMean": r3(nc_mask_diagnostic["tabular_trigger_feature_mask_share_mean"])
                                               if nc_mask_diagnostic else None,
        "NcTriggerFeatureShareChanceLevel": r3(nc_mask_diagnostic["trigger_feature_share_chance_level"])
                                           if nc_mask_diagnostic else None,
        "NcTabularFlaggedMaskConstantAllSeeds": bool(nc_mask_diagnostic["tabular_flagged_mask_constant_all_seeds"])
                                               if nc_mask_diagnostic else None,
        "H3SpectralImbalancedMean": r3(analysis["h3"]["spectral_imbalanced"][0]),
        "H3SpectralBalancedMean": r3(analysis["h3"]["spectral_balanced"][0]),
        "H3AcImbalancedMean": r3(analysis["h3"]["ac_imbalanced"][0]),
        "H3AcBalancedMean": r3(analysis["h3"]["ac_balanced"][0]),
        "H4LightgbmNoActivations": bool(analysis["h4"]["lightgbm_no_activations"]),
        "DeterminismCellsChecked": det_summary.get("n_determinism_checked"),
        "DeterminismCellsFailed": det_summary.get("n_determinism_failed"),
    }
    raw = json.dumps(macros, indent=2)
    assert "Infinity" not in raw and "NaN" not in raw, "non-strict JSON token in macros"
    (config.RESULTS / "macros.json").write_text(raw)

    print("wrote results/analysis.json and results/macros.json")
    print(json.dumps(macros, indent=2))


if __name__ == "__main__":
    main()
