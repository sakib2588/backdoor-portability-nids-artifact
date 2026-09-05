# Provenance of every reported quantity

Each number in the manuscript is read from a committed result file rather than
recomputed for the paper. This index maps each reported quantity to the file in
`results/` that produces it. It stood as Table 13 of the appendix until it was moved
here to reclaim page area; its content is unchanged.

One class of exception is stated in the appendix itself: the 30-seed UNSW-NB15 figures
merge three seed arms, and where a merged mean is quoted it is computed across the
files named for that row rather than stored as a single field.

54 rows.

| Reported quantity | Source file (`results/`) |
|---|---|
| Evasion and backdoor components | `clean_model_stamped_asr` |
| Secondary components | `secondary_clean_model_stamped_asr` |
| Random-feature control | `random_feature_evasion_control` |
| Projection displacement | `projection_displacement_diagnostic` |
| Surrogate selection | `surrogate_shap` |
| Footprint-matched control | `fraction_matched_trigger_control` |
| Main detector grid | `detectors, poison_sweep` |
| Full 25-cell MAD grid | `full_grid_mad_sweep` |
| Budget multiplier sweep | `spectral_budget_multiplier_sweep` |
| Spectral k ablation | `spectral_k1_ablation` |
| Univariate baseline | `univariate_zfilter_baseline` |
| SPECTRE | `spectre_window, spectre_vision_control` |
| STRIP | `strip_detector, strip_vision_control` |
| AC repair grid | `ac_repair_grid, ac_repair_summary` |
| AC gated score | `ac_continuous_score_pilot` |
| NC repair candidates | `nc_repair_calibration` |
| NC mask geometry | `nc_mask_geometry_pilot` |
| NC scalar residue | `nc_normpair_mining` |
| Vision controls | `vision_control, tabular_positive_control` |
| Secondary grid | `secondary_detectors, secondary_poison_sweep` |
| Secondary MAD threshold | `secondary_adaptive_threshold, _replication, _batch3` |
| Merged 30-seed UNSW column | `secondary_merged_column_n30` |
| Post-removal retraining | `secondary_post_removal_asr` |
| Adaptive attacker | `adaptive_attacker` |
| Imbalance confound | `detectors (H3 block), h3_window_auc` |
| Clean and always-benign accuracy | `detectors (clean_acc), clean_mlp_test_recall` |
| LightGBM null | `lgb_diagnostic` |
| STRIP and SPECTRE tabular control | `positive_control_strip_spectre` |
| UNSW-NB15 tabular control | `secondary_positive_control` |
| SPECTRE on UNSW-NB15 | `spectre_secondary` |
| CTU-13 replication seeds | `ctu_replication_seeds` |
| SPECTRE and Spectral independence | `spectre_spectral_independence` |
| Corpus-family sweep | `netflow_poison_sweep, netflow_data_gate` |
| Corpus-family benign-share test | `netflow_property_analysis` |
| Projection clipping, post-hoc | `netflow_clipping_diagnostic` |
| Corpus-family detectors | `netflow_detectors, netflow_detector_analysis` |
| Activation Clustering scale control | `ac_subsampling_control` |
| Displacement axis | `displacement_axis` |
| AC reduction gate | `ac_ica_vision_control` |
| Density-mitigation comparator | `density_mitigation_ctu, _control_bypass.composition` |
| Aggregated reported values | `review_response_aggregates` |
| Matched-rate vision sweep | `vision_control_rate_sweep` |
| Rule-selection statistic | `rule_selection_statistic` |
| Isolation-filter gate | `isolation_forest_window` |
| Constraint-departure gate | `boundary_departure_window` |
| Per-seed cost-8 decomposition | `cost_transition_sweep` |
| Flagged-row medians | `spectre_spectral_independence (cells[].n_mad_flagged)` |
| Rate-cap coverage | `netflow_data_gate (sum over samples[].rate_capped_values)` |
| UNSW multiplier sweep | `secondary_budget_multiplier_sweep` |
| UNSW imbalance test | `secondary_h3_window_auc` |
| STRIP published rule and bound | `strip_mad_bound` |
| Rule-capability check | `rule_capability_taxonomy` |
| Active-paths control | `active_paths_window` |
| Specificity attribution probe | `specificity_shap_comparison` |
