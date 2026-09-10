# Supplementary material

Sections the manuscript defers to rather than prints. Each is supporting detail for a claim the
paper states in full; none of it is load-bearing evidence that the paper relies on without
saying so, and each section below is one the paper itself marks as post-hoc, as not licensing a
claim, or as a control rather than a result.

The appendix names this file where each of these was previously printed.

## SPECTRE and Spectral Signatures are not one detector

Both reach recall 1.0000 under the budget-free rule on every window cell, which is
a ceiling on that metric. Scored against each other on the same activations, their
per-sample orderings have a median Spearman correlation of -0.1131 over the 20
window cells, against the 0.90 redundancy bar this paper applies to any
constructed score, and their flagged sets overlap at a median Jaccard index of
0.2999. SPECTRE flags a median of 25,460 rows against Spectral Signatures'
8,149 to reach the same recall.

## The bimodality coefficient reads the detector, not the rule

The bimodality coefficient, Sarle's b = (g_1^2 + 1)/g_2 with g_1 the skewness
and g_2 the non-excess kurtosis of the raw scores [Pfister2013bimodality],
contains no threshold from either rule and needs no poison count. We compute the
large-sample form, whose correction term is negligible at the class sizes here.
Pooled over 625 arm-cell-seed units spanning the full grid, it ranks which rule
recovers more poison at a prediction AUC of 0.8196, against 0.7138 for the granted
poison fraction. Stratified by arm, that ordering reverses. Spectral Signatures
prefers the budget-free rule on 80 of its units and the budget on 45, and there
the statistic ranks at 0.5392 against the poison rate's 0.9861; the
isolation-based filter gives 0.5156 against 0.9673 and the boundary-departure
statistic 0.6759 against 0.8526.

We state what this comparison does and does not establish. Five arms are scored:
Spectral Signatures, SPECTRE, STRIP, an isolation-based filter, and a
boundary-departure statistic. Activation Clustering and Neural Cleanse are
excluded because neither emits a ranked score for a moment statistic to read, and
the isolation-based filter and boundary-departure statistic failed their own
pre-registered gates, entering as score sources rather than working defenses.
SPECTRE prefers the budget-free rule on all 125 of its units and STRIP the budget
on all 125 of its own, so neither admits a within-arm comparison. Three arms do
vary, and in each the poison rate outranks the score shape. The shape difference
between arms is measured and remains, and still accounts for STRIP being the arm
the budget-free rule makes worse.

## Where the multiplier crosses half recall, and what it costs elsewhere

The 1,022 clean flows outranking the typical poison sample do not by themselves
locate the half-recall crossing, because the poison rows ahead of the median one
consume the same budget. Adding the 286 poison rows that outrank it places the
median poison sample at rank 1,308, between the multiplier-2 and multiplier-3
budgets, where the measured recall does cross half. At 2 held-out cells the
multiplier of 5 that buys full recall at the anchor cell costs 8.37% false
positives at rate 0.02 and 21.60% at 0.05 for no recall gain, since 1.5 already
recovers 0.94 and 1.0 there and the budget-free rule reaches the same recall at
5.20% and 4.67%.

## Post-hoc observations on the dataset family

One post-hoc observation, marked as such, points where the pre-registered
variable did not. The number of the 16 stamped features that constraint
projection clips separates the 2 modes within each corpus, at  =
-0.367 (p = 0.046) on NF-CSE-CIC-IDS2018-v2 and  = -0.506 (p =
0.005) on NF-ToN-IoT-v2, ranking the backdoor mode at AUC 0.764 and 0.750.
Heavily clipped watermarks are milder, and the clean model still calls them
attack, so the poisoning has to teach the shortcut. We do not pool the 2
corpora, because a pooled figure is inflated by a between-corpus difference.
Benign-class effective dimensionality, the participation ratio of the
rank-correlation spectrum, does not move across the manipulated levels,
holding between 3.00 and 3.02 on one corpus and between 5.56 and 5.62 on the
other, so the subsampling moved class size without moving class diversity.

## A feature-tokenizer transformer victim is bimodal across seeds

A feature-tokenizer transformer victim was tested directly at the anchor cell
under the same projected trigger. Across 10 seeds the outcome is bimodal. Four
seeds reach attack success at or above 0.9951 while the remaining 6 stay at or
below 0.0737, with no run in between. Victim clean accuracy holds between 0.9965
and 0.9996 across both groups, so the low mode is not an undertrained victim.
Which outcome a run reaches is decided by the training seed rather than by
architecture capacity alone. Ten seeds establish that both outcomes occur. They
do not fix the rate at which each occurs, since a 4-of-10 split carries a 95%
interval from 17% to 69%, so we report no rate.

## A score-aware adaptive attacker does not evade the threshold

A score-aware adaptive attacker searching a pre-registered 64-candidate budget on
a disjoint-seed surrogate keeps attack success at 1.0 while the budget-free
threshold still flags it at recall 1.0 and 6.52% false positives. A clean-model
control on that frozen trigger shows it too is largely evasion, which is the same
diagnostic applied to the adaptive case and the same conclusion.

This arm is the thinnest in the paper and we do not lean on it. The winning
trigger is evaluated on 2 seeds rather than 5, so it is an existence check
that this particular search does not evade the threshold, not an estimate of how
often an adaptive attacker would fail. A negative result at n=2 bounds very
little, and the search space is one we chose.

## Computational cost

The study is close to CPU-only. Victim training, TreeSHAP selection, the
constraint projection, the Spectral Signatures singular-value decomposition,
SPECTRE's robust covariance estimate, the Activation Clustering mixture fits, and
STRIP's blending all run on CPU. A single consumer GPU is used for the vision
control and for Neural Cleanse's inversion. Its mask-geometry replica study is
the most expensive single component, at 23.4 seconds per clean replica over 102
replicas. The Activation Clustering repair pilot runs at 128.7 seconds per cell.

## Every cell of the CTU-13 grid

Five seeds per cell, standard deviations in parentheses. MAD columns use the
budget-free rule at z=3.0. AUC is Spectral Signatures' threshold-free ranking
metric. The final column marks the 4 budget-starvation window cells. Source:
`results/full_grid_mad_sweep.json`.

| Rate | Cost | Attack success | Fixed recall | MAD recall | MAD FPR | AUC | AC fixed recall | Window |
|---|---|---|---|---|---|---|---|---|
| 0.5% | 1 | 0.0403 (0.0081) | 0.0087 (0.0033) | 0.0678 | 7.02% | 0.4967 | 0.0007 |  |
| 0.5% | 2 | 0.0423 (0.0097) | 0.0084 (0.0029) | 0.0696 | 6.95% | 0.4970 | 0.0007 |  |
| 0.5% | 4 | 0.0423 (0.0077) | 0.0091 (0.0015) | 0.0699 | 6.99% | 0.4968 | 0.0007 |  |
| 0.5% | 8 | 0.8924 (0.1722) | 0.0203 (0.0079) | 1.0000 | 6.75% | 0.9757 | 0.0007 | yes |
| 0.5% | 16 | 1.0000 (0.0000) | 0.0629 (0.0190) | 1.0000 | 6.70% | 0.9908 | 0.0000 | yes |
| 1% | 1 | 0.0423 (0.0084) | 0.0189 (0.0053) | 0.0705 | 7.04% | 0.5013 | 0.0014 |  |
| 1% | 2 | 0.0452 (0.0073) | 0.0201 (0.0063) | 0.0682 | 7.04% | 0.5017 | 0.0007 |  |
| 1% | 4 | 0.0413 (0.0082) | 0.0210 (0.0048) | 0.0727 | 7.08% | 0.5029 | 0.0005 |  |
| 1% | 8 | 0.8698 (0.2394) | 0.0376 (0.0086) | 1.0000 | 6.66% | 0.9743 | 0.0009 | yes |
| 1% | 16 | 1.0000 (0.0000) | 0.4493 (0.1343) | 1.0000 | 6.22% | 0.9894 | 0.4003 | yes |
| 2% | 1 | 0.0403 (0.0083) | 0.0334 (0.0039) | 0.0740 | 6.88% | 0.5043 | 0.0009 |  |
| 2% | 2 | 0.0432 (0.0086) | 0.0348 (0.0044) | 0.0761 | 7.00% | 0.5064 | 0.0008 |  |
| 2% | 4 | 0.0398 (0.0075) | 0.0368 (0.0047) | 0.0799 | 7.09% | 0.5062 | 0.0010 |  |
| 2% | 8 | 0.6319 (0.4781) | 0.2449 (0.1851) | 0.9999 | 6.29% | 0.9719 | 0.0011 |  |
| 2% | 16 | 1.0000 (0.0000) | 0.9407 (0.0949) | 1.0000 | 5.20% | 0.9898 | 1.0000 |  |
| 5% | 1 | 0.0398 (0.0059) | 0.0789 (0.0033) | 0.0728 | 7.11% | 0.5027 | 0.0007 |  |
| 5% | 2 | 0.0413 (0.0077) | 0.0805 (0.0033) | 0.0725 | 6.99% | 0.5035 | 0.0009 |  |
| 5% | 4 | 0.0423 (0.0080) | 0.0843 (0.0039) | 0.0774 | 6.98% | 0.5042 | 0.0009 |  |
| 5% | 8 | 0.9351 (0.0699) | 0.8829 (0.0768) | 0.9999 | 5.47% | 0.9700 | 0.0006 |  |
| 5% | 16 | 1.0000 (0.0000) | 1.0000 (0.0000) | 1.0000 | 4.67% | 0.9882 | 1.0000 |  |
| 10% | 1 | 0.0403 (0.0071) | 0.1542 (0.0042) | 0.0708 | 7.00% | 0.5028 | 0.0006 |  |
| 10% | 2 | 0.0418 (0.0090) | 0.1556 (0.0040) | 0.0713 | 6.95% | 0.5037 | 0.0007 |  |
| 10% | 4 | 0.0442 (0.0069) | 0.1594 (0.0029) | 0.0754 | 6.98% | 0.5041 | 0.0006 |  |
| 10% | 8 | 0.6369 (0.3583) | 0.9996 (0.0007) | 0.9993 | 4.44% | 0.9683 | 0.2050 |  |
| 10% | 16 | 1.0000 (0.0000) | 1.0000 (0.0000) | 1.0000 | 3.44% | 0.9863 | 1.0000 |  |

## Four families of Neural Cleanse decision statistic

Sources: `results/nc_repair_calibration.json`, `results/nc_normpair_mining.json`,
`results/nc_mask_geometry_pilot.json`.

The two-class degeneracy of the anomaly index is not a sufficient explanation either.

Calibrating instead against 102 clean-trained replicas removes class count from the decision rule altogether and does not change the verdict. Eight mask statistics, covering weight concentration, alignment with the model's own SHAP ranking, and stability across restarts, leave all 3 poisoned models inside the clean null, and a mining pass over the inversion's scalar residues clears no permutation bar. Three independent families of decision statistic return the same null while the detector succeeds on a matched vision control. We do not resolve why the ratio fails.

| Family | Statistic | Outcome |
|---|---|---|
| Decision rule | Published two-class mask ratio, recalibrated | Detection gain 0.0000 across 3 candidates |
| Scalar residue | 16 mined fields of the inversion output | Best AUC 0.6291 against a permutation bar of 0.8186, p = 0.91 |
| Mask geometry | Concentration, SHAP alignment, restart stability, 8 statistics | All 3 victims inside a 102-model clean null, p 0.41 to 0.69 |
| Structural | Gradient path on tree ensembles | Undefined; no inversion exists to score |

## The displacement axis

Source: `results/displacement_axis.json`.

A third axis, the displacement of the trigger itself, was swept during adaptive-attacker candidate selection. Over a sixfold range at poison rate 0.005 and trigger cost 16, attack success climbs to saturation and Spectral Signatures' ranking AUC climbs monotonically with it, Spearman = 1.000, so the poison becomes easier to rank rather than harder. The inherited budget stays starved across that range while the budget-free rule reaches full recall from twice the smallest displacement onward, so the window is not an artifact of a quiet trigger. These rows come from a different code path on 3 surrogate seeds and support a within-sweep trend rather than a cell-for-cell comparison with the grid.

| Displ. | Attack success | Fixed recall | Budget-free recall | AUC |
|---|---|---|---|---|
| 1 | 0.060 | 0.0082 | 0.7063 | 0.9212 |
| 2 | 0.676 | 0.0099 | 1.0000 | 0.9691 |
| 3 | 0.957 | 0.0192 | 1.0000 | 0.9768 |
| 4 | 1.000 | 0.0303 | 1.0000 | 0.9845 |
| 5 | 1.000 | 0.0373 | 1.0000 | 0.9881 |
| 6 | 1.000 | 0.0746 | 1.0000 | 0.9915 |

Aggregated from the adaptive-attacker selection stage, a different code path from
the main grid. The ledger is not a full factorial above the smallest displacement,
and absent cells are reported absent rather than filled.

## Recalibrating the removal-budget constant

Source: `results/spectral_budget_multiplier_sweep.json`.

The sweep runs from the inherited multiplier of 1.5 to the value that buys full recall.

The half-recall crossing falls between the multiplier-2 and multiplier-3 budgets, and at held-out cells a multiplier that buys full recall at the anchor cell costs far more for no recall gain, which the section above records. The constant is regime-specific, and checking whether one sufficed still requires labels.

| Multiplier | Budget (rows) | Mean recall (SD) | FPR |
|---|---|---|---|
| 1.5 | 858 | 0.0629 (0.0190) | 0.74% |
| 2 | 1,144 | 0.2762 (0.1239) | 0.89% |
| 3 | 1,716 | 0.9203 (0.1685) | 1.07% |
| 5 | 2,860 | 1.0000 (0.0000) | 2.06% |

Recall rises with the removal budget, and so does the false-positive cost of
granting it. Recall is far more dispersed across seeds at the intermediate budgets
than at either end.

## The Spectral Signatures vision-control k sweep

Source: `results/spectral_vision_k_sweep.json`.

Tran et al. compute the spectral signature from the top singular direction alone.
This study uses the top five. The sweep scores both, and two intermediate
settings, on the matched MNIST vision control at a 5% poison rate over the five
base seeds, against the control's 0.90 per-seed admission bar.

| `n_components` | mean recall | worst seed | clears 0.90 on every seed |
|---|---|---|---|
| 1 | 0.8936 | 0.8517 | no |
| 3 | 0.9348 | 0.9068 | yes |
| 5 | 0.9828 | 0.9782 | yes |
| 10 | 0.9947 | 0.9893 | yes |

Per-seed recalls, in seed order 42, 123, 456, 789, 1337: k=1 gives 0.9279,
0.8713, 0.9179, 0.8994, 0.8517; k=3 gives 0.9316, 0.9331, 0.9604, 0.9419,
0.9068; k=5 gives 0.9786, 0.9834, 0.9900, 0.9841, 0.9782; k=10 gives 0.9948,
0.9970, 0.9967, 0.9959, 0.9893.

The published k=1 is the only setting that misses the bar, and it misses it on
its worst seed rather than on the mean. Three settings clear it, so no criterion
internal to this control picks 5 over 3 or 10. The choice of 5 was fixed before
the tabular runs and not revisited.

## The Spectral Signatures k=1 window ablation

Source: `results/spectral_k1_ablation.json`.

The setting changed on the vision control is re-scored on the measurement this
paper reports, so that the deviation cannot be flattering the tabular result. The
ablation runs on the four CTU-13 window cells, rates 0.005 and 0.01 at trigger
costs 8 and 16, over the five base seeds. Both settings are scored on the
identical trained model and poisoned population per row, so only k differs. The
removal budget is given the true poison fraction as an oracle.

| setting | mean fixed-budget recall | mean ranking AUC |
|---|---|---|
| k=1, as published by Tran et al. | 0.1527 | 0.9839 |
| k=5, as used throughout this paper | 0.1454 | 0.9836 |

The published setting recovers slightly more poison in the window than the one
this paper uses, and ranks marginally better. The deviation therefore helps the
detector on the vision control and costs it a little on the tabular measurement,
which is the direction that argues against the choice having been made to favor
the reported result. Both figures come from this ablation's own retraining, so
they are close to but not identical with the 0.1425 window mean the main grid
reports.

## Spectral Signatures' removal-budget denominators

The fixed removal budget and the poison-fraction oracle it is given are computed
over different row counts, and the paper states that both differ without
printing either. A grid rate is a fraction of the 114,436-row training block
handed to the poisoner, so rate 0.005 plants 572 rows. The budget's oracle
poison fraction is that same count divided by the target class of 111,666
rows, which is 0.00512.

## Activation Clustering's independent-component substitution, per seed

Under an identical pipeline, independent-component reduction fails this
study's own positive-control gate on two of five seeds, at poison recall
0.4996 and 0.5717. Refitting on the same activation matrix with only the
solver's random state changed moves recall by as much as 0.4922, and no fit
of the 30 warned of non-convergence, so the instability is local optima
rather than a solver that fails to settle. The principal-component reduction
this paper uses instead clears the gate on all five seeds of the same
comparison, at a minimum of 0.9778, and is the reduction under which every
Activation Clustering number in the paper is computed.

## STRIP's blend parameters

STRIP blends each scored row, at ratio 0.5 in standardized space rather than
the original's pixel range, with 64 clean reference rows.

## Manipulating benign share within one dataset

Comparing datasets at their native balance is observational, because benign
share co-varies with capture year, topology, synthetic against real traffic,
and attack families. Six benign-share levels, {0.05, 0.20, 0.40, 0.60, 0.80,
0.96}, are built by subsampling one dataset at a fixed 1,200,000 rows,
holding topology, capture year, attack families, feature space, and sample
size constant so that only the ratio moves. NF-CSE-CIC-IDS2018-v2 hosts the
manipulation and NF-ToN-IoT-v2 replicates it, the only two datasets holding
enough rows of both classes to reach every level.

The hypothesis, the primary statistic and both guards were fixed before any
cell ran. The outcome is the backdoor fraction, poisoned-model attack success
rate minus the same trigger's attack success against the clean model, and the
statistic is a Spearman rank correlation between benign share and mean
backdoor fraction over the six levels at trigger cost 16, referred to an
exact permutation null. Confirmation requires |rho| >= 0.8 on the manipulated
dataset and the same sign on the replication dataset. Refutation requires
|rho| < 0.4. A cell whose clean model already accepts at least 0.95 of
watermarked flows is marked saturated and excluded, because it cannot show a
positive backdoor fraction and would otherwise be scored as support. A cell
whose poisoned attack success falls below the effectiveness bar of 0.8 is
marked ineffective and its detector numbers reported as uninterpretable
rather than as detector nulls.

NF-BoT-IoT-v2 carries no cell the attack can fund: a 1.2 million-row sample
carries 3,433 benign training rows, a 279:1 ratio against the attack class
under inverse-frequency weighting, fewer than the lowest configured poison
rate needs. Every one of its cells is infeasible for a clean-label attacker,
which is why it contributes no cell to the detector-cost figures reported
elsewhere, and is reported as that outcome rather than as a dataset dropped.
