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

## Post-hoc observations on the corpus family

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
