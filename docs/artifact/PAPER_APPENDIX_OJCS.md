# OJ-CS appendix, moved wholesale

The OJ-CS retarget (12-page limit, no appendix in the submitted manuscript)
moved the paper's entire appendix here rather than printing a trimmed version.
This is the full content that previously lived under `\appendices` in
`paper_ojcs/sections/07_appendix.tex`; every `Appendix~\ref{app:...}` in
`paper_ojcs/sections/*.tex` now points here by section name instead.

This file is specific to the OJ-CS manuscript. `paper_access/` (the frozen
IEEE Access fallback) keeps its own printed appendix and continues to defer
to `docs/artifact/SUPPLEMENTARY.md` as before -- this file does not change
that relationship.

---

## Provenance of every reported quantity
*(originally `\label{app:provenance}`)*

Each number in this paper is read from a committed result file rather than
recomputed for the manuscript, with one class of exception. The 30-seed
UNSW-NB15 figures merge three seed arms, and where a merged mean is quoted it is
computed across the three files named for that row rather than stored as a single
field. A 61-row index maps each claim to the file that produces it. All 125 CTU-13
detector-sweep cell-seed runs re-score to their committed values on fixed
hardware, as do 40 UNSW-NB15 detector runs and 35 UNSW-NB15 threshold runs. The
dataset-family arm carries no such record, and we do not claim one for it. Each long-running script writes a resumable checkpoint keyed by every input that changes its result.

The code, the result files, and that index are released at
\url{https://github.com/sakib2588/backdoor-portability-nids-artifact}, at commit
`a3f53d9aa4fc`. The index is `docs/artifact/PROVENANCE.md`, and each
of its rows names a file in the `results/` directory of that repository.
The export refuses to publish if any row names a file it does not contain, so a
row cannot point at a result file that no longer exists. That check is over file
presence, not value equality: it does not itself compare a manuscript literal
against the field it is claimed from. A separate pair of scripts does that
comparison directly, tracing each literal to a stored scalar and, where a value is
computed rather than stored outright, recomputing it from the named file. The raw
datasets are not redistributed there, and both carry their own terms. False-positive
rates throughout are false positives over the clean rows remaining in the target
class, which is the class size less the planted rows.

The UNSW-NB15 dataset of record is Moustafa's IEEE DataPort
release [Moustafa2019unswnb15data]. What this study processes is the
redistribution at \url{https://huggingface.co/datasets/Mouwiya/UNSW-NB15}, whose
2{,}280{,}090 rows and per-shard SHA-256 digests are recorded alongside the split,
because row counts differ across redistributions of this dataset and the split
would not otherwise reproduce. Citing the digest rather than the release lets a
reader confirm byte-identity with the artifact these numbers were computed on. That release contains 695{,}831 exact duplicate rows, a well-documented
artifact of flow aggregation. Dropping them leaves 1{,}584{,}259 records, cut 80/20
by flow start time into 1{,}267{,}407 training and 316{,}852 test rows. The test
block holds 26{,}166 attack rows, 8.26% of it. UNSW-NB15's attack categories run
from 16{,}949 Exploits rows to 106 Worms rows.

## Per-configuration results
Every cell of the CTU-13 grid is published with the artifact rather than printed
here (artifact repository, `SUPPLEMENTARY.md`, every cell of the CTU-13 grid), so that
the rate dimension marginalized away in Table the full CTU-13 grid table (main paper) remains available
for inspection. Two features of the grid are visible only at this resolution. Attack
success at cost 8 ranges from 0.6319 to 0.9351 across rates, so the cost-8 mean
of 0.7932 sits below the 0.8 criterion while three of its five cells sit above
it, at rates 0.5%, 1% and 5%.
Fixed-budget recall at cost 8 spans 0.0203 to 0.9996, a range wider than any
summary row can express.

## Material published with the artifact
Fourteen supporting sections are published with the artifact rather than printed
here. Thirteen are in `docs/artifact/SUPPLEMENTARY.md` of the repository
named above and the fourteenth, the per-cell dataset-family taxonomy with its
silhouette and scale evidence, is in `docs/artifact/NETFLOW_FAMILY.md`.
They are the SPECTRE and Spectral Signatures redundancy control, the
bimodality-coefficient rule-selection study, where the recalibrated multiplier
crosses half recall, the removal-budget multiplier sweep, the post-hoc
observations on the dataset family, the feature-tokenizer transformer victim, the
score-aware adaptive attacker, the computational cost of every experiment, every
cell of the CTU-13 grid, the four families of Neural Cleanse decision statistic,
the displacement axis, the Spectral Signatures vision-control $k$ sweep, and the
Spectral Signatures $k=1$ window ablation. What divides the two is enumeration depth rather than
importance. Every claim this paper makes is stated here, together with the
summary, the marginal or the control verdict a reader needs in order to check it,
and the exhaustive per-unit listing behind each one is published with the
artifact. The controls that bound the central inversion claim, the
imbalance test, the STRIP entropy bound, the Activation Clustering repair and the
dataset-family detector results are printed here rather than deferred.

## Pre-registration status
*(originally `\label{app:prereg}`)*

Three quantities central to the headline result were fixed before the runs that
produced them. They are the three MAD thresholds $\{2.0, 2.5, 3.0\}$, the
five-seed and replication-seed sets, and the four gates applied to the Activation
Clustering repair. Further quantities were also fixed in advance and are declared
where they are used rather than repeated here. They are the benign-share
hypothesis with its confirmation and refutation bars and its two guards, the gates
on the isolation-based filter and the constraint-departure statistic, the poison
recall bar on the density-mitigation comparator, and the adaptive attacker's
candidate budget. Six departures from the pre-registration are declared here.
Two are quantities that were not fixed in advance, and both are flagged wherever
they appear. The third is a rule that was fixed and then relaxed. The
pre-registration for the active-paths comparator stated that a detector failing
its control does not enter the paper. It is now reported in the lower block of
Table the window-cells detector table (main paper) instead, because two of the ported detectors failed the same
control and excluding only the comparator applied one rule two ways. Relaxing the
rule after seeing the result is the weaker of the two corrections available, and we
prefer it to a comparison that flatters us. The budget-starvation criterion, mean attack
success at or above 0.8 with fixed-budget recall below 0.5, was chosen by
inspecting the computed grid. Both cuts fall inside gaps in that grid rather than
through a crowded region. Per-cell mean attack success runs to 0.637 and resumes
at 0.870, and per-cell mean fixed-budget recall runs to 0.449 and resumes at
0.883, so any pair of cuts inside those two gaps selects the same four cells. The
thresholds are round numbers within the gaps, not tuned values. The direction of the Activation Clustering
atypicality score was reversed after a seed-42 probe. The fourth is a sample size
that grew after we saw a result. The pre-registration fixed five base seeds and five
replication seeds for the UNSW-NB15 window. When the replication arm disagreed
with the base arm we added a third block, the contiguous seeds 501 to 520, and
report the three cells at $n=30$ merged. The block was fixed and drawn without
inspecting any result from it, but the decision to run it was made after seeing
the disagreement, so it is a results-contingent increase and we report it as one.
Both arms are printed separately in Table the UNSW-NB15 window-cells table (main paper) so a reader can
see the disagreement rather than only the merge. The fifth and sixth are two
further quantities that were not fixed in advance and are disclosed at their
point of use rather than only here: the 0.90 vision-control bar, set while
building that control, and Spectral Signatures' top-$k$ setting, chosen on that
control rather than inherited. We count them here so this enumeration is the
complete one. Every cell of the grid is published with the
artifact and the disjoint replication column is printed in
Table the UNSW-NB15 window-cells table (main paper), so that a reader can assess both without relying on
our characterization of them. Assessing the criterion against the full grid
therefore means opening the artifact. Eight of the 25 cells meet the attack-success half
of the criterion, and the four named as the window are those that also starve the
fixed budget.

## Supporting diagnostics
*(originally `\label{app:diagnostics}`)*

**Why a tree ensemble admits no Neural Cleanse inversion.** A tree
ensemble's decision function is piecewise constant in its input, so the
cross-entropy term in Neural Cleanse's inversion objective contributes no
gradient. A differentiable shim does not help, because the threshold
comparison is where the graph is severed, and forcing the graph closed by a
relaxation would leave only the L1 mask penalty in the objective. That drives
every mask toward zero, producing identical norms across classes, an arbitrary
tie, and an anomaly index over a constant vector. Such a relaxation would
measure itself rather than the victim, which is why we do not run one.

**The wrong-cluster failure mode, and the subsample control.** Both are
established with the dataset-family detector results below. The cell counts are
printed there. The silhouette summary for the wrong-cluster cells is printed there. The per-cell
silhouettes and the full-scale re-run that clears the subsample cap are published
with the artifact.

## Four controls bound what explains the split
*(originally `\label{app:controls}`)*

Four controls bound what explains the cross-dataset split. Each is stated below with the measurement it rests on and the alternative it removes.

**The effect follows the ranking, not the perturbation.** Sixteen random
features at the same magnitude and under the same projection give a clean-victim
rate of 0.0584 (SD 0.0259) on CTU-13 against the ranked trigger's 1.0000.
UNSW-NB15 random draws give 0.0018 (SD 0.0060), so that victim resists any
sixteen-feature stamp rather than resisting this particular direction.

**Trigger size does not explain it.** The projection leaves comparable
perturbations on both datasets, 4.872 standard deviations against 4.789, so the
two attacks are of the same magnitude in feature space.

**White-box access is not load-bearing for evasion on CTU-13, but architecture is.** An MLP
surrogate trained on the same data at a different seed reproduces the evasion at
0.9577 and recovers 0.89 of the top-sixteen feature set. A LightGBM surrogate
reaches only 0.3179 with no overlap in the top 8. The attacker needs the
victim's architecture, not the victim.

**Enlarging the feature footprint does not leave the evasion regime.** CTU-13's cost-16
trigger stamps 2.1% of its 757 features against UNSW-NB15's 42.1% of 38, a
twenty-fold gap that could itself produce the split. Scaling CTU-13's cost to
$\{20, 40, 80, 159, 319\}$ takes it from 2.6% up to the same 42.1% of the
record that UNSW-NB15's cost-16 trigger occupies, and CTU-13
stays fully evasion-dominant throughout, evasion component 0.9577 and backdoor
component 0.0000 at every point. UNSW-NB15's own two costs span 21.1% to
42.1%, so the sweep reaches below that band as well as matching its top.

That control is bounded rather than measured, and the bound is structural. Every
cost it scans lies above 16, where the clean arm already calls every stamped flow
benign, so a backdoor component of zero is what the ceiling forces whatever
record coverage does. The bound is not avoidable by scanning lower, because the
sweep's smallest cost, 20 features, already exceeds the cost 16 at which the
clean arm saturates. Every footprint large enough to be worth matching is above
the ceiling. What it establishes is
that enlarging the footprint twenty-fold does not move CTU-13 out of the evasion
regime. It does not by itself exclude record coverage as the explanation of the
split, and we do not count it as doing so.

## A gated atypicality score restores Activation Clustering's ranking
*(originally `\label{app:ac-score}`)*

The candidate statistic is a weighted log-density of atypicality within the
minority Gaussian component, evaluated in the reduced activation space. Two
variants were pre-registered: an un-gated form scoring every sample, and a gated
form scoring only samples assigned to the minority component.

**Table.** The gated Activation Clustering score against its four pre-registered
gates, with the un-gated variant for contrast. Recall here is recall at a stated
false-positive budget, not at a label-free threshold.

| Pre-registered gate | Bar | Gated | Un-gated |
|---|---|---|---|
| Worst-cell AUC | >= 0.90 | 0.9626 | 0.9626 |
| Recall at 10% FPR | >= 0.90 | 1.0000 | 1.0000 |
| \|rho\| vs. Spectral | <= 0.90 | 0.3987 | 0.3987 |
| Clean flag rate | <= 10% | 0.09% | 14.3% |
| Verdict | | pass | informative negative |

The gated variant passes all four pre-registered gates (Table the table below)
over five seeds and the four window cells, each cell being one pairing of a poison rate
with a trigger cost. Its worst cell reaches ranking AUC 0.9626, with the four cells
at 0.9626, 0.9681, 0.9836, and 0.9864. Recall reaches
1.0 at a 10% false-positive budget on every cell. Its maximum absolute
rank correlation against Spectral Signatures' score is 0.3987, below the 0.90
redundancy bar, so it is not a restatement of the existing detector. It flags
0.09% of samples on clean models. The un-gated variant shares the same
ranking, which is unsurprising since the ordering is identical, but fails the
clean control at 14.3% false alarms (95% CI [11.5%, 16.4%]) and is recorded as an informative
negative. Gating to the minority component is therefore what makes the score
usable, not a refinement of it.

Two caveats travel with this result and we state them wherever it is quoted.
First, its recall figures are recall at a *false-positive budget*. The
budget-free rule at $z=3.0$ that rescues Spectral Signatures recovers 0.001
on this score, so what is repaired is the ranking and not yet a deployable
threshold. Second, the score's direction was reversed after a seed-42 probe
showed poison at the minority component's low-density periphery rather than its
core, a directed AUC of 0.0303. The reversal is a deviation from the
pre-registration, and both directed and two-sided values are recorded for every
row.

## Down-sampling does not rescue the detectors
*(originally `\label{app:h3}`)*

**Figure** (source: `paper_ojcs/figures/fig3_h3_confound.pdf`). Poison recall on
CTU-13 before and after down-sampling the benign class toward parity, at rate 5%
and cost 8 under the inherited fixed budget. Bars are seed-range intervals.

Class imbalance is the obvious confound for any recall collapse on a 98/2
dataset (Fig. the figure below). Activation Clustering's imbalanced value there is a
measured 0.0006 rather than an absent bar, and is printed above it. Down-sampling the benign class to parity at rate 5%
and cost 8 lowers Spectral Signatures' recall from 0.8829 (95% CI [0.8277,
0.9503]) to 0.2513 (95% CI [0.1769, 0.3653]). Activation Clustering stays near
zero either way. That operating point sits outside the window, so we repeat the
test on the four window cells and record the ranking metric beside the recall.
Balancing lowers Spectral Signatures' recall there from 0.1425 to 0.0308, and
lowers its ranking AUC from 0.9825 to 0.5559, with per-run values spanning 0.2054
to 0.8770. The
balanced condition therefore does not show a starved budget sitting on an intact
score. It shows the separability itself gone, which is a different regime from
the window rather than a repair of it. One caution belongs with that reading.
Parity is reached by discarding benign rows, so the balanced training set holds
5{,}540 rows, and the condition moves absolute training size together with the
class ratio. We read it as ruling imbalance out as the explanation of the window,
not as a clean measurement of imbalance alone.

UNSW-NB15 separates the 2. Its minority class is large enough that parity keeps
every one of 34{,}189 minority rows and matches the majority to them, leaving
68{,}378 training rows against CTU-13's 5{,}540. The ranking AUC does not fall
there. It rises from 0.9581 to 0.9924 across the three window cells and five
seeds, while Spectral Signatures' fixed-budget recall rises from 0.0216 to 0.8814
and Activation Clustering's from 0.1443 to 0.7229. Balancing on a dataset that can
afford it therefore leaves the score intact and feeds the budget, which is the
starved-budget account stated as a direct measurement. The CTU-13 arm remains what
it was, a condition in which separability is lost along with the training rows.

## The detectors on the dataset family, against a control on the same dataset
*(originally `\label{app:netflow-detectors}`)*

The two NetFlow relations rejected at the 0.999 admission bar are measured rather
than assumed away. The relation that per-size packet buckets sum to no more than
the total packet count holds on 0.7972 of NF-UNSW-NB15-v2 rows and 0.9331 of
NF-CSE-CIC-IDS2018-v2 rows. The relation that inbound bytes reach the floor
implied by packet count times smallest packet length holds on 0.9816 of
NF-UNSW-NB15-v2 rows. The two bytes-per-second columns are capped at
$1.25 x 10^{10}$ bytes per second, touching 36,713 of 38.4 million feature
values, while the two average throughput columns peak at 4.3 Gbps and are left
alone.

We score all five detectors on the same cells. A blatant constraint-violating trigger is planted
on each dataset separately, because a null on a dataset with no passing control of its own is
uninterpretable. Spectral Signatures recovers 0.9971, 1.0000 and 0.9781 of the planted rows under
the inherited budget on NF-CSE-CIC-IDS2018-v2, NF-ToN-IoT-v2 and NF-UNSW-NB15-v2, so its nulls on
all three are interpretable. The other four arms do not clear the bar everywhere, and their status is
dataset-dependent rather than uniform. Activation Clustering recovers 0.0063 on
NF-CSE-CIC-IDS2018-v2, 0.5998 on NF-ToN-IoT-v2 where it isolates the poison on three seeds and nothing
on the other 2, and 0.9984 on NF-UNSW-NB15-v2. Neural Cleanse flags the target class on every seed
of 2 datasets and on none of NF-UNSW-NB15-v2. SPECTRE recovers 0.1090 on NF-ToN-IoT-v2 and 0.0830 on NF-UNSW-NB15-v2 under the
inherited budget, the two datasets that carry their own control file, though neither number sits behind a passing control for this arm. STRIP recovers
0.9297 and 0.7387 in that same order. We therefore draw a
portability verdict from Spectral Signatures alone, and report every other arm's family cells as an
unresolved control failure on the datasets where its control did not pass. The failure modes
described below characterize what those detectors do there. They are not evidence about whether
they port.

One dataset reproduces this paper's central finding. On NF-UNSW-NB15-v2 at native balance the
inherited fixed budget of Spectral Signatures collapses while the same score still ranks the
planted rows, and the budget-free rule recovers most of them
(Table the dataset-family detector table (main paper)). The removal budget
starves while the discriminative signal survives. The window has been seen before on UNSW-NB15,
and this dataset is a NetFlow rendering of that same capture, so what is new is not the traffic but
the contrast within one schema. Two further datasets on that schema produce no window at all, the
inherited budget recovering almost every planted row on each. The fourth,
NF-BoT-IoT-v2, cannot fund the attack and yields no detector measurement. STRIP's
ranking AUC on NF-UNSW-NB15-v2's five native-balance cells is 0.1718, below chance
rather than uninformative. We report that
without explaining it.

Activation Clustering fails there in two ways a mean recall cannot separate, and
the distinction decides whether an operator would notice. Across 75 interpretable
cells it isolates the poison in 41, collapses to flagging a handful of rows in
21, and flags the wrong cluster in 13. The wrong-cluster split is confident and
genuine rather than an artifact, so the gate that exists to suppress an unreal
split cannot fire. Those 13 cells carry a mean silhouette of 0.5394 with a worst
of 0.4406, against 0.7740 across the 41 cells where the poison is isolated, so
the split is weaker than a clean isolation and far from the near-zero a spurious
partition would give. Re-running both failure modes at full scale, changing only the
number of rows scored, reproduces each one, so the subsample cap does not cause
them. The per-cell taxonomy, the silhouette evidence and the scale control are in
`docs/artifact/NETFLOW_FAMILY.md` of the artifact repository.

## Two external comparators did not clear their positive controls
*(originally `\label{app:comparators}`)*

**A tabular-native detector released as a preprint was also tested, and
also did not qualify.** The active-paths detector proposed for neural intrusion
detection [Hoyheim2026activepaths] traces local feature contributions
through a kernel principal-component decomposition into a density-based
clustering. We put in front of it the same loud constraint-violating control that
Spectral Signatures, Activation Clustering and STRIP clear, at a bar of 0.70
recall on every seed, fixed before the run. That method publishes no removal
budget, so the bar was applied under both rules this paper uses rather than under
a rule of its own. It clears neither. Under the inherited top-$k$ rule it reaches
0.7656, 0.7167, 0.0000, 0.8177 and 0.8053, failing on 1 seed of 5. Under the
budget-free rule it reaches 0.7677, 0.7167, 0.0000, 0.0000 and 0.6024, failing on
3. Its mean ranking AUC is 0.7737. Scoring it the way the circularity argument
requires, rather than under the budget alone, does not change its verdict. It is reported in the lower block
of Table the window-cells detector table (main paper) beside the two ported detectors that missed the same
bar, carrying control status only because it was never scored on the grid. No
parameter was tuned to make it pass.

That null is reportable because the control is shared. On the same trigger, cell
and rule, Spectral Signatures recovers 0.9995 and Activation Clustering 0.9998,
so the failure is a property of this detector under this attack rather than of
our harness. What it establishes is narrow and we state it exactly. The one
purpose-built, tabular-native detector we could find does not separate
a clean-label, constraint-projected attack from clean traffic under its own loud
control. It does not establish that no tabular-native defense can.

**A tabular-native comparator was attempted and did not qualify.** We
implemented the density-based mitigation proposed for this
setting [Severi2024agnosticmitigation] and put a positive control in front
of it, at a pre-registered bar of poison recall 0.90 on every seed under the
method's own published rule. The control failed on all four constructions we
tried, so no realizable cell was scored and the method does not enter the
detector comparison. A post-hoc diagnostic locates the failure. With the importance reduction bypassed
so the clustering receives the trigger's own features, it isolates all 5{,}722
planted rows in one cluster at purity 1.0 on every seed, leaving none
unclustered, and the published absorption rule then discards that cluster while
flagging seven or eight others. One reading of that is that the port is sound and the inherited decision rule is
what loses the poison, which is the shape of this paper's central result. We do
not assert it, and the threshold-free score argues against it. Scoring this
comparator the second way, as Goal 3 requires of every arm, its mean ranking AUC
over five seeds is 0.4991 under the published construction, 0.4104 with the
gradient-boosted ranking, 0.5693 with the selected-feature ranking and 0.6000
with the importance reduction bypassed. All four sit at or near chance, so unlike
Spectral Signatures this comparator does not carry an intact ranking behind a
failing rule, and the reading above is not one its own score supports. Two
things this does not license. It is not the claim that the tabular-native defense
fails on our recipe, because the gate never passed and no realizable cell ran.
And the diagnostic runs on a control pushing 8 features by 6 standard deviations,
an extremity that may sit outside the method's assumptions, so only further work
separates a mis-specified control from a faulty decision rule.

## STRIP: instability, the entropy bound, and the delivered rate
*(originally `\label{app:strip-detail}`)*

On UNSW-NB15 STRIP's mean ranking AUC of 0.6271 is unstable rather than uniformly
weak. At rate 1% and cost 16 the per-seed AUC spans 0.3808 to 0.7678 while attack
success stays above 0.99 on every seed, so one seed ranks poison below chance
under an attack that succeeds as well as the rest. The budget-free rule
returns recall 0.0 on both datasets. On the matched MNIST
control, where STRIP reaches recall 0.9945 at ranking AUC 0.9909, the same
threshold still returns 0.0 at every tested $z$.

The bound is what defeats the cutoff. At the window cell at rate 0.5% and cost
16, averaged over five seeds, the clean bulk has a median of $-0.0558$ and a median
absolute deviation of 0.0469, placing the cutoff at $+0.1530$ at $z=3.0$ and
$+0.0834$ even at $z=2.0$. Both sit above the largest value the statistic can
take, so the rule flags no row on any seed at any threshold we test.

Boundedness alone does not predict inertness. The constraint-departure count is
bounded above by the 360 relations in the manifest and still fires at every
threshold, because its clean bulk sits far below that ceiling relative to its
spread and its median absolute deviation is zero, which sends the rule into its
standard-deviation fallback and makes its $z_{\max}$ infinite. A bound is what
makes a small $z_{\max}$ possible. It does not make it happen.

Under STRIP's published rule the requested false-rejection rate is not always the
delivered one. Enough rows saturate the victim's output to tie at the entropy's
numerical floor, which puts a 1% request inside a point mass. The delivered
false-rejection rate on the clean calibration sample the cutoff is fit on averages
2.14% over the four CTU-13 window cells, with per-run values from 1.61% to 2.59%.
The false-positive rate the same cutoff then pays on the scored class is 2.17%
over those cells. On the cost-16 half of the window the rule recovers 0.9441 at a
2.12% false-positive rate on the scored class, while the budget-free rule buys
Spectral Signatures 1.0000 across all four window cells at 6.58%. UNSW-NB15 carries no
such mass, delivers 1.03% against the same request, and recovers 0.0007.

## UNSW-NB15 composition before the binary collapse

Nine attack categories and the benign and total rows. The Backdoors category is
the dataset's own label for a class of intrusion, unrelated to the backdoor
poisoning this paper studies. Moved here from `tab:unsw-categories` (main paper
Table 2 in earlier drafts) during the OJ-CS 12-page cut; the binary-collapse
distribution this table documents is steeply unequal, which the main paper
states without printing every category count.

| Category | Rows | Category | Rows |
|---|---|---|---|
| Normal | 1,523,904 | Analysis | 1,480 |
| Exploits | 16,949 | Backdoors | 1,266 |
| Generic | 14,529 | Shellcode | 918 |
| Fuzzers | 13,302 | Worms | 106 |
| Reconnaissance | 8,177 | *Total attack* | 60,355 |
| DoS | 3,628 | *Total* | 1,584,259 |

## Neural Cleanse clean-model calibration, per seed

The loud CTU-13 control scores Neural Cleanse under per-run clean-model
calibration: a clean model is trained at the same seed, the same
larger-to-smaller mask-norm ratio is taken on it, and the seed counts only where
the poisoned model exceeds its own clean reference. That is stricter than the
fixed ratio of 2.0 the grid entries use. Moved here from the main paper's
Measurement Design during the OJ-CS 12-page cut; the paper states the verdict
(the poisoned model is the more anomalous on two seeds of five) without printing
the ten underlying ratios.

| Seed | Poisoned ratio | Clean ratio | Poisoned more anomalous |
|---|---|---|---|
| 1 | 90.11 | 97.96 | no |
| 2 | 103.31 | 81.12 | yes |
| 3 | 88.29 | 81.33 | yes |
| 4 | 107.51 | 109.22 | no |
| 5 | 76.66 | 94.89 | no |

Two of five seeds have the poisoned model exceeding its own clean reference.

## Author biographies

Removed from the main paper for the OJ-CS 12-page limit, which counts
biographies toward the total. Preserved here verbatim; restore to `main.tex`
if the page budget ever allows.

**Nazmus Sakib** is currently pursuing the B.Sc. degree in computer science and engineering with American International University-Bangladesh (AIUB), Dhaka, Bangladesh. Research interests include adversarial machine learning, backdoor and poisoning attacks on learned detectors, and network intrusion detection.

**Md. Al Amin** is currently pursuing the B.Sc. degree in computer science and engineering with American International University-Bangladesh (AIUB), Dhaka, Bangladesh. Research interests include machine learning for network security and intrusion detection.

**Md. Shahriar Haque** is currently pursuing the B.Sc. degree in computer science and engineering with American International University-Bangladesh (AIUB), Dhaka, Bangladesh. Research interests include machine learning for network security and intrusion detection.

**Md. Mazid-Ul-Haque** received the B.Sc. degree in computer science and engineering and the M.Sc. degree in computer science from American International University-Bangladesh (AIUB), Dhaka, Bangladesh. Currently an Assistant Professor with the Department of Computer Science, AIUB. Research interests include computer networks, wireless communication, artificial intelligence, and software development methodologies.

## Loud-control ranking AUCs

Dropped from the main paper's positive-control table during the OJ-CS 12-page
cut, where the AUC columns duplicated what the window-cell detector table
already reports. These are the loud tabular control's ranking AUCs, per
detector and dataset, over five seeds.

| Detector | CTU-13 AUC | UNSW-NB15 AUC |
|---|---|---|
| Spectral Signatures | 0.9826 | 0.9895 |
| Activation Clustering | n/a (emits a partition, not a score) | n/a |
| STRIP | 0.9787 | 0.9795 |
| SPECTRE | 0.7909 | 0.8582 |
| Neural Cleanse | n/a (emits a flag) | not run |

SPECTRE's is the weakest of the three arms that emit a ranked score, on both
datasets, which is the comparison the paper states without printing these two
numbers.

## The matched-rate vision control

Moved from the main paper's Measurement Design during the OJ-CS 12-page cut. The
paper keeps the finding (the inherited removal budget fails on the vision
substrate itself at attacker-realistic poison rates, so the starvation window is
not a property of tabular data) and this table carries the per-rate detail.
Five seeds per row.

| Rate | Poison rows | ASR | AUC | Fixed recall | MAD recall |
|---|---|---|---|---|---|
| 0.5% | 30 | 0.8785 | 0.8411 | 0.1600 | 0.2600 |
| 1% | 60 | 0.9291 | 0.8306 | 0.2233 | 0.3100 |
| 2% | 121 | 0.9784 | 0.8662 | 0.4694 | 0.4397 |
| 5% | 312 | 0.9960 | 0.9795 | 0.8840 | 0.8276 |
| 10% | 658 | 0.9981 | 0.9901 | 0.9581 | 0.8334 |
| 31.3% | 2,704 | 0.9999 | 0.9814 | 0.9809 | 0.1852 |

The 31.3% row retrains the control reported in the paper, so its recall reads
0.9809 against that run's 0.9828. The control is dirty-label throughout, so it
matches the rate regime and not the label regime.

## Why the backdoor component turns negative on some seeds (post-hoc probe)

Moved from the main paper's Results during the OJ-CS 12-page cut. Reported there
as a candidate explanation rather than a finding, and repeated as such here.

If poisoning teaches the victim to rely less on the watermark and more on
context, the poisoned victim's attribution mass on the stamped features should
fall relative to the clean one. Comparing mean absolute Shapley share on those
features at trigger costs 6 and 7, three of the four seeds not pre-excluded
clear a threshold of a 0.02 drop fixed before the comparison, at cost 6, at
-0.0379, -0.0889 and -0.0731. At cost 7 only one of those four clears it, at
-0.0607, so the probe is weaker at the higher cost than at the lower. Seed 789
was excluded in advance because its backdoor component runs the other way.

The mechanism has support on a subset of seeds rather than uniformly, so it is
stated as a candidate explanation and not as a finding.

## STRIP's per-seed ranking AUC on UNSW-NB15

The paper reports STRIP's UNSW-NB15 ranking as a mean AUC of 0.6271 that is
unstable rather than uniformly weak, with one seed ranking poison below chance
under an attack that succeeded as well as the rest. The per-seed spread behind
that mean runs from below chance to **0.7678**. Moved here during the OJ-CS
12-page cut, where the Discussion paragraph that carried it was compressed.
