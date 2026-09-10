# Supporting measurements

Full measurements and derivations for findings the manuscript states in summary. The paper
reports each of these in condensed form and defers the detail here, so a reader can check a
summary against the measurement it came from. Every section names where the paper summarizes it.

Nothing here is a separate result: each block is the complete text and per-cell numbers behind a
claim already made in the manuscript.

## Design goals

Summarized in `04_methods.tex`; the full text and measurements follow.

```latex
\subsection{Design goals}

The measurement supports four claims at once, each imposing a requirement the
others do not (Fig.~\ref{fig:pipeline}). \textbf{Goal 1}, distinguish a planted
backdoor from pre-existing evasion, requires a never-poisoned model evaluated on
the same stamped rows. \textbf{Goal 2}, keep the attack fixed while the dataset
varies, requires one recipe and one victim family run unchanged on both.
\textbf{Goal 3}, separate a detector's ranking signal from its decision rule,
requires scoring every detector twice, under its published rule and under a
threshold-free metric. \textbf{Goal 4}, make any null interpretable, requires a
positive control on which each detector is known to work before it is trusted to
report a failure elsewhere.

```


## Victim models

Summarized in `04_methods.tex`; the full text and measurements follow.

```latex
\subsection{Victim models}

Two victims are trained on the standardized features, both with
inverse-frequency class weighting so that the clean baseline is a competent
detector rather than a majority-class predictor. The multilayer perceptron (MLP) has a
two-layer body (757 to 256 to 128, ReLU) and a linear head, trained for 20
epochs of Adam at $10^{-3}$, batch size 512, no schedule. Its 128-dimensional
penultimate layer is the representation the activation-based detectors inspect.
On UNSW-NB15 the same architecture runs at input width 38, reaching clean
accuracy 0.9762 against an always-benign baseline of 0.9174 on that
8.26\%-positive test block. The gradient-boosted tree ensemble~\cite{Ke2017lightgbm} is
grown to 200 trees of at most 64 leaves, at learning rate 0.05, with 80\% of
features sampled per tree and the positive class weighted at the
negative-to-positive ratio. As a tree ensemble it exposes no
intermediate activation and admits no gradient path, which makes it a test of
detector applicability by architecture rather than a missing result.

```


## Statistical protocol

Summarized in `04_methods.tex`; the full text and measurements follow.

```latex
\subsection{Statistical protocol}

Every headline number is computed over five seeds $\{42, 123, 456, 789, 1337\}$,
with a disjoint replication set $\{2026, 31415, 27182, 16180, 11235\}$ where
stated. One arm carries a third block, the contiguous 501 to 520, added after
the first two disagreed; it was not pre-registered and is declared as a
departure with the artifact. Confidence intervals, where quoted, are percentile
bootstrap 95\% intervals over 10{,}000 resamples~\cite{Efron1993bootstrap} whose
resampled unit is the set of values entering the mean; where that unit is five
seeds the interval describes the seed spread rather than carrying calibrated
coverage, and we read it as descriptive. Means are reported with standard deviations
over the seeds of the stated cell, and we run no parametric tests at
$n \le 30$. Four axes are kept distinct throughout: attack success, poison
recall under a stated rule, ranking AUC as a threshold-free measure of signal,
and false-positive rate as the cost of a rule, the count of clean rows a rule
flags over the clean rows remaining in the target class. All 125 CTU-13
detector-sweep cell-seed runs re-score to their committed values on fixed
hardware, as do 40 UNSW-NB15 detector runs and 35 threshold runs.

Two provenance disclosures belong here. The budget-starvation criterion, mean
attack success at or above 0.8 with fixed-budget recall below 0.5, was chosen by
inspecting the computed grid rather than fixed beforehand; we mitigate that by
applying the same thresholds unchanged to UNSW-NB15 and reporting every cell
individually. The direction of the Activation Clustering repair score was chosen
after seeing probe data, and is reported as a deviation wherever that score is
quantified.

Class imbalance is a known amplifier of backdoor threat~\cite{Lin2026rpp}, so to
test whether it rather than tabular structure drives any recall collapse we
re-run the entire pipeline, trigger selection included, after down-sampling the
benign class toward parity. A footprint-matched control tests whether the
fraction of the record the trigger touches explains the cross-dataset split,
scaling CTU-13's cost to $\{20, 40, 80, 159, 319\}$ features at fixed poison
rate, seeds and victim, spanning 2.6\% to 42.1\% of a CTU-13 record and reaching
the 42.1\% a cost-16 UNSW-NB15 trigger occupies. The adaptive attacker selects
on three surrogate seeds and is evaluated on five held-out seeds, never mixed,
from a pre-registered 64-candidate budget of which 28 survived aggregation; its
winning trigger is frozen at selection time and hash-verified before evaluation,
so no held-out seed contributes to choosing it.```


## Trigger cost, not poison rate, buys attack success

Summarized in `05_results.tex`; the full text and measurements follow.

```latex
\subsection{Trigger cost, not poison rate, buys attack success}

Attack success is governed almost entirely by trigger cost. At costs 1, 2 and 4
it stays near background, 0.040 to 0.045, at every tested rate. At cost 8 it is
potent but bimodal across seeds rather than merely variable, with mean success
spanning 0.632 to 0.935 unrelated to rate and per-cell standard deviations
reaching 0.478, which no mean over them represents; its mean over all five rates
is 0.7932, below the 0.8 criterion defining the window, so cost 8 qualifies at
some rates and not others. At cost 16 it saturates to 1.0 on every seed at every
rate including the lowest, 0.5\%. Clean accuracy holds at 0.9991 across all 125
runs against an always-benign classifier's 0.9926, with clean botnet recall
0.9577 -- which matches the cost-16 evasion component by construction, since
$1-0.0423$ is exactly that recall. LightGBM shows no attack response, pinned at
0.047 with zero variance: the same 19 of 407 botnet test rows are predicted
benign throughout, a pre-existing false-negative floor rather than a backdoor.

```


## Neural Cleanse resists repair, and two limits are structural

Summarized in `05_results.tex`; the full text and measurements follow.

```latex
\subsection{Neural Cleanse resists repair, and two limits are structural}

Neural Cleanse emits a hard flag rather than a ranked score, so the AUC
diagnostic does not apply, and it does not run on the window: it runs on a
six-cell boundary subset, rates $\{0.005, 0.05\}$ crossed with costs
$\{4, 8, 16\}$ at five seeds, so its rates are over 30 runs and its two
entries in Table~\ref{tab:detectors} are drawn from a different set of cells
than the window. Detection falls from 1.0 with zero variance on the vision
control to 0.367 on tabular NIDS, 11 of 30, and to 0.400 over the two
attack-effective costs, 8 of 20; the subset's cost-4 cells sit at background
attack success 0.0423 and contribute three of the 11 flags. UNSW-NB15 behaves
the same way, the inversion flagging the target class in nine of 20 boundary
runs and exceeding its clean-model calibration in the same nine. Neither half
of the detector is exonerated: three pre-registered repair candidates each post
0.0000 gain, and three of four families of decision statistic were run and all
three return the same null while the detector succeeds on a matched vision
control, so the two-class degeneracy of the anomaly index is not a sufficient
explanation. The fourth, structural, family is undefined on a tree ensemble,
since no inversion exists to score. This arm did not clear the loud tabular
control, which fires on only two of the five loud CTU-13 seeds and was never
run on UNSW-NB15, so an implementation fault in the port remains a live
alternative we do not exclude, and we report the failure as unresolved across
inversion and decision rule rather than locating it in one. The mask
diagnostic, the repair grid and the four statistic families are reported with
the artifact (\texttt{docs/artifact/SUPPLEMENTARY.md} and
\texttt{docs/artifact/PAPER\_APPENDIX\_OJCS.md}).

The two limits bounding LightGBM's applicability, one per activation-based
detector and one for the inversion above, are structural rather than
statistical, as reported above where the controls that did not pass are
collected.

Class imbalance is the obvious confound for a recall collapse on a 98/2
dataset, and it does not survive the test. Down-sampling toward parity
rescues neither activation-based detector on CTU-13, where parity leaves
5{,}540 training rows and the ranking AUC itself falls to 0.5559.
UNSW-NB15 supplies the arm that dataset cannot: parity there keeps
68{,}378 rows, the ranking AUC rises from 0.9581 to 0.9924, and the fixed
budget's recall rises from 0.0216 to 0.8814 (artifact repository,
\texttt{docs/artifact/PAPER\_APPENDIX\_OJCS.md}, ``Down-sampling does not
rescue the detectors''). The score survives balancing and the budget is
fed, the starvation account measured directly rather than inferred.
```


## The second dataset reproduces the window and complicates the repair

Summarized in `05_results.tex`; the full text and measurements follow.

```latex
\subsection{The second dataset reproduces the window and complicates the repair}

The same pipeline runs unchanged on UNSW-NB15. Of its eight cells, one is
attack-not-effective and four already clear recall 0.5; the remaining three
reproduce the window, with the attack succeeding, the fixed budget missing it,
and the budget-free rule flagging it (Table~\ref{tab:secondary}). Spectral
Signatures' ranking AUC on those three is 0.9287 merged over 30 seeds against a
fixed-budget recall of 0.0292, so the window is not a CTU-13 artifact. Which
cells recover follows from where the poison sits relative to the cut: the poison
median's margin above the $z=3.0$ threshold, in MADs, separates all 30 runs,
from $+0.03$ to $+30.95$ where the threshold recovers the poison and $-4.08$ to
$-0.03$ where it collapses -- an identity rather than a prediction, and
unavailable to a defender in any case.

\begin{table}[!htb]
\centering
\caption{UNSW-NB15 window cells, MAD recall over base seeds, replication seeds, and all 30 merged. The FPR and AUC columns are over the merged 30. Spreads are population standard deviations.}
\label{tab:secondary}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}cccccc@{}}
\toprule
& \multicolumn{3}{c}{MAD recall} & & \\
\cmidrule(lr){2-4}
Rate/Cost & $n=5$ base & $n=5$ repl. & $n=30$ merged & FPR & AUC \\
\midrule
0.005/16 & 0.8495 $\pm$ 0.297 & 0.7245 $\pm$ 0.351 & 0.8066 $\pm$ 0.350 & 6.75\% & 0.9357 \\
0.010/8  & 0.9266 $\pm$ 0.141 & 0.6067 $\pm$ 0.338 & 0.7440 $\pm$ 0.343 & 7.16\% & 0.9273 \\
0.010/16 & 0.8824 $\pm$ 0.220 & 0.4087 $\pm$ 0.448 & 0.7660 $\pm$ 0.369 & 6.98\% & 0.9233 \\
\bottomrule
\end{tabular}
\end{table}

That recovery is not the finding to lead with. CTU-13's four cells recover to
recall 1.0 on every seed at zero variance; UNSW-NB15's three do not, at standard
deviations of 0.297, 0.141 and 0.220 on the base seeds alone, worse on five
replication seeds, and a further 20 seeds drawn without inspecting any result
raise each cell to $n=30$ merged where the three means settle at 0.8066, 0.7440
and 0.7660 with the spread staying at 0.343 to 0.369. Those deviations describe
a bimodal outcome rather than a spread about a center, so a mean understates
what a defender faces: across all 90 merged runs the budget-free rule recovers
at least 0.9 of the poison in 62 and at most 0.5 in 23, with only five in
between. The operative quantity is the failure rate, roughly one run in 4, and
nothing the detector reports at the time tells the defender which run it is
about to be. The window cells were selected on the base seeds, so the base
column is scored on the seeds that chose it and the replication column is not;
UNSW-NB15 supports the qualitative claim that the miss and the recovery recur
outside CTU-13, not that the recovery is as reliable there.

```


## Recall is a detection statistic, not a defense outcome

Summarized in `05_results.tex`; the full text and measurements follow.

```latex
\subsection{Recall is a detection statistic, not a defense outcome}

UNSW-NB15 is the dataset on which poison removal is the security-relevant
operation, so we retrain the victim on the MAD-filtered training block at
$z=3.0$ over its three window cells and the five base seeds, and compare attack
success before and after. The removal rule drops most of the poison in every
cell, and the backdoor does not always go with it: at $(0.01, 8)$ mean attack
success falls from 0.8449 to 0.0546, but at $(0.005, 16)$ only from 0.9626 to
0.4310 and at $(0.01, 16)$ from 0.9919 to 0.6154, with clean accuracy dropping
by at most 0.0211. Poison recall in the 0.85 to 0.93 range therefore lowers
attack success by between 0.38 and 0.79 in absolute terms depending on the cell,
and does not rank the three cells in the same order as that drop. The repair
Table~\ref{tab:secondary} reports as working is, on the dataset where removal
matters, a partial defense. The replication seeds recover less poison, so
repeating it there would leave more attack success, not less.

A score-aware adaptive attacker on a disjoint-seed surrogate keeps attack
success at 1.0 while the budget-free rule still recovers all of its poison, at a
mean 6.48\% false positives across five held-out seeds, evading on none of them
under a criterion fixed before the run that counts a single held-out evasion as
transfer.
```


## What decides whether a vision-built detector ports

Summarized in `07_discussion_conclusion.tex`; the full text and measurements follow.

```latex
\subsection{What decides whether a vision-built detector ports}

The five detectors do not share a fate, and the axis separating them is not the
activation space the literature emphasizes: Spectral Signatures, SPECTRE and
Activation Clustering read the same 128-dimensional penultimate layer and end in
three different places. What separates them is what each does with what it
reads. Spectral Signatures carries a discriminative score whose calibration
constant fails, the most benign failure in the set since it is diagnosable with a
ranking metric and correctable two ways. SPECTRE shows that sharpening the score
does not substitute for fixing the multiplier, so effort spent on a better score
is spent on the wrong half of the detector. Activation Clustering has no score at
all, so its repair had to supply the missing object rather than fix an existing
one. Neural Cleanse fails in a way none of these describes and is reported as
unresolved. STRIP is the reminder that dataset dependence cuts across all of it,
and the clearest case of this paper's thesis turned on itself: scoring it only
under the inherited rules would have reported it as the weakest arm rather than
mid-ranked under its own.

```


## Both repairs work, and both have a price

Summarized in `05_results.tex`; the full text and measurements follow.

```latex
\subsection{Both repairs work, and both have a price}

The budget-free rule replaces the removal budget with a
median-absolute-deviation z-score cut, flagging score outliers rather than
assuming a poison count. Adopting it reaches recall 1.0 with zero variance
across the four window cells at all three pre-registered thresholds; at the
cheapest, $z=3.0$, mean false positives are 6.58\% at a mean precision of
0.105, and at the anchor cell it recovers all 572 poisoned rows on every seed
while flagging a mean of 8{,}017 flows. Table~\ref{tab:fullgrid} summarizes the
rule by trigger cost, with all 25 cells published with the artifact.
Budget-free recall rises from background below the attack's effective cost to
essentially one at both effective costs while the false-positive rate falls,
and where the rate is large enough to feed it the inherited budget is sometimes
cheaper: at cost 16 both reach full recall, at 5\% poison the inherited budget
costs 2.70\% against 4.67\%, and at 10\% the ordering reverses, 5.71\% against
3.44\%. The inherited budget's spread at cost 8 is as large as its own mean,
recall ranging 0.0203 to 0.9996 across rates, which is what supports treating
the window as a joint cost-rate property rather than a rate-only one.

\begin{table}[!htb]
\centering
\caption{CTU-13 grid by trigger cost, mean over five rates at $z=3.0$, standard deviations in parentheses. Costs 1, 2, and 4 are merged.}
\label{tab:fullgrid}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}cccccc@{}}
\toprule
Cost & MAD recall & MAD FPR & Fixed recall & Fixed FPR & AUC \\
\midrule
$\le$4 ($n=75$) & 0.0726 (0.0077) & 7.01\% & 0.0603 (0.0545) & 5.67\% & 0.5023 \\
8 ($n=25$)  & 0.9998 (0.0004) & 5.92\% & 0.4370 (0.4372) & 2.79\% & 0.9720 \\
16 ($n=25$) & 1.0000 (0.0000) & 5.25\% & 0.6906 (0.3895) & 2.28\% & 0.9889 \\
\bottomrule
\end{tabular}
\end{table}

\textbf{The budget-free rule is not a universal repair. Which rule applies
depends on the detector, and within a detector on the poison rate.} Scored on
identical window cells under identical rules it recovers 1.0000 for Spectral
Signatures and SPECTRE, 0.6576 for an isolation-based filter that failed its
own pre-registered gate and enters only as a score source, and 0.0000 for
STRIP, whose top-$k$ budget recovers 0.2806 on the same scores. STRIP is the
one arm the rule makes worse, and the mechanism is the bound on its own
statistic characterized above, not a property of score shape. A label-free
diagnostic cannot substitute for testing both rules directly: Sarle's
bimodality coefficient~\cite{Pfister2013bimodality} pooled over the grid ranks
which rule recovers more poison at a prediction AUC of 0.8196, but that ranking
reverses once stratified by arm, STRIP preferring the inherited budget on all
125 of its comparisons and SPECTRE the budget-free rule on all 125 of its own.
Within a detector the rate decides through the starvation condition itself:
restricted to the attack-effective costs 8 and 16, Spectral Signatures'
budget-free recall is 1.0000 from 0.5\% through 5\% and 0.9996 at 10\% while
its fixed-budget recall climbs from 0.0416 to 0.9998 and overtakes only between
5\% and 10\%. Costs 1, 2 and 4 are merged throughout because none is
attack-effective, at individual ranking AUCs of 0.5016, 0.5024 and 0.5029;
averaging them in instead reports a rate-independent 0.44 that describes no
cell of the grid.

The published rule is also repairable by recalibrating its constant. At the
anchor cell mean recall rises from 0.0629 at the original multiplier of 1.5 to
1.0 with zero variance from five onward, false positives rising from 0.74\% to
2.06\%; the constant is regime-specific, and checking whether one sufficed still
requires labels. That repair is dataset-specific: on UNSW-NB15's three window
cells the inherited constant recovers only 0.0216 and no swept multiplier
reaches the budget-free rule's recall at or below its cost, multipliers 7 and 10
bracketing it at 0.7204 for 5.36\% and 0.9234 for 7.82\% against 0.8861 at
6.43\%. Recalibrating rescues the budget on the dataset where removal does not
change attack success and fails on the dataset where it does. A univariate
baseline confirms the signal is genuinely multivariate, resting on the ranking
rather than a rule since a recall gap can be produced by a rule while an AUC gap
cannot: a per-feature z-filter granted the same oracle poison fraction reaches
AUC 0.8704 against Spectral Signatures' 0.9825 on the same cells, and recall
0.0556 against 1.0 at the budget-free rule's own cost. The multiplier sweep and
the bimodality study are reported with the artifact.
```


## Vision control

Summarized in `04_methods.tex`; the full text and measurements follow.

```latex
\subsection{Vision control}

No tabular verdict is trusted before the detector reproduces on a matched
vision control: an MNIST classifier backdoored with the dirty-label BadNets
patch~\cite{Gu2019badnets} at $4{\times}4$ maximum intensity, planted into 5\%
of non-target-class images with labels flipped, leaving poison at 31.3\% of the
target class. A detector passes at poison recall 0.90 on every seed, a bar
chosen while building the control rather than pre-registered. Which arms are
admitted does not turn on its exact value, since any bar from 0.90 to 0.97
admits the same set.

A second stage repeats the check on tabular data. The loud control stamps a
watermark of mean plus six standard deviations onto the eight top-ranked
features at a 5\% poison rate but skips the projection step, so the rows
violate the feasibility manifest and are blatant by construction. It stays
clean-label, matching this paper's label regime, but not the window's rate or
cost, running at 5\% and cost 8 against the window's 0.5 to 1\% and cost 16. It
therefore establishes only that no null below is a wiring fault, not that
behavior transfers to the window's rate regime.

Admitting on the published rule alone would make this bar circular, since
SPECTRE's published rule is the removal budget this paper identifies as the
failing half. Every arm is scored under both rules and admitted when either
recovers 0.70 on every seed rather than on the mean, which can clear 0.70 where
a seed does not. STRIP's own published rule, an entropy cut, is the only one it
defines: it clears CTU-13 there at 0.8872 but is undefined on UNSW-NB15, so no
single rule covers it on both datasets, an asymmetry we state rather than let a
column imply otherwise. The vision stage is blocking for every arm reported
here, since a null from a detector that never demonstrated it can find anything
is uninterpretable, and the tabular stage is blocking for any null reported as
a property of the published method rather than of our port.
Table~\ref{tab:vision-control} gives per-detector recall, and the controls that
did not pass are collected separately.

The 31.3\%-poison control sits on the far side of the one variable the
starvation argument turns on, so we repeat it at matched rates, reporting the
per-rate table with the artifact. The inherited budget fails on the vision
substrate itself at attacker-realistic poison, recall falling from 0.9809 to
0.1600, so the starvation window is not a property of tabular data and the
original vision evaluations did not meet it only because they never planted
poison that sparsely. Two differences travel with that reading. The ranking
also degrades on MNIST at these rates, to 0.8411 at 0.5\% where CTU-13 survives
at 0.9825, and the budget-free rule does not repair the vision case either,
recovering 0.2600 at 0.5\% and inverting to 0.1852 at 31.3\% because poison
occupying a third of a class is a mode rather than a tail. What reproduces on
vision is the budget's failure, not the intact-ranking-behind-a-starved-budget
pattern the tabular result shows.

The same control runs on UNSW-NB15 over its full 1{,}233{,}218-row benign
class, so no tabular null reported below for Spectral Signatures, Activation
Clustering or STRIP can be a wiring error. Scoring both rules turns two cells
into results rather than margins. The 0.7009 that makes Spectral Signatures'
UNSW-NB15 admission narrow is itself a starved budget, since the same runs
recover 0.9873 at worst under the budget-free rule at ranking AUC 0.9895,
against a CTU-13 ranking AUC of 0.9826. STRIP inverts that, ranking
near-perfectly on both datasets while the budget-free rule recovers nothing at
all on either.

\begin{table*}[!htb]
\centering\scriptsize
\caption{Positive controls, one row per detector. Vision recall is the mean of 5 seeds, with the worst seed in parentheses. Tabular recalls are the worst seed of 5. Neural Cleanse emits a flag, so its cell is the seed count on which it fires. Cells reading n/a are undefined for that detector, and not run means never measured. Loud-control ranking AUCs are reported with the artifact.}
\label{tab:vision-control}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & Vision & \multicolumn{2}{c}{CTU-13} & \multicolumn{2}{c}{UNSW-NB15} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Detector & recall & inherited & budget-free & inherited & budget-free \\
\midrule
\multicolumn{6}{@{}l}{\textit{Admitted}}\\
Spectral Signatures   & 0.9828 (0.9789) & 0.9981 & 1.0000 & 0.7009 & 0.9873 \\
Activation Clustering & 0.9922 (0.9760) & 0.9990 & n/a    & 0.9792 & n/a    \\
STRIP                 & 0.9945 (0.9767) & 0.9250 & 0.0000 & 0.7212 & 0.0000 \\
\midrule
\multicolumn{6}{@{}l}{\textit{Not admitted}}\\
SPECTRE               & 1.0000 (1.0000) & 0.0734 & 0.5325 & 0.0743 & 0.4835 \\
Neural Cleanse        & 1.0000 (1.0000) & 2 of 5 & n/a    & not run & n/a   \\
\bottomrule
\end{tabular}
\end{table*}
```


## Table: dataset card

Summarized in `04_methods.tex`; the full text and measurements follow.

```latex
\begin{table}[!htb]
\centering\small
\caption{Composition of the two datasets as used in this study.}
\label{tab:dataset-card}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{@{}p{2.2cm}p{2.5cm}p{2.5cm}@{}}
\toprule
Property & CTU-13 Neris & UNSW-NB15 \\
\midrule
Capture & Real botnet and background traffic, university network & IXIA PerfectStorm synthetic generator \\
Records used & $198{,}128$ & $1{,}584{,}259$ after dropping $695{,}831$ duplicates \\
Model-input features & 757 & 38 of a 49-field superset \\
Payload inspected & none & none \\
Label & binary botnet & binary, collapsed from 9 attack categories \\
Class balance & $\approx$98/2 overall; 0.74\% attack in test & 96.2/3.8 overall; 8.26\% attack in test \\
Split & TabularBench temporal, $143{,}046$ / $55{,}082$ & temporal 80/20, $1{,}267{,}407$ / $316{,}852$ \\
Attack rows in test & 407 (0.74\%) & $26{,}166$ (8.26\%) \\
Constraints & 360, published & 8 authored, 5 enforced \\
Manifest satisfied & train and test & train and test, rate 1.0 \\
Poison grid & $\{0.5,1,2,5,10\}\%$ $\times\,\{1,2,4,8,16\}$ & $\{0.5,1,5,10\}\%$ $\times\,\{8,16\}$ \\
\bottomrule
\end{tabular}
\end{table}
```


## Table: z_max inertness check

Summarized in `05_results.tex`; the full text and measurements follow.

```latex
\begin{table}[!htb]
\centering\small
\caption{The largest $z$ at which each arm's rule can still flag a row, at the anchor cell over five seeds. The isolation filter and the constraint departure failed their own pre-registered gates and enter as score sources, not as admitted detectors. An infinite $z_{\max}$ means no threshold makes the rule inert.}
\label{tab:zmax}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccc@{}}
\toprule
Arm & Bounded & Mean $z_{\max}$ & Inert \\
\midrule
Spectral Signatures  & no          & $3.4\times10^{5}$ & no \\
SPECTRE              & no          & $1.5\times10^{8}$ & no \\
Isolation filter     & no          & 38.7              & no \\
STRIP                & yes, at 0   & 0.79              & at all 3 $z$ \\
Constraint departure & yes, at 360 & infinite          & no \\
\bottomrule
\end{tabular}
\end{table}
```
