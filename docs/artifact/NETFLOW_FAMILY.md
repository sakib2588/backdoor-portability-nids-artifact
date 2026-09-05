# Activation Clustering on the corpus family: failure modes and scale control

Supporting detail for the appendix subsection "The detectors on the corpus family, against a
control on the same corpus". The appendix reports that Activation Clustering recovers 0.0063
of a blatant constraint-violating control on this family, and therefore that its cells there
are an unresolved control failure from which no portability verdict is drawn. This file
records what the detector does in those cells. None of it is evidence about whether the
detector ports.

## Two failure modes a mean recall cannot separate

The distinction decides whether an operator would notice. Across 75 interpretable cells the
detector isolates the poison in 41, collapses to flagging between 1 and 16 rows in 21, and
flags the wrong cluster in 13. The wrong-cluster cells take between 5.3% and 47.1% of clean
rows with them. Thirty-two of those 34 failing cells report a recall of exactly 0.000 and the
remaining two report 0.0635 and 0.0002, so per-cell recall is bimodal on 73 of the 75. That is
what a hard two-cluster assignment produces rather than a ranked score.

## A candidate mechanism for the wrong-cluster mode

Its silhouette sits near 0.5, far above the 0.125 at which the two-cluster structure would be
judged unreal, so the split is confident and genuine. It is not a split about poison. The
reading we cannot exclude, and do not test directly, is that the benign class on these corpora
carries dominant structure of its own and the two-cluster assumption lands on that structure.
The relative-size rule then flags the wrong half. The gate that exists to suppress a split that
is not real cannot fire, because this split is real.

## The subsample cap does not cause either failure

The target-class pool reaches 921,824 rows here against 111,666 on CTU-13, so detectors are
scored on a uniform subsample capped at 200,000 that preserves the poison fraction. Every
wrong-cluster cell was a subsampled one. We therefore re-ran both failure modes at full scale,
changing only the number of rows scored. The collapse still flags 49 to 52 rows at 921,600. The
wrong-cluster case still flags 349,767 to 356,472 at 768,000, between 45.5% and 46.4% of the
pool against 45.4% to 46.5% at the cap, and at the same silhouette. The cap does not cause
either failure.
