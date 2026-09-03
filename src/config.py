"""Central configuration: paths, seeds, and experiment constants.

Dependency-free (standard library only) so it can be imported before the ML
stack is installed. Every path and constant used across the pipeline resolves
from here; nothing downstream hardcodes a path or a seed.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- paths (all resolve from the project root, which is this file's grandparent) ---
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"

# --- reproducibility ---
# Workspace protocol: at least 5 seeds for every headline result.
SEEDS: tuple[int, ...] = (42, 123, 456, 789, 1337)

# --- experiment grid (research spec, Section 5) ---
POISON_RATES: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.10)
# Poison-rate denominator RESOLVED 2026-07-14 (was the M2 blocker): the training
# block actually passed to the poisoner, planted into the benign class (clean-label).
# CORRECTED 2026-08-01: that block is the POST-VALIDATION-SPLIT set of 114436 rows,
# NOT the 143046-row pre-split figure this comment used to quote. src/poison.py takes
# `poison_rate * len(x)` on the array it is handed, and 0.005 * 114436 = 572, which is
# the poison count every committed result records (results/detectors.json,
# results/evasion_ablation.json); 0.005 * 143046 would be 715 and matches nothing on
# disk. The stale 143046 figure had propagated into paper/sections/02_methods.tex and
# was caught by the 2026-08-01 pre-merge numeric audit. Literature-comparable;
# ~equals the benign-class denominator anyway since benign is ~98%.
# See notes/20260714-decision-m2-critical-review-resolutions.md.
POISON_DENOMINATOR = "train_set"

# --- M2 attack + split constants ---
ATTACK_TARGET = 0            # trigger makes botnet flows classified benign (evasion)
TRIGGER_COSTS: tuple[int, ...] = (1, 2, 4, 8, 16)  # number of SHAP-selected trigger features
TRAIN_ROWS = 143046          # TabularBench Ctu13Splitter temporal cut (test = remaining 55082)
VAL_FRAC = 0.2               # stratified val slice carved from the training block
VAL_SPLIT_SEED = 1319        # matches Ctu13Splitter's random_state
CONSTRAINT_TOL = 1e-6        # float tolerance for the 2 byte-conservation equalities

# --- dataset expectations (verified at load time in M0, not trusted blindly) ---
CTU_N_CONSTRAINTS_EXPECTED = 360  # TabularBench CTU botnet linear constraint count
CTU_IMBALANCE_NOTE = "~98/2 benign/malicious"  # corrected 2026-07-13 M0 data check; was ~99.3/0.7

# --- statistics (research spec, Section 5) ---
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CI = 0.95
BOOTSTRAP_DF = 4  # n=5 seeds -> df=4; no parametric tests at n<=30

# --- extension preregistration (2026-07-29) ---
# LOCKED before any extension outcome was inspected. Protocol and rationale:
# notes/20260729-decision-extension-preregistration.md. Do not retune these after seeing a
# result -- the names are deliberately verbose so a post-hoc change shows up in a diff.
EXTENSION_BASE_SEEDS: tuple[int, ...] = SEEDS
EXTENSION_REPLICATION_SEEDS: tuple[int, ...] = (2026, 31415, 27182, 16180, 11235)
SECONDARY_RATES: tuple[float, ...] = (0.005, 0.01, 0.05, 0.10)
SECONDARY_COSTS: tuple[int, ...] = (8, 16)
SPECTRAL_COMPONENTS = 5
MAD_Z_PRIMARY = 3.0
ATTACK_EFFECTIVE_ASR = 0.8
FIXED_BUDGET_MISS_RECALL = 0.5
CLEAN_ACCURACY_DROP_MAX = 0.02
ADAPTIVE_SEARCH_MAX_CANDIDATES = 64
# Calibration/held-out partition. Binding on every arm that selects a threshold, candidate, or
# attacker configuration: selection sees SURROGATE seeds only, never EVAL.
ADAPTIVE_SURROGATE_SEEDS: tuple[int, ...] = (42, 123, 456)
ADAPTIVE_EVAL_SEEDS: tuple[int, ...] = (789, 1337)

# --- secondary-dataset scoping (extension Task 3, 2026-07-31) ---
# Named dataset-id constants so the "must never silently return CTU-13 in place of the secondary
# dataset" guard in `src.data_secondary.load_secondary_setup` (and its unit test) reference one
# definition rather than duplicating the literal string in two files.
PRIMARY_DATASET_ID = "ctu_13_neris"
SECONDARY_DATASET_ID = "unsw_nb15"

# --- detector-repair gates (Tasks 7A/7B; pre-registered, see the note above) ---
AC_REPAIR_MIN_RECALL_GAIN = 0.20      # centroid repair must beat canonical AC by this much
AC_REPAIR_MAX_CLEAN_FPR = 0.01
AC_SPECTRAL_INDEPENDENCE_MAX_RHO = 0.90  # at/above this the score is Spectral-redundant, not AC
NC_REPAIR_MIN_DETECTION_GAIN = 0.10
NC_REPAIR_MAX_CLEAN_FPR = 0.01
CLEAN_CALIBRATION_QUANTILE = 0.99     # tau / distance cutoff from clean models only
NC_MIN_CONSTRAINT_VALID_FRACTION = 355 / 358  # see notes/20260802-decision-nc-track-extension-preregistration.md

# Task 7B defender-side split sizes. Fixed before any repair outcome was inspected. The benign
# training pool is ~140k rows, so holding back 20k still leaves ~120k suspect rows and every poison
# count (>=715 at rate 0.005) lands in the suspect set. AC_REFERENCE_SEED_OFFSET keeps this draw on a
# separate RNG stream from the poison draw for the same seed.
AC_REFERENCE_SIZE = 10_000
AC_CALIBRATION_SIZE = 10_000
AC_REFERENCE_SEED_OFFSET = 90_001
AC_PCA_COMPONENTS = 10   # canonical AC's reduction; the repair keeps the recipe shape
AC_N_CLUSTERS = 2

# --- Task 7B matched-FPR evaluation (2026-07-31 amendment) ---
# The original AC_REPAIR_MAX_CLEAN_FPR = 0.01 was inconsistent with this project's own accepted
# baseline: Spectral under MAD z=3.0 achieves recall 1.0 at an implied clean FPR of mean 0.0596
# (range 0.0292-0.0709) across all 25 rows of results/evasion_ablation.json, committed 2026-07-15.
# Holding the AC repair to a budget 6x stricter than its comparator answers no question. The primary
# gate is therefore a MATCHED-FPR comparison against canonical AC's own realised FPR, and recall is
# reported at every budget below so the trade-off curve is visible rather than a single chosen point.
SPECTRAL_ACCEPTED_FPR_MEAN = 0.0596      # measured, not chosen; see the note above
AC_REPORTED_FPR_BUDGETS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.06)
AC_MATCHED_FPR_MIN_RECALL_GAIN = 0.20    # same 0.20 as before, now at matched FPR rather than at 1%

# Task 7A clean-model replicas, added by the 2026-07-30 amendment, raised again by the 2026-08-02 NC
# track extension (see notes/20260802-decision-nc-track-extension-preregistration.md Section 2). The
# false-alarm rate estimated from k clean models is quantised to 1/k. 10 replicas per calibration seed
# gave ~30 models and resolution ~0.033 -- still short of the 0.01 gate, which the artifact recorded
# rather than papering over. 34 replicas per calibration seed gives 3*34=102 models, resolution
# 1/102 ~= 0.0098, which clears NC_REPAIR_MAX_CLEAN_FPR = 0.01 (the smallest replica count that does,
# with margin). Replica models differ only in init seed; the data and recipe are identical.
NC_CLEAN_REPLICAS = 34
NC_REPLICA_SEED_STRIDE = 7_919   # derives replica init seeds; prime, to avoid colliding with SEEDS

# Amended Task 7A validity gate (2026-07-30). Exact feasibility is unreachable for a gradient-inverted
# pattern: CTU has two byte-conservation EQUALITIES and the inversion moves all 757 features, while
# project_to_feasible repairs inequalities only. So the gate is: projectable (inequality) constraints
# valid after projection, AND violation magnitude cut by at least this factor versus baseline.
NC_VIOLATION_REDUCTION_MIN_FACTOR = 2.0

# --- AC track (2026-08-03). Pre-registered in notes/20260803-decision-ac-track-preregistration.md
# BEFORE any candidate ran. Canonical AC's 2-means split is the diagnosed failure mode: 12 of 20
# evasion rows carry zero minority purity (results/ac_mechanism.json) and the silhouette diagnostic
# ANTI-correlates with recall (rho = -0.5549). These constants parameterise the repair attempt.
#
# k=3 and k=10 are not arbitrary. Both were tried once informally and recorded in
# notes/20260715-finding-ac-fails-tabular-scale.md (k=3 -> poison purity 0.62, recall 0.998;
# k=10 -> purity 0.78, recall 0.97), but AT THE LOUD CONTROL CELL ONLY, never on the evasion window
# and never with CIs across seeds. This grid re-runs them properly; k=5 interpolates.
AC_K_GRID: tuple[int, ...] = (2, 3, 5, 10)
AC_CLUSTER_ALGORITHMS: tuple[str, ...] = ("kmeans", "gmm")

# Midpoint of Chen et al.'s own stated 0.10-0.15 silhouette range. Argued from their paper, NOT
# fitted to any CTU number.
AC_SILHOUETTE_GATE_THRESHOLD = 0.125

# Scoped to the k=1-vs-k=2 question only, mirroring Chen et al.'s own narrow use AND their own
# reported negative finding for it. Confirmatory, not a k-selection procedure.
AC_GAP_STATISTIC_B = 20

# ExRe tests ONLY the relative-size-flagged minority cluster, at k=2 only. Chen et al.'s method
# assumes clusters of comparable retrainable size; this dataset splits 52-vs-111614 (seed 42) and
# 30-vs-91636 (seed 1337, Task 7B control). Excluding the MAJORITY and retraining on ~100 rows would
# measure sample starvation, not backdoor removal. Argued from the diagnosed mechanism, not fitted.
AC_EXRE_CANDIDATE_K = 2
AC_EXRE_ALGORITHMS: tuple[str, ...] = ("kmeans", "gmm")
AC_EXRE_T_DEFAULT = 1.0          # Chen et al.'s published constant, kept as NC keeps tau=2.0

# 10 replicas x 3 calibration seeds = 30 clean models, resolution 1/30 ~= 0.033. Deliberately
# retained and REPORTED as too coarse for a 1% absolute gate -- the AC gate direction is matched-FPR
# (AC_MATCHED_FPR_MIN_RECALL_GAIN above), not 1% absolute. Any future 1%-absolute claim on AC must
# raise this count first, exactly as NC_CLEAN_REPLICAS went 10 -> 34.
AC_EXRE_CLEAN_REPLICAS = 10
AC_EXRE_REPLICA_SEED_STRIDE = 6_133   # prime, distinct from NC_REPLICA_SEED_STRIDE = 7_919


# ---------------------------------------------------------------------------
# NetFlow benign-share manipulation and multi-corpus replication (2026-09-03).
# See notes/20260903-decision-netflow-multicorpus-preregistration.md and
# notes/20260903-decision-netflow-plan-adversarial-review.md
# ---------------------------------------------------------------------------

# Read-only source. Belongs to the ids-compression-benchmark project; never written to.
#
# Resolved from the environment at import, defaulting to a path inside this repository, so that no
# machine-specific absolute path is baked into a tracked file. This is not tidiness:
# scripts/77_build_artifact_repo.py refuses to export any file carrying an authoring-machine path,
# and the hardcoded path that used to sit here is what blocked the artifact release, leaving the
# corpus-family code out of the published artifact while its results were in the manuscript.
#
# On a machine that holds the corpus elsewhere, do either of:
#     export NETFLOW_PARQUET=/abs/path/to/NF-UQ-NIDS-v2.parquet
#     ln -s /abs/path/to/NF-UQ-NIDS-v2.parquet data/raw/nf_uq/NF-UQ-NIDS-v2.parquet
#
# The default sits under data/raw/ rather than a new directory because .gitignore already ignores
# data/raw/*, so the symlink cannot be committed by accident, and "raw" is what a source corpus is.
#
# Pointing this at the wrong file cannot pass silently. src/data_netflow.py keys its cached row-group
# index on the source's size and mtime, and NETFLOW_POPULATION below is asserted as a gate on the
# per-corpus row counts, so a substituted corpus fails loudly instead of being studied by mistake.
NETFLOW_PARQUET = os.environ.get(
    "NETFLOW_PARQUET",
    str(DATA_RAW / "nf_uq" / "NF-UQ-NIDS-v2.parquet"),
)

NETFLOW_CORPORA: tuple[str, ...] = (
    "NF-BoT-IoT-v2",
    "NF-ToN-IoT-v2",
    "NF-CSE-CIC-IDS2018-v2",
    "NF-UNSW-NB15-v2",
)

# Measured 2026-09-03 from the Dataset column: (total rows, benign rows). Used as a gate assertion,
# so a changed source parquet fails loudly instead of silently studying different data, and as the
# population class ratio, which the loader must not have to infer.
NETFLOW_POPULATION: dict = {
    "NF-BoT-IoT-v2":         (37_763_497,    135_037),
    "NF-ToN-IoT-v2":         (16_940_496,  6_099_469),
    "NF-CSE-CIC-IDS2018-v2": (18_893_708, 16_635_567),
    "NF-UNSW-NB15-v2":       ( 2_390_275,  2_295_222),
}

NETFLOW_SAMPLE_ROWS = 1_200_000   # held constant at every gradient level and every corpus
NETFLOW_TRAIN_FRAC = 0.8
NETFLOW_SPLIT_SEED = 20260903

# Q1, the manipulation. Verified 2026-09-03: only these two corpora hold enough rows of both classes
# to reach every level at NETFLOW_SAMPLE_ROWS.
NETFLOW_MANIPULATION_CORPUS = "NF-CSE-CIC-IDS2018-v2"
NETFLOW_REPLICATION_CORPUS = "NF-ToN-IoT-v2"
NETFLOW_BENIGN_SHARES: tuple[float, ...] = (0.05, 0.20, 0.40, 0.60, 0.80, 0.96)

NETFLOW_RATE = 0.01                                  # fixed for the manipulation
NETFLOW_COSTS: tuple[int, ...] = (4, 16)
NETFLOW_PRIMARY_COST = 16                            # the cost the primary statistic is read at

# Q2, the native-balance replication.
NETFLOW_NATIVE_RATES: tuple[float, ...] = (0.005, 0.01, 0.02)

# Excluded from the property analysis by pre-registration, not by result. 3,433 benign training rows
# under 279:1 inverse-frequency weighting is memorization, not a learned benign concept.
NETFLOW_PROPERTY_EXCLUDED: tuple[str, ...] = ("NF-BoT-IoT-v2",)

NETFLOW_EXCLUDED_COLUMNS: tuple[str, ...] = (
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
    "Label", "Attack", "Dataset",
)

# NF-UQ-NIDS-v2 ships physically impossible values in its two bytes-per-second columns, in every
# one of the four corpora: measured maxima 8.9e304 (CSE-CIC), 1.9e219 (ToN-IoT), 8.6e22 (BoT-IoT)
# and 1.6e15 (UNSW-NB15). They arise from dividing a byte count by a flow duration that can be ~0.
# Squaring them overflows float64 inside StandardScaler, which sets that column's variance to NaN and
# turns EVERY row of it into NaN, collapsing the victim to a constant prediction. The cap is a line
# rate, not a fitted statistic: 100 Gbps expressed in bytes per second. The two AVG_THROUGHPUT
# columns are measured at most 4.3 Gbps and are deliberately NOT capped -- only the two columns whose
# denominator can vanish are affected. See notes/20260903-bug-netflow-second-bytes-overflow.md
NETFLOW_RATE_CAP_BYTES_PER_S = 1.25e10
NETFLOW_RATE_CAPPED_COLUMNS: tuple[str, ...] = (
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
)

NETFLOW_RELATION_ADMISSION_FRAC = 0.999

# Pre-registered guards.
NETFLOW_SATURATION_ASR = 0.95      # clean-model stamped ASR at/above this: cell is `saturated`
# attack-effectiveness gate reuses the existing ATTACK_EFFECTIVE_ASR = 0.8

# Pre-registered decision thresholds for H-NF1.
NETFLOW_RHO_CONFIRM = 0.8
NETFLOW_RHO_REFUTE = 0.4


def ensure_dirs() -> None:
    """Create the output directories if absent. Idempotent."""
    for d in (DATA_RAW, DATA_PROCESSED, CHECKPOINTS, RESULTS, FIGURES, LOGS):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"ROOT = {ROOT}")
    print(f"SEEDS = {SEEDS}")
    print(f"POISON_RATES = {POISON_RATES}")
    print(f"CTU_N_CONSTRAINTS_EXPECTED = {CTU_N_CONSTRAINTS_EXPECTED}")
