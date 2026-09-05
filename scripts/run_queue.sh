#!/usr/bin/env bash
# Sequential experiment queue for the 2026-09-04 run set.
#
# ONE JOB AT A TIME, deliberately. Two CPU-bound jobs on this machine halve each other, and the
# UNSW arms carry a 3.0-3.5 GB float64 SVD transient each (notes/20260731-bug-task5-oom-root-cause.md
# records a real kernel OOM kill at 9.96 GB max RSS). This box has 15 GB total. Running these
# concurrently is how the night gets lost, not how it gets shorter.
#
# Every job below checkpoints per unit and skips completed units on restart, so killing this script
# at any point costs at most the unit in flight. Re-running it resumes.
#
# The two --smoke jobs run FIRST on purpose. scripts/102 and scripts/103 have never been executed.
# A smoke run writes to its own _smoke paths, cannot touch a committed artifact, and surfaces an
# import or field-name error in minutes instead of after the multi-hour job ahead of it.
#
# Status lines go to logs/queue_status.txt, one per transition, for the watcher to read.
set -u

cd "$(dirname "$0")/.."
mkdir -p logs
STATUS="logs/queue_status.txt"
PY=".venv/bin/python"

exec 9> logs/.queue.lock
if ! flock -n 9; then
    echo "another queue is already running (logs/.queue.lock held) -- refusing to start a second"
    exit 1
fi

echo "QUEUE_RESTART $(date '+%Y-%m-%d %H:%M:%S')" >> "$STATUS"

run() {
    local name="$1"; shift
    local log="logs/queue_${name}.log"
    local t0 t1 rc
    # Resumable at the JOB level, not just the unit level. This queue has already been killed once
    # mid-flight (session teardown took the tmux server with it, 2026-09-04 17:47), and re-running
    # four completed jobs to reach the fifth wastes the machine. A job that has already recorded a
    # zero exit is skipped; anything else re-runs and resumes from its own per-unit checkpoint.
    if [ -f "$STATUS" ] && grep -q "^END $name rc=0 " "$STATUS"; then
        echo "SKIP $name (already completed) $(date '+%H:%M:%S')" >> "$STATUS"
        return 0
    fi
    t0=$(date +%s)
    echo "START $name $(date '+%H:%M:%S') $log" >> "$STATUS"
    PYTHONPATH=. "$PY" -u "$@" > "$log" 2>&1
    rc=$?
    t1=$(date +%s)
    echo "END $name rc=$rc secs=$((t1 - t0)) $(date '+%H:%M:%S') $log" >> "$STATUS"
    # No `set -e` and no early return: a failure in one job must not cancel the others. Each job is
    # independent and separately checkpointed, and a partial night is worth more than an empty one.
}

# 1. STRIP's positive control, rerun. The FRR rule's tie handling changed after the run on
# 2026-09-04 that reported recall 0.0000 at FPR 0.0000, so the committed rows are stale. Measured
# at 854 s.
run e1a_control70 scripts/70_positive_control_strip_spectre.py

# 2-3. Validate the never-run ports. Cheap, isolated output paths.
run smoke102 scripts/102_secondary_budget_multiplier_sweep.py --smoke
run smoke103 scripts/103_secondary_h3_window_auc.py --smoke

# 4. The UNSW timing anchor, deliberately early. No UNSW per-cell runtime exists anywhere in this
# repository, so every estimate for jobs 5 and 6 is a guess until this finishes. One seed, three
# cells. The full run in job 6 resumes these from the checkpoint rather than recomputing them.
run e2_102_seed42 scripts/102_secondary_budget_multiplier_sweep.py --seeds 42

# 5. E1b. STRIP under its published rule on both corpora. The long one: 20 CTU-scale cells plus
# 15 UNSW-scale cells, each a full retrain.
run e1b_strip62 scripts/62_strip_detector.py

# 6. E2 full.
run e2_102_full scripts/102_secondary_budget_multiplier_sweep.py

# 7-8. E3. The balanced arm keeps every minority row and matches the majority to it, so its
# training set is a small fraction of the imbalanced one and this is the cheap job.
run e3_103_seed42 scripts/103_secondary_h3_window_auc.py --seeds 42
run e3_103_full   scripts/103_secondary_h3_window_auc.py

echo "QUEUE_DONE $(date '+%Y-%m-%d %H:%M:%S')" >> "$STATUS"
