#!/usr/bin/env bash
# Sequential overnight runner, 2026-09-04.
#
# STRICTLY ONE JOB AT A TIME. Two CPU-bound jobs on this machine halved each other once
# (load average 23.9 on 12 cores, both roughly 3x slower), and every job here is
# CPU-dominated. Nothing in this file backgrounds anything.
#
# Each step is independent and checkpointed, so a kill costs at most one unit. Progress is
# the checkpoint file, not the log -- but every python call uses -u anyway, because
# `python | tee` block-buffers and a live run looks dead for minutes.
#
# Exit codes from the experiment scripts are MEANINGFUL, not failures to be retried:
#   0 = ran to completion
#   1 = a pre-registered gate refused to let it continue. That is the gate working.
# Neither case is retried and no parameter is tuned in response. A gate failure is
# recorded and the runner moves to the next step.

set -u
cd "$(dirname "$0")/.." || exit 1

PY=".venv/bin/python"
STATUS="results/overnight_20260904_status.json"
LOGDIR="logs"
mkdir -p "$LOGDIR"

started=$(date -Is)
declare -A RESULT

note() { printf '\n=== %s  [%s] ===\n' "$1" "$(date +%H:%M:%S)"; }

run_step() {
  local name="$1"; shift
  note "START $name"
  PYTHONPATH=. $PY -u "$@" > "$LOGDIR/${name}.log" 2>&1
  local code=$?
  RESULT[$name]=$code
  note "END $name exit=$code"
  tail -5 "$LOGDIR/${name}.log"
  write_status
}

write_status() {
  {
    printf '{\n  "_schema": "overnight_status/1",\n'
    printf '  "started": "%s",\n  "updated": "%s",\n' "$started" "$(date -Is)"
    printf '  "note": "exit 1 from an experiment means a pre-registered gate refused to continue, not a crash",\n'
    printf '  "steps": {\n'
    local first=1
    for k in "${!RESULT[@]}"; do
      [ $first -eq 1 ] || printf ',\n'
      printf '    "%s": %s' "$k" "${RESULT[$k]}"
      first=0
    done
    printf '\n  }\n}\n'
  } > "$STATUS"
}

# ---- 1. Isolation Forest: loud control gate, then the four window cells ----
run_step "95_isolation_forest" scripts/95_isolation_forest_window.py

# ---- 2. Boundary activation: falsifier, gate 0, then the window cells ----
# Runs regardless of step 1's verdict: they are independent detectors and a gate failure
# in one says nothing about the other.
run_step "96_boundary_activation" scripts/96_boundary_activation_window.py

# ---- 3. Full test suite, to prove nothing was broken along the way ----
note "START pytest"
PYTHONPATH=. $PY -u -m pytest tests/ -q > "$LOGDIR/pytest_after.log" 2>&1
RESULT[pytest]=$?
note "END pytest exit=${RESULT[pytest]}"
tail -3 "$LOGDIR/pytest_after.log"
write_status

note "ALL STEPS DONE"
for k in "${!RESULT[@]}"; do printf '  %-24s exit=%s\n' "$k" "${RESULT[$k]}"; done
