#!/usr/bin/env bash
# Sequential chain for the four IEEE Access review-response experiments (2026-09-02).
#
# Serialized deliberately: this project shares one RTX 3060 Ti with coursework jobs, and every
# script below trains MLPs on CUDA. Running them concurrently would contend for VRAM and would
# also make any reported wall-clock meaningless.
#
# Every script is independently checkpointed (atomic temp-then-rename, config fingerprint), so
# killing this chain costs at most the one cell in flight. Re-running the chain resumes.
#
# Order is by unblocking value: 70 licenses the Section IV-G wiring-error claim, 69 settles the
# starved-budget-versus-destroyed-score question, 71 fills the one genuinely missing detector
# column, 72 supplies the out-of-selection replication.
set -u   # NOT -e: one failing script must not cancel the rest of the chain.

cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/chain_review_response.log

echo "=== chain start $(date -Is) ===" | tee -a "$LOG"

for s in \
  70_positive_control_strip_spectre \
  69_h3_window_auc \
  71_spectre_secondary \
  72_ctu_replication_seeds
do
  echo "" | tee -a "$LOG"
  echo "--- $s start $(date -Is) ---" | tee -a "$LOG"
  if "$PY" "scripts/${s}.py" >>"$LOG" 2>&1; then
    echo "--- $s OK $(date -Is) ---" | tee -a "$LOG"
  else
    echo "--- $s FAILED (exit $?) $(date -Is) --- continuing chain" | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
echo "=== chain done $(date -Is) ===" | tee -a "$LOG"
