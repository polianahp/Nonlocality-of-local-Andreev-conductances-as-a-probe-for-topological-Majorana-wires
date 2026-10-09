#!/bin/bash
# ==============================================================================
# submit_test_batch.sh
# 
# Submits a batch of test SLURM jobs to Harpers Ferry queue.
# Usage:
#   ./submit_test_batch.sh [NUM_JOBS]
# Examples:
#   ./submit_test_batch.sh 10   # Submit first 10 realization jobs
#   ./submit_test_batch.sh 50   # Submit all 50 test jobs (default)
# ==============================================================================

MAX_JOBS=${1:-50}
COUNT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "=========================================================="
echo " Submitting Harpers Ferry Test Batch (Target: $MAX_JOBS jobs)"
echo "=========================================================="

for slurm_file in disorder_realization_*.slurm; do
    if [ ! -f "$slurm_file" ]; then
        continue
    fi
    if [ "$COUNT" -ge "$MAX_JOBS" ]; then
        break
    fi
    echo "[$((COUNT + 1))/$MAX_JOBS] Submitting $slurm_file..."
    sbatch "$slurm_file"
    COUNT=$((COUNT + 1))
    sleep 0.2  # Throttle submissions slightly to protect scheduler socket
done

echo ""
echo "Successfully queued $COUNT test jobs to Harpers Ferry scheduler."
echo "Monitor queue with: squeue -u \$USER"
