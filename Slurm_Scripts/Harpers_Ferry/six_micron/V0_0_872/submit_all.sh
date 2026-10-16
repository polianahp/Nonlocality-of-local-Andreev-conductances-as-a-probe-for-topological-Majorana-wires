#!/bin/bash
# ==============================================================================
# submit_all.sh
# 
# Submits disorder realization SLURM jobs for V0_0_872 (6 μm) to Harpers Ferry queue.
# Default submissions: 50 jobs (disorder_realization_0 through 49).
#
# Usage:
#   ./submit_all.sh [NUM_JOBS]
# Examples:
#   ./submit_all.sh      # Submits all 50 jobs (default)
#   ./submit_all.sh 10   # Submits first 10 jobs
# ==============================================================================

MAX_JOBS=${1:-50}
COUNT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

V0_NAME="$(basename "$SCRIPT_DIR")"

echo "=========================================================="
echo " Submitting Harpers Ferry 6 μm Jobs for $V0_NAME (Target: $MAX_JOBS jobs)"
echo "=========================================================="

for i in $(seq 0 $((MAX_JOBS - 1))); do
    slurm_file="disorder_realization_${i}.slurm"
    if [ ! -f "$slurm_file" ]; then
        echo "Warning: $slurm_file not found, skipping."
        continue
    fi
    echo "[$((COUNT + 1))/$MAX_JOBS] Submitting $slurm_file..."
    sbatch "$slurm_file"
    COUNT=$((COUNT + 1))
    sleep 0.2  # Throttle submissions slightly to protect scheduler socket
done

echo ""
echo "Successfully queued $COUNT jobs for $V0_NAME to Harpers Ferry scheduler."
echo "Monitor queue with: squeue -u \$USER"
