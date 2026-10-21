#!/bin/bash
# ==============================================================================
# submit_all.sh: 6 μm Nanowire Unified Finite-T Suite (V0_0_872)
# Submits 50 jobs (each runs both 30 mK and 40 mK in a single execution).
#
# Usage:
#   ./submit_all.sh [MAX_JOBS]   # default: 50
# ==============================================================================

MAX_JOBS=${1:-50}
COUNT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "=================================================================="
echo " Submitting 6 μm Unified Jobs: V0_0_872 (Target: $MAX_JOBS jobs)"
echo " Temperatures: 30 mK & 40 mK | Walltime: 02:00:00 | Cores: 256"
echo "=================================================================="

for i in $(seq 0 $((MAX_JOBS - 1))); do
    slurm_file="disorder_realization_${i}.slurm"
    if [ ! -f "$slurm_file" ]; then
        echo "Warning: $slurm_file not found, skipping."
        continue
    fi
    echo "[$((COUNT + 1))/$MAX_JOBS] Submitting $slurm_file..."
    sbatch "$slurm_file"
    COUNT=$((COUNT + 1))
    sleep 0.2
done

echo ""
echo "Successfully queued $COUNT unified jobs for V0_0_872."
echo "Monitor queue with: squeue -u \$USER"
