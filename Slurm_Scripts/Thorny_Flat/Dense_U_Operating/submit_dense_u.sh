#!/bin/bash
# ==============================================================================
# submit_dense_u.sh
# 
# Submits calibrated Dense U Operating SLURM jobs to Thorny Flat cluster queue.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

for r in 3 4; do
    slurm_file="run_dense_u_r${r}.slurm"
    if [ -f "$slurm_file" ]; then
        echo "Submitting $slurm_file..."
        sbatch "$slurm_file"
        sleep 0.2
    fi
done

echo "Submission complete. Monitor queue with: squeue -u \$USER"
