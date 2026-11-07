#!/bin/bash
# ==============================================================================
# submit_inas_pb.sh
# 
# Submits calibrated InAs-Pb (Delta_0 = 1.3 meV, Ls = 350) SLURM jobs 
# to the Thorny Flat cluster queue across Realizations 0, 1, and 2.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

for r in 0 1 2; do
    slurm_file="run_inas_pb_r${r}.slurm"
    if [ -f "$slurm_file" ]; then
        echo "Submitting $slurm_file..."
        sbatch "$slurm_file"
        sleep 0.2
    fi
done

echo "InAs-Pb batch submission complete. Monitor queue with: squeue -u \$USER"
