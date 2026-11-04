#!/bin/bash
# ==============================================================================
# submit_thorny_flat_minimal.sh
#
# Submits 12 independent minimal barrier sweep jobs to Thorny Flat
# across 4 disorder strengths (V0_0_378, V0_0_645, V0_0_872, V0_1_2)
# and 3 realizations (10, 11, 12).
#
# Usage:
#   ./submit_thorny_flat_minimal.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "=================================================================="
echo " Submitting Thorny Flat Minimal Barrier Sweeps (12 jobs)"
echo " Partition: comm_small_day | Walltime: 04:30:00 | Cores: 40"
echo "=================================================================="

for slurm_file in run_tf_minimal_*.slurm; do
    if [ -f "$slurm_file" ]; then
        echo "Submitting: $slurm_file"
        sbatch "$slurm_file"
    fi
done

echo ""
echo "=================================================================="
echo " All 12 jobs submitted. Monitor queue with: squeue -u \$USER"
echo "=================================================================="
