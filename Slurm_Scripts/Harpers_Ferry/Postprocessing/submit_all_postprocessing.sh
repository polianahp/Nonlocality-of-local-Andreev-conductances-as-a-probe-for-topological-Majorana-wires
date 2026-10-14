#!/bin/bash
# ==============================================================================
# submit_all_postprocessing.sh
# 
# Master submission script to launch post-processing jobs for all disorder strengths
# (V0_0_378, V0_0_645, V0_0_872, V0_1_2) to Harpers Ferry SLURM queue.
#
# Usage:
#   ./submit_all_postprocessing.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

SCRIPTS=(
    "post_process_V0_0_378.slurm"
    "post_process_V0_0_645.slurm"
    "post_process_V0_0_872.slurm"
    "post_process_V0_1_2.slurm"
)

echo "=================================================================="
echo " Submitting Harpers Ferry Post-Processing Batch across 4 Strengths"
echo " Target scripts: ${#SCRIPTS[@]} jobs"
echo "=================================================================="

COUNT=0
for s in "${SCRIPTS[@]}"; do
    if [ -f "$s" ]; then
        echo "[$((COUNT + 1))/${#SCRIPTS[@]}] Submitting $s..."
        sbatch "$s"
        COUNT=$((COUNT + 1))
        sleep 0.2
    else
        echo "Warning: Script $s not found, skipping."
    fi
done

echo ""
echo "=================================================================="
echo " Successfully queued $COUNT post-processing jobs to Harpers Ferry scheduler."
echo " Monitor queue with: squeue -u \$USER"
echo "=================================================================="
