#!/bin/bash
# ==============================================================================
# submit_all.sh: Master Dispatcher for 3 μm Finite-T Barrier Sweep Campaign
# Submits jobs across all 5 disorder strengths (250 jobs total).
# Each job sweeps barriers at 30 mK & 40 mK with 43-point asymmetry ratio grid.
#
# Usage:
#   ./submit_all.sh [JOBS_PER_BATCH]   # default: 50
# ==============================================================================

JOBS_PER_BATCH=${1:-50}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

DISORDER_FOLDERS=("V0_0_1" "V0_0_378" "V0_0_645" "V0_0_872" "V0_1_2")

echo "=================================================================="
echo " MASTER SUBMISSION: 3 μm Nanowire Finite-T Barrier Sweep Campaign"
echo " 5 Disorder Strengths x $JOBS_PER_BATCH Realizations (250 jobs total)"
echo " Temperatures: [30.0 mK, 40.0 mK] (Unified Engine)"
echo " Total Target: $((JOBS_PER_BATCH * ${#DISORDER_FOLDERS[@]})) jobs"
echo "=================================================================="

for folder in "${DISORDER_FOLDERS[@]}"; do
    if [ -d "$folder" ] && [ -f "$folder/submit_all.sh" ]; then
        echo ""
        echo "=============================================================="
        echo ">>> Entering Strength Directory: $folder"
        echo "=============================================================="
        (cd "$folder" && ./submit_all.sh "$JOBS_PER_BATCH")
    fi
done

echo ""
echo "=================================================================="
echo " All 3 μm jobs dispatched across 5 disorder strengths!"
echo " Monitor queue with: squeue -u $USER"
echo "=================================================================="
