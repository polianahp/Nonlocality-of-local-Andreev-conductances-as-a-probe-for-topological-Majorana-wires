#!/bin/bash
# ==============================================================================
# submit_all.sh
# 
# Submits 6-micron nanowire simulation SLURM jobs to Harpers Ferry queue.
#
# Usage:
#   ./submit_all.sh             # Submits all 6-micron simulation jobs
#   ./submit_all.sh [FILTER]    # Submits jobs matching pattern (e.g. V0_0_645)
#   ./submit_all.sh [MAX_JOBS]  # Submits first N jobs (if integer)
#
# Examples:
#   ./submit_all.sh
#   ./submit_all.sh V0_0_645
#   ./submit_all.sh 10
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

ARG="$1"
COUNT=0
MAX_JOBS=999
FILTER=""

if [[ "$ARG" =~ ^[0-9]+$ ]]; then
    MAX_JOBS="$ARG"
elif [ -n "$ARG" ]; then
    FILTER="$ARG"
fi

echo "=========================================================="
echo " Submitting Harpers Ferry 6-Micron Wire Jobs"
if [ -n "$FILTER" ]; then
    echo " Filter: *$FILTER*"
fi
echo "=========================================================="

shopt -s nullglob
if [ -n "$FILTER" ]; then
    FILES=(run_*${FILTER}*.slurm)
else
    FILES=(run_*.slurm)
fi

TOTAL_FOUND=${#FILES[@]}

if [ "$TOTAL_FOUND" -eq 0 ]; then
    echo "No matching SLURM files found in $SCRIPT_DIR for pattern: ${FILTER:-run_*.slurm}"
    exit 1
fi

for slurm_file in "${FILES[@]}"; do
    if [ "$COUNT" -ge "$MAX_JOBS" ]; then
        break
    fi
    echo "[$((COUNT + 1))/$TOTAL_FOUND] Submitting $slurm_file..."
    sbatch "$slurm_file"
    COUNT=$((COUNT + 1))
    sleep 0.2  # Throttle submissions slightly to protect scheduler socket
done

echo ""
echo "Successfully queued $COUNT jobs for 6-micron wires to Harpers Ferry scheduler."
echo "Monitor queue with: squeue -u \$USER"
