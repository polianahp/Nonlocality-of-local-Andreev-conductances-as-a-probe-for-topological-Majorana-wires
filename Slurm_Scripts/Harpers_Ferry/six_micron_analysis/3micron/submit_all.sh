#!/bin/bash
# ==============================================================================
# submit_all.sh
# Submits simulation SLURM jobs to Harpers Ferry queue.
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
echo " Submitting Harpers Ferry Simulation Jobs"
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
    sleep 0.2
done

echo "Successfully submitted $COUNT / $TOTAL_FOUND jobs."
