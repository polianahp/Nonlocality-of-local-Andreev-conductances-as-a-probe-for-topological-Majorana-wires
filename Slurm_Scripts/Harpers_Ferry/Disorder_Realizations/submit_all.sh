#!/bin/bash
# ==============================================================================
# submit_all.sh: Master Dispatcher for 3 μm Finite-T Barrier Sweep Campaign
# Submits jobs across all 5 disorder strengths (V0_0_1, V0_0_378, V0_0_645, V0_0_872, V0_1_2).
#
# Usage:
#   ./submit_all.sh [START_INDEX] [NUM_JOBS] [OPTIONS]
#
# Examples:
#   ./submit_all.sh               # Default: start=50, count=50 (submits 50..99 across all 5 strengths = 250 jobs)
#   ./submit_all.sh 50            # Starts at 50, count=50 per strength (50..99)
#   ./submit_all.sh 50 50         # Starts at 50, count=50 per strength (50..99)
#   ./submit_all.sh 0 50          # Starts at 0, count=50 per strength (0..49)
#   ./submit_all.sh 100 50        # Starts at 100, count=50 per strength (100..149)
#   ./submit_all.sh 50 50 --dry-run  # Preview without submitting
#   ./submit_all.sh 50 50 --force    # Force re-submission of existing runs
# ==============================================================================

START_INDEX=50
NUM_JOBS=50
PASSTHROUGH_ARGS=()

POS_COUNT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start|-s)
            START_INDEX="$2"
            shift 2
            ;;
        --count|-n)
            NUM_JOBS="$2"
            shift 2
            ;;
        --dry-run|-d|--force|-f)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
        --help|-h)
            echo "Usage: ./submit_all.sh [START_INDEX] [NUM_JOBS] [--dry-run] [--force]"
            exit 0
            ;;
        *)
            if [ $POS_COUNT -eq 0 ]; then
                START_INDEX="$1"
                POS_COUNT=1
            elif [ $POS_COUNT -eq 1 ]; then
                NUM_JOBS="$1"
                POS_COUNT=2
            else
                PASSTHROUGH_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

DISORDER_FOLDERS=("V0_0_1" "V0_0_378" "V0_0_645" "V0_0_872" "V0_1_2")
END_INDEX=$((START_INDEX + NUM_JOBS - 1))

echo "=================================================================="
echo " MASTER SUBMISSION: 3 μm Nanowire Finite-T Barrier Sweep Campaign"
echo " 5 Disorder Strengths: ${DISORDER_FOLDERS[*]}"
echo " Realization Range: $START_INDEX to $END_INDEX ($NUM_JOBS jobs per strength)"
echo " Total Target: $((NUM_JOBS * ${#DISORDER_FOLDERS[@]})) jobs"
echo " Temperatures: [30.0 mK, 40.0 mK] (Unified Engine)"
echo " Safeguards: Skips runs already completed on disk or active in Slurm"
echo "=================================================================="

for folder in "${DISORDER_FOLDERS[@]}"; do
    if [ -d "$folder" ] && [ -f "$folder/submit_all.sh" ]; then
        echo ""
        echo "=============================================================="
        echo ">>> Entering Strength Directory: $folder"
        echo "=============================================================="
        (cd "$folder" && ./submit_all.sh "$START_INDEX" "$NUM_JOBS" "${PASSTHROUGH_ARGS[@]}")
    fi
done

echo ""
echo "=================================================================="
echo " Master dispatch completed across 5 disorder strengths!"
echo " Monitor queue with: squeue -u \$USER"
echo "=================================================================="
