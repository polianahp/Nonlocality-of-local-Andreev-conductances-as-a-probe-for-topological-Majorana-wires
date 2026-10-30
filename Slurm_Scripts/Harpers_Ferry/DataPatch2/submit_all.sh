#!/bin/bash
# ==============================================================================
# Master Dispatcher for Harpers Ferry Spectra & Localization DataPatch2 Campaign
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

DISORDER_FOLDERS=("V0_0_378" "V0_0_645" "V0_0_872" "V0_1_2")
END_INDEX=$((START_INDEX + NUM_JOBS - 1))

echo "=================================================================="
echo " MASTER SUBMISSION: Harpers Ferry Spectra & Localization (DataPatch2)"
echo " 4 Disorder Strengths: ${DISORDER_FOLDERS[*]}"
echo " Realization Range:    $START_INDEX to $END_INDEX ($NUM_JOBS jobs per strength)"
echo " Total Target:         $((NUM_JOBS * ${#DISORDER_FOLDERS[@]})) jobs"
echo " Cores per Job:        256 | Walltime: 00:30:00"
echo " Safeguards:           Skips runs that already have spectrum_arr.npy on disk"
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
echo " Master patch dispatch completed across 4 disorder strengths!"
echo " Monitor queue with: squeue -u $USER"
echo "=================================================================="
