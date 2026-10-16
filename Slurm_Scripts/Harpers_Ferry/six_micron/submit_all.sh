#!/bin/bash
# ==============================================================================
# submit_all.sh (Harpers Ferry Master Submission - 6 μm Suite)
#
# Submits batches across all 5 disorder strengths:
# V0_0_1, V0_0_378, V0_0_645, V0_0_872, V0_1_2
#
# Usage:
#   ./submit_all.sh [NUM_JOBS_PER_DISORDER]
# Examples:
#   ./submit_all.sh      # Submits 50 jobs per disorder strength (250 jobs total)
#   ./submit_all.sh 10   # Submits 10 jobs per disorder strength (50 jobs total)
# ==============================================================================

JOBS_PER_STRENGTH=${{1:-50}}

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

DISORDER_FOLDERS=("V0_0_1" "V0_0_378" "V0_0_645" "V0_0_872" "V0_1_2")

echo "=================================================================="
echo " Submitting Harpers Ferry 6 μm Master Batch: $JOBS_PER_STRENGTH realizations x 5 strengths"
echo " Total target: $((JOBS_PER_STRENGTH * ${{#DISORDER_FOLDERS[@]}})) jobs"
echo "=================================================================="

for folder in "${{DISORDER_FOLDERS[@]}}"; do
    if [ -d "$folder" ] && [ -f "$folder/submit_all.sh" ]; then
        echo ""
        echo ">>> Entering $folder..."
        (cd "$folder" && ./submit_all.sh "$JOBS_PER_STRENGTH")
    else
        echo "Warning: Subdirectory $folder or submit_all.sh missing."
    fi
done

echo ""
echo "=================================================================="
echo " Master 6 μm submission complete. Monitor queue with: squeue -u \$USER"
echo "=================================================================="
