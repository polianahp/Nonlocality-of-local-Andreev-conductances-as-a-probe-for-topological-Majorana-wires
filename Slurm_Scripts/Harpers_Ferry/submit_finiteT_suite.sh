#!/bin/bash
# ==============================================================================
# submit_finiteT_suite.sh (Harpers Ferry Finite-T Submission Master)
#
# Submits full 15,750-point simulations across the temperature sweep:
# 10 mK, 20 mK, 30 mK, 40 mK, 50 mK (all with Upoints = 20)
#
# Usage:
#   ./submit_finiteT_suite.sh          # Submits all 5 temperatures concurrently
#   ./submit_finiteT_suite.sh 10 30 50 # Submits only selected temperatures (mK)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

AVAILABLE_TEMPS=(10 20 30 40 50)
TARGET_TEMPS=("$@")

if [ ${#TARGET_TEMPS[@]} -eq 0 ]; then
    TARGET_TEMPS=("${AVAILABLE_TEMPS[@]}")
fi

echo "=================================================================="
echo " Submitting Finite-Temperature Simulations to Harpers Ferry"
echo " Cores per Job: 256 AMD EPYC (Partition: day_3gb)"
echo " Temperatures to submit: ${TARGET_TEMPS[*]} mK"
echo "=================================================================="

for T in "${TARGET_TEMPS[@]}"; do
    SCRIPT="run_Tdis_pfaff5_${T}mK.slurm"
    if [ -f "$SCRIPT" ]; then
        echo "Submitting $SCRIPT (T = ${T} mK)..."
        sbatch "$SCRIPT"
    else
        echo "Error: $SCRIPT not found!"
    fi
done

echo ""
echo "=================================================================="
echo " All selected jobs submitted. Monitor queue with: squeue -u \$USER"
echo "=================================================================="
