#!/bin/bash
# ==============================================================================
# Helper submission script for Calibrated Dense-U Multi-Temperature Suite
# Harpers Ferry HPC Cluster (WVU)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting Calibrated Dense-U production jobs to SLURM..."

JOB0=$(sbatch "${SCRIPT_DIR}/run_dense_u_r0.slurm" | awk '{print $4}')
echo "  Submitted Realization 0 -> Job ID: ${JOB0}"

JOB1=$(sbatch "${SCRIPT_DIR}/run_dense_u_r1.slurm" | awk '{print $4}')
echo "  Submitted Realization 1 -> Job ID: ${JOB1}"

JOB2=$(sbatch "${SCRIPT_DIR}/run_dense_u_r2.slurm" | awk '{print $4}')
echo "  Submitted Realization 2 (Calibrated) -> Job ID: ${JOB2}"

# Optional: Submit legacy hardcoded comparison run for Realization 0
if [[ "$1" == "--with-hardcoded" ]]; then
    JOB_HC=$(sbatch "${SCRIPT_DIR}/run_dense_u_r0_hardcoded.slurm" | awk '{print $4}')
    echo "  Submitted Realization 0 (Legacy Hardcoded) -> Job ID: ${JOB_HC}"
fi

echo ""
echo "Jobs queued on Harpers Ferry. Monitor with:"
echo "  squeue -u $USER"
echo ""
echo "To also submit the legacy hardcoded comparison run, pass --with-hardcoded:"
echo "  bash ${SCRIPT_DIR}/submit_all_dense_u.sh --with-hardcoded"
