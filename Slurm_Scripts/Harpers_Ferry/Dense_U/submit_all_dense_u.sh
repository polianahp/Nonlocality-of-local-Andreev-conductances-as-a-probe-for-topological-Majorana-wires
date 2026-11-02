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
echo "  Submitted Realization 2 -> Job ID: ${JOB2}"

echo ""
echo "All 3 realizations queued on Harpers Ferry. Monitor with:"
echo "  squeue -u $USER"
