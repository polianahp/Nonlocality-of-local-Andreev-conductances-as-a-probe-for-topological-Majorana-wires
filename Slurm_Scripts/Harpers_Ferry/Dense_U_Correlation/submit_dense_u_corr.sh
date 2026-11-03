#!/bin/bash
# ==============================================================================
# submit_dense_u_corr.sh
# Master submission script for Tier 1: Low-Asymmetry Correlation vs Temp
# Submits jobs for Realizations 0, 1, and 2 to Harpers Ferry
# Each job runs in ~4.35 hours (#SBATCH -t 06:00:00) on partition day_3gb
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting Tier 1 Low-Asymmetry Correlation Jobs (R0, R1, R2)..."

JOB0=$(sbatch "${SCRIPT_DIR}/run_dense_u_corr_r0.slurm" | awk '{print $4}')
echo "Submitted Realization 0 -> Job ID: ${JOB0}"

JOB1=$(sbatch "${SCRIPT_DIR}/run_dense_u_corr_r1.slurm" | awk '{print $4}')
echo "Submitted Realization 1 -> Job ID: ${JOB1}"

JOB2=$(sbatch "${SCRIPT_DIR}/run_dense_u_corr_r2.slurm" | awk '{print $4}')
echo "Submitted Realization 2 -> Job ID: ${JOB2}"

echo ""
echo "All 3 jobs submitted. Track progress with: squeue -u $USER"
