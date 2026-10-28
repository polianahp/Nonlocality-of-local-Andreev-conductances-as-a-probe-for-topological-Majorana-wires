#!/bin/bash
# ==============================================================================
# Helper Script: Queue All 4 Physical Disorder Strengths as Independent Slurm Jobs
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/run_tgp_hybrids_by_v0.slurm"

V0_LIST=("V0_0_378" "V0_0_645" "V0_0_872" "V0_1_2")

echo "Submitting TGP Hybrids across 4 physical disorder strengths (excluding V0_0_1)..."

for V0 in "${V0_LIST[@]}"; do
    echo "Submitting: $V0"
    sbatch --job-name="tgp_${V0}" "$SLURM_SCRIPT" "$V0"
done

echo "All jobs submitted. Monitor with: squeue -u \$USER"
