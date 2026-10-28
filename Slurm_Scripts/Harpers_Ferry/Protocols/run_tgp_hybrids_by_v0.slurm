#!/bin/bash
#SBATCH -J tgp_hybrids_v0            # Job Name
#SBATCH -N 1                         # 1 Node
#SBATCH -n 1                         # 1 Master task
#SBATCH -c 64                        # 64 Cores
#SBATCH -t 00:30:00                  # Walltime limit (30 minutes)
#SBATCH -p day_3gb                   # Partition (day_3gb)
#SBATCH --mail-type=END,FAIL         # Email notifications
#SBATCH --mail-user=nos00002@mix.wvu.edu
#SBATCH -o Logs/Slurm/tgp_hybrids_%x_%j.out
#SBATCH -e Logs/Slurm/tgp_hybrids_%x_%j.err

# ==============================================================================
# Harpers Ferry Single-V0 Slurm Runner: MSFT_TGP_Hybrids
# Usage:
#   sbatch --job-name=tgp_V0_0_378 run_tgp_hybrids_by_v0.slurm V0_0_378
#   sbatch --job-name=tgp_V0_0_645 run_tgp_hybrids_by_v0.slurm V0_0_645
#   sbatch --job-name=tgp_V0_0_872 run_tgp_hybrids_by_v0.slurm V0_0_872
#   sbatch --job-name=tgp_V0_1_2   run_tgp_hybrids_by_v0.slurm V0_1_2
# ==============================================================================

set -eo pipefail

V0_TARGET="${1:-V0_0_378}"

# 1. Initialize Conda Environment on Harpers Ferry
if [ -f /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh ]; then
    source /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh
elif [ -f /shared/software/conda/conda_init.sh ]; then
    source /shared/software/conda/conda_init.sh
else
    source ~/.bashrc
fi

conda activate kwant_fresh

SCRATCH_ROOT="${SCRATCH:-/scratch/nos00002}"
REPO_DIR="${SCRATCH_ROOT}/nonlocalcond"

if [ ! -d "$REPO_DIR" ]; then
    REPO_DIR="$(pwd)"
fi

cd "$REPO_DIR" || exit 1
mkdir -p Logs/Slurm Outputs/Microsoft_Hybrids_Post_Processing

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

NUM_WORKERS="${SLURM_CPUS_PER_TASK:-64}"

# Auto-detect data root: check Data/Disorder_Realizations first, then Outputs/Data/Disorder_Realizations
if [ -d "${REPO_DIR}/Data/Disorder_Realizations" ]; then
    DETECTED_DATA="${REPO_DIR}/Data/Disorder_Realizations"
elif [ -d "${REPO_DIR}/Outputs/Data/Disorder_Realizations" ]; then
    DETECTED_DATA="${REPO_DIR}/Outputs/Data/Disorder_Realizations"
elif [ -d "${SCRATCH_ROOT}/Data/Disorder_Realizations" ]; then
    DETECTED_DATA="${SCRATCH_ROOT}/Data/Disorder_Realizations"
else
    DETECTED_DATA="${REPO_DIR}/Data/Disorder_Realizations"
fi

DATA_ROOT="${2:-$DETECTED_DATA}"
OUTPUTS_ROOT="${3:-${REPO_DIR}/Outputs/Microsoft_Hybrids_Post_Processing}"

echo "=================================================================="
echo "Harpers Ferry TGP Hybrids V0 Run: $(date)"
echo "Target V0:           $V0_TARGET"
echo "Host Node:           $(hostname)"
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "Repository:          $REPO_DIR"
echo "Data Root:           $DATA_ROOT"
echo "Output Directory:    $OUTPUTS_ROOT"
echo "Allocated Cores:     $NUM_WORKERS"
echo "Temperatures:        30.0 mK, 40.0 mK"
echo "Force Recalculate:   True"
echo "=================================================================="

# Precompute any missing Pfaffian ground-truth masks for this V0 target
echo "[$(date)] Checking / precomputing ground-truth Pfaffian masks for $V0_TARGET across ${NUM_WORKERS} cores..."
python3 -u Scripts-Notebooks/precompute_pfaffian_masks.py \
    --data-root "$DATA_ROOT" \
    --v0 "$V0_TARGET" \
    --workers "$NUM_WORKERS"

python3 -u run_protocol_suite.py \
    --protocols MSFT_TGP_Hybrids \
    --v0 "$V0_TARGET" \
    --data-root "$DATA_ROOT" \
    --outputs-root "$OUTPUTS_ROOT" \
    --temperatures 30.0 40.0 \
    --workers "$NUM_WORKERS" \
    --force-recalculate \
    --keep-instances-file "${REPO_DIR}/Protocols/MSFT_TGP_Hybrids/top20_union_instances.txt"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Run failed with exit code $EXIT_CODE"
    conda deactivate
    exit $EXIT_CODE
fi

echo "=================================================================="
echo "Target $V0_TARGET Completed Successfully: $(date)"
echo "=================================================================="

conda deactivate
