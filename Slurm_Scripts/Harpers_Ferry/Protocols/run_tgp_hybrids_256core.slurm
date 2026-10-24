#!/bin/bash
#SBATCH -J tgp_hybrids_256c          # Job Name
#SBATCH -N 1                         # 1 Full compute node
#SBATCH -n 1                         # 1 Master task
#SBATCH -c 256                       # 256 AMD EPYC cores for full realization-major parallelism
#SBATCH -t 02:00:00                  # Walltime limit (2 hours is generous; runtime projected < 10 mins)
#SBATCH -p day_3gb                   # Partition (day_3gb: max 1 day limit, 3GB/core = 768 GB RAM)
#SBATCH --mail-type=END,FAIL         # Email notifications
#SBATCH --mail-user=nos00002@mix.wvu.edu
#SBATCH -o Logs/Slurm/tgp_hybrids_256c_%j.out
#SBATCH -e Logs/Slurm/tgp_hybrids_256c_%j.err

# ==============================================================================
# Harpers Ferry Production Slurm Runner: MSFT_TGP_Hybrids Protocol Family
# Evaluates all 44 TGP Hybrid instances across all disorder strengths (V0 = 0.378,
# 0.645, 0.872, 1.2 meV) and temperatures (30 mK, 40 mK) using the modern
# FPE/DPE Realization-Major Engine with shared Continuous Feature Caching.
# ==============================================================================

set -eo pipefail

# 1. Initialize Conda Environment on Harpers Ferry
if [ -f /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh ]; then
    source /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh
elif [ -f /shared/software/conda/conda_init.sh ]; then
    source /shared/software/conda/conda_init.sh
else
    source ~/.bashrc
fi

conda activate kwant_fresh

# 2. Repository & Scratch Directory Resolution
SCRATCH_ROOT="${SCRATCH:-/scratch/nos00002}"
REPO_DIR="${SCRATCH_ROOT}/nonlocalcond"

if [ ! -d "$REPO_DIR" ]; then
    # Fallback to current working directory if already cloned/run in place
    REPO_DIR="$(pwd)"
fi

cd "$REPO_DIR" || exit 1
mkdir -p Logs/Slurm Outputs/Protocols

# 3. Thread Pinning: 1 physical thread per worker process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

NUM_WORKERS="${SLURM_CPUS_PER_TASK:-256}"
DATA_ROOT="${1:-${REPO_DIR}/Outputs/Data/Disorder_Realizations}"
OUTPUT_NAME="Microsoft_Hybrids_Post_Processing"
OUTPUTS_ROOT="${2:-${REPO_DIR}/Outputs/${OUTPUT_NAME}}"

echo "=================================================================="
echo "Harpers Ferry TGP Hybrids Production Suite: $(date)"
echo "Host Node:           $(hostname)"
echo "Slurm Job ID:        ${SLURM_JOB_ID:-local}"
echo "Repository:          $REPO_DIR"
echo "Data Root:           $DATA_ROOT"
echo "Common Output Dir:   $OUTPUTS_ROOT"
echo "Allocated Cores:     $NUM_WORKERS"
echo "Temperatures:        30.0 mK, 40.0 mK"
echo "=================================================================="

# 4. Execute Realization-Major FPE/DPE Runner
python3 run_protocol_suite.py \
    --protocols MSFT_TGP_Hybrids \
    --data-root "$DATA_ROOT" \
    --outputs-root "$OUTPUTS_ROOT" \
    --temperatures 30.0 40.0 \
    --workers "$NUM_WORKERS" \
    --keep-instances-file "${REPO_DIR}/Protocols/MSFT_TGP_Hybrids/top20_union_instances.txt"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Protocol suite execution failed with exit code $EXIT_CODE"
    conda deactivate
    exit $EXIT_CODE
fi

# 5. Generate Top-10 Diagnostic Step-by-Step Passing Area Figures
echo "[$(date)] Generating diagnostic passing-area figures for Top 10 protocols..."
python3 plot_protocol_steps.py \
    --top-n 10 \
    --data-root "$DATA_ROOT" \
    --outputs-root "$OUTPUTS_ROOT" || true

echo "=================================================================="
echo "TGP Hybrids Suite Completed Successfully: $(date)"
echo "Common Output Directory: $OUTPUTS_ROOT"
echo "Master Leaderboard:      ${OUTPUTS_ROOT}/master_protocols_leaderboard.md"
echo "Leaderboard CSV:         ${OUTPUTS_ROOT}/master_protocols_leaderboard.csv"
echo "Summary Report:          ${OUTPUTS_ROOT}/MSFT_TGP_Hybrids/leaderboard_summary.md"
echo ""
echo "To download all post-processing results to your local machine, run:"
echo "rsync -avz nos00002@harpersferry.hpc.wvu.edu:${OUTPUTS_ROOT}/ ./Microsoft_Hybrids_Post_Processing/"
echo "=================================================================="

conda deactivate
