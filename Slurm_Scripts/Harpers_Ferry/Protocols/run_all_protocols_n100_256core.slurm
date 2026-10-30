#!/bin/bash
#SBATCH -J nlp_suite_n100_256c       # Job Name
#SBATCH -N 1                         # 1 Full compute node
#SBATCH -n 1                         # 1 Master task
#SBATCH -c 256                       # 256 AMD EPYC cores for full realization-major parallelism
#SBATCH -t 00:30:00                  # Walltime limit (30 minutes; projected runtime ~2-3 mins)
#SBATCH -p day_3gb                   # Partition (day_3gb: max 1 day limit, 3GB/core = 768 GB RAM)
#SBATCH --mail-type=END,FAIL         # Email notifications
#SBATCH --mail-user=nos00002@mix.wvu.edu
#SBATCH -o Logs/Slurm/nlp_suite_n100_256c_%j.out
#SBATCH -e Logs/Slurm/nlp_suite_n100_256c_%j.err

# ==============================================================================
# Harpers Ferry Production Slurm Runner: Full Protocol Suite (N=100 Realizations)
# Evaluates all protocol families across 4 physical disorder strengths
# (V0 = 0.378, 0.645, 0.872, 1.2 meV, excluding V0_0_1) and temperatures
# (30 mK, 40 mK) for all 100 disorder realizations (IDs 0..99) using the
# Realization-Major FPE/DPE Engine.
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
    REPO_DIR="$(pwd)"
fi

cd "$REPO_DIR" || exit 1
mkdir -p Logs/Slurm Outputs/Protocols Outputs/Postprocessing/N100_Leaderboards

# 3. Strict Thread Pinning & Memory Discipline (Cap RAM < 400 MB per worker)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

NUM_WORKERS="${SLURM_CPUS_PER_TASK:-256}"

# Auto-detect data root
if [ -d "${REPO_DIR}/Data/Disorder_Realizations" ]; then
    DETECTED_DATA="${REPO_DIR}/Data/Disorder_Realizations"
elif [ -d "${REPO_DIR}/Outputs/Data/Disorder_Realizations" ]; then
    DETECTED_DATA="${REPO_DIR}/Outputs/Data/Disorder_Realizations"
elif [ -d "${SCRATCH_ROOT}/Data/Disorder_Realizations" ]; then
    DETECTED_DATA="${SCRATCH_ROOT}/Data/Disorder_Realizations"
else
    DETECTED_DATA="${REPO_DIR}/Outputs/Data/Disorder_Realizations"
fi

DATA_ROOT="${1:-$DETECTED_DATA}"
OUTPUTS_ROOT="${2:-${REPO_DIR}/Outputs/Protocols}"
POSTPROCESSING_ROOT="${REPO_DIR}/Outputs/Postprocessing/N100_Leaderboards"

echo "=================================================================="
echo "Harpers Ferry Complete Protocol Suite (N=100): $(date)"
echo "Host Node:           $(hostname)"
echo "Slurm Job ID:        ${SLURM_JOB_ID:-local}"
echo "Repository:          $REPO_DIR"
echo "Data Root:           $DATA_ROOT"
echo "Outputs Root:        $OUTPUTS_ROOT"
echo "Postprocessing Root: $POSTPROCESSING_ROOT"
echo "Disorder Strengths:  V0_0_378 V0_0_645 V0_0_872 V0_1_2 (Excluding V0_0_1)"
echo "Realization Count:   100 (IDs 0..99)"
echo "Allocated Cores:     $NUM_WORKERS"
echo "Temperatures:        30.0 mK, 40.0 mK"
echo "=================================================================="

# 4. Precompute any missing Pfaffian ground-truth masks across the 256 cores
echo "[$(date)] Checking / precomputing ground-truth Pfaffian masks across ${NUM_WORKERS} cores..."
python3 -u Scripts-Notebooks/precompute_pfaffian_masks.py \
    --data-root "$DATA_ROOT" \
    --v0 V0_0_378 V0_0_645 V0_0_872 V0_1_2 \
    --workers "$NUM_WORKERS"

# 5. Execute Realization-Major FPE/DPE Runner across all protocols
echo "[$(date)] Executing Realization-Major runner for all protocol configurations across 100 realizations..."
python3 -u run_protocol_suite.py \
    --data-root "$DATA_ROOT" \
    --outputs-root "$OUTPUTS_ROOT" \
    --v0 V0_0_378 V0_0_645 V0_0_872 V0_1_2 \
    --temperatures 30.0 40.0 \
    --workers "$NUM_WORKERS" \
    --keep-data auto \
    --force-recalculate

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Protocol suite execution failed with exit code $EXIT_CODE"
    conda deactivate
    exit $EXIT_CODE
fi

# 6. Build Master Outperforming TGP Leaderboards & Local Protocols Leaderboards
echo "[$(date)] Compiling master outperforming leaderboards..."
python3 -u build_unified_outperforming_leaderboard.py || true

echo "=================================================================="
echo "Protocol Suite N=100 Execution Completed Successfully: $(date)"
echo "Outputs Root:             $OUTPUTS_ROOT"
echo "Master Pooled:            ${OUTPUTS_ROOT}/master_protocols_leaderboard.md"
echo "30 mK Leaderboard:        ${OUTPUTS_ROOT}/master_protocols_leaderboard_30mK.md"
echo "40 mK Leaderboard:        ${OUTPUTS_ROOT}/master_protocols_leaderboard_40mK.md"
echo "=================================================================="

conda deactivate
