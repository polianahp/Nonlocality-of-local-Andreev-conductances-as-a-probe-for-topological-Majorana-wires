#!/bin/bash
#SBATCH -J tgp_hybrids_n50_256c      # Job Name
#SBATCH -N 1                         # 1 Full compute node
#SBATCH -n 1                         # 1 Master task
#SBATCH -c 256                       # 256 AMD EPYC cores for full realization-major parallelism
#SBATCH -t 00:30:00                  # Walltime limit (30 minutes; projected runtime < 5 mins)
#SBATCH -p day_3gb                   # Partition (day_3gb: max 1 day limit, 3GB/core = 768 GB RAM)
#SBATCH --mail-type=END,FAIL         # Email notifications
#SBATCH --mail-user=nos00002@mix.wvu.edu
#SBATCH -o Logs/Slurm/tgp_hybrids_n50_256c_%j.out
#SBATCH -e Logs/Slurm/tgp_hybrids_n50_256c_%j.err

# ==============================================================================
# Harpers Ferry Production Slurm Runner: MSFT_TGP_Hybrids N50 Suite
# Evaluates all 92 TGP Hybrid instances (44 original + 48 new LowCorr+Sym dyads)
# across the 4 physical disorder strengths (V0 = 0.378, 0.645, 0.872, 1.2 meV,
# excluding V0_0_1) and temperatures (30 mK, 40 mK) for the 50 complete disorder
# realizations (IDs 0..49) using the FPE/DPE Realization-Major Engine.
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
mkdir -p Logs/Slurm Outputs/Microsoft_Hybrids_Post_Processing Outputs/Postprocessing/MSFT_Hybrids_N50

# 3. Thread Pinning & Unbuffered Output
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

NUM_WORKERS="${SLURM_CPUS_PER_TASK:-256}"

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

DATA_ROOT="${1:-$DETECTED_DATA}"
OUTPUTS_ROOT="${2:-${REPO_DIR}/Outputs/Microsoft_Hybrids_Post_Processing}"
POSTPROCESSING_ROOT="${REPO_DIR}/Outputs/Postprocessing/MSFT_Hybrids_N50"

echo "=================================================================="
echo "Harpers Ferry TGP Hybrids Production Suite (N=50): $(date)"
echo "Host Node:           $(hostname)"
echo "Slurm Job ID:        ${SLURM_JOB_ID:-local}"
echo "Repository:          $REPO_DIR"
echo "Data Root:           $DATA_ROOT"
echo "Outputs Root:        $OUTPUTS_ROOT"
echo "Postprocessing Root: $POSTPROCESSING_ROOT"
echo "Disorder Strengths:  V0_0_378 V0_0_645 V0_0_872 V0_1_2 (Excluding V0_0_1)"
echo "Realization Cap:     50 (IDs 0..49)"
echo "Allocated Cores:     $NUM_WORKERS"
echo "Temperatures:        30.0 mK, 40.0 mK"
echo "Force Recalculate:   True"
echo "=================================================================="

# 3.5 Precompute any missing Pfaffian ground-truth masks across the 256 cores
echo "[$(date)] Checking / precomputing ground-truth Pfaffian masks across ${NUM_WORKERS} cores..."
python3 -u Scripts-Notebooks/precompute_pfaffian_masks.py \
    --data-root "$DATA_ROOT" \
    --v0 V0_0_378 V0_0_645 V0_0_872 V0_1_2 \
    --workers "$NUM_WORKERS"

# 4. Execute Realization-Major FPE/DPE Runner on complete N=50 realizations
echo "[$(date)] Executing Realization-Major runner for 92 instances across 50 realizations..."
python3 -u run_protocol_suite.py \
    --protocols MSFT_TGP_Hybrids \
    --data-root "$DATA_ROOT" \
    --outputs-root "$OUTPUTS_ROOT" \
    --v0 V0_0_378 V0_0_645 V0_0_872 V0_1_2 \
    --temperatures 30.0 40.0 \
    --max-realizations-per-v0 50 \
    --workers "$NUM_WORKERS" \
    --force-recalculate \
    --keep-instances-file "${REPO_DIR}/Protocols/MSFT_TGP_Hybrids/top20_union_instances.txt"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Protocol suite execution failed with exit code $EXIT_CODE"
    conda deactivate
    exit $EXIT_CODE
fi

# 5. Build Unified N=50 Leaderboards in Outputs/Postprocessing/MSFT_Hybrids_N50/
echo "[$(date)] Compiling unified N=50 leaderboards (92 protocol instances)..."
python3 -u build_n50_leaderboards.py \
    --src-root "$OUTPUTS_ROOT" \
    --dest-root "$POSTPROCESSING_ROOT"

# 6. Generate Top-10 Diagnostic Step-by-Step Passing Area Figures for 30 mK and 40 mK
echo "[$(date)] Generating diagnostic passing-area figures for Top 10 protocols at 30 mK..."
python3 -u plot_protocol_steps.py \
    --top-n 10 \
    --temperature 30.0 \
    --data-root "$DATA_ROOT" \
    --outputs-root "$OUTPUTS_ROOT" || true

echo "[$(date)] Generating diagnostic passing-area figures for Top 10 protocols at 40 mK..."
python3 -u plot_protocol_steps.py \
    --top-n 10 \
    --temperature 40.0 \
    --data-root "$DATA_ROOT" \
    --outputs-root "$OUTPUTS_ROOT" || true

echo "=================================================================="
echo "TGP Hybrids N50 Suite Completed Successfully: $(date)"
echo "Common Output Directory:  $OUTPUTS_ROOT"
echo "Postprocessing Directory: $POSTPROCESSING_ROOT"
echo "30 mK Leaderboard:        ${POSTPROCESSING_ROOT}/leaderboard_30mK.md"
echo "40 mK Leaderboard:        ${POSTPROCESSING_ROOT}/leaderboard_40mK.md"
echo "Executive Summary:        ${POSTPROCESSING_ROOT}/executive_summary.md"
echo "To download all post-processing results to your local machine, run:"
echo "rsync -avz nos00002@harpersferry.hpc.wvu.edu:${POSTPROCESSING_ROOT}/ /home/pseudonym/Documents/Code/NonlocalProtocol/Outputs/Postprocessing/MSFT_Hybrids_N50/"
echo "=================================================================="

conda deactivate
