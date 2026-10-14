#!/bin/bash

#SBATCH -J post_V0_1_2               # Job Name
#SBATCH -N 1                         # Single compute node
#SBATCH -n 1                         # Single master task
#SBATCH -c 64                        # Cores per task (64 AMD EPYC cores for parallel processing)
#SBATCH -t 01:00:00                  # Walltime limit (1 hour is ample for 50 realizations)
#SBATCH -p day_3gb                   # Partition (day_3gb: 1 day max limit, 3GB/core)
#SBATCH --mail-type=END,FAIL         # Email notifications
#SBATCH --mail-user=nos00002@mix.wvu.edu
#SBATCH -o %x_%j.out                 # Standard output log
#SBATCH -e %x_%j.err                 # Standard error log

# ==============================================================================
# Post-Processing Pipeline: V0 = 1.2 meV Ensemble
# Discovers disorder realizations, runs TGP Stage 2 & 3-step nonlocal protocol,
# computes FDR and Pfaffian coverage metrics, and extracts best/worst performers.
# ==============================================================================

# Initialize Conda Environment on Harpers Ferry
if [ -f /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh ]; then
    source /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh
elif [ -f /shared/software/conda/conda_init.sh ]; then
    source /shared/software/conda/conda_init.sh
else
    source ~/.bashrc
fi

conda activate kwant_fresh

# Ensure scratch and repo roots are set
SCRATCH_ROOT="${SCRATCH:-/scratch/nos00002}"
REPO_DIR="${SCRATCH_ROOT}/nonlocalcond"

if [ ! -d "$REPO_DIR" ]; then
    echo "ERROR: Repository directory not found at $REPO_DIR"
    exit 1
fi

cd "$REPO_DIR" || exit 1

# Thread Pinning: 1 thread per worker to eliminate core oversubscription in multiprocessing pool
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Directory configuration (allows CLI override: sbatch script.slurm [TARGET_DIR] [OUTPUT_DIR])
TARGET_DIR="${1:-${SCRATCH_ROOT}/nonlocalcond/Outputs/Data/Disorder_Realizations/V0_1_2}"
OUTPUT_DIR="${2:-${SCRATCH_ROOT}/nonlocalcond/Outputs/Postprocessing/V0_1_2}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-64}"

echo "=================================================================="
echo "Harpers Ferry Post-Processing Started: $(date)"
echo "Host Node:           $(hostname)"
echo "Target Directory:    $TARGET_DIR"
echo "Output Directory:    $OUTPUT_DIR"
echo "Allocated CPU Cores: $NUM_WORKERS"
echo "=================================================================="

# Run ensemble post-processing script
python -u post_process_disorder_realizations.py \
    --target_dir "$TARGET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers "$NUM_WORKERS" \
    --gap_th_factor 0.05 \
    --corr_threshold 0.85 \
    --ranking_metric fdr

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Post-processing failed with exit code $EXIT_CODE"
    conda deactivate
    exit $EXIT_CODE
fi

echo "=================================================================="
echo "Post-Processing Completed Successfully: $(date)"
echo "Ensemble summary and figures saved in: $OUTPUT_DIR"
echo "=================================================================="

conda deactivate
