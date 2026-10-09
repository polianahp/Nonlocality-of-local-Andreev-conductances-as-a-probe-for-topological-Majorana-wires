#!/bin/bash
# ==============================================================================
# run_harpers_ferry.slurm
# Production SLURM job template for Harpers Ferry cluster (AMD EPYC nodes)
# ==============================================================================

#SBATCH -J HF_Majorana_Sim          # Job name
#SBATCH -N 1                         # 1 full compute node
#SBATCH -n 1                         # 1 master task (multiprocessing pool)
#SBATCH -c 256                       # 256 CPU cores per node
#SBATCH -t 24:00:00                  # Maximum 24-hour wall-time limit
#SBATCH -p day_3gb                   # Partition (day_3gb: 1 day limit, 3GB/core = 768GB RAM)
#SBATCH --mail-type=BEGIN,END,FAIL   # Status email notifications
#SBATCH --mail-user=nos00002@mix.wvu.edu

# Initialize Conda Environment on Harpers Ferry
if [ -f /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh ]; then
    source /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh
elif [ -f /shared/software/conda/conda_init.sh ]; then
    source /shared/software/conda/conda_init.sh
else
    source ~/.bashrc
fi

conda activate kwant_fresh
cd $SCRATCH/nonlocalcond || exit 1

# ------------------------------------------------------------------------------
# Thread Pinning: 1 thread per worker to prevent CPU oversubscription across 250+ workers
# ------------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Configuration path passed as parameter $1, defaulting to standard benchmark
CONFIG_PATH=${1:-"Parameters/Disorder_Realizations/V0_0_645/disorder_realization_0.yaml"}

echo "=================================================================="
echo "Harpers Ferry Production Run Started at: $(date)"
echo "Host Node: $(hostname)"
echo "Config File: $CONFIG_PATH"
echo "Allocated CPU Cores: $SLURM_CPUS_PER_TASK"
echo "=================================================================="

# 1. Main Parallel Simulation (Kwant, BdG Spectrum, PDI, Pfaffian)
echo "Step 1: Running parallel transport & spectral simulation..."
python -u main_parallel.py --config_path "$CONFIG_PATH"
SIM_EXIT_CODE=$?

if [ $SIM_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Simulation failed with exit code $SIM_EXIT_CODE"
    conda deactivate
    exit $SIM_EXIT_CODE
fi

# 2. Parallel Post-Processing & Aggregation
echo "Step 2: Running automated post-processing and correlation aggregation..."
python -u -c '
import sys
from pathlib import Path
from src.parameter_handler import ConfigManager
from cut_analysis import analyze_dataset_cuts, PathConfigs

config_path = "'"$CONFIG_PATH"'"
cfg = ConfigManager.get_config(config_path=config_path)
data_dir = PathConfigs.DATA / cfg.dirname

print(f"Post-processing dataset: {data_dir}")
try:
    analyze_dataset_cuts(data_dir=data_dir)
    print("Post-processing successfully completed.")
except Exception as e:
    print(f"Warning during post-processing: {e}")
'

echo "=================================================================="
echo "Harpers Ferry Run Completed at: $(date)"
echo "=================================================================="

conda deactivate
