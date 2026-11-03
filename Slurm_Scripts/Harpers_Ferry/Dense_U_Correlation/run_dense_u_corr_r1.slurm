#!/bin/bash

#SBATCH -J dense_u_corr_r1      # Job Name: Tier 1 Low-Asymmetry Correlation vs Temp
#SBATCH -N 1                    # Number of nodes (single node parallel pool)
#SBATCH -n 1                    # Number of tasks
#SBATCH -c 256                  # Cores per task (256 cores on Harpers Ferry AMD EPYC node)
#SBATCH -t 06:00:00             # Time limit (6 hours: empirical runtime ~4.35h)
#SBATCH -p day_3gb              # Partition (day_3gb: 1 day limit, 3GB RAM/core)
#SBATCH --mail-type=END,FAIL    # Notifications
#SBATCH --mail-user=nos00002@mix.wvu.edu

# Initialize Conda environment on Harpers Ferry
if [ -f /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh ]; then
    source /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh
elif [ -f /shared/software/conda/conda_init.sh ]; then
    source /shared/software/conda/conda_init.sh
else
    source ~/.bashrc
fi

conda activate kwant_fresh
cd $SCRATCH/nonlocalcond

# Thread Pinning: 1 thread per worker to prevent CPU oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "========================================================================"
echo "Starting Tier 1: Low-Asymmetry Correlation vs Temp (51 U-pts, 2 baselines, 5 temps)"
echo "Disorder Realization: 1"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================================================"

python -u main_parallel.py --config Inputs/Parameters/dense_u_corr_temp_r1.yaml --workers 255

echo "Realization 1 finished successfully at $(date)!"
conda deactivate
