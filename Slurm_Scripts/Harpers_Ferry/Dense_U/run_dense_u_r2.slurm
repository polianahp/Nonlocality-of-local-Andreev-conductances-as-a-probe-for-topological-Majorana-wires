#!/bin/bash

#SBATCH -J dense_u_r2           # Job Name
#SBATCH -N 1                    # Number of nodes (single node parallel pool)
#SBATCH -n 1                    # Number of tasks
#SBATCH -c 256                  # Cores per task (256 cores on Harpers Ferry AMD EPYC node)
#SBATCH -t 04:00:00             # Time limit (4 hours)
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
echo "Starting Calibrated Dense-U Multi-Temperature Suite on Harpers Ferry"
echo "Disorder Realization: 2"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================================================"

python -u run_calibrated_dense_u_suite.py --config Inputs/Parameters/dense_u_calibrated_r2.yaml --workers 255

echo "Realization 2 finished successfully at $(date)!"
conda deactivate
