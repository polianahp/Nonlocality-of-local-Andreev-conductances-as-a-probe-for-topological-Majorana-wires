#!/bin/bash

#SBATCH -J dis_V0_0_645_139     # Job Name
#SBATCH -N 1                            # Number of nodes
#SBATCH -n 1                            # Number of tasks
#SBATCH -c 40                           # Cores (40 cores per node on Thorny Flat)
#SBATCH -t 02:00:00                     # Time limit (2 hours)
#SBATCH -p comm_small_day               # Queue partition on Thorny Flat
#SBATCH --mail-type=END,FAIL            # Notifications
#SBATCH --mail-user=nos00002@mix.wvu.edu

# Initialize Conda environment on Thorny Flat
if [ -f /shared/software/conda/conda_init.sh ]; then
    source /shared/software/conda/conda_init.sh
elif [ -f /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh ]; then
    source /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh
else
    source ~/.bashrc
fi

conda activate kwant_fresh
cd $SCRATCH/nonlocalcond

# Thread Pinning: 1 thread per worker to prevent CPU oversubscription in multiprocessing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting simulation for disorder realization 139 (V0 = 0.645 meV) on Thorny Flat..."
python -u main_parallel.py --config_path Parameters/Disorder_Realizations/V0_0_645/disorder_realization_139.yaml

echo "Realization 139 finished!"
conda deactivate
