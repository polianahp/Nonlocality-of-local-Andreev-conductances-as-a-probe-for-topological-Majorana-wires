#!/bin/bash 

#SBATCH -J dis_test_18      # Job Name
#SBATCH -N 1                  # Number of nodes (single node parallel pool)
#SBATCH -n 1                  # Number of tasks
#SBATCH -c 256                # Cores per task (256 cores on Harpers Ferry AMD EPYC node)
#SBATCH -t 10:00:00           # Time limit (10 hours)
#SBATCH -p day_3gb            # Partition (day_3gb: 1 day limit, 3GB RAM/core)
#SBATCH --mail-type=END,FAIL  # Notifications
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

echo "Starting simulation for disorder realization 18 (V0 = 0.645 meV)..."
python -u main_parallel.py --config_path Parameters/Disorder_Realizations/V0_0_645/disorder_realization_18.yaml

echo "Realization 18 finished!"
conda deactivate
