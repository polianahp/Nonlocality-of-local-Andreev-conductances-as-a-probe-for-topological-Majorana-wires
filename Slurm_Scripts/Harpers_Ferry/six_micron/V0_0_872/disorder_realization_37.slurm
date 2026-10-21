#!/bin/bash

#SBATCH -J dis_6um_V0_0_872_37  # Job Name
#SBATCH -N 1                            # Number of nodes (single node parallel pool)
#SBATCH -n 1                            # Number of tasks
#SBATCH -c 256                          # Cores per task (256 cores on Harpers Ferry AMD EPYC node)
#SBATCH -t 02:00:00                     # Time limit (2 hours for 6um multi-T full pipeline)
#SBATCH -p day_3gb                      # Partition (day_3gb: 1 day limit, 3GB RAM/core)
#SBATCH --mail-type=END,FAIL            # Notifications
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

echo "Starting simulation for 6um disorder realization 37 (V0 = 0.872 meV) on Harpers Ferry..."
python -u main_parallel.py --config_path Parameters/six_micron/V0_0_872/disorder_realization_37.yaml

echo "Realization 37 finished!"
conda deactivate
