#!/bin/bash

#SBATCH -J tf_min_0_872_r12
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH -t 04:30:00
#SBATCH -p comm_small_day
#SBATCH --mail-type=END,FAIL
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

echo "Starting minimal barrier sweeps for disorder realization 12 (V0_0_872) on Thorny Flat..."
python -u run_minimal_barrier_sweeps.py --config_path Inputs/Parameters/Thorny_Flat_Minimal/V0_0_872/disorder_realization_12.yaml

echo "Realization 12 finished successfully!"
conda deactivate
