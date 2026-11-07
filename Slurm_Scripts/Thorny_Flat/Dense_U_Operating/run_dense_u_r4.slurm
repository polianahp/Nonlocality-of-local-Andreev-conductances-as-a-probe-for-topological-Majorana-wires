#!/bin/bash

#SBATCH -J dense_u_r4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH -t 02:30:00
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

echo "Starting Dense U Operating simulation for Realization 4 on Thorny Flat..."
python -u main_parallel.py Inputs/Parameters/fixing_code_r4.yaml

echo "Realization 4 finished successfully!"
conda deactivate
