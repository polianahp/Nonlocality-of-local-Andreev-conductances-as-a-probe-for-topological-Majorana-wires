#!/bin/bash

#SBATCH -J inas_pb_r1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH -t 01:00:00
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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting InAs-Pb Realization 1 simulation on Thorny Flat..."
python -u main_parallel.py Inputs/Parameters/InAs_Pb_r1.yaml

echo "InAs-Pb Realization 1 finished successfully!"
conda deactivate
