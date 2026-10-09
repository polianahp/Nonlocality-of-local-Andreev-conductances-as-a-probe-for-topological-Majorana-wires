#!/bin/bash 

#SBATCH -J 6um_V0_0_872_r2       # Job Name
#SBATCH -N 1                    # 1 node
#SBATCH -n 1                    # 1 master task
#SBATCH -c 256                  # 256 AMD EPYC cores
#SBATCH -t 15:00:00             # Time limit (15 hours for 6um wire)
#SBATCH -p day_3gb              # Partition
#SBATCH --mail-type=END,FAIL
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
cd $SCRATCH/nonlocalcond

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting 6-micron simulation: V0_0_872 realization 2..."
python -u main_parallel.py --config_path Parameters/six_micron/V0_0_872_realization_2.yaml

echo "Simulation finished!"
conda deactivate
