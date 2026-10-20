#!/bin/bash

#SBATCH -J dis_hf_10mK              # Job Name
#SBATCH -N 1                        # Number of nodes (single node parallel pool)
#SBATCH -n 1                        # Number of tasks
#SBATCH -c 256                      # Cores per task (256 cores on Harpers Ferry AMD EPYC node)
#SBATCH -t 06:00:00                 # Time limit (6 hours: ample buffer)
#SBATCH -p day_3gb                  # Partition (day_3gb: 1 day limit, 3GB RAM/core)
#SBATCH -o Tdispfaff5_10mK_%j.out   # Standard output log
#SBATCH -e Tdispfaff5_10mK_%j.err   # Standard error log
#SBATCH --mail-type=END,FAIL        # Email notifications
#SBATCH --mail-user=nos00002@mix.wvu.edu

# ==============================================================================
# Environment Setup on Harpers Ferry
# ==============================================================================
if [ -f /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh ]; then
    source /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh
elif [ -f /shared/software/conda/conda_init.sh ]; then
    source /shared/software/conda/conda_init.sh
else
    source ~/.bashrc
fi

conda activate kwant_fresh

# Navigate to project root in scratch space
if [ -d "$SCRATCH/nonlocalcond" ]; then
    cd "$SCRATCH/nonlocalcond"
elif [ -d "$SCRATCH/NonlocalProtocol" ]; then
    cd "$SCRATCH/NonlocalProtocol"
else
    echo "Warning: SCRATCH path not found, staying in $(pwd)"
fi

export PYTHONPATH=".:$PYTHONPATH"

# Thread Pinning: 1 thread per worker to prevent CPU oversubscription across 256 cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "=================================================================="
echo " Starting Simulation: Tdispfaff5_10mK (T = 10.0 mK, Upoints = 20)"
echo " Node: $(hostname)"
echo " Start Time: $(date)"
echo " Cores Allocated: $SLURM_CPUS_PER_TASK"
echo " Configuration: Inputs/Parameters/Tdis_pfaff5_10mK.yaml"
echo " Grid: 15,750 Points (mu: 0.0-4.5 meV, Vz: 0.0-1.4 meV)"
echo " Destination: Outputs/Data/Tdispfaff5_10mK/"
echo "=================================================================="

python -u main_parallel.py --config_path Parameters/Tdis_pfaff5_10mK.yaml

echo "=================================================================="
echo " Simulation Complete at: $(date)"
echo " Results saved to: Outputs/Data/Tdispfaff5_10mK/"
echo "=================================================================="

conda deactivate
