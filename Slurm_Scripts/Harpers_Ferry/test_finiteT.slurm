#!/bin/bash

#SBATCH -J test_finiteT             # Job Name
#SBATCH -N 1                        # Number of nodes (single node parallel pool)
#SBATCH -n 1                        # Number of tasks
#SBATCH -c 256                      # Cores per task (256 cores on Harpers Ferry AMD EPYC node)
#SBATCH -t 00:30:00                 # Time limit (30 minutes: ample buffer for ~1-2 min test)
#SBATCH -p day_3gb                  # Partition (day_3gb: 1 day limit, 3GB RAM/core)
#SBATCH -o test_finiteT_%j.out      # Standard output log
#SBATCH -e test_finiteT_%j.err      # Standard error log
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
echo " Starting Finite-Temperature Validation Test on Harpers Ferry"
echo " Node: $(hostname)"
echo " Start Time: $(date)"
echo " Cores Allocated: $SLURM_CPUS_PER_TASK"
echo " Configuration: Inputs/Parameters/test_finiteT.yaml"
echo "=================================================================="

python -u main_parallel.py --config_path Parameters/test_finiteT.yaml

echo "=================================================================="
echo " Validation Test Completed at: $(date)"
echo " Verification target: Outputs/Data/test_finiteT/resolved_params.yaml"
echo "=================================================================="

conda deactivate
