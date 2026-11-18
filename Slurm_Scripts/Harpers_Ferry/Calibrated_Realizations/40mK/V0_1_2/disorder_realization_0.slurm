#!/bin/bash

#SBATCH -J cal_40mK_V0_1_2_00       # Job Name
#SBATCH -N 1                        # Number of nodes (single node parallel pool)
#SBATCH -n 1                        # Number of tasks
#SBATCH -c 256                      # Cores per task (256 cores on Harpers Ferry AMD EPYC node)
#SBATCH -t 02:00:00                 # Time limit (2 hours; ~50-60 min expected on 255 workers)
#SBATCH -p day_3gb                  # Partition (day_3gb: 1 day limit, 3GB RAM/core)
#SBATCH -o cal_40mK_V0_1_2_00_%j.out # Standard output log
#SBATCH -e cal_40mK_V0_1_2_00_%j.err # Standard error log
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

CONFIG_REL="Calibrated_Realizations/40mK/V0_1_2/disorder_realization_0.yaml"
OUT_REL="Outputs/Data/Calibrated_Realizations/40mK/V0_1_2/disorder_realization_0/"

echo "=================================================================="
echo " Starting 40 mK Test: V0 = 1.2 meV, Realization 0"
echo " Node: $(hostname)"
echo " Start Time: $(date)"
echo " Cores Allocated: $SLURM_CPUS_PER_TASK"
echo " Configuration: Inputs/Parameters/${CONFIG_REL}"
echo " Temperatures: [40] mK | Baseline: highest G_n"
echo " Grid: 15,750 Points (mu: 0.0-4.5 meV, Vz: 0.0-1.4 meV)"
echo " Destination: ${OUT_REL}"
echo "=================================================================="

python -u main_parallel.py --config_file "$CONFIG_REL"

echo "=================================================================="
echo " Simulation Complete at: $(date)"
echo " Results saved to: ${OUT_REL}"
echo "=================================================================="

conda deactivate
