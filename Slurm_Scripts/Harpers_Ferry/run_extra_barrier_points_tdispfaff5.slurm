#!/bin/bash

#SBATCH -J extra_u_tdispfaff5        # Job Name
#SBATCH -N 1                         # Single node parallel pool
#SBATCH -n 1                         # Single task
#SBATCH -c 256                       # 256 CPU cores (Harpers Ferry AMD EPYC node)
#SBATCH -t 02:00:00                  # Time limit: 2 hours (estimated run ~15-28 mins on 240 workers)
#SBATCH -p day_3gb                   # Partition: day_3gb (1 day limit, 3GB RAM/core)
#SBATCH -o extra_u_tdispfaff5_%j.out # Standard output log
#SBATCH -e extra_u_tdispfaff5_%j.err # Standard error log
#SBATCH --mail-type=END,FAIL         # Email notification
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

# Navigate to project directory in scratch space
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
echo " Starting Extra Low-Asymmetry Barrier U-Points: Tdispfaff5"
echo " Node: $(hostname)"
echo " Start Time: $(date)"
echo " Cores Allocated: $SLURM_CPUS_PER_TASK"
echo " Target Realization: Tdispfaff5_40mK (Ls = 300, 15,750 points)"
echo " Extra U-Points: 20 points in asymmetry window [-5.0, +5.0]"
echo " Workers: 240 parallel Kwant processes"
echo "=================================================================="

# Run 40 mK simulation with 240 workers and automatically merge with base dataset:
python -u run_extra_barrier_points_tdispfaff5.py \
    --data-dir Outputs/Data/Tdis_pfaff5_data/Tdispfaff5_40mK \
    --out-dir Outputs/Data/Tdis_pfaff5_data/Tdispfaff5_extra_u20 \
    --merged-dir Outputs/Data/Tdis_pfaff5_data/Tdispfaff5_40mK_dense \
    --asym-min -5.0 \
    --asym-max 5.0 \
    --num-u-points 20 \
    --temps 40.0 \
    --workers 240 \
    --merge

# Note: If evaluating BOTH 30 mK and 40 mK, replace --temps 40.0 with:
#   --temps 30.0 40.0

echo "=================================================================="
echo " Simulation and Merge Complete at: $(date)"
echo " Dense merged dataset: Outputs/Data/Tdis_pfaff5_data/Tdispfaff5_40mK_dense/"
echo "=================================================================="
