#!/bin/bash

#SBATCH -J Tdispfaff_roi_fixed          # Job Name
#SBATCH -N 1                            # Number of nodes (single node parallel pool)
#SBATCH -n 1                            # Number of tasks
#SBATCH -c 256                          # Cores per task (256 cores on Harpers Ferry AMD EPYC node)
#SBATCH -t 08:00:00                     # Time limit (previous local run was 4.57 hrs)
#SBATCH -p day_3gb                      # Partition (day_3gb: 1 day limit, 3GB RAM/core)
#SBATCH -o Tdispfaff_roi_fixed_%j.out   # Standard output log
#SBATCH -e Tdispfaff_roi_fixed_%j.err   # Standard error log
#SBATCH --mail-type=END,FAIL            # Email notifications
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

# Navigate to project root in scratch space (or submit dir if already in repo)
if [ -d "$SCRATCH/nonlocalcond" ]; then
    cd "$SCRATCH/nonlocalcond"
elif [ -d "$SCRATCH/NonlocalProtocol" ]; then
    cd "$SCRATCH/NonlocalProtocol"
elif [ -n "$SLURM_SUBMIT_DIR" ] && [ -f "$SLURM_SUBMIT_DIR/main_parallel.py" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    echo "Warning: SCRATCH repo path not found, staying in $(pwd)"
fi

export PYTHONPATH=".:$PYTHONPATH"

# Thread Pinning: 1 thread per worker to prevent CPU oversubscription across 256 cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

CONFIG_REL="Tdispfaff_roi_fixed.yaml"
OUT_REL="Outputs/Data/Tdispfaff_roi_fixed/"

echo "=================================================================="
echo " Starting Tdispfaff_roi Fixed-Code Rerun"
echo " Node: $(hostname)"
echo " Start Time: $(date)"
echo " Cores Allocated: $SLURM_CPUS_PER_TASK"
echo " Configuration: Inputs/Parameters/${CONFIG_REL}"
echo " Grid: 2,500 Points (mu: 1.5-3.5 meV, Vz: 0.4-0.9 meV)"
echo " Destination: ${OUT_REL}"
echo "=================================================================="

python -u main_parallel.py --config_file "$CONFIG_REL"

echo "=================================================================="
echo " Simulation Complete at: $(date)"
echo " Results saved to: ${OUT_REL}"
echo "=================================================================="

conda deactivate
