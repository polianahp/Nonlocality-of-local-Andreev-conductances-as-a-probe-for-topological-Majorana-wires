#!/bin/bash
#SBATCH -N 1
#SBATCH -c 256
#SBATCH -t 01:15:00
#SBATCH -p compute
#SBATCH --job-name=3um_V0_1_2_r39
#SBATCH --output=3um_V0_1_2_r39_%j.out
#SBATCH --error=3um_V0_1_2_r39_%j.err

# ==============================================================================
# Harpers Ferry SLURM Job Script: 3 μm Nanowire Finite-T Barrier Sweep
# Realization: 39 | Strength: V0_1_2
# Temperatures: 30.0 mK and 40.0 mK (Unified Execution Engine)
# Asymmetry Grid: 43 points (a in [-60, -20..20, 60], delta_G_asym at index -1)
# ==============================================================================

echo "=================================================================="
echo " Job ID: $SLURM_JOB_ID"
echo " Node: $SLURM_NODELIST"
echo " Cores: $SLURM_CPUS_PER_TASK"
echo " Realization: 39 (V0_1_2)"
echo " Start Time: $(date)"
echo "=================================================================="

# Move to Harpers Ferry working directory
cd /scratch/nos00002/nonlocalcond || exit 1

# Activate Conda environment
source ~/.bashrc
conda activate kwant_fresh || conda activate base || source activate /home/nos00002/.conda/envs/kwant_fresh

# Enforce MKL/OpenMP thread control
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Execute Python Simulation
python main_parallel.py --config_path Inputs/Parameters/Disorder_Realizations/V0_1_2/disorder_realization_39.yaml

EXIT_CODE=$?
echo ""
echo "=================================================================="
echo " Python exited with code: $EXIT_CODE"
echo " End Time: $(date)"
echo "=================================================================="

exit $EXIT_CODE
