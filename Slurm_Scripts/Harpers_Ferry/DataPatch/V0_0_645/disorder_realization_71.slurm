#!/bin/bash

#SBATCH -J patch_hf_V0_0_645_71
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 256
#SBATCH -t 01:00:00
#SBATCH -p day_3gb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nos00002@mix.wvu.edu

# Initialize Conda environment on Harpers Ferry
if [ -f /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh ]; then
    source /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh
elif [ -f /shared/software/conda/conda_init.sh ]; then
    source /shared/software/conda/conda_init.sh
else
    source ~/.bashrc
fi

conda activate kwant_fresh
cd $SCRATCH/nonlocalcond

# Thread Pinning: 1 thread per worker to prevent CPU oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Patching missing Stage 1 TGP data for realization 71 (V0_0_645) on Harpers Ferry..."
python -u patch_missing_tgp_stage1.py --disorder_folder V0_0_645 --realization_index 71 --cores 256

echo "Realization 71 Stage 1 patching finished!"
conda deactivate
