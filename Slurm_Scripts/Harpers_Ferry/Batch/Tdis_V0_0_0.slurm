#!/bin/bash 

#SBATCH -J Tdis_V0_0_0        # Job Name
#SBATCH -N 1                  # Number of nodes (multiprocessing pool runs on a single node)
#SBATCH -n 1                  # Number of tasks
#SBATCH -c 256                # Cores per task (Full 256 cores on Harpers Ferry node)
#SBATCH -t 10:00:00           # Time limit (10 hours)
#SBATCH -p day_3gb            # Partition (day_3gb: 1 day max limit, 3GB/core)
#SBATCH --mail-type=END,FAIL,BEGIN # Email notifications
#SBATCH --mail-user=nos00002@mix.wvu.edu

# Initialize Conda environment on Harpers Ferry
#test comment
if [ -f /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh ]; then
    source /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh
elif [ -f /shared/software/conda/conda_init.sh ]; then
    source /shared/software/conda/conda_init.sh
else
    source ~/.bashrc
fi

conda activate kwant_fresh
cd $SCRATCH/nonlocalcond

# Running with the Tdis configuration
python -u main_parallel.py --config_path Parameters/Tdis_V0_0_0.yaml 

conda deactivate
