#!/bin/sh 

#SBATCH -J  Tdis_sim #Job Name
#SBATCH -N 2 #Number of nodes 
#SBATCH -c 40#Cores (40 for most queues,except gpu ones 24) [https://docs.hpc.wvu.edu/text/83.ThornyFlat.html]
#SBATCH -n 2 #Tasks [https://docs.hpc.wvu.edu/text/34.WorkloadManager.html#submitting-batch-jobs] (can be 1 per node unless you do some stuff)
#SBATCH -t 0-23:00:00 #Time [https://docs.hpc.wvu.edu/text/83.ThornyFlat.html]
#SBATCH --mail-type=END,FAIL,BEGIN #email you on these events
#SBATCH --mail-user=nos00002@mix.wvu.edu #change to your email 

source /shared/software/conda/conda_init.sh

conda activate kwant_fresh
cd $SCRATCH/nonlocalcond

# Running with disorder realization 130
python -u main_parallel.py --config_path Parameters/Disorder_Realizations/disorder_realization_130.yaml

conda deactivate
