#!/bin/sh 

#SBATCH -J  pl #Job Name
#SBATCH -N 2 #Number of nodes 
#SBATCH -c 40#Cores (40 for most queues,except gpu ones 24) [https://docs.hpc.wvu.edu/text/83.ThornyFlat.html]
#SBATCH -n 2 #Tasks [https://docs.hpc.wvu.edu/text/34.WorkloadManager.html#submitting-batch-jobs] (can be 1 per node unless you do some stuff)
#SBATCH -t 1-00:00:00 #Time [https://docs.hpc.wvu.edu/text/83.ThornyFlat.html]
#SBATCH --mail-type=END,FAIL,BEGIN #email you on these events
#SBATCH --mail-user=nos00002@mix.wvu.edu #change to your email 
#SBATCH -p comm_small_day  #queue comm_small_day, comm_small_week, comm_med_day, comm_med_week

#Stuff you would normally run on bash line

source /shared/software/conda/conda_init.sh

conda activate kwant_fresh
cd $SCRATCH/nonlocalcond
python -u main_parallel.py

conda deactivate
