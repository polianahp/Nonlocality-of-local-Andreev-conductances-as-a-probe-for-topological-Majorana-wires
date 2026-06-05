#!/bin/sh 

#SBATCH -J dis_1_3-5 #Job Name
#SBATCH -N 2 #Number of nodes 
#SBATCH -c 40 #Cores
#SBATCH -n 2 #Tasks
#SBATCH -t 0-03:00:00 #Time
#SBATCH --mail-type=END,FAIL,BEGIN #email you on these events
#SBATCH --mail-user=nos00002@mix.wvu.edu #change to your email 
#SBATCH -p comm_small_day  #queue

source /shared/software/conda/conda_init.sh

conda activate kwant_fresh
cd $SCRATCH/nonlocalcond

# Running with disorder realization 1 and length scale 3.5
python -u main_parallel.py --config_path Parameters/vary_length_scale/disorder_realization_1_lambda_3-5.yaml

conda deactivate
