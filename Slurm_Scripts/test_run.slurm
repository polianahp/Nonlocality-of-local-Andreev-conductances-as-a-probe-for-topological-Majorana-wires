#!/bin/sh 

#SBATCH -J test_run #Job Name
#SBATCH -N 1 #Number of nodes 
#SBATCH -c 40 #Cores
#SBATCH -n 1 #Tasks 
#SBATCH -t 00:10:00 #Time (10 minutes max for safety, though it runs much faster)
#SBATCH -p comm_small_day  #queue

source /shared/software/conda/conda_init.sh

conda activate kwant_fresh
cd $SCRATCH/nonlocalcond

echo "Starting fast test run..."

# Running fast test using the newly created configuration
python -u main_parallel.py --config_path Parameters/test_run.yaml

echo "Test run completed!"
conda deactivate
