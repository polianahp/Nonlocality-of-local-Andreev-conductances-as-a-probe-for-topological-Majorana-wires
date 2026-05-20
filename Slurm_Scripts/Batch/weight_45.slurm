#!/bin/sh 
#SBATCH -J  w45 
#SBATCH -N 2 
#SBATCH -c 40
#SBATCH -n 2 
#SBATCH -t 0-03:00:00 
#SBATCH --mail-type=END,FAIL,BEGIN 
#SBATCH --mail-user=nos00002@mix.wvu.edu 
#SBATCH -p comm_small_day  

source /shared/software/conda/conda_init.sh
conda activate kwant_fresh
cd $SCRATCH/nonlocalcond
python -u main_parallel.py --config_path Parameters/weight_45.yaml
conda deactivate
