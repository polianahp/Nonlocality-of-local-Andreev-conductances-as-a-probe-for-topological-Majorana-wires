#!/bin/bash
#SBATCH --job-name=nlp_fpe_dpe_suite
#SBATCH --partition=day_3gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=06:00:00
#SBATCH --output=Logs/Slurm/nlp_fpe_dpe_suite_%j.out
#SBATCH --error=Logs/Slurm/nlp_fpe_dpe_suite_%j.err

# Strict thread pinning so each of the 256 worker processes uses 1 physical core
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p Logs/Slurm

echo "[$(date -u)] Starting Non-Local Protocol FPE/DPE 256-Core Suite on $(hostname)"

python3 run_protocol_suite.py \
    --workers "${SLURM_CPUS_PER_TASK:-256}" \
    --temperatures 30.0 40.0 \
    --keep-data auto

echo "[$(date -u)] Completed Non-Local Protocol FPE/DPE 256-Core Suite"
