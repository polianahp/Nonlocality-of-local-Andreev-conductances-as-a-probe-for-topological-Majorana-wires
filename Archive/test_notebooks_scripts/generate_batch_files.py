import os
from pathlib import Path

def main():
    repo_root = Path(__file__).parent.resolve()
    params_dir = repo_root / "Parameters" / "Disorder_Realizations"
    slurm_dir = repo_root / "Slurm_Scripts" / "Disorder_Realizations"

    # Create directories if they don't exist
    params_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating 150 YAMLs in {params_dir}")
    print(f"Generating 150 SLURMs in {slurm_dir}")

    for i in range(150):
        # 1. Write YAML file
        yaml_content = f"""# disorder_realization_{i}.yaml
# Configuration for system with dynamically generated disorder realization {i}.

# Disorder Management
dirname: "disorder_realization_{i}_results"
realization_index: {i}
lambda_dis: 4.5   # Envelope decay parameter

# Sweep Ranges
mu_max: 4.5
mu_min: 0.0
mu_dist: 0.02

Vz_max: 1.4
Vz_min: 0.0
Vz_dist: 0.02

# Simulation Resolution
Upoints: 10
num_engs: 51
acceleration_type: "parallel"
weight_threshold: 0.8
separation_threshold: 0.65
"""
        yaml_path = params_dir / f"disorder_realization_{i}.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        # 2. Write SLURM script
        slurm_content = f"""#!/bin/sh 

#SBATCH -J dis_{i} #Job Name
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

# Running with disorder realization {i}
python -u main_parallel.py --config_path Parameters/Disorder_Realizations/disorder_realization_{i}.yaml

conda deactivate
"""
        slurm_path = slurm_dir / f"disorder_realization_{i}.slurm"
        with open(slurm_path, "w") as f:
            f.write(slurm_content)

    print("Successfully generated all 150 YAML configuration files and SLURM scripts.")

if __name__ == "__main__":
    main()
