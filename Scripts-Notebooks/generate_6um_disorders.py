#!/usr/bin/env python3
"""
generate_6um_disorders.py

Generates and standardizes disorder realizations for 6-micron (6000 nm) nanowires.
Geometry:
  L = 6000 nm, a0 = 10 nm -> Nx = 600 lattice sites (Ls = 600).
  Decay parameter lambda = (kappa * pi * Nx) / L = (100 * pi * 600) / 6000 = 10*pi ≈ 31.4159.
  Envelope: Q_dis(i) = tanh(i/2) * exp( - (i^2 * lambda^2) / (4 * Nx^2) ).
  Discrete Sine Transform (DST) basis: psiKx(n, i) = sqrt(2 / (Nx + 1)) * sin(i * n * pi / (Nx + 1)).

Outputs:
  1. Inputs/Run_Files/six_micron/Raw_Disorders/raw_disorder_6um_realization_{0,1,2}.npy
  2. Inputs/Run_Files/six_micron/New_Disorders/V0_{tag}_6um_realization_{r}.npz
  3. Inputs/Parameters/six_micron/V0_{tag}_realization_{r}.yaml
  4. Inputs/Slurms/six_micron/run_V0_{tag}_realization_{r}.slurm
  5. Outputs/Data/six_micron/V0_{tag}_realization_{r}/ (scaffolded)
"""

import sys
import os
from pathlib import Path
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Geometry specifications
L = 6000.0          # nm
a0_physics = 10.0   # nm (discretization length)
Nx = 600            # lattice sites
Ls = 600
kappa = 100.0
lambda_val = (kappa * np.pi * Nx) / L  # 10*pi ≈ 31.4159265

NUM_REALIZATIONS = 3
BENCHMARK_DISORDERS = [
    ("V0_0_1", 0.1, -2.0, 2.0),
    ("V0_0_378", 0.378, -2.0, 2.0),
    ("V0_0_645", 0.645, -1.0, 3.5),
    ("V0_0_872", 0.872, 0.0, 4.5),
    ("V0_1_2", 1.2, 0.0, 4.5),
]

# Destination paths
RUN_FILES_6UM = ROOT_DIR / "Inputs" / "Run_Files" / "six_micron"
RAW_DIR = RUN_FILES_6UM / "Raw_Disorders"
NEW_DIR = RUN_FILES_6UM / "New_Disorders"
PARAM_DIR_6UM = ROOT_DIR / "Inputs" / "Parameters" / "six_micron"
SLURM_DIR_6UM = ROOT_DIR / "Inputs" / "Slurms" / "six_micron"
DATA_DIR_6UM = ROOT_DIR / "Outputs" / "Data" / "six_micron"

def setup_directories():
    for d in [RAW_DIR, NEW_DIR, PARAM_DIR_6UM, SLURM_DIR_6UM, DATA_DIR_6UM]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Directory scaffolding complete under {ROOT_DIR}")

def construct_basis_and_envelope(Nx, lambda_val):
    ii = np.arange(1, Nx + 1)
    # Qdis envelope: Tanh[ii/2] * exp(-ii^2 * lambda^2 / (4 * Nx^2))
    Qdis = np.tanh(ii / 2.0) * np.exp(-(ii**2) * (lambda_val**2) / (4.0 * (Nx**2)))

    # DST basis psiKx: shape (Nx, Nx)
    nn_grid = np.arange(1, Nx + 1).reshape(-1, 1)
    ii_grid = np.arange(1, Nx + 1).reshape(1, -1)
    psiKx = np.sqrt(2.0 / (Nx + 1.0)) * np.sin(ii_grid * nn_grid * np.pi / (Nx + 1.0))

    return Qdis, psiKx

def generate_disorders():
    print(f"Generating 6-micron disorders: Nx={Nx}, lambda={lambda_val:.4f}...")
    Qdis, psiKx = construct_basis_and_envelope(Nx, lambda_val)

    # Use deterministic seeds for reproducibility of the 3 realizations
    np.random.seed(42)

    raw_realizations = []
    normalized_potentials = []

    for r in range(NUM_REALIZATIONS):
        # 1. Generate raw uniform random vector in [-1, 1]
        vdk = np.random.uniform(-1.0, 1.0, Nx)
        raw_realizations.append(vdk)

        # Save raw disorder vector
        raw_path = RAW_DIR / f"raw_disorder_6um_realization_{r}.npy"
        np.save(raw_path, vdk)
        print(f"Saved: {raw_path}")

        # 2. Filter frequencies with envelope and project via DST basis
        vdkq = vdk * Qdis
        vtp = psiKx @ vdkq

        # 3. Standardize: enforce zero mean (<V> = 0) and unit variance (<V^2> = 1)
        vtp_centered = vtp - np.mean(vtp)
        v2 = np.mean(vtp_centered**2)
        v_norm = vtp_centered / np.sqrt(v2)

        mean_val = np.mean(v_norm)
        std_val = np.std(v_norm)
        print(f"  Realization {r} standardized: mean={mean_val:.2e}, std={std_val:.4f}")
        normalized_potentials.append(v_norm)

    # 4. Save pre-scaled potentials in New_Disorders/
    for tag, v0_val, mu_min, mu_max in BENCHMARK_DISORDERS:
        for r in range(NUM_REALIZATIONS):
            npz_path = NEW_DIR / f"{tag}_6um_realization_{r}.npz"
            # In Kwant pipeline, Vdisx is stored normalized and scaled by V0 in worker
            np.savez(npz_path, Vdisx=normalized_potentials[r])
            print(f"Saved: {npz_path}")

    return raw_realizations, normalized_potentials

def generate_parameter_files():
    print("Generating YAML parameter files for 6-micron configurations...")
    count = 0
    for tag, v0_val, mu_min, mu_max in BENCHMARK_DISORDERS:
        for r in range(NUM_REALIZATIONS):
            yaml_path = PARAM_DIR_6UM / f"{tag}_realization_{r}.yaml"
            data_sub = f"six_micron/{tag}_realization_{r}"
            (DATA_DIR_6UM / f"{tag}_realization_{r}").mkdir(parents=True, exist_ok=True)

            cfg = {
                # Directory and File Management
                "dirname": data_sub,
                "fname": f"six_micron/New_Disorders/{tag}_6um_realization_{r}.npz",

                # Geometry & Physics Scales for 6-micron Wire
                "Ls": 600,
                "a0": 100.0,
                "barrier0": 2.0,
                "V0": float(v0_val),

                # Sweep Ranges (Tailored per disorder strength)
                "mu_max": float(mu_max),
                "mu_min": float(mu_min),
                "mu_dist": 0.02,

                "Vz_max": 1.4,
                "Vz_min": 0.0,
                "Vz_dist": 0.02,

                # Simulation Resolution
                "Upoints": 15,
                "num_engs": 51,
                "acceleration_type": "parallel",
                "weight_threshold": 0.8,
                "separation_threshold": 0.65,

                # Ground Truth Topological Invariant
                "calc_pfaffian": True,
            }

            with open(yaml_path, "w") as fp:
                yaml.dump(cfg, fp, default_flow_style=False, sort_keys=False)
            count += 1

    print(f"Generated {count} YAML parameter files in {PARAM_DIR_6UM}")

def generate_slurm_scripts():
    print("Generating SLURM submission scripts for 6-micron runs...")
    slurm_template = """#!/bin/bash 

#SBATCH -J 6um_{tag}_r{r}       # Job Name
#SBATCH -N 1                    # 1 node
#SBATCH -n 1                    # 1 master task
#SBATCH -c 256                  # 256 AMD EPYC cores
#SBATCH -t 15:00:00             # Time limit (15 hours for 6um wire)
#SBATCH -p day_3gb              # Partition
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nos00002@mix.wvu.edu

# Initialize Conda Environment on Harpers Ferry
if [ -f /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh ]; then
    source /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh
elif [ -f /shared/software/conda/conda_init.sh ]; then
    source /shared/software/conda/conda_init.sh
else
    source ~/.bashrc
fi

conda activate kwant_fresh
cd $SCRATCH/nonlocalcond

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting 6-micron simulation: {tag} realization {r}..."
python -u main_parallel.py --config_path Parameters/six_micron/{tag}_realization_{r}.yaml

echo "Simulation finished!"
conda deactivate
"""
    count = 0
    for tag, v0_val, mu_min, mu_max in BENCHMARK_DISORDERS:
        for r in range(NUM_REALIZATIONS):
            slurm_file = SLURM_DIR_6UM / f"run_{tag}_realization_{r}.slurm"
            with open(slurm_file, "w") as fp:
                fp.write(slurm_template.format(tag=tag, r=r))
            count += 1

    print(f"Generated {count} SLURM scripts in {SLURM_DIR_6UM}")

if __name__ == "__main__":
    setup_directories()
    generate_disorders()
    generate_parameter_files()
    generate_slurm_scripts()
    print("WP-4 6-micron generation & hierarchy initialization successfully completed!")
