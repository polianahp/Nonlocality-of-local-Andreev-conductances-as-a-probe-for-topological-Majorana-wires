#!/usr/bin/env python3
"""
setup_six_micron_analysis.py

Sets up the six_micron_analysis study suite:
1. Reorganizes six_micron -> six_micron_analysis/ with 6micron/ and 3micron/ branches.
2. Truncates 600-site 6um real-space potentials to 300-site 3um potentials (first 300 sites).
3. Generates YAML parameter configs for both 6micron (Ls=600) and 3micron (Ls=300).
4. Generates SLURM submission scripts and submit_all.sh for both wire lengths on Harpers Ferry.
"""

import sys
import os
import shutil
from pathlib import Path
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Specifications for the 7 benchmark disorders and 3 realizations
BENCHMARK_DISORDERS = [
    ("V0_0_0", 0.0, -2.0, 2.0),
    ("V0_0_1", 0.1, -2.0, 2.0),
    ("V0_0_91", 0.91, 0.0, 4.5),
    ("V0_0_378", 0.378, -2.0, 2.0),
    ("V0_0_645", 0.645, -1.0, 3.5),
    ("V0_0_872", 0.872, 0.0, 4.5),
    ("V0_1_2", 1.2, 0.0, 4.5),
]
NUM_REALIZATIONS = 3

# Directory targets
RUN_FILES_BASE = REPO_ROOT / "Inputs" / "Run_Files" / "six_micron_analysis"
PARAMS_BASE = REPO_ROOT / "Inputs" / "Parameters" / "six_micron_analysis"
SLURMS_BASE = REPO_ROOT / "Slurm_Scripts" / "Harpers_Ferry" / "six_micron_analysis"
OUTPUTS_BASE = REPO_ROOT / "Outputs" / "Data" / "six_micron_analysis"

OLD_RUN_FILES_6UM = REPO_ROOT / "Inputs" / "Run_Files" / "six_micron"
OLD_PARAMS_6UM = REPO_ROOT / "Inputs" / "Parameters" / "six_micron"
OLD_SLURMS_6UM = REPO_ROOT / "Slurm_Scripts" / "Harpers_Ferry" / "six_micron"
OLD_OUTPUTS_6UM = REPO_ROOT / "Outputs" / "Data" / "six_micron"


def move_to_trash(path: Path):
    """Safely moves a path to .trash/ with collision avoidance."""
    if not path.exists():
        return
    trash_dir = REPO_ROOT / ".trash"
    trash_dir.mkdir(parents=True, exist_ok=True)
    dest = trash_dir / path.name
    count = 1
    while dest.exists():
        dest = trash_dir / f"{path.name}_{count}"
        count += 1
    print(f"Moving legacy folder to trash: {path} -> {dest}")
    shutil.move(str(path), str(dest))


def setup_disorder_files():
    print("Setting up disorder files under six_micron_analysis...")
    
    # Target directories
    dir_6um_new = RUN_FILES_BASE / "6micron" / "New_Disorders"
    dir_6um_raw = RUN_FILES_BASE / "6micron" / "Raw_Disorders"
    dir_3um_new = RUN_FILES_BASE / "3micron" / "New_Disorders"
    
    dir_6um_new.mkdir(parents=True, exist_ok=True)
    dir_6um_raw.mkdir(parents=True, exist_ok=True)
    dir_3um_new.mkdir(parents=True, exist_ok=True)

    # 1. Copy / Move existing 6um disorders into 6micron/
    src_new = OLD_RUN_FILES_6UM / "New_Disorders"
    src_raw = OLD_RUN_FILES_6UM / "Raw_Disorders"
    
    if src_raw.exists():
        for f in src_raw.glob("*.npy"):
            shutil.copy2(f, dir_6um_raw / f.name)
            
    if src_new.exists():
        for f in src_new.glob("*.npz"):
            shutil.copy2(f, dir_6um_new / f.name)

    # 2. Generate truncated 3um disorders from 6um (first 300 lattice sites)
    print("Generating truncated 3um disorders (first 300 lattice sites)...")
    count_3um = 0
    for tag, v0_val, _, _ in BENCHMARK_DISORDERS:
        for r in range(NUM_REALIZATIONS):
            src_6um_file = dir_6um_new / f"{tag}_6um_realization_{r}.npz"
            if not src_6um_file.exists():
                raise FileNotFoundError(f"Missing 6um source file: {src_6um_file}")
            
            data_6um = np.load(src_6um_file)["Vdisx"]
            if len(data_6um) != 600:
                raise ValueError(f"Expected 6um potential length 600, got {len(data_6um)} for {src_6um_file}")
            
            # Truncate directly in real space
            vdis_3um = data_6um[:300]
            dest_3um_file = dir_3um_new / f"{tag}_3um_realization_{r}.npz"
            np.savez(dest_3um_file, Vdisx=vdis_3um)
            count_3um += 1

    print(f"Generated {count_3um} truncated 3um real-space disorder files in {dir_3um_new}")


def generate_parameter_yamls():
    print("Generating YAML parameter files for 6micron and 3micron suites...")
    
    params_6um_dir = PARAMS_BASE / "6micron"
    params_3um_dir = PARAMS_BASE / "3micron"
    params_6um_dir.mkdir(parents=True, exist_ok=True)
    params_3um_dir.mkdir(parents=True, exist_ok=True)
    
    # Also scaffold output directories
    (OUTPUTS_BASE / "6micron").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_BASE / "3micron").mkdir(parents=True, exist_ok=True)
    
    count = 0
    for tag, v0_val, mu_min, mu_max in BENCHMARK_DISORDERS:
        for r in range(NUM_REALIZATIONS):
            # 6-micron YAML
            cfg_6um = {
                "dirname": f"six_micron_analysis/6micron/{tag}_realization_{r}",
                "fname": f"six_micron_analysis/6micron/New_Disorders/{tag}_6um_realization_{r}.npz",
                "Ls": 600,
                "a0": 100.0,
                "barrier0": 2.0,
                "V0": float(v0_val),
                "mu_max": float(mu_max),
                "mu_min": float(mu_min),
                "mu_dist": 0.02,
                "Vz_max": 1.4,
                "Vz_min": 0.0,
                "Vz_dist": 0.02,
                "Upoints": 15,
                "num_engs": 51,
                "acceleration_type": "parallel",
                "weight_threshold": 0.8,
                "separation_threshold": 0.65,
                "calc_pfaffian": True,
                "is_real_space": True,
            }
            with open(params_6um_dir / f"{tag}_realization_{r}.yaml", "w") as fp:
                yaml.dump(cfg_6um, fp, default_flow_style=False, sort_keys=False)

            # 3-micron YAML (Truncated disorder counterpart)
            cfg_3um = {
                "dirname": f"six_micron_analysis/3micron/{tag}_realization_{r}",
                "fname": f"six_micron_analysis/3micron/New_Disorders/{tag}_3um_realization_{r}.npz",
                "Ls": 300,
                "a0": 100.0,
                "barrier0": 2.0,
                "V0": float(v0_val),
                "mu_max": float(mu_max),
                "mu_min": float(mu_min),
                "mu_dist": 0.02,
                "Vz_max": 1.4,
                "Vz_min": 0.0,
                "Vz_dist": 0.02,
                "Upoints": 15,
                "num_engs": 51,
                "acceleration_type": "parallel",
                "weight_threshold": 0.8,
                "separation_threshold": 0.65,
                "calc_pfaffian": True,
                "is_real_space": True,
            }
            with open(params_3um_dir / f"{tag}_realization_{r}.yaml", "w") as fp:
                yaml.dump(cfg_3um, fp, default_flow_style=False, sort_keys=False)

            count += 1

    print(f"Generated {count} 6-micron YAMLs in {params_6um_dir}")
    print(f"Generated {count} 3-micron YAMLs in {params_3um_dir}")


def generate_slurm_scripts():
    print("Generating SLURM scripts for Harpers Ferry cluster execution...")
    
    slurms_6um_dir = SLURMS_BASE / "6micron"
    slurms_3um_dir = SLURMS_BASE / "3micron"
    slurms_6um_dir.mkdir(parents=True, exist_ok=True)
    slurms_3um_dir.mkdir(parents=True, exist_ok=True)

    slurm_template_6um = """#!/bin/bash 

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
python -u main_parallel.py --config_path Parameters/six_micron_analysis/6micron/{tag}_realization_{r}.yaml

echo "Simulation finished!"
conda deactivate
"""

    slurm_template_3um = """#!/bin/bash 

#SBATCH -J 3um_{tag}_r{r}       # Job Name
#SBATCH -N 1                    # 1 node
#SBATCH -n 1                    # 1 master task
#SBATCH -c 256                  # 256 AMD EPYC cores
#SBATCH -t 02:00:00             # Time limit (2 hours for 3um wire)
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

echo "Starting 3-micron simulation: {tag} realization {r}..."
python -u main_parallel.py --config_path Parameters/six_micron_analysis/3micron/{tag}_realization_{r}.yaml

echo "Simulation finished!"
conda deactivate
"""

    submit_template = """#!/bin/bash
# ==============================================================================
# submit_all.sh
# Submits simulation SLURM jobs to Harpers Ferry queue.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

ARG="$1"
COUNT=0
MAX_JOBS=999
FILTER=""

if [[ "$ARG" =~ ^[0-9]+$ ]]; then
    MAX_JOBS="$ARG"
elif [ -n "$ARG" ]; then
    FILTER="$ARG"
fi

echo "=========================================================="
echo " Submitting Harpers Ferry Simulation Jobs"
if [ -n "$FILTER" ]; then
    echo " Filter: *$FILTER*"
fi
echo "=========================================================="

shopt -s nullglob
if [ -n "$FILTER" ]; then
    FILES=(run_*${FILTER}*.slurm)
else
    FILES=(run_*.slurm)
fi

TOTAL_FOUND=${#FILES[@]}

if [ "$TOTAL_FOUND" -eq 0 ]; then
    echo "No matching SLURM files found in $SCRIPT_DIR for pattern: ${FILTER:-run_*.slurm}"
    exit 1
fi

for slurm_file in "${FILES[@]}"; do
    if [ "$COUNT" -ge "$MAX_JOBS" ]; then
        break
    fi
    echo "[$((COUNT + 1))/$TOTAL_FOUND] Submitting $slurm_file..."
    sbatch "$slurm_file"
    COUNT=$((COUNT + 1))
    sleep 0.2
done

echo "Successfully submitted $COUNT / $TOTAL_FOUND jobs."
"""

    for tag, _, _, _ in BENCHMARK_DISORDERS:
        for r in range(NUM_REALIZATIONS):
            # 6um slurm
            s6 = slurm_template_6um.format(tag=tag, r=r)
            p6 = slurms_6um_dir / f"run_{tag}_realization_{r}.slurm"
            with open(p6, "w") as fp:
                fp.write(s6)
                
            # 3um slurm
            s3 = slurm_template_3um.format(tag=tag, r=r)
            p3 = slurms_3um_dir / f"run_{tag}_realization_{r}.slurm"
            with open(p3, "w") as fp:
                fp.write(s3)

    # Write submit_all.sh
    sub6 = slurms_6um_dir / "submit_all.sh"
    with open(sub6, "w") as fp:
        fp.write(submit_template)
    sub6.chmod(0o755)

    sub3 = slurms_3um_dir / "submit_all.sh"
    with open(sub3, "w") as fp:
        fp.write(submit_template)
    sub3.chmod(0o755)

    print(f"Generated 21 SLURM scripts + submit_all.sh in {slurms_6um_dir}")
    print(f"Generated 21 SLURM scripts + submit_all.sh in {slurms_3um_dir}")


def main():
    print("=== INITIALIZING SIX_MICRON_ANALYSIS PIPELINE SETUP ===")
    setup_disorder_files()
    generate_parameter_yamls()
    generate_slurm_scripts()
    
    # Safely archive old directories to .trash/
    print("\nArchiving legacy six_micron folders to .trash/...")
    move_to_trash(OLD_RUN_FILES_6UM)
    move_to_trash(OLD_PARAMS_6UM)
    move_to_trash(OLD_SLURMS_6UM)
    move_to_trash(OLD_OUTPUTS_6UM)
    
    print("\n=== SETUP COMPLETE ===")


if __name__ == "__main__":
    main()
