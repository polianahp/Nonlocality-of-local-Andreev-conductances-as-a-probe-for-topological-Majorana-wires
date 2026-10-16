#!/usr/bin/env python3
"""
setup_6um_50_realizations.py

End-to-end setup script for the 6 μm nanowire 50-disorder realization study on Harpers Ferry.
1. Generates 50 canonical, normalized 600-site disorder potentials (L = 6 μm, Ls = 600, λ = 10π).
2. Scaffolds parameter YAML files for 5 disorder strengths (V0 = 0.1, 0.378, 0.645, 0.872, 1.2 meV),
   50 realizations each (250 YAMLs total).
3. Generates 250 SLURM scripts and 6 submit_all.sh dispatchers configured for 256-core AMD EPYC nodes.
4. Scaffolds all destination output directories.
5. Performs validation checks on generated arrays and configuration integrity.
"""

import os
import sys
import stat
from pathlib import Path
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ==============================================================================
# PHYSICS & GEOMETRY CONSTANTS
# ==============================================================================
L = 6000.0          # nm (physical wire length)
a0_physics = 10.0   # nm (discretization length)
Nx = 600            # lattice sites
Ls = 600
a0_angstrom = 100.0 # Angstroms
barrier0 = 2.0      # meV
kappa = 100.0
lambda_val = (kappa * np.pi * Nx) / L  # 10*pi ≈ 31.4159265

NUM_REALIZATIONS = 50

# 5 benchmark disorder strengths and chemical potential windows
DISORDER_STRENGTHS = [
    ("V0_0_1", 0.100, -2.0, 2.0),
    ("V0_0_378", 0.378, -2.0, 2.0),
    ("V0_0_645", 0.645, -1.0, 3.5),
    ("V0_0_872", 0.872, 0.0, 4.5),
    ("V0_1_2", 1.200, 0.0, 4.5),
]

# Destination base paths
RUN_FILES_BASE = REPO_ROOT / "Inputs" / "Run_Files" / "six_micron"
PARAMS_BASE = REPO_ROOT / "Inputs" / "Parameters" / "six_micron"
SLURMS_BASE = REPO_ROOT / "Slurm_Scripts" / "Harpers_Ferry" / "six_micron"
DATA_BASE = REPO_ROOT / "Outputs" / "Data" / "six_micron"


# ==============================================================================
# DISORDER GENERATION HELPERS
# ==============================================================================
def construct_basis_and_envelope(nx_sites: int, lam: float):
    """Constructs Discrete Sine Transform (DST) basis and Gaussian-like envelope."""
    ii = np.arange(1, nx_sites + 1)
    # Qdis envelope: Tanh[ii/2] * exp(-ii^2 * lambda^2 / (4 * Nx^2))
    q_dis = np.tanh(ii / 2.0) * np.exp(-(ii**2) * (lam**2) / (4.0 * (nx_sites**2)))

    # DST basis psiKx: shape (Nx, Nx)
    nn_grid = np.arange(1, nx_sites + 1).reshape(-1, 1)
    ii_grid = np.arange(1, nx_sites + 1).reshape(1, -1)
    psi_kx = np.sqrt(2.0 / (nx_sites + 1.0)) * np.sin(ii_grid * nn_grid * np.pi / (nx_sites + 1.0))

    return q_dis, psi_kx


def generate_disorders():
    print(f"\n[1/5] Generating 50 Canonical 6 μm Disorder Potentials (Nx={Nx}, λ={lambda_val:.4f})...")
    raw_dir = RUN_FILES_BASE / "Raw_Disorders"
    new_dir = RUN_FILES_BASE / "New_Disorders"
    raw_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)

    q_dis, psi_kx = construct_basis_and_envelope(Nx, lambda_val)

    for r in range(NUM_REALIZATIONS):
        # Deterministic random generator per realization for perfect reproducibility
        rng = np.random.RandomState(42 + r)
        vdk = rng.uniform(-1.0, 1.0, Nx)

        # 1. Save raw Fourier fluctuation vector
        raw_path = raw_dir / f"raw_disorder_{r}.npy"
        np.save(raw_path, vdk)

        # 2. Filter frequencies with envelope and project via DST basis
        vdkq = vdk * q_dis
        vtp = psi_kx @ vdkq

        # 3. Standardize: zero mean and unit variance
        vtp_centered = vtp - np.mean(vtp)
        v_norm = vtp_centered / np.sqrt(np.mean(vtp_centered**2))

        # Precision assertions
        assert abs(np.mean(v_norm)) < 1e-12, f"Realization {r} non-zero mean: {np.mean(v_norm)}"
        assert abs(np.std(v_norm) - 1.0) < 1e-6, f"Realization {r} non-unit std: {np.std(v_norm)}"

        # 4. Save normalized real-space potential
        npz_path = new_dir / f"disorder_realization_{r}.npz"
        np.savez(npz_path, Vdisx=v_norm)

    print(f"  ✓ Successfully saved 50 raw vectors in: {raw_dir}")
    print(f"  ✓ Successfully saved 50 normalized potentials in: {new_dir}")


# ==============================================================================
# PARAMETER YAML CONFIG GENERATOR
# ==============================================================================
def generate_parameter_yamls():
    print(f"\n[2/5] Generating 250 YAML Parameter Configs ({len(DISORDER_STRENGTHS)} strengths x {NUM_REALIZATIONS} realizations)...")
    count = 0
    for tag, v0_val, mu_min, mu_max in DISORDER_STRENGTHS:
        strength_dir = PARAMS_BASE / tag
        strength_dir.mkdir(parents=True, exist_ok=True)

        for r in range(NUM_REALIZATIONS):
            yaml_path = strength_dir / f"disorder_realization_{r}.yaml"

            cfg = {
                "dirname": f"six_micron/{tag}/disorder_realization_{r}",
                "fname": f"six_micron/New_Disorders/disorder_realization_{r}.npz",
                "is_real_space": True,
                "Ls": Ls,
                "a0": a0_angstrom,
                "barrier0": barrier0,
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
            }

            with open(yaml_path, "w") as fp:
                yaml.dump(cfg, fp, default_flow_style=False, sort_keys=False)
            count += 1

    print(f"  ✓ Generated {count} YAML files in {PARAMS_BASE}")


# ==============================================================================
# SLURM SCRIPT & DISPATCHER GENERATOR
# ==============================================================================
SLURM_TEMPLATE = """#!/bin/bash 

#SBATCH -J 6um_{tag}_r{r}       # Job Name
#SBATCH -N 1                    # 1 node
#SBATCH -n 1                    # 1 master task
#SBATCH -c 256                  # 256 AMD EPYC cores
#SBATCH -t 15:00:00             # Time limit (15 hours for 6um wire)
#SBATCH -p day_3gb              # Partition (day_3gb: 1 day limit, 3GB RAM/core)
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
python -u main_parallel.py --config_path Parameters/six_micron/{tag}/disorder_realization_{r}.yaml

echo "Simulation finished!"
conda deactivate
"""

SUBMIT_FOLDER_TEMPLATE = """#!/bin/bash
# ==============================================================================
# submit_all.sh
# 
# Submits disorder realization SLURM jobs for {tag} (6 μm) to Harpers Ferry queue.
# Default submissions: 50 jobs (disorder_realization_0 through 49).
#
# Usage:
#   ./submit_all.sh [NUM_JOBS]
# Examples:
#   ./submit_all.sh      # Submits all 50 jobs (default)
#   ./submit_all.sh 10   # Submits first 10 jobs
# ==============================================================================

MAX_JOBS=${{1:-50}}
COUNT=0

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

V0_NAME="$(basename "$SCRIPT_DIR")"

echo "=========================================================="
echo " Submitting Harpers Ferry 6 μm Jobs for $V0_NAME (Target: $MAX_JOBS jobs)"
echo "=========================================================="

for i in $(seq 0 $((MAX_JOBS - 1))); do
    slurm_file="disorder_realization_${{i}}.slurm"
    if [ ! -f "$slurm_file" ]; then
        echo "Warning: $slurm_file not found, skipping."
        continue
    fi
    echo "[$((COUNT + 1))/$MAX_JOBS] Submitting $slurm_file..."
    sbatch "$slurm_file"
    COUNT=$((COUNT + 1))
    sleep 0.2  # Throttle submissions slightly to protect scheduler socket
done

echo ""
echo "Successfully queued $COUNT jobs for $V0_NAME to Harpers Ferry scheduler."
echo "Monitor queue with: squeue -u \\$USER"
"""

SUBMIT_MASTER_TEMPLATE = """#!/bin/bash
# ==============================================================================
# submit_all.sh (Harpers Ferry Master Submission - 6 μm Suite)
#
# Submits batches across all 5 disorder strengths:
# V0_0_1, V0_0_378, V0_0_645, V0_0_872, V0_1_2
#
# Usage:
#   ./submit_all.sh [NUM_JOBS_PER_DISORDER]
# Examples:
#   ./submit_all.sh      # Submits 50 jobs per disorder strength (250 jobs total)
#   ./submit_all.sh 10   # Submits 10 jobs per disorder strength (50 jobs total)
# ==============================================================================

JOBS_PER_STRENGTH=${{1:-50}}

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

DISORDER_FOLDERS=("V0_0_1" "V0_0_378" "V0_0_645" "V0_0_872" "V0_1_2")

echo "=================================================================="
echo " Submitting Harpers Ferry 6 μm Master Batch: $JOBS_PER_STRENGTH realizations x 5 strengths"
echo " Total target: $((JOBS_PER_STRENGTH * ${{#DISORDER_FOLDERS[@]}})) jobs"
echo "=================================================================="

for folder in "${{DISORDER_FOLDERS[@]}}"; do
    if [ -d "$folder" ] && [ -f "$folder/submit_all.sh" ]; then
        echo ""
        echo ">>> Entering $folder..."
        (cd "$folder" && ./submit_all.sh "$JOBS_PER_STRENGTH")
    else
        echo "Warning: Subdirectory $folder or submit_all.sh missing."
    fi
done

echo ""
echo "=================================================================="
echo " Master 6 μm submission complete. Monitor queue with: squeue -u \\$USER"
echo "=================================================================="
"""


def generate_slurm_scripts():
    print(f"\n[3/5] Generating 250 Harpers Ferry SLURM Scripts & Dispatchers...")
    count = 0
    SLURMS_BASE.mkdir(parents=True, exist_ok=True)

    for tag, _, _, _ in DISORDER_STRENGTHS:
        strength_slurm_dir = SLURMS_BASE / tag
        strength_slurm_dir.mkdir(parents=True, exist_ok=True)

        for r in range(NUM_REALIZATIONS):
            slurm_path = strength_slurm_dir / f"disorder_realization_{r}.slurm"
            content = SLURM_TEMPLATE.format(tag=tag, r=r)
            with open(slurm_path, "w") as fp:
                fp.write(content)
            count += 1

        # Per-folder submit_all.sh
        folder_submit_path = strength_slurm_dir / "submit_all.sh"
        with open(folder_submit_path, "w") as fp:
            fp.write(SUBMIT_FOLDER_TEMPLATE.format(tag=tag))
        folder_submit_path.chmod(folder_submit_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Master submit_all.sh
    master_submit_path = SLURMS_BASE / "submit_all.sh"
    with open(master_submit_path, "w") as fp:
        fp.write(SUBMIT_MASTER_TEMPLATE)
    master_submit_path.chmod(master_submit_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"  ✓ Generated {count} SLURM scripts across {len(DISORDER_STRENGTHS)} directories.")
    print(f"  ✓ Created 5 folder dispatchers and 1 master submit_all.sh.")


# ==============================================================================
# OUTPUT DIRECTORY SCAFFOLDING
# ==============================================================================
def scaffold_output_directories():
    print(f"\n[4/5] Scaffolding Output Directories under {DATA_BASE}...")
    count = 0
    for tag, _, _, _ in DISORDER_STRENGTHS:
        for r in range(NUM_REALIZATIONS):
            out_dir = DATA_BASE / tag / f"disorder_realization_{r}"
            out_dir.mkdir(parents=True, exist_ok=True)
            count += 1
    print(f"  ✓ Scaffolded {count} output directories in {DATA_BASE}")


# ==============================================================================
# INTEGRITY VERIFICATION
# ==============================================================================
def verify_pipeline():
    print("\n[5/5] Running Pipeline Integrity Verification...")

    # 1. Verify disorder files
    new_dir = RUN_FILES_BASE / "New_Disorders"
    for r in range(NUM_REALIZATIONS):
        f = new_dir / f"disorder_realization_{r}.npz"
        assert f.exists(), f"Missing disorder file: {f}"
        data = np.load(f)["Vdisx"]
        assert data.shape == (600,), f"Wrong shape {data.shape} for {f}"
        assert abs(np.mean(data)) < 1e-12, f"Non-zero mean in {f}"
        assert abs(np.std(data) - 1.0) < 1e-6, f"Non-unit variance in {f}"

    # 2. Verify parameter YAMLs
    for tag, v0_val, _, _ in DISORDER_STRENGTHS:
        for r in range(NUM_REALIZATIONS):
            yp = PARAMS_BASE / tag / f"disorder_realization_{r}.yaml"
            assert yp.exists(), f"Missing YAML file: {yp}"
            with open(yp) as fp:
                cfg = yaml.safe_load(fp)
            assert cfg["Ls"] == 600
            assert cfg["is_real_space"] is True
            assert abs(cfg["V0"] - v0_val) < 1e-6

    # 3. Verify SLURM scripts
    for tag, _, _, _ in DISORDER_STRENGTHS:
        for r in range(NUM_REALIZATIONS):
            sp = SLURMS_BASE / tag / f"disorder_realization_{r}.slurm"
            assert sp.exists(), f"Missing SLURM file: {sp}"
            with open(sp) as fp:
                txt = fp.read()
            assert f"Parameters/six_micron/{tag}/disorder_realization_{r}.yaml" in txt

    print("  ✓ All 50 disorder files verified (shape 600, mean 0, std 1).")
    print("  ✓ All 250 parameter YAMLs verified.")
    print("  ✓ All 250 SLURM scripts & dispatchers verified.")
    print("\n=== PIPELINE SETUP SUCCESSFULLY COMPLETED ===")


def main():
    print("==================================================================")
    print(" INITIALIZING 6 μm NANOWIRE 50-DISORDER SUITE GENERATOR")
    print("==================================================================")
    generate_disorders()
    generate_parameter_yamls()
    generate_slurm_scripts()
    scaffold_output_directories()
    verify_pipeline()


if __name__ == "__main__":
    main()
