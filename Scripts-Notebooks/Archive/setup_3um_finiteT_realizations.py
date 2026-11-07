#!/usr/bin/env python3
"""
setup_3um_finiteT_realizations.py

Generates the complete parameter and SLURM hierarchy for the 3 μm nanowire
finite-temperature (30 mK and 40 mK) barrier conductance sweeps across 50 disorder
realizations and 5 disorder strengths (V0 = 0.1, 0.378, 0.645, 0.872, 1.2 meV).

Architecture (Unified Multi-Temperature Engine):
- Lightweight 'barrier_sweep_only: true' mode (bypasses BdG, TGP bias sweeps, PDI).
- Evaluates both 30 mK and 40 mK within the SAME execution pass via barrier_T_mK_list: [30.0, 40.0].
- 43-Point Asymmetry Ratio Grid:
    a in {-60.0} U linspace(-20, 20, 41) U {+60.0}
    Swept barrier: U_swept = a * barrier0 (barrier0 = 2.0 meV).
- Outputs written DIRECTLY into existing shared realization folders on Harpers Ferry:
    /scratch/nos00002/nonlocalcond/Outputs/Data/Disorder_Realizations/<V0_tag>/disorder_realization_{i}/
- Output arrays tagged with temperature suffix (_30mK.npy, _40mK.npy),
  preventing overwriting of any existing zero-temperature data.
- Hierarchical parameter and SLURM organization matching the Harpers Ferry namespace:
    Inputs/Parameters/Disorder_Realizations/<V0_tag>/disorder_realization_{i}.yaml
    Slurm_Scripts/Harpers_Ferry/Disorder_Realizations/<V0_tag>/disorder_realization_{i}.slurm
"""

import os
import shutil
import stat
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Geometry & Physics
Ls = 300
a0_angstrom = 100.0
barrier0 = 2.0
lambda_dis = 4.5
num_engs = 51

NUM_REALIZATIONS = 50

# 5 benchmark disorder strengths and chemical potential windows
DISORDER_STRENGTHS = [
    ("V0_0_1", 0.100, -2.0, 2.0),
    ("V0_0_378", 0.378, -2.0, 2.0),
    ("V0_0_645", 0.645, -1.0, 3.5),
    ("V0_0_872", 0.872, 0.0, 4.5),
    ("V0_1_2", 1.200, 0.0, 4.5),
]

TARGET_TEMPS = [30.0, 40.0]

PARAMS_BASE = REPO_ROOT / "Inputs" / "Parameters" / "Disorder_Realizations"
SLURMS_BASE = REPO_ROOT / "Slurm_Scripts" / "Harpers_Ferry" / "Disorder_Realizations"
DATA_BASE = REPO_ROOT / "Outputs" / "Data" / "Disorder_Realizations"


def create_strength_dispatcher(v0_dir: Path, v0_tag: str):
    """Creates submit_all.sh inside a disorder strength folder (50 jobs)."""
    script_path = v0_dir / "submit_all.sh"
    content = f"""#!/bin/bash
# ==============================================================================
# submit_all.sh: 3 μm Nanowire Unified Finite-T Barrier Sweep ({v0_tag})
# Submits 50 jobs (each runs both 30 mK and 40 mK in a single execution).
#
# Usage:
#   ./submit_all.sh [MAX_JOBS]   # default: 50
# ==============================================================================

MAX_JOBS=${{1:-50}}
COUNT=0

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "=================================================================="
echo " Submitting 3 μm Unified Jobs: {v0_tag} (Target: $MAX_JOBS jobs)"
echo " Temperatures: 30 mK & 40 mK | Walltime: 03:00:00 | Cores: 256"
echo "=================================================================="

for i in $(seq 0 $((MAX_JOBS - 1))); do
    slurm_file="disorder_realization_${{i}}.slurm"
    if [ ! -f "$slurm_file" ]; then
        echo "Warning: $slurm_file not found, skipping."
        continue
    fi
    echo "[$((COUNT + 1))/$MAX_JOBS] Submitting $slurm_file..."
    sbatch "$slurm_file"
    COUNT=$((COUNT + 1))
    sleep 0.2
done

echo ""
echo "Successfully queued $COUNT unified jobs for {v0_tag}."
echo "Monitor queue with: squeue -u \\$USER"
"""
    with open(script_path, "w") as f:
        f.write(content)
    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def create_master_dispatcher(master_dir: Path):
    """Creates master submit_all.sh in Disorder_Realizations/ root (submits all 250 jobs)."""
    script_path = master_dir / "submit_all.sh"
    content = """#!/bin/bash
# ==============================================================================
# submit_all.sh: Master Dispatcher for 3 μm Finite-T Barrier Sweep Campaign
# Submits jobs across all 5 disorder strengths (250 jobs total).
# Each job sweeps barriers at 30 mK & 40 mK with 43-point asymmetry ratio grid.
#
# Usage:
#   ./submit_all.sh [JOBS_PER_BATCH]   # default: 50
# ==============================================================================

JOBS_PER_BATCH=${1:-50}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

DISORDER_FOLDERS=("V0_0_1" "V0_0_378" "V0_0_645" "V0_0_872" "V0_1_2")

echo "=================================================================="
echo " MASTER SUBMISSION: 3 μm Nanowire Finite-T Barrier Sweep Campaign"
echo " 5 Disorder Strengths x $JOBS_PER_BATCH Realizations (250 jobs total)"
echo " Temperatures: [30.0 mK, 40.0 mK] (Unified Engine)"
echo " Total Target: $((JOBS_PER_BATCH * ${#DISORDER_FOLDERS[@]})) jobs"
echo "=================================================================="

for folder in "${DISORDER_FOLDERS[@]}"; do
    if [ -d "$folder" ] && [ -f "$folder/submit_all.sh" ]; then
        echo ""
        echo "=============================================================="
        echo ">>> Entering Strength Directory: $folder"
        echo "=============================================================="
        (cd "$folder" && ./submit_all.sh "$JOBS_PER_BATCH")
    fi
done

echo ""
echo "=================================================================="
echo " All 3 μm jobs dispatched across 5 disorder strengths!"
echo " Monitor queue with: squeue -u $USER"
echo "=================================================================="
"""
    with open(script_path, "w") as f:
        f.write(content)
    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def generate_slurm_script(
    v0_tag: str,
    v0_val: float,
    realization_idx: int,
    yaml_rel_path: str,
) -> str:
    """Generates the SLURM batch submission script for a 3 μm unified barrier sweep job."""
    return f"""#!/bin/bash

#SBATCH -J dis_hf_{v0_tag}_{realization_idx:02d}  # Job Name
#SBATCH -N 1                            # Number of nodes (single node parallel pool)
#SBATCH -n 1                            # Number of tasks
#SBATCH -c 256                          # Cores per task (256 cores on Harpers Ferry AMD EPYC node)
#SBATCH -t 03:00:00                     # Time limit (3 hours)
#SBATCH -p day_3gb                      # Partition (day_3gb: 1 day limit, 3GB RAM/core)
#SBATCH --mail-type=END,FAIL            # Notifications
#SBATCH --mail-user=nos00002@mix.wvu.edu

# Initialize Conda environment on Harpers Ferry
if [ -f /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh ]; then
    source /shared/software/miniforge/26.3.2-3/etc/profile.d/conda.sh
elif [ -f /shared/software/conda/conda_init.sh ]; then
    source /shared/software/conda/conda_init.sh
else
    source ~/.bashrc
fi

conda activate kwant_fresh
cd $SCRATCH/nonlocalcond

# Thread Pinning: 1 thread per worker to prevent CPU oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting simulation for disorder realization {realization_idx} (V0 = {v0_val:g} meV) on Harpers Ferry..."
python -u main_parallel.py --config_path {yaml_rel_path}

echo "Realization {realization_idx} finished!"
conda deactivate
"""



def main():
    print("=" * 70)
    print(" Generating 3 μm Nanowire Unified Finite-T Configuration & SLURM Suite")
    print(f" Target Temperatures: {TARGET_TEMPS} mK (Unified Multi-T Pipeline)")
    print(" 43-Point Asymmetry Ratio Grid: [-60, -20..20, 60] (delta_G_asym enabled)")
    print(f" Total Realizations: {NUM_REALIZATIONS} per strength x 5 strengths = 250 jobs")
    print(" Target Output: Outputs/Data/Disorder_Realizations/<V0_tag>/disorder_realization_{i}/")
    print("=" * 70)

    # Clean up erroneously created three_micron directories if present
    erroneous_param_dir = REPO_ROOT / "Inputs" / "Parameters" / "three_micron"
    erroneous_slurm_dir = REPO_ROOT / "Slurm_Scripts" / "Harpers_Ferry" / "three_micron"
    if erroneous_param_dir.exists():
        shutil.rmtree(erroneous_param_dir)
        print(f"Cleaned up erroneous directory: {erroneous_param_dir}")
    if erroneous_slurm_dir.exists():
        shutil.rmtree(erroneous_slurm_dir)
        print(f"Cleaned up erroneous directory: {erroneous_slurm_dir}")

    total_yaml_count = 0
    total_slurm_count = 0

    for v0_tag, v0_val, mu_min, mu_max in DISORDER_STRENGTHS:
        print(f"\nProcessing Disorder Strength: {v0_tag} (V0 = {v0_val:.3f} meV)...")

        param_dir = PARAMS_BASE / v0_tag
        slurm_dir = SLURMS_BASE / v0_tag
        param_dir.mkdir(parents=True, exist_ok=True)
        slurm_dir.mkdir(parents=True, exist_ok=True)

        for i in range(NUM_REALIZATIONS):
            # Target output folder inside Disorder_Realizations/
            target_dirname = f"Disorder_Realizations/{v0_tag}/disorder_realization_{i}"

            config_dict = {
                "dirname": target_dirname,
                "realization_index": i,
                "lambda_dis": float(lambda_dis),
                "V0": float(v0_val),
                "Ls": int(Ls),
                "a0": float(a0_angstrom),
                "barrier0": float(barrier0),
                "mu_max": float(mu_max),
                "mu_min": float(mu_min),
                "mu_dist": 0.02,
                "Vz_max": 1.4,
                "Vz_min": 0.0,
                "Vz_dist": 0.02,
                "num_engs": int(num_engs),
                "acceleration_type": "parallel",
                "weight_threshold": 0.8,
                "separation_threshold": 0.65,
                "calc_pfaffian": False,
                "barrier_conductance_mode": "finite_temperature",
                "barrier_T_mK_list": TARGET_TEMPS,
                "barrier_quadrature_nodes": 8,
                "barrier_sweep_only": True,
            }

            yaml_file = param_dir / f"disorder_realization_{i}.yaml"
            with open(yaml_file, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            total_yaml_count += 1

            rel_yaml_path = f"Parameters/Disorder_Realizations/{v0_tag}/disorder_realization_{i}.yaml"
            slurm_content = generate_slurm_script(
                v0_tag=v0_tag,
                v0_val=v0_val,
                realization_idx=i,
                yaml_rel_path=rel_yaml_path,
            )
            slurm_file = slurm_dir / f"disorder_realization_{i}.slurm"
            with open(slurm_file, "w") as f:
                f.write(slurm_content)
            os.chmod(slurm_file, os.stat(slurm_file).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            total_slurm_count += 1

        create_strength_dispatcher(slurm_dir, v0_tag)

    create_master_dispatcher(SLURMS_BASE)

    print("\n" + "=" * 70)
    print(f" Successfully generated {total_yaml_count} Parameter YAMLs in {PARAMS_BASE}")
    print(f" Successfully generated {total_slurm_count} SLURM Scripts in {SLURMS_BASE}")
    print(" Master dispatcher ready at: Slurm_Scripts/Harpers_Ferry/Disorder_Realizations/submit_all.sh")
    print("=" * 70)


if __name__ == "__main__":
    main()
