#!/usr/bin/env python3
"""
setup_6um_finiteT_realizations.py

Generates the complete parameter and SLURM hierarchy for the 6 μm nanowire
FULL PIPELINE finite-temperature (30 mK and 40 mK) simulations across 50 disorder
realizations and 5 disorder strengths (V0 = 0.1, 0.378, 0.645, 0.872, 1.2 meV).

Architecture (Unified Multi-Temperature Engine):
- FULL CALCULATION ('barrier_sweep_only: false'):
    1. Closed-system BdG diagonalization (12 eigenvalues, topological gap, MZM profiles).
    2. TGP Stage 1 bias sweep (5 barriers x 97 energy grid points = 485 Kwant solves/pixel).
    3. Loop 2: Ground-truth Pfaffian topological invariant calculation (Nx = 600, delta_N = 0).
    4. Multi-Temperature Barrier Sweep: Loops over barrier_T_mK_list: [30.0, 40.0] within
       the SAME execution run, saving 102.5 node-hours across 50 realizations.
- 43-Point Asymmetry Ratio Grid:
    a in {-60.0} U linspace(-20, 20, 41) U {+60.0}
    Swept barrier: U_swept = a * barrier0 (barrier0 = 2.0 meV).
- Shared target folder per realization:
    Outputs/Data/six_micron/<V0_tag>/disorder_realization_{i}/
- Output arrays tagged with temperature suffix (_30mK.npy, _40mK.npy),
  delta_G_asym calculated from index -1 (a = 60.0).
- Organization:
    Inputs/Parameters/six_micron/<V0_tag>/disorder_realization_{i}.yaml
    Slurm_Scripts/Harpers_Ferry/six_micron/<V0_tag>/disorder_realization_{i}.slurm
"""

import os
import shutil
import stat
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Geometry & Physics
L = 6000.0          # nm (physical wire length)
Nx = 600            # lattice sites
Ls = 600
a0_angstrom = 100.0 # Angstroms
barrier0 = 2.0      # meV
lambda_dis = 31.4159265  # 10 * pi
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

PARAMS_BASE = REPO_ROOT / "Inputs" / "Parameters" / "six_micron"
SLURMS_BASE = REPO_ROOT / "Slurm_Scripts" / "Harpers_Ferry" / "six_micron"
DATA_BASE = REPO_ROOT / "Outputs" / "Data" / "six_micron"


def create_strength_dispatcher(v0_dir: Path, v0_tag: str):
    """Creates submit_all.sh inside a disorder strength folder (50 jobs)."""
    script_path = v0_dir / "submit_all.sh"
    content = f"""#!/bin/bash
# ==============================================================================
# submit_all.sh: 6 μm Nanowire Unified Finite-T Suite ({v0_tag})
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
echo " Submitting 6 μm Unified Jobs: {v0_tag} (Target: $MAX_JOBS jobs)"
echo " Temperatures: 30 mK & 40 mK | Walltime: 02:00:00 | Cores: 256"
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
    """Creates master submit_all.sh in six_micron/ root (submits all 250 jobs)."""
    script_path = master_dir / "submit_all.sh"
    content = """#!/bin/bash
# ==============================================================================
# submit_all.sh: Master Dispatcher for 6 μm Finite-T Full Pipeline Campaign
# Submits jobs across all 5 disorder strengths (250 jobs total).
# Each job computes BdG, TGP, Pfaffian once, and sweeps barriers at 30 mK & 40 mK.
#
# Usage:
#   ./submit_all.sh [JOBS_PER_BATCH]   # default: 50
# ==============================================================================

JOBS_PER_BATCH=${1:-50}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

DISORDER_FOLDERS=("V0_0_1" "V0_0_378" "V0_0_645" "V0_0_872" "V0_1_2")

echo "=================================================================="
echo " MASTER SUBMISSION: 6 μm Nanowire Full Finite-T Pipeline Campaign"
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
echo " All 6 μm jobs dispatched across 5 disorder strengths!"
echo " Monitor queue with: squeue -u $USER"
echo "=================================================================="
"""
    with open(script_path, "w") as f:
        f.write(content)
    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def generate_slurm_script(
    v0_tag: str,
    realization_idx: int,
    yaml_rel_path: str,
) -> str:
    """Generates the SLURM batch submission script for a 6 μm unified multi-T job."""
    job_name = f"6um_{v0_tag}_r{realization_idx}"
    log_name = f"6um_{v0_tag}_r{realization_idx}_%j"

    return f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -c 256
#SBATCH -t 02:00:00
#SBATCH -p compute
#SBATCH --job-name={job_name}
#SBATCH --output={log_name}.out
#SBATCH --error={log_name}.err

# ==============================================================================
# Harpers Ferry SLURM Job Script: 6 μm Nanowire Full Finite-T Pipeline
# Realization: {realization_idx} | Strength: {v0_tag}
# Temperatures: 30.0 mK and 40.0 mK (Unified Execution Engine)
# Asymmetry Grid: 43 points (a in [-60, -20..20, 60], delta_G_asym at index -1)
# ==============================================================================

echo "=================================================================="
echo " Job ID: $SLURM_JOB_ID"
echo " Node: $SLURM_NODELIST"
echo " Cores: $SLURM_CPUS_PER_TASK"
echo " Realization: {realization_idx} ({v0_tag})"
echo " Start Time: $(date)"
echo "=================================================================="

# Move to Harpers Ferry working directory
cd /scratch/nos00002/nonlocalcond || exit 1

# Activate Conda environment
source ~/.bashrc
conda activate kwant_fresh || conda activate base || source activate /home/nos00002/.conda/envs/kwant_fresh

# Enforce MKL/OpenMP thread control
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Execute Python Simulation
python main_parallel.py --config_path {yaml_rel_path}

EXIT_CODE=$?
echo ""
echo "=================================================================="
echo " Python exited with code: $EXIT_CODE"
echo " End Time: $(date)"
echo "=================================================================="

exit $EXIT_CODE
"""


def main():
    print("=" * 70)
    print(" Generating 6 μm Nanowire Unified Finite-T Configuration & SLURM Suite")
    print(f" Target Temperatures: {TARGET_TEMPS} mK (Unified Multi-T Pipeline)")
    print(" 43-Point Asymmetry Ratio Grid: [-60, -20..20, 60] (delta_G_asym enabled)")
    print(f" Total Realizations: {NUM_REALIZATIONS} per strength x 5 strengths = 250 jobs")
    print("=" * 70)

    # Clean up legacy subdirectories if present
    for v0_tag, _, _, _ in DISORDER_STRENGTHS:
        for legacy_temp in ["30mK", "40mK"]:
            p_leg = PARAMS_BASE / v0_tag / legacy_temp
            s_leg = SLURMS_BASE / v0_tag / legacy_temp
            if p_leg.exists():
                shutil.rmtree(p_leg)
            if s_leg.exists():
                shutil.rmtree(s_leg)

    total_yaml_count = 0
    total_slurm_count = 0

    for v0_tag, v0_val, mu_min, mu_max in DISORDER_STRENGTHS:
        print(f"\nProcessing Disorder Strength: {v0_tag} (V0 = {v0_val:.3f} meV)...")

        param_dir = PARAMS_BASE / v0_tag
        slurm_dir = SLURMS_BASE / v0_tag
        param_dir.mkdir(parents=True, exist_ok=True)
        slurm_dir.mkdir(parents=True, exist_ok=True)

        for i in range(NUM_REALIZATIONS):
            # Target output folder inside six_micron/
            target_dirname = f"six_micron/{v0_tag}/disorder_realization_{i}"

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
                "calc_pfaffian": True,
                "barrier_conductance_mode": "finite_temperature",
                "barrier_T_mK_list": TARGET_TEMPS,
                "barrier_quadrature_nodes": 8,
                "barrier_sweep_only": False,
            }

            yaml_file = param_dir / f"disorder_realization_{i}.yaml"
            with open(yaml_file, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            total_yaml_count += 1

            rel_yaml_path = f"Inputs/Parameters/six_micron/{v0_tag}/disorder_realization_{i}.yaml"
            slurm_content = generate_slurm_script(
                v0_tag=v0_tag,
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
    print(" Master dispatcher ready at: Slurm_Scripts/Harpers_Ferry/six_micron/submit_all.sh")
    print("=" * 70)


if __name__ == "__main__":
    main()
