import os
import stat
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
dp2_root = repo_root / "Slurm_Scripts" / "Harpers_Ferry" / "DataPatch2"
dp2_root.mkdir(parents=True, exist_ok=True)

disorders = ["V0_0_378", "V0_0_645", "V0_0_872", "V0_1_2"]

def make_exec(path: Path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

# Master cancel_all.sh
master_cancel = dp2_root / "cancel_all.sh"
master_cancel.write_text("""#!/bin/bash
echo "Cancelling all patch2_hf_* jobs for user $USER..."
squeue -u "$USER" -o "%i %j" | grep "patch2_hf_" | awk '{print $1}' | xargs -r scancel
echo "All DataPatch2 jobs cancelled."
""", encoding="utf-8")
make_exec(master_cancel)

# Master submit_all.sh
master_submit = dp2_root / "submit_all.sh"
master_submit.write_text("""#!/bin/bash
# ==============================================================================
# Master Dispatcher for Harpers Ferry Spectra & Localization DataPatch2 Campaign
# ==============================================================================

START_INDEX=50
NUM_JOBS=50
PASSTHROUGH_ARGS=()

POS_COUNT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start|-s)
            START_INDEX="$2"
            shift 2
            ;;
        --count|-n)
            NUM_JOBS="$2"
            shift 2
            ;;
        --dry-run|-d|--force|-f)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
        --help|-h)
            echo "Usage: ./submit_all.sh [START_INDEX] [NUM_JOBS] [--dry-run] [--force]"
            exit 0
            ;;
        *)
            if [ $POS_COUNT -eq 0 ]; then
                START_INDEX="$1"
                POS_COUNT=1
            elif [ $POS_COUNT -eq 1 ]; then
                NUM_JOBS="$1"
                POS_COUNT=2
            else
                PASSTHROUGH_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

DISORDER_FOLDERS=("V0_0_378" "V0_0_645" "V0_0_872" "V0_1_2")
END_INDEX=$((START_INDEX + NUM_JOBS - 1))

echo "=================================================================="
echo " MASTER SUBMISSION: Harpers Ferry Spectra & Localization (DataPatch2)"
echo " 4 Disorder Strengths: ${DISORDER_FOLDERS[*]}"
echo " Realization Range:    $START_INDEX to $END_INDEX ($NUM_JOBS jobs per strength)"
echo " Total Target:         $((NUM_JOBS * ${#DISORDER_FOLDERS[@]})) jobs"
echo " Cores per Job:        256 | Walltime: 00:30:00"
echo " Safeguards:           Skips runs that already have spectrum_arr.npy on disk"
echo "=================================================================="

for folder in "${DISORDER_FOLDERS[@]}"; do
    if [ -d "$folder" ] && [ -f "$folder/submit_all.sh" ]; then
        echo ""
        echo "=============================================================="
        echo ">>> Entering Strength Directory: $folder"
        echo "=============================================================="
        (cd "$folder" && ./submit_all.sh "$START_INDEX" "$NUM_JOBS" "${PASSTHROUGH_ARGS[@]}")
    fi
done

echo ""
echo "=================================================================="
echo " Master patch dispatch completed across 4 disorder strengths!"
echo " Monitor queue with: squeue -u $USER"
echo "=================================================================="
""", encoding="utf-8")
make_exec(master_submit)

# Per-disorder scripts and slurm files
for v0 in disorders:
    folder = dp2_root / v0
    folder.mkdir(parents=True, exist_ok=True)

    # cancel_all.sh
    c_sh = folder / "cancel_all.sh"
    c_sh.write_text(f"""#!/bin/bash
echo "Cancelling patch2_hf_{v0}_* jobs for user $USER..."
squeue -u "$USER" -o "%i %j" | grep "patch2_hf_{v0}" | awk '{{print $1}}' | xargs -r scancel
echo "Done."
""", encoding="utf-8")
    make_exec(c_sh)

    # submit_all.sh
    s_sh = folder / "submit_all.sh"
    s_sh_content = f"""#!/bin/bash
# ==============================================================================
# submit_all.sh: Spectra & Localization DataPatch2 Dispatcher ({v0})
#
# Usage:
#   ./submit_all.sh [START_INDEX] [NUM_JOBS] [OPTIONS]
#
# Examples:
#   ./submit_all.sh               # Default: start=50, count=50 (submits 50..99)
#   ./submit_all.sh 50 50         # Submits 50..99
#   ./submit_all.sh 0 100         # Submits all 0..99
#   ./submit_all.sh 50 50 --dry-run
#   ./submit_all.sh 50 50 --force
# ==============================================================================

START_INDEX=50
NUM_JOBS=50
FORCE=0
DRY_RUN=0

POS_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start|-s)
            START_INDEX="$2"
            shift 2
            ;;
        --count|-n)
            NUM_JOBS="$2"
            shift 2
            ;;
        --force|-f)
            FORCE=1
            shift
            ;;
        --dry-run|-d)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            echo "Usage: ./submit_all.sh [START_INDEX] [NUM_JOBS] [--dry-run] [--force]"
            exit 0
            ;;
        *)
            POS_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ ${{#POS_ARGS[@]}} -ge 1 ]; then
    START_INDEX="${{POS_ARGS[0]}}"
fi
if [ ${{#POS_ARGS[@]}} -ge 2 ]; then
    NUM_JOBS="${{POS_ARGS[1]}}"
fi

END_INDEX=$((START_INDEX + NUM_JOBS - 1))

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

V0_NAME="{v0}"

echo "=================================================================="
echo " Submitting DataPatch2 Jobs (Spectra & Localization): $V0_NAME"
echo " Realization Range: $START_INDEX to $END_INDEX ($NUM_JOBS jobs requested)"
echo " Walltime: 00:30:00 | Cores: 256 per node"
if [ "$DRY_RUN" -eq 1 ]; then
    echo " MODE: DRY RUN (no jobs will be submitted)"
fi
echo "=================================================================="

SCRATCH_ROOT="${{SCRATCH:-/scratch/nos00002}}"
REPO_DIR="${{SCRATCH_ROOT}}/nonlocalcond"
if [ ! -d "$REPO_DIR" ]; then
    REPO_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi

if [ ! -d "$REPO_DIR" ]; then
    echo "ERROR: Repository directory does not exist: $REPO_DIR" >&2
    exit 1
fi

DATA_DIR="${{REPO_DIR}}/Outputs/Data/Disorder_Realizations/${{V0_NAME}}"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR" >&2
    exit 1
fi

ACTIVE_JOBS=""
if command -v squeue &>/dev/null; then
    ACTIVE_JOBS="$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)"
fi

COUNT=0
SKIPPED_EXISTING=0
SKIPPED_QUEUED=0

for i in $(seq "$START_INDEX" "$END_INDEX"); do
    slurm_file="disorder_realization_${{i}}.slurm"
    if [ ! -f "$slurm_file" ]; then
        echo "Warning: $slurm_file not found, skipping."
        continue
    fi

    job_name="patch2_hf_${{V0_NAME}}_${{i}}"

    # Safeguard 1: Check if already queued or running in Slurm
    if [ -n "$ACTIVE_JOBS" ] && echo "$ACTIVE_JOBS" | grep -qx "$job_name"; then
        echo "[SKIP] Realization $i ($job_name) is already queued/running in Slurm."
        SKIPPED_QUEUED=$((SKIPPED_QUEUED + 1))
        continue
    fi

    # Safeguard 2: Check if spectrum_arr and localization already completed on disk
    realization_out="${{DATA_DIR}}/disorder_realization_${{i}}"
    if [ "$FORCE" -eq 0 ] && [ -f "${{realization_out}}/spectrum_arr.npy" ] && [ -f "${{realization_out}}/topological_gap.npy" ] && [ -f "${{realization_out}}/weight_localization_arr.npy" ]; then
        echo "[SKIP] Realization $i already has spectra & localization (${{realization_out}}/spectrum_arr.npy). Use --force to override."
        SKIPPED_EXISTING=$((SKIPPED_EXISTING + 1))
        continue
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY-RUN] Would submit $slurm_file ($job_name)"
        COUNT=$((COUNT + 1))
    else
        echo "[$((COUNT + 1))/$NUM_JOBS] Submitting $slurm_file..."
        sbatch "$slurm_file"
        COUNT=$((COUNT + 1))
        sleep 0.1
    fi
done

echo ""
echo "=================================================================="
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run complete for $V0_NAME: $COUNT jobs ready to submit."
else
    echo "Successfully queued $COUNT DataPatch2 jobs for $V0_NAME."
fi
if [ $SKIPPED_EXISTING -gt 0 ]; then
    echo "Skipped $SKIPPED_EXISTING runs that already had spectra & localization on disk."
fi
if [ $SKIPPED_QUEUED -gt 0 ]; then
    echo "Skipped $SKIPPED_QUEUED runs that were already active in Slurm queue."
fi
echo "=================================================================="
"""
    s_sh.write_text(s_sh_content, encoding="utf-8")
    make_exec(s_sh)

    # 100 slurm files per disorder strength
    for r_idx in range(100):
        slurm_f = folder / f"disorder_realization_{r_idx}.slurm"
        slurm_content = f"""#!/bin/bash

#SBATCH -J patch2_hf_{v0}_{r_idx}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 256
#SBATCH -t 00:30:00
#SBATCH -p day_3gb
#SBATCH --mail-type=END,FAIL
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

echo "Patching missing spectra & localization for realization {r_idx} ({v0}) on Harpers Ferry..."
python -u patch_missing_spectra_and_localization.py --disorder_folder {v0} --realization_index {r_idx} --cores 256

echo "Realization {r_idx} spectra & localization patching finished!"
conda deactivate
"""
        slurm_f.write_text(slurm_content, encoding="utf-8")

print(f"Generated complete DataPatch2 structure with {4 * 100} SLURM scripts and dispatchers.")
