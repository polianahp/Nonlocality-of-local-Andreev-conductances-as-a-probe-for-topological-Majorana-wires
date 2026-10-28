#!/bin/bash
# ==============================================================================
# submit_all.sh: Stage 1 TGP Data Patch Dispatcher (V0_0_645)
#
# Usage:
#   ./submit_all.sh [START_INDEX] [NUM_JOBS] [OPTIONS]
#
# Examples:
#   ./submit_all.sh               # Default: start=50, count=50 (submits 50..99)
#   ./submit_all.sh 50 50         # Submits 50..99
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

if [ ${#POS_ARGS[@]} -ge 1 ]; then
    START_INDEX="${POS_ARGS[0]}"
fi
if [ ${#POS_ARGS[@]} -ge 2 ]; then
    NUM_JOBS="${POS_ARGS[1]}"
fi

END_INDEX=$((START_INDEX + NUM_JOBS - 1))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

V0_NAME="V0_0_645"

echo "=================================================================="
echo " Submitting Stage 1 Data Patch Jobs: $V0_NAME"
echo " Realization Range: $START_INDEX to $END_INDEX ($NUM_JOBS jobs requested)"
echo " Walltime: 01:00:00 | Cores: 256 per node"
if [ "$DRY_RUN" -eq 1 ]; then
    echo " MODE: DRY RUN (no jobs will be submitted)"
fi
echo "=================================================================="

SCRATCH_ROOT="${SCRATCH:-/scratch/nos00002}"
REPO_DIR="${SCRATCH_ROOT}/nonlocalcond"
if [ ! -d "$REPO_DIR" ]; then
    REPO_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi
DATA_DIR="${REPO_DIR}/Outputs/Data/Disorder_Realizations/${V0_NAME}"

ACTIVE_JOBS=""
if command -v squeue &>/dev/null; then
    ACTIVE_JOBS="$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)"
fi

COUNT=0
SKIPPED_EXISTING=0
SKIPPED_QUEUED=0

for i in $(seq "$START_INDEX" "$END_INDEX"); do
    slurm_file="disorder_realization_${i}.slurm"
    if [ ! -f "$slurm_file" ]; then
        echo "Warning: $slurm_file not found, skipping."
        continue
    fi

    tag=$(printf "%02d" "$i")
    job_name="patch_hf_${V0_NAME}_${tag}"

    # Safeguard 1: Check if already queued or running in Slurm
    if [ -n "$ACTIVE_JOBS" ] && echo "$ACTIVE_JOBS" | grep -qx "$job_name"; then
        echo "[SKIP] Realization $i ($job_name) is already queued/running in Slurm."
        SKIPPED_QUEUED=$((SKIPPED_QUEUED + 1))
        continue
    fi

    # Safeguard 2: Check if Stage 1 already completed on disk
    realization_out="${DATA_DIR}/disorder_realization_${i}"
    if [ "$FORCE" -eq 0 ] && [ -f "${realization_out}/tgp_stage1_dIdVl.npy" ]; then
        echo "[SKIP] Realization $i already has Stage 1 data (${realization_out}/tgp_stage1_dIdVl.npy). Use --force to override."
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
    echo "Successfully queued $COUNT Stage 1 patch jobs for $V0_NAME."
fi
if [ $SKIPPED_EXISTING -gt 0 ]; then
    echo "Skipped $SKIPPED_EXISTING runs that already had tgp_stage1_dIdVl.npy on disk."
fi
if [ $SKIPPED_QUEUED -gt 0 ]; then
    echo "Skipped $SKIPPED_QUEUED runs that were already active in Slurm queue."
fi
echo "=================================================================="
