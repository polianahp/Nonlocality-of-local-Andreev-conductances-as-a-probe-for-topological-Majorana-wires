#!/bin/bash
# ==============================================================================
# cancel_all.sh: Slurm Job Cancellation Utility for Disorder Realizations
#
# Cancels running and/or pending disorder realization jobs on Harpers Ferry.
#
# Usage:
#   ./cancel_all.sh [OPTIONS] [V0_FILTER]
#
# Arguments:
#   V0_FILTER         Optional disorder strength filter (e.g. V0_0_1, V0_0_378,
#                     V0_0_645, V0_0_872, V0_1_2). Defaults to all 5 strengths.
#
# Options:
#   -a, --all         Cancel ALL jobs for current user ($USER) across the cluster
#   -p, --pending     Cancel only PENDING (queued) jobs; leave RUNNING jobs active
#   -r, --running     Cancel only RUNNING jobs; leave PENDING jobs in queue
#   -s, --start N     Cancel only realizations with index >= N
#   -e, --end N       Cancel only realizations with index <= N
#   -d, --dry-run     Preview matching jobs without actually canceling them
#   -y, --yes, -f     Skip interactive confirmation prompt
#   -h, --help        Show this help message
#
# Examples:
#   ./cancel_all.sh                       # Cancel all disorder realization jobs (with confirmation)
#   ./cancel_all.sh -y                    # Cancel all disorder realization jobs immediately
#   ./cancel_all.sh V0_0_1                # Cancel only V0_0_1 jobs
#   ./cancel_all.sh --pending             # Cancel only queued/pending jobs
#   ./cancel_all.sh --start 50 --end 99   # Cancel realizations 50 to 99
#   ./cancel_all.sh --dry-run             # Preview what would be canceled
#   ./cancel_all.sh --all                 # Cancel ALL jobs for $USER across the cluster
# ==============================================================================

set -euo pipefail

CANCEL_ALL_USER=0
STATE_FILTER=""
START_INDEX=""
END_INDEX=""
V0_FILTER=""
DRY_RUN=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--all)
            CANCEL_ALL_USER=1
            shift
            ;;
        -p|--pending)
            STATE_FILTER="PENDING"
            shift
            ;;
        -r|--running)
            STATE_FILTER="RUNNING"
            shift
            ;;
        -s|--start)
            START_INDEX="$2"
            shift 2
            ;;
        -e|--end)
            END_INDEX="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=1
            shift
            ;;
        -y|--yes|-f|--force)
            FORCE=1
            shift
            ;;
        -h|--help)
            sed -n '2,32p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        V0_*|v0_*)
            V0_FILTER="$1"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Run './cancel_all.sh --help' for usage."
            exit 1
            ;;
    esac
done

# Check for Slurm tools
if ! command -v squeue &>/dev/null || ! command -v scancel &>/dev/null; then
    echo "Error: 'squeue' and 'scancel' commands must be available. Are you running on a Slurm cluster?" >&2
    exit 1
fi

TARGET_USER="${USER:-$(whoami)}"

echo "=================================================================="
echo " Slurm Job Cancellation Utility (Harpers Ferry)"
echo " User: $TARGET_USER"
if [ "$CANCEL_ALL_USER" -eq 1 ]; then
    echo " Target: ALL jobs for user $TARGET_USER"
else
    echo " Target: Disorder realization jobs (prefix: dis_hf_)"
    [ -n "$V0_FILTER" ]    && echo " Disorder Filter: $V0_FILTER"
    [ -n "$STATE_FILTER" ]  && echo " State Filter:    $STATE_FILTER"
    [ -n "$START_INDEX" ]   && echo " Start Realization: $START_INDEX"
    [ -n "$END_INDEX" ]     && echo " End Realization:   $END_INDEX"
fi
if [ "$DRY_RUN" -eq 1 ]; then
    echo " Mode: DRY RUN (no jobs will be cancelled)"
fi
echo "=================================================================="

# Query active jobs from Slurm
RAW_QUEUE=$(squeue -u "$TARGET_USER" -h -o "%i %T %j" 2>/dev/null || true)

if [ -z "$RAW_QUEUE" ]; then
    echo "No active or queued jobs found for user $TARGET_USER."
    exit 0
fi

MATCHING_JOB_IDS=()
MATCHING_DETAILS=()
COUNT_RUNNING=0
COUNT_PENDING=0
COUNT_OTHER=0

while read -r job_id job_state job_name; do
    [ -z "$job_id" ] && continue

    # If --all is not set, only target disorder realization jobs
    if [ "$CANCEL_ALL_USER" -eq 0 ]; then
        # Must start with dis_hf_
        if [[ ! "$job_name" =~ ^dis_hf_ ]]; then
            continue
        fi

        # Filter by V0 if requested
        if [ -n "$V0_FILTER" ] && [[ ! "$job_name" =~ $V0_FILTER ]]; then
            continue
        fi

        # Filter by realization index range if requested
        # Expected format: dis_hf_<V0>_<idx>
        if [ -n "$START_INDEX" ] || [ -n "$END_INDEX" ]; then
            idx=$(echo "$job_name" | awk -F'_' '{print $NF}')
            if [[ "$idx" =~ ^[0-9]+$ ]]; then
                idx_num=$((10#$idx))
                if [ -n "$START_INDEX" ] && [ "$idx_num" -lt "$START_INDEX" ]; then
                    continue
                fi
                if [ -n "$END_INDEX" ] && [ "$idx_num" -gt "$END_INDEX" ]; then
                    continue
                fi
            fi
        fi
    fi

    # Filter by state if requested
    if [ -n "$STATE_FILTER" ] && [ "$job_state" != "$STATE_FILTER" ]; then
        continue
    fi

    MATCHING_JOB_IDS+=("$job_id")
    MATCHING_DETAILS+=("$(printf "%-10s %-12s %s" "$job_id" "$job_state" "$job_name")")

    case "$job_state" in
        RUNNING) COUNT_RUNNING=$((COUNT_RUNNING + 1)) ;;
        PENDING) COUNT_PENDING=$((COUNT_PENDING + 1)) ;;
        *)       COUNT_OTHER=$((COUNT_OTHER + 1)) ;;
    esac
done <<< "$RAW_QUEUE"

TOTAL_MATCHING=${#MATCHING_JOB_IDS[@]}

if [ "$TOTAL_MATCHING" -eq 0 ]; then
    echo "No jobs matched the specified cancellation criteria."
    exit 0
fi

echo ""
echo "Found $TOTAL_MATCHING matching job(s):"
echo "  - RUNNING: $COUNT_RUNNING"
echo "  - PENDING: $COUNT_PENDING"
if [ "$COUNT_OTHER" -gt 0 ]; then
    echo "  - OTHER:   $COUNT_OTHER"
fi
echo ""
printf "%-10s %-12s %s\n" "JOB_ID" "STATE" "JOB_NAME"
echo "----------------------------------------------------"
if [ "$TOTAL_MATCHING" -le 30 ]; then
    for detail in "${MATCHING_DETAILS[@]}"; do
        echo "$detail"
    done
else
    for i in $(seq 0 9); do
        echo "${MATCHING_DETAILS[$i]}"
    done
    echo "... [$(($TOTAL_MATCHING - 20)) additional jobs hidden] ..."
    for i in $(seq $(($TOTAL_MATCHING - 10)) $(($TOTAL_MATCHING - 1))); do
        echo "${MATCHING_DETAILS[$i]}"
    done
fi
echo "----------------------------------------------------"
echo ""

if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN COMPLETE: $TOTAL_MATCHING job(s) identified for cancellation. None were cancelled."
    exit 0
fi

# Confirmation prompt if interactive and not forced
if [ "$FORCE" -eq 0 ] && [ -t 0 ]; then
    read -r -p "Are you sure you want to cancel these $TOTAL_MATCHING job(s)? [y/N]: " CONFIRM
    case "$CONFIRM" in
        [yY]|[yY][eE][sS])
            ;;
        *)
            echo "Cancellation aborted by user."
            exit 0
            ;;
    esac
fi

echo "Cancelling $TOTAL_MATCHING job(s)..."
printf "%s\n" "${MATCHING_JOB_IDS[@]}" | xargs -n 50 scancel

echo ""
echo "=================================================================="
echo " Successfully cancelled $TOTAL_MATCHING job(s)."
echo " Current queue status for $TARGET_USER:"
squeue -u "$TARGET_USER" || true
echo "=================================================================="
