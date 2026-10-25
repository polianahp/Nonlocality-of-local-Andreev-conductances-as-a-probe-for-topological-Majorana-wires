#!/bin/bash
# ==============================================================================
# cancel_all.sh: Cancel jobs for disorder strength V0_1_2
# Delegates to master cancel_all.sh with V0_1_2 filter
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../cancel_all.sh" "V0_1_2" "$@"
