#!/bin/bash
# ==============================================================================
# cancel_all.sh: Master Harpers Ferry Job Cancellation Dispatcher
#
# Delegates to Disorder_Realizations/cancel_all.sh by default.
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/Disorder_Realizations/cancel_all.sh" "$@"
