#!/bin/bash
echo "Cancelling all patch2_hf_* jobs for user $USER..."
squeue -u "$USER" -o "%i %j" | grep "patch2_hf_" | awk '{print $1}' | xargs -r scancel
echo "All DataPatch2 jobs cancelled."
