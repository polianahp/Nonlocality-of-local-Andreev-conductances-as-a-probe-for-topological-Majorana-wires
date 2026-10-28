#!/bin/bash
echo "Cancelling patch_hf_V0_0_872_* jobs for user $USER..."
squeue -u "$USER" -o "%i %j" | grep "patch_hf_V0_0_872" | awk '{print $1}' | xargs -r scancel
echo "Done."
