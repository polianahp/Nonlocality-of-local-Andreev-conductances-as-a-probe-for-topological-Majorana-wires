#!/bin/bash
echo "Cancelling patch2_hf_V0_1_2_* jobs for user $USER..."
squeue -u "$USER" -o "%i %j" | grep "patch2_hf_V0_1_2" | awk '{print $1}' | xargs -r scancel
echo "Done."
