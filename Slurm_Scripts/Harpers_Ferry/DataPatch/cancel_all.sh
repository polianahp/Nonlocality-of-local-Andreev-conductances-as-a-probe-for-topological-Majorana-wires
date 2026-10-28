#!/bin/bash
echo "Cancelling all patch_hf_* jobs for user $USER..."
squeue -u "$USER" -o "%i %j" | grep "patch_hf_" | awk '{print $1}' | xargs -r scancel
echo "All patch jobs cancelled."
