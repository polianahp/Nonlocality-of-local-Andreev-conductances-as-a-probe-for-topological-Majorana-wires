#!/bin/bash

# Define the extension of your Slurm files (e.g., *.slurm, *.sh, *.job)
EXTENSION="*.slurm"

# Loop through all matching files in the current directory
for file in $EXTENSION; do
    # Check if it's a regular file (prevents errors if no files match the extension)
    if [ -f "$file" ]; then
        echo "Submitting job: $file"
        sbatch "$file"
        
        # Optional: Add a small delay (e.g., half a second) between submissions 
        # to avoid overwhelming the Slurm scheduler if you have thousands of files.
        sleep 2 
    else
        echo "No files matching '$EXTENSION' found in the current directory."
    fi
done

echo "Done!"