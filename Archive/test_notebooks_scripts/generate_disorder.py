import argparse
import numpy as np
from pathlib import Path
from src.config import PathConfigs

def main():
    parser = argparse.ArgumentParser(description="Generate raw Fourier-space random vectors for disorder realizations.")
    parser.add_argument("--num_realizations", type=int, default=3, help="Number of disorder realizations to generate.")
    parser.add_argument("--Ls", type=int, default=300, help="Length of the spatial domain (Ls).")
    args = parser.parse_args()

    # Determine the save path inside Run_Files/Fourier_Component_Disorders
    output_dir = PathConfigs.RUN_FILES / "Fourier_Component_Disorders"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the maximum existing index to ensure unique naming
    existing_files = list(output_dir.glob("fourier_disorder_*.npy"))
    if existing_files:
        indices = []
        for f in existing_files:
            try:
                # Expected format: fourier_disorder_{index}.npy
                idx = int(f.stem.split('_')[-1])
                indices.append(idx)
            except ValueError:
                pass
        start_index = max(indices) + 1 if indices else 0
    else:
        start_index = 0

    print(f"Generating {args.num_realizations} Fourier component realizations of length {args.Ls}...")
    
    for i in range(args.num_realizations):
        # Generate raw random uniform vectors in [-1, 1]
        fourier_disorder = np.random.uniform(-1, 1, args.Ls)
        
        file_index = start_index + i
        output_path = output_dir / f"fourier_disorder_{file_index}.npy"
        
        # Save as individual .npy file
        np.save(output_path, fourier_disorder)
        
    print(f"Successfully generated {args.num_realizations} files.")
    print(f"Saved to directory: {output_dir}")
    print(f"Indices generated: {start_index} to {start_index + args.num_realizations - 1}")

if __name__ == "__main__":
    main()
