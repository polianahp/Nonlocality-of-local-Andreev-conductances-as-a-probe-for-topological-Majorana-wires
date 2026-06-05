import os
from pathlib import Path
import numpy as np
import hyperopt as hyp
import helpers as hp

def load_all_realizations(input_dir):
    """
    Loads all disorder realizations from the specified input data directory.
    
    Parameters:
        input_dir (str or Path): Path to the directory containing disorder realization subdirectories.
        
    Returns:
        dict: A large nested dictionary with the structure:
              {
                  "disorder_realization_name": {
                      "parameters": param_data,  # loaded from params_list.npy
                      "data": {
                          "pdi_data": pdi_data_arr,
                          "peaks_left": peaks_left_arr,
                          ...
                      }
                  }
              }
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")
        
    # Discover all subdirectories starting with "disorder_realization_"
    subdirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("disorder_realization_")]
    
    # Sort subdirectories numerically
    def get_realization_idx(path):
        name = path.name
        parts = name.split('_')
        for p in parts:
            if p.isdigit():
                return int(p)
        return 999999
    subdirs = sorted(subdirs, key=get_realization_idx)
    
    realizations_data = {}
    
    print(f"Found {len(subdirs)} realization subdirectories in {input_path}")
    
    for subdir in subdirs:
        realization_name = subdir.name
        param_file = subdir / "params_list.npy"
        
        if not param_file.exists():
            print(f"Warning: params_list.npy not found in {subdir.name}, skipping.")
            continue
            
        param_data = np.load(param_file)
        
        # Load other .npy data files
        data_dict = {}
        for filepath in subdir.glob("*.npy"):
            if filepath.name == "params_list.npy":
                continue
            # Store with key as file stem (filename without .npy)
            data_dict[filepath.stem] = np.load(filepath)
            
        realizations_data[realization_name] = {
            "parameters": param_data,
            "data": data_dict
        }
        
    return realizations_data

if __name__ == "__main__":
    # Example usage with Protocol_V2
    example_dir = "/home/pseudonym/code/Nonlocal_Conductance/Nonlocality-of-local-Andreev-conductances-as-a-probe-for-topological-Majorana-wires/Data/Protocol_V2"
    
    print(f"Loading realizations from {example_dir}...")
    try:
        data = load_all_realizations(example_dir)
        print(f"Successfully loaded {len(data)} realizations.")
        # Print info about one realization if available
        if data:
            first_key = list(data.keys())[0]
            print(f"\nExample realization: {first_key}")
            print(f"Parameters shape: {data[first_key]['parameters'].shape}")
            print("Data arrays loaded:")
            for k, v in data[first_key]['data'].items():
                print(f"  - {k}: shape {v.shape}")
    except Exception as e:
        print(f"Error: {e}")