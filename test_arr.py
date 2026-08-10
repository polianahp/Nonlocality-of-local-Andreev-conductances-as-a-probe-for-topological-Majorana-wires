import numpy as np
from pathlib import Path

dir_path = Path("Data/disorder_realization_5_results")

# List files matching barrier*
print("Files starting with 'barrier':")
for p in dir_path.glob("barrier*.npy"):
    print(" -", p.name)

# Check one array
arr_path = dir_path / "barrier_left_conductance_right_arr.npy"
if arr_path.exists():
    arr = np.load(arr_path)
    print("\nArray 'barrier_left_conductance_right_arr.npy':")
    print("Shape:", arr.shape)
    print("Dtype:", arr.dtype)
    print("Max:", np.max(arr))
    print("Min:", np.min(arr))
    print("Mean:", np.mean(arr))
    print("Non-zero count:", np.count_nonzero(arr))
    print("Is all zero?", np.all(arr == 0))
else:
    print(f"\n{arr_path.name} not found!")
    
arr_path2 = dir_path / "barrier_right_conductance_left_arr.npy"
if arr_path2.exists():
    arr2 = np.load(arr_path2)
    print("\nArray 'barrier_right_conductance_left_arr.npy':")
    print("Shape:", arr2.shape)
    print("Non-zero count:", np.count_nonzero(arr2))
    print("Is all zero?", np.all(arr2 == 0))
else:
    print(f"\n{arr_path2.name} not found!")
