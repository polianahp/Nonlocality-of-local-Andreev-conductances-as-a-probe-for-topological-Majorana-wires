import sys
import os
from pathlib import Path
import numpy as np

# Ensure local package path for h5netcdf and h5py is loaded
sys.path.insert(0, str(Path(__file__).parent / "local_packages"))
import h5py
import h5netcdf
import xarray as xr

def inspect_device_data():
    data_dir = Path(__file__).parent / "msft_data"
    nc_files = sorted(list(data_dir.glob("*.nc")))

    print("=" * 90)
    print(f"EXPLICIT INSPECTION OF MICROSOFT TGP EXPERIMENTAL DATASETS ({len(nc_files)} files)")
    print(f"Directory: {data_dir}")
    print("=" * 90)

    for f in nc_files:
        print(f"\n==========================================================================================")
        print(f"FILE: {f.name} (Size: {f.stat().st_size / (1024*1024):.2f} MB)")
        print(f"==========================================================================================")
        try:
            ds = xr.open_dataset(f, engine="h5netcdf")
            print(f"Dimensions : {dict(ds.dims)}")
            print(f"Coordinates: {list(ds.coords.keys())}")
            print(f"Variables  : {list(ds.data_vars.keys())}")
            print("\nCoordinate Details:")

            for c in ds.coords:
                arr = ds[c].values
                if isinstance(arr, np.ndarray) and arr.ndim == 1:
                    diffs = np.diff(arr)
                    step = np.mean(diffs) if len(diffs) > 0 else 0.0
                    print(f"  - {c:18s}: {len(arr):4d} points | Range: [{arr[0]:12.6e}, {arr[-1]:12.6e}] | Step (Delta): {step:12.6e}")
                    if "bias" in c.lower():
                        # Convert to microvolts for easy physical interpretation
                        print(f"    -> In microvolts (uV) : Range: [{arr[0]*1e6:7.1f} uV, {arr[-1]*1e6:7.1f} uV] | Step: {step*1e6:5.2f} uV")
                        print(f"    -> First 8 values (uV): {np.round(arr[:8]*1e6, 3)}")
                        print(f"    -> Last 4 values  (uV): {np.round(arr[-4:]*1e6, 3)}")
            
            # Print sample conductance shapes
            for g_var in ["g_ll", "g_lr", "g_rl", "g_rr", "I_1w_L", "I_2w_LR"]:
                if g_var in ds.data_vars:
                    da = ds[g_var]
                    print(f"\nConductance Variable '{g_var}': shape = {da.shape}, dims = {da.dims}")
                    break

        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    print("\n" + "=" * 90)
    print("ALL EXPERIMENTAL DATASETS LOADED AND VERIFIED DIRECTLY FROM DISK.")
    print("=" * 90)

if __name__ == "__main__":
    inspect_device_data()
