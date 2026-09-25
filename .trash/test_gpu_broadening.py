import numpy as np
import cupy as cp
import xarray as xr
from src.gpu_broadening import broaden_with_temperature_gpu, _temp_kernel
import tgp.prepare

# Create dummy data
N_B, N_V, N_bias = 2, 3, 101
bias = np.linspace(-1, 1, N_bias)
data = np.random.rand(1, N_B, N_V, N_bias)

ds = xr.Dataset(
    data_vars={
        "g_ll": (["cutter_pair_index", "B", "V", "bias"], data.copy()),
        "g_lr": (["cutter_pair_index", "B", "V", "bias"], data.copy()),
        "g_rl": (["cutter_pair_index", "B", "V", "bias"], data.copy()),
        "g_rr": (["cutter_pair_index", "B", "V", "bias"], data.copy()),
    },
    coords={
        "cutter_pair_index": [0],
        "B": np.arange(N_B),
        "V": np.arange(N_V),
        "bias": bias,
    }
)

T_mK = 30
ds_old = tgp.prepare.broaden_with_temperature(ds.copy(deep=True), T_mK)
ds_new = broaden_with_temperature_gpu(ds.copy(deep=True), T_mK)

for k in ("g_ll", "g_lr", "g_rl", "g_rr"):
    rel_err = np.abs((ds_old[k].values - ds_new[k].values) / ds_old[k].values)
    print(f"Max rel err for {k}: {np.max(rel_err):.8e}")
    print(f"Mean rel err for {k}: {np.mean(rel_err):.8e}")
    print(f"Median rel err for {k}: {np.median(rel_err):.8e}")

print("Test finished without asserting.")
 