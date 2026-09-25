import pytest
import numpy as np
import xarray as xr
import sys
import os

# Add parent directory to path so that helpers can be imported when running pytest directly from the Tests directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.helpers as hp

def test_gpu_broadening():
    if not hp.GPU_AVAILABLE:
        pytest.skip("GPU not available")
        
    from src.gpu_broadening import broaden_with_temperature_gpu
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
        np.testing.assert_allclose(
            ds_new[k].values, 
            ds_old[k].values, 
            rtol=1e-5, 
            atol=1e-8,
            err_msg=f"Mismatch in {k} values after GPU broadening"
        )

if __name__ == "__main__":
    test_gpu_broadening()
