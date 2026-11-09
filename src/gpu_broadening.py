"""
src/gpu_broadening.py - DEPRECATED

GPU thermal broadening has been deprecated in favor of high-performance chunked CPU thermal
broadening in src.cpu_broadening, which eliminates CuPy/CUDA driver dependencies and runs
in ~50 ms per realization with negligible RAM consumption.
"""

import warnings
from typing import Tuple
import numpy as np
import xarray as xr
from src.cpu_broadening import (
    _temp_kernel,
    _get_interp1d_weights,
    broaden_with_temperature_cpu,
    prepare_sim_cpu,
)


def broaden_with_temperature_gpu(
    ds: xr.Dataset,
    T_mK: float,
    bias_name: str = "bias",
    keys: tuple = ("g_ll", "g_lr", "g_rl", "g_rr"),
    chunk_size: int = 10000,
) -> xr.Dataset:
    warnings.warn(
        "broaden_with_temperature_gpu is deprecated; use src.cpu_broadening.broaden_with_temperature_cpu instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return broaden_with_temperature_cpu(
        ds=ds,
        T_mK=T_mK,
        bias_name=bias_name,
        keys=keys,
        chunk_size=chunk_size,
    )


def prepare_sim_gpu(ds_raw: xr.Dataset, T_mK: float) -> xr.Dataset:
    warnings.warn(
        "prepare_sim_gpu is deprecated; use src.cpu_broadening.prepare_sim_cpu instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return prepare_sim_cpu(ds_raw=ds_raw, T_mK=T_mK)
