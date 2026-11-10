#!/usr/bin/env python3
"""
Tests/test_worker_simulation_step_pipeline.py
=============================================
Regression and unit tests for the refactored modular worker subroutines:
- compute_spectral_slice
- compute_tgp_stage1_slice
- compute_tgp_stage2_slice
- NetCDF point slice assembly via xr.combine_by_coords
- TGPAdapter validation
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

from src.parameter_handler import ConfigManager, SimulationState
import src.helpers as hp
from main_parallel import (
    compute_spectral_slice,
    compute_tgp_stage1_slice,
    compute_tgp_stage2_slice,
    worker_simulation_step,
    worker_pfaffian_step,
)
from src.tgp_adapter import TGPAdapter


@pytest.fixture(scope="module")
def setup_micro_env():
    """Initializes a fast 1-point SimulationState for pipeline verification."""
    cfg = ConfigManager.initialize("test_micro_iteration.yaml")
    state = SimulationState(cfg)
    return cfg, state


def test_spectral_slice_structure(setup_micro_env):
    cfg, state = setup_micro_env
    mu = float(state.mu_var[0])
    vz = float(state.Vz_var[0])
    
    ds = compute_spectral_slice(mu=mu, vz=vz, config=cfg, state=state)
    
    assert isinstance(ds, xr.Dataset)
    assert "chemical_potential" in ds.coords
    assert "zeeman_field" in ds.coords
    assert "eigenvalue_index" in ds.coords
    assert ds.sizes["eigenvalue_index"] == 4
    assert "energy_0" in ds.data_vars
    assert "topological_gap" in ds.data_vars
    assert "spectrum" in ds.data_vars
    assert np.isfinite(ds["energy_0"].values[0, 0])
    assert np.isfinite(ds["topological_gap"].values[0, 0])


def test_tgp_stage1_slice_structure(setup_micro_env):
    cfg, state = setup_micro_env
    mu = float(state.mu_var[0])
    vz = float(state.Vz_var[0])
    
    import src.legendre_curvature as lc
    quad_dict = {40.0: lc.get_tanh_legendre_quadrature(T_mK=40.0, n_nodes=16)}
    
    ds = compute_tgp_stage1_slice(mu=mu, vz=vz, config=cfg, state=state, quad_dict=quad_dict)
    
    assert isinstance(ds, xr.Dataset)
    assert "cutter_pair_index" in ds.coords
    assert "temperature" in ds.coords
    for var in ["L_1w", "R_1w", "L_2w_nl", "R_2w_nl", "L_3w", "R_3w"]:
        assert var in ds.data_vars
        assert ds[var].shape == (3, 1, 1, 1)
        assert np.all(np.isfinite(ds[var].values))


def test_tgp_stage2_slice_structure(setup_micro_env):
    cfg, state = setup_micro_env
    mu = float(state.mu_var[0])
    vz = float(state.Vz_var[0])
    
    dynamic_energies = hp.make_dynamic_grid()
    ds = compute_tgp_stage2_slice(mu=mu, vz=vz, config=cfg, state=state, dynamic_energies=dynamic_energies)
    
    assert isinstance(ds, xr.Dataset)
    assert "cutter_pair_index" in ds.coords
    assert "bias" in ds.coords
    assert ds.sizes["bias"] == len(dynamic_energies)
    for var in ["g_ll", "g_rr", "g_lr", "g_rl", "L_SI", "R_SI"]:
        assert var in ds.data_vars
        assert np.all(np.isfinite(ds[var].values))


def test_worker_simulation_step_end_to_end(setup_micro_env):
    cfg, state = setup_micro_env
    iter_data = (0, float(state.mu_var[0]), float(state.Vz_var[0]))
    
    import src.legendre_curvature as lc
    worker_context = {
        "q_nodes": {},
        "quad_dict": {40.0: lc.get_tanh_legendre_quadrature(T_mK=40.0, n_nodes=16)},
        "dynamic_energies": hp.make_dynamic_grid(),
    }
    
    res = worker_simulation_step(iter_data, cfg, state, worker_context)
    
    assert res["i"] == 0
    assert "spectral" in res
    assert "tgp_stage1" in res
    assert "tgp_stage2" in res


def test_tgp_adapter_netcdf_compatibility(setup_micro_env):
    cfg, state = setup_micro_env
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "Outputs/Data/scratch/test_pipeline_micro"
    assert out_dir.exists()
    
    adapter = TGPAdapter(str(out_dir))
    ds = adapter.to_xarray()
    
    assert "B" in ds.dims and "B" in ds.coords
    assert "V" in ds.dims and "V" in ds.coords
    assert "bias" in ds.dims
    assert "g_ll" in ds.data_vars
    assert "g_rr" in ds.data_vars
    assert "g_lr" in ds.data_vars
    assert "g_rl" in ds.data_vars
    assert "L_SI" in ds.data_vars
    assert "R_SI" in ds.data_vars
