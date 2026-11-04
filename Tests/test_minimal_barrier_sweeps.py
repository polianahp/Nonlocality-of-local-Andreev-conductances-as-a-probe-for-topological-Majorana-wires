#!/usr/bin/env python3
"""
test_minimal_barrier_sweeps.py
==============================
Validates configuration contracts, SLURM generation, and exact numerical parity
between `run_minimal_barrier_sweeps.py` and `main_parallel.py`.
"""

import os
import shutil
import pytest
import numpy as np
from pathlib import Path

from src.parameter_handler import ConfigManager, SimulationState, PathConfigs
import src.helpers as hp
from main_parallel import worker_simulation_step
from run_minimal_barrier_sweeps import worker_minimal_barrier_step, main as runner_main


TEST_CONFIG_DIR = Path("Inputs/Parameters/Thorny_Flat_Minimal")
TEST_SLURM_DIR = Path("Slurm_Scripts/Thorny_Flat/Minimal_Barrier_Sweeps")


def test_thorny_flat_minimal_yaml_contracts():
    """Verify all 12 YAML parameter files exist, parse cleanly, and enforce contracts."""
    folders = ["V0_0_378", "V0_0_645", "V0_0_872", "V0_1_2"]
    realizations = [10, 11, 12]

    for f in folders:
        for r in realizations:
            yaml_path = TEST_CONFIG_DIR / f / f"disorder_realization_{r}.yaml"
            assert yaml_path.exists(), f"Missing YAML file: {yaml_path}"

            config = ConfigManager.get_config(str(yaml_path))
            state = SimulationState(config)

            # Contract verifications
            assert config.realization_index == r
            assert config.barrier0 == 2.0
            assert config.barrier_u_min == -5.0
            assert config.barrier_u_max == 5.0
            assert config.barrier_u_points == 20
            assert len(state.barrier_arr) == 20
            assert np.isclose(state.barrier_arr[0], -5.0)
            assert np.isclose(state.barrier_arr[-1], 5.0)
            assert config.barrier_T_mK_list == [0.0, 10.0, 20.0, 30.0, 40.0]
            assert config.barrier_quadrature_nodes == 8
            assert config.barrier_sweep_only is True
            assert config.dirname.startswith("Disorder_Realizations_ThornyFlat_Minimal")


def test_thorny_flat_minimal_slurm_contracts():
    """Verify all 12 SLURM scripts exist, target comm_small_day, have 04:30:00 walltime and 40 cores."""
    folders = ["V0_0_378", "V0_0_645", "V0_0_872", "V0_1_2"]
    realizations = [10, 11, 12]

    for f in folders:
        for r in realizations:
            slurm_path = TEST_SLURM_DIR / f"run_tf_minimal_{f}_r{r}.slurm"
            assert slurm_path.exists(), f"Missing SLURM file: {slurm_path}"

            with open(slurm_path) as fp:
                content = fp.read()

            assert "#SBATCH -p comm_small_day" in content
            assert "#SBATCH -t 04:30:00" in content
            assert "#SBATCH -c 40" in content
            assert "#SBATCH -N 1" in content
            assert "conda activate kwant_fresh" in content
            assert f"Inputs/Parameters/Thorny_Flat_Minimal/{f}/disorder_realization_{r}.yaml" in content

    launcher = TEST_SLURM_DIR / "submit_thorny_flat_minimal.sh"
    assert launcher.exists()
    assert os.access(launcher, os.X_OK)


def test_worker_numerical_parity_against_main_parallel():
    """Verify that worker_minimal_barrier_step matches main_parallel's worker_simulation_step identically."""
    config_path = TEST_CONFIG_DIR / "V0_1_2/disorder_realization_10.yaml"
    config = ConfigManager.get_config(str(config_path))
    state = SimulationState(config)

    # Load disorder seed
    resolved_path = PathConfigs.RUN_FILES / f"Raw_Disorders/raw_disorder_{config.realization_index}.npy"
    Vdisx = hp.initialize_vdis_from_data(
        resolved_path,
        lambda_dis=config.lambda_dis,
        Ls=config.Ls,
        is_real_space=False
    )
    static_params = state.get_static_params(Vdisx, config)

    target_temps = [0.0, 10.0, 20.0, 30.0, 40.0]
    q_nodes = {}
    for T_val in target_temps:
        if T_val > 0.0:
            qe, qw = hp.get_thermal_quadrature_nodes(T_mK=T_val, n_nodes=8, bias_meV=0.0)
            q_nodes[T_val] = (qe, qw)
        else:
            q_nodes[T_val] = (None, None)

    static_params['target_temps'] = target_temps
    static_params['q_nodes'] = q_nodes

    iter_data = state.params_list[0]  # (0, mu_val, vz_val)

    # 1. Run main_parallel worker step
    res_main = worker_simulation_step(iter_data, static_params)

    # 2. Run minimal runner worker step
    res_minimal = worker_minimal_barrier_step(iter_data, static_params)

    # Parity checks across all 5 temperatures
    for T_val in target_temps:
        Gmat_main = res_main['b_data'][T_val]['Gmat']
        Gmat_min = res_minimal['b_data'][T_val]['Gmat']
        np.testing.assert_allclose(Gmat_min, Gmat_main, rtol=1e-12, atol=1e-14,
                                   err_msg=f"Gmat mismatch at T = {T_val} mK")

        b_r_l_main = res_main['b_data'][T_val]['b_right_cond_left']
        b_r_l_min = res_minimal['b_data'][T_val]['b_right_cond_left']
        np.testing.assert_allclose(b_r_l_min, b_r_l_main, rtol=1e-12, atol=1e-14,
                                   err_msg=f"b_right_cond_left mismatch at T = {T_val} mK")

        b_l_r_main = res_main['b_data'][T_val]['b_left_cond_right']
        b_l_r_min = res_minimal['b_data'][T_val]['b_left_cond_right']
        np.testing.assert_allclose(b_l_r_min, b_l_r_main, rtol=1e-12, atol=1e-14,
                                   err_msg=f"b_left_cond_right mismatch at T = {T_val} mK")


def test_micro_execution_and_file_serialization(tmp_path):
    """Run a 1-pixel end-to-end dry-run and verify non-clashing file serialization."""
    test_yaml = tmp_path / "test_micro.yaml"
    out_dir_rel = "Disorder_Realizations_ThornyFlat_Minimal/Test_Micro"

    yaml_content = f"""
dirname: {out_dir_rel}
realization_index: 10
lambda_dis: 4.5
V0: 1.2
Ls: 300
a0: 100.0
barrier0: 2.0
barrier_u_min: -5.0
barrier_u_max: 5.0
barrier_u_points: 20
mu_max: 4.5
mu_min: 0.0
mu_dist: 0.02
Vz_max: 1.4
Vz_min: 0.0
Vz_dist: 0.02
acceleration_type: parallel
barrier_conductance_mode: finite_temperature
barrier_T_mK_list: [0.0, 10.0, 20.0, 30.0, 40.0]
barrier_quadrature_nodes: 8
barrier_sweep_only: true
"""
    with open(test_yaml, "w") as fp:
        fp.write(yaml_content)

    full_out_dir = PathConfigs.DATA / out_dir_rel
    if full_out_dir.exists():
        shutil.rmtree(full_out_dir)

    # Run CLI runner with --max_points 1
    import sys
    orig_argv = sys.argv
    try:
        sys.argv = ["run_minimal_barrier_sweeps.py", "--config_path", str(test_yaml), "--max_points", "1", "--num_workers", "1"]
        runner_main()
    finally:
        sys.argv = orig_argv

    assert full_out_dir.exists()

    # Verify all temperature files exist
    expected_temps = ["0mK", "10mK", "20mK", "30mK", "40mK"]
    for t_str in expected_temps:
        assert (full_out_dir / f"conductance_matrix_{t_str}.npy").exists()
        assert (full_out_dir / f"barrier_right_conductance_left_arr_{t_str}.npy").exists()
        assert (full_out_dir / f"barrier_left_conductance_right_arr_{t_str}.npy").exists()
        assert (full_out_dir / f"delta_G_asym_R_LL_{t_str}.npy").exists()
        assert (full_out_dir / f"delta_G_asym_L_RR_{t_str}.npy").exists()

        b_r_l = np.load(full_out_dir / f"barrier_right_conductance_left_arr_{t_str}.npy")
        assert b_r_l.shape == (1, 20)

    # Verify backward compatibility 0mK unsuffixed files
    assert (full_out_dir / "conductance_matrix.npy").exists()
    assert (full_out_dir / "barrier_right_conductance_left_arr.npy").exists()

    # Verify metadata
    assert (full_out_dir / "barrier_arr.npy").exists()
    assert (full_out_dir / "runtime.txt").exists()

    # Clean up test output
    shutil.rmtree(full_out_dir)
