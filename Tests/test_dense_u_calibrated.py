#!/usr/bin/env python3
"""
Tests/test_dense_u_calibrated.py
================================

Integration and contract test suite for the Calibrated Dense-U Multi-Temperature
Production Suite.

Verifies:
1. Configuration contract across YAML files (r0, r1, r2):
   - Grid: 100 points in [-5.0, +5.0] meV
   - Temperatures: [0, 10, 20, 30, 40] mK
   - 25-Cutter Harmonics: 51-node Double-Exponential Quadrature
   - 5 target G_N: [0.35, 0.49, 0.62, 0.76, 0.90]
2. Directory naming contract (strictly underscores, no decimal dots):
   - baseline_0_35G0, baseline_0_49G0, baseline_0_62G0, baseline_0_76G0, baseline_0_90G0
3. Serialized NPZ contract:
   - Keys in barrier sweep NPZ: '0mK', '10mK', '20mK', '30mK', '40mK'
   - Keys in 25-cutter harmonics NPZ: 'L_1w', 'R_1w', 'RL_2w', 'LR_2w', 'L_3w', 'R_3w'
   - Shapes and physical non-negativity of conductance.
4. SLURM script specifications:
   - Correct partition (day_3gb), cores (256), memory, time limit, and commands.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.parameter_handler import ConfigManager, SimulationConfig


def test_yaml_configuration_contracts():
    """Verify that all three production YAML files satisfy exact experiment specs."""
    configs = [
        ("Inputs/Parameters/dense_u_calibrated_r0.yaml", 0),
        ("Inputs/Parameters/dense_u_calibrated_r1.yaml", 1),
        ("Inputs/Parameters/dense_u_calibrated_r2.yaml", 2),
    ]

    for rel_path, expected_r in configs:
        full_path = REPO_ROOT / rel_path
        assert full_path.exists(), f"Missing config file: {full_path}"

        with open(full_path, "r") as f:
            raw_cfg = yaml.safe_load(f)

        assert raw_cfg["realization_index"] == expected_r
        assert raw_cfg["barrier_u_min"] == -5.0
        assert raw_cfg["barrier_u_max"] == 5.0
        assert raw_cfg["barrier_u_points"] == 100
        assert raw_cfg["barrier_T_mK_list"] == [0.0, 10.0, 20.0, 30.0, 40.0]
        assert raw_cfg["calc_25_cutter_harmonics"] is True
        assert raw_cfg["quadrature_nodes_25"] == 51
        assert raw_cfg["target_gns"] == [0.35, 0.49, 0.62, 0.76, 0.90]
        assert raw_cfg["dirname"] == f"Harpers_Ferry/Dense_U_Calibrated/Realization_{expected_r}"

        # Pydantic validation
        sim_cfg = ConfigManager.get_config(config_path=str(full_path), cli_args=[])
        assert sim_cfg.barrier_u_points == 100
        assert sim_cfg.quadrature_nodes_25 == 51


def test_slurm_script_contracts():
    """Verify Harper's Ferry SLURM batch scripts."""
    slurm_dir = REPO_ROOT / "Slurm_Scripts" / "Harpers_Ferry" / "Dense_U"
    assert slurm_dir.exists(), f"Missing SLURM directory: {slurm_dir}"

    for r in range(3):
        slurm_file = slurm_dir / f"run_dense_u_r{r}.slurm"
        assert slurm_file.exists(), f"Missing slurm file: {slurm_file}"

        content = slurm_file.read_text()
        assert f"#SBATCH -J dense_u_r{r}" in content
        assert "#SBATCH -c 256" in content
        assert "#SBATCH -p day_3gb" in content
        assert "#SBATCH -t 04:00:00" in content
        assert f"dense_u_calibrated_r{r}.yaml" in content
        assert "--workers 255" in content

    submit_sh = slurm_dir / "submit_all_dense_u.sh"
    assert submit_sh.exists()
    assert oct(submit_sh.stat().st_mode)[-3:] in ["755", "775", "777"]


def test_directory_and_npz_key_nomenclature():
    """Verify that folder names use underscores and NPZ keys match exact specifications."""
    tag_map = {
        0.35: "baseline_0_35G0",
        0.49: "baseline_0_49G0",
        0.62: "baseline_0_62G0",
        0.76: "baseline_0_76G0",
        0.90: "baseline_0_90G0"
    }

    # Verify no dots in baseline folder names
    for gn, folder in tag_map.items():
        assert "." not in folder, f"Baseline folder {folder} must not contain decimal dots!"
        assert "_" in folder

    # Verify temperature keys
    target_temps = [0.0, 10.0, 20.0, 30.0, 40.0]
    expected_t_keys = ['0mK', '10mK', '20mK', '30mK', '40mK']
    computed_keys = [f"{int(T)}mK" if T == int(T) else f"{T:.1f}mK" for T in target_temps]
    assert computed_keys == expected_t_keys


def test_end_to_end_mini_simulation(tmp_path):
    """
    Executes an end-to-end dry run for 1 pixel with 2 barrier points across all 5 temperatures
    and 25 cutters (51 nodes) to verify the data contract, folder structure, and NPZ keys.
    """
    from run_calibrated_dense_u_suite import run_production

    test_yaml = tmp_path / "test_mini.yaml"
    cfg_content = """
dirname: "Test_Mini"
realization_index: 0
lambda_dis: 4.5
V0: 1.2
Ls: 300
Ln: 0
Lb: 3
Lb_pdi: 3
a0: 100.0
ms: 0.023
Delta_0: 0.3
gamma: 0.2
barrier0: 2.0
mu_n: 0.0
qn: 20
calibrate_barriers: True
target_gns: [0.35, 0.49, 0.62, 0.76, 0.90]
barrier_u_min: -1.0
barrier_u_max: 1.0
barrier_u_points: 2
multi_baseline_barriers: True
calc_25_cutter_harmonics: True
quadrature_nodes_25: 51
barrier_conductance_mode: "finite_temperature"
barrier_T_mK_list: [0.0, 10.0, 20.0, 30.0, 40.0]
barrier_quadrature_nodes: 4
mu_max: 1.0
mu_min: 1.0
mu_dist: 0.1
Vz_max: 0.5
Vz_min: 0.5
Vz_dist: 0.1
"""
    test_yaml.write_text(cfg_content)
    out_dir = tmp_path / "output_test"

    run_production(
        config_path=str(test_yaml),
        workers=1,
        max_points=1,
        output_dir=str(out_dir)
    )

    # 1. Core outputs
    assert (out_dir / "spectrum_arr.npy").exists()
    assert (out_dir / "topological_gap.npy").exists()
    assert (out_dir / "energy_0.npy").exists()
    assert (out_dir / "weight_localization_arr.npy").exists()
    assert (out_dir / "mzm_separation_arr.npy").exists()
    assert (out_dir / "params_list.npy").exists()
    assert (out_dir / "tgp_barrier_arr.npy").exists()
    assert (out_dir / "tgp_barriers.txt").exists()
    assert (out_dir / "runtime.txt").exists()

    # 2. Barrier Sweeps per baseline
    barrier_dir = out_dir / "Barrier_Sweeps"
    assert barrier_dir.exists()

    expected_baselines = [
        "baseline_0_35G0", "baseline_0_49G0", "baseline_0_62G0",
        "baseline_0_76G0", "baseline_0_90G0"
    ]
    expected_t_keys = ["0mK", "10mK", "20mK", "30mK", "40mK"]
    obs_keys = [
        'b_right_cond_left', 'b_right_cond_right',
        'b_right_GLR', 'b_right_GRL',
        'b_left_cond_left', 'b_left_cond_right'
    ]

    for b_folder in expected_baselines:
        b_path = barrier_dir / b_folder
        assert b_path.exists(), f"Missing baseline directory: {b_path}"
        assert (b_path / "barrier_arr.npy").exists()
        b_arr = np.load(b_path / "barrier_arr.npy")
        assert len(b_arr) == 2

        for ok in obs_keys:
            npz_file = b_path / f"{ok}.npz"
            assert npz_file.exists(), f"Missing NPZ file: {npz_file}"
            with np.load(npz_file) as data:
                assert set(data.files) == set(expected_t_keys), f"Keys mismatch in {npz_file}: {data.files}"
                for tk in expected_t_keys:
                    arr = data[tk]
                    assert arr.shape == (1, 2)
                    assert np.all(np.isfinite(arr))
                    if 'cond' in ok:
                        assert np.all(arr >= -1e-6)

    # 3. 25-Cutter Harmonics
    harm_dir = out_dir / "Cutter_Harmonics_25"
    assert harm_dir.exists()
    harm_keys = ['L_1w', 'R_1w', 'RL_2w', 'LR_2w', 'L_3w', 'R_3w']

    for tk in expected_t_keys:
        h_file = harm_dir / f"harmonics_25_{tk}.npz"
        assert h_file.exists(), f"Missing harmonics file: {h_file}"
        with np.load(h_file) as h_data:
            assert set(h_data.files) == set(harm_keys), f"Keys mismatch in {h_file}: {h_data.files}"
            for hk in harm_keys:
                h_arr = h_data[hk]
                assert h_arr.shape == (1, 5, 5)
                assert np.all(np.isfinite(h_arr))

