#!/usr/bin/env python3
"""
Calibrated Dense-U Multi-Temperature Production Suite Runner
============================================================

Orchestrates full-scale production simulations on Harper's Ferry across
multiple disorder realizations with:
1. Automated exact self-energy barrier calibration targeting G_N in {0.35, 0.49, 0.62, 0.76, 0.90} e^2/h.
2. Dense 100-point barrier sweeps across U in [-5.0, +5.0] meV for EACH of the 5 calibrated symmetric baselines.
3. Thermal broadening across T in {0, 10, 20, 30, 40} mK.
4. 25-Cutter Consensus Harmonics (1w, 2w, 3w) via 51-node Double-Exponential Quadrature.

Usage:
  # Dry-run test on 2 pixels locally (takes ~15-20 s):
  python run_calibrated_dense_u_suite.py --config Inputs/Parameters/dense_u_calibrated_r0.yaml --dry-run --workers 2

  # Full cluster production run (255 parallel workers):
  python run_calibrated_dense_u_suite.py --config Inputs/Parameters/dense_u_calibrated_r0.yaml --workers 255
"""

import os
import sys
import time

# 1. THREAD PINNING: Must be set BEFORE importing numpy/scipy/kwant
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from functools import partial
import itertools as itr
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.helpers as hp
import src.barrier_calibration as bc
import src.legendre_curvature as lc
from src.config import PathConfigs
from src.parameter_handler import ConfigManager, SimulationConfig, SimulationState


# =============================================================================
# WORKER FUNCTION (Module level for multiprocessing pickling)
# =============================================================================

def worker_dense_u_step(
    iter_data: Tuple[int, float, float],
    static_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Worker task for a single (mu, Vz) parameter pixel.
    
    Executes:
    1. Closed wire BdG spectrum & topological gap (legacy renormalized Hamiltonian).
    2. Dense 100-point barrier sweeps for each of the 5 calibrated baselines across all temperatures.
    3. 25-cutter harmonics (1w, 2w, 3w) via 51-node Double-Exponential quadrature.
    """
    idx, mu, vz = iter_data

    # Unpack physics parameters
    t = static_data['t']
    mu_n = static_data['mu_n']
    Delta0 = static_data['Delta0']
    gamma = static_data['gamma']
    alpha = static_data['alpha']
    Ln = static_data['Ln']
    Lb = static_data['Lb']
    Ls = static_data['Ls']
    mu_leads = static_data['mu_leads']
    Vdisx = static_data['Vdisx']
    num_eigenvalues = static_data['num_eigenvalues']
    solver_type = static_data.get('solver_type', 'cpu')

    u_cutters = static_data['u_cutters']          # 5 calibrated cutter voltages
    barrier_arr = static_data['barrier_arr']      # 100 U points in [-5, +5] meV
    target_temps = static_data['target_temps']    # [0.0, 10.0, 20.0, 30.0, 40.0]
    gl_quad = static_data['gl_quad']              # Gauss-Legendre quadrature nodes for barrier sweeps
    de_quad = static_data['de_quad']              # Double-exponential quadrature nodes for 25 cutters

    num_u = len(barrier_arr)
    num_baselines = len(u_cutters)

    # -------------------------------------------------------------------------
    # 1. Closed Wire BdG Spectrum & Topological Gap
    # -------------------------------------------------------------------------
    syst_closed = hp.build_system_closed(t, mu, gamma, Delta0, vz, alpha, Ls, Vdisx)
    evals, evecs = hp.solve_ham(syst_closed, solver_type=solver_type, k=num_eigenvalues)

    spectrum = evals[:4] if len(evals) >= 4 else evals
    pos_evals = np.sort(evals[evals >= 0])
    energy_0 = float(pos_evals[0]) if len(pos_evals) > 0 else np.nan
    topological_gap = float(pos_evals[1]) if len(pos_evals) > 1 else np.nan

    weight_loc = 1.0
    mzm_sep = 0.0
    if energy_0 <= 0.1 and static_data.get('localization_flag', False) and evecs is not None:
        rho_M1, rho_M2, _ = hp.get_psiM_density(evals, evecs)
        weight_loc = float(hp.calc_weight_localization(rho_M1, rho_M2, weight_threshold=static_data.get('weight_threshold', 0.8)))
        mzm_sep = float(hp.calc_MZM_separation(rho_M1, rho_M2))

    # -------------------------------------------------------------------------
    # 2. Dense 100-Point Barrier Sweeps for Each of the 5 Calibrated Baselines
    # -------------------------------------------------------------------------
    # Structure: sweeps_by_baseline[b_idx][T_val] = { 'b_right_cond_left': ..., etc. }
    sweeps_by_baseline = []

    for b_idx, u_base in enumerate(u_cutters):
        base_dict: Dict[float, Dict[str, np.ndarray]] = {}
        for T_val in target_temps:
            base_dict[T_val] = {
                'b_right_cond_left': np.zeros(num_u, dtype=np.float64),
                'b_right_cond_right': np.zeros(num_u, dtype=np.float64),
                'b_right_GLR': np.zeros(num_u, dtype=np.float64),
                'b_right_GRL': np.zeros(num_u, dtype=np.float64),
                'b_left_cond_left': np.zeros(num_u, dtype=np.float64),
                'b_left_cond_right': np.zeros(num_u, dtype=np.float64),
            }

        for k, u_var in enumerate(barrier_arr):
            # A. Vary Right Barrier (UL = u_base, UR = u_var)
            syst_UR = hp.build_system(
                t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz,
                alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
                barrier_l=u_base, barrier_r=u_var, Vdisx=Vdisx
            )
            for T_val in target_temps:
                if T_val > 0.0:
                    qe, qw = gl_quad[T_val]
                    Gmat_UR = hp.calc_conductance_matrix_finite_T(syst_UR, qe, qw, solver_type=solver_type)
                else:
                    Gmat_UR = hp.calc_conductance_matrix(syst_UR, eng=0.0, solver_type=solver_type)
                base_dict[T_val]['b_right_cond_left'][k] = Gmat_UR[0, 0]
                base_dict[T_val]['b_right_cond_right'][k] = Gmat_UR[1, 1]
                base_dict[T_val]['b_right_GRL'][k] = Gmat_UR[1, 0]
                base_dict[T_val]['b_right_GLR'][k] = Gmat_UR[0, 1]

            # B. Vary Left Barrier (UL = u_var, UR = u_base)
            syst_UL = hp.build_system(
                t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz,
                alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
                barrier_l=u_var, barrier_r=u_base, Vdisx=Vdisx
            )
            for T_val in target_temps:
                if T_val > 0.0:
                    qe, qw = gl_quad[T_val]
                    Gmat_UL = hp.calc_conductance_matrix_finite_T(syst_UL, qe, qw, solver_type=solver_type)
                else:
                    Gmat_UL = hp.calc_conductance_matrix(syst_UL, eng=0.0, solver_type=solver_type)
                base_dict[T_val]['b_left_cond_left'][k] = Gmat_UL[0, 0]
                base_dict[T_val]['b_left_cond_right'][k] = Gmat_UL[1, 1]

        sweeps_by_baseline.append(base_dict)

    # -------------------------------------------------------------------------
    # 3. 25-Cutter Harmonics (1w, 2w, 3w) via Double-Exponential Quadrature
    # -------------------------------------------------------------------------
    # Harmonics across all 25 pairs: shape (5, 5) per observable and per temperature
    harmonics_by_temp: Dict[float, Dict[str, np.ndarray]] = {}
    for T_val in target_temps:
        harmonics_by_temp[T_val] = {
            'L_1w': np.zeros((num_baselines, num_baselines), dtype=np.float64),
            'R_1w': np.zeros((num_baselines, num_baselines), dtype=np.float64),
            'RL_2w': np.zeros((num_baselines, num_baselines), dtype=np.float64),
            'LR_2w': np.zeros((num_baselines, num_baselines), dtype=np.float64),
            'L_3w': np.zeros((num_baselines, num_baselines), dtype=np.float64),
            'R_3w': np.zeros((num_baselines, num_baselines), dtype=np.float64),
        }

    for i_l, ul in enumerate(u_cutters):
        for j_r, ur in enumerate(u_cutters):
            syst_pair = hp.build_system(
                t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz,
                alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
                barrier_l=ul, barrier_r=ur, Vdisx=Vdisx
            )

            # A. Zero Temperature (T = 0)
            G0_zero = hp.calc_conductance_matrix(syst_pair, eng=0.0, solver_type=solver_type)
            harmonics_by_temp[0.0]['L_1w'][i_l, j_r] = G0_zero[0, 0]
            harmonics_by_temp[0.0]['R_1w'][i_l, j_r] = G0_zero[1, 1]
            harmonics_by_temp[0.0]['RL_2w'][i_l, j_r] = G0_zero[1, 0]
            harmonics_by_temp[0.0]['LR_2w'][i_l, j_r] = G0_zero[0, 1]
            harmonics_by_temp[0.0]['L_3w'][i_l, j_r] = 0.0
            harmonics_by_temp[0.0]['R_3w'][i_l, j_r] = 0.0

            # B. Finite Temperatures via 51-node Double-Exponential Quadrature
            for T_val in target_temps:
                if T_val <= 0.0:
                    continue
                q = de_quad[T_val]
                e_nodes = q['energies']
                w_1w = q['weights_1w']
                w_2w = q['weights_2w']
                w_3w = q['weights_3w']

                sum_L_1w = 0.0
                sum_R_1w = 0.0
                sum_RL_2w = 0.0
                sum_LR_2w = 0.0
                sum_L_3w = 0.0
                sum_R_3w = 0.0

                for k_node, eng in enumerate(e_nodes):
                    G_mat = hp.calc_conductance_matrix(syst_pair, eng=float(eng), solver_type=solver_type)
                    g_ll = G_mat[0, 0]
                    g_rr = G_mat[1, 1]
                    g_rl = G_mat[1, 0]
                    g_lr = G_mat[0, 1]

                    sum_L_1w += w_1w[k_node] * g_ll
                    sum_R_1w += w_1w[k_node] * g_rr
                    sum_RL_2w += w_2w[k_node] * g_rl
                    sum_LR_2w += w_2w[k_node] * g_lr
                    sum_L_3w += w_3w[k_node] * g_ll
                    sum_R_3w += w_3w[k_node] * g_rr

                harmonics_by_temp[T_val]['L_1w'][i_l, j_r] = sum_L_1w
                harmonics_by_temp[T_val]['R_1w'][i_l, j_r] = sum_R_1w
                harmonics_by_temp[T_val]['RL_2w'][i_l, j_r] = sum_RL_2w
                harmonics_by_temp[T_val]['LR_2w'][i_l, j_r] = sum_LR_2w
                harmonics_by_temp[T_val]['L_3w'][i_l, j_r] = sum_L_3w
                harmonics_by_temp[T_val]['R_3w'][i_l, j_r] = sum_R_3w

    return {
        'idx': idx,
        'spectrum': spectrum,
        'energy_0': energy_0,
        'topological_gap': topological_gap,
        'weight_loc': weight_loc,
        'mzm_sep': mzm_sep,
        'sweeps_by_baseline': sweeps_by_baseline,
        'harmonics_by_temp': harmonics_by_temp
    }


# =============================================================================
# MAIN PRODUCTION DRIVER
# =============================================================================

def run_production(
    config_path: str,
    realization_index: int = None,
    workers: int = 1,
    dry_run: bool = False,
    max_points: int = None,
    output_dir: str = None
):
    t_start = time.perf_counter()

    cfg = ConfigManager.get_config(config_path=config_path, cli_args=[])
    if realization_index is not None:
        cfg.realization_index = realization_index

    # Resolve output directory
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = PathConfigs.DATA / cfg.dirname

    # Resolve disorder file
    resolved_disorder = None
    if cfg.realization_index is not None:
        raw_name = f"Raw_Disorders/raw_disorder_{cfg.realization_index}.npy"
        p_cand = PathConfigs.RUN_FILES / raw_name
        if p_cand.exists():
            resolved_disorder = p_cand
    if resolved_disorder is None and cfg.fname:
        p_cand = PathConfigs.RUN_FILES / "New_Disorders" / cfg.fname
        if p_cand.exists():
            resolved_disorder = p_cand
        else:
            resolved_disorder = PathConfigs.RUN_FILES / cfg.fname

    print("=" * 75)
    print(" CALIBRATED DENSE-U MULTI-TEMPERATURE PRODUCTION SUITE")
    print(f" Config Path:         {config_path}")
    print(f" Output Directory:    {cfg.dirname}")
    print(f" Realization Index:   {cfg.realization_index}")
    print(f" Resolved Disorder:   {resolved_disorder}")
    print(f" Dense Barrier Grid:  [{cfg.barrier_u_min}, {cfg.barrier_u_max}] meV ({cfg.barrier_u_points} points)")
    print(f" Temperatures (mK):   {cfg.barrier_T_mK_list}")
    print(f" 25-Cutter Harmonics: Double-Exponential ({cfg.quadrature_nodes_25} nodes)")
    print(f" Parallel Workers:    {workers}")
    print(f" Mode:                {'DRY-RUN (5 points)' if dry_run else 'FULL PRODUCTION'}")
    print("=" * 75)

    # Initialize State
    state = SimulationState(cfg)
    vdis_raw = hp.initialize_vdis_from_data(
        resolved_disorder, lambda_dis=cfg.lambda_dis, Ls=cfg.Ls,
        is_real_space=getattr(cfg, 'is_real_space', False)
    )
    vdis_scaled = vdis_raw * cfg.V0

    # 1. Exact Cutter Barrier Calibration
    targets = cfg.target_gns if cfg.target_gns is not None else [0.35, 0.49, 0.62, 0.76, 0.90]
    print(f"\n--- Step 1: Exact Cutter Barrier Calibration for targets: {targets} ---")
    if hasattr(state, 'tgp_barrier_arr') and state.tgp_barrier_arr is not None and len(state.tgp_barrier_arr) == len(targets):
        u_cutters = np.array(state.tgp_barrier_arr, dtype=float)
        print("Reusing pre-calibrated barrier array from simulation state.")
    else:
        calib_res = bc.calibrate_wire_cutters(
            target_gns=targets,
            t=state.t,
            gamma=cfg.gamma,
            Delta0=cfg.Delta_0,
            alpha=state.alpha,
            Ln=cfg.Ln,
            Lb=cfg.Lb,
            Ls=cfg.Ls,
            mu_leads=state.mu_leads
        )
        u_cutters = np.array(calib_res['u_left'], dtype=float)
    print(f"Calibrated Cutter Potentials (meV): {u_cutters.tolist()}")

    # 2. Precompute Quadratures
    target_temps = [float(T) for T in cfg.barrier_T_mK_list]
    print("\n--- Step 2: Precomputing Quadrature Weights ---")
    gl_quad: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    de_quad: Dict[float, Dict[str, Any]] = {}

    for T_val in target_temps:
        if T_val > 0.0:
            gl_quad[T_val] = hp.get_thermal_quadrature_nodes(
                T_mK=T_val, n_nodes=cfg.barrier_quadrature_nodes, bias_meV=0.0
            )
            de_quad[T_val] = lc.get_double_exponential_quadrature(
                T_mK=T_val, n_nodes=cfg.quadrature_nodes_25
            )
            print(f"  T = {T_val:4.1f} mK: GL ({cfg.barrier_quadrature_nodes} nodes), DE ({de_quad[T_val]['n_nodes']} nodes)")
        else:
            gl_quad[0.0] = (None, None)
            de_quad[0.0] = None
            print(f"  T =  0.0 mK: Zero-bias ground state")

    # Static data bundle for workers
    static_data = {
        't': state.t,
        'mu_n': cfg.mu_n,
        'Delta0': cfg.Delta_0,
        'gamma': cfg.gamma,
        'alpha': state.alpha,
        'Ln': cfg.Ln,
        'Lb': cfg.Lb,
        'Ls': cfg.Ls,
        'mu_leads': state.mu_leads,
        'Vdisx': vdis_scaled,
        'num_eigenvalues': cfg.num_eigenvalues,
        'localization_flag': cfg.localization_flag,
        'weight_threshold': cfg.weight_threshold,
        'solver_type': state.solver_type,
        'u_cutters': u_cutters,
        'barrier_arr': state.barrier_arr,
        'target_temps': target_temps,
        'gl_quad': gl_quad,
        'de_quad': de_quad
    }

    params_list = state.params_list
    if dry_run:
        params_list = params_list[:5]
    elif max_points is not None:
        params_list = params_list[:max_points]

    num_pixels = len(params_list)
    print(f"\n--- Step 3: Starting Parallel Execution across {num_pixels} Pixels ---")

    results = []
    if workers > 1:
        with mp.Pool(processes=workers) as pool:
            func = partial(worker_dense_u_step, static_data=static_data)
            results = list(tqdm(pool.imap(func, params_list, chunksize=1), total=num_pixels, desc="Pixel Sweep"))
    else:
        for p_item in tqdm(params_list, desc="Pixel Sweep"):
            results.append(worker_dense_u_step(p_item, static_data))

    # -------------------------------------------------------------------------
    # 4. Organizing & Serializing Results
    # -------------------------------------------------------------------------
    print(f"\n--- Step 4: Serializing Data to {out_dir} ---")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Core Arrays
    spectrum_arr = np.zeros((num_pixels, 4), dtype=np.float64)
    topological_gap_arr = np.zeros(num_pixels, dtype=np.float64)
    energy_0_arr = np.zeros(num_pixels, dtype=np.float64)
    weight_loc_arr = np.zeros(num_pixels, dtype=np.float64)
    mzm_sep_arr = np.zeros(num_pixels, dtype=np.float64)

    # Baselines tagged by target G_N with underscores (e.g. baseline_0_35G0/)
    tag_map = {
        0.35: "baseline_0_35G0",
        0.49: "baseline_0_49G0",
        0.62: "baseline_0_62G0",
        0.76: "baseline_0_76G0",
        0.90: "baseline_0_90G0"
    }

    # Pre-allocate barrier sweep arrays: sweep_data[b_idx][obs_key][T_key] -> np.ndarray (num_pixels, 100)
    obs_keys = [
        'b_right_cond_left', 'b_right_cond_right',
        'b_right_GLR', 'b_right_GRL',
        'b_left_cond_left', 'b_left_cond_right'
    ]
    
    num_u = len(state.barrier_arr)
    num_baselines = len(u_cutters)
    t_keys = [f"{int(T)}mK" if T == int(T) else f"{T:.1f}mK" for T in target_temps]

    sweep_tables = []
    for b_idx in range(num_baselines):
        b_dict = {}
        for ok in obs_keys:
            b_dict[ok] = {tk: np.zeros((num_pixels, num_u), dtype=np.float64) for tk in t_keys}
        sweep_tables.append(b_dict)

    # Pre-allocate 25-cutter harmonics: harm_tables[T_key][harm_key] -> np.ndarray (num_pixels, 5, 5)
    harm_keys = ['L_1w', 'R_1w', 'RL_2w', 'LR_2w', 'L_3w', 'R_3w']
    harm_tables = {
        tk: {hk: np.zeros((num_pixels, num_baselines, num_baselines), dtype=np.float64) for hk in harm_keys}
        for tk in t_keys
    }

    # Fill pre-allocated tables
    for res in results:
        i = res['idx']
        spectrum_arr[i, :] = res['spectrum']
        topological_gap_arr[i] = res['topological_gap']
        energy_0_arr[i] = res['energy_0']
        weight_loc_arr[i] = res['weight_loc']
        mzm_sep_arr[i] = res['mzm_sep']

        # Barrier Sweeps
        for b_idx in range(num_baselines):
            for T_val, tk in zip(target_temps, t_keys):
                p_res = res['sweeps_by_baseline'][b_idx][T_val]
                for ok in obs_keys:
                    sweep_tables[b_idx][ok][tk][i, :] = p_res[ok]

        # 25-Cutter Harmonics
        for T_val, tk in zip(target_temps, t_keys):
            h_res = res['harmonics_by_temp'][T_val]
            for hk in harm_keys:
                harm_tables[tk][hk][i, :, :] = h_res[hk]

    # Write Core Metadata & Arrays
    np.save(out_dir / "spectrum_arr.npy", spectrum_arr)
    np.save(out_dir / "topological_gap.npy", topological_gap_arr)
    np.save(out_dir / "energy_0.npy", energy_0_arr)
    np.save(out_dir / "weight_localization_arr.npy", weight_loc_arr)
    np.save(out_dir / "mzm_separation_arr.npy", mzm_sep_arr)
    np.save(out_dir / "params_list.npy", params_list)
    np.save(out_dir / "Vdisx.npy", vdis_scaled)
    np.save(out_dir / "tgp_barrier_arr.npy", u_cutters)

    # Write text audit log
    with open(out_dir / "tgp_barriers.txt", "w") as f:
        f.write(f"Simulation Directory: {cfg.dirname}\n")
        f.write(f"Realization Index: {cfg.realization_index}\n")
        f.write(f"Target Conductances G_N (e^2/h): {targets}\n")
        f.write(f"Calibrated Cutter Potentials (meV): {u_cutters.tolist()}\n")
        for b_i, (tg, uv) in enumerate(zip(targets, u_cutters)):
            f.write(f"  Baseline {b_i} (target {tg:.2f} G_0): {uv:.6f} meV\n")

    # Write Barrier Sweeps per Baseline
    barrier_dir = out_dir / "Barrier_Sweeps"
    barrier_dir.mkdir(parents=True, exist_ok=True)

    for b_idx, target_gn in enumerate(targets):
        folder_name = tag_map.get(target_gn, f"baseline_{b_idx}")
        base_dir = barrier_dir / folder_name
        base_dir.mkdir(parents=True, exist_ok=True)
        np.save(base_dir / "barrier_arr.npy", state.barrier_arr)

        for ok in obs_keys:
            # npz file where keys are '0mK', '10mK', '20mK', '30mK', '40mK'
            npz_dict = {tk: sweep_tables[b_idx][ok][tk] for tk in t_keys}
            np.savez_compressed(base_dir / f"{ok}.npz", **npz_dict)

    # Write 25-Cutter Harmonics
    harm_dir = out_dir / "Cutter_Harmonics_25"
    harm_dir.mkdir(parents=True, exist_ok=True)

    for T_val, tk in zip(target_temps, t_keys):
        # npz file where keys are 'L_1w', 'R_1w', 'RL_2w', 'LR_2w', 'L_3w', 'R_3w'
        np.savez_compressed(harm_dir / f"harmonics_25_{tk}.npz", **harm_tables[tk])

    t_end = time.perf_counter()
    runtime_sec = t_end - t_start
    runtime_min = runtime_sec / 60.0

    print("=" * 75)
    print(f" Run Complete in {runtime_sec:.2f} s ({runtime_min:.2f} min)")
    print(f" Data saved to: {out_dir}")
    print("=" * 75)

    with open(out_dir / "runtime.txt", "w") as rf:
        rf.write(f"Total Runtime Seconds: {runtime_sec:.4f}\n")
        rf.write(f"Total Runtime Minutes: {runtime_min:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrated Dense-U Multi-Temperature Production Suite")
    parser.add_argument("--config", "--config_path", dest="config", type=str, default="Inputs/Parameters/dense_u_calibrated_r0.yaml", help="Path to config YAML")
    parser.add_argument("--realization-index", type=int, default=None, help="Override disorder realization index")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument("--dry-run", action="store_true", help="Run 5 pixels for verification")
    parser.add_argument("--max-points", type=int, default=None, help="Maximum number of points to run")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")

    args = parser.parse_args()
    run_production(
        config_path=args.config,
        realization_index=args.realization_index,
        workers=args.workers,
        dry_run=args.dry_run,
        max_points=args.max_points,
        output_dir=args.output_dir
    )
