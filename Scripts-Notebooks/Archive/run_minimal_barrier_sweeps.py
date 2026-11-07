#!/usr/bin/env python3
"""
run_minimal_barrier_sweeps.py
==============================
High-throughput, minimal barrier sweep runner for Thorny Flat cluster deployment.

Computes exact zero-bias barrier sweep conductances across 5 experimental temperatures:
T in {0, 10, 20, 30, 40} mK.

For each (mu, Vz) pixel:
1. Symmetric reference conductance matrix G_sym at (barrier_l = barrier0, barrier_r = barrier0).
2. Right barrier sweep: (barrier_l = barrier0, barrier_r = U_k) for k in 0..19.
3. Left barrier sweep:  (barrier_l = U_k, barrier_r = barrier0) for k in 0..19.

At T = 0 mK: single Kwant scattering matrix solve at E = 0.
At T > 0 mK: 8-node Gauss-Legendre thermal quadrature over [-10 k_B T, +10 k_B T].

Outputs are saved with explicit temperature suffixes (e.g. _0mK, _10mK, _20mK, _30mK, _40mK)
to dedicated directories under Outputs/Data/Disorder_Realizations_ThornyFlat_Minimal/.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import multiprocessing as mp
import numpy as np
from tqdm import tqdm

import src.helpers as hp
from src.parameter_handler import ConfigManager, SimulationState, PathConfigs


def worker_minimal_barrier_step(iter_data, static_params):
    """
    Worker function evaluating symmetric reference and asymmetric barrier sweeps
    for a single (mu, Vz) parameter pixel.
    """
    i, mu, vz = iter_data

    # Unpack static physics parameters
    t = static_params['t']
    mu_n = static_params['mu_n']
    Delta0 = static_params['Delta0']
    gamma = static_params['gamma']
    alpha = static_params['alpha']
    Ln = static_params['Ln']
    Lb = static_params['Lb']
    Ls = static_params['Ls']
    mu_leads = static_params['mu_leads']
    barrier0 = static_params['barrier0']
    Vdisx = static_params['Vdisx']
    V0 = static_params['V0']
    barrier_arr = static_params['barrier_arr']
    solver_type = static_params.get('solver_type', 'cpu')
    target_temps = static_params['target_temps']
    q_nodes = static_params['q_nodes']

    Vdisx_scaled = Vdisx * V0
    barrier_tot = barrier0
    points = len(barrier_arr)

    b_data = {}
    for T_val in target_temps:
        b_data[T_val] = {
            'Gmat': np.zeros((2, 2), dtype=np.float64),
            'b_right_cond_left': np.zeros(points, dtype=np.float64),
            'b_right_cond_right': np.zeros(points, dtype=np.float64),
            'b_right_GLR': np.zeros(points, dtype=np.float64),
            'b_right_GRL': np.zeros(points, dtype=np.float64),
            'b_left_cond_left': np.zeros(points, dtype=np.float64),
            'b_left_cond_right': np.zeros(points, dtype=np.float64),
        }

    # 1. Symmetric Reference System (UL = barrier_tot, UR = barrier_tot)
    syst = hp.build_system(
        t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz,
        alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
        barrier_l=barrier_tot, barrier_r=barrier_tot, Vdisx=Vdisx_scaled
    )
    for T_val in target_temps:
        if T_val > 0.0:
            qe, qw = q_nodes[T_val]
            b_data[T_val]['Gmat'] = hp.calc_conductance_matrix_finite_T(syst, qe, qw, solver_type=solver_type)
        else:
            b_data[T_val]['Gmat'] = hp.calc_conductance_matrix(syst, 0.0, solver_type=solver_type)

    # 2. Barrier Sweeps Across Grid Points
    for k in range(points):
        barrier_var_tot = barrier_arr[k]

        # Varying Right Barrier (UL = barrier_tot, UR = barrier_var_tot)
        syst_UR = hp.build_system(
            t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma,
            V_z=vz, alpha=alpha, Ln=Ln, Lb=Lb,
            Ls=Ls, mu_leads=mu_leads, barrier_l=barrier_tot,
            barrier_r=barrier_var_tot, Vdisx=Vdisx_scaled
        )
        for T_val in target_temps:
            if T_val > 0.0:
                qe, qw = q_nodes[T_val]
                Gmat_UR = hp.calc_conductance_matrix_finite_T(syst_UR, qe, qw, solver_type=solver_type)
            else:
                Gmat_UR = hp.calc_conductance_matrix(syst_UR, eng=0.0, solver_type=solver_type)
            b_data[T_val]['b_right_cond_left'][k] = Gmat_UR[0, 0]
            b_data[T_val]['b_right_cond_right'][k] = Gmat_UR[1, 1]
            b_data[T_val]['b_right_GRL'][k] = Gmat_UR[1, 0]
            b_data[T_val]['b_right_GLR'][k] = Gmat_UR[0, 1]

        # Varying Left Barrier (UL = barrier_var_tot, UR = barrier_tot)
        syst_UL = hp.build_system(
            t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma,
            V_z=vz, alpha=alpha, Ln=Ln, Lb=Lb,
            Ls=Ls, mu_leads=mu_leads, barrier_l=barrier_var_tot,
            barrier_r=barrier_tot, Vdisx=Vdisx_scaled
        )
        for T_val in target_temps:
            if T_val > 0.0:
                qe, qw = q_nodes[T_val]
                Gmat_UL = hp.calc_conductance_matrix_finite_T(syst_UL, qe, qw, solver_type=solver_type)
            else:
                Gmat_UL = hp.calc_conductance_matrix(syst_UL, eng=0.0, solver_type=solver_type)
            b_data[T_val]['b_left_cond_left'][k] = Gmat_UL[0, 0]
            b_data[T_val]['b_left_cond_right'][k] = Gmat_UL[1, 1]

    return {
        'i': i,
        'b_data': b_data
    }


def main():
    parser = argparse.ArgumentParser(description="Thorny Flat Minimal Barrier Sweep Runner")
    parser.add_argument(
        "--config_path",
        type=str,
        default="Inputs/Parameters/custom_protocol.yaml",
        help="Path to YAML parameter file"
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=None,
        help="Maximum parameter points to simulate (for testing/dry-runs)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override worker process count"
    )

    args = parser.parse_args()

    # Load configuration
    config = ConfigManager.get_config(config_path=args.config_path)
    state = SimulationState(config)
    dirname = config.dirname

    # Resolve disorder file path
    resolved_path = None
    if config.realization_index is not None:
        resolved_path = PathConfigs.RUN_FILES / f"Raw_Disorders/raw_disorder_{config.realization_index}.npy"
    elif config.fname:
        raw_p = Path(config.fname)
        candidates = [
            PathConfigs.RUN_FILES / config.fname,
            raw_p.resolve() if raw_p.is_absolute() else None,
            Path(config.fname).resolve() if Path(config.fname).exists() else None,
            PathConfigs.RUN_FILES / "New_Disorders" / config.fname,
            PathConfigs.RUN_FILES / "six_micron_analysis" / "6micron" / "New_Disorders" / raw_p.name,
            PathConfigs.RUN_FILES / "six_micron_analysis" / "3micron" / "New_Disorders" / raw_p.name,
            PathConfigs.RUN_FILES / "six_micron" / "New_Disorders" / raw_p.name,
        ]
        for c in candidates:
            if c and c.exists():
                resolved_path = c.resolve()
                break
        if resolved_path is None:
            matches = list(PathConfigs.RUN_FILES.glob(f"**/{raw_p.name}"))
            if matches:
                resolved_path = matches[0].resolve()
            else:
                resolved_path = PathConfigs.RUN_FILES / config.fname
    else:
        resolved_path = PathConfigs.RUN_FILES / "New_Disorders/Tdis.npz"

    try:
        fname = str(resolved_path.relative_to(PathConfigs.RUN_FILES))
    except (ValueError, Exception):
        fname = str(resolved_path)

    # Initialize disorder potential
    Vdisx = hp.initialize_vdis_from_data(
        resolved_path,
        lambda_dis=config.lambda_dis,
        Ls=config.Ls,
        is_real_space=getattr(config, 'is_real_space', False)
    )

    # Build static parameters dictionary
    static_params = state.get_static_params(Vdisx, config)

    # Resolve target temperatures
    t_list = config.barrier_T_mK_list
    if t_list and len(t_list) > 0:
        target_temps = [float(t) for t in t_list]
    else:
        target_temps = [0.0, 10.0, 20.0, 30.0, 40.0]

    # Precompute Gauss-Legendre thermal quadrature nodes & weights for T > 0
    n_quad = getattr(config, 'barrier_quadrature_nodes', 8)
    q_nodes = {}
    for T_val in target_temps:
        if T_val > 0.0:
            qe, qw = hp.get_thermal_quadrature_nodes(
                T_mK=T_val,
                n_nodes=n_quad,
                bias_meV=0.0
            )
            q_nodes[T_val] = (qe, qw)
        else:
            q_nodes[T_val] = (None, None)

    static_params['target_temps'] = target_temps
    static_params['q_nodes'] = q_nodes

    # Parameter grid list
    params_list = state.params_list
    if args.max_points is not None:
        params_list = params_list[:args.max_points]
    elif getattr(config, 'max_points', None) is not None:
        params_list = params_list[:config.max_points]

    num_pixels = len(params_list)
    barrier_arr = state.barrier_arr
    num_barrier_points = len(barrier_arr)

    print("==================================================================")
    print(" Thorny Flat Minimal Barrier Sweep Runner")
    print(f" Output Directory:    {dirname}")
    print(f" Disorder File:       {fname} (V0 = {config.V0} meV)")
    print(f" Realization Index:   {config.realization_index}")
    print(f" Grid Pixels:         {num_pixels}")
    print(f" Barrier Points:      {num_barrier_points} (min={barrier_arr[0]:.2f}, max={barrier_arr[-1]:.2f} meV)")
    print(f" Baseline Barrier U0: {config.barrier0:.4f} meV")
    print(f" Target Temperatures: {target_temps} mK (quadrature nodes: {n_quad})")
    print("==================================================================")

    # Log artifact metadata
    ConfigManager.log_artifact(config, PathConfigs.DATA / dirname)

    # Determine worker process count
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = max(1, mp.cpu_count() - 1)

    print(f"Spawning {num_workers} parallel workers on {mp.cpu_count()} CPU cores...")
    start_time = time.time()

    # Pre-allocate output buffers
    conductance_data_by_temp = {}
    for T_val in target_temps:
        conductance_data_by_temp[T_val] = {
            'Conductance_matrix': np.zeros((num_pixels, 2, 2), dtype=np.float64),
            'barrier_right_conductance_left_arr': np.zeros((num_pixels, num_barrier_points), dtype=np.float64),
            'barrier_right_conductance_right_arr': np.zeros((num_pixels, num_barrier_points), dtype=np.float64),
            'barrier_right_GLR_arr': np.zeros((num_pixels, num_barrier_points), dtype=np.float64),
            'barrier_right_GRL_arr': np.zeros((num_pixels, num_barrier_points), dtype=np.float64),
            'barrier_left_conductance_left_arr': np.zeros((num_pixels, num_barrier_points), dtype=np.float64),
            'barrier_left_conductance_right_arr': np.zeros((num_pixels, num_barrier_points), dtype=np.float64),
        }

    # Parallel Execution Pool
    with mp.Pool(processes=num_workers) as pool:
        args_list = [(pms, static_params) for pms in params_list]
        for res in tqdm(pool.starmap(worker_minimal_barrier_step, args_list), total=num_pixels, desc="Barrier Sweeps"):
            idx = res['i']
            res_bdata = res['b_data']
            for T_val in target_temps:
                td = res_bdata[T_val]
                cd = conductance_data_by_temp[T_val]
                cd['Conductance_matrix'][idx, :, :] = td['Gmat']
                cd['barrier_right_conductance_left_arr'][idx, :] = td['b_right_cond_left']
                cd['barrier_right_conductance_right_arr'][idx, :] = td['b_right_cond_right']
                cd['barrier_right_GLR_arr'][idx, :] = td['b_right_GLR']
                cd['barrier_right_GRL_arr'][idx, :] = td['b_right_GRL']
                cd['barrier_left_conductance_left_arr'][idx, :] = td['b_left_cond_left']
                cd['barrier_left_conductance_right_arr'][idx, :] = td['b_left_cond_right']

    elapsed_time = time.time() - start_time
    print(f"\nSimulation finished in {elapsed_time:.2f} s ({elapsed_time/60:.2f} min / {elapsed_time/3600:.2f} hrs).")

    # Serialize Multi-Temperature Arrays
    print("\nSerializing multi-temperature datasets...")
    for T_val in target_temps:
        t_label = f"{int(T_val)}mK" if T_val == int(T_val) else f"{T_val:.1f}mK"
        cd = conductance_data_by_temp[T_val]
        Gmat_arr = cd['Conductance_matrix']
        b_r_l = cd['barrier_right_conductance_left_arr']
        b_r_r = cd['barrier_right_conductance_right_arr']
        b_r_glr = cd['barrier_right_GLR_arr']
        b_r_grl = cd['barrier_right_GRL_arr']
        b_l_l = cd['barrier_left_conductance_left_arr']
        b_l_r = cd['barrier_left_conductance_right_arr']

        delta_G_asym_R_LL = Gmat_arr[:, 0, 0] - b_r_l[:, -1]
        delta_G_asym_L_RR = Gmat_arr[:, 1, 1] - b_l_r[:, -1]

        # Explicit temperature suffix for all temperatures (e.g. _0mK, _10mK, _20mK, ...)
        hp.np_save_wrapped(Gmat_arr, f"conductance_matrix_{t_label}", dirname)
        hp.np_save_wrapped(delta_G_asym_R_LL, f"delta_G_asym_R_LL_{t_label}", dirname)
        hp.np_save_wrapped(delta_G_asym_L_RR, f"delta_G_asym_L_RR_{t_label}", dirname)
        hp.np_save_wrapped(b_r_l, f"barrier_right_conductance_left_arr_{t_label}", dirname)
        hp.np_save_wrapped(b_r_r, f"barrier_right_conductance_right_arr_{t_label}", dirname)
        hp.np_save_wrapped(b_r_glr, f"barrier_right_GLR_arr_{t_label}", dirname)
        hp.np_save_wrapped(b_r_grl, f"barrier_right_GRL_arr_{t_label}", dirname)
        hp.np_save_wrapped(b_l_l, f"barrier_left_conductance_left_arr_{t_label}", dirname)
        hp.np_save_wrapped(b_l_r, f"barrier_left_conductance_right_arr_{t_label}", dirname)

        # For base temperature (T=0), also save unsuffixed files for standard backward compatibility
        if T_val == 0.0:
            hp.np_save_wrapped(Gmat_arr, "conductance_matrix", dirname)
            hp.np_save_wrapped(delta_G_asym_R_LL, "delta_G_asym_R_LL", dirname)
            hp.np_save_wrapped(delta_G_asym_L_RR, "delta_G_asym_L_RR", dirname)
            hp.np_save_wrapped(b_r_l, "barrier_right_conductance_left_arr", dirname)
            hp.np_save_wrapped(b_r_r, "barrier_right_conductance_right_arr", dirname)
            hp.np_save_wrapped(b_r_glr, "barrier_right_GLR", dirname)
            hp.np_save_wrapped(b_r_grl, "barrier_right_GRL", dirname)
            hp.np_save_wrapped(b_l_l, "barrier_left_conductance_left_arr", dirname)
            hp.np_save_wrapped(b_l_r, "barrier_left_conductance_right_arr", dirname)

    # Save coordinate arrays and params
    if hasattr(state, 'asym_ratios'):
        hp.np_save_wrapped(state.asym_ratios, "asym_ratios", dirname)
    hp.np_save_wrapped(barrier_arr, "barrier_arr", dirname)
    hp.np_save_wrapped(params_list, "params_list", dirname)

    # Save runtime report
    runtime_path = PathConfigs.DATA / dirname / "runtime.txt"
    with open(runtime_path, "w") as f:
        f.write(f"Simulation Directory: {dirname}\n")
        f.write(f"Total Runtime Seconds: {elapsed_time:.4f}\n")
        f.write(f"Total Runtime Minutes: {elapsed_time/60:.4f}\n")
        f.write(f"Total Runtime Hours: {elapsed_time/3600:.4f}\n")
        f.write(f"Formatted Runtime: {elapsed_time:.2f} s ({elapsed_time/60:.2f} min / {elapsed_time/3600:.3f} hrs)\n")

    print(f"All files successfully written to {PathConfigs.DATA / dirname}")


if __name__ == "__main__":
    main()
