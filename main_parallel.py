import os
import time

# 1. THREAD CONTROL: Must be set BEFORE importing numpy/scipy/kwant
# Prevents oversubscription where each process tries to use all cores.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy import linalg as LA
import kwant
import tinyarray
import multiprocessing as mp
from tqdm import tqdm
import src.helpers as hp
import src.legendre_curvature as lc
from pathlib import Path
from src.config import PathConfigs
import itertools as itr
from functools import partial
import argparse
import scipy.sparse.linalg as sla
import sys
from omegaconf import OmegaConf, DictConfig
from src.parameter_handler import ConfigManager, SimulationState




# =============================================================================
# WORKER FUNCTIONS (Must be defined at module level for pickling)
# =============================================================================

def worker_simulation_step(iter_data, static_params):
    """
    Worker function for the main transport/spectral loop (Loop 1).
    iter_data: tuple (index, vz_val)
    static_params: dict containing all constant physics parameters
    """
    i, mu, vz = iter_data
    
    
    
    # Unpack static parameters
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
    energies = static_params['energies']
    barrier_arr = static_params['barrier_arr']
    num_eigenvalues = static_params['num_eigenvalues']
    eng_window_range = static_params['eng_window_range']
    solver_type = static_params.get('solver_type', 'cpu')
    
    # Barrier Sweeps ---
    points = len(barrier_arr)
    
    # Pre-allocate local arrays
    b_right_cond_left = np.zeros(points)
    b_right_cond_right = np.zeros(points)
    b_right_GLR = np.zeros(points)
    b_right_GRL = np.zeros(points)
    b_left_cond_left = np.zeros(points)
    b_left_cond_right = np.zeros(points)
    
    dIdVl, dIdVr, ldos = 0,0,0
    dIdV_LR, dIdV_RL = 0,0
    Vdisx = Vdisx * V0
    barrier_tot = barrier0 #+ mu
    gamma_sq = 0
    energy_0 = 0
    M_profile = [0]
    spectrum = None
    rho_M1 = np.zeros(Ls, dtype = complex)
    rho_M2 = np.zeros(rho_M1.shape, dtype = complex)
    site_localization = 0
    weight_localization = 1.0
    mzm_separation = 0
    topological_gap = 0.0

    
    barrier_sweep_only = static_params.get('barrier_sweep_only', False)

    tgp_barrier_arr = static_params.get('tgp_barrier_arr')
    if tgp_barrier_arr is None:
        tgp_barrier_arr = np.array([4.3548, 3.6774, 3.0000, 2.3710, 1.6935])
    else:
        tgp_barrier_arr = np.array(tgp_barrier_arr, dtype=float)

    if not barrier_sweep_only:
        # BdG Eigenvalues (using renormalized legacy Hamiltonian)
        syst_closed = hp.build_system_closed(t, mu, gamma, Delta0, vz, alpha, Ls, Vdisx)
        evals, evecs = hp.solve_ham(syst_closed, solver_type=solver_type, k=num_eigenvalues)
        
        if static_params['spectra_flag']:
            spectrum = hp.sort_spectrum(evals, evecs)
            if len(spectrum) >= 4:
                idx_closest = np.argsort(np.abs(spectrum))[:4]
                spectrum = np.sort(spectrum[idx_closest])
                
        pos_evals = np.sort(evals[evals >= 0])
        if len(pos_evals) > 0:
            energy_0 = pos_evals[0]
        else:
            energy_0 = np.nan
            
        if len(pos_evals) > 1:
            topological_gap = pos_evals[1]
        else:
            topological_gap = np.nan

        tgp_energies = hp.make_dynamic_grid()
        num_tgp = len(tgp_barrier_arr)
        tgp_stage1_dIdVl   = np.zeros((num_tgp, len(tgp_energies)))
        tgp_stage1_dIdVr   = np.zeros((num_tgp, len(tgp_energies)))
        tgp_stage1_dIdV_LR = np.zeros((num_tgp, len(tgp_energies)))
        tgp_stage1_dIdV_RL = np.zeros((num_tgp, len(tgp_energies)))
        
        tgp_barrier_pairs = static_params.get('tgp_barrier_pairs')
        for idx in range(num_tgp):
            if tgp_barrier_pairs is not None:
                b_l = tgp_barrier_pairs[idx, 0]
                b_r = tgp_barrier_pairs[idx, 1]
            else:
                b_l = tgp_barrier_arr[idx]
                b_r = tgp_barrier_arr[idx]
            syst_tgp = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz, 
                                       alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
                                       barrier_l=b_l, barrier_r=b_r, Vdisx=Vdisx)
            
            dL, dR, dLR, dRL, _ = hp.calc_dIdV(syst_tgp, tgp_energies, solver_type=solver_type)
            tgp_stage1_dIdVl[idx, :] = dL
            tgp_stage1_dIdVr[idx, :] = dR
            tgp_stage1_dIdV_LR[idx, :] = dLR
            tgp_stage1_dIdV_RL[idx, :] = dRL
    else:
        pass

    t_list = static_params.get('barrier_T_mK_list')
    if t_list and len(t_list) > 0:
        target_temps = [float(t) for t in t_list]
    elif static_params.get('barrier_conductance_mode') == 'finite_temperature' and static_params.get('barrier_T_mK', 0.0) > 0.0:
        target_temps = [float(static_params['barrier_T_mK'])]
    else:
        target_temps = [0.0]

    q_nodes = {}
    for T_val in target_temps:
        if T_val > 0.0:
            q_e, q_w = hp.get_thermal_quadrature_nodes(
                T_mK=T_val,
                n_nodes=static_params.get('barrier_quadrature_nodes', 8),
                bias_meV=0.0  # Barrier sweeps are conducted at zero bias
            )
            q_nodes[T_val] = (q_e, q_w)
        else:
            q_nodes[T_val] = (None, None)

    b_data = {}
    for T_val in target_temps:
        b_data[T_val] = {
            'Gmat': np.zeros((2, 2)),
            'b_right_cond_left': np.zeros(points),
            'b_right_cond_right': np.zeros(points),
            'b_right_GLR': np.zeros(points),
            'b_right_GRL': np.zeros(points),
            'b_left_cond_left': np.zeros(points),
            'b_left_cond_right': np.zeros(points),
        }

    sweeps_by_baseline = None
    if static_params.get('calc_barrier_sweeps', True):
        if static_params.get('multi_baseline_barriers', False):
            sweeps_by_baseline = {}
            sweep_indices = static_params.get('barrier_sweep_indices')
            tgp_barrier_pairs = static_params.get('tgp_barrier_pairs')
            if sweep_indices is None:
                active_baselines = list(enumerate(tgp_barrier_arr))
            else:
                active_baselines = [(idx, tgp_barrier_arr[idx]) for idx in sweep_indices if idx < len(tgp_barrier_arr)]

            for b_idx, u_base in active_baselines:
                ul_base = tgp_barrier_pairs[b_idx, 0] if tgp_barrier_pairs is not None else u_base
                ur_base = tgp_barrier_pairs[b_idx, 1] if tgp_barrier_pairs is not None else u_base

                base_dict = {}
                for T_val in target_temps:
                    base_dict[T_val] = {
                        'Gmat': np.zeros((2, 2)),
                        'b_right_cond_left': np.zeros(points, dtype=np.float64),
                        'b_right_cond_right': np.zeros(points, dtype=np.float64),
                        'b_right_GLR': np.zeros(points, dtype=np.float64),
                        'b_right_GRL': np.zeros(points, dtype=np.float64),
                        'b_left_cond_left': np.zeros(points, dtype=np.float64),
                        'b_left_cond_right': np.zeros(points, dtype=np.float64),
                    }

                syst_base = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma,
                                            V_z=vz, alpha=alpha, Ln=Ln, Lb=Lb,
                                            Ls=Ls, mu_leads=mu_leads,
                                            barrier_l=ul_base, barrier_r=ur_base, Vdisx=Vdisx)
                for T_val in target_temps:
                    if T_val > 0.0:
                        qe, qw = q_nodes[T_val]
                        base_dict[T_val]['Gmat'] = hp.calc_conductance_matrix_finite_T(syst_base, qe, qw, solver_type=solver_type)
                    else:
                        base_dict[T_val]['Gmat'] = hp.calc_conductance_matrix(syst_base, 0.0, solver_type=solver_type)

                for k in range(points):
                    barrier_var_tot = barrier_arr[k]

                    # Varying Right Barrier (UL = ul_base, UR = barrier_var_tot)
                    syst_UR = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma,
                                              V_z=vz, alpha=alpha, Ln=Ln, Lb=Lb,
                                              Ls=Ls, mu_leads=mu_leads, barrier_l=ul_base,
                                              barrier_r=barrier_var_tot, Vdisx=Vdisx)
                    for T_val in target_temps:
                        if T_val > 0.0:
                            qe, qw = q_nodes[T_val]
                            Gmat_UR = hp.calc_conductance_matrix_finite_T(syst_UR, qe, qw, solver_type=solver_type)
                        else:
                            Gmat_UR = hp.calc_conductance_matrix(syst_UR, eng=0.0, solver_type=solver_type)
                        base_dict[T_val]['b_right_cond_left'][k] = Gmat_UR[0, 0]
                        base_dict[T_val]['b_right_cond_right'][k] = Gmat_UR[1, 1]
                        base_dict[T_val]['b_right_GRL'][k] = Gmat_UR[1, 0]
                        base_dict[T_val]['b_right_GLR'][k] = Gmat_UR[0, 1]

                    # Varying Left Barrier (UL = barrier_var_tot, UR = ur_base)
                    syst_UL = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma,
                                              V_z=vz, alpha=alpha, Ln=Ln, Lb=Lb,
                                              Ls=Ls, mu_leads=mu_leads, barrier_l=barrier_var_tot,
                                              barrier_r=ur_base, Vdisx=Vdisx)
                    for T_val in target_temps:
                        if T_val > 0.0:
                            qe, qw = q_nodes[T_val]
                            Gmat_UL = hp.calc_conductance_matrix_finite_T(syst_UL, qe, qw, solver_type=solver_type)
                        else:
                            Gmat_UL = hp.calc_conductance_matrix(syst_UL, eng=0.0, solver_type=solver_type)
                        base_dict[T_val]['b_left_cond_left'][k] = Gmat_UL[0, 0]
                        base_dict[T_val]['b_left_cond_right'][k] = Gmat_UL[1, 1]

                sweeps_by_baseline[b_idx] = base_dict

            if len(sweeps_by_baseline) > 0:
                first_key = next(iter(sweeps_by_baseline))
                b_data = sweeps_by_baseline[first_key]
        elif static_params.get('conductance_flag', True) or barrier_sweep_only:
            syst = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz, 
                                   alpha=alpha, Ln=Ln, Lb=Lb, 
                                   Ls=Ls, mu_leads=mu_leads,
                                   barrier_l=barrier_tot, barrier_r=barrier_tot, Vdisx=Vdisx)
            for T_val in target_temps:
                if T_val > 0.0:
                    qe, qw = q_nodes[T_val]
                    b_data[T_val]['Gmat'] = hp.calc_conductance_matrix_finite_T(syst, qe, qw, solver_type=solver_type)
                else:
                    b_data[T_val]['Gmat'] = hp.calc_conductance_matrix(syst, 0.0, solver_type=solver_type)

            for k in range(points):
                barrier_var_tot = barrier_arr[k] #+ mu
                
                # Varying Right Barrier (UR)
                syst_UR = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma,
                                          V_z=vz, alpha=alpha, Ln=Ln, Lb=Lb, 
                                          Ls=Ls, mu_leads=mu_leads, barrier_l=barrier_tot,
                                          barrier_r=barrier_var_tot, Vdisx=Vdisx)
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
                
                # Varying Left Barrier (UL)
                syst_UL = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma,
                                          V_z=vz, alpha=alpha, Ln=Ln, Lb=Lb, 
                                          Ls=Ls, mu_leads=mu_leads, barrier_l=barrier_var_tot,
                                          barrier_r=barrier_tot, Vdisx=Vdisx)
                for T_val in target_temps:
                    if T_val > 0.0:
                        qe, qw = q_nodes[T_val]
                        Gmat_UL = hp.calc_conductance_matrix_finite_T(syst_UL, qe, qw, solver_type=solver_type)
                    else:
                        Gmat_UL = hp.calc_conductance_matrix(syst_UL, eng=0.0, solver_type=solver_type)
                    b_data[T_val]['b_left_cond_left'][k] = Gmat_UL[0, 0]
                    b_data[T_val]['b_left_cond_right'][k] = Gmat_UL[1, 1]

    # 25-Cutter Harmonics (1w, 2w, 3w) via Double-Exponential Quadrature
    harmonics_by_temp = None
    if static_params.get('calc_25_cutter_harmonics', False):
        num_baselines = len(tgp_barrier_arr)
        de_quad = static_params.get('de_quad', {})

        harmonics_by_temp = {}
        for T_val in target_temps:
            harmonics_by_temp[T_val] = {
                'L_1w': np.zeros((num_baselines, num_baselines), dtype=np.float64),
                'R_1w': np.zeros((num_baselines, num_baselines), dtype=np.float64),
                'RL_2w': np.zeros((num_baselines, num_baselines), dtype=np.float64),
                'LR_2w': np.zeros((num_baselines, num_baselines), dtype=np.float64),
                'L_3w': np.zeros((num_baselines, num_baselines), dtype=np.float64),
                'R_3w': np.zeros((num_baselines, num_baselines), dtype=np.float64),
            }

        tgp_barrier_pairs = static_params.get('tgp_barrier_pairs')
        for i_l in range(num_baselines):
            ul = tgp_barrier_pairs[i_l, 0] if tgp_barrier_pairs is not None else tgp_barrier_arr[i_l]
            for j_r in range(num_baselines):
                ur = tgp_barrier_pairs[j_r, 1] if tgp_barrier_pairs is not None else tgp_barrier_arr[j_r]
                syst_pair = hp.build_system(
                    t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz,
                    alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
                    barrier_l=ul, barrier_r=ur, Vdisx=Vdisx
                )

                # A. Zero Temperature (T = 0)
                if 0.0 in target_temps:
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
                    q = de_quad.get(T_val)
                    if q is None:
                        continue
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

    T_default = target_temps[0]

    if barrier_sweep_only:
        return {
            'i': i,
            'Gmat': b_data[T_default]['Gmat'],
            'b_right_cond_left': b_data[T_default]['b_right_cond_left'],
            'b_right_cond_right': b_data[T_default]['b_right_cond_right'],
            'b_right_GLR': b_data[T_default]['b_right_GLR'],
            'b_right_GRL': b_data[T_default]['b_right_GRL'],
            'b_left_cond_left': b_data[T_default]['b_left_cond_left'],
            'b_left_cond_right': b_data[T_default]['b_left_cond_right'],
            'b_data': b_data,
            'sweeps_by_baseline': sweeps_by_baseline,
            'harmonics_by_temp': harmonics_by_temp,
        }

    # --- Conditional Heavy Calculations ---
    if energy_0 <= 0.1:
        if static_params['localization_flag'] and evecs is not None:
            rho_M1, rho_M2, _ = hp.get_psiM_density(evals, evecs)
            weight_localization = hp.calc_weight_localization(rho_M1, rho_M2, weight_threshold=static_params['weight_threshold'])
            mzm_separation = hp.calc_MZM_separation(rho_M1, rho_M2)
            
    
    results = {
        'i': i,
        'tgp_stage1_dIdVl': tgp_stage1_dIdVl,
        'tgp_stage1_dIdVr': tgp_stage1_dIdVr,
        'tgp_stage1_dIdV_LR': tgp_stage1_dIdV_LR,
        'tgp_stage1_dIdV_RL': tgp_stage1_dIdV_RL,
        'Gmat': b_data[T_default]['Gmat'],
        'gamma_sq': gamma_sq,
        'energy_0': energy_0,
        'topological_gap': topological_gap,
        'M_profile': M_profile,
        'b_right_cond_left': b_data[T_default]['b_right_cond_left'],
        'b_right_cond_right': b_data[T_default]['b_right_cond_right'],
        'b_right_GLR': b_data[T_default]['b_right_GLR'],
        'b_right_GRL': b_data[T_default]['b_right_GRL'],
        'b_left_cond_left': b_data[T_default]['b_left_cond_left'],
        'b_left_cond_right': b_data[T_default]['b_left_cond_right'],
        'spectrum': spectrum,
        'weight_localization': weight_localization,
        'mzm_separation': mzm_separation,
        'tgp_barrier_arr': tgp_barrier_arr,
        'b_data': b_data,
        'sweeps_by_baseline': sweeps_by_baseline,
        'harmonics_by_temp': harmonics_by_temp,
    }
    return results

#TODO: refactor this name to "worker_pfaffian_step"
def worker_pdi_step(iter_data, static_params):
    """
    Worker function for the PDI calculation loop (Loop 2).
    """
    i, mu_pm, vz = iter_data

    # Unpack necessary static params
    ts = static_params['t']
    alphas = static_params['alpha']
    gamma = static_params['gamma']
    Ls = static_params['Ls']
    Vdisx = static_params['Vdisx']
    V0 = static_params['V0']
    qn = static_params['qn']
    Delta0 = static_params['Delta0']
    pfaffian_delta_N = static_params["pfaffian_delta_N"]

    # Apply Renormalization Factor Z to align with Kwant 
    #Z = Delta0 / (Delta0 + gamma)
    Z = 1 #TODO: double check this, should be find but need to check! If so need to remove it to avoid further confusion
    ts *= Z
    alphas *= Z
    gamma *= Z
    V0 *= Z
    mu_pm *= Z
    vz *= Z
    
    pdi_value = 0

    
    result = [mu_pm, vz, pdi_value]
    
    if static_params.get('calc_pfaffian', False):
        # Delta0 is multiplied by the same renormalization factor Z as other parameters
        pfaff_val = hp.cal_pfaffian_invariant(
            ts=ts, 
            alphas=alphas, 
            gamma=gamma, 
            delta0=Delta0 * Z, 
            Nx=Ls, 
            Vdisx=Vdisx, 
            V0=V0, 
            gm=vz, 
            mu=mu_pm,
            delta_N=pfaffian_delta_N
        )
        result.append(pfaff_val)
        
    return result



if __name__ == "__main__":
    t_start = time.perf_counter()
    
    parser = argparse.ArgumentParser(description="Nanowire simulation driver.")
    parser.add_argument("config_file", nargs="?", default=None, help="Path to config YAML file.")
    parser.add_argument("--config_path", "--config", dest="config_path", type=str, default=None, help="Path to config YAML file.")
    parser.add_argument("--workers", dest="workers", type=int, default=None, help="Number of workers.")
    parser.add_argument("--max_points", dest="max_points", type=int, default=None, help="Max points to simulate.")
    cli_args, remaining = parser.parse_known_args()

    config_path = cli_args.config_file or cli_args.config_path or "Inputs/Parameters/fixing_code.yaml"
    config = ConfigManager.get_config(config_path=config_path, cli_args=remaining or None)
    if cli_args.workers is not None:
        config.workers = cli_args.workers
    if cli_args.max_points is not None:
        config.max_points = cli_args.max_points

    state = SimulationState(config)
    
    if hasattr(config, 'outputs') and hasattr(config.outputs, 'dirname'):
        dirname = config.outputs.dirname
    else:
        dirname = getattr(config, 'dirname', 'default_run')

    if hasattr(config, 'simulation') and hasattr(config.simulation, 'geometry'):
        geom = config.simulation.geometry
        disorder_cfg = config.simulation.disorder
        res_cfg = config.simulation.resolution
        calc_cfg = getattr(config, 'calculation_includes', None)
        disorder_file = getattr(disorder_cfg, 'disorder_dirname', getattr(disorder_cfg, 'fname', 'Tdis.npz'))
        Lb = geom.Lb
        Lb_pdi = getattr(geom, 'Lb_pdi', geom.Lb)
        Ls = geom.Ls
        Ln = geom.Ln
        t_list = getattr(res_cfg, 'temperatures', getattr(res_cfg, 'barrier_T_mK_list', None))
        target_temps = [float(t) for t in t_list] if t_list else [0.0]
        quadrature_nodes_25 = getattr(res_cfg, 'quadrature_nodes_25', 51)
        barrier_sweep_only = getattr(calc_cfg, 'barrier_sweep_only', False) if calc_cfg else False
        pdi_flag = getattr(calc_cfg, 'pdi_flag', False) if calc_cfg else getattr(config, 'pdi_flag', False)
        calc_25_harmonics = getattr(calc_cfg, 'calc_25_cutter_harmonics', False) if calc_cfg else getattr(config, 'calc_25_cutter_harmonics', False)
    else:
        disorder_file = getattr(config, 'fname', 'Tdis.npz')
        Lb = config.Lb
        Lb_pdi = getattr(config, 'Lb_pdi', config.Lb)
        Ls = config.Ls
        Ln = config.Ln
        t_list = getattr(config, 'barrier_T_mK_list', None)
        target_temps = [float(t) for t in t_list] if t_list else [0.0]
        quadrature_nodes_25 = getattr(config, 'quadrature_nodes_25', 51)
        barrier_sweep_only = getattr(config, 'barrier_sweep_only', False)
        pdi_flag = getattr(config, 'pdi_flag', False)
        calc_25_harmonics = getattr(config, 'calc_25_cutter_harmonics', False)
    
    print(f"--- Starting Simulation ---")
    print(f"Output Directory: {dirname}")
    print(f"Hamiltonian Mode: legacy (renormalized)")
    print(f"Hopping: {state.t:.4f}, Spin-Orbit: {state.alpha:.4f}, Delta: {state.Delta:.4f}")
    print(f"Disorder File: {disorder_file}")
    print(f"Barrier Length (Lb): {Lb}")
    print(f"PDI Barrier Length (Lb_pdi): {Lb_pdi}")
    print(f"Acceleration Mode: {config.acceleration_type}")
    print(f"Barrier Calibration: {hasattr(state, 'symmetric_barrier_configs')}")
    print(f"TGP Barrier Array: {state.tgp_barrier_arr}")
    print(f"--------------------------------\n")
    
    # Artifact Log
    #TODO: make sure this logs the exact sim state as well
    ConfigManager.log_artifact(config, PathConfigs.DATA / dirname)
    
    # Initialize Disorder
    print(f"Run Files Path Exists: {os.path.exists(PathConfigs.RUN_FILES)}")

    # Parameter Preparation
    static_params = state.get_static_params(state.Vdisx, config)
    
    params_list = state.params_list
    max_pts = config.get('max_points', None) if hasattr(config, 'get') else getattr(config, 'max_points', None)
    if max_pts is not None:
        params_list = params_list[:max_pts]
    mu_var = state.mu_var
    Vz_var = state.Vz_var
    barrier_arr = state.barrier_arr
    energies = state.energies
    
    print("conductance_flag:", f"{static_params['conductance_flag']}")
    print("spectra_flag:", f"{static_params['spectra_flag']}")

    # Pre-allocate main arrays
    lenw = Ls + 2*(Lb + Ln)
    num_orbitals = lenw * 4
    print(f"LEN: {num_orbitals}")
    #ldos_arr = np.zeros(shape = (len(params_list), len(energies), num_orbitals)) 
    


    dIdVs_left_arr = np.zeros(shape = (len(params_list), len(energies)))
    lengrid = len(hp.make_dynamic_grid())
    num_tgp = len(state.tgp_barrier_arr) if hasattr(state, 'tgp_barrier_arr') else 5
    tgp_stage1_dIdVl   = np.zeros(shape = (len(params_list), num_tgp, lengrid))
    tgp_stage1_dIdVr   = np.zeros(shape = (len(params_list), num_tgp, lengrid))
    tgp_stage1_dIdV_LR = np.zeros(shape = (len(params_list), num_tgp, lengrid))
    tgp_stage1_dIdV_RL = np.zeros(shape = (len(params_list), num_tgp, lengrid))
    dIdVs_right_arr = np.zeros(shape = (len(params_list), len(energies)))
    dIdVs_LR_arr = np.zeros(shape = (len(params_list), len(energies)))
    dIdVs_RL_arr = np.zeros(shape = (len(params_list), len(energies)))

    num_barrier_points = len(barrier_arr)
    conductance_data_by_temp = {}
    for T_val in target_temps:
        conductance_data_by_temp[T_val] = {
            'Conductance_matrix': np.zeros((len(params_list), 2, 2)),
            'barrier_right_conductance_left_arr': np.zeros((len(params_list), num_barrier_points)),
            'barrier_right_conductance_right_arr': np.zeros((len(params_list), num_barrier_points)),
            'barrier_right_GLR_arr': np.zeros((len(params_list), num_barrier_points)),
            'barrier_right_GRL_arr': np.zeros((len(params_list), num_barrier_points)),
            'barrier_left_conductance_left_arr': np.zeros((len(params_list), num_barrier_points)),
            'barrier_left_conductance_right_arr': np.zeros((len(params_list), num_barrier_points)),
        }

    spectrum_arr = np.zeros(shape=(len(params_list), 4))

    weight_localization_arr = np.zeros(len(params_list))
    mzm_separation_arr = np.zeros(len(params_list))
    gamma_sq_arr = np.zeros_like(params_list, dtype=complex)
    mp_eng_arr = np.zeros_like(params_list)
    lenw = Ls #+ 2*(Lb + Ln)
    mp_arr = np.zeros(shape= (len(params_list), lenw))
    topological_gap_arr = np.zeros(shape=(len(params_list)))
    
    results = []
    de_quad = {}
    if calc_25_harmonics:
        print(f"--- Precomputing Double-Exponential Quadrature ({quadrature_nodes_25} nodes) ---")
        for T_val in target_temps:
            if T_val > 0.0:
                de_quad[T_val] = lc.get_double_exponential_quadrature(
                    T_mK=T_val, n_nodes=quadrature_nodes_25
                )
                print(f"  T = {T_val:4.1f} mK: DE ({de_quad[T_val]['n_nodes']} nodes)")
            else:
                de_quad[0.0] = None
    static_params['de_quad'] = de_quad

    if config.acceleration_type == 'gpu':
        print("Using GPU acceleration (Serial Sweep).")
        results = [worker_simulation_step(pms, static_params) for pms in tqdm(params_list, desc="mu/Vz Sweep")]
        
    elif config.acceleration_type == 'parallel':
        num_workers = config.get('workers', None) if (hasattr(config, 'get') and config.get('workers', None) is not None) else max(1, mp.cpu_count() - 1)
        print(f"Starting Parallel Execution with {num_workers} workers.")
        
        with mp.Pool(processes=num_workers) as pool:
            func_sim = partial(worker_simulation_step, static_params=static_params)
            results = list(tqdm(pool.imap(func_sim, params_list, chunksize=1), total=len(params_list), desc="mu/Vz Sweep"))
    else: # None
        print("Using Serial Execution (CPU).")
        for pms in tqdm(params_list, desc="mu/Vz Sweep"):
            results.append(worker_simulation_step(pms, static_params))

    if barrier_sweep_only:
        for res in results:
            idx = res['i']
            if 'b_data' in res:
                for T_val in target_temps:
                    td = res['b_data'][T_val]
                    cd = conductance_data_by_temp[T_val]
                    cd['Conductance_matrix'][idx, :, :] = td['Gmat']
                    cd['barrier_right_conductance_left_arr'][idx, :] = td['b_right_cond_left']
                    cd['barrier_right_conductance_right_arr'][idx, :] = td['b_right_cond_right']
                    cd['barrier_right_GLR_arr'][idx, :] = td['b_right_GLR']
                    cd['barrier_right_GRL_arr'][idx, :] = td['b_right_GRL']
                    cd['barrier_left_conductance_left_arr'][idx, :] = td['b_left_cond_left']
                    cd['barrier_left_conductance_right_arr'][idx, :] = td['b_left_cond_right']
            else:
                T_val = target_temps[0]
                cd = conductance_data_by_temp[T_val]
                cd['Conductance_matrix'][idx, :, :] = res['Gmat']
                cd['barrier_right_conductance_left_arr'][idx, :] = res['b_right_cond_left']
                cd['barrier_right_conductance_right_arr'][idx, :] = res['b_right_cond_right']
                cd['barrier_right_GLR_arr'][idx, :] = res['b_right_GLR']
                cd['barrier_right_GRL_arr'][idx, :] = res['b_right_GRL']
                cd['barrier_left_conductance_left_arr'][idx, :] = res['b_left_cond_left']
                cd['barrier_left_conductance_right_arr'][idx, :] = res['b_left_cond_right']
    else:
        for res in results:
            tgp_barrier_arr = res['tgp_barrier_arr']
            idx = res['i']
            if 'b_data' in res:
                for T_val in target_temps:
                    td = res['b_data'][T_val]
                    cd = conductance_data_by_temp[T_val]
                    cd['Conductance_matrix'][idx, :, :] = td['Gmat']
                    cd['barrier_right_conductance_left_arr'][idx, :] = td['b_right_cond_left']
                    cd['barrier_right_conductance_right_arr'][idx, :] = td['b_right_cond_right']
                    cd['barrier_right_GLR_arr'][idx, :] = td['b_right_GLR']
                    cd['barrier_right_GRL_arr'][idx, :] = td['b_right_GRL']
                    cd['barrier_left_conductance_left_arr'][idx, :] = td['b_left_cond_left']
                    cd['barrier_left_conductance_right_arr'][idx, :] = td['b_left_cond_right']
            else:
                T_val = target_temps[0]
                cd = conductance_data_by_temp[T_val]
                cd['Conductance_matrix'][idx, :, :] = res['Gmat']
                cd['barrier_right_conductance_left_arr'][idx, :] = res['b_right_cond_left']
                cd['barrier_right_conductance_right_arr'][idx, :] = res['b_right_cond_right']
                cd['barrier_right_GLR_arr'][idx, :] = res['b_right_GLR']
                cd['barrier_right_GRL_arr'][idx, :] = res['b_right_GRL']
                cd['barrier_left_conductance_left_arr'][idx, :] = res['b_left_cond_left']
                cd['barrier_left_conductance_right_arr'][idx, :] = res['b_left_cond_right']
            
            tgp_stage1_dIdVl[idx, :, :] = res['tgp_stage1_dIdVl']
            tgp_stage1_dIdVr[idx, :, :] = res['tgp_stage1_dIdVr']
            tgp_stage1_dIdV_LR[idx, :, :] = res['tgp_stage1_dIdV_LR']
            tgp_stage1_dIdV_RL[idx, :, :] = res['tgp_stage1_dIdV_RL']
            gamma_sq_arr[idx] = res['gamma_sq']
            mp_eng_arr[idx] = res['energy_0']
            topological_gap_arr[idx] = res['topological_gap']
            mp_arr[idx, :] = res['M_profile']
            
            if res['spectrum'] is not None:
                spectrum_arr[idx, :] = res['spectrum']
            
            weight_localization_arr[idx] = res['weight_localization']
            mzm_separation_arr[idx] = res['mzm_separation']
        
        
    pdi_data = np.array([])

    if pdi_flag and not barrier_sweep_only:
        print("\nStarting PDI Calculation Loop.")
        pdi_results = []
        if config.acceleration_type == 'gpu':
            print("Using GPU acceleration (Serial Sweep).")
            pdi_results = [worker_pdi_step(pms, static_params) for pms in tqdm(params_list, desc="PDI Sweep")]

        elif config.acceleration_type == 'parallel':
            num_workers = config.get('workers', None) if (hasattr(config, 'get') and config.get('workers', None) is not None) else max(1, mp.cpu_count() - 1)
            print(f"Starting Parallel Execution with {num_workers} workers.")

            with mp.Pool(processes=num_workers) as pool:
                func_pdi = partial(worker_pdi_step, static_params=static_params)
                pdi_results = list(tqdm(pool.imap(func_pdi, params_list, chunksize=1), total=len(params_list), desc="PDI Sweep"))
        else: # None
            print("Using Serial Execution (CPU).")
            for pms in tqdm(params_list, desc="PDI Sweep"):
                pdi_results.append(worker_pdi_step(pms, static_params))

        pdi_data = np.array(pdi_results)


    
    # -------------------------------------------------------------------------
    # Saving Results
    # -------------------------------------------------------------------------
    print(f"\nSaving data to: {dirname}")

    for T_val in target_temps:
        temp_suffix = ""
        if T_val > 0.0:
            t_str = f"{int(T_val)}mK" if T_val == int(T_val) else f"{T_val:.1f}mK"
            temp_suffix = f"_{t_str}"
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

        hp.np_save_wrapped(Gmat_arr, f"conductance_matrix{temp_suffix}", dirname)
        hp.np_save_wrapped(delta_G_asym_R_LL, f"delta_G_asym_R_LL{temp_suffix}", dirname)
        hp.np_save_wrapped(delta_G_asym_L_RR, f"delta_G_asym_L_RR{temp_suffix}", dirname)
        hp.np_save_wrapped(b_r_l, f"barrier_right_conductance_left_arr{temp_suffix}", dirname)
        hp.np_save_wrapped(b_r_r, f"barrier_right_conductance_right_arr{temp_suffix}", dirname)
        hp.np_save_wrapped(b_r_glr, f"barrier_right_GLR_arr{temp_suffix}", dirname)
        hp.np_save_wrapped(b_r_grl, f"barrier_right_GRL_arr{temp_suffix}", dirname)
        hp.np_save_wrapped(b_l_l, f"barrier_left_conductance_left_arr{temp_suffix}", dirname)
        hp.np_save_wrapped(b_l_r, f"barrier_left_conductance_right_arr{temp_suffix}", dirname)

    if hasattr(state, 'asym_ratios'):
        hp.np_save_wrapped(state.asym_ratios, "asym_ratios", dirname)
    hp.np_save_wrapped(barrier_arr, "barrier_arr", dirname)
    hp.np_save_wrapped(params_list, "params_list", dirname)

    if not barrier_sweep_only:
        hp.np_save_wrapped(state.Vdisx, "Vdisx", dirname)
        hp.np_save_wrapped(pdi_data, "pdi_data", dirname)
        hp.np_save_wrapped(energies, "energies", dirname)
        hp.np_save_wrapped(tgp_stage1_dIdVl, "tgp_stage1_dIdVl", dirname)
        hp.np_save_wrapped(tgp_stage1_dIdVr, "tgp_stage1_dIdVr", dirname)
        hp.np_save_wrapped(tgp_stage1_dIdV_LR, "tgp_stage1_dIdV_LR", dirname)
        hp.np_save_wrapped(tgp_stage1_dIdV_RL, "tgp_stage1_dIdV_RL", dirname)
        hp.np_save_wrapped(topological_gap_arr, "topological_gap", dirname)
        hp.np_save_wrapped(mp_arr, "mp_arr", dirname)
        hp.np_save_wrapped(mp_eng_arr, "energy_0", dirname)
        hp.np_save_wrapped(spectrum_arr, "spectrum_arr", dirname)
        hp.np_save_wrapped(weight_localization_arr, "weight_localization_arr", dirname)
        hp.np_save_wrapped(mzm_separation_arr, "mzm_separation_arr", dirname)
        
        # Save exact cutter barrier array (calibrated or hardcoded)
        barriers_to_save = state.tgp_barrier_arr if hasattr(state, 'tgp_barrier_arr') else ('tgp_barrier_arr' in locals() and tgp_barrier_arr)
        if barriers_to_save is not None:
            hp.np_save_wrapped(barriers_to_save, 'tgp_barrier_arr', dirname)
            if hasattr(state, 'tgp_barrier_pairs') and state.tgp_barrier_pairs is not None:
                hp.np_save_wrapped(state.tgp_barrier_pairs, 'tgp_barrier_pairs', dirname)
            txt_path = PathConfigs.DATA / dirname / "tgp_barriers.txt"
            with open(txt_path, "w") as f:
                f.write(f"Simulation Directory: {dirname}\n")
                f.write(f"Hamiltonian Mode: legacy (renormalized)\n")
                calib_flag = (hasattr(state, 'symmetric_barrier_configs') or hasattr(state, 'tgp_barrier_pairs'))
                f.write(f"Barrier Calibration: {calib_flag}\n")
                targets = getattr(state, 'target_gns', None) or (config.get('target_gns', None) if hasattr(config, 'get') else getattr(config, 'target_gns', None))
                if targets is not None:
                    f.write(f"Target Conductances G_N (e^2/h): {targets}\n")
                f.write(f"TGP Barrier Array (meV): {barriers_to_save.tolist()}\n")
                if hasattr(state, 'tgp_barrier_pairs') and state.tgp_barrier_pairs is not None:
                    f.write(f"TGP Barrier Pairs (UL, UR meV):\n")
                    for b_idx, pair in enumerate(state.tgp_barrier_pairs):
                        tg_label = targets[b_idx] if (targets is not None and b_idx < len(targets)) else b_idx
                        f.write(f"  Target {tg_label}: UL={pair[0]:.6f} meV, UR={pair[1]:.6f} meV\n")
                else:
                    for b_idx, b_val in enumerate(barriers_to_save):
                        f.write(f"  Barrier {b_idx + 1}: {b_val:.6f} meV\n")

    # Multi-baseline barrier sweeps serialization
    multi_baseline = config.get('multi_baseline_barriers', False) if hasattr(config, 'get') else getattr(config, 'multi_baseline_barriers', False)
    calc_sweeps = config.get('calc_barrier_sweeps', True) if hasattr(config, 'get') else getattr(config, 'calc_barrier_sweeps', True)
    if multi_baseline and calc_sweeps:
        obs_keys = [
            'b_right_cond_left', 'b_right_cond_right',
            'b_right_GLR', 'b_right_GRL',
            'b_left_cond_left', 'b_left_cond_right'
        ]
        u_cutters = state.tgp_barrier_arr
        targets = getattr(state, 'target_gns', None) or (config.get('target_gns', None) if hasattr(config, 'get') else getattr(config, 'target_gns', None))
        targets = targets or [0.35, 0.49, 0.62, 0.76, 0.90]
        num_baselines = len(u_cutters)
        t_keys = [f"{int(T)}mK" if T == int(T) else f"{T:.1f}mK" for T in target_temps]

        sweep_indices = config.get('barrier_sweep_indices', None) if hasattr(config, 'get') else getattr(config, 'barrier_sweep_indices', None)
        if sweep_indices is None:
            active_indices = list(range(num_baselines))
        else:
            active_indices = [idx for idx in sweep_indices if idx < num_baselines]

        sweep_tables = {}
        for b_idx in active_indices:
            b_dict = {}
            for ok in obs_keys:
                b_dict[ok] = {tk: np.zeros((len(params_list), num_barrier_points), dtype=np.float64) for tk in t_keys}
            sweep_tables[b_idx] = b_dict

        for res in results:
            idx = res['i']
            if res.get('sweeps_by_baseline') is not None:
                s_dict = res['sweeps_by_baseline']
                for b_idx in active_indices:
                    base_data = s_dict.get(b_idx) if isinstance(s_dict, dict) else (s_dict[b_idx] if b_idx < len(s_dict) else None)
                    if base_data is not None:
                        for T_val, tk in zip(target_temps, t_keys):
                            p_res = base_data[T_val]
                            for ok in obs_keys:
                                sweep_tables[b_idx][ok][tk][idx, :] = p_res[ok]

        tag_map = {
            0.35: "baseline_0_35G0",
            0.49: "baseline_0_49G0",
            0.62: "baseline_0_62G0",
            0.76: "baseline_0_76G0",
            0.90: "baseline_0_90G0"
        }
        barrier_dir = PathConfigs.DATA / dirname / "Barrier_Sweeps"
        barrier_dir.mkdir(parents=True, exist_ok=True)

        for b_idx in active_indices:
            target_gn = targets[b_idx] if b_idx < len(targets) else b_idx
            folder_name = tag_map.get(round(target_gn, 2), f"baseline_{str(target_gn).replace('.', '_')}G0")
            base_dir = barrier_dir / folder_name
            base_dir.mkdir(parents=True, exist_ok=True)
            np.save(base_dir / "barrier_arr.npy", state.barrier_arr)

            for ok in obs_keys:
                npz_dict = {tk: sweep_tables[b_idx][ok][tk] for tk in t_keys}
                np.savez_compressed(base_dir / f"{ok}.npz", **npz_dict)

    # 25-Cutter harmonics serialization
    if calc_25_harmonics:
        harm_keys = ['L_1w', 'R_1w', 'RL_2w', 'LR_2w', 'L_3w', 'R_3w']
        u_cutters = state.tgp_barrier_arr
        num_baselines = len(u_cutters)
        t_keys = [f"{int(T)}mK" if T == int(T) else f"{T:.1f}mK" for T in target_temps]

        harm_tables = {
            tk: {hk: np.zeros((len(params_list), num_baselines, num_baselines), dtype=np.float64) for hk in harm_keys}
            for tk in t_keys
        }

        for res in results:
            idx = res['i']
            if res.get('harmonics_by_temp') is not None:
                for T_val, tk in zip(target_temps, t_keys):
                    h_res = res['harmonics_by_temp'][T_val]
                    for hk in harm_keys:
                        harm_tables[tk][hk][idx, :, :] = h_res[hk]

        harm_dir = PathConfigs.DATA / dirname / "Cutter_Harmonics_25"
        harm_dir.mkdir(parents=True, exist_ok=True)

        for T_val, tk in zip(target_temps, t_keys):
            np.savez_compressed(harm_dir / f"harmonics_25_{tk}.npz", **harm_tables[tk])

    
    t_end = time.perf_counter()
    total_runtime_sec = t_end - t_start
    total_runtime_min = total_runtime_sec / 60.0
    total_runtime_hrs = total_runtime_sec / 3600.0

    print("\n" + "=" * 65)
    print(f"Simulation Complete for: {dirname}")
    print(f"Total Runtime: {total_runtime_sec:.2f} s ({total_runtime_min:.2f} min / {total_runtime_hrs:.3f} hrs)")
    print("=" * 65 + "\n")

    if isinstance(config, DictConfig):
        cfg_dict = OmegaConf.to_container(config, resolve=True)
    elif hasattr(config, 'model_dump'):
        cfg_dict = config.model_dump()
    else:
        cfg_dict = dict(config)

    all_params = {
        **cfg_dict,  # unpacks all input parameters
        **static_params,        # unpacks 't', 'mu_n', 'Delta', 'alpha', etc. (overrides None with calculated values)
        'total_runtime_seconds': total_runtime_sec,
        'total_runtime_minutes': total_runtime_min,
        'total_runtime_hours': total_runtime_hrs
    }
    hp.np_savez_wrapped("all_params", dirname, **all_params)
    
    # Write human-readable runtime log
    out_dir = PathConfigs.DATA / dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runtime.txt", "w") as rf:
        rf.write(f"Simulation Directory: {dirname}\n")
        rf.write(f"Total Runtime Seconds: {total_runtime_sec:.4f}\n")
        rf.write(f"Total Runtime Minutes: {total_runtime_min:.4f}\n")
        rf.write(f"Total Runtime Hours: {total_runtime_hrs:.4f}\n")
        rf.write(f"Formatted Runtime: {total_runtime_sec:.2f} s ({total_runtime_min:.2f} min / {total_runtime_hrs:.3f} hrs)\n")

    print("Done.")