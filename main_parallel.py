import os

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
import helpers as hp
from pathlib import Path
from config import PathConfigs
import itertools as itr
from functools import partial
import argparse
import scipy.sparse.linalg as sla
from parameter_handler import ConfigManager, SimulationState




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
    
    # --- 2. Barrier Sweeps (Nested Loop logic) ---
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
    overlap_integral = 0.0
    mzm_separation = 0
    topological_gap = 0.0

    # --- 1. Build Symmetric System & Calculate Spectral Properties ---

    if static_params['spectra_flag'] or static_params['localization_flag']:
        syst_closed = hp.build_system_closed(t, mu, gamma, Delta0, vz, alpha, Ls, Vdisx)
        evals, evecs = hp.solve_ham(syst_closed, solver_type=solver_type, k=num_eigenvalues)


        if static_params['localization_flag']:
            rho_M1, rho_M2, _ = hp.get_psiM_density(evals, evecs)
            site_localization = hp.calc_MZM_localization(rho_M1, rho_M2)
            weight_localization = hp.calc_weight_localization(rho_M1, rho_M2, weight_threshold=static_params['weight_threshold'])
            overlap_integral = hp.calc_overlap(rho_M1, rho_M2)
            mzm_separation = hp.calc_MZM_separation(rho_M1, rho_M2)
        if static_params['spectra_flag']:
            spectrum = hp.sort_spectrum(evals, evecs)
            if len(spectrum) >= 4:
                idx_closest = np.argsort(np.abs(spectrum))[:4]
                spectrum = np.sort(spectrum[idx_closest])
            pos_evals = np.sort(evals[evals >= 0])
            if len(pos_evals) > 1:
                topological_gap = pos_evals[1]
            else:
                topological_gap = np.nan
            #gamma_sq = hp.calculate_gamma_squared(evals, evecs)
            #M_profile, energy_0 = hp.calculate_local_mp(evals, evecs)
    
    
    eng_window = np.linspace(-0.15, 0.15, eng_window_range)
    csL = np.zeros_like(eng_window)
    csR = np.zeros_like(eng_window)
    
    pk_l = 0
    pk_r = 0 
    Gmat = 0
    rG_corr = 0
    lG_corr = 0
    fine_dIdVl = np.zeros(7)
    
    if static_params['conductance_flag']:
        
        syst = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma = gamma, V_z=vz, 
                           alpha=alpha, Ln=Ln, Lb=Lb, 
                           Ls=Ls, mu_leads=mu_leads,
                           barrier_l=barrier_tot, barrier_r=barrier_tot, Vdisx=Vdisx)
    
        dIdVl, dIdVr, dIdV_LR, dIdV_RL, ldos = hp.calc_dIdV(syst, energies, solver_type=solver_type)
        fine_energies = np.linspace(-0.06, 0.06, 7)
        fine_dIdVl, _, _, _, _ = hp.calc_dIdV(syst, fine_energies, solver_type=solver_type)
        Gmat = hp.calc_conductance_matrix(syst, 0.0, solver_type=solver_type)
        for k, eng in enumerate(eng_window):
            cL, cR = hp.calc_conductance(syst, energy=eng, solver_type=solver_type)
            csL[k] = cL
            csR[k] = cR
    
        pk_l_result = hp.detect_peaks_v2(csL, eng_window)
        pk_l = np.asarray([
            1.0 if pk_l_result['has_peaks'] else 0.0,
            1.0 if pk_l_result['has_both'] else 0.0,
            pk_l_result['pos_energy'] if not np.isnan(pk_l_result['pos_energy']) else 10.0,
            pk_l_result['pos_height'] if not np.isnan(pk_l_result['pos_height']) else 0.0,
            pk_l_result['neg_energy'] if not np.isnan(pk_l_result['neg_energy']) else -10.0,
            pk_l_result['neg_height'] if not np.isnan(pk_l_result['neg_height']) else 0.0,
        ])
        
        pk_r_result = hp.detect_peaks_v2(csR, eng_window)
        pk_r = np.asarray([
            1.0 if pk_r_result['has_peaks'] else 0.0,
            1.0 if pk_r_result['has_both'] else 0.0,
            pk_r_result['pos_energy'] if not np.isnan(pk_r_result['pos_energy']) else 10.0,
            pk_r_result['pos_height'] if not np.isnan(pk_r_result['pos_height']) else 0.0,
            pk_r_result['neg_energy'] if not np.isnan(pk_r_result['neg_energy']) else -10.0,
            pk_r_result['neg_height'] if not np.isnan(pk_r_result['neg_height']) else 0.0,
        ])
    

        # Note: this is run serially inside the worker because the overhead 
        # of spawning sub-processes here would be too high.
        for k in range(points):
            barrier_var_tot = barrier_arr[k] #+ mu
            
            # Varying Right Barrier (UR)
            syst_UR = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma = gamma,
                                    V_z=vz, alpha=alpha, Ln=Ln, Lb=Lb, 
                                    Ls=Ls, mu_leads=mu_leads, barrier_l=barrier_tot,
                                    barrier_r=barrier_var_tot, Vdisx=Vdisx)
            
            Gmat_UR = hp.calc_conductance_matrix(syst_UR, eng=0.0, solver_type=solver_type)
            b_right_cond_left[k] = Gmat_UR[0, 0]
            b_right_cond_right[k] = Gmat_UR[1, 1]
            b_right_GRL[k] = Gmat_UR[1, 0]
            b_right_GLR[k] = Gmat_UR[0, 1]
            
            # Varying Left Barrier (UL)
            syst_UL = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma = gamma,
                                    V_z=vz, alpha=alpha, Ln=Ln, Lb=Lb, 
                                    Ls=Ls, mu_leads=mu_leads, barrier_l=barrier_var_tot,
                                    barrier_r=barrier_tot, Vdisx=Vdisx)
            
            Gmat_UL = hp.calc_conductance_matrix(syst_UL, eng=0.0, solver_type=solver_type)
            b_left_cond_left[k] = Gmat_UL[0, 0]
            b_left_cond_right[k] = Gmat_UL[1, 1]
            
            
        idx_sym = np.argmin(np.abs(barrier_arr - barrier_tot))
        b_right_cond_left = b_right_cond_left / (b_right_cond_left[idx_sym] if b_right_cond_left[idx_sym] != 0 else 1.0)
        b_right_cond_right = b_right_cond_right / (b_right_cond_right[idx_sym] if b_right_cond_right[idx_sym] != 0 else 1.0)
        b_left_cond_left = b_left_cond_left / (b_left_cond_left[idx_sym] if b_left_cond_left[idx_sym] != 0 else 1.0)
        b_left_cond_right = b_left_cond_right / (b_left_cond_right[idx_sym] if b_left_cond_right[idx_sym] != 0 else 1.0)

        r_Gll, r_GRR = b_right_cond_left, b_right_cond_right
        l_Gll, l_GRR = b_left_cond_left, b_left_cond_right
        
        rG_corr = hp.calc_invariant_metric(r_Gll, r_GRR)
        lG_corr = hp.calc_invariant_metric(l_Gll, l_GRR)
    
    
    results = {
        'i':i,
        'dIdVl': dIdVl,
        'dIdVr': dIdVr,
        'dIdV_LR': dIdV_LR,
        'dIdV_RL': dIdV_RL,
        'fine_dIdVl': fine_dIdVl,
        'ldos': ldos,
        'Gmat': Gmat,
        'gamma_sq': gamma_sq,
        'energy_0': energy_0,
        'topological_gap': topological_gap,
        'M_profile': M_profile,
        'b_right_cond_left': b_right_cond_left,
        'b_right_cond_right': b_right_cond_right,
        'b_right_GLR': b_right_GLR,
        'b_right_GRL': b_right_GRL,
        'b_left_cond_left': b_left_cond_left,
        'b_left_cond_right': b_left_cond_right,
        'rG_corr':rG_corr,
        'lG_corr':lG_corr,
        'spectrum':spectrum,
        'peak_right':pk_r,
        'peak_left':pk_l,
        'site_localization':site_localization,
        'weight_localization': weight_localization,
        'overlap_integral': overlap_integral,
        'mzm_separation': mzm_separation
    }
    return results

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

    # Apply Renormalization Factor Z to align with Kwant physics
    #Z = Delta0 / (Delta0 + gamma)
    Z = 1
    ts *= Z
    alphas *= Z
    gamma *= Z
    V0 *= Z
    mu_pm *= Z
    vz *= Z

    NL_val = qn
    
    pdi_value = 0
    if False:
        # Pass the already-renormalized parameters
        Q_nu = hp.calculate_pdi(ts, alphas, gamma, Ls, Vdisx, V0, vz, mu_pm, NL_val)        
        if 0.05 < abs(Q_nu - int(Q_nu)) < 0.95:
            Q_nu = hp.calculate_pdi(ts, alphas, gamma, Ls, Vdisx, V0, vz, mu_pm, 2 * NL_val)

            if 0.1 < abs(Q_nu - int(Q_nu)) < 0.9:
                Q_nu = hp.calculate_pdi(ts, alphas, gamma, Ls, Vdisx, V0, vz, mu_pm, 5 * NL_val)

                if 0.1 < abs(Q_nu - int(Q_nu)) < 0.9:
                    Q_nu = hp.calculate_pdi(ts, alphas, gamma, Ls, Vdisx, V0, vz, mu_pm, 10 * NL_val)    # round the converged invariant to the nearest integer
    
        pdi_value = int(np.round(Q_nu))
    
    result = [mu_pm, vz, pdi_value]
    
    if static_params.get('calc_pfaffian', False):
        pfaffian_delta_N = static_params.get('pfaffian_delta_N', 0)
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



#### constants: 
hbar = 6.582119569e-16  # eV·s
m0   = 9.10938356e-31  # kg
e0   = 1.602176634e-19   # C
eta_m = (hbar ** 2 * e0) * (1e20)/m0 # hbar^2/m0 in eV A^2
mu_B =  5.7883818066e-2  #in meV/T
meVpK = 8.6173325e-2 # Kelvin into meV 



if __name__ == "__main__":
    
    # --- Hardcoded Configuration Path (For VS Code Debugging) ---
    # Set this to a path like "Parameters/my_config.yaml" to use it as the default.
    CONFIG_PATH = "Parameters/non_interacting.yaml" 
    # ------------------------------------------------------------

    # 1. Configuration Orchestration
    config = ConfigManager.get_config(config_path=CONFIG_PATH)
    state = SimulationState(config)
    
    dirname = config.dirname
    
    if config.realization_index is not None:
        fname = f"Raw_Disorders/raw_disorder_{config.realization_index}.npy"
    else:
        fname = f"New_Disorders/{config.fname}"

    print(f"--- Starting Simulation ---")
    print(f"Output Directory: {dirname}")
    print(f"Disorder File: {fname}")
    print(f"Barrier Length (Lb): {config.Lb}")
    print(f"PDI Barrier Length (Lb_pdi): {config.Lb_pdi}")
    print(f"Acceleration Mode: {config.acceleration_type}")
    print(f"--------------------------------\n")
    
    # 2. Artifact Logging
    ConfigManager.log_artifact(config, PathConfigs.DATA / dirname)
    
    # Initialize Disorder
    print(f"Run Files Path Exists: {os.path.exists(PathConfigs.RUN_FILES)}")
    path = Path(PathConfigs.RUN_FILES / fname)
    Vdisx = hp.initialize_vdis_from_data(path, lambda_dis=config.lambda_dis, Ls=config.Ls)

    # 3. Parameter Preparation
    static_params = state.get_static_params(Vdisx, config)
    
    params_list = state.params_list
    mu_var = state.mu_var
    Vz_var = state.Vz_var
    barrier_arr = state.barrier_arr
    energies = state.energies
    
    print("conductance_flag:", f"{static_params['conductance_flag']}")
    print("spectra_flag:", f"{static_params['spectra_flag']}")

    # Pre-allocate main arrays
    lenw = config.Ls + 2*(config.Lb + config.Ln)
    num_orbitals = lenw * 4
    print(f"LEN: {num_orbitals}")
    #ldos_arr = np.zeros(shape = (len(params_list), len(energies), num_orbitals)) 
    
    dIdVs_left_arr = np.zeros(shape = (len(params_list), len(energies)))
    fine_zero_bias_conductance = np.zeros(shape = (len(params_list), 7))
    dIdVs_right_arr = np.zeros(shape = (len(params_list), len(energies)))
    dIdVs_LR_arr = np.zeros(shape = (len(params_list), len(energies)))
    dIdVs_RL_arr = np.zeros(shape = (len(params_list), len(energies)))

    barrier_right_conductance_left_arr  = np.zeros(shape=(len(params_list), config.Upoints))
    barrier_right_conductance_right_arr = np.zeros_like(barrier_right_conductance_left_arr)
    barrier_right_GLR_arr = np.zeros_like(barrier_right_conductance_left_arr)
    barrier_right_GRL_arr = np.zeros_like(barrier_right_conductance_left_arr)
    barrier_left_conductance_left_arr   = np.zeros_like(barrier_right_conductance_left_arr)
    barrier_left_conductance_right_arr  = np.zeros_like(barrier_right_conductance_left_arr)
    rG_corr_arr = np.zeros(shape = (len(params_list)))
    lG_corr_arr = np.zeros(shape = (len(params_list)))
    spectrum_arr = np.zeros(shape=(len(params_list), 4))
    peaks_left = np.zeros(shape=(len(params_list), 6))
    peaks_right = np.zeros_like(peaks_left)

    site_localizations = np.zeros_like(rG_corr_arr)
    weight_localization_arr = np.zeros_like(rG_corr_arr)
    overlap_integral_arr = np.zeros_like(rG_corr_arr)
    mzm_separation_arr = np.zeros_like(rG_corr_arr)
    gamma_sq_arr = np.zeros_like(params_list, dtype=complex)
    mp_eng_arr = np.zeros_like(params_list)
    lenw = config.Ls #+ 2*(Lb + Ln)
    mp_arr = np.zeros(shape= (len(params_list), lenw))
    Conductance_matrix = np.zeros(shape=(len(params_list),2, 2))
    topological_gap_arr = np.zeros(shape=(len(params_list)))
    
    results = []
    if config.acceleration_type == 'gpu':
        print("Using GPU acceleration (Serial Sweep).")
        results = [worker_simulation_step(pms, static_params) for pms in tqdm(params_list, desc="mu/Vz Sweep")]
        
    elif config.acceleration_type == 'parallel':
        num_workers = max(1, mp.cpu_count() - 1)
        print(f"Starting Parallel Execution with {num_workers} workers.")
        
        with mp.Pool(processes=num_workers) as pool:
            func_sim = partial(worker_simulation_step, static_params=static_params)
            results = list(tqdm(pool.imap(func_sim, params_list, chunksize=1), total=len(params_list), desc="mu/Vz Sweep"))
    else: # None
        print("Using Serial Execution (CPU).")
        for pms in tqdm(params_list, desc="mu/Vz Sweep"):
            results.append(worker_simulation_step(pms, static_params))

    for res in results:
        
        idx = res['i']
        
        dIdVs_left_arr[idx, :] = res['dIdVl']
        fine_zero_bias_conductance[idx, :] = res['fine_dIdVl']
        dIdVs_right_arr[idx, :] = res['dIdVr']
        dIdVs_LR_arr[idx, :] = res['dIdV_LR']
        dIdVs_RL_arr[idx, :] = res['dIdV_RL']
        #ldos_arr[idx, :, :] = res['ldos']
        Conductance_matrix[idx, :, :] = res['Gmat']
        gamma_sq_arr[idx] = res['gamma_sq']
        mp_eng_arr[idx] = res['energy_0']
        topological_gap_arr[idx] = res['topological_gap']
        mp_arr[idx, :] = res['M_profile']
        
        barrier_right_conductance_left_arr[idx, :] = res['b_right_cond_left']
        barrier_right_conductance_right_arr[idx, :] = res['b_right_cond_right']
        barrier_right_GLR_arr[idx, :] = res['b_right_GLR']
        barrier_right_GRL_arr[idx, :] = res['b_right_GRL']
        barrier_left_conductance_left_arr[idx, :] = res['b_left_cond_left']
        barrier_left_conductance_right_arr[idx, :] = res['b_left_cond_right']
        if res['spectrum'] is not None:
            spectrum_arr[idx, :] = res['spectrum']
        
        rG_corr_arr[idx]= res['rG_corr']
        lG_corr_arr[idx]= res['lG_corr']
                    
        peaks_left[idx,:] = res['peak_left']
        peaks_right[idx,:] = res['peak_right']
        site_localizations[idx] = res['site_localization']
        weight_localization_arr[idx] = res['weight_localization']
        overlap_integral_arr[idx] = res['overlap_integral']
        mzm_separation_arr[idx] = res['mzm_separation']
        
        
        
    pdi_data = np.array([])

    if config.pdi_flag:
        print("\nStarting PDI Calculation Loop.")
        pdi_results = []
        if config.acceleration_type == 'gpu':
            print("Using GPU acceleration (Serial Sweep).")
            pdi_results = [worker_pdi_step(pms, static_params) for pms in tqdm(params_list, desc="PDI Sweep")]

        elif config.acceleration_type == 'parallel':
            num_workers = max(1, mp.cpu_count() - 1)
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

    hp.np_save_wrapped(Vdisx, "Vdisx", dirname)
    hp.np_save_wrapped(pdi_data, "pdi_data", dirname)
    hp.np_save_wrapped(energies, "energies", dirname)
    hp.np_save_wrapped(dIdVs_left_arr, "dIdVs_left_arr", dirname)
    hp.np_save_wrapped(fine_zero_bias_conductance, "fine_zero_bias_conductance", dirname)
    hp.np_save_wrapped(dIdVs_right_arr, "dIdVs_right_arr", dirname)
    hp.np_save_wrapped(dIdVs_LR_arr, "dIdVs_LR", dirname)
    hp.np_save_wrapped(dIdVs_RL_arr, "dIdVs_RL", dirname)
    #hp.np_save_wrapped(ldos_arr, "LDOS", dirname)
    hp.np_save_wrapped(barrier_right_conductance_left_arr, "barrier_right_conductance_left_arr", dirname)
    hp.np_save_wrapped(barrier_right_conductance_right_arr, "barrier_right_conductance_right_arr", dirname)
    hp.np_save_wrapped(barrier_right_GLR_arr, "barrier_right_GLR", dirname)
    hp.np_save_wrapped(barrier_right_GRL_arr, "barrier_right_GRL", dirname)
    hp.np_save_wrapped(topological_gap_arr, "topological_gap", dirname)
    hp.np_save_wrapped(barrier_right_conductance_right_arr, "barrier_right_conductance_right_arr", dirname)    
    hp.np_save_wrapped(barrier_left_conductance_left_arr, "barrier_left_conductance_left_arr", dirname)    
    hp.np_save_wrapped(barrier_left_conductance_right_arr, "barrier_left_conductance_right_arr", dirname)
    hp.np_save_wrapped(barrier_arr,"barrier_arr", dirname)

    #hp.np_save_wrapped(Conductance_matrix, "Conductance_matrix_zero_energy", dirname)
    #hp.np_save_wrapped(gamma_sq_arr, "gamma_sq_arr", dirname)
    #hp.np_save_wrapped(mp_eng_arr, "mp_eng_arr", dirname)
    hp.np_save_wrapped(mp_arr, "mp_arr", dirname)
    hp.np_save_wrapped(params_list, "params_list", dirname) 
    
    hp.np_save_wrapped(spectrum_arr,"spectrum_arr", dirname)
    
    hp.np_save_wrapped(rG_corr_arr,"rG_corr", dirname)
    hp.np_save_wrapped(lG_corr_arr,"lG_corr", dirname)
    
    hp.np_save_wrapped(peaks_left,"peaks_left", dirname)
    hp.np_save_wrapped(peaks_right,"peaks_right", dirname)
    hp.np_save_wrapped(site_localizations, "site_localizations", dirname)
    hp.np_save_wrapped(weight_localization_arr, "weight_localization_arr", dirname)
    hp.np_save_wrapped(overlap_integral_arr, "OverlapIntegral", dirname)
    hp.np_save_wrapped(mzm_separation_arr, "mzm_separation_arr", dirname)

    
    all_params = {
        **config.model_dump(),  # unpacks all input parameters from SimulationConfig
        **static_params         # unpacks 't', 'mu_n', 'Delta', 'alpha', etc. (overrides None with calculated values)
    }
    hp.np_savez_wrapped("all_params", dirname, **all_params)
    


    print("Done.")