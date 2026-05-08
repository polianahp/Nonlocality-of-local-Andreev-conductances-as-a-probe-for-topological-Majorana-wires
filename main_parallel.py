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
    b_left_cond_left = np.zeros(points)
    b_left_cond_right = np.zeros(points)
    
    dIdVl, dIdVr, ldos = 0,0,0
    Vdisx = Vdisx * V0
    barrier_tot = barrier0 #+ mu
    gamma_sq = 0
    energy_0 = 0
    M_profile = [0]
    spectrum = None
    rho_M1 = np.zeros(Ls, dtype = complex)
    rho_M2 = np.zeros(rho_M1.shape, dtype = complex)
    site_localization = 100
    weight_localization = 1.0
    overlap_integral = 0.0

    # --- 1. Build Symmetric System & Calculate Spectral Properties ---

    if static_params['spectra_flag'] or static_params['localization_flag']:
        syst_closed = hp.build_system_closed(t, mu, gamma, Delta0, vz, alpha, Ls, Vdisx)
        evals, evecs = hp.solve_ham(syst_closed, solver_type=solver_type, k=num_eigenvalues)


        if static_params['localization_flag']:
            rho_M1, rho_M2, _ = hp.get_psiM_density(evals, evecs)
            site_localization = hp.calc_MZM_localization(rho_M1, rho_M2)
            weight_localization = hp.calc_weight_localization(rho_M1, rho_M2, weight_threshold=static_params['weight_threshold'])
            overlap_integral = hp.calc_overlap(rho_M1, rho_M2)
        if static_params['spectra_flag']:
            spectrum = hp.sort_spectrum(evals, evecs)
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
    
    if static_params['conductance_flag']:
        
        syst = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma = gamma, V_z=vz, 
                           alpha=alpha, Ln=Ln, Lb=Lb, 
                           Ls=Ls, mu_leads=mu_leads,
                           barrier_l=barrier_tot, barrier_r=barrier_tot, Vdisx=Vdisx)
    
        dIdVl, dIdVr, ldos = hp.calc_dIdV(syst, energies, solver_type=solver_type)
        Gmat = hp.calc_conductance_matrix(syst, 0.0, solver_type=solver_type)
        for k, eng in enumerate(eng_window):
            cL, cR = hp.calc_conductance(syst, energy=eng, solver_type=solver_type)
            csL[k] = cL
            csR[k] = cR
    
        pk_l_pos = hp.detect_peaks(csL, eng_window)
        pk_r_pos = hp.detect_peaks(csR, eng_window)
    
        if pk_l_pos is not None:
            pk_l = np.asarray([1, eng_window[pk_l_pos], csL[pk_l_pos]])
        else:
            pk_l = np.asarray([0, 10, 10]) 
            #setting difference to be a huge number comparable to the actual 
            # gap so I can postprocess easily later
        
        if pk_r_pos is not None:
            pk_r = np.asarray([1, eng_window[pk_r_pos], csR[pk_r_pos]])
            #setting difference to be a huge number comparable to the actual 
            # gap so I can postprocess easily later
        else:
            pk_r = np.asarray([0, 10, 10])
    

        # Note: this is run serially inside the worker because the overhead 
        # of spawning sub-processes here would be too high.
        for k in range(points):
            barrier_var_tot = barrier_arr[k] #+ mu
            
            # Varying Right Barrier (UR)
            syst_UR = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma = gamma,
                                    V_z=vz, alpha=alpha, Ln=Ln, Lb=Lb, 
                                    Ls=Ls, mu_leads=mu_leads, barrier_l=barrier_tot,
                                    barrier_r=barrier_var_tot, Vdisx=Vdisx)
            
            cL, cR = hp.calc_conductance(syst_UR, energy=0.0, solver_type=solver_type)
            b_right_cond_left[k] = cL
            b_right_cond_right[k] = cR
            
            
        r_Gll, r_GRR = b_right_cond_left, b_right_cond_right #varying left barrier and getting local conductances
        
        rG_corr = hp.calc_correlation(r_Gll, r_GRR)
    
    
    results = {
        'i':i,
        'dIdVl': dIdVl,
        'dIdVr': dIdVr,
        'ldos': ldos,
        'Gmat': Gmat,
        'gamma_sq': gamma_sq,
        'energy_0': energy_0,
        'M_profile': M_profile,
        'b_right_cond_left': b_right_cond_left,
        'b_right_cond_right': b_right_cond_right,
        'b_left_cond_left': b_left_cond_left,
        'b_left_cond_right': b_left_cond_right,
        'rG_corr':rG_corr,
        'lG_corr':0,
        'spectrum':spectrum,
        'peak_right':pk_r,
        'peak_left':pk_l,
        'site_localization':site_localization,
        'weight_localization': weight_localization,
        'overlap_integral': overlap_integral
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
    
    NL_val = qn
    
    Q_nu = hp.calculate_pdi(ts, alphas, gamma, Ls, Vdisx, V0, vz, mu_pm, NL_val)
    
    if 0.05 < abs(Q_nu - int(Q_nu)) < 0.95:
        Q_nu = hp.calculate_pdi(ts, alphas, gamma, Ls, Vdisx, V0, vz, mu_pm, 2 * NL_val)
        
        if 0.1 < abs(Q_nu - int(Q_nu)) < 0.9:
            Q_nu = hp.calculate_pdi(ts, alphas, gamma, Ls, Vdisx, V0, vz, mu_pm, 5 * NL_val)
            
            if 0.1 < abs(Q_nu - int(Q_nu)) < 0.9:
                Q_nu = hp.calculate_pdi(ts, alphas, gamma, Ls, Vdisx, V0, vz, mu_pm, 10 * NL_val) 
                
    # round the converged invariant to the nearest integer
    pdi_value = int(np.round(Q_nu))
    
    return [mu_pm, vz, pdi_value]



#### constants: 
hbar = 6.582119569e-16  # eV·s
m0   = 9.10938356e-31  # kg
e0   = 1.602176634e-19   # C
eta_m = (hbar ** 2 * e0) * (1e20)/m0 # hbar^2/m0 in eV A^2
mu_B =  5.7883818066e-2  #in meV/T
meVpK = 8.6173325e-2 # Kelvin into meV 



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run parallel transport and PDI simulation.")
    
    action = 'store_true'
    parser.add_argument("--dirname", type=str, default="peaktesting_Agaiinnn", help="Directory name for saving output data.")
    parser.add_argument("--fname", type=str, default="Tdis.npz",help="File name for the disorder potential.")
    parser.add_argument("--Lb_pdi", type=int, default=3, help="Barrier length.")
    parser.add_argument("--no_pdi", action=action, help="Skip the time-consuming PDI calculation.")
    parser.add_argument("--no_conductance", action=action, help="Skip Conductance Calculation.")
    parser.add_argument("--no_spectra", action=action, help="Skip Spectra Calculations.")
    parser.add_argument("--no_localization", action=action, help="Skip Localization Calculations.")
    parser.add_argument("--acceleration_type", type=str, default="parallel", choices=["gpu", "parallel", "None"], help="Acceleration mode (gpu, parallel, or None).")
    
    args = parser.parse_args()


    dirname = f"peak_testing/{args.dirname}"
    fname = f"New_Disorders/{args.fname}"
    Lb = 3
    Lb_pdi = args.Lb_pdi  

    print(f"--- Starting Simulation ---")
    print(f"Output Directory: {dirname}")
    print(f"Disorder File: {fname}")
    print(f"Barrier Length (Lb): {Lb}")
    print(f"PDI Barrier Length (Lb_pdi): {Lb_pdi}")
    print(f"Acceleration Mode: {args.acceleration_type}")
    print(f"--------------------------------\n")
    
    
    
    ####### System Parameters
    
    Ls = 300 # wire length

    Ln = 0 #length of normal region. See Dourado 2023
    a0 = 100 # unit cell in A
    ms = 0.023 # effective mass
    
    t = 1000 * eta_m/(2 * a0**2 * ms) # hopping in meV
    alpha = 140.0/a0 # Rashba SOC
    
    Delta_0= 0.3 # parent SC gap
    gamma = 0.2 # SM-SC coupling strength in meV
    Delta = Delta_0 * gamma /(Delta_0 + gamma) #induced gap
    
    mu_leads = t # lead chemical potential (meV)
    
    barrier0 = 2 #barrier energy (meV)
    
    V0 = 1.2

    Upoints = 20 
    num_engs = 101  

    mu_n = 0.0

    mu_max = 4.5
    mu_min = 0
    mu_rng = mu_max - mu_min
    mu_dist = 0.02 #spacing between points
    Nmu = int(mu_rng/mu_dist) #total number of paramter space points for mu
    mu_var = np.linspace(mu_min, mu_max, Nmu)
    
    Vz_max = 1.3
    Vz_min = 0.0
    Vz_rng = Vz_max - Vz_min
    Vz_dist = 0.02 #spacing between points
    Nvz =  int(Vz_rng/Vz_dist)
    Vz_var = np.linspace(Vz_min, Vz_max, Nvz) 
    
    
    params_list = [pms for pms in itr.product(mu_var, Vz_var)]
    params_list = [[i, pms[0], pms[1]] for i, pms in enumerate(params_list)]
    
    
    barrier_arr = np.linspace(barrier0, 40*barrier0, Upoints)
    energies = np.linspace(-0.5, 0.5, num_engs)
    
    
    num_eigenvalues = 12 #number of eigenvalues to calculate in the low energy spectra, so 10 above and below the MZMs in this case

    # Initialize Disorder
    print(f"Run Files Path Exists: {os.path.exists(PathConfigs.RUN_FILES)}")
    
    
    path = Path(PathConfigs.RUN_FILES/fname)
    
    Vdisx = hp.initialize_vdis_from_data(path)  

    # Dictionary of static parameters to pass to workers
    static_params = {
        't': t,
        'mu_n': mu_n,
        'Delta0': Delta_0,
        'alpha': alpha,
        'gamma': gamma,
        'V0': V0,
        'qn': 20,

        'Ln': Ln,
        'Lb': Lb,
        #'Lb_pdi':Lb_pdi,
        'Barrier_Height': barrier0,
        'Ls': Ls,
        'mu_leads': mu_leads,

        'barrier0': barrier0,
        'Vdisx': Vdisx,

        'energies': energies,
        'barrier_arr': barrier_arr,
        
        'num_eigenvalues':num_eigenvalues,
        'weight_threshold': 0.9,
        'eng_window_range':51,
        'conductance_flag': not args.no_conductance,
        'spectra_flag': not args.no_spectra,
        'localization_flag': not args.no_localization,
        'solver_type': 'gpu' if args.acceleration_type == 'gpu' else 'cpu'
    }
    print("conductance_flag:", f"{static_params['conductance_flag']}")
    print("spectra_flag:", f"{static_params['spectra_flag']}")

    

    # Pre-allocate main arrays
    lenw = Ls + 2*(Lb + Ln)
    num_orbitals = lenw * 4
    print(f"LEN: {num_orbitals}")
    ldos_arr = np.zeros(shape = (len(params_list), len(energies), num_orbitals)) 
    
    dIdVs_left_arr = np.zeros(shape = (len(params_list), len(energies)))
    dIdVs_right_arr = np.zeros(shape = (len(params_list), len(energies)))

    barrier_right_conductance_left_arr  = np.zeros(shape=(len(params_list), Upoints))
    barrier_right_conductance_right_arr = np.zeros_like(barrier_right_conductance_left_arr)
    barrier_left_conductance_left_arr   = np.zeros_like(barrier_right_conductance_left_arr)
    barrier_left_conductance_right_arr  = np.zeros_like(barrier_right_conductance_left_arr)
    rG_corr_arr = np.zeros(shape = (len(params_list)))
    lG_corr_arr = np.zeros(shape = (len(params_list)))
    spectrum_arr = np.zeros(shape=(len(params_list), num_eigenvalues))
    peaks_left = np.zeros(shape=(len(params_list), 3))
    peaks_right = np.zeros_like(peaks_left)

    site_localizations = np.zeros_like(rG_corr_arr)
    weight_localization_arr = np.zeros_like(rG_corr_arr)
    overlap_integral_arr = np.zeros_like(rG_corr_arr)
    gamma_sq_arr = np.zeros_like(params_list, dtype=complex)
    mp_eng_arr = np.zeros_like(params_list)
    lenw = Ls #+ 2*(Lb + Ln)
    mp_arr = np.zeros(shape= (len(params_list), lenw))
    Conductance_matrix = np.zeros(shape=(len(params_list),2, 2))
    
    results = []
    if args.acceleration_type == 'gpu':
        print("Using GPU acceleration (Serial Sweep).")
        results = [worker_simulation_step(pms, static_params) for pms in tqdm(params_list, desc="mu/Vz Sweep")]
        
    elif args.acceleration_type == 'parallel':
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
        dIdVs_right_arr[idx, :] = res['dIdVr']
        ldos_arr[idx, :, :] = res['ldos']
        Conductance_matrix[idx, :, :] = res['Gmat']
        gamma_sq_arr[idx] = res['gamma_sq']
        mp_eng_arr[idx] = res['energy_0']
        mp_arr[idx, :] = res['M_profile']
        
        barrier_right_conductance_left_arr[idx, :] = res['b_right_cond_left']
        barrier_right_conductance_right_arr[idx, :] = res['b_right_cond_right']
        barrier_left_conductance_left_arr[idx, :] = res['b_left_cond_left']
        barrier_left_conductance_right_arr[idx, :] = res['b_left_cond_right']
        spectrum_arr[idx, :] = res['spectrum']
        
        rG_corr_arr[idx]= res['rG_corr']
        lG_corr_arr[idx]= res['lG_corr']
                    
        peaks_left[idx,:] = res['peak_left']
        peaks_right[idx,:] = res['peak_right']
        site_localizations[idx] = res['site_localization']
        weight_localization_arr[idx] = res['weight_localization']
        overlap_integral_arr[idx] = res['overlap_integral']
        
        
        
    pdi_data = np.array([])

    if not args.no_pdi:
        print("\nStarting PDI Calculation Loop.")
        pdi_results = []
        if args.acceleration_type == 'gpu':
            print("Using GPU acceleration (Serial Sweep).")
            pdi_results = [worker_pdi_step(pms, static_params) for pms in tqdm(params_list, desc="PDI Sweep")]

        elif args.acceleration_type == 'parallel':
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
    hp.np_save_wrapped(dIdVs_right_arr, "dIdVs_right_arr", dirname)
    #hp.np_save_wrapped(ldos_arr, "LDOS", dirname)
    hp.np_save_wrapped(barrier_right_conductance_left_arr, "barrier_right_conductance_left_arr", dirname)
    hp.np_save_wrapped(barrier_right_conductance_right_arr, "barrier_right_conductance_right_arr", dirname)    
    hp.np_save_wrapped(barrier_left_conductance_left_arr, "barrier_left_conductance_left_arr", dirname)    
    hp.np_save_wrapped(barrier_left_conductance_right_arr, "barrier_left_conductance_right_arr", dirname)
    hp.np_save_wrapped(barrier_arr,"barrier_arr", dirname)

    hp.np_save_wrapped(Conductance_matrix, "Conductance_matrix_zero_energy", dirname)
    hp.np_save_wrapped(gamma_sq_arr, "gamma_sq_arr", dirname)
    hp.np_save_wrapped(mp_eng_arr, "mp_eng_arr", dirname)
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

    
    all_params = {
        **static_params,  # unpacks 't', 'mu_n', 'Delta', 'alpha', etc.
        
        'a0': a0,
        'ms': ms,
        'Delta_0': Delta_0,
        'gamma': gamma,
        'V0': V0,
        'Upoints': Upoints,
        'num_engs': num_engs,
        
        'mu_max': mu_max,
        'mu_min': mu_min,
        'mu_rng': mu_rng,
        'mu_dist': mu_dist,
        'Nmu': Nmu,
        'mu_var': mu_var,
        
        'Vz_max': Vz_max,
        'Vz_min': Vz_min,
        'Vz_rng': Vz_rng,
        'Vz_dist': Vz_dist,
        'Nvz': Nvz,
        'Vz_var': Vz_var,
        
    }
    hp.np_savez_wrapped("all_params", dirname, **all_params)
    


    print("Done.")