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
    
    
    Vdisx = Vdisx * V0
    
    barrier_tot = barrier0 #+ mu
    
    
    
    # --- 1. Build Symmetric System & Calculate Spectral Properties ---
    syst = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma = gamma, V_z=vz, 
                           alpha=alpha, Ln=Ln, Lb=Lb, 
                           Ls=Ls, mu_leads=mu_leads,
                           barrier_l=barrier_tot, barrier_r=barrier_tot, Vdisx=Vdisx)
    
    syst_closed = hp.build_system_closed(t, mu, gamma, Delta0, vz, alpha, Ls, Vdisx)
    
    spectrum = hp.calc_spectrum(syst_closed, k=num_eigenvalues)
    
    # Majorana metrics
    M_profile, energy_0 = hp.calculate_local_mp(syst_closed)
    gamma_sq = hp.calculate_gamma_squared(syst_closed)
    
    # dI/dV and Conductance Matrix
    dIdVl, dIdVr, ldos = 0,0,0
    #dIdVl, dIdVr, ldos = hp.calc_dIdV(syst, energies)
    Gmat = hp.calc_conductance_matrix(syst, 0.0)
    
    
    eng_window = np.linspace(-0.15, 0.15, eng_window_range)
    csL = np.zeros_like(eng_window)
    csR = np.zeros_like(eng_window)
    for k, eng in enumerate(eng_window):
        cL, cR = hp.calc_conductance(syst, energy=eng)
        csL[k] = cL
        csR[k] = cR
        
    pk_l = hp.detect_peaks(csL, eng_window)
    pk_r = hp.detect_peaks(csR, eng_window)
    
    
    # --- 2. Barrier Sweeps (Nested Loop logic) ---
    points = len(barrier_arr)
    
    # Pre-allocate local arrays
    b_right_cond_left = np.zeros(points)
    b_right_cond_right = np.zeros(points)
    b_left_cond_left = np.zeros(points)
    b_left_cond_right = np.zeros(points)

    # Note: this is run serially inside the worker because the overhead 
    # of spawning sub-processes here would be too high.
    for k in range(points):
        barrier_var_tot = barrier_arr[k] #+ mu
        
        # Varying Right Barrier (UR)
        syst_UR = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma = gamma,
                                  V_z=vz, alpha=alpha, Ln=Ln, Lb=Lb, 
                                  Ls=Ls, mu_leads=mu_leads, barrier_l=barrier_tot,
                                  barrier_r=barrier_var_tot, Vdisx=Vdisx)
        
        cL, cR = hp.calc_conductance(syst_UR, energy=0.0)
        b_right_cond_left[k] = cL
        b_right_cond_right[k] = cR
        
        # Varying Left Barrier (UL)
        #syst_UL = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma = gamma,
        #                          V_z=vz, alpha=alpha, Ln=Ln, Lb=Lb, 
        #                          Ls=Ls, mu_leads=mu_leads, barrier_l=barrier_var_tot, 
        #                          barrier_r=barrier_tot, Vdisx=Vdisx)
#
        #cL, cR = hp.calc_conductance(syst_UL, energy=0.0)
        #b_left_cond_left[k] = cL
        #b_left_cond_right[k] = cR
        
    l_Gll, l_GRR = b_left_cond_left, b_left_cond_right #varying left barrier and getting local conductances
    r_Gll, r_GRR = b_right_cond_left, b_right_cond_right #varying left barrier and getting local conductances
    
    
    
    #rG_corr = np.dot(r_Gll, r_GRR)/(np.linalg.norm(r_Gll) * np.linalg.norm(r_GRR))
    #rG_corr = np.dot(r_Gll, r_GRR)/(np.linalg.norm(    r_Gll) * np.linalg.norm(r_GRR))
    
    rG_corr = hp.calc_correlation(r_Gll, r_GRR)
    lG_corr = 0#hp.calc_correlation(l_Gll, l_GRR)

    
    # Pack all results into a dictionary to return to main process
    
    
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
        'lG_corr':lG_corr,
        'spectrum':spectrum,
        'peak_left':pk_l,
        'peak_right':pk_r
    }
    return results

def worker_pdi_step(param_tuple, static_params):
    """
    Worker function for the PDI calculation loop (Loop 2).
    """
    i, mu_pm, vz = param_tuple
    
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
    
    parser.add_argument("--dirname", type=str, default="testing", help="Directory name for saving output data.")
    parser.add_argument("--fname", type=str, default="Tdis.npz",help="File name for the disorder potential.")
    parser.add_argument("--Lb_pdi", type=int, default=3, help="Barrier length.")
    
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
    print(f"--------------------------------\n")
    
    ####### System Parameters
    '''
    t = 102.0
    mu_n = 0.2
    mu_leads = 20.0
    Delta = 0.5
    alpha = 3.5
    Ln = 20 # normal metal length
    Lb = 4 #barrier length
    Lb_pdi = Lb
    Ls = 500 #super conductor length
    #V_c = np.sqrt(mu**2 + Delta**2)
    barrier0 = 5
    '''
    
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
    
    V0 = 1.2#10.5 * Delta 

    Upoints = 20 
    num_engs = 101  

    mu_n = 0.0

    mu_max = 4.5
    mu_min = 0.0
    mu_rng = mu_max - mu_min
    mu_dist = 0.02 #spacing between points
    Nmu = int(mu_rng/mu_dist) #total number of paramter space points for mu
    mu_var = np.linspace(mu_min, mu_max, Nmu)
    
    Vz_max = 1.3
    Vz_min = 0.0
    Vz_rng = Vz_max - Vz_min
    Vz_dist = 0.02 #spacing between points
    Nvz = int(Vz_rng/Vz_dist)
    Vz_var = np.linspace(Vz_min, Vz_max, Nvz) 
    
    
    params_list = [pms for pms in itr.product(mu_var, Vz_var)]
    params_list = [[i, pms[0], pms[1]] for i, pms in enumerate(params_list)]
    
    
    barrier_arr = np.linspace(barrier0, 40*barrier0, Upoints)
    energies = np.linspace(-0.5, 0.5, num_engs)
    
    
    num_eigenvalues = 22 #number of eigenvalues to calculate in the low energy spectra, so 10 above and below the MZMs in this case

    # Initialize Disorder
    print(f"Run Files Path Exists: {os.path.exists(PathConfigs.RUN_FILES)}")
    
    
    path = Path(PathConfigs.RUN_FILES/fname)
    
    Vdisx = hp.initialize_vdis_from_data(path)  

        
    #print(Vdisx)
    

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
        'eng_window_range':20
    }
    

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
    spectrum_arr = np.zeros(shape=(len(params_list), 22))
    peaks_left = np.zeros(shape=(len(params_list),2))
    peaks_right = np.zeros_like(peaks_left)
    
    gamma_sq_arr = np.zeros_like(params_list, dtype=complex)
    mp_eng_arr = np.zeros_like(params_list)
    lenw = Ls #+ 2*(Lb + Ln)
    mp_arr = np.zeros(shape= (len(params_list), lenw))
    Conductance_matrix = np.zeros(shape=(len(params_list),2, 2))
    
    
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Starting Parallel Execution with {num_workers} workers.")
    
    with mp.Pool(processes=num_workers) as pool:
        
        
        # Prepare iterable: list of (index, val)
        #vz_iterable = list(enumerate(Vz_var))
        
        func_sim = partial(worker_simulation_step, static_params=static_params)
        
        # chunksize=1 is usually fine for heavy tasks, allows better load balancing
        results_iterator = pool.imap(func_sim, params_list, chunksize=1)
        
        # Iterate and fill arrays
        for res in tqdm(results_iterator, total=len(params_list), desc="mu/Vz Sweep"):
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
            
            
            

        print("\n--- Starting PDI Calculation ---")
        
        
        # Create partial function
        func_pdi = partial(worker_pdi_step, static_params=static_params)
        
        pdi_results_iter = pool.imap(func_pdi, params_list, chunksize=1)
        
        # pdi_data structure: list of [mu*Vc, Vz_raw, pdi_val]
        pdi_data = []
        for res in tqdm(pdi_results_iter, total=len(params_list), desc="PDI Sweep"):
            pdi_data.append(res)
        pdi_data = np.array(pdi_data)

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