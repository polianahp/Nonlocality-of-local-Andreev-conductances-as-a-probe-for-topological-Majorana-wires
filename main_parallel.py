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
    Delta = static_params['Delta']
    alpha = static_params['alpha']
    Ln = static_params['Ln']
    Lb = static_params['Lb']
    Ls = static_params['Ls']
    mu_leads = static_params['mu_leads']
    barrier0 = static_params['barrier0']
    Vdisx = static_params['Vdisx']
    energies = static_params['energies']
    barrier_arr = static_params['barrier_arr']
    
    
    
    # --- 1. Build Symmetric System & Calculate Spectral Properties ---
    syst = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta=Delta, V_z=vz, 
                           alpha=alpha, Ln=Ln, Lb=Lb, 
                           Ls=Ls, mu_leads=mu_leads,
                           barrier_l=barrier0, barrier_r=barrier0, Vdisx=Vdisx)
    
    syst_closed = hp.build_system_closed(t=t, mu=mu, mu_n=mu_n, Delta=Delta, V_z=vz, 
                           alpha=alpha, Ln=Ln, Lb=Lb, 
                           Ls=Ls, mu_leads=mu_leads,
                           barrier_l=barrier0, barrier_r=barrier0, Vdisx=Vdisx)
    
    # Majorana metrics
    M_profile, energy_0 = hp.calculate_local_mp(syst_closed)
    gamma_sq = hp.calculate_gamma_squared(syst_closed)
    
    # dI/dV and Conductance Matrix
    dIdVl, dIdVr, ldos = hp.calc_dIdV(syst, energies)
    Gmat = hp.calc_conductance_matrix(syst, 0.0)
    
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
        # Varying Right Barrier (UR)
        syst_UR = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta=Delta, 
                                  V_z=vz, alpha=alpha, Ln=Ln, Lb=Lb, 
                                  Ls=Ls, mu_leads=mu_leads, barrier_l=barrier0,
                                  barrier_r=barrier_arr[k], Vdisx=Vdisx)
        
        cL, cR = hp.calc_conductance(syst_UR, energy=0.0)
        b_right_cond_left[k] = cL
        b_right_cond_right[k] = cR
        
        # Varying Left Barrier (UL)
        syst_UL = hp.build_system(t=t, mu=mu, mu_n=mu_n, Delta=Delta, 
                                  V_z=vz, alpha=alpha, Ln=Ln, Lb=Lb, 
                                  Ls=Ls, mu_leads=mu_leads, barrier_l=barrier_arr[k], 
                                  barrier_r=barrier0, Vdisx=Vdisx)

        cL, cR = hp.calc_conductance(syst_UL, energy=0.0)
        b_left_cond_left[k] = cL
        b_left_cond_right[k] = cR
        
    l_Gll, l_GRR = b_left_cond_left, b_left_cond_right #varying left barrier and getting local conductances
    r_Gll, r_GRR = b_right_cond_left, b_right_cond_right #varying left barrier and getting local conductances
    
    
    
    #rG_corr = np.dot(r_Gll, r_GRR)/(np.linalg.norm(r_Gll) * np.linalg.norm(r_GRR))
    #rG_corr = np.dot(r_Gll, r_GRR)/(np.linalg.norm(    r_Gll) * np.linalg.norm(r_GRR))
    
    rG_corr = hp.calc_invariant_metric(r_Gll, r_GRR)
    lG_corr = hp.calc_invariant_metric(l_Gll, l_GRR)

    
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
    }
    return results

def worker_pdi_step(param_tuple, static_params):
    """
    Worker function for the PDI calculation loop (Loop 2).
    """
    i, mu_pm, vz = param_tuple
    
    # Unpack necessary static params
    t = static_params['t']
    Delta = static_params['Delta']
    alpha = static_params['alpha']
    Ls = static_params['Ls']
    Vdisx = static_params['Vdisx']
    
    
    
    # Calculate PDI
    # Note: Vdisx is negated here based on convention, 
    # copying Biniyakks original script convention Vdisx --> -Vdisx
    pdi_val = hp.calculate_pdi(t, mu_pm, Delta, vz, alpha, Ls, -Vdisx, q_N=100)
    
    return [mu_pm, vz, pdi_val]


if __name__ == "__main__":
    
    ####### System Parameters
    t = 102.0
    mu_n = 0.2
    mu_leads = 20.0
    Delta = 0.5
    alpha = 3.5
    Ln = 20 # normal metal length
    Lb = 4 #barrier length
    Ls = 500 #super conductor length
    #V_c = np.sqrt(mu**2 + Delta**2)
    barrier0 = 5
    
    
    ## setting up different tests
    V0 = 0.0  * Delta 
    dirname = 'reruncorr_clean_dis_test'
    
    #V0 = 3.5  * Delta 
    #dirname = 'new_corr_med_dis_test'    #corr_med_dis_test
    
    #V0 = 10.5 * Delta 
    #dirname = 'reruncorr_stong_dis_test'  #corr_stong_dis_test

    Upoints = 50 
    num_engs = 101 
    num_vz_var = 51
    num_mu_var= 51

    # Sweeping arrays
    mu_rng = 0.5
    mu_var = np.linspace(0.5, 1.5, num_mu_var)
    Vz_var = np.linspace(0.6, 1.6, num_vz_var) 
    params_list = [pms for pms in itr.product(mu_var, Vz_var)]
    params_list = [[i, pms[0], pms[1]] for i, pms in enumerate(params_list)]
    
    barrier_arr = np.linspace(0, 40*barrier0, Upoints)
    energies = np.linspace(-0.5, 0.5, num_engs)

    # Initialize Disorder
    print(f"Run Files Path Exists: {os.path.exists(PathConfigs.RUN_FILES)}")
    fname = "Data.npz"
    path = Path(PathConfigs.RUN_FILES/fname)
    
    try:
        Vdisx = hp.initialize_vdis_from_data(path) 
        mxvdis = np.max(np.abs(Vdisx))
        Vdisx = V0 * (Vdisx / mxvdis) #renormalizing Vdis 
    except:
        print("Warning: Could not load Vdisx from file. Initializing zeros.")
        Vdisx = np.zeros(Ls)
        
    
    
    
    

    # Dictionary of static parameters to pass to workers
    static_params = {
        't': t, 'mu_n': mu_n, 'Delta': Delta, 'alpha': alpha,
        'Ln': Ln, 'Lb': Lb, 'Ls': Ls, 'mu_leads': mu_leads,
        'barrier0': barrier0, 'Vdisx': Vdisx,
        'energies': energies, 'barrier_arr': barrier_arr
    }
    

    # Pre-allocate main arrays
    dIdVs_left_arr = np.zeros(shape = (len(params_list), len(energies)))
    dIdVs_right_arr = np.zeros(shape = (len(params_list), len(energies)))
    ldos_arr = np.zeros(shape = (len(params_list), len(energies), 2192)) 

    barrier_right_conductance_left_arr  = np.zeros(shape=(len(params_list), Upoints))
    barrier_right_conductance_right_arr = np.zeros_like(barrier_right_conductance_left_arr)
    barrier_left_conductance_left_arr   = np.zeros_like(barrier_right_conductance_left_arr)
    barrier_left_conductance_right_arr  = np.zeros_like(barrier_right_conductance_left_arr)
    rG_corr_arr = np.zeros(shape = (len(params_list)))
    lG_corr_arr = np.zeros(shape = (len(params_list)))
    
    
    gamma_sq_arr = np.zeros_like(params_list, dtype=complex)
    mp_eng_arr = np.zeros_like(params_list)
    lenw = Ls + 2*(Lb + Ln)
    mp_arr = np.zeros(shape= (len(params_list), lenw))
    Conductance_matrix = np.zeros(shape=(len(params_list),2, 2))
    

    # -------------------------------------------------------------------------
    # PARALLEL EXECUTION SETUP
    # -------------------------------------------------------------------------
    
    # Determine CPUs (leave 1 or 2 free for system if possible, else use all)
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"Starting Parallel Execution with {num_workers} workers.")
    
    # Create the Pool ONCE to be reused
    with mp.Pool(processes=num_workers) as pool:
        
        
        # Prepare iterable: list of (index, val)
        #vz_iterable = list(enumerate(Vz_var))
                
        
        # Create partial function with static params frozen
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
            
            rG_corr_arr[idx]= res['rG_corr']
            lG_corr_arr[idx]= res['lG_corr']
            

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
    # SAVE RESULTS
    # -------------------------------------------------------------------------
    #dirname = "corr_test"
    print(f"\nSaving data to: {dirname}")

    hp.np_save_wrapped(pdi_data, "pdi_data", dirname)
    hp.np_save_wrapped(energies, "energies", dirname)
    hp.np_save_wrapped(dIdVs_left_arr, "dIdVs_left_arr", dirname)
    hp.np_save_wrapped(dIdVs_right_arr, "dIdVs_right_arr", dirname)
    hp.np_save_wrapped(ldos_arr, "LDOS", dirname)
    hp.np_save_wrapped(barrier_right_conductance_left_arr, "barrier_right_conductance_left_arr", dirname)
    hp.np_save_wrapped(barrier_right_conductance_right_arr, "barrier_right_conductance_right_arr", dirname)    
    hp.np_save_wrapped(barrier_left_conductance_left_arr, "barrier_left_conductance_left_arr", dirname)    
    hp.np_save_wrapped(barrier_left_conductance_right_arr, "barrier_left_conductance_right_arr", dirname)

    hp.np_save_wrapped(Conductance_matrix, "Conductance_matrix", dirname)
    hp.np_save_wrapped(gamma_sq_arr, "gamma_sq_arr", dirname)
    hp.np_save_wrapped(mp_eng_arr, "mp_eng_arr", dirname)
    hp.np_save_wrapped(mp_arr, "mp_arr", dirname)
    hp.np_save_wrapped(params_list, "params_list", dirname)
    
    hp.np_save_wrapped(rG_corr_arr,"rG_corr", dirname)
    hp.np_save_wrapped(lG_corr_arr,"lG_corr", dirname)
    


    print("Done.")