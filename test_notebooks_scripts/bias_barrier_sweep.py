# ==========================================
# Cell Block 1: Imports & Worker Definition
# ==========================================
import os
import sys
import numpy as np
import kwant
import matplotlib.pyplot as plt
from pathlib import Path
import multiprocessing as mp
from functools import partial

# Add project path to import helpers
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

import helpers as hp
from config import PathConfigs

def worker_sweep_step(task_data, t, mu_val, mu_n, gamma, Delta0, V_z_val, alpha, Ln, Lb, Ls, mu_leads, barrier_l, Vdisx, energies, solver):
    """
    Worker function executed by multiprocessing pool.
    Rebuilds system for a single barrier_r value and calculates conductance matrix for all energies.
    """
    i, br = task_data
    num_energies = len(energies)
    
    # Pre-allocate rows
    row_LL = np.zeros(num_energies)
    row_RR = np.zeros(num_energies)
    row_RL = np.zeros(num_energies)
    row_LR = np.zeros(num_energies)
    
    # Rebuild system for this specific right barrier value
    syst = hp.build_system(
        t=t, mu=mu_val, mu_n=mu_n, gamma=gamma, Delta0=Delta0, 
        V_z=V_z_val, alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, 
        mu_leads=mu_leads, barrier_l=barrier_l, barrier_r=br, 
        Vdisx=Vdisx
    )
    
    for j, eng in enumerate(energies):
        G_mat = hp.calc_conductance_matrix(syst, eng, solver_type=solver)
        row_LL[j] = G_mat[0, 0]
        row_RR[j] = G_mat[1, 1]
        row_RL[j] = G_mat[1, 0]  # G_RL: Current at Right (1) due to Left (0)
        row_LR[j] = G_mat[0, 1]  # G_LR: Current at Left (0) due to Right (1)
        
    return i, row_LL, row_RR, row_RL, row_LR


def plot_conductance_correlation(energies, G_LL, G_RR, G_LR, G_RL, plot_dir):
    """
    Calculates and plots the correlation between left and right conductances
    as a function of bias voltage (energy).
    This is done for both local (G_LL vs G_RR) and non-local (G_LR vs G_RL) conductances.
    """
    corr_local = []
    corr_nonlocal = []
    
    for j in range(len(energies)):
        # Local conductance correlation (G_LL vs G_RR)
        try:
            c_loc = hp.calc_correlation(G_LL[:, j], G_RR[:, j])
            if np.isnan(c_loc) or np.isinf(c_loc):
                c_loc = 0.0
        except Exception:
            c_loc = 0.0
        corr_local.append(c_loc)
        
        # Non-local conductance correlation (G_LR vs G_RL)
        try:
            c_nonloc = hp.calc_correlation(G_LR[:, j], G_RL[:, j])
            if np.isnan(c_nonloc) or np.isinf(c_nonloc):
                c_nonloc = 0.0
        except Exception:
            c_nonloc = 0.0
        corr_nonlocal.append(c_nonloc)
        
    corr_local = np.array(corr_local)
    corr_nonlocal = np.array(corr_nonlocal)
    
    plt.figure(figsize=(10, 6))
    plt.plot(energies, corr_local, label='Local Correlation ($G_{LL}$ vs $G_{RR}$)', color='royalblue', linewidth=2)
    plt.plot(energies, corr_nonlocal, label='Non-Local Correlation ($G_{LR}$ vs $G_{RL}$)', color='crimson', linewidth=2)
    
    # Threshold line at 0.9 (from protocol standard) and guide line at 0.0
    plt.axhline(0.9, color='gray', linestyle='--', alpha=0.7, label='Correlation Threshold (0.9)')
    plt.axhline(0.0, color='black', linestyle='-', alpha=0.3)
    
    plt.title('Conductance Correlation as a Function of Bias Voltage', fontsize=14)
    plt.xlabel('Bias Voltage / Energy (meV)', fontsize=12)
    plt.ylabel('Correlation (Cosine Similarity)', fontsize=12)
    plt.ylim(-1.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    save_path = Path(plot_dir, 'conductance_correlation_vs_bias.png')
    plt.savefig(save_path, dpi=300)
    print(f"Correlation plot saved to: {save_path}")
    plt.close()

# ==========================================
# Main Execution Guard
# ==========================================
if __name__ == '__main__':
    # ==========================================
    # Configuration Flags
    # ==========================================
    just_plot = False  # Set to True to skip running the sweep and load existing data directly
    
    # ==========================================
    # Cell Block 2: Load Data & Parameters
    # ==========================================
    save_plots = True

    dirname = Path(PathConfigs.DATA/"one_disorder/disorder_realization_00")
    plot_dir = Path(dirname, "Plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Load params and constants exactly like in protocol.ipynb
    params = np.load(Path(dirname/"all_params.npz"), allow_pickle=True)
    mu_n = float(params['mu_n'])
    t = float(params['t'])
    mu_leads = float(params['mu_leads'])
    Delta0 = float(params['Delta0'])
    gamma = float(params['gamma'])
    alpha = float(params['alpha'])
    Ln = int(params['Ln']) # normal metal length
    Lb = int(params['Lb']) # barrier length
    Ls = int(params['Ls']) # superconductor length
    barrier_l = float(params['barrier0'])
    V0 = float(params['V0'])
    points = 100 # int(params['Upoints'])

    totlen = Ln + Lb + Ls

    # Load other parameter and phase arrays
    pdi_arr = hp.np_load_wrapped("pdi_data", dirname)
    params_list = np.load(Path(dirname, 'params_list.npy'))
    Vdisx = params['Vdisx'] * V0 * 0

    # ==========================================
    # Cell Block 3: Select a Topological Point
    # ==========================================
    # Filter the PDI values using the helper function
    filtered_pdi = hp.filter_pdi(pdi_arr[:, 2])
    topological_indices = np.where(filtered_pdi > 0.9)[0]

    # Pick the first topological point; if none are > 0.9, pick the one with the maximum PDI value
    if len(topological_indices) > 0:
        test_idx = topological_indices[0]
    else:
        test_idx = np.argmax(filtered_pdi)

    # In pdi_arr: column 0 is mu, column 1 is V_z
    mu_val =  0.0   #pdi_arr[test_idx, 0]
    V_z_val =  0.2    #pdi_arr[test_idx, 1]

    print(f"Selected Point Index {test_idx}:")
    print(f"mu  = {mu_val:.6f} meV")
    print(f"V_z = {V_z_val:.6f} meV")
    print(f"PDI = {filtered_pdi[test_idx]:.4f}")

    # File path where calculated data is saved / loaded
    data_file_path = Path(dirname, "bias_barrier_sweep_data.npz")

    # ==========================================
    # Cell Block 4: Conductance Matrix Sweep (Parallel)
    # ==========================================
    if just_plot and data_file_path.exists():
        print(f"'just_plot' is set to True. Loading sweep data from {data_file_path}...")
        saved_data = np.load(data_file_path)
        G_LL = saved_data['G_LL']
        G_RR = saved_data['G_RR']
        G_LR = saved_data['G_LR']
        G_RL = saved_data['G_RL']
        energies = saved_data['energies']
        barrier_r_vals = saved_data['barrier_r_vals']
        num_barriers = len(barrier_r_vals)
        num_energies = len(energies)
        print("Data loaded successfully. Skipping parallel calculation.")
    else:
        if just_plot:
            print(f"Warning: 'just_plot' was set to True but '{data_file_path}' was not found. Running sweep instead.")

        num_barriers = 50      # Sweep 50 barrier values for a detailed heatmap
        num_energies = 51      # Sweep 51 energy values (matching original simulation density)
        energies = np.linspace(-0.5, 0.5, num_energies)
        barrier_r_vals = np.linspace(barrier_l, 70 * barrier_l, num_barriers)

        G_LL = np.zeros((num_barriers, num_energies))
        G_RR = np.zeros((num_barriers, num_energies))
        G_LR = np.zeros((num_barriers, num_energies))
        G_RL = np.zeros((num_barriers, num_energies))

        # We default to CPU solver, or use GPU if cupy is set up
        solver = 'cpu'

        print(f"Running parallel sweep over {num_barriers} barrier asymmetry values and {num_energies} energy values...")
        
        # Create the partial worker function binding all static arguments
        worker_func = partial(
            worker_sweep_step,
            t=t, mu_val=mu_val, mu_n=mu_n, gamma=gamma, Delta0=Delta0,
            V_z_val=V_z_val, alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls,
            mu_leads=mu_leads, barrier_l=barrier_l, Vdisx=Vdisx,
            energies=energies, solver=solver
        )
        
        tasks = list(enumerate(barrier_r_vals))
        num_cores = min(mp.cpu_count(), len(barrier_r_vals))
        print(f"Using multiprocessing with {num_cores} workers...")
        
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(worker_func, tasks)
            
        # Reassemble results
        for i, row_LL, row_RR, row_RL, row_LR in results:
            G_LL[i, :] = row_LL
            G_RR[i, :] = row_RR
            G_RL[i, :] = row_RL
            G_LR[i, :] = row_LR

        print("Sweep completed!")
        
        # Save calculated data to file
        np.savez(
            data_file_path,
            G_LL=G_LL,
            G_RR=G_RR,
            G_LR=G_LR,
            G_RL=G_RL,
            energies=energies,
            barrier_r_vals=barrier_r_vals
        )
        print(f"Sweep data saved to: {data_file_path}")

    # ==========================================
    # Cell Block 5: Matplotlib Static Visualization (Raw Conductance)
    # ==========================================
    plt.figure(figsize=(14, 12))
    y_axis_ratios = barrier_r_vals / barrier_l

    # Subplot 1: G_LL (Raw)
    plt.subplot(2, 2, 1)
    plt.pcolormesh(energies, y_axis_ratios, G_LL, shading='auto', cmap='inferno')
    plt.colorbar(label='Conductance ($2e^2/h$)')
    plt.title('Local Conductance Left ($G_{LL}$)')
    plt.xlabel('Bias Voltage / Energy (meV)')
    plt.ylabel('Barrier Asymmetry ($U_R / U_L$)')

    # Subplot 2: G_RR (Raw)
    plt.subplot(2, 2, 2)
    plt.pcolormesh(energies, y_axis_ratios, G_RR, shading='auto', cmap='inferno')
    plt.colorbar(label='Conductance ($2e^2/h$)')
    plt.title('Local Conductance Right ($G_{RR}$)')
    plt.xlabel('Bias Voltage / Energy (meV)')
    plt.ylabel('Barrier Asymmetry ($U_R / U_L$)')

    # Subplot 3: G_LR (Raw)
    plt.subplot(2, 2, 3)
    lr_max = max(abs(G_LR.min()), abs(G_LR.max()))
    if lr_max == 0: lr_max = 1.0
    plt.pcolormesh(energies, y_axis_ratios, G_LR, shading='auto', cmap='RdBu_r', vmin=-lr_max, vmax=lr_max)
    plt.colorbar(label='Conductance ($2e^2/h$)')
    plt.title('Non-Local Conductance ($G_{LR}$)')
    plt.xlabel('Bias Voltage / Energy (meV)')
    plt.ylabel('Barrier Asymmetry ($U_R / U_L$)')

    # Subplot 4: G_RL (Raw)
    plt.subplot(2, 2, 4)
    rl_max = max(abs(G_RL.min()), abs(G_RL.max()))
    if rl_max == 0: rl_max = 1.0
    plt.pcolormesh(energies, y_axis_ratios, G_RL, shading='auto', cmap='RdBu_r', vmin=-rl_max, vmax=rl_max)
    plt.colorbar(label='Conductance ($2e^2/h$)')
    plt.title('Non-Local Conductance ($G_{RL}$)')
    plt.xlabel('Bias Voltage / Energy (meV)')
    plt.ylabel('Barrier Asymmetry ($U_R / U_L$)')

    plt.suptitle(f"Conductance Matrix Sweeps (mu={mu_val:.4f}, Vz={V_z_val:.4f})", fontsize=16)
    plt.tight_layout()

    static_plot_path = Path(plot_dir, 'bias_barrier_sweep_matplotlib.png')
    plt.savefig(static_plot_path, dpi=300)
    print(f"Matplotlib static plot saved to: {static_plot_path}")
    plt.close()

    # ==========================================
    # Cell Block 6: Calculate & Plot Correlation vs. Bias Voltage
    # ==========================================
    print("Calculating and plotting conductance correlations...")
    plot_conductance_correlation(energies, G_LL, G_RR, G_LR, G_RL, plot_dir)

    print("All tasks finished successfully.")
