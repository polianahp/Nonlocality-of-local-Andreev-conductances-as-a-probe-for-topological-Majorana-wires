# %%
#The code below was developed by Rodrigo A. Dourado, a PhD student at the University of Sao Paulo at São Carlos co-supervised 
#by Prof. J. Carlos Egues and Dr. Poliana H. Penteado. The results obtained via this code were also independently checked by additional 
#Mathematica and Python codes for particular cases developed by Dr Penteado.
#This code is not for distribution. 

# %%
#import os
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  

# %%


# %%
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from scipy import linalg as LA
import random
import numpy.matlib
import kwant
import tinyarray
import multiprocessing as mp
import os
from tqdm import tqdm
import helpers as hp
from pathlib import Path
from config import PathConfigs
import scipy.sparse.linalg as sla
import multiprocessing as mp
from functools import partial
from scipy.signal import find_peaks
from IPython.display import display, HTML
from main_parallel import worker_pdi_step, worker_simulation_step
display(HTML('<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'))



#pauli matrices
sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])

# %%


# %%
def get_psiM_density(syst, k=2):
    """  
    Calculates the Majorana mode densities rho_M1 (Left) and rho_M2 (Right)
    at zero energy.
    
    Parameters:
    - syst: The finalized kwant system (syst_closed).
    - k: Number of eigenvalues to solve for (default 2 for the lowest pair).
    
    Returns:
    - rho_M1: Spatial density of the first Majorana mode (Left).
    - rho_M2: Spatial density of the second Majorana mode (Right).
    - energies: The eigenvalues found (for verification).
    """
    # 1. Access Hamiltonian from Kwant
    # sparse=True is essential for large systems
    ham = syst.hamiltonian_submatrix(sparse=True) 
    
    # 2. Diagonalize to find states near Zero Energy (sigma=0)
    # k=2 guarantees we find the lowest pair (E ~ +0 and E ~ -0)
    try:
        evals, evecs = sla.eigsh(ham, k=k, sigma=0, which='LM')
    except:
        # Fallback for small systems where sparse solvers might fail
        evals, evecs = np.linalg.eigh(ham.toarray())
        
    # 3. Sort by Energy (Real values)
    # We want the lowest POSITIVE energy state and its NEGATIVE partner.
    # eigsh usually returns unsorted or sorted by magnitude. We sort by value.
    sort_idx = np.argsort(evals)
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]
    
    # Identify the index of the first positive energy state
    # In a particle-hole symmetric system with 2*N states:
    # indices 0 to N-1 are negative, N to 2N-1 are positive.
    # For k retrieved states around 0, the one just above the middle is the lowest positive.
    mid_idx = len(evals) // 2
    
    # Lowest positive state (psi_+)
    psi_plus = evecs[:, mid_idx]
    # Corresponding negative state (psi_-)
    psi_minus = evecs[:, mid_idx - 1]
    
    # 4. Phase Correction (Standardize phases)
    # We enforce a phase such that the first component is real/positive to align them
    # This is similar to the 'ix' logic in the Mathematica script

    
    
    #phase_plus = np.conj(psi_plus[0]) / np.abs(psi_plus[0] + 1e-20)
    #phase_minus = np.conj(psi_minus[0]) / np.abs(psi_minus[0] + 1e-20)
    
    pmax_idx = np.argmax(psi_plus)
    mmax_idx = np.argmax(psi_minus)

    phase_plus_max = np.conj(psi_plus[pmax_idx]) / np.abs(psi_plus[pmax_idx] + 1e-20)
    phase_minus_max = np.conj(psi_minus[mmax_idx]) / np.abs(psi_minus[mmax_idx] + 1e-20)
    
    psi_plus = psi_plus * phase_plus_max
    psi_minus = psi_minus * phase_minus_max
    
    # 5. Construct Majorana Basis
    # gamma_1 = (psi_+ + psi_-) / sqrt(2)  (Usually Left)
    # gamma_2 = (psi_+ - psi_-) / sqrt(2)  (Usually Right, times i)
    gamma_1 = (psi_plus + psi_minus) / np.sqrt(2)
    gamma_2 = (psi_plus - psi_minus) / np.sqrt(2)
    
    # 6. Calculate Site Densities
    # Kwant stores wavefunctions as a flat array [site1_orb1, site1_orb2, ..., site2_orb1, ...]
    # Your system has norbs=4 (e_up, e_dn, h_dn, h_up)
    
    n_sites = len(gamma_1) // 4
    
    # Reshape to (Sites, Orbitals)
    g1_reshaped = gamma_1.reshape((n_sites, 4))
    g2_reshaped = gamma_2.reshape((n_sites, 4))
    
    # Sum over orbitals (spin/particle-hole) to get density per site
    rho_M1 = np.sum(np.abs(g1_reshaped)**2, axis=1)
    rho_M2 = np.sum(np.abs(g2_reshaped)**2, axis=1)
    
    return rho_M1, rho_M2, evals




def get_psiM_density2(syst, k=2):
    """  
    Calculates the Majorana mode densities rho_M1 (Left) and rho_M2 (Right)
    at zero energy using verified numerical phase alignment.
    
    Parameters:
    - syst: The finalized kwant system (syst_closed).
    - k: Number of eigenvalues to solve for (default 2 for the lowest pair).
    
    Returns:
    - rho_M1: Spatial density of the first Majorana mode (Left).
    - rho_M2: Spatial density of the second Majorana mode (Right).
    - energies: The eigenvalues found (for verification).
    """
    # 1. Access Hamiltonian from Kwant
    ham = syst.hamiltonian_submatrix(sparse=True) 
    
    # 2. Diagonalize to find states near Zero Energy (sigma=0)
    try:
        evals, evecs = sla.eigsh(ham, k=k, sigma=0, which='LM')
    except:
        evals, evecs = np.linalg.eigh(ham.toarray())
        
    # 3. Sort by Energy
    sort_idx = np.argsort(evals)
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]
    
    mid_idx = len(evals) // 2
    psi_plus = evecs[:, mid_idx] # Lowest positive energy state
    
    # 4. Rigorous Phase Correction (Mitigating Numerical Uncertainty)
    # Find the index with the maximum absolute probability amplitude. 
    # This ensures we extract a phase from a physically meaningful signal, 
    # not floating-point round-off error near machine epsilon.
    max_idx = np.argmax(np.abs(psi_plus))
    
    # Extract the phase of this dominant component and conjugate it
    stable_phase = np.conj(psi_plus[max_idx]) / (np.abs(psi_plus[max_idx]) + 1e-20)
    
    # Rotate psi_plus so its dominant component is strictly real and positive
    psi_plus_aligned = psi_plus * stable_phase
    
    # 5. Apply Particle-Hole Symmetry (PHS) explicitly
    # We construct the exact PHS conjugate state (Xi * psi_plus) instead of 
    # relying on the numerical solver's unsynchronized psi_minus.
    n_sites = len(psi_plus_aligned) // 4
    psi_plus_reshaped = psi_plus_aligned.reshape((n_sites, 4))
    
    psi_minus_exact = np.zeros_like(psi_plus_aligned, dtype=complex)
    psi_minus_reshaped = psi_minus_exact.reshape((n_sites, 4))
    
    # Apply Xi = U_C * K
    # Assuming standard BdG basis where U_C = tau_x (flips electron/hole blocks).
    # WARNING: If your basis includes complex signs in the hole block 
    # (e.g., basis = (u_up, u_dn, v_dn, -v_up)), you must add the corresponding 
    # negative signs or sigma matrices to the transformation below!
    for i in range(n_sites):
        e_comps = psi_plus_reshaped[i, 0:2] # Electron components
        h_comps = psi_plus_reshaped[i, 2:4] # Hole components
        
        # Complex conjugation (K) and unitary flip (tau_x)
        psi_minus_reshaped[i, 0:2] = np.conj(h_comps)
        psi_minus_reshaped[i, 2:4] = np.conj(e_comps)
        
    psi_minus_exact = psi_minus_reshaped.flatten()
    
    # 6. Construct Majorana Basis
    gamma_1 = (psi_plus_aligned + psi_minus_exact) / np.sqrt(2)
    gamma_2 = 1j * (psi_plus_aligned - psi_minus_exact) / np.sqrt(2)
    
    # 7. Calculate Site Densities
    g1_reshaped = gamma_1.reshape((n_sites, 4))
    g2_reshaped = gamma_2.reshape((n_sites, 4))
    
    rho_M1 = np.sum(np.abs(g1_reshaped)**2, axis=1)
    rho_M2 = np.sum(np.abs(g2_reshaped)**2, axis=1)
    
    return rho_M1, rho_M2, evals


def calc_MZM_separation(rho_M1, rho_M2, sep_thresh = 0.8):
    #checks if mzm wave functions are separated. Condition is that some percentage of the total weight
    #for each mzm should be on a separate side of "idx" when integrated. This is a precondition before calculating
    #how well the mzms
    tot_rho = rho_M1 + rho_M2
    
    m1_pct = 0
    m2_pct = 0
    
    idx = 3
    while (m1_pct < sep_thresh or m2_pct < sep_thresh):
        
        if idx >= len(rho_M1):
            return [None, None, None]
        
        m1sum = np.trapz(rho_M1[:idx])
        m2sum = np.trapz(rho_M2[idx:])
        
        m1_pct = m1sum
        m2_pct = m2sum
        idx +=1
        
    return [idx, m1_pct, m2_pct]
        
        
        
        
        
        

# %%
def calculate_local_mp(syst_closed):
    """
    Calculates the Local Majorana Polarization (Mj) for each site j.
    Formula: M_j = (2 * u * v) / (u^2 + v^2)
    Adapted for spinful wire: Sums spin contributions per site.
    
    Returns:
    - M_profile: Array of length L (number of sites), containing Mj for each site.
    - energy_0: The energy of the analyzed mode.
    """
    ham = syst_closed.hamiltonian_submatrix(sparse=True)
    try:
        evals, evecs = sla.eigsh(ham, k=2, sigma=0, which='LM')
    except:
        evals, evecs = np.linalg.eigh(ham.toarray())
        
    idx = np.argsort(np.abs(evals))
    psi = evecs[:, idx[0]]
    energy_0 = evals[idx[0]]

    
    n_sites = len(psi) // 4
    psi_sites = psi.reshape(n_sites, 4)
    
    
    # Basis (u_up, u_down, v_down, -v_up)
    u_up   = psi_sites[:, 0]
    u_down = psi_sites[:, 1]
    v_down = psi_sites[:, 2]
    v_up   = -psi_sites[:, 3]
    

    overlap = u_up * np.conj(v_up) + u_down * np.conj(v_down)
    numerator = 2 * np.real(overlap)
    
    denominator = (np.abs(u_up)**2 + np.abs(u_down)**2 + 
                   np.abs(v_down)**2 + np.abs(v_up)**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        M_profile = numerator / denominator
        
    M_profile = np.nan_to_num(M_profile) # Replace NaNs with 0
    
    return M_profile, energy_0

# %%
def calculate_gamma_squared(syst_closed, k=0):
    """
    Calculates the 'Squared Majorana Operator' value for the k-th energy mode.
    Formula: gamma^2 = Sum_j (u_j * v_j)
    
    Parameters:
    - syst_closed: finalized system (without leads attached).
    - k: The index of the eigenstate
    """
    ham = syst_closed.hamiltonian_submatrix(sparse=True)
    try:
        evals, evecs = sla.eigsh(ham, k=k+4, sigma=0, which='LM')
    except:
        evals, evecs = np.linalg.eigh(ham.toarray())
        
    idx = np.argsort(np.abs(evals))
    psi = evecs[:, idx[k]]  # Wavefunction of the k-th mode
    
    #basis order: (e_up, e_down, h_down, h_up) per site
    
    n_tot = len(psi)
    u_vec = psi[:n_tot//2]  
    v_vec = psi[n_tot//2:]  
    
    gamma_sq = np.sum(u_vec * v_vec)
    
    return gamma_sq

# %%
#dirname = Path(PathConfigs.DATA/"dis_realizations"/"disorder_realization_1_results")
dirname = Path(PathConfigs.DATA/"Tdis_pfaff2")

os.makedirs(Path(dirname), exist_ok=True)

#params = np.load(Path(dir/"all_params.npz"))
params = np.load(Path(dirname/"all_params.npz"), allow_pickle=True)
pdi_data = np.load(Path(dirname/"pdi_data.npy"), allow_pickle=True)
mu_n = float(params['mu_n'])
t = float(params['t'])
mu_leads =float(params['mu_leads'])
Delta0 = float(params['Delta0'])
gamma = float(params['gamma'])
alpha = float(params['alpha'])
Ln = int(params['Ln']) # normal metal length
Lb = int(params['Lb']) #barrier length
Ls = int(params['Ls']) #super conductor length
barrier_l = float(params['barrier0'])
V0 = float(params['V0'])
points = 100#int(params['Upoints'])

totlen = Ln + Lb +Ls 
#V_z = 1.0 * V_c # To generate Fig 2a V_z should be varied. For the cyan line in Fig 2, for example, V_z = 1.203 * V_c  
# 0.4864

#V_c = np.sqrt(mu**2 + Delta**2)
V_z = 0.7548387
Vz_str = str(V_z).replace('.','_')

mu = 2.432432  
mu_str = str(mu).replace('.','_')
barrier_r = np.linspace(-20*barrier_l, 40*barrier_l, points)  #Varying the right barrier U_R

num_engs = 101 
energies = np.linspace(-0.5, 0.5, num_engs)
#energies = np.linspace(-0.15, 0.15, 101)


Vdisx = params['Vdisx']  * V0


paramd = {name:params[name] for name in params.files}
del paramd['Vdisx']
del paramd['energies']
del paramd['barrier_arr']
del paramd['mu_var']


pdicalc = hp.PDICalculator(t, alpha, gamma, Ls, params['Vdisx'], V0)


vrng = 100
Vzs = np.linspace(0,1.4,vrng)

# %%
Nx = len(Vdisx)
plt.figure(figsize=(10, 2))
plt.plot(np.arange(Nx), Vdisx, color='royalblue')
plt.xlabel("x (lattice sites)")
plt.ylabel("V_xd")
plt.title("Projected Disorder Potential $V_{xd}(x)$")
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()

# mean and variance
DeltaVd = np.mean(Vdisx)
VarVd = np.mean(Vdisx**2)
print(f"ΔVd = {DeltaVd:.4f}")
print(f"sqrt[<Vxd²>] = {np.sqrt(VarVd):.4f}")
plt.savefig(Path(dirname,'Plots',"disorder.png"))

# %%
G_matrix_barrier = np.zeros(shape=(points, 2, 2))
conductance_left = np.zeros(points)
conductance_right = np.zeros(points)
for k in tqdm(range(points)):
    #print(f"running point: {k}/{points}")
    syst = hp.build_system(
        t=t, 
        mu=mu, 
        mu_n=mu_n, 
        Delta0=Delta0,
        gamma=gamma, 
        V_z=V_z, 
        alpha=alpha, 
        Ln=Ln, 
        Lb=Lb, 
        Ls=Ls, 
        mu_leads=mu_leads, 
        barrier_l=barrier_l, 
        barrier_r=barrier_r[k],
        Vdisx = Vdisx,
        a=1
        )
    
    cL, cR = hp.calc_conductance(syst, energy = 0.0, return_smatrix = False)
    conductance_left[k] = cL
    conductance_right[k] =cR
    
    

# %%
Lconductance_left = np.zeros(points)
Lconductance_right = np.zeros(points)
for k in tqdm(range(points)):
    #print(f"running point: {k}/{points}")
    syst = hp.build_system(
        t=t, 
        mu=mu, 
        mu_n=mu_n, 
        Delta0=Delta0,
        gamma=gamma, 
        V_z=V_z, 
        alpha=alpha, 
        Ln=Ln, 
        Lb=Lb, 
        Ls=Ls, 
        mu_leads=mu_leads, 
        barrier_l=barrier_r[k],
        barrier_r=barrier_l, 
        Vdisx = Vdisx,
        a=1
        )
    
    cL, cR = hp.calc_conductance(syst, energy = 0.0, return_smatrix = False)
    Lconductance_left[k] = cL
    Lconductance_right[k] =cR

# %%
# Rebuild nominal system before calculating dI/dV spectrum
syst = hp.build_system(
    t=t, 
    mu=mu, 
    mu_n=mu_n, 
    Delta0=Delta0,
    gamma=gamma, 
    V_z=V_z, 
    alpha=alpha, 
    Ln=Ln, 
    Lb=Lb, 
    Ls=Ls, 
    mu_leads=mu_leads, 
    barrier_l=barrier_l, 
    barrier_r=barrier_l, 
    Vdisx=Vdisx,
    a=1
)


dIdV_left, dIdV_right, dIdV_LR, dIdV_RL, ldos  = hp.calc_dIdV(syst, energies)


# %%
if False:

    rho_dat = np.zeros(shape=(vrng,300))

    for i, vz in enumerate(tqdm(Vzs)):
        scl = hp.build_system_closed(t, mu, gamma, Delta0, vz, alpha, Ls, Vdisx, a=1)
        ham = scl.hamiltonian_submatrix(sparse=True) 

        try:
            evals, evecs = sla.eigsh(ham, sigma=0, which='LM')
        except:
            evals, evecs = np.linalg.eigh(ham.toarray())
            
        rho_M1, rho_M2, evals = hp.get_psiM_density(evals, evecs)
        rho_dat[i,:] = rho_M1 + rho_M2

# %%
if False:
    import matplotlib.pyplot as plt

    # 1. Create the figure
    plt.figure(figsize=(10, 6), dpi=100)

    # 2. Plot the heatmap
    # We transpose rho_dat.T so x-axis = Vz and y-axis = Site
    # extent defines the physical coordinates: [left, right, bottom, top]
    # origin='lower' ensures site 0 is at the bottom
    im = plt.imshow(rho_dat.T, 
                    extent=[Vzs[0], Vzs[-1], 0, rho_dat.shape[1]], 
                    aspect='auto', 
                    origin='lower', 
                    cmap='magma')

    # 3. Add labels and styling
    plt.colorbar(im, label='Majorana Density (rho_M1 + rho_M2)')
    plt.xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=12)
    plt.ylabel("Spatial Site Index", fontsize=12)
    plt.title(r"Majorana Density Heatmap ($\mu = {}$ meV)".format(mu), fontsize=14)

    # Optional: Add a vertical line at your target_vz if needed
    # plt.axvline(target_vz, color='white', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# %%
Data_dir = Path("/home/pseudonym/code/Nonlocal_Conductance/Nonlocality-of-local-Andreev-conductances-as-a-probe-for-topological-Majorana-wires/Data")
save_dir = Path(Data_dir/"New_Results")
os.makedirs(save_dir, exist_ok=True)

# %%
scl = hp.build_system_closed(t, mu, gamma, Delta0, V_z, alpha, Ls, Vdisx, a=1)
rho_M1, rho_M2, evals = get_psiM_density(scl)
hm = scl.hamiltonian_submatrix()

# %%

# 1. Instantiate the PDICalculator using the parameters defined in the notebook
calculator = hp.PDICalculator(t, alpha, gamma, Ls, params['Vdisx'], V0)

# 2. Get the full Hamiltonian matrix from the PDI calculator
H_full = calculator.get_full_Hamiltonian(V_z, mu)

# 3. Calculate eigenvalues and eigenvectors from this Hamiltonian
evalss, evecss = np.linalg.eigh(H_full)

# 4. Use the imported hp.get_psiM_density function to get the densities
rho_M11, rho_M22, evals = hp.get_psiM_density_excited(evalss, evecss, offset = 0)

# %%
plt.figure(figsize=(9.5, 7))
# Notebook uses 'ass' = a0 * 10^-4.

plt.plot(rho_M11, label='Majorana Left (M1)', color='cyan')
plt.plot(rho_M22, label='Majorana Right (M2)', color='orange')
plt.title(f"Majorana Modes (mu={mu}, Vz={V_z})")
plt.xlabel("Site Index")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
def calc_localization(rho_M1, rho_M2, weight_thresh = 0.85):
    step = 0.00001
    n = 0
    
    totrho = rho_M1 + rho_M2
    totsum = sum(totrho)
    wpct = 0.0
    rgn_pct = 0.0
    while wpct < weight_thresh:
        thresh = np.max(totrho) - step * n
        idx = np.where(totrho>=thresh)[0]
        wpct = np.sum(totrho[idx])/totsum
        rgn_pct = (np.max(idx) - np.min(idx))/len(totrho)
        n +=1
        
    return rgn_pct


    

# %%
totrho = rho_M1 + rho_M2
step = 0.001
thresh = np.max(totrho) - step * 35

idx = np.where(totrho>=thresh)[0]
connected_region_lengh = np.max(idx) - np.min(idx)

weight_pct = np.sum(totrho[idx])/np.sum(totrho)

print(f'pct: {weight_pct} region_pct: {connected_region_lengh/len(totrho)}')
#plt.hlines(thresh, xmin = 0, xmax = 300)
#plt.plot(totrho)



# %%
qs = [30]
#Z = Delta0 / (Delta0 + gamma)
Z = 1

Qarr = []
for q in qs:
    Q_nu = hp.calculate_pdi(Z*t, Z*alpha, Z*gamma, Ls, Vdisx, 1, Z*V_z, Z*mu, q)
    Qarr.append(Q_nu)
    

#1.012191696936547

Qarr[0] 



# %%
hp.cal_pfaffian_invariant(t,alpha,gamma,Delta0,Ls, Vdisx, 1, V_z, mu, delta_N=0)

# %%

def generate_point_path(pdi_data, N, resl, mu_start, mu_end, Vz_start, Vz_end):
    pdi_params = pdi_data[:, 0:2]

    diff_vec = np.array([mu_end - mu_start, Vz_end - Vz_start])
    total_distance = np.linalg.norm(diff_vec)

    unit_vec = diff_vec / total_distance
    step_vec = resl * unit_vec 

    num_pts = int(np.floor(total_distance / resl))

    start_vec = np.array([mu_start, Vz_start])
    pts = np.asarray([start_vec + (n * step_vec) for n in range(num_pts)])

    closest_indices = []

    for i in range(num_pts):
        tst = pts[i, :]
        
        is_close_mask = np.all(np.isclose(tst, pdi_params, atol=resl), axis=1)
        matched_indices = np.where(is_close_mask)[0]
        
        if len(matched_indices) > 0:
            matched_pdi_params = pdi_params[matched_indices]
            distances = np.linalg.norm(matched_pdi_params - tst, axis=1)
            closest_subset_index = np.argmin(distances)
            closest_original_index = matched_indices[closest_subset_index]
            closest_indices.append(closest_original_index)
        else:
            closest_indices.append(-1)

    closest_indices = np.array(closest_indices)

    valid_indices = closest_indices[closest_indices != -1]
    unique_indices = np.unique(valid_indices)
    unique_points = pdi_params[unique_indices]

    dist_from_start = np.linalg.norm(unique_points - start_vec, axis=1)
    sort_order = np.argsort(dist_from_start)

    sorted_unique_indices = unique_indices[sort_order]
    sorted_unique_points = unique_points[sort_order]

    num_unique = len(sorted_unique_points)

    if num_unique == 0:
        return np.array([]), np.array([])

    if N >= num_unique:
        sampled_points = sorted_unique_points
        sampled_indices = sorted_unique_indices
    else:
        sample_idx = np.round(np.linspace(0, num_unique - 1, N)).astype(int)
        sampled_points = sorted_unique_points[sample_idx]
        sampled_indices = sorted_unique_indices[sample_idx]

    return sampled_points





# %%

kvals = 12

pts = generate_point_path(pdi_data, 100, 0.02, mu_start = mu, mu_end = mu, Vz_start = 0, Vz_end = 1.4)
evals = np.zeros(shape = (len(pts[:,1]), kvals))

for i, p in enumerate(tqdm(pts)):
    mu, vz = p
    scl = hp.build_system_closed(t, mu, gamma, Delta0, vz, alpha, Ls, Vdisx, a=1)
    evals[i,:] = hp.calc_spectrum(scl, k = kvals)

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set targets based on the current calculation block
target_mu = mu 
target_vz = V_z  # The value where the vertical line will be drawn

# Your data is already organized! No need to filter or sort.
plot_vz = pts[:,1]
plot_spectra = evals

# 1. Create the Figure
fig, ax1 = plt.subplots(figsize=(9, 6), dpi=100)

# Calculate the middle indices to isolate the states closest to E=0
# For kvals=12, mid_idx is 6. So the red states are 5 and 6.
mid_idx = kvals // 2 

# 2. Plot the bulk states in standard blue
# States below E=0
ax1.plot(plot_vz, plot_spectra[:, :mid_idx-1], color='royalblue', alpha=0.8, linewidth=1.5)
# States above E=0
ax1.plot(plot_vz, plot_spectra[:, mid_idx+1:], color='royalblue', alpha=0.8, linewidth=1.5)

# 3. Plot the two states closest to E=0 in red
ax1.plot(plot_vz, plot_spectra[:, mid_idx-1:mid_idx+1], color='red', alpha=0.9, linewidth=2.0, label=rf"$\mu = {target_mu}$ meV")

# 4. Add Zero-energy and Target Vz reference lines
ax1.axhline(0, color='black', linestyle='--', linewidth=1.2)
ax1.axvline(target_vz, color='black', linestyle='-', linewidth=1.5, label=f"Target $V_z = {target_vz}$")

# 5. Formatting
ax1.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=14)
ax1.set_ylabel("Energy (meV)", fontsize=14)
ax1.set_title(f"Low Energy Spectra ($\mu = {target_mu}$ meV)", fontsize=16)
ax1.grid(True, linestyle=':', alpha=0.6)

# Handle legend deduplication
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)

# Clean up layout
plt.tight_layout()

# Save logic (Uncomment and set plot_dir if you need to save to disk)
# plot_dir = Path("Plots")
# plot_dir.mkdir(exist_ok=True)
# fname = f"Spectra_mu_{str(target_mu).replace('.','_')}.png"
# fig.savefig(plot_dir / fname, dpi=300, bbox_inches='tight')

# Show the plot in the notebook
plt.show()

# %%
import plotly.graph_objects as go

# 1. Re-defining the specific parameters from your original list
params_to_show = {
    "t": f"{t:.2f}",
    "mu": mu,
    "mu_n": mu_n,
    "mu_leads": f"{mu_leads:.2f}",
    "Delta0": Delta0,
    "gamma": gamma,
    "alpha": alpha,
    "Ln": Ln,
    "Lb": Lb,
    "Ls": Ls,
    "V_z": V_z,
    "barrier_l": barrier_l,
    "V0 (meV)": f"{V0:.2f}"
}

# Construct the text string
text_str = '<br>'.join([f"<b>{key}</b>: {val}" for key, val in params_to_show.items()])

fig = go.Figure()

# 2. Add Traces
fig.add_trace(go.Scatter(
    x=energies, y=dIdV_right, mode='lines', name='Right dI/dV',
    line=dict(width=2, color='darkorange'),
    hovertemplate='Energy: %{x:.4f}<br>dI/dV: %{y:.6f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=energies, y=dIdV_left, mode='lines', name='Left dI/dV',
    line=dict(width=2, color='royalblue'),
    hovertemplate='Energy: %{x:.4f}<br>dI/dV: %{y:.6f}<extra></extra>'
))

# 3. Final Tightened Layout
fig.update_layout(
    width=780,   # Narrowed width to eliminate side-whitespace
    height=600,
    template="plotly_white",
    hovermode="x unified",
    xaxis_title="Energy (meV)",
    yaxis_title="Differential Conductance (dI/dV)",
    
    # Keep legend top-left inside to avoid the parameter box
    legend=dict(
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=0.02,
        bgcolor="rgba(255, 255, 255, 0.5)"
    ),

    # Trimmed right margin significantly
    margin=dict(l=60, r=130, t=40, b=60), 
    
    annotations=[
        dict(
            x=1.02, # Sits right on the edge of the plot
            y=1,
            xref="paper",
            yref="paper",
            text=text_str,
            showarrow=False,
            align="left",
            xanchor="left", # Anchor to the left side of the box
            bgcolor="rgba(245, 222, 179, 0.5)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
    ]
)

fig.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

fig, axes = plt.subplots(1,1, figsize=(5,3.5))
fig.subplots_adjust(0,0,1,1)

x_data = barrier_r / barrier_l
x_range = (np.min(x_data), np.max(x_data))

lw = 3.5

normed_GL = conductance_left / conductance_left[0]
normed_GR = conductance_right / conductance_right[0]
corr = np.dot(normed_GR, normed_GL) / (np.linalg.norm(normed_GR) * np.linalg.norm(normed_GL))
corr_new = hp.calc_invariant_metric(normed_GL, normed_GR)

print(f"New Correlation: {corr_new}")
print(f"old Correlation: {corr}")

axes.plot(x_data, normed_GL, color="green", linewidth=lw)
axes.set_xlim(x_range)
axes.set_yticks([0, 1])
axes.set_yticklabels([0, 1], fontsize=20)
axes.set_xlabel(r"$U_{R}/U_{L} $", fontsize=20)
axes.set_ylabel(r"$G_{LL}/G_{LL, sym}$", fontsize=22)
axes.yaxis.set_label_coords(-0.01, 0.5)

fig.tight_layout()
fig.savefig(Path(save_dir / "Conductances_Left.png"))
plt.savefig(Path(dirname, 'Plots', f"Conductances_Left_mu{mu_str}_vz{Vz_str}.png"))

fig, axes = plt.subplots(1,1, figsize=(5,3.5))
fig.subplots_adjust(0,0,1,1)

axes.plot(x_data, normed_GR, color="green", linewidth=lw)
axes.set_xlim(x_range)
axes.set_yticks([0, 1])
axes.set_yticklabels([0, 1], fontsize=20)
axes.set_xlabel(r"$U_{R}/U_{L}$", fontsize=20)
axes.set_ylabel(r"$G_{RR}/G_{RR, sym}$", fontsize=22)
axes.yaxis.set_label_coords(-0.01, 0.5)

fig.tight_layout()
plt.savefig(Path(dirname, 'Plots', f"Conductances_Right_mu{mu_str}_vz{Vz_str}.png"))
fig.savefig(Path(save_dir / "Conductances_Right.png"))

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

fig, axes = plt.subplots(1,1, figsize=(5,3.5))
fig.subplots_adjust(0,0,1,1)

x_data = barrier_r / barrier_l
x_range = (np.min(x_data), np.max(x_data))

lw = 3.5

normed_GL = Lconductance_left / Lconductance_left[0]
normed_GR = Lconductance_right / Lconductance_right[0]
corr = np.dot(normed_GR, normed_GL) / (np.linalg.norm(normed_GR) * np.linalg.norm(normed_GL))
corr_new = hp.calc_invariant_metric(normed_GL, normed_GR)

print(f"New Correlation: {corr_new}")
print(f"old Correlation: {corr}")

axes.plot(x_data, normed_GL, color="green", linewidth=lw)
axes.set_xlim(x_range)
axes.set_yticks([0, 1])
axes.set_yticklabels([0, 1], fontsize=20)
axes.set_xlabel(r"$U_{L}/U_{R} $", fontsize=20)
axes.set_ylabel(r"$G_{LL}/G_{LL, sym}$", fontsize=22)
axes.yaxis.set_label_coords(-0.01, 0.5)

fig.tight_layout()
fig.savefig(Path(save_dir / "Conductances_Left.png"))
plt.savefig(Path(dirname, 'Plots', f"Conductances_Left_mu{mu_str}_vz{Vz_str}.png"))

fig, axes = plt.subplots(1,1, figsize=(5,3.5))
fig.subplots_adjust(0,0,1,1)

axes.plot(x_data, normed_GR, color="green", linewidth=lw)
axes.set_xlim(x_range)
axes.set_yticks([0, 1])
axes.set_yticklabels([0, 1], fontsize=20)
axes.set_xlabel(r"$U_{L}/U_{R}$", fontsize=20)
axes.set_ylabel(r"$G_{RR}/G_{RR, sym}$", fontsize=22)
axes.yaxis.set_label_coords(-0.01, 0.5)

fig.tight_layout()
plt.savefig(Path(dirname, 'Plots', f"Conductances_Right_mu{mu_str}_vz{Vz_str}.png"))
fig.savefig(Path(save_dir / "Conductances_Right.png"))


# %%
##Saving Data
#np.savez(Path(save_dir/"Data.npz"), 
#         conductance_right = conductance_right, 
#         conductance_left = conductance_left, 
#         dIdV_left = dIdV_left,
#         dIdV_right = dIdV_right,
#         energies = energies,
#         #ldos_per_site = ldos_per_site,
#         Vdisx = Vdisx,
#         G_matrix = G_matrix,
#         G_matrix_barrier = G_matrix_barrier
#         
#         )
#
#np.savez(Path(save_dir/"Parameters.npz"),
#         
#        t = t,
#        mu = mu,
#        mu_n = mu_n,
#        mu_leads = mu_leads,
#        Delta = Delta0,
#        gamma = gamma, 
#        alpha = alpha,
#        Ln = Ln,
#        Lb = Lb,
#        Ls = Ls,
#        V_z = V_z,
#        barrier_l = barrier_l,
#        points = points,
#        barrier_r = barrier_r,
#        #lambda_dis = lambda_dis,
#        V0 = V0,
#         )


