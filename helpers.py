from tqdm import tqdm
import kwant
import tinyarray
import numpy as np
import os
from config import PathConfigs
from pathlib import Path
import scipy.sparse.linalg as sla
from scipy.signal import find_peaks




#pauli matrices
sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])


######## Path Configuration Helpers
def get_data_path(file_name, subdirectory):
    #constructs output data paths based on path configuration in config.py
    # path constructed should be DATA/subdirectory/filename
    
    pth = Path(subdirectory/ file_name)
    return PathConfigs.DATA / subdirectory / file_name

def np_save_wrapped(arr, name, dirname):
    dir = f"{PathConfigs.DATA}/{dirname}"
    Path(f'{dir}').mkdir(parents=True, exist_ok=True)
    np.save(f"{dir}/{name}.npy", arr)

def np_savez_wrapped(name, dirname, **kwargs):
    dir = f"{PathConfigs.DATA}/{dirname}"
    Path(f'{dir}').mkdir(parents=True, exist_ok=True)
    np.savez(f"{dir}/{name}.npz", **kwargs)
    
def np_load_wrapped(filename, subdirectory):
    #wrapper for np.loadtxt to load from default data bath
    path = get_data_path(f"{filename}.npy", subdirectory)
    return np.load(path)
    
    
######## Calculation Helpers ########


def calc_conductance(syst, energy = 0.0, return_smatrix = False):
    smatrix = kwant.smatrix(syst, energy)
    
    R = 0
    A = 0
    for i in range(2):
        for j in range(2):
            R = R + smatrix.transmission((0, j), (0, i))
            A = A + smatrix.transmission((0, j+2), (0, i))
    cleft = 2 - R + A
    R = 0
    A = 0
    for i in range(2):
        for j in range(2):
            R = R + smatrix.transmission((1, j), (1, i))
            A = A + smatrix.transmission((1, j+2), (1, i))
    cright = 2 - R + A
    
    if return_smatrix:
        return cleft, cright, smatrix
    
    else:
        return cleft, cright
    
    
def calc_conductance_matrix(syst, eng):
    G_matrix = np.zeros(shape=(2, 2))
    smatrix = kwant.smatrix(syst, eng)
    Rl = 0
    Al = 0
    Rr = 0
    Ar = 0
    
    # Non-Local Transmissions
    T_RL = 0; TA_RL = 0 # Left -> Right
    T_LR = 0; TA_LR = 0 # Right -> Left
    
    for i in range(2):
        for j in range(2):
            Rl = Rl + smatrix.transmission((0, j), (0, i))
            Al = Al + smatrix.transmission((0, j+2), (0, i))

            Rr = Rr + smatrix.transmission((1, j), (1, i))
            Ar = Ar + smatrix.transmission((1, j+2), (1, i))
            
            T_RL  += smatrix.transmission((1, j),   (0, i)) # Normal
            TA_RL += smatrix.transmission((1, j+2), (0, i)) # Crossed Andreev
            T_LR  += smatrix.transmission((0, j),   (1, i)) # Normal
            TA_LR += smatrix.transmission((0, j+2), (1, i)) # Crossed Andreev
            
            
    G_matrix[0,0] = 2 - Rl + Al
    G_matrix[1,1] = 2 - Rr + Ar
    
    G_matrix[1,0] = TA_RL - T_RL # Current at Right (1) due to Left (0)
    G_matrix[0,1] = TA_LR - T_LR # Current at Left (0) due to Right (1)    
    
    return G_matrix


def calculate_gamma_squared(syst_closed, k=0):
    """
    Calculates the Squared Majorana Operator 
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


def calc_dIdV(syst, energies):
    num_engs = len(energies)
    
    num_orbitals = syst.graph.num_nodes * 4
    
    ldos = np.zeros(shape = (num_engs, num_orbitals))
    dIdV_left = np.zeros_like(energies)
    dIdV_right = np.zeros_like(energies)
    
    for k, eng in tqdm(enumerate(energies), total=num_engs, desc="Calculating dI/dV"):
        #print(f"running energy {k}/{num_engs}")
        eng = energies[k]
        smatrix = kwant.smatrix(syst, eng)
        R = 0
        A = 0
        for i in range(2):
            for j in range(2):
                R = R + smatrix.transmission((0, j), (0, i))
                A = A + smatrix.transmission((0, j+2), (0, i))
        dIdV_left[k] = 2 - R + A
        R = 0
        A = 0
        for i in range(2):
            for j in range(2):
                R = R + smatrix.transmission((1, j), (1, i))
                A = A + smatrix.transmission((1, j+2), (1, i))
        dIdV_right[k] = 2 - R + A
        
        ldos[k,:] = kwant.ldos(syst, eng)
        
    return dIdV_left, dIdV_right, ldos 


def detect_peaks_oldold(conductance_arr, energy_mesh, prominence=0.01):
    """
    Detects peaks in the differential conductance array with priority for Majorana physics.
    
    Logic:
    - If a peak is found at zero (within tolerance), return (1, 0).
    - If split, return the distance between the two peaks closest to zero (hybridization splitting).
    
    Parameters:
    - conductance_arr: 1D array of conductance values.
    - energy_mesh: 1D array of energy values.
    - prominence: The required prominence of peaks.
    
    Returns:
    - has_peak: 1 if Majorana-like peaks are found, else 0.
    - splitting: Energy splitting between the two modes closest to zero.
    """
    # 1. Find all peaks based on prominence
    peaks, _ = find_peaks(conductance_arr, prominence=prominence)
    
    # 2. Boundary Filter: Exclude peaks at the very edges
    filtered_peaks = peaks[(peaks > 0) & (peaks < len(conductance_arr) - 1)]
    
    if len(filtered_peaks) == 0:
        return 0, 0.0
    
    has_peak = 1
    
    # 3. Check for a Zero-Bias Peak (ZBP)
    # Using tolerance based on mesh spacing (similar to notebook eta-scaling)
    dE = np.abs(energy_mesh[1] - energy_mesh[0])
    atol = 1.5 * dE
    
    # Check if any filtered peak is "at zero"
    peak_energies = energy_mesh[filtered_peaks]
    is_zero_peak = np.any(np.isclose(peak_energies, 0, atol=atol))
    
    if is_zero_peak:
        return 1, 0.0
    
    # 4. Hybridization Splitting: Find the two peaks flanking zero
    pos_peaks = filtered_peaks[energy_mesh[filtered_peaks] > 0]
    neg_peaks = filtered_peaks[energy_mesh[filtered_peaks] < 0]
    
    if len(pos_peaks) > 0 and len(neg_peaks) > 0:
        # Closest positive peak to zero
        e_pos = energy_mesh[pos_peaks[np.argmin(energy_mesh[pos_peaks])]]
        # Closest negative peak to zero
        e_neg = energy_mesh[neg_peaks[np.argmax(energy_mesh[neg_peaks])]]
        splitting = e_pos - e_neg
    else:
        splitting = 0.0
        
    return has_peak, splitting


def detect_peaksold(ys, xs, thresh):
    z_idx = np.where(xs.real == 0)[0][0]
    pks = find_peaks(ys, height = thresh)[0]

    if len(pks)==0:
        return None
    
    #center the peak indexes around the zero index
    zpks= pks - z_idx
    idxs = np.where(np.isclose(zpks,0.0))[0]
    if len(idxs) == 0:
        return None
    
    #get the index of the peak closest to zero energy
    min_peak = np.min(pks[idxs[0]])
    
    return min_peak

def detect_peaks(ys, xs):
    z_idx = np.where(np.isclose(xs.real, 0))[0][0]

    zpksidx =find_peaks(ys)[0] - z_idx

    are_peaks = len(zpksidx) != 0
    if not are_peaks:
        return None
    
    # Filter for positive peaks and check if any exist
    pos_pks = zpksidx[zpksidx > 0]
    if len(pos_pks) == 0:
        return None
        
    min_idx = np.min(pos_pks) + z_idx
    return min_idx



def calc_MZM_localization(rho_left, rho_right, pct_thresh = 80.0):
    """Returns the length away from either end of the wire that 80% of the weight is contained in the integration.
        The code starts at an index n = 1, which will correspond to the first and last site and integrate. If the pct weight is less than 80%, it will increase n by one and 
        integrate over the first two sites and last two sites, increasing n until 80% is reached, then it will return the site number n.

    :param rho_M1: _description_
    :type rho_M1: _type_
    :param rho_M2: _description_
    :type rho_M2: _type_
    """
    tot_area = np.trapz(rho_left + rho_right)

    
    n=1
    pct = 0
    while not pct > pct_thresh:
        rho_left_part = rho_left[0:n]
        rho_right_part = rho_right[::-1][0:n]
        
        sum_M1 = np.trapz(rho_left_part)
        sum_M2 = np.trapz(rho_right_part)
        
        pct = 100*(sum_M1 + sum_M2)/(tot_area)
        
        if pct  > pct_thresh:
            break
        else:
            n += 1
            
    return n


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
    phase_plus = np.conj(psi_plus[0]) / np.abs(psi_plus[0] + 1e-20)
    phase_minus = np.conj(psi_minus[0]) / np.abs(psi_minus[0] + 1e-20)
    
    psi_plus = psi_plus * phase_plus
    psi_minus = psi_minus * phase_minus
    
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
        
                


######### The following function builds the system ########

def build_system(t, mu, mu_n, gamma, Delta0, V_z, alpha, Ln, Lb, Ls, mu_leads, barrier_l, barrier_r, Vdisx = None, a = 1):
    syst = kwant.Builder()
    lat = kwant.lattice.square(a, norbs=4)

    # Leads
    sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
    sym_right_lead = kwant.TranslationalSymmetry((a, 0))
    
    left_lead = kwant.Builder(sym_left_lead, conservation_law=np.diag([-2, -1, 1, 2])) 
    right_lead = kwant.Builder(sym_right_lead, conservation_law=np.diag([-2, -1, 1, 2])) 
    
    Z = Delta0/(Delta0 + gamma) # renormalization for the low energy effective hamiltonian.
    
    
    mu_s = np.zeros(Ls)
    
    if Vdisx is None:
        Vdisx  = np.zeros_like(mu_s)
        
    for i in range(Ls):
        mu_s[i] = mu - Vdisx[i]  
        
    

    # 1. Left Barrier 
    for i in range(Lb):
        syst[lat(i, 0)] = (2 * t + barrier_l) * np.kron(sigma_z, sigma_0)
        if i > 0:
            syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 0.5*1j*alpha * np.kron(sigma_z, sigma_y) 
            
    # 3. Superconductor (Renormalized)
    for i in range(Lb+Ln, Lb+Ln+Ls):
        syst[lat(i, 0)] = Z * (2 * t - mu_s[i-Lb-Ln]) * np.kron(sigma_z, sigma_0) \
                          + Z * V_z * np.kron(sigma_0, sigma_x) \
                          + Z * gamma * np.kron(sigma_x, sigma_0) 
        if i > 0: 
            # Boundary hopping is sqrt(Z), bulk SC hopping is Z
            z_hop = np.sqrt(Z) if i == Lb+Ln else Z
            syst[lat(i, 0), lat(i-1, 0)] = z_hop * (-t * np.kron(sigma_z, sigma_0) + 0.5*1j*alpha * np.kron(sigma_z, sigma_y))
            

    # 5. Right Barrier 
    for i in range(Lb+Ln+Ls+Ln, Lb+Ln+Ls+Ln+Lb):
        syst[lat(i, 0)] = (2 * t + barrier_r) * np.kron(sigma_z, sigma_0)
        if i > 0: 
            # Boundary hopping exiting SC is sqrt(Z) (Applies here if Ln == 0)
            z_hop = np.sqrt(Z) if i == Lb+Ln+Ls else 1.0
            syst[lat(i, 0), lat(i-1, 0)] = z_hop * (-t * np.kron(sigma_z, sigma_0) + 0.5*1j*alpha * np.kron(sigma_z, sigma_y))
    
    
    left_lead[lat(0, 0)] = (2 * t - (mu_leads)) * np.kron(sigma_z, sigma_0) 
    left_lead[lat(1, 0), lat(0, 0)] = -t * np.kron(sigma_z, sigma_0)
    
    right_lead[lat(0, 0)] = (2 * t - (mu_leads)) * np.kron(sigma_z, sigma_0) 
    right_lead[lat(1, 0), lat(0, 0)] = -t * np.kron(sigma_z, sigma_0)

    syst.attach_lead(left_lead) 
    syst.attach_lead(right_lead)
    return syst.finalized()


def build_system_closed(t, mu, gamma, Delta0, V_z, alpha, Ls, Vdisx, a=1):
    # Strictly the SM-SC region (no barriers or leads)
    
    syst = kwant.Builder()
    lat = kwant.lattice.square(a, norbs=4)
    
    Z = Delta0 / (Delta0 + gamma)
    
    # Calculate the finite-size corrected band bottom
    epsilon0 = 2 * t * np.cos(np.pi / (Ls + 1.0))
    
    mu_s = np.zeros(Ls)
        
    for i in range(Ls):
        mu_s[i] = mu - Vdisx[i]

    # 1. Superconductor (Renormalized) - Only region left
    for i in range(Ls):
        # Replaced 2 * t with epsilon0
        syst[lat(i, 0)] = Z * (epsilon0 - mu_s[i]) * np.kron(sigma_z, sigma_0) \
                          + Z * V_z * np.kron(sigma_0, sigma_x) \
                          + Z * gamma * np.kron(sigma_x, sigma_0) 
        
        if i > 0: 
            # Uniform hopping throughout the pure SM-SC wire
            syst[lat(i, 0), lat(i-1, 0)] = Z * (-t * np.kron(sigma_z, sigma_0) + 0.5*1j*alpha * np.kron(sigma_z, sigma_y))
    
    return syst.finalized()


######## Disorder Calculation Helper Functions ########

def calc_normalized(x):
    return x/np.max(np.abs(x))
    


def initialize_vdis_from_data(path):
    
    x = np.load(path)['Vdisx']
    return x



def fVd(x, Lambda_dis):
     # Lambda_dis is the decay length (Å)
    """Exponential decay function."""
    return np.exp(-x / Lambda_dis)



def generate_Vdis(Nx, Ny, Rimp, ax, ay, lambda_dis):
    """Generate smooth random disorder potential."""
    Nimp = len(Rimp)
    vtp = np.zeros((Nx, Ny), dtype=float)

    x_grid = np.arange(1, Nx + 1)[:, None]
    y_grid = np.arange(1, Ny + 1)[None, :]

    for kk in range(Nimp):
        sign = (-1)**(kk + 1)
        dd = np.sqrt((x_grid - Rimp[kk, 0])**2 * ax**2 +
                     (y_grid - Rimp[kk, 1])**2 * ay**2)
        
        vtp += sign * fVd(dd, lambda_dis)

    # subtract mean and normalize variance
    vtp -= np.mean(vtp)
    vtp /= np.sqrt(np.mean(vtp**2))
    return vtp



def generate_1d_Vdis(Nx, ax, num_impurities, amplitude, lambda_dis):
    #generating a set of "2D" coordinate, where y coordinates are set to zero for this 1d case. 

        Ny = 1
        ay = ax
        Rimp = np.column_stack((
        np.random.uniform(0, Nx + 1, num_impurities),
        np.random.uniform(0, Ny + 1, num_impurities)
        ))
        
        Vdis = generate_Vdis(Nx, Ny, Rimp, ax, ay, lambda_dis)
        
        Y0 = np.ones(Ny)
        Vxd = np.array([np.dot(Vdis[ii, :], Y0) for ii in range(Nx)])
        #normalizing disorder and scaling by predefined amplitude
        normed_Vxd = (Vxd/np.max(np.abs(Vxd)) )* amplitude
        
        return normed_Vxd
    

def calc_invariant_metric_old(f1,f2):

    norm_f1 = (f1/np.max(f1)) 
    norm_f2 = (f2/np.max(f2))

    normf1mean = norm_f1.mean()
    normf2mean = norm_f2.mean()

    fmax = norm_f1 if normf1mean > normf2mean else f2
    fmin = norm_f1 if normf1mean < normf2mean else f2


    return (np.sum(fmin - 1)*np.sum(fmax - 1))/(np.sum(fmin - 1)*np.sum(fmin - 1))


def calc_invariant_metric(f1, f2):
    f1 = f1/f1.max()
    f2 = f2/f2.max()
    
    f1mean = f1.mean()
    f2mean = f2.mean()
    
    if f1mean > f2mean:
        fmax = f1
        fmin = f2
    else:
        fmax = f2
        fmin = f1
        
    return (np.sum(fmin - 1)*np.sum(fmax - 1))/(np.sum(fmin - 1)*np.sum(fmin - 1))
        
    

def calc_correlation(f1,f2):
    f1 = f1/f1[0]
    f2 = f2/f2[0]

    corr = np.dot(f1,f2)/(np.linalg.norm(f1)*np.linalg.norm(f2))
    
    return corr


def calc_derivatives(x, y):
    """
    Calculates the average 1st and 2nd derivatives of y(x).
    """
    if len(x) < 3:
        return 0.0, 0.0
    
    dx = x[1] - x[0]
    dy = np.gradient(y, dx)
    d2y = np.gradient(dy, dx)
    
    avg_d1 = np.mean(dy)
    avg_d2 = np.mean(d2y)
    
    return avg_d1, avg_d2


def calc_center_of_mass(x, y):
    """
    Calculates the center of mass (weighted average) of the curve y(x).
    """
    if np.sum(y) == 0:
        return np.mean(x)
    
    com = np.sum(x * y) / np.sum(y)
    return com
    
    
def calc_spectrum(syst_closed, k=22):
    """
    Calculates the k eigenvalues closest to zero for the closed system.
    Returns them sorted from lowest (most negative) to highest (most positive).
    """
    # Extract the tight-binding matrix from the Kwant system
    ham = syst_closed.hamiltonian_submatrix(sparse=True)
    
    try:
        evals, evecs = sla.eigsh(ham, k=k, sigma=0, which='LM')
    except:
        evals = np.linalg.eigvalsh(ham.toarray())
        
    # Sort by absolute value to isolate the k states closest to zero
    idx_abs = np.argsort(np.abs(evals))
    closest_evals = evals[idx_abs[:k]]
    
    sorted_evals = np.sort(closest_evals)
    
    return sorted_evals
    
        
    
    
####################### Code for Calculating Periodic Disorder Invariant #########################

# =============================================================================
# Adapted from binayyakbhroy/periodic-disorder-invariant
# =============================================================================

# Pauli matrices for PDI calculation
s0_pdi = np.eye(2)
sx_pdi = np.array([[0, 1], [1, 0]])
sy_pdi = np.array([[0, -1j], [1j, 0]])
sz_pdi = np.array([[1, 0], [0, -1]])

zeros_4 = np.zeros((4, 4))
zeros_8 = np.zeros((8, 8))

# Definitions for PDI Hamiltonian construction (Basis: Nambu x Spin)
# Note: These kron products match the basis used from PDI library
chirality_op = np.kron(sx_pdi, np.eye(2))
szs0 = np.kron(sz_pdi, s0_pdi)
szsx = np.kron(sz_pdi, sx_pdi)
sysy = np.kron(sy_pdi, sy_pdi)
szsy = np.kron(sz_pdi, sy_pdi)



# =============================================================================
# Exact Mathematica PDI Translation
# =============================================================================

class PDICalculator:
    def __init__(self, ts, alphas, gamma, Nx, Vdisx, V0):
        self.ts = ts
        self.alphas = alphas
        self.gamma = gamma
        self.Nx = Nx
        self.Vdisx = Vdisx
        self.V0 = V0

        self.s0 = np.eye(2)
        self.sx = np.array([[0, 1], [1, 0]])
        self.sy = np.array([[0, -1j], [1j, 0]])
        self.sz = np.array([[1, 0], [0, -1]])
        self.S = np.kron(self.sx, self.s0)

        self.zero4 = np.zeros((4, 4), dtype=complex)

        self.h1 = np.array([
            [-self.ts,         -self.alphas / 2,  0,               0             ],
            [ self.alphas / 2, -self.ts,          0,               0             ],
            [ 0,                0,                self.ts,         self.alphas / 2],
            [ 0,                0,               -self.alphas / 2, self.ts        ]
        ], dtype=complex)
        self.h1T = self.h1.T

        self.TT = np.block([
            [self.h1,    self.zero4],
            [self.zero4, self.h1T  ]
        ])
        self.TT1 = np.block([
            [self.h1T,   self.zero4],
            [self.zero4, self.h1   ]
        ])
        self.sigr = None

    def h0(self, gm, mu):
        return np.array([
            [ 2*self.ts - mu,  gm,              0,               self.gamma ],
            [ gm,              2*self.ts - mu, -self.gamma,      0     ],
            [ 0,              -self.gamma,     -2*self.ts + mu, -gm    ],
            [ self.gamma,      0,              -gm,             -2*self.ts + mu]
        ], dtype=complex)

    def H1(self, gm, mu, ii):
        htp = self.h0(gm, mu).copy()
        v_disorder = self.V0 * self.Vdisx[ii - 1]
        htp[0, 0] += v_disorder
        htp[1, 1] += v_disorder
        htp[2, 2] -= v_disorder
        htp[3, 3] -= v_disorder
        return htp

    def SigR(self, gm, mu):
        sgtp = np.zeros((8, 8), dtype=complex)
        sgrtp = [sgtp]
        gg0 = np.block([
            [self.H1(gm, mu, int(self.Nx/2)), self.h1],
            [self.h1T, self.H1(gm, mu, int(self.Nx/2) + 1)]
        ])
        grtp = np.linalg.inv(-gg0 - sgtp)
        sgtp = self.TT @ grtp @ self.TT1
        sgrtp.append(sgtp)
        for ii in range(int(self.Nx/2) - 1, 1, -1):
            gg0 = np.block([
                [self.H1(gm, mu, ii), self.zero4],
                [self.zero4, self.H1(gm, mu, self.Nx - ii + 1)]
            ])
            grtp = np.linalg.inv(-gg0 - sgtp)
            sgtp = self.TT @ grtp @ self.TT1
            sgrtp.append(sgtp)
        return sgrtp

    def SigL(self, gm, mu, q):
        sgtp = np.zeros((8, 8), dtype=complex)
        sgltp = [sgtp]
        gg0 = np.block([
            [self.H1(gm, mu, 1), self.h1T * np.exp(-1.0j * self.Nx * q)],
            [self.h1 * np.exp(1.0j * self.Nx * q), self.H1(gm, mu, self.Nx)]
        ])
        gltp = np.linalg.inv(-gg0 - sgtp)
        sgtp = self.TT1 @ gltp @ self.TT
        sgltp.append(sgtp)
        for ii in range(2, int(self.Nx/2)):
            gg0 = np.block([
                [self.H1(gm, mu, ii), self.zero4],
                [self.zero4, self.H1(gm, mu, self.Nx - ii + 1)]
            ])
            gltp = np.linalg.inv(-gg0 - sgtp)
            sgtp = self.TT1 @ gltp @ self.TT
            sgltp.append(sgtp)
        return sgltp

    def G0(self, gm, mu, q):
        sigl = self.SigL(gm, mu, q)
        sigr = self.sigr # Grab precalculated SigR from fq
        
        block1 = np.block([
            [-self.H1(gm, mu, 1), -self.h1T * np.exp(-1.0j * self.Nx * q)],
            [-self.h1 * np.exp(1.0j * self.Nx * q), -self.H1(gm, mu, self.Nx)]
        ])
        block2 = np.block([
            [-self.H1(gm, mu, 2), self.zero4],
            [self.zero4, -self.H1(gm, mu, self.Nx - 1)]
        ])
        gtp = np.linalg.inv(np.block([
            [block1 - sigl[0], -self.TT],
            [-self.TT1, block2 - sigr[int(self.Nx/2) - 2]]
        ]))
        gg0 = gtp
        for ii in range(2, int(self.Nx/2) - 1):
            block_ii_1 = np.block([
                [-self.H1(gm, mu, ii), self.zero4],
                [self.zero4, -self.H1(gm, mu, self.Nx - ii + 1)]
            ])
            block_ii_2 = np.block([
                [-self.H1(gm, mu, ii + 1), self.zero4],
                [self.zero4, -self.H1(gm, mu, self.Nx - ii)]
            ])
            gtp = np.linalg.inv(np.block([
                [block_ii_1 - sigl[ii - 1], -self.TT],
                [-self.TT1, block_ii_2 - sigr[int(self.Nx/2) - ii - 1]]
            ]))
            gg0 = gg0 + gtp
            
        block_end_1 = np.block([
            [-self.H1(gm, mu, int(self.Nx/2) - 1), self.zero4],
            [self.zero4, -self.H1(gm, mu, int(self.Nx/2) + 2)]
        ])
        block_end_2 = np.block([
            [-self.H1(gm, mu, int(self.Nx/2)), -self.h1],
            [-self.h1T, -self.H1(gm, mu, int(self.Nx/2) + 1)]
        ])
        gtp = np.linalg.inv(np.block([
            [block_end_1 - sigl[int(self.Nx/2) - 2], -self.TT],
            [-self.TT1, block_end_2 - sigr[0]]
        ]))
        gg0 = gg0 + gtp
        
        gg1_block = np.block([
            [-self.H1(gm, mu, 1), -self.h1T * np.exp(-1.0j * self.Nx * q)],
            [-self.h1 * np.exp(1.0j * self.Nx * q), -self.H1(gm, mu, self.Nx)]
        ])
        gg1 = np.linalg.inv(gg1_block - sigr[int(self.Nx/2) - 1])
        
        gg2_block = np.block([
            [-self.H1(gm, mu, int(self.Nx/2)), -self.h1],
            [-self.h1T, -self.H1(gm, mu, int(self.Nx/2) + 1)]
        ])
        gg2 = np.linalg.inv(gg2_block - sigl[int(self.Nx/2) - 1])
        
        return gg0, gg1, gg2

    def fg(self, gm, mu, q):
        GG = self.G0(gm, mu, q)
        gg0, gg1, gg2 = GG[0], GG[1], GG[2]
        
        gg12 = (
            gg0[0:4, 8:12] + 
            gg0[12:16, 4:8] + 
            np.exp(-1.0j * self.Nx * q) * gg1[4:8, 0:4] + 
            gg2[0:4, 4:8]
        ) / self.Nx
        
        gg21 = (
            gg0[8:12, 0:4] + 
            gg0[4:8, 12:16] + 
            np.exp(1.0j * self.Nx * q) * gg1[0:4, 4:8] + 
            gg2[4:8, 0:4]
        ) / self.Nx
        
        return gg12, gg21

    def fq(self, gm, mu, NL):
        ntp = 0.0
        delta_q = 2.0 * np.pi / (self.Nx * NL)
        q0 = -max(round(0.075 * NL), 1) * delta_q
        
        self.sigr = self.SigR(gm, mu)
        
        qq = -1.0 * np.pi / self.Nx
        while qq < q0 - 0.00001:
            g12, g21 = self.fg(gm, mu, qq)
            trace_val = np.trace(self.S @ (self.h1T @ g12 - self.h1 @ g21))
            ntp += trace_val
            qq += delta_q
            
        qq = q0 + 0.1 * delta_q / 2.0
        while qq < -0.00001:
            g12, g21 = self.fg(gm, mu, qq)
            trace_val = np.trace(self.S @ (self.h1T @ g12 - self.h1 @ g21))
            ntp += 0.1 * trace_val
            qq += 0.1 * delta_q
            
        invariant = np.real(-ntp / NL)
        if abs(invariant) < 1e-4:
            invariant = 0.0
        return invariant

def calculate_pdi(ts, alphas, gamma, Nx, Vdisx, V0, gm, mu, NL):
    """Wrapper function to instantiate the calculator and get the invariant."""
    calculator = PDICalculator(ts, alphas, gamma, Nx, Vdisx, V0)
    return calculator.fq(gm, mu, NL)