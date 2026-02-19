from tqdm import tqdm
import kwant
import tinyarray
import numpy as np
import os
from config import PathConfigs
from pathlib import Path
import scipy.sparse.linalg as sla



#pauli matrices
sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])

######## Path Configuration Helpers
def get_data_path(file_name, subdirectory):
    #constructs output data paths based on path configuration in config.py
    # path constructed should be DATA/subdirectory/filename
    os.makedirs(PathConfigs.DATA / subdirectory, exist_ok=True)
    return PathConfigs.DATA / subdirectory / file_name

def np_save_wrapped(data, filename, subdirectory):
    #wrapper for np.savetxt to save to default data bath
    path = get_data_path(f"{filename}.npy", subdirectory)
    np.save(path, data)
    
def np_load_wrapped(filename, subdirectory):
    #wrapper for np.loadtxt to load from default data bath
    path = get_data_path(f"{filename}.npy", subdirectory)
    return np.load(path)
    
    
######## Calculation Helpers ########

def calc_integrated_area_diff(conductance_left, conductance_right):
    
    normcond_L = conductance_left/conductance_left[0]
    normcond_R = conductance_right/conductance_right[0]

    Area_L = np.trapz(normcond_L)
    Area_R = np.trapz(normcond_R)

    return np.abs(Area_R - Area_L)/(Area_R + Area_L)


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
    
    ldos = np.zeros(shape = (num_engs, 2192))
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

######### The following function builds the system ########

def build_system(t, mu, mu_n, Delta, V_z, alpha, Ln, Lb, Ls, mu_leads, barrier_l, barrier_r, Vdisx = None, a = 1):
    syst = kwant.Builder()
    lat = kwant.lattice.square(a, norbs=4)

    #lead
    sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
    sym_right_lead = kwant.TranslationalSymmetry((a, 0))
    
    
    left_lead = kwant.Builder(sym_left_lead, conservation_law=np.diag([-2, -1, 1, 2])) 
    right_lead = kwant.Builder(sym_right_lead, conservation_law=np.diag([-2, -1, 1, 2])) 
    
    #print(f"left lead type: {type(left_lead)}")
    #print(f"right lead type: {type(right_lead)}")
    #print(f"syst type: {type(syst)}")
    

    #Here one can edit the chemical potential profile. The profile below is for the quasi-Majorana case, the pristine case would be 
    # mu_s[i] = mu, and for disorder we use mu_s[i] = mu + (some random value)
    # For the nontopological ABS (dotted magenta curve in Figs. 2 c and d), set alpha = 0 in the central region and add a 
    # Zeeman energy on the normal segments of the wire.
    
    mu_s = np.zeros(Ls)
    
    if Vdisx is None:
        Vdisx  = np.zeros_like(mu_s)
        
    for i in range(Ls):
        mu_s[i] = mu + Vdisx[i] #- (mu - mu_n) * (1 - np.tanh(i/80))    

    for i in range(Lb):
        syst[lat(i, 0)] = (2 * t - mu_n + barrier_l) * np.kron(sigma_z, sigma_0)
        if i > 0:
            syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y) 
            
    for i in range(Lb, Lb+Ln):
        syst[lat(i, 0)] = (2 * t - mu_n) * np.kron(sigma_z, sigma_0)
        syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y)
    for i in range(Lb+Ln, Lb+Ln+Ls):
        syst[lat(i, 0)] = (2 * t - mu_s[i-Lb-Ln]) * np.kron(sigma_z, sigma_0) + Delta * np.kron(sigma_x, sigma_0) + V_z * np.kron(sigma_0, sigma_x)
        syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y)
    for i in range(Lb+Ln+Ls, Lb+Ln+Ls+Ln):
        syst[lat(i, 0)] = (2 * t - mu_n) * np.kron(sigma_z, sigma_0)
        syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y) 
    for i in range(Lb+Ln+Ls+Ln, Lb+Ln+Ls+Ln+Lb):
        syst[lat(i, 0)] = (2 * t - mu_n + barrier_r) * np.kron(sigma_z, sigma_0)
        syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y)
    

    left_lead[lat(0, 0)] = (2 * t - mu_leads) * np.kron(sigma_z, sigma_0) 
    left_lead[lat(1, 0), lat(0, 0)] = -t * np.kron(sigma_z, sigma_0)
    right_lead[lat(0, 0)] = (2 * t - mu_leads) * np.kron(sigma_z, sigma_0) 
    right_lead[lat(1, 0), lat(0, 0)] = -t * np.kron(sigma_z, sigma_0)

    syst.attach_lead(left_lead)
    syst.attach_lead(right_lead)
    return syst.finalized()



def build_system_closed(t, mu, mu_n, Delta, V_z, alpha, Ln, Lb, Ls, mu_leads, barrier_l, barrier_r, Vdisx = None, a = 1):
    #Same as above but without leads, for calculating majorana polarization
    
    
    syst = kwant.Builder()
    lat = kwant.lattice.square(a, norbs=4)

    #lead
    sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
    sym_right_lead = kwant.TranslationalSymmetry((a, 0))
    
    
    left_lead = kwant.Builder(sym_left_lead, conservation_law=np.diag([-2, -1, 1, 2])) 
    right_lead = kwant.Builder(sym_right_lead, conservation_law=np.diag([-2, -1, 1, 2])) 
    
    #print(f"left lead type: {type(left_lead)}")
    #print(f"right lead type: {type(right_lead)}")
    #print(f"syst type: {type(syst)}")
    

    #Here one can edit the chemical potential profile. The profile below is for the quasi-Majorana case, the pristine case would be 
    # mu_s[i] = mu, and for disorder we use mu_s[i] = mu + (some random value)
    # For the nontopological ABS (dotted magenta curve in Figs. 2 c and d), set alpha = 0 in the central region and add a 
    # Zeeman energy on the normal segments of the wire.
    
    mu_s = np.zeros(Ls)
    if Vdisx is None:
        Vdisx  = np.zeros_like(mu_s)
        
    
    for i in range(Ls):
        mu_s[i] = mu + Vdisx[i] #- (mu - mu_n) * (1 - np.tanh(i/80))    

    for i in range(Lb):
        syst[lat(i, 0)] = (2 * t - mu_n + barrier_l) * np.kron(sigma_z, sigma_0)
        if i > 0:
            syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y) 
            
    for i in range(Lb, Lb+Ln):
        syst[lat(i, 0)] = (2 * t - mu_n) * np.kron(sigma_z, sigma_0)
        syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y)
        
    for i in range(Lb+Ln, Lb+Ln+Ls):
        syst[lat(i, 0)] = (2 * t - mu_s[i-Lb-Ln]) * np.kron(sigma_z, sigma_0) + Delta * np.kron(sigma_x, sigma_0) + V_z * np.kron(sigma_0, sigma_x)
        syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y)
        
    for i in range(Lb+Ln+Ls, Lb+Ln+Ls+Ln):
        syst[lat(i, 0)] = (2 * t - mu_n) * np.kron(sigma_z, sigma_0)
        syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y) 
        
    for i in range(Lb+Ln+Ls+Ln, Lb+Ln+Ls+Ln+Lb):
        syst[lat(i, 0)] = (2 * t - mu_n + barrier_r) * np.kron(sigma_z, sigma_0)
        syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y)
    

    return syst.finalized()



######## Disorder Calculation Helper Functions ########

def initialize_vdis_from_data(path):
    Vdisx = np.load(path)['Vdisx']
    return Vdisx



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
        normed_Vxd = Vxd/np.max(np.abs(Vxd)) * amplitude
        
        return normed_Vxd
    

def calc_invariant_metric(f1, f2):
    smdiff = np.sum(f1-f2)
    sgn = smdiff/np.abs(smdiff)
    fmax = f2 if sgn < 0 else f1
    fmin = f2 if sgn > 0 else f1
    
    invA = 1/np.sum(fmax)
    
    fn_max = fmax * invA
    fn_min = fmin/np.sum(fmin)
    
    corr = (np.dot(fn_max, fn_min) - invA)/(np.dot(fn_min, fn_min) - invA) 
    
    return corr
    
    
        
    
    
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

def inv_pdi(m):
    """Robust matrix inversion for PDI."""
    i = np.eye(m.shape[0])
    try:
        return np.linalg.solve(m, i)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(m, rcond=1e-15)

def fV(q, h_onsite, h_hopping, translation, translation_pr, sigma_right, disorder, N):
    """
    Recursive Green's Function calculation for the disordered supercell.
    """
    V_x = disorder
    # Ensure disorder array matches length N
    if len(V_x) > N:
        V_x = V_x[:N]
    
    # Precompute constants
    phase_pos = np.exp(1.0j * q * N)
    phase_neg = np.exp(-1.0j * q * N)

    g_block11 = lambda i: h_onsite + V_x[i] * szs0
    g_block12 = h_hopping.T * phase_neg
    g_block21 = h_hopping * phase_pos

    # ---------- Compute Sigma_L recursively ----------
    sigma_left = [zeros_8]
    green_g0 = np.block([[g_block11(0), g_block12], [g_block21, g_block11(-1)]])
    
    # Helper for intermediate blocks
    green_gi = lambda i: np.block([[g_block11(i), zeros_4], [zeros_4, g_block11(-i-1)]])
    
    green_inv = inv_pdi((-1.0 * (green_g0 + zeros_8)))
    sigma_ith = translation_pr @ (green_inv @ translation)
    sigma_left.append(sigma_ith)

    for i in range(1, int(N/2) - 1):
        green_inv = inv_pdi((-1.0 * (green_gi(i) + sigma_ith)))
        sigma_ith = translation_pr @ (green_inv @ translation)
        sigma_left.append(sigma_ith)

    green_temp0_block11 = np.block([[-1.0*g_block11(0), -1.0*g_block12],
                                    [-1.0*g_block21, -1.0*g_block11(-1)]])
    green_temp0_block22 = np.block([[-1.0*g_block11(1), zeros_4],
                                    [zeros_4, -1.0*g_block11(-2)]])

    green_temp0 = np.block([
        [green_temp0_block11 - sigma_left[0], -1.0*translation],
        [-1.0*translation_pr, green_temp0_block22 - sigma_right[-2]]
    ])

    # ---------- Full Green's function matrix ----------
    greens_func_matrix = inv_pdi(green_temp0)
    
    green_temp_block11 = lambda i: np.block([[-1.0*g_block11(i), zeros_4], [zeros_4, -1.0*g_block11(-i-1)]])
    green_temp_block22 = lambda i: np.block([[-1.0*g_block11(i + 1), zeros_4], [zeros_4, -1.0*g_block11(-i)]])
    green_temp = lambda i: np.block([
        [green_temp_block11(i) - sigma_left[i], -1.0*translation],
        [-1.0*translation_pr, green_temp_block22(i) - sigma_right[-i]]
    ])

    for i in range(1, int(N/2) - 2):
        greens_func_matrix += inv_pdi(green_temp(i))

    green_tempN_block11 = np.block([[-1.0*g_block11(int(N/2)-2), zeros_4], [zeros_4, -1.0*g_block11(int(N/2)+1)]])
    green_tempN_block22 = np.block([[-1.0*g_block11(int(N/2)-1), -1.0*h_hopping], 
                                    [-1.0*h_hopping.T, -1.0*g_block11(int(N/2))]])

    green_tempN = np.block([
        [green_tempN_block11 - sigma_left[-2], -1.0*translation],
        [-1.0*translation_pr, green_tempN_block22 - sigma_right[0]]
    ])

    greens_func_matrix += inv_pdi(green_tempN)

    greens_func_ex_right = inv_pdi(green_temp0_block11 - sigma_right[-1])
    greens_func_ex_left = inv_pdi(green_tempN_block22 - sigma_left[-1])

    # --------- Nearest-neighbor Green's functions ---------
    G_12 = (
        greens_func_matrix[0:4, 8:12]
        + greens_func_matrix[12:16, 4:8]
        + (greens_func_ex_right[4:8, 0:4] * phase_neg)
        + greens_func_ex_left[0:4, 4:8]
    ) / N

    G_21 = (
        greens_func_matrix[8:12, 0:4]
        + greens_func_matrix[4:8, 12:16]
        + (greens_func_ex_right[0:4, 4:8] * phase_pos)
        + greens_func_ex_left[4:8, 0:4]
    ) / N

    return G_12, G_21


def calculate_pdi(t, mu, Delta, V_z, alpha, Ls, Vdisx, q_N=1):
    """
    Calculates the Periodic Disorder Invariant (PDI).
    
    Parameters:
    - t, mu, Delta, V_z, alpha: System parameters.
    - Ls: Length of the superconducting region (number of sites).
    - Vdisx: Disorder potential array of length Ls.
    - q_N: Integration grid density (default 1).
    
    Returns:
    - nu: The topological invariant (winding number ~0 or ~1).
    """
    N = Ls
    
    # Ensure Vdisx is the correct length
    if len(Vdisx) != N:
        # If passed disorder is larger (e.g. from a file), crop it
        V_x = Vdisx[:N]
    else:
        V_x = Vdisx

    # Parameters from arguments
    # Note: PDI library uses a specific Hamiltonian basis
    # h_onsite = (2t - mu) SzS0 + Vz SzSx - Delta SySy
    # h_hopping = -t SzS0 - i(alpha/2) SzSy
    
    # Mapping inputs to PDI Hamiltonian terms
    e_z = V_z 
    delta_ind = Delta
    alpha_val = alpha 

    delta_k = (2 * np.pi) / (q_N * N)
    q = np.arange(-np.pi/N, -0.1*np.pi/N, delta_k)

    h_onsite = (
        (2*t - mu) * szs0
        + e_z * szsx
        - delta_ind * sysy
    )

    h_hopping = (
        -t * szs0
        - 1.0j * (alpha_val / 2.0) * szsy
    )

    translation = np.block([[h_hopping, zeros_4], [zeros_4, h_hopping.T]])
    translation_pr = np.block([[h_hopping.T, zeros_4], [zeros_4, h_hopping]])

    sigma_right = [zeros_8]

    gsr0_block11 = h_onsite + V_x[int(N/2)-1]*szs0
    gsr0_block22 = h_onsite + V_x[int(N/2)]*szs0

    green_gsr0 = np.block([[gsr0_block11, h_hopping], [h_hopping.T, gsr0_block22]])
    green_g0 = inv_pdi(-1.0 * (green_gsr0 + zeros_8)) 
    sigma_ri = translation @ (green_g0 @ translation_pr)
    sigma_right.append(sigma_ri)

    gsr_i_block11 = lambda i: h_onsite + V_x[i]*szs0
    gsr_i_block22 = lambda i: h_onsite + V_x[-i-1]*szs0

    green_gsr_i = lambda i: np.block([[gsr_i_block11(i), zeros_4], [zeros_4, gsr_i_block22(i)]])

    for i in range(int(N/2) - 2, 0, -1):
        green_inv = inv_pdi(-1.0 * (green_gsr_i(i) + sigma_ri))
        sigma_ri = translation @ (green_inv @ translation_pr)
        sigma_right.append(sigma_ri)
    
    ntp = 0
    
    for q0 in q:
        G_12, G_21 = fV(q0, h_onsite, h_hopping, translation, translation_pr, sigma_right, V_x, N)
        ntp += np.trace(chirality_op @ ((h_hopping.T @ G_12) - (h_hopping @ G_21)))
        
    q0 = (-0.1 * np.pi / N) + (0.5 * delta_k)
    while q0 < -0.00001:
        G_12, G_21 = fV(q0, h_onsite, h_hopping, translation, translation_pr, sigma_right, V_x, N)
        ntp += 0.1 * np.trace(chirality_op @ ((h_hopping.T @ G_12) - (h_hopping @ G_21)))
        q0 += 0.1 * delta_k

    return np.real(-ntp / q_N)
        
        


    
    








