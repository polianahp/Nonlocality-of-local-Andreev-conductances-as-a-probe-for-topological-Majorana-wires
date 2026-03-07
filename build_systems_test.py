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
    Delta_ind = Z * gamma # explicitly define induced gap
    
    mu_s = np.zeros(Ls)
    
    if Vdisx is None:
        Vdisx  = np.zeros_like(mu_s)
        
    for i in range(Ls):
        mu_s[i] = mu + Vdisx[i]  

    # 1. Left Barrier (Tracks mu)
    for i in range(Lb):
        syst[lat(i, 0)] = (2 * t - (mu + mu_n) + barrier_l) * np.kron(sigma_z, sigma_0)
        if i > 0:
            syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y) 
            
    # 2. Left Normal (Tracks mu)
    for i in range(Lb, Lb+Ln):
        syst[lat(i, 0)] = (2 * t - (mu + mu_n)) * np.kron(sigma_z, sigma_0)
        if i > 0: 
            syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y)
            
    # 3. Superconductor (Renormalized)
    for i in range(Lb+Ln, Lb+Ln+Ls):
        syst[lat(i, 0)] = Z * (2 * t - mu_s[i-Lb-Ln]) * np.kron(sigma_z, sigma_0) \
                          + Z * V_z * np.kron(sigma_0, sigma_x) \
                          + Delta_ind * np.kron(sigma_x, sigma_0) 
        if i > 0: 
            # Boundary hopping is sqrt(Z), bulk SC hopping is Z
            z_hop = np.sqrt(Z) if i == Lb+Ln else Z
            syst[lat(i, 0), lat(i-1, 0)] = z_hop * (-t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y))
            
    # 4. Right Normal (Tracks mu)
    for i in range(Lb+Ln+Ls, Lb+Ln+Ls+Ln):
        syst[lat(i, 0)] = (2 * t - (mu + mu_n)) * np.kron(sigma_z, sigma_0)
        if i > 0: 
            # Boundary hopping exiting SC is sqrt(Z), bulk normal is 1.0
            z_hop = np.sqrt(Z) if i == Lb+Ln+Ls else 1.0
            syst[lat(i, 0), lat(i-1, 0)] = z_hop * (-t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y)) 
            
    # 5. Right Barrier (Tracks mu)
    for i in range(Lb+Ln+Ls+Ln, Lb+Ln+Ls+Ln+Lb):
        syst[lat(i, 0)] = (2 * t - (mu + mu_n) + barrier_r) * np.kron(sigma_z, sigma_0)
        if i > 0: 
            # Boundary hopping exiting SC is sqrt(Z) (Applies here if Ln == 0)
            z_hop = np.sqrt(Z) if i == Lb+Ln+Ls else 1.0
            syst[lat(i, 0), lat(i-1, 0)] = z_hop * (-t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y))
    
    # 6. Leads (Fermi energy dynamically tracks the semiconductor band bottom)
    left_lead[lat(0, 0)] = (2 * t - (mu + mu_leads)) * np.kron(sigma_z, sigma_0) 
    left_lead[lat(1, 0), lat(0, 0)] = -t * np.kron(sigma_z, sigma_0)
    
    right_lead[lat(0, 0)] = (2 * t - (mu + mu_leads)) * np.kron(sigma_z, sigma_0) 
    right_lead[lat(1, 0), lat(0, 0)] = -t * np.kron(sigma_z, sigma_0)

    syst.attach_lead(left_lead)
    syst.attach_lead(right_lead)
    return syst.finalized()


def build_system_closed(t, mu, mu_n, gamma, Delta0, V_z, alpha, Ln, Lb, Ls, mu_leads, barrier_l, barrier_r, Vdisx = None, a = 1):
    # Same as above but without leads, for calculating majorana polarization
    
    syst = kwant.Builder()
    lat = kwant.lattice.square(a, norbs=4)
    
    Z = Delta0/(Delta0 + gamma)
    Delta_ind = Z * gamma
    
    mu_s = np.zeros(Ls)
    if Vdisx is None:
        Vdisx  = np.zeros_like(mu_s)
        
    for i in range(Ls):
        mu_s[i] = mu + Vdisx[i]    

    # 1. Left Barrier
    for i in range(Lb):
        syst[lat(i, 0)] = (2 * t - (mu + mu_n) + barrier_l) * np.kron(sigma_z, sigma_0)
        if i > 0:
            syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y) 
            
    # 2. Left Normal
    for i in range(Lb, Lb+Ln):
        syst[lat(i, 0)] = (2 * t - (mu + mu_n)) * np.kron(sigma_z, sigma_0)
        if i > 0: 
            syst[lat(i, 0), lat(i-1, 0)] = -t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y)
        
    # 3. Superconductor (Renormalized)
    for i in range(Lb+Ln, Lb+Ln+Ls):
        syst[lat(i, 0)] = Z * (2 * t - mu_s[i-Lb-Ln]) * np.kron(sigma_z, sigma_0) \
                          + Z * V_z * np.kron(sigma_0, sigma_x) \
                          + Delta_ind * np.kron(sigma_x, sigma_0) 
        if i > 0: 
            z_hop = np.sqrt(Z) if i == Lb+Ln else Z
            syst[lat(i, 0), lat(i-1, 0)] = z_hop * (-t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y))
        
    # 4. Right Normal
    for i in range(Lb+Ln+Ls, Lb+Ln+Ls+Ln):
        syst[lat(i, 0)] = (2 * t - (mu + mu_n)) * np.kron(sigma_z, sigma_0)
        if i > 0: 
            z_hop = np.sqrt(Z) if i == Lb+Ln+Ls else 1.0
            syst[lat(i, 0), lat(i-1, 0)] = z_hop * (-t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y)) 
        
    # 5. Right Barrier
    for i in range(Lb+Ln+Ls+Ln, Lb+Ln+Ls+Ln+Lb):
        syst[lat(i, 0)] = (2 * t - (mu + mu_n) + barrier_r) * np.kron(sigma_z, sigma_0)
        if i > 0: 
            z_hop = np.sqrt(Z) if i == Lb+Ln+Ls else 1.0
            syst[lat(i, 0), lat(i-1, 0)] = z_hop * (-t * np.kron(sigma_z, sigma_0) + 1j*alpha * np.kron(sigma_z, sigma_y))
    
    return syst.finalized()