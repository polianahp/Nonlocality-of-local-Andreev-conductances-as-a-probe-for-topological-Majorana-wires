from tqdm import tqdm
import kwant
import tinyarray
import numpy as np
import os
from config import PathConfigs



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
    smatrix = kwant.smatrix(syst, 0.0)
    
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


def calc_dIdV(syst, energies):
    num_engs = len(energies)
    
    ldos = np.zeros(shape = (num_engs, 20192))
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

def build_system(t, mu, mu_n, Delta, V_z, alpha, Ln, Lb, Ls, mu_leads, barrier_l, barrier_r):
    syst = kwant.Builder()
    a = 1
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
    for i in range(Ls):
        mu_s[i] = mu #- (mu - mu_n) * (1 - np.tanh(i/80))    

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


######## Disorder Calculation Helper Functions ########

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
        
        


    
    








