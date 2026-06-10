import numpy as np
from scipy.spatial import KDTree
import helpers as hp

def has_peaks(peak_dat):
    #check that there is a resonance peak 
    #return sum(peak_dat[:, 0]) > 0
    return sum(peak_dat) > 0

def has_peaks_in_window(peak_dat, window):
    #checks if peaks are within an energy window
    # 6-col format: [has_p, has_n, p_pos, p_ht, n_pos, n_ht]
    pos_peaks, neg_peaks = peak_dat[2], peak_dat[4] 
    has_p_peaks, has_n_peaks = peak_dat[0], peak_dat[1]
    
    #p_cond = bool(has_p_peaks) and bool(abs(pos_peaks) <= window)
    #n_cond = bool(has_n_peaks) and bool(abs(neg_peaks) <= window)
    p_cond = bool(abs(pos_peaks) <= window)
    n_cond = bool(abs(neg_peaks) <= window)

    return p_cond and n_cond

def has_symmetric_peaks(peak_dat):
    #check that the peaks are at symmetric voltages
    pos_peaks, neg_peaks = peak_dat[2], peak_dat[4] 
    
    return np.isclose(abs(pos_peaks), abs(neg_peaks))


def has_max_symm(conductance):
    return max(conductance) == conductance[len(conductance)//2]


def has_negative_peaks(peak_dat):
    #has_n_peaks = peak_dat[2]
    #return bool(has_n_peaks)
    
    #this tests that there is actually a peak in negative energy
    #not that the program threw a flag there is
    neg_peaks = peak_dat[4]
    return neg_peaks < 0.0

def get_inside_points_function(pdi_arr):
    #generates function with precomputed point tree
    mu = pdi_arr[:, 0]   
    V_z = pdi_arr[:, 1] 
    points = np.column_stack((V_z, mu))
    tree = KDTree(points)

    def get_inside_points(center, r):
        #center is a tuple (Vz, mu)
        inside = tree.query_ball_point(center, r)
        return inside

    return get_inside_points

def check_peak_window(peaks_data_left, peaks_data_right, window):
    condition  = lambda i: has_peaks_in_window(peaks_data_left[i,:], window) \
                            and has_peaks_in_window(peaks_data_right[i,:], window)
    return np.asarray([condition(i) for i in range(peaks_data_right.shape[0])])

def check_peak_symmetry(peaks_data_left, peaks_data_right):
    # Note: Fixed the bug where peaks_data_left was passed twice!
    condition  = lambda i: has_symmetric_peaks(peaks_data_left[i,:]) \
                            and has_symmetric_peaks(peaks_data_right[i,:])
    return np.asarray([condition(i) for i in range(peaks_data_right.shape[0])])

def check_arr_monotonic(cond_arr1, cond_arr2):
    is_monotonic_arr = np.asarray([(np.all(np.diff(cond_arr1[i,:])<=0) &
                        np.all(np.diff(cond_arr2[i,:])<=0)).astype(int) for i in range(cond_arr2.shape[0])])
    return is_monotonic_arr

def check_max_at_symm(cond_arr1, cond_arr2):
    condition = lambda i: has_max_symm(cond_arr1[i,:]) and has_max_symm(cond_arr2[i,:])
    ret = np.asarray([condition(i) for i in range(cond_arr2.shape[0])]).astype(int)
    return ret

def check_resonance_peak(cond_arr1, cond_arr2):
    return np.asarray([has_peaks(cond_arr1[i,:]) and has_peaks(cond_arr2[i,:]) for i in range(cond_arr2.shape[0])]).astype(int)

def check_negative_peaks_arr(cond_arr1, cond_arr2):
    return np.asarray([has_negative_peaks(cond_arr1[i,:]) or has_negative_peaks(cond_arr2[i,:]) for i in range(cond_arr2.shape[0])]).astype(int)

def check_correlation(cond_arr1, cond_arr2, corr_thresh):
    corrs = np.asarray([hp.calc_correlation(cond_arr1[i,:], cond_arr2[i,:]) for i in range(cond_arr2.shape[0])])
    corrs = np.asarray([1.0 if corr > 1 else corr for corr in corrs])
    corrs = np.asarray([0.0 if corr < 0.0 else corr for corr in corrs])
    corrs = np.asarray([0.0 if corr < corr_thresh else corr for corr in corrs])
    corrs = np.asarray([1.0 if corr > corr_thresh else corr for corr in corrs])
    return corrs

def check_island_stability(corrs, pdi_arr, radius=0.05, frac=1.0):
    get_inside_pts = get_inside_points_function(pdi_arr)
    mu = pdi_arr[:, 0]
    V_z = pdi_arr[:, 1]
    points = np.column_stack((V_z, mu))
    stable_corrs = np.zeros_like(corrs)
    for i in range(len(corrs)):
        if corrs[i] > 0:
            neighbors = get_inside_pts(points[i], radius)
            if len(neighbors) > 0:
                correlated_neighbors = sum(1 for n in neighbors if corrs[n] > 0)
                if (correlated_neighbors / len(neighbors)) >= frac:
                    stable_corrs[i] = 1.0
            else:
                stable_corrs[i] = 1.0
    return stable_corrs

def compute_probabilities(protocol_map, winding_map):
    A_bool = winding_map == 1.0
    B_bool = protocol_map == 1.0
    p_not_a_given_b = np.sum(~A_bool & B_bool) / np.sum(B_bool) if np.sum(B_bool) > 0 else np.nan
    p_not_b_given_a = np.sum(A_bool & ~B_bool) / np.sum(A_bool) if np.sum(A_bool) > 0 else np.nan
    return p_not_a_given_b, p_not_b_given_a

def calc_protocol(cond_arr1, cond_arr2, peaks_data_left, peaks_data_right, pdi_arr, **kwargs):
    prot_dat = np.ones_like(cond_arr1[:,0])
    
    if kwargs.get('check_correlation', False):
        corrs = check_correlation(cond_arr1, cond_arr2, kwargs["corr_thresh"])
        prot_dat *= corrs
    else:
        corrs = np.ones_like(prot_dat) # For island stability if correlation is off, though usually it's on
    
    if kwargs.get('check_resonance_peak', False):
        has_peak_arr = check_resonance_peak(peaks_data_left, peaks_data_right) 
        prot_dat *= (1 - has_peak_arr)
        
    if kwargs.get('check_negative_peaks', False):
        has_neg_peaks = check_negative_peaks_arr(peaks_data_left, peaks_data_right) 
        prot_dat *= (1 - has_neg_peaks)
        
    if kwargs.get('check_monotonic', False):
        is_monotonic = check_arr_monotonic(cond_arr1, cond_arr2)
        prot_dat *= is_monotonic
        
    if kwargs.get('check_peak_symmetry', False):
        are_symmetric = check_peak_symmetry(peaks_data_left, peaks_data_right)
        prot_dat *= are_symmetric
        
    if kwargs.get('check_peak_window', False):
        peaks_in_window = check_peak_window(peaks_data_left, peaks_data_right, kwargs["window"])
        prot_dat *= peaks_in_window
        
    if kwargs.get('check_island_stability', False):
        in_stable_island = check_island_stability(prot_dat, pdi_arr, 
                                               radius=kwargs['stability_radus'], frac=kwargs['stability_frac'])
        prot_dat *= in_stable_island
        
    return prot_dat
