import pytest
import numpy as np
import main_parallel as mp_module
import helpers as hp

def test_worker_step_cpu():
    # Minimal static params
    t = 100.0
    mu = 0.5
    mu_n = 0.0
    gamma = 0.2
    Delta0 = 0.3
    alpha = 10.0
    Ls = 10
    mu_leads = 100.0
    barrier0 = 2.0
    Vdisx = np.zeros(Ls)
    energies = np.array([0.0, 0.1])
    barrier_arr = np.array([2.0, 5.0])
    
    static_params = {
        't': t, 'mu_n': mu_n, 'Delta0': Delta0, 'alpha': alpha, 'gamma': gamma,
        'V0': 1.0, 'qn': 20, 'Ln': 0, 'Lb': 2, 'Barrier_Height': barrier0,
        'Ls': Ls, 'mu_leads': mu_leads, 'barrier0': barrier0, 'Vdisx': Vdisx,
        'energies': energies, 'barrier_arr': barrier_arr, 'num_eigenvalues': 4,
        'eng_window_range': 3, 'conductance_flag': True, 'spectra_flag': True,
        'solver_type': 'cpu'
    }
    
    iter_data = (0, 0.5, 0.5) # i, mu, vz
    res = mp_module.worker_simulation_step(iter_data, static_params)
    assert res['i'] == 0
    assert 'Gmat' in res

def test_worker_step_gpu():
    if not hp.GPU_AVAILABLE:
        pytest.skip("GPU not available")
        
    t = 100.0
    mu = 0.5
    mu_n = 0.0
    gamma = 0.2
    Delta0 = 0.3
    alpha = 10.0
    Ls = 10
    mu_leads = 100.0
    barrier0 = 2.0
    Vdisx = np.zeros(Ls)
    energies = np.array([0.0, 0.1])
    barrier_arr = np.array([2.0, 5.0])
    
    static_params = {
        't': t, 'mu_n': mu_n, 'Delta0': Delta0, 'alpha': alpha, 'gamma': gamma,
        'V0': 1.0, 'qn': 20, 'Ln': 0, 'Lb': 2, 'Barrier_Height': barrier0,
        'Ls': Ls, 'mu_leads': mu_leads, 'barrier0': barrier0, 'Vdisx': Vdisx,
        'energies': energies, 'barrier_arr': barrier_arr, 'num_eigenvalues': 4,
        'eng_window_range': 3, 'conductance_flag': True, 'spectra_flag': True,
        'solver_type': 'gpu'
    }
    
    iter_data = (0, 0.5, 0.5)
    res = mp_module.worker_simulation_step(iter_data, static_params)
    assert res['i'] == 0
    assert 'Gmat' in res

if __name__ == "__main__":
    test_worker_step_cpu()
    if hp.GPU_AVAILABLE:
        test_worker_step_gpu()
