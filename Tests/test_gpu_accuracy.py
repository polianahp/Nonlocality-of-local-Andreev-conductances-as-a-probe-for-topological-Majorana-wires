import pytest
import numpy as np
import kwant
import helpers as hp

def test_gpu_smatrix_accuracy():
    # Setup small system parameters
    t = 100.0
    mu = 0.5
    mu_n = 0.0
    gamma = 0.2
    Delta0 = 0.3
    V_z = 0.5
    alpha = 10.0
    Ln = 0
    Lb = 2
    Ls = 10
    mu_leads = 100.0
    barrier_l = 2.0
    barrier_r = 2.0
    Vdisx = np.zeros(Ls)
    
    # Build system
    syst = hp.build_system(t, mu, mu_n, gamma, Delta0, V_z, alpha, Ln, Lb, Ls, mu_leads, barrier_l, barrier_r, Vdisx)
    
    energy = 0.05
    
    # CPU S-matrix
    s_cpu = kwant.smatrix(syst, energy)
    
    # GPU S-matrix
    if not hp.GPU_AVAILABLE:
        pytest.skip("GPU (CuPy) not available for testing.")
        
    s_gpu = hp.get_gpu_smatrix(syst, energy)
    
    # Compare transmissions
    # Just checking some elements
    t_cpu = s_cpu.transmission((1, 0), (0, 0))
    t_gpu = s_gpu.transmission((1, 0), (0, 0))
    
    print(f"Transmission (CPU): {t_cpu}")
    print(f"Transmission (GPU): {t_gpu}")
    
    np.testing.assert_allclose(t_cpu, t_gpu, atol=1e-10)

    # Compare full data matrix
    np.testing.assert_allclose(s_cpu.data, s_gpu.data, atol=1e-10)

if __name__ == "__main__":
    test_gpu_smatrix_accuracy()
