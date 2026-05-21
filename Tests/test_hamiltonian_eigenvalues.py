import pytest
import numpy as np
import sys
import os

# Add parent directory to path so that helpers can be imported when running pytest directly from the Tests directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import helpers as hp

def check_eigenvalues(t, mu, gamma, Delta0, V_z, alpha, Ls, Vdisx):
    """
    Helper function to compare the eigenvalues of Kwant's build_system_closed
    and PDICalculator.get_full_Hamiltonian.
    """
    # 1. Build the Kwant system and extract its Hamiltonian
    syst_closed = hp.build_system_closed(
        t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=V_z, alpha=alpha, Ls=Ls, Vdisx=Vdisx
    )
    H_kwant = syst_closed.hamiltonian_submatrix(sparse=False)
    evals_kwant = np.sort(np.linalg.eigvalsh(H_kwant))
    
    # 2. Set up parameters for PDICalculator
    # Z renormalization factor is used to scale parameters matching the Kwant physics
    Z = Delta0 / (Delta0 + gamma)

    epsilon0 = 2 * t * np.cos(np.pi / (Ls + 1.0))
    
    # Shift chemical potential for PDI calculator to match the finite-size corrected band bottom
    mu_pdi = Z * (2 * t - epsilon0 + mu)
    
    # Scale other parameters by renormalization factor Z
    ts = Z * t
    alphas = Z * alpha
    gamma_pdi = Z * gamma
    V0 = Z
    gm = Z * V_z
    
    # Instantiate the PDI calculator
    calculator = hp.PDICalculator(ts, alphas, gamma_pdi, Ls, Vdisx, V0)
    H_pdi = calculator.get_full_Hamiltonian(gm=gm, mu=mu_pdi)
    evals_pdi = np.sort(np.linalg.eigvalsh(H_pdi))
    
    # 3. Assert eigenvalues are close
    # Use standard tolerance for double-precision float operations (1e-10 is very safe and strict)
    np.testing.assert_allclose(evals_kwant, evals_pdi, atol=1e-10, rtol=1e-10)

def test_hamiltonian_eigenvalues_clean():
    """
    Test eigenvalues for a clean wire (no disorder).
    """
    t = 1.0
    mu = 0.5
    gamma = 0.2
    Delta0 = 0.3
    V_z = 0.4
    alpha = 0.1
    Ls = 10
    Vdisx = np.zeros(Ls)
    
    check_eigenvalues(t, mu, gamma, Delta0, V_z, alpha, Ls, Vdisx)

def test_hamiltonian_eigenvalues_disordered():
    """
    Test eigenvalues for a wire with random disorder.
    """
    np.random.seed(42)
    t = 1.0
    mu = -0.2
    gamma = 0.15
    Delta0 = 0.25
    V_z = 0.5
    alpha = 0.3
    Ls = 15
    Vdisx = np.random.uniform(-0.5, 0.5, Ls)
    
    check_eigenvalues(t, mu, gamma, Delta0, V_z, alpha, Ls, Vdisx)

def test_hamiltonian_eigenvalues_sweep():
    """
    Sweep multiple parameter combinations to verify generality of eigenvalue equivalence.
    """
    np.random.seed(123)
    
    # Test combinations of different parameters
    for Ls in [5, 12]:
        for t in [0.8, 1.2]:
            for mu in [-0.5, 0.0, 0.5]:
                for V_z in [0.0, 0.3, 0.6]:
                    for gamma in [0.1, 0.25]:
                        Delta0 = 0.3
                        alpha = 0.15
                        # Include disorder
                        Vdisx = np.random.uniform(-0.3, 0.3, Ls)
                        
                        check_eigenvalues(t, mu, gamma, Delta0, V_z, alpha, Ls, Vdisx)

if __name__ == "__main__":
    pytest.main([__file__])
