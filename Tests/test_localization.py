import pytest
import numpy as np
import helpers as hp
from scipy.special import erf, erfinv

def test_weight_localization_gaussian():
    """
    Verify localization metric against an analytic Gaussian distribution.
    f(x) = exp(-a * x^2)
    Area within [-R, R] is (sqrt(pi)/sqrt(a)) * erf(sqrt(a)*R)
    Total area is sqrt(pi)/sqrt(a)
    Weight fraction = erf(sqrt(a)*R)
    """
    # Use a wider Gaussian to reduce the relative impact of the +/- 1 site error
    a = 0.0005
    L = 2000
    x = np.linspace(-L/2, L/2, L)
    
    # Single Gaussian centered at 0
    rho = np.exp(-a * x**2)
    
    # We want weight threshold 0.9
    threshold = 0.9
    
    # Solve erf(sqrt(a)*R) = 0.9 for R
    # Using scipy.special.erfinv which uses high-precision Cody-style approximations
    R_analytic = erfinv(threshold) / np.sqrt(a)
    
    # Expected number of sites is 2 * R_analytic (since it's [-R, R])
    expected_fraction = (2 * R_analytic) / L
    
    # Numerical calculation
    calc_fraction = hp.calc_weight_localization(rho, np.zeros_like(rho), weight_threshold=threshold)
    
    print(f"Analytic Expected Fraction: {expected_fraction:.4f}")
    print(f"Numerical Calculated Fraction: {calc_fraction:.4f}")
    
    # A single site error in a 2000 site wire is 1/2000 = 0.0005
    # We check within 1% absolute tolerance
    assert calc_fraction == pytest.approx(expected_fraction, abs=0.01)

def test_weight_localization_delta():
    """
    Delta function should give minimal localization (1/L).
    """
    L = 100
    rho1 = np.zeros(L)
    rho1[50] = 1.0
    rho2 = np.zeros(L)
    
    # For any threshold > 0, should only need 1 site
    assert hp.calc_weight_localization(rho1, rho2, 0.5) == 1.0/L
    assert hp.calc_weight_localization(rho1, rho2, 0.9) == 1.0/L

def test_weight_localization_uniform():
    """
    Uniform distribution should give localization equal to threshold.
    """
    L = 100
    rho1 = np.ones(L)
    rho2 = np.zeros(L)
    
    # Should need 90 sites for 0.9 threshold
    assert hp.calc_weight_localization(rho1, rho2, 0.9) == pytest.approx(0.9, abs=0.01)
    assert hp.calc_weight_localization(rho1, rho2, 0.5) == pytest.approx(0.5, abs=0.01)

def test_weight_localization_full():
    """
    Threshold of 1.0 should return 1.0.
    """
    L = 50
    rho1 = np.random.rand(L)
    rho2 = np.random.rand(L)
    
    assert hp.calc_weight_localization(rho1, rho2, 1.0) == 1.0

def test_overlap_integral():
    """
    Test that overlap integral correctly identifies disjoint vs overlapping densities.
    """
    L = 100
    rho1 = np.zeros(L)
    rho2 = np.zeros(L)
    
    # Case 1: Perfectly disjoint
    rho1[:50] = 1.0 / 50
    rho2[50:] = 1.0 / 50
    assert hp.calc_overlap(rho1, rho2) == 0.0
    
    # Case 2: Identical (maximum overlap for normalized densities)
    assert hp.calc_overlap(rho1, rho1) == np.sum(rho1**2)
    
    # Case 3: Partial overlap
    rho_overlap = np.zeros(L)
    rho_overlap[40:60] = 1.0 / 20
    # Overlap is only in region [40, 50)
    # rho1 is 1/50, rho_overlap is 1/20
    # expected = 10 * (1/50 * 1/20) = 10 / 1000 = 0.01
    assert hp.calc_overlap(rho1, rho_overlap) == pytest.approx(0.01)

if __name__ == "__main__":
    pytest.main([__file__])
