import pytest
import numpy as np
import sys
import os

# Add parent directory to path so that helpers can be imported when running pytest directly from the Tests directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.helpers as hp
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
    Delta function has max(idx) == min(idx), yielding minimal spatial span (0.0).
    """
    L = 100
    rho1 = np.zeros(L)
    rho1[50] = 1.0
    rho2 = np.zeros(L)
    
    assert hp.calc_weight_localization(rho1, rho2, 0.5) == 0.0
    assert hp.calc_weight_localization(rho1, rho2, 0.9) == 0.0

def test_weight_localization_uniform():
    """
    Uniform distribution has all sites suprathreshold, spanning (L - 1) / L.
    """
    L = 100
    rho1 = np.ones(L)
    rho2 = np.zeros(L)
    
    assert hp.calc_weight_localization(rho1, rho2, 0.9) == pytest.approx((L - 1) / L, abs=1e-4)
    assert hp.calc_weight_localization(rho1, rho2, 0.5) == pytest.approx((L - 1) / L, abs=1e-4)

def test_weight_localization_full():
    """
    Threshold of 1.0 should return 1.0.
    """
    L = 50
    rho1 = np.random.rand(L)
    rho2 = np.random.rand(L)
    
    assert hp.calc_weight_localization(rho1, rho2, 1.0) == 1.0

def test_weight_localization_legacy_parity():
    """
    Test parity of vectorized calc_weight_localization against the legacy
    threshold-stepping algorithm across multiple thresholds on Gaussian and multi-peak profiles.
    """
    def legacy_stepping(rho_M1, rho_M2, weight_threshold=0.9):
        step = 0.0001
        n = 0
        totrho = rho_M1 + rho_M2
        totsum = sum(totrho)
        wpct = 0.0
        rgn_pct = 0.0
        while wpct < weight_threshold:
            thresh = np.max(totrho) - step * n
            idx = np.where(totrho >= thresh)[0]
            wpct = np.sum(totrho[idx]) / totsum
            rgn_pct = (np.max(idx) - np.min(idx)) / len(totrho)
            n += 1
        return rgn_pct

    L = 200
    x = np.linspace(-10, 10, L)
    # Gaussian
    rho_gauss = np.exp(-0.2 * x**2)
    for th in [0.5, 0.7, 0.8, 0.9]:
        res_vec = hp.calc_weight_localization(rho_gauss, np.zeros(L), weight_threshold=th)
        res_leg = legacy_stepping(rho_gauss, np.zeros(L), weight_threshold=th)
        assert res_vec == pytest.approx(res_leg, abs=0.01)

    # Two peaks
    rho_two_peak = np.exp(-0.5 * (x - 5)**2) + np.exp(-0.5 * (x + 5)**2)
    for th in [0.5, 0.8, 0.9]:
        res_vec = hp.calc_weight_localization(rho_two_peak, np.zeros(L), weight_threshold=th)
        res_leg = legacy_stepping(rho_two_peak, np.zeros(L), weight_threshold=th)
        assert res_vec == pytest.approx(res_leg, abs=0.01)

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

def test_normalized_mzm_overlap():
    """
    Test normalized wave function amplitude overlap.
    """
    L = 100
    rho1 = np.zeros(L)
    rho2 = np.zeros(L)
    rho1[5] = 1.0
    rho2[95] = 1.0
    # Disjoint modes should have 0 overlap
    assert hp.calc_normalized_mzm_overlap(rho1, rho2) == 0.0

    # Identical modes
    assert hp.calc_normalized_mzm_overlap(rho1, rho1) > 0.0

def test_boundary_confinement():
    """
    Test normalized boundary confinement metric.
    """
    L = 100
    rho1 = np.zeros(L)
    rho2 = np.zeros(L)
    rho1[0] = 1.0
    rho2[-1] = 1.0
    conf = hp.calc_mzm_boundary_confinement(rho1, rho2, L)
    assert 0.0 <= conf <= 1.0
    assert conf > 0.9

def test_get_psiM_density_artificial_inversion():
    """
    Test that get_psiM_density strictly enforces rho_M1 as the Left mode
    and rho_M2 as the Right mode, even when the eigenvectors are artificially inverted
    such that naive linear combinations would assign the Right mode to gamma_1.
    """
    L = 50
    norbs = 4
    dim = L * norbs

    # Construct a Left-localized mode at site 5 and Right-localized mode at site 45
    mode_left = np.zeros(dim, dtype=complex)
    mode_left[5 * norbs : 5 * norbs + norbs] = 0.5  # sum |mode_left|^2 = 1.0

    mode_right = np.zeros(dim, dtype=complex)
    mode_right[45 * norbs : 45 * norbs + norbs] = 0.5  # sum |mode_right|^2 = 1.0

    # Normal case:
    # psi_plus = (mode_left + mode_right) / sqrt(2)
    # psi_minus = (mode_left - mode_right) / sqrt(2)
    evecs_normal = np.column_stack([
        (mode_left - mode_right) / np.sqrt(2),
        (mode_left + mode_right) / np.sqrt(2)
    ])
    evals = np.array([-1e-6, 1e-6])

    rho1, rho2, _ = hp.get_psiM_density(evals, evecs_normal)
    com1 = np.sum(np.arange(L) * rho1) / np.sum(rho1)
    com2 = np.sum(np.arange(L) * rho2) / np.sum(rho2)
    assert com1 < com2
    assert com1 == pytest.approx(5.0)
    assert com2 == pytest.approx(45.0)

    # Inverted case:
    # Swap role of left and right so naive projection produces Right mode for gamma_1:
    evecs_inverted = np.column_stack([
        (mode_right - mode_left) / np.sqrt(2),
        (mode_right + mode_left) / np.sqrt(2)
    ])

    rho1_inv, rho2_inv, _ = hp.get_psiM_density(evals, evecs_inverted)
    com1_inv = np.sum(np.arange(L) * rho1_inv) / np.sum(rho1_inv)
    com2_inv = np.sum(np.arange(L) * rho2_inv) / np.sum(rho2_inv)

    # The robust fix must have detected com1 > com2 and swapped them!
    assert com1_inv < com2_inv
    assert com1_inv == pytest.approx(5.0)
    assert com2_inv == pytest.approx(45.0)

def test_get_psiM_density_physical_topological():
    """
    Test get_psiM_density on a real topological Kwant Hamiltonian.
    In the topological phase (V_z > Delta), MZMs appear at the ends of the wire.
    rho_M1 must be localized at the left end, and rho_M2 at the right end.
    """
    t = 10.0
    mu = 0.0
    gamma = 0.5
    Delta0 = 1.0
    V_z = 2.0  # Well into topological regime (V_z > Delta0)
    alpha = 1.0
    Ls = 80

    Vdisx = np.zeros(Ls)
    syst = hp.build_system_closed(t, mu, gamma, Delta0, V_z, alpha, Ls, Vdisx=Vdisx, a=1)
    evals, evecs = hp.solve_ham(syst, k=2, solver_type='cpu')
    rho_M1, rho_M2, _ = hp.get_psiM_density(evals, evecs)

    com1 = np.sum(np.arange(Ls) * rho_M1) / np.sum(rho_M1)
    com2 = np.sum(np.arange(Ls) * rho_M2) / np.sum(rho_M2)

    assert com1 < com2
    assert com1 < Ls / 2  # Left half of wire
    assert com1 + com2 == pytest.approx(Ls - 1, abs=1e-3)  # Left-right inversion symmetry

def test_get_psiM_density_long_tail_defect():
    """
    Test that the composite mode assignment criterion correctly identifies
    a boundary mode at x=0 with a long exponential tail (xi=25 on L=100)
    as Left, even when a localized defect mode sits at x=20 (<x>_boundary = 22.6 > <x>_defect = 20.0),
    where CoM alone would fail.
    """
    L = 100
    norbs = 4
    dim = L * norbs

    x = np.arange(L)
    # Mode 1: Boundary mode at x=0 with xi=25
    psi_boundary = np.zeros(dim, dtype=complex)
    amp_b = np.exp(-x / 25.0)
    amp_b /= np.sqrt(np.sum(amp_b**2))
    for i in range(L):
        psi_boundary[i * norbs] = amp_b[i]

    # Mode 2: Defect mode at x=20
    psi_defect = np.zeros(dim, dtype=complex)
    amp_d = np.exp(-0.5 * ((x - 20) / 2.0)**2)
    amp_d /= np.sqrt(np.sum(amp_d**2))
    for i in range(L):
        psi_defect[i * norbs] = amp_d[i]

    # Deliberately mix such that naive projection assigns defect to gamma_1 and boundary to gamma_2
    evecs = np.column_stack([
        (psi_defect - psi_boundary) / np.sqrt(2),
        (psi_defect + psi_boundary) / np.sqrt(2)
    ])
    evals = np.array([-1e-6, 1e-6])

    rho1, rho2, _ = hp.get_psiM_density(evals, evecs)
    # rho1 MUST be the boundary mode (peaked at 0) and rho2 the defect (peaked at 20)
    assert np.argmax(rho1) == 0
    assert np.argmax(rho2) == 20

def test_disordered_topological_wire():
    """
    Test get_psiM_density and separability on a disordered Kwant closed wire (Vdis > 0).
    Verifies robust mode assignment and high separability in the topological phase with disorder.
    """
    t = 10.0
    mu = 0.0
    gamma = 0.5
    Delta0 = 1.0
    V_z = 2.0
    alpha = 1.0
    Ls = 80
    np.random.seed(42)
    Vdisx = np.random.normal(0.0, 0.3, Ls)

    syst = hp.build_system_closed(t, mu, gamma, Delta0, V_z, alpha, Ls, Vdisx=Vdisx, a=1)
    evals, evecs = hp.solve_ham(syst, k=2, solver_type='cpu')
    rho_M1, rho_M2, _ = hp.get_psiM_density(evals, evecs)

    com1 = np.sum(np.arange(Ls) * rho_M1) / np.sum(rho_M1)
    com2 = np.sum(np.arange(Ls) * rho_M2) / np.sum(rho_M2)

    # Strictly ordered Left and Right
    assert com1 < com2
    assert com1 < Ls / 2
    assert com2 > Ls / 2

    # High separability under moderate disorder
    s = hp.calc_mzm_separability(rho_M1, rho_M2)
    assert s > 0.75

def test_separability_delta_ends():
    """
    Test that delta peaks at wire ends give separability S = 1.0.
    """
    L = 100
    rho_left = np.zeros(L)
    rho_right = np.zeros(L)
    rho_left[0] = 1.0
    rho_right[-1] = 1.0

    s_cont = hp.calc_mzm_separability(rho_left, rho_right, continuous=True)
    s_disc = hp.calc_mzm_separability(rho_left, rho_right, continuous=False)

    assert s_cont == pytest.approx(1.0)
    assert s_disc == pytest.approx(1.0)

def test_separability_uniform():
    """
    Test that uniform distributions give separability S = 0.5
    for both even and odd wire lengths under continuous interpolation.
    """
    for L in [50, 51, 100, 101]:
        rho_left = np.ones(L)
        rho_right = np.ones(L)
        s = hp.calc_mzm_separability(rho_left, rho_right, continuous=True)
        assert s == pytest.approx(0.5, abs=1e-12)

def test_separability_disjoint_gaussians():
    """
    Test that disjoint Gaussians centered at opposite ends have separability ~ 1.0.
    """
    L = 100
    x = np.arange(L)
    rho_left = np.exp(-0.5 * ((x - 5) / 2.0)**2)
    rho_right = np.exp(-0.5 * ((x - 95) / 2.0)**2)

    s = hp.calc_mzm_separability(rho_left, rho_right)
    assert s == pytest.approx(1.0, abs=1e-5)

def test_separability_overlapping_modes():
    """
    Test separability on overlapping modes:
    1. Identical distributions (arbitrary profile) must yield exactly S = 0.5.
    2. Partially overlapping Gaussians must yield 0.5 < S < 1.0.
    """
    L = 100
    x = np.arange(L)

    # Arbitrary identical random distribution
    np.random.seed(42)
    rho_identical = np.random.uniform(0.1, 1.0, size=L)
    s_identical = hp.calc_mzm_separability(rho_identical, rho_identical, continuous=True)
    assert s_identical == pytest.approx(0.5, abs=1e-12)

    # Partially overlapping Gaussians
    rho_l_part = np.exp(-0.5 * ((x - 45) / 10.0)**2)
    rho_r_part = np.exp(-0.5 * ((x - 55) / 10.0)**2)
    s_part = hp.calc_mzm_separability(rho_l_part, rho_r_part)
    assert 0.5 < s_part < 1.0
    assert s_part == pytest.approx(0.6913, abs=0.01)

def test_separability_swapped_inputs():
    """
    Test behavior when modes are passed in reverse order (Right mode as rho_left, Left mode as rho_right).
    - With auto_orient=True (default): auto-detects inversion and recovers 1.0 (permutation invariant).
    - With auto_orient=False: strictly evaluates Left on left and Right on right, returning 0.0 for disjoint ends.
    """
    L = 100
    rho_left = np.zeros(L)
    rho_right = np.zeros(L)
    rho_left[0] = 1.0
    rho_right[-1] = 1.0

    # Default auto_orient=True recovers 1.0
    s_swapped_default = hp.calc_mzm_separability(rho_right, rho_left)
    assert s_swapped_default == pytest.approx(1.0)

    # Passing swapped inputs with explicit auto_orient=False
    s_swapped_strict = hp.calc_mzm_separability(rho_right, rho_left, auto_orient=False)
    assert s_swapped_strict == 0.0

def test_separability_distance_monotonicity():
    """
    Test that as separation distance between two Gaussian wave packets decreases
    from large separation to zero, separability S(d) decreases monotonically from 1.0 to 0.5.
    """
    L = 200
    x = np.arange(L)
    sigma = 3.0
    center_mid = 100

    distances = [120, 80, 50, 30, 20, 10, 5, 2, 0]
    separabilities = []

    for d in distances:
        x_L = center_mid - d / 2.0
        x_R = center_mid + d / 2.0
        rho_L = np.exp(-0.5 * ((x - x_L) / sigma)**2)
        rho_R = np.exp(-0.5 * ((x - x_R) / sigma)**2)
        s = hp.calc_mzm_separability(rho_L, rho_R)
        separabilities.append(s)

    # S at large separation is 1.0
    assert separabilities[0] == pytest.approx(1.0, abs=1e-5)
    # S at zero separation is exactly 0.5
    assert separabilities[-1] == pytest.approx(0.5, abs=1e-12)

    # Strictly monotonic non-increasing as distance decreases
    for i in range(len(separabilities) - 1):
        assert separabilities[i] >= separabilities[i + 1] - 1e-12

def test_separability_asymmetric_modes():
    """
    Test separability on highly asymmetric modes (one narrow sigma=1, one broad sigma=20).
    Verifies permutation invariance, return_cut partition coordinate, and rescaling.
    """
    L = 100
    x = np.arange(L)
    rho_narrow = np.exp(-0.5 * ((x - 10) / 1.0)**2)
    rho_broad = np.exp(-0.5 * ((x - 80) / 20.0)**2)

    s1 = hp.calc_mzm_separability(rho_narrow, rho_broad)
    s2 = hp.calc_mzm_separability(rho_broad, rho_narrow)

    # Permutation invariance via auto_orient=True default
    assert s1 == pytest.approx(s2)
    assert 0.5 < s1 <= 1.0

    # Test return_cut
    s_val, x_cut = hp.calc_mzm_separability(rho_narrow, rho_broad, return_cut=True)
    assert s_val == pytest.approx(s1)
    # Partition coordinate should fall between the two modes
    assert 10.0 < x_cut < 80.0

    # Test rescale
    s_rescaled = hp.calc_mzm_separability(rho_narrow, rho_broad, rescale=True)
    assert s_rescaled == pytest.approx(2.0 * s1 - 1.0)
    assert 0.0 < s_rescaled <= 1.0

def test_separability_edge_cases():
    """
    Test separability on unnormalized, zero, negative, and invalid inputs.
    """
    L = 50
    rho_l = np.zeros(L)
    rho_r = np.zeros(L)
    rho_l[0] = 500.0  # Large unnormalized weight
    rho_r[-1] = 0.002 # Tiny unnormalized weight

    # Unnormalized inputs should be scale invariant
    assert hp.calc_mzm_separability(rho_l, rho_r) == pytest.approx(1.0)

    # Zero densities
    assert hp.calc_mzm_separability(np.zeros(L), rho_r) == 0.0
    assert hp.calc_mzm_separability(rho_l, np.zeros(L)) == 0.0
    assert hp.calc_mzm_separability(np.zeros(L), np.zeros(L)) == 0.0

    # NaN or Inf inputs
    rho_nan = rho_l.copy()
    rho_nan[10] = np.nan
    assert hp.calc_mzm_separability(rho_nan, rho_r) == 0.0

    rho_inf = rho_r.copy()
    rho_inf[10] = np.inf
    assert hp.calc_mzm_separability(rho_l, rho_inf) == 0.0

    # Empty or mismatched
    assert hp.calc_mzm_separability([], []) == 0.0
    assert hp.calc_mzm_separability(np.ones(10), np.ones(20)) == 0.0

if __name__ == "__main__":
    pytest.main([__file__])
