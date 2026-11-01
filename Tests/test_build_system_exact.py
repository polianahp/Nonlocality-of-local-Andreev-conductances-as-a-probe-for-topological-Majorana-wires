import pytest
import numpy as np
import kwant
import sys
import os
from scipy.optimize import brentq
import scipy.sparse.linalg as sla

# Add parent directory to path so that helpers can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.helpers as hp
from Tests.test_barrier_calibration import compute_1d_analytic_transmission


# ==============================================================================
# T1: test_sigma_al_bcs_analytic
# ==============================================================================
def test_sigma_al_bcs_analytic():
    """
    Subgap: Hermitian and equal to the hand-written formula.
    Above gap (+/- E): Gamma = i(Sigma - Sigma^dagger) has eigenvalues
    2*gamma*(|E| -/+ Delta0) / sqrt(E^2 - Delta0^2) (>= 0).
    eta = 1e-9 matches eta = 0 to 1e-6.
    Raises at |E| = Delta0 when eta = 0.
    Sigma(0) = +gamma * tau_x.
    """
    gamma = 0.2
    Delta0 = 0.3
    I4 = np.eye(4, dtype=complex)
    tx = np.kron(hp.sigma_x, hp.sigma_0)

    # 1. Zero energy: Sigma(0) = +gamma * tau_x
    sig_zero = hp.sigma_al_bcs(0.0, gamma, Delta0, eta=0.0)
    expected_zero = gamma * tx
    np.testing.assert_allclose(sig_zero, expected_zero, atol=1e-15)

    # 2. Subgap: |omega| < Delta0
    for omega in [-0.25, -0.1, 0.05, 0.2]:
        sig_sub = hp.sigma_al_bcs(omega, gamma, Delta0, eta=0.0)
        # Must be strictly Hermitian
        np.testing.assert_allclose(sig_sub, sig_sub.conj().T, atol=1e-15)
        # Analytical formula: -gamma / sqrt(Delta0^2 - omega^2) * (omega * I4 - Delta0 * tx)
        s_val = np.sqrt(Delta0**2 - omega**2)
        expected_sub = -gamma / s_val * (omega * I4 - Delta0 * tx)
        np.testing.assert_allclose(sig_sub, expected_sub, atol=1e-15)

    # 3. Above gap: |omega| > Delta0 for both positive and negative E
    for omega in [0.5, -0.5, 0.35, -0.35]:
        sig_above = hp.sigma_al_bcs(omega, gamma, Delta0, eta=0.0)
        Gamma = 1j * (sig_above - sig_above.conj().T)
        eigs_gamma = np.sort(np.linalg.eigvalsh(Gamma))
        
        # Eigenvalues must be 2*gamma*(|omega| -/+ Delta0) / sqrt(omega^2 - Delta0^2)
        denom = np.sqrt(omega**2 - Delta0**2)
        eig1 = 2.0 * gamma * (abs(omega) - Delta0) / denom
        eig2 = 2.0 * gamma * (abs(omega) + Delta0) / denom
        expected_eigs = np.sort([eig1, eig1, eig2, eig2])
        
        np.testing.assert_allclose(eigs_gamma, expected_eigs, atol=1e-14)
        assert np.all(eigs_gamma >= 0.0), f"Gamma has negative eigenvalues at omega={omega}"

    # 4. Dynes broadening: eta = 1e-9 matches eta = 0 to 1e-6
    for omega in [0.1, 0.5, -0.1, -0.5]:
        sig_clean = hp.sigma_al_bcs(omega, gamma, Delta0, eta=0.0)
        sig_dynes = hp.sigma_al_bcs(omega, gamma, Delta0, eta=1e-9)
        np.testing.assert_allclose(sig_dynes, sig_clean, atol=1e-6)

    # 5. Raises at |omega| == Delta0 when eta == 0
    with pytest.raises(ValueError):
        hp.sigma_al_bcs(Delta0, gamma, Delta0, eta=0.0)
    with pytest.raises(ValueError):
        hp.sigma_al_bcs(-Delta0, gamma, Delta0, eta=0.0)

    # 6. Parameter validation
    with pytest.raises(ValueError):
        hp.sigma_al_bcs(0.1, -0.2, Delta0)
    with pytest.raises(ValueError):
        hp.sigma_al_bcs(0.1, gamma, -0.3)
    with pytest.raises(ValueError):
        hp.sigma_al_bcs(0.1, gamma, 0.0)
    with pytest.raises(ValueError):
        hp.sigma_al_bcs(0.1, gamma, Delta0, eta=-1e-4)


# ==============================================================================
# T2: test_exact_self_energy_matrix
# ==============================================================================
def test_exact_self_energy_matrix():
    """
    diff = H_exact(omega=E) - H_exact(gamma=0, omega=E), never sliced or modified,
    equals the full block-diagonal M (Sigma on SC sites, zero elsewhere), atol=1e-12,
    for E in {0.05, 0.5} and for both closed and open systems.
    """
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3
    V_z = 0.4
    alpha = 1.4
    Ls = 300
    Lb = 3
    mu = 0.5

    for E in [0.05, 0.5]:
        sig_E = hp.sigma_al_bcs(E, gamma, Delta0)

        # 1. Closed system
        syst_closed_exact = hp.build_system_exact(
            t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=V_z, alpha=alpha, Ls=Ls, closed=True
        )
        syst_closed_zero = hp.build_system_exact(
            t=t, mu=mu, gamma=0.0, Delta0=Delta0, V_z=V_z, alpha=alpha, Ls=Ls, closed=True
        )
        H_closed_exact = syst_closed_exact.hamiltonian_submatrix(params=dict(omega=E), sparse=False)
        H_closed_zero = syst_closed_zero.hamiltonian_submatrix(params=dict(omega=E), sparse=False)
        diff_closed = H_closed_exact - H_closed_zero

        M_closed = np.kron(np.eye(Ls, dtype=complex), sig_E)
        np.testing.assert_allclose(
            diff_closed, M_closed, atol=1e-12, rtol=1e-12,
            err_msg=f"Closed wire self-energy matrix diff mismatch at E={E}"
        )

        # 2. Open system
        syst_open_exact = hp.build_system_exact(
            t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=V_z, alpha=alpha, Ls=Ls, Lb=Lb,
            closed=False, barrier_l=1.5, barrier_r=2.5
        )
        syst_open_zero = hp.build_system_exact(
            t=t, mu=mu, gamma=0.0, Delta0=Delta0, V_z=V_z, alpha=alpha, Ls=Ls, Lb=Lb,
            closed=False, barrier_l=1.5, barrier_r=2.5
        )
        H_open_exact = syst_open_exact.hamiltonian_submatrix(params=dict(omega=E), sparse=False)
        H_open_zero = syst_open_zero.hamiltonian_submatrix(params=dict(omega=E), sparse=False)
        diff_open = H_open_exact - H_open_zero

        N_total = 2 * Lb + Ls
        M_open = np.zeros((4 * N_total, 4 * N_total), dtype=complex)
        for i in range(Lb, Lb + Ls):
            M_open[4 * i : 4 * i + 4, 4 * i : 4 * i + 4] = sig_E

        np.testing.assert_allclose(
            diff_open, M_open, atol=1e-12, rtol=1e-12,
            err_msg=f"Open wire self-energy matrix diff mismatch at E={E}"
        )


# ==============================================================================
# T3: test_bare_hoppings_everywhere
# ==============================================================================
def test_bare_hoppings_everywhere():
    """
    Open system: every bond, including both barrier <-> SC interfaces,
    equals T exactly (no Z, no sqrt(Z)).
    """
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3
    alpha = 1.4
    Ls = 300
    Lb = 3
    
    syst = hp.build_system_exact(
        t=t, mu=0.5, gamma=gamma, Delta0=Delta0, V_z=0.4, alpha=alpha,
        Ls=Ls, Lb=Lb, closed=False
    )
    H = syst.hamiltonian_submatrix(params=dict(omega=0.05), sparse=False)

    tzs0 = np.kron(hp.sigma_z, hp.sigma_0)
    tzsy = np.kron(hp.sigma_z, hp.sigma_y)
    T_expected = -t * tzs0 + 0.5j * alpha * tzsy

    # Spatial order: sites 0..2*Lb+Ls-1
    # Check all adjacent hopping blocks H[site, site-1]
    N_total = 2 * Lb + Ls
    for i in range(1, N_total):
        hop_block = H[4 * i : 4 * i + 4, 4 * (i - 1) : 4 * i]
        np.testing.assert_allclose(
            hop_block, T_expected, atol=1e-14, rtol=1e-14,
            err_msg=f"Hopping at bond ({i}, {i-1}) deviates from bare hopping T!"
        )


# ==============================================================================
# T4: test_omega_equals_energy
# ==============================================================================
def test_omega_equals_energy(monkeypatch):
    """
    Spy on kwant.smatrix with monkeypatch and assert params['omega'] == energy
    and check_hermiticity is False.
    Calling legacy calc_conductance_matrix on the new system raises TypeError (missing omega).
    """
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3
    alpha = 1.4
    Ls = 50
    Lb = 3

    syst = hp.build_system_exact(
        t=t, mu=0.5, gamma=gamma, Delta0=Delta0, V_z=0.4, alpha=alpha,
        Ls=Ls, Lb=Lb, closed=False
    )

    recorded_calls = []
    real_smatrix = kwant.smatrix

    def spy_smatrix(fsyst, energy=0.0, **kwargs):
        recorded_calls.append({'energy': energy, 'kwargs': kwargs})
        return real_smatrix(fsyst, energy=energy, **kwargs)

    monkeypatch.setattr(kwant, 'smatrix', spy_smatrix)

    E_test = 0.25
    hp.smatrix_exact(syst, E_test)

    assert len(recorded_calls) == 1
    call = recorded_calls[0]
    assert call['energy'] == E_test
    assert call['kwargs'].get('check_hermiticity') is False
    assert call['kwargs'].get('params') == {'omega': E_test}

    # Calling legacy calc_conductance_matrix must fail loudly
    with pytest.raises((TypeError, kwant.UserCodeError)):
        hp.calc_conductance_matrix(syst, eng=E_test)


# ==============================================================================
# T5: test_single_mode_analytic_oracle
# ==============================================================================
def test_single_mode_analytic_oracle():
    """
    Clean closed wire with alpha = 0: modes decouple.
    xi_n = (eps_ref - mu) - 2t * cos(n * pi / (Ls + 1)), spin s = +/- 1.
    Each 2x2 root of det[(E - s*V_z) - xi_n*tau_z - Sigma(E)] = 0,
    found with Brent in pure numpy (no Kwant), must match solve_exact_spectrum to 1e-9.
    Cover V_z = 0 (multiplicity 2) and V_z = 0.6.
    """
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3
    alpha = 0.0
    Ls = 300
    mu = 0.3

    for V_z in [0.0, 0.6]:
        fsyst = hp.build_system_exact(
            t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=V_z, alpha=alpha,
            Ls=Ls, closed=True
        )

        # Analytical 1D mode roots:
        # eps_ref = 2t
        n_vals = np.arange(1, Ls + 1)
        xi_n = (2.0 * t - mu) - 2.0 * t * np.cos(n_vals * np.pi / (Ls + 1.0))

        # Root finder for single mode (xi, s):
        # det[ E - H_mode(E) ] = 0
        # where H_mode(E) = xi * tau_z + s * V_z * tau_0 + Sigma(E)
        # and Sigma(E) = -gamma / sqrt(Delta0^2 - E^2) * (E * tau_0 - Delta0 * tau_x)
        # det = (E - s*V_z + gamma*E / sqrt(Delta0^2 - E^2))^2 - xi^2 - (gamma*Delta0 / sqrt(Delta0^2 - E^2))^2
        analytic_roots = []

        def det_mode(E, xi, s):
            s_val = np.sqrt(Delta0**2 - E**2)
            diag_term = E - s * V_z + gamma * E / s_val
            offdiag_term = gamma * Delta0 / s_val
            return diag_term**2 - xi**2 - offdiag_term**2

        for xi in xi_n:
            for s in [1.0, -1.0]:
                # Check for roots in (0, Delta0 - 1e-6)
                # Note: f(E) has roots where one of the two eigenvalues of H_mode(E) equals E.
                # Since H_mode(E) eigenvalues decrease strictly with E, each branch has at most one root.
                # Evaluate at endpoints:
                E_a = 1e-12
                E_b = Delta0 - 1e-6
                
                # We can track the two eigenvalue branches:
                # lam_pm(E) = s * V_z - gamma * E / sqrt(Delta0^2 - E^2) +/- sqrt(xi^2 + (gamma*Delta0 / sqrt(Delta0^2 - E^2))^2)
                for sign in [+1.0, -1.0]:
                    f = lambda E: (
                        s * V_z - gamma * E / np.sqrt(Delta0**2 - E**2)
                        + sign * np.sqrt(xi**2 + (gamma * Delta0 / np.sqrt(Delta0**2 - E**2))**2)
                        - E
                    )
                    fa = f(E_a)
                    fb = f(E_b)
                    if fa * fb < 0:
                        r = brentq(f, E_a, E_b, xtol=1e-13)
                        analytic_roots.append(float(r))

        analytic_roots = np.sort(analytic_roots)
        # Take lowest 8 roots
        analytic_lowest = analytic_roots[:8]

        # Kwant exact solver
        res = hp.solve_exact_spectrum(fsyst, n_states=8, gamma=gamma, Delta0=Delta0)
        kwant_energies = res['energies']

        np.testing.assert_allclose(
            kwant_energies, analytic_lowest, atol=1e-9, rtol=1e-9,
            err_msg=f"Analytical single-mode oracle mismatch at V_z={V_z}"
        )


# ==============================================================================
# T6: test_sturm_count_matches_dense
# ==============================================================================
def test_sturm_count_matches_dense():
    """
    Disordered wire, 10 random E in (-Delta0, Delta0):
    sturm_count_exact equals the dense eigenvalue count of H(E) - E*I < 0 exactly.
    """
    rng = np.random.RandomState(42)
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3
    alpha = 1.4
    Ls = 300
    mu = 0.3
    V_z = 0.6
    Vdisx = rng.uniform(-0.3, 0.3, size=Ls)

    fsyst = hp.build_system_exact(
        t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=V_z, alpha=alpha,
        Ls=Ls, Vdisx=Vdisx, closed=True
    )

    test_energies = rng.uniform(-0.28, 0.28, size=10)
    for Eval in test_energies:
        s_count = hp.sturm_count_exact(fsyst, Eval, Delta0=Delta0)
        H_dense = fsyst.hamiltonian_submatrix(params=dict(omega=Eval), sparse=False)
        d_count = int(np.sum(np.linalg.eigvalsh(H_dense) < Eval))
        assert s_count == d_count, f"Sturm count ({s_count}) != dense count ({d_count}) at E={Eval}"


# ==============================================================================
# T7: test_spectrum_completeness_and_residuals
# ==============================================================================
def test_spectrum_completeness_and_residuals():
    """
    10 random (mu, V_z) pairs (seed 42; 5 clean, 5 disordered):
    total states returned equals N(E_hi) - N(0+); every residual < 1e-9; energies sorted.
    """
    rng = np.random.RandomState(42)
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3
    alpha = 1.4
    Ls = 300

    mu_samples = rng.uniform(-0.5, 1.5, size=10)
    Vz_samples = rng.uniform(0.0, 0.8, size=10)

    for i in range(10):
        mu = float(mu_samples[i])
        V_z = float(Vz_samples[i])
        Vdisx = None if i < 5 else rng.uniform(-0.3, 0.3, size=Ls)

        fsyst = hp.build_system_exact(
            t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=V_z, alpha=alpha,
            Ls=Ls, Vdisx=Vdisx, closed=True
        )

        n_target = 10
        res = hp.solve_exact_spectrum(fsyst, n_states=n_target, gamma=gamma, Delta0=Delta0)

        energies = res['energies']
        residuals = res['residuals']

        # 1. Residuals < 1e-9
        assert np.all(residuals < 1e-9), f"Pair {i}: Residuals exceeded 1e-9: max={np.max(residuals)}"

        # 2. Energies strictly sorted
        assert np.all(np.diff(energies) >= -1e-12), f"Pair {i}: Energies not sorted!"

        # 3. Completeness: total states returned equals N(E_hi) - N(0+) or n_target
        if not res['truncated']:
            assert len(energies) == n_target
        else:
            N0 = hp.sturm_count_exact(fsyst, 1e-14, Delta0=Delta0)
            Nmax = hp.sturm_count_exact(fsyst, Delta0 * (1 - 1e-9), Delta0=Delta0)
            assert len(energies) == Nmax - N0


# ==============================================================================
# T8: test_particle_hole_symmetry_of_roots
# ==============================================================================
def test_particle_hole_symmetry_of_roots():
    """
    Number of roots in (0, E) equals number in (-E, 0), via sturm_count_exact,
    at 5 values of E.
    """
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3
    alpha = 1.4
    Ls = 300
    mu = 0.3
    V_z = 0.5

    fsyst = hp.build_system_exact(
        t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=V_z, alpha=alpha,
        Ls=Ls, closed=True
    )

    for E_pos in [0.05, 0.10, 0.18, 0.22, 0.27]:
        pos_roots = hp.sturm_count_exact(fsyst, E_pos, Delta0=Delta0) - hp.sturm_count_exact(fsyst, 1e-13, Delta0=Delta0)
        neg_roots = hp.sturm_count_exact(fsyst, -1e-13, Delta0=Delta0) - hp.sturm_count_exact(fsyst, -E_pos, Delta0=Delta0)
        assert pos_roots == neg_roots, (
            f"Particle-hole asymmetry in root count at E={E_pos}: pos={pos_roots}, neg={neg_roots}"
        )


# ==============================================================================
# T9: test_valid_regime_eigenvalue_agreement_with_legacy
# ==============================================================================
def test_valid_regime_eigenvalue_agreement_with_legacy():
    """
    (a) Large parent gap (Delta0 = 30.0 meV, gamma = 0.2 meV, (E/Delta0)^2 < 1e-4)
        across 10 random (mu, V_z) pairs (5 clean, 5 disordered, Ls=300),
        comparing lowest 10 positive eigenvalues via assert_allclose(atol=2e-5, rtol=1e-4).
    (b) Physical strong coupling (Delta0 = 0.3 meV, gamma = 0.2 meV, Z = 0.6)
        in topological regime (E < 0.03 meV = 0.1*Delta0, clean & disordered Ls=300),
        asserting near-zero MZM eigenvalues agree to atol=1e-5 and subgap states below 0.1*Delta0
        agree via assert_allclose(atol=1e-3, rtol=1e-2).
    """
    rng = np.random.RandomState(42)
    t = 16.529
    alpha = 1.4
    Ls = 300
    eps0 = 2.0 * t * np.cos(np.pi / (Ls + 1.0))

    # --- Part (a): Large parent gap ---
    Delta0_large = 30.0
    gamma_large = 0.2
    mu_samples = rng.uniform(-0.5, 1.5, size=10)
    Vz_samples = rng.uniform(0.0, 0.8, size=10)

    for i in range(10):
        mu = float(mu_samples[i])
        V_z = float(Vz_samples[i])
        Vdisx = None if i < 5 else rng.uniform(-0.3, 0.3, size=Ls)

        # Exact system
        syst_exact = hp.build_system_exact(
            t=t, mu=mu, gamma=gamma_large, Delta0=Delta0_large, V_z=V_z, alpha=alpha,
            Ls=Ls, Vdisx=Vdisx, closed=True
        )
        res_exact = hp.solve_exact_spectrum(syst_exact, n_states=10, gamma=gamma_large, Delta0=Delta0_large)
        exact_eigs = res_exact['energies']

        # Legacy system: pass mu_closed = mu - (2t - eps0)
        mu_closed = mu - (2.0 * t - eps0)
        syst_legacy = hp.build_system_closed(
            t=t, mu=mu_closed, gamma=gamma_large, Delta0=Delta0_large, V_z=V_z, alpha=alpha,
            Ls=Ls, Vdisx=Vdisx if Vdisx is not None else np.zeros(Ls)
        )
        H_legacy = syst_legacy.hamiltonian_submatrix(sparse=False)
        eigs_all = np.sort(np.linalg.eigvalsh(H_legacy))
        legacy_pos = eigs_all[eigs_all > 1e-12][:10]

        np.testing.assert_allclose(
            exact_eigs, legacy_pos, atol=2e-5, rtol=1e-4,
            err_msg=f"Part (a) mismatch at pair {i}: mu={mu}, Vz={V_z}"
        )

    # --- Part (b): Physical strong coupling ---
    Delta0_phys = 0.3
    gamma_phys = 0.2
    mu_phys = 0.3
    V_z_phys = 0.6  # Topological regime

    for disordered in [False, True]:
        Vdisx = rng.uniform(-0.2, 0.2, size=Ls) if disordered else None

        syst_exact = hp.build_system_exact(
            t=t, mu=mu_phys, gamma=gamma_phys, Delta0=Delta0_phys, V_z=V_z_phys, alpha=alpha,
            Ls=Ls, Vdisx=Vdisx, closed=True
        )
        res_exact = hp.solve_exact_spectrum(syst_exact, n_states=6, gamma=gamma_phys, Delta0=Delta0_phys)
        exact_eigs = res_exact['energies']

        mu_closed = mu_phys - (2.0 * t - eps0)
        syst_legacy = hp.build_system_closed(
            t=t, mu=mu_closed, gamma=gamma_phys, Delta0=Delta0_phys, V_z=V_z_phys, alpha=alpha,
            Ls=Ls, Vdisx=Vdisx if Vdisx is not None else np.zeros(Ls)
        )
        H_legacy = syst_legacy.hamiltonian_submatrix(sparse=False)
        eigs_all = np.sort(np.linalg.eigvalsh(H_legacy))
        legacy_pos = eigs_all[eigs_all > 0]

        # 1. MZM near-zero eigenvalue agrees to atol=1e-5
        assert abs(exact_eigs[0] - legacy_pos[0]) < 1e-5, (
            f"MZM eigenvalue mismatch: exact={exact_eigs[0]}, legacy={legacy_pos[0]}"
        )

        # 2. States below 0.1 * Delta0 (0.03 meV)
        sub_mask_exact = exact_eigs < 0.03
        sub_mask_legacy = legacy_pos[:len(exact_eigs)] < 0.03
        mask = sub_mask_exact & sub_mask_legacy
        if np.any(mask):
            np.testing.assert_allclose(
                exact_eigs[mask], legacy_pos[:len(exact_eigs)][mask],
                atol=1e-3, rtol=1e-2,
                err_msg=f"Subgap states below 0.1*Delta0 mismatch (disordered={disordered})"
            )


# ==============================================================================
# T10: test_renormalization_breakdown_near_gap_edge
# ==============================================================================
def test_renormalization_breakdown_near_gap_edge():
    """
    Test the limit where the low-energy static renormalization approximation breaks down
    and diverges from the exact frequency-dependent Green's function pole solution.
    (Kept unchanged from existing test).
    """
    Delta0 = 0.3  # meV
    gamma = 0.2   # meV
    Z = Delta0 / (Delta0 + gamma)

    def E_renorm(eps):
        return float(np.sqrt((Z * eps)**2 + (Z * gamma)**2))

    def exact_det(E, eps):
        denom = np.sqrt(Delta0**2 - E**2)
        sig_0 = gamma * E / denom
        sig_x = gamma * Delta0 / denom
        return float((E + sig_0)**2 - eps**2 - sig_x**2)

    def E_exact(eps):
        return float(brentq(exact_det, 0.0, Delta0 - 1e-6, args=(eps,)))

    # 1. Low-energy valid regime (eps = 0.05 meV << Delta0)
    e_renorm_low = E_renorm(0.05)
    e_exact_low = E_exact(0.05)
    assert abs(e_renorm_low - e_exact_low) < 0.01

    # 2. Breakdown regime near/above gap edge (eps = 0.50 meV > Delta0)
    e_renorm_high = E_renorm(0.50)
    e_exact_high = E_exact(0.50)

    assert e_renorm_high > Delta0
    assert e_exact_high < Delta0
    discrepancy = abs(e_renorm_high - e_exact_high)
    assert discrepancy > 0.05


# ==============================================================================
# T11: test_flux_balance_above_gap
# ==============================================================================
def test_flux_balance_above_gap():
    """
    E in {0.35, 0.5}, V_z in {0, 0.4}:
    for every incoming mode m, 1 - sum_n |S_nm|^2 equals psi_m^dagger Gamma psi_m
    summed over SC sites (using kwant.wave_function), atol=1e-8.
    """
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3
    alpha = 1.4
    Ls = 300
    Lb = 3
    mu = 0.5

    for E in [0.35, 0.5]:
        for V_z in [0.0, 0.4]:
            fsyst = hp.build_system_exact(
                t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=V_z, alpha=alpha,
                Ls=Ls, Lb=Lb, barrier_l=2.0, barrier_r=2.0, closed=False
            )

            sm = hp.smatrix_exact(fsyst, E)
            S = sm.data
            loss = 1.0 - np.sum(np.abs(S)**2, axis=0)  # Unitarity deficit per incoming mode

            wf = kwant.wave_function(fsyst, energy=E, params=dict(omega=E), check_hermiticity=False)
            Sig = hp.sigma_al_bcs(E, gamma, Delta0)
            Gamma = 1j * (Sig - Sig.conj().T)

            # SC sites are located at x in [Lb, Lb + Ls)
            sc_site_indices = [idx for idx, s in enumerate(fsyst.sites) if Lb <= s.tag[0] < Lb + Ls]

            absorbed = []
            for lead in (0, 1):
                psi = wf(lead)
                n_modes = psi.shape[0]
                for m in range(n_modes):
                    abs_m = 0.0
                    for site_idx in sc_site_indices:
                        v = psi[m, 4 * site_idx : 4 * site_idx + 4]
                        abs_m += np.real(v.conj() @ Gamma @ v)
                    absorbed.append(abs_m)

            absorbed = np.array(absorbed)
            np.testing.assert_allclose(
                loss, absorbed, atol=1e-8,
                err_msg=f"Flux balance deficit mismatch at E={E}, V_z={V_z}"
            )


# ==============================================================================
# T12: test_subgap_open_system_is_hermitian_and_unitary
# ==============================================================================
def test_subgap_open_system_is_hermitian_and_unitary():
    """
    E = 0.05: kwant.smatrix(..., check_hermiticity=True) does not raise,
    and every column of S has unit norm to 1e-10.
    """
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3
    alpha = 1.4
    Ls = 300
    Lb = 3
    mu = 0.5
    E_sub = 0.05

    fsyst = hp.build_system_exact(
        t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=0.4, alpha=alpha,
        Ls=Ls, Lb=Lb, barrier_l=1.0, barrier_r=1.0, closed=False
    )

    # Calling kwant.smatrix with check_hermiticity=True must NOT raise
    sm = kwant.smatrix(fsyst, energy=E_sub, params=dict(omega=E_sub), check_hermiticity=True)
    S = sm.data
    col_norms = np.sum(np.abs(S)**2, axis=0)

    np.testing.assert_allclose(col_norms, 1.0, atol=1e-10)


# ==============================================================================
# T13: test_above_gap_absorption_decoupling_symmetric
# ==============================================================================
def test_above_gap_absorption_decoupling_symmetric():
    """
    Test that the physical parent-Al self-energy Sigma_Al^R(E=0.5 meV) added to SC sites
    decouples the left junction conductance G_LL from the right barrier U_R,
    AND decouples the right junction conductance G_RR from the left barrier U_L on a 3-micron wire (Ls=300).
    """
    t = 16.529
    mu = 0.5
    gamma = 0.2
    Delta0 = 0.3
    V_z = 0.0
    alpha = 1.4
    Ln = 0
    Lb = 3
    Ls = 300
    mu_leads = t
    E = 0.5

    # 1. Left junction decoupling: vary U_R in {0.0, 5.0, 50.0}
    g_ll_values = []
    ur_values = [0.0, 5.0, 50.0]
    for ur in ur_values:
        syst = hp.build_system_exact(
            t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=V_z,
            alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
            barrier_l=2.0, barrier_r=ur, closed=False
        )
        G_mat = hp.calc_conductance_matrix_exact(syst, E=E)
        g_ll_values.append(G_mat[0, 0])

    g_ll_arr = np.array(g_ll_values)
    spread_l = np.max(g_ll_arr) - np.min(g_ll_arr)
    assert spread_l < 1e-6, f"Left conductance depends on right barrier! spread = {spread_l}"

    # 2. Right junction decoupling: vary U_L in {0.0, 5.0, 50.0}
    g_rr_values = []
    ul_values = [0.0, 5.0, 50.0]
    for ul in ul_values:
        syst = hp.build_system_exact(
            t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=V_z,
            alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
            barrier_l=ul, barrier_r=2.0, closed=False
        )
        G_mat = hp.calc_conductance_matrix_exact(syst, E=E)
        g_rr_values.append(G_mat[1, 1])

    g_rr_arr = np.array(g_rr_values)
    spread_r = np.max(g_rr_arr) - np.min(g_rr_arr)
    assert spread_r < 1e-6, f"Right conductance depends on left barrier! spread = {spread_r}"


# ==============================================================================
# T14: test_gamma_zero_matches_transfer_matrix
# ==============================================================================
def test_gamma_zero_matches_transfer_matrix():
    """
    gamma = 0, alpha = 0, V_z = 0: G_LL equals the analytic 1D transfer-matrix result.
    """
    t = 16.529
    mu_leads = t
    Ub = 2.0
    Lb = 3
    Ls = 300
    E = 0.5

    # 1. Analytical 1D transmission for a single channel
    T_analytic = compute_1d_analytic_transmission(t=t, mu_leads=mu_leads, Ub=Ub, Lb=Lb, E=E)

    # 2. Kwant system with gamma=0, alpha=0, V_z=0
    syst = hp.build_system_exact(
        t=t, mu=mu_leads, gamma=0.0, Delta0=0.3, V_z=0.0, alpha=0.0,
        Ls=Ls, Lb=Lb, mu_leads=mu_leads, barrier_l=Ub, barrier_r=-mu_leads,
        closed=False
    )
    G_mat = hp.calc_conductance_matrix_exact(syst, E=E)
    G_LL = G_mat[0, 0]

    # In a normal lead with 2 spin channels (electron modes), G_LL = 2 * T_analytic
    np.testing.assert_allclose(
        G_LL, 2.0 * T_analytic, atol=1e-10, rtol=1e-10,
        err_msg="G_LL at gamma=0 does not match 2 * T_analytic"
    )


# ==============================================================================
# T15: test_input_validation
# ==============================================================================
def test_input_validation():
    """
    Raises for:
    - |E| = Delta0 in Sigma (eta=0)
    - Ln != 0 in build_system_exact
    - Sturm or solve on an open system
    - Sturm with eta > 0
    - Sturm at |E| >= Delta0
    - calc_conductance_matrix_exact with solver_type='gpu'
    """
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3

    # 1. |E| = Delta0 in Sigma
    with pytest.raises(ValueError):
        hp.sigma_al_bcs(Delta0, gamma, Delta0, eta=0.0)

    # 2. Ln != 0 in build_system_exact
    with pytest.raises(NotImplementedError):
        hp.build_system_exact(
            t=t, mu=0.5, gamma=gamma, Delta0=Delta0, V_z=0.0, alpha=1.4,
            Ls=50, Ln=2, closed=False
        )

    # 3. Open system passed to sturm_count_exact or solve_exact_spectrum
    open_syst = hp.build_system_exact(
        t=t, mu=0.5, gamma=gamma, Delta0=Delta0, V_z=0.0, alpha=1.4,
        Ls=50, closed=False
    )
    with pytest.raises(ValueError, match="closed"):
        hp.sturm_count_exact(open_syst, 0.05, Delta0=Delta0)
    with pytest.raises(ValueError, match="closed"):
        hp.solve_exact_spectrum(open_syst, n_states=4, gamma=gamma, Delta0=Delta0)

    # 4. Sturm with eta > 0 in build_system_exact (causes non-Hermitian H)
    closed_broadened = hp.build_system_exact(
        t=t, mu=0.5, gamma=gamma, Delta0=Delta0, V_z=0.0, alpha=1.4,
        Ls=50, closed=True, eta=1e-4
    )
    with pytest.raises(ValueError, match="Hermitian"):
        hp.sturm_count_exact(closed_broadened, 0.05, Delta0=Delta0)

    # 5. Sturm at |E| >= Delta0
    closed_syst = hp.build_system_exact(
        t=t, mu=0.5, gamma=gamma, Delta0=Delta0, V_z=0.0, alpha=1.4,
        Ls=50, closed=True
    )
    with pytest.raises(ValueError, match="Delta0"):
        hp.sturm_count_exact(closed_syst, Delta0, Delta0=Delta0)
    with pytest.raises(ValueError, match="Delta0"):
        hp.sturm_count_exact(closed_syst, 0.5, Delta0=Delta0)

    # 6. calc_conductance_matrix_exact with solver_type='gpu'
    with pytest.raises(NotImplementedError):
        hp.calc_conductance_matrix_exact(open_syst, 0.05, solver_type='gpu')


if __name__ == "__main__":
    pytest.main([__file__])
