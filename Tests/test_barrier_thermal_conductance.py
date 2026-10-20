"""
Tests/test_barrier_thermal_conductance.py

Comprehensive pytest verification suite for finite-temperature barrier sweep conductances:
1. test_zero_temperature_rollback_parity: Bitwise identical to legacy E=0 calculation when T=0.
2. test_quadrature_convergence: Verifies bipolar Gauss-Legendre quadrature matches scipy.integrate.quad.
3. test_arbitrary_bias_t0_rollback: Verifies T=0 quadrature at non-zero bias collapses to [V] with weight 1.0.
4. test_arbitrary_bias_quadrature_parity: Verifies Gauss-Legendre quadrature at V != 0 matches scipy.integrate.quad.
5. test_parity_with_broadened_bias_trace: Validates cross-parity between the newly computed quadrature
   conductance at V=0 and the thermally broadened dI/dV bias curve at V=0.
6. test_physical_plateau_emergence: Verifies the emergence of the finite G_LL plateau at UR >> UL
   under T = 30 mK, reproducing Dourado et al. (PRB 110, 014504) Fig. 5a.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from scipy.integrate import quad
import scipy.signal

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.helpers as hp
from src.cpu_broadening import _temp_kernel


@pytest.fixture(scope="module")
def wire_params():
    """parameters for test nanowires."""
    t = 100.0
    mu = 1.0
    mu_n = 0.0
    Delta0 = 0.5
    gamma = 0.2
    alpha = 3.0
    Ln = 0
    Lb = 3
    Ls = 300
    mu_leads = t
    barrier0 = 2.0
    Vdisx = np.zeros(Ls)
    Vc = np.sqrt(mu**2 + Delta0**2)
    Vz = 1.3 * Vc  # Topological regime

    T_mK = 30.0
    k_B = 8.617333262145e-5  # eV/K
    T_meV = (T_mK * 1e-3) * k_B * 1e3

    return {
        "t": t, "mu": mu, "mu_n": mu_n, "Delta0": Delta0, "gamma": gamma,
        "alpha": alpha, "Ln": Ln, "Lb": Lb, "Ls": Ls, "mu_leads": mu_leads,
        "barrier0": barrier0, "Vdisx": Vdisx, "Vc": Vc, "Vz": Vz,
        "T_mK": T_mK, "k_B": k_B, "T_meV": T_meV
    }


def test_zero_temperature_rollback_parity(wire_params):
    """Verify that T=0 quadrature reproduces the legacy E=0 calculation exactly."""
    p = wire_params
    syst = hp.build_system(
        t=p["t"], mu=p["mu"], mu_n=p["mu_n"], Delta0=p["Delta0"], gamma=p["gamma"],
        V_z=p["Vz"], alpha=p["alpha"], Ln=p["Ln"], Lb=p["Lb"], Ls=p["Ls"],
        mu_leads=p["mu_leads"], barrier_l=p["barrier0"], barrier_r=p["barrier0"],
        Vdisx=p["Vdisx"]
    )

    # Legacy zero-temperature calculation
    g_legacy = hp.calc_conductance_matrix(syst, eng=0.0)

    # Finite-temperature quadrature evaluated at T_mK = 0.0
    q_engs, q_wts = hp.get_thermal_quadrature_nodes(T_mK=0.0, n_nodes=10)
    assert len(q_engs) == 1
    assert q_engs[0] == 0.0
    assert q_wts[0] == 1.0

    g_quad_t0 = hp.calc_conductance_matrix_finite_T(syst, q_engs, q_wts)
    max_diff = float(np.max(np.abs(g_legacy - g_quad_t0)))
    assert max_diff == 0.0, f"Zero-temperature rollback mismatch: max_diff = {max_diff}"


def test_quadrature_convergence(wire_params):
    """Verify bipolar Gauss-Legendre quadrature achieves < 0.05% error vs scipy.integrate.quad."""
    p = wire_params
    def g_kernel(E):
        return 0.2 + 0.5 * (E**2) / (E**2 + 0.01**2) + 0.1 * np.tanh(E / 0.01)

    beta = 1.0 / p["T_meV"]
    def integrand(E):
        u = 0.5 * beta * E
        u_clipped = np.clip(u, -50.0, 50.0)
        sech = 2.0 / (np.exp(u_clipped) + np.exp(-u_clipped))
        return g_kernel(E) * 0.25 * beta * (sech**2)

    exact_val, _ = quad(integrand, -0.5, 0.5)
    q_engs, q_wts = hp.get_thermal_quadrature_nodes(T_mK=p["T_mK"], n_nodes=10)
    quad_val = np.sum(q_wts * g_kernel(q_engs))

    rel_error = abs(exact_val - quad_val) / exact_val
    assert rel_error < 5e-4, f"Quadrature relative error {rel_error:.2e} exceeds threshold 5e-4 (0.05%)"


def test_arbitrary_bias_t0_rollback():
    """Verify that T=0 quadrature at an arbitrary bias collapses to the exact bias point with weight 1.0."""
    for v_bias in [0.0, 0.025, -0.04, 0.1]:
        q_engs, q_wts = hp.get_thermal_quadrature_nodes(T_mK=0.0, n_nodes=10, bias_meV=v_bias)
        assert len(q_engs) == 1
        assert q_engs[0] == v_bias
        assert q_wts[0] == 1.0


def test_arbitrary_bias_quadrature_parity(wire_params):
    """
    Verify that Gauss-Legendre quadrature at arbitrary bias voltages V != 0 matches
    the exact numerical integral of T(E) * [-df(E - eV)/dE] via scipy.integrate.quad.
    """
    p = wire_params
    def g_kernel(E):
        return 0.3 + 0.5 * (E**2) / (E**2 + 0.015**2) + 0.2 * np.sin(20.0 * E)

    beta = 1.0 / p["T_meV"]
    test_biases = [-0.035, -0.01, 0.0, 0.02, 0.05]

    for v_bias in test_biases:
        def integrand(E):
            u = 0.5 * beta * (E - v_bias)
            u_clipped = np.clip(u, -50.0, 50.0)
            sech = 2.0 / (np.exp(u_clipped) + np.exp(-u_clipped))
            return g_kernel(E) * 0.25 * beta * (sech**2)

        exact_val, _ = quad(integrand, v_bias - 0.5, v_bias + 0.5)

        # Standard 10-node evaluation (< 1% tolerance)
        q_engs, q_wts = hp.get_thermal_quadrature_nodes(
            T_mK=p["T_mK"], n_nodes=10, bias_meV=v_bias
        )
        assert np.isclose(np.mean(q_engs), v_bias, atol=1e-7)
        quad_val = np.sum(q_wts * g_kernel(q_engs))
        rel_error = abs(exact_val - quad_val) / exact_val
        assert rel_error < 1e-2, (
            f"Arbitrary bias quadrature (n=10) at V={v_bias} meV rel_error {rel_error:.2e} exceeds threshold 1e-2"
        )

        # High-precision 16-node evaluation (< 0.1% tolerance)
        q_engs16, q_wts16 = hp.get_thermal_quadrature_nodes(
            T_mK=p["T_mK"], n_nodes=16, bias_meV=v_bias
        )
        quad_val16 = np.sum(q_wts16 * g_kernel(q_engs16))
        rel_error16 = abs(exact_val - quad_val16) / exact_val
        assert rel_error16 < 1e-3, (
            f"Arbitrary bias quadrature (n=16) at V={v_bias} meV rel_error {rel_error16:.2e} exceeds threshold 1e-3"
        )


def test_parity_with_broadened_bias_trace(wire_params):
    """
    Cross-Parity Validation:
    Asserts that the zero-bias conductance from the broadened dI/dV bias spectrum
    matches the direct quadrature calculation at the same temperature.
    """
    p = wire_params
    syst = hp.build_system(
        t=p["t"], mu=p["mu"], mu_n=p["mu_n"], Delta0=p["Delta0"], gamma=p["gamma"],
        V_z=p["Vz"], alpha=p["alpha"], Ln=p["Ln"], Lb=p["Lb"], Ls=p["Ls"],
        mu_leads=p["mu_leads"], barrier_l=p["barrier0"], barrier_r=p["barrier0"],
        Vdisx=p["Vdisx"]
    )

    # Route A: Broadened Bias Trace via fine energy grid
    energies = np.linspace(-0.06, 0.06, 301)
    dIdV_left = np.zeros_like(energies)
    dIdV_right = np.zeros_like(energies)

    for idx, eng in enumerate(energies):
        gmat = hp.calc_conductance_matrix(syst, eng=eng)
        dIdV_left[idx] = gmat[0, 0]
        dIdV_right[idx] = gmat[1, 1]

    K = _temp_kernel(energies, p["T_meV"])
    delta_bias = energies[1] - energies[0]
    K_norm = K * delta_bias
    dIdV_left_broadened = scipy.signal.fftconvolve(dIdV_left, K_norm, mode='same')
    dIdV_right_broadened = scipy.signal.fftconvolve(dIdV_right, K_norm, mode='same')

    zero_idx = len(energies) // 2
    g_ll_bias_conv = dIdV_left_broadened[zero_idx]
    g_rr_bias_conv = dIdV_right_broadened[zero_idx]

    # Route B: Direct Quadrature Solver at V=0
    q_engs, q_wts = hp.get_thermal_quadrature_nodes(T_mK=p["T_mK"], n_nodes=10)
    gmat_quad = hp.calc_conductance_matrix_finite_T(syst, q_engs, q_wts)
    g_ll_quad = gmat_quad[0, 0]
    g_rr_quad = gmat_quad[1, 1]

    rel_diff_L = abs(g_ll_bias_conv - g_ll_quad) / g_ll_bias_conv
    rel_diff_R = abs(g_rr_bias_conv - g_rr_quad) / g_rr_bias_conv

    # Agreement within 0.5% (difference due solely to discrete grid discretization in bias convolution)
    assert rel_diff_L < 0.005, (
        f"G_LL bias parity mismatch: conv={g_ll_bias_conv:.6f}, quad={g_ll_quad:.6f}, rel_diff={rel_diff_L:.2e}"
    )
    assert rel_diff_R < 0.005, (
        f"G_RR bias parity mismatch: conv={g_rr_bias_conv:.6f}, quad={g_rr_quad:.6f}, rel_diff={rel_diff_R:.2e}"
    )


def test_physical_plateau_emergence(wire_params):
    """
    Verify Dourado et al. Fig. 5a physics:
    In a well-localized wire (Ls = 300), raising UR >> UL at T=0 produces a zero-energy dip (G_LL -> 0).
    At T = 30 mK, G_LL saturates at a finite plateau significantly above the T=0 value.
    """
    p = wire_params
    syst_asym = hp.build_system(
        t=p["t"], mu=p["mu"], mu_n=p["mu_n"], Delta0=p["Delta0"], gamma=p["gamma"],
        V_z=p["Vz"], alpha=p["alpha"], Ln=p["Ln"], Lb=p["Lb"], Ls=p["Ls"],
        mu_leads=p["mu_leads"], barrier_l=p["barrier0"], barrier_r=40.0 * p["barrier0"],
        Vdisx=p["Vdisx"]
    )

    # Zero temperature
    g_asym_t0 = hp.calc_conductance_matrix(syst_asym, eng=0.0)[0, 0]

    # Finite temperature (T = 30 mK)
    q_engs, q_wts = hp.get_thermal_quadrature_nodes(T_mK=p["T_mK"], n_nodes=10)
    g_asym_t30 = hp.calc_conductance_matrix_finite_T(syst_asym, q_engs, q_wts)[0, 0]

    # Plateau emergence: finite T conductance is strictly higher than zero-energy dip
    assert g_asym_t30 > g_asym_t0, (
        f"Finite-T plateau expected to exceed T=0 dip value: T=30mK={g_asym_t30:.6f}, T=0={g_asym_t0:.6f}"
    )
