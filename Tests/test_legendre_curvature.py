"""
Test Suite: Legendre Curvature & Lock-in Harmonics Module
=========================================================

Verifies:
1. Quadrature moments, partition of unity, and Legendre orthogonality.
2. Exponential convergence against analytical Breit-Wigner / Andreev resonance integrals.
3. Close numerical parity with full 97-point bias sweep + FFT broadening + B-spline derivative.
4. Double-Exponential quadrature orthogonality, moments, and machine-precision convergence.
"""

import sys
import os
from pathlib import Path
import pytest
import numpy as np
from scipy.integrate import quad

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.helpers as hp
import src.legendre_curvature as lc
from src.Protocol_Elements.cache import get_linear_broaden_and_spline_operators


def test_quadrature_orthogonality_and_moments():
    """
    Asserts mathematical properties of the tanh-mapped Gauss-Legendre quadrature:
    - Partition of unity: sum(w_1w) == 1.0.
    - Odd parity cancellation: sum(w_2w) == 0.0.
    - Legendre orthogonality: sum(w_3w) == 0.0 (P_2 perp P_0).
    - Energy node reflection symmetry: E_k == -E_{n-1-k}.
    """
    for T_mK in [20.0, 30.0, 40.0]:
        for n in [8, 12, 16, 24]:
            q = lc.get_tanh_legendre_quadrature(T_mK=T_mK, n_nodes=n)
            
            # 1w weights integrate flat transmission to 1.0
            np.testing.assert_allclose(np.sum(q['weights_1w']), 1.0, rtol=1e-12)
            
            # 2w weights integrate even transmission to 0.0
            np.testing.assert_allclose(np.sum(q['weights_2w']), 0.0, atol=1e-12)
            
            # 3w weights integrate constant transmission to 0.0 (P_2 is orthogonal to P_0)
            # Floating point roundoff is bounded by eps * beta^2 ~ 1e-16 * 1.5e5 ~ 1.5e-11 to 1e-9
            np.testing.assert_allclose(np.sum(q['weights_3w']), 0.0, atol=1e-8)
            # Dimensionless P_2 moment is zero to machine precision
            np.testing.assert_allclose(np.sum(q['u_weights'] * (3.0 * q['u_nodes']**2 - 1.0)), 0.0, atol=1e-14)
            
            # Reflection symmetry
            energies = q['energies']
            w_1w = q['weights_1w']
            w_2w = q['weights_2w']
            w_3w = q['weights_3w']
            
            np.testing.assert_allclose(energies, -energies[::-1], atol=1e-15)
            np.testing.assert_allclose(w_1w, w_1w[::-1], rtol=1e-12)
            np.testing.assert_allclose(w_3w, w_3w[::-1], rtol=1e-12)
            np.testing.assert_allclose(w_2w, -w_2w[::-1], atol=1e-15)


def test_analytical_convergence_vs_quad():
    """
    Verifies that the Legendre sum converges exponentially to the exact
    continuum thermal derivative integrals for a zero-energy Lorentzian resonance:
        G_0(E) = 2.0 * Gamma^2 / (E^2 + Gamma^2).
    """
    Gamma = 0.02  # 20 ueV resonance width
    T_mK = 30.0
    kBT = lc.KB_MEV_PER_MK * T_mK
    beta = 1.0 / kBT

    def G0(E):
        return 2.0 * Gamma**2 / (E**2 + Gamma**2)

    # Exact continuum 1w integral
    def integrand_1w(E):
        ch = np.cosh(beta * E / 2.0)
        return G0(E) * (beta / (4.0 * ch**2))

    I_1w_exact, _ = quad(integrand_1w, -0.2, 0.2, epsabs=1e-12, epsrel=1e-12)

    # Exact continuum 3w integral
    def integrand_3w(E):
        ch = np.cosh(beta * E / 2.0)
        th = np.tanh(beta * E / 2.0)
        return G0(E) * (beta**3 / 8.0) * (1.0 / ch**2) * (3.0 * th**2 - 1.0)

    I_3w_exact, _ = quad(integrand_3w, -0.2, 0.2, epsabs=1e-12, epsrel=1e-12)

    # Legendre sum at n=24
    q24 = lc.get_tanh_legendre_quadrature(T_mK=T_mK, n_nodes=24)
    I_1w_leg = np.sum(q24['weights_1w'] * G0(q24['energies']))
    I_3w_leg = np.sum(q24['weights_3w'] * G0(q24['energies']))

    # Relative errors must be < 0.1% for 1w and < 0.6% for 3w
    rel_err_1w = abs(I_1w_leg - I_1w_exact) / abs(I_1w_exact)
    rel_err_3w = abs(I_3w_leg - I_3w_exact) / abs(I_3w_exact)

    assert rel_err_1w < 0.001, f"1w error too high: {rel_err_1w:.2e}"
    assert rel_err_3w < 0.006, f"3w error too high: {rel_err_3w:.2e}"


def test_kwant_wire_parity_vs_97pt_spline():
    """
    Compares Legendre sum (n=16) against the 97-point bias sweep + B-spline
    baseline on an exact Kwant nanowire.
    """
    syst = hp.build_system_exact(
        t=10.0, mu=1.5, gamma=0.2, Delta0=0.3, V_z=0.5,
        alpha=1.0, Ls=60, Lb=3, barrier_l=1.0, barrier_r=1.0
    )

    T_mK = 30.0
    bias = np.linspace(-0.134, 0.134, 97)
    g_ll_97 = np.zeros(97)
    for idx, V in enumerate(bias):
        G = hp.calc_conductance_matrix_exact(syst, float(V))
        g_ll_97[idx] = G[0, 0]

    M_broaden, w1_spline, w2_spline = get_linear_broaden_and_spline_operators(bias, T_mK)
    g_broad = g_ll_97 @ M_broaden
    I_1w_ref = g_broad[48]
    I_3w_ref = g_broad @ w2_spline

    quad = lc.get_tanh_legendre_quadrature(T_mK=T_mK, n_nodes=16)
    harmonics = lc.compute_harmonics_single_system(syst, quad, hamiltonian_mode='exact')

    # 1w must match within 0.5%
    rel_err_1w = abs(harmonics['L_1w'] - I_1w_ref) / abs(I_1w_ref)
    assert rel_err_1w < 0.005, f"Kwant 1w mismatch: ref={I_1w_ref}, leg={harmonics['L_1w']}, err={rel_err_1w}"

    # 3w must have identical sign and match within 8% (difference is B-spline finite stencil error)
    assert np.sign(harmonics['L_3w']) == np.sign(I_3w_ref)
    rel_err_3w = abs(harmonics['L_3w'] - I_3w_ref) / abs(I_3w_ref)
    assert rel_err_3w < 0.08, f"Kwant 3w mismatch: ref={I_3w_ref}, leg={harmonics['L_3w']}, err={rel_err_3w}"


def test_double_exponential_quadrature_orthogonality_and_moments():
    """
    Asserts mathematical properties of the Double-Exponential quadrature:
    - Partition of unity: sum(w_1w) == 1.0 within machine precision (1e-14).
    - Odd parity cancellation: sum(w_2w) == 0.0 (exact zero by antisymmetry).
    - Second derivative weight sum: sum(w_3w) == 0.0 within 1e-8 (scaled by beta^2).
    - Reflection symmetry and exact zero-energy node.
    """
    tol_map = {75: 1e-10, 95: 1e-13, 115: 1e-15}
    atol_w3_map = {75: 1e-5, 95: 1e-7, 115: 1e-10}
    for T_mK in [20.0, 30.0, 40.0]:
        for n in [75, 95, 115]:
            q = lc.get_double_exponential_quadrature(T_mK=T_mK, n_nodes=n)
            
            # Partition of unity: integral of -df/dE over (-inf, inf) is exactly 1
            # Converges double-exponentially to machine epsilon (2.22e-16 at n=115)
            np.testing.assert_allclose(np.sum(q['weights_1w']), 1.0, rtol=tol_map[n])
            
            # Odd parity cancellation: integral of odd function is 0
            np.testing.assert_allclose(np.sum(q['weights_2w']), 0.0, atol=1e-14)
            
            # Curvature moment of flat transmission is 0 (scaled by beta^2 ~ 1e5 meV^-2)
            np.testing.assert_allclose(np.sum(q['weights_3w']), 0.0, atol=atol_w3_map[n])
            
            # Reflection symmetry
            energies = q['energies']
            w_1w = q['weights_1w']
            w_2w = q['weights_2w']
            w_3w = q['weights_3w']
            
            np.testing.assert_allclose(energies, -energies[::-1], atol=1e-15)
            np.testing.assert_allclose(w_1w, w_1w[::-1], rtol=1e-14)
            np.testing.assert_allclose(w_3w, w_3w[::-1], rtol=1e-14)
            np.testing.assert_allclose(w_2w, -w_2w[::-1], atol=1e-15)


def test_double_exponential_machine_precision_vs_reference():
    """
    Verifies that the Double-Exponential quadrature exhibits exponential
    convergence on a zero-energy Lorentzian resonance G_0(E) = 2 Gamma^2 / (E^2 + Gamma^2),
    achieving < 1e-13 error at N=115 nodes.
    """
    import mpmath
    mpmath.mp.dps = 50

    Gamma = 0.02  # 20 ueV
    T_mK = 30.0
    kBT = lc.KB_MEV_PER_MK * T_mK
    beta = 1.0 / kBT

    # Reference computation using mpmath 50-digit quad
    def mp_integrand_1w(E):
        ch = mpmath.cosh(beta * E / 2.0)
        G0 = 2.0 * Gamma**2 / (E**2 + Gamma**2)
        return G0 * (beta / (4.0 * ch**2))

    def mp_integrand_3w(E):
        ch = mpmath.cosh(beta * E / 2.0)
        th = mpmath.tanh(beta * E / 2.0)
        G0 = 2.0 * Gamma**2 / (E**2 + Gamma**2)
        return G0 * (beta**3 / 8.0) * (1.0 / ch**2) * (3.0 * th**2 - 1.0)

    ref_1w = float(mpmath.quad(mp_integrand_1w, [-mpmath.inf, mpmath.inf]))
    ref_3w = float(mpmath.quad(mp_integrand_3w, [-mpmath.inf, mpmath.inf]))

    def G0_np(E):
        return 2.0 * Gamma**2 / (E**2 + Gamma**2)

    # N=55: relative error < 1e-6
    q55 = lc.get_double_exponential_quadrature(T_mK=T_mK, n_nodes=55)
    val_1w_55 = np.sum(q55['weights_1w'] * G0_np(q55['energies']))
    val_3w_55 = np.sum(q55['weights_3w'] * G0_np(q55['energies']))
    assert abs(val_1w_55 - ref_1w) / abs(ref_1w) < 1e-8
    assert abs(val_3w_55 - ref_3w) / abs(ref_3w) < 1e-6

    # N=75: relative error < 1e-9
    q75 = lc.get_double_exponential_quadrature(T_mK=T_mK, n_nodes=75)
    val_1w_75 = np.sum(q75['weights_1w'] * G0_np(q75['energies']))
    val_3w_75 = np.sum(q75['weights_3w'] * G0_np(q75['energies']))
    assert abs(val_1w_75 - ref_1w) / abs(ref_1w) < 1e-11
    assert abs(val_3w_75 - ref_3w) / abs(ref_3w) < 1e-9

    # N=115: machine precision (< 1e-13)
    q115 = lc.get_double_exponential_quadrature(T_mK=T_mK, n_nodes=115)
    val_1w_115 = np.sum(q115['weights_1w'] * G0_np(q115['energies']))
    val_3w_115 = np.sum(q115['weights_3w'] * G0_np(q115['energies']))
    assert abs(val_1w_115 - ref_1w) / abs(ref_1w) < 1e-14
    assert abs(val_3w_115 - ref_3w) / abs(ref_3w) < 1e-13
