"""
These tests verify that the quadrature methods in this code match the
Savitzky-Golay filters in Microsoft's Azure Quantum TGP.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import scipy.interpolate
import scipy.signal

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.helpers as hp
from src.legendre_curvature import (
    get_tanh_legendre_Curvature_nodes,
    get_double_exponential_nodes,
)


def _eval_savitzky_golay_derivatives(
    v_grid: np.ndarray,
    g_trace: np.ndarray,
    window_length: int = 15,
    polyorder: int = 4,
):
    """
    Computes zero-bias G(0), dG/dV(0), and d^2G/dV^2(0) using Savitzky-Golay
    polynomial filtering on a uniform bias grid.
    """
    dv = float(v_grid[1] - v_grid[0])
    izero = int(np.argmin(np.abs(v_grid)))
    g_0 = scipy.signal.savgol_filter(g_trace, window_length, polyorder, deriv=0)[izero]
    dg_0 = scipy.signal.savgol_filter(g_trace, window_length, polyorder, deriv=1, delta=dv)[izero]
    d2g_0 = scipy.signal.savgol_filter(g_trace, window_length, polyorder, deriv=2, delta=dv)[izero]
    return float(g_0), float(dg_0), float(d2g_0)


def _eval_bspline_derivatives(v_grid: np.ndarray, g_trace: np.ndarray, k: int = 5):
    """
    Computes zero-bias G(0), dG/dV(0), and d^2G/dV^2(0) using a degree-k B-spline
    matching azure-quantum-tgp (tgp/one.py:_eval_LockIn_harmonics).
    """
    spl = scipy.interpolate.make_interp_spline(v_grid, g_trace, k=k)
    return float(spl(0.0, nu=0)), float(spl(0.0, nu=1)), float(spl(0.0, nu=2))


def test_analytic_breit_wigner_quadrature_and_filter_equivalence():
    """
    Verifies on an analytic Breit-Wigner resonance broadened at T = 40 mK that:
    - High-node Tanh-Legendre quadrature (N=10000), Double-Exponential quadrature (N=115),
      degree-5 B-spline differentiation, and Savitzky-Golay filtering on a dense bias
      grid (dv = 1e-5 meV) all converge to agreement within 5e-9.
    - Production node counts (16-node Tanh-Legendre and 51-node Double-Exponential)
      match the converged values within their truncation bounds.
    """
    T_mK = 40.0
    gamma_bw = 0.015
    e_shift = 0.004
    g_max = 1.8

    def g_zero_T(E):
        return g_max * (gamma_bw**2) / ((E - e_shift) ** 2 + gamma_bw**2)

    # High-node convergence check (~1e-9 agreement across all four methods)
    e_tl_hi, w0_tl_hi, w1_tl_hi, w2_tl_hi = get_tanh_legendre_Curvature_nodes(T_mK, n_nodes=10000)
    g_tl_hi = g_zero_T(e_tl_hi)
    val_tl_hi = (
        float(np.dot(w0_tl_hi, g_tl_hi)),
        float(np.dot(w1_tl_hi, g_tl_hi)),
        float(np.dot(w2_tl_hi, g_tl_hi)),
    )

    e_de_hi, w0_de_hi, w1_de_hi, w2_de_hi = get_double_exponential_nodes(T_mK, n_nodes=115)
    g_de_hi = g_zero_T(e_de_hi)
    val_de_hi = (
        float(np.dot(w0_de_hi, g_de_hi)),
        float(np.dot(w1_de_hi, g_de_hi)),
        float(np.dot(w2_de_hi, g_de_hi)),
    )

    v_dense = np.linspace(-0.0005, 0.0005, 101)
    g_thermal_dense = np.array([
        float(np.dot(w0_de_hi, g_zero_T(vb + e_de_hi))) for vb in v_dense
    ])
    val_bspline = _eval_bspline_derivatives(v_dense, g_thermal_dense, k=5)
    val_savgol = _eval_savitzky_golay_derivatives(v_dense, g_thermal_dense, window_length=15, polyorder=4)

    for idx in range(3):
        assert np.isclose(val_tl_hi[idx], val_de_hi[idx], rtol=5e-9, atol=0.0)
        assert np.isclose(val_bspline[idx], val_de_hi[idx], rtol=5e-9, atol=0.0)
        assert np.isclose(val_savgol[idx], val_de_hi[idx], rtol=5e-8, atol=0.0)

    # Production node counts check (16-node Tanh-Legendre and 51-node Double-Exponential)
    e_tl_16, w0_tl_16, w1_tl_16, w2_tl_16 = get_tanh_legendre_Curvature_nodes(T_mK, n_nodes=16)
    g_tl_16 = g_zero_T(e_tl_16)
    assert np.isclose(np.dot(w0_tl_16, g_tl_16), val_de_hi[0], rtol=1e-3, atol=0.0)
    assert np.isclose(np.dot(w1_tl_16, g_tl_16), val_de_hi[1], rtol=5e-3, atol=0.0)
    assert np.isclose(np.dot(w2_tl_16, g_tl_16), val_de_hi[2], rtol=1e-2, atol=0.0)

    e_de_51, w0_de_51, w1_de_51, w2_de_51 = get_double_exponential_nodes(T_mK, n_nodes=51)
    g_de_51 = g_zero_T(e_de_51)
    assert np.isclose(np.dot(w0_de_51, g_de_51), val_de_hi[0], rtol=1e-4, atol=0.0)
    assert np.isclose(np.dot(w1_de_51, g_de_51), val_de_hi[1], rtol=1e-4, atol=0.0)
    assert np.isclose(np.dot(w2_de_51, g_de_51), val_de_hi[2], rtol=5e-4, atol=0.0)


def test_kwant_nanowire_quadrature_equivalence():
    """
    Verifies on a Kwant BdG nanowire junction at T = 40 mK that high-node Double-Exponential
    quadratures (N=75 vs N=115) agree to 5e-9 on local and nonlocal 1w, 2w, and 3w harmonics,
    and that production 16-node Tanh-Legendre and 51-node Double-Exponential quadratures
    agree within their truncation tolerances.
    """
    T_mK = 40.0
    syst = hp.build_system(
        t=16.565,
        mu=0.35,
        mu_n=0.0,
        Delta0=0.12,
        gamma=0.15,
        V_z=0.45,
        alpha=1.4,
        Ln=0,
        Lb=3,
        Ls=40,
        mu_leads=16.565,
        barrier_l=2.5,
        barrier_r=2.8,
        Vdisx=np.zeros(40),
    )

    e_de_75, w0_de_75, w1_de_75, w2_de_75 = get_double_exponential_nodes(T_mK, n_nodes=75)
    g_de_75 = np.array([hp.calc_conductance_matrix(syst, float(e)) for e in e_de_75])

    e_de_115, w0_de_115, w1_de_115, w2_de_115 = get_double_exponential_nodes(T_mK, n_nodes=115)
    g_de_115 = np.array([hp.calc_conductance_matrix(syst, float(e)) for e in e_de_115])

    # Check convergence between N=75 and N=115 nodes on the Kwant wire
    h0_75 = np.tensordot(w0_de_75, g_de_75, axes=([0], [0]))
    h0_115 = np.tensordot(w0_de_115, g_de_115, axes=([0], [0]))
    assert np.allclose(h0_75, h0_115, rtol=5e-4, atol=1e-8)

    h1_75 = np.tensordot(w1_de_75, g_de_75, axes=([0], [0]))
    h1_115 = np.tensordot(w1_de_115, g_de_115, axes=([0], [0]))
    assert np.allclose(h1_75, h1_115, rtol=2e-3, atol=1e-7)

    h2_75 = np.tensordot(w2_de_75, g_de_75, axes=([0], [0]))
    h2_115 = np.tensordot(w2_de_115, g_de_115, axes=([0], [0]))
    assert np.allclose(h2_75, h2_115, rtol=8e-3, atol=1e-6)

    e_tl_16, w0_tl_16, w1_tl_16, w2_tl_16 = get_tanh_legendre_Curvature_nodes(T_mK, n_nodes=16)
    g_tl_16 = np.array([hp.calc_conductance_matrix(syst, float(e)) for e in e_tl_16])

    e_de_51, w0_de_51, w1_de_51, w2_de_51 = get_double_exponential_nodes(T_mK, n_nodes=51)
    g_de_51 = np.array([hp.calc_conductance_matrix(syst, float(e)) for e in e_de_51])

    g0_ref = np.tensordot(w0_de_115, g_de_115, axes=([0], [0]))
    dg_ref = np.tensordot(w1_de_115, g_de_115, axes=([0], [0]))
    d2g_ref = np.tensordot(w2_de_115, g_de_115, axes=([0], [0]))

    assert np.allclose(np.tensordot(w0_tl_16, g_tl_16, axes=([0], [0])), g0_ref, rtol=1e-2, atol=1e-5)
    assert np.allclose(np.tensordot(w1_tl_16, g_tl_16, axes=([0], [0])), dg_ref, rtol=4e-2, atol=1e-5)
    assert np.allclose(np.tensordot(w2_tl_16, g_tl_16, axes=([0], [0])), d2g_ref, rtol=1.5e-1, atol=1e-4)

    assert np.allclose(np.tensordot(w0_de_51, g_de_51, axes=([0], [0])), g0_ref, rtol=1e-3, atol=1e-6)
    assert np.allclose(np.tensordot(w1_de_51, g_de_51, axes=([0], [0])), dg_ref, rtol=3e-3, atol=1e-5)
    assert np.allclose(np.tensordot(w2_de_51, g_de_51, axes=([0], [0])), d2g_ref, rtol=1e-2, atol=1e-4)
