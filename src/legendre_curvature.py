"""
Legendre Curvature & Lock-in Harmonics Module
============================================

Implements the tanh-mapped Gauss-Legendre quadrature ("Legendre sum") and
Double-Exponential quadrature to compute zero-bias Andreev conductance (1w),
nonlocal slope (2w), and zero-bias curvature (3w).
"""

from typing import Optional, List, Dict, Union, Tuple, Any
import os
import multiprocessing as mp
from functools import partial
import numpy as np

import src.helpers as hp


# Boltzmann constant in meV / mK:
# k_B = 8.617333262145e-5 eV / K = (8.617333262145e-5 * 1e3 meV) / (1e3 mK) = 8.617333262145e-5 meV / mK
KB_MEV_PER_MK = 8.617333262145e-5


def get_double_exponential_quadrature(
    T_mK: float,
    n_nodes: int = 115
) -> Dict[str, Any]:
    r"""
    Computes Double-Exponential (Tanh-Sinh variant on (-inf, +inf)) quadrature
    nodes and weights achieving machine precision (~1e-15 to 1e-16 in float64)
    for thermal lock-in harmonics (1w, 2w, 3w) at electronic temperature T_mK.

    Mathematical Formulation (Takahasi & Mori 1974):
    ------------------------------------------------
    The thermal convolution integrals on E in (-inf, +inf) possess exponential
    decay ~ exp(-beta |E|). While the algebraic mapping u = tanh(beta E / 2)
    leaves logarithmic branch singularities at u = +-1, the double-exponential
    coordinate transformation:
        E(t) = (2 / beta) * sinh(pi / 2 * sinh(t)),   t in (-inf, +inf)
    forces the integrand and all its derivatives to vanish double-exponentially
    at |t| -> inf. By the Euler-Maclaurin formula, the equispaced trapezoidal rule
    in t converges with exponential rate O(exp(-2 pi d / h)) = O(exp(-C * N)):
        - N = 35 nodes: relative error ~ 2e-4  (0.02%)
        - N = 55 nodes: relative error ~ 9e-8  (0.00001%)
        - N = 75 nodes: relative error ~ 3e-11
        - N = 95 nodes: relative error ~ 1e-13
        - N = 115 nodes: relative error ~ 4e-15 to 1e-16 (machine precision)

    Parameters
    ----------
    T_mK : float
        Electronic temperature in millikelvin (mK). Must be strictly positive.
    n_nodes : int, default 115
        Number of quadrature nodes (must be odd, e.g. 75, 95, 115).
        115 nodes delivers bit-for-bit machine precision in IEEE-754 double precision.

    Returns
    -------
    dict
        Dictionary containing energy nodes and exact weights for 1w, 2w, 3w.
    """
    if T_mK <= 0:
        raise ValueError(f"T_mK must be strictly positive, got {T_mK}")
    N = int(n_nodes)
    if N < 5:
        raise ValueError(f"n_nodes must be at least 5, got {N}")
    if N % 2 == 0:
        N += 1  # Force odd for exact zero-energy centering

    kBT = KB_MEV_PER_MK * float(T_mK)  # in meV
    beta = 1.0 / kBT                   # in meV^-1

    # In double precision, the double-exponential weight cutoff occurs when
    # (pi / 2) * sinh(t_max) ~ 37 (since exp(-37) ~ 1e-16)
    # sinh(t_max) ~ 37 * 2 / pi ~ 23.55 => t_max ~ asinh(23.55) ~ 3.85
    t_max = 3.85
    half_N = N // 2
    h = (2.0 * t_max) / (N - 1)
    k_indices = np.arange(-half_N, half_N + 1)
    t = k_indices * h

    arg = 0.5 * np.pi * np.sinh(t)
    E_nodes = (2.0 / beta) * np.sinh(arg)
    dE_dt = (2.0 / beta) * np.cosh(arg) * (0.5 * np.pi * np.cosh(t))
    w_trap = h * dE_dt

    # Numerically safe thermal derivative kernels
    x = beta * E_nodes / 2.0
    sech = 1.0 / np.cosh(np.clip(x, -350.0, 350.0))
    th = np.tanh(x)

    # Harmonics weights
    # 1w: G(0, T) = int dE G0(E) * [-df/dE] = int dE G0(E) * (beta / 4 cosh^2)
    w_1w = w_trap * (beta / 4.0) * (sech**2)

    # 2w: dG/dV = int dE G0(E) * (beta^2 / 4 cosh^2 * tanh)
    w_2w = w_trap * (beta**2 / 4.0) * (sech**2) * th

    # 3w: d2G/dV2 = int dE G0(E) * (beta^3 / 8 cosh^2 * (3 tanh^2 - 1))
    w_3w = w_trap * (beta**3 / 8.0) * (sech**2) * (3.0 * (th**2) - 1.0)

    # Machine-precision odd parity and reflection antisymmetry
    w_2w[half_N] = 0.0
    E_nodes[half_N] = 0.0

    return {
        'energies': E_nodes,
        'weights_1w': w_1w,
        'weights_2w': w_2w,
        'weights_3w': w_3w,
        't_nodes': t,
        'h_step': float(h),
        'T_mK': float(T_mK),
        'kBT': float(kBT),
        'beta': float(beta),
        'n_nodes': N,
        'quadrature_type': 'double_exponential'
    }


def get_tanh_legendre_quadrature(
    T_mK: float,
    n_nodes: int = 16
) -> Dict[str, Any]:
    """
    Computes tanh-mapped Gauss-Legendre quadrature nodes and weights for the
    thermal lock-in harmonics (1w, 2w, 3w) at electronic temperature T_mK.

    Parameters
    ----------
    T_mK : float
        Electronic temperature in millikelvin (mK). Must be strictly positive.
    n_nodes : int, default 16
        Number of Gauss-Legendre quadrature nodes on u in (-1, +1).
        Typically 12 to 24 nodes are sufficient for < 1% relative error.

    Returns
    -------
    dict
        {
            'energies': np.ndarray,      # 1D array of energy evaluation points E_k in meV
            'weights_1w': np.ndarray,    # 1D array of weights for 1w conductance (dimensionless)
            'weights_2w': np.ndarray,    # 1D array of weights for 2w slope (meV^-1)
            'weights_3w': np.ndarray,    # 1D array of weights for 3w curvature (meV^-2)
            'u_nodes': np.ndarray,       # Standard Gauss-Legendre nodes u_k in (-1, 1)
            'u_weights': np.ndarray,     # Standard Gauss-Legendre weights w_k
            'T_mK': float,               # Temperature in mK
            'kBT': float,                # k_B * T in meV
            'beta': float,               # Inverse thermal energy beta = 1 / (k_B * T) in meV^-1
            'n_nodes': int               # Number of nodes
        }
    """
    if T_mK <= 0:
        raise ValueError(f"T_mK must be strictly positive, got {T_mK}")
    if n_nodes < 4:
        raise ValueError(f"n_nodes must be at least 4, got {n_nodes}")

    kBT = KB_MEV_PER_MK * float(T_mK)  # in meV
    beta = 1.0 / kBT                   # in meV^-1

    # Standard Gauss-Legendre nodes and weights on [-1, +1]
    u, w = np.polynomial.legendre.leggauss(int(n_nodes))

    # Energy mapping: u = tanh(beta * E / 2) <=> E = (2 / beta) * arctanh(u)
    # Clamp u slightly away from +-1 to prevent numerical infinities if n_nodes is huge
    u_clamped = np.clip(u, -1.0 + 1e-15, 1.0 - 1e-15)
    E_nodes = (2.0 / beta) * np.arctanh(u_clamped)

    # Harmonics weights
    # 1w: 1/2 * w_k
    w_1w = 0.5 * w

    # 2w: (beta / 2) * w_k * u_k
    w_2w = 0.5 * beta * w * u

    # 3w: (beta^2 / 4) * w_k * (3 u_k^2 - 1) = (beta^2 / 2) * w_k * P_2(u_k)
    w_3w = 0.25 * (beta**2) * w * (3.0 * (u**2) - 1.0)

    return {
        'energies': E_nodes,
        'weights_1w': w_1w,
        'weights_2w': w_2w,
        'weights_3w': w_3w,
        'u_nodes': u,
        'u_weights': w,
        'T_mK': float(T_mK),
        'kBT': float(kBT),
        'beta': float(beta),
        'n_nodes': int(n_nodes)
    }


def compute_harmonics_single_system(
    syst: Any,
    quad: Dict[str, Any],
    hamiltonian_mode: str = 'exact'
) -> Dict[str, Any]:
    """
    Computes 1w, 2w, and 3w lock-in harmonics for a pre-built Kwant system
    using pre-computed tanh-mapped Gauss-Legendre quadrature weights.
    """
    energies = quad['energies']
    w_1w = quad['weights_1w']
    w_2w = quad['weights_2w']
    w_3w = quad['weights_3w']
    n_nodes = len(energies)

    G_matrices = np.zeros((n_nodes, 2, 2), dtype=float)

    is_exact = (hamiltonian_mode == 'exact')

    for k in range(n_nodes):
        E_val = float(energies[k])
        if is_exact:
            G_mat = hp.calc_conductance_matrix_exact(syst, E=E_val)
        else:
            G_mat = hp.calc_conductance_matrix(syst, eng=E_val)
        G_matrices[k] = G_mat

    g_ll = G_matrices[:, 0, 0]
    g_rr = G_matrices[:, 1, 1]
    g_rl = G_matrices[:, 1, 0]
    g_lr = G_matrices[:, 0, 1]

    L_1w = float(np.sum(w_1w * g_ll))
    R_1w = float(np.sum(w_1w * g_rr))

    RL_2w = float(np.sum(w_2w * g_rl))
    LR_2w = float(np.sum(w_2w * g_lr))

    L_3w = float(np.sum(w_3w * g_ll))
    R_3w = float(np.sum(w_3w * g_rr))

    return {
        'L_1w': L_1w,
        'R_1w': R_1w,
        'RL_2w': RL_2w,
        'LR_2w': LR_2w,
        'L_3w': L_3w,
        'R_3w': R_3w,
        'G_matrices': G_matrices
    }


def get_tanh_legendre_Curvature_nodes(T_mK: float, n_nodes: int = 16):
    """Convenience alias returning (energies, w_1w, w_2w, w_3w)."""
    q = get_tanh_legendre_quadrature(T_mK=T_mK, n_nodes=n_nodes)
    return q['energies'], q['weights_1w'], q['weights_2w'], q['weights_3w']


def get_double_exponential_nodes(T_mK: float, n_nodes: int = 115):
    """Convenience alias returning (energies, w_1w, w_2w, w_3w)."""
    q = get_double_exponential_quadrature(T_mK=T_mK, n_nodes=n_nodes)
    return q['energies'], q['weights_1w'], q['weights_2w'], q['weights_3w']
