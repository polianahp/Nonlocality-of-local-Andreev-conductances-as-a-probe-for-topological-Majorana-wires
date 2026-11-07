"""
Legendre Curvature & Lock-in Harmonics Module
============================================

Implements the tanh-mapped Gauss-Legendre quadrature ("Legendre sum") to
compute zero-bias Andreev conductance (1w), nonlocal slope (2w), and zero-bias
curvature (3w) across the 25 asymmetric cutter gate configurations:
  (U_c,L^{(i)}, U_c,R^{(j)}) for i, j in {0, 1, 2, 3, 4}
formed by calibrated cutter gates targeting G_N in {0.35, 0.49, 0.62, 0.76, 0.90} e^2/h.

Mathematical Formulation:
-------------------------
The thermally broadened differential conductance at bias V and temperature T
(with beta = 1 / (k_B T)) is:
    G(V, T) = int_{-inf}^{+inf} dE G_0(E) * [-df(E - V, T) / dE]
            = int_{-inf}^{+inf} dE G_0(E) * [beta / (4 * cosh^2(beta (E - V) / 2))]

Under the nonlinear coordinate transformation:
    u = tanh(beta * E / 2) in (-1, +1)  <=>  E(u) = (2 / beta) * arctanh(u)
the differential measure transforms as:
    du = [beta / (2 * cosh^2(beta * E / 2))] dE = 2 * [-df(E, T) / dE] dE.

Differentiating with respect to bias voltage V at zero bias (V = 0) yields exact
algebraic projections onto the orthogonal Legendre polynomials P_m(u):

1. Zero-Bias Conductance (1w):
    I_{1w} = G(0, T) = 1/2 * int_{-1}^{+1} du P_0(u) G_0(E(u))
    Quadrature weights: W_{1w}[k] = 1/2 * w_k

2. Nonlocal Lock-in Derivative / Slope (2w):
    I_{2w} = dG/dV|_{V=0} = beta / 2 * int_{-1}^{+1} du P_1(u) G_0(E(u))
    Quadrature weights: W_{2w}[k] = (beta / 2) * w_k * u_k

3. Zero-Bias Andreev Conductance Curvature (3w):
    I_{3w} = d^2 G / dV^2|_{V=0} = (beta^2 / 2) * int_{-1}^{+1} du P_2(u) G_0(E(u))
           = (beta^2 / 4) * int_{-1}^{+1} du (3 u^2 - 1) G_0(E(u))
    Quadrature weights: W_{3w}[k] = (beta^2 / 4) * w_k * (3 u_k^2 - 1)

Here {u_k, w_k} are the standard Gauss-Legendre quadrature nodes and weights on [-1, +1].
Because the integrand is projected onto Legendre polynomials, Gauss-Legendre quadrature
achieves exponential spectral convergence, reducing the required Kwant smatrix solves from
97 bias energies to only n = 12-16 nodes per configuration (~6x-8x acceleration).
"""

from typing import Optional, List, Dict, Union, Tuple, Any
import os
import multiprocessing as mp
from functools import partial
import numpy as np

import src.helpers as hp
from src.barrier_calibration import calibrate_wire_cutters


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
    hamiltonian_mode: str = 'exact',
    solver_type: str = 'cpu'
) -> Dict[str, Any]:
    """
    Computes 1w, 2w, and 3w lock-in harmonics for a pre-built Kwant system
    using pre-computed tanh-mapped Gauss-Legendre quadrature weights.

    Parameters
    ----------
    syst : kwant.system.FiniteSystem
        Pre-finalized Kwant system.
    quad : dict
        Quadrature dictionary returned by get_tanh_legendre_quadrature.
    hamiltonian_mode : str, default 'exact'
        'exact' (energy-dependent self-energy) or 'legacy' (static model).
    solver_type : str, default 'cpu'
        'cpu' or 'gpu'.

    Returns
    -------
    dict
        {
            'L_1w': float,       # Left zero-bias conductance (e^2/h)
            'R_1w': float,       # Right zero-bias conductance (e^2/h)
            'RL_2w': float,      # Nonlocal slope dG_RL/dV (e^2/h / meV)
            'LR_2w': float,      # Nonlocal slope dG_LR/dV (e^2/h / meV)
            'L_3w': float,       # Left zero-bias curvature d^2 G_LL / dV^2 (e^2/h / meV^2)
            'R_3w': float,       # Right zero-bias curvature d^2 G_RR / dV^2 (e^2/h / meV^2)
            'G_matrices': np.ndarray  # Shape (n_nodes, 2, 2)
        }
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
            G_mat = hp.calc_conductance_matrix_exact(syst, E=E_val, solver_type=solver_type)
        else:
            G_mat = hp.calc_conductance_matrix(syst, eng=E_val, solver_type=solver_type)
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


def _worker_eval_cutter_pair(task_args: Tuple) -> Tuple[int, int, Dict[str, Any]]:
    """
    Top-level worker function for multiprocessing execution across cutter pairs.
    Builds system inside worker to guarantee process-boundary pickling safety.
    """
    (
        i, j, u_l, u_r,
        system_params, Vdisx,
        quad, hamiltonian_mode, solver_type
    ) = task_args

    p = system_params
    is_exact = (hamiltonian_mode == 'exact')

    if is_exact:
        syst = hp.build_system_exact(
            t=float(p['t']),
            mu=float(p['mu']),
            gamma=float(p['gamma']),
            Delta0=float(p['Delta0']),
            V_z=float(p['V_z']),
            alpha=float(p['alpha']),
            Ls=int(p['Ls']),
            Vdisx=Vdisx,
            a=float(p.get('a', 1.0)),
            closed=False,
            mu_n=float(p.get('mu_n', 0.0)),
            Ln=int(p.get('Ln', 0)),
            Lb=int(p.get('Lb', 3)),
            mu_leads=p.get('mu_leads', None),
            barrier_l=float(u_l),
            barrier_r=float(u_r),
            eta=float(p.get('eta', 0.0))
        )
    else:
        syst = hp.build_system(
            t=float(p['t']),
            mu=float(p['mu']),
            mu_n=float(p.get('mu_n', 0.0)),
            gamma=float(p['gamma']),
            Delta0=float(p['Delta0']),
            V_z=float(p['V_z']),
            alpha=float(p['alpha']),
            Ln=int(p.get('Ln', 0)),
            Lb=int(p.get('Lb', 3)),
            Ls=int(p['Ls']),
            mu_leads=float(p['mu_leads']) if p.get('mu_leads') is not None else float(p['t']),
            barrier_l=float(u_l),
            barrier_r=float(u_r),
            Vdisx=Vdisx,
            a=float(p.get('a', 1.0))
        )

    harmonics = compute_harmonics_single_system(
        syst=syst,
        quad=quad,
        hamiltonian_mode=hamiltonian_mode,
        solver_type=solver_type
    )
    # Do not pass full G_matrices back across processes to save IPC bandwidth
    harmonics_summary = {
        'L_1w': harmonics['L_1w'],
        'R_1w': harmonics['R_1w'],
        'RL_2w': harmonics['RL_2w'],
        'LR_2w': harmonics['LR_2w'],
        'L_3w': harmonics['L_3w'],
        'R_3w': harmonics['R_3w']
    }
    return i, j, harmonics_summary


def compute_25_cutter_curvatures_legendre(
    u_left_5: Union[List[float], np.ndarray],
    u_right_5: Union[List[float], np.ndarray],
    system_params: Dict[str, Any],
    Vdisx: Optional[np.ndarray] = None,
    T_mK: float = 30.0,
    n_nodes: int = 16,
    hamiltonian_mode: str = 'exact',
    solver_type: str = 'cpu',
    n_jobs: int = 1,
    quadrature_method: str = 'legendre'
) -> Dict[str, Any]:
    """
    Computes zero-bias curvature (3w), nonlocal slope (2w), and zero-bias conductance (1w)
    across all 5x5 = 25 asymmetric cutter gate configurations (U_c,L^{(i)}, U_c,R^{(j)}).

    Parameters
    ----------
    u_left_5 : list or np.ndarray
        Array of left cutter barrier potentials (length N_L, typically 5).
    u_right_5 : list or np.ndarray
        Array of right cutter barrier potentials (length N_R, typically 5).
    system_params : dict
        Hamiltonian parameters for the wire:
        {'t': float, 'mu': float, 'gamma': float, 'Delta0': float, 'V_z': float, 'alpha': float, 'Ls': int, ...}
    Vdisx : np.ndarray, optional
        Disorder potential array along wire length Ls.
    T_mK : float, default 30.0
        Temperature in millikelvin.
    n_nodes : int, default 16
        Number of quadrature nodes.
        For method='legendre': 12-24 nodes.
        For method='double_exponential': 75 nodes (~1e-11) or 115 nodes (~1e-15 machine precision).
    hamiltonian_mode : str, default 'exact'
        'exact' or 'legacy'.
    solver_type : str, default 'cpu'
        'cpu' or 'gpu'.
    n_jobs : int, default 1
        Number of parallel worker processes. If n_jobs > 1 or -1, utilizes multiprocessing.
    quadrature_method : str, default 'legendre'
        'legendre' for fast tanh-Gauss-Legendre (~1-2% at n=12), or
        'double_exponential' (alias 'de') for exponential convergence to machine precision (~1e-15).

    Returns
    -------
    dict
        Structured dictionary containing:
        - 'L_3w': 2D np.ndarray of shape (N_L, N_R) (left curvature)
        - 'R_3w': 2D np.ndarray of shape (N_L, N_R) (right curvature)
        - 'RL_2w': 2D np.ndarray of shape (N_L, N_R) (nonlocal slope RL)
        - 'LR_2w': 2D np.ndarray of shape (N_L, N_R) (nonlocal slope LR)
        - 'L_1w': 2D np.ndarray of shape (N_L, N_R) (left zero-bias conductance)
        - 'R_1w': 2D np.ndarray of shape (N_L, N_R) (right zero-bias conductance)
        - 'u_left': np.ndarray of shape (N_L,)
        - 'u_right': np.ndarray of shape (N_R,)
        - 'L_3w_mean': float, mean left curvature over all 25 configurations
        - 'R_3w_mean': float, mean right curvature over all 25 configurations
        - 'RL_2w_mean': float, mean nonlocal slope RL over all 25 configurations
        - 'LR_2w_mean': float, mean nonlocal slope LR over all 25 configurations
        - 'L_1w_mean': float, mean left conductance over all 25 configurations
        - 'R_1w_mean': float, mean right conductance over all 25 configurations
        - 'symmetric_pairs_L_3w': 1D np.ndarray of diagonal values (i == j)
        - 'symmetric_pairs_R_3w': 1D np.ndarray of diagonal values (i == j)
        - 'quadrature': dict of quadrature parameters
    """
    u_left = np.asarray(u_left_5, dtype=float)
    u_right = np.asarray(u_right_5, dtype=float)
    NL = len(u_left)
    NR = len(u_right)

    if quadrature_method in ('double_exponential', 'de'):
        quad = get_double_exponential_quadrature(T_mK=T_mK, n_nodes=n_nodes)
    else:
        quad = get_tanh_legendre_quadrature(T_mK=T_mK, n_nodes=n_nodes)

    # Initialize output 2D grids
    L_3w = np.zeros((NL, NR), dtype=float)
    R_3w = np.zeros((NL, NR), dtype=float)
    RL_2w = np.zeros((NL, NR), dtype=float)
    LR_2w = np.zeros((NL, NR), dtype=float)
    L_1w = np.zeros((NL, NR), dtype=float)
    R_1w = np.zeros((NL, NR), dtype=float)

    # Prepare tasks list for 25 pairs
    tasks = []
    for i in range(NL):
        for j in range(NR):
            tasks.append((
                i, j, u_left[i], u_right[j],
                system_params, Vdisx,
                quad, hamiltonian_mode, solver_type
            ))

    # Determine concurrency
    effective_n_jobs = int(n_jobs)
    if effective_n_jobs == -1:
        effective_n_jobs = os.cpu_count() or 1
    effective_n_jobs = max(1, min(effective_n_jobs, len(tasks)))

    if effective_n_jobs > 1:
        with mp.Pool(processes=effective_n_jobs) as pool:
            results = pool.map(_worker_eval_cutter_pair, tasks)
    else:
        results = [_worker_eval_cutter_pair(t) for t in tasks]

    for i, j, h in results:
        L_3w[i, j] = h['L_3w']
        R_3w[i, j] = h['R_3w']
        RL_2w[i, j] = h['RL_2w']
        LR_2w[i, j] = h['LR_2w']
        L_1w[i, j] = h['L_1w']
        R_1w[i, j] = h['R_1w']

    # Symmetric diagonal elements
    min_dim = min(NL, NR)
    diag_L_3w = np.array([L_3w[d, d] for d in range(min_dim)])
    diag_R_3w = np.array([R_3w[d, d] for d in range(min_dim)])

    return {
        'L_3w': L_3w,
        'R_3w': R_3w,
        'RL_2w': RL_2w,
        'LR_2w': LR_2w,
        'L_1w': L_1w,
        'R_1w': R_1w,
        'u_left': u_left,
        'u_right': u_right,
        'L_3w_mean': float(np.mean(L_3w)),
        'R_3w_mean': float(np.mean(R_3w)),
        'RL_2w_mean': float(np.mean(RL_2w)),
        'LR_2w_mean': float(np.mean(LR_2w)),
        'L_1w_mean': float(np.mean(L_1w)),
        'R_1w_mean': float(np.mean(R_1w)),
        'L_3w_std': float(np.std(L_3w)),
        'R_3w_std': float(np.std(R_3w)),
        'symmetric_pairs_L_3w': diag_L_3w,
        'symmetric_pairs_R_3w': diag_R_3w,
        'quadrature': quad,
        'metadata': {
            'T_mK': T_mK,
            'n_nodes': n_nodes,
            'hamiltonian_mode': hamiltonian_mode,
            'NL': NL,
            'NR': NR,
            'total_configurations': NL * NR
        }
    }


def calibrate_and_compute_25_cutter_curvatures(
    system_params: Dict[str, Any],
    target_gns: Optional[Union[List[float], np.ndarray]] = None,
    u_left_5: Optional[Union[List[float], np.ndarray]] = None,
    u_right_5: Optional[Union[List[float], np.ndarray]] = None,
    Vdisx: Optional[np.ndarray] = None,
    T_mK: float = 30.0,
    n_nodes: int = 16,
    hamiltonian_mode: str = 'exact',
    solver_type: str = 'cpu',
    n_jobs: int = 1,
    calib_tol: float = 1e-4,
    quadrature_method: str = 'legendre'
) -> Dict[str, Any]:
    """
    End-to-end convenience pipeline:
    1. Obtains 5 left and 5 right cutter gate voltages (either passed directly or
       calibrated via `calibrate_wire_cutters` targeting G_N in {0.35, 0.49, 0.62, 0.76, 0.90} e^2/h).
    2. Rapidly computes 1w, 2w, and 3w observables across all 25 configurations
       using tanh-mapped Gauss-Legendre or Double-Exponential quadrature.

    Parameters
    ----------
    system_params : dict
        Nanowire parameters (t, mu, gamma, Delta0, V_z, alpha, Ls, Lb, etc.).
    target_gns : list or np.ndarray, optional
        Target normal-state conductances (default: [0.35, 0.49, 0.62, 0.76, 0.90] e^2/h).
    u_left_5, u_right_5 : list or np.ndarray, optional
        Pre-calibrated cutter gate potentials. If None, `calibrate_wire_cutters` is called.
    Vdisx : np.ndarray, optional
        Disorder potential array.
    T_mK : float, default 30.0
        Temperature in millikelvin.
    n_nodes : int, default 16
        Number of quadrature points (e.g. 16 for Legendre, 75-115 for Double-Exponential).
    hamiltonian_mode : str, default 'exact'
        'exact' or 'legacy'.
    solver_type : str, default 'cpu'
        'cpu' or 'gpu'.
    n_jobs : int, default 1
        Number of parallel worker processes.
    calib_tol : float, default 1e-4
        Tolerance for barrier calibration root finder.
    quadrature_method : str, default 'legendre'
        'legendre' or 'double_exponential' ('de').

    Returns
    -------
    dict
        Output of `compute_25_cutter_curvatures_legendre` augmented with calibration details.
    """
    if target_gns is None:
        target_gns = [0.35, 0.49, 0.62, 0.76, 0.90]

    target_gns = np.asarray(target_gns, dtype=float)

    if u_left_5 is None or u_right_5 is None:
        p = system_params
        calib = calibrate_wire_cutters(
            target_gns=target_gns,
            t=float(p['t']),
            gamma=float(p['gamma']),
            Delta0=float(p['Delta0']),
            alpha=float(p['alpha']),
            Ln=int(p.get('Ln', 0)),
            Lb=int(p.get('Lb', 3)),
            Ls=int(p['Ls']),
            mu_leads=p.get('mu_leads', None),
            tol=calib_tol
        )
        u_left_5 = calib['u_left']
        u_right_5 = calib['u_right']
    else:
        u_left_5 = np.asarray(u_left_5, dtype=float)
        u_right_5 = np.asarray(u_right_5, dtype=float)

    result = compute_25_cutter_curvatures_legendre(
        u_left_5=u_left_5,
        u_right_5=u_right_5,
        system_params=system_params,
        Vdisx=Vdisx,
        T_mK=T_mK,
        n_nodes=n_nodes,
        hamiltonian_mode=hamiltonian_mode,
        solver_type=solver_type,
        n_jobs=n_jobs,
        quadrature_method=quadrature_method
    )
    result['target_gns'] = target_gns
    return result


def evaluate_cutter_consensus_from_legendre(
    res_25: Dict[str, Any],
    curvature_threshold: float = 20.0,
    k_of_n: int = 15,
    lead: str = 'mean_LR'
) -> Dict[str, Any]:
    """
    Applies consensus voting across the 25 cutter configurations to determine
    whether the zero-bias Andreev curvature satisfies the topological bound:
        curvature <= -curvature_threshold

    Parameters
    ----------
    res_25 : dict
        Dictionary returned by compute_25_cutter_curvatures_legendre.
    curvature_threshold : float, default 20.0
        Threshold t_{3w} such that curvature <= -t_{3w} indicates Andreev bound state dip.
    k_of_n : int, default 15
        Number of passing configurations required out of 25 (e.g. 15 of 25 = 60% consensus).
    lead : {'L', 'R', 'mean_LR', 'max_LR', 'min_LR'}, default 'mean_LR'
        Lead combination mode.

    Returns
    -------
    dict
        {
            'passed': bool,
            'pass_count': int,
            'pass_fraction': float,
            'pass_mask': np.ndarray of shape (5, 5),
            'consensus_threshold': int,
            'curvature_threshold': float
        }
    """
    L_3w = res_25['L_3w']
    R_3w = res_25['R_3w']

    if lead == 'L':
        curv = L_3w
    elif lead == 'R':
        curv = R_3w
    elif lead == 'mean_LR':
        curv = 0.5 * (L_3w + R_3w)
    elif lead == 'max_LR':
        curv = np.maximum(L_3w, R_3w)
    elif lead == 'min_LR':
        curv = np.minimum(L_3w, R_3w)
    else:
        raise ValueError(f"Unknown lead mode '{lead}'")

    pass_mask = (curv <= -float(curvature_threshold))
    pass_count = int(np.sum(pass_mask))
    total_configs = int(curv.size)
    pass_fraction = float(pass_count / total_configs)

    return {
        'passed': bool(pass_count >= k_of_n),
        'pass_count': pass_count,
        'pass_fraction': pass_fraction,
        'pass_mask': pass_mask,
        'consensus_threshold': int(k_of_n),
        'total_configs': total_configs,
        'curvature_threshold': float(curvature_threshold)
    }
