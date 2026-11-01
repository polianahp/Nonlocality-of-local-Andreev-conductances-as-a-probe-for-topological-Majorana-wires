"""
Barrier Calibration Module (Aghaee et al. 2023)
================================================

Calibrates nanowire cutter gate potentials (U_c,L and U_c,R) to achieve target
normal-state differential conductances G_N at zero magnetic field (V_z = 0) and
above-gap bias (E = 0.5 meV).

- Proximitized superconductor sites include the physical BCS parent-Al self-energy
  Sigma_Al(omega) = -gamma / s(omega) * (omega * tau_0 - Delta0 * tau_x) (x) sigma_0,
  with s(omega) = sqrt(Delta0^2 - omega^2) for |omega| < Delta0 and
  s(omega) = -i * sgn(omega) * sqrt(omega^2 - Delta0^2) for |omega| > Delta0.
  At omega = 0, Sigma_Al(0) = +gamma * tau_x (x) sigma_0.
- This drains transmitted quasiparticles into the grounded parent Al reservoir,
  eliminating finite-size cavity reflections and strictly decoupling G_LL from U_R
  (and G_RR from U_L).
- Target inversion: The normal conductance G_N(U) is monotonic with barrier height U.
  We invert G_N(U_c) = G_target by computing the median conductance across 15
  equidistant chemical potential points in the lowest subband (mu in [0.0, 2.0] meV)
  via Brent's root-finding method (scipy.optimize.brentq).
"""

from typing import Optional, List, Dict, Union
import numpy as np
from scipy.optimize import brentq
import kwant

import src.helpers as hp


def calculate_gn_above_gap(
    u_barrier: float,
    side: str,
    t: float,
    mu: float,
    gamma: float,
    Delta0: float,
    alpha: float,
    Ln: int = 0,
    Lb: int = 3,
    Ls: int = 300,
    mu_leads: Optional[float] = None,
    energy: float = 0.5,
    solver_type: str = 'cpu'
) -> float:
    """
    Computes single-junction normal conductance G_N at energy E > Delta0 for a given barrier height.

    Parameters
    ----------
    u_barrier : float
        Cutter gate potential in meV.
    side : str
        'left' or 'right' indicating which junction to measure.
    t : float
        Hopping energy in meV.
    mu : float
        Superconductor chemical potential in meV.
    gamma : float
        SM-SC coupling parameter in meV.
    Delta0 : float
        Bulk superconducting gap in meV.
    alpha : float
        Rashba spin-orbit coupling in meV.
    Ln : int
        Normal region length in lattice sites (default: 0).
    Lb : int
        Barrier length in lattice sites (default: 3).
    Ls : int
        Superconductor length in lattice sites (default: 300).
    mu_leads : float, optional
        Lead chemical potential in meV (default: t).
    energy : float
        Above-gap bias energy in meV (default: 0.5).
    solver_type : str
        'cpu' or 'gpu'.

    Returns
    -------
    float
        Differential conductance in units of e^2/h.
    """
    if side not in ('left', 'right'):
        raise ValueError(f"side must be 'left' or 'right', got '{side}'")

    if energy <= Delta0:
        raise ValueError(
            f"energy ({energy} meV) must be strictly greater than Delta0 ({Delta0} meV) "
            "for normal above-gap calibration."
        )

    # Pin remote barrier to ensure strong decoupling even on small test wires
    remote_barrier = 50.0

    if side == 'left':
        barrier_l = float(u_barrier)
        barrier_r = remote_barrier
    else:
        barrier_l = remote_barrier
        barrier_r = float(u_barrier)

    syst = hp.build_system_exact(
        t=t,
        mu=mu,
        gamma=gamma,
        Delta0=Delta0,
        V_z=0.0,
        alpha=alpha,
        Ln=Ln,
        Lb=Lb,
        Ls=Ls,
        mu_leads=mu_leads,
        barrier_l=barrier_l,
        barrier_r=barrier_r,
        closed=False
    )

    G_mat = hp.calc_conductance_matrix_exact(
        syst,
        E=energy,
        solver_type=solver_type
    )

    if side == 'left':
        return float(G_mat[0, 0])
    else:
        return float(G_mat[1, 1])


def calibrate_single_junction(
    target_gn: float,
    side: str,
    t: float,
    gamma: float,
    Delta0: float,
    alpha: float,
    Ln: int = 0,
    Lb: int = 3,
    Ls: int = 300,
    mu_leads: Optional[float] = None,
    u_min: float = -3.0,
    u_max: float = 6.0,
    energy: float = 0.5,
    mu_points: Optional[np.ndarray] = None,
    tol: float = 1e-4,
    max_iter: int = 50
) -> float:
    """
    Inverts the normal-state conductance to find the cutter barrier potential U_c
    satisfying median_{mu}[G_N(U_c, mu)] = target_gn.

    Parameters
    ----------
    target_gn : float
        Target differential conductance in units of e^2/h.
    side : str
        'left' or 'right'.
    t, gamma, Delta0, alpha, Ln, Lb, Ls, mu_leads : float / int
        Nanowire Hamiltonian parameters.
    u_min, u_max : float
        Search bracket boundaries in meV.
    energy : float
        Above-gap bias energy in meV.
    mu_points : np.ndarray, optional
        Array of chemical potentials to median-average over (default: 15 points in [0.0, 2.0] meV).
    tol : float
        Tolerance on root finding in meV.
    max_iter : int
        Maximum iterations for brentq.

    Returns
    -------
    float
        Calibrated cutter gate potential U_c in meV.
    """
    if side not in ('left', 'right'):
        raise ValueError(f"side must be 'left' or 'right', got '{side}'")

    if u_min >= u_max:
        raise ValueError(f"u_min must be strictly less than u_max: u_min={u_min}, u_max={u_max}")

    # Physical conductance bounds for a 1-channel spin-degenerate normal lead
    if target_gn <= 0.0 or target_gn >= 2.0:
        raise ValueError(
            f"Target G_N = {target_gn} cannot be bracketed in [{u_min}, {u_max}] (out of physical bounds)"
        )

    if mu_points is None:
        mu_points = np.linspace(0.0, 2.0, 15)

    def objective(u: float) -> float:
        gns = [
            calculate_gn_above_gap(
                u_barrier=u,
                side=side,
                t=t,
                mu=mu,
                gamma=gamma,
                Delta0=Delta0,
                alpha=alpha,
                Ln=Ln,
                Lb=Lb,
                Ls=Ls,
                mu_leads=mu_leads,
                energy=energy
            )
            for mu in mu_points
        ]
        return float(np.median(gns) - target_gn)

    # Bracket validation
    f_min = objective(u_min)
    f_max = objective(u_max)

    if f_min * f_max > 0:
        raise ValueError(
            f"Target G_N = {target_gn} cannot be bracketed in [{u_min}, {u_max}] (out of physical bounds): "
            f"f({u_min}) = {f_min:+.4f}, f({u_max}) = {f_max:+.4f}"
        )

    root = brentq(objective, u_min, u_max, xtol=tol, maxiter=max_iter)
    return float(root)


def calibrate_wire_cutters(
    target_gns: Union[float, List[float], np.ndarray],
    t: float,
    gamma: float,
    Delta0: float,
    alpha: float,
    Ln: int = 0,
    Lb: int = 3,
    Ls: int = 300,
    mu_leads: Optional[float] = None,
    u_min: float = -3.0,
    u_max: float = 6.0,
    energy: float = 0.5,
    mu_points: Optional[np.ndarray] = None,
    tol: float = 1e-4
) -> Dict[str, np.ndarray]:
    """
    Calibrates both left and right cutter potentials for a suite of target G_N values.

    Returns
    -------
    dict
        {
            'targets': np.ndarray of target G_N values,
            'u_left': np.ndarray of calibrated U_c,L,
            'u_right': np.ndarray of calibrated U_c,R
        }
    """
    targets = np.atleast_1d(np.asarray(target_gns, dtype=float))
    u_left = np.zeros(len(targets), dtype=float)
    u_right = np.zeros(len(targets), dtype=float)

    for idx, tg in enumerate(targets):
        u_left[idx] = calibrate_single_junction(
            target_gn=tg,
            side='left',
            t=t,
            gamma=gamma,
            Delta0=Delta0,
            alpha=alpha,
            Ln=Ln,
            Lb=Lb,
            Ls=Ls,
            mu_leads=mu_leads,
            u_min=u_min,
            u_max=u_max,
            energy=energy,
            mu_points=mu_points,
            tol=tol
        )
        u_right[idx] = calibrate_single_junction(
            target_gn=tg,
            side='right',
            t=t,
            gamma=gamma,
            Delta0=Delta0,
            alpha=alpha,
            Ln=Ln,
            Lb=Lb,
            Ls=Ls,
            mu_leads=mu_leads,
            u_min=u_min,
            u_max=u_max,
            energy=energy,
            mu_points=mu_points,
            tol=tol
        )

    return {
        'targets': targets,
        'u_left': u_left,
        'u_right': u_right
    }
