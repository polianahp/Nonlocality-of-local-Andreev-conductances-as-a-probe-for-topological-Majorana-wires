"""
Barrier Calibration Module (Aghaee et al. 2023)
================================================


    
"""

from typing import List, Tuple, Any
import numpy as np
from scipy.optimize import brentq
from omegaconf import DictConfig

import src.helpers as hp


def calculate_junction_conductances(
    u_left: float,
    u_right: float,
    mu: float,
    calibration_cfg: DictConfig,
    geometry_cfg: DictConfig,
    Vdisx: np.ndarray,
    t: float,
    alpha: float,
    mu_leads: float,
    solver_type: str = "cpu"
) -> Tuple[float, float]:
    """
    Simultaneously computes left and right normal conductances (G_LL, G_RR)
    for a wire with cutter gate pair (u_left, u_right) at bias energy > Delta_0.

    Parameters:
    -----------
    u_left : float
        Left cutter gate potential in meV.
    u_right : float
        Right cutter gate potential in meV.
    mu : float
        Superconductor chemical potential in meV.
    calibration_cfg : DictConfig
        Configuration dictionary for barrier calibration (from barrier_calibration.yaml).
    geometry_cfg : DictConfig
        Configuration dictionary for wire geometry and material parameters.
    Vdisx : np.ndarray
        1D spatial disorder profile of length geometry_cfg.Ls.
    t : float
        Hopping energy in meV (derived in SimulationState).
    alpha : float
        Rashba spin-orbit coupling in meV (derived in SimulationState).
    mu_leads : float
        Chemical potential in the normal leads in meV (derived in SimulationState).
    solver_type : str
        Kwant solver type ('cpu' or 'gpu').

    Returns:
    --------
    Tuple[float, float]
        (G_LL, G_RR) differential conductances in units of e^2/h.
    """
    if len(Vdisx) != geometry_cfg.Ls:
        raise ValueError(
            f"Vdisx length mismatch: got {len(Vdisx)}, expected geometry Ls={geometry_cfg.Ls}"
        )

    energy = calibration_cfg.energy
    Delta0 = geometry_cfg.Delta_0
    if energy <= Delta0:
        raise ValueError(
            f"Calibration bias energy ({energy} meV) must be strictly greater than Delta_0 ({Delta0} meV)"
        )

    syst = hp.build_system_exact(
        t=t,
        mu=mu,
        gamma=geometry_cfg.gamma,
        Delta0=Delta0,
        V_z=0.0,
        alpha=alpha,
        Ln=geometry_cfg.Ln,
        Lb=geometry_cfg.Lb,
        Ls=geometry_cfg.Ls,
        mu_leads=mu_leads,
        barrier_l=u_left,
        barrier_r=u_right,
        Vdisx=Vdisx,
        closed=False
    )

    G_mat = hp.calc_conductance_matrix_exact(syst, E=energy)
    return G_mat[0, 0], G_mat[1, 1]


def calibrate_symmetric_pair(
    target_gn: float,
    calibration_cfg: DictConfig,
    geometry_cfg: DictConfig,
    Vdisx: np.ndarray,
    t: float,
    alpha: float,
    mu_leads: float,
    solver_type: str = "cpu"
) -> List[float]:
    """
    Finds the symmetric cutter gate pair [U_L*, U_R*] that simultaneously satisfies:
        median_mu[G_LL(U_L*, U_R*)] = target_gn
        median_mu[G_RR(U_L*, U_R*)] = target_gn

    Returns:
    --------
    List[float]
        [U_L*, U_R*] in meV.
    """
    u_min = calibration_cfg.u_min
    u_max = calibration_cfg.u_max
    tol = calibration_cfg.tol
    max_iter = calibration_cfg.max_iter

    if u_min >= u_max:
        raise ValueError(f"u_min must be strictly less than u_max: u_min={u_min}, u_max={u_max}")

    if target_gn <= 0.0 or target_gn >= 2.0:
        raise ValueError(f"Target G_N = {target_gn} out of physical bounds (0, 2) e^2/h")

    mu_points = np.linspace(
        calibration_cfg.mu_min,
        calibration_cfg.mu_max,
        calibration_cfg.n_mu_points
    )

    # 1. Symmetric baseline solve: find u_sym where the average conductance matches target_gn
    def objective_symmetric(u: float) -> float:
        joint_gns = [
            calculate_junction_conductances(
                u_left=u,
                u_right=u,
                mu=mu_val,
                calibration_cfg=calibration_cfg,
                geometry_cfg=geometry_cfg,
                Vdisx=Vdisx,
                t=t,
                alpha=alpha,
                mu_leads=mu_leads,
                solver_type=solver_type
            )
            for mu_val in mu_points
        ]
        avg_g = np.median([(g[0] + g[1]) / 2.0 for g in joint_gns])
        return avg_g - target_gn

    f_sym_min = objective_symmetric(u_min)
    f_sym_max = objective_symmetric(u_max)
    if f_sym_min * f_sym_max > 0:
        raise ValueError(
            f"Target G_N={target_gn} cannot be bracketed in [{u_min}, {u_max}]: "
            f"f({u_min})={f_sym_min:+.4f}, f({u_max})={f_sym_max:+.4f}"
        )

    u_sym = brentq(objective_symmetric, u_min, u_max, xtol=tol, maxiter=max_iter)

    # 2. Refine left and right barriers around u_sym to account for spatial disorder asymmetry
    def objective_left(ul: float) -> float:
        g_lefts = [
            calculate_junction_conductances(
                u_left=ul,
                u_right=u_sym,
                mu=mu_val,
                calibration_cfg=calibration_cfg,
                geometry_cfg=geometry_cfg,
                Vdisx=Vdisx,
                t=t,
                alpha=alpha,
                mu_leads=mu_leads,
                solver_type=solver_type
            )[0]
            for mu_val in mu_points
        ]
        return np.median(g_lefts) - target_gn

    def objective_right(ur: float) -> float:
        g_rights = [
            calculate_junction_conductances(
                u_left=u_sym,
                u_right=ur,
                mu=mu_val,
                calibration_cfg=calibration_cfg,
                geometry_cfg=geometry_cfg,
                Vdisx=Vdisx,
                t=t,
                alpha=alpha,
                mu_leads=mu_leads,
                solver_type=solver_type
            )[1]
            for mu_val in mu_points
        ]
        return np.median(g_rights) - target_gn

    f_l_min = objective_left(u_min)
    f_l_max = objective_left(u_max)
    if f_l_min * f_l_max > 0:
        raise ValueError(
            f"Target G_N={target_gn} cannot be bracketed on Left junction in [{u_min}, {u_max}]: "
            f"f({u_min})={f_l_min:+.4f}, f({u_max})={f_l_max:+.4f}. "
            f"Impurity scattering or potential limits may restrict maximum attainable transmission."
        )

    f_r_min = objective_right(u_min)
    f_r_max = objective_right(u_max)
    if f_r_min * f_r_max > 0:
        raise ValueError(
            f"Target G_N={target_gn} cannot be bracketed on Right junction in [{u_min}, {u_max}]: "
            f"f({u_min})={f_r_min:+.4f}, f({u_max})={f_r_max:+.4f}. "
            f"Impurity scattering or potential limits may restrict maximum attainable transmission."
        )

    u_l_star = brentq(objective_left, u_min, u_max, xtol=tol, maxiter=max_iter)
    u_r_star = brentq(objective_right, u_min, u_max, xtol=tol, maxiter=max_iter)

    # 2. Joint verification with both calibrated gates active simultaneously
    joint_conductances = [
        calculate_junction_conductances(
            u_left=u_l_star,
            u_right=u_r_star,
            mu=mu_val,
            calibration_cfg=calibration_cfg,
            geometry_cfg=geometry_cfg,
            Vdisx=Vdisx,
            t=t,
            alpha=alpha,
            mu_leads=mu_leads,
            solver_type=solver_type
        )
        for mu_val in mu_points
    ]
    med_gll = np.median([g[0] for g in joint_conductances])
    med_grr = np.median([g[1] for g in joint_conductances])

    err_l = abs(med_gll - target_gn)
    err_r = abs(med_grr - target_gn)
    # Allow 2 * tol for residual joint coupling
    tolerance_threshold = max(2.0 * tol, 0.005)
    if err_l > tolerance_threshold or err_r > tolerance_threshold:
        raise RuntimeError(
            f"Simultaneous calibration failed tolerance for G_N={target_gn}: "
            f"G_LL error={err_l:.4f}, G_RR error={err_r:.4f} (threshold={tolerance_threshold})"
        )

    return [u_l_star, u_r_star]


def calibrate_single_junction(
    target_gn: float,
    side: str,
    calibration_cfg: DictConfig,
    geometry_cfg: DictConfig,
    Vdisx: np.ndarray,
    t: float,
    alpha: float,
    mu_leads: float,
    solver_type: str = "cpu"
) -> float:
    """
    Calibrates a single junction ('left' or 'right') as part of the symmetric pair.
    Convenience wrapper returning a scalar potential.
    """
    if side not in ('left', 'right'):
        raise ValueError(f"side must be 'left' or 'right', got '{side}'")

    pair = calibrate_symmetric_pair(
        target_gn=target_gn,
        calibration_cfg=calibration_cfg,
        geometry_cfg=geometry_cfg,
        Vdisx=Vdisx,
        t=t,
        alpha=alpha,
        mu_leads=mu_leads,
        solver_type=solver_type
    )
    return pair[0] if side == 'left' else pair[1]


def calibrate_wire_cutters(
    calibration_cfg: DictConfig,
    geometry_cfg: DictConfig,
    Vdisx: np.ndarray,
    t: float,
    alpha: float,
    mu_leads: float,
) -> List[List[Any]]:
    """
    Calibrates symmetric barrier configurations for all target conductances.

    Parameters:
    -----------
    calibration_cfg : DictConfig
        Configuration dictionary containing target_gns, u_min, u_max, energy, tol, etc.
    geometry_cfg : DictConfig
        Configuration dictionary containing Ls, Ln, Lb, gamma, Delta_0, etc.
    Vdisx : np.ndarray
        MANDATORY 1D spatial disorder profile of length geometry_cfg.Ls.
    t : float
        Hopping energy in meV.
    alpha : float
        Rashba spin-orbit coupling in meV.
    mu_leads : float
        Chemical potential in the normal leads in meV.
    solver_type : str
        Kwant solver type ('cpu' or 'gpu').

    Returns:
    --------
    List of entries: [G_N, [UL, UR]]
    Example:
        [
            [0.35, [3.120, 2.985]],
            [0.49, [2.410, 2.350]],
            [0.62, [1.820, 1.790]],
            [0.76, [1.110, 1.090]],
            [0.90, [0.110, 0.095]]
        ]
    """
    targets = np.atleast_1d(np.asarray(calibration_cfg.target_gns, dtype=float))
    calibrated_table = []

    for tg in targets:
        pair = calibrate_symmetric_pair(
            target_gn=tg,
            calibration_cfg=calibration_cfg,
            geometry_cfg=geometry_cfg,
            Vdisx=Vdisx,
            t=t,
            alpha=alpha,
            mu_leads=mu_leads,
        )
        calibrated_table.append([tg, pair])

    return calibrated_table
