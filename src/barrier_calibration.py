"""
Barrier Calibration Module (Aghaee et al. 2023)
================================================

Calibrates nanowire cutter gate potentials (UL, UR) to establish standardized
normal-state tunneling conductances G_N at above-gap bias energy E > Delta_0.
Each cutter gate is tuned individually to achieve target G_N.
"""

from typing import List, Tuple, Any
import numpy as np
from scipy.optimize import brentq
from omegaconf import DictConfig
import xarray as xr
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
    mu_leads: float
) -> Tuple[float, float]:
    """
    Computes left and right normal conductances (G_LL, G_RR)
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

    Returns:
    --------
    Tuple[float, float]
        (G_LL, G_RR) differential conductances in units of e^2/h.
    """
    if len(Vdisx) != geometry_cfg.Ls:
        raise ValueError(
            f"Vdisx length mismatch: got {len(Vdisx)}, expected length Ls={geometry_cfg.Ls}"
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


def calibrate_single_barrier(
    target_gn: float,
    side: str,
    calibration_cfg: DictConfig,
    geometry_cfg: DictConfig,
    Vdisx: np.ndarray,
    t: float,
    alpha: float,
    mu_leads: float,
    mu_points: np.array = None
) -> float:
    """
    Calibrates a single junction gate ('left' or 'right') individually to achieve
    target normal conductance G_N (averaged over chemical potentials mu).

    Parameters:
    -----------
    target_gn : float
        Target normal-state conductance in units of e^2/h.
    side : str
        Which junction cutter gate to tune: 'left' or 'right'.
    calibration_cfg : DictConfig
        Configuration dictionary for barrier calibration (from barrier_calibration.yaml).
    geometry_cfg : DictConfig
        Configuration dictionary for wire geometry and material parameters.
    Vdisx : np.ndarray
        1D spatial disorder profile of length geometry_cfg.Ls.
    t : float
        Hopping energy in meV.
    alpha : float
        Rashba spin-orbit coupling in meV.
    mu_leads : float
        Chemical potential in the normal leads in meV.

    Returns:
    --------
    float
        Calibrated cutter gate potential in meV.
    """
    if side not in ('left', 'right'):
        raise ValueError(f"side must be 'left' or 'right', got '{side}'")

    if target_gn <= 0.0 or target_gn >= 2.0:
        raise ValueError(f"Target G_N = {target_gn} out of physical bounds (0, 2) e^2/h")

    u_min = calibration_cfg.u_min
    u_max = calibration_cfg.u_max
    tol = calibration_cfg.tol
    max_iter = calibration_cfg.max_iter

    if u_min >= u_max:
        raise ValueError(f"u_min must be strictly less than u_max: u_min={u_min}, u_max={u_max}")

    if mu_points == None:
        mu_points = np.linspace(
            calibration_cfg.mu_min,
            calibration_cfg.mu_max,
            calibration_cfg.n_mu_points
        )

    lead_idx = 0 if side == 'left' else 1

    def objective(u_val: float) -> float:
        u_l = u_val if side == 'left' else 0.0
        u_r = 0.0 if side == 'left' else u_val
        g_vals = [
            calculate_junction_conductances(
                u_left=u_l,
                u_right=u_r,
                mu=mu_val,
                calibration_cfg=calibration_cfg,
                geometry_cfg=geometry_cfg,
                Vdisx=Vdisx,
                t=t,
                alpha=alpha,
                mu_leads=mu_leads
            )[lead_idx]
            for mu_val in mu_points
        ]
        return float(np.median(g_vals) - target_gn)

    # Robust monotonic bracketing
    # Left bracket: must have objective(u) > 0 (conductance above target)
    u_left_bracket = u_min
    f_min = objective(u_left_bracket)
    if f_min <= 0.0:
        # Search candidate points in the physical open-well regime
        peak_candidates = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]
        found_pos = False
        for u_cand in peak_candidates:
            if u_min <= u_cand <= u_max:
                f_cand = objective(u_cand)
                if f_cand > 0.0:
                    u_left_bracket = u_cand
                    f_min = f_cand
                    found_pos = True
                    break
        if not found_pos:
            raise ValueError(
                f"Target G_N={target_gn} cannot be bracketed on {side} junction in [{u_min}, {u_max}]: "
                f"maximum conductance is below target (f={f_min:+.4f})."
            )

    # Right bracket: must have objective(u) < 0 (conductance below target)
    u_right_bracket = u_max
    f_max = objective(u_right_bracket)
    if f_max >= 0.0:
        raise ValueError(
            f"Target G_N={target_gn} cannot be bracketed on {side} junction in [{u_min}, {u_max}]: "
            f"junction does not pinch off sufficiently (f={f_max:+.4f})."
        )

    # Contract right bracket if u_max is large to avoid wasting bisections in flat zero conductance
    if u_max > 15.0:
        for u_pinch in [5.0, 8.0, 12.0, 15.0]:
            if u_left_bracket < u_pinch < u_max:
                f_pinch = objective(u_pinch)
                if f_pinch < 0.0:
                    u_right_bracket = u_pinch
                    break

    u_star = brentq(objective, u_left_bracket, u_right_bracket, xtol=tol, maxiter=max_iter)
    return float(u_star)


def calibrate_symmetric_pair(
    target_gn: float,
    calibration_cfg: DictConfig,
    geometry_cfg: DictConfig,
    Vdisx: np.ndarray,
    t: float,
    alpha: float,
    mu_leads: float,
    mu_points: np.array = None
) -> List[float]:
    """
    Calibrates the left and right cutter gates individually to each yield target G_N.

    Parameters:
    -----------
    target_gn : float
        Target normal-state conductance in units of e^2/h.
    calibration_cfg : DictConfig
        Configuration dictionary for barrier calibration (from barrier_calibration.yaml).
    geometry_cfg : DictConfig
        Configuration dictionary for wire geometry and material parameters.
    Vdisx : np.ndarray
        1D spatial disorder profile of length geometry_cfg.Ls.
    t : float
        Hopping energy in meV.
    alpha : float
        Rashba spin-orbit coupling in meV.
    mu_leads : float
        Chemical potential in the normal leads in meV.

    Returns:
    --------
    List[float]
        [U_L*, U_R*] in meV.
    """
    if target_gn <= 0.0 or target_gn >= 2.0:
        raise ValueError(f"Target G_N = {target_gn} out of physical bounds (0, 2) e^2/h")

    u_l = calibrate_single_barrier(
        target_gn=target_gn,
        side='left',
        calibration_cfg=calibration_cfg,
        geometry_cfg=geometry_cfg,
        Vdisx=Vdisx,
        t=t,
        alpha=alpha,
        mu_leads=mu_leads,
        mu_points=mu_points
    )
    u_r = calibrate_single_barrier(
        target_gn=target_gn,
        side='right',
        calibration_cfg=calibration_cfg,
        geometry_cfg=geometry_cfg,
        Vdisx=Vdisx,
        t=t,
        alpha=alpha,
        mu_leads=mu_leads,
        mu_points=mu_points
    )
    return [u_l, u_r]


def calibrate_wire_cutters(
    targets: np.array,
    calibration_cfg: DictConfig,
    geometry_cfg: DictConfig,
    Vdisx: np.ndarray,
    t: float,
    alpha: float,
    mu_leads: float,
) -> List[List[Any]]:
    """
    Calibrates cutter barriers for all target conductances by tuning each gate individually.

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

    Returns:
    --------
    List of entries: [G_N, [UL, UR]]
    Example:
        [
            [0.25, [4.228, 4.228]],
            [0.35, [3.120, 3.120]],
            [0.45, [2.319, 2.319]],
            [0.55, [1.688, 1.688]],
            [0.65, [1.165, 1.165]]
        ]
    """
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
    g_n_coords = [row[0] for row in calibrated_table]
    matrix_values = [row[1] for row in calibrated_table]
    side_coords = ["left", "right"]

    ret = xr.DataArray(
        data=matrix_values,
        dims=["G_n", "side"],
        coords={
            "G_n": g_n_coords,
            "side": side_coords
        }
    )



    return ret
