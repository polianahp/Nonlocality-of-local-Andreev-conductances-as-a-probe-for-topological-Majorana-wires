"""
Barrier Calibration Module (Aghaee et al. 2023)
================================================

Calibrates nanowire cutter gate potentials (UL, UR) to establish standardized
normal-state tunneling conductances G_N at above-gap bias energy E > Delta_0.
Each cutter gate is tuned individually to achieve target G_N.
"""

from typing import List, Tuple, Any, Optional, Union
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
    mu_leads: float,
) -> Tuple[float, float]:
    """
    Computes left and right normal conductances (G_LL, G_RR)
    for a wire with cutter gate pair (u_left, u_right) at bias energy > Delta_0.
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
        closed=False,
    )

    G_mat = hp.calc_conductance_matrix_exact(syst, E=energy)
    return G_mat[0, 0], G_mat[1, 1]


def _eval_mu_conductance_worker(args: Tuple) -> float:
    """Module-level worker for evaluating single-mu conductance across an injected Pool."""
    u_l, u_r, mu_val, calibration_cfg, geometry_cfg, Vdisx, t, alpha, mu_leads, lead_idx = args
    return calculate_junction_conductances(
        u_left=u_l,
        u_right=u_r,
        mu=mu_val,
        calibration_cfg=calibration_cfg,
        geometry_cfg=geometry_cfg,
        Vdisx=Vdisx,
        t=t,
        alpha=alpha,
        mu_leads=mu_leads,
    )[lead_idx]


def _eval_median_conductance(
    u_val: float,
    side: str,
    calibration_cfg: DictConfig,
    geometry_cfg: DictConfig,
    Vdisx: np.ndarray,
    t: float,
    alpha: float,
    mu_leads: float,
    mu_points: Union[List[float], np.ndarray],
    pool: Optional[Any] = None,
) -> float:
    """Evaluates the median junction conductance across mu_points at barrier voltage u_val."""
    lead_idx = 0 if side == "left" else 1
    u_l = float(u_val) if side == "left" else 0.0
    u_r = 0.0 if side == "left" else float(u_val)

    if pool is not None and len(mu_points) > 1:
        tasks = [
            (u_l, u_r, float(mu_val), calibration_cfg, geometry_cfg, Vdisx, t, alpha, mu_leads, lead_idx)
            for mu_val in mu_points
        ]
        g_vals = pool.map(_eval_mu_conductance_worker, tasks)
    else:
        g_vals = [
            calculate_junction_conductances(
                u_left=u_l,
                u_right=u_r,
                mu=float(mu_val),
                calibration_cfg=calibration_cfg,
                geometry_cfg=geometry_cfg,
                Vdisx=Vdisx,
                t=t,
                alpha=alpha,
                mu_leads=mu_leads,
            )[lead_idx]
            for mu_val in mu_points
        ]
    return float(np.median(g_vals))


def _build_coarse_u_grid(u_min: float, u_max: float) -> np.ndarray:
    """
    Builds a coarse barrier potential grid spanning [u_min, u_max] to locate the
    disorder-dependent maximum conductance peak U_peak and bracket the pinch-off branch.
    """
    base_candidates = np.array(
        [-20.0, -16.0, -12.0, -9.0, -6.5, -4.5, -3.0, -1.5, 0.0, 1.5, 3.5, 6.0, 10.0, 18.0, 35.0],
        dtype=float,
    )
    interior = base_candidates[(base_candidates > u_min) & (base_candidates < u_max)]
    grid = np.unique(np.concatenate([[float(u_min)], interior, [float(u_max)]]))
    if len(grid) < 5:
        grid = np.linspace(float(u_min), float(u_max), 7)
    return grid


def calibrate_single_barrier(
    target_gn: float,
    side: str,
    calibration_cfg: DictConfig,
    geometry_cfg: DictConfig,
    Vdisx: np.ndarray,
    t: float,
    alpha: float,
    mu_leads: float,
    mu_points: Union[List[float], np.ndarray],
    pool: Optional[Any] = None,
    precomputed_sweep: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> float:
    """
    Calibrates a single junction gate ('left' or 'right') individually to achieve
    target normal conductance G_N (median over chemical potentials mu_points).
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got '{side}'")

    if target_gn <= 0.0 or target_gn >= 2.0:
        raise ValueError(f"Target G_N = {target_gn} out of physical bounds (0, 2) e^2/h")

    u_min = float(calibration_cfg.u_min)
    u_max = float(calibration_cfg.u_max)
    tol = float(calibration_cfg.tol)
    max_iter = int(calibration_cfg.max_iter)

    if u_min >= u_max:
        raise ValueError(f"u_min must be strictly less than u_max: u_min={u_min}, u_max={u_max}")

    if precomputed_sweep is not None:
        u_coarse, g_coarse = precomputed_sweep
    else:
        u_coarse = _build_coarse_u_grid(u_min, u_max)
        g_coarse = np.array([
            _eval_median_conductance(
                u_val=u_c,
                side=side,
                calibration_cfg=calibration_cfg,
                geometry_cfg=geometry_cfg,
                Vdisx=Vdisx,
                t=t,
                alpha=alpha,
                mu_leads=mu_leads,
                mu_points=mu_points,
                pool=pool,
            )
            for u_c in u_coarse
        ])

    # Locate the disorder-dependent global conductance peak across [u_min, u_max]
    idx_peak = int(np.argmax(g_coarse))
    u_peak = float(u_coarse[idx_peak])
    g_max = float(g_coarse[idx_peak])
    f_peak = g_max - target_gn

    if f_peak <= 0.0:
        raise ValueError(
            f"Target G_N={target_gn} cannot be bracketed on {side} junction in [{u_min}, {u_max}]: "
            f"maximum conductance across sweep is G_max={g_max:.4f} at U={u_peak:.2f} meV (f={f_peak:+.4f})."
        )

    # Restrict to the monotonic repulsive pinch-off branch U >= U_peak
    u_mon = u_coarse[idx_peak:]
    g_mon = g_coarse[idx_peak:]
    f_mon = g_mon - target_gn

    if f_mon[-1] >= 0.0:
        raise ValueError(
            f"Target G_N={target_gn} cannot be bracketed on {side} junction in [{u_min}, {u_max}]: "
            f"junction does not pinch off sufficiently at u_max={u_max} (f={f_mon[-1]:+.4f})."
        )

    # Select tightest adjacent bracket [u_left_bracket, u_right_bracket] on U >= U_peak
    below_indices = np.where(f_mon < 0.0)[0]
    first_below = int(below_indices[0])
    last_above = max(0, first_below - 1)

    u_left_bracket = float(u_mon[last_above])
    u_right_bracket = float(u_mon[first_below])
    f_left = float(f_mon[last_above])
    f_right = float(f_mon[first_below])

    if np.isclose(f_left, 0.0, atol=1e-12):
        return u_left_bracket

    eval_cache = {
        u_left_bracket: f_left,
        u_right_bracket: f_right,
    }

    def objective(u_val: float) -> float:
        u_key = float(u_val)
        if u_key in eval_cache:
            return eval_cache[u_key]
        g_med = _eval_median_conductance(
            u_val=u_key,
            side=side,
            calibration_cfg=calibration_cfg,
            geometry_cfg=geometry_cfg,
            Vdisx=Vdisx,
            t=t,
            alpha=alpha,
            mu_leads=mu_leads,
            mu_points=mu_points,
            pool=pool,
        )
        val = float(g_med - target_gn)
        eval_cache[u_key] = val
        return val

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
    mu_points: Union[List[float], np.ndarray],
    pool: Optional[Any] = None,
    precomputed_left_sweep: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    precomputed_right_sweep: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> List[float]:
    """
    Calibrates the left and right cutter gates individually to each yield target G_N.
    """
    if target_gn <= 0.0 or target_gn >= 2.0:
        raise ValueError(f"Target G_N = {target_gn} out of physical bounds (0, 2) e^2/h")

    u_l = calibrate_single_barrier(
        target_gn=target_gn,
        side="left",
        calibration_cfg=calibration_cfg,
        geometry_cfg=geometry_cfg,
        Vdisx=Vdisx,
        t=t,
        alpha=alpha,
        mu_leads=mu_leads,
        mu_points=mu_points,
        pool=pool,
        precomputed_sweep=precomputed_left_sweep,
    )
    u_r = calibrate_single_barrier(
        target_gn=target_gn,
        side="right",
        calibration_cfg=calibration_cfg,
        geometry_cfg=geometry_cfg,
        Vdisx=Vdisx,
        t=t,
        alpha=alpha,
        mu_leads=mu_leads,
        mu_points=mu_points,
        pool=pool,
        precomputed_sweep=precomputed_right_sweep,
    )
    return [u_l, u_r]


def calibrate_wire_cutters(
    targets: Union[List[float], np.ndarray],
    calibration_cfg: DictConfig,
    geometry_cfg: DictConfig,
    Vdisx: np.ndarray,
    t: float,
    alpha: float,
    mu_leads: float,
    mu_points: Union[List[float], np.ndarray],
    pool: Optional[Any] = None,
) -> xr.DataArray:
    """
    Calibrates cutter barriers for all target conductances by tuning each gate individually
    over the chemical potential set mu_points.
    Returns xr.DataArray with dims ["G_n", "side"].
    """
    u_min = float(calibration_cfg.u_min)
    u_max = float(calibration_cfg.u_max)
    if u_min >= u_max:
        raise ValueError(f"u_min must be strictly less than u_max: u_min={u_min}, u_max={u_max}")

    g_n_coords = [float(tg) for tg in targets]
    for tg in g_n_coords:
        if tg <= 0.0 or tg >= 2.0:
            raise ValueError(f"Target G_N = {tg} out of physical bounds (0, 2) e^2/h")

    # Evaluate coarse open-barrier sweep once per side to locate disorder-dependent U_peak
    u_coarse = _build_coarse_u_grid(u_min, u_max)
    g_coarse_left = np.array([
        _eval_median_conductance(
            u_val=u_c,
            side="left",
            calibration_cfg=calibration_cfg,
            geometry_cfg=geometry_cfg,
            Vdisx=Vdisx,
            t=t,
            alpha=alpha,
            mu_leads=mu_leads,
            mu_points=mu_points,
            pool=pool,
        )
        for u_c in u_coarse
    ])
    g_coarse_right = np.array([
        _eval_median_conductance(
            u_val=u_c,
            side="right",
            calibration_cfg=calibration_cfg,
            geometry_cfg=geometry_cfg,
            Vdisx=Vdisx,
            t=t,
            alpha=alpha,
            mu_leads=mu_leads,
            mu_points=mu_points,
            pool=pool,
        )
        for u_c in u_coarse
    ])

    left_sweep = (u_coarse, g_coarse_left)
    right_sweep = (u_coarse, g_coarse_right)

    matrix_values = []
    for tg in g_n_coords:
        pair = calibrate_symmetric_pair(
            target_gn=tg,
            calibration_cfg=calibration_cfg,
            geometry_cfg=geometry_cfg,
            Vdisx=Vdisx,
            t=t,
            alpha=alpha,
            mu_leads=mu_leads,
            mu_points=mu_points,
            pool=pool,
            precomputed_left_sweep=left_sweep,
            precomputed_right_sweep=right_sweep,
        )
        matrix_values.append(pair)

    return xr.DataArray(
        data=matrix_values,
        dims=["G_n", "side"],
        coords={
            "G_n": g_n_coords,
            "side": ["left", "right"],
        },
    )
