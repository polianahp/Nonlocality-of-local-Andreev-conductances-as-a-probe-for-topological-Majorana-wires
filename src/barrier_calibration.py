"""
Barrier Calibration Module (Aghaee et al. 2023)
================================================

Calibrates nanowire cutter gate potentials (UL, UR) to establish standardized
normal-state tunneling conductances G_N at above-gap bias energy E > Delta_0.
Each cutter gate is tuned individually to achieve target G_N.
"""

from typing import List, Tuple, Any, Optional, Union
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from omegaconf import DictConfig
import xarray as xr
import src.helpers as hp


class UnhealthyWireError(RuntimeError):
    """Raised when a nanowire fails to pinch off below pinch_off_tol at U = u_max."""


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


def check_wire_health(
    calibration_cfg: DictConfig,
    geometry_cfg: DictConfig,
    Vdisx: np.ndarray,
    t: float,
    alpha: float,
    mu_leads: float,
    mu_points: Union[List[float], np.ndarray],
    pool: Optional[Any] = None,
) -> Tuple[float, float]:
    """
    Pre-health check at the pinch-off limit U = u_max.

    Evaluates median junction conductances on both left and right junctions.
    Raises UnhealthyWireError if either junction exceeds pinch_off_tol.
    """
    u_pinch = float(calibration_cfg.u_max)
    pinch_tol = float(calibration_cfg.pinch_off_tol)

    g_left = _eval_median_conductance(
        u_pinch, "left", calibration_cfg, geometry_cfg,
        Vdisx, t, alpha, mu_leads, mu_points, pool=pool,
    )
    g_right = _eval_median_conductance(
        u_pinch, "right", calibration_cfg, geometry_cfg,
        Vdisx, t, alpha, mu_leads, mu_points, pool=pool,
    )

    if g_left > pinch_tol or g_right > pinch_tol:
        raise UnhealthyWireError(
            f"Unhealthy wire at pinch-off U={u_pinch:.2f} meV: "
            f"G_left={g_left:.3e}, G_right={g_right:.3e} > tol={pinch_tol:.3e}"
        )
    return g_left, g_right


def _find_conductance_top(
    side: str,
    calibration_cfg: DictConfig,
    geometry_cfg: DictConfig,
    Vdisx: np.ndarray,
    t: float,
    alpha: float,
    mu_leads: float,
    mu_points: Union[List[float], np.ndarray],
    pool: Optional[Any] = None,
) -> Tuple[float, float]:
    """
    Locates the open-barrier conductance peak (U_top, G_max) by stepping inward
    from peak_search_start_u (repulsive side) via scipy.optimize.minimize_scalar(method='brent').
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    start_u = float(calibration_cfg.peak_search_start_u)
    step = float(calibration_cfg.peak_search_step)
    u_lo_limit = float(calibration_cfg.u_min)
    u_hi_limit = float(calibration_cfg.u_max)

    def obj(u_val: float) -> float:
        u_f = float(u_val)
        if u_f < u_lo_limit:
            return 1e6 + (u_lo_limit - u_f)
        if u_f > u_hi_limit:
            return 1e6 + (u_f - u_hi_limit)
        return -_eval_median_conductance(
            u_f, side, calibration_cfg, geometry_cfg,
            Vdisx, t, alpha, mu_leads, mu_points, pool=pool,
        )

    res = minimize_scalar(
        obj,
        bracket=(start_u, start_u - step),
        method="brent",
        options={"xtol": float(calibration_cfg.tol)},
    )
    if not res.success:
        raise RuntimeError(f"Peak search failed for side={side!r}: {res.message}")

    u_top = float(np.clip(res.x, u_lo_limit, u_hi_limit))
    g_max = _eval_median_conductance(
        u_top, side, calibration_cfg, geometry_cfg,
        Vdisx, t, alpha, mu_leads, mu_points, pool=pool,
    )
    return u_top, g_max


def _solve_repulsive_barrier(
    target_gn: float,
    side: str,
    u_top: float,
    g_max: float,
    calibration_cfg: DictConfig,
    geometry_cfg: DictConfig,
    Vdisx: np.ndarray,
    t: float,
    alpha: float,
    mu_leads: float,
    mu_points: Union[List[float], np.ndarray],
    pool: Optional[Any] = None,
) -> float:
    """
    Solves G(U) = target_gn on the monotonic repulsive branch [u_top, u_max].
    """
    u_max = float(calibration_cfg.u_max)
    if u_top >= u_max:
        raise ValueError(f"u_top ({u_top}) must be strictly less than u_max ({u_max})")
    if target_gn <= 0.0 or target_gn > g_max + 1e-4:
        raise ValueError(
            f"Target G_N={target_gn:.4f} on side={side!r} is outside reachable "
            f"repulsive range (0, {g_max:.4f}]"
        )

    def objective(u_val: float) -> float:
        return _eval_median_conductance(
            u_val, side, calibration_cfg, geometry_cfg,
            Vdisx, t, alpha, mu_leads, mu_points, pool=pool,
        ) - target_gn

    f_lo = objective(u_top)
    if -1e-4 <= f_lo < 0.0:
        return float(u_top)

    f_hi = objective(u_max)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"Root not bracketed on [{u_top:.3f}, {u_max:.3f}] for side={side!r}, "
            f"target={target_gn:.4f}: f(u_top)={f_lo:.4e}, f(u_max)={f_hi:.4e}"
        )

    return float(
        brentq(
            objective,
            u_top,
            u_max,
            xtol=float(calibration_cfg.tol),
            maxiter=int(calibration_cfg.max_iter),
        )
    )


def calibrate_stage1_wire_cutters(
    nominal_targets: Union[List[float], np.ndarray],
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
    Calibrates Stage 1 (and barrier-sweep) cutter gate potentials over the
    median of mu_points on the monotonic repulsive branch [u_top, u_max].

    Targets above G_top = min(G_max_left, G_max_right) are truncated and
    floor_2dec(G_top) is included (replacing the last kept target if within
    sweep_gnres so the widest boundary is preserved). Returns an xr.DataArray
    named 'stage1_cutters' with dims ('G_n', 'barrier_side').
    """
    u_top_l, g_max_l = _find_conductance_top(
        "left", calibration_cfg, geometry_cfg, Vdisx, t, alpha, mu_leads, mu_points, pool=pool
    )
    u_top_r, g_max_r = _find_conductance_top(
        "right", calibration_cfg, geometry_cfg, Vdisx, t, alpha, mu_leads, mu_points, pool=pool
    )

    g_top_raw = min(g_max_l, g_max_r)
    if g_top_raw <= 0.0:
        raise ValueError(f"Non-positive G_top={g_top_raw:.4f} (g_max_l={g_max_l:.4f}, g_max_r={g_max_r:.4f})")

    nominal_arr = np.asarray(nominal_targets, dtype=float)
    kept = nominal_arr[nominal_arr <= g_top_raw]
    truncated = len(kept) < len(nominal_arr)

    g_top_rounded = float(np.floor(g_top_raw * 100.0) / 100.0)
    gn_res = float(calibration_cfg.sweep_gnres)

    if truncated and g_top_rounded > 0.0:
        targets_list = sorted(set(np.round(kept, 2).tolist()))
        if not targets_list:
            targets_list = [g_top_rounded]
        elif abs(g_top_rounded - targets_list[-1]) < gn_res:
            targets_list[-1] = max(targets_list[-1], g_top_rounded)
        else:
            targets_list.append(g_top_rounded)
        targets = np.sort(np.unique(np.asarray(targets_list, dtype=float)))
    else:
        targets = np.sort(np.unique(np.round(kept, 2)))

    if len(targets) == 0:
        raise ValueError(f"No reachable Stage 1 targets below G_top={g_top_raw:.4f}")

    data = np.full((len(targets), 2), np.nan, dtype=float)
    for idx, gn in enumerate(targets):
        data[idx, 0] = _solve_repulsive_barrier(
            float(gn), "left", u_top_l, g_max_l,
            calibration_cfg, geometry_cfg, Vdisx, t, alpha, mu_leads, mu_points, pool=pool,
        )
        data[idx, 1] = _solve_repulsive_barrier(
            float(gn), "right", u_top_r, g_max_r,
            calibration_cfg, geometry_cfg, Vdisx, t, alpha, mu_leads, mu_points, pool=pool,
        )

    assert not np.isnan(data).any(), "Stage 1 cutter calibration produced NaN"

    return xr.DataArray(
        data=data,
        dims=("G_n", "barrier_side"),
        coords={
            "G_n": targets,
            "barrier_side": ["left", "right"],
        },
        name="stage1_cutters",
        attrs={
            "u_top_left": u_top_l,
            "u_top_right": u_top_r,
            "g_max_left": g_max_l,
            "g_max_right": g_max_r,
            "g_top": g_top_rounded,
        },
    )


def _stage2_mu_peak_worker(args: Tuple) -> Tuple[float, float, float, float]:
    """Module-level worker for parallelizing Stage 2 peak finding over mu."""
    mu_val, calib, geom, Vdisx, t, alpha, mu_leads = args
    mu_list = [float(mu_val)]
    u_top_l, g_max_l = _find_conductance_top(
        "left", calib, geom, Vdisx, t, alpha, mu_leads, mu_list, pool=None
    )
    u_top_r, g_max_r = _find_conductance_top(
        "right", calib, geom, Vdisx, t, alpha, mu_leads, mu_list, pool=None
    )
    return u_top_l, g_max_l, u_top_r, g_max_r


def _stage2_mu_solve_worker(args: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Module-level worker for parallelizing Stage 2 root finding over mu."""
    mu_val, targets, u_top_l, g_max_l, u_top_r, g_max_r, calib, geom, Vdisx, t, alpha, mu_leads = args
    mu_list = [float(mu_val)]
    u_l_row = np.full(len(targets), np.nan, dtype=float)
    u_r_row = np.full(len(targets), np.nan, dtype=float)
    for idx, gn in enumerate(targets):
        u_l_row[idx] = _solve_repulsive_barrier(
            float(gn), "left", u_top_l, g_max_l,
            calib, geom, Vdisx, t, alpha, mu_leads, mu_list, pool=None,
        )
        u_r_row[idx] = _solve_repulsive_barrier(
            float(gn), "right", u_top_r, g_max_r,
            calib, geom, Vdisx, t, alpha, mu_leads, mu_list, pool=None,
        )
    return u_l_row, u_r_row


def calibrate_stage2_wire_cutters(
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
    Calibrates Stage 2 cutter gate potentials at each chemical potential on the
    monotonic repulsive branch [u_top(mu), u_max].

    Uses nominal calibration_cfg.stage2_target_Gns whenever reachable across all
    chemical potentials (G_top >= max(stage2_target_Gns)). Otherwise reconstructs
    evenly spaced targets on [stage2_gn_min, G_top], decrementing n_c from
    len(stage2_target_Gns) down to 2 until min(diff(G_n)) >= stage2_min_gn_spacing.
    Returns an xr.DataArray named 'stage2_cutters' with dims
    ('G_n', 'chemical_potential', 'barrier_side').
    """
    nominal_targets = np.sort(np.asarray(calibration_cfg.stage2_target_Gns, dtype=float))
    if nominal_targets.ndim != 1 or len(nominal_targets) < 2:
        raise ValueError(f"stage2_target_Gns must be a 1D list of >= 2 targets, got {nominal_targets}")

    mu_arr = np.asarray(mu_points, dtype=float)
    peak_tasks = [
        (float(mu_val), calibration_cfg, geometry_cfg, Vdisx, t, alpha, mu_leads)
        for mu_val in mu_arr
    ]
    if pool is not None:
        peak_results = pool.map(_stage2_mu_peak_worker, peak_tasks)
    else:
        peak_results = [_stage2_mu_peak_worker(task) for task in peak_tasks]

    u_tops_l = np.array([r[0] for r in peak_results], dtype=float)
    g_maxs_l = np.array([r[1] for r in peak_results], dtype=float)
    u_tops_r = np.array([r[2] for r in peak_results], dtype=float)
    g_maxs_r = np.array([r[3] for r in peak_results], dtype=float)

    raw_top = min(float(np.min(g_maxs_l)), float(np.min(g_maxs_r)))
    g_top = float(np.floor(raw_top * 100.0) / 100.0)

    if g_top >= float(np.max(nominal_targets)):
        chosen_targets = nominal_targets
    else:
        gn_min = float(calibration_cfg.stage2_gn_min)
        if raw_top <= gn_min:
            raise ValueError(
                f"Stage 2 G_top ({raw_top:.4f}) is not above stage2_gn_min ({gn_min})"
            )
        min_spacing = float(calibration_cfg.stage2_min_gn_spacing)
        max_n_c = len(nominal_targets)

        chosen_targets: Optional[np.ndarray] = None
        for n_c in range(max_n_c, 1, -1):
            candidate = np.round(np.linspace(gn_min, g_top, n_c), 2)
            if len(np.unique(candidate)) == n_c and np.min(np.diff(candidate)) >= min_spacing - 1e-12:
                chosen_targets = candidate
                break

        if chosen_targets is None:
            raise ValueError(
                f"Cannot satisfy stage2_min_gn_spacing={min_spacing} between "
                f"stage2_gn_min={gn_min} and g_top={g_top}"
            )

    solve_tasks = [
        (
            float(mu_arr[i]),
            chosen_targets,
            float(u_tops_l[i]),
            float(g_maxs_l[i]),
            float(u_tops_r[i]),
            float(g_maxs_r[i]),
            calibration_cfg,
            geometry_cfg,
            Vdisx,
            t,
            alpha,
            mu_leads,
        )
        for i in range(len(mu_arr))
    ]
    if pool is not None:
        solve_results = pool.map(_stage2_mu_solve_worker, solve_tasks)
    else:
        solve_results = [_stage2_mu_solve_worker(task) for task in solve_tasks]

    data = np.full((len(chosen_targets), len(mu_arr), 2), np.nan, dtype=float)
    for mu_idx, (u_l_row, u_r_row) in enumerate(solve_results):
        data[:, mu_idx, 0] = u_l_row
        data[:, mu_idx, 1] = u_r_row

    assert not np.isnan(data).any(), "Stage 2 cutter calibration produced NaN"

    return xr.DataArray(
        data=data,
        dims=("G_n", "chemical_potential", "barrier_side"),
        coords={
            "G_n": chosen_targets,
            "chemical_potential": np.round(mu_arr, 2),
            "barrier_side": ["left", "right"],
        },
        name="stage2_cutters",
        attrs={
            "g_top": g_top,
        },
    )
