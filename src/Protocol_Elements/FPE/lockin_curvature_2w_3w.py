r"""Zero-Bias $3\omega$ Curvature & Nonlocal $2\omega$ Lock-In Observables FPEs ($C_1$).

Implements:
- `fpe_zero_bias_curvature_single_cutter` (User's $C_1$):
    $$\left.\frac{\partial^2 G_\alpha^{(c)}}{\partial V^2}\right|_{V=0} \le -t_{3\omega}$$
- `fpe_cutter_averaged_curvature`: Continuous cutter/lead average of zero-bias curvature $\le -t_{3\omega}$
- `fpe_nonlocal_2w_bounded`: Nonlocal $2\omega$ lock-in derivative bounded within $[-t_{2\omega}, +t_{2\omega}]$.

Explicitly resolves the Round 1 Critique 1 scaling and thermal-broadening distinctions via:
- `curvature_source`: `"thermal_lockin_3w"` (quadratic B-spline 2nd derivative on thermally
  broadened $G(V)$ at temperature $T$), `"savgol_thermal"`, or `"savgol_T0"`
- `scale_mode`: `"physical_dynamic_grid"` ($\Delta V_{\text{center}} = 0.0005\,\text{meV}$ from
  `hp.make_dynamic_grid()`) vs. `"legacy_uniform_97"` ($\Delta V_{\text{legacy}} = 0.268 / 96 = 0.00279167\,\text{meV}$).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from ..cache import RealizationFeatureCache


def extract_zero_bias_curvature_3d(
    cache: RealizationFeatureCache,
    *,
    curvature_source: Literal["thermal_lockin_3w", "savgol_thermal", "savgol_T0"] = "thermal_lockin_3w",
    scale_mode: Literal["physical_dynamic_grid", "legacy_uniform_97"] = "physical_dynamic_grid",
    savgol_window_length: int = 7,
    savgol_polyorder: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return memoized Rank-3 `(d2G_L, d2G_R)` arrays of shape `(N_cutters, N_mu, N_Vz)` at $V=0$."""
    if curvature_source == "thermal_lockin_3w":
        lockin = cache.get_stage1_broadened_and_lockin()
        return lockin["I_3w_L"], lockin["I_3w_R"]

    w_len = int(savgol_window_length)
    if w_len % 2 == 0:
        w_len += 1
    # Guard against crossing the non-uniform dynamic grid spacing boundary at indices 43..53
    if scale_mode == "physical_dynamic_grid" and w_len > 11:
        w_len = 11

    key = (
        "savgol_zero_bias_curvature_3d",
        curvature_source,
        scale_mode,
        w_len,
        int(savgol_polyorder),
        round(cache.temperature_mK, 4),
    )

    def _compute() -> Tuple[np.ndarray, np.ndarray]:
        bias = cache.bias_grid
        mid_idx = int(np.argmin(np.abs(bias - 0.0)))
        if scale_mode == "physical_dynamic_grid":
            delta = float(bias[mid_idx + 1] - bias[mid_idx])  # 0.0005 meV
        elif scale_mode == "legacy_uniform_97":
            delta = float((bias[-1] - bias[0]) / max(len(bias) - 1, 1))  # 0.268 / 96 meV
        else:
            raise ValueError(f"Unknown scale_mode '{scale_mode}'.")

        if curvature_source == "savgol_thermal":
            src = cache.get_stage1_broadened_and_lockin()
            g_ll, g_rr = src["g_ll"], src["g_rr"]
        elif curvature_source == "savgol_T0":
            src = cache.get_stage1_raw()
            g_ll = np.nan_to_num(src["g_ll"], nan=0.0)
            g_rr = np.nan_to_num(src["g_rr"], nan=0.0)
        else:
            raise ValueError(f"Unknown curvature_source '{curvature_source}'.")

        # Slice stencil around zero bias so savgol_filter only touches the central window!
        half = w_len // 2
        sl = slice(max(0, mid_idx - half), min(len(bias), mid_idx + half + 1))
        rel_mid = mid_idx - sl.start

        d2_L = savgol_filter(
            g_ll[:, :, :, sl],
            window_length=w_len,
            polyorder=int(savgol_polyorder),
            deriv=2,
            delta=delta,
            axis=-1,
        )[:, :, :, rel_mid]
        d2_R = savgol_filter(
            g_rr[:, :, :, sl],
            window_length=w_len,
            polyorder=int(savgol_polyorder),
            deriv=2,
            delta=delta,
            axis=-1,
        )[:, :, :, rel_mid]
        return d2_L, d2_R

    return cache.get_or_compute(key, _compute)


def fpe_zero_bias_curvature_single_cutter(
    cache: RealizationFeatureCache,
    *,
    curvature_threshold: float = 20.0,
    lead: Literal["L", "R", "max_LR", "min_LR", "mean_LR"] = "R",
    cutter_idx: Optional[int] = None,
    curvature_source: Literal["thermal_lockin_3w", "savgol_thermal", "savgol_T0"] = "thermal_lockin_3w",
    scale_mode: Literal["physical_dynamic_grid", "legacy_uniform_97"] = "physical_dynamic_grid",
    savgol_window_length: int = 7,
    savgol_polyorder: int = 2,
) -> np.ndarray:
    r"""User's FPE $C_1(D, t)$: Zero-bias conductance curvature $\left.\frac{\partial^2 G_\alpha^{(c)}}{\partial V^2}\right|_{V=0} \le -t_{3\omega}$.

    Parameters
    ----------
    curvature_threshold : float, default 20.0
        Positive threshold $t_{3\omega}$ such that curvature $\le -t_{3\omega}$ passes.
    lead : {"L", "R", "max_LR", "min_LR", "mean_LR"}, default "R"
        - `"L"` or `"R"`: Single-lead curvature
        - `"max_LR"`: $\max(\kappa_L, \kappa_R) \le -t$ (both leads must be $\le -t$)
        - `"min_LR"`: $\min(\kappa_L, \kappa_R) \le -t$ (either lead $\le -t$)
        - `"mean_LR"`: $\frac{1}{2}(\kappa_L + \kappa_R) \le -t$.
    cutter_idx : int, optional
        If provided (`0 <= cutter_idx < N_cutters`), returns Rank-2 `(N_mu, N_Vz)` mask for
        that cutter. If `None` (default), returns Rank-3 `(N_cutters, N_mu, N_Vz)` tensor for
        downstream `dpe_k_of_n_cutter_consensus`.
    """
    d2_L, d2_R = extract_zero_bias_curvature_3d(
        cache,
        curvature_source=curvature_source,
        scale_mode=scale_mode,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
    )
    if lead == "L":
        obs = d2_L
    elif lead == "R":
        obs = d2_R
    elif lead == "max_LR":
        obs = np.maximum(d2_L, d2_R)
    elif lead == "min_LR":
        obs = np.minimum(d2_L, d2_R)
    elif lead == "mean_LR":
        obs = 0.5 * (d2_L + d2_R)
    else:
        raise ValueError(f"Unknown lead '{lead}' in fpe_zero_bias_curvature_single_cutter.")

    if cutter_idx is not None:
        obs = obs[int(cutter_idx)]
    return obs <= -float(curvature_threshold)


def fpe_cutter_averaged_curvature(
    cache: RealizationFeatureCache,
    *,
    curvature_threshold: float = 0.0,
    lead: Literal["L", "R", "mean_LR", "max_LR", "min_LR"] = "mean_LR",
    curvature_source: Literal["thermal_lockin_3w", "savgol_thermal", "savgol_T0"] = "thermal_lockin_3w",
    scale_mode: Literal["physical_dynamic_grid", "legacy_uniform_97"] = "physical_dynamic_grid",
    savgol_window_length: int = 7,
    savgol_polyorder: int = 2,
    cap_limit: Optional[float] = None,
) -> np.ndarray:
    r"""Cutter-averaged zero-bias curvature FPE $\langle 3\omega \rangle \le -t_{3\omega}$ (`2026-10-23` log).

    Computes the continuous mean across all $N_c = 5$ cutter pairs for Left and Right leads
    before thresholding. Returns Rank-2 `(N_mu, N_Vz)` boolean mask.
    """
    d2_L, d2_R = extract_zero_bias_curvature_3d(
        cache,
        curvature_source=curvature_source,
        scale_mode=scale_mode,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
    )
    if cap_limit is not None:
        lim = abs(float(cap_limit))
        d2_L = np.clip(d2_L, -lim, lim)
        d2_R = np.clip(d2_R, -lim, lim)

    L_3w = np.mean(d2_L, axis=0)
    R_3w = np.mean(d2_R, axis=0)

    if lead == "L":
        obs = L_3w
    elif lead == "R":
        obs = R_3w
    elif lead == "mean_LR":
        obs = 0.5 * (L_3w + R_3w)
    elif lead == "max_LR":
        obs = np.maximum(L_3w, R_3w)
    elif lead == "min_LR":
        obs = np.minimum(L_3w, R_3w)
    else:
        raise ValueError(f"Unknown lead '{lead}' in fpe_cutter_averaged_curvature.")

    return obs <= -float(curvature_threshold)


def fpe_nonlocal_2w_bounded(
    cache: RealizationFeatureCache,
    *,
    max_abs_2w: float = 1.0,
    lead: Literal["L", "R", "signed_mean", "magnitude_mean", "max_abs_LR", "min_abs_LR"] = "signed_mean",
    cutter_idx: Optional[int] = None,
) -> np.ndarray:
    r"""Nonlocal $2\omega$ lock-in antisymmetric derivative bound FPE (`2026-10-23` log).

    Evaluates whether the nonlocal zero-bias first derivative (`I_2w_RL` for Left, `I_2w_LR`
    for Right) lies within $[-t_{2\omega}, +t_{2\omega}]$.

    Supports both `"signed_mean"` ($|\frac{1}{2}(\langle 2\omega \rangle_L + \langle 2\omega \rangle_R)| \le t_{2\omega}$)
    and `"magnitude_mean"` ($\frac{1}{2}(|\langle 2\omega \rangle_L| + |\langle 2\omega \rangle_R|) \le t_{2\omega}$)
    per Round 1 Critique 6.
    """
    lockin = cache.get_stage1_broadened_and_lockin()
    if cutter_idx is not None:
        c = int(cutter_idx)
        L_2w = lockin["I_2w_RL"][c]
        R_2w = lockin["I_2w_LR"][c]
    else:
        L_2w = lockin["L_2w_mean"]
        R_2w = lockin["R_2w_mean"]

    if lead == "L":
        abs_val = np.abs(L_2w)
    elif lead == "R":
        abs_val = np.abs(R_2w)
    elif lead == "signed_mean":
        abs_val = np.abs(0.5 * (L_2w + R_2w))
    elif lead == "magnitude_mean":
        abs_val = 0.5 * (np.abs(L_2w) + np.abs(R_2w))
    elif lead == "max_abs_LR":
        abs_val = np.maximum(np.abs(L_2w), np.abs(R_2w))
    elif lead == "min_abs_LR":
        abs_val = np.minimum(np.abs(L_2w), np.abs(R_2w))
    else:
        raise ValueError(f"Unknown lead mode '{lead}' in fpe_nonlocal_2w_bounded.")

    return abs_val <= float(max_abs_2w)
