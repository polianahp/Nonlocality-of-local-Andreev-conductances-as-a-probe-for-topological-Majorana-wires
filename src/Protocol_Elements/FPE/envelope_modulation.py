"""1D Zeeman Foliation Energy & Analytic Quadrature Envelope FPEs (`2026-10-21`).

Implements the Fundamental Protocol Elements from the 3 Consciousness Council architectures:
- Option 1 (`AC-RMS`): `fpe_ac_rms_energy`, `fpe_ac_rms_inflection`
- Option 2 (`LQM-P`): `fpe_lqmp_detrended_power`, `fpe_lqmp_quadrature_coherence`
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter

from ..cache import RealizationFeatureCache
from .barrier_asymmetry import extract_lead_acd_maps


def extract_ac_rms_fields(
    cache: RealizationFeatureCache,
    *,
    window_pts: int = 5,
    floor_val: float = 1e-3,
    sym_asym_ratio: float = 0.0,
    pinch_asym_ratio: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute memoized Option 1 AC-RMS coincident envelope `C_asym` and its $V_z$ 2nd derivative `d2_C`.

    $$\mathcal{E}_\alpha(\mu, V_z) = \sqrt{\frac{1}{W_{V_z}} \sum_{k} a_\alpha(\mu, V_z + k\Delta V_z)^2}, \quad
      \mathcal{C}_{\text{asym}} = \sqrt{\mathcal{E}_L \mathcal{E}_R}$$
    """
    w = max(int(window_pts), 3)
    if w % 2 == 0:
        w += 1
    key = (
        "ac_rms_fields",
        w,
        round(float(floor_val), 8),
        round(float(sym_asym_ratio), 4),
        round(float(pinch_asym_ratio), 4),
    )

    def _compute() -> Tuple[np.ndarray, np.ndarray]:
        a_L, a_R, _ = extract_lead_acd_maps(
            cache,
            floor_val=floor_val,
            sym_asym_ratio=sym_asym_ratio,
            pinch_asym_ratio=pinch_asym_ratio,
            use_signed_diff=False,
        )
        e_L = np.sqrt(
            np.maximum(ndimage.uniform_filter1d(a_L**2, size=w, axis=1, mode="nearest"), 0.0)
        )
        e_R = np.sqrt(
            np.maximum(ndimage.uniform_filter1d(a_R**2, size=w, axis=1, mode="nearest"), 0.0)
        )
        c_asym = np.sqrt(e_L * e_R)
        d2_c = ndimage.laplace(ndimage.gaussian_filter1d(c_asym, sigma=1.0, axis=1))
        return c_asym, d2_c

    return cache.get_or_compute(key, _compute)


def fpe_ac_rms_energy(
    cache: RealizationFeatureCache,
    *,
    window_pts: int = 5,
    use_adaptive_percentile: bool = True,
    percentile_rank: float = 95.0,
    threshold_factor: float = 0.35,
    static_threshold: float = 0.20,
    multiplier: float = 1.0,
    floor_val: float = 1e-3,
    sym_asym_ratio: float = 0.0,
    pinch_asym_ratio: float = 60.0,
) -> np.ndarray:
    r"""Option 1 (`AC-RMS`) Coincident RMS Energy FPE $\mathcal{C}_{\text{asym}} \ge m \cdot \theta(T)$.

    Supports both the base threshold (`multiplier=1.0`) and the high-plateau bypass threshold
    (`multiplier=1.30`, Round 1 Critique 3) so that Option 1 is algebraically:
        $$\mathcal{D}_{\text{close}, V_z=1}\!\Big(\mathcal{F}_{\text{AC-RMS}}(m=1.30) \lor \big(\mathcal{F}_{\text{AC-RMS}}(m=1.0) \land \mathcal{F}_{\text{inflect}}(0)\big)\Big)$$
    """
    c_asym, _ = extract_ac_rms_fields(
        cache,
        window_pts=window_pts,
        floor_val=floor_val,
        sym_asym_ratio=sym_asym_ratio,
        pinch_asym_ratio=pinch_asym_ratio,
    )
    if use_adaptive_percentile:
        pos = c_asym[c_asym > 1e-4]
        p_val = float(np.percentile(pos, float(percentile_rank))) if pos.size > 10 else 0.0
        base_theta = max(float(threshold_factor) * p_val, 0.03)
    else:
        base_theta = float(static_threshold)
    eff_theta = float(multiplier) * base_theta
    return c_asym >= eff_theta


def fpe_ac_rms_inflection(
    cache: RealizationFeatureCache,
    *,
    window_pts: int = 5,
    floor_val: float = 1e-3,
    max_d2: float = 0.0,
    sym_asym_ratio: float = 0.0,
    pinch_asym_ratio: float = 60.0,
) -> np.ndarray:
    r"""Option 1 (`AC-RMS`) 2nd-derivative inflection sharpening FPE $\partial^2 \mathcal{C}_{\text{asym}} / \partial V_z^2 \le \text{max\_d2}$."""
    _, d2_c = extract_ac_rms_fields(
        cache,
        window_pts=window_pts,
        floor_val=floor_val,
        sym_asym_ratio=sym_asym_ratio,
        pinch_asym_ratio=pinch_asym_ratio,
    )
    return d2_c <= float(max_d2)


def extract_lqmp_fields(
    cache: RealizationFeatureCache,
    *,
    win_vz: int = 5,
    expected_lambda_mev: float = 0.18,
    sym_asym_ratio: float = 0.0,
    pinch_asym_ratio: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute memoized Option 2 (`LQM-P`) detrended oscillatory power `(P_L, P_R)` and quadrature coherence `C_LR_sq`.

    Uses Savitzky-Golay 1st derivatives scaled by $\lambda_{V_z}/(2\pi)$ to construct real-space
    analytic quadrature envelopes that are 100% invariant to disorder-induced phase shifts $\Delta\Phi$.
    """
    w = max(int(win_vz), 5)
    if w % 2 == 0:
        w += 1
    key = (
        "lqmp_fields",
        w,
        round(float(expected_lambda_mev), 5),
        round(float(sym_asym_ratio), 4),
        round(float(pinch_asym_ratio), 4),
    )

    def _compute() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bdata = cache.get_barrier_data()
        asym = bdata["asym_ratios"]
        idx_sym = int(np.argmin(np.abs(asym - float(sym_asym_ratio))))
        idx_pinch = int(np.argmin(np.abs(asym - float(pinch_asym_ratio))))

        # Signed asymmetric conductance deviation delta_G_L (br_l) and delta_G_R (bl_r)
        dg_L = bdata["br_l"][:, :, idx_sym] - bdata["br_l"][:, :, idx_pinch]
        dg_R = bdata["bl_r"][:, :, idx_sym] - bdata["bl_r"][:, :, idx_pinch]

        mean_L = ndimage.uniform_filter1d(dg_L, size=w, axis=1, mode="nearest")
        mean_R = ndimage.uniform_filter1d(dg_R, size=w, axis=1, mode="nearest")
        tilde_L = dg_L - mean_L
        tilde_R = dg_R - mean_R

        P_L = ndimage.uniform_filter1d(tilde_L**2, size=w, axis=1, mode="nearest")
        P_R = ndimage.uniform_filter1d(tilde_R**2, size=w, axis=1, mode="nearest")

        q_scale = float(expected_lambda_mev) / (2.0 * np.pi)
        dvz = max(cache.d_vz_mev, 1e-6)
        Q_L = q_scale * savgol_filter(
            tilde_L, window_length=w, polyorder=2, deriv=1, delta=dvz, axis=1, mode="nearest"
        )
        Q_R = q_scale * savgol_filter(
            tilde_R, window_length=w, polyorder=2, deriv=1, delta=dvz, axis=1, mode="nearest"
        )

        I_re = ndimage.uniform_filter1d(
            tilde_L * tilde_R + Q_L * Q_R, size=w, axis=1, mode="nearest"
        )
        I_im = ndimage.uniform_filter1d(
            Q_L * tilde_R - tilde_L * Q_R, size=w, axis=1, mode="nearest"
        )
        E_L = ndimage.uniform_filter1d(tilde_L**2 + Q_L**2, size=w, axis=1, mode="nearest")
        E_R = ndimage.uniform_filter1d(tilde_R**2 + Q_R**2, size=w, axis=1, mode="nearest")

        C_LR_sq = (I_re**2 + I_im**2) / (E_L * E_R + 1e-12)
        return P_L, P_R, np.clip(C_LR_sq, 0.0, 1.0)

    return cache.get_or_compute(key, _compute)


def fpe_lqmp_detrended_power(
    cache: RealizationFeatureCache,
    *,
    lead: Literal["L", "R", "min_LR", "max_LR", "mean_LR"] = "min_LR",
    theta_P: float = 0.002,
    win_vz: int = 5,
    expected_lambda_mev: float = 0.18,
    sym_asym_ratio: float = 0.0,
    pinch_asym_ratio: float = 60.0,
) -> np.ndarray:
    r"""Option 2 (`LQM-P`) detrended oscillatory power FPE $P_\alpha(V_z, \mu) \ge \theta_P$."""
    P_L, P_R, _ = extract_lqmp_fields(
        cache,
        win_vz=win_vz,
        expected_lambda_mev=expected_lambda_mev,
        sym_asym_ratio=sym_asym_ratio,
        pinch_asym_ratio=pinch_asym_ratio,
    )
    if lead == "L":
        obs = P_L
    elif lead == "R":
        obs = P_R
    elif lead == "min_LR":
        obs = np.minimum(P_L, P_R)
    elif lead == "max_LR":
        obs = np.maximum(P_L, P_R)
    elif lead == "mean_LR":
        obs = 0.5 * (P_L + P_R)
    else:
        raise ValueError(f"Unknown lead mode '{lead}' in fpe_lqmp_detrended_power.")
    return obs >= float(theta_P)


def fpe_lqmp_quadrature_coherence(
    cache: RealizationFeatureCache,
    *,
    theta_C: float = 0.25,
    win_vz: int = 5,
    expected_lambda_mev: float = 0.18,
    sym_asym_ratio: float = 0.0,
    pinch_asym_ratio: float = 60.0,
) -> np.ndarray:
    r"""Option 2 (`LQM-P`) quadrature cross-envelope coherence FPE $\mathcal{C}_{LR}^2(V_z, \mu) \ge \theta_C$."""
    _, _, C_LR_sq = extract_lqmp_fields(
        cache,
        win_vz=win_vz,
        expected_lambda_mev=expected_lambda_mev,
        sym_asym_ratio=sym_asym_ratio,
        pinch_asym_ratio=pinch_asym_ratio,
    )
    return C_LR_sq >= float(theta_C)
