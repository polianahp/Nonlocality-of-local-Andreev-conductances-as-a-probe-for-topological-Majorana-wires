r"""Barrier Sweep Reflection Correlation & Local $V_z$ Oscillation Correlation FPEs.

Unifies:
- Low-Asymmetry Window Reflection Correlation ($a \in [-5, +5]$, $\Delta U \in [-10, +10]\,\text{meV}$,
  from `2026-10-22` / `2026-10-23` logs)
- Legacy Full-Sweep Reflection Correlation ($a \in [0, 60]$, $\Delta U \in [0, 120]\,\text{meV}$)
- Dourado et al. PRB 110, 014504 (2024) Appendix G Figs. 20–21 1D sliding-window Pearson
  correlation along $V_z$ (`fpe_local_vz_oscillation_correlation`).
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from ..cache import RealizationFeatureCache


def extract_barrier_reflection_correlations(
    cache: RealizationFeatureCache,
    *,
    asym_min: float = -5.0,
    asym_max: float = 5.0,
    mean_subtract: bool = False,
    normalize_by_first_point: bool = False,
    norm_floor: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute memoized Left-sweep and Right-sweep barrier reflection correlations `(cL, cR)`.

    Resolves barrier indices dynamically from `cache.get_barrier_data()["asym_ratios"]`
    (Round 1 Critique 7) and guards zero-norm blockade regions with `norm_floor`.
    """
    key = (
        "barrier_reflection_corr_maps",
        round(float(asym_min), 4),
        round(float(asym_max), 4),
        bool(mean_subtract),
        bool(normalize_by_first_point),
        float(norm_floor),
    )

    def _compute() -> Tuple[np.ndarray, np.ndarray]:
        bdata = cache.get_barrier_data()
        asym = bdata["asym_ratios"]
        win_mask = (asym >= float(asym_min) - 1e-9) & (asym <= float(asym_max) + 1e-9)
        if not np.any(win_mask):
            # Fallback if positive-only sweep is queried with [-5, 5]: use [0, abs(asym_max)]
            win_mask = (asym >= 0.0) & (asym <= max(abs(float(asym_min)), abs(float(asym_max))) + 1e-9)
        if not np.any(win_mask):
            win_mask = np.ones_like(asym, dtype=bool)

        def _corr_pair(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
            x = np.nan_to_num(f1[:, :, win_mask], nan=0.0).copy()
            y = np.nan_to_num(f2[:, :, win_mask], nan=0.0).copy()
            # Note (Round 1 Critique 7): For uncentered cosine similarity, dividing by a positive
            # scalar cancels out identically; only apply first-point scaling when mean_subtract=True
            # and normalize_by_first_point=True.
            if normalize_by_first_point and mean_subtract and x.shape[-1] > 0:
                x0 = np.where(np.abs(x[:, :, :1]) > norm_floor, x[:, :, :1], 1.0)
                y0 = np.where(np.abs(y[:, :, :1]) > norm_floor, y[:, :, :1], 1.0)
                x = x / x0
                y = y / y0
            if mean_subtract:
                x = x - np.mean(x, axis=-1, keepdims=True)
                y = y - np.mean(y, axis=-1, keepdims=True)
            num = np.sum(x * y, axis=-1)
            nx = np.linalg.norm(x, axis=-1)
            ny = np.linalg.norm(y, axis=-1)
            denom = nx * ny
            valid = denom > float(norm_floor)
            corr = np.full(x.shape[:2], -1.0, dtype=np.float64)
            corr[valid] = np.clip(num[valid] / denom[valid], -1.0, 1.0)
            return corr

        cL = _corr_pair(bdata["bl_l"], bdata["bl_r"])
        cR = _corr_pair(bdata["br_l"], bdata["br_r"])
        return cL, cR

    return cache.get_or_compute(key, _compute)


def fpe_barrier_reflection_correlation(
    cache: RealizationFeatureCache,
    *,
    corr_threshold: float = 0.85,
    asym_min: float = -5.0,
    asym_max: float = 5.0,
    lead: Literal["L", "R", "min_LR", "max_LR", "mean_LR"] = "R",
    mean_subtract: bool = False,
    normalize_by_first_point: bool = False,
    norm_floor: float = 1e-12,
) -> np.ndarray:
    r"""Barrier reflection cosine/Pearson correlation FPE $C_\alpha([a_{\min}, a_{\max}]) \ge \theta_{\text{corr}}$.

    Covers both:
    - Low-Asymmetry Window Correlation: `asym_min=-5.0, asym_max=5.0` ($U \in [-10, +10]\,\text{meV}$)
    - Full-Sweep Legacy Correlation: `asym_min=0.0, asym_max=60.0` ($U \in [0, 120]\,\text{meV}$).

    Returns a Rank-2 boolean mask of shape `(N_mu, N_Vz)`.
    """
    cL, cR = extract_barrier_reflection_correlations(
        cache,
        asym_min=asym_min,
        asym_max=asym_max,
        mean_subtract=mean_subtract,
        normalize_by_first_point=normalize_by_first_point,
        norm_floor=norm_floor,
    )
    if lead == "L":
        obs = cL
    elif lead == "R":
        obs = cR
    elif lead == "min_LR":
        obs = np.minimum(cL, cR)
    elif lead == "max_LR":
        obs = np.maximum(cL, cR)
    elif lead == "mean_LR":
        obs = 0.5 * (cL + cR)
    else:
        raise ValueError(f"Unknown lead mode '{lead}' in fpe_barrier_reflection_correlation.")
    return obs >= float(corr_threshold)


def fpe_local_vz_oscillation_correlation(
    cache: RealizationFeatureCache,
    *,
    r_threshold: float = 0.50,
    window_pts: int = 5,
    observable: Literal["G_sym", "delta_G_asym", "acd_aL_aR"] = "G_sym",
    sym_asym_ratio: float = 0.0,
    pinch_asym_ratio: float = 60.0,
    variance_floor: float = 1e-10,
    use_cosine: bool = False,
) -> np.ndarray:
    r"""Dourado et al. PRB 110, 014504 (2024) Appendix G 1D sliding-window Pearson/Cosine correlation along $V_z$.

    Computes local sliding-window Pearson correlation (or uncentered cosine similarity if `use_cosine=True`)
    $r_{\text{local}}(\mu, V_z)$ along the Zeeman axis (`axis=1`) over a window of `window_pts` points
    ($3\,\text{pts} \approx 0.06\,\text{meV}$ or $5\,\text{pts} \approx 0.10\,\text{meV}$) between Left and Right
    observables, returning `r_local >= r_threshold`.
    """
    w = max(int(window_pts), 3)
    if w % 2 == 0:
        w += 1
    key = (
        "local_vz_osc_corr_map",
        w,
        observable,
        round(float(sym_asym_ratio), 4),
        round(float(pinch_asym_ratio), 4),
        float(variance_floor),
        bool(use_cosine),
    )

    def _compute() -> np.ndarray:
        bdata = cache.get_barrier_data()
        asym = bdata["asym_ratios"]
        idx_sym = int(np.argmin(np.abs(asym - float(sym_asym_ratio))))
        idx_pinch = int(np.argmin(np.abs(asym - float(pinch_asym_ratio))))

        if observable == "G_sym":
            x = bdata["br_l"][:, :, idx_sym]
            y = bdata["bl_r"][:, :, idx_sym]
        elif observable == "delta_G_asym":
            x = np.abs(bdata["br_l"][:, :, idx_sym] - bdata["br_l"][:, :, idx_pinch])
            y = np.abs(bdata["bl_r"][:, :, idx_sym] - bdata["bl_r"][:, :, idx_pinch])
        elif observable == "acd_aL_aR":
            from .barrier_asymmetry import extract_lead_acd_maps

            x, y, _ = extract_lead_acd_maps(
                cache,
                sym_asym_ratio=sym_asym_ratio,
                pinch_asym_ratio=pinch_asym_ratio,
            )
        else:
            raise ValueError(f"Unknown observable '{observable}' in fpe_local_vz_oscillation_correlation.")

        pad = w // 2
        x_pad = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
        y_pad = np.pad(y, ((0, 0), (pad, pad)), mode="edge")

        # Vectorized sliding window along Vz (axis=1)
        n_vz = x.shape[1]
        x_win = np.stack([x_pad[:, k : k + n_vz] for k in range(w)], axis=-1)
        y_win = np.stack([y_pad[:, k : k + n_vz] for k in range(w)], axis=-1)

        if use_cosine:
            cov = np.sum(x_win * y_win, axis=-1)
            vx = np.sum(x_win * x_win, axis=-1)
            vy = np.sum(y_win * y_win, axis=-1)
            denom = np.sqrt(vx * vy)
        else:
            dx = x_win - np.mean(x_win, axis=-1, keepdims=True)
            dy = y_win - np.mean(y_win, axis=-1, keepdims=True)
            cov = np.sum(dx * dy, axis=-1)
            vx = np.sum(dx * dx, axis=-1)
            vy = np.sum(dy * dy, axis=-1)
            denom = np.sqrt(vx * vy)
        valid = denom > float(variance_floor)
        r_map = np.full(x.shape, -1.0, dtype=np.float64)
        r_map[valid] = np.clip(cov[valid] / denom[valid], -1.0, 1.0)
        return r_map

    r_map = cache.get_or_compute(key, _compute)
    return r_map >= float(r_threshold)


def extract_cross_barrier_correlations(
    cache: RealizationFeatureCache,
    *,
    asym_min: float = -5.0,
    asym_max: float = 5.0,
    mean_subtract: bool = False,
    normalize_by_first_point: bool = False,
    norm_floor: float = 1e-12,
) -> np.ndarray:
    r"""Compute memoized cross-swept correlation between $G_{LL}(U_R)$ and $G_{RR}(U_L)$.

    Correlates `bdata["br_l"]` ($G_{LL}$ as $U_R$ is swept) with `bdata["bl_r"]`
    ($G_{RR}$ as $U_L$ is swept) across the asymmetry window `[asym_min, asym_max]`.
    """
    key = (
        "cross_barrier_corr_maps",
        round(float(asym_min), 4),
        round(float(asym_max), 4),
        bool(mean_subtract),
        bool(normalize_by_first_point),
        float(norm_floor),
    )

    def _compute() -> np.ndarray:
        bdata = cache.get_barrier_data()
        asym = bdata["asym_ratios"]
        win_mask = (asym >= float(asym_min) - 1e-9) & (asym <= float(asym_max) + 1e-9)
        if not np.any(win_mask):
            win_mask = (asym >= 0.0) & (
                asym <= max(abs(float(asym_min)), abs(float(asym_max))) + 1e-9
            )
        if not np.any(win_mask):
            win_mask = np.ones_like(asym, dtype=bool)

        x = np.nan_to_num(bdata["br_l"][:, :, win_mask], nan=0.0).copy()
        y = np.nan_to_num(bdata["bl_r"][:, :, win_mask], nan=0.0).copy()

        if normalize_by_first_point and mean_subtract and x.shape[-1] > 0:
            x0 = np.where(np.abs(x[:, :, :1]) > norm_floor, x[:, :, :1], 1.0)
            y0 = np.where(np.abs(y[:, :, :1]) > norm_floor, y[:, :, :1], 1.0)
            x = x / x0
            y = y / y0
        if mean_subtract:
            x = x - np.mean(x, axis=-1, keepdims=True)
            y = y - np.mean(y, axis=-1, keepdims=True)
        num = np.sum(x * y, axis=-1)
        nx = np.linalg.norm(x, axis=-1)
        ny = np.linalg.norm(y, axis=-1)
        denom = nx * ny
        valid = denom > float(norm_floor)
        corr = np.full(x.shape[:2], -1.0, dtype=np.float64)
        corr[valid] = np.clip(num[valid] / denom[valid], -1.0, 1.0)
        return corr

    return cache.get_or_compute(key, _compute)


def fpe_cross_barrier_conjugate_correlation(
    cache: RealizationFeatureCache,
    *,
    corr_threshold: float = 0.80,
    asym_min: float = -5.0,
    asym_max: float = 5.0,
    mean_subtract: bool = False,
    normalize_by_first_point: bool = False,
    norm_floor: float = 1e-12,
) -> np.ndarray:
    r"""Cross-swept conductance correlation FPE: $\operatorname{Corr}(G_{LL}(U_R), G_{RR}(U_L)) \ge \theta_{\text{corr}}$.

    Evaluates whether sweeping the Right barrier causes Left conductance $G_{LL}$ to drop in conjugate
    symmetry with Right conductance $G_{RR}$ when sweeping the Left barrier (Dourado et al. Fig. 13).
    Supports low asymmetry ($[-5, +5]$) and higher asymmetry ($[0, 60]$ or $[-60, 60]$).
    """
    corr = extract_cross_barrier_correlations(
        cache,
        asym_min=asym_min,
        asym_max=asym_max,
        mean_subtract=mean_subtract,
        normalize_by_first_point=normalize_by_first_point,
        norm_floor=norm_floor,
    )
    return corr >= float(corr_threshold)
