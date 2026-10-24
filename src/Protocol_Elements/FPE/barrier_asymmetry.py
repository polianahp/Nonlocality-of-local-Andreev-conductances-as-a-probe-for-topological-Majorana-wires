r"""Barrier Conductance Asymmetry & Symmetry Balance FPEs.

Fundamental Protocol Elements (FPEs) mapping continuous barrier sweep conductances
$G_{\alpha\beta}(U_L, U_R)$ and free parameters $\Theta$ to binary outputs $\{0, 1\}^{N_\mu \times N_{V_z}}$:
- `fpe_acd_geometric_mean` (User's $C_2$): $\sqrt{a_L a_R} \ge \tau$
- `fpe_single_lead_acd`: $a_\alpha \ge \tau_\alpha$
- `fpe_lead_symmetry_balance`: Dourado et al. PRB 110, 014504 (2024) Fig. 13 symmetry balance $S_{\text{sym}} \le s_{\max}$
- `fpe_pinch_off_conductance_gate`: Pinch-off suppression $G_{\alpha\alpha}^{\text{pinch}} < G_{\text{thresh}}$
- `fpe_barrier_monotonicity`: Monotonic barrier decay $\Delta_U G_{\alpha\alpha}(U) \le \epsilon_{\text{mono}}$
- `fpe_barrier_max_at_symmetric`: Conductance maximum at symmetric barrier $\Delta U = 0$.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from ..cache import RealizationFeatureCache


def _resolve_asym_index(asym_ratios: np.ndarray, target_ratio: float) -> int:
    """Resolve the index in `asym_ratios` closest to `target_ratio`."""
    return int(np.argmin(np.abs(asym_ratios - float(target_ratio))))


def extract_lead_acd_maps(
    cache: RealizationFeatureCache,
    *,
    floor_val: float = 1e-3,
    sym_asym_ratio: float = 0.0,
    pinch_asym_ratio: float = 60.0,
    use_signed_diff: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute memoized normalized Asymmetric Conductance Deviation maps `(a_L, a_R, sqrt_aL_aR)`.

    Following the S-ACD formulation (`2026-10-21/22`):
    - $a_L(\mu, V_z)$ measures the normalized change in Left conductance $G_{LL}$ when the
      distant Right barrier is pinched off (`br_l` at `sym_asym_ratio` vs `pinch_asym_ratio`).
    - $a_R(\mu, V_z)$ measures the normalized change in Right conductance $G_{RR}$ when the
      distant Left barrier is pinched off (`bl_r` at `sym_asym_ratio` vs `pinch_asym_ratio`).
    """
    key = (
        "lead_acd_maps",
        round(float(floor_val), 8),
        round(float(sym_asym_ratio), 4),
        round(float(pinch_asym_ratio), 4),
        bool(use_signed_diff),
    )

    def _compute() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bdata = cache.get_barrier_data()
        asym = bdata["asym_ratios"]
        idx_sym = _resolve_asym_index(asym, sym_asym_ratio)
        idx_pinch = _resolve_asym_index(asym, pinch_asym_ratio)

        # Left lead conductance G_LL under Right barrier sweep (br_l)
        g_ll_sym = bdata["br_l"][:, :, idx_sym]
        g_ll_pinch = bdata["br_l"][:, :, idx_pinch]

        # Right lead conductance G_RR under Left barrier sweep (bl_r)
        g_rr_sym = bdata["bl_r"][:, :, idx_sym]
        g_rr_pinch = bdata["bl_r"][:, :, idx_pinch]

        fl = max(float(floor_val), 1e-12)
        if use_signed_diff:
            diff_l = g_ll_sym - g_ll_pinch
            diff_r = g_rr_sym - g_rr_pinch
        else:
            diff_l = np.abs(g_ll_sym - g_ll_pinch)
            diff_r = np.abs(g_rr_sym - g_rr_pinch)

        a_L = np.clip(diff_l / np.maximum(g_ll_sym, fl), 0.0, 1.0) * (g_ll_sym >= fl)
        a_R = np.clip(diff_r / np.maximum(g_rr_sym, fl), 0.0, 1.0) * (g_rr_sym >= fl)
        sqrt_aL_aR = np.sqrt(a_L * a_R)
        return a_L, a_R, sqrt_aL_aR

    return cache.get_or_compute(key, _compute)


def fpe_acd_geometric_mean(
    cache: RealizationFeatureCache,
    *,
    tau: float = 0.05,
    floor_val: float = 1e-3,
    sym_asym_ratio: float = 0.0,
    pinch_asym_ratio: float = 60.0,
    use_signed_diff: bool = False,
) -> np.ndarray:
    r"""User's FPE $C_2(\alpha_L, \alpha_R, \tau)$: Geometric-mean ACD gate $\sqrt{a_L a_R} \ge \tau$.

    Returns a Rank-2 boolean mask of shape `(N_mu, N_Vz)`.
    """
    _, _, geom = extract_lead_acd_maps(
        cache,
        floor_val=floor_val,
        sym_asym_ratio=sym_asym_ratio,
        pinch_asym_ratio=pinch_asym_ratio,
        use_signed_diff=use_signed_diff,
    )
    return geom >= float(tau)


def fpe_single_lead_acd(
    cache: RealizationFeatureCache,
    *,
    lead: Literal["L", "R", "min_LR", "max_LR", "mean_LR"] = "L",
    tau: float = 0.05,
    floor_val: float = 1e-3,
    sym_asym_ratio: float = 0.0,
    pinch_asym_ratio: float = 60.0,
    use_signed_diff: bool = False,
) -> np.ndarray:
    r"""Single-lead or pooled Asymmetric Conductance Deviation FPE $a_\alpha \ge \tau$."""
    a_L, a_R, _ = extract_lead_acd_maps(
        cache,
        floor_val=floor_val,
        sym_asym_ratio=sym_asym_ratio,
        pinch_asym_ratio=pinch_asym_ratio,
        use_signed_diff=use_signed_diff,
    )
    if lead == "L":
        obs = a_L
    elif lead == "R":
        obs = a_R
    elif lead == "min_LR":
        obs = np.minimum(a_L, a_R)
    elif lead == "max_LR":
        obs = np.maximum(a_L, a_R)
    elif lead == "mean_LR":
        obs = 0.5 * (a_L + a_R)
    else:
        raise ValueError(f"Unknown lead mode '{lead}' in fpe_single_lead_acd.")
    return obs >= float(tau)


def fpe_lead_symmetry_balance(
    cache: RealizationFeatureCache,
    *,
    s_max: float = 0.20,
    eps_sym: float = 1e-4,
    sym_asym_ratio: float = 0.0,
) -> np.ndarray:
    r"""Dourado et al. PRB 110, 014504 (2024) Fig. 13 Conjugate Lead Symmetry Balance FPE.

    Evaluates:
        $$S_{\text{sym}}(\mu, V_z) = \frac{|G_{LL}^{\text{sym}} - G_{RR}^{\text{sym}}|}{\max(G_{LL}^{\text{sym}} + G_{RR}^{\text{sym}}, \epsilon_{\text{sym}})} \le s_{\max}$$
    """
    key = ("lead_symmetry_balance_obs", round(float(eps_sym), 8), round(float(sym_asym_ratio), 4))

    def _compute() -> np.ndarray:
        bdata = cache.get_barrier_data()
        idx_sym = _resolve_asym_index(bdata["asym_ratios"], sym_asym_ratio)
        g_ll = bdata["br_l"][:, :, idx_sym]
        g_rr = bdata["bl_r"][:, :, idx_sym]
        denom = np.maximum(g_ll + g_rr, max(float(eps_sym), 1e-12))
        s_sym = np.abs(g_ll - g_rr) / denom
        # Require non-zero total conductance above eps_sym
        s_sym = np.where((g_ll + g_rr) >= float(eps_sym), s_sym, np.inf)
        return s_sym

    s_sym = cache.get_or_compute(key, _compute)
    return s_sym <= float(s_max)


def fpe_pinch_off_conductance_gate(
    cache: RealizationFeatureCache,
    *,
    lead: Literal["L", "R", "both", "either"] = "both",
    g_thresh: float = 0.05,
    pinch_asym_ratio: float = 60.0,
) -> np.ndarray:
    r"""Local pinch-off suppression gate $G_{\alpha\alpha}^{\text{pinch}} < G_{\text{thresh}}$ (LQM-P Option 2).

    Verifies that raising barrier $\alpha$ to `pinch_asym_ratio` suppresses local reflection
    conductance $G_{\alpha\alpha}$ below `g_thresh`.
    """
    bdata = cache.get_barrier_data()
    idx_pinch = _resolve_asym_index(bdata["asym_ratios"], pinch_asym_ratio)
    g_ll_pinch = bdata["bl_l"][:, :, idx_pinch]
    g_rr_pinch = bdata["br_r"][:, :, idx_pinch]

    th = float(g_thresh)
    if lead == "L":
        return g_ll_pinch < th
    if lead == "R":
        return g_rr_pinch < th
    if lead == "both":
        return (g_ll_pinch < th) & (g_rr_pinch < th)
    if lead == "either":
        return (g_ll_pinch < th) | (g_rr_pinch < th)
    raise ValueError(f"Unknown lead mode '{lead}' in fpe_pinch_off_conductance_gate.")


def fpe_barrier_monotonicity(
    cache: RealizationFeatureCache,
    *,
    lead: Literal["L", "R"] = "L",
    asym_min: float = 0.0,
    asym_max: float = 60.0,
    mono_tol: float = 1e-4,
) -> np.ndarray:
    """Legacy FPE (`Archive/reference_protocol.py:check_arr_monotonic`): monotonic decay along barrier sweep."""
    bdata = cache.get_barrier_data()
    asym = bdata["asym_ratios"]
    win = (asym >= float(asym_min) - 1e-9) & (asym <= float(asym_max) + 1e-9)
    arr = bdata["bl_l" if lead == "L" else "br_r"][:, :, win]
    if arr.shape[-1] <= 1:
        return np.ones(cache.grid_shape, dtype=bool)
    diffs = np.diff(arr, axis=-1)
    return np.all(diffs <= float(mono_tol), axis=-1)


def fpe_barrier_max_at_symmetric(
    cache: RealizationFeatureCache,
    *,
    lead: Literal["L", "R"] = "L",
    sym_asym_ratio: float = 0.0,
    asym_min: float = 0.0,
    asym_max: float = 60.0,
) -> np.ndarray:
    r"""Legacy FPE (`Archive/reference_protocol.py:check_max_at_symm`): maximum conductance at $\Delta U = 0$."""
    bdata = cache.get_barrier_data()
    asym = bdata["asym_ratios"]
    win_idx = np.flatnonzero(
        (asym >= float(asym_min) - 1e-9) & (asym <= float(asym_max) + 1e-9)
    )
    if win_idx.size == 0:
        return np.zeros(cache.grid_shape, dtype=bool)
    target_rel_idx = int(np.argmin(np.abs(asym[win_idx] - float(sym_asym_ratio))))
    arr = bdata["bl_l" if lead == "L" else "br_r"][:, :, win_idx]
    max_idx = np.argmax(arr, axis=-1)
    return max_idx == target_rel_idx
