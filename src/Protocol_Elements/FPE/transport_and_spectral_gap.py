r"""Spectral Excitation Gap, Nonlocal Conductance Transport Gap & Legacy Peak FPEs.

Explicitly separates (per Round 1 Critique 2):
1. `fpe_spectral_excitation_gap`: Closed-system BdG excitation gap $E_1$ (`topological_gap.npy`).
2. `fpe_nonlocal_conductance_transport_gap`: Open-system nonlocal transport gap extracted from
   antisymmetrized nonlocal conductances $A(G_{RL}(V)) = \frac{1}{2}(G_{RL}(V) - G_{RL}(-V))$
   and $A(G_{LR}(V))$.
3. `fpe_legacy_peak_gates`: All 5 zero-bias/finite-bias resonance peak gates from Protocol v2.0.
4. `fpe_quantum_dot_gamma_squared`: Side-coupled detuned quantum dot resonant conductance proxy
   for Dourado et al. $|\gamma^2| \ge \gamma_{\min}^2$ (`2026-10-22` log).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

import src.helpers as hp

from ..cache import RealizationFeatureCache


def fpe_spectral_excitation_gap(
    cache: RealizationFeatureCache,
    *,
    min_gap_mev: float = 0.015,
    max_gap_mev: Optional[float] = None,
) -> np.ndarray:
    r"""Closed-system spectral excitation gap FPE $E_1(\mu, V_z) \ge \Delta_{\min}$ (`topological_gap.npy`).

    Returns a Rank-2 boolean mask of shape `(N_mu, N_Vz)`.
    """
    gap = cache.topological_gap
    mask = gap >= float(min_gap_mev)
    if max_gap_mev is not None:
        mask = mask & (gap <= float(max_gap_mev))
    return mask


def extract_nonlocal_transport_gap_maps(
    cache: RealizationFeatureCache,
    *,
    gap_threshold_factor: float = 0.05,
    noise_threshold: float = 1e-4,
    median_size: int = 2,
    use_thermal_broadened: bool = True,
    global_field_threshold: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute memoized open-system nonlocal conductance transport gap maps `(gap_l_3d, gap_r_3d, gap_min_2d)`.

    Matches Microsoft `tgp.two.determine_gap` (`Aghaee 2023`, App. D.2, Eq. D2) and
    `helpers.extract_nonlocal_gap(..., global_threshold=True)`:
    1. Evaluates thermally broadened nonlocal conductances $G_{RL}(V, T), G_{LR}(V, T)$ by default
       (`use_thermal_broadened=True`, reducing to $T=0$ when `cache.temperature_mK == 0`).
    2. Applies optional 2D `(Vz, bias)` median filter (`median_size=2`) prior to antisymmetrization.
    3. Normalizes dynamic threshold over each `(Vz, bias)` field-bias slice (`global_field_threshold=True`,
       `axis=(-2, -1)`) rather than a single pixel so deeply gapped topological pixels do not
       collapse their reference amplitude.
    4. Applies the TGP half-bin midpoint correction `pos_bias[i] + 0.5 * (pos_bias[i+1] - pos_bias[i])`.
    """
    key = (
        "nonlocal_transport_gap_maps",
        round(float(gap_threshold_factor), 6),
        round(float(noise_threshold), 8),
        int(median_size),
        bool(use_thermal_broadened),
        bool(global_field_threshold),
        round(cache.temperature_mK, 4) if use_thermal_broadened else 0.0,
    )

    def _compute() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import scipy.ndimage

        if use_thermal_broadened:
            src = cache.get_stage1_broadened_and_lockin()
            g_rl, g_lr = src["g_rl"], src["g_lr"]
        else:
            src = cache.get_stage1_raw()
            g_rl = np.nan_to_num(src["g_rl"], nan=0.0)
            g_lr = np.nan_to_num(src["g_lr"], nan=0.0)

        bias = cache.bias_grid
        mid = int(np.argmin(np.abs(bias)))
        pos_bias = bias[mid:]  # shape (N_pos=49,), starting at 0.0 meV
        # Midpoint lookup matching tgp.two.extract_gap_from_trace & helpers.extract_nonlocal_gap
        pos_midpoints = pos_bias.copy()
        pos_midpoints[:-1] = pos_bias[:-1] + 0.5 * np.diff(pos_bias)

        def _extract_3d(g_4d: np.ndarray) -> np.ndarray:
            if int(median_size) > 1:
                # Filter along (B=Vz, bias) per (cutter, mu) slice matching tgp.two.determine_gap
                g_work = scipy.ndimage.median_filter(
                    g_4d, size=(1, 1, int(median_size), int(median_size))
                )
            else:
                g_work = g_4d

            # Antisymmetric component |A(G(V))| = 0.5 * |G(V) - G(-V)|
            anti = 0.5 * np.abs(g_work[:, :, :, mid:] - g_work[:, :, :, mid::-1])
            norm_axis = (-2, -1) if global_field_threshold else -1
            peak_anti = np.max(anti, axis=norm_axis, keepdims=True)
            thresh = np.maximum(float(gap_threshold_factor) * peak_anti, float(noise_threshold))
            above = anti > thresh
            above[:, :, :, 0] = False
            has_crossing = np.any(above, axis=-1)
            first_idx = np.argmax(above, axis=-1)
            gap_3d = np.where(has_crossing, pos_midpoints[first_idx], pos_bias[-1])
            return gap_3d

        gap_l_3d = _extract_3d(g_rl)
        gap_r_3d = _extract_3d(g_lr)
        gap_pair_min_3d = np.minimum(gap_l_3d, gap_r_3d)
        gap_min_2d = np.mean(gap_pair_min_3d, axis=0)
        return gap_l_3d, gap_r_3d, gap_min_2d

    return cache.get_or_compute(key, _compute)


def fpe_nonlocal_conductance_transport_gap(
    cache: RealizationFeatureCache,
    *,
    min_gap_mev: float = 0.010,
    max_gap_mev: Optional[float] = 0.067,
    gap_threshold_factor: float = 0.05,
    noise_threshold: float = 1e-4,
    median_size: int = 2,
    use_thermal_broadened: bool = True,
    global_field_threshold: bool = True,
    pooling: Literal["cutter_mean_min_LR", "per_cutter_min_LR", "k_of_n_min_LR", "L_mean", "R_mean"] = "cutter_mean_min_LR",
    k_min_cutters: int = 3,
) -> np.ndarray:
    r"""Open-system nonlocal conductance transport gap FPE $\Delta_{\text{trans}}^{\text{NL}} \in [\Delta_{\min}, \Delta_{\max}]$.

    Returns Rank-2 `(N_mu, N_Vz)` mask (or Rank-3 `(N_cutters, N_mu, N_Vz)` when
    `pooling == "per_cutter_min_LR"`).
    """
    gap_l_3d, gap_r_3d, gap_min_2d = extract_nonlocal_transport_gap_maps(
        cache,
        gap_threshold_factor=gap_threshold_factor,
        noise_threshold=noise_threshold,
        median_size=median_size,
        use_thermal_broadened=use_thermal_broadened,
        global_field_threshold=global_field_threshold,
    )
    if pooling == "k_of_n_min_LR":
        per_c = np.minimum(gap_l_3d, gap_r_3d)
        c_mask = per_c >= float(min_gap_mev)
        if max_gap_mev is not None:
            c_mask = c_mask & (per_c <= float(max_gap_mev))
        return np.sum(c_mask, axis=0) >= int(k_min_cutters)

    if pooling == "cutter_mean_min_LR":
        obs = gap_min_2d
    elif pooling == "per_cutter_min_LR":
        obs = np.minimum(gap_l_3d, gap_r_3d)
    elif pooling == "L_mean":
        obs = np.mean(gap_l_3d, axis=0)
    elif pooling == "R_mean":
        obs = np.mean(gap_r_3d, axis=0)
    else:
        raise ValueError(f"Unknown pooling '{pooling}' in fpe_nonlocal_conductance_transport_gap.")

    mask = obs >= float(min_gap_mev)
    if max_gap_mev is not None:
        mask = mask & (obs <= float(max_gap_mev))
    return mask


def fpe_legacy_peak_gates(
    cache: RealizationFeatureCache,
    *,
    require_pos_resonance: bool = True,
    height_thresh: float = 0.02,
    require_lr_energy_agreement: bool = True,
    peak_diff_tol: float = 0.010,
    require_neg_companion: bool = True,
    min_energy_mev: float = 0.0,
    max_energy_mev: float = 0.030,
    require_eh_symmetry: bool = True,
    sym_tol: float = 0.008,
    cutter_idx: int = 0,
    use_thermal_broadened: bool = True,
) -> np.ndarray:
    r"""Legacy Protocol v2.0 5-Gate Resonance Peak FPE (`Archive/reference_protocol.py`, Round 1 Critique 6).

    Evaluates the 5 local conductance peak gates on $G_{LL}(V)$ and $G_{RR}(V)$:
    1. Positive-bias peak existence & height (`peak_G >= height_thresh`) on both leads
    2. Left-Right positive peak energy agreement ($|E_{+,L} - E_{+,R}| \le \text{peak\_diff\_tol}$)
    3. Negative-bias companion peak existence
    4. Peak energy window ($E_{\pm} \in [\text{min\_energy\_mev}, \text{max\_energy\_mev}]$)
    5. Electron-hole bias voltage symmetry ($|E_+ + E_-| \le \text{sym\_tol}$).
    """
    c_idx = int(cutter_idx)
    key = (
        "legacy_peak_features",
        c_idx,
        bool(use_thermal_broadened),
        round(cache.temperature_mK, 4) if use_thermal_broadened else 0.0,
    )

    def _compute_peaks():
        if use_thermal_broadened:
            src = cache.get_stage1_broadened_and_lockin()
            g_ll = src["g_ll"][c_idx]
            g_rr = src["g_rr"][c_idx]
        else:
            src = cache.get_stage1_raw()
            g_ll = np.nan_to_num(src["g_ll"][c_idx], nan=0.0)
            g_rr = np.nan_to_num(src["g_rr"][c_idx], nan=0.0)

        bias = cache.bias_grid
        mid = len(bias) // 2

        def _find_pos_neg_peaks(g_3d: np.ndarray):
            # Local maximum in positive window [mid:, ...] and negative window [:mid+1, ...]
            g_pos = g_3d[:, :, mid:]
            b_pos = bias[mid:]
            idx_pos = np.argmax(g_pos, axis=-1)
            e_pos = b_pos[idx_pos]
            h_pos = np.max(g_pos, axis=-1)
            # Prominence check: peak cannot strictly be at the outer boundary index
            is_local_pos = idx_pos < (len(b_pos) - 1)

            g_neg = g_3d[:, :, : mid + 1]
            b_neg = bias[: mid + 1]
            idx_neg = np.argmax(g_neg, axis=-1)
            e_neg = b_neg[idx_neg]
            h_neg = np.max(g_neg, axis=-1)
            is_local_neg = idx_neg > 0
            return e_pos, h_pos, is_local_pos, e_neg, h_neg, is_local_neg

        return _find_pos_neg_peaks(g_ll), _find_pos_neg_peaks(g_rr)

    (ep_L, hp_L, locp_L, en_L, hn_L, locn_L), (
        ep_R,
        hp_R,
        locp_R,
        en_R,
        hn_R,
        locn_R,
    ) = cache.get_or_compute(key, _compute_peaks)

    mask = np.ones(cache.grid_shape, dtype=bool)
    if require_pos_resonance:
        mask &= locp_L & locp_R & (hp_L >= float(height_thresh)) & (hp_R >= float(height_thresh))
    if require_lr_energy_agreement:
        mask &= np.abs(ep_L - ep_R) <= float(peak_diff_tol)
    if require_neg_companion:
        mask &= locn_L & locn_R & (hn_L >= float(height_thresh)) & (hn_R >= float(height_thresh))
    mask &= (
        (ep_L >= float(min_energy_mev))
        & (ep_L <= float(max_energy_mev))
        & (ep_R >= float(min_energy_mev))
        & (ep_R <= float(max_energy_mev))
    )
    if require_eh_symmetry:
        mask &= (np.abs(ep_L + en_L) <= float(sym_tol)) & (
            np.abs(ep_R + en_R) <= float(sym_tol)
        )
    return mask


def fpe_quantum_dot_gamma_squared(
    cache: RealizationFeatureCache,
    *,
    resonant_g_target: float = 0.50,
    max_g_dev: float = 0.25,
    sym_asym_ratio: float = 0.0,
) -> np.ndarray:
    r"""Dourado et al. PRB 110, 014504 (2024) $\gamma^2 \approx 0.5$ Sweet-Spot / Detuned QD Analog FPE (`2026-10-22` log).

    Filters for phase-space regions where the symmetric-barrier zero-bias reflection
    conductance on both leads sits near the $e^2/(2h) = 0.5$ side-coupled quantum dot
    universal resonance value ($|G_{\alpha\alpha}^{\text{sym}} - G_{\text{target}}| \le \delta_G$).
    """
    bdata = cache.get_barrier_data()
    idx_sym = int(np.argmin(np.abs(bdata["asym_ratios"] - float(sym_asym_ratio))))
    g_ll = bdata["br_l"][:, :, idx_sym]
    g_rr = bdata["bl_r"][:, :, idx_sym]
    target = float(resonant_g_target)
    dev = float(max_g_dev)
    return (np.abs(g_ll - target) <= dev) & (np.abs(g_rr - target) <= dev)
