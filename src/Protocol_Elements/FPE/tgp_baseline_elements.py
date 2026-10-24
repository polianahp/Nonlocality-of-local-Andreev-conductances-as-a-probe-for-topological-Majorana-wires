"""Microsoft Azure Quantum TGP Stage 1 & Stage 2 Primitives as FPEs.

Exposes Microsoft's Topological Gap Protocol (`azure-quantum-tgp`) primitives inside the
same canonical `(N_cutters, N_mu, N_Vz)` and `(N_mu, N_Vz)` FPE/DPE algebra.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np

from ..cache import RealizationFeatureCache
from .lockin_curvature_2w_3w import fpe_zero_bias_curvature_single_cutter
from .transport_and_spectral_gap import extract_nonlocal_transport_gap_maps


def compute_tgp_canonical_dataset(
    cache: RealizationFeatureCache,
    *,
    derivative_threshold: float = 100.0,
    zbp_probability_threshold: float = 0.70,
    bias_window: float = 10e-3,
    polyorder: int = 2,
    min_gap_mev: float = 0.010,
    max_gap_mev: float = 0.067,
    gap_threshold_factor: float = 0.05,
    noise_threshold: float = 1e-4,
    min_cluster_size: int = 7,
    xi: float = 0.05,
    max_eps: float = 2.0,
    min_samples: int = 3,
    variance: int = 3,
    boundary_pct_threshold: float = 0.60,
    min_cutter_fraction: float = 0.50,
) -> Dict[str, Any]:
    r"""Execute canonical Microsoft `azure-quantum-tgp` Stage 1 & Stage 2 pipeline on cached data.

    Utilizes `RealizationFeatureCache.get_stage1_broadened_and_lockin()` to construct the
    xarray dataset and runs `tgp.two.extract_gap`, `tgp.two.zbp_dataset_derivative`,
    `tgp.two.set_gap_threshold`, and `tgp.two.cluster_and_score`.
    Memoized directly within `cache._memo`.
    """
    key = (
        "tgp_canonical_dataset",
        round(float(derivative_threshold), 4),
        round(float(zbp_probability_threshold), 4),
        round(float(bias_window), 6),
        int(polyorder),
        round(float(min_gap_mev), 6),
        round(float(max_gap_mev), 6),
        round(float(gap_threshold_factor), 6),
        round(float(noise_threshold), 8),
        int(min_cluster_size),
        round(float(xi), 4),
        round(float(max_eps), 4),
        int(min_samples),
        int(variance),
        round(float(boundary_pct_threshold), 4),
        round(float(min_cutter_fraction), 4),
        round(cache.temperature_mK, 4),
    )

    def _compute() -> Dict[str, Any]:
        import xarray as xr
        import tgp.two

        broad = cache.get_stage1_broadened_and_lockin()
        # Shape of broad arrays is (5, Nmu, Nvz, 97)
        # Transpose to (5, Nvz, Nmu, 97) so B=Vz and V=mu to match TGPAdapter coordinate convention
        g_ll = broad["g_ll"].transpose(0, 2, 1, 3)
        g_rr = broad["g_rr"].transpose(0, 2, 1, 3)
        g_lr = broad["g_lr"].transpose(0, 2, 1, 3)
        g_rl = broad["g_rl"].transpose(0, 2, 1, 3)

        ds = xr.Dataset(
            data_vars=dict(
                g_ll=(["cutter_pair_index", "B", "V", "bias"], g_ll),
                g_rr=(["cutter_pair_index", "B", "V", "bias"], g_rr),
                g_lr=(["cutter_pair_index", "B", "V", "bias"], g_lr),
                g_rl=(["cutter_pair_index", "B", "V", "bias"], g_rl),
            ),
            coords=dict(
                cutter_pair_index=np.arange(5),
                B=cache.vz_vals,
                V=cache.mu_vals,
                bias=cache.bias_grid,
            ),
        )

        t_l = ds.rename({"bias": "left_bias"})
        t_r = ds.rename({"bias": "right_bias"})
        t_l, t_r = tgp.two.extract_gap(
            t_l,
            t_r,
            gap_threshold_factor=gap_threshold_factor,
            noise_threshold=noise_threshold,
        )
        z_ds = tgp.two.zbp_dataset_derivative(
            t_l,
            t_r,
            derivative_threshold=derivative_threshold,
            zbp_probability_threshold=zbp_probability_threshold,
            bias_window=bias_window,
            polyorder=polyorder,
            average_over_cutter=False,
        )
        tgp.two.set_gap_threshold(
            z_ds,
            threshold_low=min_gap_mev,
            threshold_high=max_gap_mev,
        )
        s2 = tgp.two.cluster_and_score(
            z_ds,
            min_cluster_size=min_cluster_size,
            xi=xi,
            max_eps=max_eps,
            min_samples=min_samples,
            variance=variance,
            cluster_gap_threshold=min_gap_mev,
            cluster_percentage_boundary_threshold=boundary_pct_threshold,
            cluster_ncutter_threshold=min_cutter_fraction,
        )

        # Transpose back from (B, V) or (V, B) to canonical (Nmu, Nvz):
        zbp_3d = (
            z_ds["zbp"].transpose("cutter_pair_index", "V", "B").values.astype(bool)
        )
        gapped_zbp_3d = (
            z_ds["gapped_zbp"]
            .transpose("cutter_pair_index", "V", "B")
            .values.astype(bool)
        )

        roi2_vb = (
            s2.roi2.values.T if s2.roi2.dims == ("B", "V") else s2.roi2.values
        )
        roi2_2d = roi2_vb > 0

        # Gap maps (Nmu, Nvz)
        gap_arr = z_ds["gap"].transpose("cutter_pair_index", "V", "B").values
        gapless_2d = np.nanmin(gap_arr, axis=0) <= min_gap_mev

        return {
            "zbp_3d": zbp_3d,
            "gapped_zbp_3d": gapped_zbp_3d,
            "gap_arr_3d": gap_arr,
            "gapless_2d": gapless_2d,
            "roi2_2d": roi2_2d,
            "z_ds": z_ds,
            "s2": s2,
        }

    return cache.get_or_compute(key, _compute)


def fpe_tgp_zbp_per_cutter(
    cache: RealizationFeatureCache,
    *,
    zbp_curvature_threshold: float = 100.0,
    lead: Literal["L", "R", "max_LR"] = "max_LR",
    cutter_idx: Optional[int] = None,
    use_canonical_tgp: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """Microsoft TGP Stage 1/2 Zero-Bias Peak (ZBP) per-cutter FPE (`I_3w < -zbp_curvature_threshold`).

    When `use_canonical_tgp=True` (default), computes the exact `zbp` tensor from
    `compute_tgp_canonical_dataset` using Savitzky-Golay 2nd derivative ($t=100.0$).
    Returns Rank-3 `(N_cutters, N_mu, N_Vz)` (or Rank-2 `(N_mu, N_Vz)` if `cutter_idx` is given).
    """
    if use_canonical_tgp:
        canon = compute_tgp_canonical_dataset(
            cache,
            derivative_threshold=zbp_curvature_threshold,
            **{k: v for k, v in kwargs.items() if k in {"bias_window", "polyorder", "zbp_probability_threshold"}},
        )
        zbp_3d = canon["zbp_3d"]
        if cutter_idx is not None:
            return zbp_3d[int(cutter_idx)]
        return zbp_3d

    return fpe_zero_bias_curvature_single_cutter(
        cache,
        curvature_threshold=zbp_curvature_threshold,
        lead=lead,
        cutter_idx=cutter_idx,
        curvature_source="thermal_lockin_3w",
    )


def fpe_tgp_gapped_zbp_per_cutter(
    cache: RealizationFeatureCache,
    *,
    zbp_curvature_threshold: float = 100.0,
    min_gap_mev: float = 0.010,
    max_gap_mev: float = 0.067,
    gap_threshold_factor: float = 0.05,
    noise_threshold: float = 1e-4,
    use_thermal_broadened: bool = True,
    use_canonical_tgp: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """Microsoft TGP Stage 2 per-cutter `gapped_zbp` FPE (`zbp_L & zbp_R & (min_gap <= gap <= max_gap)`).

    Returns a Rank-3 `(N_cutters, N_mu, N_Vz)` boolean array indicating where each cutter pair
    simultaneously exhibits a dual-end ZBP and an open nonlocal transport gap.
    """
    if use_canonical_tgp:
        canon = compute_tgp_canonical_dataset(
            cache,
            derivative_threshold=zbp_curvature_threshold,
            min_gap_mev=min_gap_mev,
            max_gap_mev=max_gap_mev,
            gap_threshold_factor=gap_threshold_factor,
            noise_threshold=noise_threshold,
            **kwargs,
        )
        return canon["gapped_zbp_3d"]

    zbp_3d = fpe_tgp_zbp_per_cutter(
        cache,
        zbp_curvature_threshold=zbp_curvature_threshold,
        lead="max_LR",
        use_canonical_tgp=False,
    )
    gap_l_3d, gap_r_3d, _ = extract_nonlocal_transport_gap_maps(
        cache,
        gap_threshold_factor=gap_threshold_factor,
        noise_threshold=noise_threshold,
        use_thermal_broadened=use_thermal_broadened,
    )
    gapped_3d = (gap_l_3d >= float(min_gap_mev)) & (gap_r_3d >= float(min_gap_mev))
    if max_gap_mev is not None:
        gapped_3d = gapped_3d & (gap_l_3d <= float(max_gap_mev)) & (gap_r_3d <= float(max_gap_mev))
    return zbp_3d & gapped_3d
