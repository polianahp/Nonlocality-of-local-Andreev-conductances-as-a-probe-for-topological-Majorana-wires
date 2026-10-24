r"""Spatial Neighborhood & Directional Morphological DPEs ($C_3$).

Derived Protocol Elements operating on 2D phase-space binary masks
$D \in \{0, 1\}^{N_\mu \times N_{V_z}}$ via local spatial neighborhoods or
directional morphological operations.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def dpe_radius_neighborhood_consensus(
    mask: np.ndarray,
    *,
    radius: float = 1.0,
    t_N: float = 0.50,
    radius_mu: Optional[float] = None,
    radius_vz: Optional[float] = None,
    metric: Literal[
        "chebyshev_grid",
        "manhattan_grid",
        "euclidean_grid",
        "euclidean_physical",
        "anisotropic_box",
    ] = "chebyshev_grid",
    include_center: bool = True,
    require_center_pass: bool = False,
    boundary_mode: Literal["valid_support_only", "zero_pad"] = "valid_support_only",
    mu_vals: Optional[np.ndarray] = None,
    vz_vals: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Spatial neighborhood consensus DPE $C_3(D, \mu, V_z, r, t_N) \in \{0, 1\}^{N_\mu \times N_{V_z}}$.

    For each point $(\mu_i, V_{z,j})$ on the phase-space grid, computes the local fraction
    of passing neighbors inside a ball/box of radius $r$ and returns `True` where that
    fraction is $\ge t_N$.

    Parameters
    ----------
    mask : np.ndarray
        Rank-2 boolean mask of shape `(N_mu, N_Vz)`.
    radius : float, default 1.0
        Neighborhood radius (in grid pixels for `*_grid` metrics, or in meV for
        `euclidean_physical`).
    t_N : float, default 0.50
        Minimum neighborhood pass fraction in `[0.0, 1.0]`.
    radius_mu, radius_vz : float, optional
        Half-widths along $\mu$ and $V_z$ when `metric == "anisotropic_box"`.
    metric : {"chebyshev_grid", "manhattan_grid", "euclidean_grid", "euclidean_physical", "anisotropic_box"}
        Distance metric defining the local neighborhood $B_r(i, j)$.
    include_center : bool, default True
        Whether the center pixel $(i, j)$ is included in its own neighborhood average.
    require_center_pass : bool, default False
        If `True`, computes $D \land \mathbb{I}[\text{density}_{B_r}(D) \ge t_N]$ (strictly
        erosive, matching `check_island_stability_v3`). If `False` (default), returns the pure
        unconditional neighborhood consensus $\mathbb{I}[\text{density}_{B_r}(D) \ge t_N]$ so
        arbitrary algebraic combinations such as $(C_1 \lor C_2) \land C_3(C_2)$ or
        $C_2 \land C_3(C_2)$ can be composed explicitly.
    boundary_mode : {"valid_support_only", "zero_pad"}, default "valid_support_only"
        - `"valid_support_only"`: Divides the passing neighbor count by the actual number of
          in-bounds grid points within $B_r(i, j)$ so boundary pixels are not artificially
          penalized.
        - `"zero_pad"`: Treats out-of-bounds points outside the grid as `0` (failed).
    mu_vals, vz_vals : np.ndarray, optional
        1D physical coordinate arrays of shape `(N_mu,)` and `(N_Vz,)`, required when
        `metric == "euclidean_physical"`.

    Returns
    -------
    np.ndarray
        Rank-2 boolean mask of shape `(N_mu, N_Vz)`.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"dpe_radius_neighborhood_consensus expects 2D mask, got shape {m.shape}.")

    if radius <= 0.0 and radius_mu is None and radius_vz is None:
        return m.copy()

    if metric == "euclidean_physical":
        if mu_vals is None or vz_vals is None:
            raise ValueError(
                "mu_vals and vz_vals must be provided when metric='euclidean_physical'."
            )
        mu_grid, vz_grid = np.meshgrid(mu_vals, vz_vals, indexing="ij")
        coords = np.column_stack([mu_grid.ravel(), vz_grid.ravel()])
        tree = cKDTree(coords)
        neighbors_list = tree.query_ball_point(coords, r=float(radius))
        flat_m = m.ravel().astype(np.float64)
        out_flat = np.zeros(flat_m.shape[0], dtype=bool)
        for idx, nbrs in enumerate(neighbors_list):
            if require_center_pass and not flat_m[idx]:
                continue
            if not include_center:
                nbrs = [n for n in nbrs if n != idx]
            if not nbrs:
                out_flat[idx] = bool(flat_m[idx])
                continue
            frac = float(np.sum(flat_m[nbrs])) / float(len(nbrs))
            out_flat[idx] = frac >= (float(t_N) - 1e-12)
        return out_flat.reshape(m.shape)

    # Grid-based kernel construction
    if metric == "anisotropic_box":
        r_mu = int(round(radius_mu if radius_mu is not None else radius))
        r_vz = int(round(radius_vz if radius_vz is not None else radius))
        footprint = np.ones((2 * r_mu + 1, 2 * r_vz + 1), dtype=np.float64)
        if not include_center:
            footprint[r_mu, r_vz] = 0.0
    else:
        r_int = max(int( np.ceil(radius) ), 1)
        di, dj = np.mgrid[-r_int : r_int + 1, -r_int : r_int + 1]
        if metric == "chebyshev_grid":
            dist = np.maximum(np.abs(di), np.abs(dj))
        elif metric == "manhattan_grid":
            dist = np.abs(di) + np.abs(dj)
        elif metric == "euclidean_grid":
            dist = np.hypot(di, dj)
        else:
            raise ValueError(f"Unknown metric '{metric}' in dpe_radius_neighborhood_consensus.")
        footprint = (dist <= float(radius) + 1e-12).astype(np.float64)
        if not include_center:
            footprint[r_int, r_int] = 0.0

    if np.sum(footprint) == 0.0:
        return m.copy()

    m_float = m.astype(np.float64)
    pass_sum = ndimage.convolve(m_float, footprint, mode="constant", cval=0.0)
    if boundary_mode == "valid_support_only":
        support = ndimage.convolve(np.ones_like(m_float), footprint, mode="constant", cval=0.0)
        support = np.maximum(support, 1.0)
    elif boundary_mode == "zero_pad":
        support = float(np.sum(footprint))
    else:
        raise ValueError(f"Unknown boundary_mode '{boundary_mode}'.")

    frac_map = pass_sum / support
    out = frac_map >= (float(t_N) - 1e-12)
    if require_center_pass:
        out = np.logical_and(m, out)
    return out


def dpe_1d_zeeman_closing(
    mask: np.ndarray,
    *,
    radius_vz: int = 1,
) -> np.ndarray:
    r"""1D horizontal morphological closing along the Zeeman axis $V_z$ (`axis=1`).

    Uses structuring element of shape `(1, 2 * radius_vz + 1)` to bridge nodal Majorana
    oscillation zero-crossings along $V_z$ without bleeding across adjacent $\mu$ subbands.
    Used in AC-RMS (Option 1) and ZMC-2 (Option 3).
    """
    m = np.asarray(mask, dtype=bool)
    if radius_vz <= 0 or not np.any(m):
        return m.copy()
    struct = np.ones((1, 2 * int(radius_vz) + 1), dtype=bool)
    return ndimage.binary_closing(m, structure=struct)


def dpe_hysteresis_propagation(
    high_mask: np.ndarray,
    low_mask: np.ndarray,
    *,
    connectivity: Literal["1d_vz", "4conn", "8conn"] = "1d_vz",
) -> np.ndarray:
    """Binary hysteresis propagation DPE mapping `(high_mask, low_mask) -> {0, 1}^{N_mu x N_Vz}`.

    Retains connected regions of `low_mask` that are seeded by at least one pixel in `high_mask`.
    """
    hi = np.asarray(high_mask, dtype=bool)
    lo = np.asarray(low_mask, dtype=bool)
    if not np.any(hi) or not np.any(lo):
        return np.zeros_like(lo, dtype=bool)
    if connectivity == "1d_vz":
        struct = np.ones((1, 3), dtype=bool)
    elif connectivity == "4conn":
        struct = ndimage.generate_binary_structure(2, 1)
    elif connectivity == "8conn":
        struct = np.ones((3, 3), dtype=bool)
    else:
        raise ValueError(f"Unknown connectivity '{connectivity}'.")
    return ndimage.binary_propagation(hi & lo, structure=struct, mask=lo)


def dpe_directional_manifold_closing(
    mask: np.ndarray,
    *,
    kernel_mu: int = 3,
    kernel_vz: int = 5,
) -> np.ndarray:
    """Legacy anisotropic 2D morphological closing (`kernel_mu x kernel_vz`).

    Included for completeness of the 2-week protocol variation catalog (`2026-10-21/22`).
    Note: Can add non-candidate bridge pixels between nearby specks; prefer subtractive DPEs
    for low-FDR operation.
    """
    m = np.asarray(mask, dtype=bool)
    if kernel_mu <= 1 and kernel_vz <= 1:
        return m.copy()
    struct = np.ones((int(kernel_mu), int(kernel_vz)), dtype=bool)
    return ndimage.binary_closing(m, structure=struct)
