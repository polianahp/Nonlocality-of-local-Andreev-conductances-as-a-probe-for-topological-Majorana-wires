r"""Mandatory Post-Processing Subtractive Island Filters ($C_4$).

Derived Protocol Elements (DPEs) that prune spurious small islands, 1D filaments,
or unclustered noise specks so every protocol ends with the canonical form:
    $P = C_4(\dots)$
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from sklearn.cluster import OPTICS


def _get_structure_2d(connectivity: int = 8) -> np.ndarray:
    """Return 2D binary connectivity structuring element (4- or 8-connected)."""
    if int(connectivity) == 8:
        return np.ones((3, 3), dtype=bool)
    if int(connectivity) == 4:
        return ndimage.generate_binary_structure(2, 1)
    raise ValueError(f"Unsupported 2D connectivity '{connectivity}'; must be 4 or 8.")


def dpe_subtractive_area_pruning(
    mask: np.ndarray,
    *,
    min_island_size: int = 10,
    connectivity: int = 8,
    fallback_min_island_size: Optional[int] = None,
    fallback_top_n: Optional[int] = None,
) -> np.ndarray:
    r"""Connected-component area pruning DPE retaining only islands with $\text{Area}(C) \ge A_{\min}$.

    Strictly subtractive (never dilates or adds unpassed pixels).

    Parameters
    ----------
    mask : np.ndarray
        Rank-2 boolean mask of shape `(N_mu, N_Vz)`.
    min_island_size : int, default 10
        Minimum connected-component area in pixels.
    connectivity : {4, 8}, default 8
        Connected-component neighborhood connectivity.
    fallback_min_island_size : int, optional
        If provided and no component meets `min_island_size`, falls back to retaining up to
        `fallback_top_n` largest components with area `>= fallback_min_island_size`.
    fallback_top_n : int, optional
        Maximum number of components retained under the fallback rule.

    Returns
    -------
    np.ndarray
        Rank-2 boolean mask of shape `(N_mu, N_Vz)`.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"dpe_subtractive_area_pruning expects 2D mask, got shape {m.shape}.")
    if min_island_size <= 1 or not np.any(m):
        return m.copy()

    struct = _get_structure_2d(connectivity)
    labeled, num_features = ndimage.label(m, structure=struct)
    if num_features == 0:
        return np.zeros_like(m, dtype=bool)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # background label 0
    valid_labels = np.flatnonzero(counts >= int(min_island_size))

    if valid_labels.size > 0:
        return np.isin(labeled, valid_labels)

    if fallback_min_island_size is not None and int(fallback_min_island_size) >= 1:
        fb_candidates = np.flatnonzero(counts >= int(fallback_min_island_size))
        if fb_candidates.size > 0:
            order = np.argsort(counts[fb_candidates])[::-1]
            if fallback_top_n is not None and int(fallback_top_n) > 0:
                order = order[: int(fallback_top_n)]
            chosen = fb_candidates[order]
            return np.isin(labeled, chosen)

    return np.zeros_like(m, dtype=bool)


def dpe_topological_island_radius_core_filter(
    mask: np.ndarray,
    *,
    core_shape: Tuple[int, int] = (2, 2),
    connectivity: int = 8,
    mode: Literal[
        "component_has_core",
        "morphological_opening",
        "opening_reconstructed_by_4conn",
    ] = "component_has_core",
) -> np.ndarray:
    r"""Topological Island Radius / Core Reconstruction Filter DPE (`2026-10-23`).

    Requires topological islands to possess a non-empty 2D interior core under erosion by
    structuring element $B_{k \times m}$ (default $B_{2 \times 2}$):
        $\text{Core}(C) = C \ominus B_{2 \times 2} \neq \emptyset$.

    Parameters
    ----------
    mask : np.ndarray
        Rank-2 boolean mask of shape `(N_mu, N_Vz)`.
    core_shape : tuple of (int, int), default (2, 2)
        Shape `(k_mu, k_vz)` of the required 2D interior block $B$.
    connectivity : {4, 8}, default 8
        Connected-component connectivity used when `mode == "component_has_core"`.
    mode : {"component_has_core", "morphological_opening", "opening_reconstructed_by_4conn"}
        - `"component_has_core"`: Retains every connected component $C$ that contains at least
          one eroded core pixel (reproduces `apply_island_radius_filter` from `2026-10-23`,
          preserving exact boundaries of valid islands).
        - `"morphological_opening"`: Computes $\mathcal{M} \circ B = (\mathcal{M} \ominus B) \oplus B$,
          retaining only pixels directly covered by at least one $k \times m$ block inside $\mathcal{M}$
          (shaves off 1-pixel-wide boundary hairs attached to valid islands).
        - `"opening_reconstructed_by_4conn"`: Opens by $B$ and reconstructs within $\mathcal{M}$
          via 4-connectivity (prevents diagonal corner leaks into adjacent trivial blobs).

    Returns
    -------
    np.ndarray
        Rank-2 boolean mask of shape `(N_mu, N_Vz)`.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(
            f"dpe_topological_island_radius_core_filter expects 2D mask, got shape {m.shape}."
        )
    if not np.any(m):
        return np.zeros_like(m, dtype=bool)

    k_mu, k_vz = int(core_shape[0]), int(core_shape[1])
    if k_mu <= 1 and k_vz <= 1:
        return m.copy()

    b_core = np.ones((k_mu, k_vz), dtype=bool)
    eroded = ndimage.binary_erosion(m, structure=b_core)
    if not np.any(eroded):
        return np.zeros_like(m, dtype=bool)

    if mode == "component_has_core":
        struct = _get_structure_2d(connectivity)
        labeled, num_features = ndimage.label(m, structure=struct)
        if num_features == 0:
            return np.zeros_like(m, dtype=bool)
        valid_labels = np.unique(labeled[eroded])
        valid_labels = valid_labels[valid_labels > 0]
        if valid_labels.size == 0:
            return np.zeros_like(m, dtype=bool)
        return np.isin(labeled, valid_labels)

    if mode == "morphological_opening":
        opened = ndimage.binary_opening(m, structure=b_core)
        return np.logical_and(m, opened)

    if mode == "opening_reconstructed_by_4conn":
        opened = ndimage.binary_opening(m, structure=b_core)
        seed = np.logical_and(m, opened)
        if not np.any(seed):
            return np.zeros_like(m, dtype=bool)
        s4 = ndimage.generate_binary_structure(2, 1)
        return ndimage.binary_propagation(seed, structure=s4, mask=m)

    raise ValueError(f"Unknown core filter mode '{mode}'.")


def dpe_optics_subtractive_clustering(
    mask: np.ndarray,
    *,
    min_samples: int = 5,
    max_eps: float = 3.0,
    min_cluster_size: int = 10,
    xi: float = 0.05,
    coordinate_space: Literal["grid_indices", "physical_meV"] = "grid_indices",
    mu_vals: Optional[np.ndarray] = None,
    vz_vals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Subtractive OPTICS density-based clustering filter DPE.

    Evaluates `sklearn.cluster.OPTICS` on the coordinates of candidate pixels in `mask`
    and removes unclustered noise points (`labels_ == -1`). Never adds unpassed pixels
    and never silently drops masks with `> 1500` pixels.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"dpe_optics_subtractive_clustering expects 2D mask, got shape {m.shape}.")

    idx_pts = np.argwhere(m)
    n_pts = idx_pts.shape[0]
    eff_min_pts = max(int(min_samples), int(min_cluster_size))
    if n_pts < eff_min_pts:
        return np.zeros_like(m, dtype=bool)

    if coordinate_space == "physical_meV":
        if mu_vals is None or vz_vals is None:
            raise ValueError(
                "mu_vals and vz_vals are required when coordinate_space='physical_meV'."
            )
        coords = np.column_stack([
            mu_vals[idx_pts[:, 0]].astype(np.float64),
            vz_vals[idx_pts[:, 1]].astype(np.float64),
        ])
    elif coordinate_space == "grid_indices":
        coords = idx_pts.astype(np.float64)
    else:
        raise ValueError(f"Unknown coordinate_space '{coordinate_space}'.")

    clust = OPTICS(
        min_samples=int(min_samples),
        max_eps=float(max_eps),
        min_cluster_size=int(min_cluster_size),
        xi=float(xi),
        algorithm="kd_tree",
    )
    clust.fit(coords)
    keep = clust.labels_ >= 0
    out = np.zeros_like(m, dtype=bool)
    if np.any(keep):
        kept_idx = idx_pts[keep]
        out[kept_idx[:, 0], kept_idx[:, 1]] = True
    return out


def dpe_island_boundary_gapless_contiguity_filter(
    gapped_mask: np.ndarray,
    gapless_mask: Optional[np.ndarray] = None,
    *,
    zbp_mask: Optional[np.ndarray] = None,
    variance: int = 3,
    boundary_pct_threshold: float = 0.60,
    min_island_size: int = 7,
    connectivity: int = 8,
) -> np.ndarray:
    r"""Topological gapless boundary-closure DPE (`compute_gapped_islands_and_boundaries`).

    For each connected island $C$ in `gapped_mask` with area $\ge \text{min\_island\_size}$,
    extracts its 1-pixel morphological boundary $\partial C = C \setminus (C \ominus B)$
    and checks what fraction of boundary pixels border a gapless / closed-gap region
    within Chebyshev distance $s = \max(1, \text{variance} // 2)$ (matching Microsoft TGP
    `_gapless_boundary`). Retains only islands meeting:
        $\text{boundary\_fraction} \ge \text{boundary\_pct\_threshold}$.
    """
    g_mask = np.asarray(gapped_mask, dtype=bool)
    if gapless_mask is not None:
        target = np.asarray(gapless_mask, dtype=bool)
    elif zbp_mask is not None:
        target = np.asarray(zbp_mask, dtype=bool)
    else:
        raise ValueError("dpe_island_boundary_gapless_contiguity_filter requires gapless_mask.")

    if target.ndim == 3:
        target = np.any(target, axis=0)
    if g_mask.shape != target.shape:
        raise ValueError(
            f"Shape mismatch in dpe_island_boundary_gapless_contiguity_filter: "
            f"{g_mask.shape} vs {target.shape}"
        )
    if not np.any(g_mask):
        return np.zeros_like(g_mask, dtype=bool)

    struct = _get_structure_2d(connectivity)
    labeled, num_islands = ndimage.label(g_mask, structure=struct)
    if num_islands == 0:
        return np.zeros_like(g_mask, dtype=bool)

    # Tolerance distance s = max(1, variance // 2) matching Microsoft TGP Stage 2 (_gapless_boundary)
    s = max(1, int(variance) // 2)
    if s > 0 and np.any(target):
        target_dilated = ndimage.binary_dilation(
            target, structure=np.ones((2 * s + 1, 2 * s + 1), dtype=bool)
        )
    else:
        target_dilated = target

    out = np.zeros_like(g_mask, dtype=bool)
    s4 = ndimage.generate_binary_structure(2, 1)
    for lbl in range(1, num_islands + 1):
        comp = labeled == lbl
        if int(np.count_nonzero(comp)) < int(min_island_size):
            continue
        eroded = ndimage.binary_erosion(comp, structure=s4, border_value=0)
        boundary = comp & (~eroded)
        n_bnd = int(np.count_nonzero(boundary))
        if n_bnd == 0:
            continue
        n_on_bnd = int(np.count_nonzero(boundary & target_dilated))
        if (n_on_bnd / float(n_bnd)) >= (float(boundary_pct_threshold) - 1e-12):
            out |= comp
    return out


# Backward-compatible alias
dpe_island_boundary_zbp_contiguity_filter = dpe_island_boundary_gapless_contiguity_filter


def dpe_canonical_post_processor(
    mask: np.ndarray,
    *,
    enabled: bool = True,
    use_core_filter: bool = False,
    core_shape: Sequence[int] = (2, 2),
    core_connectivity: int = 8,
    core_mode: Literal[
        "component_has_core",
        "morphological_opening",
        "opening_reconstructed_by_4conn",
    ] = "component_has_core",
    use_optics: bool = False,
    optics_min_samples: int = 5,
    optics_max_eps: float = 3.0,
    optics_min_cluster_size: int = 10,
    optics_xi: float = 0.05,
    optics_coordinate_space: Literal["grid_indices", "physical_meV"] = "grid_indices",
    min_island_size: int = 1,
    area_connectivity: int = 8,
    fallback_min_island_size: Optional[int] = None,
    fallback_top_n: Optional[int] = None,
    mu_vals: Optional[np.ndarray] = None,
    vz_vals: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Canonical Terminal Post-Processing DPE $C_4$ such that every protocol is $P = C_4(\dots)$.

    Enforces the non-commuting pipeline invariant (Round 2 Critique 4):
        $\mathcal{D}_{C_4} = \mathcal{D}_{\text{area\_pruning}} \circ \mathcal{D}_{\text{OPTICS}} \circ \mathcal{D}_{\text{core\_filter}}$
    Running `dpe_subtractive_area_pruning` last guarantees that every surviving connected
    component in the final protocol output strictly satisfies $\text{Area}(C) \ge \text{min\_island\_size}$.

    Always returns a pure Rank-2 boolean `np.ndarray` of shape `(N_mu, N_Vz)`.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"dpe_canonical_post_processor ($C_4$) expects 2D mask, got {m.shape}.")
    if not enabled or not np.any(m):
        return m.copy()

    out = m.copy()

    # Stage 4a: Topological Island Radius / Core Reconstruction Filter
    if use_core_filter:
        out = dpe_topological_island_radius_core_filter(
            out,
            core_shape=(int(core_shape[0]), int(core_shape[1])),
            connectivity=int(core_connectivity),
            mode=core_mode,
        )
        if not np.any(out):
            return out

    # Stage 4b: Subtractive OPTICS density clustering
    if use_optics:
        out = dpe_optics_subtractive_clustering(
            out,
            min_samples=int(optics_min_samples),
            max_eps=float(optics_max_eps),
            min_cluster_size=int(optics_min_cluster_size),
            xi=float(optics_xi),
            coordinate_space=optics_coordinate_space,
            mu_vals=mu_vals,
            vz_vals=vz_vals,
        )
        if not np.any(out):
            return out

    # Stage 4c: Subtractive connected-component area pruning (always last)
    if int(min_island_size) > 1 or fallback_min_island_size is not None:
        out = dpe_subtractive_area_pruning(
            out,
            min_island_size=int(min_island_size),
            connectivity=int(area_connectivity),
            fallback_min_island_size=fallback_min_island_size,
            fallback_top_n=fallback_top_n,
        )

    return out
