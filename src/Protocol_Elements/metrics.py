r"""Canonical Metric Evaluation & Medoid Extraction for Non-Local Protocols.

Adheres strictly to the project metric standards (`AGENTS.md` & PBrain `projects/non-local-protocol`):
1. Primary Metric: Per-realization False Discovery Rate (`FDR_pct` = $\text{FP} / (\text{TP} + \text{FP}) \times 100\%$).
   Note: When a realization predicts 0 pixels ($\text{TP} + \text{FP} = 0$), its conditional FDR is
   undefined (`NaN` for conditional active-realization FDR, `0.0` for unconditional FDR—both are
   tracked explicitly along with `realization_pass_rate_pct`).
2. Secondary Metric: True Positive Rate / Topological Area Passed (`TPR_pct` = $\text{TP} / |\Omega_{\text{topo}}| \times 100\%$).
3. Physical Quality Metrics:
   - MZM Boundary Confinement $\mathcal{C} \in [0, 1]$
   - MZM Spatial Separability $\mathcal{S} \in [0, 1]$
   - Island count, Mean Island Area (px), Max Island Area (px), and Medoid Coordinates
     $\mathbf{r}_{\text{med}} = \arg\min_{\mathbf{r} \in C_k} \|\mathbf{r} - \mathbf{r}_{\text{centroid}}\|^2$.
4. Per-Step Waterfall Diagnostics (`step_waterfall_metrics`) for every intermediate step in
   `OrderedDict[str, np.ndarray]`, saved even when `keep_data: false`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from scipy import ndimage

from .cache import RealizationFeatureCache


def extract_island_medoids_and_areas(
    mask_2d: np.ndarray,
    mu_vals: np.ndarray,
    vz_vals: np.ndarray,
    connectivity: int = 8,
) -> Dict[str, Any]:
    r"""Extract connected islands, areas, and exact in-island medoid coordinates $(\mu, V_z)$.

    Guarantees $\mathbf{r}_{\text{med}} \in C_k$ even for non-convex crescent-shaped islands.
    """
    m = np.asarray(mask_2d, dtype=bool)
    if not np.any(m):
        return {
            "island_count": 0,
            "mean_island_area_px": 0.0,
            "max_island_area_px": 0,
            "medoids_mu_vz": [],
            "centers_of_mass_mu_vz": [],
            "clusters": [],
        }

    struct = (
        np.ones((3, 3), dtype=bool)
        if int(connectivity) == 8
        else ndimage.generate_binary_structure(2, 1)
    )
    labeled, num_islands = ndimage.label(m, structure=struct)
    if num_islands == 0:
        return {
            "island_count": 0,
            "mean_island_area_px": 0.0,
            "max_island_area_px": 0,
            "medoids_mu_vz": [],
            "centers_of_mass_mu_vz": [],
            "clusters": [],
        }

    areas: List[int] = []
    clusters: List[Dict[str, Any]] = []
    centers_of_mass: List[Dict[str, Any]] = []
    for lbl in range(1, num_islands + 1):
        pts = np.argwhere(labeled == lbl)
        area = int(pts.shape[0])
        if area == 0:
            continue
        areas.append(area)
        phys_pts = np.column_stack([mu_vals[pts[:, 0]], vz_vals[pts[:, 1]]])
        centroid = np.mean(phys_pts, axis=0)
        com_idx = np.mean(pts, axis=0)
        dist_sq = np.sum((phys_pts - centroid) ** 2, axis=1)
        med_idx = int(np.argmin(dist_sq))
        cluster_entry = {
            "island_id": int(lbl),
            "area_px": area,
            # Exact continuous center of mass (centroid)
            "center_of_mass_mu_meV": round(float(centroid[0]), 5),
            "center_of_mass_Vz_meV": round(float(centroid[1]), 5),
            "center_of_mass_mu_idx": round(float(com_idx[0]), 2),
            "center_of_mass_Vz_idx": round(float(com_idx[1]), 2),
            # In-cluster medoid point (guaranteed to lie within cluster C_k)
            "medoid_mu_meV": round(float(phys_pts[med_idx, 0]), 5),
            "medoid_Vz_meV": round(float(phys_pts[med_idx, 1]), 5),
            "medoid_mu_idx": int(pts[med_idx, 0]),
            "medoid_Vz_idx": int(pts[med_idx, 1]),
            # Backward-compatible aliases
            "mu_meV": round(float(phys_pts[med_idx, 0]), 5),
            "Vz_meV": round(float(phys_pts[med_idx, 1]), 5),
            "mu_idx": int(pts[med_idx, 0]),
            "Vz_idx": int(pts[med_idx, 1]),
        }
        clusters.append(cluster_entry)
        centers_of_mass.append({
            "island_id": int(lbl),
            "area_px": area,
            "mu_meV": round(float(centroid[0]), 5),
            "Vz_meV": round(float(centroid[1]), 5),
            "mu_idx": round(float(com_idx[0]), 2),
            "Vz_idx": round(float(com_idx[1]), 2),
        })

    return {
        "island_count": int(len(areas)),
        "mean_island_area_px": float(np.mean(areas)) if areas else 0.0,
        "max_island_area_px": int(np.max(areas)) if areas else 0,
        "medoids_mu_vz": clusters,
        "centers_of_mass_mu_vz": centers_of_mass,
        "clusters": clusters,
    }


def compute_step_diagnostic(
    step_mask: np.ndarray,
    pf_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute lightweight `(survival_pct, fdr_pct, tpr_pct)` for a single step mask (Rank-2 or Rank-3)."""
    m = np.asarray(step_mask, dtype=bool)
    gt = np.asarray(pf_mask, dtype=bool)
    n_total = float(max(gt.size, 1))
    n_gt = float(np.count_nonzero(gt))

    if m.ndim == 3:
        # Per-cutter average diagnostic
        surv_list, fdr_list, tpr_list = [], [], []
        for c in range(m.shape[0]):
            mc = m[c]
            n_pred = int(np.count_nonzero(mc))
            tp = int(np.count_nonzero(mc & gt))
            fp = n_pred - tp
            surv_list.append(100.0 * n_pred / n_total)
            fdr_list.append(100.0 * fp / n_pred if n_pred > 0 else 0.0)
            tpr_list.append(100.0 * tp / n_gt if n_gt > 0 else 0.0)
        return {
            "rank": 3.0,
            "survival_pct": float(np.mean(surv_list)),
            "fdr_pct": float(np.mean(fdr_list)),
            "tpr_pct": float(np.mean(tpr_list)),
        }

    n_pred = int(np.count_nonzero(m))
    tp = int(np.count_nonzero(m & gt))
    fp = n_pred - tp
    surv_pct = 100.0 * n_pred / n_total
    fdr_pct = 100.0 * fp / n_pred if n_pred > 0 else 0.0
    tpr_pct = 100.0 * tp / n_gt if n_gt > 0 else 0.0
    return {
        "rank": 2.0,
        "survival_pct": float(surv_pct),
        "fdr_pct": float(fdr_pct),
        "tpr_pct": float(tpr_pct),
    }


def evaluate_realization_metrics(
    cache: RealizationFeatureCache,
    steps: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Evaluate complete per-realization metrics on the final mask and all intermediate steps."""
    if not steps:
        raise ValueError("evaluate_realization_metrics received empty steps dict.")

    final_key = list(steps.keys())[-1]
    final_mask = np.asarray(steps[final_key], dtype=bool)
    if final_mask.ndim != 2:
        raise ValueError(
            f"Final step '{final_key}' must be Rank-2 `(N_mu, N_Vz)`, got shape {final_mask.shape}."
        )

    pf = cache.pfaffian_mask
    conf = cache.boundary_confinement
    sep = cache.mzm_separability

    n_total = int(pf.size)
    n_gt = int(np.count_nonzero(pf))
    n_pred = int(np.count_nonzero(final_mask))
    tp = int(np.count_nonzero(final_mask & pf))
    fp = n_pred - tp

    has_prediction = n_pred > 0
    fdr_unconditional_pct = 100.0 * fp / n_pred if has_prediction else 0.0
    fdr_conditional_pct = 100.0 * fp / n_pred if has_prediction else float("nan")
    tpr_pct = 100.0 * tp / n_gt if n_gt > 0 else 0.0
    survival_pct = 100.0 * n_pred / max(n_total, 1)

    if has_prediction:
        mean_conf = float(np.mean(conf[final_mask]))
        mean_sep = float(np.mean(sep[final_mask]))
    else:
        mean_conf = 0.0
        mean_sep = 0.0

    island_info = extract_island_medoids_and_areas(
        final_mask, cache.mu_vals, cache.vz_vals, connectivity=8
    )

    step_diag = {
        step_name: compute_step_diagnostic(step_arr, pf)
        for step_name, step_arr in steps.items()
    }

    return {
        "realization_name": cache.realization_name,
        "v0_label": cache.v0_label,
        "v0_mev": float(cache.v0_mev),
        "temperature_mK": float(cache.temperature_mK),
        "has_prediction": bool(has_prediction),
        "n_predicted_px": int(n_pred),
        "n_true_positive_px": int(tp),
        "n_false_positive_px": int(fp),
        "n_ground_truth_topo_px": int(n_gt),
        "survival_pct": round(float(survival_pct), 4),
        "FDR_pct": round(float(fdr_unconditional_pct), 4),
        "FDR_active_only_pct": (
            round(float(fdr_conditional_pct), 4) if has_prediction else None
        ),
        "TPR_pct": round(float(tpr_pct), 4),
        "MZM_confinement_C": round(float(mean_conf), 4),
        "MZM_separability_S": round(float(mean_sep), 4),
        "island_count": int(island_info.get("island_count", 0)),
        "mean_island_area_px": round(float(island_info.get("mean_island_area_px", 0.0)), 2),
        "max_island_area_px": int(island_info.get("max_island_area_px", 0)),
        "medoids_mu_vz": island_info.get("medoids_mu_vz", []),
        "centers_of_mass_mu_vz": island_info.get("centers_of_mass_mu_vz", []),
        "clusters": island_info.get("clusters", []),
        "step_diagnostics": step_diag,
    }


def aggregate_ensemble_metrics(
    realization_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-realization metrics across an ensemble (never reporting F1 or pooled FDR as headline)."""
    if not realization_records:
        return {}

    def _agg_subset(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        n_r = len(records)
        if n_r == 0:
            return {}
        fdrs = np.array([r["FDR_pct"] for r in records], dtype=np.float64)
        active_fdrs = np.array(
            [r["FDR_active_only_pct"] for r in records if r["FDR_active_only_pct"] is not None],
            dtype=np.float64,
        )
        tprs = np.array([r["TPR_pct"] for r in records], dtype=np.float64)
        survs = np.array([r["survival_pct"] for r in records], dtype=np.float64)
        confs = np.array([r["MZM_confinement_C"] for r in records], dtype=np.float64)
        seps = np.array([r["MZM_separability_S"] for r in records], dtype=np.float64)
        active_confs = np.array(
            [r["MZM_confinement_C"] for r in records if r["has_prediction"]], dtype=np.float64
        )
        active_seps = np.array(
            [r["MZM_separability_S"] for r in records if r["has_prediction"]], dtype=np.float64
        )
        islands = np.array([r["island_count"] for r in records], dtype=np.float64)
        mean_areas = np.array([r["mean_island_area_px"] for r in records], dtype=np.float64)
        max_areas = np.array([r["max_island_area_px"] for r in records], dtype=np.float64)
        pass_rate = 100.0 * sum(1 for r in records if r["has_prediction"]) / float(n_r)

        # Aggregate per-step waterfall metrics
        step_names = list(records[0]["step_diagnostics"].keys())
        waterfall: Dict[str, Dict[str, float]] = {}
        for sname in step_names:
            s_surv = np.array([r["step_diagnostics"][sname]["survival_pct"] for r in records])
            s_fdr = np.array([r["step_diagnostics"][sname]["fdr_pct"] for r in records])
            s_tpr = np.array([r["step_diagnostics"][sname]["tpr_pct"] for r in records])
            waterfall[sname] = {
                "rank": int(records[0]["step_diagnostics"][sname]["rank"]),
                "mean_survival_pct": round(float(np.mean(s_surv)), 4),
                "mean_FDR_pct": round(float(np.mean(s_fdr)), 4),
                "median_FDR_pct": round(float(np.median(s_fdr)), 4),
                "mean_TPR_pct": round(float(np.mean(s_tpr)), 4),
                "median_TPR_pct": round(float(np.median(s_tpr)), 4),
            }

        return {
            "n_realizations": int(n_r),
            "realization_pass_rate_pct": round(float(pass_rate), 2),
            "mean_FDR_pct": round(float(np.mean(fdrs)), 4),
            "median_FDR_pct": round(float(np.median(fdrs)), 4),
            "std_FDR_pct": round(float(np.std(fdrs)), 4),
            "mean_FDR_active_only_pct": (
                round(float(np.mean(active_fdrs)), 4) if active_fdrs.size > 0 else 0.0
            ),
            "mean_TPR_pct": round(float(np.mean(tprs)), 4),
            "median_TPR_pct": round(float(np.median(tprs)), 4),
            "std_TPR_pct": round(float(np.std(tprs)), 4),
            "mean_survival_pct": round(float(np.mean(survs)), 4),
            "mean_MZM_confinement_C": round(float(np.mean(confs)), 4),
            "mean_MZM_confinement_C_active": (
                round(float(np.mean(active_confs)), 4) if active_confs.size > 0 else 0.0
            ),
            "mean_MZM_separability_S": round(float(np.mean(seps)), 4),
            "mean_MZM_separability_S_active": (
                round(float(np.mean(active_seps)), 4) if active_seps.size > 0 else 0.0
            ),
            "mean_island_count": round(float(np.mean(islands)), 3),
            "mean_island_area_px": round(float(np.mean(mean_areas)), 2),
            "max_island_area_px": int(np.max(max_areas)),
            "step_waterfall_metrics": waterfall,
        }

    overall = _agg_subset(realization_records)

    # Breakdown by (temperature_mK, v0_label)
    by_temp_and_v0: Dict[str, Any] = {}
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in realization_records:
        gkey = f"{int(round(rec['temperature_mK']))}mK__{rec['v0_label']}"
        groups.setdefault(gkey, []).append(rec)
    for gkey, recs in sorted(groups.items()):
        by_temp_and_v0[gkey] = _agg_subset(recs)

    overall["by_cohort"] = by_temp_and_v0
    return overall
