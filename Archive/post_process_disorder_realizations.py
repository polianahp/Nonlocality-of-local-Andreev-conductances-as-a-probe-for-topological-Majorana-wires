#!/usr/bin/env python3
"""
post_process_disorder_realizations.py

High-performance single-disorder ensemble post-processing and benchmarking pipeline.
Discovers and processes disorder realizations belonging to a single disorder strength
(e.g., across 50 realizations per disorder strength on Harpers Ferry).

Evaluates and compares:
  1. Microsoft Azure Quantum TGP Baseline (Stage 2 ROI 2)
  2. Legacy Reflection Correlation Protocol (Correlation + 3w + Gap)
  3. S-ACD Stage 1: Asymmetric Conductance Deviation + Conjugate Lead Symmetry Gate
  4. S-ACD Stage 2: Multi-Cutter 3w Curvature Consensus
  5. S-ACD Stage 3: Unclustered Pointwise Intersection (+ Transport Gap)
  6. S-ACD Stage 4 [Directional Manifold Clustered]: Anisotropic morphological closing + island pruning
  7. S-ACD Stage 4 [OPTICS Clustered]: Density-based reachability clustering

Calculates strictly True Positive Rate (TPR %), False Discovery Rate (FDR %),
island counts, primary island size (px), and microscopic wavefunction localization
metrics strictly on True Positives (Boundary Confinement and MZM Separability).
"""

import os
import sys
import argparse
import time
import json
import shutil
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter
from scipy.ndimage import label
from sklearn.cluster import OPTICS
from multiprocessing import Pool, cpu_count

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tgp_adapter import TGPAdapter
from src.cpu_broadening import prepare_sim_cpu
import tgp.prepare
import tgp.two

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PostProcessDisorder")


# ==============================================================================
# 1. CORE PROTOCOL OPERATORS & ALGORITHMIC ROUTINES
# ==============================================================================

def vectorized_reflection_correlation(
    f1: np.ndarray,
    f2: np.ndarray,
    barrier_arr: np.ndarray
) -> np.ndarray:
    """
    Computes normalized correlation C(f1, f2) across barrier traces
    restricted to the non-negative barrier regime (U >= 0).
    f1, f2: shape (N_pixels, N_barriers)
    barrier_arr: shape (N_barriers,)
    Returns: corr array of shape (N_pixels,)
    """
    pos_mask = barrier_arr >= 0.0
    f1_m = f1[:, pos_mask]
    f2_m = f2[:, pos_mask]

    denom1 = np.where(f1_m[:, 0:1] != 0, f1_m[:, 0:1], 1.0)
    denom2 = np.where(f2_m[:, 0:1] != 0, f2_m[:, 0:1], 1.0)
    f1_norm = f1_m / denom1
    f2_norm = f2_m / denom2

    norm1 = np.linalg.norm(f1_norm, axis=1)
    norm2 = np.linalg.norm(f2_norm, axis=1)

    dot = np.sum(f1_norm * f2_norm, axis=1)
    denom = norm1 * norm2
    return np.where(denom > 0, dot / denom, 0.0)


def compute_sacd_stage1(
    g_LL_sym: np.ndarray,
    g_LL_pinch: np.ndarray,
    g_RR_sym: np.ndarray,
    g_RR_pinch: np.ndarray,
    tau: float = 0.05,
    s_max: float = 0.20,
    floor_val: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Stage 1 Asymmetric Conductance Deviation (aL, aR) and Lead Symmetry Gate (S_sym)
    Grounded in Dourado et al. [Phys. Rev. B 110, 014504 (2024), Fig. 13].
    """
    dG_L = np.abs(g_LL_sym - g_LL_pinch)
    dG_R = np.abs(g_RR_sym - g_RR_pinch)

    denom_l = np.where(g_LL_sym >= floor_val, g_LL_sym, 1.0)
    aL = np.clip(dG_L / denom_l, 0.0, 1.0)
    aL[g_LL_sym < floor_val] = 0.0

    denom_r = np.where(g_RR_sym >= floor_val, g_RR_sym, 1.0)
    aR = np.clip(dG_R / denom_r, 0.0, 1.0)
    aR[g_RR_sym < floor_val] = 0.0

    sqrt_a = np.sqrt(aL * aR)
    denom_sym = np.maximum(g_LL_sym + g_RR_sym, 1e-4)
    S_sym = np.abs(g_LL_sym - g_RR_sym) / denom_sym

    m_stage1 = (sqrt_a >= tau) & (S_sym <= s_max)
    return m_stage1, sqrt_a, S_sym


def compute_sacd_stage2_curvature(
    dIdVl: np.ndarray,
    dIdVr: np.ndarray,
    Nmu: int,
    Nvz: int,
    d2_thresh: float = 20.0,
    min_cutters: int = 2,
    win_len: int = 7,
    polyorder: int = 2
) -> np.ndarray:
    """
    Computes 3w Zero-Bias Negative Curvature Consensus across 5 cutter gates:
      (d^2 G / dV^2)|_{V=0} <= -d2_thresh on BOTH Left and Right leads
    """
    bias = np.linspace(-0.134, 0.134, 97)
    delta = bias[1] - bias[0]
    zero_bias_idx = 48
    pass_cutters = np.zeros((Nmu * Nvz, 5), dtype=bool)

    for c in range(5):
        d2_l = savgol_filter(dIdVl[:, c, :], window_length=win_len, polyorder=polyorder, deriv=2, delta=delta, axis=1)[:, zero_bias_idx]
        d2_r = savgol_filter(dIdVr[:, c, :], window_length=win_len, polyorder=polyorder, deriv=2, delta=delta, axis=1)[:, zero_bias_idx]
        pass_cutters[:, c] = (d2_l <= -d2_thresh) & (d2_r <= -d2_thresh)

    pass_consensus = (np.sum(pass_cutters, axis=1) >= min_cutters)
    return pass_consensus.reshape(Nmu, Nvz)


def compute_sacd_stage3_gap(
    topological_gap_2d: np.ndarray,
    min_gap_uev: float = 10.0
) -> np.ndarray:
    """Computes Non-Local Transport Gap localization gate (Delta_trans >= min_gap_uev)."""
    return topological_gap_2d >= (min_gap_uev / 1000.0)


def apply_subtractive_island_filter(
    raw_mask: np.ndarray,
    min_island_size: int = 15
) -> Tuple[np.ndarray, int, int]:
    """
    Applies Subtractive Island Filtering:
      (clustered_mask, total_islands_retained, max_island_size)
    """
    lbl, n_components = label(raw_mask, structure=np.ones((3, 3), dtype=int))
    sizes = [int(np.sum(lbl == i)) for i in range(1, n_components + 1)]

    clustered_mask = np.zeros_like(raw_mask, dtype=bool)
    for i, s in enumerate(sizes, 1):
        if s >= min_island_size:
            clustered_mask[lbl == i] = True

    lbl_out, n_out = label(clustered_mask, structure=np.ones((3, 3), dtype=int))
    out_sizes = sorted([int(np.sum(lbl_out == i)) for i in range(1, n_out + 1)], reverse=True)
    max_island = out_sizes[0] if out_sizes else 0

    return clustered_mask, n_out, max_island


def apply_directional_manifold_clustering(
    raw_mask: np.ndarray,
    min_island_size: int = 15,
    **kwargs
) -> Tuple[np.ndarray, int, int]:
    """
    Subtractive Island Filtering (Legacy alias for apply_subtractive_island_filter).
    Note: kernel_mu and kernel_vz are obsolete and ignored; no dilation or morphological closing is performed.
    """
    return apply_subtractive_island_filter(raw_mask, min_island_size=min_island_size)


def apply_optics_density_clustering(
    raw_mask: np.ndarray,
    min_samples: int = 5,
    max_eps: float = 3.0,
    min_cluster_size: int = 15,
    xi: float = 0.05
) -> Tuple[np.ndarray, int, int]:
    """
    Applies Scikit-Learn OPTICS Density Reachability Clustering subtractively:
      Clusters 2D candidate coordinates and discards unclustered noise (label == -1).
    Returns:
      (clustered_mask, total_islands_retained, max_island_size)
    """
    X = np.argwhere(raw_mask)
    clustered_mask = np.zeros_like(raw_mask, dtype=bool)

    if min_cluster_size <= len(X) <= 1500:
        opt = OPTICS(min_samples=min_samples, max_eps=max_eps, min_cluster_size=min_cluster_size, xi=xi)
        labs = opt.fit_predict(X)
        for idx, lab in enumerate(labs):
            if lab >= 0:
                clustered_mask[X[idx, 0], X[idx, 1]] = True

    lbl_out, n_out = label(clustered_mask, structure=np.ones((3, 3), dtype=int))
    out_sizes = sorted([int(np.sum(lbl_out == i)) for i in range(1, n_out + 1)], reverse=True)
    max_island = out_sizes[0] if out_sizes else 0

    return clustered_mask, n_out, max_island


# ==============================================================================
# 2. LOCALIZATION METRIC LOADERS & EVALUATOR
# ==============================================================================

def load_boundary_confinement(realization_dir: Path, Nmu: int, Nvz: int, Ls: int = 150) -> np.ndarray:
    """
    Loads 2D boundary confinement map from realization folder using fallback priority:
    1. mzm_boundary_confinement_arr.npy
    2. site_localizations.npy (normalized by Ls if integer penetration depth)
    3. weight_localization_arr.npy
    """
    candidates = [
        "mzm_boundary_confinement_arr.npy",
        "site_localizations.npy",
        "weight_localization_arr.npy",
    ]
    for c in candidates:
        p = realization_dir / c
        if p.exists():
            arr = np.load(p)
            if c == "site_localizations.npy" and np.nanmax(arr) > 1.0:
                arr = 1.0 - (arr.astype(float) / float(Ls))
                arr = np.clip(arr, 0.0, 1.0)
            return arr.reshape(Nmu, Nvz)

    # Check parent directory fallback if temperature subfolder
    if realization_dir.parent.exists():
        for c in candidates:
            p = realization_dir.parent / realization_dir.name.split("_")[0] / c
            if p.exists():
                arr = np.load(p)
                return arr.reshape(Nmu, Nvz)

    return np.zeros((Nmu, Nvz), dtype=float)


def load_separability(realization_dir: Path, Nmu: int, Nvz: int) -> np.ndarray:
    """
    Loads 2D MZM separability map from realization folder using fallback priority:
    1. mzm_separability_arr.npy
    2. mzm_separation_arr.npy
    """
    candidates = [
        "mzm_separability_arr.npy",
        "mzm_separation_arr.npy",
    ]
    for c in candidates:
        p = realization_dir / c
        if p.exists():
            arr = np.load(p)
            return arr.reshape(Nmu, Nvz)

    # Check parent directory fallback if temperature subfolder
    if realization_dir.parent.exists():
        for c in candidates:
            p = realization_dir.parent / realization_dir.name.split("_")[0] / c
            if p.exists():
                arr = np.load(p)
                return arr.reshape(Nmu, Nvz)

    return np.zeros((Nmu, Nvz), dtype=float)


def calc_protocol_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    conf_2d: np.ndarray = None,
    sep_2d: np.ndarray = None,
    tgp_mask: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Calculates confusion matrix, performance rates (TPR, FDR, PPV, F1),
    island contiguity metrics (island count, primary island size), and
    microscopic wavefunction localization metrics strictly on True Positives (TP).
    """
    pred_b = pred_mask.astype(bool)
    gt_b = gt_mask.astype(bool)

    tp_b = pred_b & gt_b
    fp_b = pred_b & (~gt_b)
    fn_b = (~pred_b) & gt_b
    tn_b = (~pred_b) & (~gt_b)

    tp = int(np.sum(tp_b))
    fp = int(np.sum(fp_b))
    fn = int(np.sum(fn_b))
    tn = int(np.sum(tn_b))

    tot_pass = tp + fp
    tot_topological = int(np.sum(gt_b))

    # Rates
    pfaffian_passed_pct = float((tp / tot_topological) * 100.0) if tot_topological > 0 else 0.0
    fdr_pct = float((fp / tot_pass) * 100.0) if tot_pass > 0 else 0.0
    ppv_pct = float((tp / tot_pass) * 100.0) if tot_pass > 0 else 0.0

    precision = tp / tot_pass if tot_pass > 0 else 0.0
    recall = tp / tot_topological if tot_topological > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # TGP True Positive coverage
    tgp_tp_coverage_pct = 0.0
    if tgp_mask is not None:
        tgp_tp_b = tgp_mask.astype(bool) & gt_b
        n_tgp_tp = int(np.sum(tgp_tp_b))
        if n_tgp_tp > 0:
            tgp_tp_coverage_pct = round(float(np.sum(pred_b & tgp_tp_b) / n_tgp_tp * 100.0), 2)

    # Contiguity & Island statistics
    lbl, n_components = label(pred_b, structure=np.ones((3, 3), dtype=int))
    sizes = sorted([int(np.sum(lbl == i)) for i in range(1, n_components + 1)], reverse=True)
    max_island = sizes[0] if sizes else 0
    mean_island_size = float(np.mean(sizes)) if sizes else 0.0

    # Localization metrics on True Positives strictly
    tp_mean_conf, tp_var_conf, tp_std_conf = 0.0, 0.0, 0.0
    tp_mean_sep, tp_var_sep, tp_std_sep = 0.0, 0.0, 0.0

    if tp > 0:
        if conf_2d is not None:
            c_tp = conf_2d[tp_b]
            tp_mean_conf = float(np.nanmean(c_tp))
            tp_var_conf = float(np.nanvar(c_tp))
            tp_std_conf = float(np.nanstd(c_tp))
        if sep_2d is not None:
            s_tp = sep_2d[tp_b]
            tp_mean_sep = float(np.nanmean(s_tp))
            tp_var_sep = float(np.nanvar(s_tp))
            tp_std_sep = float(np.nanstd(s_tp))

    # Also record localization metrics across all positive predictions (pass)
    pass_mean_conf, pass_var_conf = 0.0, 0.0
    pass_mean_sep, pass_var_sep = 0.0, 0.0

    if tot_pass > 0:
        if conf_2d is not None:
            c_pass = conf_2d[pred_b]
            pass_mean_conf = float(np.nanmean(c_pass))
            pass_var_conf = float(np.nanvar(c_pass))
        if sep_2d is not None:
            s_pass = sep_2d[pred_b]
            pass_mean_sep = float(np.nanmean(s_pass))
            pass_var_sep = float(np.nanvar(s_pass))

    return {
        "tot_passed": tot_pass,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "total_topological_points": tot_topological,
        "total_pfaffian_area": tot_topological,
        "pfaffian_passed_pct": round(pfaffian_passed_pct, 4),
        "fdr_percent": round(fdr_pct, 4),
        "ppv_percent": round(ppv_pct, 4),
        "f1_score": round(float(f1), 4),
        "n_islands": n_components,
        "max_island": max_island,
        "mean_island_size": round(mean_island_size, 2),
        # True Positive localization
        "tp_mean_boundary_confinement": round(tp_mean_conf, 4),
        "tp_var_boundary_confinement": round(tp_var_conf, 4),
        "tp_std_boundary_confinement": round(tp_std_conf, 4),
        "tp_mean_separability": round(tp_mean_sep, 4),
        "tp_var_separability": round(tp_var_sep, 4),
        "tp_std_separability": round(tp_std_sep, 4),
        # All passed predictions localization
        "mean_boundary_confinement": round(pass_mean_conf, 4),
        "var_boundary_confinement": round(pass_var_conf, 4),
        "mean_separability": round(pass_mean_sep, 4),
        "var_separability": round(pass_var_sep, 4),
        "tgp_tp_coverage_pct": tgp_tp_coverage_pct,
    }


# ==============================================================================
# 3. WORKER FUNCTION FOR SINGLE REALIZATION PROCESSING
# ==============================================================================

def process_single_realization(args_tuple: Tuple) -> Dict[str, Any]:
    """
    Worker task: Processes one realization directory across all protocols.
    """
    realization_dir_str, config_dict = args_tuple
    realization_dir = Path(realization_dir_str)
    r_name = realization_dir.name

    t0 = time.perf_counter()

    # Protocol configuration parameters
    gap_th_factor = config_dict.get("gap_th_factor", 0.05)
    corr_threshold = config_dict.get("corr_threshold", 0.85)
    tau = config_dict.get("tau", 0.01)
    s_max = config_dict.get("s_max", 0.20)
    d2_threshold = config_dict.get("d2_threshold", 100.0)
    min_cutters = config_dict.get("min_cutters", 3)
    min_gap_uev = config_dict.get("min_gap_uev", 10.0)
    min_island_size = config_dict.get("min_island_size", 10)
    optics_min_samples = config_dict.get("optics_min_samples", 5)
    optics_max_eps = config_dict.get("optics_max_eps", 3.0)
    optics_min_cluster_size = config_dict.get("optics_min_cluster_size", 10)
    optics_xi = config_dict.get("optics_xi", 0.05)
    barrier_T = config_dict.get("barrier_T_mK", 40.0)

    protocol_mode = config_dict.get("protocol_mode", "strict_and")

    try:
        # 1. Parameter extraction
        params_path = realization_dir / "all_params.npz"
        params = np.load(params_path, allow_pickle=True)
        mu_var = params["mu_var"]
        Vz_var = params["Vz_var"]
        Nmu = len(mu_var)
        Nvz = len(Vz_var)
        v0_val = float(params.get("V0", 0.0))

        # 2. Ground truth topological invariant
        pdi_path = realization_dir / "pdi_data.npy"
        pdi_data = np.load(pdi_path)
        col_idx = 3 if pdi_data.shape[1] >= 4 else 2
        gt_flat = (pdi_data[:, col_idx] == 1.0)
        gt_2d = gt_flat.reshape(Nmu, Nvz)

        # 3. Load barrier conductance data
        temp_suffix = ""
        if barrier_T is not None and float(barrier_T) > 0:
            b_val = float(barrier_T)
            t_str = f"{int(b_val)}mK" if b_val == int(b_val) else f"{b_val:.1f}mK"
            temp_suffix = f"_{t_str}"

        b_arr = np.load(realization_dir / "barrier_arr.npy")

        def _load_barrier_arr(base_name: str) -> np.ndarray:
            if temp_suffix:
                p_suffixed = realization_dir / f"{base_name}{temp_suffix}.npy"
                if p_suffixed.exists():
                    return np.load(p_suffixed)
            return np.load(realization_dir / f"{base_name}.npy")

        bl_l = _load_barrier_arr("barrier_left_conductance_left_arr")
        bl_r = _load_barrier_arr("barrier_left_conductance_right_arr")
        br_l = _load_barrier_arr("barrier_right_conductance_left_arr")
        br_r = _load_barrier_arr("barrier_right_conductance_right_arr")
        # Dynamic barrier indexing: U0 = 0.0 meV symmetric reference, max(b_arr) pinchoff
        idx_sym = int(np.argmin(np.abs(b_arr - 0.0)))
        idx_pinch = int(np.argmax(b_arr))

        g_LL_sym = br_l[:, idx_sym].reshape(Nmu, Nvz)
        g_LL_pinch = br_l[:, idx_pinch].reshape(Nmu, Nvz)
        g_RR_sym = bl_r[:, idx_sym].reshape(Nmu, Nvz)
        g_RR_pinch = bl_r[:, idx_pinch].reshape(Nmu, Nvz)

        dIdVl = np.load(realization_dir / "tgp_stage1_dIdVl.npy")
        dIdVr = np.load(realization_dir / "tgp_stage1_dIdVr.npy")
        topological_gap_2d = np.load(realization_dir / "topological_gap.npy").reshape(Nmu, Nvz)

        # 4. Load boundary confinement and separability maps
        conf_2d = load_boundary_confinement(realization_dir, Nmu, Nvz, Ls=int(params.get("Ls", 150)))
        sep_2d = load_separability(realization_dir, Nmu, Nvz)

        # 5. Microsoft Azure Quantum TGP Baseline (Stage 2 ROI 2)
        adapter = TGPAdapter(str(realization_dir))
        ds = adapter.to_xarray()
        tgp_T = float(barrier_T) if (barrier_T is not None and float(barrier_T) > 0) else 30.0
        tprep = prepare_sim_cpu(ds, T_mK=tgp_T)

        t_l = tprep.rename({"bias": "left_bias"})
        t_r = tprep.rename({"bias": "right_bias"})
        t_l, t_r = tgp.two.extract_gap(
            t_l, t_r,
            gap_threshold_factor=gap_th_factor,
            noise_threshold=1e-4
        )
        z_ds = tgp.two.zbp_dataset_derivative(
            t_l, t_r,
            zbp_probability_threshold=0.7,
            average_over_cutter=False
        )
        tgp.two.set_gap_threshold(z_ds, threshold_low=10e-3, threshold_high=0.067)

        s2 = tgp.two.cluster_and_score(
            z_ds,
            min_cluster_size=7,
            cluster_gap_threshold=10e-3,
            cluster_percentage_boundary_threshold=0.6,
            cluster_ncutter_threshold=0.5
        )
        roi2_vb = s2.roi2.values.T if s2.roi2.dims == ("B", "V") else s2.roi2.values
        m_tgp = (roi2_vb > 0)

        # 6. Legacy Reflection Correlation Protocol
        cL = vectorized_reflection_correlation(bl_l, bl_r, b_arr).reshape(Nmu, Nvz)
        cR = vectorized_reflection_correlation(br_l, br_r, b_arr).reshape(Nmu, Nvz)
        m_legacy_s1 = (cL >= corr_threshold) & (cR >= corr_threshold)
        gzbp_2d = (z_ds["gapped_zbp"].mean(dim="cutter_pair_index") >= 0.5).transpose("V", "B").values
        m_legacy_s3 = m_legacy_s1 & gzbp_2d

        # 7. S-ACD Protocol Suite (Stages 1-4)
        m_sacd_s1, _, _ = compute_sacd_stage1(g_LL_sym, g_LL_pinch, g_RR_sym, g_RR_pinch, tau=tau, s_max=s_max)
        m_curv = compute_sacd_stage2_curvature(dIdVl, dIdVr, Nmu, Nvz, d2_thresh=d2_threshold, min_cutters=min_cutters)
        m_gap = compute_sacd_stage3_gap(topological_gap_2d, min_gap_uev=min_gap_uev)

        if protocol_mode == "or_gate":
            m_sacd_s2 = m_sacd_s1 & (m_curv | m_gap)
            m_sacd_s3_raw = m_sacd_s2
        elif protocol_mode == "curv_only":
            m_sacd_s2 = m_sacd_s1 & m_curv
            m_sacd_s3_raw = m_sacd_s2
        elif protocol_mode == "gap_only":
            m_sacd_s2 = m_sacd_s1 & m_gap
            m_sacd_s3_raw = m_sacd_s2
        else:  # strict_and (default baseline)
            m_sacd_s2 = m_sacd_s1 & m_curv
            m_sacd_s3_raw = m_sacd_s2 & m_gap

        # Stage 4A: Purely Subtractive Island Filtering (Zero Dilation)
        m_sacd_manifold, n_isl_man, max_isl_man = apply_subtractive_island_filter(
            m_sacd_s3_raw, min_island_size=min_island_size
        )

        # Stage 4B: OPTICS Density Clustering
        m_sacd_optics, n_isl_opt, max_isl_opt = apply_optics_density_clustering(
            m_sacd_s3_raw, min_samples=optics_min_samples, max_eps=optics_max_eps,
            min_cluster_size=optics_min_cluster_size, xi=optics_xi
        )

        # 8. Compute detailed metrics for all protocols
        gt_b = gt_2d.astype(bool)
        gt_metrics = {
            "gt_total_area": int(np.sum(gt_b)),
            "gt_mean_boundary_confinement": round(float(np.nanmean(conf_2d[gt_b])), 4) if np.sum(gt_b) > 0 else 0.0,
            "gt_var_boundary_confinement": round(float(np.nanvar(conf_2d[gt_b])), 4) if np.sum(gt_b) > 0 else 0.0,
            "gt_mean_separability": round(float(np.nanmean(sep_2d[gt_b])), 4) if np.sum(gt_b) > 0 else 0.0,
            "gt_var_separability": round(float(np.nanvar(sep_2d[gt_b])), 4) if np.sum(gt_b) > 0 else 0.0,
        }

        metrics_tgp = calc_protocol_metrics(m_tgp, gt_2d, conf_2d, sep_2d, tgp_mask=m_tgp)
        metrics_legacy = calc_protocol_metrics(m_legacy_s3, gt_2d, conf_2d, sep_2d, tgp_mask=m_tgp)
        metrics_sacd_s1 = calc_protocol_metrics(m_sacd_s1, gt_2d, conf_2d, sep_2d, tgp_mask=m_tgp)
        metrics_sacd_s2 = calc_protocol_metrics(m_sacd_s2, gt_2d, conf_2d, sep_2d, tgp_mask=m_tgp)
        metrics_sacd_s3 = calc_protocol_metrics(m_sacd_s3_raw, gt_2d, conf_2d, sep_2d, tgp_mask=m_tgp)
        metrics_sacd_man = calc_protocol_metrics(m_sacd_manifold, gt_2d, conf_2d, sep_2d, tgp_mask=m_tgp)
        metrics_sacd_opt = calc_protocol_metrics(m_sacd_optics, gt_2d, conf_2d, sep_2d, tgp_mask=m_tgp)

        elapsed = time.perf_counter() - t0

        return {
            "success": True,
            "realization": r_name,
            "realization_dir": str(realization_dir),
            "V0": v0_val,
            "Nmu": Nmu,
            "Nvz": Nvz,
            "elapsed_seconds": round(elapsed, 2),
            "ground_truth_metrics": gt_metrics,
            # Protocol metrics
            "regular_tgp": metrics_tgp,
            "legacy_step3": metrics_legacy,
            "sacd_stage1": metrics_sacd_s1,
            "sacd_stage2": metrics_sacd_s2,
            "sacd_stage3_raw": metrics_sacd_s3,
            "sacd_manifold": metrics_sacd_man,
            "sacd_optics": metrics_sacd_opt,
            # Phase diagrams for visualization
            "mu_var": mu_var,
            "Vz_var": Vz_var,
            "gt_2d": gt_2d,
            "m_tgp": m_tgp,
            "m_sacd_s3_raw": m_sacd_s3_raw,
            "m_sacd_manifold": m_sacd_manifold,
            "m_sacd_optics": m_sacd_optics,
            "conf_2d": conf_2d,
            "sep_2d": sep_2d,
        }

    except Exception as e:
        logger.error(f"Error processing {r_name}: {str(e)}", exc_info=True)
        return {
            "success": False,
            "realization": r_name,
            "realization_dir": str(realization_dir),
            "error": str(e),
        }


# ==============================================================================
# 4. ENSEMBLE STATISTICAL AGGREGATION
# ==============================================================================

def compute_ensemble_statistics(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes distribution aggregates (mean, variance, std, median, min, max, IQR, pooled)
    for each protocol pipeline across the entire realization ensemble.
    """
    steps = [
        "regular_tgp",
        "legacy_step3",
        "sacd_stage1",
        "sacd_stage2",
        "sacd_stage3_raw",
        "sacd_manifold",
        "sacd_optics",
    ]
    summary: Dict[str, Any] = {}

    def _calc_stats(series: pd.Series, decimals: int = 4) -> Dict[str, float]:
        s = series.dropna()
        if len(s) == 0:
            return {
                "mean": 0.0, "variance": 0.0, "std": 0.0, "median": 0.0,
                "iqr_25": 0.0, "iqr_75": 0.0, "min": 0.0, "max": 0.0,
            }
        return {
            "mean": round(float(s.mean()), decimals),
            "variance": round(float(s.var()), decimals),
            "std": round(float(s.std()), decimals),
            "median": round(float(s.median()), decimals),
            "iqr_25": round(float(s.quantile(0.25)), decimals),
            "iqr_75": round(float(s.quantile(0.75)), decimals),
            "min": round(float(s.min()), decimals),
            "max": round(float(s.max()), decimals),
        }

    for step in steps:
        tpr_col = f"{step}_pfaffian_passed_pct"
        fdr_col = f"{step}_fdr_percent"
        ppv_col = f"{step}_ppv_percent"
        f1_col = f"{step}_f1_score"
        isl_col = f"{step}_n_islands"
        max_isl_col = f"{step}_max_island"

        tp_conf_col = f"{step}_tp_mean_boundary_confinement"
        tp_sep_col = f"{step}_tp_mean_separability"
        pass_conf_col = f"{step}_mean_boundary_confinement"
        pass_sep_col = f"{step}_mean_separability"
        tgp_cov_col = f"{step}_tgp_tp_coverage_pct"

        tp_col = f"{step}_true_positives"
        fp_col = f"{step}_false_positives"
        tot_gt_col = f"{step}_total_pfaffian_area"
        tot_pass_col = f"{step}_tot_passed"
        mean_isl_col = f"{step}_mean_island_size"

        # Pooled rates across all pixels in the ensemble
        sum_tp = metrics_df[tp_col].sum() if tp_col in metrics_df.columns else 0
        sum_fp = metrics_df[fp_col].sum() if fp_col in metrics_df.columns else 0
        sum_gt = metrics_df[tot_gt_col].sum() if tot_gt_col in metrics_df.columns else 0
        pooled_fdr = round((sum_fp / (sum_tp + sum_fp) * 100.0) if (sum_tp + sum_fp) > 0 else 0.0, 4)
        pooled_tpr = round((sum_tp / sum_gt * 100.0) if sum_gt > 0 else 0.0, 4)

        tpr_dict = _calc_stats(metrics_df[tpr_col], decimals=2)
        tpr_dict["pooled"] = pooled_tpr

        fdr_dict = _calc_stats(metrics_df[fdr_col], decimals=2)
        fdr_dict["pooled"] = pooled_fdr

        summary[step] = {
            "tot_passed": _calc_stats(metrics_df[tot_pass_col], decimals=1) if tot_pass_col in metrics_df.columns else {},
            "pfaffian_passed_pct": tpr_dict,
            "fdr_percent": fdr_dict,
            "ppv_percent": _calc_stats(metrics_df[ppv_col], decimals=2),
            "f1_score": _calc_stats(metrics_df[f1_col], decimals=4),
            "n_islands": _calc_stats(metrics_df[isl_col], decimals=1),
            "max_island": _calc_stats(metrics_df[max_isl_col], decimals=1),
            "mean_island_size": _calc_stats(metrics_df[mean_isl_col], decimals=2) if mean_isl_col in metrics_df.columns else {},
            "tp_mean_boundary_confinement": _calc_stats(metrics_df[tp_conf_col], decimals=4),
            "tp_mean_separability": _calc_stats(metrics_df[tp_sep_col], decimals=4),
            "mean_boundary_confinement": _calc_stats(metrics_df[pass_conf_col], decimals=4),
            "mean_separability": _calc_stats(metrics_df[pass_sep_col], decimals=4),
            "tgp_tp_coverage_pct": _calc_stats(metrics_df[tgp_cov_col], decimals=2) if tgp_cov_col in metrics_df.columns else {},
        }

    # Include ground truth reference statistics
    if "gt_mean_boundary_confinement" in metrics_df.columns:
        summary["ground_truth_reference"] = {
            "gt_total_area": _calc_stats(metrics_df["gt_total_area"], decimals=1),
            "gt_mean_boundary_confinement": _calc_stats(metrics_df["gt_mean_boundary_confinement"], decimals=4),
            "gt_var_boundary_confinement": _calc_stats(metrics_df["gt_var_boundary_confinement"], decimals=4),
            "gt_mean_separability": _calc_stats(metrics_df["gt_mean_separability"], decimals=4),
            "gt_var_separability": _calc_stats(metrics_df["gt_var_separability"], decimals=4),
        }

    return summary


# ==============================================================================
# 5. PUBLICATION-QUALITY PLOTTING
# ==============================================================================

def plot_realization_phase_maps(res: Dict[str, Any], save_path: Path):
    """
    Generates a 6-panel diagnostic phase diagram for a single realization:
    1. Ground Truth Pfaffian Invariant (Q = -1)
    2. Microsoft Azure Quantum TGP Baseline (Stage 2 ROI 2)
    3. Unclustered S-ACD (Pointwise)
    4. Clustered S-ACD [Directional Manifold Regularization]
    5. Clustered S-ACD [OPTICS Density Clustering]
    6. Manifold Clustered Error Classification (TP, FP, FN, TN)
    """
    gt = res["gt_2d"]
    m_tgp = res["m_tgp"]
    m_s3_raw = res["m_sacd_s3_raw"]
    m_man = res["m_sacd_manifold"]
    m_opt = res["m_sacd_optics"]

    mu_var = res["mu_var"]
    Vz_var = res["Vz_var"]
    r_name = res["realization"]
    extent = [Vz_var[0], Vz_var[-1], mu_var[0], mu_var[-1]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)

    # 1. Ground Truth
    ax = axes[0, 0]
    ax.imshow(gt, origin="lower", extent=extent, aspect="auto", cmap="Blues", interpolation="nearest")
    gt_met = res.get("ground_truth_metrics", {})
    ax.set_title(
        f"Ground Truth Pfaffian Phase (Q = -1)\n"
        f"Area={int(np.sum(gt))} px | Conf={gt_met.get('gt_mean_boundary_confinement', 0):.2f} | "
        f"Sep={gt_met.get('gt_mean_separability', 0):.2f}",
        fontsize=10, fontweight="bold"
    )
    ax.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)

    # 2. Microsoft TGP
    ax = axes[0, 1]
    ax.imshow(m_tgp, origin="lower", extent=extent, aspect="auto", cmap="Oranges", interpolation="nearest")
    tgp_m = res["regular_tgp"]
    ax.set_title(
        f"Microsoft TGP Baseline\n"
        f"TPR={tgp_m['pfaffian_passed_pct']:.1f}% | FDR={tgp_m['fdr_percent']:.1f}% | "
        f"Isl={tgp_m['n_islands']} (Max={tgp_m['max_island']} px)",
        fontsize=10, fontweight="bold"
    )

    # 3. Unclustered S-ACD (Pointwise)
    ax = axes[0, 2]
    ax.imshow(m_s3_raw, origin="lower", extent=extent, aspect="auto", cmap="Purples", interpolation="nearest")
    s3_m = res["sacd_stage3_raw"]
    ax.set_title(
        f"Unclustered S-ACD (Pointwise)\n"
        f"TPR={s3_m['pfaffian_passed_pct']:.1f}% | FDR={s3_m['fdr_percent']:.1f}% | "
        f"Isl={s3_m['n_islands']} (Max={s3_m['max_island']} px)",
        fontsize=10, fontweight="bold"
    )

    # 4. Clustered S-ACD (Manifold)
    ax = axes[1, 0]
    ax.imshow(m_man, origin="lower", extent=extent, aspect="auto", cmap="Greens", interpolation="nearest")
    man_m = res["sacd_manifold"]
    ax.set_title(
        f"Clustered S-ACD [Directional Manifold]\n"
        f"TPR={man_m['pfaffian_passed_pct']:.1f}% | FDR={man_m['fdr_percent']:.1f}% | "
        f"Isl={man_m['n_islands']} (Max={man_m['max_island']} px)",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10)
    ax.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)

    # 5. Clustered S-ACD (OPTICS)
    ax = axes[1, 1]
    ax.imshow(m_opt, origin="lower", extent=extent, aspect="auto", cmap="YlGnBu", interpolation="nearest")
    opt_m = res["sacd_optics"]
    ax.set_title(
        f"Clustered S-ACD [OPTICS Clustering]\n"
        f"TPR={opt_m['pfaffian_passed_pct']:.1f}% | FDR={opt_m['fdr_percent']:.1f}% | "
        f"Isl={opt_m['n_islands']} (Max={opt_m['max_island']} px)",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10)

    # 6. Manifold Error Classification (TP, FP, FN, TN)
    ax = axes[1, 2]
    conf_map = np.zeros_like(gt, dtype=int)
    conf_map[(~m_man) & gt] = 1   # FN (Blue)
    conf_map[m_man & (~gt)] = 2   # FP (Red)
    conf_map[m_man & gt] = 3      # TP (Green)

    cmap_conf = mcolors.ListedColormap(["#f0f0f0", "#1f77b4", "#d62728", "#2ca02c"])
    ax.imshow(conf_map, origin="lower", extent=extent, aspect="auto", cmap=cmap_conf, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(
        f"Manifold Error Classification\n"
        f"TP={man_m['true_positives']} (Green) | FP={man_m['false_positives']} (Red) | "
        f"TP Conf={man_m['tp_mean_boundary_confinement']:.2f} | TP Sep={man_m['tp_mean_separability']:.2f}",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10)

    plt.suptitle(f"Realization Diagnostics: {r_name} ($V_0 = {res['V0']}$ meV)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_ensemble_figures(metrics_df: pd.DataFrame, figures_dir: Path):
    """
    Generates summary ensemble plots across all protocol steps:
      1. FDR Distribution Boxplot
      2. TPR Coverage Boxplot
      3. TPR vs FDR Trade-off Scatter with Pareto Frontier
      4. Island Contiguity Boxplot (Total Islands & Max Island Size)
      5. True Positive Boundary Confinement Boxplot
      6. True Positive MZM Separability Boxplot
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    steps = ["regular_tgp", "sacd_stage3_raw", "sacd_manifold", "sacd_optics"]
    labels = ["Microsoft TGP", "Raw S-ACD (Pointwise)", "S-ACD (Manifold)", "S-ACD (OPTICS)"]
    colors = ["#ff7f0e", "#9467bd", "#2ca02c", "#17becf"]

    # 1. FDR Distribution Boxplot
    fig, ax = plt.subplots(figsize=(9, 5))
    fdr_data = [metrics_df[f"{s}_fdr_percent"].dropna().values for s in steps]
    bp = ax.boxplot(fdr_data, patch_artist=True, labels=labels, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.axhline(20.0, color="crimson", linestyle="--", linewidth=1.5, label="Target Ceiling (FDR = 20%)")
    ax.set_ylabel("False Discovery Rate (%)", fontsize=11)
    ax.set_title("FDR Distribution Across Ensemble (Target < 20%)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "fdr_distribution_boxplot.png", dpi=200)
    plt.close(fig)

    # 2. TPR Distribution Boxplot
    fig, ax = plt.subplots(figsize=(9, 5))
    tpr_data = [metrics_df[f"{s}_pfaffian_passed_pct"].dropna().values for s in steps]
    bp = ax.boxplot(tpr_data, patch_artist=True, labels=labels, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_ylabel("True Positive Rate (Recall %)", fontsize=11)
    ax.set_title("Pfaffian Topological Area Coverage (TPR %)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "tpr_distribution_boxplot.png", dpi=200)
    plt.close(fig)

    # 3. TPR vs FDR Scatter with Pareto Frontier
    fig, ax = plt.subplots(figsize=(9, 6))
    for s, l, c in zip(steps, labels, colors):
        ax.scatter(
            metrics_df[f"{s}_fdr_percent"],
            metrics_df[f"{s}_pfaffian_passed_pct"],
            alpha=0.6,
            label=l,
            color=c,
            edgecolors="none",
            s=45
        )
        # Mean point
        ax.scatter(
            metrics_df[f"{s}_fdr_percent"].mean(),
            metrics_df[f"{s}_pfaffian_passed_pct"].mean(),
            color=c,
            edgecolors="black",
            s=130,
            marker="D"
        )
    ax.axvline(20.0, color="crimson", linestyle="--", alpha=0.7, label="FDR = 20% Threshold")
    ax.set_xlabel("False Discovery Rate (%)", fontsize=11)
    ax.set_ylabel("True Positive Rate (%)", fontsize=11)
    ax.set_title("TPR vs. FDR Trade-Off (Diamonds = Ensemble Means)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "tpr_vs_fdr_scatter.png", dpi=200)
    plt.close(fig)

    # 4. Island Contiguity Boxplot (Dual Panel: Max Island Size & Island Count)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    max_isl_data = [metrics_df[f"{s}_max_island"].dropna().values for s in steps]
    bp1 = ax1.boxplot(max_isl_data, patch_artist=True, labels=labels, showmeans=True)
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax1.set_ylabel("Primary Island Size (pixels)", fontsize=11)
    ax1.set_title("Primary Contiguous Island Size", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    n_isl_data = [metrics_df[f"{s}_n_islands"].dropna().values for s in steps]
    bp2 = ax2.boxplot(n_isl_data, patch_artist=True, labels=labels, showmeans=True)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax2.set_ylabel("Total Number of Islands", fontsize=11)
    ax2.set_title("Topological Island Count (Fragmentation)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(figures_dir / "island_contiguity_boxplot.png", dpi=200)
    plt.close(fig)

    # 5. True Positive Boundary Confinement Boxplot
    fig, ax = plt.subplots(figsize=(9, 5))
    conf_data = [metrics_df[f"{s}_tp_mean_boundary_confinement"].dropna().values for s in steps]
    bp = ax.boxplot(conf_data, patch_artist=True, labels=labels, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    if "gt_mean_boundary_confinement" in metrics_df.columns:
        gt_mean_c = metrics_df["gt_mean_boundary_confinement"].mean()
        ax.axhline(gt_mean_c, color="blue", linestyle=":", linewidth=1.5, label=f"Ground Truth Average ({gt_mean_c:.3f})")
        ax.legend(loc="lower left")
    ax.set_ylabel("True Positive Boundary Confinement (0 to 1)", fontsize=11)
    ax.set_title("Boundary Confinement on True Positives", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "boundary_confinement_boxplot.png", dpi=200)
    plt.close(fig)

    # 6. True Positive MZM Separability Boxplot
    fig, ax = plt.subplots(figsize=(9, 5))
    sep_data = [metrics_df[f"{s}_tp_mean_separability"].dropna().values for s in steps]
    bp = ax.boxplot(sep_data, patch_artist=True, labels=labels, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    if "gt_mean_separability" in metrics_df.columns:
        gt_mean_s = metrics_df["gt_mean_separability"].mean()
        ax.axhline(gt_mean_s, color="blue", linestyle=":", linewidth=1.5, label=f"Ground Truth Average ({gt_mean_s:.3f})")
        ax.legend(loc="lower left")
    ax.set_ylabel("True Positive MZM Separability (0.5 to 1.0)", fontsize=11)
    ax.set_title("Majorana Spatial Separability on True Positives", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "separability_boxplot.png", dpi=200)
    plt.close(fig)


# ==============================================================================
# 6. ENSEMBLE RUNNER FOR SINGLE DISORDER & TEMPERATURE
# ==============================================================================

def run_single_disorder_temperature(
    target_dir: Path,
    output_dir: Path,
    temp_mK: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Executes ensemble post-processing on a single disorder strength at a specified temperature.
    Saves metrics CSVs, JSONs, diagnostic figures, and best/worst bundles into output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    b_val = float(temp_mK)
    t_str = f"{int(b_val)}mK" if b_val == int(b_val) else f"{b_val:.1f}mK"
    temp_suffix = f"_{t_str}"

    base_required = [
        "all_params.npz",
        "pdi_data.npy",
        "tgp_stage1_dIdVl.npy",
        "barrier_arr.npy",
    ]
    barrier_names = [
        "barrier_left_conductance_left_arr",
        "barrier_left_conductance_right_arr",
        "barrier_right_conductance_left_arr",
        "barrier_right_conductance_right_arr",
    ]

    all_subs = sorted([p for p in target_dir.iterdir() if p.is_dir()])
    valid_folders = []
    skipped_folders = []

    for sub in all_subs:
        missing = [f for f in base_required if not (sub / f).exists()]
        for b_name in barrier_names:
            has_suffixed = (sub / f"{b_name}{temp_suffix}.npy").exists()
            has_base = (sub / f"{b_name}.npy").exists()
            if not (has_suffixed or has_base):
                missing.append(f"{b_name}{temp_suffix}.npy")
        if not missing:
            valid_folders.append(sub)
        else:
            skipped_folders.append((sub.name, missing))

    logger.info(f"[{target_dir.name} @ {temp_mK} mK] Discovered {len(valid_folders)} valid realization folders.")
    if skipped_folders:
        logger.warning(f"Skipped {len(skipped_folders)} incomplete folders: {[s[0] for s in skipped_folders]}")

    if not valid_folders:
        logger.error(f"No valid realization directories found in {target_dir} for {temp_mK} mK.")
        return {}

    if args.max_realizations and args.max_realizations > 0:
        logger.info(f"Limiting to first {args.max_realizations} realizations per --max_realizations.")
        valid_folders = valid_folders[:args.max_realizations]

    workers = args.num_workers or min(32, cpu_count() or 4)
    logger.info(f"Processing {len(valid_folders)} realizations using {workers} parallel workers...")

    config_dict = {
        "protocol_mode": getattr(args, "protocol_mode", "strict_and"),
        "gap_th_factor": args.gap_th_factor,
        "corr_threshold": args.corr_threshold,
        "barrier_T_mK": temp_mK,
        "tau": args.tau,
        "s_max": args.s_max,
        "d2_threshold": args.d2_threshold,
        "min_cutters": args.min_cutters,
        "min_gap_uev": args.min_gap_uev,
        "min_island_size": args.min_island_size,
        "optics_min_samples": args.optics_min_samples,
        "optics_max_eps": args.optics_max_eps,
        "optics_min_cluster_size": args.optics_min_cluster_size,
        "optics_xi": args.optics_xi,
    }
    work_items = [(str(p), config_dict) for p in valid_folders]

    t_start = time.perf_counter()
    if workers > 1:
        with Pool(processes=workers) as pool:
            results = pool.map(process_single_realization, work_items)
    else:
        results = [process_single_realization(item) for item in work_items]
    t_end = time.perf_counter()

    logger.info(f"[{target_dir.name} @ {temp_mK} mK] Processed {len(valid_folders)} realizations in {t_end - t_start:.2f} s.")

    successful_results = [r for r in results if r.get("success")]
    failed_results = [r for r in results if not r.get("success")]

    if not successful_results:
        logger.error(f"All realizations in {target_dir} failed processing at {temp_mK} mK.")
        return {}

    all_steps = [
        "regular_tgp",
        "legacy_step3",
        "sacd_stage1",
        "sacd_stage2",
        "sacd_stage3_raw",
        "sacd_manifold",
        "sacd_optics",
    ]

    rows = []
    for r in successful_results:
        row = {
            "realization": r["realization"],
            "V0": r["V0"],
            "Nmu": r["Nmu"],
            "Nvz": r["Nvz"],
            "elapsed_seconds": r["elapsed_seconds"],
        }
        for k, v in r.get("ground_truth_metrics", {}).items():
            row[k] = v
        for step in all_steps:
            met = r[step]
            for k, v in met.items():
                row[f"{step}_{k}"] = v
        rows.append(row)

    metrics_df = pd.DataFrame(rows)

    def natural_sort_key(s):
        match = re.search(r"\d+", s)
        return int(match.group()) if match else s
    metrics_df["sort_id"] = metrics_df["realization"].apply(natural_sort_key)
    metrics_df = metrics_df.sort_values(by="sort_id").drop(columns=["sort_id"]).reset_index(drop=True)

    csv_path = output_dir / "realization_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Saved realization metrics to {csv_path}")

    ensemble_stats = compute_ensemble_statistics(metrics_df)
    ensemble_stats["metadata"] = {
        "target_dir": str(target_dir),
        "total_realizations_evaluated": len(successful_results),
        "failed_realizations": len(failed_results),
        "disorder_strength_V0": float(successful_results[0]["V0"]),
        "temperature_mK": float(temp_mK),
        "total_runtime_seconds": round(t_end - t_start, 2),
    }

    json_path = output_dir / "ensemble_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ensemble_stats, f, indent=2)
    logger.info(f"Saved ensemble summary JSON to {json_path}")

    summary_rows = []
    step_display_names = {
        "regular_tgp": "Microsoft Azure Quantum TGP Baseline",
        "legacy_step3": "Legacy Reflection Correlation Protocol",
        "sacd_stage1": "S-ACD Stage 1 (Asymmetry + Lead Symmetry)",
        "sacd_stage2": "S-ACD Stage 2 (Multi-Cutter 3w Curvature)",
        "sacd_stage3_raw": "Unclustered S-ACD (Pointwise Intersection)",
        "sacd_manifold": "Clustered S-ACD [Subtractive Island Filter]",
        "sacd_optics": "Clustered S-ACD [OPTICS Density Clustering]",
    }

    for step_key, step_label in step_display_names.items():
        st = ensemble_stats[step_key]
        summary_rows.append({
            "Protocol Pipeline": step_label,
            "Mean Passed Area (px)": st["tot_passed"]["mean"] if "tot_passed" in st else 0.0,
            "Mean TPR (%)": st["pfaffian_passed_pct"]["mean"],
            "Median TPR (%)": st["pfaffian_passed_pct"]["median"],
            "Pooled TPR (%)": st["pfaffian_passed_pct"]["pooled"],
            "Mean FDR (%)": st["fdr_percent"]["mean"],
            "Median FDR (%)": st["fdr_percent"]["median"],
            "Pooled FDR (%)": st["fdr_percent"]["pooled"],
            "Mean Islands": st["n_islands"]["mean"],
            "Mean Cluster Area (px)": st["mean_island_size"]["mean"] if "mean_island_size" in st else 0.0,
            "Mean Max Island (px)": st["max_island"]["mean"],
            "TP Mean Confinement": st["tp_mean_boundary_confinement"]["mean"],
            "TP Mean Separability": st["tp_mean_separability"]["mean"],
            "Mean F1 Score": st["f1_score"]["mean"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "ensemble_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved readable ensemble summary table to {summary_csv_path}")

    plot_ensemble_figures(metrics_df, figures_dir)
    logger.info(f"Generated ensemble diagnostic figures in {figures_dir}")

    # Robust ranking of best and worst
    valid_mask = (metrics_df["sacd_manifold_tot_passed"] > 0)
    if args.ranking_metric in ["manifold_fdr", "fdr"]:
        if valid_mask.any():
            valid_df = metrics_df[valid_mask].sort_values(
                by=["sacd_manifold_fdr_percent", "sacd_manifold_tot_passed", "sacd_manifold_pfaffian_passed_pct"],
                ascending=[True, False, False]
            )
            best_name = valid_df.iloc[0]["realization"]
            invalid_df = metrics_df[~valid_mask]
            if len(invalid_df) > 0:
                worst_name = invalid_df.iloc[0]["realization"]
            else:
                worst_name = valid_df.iloc[-1]["realization"]
        else:
            best_name = metrics_df.iloc[0]["realization"]
            worst_name = metrics_df.iloc[-1]["realization"]
    elif args.ranking_metric in ["manifold_tpr", "tpr"]:
        ranked_df = metrics_df.sort_values(
            by=["sacd_manifold_pfaffian_passed_pct", "sacd_manifold_fdr_percent"],
            ascending=[False, True]
        )
        best_name = ranked_df.iloc[0]["realization"]
        worst_name = ranked_df.iloc[-1]["realization"]
    else:  # f1
        ranked_df = metrics_df.sort_values(
            by=["sacd_manifold_f1_score", "sacd_manifold_fdr_percent"],
            ascending=[False, True]
        )
        best_name = ranked_df.iloc[0]["realization"]
        worst_name = ranked_df.iloc[-1]["realization"]

    best_res = next(r for r in successful_results if r["realization"] == best_name)
    worst_res = next(r for r in successful_results if r["realization"] == worst_name)

    logger.info(f"[{target_dir.name} @ {temp_mK} mK] Best: {best_name} (FDR={best_res['sacd_manifold']['fdr_percent']}%, TPR={best_res['sacd_manifold']['pfaffian_passed_pct']}%)")
    logger.info(f"[{target_dir.name} @ {temp_mK} mK] Worst: {worst_name} (FDR={worst_res['sacd_manifold']['fdr_percent']}%, TPR={worst_res['sacd_manifold']['pfaffian_passed_pct']}%)")

    best_and_worst = {
        "best_performing": {
            "realization": best_name,
            "ground_truth_metrics": best_res.get("ground_truth_metrics", {}),
            "metrics": {s: best_res[s] for s in all_steps}
        },
        "worst_performing": {
            "realization": worst_name,
            "ground_truth_metrics": worst_res.get("ground_truth_metrics", {}),
            "metrics": {s: worst_res[s] for s in all_steps}
        }
    }

    bw_json_path = output_dir / "best_and_worst_summary.json"
    with open(bw_json_path, "w", encoding="utf-8") as f:
        json.dump(best_and_worst, f, indent=2)

    for category, res_obj in [("best_performing", best_res), ("worst_performing", worst_res)]:
        cat_dir = output_dir / f"{category}_{res_obj['realization']}"
        cat_dir.mkdir(parents=True, exist_ok=True)

        cat_json = cat_dir / "performance_metrics.json"
        with open(cat_json, "w", encoding="utf-8") as f:
            json.dump({
                "realization": res_obj["realization"],
                "V0": res_obj["V0"],
                "ground_truth_metrics": res_obj.get("ground_truth_metrics", {}),
                "metrics": {s: res_obj[s] for s in all_steps}
            }, f, indent=2)

        cat_png = cat_dir / "diagnostic_phase_maps.png"
        plot_realization_phase_maps(res_obj, cat_png)

        src_data_dir = Path(res_obj["realization_dir"])
        dest_data_link = cat_dir / "data"

        if dest_data_link.exists() or dest_data_link.is_symlink():
            if dest_data_link.is_symlink():
                dest_data_link.unlink()
            elif dest_data_link.is_dir():
                shutil.rmtree(dest_data_link)

        if args.copy_data:
            shutil.copytree(src_data_dir, dest_data_link)
        else:
            try:
                dest_data_link.symlink_to(src_data_dir.resolve(), target_is_directory=True)
            except Exception as e:
                shutil.copytree(src_data_dir, dest_data_link)

    return {
        "disorder_folder": target_dir.name,
        "disorder_strength_V0": float(successful_results[0]["V0"]),
        "temperature_mK": float(temp_mK),
        "total_realizations": len(successful_results),
        # Best Realization
        "best_realization": best_name,
        "best_passed_area": best_res["sacd_manifold"]["tot_passed"],
        "best_fdr": best_res["sacd_manifold"]["fdr_percent"],
        "best_tpr": best_res["sacd_manifold"]["pfaffian_passed_pct"],
        "best_confinement": best_res["sacd_manifold"]["tp_mean_boundary_confinement"],
        "best_separability": best_res["sacd_manifold"]["tp_mean_separability"],
        "best_n_clusters": best_res["sacd_manifold"]["n_islands"],
        "best_cluster_area": best_res["sacd_manifold"]["mean_island_size"],
        # Worst Realization
        "worst_realization": worst_name,
        "worst_passed_area": worst_res["sacd_manifold"]["tot_passed"],
        "worst_fdr": worst_res["sacd_manifold"]["fdr_percent"],
        "worst_tpr": worst_res["sacd_manifold"]["pfaffian_passed_pct"],
        "worst_confinement": worst_res["sacd_manifold"]["tp_mean_boundary_confinement"],
        "worst_separability": worst_res["sacd_manifold"]["tp_mean_separability"],
        "worst_n_clusters": worst_res["sacd_manifold"]["n_islands"],
        "worst_cluster_area": worst_res["sacd_manifold"]["mean_island_size"],
        # Ensemble Average
        "avg_passed_area": ensemble_stats["sacd_manifold"]["tot_passed"]["mean"],
        "fdr_mean": ensemble_stats["sacd_manifold"]["fdr_percent"]["mean"],
        "fdr_pooled": ensemble_stats["sacd_manifold"]["fdr_percent"]["pooled"],
        "tpr_mean": ensemble_stats["sacd_manifold"]["pfaffian_passed_pct"]["mean"],
        "tp_confinement_mean": ensemble_stats["sacd_manifold"]["tp_mean_boundary_confinement"]["mean"],
        "tp_separability_mean": ensemble_stats["sacd_manifold"]["tp_mean_separability"]["mean"],
        "avg_n_clusters": ensemble_stats["sacd_manifold"]["n_islands"]["mean"],
        "avg_cluster_area": ensemble_stats["sacd_manifold"]["mean_island_size"]["mean"],
        "f1_mean": ensemble_stats["sacd_manifold"]["f1_score"]["mean"],
        "tgp_tp_coverage_mean": ensemble_stats["sacd_manifold"]["tgp_tp_coverage_pct"].get("mean", 0.0) if "tgp_tp_coverage_pct" in ensemble_stats["sacd_manifold"] else 0.0,
        # Microsoft Baseline reference
        "tgp_avg_passed_area": ensemble_stats["regular_tgp"]["tot_passed"]["mean"],
        "tgp_fdr_mean": ensemble_stats["regular_tgp"]["fdr_percent"]["mean"],
        "tgp_tpr_mean": ensemble_stats["regular_tgp"]["pfaffian_passed_pct"]["mean"],
        "tgp_tp_confinement": ensemble_stats["regular_tgp"]["tp_mean_boundary_confinement"]["mean"],
        "tgp_tp_separability": ensemble_stats["regular_tgp"]["tp_mean_separability"]["mean"],
        "tgp_avg_n_clusters": ensemble_stats["regular_tgp"]["n_islands"]["mean"],
        "tgp_avg_cluster_area": ensemble_stats["regular_tgp"]["mean_island_size"]["mean"],
        # Output paths
        "output_dir": str(output_dir),
        "ensemble_summary_csv": str(summary_csv_path),
        "ensemble_summary_json": str(json_path),
    }


# ==============================================================================
# 7. MASTER DISCOVERY & BATCH RUNNER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-process single-disorder realizations for FDR, TPR, contiguity, and MZM localization."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Path to directory containing realization subfolders, or parent directory of disorder strengths."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to output destination directory. Defaults to Outputs/Postprocessing."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of multiprocessing workers (defaults to min(32, cpu_count()))."
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help="List of temperatures in mK to evaluate (default: [30.0, 40.0])."
    )
    parser.add_argument(
        "--barrier_T_mK",
        type=float,
        default=40.0,
        help="Target temperature in mK if --temperatures is not specified (default: 40.0)."
    )
    parser.add_argument(
        "--gap_th_factor",
        type=float,
        default=0.05,
        help="Dynamic gap threshold factor for tgp.two.extract_gap (default: 0.05)."
    )
    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=0.85,
        help="Reflection correlation cutoff for Legacy Step 1 (default: 0.85)."
    )
    parser.add_argument(
        "--protocol_mode",
        type=str,
        default="strict_and",
        choices=["strict_and", "or_gate", "curv_only", "gap_only"],
        help="S-ACD logic combination: strict_and (S1 & Curv & Gap), or_gate (S1 & (Curv | Gap)), curv_only (S1 & Curv), gap_only (S1 & Gap)."
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.01,
        help="S-ACD asymmetric deviation cutoff sqrt(aL * aR) >= tau (default: 0.01)."
    )
    parser.add_argument(
        "--s_max",
        type=float,
        default=0.20,
        help="S-ACD lead symmetry balance threshold S_sym <= s_max (default: 0.20)."
    )
    parser.add_argument(
        "--d2_threshold",
        type=float,
        default=100.0,
        help="S-ACD zero-bias 3w negative curvature threshold in (e^2/h)/meV^2 (default: 100.0)."
    )
    parser.add_argument(
        "--min_cutters",
        type=int,
        default=3,
        help="Minimum number of passing cutters for S-ACD consensus (default: 3)."
    )
    parser.add_argument(
        "--min_gap_uev",
        type=float,
        default=10.0,
        help="Minimum non-local transport gap threshold in ueV (default: 10.0)."
    )
    parser.add_argument(
        "--min_island_size",
        type=int,
        default=10,
        help="Minimum island size in pixels to retain after clustering (default: 10)."
    )
    parser.add_argument(
        "--optics_min_samples",
        type=int,
        default=5,
        help="OPTICS min_samples neighborhood parameter (default: 5)."
    )
    parser.add_argument(
        "--optics_max_eps",
        type=float,
        default=3.0,
        help="OPTICS max_eps distance parameter (default: 3.0)."
    )
    parser.add_argument(
        "--optics_min_cluster_size",
        type=int,
        default=10,
        help="OPTICS min_cluster_size parameter (default: 10)."
    )
    parser.add_argument(
        "--optics_xi",
        type=float,
        default=0.05,
        help="OPTICS xi reachability steepness parameter (default: 0.05)."
    )
    parser.add_argument(
        "--ranking_metric",
        type=str,
        default="fdr",
        choices=["fdr", "tpr", "manifold_fdr", "manifold_tpr", "f1"],
        help="Metric used to rank best and worst realizations (default: fdr)."
    )
    parser.add_argument(
        "--max_realizations",
        type=int,
        default=None,
        help="Optional maximum number of realizations to process (useful for rapid dry-runs)."
    )
    parser.add_argument(
        "--copy_data",
        action="store_true",
        help="If set, copies the full data directory for best/worst realizations instead of creating symlinks."
    )

    args = parser.parse_args()

    target_path = Path(args.target_dir).resolve()
    if not target_path.exists():
        logger.error(f"Target directory does not exist: {target_path}")
        sys.exit(1)

    # Base output root
    if args.output_dir:
        base_output_dir = Path(args.output_dir).resolve()
    else:
        base_output_dir = REPO_ROOT / "Outputs" / "Postprocessing"

    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine whether target_path is a parent directory of disorders or a single disorder directory
    sub_disorders = [
        d for d in sorted(target_path.iterdir())
        if d.is_dir() and any((r / "all_params.npz").exists() for r in d.iterdir() if r.is_dir())
    ]

    if sub_disorders:
        logger.info(f"Target directory is a multi-disorder parent containing: {[d.name for d in sub_disorders]}")
        disorder_dirs = sub_disorders
    else:
        logger.info(f"Target directory is a single disorder folder: {target_path.name}")
        disorder_dirs = [target_path]

    # Temperatures to evaluate
    if args.temperatures is not None:
        eval_temperatures = args.temperatures
    else:
        eval_temperatures = [30.0, 40.0]

    logger.info(f"Evaluating temperatures: {eval_temperatures} mK across {len(disorder_dirs)} disorder folders.")

    all_ensemble_results: List[Dict[str, Any]] = []

    for d_dir in disorder_dirs:
        d_name = d_dir.name
        d_out = base_output_dir / d_name
        d_out.mkdir(parents=True, exist_ok=True)

        disorder_results = []
        for temp_mK in eval_temperatures:
            temp_str = f"{int(temp_mK)}mK" if temp_mK == int(temp_mK) else f"{temp_mK:.1f}mK"
            temp_out = d_out / temp_str
            res = run_single_disorder_temperature(d_dir, temp_out, temp_mK, args)
            if res:
                disorder_results.append(res)
                all_ensemble_results.append(res)

        # Generate comparative summary between temperatures for this disorder strength
        if disorder_results:
            disorder_comp_rows = []
            for r in disorder_results:
                disorder_comp_rows.append({
                    "Disorder Strength V0 (meV)": r["disorder_strength_V0"],
                    "Temperature (mK)": r["temperature_mK"],
                    "Realizations Evaluated": r["total_realizations"],
                    "Best Realization": f"{r['best_realization']} (FDR={r['best_fdr']}%, TPR={r['best_tpr']}%)",
                    "Worst Realization": f"{r['worst_realization']} (FDR={r['worst_fdr']}%, TPR={r['worst_tpr']}%)",
                    "Avg Passed Area (px)": r["avg_passed_area"],
                    "Mean FDR (%)": r["fdr_mean"],
                    "Pooled FDR (%)": r["fdr_pooled"],
                    "Mean TPR (%)": r["tpr_mean"],
                    "TP Mean Confinement": r["tp_confinement_mean"],
                    "TP Mean Separability": r["tp_separability_mean"],
                    "Avg Clusters (N)": r["avg_n_clusters"],
                    "Avg Cluster Area (px)": r["avg_cluster_area"],
                    "Mean F1 Score": r["f1_mean"],
                    "TGP TP Coverage (%)": r.get("tgp_tp_coverage_mean", 0.0),
                })

            d_comp_df = pd.DataFrame(disorder_comp_rows)
            d_comp_csv = d_out / "disorder_summary_30mK_vs_40mK.csv"
            d_comp_df.to_csv(d_comp_csv, index=False)
            logger.info(f"Saved disorder temperature comparison table to {d_comp_csv}")

            d_comp_json = d_out / "disorder_summary_30mK_vs_40mK.json"
            with open(d_comp_json, "w", encoding="utf-8") as f:
                json.dump(disorder_results, f, indent=2)

    # Global master table across all disorder strengths and temperatures
    if all_ensemble_results:
        master_rows = []
        for r in all_ensemble_results:
            master_rows.append({
                "Disorder Folder": r["disorder_folder"],
                "Disorder V0 (meV)": r["disorder_strength_V0"],
                "Temperature (mK)": r["temperature_mK"],
                "Realizations": r["total_realizations"],
                "Best Realization": r["best_realization"],
                "Best FDR (%)": r["best_fdr"],
                "Best TPR (%)": r["best_tpr"],
                "Best Passed (px)": r["best_passed_area"],
                "Worst Realization": r["worst_realization"],
                "Worst FDR (%)": r["worst_fdr"],
                "Worst TPR (%)": r["worst_tpr"],
                "Worst Passed (px)": r["worst_passed_area"],
                "Avg Passed Area (px)": r["avg_passed_area"],
                "Mean FDR (%)": r["fdr_mean"],
                "Pooled FDR (%)": r["fdr_pooled"],
                "Mean TPR (%)": r["tpr_mean"],
                "TP Boundary Confinement": r["tp_confinement_mean"],
                "TP Separability": r["tp_separability_mean"],
                "Avg Clusters (N)": r["avg_n_clusters"],
                "Avg Cluster Area (px)": r["avg_cluster_area"],
                "Mean F1 Score": r["f1_mean"],
                "TGP TP Coverage (%)": r.get("tgp_tp_coverage_mean", 0.0),
                # TGP baseline
                "TGP Avg Passed (px)": r["tgp_avg_passed_area"],
                "TGP Mean FDR (%)": r["tgp_fdr_mean"],
                "TGP Mean TPR (%)": r["tgp_tpr_mean"],
            })

        master_df = pd.DataFrame(master_rows)
        master_csv_path = base_output_dir / "all_disorders_master_summary.csv"
        master_df.to_csv(master_csv_path, index=False)
        logger.info(f"Saved master multi-disorder summary table to {master_csv_path}")

        master_json_path = base_output_dir / "all_disorders_master_summary.json"
        with open(master_json_path, "w", encoding="utf-8") as f:
            json.dump(all_ensemble_results, f, indent=2)

        print("\n" + "=" * 120)
        print("MASTER ENSEMBLE BENCHMARK SUMMARY (S-ACD SUBTRACTIVE PROTOCOL)")
        print("=" * 120)
        display_cols = [
            "Disorder V0 (meV)",
            "Temperature (mK)",
            "Best Realization",
            "Worst Realization",
            "Avg Passed Area (px)",
            "Mean FDR (%)",
            "TGP TP Coverage (%)",
            "TP Boundary Confinement",
            "TP Separability",
            "Avg Clusters (N)",
            "Avg Cluster Area (px)",
        ]
        print(master_df[display_cols].to_string(index=False))
        print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
