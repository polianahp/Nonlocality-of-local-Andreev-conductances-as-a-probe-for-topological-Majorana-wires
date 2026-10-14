#!/usr/bin/env python3
"""
post_process_disorder_realizations.py

High-performance single-disorder ensemble post-processing and benchmarking pipeline.
Discovers and processes disorder realizations belonging to a single disorder strength,
evaluates Microsoft TGP Stage 2 alongside the 3-step nonlocal protocol, computes
False Discovery Rate (FDR) and % Pfaffian topological area passed (% TPR), and
bundles the best and worst performing realizations with diagnostic phase diagrams.
"""

import os
import sys
import argparse
import time
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

    # Normalize each trace by its first element (if non-zero)
    denom1 = np.where(f1_m[:, 0:1] != 0, f1_m[:, 0:1], 1.0)
    denom2 = np.where(f2_m[:, 0:1] != 0, f2_m[:, 0:1], 1.0)
    f1_norm = f1_m / denom1
    f2_norm = f2_m / denom2

    # L2 norms along barrier dimension
    norm1 = np.linalg.norm(f1_norm, axis=1)
    norm2 = np.linalg.norm(f2_norm, axis=1)

    dot = np.sum(f1_norm * f2_norm, axis=1)
    denom = norm1 * norm2
    return np.where(denom > 0, dot / denom, 0.0)


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
    return np.zeros((Nmu, Nvz), dtype=float)


def calc_protocol_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    conf_2d: np.ndarray = None,
    sep_2d: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Calculates confusion matrix, performance rates, boundary confinement, and separability
    for a binary decision mask against the ground truth Pfaffian topological invariant.
    """
    pred_b = pred_mask.astype(bool)
    gt_b = gt_mask.astype(bool)

    tp = int(np.sum(pred_b & gt_b))
    fp = int(np.sum(pred_b & (~gt_b)))
    fn = int(np.sum((~pred_b) & gt_b))
    tn = int(np.sum((~pred_b) & (~gt_b)))

    tot_pass = tp + fp
    tot_topological = int(np.sum(gt_b))  # Total points that are topological

    # Pfaffian Area Passed: True Positives divided by Total Topological Points
    pfaffian_passed_pct = float((tp / tot_topological) * 100.0) if tot_topological > 0 else 0.0
    fdr_pct = float((fp / tot_pass) * 100.0) if tot_pass > 0 else 0.0
    ppv_pct = float((tp / tot_pass) * 100.0) if tot_pass > 0 else 0.0

    # F1 score balancing precision (PPV) and recall (TPR)
    precision = tp / tot_pass if tot_pass > 0 else 0.0
    recall = tp / tot_topological if tot_topological > 0 else 0.0
    if (precision + recall) > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # Boundary confinement and separability on passed region
    mean_conf = 0.0
    var_conf = 0.0
    mean_sep = 0.0
    var_sep = 0.0

    if tot_pass > 0:
        if conf_2d is not None:
            c_vals = conf_2d[pred_b]
            mean_conf = float(np.nanmean(c_vals))
            var_conf = float(np.nanvar(c_vals))
        if sep_2d is not None:
            s_vals = sep_2d[pred_b]
            mean_sep = float(np.nanmean(s_vals))
            var_sep = float(np.nanvar(s_vals))

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
        "mean_boundary_confinement": round(float(mean_conf), 4),
        "var_boundary_confinement": round(float(var_conf), 4),
        "mean_separability": round(float(mean_sep), 4),
        "var_separability": round(float(var_sep), 4),
    }


def process_single_realization(args_tuple: Tuple) -> Dict[str, Any]:
    """
    Worker function to process one realization directory.
    Executes TGP Stage 2, Steps 1-3, and returns metrics + masks.
    """
    realization_dir_str, config_dict = args_tuple
    realization_dir = Path(realization_dir_str)
    r_name = realization_dir.name

    t0 = time.perf_counter()
    gap_th_factor = config_dict.get("gap_th_factor", 0.05)
    corr_threshold = config_dict.get("corr_threshold", 0.85)

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

        # 3. Vectorized reflection correlation (Step 1)
        b_arr = np.load(realization_dir / "barrier_arr.npy")
        bl_l = np.load(realization_dir / "barrier_left_conductance_left_arr.npy")
        bl_r = np.load(realization_dir / "barrier_left_conductance_right_arr.npy")
        br_l = np.load(realization_dir / "barrier_right_conductance_left_arr.npy")
        br_r = np.load(realization_dir / "barrier_right_conductance_right_arr.npy")

        cL = vectorized_reflection_correlation(bl_l, bl_r, b_arr).reshape(Nmu, Nvz)
        cR = vectorized_reflection_correlation(br_l, br_r, b_arr).reshape(Nmu, Nvz)
        m1 = (cL >= corr_threshold) & (cR >= corr_threshold)

        # 4. TGP Stage 1 & 2 via TGPAdapter
        adapter = TGPAdapter(str(realization_dir))
        ds = adapter.to_xarray()
        tprep = prepare_sim_cpu(ds, T_mK=30.0)

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

        # Microsoft Stage 2 ROI 2
        s2 = tgp.two.cluster_and_score(
            z_ds,
            min_cluster_size=7,
            cluster_gap_threshold=10e-3,
            cluster_percentage_boundary_threshold=0.6,
            cluster_ncutter_threshold=0.5
        )
        roi2_vb = s2.roi2.values.T if s2.roi2.dims == ("B", "V") else s2.roi2.values
        m_tgp = (roi2_vb > 0)

        # Step 2: Correlation + 3w consensus (>= 50% cutters)
        zbp_2d = (z_ds["zbp"].mean(dim="cutter_pair_index") >= 0.5).transpose("V", "B").values
        m2 = m1 & zbp_2d

        # Step 3: Full ROI 3 (Correlation + 3w + Transport Gap >= 10 ueV)
        gzbp_2d = (z_ds["gapped_zbp"].mean(dim="cutter_pair_index") >= 0.5).transpose("V", "B").values
        m3 = m1 & gzbp_2d

        used_fallback = False
        if np.sum(m3) == 0:
            cand_single = (z_ds["gapped_zbp"].values[0] > 0).T
            m3 = m1 & cand_single
            used_fallback = True

        # Load boundary confinement and separability maps
        conf_2d = load_boundary_confinement(realization_dir, Nmu, Nvz, Ls=int(params.get("Ls", 150)))
        sep_2d = load_separability(realization_dir, Nmu, Nvz)

        # Ground truth topological reference properties
        gt_b = gt_2d.astype(bool)
        gt_metrics = {
            "gt_total_area": int(np.sum(gt_b)),
            "gt_mean_boundary_confinement": round(float(np.nanmean(conf_2d[gt_b])), 4) if np.sum(gt_b) > 0 else 0.0,
            "gt_var_boundary_confinement": round(float(np.nanvar(conf_2d[gt_b])), 4) if np.sum(gt_b) > 0 else 0.0,
            "gt_mean_separability": round(float(np.nanmean(sep_2d[gt_b])), 4) if np.sum(gt_b) > 0 else 0.0,
            "gt_var_separability": round(float(np.nanvar(sep_2d[gt_b])), 4) if np.sum(gt_b) > 0 else 0.0,
        }

        # Calculate metrics for each step
        metrics_tgp = calc_protocol_metrics(m_tgp, gt_2d, conf_2d, sep_2d)
        metrics_s1 = calc_protocol_metrics(m1, gt_2d, conf_2d, sep_2d)
        metrics_s2 = calc_protocol_metrics(m2, gt_2d, conf_2d, sep_2d)
        metrics_s3 = calc_protocol_metrics(m3, gt_2d, conf_2d, sep_2d)

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
            "regular_tgp": metrics_tgp,
            "step1_correlation": metrics_s1,
            "step2_corr_plus_3w": metrics_s2,
            "step3_full_roi3": metrics_s3,
            "used_fallback": used_fallback,
            # Store 2D arrays for best/worst visualization
            "mu_var": mu_var,
            "Vz_var": Vz_var,
            "gt_2d": gt_2d,
            "m_tgp": m_tgp,
            "m1": m1,
            "m2": m2,
            "m3": m3,
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


def compute_ensemble_statistics(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes distribution aggregates (mean, variance, std, median, min, max, IQR, pooled)
    for each protocol step across the ensemble.
    """
    steps = ["regular_tgp", "step1_correlation", "step2_corr_plus_3w", "step3_full_roi3"]
    summary: Dict[str, Any] = {}

    def _calc_stats(series: pd.Series, decimals: int = 4) -> Dict[str, float]:
        s = series.dropna()
        if len(s) == 0:
            return {
                "mean": 0.0,
                "variance": 0.0,
                "std": 0.0,
                "median": 0.0,
                "iqr_25": 0.0,
                "iqr_75": 0.0,
                "min": 0.0,
                "max": 0.0,
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
        conf_col = f"{step}_mean_boundary_confinement"
        conf_var_col = f"{step}_var_boundary_confinement"
        sep_col = f"{step}_mean_separability"
        sep_var_col = f"{step}_var_separability"

        tp_col = f"{step}_true_positives"
        fp_col = f"{step}_false_positives"
        tot_gt_col = f"{step}_total_pfaffian_area"

        # Pooled rates across all pixels
        sum_tp = metrics_df[tp_col].sum()
        sum_fp = metrics_df[fp_col].sum()
        sum_gt = metrics_df[tot_gt_col].sum()
        pooled_fdr = round((sum_fp / (sum_tp + sum_fp) * 100.0) if (sum_tp + sum_fp) > 0 else 0.0, 4)
        pooled_tpr = round((sum_tp / sum_gt * 100.0) if sum_gt > 0 else 0.0, 4)

        tpr_dict = _calc_stats(metrics_df[tpr_col], decimals=2)
        tpr_dict["pooled"] = pooled_tpr

        fdr_dict = _calc_stats(metrics_df[fdr_col], decimals=2)
        fdr_dict["pooled"] = pooled_fdr

        summary[step] = {
            "pfaffian_passed_pct": tpr_dict,
            "fdr_percent": fdr_dict,
            "ppv_percent": _calc_stats(metrics_df[ppv_col], decimals=2),
            "f1_score": _calc_stats(metrics_df[f1_col], decimals=4),
            "mean_boundary_confinement": _calc_stats(metrics_df[conf_col], decimals=4),
            "var_boundary_confinement": _calc_stats(metrics_df[conf_var_col], decimals=4),
            "mean_separability": _calc_stats(metrics_df[sep_col], decimals=4),
            "var_separability": _calc_stats(metrics_df[sep_var_col], decimals=4),
        }

    # Include ground truth reference statistics if available
    if "gt_mean_boundary_confinement" in metrics_df.columns:
        summary["ground_truth_reference"] = {
            "gt_total_area": _calc_stats(metrics_df["gt_total_area"], decimals=1),
            "gt_mean_boundary_confinement": _calc_stats(metrics_df["gt_mean_boundary_confinement"], decimals=4),
            "gt_var_boundary_confinement": _calc_stats(metrics_df["gt_var_boundary_confinement"], decimals=4),
            "gt_mean_separability": _calc_stats(metrics_df["gt_mean_separability"], decimals=4),
            "gt_var_separability": _calc_stats(metrics_df["gt_var_separability"], decimals=4),
        }

    return summary


def plot_realization_phase_maps(res: Dict[str, Any], save_path: Path):
    """
    Generates a 6-panel publication-quality diagnostic figure for a single realization
    showing Ground Truth Pfaffian, Step 1, TGP Stage 2, Step 3, Confusion Classification, and Overlap.
    """
    gt = res["gt_2d"]
    m1 = res["m1"]
    m_tgp = res["m_tgp"]
    m3 = res["m3"]
    mu_var = res["mu_var"]
    Vz_var = res["Vz_var"]
    r_name = res["realization"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)

    extent = [Vz_var[0], Vz_var[-1], mu_var[0], mu_var[-1]]

    # 1. Ground Truth
    ax = axes[0, 0]
    im0 = ax.imshow(gt, origin="lower", extent=extent, aspect="auto", cmap="Blues", interpolation="nearest")
    gt_met = res.get("ground_truth_metrics", {})
    gt_conf_str = f", Conf={gt_met['gt_mean_boundary_confinement']:.2f}" if "gt_mean_boundary_confinement" in gt_met else ""
    gt_sep_str = f", Sep={gt_met['gt_mean_separability']:.2f}" if "gt_mean_separability" in gt_met else ""
    ax.set_title(f"Ground Truth Pfaffian Invariant\n(Total Area = {int(np.sum(gt))} pts{gt_conf_str}{gt_sep_str})", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)

    # 2. Step 1 (Correlation)
    ax = axes[0, 1]
    ax.imshow(m1, origin="lower", extent=extent, aspect="auto", cmap="Purples", interpolation="nearest")
    s1_met = res["step1_correlation"]
    ax.set_title(fr"Step 1: Dual Correlation $\geq 0.85$" + f"\n(Passed={s1_met['tot_passed']}, FDR={s1_met['fdr_percent']}%, Conf={s1_met['mean_boundary_confinement']:.2f}, Sep={s1_met['mean_separability']:.2f})", fontsize=10, fontweight="bold")

    # 3. Regular TGP Stage 2
    ax = axes[0, 2]
    ax.imshow(m_tgp, origin="lower", extent=extent, aspect="auto", cmap="Oranges", interpolation="nearest")
    tgp_met = res["regular_tgp"]
    ax.set_title(f"Regular Microsoft TGP Stage 2\n(Passed={tgp_met['tot_passed']}, FDR={tgp_met['fdr_percent']}%, Conf={tgp_met['mean_boundary_confinement']:.2f}, Sep={tgp_met['mean_separability']:.2f})", fontsize=10, fontweight="bold")

    # 4. Step 3 (ROI 3)
    ax = axes[1, 0]
    ax.imshow(m3, origin="lower", extent=extent, aspect="auto", cmap="Greens", interpolation="nearest")
    s3_met = res["step3_full_roi3"]
    ax.set_title(f"Step 3: Full ROI 3 (Corr + $3\\omega$ + Gap)\n(Passed={s3_met['tot_passed']}, FDR={s3_met['fdr_percent']}%, Conf={s3_met['mean_boundary_confinement']:.2f}, Sep={s3_met['mean_separability']:.2f})", fontsize=10, fontweight="bold")
    ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10)
    ax.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)

    # 5. Step 3 Error Classification (TP, FP, FN, TN)
    ax = axes[1, 1]
    # Encode: TN=0 (white/lightgray), FN=1 (blue), FP=2 (red), TP=3 (green)
    conf_map = np.zeros_like(gt, dtype=int)
    conf_map[(~m3) & gt] = 1   # FN (Missed topological)
    conf_map[m3 & (~gt)] = 2   # FP (False discovery)
    conf_map[m3 & gt] = 3      # TP (Correct detection)

    cmap_conf = mcolors.ListedColormap(["#f0f0f0", "#1f77b4", "#d62728", "#2ca02c"])
    ax.imshow(conf_map, origin="lower", extent=extent, aspect="auto", cmap=cmap_conf, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(f"Step 3 Error Classification\n(TP={s3_met['true_positives']} (Green), FP={s3_met['false_positives']} (Red))", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10)

    # 6. Comparative Boundary Overlay
    ax = axes[1, 2]
    ax.imshow(gt, origin="lower", extent=extent, aspect="auto", cmap="Blues", alpha=0.3, interpolation="nearest")
    # Overlay contours of m_tgp (orange) and m3 (green)
    ax.contour(m_tgp, levels=[0.5], extent=extent, colors=["#ff7f0e"], linewidths=1.5, linestyles="dashed")
    ax.contour(m3, levels=[0.5], extent=extent, colors=["#2ca02c"], linewidths=2.0)
    ax.set_title(f"Overlay: TGP (Orange Dash) vs ROI 3 (Green)\nOver Ground Truth (Blue)", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10)

    plt.suptitle(f"Diagnostic Phase Maps: {r_name} ($V_0 = {res['V0']}$ meV)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_ensemble_figures(metrics_df: pd.DataFrame, figures_dir: Path):
    """
    Generates summary ensemble plots:
    1. FDR Distribution Boxplot
    2. Pfaffian Coverage Boxplot
    3. TPR vs FDR Scatter Trade-off
    4. Realization Score Ranking Bar
    5. Boundary Confinement Boxplot
    6. Separability Boxplot
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    steps = ["regular_tgp", "step1_correlation", "step2_corr_plus_3w", "step3_full_roi3"]
    labels = ["Regular TGP", "Step 1 (Corr)", "Step 2 (Corr+3w)", "Step 3 (ROI 3)"]
    colors = ["#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c"]

    # 1. FDR Distribution Boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    fdr_data = [metrics_df[f"{s}_fdr_percent"].dropna().values for s in steps]
    bp = ax.boxplot(fdr_data, patch_artist=True, labels=labels)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)
    ax.set_ylabel("False Discovery Rate (%)", fontsize=11)
    ax.set_title("FDR Distribution Across Protocol Steps (Single-Disorder Ensemble)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "fdr_distribution_boxplot.png", dpi=200)
    plt.close(fig)

    # 2. % Pfaffian Passed Coverage Boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    tpr_data = [metrics_df[f"{s}_pfaffian_passed_pct"].dropna().values for s in steps]
    bp = ax.boxplot(tpr_data, patch_artist=True, labels=labels)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)
    ax.set_ylabel("Pfaffian Topological Area Passed (%)", fontsize=11)
    ax.set_title("Topological Coverage Distribution Across Protocol Steps", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "pfaffian_coverage_boxplot.png", dpi=200)
    plt.close(fig)

    # 3. TPR vs FDR Scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    for s, lab, c in zip(steps, labels, colors):
        tpr = metrics_df[f"{s}_pfaffian_passed_pct"].values
        fdr = metrics_df[f"{s}_fdr_percent"].values
        ax.scatter(fdr, tpr, label=lab, color=c, alpha=0.8, s=60, edgecolors="none")
    ax.set_xlabel("False Discovery Rate FDR (%) [Lower is Better]", fontsize=11)
    ax.set_ylabel("Pfaffian Topological Area Passed (%) [Higher is Better]", fontsize=11)
    ax.set_title("Protocol Trade-Off: Topological Area Passed vs. FDR", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "tpr_vs_fdr_scatter.png", dpi=200)
    plt.close(fig)

    # 4. Realization Ranking Bar (Ranked by False Discovery Rate FDR, then Pfaffian Area Passed)
    fig, ax = plt.subplots(figsize=(10, max(5, len(metrics_df) * 0.35)))
    df_sorted = metrics_df.sort_values(
        by=["step3_full_roi3_fdr_percent", "step3_full_roi3_pfaffian_passed_pct"],
        ascending=[False, True]
    )
    y_pos = np.arange(len(df_sorted))
    fdr_vals = df_sorted["step3_full_roi3_fdr_percent"].values
    tpr_vals = df_sorted["step3_full_roi3_pfaffian_passed_pct"].values
    names = df_sorted["realization"].values
    bars = ax.barh(y_pos, fdr_vals, color="#d62728", alpha=0.75, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Step 3 (ROI 3) False Discovery Rate FDR (%) [Lower is Better]", fontsize=11)
    ax.set_title("Disorder Realizations Ranked by Step 3 FDR (Lower is Cleaner)", fontsize=12, fontweight="bold")
    for bar, fdr, tpr in zip(bars, fdr_vals, tpr_vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"FDR={fdr:.1f}%, Pfaffian Passed={tpr:.1f}%",
                va="center", ha="left", fontsize=8, color="#333333")
    max_fdr = np.nanmax(fdr_vals) if len(fdr_vals) > 0 and not np.all(np.isnan(fdr_vals)) else 10.0
    ax.set_xlim(0, max(15.0, max_fdr * 1.35))
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "realization_ranking_bar.png", dpi=200)
    plt.close(fig)

    # 5. Boundary Confinement Boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    conf_data = [metrics_df[f"{s}_mean_boundary_confinement"].dropna().values for s in steps]
    bp = ax.boxplot(conf_data, patch_artist=True, labels=labels)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)
    ax.set_ylabel("Mean Boundary Confinement (0 to 1)", fontsize=11)
    ax.set_title("MZM Boundary Confinement Across Protocol Steps", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "boundary_confinement_boxplot.png", dpi=200)
    plt.close(fig)

    # 6. Separability Boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    sep_data = [metrics_df[f"{s}_mean_separability"].dropna().values for s in steps]
    bp = ax.boxplot(sep_data, patch_artist=True, labels=labels)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)
    ax.set_ylabel("Mean Separability (0 to 1)", fontsize=11)
    ax.set_title("MZM Separability Across Protocol Steps", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "separability_boxplot.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Post-process single-disorder realizations for FDR and topological metrics."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Path to root directory containing realization subfolders for a single disorder strength."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to output destination directory. Defaults to Outputs/Postprocessing/<folder_name>."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of multiprocessing workers (defaults to min(8, cpu_count()))."
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
        help="Reflection correlation cutoff for Step 1 (default: 0.85)."
    )
    parser.add_argument(
        "--ranking_metric",
        type=str,
        default="fdr",
        choices=["fdr", "tpr", "f1"],
        help="Metric used to rank best and worst realizations (default: fdr, ranks by lowest FDR then highest Pfaffian area passed)."
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

    target_dir = Path(args.target_dir).resolve()
    if not target_dir.exists():
        logger.error(f"Target directory does not exist: {target_dir}")
        sys.exit(1)

    folder_name = target_dir.name
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = REPO_ROOT / "Outputs" / "Postprocessing" / folder_name

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover realization folders
    required_files = [
        "all_params.npz",
        "pdi_data.npy",
        "tgp_stage1_dIdVl.npy",
        "barrier_arr.npy",
        "barrier_left_conductance_left_arr.npy",
        "barrier_left_conductance_right_arr.npy",
        "barrier_right_conductance_left_arr.npy",
        "barrier_right_conductance_right_arr.npy",
    ]

    all_subs = sorted([p for p in target_dir.iterdir() if p.is_dir()])
    valid_folders = []
    skipped_folders = []

    for sub in all_subs:
        missing = [f for f in required_files if not (sub / f).exists()]
        if not missing:
            valid_folders.append(sub)
        else:
            skipped_folders.append((sub.name, missing))

    logger.info(f"Discovered {len(valid_folders)} valid realization folders in {target_dir}.")
    if skipped_folders:
        logger.warning(f"Skipped {len(skipped_folders)} incomplete folders: {[s[0] for s in skipped_folders]}")

    if not valid_folders:
        logger.error("No valid realization directories found. Exiting.")
        sys.exit(1)

    if args.max_realizations and args.max_realizations > 0:
        logger.info(f"Limiting to first {args.max_realizations} realizations per --max_realizations.")
        valid_folders = valid_folders[:args.max_realizations]

    # Multiprocessing configuration
    workers = args.num_workers or min(8, cpu_count() or 4)
    logger.info(f"Processing {len(valid_folders)} realizations using {workers} parallel workers...")

    config_dict = {
        "gap_th_factor": args.gap_th_factor,
        "corr_threshold": args.corr_threshold,
    }
    work_items = [(str(p), config_dict) for p in valid_folders]

    t_start = time.perf_counter()
    if workers > 1:
        with Pool(processes=workers) as pool:
            results = pool.map(process_single_realization, work_items)
    else:
        results = [process_single_realization(item) for item in work_items]
    t_end = time.perf_counter()

    logger.info(f"All {len(valid_folders)} realizations processed in {t_end - t_start:.2f} seconds.")

    # Flatten successful results into table
    successful_results = [r for r in results if r.get("success")]
    failed_results = [r for r in results if not r.get("success")]

    if not successful_results:
        logger.error("All realizations failed processing. Check errors.")
        sys.exit(1)

    rows = []
    for r in successful_results:
        row = {
            "realization": r["realization"],
            "V0": r["V0"],
            "Nmu": r["Nmu"],
            "Nvz": r["Nvz"],
            "elapsed_seconds": r["elapsed_seconds"],
            "used_fallback": r["used_fallback"],
        }
        for k, v in r.get("ground_truth_metrics", {}).items():
            row[k] = v
        for step in ["regular_tgp", "step1_correlation", "step2_corr_plus_3w", "step3_full_roi3"]:
            met = r[step]
            for k, v in met.items():
                row[f"{step}_{k}"] = v
        rows.append(row)

    metrics_df = pd.DataFrame(rows)

    # Sort table by realization name naturally (extract numeric id if available)
    import re
    def natural_sort_key(s):
        match = re.search(r"\d+", s)
        return int(match.group()) if match else s
    metrics_df["sort_id"] = metrics_df["realization"].apply(natural_sort_key)
    metrics_df = metrics_df.sort_values(by="sort_id").drop(columns=["sort_id"]).reset_index(drop=True)

    # Save realization metrics CSV
    csv_path = output_dir / "realization_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Saved realization metrics to {csv_path}")

    # Compute Ensemble Statistics
    ensemble_stats = compute_ensemble_statistics(metrics_df)
    ensemble_stats["metadata"] = {
        "target_dir": str(target_dir),
        "total_realizations_evaluated": len(successful_results),
        "failed_realizations": len(failed_results),
        "disorder_strength_V0": float(successful_results[0]["V0"]),
        "total_runtime_seconds": round(t_end - t_start, 2),
    }

    json_path = output_dir / "ensemble_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ensemble_stats, f, indent=2)
    logger.info(f"Saved ensemble summary JSON to {json_path}")

    # Create clean readable summary table CSV
    summary_rows = []
    step_names = {
        "regular_tgp": "Regular Microsoft TGP Stage 2",
        "step1_correlation": "Step 1 (Dual Correlation >= 0.85)",
        "step2_corr_plus_3w": "Step 2 (Corr + 3w Consensus)",
        "step3_full_roi3": "Step 3 / ROI 3 (Corr + 3w + Gap >= 10 ueV)"
    }
    for step_key, step_label in step_names.items():
        st = ensemble_stats[step_key]
        summary_rows.append({
            "Protocol Step": step_label,
            "Mean Pfaffian Passed (%)": st["pfaffian_passed_pct"]["mean"],
            "Variance Pfaffian Passed (%)": st["pfaffian_passed_pct"]["variance"],
            "Median Pfaffian Passed (%)": st["pfaffian_passed_pct"]["median"],
            "Pooled Pfaffian Passed (%)": st["pfaffian_passed_pct"]["pooled"],
            "Mean FDR (%)": st["fdr_percent"]["mean"],
            "Variance FDR (%)": st["fdr_percent"]["variance"],
            "Median FDR (%)": st["fdr_percent"]["median"],
            "Pooled FDR (%)": st["fdr_percent"]["pooled"],
            "Mean F1 Score": st["f1_score"]["mean"],
            "Variance F1 Score": st["f1_score"]["variance"],
            "Median F1 Score": st["f1_score"]["median"],
            "Mean Boundary Confinement": st["mean_boundary_confinement"]["mean"],
            "Variance Boundary Confinement": st["mean_boundary_confinement"]["variance"],
            "Median Boundary Confinement": st["mean_boundary_confinement"]["median"],
            "Mean Separability": st["mean_separability"]["mean"],
            "Variance Separability": st["mean_separability"]["variance"],
            "Median Separability": st["mean_separability"]["median"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "ensemble_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved readable ensemble summary table to {summary_csv_path}")

    # Generate Figures
    plot_ensemble_figures(metrics_df, figures_dir)
    logger.info(f"Generated ensemble diagnostic figures in {figures_dir}")

    # Identify Best and Worst Performing Realizations
    # Primary goal: Lowest False Discovery Rate (FDR) so passed predictions are reliably topological.
    # Secondary goal: Highest Pfaffian Area Passed (maximizing detected topological phase space).
    metrics_df["valid_passed"] = metrics_df["step3_full_roi3_tot_passed"] > 0

    if args.ranking_metric == "fdr":
        ranked_df = metrics_df.sort_values(
            by=["valid_passed", "step3_full_roi3_fdr_percent", "step3_full_roi3_pfaffian_passed_pct"],
            ascending=[False, True, False]
        )
    elif args.ranking_metric == "tpr":
        ranked_df = metrics_df.sort_values(
            by=["valid_passed", "step3_full_roi3_pfaffian_passed_pct", "step3_full_roi3_fdr_percent"],
            ascending=[False, False, True]
        )
    else:  # f1
        ranked_df = metrics_df.sort_values(
            by=["step3_full_roi3_f1_score", "step3_full_roi3_fdr_percent"],
            ascending=[False, True]
        )

    best_name = ranked_df.iloc[0]["realization"]
    worst_name = ranked_df.iloc[-1]["realization"]

    best_res = next(r for r in successful_results if r["realization"] == best_name)
    worst_res = next(r for r in successful_results if r["realization"] == worst_name)

    logger.info(f"Best Performing Realization: {best_name} (Step 3 FDR={best_res['step3_full_roi3']['fdr_percent']}%, Pfaffian Passed={best_res['step3_full_roi3']['pfaffian_passed_pct']}%)")
    logger.info(f"Worst Performing Realization: {worst_name} (Step 3 FDR={worst_res['step3_full_roi3']['fdr_percent']}%, Pfaffian Passed={worst_res['step3_full_roi3']['pfaffian_passed_pct']}%)")

    # Bundle Best and Worst
    best_and_worst = {
        "best_performing": {
            "realization": best_name,
            "ground_truth_metrics": best_res.get("ground_truth_metrics", {}),
            "metrics": {
                "regular_tgp": best_res["regular_tgp"],
                "step1_correlation": best_res["step1_correlation"],
                "step2_corr_plus_3w": best_res["step2_corr_plus_3w"],
                "step3_full_roi3": best_res["step3_full_roi3"],
            }
        },
        "worst_performing": {
            "realization": worst_name,
            "ground_truth_metrics": worst_res.get("ground_truth_metrics", {}),
            "metrics": {
                "regular_tgp": worst_res["regular_tgp"],
                "step1_correlation": worst_res["step1_correlation"],
                "step2_corr_plus_3w": worst_res["step2_corr_plus_3w"],
                "step3_full_roi3": worst_res["step3_full_roi3"],
            }
        }
    }

    bw_json_path = output_dir / "best_and_worst_summary.json"
    with open(bw_json_path, "w", encoding="utf-8") as f:
        json.dump(best_and_worst, f, indent=2)

    for category, res_obj in [("best_performing", best_res), ("worst_performing", worst_res)]:
        cat_dir = output_dir / f"{category}_{res_obj['realization']}"
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Performance metrics json
        cat_json = cat_dir / "performance_metrics.json"
        with open(cat_json, "w", encoding="utf-8") as f:
            json.dump({
                "realization": res_obj["realization"],
                "V0": res_obj["V0"],
                "ground_truth_metrics": res_obj.get("ground_truth_metrics", {}),
                "regular_tgp": res_obj["regular_tgp"],
                "step1_correlation": res_obj["step1_correlation"],
                "step2_corr_plus_3w": res_obj["step2_corr_plus_3w"],
                "step3_full_roi3": res_obj["step3_full_roi3"],
            }, f, indent=2)

        # Phase diagram plot
        cat_png = cat_dir / "diagnostic_phase_maps.png"
        plot_realization_phase_maps(res_obj, cat_png)
        logger.info(f"Generated diagnostic phase maps for {category}: {cat_png}")

        # Data folder linkage or copy
        src_data_dir = Path(res_obj["realization_dir"])
        dest_data_link = cat_dir / "data"

        if dest_data_link.exists() or dest_data_link.is_symlink():
            if dest_data_link.is_symlink():
                dest_data_link.unlink()
            elif dest_data_link.is_dir():
                shutil.rmtree(dest_data_link)

        if args.copy_data:
            shutil.copytree(src_data_dir, dest_data_link)
            logger.info(f"Copied raw data directory for {category} into {dest_data_link}")
        else:
            try:
                dest_data_link.symlink_to(src_data_dir.resolve(), target_is_directory=True)
                logger.info(f"Symlinked raw data directory for {category} -> {src_data_dir}")
            except Exception as e:
                logger.warning(f"Could not create symlink for {category}: {e}. Falling back to copy.")
                shutil.copytree(src_data_dir, dest_data_link)

    print("\n" + "=" * 70)
    print("ENSEMBLE POST-PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Target Directory:      {target_dir}")
    print(f"Output Directory:      {output_dir}")
    print(f"Realizations Processed: {len(successful_results)} / {len(valid_folders)}")
    print("-" * 70)
    print(summary_df.to_string(index=False))
    print("-" * 70)
    print(f"Best Performing Realization:  {best_name} (ROI 3 FDR={best_res['step3_full_roi3']['fdr_percent']}%, Pfaffian Passed={best_res['step3_full_roi3']['pfaffian_passed_pct']}%)")
    print(f"Worst Performing Realization: {worst_name} (ROI 3 FDR={worst_res['step3_full_roi3']['fdr_percent']}%, Pfaffian Passed={worst_res['step3_full_roi3']['pfaffian_passed_pct']}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
