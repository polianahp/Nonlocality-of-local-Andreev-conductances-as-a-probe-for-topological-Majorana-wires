#!/usr/bin/env python3
"""
generate_finite_T_correlation_phase_maps.py

Generates finite-temperature conductance correlation phase maps across temperatures
(10 mK, 20 mK, 30 mK, 40 mK, 50 mK) for Tdis_pfaff5 disorder dataset:
  1. Dual plot for raw reflection correlation for each side (Left C_L and Right C_R).
  2. Combined binarized plot where:
     - Blue: At least one side meets threshold (C_L >= threshold or C_R >= threshold)
     - Green: Both sides meet threshold (C_L >= threshold and C_R >= threshold)
     - Red dashed line: Topological ground truth Pfaffian invariant boundary (Q = -1)

Exactly two plots generated for each temperature, saving cleanly to Plots/Finite_T_correllation_analysis.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FiniteTCorrelation")

REPO_ROOT = Path(__file__).resolve().parent
DATA_BASE = REPO_ROOT / "Outputs" / "Data" / "Tdis_pfaff5_data"
PRIMARY_OUT_DIR = REPO_ROOT / "Outputs" / "Plots" / "Finite_T_correllation_analysis"
ALIAS_OUT_DIR = REPO_ROOT / "Plots" / "Finite_T_correllation_analysis"

TEMPERATURE_DIRS = [
    ("Tdispfaff5_10mK", 10.0),
    ("Tdispfaff5_20mK", 20.0),
    ("Tdispfaff5_30mK", 30.0),
    ("Tdispfaff5_40mK", 40.0),
    ("Tdispfaff5_50mK", 50.0),
]


def vectorized_reflection_correlation(
    f1: np.ndarray,
    f2: np.ndarray,
    barrier_arr: np.ndarray
) -> np.ndarray:
    """
    Computes normalized correlation C(f1, f2) across barrier traces
    restricted to the non-negative barrier regime (U >= 0).
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


def setup_output_directories():
    PRIMARY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_root = REPO_ROOT / "Plots"
    if not plots_root.exists():
        try:
            plots_root.symlink_to(REPO_ROOT / "Outputs" / "Plots")
            logger.info("Created symlink: Plots -> Outputs/Plots")
        except Exception:
            ALIAS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    elif not ALIAS_OUT_DIR.exists():
        ALIAS_OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_dual_raw_correlation_map(
    out_path: Path,
    cL_2d: np.ndarray,
    cR_2d: np.ndarray,
    gt_2d: np.ndarray,
    B_vals: np.ndarray,
    V_vals: np.ndarray,
    T_mK: float,
    stats: Dict[str, Any],
    threshold: float
):
    """Renders 1x2 comparative PNG for Left vs Right barrier sweep raw correlations."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.5), dpi=300, sharey=True, layout='constrained')
    
    mesh1 = axes[0].pcolormesh(B_vals, V_vals, cL_2d, cmap="Blues", vmin=0.0, vmax=1.0, shading="nearest")
    mesh2 = axes[1].pcolormesh(B_vals, V_vals, cR_2d, cmap="Blues", vmin=0.0, vmax=1.0, shading="nearest")

    for ax in axes:
        ax.contour(
            B_vals, V_vals, gt_2d.astype(float),
            levels=[0.5], colors=["cyan"], linewidths=[1.8], linestyles=["--"]
        )
        ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11.5)
    axes[0].set_title(
        f"Left Barrier Sweep ($C_L$) [Mean = {stats['cL_mean']:.3f}, $\\geq {threshold:.2f}$: {stats['cL_bin_pct']:.1f}%]",
        fontsize=11.5, fontweight="bold"
    )
    axes[1].set_title(
        f"Right Barrier Sweep ($C_R$) [Mean = {stats['cR_mean']:.3f}, $\\geq {threshold:.2f}$: {stats['cR_bin_pct']:.1f}%]",
        fontsize=11.5, fontweight="bold"
    )

    legend_line = plt.Line2D([0], [0], color="cyan", linestyle="--", linewidth=1.8, label=r"Pfaffian Boundary ($\mathcal{Q} = -1$)")
    axes[0].legend(handles=[legend_line], loc="lower right", framealpha=0.9, fontsize=9.5)
    axes[1].legend(handles=[legend_line], loc="lower right", framealpha=0.9, fontsize=9.5)

    cbar = fig.colorbar(mesh2, ax=axes, pad=0.02)
    cbar.set_label("Reflection Correlation (Cosine Similarity)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        f"Raw Barrier Reflection Correlation Phase Maps — T = {int(T_mK)} mK",
        fontsize=14, fontweight="bold"
    )

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def render_combined_binarized_map(
    out_path: Path,
    cL_bin_2d: np.ndarray,
    cR_bin_2d: np.ndarray,
    gt_2d: np.ndarray,
    B_vals: np.ndarray,
    V_vals: np.ndarray,
    T_mK: float,
    stats: Dict[str, Any],
    threshold: float
):
    """
    Renders combined binarized correlation phase map where:
      - Blue: at least one side meets threshold (one side only)
      - Green: both sides meet threshold
      - Off-white/Grey: neither side meets threshold
    """
    fig, ax = plt.subplots(figsize=(9.0, 7.8), dpi=300, layout='constrained')

    both_pass = cL_bin_2d & cR_bin_2d
    any_pass = cL_bin_2d | cR_bin_2d
    one_pass = any_pass & (~both_pass)

    combined_2d = np.zeros_like(cL_bin_2d, dtype=int)
    combined_2d[one_pass] = 1   # Blue: one side only meets threshold
    combined_2d[both_pass] = 2  # Green: both sides meet threshold

    # Colormap: 0 = #f8fafc (neither), 1 = #0284c7 (blue), 2 = #059669 (green)
    cmap = mcolors.ListedColormap(["#f8fafc", "#0284c7", "#059669"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    mesh = ax.pcolormesh(
        B_vals, V_vals, combined_2d,
        cmap=cmap, norm=norm, shading="nearest"
    )

    # Pfaffian ground truth boundary contour
    ax.contour(
        B_vals, V_vals, gt_2d.astype(float),
        levels=[0.5], colors=["#dc2626"], linewidths=[2.0], linestyles=["--"]
    )

    ax.set_title(
        rf"Combined Binarized Correlation Phase Map — T = {int(T_mK)} mK (Threshold $\geq {threshold:.2f}$)",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11)
    ax.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11)
    ax.tick_params(labelsize=10)

    # Info text box
    both_pts = stats["both_pts"]
    both_pct = stats["both_pct"]
    both_tp = stats["both_tp"]
    both_fp = stats["both_fp"]
    both_tpr = stats["both_tpr"]
    both_fdr = stats["both_fdr"]

    any_pts = stats["any_pts"]
    any_pct = stats["any_pct"]
    any_tp = stats["any_tp"]
    any_fp = stats["any_fp"]
    any_tpr = stats["any_tpr"]
    any_fdr = stats["any_fdr"]

    info_text = (
        f"Temperature: {int(T_mK)} mK (Threshold $\\geq {threshold:.2f}$)\n"
        f"Green (Both Meet): {both_pct:.2f}% ({both_pts} pts) | TPR: {both_tpr:.1f}% | FDR: {both_fdr:.1f}%\n"
        f"Blue (At least one): {any_pct:.2f}% ({any_pts} pts) | TPR: {any_tpr:.1f}% | FDR: {any_fdr:.1f}%\n"
        f"Topological Ground Truth: {stats['gt_pts']} pts ({stats['gt_pct']:.2f}%)"
    )
    ax.text(
        0.03, 0.97, info_text,
        transform=ax.transAxes,
        fontsize=9.2,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='#94a3b8')
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#059669", edgecolor="#047857", label=rf"Green: Both meet threshold ($C_L, C_R \geq {threshold:.2f}$)"),
        mpatches.Patch(facecolor="#0284c7", edgecolor="#0369a1", label=rf"Blue: At least one meets threshold ($C_L$ or $C_R \geq {threshold:.2f}$)"),
        mpatches.Patch(facecolor="#f8fafc", edgecolor="#cbd5e1", label=rf"Grey: Neither meets threshold ($< {threshold:.2f}$)"),
        plt.Line2D([0], [0], color="#dc2626", linestyle="--", linewidth=2.0, label=r"Pfaffian Boundary ($\mathcal{Q} = -1$)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.92, fontsize=9.2)

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_markdown_summary(
    out_path: Path,
    all_data: List[Dict[str, Any]],
    threshold: float
):
    """Generates clean, structured README.md summarizing all finite-temperature correlation findings."""
    md = [
        f"# Finite-Temperature Conductance Correlation Analysis (Threshold $\\geq {threshold:.2f}$)",
        "",
        "## Overview",
        "This directory contains publication-quality raw and combined binarized reflection correlation phase maps ",
        "for the finite-temperature disorder dataset `Tdis_pfaff5_data` ($V_0 = 1.2$ meV, $L_s = 300$ sites).",
        "",
        f"For each temperature, **exactly two figures** are produced:",
        f"1. `dual_raw_correlation_{{T}}mK.png`: Side-by-side continuous reflection correlation maps ($C_L$ and $C_R$).",
        f"2. `binarized_combined_correlation_{{T}}mK.png`: Combined binarized phase map with Pfaffian ground truth boundary:",
        f"   - **Blue**: At least one side meets the correlation threshold ($C_L \\ge {threshold:.2f} \\lor C_R \\ge {threshold:.2f}$).",
        f"   - **Green**: Both sides meet the correlation threshold ($C_L \\ge {threshold:.2f} \\land C_R \\ge {threshold:.2f}$).",
        f"   - **Red Dashed Contour**: Topological ground truth Pfaffian invariant boundary ($\\mathcal{{Q}} = -1$).",
        "",
        "---",
        "",
        f"## Quantitative Summary Table (Threshold = {threshold:.2f})",
        "",
        f"| Temperature | Mean $C_L$ | Mean $C_R$ | At Least One Pass (Blue) | Blue TPR | Blue FDR | Both Pass (Green) | Green TPR | Green FDR |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for d in all_data:
        s = d["stats"]
        T = int(d["T_mK"])
        row = (
            f"| **{T} mK** | {s['cL_mean']:.3f} | {s['cR_mean']:.3f} | "
            f"{s['any_pct']:.2f}% ({s['any_pts']} pts) | {s['any_tpr']:.2f}% | {s['any_fdr']:.2f}% | "
            f"**{s['both_pct']:.2f}% ({s['both_pts']} pts)** | **{s['both_tpr']:.2f}%** | **{s['both_fdr']:.2f}%** |"
        )
        md.append(row)

    md.extend([
        "",
        "---",
        "",
        "## Multi-Threshold Evolution: 0.85 vs. 0.50 vs. 0.40 Impact (Both Pass / Green)",
        "",
        "| Temperature | Metric | Threshold $\\ge 0.85$ | Threshold $\\ge 0.50$ | Threshold $\\ge 0.40$ |",
        "| :--- | :--- | :---: | :---: | :---: |",
        "| **10 mK** | Passing Area % (Pts) <br> True Positives (TP) <br> TPR (%) <br> FDR (%) | 4.69% (739 pts) <br> 582 <br> 18.82% <br> 21.24% | 12.94% (2,038 pts) <br> 1,340 <br> 43.32% <br> 34.25% | **17.62% (2,775 pts)** <br> **1,734** <br> **56.06%** <br> 37.51% |",
        "| **20 mK** | Passing Area % (Pts) <br> True Positives (TP) <br> TPR (%) <br> FDR (%) | 1.71% (270 pts) <br> 233 <br> 7.27% <br> 16.67% | 8.20% (1,292 pts) <br> 836 <br> 27.03% <br> 35.29% | **12.59% (1,983 pts)** <br> **1,203** <br> **38.89%** <br> 39.33% |",
        "| **30 mK** | Passing Area % (Pts) <br> True Positives (TP) <br> TPR (%) <br> FDR (%) | 0.65% (103 pts) <br> 79 <br> 2.55% <br> 13.19% | 5.49% (864 pts) <br> 571 <br> 18.46% <br> 33.91% | **9.68% (1,524 pts)** <br> **915** <br> **29.58%** <br> 39.96% |",
        "| **40 mK** | Passing Area % (Pts) <br> True Positives (TP) <br> TPR (%) <br> FDR (%) | 0.15% (23 pts) <br> 14 <br> 0.45% <br> 17.65% | 3.82% (602 pts) <br> 377 <br> 12.19% <br> 37.38% | **7.90% (1,245 pts)** <br> **738** <br> **23.86%** <br> 40.72% |",
        "| **50 mK** | Passing Area % (Pts) <br> True Positives (TP) <br> TPR (%) <br> FDR (%) | 0.01% (2 pts) <br> 0 <br> 0.00% <br> 100.0% | 2.91% (458 pts) <br> 265 <br> 8.57% <br> 42.14% | **6.88% (1,084 pts)** <br> **610** <br> **19.72%** <br> 43.73% |",
        "",
        "---",
        "",
        "## File Inventory (Exactly 2 plots per temperature)",
        "",
        "- `dual_raw_correlation_10mK.png` & `binarized_combined_correlation_10mK.png`",
        "- `dual_raw_correlation_20mK.png` & `binarized_combined_correlation_20mK.png`",
        "- `dual_raw_correlation_30mK.png` & `binarized_combined_correlation_30mK.png`",
        "- `dual_raw_correlation_40mK.png` & `binarized_combined_correlation_40mK.png`",
        "- `dual_raw_correlation_50mK.png` & `binarized_combined_correlation_50mK.png`",
        "- `finite_temperature_correlation_metrics.json`",
        ""
    ])

    out_path.write_text("\n".join(md), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate finite-T conductance correlation phase maps.")
    parser.add_argument("--threshold", type=float, default=0.40, help="Correlation threshold for binarization (default: 0.40)")
    args = parser.parse_args()

    threshold = float(args.threshold)
    setup_output_directories()
    logger.info(f"Target primary output directory: {PRIMARY_OUT_DIR}")
    logger.info(f"Using correlation threshold: {threshold:.2f}")

    all_data = []

    for dir_name, T_mK in TEMPERATURE_DIRS:
        data_dir = DATA_BASE / dir_name
        if not data_dir.exists():
            logger.error(f"Directory {data_dir} not found! Skipping.")
            continue

        logger.info(f"Processing {dir_name} (T = {T_mK} mK)...")
        
        # Load parameters and axes
        params = np.load(data_dir / "all_params.npz", allow_pickle=True)
        mu_var = params["mu_var"]
        Vz_var = params["Vz_var"]
        Nmu = len(mu_var)
        Nvz = len(Vz_var)

        # Load Pfaffian ground truth
        pdi_data = np.load(data_dir / "pdi_data.npy")
        col_idx = 3 if pdi_data.shape[1] >= 4 else 2
        gt_2d = (pdi_data[:, col_idx] == 1.0).reshape(Nmu, Nvz)
        gt_pts = int(np.sum(gt_2d))
        gt_pct = float(gt_pts / gt_2d.size * 100.0)

        # Load barrier conductances
        b_arr = np.load(data_dir / "barrier_arr.npy")
        bl_l = np.load(data_dir / "barrier_left_conductance_left_arr.npy")
        bl_r = np.load(data_dir / "barrier_left_conductance_right_arr.npy")
        br_l = np.load(data_dir / "barrier_right_conductance_left_arr.npy")
        br_r = np.load(data_dir / "barrier_right_conductance_right_arr.npy")

        # Compute or load correlation
        corr_L_file = data_dir / "barrier_left_correlation_arr.npy"
        corr_R_file = data_dir / "barrier_right_correlation_arr.npy"

        if corr_L_file.exists():
            cL = np.load(corr_L_file)
        else:
            cL = vectorized_reflection_correlation(bl_l, bl_r, b_arr)
            np.save(corr_L_file, cL)

        if corr_R_file.exists():
            cR = np.load(corr_R_file)
        else:
            cR = vectorized_reflection_correlation(br_l, br_r, b_arr)
            np.save(corr_R_file, cR)

        cL_2d = cL.reshape(Nmu, Nvz)
        cR_2d = cR.reshape(Nmu, Nvz)
        
        # Binarized masks
        cL_bin_2d = cL_2d >= threshold
        cR_bin_2d = cR_2d >= threshold
        both_pass = cL_bin_2d & cR_bin_2d
        any_pass = cL_bin_2d | cR_bin_2d

        # Left side stats
        cL_tp = int(np.sum(cL_bin_2d & gt_2d))
        cL_fp = int(np.sum(cL_bin_2d & (~gt_2d)))
        cL_bin_pts = int(np.sum(cL_bin_2d))
        cL_bin_pct = float(cL_bin_pts / cL_bin_2d.size * 100.0)
        cL_tpr = float(cL_tp / gt_pts * 100.0) if gt_pts > 0 else 0.0
        cL_fdr = float(cL_fp / cL_bin_pts * 100.0) if cL_bin_pts > 0 else 0.0

        # Right side stats
        cR_tp = int(np.sum(cR_bin_2d & gt_2d))
        cR_fp = int(np.sum(cR_bin_2d & (~gt_2d)))
        cR_bin_pts = int(np.sum(cR_bin_2d))
        cR_bin_pct = float(cR_bin_pts / cR_bin_2d.size * 100.0)
        cR_tpr = float(cR_tp / gt_pts * 100.0) if gt_pts > 0 else 0.0
        cR_fdr = float(cR_fp / cR_bin_pts * 100.0) if cR_bin_pts > 0 else 0.0

        # Both meet threshold (Green)
        both_tp = int(np.sum(both_pass & gt_2d))
        both_fp = int(np.sum(both_pass & (~gt_2d)))
        both_pts = int(np.sum(both_pass))
        both_pct = float(both_pts / both_pass.size * 100.0)
        both_tpr = float(both_tp / gt_pts * 100.0) if gt_pts > 0 else 0.0
        both_fdr = float(both_fp / both_pts * 100.0) if both_pts > 0 else 0.0

        # At least one meets threshold (Blue + Green)
        any_tp = int(np.sum(any_pass & gt_2d))
        any_fp = int(np.sum(any_pass & (~gt_2d)))
        any_pts = int(np.sum(any_pass))
        any_pct = float(any_pts / any_pass.size * 100.0)
        any_tpr = float(any_tp / gt_pts * 100.0) if gt_pts > 0 else 0.0
        any_fdr = float(any_fp / any_pts * 100.0) if any_pts > 0 else 0.0

        stats = {
            "T_mK": T_mK,
            "threshold": threshold,
            "cL_mean": float(cL.mean()),
            "cL_min": float(cL.min()),
            "cL_max": float(cL.max()),
            "cL_bin_pts": cL_bin_pts,
            "cL_bin_pct": cL_bin_pct,
            "cL_tp": cL_tp,
            "cL_fp": cL_fp,
            "cL_tpr": cL_tpr,
            "cL_fdr": cL_fdr,
            "cR_mean": float(cR.mean()),
            "cR_min": float(cR.min()),
            "cR_max": float(cR.max()),
            "cR_bin_pts": cR_bin_pts,
            "cR_bin_pct": cR_bin_pct,
            "cR_tp": cR_tp,
            "cR_fp": cR_fp,
            "cR_tpr": cR_tpr,
            "cR_fdr": cR_fdr,
            "both_pts": both_pts,
            "both_pct": both_pct,
            "both_tp": both_tp,
            "both_fp": both_fp,
            "both_tpr": both_tpr,
            "both_fdr": both_fdr,
            "any_pts": any_pts,
            "any_pct": any_pct,
            "any_tp": any_tp,
            "any_fp": any_fp,
            "any_tpr": any_tpr,
            "any_fdr": any_fdr,
            "gt_pts": gt_pts,
            "gt_pct": gt_pct,
        }

        entry = {
            "dir_name": dir_name,
            "T_mK": T_mK,
            "cL_2d": cL_2d,
            "cR_2d": cR_2d,
            "cL_bin_2d": cL_bin_2d,
            "cR_bin_2d": cR_bin_2d,
            "stats": stats,
            "B_vals": Vz_var,
            "V_vals": mu_var,
            "gt_2d": gt_2d
        }
        all_data.append(entry)

        # 1. Dual raw correlation plot for Left and Right
        render_dual_raw_correlation_map(
            PRIMARY_OUT_DIR / f"dual_raw_correlation_{int(T_mK)}mK.png",
            cL_2d, cR_2d, gt_2d, Vz_var, mu_var, T_mK, stats, threshold
        )

        # 2. Combined binarized correlation plot (Blue = at least one, Green = both)
        render_combined_binarized_map(
            PRIMARY_OUT_DIR / f"binarized_combined_correlation_{int(T_mK)}mK.png",
            cL_bin_2d, cR_bin_2d, gt_2d, Vz_var, mu_var, T_mK, stats, threshold
        )

    # Export machine-readable metrics JSON
    metrics_export = {
        "dataset": "Tdis_pfaff5_data",
        "Ls": 300,
        "V0": 1.2,
        "threshold": threshold,
        "temperatures_mK": [d["T_mK"] for d in all_data],
        "metrics_by_temperature": {f"{int(d['T_mK'])}mK": d["stats"] for d in all_data}
    }
    with open(PRIMARY_OUT_DIR / "finite_temperature_correlation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)

    # Structured README.md
    generate_markdown_summary(PRIMARY_OUT_DIR / "README.md", all_data, threshold)

    logger.info(f"Successfully generated all {len(all_data) * 2} plots in {PRIMARY_OUT_DIR}")


if __name__ == "__main__":
    main()
