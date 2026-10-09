#!/usr/bin/env python3
"""
Generate:
1. Three-Stage Progressive FDR Evolution Figures (progressive_fdr_evolution.png)
   and metrics JSON (progressive_fdr_metrics.json) for all 7 disorder realizations:
   - Stage 1: Dual Barrier Correlation alone (C_L >= 0.85 & C_R >= 0.85) [Blue #0284c7]
   - Stage 2: Correlation + 3w Curvature Agreement across Cutters [Amber #f59e0b]
   - Stage 3: Correlation + 3w + Transport Gap (ROI 3) [Emerald #10b981]
   with Pfaffian topological boundary (cyan) overlaid.

2. Multi-column Separability vs. Boundary Confinement Comparison Phase Maps:
   - Group A (Slide A): V0 = 0.0 (Clean), V0 = 0.1 (Low), V0 = 0.378 (Intermediate)
   - Group B (Slide B): V0 = 0.645, V0 = 0.872, V0 = 0.91, V0 = 1.2
   Row 1: MZM Separability phase map (S in [0.5, 1.0])
   Row 2: Boundary Confinement phase map (outer wire end localization)
   Colored inside topological boundary, dimmed/grey outside.
"""

import sys
import os
import json
import logging
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import ndimage

import tgp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProgressiveAndLocalizationPlots")

BASE_DIR = Path("/home/pseudonym/Documents/Code/NonlocalProtocol")
DATA_DIR = BASE_DIR / "Outputs/Data"
PLOTS_DIR = BASE_DIR / "Outputs/Plots"

DATASETS = [
    ("Tdis_pfaff5_V0_0_0", 0.001, "V0 = 0.0 (Clean Wire)"),
    ("Tdis_pfaff5_V0_0_1", 0.01, "V0 = 0.1 (Low Disorder)"),
    ("Tdis_pfaff5_V0_0_378", 0.01, "V0 = 0.378 (Intermediate Disorder)"),
    ("Tdis_pfaff5_V0_0_645", 0.05, "V0 = 0.645 (Strong Disorder)"),
    ("Tdis_pfaff5_V0_0_872", 0.05, "V0 = 0.872 (Strong Disorder)"),
    ("Tdis_pfaff5_V0_0_91", 0.05, "V0 = 0.91 (Strong Disorder)"),
    ("Tdis_pfaff5", 0.05, "V0 = 1.2 (Benchmark Disorder)"),
]

def calc_confusion_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    tp = int(np.sum(pred_mask & gt_mask))
    fp = int(np.sum(pred_mask & (~gt_mask)))
    fn = int(np.sum((~pred_mask) & gt_mask))
    tn = int(np.sum((~pred_mask) & (~gt_mask)))
    tot_pass = tp + fp
    tot_gt = tp + fn
    ppv = float(tp / tot_pass) if tot_pass > 0 else 0.0
    fdr = float(fp / tot_pass) if tot_pass > 0 else 0.0
    tpr = float(tp / tot_gt) if tot_gt > 0 else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "total_passed": tot_pass,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "ppv_percent": round(ppv * 100.0, 2),
        "fdr_percent": round(fdr * 100.0, 2),
        "tpr_percent": round(tpr * 100.0, 2),
        "tnr_percent": round(tnr * 100.0, 2),
    }

def generate_progressive_evolution_plots():
    logger.info("Generating Progressive FDR Evolution Plots...")
    all_summary = {}

    for ds_name, gap_th_factor, label in DATASETS:
        p_data = DATA_DIR / ds_name
        p_plot = PLOTS_DIR / f"{ds_name}_Plots"
        p_plot.mkdir(parents=True, exist_ok=True)
        roi3_dir = p_plot / "ROI3_Cuts"
        roi3_dir.mkdir(parents=True, exist_ok=True)

        tprep = xr.load_dataset(p_data / "tprep.nc")
        V_vals = tprep["V"].values
        B_vals = tprep["B"].values
        gt = tprep["ground_truth_topological"].values  # (V, B)

        cL = np.load(p_data / "barrier_left_correlation_arr.npy").reshape(len(V_vals), len(B_vals))
        cR = np.load(p_data / "barrier_right_correlation_arr.npy").reshape(len(V_vals), len(B_vals))
        corr_both = (cL >= 0.85) & (cR >= 0.85)

        t_l = tprep.rename({"bias": "left_bias"})
        t_r = tprep.rename({"bias": "right_bias"})
        t_l, t_r = tgp.two.extract_gap(t_l, t_r, gap_threshold_factor=gap_th_factor, noise_threshold=1e-4)
        zbp_ds = tgp.two.zbp_dataset_derivative(t_l, t_r, zbp_probability_threshold=0.7, average_over_cutter=False)
        tgp.two.set_gap_threshold(zbp_ds, threshold_low=10e-3, threshold_high=0.067)

        # Stage 1: Correlation alone
        m1 = corr_both

        # Stage 2: Correlation + 3w across cutters (>= 50% cutters)
        zbp_2d = (zbp_ds["zbp"].mean(dim="cutter_pair_index") >= 0.5).transpose("V", "B").values
        m2 = m1 & zbp_2d

        # Stage 3: Full ROI 3: Correlation + 3w + Gap
        gzbp_2d = (zbp_ds["gapped_zbp"].mean(dim="cutter_pair_index") >= 0.5).values
        m3 = m1 & gzbp_2d

        # Fallback if m3 has 0 pixels
        used_fallback = False
        if np.sum(m3) == 0:
            logger.info(f"[{ds_name}] Fallback triggered for M3")
            # candidate orange & corr
            cand_orange = (zbp_ds["gapped_zbp"].values[0] > 0)
            m3 = cand_orange & corr_both
            used_fallback = True

        m1_metrics = calc_confusion_metrics(m1, gt)
        m2_metrics = calc_confusion_metrics(m2, gt)
        m3_metrics = calc_confusion_metrics(m3, gt)

        metrics_dict = {
            "dataset": ds_name,
            "label": label,
            "stage1_correlation": m1_metrics,
            "stage2_corr_plus_3w": m2_metrics,
            "stage3_full_roi3": m3_metrics,
            "used_fallback": used_fallback
        }
        all_summary[ds_name] = metrics_dict

        with open(roi3_dir / "progressive_fdr_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)

        # Render 3-panel widescreen figure: 1 row, 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True, sharey=True)

        B_grid, V_grid = np.meshgrid(B_vals, V_vals)
        extent = [B_vals[0], B_vals[-1], V_vals[0], V_vals[-1]]

        stages = [
            (axes[0], m1, "#0284c7", f"Step 1: Barrier Correlation ($C_L, C_R \geq 0.85$)\nPassed: {m1_metrics['total_passed']} px | FDR: {m1_metrics['fdr_percent']}%"),
            (axes[1], m2, "#f59e0b", f"Step 2: Corr + $3\omega$ Curvature Agreement\nPassed: {m2_metrics['total_passed']} px | FDR: {m2_metrics['fdr_percent']}%"),
            (axes[2], m3, "#10b981", f"Step 3: Full ROI 3 (Corr + $3\omega$ + Transport Gap)\nPassed: {m3_metrics['total_passed']} px | FDR: {m3_metrics['fdr_percent']}%"),
        ]

        for ax, mask, color, title in stages:
            # Base failed background: subtle slate gray
            ax.set_facecolor("#1e293b")
            
            # Display passed mask as solid color overlay
            img_rgba = np.zeros((len(V_vals), len(B_vals), 4))
            # Background failed: dark grey with faint alpha
            img_rgba[~mask] = [0.12, 0.16, 0.23, 0.95]
            # Passed: theme color
            rgb_col = matplotlib.colors.to_rgb(color)
            img_rgba[mask] = [rgb_col[0], rgb_col[1], rgb_col[2], 0.95]

            ax.imshow(img_rgba, origin='lower', extent=extent, aspect='auto')

            # Overlay Ground Truth Pfaffian contour in cyan
            ax.contour(B_vals, V_vals, gt.astype(float), levels=[0.5], colors=['#38bdf8'], linewidths=1.8, linestyles='--')

            ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.3, color='white')

        axes[0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)

        # Legend on panel 1
        custom_lines = [
            matplotlib.lines.Line2D([0], [0], color='#38bdf8', lw=2, linestyle='--', label=r'Pfaffian Boundary ($\mathcal{Q} = -1$)'),
            matplotlib.patches.Patch(facecolor='#0284c7', label='Step 1 Passed'),
            matplotlib.patches.Patch(facecolor='#f59e0b', label='Step 2 Passed'),
            matplotlib.patches.Patch(facecolor='#10b981', label='Step 3 Passed (ROI 3)'),
        ]
        axes[0].legend(handles=custom_lines, loc='upper left', fontsize=8, facecolor='#0f172a', edgecolor='#334155', labelcolor='white')

        fig.suptitle(f"Progressive Protocol Filtering Evolution: {label}", fontsize=13, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        
        out_path1 = roi3_dir / "progressive_fdr_evolution.png"
        out_path2 = p_plot / "progressive_fdr_evolution.png"
        fig.savefig(out_path1, dpi=200)
        fig.savefig(out_path2, dpi=200)
        plt.close(fig)
        logger.info(f"[{ds_name}] Saved progressive_fdr_evolution.png")

    with open(PLOTS_DIR / "all_progressive_fdr_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)
    logger.info("Saved all_progressive_fdr_summary.json")

def generate_separability_vs_confinement_slides():
    logger.info("Generating Separability vs. Boundary Confinement Slides...")

    groups = [
        ("group_A", [
            ("Tdis_pfaff5_V0_0_0", "V0 = 0.0 (Clean)"),
            ("Tdis_pfaff5_V0_0_1", "V0 = 0.1 (Low)"),
            ("Tdis_pfaff5_V0_0_378", "V0 = 0.378 (Intermediate)")
        ]),
        ("group_B", [
            ("Tdis_pfaff5_V0_0_645", "V0 = 0.645 (Strong)"),
            ("Tdis_pfaff5_V0_0_872", "V0 = 0.872 (Strong)"),
            ("Tdis_pfaff5_V0_0_91", "V0 = 0.91 (Severe)"),
            ("Tdis_pfaff5", "V0 = 1.2 (Benchmark)")
        ])
    ]

    for grp_name, ds_list in groups:
        n_cols = len(ds_list)
        fig, axes = plt.subplots(2, n_cols, figsize=(5.2 * n_cols, 8.5), sharex=True, sharey=True)

        for col_idx, (ds_name, col_title) in enumerate(ds_list):
            p_data = DATA_DIR / ds_name
            tprep = xr.load_dataset(p_data / "tprep.nc")
            V_vals = tprep["V"].values
            B_vals = tprep["B"].values
            gt = tprep["ground_truth_topological"].values  # (V, B)

            extent = [B_vals[0], B_vals[-1], V_vals[0], V_vals[-1]]

            sep = np.load(p_data / "mzm_separability_arr.npy").reshape(len(V_vals), len(B_vals))
            conf = np.load(p_data / "mzm_boundary_confinement_arr.npy").reshape(len(V_vals), len(B_vals))

            # Row 1: Separability S in [0.5, 1.0]
            ax_sep = axes[0, col_idx]
            # Muted grey outside topological island, viridis inside
            sep_display = np.copy(sep)
            # Clip between 0.5 and 1.0
            sep_display = np.clip(sep_display, 0.5, 1.0)
            
            im_sep = ax_sep.imshow(sep_display, origin='lower', extent=extent, aspect='auto', cmap='magma', vmin=0.5, vmax=1.0)
            
            # Dim region outside Pfaffian
            dim_outside = np.zeros((len(V_vals), len(B_vals), 4))
            dim_outside[~gt] = [0.1, 0.1, 0.15, 0.65]  # semi-transparent dark mask
            ax_sep.imshow(dim_outside, origin='lower', extent=extent, aspect='auto')

            # Cyan contour for Pfaffian boundary
            ax_sep.contour(B_vals, V_vals, gt.astype(float), levels=[0.5], colors=['#38bdf8'], linewidths=1.8, linestyles='--')
            ax_sep.set_title(f"{col_title}\nMZM Separability ($S$)", fontsize=11, fontweight='bold')
            ax_sep.grid(True, linestyle=':', alpha=0.3, color='white')

            # Row 2: Boundary Confinement in [0.0, 1.0]
            ax_conf = axes[1, col_idx]
            conf_display = np.clip(conf, 0.0, 1.0)
            im_conf = ax_conf.imshow(conf_display, origin='lower', extent=extent, aspect='auto', cmap='viridis', vmin=0.0, vmax=0.9)
            
            # Dim region outside Pfaffian
            ax_conf.imshow(dim_outside, origin='lower', extent=extent, aspect='auto')
            ax_conf.contour(B_vals, V_vals, gt.astype(float), levels=[0.5], colors=['#38bdf8'], linewidths=1.8, linestyles='--')
            ax_conf.set_title(f"{col_title}\nBoundary Confinement", fontsize=11, fontweight='bold')
            ax_conf.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10)
            ax_conf.grid(True, linestyle=':', alpha=0.3, color='white')

        axes[0, 0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)
        axes[1, 0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)

        # Colorbars
        cbar_sep = fig.colorbar(im_sep, ax=axes[0, :].ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02)
        cbar_sep.set_label("Separability $S \in [0.5, 1.0]$", fontsize=10)

        cbar_conf = fig.colorbar(im_conf, ax=axes[1, :].ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02)
        cbar_conf.set_label("Boundary Confinement $B_c \in [0, 1]$", fontsize=10)

        fig.suptitle(f"Spatial Localization Evolution vs. Disorder ({grp_name.replace('_', ' ').title()})\n[Highlighted inside Topological Invariant $\\mathcal{{Q}}=-1$]", fontsize=13, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 0.92, 0.95])

        out_name = f"separability_vs_confinement_{grp_name}.png"
        fig.savefig(PLOTS_DIR / out_name, dpi=200)
        plt.close(fig)
        logger.info(f"Saved {out_name} to {PLOTS_DIR}")

if __name__ == "__main__":
    generate_progressive_evolution_plots()
    generate_separability_vs_confinement_slides()
    logger.info("All progressive FDR and localization comparison plots successfully generated!")
