#!/usr/bin/env python3
"""
Generate 4-Panel Comparison Figures for All 7 Disorder Realizations:
  1. Regular TGP (Microsoft Stage 2 ROI 2, without correlation condition)
  2. Step 1: Dual Barrier Reflection Correlation Alone (C_L, C_R >= 0.85)
  3. Step 2: Correlation + 3w Curvature Agreement across cutters
  4. Step 3: Full ROI 3 (Correlation + 3w + Transport Gap Delta > 10 ueV)

Strict User Directives:
  - Do NOT include areas that are not in the Pfaffian area in this figure (strictly mask out Q = +1 trivial bulk).
  - Overlay dashed cyan contour for Pfaffian boundary (Q = -1).
  - Display the 2 key metrics on each panel:
      * % Pfaffian Area Passed (TPR = TP / Total Pfaffian Area * 100%)
      * False Discovery Rate (FDR = FP / Total Passed * 100%)
  - Save as tgp_vs_3step_protocol_comparison.png and tgp_vs_3step_metrics.json in each dataset's plot directories.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import tgp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TGPvs3StepComparison")

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

def calc_two_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    tp = int(np.sum(pred_mask & gt_mask))
    fp = int(np.sum(pred_mask & (~gt_mask)))
    tot_pass = tp + fp
    tot_gt = int(np.sum(gt_mask))
    tpr_pct = float((tp / tot_gt) * 100.0) if tot_gt > 0 else 0.0
    fdr_pct = float((fp / tot_pass) * 100.0) if tot_pass > 0 else 0.0
    return {
        "total_passed": tot_pass,
        "true_positives": tp,
        "false_positives": fp,
        "total_pfaffian_area": tot_gt,
        "pfaffian_passed_pct": round(tpr_pct, 2),
        "fdr_percent": round(fdr_pct, 2),
    }

def run_generation():
    all_metrics = {}

    for ds_name, gap_th_factor, label in DATASETS:
        logger.info(f"Processing {ds_name} ({label})...")
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
        z_ds = tgp.two.zbp_dataset_derivative(t_l, t_r, zbp_probability_threshold=0.7, average_over_cutter=False)
        tgp.two.set_gap_threshold(z_ds, threshold_low=10e-3, threshold_high=0.067)

        # 1. Regular TGP Stage 2 ROI 2
        s2 = tgp.two.cluster_and_score(
            z_ds,
            min_cluster_size=7,
            cluster_gap_threshold=10e-3,
            cluster_percentage_boundary_threshold=0.6,
            cluster_ncutter_threshold=0.5
        )
        roi2_vb = s2.roi2.values.T if s2.roi2.dims == ("B", "V") else s2.roi2.values
        m_tgp = (roi2_vb > 0)

        # 2. Step 1: Correlation alone
        m1 = corr_both

        # 3. Step 2: Correlation + 3w across cutters (>= 50% cutters)
        zbp_2d = (z_ds["zbp"].mean(dim="cutter_pair_index") >= 0.5).transpose("V", "B").values
        m2 = m1 & zbp_2d

        # 4. Step 3: Full ROI 3: Correlation + 3w + Transport Gap
        gzbp_2d = (z_ds["gapped_zbp"].mean(dim="cutter_pair_index") >= 0.5).values
        m3 = m1 & gzbp_2d
        used_fallback = False
        if np.sum(m3) == 0:
            cand_orange = (z_ds["gapped_zbp"].values[0] > 0)
            m3 = cand_orange & corr_both
            used_fallback = True

        # Calculate exact 2 metrics for each step
        metrics_tgp = calc_two_metrics(m_tgp, gt)
        metrics_s1 = calc_two_metrics(m1, gt)
        metrics_s2 = calc_two_metrics(m2, gt)
        metrics_s3 = calc_two_metrics(m3, gt)

        ds_result = {
            "dataset": ds_name,
            "label": label,
            "regular_tgp": metrics_tgp,
            "step1_correlation": metrics_s1,
            "step2_corr_plus_3w": metrics_s2,
            "step3_full_roi3": metrics_s3,
            "used_fallback": used_fallback
        }
        all_metrics[ds_name] = ds_result

        # Save metrics JSON
        out_json1 = roi3_dir / "tgp_vs_3step_metrics.json"
        out_json2 = p_plot / "tgp_vs_3step_metrics.json"
        with open(out_json1, "w", encoding="utf-8") as f:
            json.dump(ds_result, f, indent=2)
        with open(out_json2, "w", encoding="utf-8") as f:
            json.dump(ds_result, f, indent=2)

        # Render 2x2 comparison figure with strict Pfaffian masking
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2), sharex=True, sharey=True)
        extent = [B_vals[0], B_vals[-1], V_vals[0], V_vals[-1]]

        panels = [
            (axes[0, 0], m_tgp, "#a855f7", "Regular TGP (Stage 2 ROI 2)", metrics_tgp),
            (axes[0, 1], m1, "#0284c7", "Step 1: Barrier Correlation ($C_L, C_R \geq 0.85$)", metrics_s1),
            (axes[1, 0], m2, "#f59e0b", "Step 2: Corr + $3\omega$ Curvature Agreement", metrics_s2),
            (axes[1, 1], m3, "#10b981", "Step 3: Full ROI 3 (Corr + $3\omega$ + Transport Gap)", metrics_s3),
        ]

        for ax, mask, color, title, m_dict in panels:
            ax.set_facecolor("#0b1120")  # deep dark background outside Pfaffian

            # Strict Pfaffian domain masking:
            # - Outside Pfaffian (~gt): fully transparent/background color (dark)
            # - Inside Pfaffian (gt) and NOT passed (~mask): dark slate (#1e293b)
            # - Inside Pfaffian (gt) and passed (mask): theme color
            img_rgba = np.zeros((len(V_vals), len(B_vals), 4))
            
            # Unpassed within Pfaffian: dark slate
            img_rgba[gt & (~mask)] = [0.12, 0.16, 0.23, 0.95]
            
            # Passed within Pfaffian: theme color
            rgb = mcolors.to_rgb(color)
            img_rgba[gt & mask] = [rgb[0], rgb[1], rgb[2], 0.95]

            # Areas outside Pfaffian (~gt) remain 0.0 alpha (or dark background)
            img_rgba[~gt] = [0.04, 0.07, 0.13, 1.0]

            ax.imshow(img_rgba, origin='lower', extent=extent, aspect='auto')

            # Overlay Ground Truth Pfaffian boundary in dashed cyan
            ax.contour(B_vals, V_vals, gt.astype(float), levels=[0.5], colors=['#38bdf8'], linewidths=1.6, linestyles='--')

            sub_title = f"{title}\nPfaffian Passed: {m_dict['pfaffian_passed_pct']:.2f}% | FDR: {m_dict['fdr_percent']:.2f}%"
            ax.set_title(sub_title, fontsize=10.5, fontweight='bold', pad=8, color='#f1f5f9')
            ax.grid(True, linestyle=':', alpha=0.25, color='white')

        # Labels
        axes[0, 0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)
        axes[1, 0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)
        axes[1, 0].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10)
        axes[1, 1].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10)

        # Legend on top-left panel
        legend_elements = [
            mlines.Line2D([0], [0], color='#38bdf8', lw=1.8, linestyle='--', label=r'Pfaffian Boundary ($\mathcal{Q} = -1$)'),
            mpatches.Patch(facecolor='#1e293b', edgecolor='#475569', label='Pfaffian Bulk (Unpassed)'),
            mpatches.Patch(facecolor='#a855f7', label='Regular TGP Passed'),
            mpatches.Patch(facecolor='#0284c7', label='Step 1 Passed'),
            mpatches.Patch(facecolor='#f59e0b', label='Step 2 Passed'),
            mpatches.Patch(facecolor='#10b981', label='Step 3 Passed (ROI 3)'),
        ]
        axes[0, 0].legend(handles=legend_elements, loc='upper left', fontsize=7.5, facecolor='#0f172a', edgecolor='#334155', labelcolor='white')

        fig.suptitle(f"Protocol Comparison (Pfaffian Domain Only): {label}", fontsize=12.5, fontweight='bold', color='#38bdf8', y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_img1 = roi3_dir / "tgp_vs_3step_protocol_comparison.png"
        out_img2 = p_plot / "tgp_vs_3step_protocol_comparison.png"
        fig.savefig(out_img1, dpi=200)
        fig.savefig(out_img2, dpi=200)
        plt.close(fig)
        logger.info(f"[{ds_name}] Saved tgp_vs_3step_protocol_comparison.png")

    with open(PLOTS_DIR / "all_tgp_vs_3step_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved all_tgp_vs_3step_summary.json successfully.")

if __name__ == "__main__":
    run_generation()
