#!/usr/bin/env python3
"""
generate_weekly_presentation_figures.py

Generates all required figures for the 2026-10-28 weekly research slides:
1. Fig 1: 0 mK vs 40 mK legacy full-sweep [0, 60] meV reflection correlation comparison on realization 10 (V0 = 1.2 meV).
2. Fig 2: 40 mK comparison between legacy [0, 60] meV and low-asymmetry [-5, +5] meV reflection correlation on realization 10.
3. Fig 3: 40 mK Binarized low-asymmetry correlation phase map (CL, CR >= 0.90) on realization 10.
4. Fig 4: 40 mK Symmetry balance ratio phase map (continuous ratio and binarized S_sym <= 0.15) on realization 10.
5. Figs 5-8: Step-by-step 4-panel comparison figures for V0 = 0.378, 0.645, 0.872, 1.200 meV on realization 10,
   matching the style of Slides 2-7 of slides-2026-10-13.pdf.
6. Figs 9-12: Protocol cluster phase maps showing surviving clusters of tgp_bnd000_lowcorr090_sym015 with labeled IDs and COM stars.
7. Cut analytics, 5-panel conductance, and COM topological point deep-dives for every surviving cluster:
   - V0 = 0.378: Island 1 and Island 2
   - V0 = 0.872: Island 1
   - V0 = 1.200: Island 1
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter

REPO_ROOT = Path("/home/pseudonym/Documents/Code/NonlocalProtocol").resolve()
sys.path.insert(0, str(REPO_ROOT))

import src.helpers as hp
from src.Protocol_Elements import (
    RealizationFeatureCache,
    load_protocol_class_from_file,
    compute_step_diagnostic,
)
from src.Protocol_Elements.FPE.barrier_correlation import extract_barrier_reflection_correlations
from src.Protocol_Elements.FPE.barrier_asymmetry import fpe_lead_symmetry_balance
from src.Protocol_Elements.FPE.lockin_curvature_2w_3w import extract_zero_bias_curvature_3d
from src.Protocol_Elements.FPE.transport_and_spectral_gap import extract_nonlocal_transport_gap_maps

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WeeklySlideFigures")

OUT_FIG_DIR = REPO_ROOT / "Outputs" / "slides" / "weekly research slides" / "2026-10-28" / "figures"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = REPO_ROOT / "Outputs" / "Data" / "Disorder_Realizations"

# Colors & Styling
BG_DARK = "#0b1120"
BG_CARD = "#0f172a"
CYAN_PFAFF = "#38bdf8"
DARK_SLATE = "#1e293b"


def calc_two_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, Any]:
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


def generate_finite_t_correlation_figures():
    logger.info("Generating Finite-T Correlation & Symmetry Balance figures on realization 10 (V0 = 1.2 meV)...")
    real_dir = DATA_ROOT / "V0_1_2" / "disorder_realization_10"
    
    # 0 mK cache
    c0 = RealizationFeatureCache(real_dir, temperature_mK=0.0, v0_label="V0_1_2")
    # 40 mK cache
    c40 = RealizationFeatureCache(real_dir, temperature_mK=40.0, v0_label="V0_1_2")
    
    gt = c40.pfaffian_mask
    mu_vals = c40.mu_vals
    vz_vals = c40.vz_vals
    extent = [vz_vals[0], vz_vals[-1], mu_vals[0], mu_vals[-1]]
    
    # --- FIG 1: 0 mK vs 40 mK Legacy Correlation [0, 60] meV ---
    cL_0, cR_0 = extract_barrier_reflection_correlations(c0, asym_min=0.0, asym_max=60.0)
    cL_40, cR_40 = extract_barrier_reflection_correlations(c40, asym_min=0.0, asym_max=60.0)
    c_both_0 = np.minimum(cL_0, cR_0)
    c_both_40 = np.minimum(cL_40, cR_40)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG_DARK)
    
    for ax, c_map, title, temp_lbl in [
        (axes[0], c_both_0, "T = 0 mK (Legacy [0, 60] meV Sweep)", "0 mK"),
        (axes[1], c_both_40, "T = 40 mK (Legacy [0, 60] meV Sweep)", "40 mK"),
    ]:
        ax.set_facecolor("#030712")
        im = ax.imshow(c_map, origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=1.8, linestyles="--")
        ax.set_title(title, fontsize=12, fontweight="bold", color="#f8fafc", pad=8)
        ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11, color="white")
        ax.tick_params(colors="white")
        ax.grid(True, linestyle=":", alpha=0.25, color="white")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Dual Correlation $\min(C_L, C_R)$", color="white", fontsize=10)
        cbar.ax.tick_params(colors="white")
        
    axes[0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11, color="white")
    legend_line = mlines.Line2D([0], [0], color=CYAN_PFAFF, lw=1.8, linestyle="--", label=r"Pfaffian Boundary ($\mathcal{Q} = -1$)")
    axes[0].legend(handles=[legend_line], loc="upper left", fontsize=8.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
    
    fig.suptitle(r"Finite-Temperature Correlation Breakdown: $T = 0$ mK vs $T = 40$ mK ($V_0 = 1.2$ meV, Realization 10)", fontsize=13, fontweight="bold", color=CYAN_PFAFF, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG_DIR / "fig1_0mK_vs_40mK_legacy_correlation.png", dpi=200)
    plt.close(fig)
    logger.info("Saved fig1_0mK_vs_40mK_legacy_correlation.png")

    # --- FIG 2: 40 mK Legacy [0, 60] vs Low-Asymmetry [-5, +5] meV ---
    cL_low, cR_low = extract_barrier_reflection_correlations(c40, asym_min=-5.0, asym_max=5.0)
    c_both_low = np.minimum(cL_low, cR_low)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG_DARK)
    
    for ax, c_map, title in [
        (axes[0], c_both_40, "Legacy Full-Sweep Asymmetry ([0, 60] meV)"),
        (axes[1], c_both_low, r"Low-Asymmetry Gate Sweep ([-5, +5] meV)"),
    ]:
        ax.set_facecolor("#030712")
        im = ax.imshow(c_map, origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=1.8, linestyles="--")
        ax.set_title(title, fontsize=12, fontweight="bold", color="#f8fafc", pad=8)
        ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11, color="white")
        ax.tick_params(colors="white")
        ax.grid(True, linestyle=":", alpha=0.25, color="white")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Dual Correlation $\min(C_L, C_R)$", color="white", fontsize=10)
        cbar.ax.tick_params(colors="white")
        
    axes[0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11, color="white")
    axes[0].legend(handles=[legend_line], loc="upper left", fontsize=8.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
    
    fig.suptitle(r"Restoring Topological Correlation at $T = 40$ mK via Low-Asymmetry Window ([-5, +5] meV)", fontsize=13, fontweight="bold", color=CYAN_PFAFF, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG_DIR / "fig2_40mK_legacy_vs_low_asymmetry.png", dpi=200)
    plt.close(fig)
    logger.info("Saved fig2_40mK_legacy_vs_low_asymmetry.png")

    # --- FIG 3: 40 mK Binarized Low-Asymmetry Correlation Condition (CL, CR >= 0.75) ---
    m_bin_corr = (cL_low >= 0.75) & (cR_low >= 0.75)
    met_bin_corr = calc_two_metrics(m_bin_corr, gt)
    
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor("#030712")
    
    img_rgba = np.zeros((len(mu_vals), len(vz_vals), 4))
    img_rgba[~gt & (~m_bin_corr)] = [0.04, 0.07, 0.13, 1.0]
    img_rgba[gt & (~m_bin_corr)] = [0.12, 0.16, 0.23, 0.95]
    img_rgba[m_bin_corr] = [0.01, 0.52, 0.78, 0.95]  # cyan blue
    
    ax.imshow(img_rgba, origin="lower", extent=extent, aspect="auto")
    ax.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=1.8, linestyles="--")
    
    sub_title = (
        r"Binarized Low-Asymmetry Correlation Filter ($C_L, C_R \geq 0.75$ at $T = 40$ mK)"
        f"\nPfaffian Area Passed: {met_bin_corr['pfaffian_passed_pct']:.2f}% | FDR: {met_bin_corr['fdr_percent']:.2f}%"
    )
    ax.set_title(sub_title, fontsize=12, fontweight="bold", color="#f8fafc", pad=10)
    ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11, color="white")
    ax.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11, color="white")
    ax.tick_params(colors="white")
    ax.grid(True, linestyle=":", alpha=0.25, color="white")
    
    legend_elements = [
        mlines.Line2D([0], [0], color=CYAN_PFAFF, lw=1.8, linestyle="--", label=r"Pfaffian Boundary ($\mathcal{Q} = -1$)"),
        mpatches.Patch(facecolor="#1e293b", edgecolor="#475569", label="Pfaffian Bulk (Unpassed)"),
        mpatches.Patch(facecolor="#0284c7", label=r"Passed Pixels ($C_L, C_R \geq 0.75$)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
    
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "fig3_binarized_low_asymmetry_correlation_40mK.png", dpi=200)
    plt.close(fig)
    logger.info("Saved fig3_binarized_low_asymmetry_correlation_40mK.png")

    # --- FIG 4: 40 mK Symmetry Balance Ratio (Continuous + Binarized S_sym <= 0.15) ---
    m_sym = fpe_lead_symmetry_balance(c40, s_max=0.15, eps_sym=1e-4)
    s_sym = c40._memo[("lead_symmetry_balance_obs", 0.0001, 0.0)].copy()
    s_sym_plot = np.clip(np.nan_to_num(s_sym, nan=1.0, posinf=1.0), 0.0, 1.0)
    met_sym = calc_two_metrics(m_sym, gt)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG_DARK)
    
    # Left: Continuous S_sym map
    ax0 = axes[0]
    ax0.set_facecolor("#030712")
    im0 = ax0.imshow(s_sym_plot, origin="lower", extent=extent, aspect="auto", cmap="viridis_r", vmin=0.0, vmax=0.5)
    ax0.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=1.8, linestyles="--")
    ax0.set_title(r"Continuous Symmetry Balance Ratio $S_{\mathrm{sym}}(\mu, V_z)$", fontsize=11.5, fontweight="bold", color="#f8fafc", pad=8)
    ax0.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11, color="white")
    ax0.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11, color="white")
    ax0.tick_params(colors="white")
    ax0.grid(True, linestyle=":", alpha=0.25, color="white")
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label(r"$S_{\mathrm{sym}} = |G_{LL}-G_{RR}|/(G_{LL}+G_{RR})$", color="white", fontsize=10)
    cbar0.ax.tick_params(colors="white")
    
    # Right: Binarized S_sym <= 0.15
    ax1 = axes[1]
    ax1.set_facecolor("#030712")
    img_sym_rgba = np.zeros((len(mu_vals), len(vz_vals), 4))
    img_sym_rgba[~gt & (~m_sym)] = [0.04, 0.07, 0.13, 1.0]
    img_sym_rgba[gt & (~m_sym)] = [0.12, 0.16, 0.23, 0.95]
    img_sym_rgba[m_sym] = [0.96, 0.62, 0.04, 0.95]  # amber/gold
    
    ax1.imshow(img_sym_rgba, origin="lower", extent=extent, aspect="auto")
    ax1.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=1.8, linestyles="--")
    sub_title_sym = (
        r"Binarized Lead Symmetry Balance Gate ($S_{\mathrm{sym}} \leq 0.15$)"
        f"\nPfaffian Area Passed: {met_sym['pfaffian_passed_pct']:.2f}% | FDR: {met_sym['fdr_percent']:.2f}%"
    )
    ax1.set_title(sub_title_sym, fontsize=11.5, fontweight="bold", color="#f8fafc", pad=8)
    ax1.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11, color="white")
    ax1.tick_params(colors="white")
    ax1.grid(True, linestyle=":", alpha=0.25, color="white")
    
    legend_sym = [
        mlines.Line2D([0], [0], color=CYAN_PFAFF, lw=1.8, linestyle="--", label=r"Pfaffian Boundary ($\mathcal{Q} = -1$)"),
        mpatches.Patch(facecolor="#1e293b", edgecolor="#475569", label="Pfaffian Bulk (Unpassed)"),
        mpatches.Patch(facecolor="#f59e0b", label=r"Passed Pixels ($S_{\mathrm{sym}} \leq 0.15$)"),
    ]
    ax1.legend(handles=legend_sym, loc="upper left", fontsize=8.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
    
    fig.suptitle(r"Conjugate Lead Symmetry Balance at $T = 40$ mK ($V_0 = 1.2$ meV, Realization 10)", fontsize=13, fontweight="bold", color=CYAN_PFAFF, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG_DIR / "fig4_symmetry_balance_phase_map_40mK.png", dpi=200)
    plt.close(fig)
    logger.info("Saved fig4_symmetry_balance_phase_map_40mK.png")


def generate_step_by_step_comparison_figures():
    logger.info("Generating Step-by-Step 4-Panel Comparison Figures for V0 = 0.378, 0.645, 0.872, 1.200 meV...")
    proto_cls = load_protocol_class_from_file(REPO_ROOT / "Protocols" / "MSFT_TGP_Hybrids" / "protocol.py")
    proto = proto_cls()

    cfg_path = REPO_ROOT / "Outputs/MSFT_Hybrids_Test_Raw/MSFT_TGP_Hybrids/tgp_bnd000_lowcorr075_sym015/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    params = cfg["free_parameters"]

    v0_info = [
        ("V0_0_378", 0.378, "Weak Disorder $V_0 = 0.378$ meV", "fig5_step_by_step_V0_0_378.png"),
        ("V0_0_645", 0.645, "Intermediate Disorder $V_0 = 0.645$ meV", "fig6_step_by_step_V0_0_645.png"),
        ("V0_0_872", 0.872, "Strong Disorder $V_0 = 0.872$ meV", "fig7_step_by_step_V0_0_872.png"),
        ("V0_1_2",   1.200, "Severe Benchmark Disorder $V_0 = 1.200$ meV", "fig8_step_by_step_V0_1_2.png"),
    ]

    all_step_metrics = {}

    for v0_lbl, v0_val, v0_title, out_filename in v0_info:
        real_dir = DATA_ROOT / v0_lbl / "disorder_realization_10"
        cache = RealizationFeatureCache(real_dir, temperature_mK=40.0, v0_label=v0_lbl)
        steps = proto.evaluate_steps(cache, params)
        gt = cache.pfaffian_mask
        mu_vals = cache.mu_vals
        vz_vals = cache.vz_vals
        extent = [vz_vals[0], vz_vals[-1], mu_vals[0], mu_vals[-1]]

        m_tgp = steps["step_3_tgp_stage2_candidate"]
        m_s1 = steps["step_4_fpe_low_asym_corr"]
        m_s2 = m_s1 & steps["step_5_fpe_lead_symmetry_balance"]
        m_s3 = steps["step_8_dpe_C4_final_mask"]

        met_tgp = calc_two_metrics(m_tgp, gt)
        met_s1 = calc_two_metrics(m_s1, gt)
        met_s2 = calc_two_metrics(m_s2, gt)
        met_s3 = calc_two_metrics(m_s3, gt)

        all_step_metrics[v0_lbl] = {
            "v0_title": v0_title,
            "v0_mev": v0_val,
            "regular_tgp": met_tgp,
            "step1_lowcorr": met_s1,
            "step2_corr_sym": met_s2,
            "step3_full_c4": met_s3,
        }

        # 4-panel figure matching template Slides 2-7
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.8), sharex=True, sharey=True)
        fig.patch.set_facecolor(BG_DARK)

        panels = [
            (axes[0, 0], m_tgp, "#a855f7", "Regular TGP (Stage 2 ROI 2)", met_tgp),
            (axes[0, 1], m_s1, "#0284c7", r"Step 1: Low-Asym Corr ($C_L, C_R \geq 0.90$)", met_s1),
            (axes[1, 0], m_s2, "#f59e0b", r"Step 2: Corr + Symmetry Balance ($S_{\mathrm{sym}} \leq 0.15$)", met_s2),
            (axes[1, 1], m_s3, "#10b981", r"Step 3: Full Protocol ($C_4$ Canonical Filter)", met_s3),
        ]

        for ax, mask, color, title, m_dict in panels:
            ax.set_facecolor("#030712")
            img_rgba = np.zeros((len(mu_vals), len(vz_vals), 4))
            img_rgba[~gt & (~mask)] = [0.04, 0.07, 0.13, 1.0]
            img_rgba[gt & (~mask)] = [0.12, 0.16, 0.23, 0.95]
            rgb = mcolors.to_rgb(color)
            img_rgba[mask] = [rgb[0], rgb[1], rgb[2], 0.95]

            ax.imshow(img_rgba, origin="lower", extent=extent, aspect="auto")
            ax.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=1.6, linestyles="--")

            sub_title = f"{title}\nPfaffian Passed: {m_dict['pfaffian_passed_pct']:.2f}% | FDR: {m_dict['fdr_percent']:.2f}%"
            ax.set_title(sub_title, fontsize=10.5, fontweight="bold", pad=8, color="#f1f5f9")
            ax.grid(True, linestyle=":", alpha=0.25, color="white")
            ax.tick_params(colors="white")

        axes[0, 0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10.5, color="white")
        axes[1, 0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10.5, color="white")
        axes[1, 0].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10.5, color="white")
        axes[1, 1].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10.5, color="white")

        legend_elements = [
            mlines.Line2D([0], [0], color=CYAN_PFAFF, lw=1.8, linestyle="--", label=r"Pfaffian Boundary ($\mathcal{Q} = -1$)"),
            mpatches.Patch(facecolor="#1e293b", edgecolor="#475569", label="Pfaffian Bulk (Unpassed)"),
            mpatches.Patch(facecolor="#a855f7", label="Regular TGP Passed"),
            mpatches.Patch(facecolor="#0284c7", label="Step 1: Low-Asym Corr"),
            mpatches.Patch(facecolor="#f59e0b", label="Step 2: Corr + SymBal"),
            mpatches.Patch(facecolor="#10b981", label="Step 3: Full Protocol (C4)"),
        ]
        axes[0, 0].legend(handles=legend_elements, loc="upper left", fontsize=7.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")

        fig.suptitle(f"Step-by-Step Protocol Elimination: {v0_title} at $T = 40$ mK", fontsize=12.5, fontweight="bold", color=CYAN_PFAFF, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT_FIG_DIR / out_filename, dpi=200)
        plt.close(fig)
        logger.info("Saved %s", out_filename)

    with open(OUT_FIG_DIR / "step_by_step_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_step_metrics, f, indent=2)
    logger.info("Saved step_by_step_metrics.json")
    return all_step_metrics


def generate_cluster_phasemaps_and_cuts():
    logger.info("Generating Phase Maps with Clusters and Full Cut Analytics for all surviving clusters...")
    from scipy.ndimage import label as nd_label

    proto_cls = load_protocol_class_from_file(REPO_ROOT / "Protocols" / "MSFT_TGP_Hybrids" / "protocol.py")
    proto = proto_cls()

    cfg_path = REPO_ROOT / "Outputs/MSFT_Hybrids_Test_Raw/MSFT_TGP_Hybrids/tgp_bnd000_lowcorr075_sym015/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    params = cfg["free_parameters"]

    cluster_targets = [
        ("V0_0_378", "Weak Disorder $V_0 = 0.378$ meV"),
        ("V0_0_645", "Intermediate Disorder $V_0 = 0.645$ meV"),
        ("V0_0_872", "Strong Disorder $V_0 = 0.872$ meV"),
        ("V0_1_2",   "Severe Benchmark Disorder $V_0 = 1.200$ meV"),
    ]

    cluster_catalog = {}

    for v0_lbl, v0_title in cluster_targets:
        real_dir = DATA_ROOT / v0_lbl / "disorder_realization_10"
        cache = RealizationFeatureCache(real_dir, temperature_mK=40.0, v0_label=v0_lbl)
        steps = proto.evaluate_steps(cache, params)
        gt = cache.pfaffian_mask
        mu_vals = cache.mu_vals
        vz_vals = cache.vz_vals
        Nmu, Nvz = len(mu_vals), len(vz_vals)
        extent = [vz_vals[0], vz_vals[-1], mu_vals[0], mu_vals[-1]]

        m_tgp = steps["step_3_tgp_stage2_candidate"]
        m_final = steps["step_8_dpe_C4_final_mask"]
        
        # Load precomputed simulation observables for cuts
        all_p = np.load(real_dir / "all_params.npz", allow_pickle=True)
        t_val = float(all_p["t"])
        gamma = float(all_p["gamma"])
        Delta0 = float(all_p["Delta0"])
        alpha = float(all_p["alpha"])
        Ls = int(all_p["Ls"])
        Vdisx = np.load(real_dir / "Vdisx.npy")

        # Conductance 4D tensors for cuts (Nmu, Nvz, 5 cutters, 97 bias points)
        d_ll = np.load(real_dir / "tgp_stage1_dIdVl.npy").reshape(Nmu, Nvz, 5, 97)
        d_rr = np.load(real_dir / "tgp_stage1_dIdVr.npy").reshape(Nmu, Nvz, 5, 97)
        d_rl = np.load(real_dir / "tgp_stage1_dIdV_RL.npy").reshape(Nmu, Nvz, 5, 97)
        d_lr = np.load(real_dir / "tgp_stage1_dIdV_LR.npy").reshape(Nmu, Nvz, 5, 97)
        bias_grid = cache.bias_grid

        # 3w curvature and transport gap
        d2G_L, d2G_R = extract_zero_bias_curvature_3d(cache)
        c3w_2d = 0.5 * (d2G_L.mean(axis=0) + d2G_R.mean(axis=0))
        gap_l_3d, gap_r_3d, gap_min_2d = extract_nonlocal_transport_gap_maps(cache)
        gap_2d = gap_min_2d
        gap_l_2d = gap_l_3d.mean(axis=0)
        gap_r_2d = gap_r_3d.mean(axis=0)

        # Correlation and symmetry balance continuous tracks
        cL_low, cR_low = extract_barrier_reflection_correlations(cache, asym_min=-5.0, asym_max=5.0)
        c_low_2d = np.minimum(cL_low, cR_low)
        m_sym_2d = fpe_lead_symmetry_balance(cache, s_max=0.15)
        s_sym_2d = cache._memo[("lead_symmetry_balance_obs", 0.0001, 0.0)]
        mzm_conf_2d = cache.boundary_confinement
        mzm_sep_2d = cache.mzm_separability

        # Barrier conductances at 40 mK for deep dive
        bl_l = np.load(real_dir / "barrier_left_conductance_left_arr_40mK.npy")
        br_r = np.load(real_dir / "barrier_right_conductance_right_arr_40mK.npy")
        asym_ratios = np.load(real_dir / "asym_ratios.npy")

        # Cluster labeling and filtering for area >= 10 px
        lbl_arr, num_isl = nd_label(m_final, structure=np.ones((3, 3), dtype=int))
        logger.info("[%s] Found %d raw clusters", v0_lbl, num_isl)

        clusters_info = []
        retained_id = 1
        for raw_isl_idx in range(1, num_isl + 1):
            pts = np.argwhere(lbl_arr == raw_isl_idx)
            sz = len(pts)
            if sz < 10:
                logger.info("[%s] Pruning raw cluster %d: area %d px < 10 px", v0_lbl, raw_isl_idx, sz)
                continue

            com_mu = float(np.mean(mu_vals[pts[:, 0]]))
            com_vz = float(np.mean(vz_vals[pts[:, 1]]))

            # Find medoid: passing topological pixel closest to COM
            topo_pts = [p for p in pts if gt[p[0], p[1]]]
            if len(topo_pts) > 0:
                cand_pts = topo_pts
            else:
                all_topo = np.argwhere(gt)
                cand_pts = all_topo if len(all_topo) > 0 else pts
            dists = [(mu_vals[p[0]] - com_mu)**2 + (vz_vals[p[1]] - com_vz)**2 for p in cand_pts]
            best_p = cand_pts[int(np.argmin(dists))]
            med_mu_idx, med_vz_idx = int(best_p[0]), int(best_p[1])
            med_mu = float(mu_vals[med_mu_idx])
            med_vz = float(vz_vals[med_vz_idx])

            # Compute MZM boundary confinement and separability for the medoid
            syst_pt = hp.build_system_closed(t_val, med_mu, gamma, Delta0, med_vz, alpha, Ls, Vdisx, a=1)
            ev_pt, ec_pt = hp.solve_ham(syst_pt, k=2, solver_type="cpu")
            rho1, rho2, _ = hp.get_psiM_density(ev_pt, ec_pt)
            conf_val = float(hp.calc_mzm_boundary_confinement(rho1, rho2, Ls))
            sep_val = float(hp.calc_mzm_separability(rho1, rho2))

            clusters_info.append({
                "island_id": retained_id,
                "raw_island_id": raw_isl_idx,
                "area_px": sz,
                "com_mu": com_mu,
                "com_vz": com_vz,
                "med_mu": med_mu,
                "med_vz": med_vz,
                "med_mu_idx": med_mu_idx,
                "med_vz_idx": med_vz_idx,
                "confinement_C": conf_val,
                "separability_S": sep_val,
            })
            retained_id += 1

        cluster_catalog[v0_lbl] = {
            "v0_title": v0_title,
            "num_clusters": len(clusters_info),
            "clusters": clusters_info,
        }

        # --- Phase Map of Surviving Clusters ---
        fig_pm, ax_pm = plt.subplots(figsize=(8.5, 6.2))
        fig_pm.patch.set_facecolor(BG_DARK)
        ax_pm.set_facecolor("#030712")

        img_pm = np.zeros((Nmu, Nvz, 4))
        img_pm[~gt & (~m_final)] = [0.04, 0.07, 0.13, 1.0]
        img_pm[gt & (~m_final)] = [0.12, 0.16, 0.23, 0.95]
        
        star_palette = ["#00f5d4", "#f72585", "#ffbe0b", "#4cc9f0", "#a78bfa", "#f97316", "#10b981", "#e11d48"]
        for c_entry in clusters_info:
            c_mask = (lbl_arr == c_entry["raw_island_id"])
            col_rgb = mcolors.to_rgb(star_palette[(c_entry["island_id"] - 1) % len(star_palette)])
            img_pm[c_mask] = [col_rgb[0], col_rgb[1], col_rgb[2], 0.95]

        ax_pm.imshow(img_pm, origin="lower", extent=extent, aspect="auto")
        ax_pm.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=1.8, linestyles="--")

        # Mark clusters and COM stars
        for c_entry in clusters_info:
            c_col = star_palette[(c_entry["island_id"] - 1) % len(star_palette)]
            ax_pm.plot(c_entry["med_vz"], c_entry["med_mu"], marker="*", color=c_col, markersize=14, markeredgecolor="white", markeredgewidth=1.2, zorder=10)
            ax_pm.annotate(f"Island {c_entry['island_id']}\n({c_entry['area_px']} px)\nC = {c_entry['confinement_C']:.2f}",
                           xy=(c_entry["med_vz"], c_entry["med_mu"]),
                           xytext=(c_entry["med_vz"] + 0.04, c_entry["med_mu"] + 0.12),
                           color="white", fontsize=8.5, fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.25", facecolor=BG_CARD, edgecolor=c_col, alpha=0.9),
                           arrowprops=dict(arrowstyle="->", color=c_col, lw=1.2))

        ax_pm.set_title(f"Protocol `tgp_bnd000_lowcorr075_sym015` Clusters: {v0_title} (T = 40 mK)", fontsize=11.5, fontweight="bold", color="#f8fafc", pad=8)
        ax_pm.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10.5, color="white")
        ax_pm.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10.5, color="white")
        ax_pm.tick_params(colors="white")
        ax_pm.grid(True, linestyle=":", alpha=0.25, color="white")

        pm_filename = f"phasemap_clusters_{v0_lbl}.png"
        fig_pm.tight_layout()
        fig_pm.savefig(OUT_FIG_DIR / pm_filename, dpi=200)
        plt.close(fig_pm)
        logger.info("Saved %s", pm_filename)

        # --- Generate Cuts for Each Surviving Cluster ---
        for c_entry in clusters_info:
            isl_id = c_entry["island_id"]
            mu_cut = c_entry["med_mu"]
            mu_idx = c_entry["med_mu_idx"]
            vz_med = c_entry["med_vz"]
            vz_idx = c_entry["med_vz_idx"]
            star_col = star_palette[(isl_id - 1) % len(star_palette)]

            logger.info("Solving BdG spectrum along cut mu = %.3f meV for %s Island %d...", mu_cut, v0_lbl, isl_id)
            n_evals_cut = 14
            evals_cut = np.zeros((Nvz, n_evals_cut))
            for j in range(Nvz):
                syst_j = hp.build_system_closed(t_val, mu_cut, gamma, Delta0, vz_vals[j], alpha, Ls, Vdisx, a=1)
                ev_j, _ = hp.solve_ham(syst_j, k=n_evals_cut, solver_type="cpu")
                evals_cut[j, :] = np.sort(ev_j)

            # 1. Standalone Spectra along cut
            fig_sp, ax_sp = plt.subplots(figsize=(7.5, 4.2))
            fig_sp.patch.set_facecolor(BG_DARK)
            ax_sp.set_facecolor("#030712")
            
            gt_cut = gt[mu_idx, :]
            for j in range(Nvz):
                if gt_cut[j]:
                    dvz = (vz_vals[-1] - vz_vals[0]) / (Nvz - 1)
                    ax_sp.axvspan(vz_vals[j] - 0.5 * dvz, vz_vals[j] + 0.5 * dvz, color=DARK_SLATE, alpha=0.75, zorder=0)

            mid_ev = n_evals_cut // 2
            for ev_i in range(n_evals_cut):
                col_ev = "#ef4444" if ev_i in [mid_ev - 1, mid_ev] else CYAN_PFAFF
                lw_ev = 2.2 if ev_i in [mid_ev - 1, mid_ev] else 1.1
                ax_sp.plot(vz_vals, evals_cut[:, ev_i], color=col_ev, lw=lw_ev, alpha=0.85, zorder=2)

            ax_sp.axhline(0.0, color="white", lw=0.8, linestyle=":", alpha=0.6, zorder=3)
            ax_sp.axvline(vz_med, color=star_col, linestyle="--", lw=1.5, alpha=0.9, zorder=4)
            ax_sp.plot(vz_med, 0.0, marker="*", color=star_col, markersize=14, markeredgecolor="black", markeredgewidth=1.2, zorder=10)

            ax_sp.set_ylim(-0.16, 0.16)
            ax_sp.set_xlim(vz_vals[0], vz_vals[-1])
            ax_sp.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10, color="white")
            ax_sp.set_ylabel("Energy (meV)", fontsize=10, color="white")
            ax_sp.set_title(f"BdG Energy Spectrum along Cut ($\mu = {mu_cut:.3f}$ meV)\n{v0_title} — Island {isl_id}", fontsize=11, fontweight="bold", color=CYAN_PFAFF, pad=6)
            ax_sp.tick_params(colors="white")
            ax_sp.grid(True, linestyle=":", alpha=0.25, color="white")

            spec_filename = f"cut_spectra_{v0_lbl}_isl{isl_id}.png"
            fig_sp.tight_layout()
            fig_sp.savefig(OUT_FIG_DIR / spec_filename, dpi=200)
            plt.close(fig_sp)
            logger.info("Saved %s", spec_filename)

            # 2. 5-Panel Conductance Plot along Cut
            fig_cond, axes_c = plt.subplots(5, 1, figsize=(8.5, 10.5), sharex=True, gridspec_kw={"height_ratios": [1.8, 1.0, 1.0, 1.0, 1.0], "hspace": 0.18})
            fig_cond.patch.set_facecolor(BG_DARK)

            # Panel 0: Spectrum
            ax0 = axes_c[0]
            ax0.set_facecolor("#030712")
            for j in range(Nvz):
                if gt_cut[j]:
                    dvz = (vz_vals[-1] - vz_vals[0]) / (Nvz - 1)
                    ax0.axvspan(vz_vals[j] - 0.5 * dvz, vz_vals[j] + 0.5 * dvz, color=DARK_SLATE, alpha=0.75, zorder=0)
            for ev_i in range(n_evals_cut):
                col_ev = "#ef4444" if ev_i in [mid_ev - 1, mid_ev] else CYAN_PFAFF
                lw_ev = 2.0 if ev_i in [mid_ev - 1, mid_ev] else 1.0
                ax0.plot(vz_vals, evals_cut[:, ev_i], color=col_ev, lw=lw_ev, alpha=0.85, zorder=2)
            ax0.axvline(vz_med, color=star_col, linestyle="--", lw=1.4, alpha=0.9, zorder=4)
            ax0.plot(vz_med, 0.0, marker="*", color=star_col, markersize=12, markeredgecolor="black", markeredgewidth=1.0, zorder=10)
            ax0.set_ylim(-0.16, 0.16)
            ax0.set_ylabel("Energy (meV)", fontsize=9, color="white")
            ax0.set_title(f"Spectra & Conductance along Cluster Cut ($\mu = {mu_cut:.3f}$ meV, Island {isl_id})", fontsize=11, fontweight="bold", color=CYAN_PFAFF, pad=6)
            ax0.tick_params(colors="white")
            ax0.grid(True, linestyle=":", alpha=0.25, color="white")

            # Extract 2D slices along cut: (Nvz, 97 bias)
            # Transpose to (97 bias, Nvz) so x=Vz, y=bias in µV
            bias_uV = bias_grid * 1000.0
            extent_cut_uV = [vz_vals[0], vz_vals[-1], bias_uV[0], bias_uV[-1]]
            gll_cut = d_ll[mu_idx, :, 0, :].T
            grr_cut = d_rr[mu_idx, :, 0, :].T
            grl_cut = d_rl[mu_idx, :, 0, :].T
            glr_cut = d_lr[mu_idx, :, 0, :].T
            agrl_cut = 0.5 * (grl_cut - grl_cut[::-1, :])
            aglr_cut = 0.5 * (glr_cut - glr_cut[::-1, :])

            cond_maps = [
                (axes_c[1], gll_cut, r"$G_{LL}(V, V_z)$ [$e^2/h$]", "viridis", 0.0, 2.0, r"Left bias [$\mu$V]"),
                (axes_c[2], grr_cut, r"$G_{RR}(V, V_z)$ [$e^2/h$]", "viridis", 0.0, 2.0, r"Right bias [$\mu$V]"),
                (axes_c[3], agrl_cut, r"$\mathcal{A}(G_{RL})$ [$e^2/h$]", "PuOr_r", -0.10, 0.10, r"Left bias [$\mu$V]"),
                (axes_c[4], aglr_cut, r"$\mathcal{A}(G_{LR})$ [$e^2/h$]", "PuOr_r", -0.10, 0.10, r"Right bias [$\mu$V]"),
            ]

            for ax_m, cdata, lbl_m, cmap_m, vmin_m, vmax_m, ylbl in cond_maps:
                ax_m.set_facecolor("#030712")
                im_m = ax_m.imshow(cdata, origin="lower", extent=extent_cut_uV, aspect="auto", cmap=cmap_m, vmin=vmin_m, vmax=vmax_m)
                ax_m.axvline(vz_med, color=star_col, linestyle="--", lw=1.2, alpha=0.8, zorder=4)
                ax_m.set_ylabel(ylbl, fontsize=8.5, color="white")
                ax_m.tick_params(colors="white")
                div_m = make_axes_locatable(ax_m)
                cax_m = div_m.append_axes("right", size="2.5%", pad=0.08)
                cb_m = fig_cond.colorbar(im_m, cax=cax_m)
                cb_m.set_label(lbl_m, color="white", fontsize=8)
                cb_m.ax.tick_params(colors="white")

            # Overlay solid black lines indicating extracted transport gap on panels 3 & 4
            max_b = np.max(np.abs(bias_uV))
            gap_l_cut_uV = gap_l_2d[mu_idx, :] * 1000.0
            valid_l = (gap_l_cut_uV > 0) & (gap_l_cut_uV <= max_b)
            if np.any(valid_l):
                gap_plot_l = np.where(valid_l, gap_l_cut_uV, np.nan)
                axes_c[3].plot(vz_vals, gap_plot_l, color="black", lw=1.6, label=r"$\pm \Delta_{\mathrm{trans}}$")
                axes_c[3].plot(vz_vals, -gap_plot_l, color="black", lw=1.6)
                axes_c[3].legend(loc="upper right", fontsize=7.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")

            gap_r_cut_uV = gap_r_2d[mu_idx, :] * 1000.0
            valid_r = (gap_r_cut_uV > 0) & (gap_r_cut_uV <= max_b)
            if np.any(valid_r):
                gap_plot_r = np.where(valid_r, gap_r_cut_uV, np.nan)
                axes_c[4].plot(vz_vals, gap_plot_r, color="black", lw=1.6, label=r"$\pm \Delta_{\mathrm{trans}}$")
                axes_c[4].plot(vz_vals, -gap_plot_r, color="black", lw=1.6)
                axes_c[4].legend(loc="upper right", fontsize=7.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")

            axes_c[4].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10, color="white")
            fig_cond.tight_layout()
            cond_filename = f"cut_conductance_5panel_{v0_lbl}_isl{isl_id}.png"
            fig_cond.savefig(OUT_FIG_DIR / cond_filename, dpi=200)
            plt.close(fig_cond)
            logger.info("Saved %s", cond_filename)

            # 3. Cut Analytics Multiplot (6 Tracks: Spectrum, 3w, Gap, Low-Asym Corr, SymBal, MZM Confinement & Separability)
            fig_an, axes_a = plt.subplots(6, 1, figsize=(8.5, 12.5), sharex=True, gridspec_kw={"height_ratios": [1.8, 0.9, 0.9, 0.8, 0.8, 0.8], "hspace": 0.22})
            fig_an.patch.set_facecolor(BG_DARK)

            # Panel 0: BdG Spectrum with protocol runner ticks
            ax_a0 = axes_a[0]
            ax_a0.set_facecolor("#030712")
            for j in range(Nvz):
                if gt_cut[j]:
                    dvz = (vz_vals[-1] - vz_vals[0]) / (Nvz - 1)
                    ax_a0.axvspan(vz_vals[j] - 0.5 * dvz, vz_vals[j] + 0.5 * dvz, color=DARK_SLATE, alpha=0.75, zorder=0)
            for ev_i in range(n_evals_cut):
                col_ev = "#ef4444" if ev_i in [mid_ev - 1, mid_ev] else CYAN_PFAFF
                lw_ev = 2.2 if ev_i in [mid_ev - 1, mid_ev] else 1.1
                ax_a0.plot(vz_vals, evals_cut[:, ev_i], color=col_ev, lw=lw_ev, alpha=0.85, zorder=2)
            ax_a0.axvline(vz_med, color=star_col, linestyle="--", lw=1.5, alpha=0.9, zorder=4)
            ax_a0.plot(vz_med, 0.0, marker="*", color=star_col, markersize=14, markeredgecolor="black", markeredgewidth=1.2, zorder=10, label=f"COM Medoid (Island {isl_id})")

            # Runner ticks for Regular TGP and New Protocol
            tgp_cut_m = m_tgp[mu_idx, :]
            c4_cut_m = m_final[mu_idx, :]
            ymin_a = -0.16
            ax_a0.plot(vz_vals[tgp_cut_m], np.full(np.sum(tgp_cut_m), ymin_a + 0.015), "|", color="#a855f7", markersize=8, markeredgewidth=1.5, label="Regular TGP (ROI 2)")
            ax_a0.plot(vz_vals[c4_cut_m], np.full(np.sum(c4_cut_m), ymin_a + 0.025), "|", color="#10b981", markersize=10, markeredgewidth=2.2, label=r"New Protocol ($C_4$)")

            ax_a0.set_ylim(-0.16, 0.16)
            ax_a0.set_ylabel("Energy (meV)", fontsize=9.5, color="white")
            ax_a0.set_title(f"Cut Analytics through Island {isl_id}: {v0_title} ($\mu = {mu_cut:.3f}$ meV)", fontsize=11.5, fontweight="bold", color=CYAN_PFAFF, pad=6)
            ax_a0.legend(loc="upper right", fontsize=8, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
            ax_a0.tick_params(colors="white")
            ax_a0.grid(True, linestyle=":", alpha=0.25, color="white")

            # Panel 1: 3w Curvature
            ax_a1 = axes_a[1]
            ax_a1.set_facecolor("#030712")
            c3w_cut = c3w_2d[mu_idx, :]
            ax_a1.plot(vz_vals, np.clip(c3w_cut, -150.0, 50.0), color="#f97316", lw=1.6, label=r"$3\omega$ Curvature $\partial_V^2 G$")
            ax_a1.axhline(-100.0, color="white", linestyle=":", lw=1.2, label="ZBP Curvature Threshold (-100)")
            ax_a1.axvline(vz_med, color=star_col, linestyle="--", lw=1.2, alpha=0.8)
            ax_a1.set_ylabel(r"$3\omega$ ($e^2/h/\mathrm{meV}^2$)", fontsize=8.5, color="white")
            ax_a1.legend(loc="upper right", fontsize=7.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
            ax_a1.tick_params(colors="white")
            ax_a1.grid(True, linestyle=":", alpha=0.25, color="white")

            # Panel 2: Transport Gap vs Lowest State
            ax_a2 = axes_a[2]
            ax_a2.set_facecolor("#030712")
            gap_cut = gap_2d[mu_idx, :]
            ax_a2.plot(vz_vals, gap_cut * 1000.0, color="#a855f7", lw=1.6, label=r"Transport Gap $\Delta_{\mathrm{trans}}$ ($\mu$eV)")
            ax_a2.plot(vz_vals, np.abs(evals_cut[:, mid_ev]) * 1000.0, color="#ef4444", lw=1.4, linestyle="--", label=r"Lowest State $|E_0|$ ($\mu$eV)")
            ax_a2.axhline(10.0, color="white", linestyle="--", lw=1.1, label=r"Gap Floor ($10\ \mu\mathrm{eV}$)")
            ax_a2.axvline(vz_med, color=star_col, linestyle="--", lw=1.2, alpha=0.8)
            ax_a2.set_ylabel(r"Gap ($\mu$eV)", fontsize=8.5, color="white")
            ax_a2.legend(loc="upper right", fontsize=7.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
            ax_a2.tick_params(colors="white")
            ax_a2.grid(True, linestyle=":", alpha=0.25, color="white")

            # Panel 3: Low-Asymmetry Reflection Correlation Track
            ax_a3 = axes_a[3]
            ax_a3.set_facecolor("#030712")
            corr_cut = c_low_2d[mu_idx, :]
            ax_a3.plot(vz_vals, corr_cut, color="#0284c7", lw=1.8, label=r"Low-Asym Corr $\min(C_L, C_R)$ ([-5, +5] meV)")
            ax_a3.axhline(0.75, color="#38bdf8", linestyle="--", lw=1.2, label=r"Threshold $\theta_{\mathrm{corr}} = 0.75$")
            ax_a3.axvline(vz_med, color=star_col, linestyle="--", lw=1.2, alpha=0.8)
            ax_a3.set_ylim(-0.1, 1.05)
            ax_a3.set_ylabel("Corr $C$", fontsize=8.5, color="white")
            ax_a3.legend(loc="lower right", fontsize=7.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
            ax_a3.tick_params(colors="white")
            ax_a3.grid(True, linestyle=":", alpha=0.25, color="white")

            # Panel 4: Symmetry Balance Ratio Track
            ax_a4 = axes_a[4]
            ax_a4.set_facecolor("#030712")
            sym_cut = np.clip(np.nan_to_num(s_sym_2d[mu_idx, :], nan=1.0, posinf=1.0), 0.0, 1.0)
            ax_a4.plot(vz_vals, sym_cut, color="#f59e0b", lw=1.8, label=r"Symmetry Balance $S_{\mathrm{sym}} = |G_{LL}-G_{RR}|/\Sigma G$")
            ax_a4.axhline(0.15, color="#fbbf24", linestyle="--", lw=1.2, label=r"Bound $s_{\max} = 0.15$")
            ax_a4.axvline(vz_med, color=star_col, linestyle="--", lw=1.2, alpha=0.8)
            ax_a4.set_ylim(-0.05, 0.8)
            ax_a4.set_ylabel(r"$S_{\mathrm{sym}}$", fontsize=8.5, color="white")
            ax_a4.legend(loc="upper right", fontsize=7.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
            ax_a4.tick_params(colors="white")
            ax_a4.grid(True, linestyle=":", alpha=0.25, color="white")

            # Panel 5: MZM Boundary Confinement & Separability Track
            ax_a5 = axes_a[5]
            ax_a5.set_facecolor("#030712")
            conf_cut = mzm_conf_2d[mu_idx, :]
            sep_cut = mzm_sep_2d[mu_idx, :]
            ax_a5.plot(vz_vals, conf_cut, color="#10b981", lw=1.8, label=r"Boundary Confinement $\mathcal{C}$")
            ax_a5.plot(vz_vals, sep_cut, color="#06b6d4", lw=1.6, linestyle="--", label=r"MZM Separability $\mathcal{S}$")
            ax_a5.plot(vz_med, c_entry["confinement_C"], marker="*", color=star_col, markersize=12, markeredgecolor="black", zorder=5)
            ax_a5.set_ylim(-0.05, 1.05)
            ax_a5.set_ylabel(r"$\mathcal{C}, \mathcal{S}$", fontsize=8.5, color="white")
            ax_a5.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10, color="white")
            ax_a5.legend(loc="upper right", fontsize=7.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
            ax_a5.tick_params(colors="white")
            ax_a5.grid(True, linestyle=":", alpha=0.25, color="white")

            fig_an.tight_layout()
            an_filename = f"cut_analytics_{v0_lbl}_isl{isl_id}.png"
            fig_an.savefig(OUT_FIG_DIR / an_filename, dpi=200)
            plt.close(fig_an)
            logger.info("Saved %s", an_filename)

            # 4. Point Deep-Dive at COM Medoid
            # a. dI/dV vs Bias
            didv_l_pt = d_ll[mu_idx, vz_idx, 0, :]
            didv_r_pt = d_rr[mu_idx, vz_idx, 0, :]
            fig_pt, ax_pt = plt.subplots(figsize=(5.5, 3.8))
            fig_pt.patch.set_facecolor(BG_DARK)
            ax_pt.set_facecolor("#030712")
            ax_pt.plot(bias_grid * 1000.0, didv_l_pt, color="#38bdf8", lw=1.8, label=r"$dI/dV_L$")
            ax_pt.plot(bias_grid * 1000.0, didv_r_pt, color="#f43f5e", lw=1.8, linestyle="--", label=r"$dI/dV_R$")
            ax_pt.set_title(f"Zero-Bias Peak at Medoid ($\mu={mu_cut:.3f}, V_z={vz_med:.3f}$ meV)", fontsize=10, fontweight="bold", color="white")
            ax_pt.set_xlabel(r"Bias Voltage $V$ ($\mu$eV)", fontsize=9, color="white")
            ax_pt.set_ylabel(r"Conductance ($2e^2/h$)", fontsize=9, color="white")
            ax_pt.legend(loc="upper right", fontsize=8, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
            ax_pt.tick_params(colors="white")
            ax_pt.grid(True, linestyle=":", alpha=0.25, color="white")
            pt_didv_filename = f"deepdive_dIdV_{v0_lbl}_isl{isl_id}.png"
            fig_pt.tight_layout()
            fig_pt.savefig(OUT_FIG_DIR / pt_didv_filename, dpi=200)
            plt.close(fig_pt)

            # b. Majorana Wavefunctions
            syst_pt = hp.build_system_closed(t_val, mu_cut, gamma, Delta0, vz_med, alpha, Ls, Vdisx, a=1)
            ev_pt, ec_pt = hp.solve_ham(syst_pt, k=2, solver_type="cpu")
            rho1, rho2, _ = hp.get_psiM_density(ev_pt, ec_pt)
            conf_val = hp.calc_mzm_boundary_confinement(rho1, rho2, Ls)
            sep_val = hp.calc_mzm_separability(rho1, rho2)

            fig_wf, ax_wf = plt.subplots(figsize=(5.5, 3.8))
            fig_wf.patch.set_facecolor(BG_DARK)
            ax_wf.set_facecolor("#030712")
            x_sites = np.arange(Ls)
            ax_wf.plot(x_sites, rho1, color="#38bdf8", lw=1.8, label=r"$|\psi_L|^2$ (Left MZM)")
            ax_wf.plot(x_sites, rho2, color="#f43f5e", lw=1.8, label=r"$|\psi_R|^2$ (Right MZM)")
            sub_wf = f"Majorana Wavefunctions ($\mathcal{{C}} = {conf_val:.3f}, \mathcal{{S}} = {sep_val:.3f}$)"
            ax_wf.set_title(sub_wf, fontsize=10, fontweight="bold", color="white")
            ax_wf.set_xlabel("Wire Site $x$", fontsize=9, color="white")
            ax_wf.set_ylabel("Probability Density", fontsize=9, color="white")
            ax_wf.text(0.04, 0.88, f"Boundary Confinement: $\mathcal{{C}} = {conf_val:.3f}$\nMZM Separability: $\mathcal{{S}} = {sep_val:.3f}$",
                       transform=ax_wf.transAxes, fontsize=8.5, color="white",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_CARD, edgecolor="#38bdf8", alpha=0.9))
            ax_wf.legend(loc="upper right", fontsize=8, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
            ax_wf.tick_params(colors="white")
            ax_wf.grid(True, linestyle=":", alpha=0.25, color="white")
            pt_wf_filename = f"deepdive_wavefunctions_{v0_lbl}_isl{isl_id}.png"
            fig_wf.tight_layout()
            fig_wf.savefig(OUT_FIG_DIR / pt_wf_filename, dpi=200)
            plt.close(fig_wf)

            # c. Barrier Asymmetry Sweeps at Medoid Point
            flat_pt_idx = mu_idx * Nvz + vz_idx
            bl_l_pt = bl_l[flat_pt_idx, :]
            br_r_pt = br_r[flat_pt_idx, :]
            
            fig_ba, ax_ba = plt.subplots(figsize=(5.5, 3.8))
            fig_ba.patch.set_facecolor(BG_DARK)
            ax_ba.set_facecolor("#030712")
            ax_ba.plot(asym_ratios, bl_l_pt, color="#0284c7", lw=1.8, label=r"$G_{LL}(U_{\mathrm{asym}})$")
            ax_ba.plot(asym_ratios, br_r_pt, color="#f59e0b", lw=1.8, linestyle="--", label=r"$G_{RR}(U_{\mathrm{asym}})$")
            ax_ba.axvspan(-5.0, 5.0, color="#38bdf8", alpha=0.15, label="Low-Asym Window ([-5, 5] meV)")
            ax_ba.set_title(r"Dual Barrier Conductance Sweeps ($T = 40$ mK)", fontsize=10, fontweight="bold", color="white")
            ax_ba.set_xlabel(r"Asymmetry Parameter $\Delta U$ (meV)", fontsize=9, color="white")
            ax_ba.set_ylabel(r"Conductance ($2e^2/h$)", fontsize=9, color="white")
            ax_ba.set_xlim(-20.0, 20.0)
            ax_ba.legend(loc="upper right", fontsize=8, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
            ax_ba.tick_params(colors="white")
            ax_ba.grid(True, linestyle=":", alpha=0.25, color="white")
            pt_ba_filename = f"deepdive_barrier_asym_{v0_lbl}_isl{isl_id}.png"
            fig_ba.tight_layout()
            fig_ba.savefig(OUT_FIG_DIR / pt_ba_filename, dpi=200)
            plt.close(fig_ba)
            logger.info("Saved point deep-dive figures for %s Island %d", v0_lbl, isl_id)

    with open(OUT_FIG_DIR / "cluster_catalog.json", "w", encoding="utf-8") as f:
        json.dump(cluster_catalog, f, indent=2)
    logger.info("Saved cluster_catalog.json successfully.")


if __name__ == "__main__":
    # Remove old cluster and cut figures to ensure clean state after area pruning
    for pattern in ["cut_*.png", "deepdive_*.png", "phasemap_clusters_*.png"]:
        for old_f in OUT_FIG_DIR.glob(pattern):
            try:
                old_f.unlink()
            except Exception:
                pass
    generate_cluster_phasemaps_and_cuts()
    logger.info("All cluster cuts and deep-dive figures updated successfully in %s", OUT_FIG_DIR)
