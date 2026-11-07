#!/usr/bin/env python3
"""
build_unified_weekly_presentation_2026_10_28.py

Assembles the complete, unified weekly research presentation (2026-10-28):
  - Section 1: Intro Motivation Figures (0 mK vs 40 mK, low-asymmetry recovery, binarized correlation, symmetry balance)
               evaluated strictly on champion realization (Realization 14 at V0 = 1.2 meV).
  - Section 2: Protocol 0 — tgp_bnd000_lowcorr075_sym015 (MSFT_TGP_Hybrids weekly baseline).
  - Section 3: Protocol 1 — lowcorr080_curv500_5of5_gap10_sym025_c20 (Universal Killer).
  - Section 4: Protocol 2 — lax_base_strict_cluster50_core3 (Lax Base + Strict Cluster).
  - Section 5: Protocol 3 — lax_k4_sym030_strict_cluster35_core3 (Balanced Hybrid).
  - Section 6: Master Verification Statistics Tables across all disorder regimes at 30 mK and 40 mK.

Key Invariants:
  - Unified Disorder Realization: Every single diagnostic figure is run on Realization 14.
  - No realization count indications (N=50, N=80, etc.) in any headers, titles, or tables.
  - Strict layout & styling matching slides-lowcorr080_curv500_5of5_gap10_sym025_c20.pdf:
    - Pure plots with titles and slide counter pills (Slide X of 62).
    - 4-panel step-by-step diagnostic slides with Full Standard Microsoft TGP in Panel 1.
    - Dual barrier conductance sweep subpanels showing all 4 curves side-by-side with [-5, +5] meV shaded window.
    - BdG spectra cuts with both Pfaffian ground truth and Protocol Topological Phase indicator bars.
    - Maximized 16:9 canvas coverage (100% width, 98%+ height).
"""

import os
import sys
import shutil
import pickle
import logging
import subprocess
import shlex
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import label as nd_label

REPO_ROOT = Path("/home/pseudonym/Documents/Code/NonlocalProtocol").resolve()
sys.path.insert(0, str(REPO_ROOT))

import src.helpers as hp
from src.Protocol_Elements import (
    RealizationFeatureCache,
    load_protocol_class_from_file,
    DPE,
    FPE,
)
from src.Protocol_Elements.FPE.barrier_correlation import extract_barrier_reflection_correlations
from src.Protocol_Elements.FPE.barrier_asymmetry import fpe_lead_symmetry_balance
from src.Protocol_Elements.FPE.transport_and_spectral_gap import extract_nonlocal_transport_gap_maps

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("UnifiedWeeklyBuilder")

WEEKLY_DIR = REPO_ROOT / "Outputs" / "slides" / "weekly research slides" / "2026-10-28"
FIG_ROOT = WEEKLY_DIR / "figures"
HTML_OUT = WEEKLY_DIR / "slides-2026-10-28.html"
PDF_OUT = WEEKLY_DIR / "slides-2026-10-28.pdf"

DATA_ROOT = REPO_ROOT / "Outputs" / "Data" / "Disorder_Realizations"
CHAMPION_REAL = "disorder_realization_14"

# Global theme colors
BG_DARK = "#0b1120"
BG_CARD = "#0f172a"
CYAN_PFAFF = "#38bdf8"
DARK_SLATE = "#1e293b"

PROTOCOLS = {
    "tgp_bnd000_lowcorr075_sym015": {
        "title": "tgp_bnd000_lowcorr075_sym015",
        "subtitle": "Weekly Research Baseline: Boundaryless TGP (bnd=0%) + Low-Asym Corr (C >= 0.75) + Symmetry Balance (S_sym <= 0.15)",
        "family": "MSFT_TGP_Hybrids",
        "archetype": "Boundaryless TGP + Transport Reflection & Symmetry Gating",
        "theme_color": "#818cf8", # Slate / Indigo
        "c_th": 0.75,
        "curv_th": 100.0,
        "gap_mev": 0.010,
        "s_sym": 0.15,
        "min_island_size": 7,
        "core_shape": None,
        "stats_pass_rate": "92.5%",
        "stats_fdr": "5.58% (30 mK) / 10.09% (40 mK)",
        "stats_area": "22.4 px (max 311 px)",
        "stats_mzm": "C = 0.977 | S = 0.567",
    },
    "lowcorr080_curv500_5of5_gap10_sym025_c20": {
        "title": "lowcorr080_curv500_5of5_gap10_sym025_c20",
        "subtitle": "Strict Gating: C >= 0.80, Curv <= -500 (5/5), Gap >= 10 ueV, S_sym <= 0.25, Area >= 20 px",
        "family": "Low_Asym_Corr_Curv3w_Transport_Gap_Protocol",
        "archetype": "Universal Killer + Tight Physical Gating",
        "theme_color": "#10b981", # Emerald
        "c_th": 0.80,
        "curv_th": 500.0,
        "k_min": 5,
        "gap_mev": 0.010,
        "s_sym": 0.25,
        "min_island_size": 20,
        "core_shape": [2, 2],
        "stats_pass_rate": "100.0%",
        "stats_fdr": "0.17% (30 mK) / 14.86% (40 mK)",
        "stats_area": "25.4 px (max 39 px)",
        "stats_mzm": "C = 0.988 | S = 0.584",
    },
    "lax_base_strict_cluster50_core3": {
        "title": "lax_base_strict_cluster50_core3",
        "subtitle": "Lax Base + Strict Cluster: C >= 0.75, Curv <= -100 (3/5), Gap >= 10 ueV, Area >= 50 px, Core 3x3",
        "family": "Low_Asym_Corr_Curv3w_Transport_Gap_Protocol",
        "archetype": "Lax Gating + Strict Spatial Morphological Core",
        "theme_color": "#f59e0b", # Amber
        "c_th": 0.75,
        "curv_th": 100.0,
        "k_min": 3,
        "gap_mev": 0.010,
        "s_sym": None,
        "min_island_size": 50,
        "core_shape": [3, 3],
        "stats_pass_rate": "100.0%",
        "stats_fdr": "0.25% (30 mK) / 13.91% (40 mK)",
        "stats_area": "68.2 px (max 104 px)",
        "stats_mzm": "C = 0.989 | S = 0.591",
    },
    "lax_k4_sym030_strict_cluster35_core3": {
        "title": "lax_k4_sym030_strict_cluster35_core3",
        "subtitle": "Balanced Hybrid: C >= 0.75, Curv <= -100 (4/5), Gap >= 10 ueV, S_sym <= 0.30, Area >= 35 px, Core 3x3",
        "family": "Low_Asym_Corr_Curv3w_Transport_Gap_Protocol",
        "archetype": "Consensus Gating + Relaxed Symmetry + Strict Core",
        "theme_color": "#38bdf8", # Sky Blue
        "c_th": 0.75,
        "curv_th": 100.0,
        "k_min": 4,
        "gap_mev": 0.010,
        "s_sym": 0.30,
        "min_island_size": 35,
        "core_shape": [3, 3],
        "stats_pass_rate": "100.0%",
        "stats_fdr": "0.21% (30 mK) / 14.12% (40 mK)",
        "stats_area": "47.8 px (max 76 px)",
        "stats_mzm": "C = 0.988 | S = 0.588",
    },
}

CACHE_FILE = REPO_ROOT / "Outputs/slides/.bdg_cut_cache.pkl"
try:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            BDG_CUT_CACHE = pickle.load(f)
    else:
        BDG_CUT_CACHE = {}
except Exception:
    BDG_CUT_CACHE = {}


def save_bdg_cache():
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(BDG_CUT_CACHE, f)
    except Exception as e:
        logger.warning("Could not save BdG cache: %s", e)


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


def solve_bdg_cut_evals(real_dir: Path, mu_cut: float, vz_vals: np.ndarray, n_evals: int = 14) -> np.ndarray:
    cache_key = (str(real_dir), round(mu_cut, 4), len(vz_vals))
    if cache_key in BDG_CUT_CACHE:
        return BDG_CUT_CACHE[cache_key].copy()

    all_p = np.load(real_dir / "all_params.npz", allow_pickle=True)
    t_val = float(all_p["t"])
    gamma = float(all_p["gamma"])
    Delta0 = float(all_p["Delta0"])
    alpha = float(all_p["alpha"])
    Ls = int(all_p["Ls"])
    Vdisx = np.load(real_dir / "Vdisx.npy")

    Nvz = len(vz_vals)
    evals = np.zeros((Nvz, n_evals))
    for j in range(Nvz):
        syst_j = hp.build_system_closed(t_val, mu_cut, gamma, Delta0, vz_vals[j], alpha, Ls, Vdisx, a=1)
        ev_j, _ = hp.solve_ham(syst_j, k=n_evals, solver_type="cpu")
        evals[j, :] = np.sort(ev_j)
    BDG_CUT_CACHE[cache_key] = evals.copy()
    save_bdg_cache()
    return evals


def generate_intro_figures(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating Intro Motivation figures on Realization 14 (V0 = 1.2 meV)...")
    real_dir = DATA_ROOT / "V0_1_2" / CHAMPION_REAL

    c0 = RealizationFeatureCache(real_dir, temperature_mK=0.0, v0_label="V0_1_2")
    c40 = RealizationFeatureCache(real_dir, temperature_mK=40.0, v0_label="V0_1_2")

    gt = c40.pfaffian_mask
    mu_vals = c40.mu_vals
    vz_vals = c40.vz_vals
    extent = [vz_vals[0], vz_vals[-1], mu_vals[0], mu_vals[-1]]

    # FIG 1: 0 mK vs 40 mK Legacy Correlation
    cL_0, cR_0 = extract_barrier_reflection_correlations(c0, asym_min=0.0, asym_max=60.0)
    cL_40, cR_40 = extract_barrier_reflection_correlations(c40, asym_min=0.0, asym_max=60.0)
    c_both_0 = np.minimum(cL_0, cR_0)
    c_both_40 = np.minimum(cL_40, cR_40)

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 9.0), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG_DARK)

    for ax, c_map, title in [
        (axes[0], c_both_0, "T = 0 mK (Legacy [0, 60] meV Sweep)"),
        (axes[1], c_both_40, "T = 40 mK (Legacy [0, 60] meV Sweep)"),
    ]:
        ax.set_facecolor("#030712")
        im = ax.imshow(c_map, origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=2.0, linestyles="--")
        ax.set_title(title, fontsize=13.5, fontweight="bold", color="#f8fafc", pad=8)
        ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11.5, color="white")
        ax.tick_params(colors="white")
        ax.grid(True, linestyle=":", alpha=0.25, color="white")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Dual Correlation $\min(C_L, C_R)$", color="white", fontsize=11)
        cbar.ax.tick_params(colors="white")

    axes[0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11.5, color="white")
    legend_line = mlines.Line2D([0], [0], color=CYAN_PFAFF, lw=2.0, linestyle="--", label=r"Pfaffian Boundary ($\mathcal{Q} = -1$)")
    axes[0].legend(handles=[legend_line], loc="upper left", fontsize=9.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")

    fig.suptitle(r"Finite-Temperature Correlation Breakdown: $T = 0$ mK vs $T = 40$ mK (Severe Disorder $V_0 = 1.2$ meV, Realization 14)",
                 fontsize=14.5, fontweight="bold", color=CYAN_PFAFF, y=0.98)
    fig.tight_layout(pad=0.5, rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "fig1_0mK_vs_40mK_legacy_correlation.png", dpi=200)
    plt.close(fig)

    # FIG 2: 40 mK Legacy vs Low-Asymmetry
    cL_low, cR_low = extract_barrier_reflection_correlations(c40, asym_min=-5.0, asym_max=5.0)
    c_both_low = np.minimum(cL_low, cR_low)

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 9.0), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG_DARK)

    for ax, c_map, title in [
        (axes[0], c_both_40, "Legacy Full-Sweep Asymmetry ([0, 60] meV)"),
        (axes[1], c_both_low, r"Low-Asymmetry Gate Sweep ([-5, +5] meV)"),
    ]:
        ax.set_facecolor("#030712")
        im = ax.imshow(c_map, origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        ax.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=2.0, linestyles="--")
        ax.set_title(title, fontsize=13.5, fontweight="bold", color="#f8fafc", pad=8)
        ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11.5, color="white")
        ax.tick_params(colors="white")
        ax.grid(True, linestyle=":", alpha=0.25, color="white")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Dual Correlation $\min(C_L, C_R)$", color="white", fontsize=11)
        cbar.ax.tick_params(colors="white")

    axes[0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11.5, color="white")
    axes[0].legend(handles=[legend_line], loc="upper left", fontsize=9.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")

    fig.suptitle(r"Restoring Topological Correlation at $T = 40$ mK via Low-Asymmetry Window ([-5, +5] meV, Realization 14)",
                 fontsize=14.5, fontweight="bold", color=CYAN_PFAFF, y=0.98)
    fig.tight_layout(pad=0.5, rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "fig2_40mK_legacy_vs_low_asymmetry.png", dpi=200)
    plt.close(fig)

    # FIG 3: 40 mK Binarized Low-Asymmetry Correlation Condition (CL, CR >= 0.75)
    m_bin_corr = (cL_low >= 0.75) & (cR_low >= 0.75)
    met_bin_corr = calc_two_metrics(m_bin_corr, gt)

    fig, ax = plt.subplots(figsize=(16.5, 9.0))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor("#030712")

    img_rgba = np.zeros((len(mu_vals), len(vz_vals), 4))
    img_rgba[~gt & (~m_bin_corr)] = [0.04, 0.07, 0.13, 1.0]
    img_rgba[gt & (~m_bin_corr)] = [0.12, 0.16, 0.23, 0.95]
    img_rgba[m_bin_corr] = [0.01, 0.52, 0.78, 0.95]

    ax.imshow(img_rgba, origin="lower", extent=extent, aspect="auto")
    ax.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=2.0, linestyles="--")

    sub_title = (
        r"Binarized Low-Asymmetry Correlation Filter ($C_L, C_R \geq 0.75$ at $T = 40$ mK, Realization 14)"
        f"\nPfaffian Area Passed (TPR): {met_bin_corr['pfaffian_passed_pct']:.2f}% | Active FDR: {met_bin_corr['fdr_percent']:.2f}%"
    )
    ax.set_title(sub_title, fontsize=13.5, fontweight="bold", color="#f8fafc", pad=10)
    ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11.5, color="white")
    ax.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11.5, color="white")
    ax.tick_params(colors="white")
    ax.grid(True, linestyle=":", alpha=0.25, color="white")

    legend_elements = [
        mlines.Line2D([0], [0], color=CYAN_PFAFF, lw=2.0, linestyle="--", label=r"Pfaffian Boundary ($\mathcal{Q} = -1$)"),
        mpatches.Patch(facecolor="#1e293b", edgecolor="#475569", label="Pfaffian Bulk (Unpassed)"),
        mpatches.Patch(facecolor="#0284c7", label=r"Passed Pixels ($C_L, C_R \geq 0.75$)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10.0, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")

    fig.tight_layout(pad=0.5)
    fig.savefig(out_dir / "fig3_binarized_low_asymmetry_correlation_40mK.png", dpi=200)
    plt.close(fig)

    # FIG 4: 40 mK Symmetry Balance Ratio
    m_sym = fpe_lead_symmetry_balance(c40, s_max=0.15, eps_sym=1e-4)
    s_sym = c40._memo[("lead_symmetry_balance_obs", 0.0001, 0.0)].copy()
    s_sym_plot = np.clip(np.nan_to_num(s_sym, nan=1.0, posinf=1.0), 0.0, 1.0)
    met_sym = calc_two_metrics(m_sym, gt)

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 9.0), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG_DARK)

    # Left: Continuous S_sym map
    ax0 = axes[0]
    ax0.set_facecolor("#030712")
    im0 = ax0.imshow(s_sym_plot, origin="lower", extent=extent, aspect="auto", cmap="viridis_r", vmin=0.0, vmax=0.6)
    ax0.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=2.0, linestyles="--")
    ax0.set_title(r"Continuous Symmetry Balance Ratio $\mathcal{S}_{\mathrm{sym}}(\mu, V_z)$", fontsize=13.0, fontweight="bold", color="white", pad=8)
    ax0.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11.5, color="white")
    ax0.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11.5, color="white")
    ax0.tick_params(colors="white")
    ax0.grid(True, linestyle=":", alpha=0.25, color="white")
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label(r"Asymmetry Ratio $\mathcal{S}_{\mathrm{sym}}$", color="white", fontsize=11)
    cbar0.ax.tick_params(colors="white")

    # Right: Binarized S_sym <= 0.15
    ax1 = axes[1]
    ax1.set_facecolor("#030712")
    img_sym_rgba = np.zeros((len(mu_vals), len(vz_vals), 4))
    img_sym_rgba[~gt & (~m_sym)] = [0.04, 0.07, 0.13, 1.0]
    img_sym_rgba[gt & (~m_sym)] = [0.12, 0.16, 0.23, 0.95]
    img_sym_rgba[m_sym] = [0.96, 0.62, 0.04, 0.95] # amber

    ax1.imshow(img_sym_rgba, origin="lower", extent=extent, aspect="auto")
    ax1.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=2.0, linestyles="--")
    sub_title_sym = (
        r"Binarized Symmetry Gate ($\mathcal{S}_{\mathrm{sym}} \leq 0.15$ at $T = 40$ mK)"
        f"\nPfaffian Area Passed (TPR): {met_sym['pfaffian_passed_pct']:.2f}% | Active FDR: {met_sym['fdr_percent']:.2f}%"
    )
    ax1.set_title(sub_title_sym, fontsize=13.0, fontweight="bold", color="white", pad=8)
    ax1.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11.5, color="white")
    ax1.tick_params(colors="white")
    ax1.grid(True, linestyle=":", alpha=0.25, color="white")

    leg_sym = [
        mlines.Line2D([0], [0], color=CYAN_PFAFF, lw=2.0, linestyle="--", label=r"Pfaffian Boundary ($\mathcal{Q} = -1$)"),
        mpatches.Patch(facecolor="#1e293b", edgecolor="#475569", label="Pfaffian Bulk (Unpassed)"),
        mpatches.Patch(facecolor="#f59e0b", label=r"Passed ($\mathcal{S}_{\mathrm{sym}} \leq 0.15$)"),
    ]
    ax1.legend(handles=leg_sym, loc="upper left", fontsize=9.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")

    fig.suptitle(r"Conjugate Lead Symmetry Balance Filter at $T = 40$ mK (Severe Disorder $V_0 = 1.2$ meV, Realization 14)",
                 fontsize=14.5, fontweight="bold", color=CYAN_PFAFF, y=0.98)
    fig.tight_layout(pad=0.5, rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "fig4_symmetry_balance_phase_map_40mK.png", dpi=200)
    plt.close(fig)
    logger.info("Saved all Intro figures to %s", out_dir)


def generate_protocol0_figures(out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating figure suite for Protocol 0 ('tgp_bnd000_lowcorr075_sym015') in %s...", out_dir)

    proto_cls = load_protocol_class_from_file(REPO_ROOT / "Protocols/MSFT_TGP_Hybrids/protocol.py")
    proto = proto_cls()

    cfg_path = REPO_ROOT / "Outputs/MSFT_Hybrids_Test_Raw/MSFT_TGP_Hybrids/tgp_bnd000_lowcorr075_sym015/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    params = cfg["free_parameters"]

    v0_targets = [
        ("V0_0_378", 0.378, "Weak Disorder (V0 = 0.378 meV)"),
        ("V0_0_645", 0.645, "Intermediate Disorder (V0 = 0.645 meV)"),
        ("V0_0_872", 0.872, "Strong Disorder (V0 = 0.872 meV)"),
        ("V0_1_2",   1.200, "Severe Benchmark Disorder (V0 = 1.200 meV)"),
    ]

    caches = {}
    step_results = {}
    cluster_catalogs = {}

    for v0_lbl, v0_val, v0_title in v0_targets:
        real_dir = DATA_ROOT / v0_lbl / CHAMPION_REAL
        cache = RealizationFeatureCache(real_dir, temperature_mK=40.0, v0_label=v0_lbl)
        caches[v0_lbl] = cache
        steps = proto.evaluate_steps(cache, params)
        step_results[v0_lbl] = steps
        gt = cache.pfaffian_mask
        mu_vals = cache.mu_vals
        vz_vals = cache.vz_vals
        extent = [vz_vals[0], vz_vals[-1], mu_vals[0], mu_vals[-1]]

        m_tgp_bnd0 = steps["step_3_tgp_stage2_candidate"]
        m_corr = steps["step_4_fpe_low_asym_corr"]
        m_phys = steps["step_5_fpe_lead_symmetry_balance"]
        m_final = steps["step_8_dpe_C4_final_mask"]

        # Full Standard Microsoft TGP Protocol (bnd=60%, Stage 2 Final Mask)
        tgp_step_dir = REPO_ROOT / "Outputs/Postprocessing/Microsoft_Hybrids_Post_Processing/MSFT_TGP_Hybrids/tgp_std_bnd060/step_data"
        tgp_npz_path = tgp_step_dir / f"ensemble_40mK__{v0_lbl}_steps.npz"
        tgp_data = np.load(tgp_npz_path, allow_pickle=True)
        tgp_keys = list(tgp_data["realization_keys"])
        tgp_masks = tgp_data["step_8_dpe_C4_final_mask"]
        tgp_idx = [i for i, k in enumerate(tgp_keys) if CHAMPION_REAL in k][0]
        full_tgp_mask = tgp_masks[tgp_idx]

        met_tgp = calc_two_metrics(full_tgp_mask, gt)
        met_bnd0 = calc_two_metrics(m_tgp_bnd0, gt)
        met_phys = calc_two_metrics(m_phys, gt)
        met_final = calc_two_metrics(m_final, gt)

        # 1a. Direct Simulation Observables 4-Panel Plot (Continuous Physical Maps)
        cL, cR = extract_barrier_reflection_correlations(cache, asym_min=-5.0, asym_max=5.0)
        c_min = np.minimum(cL, cR)

        lockin = cache.get_stage1_broadened_and_lockin()
        both_3w = np.maximum(lockin["I_3w_L"], lockin["I_3w_R"])
        sorted_3w = np.sort(both_3w, axis=0)
        k_min = 3
        curv_consensus = sorted_3w[k_min - 1]

        _, _, gap_mev = extract_nonlocal_transport_gap_maps(cache)
        gap_uev = gap_mev * 1000.0

        bdata = cache.get_barrier_data()
        idx_sym = int(np.argmin(np.abs(bdata["asym_ratios"] - 0.0)))
        g_ll = bdata["br_l"][:, :, idx_sym]
        g_rr = bdata["bl_r"][:, :, idx_sym]
        s_sym = np.abs(g_ll - g_rr) / np.maximum(g_ll + g_rr, 1e-4)

        fig_raw, axes_raw = plt.subplots(2, 2, figsize=(16.5, 9.0), sharex=True, sharey=True)
        fig_raw.patch.set_facecolor(BG_DARK)

        # Panel 1: Reflection Correlation min(C_L, C_R)
        ax0 = axes_raw[0, 0]
        ax0.set_facecolor("#030712")
        im0 = ax0.imshow(c_min, origin="lower", extent=extent, aspect="auto", cmap="viridis", vmin=-0.2, vmax=1.0)
        ax0.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=1.8, linestyles="--")
        ax0.contour(vz_vals, mu_vals, c_min, levels=[0.75], colors=["#f59e0b"], linewidths=2.0)
        ax0.set_title(r"1. Low-Asymmetry Reflection Correlation $\min(C_L, C_R)$" + "\n" + r"Isoline: $C \geq 0.75$ (Amber) | Pfaffian Ground Truth (Cyan)", fontsize=10.5, fontweight="bold", pad=6, color="#f1f5f9")
        ax0.tick_params(colors="white")
        ax0.grid(True, linestyle=":", alpha=0.25, color="white")
        div0 = make_axes_locatable(ax0)
        cax0 = div0.append_axes("right", size="3.5%", pad=0.08)
        cb0 = fig_raw.colorbar(im0, cax=cax0)
        cb0.ax.tick_params(colors="white", labelsize=8.5)
        cb0.set_label(r"$\min(C_L, C_R)$", color="white", fontsize=9.5)

        # Panel 2: 3w Curvature Consensus
        ax1 = axes_raw[0, 1]
        ax1.set_facecolor("#030712")
        im1 = ax1.imshow(curv_consensus, origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=-1200, vmax=200)
        ax1.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=1.8, linestyles="--")
        ax1.contour(vz_vals, mu_vals, curv_consensus, levels=[-100.0], colors=["#f59e0b"], linewidths=2.0)
        ax1.set_title(r"2. $3\omega$ Consensus Curvature $\kappa_{3/5}$ ($\mu$S/V$^2$)" + "\n" + r"Isoline: $\kappa \leq -100\,\mu$S/V$^2$ (Amber) | Pfaffian Ground Truth (Cyan)", fontsize=10.5, fontweight="bold", pad=6, color="#f1f5f9")
        ax1.tick_params(colors="white")
        ax1.grid(True, linestyle=":", alpha=0.25, color="white")
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes("right", size="3.5%", pad=0.08)
        cb1 = fig_raw.colorbar(im1, cax=cax1)
        cb1.ax.tick_params(colors="white", labelsize=8.5)
        cb1.set_label(r"Curvature ($\mu$S/V$^2$)", color="white", fontsize=9.5)

        # Panel 3: Extracted Open-System Transport Gap
        ax2 = axes_raw[1, 0]
        ax2.set_facecolor("#030712")
        im2 = ax2.imshow(gap_uev, origin="lower", extent=extent, aspect="auto", cmap="plasma", vmin=0.0, vmax=100.0)
        ax2.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=1.8, linestyles="--")
        ax2.contour(vz_vals, mu_vals, gap_uev, levels=[10.0], colors=["#f59e0b"], linewidths=2.0)
        ax2.set_title(r"3. Extracted Open-System Transport Gap $\Delta_{\rm trans}$ ($\mu$eV)" + "\n" + r"Isoline: $\Delta \geq 10\,\mu$eV (Amber) | Pfaffian Ground Truth (Cyan)", fontsize=10.5, fontweight="bold", pad=6, color="#f1f5f9")
        ax2.tick_params(colors="white")
        ax2.grid(True, linestyle=":", alpha=0.25, color="white")
        div2 = make_axes_locatable(ax2)
        cax2 = div2.append_axes("right", size="3.5%", pad=0.08)
        cb2 = fig_raw.colorbar(im2, cax=cax2)
        cb2.ax.tick_params(colors="white", labelsize=8.5)
        cb2.set_label(r"Transport Gap ($\mu$eV)", color="white", fontsize=9.5)

        # Panel 4: Conjugate Lead Symmetry Balance
        ax3 = axes_raw[1, 1]
        ax3.set_facecolor("#030712")
        im3 = ax3.imshow(s_sym, origin="lower", extent=extent, aspect="auto", cmap="Blues_r", vmin=0.0, vmax=0.6)
        ax3.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=1.8, linestyles="--")
        ax3.contour(vz_vals, mu_vals, s_sym, levels=[0.15], colors=["#f59e0b"], linewidths=2.0)
        ax3.set_title(r"4. Conjugate Lead Symmetry Balance $\mathcal{S}_{\rm sym}$" + "\n" + r"Isoline: $\mathcal{S}_{\rm sym} \leq 0.15$ (Amber) | Pfaffian Ground Truth (Cyan)", fontsize=10.5, fontweight="bold", pad=6, color="#f1f5f9")
        ax3.tick_params(colors="white")
        ax3.grid(True, linestyle=":", alpha=0.25, color="white")
        div3 = make_axes_locatable(ax3)
        cax3 = div3.append_axes("right", size="3.5%", pad=0.08)
        cb3 = fig_raw.colorbar(im3, cax=cax3)
        cb3.ax.tick_params(colors="white", labelsize=8.5)
        cb3.set_label(r"$\mathcal{S}_{\rm sym}$", color="white", fontsize=9.5)

        axes_raw[0, 0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10.5, color="white")
        axes_raw[1, 0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10.5, color="white")
        axes_raw[1, 0].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10.5, color="white")
        axes_raw[1, 1].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10.5, color="white")

        fig_raw.suptitle(f"Direct Simulation Observables & Continuous Sensor Phase Maps: {v0_title} at T = 40 mK (Realization 14)", fontsize=12.5, fontweight="bold", color=CYAN_PFAFF, y=0.99)
        fig_raw.tight_layout(pad=0.5, rect=[0, 0, 1, 0.96])
        raw_filename = f"raw_observables_{v0_lbl}.png"
        fig_raw.savefig(out_dir / raw_filename, dpi=200)
        plt.close(fig_raw)
        logger.info("Saved %s to %s", raw_filename, out_dir)

        # 1b. Step-by-Step 4-Panel Plot (16:9 optimized aspect)
        fig, axes = plt.subplots(2, 2, figsize=(16.5, 9.0), sharex=True, sharey=True)
        fig.patch.set_facecolor(BG_DARK)

        panels = [
            (axes[0, 0], full_tgp_mask, "#a855f7", "Full Standard Microsoft TGP (bnd=60%)", met_tgp),
            (axes[0, 1], m_tgp_bnd0, "#0284c7", "Step 1: Base Candidate (bnd=0%)", met_bnd0),
            (axes[1, 0], m_phys, "#f59e0b", "Step 2–3: Low-Asym Corr + Symmetry Balance", met_phys),
            (axes[1, 1], m_final, "#818cf8", "Step 4: Canonical C4 Final Island Mask (Area >= 7 px)", met_final),
        ]

        for ax, mask, color, ptitle, m_dict in panels:
            ax.set_facecolor("#030712")
            img_rgba = np.zeros((len(mu_vals), len(vz_vals), 4))
            img_rgba[~gt & (~mask)] = [0.04, 0.07, 0.13, 1.0]
            img_rgba[gt & (~mask)] = [0.12, 0.16, 0.23, 0.95]
            rgb = mcolors.to_rgb(color)
            img_rgba[mask] = [rgb[0], rgb[1], rgb[2], 0.95]

            ax.imshow(img_rgba, origin="lower", extent=extent, aspect="auto")
            ax.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=1.6, linestyles="--")

            sub_title = f"{ptitle}\nPfaffian Passed (TPR): {m_dict['pfaffian_passed_pct']:.2f}% | Active FDR: {m_dict['fdr_percent']:.2f}%"
            ax.set_title(sub_title, fontsize=10.5, fontweight="bold", pad=6, color="#f1f5f9")
            ax.grid(True, linestyle=":", alpha=0.25, color="white")
            ax.tick_params(colors="white")

        axes[0, 0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10.5, color="white")
        axes[1, 0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10.5, color="white")
        axes[1, 0].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10.5, color="white")
        axes[1, 1].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10.5, color="white")

        legend_elements = [
            mlines.Line2D([0], [0], color=CYAN_PFAFF, lw=1.8, linestyle="--", label=r"Pfaffian Boundary ($\mathcal{Q} = -1$)"),
            mpatches.Patch(facecolor="#1e293b", edgecolor="#475569", label="Pfaffian Bulk (Unpassed)"),
            mpatches.Patch(facecolor="#a855f7", label="Full Standard TGP (bnd=60%)"),
            mpatches.Patch(facecolor="#0284c7", label="Step 1: Base Candidate (bnd=0%)"),
            mpatches.Patch(facecolor="#f59e0b", label="Step 2-3: Corr + SymBal"),
            mpatches.Patch(facecolor="#818cf8", label="Final Island Mask"),
        ]
        axes[0, 0].legend(handles=legend_elements, loc="upper left", fontsize=7.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")

        fig.suptitle(f"Step-by-Step Protocol Elimination: {v0_title} at T = 40 mK (Realization 14)", fontsize=12.5, fontweight="bold", color=CYAN_PFAFF, y=0.99)
        fig.tight_layout(pad=0.5, rect=[0, 0, 1, 0.96])
        s_filename = f"step_by_step_{v0_lbl}.png"
        fig.savefig(out_dir / s_filename, dpi=200)
        plt.close(fig)

    # 2. Surviving Clusters 4-Panel Phase Map
    fig_pm, axes_pm = plt.subplots(2, 2, figsize=(16.5, 9.0), sharex=True, sharey=True)
    fig_pm.patch.set_facecolor(BG_DARK)
    star_palette = ["#00f5d4", "#f72585", "#ffbe0b", "#4cc9f0", "#a78bfa", "#f97316"]

    for idx, (v0_lbl, v0_val, v0_title) in enumerate(v0_targets):
        ax = axes_pm[idx // 2, idx % 2]
        ax.set_facecolor("#030712")
        cache = caches[v0_lbl]
        gt = cache.pfaffian_mask
        m_final = step_results[v0_lbl]["step_8_dpe_C4_final_mask"]
        mu_vals = cache.mu_vals
        vz_vals = cache.vz_vals
        extent = [vz_vals[0], vz_vals[-1], mu_vals[0], mu_vals[-1]]

        lbl_arr, num_isl = nd_label(m_final, structure=np.ones((3, 3), dtype=int))
        img_pm = np.zeros((len(mu_vals), len(vz_vals), 4))
        img_pm[~gt & (~m_final)] = [0.04, 0.07, 0.13, 1.0]
        img_pm[gt & (~m_final)] = [0.12, 0.16, 0.23, 0.95]

        c_sizes = [(int(np.sum(lbl_arr == i)), i) for i in range(1, num_isl + 1)]
        c_sizes.sort(reverse=True)

        isl_info_list = []
        for r_rank, (sz, raw_id) in enumerate(c_sizes):
            pts = np.argwhere(lbl_arr == raw_id)
            c_col = star_palette[r_rank % len(star_palette)]
            col_rgb = mcolors.to_rgb(c_col)
            img_pm[lbl_arr == raw_id] = [col_rgb[0], col_rgb[1], col_rgb[2], 0.95]

            com_mu = float(np.mean(mu_vals[pts[:, 0]]))
            com_vz = float(np.mean(vz_vals[pts[:, 1]]))

            topo_pts = [p for p in pts if gt[p[0], p[1]]]
            cand_pts = topo_pts if len(topo_pts) > 0 else pts
            dists = [(mu_vals[p[0]] - com_mu)**2 + (vz_vals[p[1]] - com_vz)**2 for p in cand_pts]
            best_p = cand_pts[int(np.argmin(dists))]
            med_mu = float(mu_vals[best_p[0]])
            med_vz = float(vz_vals[best_p[1]])

            isl_info_list.append({
                "rank": r_rank + 1,
                "raw_id": raw_id,
                "area_px": sz,
                "com_mu": com_mu,
                "com_vz": com_vz,
                "med_mu": med_mu,
                "med_vz": med_vz,
                "med_mu_idx": int(best_p[0]),
                "med_vz_idx": int(best_p[1]),
                "color": c_col,
            })

            ax.plot(med_vz, med_mu, marker="*", color=c_col, markersize=14, markeredgecolor="white", markeredgewidth=1.2, zorder=10)
            ax.annotate(f"Island {r_rank + 1}: {sz} px",
                        xy=(med_vz, med_mu),
                        xytext=(med_vz + 0.04, med_mu + 0.10),
                        color="white", fontsize=8.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=BG_CARD, edgecolor=c_col, alpha=0.9),
                        arrowprops=dict(arrowstyle="->", color=c_col, lw=1.2))

        cluster_catalogs[v0_lbl] = isl_info_list
        ax.imshow(img_pm, origin="lower", extent=extent, aspect="auto")
        ax.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=1.8, linestyles="--")
        ax.set_title(f"{v0_title} — {num_isl} Surviving Island(s) ({sum(s for s, _ in c_sizes)} Total px)", fontsize=11, fontweight="bold", pad=6, color="#f1f5f9")
        ax.grid(True, linestyle=":", alpha=0.25, color="white")
        ax.tick_params(colors="white")

    axes_pm[0, 0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10.5, color="white")
    axes_pm[1, 0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10.5, color="white")
    axes_pm[1, 0].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10.5, color="white")
    axes_pm[1, 1].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10.5, color="white")

    leg_pm = [
        mlines.Line2D([0], [0], color=CYAN_PFAFF, lw=1.8, linestyle="--", label=r"Pfaffian Boundary ($\mathcal{Q} = -1$)"),
        mpatches.Patch(facecolor="#1e293b", edgecolor="#475569", label="Pfaffian Bulk (Unpassed)"),
        mlines.Line2D([0], [0], marker="*", color="#00f5d4", markeredgecolor="white", markersize=12, linestyle="None", label="Island 1 Medoid"),
        mlines.Line2D([0], [0], marker="*", color="#f72585", markeredgecolor="white", markersize=12, linestyle="None", label="Island 2 Medoid"),
    ]
    axes_pm[0, 0].legend(handles=leg_pm, loc="upper left", fontsize=8.0, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")

    fig_pm.suptitle(f"Protocol tgp_bnd000_lowcorr075_sym015: Surviving Topological Clusters & Medoids across Disorders (T = 40 mK)",
                    fontsize=12.5, fontweight="bold", color=CYAN_PFAFF, y=0.99)
    fig_pm.tight_layout(pad=0.5, rect=[0, 0, 1, 0.96])
    fig_pm.savefig(out_dir / "phasemap_clusters_all_V0.png", dpi=200)
    plt.close(fig_pm)

    # 3. Cut Overviews and Point Deep-Dives for Island 1 of each disorder
    for v0_lbl, v0_val, v0_title in v0_targets:
        c_v0 = caches[v0_lbl]
        real_dir = DATA_ROOT / v0_lbl / CHAMPION_REAL
        gt = c_v0.pfaffian_mask
        m_final = step_results[v0_lbl]["step_8_dpe_C4_final_mask"]
        mu_vals = c_v0.mu_vals
        vz_vals = c_v0.vz_vals
        extent = [vz_vals[0], vz_vals[-1], mu_vals[0], mu_vals[-1]]
        Nmu, Nvz = len(mu_vals), len(vz_vals)

        isl_list = cluster_catalogs[v0_lbl]
        if len(isl_list) == 0:
            logger.warning("No island found for %s! Falling back to max pfaffian point.", v0_lbl)
            mu_idx, vz_idx = 48, 64
            mu_cut, vz_med = mu_vals[mu_idx], vz_vals[vz_idx]
            star_col = "#00f5d4"
        else:
            top_isl = isl_list[0]
            mu_idx = top_isl["med_mu_idx"]
            vz_idx = top_isl["med_vz_idx"]
            mu_cut = top_isl["med_mu"]
            vz_med = top_isl["med_vz"]
            star_col = top_isl["color"]

        # 3a. Phase map cut line
        fig_cut, ax_c = plt.subplots(figsize=(7.5, 4.4))
        fig_cut.patch.set_facecolor(BG_DARK)
        ax_c.set_facecolor("#030712")

        img_pm_cut = np.zeros((Nmu, Nvz, 4))
        img_pm_cut[~gt & (~m_final)] = [0.04, 0.07, 0.13, 1.0]
        img_pm_cut[gt & (~m_final)] = [0.12, 0.16, 0.23, 0.95]
        col_rgb = mcolors.to_rgb("#818cf8")
        img_pm_cut[m_final] = [col_rgb[0], col_rgb[1], col_rgb[2], 0.95]

        ax_c.imshow(img_pm_cut, origin="lower", extent=extent, aspect="auto")
        ax_c.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=1.6, linestyles="--")
        ax_c.axhline(mu_cut, color="#f43f5e", linestyle="-", lw=1.8, alpha=0.9, label=rf"Cut $\mu = {mu_cut:.3f}$ meV")
        ax_c.plot(vz_med, mu_cut, marker="*", color=star_col, markersize=14, markeredgecolor="white", markeredgewidth=1.2, zorder=10)

        ax_c.set_title(rf"{v0_title} | Cut Line ($\mu = {mu_cut:.3f}$ meV)", fontsize=10.5, fontweight="bold", color="white", pad=6)
        ax_c.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=9.5, color="white")
        ax_c.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=9.5, color="white")
        ax_c.tick_params(colors="white")
        ax_c.grid(True, linestyle=":", alpha=0.25, color="white")
        ax_c.legend(loc="upper left", fontsize=7.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
        fig_cut.tight_layout()
        cut_pm_filename = f"phasemap_cut_{v0_lbl}_isl1.png"
        fig_cut.savefig(out_dir / cut_pm_filename, dpi=200)
        plt.close(fig_cut)

        # 3b. BdG cut spectra
        evals_cut = solve_bdg_cut_evals(real_dir, mu_cut, vz_vals, n_evals=14)
        gt_cut = gt[mu_idx, :]
        n_evals_cut = evals_cut.shape[1]
        mid_ev = n_evals_cut // 2

        fig_sp, ax_sp = plt.subplots(figsize=(7.5, 4.4))
        fig_sp.patch.set_facecolor(BG_DARK)
        ax_sp.set_facecolor("#030712")

        for j in range(Nvz):
            if gt_cut[j]:
                dvz = (vz_vals[-1] - vz_vals[0]) / (Nvz - 1)
                ax_sp.axvspan(vz_vals[j] - 0.5 * dvz, vz_vals[j] + 0.5 * dvz, color=DARK_SLATE, alpha=0.75, zorder=0)

        for ev_i in range(n_evals_cut):
            col_ev = "#ef4444" if ev_i in [mid_ev - 1, mid_ev] else CYAN_PFAFF
            lw_ev = 2.2 if ev_i in [mid_ev - 1, mid_ev] else 1.1
            ax_sp.plot(vz_vals, evals_cut[:, ev_i], color=col_ev, lw=lw_ev, alpha=0.85, zorder=2)

        ax_sp.axhline(0.0, color="white", lw=0.8, linestyle=":", alpha=0.6, zorder=3)
        ax_sp.axvline(vz_med, color=star_col, linestyle="--", lw=1.5, alpha=0.9, zorder=4)
        ax_sp.plot(vz_med, 0.0, marker="*", color=star_col, markersize=14, markeredgecolor="black", markeredgewidth=1.2, zorder=10)

        # Horizontal indicator lines along bottom of spectra
        dvz = (vz_vals[-1] - vz_vals[0]) / (Nvz - 1)
        m_cut = m_final[mu_idx, :]
        for j in range(Nvz):
            if gt_cut[j]:
                ax_sp.plot([vz_vals[j] - 0.5 * dvz, vz_vals[j] + 0.5 * dvz], [-0.142, -0.142],
                           color=CYAN_PFAFF, lw=3.8, solid_capstyle="butt", zorder=6)
            if m_cut[j]:
                ax_sp.plot([vz_vals[j] - 0.5 * dvz, vz_vals[j] + 0.5 * dvz], [-0.154, -0.154],
                           color="#818cf8", lw=4.8, solid_capstyle="butt", zorder=7)

        ax_sp.set_ylim(-0.168, 0.16)
        ax_sp.set_xlim(vz_vals[0], vz_vals[-1])
        ax_sp.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=9.5, color="white")
        ax_sp.set_ylabel("Energy (meV)", fontsize=9.5, color="white")
        ax_sp.set_title(rf"BdG Energy Spectrum along Cut ($\mu = {mu_cut:.3f}$ meV)", fontsize=10.5, fontweight="bold", color=CYAN_PFAFF, pad=6)

        leg_sp = [
            mlines.Line2D([0], [0], color="#ef4444", lw=2.2, label="Lowest BdG States"),
            mlines.Line2D([0], [0], color=CYAN_PFAFF, lw=1.2, label="Bulk Excitations"),
            mlines.Line2D([0], [0], color=CYAN_PFAFF, lw=3.0, linestyle="--", label=r"Pfaffian Ground Truth ($\mathcal{Q}=-1$)"),
            mlines.Line2D([0], [0], color="#818cf8", lw=4.0, label="Protocol Topological Phase"),
            mlines.Line2D([0], [0], marker="*", color=star_col, markeredgecolor="white", markersize=10, linestyle="None", label=f"Medoid Star: $V_z={vz_med:.3f}$"),
        ]
        ax_sp.legend(handles=leg_sp, loc="upper right", fontsize=7.2, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white", framealpha=0.9)
        ax_sp.tick_params(colors="white")
        ax_sp.grid(True, linestyle=":", alpha=0.25, color="white")
        fig_sp.tight_layout()
        sp_filename = f"cut_spectra_{v0_lbl}_isl1.png"
        fig_sp.savefig(out_dir / sp_filename, dpi=200)
        plt.close(fig_sp)

        # 3c. 5-panel conductance along cut
        d_ll = np.load(real_dir / "tgp_stage1_dIdVl.npy").reshape(Nmu, Nvz, 5, 97)
        d_rr = np.load(real_dir / "tgp_stage1_dIdVr.npy").reshape(Nmu, Nvz, 5, 97)
        d_rl = np.load(real_dir / "tgp_stage1_dIdV_RL.npy").reshape(Nmu, Nvz, 5, 97)
        d_lr = np.load(real_dir / "tgp_stage1_dIdV_LR.npy").reshape(Nmu, Nvz, 5, 97)

        fig_cond, axes_c = plt.subplots(5, 1, figsize=(7.2, 9.8), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.0, 1.0, 1.0, 1.0], "hspace": 0.18})
        fig_cond.patch.set_facecolor(BG_DARK)

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
        for j in range(Nvz):
            if gt_cut[j]:
                ax0.plot([vz_vals[j] - 0.5 * dvz, vz_vals[j] + 0.5 * dvz], [-0.142, -0.142],
                         color=CYAN_PFAFF, lw=3.2, solid_capstyle="butt", zorder=6)
            if m_cut[j]:
                ax0.plot([vz_vals[j] - 0.5 * dvz, vz_vals[j] + 0.5 * dvz], [-0.154, -0.154],
                         color="#818cf8", lw=4.2, solid_capstyle="butt", zorder=7)
        ax0.set_ylim(-0.168, 0.16)
        ax0.set_ylabel("Energy (meV)", fontsize=8.5, color="white")
        ax0.set_title(rf"Spectra & Conductance along Cut ($\mu = {mu_cut:.3f}$ meV)", fontsize=10, fontweight="bold", color=CYAN_PFAFF, pad=5)
        ax0.tick_params(colors="white")
        ax0.grid(True, linestyle=":", alpha=0.25, color="white")

        bias_uV = c_v0.bias_grid * 1000.0
        extent_cut_uV = [vz_vals[0], vz_vals[-1], bias_uV[0], bias_uV[-1]]
        gll_cut = d_ll[mu_idx, :, 0, :].T
        grr_cut = d_rr[mu_idx, :, 0, :].T
        grl_cut = d_rl[mu_idx, :, 0, :].T
        glr_cut = d_lr[mu_idx, :, 0, :].T
        agrl_cut = 0.5 * (grl_cut - grl_cut[::-1, :])
        aglr_cut = 0.5 * (glr_cut - glr_cut[::-1, :])

        cond_maps = [
            (axes_c[1], gll_cut, r"$G_{LL}$ [$e^2/h$]", "viridis", 0.0, 2.0, r"Left bias [$\mu$V]"),
            (axes_c[2], grr_cut, r"$G_{RR}$ [$e^2/h$]", "viridis", 0.0, 2.0, r"Right bias [$\mu$V]"),
            (axes_c[3], agrl_cut, r"$\mathcal{A}(G_{RL})$", "PuOr_r", -0.10, 0.10, r"Left bias [$\mu$V]"),
            (axes_c[4], aglr_cut, r"$\mathcal{A}(G_{LR})$", "PuOr_r", -0.10, 0.10, r"Right bias [$\mu$V]"),
        ]

        for ax_m, cdata, lbl_m, cmap_m, vmin_m, vmax_m, ylbl in cond_maps:
            ax_m.set_facecolor("#030712")
            im_m = ax_m.imshow(cdata, origin="lower", extent=extent_cut_uV, aspect="auto", cmap=cmap_m, vmin=vmin_m, vmax=vmax_m)
            ax_m.axvline(vz_med, color=star_col, linestyle="--", lw=1.2, alpha=0.8, zorder=4)
            ax_m.set_ylabel(ylbl, fontsize=8.0, color="white")
            ax_m.tick_params(colors="white")

        axes_c[-1].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=9.0, color="white")
        fig_cond.tight_layout()
        cond_filename = f"cut_conductance_5panel_{v0_lbl}_isl1.png"
        fig_cond.savefig(out_dir / cond_filename, dpi=200)
        plt.close(fig_cond)

        # 3d. Point Deep-Dive: dI/dV at medoid
        fig_didv, ax_didv = plt.subplots(figsize=(8.2, 2.8))
        fig_didv.patch.set_facecolor(BG_DARK)
        ax_didv.set_facecolor("#030712")

        gll_pt = d_ll[mu_idx, vz_idx, 0, :]
        grr_pt = d_rr[mu_idx, vz_idx, 0, :]
        ax_didv.plot(bias_uV, gll_pt, color="#38bdf8", lw=2.0, label=r"$dI/dV_L$")
        ax_didv.plot(bias_uV, grr_pt, color="#f43f5e", lw=2.0, linestyle="--", label=r"$dI/dV_R$")
        ax_didv.set_title(rf"Zero-Bias Peak at Medoid ($\mu = {mu_cut:.3f}$, $V_z = {vz_med:.3f}$ meV)", fontsize=9.5, fontweight="bold", color="white")
        ax_didv.set_xlabel(r"Bias Voltage $V$ ($\mu$eV)", fontsize=8.5, color="white")
        ax_didv.set_ylabel(r"Conductance ($2e^2/h$)", fontsize=8.5, color="white")
        ax_didv.legend(loc="upper right", fontsize=7.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
        ax_didv.tick_params(colors="white")
        ax_didv.grid(True, linestyle=":", alpha=0.25, color="white")
        fig_didv.tight_layout()
        pt_didv_filename = f"deepdive_dIdV_{v0_lbl}_isl1.png"
        fig_didv.savefig(out_dir / pt_didv_filename, dpi=200)
        plt.close(fig_didv)

        # 3e. Point Deep-Dive: Wavefunctions
        fig_wf, ax_wf = plt.subplots(figsize=(8.2, 2.8))
        fig_wf.patch.set_facecolor(BG_DARK)
        ax_wf.set_facecolor("#030712")

        all_p = np.load(real_dir / "all_params.npz", allow_pickle=True)
        t_val = float(all_p["t"])
        gamma = float(all_p["gamma"])
        Delta0 = float(all_p["Delta0"])
        alpha = float(all_p["alpha"])
        Ls = int(all_p["Ls"])
        Vdisx = np.load(real_dir / "Vdisx.npy")

        syst_pt = hp.build_system_closed(t_val, mu_cut, gamma, Delta0, vz_med, alpha, Ls, Vdisx, a=1)
        _, evecs_pt = hp.solve_ham(syst_pt, k=4, solver_type="cpu")
        wf_l = np.abs(evecs_pt[:, 0])**2 + np.abs(evecs_pt[:, 1])**2
        wf_r = np.abs(evecs_pt[:, 2])**2 + np.abs(evecs_pt[:, 3])**2
        x_sites = np.arange(len(wf_l) // 4)
        wf_l_site = wf_l.reshape(-1, 4).sum(axis=1)
        wf_r_site = wf_r.reshape(-1, 4).sum(axis=1)

        ax_wf.plot(x_sites, wf_l_site, color="#38bdf8", lw=2.0, label=r"$|\psi_L|^2$ (Left MZM)")
        ax_wf.plot(x_sites, wf_r_site, color="#f43f5e", lw=2.0, label=r"$|\psi_R|^2$ (Right MZM)")
        ax_wf.set_title(rf"Majorana Wavefunctions (Medoid: $\mu = {mu_cut:.3f}$, $V_z = {vz_med:.3f}$ meV)", fontsize=9.5, fontweight="bold", color="white")
        ax_wf.set_xlabel("Wire Site $x$", fontsize=8.5, color="white")
        ax_wf.set_ylabel("Probability Density", fontsize=8.5, color="white")
        ax_wf.legend(loc="upper right", fontsize=7.5, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
        ax_wf.tick_params(colors="white")
        ax_wf.grid(True, linestyle=":", alpha=0.25, color="white")
        fig_wf.tight_layout()
        pt_wf_filename = f"deepdive_wavefunctions_{v0_lbl}_isl1.png"
        fig_wf.savefig(out_dir / pt_wf_filename, dpi=200)
        plt.close(fig_wf)

        # 3f. Point Deep-Dive: Dual Barrier Conductance Sweeps (4 Curves across both sweeps)
        bdata = c_v0.get_barrier_data()
        asym_ratios = bdata["asym_ratios"]
        bl_l_pt = bdata["bl_l"][mu_idx, vz_idx, :]
        bl_r_pt = bdata["bl_r"][mu_idx, vz_idx, :]
        br_l_pt = bdata["br_l"][mu_idx, vz_idx, :]
        br_r_pt = bdata["br_r"][mu_idx, vz_idx, :]

        win_corr = (asym_ratios >= -5.0 - 1e-9) & (asym_ratios <= 5.0 + 1e-9)
        num_l = np.sum(bl_l_pt[win_corr] * bl_r_pt[win_corr])
        den_l = np.linalg.norm(bl_l_pt[win_corr]) * np.linalg.norm(bl_r_pt[win_corr])
        c_l_val = float(num_l / den_l) if den_l > 1e-12 else 0.0

        num_r = np.sum(br_l_pt[win_corr] * br_r_pt[win_corr])
        den_r = np.linalg.norm(br_l_pt[win_corr]) * np.linalg.norm(br_r_pt[win_corr])
        c_r_val = float(num_r / den_r) if den_r > 1e-12 else 0.0

        fig_ba, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
        fig_ba.patch.set_facecolor(BG_DARK)

        # Left Barrier Sweep Subplot
        ax_l.set_facecolor("#030712")
        ax_l.plot(asym_ratios, bl_l_pt, color="#38bdf8", lw=2.0, label=r"$G_{LL}(U_L)$ [local]")
        ax_l.plot(asym_ratios, bl_r_pt, color="#f59e0b", lw=2.0, linestyle="--", label=r"$G_{RR}(U_L)$ [nonlocal]")
        ax_l.axvspan(-5.0, 5.0, color="#38bdf8", alpha=0.15, label=r"Window $[-5, 5]$ meV")
        ax_l.set_title(rf"Left Barrier Sweep ($U_L$): $\mathcal{{C}}_L = {c_l_val:.2f}$", fontsize=9.2, fontweight="bold", color="white")
        ax_l.set_xlabel(r"Asymmetry $\Delta U$ (meV)", fontsize=8.5, color="white")
        ax_l.set_ylabel(r"Conductance ($2e^2/h$)", fontsize=8.5, color="white")
        ax_l.set_xlim(-20.0, 20.0)
        ax_l.legend(loc="upper right", fontsize=7.2, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
        ax_l.tick_params(colors="white")
        ax_l.grid(True, linestyle=":", alpha=0.25, color="white")

        # Right Barrier Sweep Subplot
        ax_r.set_facecolor("#030712")
        ax_r.plot(asym_ratios, br_l_pt, color="#38bdf8", lw=2.0, linestyle="--", label=r"$G_{LL}(U_R)$ [nonlocal]")
        ax_r.plot(asym_ratios, br_r_pt, color="#f59e0b", lw=2.0, label=r"$G_{RR}(U_R)$ [local]")
        ax_r.axvspan(-5.0, 5.0, color="#38bdf8", alpha=0.15, label=r"Window $[-5, 5]$ meV")
        ax_r.set_title(rf"Right Barrier Sweep ($U_R$): $\mathcal{{C}}_R = {c_r_val:.2f}$", fontsize=9.2, fontweight="bold", color="white")
        ax_r.set_xlabel(r"Asymmetry $\Delta U$ (meV)", fontsize=8.5, color="white")
        ax_r.set_xlim(-20.0, 20.0)
        ax_r.legend(loc="upper right", fontsize=7.2, facecolor=BG_CARD, edgecolor="#334155", labelcolor="white")
        ax_r.tick_params(colors="white")
        ax_r.grid(True, linestyle=":", alpha=0.25, color="white")

        fig_ba.suptitle(rf"Dual Barrier Conductance Sweeps ($T = 40$ mK, Medoid: $\mu = {mu_cut:.3f}$, $V_z = {vz_med:.3f}$ meV)",
                        fontsize=10.2, fontweight="bold", color="white", y=0.98)
        fig_ba.tight_layout(rect=[0, 0, 1, 0.94])
        pt_ba_filename = f"deepdive_barrier_asym_{v0_lbl}_isl1.png"
        fig_ba.savefig(out_dir / pt_ba_filename, dpi=200)
        plt.close(fig_ba)

    return {"cluster_catalogs": cluster_catalogs}


def build_table_rows_html(temp_mK) -> str:
    temp_int = int(round(float(temp_mK)))
    local_csv = REPO_ROOT / "Outputs" / "Postprocessing" / "Local_Protocols_Leaderboard" / f"leaderboard_{temp_int}mK.csv"
    msft_csv = REPO_ROOT / "Outputs" / "Postprocessing" / "MSFT_Hybrids_N50_new" / f"leaderboard_{temp_int}mK.csv"

    df_local = pd.read_csv(local_csv)
    df_msft = pd.read_csv(msft_csv)

    regimes = [
        ("overall", "Pooled All Regimes"),
        ("V0_0_378", "Weak Disorder (V0 = 0.378 meV)"),
        ("V0_0_645", "Intermediate Disorder (V0 = 0.645 meV)"),
        ("V0_0_872", "Strong Disorder (V0 = 0.872 meV)"),
        ("V0_1_2",   "Severe Disorder (V0 = 1.200 meV)"),
    ]

    proto_display = {
        "tgp_std_bnd060": ("Standard TGP (bnd=60%)", "#94a3b8"),
        "tgp_nobnd_bnd000": ("TGP (No Boundary)", "#64748b"),
        "tgp_bnd000_lowcorr075_sym015": ("Weekly Baseline: tgp_bnd000_lowcorr075_sym015", "#818cf8"),
        "lowcorr080_curv500_5of5_gap10_sym025_c20": ("lowcorr080_curv500_5of5_gap10_sym025_c20", "#10b981"),
        "lax_base_strict_cluster50_core3": ("lax_base_strict_cluster50_core3", "#f59e0b"),
        "lax_k4_sym030_strict_cluster35_core3": ("lax_k4_sym030_strict_cluster35_core3", "#38bdf8"),
    }

    rows_html = []
    for reg_key, reg_label in regimes:
        rows_html.append(f"""
        <tr style="background-color: #1e293b; border-top: 2px solid #334155;">
          <td colspan="7" style="font-weight:700; color:#38bdf8; font-size:11.5px; padding:6px 10px;">
            {reg_label}
          </td>
        </tr>
        """)

        for p_name in ["tgp_std_bnd060", "tgp_nobnd_bnd000", "tgp_bnd000_lowcorr075_sym015", "lowcorr080_curv500_5of5_gap10_sym025_c20", "lax_base_strict_cluster50_core3", "lax_k4_sym030_strict_cluster35_core3"]:
            disp_name, color_hex = proto_display[p_name]

            if p_name in ["tgp_std_bnd060", "tgp_nobnd_bnd000", "tgp_bnd000_lowcorr075_sym015"]:
                reg_col = "scope" if "scope" in df_msft.columns else "disorder_regime"
                sub = df_msft[(df_msft["instance_name"] == p_name) & (df_msft[reg_col] == reg_key)]
            else:
                reg_col = "disorder_regime" if "disorder_regime" in df_local.columns else "scope"
                sub = df_local[(df_local["instance_name"] == p_name) & (df_local[reg_col] == reg_key)]

            if len(sub) == 0:
                continue

            r = sub.iloc[0]
            fdr = float(r["mean_FDR_active_only_pct"])
            tpr = float(r["mean_TPR_pct"])
            area = float(r["mean_island_area_px"])
            pred = float(r.get("mean_predicted_px", area * float(r.get("mean_island_count", 1.0))))
            pass_rate = float(r["pass_rate_pct"])

            if fdr <= 10.0:
                fdr_badge = f'<span class="badge-pass">{fdr:.2f}%</span>'
            elif fdr <= 20.0:
                fdr_badge = f'<span class="badge-warn">{fdr:.2f}%</span>'
            else:
                fdr_badge = f'<span class="badge-fail">{fdr:.2f}%</span>'

            pass_badge = f'<span class="badge-pass">{pass_rate:.1f}%</span>' if pass_rate >= 90.0 else f'<span class="badge-warn">{pass_rate:.1f}%</span>'

            is_highlight = p_name in ["lowcorr080_curv500_5of5_gap10_sym025_c20", "lax_k4_sym030_strict_cluster35_core3"]
            row_class = 'class="highlight-new"' if is_highlight else ''
            prefix = '★ ' if is_highlight else ''

            rows_html.append(f"""
            <tr {row_class}>
              <td style="font-weight:{'700' if is_highlight else '500'}; color:{color_hex};">{prefix}{disp_name}</td>
              <td style="text-align:center;">{pass_badge}</td>
              <td style="text-align:center;">{fdr_badge}</td>
              <td style="text-align:center; font-weight:600; color:#f1f5f9;">{tpr:.2f}%</td>
              <td style="text-align:center; color:#38bdf8; font-weight:700;">{area:.1f} px</td>
              <td style="text-align:center; color:#94a3b8;">{pred:.0f} px</td>
              <td style="text-align:center; color:#cbd5e1;">C={float(r['mean_MZM_confinement_C']):.3f} | S={float(r['mean_MZM_separability_S']):.3f}</td>
            </tr>
            """)

    return "".join(rows_html)


def build_front_slide_html(inst_name: str, meta: Dict[str, Any], total_slides: int) -> str:
    c_th = meta["c_th"]
    curv_th = meta["curv_th"]
    k_min = meta.get("k_min", 3)
    gap_mev = meta["gap_mev"]
    gap_uev = gap_mev * 1000.0
    s_sym = meta["s_sym"]
    min_sz = meta["min_island_size"]
    core = meta["core_shape"] or [2, 2]

    if s_sym is not None:
        sym_formula_html = f"<i>S</i><sub>sym</sub> &le; {s_sym:.2f}"
        sym_status_html = f"Active (&le; {s_sym:.2f})"
    else:
        sym_formula_html = "<i>S</i><sub>sym</sub> unconstrained"
        sym_status_html = "Disabled"

    if meta.get("family") == "MSFT_TGP_Hybrids":
        step1_title = "Base Candidate (bnd=0%)"
        step1_formula = "Boundaryless TGP"
        step1_p1 = "Boundary threshold: 0%"
        step1_p2 = "Preserves wire ends"
        step1_p3 = "High spatial sensitivity"

        step2_title = "Barrier Reflection Correlation"
        step2_formula = f"<i>C<sub>L</sub></i> &ge; {c_th:.2f} &amp; <i>C<sub>R</sub></i> &ge; {c_th:.2f}"
        step2_p1 = "Window: &Delta;<i>U</i> &isin; [-5, +5] meV"
        step2_p2 = "Mode: Both leads independently"
        step2_p3 = "Local gate invariance"

        step3_title = "Lead Symmetry Balance"
        step3_formula = f"<i>S</i><sub>sym</sub> &le; {s_sym:.2f}"
        step3_p1 = "Metric: |G_LL - G_RR| / &Sigma;G"
        step3_p2 = "Threshold: &le; 0.15"
        step3_p3 = "End-to-end symmetry"

        step4_title = "Transport Gap Threshold"
        step4_formula = f"&Delta; &ge; {gap_uev:.0f} &mu;eV"
        step4_p1 = "Bulk transconductance"
        step4_p2 = "Continuum protection"
        step4_p3 = f"{gap_mev:.3f} meV threshold"
    else:
        step1_title = "Barrier Reflection Correlation"
        step1_formula = f"<i>C<sub>L</sub></i> &ge; {c_th:.2f} &amp; <i>C<sub>R</sub></i> &ge; {c_th:.2f}"
        step1_p1 = "Window: &Delta;<i>U</i> &isin; [-5, +5] meV"
        step1_p2 = "Mode: Both leads independently"
        step1_p3 = "Local gate invariance"

        step2_title = "Consensus 3&omega; Curvature"
        step2_formula = f"<i>I</i><sub>3&omega;</sub> &le; -{curv_th:.0f} (<i>e</i><sup>2</sup>/<i>h</i>)/meV<sup>2</sup>"
        step2_p1 = f"Consensus: &ge; {k_min} of 5 cutter leads"
        step2_p2 = "Sides: Simultaneous <i>L</i> &amp; <i>R</i>"
        step2_p3 = "Negative zero-bias peak"

        step3_title = "Nonlocal Transport Gap"
        step3_formula = f"&Delta;<sub>transport</sub> &ge; {gap_uev:.0f} &mu;eV"
        step3_p1 = "Bulk transconductance"
        step3_p2 = "Continuum protection"
        step3_p3 = f"{gap_mev:.3f} meV threshold"

        step4_title = "Lead Symmetry Balance"
        step4_formula = sym_formula_html
        step4_p1 = "Metric: |<i>G<sub>LL</sub></i> - <i>G<sub>RR</sub></i>| / &Sigma;<i>G</i>"
        step4_p2 = f"Status: {sym_status_html}"
        step4_p3 = "End-to-end balance"

    return f"""
    <div class="slide-header-bar">
      <span class="slide-header-title">Protocol Specification & Architecture | Exact Operational Steps</span>
      <span class="slide-counter-pill">Slide -- of {total_slides}</span>
    </div>
    <div class="protocol-front-container">
      <div class="front-header">
        <div class="front-badge-row">
          <span class="front-badge">{meta.get('family', 'Low_Asym_Corr_Curv3w_Transport_Gap_Protocol')}</span>
          <span class="front-tag">Champion State: Realization 14 (100% Pass)</span>
          <span class="front-tag">Archetype: {meta.get('archetype', 'Topological Transport Gating')}</span>
        </div>
        <h1 class="front-title">{inst_name}</h1>
        <p class="front-subtitle">{meta['subtitle']}</p>
      </div>

      <div class="steps-grid">
        <!-- Step 1 -->
        <div class="step-card">
          <div class="step-card-header">
            <span class="step-num-pill">Step 1</span>
            <h4>{step1_title}</h4>
          </div>
          <div class="step-formula-box">
            {step1_formula}
          </div>
          <div class="step-params-list">
            <div><strong>Parameter 1:</strong> {step1_p1}</div>
            <div><strong>Parameter 2:</strong> {step1_p2}</div>
            <div><strong>Physical Target:</strong> {step1_p3}</div>
          </div>
        </div>

        <!-- Step 2 -->
        <div class="step-card">
          <div class="step-card-header">
            <span class="step-num-pill">Step 2</span>
            <h4>{step2_title}</h4>
          </div>
          <div class="step-formula-box">
            {step2_formula}
          </div>
          <div class="step-params-list">
            <div><strong>Parameter 1:</strong> {step2_p1}</div>
            <div><strong>Parameter 2:</strong> {step2_p2}</div>
            <div><strong>Physical Target:</strong> {step2_p3}</div>
          </div>
        </div>

        <!-- Step 3 -->
        <div class="step-card">
          <div class="step-card-header">
            <span class="step-num-pill">Step 3</span>
            <h4>{step3_title}</h4>
          </div>
          <div class="step-formula-box">
            {step3_formula}
          </div>
          <div class="step-params-list">
            <div><strong>Parameter 1:</strong> {step3_p1}</div>
            <div><strong>Parameter 2:</strong> {step3_p2}</div>
            <div><strong>Physical Target:</strong> {step3_p3}</div>
          </div>
        </div>

        <!-- Step 4 -->
        <div class="step-card">
          <div class="step-card-header">
            <span class="step-num-pill">Step 4</span>
            <h4>{step4_title}</h4>
          </div>
          <div class="step-formula-box">
            {step4_formula}
          </div>
          <div class="step-params-list">
            <div><strong>Parameter 1:</strong> {step4_p1}</div>
            <div><strong>Parameter 2:</strong> {step4_p2}</div>
            <div><strong>Physical Target:</strong> {step4_p3}</div>
          </div>
        </div>

        <!-- Step 5 -->
        <div class="step-card highlight-step">
          <div class="step-card-header">
            <span class="step-num-pill highlight-pill">Step 5</span>
            <h4 style="color:{meta['theme_color']};">Canonical C4 Spatial Filtering</h4>
          </div>
          <div class="step-formula-box highlight-box">
            Area &ge; {min_sz} px | Core [{core[0]}&times;{core[1]}]
          </div>
          <div class="step-params-list">
            <div><strong>Core Erosion:</strong> [{core[0]}&times;{core[1]}] kernel (8-conn)</div>
            <div><strong>Min Area:</strong> &ge; {min_sz} connected pixels</div>
            <div><strong>Physics:</strong> Prunes noise &amp; filaments</div>
          </div>
        </div>
      </div>

      <div class="front-footer-stats">
        <div class="front-stat-box">
          <span class="stat-label">Pass Rate (Mean across V0):</span>
          <span class="stat-value">{meta.get('stats_pass_rate', '100.0%')}</span>
        </div>
        <div class="front-stat-box">
          <span class="stat-label">Active Region FDR (30 / 40 mK):</span>
          <span class="stat-value">{meta.get('stats_fdr', '< 6.0%')}</span>
        </div>
        <div class="front-stat-box">
          <span class="stat-label">Mean Island Area:</span>
          <span class="stat-value">{meta.get('stats_area', '35+ px')}</span>
        </div>
        <div class="front-stat-box">
          <span class="stat-label">MZM Confinement / Separability:</span>
          <span class="stat-value">{meta.get('stats_mzm', 'C >= 0.988 | S >= 0.58')}</span>
        </div>
      </div>
    </div>
    """


def main():
    logger.info("Initializing Unified Weekly Presentation Build (2026-10-28)...")
    WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Generate Intro Figures on Realization 14
    intro_dir = FIG_ROOT / "intro"
    generate_intro_figures(intro_dir)

    # 2. Generate Protocol 0 Figures on Realization 14
    proto0_dir = FIG_ROOT / "tgp_bnd000_lowcorr075_sym015"
    generate_protocol0_figures(proto0_dir)

    # 3. Verify figures for Protocols 1, 2, 3 in FIG_ROOT
    proto_keys = [
        "tgp_bnd000_lowcorr075_sym015",
        "lowcorr080_curv500_5of5_gap10_sym025_c20",
        "lax_base_strict_cluster50_core3",
        "lax_k4_sym030_strict_cluster35_core3",
    ]
    for pk in proto_keys[1:]:
        p_dir = FIG_ROOT / pk
        src_dir = REPO_ROOT / f"Outputs/slides/{pk}/figures"
        logger.info("Copying/updating figures for %s from %s to %s...", pk, src_dir, p_dir)
        p_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.glob("*.png"):
            shutil.copy2(f, p_dir / f.name)

    # 4. Assemble Slides List
    slides = []
    total_slides = 78  # 4 intro + 4 * 18 + 2 tables = 78

    # --- Section 1: Intro Motivation Slides (Slides 1–4) ---
    slides.append({
        "title": "Finite-T Correlation Breakdown: 0 mK vs 40 mK (Realization 14)",
        "content": f"""
        <div class="slide-header-bar">
          <span class="slide-header-title">Finite-Temperature Correlation Breakdown | Realization 14 (V0 = 1.2 meV, 0 mK vs 40 mK)</span>
          <span class="slide-counter-pill">Slide -- of {total_slides}</span>
        </div>
        <div class="layout-single-centered">
          <div class="img-card flex-1">
            <img src="figures/intro/fig1_0mK_vs_40mK_legacy_correlation.png" alt="0 mK vs 40 mK Legacy Correlation">
          </div>
        </div>
        """
    })

    slides.append({
        "title": "Restoring Topological Sensitivity via Low-Asymmetry Window (Realization 14)",
        "content": f"""
        <div class="slide-header-bar">
          <span class="slide-header-title">Restoring Topological Correlation | Low-Asymmetry Window ([-5, +5] meV at 40 mK)</span>
          <span class="slide-counter-pill">Slide -- of {total_slides}</span>
        </div>
        <div class="layout-single-centered">
          <div class="img-card flex-1">
            <img src="figures/intro/fig2_40mK_legacy_vs_low_asymmetry.png" alt="Legacy vs Low-Asymmetry Correlation">
          </div>
        </div>
        """
    })

    slides.append({
        "title": "Binarized Low-Asymmetry Correlation Condition Phase Map (Realization 14)",
        "content": f"""
        <div class="slide-header-bar">
          <span class="slide-header-title">Binarized Low-Asymmetry Correlation Condition Phase Map | C_L, C_R &ge; 0.75 (T = 40 mK)</span>
          <span class="slide-counter-pill">Slide -- of {total_slides}</span>
        </div>
        <div class="layout-single-centered">
          <div class="img-card flex-1">
            <img src="figures/intro/fig3_binarized_low_asymmetry_correlation_40mK.png" alt="Binarized Low-Asymmetry Correlation">
          </div>
        </div>
        """
    })

    slides.append({
        "title": "Conjugate Lead Symmetry Balance Condition Phase Map (Realization 14)",
        "content": f"""
        <div class="slide-header-bar">
          <span class="slide-header-title">Conjugate Lead Symmetry Balance Condition | S_sym &le; 0.15 (T = 40 mK)</span>
          <span class="slide-counter-pill">Slide -- of {total_slides}</span>
        </div>
        <div class="layout-single-centered">
          <div class="img-card flex-1">
            <img src="figures/intro/fig4_symmetry_balance_phase_map_40mK.png" alt="Symmetry Balance Condition Phase Map">
          </div>
        </div>
        """
    })

    # --- Sections 2–5: Four Protocols (14 Slides each = 56 slides) ---
    v0_list = [
        ("V0_0_378", "Weak Disorder (V0 = 0.378 meV)"),
        ("V0_0_645", "Intermediate Disorder (V0 = 0.645 meV)"),
        ("V0_0_872", "Strong Disorder (V0 = 0.872 meV)"),
        ("V0_1_2",   "Severe Benchmark Disorder (V0 = 1.200 meV)"),
    ]

    for p_key in proto_keys:
        meta = PROTOCOLS[p_key]
        p_sub = f"figures/{p_key}"

        # 1. Front Specification Slide
        slides.append({
            "title": f"Protocol Specification & Architecture: {p_key}",
            "content": build_front_slide_html(p_key, meta, total_slides)
        })

        # 2–9. Observables & Step-by-Step Diagnostic Slides
        for v0_lbl, v0_name in v0_list:
            slides.append({
                "title": f"[{p_key} | {v0_lbl}] Direct Simulation Observables & Continuous Sensor Phase Maps (Realization 14, T = 40 mK)",
                "content": f"""
                <div class="slide-header-bar">
                  <span class="slide-header-title">[{p_key}] Direct Simulation Observables & Continuous Sensor Phase Maps | {v0_name}</span>
                  <span class="slide-counter-pill">Slide -- of {total_slides}</span>
                </div>
                <div class="layout-single-centered">
                  <div class="img-card flex-1">
                    <img src="{p_sub}/raw_observables_{v0_lbl}.png" alt="Direct Simulation Observables {v0_lbl}">
                  </div>
                </div>
                """
            })

            slides.append({
                "title": f"[{p_key} | {v0_lbl}] Step-by-Step Protocol Elimination (Realization 14, T = 40 mK)",
                "content": f"""
                <div class="slide-header-bar">
                  <span class="slide-header-title">[{p_key}] Step-by-Step Protocol Elimination | {v0_name}</span>
                  <span class="slide-counter-pill">Slide -- of {total_slides}</span>
                </div>
                <div class="layout-single-centered">
                  <div class="img-card flex-1">
                    <img src="{p_sub}/step_by_step_{v0_lbl}.png" alt="Step-by-Step Elimination {v0_lbl}">
                  </div>
                </div>
                """
            })

        # 6. Surviving Clusters 4-Panel Phase Map
        slides.append({
            "title": f"[{p_key}] Surviving Topological Clusters & Medoids Across All 4 Disorders (T = 40 mK)",
            "content": f"""
            <div class="slide-header-bar">
              <span class="slide-header-title">[{p_key}] Surviving Topological Clusters & Medoids Across All 4 Disorders (T = 40 mK)</span>
              <span class="slide-counter-pill">Slide -- of {total_slides}</span>
            </div>
            <div class="layout-single-centered">
              <div class="img-card flex-1">
                <img src="{p_sub}/phasemap_clusters_all_V0.png" alt="Surviving Clusters Phase Map">
              </div>
            </div>
            """
        })

        # 7–14. Cut Overview & Point Deep-Dives for Island 1 across all 4 disorders
        cut_order = [
            ("V0_0_645", "Intermediate Disorder (V0 = 0.645 meV)"),
            ("V0_0_378", "Weak Disorder (V0 = 0.378 meV)"),
            ("V0_0_872", "Strong Disorder (V0 = 0.872 meV)"),
            ("V0_1_2",   "Severe Benchmark Disorder (V0 = 1.200 meV)"),
        ]

        for v0_lbl, v0_name in cut_order:
            # Cut Overview
            slides.append({
                "title": f"[{p_key} | {v0_lbl}] Island 1 Medoid | Cut Overview (Conductance Multiplot & BdG Spectrum)",
                "content": f"""
                <div class="slide-header-bar">
                  <span class="slide-header-title">[{p_key}] Island 1 Medoid | Cut Overview | {v0_name}</span>
                  <span class="slide-counter-pill">Slide -- of {total_slides}</span>
                </div>
                <div class="layout-cut-overview">
                  <div class="col-stacked">
                    <div class="img-card flex-1">
                      <img src="{p_sub}/phasemap_cut_{v0_lbl}_isl1.png" alt="Phase Map & Cut Line">
                    </div>
                    <div class="img-card flex-1">
                      <img src="{p_sub}/cut_spectra_{v0_lbl}_isl1.png" alt="BdG Energy Spectrum along cut">
                    </div>
                  </div>
                  <div class="img-card">
                    <img src="{p_sub}/cut_conductance_5panel_{v0_lbl}_isl1.png" alt="5-Panel Conductance along cut">
                  </div>
                </div>
                """
            })

            # Point Deep-Dive
            slides.append({
                "title": f"[{p_key} | {v0_lbl}] Island 1 Medoid | Point Deep-Dive (dI/dV, Wavefunctions, Barrier Sweeps)",
                "content": f"""
                <div class="slide-header-bar">
                  <span class="slide-header-title">[{p_key}] Island 1 Medoid | Point Deep-Dive | {v0_name}</span>
                  <span class="slide-counter-pill">Slide -- of {total_slides}</span>
                </div>
                <div class="layout-point-deepdive">
                  <div class="img-card">
                    <img src="{p_sub}/cut_conductance_5panel_{v0_lbl}_isl1.png" alt="5-Panel Conductance along cut">
                  </div>
                  <div class="col-stacked-3">
                    <div class="img-card flex-1">
                      <img src="{p_sub}/deepdive_dIdV_{v0_lbl}_isl1.png" alt="Zero-Bias Peak at Medoid">
                    </div>
                    <div class="img-card flex-1">
                      <img src="{p_sub}/deepdive_wavefunctions_{v0_lbl}_isl1.png" alt="Majorana Wavefunctions">
                    </div>
                    <div class="img-card flex-1">
                      <img src="{p_sub}/deepdive_barrier_asym_{v0_lbl}_isl1.png" alt="Barrier Asymmetry Conductance Sweeps">
                    </div>
                  </div>
                </div>
                """
            })

    # --- Section 6: Master Verification Statistics Tables (Slides 61 & 62) ---
    t30_html = build_table_rows_html(30)
    slides.append({
        "title": "Master Verification Statistics Comparison at T = 30 mK",
        "content": f"""
        <div class="slide-header-bar">
          <span class="slide-header-title">Master Verification Statistics at T = 30 mK</span>
          <span class="slide-counter-pill">Slide -- of {total_slides}</span>
        </div>
        <div class="layout-table-slide">
          <div class="table-card">
            <h2>Post-Processing Verification Statistics Comparison across Regimes (T = 30 mK)</h2>
            <table class="data-table">
              <thead>
                <tr>
                  <th>Protocol Instance</th>
                  <th style="text-align:center;">Pass Rate</th>
                  <th style="text-align:center;">Active Region FDR</th>
                  <th style="text-align:center;">Mean TPR</th>
                  <th style="text-align:center;">Mean Island Area</th>
                  <th style="text-align:center;">Total Passed</th>
                  <th style="text-align:center;">MZM Quality (C / S)</th>
                </tr>
              </thead>
              <tbody>
                {t30_html}
              </tbody>
            </table>
          </div>
        </div>
        """
    })

    t40_html = build_table_rows_html(40)
    slides.append({
        "title": "Master Verification Statistics Comparison at T = 40 mK",
        "content": f"""
        <div class="slide-header-bar">
          <span class="slide-header-title">Master Verification Statistics at T = 40 mK</span>
          <span class="slide-counter-pill">Slide -- of {total_slides}</span>
        </div>
        <div class="layout-table-slide">
          <div class="table-card">
            <h2>Post-Processing Verification Statistics Comparison across Regimes (T = 40 mK)</h2>
            <table class="data-table">
              <thead>
                <tr>
                  <th>Protocol Instance</th>
                  <th style="text-align:center;">Pass Rate</th>
                  <th style="text-align:center;">Active Region FDR</th>
                  <th style="text-align:center;">Mean TPR</th>
                  <th style="text-align:center;">Mean Island Area</th>
                  <th style="text-align:center;">Total Passed</th>
                  <th style="text-align:center;">MZM Quality (C / S)</th>
                </tr>
              </thead>
              <tbody>
                {t40_html}
              </tbody>
            </table>
          </div>
        </div>
        """
    })

    # 5. Build Complete HTML
    actual_total = len(slides)
    logger.info("Total slides compiled: %d", actual_total)

    # Replace slide pill numbers
    for idx, s in enumerate(slides):
        s_num = idx + 1
        s["content"] = s["content"].replace(f"Slide -- of {total_slides}", f"Slide {s_num} of {actual_total}")
        s["content"] = re.sub(r"Slide (?:--|\d+) of \d+", f"Slide {s_num} of {actual_total}", s["content"])

    # Build dropdown options and section HTML
    options_html = []
    sections_html = []
    for idx, s in enumerate(slides):
        active_cls = " active" if idx == 0 else ""
        opt = f'<option value="{idx}">{idx + 1}: {s["title"]}</option>'
        options_html.append(opt)
        sec = f"""
        <section class="slide{active_cls}" data-title="{s['title']}">
          {s['content']}
        </section>
        """
        sections_html.append(sec)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Weekly Research Presentation (2026-10-28) — Unified Protocol Comparison</title>
<style>
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  body {{
    background-color: #0b1120;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    overflow: hidden;
    height: 100vh;
    width: 100vw;
    display: flex;
    flex-direction: column;
  }}

  /* Top Navigation Bar */
  header {{
    background-color: #0f172a;
    border-bottom: 1px solid #1e293b;
    padding: 0 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 42px;
    user-select: none;
    z-index: 100;
    flex-shrink: 0;
  }}
  .nav-left {{
    display: flex;
    align-items: center;
    gap: 12px;
    overflow: hidden;
    min-width: 0;
  }}
  .nav-title {{
    font-size: 13.5px;
    font-weight: 700;
    color: #38bdf8;
    white-space: nowrap;
    letter-spacing: 0.02em;
  }}
  .nav-sep {{
    color: #475569;
    font-size: 13px;
  }}
  .slide-title-display {{
    font-size: 13.5px;
    font-weight: 500;
    color: #f1f5f9;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 55vw;
  }}
  .nav-right {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
  }}
  .slide-select {{
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 12px;
    outline: none;
    max-width: 380px;
  }}
  .btn {{
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 3px 12px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .btn:hover {{
    background-color: #334155;
    color: #fff;
  }}
  .counter {{
    font-size: 12px;
    color: #94a3b8;
    min-width: 100px;
    text-align: center;
    font-variant-numeric: tabular-nums;
  }}
  .hint {{
    font-size: 11px;
    color: #64748b;
  }}

  /* Main Workspace */
  main {{
    flex: 1;
    position: relative;
    width: 100vw;
    height: calc(100vh - 42px);
    overflow: hidden;
  }}
  .slide {{
    display: none;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    padding: 8px 16px;
    background-color: #0b1120;
    overflow: hidden;
  }}
  .slide.active {{
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
  }}

  /* Slide Sub-Header Bar */
  .slide-header-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 10px 8px 10px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 6px;
    flex-shrink: 0;
  }}
  .slide-header-title {{
    color: #38bdf8;
    font-size: 16px;
    font-weight: 700;
  }}
  .slide-counter-pill {{
    background: #1e293b;
    color: #38bdf8;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 11.5px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }}

  /* Layout Containers */
  .layout-single-centered {{
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
    height: 100%;
    width: 100%;
    padding: 4px;
  }}

  .layout-cut-overview {{
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1.05fr;
    gap: 10px;
    min-height: 0;
    height: 100%;
    width: 100%;
  }}
  .col-stacked {{
    display: flex;
    flex-direction: column;
    gap: 8px;
    height: 100%;
    min-height: 0;
  }}

  .layout-point-deepdive {{
    flex: 1;
    display: grid;
    grid-template-columns: 1.05fr 1.35fr;
    gap: 10px;
    min-height: 0;
    height: 100%;
    width: 100%;
  }}
  .col-stacked-3 {{
    display: flex;
    flex-direction: column;
    gap: 6px;
    height: 100%;
    min-height: 0;
  }}

  .layout-table-slide {{
    flex: 1;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: 10px 14px;
    overflow-y: auto;
    height: 100%;
  }}

  /* Image card */
  .img-card {{
    background-color: transparent;
    border: none;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    min-height: 0;
    min-width: 0;
    width: 100%;
    height: 100%;
  }}
  .img-card img {{
    max-width: 100%;
    max-height: 100%;
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    cursor: zoom-in;
  }}
  .flex-1 {{
    flex: 1;
    min-height: 0;
  }}

  /* Front Specification Slide */
  .protocol-front-container {{
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 10px 16px;
    background: #030712;
    border: 1px solid #1e293b;
    border-radius: 8px;
    min-height: 0;
  }}
  .front-header {{
    margin-bottom: 8px;
  }}
  .front-badge-row {{
    display: flex;
    gap: 10px;
    margin-bottom: 6px;
  }}
  .front-badge {{
    background: #1e293b;
    color: #38bdf8;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .front-tag {{
    background: #0f172a;
    color: #94a3b8;
    border: 1px solid #334155;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
  }}
  .front-title {{
    font-size: 20px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 4px;
    font-family: monospace;
  }}
  .front-subtitle {{
    font-size: 12.5px;
    color: #94a3b8;
    line-height: 1.4;
  }}

  .steps-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin: 8px 0;
  }}
  .step-card {{
    background: #0b1120;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 10px;
    display: flex;
    flex-direction: column;
  }}
  .step-card.highlight-step {{
    border-color: #38bdf8;
    background: #0b172a;
  }}
  .step-card-header {{
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
  }}
  .step-num-pill {{
    background: #1e293b;
    color: #38bdf8;
    font-size: 10px;
    font-weight: 800;
    padding: 2px 6px;
    border-radius: 4px;
    text-transform: uppercase;
  }}
  .step-num-pill.highlight-pill {{
    background: #0284c7;
    color: white;
  }}
  .step-card-header h4 {{
    font-size: 11.5px;
    font-weight: 700;
    color: #f1f5f9;
  }}
  .step-formula-box {{
    background: #030712;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 6px 8px;
    font-family: monospace;
    font-size: 11px;
    color: #38bdf8;
    text-align: center;
    margin-bottom: 8px;
  }}
  .step-formula-box.highlight-box {{
    border-color: #0284c7;
    color: #38bdf8;
    font-weight: bold;
  }}
  .step-params-list {{
    font-size: 10.5px;
    color: #cbd5e1;
    line-height: 1.4;
  }}
  .step-params-list div {{
    margin-bottom: 3px;
  }}

  .front-footer-stats {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    border-top: 1px solid #1e293b;
    padding-top: 10px;
    margin-top: 6px;
  }}
  .front-stat-box {{
    background: #0b1120;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 6px 10px;
    display: flex;
    flex-direction: column;
  }}
  .stat-label {{
    font-size: 10.5px;
    color: #94a3b8;
    margin-bottom: 2px;
  }}
  .stat-value {{
    font-size: 12.5px;
    font-weight: 700;
    color: #f1f5f9;
  }}

  /* Data Table Card */
  .table-card {{
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 14px;
    width: 100%;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5);
  }}
  .table-card h2 {{
    font-size: 15px;
    color: #38bdf8;
    margin-bottom: 12px;
  }}
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
    text-align: left;
  }}
  .data-table th {{
    background-color: #1e293b;
    color: #f1f5f9;
    padding: 8px 10px;
    font-weight: 600;
    border-bottom: 2px solid #334155;
    white-space: nowrap;
  }}
  .data-table td {{
    padding: 6px 10px;
    border-bottom: 1px solid #1e293b;
    color: #cbd5e1;
    white-space: nowrap;
  }}
  .data-table tr.highlight-new {{
    background-color: #042f2e;
  }}
  .data-table tr.highlight-new td {{
    color: #f0fdf4;
    font-weight: 600;
  }}
  .badge-pass {{
    background-color: #064e3b;
    color: #34d399;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 10.5px;
  }}
  .badge-warn {{
    background-color: #713f12;
    color: #fde047;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 10.5px;
  }}
  .badge-fail {{
    background-color: #7f1d1d;
    color: #f87171;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 10.5px;
  }}

  /* Print to PDF Styles */
  @page {{
    size: 1920px 1080px;
    margin: 0;
  }}
  @media print {{
    * {{
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
    }}
    header, .slide-select, .nav-right, .hint, .counter {{
      display: none !important;
    }}
    html, body {{
      background-color: #0b1120 !important;
      width: 1920px !important;
      height: auto !important;
      margin: 0 !important;
      padding: 0 !important;
      overflow: visible !important;
    }}
    main {{
      width: 1920px !important;
      height: auto !important;
      overflow: visible !important;
      position: static !important;
    }}
    .slide {{
      display: flex !important;
      flex-direction: column !important;
      position: relative !important;
      width: 1920px !important;
      height: 1080px !important;
      page-break-after: always !important;
      break-after: page !important;
      page-break-inside: avoid !important;
      break-inside: avoid !important;
      overflow: hidden !important;
      padding: 12px 18px !important;
      box-sizing: border-box !important;
      background-color: #0b1120 !important;
    }}
    .layout-single-centered, .layout-cut-overview, .layout-point-deepdive, .layout-table-slide {{
      min-height: 0 !important;
      height: 100% !important;
      width: 100% !important;
    }}
  }}
</style>
</head>
<body>
<header>
  <div class="nav-left">
    <span class="nav-title">Weekly Research Slides (2026-10-28)</span>
    <span class="nav-sep">|</span>
    <span id="slideTitleDisplay" class="slide-title-display"></span>
  </div>
  <div class="nav-right">
    <span class="hint">&larr; / &rarr;</span>
    <button class="btn" onclick="prevSlide()" title="Previous slide (Left Arrow)">&ltrif;</button>
    <span id="slideCounter" class="counter">Slide 1 of {actual_total}</span>
    <button class="btn" onclick="nextSlide()" title="Next slide (Right Arrow / Space)">&rtrif;</button>
    <select id="slideSelect" class="slide-select" onchange="goToSlide(parseInt(this.value))">
      {"".join(options_html)}
    </select>
  </div>
</header>
<main>
  {"".join(sections_html)}
</main>
<script>
  let currentSlide = 0;
  const slides = document.querySelectorAll('.slide');
  const totalSlides = slides.length;
  const slideCounter = document.getElementById('slideCounter');
  const slideTitleDisplay = document.getElementById('slideTitleDisplay');
  const slideSelect = document.getElementById('slideSelect');

  function showSlide(index) {{
    if (index < 0) index = 0;
    if (index >= totalSlides) index = totalSlides - 1;
    slides[currentSlide].classList.remove('active');
    currentSlide = index;
    slides[currentSlide].classList.add('active');

    const title = slides[currentSlide].getAttribute('data-title') || '';
    slideTitleDisplay.textContent = title;
    slideCounter.textContent = `Slide ${{currentSlide + 1}} of ${{totalSlides}}`;
    slideSelect.value = currentSlide;
  }}

  function nextSlide() {{ showSlide(currentSlide + 1); }}
  function prevSlide() {{ showSlide(currentSlide - 1); }}
  function goToSlide(index) {{ showSlide(index); }}

  document.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {{
      nextSlide();
    }} else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{
      prevSlide();
    }} else if (e.key === 'Home') {{
      showSlide(0);
    }} else if (e.key === 'End') {{
      showSlide(totalSlides - 1);
    }}
  }});

  showSlide(0);
</script>
</body>
</html>
"""

    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info("Saved interactive HTML presentation (%d slides) to: %s", actual_total, HTML_OUT)

    # 6. Compile PDF via Google Chrome
    cmd = [
        "/usr/bin/google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUT}",
        "--window-size=1920,1080",
        f"file://{HTML_OUT}",
    ]
    logger.info("Compiling PDF via Chrome: %s", " ".join(shlex.quote(c) for c in cmd))
    res = subprocess.run(cmd).returncode
    if res != 0:
        logger.error("Error compiling PDF via chrome (exit %d)", res)
    else:
        logger.info("Successfully compiled %d-page PDF to: %s (%.2f MB)", actual_total, PDF_OUT, PDF_OUT.stat().st_size / (1024 * 1024))

    # Also sync root build_weekly_slides_2026_10_28.py
    shutil.copy(__file__, REPO_ROOT / "build_weekly_slides_2026_10_28.py")
    logger.info("Synchronized script to root build_weekly_slides_2026_10_28.py")


if __name__ == "__main__":
    main()
