#!/usr/bin/env python3
"""
generate_exact_vs_calibrated_slides.py

Comprehensive Slide Deck Generator comparing Zero-Temperature (T = 0 mK) Simulation Data:
  1. Tdis_pfaff5: Original Legacy Kwant Hamiltonian Solver (Hardcoded Barriers)
  2. Tdispfaff5_exact: Exact Energy-Dependent Self-Energy Solver (Hardcoded Barriers)
  3. Tdispfaff5_exact_calibrated: Exact Self-Energy Solver (Dynamically Calibrated Barriers G_N in [0.35, 0.90] e^2/h)

Observables Evaluated across 3-Panel Side-by-Side Slides:
  - 3w Local Conductance Curvature (Left Lead, Right Lead, Dual-Lead Mean, Peak Gating Mask)
  - 2w Nonlocal Conductance Derivative (Left Lead, Right Lead, Dual-Lead Mean)
  - Nonlocal Conductance Transport Gap (Pair Minimum, Left Lead, Right Lead)
  - Closed-System Spectral Excitation Gap & Pfaffian Topological Ground Truth
  - Low-Asymmetry Barrier Reflection Correlations (Left Sweep, Right Sweep, Dual-Lead Mean, Pair Minimum)
  - Multi-Observable 3-Step Protocol Candidate Phase Maps & Master Verification Table
"""

import os
import sys
import shutil
import subprocess
import shlex
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.Protocol_Elements.cache import RealizationFeatureCache
from src.Protocol_Elements.FPE.barrier_correlation import extract_barrier_reflection_correlations
from src.Protocol_Elements.FPE.transport_and_spectral_gap import extract_nonlocal_transport_gap_maps

# Directories
OUT_DIR = REPO_ROOT / "Outputs" / "slides" / "exact_vs_calibrated_comparison"
FIG_DIR = OUT_DIR / "figures"
HTML_OUT = OUT_DIR / "slides.html"
PDF_OUT = OUT_DIR / "slides.pdf"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
DATA_PATHS = {
    "legacy": REPO_ROOT / "Outputs" / "Data" / "Tdis_pfaff5_data" / "Tdis_pfaff5",
    "exact": REPO_ROOT / "Outputs" / "Data" / "Tdispfaff5_exact",
    "calibrated": REPO_ROOT / "Outputs" / "Data" / "Tdispfaff5_exact_calibrated",
}

DATA_LABELS = {
    "legacy": "Original (Legacy Solver, Hardcoded Barriers)",
    "exact": "Exact (Self-Energy Solver, Hardcoded Barriers)",
    "calibrated": "Exact Calibrated (Self-Energy Solver, Calibrated Barriers)",
}

DATA_SHORT = {
    "legacy": "Tdispfaff5 (Legacy)",
    "exact": "Tdispfaff5 (Exact Hardcoded)",
    "calibrated": "Tdispfaff5 (Exact Calibrated)",
}

# Theme Colors
BG_DARK = "#0b1120"
BG_CARD = "#0f172a"
CYAN_PFAFF = "#38bdf8"
DARK_SLATE = "#1e293b"
WHITE = "#f8fafc"
GRAY = "#94a3b8"

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "figure.facecolor": BG_DARK,
    "axes.facecolor": BG_CARD,
    "axes.edgecolor": "#334155",
    "axes.labelcolor": WHITE,
    "xtick.color": WHITE,
    "ytick.color": WHITE,
    "text.color": WHITE,
    "grid.color": "#1e293b",
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
})


def load_all_features() -> Dict[str, Dict[str, Any]]:
    """Load and compute all zero-temperature continuous observables for each dataset."""
    features = {}
    for key, path in DATA_PATHS.items():
        print(f"Loading {key} from {path}...")
        cache = RealizationFeatureCache(path, temperature_mK=0.0)
        s1 = cache.get_stage1_broadened_and_lockin(0.0)
        cL, cR = extract_barrier_reflection_correlations(cache, asym_min=-5.0, asym_max=5.0)
        gap_l, gap_r, gap_min = extract_nonlocal_transport_gap_maps(cache, use_thermal_broadened=True)
        topo_gap = cache.topological_gap
        gt = cache.pfaffian_mask

        # Lock-in observables
        l_2w = s1["L_2w_mean"]
        r_2w = s1["R_2w_mean"]
        mean_2w = 0.5 * (l_2w + r_2w)

        l_3w = s1["L_3w_mean"]
        r_3w = s1["R_3w_mean"]
        mean_3w = 0.5 * (l_3w + r_3w)

        # 3-step candidate mask (C >= 0.75, 3w <= 0, gap >= 10 ueV)
        c_mean = 0.5 * (cL + cR)
        c_min = np.minimum(cL, cR)
        step1 = (cL >= 0.75) & (cR >= 0.75)
        step2 = (l_3w <= 0.0) & (r_3w <= 0.0)
        step3 = (gap_min >= 0.010)
        cand_mask = step1 & step2 & step3

        # Candidate metrics
        tp = int(np.sum(cand_mask & gt))
        fp = int(np.sum(cand_mask & (~gt)))
        tpr = tp / np.sum(gt) * 100.0 if np.sum(gt) > 0 else 0.0
        fdr = fp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0

        features[key] = {
            "cache": cache,
            "mu_vals": cache.mu_vals,
            "vz_vals": cache.vz_vals,
            "gt": gt,
            "extent": [cache.vz_vals[0], cache.vz_vals[-1], cache.mu_vals[0], cache.mu_vals[-1]],
            "l_2w": l_2w,
            "r_2w": r_2w,
            "mean_2w": mean_2w,
            "l_3w": l_3w,
            "r_3w": r_3w,
            "mean_3w": mean_3w,
            "cL": cL,
            "cR": cR,
            "c_mean": c_mean,
            "c_min": c_min,
            "gap_l": gap_l,
            "gap_r": gap_r,
            "gap_min": gap_min,
            "topo_gap": topo_gap,
            "cand_mask": cand_mask,
            "tpr": tpr,
            "fdr": fdr,
            "tp": tp,
            "fp": fp,
        }
    return features


def plot_3panel_continuous(
    features: Dict[str, Dict[str, Any]],
    var_extractor,
    title: str,
    cbar_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    out_path: Path,
    slide_num: int,
    total_slides: int,
    section_name: str,
    note: str = "",
    clip_val: float = None,
    symlog: bool = False,
):
    """Render a 3-panel side-by-side comparison figure across the 3 simulation datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2), sharex=True, sharey=True, dpi=300, layout="constrained")
    fig.patch.set_facecolor(BG_DARK)

    im = None
    keys = ["legacy", "exact", "calibrated"]

    for idx, (ax, k) in enumerate(zip(axes, keys)):
        d = features[k]
        raw_arr = var_extractor(d)
        if clip_val is not None:
            arr = np.clip(raw_arr, -clip_val, clip_val)
        else:
            arr = raw_arr

        gt = d["gt"]
        extent = d["extent"]

        if symlog:
            norm = mcolors.SymLogNorm(linthresh=10.0, linscale=1.0, vmin=vmin, vmax=vmax, base=10)
            im = ax.imshow(arr, origin="lower", extent=extent, aspect="auto", cmap=cmap, norm=norm)
        else:
            im = ax.imshow(arr, origin="lower", extent=extent, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        # Ground Truth Pfaffian contour
        cs = ax.contour(d["vz_vals"], d["mu_vals"], gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=[1.8], linestyles=["--"])

        # Panel title and subtext
        topo_mean = np.mean(raw_arr[gt]) if np.sum(gt) > 0 else 0.0
        global_mean = np.mean(raw_arr)

        short_title = DATA_SHORT[k]
        ax.set_title(f"{short_title}\n[Topological Mean: {topo_mean:.3g} | Global Mean: {global_mean:.3g}]",
                     fontsize=11.5, fontweight="bold", color=WHITE, pad=10)
        ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11, fontweight="bold", labelpad=6)
        if idx == 0:
            ax.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11, fontweight="bold", labelpad=6)
        ax.tick_params(labelsize=10, colors=WHITE)
        ax.grid(True, linestyle=":", alpha=0.3, color="#475569")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.018, pad=0.015)
    cbar.set_label(cbar_label, fontsize=11, fontweight="bold", color=WHITE, labelpad=8)
    cbar.ax.tick_params(labelsize=9.5, colors=WHITE)
    cbar.outline.set_edgecolor("#334155")

    # Legend for contour
    legend_line = plt.Line2D([0], [0], color=CYAN_PFAFF, lw=2.0, ls="--", label=r"Pfaffian Ground Truth ($\mathcal{Q} = -1$)")
    axes[-1].legend(handles=[legend_line], loc="lower right", facecolor=BG_CARD, edgecolor="#334155", labelcolor=WHITE, fontsize=9.5)

    # Super title & annotation
    fig.suptitle(f"{section_name.upper()}  |  {title}", fontsize=14.5, fontweight="heavy", color=WHITE, y=1.03)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_3panel_confusion(
    features: Dict[str, Dict[str, Any]],
    mask_extractor,
    title: str,
    out_path: Path,
    slide_num: int,
    total_slides: int,
    section_name: str,
    note: str = "",
):
    """Render a 3-panel boolean confusion phase map (TP, FP, TN, FN) across the 3 datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2), sharex=True, sharey=True, dpi=300, layout="constrained")
    fig.patch.set_facecolor(BG_DARK)

    cmap_conf = mcolors.ListedColormap(["#0f172a", "#f59e0b", "#059669"])
    norm_conf = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_conf.N)

    keys = ["legacy", "exact", "calibrated"]

    for idx, (ax, k) in enumerate(zip(axes, keys)):
        d = features[k]
        mask = mask_extractor(d)
        gt = d["gt"]
        extent = d["extent"]

        c2d = np.zeros_like(mask, dtype=int)
        c2d[mask & (~gt)] = 1  # False Positive
        c2d[mask & gt] = 2     # True Positive

        ax.pcolormesh(d["vz_vals"], d["mu_vals"], c2d, cmap=cmap_conf, norm=norm_conf, shading="nearest")
        ax.contour(d["vz_vals"], d["mu_vals"], gt.astype(float), levels=[0.5], colors=[CYAN_PFAFF], linewidths=[1.8], linestyles=["--"])

        n_pass = int(np.sum(mask))
        tp = int(np.sum(mask & gt))
        fp = int(np.sum(mask & (~gt)))
        tpr = tp / np.sum(gt) * 100.0 if np.sum(gt) > 0 else 0.0
        fdr = fp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0

        short_title = DATA_SHORT[k]
        ax.set_title(f"{short_title}\n[Passed: {n_pass} px | TPR: {tpr:.1f}% | FDR: {fdr:.1f}%]",
                     fontsize=11.5, fontweight="bold", color=WHITE, pad=10)
        ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11, fontweight="bold", labelpad=6)
        if idx == 0:
            ax.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11, fontweight="bold", labelpad=6)
        ax.tick_params(labelsize=10, colors=WHITE)
        ax.grid(True, linestyle=":", alpha=0.3, color="#475569")

    legend_patches = [
        mpatches.Patch(facecolor="#059669", edgecolor="none", label="True Positive (TP)"),
        mpatches.Patch(facecolor="#f59e0b", edgecolor="none", label="False Positive (FP)"),
        plt.Line2D([0], [0], color=CYAN_PFAFF, lw=2.0, ls="--", label=r"Pfaffian Ground Truth ($\mathcal{Q} = -1$)"),
    ]
    axes[-1].legend(handles=legend_patches, loc="lower right", facecolor=BG_CARD, edgecolor="#334155", labelcolor=WHITE, fontsize=9.5)

    fig.suptitle(f"{section_name.upper()}  |  {title}", fontsize=14.5, fontweight="heavy", color=WHITE, y=1.03)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def render_title_slide(out_path: Path):
    """Render Slide 1: Deck Title Slide."""
    fig = plt.figure(figsize=(19.2, 10.8), dpi=300)
    fig.patch.set_facecolor(BG_DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Background subtle styling
    ax.text(0.5, 0.72, "EXACT SELF-ENERGY HAMILTONIAN &\nBARRIER CALIBRATION BENCHMARK",
            fontsize=36, fontweight="heavy", color=WHITE, ha="center", va="center", linespacing=1.2)
    ax.text(0.5, 0.56, "Zero-Temperature (T = 0 mK) Phase Space Comparison across 15,750 Parameter Points",
            fontsize=20, fontweight="medium", color=CYAN_PFAFF, ha="center", va="center")

    ax.text(0.5, 0.44,
            "1. Tdis_pfaff5 (Legacy Static Approximation, Hardcoded Barriers)\n"
            "2. Tdispfaff5_exact (Exact Energy-Dependent Self-Energy, Hardcoded Barriers)\n"
            "3. Tdispfaff5_exact_calibrated (Exact Self-Energy, Dynamically Calibrated G_N)",
            fontsize=15, color="#cbd5e1", ha="center", va="center", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=1.2", facecolor=BG_CARD, edgecolor="#334155", alpha=0.9))

    ax.text(0.5, 0.22,
            "Observables: 2w & 3w Harmonics | Nonlocal Transport Gap | Closed-System Excitation Gap | Low-Asymmetry Correlations",
            fontsize=13, color=GRAY, ha="center", va="center")
    ax.text(0.5, 0.12, "Disorder Realization: Tdis.npz (V0 = 1.20 meV, Ls = 300) | Full 225 x 70 Field-Chemical Potential Grid",
            fontsize=12, color="#64748b", ha="center", va="center")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def render_params_table_slide(out_path: Path):
    """Render Slide 2: Parameters and Barrier Comparison Slide."""
    fig = plt.figure(figsize=(19.2, 10.8), dpi=300)
    fig.patch.set_facecolor(BG_DARK)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.axis("off")

    ax.text(0.5, 0.94, "SIMULATION SPECIFICATIONS & BARRIER CONFIGURATIONS",
            fontsize=26, fontweight="heavy", color=WHITE, ha="center")
    ax.text(0.5, 0.89, "Quantitative Comparison of Numerical Models & Lead Potentials at T = 0 mK",
            fontsize=14, color=CYAN_PFAFF, ha="center")

    table_data = [
        ["Parameter / Property", "Tdis_pfaff5 (Legacy)", "Tdispfaff5_exact", "Tdispfaff5_exact_calibrated"],
        ["Disorder Realization", "Tdis.npz (Seed 5)", "Tdis.npz (Seed 5)", "Tdis.npz (Seed 5)"],
        ["Wire Length (Ls)", "300 sites (3.0 um)", "300 sites (3.0 um)", "300 sites (3.0 um)"],
        ["Disorder Amplitude (V0)", "1.20 meV", "1.20 meV", "1.20 meV"],
        ["Superconducting Gap (Delta0)", "0.30 meV", "0.30 meV", "0.30 meV"],
        ["Proximity Coupling (gamma)", "0.20 meV", "0.20 meV", "0.20 meV"],
        ["Effective Mass (m*)", "0.023 m_e", "0.023 m_e", "0.023 m_e"],
        ["Hamiltonian Solver", "Static Approximation", "Exact Energy-Dep Sigma(omega)", "Exact Energy-Dep Sigma(omega)"],
        ["Total Grid Points (N_mu x N_Vz)", "225 x 70 = 15,750", "225 x 70 = 15,750", "225 x 70 = 15,750"],
        ["Barrier Calibration", "False (Legacy Hardcoded)", "False (Hardcoded Match)", "True (Newton Root-Finder)"],
        ["Target Normal Conductance G_N", "N/A (Fixed Gates)", "N/A (Fixed Gates)", "[0.35, 0.49, 0.62, 0.76, 0.90] e^2/h"],
        ["Cutter Gate 1 (meV)", "4.3548 meV", "4.3548 meV", "3.1197 meV (G_N = 0.35)"],
        ["Cutter Gate 2 (meV)", "3.6774 meV", "3.6774 meV", "2.0505 meV (G_N = 0.49)"],
        ["Cutter Gate 3 (meV)", "3.0000 meV", "3.0000 meV", "1.3135 meV (G_N = 0.62)"],
        ["Cutter Gate 4 (meV)", "2.3710 meV", "2.3710 meV", "0.6723 meV (G_N = 0.76)"],
        ["Cutter Gate 5 (meV)", "1.6935 meV", "1.6935 meV", "0.1107 meV (G_N = 0.90)"],
        ["Barrier Sweep Window", "[-120, +120] meV (101 pts)", "[-120, +120] meV (43 pts)", "[-120, +120] meV (43 pts)"],
        ["Low-Asym Correlation Window", "[-10, +10] meV ([-5, 5])", "[-10, +10] meV ([-5, 5])", "[-10, +10] meV ([-5, 5])"],
    ]

    table = ax.table(cellText=table_data, loc="center", cellLoc="center",
                     colWidths=[0.28, 0.24, 0.24, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        if row == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color=CYAN_PFAFF, fontweight="bold")
        elif col == 0:
            cell.set_facecolor("#0f172a")
            cell.set_text_props(color=WHITE, fontweight="bold")
        else:
            cell.set_facecolor("#090d16" if row % 2 == 0 else "#0f172a")
            cell.set_text_props(color="#e2e8f0")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def render_section_divider(out_path: Path, section_title: str, subtitle: str, description_bullets: List[str]):
    """Render a clean section divider slide."""
    fig = plt.figure(figsize=(19.2, 10.8), dpi=300)
    fig.patch.set_facecolor(BG_DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.12, 0.75, section_title.upper(), fontsize=32, fontweight="heavy", color=WHITE, ha="left")
    ax.text(0.12, 0.67, subtitle, fontsize=18, fontweight="medium", color=CYAN_PFAFF, ha="left")

    bullet_text = "\n\n".join([f"•  {b}" for b in description_bullets])
    ax.text(0.12, 0.42, bullet_text, fontsize=15, color="#cbd5e1", ha="left", va="center", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=1.4", facecolor=BG_CARD, edgecolor="#334155", alpha=0.9))

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def render_summary_table_slide(features: Dict[str, Dict[str, Any]], out_path: Path):
    """Render Slide 26: Master Quantitative Summary Table Slide."""
    fig = plt.figure(figsize=(19.2, 10.8), dpi=300)
    fig.patch.set_facecolor(BG_DARK)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.axis("off")

    ax.text(0.5, 0.94, "MASTER QUANTITATIVE SUMMARY: ZERO TEMPERATURE COMPARISON",
            fontsize=24, fontweight="heavy", color=WHITE, ha="center")
    ax.text(0.5, 0.89, "Statistical Verification across 15,750 Points & 3-Step Protocol Candidates",
            fontsize=14, color=CYAN_PFAFF, ha="center")

    d_leg = features["legacy"]
    d_exc = features["exact"]
    d_cal = features["calibrated"]
    gt = d_leg["gt"]
    n_gt = int(np.sum(gt))

    def fmt_stat(arr, in_topo=True):
        sub = arr[gt] if in_topo else arr
        return f"{np.mean(sub):.3f} +/- {np.std(sub):.3f}"

    table_data = [
        ["Observable / Metric", "Tdis_pfaff5 (Legacy)", "Tdispfaff5_exact", "Tdispfaff5_exact_calibrated"],
        ["Ground Truth Topological Pixels", f"{n_gt} (19.64%)", f"{n_gt} (19.64%)", f"{n_gt} (19.64%)"],
        ["Mean L_3w in Topo Phase (cap 100)", fmt_stat(np.clip(d_leg["l_3w"], -100, 100)), fmt_stat(np.clip(d_exc["l_3w"], -100, 100)), fmt_stat(np.clip(d_cal["l_3w"], -100, 100))],
        ["Mean R_3w in Topo Phase (cap 100)", fmt_stat(np.clip(d_leg["r_3w"], -100, 100)), fmt_stat(np.clip(d_exc["r_3w"], -100, 100)), fmt_stat(np.clip(d_cal["r_3w"], -100, 100))],
        ["Mean L_2w in Topo Phase (e^2/h / meV)", fmt_stat(d_leg["l_2w"]), fmt_stat(d_exc["l_2w"]), fmt_stat(d_cal["l_2w"])],
        ["Mean R_2w in Topo Phase (e^2/h / meV)", fmt_stat(d_leg["r_2w"]), fmt_stat(d_exc["r_2w"]), fmt_stat(d_cal["r_2w"])],
        ["Nonlocal Transport Gap (mean, meV)", fmt_stat(d_leg["gap_min"]), fmt_stat(d_exc["gap_min"]), fmt_stat(d_cal["gap_min"])],
        ["Spectral Excitation Gap E_1 (mean, meV)", fmt_stat(d_leg["topo_gap"]), fmt_stat(d_exc["topo_gap"]), fmt_stat(d_cal["topo_gap"])],
        ["Low-Asym Reflection Corr C_L (mean)", fmt_stat(d_leg["cL"]), fmt_stat(d_exc["cL"]), fmt_stat(d_cal["cL"])],
        ["Low-Asym Reflection Corr C_R (mean)", fmt_stat(d_leg["cR"]), fmt_stat(d_exc["cR"]), fmt_stat(d_cal["cR"])],
        ["3-Step Protocol Candidate Pixels", f"{int(np.sum(d_leg['cand_mask']))}", f"{int(np.sum(d_exc['cand_mask']))}", f"{int(np.sum(d_cal['cand_mask']))}"],
        ["True Positives (TP)", f"{d_leg['tp']}", f"{d_exc['tp']}", f"{d_cal['tp']}"],
        ["False Positives (FP)", f"{d_leg['fp']}", f"{d_exc['fp']}", f"{d_cal['fp']}"],
        ["True Positive Rate (TPR)", f"{d_leg['tpr']:.2f}%", f"{d_exc['tpr']:.2f}%", f"{d_cal['tpr']:.2f}%"],
        ["False Discovery Rate (FDR)", f"{d_leg['fdr']:.2f}%", f"{d_exc['fdr']:.2f}%", f"{d_cal['fdr']:.2f}%"],
    ]

    table = ax.table(cellText=table_data, loc="center", cellLoc="center",
                     colWidths=[0.32, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1.0, 1.9)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        if row == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color=CYAN_PFAFF, fontweight="bold")
        elif col == 0:
            cell.set_facecolor("#0f172a")
            cell.set_text_props(color=WHITE, fontweight="bold")
        else:
            cell.set_facecolor("#090d16" if row % 2 == 0 else "#0f172a")
            cell.set_text_props(color="#e2e8f0")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def build_html_presentation(slide_metadata: List[Dict[str, str]], html_path: Path):
    """Compile HTML presentation with interactive navigation viewer."""
    total = len(slide_metadata)

    options_html = "\n".join([
        f'<option value="{idx}">Slide {idx+1}: {item["title"]}</option>'
        for idx, item in enumerate(slide_metadata)
    ])

    slides_html = "\n".join([
        f'''
        <div class="slide" id="slide-{idx}" data-title="{item["title"]}" data-section="{item["section"]}">
          <div class="slide-header-bar">
            <span class="slide-header-title">{item["section"]}  |  {item["title"]}</span>
            <span class="slide-counter-pill">Slide {idx+1} of {total}</span>
          </div>
          <div class="slide-content">
            <img src="figures/{item["filename"]}" alt="{item["title"]}">
          </div>
        </div>
        '''
        for idx, item in enumerate(slide_metadata)
    ])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exact Self-Energy & Barrier Calibration Benchmark (T = 0 mK)</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
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
  header {{
    background-color: #0f172a;
    border-bottom: 1px solid #1e293b;
    padding: 0 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 40px;
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
  .nav-sep {{ color: #475569; font-size: 13px; }}
  .slide-title-display {{
    font-size: 12.5px;
    font-weight: 500;
    color: #f1f5f9;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 50vw;
  }}
  .nav-right {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }}
  .slide-select {{
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11.5px;
    outline: none;
    max-width: 380px;
  }}
  .btn {{
    background-color: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .btn:hover {{
    background-color: #334155;
    color: #f8fafc;
    border-color: #475569;
  }}
  .counter {{
    font-size: 12px;
    color: #94a3b8;
    min-width: 90px;
    text-align: center;
    font-variant-numeric: tabular-nums;
  }}
  main {{
    flex: 1;
    position: relative;
    width: 100vw;
    height: calc(100vh - 40px);
    overflow: hidden;
  }}
  .slide {{
    display: none;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    padding: 6px 14px;
    background-color: #0b1120;
    overflow: hidden;
  }}
  .slide.active {{
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
  }}
  .slide-header-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
    flex-shrink: 0;
    height: 22px;
  }}
  .slide-header-title {{
    font-size: 12.5px;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .slide-counter-pill {{
    font-size: 11px;
    font-weight: 700;
    color: #cbd5e1;
    background: #1e293b;
    padding: 2px 9px;
    border-radius: 12px;
    border: 1px solid #334155;
  }}
  .slide-content {{
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
    height: calc(100% - 26px);
  }}
  .slide-content img {{
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
    object-fit: contain;
    border-radius: 4px;
  }}
  @media print {{
    body {{ background: #0b1120 !important; }}
    header {{ display: none !important; }}
    main {{ height: auto !important; overflow: visible !important; }}
    .slide {{
      display: flex !important;
      position: relative !important;
      width: 100vw !important;
      height: 100vh !important;
      page-break-after: always;
      break-after: page;
      padding: 0 !important;
    }}
    .slide-content img {{
      width: 100vw !important;
      height: 100vh !important;
      object-fit: contain !important;
    }}
  }}
</style>
</head>
<body>
<header>
  <div class="nav-left">
    <span class="nav-title">AGY RESEARCH</span>
    <span class="nav-sep">/</span>
    <span class="slide-title-display" id="slideTitleDisplay">Slide 1 of {total}</span>
  </div>
  <div class="nav-right">
    <button class="btn" onclick="prevSlide()" title="Previous slide (Left Arrow)">&#8592; Prev</button>
    <span class="counter" id="slideCounter">Slide 1 of {total}</span>
    <button class="btn" onclick="nextSlide()" title="Next slide (Right Arrow)">Next &#8594;</button>
    <select class="slide-select" id="slideSelect" onchange="goToSlide(parseInt(this.value))">
      {options_html}
    </select>
  </div>
</header>
<main>
  {slides_html}
</main>
<script>
  let currentSlide = 0;
  const totalSlides = {total};
  const slides = document.querySelectorAll('.slide');
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
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved interactive HTML presentation ({total} slides) to: {html_path}")


def main():
    print("=" * 80)
    print("STARTING EXACT VS. CALIBRATED SLIDE DECK GENERATION (T = 0 mK)")
    print("=" * 80)

    # 1. Load data
    features = load_all_features()

    TOTAL_SLIDES = 26
    slide_meta = []

    def register(filename: str, title: str, section: str):
        slide_meta.append({
            "filename": filename,
            "title": title,
            "section": section,
        })

    # Slide 1: Title
    s1_name = "slide_01_title.png"
    render_title_slide(FIG_DIR / s1_name)
    register(s1_name, "Title: Benchmark Overview", "Overview")

    # Slide 2: Parameters Table
    s2_name = "slide_02_parameters_table.png"
    render_params_table_slide(FIG_DIR / s2_name)
    register(s2_name, "Simulation Specifications & Barrier Table", "Overview")

    # Section 1: 3w Local Conductance Curvature
    s3_name = "slide_03_section_3w.png"
    render_section_divider(
        FIG_DIR / s3_name,
        "Section 1: 3w Local Conductance Curvature",
        r"Zero-Bias Second Derivative of Local Andreev Conductance $\partial^2 G / \partial V^2$",
        [
            r"Local zero-bias Andreev conductance curvature $I_{3\omega} = \left.\frac{\partial^2 G}{\partial V^2}\right|_{V=0}$ is the primary spectroscopic probe of zero-energy Majorana bound states.",
            r"Topological Majorana modes create zero-bias conductance peaks, corresponding to strongly negative curvature ($I_{3\omega} \leq 0$ or $I_{3\omega} \leq -100$).",
            "Comparing Left lead, Right lead, Dual-lead mean, and strict joint peak gating ($L_{3\omega} \leq 0 & R_{3\omega} \leq 0$) across Legacy, Exact Hardcoded, and Exact Calibrated solvers.",
        ]
    )
    register(s3_name, "Section 1 Divider: 3w Curvature", "Section 1: 3w Curvature")

    # Slide 4: 3w Left Lead (The user's specific prompt example!)
    s4_name = "slide_04_3w_left_lead.png"
    plot_3panel_continuous(
        features,
        lambda d: d["l_3w"],
        r"Average $3\omega$ Curvature — Left Lead ($L_{3\omega}$)",
        r"$L_{3\omega}$ Curvature [cap: $\pm 100$]",
        "bwr", -100.0, 100.0,
        FIG_DIR / s4_name,
        4, TOTAL_SLIDES,
        "Section 1: 3w Curvature",
        clip_val=100.0
    )
    register(s4_name, "Average 3w Curvature: Left Lead", "Section 1: 3w Curvature")

    # Slide 5: 3w Right Lead
    s5_name = "slide_05_3w_right_lead.png"
    plot_3panel_continuous(
        features,
        lambda d: d["r_3w"],
        r"Average $3\omega$ Curvature — Right Lead ($R_{3\omega}$)",
        r"$R_{3\omega}$ Curvature [cap: $\pm 100$]",
        "bwr", -100.0, 100.0,
        FIG_DIR / s5_name,
        5, TOTAL_SLIDES,
        "Section 1: 3w Curvature",
        clip_val=100.0
    )
    register(s5_name, "Average 3w Curvature: Right Lead", "Section 1: 3w Curvature")

    # Slide 6: 3w Dual-Lead Mean
    s6_name = "slide_06_3w_mean.png"
    plot_3panel_continuous(
        features,
        lambda d: d["mean_3w"],
        r"Average $3\omega$ Curvature — Dual-Lead Mean $\frac{1}{2}(L_{3\omega} + R_{3\omega})$",
        r"Mean $3\omega$ Curvature [cap: $\pm 100$]",
        "bwr", -100.0, 100.0,
        FIG_DIR / s6_name,
        6, TOTAL_SLIDES,
        "Section 1: 3w Curvature",
        clip_val=100.0
    )
    register(s6_name, "Average 3w Curvature: Dual-Lead Mean", "Section 1: 3w Curvature")

    # Slide 7: 3w Joint Peak Mask (Both <= 0)
    s7_name = "slide_07_3w_peak_mask.png"
    plot_3panel_confusion(
        features,
        lambda d: (d["l_3w"] <= 0.0) & (d["r_3w"] <= 0.0),
        r"Zero-Bias $3\omega$ Peak Pass Region ($L_{3\omega} \leq 0 & R_{3\omega} \leq 0$)",
        FIG_DIR / s7_name,
        7, TOTAL_SLIDES,
        "Section 1: 3w Curvature",
    )
    register(s7_name, "Joint 3w Peak Gating Phase Map", "Section 1: 3w Curvature")

    # Section 2: 2w Nonlocal Conductance Derivative
    s8_name = "slide_08_section_2w.png"
    render_section_divider(
        FIG_DIR / s8_name,
        "Section 2: 2w Nonlocal Conductance Derivative",
        r"Zero-Bias First Derivative of Nonlocal Conductance $\partial G / \partial V$",
        [
            r"Nonlocal conductance zero-bias slope $I_{2\omega} = \left.\frac{\partial G_{RL/LR}}{\partial V}\right|_{V=0}$ probes asymmetry in nonlocal transmission.",
            r"In the Microsoft TGP architecture, $L_{2\omega} \equiv I_{2\omega, RL}$ (current in right lead from left bias) and $R_{2\omega} \equiv I_{2\omega, LR}$.",
            "Evaluated across all 5 cutter gate configurations and compared at T = 0 mK.",
        ]
    )
    register(s8_name, "Section 2 Divider: 2w Derivative", "Section 2: 2w Derivative")

    # Slide 9: 2w Left Lead
    s9_name = "slide_09_2w_left_lead.png"
    plot_3panel_continuous(
        features,
        lambda d: d["l_2w"],
        r"Average $2\omega$ Derivative — Left Lead ($L_{2\omega} = I_{2\omega, RL}$)",
        r"$L_{2\omega}$ Slope [cap: $\pm 25\ e^2/h/\mathrm{meV}$]",
        "bwr", -25.0, 25.0,
        FIG_DIR / s9_name,
        9, TOTAL_SLIDES,
        "Section 2: 2w Derivative",
        clip_val=25.0
    )
    register(s9_name, "Average 2w Derivative: Left Lead", "Section 2: 2w Derivative")

    # Slide 10: 2w Right Lead
    s10_name = "slide_10_2w_right_lead.png"
    plot_3panel_continuous(
        features,
        lambda d: d["r_2w"],
        r"Average $2\omega$ Derivative — Right Lead ($R_{2\omega} = I_{2\omega, LR}$)",
        r"$R_{2\omega}$ Slope [cap: $\pm 25\ e^2/h/\mathrm{meV}$]",
        "bwr", -25.0, 25.0,
        FIG_DIR / s10_name,
        10, TOTAL_SLIDES,
        "Section 2: 2w Derivative",
        clip_val=25.0
    )
    register(s10_name, "Average 2w Derivative: Right Lead", "Section 2: 2w Derivative")

    # Slide 11: 2w Dual-Lead Mean
    s11_name = "slide_11_2w_mean.png"
    plot_3panel_continuous(
        features,
        lambda d: d["mean_2w"],
        r"Average $2\omega$ Derivative — Dual-Lead Mean $\frac{1}{2}(L_{2\omega} + R_{2\omega})$",
        r"Mean $2\omega$ Slope [cap: $\pm 25\ e^2/h/\mathrm{meV}$]",
        "bwr", -25.0, 25.0,
        FIG_DIR / s11_name,
        11, TOTAL_SLIDES,
        "Section 2: 2w Derivative",
        clip_val=25.0
    )
    register(s11_name, "Average 2w Derivative: Dual-Lead Mean", "Section 2: 2w Derivative")

    # Section 3: Nonlocal Transport Gap
    s12_name = "slide_12_section_transport_gap.png"
    render_section_divider(
        FIG_DIR / s12_name,
        "Section 3: Open-System Nonlocal Transport Gap",
        r"Nonlocal Conductance Transport Gap $\Delta_{\mathrm{trans}}^{\mathrm{NL}}$ from Antisymmetric Nonlocal Bias Traces",
        [
            r"Extracted from antisymmetrized nonlocal conductances $A(G_{RL}(V)) = \frac{1}{2}|G_{RL}(V) - G_{RL}(-V)|$ and $A(G_{LR}(V))$ using the Microsoft TGP dynamic threshold algorithm.",
            r"Threshold criterion: first bias crossing above $\max(0.05 \cdot \max |A(G)|, 10^{-4}\ e^2/h)$.",
            r"Physical significance: probes bulk quasiparticle transmission gap through the hybrid wire in open transport conditions.",
        ]
    )
    register(s12_name, "Section 3 Divider: Transport Gap", "Section 3: Transport Gap")

    # Slide 13: Transport Gap Pair Minimum
    s13_name = "slide_13_transport_gap_min.png"
    plot_3panel_continuous(
        features,
        lambda d: d["gap_min"] * 1e3,  # in ueV
        r"Nonlocal Transport Gap — Pair Minimum $\min(\Delta_{\mathrm{trans}}^L, \Delta_{\mathrm{trans}}^R)$",
        r"$\Delta_{\mathrm{trans}}^{\mathrm{NL}}$ ($\mu$eV)",
        "inferno", 0.0, 140.0,
        FIG_DIR / s13_name,
        13, TOTAL_SLIDES,
        "Section 3: Transport Gap",
    )
    register(s13_name, "Nonlocal Transport Gap: Pair Minimum", "Section 3: Transport Gap")

    # Slide 14: Transport Gap Left Lead
    s14_name = "slide_14_transport_gap_left.png"
    plot_3panel_continuous(
        features,
        lambda d: np.mean(d["gap_l"], axis=0) * 1e3,
        r"Nonlocal Transport Gap — Left Lead $\Delta_{\mathrm{trans}}^L$ (from $G_{RL}$)",
        r"$\Delta_{\mathrm{trans}}^L$ ($\mu$eV)",
        "inferno", 0.0, 140.0,
        FIG_DIR / s14_name,
        14, TOTAL_SLIDES,
        "Section 3: Transport Gap",
    )
    register(s14_name, "Nonlocal Transport Gap: Left Lead", "Section 3: Transport Gap")

    # Slide 15: Transport Gap Right Lead
    s15_name = "slide_15_transport_gap_right.png"
    plot_3panel_continuous(
        features,
        lambda d: np.mean(d["gap_r"], axis=0) * 1e3,
        r"Nonlocal Transport Gap — Right Lead $\Delta_{\mathrm{trans}}^R$ (from $G_{LR}$)",
        r"$\Delta_{\mathrm{trans}}^R$ ($\mu$eV)",
        "inferno", 0.0, 140.0,
        FIG_DIR / s15_name,
        15, TOTAL_SLIDES,
        "Section 3: Transport Gap",
    )
    register(s15_name, "Nonlocal Transport Gap: Right Lead", "Section 3: Transport Gap")

    # Section 4: Closed-System Spectral Excitation Gap
    s16_name = "slide_16_section_excitation_gap.png"
    render_section_divider(
        FIG_DIR / s16_name,
        "Section 4: Closed-System Spectral Excitation Gap",
        r"BdG Spectrum Excitation Gap $E_1$ & Pfaffian Ground Truth Topology",
        [
            r"Closed-system BdG spectral excitation gap $E_1(\mu, V_z)$ (`topological_gap.npy`) computed from the 4 lowest positive eigenvalues of the isolated wire Hamiltonian.",
            r"Pfaffian topological invariant $\mathcal{Q} \in \{-1, +1\}$ computed via scattering matrix parity (`pdi_data.npy`).",
            r"The closed-system gap is intrinsic to the wire Hamiltonian; barrier calibration only affects leads and open boundary coupling, leaving $E_1$ and $\mathcal{Q}$ invariant.",
        ]
    )
    register(s16_name, "Section 4 Divider: Excitation Gap", "Section 4: Excitation Gap")

    # Slide 17: Closed-System Excitation Gap
    s17_name = "slide_17_excitation_gap.png"
    plot_3panel_continuous(
        features,
        lambda d: d["topo_gap"] * 1e3,  # in ueV
        r"Closed-System BdG Spectral Excitation Gap $E_1$ (`topological_gap.npy`)",
        r"$E_1$ Gap ($\mu$eV)",
        "viridis", 0.0, 140.0,
        FIG_DIR / s17_name,
        17, TOTAL_SLIDES,
        "Section 4: Excitation Gap",
    )
    register(s17_name, "Spectral Excitation Gap E_1", "Section 4: Excitation Gap")

    # Slide 18: Ground Truth Topological Pfaffian Phase Map
    s18_name = "slide_18_pfaffian_ground_truth.png"
    plot_3panel_confusion(
        features,
        lambda d: d["gt"],
        r"Pfaffian Ground Truth Topological Phase ($\mathcal{Q} = -1$)",
        FIG_DIR / s18_name,
        18, TOTAL_SLIDES,
        "Section 4: Excitation Gap",
    )
    register(s18_name, "Ground Truth Pfaffian Phase Map", "Section 4: Excitation Gap")

    # Section 5: Low-Asymmetry Barrier Reflection Correlations
    s19_name = "slide_19_section_correlation.png"
    render_section_divider(
        FIG_DIR / s19_name,
        "Section 5: Low-Asymmetry Barrier Reflection Correlations",
        r"Lead Reflection Correlation $\mathcal{C}_{[-5, +5]}$ in the Narrow Barrier Offset Window",
        [
            r"Reflection correlation $\mathcal{C}_\alpha = \frac{\mathbf{G}_{LL} \cdot \mathbf{G}_{RR}}{\|\mathbf{G}_{LL}\| \|\mathbf{G}_{RR}\|}$ evaluated within the low-asymmetry offset window $\Delta U \in [-10, +10]\,\mathrm{meV}$ ($a \in [-5, +5]$).",
            r"Pinch-off protection: narrow low-asymmetry gating avoids severe tunnel-coupling collapse at large barrier offsets while maintaining sensitivity to topological edge states.",
            "Comparing Left lead sweep ($C_L$), Right lead sweep ($C_R$), Dual-lead mean, and Strict pair minimum ($C_L \geq 0.75 & C_R \geq 0.75$).",
        ]
    )
    register(s19_name, "Section 5 Divider: Reflection Correlation", "Section 5: Correlation")

    # Slide 20: Low-Asym Correlation Left Sweep
    s20_name = "slide_20_corr_left.png"
    plot_3panel_continuous(
        features,
        lambda d: d["cL"],
        r"Low-Asymmetry Reflection Correlation — Left Sweep $C_L([-5, +5])$",
        r"Correlation $C_L$",
        "magma", 0.4, 1.0,
        FIG_DIR / s20_name,
        20, TOTAL_SLIDES,
        "Section 5: Correlation",
    )
    register(s20_name, "Low-Asymmetry Correlation: Left Lead", "Section 5: Correlation")

    # Slide 21: Low-Asym Correlation Right Sweep
    s21_name = "slide_21_corr_right.png"
    plot_3panel_continuous(
        features,
        lambda d: d["cR"],
        r"Low-Asymmetry Reflection Correlation — Right Sweep $C_R([-5, +5])$",
        r"Correlation $C_R$",
        "magma", 0.4, 1.0,
        FIG_DIR / s21_name,
        21, TOTAL_SLIDES,
        "Section 5: Correlation",
    )
    register(s21_name, "Low-Asymmetry Correlation: Right Lead", "Section 5: Correlation")

    # Slide 22: Low-Asym Correlation Dual-Lead Mean
    s22_name = "slide_22_corr_mean.png"
    plot_3panel_continuous(
        features,
        lambda d: d["c_mean"],
        r"Low-Asymmetry Reflection Correlation — Dual-Lead Mean $\frac{1}{2}(C_L + C_R)$",
        r"Mean Correlation $\bar{C}$",
        "magma", 0.4, 1.0,
        FIG_DIR / s22_name,
        22, TOTAL_SLIDES,
        "Section 5: Correlation",
    )
    register(s22_name, "Low-Asymmetry Correlation: Dual-Lead Mean", "Section 5: Correlation")

    # Slide 23: Low-Asym Correlation Strict Pair Minimum Mask
    s23_name = "slide_23_corr_strict_mask.png"
    plot_3panel_confusion(
        features,
        lambda d: (d["cL"] >= 0.75) & (d["cR"] >= 0.75),
        r"Strict Reflection Correlation Pass Region ($C_L \geq 0.75 & C_R \geq 0.75$)",
        FIG_DIR / s23_name,
        23, TOTAL_SLIDES,
        "Section 5: Correlation",
    )
    register(s23_name, "Strict Reflection Correlation Pass Map", "Section 5: Correlation")

    # Section 6: Protocol Synthesis & Benchmark Summary
    s24_name = "slide_24_section_synthesis.png"
    render_section_divider(
        FIG_DIR / s24_name,
        "Section 6: Multi-Observable Protocol Synthesis",
        r"Integrated 3-Step Topological Protocol ($C \geq 0.75 & 3\omega \leq 0 & \Delta_{\mathrm{trans}} \geq 10\,\mu\mathrm{eV}$)",
        [
            r"Combines: (1) Low-asymmetry reflection correlation $\geq 0.75$, (2) Local zero-bias conductance curvature $\leq 0$, (3) Nonlocal transport gap $\geq 10\,\mu\mathrm{eV}$.",
            r"Measures the simultaneous topological selectivity and false discovery suppression across Legacy, Exact Hardcoded, and Exact Calibrated datasets.",
            r"Demonstrates the physical impact of dynamic barrier calibration on topological gap yield and diagnostic certainty.",
        ]
    )
    register(s24_name, "Section 6 Divider: Protocol Synthesis", "Section 6: Synthesis")

    # Slide 25: 3-Step Protocol Candidates
    s25_name = "slide_25_3step_protocol_candidates.png"
    plot_3panel_confusion(
        features,
        lambda d: d["cand_mask"],
        r"3-Step Protocol Candidate Phase Map ($C \geq 0.75 & 3\omega \leq 0 & \Delta_{\mathrm{trans}} \geq 10\,\mu\mathrm{eV}$)",
        FIG_DIR / s25_name,
        25, TOTAL_SLIDES,
        "Section 6: Synthesis",
    )
    register(s25_name, "3-Step Protocol Candidate Phase Map", "Section 6: Synthesis")

    # Slide 26: Master Quantitative Summary Table
    s26_name = "slide_26_master_summary_table.png"
    render_summary_table_slide(features, FIG_DIR / s26_name)
    register(s26_name, "Master Quantitative Summary Table", "Section 6: Synthesis")

    # 4. Compile HTML slide deck
    build_html_presentation(slide_meta, HTML_OUT)

    # 5. Compile PDF via Google Chrome
    cmd = [
        "/usr/bin/google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUT}",
        "--window-size=1920,1080",
        f"file://{HTML_OUT}",
    ]
    print(f"Compiling PDF via Chrome: {' '.join(shlex.quote(c) for c in cmd)}")
    res = subprocess.run(cmd).returncode
    if res != 0:
        print(f"Error compiling PDF via chrome (exit {res})")
    else:
        print(f"Successfully compiled {TOTAL_SLIDES}-page PDF to: {PDF_OUT} ({PDF_OUT.stat().st_size / (1024 * 1024):.2f} MB)")

    print("=" * 80)
    print("SLIDE GENERATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
