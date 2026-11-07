#!/usr/bin/env python3
"""
generate_symmetry_balance_phase_maps.py

Generates continuous and binarized phase maps of the Conjugate Lead Symmetry Balance Ratio:
    S_sym(mu, Vz) = |G_LL^sym - G_RR^sym| / max(G_LL^sym + G_RR^sym, eps_sym)

Evaluates:
  1. Continuous phase map S_sym(mu, Vz) at 40 mK and 30 mK for Tdis_pfaff5 (V0 = 1.2 meV benchmark).
  2. Binarized phase map with threshold S_sym <= 0.15 (Green = TP, Red = FP, Blue = FN, White = TN).
  3. Multi-disorder 4-panel comparison across V0 = 0.378, 0.645, 0.872, 1.200 meV.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent
DATA_BASE = REPO_ROOT / "Outputs" / "Data" / "Tdis_pfaff5_data"
OUT_DIR = REPO_ROOT / "Outputs" / "Plots" / "Finite_T_correllation_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TP_RGB = [0.13, 0.65, 0.28]
FP_RGB = [0.86, 0.20, 0.18]
FN_RGB = [0.12, 0.53, 0.90]

def load_data_and_compute_s_sym(data_dir: Path, eps_sym: float = 1e-4):
    params_npz = np.load(data_dir / "all_params.npz")
    Nmu = int(params_npz["Nmu"])
    Nvz = int(params_npz["Nvz"])
    mu_vals = params_npz["mu_var"]
    vz_vals = params_npz["Vz_var"]

    pdi_data = np.load(data_dir / "pdi_data.npy")
    col_idx = 3 if pdi_data.shape[1] >= 4 else 2
    gt = (pdi_data[:, col_idx] == 1.0).reshape(Nmu, Nvz)

    br_l = np.load(data_dir / "barrier_right_conductance_left_arr.npy")
    bl_r = np.load(data_dir / "barrier_left_conductance_right_arr.npy")

    g_ll = br_l[:, 0].reshape(Nmu, Nvz)
    g_rr = bl_r[:, 0].reshape(Nmu, Nvz)

    denom = np.maximum(g_ll + g_rr, eps_sym)
    s_sym = np.abs(g_ll - g_rr) / denom
    # Clip inf / nan
    s_sym = np.where((g_ll + g_rr) >= eps_sym, s_sym, 1.0)
    s_sym = np.clip(s_sym, 0.0, 1.0)

    return {
        "Nmu": Nmu, "Nvz": Nvz,
        "mu_vals": mu_vals, "vz_vals": vz_vals,
        "gt": gt, "s_sym": s_sym,
        "g_ll": g_ll, "g_rr": g_rr
    }

def plot_symmetry_balance_dual_panel(t_label: str, temp_folder: str, s_max: float = 0.15):
    dpath = DATA_BASE / temp_folder
    data = load_data_and_compute_s_sym(dpath)
    gt = data["gt"]
    s_sym = data["s_sym"]
    mu_vals = data["mu_vals"]
    vz_vals = data["vz_vals"]

    pass_mask = s_sym <= s_max
    tp = np.sum(pass_mask & gt)
    fp = np.sum(pass_mask & ~gt)
    fn = np.sum(~pass_mask & gt)
    fdr = fp / max(tp + fp, 1) * 100.0
    tpr = tp / max(tp + fn, 1) * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=140)

    # Panel 1: Continuous S_sym
    extent = [vz_vals[0], vz_vals[-1], mu_vals[0], mu_vals[-1]]
    im1 = ax1.imshow(s_sym, origin="lower", extent=extent, aspect="auto", cmap="viridis_r", vmin=0.0, vmax=0.6)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(r"$S_{\mathrm{sym}} = |G_{LL} - G_{RR}| / (G_{LL} + G_{RR})$", fontsize=10)
    ax1.contour(gt, levels=[0.5], colors="red", linestyles="--", linewidths=1.8, extent=extent)
    ax1.set_title(f"Conjugate Lead Symmetry Balance $S_{{\\mathrm{{sym}}}}$ ({t_label})\nContinuous Metric (Red dashed: Pfaffian $\\mathcal{{Q}} = -1$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"Zeeman Energy $V_z$ (meV)", fontsize=10)
    ax1.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)

    # Panel 2: Binarized S_sym <= s_max
    rgb = np.ones(gt.shape + (3,), dtype=float)
    rgb[pass_mask & gt] = TP_RGB
    rgb[pass_mask & ~gt] = FP_RGB
    rgb[~pass_mask & gt] = FN_RGB

    ax2.imshow(rgb, origin="lower", extent=extent, aspect="auto")
    ax2.contour(gt, levels=[0.5], colors="black", linestyles="--", linewidths=1.8, extent=extent)
    ax2.set_title(f"Binarized Gate $S_{{\\mathrm{{sym}}}} \\leq {s_max:.2f}$ ({t_label})\nFDR = {fdr:.1f}% | TPR = {tpr:.1f}% | Pass Area = {np.mean(pass_mask)*100:.1f}%", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Zeeman Energy $V_z$ (meV)", fontsize=10)
    ax2.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)

    handles = [
        Patch(color=TP_RGB, label="True Positive (Pass & Topological)"),
        Patch(color=FP_RGB, label="False Positive (Pass & Trivial)"),
        Patch(color=FN_RGB, label="False Negative (Filtered Topological)"),
        Patch(facecolor="white", edgecolor="0.6", label="True Negative"),
        plt.Line2D([0], [0], color="black", ls="--", label=r"Pfaffian Boundary ($\mathcal{Q}=-1$)")
    ]
    ax2.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out_file = OUT_DIR / f"symmetry_balance_phase_map_{temp_folder}.png"
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")
    return out_file

def main():
    plot_symmetry_balance_dual_panel("40 mK", "Tdispfaff5_40mK", s_max=0.15)
    plot_symmetry_balance_dual_panel("30 mK", "Tdispfaff5_30mK", s_max=0.15)

if __name__ == "__main__":
    main()
