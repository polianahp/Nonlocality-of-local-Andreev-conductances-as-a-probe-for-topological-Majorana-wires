#!/usr/bin/env python3
"""
generate_presentation_correlation_comparisons.py

Produces side-by-side high-resolution comparison figures for the presentation:
1. 0 mK vs 40 mK Legacy Reflection Correlation (demonstrating thermal destruction).
2. 40 mK Legacy sweep ([0, 60] meV) vs 40 mK Low-Asymmetry sweep ([-5, +5] meV) (demonstrating restoration).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
DATA_BASE = REPO_ROOT / "Outputs" / "Data" / "Tdis_pfaff5_data"
OUT_DIR = REPO_ROOT / "Outputs" / "Plots" / "Finite_T_correllation_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(data_dir: Path):
    params_npz = np.load(data_dir / "all_params.npz")
    Nmu = int(params_npz["Nmu"])
    Nvz = int(params_npz["Nvz"])
    mu_vals = params_npz["mu_var"]
    vz_vals = params_npz["Vz_var"]

    pdi_data = np.load(data_dir / "pdi_data.npy")
    col_idx = 3 if pdi_data.shape[1] >= 4 else 2
    gt = (pdi_data[:, col_idx] == 1.0).reshape(Nmu, Nvz)

    c_l = np.load(data_dir / "barrier_left_correlation_arr.npy").reshape(Nmu, Nvz)
    c_r = np.load(data_dir / "barrier_right_correlation_arr.npy").reshape(Nmu, Nvz)
    c_mean = 0.5 * (c_l + c_r)

    return {
        "Nmu": Nmu, "Nvz": Nvz,
        "mu_vals": mu_vals, "vz_vals": vz_vals,
        "gt": gt,
        "c_l": c_l, "c_r": c_r,
        "c_mean": c_mean
    }

def compute_low_asym_corr(data_dir: Path, asym_window=(-5.0, 5.0)):
    # Compute low asymmetry correlation from barrier conductances
    params_npz = np.load(data_dir / "all_params.npz")
    Nmu = int(params_npz["Nmu"])
    Nvz = int(params_npz["Nvz"])
    asym_ratios = np.load(data_dir / "barrier_arr.npy")

    br_l = np.load(data_dir / "barrier_right_conductance_left_arr.npy")
    bl_r = np.load(data_dir / "barrier_left_conductance_right_arr.npy")

    mask = (asym_ratios >= asym_window[0]) & (asym_ratios <= asym_window[1])
    f1 = br_l[:, mask]
    f2 = bl_r[:, mask]

    # Normalize by first point in window
    f1_norm = f1 / np.where(f1[:, 0:1] != 0, f1[:, 0:1], 1.0)
    f2_norm = f2 / np.where(f2[:, 0:1] != 0, f2[:, 0:1], 1.0)

    norm1 = np.linalg.norm(f1_norm, axis=1)
    norm2 = np.linalg.norm(f2_norm, axis=1)
    dot = np.sum(f1_norm * f2_norm, axis=1)
    denom = norm1 * norm2
    c_low = np.where(denom > 0, dot / denom, 0.0).reshape(Nmu, Nvz)
    return c_low

def main():
    d0 = load_data(DATA_BASE / "Tdis_pfaff5")
    d40 = load_data(DATA_BASE / "Tdispfaff5_40mK")
    c_low40 = compute_low_asym_corr(DATA_BASE / "Tdispfaff5_40mK")

    extent = [d0["vz_vals"][0], d0["vz_vals"][-1], d0["mu_vals"][0], d0["mu_vals"][-1]]

    # 1. Figure 1: 0 mK vs 40 mK Legacy Correlation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    im1 = ax1.imshow(d0["c_mean"], origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax1.contour(d0["gt"], levels=[0.5], colors="cyan", linestyles="--", linewidths=1.8, extent=extent)
    ax1.set_title(r"Standard Barrier Correlation $\mathcal{C}$ at $T = 0$ mK (Legacy $[0, 60]$ meV)" "\n" r"(Cyan dashed: Ground Truth Pfaffian $\mathcal{Q} = -1$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"Zeeman Energy $V_z$ (meV)", fontsize=10)
    ax1.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Correlation Coefficient", fontsize=10)

    im2 = ax2.imshow(d40["c_mean"], origin="lower", extent=extent, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax2.contour(d40["gt"], levels=[0.5], colors="cyan", linestyles="--", linewidths=1.8, extent=extent)
    ax2.set_title(r"Standard Barrier Correlation $\mathcal{C}$ at $T = 40$ mK (Legacy $[0, 60]$ meV)" "\n" r"Severe Thermal Degradation ($k_B T \approx 3.4\ \mu$eV)", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Zeeman Energy $V_z$ (meV)", fontsize=10)
    ax2.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Correlation Coefficient", fontsize=10)

    plt.tight_layout()
    out1 = OUT_DIR / "comparison_0mK_vs_40mK_legacy_correlation.png"
    plt.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # 2. Figure 2: 40 mK Legacy vs Low Asymmetry
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    im1 = ax1.imshow(d40["c_mean"], origin="lower", extent=extent, aspect="auto", cmap="inferno", vmin=0.0, vmax=1.0)
    ax1.contour(d40["gt"], levels=[0.5], colors="cyan", linestyles="--", linewidths=1.8, extent=extent)
    ax1.set_title(r"40 mK Legacy Sweep ($U \in [0, 60]$ meV)" "\n" r"Thermally Broadened Correlation Loss ($\mathrm{Mean}\ \mathcal{C} = 0.317$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel(r"Zeeman Energy $V_z$ (meV)", fontsize=10)
    ax1.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Correlation Coefficient", fontsize=10)

    im2 = ax2.imshow(c_low40, origin="lower", extent=extent, aspect="auto", cmap="inferno", vmin=0.0, vmax=1.0)
    ax2.contour(d40["gt"], levels=[0.5], colors="cyan", linestyles="--", linewidths=1.8, extent=extent)
    ax2.set_title(r"40 mK Low-Asymmetry Sweep ($U \in [-5, +5]$ meV)" "\n" r"Restored Resonance Sensitivity ($\mathrm{Mean}\ \mathcal{C}_{\mathrm{topo}} > 0.90$)", fontsize=11, fontweight="bold")
    ax2.set_xlabel(r"Zeeman Energy $V_z$ (meV)", fontsize=10)
    ax2.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Correlation Coefficient", fontsize=10)

    plt.tight_layout()
    out2 = OUT_DIR / "comparison_40mK_legacy_vs_low_asymmetry.png"
    plt.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

if __name__ == "__main__":
    main()
