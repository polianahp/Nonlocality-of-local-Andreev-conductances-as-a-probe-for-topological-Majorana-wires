#!/usr/bin/env python3
"""
Elastic FPCA + K-Means Clustering Analysis of Conductance Curves

This script performs Elastic Functional Principal Component Analysis (Joint FPCA) 
using the fdasrsf package and K-Means clustering on barrier-sweep conductance curves.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans

import fdasrsf as fs
from fdasrsf import time_warping, fPCA

# ========================== CONFIGURATION ==========================
DATA_DIRS = ["Data/Tdis_pfaff4", "Data/disorder_realization_0_results", "Data/disorder_realization_1_results"]
PLOT_DIR = "Plots/Elastic_FPCA"

ALL_CURVE_TYPES = {
    "BR_LL": "barrier_right_conductance_left_arr",
    "BR_RR": "barrier_right_conductance_right_arr",
    "BL_LL": "barrier_left_conductance_left_arr",
    "BL_RR": "barrier_left_conductance_right_arr",
}
INCLUDE_CURVE_TYPES = ["BR_LL", "BR_RR", "BL_LL", "BL_RR"] # Just one for speed during initial testing

NORMALIZE_METHOD = "max"
SYMMETRIC_BARRIER_VALUE = 2.0

N_CLUSTERS = 6
N_COMPONENTS = 4

BARRIER_FILE = "barrier_arr"
PARAMS_FILE = "params_list"
PDI_FILE = "pdi_data"
# ===================================================================

def main():
    script_dir = Path(__file__).parent.resolve()
    plot_path = script_dir / PLOT_DIR
    plot_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STARTING ELASTIC FPCA + K-MEANS CONDUCTANCE ANALYSIS")
    print("=" * 70)

    # 1. Inspect dataset sources
    print("\n[1/6] Loading datasets...")
    subdirs = []
    for dir_str in DATA_DIRS:
        data_path = script_dir / dir_str
        print(f"Checking data_path: {data_path} exists: {data_path.exists()}")
        if not data_path.exists():
            continue
        if (data_path / f"{BARRIER_FILE}.npy").exists():
            print(f"Found {BARRIER_FILE}.npy in {data_path}")
            subdirs.append(data_path)
        else:
            found = sorted([d for d in data_path.iterdir() if d.is_dir() and (d / f"{BARRIER_FILE}.npy").exists()])
            print(f"Found subdirs: {found}")
            subdirs.extend(found)

    if not subdirs:
        raise RuntimeError("No valid dataset folders found.")

    master_barrier_arr = np.load(subdirs[0] / f"{BARRIER_FILE}.npy")
    all_curves = []

    for sub in subdirs:
        sub_barrier = np.load(sub / f"{BARRIER_FILE}.npy")
        sub_idx_sym = np.argmin(np.abs(sub_barrier - SYMMETRIC_BARRIER_VALUE))
        
        for label in INCLUDE_CURVE_TYPES:
            fname = ALL_CURVE_TYPES[label]
            arr = np.load(sub / f"{fname}.npy")
            
            if NORMALIZE_METHOD == "max":
                max_vals = arr.max(axis=1, keepdims=True)
                denom = np.where(np.abs(max_vals) > 1e-12, max_vals, 1.0)
                arr = arr / denom
            
            all_curves.append(arr)

    all_curves = np.vstack(all_curves)
    
    # Subsampling for speed during development
    max_curves = 500
    if len(all_curves) > max_curves:
        print(f"Subsampling from {len(all_curves)} to {max_curves} curves for testing.")
        idx = np.random.choice(len(all_curves), max_curves, replace=False)
        all_curves = all_curves[idx]
        
    print(f"Data shape: {all_curves.shape}")

    # 2. Prepare data for fdasrsf
    print("\n[2/6] Preparing data for Elastic Alignment...")
    # fdasrsf expects M x N (M time points, N functions)
    f = all_curves.T
    time = master_barrier_arr

    # 3. Elastic Alignment
    print("\n[3/6] Performing Elastic Curve Alignment (srsf_align)...")
    warp_obj = time_warping.fdawarp(f, time)
    # Using smoothdata=False to avoid B-spline smoothing issues if not needed
    warp_obj.srsf_align(parallel=True, cores=-1)

    # 4. Joint FPCA
    print(f"\n[4/6] Computing Joint FPCA (n_components={N_COMPONENTS})...")
    jpca = fPCA.fdajpca(warp_obj)
    jpca.calc_fpca(no=N_COMPONENTS)
    
    # 5. K-Means Clustering
    print(f"\n[5/6] Performing K-Means Clustering (K={N_CLUSTERS})...")
    scores = jpca.coef  # Shape: (N, no)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    cluster_labels = kmeans.fit_predict(scores)

    # 6. Plotting
    print("\n[6/6] Generating Plots...")
    
    # Plot A: Principal Component Functions
    # The jpca object has q_pca and f_pca. f_pca contains the principal directions in the f domain
    # shape of f_pca is (M, Nstd, no). Usually Nstd=3 (-1, 0, 1 std dev)
    fig, axes = plt.subplots(N_COMPONENTS, 1, figsize=(9, 3 * N_COMPONENTS), sharex=True)
    if N_COMPONENTS == 1:
        axes = [axes]
        
    colors = ['#FC8D62', 'k', '#66C2A5'] # -1 std, mean, +1 std
    
    for k in range(N_COMPONENTS):
        ax = axes[k]
        for l in range(jpca.stds.shape[0]):
            ax.plot(time, jpca.f_pca[:, l, k], color=colors[l], linewidth=2, 
                    label=f"std={jpca.stds[l]}" if k==0 else "")
        ax.set_title(f"Joint FPC {k+1} (Amplitude & Phase variation)")
        ax.set_ylabel(f"f(U)")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend()
            
    axes[-1].set_xlabel("Barrier U")
    fig.tight_layout()
    fig.savefig(plot_path / "elastic_fpca_components.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot B: Clusters (Mean + Transparent Individuals)
    fig, axes = plt.subplots(1, N_CLUSTERS, figsize=(4.5 * N_CLUSTERS, 4), sharey=True)
    if N_CLUSTERS == 1:
        axes = [axes]

    cmap_clusters = matplotlib.colormaps["tab10"].resampled(10)
    
    for c in range(N_CLUSTERS):
        ax = axes[c]
        mask = cluster_labels == c
        curves_in_cluster = all_curves[mask]
        n_in_cluster = len(curves_in_cluster)
        
        # Plot individual functions in very transparent lines
        for curve in curves_in_cluster:
            ax.plot(time, curve, alpha=0.1, color=cmap_clusters(c), linewidth=1.5)
            
        # Plot the mean of the cluster
        if n_in_cluster > 0:
            mean_curve = curves_in_cluster.mean(axis=0)
            ax.plot(time, mean_curve, color="black", linewidth=3, label="Mean")
            
        ax.set_title(f"Cluster {c}\n(N={n_in_cluster})")
        ax.set_xlabel("Barrier U")
        ax.grid(True, alpha=0.3)
        if n_in_cluster > 0:
            ax.legend(loc="upper right")

    axes[0].set_ylabel("Normalized Conductance")
    fig.suptitle(f"Elastic FPCA Clusters (K={N_CLUSTERS})", fontsize=16, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(plot_path / "elastic_fpca_clusters.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 70)
    print(f"ANALYSIS COMPLETE! Outputs written to: {plot_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
