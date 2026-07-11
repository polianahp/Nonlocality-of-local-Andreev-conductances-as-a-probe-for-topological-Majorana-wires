#!/usr/bin/env python3
"""
FPCA + K-Means Clustering Analysis of Conductance Curves

This script performs Functional Principal Component Analysis (FPCA) and
K-Means clustering on barrier-sweep conductance curves from Majorana
nanowire simulations, explicitly normalized by the conductance at the
symmetric barrier point.

All user-configurable parameters are defined in the CONFIGURATION block below.
"""

import os
# Ensure single-threaded NumPy/SciPy execution inside parallel GridSearchCV workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skfda.representation import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import silhouette_score


# ========================== CONFIGURATION ==========================
# Relative path from this script to the dataset folder
DATA_DIR = "Data/Tdis_pfaff4"

# Output folder where plots and CSV results will be saved
PLOT_DIR = "Plots/FPCA"

# ---------- Curve Type Include List ----------
# Registry of all possible curve types and their corresponding .npy filenames.
ALL_CURVE_TYPES = {
    "BR_LL": "barrier_right_conductance_left_arr",   # Right barrier sweep, left local conductance
    "BR_RR": "barrier_right_conductance_right_arr",  # Right barrier sweep, right local conductance
    "BL_LL": "barrier_left_conductance_left_arr",    # Left barrier sweep, left local conductance
    "BL_RR": "barrier_left_conductance_right_arr",   # Left barrier sweep, right local conductance
}

# Only curve types listed in INCLUDE_CURVE_TYPES will be loaded and analyzed.
# When left-barrier data is computed, add "BL_LL" and "BL_RR" to this list.
INCLUDE_CURVE_TYPES = ["BR_LL", "BR_RR"]

# ---------- Conductance Normalization ----------
# NORMALIZE_METHOD options:
#   "max"       : Normalize each curve G(U) by its maximum value across the barrier sweep.
#   "symmetric" : Normalize each curve G(U) by its value at the symmetric barrier point U_sym.
#   "none"      : Use raw conductance curves.
NORMALIZE_METHOD = "max"
SYMMETRIC_BARRIER_VALUE = 2.0             # Only used if NORMALIZE_METHOD == "symmetric"

# Option to save the normalized dataset to disk (.npy files)
SAVE_NORMALIZED_DATASET = True
NORMALIZED_DATA_DIR = "Data/Tdis_pfaff4_maxnorm"

# ---------- FPCA Grid Search Range ----------
# Number of Functional Principal Components to test
FPCA_COMPONENTS_RANGE = range(1, 6)       # 1, 2, 3, 4, 5

# ---------- K-Means Grid Search Range ----------
# Number of clusters to test
KMEANS_CLUSTERS_RANGE = range(2, 21)      # 2 to 20

# ---------- Train-Test Split Settings ----------
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ---------- Grid Search Output & CV ----------
TOP_N_RESULTS = 20
TOP_RANKS_TO_PLOT = 10
CV_FOLDS = 5

# ---------- Filenames for Coordinates & Topological Invariants ----------
PARAMS_FILE = "params_list"
PDI_FILE = "pdi_data"
BARRIER_FILE = "barrier_arr"
# ===================================================================


def silhouette_scorer(estimator, X):
    """
    Custom scorer for Pipeline(FPCA, KMeans).
    Transforms X through the FPCA step, predicts cluster labels via KMeans,
    and computes the silhouette score on the FPC score vectors.
    """
    fpca_step = estimator.named_steps["fpca"]
    scores = fpca_step.transform(X)
    labels = estimator.named_steps["kmeans"].predict(scores)

    # Silhouette score requires at least 2 distinct clusters
    if len(np.unique(labels)) < 2:
        return -1.0
    return silhouette_score(scores, labels)


def main():
    script_dir = Path(__file__).parent.resolve()
    data_path = script_dir / DATA_DIR
    plot_path = script_dir / PLOT_DIR
    plot_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STARTING FPCA + K-MEANS CONDUCTANCE ANALYSIS")
    print("=" * 70)

    # 1. Load coordinates, barrier grid, and topological invariant
    print(f"\n[1/7] Loading dataset from: {data_path}")
    barrier_arr = np.load(data_path / f"{BARRIER_FILE}.npy")
    params_list = np.load(data_path / f"{PARAMS_FILE}.npy")
    pdi_data = np.load(data_path / f"{PDI_FILE}.npy")

    mu_arr = params_list[:, 1]
    vz_arr = params_list[:, 2]
    pfaffian_arr = pdi_data[:, 3]  # Column 3 is 0 (trivial) or 1 (topological)

    unique_mu = np.unique(mu_arr)
    unique_vz = np.unique(vz_arr)
    Nmu = len(unique_mu)
    Nvz = len(unique_vz)
    print(f"  Parameter space grid: N_mu={Nmu}, N_vz={Nvz} (Total points={len(params_list)})")
    print(f"  Barrier sweep grid points: {len(barrier_arr)}")

    # Locate symmetric barrier index
    idx_sym = np.argmin(np.abs(barrier_arr - SYMMETRIC_BARRIER_VALUE))
    print(f"  Symmetric barrier normalization point: U_sym={barrier_arr[idx_sym]:.4f} (index {idx_sym})")

    # 2. Load and normalize included curve types
    curve_data = {}
    for label in INCLUDE_CURVE_TYPES:
        if label not in ALL_CURVE_TYPES:
            raise ValueError(
                f"Unknown curve type '{label}'. Valid types: {list(ALL_CURVE_TYPES.keys())}"
            )
        fname = ALL_CURVE_TYPES[label]
        arr = np.load(data_path / f"{fname}.npy")
        print(f"  Loaded {label} ({fname}.npy): shape={arr.shape}, raw range=[{arr.min():.4e}, {arr.max():.4e}]")

        if NORMALIZE_METHOD == "max":
            max_vals = arr.max(axis=1, keepdims=True)
            denom = np.where(np.abs(max_vals) > 1e-12, max_vals, 1.0)
            arr = arr / denom
            print(f"    -> Normalized {label} by maximum curve value: range=[{arr.min():.4e}, {arr.max():.4e}]")
        elif NORMALIZE_METHOD == "symmetric":
            sym_vals = arr[:, idx_sym : idx_sym + 1]
            denom = np.where(np.abs(sym_vals) > 1e-12, sym_vals, 1.0)
            arr = arr / denom
            print(f"    -> Normalized {label} by value at symmetric barrier (U_sym={barrier_arr[idx_sym]:.2f}): range=[{arr.min():.4e}, {arr.max():.4e}]")
        elif NORMALIZE_METHOD != "none":
            raise ValueError(f"Unknown NORMALIZE_METHOD: '{NORMALIZE_METHOD}'")

        curve_data[label] = arr

    if SAVE_NORMALIZED_DATASET:
        norm_dir = script_dir / NORMALIZED_DATA_DIR
        norm_dir.mkdir(parents=True, exist_ok=True)
        np.save(norm_dir / f"{BARRIER_FILE}.npy", barrier_arr)
        np.save(norm_dir / f"{PARAMS_FILE}.npy", params_list)
        np.save(norm_dir / f"{PDI_FILE}.npy", pdi_data)
        for label, arr in curve_data.items():
            fname = ALL_CURVE_TYPES[label]
            np.save(norm_dir / f"{fname}.npy", arr)
        print(f"  Saved normalized dataset to: {norm_dir}")

    if not curve_data:
        raise RuntimeError("No curve types included in INCLUDE_CURVE_TYPES.")

    # 3. Aggregate curves into a single functional dataset
    print("\n[2/7] Aggregating functional data...")
    all_curves = []
    all_labels = []
    all_param_idx = []

    for label, arr in curve_data.items():
        n_params, _ = arr.shape
        all_curves.append(arr)
        all_labels.extend([label] * n_params)
        all_param_idx.extend(range(n_params))

    all_curves = np.vstack(all_curves)
    all_labels = np.array(all_labels)
    all_param_idx = np.array(all_param_idx)

    print(f"  Aggregated dataset shape: {all_curves.shape}")

    # Convert to FDataGrid object
    fd = FDataGrid(data_matrix=all_curves, grid_points=barrier_arr)

    # 4. Train-Test split
    print(f"\n[3/7] Creating {int((1-TEST_SIZE)*100)}-{int(TEST_SIZE*100)} train-test split...")
    indices = np.arange(len(all_curves))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    fd_train = fd[train_idx]
    print(f"  Training set size: {len(fd_train)} curves")
    print(f"  Test set size: {len(fd[test_idx])} curves")

    # 5. Build pipeline and run GridSearchCV
    print("\n[4/7] Performing GridSearchCV (FPCA + K-Means)...")
    pipe = Pipeline([
        ("fpca", FPCA(n_components=2)),
        ("kmeans", KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "fpca__n_components": list(FPCA_COMPONENTS_RANGE),
        "kmeans__n_clusters": list(KMEANS_CLUSTERS_RANGE),
    }

    gscv = GridSearchCV(
        pipe,
        param_grid,
        scoring=silhouette_scorer,
        cv=CV_FOLDS,
        refit=True,
        n_jobs=-1,
        verbose=1,
    )

    gscv.fit(fd_train)

    # 6. Report and save Top 20 results
    print("\n[5/7] Processing Grid Search results...")
    results_df = pd.DataFrame(gscv.cv_results_)
    results_df = results_df.sort_values("mean_test_score", ascending=False)

    top_20 = results_df.head(TOP_N_RESULTS)[
        [
            "param_fpca__n_components",
            "param_kmeans__n_clusters",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
    ].copy()
    top_20.columns = [
        "FPCA_Components",
        "KMeans_Clusters",
        "Mean_Silhouette",
        "Std_Silhouette",
        "Rank",
    ]

    print("\n" + "=" * 70)
    print("TOP 20 GRID SEARCH RESULTS (by Silhouette Score)")
    print("=" * 70)
    print(top_20.to_string(index=False))

    results_df.to_csv(plot_path / "grid_search_results.csv", index=False)
    top_20.to_csv(plot_path / "top_20_results.csv", index=False)
    print(f"\nSaved grid search CSVs to: {plot_path}")

    # 7. Generate Rank Subfolders for Top 10 FPCA + K-Means Combinations
    print(f"\n[6/7] Cleaning root plot folder and preparing top {TOP_RANKS_TO_PLOT} rank subfolders...")
    for old_png in plot_path.glob("*.png"):
        old_png.unlink()

    top_ranks = top_20.head(TOP_RANKS_TO_PLOT)
    print(f"\n[7/7] Generating visualizations for top {len(top_ranks)} ranked combinations...")

    for idx, row in top_ranks.iterrows():
        rank = int(row["Rank"])
        n_comp = int(row["FPCA_Components"])
        n_clust = int(row["KMeans_Clusters"])
        score = float(row["Mean_Silhouette"])

        rank_dir = plot_path / f"Rank_{rank:02d}_FPC{n_comp}_K{n_clust}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        print(f"  -> Rank {rank:02d}: FPC={n_comp}, K={n_clust} (Silhouette={score:.4f}) -> {rank_dir.name}")

        fpca = FPCA(n_components=n_comp)
        fpca.fit(fd)
        scores_all = fpca.transform(fd)

        kmeans = KMeans(n_clusters=n_clust, n_init=10, random_state=RANDOM_STATE)
        kmeans.fit(scores_all)

        generate_plots_for_model(
            fpca_fitted=fpca,
            kmeans_fitted=kmeans,
            fd=fd,
            all_curves=all_curves,
            all_labels=all_labels,
            barrier_arr=barrier_arr,
            pfaffian_arr=pfaffian_arr,
            Nmu=Nmu,
            Nvz=Nvz,
            unique_mu=unique_mu,
            unique_vz=unique_vz,
            out_dir=rank_dir,
            n_components=n_comp,
            n_clusters=n_clust,
            normalize_method=NORMALIZE_METHOD,
            curve_labels=list(curve_data.keys()),
        )

    print("\n" + "=" * 70)
    print(f"ANALYSIS COMPLETE! All outputs written to: {plot_path}")
    print("=" * 70)


def generate_plots_for_model(
    fpca_fitted,
    kmeans_fitted,
    fd,
    all_curves,
    all_labels,
    barrier_arr,
    pfaffian_arr,
    Nmu,
    Nvz,
    unique_mu,
    unique_vz,
    out_dir,
    n_components,
    n_clusters,
    normalize_method,
    curve_labels,
):
    scores_all = fpca_fitted.transform(fd)
    cluster_labels_all = kmeans_fitted.predict(scores_all)

    # --- Plot A: Individual Functional Principal Components ---
    fig, axes = plt.subplots(
        n_components,
        1,
        figsize=(9, 2.5 * n_components),
        sharex=True,
    )
    if n_components == 1:
        axes = [axes]

    components = fpca_fitted.components_
    for k in range(n_components):
        ax = axes[k]
        comp_data = components.data_matrix[k, :, 0]
        ax.plot(barrier_arr, comp_data, linewidth=2, color="#1f77b4")
        var_pct = fpca_fitted.explained_variance_ratio_[k] * 100
        ax.set_title(f"FPC {k+1} (Explains {var_pct:.1f}% Variance)")
        ax.set_ylabel(f"$\\xi_{{{k+1}}}(U)$")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Barrier Potential $U$")
    fig.suptitle(
        f"Functional Principal Components (K={n_components})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fpca_components.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Plot B: Explained Variance Ratio Bar Chart ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fpc_indices = np.arange(1, n_components + 1)
    var_ratios = fpca_fitted.explained_variance_ratio_ * 100
    ax.bar(fpc_indices, var_ratios, color="#2ca02c", alpha=0.8, edgecolor="black")
    ax.set_xlabel("FPC Index")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("FPCA Explained Variance Ratio")
    ax.set_xticks(fpc_indices)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "explained_variance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Plot C: Conductance Curves Grouped by Cluster ---
    fig, axes = plt.subplots(
        1,
        n_clusters,
        figsize=(4.5 * n_clusters, 4),
        sharey=True,
    )
    if n_clusters == 1:
        axes = [axes]

    cmap_clusters = matplotlib.colormaps["tab20"].resampled(n_clusters)

    for c in range(n_clusters):
        ax = axes[c]
        mask = cluster_labels_all == c
        curves_in_cluster = all_curves[mask]
        n_in_cluster = len(curves_in_cluster)

        max_plot = 400
        if n_in_cluster > max_plot:
            plot_idx = np.random.choice(n_in_cluster, max_plot, replace=False)
            curves_to_plot = curves_in_cluster[plot_idx]
        else:
            curves_to_plot = curves_in_cluster

        for curve in curves_to_plot:
            ax.plot(
                barrier_arr,
                curve,
                alpha=0.06,
                color=cmap_clusters(c),
                linewidth=0.6,
            )

        if n_in_cluster > 0:
            mean_curve = curves_in_cluster.mean(axis=0)
            ax.plot(
                barrier_arr,
                mean_curve,
                color="black",
                linewidth=2.2,
                label="Mean",
            )

        ax.set_title(f"Cluster {c}\n(N={n_in_cluster})")
        ax.set_xlabel("Barrier Potential $U$")
        ax.grid(True, alpha=0.3)
        if n_in_cluster > 0:
            ax.legend(loc="upper right")

    ylabel_str = "Normalized Conductance $G / G_{max}$" if normalize_method == "max" else "Normalized Conductance $G / G_{sym}$"
    axes[0].set_ylabel(ylabel_str)
    fig.suptitle(
        f"Conductance Curves by Cluster (FPC={n_components}, K={n_clusters})",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "curves_by_cluster.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Plot D: Topological Phase Maps per Included Curve Type ---
    pfaff_grid = pfaffian_arr.reshape(Nmu, Nvz)

    for label in curve_labels:
        type_mask = all_labels == label
        type_cluster_labels = cluster_labels_all[type_mask]
        cluster_grid = type_cluster_labels.reshape(Nmu, Nvz)

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.pcolormesh(
            unique_vz,
            unique_mu,
            cluster_grid,
            cmap=cmap_clusters,
            shading="nearest",
            vmin=-0.5,
            vmax=n_clusters - 0.5,
        )

        ax.contour(
            unique_vz,
            unique_mu,
            pfaff_grid,
            levels=[0.5],
            colors=["gray"],
            linewidths=2.2,
            alpha=0.6,
        )
        ax.contourf(
            unique_vz,
            unique_mu,
            pfaff_grid,
            levels=[0.5, 1.5],
            colors=["gray"],
            alpha=0.25,
        )

        ax.set_xlabel("$V_Z$", fontsize=13)
        ax.set_ylabel("$\\mu$", fontsize=13)
        ax.set_title(
            f"Phase Map: {label} Conductance Curves\n(FPC={n_components}, Clusters={n_clusters} | Gray overlay = Topological Phase)",
            fontsize=13,
        )

        cbar = fig.colorbar(im, ax=ax, ticks=range(n_clusters))
        cbar.set_label("K-Means Cluster Label")

        fig.tight_layout()
        fig.savefig(out_dir / f"phase_map_{label}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
