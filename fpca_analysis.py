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
# Relative paths from this script to the dataset folders to include
DATA_DIRS = [
    "Data/dis_realizations",
    "Data/Tdis_pfaff4",
]
DATA_DIR = DATA_DIRS[0]  # Fallback reference

# Output folder where plots and CSV results will be saved
PLOT_DIR = "Plots/FPCA_dis_realizations_maxnorm"

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
INCLUDE_CURVE_TYPES = ["BR_LL", "BR_RR", "BL_LL", "BL_RR"]

# ---------- Conductance Normalization ----------
# NORMALIZE_METHOD options:
#   "max"       : Normalize each curve G(U) by its maximum value across the barrier sweep.
#   "symmetric" : Normalize each curve G(U) by its value at the symmetric barrier point U_sym.
#   "none"      : Use raw conductance curves.
NORMALIZE_METHOD = "max"
SYMMETRIC_BARRIER_VALUE = 2.0             # Only used if NORMALIZE_METHOD == "symmetric"

# ---------- Functional Representation (FDataGrid vs FDataBasis) ----------
USE_BSPLINE_BASIS = True
# Custom interior knots with higher density in [-25, 5] (16 total basis functions)
CUSTOM_BSPLINE_KNOTS = [-100.0, -70.0, -45.0, -25.0, -18.0, -11.0, -4.0, 3.0, 15.0, 40.0, 75.0, 110.0]

# Option to save the normalized dataset to disk (.npy files)
SAVE_NORMALIZED_DATASET = True
NORMALIZED_DATA_DIR = f"Data/dis_realizations_{NORMALIZE_METHOD}norm"

# Option to skip grid search if optimum results are already saved in top_20_results.csv
SKIP_GRID_SEARCH = True

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
    return silhouette_score(scores, labels, sample_size=10000, random_state=RANDOM_STATE)


def main():
    script_dir = Path(__file__).parent.resolve()
    plot_path = script_dir / PLOT_DIR
    plot_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STARTING FPCA + K-MEANS CONDUCTANCE ANALYSIS")
    print("=" * 70)

    # 1. Inspect dataset sources across all folders in DATA_DIRS
    print("\n[1/7] Inspecting dataset sources...")
    subdirs = []
    for dir_str in DATA_DIRS:
        data_path = script_dir / dir_str
        if not data_path.exists():
            print(f"  Warning: Dataset path does not exist: {data_path}")
            continue
        if (data_path / f"{BARRIER_FILE}.npy").exists():
            subdirs.append(data_path)
            print(f"  Detected single dataset directory: {data_path.name}")
        else:
            found = sorted([d for d in data_path.iterdir() if d.is_dir() and (d / f"{BARRIER_FILE}.npy").exists()])
            if not found:
                print(f"  Warning: No valid dataset folders found in: {data_path}")
            else:
                subdirs.extend(found)
                print(f"  Detected ensemble directory '{data_path.name}' with {len(found)} realization subfolders")

    if not subdirs:
        raise RuntimeError(f"No valid dataset folders containing {BARRIER_FILE}.npy found across: {DATA_DIRS}")
    print(f"  Total realization subfolders to process: {len(subdirs)}")

    # Establish reference barrier_arr across all subfolders
    master_barrier_arr = np.load(subdirs[0] / f"{BARRIER_FILE}.npy")

    all_curves = []
    all_labels = []
    all_realizations = []
    all_param_idx = []
    all_pfaffians = []

    barrier_arr = master_barrier_arr
    mu_arr = None
    vz_arr = None
    unique_mu = None
    unique_vz = None
    Nmu = None
    Nvz = None

    for sub_idx, sub in enumerate(subdirs):
        print(f"\n  [{sub_idx+1}/{len(subdirs)}] Loading from {sub.name}...")
        sub_barrier = np.load(sub / f"{BARRIER_FILE}.npy")
        sub_params = np.load(sub / f"{PARAMS_FILE}.npy")
        sub_pdi = np.load(sub / f"{PDI_FILE}.npy")

        if mu_arr is None:
            mu_arr = sub_params[:, 1]
            vz_arr = sub_params[:, 2]
            unique_mu = np.unique(mu_arr)
            unique_vz = np.unique(vz_arr)
            Nmu = len(unique_mu)
            Nvz = len(unique_vz)
            print(f"    Master parameter space grid: N_mu={Nmu}, N_vz={Nvz} (Total points={len(sub_params)})")
            print(f"    Master barrier sweep grid points: {len(barrier_arr)}")
        else:
            if not np.allclose(sub_params[:, 1:3], np.column_stack((mu_arr, vz_arr))):
                raise ValueError(f"Inconsistent (mu, V_z) parameter grid in folder {sub.name}")

        sub_idx_sym = np.argmin(np.abs(sub_barrier - SYMMETRIC_BARRIER_VALUE))
        print(f"    Realization U_sym index={sub_idx_sym} (U={sub_barrier[sub_idx_sym]:.4f}, grid len={len(sub_barrier)})")
        sub_pfaffian = sub_pdi[:, 3]

        # Load and normalize included curve types for this realization
        sub_curve_data = {}
        for label in INCLUDE_CURVE_TYPES:
            if label not in ALL_CURVE_TYPES:
                raise ValueError(f"Unknown curve type '{label}'. Valid types: {list(ALL_CURVE_TYPES.keys())}")
            fname = ALL_CURVE_TYPES[label]
            arr = np.load(sub / f"{fname}.npy")

            if NORMALIZE_METHOD == "max":
                max_vals = arr.max(axis=1, keepdims=True)
                denom = np.where(np.abs(max_vals) > 1e-12, max_vals, 1.0)
                arr = arr / denom
            elif NORMALIZE_METHOD == "symmetric":
                sym_vals = arr[:, sub_idx_sym : sub_idx_sym + 1]
                denom = np.where(np.abs(sym_vals) > 1e-8, sym_vals, np.where(sym_vals < 0, -1e-8, 1e-8))
                denom = np.where(denom == 0, 1e-8, denom)
                arr = arr / denom
            elif NORMALIZE_METHOD != "none":
                raise ValueError(f"Unknown NORMALIZE_METHOD: '{NORMALIZE_METHOD}'")

            # Verify sweep resolutions match exactly
            if len(sub_barrier) != len(barrier_arr) or not np.allclose(sub_barrier, barrier_arr):
                raise ValueError(
                    f"Resolution mismatch in realization '{sub.name}': "
                    f"sweep grid length is {len(sub_barrier)}, but expected {len(barrier_arr)} to match master grid."
                )

            sub_curve_data[label] = arr
            n_params, _ = arr.shape
            all_curves.append(arr)
            all_labels.extend([label] * n_params)
            all_realizations.extend([sub.name] * n_params)
            all_param_idx.extend(range(n_params))
            all_pfaffians.extend(sub_pfaffian)

        if SAVE_NORMALIZED_DATASET:
            norm_dir = script_dir / NORMALIZED_DATA_DIR / sub.name
            norm_dir.mkdir(parents=True, exist_ok=True)
            np.save(norm_dir / f"{BARRIER_FILE}.npy", barrier_arr)
            np.save(norm_dir / f"{PARAMS_FILE}.npy", sub_params)
            np.save(norm_dir / f"{PDI_FILE}.npy", sub_pdi)
            for label, arr in sub_curve_data.items():
                fname = ALL_CURVE_TYPES[label]
                np.save(norm_dir / f"{fname}.npy", arr)
            print(f"    Saved normalized realization data to: {norm_dir}")

    all_curves = np.vstack(all_curves)
    all_labels = np.array(all_labels)
    all_realizations = np.array(all_realizations)
    all_param_idx = np.array(all_param_idx)
    all_pfaffians = np.array(all_pfaffians)
    unique_realizations = [s.name for s in subdirs]

    print(f"\n[2/7] Aggregated functional data shape across {len(unique_realizations)} realizations: {all_curves.shape}")

    # Convert to FDataGrid object
    fd = FDataGrid(data_matrix=all_curves, grid_points=barrier_arr)

    if USE_BSPLINE_BASIS:
        from skfda.representation.basis import BSplineBasis
        full_knots = [float(barrier_arr[0])] + [k for k in CUSTOM_BSPLINE_KNOTS if barrier_arr[0] < k < barrier_arr[-1]] + [float(barrier_arr[-1])]
        basis = BSplineBasis(domain_range=(float(barrier_arr[0]), float(barrier_arr[-1])), knots=full_knots, order=4)
        print(f"\n[2b/7] Projecting {len(all_curves)} curves onto B-spline basis ({basis.n_basis} basis functions)...")
        fd = fd.to_basis(basis)

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
    print("\n[4/7] Performing GridSearchCV (FPCA + K-Means) with Pipeline Caching...")
    from joblib import Memory
    cache_dir = Path("scratch/fpca_pipeline_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    memory = Memory(location=cache_dir, verbose=0)

    pipe = Pipeline([
        ("fpca", FPCA(n_components=2)),
        ("kmeans", KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)),
    ], memory=memory)

    param_grid = {
        "fpca__n_components": list(FPCA_COMPONENTS_RANGE),
        "kmeans__n_clusters": list(KMEANS_CLUSTERS_RANGE),
    }

    top20_csv = plot_path / "top_20_results.csv"
    if SKIP_GRID_SEARCH and top20_csv.exists():
        print("\n[4/7 & 5/7] Skipping GridSearchCV — loading existing top_20_results.csv...")
        top_20 = pd.read_csv(top20_csv)
        print("\n" + "=" * 70)
        print("TOP 20 GRID SEARCH RESULTS (Loaded from CSV)")
        print("=" * 70)
        print(top_20.to_string(index=False))
    else:
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
        top_20.to_csv(top20_csv, index=False)
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
            all_realizations=all_realizations,
            all_param_idx=all_param_idx,
            all_pfaffians=all_pfaffians,
            barrier_arr=barrier_arr,
            unique_realizations=unique_realizations,
            Nmu=Nmu,
            Nvz=Nvz,
            unique_mu=unique_mu,
            unique_vz=unique_vz,
            out_dir=rank_dir,
            n_components=n_comp,
            n_clusters=n_clust,
            normalize_method=NORMALIZE_METHOD,
            curve_labels=INCLUDE_CURVE_TYPES,
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
    all_realizations,
    all_param_idx,
    all_pfaffians,
    barrier_arr,
    unique_realizations,
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

    # Calculate barrier ratio U_swept / U_const dynamically at symmetric anchor point
    idx_sym = np.argmin(np.abs(barrier_arr - SYMMETRIC_BARRIER_VALUE))
    u_const = barrier_arr[idx_sym] if np.abs(barrier_arr[idx_sym]) > 1e-12 else 1.0
    barrier_ratio_arr = barrier_arr / u_const
    x_label_str = "Barrier Ratio $U_{\\mathrm{swept}} / U_{\\mathrm{const}}$ ($U_L/U_R$ or $U_R/U_L$)"

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
    if hasattr(components, "data_matrix") and components.data_matrix is not None:
        comp_eval = components.data_matrix[:, :, 0]
    else:
        comp_eval = components(barrier_arr)[:, :, 0]

    for k in range(n_components):
        ax = axes[k]
        comp_data = comp_eval[k, :]
        ax.plot(barrier_ratio_arr, comp_data, linewidth=2, color="#1f77b4")
        var_pct = fpca_fitted.explained_variance_ratio_[k] * 100
        ax.set_title(f"FPC {k+1} (Explains {var_pct:.1f}% Variance)")
        ax.set_ylabel(f"$\\xi_{{{k+1}}}(U)$")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label_str)
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
                barrier_ratio_arr,
                curve,
                alpha=0.06,
                color=cmap_clusters(c),
                linewidth=0.6,
            )

        if n_in_cluster > 0:
            mean_curve = curves_in_cluster.mean(axis=0)
            ax.plot(
                barrier_ratio_arr,
                mean_curve,
                color="black",
                linewidth=2.2,
                label="Mean",
            )

        ax.set_title(f"Cluster {c}\n(N={n_in_cluster})")
        ax.set_xlabel(x_label_str)
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
    M = len(unique_realizations)

    if M == 1:
        # Single realization case: direct grid mapping
        pfaff_grid = all_pfaffians[:Nmu * Nvz].reshape(Nmu, Nvz)
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
                colors=["black"],
                linewidths=3.0,
                alpha=0.9,
            )
            ax.contourf(
                unique_vz,
                unique_mu,
                pfaff_grid,
                levels=[0.5, 1.5],
                colors=["black"],
                alpha=0.45,
            )

            ax.set_xlabel("$V_Z$", fontsize=13)
            ax.set_ylabel("$\\mu$", fontsize=13)
            ax.set_title(
                f"Phase Map: {label} Conductance Curves\n(FPC={n_components}, Clusters={n_clusters} | Dark overlay = Topological Phase)",
                fontsize=13,
            )

            cbar = fig.colorbar(im, ax=ax, ticks=range(n_clusters))
            cbar.set_label("K-Means Cluster Label")

            fig.tight_layout()
            fig.savefig(out_dir / f"phase_map_{label}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
    else:
        # Ensemble multi-realization case:
        # 1. Ensemble majority phase map and average topological probability overlay
        for label in curve_labels:
            label_mask = all_labels == label
            clusters_for_label = cluster_labels_all[label_mask].reshape(M, Nmu, Nvz)
            pfaffians_for_label = all_pfaffians[label_mask].reshape(M, Nmu, Nvz)

            # Majority cluster assignment across realizations for each (mu, V_z) point
            majority_cluster_grid = np.apply_along_axis(
                lambda x: np.bincount(x, minlength=n_clusters).argmax(), axis=0, arr=clusters_for_label
            )
            mean_pfaff_grid = pfaffians_for_label.mean(axis=0)

            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.pcolormesh(
                unique_vz,
                unique_mu,
                majority_cluster_grid,
                cmap=cmap_clusters,
                shading="nearest",
                vmin=-0.5,
                vmax=n_clusters - 0.5,
            )

            # Contour for ensemble average topological phase boundary (>= 75% of realizations topological)
            ax.contour(
                unique_vz,
                unique_mu,
                mean_pfaff_grid,
                levels=[0.75],
                colors=["black"],
                linewidths=3.0,
                alpha=0.9,
            )
            ax.contourf(
                unique_vz,
                unique_mu,
                mean_pfaff_grid,
                levels=[0.75, 1.0],
                colors=["black"],
                alpha=0.45,
            )

            ax.set_xlabel("$V_Z$", fontsize=13)
            ax.set_ylabel("$\\mu$", fontsize=13)
            ax.set_title(
                f"Ensemble Majority Phase Map: {label} Conductance Curves\n(FPC={n_components}, K={n_clusters} | Dark overlay = Topological in >=75% realizations)",
                fontsize=13,
            )

            cbar = fig.colorbar(im, ax=ax, ticks=range(n_clusters))
            cbar.set_label("Majority K-Means Cluster Label")

            fig.tight_layout()
            fig.savefig(out_dir / f"phase_map_ensemble_{label}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        # 2. Individual realization phase maps stored in subfolder
        realiz_dir = out_dir / "phase_maps_by_realization"
        realiz_dir.mkdir(parents=True, exist_ok=True)
        for m, realiz_name in enumerate(unique_realizations):
            for label in curve_labels:
                realiz_mask = (all_labels == label) & (all_realizations == realiz_name)
                cluster_grid = cluster_labels_all[realiz_mask].reshape(Nmu, Nvz)
                pfaff_grid = all_pfaffians[realiz_mask].reshape(Nmu, Nvz)

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
                    colors=["black"],
                    linewidths=3.0,
                    alpha=0.9,
                )
                ax.contourf(
                    unique_vz,
                    unique_mu,
                    pfaff_grid,
                    levels=[0.5, 1.5],
                    colors=["black"],
                    alpha=0.45,
                )

                ax.set_xlabel("$V_Z$", fontsize=13)
                ax.set_ylabel("$\\mu$", fontsize=13)
                ax.set_title(
                    f"Phase Map ({realiz_name}): {label}\n(FPC={n_components}, K={n_clusters} | Dark overlay = Topological Phase)",
                    fontsize=12,
                )

                cbar = fig.colorbar(im, ax=ax, ticks=range(n_clusters))
                cbar.set_label("K-Means Cluster Label")

                fig.tight_layout()
                fig.savefig(realiz_dir / f"phase_map_{label}_{realiz_name}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)


if __name__ == "__main__":
    main()
