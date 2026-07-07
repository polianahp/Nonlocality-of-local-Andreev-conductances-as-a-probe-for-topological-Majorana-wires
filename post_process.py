#!/usr/bin/env python3
import os
import sys
import ast
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Ensure local imports work
sys.path.append(str(Path(__file__).parent.resolve()))

from config import PathConfigs
import helpers as hp

def generate_point_path(pdi_data, N, resl, mu_start, mu_end, Vz_start, Vz_end):
    """
    Extracts a set of N unique, evenly spaced points along a defined straight-line cut
    through the 2D parameter space (mu and V_z) from the dataset.
    Returns both the sorted points and their original indices in pdi_data.
    """
    pdi_params = pdi_data[:, 0:2]

    diff_vec = np.array([mu_end - mu_start, Vz_end - Vz_start])
    total_distance = np.linalg.norm(diff_vec)

    if total_distance == 0:
        unit_vec = np.array([0.0, 0.0])
        step_vec = np.array([0.0, 0.0])
        num_pts = 1
    else:
        unit_vec = diff_vec / total_distance
        step_vec = resl * unit_vec 
        num_pts = int(np.floor(total_distance / resl)) + 1

    start_vec = np.array([mu_start, Vz_start])
    pts = np.asarray([start_vec + (n * step_vec) for n in range(num_pts)])

    closest_indices = []

    for i in range(num_pts):
        tst = pts[i, :]
        is_close_mask = np.all(np.isclose(tst, pdi_params, atol=resl), axis=1)
        matched_indices = np.where(is_close_mask)[0]
        
        if len(matched_indices) > 0:
            matched_pdi_params = pdi_params[matched_indices]
            distances = np.linalg.norm(matched_pdi_params - tst, axis=1)
            closest_subset_index = np.argmin(distances)
            closest_original_index = matched_indices[closest_subset_index]
            closest_indices.append(closest_original_index)
        else:
            closest_indices.append(-1)

    closest_indices = np.array(closest_indices)

    valid_indices = closest_indices[closest_indices != -1]
    if len(valid_indices) == 0:
        print("Warning: No matching points found in the dataset within the search resolution.")
        return np.array([]), np.array([])

    unique_indices = np.unique(valid_indices)
    unique_points = pdi_params[unique_indices]

    dist_from_start = np.linalg.norm(unique_points - start_vec, axis=1)
    sort_order = np.argsort(dist_from_start)

    sorted_unique_points = unique_points[sort_order]
    sorted_unique_indices = unique_indices[sort_order]
    num_unique = len(sorted_unique_points)

    if N >= num_unique:
        sampled_points = sorted_unique_points
        sampled_indices = sorted_unique_indices
    else:
        sample_idx = np.round(np.linspace(0, num_unique - 1, N)).astype(int)
        sampled_points = sorted_unique_points[sample_idx]
        sampled_indices = sorted_unique_indices[sample_idx]

    return sampled_points, sampled_indices

def create_overlay_rgba(mu, V_z, I, prot_dat):
    unique_mu = np.unique(mu)
    unique_vz = np.unique(V_z)
    Nmu = len(unique_mu)
    Nvz = len(unique_vz)

    I_grid = np.zeros((Nmu, Nvz))
    prot_grid = np.zeros((Nmu, Nvz))

    mu_to_idx = {val: idx for idx, val in enumerate(unique_mu)}
    vz_to_idx = {val: idx for idx, val in enumerate(unique_vz)}

    for idx in range(len(mu)):
        mu_val = mu[idx]
        vz_val = V_z[idx]
        mu_i = mu_to_idx.get(mu_val, -1)
        vz_j = vz_to_idx.get(vz_val, -1)
        if mu_i != -1 and vz_j != -1:
            I_grid[mu_i, vz_j] = I[idx]
            prot_grid[mu_i, vz_j] = prot_dat[idx]

    rgba = np.ones((Nmu, Nvz, 4))
    for i in range(Nmu):
        for j in range(Nvz):
            val_I = I_grid[i, j]
            val_P = prot_grid[i, j]
            if val_I == 1.0 and val_P == 1.0:
                rgba[i, j] = [0.6, 0.25, 0.6, 1.0]
            elif val_I == 1.0:
                rgba[i, j] = [0.5, 0.5, 1.0, 1.0]
            elif val_P == 1.0:
                rgba[i, j] = [1.0, 0.5, 0.5, 1.0]
            else:
                rgba[i, j] = [1.0, 1.0, 1.0, 1.0]
    return unique_mu, unique_vz, rgba

def main():
    parser = argparse.ArgumentParser(description="Run full post-processing pipeline (2D overlays and 1D parameter cuts).")

    #indir = "Data/dis_realizations/disorder_realization_6_results"
    indir = "Data/Tdis_pfaff3"

    parser.add_argument("--dirname", type=str, default=indir, help="Directory name containing the simulation parameters and results.")
    parser.add_argument("-N", type=int, default=100, help="Number of points to sample along the cut.")
    parser.add_argument("--resl", type=float, default=0.02, help="Resolution/tolerance for matching points.")
    parser.add_argument("--mu-start", type=float, default=3.133929, help="Start Chemical Potential (mu) for the cut.")
    parser.add_argument("--mu-end", type=float, default=3.133929, help="End Chemical Potential (mu) for the cut.")
    parser.add_argument("--vz-start", type=float, default=0.0, help="Start Zeeman Field (V_z) for the cut.")
    parser.add_argument("--vz-end", type=float, default=1.4, help="End Zeeman Field (V_z) for the cut.")
    parser.add_argument("--target-vz", type=float, default=0.9140625, help="Predefined target V_z to draw a vertical reference line.")
    parser.add_argument("--kvals", type=int, default=12, help="Number of lowest eigenvalues to solve for.")
    parser.add_argument("--pdi-thresh", type=float, default=0.9, help="PDI filtering threshold.")
    parser.add_argument("--corr-thresh", type=float, default=0.7, help="Correlation threshold.")
    parser.add_argument("--window", type=float, default=0.03, help="Peak window threshold.")
    parser.add_argument("--stability-radius", type=float, default=0.025, help="Stability radius.")
    parser.add_argument("--stability-frac", type=float, default=1.0, help="Stability fraction.")
    parser.add_argument("--output", type=str, default="Plots/spectra_cut.png", help="Path to save the spectra plot.")
    parser.add_argument("--corr-output", type=str, default="Plots/correlation_cut.png", help="Path to save the correlation plot.")
    parser.add_argument("--gap-output", type=str, default="Plots/gap_nonlocal_cut.png", help="Path to save the gap and nonlocal conductance plot.")
    parser.add_argument("--peaks-output", type=str, default="Plots/peaks_cut.png", help="Path to save the peak width and height plot.")
    parser.add_argument("--overlap-output", type=str, default="Plots/overlap_plot.png", help="Path to save the 2D overlay plot.")
    parser.add_argument("--single-points", type=str, default="[]", help="List of (V_z, mu) tuples for single point plotting, e.g. '[(0.45323, 3.4564)]'")
    args = parser.parse_args()

    try:
        single_points = ast.literal_eval(args.single_points)
        if not isinstance(single_points, list):
            single_points = []
    except Exception as e:
        print(f"Warning: Could not parse --single-points ({e}). Defaulting to empty list.")
        single_points = []

    # Resolve input directory
    if os.path.isabs(args.dirname):
        dirname = Path(args.dirname)
    else:
        dirname = Path(PathConfigs.ROOT) / args.dirname
        if not dirname.exists():
            dirname = Path(args.dirname)

    print(f"Loading data from: {dirname}")

    params_path = dirname / "all_params.npz"
    pdi_data_path = dirname / "pdi_data.npy"
    params_list_path = dirname / "params_list.npy"
    peaks_left_path = dirname / "peaks_left.npy"
    peaks_right_path = dirname / "peaks_right.npy"
    brcl_path = dirname / "barrier_right_conductance_left_arr.npy"
    brcr_path = dirname / "barrier_right_conductance_right_arr.npy"
    glr_path = dirname / "barrier_right_GLR.npy"
    grl_path = dirname / "barrier_right_GRL.npy"
    gap_path = dirname / "topological_gap.npy"

    for filepath in [params_path, pdi_data_path, params_list_path, peaks_left_path, peaks_right_path,
                     brcl_path, brcr_path, glr_path, grl_path, gap_path]:
        if not filepath.exists():
            print(f"Error: Missing required file {filepath}")
            sys.exit(1)

    params_config = np.load(params_path, allow_pickle=True)
    pdi_data = np.load(pdi_data_path, allow_pickle=True)
    params_list = np.load(params_list_path, allow_pickle=True)
    peaks_left = np.load(peaks_left_path, allow_pickle=True)
    peaks_right = np.load(peaks_right_path, allow_pickle=True)
    brcl_all = np.load(brcl_path, allow_pickle=True)
    brcr_all = np.load(brcr_path, allow_pickle=True)
    glr_all = np.load(glr_path, allow_pickle=True)
    grl_all = np.load(grl_path, allow_pickle=True)
    gap_all = np.load(gap_path, allow_pickle=True)

    # Extract required parameters from config params
    t = float(params_config['t'])
    Delta0 = float(params_config['Delta0'])
    gamma = float(params_config['gamma'])
    alpha = float(params_config['alpha'])
    Ls = int(params_config['Ls'])
    V0 = float(params_config['V0'])
    Vdisx = params_config['Vdisx'] * V0

    mu = params_list[:, 1]
    V_z = params_list[:, 2]

    # Calculate I (Topological Winding Number)
    if pdi_data.shape[1] > 3:
        pdi_winding = pdi_data[:, 3]
    else:
        pdi_winding = pdi_data[:, 2]
    I = hp.filter_pdi(pdi_winding, thresh=args.pdi_thresh)

    # Calculate full correlations
    print("Computing correlations...")
    corrs = np.array([hp.calc_correlation(brcl_all[i], brcr_all[i]) for i in range(brcl_all.shape[0])])

    # Calculate decision map prot_dat
    print("Calculating protocol decision map (prot_dat)...")
    params = {
        "check_correlation"      : True,
        "check_resonance_peak"   : False,
        "check_negative_peaks"   : False,
        "check_monotonic"        : False,
        "check_peak_symmetry"    : False, 
        "check_peak_window"      : True,
        "check_island_stability" : True,
        
        "corr_thresh"     : 0.7,
        "window"          : 0.03,
        "stability_radius" : 0.025,
        "stability_frac"  : 1.0
    }

    prot_dat = hp.calc_protocol_v3(
        corrs,
        peaks_left,
        peaks_right,
        None,
        pdi_data,
        params
    )

    # ==========================================
    # STAGE 1: Generate Path and Plot 2D Overlay Map (Matplotlib)
    # ==========================================
    print(f"Generating point path along the cut: mu=({args.mu_start} -> {args.mu_end}), Vz=({args.vz_start} -> {args.vz_end})...")
    pts, sampled_indices = generate_point_path(pdi_data, args.N, args.resl, args.mu_start, args.mu_end, args.vz_start, args.vz_end)

    if len(pts) == 0:
        print("Error: No points generated.")
        sys.exit(1)

    actual_N = len(pts)
    print(f"Sampled {actual_N} points along the path.")

    # Calculate target point along the cut
    vz_values = pts[:, 1]
    closest_vz_idx = int(np.argmin(np.abs(vz_values - args.target_vz)))
    target_mu_val = pts[closest_vz_idx, 0]

    print("Generating 2D overlay plot...")
    unique_mu, unique_vz, rgba = create_overlay_rgba(mu, V_z, I, prot_dat)

    fig_overlay, ax_overlay = plt.subplots(figsize=(6, 8), dpi=150)
    ax_overlay.imshow(rgba, origin='lower', extent=[unique_vz[0], unique_vz[-1], unique_mu[0], unique_mu[-1]], aspect='auto')

    # Draw the exact cut path taken
    ax_overlay.plot([args.vz_start, args.vz_end], [args.mu_start, args.mu_end], color='black', linestyle='-', linewidth=2, zorder=4)
    # Draw the single point at the target Vz along the cut path
    target_vz_pt = vz_values[closest_vz_idx]
    target_mu_pt = target_mu_val
    ax_overlay.scatter(target_vz_pt, target_mu_pt, color='black', edgecolor='white', s=40, zorder=5)

    ax_overlay.set_title('Topological Region vs Protocol Positive Overlay', fontsize=12, pad=15)
    ax_overlay.set_xlabel(r'Zeeman Field $V_z$', fontsize=11)
    ax_overlay.set_ylabel(r'Chemical Potential $\mu$', fontsize=11)
    ax_overlay.set_xlim(0.0, 1.2)
    ax_overlay.set_ylim(0.0, 4.5)
    ax_overlay.spines['top'].set_visible(False)
    ax_overlay.spines['right'].set_visible(False)

    red_patch = mpatches.Patch(color=(1.0, 0.5, 0.5), label='Protocol Positive (prot_dat = 1)')
    blue_patch = mpatches.Patch(color=(0.5, 0.5, 1.0), label='Topological (I = 1)')
    purple_patch = mpatches.Patch(color=(0.6, 0.25, 0.6), label='Overlap (Both = 1)')
    
    cut_line_handle = mlines.Line2D([], [], color='black', linestyle='-', label='1D Cut Path')
    target_pt_handle = mlines.Line2D([], [], color='black', marker='o', markerfacecolor='black', markeredgecolor='white', markersize=6, linestyle='None', label=rf'Target $V_z = {target_vz_pt:.4f}$')
    
    ax_overlay.legend(handles=[red_patch, blue_patch, purple_patch, cut_line_handle, target_pt_handle], bbox_to_anchor=(0.5, -0.15),
                      loc='upper center', ncol=2, fontsize=9, frameon=True)

    overlay_out = Path(args.overlap_output)
    if not overlay_out.is_absolute():
        overlay_out = Path(PathConfigs.ROOT) / overlay_out
    overlay_out.parent.mkdir(parents=True, exist_ok=True)

    fig_overlay.tight_layout()
    fig_overlay.subplots_adjust(bottom=0.22)
    fig_overlay.savefig(overlay_out, dpi=300, bbox_inches='tight')
    plt.close(fig_overlay)
    print(f"Saved 2D overlay plot to: {overlay_out}")

    # ==========================================
    # STAGE 2: Slice Path-Specific Data (1D)
    # ==========================================

    # Slice path-specific data
    gap = gap_all[sampled_indices]
    pl = peaks_left[sampled_indices]
    pr = peaks_right[sampled_indices]
    brcl = brcl_all[sampled_indices]
    brcr = brcr_all[sampled_indices]
    glr_sym = glr_all[sampled_indices, 0]
    grl_sym = grl_all[sampled_indices, 0]
    glr_path_data = glr_all[sampled_indices]
    grl_path_data = grl_all[sampled_indices]


    # Physics Calculation for Spectrum and Pfaffian Invariants
    print(f"Calculating closed system spectra and Pfaffian invariants for {actual_N} points...")
    evals = np.zeros(shape=(actual_N, args.kvals))
    pf_invariants = np.zeros(actual_N)

    for i in range(actual_N):
        mu_val, vz_val = pts[i]
        scl = hp.build_system_closed(t, mu_val, gamma, Delta0, vz_val, alpha, Ls, Vdisx, a=1)
        evals[i, :] = hp.calc_spectrum(scl, k=args.kvals)
        pf_invariants[i] = hp.cal_pfaffian_invariant(
            ts=t, alphas=alpha, gamma=gamma, delta0=Delta0, Nx=Ls, Vdisx=Vdisx, V0=1.0, gm=vz_val, mu=mu_val
        )

    # Compute correlation and peaks details along the cut
    corr_vals = np.zeros(actual_N)
    corr_nonlocal_vals = np.zeros(actual_N)
    for i in range(actual_N):
        corr_vals[i] = hp.calc_correlation(brcl[i], brcr[i])
        corr_nonlocal_vals[i] = hp.calc_correlation(glr_path_data[i], grl_path_data[i])

    # Left peak widths and heights
    has_both_L = pl[:, 1]
    width_L = np.where(has_both_L == 1.0, pl[:, 2] - pl[:, 4], np.nan)
    height_L = pl[:, 3]

    # Right peak widths and heights
    has_both_R = pr[:, 1]
    width_R = np.where(has_both_R == 1.0, pr[:, 2] - pr[:, 4], np.nan)
    height_R = pr[:, 3]

    # Visualization of the cuts
    print("Visualizing parameter cut results...")

    def add_common_elements(ax):
        """Adds grid, target Vz line, dual x-axes, and overlay background shading."""
        # Background overlays (Topological / Protocol / Overlap)
        topo_added = False
        prot_added = False
        overlap_added = False

        for idx in range(actual_N):
            is_topo = (np.abs(pf_invariants[idx]) > 1e-5)
            is_prot = (prot_dat[sampled_indices[idx]] == 1.0)
            
            if is_topo and is_prot:
                ax.axvspan(idx - 0.5, idx + 0.5, color='forestgreen', alpha=0.25, zorder=0,
                           label="Pfaffian & Protocol" if not overlap_added else "")
                overlap_added = True
            elif is_topo:
                ax.axvspan(idx - 0.5, idx + 0.5, color='gold', alpha=0.25, zorder=0,
                           label="Pfaffian Only" if not topo_added else "")
                topo_added = True
            elif is_prot:
                ax.axvspan(idx - 0.5, idx + 0.5, color='deepskyblue', alpha=0.2, zorder=0,
                           label="Protocol Only" if not prot_added else "")
                prot_added = True

        # Vertical line for target Vz
        ax.axvline(closest_vz_idx, color='black', linestyle='-', linewidth=1.5, zorder=1,
                   label=rf"Target $V_z = {vz_values[closest_vz_idx]:.4f}$ ($\mu = {target_mu_val:.4f}$)")

        ax.grid(True, linestyle=':', alpha=0.5, zorder=1)
        ax.set_xlim(-0.5, actual_N - 0.5)

        # Select evenly spaced ticks to keep spacing uniform and reduce density
        num_ticks = min(6, actual_N)
        tick_indices = np.round(np.linspace(0, actual_N - 1, num_ticks)).astype(int)
        tick_indices = np.unique(tick_indices)

        # Primary X-Axis (mu)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f"{pts[idx, 0]:.3f}" for idx in tick_indices], rotation=45, ha='right', fontsize=9)
        ax.set_xlabel(r"Chemical Potential $\mu$ (meV)", fontsize=11, labelpad=15)

        # Secondary X-Axis (V_z)
        ax2 = ax.twiny()
        ax2.spines['bottom'].set_position(('outward', 65))
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')

        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels([f"{pts[idx, 1]:.3f}" for idx in tick_indices], rotation=45, ha='right', fontsize=9)
        ax2.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11, labelpad=15)

        # Hide top/right spines of both axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        return ax2

    # --- Plot 1: Low Energy Spectra ---
    fig_spectra, ax_spec = plt.subplots(figsize=(10, 7.5), dpi=150)
    add_common_elements(ax_spec)
    
    mid_idx = args.kvals // 2
    for j in range(mid_idx - 1):
        ax_spec.plot(range(actual_N), evals[:, j], '-', color='royalblue', alpha=0.7, linewidth=1.5, zorder=2)
    for j in range(mid_idx + 1, args.kvals):
        ax_spec.plot(range(actual_N), evals[:, j], '-', color='royalblue', alpha=0.7, linewidth=1.5, zorder=2)

    ax_spec.plot(range(actual_N), evals[:, mid_idx - 1], '-', color='red', alpha=0.9, linewidth=2.5, zorder=3)
    ax_spec.plot(range(actual_N), evals[:, mid_idx], '-', color='red', alpha=0.9, linewidth=2.5, zorder=3)
    ax_spec.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6, zorder=1)

    ax_spec.set_ylabel("Energy (meV)", fontsize=12)
    ax_spec.set_title("Low Energy Spectra along Cut", fontsize=13)
    
    handles, labels = ax_spec.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_spec.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=3, fontsize=10)

    # --- Plot 2: Local & Nonlocal Conductance Correlation ---
    fig_corr, ax_corr = plt.subplots(figsize=(10, 7.5), dpi=150)
    add_common_elements(ax_corr)
    
    ax_corr.plot(range(actual_N), corr_vals, '-', color='forestgreen', linewidth=2.5, label="Local Conductance Correlation", zorder=3)
    ax_corr.plot(range(actual_N), corr_nonlocal_vals, '--', color='darkorchid', linewidth=2.0, label="Nonlocal Conductance Correlation", zorder=3)
    ax_corr.set_ylabel("Correlation", fontsize=12)
    ax_corr.set_title("Conductance Correlation along Cut", fontsize=13)
    
    handles, labels = ax_corr.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_corr.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=3, fontsize=10)

    # --- Plot 3: Topological Gap and Nonlocal Conductance (twin-y) ---
    fig_gap, ax_gap = plt.subplots(figsize=(10, 7.5), dpi=150)
    ax2_gap = add_common_elements(ax_gap)
    ax_nonlocal = ax_gap.twinx()
    
    ax2_gap.spines['right'].set_visible(False)
    
    ax_gap.plot(range(actual_N), gap, '-', color='darkorange', linewidth=2.5, label="Topological Gap", zorder=3)
    ax_gap.set_ylabel("Gap (meV)", color='darkorange', fontsize=12)
    ax_gap.tick_params(axis='y', labelcolor='darkorange')

    ax_nonlocal.plot(range(actual_N), glr_sym, '--', color='purple', linewidth=1.5, alpha=0.8, label=r"$G_{LR}$ (Symmetric)", zorder=3)
    ax_nonlocal.plot(range(actual_N), grl_sym, '--', color='crimson', linewidth=1.5, alpha=0.8, label=r"$G_{RL}$ (Symmetric)", zorder=3)
    ax_nonlocal.set_ylabel(r"Nonlocal Conductance ($e^2/h$)", color='purple', fontsize=12)
    ax_nonlocal.tick_params(axis='y', labelcolor='purple')
    
    ax_nonlocal.spines['top'].set_visible(False)
    ax_nonlocal.spines['left'].set_visible(False)
    ax_nonlocal.spines['right'].set_visible(True)
    
    ax_gap.set_title("Topological Gap & Symmetric Nonlocal Conductance along Cut", fontsize=13)
    
    handles_gap, labels_gap = ax_gap.get_legend_handles_labels()
    handles_nl, labels_nl = ax_nonlocal.get_legend_handles_labels()
    by_label = dict(zip(labels_gap + labels_nl, handles_gap + handles_nl))
    ax_gap.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=3, fontsize=10)

    # --- Plot 4: Peak Width and Height (twin-y) ---
    fig_peaks, ax_width = plt.subplots(figsize=(10, 7.5), dpi=150)
    ax2_peaks = add_common_elements(ax_width)
    ax_height = ax_width.twinx()
    
    ax2_peaks.spines['right'].set_visible(False)

    ax_width.plot(range(actual_N), width_L, '-', color='navy', linewidth=2.5, label="Left Peak Width", zorder=3)
    ax_width.plot(range(actual_N), width_R, '--', color='royalblue', linewidth=2.5, label="Right Peak Width", zorder=3)
    ax_width.set_ylabel("Peak Width (meV)", color='navy', fontsize=12)
    ax_width.tick_params(axis='y', labelcolor='navy')

    ax_height.plot(range(actual_N), height_L, '-', color='darkred', linewidth=1.5, alpha=0.8, label="Left Peak Height", zorder=3)
    ax_height.plot(range(actual_N), height_R, '--', color='crimson', linewidth=1.5, alpha=0.8, label="Right Peak Height", zorder=3)
    ax_height.set_ylabel(r"Peak Height ($e^2/h$)", color='darkred', fontsize=12)
    ax_height.tick_params(axis='y', labelcolor='darkred')

    ax_height.spines['top'].set_visible(False)
    ax_height.spines['left'].set_visible(False)
    ax_height.spines['right'].set_visible(True)

    ax_width.set_title("Peak Width & Height along Cut", fontsize=13)

    handles_w, labels_w = ax_width.get_legend_handles_labels()
    handles_h, labels_h = ax_height.get_legend_handles_labels()
    by_label = dict(zip(labels_w + labels_h, handles_w + handles_h))
    ax_width.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=3, fontsize=10)

    # Save all 4 plots
    for output_name, figure in [
        (args.output, fig_spectra),
        (args.corr_output, fig_corr),
        (args.gap_output, fig_gap),
        (args.peaks_output, fig_peaks)
    ]:
        output_path = Path(output_name)
        if not output_path.is_absolute():
            output_path = Path(PathConfigs.ROOT) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure.tight_layout()
        figure.subplots_adjust(bottom=0.4)
        figure.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(figure)
        print(f"Saved plot to: {output_path}")

    # ==========================================
    # STAGE 3: Single Points Plotting & Point Map
    # ==========================================
    if len(single_points) > 0:
        print(f"Processing {len(single_points)} single points for detailed plotting...")
        single_points_dir = dirname / "Plots" / "Single_Points"
        single_points_dir.mkdir(parents=True, exist_ok=True)

        mu_n = float(params_config['mu_n'])
        mu_leads = float(params_config['mu_leads'])
        Ln = int(params_config['Ln'])
        Lb = int(params_config['Lb'])
        try:
            barrier_l = float(params_config['barrier0'])
        except Exception:
            try:
                barrier_l = float(params_config['barrier_l'])
            except Exception:
                barrier_l = 2.0

        num_sweep_points = 100
        barrier_sweep = np.linspace(-70 * barrier_l, 70 * barrier_l, num_sweep_points)
        energies = np.linspace(-0.5, 0.5, 101)

        for pt_idx, pt in enumerate(single_points):
            vz_val, mu_val = float(pt[0]), float(pt[1])
            folder_name = f"Vz{vz_val:.3f}".replace('.', '') + "_" + f"mu{mu_val:.3f}".replace('.', '')
            pt_dir = single_points_dir / folder_name
            pt_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{pt_idx+1}/{len(single_points)}] Generating plots for Vz={vz_val}, mu={mu_val} -> {pt_dir}")

            # 1. Right Barrier Sweep (varying UR, holding UL=barrier_l)
            cond_left_R = np.zeros(num_sweep_points)
            cond_right_R = np.zeros(num_sweep_points)
            for k in range(num_sweep_points):
                syst_R = hp.build_system(
                    t=t, mu=mu_val, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz_val,
                    alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
                    barrier_l=barrier_l, barrier_r=barrier_sweep[k], Vdisx=Vdisx, a=1
                )
                cL, cR = hp.calc_conductance(syst_R, energy=0.0, return_smatrix=False)
                cond_left_R[k] = cL
                cond_right_R[k] = cR

            # 2. Left Barrier Sweep (varying UL, holding UR=barrier_l)
            cond_left_L = np.zeros(num_sweep_points)
            cond_right_L = np.zeros(num_sweep_points)
            for k in range(num_sweep_points):
                syst_L = hp.build_system(
                    t=t, mu=mu_val, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz_val,
                    alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
                    barrier_l=barrier_sweep[k], barrier_r=barrier_l, Vdisx=Vdisx, a=1
                )
                cL, cR = hp.calc_conductance(syst_L, energy=0.0, return_smatrix=False)
                cond_left_L[k] = cL
                cond_right_L[k] = cR

            x_data = barrier_sweep / (barrier_l if barrier_l != 0 else 1.0)
            idx_sym = np.argmin(np.abs(barrier_sweep - barrier_l))
            lw = 3.0

            # Save Right Sweep Left Conductance
            fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
            normed_GL_R = cond_left_R / (cond_left_R[idx_sym] if cond_left_R[idx_sym] != 0 else 1.0)
            ax.plot(x_data, normed_GL_R, color="green", linewidth=lw)
            ax.set_xlabel(r"$U_{R}/U_{L}$", fontsize=14)
            ax.set_ylabel(r"$G_{LL}/G_{LL, sym}$", fontsize=14)
            ax.set_title(rf"Right Sweep $G_{{LL}}$ ($V_z={vz_val:.3f}, \mu={mu_val:.3f}$)")
            fig.tight_layout()
            fig.savefig(pt_dir / "Conductances_Left_RightSweep.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Save Right Sweep Right Conductance
            fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
            normed_GR_R = cond_right_R / (cond_right_R[idx_sym] if cond_right_R[idx_sym] != 0 else 1.0)
            ax.plot(x_data, normed_GR_R, color="green", linewidth=lw)
            ax.set_xlabel(r"$U_{R}/U_{L}$", fontsize=14)
            ax.set_ylabel(r"$G_{RR}/G_{RR, sym}$", fontsize=14)
            ax.set_title(rf"Right Sweep $G_{{RR}}$ ($V_z={vz_val:.3f}, \mu={mu_val:.3f}$)")
            fig.tight_layout()
            fig.savefig(pt_dir / "Conductances_Right_RightSweep.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Save Left Sweep Left Conductance
            fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
            normed_GL_L = cond_left_L / (cond_left_L[idx_sym] if cond_left_L[idx_sym] != 0 else 1.0)
            ax.plot(x_data, normed_GL_L, color="green", linewidth=lw)
            ax.set_xlabel(r"$U_{L}/U_{R}$", fontsize=14)
            ax.set_ylabel(r"$G_{LL}/G_{LL, sym}$", fontsize=14)
            ax.set_title(rf"Left Sweep $G_{{LL}}$ ($V_z={vz_val:.3f}, \mu={mu_val:.3f}$)")
            fig.tight_layout()
            fig.savefig(pt_dir / "Conductances_Left_LeftSweep.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Save Left Sweep Right Conductance
            fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
            normed_GR_L = cond_right_L / (cond_right_L[idx_sym] if cond_right_L[idx_sym] != 0 else 1.0)
            ax.plot(x_data, normed_GR_L, color="green", linewidth=lw)
            ax.set_xlabel(r"$U_{L}/U_{R}$", fontsize=14)
            ax.set_ylabel(r"$G_{RR}/G_{RR, sym}$", fontsize=14)
            ax.set_title(rf"Left Sweep $G_{{RR}}$ ($V_z={vz_val:.3f}, \mu={mu_val:.3f}$)")
            fig.tight_layout()
            fig.savefig(pt_dir / "Conductances_Right_LeftSweep.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            # 3. dIdV Spectrum at Nominal Barrier
            syst_nom = hp.build_system(
                t=t, mu=mu_val, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz_val,
                alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
                barrier_l=barrier_l, barrier_r=barrier_l, Vdisx=Vdisx, a=1
            )
            dIdV_left, dIdV_right, _, _, _ = hp.calc_dIdV(syst_nom, energies)
            fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
            ax.plot(energies, dIdV_left, color='royalblue', linewidth=2.0, label='Left dI/dV')
            ax.plot(energies, dIdV_right, color='darkorange', linewidth=2.0, label='Right dI/dV')
            ax.set_xlabel("Energy (meV)", fontsize=13)
            ax.set_ylabel("Differential Conductance (dI/dV)", fontsize=13)
            ax.set_title(rf"dI/dV Spectrum ($V_z={vz_val:.3f}, \mu={mu_val:.3f}$)")
            ax.legend(fontsize=11)
            ax.grid(True, linestyle=':', alpha=0.5)
            fig.tight_layout()
            fig.savefig(pt_dir / "dIdV.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            # 4. MZM Wavefunction
            scl = hp.build_system_closed(t, mu_val, gamma, Delta0, vz_val, alpha, Ls, Vdisx, a=1)
            evals_cl, evecs_cl = hp.solve_ham(scl, k=2)
            rho_M1, rho_M2, _ = hp.get_psiM_density(evals_cl, evecs_cl)
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            ax.plot(rho_M1, label='Majorana Left (M1)', color='cyan', linewidth=2.0)
            ax.plot(rho_M2, label='Majorana Right (M2)', color='orange', linewidth=2.0)
            ax.set_xlabel("Site Index", fontsize=13)
            ax.set_ylabel("Probability Density", fontsize=13)
            ax.set_title(rf"Majorana Modes ($V_z={vz_val:.3f}, \mu={mu_val:.3f}$)")
            ax.legend(fontsize=11)
            ax.grid(True, linestyle=':', alpha=0.5)
            fig.tight_layout()
            fig.savefig(pt_dir / "wavefunction.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

        # 5. Generate point_map.png
        print("Generating 2D single point map (point_map.png)...")
        fig_map, ax_map = plt.subplots(figsize=(6, 8), dpi=150)
        ax_map.imshow(rgba, origin='lower', extent=[unique_vz[0], unique_vz[-1], unique_mu[0], unique_mu[-1]], aspect='auto')

        pts_array = np.array(single_points, dtype=float)
        ax_map.scatter(pts_array[:, 0], pts_array[:, 1], color='black', edgecolor='white', s=50, zorder=5)

        ax_map.set_title('Topological Region vs Protocol Overlay (Single Points)', fontsize=12, pad=15)
        ax_map.set_xlabel(r'Zeeman Field $V_z$', fontsize=11)
        ax_map.set_ylabel(r'Chemical Potential $\mu$', fontsize=11)
        ax_map.set_xlim(0.0, 1.2)
        ax_map.set_ylim(0.0, 4.5)
        ax_map.spines['top'].set_visible(False)
        ax_map.spines['right'].set_visible(False)

        red_patch = mpatches.Patch(color=(1.0, 0.5, 0.5), label='Protocol Positive (prot_dat = 1)')
        blue_patch = mpatches.Patch(color=(0.5, 0.5, 1.0), label='Topological (I = 1)')
        purple_patch = mpatches.Patch(color=(0.6, 0.25, 0.6), label='Overlap (Both = 1)')
        pts_handle = mlines.Line2D([], [], color='black', marker='o', markerfacecolor='black', markeredgecolor='white', markersize=7, linestyle='None', label='Single Points')

        ax_map.legend(handles=[red_patch, blue_patch, purple_patch, pts_handle], bbox_to_anchor=(0.5, -0.15),
                      loc='upper center', ncol=2, fontsize=9, frameon=True)

        fig_map.tight_layout()
        fig_map.subplots_adjust(bottom=0.22)
        map_out = single_points_dir / "point_map.png"
        fig_map.savefig(map_out, dpi=300, bbox_inches='tight')
        plt.close(fig_map)
        print(f"Saved point map to: {map_out}")

if __name__ == "__main__":
    main()
