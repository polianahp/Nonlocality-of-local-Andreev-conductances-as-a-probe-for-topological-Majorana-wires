#!/usr/bin/env python3
import os
import sys
import ast
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors

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

def generate_continuous_phase_maps(dirname, mu, V_z, I, eff_gap, gap_transport_all, site_localizations, weight_localization_arr, curvature_arr=None):
    print("\nGenerating global 2D continuous phase maps (effective gap, transport gap, site & weight localizations)...")
    plots_dir = dirname / "Plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def save_continuous_phase_map(values_1d, title_str, filename_str, cbar_label, cmap='viridis', vmin=None, vmax=None):
        if values_1d is None or len(values_1d) != len(mu):
            print(f"Warning: Skipping {filename_str} because data is missing or size mismatch (len={len(values_1d) if values_1d is not None else 'None'}, expected {len(mu)}).")
            return

        unique_mu_grid = np.unique(mu)
        unique_vz_grid = np.unique(V_z)
        Nmu = len(unique_mu_grid)
        Nvz = len(unique_vz_grid)

        val_grid = np.full((Nmu, Nvz), np.nan)
        I_grid = np.zeros((Nmu, Nvz))

        mu_to_idx = {val: idx for idx, val in enumerate(unique_mu_grid)}
        vz_to_idx = {val: idx for idx, val in enumerate(unique_vz_grid)}

        for idx in range(len(mu)):
            mu_i = mu_to_idx.get(mu[idx], -1)
            vz_j = vz_to_idx.get(V_z[idx], -1)
            if mu_i != -1 and vz_j != -1:
                val_grid[mu_i, vz_j] = values_1d[idx]
                I_grid[mu_i, vz_j] = I[idx]

        fig_map, ax_map = plt.subplots(figsize=(7, 8), dpi=150)
        im = ax_map.imshow(val_grid, origin='lower', extent=[unique_vz_grid[0], unique_vz_grid[-1], unique_mu_grid[0], unique_mu_grid[-1]], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

        cbar = fig_map.colorbar(im, ax=ax_map, pad=0.03, extend='max' if vmax is not None else 'neither')
        cbar.set_label(cbar_label, fontsize=11)

        # Overlay topological phase volume (I == 1) with dark gray overlay and distinct boundary line
        if np.any(I_grid == 1) and np.any(I_grid == 0):
            VZ, MU = np.meshgrid(unique_vz_grid, unique_mu_grid)
            ax_map.contourf(VZ, MU, I_grid, levels=[0.5, 1.5], colors=['#404040'], alpha=0.55)
            ax_map.contour(VZ, MU, I_grid, levels=[0.5], colors=['#202020'], linewidths=1.5, alpha=0.9)
            topological_patch = mpatches.Patch(facecolor='#404040', edgecolor='#202020', linewidth=1.5, alpha=0.6, label='Topological Phase ($I = 1$)')
            ax_map.legend(handles=[topological_patch], loc='upper right', fontsize=9, frameon=True)

        ax_map.set_title(title_str, fontsize=12, pad=15)
        ax_map.set_xlabel(r'Zeeman Field $V_z$', fontsize=11)
        ax_map.set_ylabel(r'Chemical Potential $\mu$', fontsize=11)
        ax_map.set_xlim(0.0, 1.2)
        ax_map.set_ylim(0.0, 4.5)
        ax_map.spines['top'].set_visible(False)
        ax_map.spines['right'].set_visible(False)

        fig_map.tight_layout()
        map_out = plots_dir / filename_str
        fig_map.savefig(map_out, dpi=300, bbox_inches='tight')
        plt.close(fig_map)
        print(f"Saved phase map to: {map_out}")

    # 1. Effective Topological Gap (First Excited State E_1)
    if eff_gap is not None:
        save_continuous_phase_map(eff_gap, r'Phase Map: First Excited State ($E_1$)', 'phase_map_effective_topological_gap.png', r'First Excited State ($E_1$)', cmap='viridis')
    else:
        print("Warning: eff_gap not available for Effective Topological Gap phase map.")

    # 2. Transport Gap
    if gap_transport_all is not None:
        save_continuous_phase_map(gap_transport_all, r'Phase Map: Transport Gap ($\Delta_{\mathrm{transport}}$)', 'phase_map_transport_gap.png', r'Transport Gap ($\Delta_{\mathrm{transport}}$)', cmap='viridis', vmin=0.0, vmax=0.25)
    else:
        print("Warning: gap_transport_all not available for Transport Gap phase map.")

    # 3. Site Localizations
    if site_localizations is not None:
        save_continuous_phase_map(site_localizations, r'Phase Map: Site Localizations ($80\%$ Density)', 'phase_map_site_localizations.png', r'Sites from Ends ($80\%$ Density)', cmap='magma_r')
    else:
        print("Warning: site_localizations not available for Site Localizations phase map.")

    # 4. Weight Localizations
    if weight_localization_arr is not None:
        save_continuous_phase_map(weight_localization_arr, r'Phase Map: Weight Localizations ($90\%$ Density Fraction)', 'phase_map_weight_localizations.png', r'Fractional Wire Length ($90\%$ Density)', cmap='magma_r')
    else:
        print("Warning: weight_localization_arr not available for Weight Localizations phase map.")

    # 4b. Curvature
    if curvature_arr is not None and not np.all(np.isnan(curvature_arr)):
        save_continuous_phase_map(curvature_arr, r'Phase Map: Zero-Bias Curvature (Gapless Points)', 'phase_map_curvature.png', r'Curvature ($d^2I/dV^2$)', cmap='coolwarm')
    elif curvature_arr is not None:
        print("Warning: curvature_arr exists but contains only NaNs.")

    # 5. Gapped Island Binary Classification & Qualified Gap Maps (70% Gapless Boundary)
    if gap_transport_all is not None:
        unique_mu_grid = np.unique(mu)
        unique_vz_grid = np.unique(V_z)
        Nmu = len(unique_mu_grid)
        Nvz = len(unique_vz_grid)

        gap_grid = np.full((Nmu, Nvz), np.nan)
        I_grid = np.zeros((Nmu, Nvz))

        mu_to_idx = {val: idx for idx, val in enumerate(unique_mu_grid)}
        vz_to_idx = {val: idx for idx, val in enumerate(unique_vz_grid)}

        for idx in range(len(mu)):
            mu_i = mu_to_idx.get(mu[idx], -1)
            vz_j = vz_to_idx.get(V_z[idx], -1)
            if mu_i != -1 and vz_j != -1:
                gap_grid[mu_i, vz_j] = gap_transport_all[idx]
                I_grid[mu_i, vz_j] = I[idx]

        island_res = hp.compute_gapped_islands_and_boundaries(
            gap_grid,
            min_gap_threshold=1e-4,
            pct_boundary_threshold=0.70,
            variance=3
        )
        classified_grid = island_res["classified_grid"]

        # Plot 5: Binary Gapped Island Map
        fig_bin, ax_bin = plt.subplots(figsize=(7, 8), dpi=150)
        cmap_bin = mcolors.ListedColormap(['#1e1e2e', '#d62828', '#06d6a0'])
        norm_bin = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_bin.N)

        im_bin = ax_bin.imshow(
            classified_grid,
            origin='lower',
            extent=[unique_vz_grid[0], unique_vz_grid[-1], unique_mu_grid[0], unique_mu_grid[-1]],
            aspect='auto',
            cmap=cmap_bin,
            norm=norm_bin
        )

        cbar_bin = fig_bin.colorbar(im_bin, ax=ax_bin, pad=0.03, ticks=[0, 1, 2])
        cbar_bin.ax.set_yticklabels(['Gapless', 'Rejected (<70%)', 'Valid Island (≥70%)'], fontsize=9)

        if np.any(I_grid == 1) and np.any(I_grid == 0):
            VZ, MU = np.meshgrid(unique_vz_grid, unique_mu_grid)
            ax_bin.contour(VZ, MU, I_grid, levels=[0.5], colors=['#ffffff'], linewidths=1.2, linestyles='--')
            topo_line = mlines.Line2D([], [], color='#ffffff', linestyle='--', linewidth=1.2, label='Topological Boundary ($I=1$)')
            ax_bin.legend(handles=[topo_line], loc='upper right', fontsize=9, frameon=True, facecolor='#1e1e2e', edgecolor='none', labelcolor='white')

        ax_bin.set_title(r'Phase Map: Gapped Islands ($\geq 70\%$ Gapless Boundary)', fontsize=12, pad=15)
        ax_bin.set_xlabel(r'Zeeman Field $V_z$', fontsize=11)
        ax_bin.set_ylabel(r'Chemical Potential $\mu$', fontsize=11)
        ax_bin.set_xlim(0.0, 1.2)
        ax_bin.set_ylim(0.0, 4.5)
        ax_bin.spines['top'].set_visible(False)
        ax_bin.spines['right'].set_visible(False)

        fig_bin.tight_layout()
        bin_out = plots_dir / "phase_map_gapped_islands_binary.png"
        fig_bin.savefig(bin_out, dpi=300, bbox_inches='tight')
        plt.close(fig_bin)
        print(f"Saved phase map to: {bin_out}")

        # Plot 6: Qualified Transport Gap (Masked) Map
        qualified_gap_grid = np.where(classified_grid == 2, gap_grid, np.nan)
        fig_qual, ax_qual = plt.subplots(figsize=(7, 8), dpi=150)
        ax_qual.set_facecolor('#2b2b36')

        im_qual = ax_qual.imshow(
            qualified_gap_grid,
            origin='lower',
            extent=[unique_vz_grid[0], unique_vz_grid[-1], unique_mu_grid[0], unique_mu_grid[-1]],
            aspect='auto',
            cmap='viridis',
            vmin=0.0,
            vmax=0.25
        )

        cbar_qual = fig_qual.colorbar(im_qual, ax=ax_qual, pad=0.03, extend='max')
        cbar_qual.set_label(r'Qualified Transport Gap ($\Delta_{\mathrm{transport}}$)', fontsize=11)

        VZ, MU = np.meshgrid(unique_vz_grid, unique_mu_grid)
        if np.any(classified_grid == 2):
            ax_qual.contour(VZ, MU, (classified_grid == 2).astype(int), levels=[0.5], colors=['#00ffff'], linewidths=1.5)

        if np.any(I_grid == 1) and np.any(I_grid == 0):
            ax_qual.contour(VZ, MU, I_grid, levels=[0.5], colors=['#ffffff'], linewidths=1.0, linestyles=':')

        ax_qual.set_title(r'Phase Map: Qualified Transport Gap (Islands $\geq 70\%$ Gapless Boundary)', fontsize=12, pad=15)
        ax_qual.set_xlabel(r'Zeeman Field $V_z$', fontsize=11)
        ax_qual.set_ylabel(r'Chemical Potential $\mu$', fontsize=11)
        ax_qual.set_xlim(0.0, 1.2)
        ax_qual.set_ylim(0.0, 4.5)
        ax_qual.spines['top'].set_visible(False)
        ax_qual.spines['right'].set_visible(False)

        fig_qual.tight_layout()
        qual_out = plots_dir / "phase_map_qualified_transport_gap.png"
        fig_qual.savefig(qual_out, dpi=300, bbox_inches='tight')
        plt.close(fig_qual)
        print(f"Saved phase map to: {qual_out}")



def process_single_directory(dirname, args, single_points):
    print(f"\n==================================================")
    print(f"Loading data from: {dirname}")
    print(f"==================================================")

    params_path = dirname / "all_params.npz"
    pdi_data_path = dirname / "pdi_data.npy"
    params_list_path = dirname / "params_list.npy"
    peaks_left_path = dirname / "peaks_left.npy"
    peaks_right_path = dirname / "peaks_right.npy"
    brcl_path = dirname / "barrier_right_conductance_left_arr.npy"
    brcr_path = dirname / "barrier_right_conductance_right_arr.npy"
    blcl_path = dirname / "barrier_left_conductance_left_arr.npy"
    blcr_path = dirname / "barrier_left_conductance_right_arr.npy"
    glr_path = dirname / "barrier_right_GLR.npy"
    grl_path = dirname / "barrier_right_GRL.npy"
    gap_path = dirname / "topological_gap.npy"

    for filepath in [params_path, pdi_data_path, params_list_path, peaks_left_path, peaks_right_path,
                     brcl_path, brcr_path, blcl_path, blcr_path, glr_path, grl_path, gap_path]:
        if not filepath.exists():
            print(f"Error: Missing required file {filepath} in {dirname}. Skipping directory.")
            return

    params_config = np.load(params_path, allow_pickle=True)
    pdi_data = np.load(pdi_data_path, allow_pickle=True)
    params_list = np.load(params_list_path, allow_pickle=True)
    peaks_left = np.load(peaks_left_path, allow_pickle=True)
    peaks_right = np.load(peaks_right_path, allow_pickle=True)
    brcl_all = np.load(brcl_path, allow_pickle=True)
    brcr_all = np.load(brcr_path, allow_pickle=True)
    blcl_all = np.load(blcl_path, allow_pickle=True)
    blcr_all = np.load(blcr_path, allow_pickle=True)
    glr_all = np.load(glr_path, allow_pickle=True)
    grl_all = np.load(grl_path, allow_pickle=True)
    gap_all = np.load(gap_path, allow_pickle=True)

    dIdVs_LR_path = dirname / "dIdVs_LR.npy"
    dIdVs_RL_path = dirname / "dIdVs_RL.npy"
    energies_path = dirname / "energies.npy"
    gap_transport_path = dirname / "gap_transport_all.npy"
    spectrum_arr_path = dirname / "spectrum_arr.npy"
    site_localizations_path = dirname / "site_localizations.npy"
    weight_localization_path = dirname / "weight_localization_arr.npy"
    fine_zero_bias_path = dirname / "fine_zero_bias_conductance.npy"

    spectrum_arr = np.load(spectrum_arr_path, allow_pickle=True) if spectrum_arr_path.exists() else None
    site_localizations = np.load(site_localizations_path, allow_pickle=True) if site_localizations_path.exists() else None
    weight_localization_arr = np.load(weight_localization_path, allow_pickle=True) if weight_localization_path.exists() else None
    fine_zero_bias_conductance = np.load(fine_zero_bias_path, allow_pickle=True) if fine_zero_bias_path.exists() else None
    
    if dIdVs_LR_path.exists() and dIdVs_RL_path.exists() and energies_path.exists():
        print("Extracting anti-symmetrized non-local transport gaps (Delta_LR, Delta_RL, Mutual)...")
        dIdVs_LR_all = np.load(dIdVs_LR_path, mmap_mode='r')
        dIdVs_RL_all = np.load(dIdVs_RL_path, mmap_mode='r')
        energies_all = np.load(energies_path)
        
        if gap_transport_path.exists() and args.skip_recalc:
            gap_transport_data = np.load(gap_transport_path, allow_pickle=True)
            if isinstance(gap_transport_data, np.ndarray) and gap_transport_data.ndim == 2 and gap_transport_data.shape[1] == 3:
                gap_LR_all, gap_RL_all, gap_transport_all = gap_transport_data[:, 0], gap_transport_data[:, 1], gap_transport_data[:, 2]
            else:
                gap_LR_all, _, _ = hp.extract_nonlocal_gap(energies_all, dIdVs_LR_all)
                gap_RL_all, _, _ = hp.extract_nonlocal_gap(energies_all, dIdVs_RL_all)
                gap_transport_all = hp.nanmin_gap_combination(gap_LR_all, gap_RL_all)
                np.save(gap_transport_path, np.column_stack([gap_LR_all, gap_RL_all, gap_transport_all]))
        else:
            gap_LR_all, _, _ = hp.extract_nonlocal_gap(energies_all, dIdVs_LR_all)
            gap_RL_all, _, _ = hp.extract_nonlocal_gap(energies_all, dIdVs_RL_all)
            gap_transport_all = hp.nanmin_gap_combination(gap_LR_all, gap_RL_all)
            np.save(gap_transport_path, np.column_stack([gap_LR_all, gap_RL_all, gap_transport_all]))
        has_nonlocal_gaps = True
    else:
        gap_LR_all = gap_RL_all = gap_transport_all = None
        has_nonlocal_gaps = False

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

    # Apply gapless mask and compute E1 effective gap
    if spectrum_arr is not None:
        mask_gapless = np.abs(spectrum_arr[:, 2]) <= args.epsilon
        eff_gap_val = spectrum_arr[:, 3]
        eff_gap = np.where(mask_gapless, eff_gap_val, np.nan)
    else:
        mask_gapless = np.ones(len(mu), dtype=bool)
        eff_gap = None
        
    # Mask the arrays that are plotted
    if gap_transport_all is not None:
        gap_transport_all = np.where(mask_gapless, gap_transport_all, np.nan)
    if site_localizations is not None:
        site_localizations = np.where(mask_gapless, site_localizations, np.nan)
    if weight_localization_arr is not None:
        weight_localization_arr = np.where(mask_gapless, weight_localization_arr, np.nan)

    # Compute Curvature from pre-calculated 7-point array
    curvature_arr = np.full(len(mu), np.nan)
    if spectrum_arr is not None and fine_zero_bias_conductance is not None:
        print("Computing zero-bias curvature for gapless points...")
        eps = args.curvature_epsilon
        # 7 points are at [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06]
        # Central point (0.0) is at index 3
        idx_zero = 3
        # Calculate step based on epsilon (e.g., eps=0.02 -> step=1, eps=0.04 -> step=2)
        step = int(np.round(eps / 0.02))
        if 1 <= step <= 3:
            f_minus = fine_zero_bias_conductance[:, idx_zero - step]
            f_zero = fine_zero_bias_conductance[:, idx_zero]
            f_plus = fine_zero_bias_conductance[:, idx_zero + step]
            
            cond_valid = f_zero > args.cond_threshold
            valid_mask = cond_valid & mask_gapless
            
            curvature_arr[valid_mask] = (f_plus[valid_mask] - 2 * f_zero[valid_mask] + f_minus[valid_mask]) / (eps ** 2)
        else:
            print(f"Warning: Epsilon {eps} cannot be mapped to the 0.02 spacing grid for curvature calculation.")

    if getattr(args, 'phase_maps_only', False) or getattr(args, 'skip_single_points', False):
        print("Flag --phase-maps-only set: Skipping single point cut calculations and point maps.")
        generate_continuous_phase_maps(dirname, mu, V_z, I, eff_gap, gap_transport_all, site_localizations, weight_localization_arr, curvature_arr)
        return

    # Calculate full correlations for Right and Left sweeps
    print("Computing correlations for Right and Left barrier sweeps...")
    barrier_arr_path = dirname / "barrier_arr.npy"
    barrier_arr = np.load(barrier_arr_path, allow_pickle=True) if barrier_arr_path.exists() else None
    corrs_R = np.array([hp.calc_correlation(brcl_all[i], brcr_all[i], barrier_arr=barrier_arr) for i in range(brcl_all.shape[0])])
    corrs_L = np.array([hp.calc_correlation(blcl_all[i], blcr_all[i], barrier_arr=barrier_arr) for i in range(blcl_all.shape[0])])

    # Calculate decision map prot_dat for Right and Left sweeps
    print("Calculating protocol decision maps (prot_dat)...")
    params = {
        "check_correlation"      : True,
        "check_resonance_peak"   : False,
        "check_negative_peaks"   : False,
        "check_monotonic"        : False,
        "check_peak_symmetry"    : False, 
        "check_peak_window"      : True,
        "check_island_stability" : True,
        
        "corr_thresh"     : args.corr_thresh,
        "window"          : args.window,
        "stability_radius" : args.stability_radius,
        "stability_frac"  : args.stability_frac
    }

    prot_dat_R = hp.calc_protocol_v3(corrs_R, peaks_left, peaks_right, None, pdi_data, params)
    prot_dat_L = hp.calc_protocol_v3(corrs_L, peaks_left, peaks_right, None, pdi_data, params)

    both_corr_above = ((corrs_L > args.corr_thresh) & (corrs_R > args.corr_thresh)).astype(float)
    both_prot_dat = ((prot_dat_L == 1.0) & (prot_dat_R == 1.0)).astype(float)

    # Define global dictionary of all 10 target and endpoint tuples
    ALL_POINTS_DICT = {
        # Blue points
        "pt_1_blue": {
            "target": {"mu": 3.535714, "vz": 1.115942},
            "endpoint": {"mu": 3.033482, "vz": 0.6289855},
            "color": "blue"
        },
        "pt_2_blue": {
            "target": {"mu": 2.852679, "vz": 0.8115942},
            "endpoint": {"mu": 2.691964, "vz": 0.6289855},
            "color": "blue"
        },
        "pt_3_blue": {
            "target": {"mu": 1.727679, "vz": 0.4869565},
            "endpoint": {"mu": 2.209821, "vz": 1.014483},
            "color": "blue"
        },
        "pt_4_blue": {
            "target": {"mu": 1.627232, "vz": 0.8521739},
            "endpoint": {"mu": 1.446429, "vz": 0.6289855},
            "color": "blue"
        },
        # Red points
        "pt_5_red": {
            "target": {"mu": 2.691964, "vz": 0.6289855},
            "endpoint": {"mu": 2.852679, "vz": 0.8115942},
            "color": "red"
        },
        "pt_6_red": {
            "target": {"mu": 1.446429, "vz": 0.6289855},
            "endpoint": {"mu": 1.627232, "vz": 0.8521739},
            "color": "red"
        },
        "pt_7_red": {
            "target": {"mu": 2.651786, "vz": 0.6086957},
            "endpoint": {"mu": 2.491071, "vz": 0.426087},
            "color": "red"
        },
        "pt_8_red": {
            "target": {"mu": 1.466518, "vz": 0.6086957},
            "endpoint": {"mu": 1.305804, "vz": 0.426087},
            "color": "red"
        },
        # Purple points
        "pt_9_purple": {
            "target": {"mu": 3.033482, "vz": 0.6289855},
            "endpoint": {"mu": 3.535714, "vz": 1.115942},
            "color": "purple"
        },
        "pt_10_purple": {
            "target": {"mu": 2.209821, "vz": 1.014483},
            "endpoint": {"mu": 1.727679, "vz": 0.4869565},
            "color": "purple"
        }
    }

    SUBSET_POINT_KEYS = [
        "pt_1_blue",   # Vz1116_mu3536
        "pt_2_blue",   # Vz0812_mu2853
        "pt_3_blue",   # Vz0487_mu1728
        "pt_4_blue",   # Vz0852_mu1627
        "pt_6_red",    # Vz0629_mu1446
        "pt_8_red",    # Vz0609_mu1467
        "pt_10_purple" # Vz1014_mu2210
    ]
    if single_points:
        ACTIVE_POINTS_DICT = {}
        for pt_name, pt_info in ALL_POINTS_DICT.items():
            for sp in single_points:
                if np.isclose(pt_info["target"]["vz"], sp[0], atol=1e-4) and np.isclose(pt_info["target"]["mu"], sp[1], atol=1e-4):
                    ACTIVE_POINTS_DICT[pt_name] = pt_info
    else:
        ACTIVE_POINTS_DICT = {k: ALL_POINTS_DICT[k] for k in SUBSET_POINT_KEYS}

    single_points_dir = dirname / "Plots" / "single_points"
    single_points_dir.mkdir(parents=True, exist_ok=True)

    # Extract required parameters for single point calculations
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

    try:
        d_mu = float(params_config['mu_dist'])
    except Exception:
        unique_mus = np.unique(mu)
        d_mu = float(np.min(np.diff(unique_mus))) if len(unique_mus) > 1 else 0.02

    try:
        d_vz = float(params_config['Vz_dist'])
    except Exception:
        unique_vzs = np.unique(V_z)
        d_vz = float(np.min(np.diff(unique_vzs))) if len(unique_vzs) > 1 else 0.02

    if args.skip_recalc:
        barrier_arr_path = dirname / "barrier_arr.npy"
        if barrier_arr_path.exists():
            barrier_sweep = np.load(barrier_arr_path, allow_pickle=True)
            num_sweep_points = len(barrier_sweep)
        else:
            print(f"Warning: {barrier_arr_path} not found. Using default linspace for barrier_sweep.")
            num_sweep_points = brcl_all.shape[1] if brcl_all.ndim > 1 else 101
            barrier_sweep = np.linspace(-70 * barrier_l, 70 * barrier_l, num_sweep_points)
    else:
        num_sweep_points = 100
        barrier_sweep = np.linspace(-70 * barrier_l, 70 * barrier_l, num_sweep_points)

    energies = np.linspace(-0.5, 0.5, 101)

    for pt_name, pt_info in ACTIVE_POINTS_DICT.items():
        print(f"\n==========================================")
        print(f"Processing Point: {pt_name}")
        print(f"==========================================")
        target_mu = float(pt_info["target"]["mu"])
        target_vz = float(pt_info["target"]["vz"])
        folder_name = f"Vz{target_vz:.3f}".replace('.', '') + "_" + f"mu{target_mu:.3f}".replace('.', '')
        pt_dir = single_points_dir / folder_name
        pt_dir.mkdir(parents=True, exist_ok=True)

        # ==========================================
        # 1D Cut Analysis along endpoint vector (if endpoint exists)
        # ==========================================
        if "endpoint" in pt_info and pt_info["endpoint"] is not None:
            end_mu = float(pt_info["endpoint"]["mu"])
            end_vz = float(pt_info["endpoint"]["vz"])
            print(f"Endpoint detected: ({end_mu:.4f}, {end_vz:.4f}). Performing 1D cut along line...")

            mu_min_b, mu_max_b = np.min(mu), np.max(mu)
            # Limit cut along V_z to start at minimum V_z = 0.2
            vz_min_b, vz_max_b = max(float(np.min(V_z)), 0.2), float(np.max(V_z))

            P0 = np.array([target_mu, target_vz], dtype=float)
            diff_v = np.array([end_mu - target_mu, end_vz - target_vz], dtype=float)
            dist_v = np.linalg.norm(diff_v)
            if dist_v > 1e-12:
                u_vec = diff_v / dist_v
                t_cands_mu = []
                if abs(u_vec[0]) > 1e-12:
                    t_cands_mu = [(mu_min_b - P0[0])/u_vec[0], (mu_max_b - P0[0])/u_vec[0]]
                t_cands_vz = []
                if abs(u_vec[1]) > 1e-12:
                    t_cands_vz = [(vz_min_b - P0[1])/u_vec[1], (vz_max_b - P0[1])/u_vec[1]]
                
                t_mu = sorted(t_cands_mu) if t_cands_mu else ([-np.inf, np.inf] if mu_min_b <= P0[0] <= mu_max_b else [0.0, 0.0])
                t_vz = sorted(t_cands_vz) if t_cands_vz else ([-np.inf, np.inf] if vz_min_b <= P0[1] <= vz_max_b else [0.0, 0.0])

                t_min = max(t_mu[0], t_vz[0])
                t_max = min(t_mu[1], t_vz[1])
                if t_min < t_max:
                    P_start = P0 + t_min * u_vec
                    P_end = P0 + t_max * u_vec
                else:
                    P_start, P_end = P0, P0 + diff_v
            else:
                P_start, P_end = P0, P0

            # Ensure P_start has smaller V_z so that V_z is monotonically increasing along the cut from left to right
            if P_start[1] > P_end[1]:
                P_start, P_end = P_end, P_start
            elif P_start[1] == P_end[1] and P_start[0] > P_end[0]:
                P_start, P_end = P_end, P_start

            mu_start_ext, vz_start_ext = P_start[0], P_start[1]
            mu_end_ext, vz_end_ext = P_end[0], P_end[1]

            print(f"Generating point path along cut: mu=({mu_start_ext:.4f} -> {mu_end_ext:.4f}), Vz=({vz_start_ext:.4f} -> {vz_end_ext:.4f})...")
            pts, sampled_indices = generate_point_path(pdi_data, args.N, args.resl, mu_start_ext, mu_end_ext, vz_start_ext, vz_end_ext)

            if len(pts) > 0:
                # Filter to enforce V_z >= 0.2 and sort monotonically increasing by V_z
                mask_vz = pts[:, 1] >= 0.2 - 1e-6
                pts = pts[mask_vz]
                sampled_indices = sampled_indices[mask_vz]

                if len(pts) > 1:
                    if abs(pts[-1, 1] - pts[0, 1]) > 1e-6:
                        sort_idx = np.argsort(pts[:, 1])
                    else:
                        sort_idx = np.argsort(pts[:, 0])
                    pts = pts[sort_idx]
                    sampled_indices = sampled_indices[sort_idx]

                actual_N = len(pts)
                print(f"Sampled {actual_N} points along the path after filtering for V_z >= 0.2 and sorting monotonically.")

                if actual_N > 0:
                    vz_values = pts[:, 1]
                    closest_vz_idx = int(np.argmin(np.linalg.norm(pts - np.array([target_mu, target_vz]), axis=1)))
                    target_mu_val = pts[closest_vz_idx, 0]
                    target_vz_pt = vz_values[closest_vz_idx]
                    target_mu_pt = target_mu_val

                    # Helper for saving 2D overlay plots inside pt_dir
                    def save_2d_overlay_plot(indicator_dat, label_positive, out_path, title_text):
                        unique_mu_g, unique_vz_g, rgba = create_overlay_rgba(mu, V_z, I, indicator_dat)
                        fig_overlay, ax_overlay = plt.subplots(figsize=(6, 8), dpi=150)
                        ax_overlay.imshow(rgba, origin='lower', extent=[unique_vz_g[0], unique_vz_g[-1], unique_mu_g[0], unique_mu_g[-1]], aspect='auto')

                        ax_overlay.plot([vz_start_ext, vz_end_ext], [mu_start_ext, mu_end_ext], color='black', linestyle='-', linewidth=2, zorder=4)
                        ax_overlay.scatter(target_vz_pt, target_mu_pt, color='black', edgecolor='white', s=40, zorder=5)

                        ax_overlay.set_title(title_text, fontsize=12, pad=15)
                        ax_overlay.set_xlabel(r'Zeeman Field $V_z$', fontsize=11)
                        ax_overlay.set_ylabel(r'Chemical Potential $\mu$', fontsize=11)
                        ax_overlay.set_xlim(0.0, 1.2)
                        ax_overlay.set_ylim(0.0, 4.5)
                        ax_overlay.spines['top'].set_visible(False)
                        ax_overlay.spines['right'].set_visible(False)

                        red_patch = mpatches.Patch(color=(1.0, 0.5, 0.5), label=label_positive)
                        blue_patch = mpatches.Patch(color=(0.5, 0.5, 1.0), label='Topological (I = 1)')
                        purple_patch = mpatches.Patch(color=(0.6, 0.25, 0.6), label='Overlap (Both = 1)')
                        
                        cut_line_handle = mlines.Line2D([], [], color='black', linestyle='-', label='1D Cut Path')
                        target_pt_handle = mlines.Line2D([], [], color='black', marker='o', markerfacecolor='black', markeredgecolor='white', markersize=6, linestyle='None', label=rf'Target $V_z = {target_vz_pt:.4f}$')
                        
                        ax_overlay.legend(handles=[red_patch, blue_patch, purple_patch, cut_line_handle, target_pt_handle], bbox_to_anchor=(0.5, -0.15),
                                          loc='upper center', ncol=2, fontsize=9, frameon=True)

                        fig_overlay.tight_layout()
                        fig_overlay.subplots_adjust(bottom=0.22)
                        fig_overlay.savefig(out_path, dpi=300, bbox_inches='tight')
                        plt.close(fig_overlay)
                        print(f"Saved 2D overlay plot to: {out_path}")

                    print("Generating 2D overlay plots for Right, Left, and Both sweeps...")
                    save_2d_overlay_plot(prot_dat_R, 'Right Sweep Protocol Positive', pt_dir / "overlap_plot_R.png", 'Topological Region vs Right Sweep Protocol Overlay')
                    save_2d_overlay_plot(prot_dat_L, 'Left Sweep Protocol Positive', pt_dir / "overlap_plot_L.png", 'Topological Region vs Left Sweep Protocol Overlay')
                    save_2d_overlay_plot(both_prot_dat, 'Both Sweeps Protocol Positive', pt_dir / "overlap_both_prot.png", 'Topological Region vs Both Sweeps Protocol Overlay')
                    save_2d_overlay_plot(prot_dat_R, 'Protocol Positive (prot_dat_R = 1)', pt_dir / "overlap_plot.png", 'Topological Region vs Right Sweep Protocol Overlay')

                    # Slice path-specific data
                    gap = gap_all[sampled_indices]
                    pl = peaks_left[sampled_indices]
                    pr = peaks_right[sampled_indices]
                    brcl = brcl_all[sampled_indices]
                    brcr = brcr_all[sampled_indices]
                    blcl = blcl_all[sampled_indices]
                    blcr = blcr_all[sampled_indices]
                    glr_sym = glr_all[sampled_indices, 0]
                    grl_sym = grl_all[sampled_indices, 0]
                    glr_path_data = glr_all[sampled_indices]
                    grl_path_data = grl_all[sampled_indices]

                    if has_nonlocal_gaps:
                        gap_LR_cut = gap_LR_all[sampled_indices]
                        gap_RL_cut = gap_RL_all[sampled_indices]
                        gap_transport_cut = gap_transport_all[sampled_indices]
                        izero = np.argmin(np.abs(energies_all))
                        glr_zerobias_cut = dIdVs_LR_all[sampled_indices, izero]
                        grl_zerobias_cut = dIdVs_RL_all[sampled_indices, izero]

                    evals_path = pt_dir / "cut_evals.npy"
                    pf_path = pt_dir / "cut_pf_invariants.npy"
                    corr_R_path = pt_dir / "cut_corr_R.npy"
                    corr_L_path = pt_dir / "cut_corr_L.npy"
                    corr_nl_path = pt_dir / "cut_corr_nl.npy"

                    if args.skip_recalc and evals_path.exists() and pf_path.exists() and corr_R_path.exists() and corr_L_path.exists() and corr_nl_path.exists():
                        print(f"Loading cached cut spectra and correlations from {pt_dir}...")
                        evals = np.load(evals_path)
                        pf_invariants = np.load(pf_path)
                        corr_vals_R = np.load(corr_R_path)
                        corr_vals_L = np.load(corr_L_path)
                        corr_nonlocal_vals = np.load(corr_nl_path)
                    else:
                        print(f"Calculating closed system spectra and Pfaffian invariants for {actual_N} points along cut...")
                        evals = np.zeros(shape=(actual_N, args.kvals))
                        pf_invariants = np.zeros(actual_N)

                        for i in range(actual_N):
                            mu_val, vz_val = pts[i]
                            scl = hp.build_system_closed(t, mu_val, gamma, Delta0, vz_val, alpha, Ls, Vdisx, a=1)
                            evals[i, :] = hp.calc_spectrum(scl, k=args.kvals)
                            pf_invariants[i] = hp.cal_pfaffian_invariant(
                                ts=t, alphas=alpha, gamma=gamma, delta0=Delta0, Nx=Ls, Vdisx=Vdisx, V0=1.0, gm=vz_val, mu=mu_val
                            )

                        corr_vals_R = np.zeros(actual_N)
                        corr_vals_L = np.zeros(actual_N)
                        corr_nonlocal_vals = np.zeros(actual_N)
                        for i in range(actual_N):
                            corr_vals_R[i] = hp.calc_correlation(brcl[i], brcr[i], barrier_arr=barrier_arr)
                            corr_vals_L[i] = hp.calc_correlation(blcl[i], blcr[i], barrier_arr=barrier_arr)
                            corr_nonlocal_vals[i] = hp.calc_correlation(glr_path_data[i], grl_path_data[i], barrier_arr=barrier_arr)

                        np.save(evals_path, evals)
                        np.save(pf_path, pf_invariants)
                        np.save(corr_R_path, corr_vals_R)
                        np.save(corr_L_path, corr_vals_L)
                        np.save(corr_nl_path, corr_nonlocal_vals)

                    has_both_L = pl[:, 1]
                    width_L = np.where(has_both_L == 1.0, pl[:, 2] - pl[:, 4], np.nan)
                    height_L = pl[:, 3]

                    has_both_R = pr[:, 1]
                    width_R = np.where(has_both_R == 1.0, pr[:, 2] - pr[:, 4], np.nan)
                    height_R = pr[:, 3]

                    def add_common_elements(ax, prot_data_array=None):
                        if prot_data_array is None:
                            prot_data_array = prot_dat_R

                        topo_added = False
                        prot_added = False
                        overlap_added = False

                        for idx in range(actual_N):
                            is_topo = (I[sampled_indices[idx]] == 1.0)
                            is_prot = (prot_data_array[sampled_indices[idx]] == 1.0)
                            
                            if is_topo and is_prot:
                                ax.axvspan(idx - 0.5, idx + 0.5, color=(0.6, 0.25, 0.6), alpha=0.3, zorder=0,
                                           label="Pfaffian & Protocol" if not overlap_added else "")
                                overlap_added = True
                            elif is_topo:
                                ax.axvspan(idx - 0.5, idx + 0.5, color=(0.5, 0.5, 1.0), alpha=0.3, zorder=0,
                                           label="Pfaffian Only" if not topo_added else "")
                                topo_added = True
                            elif is_prot:
                                ax.axvspan(idx - 0.5, idx + 0.5, color=(1.0, 0.5, 0.5), alpha=0.3, zorder=0,
                                           label="Protocol Only" if not prot_added else "")
                                prot_added = True

                        ax.axvline(closest_vz_idx, color='black', linestyle='-', linewidth=1.5, zorder=1,
                                   label=rf"Target $V_z = {vz_values[closest_vz_idx]:.4f}$ ($\mu = {target_mu_val:.4f}$)")

                        ax.grid(True, linestyle=':', alpha=0.5, zorder=1)
                        ax.set_xlim(-0.5, actual_N - 0.5)

                        num_ticks = min(6, actual_N)
                        tick_indices = np.round(np.linspace(0, actual_N - 1, num_ticks)).astype(int)
                        tick_indices = np.unique(tick_indices)

                        # Primary X-Axis (mu)
                        ax.set_xticks(tick_indices)
                        ax.set_xticklabels([f"{pts[idx, 0]:.3f}" for idx in tick_indices], rotation=45, ha='right', fontsize=9)
                        ax.set_xlabel(r"Chemical Potential $\mu$ (meV)", fontsize=11, labelpad=15)

                        # Secondary X-Axis (V_z) monotonically increasing
                        ax2 = ax.twiny()
                        ax2.spines['bottom'].set_position(('outward', 65))
                        ax2.xaxis.set_ticks_position('bottom')
                        ax2.xaxis.set_label_position('bottom')

                        ax2.set_xlim(ax.get_xlim())
                        ax2.set_xticks(tick_indices)
                        ax2.set_xticklabels([f"{pts[idx, 1]:.3f}" for idx in tick_indices], rotation=45, ha='right', fontsize=9)
                        ax2.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11, labelpad=15)

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

                    # --- Plot 2A: Right Sweep Local Conductance Correlation ---
                    fig_corr_R, ax_corr_R = plt.subplots(figsize=(10, 7.5), dpi=150)
                    add_common_elements(ax_corr_R, prot_data_array=prot_dat_R)
                    ax_corr_R.plot(range(actual_N), corr_vals_R, '-', color='forestgreen', linewidth=2.5, label="Right Sweep Conductance Correlation", zorder=3)
                    ax_corr_R.axhline(args.corr_thresh, color='black', linestyle=':', linewidth=1.5, label=f"Threshold ({args.corr_thresh})", zorder=2)
                    ax_corr_R.set_ylabel("Correlation", fontsize=12)
                    ax_corr_R.set_title("Right Barrier Sweep Conductance Correlation along Cut", fontsize=13)
                    handles, labels = ax_corr_R.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax_corr_R.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=3, fontsize=10)

                    # --- Plot 2C: Combined Both Sweeps Conductance Correlation ---
                    fig_corr_both, ax_corr_both = plt.subplots(figsize=(10, 7.5), dpi=150)
                    add_common_elements(ax_corr_both, prot_data_array=both_prot_dat)
                    ax_corr_both.plot(range(actual_N), corr_vals_R, '-', color='forestgreen', linewidth=2.5, label="Right Sweep Conductance Correlation", zorder=3)
                    ax_corr_both.plot(range(actual_N), corr_vals_L, '-', color='dodgerblue', linewidth=2.0, label="Left Sweep Conductance Correlation", zorder=3)
                    ax_corr_both.axhline(args.corr_thresh, color='red', linestyle='--', linewidth=1.5, label=f"Threshold ({args.corr_thresh})", zorder=3)

                    ax_corr_both.set_ylabel("Correlation", fontsize=12)
                    ax_corr_both.set_title("Both Barrier Sweeps Conductance Correlation along Cut", fontsize=13)
                    handles, labels = ax_corr_both.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax_corr_both.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=3, fontsize=10)

                    # --- Plot 3: Effective Topological Gap and Transport Gap ---
                    fig_gap, ax_gap = plt.subplots(figsize=(10, 7.5), dpi=150)
                    ax2_gap = add_common_elements(ax_gap)
                    ax2_gap.spines['right'].set_visible(False)
                    
                    mid_idx = args.kvals // 2
                    effective_gap_cut = evals[:, mid_idx + 1] - evals[:, mid_idx]
                    ax_gap.plot(range(actual_N), effective_gap_cut, '-', color='darkorange', linewidth=2.5, label="Effective Topological Gap", zorder=3)
                    if has_nonlocal_gaps:
                        ax_gap.plot(range(actual_N), gap_transport_cut, '-', color='forestgreen', linewidth=2.5, label=r"Transport Gap ($\Delta_{transport}$)", zorder=3)
                        ax_gap.plot(range(actual_N), gap_LR_cut, ':', color='dodgerblue', linewidth=1.5, alpha=0.8, label=r"$\Delta_{LR}$", zorder=3)
                        ax_gap.plot(range(actual_N), gap_RL_cut, ':', color='brown', linewidth=1.5, alpha=0.8, label=r"$\Delta_{RL}$", zorder=3)
                    ax_gap.set_ylabel("Gap (meV)", fontsize=12)
                    ax_gap.set_ylim(bottom=0.0, top=0.25)
                    ax_gap.tick_params(axis='y')
                    ax_gap.set_title("Effective Topological Gap vs Anti-Symmetrized Transport Gap along Cut", fontsize=13)
                    
                    handles_gap, labels_gap = ax_gap.get_legend_handles_labels()
                    by_label = dict(zip(labels_gap, handles_gap))
                    ax_gap.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=3, fontsize=10)

                    # --- Plot 4: Peak Width and Height ---
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

                    # Save all cut figures right into pt_dir
                    for filename, figure in [
                        ("spectra_cut.png", fig_spectra),
                        ("correlation_cut_R.png", fig_corr_R),
                        ("correlation_cut_both.png", fig_corr_both),
                        ("gap_nonlocal_cut.png", fig_gap),
                        ("peaks_cut.png", fig_peaks)
                    ]:
                        out_path = pt_dir / filename
                        figure.tight_layout()
                        figure.subplots_adjust(bottom=0.4)
                        figure.savefig(out_path, dpi=300, bbox_inches='tight')
                        plt.close(figure)
                        print(f"Saved cut plot to: {out_path}")

        # ==========================================
        # Single Point Calculations & Detailed Plotting
        # ==========================================
        print(f"Generating single point calculations for Vz={target_vz}, mu={target_mu}...")
        idx_match = None
        if args.skip_recalc:
            dist_sq = ((V_z - target_vz) / d_vz)**2 + ((mu - target_mu) / d_mu)**2
            best_idx = int(np.argmin(dist_sq))
            if abs(V_z[best_idx] - target_vz) <= d_vz + 1e-6 and abs(mu[best_idx] - target_mu) <= d_mu + 1e-6:
                idx_match = best_idx
                print(f"  [Skip-Recalc] Found existing data at index {idx_match} (Vz={V_z[best_idx]:.4f}, mu={mu[best_idx]:.4f}).")
            else:
                print(f"  [Skip-Recalc] Warning: No data point found within resolution window of (Vz={target_vz:.4f}, mu={target_mu:.4f}). Falling back to live Kwant calculation.")

        if idx_match is not None:
            cond_left_R = brcl_all[idx_match]
            cond_right_R = brcr_all[idx_match]
            cond_left_L = blcl_all[idx_match]
            cond_right_L = blcr_all[idx_match]
        else:
            print(f"  Holding Left barrier constant at non-zero UL={barrier_l:.3f} meV while sweeping Right barrier UR...")
            cond_left_R = np.zeros(num_sweep_points)
            cond_right_R = np.zeros(num_sweep_points)
            for k in tqdm(range(num_sweep_points), desc=f"Right Sweep (UL={barrier_l:.2f})"):
                syst_R = hp.build_system(
                    t=t, mu=target_mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=target_vz,
                    alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
                    barrier_l=barrier_l, barrier_r=barrier_sweep[k], Vdisx=Vdisx, a=1
                )
                cL, cR = hp.calc_conductance(syst_R, energy=0.0, return_smatrix=False)
                cond_left_R[k] = cL
                cond_right_R[k] = cR

            print(f"  Holding Right barrier constant at non-zero UR={barrier_l:.3f} meV while sweeping Left barrier UL...")
            cond_left_L = np.zeros(num_sweep_points)
            cond_right_L = np.zeros(num_sweep_points)
            for k in tqdm(range(num_sweep_points), desc=f"Left Sweep (UR={barrier_l:.2f})"):
                syst_L = hp.build_system(
                    t=t, mu=target_mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=target_vz,
                    alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
                    barrier_l=barrier_sweep[k], barrier_r=barrier_l, Vdisx=Vdisx, a=1
                )
                cL, cR = hp.calc_conductance(syst_L, energy=0.0, return_smatrix=False)
                cond_left_L[k] = cL
                cond_right_L[k] = cR

        np.save(pt_dir / "cond_left_LeftSweep.npy", cond_left_L)
        np.save(pt_dir / "cond_right_LeftSweep.npy", cond_right_L)

        x_data = barrier_sweep / (barrier_l if barrier_l != 0 else 1.0)
        idx_sym = np.argmin(np.abs(barrier_sweep - barrier_l))
        lw = 3.0

        fig_cond, axs_cond = plt.subplots(2, 2, figsize=(11, 8.5), dpi=150)
        normed_GL_L = cond_left_L / (cond_left_L[idx_sym] if cond_left_L[idx_sym] != 0 else 1.0)
        normed_GR_L = cond_right_L / (cond_right_L[idx_sym] if cond_right_L[idx_sym] != 0 else 1.0)
        normed_GL_R = cond_left_R / (cond_left_R[idx_sym] if cond_left_R[idx_sym] != 0 else 1.0)
        normed_GR_R = cond_right_R / (cond_right_R[idx_sym] if cond_right_R[idx_sym] != 0 else 1.0)

        # [0, 0] Left Sweep G_LL
        axs_cond[0, 0].plot(x_data, normed_GL_L, color="green", linewidth=lw)
        axs_cond[0, 0].set_xlabel(r"$U_{L}/U_{R}$", fontsize=12)
        axs_cond[0, 0].set_ylabel(r"$G_{LL}/G_{LL, sym}$", fontsize=12)
        axs_cond[0, 0].set_title(r"Left Sweep $G_{LL}$")
        axs_cond[0, 0].grid(True, linestyle=':', alpha=0.5)

        # [0, 1] Left Sweep G_RR
        axs_cond[0, 1].plot(x_data, normed_GR_L, color="green", linewidth=lw)
        axs_cond[0, 1].set_xlabel(r"$U_{L}/U_{R}$", fontsize=12)
        axs_cond[0, 1].set_ylabel(r"$G_{RR}/G_{RR, sym}$", fontsize=12)
        axs_cond[0, 1].set_title(r"Left Sweep $G_{RR}$")
        axs_cond[0, 1].grid(True, linestyle=':', alpha=0.5)

        # [1, 0] Right Sweep G_LL
        axs_cond[1, 0].plot(x_data, normed_GL_R, color="green", linewidth=lw)
        axs_cond[1, 0].set_xlabel(r"$U_{R}/U_{L}$", fontsize=12)
        axs_cond[1, 0].set_ylabel(r"$G_{LL}/G_{LL, sym}$", fontsize=12)
        axs_cond[1, 0].set_title(r"Right Sweep $G_{LL}$")
        axs_cond[1, 0].grid(True, linestyle=':', alpha=0.5)

        # [1, 1] Right Sweep G_RR
        axs_cond[1, 1].plot(x_data, normed_GR_R, color="green", linewidth=lw)
        axs_cond[1, 1].set_xlabel(r"$U_{R}/U_{L}$", fontsize=12)
        axs_cond[1, 1].set_ylabel(r"$G_{RR}/G_{RR, sym}$", fontsize=12)
        axs_cond[1, 1].set_title(r"Right Sweep $G_{RR}$")
        axs_cond[1, 1].grid(True, linestyle=':', alpha=0.5)

        fig_cond.suptitle(rf"Normalized Conductance Sweeps ($V_z={target_vz:.3f}, \mu={target_mu:.3f}$)", fontsize=14, y=0.98)
        fig_cond.tight_layout()
        fig_cond.subplots_adjust(top=0.91)
        fig_cond.savefig(pt_dir / "Conductances_Combined.png", dpi=300, bbox_inches='tight')
        plt.close(fig_cond)

        # dIdV Spectrum at Nominal Barrier
        skip_dIdV = False
        if idx_match is not None:
            dIdV_left_path = dirname / "dIdVs_left_arr.npy"
            dIdV_right_path = dirname / "dIdVs_right_arr.npy"
            energies_path = dirname / "energies.npy"
            if dIdV_left_path.exists() and dIdV_right_path.exists() and energies_path.exists():
                try:
                    dIdVs_L_all = np.load(dIdV_left_path, mmap_mode='r')
                    dIdVs_R_all = np.load(dIdV_right_path, mmap_mode='r')
                    energies_dIdV = np.load(energies_path)
                    dIdV_left = dIdVs_L_all[idx_match]
                    dIdV_right = dIdVs_R_all[idx_match]
                    skip_dIdV = True
                except Exception:
                    skip_dIdV = False

        if not skip_dIdV:
            syst_nom = hp.build_system(
                t=t, mu=target_mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=target_vz,
                alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
                barrier_l=barrier_l, barrier_r=barrier_l, Vdisx=Vdisx, a=1
            )
            dIdV_left, dIdV_right, _, _, _ = hp.calc_dIdV(syst_nom, energies)
            energies_dIdV = energies

        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        ax.plot(energies_dIdV, dIdV_left, color='royalblue', linewidth=2.0, label='Left dI/dV')
        ax.plot(energies_dIdV, dIdV_right, color='darkorange', linewidth=2.0, label='Right dI/dV')
        ax.set_xlabel("Energy (meV)", fontsize=13)
        ax.set_ylabel("Differential Conductance (dI/dV)", fontsize=13)
        ax.set_title(rf"dI/dV Spectrum ($V_z={target_vz:.3f}, \mu={target_mu:.3f}$)")
        ax.legend(fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.5)
        fig.tight_layout()
        fig.savefig(pt_dir / "dIdV.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # MZM Wavefunction
        scl = hp.build_system_closed(t, target_mu, gamma, Delta0, target_vz, alpha, Ls, Vdisx, a=1)
        evals_cl, evecs_cl = hp.solve_ham(scl, k=2)
        rho_M1, rho_M2, _ = hp.get_psiM_density(evals_cl, evecs_cl)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.plot(rho_M1, label='Majorana Left (M1)', color='cyan', linewidth=2.0)
        ax.plot(rho_M2, label='Majorana Right (M2)', color='orange', linewidth=2.0)
        ax.set_xlabel("Site Index", fontsize=13)
        ax.set_ylabel("Probability Density", fontsize=13)
        ax.set_title(rf"Majorana Modes ($V_z={target_vz:.3f}, \mu={target_mu:.3f}$)")
        ax.legend(fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.5)
        fig.tight_layout()
        fig.savefig(pt_dir / "wavefunction.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ==========================================
    # Global 2D Single Point Maps (point_map_R, point_map_L, point_map_both)
    # ==========================================
    print("\nGenerating global 2D single point maps (point_map_R.png, point_map_L.png, point_map_both.png)...")
    pts_array = np.array([[pt["target"]["vz"], pt["target"]["mu"]] for pt in ACTIVE_POINTS_DICT.values()], dtype=float)

    def save_single_points_map(prot_indicator, title_str, filename_str, label_pos):
        unique_mu_grid, unique_vz_grid, rgba_map = create_overlay_rgba(mu, V_z, I, prot_indicator)
        fig_map, ax_map = plt.subplots(figsize=(6, 8), dpi=150)
        ax_map.imshow(rgba_map, origin='lower', extent=[unique_vz_grid[0], unique_vz_grid[-1], unique_mu_grid[0], unique_mu_grid[-1]], aspect='auto')
        if len(pts_array) > 0:
            ax_map.scatter(pts_array[:, 0], pts_array[:, 1], color='black', edgecolor='white', s=50, zorder=5)

        ax_map.set_title(title_str, fontsize=12, pad=15)
        ax_map.set_xlabel(r'Zeeman Field $V_z$', fontsize=11)
        ax_map.set_ylabel(r'Chemical Potential $\mu$', fontsize=11)
        ax_map.set_xlim(0.0, 1.2)
        ax_map.set_ylim(0.0, 4.5)
        ax_map.spines['top'].set_visible(False)
        ax_map.spines['right'].set_visible(False)

        red_patch = mpatches.Patch(color=(1.0, 0.5, 0.5), label=label_pos)
        blue_patch = mpatches.Patch(color=(0.5, 0.5, 1.0), label='Topological (I = 1)')
        purple_patch = mpatches.Patch(color=(0.6, 0.25, 0.6), label='Overlap (Both = 1)')
        pts_handle = mlines.Line2D([], [], color='black', marker='o', markerfacecolor='black', markeredgecolor='white', markersize=7, linestyle='None', label='Single Points')

        ax_map.legend(handles=[red_patch, blue_patch, purple_patch, pts_handle], bbox_to_anchor=(0.5, -0.15),
                      loc='upper center', ncol=2, fontsize=9, frameon=True)

        fig_map.tight_layout()
        fig_map.subplots_adjust(bottom=0.22)
        map_out = single_points_dir / filename_str
        fig_map.savefig(map_out, dpi=300, bbox_inches='tight')
        plt.close(fig_map)
        print(f"Saved point map to: {map_out}")

    save_single_points_map(prot_dat_R, 'Topological Region vs Right Sweep Protocol (Single Points)', 'point_map_R.png', 'Right Protocol Positive')
    save_single_points_map(prot_dat_L, 'Topological Region vs Left Sweep Protocol (Single Points)', 'point_map_L.png', 'Left Protocol Positive')
    save_single_points_map(both_prot_dat, 'Topological Region vs Both Sweeps Protocol (Single Points)', 'point_map_both.png', 'Both Protocol Positive')

    # ==========================================
    # Global 2D Continuous Phase Maps (Effective Gap, Transport Gap, Localizations)
    # ==========================================
    generate_continuous_phase_maps(dirname, mu, V_z, I, eff_gap, gap_transport_all, site_localizations, weight_localization_arr, curvature_arr)


def main():
    parser = argparse.ArgumentParser(description="Run full post-processing pipeline (2D overlays and 1D parameter cuts).")

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
    parser.add_argument("--output", type=str, default="Plots/Tdis_pfaff4/spectra_cut.png", help="Path to save the spectra plot.")
    parser.add_argument("--corr-output", type=str, default="Plots/Tdis_pfaff4/correlation_cut.png", help="Path to save the correlation plot.")
    parser.add_argument("--gap-output", type=str, default="Plots/Tdis_pfaff4/gap_nonlocal_cut.png", help="Path to save the gap and nonlocal conductance plot.")
    parser.add_argument("--peaks-output", type=str, default="Plots/Tdis_pfaff4/peaks_cut.png", help="Path to save the peak width and height plot.")
    parser.add_argument("--overlap-output", type=str, default="Plots/Tdis_pfaff4/overlap_plot.png", help="Path to save the 2D overlay plot.")
    parser.add_argument("--single-points", type=str, default="[]", help="List of (V_z, mu) tuples for single point plotting, e.g. '[(0.45323, 3.4564)]'")
    parser.add_argument("--skip-recalc", action="store_true", help="Skip live Kwant recalculation for single points and pull from pre-computed data within parameter grid resolution.")
    parser.add_argument("--epsilon", type=float, default=0.005, help="Resolution threshold to define gapless points.")
    parser.add_argument("--curvature-epsilon", type=float, default=0.02, help="Spacing step size for finite difference curvature (0.02, 0.04, or 0.06).")
    parser.add_argument("--cond-threshold", type=float, default=1e-9, help="Conductance threshold for finite zero-bias check.")
    parser.add_argument("--process-all-realizations", action="store_true", help="Automatically run post-processing pipeline across all disorder realizations (disorder_realization_0..9_results and Tdis_pfaff4).")
    parser.add_argument("--phase-maps-only", action="store_true", help="Skip single-point cut and map calculations and only generate continuous 2D phase maps.")
    parser.add_argument("--skip-single-points", action="store_true", help="Alias for --phase-maps-only.")
    args = parser.parse_args()

    try:
        single_points = ast.literal_eval(args.single_points)
        if not isinstance(single_points, list):
            single_points = []
    except Exception as e:
        print(f"Warning: Could not parse --single-points ({e}). Defaulting to empty list.")
        single_points = []

    if args.process_all_realizations:
        base_root = Path(PathConfigs.ROOT) if not os.path.isabs(args.dirname) else Path(args.dirname).parent.parent
        data_dir = base_root / "Data"
        all_dirs = sorted(list(data_dir.glob("disorder_realization_*_results"))) if data_dir.exists() else []
        pfaff4_dir = data_dir / "Tdis_pfaff4"
        if pfaff4_dir.exists():
            all_dirs.append(pfaff4_dir)
        non_int_dir = data_dir / "non_interacting_results_local"
        if non_int_dir.exists():
            all_dirs.append(non_int_dir)

        print(f"Found {len(all_dirs)} directories to process: {[d.name for d in all_dirs]}")
        for d in all_dirs:
            process_single_directory(d, args, single_points)
    else:
        if os.path.isabs(args.dirname):
            dirname = Path(args.dirname)
        else:
            dirname = Path(PathConfigs.ROOT) / args.dirname
            if not dirname.exists():
                dirname = Path(args.dirname)
        process_single_directory(dirname, args, single_points)

if __name__ == "__main__":
    main()
