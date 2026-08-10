import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import argparse
import json
from pathlib import Path
import os
import helpers as hp

def generate_continuous_phase_map(val_grid, unique_mu, unique_vz, title_str, filename_str, cbar_label, cmap='viridis', vmin=None, vmax=None, I_grid=None, I_color='#404040', I_alpha=0.55, dark_theme=False):
    if dark_theme:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
        
    fig, ax = plt.subplots(figsize=(7, 8), dpi=150)
    im = ax.imshow(val_grid, origin='lower', extent=[unique_vz[0], unique_vz[-1], unique_mu[0], unique_mu[-1]], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, pad=0.03, extend='max' if vmax is not None else 'neither')
    cbar.set_label(cbar_label, fontsize=11)
    
    if I_grid is not None and np.any(I_grid == 1) and np.any(I_grid == 0):
        VZ, MU = np.meshgrid(unique_vz, unique_mu)
        ax.contourf(VZ, MU, I_grid, levels=[0.5, 1.5], colors=[I_color], alpha=I_alpha)
        ax.contour(VZ, MU, I_grid, levels=[0.5], colors=['#202020' if not dark_theme else '#ffffff'], linewidths=1.5, alpha=0.9)
        patch = mpatches.Patch(facecolor=I_color, edgecolor='#202020' if not dark_theme else '#ffffff', linewidth=1.5, alpha=I_alpha, label='Topological Phase')
        ax.legend(handles=[patch], loc='upper right', fontsize=9, frameon=True)
        
    ax.set_title(title_str, fontsize=12, pad=15)
    ax.set_xlabel(r'Zeeman Field $V_z$', fontsize=11)
    ax.set_ylabel(r'Chemical Potential $\mu$', fontsize=11)
    ax.set_xlim(unique_vz[0], unique_vz[-1])
    ax.set_ylim(unique_mu[0], unique_mu[-1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(filename_str, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.style.use('default')

def main():
    parser = argparse.ArgumentParser(description="TGP Post-Processing Pipeline")
    parser.add_argument("--dirname", type=str, required=True, help="Directory name containing the results (e.g. Data/Tdis_pfaff4).")
    args = parser.parse_args()

    dirname = Path(args.dirname)
    if not dirname.is_absolute():
        dirname = Path("Data") / args.dirname
        if not dirname.exists():
            dirname = Path(args.dirname)
            
    print(f"Processing TGP for {dirname}...")
    
    # Load parameters
    params_list = np.load(dirname / "params_list.npy", allow_pickle=True)
    mu = params_list[:, 1]
    V_z = params_list[:, 2]
    unique_mu = np.unique(mu)
    unique_vz = np.unique(V_z)
    Nmu = len(unique_mu)
    Nvz = len(unique_vz)
    
    mu_to_idx = {val: idx for idx, val in enumerate(unique_mu)}
    vz_to_idx = {val: idx for idx, val in enumerate(unique_vz)}
    
    def to_grid(val_1d):
        grid = np.full((Nmu, Nvz), np.nan)
        for i in range(len(mu)):
            mu_i = mu_to_idx.get(mu[i], -1)
            vz_j = vz_to_idx.get(V_z[i], -1)
            if mu_i != -1 and vz_j != -1:
                grid[mu_i, vz_j] = val_1d[i]
        return grid
        
    # Load TGP arrays
    tgp_l = np.load(dirname / "tgp_stage1_dIdVl.npy")  # (N, 5, 5, 7)
    tgp_r = np.load(dirname / "tgp_stage1_dIdVr.npy")  # (N, 5, 5, 7)
    
    bar_arr_path = dirname / "barrier_arr.npy"
    if bar_arr_path.exists():
        barrier_arr = np.load(bar_arr_path)
        bar_L_GL = np.load(dirname / "barrier_left_conductance_left_arr.npy")
        bar_L_GR = np.load(dirname / "barrier_left_conductance_right_arr.npy")
        bar_R_GL = np.load(dirname / "barrier_right_conductance_left_arr.npy")
        bar_R_GR = np.load(dirname / "barrier_right_conductance_right_arr.npy")
    else:
        barrier_arr = None
    
    # E0 and E1 (topological gap)
    mp_eng_arr_path = dirname / "mp_eng_arr.npy"
    if mp_eng_arr_path.exists():
        E0 = np.load(mp_eng_arr_path)
    else:
        spectrum_arr = np.load(dirname / "spectrum_arr.npy")
        E0 = np.abs(spectrum_arr[:, 2])
        
    topological_gap_path = dirname / "topological_gap.npy"
    E1 = np.load(topological_gap_path) if topological_gap_path.exists() else None
    
    # Weight localization
    weight_loc = np.load(dirname / "weight_localization_arr.npy")
    
    # Pfaffian invariant
    pfaffian_grid = None
    pdi_data_path = dirname / "pdi_data.npy"
    if pdi_data_path.exists():
        pdi_data = np.load(pdi_data_path)
        if pdi_data.shape[1] >= 4:
            pfaffian_grid = to_grid(pdi_data[:, 3])
    
    # Transport gap
    gap_transport_path = dirname / "gap_transport_all.npy"
    dIdVs_LR = np.load(dirname / "dIdVs_LR.npy")
    dIdVs_RL = np.load(dirname / "dIdVs_RL.npy")
    energies = np.load(dirname / "energies.npy")
    gap_LR = np.zeros(len(mu))
    gap_RL = np.zeros(len(mu))
    
    for m in unique_mu:
        idx_m = np.where(mu == m)[0]
        
        # Left-to-Right gap
        g_LR_m = dIdVs_LR[idx_m]
        gap_LR_m, _, _ = hp.extract_nonlocal_gap(energies, g_LR_m, global_threshold=True)
        gap_LR[idx_m] = gap_LR_m
        
        # Right-to-Left gap
        g_RL_m = dIdVs_RL[idx_m]
        gap_RL_m, _, _ = hp.extract_nonlocal_gap(energies, g_RL_m, global_threshold=True)
        gap_RL[idx_m] = gap_RL_m
        
    gap_transport_all = hp.nanmin_gap_combination(gap_LR, gap_RL)
    np.save(gap_transport_path, np.column_stack([gap_LR, gap_RL, gap_transport_all]))

    # Setup directories
    plots_dir = dirname / "Plots"
    stage1_dir = plots_dir / "Stage1"
    stage2_dir = plots_dir / "Stage2"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)
    
    resolutions = [0.1, 0.05, 0.02]
    # Energy array: [-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1]
    res_steps = {0.1: 3, 0.05: 2, 0.02: 1}
    idx_zero = 3
    
    E0_grid = to_grid(E0)
    
    tp_fp_counts = {}
    
    for res in resolutions:
        # Calculate Curvature L and R using savgol_filter on the fine 11-point ZBP mesh
        # Re-implemented from Microsoft's azure-quantum-tgp (tgp/two.py: derivative_threshold)
        import scipy.signal
        
        # bias_window = 0.01 (10 uV). delta = 0.0025 (2.5 uV).
        # Microsoft window_length = max(3, (int(2 * 0.01 / 0.0025) // 2) * 2 + 1) = 9
        # savgol_filter returns the smoothed derivatives at all points, we want the center point (index 5 of 11)
        curv_L = scipy.signal.savgol_filter(tgp_l, window_length=9, polyorder=2, deriv=2, delta=0.0025, axis=-1)[:, :, :, 5]
        curv_R = scipy.signal.savgol_filter(tgp_r, window_length=9, polyorder=2, deriv=2, delta=0.0025, axis=-1)[:, :, :, 5]
        
        # ZBP condition: strictly negative curvature <= -100.0
        zbp_mask_L = (curv_L <= -100.0) # Shape: (N, 5, 5)
        zbp_mask_R = (curv_R <= -100.0)
        
        # Marginal probability logic (Microsoft Stage 2)
        # Re-implemented from Microsoft's azure-quantum-tgp (tgp/two.py: zbp_dataset_derivative)
        # using the 0.60 independent marginal passing fraction.
        prob_L = np.mean(zbp_mask_L, axis=(1, 2)) # shape (N,)
        prob_R = np.mean(zbp_mask_R, axis=(1, 2))
        
        stage1_pass = (prob_L >= 0.60) & (prob_R >= 0.60)
        stage1_pass = stage1_pass.astype(float)
        
        # Keep old stage2_pass as well for symmetric requirement (or set to stage1_pass if 1:1)
        # Microsoft only has `passed_TGP` (which is gap & ZBP probability), but we'll use stage1_pass
        stage1_grid = to_grid(stage1_pass)
        
        # Qualified Transport Gap
        gap_grid = to_grid(gap_transport_all)
        gap_grid_qual = np.where(E0_grid <= res, gap_grid, np.nan)
        
        # Topological Island Mask
        island_mask = (gap_grid > res) & (stage1_grid == 1)
        
        # Cluster Pruning (Connected Components)
        # Re-implemented from Microsoft's azure-quantum-tgp (tgp/two.py: cluster_and_score)
        # Discard any topological 'island' that contains fewer than min_samples = 3 pixels.
        import scipy.ndimage
        labels, num_features = scipy.ndimage.label(island_mask, structure=np.ones((3,3)))
        for i in range(1, num_features + 1):
            if np.sum(labels == i) < 3:
                island_mask[labels == i] = False
                
        # Update stage1_grid to reflect the pruned islands
        stage1_grid[~island_mask] = 0
        stage2_grid = stage1_grid.copy() # Match stage1 and stage2 since we unified the probability check
        
        # Plot Binary Maps
        generate_continuous_phase_map(stage1_grid, unique_mu, unique_vz, f'Stage 1 (Res {res})', stage1_dir / f'phase_map_TGP_stage1_res_{res}.png', 'Pass (1) / Fail (0)', cmap='gray', vmin=0, vmax=1)
        generate_continuous_phase_map(stage2_grid, unique_mu, unique_vz, f'Stage 2 (Res {res})', stage2_dir / f'phase_map_TGP_stage2_res_{res}.png', 'Pass (1) / Fail (0)', cmap='gray', vmin=0, vmax=1)
        
        generate_continuous_phase_map(gap_grid_qual, unique_mu, unique_vz, f'Qualified Transport Gap (Res {res})', stage2_dir / f'phase_map_qualified_transport_gap_res_{res}.png', 'Gap', cmap='viridis')
        
        # TGP Summary Phase Diagram
        summary_grid = np.zeros_like(gap_grid)
        summary_grid[(gap_grid > res) & (stage2_grid == 0)] = 1
        summary_grid[(gap_grid > res) & (stage2_grid == 1)] = 2
        summary_grid[(gap_grid <= res) & (stage2_grid == 0)] = 3
        summary_grid[(gap_grid <= res) & (stage2_grid == 1)] = 4
        
        fig, ax = plt.subplots(figsize=(7, 8), dpi=150)
        cmap_sum = mcolors.ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds, cmap_sum.N)
        im = ax.imshow(summary_grid, origin='lower', extent=[unique_vz[0], unique_vz[-1], unique_mu[0], unique_mu[-1]], aspect='auto', cmap=cmap_sum, norm=norm)
        
        cbar = fig.colorbar(im, ax=ax, pad=0.03, ticks=[1, 2, 3, 4])
        cbar.ax.set_yticklabels(['Gapped, no ZBP', 'Gapped, ZBP', 'Gapless, no ZBP', 'Gapless, ZBP'])
        
        handles_list = []
        if np.any(island_mask):
            VZ, MU = np.meshgrid(unique_vz, unique_mu)
            ax.contourf(VZ, MU, island_mask, levels=[0.5, 1.5], colors=['white'], alpha=0.3, hatches=['//'])
            ax.contour(VZ, MU, island_mask, levels=[0.5], colors=['black'], linewidths=1.5)
            patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', alpha=0.3, label='Topological Island')
            handles_list.append(patch)
            
        if pfaffian_grid is not None and np.any(pfaffian_grid > 0.5):
            VZ, MU = np.meshgrid(unique_vz, unique_mu)
            ax.contourf(VZ, MU, pfaffian_grid, levels=[0.5, 1.5], colors=['#303030'], alpha=0.6)
            patch_pfaff = mpatches.Patch(facecolor='#303030', alpha=0.6, label='True Topological (Pfaffian)')
            handles_list.append(patch_pfaff)
            
        if handles_list:
            ax.legend(handles=handles_list, loc='upper right', fontsize=9, frameon=True)
            
        ax.set_title(f'TGP Summary (Res {res})', fontsize=12)
        ax.set_xlabel(r'Zeeman Field $V_z$')
        ax.set_ylabel(r'Chemical Potential $\mu$')
        fig.tight_layout()
        fig.savefig(stage2_dir / f'tgp_summary_res_{res}.png', dpi=300)
        plt.close(fig)

        # Stage 3 Correlation Checks
        if barrier_arr is not None:
            stage3_dir = plots_dir / "Stage3"
            stage3_dir.mkdir(parents=True, exist_ok=True)
            
            corr_L_grid = np.zeros_like(gap_grid)
            corr_R_grid = np.zeros_like(gap_grid)
            
            for i in range(Nmu):
                for j in range(Nvz):
                    if island_mask[i, j]:
                        idx = i * Nvz + j
                        cL = hp.calc_correlation(bar_L_GL[idx], bar_L_GR[idx], barrier_arr)
                        cR = hp.calc_correlation(bar_R_GR[idx], bar_R_GL[idx], barrier_arr)
                        corr_L_grid[i, j] = cL
                        corr_R_grid[i, j] = cR
                        
            s3_pass_L = (corr_L_grid >= 0.90).astype(float)
            s3_pass_R = (corr_R_grid >= 0.90).astype(float)
            s3_pass_both = ((corr_L_grid >= 0.90) & (corr_R_grid >= 0.90)).astype(float)
            
            def create_stage3_summary(s3_grid, suffix):
                summary_grid_s3 = np.zeros_like(gap_grid)
                summary_grid_s3[(gap_grid > res) & (stage2_grid == 0)] = 1
                summary_grid_s3[(gap_grid > res) & (stage2_grid == 1) & (s3_grid == 1)] = 2
                summary_grid_s3[(gap_grid > res) & (stage2_grid == 1) & (s3_grid == 0)] = 3
                summary_grid_s3[(gap_grid <= res) & (stage2_grid == 0)] = 4
                summary_grid_s3[(gap_grid <= res) & (stage2_grid == 1)] = 5
                
                fig, ax = plt.subplots(figsize=(7, 8), dpi=150)
                cmap_sum_s3 = mcolors.ListedColormap(['#1f77b4', '#9467bd', '#ff7f0e', '#2ca02c', '#d62728'])
                bounds_s3 = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
                norm_s3 = mcolors.BoundaryNorm(bounds_s3, cmap_sum_s3.N)
                im = ax.imshow(summary_grid_s3, origin='lower', extent=[unique_vz[0], unique_vz[-1], unique_mu[0], unique_mu[-1]], aspect='auto', cmap=cmap_sum_s3, norm=norm_s3)
                
                cbar = fig.colorbar(im, ax=ax, pad=0.03, ticks=[1, 2, 3, 4, 5])
                cbar.ax.set_yticklabels(['Gapped, no ZBP', 'Gapped, ZBP (Corr)', 'Gapped, ZBP (No Corr)', 'Gapless, no ZBP', 'Gapless, ZBP'])
                
                handles_list = []
                if np.any(island_mask):
                    VZ, MU = np.meshgrid(unique_vz, unique_mu)
                    ax.contourf(VZ, MU, island_mask, levels=[0.5, 1.5], colors=['white'], alpha=0.3, hatches=['//'])
                    ax.contour(VZ, MU, island_mask, levels=[0.5], colors=['black'], linewidths=1.5)
                    patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', alpha=0.3, label='Topological Island')
                    handles_list.append(patch)
                    
                if pfaffian_grid is not None and np.any(pfaffian_grid > 0.5):
                    VZ, MU = np.meshgrid(unique_vz, unique_mu)
                    ax.contourf(VZ, MU, pfaffian_grid, levels=[0.5, 1.5], colors=['#303030'], alpha=0.6)
                    patch_pfaff = mpatches.Patch(facecolor='#303030', alpha=0.6, label='True Topological (Pfaffian)')
                    handles_list.append(patch_pfaff)
                    
                if handles_list:
                    ax.legend(handles=handles_list, loc='upper right', fontsize=9, frameon=True)
                    
                ax.set_title(f'Stage 3 Summary {suffix.capitalize()} (Res {res})', fontsize=12)
                ax.set_xlabel(r'Zeeman Field $V_z$')
                ax.set_ylabel(r'Chemical Potential $\mu$')
                fig.tight_layout()
                fig.savefig(stage3_dir / f'stage3_summary_{suffix}_res_{res}.png', dpi=300)
                plt.close(fig)

            create_stage3_summary(s3_pass_L, "left")
            create_stage3_summary(s3_pass_R, "right")
            create_stage3_summary(s3_pass_both, "both")

        # Save TP/FP counts for FDR analysis
        if pfaffian_grid is not None:
            true_top = (pfaffian_grid > 0.98)
            TP_stage2 = int(np.sum(island_mask & true_top))
            FP_stage2 = int(np.sum(island_mask & ~true_top))
            
            if barrier_arr is not None:
                TP_stage3 = int(np.sum((s3_pass_both == 1) & island_mask & true_top))
                FP_stage3 = int(np.sum((s3_pass_both == 1) & island_mask & ~true_top))
            else:
                TP_stage3 = FP_stage3 = 0
                
            tp_fp_counts[str(res)] = {
                "Total_True": int(np.sum(true_top)),
                "Stage2": {"TP": TP_stage2, "FP": FP_stage2},
                "Stage3": {"TP": TP_stage3, "FP": FP_stage3}
            }

    # Raw Transport Gap
    gap_grid = to_grid(gap_transport_all)
    generate_continuous_phase_map(gap_grid, unique_mu, unique_vz, 'Raw Transport Gap', stage2_dir / 'phase_map_transport_gap.png', 'Gap', cmap='viridis')
    
    # Weight Localization (filtered by E0 <= 0.1)
    wl_grid = to_grid(weight_loc)
    wl_grid = np.where(E0_grid <= 0.1, wl_grid, np.nan)
    generate_continuous_phase_map(wl_grid, unique_mu, unique_vz, 'Weight Localizations', plots_dir / 'phase_map_weight_localizations.png', 'Localization', cmap='magma_r', I_grid=island_mask, I_color='white', I_alpha=0.25)
    
    # Topological Gap (E1 filtered by E0 <= 0.1)
    if E1 is not None:
        E1_grid = to_grid(E1)
        E1_grid = np.where(E0_grid <= 0.1, E1_grid, np.nan)
        generate_continuous_phase_map(E1_grid, unique_mu, unique_vz, 'Topological Gap (E1)', plots_dir / 'phase_map_topological_gap_E1.png', 'E1 Gap', cmap='viridis')
        
    # Conductance Correlation
    G_L = tgp_l[:, :, :, 3]
    G_R = tgp_r[:, :, :, 3]
    corr_arr = np.array([hp.calc_invariant_metric(G_L[i].flatten(), G_R[i].flatten()) for i in range(len(mu))])
    corr_grid = to_grid(corr_arr)
    generate_continuous_phase_map(corr_grid, unique_mu, unique_vz, 'Conductance Correlation', plots_dir / 'phase_map_conductance_correlation.png', 'Correlation', cmap='gray_r', vmin=0, vmax=1, dark_theme=False)
    
    if tp_fp_counts:
        with open(dirname / "tp_fp_counts.json", "w") as f:
            json.dump(tp_fp_counts, f, indent=4)
            
    print("Post-processing complete!")
    
if __name__ == "__main__":
    main()
