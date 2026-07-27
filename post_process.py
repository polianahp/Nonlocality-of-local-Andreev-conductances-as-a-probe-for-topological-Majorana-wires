import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import argparse
from pathlib import Path
import os
import helpers as hp

def generate_continuous_phase_map(val_grid, unique_mu, unique_vz, title_str, filename_str, cbar_label, cmap='viridis', vmin=None, vmax=None, I_grid=None, I_color='#404040', dark_theme=False):
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
        ax.contourf(VZ, MU, I_grid, levels=[0.5, 1.5], colors=[I_color], alpha=0.55)
        ax.contour(VZ, MU, I_grid, levels=[0.5], colors=['#202020' if not dark_theme else '#ffffff'], linewidths=1.5, alpha=0.9)
        patch = mpatches.Patch(facecolor=I_color, edgecolor='#202020' if not dark_theme else '#ffffff', linewidth=1.5, alpha=0.6, label='Topological Phase')
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
    
    # Transport gap
    gap_transport_path = dirname / "gap_transport_all.npy"
    if gap_transport_path.exists():
        gap_transport_data = np.load(gap_transport_path)
        if gap_transport_data.ndim == 2:
            gap_transport_all = gap_transport_data[:, 2]
        else:
            gap_transport_all = gap_transport_data
    else:
        dIdVs_LR = np.load(dirname / "dIdVs_LR.npy")
        dIdVs_RL = np.load(dirname / "dIdVs_RL.npy")
        energies = np.load(dirname / "energies.npy")
        gap_LR, _, _ = hp.extract_nonlocal_gap(energies, dIdVs_LR)
        gap_RL, _, _ = hp.extract_nonlocal_gap(energies, dIdVs_RL)
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
    
    for res in resolutions:
        step = res_steps[res]
        
        # Calculate Curvature L and R for all 25 combinations (N, 5, 5)
        fL_m = tgp_l[:, :, :, idx_zero - step]
        fL_0 = tgp_l[:, :, :, idx_zero]
        fL_p = tgp_l[:, :, :, idx_zero + step]
        curv_L = (fL_p - 2*fL_0 + fL_m) / (res**2)
        
        fR_m = tgp_r[:, :, :, idx_zero - step]
        fR_0 = tgp_r[:, :, :, idx_zero]
        fR_p = tgp_r[:, :, :, idx_zero + step]
        curv_R = (fR_p - 2*fR_0 + fR_m) / (res**2)
        
        # Stage 1 and Stage 2 ZBP condition: negative curvature on BOTH sides
        zbp_mask = (curv_L < 0) & (curv_R < 0) # Shape: (N, 5, 5)
        
        # Stage 1: ZBP in >= 70% (18/25) of combinations
        stage1_count = np.sum(zbp_mask, axis=(1, 2))
        stage1_pass = (stage1_count >= 18).astype(float)
        
        # Stage 2: ZBP in >= 3 of 5 symmetric combinations
        zbp_sym = np.diagonal(zbp_mask, axis1=1, axis2=2) # Shape: (N, 5)
        stage2_count = np.sum(zbp_sym, axis=1)
        stage2_pass = (stage2_count >= 3).astype(float)
        
        stage1_grid = to_grid(stage1_pass)
        stage2_grid = to_grid(stage2_pass)
        
        # Plot Binary Maps
        generate_continuous_phase_map(stage1_grid, unique_mu, unique_vz, f'Stage 1 (Res {res})', stage1_dir / f'phase_map_TGP_stage1_res_{res}.png', 'Pass (1) / Fail (0)', cmap='gray', vmin=0, vmax=1)
        generate_continuous_phase_map(stage2_grid, unique_mu, unique_vz, f'Stage 2 (Res {res})', stage2_dir / f'phase_map_TGP_stage2_res_{res}.png', 'Pass (1) / Fail (0)', cmap='gray', vmin=0, vmax=1)
        
        # Qualified Transport Gap
        gap_grid = to_grid(gap_transport_all)
        gap_grid_qual = np.where(E0_grid <= res, gap_grid, np.nan)
        generate_continuous_phase_map(gap_grid_qual, unique_mu, unique_vz, f'Qualified Transport Gap (Res {res})', stage2_dir / f'phase_map_qualified_transport_gap_res_{res}.png', 'Gap', cmap='viridis')
        
        # TGP Summary Phase Diagram
        # 1. Gapped without ZBP
        # 2. Gapped with ZBP
        # 3. Gapless without ZBP
        # 4. Gapless with ZBP
        summary_grid = np.zeros_like(E0_grid)
        summary_grid[(E0_grid > res) & (stage2_grid == 0)] = 1
        summary_grid[(E0_grid > res) & (stage2_grid == 1)] = 2
        summary_grid[(E0_grid <= res) & (stage2_grid == 0)] = 3
        summary_grid[(E0_grid <= res) & (stage2_grid == 1)] = 4
        
        island_mask = (E0_grid > res) & (stage1_grid == 1) & (stage2_grid == 1)
        
        fig, ax = plt.subplots(figsize=(7, 8), dpi=150)
        cmap_sum = mcolors.ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds, cmap_sum.N)
        im = ax.imshow(summary_grid, origin='lower', extent=[unique_vz[0], unique_vz[-1], unique_mu[0], unique_mu[-1]], aspect='auto', cmap=cmap_sum, norm=norm)
        
        cbar = fig.colorbar(im, ax=ax, pad=0.03, ticks=[1, 2, 3, 4])
        cbar.ax.set_yticklabels(['Gapped, no ZBP', 'Gapped, ZBP', 'Gapless, no ZBP', 'Gapless, ZBP'])
        
        if np.any(island_mask):
            VZ, MU = np.meshgrid(unique_vz, unique_mu)
            ax.contourf(VZ, MU, island_mask, levels=[0.5, 1.5], colors=['white'], alpha=0.3, hatches=['//'])
            ax.contour(VZ, MU, island_mask, levels=[0.5], colors=['black'], linewidths=1.5)
            patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', alpha=0.3, label='Topological Island')
            ax.legend(handles=[patch], loc='upper right', fontsize=9, frameon=True)
            
        ax.set_title(f'TGP Summary (Res {res})', fontsize=12)
        ax.set_xlabel(r'Zeeman Field $V_z$')
        ax.set_ylabel(r'Chemical Potential $\mu$')
        fig.tight_layout()
        fig.savefig(stage2_dir / f'tgp_summary_res_{res}.png', dpi=300)
        plt.close(fig)

    # Raw Transport Gap
    gap_grid = to_grid(gap_transport_all)
    generate_continuous_phase_map(gap_grid, unique_mu, unique_vz, 'Raw Transport Gap', stage2_dir / 'phase_map_transport_gap.png', 'Gap', cmap='viridis')
    
    # Weight Localization (filtered by E0 <= 0.1)
    wl_grid = to_grid(weight_loc)
    wl_grid = np.where(E0_grid <= 0.1, wl_grid, np.nan)
    generate_continuous_phase_map(wl_grid, unique_mu, unique_vz, 'Weight Localizations', plots_dir / 'phase_map_weight_localizations.png', 'Localization', cmap='magma_r')
    
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
    generate_continuous_phase_map(corr_grid, unique_mu, unique_vz, 'Conductance Correlation', plots_dir / 'phase_map_conductance_correlation.png', 'Correlation', cmap='gray', vmin=0, vmax=1, dark_theme=True)
    
    print("Post-processing complete!")
    
if __name__ == "__main__":
    main()
