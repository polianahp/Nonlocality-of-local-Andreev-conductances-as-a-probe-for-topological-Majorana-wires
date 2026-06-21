#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure local imports work
sys.path.append(str(Path(__file__).parent.resolve()))

from config import PathConfigs
import helpers as hp

def main():
    parser = argparse.ArgumentParser(description="Plot overlay of Topological Region (I) and Protocol (prot_dat) using Matplotlib.")
    parser.add_argument("--dirname", type=str, default="Data/Tdis_pfaff3",
                        help="Directory containing the simulation results.")
    parser.add_argument("--pdi-thresh", type=float, default=0.9, help="PDI filtering threshold.")
    parser.add_argument("--corr-thresh", type=float, default=0.9, help="Correlation threshold.")
    parser.add_argument("--stability-radius", type=float, default=0.027, help="Stability radius.")
    parser.add_argument("--stability-frac", type=float, default=0.8, help="Stability fraction.")
    parser.add_argument("--output", type=str, default="Plots/overlap_plot.png", help="Path to save the output PNG plot.")
    args = parser.parse_args()

    # Resolve dirname
    if os.path.isabs(args.dirname):
        dirname = Path(args.dirname)
    else:
        dirname = Path(PathConfigs.ROOT) / args.dirname
        if not dirname.exists():
            dirname = Path(args.dirname)

    print(f"Loading data from: {dirname}")

    pdi_data_path = dirname / "pdi_data.npy"
    params_list_path = dirname / "params_list.npy"
    peaks_left_path = dirname / "peaks_left.npy"
    peaks_right_path = dirname / "peaks_right.npy"
    brcl_path = dirname / "barrier_right_conductance_left_arr.npy"
    brcr_path = dirname / "barrier_right_conductance_right_arr.npy"

    for path in [pdi_data_path, params_list_path, peaks_left_path, peaks_right_path, brcl_path, brcr_path]:
        if not path.exists():
            print(f"Error: Missing required file {path}")
            sys.exit(1)

    pdi_data = np.load(pdi_data_path)
    params_list = np.load(params_list_path)
    peaks_left = np.load(peaks_left_path)
    peaks_right = np.load(peaks_right_path)
    brcl = np.load(brcl_path)
    brcr = np.load(brcr_path)

    mu = params_list[:, 1]
    V_z = params_list[:, 2]

    # Calculate I
    if pdi_data.shape[1] > 3:
        pdi_winding = pdi_data[:, 3]
    else:
        pdi_winding = pdi_data[:, 2]
    
    I = hp.filter_pdi(pdi_winding, thresh=args.pdi_thresh)

    # Calculate correlations
    print("Computing correlations...")
    corrs = np.array([hp.calc_correlation(brcl[i], brcr[i]) for i in range(brcl.shape[0])])

    # Calculate decision map prot_dat
    print("Calculating protocol decision map (prot_dat)...")
    params = {
        "check_correlation": True,
        "corr_thresh": args.corr_thresh,
        "check_resonance_peak": False,
        "check_negative_peaks": False,
        "check_monotonic": False,
        "check_peak_symmetry": False,
        "check_peak_window": False,
        "check_island_stability": True,
        "stability_radius": args.stability_radius,
        "stability_frac": args.stability_frac
    }

    prot_dat = hp.calc_protocol_v3(
        corrs,
        peaks_left,
        peaks_right,
        None,
        pdi_data,
        params
    )

    # Map to 2D grid
    unique_mu = np.unique(mu)
    unique_vz = np.unique(V_z)
    Nmu = len(unique_mu)
    Nvz = len(unique_vz)
    print(f"Reconstructed grid: {Nmu} mu points, {Nvz} V_z points.")

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

    # Reconstruct RGBA Image Matrix directly to prevent alpha-blending artifacts
    rgba = np.ones((Nmu, Nvz, 4))
    for i in range(Nmu):
        for j in range(Nvz):
            val_I = I_grid[i, j]
            val_P = prot_grid[i, j]
            if val_I == 1.0 and val_P == 1.0:
                # Overlap: Purple
                rgba[i, j] = [0.6, 0.25, 0.6, 1.0]
            elif val_I == 1.0:
                # Topological only: Coral Red
                rgba[i, j] = [1.0, 0.5, 0.5, 1.0]
            elif val_P == 1.0:
                # Protocol only: Light Blue
                rgba[i, j] = [0.5, 0.5, 1.0, 1.0]
            else:
                # Neither: White
                rgba[i, j] = [1.0, 1.0, 1.0, 1.0]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 8), dpi=150)
    
    # Plot using imshow
    ax.imshow(rgba, origin='lower', extent=[unique_vz[0], unique_vz[-1], unique_mu[0], unique_mu[-1]], aspect='auto')

    # Label axes and title
    ax.set_title('Topological Region vs Protocol Positive Overlay', fontsize=12, pad=15)
    ax.set_xlabel(r'Zeeman Field $V_z$', fontsize=11)
    ax.set_ylabel(r'Chemical Potential $\mu$', fontsize=11)
    
    # Set axis ranges matching your notebook
    ax.set_xlim(0.0, 1.2)
    ax.set_ylim(0.0, 4.5)

    # Clean up top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create custom legend handles
    red_patch = mpatches.Patch(color=(1.0, 0.5, 0.5), label='Topological (I = 1)')
    blue_patch = mpatches.Patch(color=(0.5, 0.5, 1.0), label='Protocol Positive (prot_dat = 1)')
    purple_patch = mpatches.Patch(color=(0.6, 0.25, 0.6), label='Overlap (Both = 1)')
    
    # Add legend at the bottom
    ax.legend(handles=[red_patch, blue_patch, purple_patch], bbox_to_anchor=(0.5, -0.15),
              loc='upper center', ncol=3, fontsize=9, frameon=True)

    # Ensure plots directory exists
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(PathConfigs.ROOT) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save and close
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {output_path}")

if __name__ == "__main__":
    main()
