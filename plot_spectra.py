#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.append(str(Path(__file__).parent.resolve()))

from config import PathConfigs
import helpers as hp

def generate_point_path(pdi_data, N, resl, mu_start, mu_end, Vz_start, Vz_end):
    """
    Extracts a set of N unique, evenly spaced points along a defined straight-line cut
    through the 2D parameter space (mu and V_z) from the dataset.
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
        return np.array([])

    unique_indices = np.unique(valid_indices)
    unique_points = pdi_params[unique_indices]

    dist_from_start = np.linalg.norm(unique_points - start_vec, axis=1)
    sort_order = np.argsort(dist_from_start)

    sorted_unique_points = unique_points[sort_order]
    num_unique = len(sorted_unique_points)

    if N >= num_unique:
        sampled_points = sorted_unique_points
    else:
        sample_idx = np.round(np.linspace(0, num_unique - 1, N)).astype(int)
        sampled_points = sorted_unique_points[sample_idx]

    return sampled_points

def main():
    parser = argparse.ArgumentParser(description="Calculate and plot energy spectra along a cut in parameter space.")
    parser.add_argument("--dirname", type=str, default="Data/dis_realizations/disorder_realization_1_results",
                        help="Directory name containing the simulation parameters and pdi data.")
    parser.add_argument("-N", type=int, default=20, help="Number of points to sample along the cut.")
    parser.add_argument("--resl", type=float, default=0.02, help="Resolution/tolerance for matching points.")
    parser.add_argument("--mu-start", type=float, default=2.189732, help="Start Chemical Potential (mu) for the cut.")
    parser.add_argument("--mu-end", type=float, default=2.189732, help="End Chemical Potential (mu) for the cut.")
    parser.add_argument("--vz-start", type=float, default=0.0, help="Start Zeeman Field (V_z) for the cut.")
    parser.add_argument("--vz-end", type=float, default=1.4, help="End Zeeman Field (V_z) for the cut.")
    parser.add_argument("--target-vz", type=float, default=1.0, help="Predefined target V_z to draw a vertical reference line.")
    parser.add_argument("--kvals", type=int, default=12, help="Number of lowest eigenvalues to solve for.")
    parser.add_argument("--output", type=str, default="Plots/spectra_cut.png", help="Path to save the output plot.")
    args = parser.parse_args()

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

    if not params_path.exists() or not pdi_data_path.exists():
        print(f"Error: Missing params or pdi_data in {dirname}")
        sys.exit(1)

    params = np.load(params_path, allow_pickle=True)
    pdi_data = np.load(pdi_data_path, allow_pickle=True)

    # Extract required parameters from params
    t = float(params['t'])
    Delta0 = float(params['Delta0'])
    gamma = float(params['gamma'])
    alpha = float(params['alpha'])
    Ls = int(params['Ls'])
    V0 = float(params['V0'])
    Vdisx = params['Vdisx'] * V0

    # Step 1: Parameter Path Generation
    print(f"Generating point path along the cut: mu=({args.mu_start} -> {args.mu_end}), Vz=({args.vz_start} -> {args.vz_end})...")
    pts = generate_point_path(pdi_data, args.N, args.resl, args.mu_start, args.mu_end, args.vz_start, args.vz_end)

    if len(pts) == 0:
        print("Error: No points generated.")
        sys.exit(1)

    actual_N = len(pts)
    print(f"Sampled {actual_N} points along the path.")

    # Step 2: Physics Calculation
    print(f"Calculating closed system spectra for {actual_N} points (kvals={args.kvals})...")
    evals = np.zeros(shape=(actual_N, args.kvals))

    for i in range(actual_N):
        mu_val, vz_val = pts[i]
        # Build closed system
        scl = hp.build_system_closed(t, mu_val, gamma, Delta0, vz_val, alpha, Ls, Vdisx, a=1)
        # Calculate spectrum
        evals[i, :] = hp.calc_spectrum(scl, k=args.kvals)

    # Step 3: Dual X-Axis Visualization
    print("Visualizing results...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot the eigenvalues
    mid_idx = args.kvals // 2

    # Plot bulk states in standard blue
    # States below E=0 (indices 0 to mid_idx-2)
    for j in range(mid_idx - 1):
        ax.plot(range(actual_N), evals[:, j], 'o-', color='royalblue', alpha=0.7, linewidth=1.5, markersize=4)
    # States above E=0 (indices mid_idx+1 to kvals-1)
    for j in range(mid_idx + 1, args.kvals):
        ax.plot(range(actual_N), evals[:, j], 'o-', color='royalblue', alpha=0.7, linewidth=1.5, markersize=4)

    # Plot the two states closest to E=0 (indices mid_idx-1 and mid_idx) in red with a thicker line width
    ax.plot(range(actual_N), evals[:, mid_idx - 1], 'o-', color='red', alpha=0.9, linewidth=2.5, markersize=6)
    ax.plot(range(actual_N), evals[:, mid_idx], 'o-', color='red', alpha=0.9, linewidth=2.5, markersize=6)

    # Add reference lines
    ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)

    # Find the index closest to target_vz
    vz_values = pts[:, 1]
    closest_vz_idx = np.argmin(np.abs(vz_values - args.target_vz))
    target_mu_val = pts[closest_vz_idx, 0]
    ax.axvline(closest_vz_idx, color='black', linestyle='-', linewidth=1.5,
               label=f"$V_z = {vz_values[closest_vz_idx]:.4f}$ ($\mu = {target_mu_val:.4f}$)")

    # Set labels and formatting
    ax.set_ylabel("Energy (meV)", fontsize=12)
    ax.set_title(f"Low Energy Spectra along Cut", fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.5)

    # Primary X-Axis (mu)
    ax.set_xticks(range(actual_N))
    ax.set_xticklabels([f"{pts[i, 0]:.4f}" for i in range(actual_N)], rotation=45, ha='right', fontsize=9)
    ax.set_xlabel(r"Chemical Potential $\mu$ (meV)", fontsize=11, labelpad=15)

    # Secondary X-Axis (V_z)
    ax2 = ax.twiny()
    # Move its spine physically below the primary x-axis
    ax2.spines['bottom'].set_position(('outward', 65))
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')

    # Ensure the limits and ticks match the primary axis
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(actual_N))
    ax2.set_xticklabels([f"{pts[i, 1]:.4f}" for i in range(actual_N)], rotation=45, ha='right', fontsize=9)
    ax2.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11, labelpad=15)

    # Clean up top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

    # Ensure plots directory exists
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(PathConfigs.ROOT) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Adjust layout to fit both x-axes at the bottom
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to: {output_path}")

if __name__ == "__main__":
    main()
