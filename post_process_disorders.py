#!/usr/bin/env python3
"""
post_process_disorders.py

A Python CLI script to aggregate and post-process simulation results across multiple
disorder realizations for non-local Andreev conductance and topological PDI phase maps.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np

# Use Agg backend for headless matplotlib operations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def filter_pdi(pdis, thresh=0.8):
    """
    Binarize PDI winding number values.
    
    - Clamp raw winding numbers to [0.0, 1.0]
    - Threshold: > thresh -> 1.0, else -> 0.0
    
    Operates on a copy to avoid mutating the original array.
    """
    result = np.array(pdis, copy=True)
    result = np.clip(result, 0.0, 1.0)
    result = np.where(result > thresh, 1.0, 0.0)
    return result


def filter_peaks(peak_data, width_thresh=None, height_thresh=None):
    """
    Filter peaks by energy width and conductance height.
    
    peak_data columns: [has_peak (0/1), energy_of_peak, height_of_peak]
    
    - width_thresh: maximum allowed energy of the peak (distance from zero-bias).
      Keep peaks where energy <= width_thresh.
    - height_thresh: minimum conductance height. Keep peaks where height >= height_thresh.
    
    Returns: 1D array of 0 or 1 representing peak presence after filtering.
    """
    peaks = np.array(peak_data[:, 0], copy=True)
    widths = peak_data[:, 1]
    heights = peak_data[:, 2]
    
    mask = np.ones_like(peaks)
    
    if height_thresh is not None:
        mask = mask * (heights >= height_thresh).astype(float)
        
    if width_thresh is not None:
        mask = mask * (widths <= width_thresh).astype(float)
        
    return peaks * mask


def load_realization(subdir_path):
    """
    Loads all required numpy data files from a single realization subdirectory.
    
    Raises FileNotFoundError with a clear message if any required file is missing.
    """
    subdir_path = Path(subdir_path)
    required_files = {
        'pdi_data': 'pdi_data.npy',
        'peaks_left': 'peaks_left.npy',
        'peaks_right': 'peaks_right.npy',
        'rG_corr': 'rG_corr.npy',
        'params_list': 'params_list.npy',
        'weight_localization': 'weight_localization_arr.npy'
    }
    
    data = {}
    for key, filename in required_files.items():
        filepath = subdir_path / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Required file '{filename}' not found in realization directory '{subdir_path}'"
            )
        data[key] = np.load(filepath)
        
    return data


def process_single_realization(data, width_thresh, height_thresh, pdi_thresh):
    """
    Processes the data of a single realization to calculate:
    - Binarized PDI map
    - Conductance map and binarized conductance map
    - P(A|B) and P(B|A) conditional probabilities
    - Counts for the confusion matrix (TP, FP, FN, TN)
    """
    # A = PDI is topological
    # B = Conductance protocol is positive
    
    # 1. PDI binarization (Col index 2 of pdi_data is PDI_winding_number)
    pdi_winding = data['pdi_data'][:, 2]
    pdi_binary = filter_pdi(pdi_winding, thresh=pdi_thresh)
    
    # 2. Conductance protocol map
    int_pks_left = filter_peaks(data['peaks_left'], width_thresh=width_thresh, height_thresh=height_thresh)
    int_pks_right = filter_peaks(data['peaks_right'], width_thresh=width_thresh, height_thresh=height_thresh)
    conductance_map = (int_pks_left * int_pks_right) * data['rG_corr']
    
    # Binarize conductance map (non-zero means positive protocol detection)
    conductance_binary = (conductance_map > 0.0).astype(float)
    
    # 3. Boolean masks
    A_bool = pdi_binary == 1.0
    B_bool = conductance_binary == 1.0
    total_points = len(A_bool)
    
    # 4. Probabilities
    PA = np.sum(A_bool) / total_points
    PB = np.sum(B_bool) / total_points
    PAB = np.sum(A_bool & B_bool) / total_points
    P_notAB = np.sum(~A_bool & B_bool) / total_points
    
    p_a_given_b = PAB / PB 
    p_b_given_a = PAB / PA 
    p_not_a_given_b = P_notAB / PB 
    
    # 5. Confusion matrix counts
    tp = np.sum(A_bool & B_bool)
    fp = np.sum(~A_bool & B_bool)
    fn = np.sum(A_bool & ~B_bool)
    tn = np.sum(~A_bool & ~B_bool)
    
    return {
        'pdi_binary': pdi_binary,
        'conductance_map': conductance_map,
        'conductance_binary': conductance_binary,
        'p_a_given_b': p_a_given_b,
        'p_b_given_a': p_b_given_a,
        'p_not_a_given_b': p_not_a_given_b,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def plot_phase_map(mu, vz, z, title, savepath, cmap='viridis', vmin=None, vmax=None,
                   xlabel=r'$V_z$ (meV)', ylabel=r'$\mu$ (meV)'):
    """
    Plots a 2D phase map as a filled contour plot on the (Vz, mu) grid.
    mu, vz, z are all 1D arrays of length n_mu * n_vz.
    """
    mu_vals = np.unique(mu)
    vz_vals = np.unique(vz)
    
    n_mu = len(mu_vals)
    n_vz = len(vz_vals)
    
    if n_mu * n_vz != len(z):
        raise ValueError(
            f"Dimensions mismatch: unique mu ({n_mu}) x unique Vz ({n_vz}) "
            f"= {n_mu * n_vz} points, but data length is {len(z)}."
        )
        
    Z = z.reshape(n_mu, n_vz)
    
    fig, ax = plt.subplots(figsize=(6, 8))
    
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin, vmax, 101)
    else:
        levels = 100
        
    cf = ax.contourf(vz_vals, mu_vals, Z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cf, ax=ax)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {savepath}")


def plot_histogram(values, title, xlabel, savepath, bins=15):
    """
    Plots a histogram of scalar values (one per realization) with a mean line.
    """
    clean = np.array(values)
    clean = clean[~np.isnan(clean)]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(clean, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(f'{title} (N={len(clean)})')
    
    if len(clean) > 0:
        mean_val = np.mean(clean)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_val:.4f}')
        ax.legend()
        
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {savepath}")


def main(input_dir, width_thresh=0.015, height_thresh=0.0, pdi_thresh=0.8, output_dir=None):
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")
        
    if output_dir is None:
        output_path = input_path / "post_process_results"
    else:
        output_path = Path(output_dir)
        
    plot_path = output_path / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Discover and sort subdirectories
    subdirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("disorder_realization_")]
    
    def get_realization_idx(path):
        name = path.name
        parts = name.split('_')
        for p in parts:
            if p.isdigit():
                return int(p)
        return 999999
        
    subdirs = sorted(subdirs, key=get_realization_idx)
    num_realizations = len(subdirs)
    
    if num_realizations == 0:
        print(f"No subdirectories starting with 'disorder_realization_' found in {input_path}")
        return
        
    print(f"Found {num_realizations} realization subdirectories in {input_path}")
    print(f"Saving output to: {output_path}")
    
    # 2. Accumulators
    all_pdi_maps = []
    all_conductance_maps = []
    all_conductance_binary_maps = []
    all_p_a_given_b = []
    all_p_b_given_a = []
    all_p_not_a_given_b = []
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    params_grid = None
    
    # 3. Processing loop
    for i, subdir in enumerate(subdirs):
        print(f"[{i+1}/{num_realizations}] Processing {subdir.name}...")
        data = load_realization(subdir)
        
        # Save params_list from the first realization
        if params_grid is None:
            params_grid = data['params_list']
            
        res = process_single_realization(
            data,
            width_thresh=width_thresh,
            height_thresh=height_thresh,
            pdi_thresh=pdi_thresh
        )
        
        all_pdi_maps.append(res['pdi_binary'])
        all_conductance_maps.append(res['conductance_map'])
        all_conductance_binary_maps.append(res['conductance_binary'])
        all_p_a_given_b.append(res['p_a_given_b'])
        all_p_b_given_a.append(res['p_b_given_a'])
        all_p_not_a_given_b.append(res['p_not_a_given_b'])
        
        total_tp += res['tp']
        total_fp += res['fp']
        total_fn += res['fn']
        total_tn += res['tn']
        
    # 4. Stack and compute aggregates
    pdi_stack = np.stack(all_pdi_maps)
    cond_stack = np.stack(all_conductance_maps)
    cond_binary_stack = np.stack(all_conductance_binary_maps)
    
    avg_pdi = np.mean(pdi_stack, axis=0)
    avg_cond = np.mean(cond_stack, axis=0)
    std_pdi = np.std(pdi_stack, axis=0)
    std_cond = np.std(cond_stack, axis=0)
    avg_diff = np.mean(pdi_stack - cond_binary_stack, axis=0)
    agreement = np.mean((pdi_stack == cond_binary_stack).astype(float), axis=0)
    
    # 5. Save data outputs
    np.save(output_path / 'avg_pdi_map.npy', avg_pdi)
    np.save(output_path / 'avg_conductance_map.npy', avg_cond)
    np.save(output_path / 'std_pdi_map.npy', std_pdi)
    np.save(output_path / 'std_conductance_map.npy', std_cond)
    np.save(output_path / 'avg_difference_map.npy', avg_diff)
    np.save(output_path / 'agreement_fraction_map.npy', agreement)
    np.save(output_path / 'p_a_given_b_values.npy', np.array(all_p_a_given_b))
    np.save(output_path / 'p_b_given_a_values.npy', np.array(all_p_b_given_a))
    np.save(output_path / 'p_not_a_given_b_values.npy', np.array(all_p_not_a_given_b))
    if params_grid is not None:
        np.save(output_path / 'params_grid.npy', params_grid)
        
    # 6. Save confusion matrix
    total_pooled_points = total_tp + total_fp + total_fn + total_tn
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
    accuracy = (total_tp + total_tn) / total_pooled_points if total_pooled_points > 0 else 0.0
    
    tp_pct = (total_tp / total_pooled_points * 100.0) if total_pooled_points > 0 else 0.0
    fp_pct = (total_fp / total_pooled_points * 100.0) if total_pooled_points > 0 else 0.0
    fn_pct = (total_fn / total_pooled_points * 100.0) if total_pooled_points > 0 else 0.0
    tn_pct = (total_tn / total_pooled_points * 100.0) if total_pooled_points > 0 else 0.0
    
    p_not_a_given_b_pooled = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    
    confusion = {
        'tp': tp_pct,
        'fp': fp_pct,
        'fn': fn_pct,
        'tn': tn_pct,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'p_not_a_given_b': p_not_a_given_b_pooled,
        'n_realizations': num_realizations,
        'n_points_per_realization': len(avg_pdi) if len(all_pdi_maps) > 0 else 0,
        'total_points_pooled': total_pooled_points
    }
    
    with open(output_path / 'confusion_matrix_summary.json', 'w') as f:
        json.dump(confusion, f, indent=2)
        
    # 7. Generate plots
    if params_grid is not None:
        mu = params_grid[:, 1]
        vz = params_grid[:, 2]
        
        print("\nGenerating phase map and statistical plots...")
        plot_phase_map(mu, vz, avg_pdi, 'Disorder-Averaged Winding Number',
                       plot_path / 'avg_pdi_map.png', cmap='RdYlBu_r', vmin=0.0, vmax=1.0)
                       
        plot_phase_map(mu, vz, avg_cond, 'Disorder-Averaged Conductance Protocol',
                       plot_path / 'avg_conductance_map.png', cmap='inferno', vmin=0.0)
                       
        plot_phase_map(mu, vz, std_pdi, 'PDI Std. Dev. Across Realizations',
                       plot_path / 'std_pdi_map.png', cmap='viridis', vmin=0.0)
                       
        plot_phase_map(mu, vz, std_cond, 'Conductance Protocol Std. Dev.',
                       plot_path / 'std_conductance_map.png', cmap='viridis', vmin=0.0)
                       
        plot_phase_map(mu, vz, avg_diff, 'Avg Difference (PDI - Conductance)',
                       plot_path / 'avg_difference_map.png', cmap='RdBu', vmin=-1.0, vmax=1.0)
                       
        plot_phase_map(mu, vz, agreement, 'Agreement Fraction',
                       plot_path / 'agreement_fraction_map.png', cmap='Greens', vmin=0.0, vmax=1.0)
                       
        plot_histogram(all_p_a_given_b, 'P(Topological | Protocol Positive)',
                       'P(A|B)', plot_path / 'p_a_given_b_histogram.png')
                       
        plot_histogram(all_p_b_given_a, 'P(Protocol Positive | Topological)',
                       'P(B|A)', plot_path / 'p_b_given_a_histogram.png')
                       
    # 8. Print and save summary
    p_a_given_b_clean = np.array(all_p_a_given_b)[~np.isnan(all_p_a_given_b)]
    p_b_given_a_clean = np.array(all_p_b_given_a)[~np.isnan(all_p_b_given_a)]
    p_not_a_given_b_clean = np.array(all_p_not_a_given_b)[~np.isnan(all_p_not_a_given_b)]
    
    summary_lines = [
        "=" * 60,
        "POST-PROCESSING SUMMARY",
        "=" * 60,
        f"Input directory:                  {input_path}",
        f"Output directory:                 {output_path}",
        f"Realizations processed:           {num_realizations}",
        f"Grid points per realization:      {len(avg_pdi) if len(all_pdi_maps) > 0 else 0}",
        f"Thresholds:                       PDI={pdi_thresh}, Width={width_thresh}, Height={height_thresh}",
        "",
        "--- Conditional Probabilities (Mean ± Std Across Realizations) ---",
        f"  P(A|B) = P(Topological | Protocol Positive):   {np.mean(p_a_given_b_clean):.4f} ± {np.std(p_a_given_b_clean):.4f}" if len(p_a_given_b_clean) > 0 else "  P(A|B): N/A",
        f"  P(~A|B) = P(Trivial | Protocol Positive):       {np.mean(p_not_a_given_b_clean):.4f} ± {np.std(p_not_a_given_b_clean):.4f}" if len(p_not_a_given_b_clean) > 0 else "  P(~A|B): N/A",
        f"  P(B|A) = P(Protocol Positive | Topological):   {np.mean(p_b_given_a_clean):.4f} ± {np.std(p_b_given_a_clean):.4f}" if len(p_b_given_a_clean) > 0 else "  P(B|A): N/A",
        "",
        "--- Pooled Confusion Matrix (% of Total Grid Points) ---",
        f"  TP: {tp_pct:>7.2f}%   FP: {fp_pct:>7.2f}%",
        f"  FN: {fn_pct:>7.2f}%   TN: {tn_pct:>7.2f}%",
        f"  Precision: {precision:.4f}",
        f"  Recall:    {recall:.4f}",
        f"  F1 Score:  {f1:.4f}",
        f"  Accuracy:  {accuracy:.4f}",
        "=" * 60,
    ]
    summary = "\n".join(summary_lines)
    print(summary)
    
    with open(output_path / 'summary.txt', 'w') as f:
        f.write(summary)
        
    print(f"\nAll files successfully saved to {output_path}.")


if __name__ == '__main__':
    from config import PathConfigs
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="Aggregate and post-process simulation results across disorder realizations."
    )
    parser.add_argument(
        'input_dir',
        type=str,
        nargs='?',
        default='/home/pseudonym/code/Nonlocal_Conductance/Nonlocality-of-local-Andreev-conductances-as-a-probe-for-topological-Majorana-wires/Data/Disorder_Realizations',
        help="Path to directory containing realization subdirectories."
    )
    parser.add_argument(
        '--width_thresh',
        type=float,
        default=0.01,
        help="Peak energy width threshold (default: 0.015)"
    )
    parser.add_argument(
        '--height_thresh',
        type=float,
        default=0.0,
        help="Peak height threshold (default: 0.0)"
    )
    parser.add_argument(
        '--pdi_thresh',
        type=float,
        default=0.98,
        help="PDI binarization threshold (default: 0.8)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Override output directory (default: input_dir/post_process_results)"
    )
    
    args = parser.parse_args()
    
    main(
        input_dir=args.input_dir,
        width_thresh=args.width_thresh,
        height_thresh=args.height_thresh,
        pdi_thresh=args.pdi_thresh,
        output_dir=args.output_dir
    )
