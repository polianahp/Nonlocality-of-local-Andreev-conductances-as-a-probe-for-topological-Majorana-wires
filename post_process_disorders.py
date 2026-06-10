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
import helpers as hp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def filter_pdi(pdis, thresh=0.8):
    result = np.array(pdis, copy=True)
    result = np.clip(result, 0.0, 1.0)
    result = np.where(result > thresh, 1.0, 0.0)
    return result

def load_realization(subdir_path):
    subdir_path = Path(subdir_path)
    required_files = {
        'pdi_data': 'pdi_data.npy',
        'peaks_left': 'peaks_left.npy',
        'peaks_right': 'peaks_right.npy',
        'params_list': 'params_list.npy',
        'barrier_right_conductance_left_arr': 'barrier_right_conductance_left_arr.npy',
        'barrier_right_conductance_right_arr': 'barrier_right_conductance_right_arr.npy'
    }
    
    data = {}
    for key, filename in required_files.items():
        filepath = subdir_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file '{filename}' not found in '{subdir_path}'")
        data[key] = np.load(filepath)
    return data

def process_single_realization(data, params, pdi_thresh):
    I = filter_pdi(data['pdi_data'][:, 2], thresh=pdi_thresh)
    
    brcl = data['barrier_right_conductance_left_arr']
    brcr = data['barrier_right_conductance_right_arr']
    
    if np.ndim(brcl) == 1:
        corr_map = np.array([hp.calc_correlation(brcl, brcr)])
    else:
        corr_map = np.array([hp.calc_correlation(brcl[i, :], brcr[i, :]) for i in range(len(brcl))])

    mono_pass = None
    if params.get('check_monotonic', False):
        mono_left = np.all(np.diff(brcl, axis=-1) <= 0, axis=-1)
        mono_right = np.all(np.diff(brcr, axis=-1) <= 0, axis=-1)
        mono_pass = (mono_left & mono_right).astype(float)
        if np.ndim(mono_pass) == 0:
            mono_pass = np.array([mono_pass])

    prot_dat = hp.calc_protocol_v3(
        corr_map,
        data['peaks_left'],
        data['peaks_right'],
        mono_pass,
        data['pdi_data'],
        params
    )
    
    A_bool = I == 1.0
    B_bool = prot_dat == 1.0
    tp = np.sum(A_bool & B_bool)
    fp = np.sum(~A_bool & B_bool)
    fn = np.sum(A_bool & ~B_bool)
    tn = np.sum(~A_bool & ~B_bool)
    
    p_b_given_a = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    p_not_b_given_a = fn / (tp + fn) if (tp + fn) > 0 else np.nan
    
    Ppcor = np.sum(B_bool) / len(B_bool)
    PpcornI = fp / len(B_bool)
    PpcorpI = tp / len(B_bool)
    
    PnIgcor = PpcornI / Ppcor if Ppcor > 0 else np.nan
    PIgcor = PpcorpI / Ppcor if Ppcor > 0 else np.nan

    return {
        'pdi_binary': I,
        'conductance_map': prot_dat,
        'p_a_given_b': PIgcor,
        'p_b_given_a': p_b_given_a,
        'p_not_a_given_b': PnIgcor,
        'p_not_b_given_a': p_not_b_given_a,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
    }

def plot_phase_map(mu, vz, z, title, savepath, cmap='viridis', vmin=None, vmax=None):
    mu_vals, vz_vals = np.unique(mu), np.unique(vz)
    n_mu, n_vz = len(mu_vals), len(vz_vals)
    if n_mu * n_vz != len(z): return
    Z = z.reshape(n_mu, n_vz)
    fig, ax = plt.subplots(figsize=(6, 8))
    levels = np.linspace(vmin, vmax, 101) if vmin is not None and vmax is not None else 100
    cf = ax.contourf(vz_vals, mu_vals, Z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel(r'$V_z$ (meV)')
    ax.set_ylabel(r'$\mu$ (meV)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)

def plot_histogram(values, title, xlabel, savepath, bins=15):
    clean = np.array(values)
    clean = clean[~np.isnan(clean)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(clean, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(f'{title} (N={len(clean)})')
    if len(clean) > 0:
        ax.axvline(np.mean(clean), color='red', linestyle='--', linewidth=1.5, label=f'Mean = {np.mean(clean):.4f}')
        ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)

params = {
    "check_correlation"      : True,
    "check_resonance_peak"   : False,
    "check_negative_peaks"   : False,
    "check_monotonic"        : False,
    "check_peak_symmetry"    : False, 
    "check_peak_window"      : False,
    "check_island_stability" : False,
    
    "corr_thresh"     : 0.99,
    "window"          : 0.035,
    "stability_radus" : 0.045,
    "stability_frac"  : 1.0,
    "symmetry_tol"    : 1e-8
}

def main():
    parser = argparse.ArgumentParser(description="Aggregate and post-process simulation results.")
    parser.add_argument('--input_dir', type=str, default='/home/pseudonym/code/Nonlocal_Conductance/Nonlocality-of-local-Andreev-conductances-as-a-probe-for-topological-Majorana-wires/Data/Protocol_V2')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--pdi_thresh', type=float, default=0.9)
    
    args = parser.parse_args()

    # Map 'stability_radus' to 'stability_radius' for the internal function just in case
    if 'stability_radius' not in params and 'stability_radus' in params:
        params['stability_radius'] = params['stability_radus']

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir) if args.output_dir else input_path / 'post_process_results'
    plot_path = output_path / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    subdirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("disorder_realization_")], key=lambda p: int(p.name.split('_')[-1]) if p.name.split('_')[-1].isdigit() else 999)
    if not subdirs: return

    all_pdi = []; all_cond = []
    p_a_b = []; p_b_a = []; p_not_a_b = []; p_not_b_a = []
    tp_tot = fp_tot = fn_tot = tn_tot = 0
    params_grid = None

    for i, subdir in enumerate(subdirs):
        print(f"[{i+1}/{len(subdirs)}] Processing {subdir.name}...")
        data = load_realization(subdir)
        if params_grid is None: params_grid = data['params_list']
        
        res = process_single_realization(data, params, args.pdi_thresh)
        all_pdi.append(res['pdi_binary'])
        all_cond.append(res['conductance_map'])
        p_a_b.append(res['p_a_given_b'])
        p_b_a.append(res['p_b_given_a'])
        p_not_a_b.append(res['p_not_a_given_b'])
        p_not_b_a.append(res['p_not_b_given_a'])
        tp_tot += res['tp']; fp_tot += res['fp']; fn_tot += res['fn']; tn_tot += res['tn']

    avg_pdi = np.mean(all_pdi, axis=0)
    avg_cond = np.mean(all_cond, axis=0)
    
    np.save(output_path / 'avg_pdi_map.npy', avg_pdi)
    np.save(output_path / 'avg_conductance_map.npy', avg_cond)
    
    print("\nGenerating plots...")
    if params_grid is not None:
        mu, vz = params_grid[:, 1], params_grid[:, 2]
        plot_phase_map(mu, vz, avg_pdi, 'Avg Winding Number', plot_path / 'avg_pdi_map.png', vmin=0.0, vmax=1.0)
        plot_phase_map(mu, vz, avg_cond, 'Avg Protocol Map', plot_path / 'avg_conductance_map.png', cmap='inferno', vmin=0.0)
        
    p_trivial_given_positive = fp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else np.nan
    p_negative_given_topological = fn_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else np.nan
    
    summary = f"Processed {len(subdirs)} realizations.\n"
    summary += f"P(Trivial | Positive Protocol):  {p_trivial_given_positive} |  P(Negative Protocol | Topological):   {p_negative_given_topological}\n"
    summary += f"Precision: {tp_tot/(tp_tot+fp_tot):.4f}\nRecall: {tp_tot/(tp_tot+fn_tot):.4f}"
    
    with open(output_path / 'summary.txt', 'w') as f: 
        f.write(summary)
    print(summary)

if __name__ == '__main__':
    main()
