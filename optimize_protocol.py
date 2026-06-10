#!/usr/bin/env python3
"""
optimize_protocol.py

Grid Search hyperparameter optimization of Protocol V3.

Runs an exhaustive search over the `search_space` dictionary defined below.
Evaluates conditional probabilities:
    P(~A|B) = P(Trivial | Protocol Positive)    — false discovery rate
    P(~B|A) = P(Protocol Negative | Topological) — false negative rate

Results are saved to a structured .npz file for later plotting.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import optuna
import helpers as hp

# ==========================================
# DEFINE YOUR GRID SEARCH SPACE HERE
# ==========================================
# Lists define values to sweep. 
# A single-element list means the parameter is fixed.
search_space = {
    "corr_thresh": np.linspace(0.4,0.98,5),
    "window": [0.05],#np.linspace(0.001,0.05,10),
    "symmetry_tol": [1e-8],
    "stability_radius":np.linspace(0.02,0.05,5),
    "stability_frac":  [1] #np.linspace(0.3,1,10)
}

# Base parameters toggling the checks
base_params = {
    "check_correlation": True,
    "check_resonance_peak": False,
    "check_negative_peaks": False,
    "check_monotonic": False,
    "check_peak_symmetry": False,
    "check_peak_window": False,
    "check_island_stability": True,
}
# ==========================================

def load_all_realizations(input_dir):
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    subdirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("disorder_realization_")]

    def get_realization_idx(path):
        name = path.name
        parts = name.split('_')
        for p in parts:
            if p.isdigit(): return int(p)
        return 999999

    subdirs = sorted(subdirs, key=get_realization_idx)
    realizations_data = {}

    print(f"Found {len(subdirs)} realization subdirectories in {input_path}")
    for subdir in subdirs:
        realization_name = subdir.name
        param_file = subdir / "params_list.npy"
        if not param_file.exists(): continue
        param_data = np.load(param_file)

        data_dict = {}
        for filepath in subdir.glob("*.npy"):
            if filepath.name == "params_list.npy": continue
            data_dict[filepath.stem] = np.load(filepath)

        realizations_data[realization_name] = {"parameters": param_data, "data": data_dict}
    return realizations_data

def precompute_correlations(all_data):
    print("Pre-computing correlation values...")
    for name, realization in all_data.items():
        data = realization['data']
        brcl = data['barrier_right_conductance_left_arr']
        brcr = data['barrier_right_conductance_right_arr']
        corrs = np.array([hp.calc_correlation(brcl[i, :], brcr[i, :]) for i in range(brcr.shape[0])])
        data['precomputed_corrs'] = corrs
    print("  Done.")

def precompute_monotonicity(all_data):
    print("Pre-computing monotonicity checks...")
    for name, realization in all_data.items():
        data = realization['data']
        brcl = data['barrier_right_conductance_left_arr']
        brcr = data['barrier_right_conductance_right_arr']
        mono_left = np.all(np.diff(brcl, axis=-1) <= 0, axis=-1)
        mono_right = np.all(np.diff(brcr, axis=-1) <= 0, axis=-1)
        mono_pass = (mono_left & mono_right).astype(float)
        data['precomputed_monotonicity'] = mono_pass
    print("  Done.")

def precompute_pdi_ground_truth(all_data, pdi_thresh=0.9):
    print(f"Pre-computing PDI ground truth (thresh={pdi_thresh})...")
    for name, realization in all_data.items():
        data = realization['data']
        pdi_winding = data['pdi_data'][:, 2]
        pdi_binary = np.clip(pdi_winding, 0.0, 1.0)
        pdi_binary = np.where(pdi_binary > pdi_thresh, 1.0, 0.0)
        data['pdi_binary'] = pdi_binary
    print("  Done.")

def evaluate_protocol_on_realization(data, params_list, trial_params):
    params = base_params.copy()
    params.update(trial_params)
    
    mono_pass = data.get('precomputed_monotonicity', None)
    
    prot_dat = hp.calc_protocol_v3(
        data['precomputed_corrs'],
        data['peaks_left'],
        data['peaks_right'],
        mono_pass,
        data['pdi_data'],
        params
    )

    I = data['pdi_binary']
    A_bool = (I == 1.0)
    B_bool = (prot_dat == 1.0)

    tp = np.sum(A_bool & B_bool)
    fp = np.sum(~A_bool & B_bool)
    fn = np.sum(A_bool & ~B_bool)

    p_not_a_given_b = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    p_not_b_given_a = fn / (tp + fn) if (tp + fn) > 0 else np.nan
    return p_not_a_given_b, p_not_b_given_a

def objective(trial, all_data):
    trial_params = {}
    if base_params.get("check_correlation", False):
        trial_params["corr_thresh"] = trial.suggest_categorical("corr_thresh", search_space["corr_thresh"])
    if base_params.get("check_peak_window", False):
        trial_params["window"] = trial.suggest_categorical("window", search_space["window"])
    if base_params.get("check_peak_symmetry", False):
        trial_params["symmetry_tol"] = trial.suggest_categorical("symmetry_tol", search_space["symmetry_tol"])
    if base_params.get("check_island_stability", False):
        trial_params["stability_radius"] = trial.suggest_categorical("stability_radius", search_space["stability_radius"])
        trial_params["stability_frac"] = trial.suggest_categorical("stability_frac", search_space["stability_frac"])

    nab_list = []
    nba_list = []
    for name, realization in all_data.items():
        p_nab, p_nba = evaluate_protocol_on_realization(
            realization['data'], realization['parameters'], trial_params
        )
        nab_list.append(p_nab)
        nba_list.append(p_nba)
        
    avg_nab = np.nanmean(nab_list)
    avg_nba = np.nanmean(nba_list)
    
    # We save these custom attributes to the trial so we can extract them later
    trial.set_user_attr("p_not_a_given_b", avg_nab)
    trial.set_user_attr("p_not_b_given_a", avg_nba)
    
    # Return a combined cost just for optuna's sorting, but grid search explores everything anyway
    w = 0.8
    cost = w * avg_nab + (1 - w) * avg_nba
    return cost

def main():
    parser = argparse.ArgumentParser(description="Grid Search Protocol V3 hyperparameters.")
    parser.add_argument('--data-dir', type=str, default='/home/pseudonym/code/Nonlocal_Conductance/Nonlocality-of-local-Andreev-conductances-as-a-probe-for-topological-Majorana-wires/Data/Protocol_V2')
    parser.add_argument('--pdi-thresh', type=float, default=0.9)
    parser.add_argument('--study-name', type=str, default='protocol_v3_grid')
    parser.add_argument('--output', type=str, default='grid_search_results.npz')
    parser.add_argument('--n-jobs', type=int, default=10, help='Number of parallel jobs for Optuna (-1 for all cores)')
    
    args = parser.parse_args()

    all_data = load_all_realizations(args.data_dir)
    precompute_correlations(all_data)
    if base_params.get("check_monotonic", False): 
        precompute_monotonicity(all_data)
    precompute_pdi_ground_truth(all_data, pdi_thresh=args.pdi_thresh)

    # Set up Optuna Grid Search
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(study_name=args.study_name, sampler=sampler, direction="minimize")

    # n_jobs > 1 runs trials in parallel using multithreading.
    study.optimize(lambda trial: objective(trial, all_data), n_jobs=args.n_jobs, show_progress_bar=True)

    print("\nGRID SEARCH COMPLETE")
    print(f"Best cost: {study.best_trial.value:.6f}")
    print("Best params:", study.best_trial.params)
    
    # Extract data into structured arrays for .npz
    trials = study.trials
    results_dict = {}
    
    # Store parameter lists
    for key in search_space.keys():
        results_dict[key] = np.array([t.params.get(key, np.nan) for t in trials if t.state == optuna.trial.TrialState.COMPLETE])
        
    # Store probability outcomes
    results_dict['p_not_a_given_b'] = np.array([t.user_attrs.get("p_not_a_given_b", np.nan) for t in trials if t.state == optuna.trial.TrialState.COMPLETE])
    results_dict['p_not_b_given_a'] = np.array([t.user_attrs.get("p_not_b_given_a", np.nan) for t in trials if t.state == optuna.trial.TrialState.COMPLETE])
    results_dict['cost'] = np.array([t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE])
    
    out_path = Path(args.data_dir) / args.output
    np.savez(out_path, **results_dict)
    print(f"\nSaved grid search results and conditional probabilities to: {out_path}")

if __name__ == "__main__":
    main()