#!/usr/bin/env python3
"""
optimize_protocol.py

Bayesian hyperparameter optimization of Protocol V2 using Optuna.

Optimizes protocol hyperparameters to minimize a weighted cost function
combining two conditional probability errors:
    P(~A|B) = P(Trivial | Protocol Positive)    — false discovery rate
    P(~B|A) = P(Protocol Negative | Topological) — false negative rate

The cost function is:
    cost = w * P(~A|B) + (1 - w) * P(~B|A)

where w=0.8 by default (prioritize minimizing false discoveries).
"""

import os
import argparse
from pathlib import Path
import numpy as np
import optuna
import helpers as hp

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

def generate_cost_function(weight_false_discovery=0.8):
    w = weight_false_discovery
    def cost_fn(p_not_a_given_b, p_not_b_given_a):
        if np.isnan(p_not_a_given_b) and np.isnan(p_not_b_given_a): return 1.0
        if np.isnan(p_not_a_given_b): p_not_a_given_b = 0.0
        if np.isnan(p_not_b_given_a): p_not_b_given_a = 0.0
        return w * p_not_a_given_b + (1 - w) * p_not_b_given_a
    return cost_fn

def evaluate_protocol_on_realization(data, params_list, base_params, trial_params):
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

def objective(trial, all_data, cost_fn, base_params):
    trial_params = {}
    if base_params.get("check_correlation", False):
        trial_params["corr_thresh"] = trial.suggest_float("corr_thresh", 0.5, 0.99)
    if base_params.get("check_peak_window", False):
        trial_params["window"] = trial.suggest_float("window", 0.001, 0.1, log=True)
    if base_params.get("check_peak_symmetry", False):
        trial_params["symmetry_tol"] = trial.suggest_float("symmetry_tol", 0.0, 0.05)
    if base_params.get("check_island_stability", False):
        trial_params["stability_radius"] = trial.suggest_float("stability_radius", 0.01, 0.1)
        trial_params["stability_frac"] = trial.suggest_float("stability_frac", 0.3, 0.99)

    costs = []
    for name, realization in all_data.items():
        p_nab, p_nba = evaluate_protocol_on_realization(
            realization['data'], realization['parameters'], base_params, trial_params
        )
        costs.append(cost_fn(p_nab, p_nba))
    return np.mean(costs)

def main():
    parser = argparse.ArgumentParser(description="Optimize Protocol V3 hyperparameters.")
    parser.add_argument('--data-dir', type=str, default='/home/pseudonym/code/Nonlocal_Conductance/Nonlocality-of-local-Andreev-conductances-as-a-probe-for-topological-Majorana-wires/Data/Protocol_V2')
    parser.add_argument('--n-trials', type=int, default=200)
    parser.add_argument('--weight', type=float, default=0.6)
    parser.add_argument('--pdi-thresh', type=float, default=0.9)
    parser.add_argument('--study-name', type=str, default='protocol_v3_opt')
    parser.add_argument('--journal-path', type=str, default=None)
    
    # Boolean toggles matching the notebook
    parser.add_argument('--check-correlation', action='store_true', default=True)
    parser.add_argument('--check-resonance-peak', action='store_true')
    parser.add_argument('--check-negative-peaks', action='store_true')
    parser.add_argument('--check-monotonic', action='store_true')
    parser.add_argument('--check-peak-symmetry', action='store_true')
    parser.add_argument('--check-peak-window', action='store_true')
    parser.add_argument('--check-island-stability', action='store_true', default=True)
    
    args = parser.parse_args()

    base_params = {
        "check_correlation": args.check_correlation,
        "check_resonance_peak": args.check_resonance_peak,
        "check_negative_peaks": args.check_negative_peaks,
        "check_monotonic": args.check_monotonic,
        "check_peak_symmetry": args.check_peak_symmetry,
        "check_peak_window": args.check_peak_window,
        "check_island_stability": args.check_island_stability,
    }

    all_data = load_all_realizations(args.data_dir)
    precompute_correlations(all_data)
    if args.check_monotonic: precompute_monotonicity(all_data)
    precompute_pdi_ground_truth(all_data, pdi_thresh=args.pdi_thresh)

    cost_fn = generate_cost_function(weight_false_discovery=args.weight)
    
    if args.journal_path:
        from optuna.storages import JournalStorage, JournalFileBackend
        storage = JournalStorage(JournalFileBackend(args.journal_path))
        study = optuna.create_study(study_name=args.study_name, storage=storage, load_if_exists=True, direction="minimize")
    else:
        study = optuna.create_study(study_name=args.study_name, direction="minimize")

    study.optimize(lambda trial: objective(trial, all_data, cost_fn, base_params), n_trials=args.n_trials, show_progress_bar=True)

    print("\nOPTIMIZATION COMPLETE")
    print(f"Best cost: {study.best_trial.value:.6f}")
    for key, val in study.best_trial.params.items():
        print(f"  {key} = {val}")

if __name__ == "__main__":
    main()