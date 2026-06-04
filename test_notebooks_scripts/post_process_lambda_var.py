#!/usr/bin/env python3
"""
post_process_lambda_var.py
Computes and plots conditional probabilities as a function of the characteristic
disorder length scale (lambda), grouping and averaging across realizations.
"""

import sys
import os
import re
from collections import defaultdict
from pathlib import Path
import numpy as np

# Use Agg backend for headless matplotlib operations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add current directory to path to import post_process_disorders
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

try:
    from post_process_disorders import load_realization, process_single_realization
except ImportError:
    # Fallback to local import if run from a sibling folder
    sys.path.append(os.getcwd())
    from post_process_disorders import load_realization, process_single_realization


def main():
    lambda_var_dir = project_root / "Data" / "lambda_var"
    if not lambda_var_dir.exists():
        print(f"Error: lambda_var directory does not exist: {lambda_var_dir}")
        sys.exit(1)
        
    subdirs = [d for d in lambda_var_dir.iterdir() if d.is_dir() and "lambda_" in d.name]
    if not subdirs:
        print(f"No results directories found in {lambda_var_dir}")
        sys.exit(1)
        
    # Group results by lambda value
    # key: lambda (float), value: list of dicts with 'p_a_given_b' and 'p_b_given_a'
    results_by_lambda = defaultdict(list)
    
    # Regex to match: disorder_realization_{R}_lambda_{L_dash}_results
    pattern = re.compile(r"disorder_realization_\d+_lambda_([\d-]+)_results")
    
    for subdir in subdirs:
        match = pattern.match(subdir.name)
        if not match:
            continue
        
        l_dash = match.group(1)
        l_val = float(l_dash.replace('-', '.'))
        
        try:
            data = load_realization(subdir)
            res = process_single_realization(
                data,
                width_thresh=0.015,
                height_thresh=0.0,
                pdi_thresh=0.8
            )
            
            p_a_given_b = res['p_a_given_b']
            p_b_given_a = res['p_b_given_a']
            
            # Append if not NaN
            results_by_lambda[l_val].append({
                'p_a_given_b': p_a_given_b,
                'p_b_given_a': p_b_given_a
            })
            print(f"Processed {subdir.name}: lambda={l_val}, P(A|B)={p_a_given_b:.4f}, P(B|A)={p_b_given_a:.4f}")
        except Exception as e:
            print(f"Error processing {subdir.name}: {e}")
            
    if not results_by_lambda:
        print("No valid results computed.")
        sys.exit(1)
        
    # Aggregate and average
    sorted_lambdas = sorted(results_by_lambda.keys())
    avg_p_a_given_b = []
    avg_p_b_given_a = []
    
    for l_val in sorted_lambdas:
        runs = results_by_lambda[l_val]
        
        p_a_b_list = [r['p_a_given_b'] for r in runs if not np.isnan(r['p_a_given_b'])]
        p_b_a_list = [r['p_b_given_a'] for r in runs if not np.isnan(r['p_b_given_a'])]
        
        avg_p_a_given_b.append(np.mean(p_a_b_list) if p_a_b_list else np.nan)
        avg_p_b_given_a.append(np.mean(p_b_a_list) if p_b_a_list else np.nan)
        
    # Plotting
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(sorted_lambdas, avg_p_a_given_b, marker='o', linestyle='-', color='crimson', label=r'$P(A|B) = P(\mathrm{Topological} \mid \mathrm{Protocol\ Positive})$')
    ax.plot(sorted_lambdas, avg_p_b_given_a, marker='s', linestyle='--', color='navy', label=r'$P(B|A) = P(\mathrm{Protocol\ Positive} \mid \mathrm{Topological})$')
    
    ax.set_xlabel(r'Disorder Characteristic Length Scale $\lambda$')
    ax.set_ylabel('Probability')
    ax.set_title('Conditional Probabilities vs. Disorder Length Scale')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(loc='best')
    ax.set_ylim(-0.05, 1.05)
    
    savepath = project_root / "conditional_probabilities_vs_lambda.png"
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    
    print(f"\nPlot successfully saved to: {savepath}")


if __name__ == "__main__":
    main()
