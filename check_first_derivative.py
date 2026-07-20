import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate average first derivatives of conductance curves across parameter space.")
    parser.add_argument("--dirname", type=str, default="Data/Tdis_pfaff4", help="Path to data directory")
    parser.add_argument("--outdir", type=str, default=None, help="Directory to save plots (default: dirname/Plots/First_Derivatives)")
    return parser.parse_args()

def load_data(dirname: Path):
    print(f"Loading conductance arrays from: {dirname}")
    brcl_path = dirname / "barrier_right_conductance_left_arr.npy"
    brcr_path = dirname / "barrier_right_conductance_right_arr.npy"
    blcl_path = dirname / "barrier_left_conductance_left_arr.npy"
    blcr_path = dirname / "barrier_left_conductance_right_arr.npy"
    barrier_path = dirname / "barrier_arr.npy"
    cfg_path = dirname / "resolved_params.yaml"

    if not all(p.exists() for p in [brcl_path, brcr_path, blcl_path, blcr_path, barrier_path]):
        raise FileNotFoundError(f"Missing one or more required .npy arrays in {dirname}")

    brcl = np.load(brcl_path, allow_pickle=True)
    brcr = np.load(brcr_path, allow_pickle=True)
    blcl = np.load(blcl_path, allow_pickle=True)
    blcr = np.load(blcr_path, allow_pickle=True)
    barrier_arr = np.load(barrier_path, allow_pickle=True)

    barrier_l = 2.0
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        barrier_l = float(cfg.get("barrier0", cfg.get("barrier_l", 2.0)))

    return brcl, brcr, blcl, blcr, barrier_arr, barrier_l

def normalize_max(arr: np.ndarray):
    """Normalize each 1D curve across axis=1 by its maximum value."""
    # arr has shape (M, N)
    max_vals = np.max(arr, axis=1, keepdims=True)
    # Safe division
    safe_max = np.where(max_vals > 1e-15, max_vals, 1.0)
    normed = arr / safe_max
    return normed

def normalize_sym(arr: np.ndarray, idx_sym: int):
    """Normalize each 1D curve across axis=1 by its value at idx_sym."""
    sym_vals = arr[:, idx_sym:idx_sym+1]
    # Check how many curves have near-zero conductance at symmetric point
    valid_mask = (np.abs(sym_vals.squeeze()) > 1e-15)
    invalid_count = np.sum(~valid_mask)
    if invalid_count > 0:
        print(f"  Note: {invalid_count} curves have near-zero symmetric conductance (<1e-15); using safe fallback.")
    safe_sym = np.where(np.abs(sym_vals) > 1e-15, sym_vals, 1.0)
    normed = arr / safe_sym
    return normed, valid_mask

def compute_derivative(arr_normed: np.ndarray, x: np.ndarray):
    """Compute first derivative dG/dx along axis=1."""
    return np.gradient(arr_normed, x, axis=1)

def make_plot(x: np.ndarray, mean_deriv: np.ndarray, std_deriv: np.ndarray, sem_deriv: np.ndarray,
              xlabel: str, title: str, outfile: Path, color="royalblue"):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    
    # Plot shaded standard deviation band (total spread across curves)
    ax.fill_between(x, mean_deriv - std_deriv, mean_deriv + std_deriv,
                    color=color, alpha=0.2, label=r"$\pm 1\sigma$ (Curve-to-Curve Spread)")
    
    # Plot mean curve
    ax.plot(x, mean_deriv, color=color, linewidth=2.5, label="Average First Derivative")
    
    # Plot error bars (using std or sem at sampled intervals to avoid clutter)
    step = max(1, len(x) // 25) # approx 25 error bars along x
    ax.errorbar(x[::step], mean_deriv[::step], yerr=std_deriv[::step],
                fmt='o', color=color, markersize=4, capsize=3, elinewidth=1.2,
                linestyle='None', label=r"Error Bars ($\pm 1\sigma$)")
    
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(r"First Derivative $\frac{d(G/G_{\mathrm{norm}})}{d(U/U_{\mathrm{sym}})}$", fontsize=13)
    ax.set_title(title, fontsize=14, pad=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {outfile}")

def main():
    args = parse_args()
    dirname = Path(args.dirname)
    outdir = Path(args.outdir) if args.outdir else dirname / "Plots" / "First_Derivatives"
    outdir.mkdir(parents=True, exist_ok=True)

    brcl, brcr, blcl, blcr, barrier_arr, barrier_l = load_data(dirname)
    M, N = brcl.shape
    print(f"Loaded {M} curves, each with {N} barrier sweep points.")
    print(f"Nominal/symmetric barrier_l = {barrier_l}")

    idx_sym = int(np.argmin(np.abs(barrier_arr - barrier_l)))
    print(f"Symmetric barrier index: {idx_sym} (value = {barrier_arr[idx_sym]:.4f})")

    # Dimensionless x coordinate: ratio of barrier potential
    x_data = barrier_arr / (barrier_l if barrier_l != 0 else 1.0)

    # Dictionary of configurations
    configs = {
        "brcl": (brcl, r"$U_{R}/U_{L}$", r"Right Sweep $G_{LL}$"),
        "brcr": (brcr, r"$U_{R}/U_{L}$", r"Right Sweep $G_{RR}$"),
        "blcl": (blcl, r"$U_{L}/U_{R}$", r"Left Sweep $G_{LL}$"),
        "blcr": (blcr, r"$U_{L}/U_{R}$", r"Left Sweep $G_{RR}$")
    }

    stats_save = {"x_data": x_data, "barrier_arr": barrier_arr}

    # 1. Normalization Method 1: Max Value
    print("\n--- Processing Normalization 1: Max Value in Conductance Curve ---")
    for key, (arr, xlabel, label_str) in configs.items():
        normed = normalize_max(arr)
        deriv = compute_derivative(normed, x_data)
        
        mean_deriv = np.mean(deriv, axis=0)
        std_deriv = np.std(deriv, axis=0)
        sem_deriv = std_deriv / np.sqrt(M)
        
        stats_save[f"{key}_max_mean"] = mean_deriv
        stats_save[f"{key}_max_std"] = std_deriv
        stats_save[f"{key}_max_sem"] = sem_deriv

        outfile = outdir / f"{key}_deriv_max.png"
        title = f"Average First Derivative (Max Normalized) - {label_str}"
        make_plot(x_data, mean_deriv, std_deriv, sem_deriv, xlabel, title, outfile, color="royalblue")

    # 2. Normalization Method 2: Symmetric Barrier Value
    print("\n--- Processing Normalization 2: Symmetric Barrier Conductance ---")
    for key, (arr, xlabel, label_str) in configs.items():
        normed, valid_mask = normalize_sym(arr, idx_sym)
        # Filter to valid curves for stats
        deriv = compute_derivative(normed[valid_mask], x_data)
        
        mean_deriv = np.mean(deriv, axis=0)
        std_deriv = np.std(deriv, axis=0)
        sem_deriv = std_deriv / np.sqrt(np.sum(valid_mask))
        
        stats_save[f"{key}_sym_mean"] = mean_deriv
        stats_save[f"{key}_sym_std"] = std_deriv
        stats_save[f"{key}_sym_sem"] = sem_deriv

        outfile = outdir / f"{key}_deriv_sym.png"
        title = f"Average First Derivative (Symmetric Barrier Normalized) - {label_str}"
        make_plot(x_data, mean_deriv, std_deriv, sem_deriv, xlabel, title, outfile, color="darkorange")

    # Save summary stats
    stats_file = outdir / "first_derivative_stats.npz"
    np.savez(stats_file, **stats_save)
    print(f"\nSaved aggregated derivative statistics to: {stats_file}")

    # Also generate summary 4-panel figures for quick comparison
    for norm_type, color, norm_name in [("max", "royalblue", "Max Value"), ("sym", "darkorange", "Symmetric Barrier")]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
        axes = axes.flatten()
        for i, (key, (_, xlabel, label_str)) in enumerate(configs.items()):
            ax = axes[i]
            mean_d = stats_save[f"{key}_{norm_type}_mean"]
            std_d = stats_save[f"{key}_{norm_type}_std"]
            
            ax.fill_between(x_data, mean_d - std_d, mean_d + std_d, color=color, alpha=0.2, label=r"$\pm 1\sigma$")
            ax.plot(x_data, mean_d, color=color, linewidth=2.0, label="Average Derivative")
            step = max(1, len(x_data) // 20)
            ax.errorbar(x_data[::step], mean_d[::step], yerr=std_d[::step], fmt='o', color=color,
                        markersize=3.5, capsize=2.5, elinewidth=1.0, linestyle='None')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(r"$\frac{d(G/G_{\mathrm{norm}})}{dx}$", fontsize=12)
            ax.set_title(label_str, fontsize=13)
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 0:
                ax.legend(fontsize=10)
        
        fig.suptitle(f"Average First Derivatives Across All {M} Curves ({norm_name} Normalized)", fontsize=15, y=0.98)
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        summary_out = outdir / f"all_derivs_{norm_type}_summary.png"
        fig.savefig(summary_out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved 4-panel summary: {summary_out}")

if __name__ == "__main__":
    main()
