import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
data_path = Path("/home/pseudonym/code/Nonlocal_Conductance/Nonlocality-of-local-Andreev-conductances-as-a-probe-for-topological-Majorana-wires/Data/Protocol_V2/grid_search_results.npz")
if not data_path.exists():
    raise FileNotFoundError(f"Could not find grid search data at {data_path}")

data = np.load(data_path)

corr_thresh = data['corr_thresh']
stability_radius = data['stability_radius']
p_nab = data['p_not_a_given_b']
p_nba = data['p_not_b_given_a']

# Get unique values
unique_stability = np.unique(stability_radius)
unique_corrs = np.unique(corr_thresh)

# Determine global y-axis limits for uniform scaling across all plots
valid_nab = p_nab[~np.isnan(p_nab)]
valid_nba = p_nba[~np.isnan(p_nba)]
ymax = max(
    np.max(valid_nab) if len(valid_nab) > 0 else 1.0,
    np.max(valid_nba) if len(valid_nba) > 0 else 1.0
)
y_limits = (-0.02, ymax * 1.1)

root_dir = Path("/home/pseudonym/code/Nonlocal_Conductance/Nonlocality-of-local-Andreev-conductances-as-a-probe-for-topological-Majorana-wires")

# =========================================================================
# 1. Probabilities vs corr_thresh (lines for each stability_radius)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for r in unique_stability:
    mask = (stability_radius == r)
    x = corr_thresh[mask]
    y_nab = p_nab[mask]
    y_nba = p_nba[mask]
    
    # Sort by x for clean line plotting
    sort_idx = np.argsort(x)
    
    ax1.plot(x[sort_idx], y_nab[sort_idx], marker='o', linewidth=2, label=f'Radius = {r:.4f}')
    ax2.plot(x[sort_idx], y_nba[sort_idx], marker='o', linewidth=2, label=f'Radius = {r:.4f}')

ax1.set_xlabel('Correlation Threshold', fontsize=12)
ax1.set_ylabel('P(Trivial | Protocol Positive)', fontsize=12)
ax1.set_title('False Discovery Rate vs Corr Thresh', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.set_xlabel('Correlation Threshold', fontsize=12)
ax2.set_ylabel('P(Protocol Negative | Topological)', fontsize=12)
ax2.set_title('False Negative Rate vs Corr Thresh', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)

ax1.set_ylim(y_limits)
ax2.set_ylim(y_limits)

plt.tight_layout()
plt.savefig(root_dir / 'probs_vs_corr.png', dpi=150)
plt.close()

# =========================================================================
# 2. Probabilities vs stability_radius (lines for each corr_thresh)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for c in unique_corrs:
    mask = (corr_thresh == c)
    x = stability_radius[mask]
    y_nab = p_nab[mask]
    y_nba = p_nba[mask]
    
    # Sort by x
    sort_idx = np.argsort(x)
    
    ax1.plot(x[sort_idx], y_nab[sort_idx], marker='s', linewidth=2, label=f'Corr Thresh = {c:.2f}')
    ax2.plot(x[sort_idx], y_nba[sort_idx], marker='s', linewidth=2, label=f'Corr Thresh = {c:.2f}')

ax1.set_xlabel('Stability Radius (meV)', fontsize=12)
ax1.set_ylabel('P(Trivial | Protocol Positive)', fontsize=12)
ax1.set_title('False Discovery Rate vs Stability Radius', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.set_xlabel('Stability Radius (meV)', fontsize=12)
ax2.set_ylabel('P(Protocol Negative | Topological)', fontsize=12)
ax2.set_title('False Negative Rate vs Stability Radius', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)

ax1.set_ylim(y_limits)
ax2.set_ylim(y_limits)

plt.tight_layout()
plt.savefig(root_dir / 'probs_vs_stability.png', dpi=150)
plt.close()

print(f"Successfully generated plots:")
print(f"  - {root_dir / 'probs_vs_corr.png'}")
print(f"  - {root_dir / 'probs_vs_stability.png'}")
