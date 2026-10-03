import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import glob
import tgp
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.tgp_adapter import TGPAdapter
import sys
from pathlib import Path
sys.path.append("/home/pseudonym/Documents/Code/azure-quantum-tgp/notebooks")
from yield_analysis import analyze_2
from src.parameter_handler import ConfigManager
from src.config import PathConfigs

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
# Gather all disorder realization directories
DATA_DIRS = [str(p) for p in PathConfigs.DATA.glob('Tdis_pfaff5_V0_0_*')]
# Add any additional directories you want to include here:
ADDITIONAL_DIRS = []
DATA_DIRS.extend(ADDITIONAL_DIRS)

# Remove duplicates 
def sort_key(x):
    import re
    m = re.search(r'V0_(\d+)_(\d+)', x)
    if m:
        try:
            return float(f"{m.group(1)}.{m.group(2)}")
        except:
            return 999.0
    try:
        return int(x.split('_')[2])
    except:
        return 999.0
DATA_DIRS = sorted(list(set(DATA_DIRS)), key=sort_key)

OUTPUT_DIR = str(PathConfigs.DATA / 'Transport_Gap_Analysis')
# Load protocol parameters
p = ConfigManager.get_protocol_config('Parameters/default_protocol.yaml')

BIN_WIDTH_E0 = 0.005
BIN_WIDTH_E1 = 0.005
BIN_WIDTH_E1_E0 = 0.005
BIN_WIDTH_GAP = 0.005

E0_THRESHOLDS = [0.005, 0.01, 0.02]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load and Aggregate Data across all directories
# ---------------------------------------------------------
all_dfs = []
all_cond_dfs = []

print(f"Aggregating data across {len(DATA_DIRS)} directories...")
for data_dir in DATA_DIRS:
    if not os.path.exists(f"{data_dir}/all_params.npz"):
        print(f"Skipping {data_dir} because all_params.npz is missing.")
        continue

    print(f"Processing {data_dir}...")
    params = np.load(f"{data_dir}/all_params.npz", allow_pickle=True)
    mu_var = params['mu_var']
    Vz_var = params['Vz_var']

    spec = np.load(f"{data_dir}/spectrum_arr.npy", allow_pickle=True)
    spec_abs = np.sort(np.abs(spec), axis=-1)

    E0_flat = spec_abs[:, 0]
    E1_flat = spec_abs[:, 2]

    Nmu = len(mu_var)
    Nvz = len(Vz_var)

    E0 = E0_flat.reshape(Nmu, Nvz)
    E1 = E1_flat.reshape(Nmu, Nvz)

    E0_xr = xr.DataArray(E0, coords=[('V', mu_var), ('B', Vz_var)])
    E1_xr = xr.DataArray(E1, coords=[('V', mu_var), ('B', Vz_var)])
    E1_E0_xr = E1_xr - E0_xr

    # Transport gap extraction
    adapter = TGPAdapter(data_dir)
    ds = adapter.to_xarray()
    res = analyze_2(ds, T_mK=p.T_mK, force=False)
    zbp_ds = res['zbp_ds']
    transport_gap = zbp_ds.gap.min(dim="cutter_pair_index")

    E0_xr, E1_xr, E1_E0_xr, transport_gap = xr.align(E0_xr, E1_xr, E1_E0_xr, transport_gap, join='inner')

    df = pd.DataFrame({
        'E0': E0_xr.values.flatten(),
        'E1': E1_xr.values.flatten(),
        'E1_E0': E1_E0_xr.values.flatten(),
        'gap': transport_gap.values.flatten()
    })
    all_dfs.append(df)
    
    # Conditional Prob Logic (2w)
    tprep = tgp.prepare.prepare_sim(ds, p.T_mK)
    tgp.one.set_2w_th(tprep, n_tiles=p.n_tiles, th_2w=p.th_2w)
    tgp.one.set_gapped(tprep, th_2w_p=p.th_2w_p)
    
    E0_xr_aligned = E0_xr.transpose('B', 'V')
    E1_xr_aligned = E1_xr.transpose('B', 'V')
    is_gapless = (tprep.gapped == 0).values
    
    # Ensure pfaffian is topologically non-trivial (L_SI == 1.0)
    # L_SI corresponds to pfaffian invariant mapped to {0, 1}
    is_topological = (tprep.L_SI.isel(cutter_pair_index=0).values == 1.0)
    
    df_cond = pd.DataFrame({
        'E0': E0_xr_aligned.values.flatten(),
        'E1': E1_xr_aligned.values.flatten(),
        'gapless': is_gapless.flatten(),
        'is_topological': is_topological.flatten()
    })
    all_cond_dfs.append(df_cond)

# Combine into master dataframes
df_master = pd.concat(all_dfs, ignore_index=True)
df_master = df_master.dropna()

df_cond_master = pd.concat(all_cond_dfs, ignore_index=True)

# ---------------------------------------------------------
# Correlations
# ---------------------------------------------------------
print("\n--- Pearson Correlations ---")
corr_E0 = df_master['gap'].corr(df_master['E0'])
corr_E1 = df_master['gap'].corr(df_master['E1'])
corr_E1_E0 = df_master['gap'].corr(df_master['E1_E0'])

print(f"Correlation (Transport Gap, E0): {corr_E0:.4f}")
print(f"Correlation (Transport Gap, E1): {corr_E1:.4f}")
print(f"Correlation (Transport Gap, E1 - E0): {corr_E1_E0:.4f}")

with open(f"{OUTPUT_DIR}/raw_correlations.txt", "w") as f:
    f.write(f"Correlation (Transport Gap, E0): {corr_E0:.4f}\n")
    f.write(f"Correlation (Transport Gap, E1): {corr_E1:.4f}\n")
    f.write(f"Correlation (Transport Gap, E1 - E0): {corr_E1_E0:.4f}\n")

# ---------------------------------------------------------
# Binned and Cumulative Correlations (binned by E_x)
# ---------------------------------------------------------
def get_binned_data(df, var_name, bin_width):
    min_val = df[var_name].min()
    max_val = df[var_name].max()
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    
    disjoint_corrs, disjoint_centers = [], []
    for i in range(len(bins)-1):
        mask = (df[var_name] >= bins[i]) & (df[var_name] < bins[i+1])
        sub_df = df[mask]
        if len(sub_df) > 5:
            c = sub_df['gap'].corr(sub_df[var_name])
            if not np.isnan(c):
                disjoint_corrs.append(c)
                disjoint_centers.append(bins[i] + bin_width/2.0)
                
    cumulative_corrs, cumulative_edges = [], []
    for edge in bins[1:]:
        mask = (df[var_name] <= edge)
        sub_df = df[mask]
        if len(sub_df) > 5:
            c = sub_df['gap'].corr(sub_df[var_name])
            if not np.isnan(c):
                cumulative_corrs.append(c)
                cumulative_edges.append(edge)
                
    return disjoint_centers, disjoint_corrs, cumulative_edges, cumulative_corrs

def plot_combined_binned_correlations(df, bin_width_E0, bin_width_E1, bin_width_E1_E0, filename):
    dis_c_E0, dis_corr_E0, cum_e_E0, cum_corr_E0 = get_binned_data(df, 'E0', bin_width_E0)
    dis_c_E1, dis_corr_E1, cum_e_E1, cum_corr_E1 = get_binned_data(df, 'E1', bin_width_E1)
    dis_c_E1_E0, dis_corr_E1_E0, cum_e_E1_E0, cum_corr_E1_E0 = get_binned_data(df, 'E1_E0', bin_width_E1_E0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(dis_c_E0, dis_corr_E0, 'o-', label='E0')
    axes[0].plot(dis_c_E1, dis_corr_E1, 's-', label='E1')
    axes[0].plot(dis_c_E1_E0, dis_corr_E1_E0, '^-', label='E1 - E0')
    axes[0].set_xlabel('Energy Bin Center (Disjoint Bins)')
    axes[0].set_ylabel('Pearson Correlation with Transport Gap')
    axes[0].set_title('Correlation vs Energy (Disjoint)')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()
    
    axes[1].plot(cum_e_E0, cum_corr_E0, 'o-', label='E0')
    axes[1].plot(cum_e_E1, cum_corr_E1, 's-', label='E1')
    axes[1].plot(cum_e_E1_E0, cum_corr_E1_E0, '^-', label='E1 - E0')
    axes[1].set_xlabel('Energy Bin Edge (Cumulative Bins <= E)')
    axes[1].set_ylabel('Pearson Correlation with Transport Gap')
    axes[1].set_title('Correlation vs Energy (Cumulative)')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}.png")
    plt.close()

plot_combined_binned_correlations(df_master, BIN_WIDTH_E0, BIN_WIDTH_E1, BIN_WIDTH_E1_E0, 'corr_binned_combined')
# ---------------------------------------------------------
# Correlations Binned by Transport Gap
# ---------------------------------------------------------
def plot_correlations_binned_by_gap(df, bin_width, filename):
    min_val = df['gap'].min()
    max_val = df['gap'].max()
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    
    dis_centers = []
    dis_corr_E0, dis_corr_E1, dis_corr_E1_E0 = [], [] ,[]
    
    for i in range(len(bins)-1):
        mask = (df['gap'] >= bins[i]) & (df['gap'] < bins[i+1])
        sub_df = df[mask]
        if len(sub_df) > 5:
            c0 = sub_df['gap'].corr(sub_df['E0'])
            c1 = sub_df['gap'].corr(sub_df['E1'])
            cd = sub_df['gap'].corr(sub_df['E1_E0'])
            if not (np.isnan(c0) and np.isnan(c1) and np.isnan(cd)):
                dis_centers.append(bins[i] + bin_width/2.0)
                dis_corr_E0.append(c0)
                dis_corr_E1.append(c1)
                dis_corr_E1_E0.append(cd)

    cum_edges = []
    cum_corr_E0, cum_corr_E1, cum_corr_E1_E0 = [], [], []
    
    for edge in bins[1:]:
        mask = (df['gap'] <= edge)
        sub_df = df[mask]
        if len(sub_df) > 5:
            c0 = sub_df['gap'].corr(sub_df['E0'])
            c1 = sub_df['gap'].corr(sub_df['E1'])
            cd = sub_df['gap'].corr(sub_df['E1_E0'])
            if not (np.isnan(c0) and np.isnan(c1) and np.isnan(cd)):
                cum_edges.append(edge)
                cum_corr_E0.append(c0)
                cum_corr_E1.append(c1)
                cum_corr_E1_E0.append(cd)
                
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(dis_centers, dis_corr_E0, 'o-', label='E0')
    axes[0].plot(dis_centers, dis_corr_E1, 's-', label='E1')
    axes[0].plot(dis_centers, dis_corr_E1_E0, '^-', label='E1 - E0')
    axes[0].set_xlabel('Transport Gap (Disjoint Bins)')
    axes[0].set_ylabel('Pearson Correlation')
    axes[0].set_title('Correlation vs Transport Gap (Disjoint)')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()
    
    axes[1].plot(cum_edges, cum_corr_E0, 'o-', label='E0')
    axes[1].plot(cum_edges, cum_corr_E1, 's-', label='E1')
    axes[1].plot(cum_edges, cum_corr_E1_E0, '^-', label='E1 - E0')
    axes[1].set_xlabel('Transport Gap (Cumulative Bins <= Gap)')
    axes[1].set_ylabel('Pearson Correlation')
    axes[1].set_title('Correlation vs Transport Gap (Cumulative)')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}.png")
    plt.close()

plot_correlations_binned_by_gap(df_master, BIN_WIDTH_GAP, 'corr_binned_by_gap')

# ---------------------------------------------------------
# Probability of Gap and Energies being in the same bin
# ---------------------------------------------------------
def plot_same_bin_probability(df, var_name, bin_width, filename):
    min_val = min(df[var_name].min(), df['gap'].min())
    max_val = max(df[var_name].max(), df['gap'].max())

    bins_shared = np.arange(min_val, max_val + bin_width, bin_width)

    centers = []
    p_gap_given_var, p_var_given_gap = [], []
    p_gap_given_var_adj, p_var_given_gap_adj = [], []

    for i in range(len(bins_shared)-1):
        low = bins_shared[i]
        high = bins_shared[i+1]
        
        # Adjacent bins range
        low_adj = bins_shared[max(0, i-1)]
        high_adj = bins_shared[min(len(bins_shared)-1, i+2)]
        
        var_mask = (df[var_name] >= low) & (df[var_name] < high)
        gap_mask = (df['gap'] >= low) & (df['gap'] < high)
        
        var_mask_adj = (df[var_name] >= low_adj) & (df[var_name] < high_adj)
        gap_mask_adj = (df['gap'] >= low_adj) & (df['gap'] < high_adj)
        
        # Strict overlaps
        both_mask = var_mask & gap_mask
        
        # Adjacent overlaps
        both_var_gapadj = var_mask & gap_mask_adj
        both_gap_varadj = gap_mask & var_mask_adj
        
        count_var = var_mask.sum()
        count_gap = gap_mask.sum()
        
        centers.append(low + bin_width / 2.0)
        
        if count_var > 0:
            p_gap_given_var.append(both_mask.sum() / count_var)
            p_gap_given_var_adj.append(both_var_gapadj.sum() / count_var)
        else:
            p_gap_given_var.append(np.nan)
            p_gap_given_var_adj.append(np.nan)
            
        if count_gap > 0:
            p_var_given_gap.append(both_mask.sum() / count_gap)
            p_var_given_gap_adj.append(both_gap_varadj.sum() / count_gap)
        else:
            p_var_given_gap.append(np.nan)
            p_var_given_gap_adj.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(centers, p_gap_given_var, 'o-', label=f'P(Gap in bin | {var_name} in bin)', color='blue')
    ax.plot(centers, p_gap_given_var_adj, 'o--', label=f'P(Gap in same or adj bin | {var_name} in bin)', color='blue', alpha=0.6)
    
    ax.plot(centers, p_var_given_gap, 's-', label=f'P({var_name} in bin | Gap in bin)', color='red')
    ax.plot(centers, p_var_given_gap_adj, 's--', label=f'P({var_name} in same or adj bin | Gap in bin)', color='red', alpha=0.6)
    
    ax.set_xlabel('Bin Center (meV)')
    ax.set_ylabel('Probability')
    ax.set_title(f'Probability of {var_name} and Transport Gap falling in the same/adjacent bin')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}.png")
    plt.close()

plot_same_bin_probability(df_master, 'E0', BIN_WIDTH_E0, 'prob_same_bin_e0_gap')
plot_same_bin_probability(df_master, 'E1', BIN_WIDTH_E1, 'prob_same_bin_e1_gap')
plot_same_bin_probability(df_master, 'E1_E0', BIN_WIDTH_E1_E0, 'prob_same_bin_e1_e0_gap')

# ---------------------------------------------------------
# Conditional Probability (2w Measurement) - TOPOLOGICAL ONLY
# ---------------------------------------------------------
# Filter the conditional dataframe to ONLY include non-trivial topological regions
df_cond_master = df_cond_master[df_cond_master['is_topological'] == True]

min_e1 = df_cond_master['E1'].min()
max_e1 = df_cond_master['E1'].max()
bins_e1 = np.arange(min_e1, max_e1 + BIN_WIDTH_E1, BIN_WIDTH_E1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for thresh in E0_THRESHOLDS:
    df_filtered = df_cond_master[df_cond_master['E0'] < thresh]
    
    dis_centers, dis_prob = [], []
    for i in range(len(bins_e1)-1):
        mask = (df_filtered['E1'] >= bins_e1[i]) & (df_filtered['E1'] < bins_e1[i+1])
        sub_df = df_filtered[mask]
        if len(sub_df) > 0:
            dis_prob.append(sub_df['gapless'].mean())
            dis_centers.append(bins_e1[i] + BIN_WIDTH_E1/2.0)
            
    cum_edges, cum_prob = [], []
    for edge in bins_e1[1:]:
        mask = (df_filtered['E1'] <= edge)
        sub_df = df_filtered[mask]
        if len(sub_df) > 0:
            cum_prob.append(sub_df['gapless'].mean())
            cum_edges.append(edge)
            
    axes[0].plot(dis_centers, dis_prob, 'o-', label=f'E0 < {thresh}')
    axes[1].plot(cum_edges, cum_prob, 's-', label=f'E0 < {thresh}')

axes[0].set_xlabel('E1 (Disjoint Bins)')
axes[0].set_ylabel('P(Gapless | E0 < Thresh, E1 in bin, Topological)')
axes[0].set_title('Conditional Probability (Disjoint Bins)')
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].set_ylim(-0.05, 1.05)
axes[0].legend()

axes[1].set_xlabel('E1 (Cumulative Bins <= E1)')
axes[1].set_ylabel('P(Gapless | E0 < Thresh, E1 <= Edge, Topological)')
axes[1].set_title('Conditional Probability (Cumulative Bins)')
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/conditional_prob_gapless_multiple_E0.png")
plt.close()

print(f"\nAnalysis complete. Results saved in {OUTPUT_DIR}")
