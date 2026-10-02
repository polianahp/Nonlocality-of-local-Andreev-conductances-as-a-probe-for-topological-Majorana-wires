#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import argparse
import logging
import gc

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from src.config import PathConfigs
import src.helpers as hp
from src.parameter_handler import ConfigManager
from scipy.constants import physical_constants
from scipy.ndimage import convolve1d
from src.gpu_broadening import _temp_kernel

# ==========================================
# USER CONFIGURATION
# ==========================================
DEFAULT_DATA_DIRS = ["Tdis_pfaff5"] # Add your default target folders here

N_CUT_POINTS = 100                           # Number of points to sample along each cut
BARRIER_SWEEP_SCALE = 70                    # Scale multiplier for barrier sweeps

# Load official parameters from custom_protocol.yaml
p = ConfigManager.get_protocol_config(PathConfigs.PARAMETERS / "default_protocol.yaml")
DERIVATIVE_THRESHOLD = p.derivative_threshold

#Cuts and the points that should mathematically "snap" to them
#NOTE: All coordinates MUST be input as (Vz, mu)
CUTS = [
    {
        "start": (0.162, 2.391), "end": (1.4, 3.496), "color": "blue", "label": "Cut_1",
        "snap_points": [
            {"raw_coords": (0.954, 3.054), "color": "red", "label": "Point 1A"},

            {"raw_coords": (1.014, 3.114), "color": "cyan", "label": "Point 1B"},

            {"raw_coords": (0.893, 3.033), "color": "green", "label": "Point 1C"},

            {"raw_coords": (1.136, 3.234), "color": "yellow", "label": "Point 1D"},

            {"raw_coords": (1.38, 3.455), "color": "purple", "label": "Point 1E"},
            

        ]
    },

    {
        "start": (0.0, 2.511), "end": (1.4, 2.511), "color": "red", "label": "Cut_2",
        "snap_points": [
            {"raw_coords": (0.730, 2.511), "color": "red", "label": "Point 2A"},

            {"raw_coords": (0.568, 2.511), "color": "yellow", "label": "Point 2B"},

            {"raw_coords": (0.994, 2.511), "color": "green", "label": "Point 2C"},
            
            {"raw_coords": (0.852, 2.511), "color": "green", "label": "Point 2D"},
        ]
    },

        {
        "start": (0.35, 2.15), "end": (1.4, 3.134), "color": "green", "label": "Cut_3",
        "snap_points": [
            {"raw_coords": (1.319, 3.074), "color": "red", "label": "Point 3A"},

            {"raw_coords": (1.096, 2.813), "color": "yellow", "label": "Point 3B"},

            {"raw_coords": (0.771, 2.511), "color": "green", "label": "Point 3C"},
        ]
    }
]


# Points evaluated exactly as given (off-cut)
# NOTE: All coordinates MUST be input as (Vz, mu)
FREESTANDING_POINTS = [
    {"coords": (0.852, 2.511), "color": "purple", "label": "Point 2D"},
]

# ==========================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_point_path(pdi_data, N, resl, mu_start, mu_end, Vz_start, Vz_end):
    pdi_params = pdi_data[:, 0:2]
    diff_vec = np.array([mu_end - mu_start, Vz_end - Vz_start])
    total_distance = np.linalg.norm(diff_vec)

    if total_distance == 0:
        return np.array([]), np.array([])

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
            closest_indices.append(matched_indices[np.argmin(distances)])
        else:
            closest_indices.append(-1)

    valid_indices = np.array(closest_indices)[np.array(closest_indices) != -1]
    if len(valid_indices) == 0:
        return np.array([]), np.array([])

    unique_indices = np.unique(valid_indices)
    unique_points = pdi_params[unique_indices]
    dist_from_start = np.linalg.norm(unique_points - start_vec, axis=1)
    sort_order = np.argsort(dist_from_start)

    sorted_unique_points = unique_points[sort_order]
    sorted_unique_indices = unique_indices[sort_order]
    
    if N >= len(sorted_unique_points):
        return sorted_unique_points, sorted_unique_indices
    else:
        sample_idx = np.round(np.linspace(0, len(sorted_unique_points) - 1, N)).astype(int)
        return sorted_unique_points[sample_idx], sorted_unique_indices[sample_idx]

def export_phase_map_pair(name_prefix, z_left, z_right, z_inv, B_vals, V_vals, 
                          zmin, zmax, cmap_mpl, cmap_plotly, label_left, label_right, 
                          cbar_label, resolved_cuts, freestanding_points, out_dir,
                          contour_color='black', contour_dash='solid'):
    # Interactive HTML
    fig_html = make_subplots(rows=1, cols=2, subplot_titles=(label_left, label_right), shared_yaxes=True, horizontal_spacing=0.05)
    
    def add_html_overlays(col_idx):
        line_dict = dict(color=contour_color, width=2)
        if contour_dash != 'solid':
            line_dict['dash'] = contour_dash
        fig_html.add_trace(go.Contour(z=z_inv, x=B_vals, y=V_vals, contours=dict(start=0, end=0, size=1), contours_coloring='lines', line=line_dict, showscale=False, hoverinfo='skip'), row=1, col=col_idx)
        for rcut in resolved_cuts:
            cut = rcut['config']
            fig_html.add_trace(go.Scatter(x=[cut['start'][1], cut['end'][1]], y=[cut['start'][0], cut['end'][0]], mode='lines', line=dict(color=cut['color'], width=1.5), name=cut['label'], showlegend=(col_idx==1)), row=1, col=col_idx)
            for sp in rcut['resolved_snaps']:
                fig_html.add_trace(go.Scatter(x=[sp['raw_coords'][1]], y=[sp['raw_coords'][0]], mode='markers', marker=dict(symbol='circle-open', color=sp['color'], size=5, line=dict(width=1)), name=f"{sp['label']} (Raw)", showlegend=False), row=1, col=col_idx)
                fig_html.add_trace(go.Scatter(x=[sp['snapped_coords'][1]], y=[sp['snapped_coords'][0]], mode='markers', marker=dict(symbol='star', color=sp['color'], size=6), name=f"{sp['label']} (Snapped)", showlegend=False), row=1, col=col_idx)
                fig_html.add_trace(go.Scatter(x=[sp['raw_coords'][1], sp['snapped_coords'][1]], y=[sp['raw_coords'][0], sp['snapped_coords'][0]], mode='lines', line=dict(color=sp['color'], width=0.8, dash='dot'), showlegend=False, hoverinfo='skip'), row=1, col=col_idx)
        for fp in freestanding_points:
            fig_html.add_trace(go.Scatter(x=[fp['coords'][1]], y=[fp['coords'][0]], mode='markers', marker=dict(symbol='star', color=fp['color'], size=6), name=fp['label'], showlegend=(col_idx==1)), row=1, col=col_idx)

    hover_temp = "<b>B (Vz)</b>: %{x:.3f} meV<br><b>V (mu)</b>: %{y:.3f} meV<br><b>Value</b>: %{z:.3f}<extra></extra>"
    fig_html.add_trace(go.Heatmap(z=z_left, x=B_vals, y=V_vals, zmin=zmin, zmax=zmax, colorscale=cmap_plotly, showscale=False, hovertemplate=hover_temp), row=1, col=1)
    fig_html.add_trace(go.Heatmap(z=z_right, x=B_vals, y=V_vals, zmin=zmin, zmax=zmax, colorscale=cmap_plotly, colorbar=dict(title=cbar_label), hovertemplate=hover_temp), row=1, col=2)
    add_html_overlays(1)
    add_html_overlays(2)

    fig_html.update_layout(title=f"Interactive Phase Map: {cbar_label}", xaxis_title="Zeeman Field Vz (B) [meV]", yaxis_title="Chemical Potential µ (V) [meV]", xaxis2_title="Zeeman Field Vz (B) [meV]", width=1200, height=700, legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.8)'))
    fig_html.write_html(str(out_dir / f"{name_prefix}_phase_map.html"))

    # Static PNG
    fig_map, axes = plt.subplots(1, 2, figsize=(15.5, 8.5), dpi=300, sharey=True, layout='constrained')
    im1 = axes[0].pcolormesh(B_vals, V_vals, z_left, cmap=cmap_mpl, vmin=zmin, vmax=zmax, shading='nearest')
    im2 = axes[1].pcolormesh(B_vals, V_vals, z_right, cmap=cmap_mpl, vmin=zmin, vmax=zmax, shading='nearest')
    
    for ax in axes:
        ax.contour(B_vals, V_vals, z_inv, levels=[0], colors=contour_color, linewidths=1.5, linestyles='dashed' if contour_dash != 'solid' else 'solid')
        for rcut in resolved_cuts:
            cut = rcut['config']
            ax.plot([cut['start'][1], cut['end'][1]], [cut['start'][0], cut['end'][0]], color=cut['color'], linewidth=1.2, label=cut['label'])
            for sp in rcut['resolved_snaps']:
                ax.plot(sp['raw_coords'][1], sp['raw_coords'][0], marker='o', markerfacecolor='none', markeredgecolor=sp['color'], markersize=4, linestyle='None')
                ax.plot(sp['snapped_coords'][1], sp['snapped_coords'][0], marker='*', color=sp['color'], markersize=6, linestyle='None')
                ax.plot([sp['raw_coords'][1], sp['snapped_coords'][1]], [sp['raw_coords'][0], sp['snapped_coords'][0]], color=sp['color'], linestyle=':', linewidth=0.8)
        for fp in freestanding_points:
            ax.plot(fp['coords'][1], fp['coords'][0], marker='*', color=fp['color'], markersize=6, label=fp['label'], linestyle='None')
        ax.set_xlabel(r"Zeeman Field $V_z$ (meV)")
        
    axes[0].set_ylabel(r"Chemical Potential $\mu$ (meV)")
    axes[0].set_title(label_left)
    axes[1].set_title(label_right)
    axes[0].legend(loc='upper left', framealpha=0.8)
    fig_map.colorbar(im2, ax=axes, label=cbar_label)
    fig_map.suptitle(f"Global Phase Map: {cbar_label}")
    fig_map.savefig(out_dir / f"{name_prefix}_phase_map.png", bbox_inches='tight')
    plt.close(fig_map)



def export_phase_map_single(name_prefix, z_data, z_inv, B_vals, V_vals, 
                            zmin, zmax, cmap_mpl, cmap_plotly, label, 
                            cbar_label, resolved_cuts, freestanding_points, out_dir,
                            contour_color='black', contour_dash='solid'):
    # Interactive HTML
    fig_html = go.Figure()
    
    hover_temp = "<b>B (Vz)</b>: %{x:.3f} meV<br><b>V (mu)</b>: %{y:.3f} meV<br><b>Value</b>: %{z:.3f}<extra></extra>"
    fig_html.add_trace(go.Heatmap(z=z_data, x=B_vals, y=V_vals, zmin=zmin, zmax=zmax, colorscale=cmap_plotly, colorbar=dict(title=cbar_label), hovertemplate=hover_temp))

    line_dict = dict(color=contour_color, width=2)
    if contour_dash != 'solid':
        line_dict['dash'] = contour_dash
    fig_html.add_trace(go.Contour(z=z_inv, x=B_vals, y=V_vals, contours=dict(start=0, end=0, size=1), contours_coloring='lines', line=line_dict, showscale=False, hoverinfo='skip'))
    
    for rcut in resolved_cuts:
        cut = rcut['config']
        fig_html.add_trace(go.Scatter(x=[cut['start'][1], cut['end'][1]], y=[cut['start'][0], cut['end'][0]], mode='lines', line=dict(color=cut['color'], width=1.5), name=cut['label'], showlegend=True))
        for sp in rcut['resolved_snaps']:
            fig_html.add_trace(go.Scatter(x=[sp['raw_coords'][1]], y=[sp['raw_coords'][0]], mode='markers', marker=dict(symbol='circle-open', color=sp['color'], size=5, line=dict(width=1)), name=f"{sp['label']} (Raw)", showlegend=False))
            fig_html.add_trace(go.Scatter(x=[sp['snapped_coords'][1]], y=[sp['snapped_coords'][0]], mode='markers', marker=dict(symbol='star', color=sp['color'], size=6), name=f"{sp['label']} (Snapped)", showlegend=False))
            fig_html.add_trace(go.Scatter(x=[sp['raw_coords'][1], sp['snapped_coords'][1]], y=[sp['raw_coords'][0], sp['snapped_coords'][0]], mode='lines', line=dict(color=sp['color'], width=0.8, dash='dot'), showlegend=False, hoverinfo='skip'))
    for fp in freestanding_points:
        fig_html.add_trace(go.Scatter(x=[fp['coords'][1]], y=[fp['coords'][0]], mode='markers', marker=dict(symbol='star', color=fp['color'], size=6), name=fp['label'], showlegend=True))

    fig_html.update_layout(title=f"Interactive Phase Map: {cbar_label}", xaxis_title="Zeeman Field Vz (B) [meV]", yaxis_title="Chemical Potential µ (V) [meV]", width=800, height=700, legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.8)'))
    fig_html.write_html(str(out_dir / f"{name_prefix}_phase_map.html"))

    # Static PNG
    fig_map, ax = plt.subplots(figsize=(8.5, 8.5), dpi=300, layout='constrained')
    im1 = ax.pcolormesh(B_vals, V_vals, z_data, cmap=cmap_mpl, vmin=zmin, vmax=zmax, shading='nearest')
    
    ax.contour(B_vals, V_vals, z_inv, levels=[0], colors=contour_color, linewidths=1.5, linestyles='dashed' if contour_dash != 'solid' else 'solid')
    for rcut in resolved_cuts:
        cut = rcut['config']
        ax.plot([cut['start'][1], cut['end'][1]], [cut['start'][0], cut['end'][0]], color=cut['color'], linewidth=1.2, label=cut['label'])
        for sp in rcut['resolved_snaps']:
            ax.plot(sp['raw_coords'][1], sp['raw_coords'][0], marker='o', markerfacecolor='none', markeredgecolor=sp['color'], markersize=4, linestyle='None')
            ax.plot(sp['snapped_coords'][1], sp['snapped_coords'][0], marker='*', color=sp['color'], markersize=6, linestyle='None')
            ax.plot([sp['raw_coords'][1], sp['snapped_coords'][1]], [sp['raw_coords'][0], sp['snapped_coords'][0]], color=sp['color'], linestyle=':', linewidth=0.8)
    for fp in freestanding_points:
        ax.plot(fp['coords'][1], fp['coords'][0], marker='*', color=fp['color'], markersize=6, label=fp['label'], linestyle='None')
    
    ax.set_xlabel(r"Zeeman Field $V_z$ (meV)")
    ax.set_ylabel(r"Chemical Potential $\mu$ (meV)")
    ax.set_title(label)
    ax.legend(loc='upper left', framealpha=0.8)
    fig_map.colorbar(im1, ax=ax, label=cbar_label)
    fig_map.suptitle(f"Global Phase Map: {cbar_label}")
    fig_map.savefig(out_dir / f"{name_prefix}_phase_map.png", bbox_inches='tight')
    plt.close(fig_map)

def process_data_dir(data_dir):
    OUT_DIR = PathConfigs.PLOTS / f"{data_dir.name}_Plots"
    logger.info(f"Creating output directory: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TPREP_PATH = data_dir / "tprep.nc"

    # Convert all user inputs from (Vz, mu) to internal (mu, Vz) convention
    # Make deep copies so we don't mutate the global config for subsequent runs
    import copy
    local_cuts = copy.deepcopy(CUTS)
    local_freestanding = copy.deepcopy(FREESTANDING_POINTS)

    for cut in local_cuts:
        cut['start'] = (cut['start'][1], cut['start'][0])
        cut['end'] = (cut['end'][1], cut['end'][0])
        for sp in cut.get('snap_points', []):
            sp['raw_coords'] = (sp['raw_coords'][1], sp['raw_coords'][0])
            
    for fp in local_freestanding:
        fp['coords'] = (fp['coords'][1], fp['coords'][0])

    # 1. Load Data
    params = np.load(data_dir / "all_params.npz", allow_pickle=True)
    pdi_data = np.load(data_dir / "pdi_data.npy", allow_pickle=True)

    if not TPREP_PATH.exists():
        logger.info(f"tprep.nc not found in {data_dir}. Generating it now...")
        from src.tgp_adapter import TGPAdapter
        from src.gpu_broadening import prepare_sim_gpu
        data = TGPAdapter(data_dir).to_xarray()
        tprep = prepare_sim_gpu(data, p.T_mK)
        
        tmp_path = TPREP_PATH.with_suffix('.nc.tmp')
        tprep.to_netcdf(tmp_path)
        tmp_path.rename(TPREP_PATH)
        logger.info("tprep.nc successfully generated.")
        
    tprep = xr.open_dataset(TPREP_PATH)
    
    # Dynamically extract transport gap using tgp
    try:
        import tgp
        tprep_left = tprep.rename({"bias": "left_bias"})
        tprep_right = tprep.rename({"bias": "right_bias"})
        upper_th = float('inf') if p.upper_conductance_threshold is None else p.upper_conductance_threshold
        tprep_left, tprep_right = tgp.two.extract_gap(
            tprep_left, 
            tprep_right,
            gap_threshold_factor=p.gap_threshold_factor,
            upper_conductance_threshold=upper_th,
            noise_threshold=p.noise_threshold
        )
        gap_left_avg = tprep_left.gap.mean(dim='cutter_pair_index')
        gap_right_avg = tprep_right.gap.mean(dim='cutter_pair_index')
        has_tgp_gap = True
    except ImportError:
        logger.info("Warning: 'tgp' module not found. Transport gap will be plotted as zeros.")
        has_tgp_gap = False

    # Extract physics params
    t_val = float(params['t'])
    mu_n = float(params['mu_n'])
    mu_leads = float(params['mu_leads'])
    Delta0 = float(params['Delta0'])
    gamma = float(params['gamma'])
    alpha = float(params['alpha'])
    Ln = int(params['Ln'])
    Lb = int(params['Lb'])
    Ls = int(params['Ls'])
    V0 = float(params['V0'])
    barrier_l_base = float(params['barrier0'])
    Vdisx = params['Vdisx'] * V0

    # 2. Resolve Cut Paths and Snapped Points
    logger.info("Resolving cuts and snapping points...")
    resolved_cuts = []
    points_to_analyze = []
    
    for cut in local_cuts:
        pts, sampled_indices = generate_point_path(pdi_data, N_CUT_POINTS, 0.02, cut['start'][0], cut['end'][0], cut['start'][1], cut['end'][1])
        actual_N = len(pts)
        
        resolved_snaps = []
        if actual_N > 0:
            for sp in cut.get('snap_points', []):
                raw_coords = np.array(sp['raw_coords'])
                snap_idx = np.argmin(np.linalg.norm(pts - raw_coords, axis=1))
                snapped_coords = pts[snap_idx]
                
                resolved_snaps.append({
                    "raw_coords": raw_coords,
                    "snapped_coords": snapped_coords,
                    "snap_idx": snap_idx,
                    "color": sp['color'],
                    "label": sp['label']
                })
                points_to_analyze.append({
                    "coords": tuple(snapped_coords),
                    "color": sp['color'],
                    "label": sp['label'],
                    "dir_path": OUT_DIR / "Cuts" / cut['label'] / "Points" / sp['label']
                })
                
        resolved_cuts.append({
            "config": cut,
            "pts": pts,
            "sampled_indices": sampled_indices,
            "resolved_snaps": resolved_snaps,
            "actual_N": actual_N
        })

    for fp in local_freestanding:
        fp_dict = fp.copy()
        fp_dict['dir_path'] = OUT_DIR / "Freestanding_Points" / fp['label']
        points_to_analyze.append(fp_dict)

    # 3. Global Phase Maps (Plotly HTML & Matplotlib PNG)
    logger.info("Generating Global Phase Maps (HTML & PNG)...")
    
    B_vals = tprep['B'].values
    V_vals = tprep['V'].values
    z_inv = tprep['L_SI'].mean(dim='cutter_pair_index').transpose('V', 'B').values
    
    # A. 2w Phase Map
    export_phase_map_pair(
        "global_2w",
        tprep['L_2w_nl'].mean(dim='cutter_pair_index').transpose('V', 'B').values,
        tprep['R_2w_nl'].mean(dim='cutter_pair_index').transpose('V', 'B').values,
        z_inv, B_vals, V_vals, -2.0, 2.0, 'RdBu_r', 'RdBu_r',
        "Average Left 2ω", "Average Right 2ω", "2ω Conductance",
        resolved_cuts, local_freestanding, OUT_DIR
    )

    # B. 3w Phase Map
    zrng = DERIVATIVE_THRESHOLD * 1.3
    export_phase_map_pair(
        "global_3w",
        tprep['L_3w'].mean(dim='cutter_pair_index').transpose('V', 'B').values,
        tprep['R_3w'].mean(dim='cutter_pair_index').transpose('V', 'B').values,
        z_inv, B_vals, V_vals, -zrng, zrng, 'RdBu_r', 'RdBu_r',
        "Average Left 3ω", "Average Right 3ω", "3ω Curvature",
        resolved_cuts, local_freestanding, OUT_DIR
    )

    # C. Transport Gap Phase Map
    if has_tgp_gap:
        export_phase_map_pair(
            "global_transport_gap",
            gap_left_avg.transpose('V', 'B').values,
            gap_right_avg.transpose('V', 'B').values,
            z_inv, B_vals, V_vals, 0.0, 0.05, 'gist_heat_r', 'hot_r',
            "Gap from G_RL (Left)", "Gap from G_LR (Right)", "Extracted Gap (µV)",
            resolved_cuts, local_freestanding, OUT_DIR,
            contour_color='dimgray', contour_dash='dash'
        )

    # D. Pfaffian Invariant Phase Map
    export_phase_map_single(
        "global_pfaffian",
        z_inv,
        z_inv, B_vals, V_vals, -1.1, 1.1, 'gray', 'gray',
        "Pfaffian Invariant", "Pfaffian Sign",
        resolved_cuts, local_freestanding, OUT_DIR,
        contour_color='cyan'
    )

    # 4. Cut Analysis (Multi-panel plotting)
    logger.info(f"Analyzing {len(resolved_cuts)} cuts...")
    L_2w_avg = tprep['L_2w_nl'].mean(dim='cutter_pair_index')
    invariant_avg = tprep['L_SI'].mean(dim='cutter_pair_index')
    
    for rcut in resolved_cuts:
        cut = rcut['config']
        actual_N = rcut['actual_N']
        pts = rcut['pts']
        
        if actual_N == 0:
            logger.info(f"  Skipping {cut['label']}: No points found along path.")
            continue
            
        cut_dir = OUT_DIR / "Cuts" / cut['label']
        cut_dir.mkdir(parents=True, exist_ok=True)
        
        kvals = 14
        evals = np.zeros((actual_N, kvals))
        val_2w = np.zeros(actual_N)
        val_3w = np.zeros(actual_N)
        pfaffians = np.zeros(actual_N)
        val_gap_left = np.zeros(actual_N)
        val_gap_right = np.zeros(actual_N)
        
        for i, (mu_val, vz_val) in enumerate(pts):
            scl = hp.build_system_closed(t_val, mu_val, gamma, Delta0, vz_val, alpha, Ls, Vdisx, a=1)
            evals[i, :] = hp.calc_spectrum(scl, k=kvals)
            
            val_2w[i] = L_2w_avg.sel(V=mu_val, B=vz_val, method='nearest').values.item()
            val_3w[i] = tprep['L_3w'].mean(dim='cutter_pair_index').sel(V=mu_val, B=vz_val, method='nearest').values.item()
            pfaffians[i] = invariant_avg.sel(V=mu_val, B=vz_val, method='nearest').values.item()

            if has_tgp_gap:
                val_gap_left[i] = gap_left_avg.sel(V=mu_val, B=vz_val, method='nearest').values.item()
                val_gap_right[i] = gap_right_avg.sel(V=mu_val, B=vz_val, method='nearest').values.item()

        fig, axes = plt.subplots(4, 1, figsize=(7.5, 8.8), gridspec_kw={'height_ratios': [2, 1, 1, 1]}, sharex=True)
        
        # Shading
        for ax in axes:
            for idx in range(actual_N):
                if pfaffians[idx] > 0: 
                    ax.axvspan(idx - 0.5, idx + 0.5, color='lightgrey', alpha=0.5, zorder=0)

        # Draw Snapped Point Vertical Lines
        for sp in rcut['resolved_snaps']:
            for ax in axes:
                ax.axvline(sp['snap_idx'], color=sp['color'], linestyle='--', linewidth=1, alpha=0.7)

        # Panel 1: Spectra
        mid_idx = kvals // 2
        for j in range(kvals):
            color = 'red' if j in [mid_idx-1, mid_idx] else 'royalblue'
            lw = 2.5 if j in [mid_idx-1, mid_idx] else 1.5
            axes[0].plot(range(actual_N), evals[:, j], color=color, linewidth=lw, alpha=0.8, zorder=2)
        axes[0].axhline(0, color='black', linestyle='--', alpha=0.6)
        axes[0].set_ylabel("Energy (meV)")
        axes[0].set_title(f"Spectra along {cut['label']}")

        # Panel 2: 2w (Capped to match phase map)
        cap_2w = 2.0
        val_2w_capped = np.clip(val_2w, -cap_2w, cap_2w)
        axes[1].plot(range(actual_N), val_2w_capped, color='forestgreen', linewidth=1.5, marker='o', markersize=3, zorder=2)
        axes[1].set_ylim(-cap_2w, cap_2w)
        axes[1].set_ylabel("2ω Conductance")
        axes[1].set_title(f"Signed 2ω Measurement (Capped to ±{cap_2w})")

        # Panel 3: 3w (Capped)
        cap_val = 1.4 * DERIVATIVE_THRESHOLD
        val_3w_capped = np.clip(val_3w, -cap_val, cap_val)
        
        axes[2].plot(range(actual_N), val_3w_capped, color='darkorange', linewidth=1.5, marker='s', markersize=3, zorder=2)
        axes[2].axhline(-DERIVATIVE_THRESHOLD, color='black', linestyle=':', linewidth=1.5, label='ZBP Threshold', zorder=3)
        
        axes[2].set_ylim(-cap_val, cap_val)
        axes[2].set_ylabel("3ω Curvature")
        axes[2].set_title(f"3ω Curvature (Strictly Capped to ±{cap_val:.3f})")
        axes[2].legend(loc='upper right')

        # Panel 4: Transport Gap and Lowest States
        axes[3].plot(range(actual_N), np.minimum(val_gap_left, val_gap_right), color='purple', linewidth=1.5, marker='^', markersize=3, zorder=2, label=r"Min $\Delta_{ex}$")
        axes[3].plot(range(actual_N), evals[:, mid_idx], color='red', linewidth=1.5, linestyle='--', zorder=1, label=r"$E_0$")
        axes[3].plot(range(actual_N), evals[:, mid_idx + 1], color='blue', linewidth=1.5, linestyle='--', zorder=1, label=r"$E_1$")
        axes[3].set_ylabel(r"Gap / Energy (meV)")
        axes[3].set_title("Transport Gap & Lowest Spectral States")
        axes[3].legend(loc='upper right')

        # X-axes (Dual)
        tick_indices = np.round(np.linspace(0, actual_N - 1, min(10, actual_N))).astype(int)
        axes[3].set_xticks(tick_indices)
        axes[3].set_xticklabels([f"{pts[idx, 0]:.3f}" for idx in tick_indices], rotation=45, ha='right')
        axes[3].set_xlabel(r"Chemical Potential $\mu$ (meV)")
        
        ax_top = axes[0].twiny()
        ax_top.set_xlim(axes[0].get_xlim())
        ax_top.set_xticks(tick_indices)
        ax_top.set_xticklabels([f"{pts[idx, 1]:.3f}" for idx in tick_indices], rotation=45, ha='left')
        ax_top.set_xlabel(r"Zeeman Field $V_z$ (meV)")

        plt.tight_layout()
        fig.savefig(cut_dir / "multi_panel.png", dpi=300)
        plt.close(fig)

    # 5. Point Analysis (Unified loop)
    logger.info(f"Analyzing {len(points_to_analyze)} deep-dive points...")
    for pt in points_to_analyze:
        pt_dir = pt['dir_path']
        pt_dir.mkdir(parents=True, exist_ok=True)
        mu_val, vz_val = pt['coords']
        
        # A. dI/dV
        logger.info(f"  Calculating dI/dV for {pt['label']}...")
        syst = hp.build_system(t_val, mu_val, mu_n, gamma, Delta0, vz_val, alpha, Ln, Lb, Ls, mu_leads, barrier_l_base, barrier_l_base, Vdisx)
        energies = np.linspace(-0.5, 0.5, 101)
        dIdV_left, dIdV_right, _, _, _ = hp.calc_dIdV(syst, energies)
        
        # Apply Thermal Broadening if T_mK > 0
        k_B = physical_constants["Boltzmann constant in eV/K"][0]
        T_meV = k_B * (p.T_mK * 1e-3) * 1e3
        if T_meV > 0.0:
            K = _temp_kernel(energies, T_meV)
            delta_bias = np.diff(energies)[0]
            K_norm = K * delta_bias
            dIdV_left = convolve1d(dIdV_left, K_norm, mode='constant', cval=0.0)
            dIdV_right = convolve1d(dIdV_right, K_norm, mode='constant', cval=0.0)

        
        fig, ax = plt.subplots(figsize=(8.2, 2.8))
        ax.plot(energies, dIdV_left, label="Left dI/dV", color='royalblue')
        ax.plot(energies, dIdV_right, label="Right dI/dV", color='darkorange')
        
        # Add a dummy trace to the legend so the map color is explicitly visible
        ax.plot([], [], color=pt['color'], marker='*', markersize=10, linestyle='None', label=f"Map Marker ({pt['color']})")
        
        # Extract local 2w and 3w values (using the Left side averages as representative)
        pt_2w = L_2w_avg.sel(V=mu_val, B=vz_val, method='nearest').values.item()
        pt_3w = tprep['L_3w'].mean(dim='cutter_pair_index').sel(V=mu_val, B=vz_val, method='nearest').values.item()
        
        ax.set_title(f"dI/dV for {pt['label']} [Color: {pt['color']}]\n(mu={mu_val:.3f}, Vz={vz_val:.3f}) | 2ω_L={pt_2w:.3f}, 3ω_L={pt_3w:.3f}")
        ax.set_xlabel("Energy (meV)")
        ax.set_ylabel("Conductance")
        ax.legend()
        plt.tight_layout()
        fig.savefig(pt_dir / f"{pt['label']}_{pt['color']}_dIdV.png", dpi=150)
        plt.close(fig)

        # B. Majorana Wave Functions
        logger.info(f"  Calculating Wavefunctions for {pt['label']}...")
        scl = hp.build_system_closed(t_val, mu_val, gamma, Delta0, vz_val, alpha, Ls, Vdisx, a=1)
        H_full = scl.hamiltonian_submatrix(sparse=False)
        evals_full, evecs_full = np.linalg.eigh(H_full)
        rho_M1, rho_M2, _ = hp.get_psiM_density(evals_full, evecs_full)
        
        fig, ax = plt.subplots(figsize=(8.2, 2.8))
        ax.plot(rho_M1, label="Majorana Left (M1)", color='cyan')
        ax.plot(rho_M2, label="Majorana Right (M2)", color='orange')
        ax.set_title(f"Wavefunctions for {pt['label']}")
        ax.set_xlabel("Site Index")
        ax.set_ylabel("Probability Density")
        ax.legend()
        plt.tight_layout()
        fig.savefig(pt_dir / f"{pt['label']}_{pt['color']}_wavefunctions.png", dpi=150)
        plt.close(fig)

        # C. Barrier Sweeps
        logger.info(f"  Calculating Barrier Sweeps for {pt['label']}...")
        barrier_sweep_vals = np.linspace(-BARRIER_SWEEP_SCALE * barrier_l_base, BARRIER_SWEEP_SCALE * barrier_l_base, 50)
        
        gL_varying_R, gR_varying_R = [], []
        for br in barrier_sweep_vals:
            s = hp.build_system(t_val, mu_val, mu_n, gamma, Delta0, vz_val, alpha, Ln, Lb, Ls, mu_leads, barrier_l_base, br, Vdisx)
            cL, cR = hp.calc_conductance(s, energy=0.0)
            gL_varying_R.append(cL)
            gR_varying_R.append(cR)
            
        gL_varying_L, gR_varying_L = [], []
        for bl in barrier_sweep_vals:
            s = hp.build_system(t_val, mu_val, mu_n, gamma, Delta0, vz_val, alpha, Ln, Lb, Ls, mu_leads, bl, barrier_l_base, Vdisx)
            cL, cR = hp.calc_conductance(s, energy=0.0)
            gL_varying_L.append(cL)
            gR_varying_L.append(cR)
            
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.8))
        
        x_data_R = barrier_sweep_vals / barrier_l_base
        axes[0].plot(x_data_R, np.array(gL_varying_R) / gL_varying_R[len(gL_varying_R)//2], label="G_LL", color='green')
        axes[0].plot(x_data_R, np.array(gR_varying_R) / gR_varying_R[len(gR_varying_R)//2], label="G_RR", color='blue')
        axes[0].set_title(f"Varying Right Barrier ({pt['label']})")
        axes[0].set_xlabel("U_R / U_L")
        axes[0].set_ylabel("Normalized Conductance")
        axes[0].legend()
        
        x_data_L = barrier_sweep_vals / barrier_l_base
        axes[1].plot(x_data_L, np.array(gL_varying_L) / gL_varying_L[len(gL_varying_L)//2], label="G_LL", color='green')
        axes[1].plot(x_data_L, np.array(gR_varying_L) / gR_varying_L[len(gR_varying_L)//2], label="G_RR", color='blue')
        axes[1].set_title(f"Varying Left Barrier ({pt['label']})")
        axes[1].set_xlabel("U_L / U_R")
        axes[1].set_ylabel("Normalized Conductance")
        axes[1].legend()
        
        plt.tight_layout()
        fig.savefig(pt_dir / f"{pt['label']}_{pt['color']}_barrier_asymmetry.png", dpi=150)
        plt.close(fig)

    logger.info("Done!")

    # Explicitly free memory for sequential processing
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
    except ImportError:
        pass
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Analyze cuts for a series of data_dir folders")
    parser.add_argument("data_dirs", nargs="*", help="List of active data directory names (relative to PathConfigs.DATA) or absolute paths")
    args = parser.parse_args()

    if not args.data_dirs:
        # Default behavior
        data_dirs_to_process = DEFAULT_DATA_DIRS
    else:
        data_dirs_to_process = args.data_dirs

    for d in data_dirs_to_process:
        p_dir = Path(d)
        if not p_dir.is_absolute():
            p_dir = PathConfigs.DATA / d
        
        if not p_dir.exists():
            logger.error(f"Directory {p_dir} does not exist. Skipping.")
            continue
            
        try:
            logger.info(f"--- Processing Directory: {p_dir} ---")
            process_data_dir(p_dir)
        except Exception as e:
            logger.error(f"Failed processing {p_dir}: {e}", exc_info=True)
            continue

if __name__ == "__main__":

    main()
