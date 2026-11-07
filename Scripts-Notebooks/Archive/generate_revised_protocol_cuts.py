#!/usr/bin/env python3
"""
generate_revised_protocol_cuts.py

Comprehensive cut analysis and step-by-step protocol visualization.
Completely driven by a reproducible YAML configuration file (zero hardcoded parameters).

Outputs:
1. Protocol step-by-step comparison map (2x2 grid):
   - Panel 1: Regular TGP (Stage 2 ROI 2)
   - Panel 2: S-ACD Stage 1 (Asymmetric Conductance Deviation + Lead Symmetry Gate)
   - Panel 3: S-ACD Stage 2 (Multi-Cutter 3w Curvature Agreement)
   - Panel 4: Subtractive S-ACD (Non-Local Gap Gate + Subtractive Island Filtering)
2. Final passing phase map with labeled clusters, dashed Pfaffian boundary, and marked Medoid stars.
3. Multi-panel cut diagnostic figures for each cluster Medoid (actual passing pixel):
   - Left Column: BdG Spectrum along cut, G_LL(V, Vz), G_RR(V, Vz), A(G_RL), A(G_LR) with pixel-perfect axis alignment.
   - Right Column: dI/dV at Medoid, Majorana wavefunctions rho_M1 & rho_M2 at Medoid, and dual barrier sweeps.
4. Summary JSON cataloging cluster properties, medoid coordinates, curvatures, and localization metrics.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import label
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "Archive") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "Archive"))

import src.helpers as hp
from src.tgp_adapter import TGPAdapter
from src.cpu_broadening import prepare_sim_cpu
import tgp.two

from post_process_disorder_realizations import (
    compute_sacd_stage1,
    compute_sacd_stage2_curvature,
    compute_sacd_stage3_gap,
    apply_subtractive_island_filter,
    calc_protocol_metrics,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RevisedProtocolCuts")

# ==============================================================================
# TARGET CONFIGURATION FILE (YAML)
# ==============================================================================
# Specify your target configuration YAML file here.
# You can also override it by passing a file or directory as the first CLI argument:
#   python generate_revised_protocol_cuts.py [path/to/protocol_config.yaml]
#   python generate_revised_protocol_cuts.py [path/to/disorder_realization_dir]
TARGET_CONFIG_PATH = REPO_ROOT / "Inputs" / "Parameters" / "protocol_config.yaml"


def load_yaml_config(config_target: Union[str, Path]) -> Dict[str, Any]:
    """Loads and validates protocol configuration from a YAML file."""
    cpath = Path(config_target).resolve()
    if not cpath.exists():
        raise FileNotFoundError(f"Configuration YAML file not found: {cpath}")
    logger.info("Loading configuration from: %s", cpath)
    with open(cpath, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    # --------------------------------------------------------------------------
    # 1. Resolve Target Configuration & Data Path
    # --------------------------------------------------------------------------
    override_data_dir = None
    if len(sys.argv) > 1:
        cli_arg = sys.argv[1].strip()
        arg_path = Path(cli_arg)
        if not arg_path.is_absolute():
            arg_path = (REPO_ROOT / arg_path).resolve()

        if arg_path.is_file() and arg_path.suffix.lower() in [".yaml", ".yml"]:
            active_config_path = arg_path
        elif arg_path.is_dir():
            active_config_path = TARGET_CONFIG_PATH
            override_data_dir = arg_path
            logger.info("Target data directory overridden from CLI: %s", override_data_dir)
        else:
            active_config_path = arg_path
    else:
        active_config_path = TARGET_CONFIG_PATH

    cfg = load_yaml_config(active_config_path)

    # --- Unpack Target Section ---
    tgt_cfg = cfg.get("target", {})
    if override_data_dir is not None:
        data_dir = override_data_dir
    else:
        raw_data_dir = tgt_cfg.get("data_dir", "Outputs/Data/Disorder_Realizations/V0_0_872/disorder_realization_17")
        data_dir = Path(raw_data_dir)
        if not data_dir.is_absolute():
            data_dir = (REPO_ROOT / data_dir).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Target data directory does not exist: {data_dir}")

    temp_mK = float(tgt_cfg.get("temp_mK", 30.0))

    if tgt_cfg.get("disorder_name"):
        disorder_name = str(tgt_cfg["disorder_name"])
    else:
        if data_dir.parent.name.startswith("V0_"):
            disorder_name = f"{data_dir.parent.name}_{data_dir.name}"
        else:
            disorder_name = data_dir.name

    if tgt_cfg.get("plots_dir"):
        plots_dir = Path(tgt_cfg["plots_dir"])
        if not plots_dir.is_absolute():
            plots_dir = (REPO_ROOT / plots_dir).resolve()
    else:
        plots_dir = REPO_ROOT / "Plots" / f"revised_protocol_cuts_{disorder_name}_{int(temp_mK)}mK"

    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Target Data Directory: %s", data_dir)
    logger.info("Output Plots Directory: %s", plots_dir)

    # --- Unpack S-ACD Protocol Section ---
    sacd_cfg = cfg.get("sacd_protocol", {})
    s1_cfg = sacd_cfg.get("stage1", {})
    tau = float(s1_cfg.get("tau", 0.01))
    s_max = float(s1_cfg.get("s_max", 0.20))
    floor_val = float(s1_cfg.get("floor_val", 0.001))
    sym_barrier_val = float(s1_cfg.get("sym_barrier_val", 0.0))

    s2_cfg = sacd_cfg.get("stage2_curvature", {})
    d2_threshold = float(s2_cfg.get("d2_threshold", 100.0))
    min_cutters_consensus = int(s2_cfg.get("min_cutters_consensus", 3))
    savgol_window_length = int(s2_cfg.get("savgol_window_length", 7))
    savgol_polyorder = int(s2_cfg.get("savgol_polyorder", 2))
    bias_window_min = float(s2_cfg.get("bias_window_min_mev", -0.134))
    bias_window_max = float(s2_cfg.get("bias_window_max_mev", 0.134))
    bias_num_points = int(s2_cfg.get("bias_num_points", 97))
    zero_bias_index = int(s2_cfg.get("zero_bias_index", 48))

    s3_cfg = sacd_cfg.get("stage3_gap_and_filter", {})
    min_gap_uev = float(s3_cfg.get("min_gap_uev", 10.0))
    min_island_size = int(s3_cfg.get("min_island_size", 10))
    fallback_min_island_size = int(s3_cfg.get("fallback_min_island_size", 4))
    fallback_top_n_clusters = int(s3_cfg.get("fallback_top_n_clusters", 6))

    # --- Unpack TGP Baseline Section ---
    tgp_cfg = cfg.get("tgp_baseline", {})
    broadening_chunk_size = int(tgp_cfg.get("broadening_chunk_size", 5000))
    gap_th_factor = float(tgp_cfg.get("gap_threshold_factor", 0.05))
    noise_threshold = float(tgp_cfg.get("noise_threshold", 1e-4))
    zbp_prob_thresh = float(tgp_cfg.get("zbp_probability_threshold", 0.7))
    avg_over_cutter = bool(tgp_cfg.get("average_over_cutter", False))
    gap_th_low = float(tgp_cfg.get("threshold_low_mev", 0.010))
    gap_th_high = float(tgp_cfg.get("threshold_high_mev", 0.067))
    tgp_min_cluster_size = int(tgp_cfg.get("min_cluster_size", 7))
    cluster_gap_threshold = float(tgp_cfg.get("cluster_gap_threshold_mev", 0.010))
    cluster_pct_boundary = float(tgp_cfg.get("cluster_percentage_boundary_threshold", 0.6))
    cluster_ncutter_threshold = float(tgp_cfg.get("cluster_ncutter_threshold", 0.5))

    # --- Unpack Diagnostics Section ---
    diag_cfg = cfg.get("diagnostics", {})
    ref_barrier_u0 = float(diag_cfg.get("reference_barrier_u0", 2.0))
    n_evals_cut = int(diag_cfg.get("num_eigenvalues_cut", 14))
    n_evals_medoid = int(diag_cfg.get("num_eigenvalues_medoid", 2))
    mzm_conf_pct = float(diag_cfg.get("mzm_boundary_confinement_pct", 80.0))
    mzm_sep_continuous = bool(diag_cfg.get("mzm_separability_continuous", True))
    mzm_sep_orient = bool(diag_cfg.get("mzm_separability_auto_orient", True))
    max_clusters_plot = int(diag_cfg.get("max_clusters_to_plot", 6))
    max_clusters_annotated = int(diag_cfg.get("max_clusters_annotated_on_map", 10))

    # --- Unpack Visualization Section ---
    vis_cfg = cfg.get("visualization", {})
    fig_dpi = int(vis_cfg.get("dpi", 200))
    spec_ylim = list(vis_cfg.get("spec_ylim_mev", [-0.16, 0.16]))
    nonlocal_range = list(vis_cfg.get("nonlocal_conductance_range", [-0.10, 0.10]))
    c_bg_fig = str(vis_cfg.get("figure_bg_color", "#0b132b"))
    c_bg_axes = str(vis_cfg.get("axes_bg_color", "#0b1120"))
    c_bg_legend = str(vis_cfg.get("legend_bg_color", "#0f172a"))
    c_grid = str(vis_cfg.get("grid_color", "white"))
    grid_alpha = float(vis_cfg.get("grid_alpha", 0.25))
    c_pfaff_bound = str(vis_cfg.get("pfaffian_boundary_color", "#38bdf8"))
    c_pfaff_bulk = str(vis_cfg.get("pfaffian_bulk_color", "#1e293b"))
    palette = list(vis_cfg.get("cluster_palette", [
        "#00f5d4", "#ff006e", "#3a86ff", "#ffbe0b", "#06d6a0", "#fb5607",
        "#8338ec", "#00b4d8", "#90be6d", "#f72585", "#4cc9f0", "#e63946",
        "#a855f7", "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16", "#06b6d4"
    ]))

    # Save a frozen copy of the configuration into the plots directory for exact replicability
    with open(plots_dir / "executed_protocol_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    logger.info("Saved replica config to: %s", plots_dir / "executed_protocol_config.yaml")

    # --------------------------------------------------------------------------
    # 2. Load Simulation Arrays & Ground Truth
    # --------------------------------------------------------------------------
    logger.info("Loading simulation parameters from %s...", data_dir)
    params = np.load(data_dir / "all_params.npz", allow_pickle=True)
    mu_var = params["mu_var"]
    Vz_var = params["Vz_var"]
    Nmu = len(mu_var)
    Nvz = len(Vz_var)
    extent = [Vz_var[0], Vz_var[-1], mu_var[0], mu_var[-1]]

    # Ground truth Pfaffian
    pdi_data = np.load(data_dir / "pdi_data.npy")
    col_idx = 3 if pdi_data.shape[1] >= 4 else 2
    gt = (pdi_data[:, col_idx] == 1.0).reshape(Nmu, Nvz)

    # Dedicated temperature-dependent barrier conductances
    temp_suffix = ""
    if temp_mK > 0:
        t_str = f"{int(temp_mK)}mK" if temp_mK == int(temp_mK) else f"{temp_mK:.1f}mK"
        temp_suffix = f"_{t_str}"

    b_arr = np.load(data_dir / "barrier_arr.npy")

    def _load_barrier_arr(base_name: str) -> np.ndarray:
        if temp_suffix:
            p_suffixed = data_dir / f"{base_name}{temp_suffix}.npy"
            if p_suffixed.exists():
                logger.info("Loaded barrier file: %s", p_suffixed.name)
                return np.load(p_suffixed)
        p_base = data_dir / f"{base_name}.npy"
        if p_base.exists():
            logger.info("Loaded barrier file: %s", p_base.name)
            return np.load(p_base)
        raise FileNotFoundError(f"Neither {base_name}{temp_suffix}.npy nor {base_name}.npy found in {data_dir}")

    bl_l = _load_barrier_arr("barrier_left_conductance_left_arr")
    bl_r = _load_barrier_arr("barrier_left_conductance_right_arr")
    br_l = _load_barrier_arr("barrier_right_conductance_left_arr")
    br_r = _load_barrier_arr("barrier_right_conductance_right_arr")

    idx_sym = int(np.argmin(np.abs(b_arr - sym_barrier_val)))
    idx_pinch = int(np.argmax(b_arr))

    g_LL_sym = br_l[:, idx_sym].reshape(Nmu, Nvz)
    g_LL_pinch = br_l[:, idx_pinch].reshape(Nmu, Nvz)
    g_RR_sym = bl_r[:, idx_sym].reshape(Nmu, Nvz)
    g_RR_pinch = bl_r[:, idx_pinch].reshape(Nmu, Nvz)

    dIdVl = np.load(data_dir / "tgp_stage1_dIdVl.npy")
    dIdVr = np.load(data_dir / "tgp_stage1_dIdVr.npy")
    topological_gap_2d = np.load(data_dir / "topological_gap.npy").reshape(Nmu, Nvz)

    # --------------------------------------------------------------------------
    # 3. Microsoft Azure Quantum TGP Baseline Pipeline
    # --------------------------------------------------------------------------
    logger.info("Executing Microsoft TGP Baseline pipeline at %s mK...", temp_mK)
    adapter = TGPAdapter(str(data_dir))
    ds = adapter.to_xarray()
    tprep = prepare_sim_cpu(ds, T_mK=temp_mK, chunk_size=broadening_chunk_size)

    t_l = tprep.rename({"bias": "left_bias"})
    t_r = tprep.rename({"bias": "right_bias"})
    t_l, t_r = tgp.two.extract_gap(
        t_l, t_r,
        gap_threshold_factor=gap_th_factor,
        noise_threshold=noise_threshold
    )
    z_ds = tgp.two.zbp_dataset_derivative(
        t_l, t_r,
        zbp_probability_threshold=zbp_prob_thresh,
        average_over_cutter=avg_over_cutter
    )
    tgp.two.set_gap_threshold(
        z_ds,
        threshold_low=gap_th_low,
        threshold_high=gap_th_high
    )
    s2 = tgp.two.cluster_and_score(
        z_ds,
        min_cluster_size=tgp_min_cluster_size,
        cluster_gap_threshold=cluster_gap_threshold,
        cluster_percentage_boundary_threshold=cluster_pct_boundary,
        cluster_ncutter_threshold=cluster_ncutter_threshold
    )
    roi2_vb = s2.roi2.values.T if s2.roi2.dims == ("B", "V") else s2.roi2.values
    m_tgp = (roi2_vb > 0)

    # --------------------------------------------------------------------------
    # 4. User S-ACD Protocol Suite Stages 1-3 with Subtractive Filtering
    # --------------------------------------------------------------------------
    logger.info("Evaluating S-ACD Protocol Suite Stages 1-3...")
    m_stage1, sqrt_a, S_sym = compute_sacd_stage1(
        g_LL_sym, g_LL_pinch, g_RR_sym, g_RR_pinch,
        tau=tau, s_max=s_max, floor_val=floor_val
    )
    corr_cfg = sacd_cfg.get("stage_corr", {})
    if corr_cfg.get("enabled", False):
        from src.Protocol_Elements import FPE, RealizationFeatureCache
        v0_lbl = data_dir.parent.name if data_dir.parent.name.startswith("V0_") else "V0_0_378"
        cache = RealizationFeatureCache(data_dir, temperature_mK=temp_mK, v0_label=v0_lbl)
        m_corr = FPE.fpe_barrier_reflection_correlation(
            cache,
            corr_threshold=float(corr_cfg.get("corr_threshold", 0.80)),
            asym_min=float(corr_cfg.get("asym_min", -5.0)),
            asym_max=float(corr_cfg.get("asym_max", 5.0)),
            lead=str(corr_cfg.get("lead", "R")),
        )
        m_stage1 = m_stage1 & m_corr
        logger.info("Applied Low-Asymmetry Correlation gate (thresh=%.2f, passed=%d px)",
                    float(corr_cfg.get("corr_threshold", 0.80)), int(np.sum(m_corr)))

    m_stage2 = m_stage1 & compute_sacd_stage2_curvature(
        dIdVl, dIdVr, Nmu, Nvz,
        d2_thresh=d2_threshold,
        min_cutters=min_cutters_consensus,
        win_len=savgol_window_length,
        polyorder=savgol_polyorder
    )
    m_stage3 = m_stage2 & compute_sacd_stage3_gap(topological_gap_2d, min_gap_uev=min_gap_uev)

    # Purely Subtractive Island Filter (Zero dilation, zero morphological closing)
    lbl_raw, n_raw = label(m_stage3, structure=np.ones((3, 3), dtype=int))
    sizes_raw = np.bincount(lbl_raw.ravel())
    valid_components = np.where(sizes_raw[1:] >= min_island_size)[0] + 1
    if len(valid_components) == 0:
        logger.warning("No components >= %d px found under strict thresholds. Retaining top clusters >= %d px...",
                       min_island_size, fallback_min_island_size)
        valid_components = np.argsort(sizes_raw[1:])[::-1][:fallback_top_n_clusters] + 1
        valid_components = [c for c in valid_components if sizes_raw[c] >= fallback_min_island_size]

    m_subtractive = np.isin(lbl_raw, valid_components)

    met_tgp = calc_protocol_metrics(m_tgp, gt)
    met_s1 = calc_protocol_metrics(m_stage1, gt)
    met_s2 = calc_protocol_metrics(m_stage2, gt)
    met_sub = calc_protocol_metrics(m_subtractive, gt)

    logger.info("Stage 3 Raw: %d px | Subtractive Filtered: %d px (TPR=%.2f%%, FDR=%.2f%%)",
                int(np.sum(m_stage3)), int(np.sum(m_subtractive)),
                met_sub["pfaffian_passed_pct"], met_sub["fdr_percent"])

    # --------------------------------------------------------------------------
    # 5. Generate Step-by-Step Protocol Comparison Map (4 Panels)
    # --------------------------------------------------------------------------
    logger.info("Generating 4-panel step-by-step comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
    fig.patch.set_facecolor(c_bg_fig)

    panels = [
        (axes[0, 0], m_tgp, "#a855f7", "Regular TGP (Stage 2 ROI 2)", met_tgp),
        (axes[0, 1], m_stage1, "#0284c7", f"Step 1: Asymmetric Conductance Deviation (τ={tau}, S_max={s_max})", met_s1),
        (axes[1, 0], m_stage2, "#f59e0b", f"Step 2: Curvature Consensus (d²G/dV² ≤ -{int(d2_threshold)})", met_s2),
        (axes[1, 1], m_subtractive, "#10b981", f"Step 3: Subtractive S-ACD (Gap ≥ {int(min_gap_uev)} μeV + Size Filter)", met_sub),
    ]

    for ax, mask, color, title, m_dict in panels:
        ax.set_facecolor(c_bg_axes)
        img_rgba = np.zeros((Nmu, Nvz, 4))
        img_rgba[~gt & (~mask)] = [0.04, 0.07, 0.13, 1.0]
        img_rgba[gt & (~mask)] = [0.12, 0.16, 0.23, 0.95]
        rgb = mcolors.to_rgb(color)
        img_rgba[mask] = [rgb[0], rgb[1], rgb[2], 0.95]

        ax.imshow(img_rgba, origin="lower", extent=extent, aspect="auto")
        ax.contour(Vz_var, mu_var, gt.astype(float), levels=[0.5], colors=[c_pfaff_bound], linewidths=1.6, linestyles="--")

        sub_title = f"{title}\nPfaffian Passed: {m_dict['pfaffian_passed_pct']:.2f}% | FDR: {m_dict['fdr_percent']:.2f}%"
        ax.set_title(sub_title, fontsize=11, fontweight="bold", pad=8, color="#f1f5f9")
        ax.grid(True, linestyle=":", alpha=grid_alpha, color=c_grid)
        ax.tick_params(colors="white")

    axes[0, 0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11, color="white")
    axes[1, 0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11, color="white")
    axes[1, 0].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11, color="white")
    axes[1, 1].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11, color="white")

    legend_elements = [
        mlines.Line2D([0], [0], color=c_pfaff_bound, lw=1.8, linestyle="--", label=r"Pfaffian Boundary ($\mathcal{Q} = -1$)"),
        mpatches.Patch(facecolor=c_pfaff_bulk, edgecolor="#475569", label="Pfaffian Bulk (Unpassed)"),
        mpatches.Patch(facecolor="#a855f7", label="Regular TGP Passed"),
        mpatches.Patch(facecolor="#0284c7", label="Step 1 Passed (S-ACD)"),
        mpatches.Patch(facecolor="#f59e0b", label="Step 2 Passed (Curvature)"),
        mpatches.Patch(facecolor="#10b981", label="Step 3 Passed (Subtractive Filter)"),
    ]
    axes[0, 0].legend(handles=legend_elements, loc="upper left", fontsize=8, facecolor=c_bg_legend, edgecolor="#334155", labelcolor="white")

    fig.suptitle(f"Subtractive Protocol Comparison: {disorder_name} at T = {int(temp_mK)} mK (Curvature ≤ -{int(d2_threshold)}, Gap ≥ {int(min_gap_uev)} μeV)", fontsize=13, fontweight="bold", color=c_pfaff_bound, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    step_comp_path = plots_dir / "protocol_step_by_step_comparison.png"
    fig.savefig(step_comp_path, dpi=fig_dpi)
    plt.close(fig)
    logger.info("Saved %s", step_comp_path)

    # --------------------------------------------------------------------------
    # 6. Cluster Identification & Medoid Determination
    # --------------------------------------------------------------------------
    lbl, n_comp = label(m_subtractive, structure=np.ones((3, 3), dtype=int))
    logger.info("Retained %d subtractive clusters", n_comp)

    raw_clusters = []
    for comp_id in range(1, n_comp + 1):
        pts = np.argwhere(lbl == comp_id)
        sz = len(pts)

        # Centroid coordinates
        com_mu_exact = float(np.mean(mu_var[pts[:, 0]]))
        com_vz_exact = float(np.mean(Vz_var[pts[:, 1]]))

        # Medoid: the actual passing pixel within the cluster closest to centroid
        dists = (mu_var[pts[:, 0]] - com_mu_exact)**2 + (Vz_var[pts[:, 1]] - com_vz_exact)**2
        best_idx = int(np.argmin(dists))
        medoid_mu_idx = int(pts[best_idx, 0])
        medoid_vz_idx = int(pts[best_idx, 1])
        medoid_mu = float(mu_var[medoid_mu_idx])
        medoid_vz = float(Vz_var[medoid_vz_idx])

        n_tp = int(np.sum(gt[lbl == comp_id]))
        pct_tp = float((n_tp / sz) * 100.0)

        raw_clusters.append({
            "orig_id": comp_id,
            "pts": pts,
            "size": sz,
            "n_tp": n_tp,
            "pct_tp": pct_tp,
            "medoid_mu_idx": medoid_mu_idx,
            "medoid_vz_idx": medoid_vz_idx,
            "medoid_mu": medoid_mu,
            "medoid_vz": medoid_vz,
        })

    # Sort clusters by size descending
    raw_clusters = sorted(raw_clusters, key=lambda x: x["size"], reverse=True)

    cluster_info = []
    new_lbl = np.zeros_like(lbl)
    for new_id, c in enumerate(raw_clusters, 1):
        new_lbl[lbl == c["orig_id"]] = new_id
        col = palette[(new_id - 1) % len(palette)]
        c["id"] = new_id
        c["color"] = col
        cluster_info.append(c)

    # --------------------------------------------------------------------------
    # 7. Plot Final Subtractive Phase Map with Medoid Markers
    # --------------------------------------------------------------------------
    logger.info("Plotting final phase map with Medoid markers...")
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(c_bg_fig)
    ax.set_facecolor(c_bg_axes)

    base_rgba = np.zeros((Nmu, Nvz, 4))
    base_rgba[~gt] = [0.04, 0.07, 0.13, 1.0]
    base_rgba[gt] = [0.12, 0.16, 0.23, 0.95]
    ax.imshow(base_rgba, origin="lower", extent=extent, aspect="auto")

    for c in cluster_info:
        c_mask = (new_lbl == c["id"])
        c_rgb = mcolors.to_rgb(c["color"])
        c_rgba = np.zeros((Nmu, Nvz, 4))
        c_rgba[c_mask] = [c_rgb[0], c_rgb[1], c_rgb[2], 0.95]
        ax.imshow(c_rgba, origin="lower", extent=extent, aspect="auto")

    ax.contour(Vz_var, mu_var, gt.astype(float), levels=[0.5], colors=[c_pfaff_bound], linewidths=1.8, linestyles="--")

    for rank, c in enumerate(cluster_info[:max_clusters_annotated], 1):
        ax.axhline(c["medoid_mu"], color=c["color"], linestyle=":", linewidth=0.8, alpha=0.5)
        ax.plot(c["medoid_vz"], c["medoid_mu"], marker="*", markersize=14, color=c["color"], markeredgecolor="black", markeredgewidth=1.2)
        ax.annotate(
            f"Cluster {c['id']} ({c['size']} px)\nμ={c['medoid_mu']:.2f}, Vz={c['medoid_vz']:.2f}\nTP={c['pct_tp']:.0f}%",
            xy=(c["medoid_vz"], c["medoid_mu"]),
            xytext=(c["medoid_vz"] + 0.03, c["medoid_mu"] + 0.07),
            fontsize=8,
            fontweight="bold",
            color=c["color"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor=c_bg_legend, edgecolor=c["color"], alpha=0.85),
            arrowprops=dict(arrowstyle="->", color=c["color"], lw=1.0)
        )

    ax.set_title(f"Subtractive Filtered Topological Clusters on {disorder_name} ({int(temp_mK)} mK)\n(Zero Dilation | Medoid Stars = Verified True Passing Points)", fontsize=13, fontweight="bold", color=c_pfaff_bound, pad=12)
    ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11, color="white")
    ax.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11, color="white")
    ax.grid(True, linestyle=":", alpha=grid_alpha, color=c_grid)
    ax.tick_params(colors="white")

    plt.tight_layout()
    phase_map_path = plots_dir / "final_clustered_phase_map_with_com.png"
    fig.savefig(phase_map_path, dpi=fig_dpi)
    plt.close(fig)
    logger.info("Saved %s", phase_map_path)

    # --------------------------------------------------------------------------
    # 8. Generate Multi-Panel Cut Diagnostics with Pixel-Perfect Axis Alignment
    # --------------------------------------------------------------------------
    t_val = float(params["t"])
    Delta0 = float(params["Delta0"])
    gamma = float(params["gamma"])
    alpha = float(params["alpha"])
    Ls = int(params["Ls"])
    V0 = float(params.get("V0", 1.2))
    Vdisx = params.get("Vdisx", np.zeros(Ls)) * V0

    bias_axis = np.linspace(bias_window_min, bias_window_max, bias_num_points)
    bias_uV = bias_axis * 1e3

    g_ll_all = tprep.g_ll.mean(dim="cutter_pair_index").values if "cutter_pair_index" in tprep.g_ll.dims else tprep.g_ll.values
    g_rr_all = tprep.g_rr.mean(dim="cutter_pair_index").values if "cutter_pair_index" in tprep.g_rr.dims else tprep.g_rr.values
    g_rl_all = tprep.g_rl.mean(dim="cutter_pair_index").values if "cutter_pair_index" in tprep.g_rl.dims else tprep.g_rl.values
    g_lr_all = tprep.g_lr.mean(dim="cutter_pair_index").values if "cutter_pair_index" in tprep.g_lr.dims else tprep.g_lr.values

    L_3w_all = tprep["L_3w"].mean(dim="cutter_pair_index").values if "cutter_pair_index" in tprep["L_3w"].dims else tprep["L_3w"].values

    gap_l_all = t_l.gap.mean(dim="cutter_pair_index").values if "cutter_pair_index" in t_l.gap.dims else t_l.gap.values
    gap_r_all = t_r.gap.mean(dim="cutter_pair_index").values if "cutter_pair_index" in t_r.gap.dims else t_r.gap.values

    selected_clusters = cluster_info[:max_clusters_plot]
    logger.info("Generating multi-panel diagnostic slides for %d key clusters...", len(selected_clusters))

    slides_metadata = []

    for c in selected_clusters:
        isl_id = c["id"]
        mu_idx = c["medoid_mu_idx"]
        vz_idx = c["medoid_vz_idx"]
        mu_cut = c["medoid_mu"]
        vz_medoid = c["medoid_vz"]
        col = c["color"]

        logger.info("Processing Cluster %d (size %d px, Medoid mu=%.3f, Vz=%.3f)...", isl_id, c["size"], mu_cut, vz_medoid)

        # 1. BdG Spectrum along horizontal cut mu = mu_cut
        n_cut_pts = Nvz
        cut_vz = Vz_var
        cut_evals = np.zeros((n_cut_pts, n_evals_cut))

        for j in range(n_cut_pts):
            vz_j = cut_vz[j]
            syst_j = hp.build_system_closed(t_val, mu_cut, gamma, Delta0, vz_j, alpha, Ls, Vdisx, a=1)
            evals_j, _ = hp.solve_ham(syst_j, k=n_evals_cut, solver_type="cpu")
            cut_evals[j, :] = np.sort(evals_j)

        # 2. Extract 2D maps along cut
        g_ll_cut = g_ll_all[:, mu_idx, :]
        g_rr_cut = g_rr_all[:, mu_idx, :]
        g_rl_cut = g_rl_all[:, mu_idx, :]
        g_lr_cut = g_lr_all[:, mu_idx, :]

        a_grl_cut = 0.5 * (g_rl_cut - g_rl_cut[:, ::-1])
        a_glr_cut = 0.5 * (g_lr_cut - g_lr_cut[:, ::-1])

        gap_l_cut = gap_l_all[mu_idx, :]
        gap_r_cut = gap_r_all[mu_idx, :]

        # 3. Wavefunctions and localization at MEDOID point
        syst_med = hp.build_system_closed(t_val, mu_cut, gamma, Delta0, vz_medoid, alpha, Ls, Vdisx, a=1)
        evals_med, evecs_med = hp.solve_ham(syst_med, k=n_evals_medoid, solver_type="cpu")
        rho_M1, rho_M2, _ = hp.get_psiM_density(evals_med, evecs_med)

        conf_val = hp.calc_mzm_boundary_confinement(rho_M1, rho_M2, Ls, pct_thresh=mzm_conf_pct)
        sep_val = hp.calc_mzm_separability(rho_M1, rho_M2, continuous=mzm_sep_continuous, auto_orient=mzm_sep_orient)

        # 4. Differential conductance dI/dV at MEDOID
        didv_l_med = g_ll_all[vz_idx, mu_idx, :]
        didv_r_med = g_rr_all[vz_idx, mu_idx, :]
        val_2w = float(didv_l_med[zero_bias_index])
        val_3w = float(L_3w_all[vz_idx, mu_idx])

        # 5. Barrier Sweeps at MEDOID
        pt_flat_idx = mu_idx * Nvz + vz_idx
        x_barrier = b_arr / ref_barrier_u0

        br_l_med = br_l[pt_flat_idx, :]
        br_r_med = br_r[pt_flat_idx, :]
        bl_l_med = bl_l[pt_flat_idx, :]
        bl_r_med = bl_r[pt_flat_idx, :]

        norm_br_l = br_l_med / max(br_l_med[idx_sym], 1e-4)
        norm_br_r = br_r_med / max(br_r_med[idx_sym], 1e-4)
        norm_bl_l = bl_l_med / max(bl_l_med[idx_sym], 1e-4)
        norm_bl_r = bl_r_med / max(bl_r_med[idx_sym], 1e-4)

        # ----------------------------------------------------------------------
        # Assemble Diagnostic Figure with Pixel-Perfect Axis Alignment
        # ----------------------------------------------------------------------
        fig = plt.figure(figsize=(17, 9.5))
        fig.patch.set_facecolor(c_bg_fig)

        gs_master = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.18)

        # --- LEFT COLUMN (5 Subpanels with matching width dividers) ---
        gs_left = gs_master[0].subgridspec(5, 1, height_ratios=[2.0, 1.0, 1.0, 1.0, 1.0], hspace=0.16)

        # Panel 1: BdG Energy Spectrum
        ax_spec = fig.add_subplot(gs_left[0])
        ax_spec.set_facecolor(c_bg_axes)

        # Topological shading
        gt_cut = gt[mu_idx, :]
        for k_idx in range(n_cut_pts):
            if gt_cut[k_idx]:
                dvz = (cut_vz[-1] - cut_vz[0]) / (n_cut_pts - 1)
                ax_spec.axvspan(cut_vz[k_idx] - 0.5 * dvz, cut_vz[k_idx] + 0.5 * dvz, color=c_pfaff_bulk, alpha=0.75, zorder=0)

        for ev_i in range(n_evals_cut):
            ax_spec.plot(cut_vz, cut_evals[:, ev_i], color=c_pfaff_bound, lw=1.1, alpha=0.85, zorder=2)
        ax_spec.axhline(0.0, color="#ef4444", lw=1.2, linestyle="-", zorder=3)
        ax_spec.axvline(vz_medoid, color=col, linestyle="--", lw=1.4, alpha=0.9, zorder=4)

        # Runner ticks
        tgp_cut = m_tgp[mu_idx, :]
        sub_cut = m_subtractive[mu_idx, :]
        ax_spec.plot(cut_vz[tgp_cut], np.full(np.sum(tgp_cut), spec_ylim[0] + 0.015), "|", color="#a855f7", markersize=6, label="Regular TGP (ROI 2)")
        ax_spec.plot(cut_vz[sub_cut], np.full(np.sum(sub_cut), spec_ylim[0] + 0.020), "|", color="#10b981", markersize=6, label="Subtractive Filtered")

        ax_spec.set_ylim(spec_ylim[0], spec_ylim[1])
        ax_spec.set_xlim(cut_vz[0], cut_vz[-1])
        ax_spec.set_ylabel("Energy (meV)", fontsize=9.5, color="white")
        ax_spec.set_title(f"Spectra and Conductance along Cluster_{isl_id}_Cut\n($\mu = {mu_cut:.3f}$ meV, T = {int(temp_mK)} mK)", fontsize=11, fontweight="bold", color=c_pfaff_bound, pad=6)
        ax_spec.legend(loc="upper right", fontsize=7.5, facecolor=c_bg_legend, edgecolor="#334155", labelcolor="white")
        ax_spec.tick_params(colors="white", labelbottom=False)
        ax_spec.grid(True, linestyle=":", alpha=0.2, color=c_grid)

        div_spec = make_axes_locatable(ax_spec)
        cax_dummy = div_spec.append_axes("right", size="3%", pad=0.08)
        cax_dummy.set_visible(False)

        # Top coordinate axis
        ax_top = ax_spec.twiny()
        ax_top.set_xlim(ax_spec.get_xlim())
        ax_top.set_xticks(np.linspace(cut_vz[0], cut_vz[-1], 6))
        ax_top.set_xticklabels([f"{mu_cut:.3f}" for _ in range(6)], fontsize=8, color="#94a3b8")
        ax_top.set_xlabel(r"Chemical Potential $\mu$ (meV)", fontsize=9, color="#94a3b8", labelpad=2)
        ax_top.tick_params(colors="#94a3b8")

        # Panel 2: G_LL
        ax_gll = fig.add_subplot(gs_left[1], sharex=ax_spec)
        ax_gll.set_facecolor(c_bg_axes)
        im_gll = ax_gll.pcolormesh(cut_vz, bias_uV, g_ll_cut.T, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0, rasterized=True)
        ax_gll.axvline(vz_medoid, color=col, linestyle="--", lw=1.4, alpha=0.9)
        ax_gll.axhline(0, color="gray", linestyle=":", lw=0.8, alpha=0.5)
        ax_gll.set_ylabel("Left bias\n[$\mu$V]", fontsize=8.5, color="white")
        ax_gll.tick_params(colors="white", labelbottom=False)

        div_gll = make_axes_locatable(ax_gll)
        cax_gll = div_gll.append_axes("right", size="3%", pad=0.08)
        cb_gll = fig.colorbar(im_gll, cax=cax_gll)
        cb_gll.set_label(r"$G_{\mathrm{LL}}\,[e^2/h]$", fontsize=7.5, color="white")
        cb_gll.ax.tick_params(colors="white", labelsize=7)

        # Panel 3: G_RR
        ax_grr = fig.add_subplot(gs_left[2], sharex=ax_spec)
        ax_grr.set_facecolor(c_bg_axes)
        im_grr = ax_grr.pcolormesh(cut_vz, bias_uV, g_rr_cut.T, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0, rasterized=True)
        ax_grr.axvline(vz_medoid, color=col, linestyle="--", lw=1.4, alpha=0.9)
        ax_grr.axhline(0, color="gray", linestyle=":", lw=0.8, alpha=0.5)
        ax_grr.set_ylabel("Right bias\n[$\mu$V]", fontsize=8.5, color="white")
        ax_grr.tick_params(colors="white", labelbottom=False)

        div_grr = make_axes_locatable(ax_grr)
        cax_grr = div_grr.append_axes("right", size="3%", pad=0.08)
        cb_grr = fig.colorbar(im_grr, cax=cax_grr)
        cb_grr.set_label(r"$G_{\mathrm{RR}}\,[e^2/h]$", fontsize=7.5, color="white")
        cb_grr.ax.tick_params(colors="white", labelsize=7)

        # Panel 4: A(G_RL) Non-local conductance
        ax_grl = fig.add_subplot(gs_left[3], sharex=ax_spec)
        ax_grl.set_facecolor(c_bg_axes)
        im_grl = ax_grl.pcolormesh(cut_vz, bias_uV, a_grl_cut.T, shading="auto", cmap="PuOr_r",
                                  vmin=nonlocal_range[0], vmax=nonlocal_range[1], rasterized=True)
        ax_grl.axvline(vz_medoid, color=col, linestyle="--", lw=1.4, alpha=0.9)
        ax_grl.axhline(0, color="gray", linestyle=":", lw=0.8, alpha=0.5)
        valid_gap_l = (gap_l_cut * 1e3 > 0) & (gap_l_cut * 1e3 <= bias_window_max * 1e3)
        if np.any(valid_gap_l):
            ax_grl.plot(cut_vz[valid_gap_l], gap_l_cut[valid_gap_l] * 1e3, color="black", lw=1.4)
            ax_grl.plot(cut_vz[valid_gap_l], -gap_l_cut[valid_gap_l] * 1e3, color="black", lw=1.4)
        ax_grl.set_ylabel("Left bias\n[$\mu$V]", fontsize=8.5, color="white")
        ax_grl.tick_params(colors="white", labelbottom=False)

        div_grl = make_axes_locatable(ax_grl)
        cax_grl = div_grl.append_axes("right", size="3%", pad=0.08)
        cb_grl = fig.colorbar(im_grl, cax=cax_grl)
        cb_grl.set_label(r"$A(G_{\mathrm{RL}})\,[e^2/h]$", fontsize=7.5, color="white")
        cb_grl.ax.tick_params(colors="white", labelsize=7)

        # Panel 5: A(G_LR) Non-local conductance
        ax_glr = fig.add_subplot(gs_left[4], sharex=ax_spec)
        ax_glr.set_facecolor(c_bg_axes)
        im_glr = ax_glr.pcolormesh(cut_vz, bias_uV, a_glr_cut.T, shading="auto", cmap="PuOr_r",
                                  vmin=nonlocal_range[0], vmax=nonlocal_range[1], rasterized=True)
        ax_glr.axvline(vz_medoid, color=col, linestyle="--", lw=1.4, alpha=0.9)
        ax_glr.axhline(0, color="gray", linestyle=":", lw=0.8, alpha=0.5)
        valid_gap_r = (gap_r_cut * 1e3 > 0) & (gap_r_cut * 1e3 <= bias_window_max * 1e3)
        if np.any(valid_gap_r):
            ax_glr.plot(cut_vz[valid_gap_r], gap_r_cut[valid_gap_r] * 1e3, color="black", lw=1.4)
            ax_glr.plot(cut_vz[valid_gap_r], -gap_r_cut[valid_gap_r] * 1e3, color="black", lw=1.4)
        ax_glr.set_ylabel("Right bias\n[$\mu$V]", fontsize=8.5, color="white")
        ax_glr.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10, color="white")
        ax_glr.tick_params(colors="white")

        div_glr = make_axes_locatable(ax_glr)
        cax_glr = div_glr.append_axes("right", size="3%", pad=0.08)
        cb_glr = fig.colorbar(im_glr, cax=cax_glr)
        cb_glr.set_label(r"$A(G_{\mathrm{LR}})\,[e^2/h]$", fontsize=7.5, color="white")
        cb_glr.ax.tick_params(colors="white", labelsize=7)

        # --- RIGHT COLUMN (3 Main Panels) ---
        gs_right = gs_master[1].subgridspec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.32)

        # Panel 6: dI/dV at MEDOID
        ax_didv = fig.add_subplot(gs_right[0])
        ax_didv.set_facecolor(c_bg_axes)
        ax_didv.plot(bias_axis, didv_l_med, label="Left dI/dV", color="#3b82f6", lw=1.6)
        ax_didv.plot(bias_axis, didv_r_med, label="Right dI/dV", color="#f97316", lw=1.6)
        ax_didv.plot(0.0, val_2w, marker="*", markersize=12, color=col, markeredgecolor="black", label=f"Medoid Marker ({col})")

        ax_didv.set_title(f"dI/dV for Cluster_{isl_id}_Medoid [Verified Passing Pixel]\n($\mu = {mu_cut:.3f}$, $V_z = {vz_medoid:.3f}$) | $2\omega_L = {val_2w:.3f}$, $3\omega_L = {val_3w:.1f}$", fontsize=10, fontweight="bold", color=c_pfaff_bound, pad=4)
        ax_didv.set_xlabel("Energy (meV)", fontsize=9, color="white")
        ax_didv.set_ylabel("Conductance [$e^2/h$]", fontsize=9, color="white")
        ax_didv.legend(loc="upper right", fontsize=8, facecolor=c_bg_legend, edgecolor="#334155", labelcolor="white")
        ax_didv.tick_params(colors="white")
        ax_didv.grid(True, linestyle=":", alpha=grid_alpha, color=c_grid)

        # Panel 7: Wavefunctions at MEDOID
        ax_wf = fig.add_subplot(gs_right[1])
        ax_wf.set_facecolor(c_bg_axes)
        ax_wf.plot(np.abs(rho_M1), label=r"Majorana Left ($M_1$)", color="#00f5d4", lw=1.8)
        ax_wf.plot(np.abs(rho_M2), label=r"Majorana Right ($M_2$)", color="#fb5607", lw=1.8)

        ax_wf.set_title(f"Wavefunctions for Cluster_{isl_id}_Medoid\n(Boundary Confinement $\mathcal{{C}} = {conf_val:.3f}$, Separability $\mathcal{{S}} = {sep_val:.3f}$)", fontsize=10, fontweight="bold", color=c_pfaff_bound, pad=4)
        ax_wf.set_xlabel(f"Site Index (0 to {Ls})", fontsize=9, color="white")
        ax_wf.set_ylabel("Probability Density", fontsize=9, color="white")
        ax_wf.legend(loc="upper center", fontsize=8, facecolor=c_bg_legend, edgecolor="#334155", labelcolor="white")
        ax_wf.tick_params(colors="white")
        ax_wf.grid(True, linestyle=":", alpha=grid_alpha, color=c_grid)

        # Panel 8: Dual Barrier Sweeps
        gs_barr = gs_right[2].subgridspec(1, 2, wspace=0.28)

        ax_br = fig.add_subplot(gs_barr[0])
        ax_br.set_facecolor(c_bg_axes)
        ax_br.plot(x_barrier, norm_br_l, label=r"$G_{\mathrm{LL}}$", color="#10b981", lw=1.6)
        ax_br.plot(x_barrier, norm_br_r, label=r"$G_{\mathrm{RR}}$", color="#3b82f6", lw=1.6)
        ax_br.set_title(f"Varying Right Barrier ({int(temp_mK)} mK)\n(Cluster_{isl_id}_Medoid)", fontsize=9, fontweight="bold", color="#f1f5f9", pad=4)
        ax_br.set_xlabel(r"$U_R / U_0$", fontsize=8.5, color="white")
        ax_br.set_ylabel("Normalized Conductance", fontsize=8.5, color="white")
        ax_br.legend(loc="upper right", fontsize=7.5, facecolor=c_bg_legend, edgecolor="#334155", labelcolor="white")
        ax_br.tick_params(colors="white")
        ax_br.grid(True, linestyle=":", alpha=grid_alpha, color=c_grid)

        ax_bl = fig.add_subplot(gs_barr[1])
        ax_bl.set_facecolor(c_bg_axes)
        ax_bl.plot(x_barrier, norm_bl_l, label=r"$G_{\mathrm{LL}}$", color="#10b981", lw=1.6)
        ax_bl.plot(x_barrier, norm_bl_r, label=r"$G_{\mathrm{RR}}$", color="#3b82f6", lw=1.6)
        ax_bl.set_title(f"Varying Left Barrier ({int(temp_mK)} mK)\n(Cluster_{isl_id}_Medoid)", fontsize=9, fontweight="bold", color="#f1f5f9", pad=4)
        ax_bl.set_xlabel(r"$U_L / U_0$", fontsize=8.5, color="white")
        ax_bl.set_ylabel("Normalized Conductance", fontsize=8.5, color="white")
        ax_bl.legend(loc="upper right", fontsize=7.5, facecolor=c_bg_legend, edgecolor="#334155", labelcolor="white")
        ax_bl.tick_params(colors="white")
        ax_bl.grid(True, linestyle=":", alpha=grid_alpha, color=c_grid)

        out_slide_path = plots_dir / f"island_{isl_id}_com_cut_diagnostics.png"
        fig.savefig(out_slide_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved slide %s", out_slide_path)

        slides_metadata.append({
            "cluster_id": isl_id,
            "size_px": c["size"],
            "pct_tp": c["pct_tp"],
            "mu_cut": round(mu_cut, 4),
            "vz_medoid": round(vz_medoid, 4),
            "confinement": round(conf_val, 4),
            "separability": round(sep_val, 4),
            "val_2w": round(val_2w, 4),
            "val_3w": round(val_3w, 2),
            "image_filename": f"island_{isl_id}_com_cut_diagnostics.png"
        })

    meta_path = plots_dir / "revised_clusters_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(slides_metadata, f, indent=2)
    logger.info("Saved metadata summary to %s", meta_path)
    print("\n" + "=" * 80)
    print(f"REVISED PROTOCOL ({int(temp_mK)} mK) CUT DIAGNOSTICS GENERATED SUCCESSFULLY")
    print(f"Output Directory: {plots_dir}")
    print(f"Executed Config:  {plots_dir / 'executed_protocol_config.yaml'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
