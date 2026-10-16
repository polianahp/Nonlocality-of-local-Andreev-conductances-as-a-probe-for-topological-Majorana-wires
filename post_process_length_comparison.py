#!/usr/bin/env python3
"""
post_process_length_comparison.py

Post-processes and compares paired 3um vs 6um simulation runs:
- Benchmarks Microsoft TGP Stage 2 against the 3-step Nonlocal Protocol:
    1. Step 1: Correlation Condition (C_L >= 0.85, C_R >= 0.85)
    2. Step 2: 3w Measurement (Step 1 + ZBP derivative consensus >= 50%)
    3. Step 3: Extracted Transport Gap (Step 2 + Gap >= 10 ueV consensus >= 50%)
- Computes length sensitivity deltas (Delta FDR, Delta TPR, Delta Area, Confinement, Separability).
- Generates 4-panel high-resolution comparison and transition figures for every disorder.
- Generates cross-disorder global sensitivity trend figures for Correlation, 3w, and Transport Gap.
- Exports metrics to JSON in Outputs/slides/six_micron_analysis/metrics/.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tgp_adapter import TGPAdapter
from src.cpu_broadening import prepare_sim_cpu
import tgp.two

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("LengthComparison")

OUTPUT_BASE = REPO_ROOT / "Outputs" / "slides" / "six_micron_analysis"
FIGURES_DIR = OUTPUT_BASE / "figures"
METRICS_DIR = OUTPUT_BASE / "metrics"

PAIRED_DISORDERS = [
    ("V0_0_0", 0.0, "V0 = 0.0 meV (Clean Wire)"),
    ("V0_0_1", 0.1, "V0 = 0.1 meV (Low Disorder)"),
    ("V0_0_645", 0.645, "V0 = 0.645 meV (Strong Disorder)"),
    ("V0_0_872", 0.872, "V0 = 0.872 meV (Strong Disorder)"),
    ("V0_0_91", 0.91, "V0 = 0.91 meV (Strong Disorder)"),
    ("V0_1_2", 1.2, "V0 = 1.2 meV (Benchmark Disorder)"),
]


def vectorized_reflection_correlation(
    f1: np.ndarray,
    f2: np.ndarray,
    barrier_arr: np.ndarray
) -> np.ndarray:
    """Computes normalized correlation C(f1, f2) across non-negative barrier regime."""
    pos_mask = barrier_arr >= 0.0
    f1_m = f1[:, pos_mask]
    f2_m = f2[:, pos_mask]

    denom1 = np.where(f1_m[:, 0:1] != 0, f1_m[:, 0:1], 1.0)
    denom2 = np.where(f2_m[:, 0:1] != 0, f2_m[:, 0:1], 1.0)
    f1_norm = f1_m / denom1
    f2_norm = f2_m / denom2

    norm1 = np.linalg.norm(f1_norm, axis=1)
    norm2 = np.linalg.norm(f2_norm, axis=1)

    dot = np.sum(f1_norm * f2_norm, axis=1)
    denom = norm1 * norm2
    return np.where(denom > 0, dot / denom, 0.0)


def load_boundary_confinement(realization_dir: Path, Nmu: int, Nvz: int, Ls: int = 300) -> np.ndarray:
    """Loads 2D boundary confinement map from realization folder."""
    candidates = [
        "mzm_boundary_confinement_arr.npy",
        "site_localizations.npy",
        "weight_localization_arr.npy",
    ]
    for c in candidates:
        p = realization_dir / c
        if p.exists():
            arr = np.load(p)
            if c == "site_localizations.npy" and np.nanmax(arr) > 1.0:
                arr = 1.0 - (arr.astype(float) / float(Ls))
                arr = np.clip(arr, 0.0, 1.0)
            return arr.reshape(Nmu, Nvz)
    return np.zeros((Nmu, Nvz), dtype=float)


def load_separability(realization_dir: Path, Nmu: int, Nvz: int) -> np.ndarray:
    """Loads 2D MZM separability map from realization folder."""
    candidates = ["mzm_separability_arr.npy", "mzm_separation_arr.npy"]
    for c in candidates:
        p = realization_dir / c
        if p.exists():
            arr = np.load(p)
            return arr.reshape(Nmu, Nvz)
    return np.zeros((Nmu, Nvz), dtype=float)


def calc_protocol_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    conf_2d: np.ndarray = None,
    sep_2d: np.ndarray = None,
) -> Dict[str, Any]:
    """Calculates confusion matrix, TPR, FDR, PPV, F1, Confinement, and Separability."""
    pred_b = pred_mask.astype(bool)
    gt_b = gt_mask.astype(bool)

    tp = int(np.sum(pred_b & gt_b))
    fp = int(np.sum(pred_b & (~gt_b)))
    fn = int(np.sum((~pred_b) & gt_b))
    tn = int(np.sum((~pred_b) & (~gt_b)))

    tot_pass = tp + fp
    tot_gt = tp + fn

    tpr_pct = float((tp / tot_gt) * 100.0) if tot_gt > 0 else 0.0
    fdr_pct = float((fp / tot_pass) * 100.0) if tot_pass > 0 else 0.0
    ppv_pct = float((tp / tot_pass) * 100.0) if tot_pass > 0 else 0.0

    precision = tp / tot_pass if tot_pass > 0 else 0.0
    recall = tp / tot_gt if tot_gt > 0 else 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    mean_conf, var_conf = 0.0, 0.0
    mean_sep, var_sep = 0.0, 0.0

    if tot_pass > 0:
        if conf_2d is not None:
            c_vals = conf_2d[pred_b]
            mean_conf = float(np.nanmean(c_vals))
            var_conf = float(np.nanvar(c_vals))
        if sep_2d is not None:
            s_vals = sep_2d[pred_b]
            mean_sep = float(np.nanmean(s_vals))
            var_sep = float(np.nanvar(s_vals))

    return {
        "tot_passed": tot_pass,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "total_pfaffian_area": tot_gt,
        "pfaffian_passed_pct": round(tpr_pct, 4),
        "fdr_percent": round(fdr_pct, 4),
        "ppv_percent": round(ppv_pct, 4),
        "f1_score": round(float(f1), 4),
        "mean_boundary_confinement": round(float(mean_conf), 4),
        "var_boundary_confinement": round(float(var_conf), 4),
        "mean_separability": round(float(mean_sep), 4),
        "var_separability": round(float(var_sep), 4),
    }


def process_single_dataset(args_tuple: Tuple) -> Dict[str, Any]:
    """Processes a single realization folder."""
    data_dir_str, wire_len_tag, v0_tag = args_tuple
    data_dir = Path(data_dir_str)
    t0 = time.perf_counter()

    logger.info(f"Processing {wire_len_tag} {v0_tag} ({data_dir.name})...")

    # 1. Load parameters
    params = np.load(data_dir / "all_params.npz", allow_pickle=True)
    mu_var = params["mu_var"]
    Vz_var = params["Vz_var"]
    Nmu = len(mu_var)
    Nvz = len(Vz_var)
    v0_val = float(params.get("V0", 0.0))
    Ls = int(params.get("Ls", 300 if wire_len_tag == "3micron" else 600))

    # 2. Ground truth Pfaffian invariant
    pdi_data = np.load(data_dir / "pdi_data.npy")
    col_idx = 3 if pdi_data.shape[1] >= 4 else 2
    gt_flat = (pdi_data[:, col_idx] == 1.0)
    gt_2d = gt_flat.reshape(Nmu, Nvz)

    # 3. Vectorized Barrier Correlation (Step 1)
    b_arr = np.load(data_dir / "barrier_arr.npy")
    bl_l = np.load(data_dir / "barrier_left_conductance_left_arr.npy")
    bl_r = np.load(data_dir / "barrier_left_conductance_right_arr.npy")
    br_l = np.load(data_dir / "barrier_right_conductance_left_arr.npy")
    br_r = np.load(data_dir / "barrier_right_conductance_right_arr.npy")

    cL = vectorized_reflection_correlation(bl_l, bl_r, b_arr).reshape(Nmu, Nvz)
    cR = vectorized_reflection_correlation(br_l, br_r, b_arr).reshape(Nmu, Nvz)
    c_min = np.minimum(cL, cR)
    m1 = (cL >= 0.85) & (cR >= 0.85)

    # 4. TGP Stage 1 & 2 via chunked CPU broadening
    adapter = TGPAdapter(str(data_dir))
    ds = adapter.to_xarray()
    tprep = prepare_sim_cpu(ds, T_mK=30.0)

    t_l = tprep.rename({"bias": "left_bias"})
    t_r = tprep.rename({"bias": "right_bias"})
    t_l, t_r = tgp.two.extract_gap(
        t_l, t_r,
        gap_threshold_factor=0.05,
        noise_threshold=1e-4
    )
    z_ds = tgp.two.zbp_dataset_derivative(
        t_l, t_r,
        zbp_probability_threshold=0.7,
        average_over_cutter=False
    )
    tgp.two.set_gap_threshold(z_ds, threshold_low=10e-3, threshold_high=0.067)

    # Microsoft Stage 2 ROI 2
    s2 = tgp.two.cluster_and_score(
        z_ds,
        min_cluster_size=7,
        cluster_gap_threshold=10e-3,
        cluster_percentage_boundary_threshold=0.6,
        cluster_ncutter_threshold=0.5
    )
    roi2_vb = s2.roi2.values.T if s2.roi2.dims == ("B", "V") else s2.roi2.values
    m_tgp = (roi2_vb > 0)

    # Step 2: Correlation + 3w consensus (>= 50% cutters)
    zbp_mean = z_ds["zbp"].mean(dim="cutter_pair_index").transpose("V", "B").values
    m2 = m1 & (zbp_mean >= 0.5)

    # Step 3: Correlation + 3w + Transport Gap >= 10 ueV consensus
    gzbp_mean = z_ds["gapped_zbp"].mean(dim="cutter_pair_index").transpose("V", "B").values
    m3 = m1 & (gzbp_mean >= 0.5)

    # Average extracted transport gap map in ueV
    gap_uev = (z_ds["gap"].mean(dim="cutter_pair_index").transpose("V", "B").values) * 1000.0

    # Extract individual physical measurements for comparative phase maps
    L_3w = tprep['L_3w'].mean(dim='cutter_pair_index').transpose('V', 'B').values if 'cutter_pair_index' in tprep['L_3w'].dims else tprep['L_3w'].transpose('V', 'B').values
    R_3w = tprep['R_3w'].mean(dim='cutter_pair_index').transpose('V', 'B').values if 'cutter_pair_index' in tprep['R_3w'].dims else tprep['R_3w'].transpose('V', 'B').values
    L_2w = tprep['L_2w_nl'].mean(dim='cutter_pair_index').transpose('V', 'B').values if 'cutter_pair_index' in tprep['L_2w_nl'].dims else tprep['L_2w_nl'].transpose('V', 'B').values
    R_2w = tprep['R_2w_nl'].mean(dim='cutter_pair_index').transpose('V', 'B').values if 'cutter_pair_index' in tprep['R_2w_nl'].dims else tprep['R_2w_nl'].transpose('V', 'B').values
    gap_L = t_l['gap'].mean(dim='cutter_pair_index').transpose('V', 'B').values if 'cutter_pair_index' in t_l['gap'].dims else t_l['gap'].transpose('V', 'B').values
    gap_R = t_r['gap'].mean(dim='cutter_pair_index').transpose('V', 'B').values if 'cutter_pair_index' in t_r['gap'].dims else t_r['gap'].transpose('V', 'B').values

    # Load Confinement and Separability
    conf_2d = load_boundary_confinement(data_dir, Nmu, Nvz, Ls=Ls)
    sep_2d = load_separability(data_dir, Nmu, Nvz)

    # Metrics
    metrics_tgp = calc_protocol_metrics(m_tgp, gt_2d, conf_2d, sep_2d)
    metrics_s1 = calc_protocol_metrics(m1, gt_2d, conf_2d, sep_2d)
    metrics_s2 = calc_protocol_metrics(m2, gt_2d, conf_2d, sep_2d)
    metrics_s3 = calc_protocol_metrics(m3, gt_2d, conf_2d, sep_2d)

    # Load BdG spectrum for E0 and E1 - E0
    spec_file = data_dir / "spectrum_arr.npy"
    if spec_file.exists():
        spec_arr = np.load(spec_file)
        E0 = spec_arr[:, 2].reshape(Nmu, Nvz)
        E1 = spec_arr[:, 3].reshape(Nmu, Nvz)
        E_diff = (E1 - E0).reshape(Nmu, Nvz)
    else:
        E0 = np.zeros((Nmu, Nvz))
        E_diff = np.zeros((Nmu, Nvz))

    return {
        "wire_len_tag": wire_len_tag,
        "v0_tag": v0_tag,
        "V0": v0_val,
        "Ls": Ls,
        "mu_var": mu_var.tolist(),
        "Vz_var": Vz_var.tolist(),
        "extent": [float(Vz_var[0]), float(Vz_var[-1]), float(mu_var[0]), float(mu_var[-1])],
        "gt_mask": gt_2d,
        "c_min": c_min,
        "zbp_mean": zbp_mean,
        "gap_uev": gap_uev,
        "m_tgp": m_tgp,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "metrics_tgp": metrics_tgp,
        "metrics_s1": metrics_s1,
        "metrics_s2": metrics_s2,
        "metrics_s3": metrics_s3,
        "L_3w": np.nan_to_num(L_3w, nan=0.0),
        "R_3w": np.nan_to_num(R_3w, nan=0.0),
        "L_2w": np.nan_to_num(L_2w, nan=0.0),
        "R_2w": np.nan_to_num(R_2w, nan=0.0),
        "gap_L": np.nan_to_num(gap_L, nan=0.0),
        "gap_R": np.nan_to_num(gap_R, nan=0.0),
        "cL": np.nan_to_num(cL, nan=0.0),
        "cR": np.nan_to_num(cR, nan=0.0),
        "E0": np.nan_to_num(E0, nan=0.0),
        "E_diff": np.nan_to_num(E_diff, nan=0.0),
    }


def render_4panel_phase_map(
    panels_data: List[Tuple[np.ndarray, str, str, Dict[str, Any]]],
    gt_mask: np.ndarray,
    extent: List[float],
    title: str,
    out_path: Path
):
    """
    Renders a 2x2 grid of high-contrast phase maps on #0b1120 dark background.
    panels_data: list of 4 items: (mask, color_hex, panel_title, metrics_dict)
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.8), facecolor="#0b1120")

    gt = gt_mask.astype(bool)

    for idx, (ax, (mask, color, p_title, m_dict)) in enumerate(zip(axes.flat, panels_data)):
        ax.set_facecolor("#070d1a")

        # Full passing map representation:
        # - Failed trivial bulk: deep navy #070d1a
        # - Failed topological bulk: slate #1e293b
        # - Passed pixels (both inside AND outside Pfaffian): theme color
        img_rgba = np.zeros((gt.shape[0], gt.shape[1], 4))
        img_rgba[~gt & (~mask)] = [0.03, 0.05, 0.10, 1.0]
        img_rgba[gt & (~mask)] = [0.12, 0.17, 0.25, 0.95]
        
        rgb = mcolors.to_rgb(color)
        img_rgba[mask] = [rgb[0], rgb[1], rgb[2], 0.95]

        ax.imshow(img_rgba, origin="lower", extent=extent, aspect="auto")

        # Overlay Pfaffian Ground Truth Boundary
        if np.sum(gt) > 0:
            ax.contour(gt, levels=[0.5], extent=extent, colors=["#38bdf8"],
                       linestyles=["--"], linewidths=[1.8], origin="lower")

        ax.set_title(p_title, fontsize=12, fontweight="bold", color="#f1f5f9", pad=8)
        ax.set_xlabel(r"$V_z\ [\mathrm{T}]$", fontsize=10.5, color="#cbd5e1")
        ax.set_ylabel(r"$\mu\ [\mathrm{meV}]$", fontsize=10.5, color="#cbd5e1")
        ax.tick_params(colors="#94a3b8", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#334155")

        # Informative badge in upper-left corner
        tpr = m_dict.get("pfaffian_passed_pct", 0.0)
        fdr = m_dict.get("fdr_percent", 0.0)
        text_badge = f"TPR: {tpr:.1f}% | FDR: {fdr:.1f}%"
        ax.text(0.03, 0.94, text_badge, transform=ax.transAxes, fontsize=9.2,
                fontweight="bold", color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", edgecolor="#475569", alpha=0.9))

    # Single unified legend on first axis
    legend_elements = [
        mpatches.Patch(facecolor="#1e293b", edgecolor="#475569", label="Unpassed Topological Bulk (Q = -1)"),
        mpatches.Patch(facecolor="#070d1a", edgecolor="#334155", label="Unpassed Trivial Bulk"),
        plt.Line2D([0], [0], color="#38bdf8", linestyle="--", linewidth=1.8, label="Pfaffian Boundary (Q = -1)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975),
               fontsize=9.5, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

    fig.suptitle(title, fontsize=14, fontweight="bold", color="#38bdf8", y=1.005)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def render_4panel_transition_map(
    d3: Dict[str, Any],
    d6: Dict[str, Any],
    title: str,
    out_path: Path
):
    """
    Renders Slide B for a disorder:
    - Panel 1: Step 1 (Correlation) 3um vs 6um Difference Map
    - Panel 2: Step 2 (3w) 3um vs 6um Difference Map
    - Panel 3: Step 3 (Transport Gap) 3um vs 6um Difference Map
    - Panel 4: Final Protocol Spatial Transition Map (Gained at 6um [Green], Lost at 6um [Red], Preserved [Cyan])
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.8), facecolor="#0b1120")
    extent = d3["extent"]
    gt = d3["gt_mask"].astype(bool)

    steps_cfg = [
        ("Step 1: Correlation (C >= 0.85)", d3["m1"], d6["m1"], d3["metrics_s1"], d6["metrics_s1"]),
        ("Step 2: 3w Measurement Consensus", d3["m2"], d6["m2"], d3["metrics_s2"], d6["metrics_s2"]),
        ("Step 3: Extracted Transport Gap", d3["m3"], d6["m3"], d3["metrics_s3"], d6["metrics_s3"]),
    ]

    # Colors for transition map:
    # Gained at 6um: Vibrant Amber #f59e0b
    # Lost at 6um: Crimson #ef4444
    # Preserved at both: Royal Blue #2563eb
    c_gained = [0.96, 0.62, 0.04, 0.95]
    c_lost = [0.94, 0.27, 0.27, 0.95]
    c_kept = [0.15, 0.39, 0.92, 0.95]

    for idx, (ax, (s_title, m3_b, m6_b, m3_met, m6_met)) in enumerate(zip(axes.flat[:3], steps_cfg)):
        ax.set_facecolor("#070d1a")
        img_rgba = np.zeros((gt.shape[0], gt.shape[1], 4))
        img_rgba[~gt] = [0.03, 0.05, 0.10, 1.0]
        img_rgba[gt] = [0.12, 0.17, 0.25, 0.95]

        # Classification
        kept = m3_b & m6_b
        lost = m3_b & (~m6_b)
        gained = (~m3_b) & m6_b

        img_rgba[kept] = c_kept
        img_rgba[lost] = c_lost
        img_rgba[gained] = c_gained

        ax.imshow(img_rgba, origin="lower", extent=extent, aspect="auto")
        if np.sum(gt) > 0:
            ax.contour(gt, levels=[0.5], extent=extent, colors=["#ffffff"],
                       linestyles=["--"], linewidths=[1.5], origin="lower")

        delta_fdr = m6_met["fdr_percent"] - m3_met["fdr_percent"]
        delta_tpr = m6_met["pfaffian_passed_pct"] - m3_met["pfaffian_passed_pct"]
        d_sign = "+" if delta_fdr >= 0 else ""
        t_sign = "+" if delta_tpr >= 0 else ""

        badge = f"3um: TPR {m3_met['pfaffian_passed_pct']:.1f}%, FDR {m3_met['fdr_percent']:.1f}%\n6um: TPR {m6_met['pfaffian_passed_pct']:.1f}%, FDR {m6_met['fdr_percent']:.1f}%\nΔTPR: {t_sign}{delta_tpr:.1f}% | ΔFDR: {d_sign}{delta_fdr:.1f}%"
        ax.text(0.03, 0.94, badge, transform=ax.transAxes, fontsize=8.8,
                fontweight="bold", color="#ffffff", va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", edgecolor="#475569", alpha=0.9))

        ax.set_title(s_title, fontsize=12, fontweight="bold", color="#f1f5f9", pad=8)
        ax.set_xlabel(r"$V_z\ [\mathrm{T}]$", fontsize=10.5, color="#cbd5e1")
        ax.set_ylabel(r"$\mu\ [\mathrm{meV}]$", fontsize=10.5, color="#cbd5e1")
        ax.tick_params(colors="#94a3b8", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Panel 4: Final Protocol Step 3 Comprehensive Transition Breakdown
    ax4 = axes.flat[3]
    ax4.set_facecolor("#070d1a")
    img_rgba = np.zeros((gt.shape[0], gt.shape[1], 4))
    img_rgba[~gt] = [0.03, 0.05, 0.10, 1.0]
    img_rgba[gt] = [0.12, 0.17, 0.25, 0.95]

    m3_final = d3["m3"]
    m6_final = d6["m3"]
    kept_f = m3_final & m6_final
    lost_f = m3_final & (~m6_final)
    gained_f = (~m3_final) & m6_final

    img_rgba[kept_f] = c_kept
    img_rgba[lost_f] = c_lost
    img_rgba[gained_f] = c_gained

    ax4.imshow(img_rgba, origin="lower", extent=extent, aspect="auto")
    if np.sum(gt) > 0:
        ax4.contour(gt, levels=[0.5], extent=extent, colors=["#ffffff"],
                    linestyles=["--"], linewidths=[1.5], origin="lower")

    ax4.set_title("Full Protocol Spatial Transition (6um vs 3um)", fontsize=12, fontweight="bold", color="#38bdf8", pad=8)
    ax4.set_xlabel(r"$V_z\ [\mathrm{T}]$", fontsize=10.5, color="#cbd5e1")
    ax4.set_ylabel(r"$\mu\ [\mathrm{meV}]$", fontsize=10.5, color="#cbd5e1")
    ax4.tick_params(colors="#94a3b8", labelsize=9.5)
    for spine in ax4.spines.values():
        spine.set_color("#38bdf8")

    total_topo = max(int(np.sum(gt)), 1)
    pct_kept = (int(np.sum(kept_f)) / total_topo) * 100
    pct_gained = (int(np.sum(gained_f)) / total_topo) * 100
    pct_lost = (int(np.sum(lost_f)) / total_topo) * 100
    badge_f = f"Preserved: {pct_kept:.1f}% | Gained: {pct_gained:.1f}% | Lost: {pct_lost:.1f}%"
    ax4.text(0.03, 0.94, badge_f, transform=ax4.transAxes, fontsize=8.8,
             fontweight="bold", color="#ffffff", va="top",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", edgecolor="#0284c7", alpha=0.9))

    # Global legend
    legend_elements = [
        mpatches.Patch(facecolor="#2563eb", edgecolor="none", label="Preserved Across Lengths (3um & 6um)"),
        mpatches.Patch(facecolor="#f59e0b", edgecolor="none", label="Gained at 6um (Stabilized by Length)"),
        mpatches.Patch(facecolor="#ef4444", edgecolor="none", label="Lost at 6um (Suppressed by Length/Disorder)"),
        plt.Line2D([0], [0], color="#ffffff", linestyle="--", linewidth=1.5, label="Pfaffian Boundary (Q = -1)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.975),
               fontsize=9.2, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

    fig.suptitle(title, fontsize=14, fontweight="bold", color="#38bdf8", y=1.005)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def render_measurement_comparison_2x2(
    z_3_left: np.ndarray,
    z_3_right: np.ndarray,
    z_6_left: np.ndarray,
    z_6_right: np.ndarray,
    gt_3: np.ndarray,
    gt_6: np.ndarray,
    extent: List[float],
    vmin: float,
    vmax: float,
    cmap: str,
    cbar_label: str,
    title: str,
    subtitle_left: str,
    subtitle_right: str,
    out_path: Path,
):
    """
    Renders a 2x2 comparative phase map on #0b1120 dark background:
      Row 1: 3 μm Wire (Left Lead vs. Right Lead)
      Row 2: 6 μm Wire (Left Lead vs. Right Lead)
    Overlays only the ground truth topological boundary (cyan dashed).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.2), facecolor="#0b1120")
    
    gt3_b = gt_3.astype(bool)
    gt6_b = gt_6.astype(bool)

    panels = [
        (axes[0, 0], z_3_left, gt3_b, f"3 μm Wire: {subtitle_left}"),
        (axes[0, 1], z_3_right, gt3_b, f"3 μm Wire: {subtitle_right}"),
        (axes[1, 0], z_6_left, gt6_b, f"6 μm Wire: {subtitle_left}"),
        (axes[1, 1], z_6_right, gt6_b, f"6 μm Wire: {subtitle_right}"),
    ]

    im_ref = None
    for ax, data, gt_mask, p_title in panels:
        ax.set_facecolor("#070d1a")
        
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )
        if im_ref is None:
            im_ref = im

        # Overlay Pfaffian ground truth topological boundary
        if np.sum(gt_mask) > 0:
            ax.contour(
                gt_mask,
                levels=[0.5],
                extent=extent,
                colors=["#38bdf8"],
                linestyles=["--"],
                linewidths=[1.8],
                origin="lower"
            )

        ax.set_title(p_title, fontsize=12, fontweight="bold", color="#f1f5f9", pad=8)
        ax.set_xlabel(r"$V_z\ [\mathrm{T}]$", fontsize=10.5, color="#cbd5e1")
        ax.set_ylabel(r"$\mu\ [\mathrm{meV}]$", fontsize=10.5, color="#cbd5e1")
        ax.tick_params(colors="#94a3b8", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Unified legend at the top: only Pfaffian boundary
    legend_elements = [
        plt.Line2D([0], [0], color="#38bdf8", linestyle="--", linewidth=1.8, label="Pfaffian Boundary (Q = -1)"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=1,
        bbox_to_anchor=(0.48, 0.98),
        fontsize=10,
        facecolor="#0f172a",
        edgecolor="#334155",
        labelcolor="#e2e8f0"
    )

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.915, 0.12, 0.02, 0.76])
    cbar = fig.colorbar(im_ref, cax=cbar_ax)
    cbar.set_label(cbar_label, color="#e2e8f0", fontsize=11, fontweight="bold")
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#cbd5e1", fontsize=9.5)
    cbar.outline.set_edgecolor("#334155")

    fig.suptitle(title, fontsize=14, fontweight="bold", color="#38bdf8", y=1.01)
    fig.subplots_adjust(left=0.07, right=0.895, bottom=0.08, top=0.92, wspace=0.18, hspace=0.25)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def render_energy_comparison_2x2(
    extent: List[float],
    gt_3: np.ndarray,
    gt_6: np.ndarray,
    e0_3: np.ndarray,
    diff_3: np.ndarray,
    e0_6: np.ndarray,
    diff_6: np.ndarray,
    title: str,
    out_path: Path,
):
    """
    Renders a 2x2 comparative phase map on #0b1120 dark background:
      Row 1: 3 μm Wire: Lowest Positive Energy State E_0 (Left) vs. Excited State Difference E_1 - E_0 (Right)
      Row 2: 6 μm Wire: Lowest Positive Energy State E_0 (Left) vs. Excited State Difference E_1 - E_0 (Right)
    Overlays only the ground truth topological boundary (cyan dashed).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.2), facecolor="#0b1120")
    
    gt3_b = gt_3.astype(bool)
    gt6_b = gt_6.astype(bool)

    panels = [
        (axes[0, 0], e0_3, gt3_b, r"3 μm Wire: Lowest Positive Energy $E_0$", "magma", 0.0, 0.06),
        (axes[0, 1], diff_3, gt3_b, r"3 μm Wire: Quasiparticle Gap $(E_1 - E_0)$", "viridis", 0.0, 0.05),
        (axes[1, 0], e0_6, gt6_b, r"6 μm Wire: Lowest Positive Energy $E_0$", "magma", 0.0, 0.06),
        (axes[1, 1], diff_6, gt6_b, r"6 μm Wire: Quasiparticle Gap $(E_1 - E_0)$", "viridis", 0.0, 0.05),
    ]

    im_e0, im_diff = None, None
    for (ax, data, gt_mask, p_title, cmap, vmin, vmax) in panels:
        ax.set_facecolor("#070d1a")
        
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )
        if cmap == "magma" and im_e0 is None:
            im_e0 = im
        elif cmap == "viridis" and im_diff is None:
            im_diff = im

        # Overlay Pfaffian ground truth topological boundary
        if np.sum(gt_mask) > 0:
            ax.contour(
                gt_mask,
                levels=[0.5],
                extent=extent,
                colors=["#38bdf8"],
                linestyles=["--"],
                linewidths=[1.8],
                origin="lower"
            )

        ax.set_title(p_title, fontsize=12, fontweight="bold", color="#f1f5f9", pad=8)
        ax.set_xlabel(r"$V_z\ [\mathrm{T}]$", fontsize=10.5, color="#cbd5e1")
        ax.set_ylabel(r"$\mu\ [\mathrm{meV}]$", fontsize=10.5, color="#cbd5e1")
        ax.tick_params(colors="#94a3b8", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Unified legend at the top
    legend_elements = [
        plt.Line2D([0], [0], color="#38bdf8", linestyle="--", linewidth=1.8, label="Pfaffian Boundary (Q = -1)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=1,
        bbox_to_anchor=(0.50, 0.93),
        fontsize=10,
        facecolor="#0f172a",
        edgecolor="#334155",
        labelcolor="#e2e8f0"
    )

    # Colorbars: one for column 0 (E0) and one for column 1 (E1 - E0)
    cbar_ax0 = fig.add_axes([0.455, 0.08, 0.015, 0.77])
    cbar0 = fig.colorbar(im_e0, cax=cbar_ax0)
    cbar0.set_label(r"Lowest Positive Energy $E_0$ (meV)", color="#e2e8f0", fontsize=10.5, fontweight="bold")
    cbar0.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar0.ax.axes, 'yticklabels'), color="#cbd5e1", fontsize=9.0)
    cbar0.outline.set_edgecolor("#334155")

    cbar_ax1 = fig.add_axes([0.935, 0.08, 0.015, 0.77])
    cbar1 = fig.colorbar(im_diff, cax=cbar_ax1)
    cbar1.set_label(r"Excited State Difference $E_1 - E_0$ (meV)", color="#e2e8f0", fontsize=10.5, fontweight="bold")
    cbar1.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color="#cbd5e1", fontsize=9.0)
    cbar1.outline.set_edgecolor("#334155")

    fig.suptitle(title, fontsize=14, fontweight="bold", color="#38bdf8", y=0.985)
    fig.subplots_adjust(left=0.06, right=0.915, bottom=0.08, top=0.88, wspace=0.42, hspace=0.25)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def render_global_sensitivity_plots(summary_list: List[Dict[str, Any]]):
    """
    Renders the 3 global sensitivity overview figures:
    - Slide 2: Step 1 (Correlation Condition) Sensitivity
    - Slide 3: Step 2 (3w Measurement) Sensitivity
    - Slide 4: Step 3 (Extracted Transport Gap) Sensitivity
    """
    v0_vals = [s["V0"] for s in summary_list]
    v0_labels = [s["tag"] for s in summary_list]
    x_indices = np.arange(len(v0_vals))

    # Helper for dual-axis trend plots
    def plot_step_global(step_key: str, step_name: str, color_main: str, out_filename: str):
        fig, (ax_area, ax_fdr) = plt.subplots(1, 2, figsize=(13.0, 5.5), facecolor="#0b1120")

        # 1. Total Passed Area & TPR
        area_3um = [s["3micron"][step_key]["tot_passed"] for s in summary_list]
        area_6um = [s["6micron"][step_key]["tot_passed"] for s in summary_list]
        tpr_3um = [s["3micron"][step_key]["pfaffian_passed_pct"] for s in summary_list]
        tpr_6um = [s["6micron"][step_key]["pfaffian_passed_pct"] for s in summary_list]

        width = 0.35
        ax_area.set_facecolor("#070d1a")
        bars1 = ax_area.bar(x_indices - width/2, area_3um, width, label="3 μm Wire", color="#38bdf8", alpha=0.85)
        bars2 = ax_area.bar(x_indices + width/2, area_6um, width, label="6 μm Wire", color="#818cf8", alpha=0.85)
        
        ax_area.set_xticks(x_indices)
        ax_area.set_xticklabels(v0_labels, rotation=25, ha="right", color="#cbd5e1", fontsize=9.5)
        ax_area.set_ylabel("Total Passed Pixels", color="#cbd5e1", fontsize=11)
        ax_area.set_title(f"Passed Area Scaling across Wire Length\n({step_name})", color="#f1f5f9", fontsize=12, fontweight="bold")
        ax_area.tick_params(colors="#94a3b8")
        ax_area.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
        for spine in ax_area.spines.values():
            spine.set_color("#334155")
        ax_area.grid(axis="y", color="#1e293b", linestyle=":", alpha=0.7)

        # 2. FDR & Topological Area Retention (TPR)
        fdr_3um = [s["3micron"][step_key]["fdr_percent"] for s in summary_list]
        fdr_6um = [s["6micron"][step_key]["fdr_percent"] for s in summary_list]

        ax_fdr.set_facecolor("#070d1a")
        ax_fdr.plot(x_indices, fdr_3um, marker="o", markersize=8, linewidth=2.4, color="#38bdf8", label="3 μm FDR (%)")
        ax_fdr.plot(x_indices, fdr_6um, marker="s", markersize=8, linewidth=2.4, color="#f43f5e", label="6 μm FDR (%)")
        
        ax_fdr.plot(x_indices, tpr_3um, marker="^", markersize=7, linewidth=1.8, linestyle="--", color="#34d399", label="3 μm TPR (% Area)")
        ax_fdr.plot(x_indices, tpr_6um, marker="v", markersize=7, linewidth=1.8, linestyle="--", color="#a7f3d0", label="6 μm TPR (% Area)")

        ax_fdr.set_xticks(x_indices)
        ax_fdr.set_xticklabels(v0_labels, rotation=25, ha="right", color="#cbd5e1", fontsize=9.5)
        ax_fdr.set_ylabel("Rate (%)", color="#cbd5e1", fontsize=11)
        ax_fdr.set_title(f"False Discovery Rate & Topological Retention\n({step_name})", color="#f1f5f9", fontsize=12, fontweight="bold")
        ax_fdr.tick_params(colors="#94a3b8")
        ax_fdr.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
        for spine in ax_fdr.spines.values():
            spine.set_color("#334155")
        ax_fdr.grid(color="#1e293b", linestyle=":", alpha=0.7)

        fig.suptitle(f"Global Length Sensitivity: {step_name} (6 μm vs. 3 μm)", fontsize=13.5, fontweight="bold", color="#38bdf8", y=0.99)
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
        fig.savefig(FIGURES_DIR / out_filename, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

    plot_step_global("metrics_s1", "Step 1: Barrier Reflection Correlation (C >= 0.85)", "#38bdf8", "global_slide2_correlation_sensitivity.png")
    plot_step_global("metrics_s2", "Step 2: 3ω Peak Derivative Consensus", "#818cf8", "global_slide3_3w_sensitivity.png")
    plot_step_global("metrics_s3", "Step 3: Full Protocol (Extracted Gap >= 10 ueV)", "#34d399", "global_slide4_gap_sensitivity.png")


def main():
    logger.info("=== STARTING 6um vs. 3um WIRE LENGTH SENSITIVITY POST-PROCESSING ===")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    base_data = REPO_ROOT / "Outputs" / "Data" / "six_micron_analysis"
    tasks = []

    for tag, v0_val, desc in PAIRED_DISORDERS:
        d3_path = base_data / "3micron" / f"{tag}_realization_0"
        d6_path = base_data / "6micron" / f"{tag}_realization_0"

        if not d3_path.exists():
            raise FileNotFoundError(f"Missing 3micron directory: {d3_path}")
        if not d6_path.exists():
            raise FileNotFoundError(f"Missing 6micron directory: {d6_path}")

        tasks.append((str(d3_path), "3micron", tag))
        tasks.append((str(d6_path), "6micron", tag))

    n_workers = min(len(tasks), max(1, cpu_count() - 2))
    logger.info(f"Dispatching {len(tasks)} simulation datasets across {n_workers} CPU workers...")

    with Pool(n_workers) as pool:
        results_list = pool.map(process_single_dataset, tasks)

    # Organize results by disorder tag and wire length
    results_dict = {}
    for res in results_list:
        tag = res["v0_tag"]
        wtag = res["wire_len_tag"]
        if tag not in results_dict:
            results_dict[tag] = {}
        results_dict[tag][wtag] = res

    # Generate figures and compile summary metrics
    summary_data = []

    for tag, v0_val, desc in PAIRED_DISORDERS:
        d3 = results_dict[tag]["3micron"]
        d6 = results_dict[tag]["6micron"]
        extent = d3["extent"]
        gt_mask = d3["gt_mask"]

        logger.info(f"Rendering figures for {tag} ({desc})...")

        # Slide A: Protocol Comparison (4 large panels)
        # Top-Left: 3um TGP; Top-Right: 6um TGP; Bottom-Left: 3um Step 3; Bottom-Right: 6um Step 3
        slideA_panels = [
            (d3["m_tgp"], "#10b981", "3 μm Wire: Microsoft TGP Stage 2 (ROI 2)", d3["metrics_tgp"]),
            (d6["m_tgp"], "#10b981", "6 μm Wire: Microsoft TGP Stage 2 (ROI 2)", d6["metrics_tgp"]),
            (d3["m3"], "#10b981", "3 μm Wire: 3-Step Protocol (Full ROI 3)", d3["metrics_s3"]),
            (d6["m3"], "#10b981", "6 μm Wire: 3-Step Protocol (Full ROI 3)", d6["metrics_s3"]),
        ]
        slideA_title = f"Protocol Comparison (TGP vs. 3-Step Protocol): {desc}"
        slideA_path = FIGURES_DIR / f"{tag}_slideA_protocol_comparison.png"
        render_4panel_phase_map(slideA_panels, gt_mask, extent, slideA_title, slideA_path)

        # Slide B: Step-by-Step Sensitivity & Transition (4 large panels)
        slideB_title = f"Protocol Steps Length Sensitivity & Spatial Transition: {desc}"
        slideB_path = FIGURES_DIR / f"{tag}_slideB_step_sensitivity.png"
        render_4panel_transition_map(d3, d6, slideB_title, slideB_path)

        # Measurement 1: 3w Peak Curvature Comparison (3um vs 6um)
        m1_title = f"3ω Peak Curvature Comparison (3μm vs. 6μm): {desc}"
        m1_path = FIGURES_DIR / f"{tag}_comparison_3w.png"
        render_measurement_comparison_2x2(
            z_3_left=np.clip(d3["L_3w"], -130.0, 130.0),
            z_3_right=np.clip(d3["R_3w"], -130.0, 130.0),
            z_6_left=np.clip(d6["L_3w"], -130.0, 130.0),
            z_6_right=np.clip(d6["R_3w"], -130.0, 130.0),
            gt_3=d3["gt_mask"],
            gt_6=d6["gt_mask"],
            extent=extent,
            vmin=-130.0,
            vmax=130.0,
            cmap="RdBu_r",
            cbar_label=r"3ω Peak Curvature $d^2G/dV^2$",
            title=m1_title,
            subtitle_left="Average Left Lead 3ω",
            subtitle_right="Average Right Lead 3ω",
            out_path=m1_path,
        )

        # Measurement 2: Extracted Transport Gap Comparison (3um vs 6um)
        m2_title = f"Extracted Transport Gap Comparison (3μm vs. 6μm): {desc}"
        m2_path = FIGURES_DIR / f"{tag}_comparison_transport_gap.png"
        render_measurement_comparison_2x2(
            z_3_left=d3["gap_L"],
            z_3_right=d3["gap_R"],
            z_6_left=d6["gap_L"],
            z_6_right=d6["gap_R"],
            gt_3=d3["gt_mask"],
            gt_6=d6["gt_mask"],
            extent=extent,
            vmin=0.0,
            vmax=0.05,
            cmap="gist_heat_r",
            cbar_label=r"Extracted Transport Gap $\Delta_{\mathrm{transport}}$ (meV)",
            title=m2_title,
            subtitle_left=r"Gap from Nonlocal $G_{RL}$ (Left Injection)",
            subtitle_right=r"Gap from Nonlocal $G_{LR}$ (Right Injection)",
            out_path=m2_path,
        )

        # Measurement 3: 2w Conductance Comparison (3um vs 6um)
        m3_title = f"2ω Zero-Bias Conductance Comparison (3μm vs. 6μm): {desc}"
        m3_path = FIGURES_DIR / f"{tag}_comparison_2w.png"
        render_measurement_comparison_2x2(
            z_3_left=d3["L_2w"],
            z_3_right=d3["R_2w"],
            z_6_left=d6["L_2w"],
            z_6_right=d6["R_2w"],
            gt_3=d3["gt_mask"],
            gt_6=d6["gt_mask"],
            extent=extent,
            vmin=-2.0,
            vmax=2.0,
            cmap="RdBu_r",
            cbar_label=r"Average $2\omega$ Conductance ($e^2/h$)",
            title=m3_title,
            subtitle_left="Average Left Lead 2ω Conductance",
            subtitle_right="Average Right Lead 2ω Conductance",
            out_path=m3_path,
        )

        # Measurement 4: Barrier Reflection Correlation Comparison (3um vs 6um)
        m4_title = f"Barrier Reflection Correlation Comparison (3μm vs. 6μm): {desc}"
        m4_path = FIGURES_DIR / f"{tag}_comparison_barrier_correlation.png"
        render_measurement_comparison_2x2(
            z_3_left=d3["cL"],
            z_3_right=d3["cR"],
            z_6_left=d6["cL"],
            z_6_right=d6["cR"],
            gt_3=d3["gt_mask"],
            gt_6=d6["gt_mask"],
            extent=extent,
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            cbar_label=r"Reflection Correlation $C(f_1, f_2)$",
            title=m4_title,
            subtitle_left="Left Barrier Reflection Correlation (C_L)",
            subtitle_right="Right Barrier Reflection Correlation (C_R)",
            out_path=m4_path,
        )

        # Measurement 5: BdG Ground State Energy (E0) & Quasiparticle Gap (E1 - E0) Comparison (3um vs 6um)
        m5_title = f"BdG Energy ($E_0$) & Quasiparticle Excitation Gap ($E_1 - E_0$): {desc}"
        m5_path = FIGURES_DIR / f"{tag}_comparison_energy_diff.png"
        render_energy_comparison_2x2(
            extent=extent,
            gt_3=d3["gt_mask"],
            gt_6=d6["gt_mask"],
            e0_3=d3["E0"],
            diff_3=d3["E_diff"],
            e0_6=d6["E0"],
            diff_6=d6["E_diff"],
            title=m5_title,
            out_path=m5_path,
        )

        # Summary Record
        summary_record = {
            "tag": tag,
            "V0": v0_val,
            "description": desc,
            "3micron": {
                "tgp": d3["metrics_tgp"],
                "metrics_s1": d3["metrics_s1"],
                "metrics_s2": d3["metrics_s2"],
                "metrics_s3": d3["metrics_s3"],
            },
            "6micron": {
                "tgp": d6["metrics_tgp"],
                "metrics_s1": d6["metrics_s1"],
                "metrics_s2": d6["metrics_s2"],
                "metrics_s3": d6["metrics_s3"],
            },
            "deltas_step3": {
                "delta_passed_pixels": d6["metrics_s3"]["tot_passed"] - d3["metrics_s3"]["tot_passed"],
                "delta_tpr": round(d6["metrics_s3"]["pfaffian_passed_pct"] - d3["metrics_s3"]["pfaffian_passed_pct"], 2),
                "delta_fdr": round(d6["metrics_s3"]["fdr_percent"] - d3["metrics_s3"]["fdr_percent"], 2),
                "delta_confinement": round(d6["metrics_s3"]["mean_boundary_confinement"] - d3["metrics_s3"]["mean_boundary_confinement"], 4),
                "delta_separability": round(d6["metrics_s3"]["mean_separability"] - d3["metrics_s3"]["mean_separability"], 4),
            },
            "deltas_tgp": {
                "delta_passed_pixels": d6["metrics_tgp"]["tot_passed"] - d3["metrics_tgp"]["tot_passed"],
                "delta_tpr": round(d6["metrics_tgp"]["pfaffian_passed_pct"] - d3["metrics_tgp"]["pfaffian_passed_pct"], 2),
                "delta_fdr": round(d6["metrics_tgp"]["fdr_percent"] - d3["metrics_tgp"]["fdr_percent"], 2),
                "delta_confinement": round(d6["metrics_tgp"]["mean_boundary_confinement"] - d3["metrics_tgp"]["mean_boundary_confinement"], 4),
                "delta_separability": round(d6["metrics_tgp"]["mean_separability"] - d3["metrics_tgp"]["mean_separability"], 4),
            }
        }
        summary_data.append(summary_record)

    # Render global sensitivity overview plots
    logger.info("Rendering global sensitivity overview figures...")
    render_global_sensitivity_plots(summary_data)

    # Save summary metrics JSON
    metrics_json_path = METRICS_DIR / "length_comparison_summary.json"
    with open(metrics_json_path, "w") as fp:
        json.dump(summary_data, fp, indent=2)

    logger.info(f"Successfully saved summary metrics to {metrics_json_path}")
    logger.info("=== POST-PROCESSING & FIGURE GENERATION COMPLETE ===")


if __name__ == "__main__":
    main()
