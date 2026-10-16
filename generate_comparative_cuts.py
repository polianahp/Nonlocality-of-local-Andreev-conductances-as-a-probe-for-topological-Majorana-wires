#!/usr/bin/env python3
"""
generate_comparative_cuts.py

Generates 1:1 comparative single-cut analyses comparing 3 μm vs 6 μm nanowires:
1. 10-Panel Conductance Multiplot:
   - Row 0: BdG Energy Spectra (lowest 14 eigenvalues, highlighting E0/E1, with TGP ROI 2 and Protocol ROI 3 runners)
   - Row 1: G_LL vs. bias
   - Row 2: G_RR vs. bias
   - Row 3: A(G_RL) vs. bias with +/- Delta_transport overlay
   - Row 4: A(G_LR) vs. bias with +/- Delta_transport overlay
2. Comparative Cut Analytics Multiplot (4 rows x 2 columns):
   - Row 0: BdG Spectrum + Protocol Acceptance Runners + Ground Truth Shading
   - Row 1: 3w Peak Curvature vs. -100 threshold
   - Row 2: Extracted Transport Gap vs. Lowest BdG States (E0, E1) and 10 ueV threshold
   - Row 3: Binarized Barrier Reflection Correlation Track (C_L, C_R >= 0.85)
3. 2D Gapless Boundary Diagnostic Map (2x2):
   - Directly illustrates why Microsoft TGP fails on 6 μm wire due to the collapse of the gapless boundary halo.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr
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
from cut_analysis import _eval_cut_point_worker
from post_process_length_comparison import (
    vectorized_reflection_correlation,
    calc_protocol_metrics,
    load_boundary_confinement,
    load_separability,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ComparativeCuts")

OUTPUT_DIR = REPO_ROOT / "Outputs" / "slides" / "six_micron_analysis" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_DATA = REPO_ROOT / "Outputs" / "Data" / "six_micron_analysis"

TARGET_CUTS = [
    {
        "disorder_tag": "V0_0_645",
        "title": "Strong Benchmark Disorder (V0 = 0.645 meV) — Island 1 COM Cut",
        "mu_cut": 0.975,
        "vz_min": 0.15,
        "vz_max": 0.95,
        "n_pts": 100,
        "filename_tag": "V0_0_645_cut_island1",
    },
    {
        "disorder_tag": "V0_0_0",
        "title": "Clean Nanowire (V0 = 0.0 meV) — Transverse Cut (μ = 0.0 meV)",
        "mu_cut": 0.000,
        "vz_min": 0.00,
        "vz_max": 1.20,
        "n_pts": 100,
        "filename_tag": "V0_0_0_cut_clean",
    },
]


def load_wire_simulation(data_dir: Path, wire_len_tag: str):
    """Loads simulation data, computes tprep, extracted gap, and protocol masks."""
    logger.info(f"Loading {wire_len_tag} data from {data_dir.name}...")
    params = np.load(data_dir / "all_params.npz", allow_pickle=True)
    mu_var = params["mu_var"]
    Vz_var = params["Vz_var"]
    Nmu = len(mu_var)
    Nvz = len(Vz_var)
    Ls = int(params.get("Ls", 300 if wire_len_tag == "3micron" else 600))
    V0 = float(params.get("V0", 0.0))
    Vdisx = params.get("Vdisx", np.zeros(Ls)) * V0

    # Ground truth Pfaffian invariant
    pdi_data = np.load(data_dir / "pdi_data.npy")
    col_idx = 3 if pdi_data.shape[1] >= 4 else 2
    gt_2d = (pdi_data[:, col_idx] == 1.0).reshape(Nmu, Nvz)

    # Barrier correlation
    b_arr = np.load(data_dir / "barrier_arr.npy")
    bl_l = np.load(data_dir / "barrier_left_conductance_left_arr.npy")
    bl_r = np.load(data_dir / "barrier_left_conductance_right_arr.npy")
    br_l = np.load(data_dir / "barrier_right_conductance_left_arr.npy")
    br_r = np.load(data_dir / "barrier_right_conductance_right_arr.npy")
    cL = vectorized_reflection_correlation(bl_l, bl_r, b_arr).reshape(Nmu, Nvz)
    cR = vectorized_reflection_correlation(br_l, br_r, b_arr).reshape(Nmu, Nvz)
    m1 = (cL >= 0.85) & (cR >= 0.85)

    # TGP adapter and broadening
    adapter = TGPAdapter(str(data_dir))
    ds = adapter.to_xarray()
    tprep = prepare_sim_cpu(ds, T_mK=30.0)

    t_l = tprep.rename({"bias": "left_bias"})
    t_r = tprep.rename({"bias": "right_bias"})
    t_l, t_r = tgp.two.extract_gap(t_l, t_r, gap_threshold_factor=0.05, noise_threshold=1e-4)
    z_ds = tgp.two.zbp_dataset_derivative(t_l, t_r, zbp_probability_threshold=0.7, average_over_cutter=False)
    tgp.two.set_gap_threshold(z_ds, threshold_low=10e-3, threshold_high=0.067)

    s2 = tgp.two.cluster_and_score(
        z_ds,
        min_cluster_size=7,
        cluster_gap_threshold=10e-3,
        cluster_percentage_boundary_threshold=0.6,
        cluster_ncutter_threshold=0.5
    )
    roi2_vb = s2.roi2.values.T if s2.roi2.dims == ("B", "V") else s2.roi2.values
    m_tgp = (roi2_vb > 0)

    gzbp_mean = z_ds["gapped_zbp"].mean(dim="cutter_pair_index").transpose("V", "B").values
    m3 = m1 & (gzbp_mean >= 0.5)

    # Extracted gap
    gap_left = t_l["gap"].mean(dim="cutter_pair_index").transpose("V", "B").values if "cutter_pair_index" in t_l["gap"].dims else t_l["gap"].transpose("V", "B").values
    gap_right = t_r["gap"].mean(dim="cutter_pair_index").transpose("V", "B").values if "cutter_pair_index" in t_r["gap"].dims else t_r["gap"].transpose("V", "B").values
    gap_min_2d = np.minimum(gap_left, gap_right)

    # 3w curvature
    L_3w = tprep['L_3w'].mean(dim='cutter_pair_index').transpose('V', 'B').values if 'cutter_pair_index' in tprep['L_3w'].dims else tprep['L_3w'].transpose('V', 'B').values
    R_3w = tprep['R_3w'].mean(dim='cutter_pair_index').transpose('V', 'B').values if 'cutter_pair_index' in tprep['R_3w'].dims else tprep['R_3w'].transpose('V', 'B').values
    avg_3w = 0.5 * (L_3w + R_3w)

    return {
        "params": params,
        "mu_var": mu_var,
        "Vz_var": Vz_var,
        "Ls": Ls,
        "V0": V0,
        "Vdisx": Vdisx,
        "gt_2d": gt_2d,
        "tprep": tprep,
        "t_l": t_l,
        "t_r": t_r,
        "m_tgp": m_tgp,
        "m3": m3,
        "gap_left": gap_left,
        "gap_right": gap_right,
        "gap_min_2d": gap_min_2d,
        "avg_3w": avg_3w,
        "cL": cL,
        "cR": cR,
    }


def extract_cut_data(sim_dict: Dict[str, Any], mu_cut: float, vz_pts: np.ndarray, n_evals: int = 14):
    """Solves BdG Hamiltonian and slices conductances along (mu_cut, vz_pts)."""
    params = sim_dict["params"]
    Ls = sim_dict["Ls"]
    Vdisx = sim_dict["Vdisx"]
    t_val = float(params["t"])
    gamma = float(params["gamma"])
    Delta0 = float(params["Delta0"])
    alpha = float(params["alpha"])

    actual_N = len(vz_pts)
    worker_args = [
        (i, mu_cut, vz_pts[i], t_val, gamma, Delta0, alpha, Ls, Vdisx, n_evals)
        for i in range(actual_N)
    ]
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_eval_cut_point_worker, worker_args))
    results.sort(key=lambda x: x[0])
    evals = np.array([r[1] for r in results])  # (N, n_evals)

    tprep = sim_dict["tprep"]
    t_l = sim_dict["t_l"]
    t_r = sim_dict["t_r"]
    bias = tprep["bias"].values
    bias_uV = bias * 1e3

    tprep_sub = tprep.mean(dim="cutter_pair_index") if "cutter_pair_index" in tprep.dims else tprep
    t_l_sub = t_l.mean(dim="cutter_pair_index") if "cutter_pair_index" in t_l.dims else t_l
    t_r_sub = t_r.mean(dim="cutter_pair_index") if "cutter_pair_index" in t_r.dims else t_r

    g_ll_cut = np.zeros((actual_N, len(bias)))
    g_rr_cut = np.zeros((actual_N, len(bias)))
    g_rl_cut = np.zeros((actual_N, len(bias)))
    g_lr_cut = np.zeros((actual_N, len(bias)))
    gap_l_cut = np.zeros(actual_N)
    gap_r_cut = np.zeros(actual_N)
    val_3w = np.zeros(actual_N)
    val_corr = np.zeros(actual_N, dtype=bool)
    val_pfaff = np.zeros(actual_N, dtype=bool)
    val_tgp = np.zeros(actual_N, dtype=bool)
    val_roi3 = np.zeros(actual_N, dtype=bool)

    mu_var = sim_dict["mu_var"]
    vz_var = sim_dict["Vz_var"]
    gt_2d = sim_dict["gt_2d"]
    cL = sim_dict["cL"]
    cR = sim_dict["cR"]
    m_tgp = sim_dict["m_tgp"]
    m3 = sim_dict["m3"]
    avg_3w = sim_dict["avg_3w"]

    mu_idx = np.argmin(np.abs(mu_var - mu_cut))

    for i in range(actual_N):
        bv = vz_pts[i]
        b_idx = np.argmin(np.abs(vz_var - bv))

        g_ll_cut[i, :] = tprep_sub.g_ll.sel(V=mu_cut, B=bv, method="nearest").values
        g_rr_cut[i, :] = tprep_sub.g_rr.sel(V=mu_cut, B=bv, method="nearest").values
        g_rl_cut[i, :] = tprep_sub.g_rl.sel(V=mu_cut, B=bv, method="nearest").values
        g_lr_cut[i, :] = tprep_sub.g_lr.sel(V=mu_cut, B=bv, method="nearest").values

        gap_l_cut[i] = t_l_sub.gap.sel(V=mu_cut, B=bv, method="nearest").values.item()
        gap_r_cut[i] = t_r_sub.gap.sel(V=mu_cut, B=bv, method="nearest").values.item()

        val_3w[i] = avg_3w[mu_idx, b_idx]
        val_corr[i] = bool((cL[mu_idx, b_idx] >= 0.85) and (cR[mu_idx, b_idx] >= 0.85))
        val_pfaff[i] = bool(gt_2d[mu_idx, b_idx])
        val_tgp[i] = bool(m_tgp[mu_idx, b_idx])
        val_roi3[i] = bool(m3[mu_idx, b_idx])

    a_grl_cut = 0.5 * (g_rl_cut - g_rl_cut[:, ::-1])
    a_glr_cut = 0.5 * (g_lr_cut - g_lr_cut[:, ::-1])

    return {
        "vz_pts": vz_pts,
        "bias_uV": bias_uV,
        "evals": evals,
        "g_ll": g_ll_cut,
        "g_rr": g_rr_cut,
        "a_grl": a_grl_cut,
        "a_glr": a_glr_cut,
        "gap_l": gap_l_cut,
        "gap_r": gap_r_cut,
        "val_3w": val_3w,
        "val_corr": val_corr,
        "val_pfaff": val_pfaff,
        "val_tgp": val_tgp,
        "val_roi3": val_roi3,
    }


def render_10panel_comparison(
    c3: Dict[str, Any],
    c6: Dict[str, Any],
    mu_cut: float,
    title: str,
    out_path: Path
):
    """
    Renders 10-panel comparative conductance multiplot (3 μm left, 6 μm right).
    Rows:
      0: BdG Energy Spectrum (with TGP and ROI 3 indicators)
      1: G_LL
      2: G_RR
      3: A(G_RL) with +/- Delta_transport
      4: A(G_LR) with +/- Delta_transport
    """
    fig, axes = plt.subplots(5, 2, figsize=(15.0, 13.5), sharex=True, facecolor="#0b1120")
    plt.subplots_adjust(left=0.07, right=0.92, bottom=0.06, top=0.93, wspace=0.18, hspace=0.12)

    g_local_max = max(1.0, float(np.nanpercentile([c3["g_ll"], c3["g_rr"], c6["g_ll"], c6["g_rr"]], 99.0)))
    g_nonlocal_max = max(0.04, float(np.nanpercentile(np.abs([c3["a_grl"], c3["a_glr"], c6["a_grl"], c6["a_glr"]]), 99.0)))

    wire_configs = [
        (0, c3, "3 μm Nanowire"),
        (1, c6, "6 μm Nanowire"),
    ]

    for col, cdata, wlabel in wire_configs:
        vz_pts = cdata["vz_pts"]
        bias_uV = cdata["bias_uV"]
        actual_N = len(vz_pts)
        dx = (vz_pts[-1] - vz_pts[0]) / (actual_N - 1) if actual_N > 1 else 1.0

        for r in range(5):
            axes[r, col].set_facecolor("#070d1a")
            for spine in axes[r, col].spines.values():
                spine.set_color("#334155")
            axes[r, col].tick_params(colors="#94a3b8", labelsize=9)

        # ----------------------------------------------------
        # Row 0: BdG Energy Spectrum
        # ----------------------------------------------------
        ax_sp = axes[0, col]
        evals = cdata["evals"]
        kvals = evals.shape[1]
        mid = kvals // 2

        # Shade ground truth Pfaffian domain in slate blue
        for i in range(actual_N):
            if cdata["val_pfaff"][i]:
                ax_sp.axvspan(vz_pts[i] - 0.5*dx, vz_pts[i] + 0.5*dx, color="#38bdf8", alpha=0.15, zorder=0)

        for j in range(kvals):
            is_mzm = j in [mid - 1, mid]
            col_line = "#ef4444" if is_mzm else "#3b82f6"
            lw = 2.2 if is_mzm else 1.1
            ax_sp.plot(vz_pts, evals[:, j], color=col_line, linewidth=lw, alpha=0.85, zorder=2)

        ax_sp.axhline(0, color="#64748b", linestyle="--", linewidth=0.8, alpha=0.6)
        ymin = np.min(evals) - 0.08 * (np.max(evals) - np.min(evals))
        ymax = np.max(evals) + 0.08 * (np.max(evals) - np.min(evals))
        ax_sp.set_ylim(ymin, ymax)
        y_bar = ymin + 0.04 * (ymax - ymin)

        # Protocol Runners at bottom
        tgp_plotted = False
        for i in range(actual_N):
            if cdata["val_tgp"][i]:
                ax_sp.plot(vz_pts[i], ymin, marker="|", color="#ef4444", markersize=9, markeredgewidth=2.0, zorder=6,
                           label="Microsoft TGP (ROI 2)" if not tgp_plotted else "")
                tgp_plotted = True

        roi3_plotted = False
        for i in range(actual_N):
            if cdata["val_roi3"][i]:
                ax_sp.plot([vz_pts[i] - 0.5*dx, vz_pts[i] + 0.5*dx], [y_bar, y_bar], color="#10b981", linewidth=4.5,
                           solid_capstyle="butt", zorder=5, label="3-Step Protocol (ROI 3)" if not roi3_plotted else "")
                roi3_plotted = True

        ax_sp.set_title(f"{wlabel}: BdG Energy Spectrum", fontsize=11.5, fontweight="bold", color="#f1f5f9", pad=6)
        ax_sp.set_ylabel("Energy (meV)", color="#cbd5e1", fontsize=9.5)
        ax_sp.legend(loc="upper right", fontsize=8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

        # ----------------------------------------------------
        # Row 1: G_LL
        # ----------------------------------------------------
        ax_gll = axes[1, col]
        pcm1 = ax_gll.pcolormesh(vz_pts, bias_uV, cdata["g_ll"].T, shading="auto", cmap="viridis", vmin=0.0, vmax=g_local_max, rasterized=True)
        ax_gll.set_title(f"{wlabel}: Local Conductance $G_{{LL}}$", fontsize=11, color="#f1f5f9", pad=4)
        ax_gll.set_ylabel(r"$V_{\mathrm{bias}}\ [\mu\mathrm{V}]$", color="#cbd5e1", fontsize=9.5)

        # ----------------------------------------------------
        # Row 2: G_RR
        # ----------------------------------------------------
        ax_grr = axes[2, col]
        pcm2 = ax_grr.pcolormesh(vz_pts, bias_uV, cdata["g_rr"].T, shading="auto", cmap="viridis", vmin=0.0, vmax=g_local_max, rasterized=True)
        ax_grr.set_title(f"{wlabel}: Local Conductance $G_{{RR}}$", fontsize=11, color="#f1f5f9", pad=4)
        ax_grr.set_ylabel(r"$V_{\mathrm{bias}}\ [\mu\mathrm{V}]$", color="#cbd5e1", fontsize=9.5)

        # ----------------------------------------------------
        # Row 3: A(G_RL) with +/- Delta_transport
        # ----------------------------------------------------
        ax_grl = axes[3, col]
        pcm3 = ax_grl.pcolormesh(vz_pts, bias_uV, cdata["a_grl"].T, shading="auto", cmap="PuOr_r", vmin=-g_nonlocal_max, vmax=g_nonlocal_max, rasterized=True)
        gap_l_uV = cdata["gap_l"] * 1e3
        max_b = np.max(np.abs(bias_uV))
        valid_l = (gap_l_uV > 0) & (gap_l_uV <= max_b)
        ax_grl.plot(vz_pts[valid_l], gap_l_uV[valid_l], color="#0f172a", linewidth=2.4)
        ax_grl.plot(vz_pts[valid_l], gap_l_uV[valid_l], color="#f8fafc", linewidth=1.4, linestyle="--", label=r"$\pm \Delta_{\mathrm{transport}}$")
        ax_grl.plot(vz_pts[valid_l], -gap_l_uV[valid_l], color="#0f172a", linewidth=2.4)
        ax_grl.plot(vz_pts[valid_l], -gap_l_uV[valid_l], color="#f8fafc", linewidth=1.4, linestyle="--")
        ax_grl.set_title(f"{wlabel}: Nonlocal Antisymmetric $A(G_{{RL}})$", fontsize=11, color="#f1f5f9", pad=4)
        ax_grl.set_ylabel(r"$V_{\mathrm{bias}}\ [\mu\mathrm{V}]$", color="#cbd5e1", fontsize=9.5)
        ax_grl.legend(loc="upper right", fontsize=7.5, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

        # ----------------------------------------------------
        # Row 4: A(G_LR) with +/- Delta_transport
        # ----------------------------------------------------
        ax_glr = axes[4, col]
        pcm4 = ax_glr.pcolormesh(vz_pts, bias_uV, cdata["a_glr"].T, shading="auto", cmap="PuOr_r", vmin=-g_nonlocal_max, vmax=g_nonlocal_max, rasterized=True)
        gap_r_uV = cdata["gap_r"] * 1e3
        valid_r = (gap_r_uV > 0) & (gap_r_uV <= max_b)
        ax_glr.plot(vz_pts[valid_r], gap_r_uV[valid_r], color="#0f172a", linewidth=2.4)
        ax_glr.plot(vz_pts[valid_r], gap_r_uV[valid_r], color="#f8fafc", linewidth=1.4, linestyle="--", label=r"$\pm \Delta_{\mathrm{transport}}$")
        ax_glr.plot(vz_pts[valid_r], -gap_r_uV[valid_r], color="#0f172a", linewidth=2.4)
        ax_glr.plot(vz_pts[valid_r], -gap_r_uV[valid_r], color="#f8fafc", linewidth=1.4, linestyle="--")
        ax_glr.set_title(f"{wlabel}: Nonlocal Antisymmetric $A(G_{{LR}})$", fontsize=11, color="#f1f5f9", pad=4)
        ax_glr.set_ylabel(r"$V_{\mathrm{bias}}\ [\mu\mathrm{V}]$", color="#cbd5e1", fontsize=9.5)
        ax_glr.set_xlabel(r"Zeeman Field $V_z\ [\mathrm{T}]$", color="#cbd5e1", fontsize=10.5)

    # Dedicated Colorbars on right side
    cax_local = fig.add_axes([0.93, 0.44, 0.015, 0.30])
    cb_loc = fig.colorbar(pcm1, cax=cax_local)
    cb_loc.set_label(r"Local Conductance ($e^2/h$)", color="#e2e8f0", fontsize=10)
    cb_loc.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cb_loc.ax.axes, 'yticklabels'), color="#cbd5e1", fontsize=8.5)
    cb_loc.outline.set_edgecolor("#334155")

    cax_nonlocal = fig.add_axes([0.93, 0.065, 0.015, 0.30])
    cb_non = fig.colorbar(pcm3, cax=cax_nonlocal)
    cb_non.set_label(r"Nonlocal $A(G)$ ($e^2/h$)", color="#e2e8f0", fontsize=10)
    cb_non.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cb_non.ax.axes, 'yticklabels'), color="#cbd5e1", fontsize=8.5)
    cb_non.outline.set_edgecolor("#334155")

    fig.suptitle(f"{title}\nCut along $\\mu = {mu_cut:.3f}\\ \\mathrm{{meV}}$", fontsize=13.5, fontweight="bold", color="#38bdf8", y=0.985)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    logger.info(f"Saved 10-panel conductance multiplot: {out_path.name}")


def render_analytics_multiplot(
    c3: Dict[str, Any],
    c6: Dict[str, Any],
    mu_cut: float,
    title: str,
    out_path: Path
):
    """
    Renders 8-panel comparative analytics multiplot (4 rows x 2 columns).
    Rows:
      0: BdG Spectrum + Protocol Acceptance Runners + Ground Truth Shading
      1: 3w Peak Curvature vs -100 threshold
      2: Extracted Transport Gap vs Lowest BdG States (E0, E1) and 10 ueV threshold
      3: Binarized Barrier Reflection Correlation Track (C_L, C_R >= 0.85)
    """
    fig, axes = plt.subplots(4, 2, figsize=(15.0, 11.5), sharex=True, facecolor="#0b1120")
    plt.subplots_adjust(left=0.07, right=0.94, bottom=0.07, top=0.92, wspace=0.18, hspace=0.15)

    wire_configs = [
        (0, c3, "3 μm Nanowire"),
        (1, c6, "6 μm Nanowire"),
    ]

    for col, cdata, wlabel in wire_configs:
        vz_pts = cdata["vz_pts"]
        actual_N = len(vz_pts)
        dx = (vz_pts[-1] - vz_pts[0]) / (actual_N - 1) if actual_N > 1 else 1.0

        for r in range(4):
            axes[r, col].set_facecolor("#070d1a")
            for spine in axes[r, col].spines.values():
                spine.set_color("#334155")
            axes[r, col].tick_params(colors="#94a3b8", labelsize=9)

        # ----------------------------------------------------
        # Row 0: BdG Spectrum + Protocol Runners
        # ----------------------------------------------------
        ax0 = axes[0, col]
        evals = cdata["evals"]
        kvals = evals.shape[1]
        mid = kvals // 2

        for i in range(actual_N):
            if cdata["val_pfaff"][i]:
                ax0.axvspan(vz_pts[i] - 0.5*dx, vz_pts[i] + 0.5*dx, color="#38bdf8", alpha=0.15, zorder=0)

        for j in range(kvals):
            is_mzm = j in [mid - 1, mid]
            col_l = "#ef4444" if is_mzm else "#3b82f6"
            lw = 2.2 if is_mzm else 1.1
            ax0.plot(vz_pts, evals[:, j], color=col_l, linewidth=lw, alpha=0.85, zorder=2)

        ax0.axhline(0, color="#64748b", linestyle="--", linewidth=0.8, alpha=0.6)
        ymin = np.min(evals) - 0.08 * (np.max(evals) - np.min(evals))
        ymax = np.max(evals) + 0.08 * (np.max(evals) - np.min(evals))
        ax0.set_ylim(ymin, ymax)
        y_bar = ymin + 0.04 * (ymax - ymin)

        tgp_plotted = False
        for i in range(actual_N):
            if cdata["val_tgp"][i]:
                ax0.plot(vz_pts[i], ymin, marker="|", color="#ef4444", markersize=9, markeredgewidth=2.0, zorder=6,
                         label="Microsoft TGP" if not tgp_plotted else "")
                tgp_plotted = True

        roi3_plotted = False
        for i in range(actual_N):
            if cdata["val_roi3"][i]:
                ax0.plot([vz_pts[i] - 0.5*dx, vz_pts[i] + 0.5*dx], [y_bar, y_bar], color="#10b981", linewidth=4.5,
                         solid_capstyle="butt", zorder=5, label="3-Step Protocol" if not roi3_plotted else "")
                roi3_plotted = True

        ax0.set_title(f"{wlabel}: BdG Spectrum & Protocol Acceptance", fontsize=11, fontweight="bold", color="#f1f5f9", pad=4)
        ax0.set_ylabel("Energy (meV)", color="#cbd5e1", fontsize=9.5)
        ax0.legend(loc="upper right", fontsize=8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")

        # ----------------------------------------------------
        # Row 1: 3w Peak Curvature
        # ----------------------------------------------------
        ax1 = axes[1, col]
        y_3w = np.clip(cdata["val_3w"], -110.0, 110.0)
        ax1.plot(vz_pts, y_3w, color="#f59e0b", linewidth=1.8, label=r"Average $3\omega$ Curvature")
        ax1.axhline(-100.0, color="#ef4444", linestyle="--", linewidth=1.2, label=r"ZBP Threshold ($-100$)")
        ax1.set_ylabel(r"$3\omega\ [d^2G/dV^2]$", color="#cbd5e1", fontsize=9.5)
        ax1.set_title(f"{wlabel}: $3\\omega$ Zero-Bias Peak Curvature Consensus", fontsize=11, color="#f1f5f9", pad=4)
        ax1.legend(loc="upper right", fontsize=8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
        ax1.set_ylim(-110.0, 110.0)

        # ----------------------------------------------------
        # Row 2: Transport Gap vs. Lowest BdG States (E0, E1)
        # ----------------------------------------------------
        ax2 = axes[2, col]
        gap_min = np.minimum(cdata["gap_l"], cdata["gap_r"])
        e0 = evals[:, mid]
        e1 = evals[:, mid + 1]
        ax2.plot(vz_pts, gap_min, color="#a855f7", linewidth=2.0, label=r"$\Delta_{\mathrm{transport}}$ (Extracted)")
        ax2.plot(vz_pts, e0, color="#ef4444", linestyle=":", linewidth=1.8, label=r"BdG Ground State $E_0$")
        ax2.plot(vz_pts, e1, color="#38bdf8", linestyle="--", linewidth=1.4, label=r"BdG First Excited $E_1$")
        ax2.axhline(0.010, color="#facc15", linestyle="-.", linewidth=1.2, label=r"Gap Threshold ($10\ \mu\mathrm{eV}$)")
        ax2.set_ylabel("Gap / Energy (meV)", color="#cbd5e1", fontsize=9.5)
        ax2.set_title(f"{wlabel}: Transport Gap vs. True BdG Energy Levels", fontsize=11, color="#f1f5f9", pad=4)
        ax2.legend(loc="upper right", fontsize=8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
        ax2.set_ylim(-0.002, 0.065)

        # ----------------------------------------------------
        # Row 3: Binarized Correlation Track
        # ----------------------------------------------------
        ax3 = axes[3, col]
        corr_bin = np.where(cdata["val_corr"], 1.0, -1.0)
        ax3.step(vz_pts, corr_bin, where="mid", color="#38bdf8", linewidth=1.8)
        for i in range(actual_N):
            ccol = "#10b981" if cdata["val_corr"][i] else "#475569"
            calpha = 0.45 if cdata["val_corr"][i] else 0.20
            ax3.axvspan(vz_pts[i] - 0.5*dx, vz_pts[i] + 0.5*dx,
                        ymin=0.5 if cdata["val_corr"][i] else 0.0,
                        ymax=1.0 if cdata["val_corr"][i] else 0.5,
                        color=ccol, alpha=calpha, zorder=1)
        ax3.axhline(0, color="#64748b", linewidth=0.8, alpha=0.5)
        ax3.set_ylim(-1.4, 1.4)
        ax3.set_yticks([-1.0, 0.0, 1.0])
        ax3.set_yticklabels(["Fail", "0", "Pass"], color="#cbd5e1", fontsize=8.5)
        ax3.set_ylabel("Barrier Corr", color="#cbd5e1", fontsize=9.5)
        ax3.set_xlabel(r"Zeeman Field $V_z\ [\mathrm{T}]$", color="#cbd5e1", fontsize=10.5)
        ax3.set_title(f"{wlabel}: Dual Reflection Correlation Track ($C_L, C_R \\geq 0.85$)", fontsize=11, color="#f1f5f9", pad=4)

    fig.suptitle(f"{title}\nCut Analytics along $\\mu = {mu_cut:.3f}\\ \\mathrm{{meV}}$", fontsize=13.5, fontweight="bold", color="#38bdf8", y=0.985)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    logger.info(f"Saved analytics multiplot: {out_path.name}")


def render_gapless_boundary_diagnostic(
    sim3: Dict[str, Any],
    sim6: Dict[str, Any],
    desc: str,
    title: str,
    out_path: Path
):
    """
    Renders 2x2 spatial diagnostic map illustrating the collapse of the gapless boundary.
      Row 1: 3 μm Wire: Gapless Points Map (Left) vs. Microsoft TGP Passed ROI 2 (Right)
      Row 2: 6 μm Wire: Gapless Points Map (Left) vs. Microsoft TGP Passed ROI 2 (Right)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.2), facecolor="#0b1120")
    extent = [
        float(sim3["Vz_var"][0]), float(sim3["Vz_var"][-1]),
        float(sim3["mu_var"][0]), float(sim3["mu_var"][-1])
    ]

    gapless_3 = (sim3["gap_min_2d"] <= 0.010).astype(int)
    gapless_6 = (sim6["gap_min_2d"] <= 0.010).astype(int)
    tgp_3 = sim3["m_tgp"].astype(int)
    tgp_6 = sim6["m_tgp"].astype(int)

    gt3_b = sim3["gt_2d"].astype(bool)
    gt6_b = sim6["gt_2d"].astype(bool)

    panels = [
        (axes[0, 0], gapless_3, gt3_b, r"3 μm Wire: Gapless Map ($\Delta_{\mathrm{transport}} \leq 10\ \mu\mathrm{eV}$)", "#10b981"),
        (axes[0, 1], tgp_3, gt3_b, "3 μm Wire: Microsoft TGP Stage 2 Passed Region", "#10b981"),
        (axes[1, 0], gapless_6, gt6_b, r"6 μm Wire: Gapless Map ($\Delta_{\mathrm{transport}} \leq 10\ \mu\mathrm{eV}$)", "#ef4444"),
        (axes[1, 1], tgp_6, gt6_b, "6 μm Wire: Microsoft TGP Stage 2 Passed Region (TPR: 0.0%)", "#ef4444"),
    ]

    for ax, data, gt_mask, p_title, active_color in panels:
        ax.set_facecolor("#070d1a")
        cmap_bin = mcolors.ListedColormap(["#070d1a", active_color])
        ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap=cmap_bin, vmin=0, vmax=1, interpolation="nearest")

        if np.sum(gt_mask) > 0:
            ax.contour(gt_mask, levels=[0.5], extent=extent, colors=["#38bdf8"], linestyles=["--"], linewidths=[1.8], origin="lower")

        ax.set_title(p_title, fontsize=11.5, fontweight="bold", color="#f1f5f9", pad=8)
        ax.set_xlabel(r"$V_z\ [\mathrm{T}]$", fontsize=10.5, color="#cbd5e1")
        ax.set_ylabel(r"$\mu\ [\mathrm{meV}]$", fontsize=10.5, color="#cbd5e1")
        ax.tick_params(colors="#94a3b8", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    legend_elements = [
        plt.Line2D([0], [0], color="#38bdf8", linestyle="--", linewidth=1.8, label="Pfaffian Boundary (Q = -1)"),
        mpatches.Patch(facecolor="#10b981", edgecolor="none", label="3 μm Passing Region"),
        mpatches.Patch(facecolor="#ef4444", edgecolor="none", label="6 μm Collapsed / Suppressed"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.50, 0.925),
        fontsize=10,
        facecolor="#0f172a",
        edgecolor="#334155",
        labelcolor="#e2e8f0"
    )

    fig.suptitle(f"{title}\nDiagnostic: Collapse of the Nonlocal Gapless Boundary at 6 μm", fontsize=13.5, fontweight="bold", color="#38bdf8", y=0.985)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.87, wspace=0.18, hspace=0.25)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    logger.info(f"Saved gapless boundary diagnostic map: {out_path.name}")


def main():
    logger.info("=== STARTING COMPARATIVE WIRE LENGTH CUT EVALUATION ===")

    for cdef in TARGET_CUTS:
        dtag = cdef["disorder_tag"]
        mu_cut = cdef["mu_cut"]
        vz_min = cdef["vz_min"]
        vz_max = cdef["vz_max"]
        n_pts = cdef["n_pts"]
        ftag = cdef["filename_tag"]
        title = cdef["title"]

        d3_path = BASE_DATA / "3micron" / f"{dtag}_realization_0"
        d6_path = BASE_DATA / "6micron" / f"{dtag}_realization_0"

        logger.info(f"Processing cut for {dtag} at mu = {mu_cut:.3f} meV...")
        sim3 = load_wire_simulation(d3_path, "3micron")
        sim6 = load_wire_simulation(d6_path, "6micron")

        vz_pts = np.linspace(vz_min, vz_max, n_pts)
        c3 = extract_cut_data(sim3, mu_cut, vz_pts)
        c6 = extract_cut_data(sim6, mu_cut, vz_pts)

        # 1. 10-Panel Conductance Multiplot
        p_10p = OUTPUT_DIR / f"{ftag}_10panel_conductance.png"
        render_10panel_comparison(c3, c6, mu_cut, title, p_10p)

        # 2. Analytics Multiplot
        p_ana = OUTPUT_DIR / f"{ftag}_analytics_multiplot.png"
        render_analytics_multiplot(c3, c6, mu_cut, title, p_ana)

        # 3. 2D Gapless Boundary Diagnostic Map
        p_diag = OUTPUT_DIR / f"{ftag}_diagnostic_gapless_boundary.png"
        render_gapless_boundary_diagnostic(sim3, sim6, dtag, title, p_diag)

    logger.info("=== COMPARATIVE WIRE LENGTH CUT EVALUATION COMPLETE ===")


if __name__ == "__main__":
    main()
