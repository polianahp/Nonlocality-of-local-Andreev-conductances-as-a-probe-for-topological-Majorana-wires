#!/usr/bin/env python3
"""
Batch script to:
1. Generate Stage 2 bias voltage gap threshold sensitivity sweep diagrams
   across all 7 disorder realizations for threshold_low in [2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0] ueV.
2. Compute and extract ROI 3 (Stage 2 ROI 2 intersected with dual barrier sweep correlation >= 0.85).
3. Find the top 3 islands of ROI 3, generate ROI3_stage2_phase_map.png, and execute horizontal TGP cuts
   through their centers of mass, saving all plots with 'ROI3_' prefix in ROI3_Cuts/.
4. Save ROI3_fdr_metrics.json comparing Ground Truth, ROI 2, and ROI 3.
"""

import sys
import os
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

import tgp
import src.helpers as hp
from cut_analysis import (
    plot_stage2_diagram_clean,
    export_cut_conductance_plots,
    _eval_cut_point_worker,
    _process_deep_dive_point_worker,
    PathConfigs
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BiasSweepAndROI3")

DATASETS = [
    ("Tdis_pfaff5_V0_0_0", 0.001),
    ("Tdis_pfaff5_V0_0_1", 0.01),
    ("Tdis_pfaff5_V0_0_378", 0.01),
    ("Tdis_pfaff5_V0_0_645", 0.05),
    ("Tdis_pfaff5_V0_0_872", 0.05),
    ("Tdis_pfaff5_V0_0_91", 0.05),
    ("Tdis_pfaff5", 0.05),
]

BIAS_GAP_THRESHOLDS_UEV = [2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0]

def process_dataset(dataset_name: str, gap_th_factor: float):
    logger.info(f"==================================================")
    logger.info(f"Processing dataset: {dataset_name} (factor={gap_th_factor})")
    logger.info(f"==================================================")

    data_dir = PathConfigs.DATA / dataset_name
    plot_dir = PathConfigs.PLOTS / f"{dataset_name}_Plots"
    sweeps_dir = plot_dir / "Stage2_Gap_Sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    bias_sweep_dir = plot_dir / "bias_threshold_sweep"
    bias_sweep_dir.mkdir(parents=True, exist_ok=True)
    roi3_dir = plot_dir / "ROI3_Cuts"
    roi3_dir.mkdir(parents=True, exist_ok=True)

    tprep = xr.load_dataset(data_dir / "tprep.nc")
    B_vals = tprep.coords["B"].values
    V_vals = tprep.coords["V"].values

    if 'L_SI' in tprep:
        pfaff_da = (tprep['L_SI'].mean(dim='cutter_pair_index') if 'cutter_pair_index' in tprep['L_SI'].dims else tprep['L_SI']).astype(int)
    else:
        pfaff_da = None

    pfaff_vb = (tprep["L_SI"].values[0] == 1).T if 'L_SI' in tprep else None  # (V, B)

    cL_file = data_dir / "barrier_left_correlation_arr.npy"
    cR_file = data_dir / "barrier_right_correlation_arr.npy"
    if not cL_file.exists() or not cR_file.exists():
        logger.error(f"Correlation arrays missing for {dataset_name}!")
        return

    cL = np.load(cL_file)
    cR = np.load(cR_file)
    corr_both_1d = (cL >= 0.85) & (cR >= 0.85)
    corr_both_2d = corr_both_1d.reshape((len(V_vals), len(B_vals)))

    t_l = tprep.rename({"bias": "left_bias"})
    t_r = tprep.rename({"bias": "right_bias"})
    t_l, t_r = tgp.two.extract_gap(t_l, t_r, gap_threshold_factor=gap_th_factor, noise_threshold=1e-4)
    z_ds_base = tgp.two.zbp_dataset_derivative(t_l, t_r, zbp_probability_threshold=0.7, average_over_cutter=False)

    # --------------------------------------------------------------------------
    # 1. Bias Voltage Gap Threshold Sensitivity Sweep Diagrams & Transport Gap Maps
    # --------------------------------------------------------------------------
    logger.info(f"[{dataset_name}] Generating bias voltage gap threshold sweep diagrams & transport gap maps...")
    gap_left_mean = t_l.gap.mean(dim='cutter_pair_index').values if 'cutter_pair_index' in t_l.gap.dims else t_l.gap.values
    gap_right_mean = t_r.gap.mean(dim='cutter_pair_index').values if 'cutter_pair_index' in t_r.gap.dims else t_r.gap.values
    gap_2d = np.minimum(gap_left_mean, gap_right_mean)  # (V, B)

    for th_u in BIAS_GAP_THRESHOLDS_UEV:
        th_mev = th_u * 1e-3
        out_png_legacy = sweeps_dir / f"stage2_bias_gap_{th_u:.1f}uev.png"
        out_s2 = bias_sweep_dir / f"stage2_phase_map_{th_u:.1f}uev.png"
        out_roi3 = bias_sweep_dir / f"ROI3_stage2_phase_map_{th_u:.1f}uev.png"
        out_tg = bias_sweep_dir / f"transport_gap_phase_map_{th_u:.1f}uev.png"
        try:
            z_ds = z_ds_base.copy(deep=True)
            tgp.two.set_gap_threshold(z_ds, threshold_low=th_mev, threshold_high=0.067)
            s2 = tgp.two.cluster_and_score(
                z_ds,
                min_cluster_size=7,
                cluster_gap_threshold=th_mev,
                cluster_percentage_boundary_threshold=0.6,
                cluster_ncutter_threshold=0.5
            )
            if pfaff_da is not None:
                s2['pfaffian'] = pfaff_da

            # 1a. Stage 2 standard phase map (no correlation mask)
            fig_s2, _ = plot_stage2_diagram_clean(s2, draw_cut_lines=False, title_suffix=f"Bias Gap Th={th_u:.1f} ueV", correlation_mask=None)
            fig_s2.savefig(out_s2, dpi=150, bbox_inches='tight')
            fig_s2.savefig(out_png_legacy, dpi=150, bbox_inches='tight')
            plt.close(fig_s2)

            # 1b. ROI 3 Stage 2 phase map (grey shading for failed correlation)
            fig_r3, _ = plot_stage2_diagram_clean(s2, draw_cut_lines=False, title_suffix=f"Bias Gap Th={th_u:.1f} ueV [ROI 3: Grey = Failed Corr]", correlation_mask=corr_both_2d)
            fig_r3.savefig(out_roi3, dpi=150, bbox_inches='tight')
            plt.close(fig_r3)

            # 1c. Transport Gap Phase Map
            fig_tg, ax_tg = plt.subplots(figsize=(6.5, 5.2))
            im_tg = ax_tg.pcolormesh(B_vals, V_vals, gap_2d, cmap='hot_r', vmin=0.0, vmax=0.05, shading='nearest', rasterized=True)
            if pfaff_vb is not None:
                try:
                    ax_tg.contour(B_vals, V_vals, pfaff_vb.astype(float), levels=[0.5], colors=['cyan'], linewidths=[1.2])
                except Exception:
                    pass
            cb_tg = fig_tg.colorbar(im_tg, ax=ax_tg)
            cb_tg.set_label("Transport Gap (meV)", fontsize=10)
            ax_tg.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10)
            ax_tg.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)
            ax_tg.set_title(r"Transport Gap Phase Map $\Delta_{\mathrm{transport}}$" + f"\n(Cutoff $\Delta_{{\mathrm{{low}}}} = {th_u:.1f}\ \mu\mathrm{{eV}}$, Cyan: Pfaffian $\mathcal{{Q}}=-1$)", fontsize=11)
            fig_tg.tight_layout()
            fig_tg.savefig(out_tg, dpi=150)
            plt.close(fig_tg)

            logger.info(f"Saved {out_s2.name}, {out_roi3.name}, and {out_tg.name}")
        except Exception as e:
            logger.warning(f"Failed bias sweep for {th_u} ueV: {e}")

    # 1d. Global Bias Gap Sensitivity Phase Map (Difference between 2.0 ueV and 30.0 ueV)
    try:
        gap_th_2 = np.where(gap_2d >= 2.0e-3, gap_2d, 0.0)
        gap_th_30 = np.where(gap_2d >= 30.0e-3, gap_2d, 0.0)
        bias_gap_sens_2d = gap_th_2 - gap_th_30
        fig_bs, ax_bs = plt.subplots(figsize=(6.5, 5.2))
        sens_max = max(0.01, float(np.nanmax(bias_gap_sens_2d)))
        im_bs = ax_bs.pcolormesh(B_vals, V_vals, bias_gap_sens_2d, cmap='plasma', vmin=0.0, vmax=sens_max, shading='nearest', rasterized=True)
        if pfaff_vb is not None:
            try:
                ax_bs.contour(B_vals, V_vals, pfaff_vb.astype(float), levels=[0.5], colors=['cyan'], linewidths=[1.2])
            except Exception:
                pass
        cb_bs = fig_bs.colorbar(im_bs, ax=ax_bs)
        cb_bs.set_label("Bias Gap Sensitivity (meV)", fontsize=10)
        ax_bs.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10)
        ax_bs.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)
        ax_bs.set_title(r"Global Bias Gap Threshold Sensitivity ($\Delta_{2.0\mu\mathrm{eV}} - \Delta_{30.0\mu\mathrm{eV}}$)", fontsize=11)
        fig_bs.tight_layout()
        fig_bs.savefig(plot_dir / "global_bias_gap_sensitivity_phase_map.png", dpi=150)
        plt.close(fig_bs)
        logger.info(f"Saved global_bias_gap_sensitivity_phase_map.png to {plot_dir}")
    except Exception as e_bs:
        logger.warning(f"Could not generate global_bias_gap_sensitivity_phase_map: {e_bs}")

    # --------------------------------------------------------------------------
    # 2. Base Stage 2 Setup & ROI 3 Intersection
    # --------------------------------------------------------------------------
    z_ds_default = z_ds_base.copy(deep=True)
    tgp.two.set_gap_threshold(z_ds_default, threshold_low=10e-3, threshold_high=0.067)
    s2_default = tgp.two.cluster_and_score(
        z_ds_default,
        min_cluster_size=7,
        cluster_gap_threshold=10e-3,
        cluster_percentage_boundary_threshold=0.6,
        cluster_ncutter_threshold=0.5
    )
    if pfaff_da is not None:
        s2_default['pfaffian'] = pfaff_da

    gzbp_vb = z_ds_default.gapped_zbp.values[0]  # (V, B)
    roi2_vb = s2_default.roi2.values.T if s2_default.roi2.dims == ("B", "V") else s2_default.roi2.values
    roi2_passed = (roi2_vb > 0)

    # --------------------------------------------------------------------------
    # 3. Define ROI 3: ROI2 & (corr_L >= 0.85 & corr_R >= 0.85)
    # --------------------------------------------------------------------------
    roi3_passed = roi2_passed & corr_both_2d
    used_fallback = False
    if np.sum(roi3_passed) == 0:
        logger.info(f"[{dataset_name}] ROI2 passed has 0 pixels intersecting corr >= 0.85. Falling back to candidate orange & corr >= 0.85.")
        roi3_passed = gzbp_vb & corr_both_2d
        used_fallback = True

    # Compute Confusion Matrices: Ground Truth vs ROI2 vs ROI3
    def get_metrics(pred_mask, gt_mask):
        tp = int(np.sum(pred_mask & gt_mask))
        fp = int(np.sum(pred_mask & (~gt_mask)))
        fn = int(np.sum((~pred_mask) & gt_mask))
        tn = int(np.sum((~pred_mask) & (~gt_mask)))
        tot_pass = tp + fp
        tot_gt = tp + fn
        ppv = float(tp / tot_pass) if tot_pass > 0 else 0.0
        fdr = float(fp / tot_pass) if tot_pass > 0 else 0.0
        tpr = float(tp / tot_gt) if tot_gt > 0 else 0.0
        tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        return {
            "total_passed": tot_pass,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "p_topological_given_passed_ppv": ppv,
            "false_discovery_rate_fdr": fdr,
            "sensitivity_tpr": tpr,
            "specificity_tnr": tnr,
        }

    roi2_m = get_metrics(roi2_passed, pfaff_vb)
    roi3_m = get_metrics(roi3_passed, pfaff_vb)

    logger.info(f"[{dataset_name}] ROI2 Passed: {roi2_m['total_passed']} px (PPV: {roi2_m['p_topological_given_passed_ppv']*100:.1f}%, FDR: {roi2_m['false_discovery_rate_fdr']*100:.1f}%)")
    logger.info(f"[{dataset_name}] ROI3 Passed: {roi3_m['total_passed']} px (PPV: {roi3_m['p_topological_given_passed_ppv']*100:.1f}%, FDR: {roi3_m['false_discovery_rate_fdr']*100:.1f}%)")

    # --------------------------------------------------------------------------
    # 4. Extract Top 3 ROI 3 Islands
    # --------------------------------------------------------------------------
    lbl_roi3, n_roi3 = ndimage.label(roi3_passed)
    sizes = [int((lbl_roi3 == i).sum()) for i in range(1, n_roi3 + 1)]
    s_idx = np.argsort(sizes)[::-1]
    star_colors = ["cyan", "magenta", "gold"]
    star_names = ["Cyan Star", "Magenta Star", "Gold Star"]

    top3_islands = []
    for rank, idx in enumerate(s_idx[:3], 1):
        comp_id = idx + 1
        mask = (lbl_roi3 == comp_id)
        com_v, com_b = ndimage.center_of_mass(mask)
        mu_com = float(np.interp(com_v, np.arange(len(V_vals)), V_vals))
        vz_com = float(np.interp(com_b, np.arange(len(B_vals)), B_vals))
        top3_islands.append({
            "rank": rank,
            "size_pixels": sizes[idx],
            "vz_com": vz_com,
            "mu_com": mu_com,
            "color": star_colors[rank - 1],
            "star_name": star_names[rank - 1]
        })

    roi3_report = {
        "dataset_name": dataset_name,
        "total_pixels": int(pfaff_vb.size),
        "ground_truth_topological_pixels": int(np.sum(pfaff_vb)),
        "used_candidate_orange_fallback": used_fallback,
        "stage2_roi2_metrics": roi2_m,
        "stage3_roi3_metrics": roi3_m,
        "top3_islands": top3_islands
    }
    with open(roi3_dir / "ROI3_fdr_metrics.json", "w") as f_json:
        json.dump(roi3_report, f_json, indent=2)
    # Also save as fdr_metrics.json for compatibility
    with open(roi3_dir / "fdr_metrics.json", "w") as f_json:
        json.dump(roi3_report, f_json, indent=2)

    # --------------------------------------------------------------------------
    # 5. Export ROI3_stage2_phase_map.png
    # --------------------------------------------------------------------------
    try:
        stars_meta = [
            {"vz_com": isl["vz_com"], "mu_com": isl["mu_com"], "color": isl["color"], "label": f"ROI3 Island {isl['rank']} ({isl['star_name']})"}
            for isl in top3_islands
        ]
        # Create a modified dataset to highlight ROI 3 clusters
        roi3_ds = s2_default.copy(deep=True)
        # Update roi2 data with roi3_passed
        if 'roi2' in roi3_ds:
            if roi3_ds.roi2.dims == ("B", "V"):
                roi3_ds.roi2.values = roi3_passed.T.astype(int)
            else:
                roi3_ds.roi2.values = roi3_passed.astype(int)
        
        fig_stars, _ = plot_stage2_diagram_clean(
            roi3_ds,
            stars=stars_meta,
            draw_cut_lines=False,
            title_suffix="[ROI 3: Barrier Correlation >= 0.85 (Grey = Failed Corr)]",
            correlation_mask=corr_both_2d
        )
        fig_stars.savefig(roi3_dir / "ROI3_stage2_phase_map.png", dpi=200, bbox_inches='tight')
        # Also save without prefix for slide rendering compatibility
        fig_stars.savefig(roi3_dir / "stage2_phase_map.png", dpi=200, bbox_inches='tight')
        plt.close(fig_stars)
        logger.info(f"Saved ROI3_stage2_phase_map.png to {roi3_dir}")
    except Exception as e_stars:
        logger.warning(f"Could not generate ROI3_stage2_phase_map: {e_stars}")

    # --------------------------------------------------------------------------
    # 6. Execute TGP Cuts through the 3 ROI 3 Islands
    # --------------------------------------------------------------------------
    params = np.load(data_dir / "all_params.npz", allow_pickle=True)
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

    physics_params = {
        't_val': t_val,
        'mu_n': mu_n,
        'mu_leads': mu_leads,
        'gamma': gamma,
        'Delta0': Delta0,
        'alpha': alpha,
        'Ln': Ln,
        'Lb': Lb,
        'Ls': Ls,
        'barrier_l_base': barrier_l_base,
        'Vdisx': Vdisx,
    }

    L_2w_avg = tprep['L_2w_nl'].mean(dim='cutter_pair_index') if 'cutter_pair_index' in tprep['L_2w_nl'].dims else tprep['L_2w_nl']
    L_3w_avg = tprep['L_3w'].mean(dim='cutter_pair_index') if 'cutter_pair_index' in tprep['L_3w'].dims else tprep['L_3w']
    gap_left_avg = t_l.gap.mean(dim='cutter_pair_index')
    gap_right_avg = t_r.gap.mean(dim='cutter_pair_index')
    invariant_avg = tprep['L_SI'].mean(dim='cutter_pair_index')

    for isl in top3_islands:
        rank = isl["rank"]
        vz_com = isl["vz_com"]
        mu_com = isl["mu_com"]
        star_color = isl["color"]
        star_name = isl["star_name"]

        isl_dir = roi3_dir / f"ROI3_Island_{rank}_{star_color.capitalize()}_Star"
        isl_dir.mkdir(parents=True, exist_ok=True)

        half_w = 0.35
        vz_start = max(0.0, vz_com - half_w)
        vz_end = min(1.4, vz_com + half_w)
        if (vz_end - vz_start) < 0.6:
            if vz_start == 0.0: vz_end = min(1.4, vz_start + 0.7)
            elif vz_end == 1.4: vz_start = max(0.0, vz_end - 0.7)

        actual_N = 100
        vz_pts = np.linspace(vz_start, vz_end, actual_N)
        mu_pts = np.full(actual_N, mu_com)
        pts = np.column_stack([mu_pts, vz_pts])
        snap_idx = int(np.argmin(np.abs(vz_pts - vz_com)))

        rcut = {
            'config': {
                'start': (mu_com, vz_start),
                'end': (mu_com, vz_end),
                'color': star_color,
                'label': f"ROI3_Cut_Island_{rank}_{star_color.capitalize()}"
            },
            'resolved_snaps': [
                {
                    'snap_idx': snap_idx,
                    'snapped_coords': (mu_com, vz_pts[snap_idx]),
                    'color': star_color,
                    'label': f"ROI3_Island_{rank}_COM"
                }
            ]
        }

        kvals = 14
        worker_args = [
            (i, mu_pts[i], vz_pts[i], t_val, gamma, Delta0, alpha, Ls, Vdisx, kvals)
            for i in range(actual_N)
        ]
        with ProcessPoolExecutor(max_workers=12) as executor:
            results = list(executor.map(_eval_cut_point_worker, worker_args))
        results.sort(key=lambda x: x[0])

        evals = np.zeros((actual_N, kvals))
        val_overlap = np.zeros(actual_N)
        for i, s_evals, ov in results:
            evals[i, :] = s_evals
            val_overlap[i] = ov

        val_2w = np.zeros(actual_N)
        val_3w = np.zeros(actual_N)
        val_gap_left = np.zeros(actual_N)
        val_gap_right = np.zeros(actual_N)
        pfaffians = np.zeros(actual_N)
        val_corr_thresh = np.zeros(actual_N, dtype=bool)
        val_tgp_roi3 = np.zeros(actual_N, dtype=bool)

        for i in range(actual_N):
            mv, bv = mu_pts[i], vz_pts[i]
            val_2w[i] = L_2w_avg.sel(V=mv, B=bv, method='nearest').values.item()
            val_3w[i] = L_3w_avg.sel(V=mv, B=bv, method='nearest').values.item()
            pfaffians[i] = invariant_avg.sel(V=mv, B=bv, method='nearest').values.item()
            val_gap_left[i] = gap_left_avg.sel(V=mv, B=bv, method='nearest').values.item()
            val_gap_right[i] = gap_right_avg.sel(V=mv, B=bv, method='nearest').values.item()
            v_i = np.argmin(np.abs(V_vals - mv))
            b_i = np.argmin(np.abs(B_vals - bv))
            val_corr_thresh[i] = bool(corr_both_2d[v_i, b_i])
            val_tgp_roi3[i] = bool(roi3_passed[v_i, b_i])

        # 6a. Export Conductance Multiplot
        export_cut_conductance_plots(
            cut_dir=isl_dir,
            cut=rcut['config'],
            pts=pts,
            rcut=rcut,
            tprep=tprep,
            tprep_left=t_l,
            tprep_right=t_r,
            has_tgp_gap=True,
            selected_cutter=0,
            evals=evals,
            pfaffians=pfaffians,
            corr_thresh=val_corr_thresh,
            tgp_roi2=val_tgp_roi3
        )
        # Copy / duplicate to ROI3_ prefixed files
        if (isl_dir / "multi_panel_conductance.png").exists():
            import shutil
            shutil.copyfile(isl_dir / "multi_panel_conductance.png", isl_dir / "ROI3_multi_panel_conductance.png")

        # 6b. Cut Analytics Multiplot (ROI3_multi_panel.png)
        fig_an, axes = plt.subplots(4, 1, figsize=(7.5, 9.2), gridspec_kw={'height_ratios': [2, 1, 1, 0.7]}, sharex=True)
        dx_val = vz_pts[1] - vz_pts[0] if len(vz_pts) > 1 else 1.0
        for idx in range(actual_N):
            is_pfaff = pfaffians[idx] > 0
            is_corr = val_corr_thresh[idx]
            if is_pfaff and is_corr: color = '#5d768d'; alpha = 0.55
            elif is_pfaff and not is_corr: color = '#555555'; alpha = 0.45
            elif is_corr and not is_pfaff: color = '#99ccff'; alpha = 0.5
            else: continue
            for ax in axes:
                ax.axvspan(vz_pts[idx] - 0.5*dx_val, vz_pts[idx] + 0.5*dx_val, color=color, alpha=alpha, zorder=0)

        for ax in axes:
            ax.axvline(vz_pts[snap_idx], color=star_color, linestyle='--', linewidth=1.5, alpha=0.9, zorder=3)

        mid_idx = kvals // 2
        for j in range(kvals):
            color = 'red' if j in [mid_idx-1, mid_idx] else 'royalblue'
            lw = 2.5 if j in [mid_idx-1, mid_idx] else 1.2
            axes[0].plot(vz_pts, evals[:, j], color=color, linewidth=lw, alpha=0.85, zorder=2)
        axes[0].plot(vz_pts[snap_idx], 0.0, marker='*', color=star_color, markersize=14, markeredgecolor='black', markeredgewidth=1.2, zorder=10, label=f"ROI3 COM ({star_name})")
        axes[0].axhline(0, color='black', linestyle='--', alpha=0.6, linewidth=1.0)
        axes[0].set_ylabel("Energy (meV)")
        axes[0].set_title(f"ROI 3 Cut Analytics through Island {rank} [{star_name}]\n($\mu = {mu_com:.3f}$ meV, $V_z \in [{vz_start:.3f}, {vz_end:.3f}]$ meV)", fontsize=11)
        
        # Highlight full ROI 3 Island along bottom runner with red ticks on bottom axis
        ymin_an = np.min(evals) - 0.08 * (np.max(evals) - np.min(evals))
        ymax_an = np.max(evals) + 0.08 * (np.max(evals) - np.min(evals))
        axes[0].set_ylim(ymin_an, ymax_an)
        y_bar_an = ymin_an + 0.03 * (ymax_an - ymin_an)
        isl_an_plotted = False
        for idx in range(actual_N):
            if val_tgp_roi3[idx]:
                axes[0].plot([vz_pts[idx] - 0.5*dx_val, vz_pts[idx] + 0.5*dx_val], [y_bar_an, y_bar_an], color='#2ca02c', linewidth=4.5, solid_capstyle='butt', zorder=5, label="ROI 3 Island" if not isl_an_plotted else "")
                axes[0].plot(vz_pts[idx], ymin_an, marker='|', color='red', markersize=8, markeredgewidth=1.8, zorder=6)
                isl_an_plotted = True
        axes[0].legend(loc='upper right', fontsize=8)

        # Panel 2: 3w Measurement (Capped)
        cap_val = 140.0
        axes[1].plot(vz_pts, np.clip(val_3w, -cap_val, cap_val), color='darkorange', linewidth=1.5, zorder=2)
        axes[1].axhline(-100.0, color='black', linestyle=':', label='ZBP Threshold')
        axes[1].set_ylabel("3ω Curvature")
        axes[1].legend(loc='upper right', fontsize=8)

        # Panel 3: Transport Gap and Lowest States
        axes[2].plot(vz_pts, np.minimum(val_gap_left, val_gap_right), color='purple', linewidth=1.5, label=r"Min $\Delta_{ex}$")
        axes[2].plot(vz_pts, evals[:, mid_idx], color='red', linestyle='--', label=r"$E_0$")
        axes[2].plot(vz_pts, evals[:, mid_idx + 1], color='blue', linestyle='--', label=r"$E_1$")
        axes[2].axhline(0.010, color='black', linestyle='--', linewidth=1.2, label=r'Gap Th ($10\ \mu\mathrm{eV}$)', zorder=3)
        axes[2].set_ylabel("Gap / Energy (meV)")
        axes[2].legend(loc='upper right', fontsize=8)

        # Panel 4: Thresholded Correlation Condition (±1.5 binarized track)
        corr_bin = np.where(val_corr_thresh, 1.0, -1.0)
        axes[3].step(vz_pts, corr_bin, where='mid', color='#0284c7', linewidth=1.8, zorder=3)
        for idx in range(actual_N):
            c_col = '#10b981' if val_corr_thresh[idx] else '#94a3b8'
            c_alpha = 0.55 if val_corr_thresh[idx] else 0.35
            axes[3].axvspan(vz_pts[idx] - 0.5*dx_val, vz_pts[idx] + 0.5*dx_val,
                            ymin=0.5 if val_corr_thresh[idx] else 0.0,
                            ymax=1.0 if val_corr_thresh[idx] else 0.5,
                            color=c_col, alpha=c_alpha, zorder=2)
        axes[3].axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=4)
        axes[3].axhline(1.0, color='#10b981', linestyle='--', linewidth=0.8, alpha=0.7, zorder=4)
        axes[3].axhline(-1.0, color='#94a3b8', linestyle='--', linewidth=0.8, alpha=0.7, zorder=4)
        axes[3].set_ylim(-1.5, 1.5)
        axes[3].set_yticks([-1.0, 0.0, 1.0])
        axes[3].set_yticklabels(["Fail (-1)", "0", "Pass (+1)"], fontsize=8)
        axes[3].set_ylabel("Barrier Corr")
        axes[3].set_xlabel(r"Zeeman Field $V_z$ (meV)")
        axes[3].set_title(r"Thresholded Barrier Correlation ($C_L \geq 0.85$ & $C_R \geq 0.85$)", fontsize=9)

        fig_an.tight_layout()
        fig_an.savefig(isl_dir / "ROI3_multi_panel.png", dpi=200)
        fig_an.savefig(isl_dir / "multi_panel.png", dpi=200)
        plt.close(fig_an)

        # 6c. Standalone Spectra Plot
        fig_sp, ax_sp = plt.subplots(figsize=(7.5, 4.2))
        for j in range(kvals):
            color = 'red' if j in [mid_idx-1, mid_idx] else 'royalblue'
            lw = 2.5 if j in [mid_idx-1, mid_idx] else 1.2
            ax_sp.plot(vz_pts, evals[:, j], color=color, linewidth=lw, alpha=0.85)
        ax_sp.plot(vz_pts[snap_idx], 0.0, marker='*', color=star_color, markersize=14, markeredgecolor='black', markeredgewidth=1.2, zorder=10, label=f"ROI3 Center of Mass ({star_name})")
        ax_sp.axvline(vz_pts[snap_idx], color=star_color, linestyle='--', linewidth=1.5, alpha=0.8)
        ax_sp.axhline(0, color='black', linestyle='--', alpha=0.6, linewidth=1.0)
        ax_sp.set_xlabel(r"Zeeman Field $V_z$ [meV]", fontsize=10)
        ax_sp.set_ylabel("Energy [meV]", fontsize=10)
        ax_sp.set_title(f"BdG Energy Spectrum through ROI 3 Island {rank} [{star_name}]\n($\mu = {mu_com:.3f}$ meV, $V_z \in [{vz_start:.3f}, {vz_end:.3f}]$ meV)", fontsize=11)
        
        ymin_sp = np.min(evals) - 0.08 * (np.max(evals) - np.min(evals))
        ymax_sp = np.max(evals) + 0.08 * (np.max(evals) - np.min(evals))
        ax_sp.set_ylim(ymin_sp, ymax_sp)
        y_bar_sp = ymin_sp + 0.03 * (ymax_sp - ymin_sp)
        isl_sp_plotted = False
        for idx in range(actual_N):
            if val_tgp_roi3[idx]:
                ax_sp.plot([vz_pts[idx] - 0.5*dx_val, vz_pts[idx] + 0.5*dx_val], [y_bar_sp, y_bar_sp], color='#2ca02c', linewidth=4.5, solid_capstyle='butt', zorder=5, label="ROI 3 Island" if not isl_sp_plotted else "")
                ax_sp.plot(vz_pts[idx], ymin_sp, marker='|', color='red', markersize=8, markeredgewidth=1.8, zorder=6)
                isl_sp_plotted = True
        ax_sp.legend(loc='upper right', fontsize=9)
        fig_sp.tight_layout()
        fig_sp.savefig(isl_dir / "ROI3_spectra.png", dpi=200)
        fig_sp.savefig(isl_dir / "spectra.png", dpi=200)
        fig_sp.savefig(roi3_dir / f"ROI3_tgp_cut_island_{rank}_spectra.png", dpi=200)
        fig_sp.savefig(roi3_dir / f"tgp_cut_island_{rank}_spectra.png", dpi=200)
        plt.close(fig_sp)

        # 6d. Point Deep Dive at COM
        pt_dict = {
            'coords': (mu_com, vz_com),
            'color': star_color,
            'label': f"ROI3_Island_{rank}_COM",
            'dir_path': str(isl_dir)
        }
        pt_2w = L_2w_avg.sel(V=mu_com, B=vz_com, method='nearest').values.item()
        pt_3w = L_3w_avg.sel(V=mu_com, B=vz_com, method='nearest').values.item()
        _process_deep_dive_point_worker((pt_dict, physics_params, 30.0, pt_2w, pt_3w, 70))
        logger.info(f"[{dataset_name}] Completed ROI 3 Cut & Deep Dive for Island {rank} ({star_name})")

    logger.info(f"[{dataset_name}] Finished all processing successfully.")


def main():
    logger.info("Starting batch execution for Bias Voltage Gap Sweeps and ROI 3 Analysis...")
    for ds_name, fac in DATASETS:
        process_dataset(ds_name, fac)
    logger.info("Batch execution completed!")

if __name__ == "__main__":
    main()
