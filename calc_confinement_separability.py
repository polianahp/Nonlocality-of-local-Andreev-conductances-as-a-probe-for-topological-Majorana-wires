#!/usr/bin/env python3
"""
calc_confinement_separability.py

Computes boundary confinement and MZM separability specifically on the True Positives (TP)
for:
  1. Ground Truth Topological Phase (Q = -1)
  2. Microsoft Azure Quantum TGP Baseline
  3. Unclustered Revised S-ACD (Pointwise)
  4. Clustered S-ACD (Directional Manifold Regularization)
  5. Clustered S-ACD (OPTICS Density Clustering)
at 30 mK and 40 mK on realistic disordered nanowire Tdis_pfaff5.
"""

from pathlib import Path
import numpy as np

from run_sacd_clustered_protocol import (
    load_nanowire_dataset,
    compute_stage1_sacd_mask,
    compute_stage2_curvature_consensus_mask,
    compute_stage3_transport_gap_mask,
    apply_manifold_clustering,
    apply_optics_clustering,
)
from generate_curvature_threshold_suite import execute_microsoft_tgp

REPO_ROOT = Path(__file__).resolve().parent

def evaluate_tp_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, conf_2d: np.ndarray, sep_2d: np.ndarray):
    pred_b = pred_mask.astype(bool)
    gt_b = gt_mask.astype(bool)
    
    tp_b = pred_b & gt_b
    fp_b = pred_b & (~gt_b)
    fn_b = (~pred_b) & gt_b
    
    tp = int(np.sum(tp_b))
    fp = int(np.sum(fp_b))
    fn = int(np.sum(fn_b))
    total_topo = int(np.sum(gt_b))
    total_pass = tp + fp
    
    tpr_pct = (tp / total_topo) * 100.0 if total_topo > 0 else 0.0
    fdr_pct = (fp / total_pass) * 100.0 if total_pass > 0 else 0.0
    
    # Boundary confinement & separability strictly on True Positives
    tp_conf = float(np.nanmean(conf_2d[tp_b])) if tp > 0 else 0.0
    tp_sep = float(np.nanmean(sep_2d[tp_b])) if tp > 0 else 0.0
    
    # Also record variance/std on True Positives
    tp_conf_std = float(np.nanstd(conf_2d[tp_b])) if tp > 0 else 0.0
    tp_sep_std = float(np.nanstd(sep_2d[tp_b])) if tp > 0 else 0.0
    
    # For comparison, also on all predicted positives (pass)
    pass_conf = float(np.nanmean(conf_2d[pred_b])) if total_pass > 0 else 0.0
    pass_sep = float(np.nanmean(sep_2d[pred_b])) if total_pass > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_pass": total_pass,
        "tpr_pct": tpr_pct,
        "fdr_pct": fdr_pct,
        "tp_conf": tp_conf,
        "tp_conf_std": tp_conf_std,
        "tp_sep": tp_sep,
        "tp_sep_std": tp_sep_std,
        "pass_conf": pass_conf,
        "pass_sep": pass_sep,
    }

def main():
    temperatures = [
        ("30mK", 30.0, REPO_ROOT / "Outputs" / "Data" / "Tdis_pfaff5_data" / "Tdispfaff5_30mK"),
        ("40mK", 40.0, REPO_ROOT / "Outputs" / "Data" / "Tdis_pfaff5_data" / "Tdispfaff5_40mK"),
    ]
    
    results = {}
    
    for temp_str, T_mK, data_dir in temperatures:
        print(f"\n=======================================================")
        print(f"PROCESSING TEMPERATURE: {temp_str} ({T_mK} mK)")
        print(f"=======================================================")
        
        d = load_nanowire_dataset(data_dir)
        gt_mask = d["gt_2d"]
        Nmu, Nvz = d["Nmu"], d["Nvz"]
        
        conf_2d = np.load(data_dir / "mzm_boundary_confinement_arr.npy").reshape(Nmu, Nvz)
        sep_2d = np.load(data_dir / "mzm_separability_arr.npy").reshape(Nmu, Nvz)
        
        # Ground Truth statistics
        gt_b = gt_mask.astype(bool)
        gt_conf = float(np.nanmean(conf_2d[gt_b]))
        gt_conf_std = float(np.nanstd(conf_2d[gt_b]))
        gt_sep = float(np.nanmean(sep_2d[gt_b]))
        gt_sep_std = float(np.nanstd(sep_2d[gt_b]))
        
        print(f"Ground Truth Pfaffian (Q = -1, Total {np.sum(gt_b)} px):")
        print(f"  Boundary Confinement: {gt_conf:.4f} +/- {gt_conf_std:.4f}")
        print(f"  MZM Separability:     {gt_sep:.4f} +/- {gt_sep_std:.4f}")
        
        # 1. Microsoft TGP Baseline
        _, _, m_tgp = execute_microsoft_tgp(data_dir, T_mK)
        
        # 2. S-ACD (Pointwise)
        m_s1, _, _ = compute_stage1_sacd_mask(d["g_LL_sym"], d["g_LL_pinch"], d["g_RR_sym"], d["g_RR_pinch"])
        m_s2 = compute_stage2_curvature_consensus_mask(d["dIdVl"], d["dIdVr"], Nmu, Nvz, d2_thresh=20.0, min_cutters=2)
        m_s3_gap = compute_stage3_transport_gap_mask(d["topological_gap_2d"], min_gap_uev=10.0)
        m_raw = m_s1 & m_s2 & m_s3_gap
        
        # 3. Clustered S-ACD: Manifold Regularization
        m_man, n_islands_man, max_island_man = apply_manifold_clustering(m_raw, kernel_mu=3, kernel_vz=5, min_island_size=15)
        
        # 4. Clustered S-ACD: OPTICS Clustering
        m_opt, n_islands_opt, max_island_opt = apply_optics_clustering(m_raw, min_samples=5, max_eps=3.0, min_cluster_size=15, xi=0.05)
        
        pipelines = [
            ("Microsoft TGP Baseline", m_tgp, None, None),
            ("Unclustered S-ACD (Pointwise)", m_raw, None, None),
            ("Clustered S-ACD (Directional Manifold)", m_man, n_islands_man, max_island_man),
            ("Clustered S-ACD (OPTICS Clustering)", m_opt, n_islands_opt, max_island_opt),
        ]
        
        res_temp = {"gt": {"conf": gt_conf, "conf_std": gt_conf_std, "sep": gt_sep, "sep_std": gt_sep_std}}
        
        for name, mask, n_isl, max_isl in pipelines:
            met = evaluate_tp_metrics(mask, gt_mask, conf_2d, sep_2d)
            if n_isl is not None:
                met["n_islands"] = n_isl
                met["max_island"] = max_isl
            res_temp[name] = met
            
            print(f"\n--- {name} ---")
            print(f"  TPR = {met['tpr_pct']:.2f}% | FDR = {met['fdr_pct']:.2f}% | TP = {met['tp']} px | Total Passed = {met['total_pass']} px")
            print(f"  TP Mean Confinement: {met['tp_conf']:.4f} +/- {met['tp_conf_std']:.4f}")
            print(f"  TP Mean Separability: {met['tp_sep']:.4f} +/- {met['tp_sep_std']:.4f}")
            print(f"  All-Predictions Mean Confinement: {met['pass_conf']:.4f}")
            print(f"  All-Predictions Mean Separability: {met['pass_sep']:.4f}")
            
        results[temp_str] = res_temp

if __name__ == "__main__":
    main()
