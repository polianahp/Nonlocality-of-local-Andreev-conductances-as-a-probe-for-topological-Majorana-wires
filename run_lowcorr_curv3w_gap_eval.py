#!/usr/bin/env python3
"""Evaluation Runner for Low-Asymmetry Correlation + 3w Curvature + Transport Gap Protocol.

User Specification (2026-10-28):
- Low asymmetry correlation: above 0.75 in [-5, +5] window
- 3w curvature: less than -100 on both sides (L and R) for 3 out of 5 cutters (leads)
- Transport gap: above 10 mu eV (0.010 meV)
- Runs across all local disorder realizations (10 per disorder strength V0 in {0.378, 0.645, 0.872, 1.2} meV)
"""

import os
import sys
from pathlib import Path
import json
from collections import OrderedDict
from typing import Dict, Any, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.Protocol_Elements import BaseProtocol, RealizationFeatureCache
from src.Protocol_Elements.metrics import evaluate_realization_metrics, aggregate_ensemble_metrics
from Protocols.Low_Asym_Corr_Curv3w_Transport_Gap_Protocol.protocol import Protocol

DATA_ROOT = REPO_ROOT / "Outputs" / "Data" / "Disorder_Realizations"
OUT_DIR = REPO_ROOT / "Outputs" / "Postprocessing" / "LowCorr_Curv3w_TransportGap_Protocol"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    protocol = Protocol()
    params = protocol.default_parameters

    v0_order = ["V0_0_378", "V0_0_645", "V0_0_872", "V0_1_2"]
    temperatures = [30.0, 40.0]

    all_records: List[Dict[str, Any]] = []

    print("=" * 78)
    print(" EVALUATION: Low-Asymmetry Correlation + 3w Curvature + Transport Gap Protocol")
    print(" Specs: LowCorr >= 0.75 (both), 3w <= -100 (both, 3 of 5), Gap >= 10 ueV, Area >= 10")
    print("=" * 78)

    for temp in temperatures:
        for v0_lbl in v0_order:
            v0_dir = DATA_ROOT / v0_lbl
            if not v0_dir.exists():
                continue
            reals = sorted([d for d in v0_dir.iterdir() if d.is_dir() and "realization" in d.name])
            print(f"\n--- Evaluating {v0_lbl} ({len(reals)} realizations) at T = {temp} mK ---")
            for r_dir in reals:
                cache = RealizationFeatureCache(r_dir, temperature_mK=temp)
                steps = protocol.evaluate_steps(cache, params)
                rec = evaluate_realization_metrics(cache, steps)
                all_records.append(rec)
                print(
                    f"  [{r_dir.name}] Final: {rec['n_predicted_px']:4d} px | "
                    f"TP: {rec['n_true_positive_px']:4d} | FP: {rec['n_false_positive_px']:4d} | "
                    f"TPR: {rec['TPR_pct']:5.2f}% | FDR: {rec['FDR_pct']:5.2f}% | "
                    f"Clusters: {rec['island_count']:2d} (avg area: {rec['mean_island_area_px']:.1f} px)"
                )

    # Flatten records for CSV
    flattened = []
    for r in all_records:
        row = {
            "v0_label": r["v0_label"],
            "v0_mev": r["v0_mev"],
            "realization": r["realization_name"],
            "temperature_mK": r["temperature_mK"],
            "gt_topo_px": r["n_ground_truth_topo_px"],
            "n_predicted_px": r["n_predicted_px"],
            "n_true_positive_px": r["n_true_positive_px"],
            "n_false_positive_px": r["n_false_positive_px"],
            "TPR_pct": r["TPR_pct"],
            "FDR_pct": r["FDR_pct"],
            "FDR_active_only_pct": r["FDR_active_only_pct"],
            "island_count": r["island_count"],
            "mean_island_area_px": r["mean_island_area_px"],
            "max_island_area_px": r["max_island_area_px"],
            "MZM_confinement_C": r["MZM_confinement_C"],
            "MZM_separability_S": r["MZM_separability_S"],
        }
        # Add waterfall metrics
        for sname, sdiag in r["step_diagnostics"].items():
            row[f"{sname}_survival_pct"] = sdiag["survival_pct"]
            row[f"{sname}_fdr_pct"] = sdiag["fdr_pct"]
            row[f"{sname}_tpr_pct"] = sdiag["tpr_pct"]
        flattened.append(row)

    df = pd.DataFrame(flattened)
    csv_path = OUT_DIR / "realization_level_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed realization metrics to {csv_path}")

    # Compute Ensemble Summaries by Disorder and Temperature
    summary_rows = []
    for temp in temperatures:
        for v0_lbl in v0_order:
            sub = df[(df["temperature_mK"] == temp) & (df["v0_label"] == v0_lbl)]
            if len(sub) == 0:
                continue
            v0_val = sub["v0_mev"].iloc[0]
            n_reals = len(sub)
            mean_passed = sub["n_predicted_px"].mean()
            mean_tp = sub["n_true_positive_px"].mean()
            mean_fp = sub["n_false_positive_px"].mean()
            mean_tpr = sub["TPR_pct"].mean()
            mean_fdr = sub["FDR_pct"].mean()
            pooled_fdr = (sub["n_false_positive_px"].sum() / max(1, sub["n_predicted_px"].sum())) * 100
            pass_rate = 100.0 * (sub["n_predicted_px"] > 0).sum() / n_reals
            mean_clusters = sub["island_count"].mean()
            mean_cluster_area = sub["mean_island_area_px"].mean()
            mean_bnd = sub["MZM_confinement_C"].mean()
            mean_sep = sub["MZM_separability_S"].mean()

            # Best and worst realizations by FDR
            with_pass = sub[sub["n_predicted_px"] > 0]
            if len(with_pass) > 0:
                best_row = with_pass.sort_values(by=["FDR_pct", "TPR_pct"], ascending=[True, False]).iloc[0]
                worst_row = with_pass.sort_values(by=["FDR_pct", "TPR_pct"], ascending=[False, True]).iloc[0]
                best_real = f"{best_row['realization']} (FDR {best_row['FDR_pct']}%, TPR {best_row['TPR_pct']}%)"
                worst_real = f"{worst_row['realization']} (FDR {worst_row['FDR_pct']}%, TPR {worst_row['TPR_pct']}%)"
            else:
                best_real = "N/A (0 px passed)"
                worst_real = "N/A (0 px passed)"

            summary_rows.append({
                "v0_label": v0_lbl,
                "v0_mev": v0_val,
                "temperature_mK": temp,
                "n_realizations": n_reals,
                "realization_pass_rate_pct": round(pass_rate, 1),
                "mean_final_px": round(mean_passed, 1),
                "mean_tp_px": round(mean_tp, 1),
                "mean_fp_px": round(mean_fp, 1),
                "mean_tpr_pct": round(mean_tpr, 2),
                "mean_fdr_pct": round(mean_fdr, 2),
                "pooled_fdr_pct": round(pooled_fdr, 2),
                "mean_clusters": round(mean_clusters, 1),
                "mean_cluster_area": round(mean_cluster_area, 1),
                "mean_MZM_confinement": round(mean_bnd, 4),
                "mean_MZM_separability": round(mean_sep, 4),
                "best_realization": best_real,
                "worst_realization": worst_real,
            })

    df_summary = pd.DataFrame(summary_rows)
    sum_csv_path = OUT_DIR / "disorder_ensemble_summary.csv"
    df_summary.to_csv(sum_csv_path, index=False)
    print(f"Saved ensemble summary to {sum_csv_path}\n")

    # Pretty print summary table
    print("=" * 115)
    print(" SUMMARY TABLE: Low-Asymmetry Correlation + 3w Curvature <= -100 + Transport Gap >= 10 ueV Protocol")
    print("=" * 115)
    print(df_summary[["v0_label", "temperature_mK", "realization_pass_rate_pct", "mean_final_px", "mean_tpr_pct", "mean_fdr_pct", "pooled_fdr_pct", "mean_clusters", "mean_cluster_area"]].to_string(index=False))
    print("=" * 115)

    with open(OUT_DIR / "evaluation_summary.json", "w") as f:
        json.dump({
            "protocol_name": protocol.name,
            "parameters": params,
            "disorder_summaries": summary_rows,
            "all_realizations": all_records,
        }, f, indent=2)


if __name__ == "__main__":
    main()
