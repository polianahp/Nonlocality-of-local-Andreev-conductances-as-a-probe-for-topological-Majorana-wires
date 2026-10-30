#!/usr/bin/env python3
"""Sensitivity Variations Benchmark for Low-Asymmetry Correlation + 3w Curvature + Transport Gap Protocol.

User Directive (2026-10-28):
Evaluate 4 new variations alongside the baseline:
1. Baseline: LowCorr >= 0.75, Curv3w <= -100, Gap >= 10 ueV, C4 (Area >= 10, Core 2x2)
2. Variation Curv1000: Curv3w <= -1000 (with LowCorr >= 0.75, Gap >= 10 ueV)
3. Variation Gap20uev: Gap >= 20 ueV (with LowCorr >= 0.75, Curv3w <= -100)
4. Variation LowCorr080: LowCorr >= 0.80 (with Curv3w <= -100, Gap >= 10 ueV)
5. Variation LowCorr090: LowCorr >= 0.90 (with Curv3w <= -100, Gap >= 10 ueV)

Evaluated across all 40 local disorder realizations (10 per disorder strength V0 in {0.378, 0.645, 0.872, 1.2} meV)
at both T = 30 mK and T = 40 mK (total 80 realization-runs x 5 variations = 400 evaluations).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.Protocol_Elements import BaseProtocol, RealizationFeatureCache
from src.Protocol_Elements.metrics import evaluate_realization_metrics
from Protocols.Low_Asym_Corr_Curv3w_Transport_Gap_Protocol.protocol import Protocol

DATA_ROOT = REPO_ROOT / "Outputs" / "Data" / "Disorder_Realizations"
BASE_OUT_DIR = REPO_ROOT / "Outputs" / "Postprocessing" / "LowCorr_Curv3w_TransportGap_Protocol"
PARAM_DIR = REPO_ROOT / "Protocols" / "Low_Asym_Corr_Curv3w_Transport_Gap_Protocol" / "Parameters"


def load_instances() -> List[Dict[str, Any]]:
    configs = [
        {
            "id": "baseline_curv100_gap10_corr075",
            "file": "lowcorr075_curv100_gap10uev.yaml",
            "name": "Baseline (Curv <= -100, Gap >= 10 ueV, Corr >= 0.75)",
        },
        {
            "id": "var_curv1000",
            "file": "lowcorr075_curv1000_gap10uev.yaml",
            "name": "Curv <= -1000 (Gap >= 10 ueV, Corr >= 0.75)",
        },
        {
            "id": "var_gap20uev",
            "file": "lowcorr075_curv100_gap20uev.yaml",
            "name": "Gap >= 20 ueV (Curv <= -100, Corr >= 0.75)",
        },
        {
            "id": "var_corr080",
            "file": "lowcorr080_curv100_gap10uev.yaml",
            "name": "Corr >= 0.80 (Curv <= -100, Gap >= 10 ueV)",
        },
        {
            "id": "var_corr090",
            "file": "lowcorr090_curv100_gap10uev.yaml",
            "name": "Corr >= 0.90 (Curv <= -100, Gap >= 10 ueV)",
        },
    ]

    loaded = []
    for c in configs:
        p_path = PARAM_DIR / c["file"]
        with open(p_path) as f:
            raw = yaml.safe_load(f)
        loaded.append({
            "id": c["id"],
            "name": c["name"],
            "filename": c["file"],
            "instance_name": raw.get("instance_name", c["id"]),
            "description": raw.get("description", ""),
            "parameters": raw["parameters"],
        })
    return loaded


def main():
    protocol = Protocol()
    instances = load_instances()
    v0_order = ["V0_0_378", "V0_0_645", "V0_0_872", "V0_1_2"]
    temperatures = [30.0, 40.0]

    print("=" * 85)
    print(" SENSITIVITY BENCHMARK: Low-Asymmetry Correlation + 3w Curvature + Transport Gap Protocol")
    print(f" Loaded {len(instances)} Variations:")
    for inst in instances:
        print(f"   * [{inst['instance_name']}] {inst['name']}")
    print("=" * 85)

    all_instance_records: Dict[str, List[Dict[str, Any]]] = {
        inst["instance_name"]: [] for inst in instances
    }

    # Evaluate across temperatures and disorder regimes
    for temp in temperatures:
        for v0_lbl in v0_order:
            v0_dir = DATA_ROOT / v0_lbl
            if not v0_dir.exists():
                continue
            reals = sorted([d for d in v0_dir.iterdir() if d.is_dir() and "realization" in d.name])
            print(f"\n>>> Processing {v0_lbl} ({len(reals)} realizations) at T = {temp} mK <<<")

            for r_dir in reals:
                # Instantiate cache ONCE per realization & temperature to maximize cache reuse
                cache = RealizationFeatureCache(r_dir, temperature_mK=temp)

                for inst in instances:
                    inst_name = inst["instance_name"]
                    params = inst["parameters"]

                    steps = protocol.evaluate_steps(cache, params)
                    rec = evaluate_realization_metrics(cache, steps)
                    rec["instance_name"] = inst_name
                    rec["instance_label"] = inst["name"]
                    all_instance_records[inst_name].append(rec)

                # Log brief status for baseline and extremes
                b_rec = all_instance_records[instances[0]["instance_name"]][-1]
                c_rec = all_instance_records[instances[1]["instance_name"]][-1]
                g_rec = all_instance_records[instances[2]["instance_name"]][-1]
                print(
                    f"  [{r_dir.name}] "
                    f"Base: {b_rec['n_predicted_px']:3d}px (FDR {b_rec['FDR_pct']:4.1f}%, TPR {b_rec['TPR_pct']:4.1f}%) | "
                    f"Curv1000: {c_rec['n_predicted_px']:3d}px (FDR {c_rec['FDR_pct']:4.1f}%) | "
                    f"Gap20: {g_rec['n_predicted_px']:3d}px (FDR {g_rec['FDR_pct']:4.1f}%)"
                )

    # Compile outputs for each instance
    var_dir = BASE_OUT_DIR / "Variations"
    var_dir.mkdir(parents=True, exist_ok=True)

    master_summary_rows: List[Dict[str, Any]] = []

    for inst in instances:
        inst_name = inst["instance_name"]
        records = all_instance_records[inst_name]
        inst_out_dir = var_dir / inst_name
        inst_out_dir.mkdir(parents=True, exist_ok=True)

        flattened = []
        for r in records:
            row = {
                "instance_name": inst_name,
                "instance_label": r["instance_label"],
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
            flattened.append(row)

        df_inst = pd.DataFrame(flattened)
        df_inst.to_csv(inst_out_dir / "realization_level_metrics.csv", index=False)

        # Compute summary by (temp, disorder)
        inst_summaries = []
        for temp in temperatures:
            for v0_lbl in v0_order:
                sub = df_inst[(df_inst["temperature_mK"] == temp) & (df_inst["v0_label"] == v0_lbl)]
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

                summary_item = {
                    "instance_name": inst_name,
                    "instance_label": inst["name"],
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
                }
                inst_summaries.append(summary_item)
                master_summary_rows.append(summary_item)

        df_inst_sum = pd.DataFrame(inst_summaries)
        df_inst_sum.to_csv(inst_out_dir / "disorder_ensemble_summary.csv", index=False)

    # Master Comparative Summary
    df_master = pd.DataFrame(master_summary_rows)
    master_csv_path = BASE_OUT_DIR / "sensitivity_comparative_summary.csv"
    df_master.to_csv(master_csv_path, index=False)
    print(f"\nSaved master comparative summary to {master_csv_path}")

    # Also compute aggregate across all disorder regimes for each (instance, temperature)
    agg_rows = []
    for temp in temperatures:
        for inst in instances:
            inst_name = inst["instance_name"]
            sub = df_master[(df_master["temperature_mK"] == temp) & (df_master["instance_name"] == inst_name)]
            agg_rows.append({
                "instance_name": inst_name,
                "instance_label": inst["name"],
                "temperature_mK": temp,
                "avg_pass_rate_pct": round(sub["realization_pass_rate_pct"].mean(), 1),
                "avg_final_px": round(sub["mean_final_px"].mean(), 1),
                "avg_tpr_pct": round(sub["mean_tpr_pct"].mean(), 2),
                "avg_fdr_pct": round(sub["mean_fdr_pct"].mean(), 2),
                "avg_clusters": round(sub["mean_clusters"].mean(), 1),
                "avg_cluster_area": round(sub["mean_cluster_area"].mean(), 1),
                "avg_MZM_confinement": round(sub["mean_MZM_confinement"].mean(), 4),
                "avg_MZM_separability": round(sub["mean_MZM_separability"].mean(), 4),
            })

    df_agg = pd.DataFrame(agg_rows)
    agg_csv_path = BASE_OUT_DIR / "sensitivity_overall_aggregate.csv"
    df_agg.to_csv(agg_csv_path, index=False)
    print(f"Saved overall aggregate summary to {agg_csv_path}")

    # Display comparison tables
    print("\n" + "=" * 115)
    print(" OVERALL ENSEMBLE COMPARISON (Averaged across all 4 disorder strengths V0 in {0.378, 0.645, 0.872, 1.2} meV)")
    print("=" * 115)
    for temp in temperatures:
        print(f"\n--- Temperature T = {temp} mK ---")
        sub_agg = df_agg[df_agg["temperature_mK"] == temp]
        print(sub_agg[["instance_name", "avg_pass_rate_pct", "avg_final_px", "avg_tpr_pct", "avg_fdr_pct", "avg_clusters", "avg_cluster_area"]].to_string(index=False))
    print("=" * 115)

    # Save comprehensive JSON
    with open(BASE_OUT_DIR / "sensitivity_analysis_summary.json", "w") as f:
        json.dump({
            "instances": instances,
            "overall_aggregate": agg_rows,
            "disorder_summaries": master_summary_rows,
        }, f, indent=2)


if __name__ == "__main__":
    main()
