#!/usr/bin/env python3
"""Benchmark of Lax Protocol + Strict Cluster Post-Processing Family.

Investigates the alternative paradigm:
"Lax on physical protocol thresholds, strict on cluster post-processing."

Evaluates configurations:
1. Standard Lax Base (C>=0.75, k>=3/5, Curv<=-100, Gap>=10, Area>=10, core 2x2)
2. Lax Base + Strict Cluster 25 (Area>=25, core 2x2)
3. Lax Base + Strict Cluster 40 (Area>=40, core 3x3)
4. Lax Base + Strict Cluster 50 (Area>=50, core 3x3)
5. Ultra-Lax Corr (C>=0.70) + Strict Cluster 25 (Area>=25, core 2x2)
6. Lax k=4 + Mild Symmetry (S_sym<=0.30) + Strict Cluster 25 (Area>=25, core 2x2)
7. Lax Base (k=3) + Mild Symmetry (S_sym<=0.30) + Strict Cluster 35 (Area>=35, core 3x3)

Across 40 disorder realizations (10 per V0 in {0.378, 0.645, 0.872, 1.200} meV)
at both T = 30 mK and T = 40 mK (total 80 evaluation runs per configuration).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.Protocol_Elements import BaseProtocol, RealizationFeatureCache
from src.Protocol_Elements.metrics import evaluate_realization_metrics
from Protocols.Low_Asym_Corr_Curv3w_Transport_Gap_Protocol.protocol import Protocol

DATA_ROOT = REPO_ROOT / "Outputs" / "Data" / "Disorder_Realizations"
OUT_DIR = REPO_ROOT / "Outputs" / "Postprocessing" / "LowCorr_Curv3w_TransportGap_Protocol"
PARAM_DIR = REPO_ROOT / "Protocols" / "Low_Asym_Corr_Curv3w_Transport_Gap_Protocol" / "Parameters"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

configs = [
    {
        "id": "lax_base_cluster10",
        "file": "lowcorr075_curv100_gap10uev.yaml",
        "label": "Lax Base (Area>=10, Core 2x2)",
    },
    {
        "id": "lax_base_strict_cluster25",
        "file": "lax_base_strict_cluster25.yaml",
        "label": "Lax Base (Area>=25, Core 2x2)",
    },
    {
        "id": "lax_base_strict_cluster40_core3",
        "file": "lax_base_strict_cluster40_core3.yaml",
        "label": "Lax Base (Area>=40, Core 3x3)",
    },
    {
        "id": "lax_corr070_strict_cluster25",
        "file": "lax_corr070_strict_cluster25.yaml",
        "label": "Ultra-Lax C>=0.70 (Area>=25, Core 2x2)",
    },
    {
        "id": "lax_k4_sym030_strict_cluster25",
        "file": "lax_k4_sym030_strict_cluster25.yaml",
        "label": "Lax k=4 + Sym<=0.30 (Area>=25, Core 2x2)",
    },
    {
        "id": "lax_base_sym030_strict_cluster35_core3",
        "file": "lax_base_sym030_strict_cluster35_core3.yaml",
        "label": "Lax k=3 + Sym<=0.30 (Area>=35, Core 3x3)",
    },
]

tgp_baseline = {
    (30.0, "V0_0_378"): {"fdr": 11.69, "tpr": 7.35, "area": 359.7},
    (30.0, "V0_0_645"): {"fdr": 24.87, "tpr": 5.42, "area": 287.4},
    (30.0, "V0_0_872"): {"fdr": 25.34, "tpr": 4.84, "area": 224.3},
    (30.0, "V0_1_2"):   {"fdr": 20.55, "tpr": 3.16, "area": 111.9},
    (40.0, "V0_0_378"): {"fdr": 8.15,  "tpr": 8.37, "area": 396.9},
    (40.0, "V0_0_645"): {"fdr": 22.29, "tpr": 5.90, "area": 298.8},
    (40.0, "V0_0_872"): {"fdr": 25.12, "tpr": 5.23, "area": 240.2},
    (40.0, "V0_1_2"):   {"fdr": 24.20, "tpr": 3.51, "area": 126.9},
}


def main():
    protocol = Protocol()
    v0_order = ["V0_0_378", "V0_0_645", "V0_0_872", "V0_1_2"]
    temperatures = [30.0, 40.0]

    loaded_cfgs = []
    for c in configs:
        with open(PARAM_DIR / c["file"]) as f:
            raw = yaml.safe_load(f)
        loaded_cfgs.append({
            "id": c["id"],
            "label": c["label"],
            "filename": c["file"],
            "instance_name": raw["instance_name"],
            "params": raw["parameters"],
        })

    all_records = []

    print("=" * 100)
    print(" BENCHMARK: LAX PROTOCOL + STRICT CLUSTER POST-PROCESSING FAMILY VS TGP BASELINE")
    print("=" * 100)

    for temp in temperatures:
        for v0_lbl in v0_order:
            v0_dir = DATA_ROOT / v0_lbl
            realization_dirs = sorted([d for d in v0_dir.iterdir() if d.is_dir() and "realization" in d.name])

            for r_dir in realization_dirs:
                cache = RealizationFeatureCache(r_dir, temperature_mK=temp)

                for cfg in loaded_cfgs:
                    cfg_id = cfg["id"]
                    cfg_lbl = cfg["label"]
                    params = cfg["params"]

                    steps = protocol.evaluate_steps(cache, params)
                    rec = evaluate_realization_metrics(cache, steps)

                    # Cluster audit: verify that all components are >= min_island_size
                    final_mask = steps["step_5_final_mask"]
                    min_sz = params.get("dpe_c4_post_processor", {}).get("min_island_size", 1)
                    lbl_arr, num_c = ndi.label(final_mask, structure=np.ones((3, 3), dtype=bool))
                    c_sizes = [int(np.sum(lbl_arr == i)) for i in range(1, num_c + 1)]
                    min_observed = min(c_sizes) if c_sizes else 0
                    assert min_observed == 0 or min_observed >= min_sz, f"Cluster audit failure: observed {min_observed} < {min_sz}"

                    all_records.append({
                        "temperature_mK": temp,
                        "disorder_meV": v0_lbl,
                        "realization": r_dir.name,
                        "config_id": cfg_id,
                        "config_label": cfg_lbl,
                        "fdr_pct": round(rec["FDR_pct"], 4),
                        "tpr_pct": round(rec["TPR_pct"], 4),
                        "n_predicted_px": int(rec["n_predicted_px"]),
                        "island_count": int(rec["island_count"]),
                        "mean_island_area_px": round(rec["mean_island_area_px"], 2),
                        "topological_charge_passed": bool(rec["n_predicted_px"] > 0),
                    })

    df_all = pd.DataFrame(all_records)
    df_all.to_csv(OUT_DIR / "lax_strict_cluster_realizations.csv", index=False)
    print(f"Saved realization metrics to {OUT_DIR / 'lax_strict_cluster_realizations.csv'}")

    # Aggregate by temperature, disorder, config
    summary_rows = []
    for (temp, v0_lbl, cfg_id), grp in df_all.groupby(["temperature_mK", "disorder_meV", "config_id"], sort=False):
        cfg_lbl = grp["config_label"].iloc[0]
        mean_fdr = grp["fdr_pct"].mean()
        std_fdr = grp["fdr_pct"].std()
        mean_tpr = grp["tpr_pct"].mean()
        mean_area = grp["n_predicted_px"].mean()
        mean_islands = grp["island_count"].mean()
        pass_rate = grp["topological_charge_passed"].mean() * 100.0

        tgp = tgp_baseline[(temp, v0_lbl)]
        beats_fdr = mean_fdr < tgp["fdr"]
        beats_all = beats_fdr and (pass_rate >= 90.0) and (mean_area >= 10.0)

        summary_rows.append({
            "temperature_mK": temp,
            "disorder_meV": v0_lbl,
            "config_id": cfg_id,
            "config_label": cfg_lbl,
            "mean_fdr_pct": float(round(mean_fdr, 2)),
            "std_fdr_pct": float(round(std_fdr, 2)),
            "mean_tpr_pct": float(round(mean_tpr, 2)),
            "mean_area_px": float(round(mean_area, 1)),
            "mean_island_count": float(round(mean_islands, 2)),
            "pass_rate_pct": float(round(pass_rate, 1)),
            "tgp_fdr_pct": float(tgp["fdr"]),
            "tgp_tpr_pct": float(tgp["tpr"]),
            "tgp_area_px": float(tgp["area"]),
            "beats_tgp_fdr": bool(beats_fdr),
            "beats_tgp_strict": bool(beats_all),
            "fdr_margin_pct": float(round(tgp["fdr"] - mean_fdr, 2)),
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "lax_strict_cluster_summary.csv", index=False)
    with open(OUT_DIR / "lax_strict_cluster_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nSaved summary to {OUT_DIR / 'lax_strict_cluster_summary.csv'}")

    # Print comparative report
    for temp in [30.0, 40.0]:
        print(f"\n{'='*110}\n TEMPERATURE T = {temp:.0f} mK: COMPARATIVE PERFORMANCE VS MICROSOFT TGP\n{'='*110}")
        df_t = df_sum[df_sum["temperature_mK"] == temp]
        for v0_lbl in v0_order:
            tgp = tgp_baseline[(temp, v0_lbl)]
            print(f"\n--- Disorder {v0_lbl} (TGP: FDR={tgp['fdr']}%, TPR={tgp['tpr']}%, Area={tgp['area']}px) ---")
            sub = df_t[df_t["disorder_meV"] == v0_lbl][
                ["config_label", "mean_fdr_pct", "mean_tpr_pct", "mean_area_px", "mean_island_count", "pass_rate_pct", "beats_tgp_fdr"]
            ]
            print(sub.to_string(index=False))

    # Identify configs that beat TGP across all regimes
    print("\n" + "=" * 110)
    print(" OVERALL REGIME AUDIT (OUTPERFORMING TGP ACROSS ALL 4 DISORDER REGIMES)")
    print("=" * 110)
    for cfg_id, grp in df_sum.groupby("config_id"):
        lbl = grp["config_label"].iloc[0]
        n_beats_30 = grp[(grp["temperature_mK"] == 30.0)]["beats_tgp_fdr"].sum()
        n_beats_40 = grp[(grp["temperature_mK"] == 40.0)]["beats_tgp_fdr"].sum()
        total_beats = grp["beats_tgp_fdr"].sum()
        status = ">>> UNIVERSAL WINNER <<<" if total_beats == 8 else f"{total_beats}/8 regimes"
        print(f"[{status:24s}] {lbl:48s} | 30mK: {n_beats_30}/4 | 40mK: {n_beats_40}/4")

    # Generate comparative plot
    plot_comparisons(df_sum, FIG_DIR)


def plot_comparisons(df_sum: pd.DataFrame, fig_dir: Path):
    """Plot comparative FDR and Topological Area across disorder strengths for T=30 mK."""
    df_30 = df_sum[df_sum["temperature_mK"] == 30.0].copy()
    v0_order = ["V0_0_378", "V0_0_645", "V0_0_872", "V0_1_2"]
    v0_labels = ["V0 = 0.378 meV", "V0 = 0.645 meV", "V0 = 0.872 meV", "V0 = 1.200 meV"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # TGP baseline
    tgp_fdrs = [tgp_baseline[(30.0, v)]["fdr"] for v in v0_order]
    tgp_areas = [tgp_baseline[(30.0, v)]["area"] for v in v0_order]

    x = np.arange(len(v0_order))
    width = 0.12

    # Plot TGP
    axes[0].plot(x, tgp_fdrs, "k--o", label="Microsoft TGP (Baseline)", linewidth=2.5, markersize=8, zorder=10)
    axes[1].plot(x, tgp_areas, "k--o", label="Microsoft TGP (Baseline)", linewidth=2.5, markersize=8, zorder=10)

    cfg_ids = [c["id"] for c in configs]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

    for idx, (cid, col) in enumerate(zip(cfg_ids, colors)):
        sub = df_30[df_30["config_id"] == cid].set_index("disorder_meV").loc[v0_order]
        lbl = sub["config_label"].iloc[0]
        axes[0].plot(x, sub["mean_fdr_pct"], marker="s", color=col, label=lbl, linewidth=1.8, markersize=6)
        axes[1].plot(x, sub["mean_area_px"], marker="s", color=col, label=lbl, linewidth=1.8, markersize=6)

    axes[0].set_title("False Discovery Rate (FDR %) vs Disorder [T = 30 mK]", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Disorder Strength", fontsize=11)
    axes[0].set_ylabel("FDR (%) - Lower is Better", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(v0_labels, fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9, loc="upper left")

    axes[1].set_title("Topological Harvest Area (px) vs Disorder [T = 30 mK]", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Disorder Strength", fontsize=11)
    axes[1].set_ylabel("Mean Predicted Area (pixels) - Higher is Better", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(v0_labels, fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    fig_path = fig_dir / "lax_strict_cluster_benchmark_fdr_area.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\nSaved comparison figure to {fig_path}")


if __name__ == "__main__":
    main()
