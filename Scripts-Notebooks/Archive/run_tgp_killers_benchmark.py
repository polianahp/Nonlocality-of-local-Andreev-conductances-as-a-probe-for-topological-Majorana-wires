#!/usr/bin/env python3
"""Benchmark of TGP-Killing Protocol Configurations vs Standard Microsoft TGP Baseline.

Evaluates:
1. Microsoft TGP Stage 2 Reference Baseline
2. Baseline LowCorr+Curv+Gap: C>=0.75, k>=3/5, Curv<=-100, Gap>=10
3. Universal TGP-Killer (5/5 cutters + SymBalance): C>=0.80, k>=5/5, Curv<=-500, Gap>=10, S_sym<=0.25
4. High-Yield TGP-Killer (4/5 cutters + SymBalance): C>=0.80, k>=4/5, Curv<=-100, Gap>=10, S_sym<=0.25
5. Pure 3-Condition Winner: C>=0.85, k>=5/5, Curv<=-500, Gap>=10 (No SymBalance)

Evaluated across all 40 local disorder realizations (10 per V0 in {0.378, 0.645, 0.872, 1.2} meV)
at both T = 30 mK and T = 40 mK.
"""

from __future__ import annotations

import json
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
OUT_DIR = REPO_ROOT / "Outputs" / "Postprocessing" / "LowCorr_Curv3w_TransportGap_Protocol"
PARAM_DIR = REPO_ROOT / "Protocols" / "Low_Asym_Corr_Curv3w_Transport_Gap_Protocol" / "Parameters"
OUT_DIR.mkdir(parents=True, exist_ok=True)

configs = [
    {
        "id": "baseline_corr075_curv100_gap10",
        "file": "lowcorr075_curv100_gap10uev.yaml",
        "label": "Baseline (C>=0.75, k>=3, Curv<=-100, Gap>=10)",
    },
    {
        "id": "universal_tgp_killer_5of5_sym025",
        "file": "lowcorr080_curv500_5of5_gap10_sym025.yaml",
        "label": "Universal TGP-Killer (C>=0.80, k>=5/5, Curv<=-500, S_sym<=0.25)",
    },
    {
        "id": "high_yield_tgp_killer_4of5_sym025",
        "file": "lowcorr080_curv100_4of5_gap10_sym025.yaml",
        "label": "High-Yield TGP-Killer (C>=0.80, k>=4/5, Curv<=-100, S_sym<=0.25)",
    },
    {
        "id": "pure3_tgp_killer_5of5_corr085",
        "file": "lowcorr085_curv500_5of5_gap10_pure3.yaml",
        "label": "Pure 3-Condition Winner (C>=0.85, k>=5/5, Curv<=-500, Gap>=10)",
    },
]

# TGP baselines from all_disorders_master_summary.csv
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

    print("=" * 90)
    print(" BENCHMARK: TGP-KILLING PROTOCOL CONFIGURATIONS VS MICROSOFT TGP BASELINE")
    print("=" * 90)

    for temp in temperatures:
        for v0_lbl in v0_order:
            v0_dir = DATA_ROOT / v0_lbl
            if not v0_dir.exists():
                continue
            reals = sorted([d for d in v0_dir.iterdir() if d.is_dir() and "realization" in d.name])
            print(f"\n--- Evaluating {v0_lbl} at T = {temp} mK ({len(reals)} realizations) ---")

            for r_dir in reals:
                cache = RealizationFeatureCache(r_dir, temperature_mK=temp)

                for cfg in loaded_cfgs:
                    steps = protocol.evaluate_steps(cache, cfg["params"])
                    rec = evaluate_realization_metrics(cache, steps)
                    rec["config_id"] = cfg["id"]
                    rec["config_label"] = cfg["label"]
                    all_records.append(rec)

    # Flatten records
    flattened = []
    for r in all_records:
        flattened.append({
            "config_id": r["config_id"],
            "config_label": r["config_label"],
            "v0_label": r["v0_label"],
            "v0_mev": r["v0_mev"],
            "realization": r["realization_name"],
            "temperature_mK": r["temperature_mK"],
            "n_predicted_px": r["n_predicted_px"],
            "n_true_positive_px": r["n_true_positive_px"],
            "n_false_positive_px": r["n_false_positive_px"],
            "TPR_pct": r["TPR_pct"],
            "FDR_pct": r["FDR_pct"],
            "island_count": r["island_count"],
            "mean_island_area_px": r["mean_island_area_px"],
            "MZM_confinement_C": r["MZM_confinement_C"],
            "MZM_separability_S": r["MZM_separability_S"],
        })

    df = pd.DataFrame(flattened)
    df.to_csv(OUT_DIR / "tgp_killers_detailed.csv", index=False)

    # Ensemble summary
    summary_rows = []
    for temp in temperatures:
        for v0_lbl in v0_order:
            tgp_ref = tgp_baseline[(temp, v0_lbl)]

            # First add TGP reference row
            summary_rows.append({
                "temperature_mK": temp,
                "v0_label": v0_lbl,
                "protocol": "Microsoft TGP Stage 2 Baseline",
                "mean_fdr_pct": tgp_ref["fdr"],
                "mean_tpr_pct": tgp_ref["tpr"],
                "mean_final_px": tgp_ref["area"],
                "pass_rate_pct": 100.0,
                "fdr_vs_tgp_pct": 0.0,
                "beats_tgp": False,
            })

            # Then add each configuration
            for cfg in loaded_cfgs:
                sub = df[(df["temperature_mK"] == temp) & (df["v0_label"] == v0_lbl) & (df["config_id"] == cfg["id"])]
                if len(sub) == 0:
                    continue
                mfdr = sub["FDR_pct"].mean()
                mtpr = sub["TPR_pct"].mean()
                marea = sub["n_predicted_px"].mean()
                pass_rate = 100.0 * (sub["n_predicted_px"] > 0).sum() / len(sub)
                delta_fdr = mfdr - tgp_ref["fdr"]
                beats = (mfdr < tgp_ref["fdr"]) and (pass_rate >= 90.0) and (marea >= 10.0)

                summary_rows.append({
                    "temperature_mK": temp,
                    "v0_label": v0_lbl,
                    "protocol": cfg["label"],
                    "mean_fdr_pct": round(mfdr, 2),
                    "mean_tpr_pct": round(mtpr, 2),
                    "mean_final_px": round(marea, 1),
                    "pass_rate_pct": round(pass_rate, 1),
                    "fdr_vs_tgp_pct": round(delta_fdr, 2),
                    "beats_tgp": beats,
                })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "tgp_killers_summary.csv", index=False)

    print("\n" + "=" * 120)
    print(" SUMMARY: TGP-KILLING CONFIGURATIONS VS MICROSOFT TGP BASELINE (T = 30 mK)")
    print("=" * 120)
    sub30 = df_sum[df_sum["temperature_mK"] == 30.0]
    print(sub30[["v0_label", "protocol", "mean_fdr_pct", "fdr_vs_tgp_pct", "mean_tpr_pct", "mean_final_px", "beats_tgp"]].to_string(index=False))

    print("\n" + "=" * 120)
    print(" SUMMARY: TGP-KILLING CONFIGURATIONS VS MICROSOFT TGP BASELINE (T = 40 mK)")
    print("=" * 120)
    sub40 = df_sum[df_sum["temperature_mK"] == 40.0]
    print(sub40[["v0_label", "protocol", "mean_fdr_pct", "fdr_vs_tgp_pct", "mean_tpr_pct", "mean_final_px", "beats_tgp"]].to_string(index=False))
    print("=" * 120)


if __name__ == "__main__":
    main()
