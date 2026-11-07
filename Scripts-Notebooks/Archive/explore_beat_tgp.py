#!/usr/bin/env python3
"""Explore parameter variations to beat TGP across all disorder strengths."""

from pathlib import Path
import numpy as np
import pandas as pd
from src.Protocol_Elements import BaseProtocol, RealizationFeatureCache, FPE, DPE

DATA_ROOT = Path("Outputs/Data/Disorder_Realizations")
v0_dirs = ["V0_0_378", "V0_0_645", "V0_0_872", "V0_1_2"]

tgp_targets = {
    "V0_0_378": 11.69,
    "V0_0_645": 24.87,
    "V0_0_872": 25.34,
    "V0_1_2": 20.55,
}

print("Loading caches...")
caches = {
    v0: [RealizationFeatureCache(d, temperature_mK=30.0) for d in sorted((DATA_ROOT / v0).iterdir()) if d.is_dir() and "realization" in d.name]
    for v0 in v0_dirs
}

features = {}
for v0, c_list in caches.items():
    features[v0] = []
    for c in c_list:
        cL_5, cR_5 = FPE.barrier_correlation.extract_barrier_reflection_correlations(c, asym_min=-5.0, asym_max=5.0)
        cL_3, cR_3 = FPE.barrier_correlation.extract_barrier_reflection_correlations(c, asym_min=-3.0, asym_max=3.0)
        lockin = c.get_stage1_broadened_and_lockin()
        _, _, gap = FPE.transport_and_spectral_gap.extract_nonlocal_transport_gap_maps(c)
        pf = c.pfaffian_mask
        features[v0].append({
            "cL_5": cL_5, "cR_5": cR_5,
            "cL_3": cL_3, "cR_3": cR_3,
            "I_3w_L": lockin["I_3w_L"],
            "I_3w_R": lockin["I_3w_R"],
            "gap": gap,
            "pf": pf,
            "cache": c,
        })

print("Features extracted. Starting parameter sweeps...")

# Let's test a broad grid
# We want to explore:
# - corr_th: 0.75, 0.80, 0.82, 0.85
# - window: 5.0 vs 3.0
# - k_min: 3 vs 4 vs 5
# - curv_th: 100, 300, 500, 1000
# - gap_mev: 0.005, 0.010, 0.015
# - s_sym: None, 0.35, 0.30, 0.25, 0.20

results = []

grid = [
    # Baseline & variants
    {"corr_th": 0.75, "asym_win": 5.0, "k_min": 3, "curv_th": 100.0, "gap": 0.010, "s_sym": None},
    {"corr_th": 0.80, "asym_win": 5.0, "k_min": 3, "curv_th": 100.0, "gap": 0.010, "s_sym": None},
    {"corr_th": 0.82, "asym_win": 5.0, "k_min": 3, "curv_th": 100.0, "gap": 0.010, "s_sym": None},
    {"corr_th": 0.85, "asym_win": 5.0, "k_min": 3, "curv_th": 100.0, "gap": 0.010, "s_sym": None},
    # Cutter consensus k >= 4
    {"corr_th": 0.75, "asym_win": 5.0, "k_min": 4, "curv_th": 100.0, "gap": 0.010, "s_sym": None},
    {"corr_th": 0.80, "asym_win": 5.0, "k_min": 4, "curv_th": 100.0, "gap": 0.010, "s_sym": None},
    {"corr_th": 0.82, "asym_win": 5.0, "k_min": 4, "curv_th": 100.0, "gap": 0.010, "s_sym": None},
    # Cutter consensus k == 5
    {"corr_th": 0.75, "asym_win": 5.0, "k_min": 5, "curv_th": 100.0, "gap": 0.010, "s_sym": None},
    {"corr_th": 0.80, "asym_win": 5.0, "k_min": 5, "curv_th": 100.0, "gap": 0.010, "s_sym": None},
    # Higher curvature threshold
    {"corr_th": 0.75, "asym_win": 5.0, "k_min": 3, "curv_th": 500.0, "gap": 0.010, "s_sym": None},
    {"corr_th": 0.80, "asym_win": 5.0, "k_min": 3, "curv_th": 500.0, "gap": 0.010, "s_sym": None},
    {"corr_th": 0.75, "asym_win": 5.0, "k_min": 4, "curv_th": 500.0, "gap": 0.010, "s_sym": None},
    {"corr_th": 0.80, "asym_win": 5.0, "k_min": 4, "curv_th": 500.0, "gap": 0.010, "s_sym": None},
    # Narrower window [-3, 3]
    {"corr_th": 0.75, "asym_win": 3.0, "k_min": 3, "curv_th": 100.0, "gap": 0.010, "s_sym": None},
    {"corr_th": 0.80, "asym_win": 3.0, "k_min": 3, "curv_th": 100.0, "gap": 0.010, "s_sym": None},
    {"corr_th": 0.80, "asym_win": 3.0, "k_min": 4, "curv_th": 100.0, "gap": 0.010, "s_sym": None},
    # With Lead Symmetry Balance S_sym <= s_max
    {"corr_th": 0.75, "asym_win": 5.0, "k_min": 3, "curv_th": 100.0, "gap": 0.010, "s_sym": 0.35},
    {"corr_th": 0.75, "asym_win": 5.0, "k_min": 3, "curv_th": 100.0, "gap": 0.010, "s_sym": 0.30},
    {"corr_th": 0.75, "asym_win": 5.0, "k_min": 3, "curv_th": 100.0, "gap": 0.010, "s_sym": 0.25},
    {"corr_th": 0.75, "asym_win": 5.0, "k_min": 3, "curv_th": 100.0, "gap": 0.010, "s_sym": 0.20},
    {"corr_th": 0.80, "asym_win": 5.0, "k_min": 3, "curv_th": 100.0, "gap": 0.010, "s_sym": 0.30},
    {"corr_th": 0.80, "asym_win": 5.0, "k_min": 3, "curv_th": 100.0, "gap": 0.010, "s_sym": 0.25},
    {"corr_th": 0.80, "asym_win": 5.0, "k_min": 3, "curv_th": 100.0, "gap": 0.010, "s_sym": 0.20},
    {"corr_th": 0.80, "asym_win": 5.0, "k_min": 4, "curv_th": 100.0, "gap": 0.010, "s_sym": 0.30},
    {"corr_th": 0.80, "asym_win": 5.0, "k_min": 4, "curv_th": 100.0, "gap": 0.010, "s_sym": 0.25},
    {"corr_th": 0.80, "asym_win": 5.0, "k_min": 4, "curv_th": 100.0, "gap": 0.010, "s_sym": 0.20},
    # Gap variation
    {"corr_th": 0.80, "asym_win": 5.0, "k_min": 3, "curv_th": 100.0, "gap": 0.015, "s_sym": 0.25},
    {"corr_th": 0.80, "asym_win": 5.0, "k_min": 4, "curv_th": 100.0, "gap": 0.015, "s_sym": 0.25},
]

for g in grid:
    row = dict(g)
    beats_count = 0
    beats_all = True
    for v0 in v0_dirs:
        fdrs, tprs, areas, passes = [], [], [], []
        for feat in features[v0]:
            cL = feat["cL_5"] if g["asym_win"] == 5.0 else feat["cL_3"]
            cR = feat["cR_5"] if g["asym_win"] == 5.0 else feat["cR_3"]
            m_corr = (cL >= g["corr_th"]) & (cR >= g["corr_th"])
            both_curv = (feat["I_3w_L"] <= -g["curv_th"]) & (feat["I_3w_R"] <= -g["curv_th"])
            m_curv = np.sum(both_curv, axis=0) >= g["k_min"]
            m_gap = feat["gap"] >= g["gap"]
            raw = m_corr & m_curv & m_gap
            if g["s_sym"] is not None:
                s_mask = FPE.barrier_asymmetry.fpe_lead_symmetry_balance(feat["cache"], s_max=g["s_sym"])
                raw = raw & s_mask

            final = DPE.dpe_canonical_post_processor(
                raw,
                use_core_filter=True,
                core_shape=(2, 2),
                min_island_size=10,
                area_connectivity=8,
                core_connectivity=8,
            )
            n_pred = int(np.sum(final))
            tp = int(np.sum(final & feat["pf"]))
            fp = n_pred - tp
            fdr = fp / n_pred * 100 if n_pred > 0 else 0.0
            tpr = tp / np.sum(feat["pf"]) * 100 if np.sum(feat["pf"]) > 0 else 0.0
            fdrs.append(fdr)
            tprs.append(tpr)
            areas.append(n_pred)
            passes.append(n_pred > 0)

        mean_fdr = float(np.mean(fdrs))
        mean_tpr = float(np.mean(tprs))
        mean_area = float(np.mean(areas))
        pass_rate = float(np.mean(passes) * 100)

        row[f"{v0}_fdr"] = round(mean_fdr, 2)
        row[f"{v0}_tpr"] = round(mean_tpr, 2)
        row[f"{v0}_area"] = round(mean_area, 1)
        row[f"{v0}_pass"] = round(pass_rate, 1)

        tgt = tgp_targets[v0]
        # Beats TGP if FDR < TGP FDR and pass rate >= 90% and area >= 10 px
        beats = (mean_fdr < tgt) and (pass_rate >= 90.0) and (mean_area >= 10.0)
        row[f"{v0}_beats"] = beats
        if beats:
            beats_count += 1
        else:
            beats_all = False

    row["beats_count"] = beats_count
    row["beats_all"] = beats_all
    results.append(row)

df_res = pd.DataFrame(results)
df_res = df_res.sort_values(by=["beats_count", "V0_1_2_fdr", "V0_0_872_fdr"], ascending=[False, True, True])

print("\n" + "=" * 120)
print(" TOP CONFIGURATIONS SORTED BY DISORDER BEAT COUNT & STRONG-DISORDER FDR")
print("=" * 120)
cols = [
    "corr_th", "k_min", "curv_th", "s_sym", "beats_count",
    "V0_0_378_fdr", "V0_0_645_fdr", "V0_0_872_fdr", "V0_1_2_fdr",
    "V0_0_378_area", "V0_0_645_area", "V0_0_872_area", "V0_1_2_area",
    "V0_1_2_pass"
]
print(df_res[cols].head(15).to_string(index=False))

# Filter configs where beats_count == 4
perfect = df_res[df_res["beats_all"] == True]
print(f"\nFound {len(perfect)} configurations that BEAT TGP ACROSS ALL 4 DISORDER REGIMES!")
if len(perfect) > 0:
    print(perfect[cols].to_string(index=False))
