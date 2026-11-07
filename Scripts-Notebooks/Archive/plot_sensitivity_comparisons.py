#!/usr/bin/env python3
"""Plot 5-Variation Sensitivity Comparisons for Representative Realizations."""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.Protocol_Elements import RealizationFeatureCache
from Protocols.Low_Asym_Corr_Curv3w_Transport_Gap_Protocol.protocol import Protocol

DATA_ROOT = REPO_ROOT / "Outputs" / "Data" / "Disorder_Realizations"
FIG_DIR = REPO_ROOT / "Outputs" / "Postprocessing" / "LowCorr_Curv3w_TransportGap_Protocol" / "figures"
PARAM_DIR = REPO_ROOT / "Protocols" / "Low_Asym_Corr_Curv3w_Transport_Gap_Protocol" / "Parameters"
FIG_DIR.mkdir(parents=True, exist_ok=True)

configs = [
    ("Baseline\n(Curv<=-100, Gap>=10, Corr>=0.75)", "lowcorr075_curv100_gap10uev.yaml"),
    ("Variation: Curv <= -1000\n(Gap>=10, Corr>=0.75)", "lowcorr075_curv1000_gap10uev.yaml"),
    ("Variation: Gap >= 20 ueV\n(Curv<=-100, Corr>=0.75)", "lowcorr075_curv100_gap20uev.yaml"),
    ("Variation: Corr >= 0.80\n(Curv<=-100, Gap>=10)", "lowcorr080_curv100_gap10uev.yaml"),
    ("Variation: Corr >= 0.90\n(Curv<=-100, Gap>=10)", "lowcorr090_curv100_gap10uev.yaml"),
]

targets = [
    ("V0_0_872", "disorder_realization_17", 30.0, "Realization 17 (V0 = 0.872 meV, T = 30 mK)"),
    ("V0_1_2", "disorder_realization_10", 30.0, "Realization 10 (V0 = 1.200 meV, T = 30 mK)"),
    ("V0_0_378", "disorder_realization_19", 30.0, "Realization 19 (V0 = 0.378 meV, T = 30 mK)"),
]

protocol = Protocol()
cmap_conf = mcolors.ListedColormap(["#0f172a", "#f59e0b", "#059669"])
norm_conf = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_conf.N)

for v0_lbl, r_name, temp, title in targets:
    r_dir = DATA_ROOT / v0_lbl / r_name
    if not r_dir.exists():
        continue
    cache = RealizationFeatureCache(r_dir, temperature_mK=temp)
    gt = cache.pfaffian_mask
    vz_vals = cache.vz_vals
    mu_vals = cache.mu_vals

    fig, axes = plt.subplots(1, 5, figsize=(32, 6.5), sharex=True, sharey=True, layout="constrained")
    fig.patch.set_facecolor("#020617")

    for ax, (lbl, cfg_file) in zip(axes, configs):
        with open(PARAM_DIR / cfg_file) as f:
            cfg = yaml.safe_load(f)
        steps = protocol.evaluate_steps(cache, cfg["parameters"])
        mask = steps["step_5_final_mask"]

        ax.set_facecolor("#030712")
        c2d = np.zeros_like(mask, dtype=int)
        c2d[mask & (~gt)] = 1
        c2d[mask & gt] = 2
        ax.pcolormesh(vz_vals, mu_vals, c2d, cmap=cmap_conf, norm=norm_conf, shading="nearest")
        ax.contour(vz_vals, mu_vals, gt.astype(float), levels=[0.5], colors=["#06b6d4"], linewidths=[2.0], linestyles="--")

        n_pass = int(np.sum(mask))
        tp = int(np.sum(mask & gt))
        fp = int(np.sum(mask & (~gt)))
        tpr = tp / np.sum(gt) * 100 if np.sum(gt) > 0 else 0
        fdr = fp / (tp + fp) * 100 if (tp + fp) > 0 else 0

        ax.set_title(f"{lbl}\n[Passed: {n_pass} px | TPR: {tpr:.1f}% | FDR: {fdr:.1f}%]", fontsize=10.5, fontweight="bold", color="white", pad=8)
        ax.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=11, color="white")
        ax.tick_params(colors="white")
        ax.grid(True, linestyle=":", alpha=0.25, color="gray")

    axes[0].set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=11, color="white")

    legend_patches = [
        mpatches.Patch(facecolor="#059669", edgecolor="none", label="True Positive (TP)"),
        mpatches.Patch(facecolor="#f59e0b", edgecolor="none", label="False Positive (FP)"),
        plt.Line2D([0], [0], color="#06b6d4", lw=2.0, ls="--", label=r"Pfaffian Ground Truth ($\mathcal{Q} = -1$)"),
    ]
    axes[-1].legend(handles=legend_patches, loc="lower right", facecolor="#0f172a", edgecolor="#334155", labelcolor="white", fontsize=9.5)

    out_file = FIG_DIR / f"sensitivity_comparison_{v0_lbl}_{r_name}_{int(temp)}mK.png"
    plt.savefig(out_file, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved sensitivity comparison map to {out_file}")

print("All sensitivity maps generated successfully.")
