#!/usr/bin/env python3
"""Diagnostic Phase Map Generator for Low-Asymmetry Correlation + 3w Curvature + Transport Gap Protocol."""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.Protocol_Elements import RealizationFeatureCache
from Protocols.Low_Asym_Corr_Curv3w_Transport_Gap_Protocol.protocol import Protocol

DATA_ROOT = REPO_ROOT / "Outputs" / "Data" / "Disorder_Realizations"
FIG_DIR = REPO_ROOT / "Outputs" / "Postprocessing" / "LowCorr_Curv3w_TransportGap_Protocol" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

protocol = Protocol()
params = protocol.default_parameters

targets = [
    ("V0_0_872", "disorder_realization_17", 30.0, "Realization 17 (V0 = 0.872 meV, 30 mK)"),
    ("V0_1_2", "disorder_realization_10", 30.0, "Realization 10 (V0 = 1.200 meV, 30 mK)"),
    ("V0_0_378", "disorder_realization_19", 30.0, "Realization 19 (V0 = 0.378 meV, 30 mK - Lowest FDR)"),
    ("V0_1_2", "disorder_realization_10", 40.0, "Realization 10 (V0 = 1.200 meV, 40 mK)"),
]

cmap_conf = mcolors.ListedColormap(["#0f172a", "#f59e0b", "#059669"])
norm_conf = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_conf.N)

for v0_lbl, r_name, temp, title in targets:
    r_dir = DATA_ROOT / v0_lbl / r_name
    if not r_dir.exists():
        continue
    cache = RealizationFeatureCache(r_dir, temperature_mK=temp)
    gt = cache.pfaffian_mask
    steps = protocol.evaluate_steps(cache, params)

    m_corr = steps["step_1_m_corr"]
    m_curv = steps["step_2_m_curv_3of5"]
    m_gap = steps["step_3_m_gap"]
    m_raw = steps["step_4_raw_candidate"]
    m_final = steps["step_5_final_mask"]

    fig, axes = plt.subplots(1, 5, figsize=(32, 6.5), sharex=True, sharey=True, layout="constrained")
    fig.patch.set_facecolor("#020617")

    vz_vals = cache.vz_vals
    mu_vals = cache.mu_vals
    extent = [vz_vals[0], vz_vals[-1], mu_vals[0], mu_vals[-1]]

    panels = [
        ("Step 1: Low-Asym Corr (>= 0.75)", m_corr, "#38bdf8"),
        ("Step 2: 3w <= -100 (Both, 3/5)", m_curv, "#a855f7"),
        ("Step 3: Transport Gap (>= 10 ueV)", m_gap, "#f59e0b"),
        ("Step 4: Raw Intersection", m_raw, None),
        ("Step 5: Final Filtered (Area>=10 + Core)", m_final, None),
    ]

    for ax, (p_title, mask, col) in zip(axes, panels):
        ax.set_facecolor("#030712")
        if col is not None:
            # Single condition binary map
            img = np.zeros((*mask.shape, 4))
            img[mask] = mcolors.to_rgba(col, alpha=0.85)
            ax.imshow(img, origin="lower", extent=extent, aspect="auto")
        else:
            # TP/FP confusion map
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

        ax.set_title(f"{p_title}\n[Passed: {n_pass} px | TPR: {tpr:.1f}% | FDR: {fdr:.1f}%]", fontsize=11, fontweight="bold", color="white", pad=8)
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

    fig.suptitle(f"{title}: Protocol Waterfall (LowCorr >= 0.75 + 3w <= -100 + Gap >= 10 ueV)", fontsize=14, fontweight="bold", color="#38bdf8", y=1.03)

    out_name = f"waterfall_{v0_lbl}_{r_name}_{int(temp)}mK.png"
    out_path = FIG_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic phase map to {out_path}")
