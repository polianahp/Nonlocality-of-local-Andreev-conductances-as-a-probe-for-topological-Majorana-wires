#!/usr/bin/env python3
"""Generate diagnostic visual demonstration of Lax Protocol + Strict Clustering.

Plots for a representative realization (e.g., V0 = 0.645 meV, Realization 10):
Panel 1: Ground Truth Pfaffian Topological Mask
Panel 2: Raw Candidate Mask (before C4 post-processing)
Panel 3: Standard Post-Processing (Area >= 10 px, Core 2x2)
Panel 4: Strict Cluster Pruning (Area >= 25 px, Core 2x2)
Panel 5: Ultra-Strict Cluster Pruning (Area >= 40 px, Core 3x3)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.Protocol_Elements import RealizationFeatureCache, FPE, DPE
from Protocols.Low_Asym_Corr_Curv3w_Transport_Gap_Protocol.protocol import Protocol

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "Outputs" / "Data" / "Disorder_Realizations"
FIG_DIR = REPO_ROOT / "Outputs" / "Postprocessing" / "LowCorr_Curv3w_TransportGap_Protocol" / "figures"
PARAM_DIR = REPO_ROOT / "Protocols" / "Low_Asym_Corr_Curv3w_Transport_Gap_Protocol" / "Parameters"

# Choose V0 = 0.645 meV, disorder_realization_10 at T = 30 mK
r_dir = DATA_ROOT / "V0_0_645" / "disorder_realization_10"
cache = RealizationFeatureCache(r_dir, temperature_mK=30.0)
protocol = Protocol()

with open(PARAM_DIR / "lax_base_strict_cluster25.yaml") as f:
    cfg = yaml.safe_load(f)["parameters"]

steps = protocol.evaluate_steps(cache, cfg)

raw = steps["step_4_raw_candidate"]
c10 = DPE.dpe_canonical_post_processor(raw, use_core_filter=True, core_shape=(2, 2), min_island_size=10, area_connectivity=8, core_connectivity=8)
c25 = steps["step_5_final_mask"]
c40 = DPE.dpe_canonical_post_processor(raw, use_core_filter=True, core_shape=(3, 3), min_island_size=40, area_connectivity=8, core_connectivity=8)
pf = cache.pfaffian_mask

mu = cache.mu_vals
vz = cache.vz_vals
extent = [vz[0], vz[-1], mu[0], mu[-1]]

fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharex=True, sharey=True)

cmap_pf = "Purples"
cmap_pred = "Blues"

axes[0].imshow(pf, origin="lower", extent=extent, cmap="Greens", aspect="auto")
axes[0].set_title(f"1. Pfaffian Ground Truth\n(Area = {np.sum(pf)} px)", fontsize=11, fontweight="bold")

axes[1].imshow(raw, origin="lower", extent=extent, cmap=cmap_pred, aspect="auto")
fp_raw = np.sum(raw & ~pf) / np.sum(raw) * 100 if np.sum(raw) > 0 else 0
axes[1].set_title(f"2. Raw Candidate Mask\n(Area = {np.sum(raw)} px, FDR = {fp_raw:.1f}%)", fontsize=11, fontweight="bold")

axes[2].imshow(c10, origin="lower", extent=extent, cmap=cmap_pred, aspect="auto")
fp_c10 = np.sum(c10 & ~pf) / np.sum(c10) * 100 if np.sum(c10) > 0 else 0
axes[2].set_title(f"3. C4 Standard (>=10 px, 2x2)\n(Area = {np.sum(c10)} px, FDR = {fp_c10:.1f}%)", fontsize=11, fontweight="bold")

axes[3].imshow(c25, origin="lower", extent=extent, cmap=cmap_pred, aspect="auto")
fp_c25 = np.sum(c25 & ~pf) / np.sum(c25) * 100 if np.sum(c25) > 0 else 0
axes[3].set_title(f"4. C4 Strict (>=25 px, 2x2)\n(Area = {np.sum(c25)} px, FDR = {fp_c25:.1f}%)", fontsize=11, fontweight="bold")

axes[4].imshow(c40, origin="lower", extent=extent, cmap=cmap_pred, aspect="auto")
fp_c40 = np.sum(c40 & ~pf) / np.sum(c40) * 100 if np.sum(c40) > 0 else 0
axes[4].set_title(f"5. C4 Strict (>=40 px, 3x3)\n(Area = {np.sum(c40)} px, FDR = {fp_c40:.1f}%)", fontsize=11, fontweight="bold")

# Overlay Pfaffian Ground Truth contour on all panels
for ax in axes:
    ax.contour(vz, mu, pf.astype(float), levels=[0.5], colors=["#dc2626"], linewidths=1.8, linestyles="--")
    ax.set_xlabel(r"$V_z$ (meV)", fontsize=10)
axes[0].set_ylabel(r"$\mu$ (meV)", fontsize=10)

plt.suptitle(r"Lax Protocol + Strict Cluster Pruning ($V_0 = 0.645$ meV, Realization 10, $T = 30$ mK) [Red Dashed Line: $\mathcal{Q} = -1$ Boundary]", fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
out_path = FIG_DIR / "lax_cluster_pruning_demo_V0_0_645.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved demo figure to {out_path}")
