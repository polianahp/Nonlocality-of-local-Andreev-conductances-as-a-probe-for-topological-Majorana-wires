"""Tier-0 exploratory side-by-side comparison of calc_correlation at 0 mK vs 40 mK (baseline G_N = 0.85)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exploratory.plot_quick_correlations import normalize_by_symmetric, compute_calc_correlation_map


def main() -> None:
    data_dir = REPO_ROOT / "Outputs" / "Data" / "Tdispfaff_roi"
    out_dir = REPO_ROOT / "exploratory"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cond = xr.open_dataset(data_dir / "conductance_sweeps.nc", engine="h5netcdf")
    ds_spec = xr.open_dataset(data_dir / "spectra_and_topology.nc", engine="h5netcdf")

    b_gn = 0.85
    temps = (0.0, 40.0)
    sides = ("left", "right")

    pfaff = (ds_spec["pfaffian_invariant"] > 0).transpose("chemical_potential", "zeeman_field").values
    mu_vals = ds_cond.coords["chemical_potential"].values
    vz_vals = ds_cond.coords["zeeman_field"].values
    gn_vals = ds_cond.coords["G_n"].values
    idx_sym = int(np.argmin(np.abs(gn_vals - b_gn)))
    asym_ratio = gn_vals / b_gn

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0), sharex=True, sharey=True, constrained_layout=True)

    pcm_last = None
    for r, side in enumerate(sides):
        for c, T_val in enumerate(temps):
            ax = axes[r, c]
            gmat = ds_cond["Gmat"].sel(temperature=T_val, baseline_Gn=b_gn)
            sub = gmat.sel(swept_barrier=side)
            g_ll = normalize_by_symmetric(
                sub.sel(lead_row="left", lead_col="left").transpose("chemical_potential", "zeeman_field", "G_n").values,
                idx_sym,
            )
            g_rr = normalize_by_symmetric(
                sub.sel(lead_row="right", lead_col="right").transpose("chemical_potential", "zeeman_field", "G_n").values,
                idx_sym,
            )
            corr_map = compute_calc_correlation_map(g_ll, g_rr, asym_ratio)

            pcm_last = ax.pcolormesh(vz_vals, mu_vals, corr_map, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
            if np.any(pfaff):
                ax.contour(vz_vals, mu_vals, pfaff.astype(float), levels=[0.5], colors="black", linewidths=1.4, linestyles="--")

            side_lbl = "Left Swept ($C_L$)" if side == "left" else "Right Swept ($C_R$)"
            ax.set_title(f"{side_lbl} — T = {T_val:.0f} mK", fontsize=11, fontweight="bold")
            if r == 1:
                ax.set_xlabel(r"$V_z$ (meV)", fontsize=10.5)
            if c == 0:
                ax.set_ylabel(r"$\mu$ (meV)", fontsize=10.5)
            ax.set_xlim(float(vz_vals[0]), float(vz_vals[-1]))
            ax.set_ylim(float(mu_vals[0]), float(mu_vals[-1]))

    cb = fig.colorbar(pcm_last, ax=axes, shrink=0.88, pad=0.02)
    cb.set_label(r"`calc_correlation` ($C_L$ / $C_R$)", fontsize=10.5)

    fig.suptitle(
        f"[EXPLORATORY / QUICK-INSPECT — Unregistered] Correlation Phase Map: 0 mK vs 40 mK (Baseline G_N = {b_gn:.2f} e²/h, black dashed = Pfaffian)",
        fontsize=12,
    )

    out_path = out_dir / "Tdispfaff_roi_corr_0.85_0mK_vs_40mK.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")

    ds_cond.close()
    ds_spec.close()


if __name__ == "__main__":
    main()
