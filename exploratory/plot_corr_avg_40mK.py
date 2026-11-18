"""Tier-0 exploratory plot: simple average 0.5*(C_L + C_R) at T = 40 mK, baseline G_N = 0.85."""
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

    T_val = 40.0
    b_gn = 0.85
    pfaff = (ds_spec["pfaffian_invariant"] > 0).transpose("chemical_potential", "zeeman_field").values
    mu_vals = ds_cond.coords["chemical_potential"].values
    vz_vals = ds_cond.coords["zeeman_field"].values
    gn_vals = ds_cond.coords["G_n"].values
    idx_sym = int(np.argmin(np.abs(gn_vals - b_gn)))
    asym_ratio = gn_vals / b_gn

    gmat = ds_cond["Gmat"].sel(temperature=T_val, baseline_Gn=b_gn)
    corr_sides = {}
    for side in ("left", "right"):
        sub = gmat.sel(swept_barrier=side)
        g_ll = normalize_by_symmetric(
            sub.sel(lead_row="left", lead_col="left").transpose("chemical_potential", "zeeman_field", "G_n").values,
            idx_sym,
        )
        g_rr = normalize_by_symmetric(
            sub.sel(lead_row="right", lead_col="right").transpose("chemical_potential", "zeeman_field", "G_n").values,
            idx_sym,
        )
        corr_sides[side] = compute_calc_correlation_map(g_ll, g_rr, asym_ratio)

    corr_avg = 0.5 * (corr_sides["left"] + corr_sides["right"])

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    pcm = ax.pcolormesh(vz_vals, mu_vals, corr_avg, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    if np.any(pfaff):
        ax.contour(vz_vals, mu_vals, pfaff.astype(float), levels=[0.5], colors="black", linewidths=1.5, linestyles="--")

    cb = fig.colorbar(pcm, ax=ax, shrink=0.92)
    cb.set_label(r"$\frac{1}{2}(C_L + C_R)$", fontsize=11)

    ax.set_xlabel(r"$V_z$ (meV)", fontsize=11)
    ax.set_ylabel(r"$\mu$ (meV)", fontsize=11)
    ax.set_title(
        f"[EXPLORATORY / QUICK-INSPECT — Unregistered]\nSimple Average Correlation (T = 40 mK, Baseline G_N = {b_gn:.2f} e²/h, black dashed = Pfaffian)",
        fontsize=10.5,
    )

    out_path = out_dir / "Tdispfaff_roi_corr_avg_0.85_40mK.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")

    ds_cond.close()
    ds_spec.close()


if __name__ == "__main__":
    main()
