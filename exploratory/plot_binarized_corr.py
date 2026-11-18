"""Tier-0 exploratory binarized phase map at baseline G_N = 0.85 with C_L >= 0.75 and C_R >= 0.75."""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exploratory.plot_quick_correlations import normalize_by_symmetric, compute_calc_correlation_map


def main() -> None:
    data_dir = REPO_ROOT / "Outputs" / "Data" / "Tdispfaff_roi"
    out_dir = REPO_ROOT / "exploratory"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cond = xr.open_dataset(data_dir / "conductance_sweeps.nc", engine="h5netcdf")
    ds_spec = xr.open_dataset(data_dir / "spectra_and_topology.nc", engine="h5netcdf")

    T_plot = 40.0
    b_gn = 0.85
    thresholds = (0.90, 0.95)

    gmat = ds_cond["Gmat"].sel(temperature=T_plot, baseline_Gn=b_gn)
    pfaff = ds_spec["pfaffian_invariant"].transpose("chemical_potential", "zeeman_field").values > 0
    mu_vals = ds_cond.coords["chemical_potential"].values
    vz_vals = ds_cond.coords["zeeman_field"].values
    gn_vals = ds_cond.coords["G_n"].values

    idx_sym = int(np.argmin(np.abs(gn_vals - b_gn)))
    asym_ratio = gn_vals / b_gn

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

    for thresh in thresholds:
        mask_l = corr_sides["left"] >= thresh
        mask_r = corr_sides["right"] >= thresh
        mask_both = mask_l & mask_r

        n_tot = mask_both.size
        n_pf = int(np.count_nonzero(pfaff))
        n_pass = int(np.count_nonzero(mask_both))
        tp = int(np.count_nonzero(mask_both & pfaff))
        fp = n_pass - tp
        fdr = 100.0 * fp / n_pass if n_pass > 0 else 0.0
        tpr = 100.0 * tp / n_pf if n_pf > 0 else 0.0
        print(f"Baseline={b_gn}, thresh={thresh:.2f}: L={mask_l.sum()}/{n_tot}, R={mask_r.sum()}/{n_tot}, Both={n_pass}/{n_tot} | TP={tp}, FP={fp}, TPR={tpr:.1f}%, FDR={fdr:.1f}%")

        cmap_bin = ListedColormap(["#f0f2f5", "#1f77b4"])
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), sharey=True, constrained_layout=True)

        panels = [
            (mask_l, rf"Left Swept: $C_L \geq {thresh:.2f}$ ({int(mask_l.sum())}/{n_tot} px)"),
            (mask_r, rf"Right Swept: $C_R \geq {thresh:.2f}$ ({int(mask_r.sum())}/{n_tot} px)"),
            (mask_both, rf"Both Sides ({n_pass}/{n_tot} px | TPR {tpr:.1f}%, FDR {fdr:.1f}%)"),
        ]

        for ax, (mask, title) in zip(axes, panels):
            ax.pcolormesh(vz_vals, mu_vals, mask.astype(float), shading="auto", cmap=cmap_bin, vmin=0.0, vmax=1.0)
            if np.any(pfaff):
                ax.contour(vz_vals, mu_vals, pfaff.astype(float), levels=[0.5], colors="magenta", linewidths=1.3, linestyles="--")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(r"$V_z$ (meV)", fontsize=10)
            ax.set_xlim(float(vz_vals[0]), float(vz_vals[-1]))
            ax.set_ylim(float(mu_vals[0]), float(mu_vals[-1]))

        axes[0].set_ylabel(r"$\mu$ (meV)", fontsize=10)
        fig.suptitle(
            f"[EXPLORATORY / QUICK-INSPECT — Unregistered] Binarized `calc_correlation` >= {thresh:.2f} (T={T_plot:.0f} mK, Baseline G_N={b_gn:.2f}, magenta dashed = Pfaffian)",
            fontsize=11,
        )

        out_path = out_dir / f"Tdispfaff_roi_binarized_corr_{b_gn:.2f}_thresh_{thresh:.2f}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Saved: {out_path}")

    ds_cond.close()
    ds_spec.close()


if __name__ == "__main__":
    main()
