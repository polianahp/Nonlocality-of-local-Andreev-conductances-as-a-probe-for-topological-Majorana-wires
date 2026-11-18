"""Tier-0 exploratory phase maps for spline-derivative Options A, B, and C at baseline G_N = 0.85."""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.interpolate import make_interp_spline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exploratory.plot_quick_correlations import normalize_by_symmetric


def compute_options_abc(g_swept: np.ndarray, g_const: np.ndarray, x: np.ndarray, n_dense: int = 200):
    """Compute vectorized cubic B-spline derivative Options A, B, C along axis=-1."""
    x_dense = np.linspace(float(x[0]), float(x[-1]), n_dense)
    d_sw = make_interp_spline(x, g_swept, k=3, axis=-1)(x_dense, nu=1)
    d_co = make_interp_spline(x, g_const, k=3, axis=-1)(x_dense, nu=1)

    opt_a = np.mean(d_co, axis=-1)
    dot_sc = np.sum(d_sw * d_co, axis=-1)
    norm2_sw = np.sum(d_sw * d_sw, axis=-1)
    norm2_co = np.sum(d_co * d_co, axis=-1)

    opt_b = np.where(norm2_sw > 1e-12, dot_sc / np.maximum(norm2_sw, 1e-12), 0.0)
    opt_c = np.where((norm2_sw + norm2_co) > 1e-12, (2.0 * dot_sc) / np.maximum(norm2_sw + norm2_co, 1e-12), 0.0)
    return opt_a, opt_b, opt_c


def main() -> None:
    data_dir = REPO_ROOT / "Outputs" / "Data" / "Tdispfaff_roi"
    out_dir = REPO_ROOT / "exploratory"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cond = xr.open_dataset(data_dir / "conductance_sweeps.nc", engine="h5netcdf")
    ds_spec = xr.open_dataset(data_dir / "spectra_and_topology.nc", engine="h5netcdf")

    T_plot = 40.0
    b_gn = 0.85

    gmat = ds_cond["Gmat"].sel(temperature=T_plot, baseline_Gn=b_gn)
    pfaff = ds_spec["pfaffian_invariant"].transpose("chemical_potential", "zeeman_field").values > 0
    mu_vals = ds_cond.coords["chemical_potential"].values
    vz_vals = ds_cond.coords["zeeman_field"].values
    gn_vals = ds_cond.coords["G_n"].values

    idx_sym = int(np.argmin(np.abs(gn_vals - b_gn)))
    x = gn_vals / b_gn

    def _nearest(mu_t: float, vz_t: float):
        mi = int(np.argmin(np.abs(mu_vals - mu_t)))
        vi = int(np.argmin(np.abs(vz_vals - vz_t)))
        return mi, vi, float(mu_vals[mi]), float(vz_vals[vi])

    points = [
        ("R1 Topo", "blue", "o", _nearest(2.46, 0.52)),
        ("R1 Triv", "red", "o", _nearest(2.38, 0.52)),
        ("R2 Topo", "deepskyblue", "D", _nearest(2.20, 0.70)),
        ("R2 Triv", "orange", "D", _nearest(2.14, 0.72)),
    ]

    results = {"A": {}, "B": {}, "C": {}}
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
        g_sw, g_co = (g_ll, g_rr) if side == "left" else (g_rr, g_ll)
        opt_a, opt_b, opt_c = compute_options_abc(g_sw, g_co, x)
        results["A"][side] = opt_a
        results["B"][side] = opt_b
        results["C"][side] = opt_c

    specs = [
        ("A", "Option A: Avg Partial Deriv <dG_const / dx>", r"$\langle \partial \tilde{G}_{\mathrm{const}} / \partial x \rangle$", -0.6, 0.6),
        ("B", "Option B: Spline Derivative Slope beta_spline", r"$\beta_{\mathrm{spline}}$", -1.0, 1.0),
        ("C", "Option C: Derivative Concordance rho_c'", r"$\rho_c'$", -1.0, 1.0),
    ]

    for key, label, cbar_lbl, vmin, vmax in specs:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True, constrained_layout=True)
        for ax, side, title in zip(
            axes,
            ["left", "right"],
            [f"Left Barrier Swept ($G_N^R={b_gn:.2f}$)", f"Right Barrier Swept ($G_N^L={b_gn:.2f}$)"],
        ):
            data = results[key][side]
            im = ax.pcolormesh(vz_vals, mu_vals, data, shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
            if np.any(pfaff):
                ax.contour(vz_vals, mu_vals, pfaff.astype(float), levels=[0.5], colors="lime", linewidths=1.3, linestyles="--")

            for pt_lbl, col, mkr, pt in points:
                edge = "white" if mkr == "o" else "black"
                ax.scatter([pt[3]], [pt[2]], c=col, marker=mkr, edgecolors=edge, s=75, linewidths=1.1, zorder=5,
                           label=f"{pt_lbl} ({pt[2]:.2f}, {pt[3]:.2f})")

            ax.set_title(title, fontsize=11)
            ax.set_xlabel(r"$V_z$ (meV)", fontsize=10)
            ax.set_xlim(float(vz_vals[0]), float(vz_vals[-1]))
            ax.set_ylim(float(mu_vals[0]), float(mu_vals[-1]))
            ax.legend(loc="upper left", fontsize=7.8, framealpha=0.88, ncol=2)

        axes[0].set_ylabel(r"$\mu$ (meV)", fontsize=10)
        cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
        cbar.set_label(cbar_lbl, fontsize=10)

        fig.suptitle(
            f"[EXPLORATORY / QUICK-INSPECT — Unregistered] {label} (T={T_plot:.0f} mK, Baseline G_N={b_gn:.2f}, green = Pfaffian)",
            fontsize=11,
        )
        out_path = out_dir / f"Tdispfaff_roi_option_{key}_baseline_{b_gn:.2f}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Saved: {out_path}")

    ds_cond.close()
    ds_spec.close()


if __name__ == "__main__":
    main()
