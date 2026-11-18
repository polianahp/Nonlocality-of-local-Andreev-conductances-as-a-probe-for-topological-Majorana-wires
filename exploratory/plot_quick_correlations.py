"""Tier-0 exploratory diagnostic plots using src.helpers.calc_correlation on symmetrically normalized sweeps."""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.helpers as hp


def normalize_by_symmetric(arr: np.ndarray, idx_sym: int, norm_floor: float = 1e-12) -> np.ndarray:
    """Normalize conductance curves along the last axis by their value at the symmetric cutter point."""
    sym_val = arr[..., idx_sym : idx_sym + 1]
    safe_sym = np.where(np.abs(sym_val) > norm_floor, sym_val, 1.0)
    out = arr / safe_sym
    out = np.where(np.abs(sym_val) > norm_floor, out, 0.0)
    return out


def compute_calc_correlation_map(g_ll: np.ndarray, g_rr: np.ndarray, barrier_axis: np.ndarray) -> np.ndarray:
    """Apply hp.calc_correlation over every (mu, Vz) pixel."""
    n_mu, n_vz, _ = g_ll.shape
    corr = np.zeros((n_mu, n_vz), dtype=np.float64)
    for i in range(n_mu):
        for j in range(n_vz):
            corr[i, j] = hp.calc_correlation(g_ll[i, j], g_rr[i, j], barrier_axis)
    return corr


def main() -> None:
    data_dir = REPO_ROOT / "Outputs" / "Data" / "Tdispfaff_roi"
    out_dir = REPO_ROOT / "exploratory"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cond = xr.open_dataset(data_dir / "conductance_sweeps.nc", engine="h5netcdf")
    ds_spec = xr.open_dataset(data_dir / "spectra_and_topology.nc", engine="h5netcdf")
    ds_s1 = xr.open_dataset(data_dir / "tgp_stage1.nc", engine="h5netcdf")

    T_plot = 0.0
    t_tag = f"{int(round(T_plot))}mK"
    gmat = ds_cond["Gmat"].sel(temperature=T_plot)
    pfaff = ds_spec["pfaffian_invariant"].transpose("chemical_potential", "zeeman_field").values
    mu_vals = ds_cond.coords["chemical_potential"].values
    vz_vals = ds_cond.coords["zeeman_field"].values
    gn_vals = ds_cond.coords["G_n"].values
    baselines = [float(b) for b in ds_cond.coords["baseline_Gn"].values]

    # Recover calibrated barrier heights U_L(G_n) and U_R(G_n) in meV from tgp_stage1.nc
    ord_l = np.argsort(ds_s1.coords["G_n_left"].values)
    ord_r = np.argsort(ds_s1.coords["G_n_right"].values)
    np.testing.assert_allclose(ds_s1.coords["G_n_left"].values[ord_l], gn_vals, atol=1e-6)
    np.testing.assert_allclose(ds_s1.coords["G_n_right"].values[ord_r], gn_vals, atol=1e-6)
    u_left_mev = ds_s1.coords["V_leftcutter"].values[ord_l]
    u_right_mev = ds_s1.coords["V_rightcutter"].values[ord_r]

    def _nearest_idx(mu_t: float, vz_t: float):
        mi = int(np.argmin(np.abs(mu_vals - mu_t)))
        vi = int(np.argmin(np.abs(vz_vals - vz_t)))
        return mi, vi, float(mu_vals[mi]), float(vz_vals[vi])

    # Selected pairs inside the two user-specified regions:
    # Region 1: Vz in [0.50, 0.60], mu in [2.25, 2.50]
    r1_t = _nearest_idx(2.46, 0.52)
    r1_v = _nearest_idx(2.38, 0.52)
    # Region 2: Vz in [0.65, 0.75], mu in [2.00, 2.25]
    r2_t = _nearest_idx(2.20, 0.70)
    r2_v = _nearest_idx(2.14, 0.72)

    assert pfaff[r1_t[0], r1_t[1]] == 1 and pfaff[r1_v[0], r1_v[1]] == 0
    assert pfaff[r2_t[0], r2_t[1]] == 1 and pfaff[r2_v[0], r2_v[1]] == 0

    points = [
        ("Region 1 — Topological", "blue", "o", r1_t),
        ("Region 1 — Trivial", "red", "o", r1_v),
        ("Region 2 — Topological", "deepskyblue", "D", r2_t),
        ("Region 2 — Trivial", "orange", "D", r2_v),
    ]

    for b_gn in baselines:
        idx_sym = int(np.argmin(np.abs(gn_vals - b_gn)))
        asym_ratio = gn_vals / b_gn

        # 1. Phase-diagram correlation plot via hp.calc_correlation (normalized before correlation)
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharey=True, constrained_layout=True)
        for ax, side, title in zip(
            axes,
            ["left", "right"],
            [f"Left Barrier Swept ($G_N^R = {b_gn:.2f}$ held constant)", f"Right Barrier Swept ($G_N^L = {b_gn:.2f}$ held constant)"],
        ):
            sub = gmat.sel(baseline_Gn=b_gn, swept_barrier=side)
            g_ll_raw = sub.sel(lead_row="left", lead_col="left").transpose("chemical_potential", "zeeman_field", "G_n").values
            g_rr_raw = sub.sel(lead_row="right", lead_col="right").transpose("chemical_potential", "zeeman_field", "G_n").values

            g_ll = normalize_by_symmetric(g_ll_raw, idx_sym)
            g_rr = normalize_by_symmetric(g_rr_raw, idx_sym)
            corr = compute_calc_correlation_map(g_ll, g_rr, asym_ratio)

            im = ax.pcolormesh(vz_vals, mu_vals, corr, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
            if np.any(pfaff > 0):
                ax.contour(vz_vals, mu_vals, pfaff, levels=[0.5], colors="magenta", linewidths=1.3, linestyles="--")

            for pt_lbl, col, mkr, pt in points:
                edge = "white" if mkr == "o" else "black"
                ax.scatter([pt[3]], [pt[2]], c=col, marker=mkr, edgecolors=edge, s=80, linewidths=1.1, zorder=5,
                           label=f"{pt_lbl.split(' — ')[0]} {'Topo' if 'Topological' in pt_lbl else 'Triv'} ({pt[2]:.2f}, {pt[3]:.2f})")

            ax.set_title(title, fontsize=11)
            ax.set_xlabel(r"$V_z$ (meV)", fontsize=10)
            ax.set_xlim(float(vz_vals[0]), float(vz_vals[-1]))
            ax.set_ylim(float(mu_vals[0]), float(mu_vals[-1]))
            ax.legend(loc="upper left", fontsize=7.8, framealpha=0.88, ncol=2)

        axes[0].set_ylabel(r"$\mu$ (meV)", fontsize=10)
        cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
        cbar.set_label(r"`hp.calc_correlation` ($G_{LL}/G_{LL,\mathrm{sym}}$, $G_{RR}/G_{RR,\mathrm{sym}}$)", fontsize=10)

        fig.suptitle(
            f"[EXPLORATORY / QUICK-INSPECT — Unregistered] Tdispfaff_roi `calc_correlation` ($T={T_plot:.0f}$ mK, Baseline $G_N={b_gn:.2f}$)",
            fontsize=11,
        )
        corr_path = out_dir / f"Tdispfaff_roi_corr_baseline_{b_gn:.2f}_{t_tag}.png"
        fig.savefig(corr_path, dpi=180)
        plt.close(fig)
        print(f"Saved: {corr_path}")

        # 2. Per-point sweep curves: 4 rows x 2 cols (one point per row, one sweep side per subpanel)
        fig2, axes2 = plt.subplots(4, 2, figsize=(13.2, 13.8), sharex="col", constrained_layout=True)

        tick_indices = sorted(set([0, 3, 7, 10, 14, 17, len(gn_vals) - 1]))
        side_specs = [
            ("left", u_left_mev, rf"Conductance Ratio $G_N^L / G_N^R$ ($G_N^R={b_gn:.2f}$)", r"Left Barrier Height $U_L$ (meV)"),
            ("right", u_right_mev, rf"Conductance Ratio $G_N^R / G_N^L$ ($G_N^L={b_gn:.2f}$)", r"Right Barrier Height $U_R$ (meV)"),
        ]

        side_cache = {}
        for side, _, _, _ in side_specs:
            sub = gmat.sel(baseline_Gn=b_gn, swept_barrier=side)
            g_ll_raw = sub.sel(lead_row="left", lead_col="left").transpose("chemical_potential", "zeeman_field", "G_n").values
            g_rr_raw = sub.sel(lead_row="right", lead_col="right").transpose("chemical_potential", "zeeman_field", "G_n").values
            g_ll_norm = normalize_by_symmetric(g_ll_raw, idx_sym)
            g_rr_norm = normalize_by_symmetric(g_rr_raw, idx_sym)
            corr_map = compute_calc_correlation_map(g_ll_norm, g_rr_norm, asym_ratio)
            side_cache[side] = (g_ll_norm, g_rr_norm, corr_map)

        for row_idx, (pt_lbl, _, _, pt) in enumerate(points):
            is_topo = "Topological" in pt_lbl
            phase_tag = "Topological" if is_topo else "Trivial"
            title_color = "navy" if is_topo else "darkred"
            c_ll = "tab:blue" if is_topo else "tab:red"
            c_rr = "tab:cyan" if is_topo else "darkorange"

            for col_idx, (side, u_swept, bottom_lbl, top_lbl) in enumerate(side_specs):
                ax = axes2[row_idx, col_idx]
                g_ll_norm, g_rr_norm, corr_map = side_cache[side]
                c_val = corr_map[pt[0], pt[1]]

                ax.plot(asym_ratio, g_ll_norm[pt[0], pt[1]], color=c_ll, linestyle="-", marker="o", markersize=3.2,
                        label=rf"$G_{{LL}}/G_{{LL,\mathrm{{sym}}}}$ ({phase_tag})")
                ax.plot(asym_ratio, g_rr_norm[pt[0], pt[1]], color=c_rr, linestyle="--", marker="s", markersize=3.2,
                        label=rf"$G_{{RR}}/G_{{RR,\mathrm{{sym}}}}$ ({phase_tag})")
                ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.1, label=r"Symmetric ($=1$)")
                ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
                ax.set_xlim(float(asym_ratio[0]), float(asym_ratio[-1]))
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc="best")

                sweep_name = "Left Swept" if side == "left" else "Right Swept"
                sub_title = f"{pt_lbl} | $\\mu={pt[2]:.2f},\\ V_z={pt[3]:.2f}$ meV | {sweep_name} ($C={c_val:.3f}$)"

                if row_idx == 0:
                    ax_top = ax.twiny()
                    ax_top.set_xlim(ax.get_xlim())
                    ax_top.set_xticks(asym_ratio[tick_indices])
                    ax_top.set_xticklabels([f"{u_swept[i]:.2f}" for i in tick_indices], fontsize=8.5)
                    ax_top.set_xlabel(top_lbl, fontsize=9.5)
                    ax_top.set_title(sub_title, fontsize=10, color=title_color, pad=8)
                else:
                    ax.set_title(sub_title, fontsize=10, color=title_color)

                if row_idx == len(points) - 1:
                    ax.set_xlabel(bottom_lbl, fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(r"Normalized $G / G_{\mathrm{sym}}$", fontsize=9.5)

        fig2.suptitle(
            f"[EXPLORATORY / QUICK-INSPECT — Unregistered] Per-Point Normalized Sweeps w/ `calc_correlation` ($T={T_plot:.0f}$ mK, Baseline $G_N={b_gn:.2f}$)",
            fontsize=11.5,
        )
        sweep_path = out_dir / f"Tdispfaff_roi_sweep_curves_baseline_{b_gn:.2f}_{t_tag}.png"
        fig2.savefig(sweep_path, dpi=180)
        plt.close(fig2)
        print(f"Saved: {sweep_path}")

    ds_cond.close()
    ds_spec.close()
    ds_s1.close()


if __name__ == "__main__":
    main()
