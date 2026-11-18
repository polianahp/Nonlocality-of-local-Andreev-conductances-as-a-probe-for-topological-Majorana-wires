"""Tier-0 exploratory plots: selected mu cuts (spectra + local/nonlocal conductance) & color-coded phase map."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import xarray as xr
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import tgp.two
import tgp.plot
import tgp.plot.paper

from exploratory.plot_quick_correlations import normalize_by_symmetric, compute_calc_correlation_map


def antisymmetrize(g_arr: np.ndarray) -> np.ndarray:
    """Compute A(G)(V) = 0.5 * (G(V) - G(-V)) along the last (bias) axis."""
    return 0.5 * (g_arr - g_arr[..., ::-1])


def plot_single_cut(
    *,
    cut_name: str,
    mu_val: float,
    vz_vals: np.ndarray,
    bias_uev: np.ndarray,
    spec_slice: np.ndarray,
    pfaff_slice: np.ndarray,
    s2_slice: np.ndarray,
    s3_slice: np.ndarray,
    g_ll: np.ndarray,
    g_rr: np.ndarray,
    g_rl: np.ndarray,
    g_lr: np.ndarray,
    gap_rl_uev: np.ndarray,
    gap_lr_uev: np.ndarray,
    vz_markers: list[tuple[float, str]],
    out_path: Path,
) -> None:
    """Plot spectra + G_LL + G_RR + A(G_RL) + A(G_LR) along a fixed-mu cut."""
    dvz = float(np.median(np.diff(vz_vals)))
    # Sort eigenvalues per Vz so innermost +/-E0 are red and outer +/-E1 are blue
    spec_sorted = np.sort(spec_slice, axis=-1)  # (N_Vz, 4)

    fig = plt.figure(figsize=(6.8, 11.2))
    gs = fig.add_gridspec(
        5,
        2,
        width_ratios=[1.0, 0.032],
        height_ratios=[1.55, 1.0, 1.0, 1.0, 1.0],
        hspace=0.14,
        wspace=0.04,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    axes = [ax0] + [fig.add_subplot(gs[r, 0], sharex=ax0) for r in range(1, 5)]
    caxes = [fig.add_subplot(gs[r, 1]) for r in range(1, 5)]
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis("off")

    # --- Panel 0: Spectra ---
    for j, vz in enumerate(vz_vals):
        if pfaff_slice[j]:
            ax0.axvspan(vz - 0.5 * dvz, vz + 0.5 * dvz, color="#dcdcdc", alpha=0.85, zorder=0)

    ax0.axhline(0.0, color="gray", linestyle="--", linewidth=0.9, zorder=1)
    # Outer states (indices 0, 3) in blue; innermost states (indices 1, 2) in red
    for idx in (0, 3):
        ax0.plot(vz_vals, spec_sorted[:, idx], color="royalblue", linewidth=1.3, zorder=2)
    for idx in (1, 2):
        ax0.plot(vz_vals, spec_sorted[:, idx], color="red", linewidth=1.6, zorder=3)

    ymax = max(float(np.nanmax(np.abs(spec_sorted))) * 1.18, 0.08)
    ax0.set_ylim(-ymax, ymax)
    ax0.set_ylabel("Energy (meV)", fontsize=10)
    ax0.tick_params(labelbottom=False)
    ax0.set_title(
        f"[EXPLORATORY] Spectra and Conductance along {cut_name}\n"
        rf"($\mu = {mu_val:.2f}$ meV, $T = 40$ mK, cutter #0)",
        fontsize=11,
        pad=8,
    )

    # Bottom indicator bars for Stage 2 (orange) and Stage 3 (green)
    y_s2 = -ymax * 0.90
    y_s3 = -ymax * 0.80
    for j, vz in enumerate(vz_vals):
        x0, x1 = vz - 0.46 * dvz, vz + 0.46 * dvz
        if s2_slice[j]:
            ax0.plot([x0, x1], [y_s2, y_s2], color="tab:orange", linewidth=3.5, solid_capstyle="butt", zorder=5)
        if s3_slice[j]:
            ax0.plot([x0, x1], [y_s3, y_s3], color="#2ca02c", linewidth=3.5, solid_capstyle="butt", zorder=5)

    legend_handles = [
        Patch(facecolor="#dcdcdc", edgecolor="none", label="Topological (Pfaffian)"),
        Line2D([0], [0], color="tab:orange", lw=3.5, label="Stage 2 pass"),
        Line2D([0], [0], color="#2ca02c", lw=3.5, label="Stage 3 pass"),
    ]
    ax0.legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=True, framealpha=0.88, ncol=3)

    # --- Panels 1 & 2: Local conductances G_LL, G_RR ---
    vmax_loc = max(float(np.nanpercentile(g_ll, 99)), float(np.nanpercentile(g_rr, 99)), 0.5)
    for ax, cax, arr, lbl in [
        (axes[1], caxes[0], g_ll, r"$G_{LL}$ [$e^2/h$]"),
        (axes[2], caxes[1], g_rr, r"$G_{RR}$ [$e^2/h$]"),
    ]:
        pcm = ax.pcolormesh(vz_vals, bias_uev, arr.T, shading="auto", cmap="viridis", vmin=0.0, vmax=vmax_loc)
        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label(lbl, fontsize=9)
        ax.set_ylabel("Left bias [µV]" if "LL" in lbl else "Right bias [µV]", fontsize=9.5)
        ax.tick_params(labelbottom=False)

    # --- Panels 3 & 4: Antisymmetric nonlocal conductances A(G_RL), A(G_LR) ---
    a_rl = antisymmetrize(g_rl)
    a_lr = antisymmetrize(g_lr)
    vlim_nl = max(float(np.nanpercentile(np.abs(a_rl), 98)), float(np.nanpercentile(np.abs(a_lr), 98)), 0.05)

    for ax, cax, arr, gap_u, lbl, ylab, hide_x in [
        (axes[3], caxes[2], a_rl, gap_rl_uev, r"$A(G_{RL})$ [$e^2/h$]", "Left bias [µV]", True),
        (axes[4], caxes[3], a_lr, gap_lr_uev, r"$A(G_{LR})$ [$e^2/h$]", "Right bias [µV]", False),
    ]:
        pcm = ax.pcolormesh(vz_vals, bias_uev, arr.T, shading="auto", cmap="PuOr_r", vmin=-vlim_nl, vmax=vlim_nl)
        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label(lbl, fontsize=9)
        ax.plot(vz_vals, gap_u, color="k", linewidth=1.2, zorder=4)
        ax.plot(vz_vals, -gap_u, color="k", linewidth=1.2, zorder=4)
        ax.set_ylabel(ylab, fontsize=9.5)
        if hide_x:
            ax.tick_params(labelbottom=False)

    axes[4].set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10)
    axes[4].set_xlim(float(vz_vals[0] - 0.5 * dvz), float(vz_vals[-1] + 0.5 * dvz))

    # Color-coded vertical dashed lines across all 5 panels
    for vz_m, col in vz_markers:
        for ax in axes:
            ax.axvline(vz_m, color=col, linestyle="--", linewidth=1.35, alpha=0.95, zorder=6)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    data_dir = REPO_ROOT / "Outputs" / "Data" / "Tdispfaff_roi"
    out_dir = REPO_ROOT / "exploratory"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(data_dir / "resolved_params.yaml")
    tgp2_cfg = cfg.protocol.tgp.stage2
    T_plot = 40.0
    b_gn = 0.85
    corr_thresh = 0.90
    selected_cutter = 0

    # 1. Spectra & Pfaffian
    ds_spec = xr.open_dataset(data_dir / "spectra_and_topology.nc", engine="h5netcdf")
    pfaff_da = (ds_spec["pfaffian_invariant"] > 0).rename({"zeeman_field": "B", "chemical_potential": "V"})
    spec_da = ds_spec["spectrum"].transpose("chemical_potential", "zeeman_field", "eigenvalue_index")

    # 2. Stage 2 (10 ueV threshold, gb_th = 0.0, matching latest Stage 3 plot)
    ds2_raw = xr.open_dataset(data_dir / "tgp_stage2.nc", engine="h5netcdf").sel(temperature=T_plot, drop=True)
    ds2 = ds2_raw.rename({"zeeman_field": "B", "chemical_potential": "V"}).transpose(
        "cutter_pair_index", "B", "V", "bias"
    )
    ds2_left = ds2.rename({"bias": "left_bias"})
    ds2_right = ds2.rename({"bias": "right_bias"})

    v0 = float(cfg.simulation.disorder.V0)
    gap_factor = float(tgp2_cfg.gap_threshold_factor[v0])
    noise_th = float(tgp2_cfg.noise_threshold)

    ds2_left, ds2_right = tgp.two.extract_gap(
        ds2_left, ds2_right, gap_threshold_factor=gap_factor, noise_threshold=noise_th
    )
    zbp_ds = tgp.two.zbp_dataset_derivative(
        ds2_left,
        ds2_right,
        derivative_threshold=float(tgp2_cfg.derivative_threshold),
        bias_window=float(tgp2_cfg.bias_window),
        average_over_cutter=bool(tgp2_cfg.average_over_cutter),
    )
    tgp.two.set_gap_threshold(zbp_ds, threshold_low=0.010, threshold_high=float(tgp2_cfg.threshold_high))
    zbp_ds = tgp.two.cluster_and_score(
        zbp_ds,
        min_cluster_size=int(tgp2_cfg.min_cluster_size),
        cluster_gap_threshold=float(tgp2_cfg.cluster_gap_threshold),
        cluster_percentage_boundary_threshold=0.0,
        cluster_ncutter_threshold=float(tgp2_cfg.cutter_overlap_th),
    )
    zbp_ds["SI"] = pfaff_da.transpose("B", "V")

    # 3. Stage 3 correlation mask
    ds_cond = xr.open_dataset(data_dir / "conductance_sweeps.nc", engine="h5netcdf")
    gmat = ds_cond["Gmat"].sel(temperature=T_plot, baseline_Gn=b_gn)
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

    corr_pass_vb = (corr_sides["left"] >= corr_thresh) & (corr_sides["right"] >= corr_thresh)

    mu_vals = ds_spec.coords["chemical_potential"].values
    vz_vals = ds_spec.coords["zeeman_field"].values
    bias_uev = ds2_raw.coords["bias"].values * 1e3  # meV -> ueV

    s2_vb = (zbp_ds.sel(cutter_pair_index=selected_cutter).gapped_zbp_cluster > 0).transpose("V", "B").values
    s3_vb = s2_vb & corr_pass_vb
    pfaff_vb = pfaff_da.transpose("V", "B").values

    # Selected chemical potentials across the green island in [2.25, 2.50] meV & color-coded Vz markers
    cuts = [
        ("Cut_1", 2.36),
        ("Cut_2", 2.42),
        ("Cut_3", 2.46),
        ("Cut_4", 2.50),
    ]
    vz_markers = [
        (0.50, "crimson"),
        (0.64, "deepskyblue"),
        (0.74, "limegreen"),
        (0.84, "gold"),
    ]

    ds2_c0 = ds2.sel(cutter_pair_index=selected_cutter)
    gap_rl_all = ds2_left.sel(cutter_pair_index=selected_cutter)["gap"].transpose("V", "B").values * 1e3
    gap_lr_all = ds2_right.sel(cutter_pair_index=selected_cutter)["gap"].transpose("V", "B").values * 1e3

    for cut_name, mu_target in cuts:
        i_mu = int(np.argmin(np.abs(mu_vals - mu_target)))
        mu_actual = float(mu_vals[i_mu])
        plot_single_cut(
            cut_name=cut_name,
            mu_val=mu_actual,
            vz_vals=vz_vals,
            bias_uev=bias_uev,
            spec_slice=spec_da.isel(chemical_potential=i_mu).values,
            pfaff_slice=pfaff_vb[i_mu],
            s2_slice=s2_vb[i_mu],
            s3_slice=s3_vb[i_mu],
            g_ll=ds2_c0["g_ll"].isel(V=i_mu).transpose("B", "bias").values,
            g_rr=ds2_c0["g_rr"].isel(V=i_mu).transpose("B", "bias").values,
            g_rl=ds2_c0["g_rl"].isel(V=i_mu).transpose("B", "bias").values,
            g_lr=ds2_c0["g_lr"].isel(V=i_mu).transpose("B", "bias").values,
            gap_rl_uev=gap_rl_all[i_mu],
            gap_lr_uev=gap_lr_all[i_mu],
            vz_markers=vz_markers,
            out_path=out_dir / f"Tdispfaff_roi_{cut_name}_mu_{mu_actual:.2f}.png",
        )

    # --- Phase map showing all cuts and color-coded Vz points ---
    tgp.plot.set_mpl_rc()
    fig, axs = tgp.plot.paper.plot_stage2_diagram(
        zbp_ds,
        cutter_value=selected_cutter,
        zbp_cluster_numbers="all",
        pct_boundary_shifts={1: [-0.02, -0.24]},
        gap_lim=60.0,
        invariant="SI",
    )

    corr_da = xr.DataArray(corr_pass_vb, coords={"V": mu_vals, "B": vz_vals}, dims=("V", "B"))
    stage3_da = corr_da.where(
        (zbp_ds.sel(cutter_pair_index=selected_cutter).gapped_zbp_cluster > 0).transpose("V", "B") & corr_da,
        np.nan,
    )
    stage3_da.astype(float).plot.pcolormesh(
        ax=axs[0],
        cmap=ListedColormap(["#2ca02c"]),
        add_colorbar=False,
        shading="nearest",
        infer_intervals=False,
        linewidth=0,
        rasterized=True,
        vmin=0.5,
        vmax=1.5,
        zorder=3,
    )
    axs[0].set_title("")

    # Overlay horizontal cut lines & color-coded Vz markers on both panels
    for cut_name, mu_target in cuts:
        i_mu = int(np.argmin(np.abs(mu_vals - mu_target)))
        mu_actual = float(mu_vals[i_mu])
        for ax in axs:
            ax.axhline(mu_actual, color="black", linestyle=":", linewidth=1.1, alpha=0.85, zorder=1050)
            for vz_m, col in vz_markers:
                ax.scatter(
                    [vz_m],
                    [mu_actual],
                    c=col,
                    edgecolors="black",
                    linewidths=0.9,
                    s=55,
                    zorder=1100,
                )
        axs[0].text(
            float(vz_vals[0]) + 0.008,
            mu_actual + 0.025,
            f"{cut_name} ({mu_actual:.2f})",
            fontsize=8.5,
            color="black",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.78),
            zorder=1150,
        )

    cax_s3 = fig.add_axes([-0.12, -0.04, 0.03, 0.11])
    cb_s3 = fig.colorbar(
        plt.cm.ScalarMappable(cmap=ListedColormap(["#2ca02c"]), norm=plt.Normalize(0, 1)),
        cax=cax_s3,
        orientation="vertical",
        ticks=[0.5],
    )
    cb_s3.ax.set_yticklabels([f"Stage 3 Pass\n(Corr ≥ {corr_thresh:.2f})"])

    gb_texts = [t for t in axs[0].texts if "gapless boundary" in t.get_text()]
    for t in gb_texts[1:]:
        t.remove()
    if gb_texts:
        gb_texts[0].set_bbox(dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.80))

    axs[0].yaxis.set_label_text(r"$\mu$ [meV]")
    axs[0].xaxis.set_label_text(r"$V_z$ [meV]")
    axs[1].yaxis.set_label_text("")
    axs[1].xaxis.set_label_text(r"$V_z$ [meV]")

    text_kw = dict(c="k", ha="left", va="bottom", size=14, transform=axs[0].transAxes, zorder=2000)
    axs[0].text(-0.46, 0.90, r"$\mathrm{Tdispfaff\_roi}$,", **text_kw)
    axs[0].text(-0.46, 0.84, f"T = {T_plot:.0f} mK,", **text_kw)
    axs[0].text(-0.46, 0.78, f"cutter pair #{selected_cutter}", **text_kw)

    fig.suptitle(
        "[EXPLORATORY / QUICK-INSPECT — Unregistered] Phase Map with Selected Cuts & Color-Coded $V_z$ Points",
        fontsize=13,
        y=1.04,
    )

    pm_path = out_dir / "Tdispfaff_roi_cuts_phase_map.png"
    fig.savefig(pm_path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved: {pm_path}")

    ds_spec.close()
    ds2_raw.close()
    ds_cond.close()


if __name__ == "__main__":
    main()
