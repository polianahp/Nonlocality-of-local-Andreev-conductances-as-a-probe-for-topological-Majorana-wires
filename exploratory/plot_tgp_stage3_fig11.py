"""Tier-0 exploratory Stage 3 (Stage 2 + calc_correlation >= 0.85) reproduction in Fig. 11 style."""
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
import tgp.two
import tgp.common
import tgp.plot
import tgp.plot.paper

from exploratory.plot_quick_correlations import normalize_by_symmetric, compute_calc_correlation_map


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "Outputs" / "Data" / "Tdispfaff_roi"
    out_dir = repo_root / "exploratory"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(data_dir / "resolved_params.yaml")
    tgp2_cfg = cfg.protocol.tgp.stage2
    T_plot = 40.0
    b_gn = 0.85
    corr_thresh = 0.90

    # 1. Ground truth Pfaffian
    ds_spec = xr.open_dataset(data_dir / "spectra_and_topology.nc", engine="h5netcdf")
    pfaff = (ds_spec["pfaffian_invariant"] > 0).rename({"zeeman_field": "B", "chemical_potential": "V"})

    # 2. Stage 2 (tgp.two)
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
    zbp_ds_base = tgp.two.zbp_dataset_derivative(
        ds2_left,
        ds2_right,
        derivative_threshold=float(tgp2_cfg.derivative_threshold),
        bias_window=float(tgp2_cfg.bias_window),
        average_over_cutter=bool(tgp2_cfg.average_over_cutter),
    )

    # 3. Stage 3 correlation metric (calc_correlation >= corr_thresh on both swept sides)
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
    corr_da = xr.DataArray(
        corr_pass_vb,
        coords={"V": zbp_ds_base.coords["V"].values, "B": zbp_ds_base.coords["B"].values},
        dims=("V", "B"),
    )

    selected_cutter = 0
    selected_field = 0.72
    selected_plunger = 2.42
    pfaff_vb = pfaff.transpose("V", "B").values

    tgp.plot.set_mpl_rc()

    runs = [
        (float(tgp2_cfg.threshold_low), float(tgp2_cfg.gapless_boundary_th), "", f"gap_th={float(tgp2_cfg.threshold_low)*1e3:.0f}ueV, gb_th={float(tgp2_cfg.gapless_boundary_th):.2f}"),
        (float(tgp2_cfg.threshold_low), 0.0, "_no_gb", f"gap_th={float(tgp2_cfg.threshold_low)*1e3:.0f}ueV, gb_th=0.0"),
        (0.010, 0.0, "_no_gb_gap10uev", "gap_th=10ueV, gb_th=0.0"),
    ]

    for gap_th_low, gb_th, suffix, run_label in runs:
        zds_run = zbp_ds_base.copy(deep=True)
        tgp.two.set_gap_threshold(
            zds_run,
            threshold_low=gap_th_low,
            threshold_high=float(tgp2_cfg.threshold_high),
        )
        zds_run = tgp.two.cluster_and_score(
            zds_run,
            min_cluster_size=int(tgp2_cfg.min_cluster_size),
            cluster_gap_threshold=float(tgp2_cfg.cluster_gap_threshold),
            cluster_percentage_boundary_threshold=gb_th,
            cluster_ncutter_threshold=float(tgp2_cfg.cutter_overlap_th),
        )
        zds_run["SI"] = pfaff.transpose("B", "V")

        stage2_cutter_mask = (zds_run.sel(cutter_pair_index=selected_cutter).gapped_zbp_cluster > 0).transpose("V", "B").values
        stage3_cutter_mask = stage2_cutter_mask & corr_pass_vb

        n_s2 = int(stage2_cutter_mask.sum())
        n_s3 = int(stage3_cutter_mask.sum())
        tp_s3 = int((stage3_cutter_mask & pfaff_vb).sum())
        fp_s3 = n_s3 - tp_s3
        tpr_s3 = 100.0 * tp_s3 / max(int(pfaff_vb.sum()), 1)
        fdr_s3 = 100.0 * fp_s3 / max(n_s3, 1) if n_s3 > 0 else 0.0
        print(f"[{run_label}] Cutter #{selected_cutter} Stage 2: {n_s2} px | Stage 3 (corr>={corr_thresh}): {n_s3} px | TP={tp_s3}, FP={fp_s3}, TPR={tpr_s3:.1f}%, FDR={fdr_s3:.1f}%")

        roi2_mask = (zds_run["roi2"] > 0).transpose("V", "B").values
        roi2_s3 = roi2_mask & corr_pass_vb
        tp_r3 = int((roi2_s3 & pfaff_vb).sum())
        fp_r3 = int(roi2_s3.sum()) - tp_r3
        print(
            f"[{run_label}] Full ROI2 Stage 2: {int(roi2_mask.sum())} px | Stage 3: {int(roi2_s3.sum())} px | "
            f"TP={tp_r3}, FP={fp_r3}, TPR={100*tp_r3/max(int(pfaff_vb.sum()),1):.1f}%, FDR={100*fp_r3/max(int(roi2_s3.sum()),1):.1f}%"
        )

        fig, axs = tgp.plot.paper.plot_stage2_diagram(
            zds_run,
            cutter_value=selected_cutter,
            zbp_cluster_numbers="all",
            plunger_cut=selected_plunger,
            field_cut=selected_field,
            pct_boundary_shifts={1: [-0.02, -0.24]},
            gap_lim=60.0,
            invariant="SI",
        )

        # Overlay Stage 3 passing regions in green on panel (a)
        stage3_da = corr_da.where(
            (zds_run.sel(cutter_pair_index=selected_cutter).gapped_zbp_cluster > 0).transpose("V", "B") & corr_da,
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

        # Add Stage 3 green swatch to left-side legend column
        cax_s3 = fig.add_axes([-0.12, -0.04, 0.03, 0.11])
        cb_s3 = fig.colorbar(
            plt.cm.ScalarMappable(cmap=ListedColormap(["#2ca02c"]), norm=plt.Normalize(0, 1)),
            cax=cax_s3,
            orientation="vertical",
            ticks=[0.5],
        )
        cb_s3.ax.set_yticklabels([f"Stage 3 Pass\n(Corr ≥ {corr_thresh:.2f})"])

        # Keep only main cluster #1's gapless boundary label
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
            f"[EXPLORATORY / QUICK-INSPECT — Unregistered] TGP Stage 3 (Stage 2 [{run_label}] + Corr ≥ {corr_thresh:.2f} at G_N={b_gn:.2f})",
            fontsize=13,
            y=1.04,
        )

        out_path = out_dir / f"Tdispfaff_roi_tgp_stage3_fig11_cutter{selected_cutter}{suffix}.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        print(f"Saved: {out_path}")

    ds_spec.close()
    ds2_raw.close()
    ds_cond.close()


if __name__ == "__main__":
    main()
