"""Tier-0 exploratory plots for TGP Stage 1, Stage 2, and Final Predicted TGP Clusters (T = 40 mK)."""
from pathlib import Path
import numpy as np
import xarray as xr
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tgp.one
import tgp.two


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "Outputs" / "Data" / "Tdispfaff_roi"
    out_dir = repo_root / "exploratory"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(data_dir / "resolved_params.yaml")
    tgp1_cfg = cfg.protocol.tgp.stage1
    tgp2_cfg = cfg.protocol.tgp.stage2

    T_plot = 40.0
    ds_spec = xr.open_dataset(data_dir / "spectra_and_topology.nc", engine="h5netcdf")
    pfaff = ds_spec["pfaffian_invariant"].transpose("chemical_potential", "zeeman_field").values > 0
    mu_vals = ds_spec.coords["chemical_potential"].values
    vz_vals = ds_spec.coords["zeeman_field"].values

    # -------------------------------------------------------------------------
    # 1. TGP Stage 1 (tgp.one.analyze)
    # -------------------------------------------------------------------------
    ds1_raw = xr.open_dataset(data_dir / "tgp_stage1.nc", engine="h5netcdf").sel(temperature=T_plot, drop=True)
    ds1 = ds1_raw.rename({"zeeman_field": "B", "chemical_potential": "V"}).transpose("cutter_pair_index", "B", "V")

    th_s1 = dict(
        set_2w_th={"n_tiles": int(tgp1_cfg.n_tiles), "percentile": int(tgp1_cfg.percentile)},
        set_gapped={"th_2w_p": float(tgp1_cfg.th_2w_p)},
        set_3w_th={"th_3w": float(tgp1_cfg.th_3w)},
        set_3w_tat={"th_3w_tat": float(tgp1_cfg.th_3w_tat)},
    )

    s1_ok = True
    try:
        tgp.one.analyze(ds1, th_s1, force=True)
    except Exception as exc:
        print(f"tgp.one.analyze note: {exc}")
        s1_ok = False

    fig1, axes1 = plt.subplots(1, 3, figsize=(15.5, 4.9), sharey=True, constrained_layout=True)

    # Panel 1: Mean |2w| nonlocal amplitude
    w2_mean = 0.5 * (np.abs(ds1["L_2w_nl"]).mean("cutter_pair_index") + np.abs(ds1["R_2w_nl"]).mean("cutter_pair_index"))
    w2_arr = w2_mean.transpose("V", "B").values
    im1 = axes1[0].pcolormesh(vz_vals, mu_vals, w2_arr, shading="auto", cmap="magma")
    fig1.colorbar(im1, ax=axes1[0], shrink=0.9)
    axes1[0].set_title(r"Mean $|I_{2\omega}^{\mathrm{NL}}|$ (Gapless Signal)", fontsize=10.5)

    # Panel 2: Mean 3w curvature (min of L/R)
    w3_min = np.minimum(ds1["L_3w"].mean("cutter_pair_index").transpose("V", "B").values,
                        ds1["R_3w"].mean("cutter_pair_index").transpose("V", "B").values)
    vlim3 = max(float(np.nanpercentile(np.abs(w3_min), 95)), 100.0)
    im2 = axes1[1].pcolormesh(vz_vals, mu_vals, w3_min, shading="auto", cmap="RdBu_r", vmin=-vlim3, vmax=vlim3)
    fig1.colorbar(im2, ax=axes1[1], shrink=0.9)
    axes1[1].set_title(r"Mean $\min(I_{3\omega}^L, I_{3\omega}^R)$ ($<0 \Rightarrow$ ZBP)", fontsize=10.5)

    # Panel 3: Stage 1 Clusters (ROI1) or fallback 3w ZBP fraction
    if s1_ok and "clusters" in ds1:
        s1_mask = (ds1["clusters"].transpose("V", "B").values > 0)
        n_s1 = int(np.count_nonzero(s1_mask))
        tp_s1 = int(np.count_nonzero(s1_mask & pfaff))
        fp_s1 = n_s1 - tp_s1
        tpr_s1 = 100.0 * tp_s1 / max(int(pfaff.sum()), 1)
        fdr_s1 = 100.0 * fp_s1 / max(n_s1, 1) if n_s1 > 0 else 0.0
        axes1[2].pcolormesh(vz_vals, mu_vals, s1_mask.astype(float), shading="auto",
                            cmap=ListedColormap(["#f0f2f5", "#ff7f0e"]), vmin=0.0, vmax=1.0)
        axes1[2].set_title(f"Stage 1 Clusters ({n_s1} px | TPR {tpr_s1:.1f}%, FDR {fdr_s1:.1f}%)", fontsize=10)
        print(f"Stage 1 Clusters: {n_s1}/2500 px | TP={tp_s1}, FP={fp_s1}, TPR={tpr_s1:.1f}%, FDR={fdr_s1:.1f}%")
    else:
        zbp_frac_s1 = (0.5 * ((ds1["L_3w"] < -float(tgp1_cfg.th_3w)).mean("cutter_pair_index") +
                              (ds1["R_3w"] < -float(tgp1_cfg.th_3w)).mean("cutter_pair_index"))).transpose("V", "B").values
        im3 = axes1[2].pcolormesh(vz_vals, mu_vals, zbp_frac_s1, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        fig1.colorbar(im3, ax=axes1[2], shrink=0.9)
        axes1[2].set_title(r"Stage 1 $3\omega$ ZBP Fraction (no clusters passed)", fontsize=10)

    for ax in axes1:
        if np.any(pfaff):
            ax.contour(vz_vals, mu_vals, pfaff.astype(float), levels=[0.5], colors="lime", linewidths=1.2, linestyles="--")
        ax.set_xlabel(r"$V_z$ (meV)", fontsize=10)
    axes1[0].set_ylabel(r"$\mu$ (meV)", fontsize=10)
    fig1.suptitle(
        f"[EXPLORATORY / QUICK-INSPECT — Unregistered] TGP Stage 1 (T={T_plot:.0f} mK, green dashed = Pfaffian)",
        fontsize=11,
    )
    p1 = out_dir / "Tdispfaff_roi_tgp_stage1.png"
    fig1.savefig(p1, dpi=180)
    plt.close(fig1)
    print(f"Saved: {p1}")

    # -------------------------------------------------------------------------
    # 2. TGP Stage 2 & Final Predicted Clusters (tgp.two)
    # -------------------------------------------------------------------------
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
    tgp.two.set_gap_threshold(
        zbp_ds,
        threshold_low=float(tgp2_cfg.threshold_low),
        threshold_high=float(tgp2_cfg.threshold_high),
    )
    zbp_ds = tgp.two.cluster_and_score(
        zbp_ds,
        min_cluster_size=int(tgp2_cfg.min_cluster_size),
        cluster_gap_threshold=float(tgp2_cfg.cluster_gap_threshold),
        cluster_percentage_boundary_threshold=float(tgp2_cfg.gapless_boundary_th),
        cluster_ncutter_threshold=float(tgp2_cfg.cutter_overlap_th),
    )

    gap_mean_uev = zbp_ds["gap"].mean("cutter_pair_index").transpose("V", "B").values * 1e3
    zbp_frac = zbp_ds["zbp"].mean("cutter_pair_index").transpose("V", "B").values
    gapped_zbp_frac = zbp_ds["gapped_zbp"].mean("cutter_pair_index").transpose("V", "B").values

    # Final predicted TGP clusters: roi2 (passed OPTICS + boundary + ncutter criteria) & raw gapped_zbp_cluster union
    roi2_mask = zbp_ds["roi2"].transpose("V", "B").values > 0
    if "gapped_zbp_cluster" in zbp_ds:
        raw_cluster_mask = (zbp_ds["gapped_zbp_cluster"] > 0).any("cutter_pair_index").transpose("V", "B").values
    else:
        raw_cluster_mask = np.zeros_like(pfaff, dtype=bool)

    n_roi2 = int(np.count_nonzero(roi2_mask))
    tp_roi2 = int(np.count_nonzero(roi2_mask & pfaff))
    fp_roi2 = n_roi2 - tp_roi2
    tpr_roi2 = 100.0 * tp_roi2 / max(int(pfaff.sum()), 1)
    fdr_roi2 = 100.0 * fp_roi2 / max(n_roi2, 1) if n_roi2 > 0 else 0.0
    print(f"Stage 2 Final ROI2 Clusters: {n_roi2}/2500 px | TP={tp_roi2}, FP={fp_roi2}, TPR={tpr_roi2:.1f}%, FDR={fdr_roi2:.1f}%")
    print(f"Stage 2 Raw gapped_zbp_cluster union: {int(raw_cluster_mask.sum())}/2500 px")

    fig2, axes2 = plt.subplots(1, 3, figsize=(15.5, 4.9), sharey=True, constrained_layout=True)

    im_g = axes2[0].pcolormesh(vz_vals, mu_vals, gap_mean_uev, shading="auto", cmap="inferno", vmin=0.0, vmax=80.0)
    fig2.colorbar(im_g, ax=axes2[0], shrink=0.9, label=r"$\Delta_{\mathrm{trans}}$ ($\mu$eV)")
    axes2[0].set_title(r"Stage 2 Mean Nonlocal Gap ($\mu$eV)", fontsize=10.5)

    im_z = axes2[1].pcolormesh(vz_vals, mu_vals, zbp_frac, shading="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    fig2.colorbar(im_z, ax=axes2[1], shrink=0.9, label="Cutter fraction")
    axes2[1].set_title("Stage 2 Dual-End ZBP Fraction", fontsize=10.5)

    im_gz = axes2[2].pcolormesh(vz_vals, mu_vals, gapped_zbp_frac, shading="auto", cmap="Purples", vmin=0.0, vmax=1.0)
    fig2.colorbar(im_gz, ax=axes2[2], shrink=0.9, label="Cutter fraction")
    axes2[2].set_title("Stage 2 Gapped-ZBP Fraction", fontsize=10.5)

    for ax in axes2:
        if np.any(pfaff):
            ax.contour(vz_vals, mu_vals, pfaff.astype(float), levels=[0.5], colors="lime", linewidths=1.2, linestyles="--")
        ax.set_xlabel(r"$V_z$ (meV)", fontsize=10)
    axes2[0].set_ylabel(r"$\mu$ (meV)", fontsize=10)
    fig2.suptitle(
        f"[EXPLORATORY / QUICK-INSPECT — Unregistered] TGP Stage 2 Maps (T={T_plot:.0f} mK, green dashed = Pfaffian)",
        fontsize=11,
    )
    p2 = out_dir / "Tdispfaff_roi_tgp_stage2.png"
    fig2.savefig(p2, dpi=180)
    plt.close(fig2)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------------------
    # 3. Final Predicted TGP Clusters (ROI2 + cluster_sets overlay)
    # -------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=(12.5, 4.9), sharey=True, constrained_layout=True)

    axes3[0].pcolormesh(vz_vals, mu_vals, raw_cluster_mask.astype(float), shading="auto",
                        cmap=ListedColormap(["#f0f2f5", "#9467bd"]), vmin=0.0, vmax=1.0)
    axes3[0].set_title(f"Stage 2 All Scored Clusters ({int(raw_cluster_mask.sum())}/2500 px)", fontsize=10.5)

    axes3[1].pcolormesh(vz_vals, mu_vals, roi2_mask.astype(float), shading="auto",
                        cmap=ListedColormap(["#f0f2f5", "#2ca02c"]), vmin=0.0, vmax=1.0)
    axes3[1].set_title(f"Final TGP ROI2 Prediction ({n_roi2}/2500 px | TPR {tpr_roi2:.1f}%, FDR {fdr_roi2:.1f}%)", fontsize=10)

    for ax in axes3:
        if np.any(pfaff):
            ax.contour(vz_vals, mu_vals, pfaff.astype(float), levels=[0.5], colors="magenta", linewidths=1.3, linestyles="--")
        ax.set_xlabel(r"$V_z$ (meV)", fontsize=10)
    axes3[0].set_ylabel(r"$\mu$ (meV)", fontsize=10)
    fig3.suptitle(
        f"[EXPLORATORY / QUICK-INSPECT — Unregistered] Final TGP Predicted Clusters (T={T_plot:.0f} mK, magenta dashed = Pfaffian)",
        fontsize=11,
    )
    p3 = out_dir / "Tdispfaff_roi_tgp_final_clusters.png"
    fig3.savefig(p3, dpi=180)
    plt.close(fig3)
    print(f"Saved: {p3}")

    ds_spec.close()
    ds1_raw.close()
    ds2_raw.close()


if __name__ == "__main__":
    main()
