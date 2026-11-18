"""Tier-0 exploratory reproduction of azure-quantum-tgp Fig. 11 (plot_stage2_diagram) on Tdispfaff_roi."""
from pathlib import Path
import numpy as np
import xarray as xr
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tgp.two
import tgp.plot
import tgp.plot.paper


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "Outputs" / "Data" / "Tdispfaff_roi"
    out_dir = repo_root / "exploratory"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(data_dir / "resolved_params.yaml")
    tgp2_cfg = cfg.protocol.tgp.stage2
    T_plot = 40.0

    ds_spec = xr.open_dataset(data_dir / "spectra_and_topology.nc", engine="h5netcdf")
    pfaff = (ds_spec["pfaffian_invariant"] > 0).rename({"zeeman_field": "B", "chemical_potential": "V"})

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
    zbp_ds["SI"] = pfaff.transpose("B", "V")

    tgp.plot.set_mpl_rc()

    selected_cutter = 0
    # Center of mass of main cluster #1 on cutter 0 is (Vz=0.724, mu=2.425)
    selected_field = 0.72
    selected_plunger = 2.42

    fig, axs = tgp.plot.paper.plot_stage2_diagram(
        zbp_ds,
        cutter_value=selected_cutter,
        zbp_cluster_numbers="all",
        plunger_cut=selected_plunger,
        field_cut=selected_field,
        pct_boundary_shifts={1: [-0.02, -0.24]},
        gap_lim=60.0,
        invariant="SI",
    )

    # Remove boundary text annotations from smaller clusters, keeping only main cluster #1's label
    gb_texts = [t for t in axs[0].texts if "gapless boundary" in t.get_text()]
    for t in gb_texts[1:]:
        t.remove()
    if gb_texts:
        gb_texts[0].set_bbox(dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.80))

    # Replace experimental device axis labels with simulation coordinates (Vz, mu)
    axs[0].yaxis.set_label_text(r"$\mu$ [meV]")
    axs[0].xaxis.set_label_text(r"$V_z$ [meV]")
    axs[1].yaxis.set_label_text("")
    axs[1].xaxis.set_label_text(r"$V_z$ [meV]")

    # Update legend position/labels on the left panel
    text_kw = dict(c="k", ha="left", va="bottom", size=14, transform=axs[0].transAxes, zorder=2000)
    axs[0].text(-0.46, 0.90, r"$\mathrm{Tdispfaff\_roi}$,", **text_kw)
    axs[0].text(-0.46, 0.84, f"T = {T_plot:.0f} mK,", **text_kw)
    axs[0].text(-0.46, 0.78, f"cutter pair #{selected_cutter}", **text_kw)

    fig.suptitle(
        f"[EXPLORATORY / QUICK-INSPECT — Unregistered] TGP Stage 2 Diagram (Fig. 11 style, gapless_boundary_th={float(tgp2_cfg.gapless_boundary_th):.2f})",
        fontsize=13,
        y=1.04,
    )

    out_path = out_dir / f"Tdispfaff_roi_tgp_stage2_fig11_cutter{selected_cutter}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved: {out_path}")

    ds_spec.close()
    ds2_raw.close()


if __name__ == "__main__":
    main()
