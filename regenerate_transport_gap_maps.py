#!/usr/bin/env python3
"""
Regenerate transport gap phase map plots across all 7 datasets without the cluttered blue outline.
Preserves the clean heatmap and the cyan contour for the Pfaffian topological phase boundary.
"""
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tgp
from cut_analysis import PathConfigs

DATASETS = [
    ("Tdis_pfaff5_V0_0_0", 0.001),
    ("Tdis_pfaff5_V0_0_1", 0.01),
    ("Tdis_pfaff5_V0_0_378", 0.01),
    ("Tdis_pfaff5_V0_0_645", 0.05),
    ("Tdis_pfaff5_V0_0_872", 0.05),
    ("Tdis_pfaff5_V0_0_91", 0.05),
    ("Tdis_pfaff5", 0.05),
]

BIAS_GAP_THRESHOLDS_UEV = [2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0]

def main():
    for ds_name, gap_th_factor in DATASETS:
        print(f"Processing transport gap plots for {ds_name}...")
        data_dir = PathConfigs.DATA / ds_name
        bias_sweep_dir = PathConfigs.PLOTS / f"{ds_name}_Plots" / "bias_threshold_sweep"
        bias_sweep_dir.mkdir(parents=True, exist_ok=True)

        tprep = xr.load_dataset(data_dir / "tprep.nc")
        B_vals = tprep.coords["B"].values
        V_vals = tprep.coords["V"].values
        pfaff_vb = (tprep["L_SI"].values[0] == 1).T if 'L_SI' in tprep else None

        t_l = tprep.rename({"bias": "left_bias"})
        t_r = tprep.rename({"bias": "right_bias"})
        t_l, t_r = tgp.two.extract_gap(t_l, t_r, gap_threshold_factor=gap_th_factor, noise_threshold=1e-4)

        gap_left_mean = t_l.gap.mean(dim='cutter_pair_index').values if 'cutter_pair_index' in t_l.gap.dims else t_l.gap.values
        gap_right_mean = t_r.gap.mean(dim='cutter_pair_index').values if 'cutter_pair_index' in t_r.gap.dims else t_r.gap.values
        gap_2d = np.minimum(gap_left_mean, gap_right_mean)

        for th_u in BIAS_GAP_THRESHOLDS_UEV:
            out_tg = bias_sweep_dir / f"transport_gap_phase_map_{th_u:.1f}uev.png"
            fig_tg, ax_tg = plt.subplots(figsize=(6.5, 5.2))
            im_tg = ax_tg.pcolormesh(B_vals, V_vals, gap_2d, cmap='hot_r', vmin=0.0, vmax=0.05, shading='nearest', rasterized=True)
            if pfaff_vb is not None:
                try:
                    ax_tg.contour(B_vals, V_vals, pfaff_vb.astype(float), levels=[0.5], colors=['cyan'], linewidths=[1.2])
                except Exception:
                    pass
            cb_tg = fig_tg.colorbar(im_tg, ax=ax_tg)
            cb_tg.set_label("Transport Gap (meV)", fontsize=10)
            ax_tg.set_xlabel(r"Zeeman Field $V_z$ (meV)", fontsize=10)
            ax_tg.set_ylabel(r"Chemical Potential $\mu$ (meV)", fontsize=10)
            ax_tg.set_title(r"Transport Gap Phase Map $\Delta_{\mathrm{transport}}$" + f"\n(Cutoff $\Delta_{{\mathrm{{low}}}} = {th_u:.1f}\ \mu\mathrm{{eV}}$, Cyan: Pfaffian $\mathcal{{Q}}=-1$)", fontsize=11)
            fig_tg.tight_layout()
            fig_tg.savefig(out_tg, dpi=150)
            plt.close(fig_tg)

    print("Finished regenerating all transport gap phase map plots without blue outline!")

if __name__ == "__main__":
    main()
