"""Tier-0 exploratory companion plots: Majorana wavefunctions (M1, M2) at the 4 Vz points for each cut."""
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

from src import helpers as hp


def main() -> None:
    data_dir = REPO_ROOT / "Outputs" / "Data" / "Tdispfaff_roi"
    out_dir = REPO_ROOT / "exploratory"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(data_dir / "resolved_params.yaml")
    num_evals = int(cfg.simulation.resolution.num_eigenvalues)

    st = np.load(data_dir / "state.npz")
    t_hop = float(st["t"])
    gamma = float(st["gamma"])
    delta0 = float(st["Delta0"])
    alpha = float(st["alpha"])
    Ls = int(st["Ls"])
    Vdisx = np.asarray(st["Vdisx"], dtype=float)

    ds_spec = xr.open_dataset(data_dir / "spectra_and_topology.nc", engine="h5netcdf")
    mu_vals = ds_spec.coords["chemical_potential"].values
    vz_vals = ds_spec.coords["zeeman_field"].values
    pfaff_da = ds_spec["pfaffian_invariant"] > 0

    cuts = [
        ("Cut_1", 2.36),
        ("Cut_2", 2.42),
        ("Cut_3", 2.46),
        ("Cut_4", 2.50),
    ]
    vz_markers = [
        (0.50, "crimson", "A"),
        (0.64, "deepskyblue", "B"),
        (0.74, "limegreen", "C"),
        (0.84, "gold", "D"),
    ]
    sites = np.arange(Ls)

    for cut_idx, (cut_name, mu_target) in enumerate(cuts, start=1):
        i_mu = int(np.argmin(np.abs(mu_vals - mu_target)))
        mu_actual = float(mu_vals[i_mu])

        fig, axes = plt.subplots(4, 1, figsize=(7.6, 8.6), sharex=True, constrained_layout=True)

        for ax, (vz_target, col, pt_letter) in zip(axes, vz_markers):
            j_vz = int(np.argmin(np.abs(vz_vals - vz_target)))
            vz_actual = float(vz_vals[j_vz])
            is_topo = bool(pfaff_da.isel(chemical_potential=i_mu, zeeman_field=j_vz).values)

            syst_closed = hp.build_system_closed(
                t=t_hop,
                mu=mu_actual,
                gamma=gamma,
                Delta0=delta0,
                V_z=vz_actual,
                alpha=alpha,
                Ls=Ls,
                Vdisx=Vdisx,
            )
            evals, evecs = hp.solve_ham(syst_closed, k=num_evals)
            rho_m1, rho_m2, _ = hp.get_psiM_density(evals, evecs)

            ax.plot(sites, rho_m1, color="cyan", linewidth=1.6, label="Majorana Left (M1)")
            ax.plot(sites, rho_m2, color="orange", linewidth=1.6, label="Majorana Right (M2)")
            ax.set_ylabel("Probability Density", fontsize=9.5)
            ax.set_xlim(-8, Ls + 8)
            ax.set_ylim(bottom=0.0)
            topo_str = "Topological" if is_topo else "Trivial"
            ax.set_title(
                rf"Wavefunctions for Point {cut_idx}{pt_letter} ($V_z = {vz_actual:.2f}$ meV, {topo_str})",
                fontsize=10,
            )
            ax.legend(loc="upper center", fontsize=8.5, frameon=True, ncol=2)

            # Color-coded border matching the vertical line on the cut plot
            for spine in ax.spines.values():
                spine.set_edgecolor(col)
                spine.set_linewidth(2.6)

        axes[-1].set_xlabel("Site Index", fontsize=10)
        fig.suptitle(
            rf"[EXPLORATORY] Majorana Wavefunction Localization — {cut_name} ($\mu = {mu_actual:.2f}$ meV)",
            fontsize=11.5,
        )

        out_path = out_dir / f"Tdispfaff_roi_{cut_name}_mu_{mu_actual:.2f}_wavefunctions.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_path}")

    ds_spec.close()


if __name__ == "__main__":
    main()
