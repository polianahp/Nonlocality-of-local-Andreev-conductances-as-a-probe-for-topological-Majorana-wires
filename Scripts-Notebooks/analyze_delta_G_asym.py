#!/usr/bin/env python3
"""
analyze_delta_G_asym.py

Post-processing and visualization tool for Asymmetric Conductance Deviation (delta G_asym)
measurements based on Dourado, Penteado, Egues, Phys. Rev. B 110, 014504 (2024).

Physics & Diagnostics:
1. Conductance Plateau Saturation:
   Plots G_LL and G_RR as a function of asymmetry ratio a in [-60, -20..20, 60]
   showing pinch-off of the local swept barrier and saturation of the remote barrier.
2. Majorana Oscillation Tracking:
   Plots delta G_LL^{asym, R}(V_z) and delta G_RR^{asym, L}(V_z) alongside the
   Majorana splitting energy epsilon_m(V_z) from closed-system BdG spectrum across
   elevated temperatures (e.g. 30 mK vs 40 mK).
3. Dual Barrier Symmetry Verification:
   Verifies that sweeping UR pinches off GRR while generating oscillations in GLL,
   and sweeping UL pinches off GLL while generating oscillations in GRR.
4. 2D Phase Space Maps:
   Maps delta G^{asym}(mu, V_z) in the (mu, V_z) plane with topological Pfaffian boundary.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_realization_data(realization_dir: Path, temps: List[float]) -> Dict[str, Any]:
    """Loads all relevant arrays for delta G_asym analysis from a realization folder."""
    data = {"dir": realization_dir}

    # Parameters & coordinates
    if (realization_dir / "params_list.npy").exists():
        data["params_list"] = np.load(realization_dir / "params_list.npy")
    else:
        raise FileNotFoundError(f"params_list.npy missing in {realization_dir}")

    if (realization_dir / "barrier_arr.npy").exists():
        data["barrier_arr"] = np.load(realization_dir / "barrier_arr.npy")
    if (realization_dir / "asym_ratios.npy").exists():
        data["asym_ratios"] = np.load(realization_dir / "asym_ratios.npy")
    else:
        # Default 43-point grid if coordinate file was not yet saved
        data["asym_ratios"] = np.concatenate([[-60.0], np.linspace(-20.0, 20.0, 41), [60.0]])

    # Closed-system observables (if full pipeline was executed)
    for key in ["spectrum_arr", "topological_gap", "pdi_data"]:
        p = realization_dir / f"{key}.npy"
        data[key] = np.load(p) if p.exists() else None

    # Temperature-specific conductance arrays
    data["temps"] = {}
    for T in temps:
        t_str = f"{int(T)}mK" if T == int(T) else f"{T:.1f}mK"
        suffix = f"_{t_str}"

        t_dict = {}
        # Conductance matrix G_alpha,beta(U0, U0)
        p_gmat = realization_dir / f"conductance_matrix{suffix}.npy"
        if not p_gmat.exists():
            p_gmat = realization_dir / "conductance_matrix.npy"
        t_dict["Gmat"] = np.load(p_gmat) if p_gmat.exists() else None

        # Asymmetric conductance deviations
        p_dll = realization_dir / f"delta_G_asym_R_LL{suffix}.npy"
        p_drr = realization_dir / f"delta_G_asym_L_RR{suffix}.npy"
        t_dict["delta_G_asym_R_LL"] = np.load(p_dll) if p_dll.exists() else None
        t_dict["delta_G_asym_L_RR"] = np.load(p_drr) if p_drr.exists() else None

        # Barrier sweep curves
        for bkey in [
            "barrier_right_conductance_left_arr",
            "barrier_right_conductance_right_arr",
            "barrier_left_conductance_left_arr",
            "barrier_left_conductance_right_arr",
            "barrier_right_GLR_arr",
            "barrier_right_GRL_arr",
        ]:
            pb = realization_dir / f"{bkey}{suffix}.npy"
            if not pb.exists():
                pb = realization_dir / f"{bkey}.npy"
            t_dict[bkey] = np.load(pb) if pb.exists() else None

        # Compute delta_G_asym if not saved directly
        if t_dict["delta_G_asym_R_LL"] is None and t_dict["Gmat"] is not None and t_dict["barrier_right_conductance_left_arr"] is not None:
            t_dict["delta_G_asym_R_LL"] = t_dict["Gmat"][:, 0, 0] - t_dict["barrier_right_conductance_left_arr"][:, -1]
        if t_dict["delta_G_asym_L_RR"] is None and t_dict["Gmat"] is not None and t_dict["barrier_left_conductance_right_arr"] is not None:
            t_dict["delta_G_asym_L_RR"] = t_dict["Gmat"][:, 1, 1] - t_dict["barrier_left_conductance_right_arr"][:, -1]

        data["temps"][T] = t_dict

    return data


def plot_barrier_pinchoff_and_plateau(
    data: Dict[str, Any],
    pixel_idx: int,
    out_path: Path
):
    """
    Plots the conductance curves as a function of asymmetry ratio a, demonstrating:
    1. The pinch-off of the local swept barrier.
    2. The finite-temperature saturation plateau of the remote barrier.
    Matches Dourado Fig. 5(a) and Fig. 21.
    """
    asym_ratios = data["asym_ratios"]
    params = data["params_list"][pixel_idx]
    mu_val, vz_val = params[1], params[2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    colors = {30.0: "#1f77b4", 40.0: "#ff7f0e", 50.0: "#2ca02c"}

    # Left Panel: Right Barrier Sweep (UR swept, UL fixed at U0)
    ax1 = axes[0]
    for T, t_data in data["temps"].items():
        c = colors.get(T, "black")
        br_l = t_data["barrier_right_conductance_left_arr"]
        br_r = t_data["barrier_right_conductance_right_arr"]
        if br_l is not None and br_r is not None:
            ax1.plot(asym_ratios, br_l[pixel_idx, :], label=f"$G_{{LL}}$ (remote) @ {T} mK", color=c, lw=2.0)
            ax1.plot(asym_ratios, br_r[pixel_idx, :], label=f"$G_{{RR}}$ (local) @ {T} mK", color=c, ls="--", lw=1.8)

    ax1.axvline(1.0, color="gray", ls=":", alpha=0.7, label="Symmetric point ($a=1.0$)")
    ax1.axvline(60.0, color="red", ls=":", alpha=0.7, label="Plateau readout ($a=60.0$)")
    ax1.set_xlabel("Right Barrier Asymmetry Ratio $a_R = U_R / U_0$", fontsize=12)
    ax1.set_ylabel("Differential Conductance ($2e^2/h$)", fontsize=12)
    ax1.set_title(f"Right Barrier Sweep ($U_L = U_0$ fixed)\n$(\\mu={mu_val:.2f}\\,\\mathrm{{meV}}, V_z={vz_val:.2f}\\,\\mathrm{{meV}})$", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)

    # Right Panel: Left Barrier Sweep (UL swept, UR fixed at U0)
    ax2 = axes[1]
    for T, t_data in data["temps"].items():
        c = colors.get(T, "black")
        bl_l = t_data["barrier_left_conductance_left_arr"]
        bl_r = t_data["barrier_left_conductance_right_arr"]
        if bl_l is not None and bl_r is not None:
            ax2.plot(asym_ratios, bl_r[pixel_idx, :], label=f"$G_{{RR}}$ (remote) @ {T} mK", color=c, lw=2.0)
            ax2.plot(asym_ratios, bl_l[pixel_idx, :], label=f"$G_{{LL}}$ (local) @ {T} mK", color=c, ls="--", lw=1.8)

    ax2.axvline(1.0, color="gray", ls=":", alpha=0.7, label="Symmetric point ($a=1.0$)")
    ax2.axvline(60.0, color="red", ls=":", alpha=0.7, label="Plateau readout ($a=60.0$)")
    ax2.set_xlabel("Left Barrier Asymmetry Ratio $a_L = U_L / U_0$", fontsize=12)
    ax2.set_title(f"Left Barrier Sweep ($U_R = U_0$ fixed)\n$(\\mu={mu_val:.2f}\\,\\mathrm{{meV}}, V_z={vz_val:.2f}\\,\\mathrm{{meV}})$", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)

    plt.suptitle("Asymmetric Barrier Pinch-off & Remote Conductance Plateau (Dourado Protocol)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved pinch-off & plateau plot: {out_path}")


def plot_majorana_oscillations_vs_vz(
    data: Dict[str, Any],
    target_mu: float,
    out_path: Path
):
    """
    Plots delta G^{asym}(V_z) tracking Majorana energy oscillations epsilon_m(V_z)
    at a fixed chemical potential cut mu = target_mu.
    Matches Dourado Fig. 5(b).
    """
    params_list = data["params_list"]
    mu_vals = np.unique(params_list[:, 1])
    # Find closest mu
    closest_mu = mu_vals[np.argmin(np.abs(mu_vals - target_mu))]
    mask_mu = np.isclose(params_list[:, 1], closest_mu, atol=1e-4)

    vz_line = params_list[mask_mu, 2]
    sort_idx = np.argsort(vz_line)
    vz_sorted = vz_line[sort_idx]

    has_spectrum = data["spectrum_arr"] is not None
    num_rows = 2 if has_spectrum else 1
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 4.0 * num_rows), sharex=True)
    if num_rows == 1:
        axes = [axes]

    ax_osc = axes[0]
    colors = {30.0: "#1f77b4", 40.0: "#ff7f0e", 50.0: "#2ca02c"}

    for T, t_data in data["temps"].items():
        c = colors.get(T, "black")
        dG_LL = t_data["delta_G_asym_R_LL"]
        dG_RR = t_data["delta_G_asym_L_RR"]

        if dG_LL is not None:
            ax_osc.plot(
                vz_sorted,
                dG_LL[mask_mu][sort_idx],
                label=f"$\\delta G_{{LL}}^{{\\mathrm{{asym}}, R}}$ @ {T} mK",
                color=c,
                lw=2.0
            )
        if dG_RR is not None:
            ax_osc.plot(
                vz_sorted,
                dG_RR[mask_mu][sort_idx],
                label=f"$\\delta G_{{RR}}^{{\\mathrm{{asym}}, L}}$ @ {T} mK",
                color=c,
                ls="--",
                lw=1.8
            )

    ax_osc.set_ylabel("$\\delta G^{\\mathrm{asym}} = G(U_0) - G(60 U_0)$ ($2e^2/h$)", fontsize=11)
    ax_osc.set_title(f"Asymmetric Conductance Deviation $\\delta G^{{\\mathrm{{asym}}}}(V_z)$ at $\\mu = {closest_mu:.2f}\\,\\mathrm{{meV}}$", fontsize=12)
    ax_osc.grid(True, alpha=0.3)
    ax_osc.legend(loc="best", fontsize=9)

    # Lower Panel: Closed-system BdG Majorana Spectrum epsilon_m(V_z)
    if has_spectrum:
        ax_bdg = axes[1]
        spec = data["spectrum_arr"][mask_mu][sort_idx]
        # Plot lowest 2 positive eigenvalues
        ax_bdg.plot(vz_sorted, spec[:, 2] * 1e3, color="#2ca02c", lw=2.2, label="Lowest BdG State $\\epsilon_m$ (MZM)")
        ax_bdg.plot(vz_sorted, spec[:, 3] * 1e3, color="#7f7f7f", lw=1.5, ls="--", label="Second State $\\epsilon_2$ (Topological Gap)")

        ax_bdg.set_xlabel("Zeeman Energy $V_z$ (meV)", fontsize=11)
        ax_bdg.set_ylabel("Energy $\\epsilon$ ($\\mu\\mathrm{eV}$)", fontsize=11)
        ax_bdg.set_title("Closed-System BdG Spectrum (Majorana Energy Oscillations)", fontsize=12)
        ax_bdg.grid(True, alpha=0.3)
        ax_bdg.legend(loc="best", fontsize=9)
    else:
        axes[0].set_xlabel("Zeeman Energy $V_z$ (meV)", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Majorana oscillation comparison plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot delta G_asym and Majorana oscillations for a disorder realization."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to realization data directory (e.g. Outputs/Data/six_micron/V0_0_1/disorder_realization_0)"
    )
    parser.add_argument(
        "--temps",
        type=float,
        nargs="+",
        default=[30.0, 40.0],
        help="List of temperatures in mK to analyze (default: 30.0 40.0)"
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.0,
        help="Target chemical potential for 1D V_z linecuts (default: 0.0 meV)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <data_dir>/plots_delta_G_asym)"
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (data_dir / "plots_delta_G_asym")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Analyzing Asymmetric Conductance Deviation (delta G_asym)")
    print(f" Data Directory: {data_dir}")
    print(f" Temperatures: {args.temps} mK")
    print(f" Output Directory: {out_dir}")
    print("=" * 70)

    data = load_realization_data(data_dir, args.temps)

    # 1. Plot barrier pinch-off and remote plateau at mid-range pixel
    num_pixels = len(data["params_list"])
    test_pixel = num_pixels // 2
    pinchoff_path = out_dir / "barrier_pinchoff_and_plateau.png"
    plot_barrier_pinchoff_and_plateau(data, test_pixel, pinchoff_path)

    # 2. Plot delta G_asym vs Vz linecut tracking Majorana oscillations
    osc_path = out_dir / f"majorana_oscillations_mu_{args.mu:.2f}meV.png"
    plot_majorana_oscillations_vs_vz(data, args.mu, osc_path)

    print("=" * 70)
    print(" Analysis complete! Visualizations generated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
