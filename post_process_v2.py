import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add Microsoft TGP pipeline to path
tgp_path = "/home/pseudonym/Documents/Code/azure-quantum-tgp"
tgp_notebooks = "/home/pseudonym/Documents/Code/azure-quantum-tgp/notebooks"
if tgp_path not in sys.path:
    sys.path.append(tgp_path)
if tgp_notebooks not in sys.path:
    sys.path.append(tgp_notebooks)

import tgp.prepare
from yield_analysis import analyze_2

from tgp_adapter import TGPAdapter

def plot_phase_diagram(zbp_ds, save_dir):
    """
    Plots the topological invariant overlayed with the protocol passing regions.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract Topological Invariant (from our Pfaffian, which was copied to SI)
    # L_SI and R_SI are identical in our adapter. Negative means topological.
    # SI has shape (B, V).
    SI = zbp_ds.SI.values
    is_topological = SI < 0

    B_coords = zbp_ds.coords["B"].values
    V_coords = zbp_ds.coords["V"].values

    # Plot Topological region
    V_mesh, B_mesh = np.meshgrid(V_coords, B_coords)
    ax.contourf(V_mesh, B_mesh, is_topological, levels=[0.5, 1.5], colors=['#a1c9f4'], alpha=0.6)
    
    # We want to highlight the protocol passing region.
    # zbp_ds.roi2 contains integer IDs of the zero-bias peak clusters.
    if "roi2" in zbp_ds.data_vars and zbp_ds.roi2.max() > 0:
        # zbp_ds.roi2 is flattened across cutters, shape (B, V)
        roi2_mask = zbp_ds.roi2.values > 0
        ax.contour(V_mesh, B_mesh, roi2_mask, levels=[0.5], colors=['red'], linewidths=2)
        ax.plot([], [], color='red', linewidth=2, label="Protocol ROI2 Region")

    ax.plot([], [], color='#a1c9f4', alpha=0.6, linewidth=8, label="Pfaffian Topological Phase")

    ax.set_xlabel("Chemical Potential (meV)")
    ax.set_ylabel("Zeeman Field (meV)")
    ax.set_title("TGP Protocol vs Topological Phase")
    ax.legend()
    
    plt.tight_layout()
    save_path = Path(save_dir) / "phase_diagram.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved phase diagram to: {save_path}")
    # plt.show()

def main():
    data_dir = "/home/pseudonym/Documents/Code/NonlocalProtocol/Data/TGP_Test_Run"
    
    print(f"Loading TGP Adapter for data in: {data_dir}")
    adapter = TGPAdapter(data_dir)
    ds = adapter.to_xarray()

    print("Running Microsoft TGP Pipeline: prepare_sim...")
    # T_mK=0.0 means no thermal broadening is applied, matching the raw data.
    # Use T_mK=40.0 if you want to mirror their 40mK broadening exactly.
    ds_prepared = tgp.prepare.prepare_sim(ds, T_mK=0.0)

    print("Running Microsoft TGP Pipeline: analyze_2...")
    result = analyze_2(ds_prepared, T_mK=0.0, return_datasets=True)
    
    print("Analysis Results:")
    print("-----------------")
    print("ROI2 Stats (Passed/Failed):", result.get("roi2_stats"))
    print("Passing List for each ROI2:", result.get("passing_list"))
    
    zbp_ds = result["zbp_ds"]
    
    print("\nGenerating Phase Diagram...")
    plot_phase_diagram(zbp_ds, data_dir)

if __name__ == "__main__":
    main()
