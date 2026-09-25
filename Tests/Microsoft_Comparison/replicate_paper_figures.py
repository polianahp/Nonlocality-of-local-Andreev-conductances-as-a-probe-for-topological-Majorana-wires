import json
import io
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.constants import physical_constants

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import tgp
import tgp.one
import tgp.two
import tgp.common
import tgp.plot.paper

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.parameter_handler import ConfigManager

# Configuration based on paper-figures.ipynb
DATASETS = [
    {
        "name": "deviceA1_stage2",
        "type": "experimental",
        "selected_cutter": 0,
        "zbp_cluster_numbers": [1],
        "selected_plunger": -1.17175,
        "selected_field": 1.66,
        "pct_boundary_shifts": {1: [0.28, -0.001]},
        "device_info_key": "deviceA1",
        "override_params": {"gap_threshold_high": 70e-3},
        "correct_kwargs": {
            "lockin_left": "lockin_1", "lockin_right": "lockin_2",
            "fridge_parameters": "deviceA1.yaml",
            "drop_indices": [0, 1, 2], "max_bias_index": -2, "norm": 1e3,
            "phase_shift_left": -3.3, "phase_shift_right": -5.57
        }
    },
    {
        "name": "deviceA2_stage2",
        "type": "experimental",
        "selected_cutter": 1,
        "zbp_cluster_numbers": [1],
        "selected_plunger": -1.4045,
        "selected_field": 1.14666667,
        "pct_boundary_shifts": {1: [0.24, -0.0016]},
        "device_info_key": "deviceA2",
        "override_params": {"gap_threshold_high": 70e-3},
        "correct_kwargs": {
            "lockin_left": "mfli_5510", "lockin_right": "mfli_5602",
            "fridge_parameters": "deviceA2.yaml",
            "drop_indices": [0, 1, 2], "max_bias_index": -2, "norm": 1e3
        }
    },
    {
        "name": "deviceA3_stage2",
        "type": "experimental",
        "selected_cutter": 1,
        "zbp_cluster_numbers": [1],
        "selected_plunger": -1.4083,
        "selected_field": 1.1,
        "pct_boundary_shifts": {1: [0.34, -0.0002]},
        "device_info_key": "deviceA3",
        "override_params": {"gap_threshold_high": 70e-3},
        "correct_kwargs": {
            "lockin_left": "mfli_5510", "lockin_right": "mfli_5602",
            "fridge_parameters": "deviceA2.yaml",
            "drop_indices": [], "max_bias_index": -1, "norm": 1e3
        }
    },
    {
        "name": "deviceB_stage2",
        "type": "experimental",
        "selected_cutter": 1,
        "zbp_cluster_numbers": [1],
        "selected_plunger": -1.15775,
        "selected_field": 2.12,
        "pct_boundary_shifts": {1: [0.33, 0.00033]},
        "device_info_key": "deviceB",
        "override_params": {"gap_threshold_high": 70e-3},
        "correct_kwargs": {
            "lockin_left": "lockin_1", "lockin_right": "lockin_2",
            "fridge_parameters": "deviceB.yaml",
            "drop_indices": [0, 1, 2, 3], "max_bias_index": -4, "norm": 1e3,
            "phase_shift_left": -3.78, "phase_shift_right": -6.89
        }
    },
    {
        "name": "deviceC_stage2",
        "type": "experimental",
        "selected_cutter": 0,
        "zbp_cluster_numbers": [1],
        "selected_plunger": -2.3655,
        "selected_field": 0.98,
        "pct_boundary_shifts": {1: [0.20, 0.0001]},
        "device_info_key": "deviceC",
        "override_params": {"gap_threshold_high": 0.1},
        "correct_kwargs": {
            "lockin_left": "mfli_5583", "lockin_right": "mfli_5591",
            "fridge_parameters": "deviceC.yaml",
            "drop_indices": ([0, 1, 2, 3, 4, 5], []), "max_bias_index": (-4, -1), "norm": 1e3
        }
    },
    {
        "name": "deviceD_stage2",
        "type": "experimental",
        "selected_cutter": 0,
        "zbp_cluster_numbers": [1, 2],
        "selected_plunger": -2.721,
        "selected_field": 1.82,
        "pct_boundary_shifts": {1: [-0.3, 0.0012], 2: [0.32, -0.0012]},
        "device_info_key": "deviceD",
        "override_params": {"gap_threshold_high": 70e-3},
        "correct_kwargs": {
            "lockin_left": "mfli_4909", "lockin_right": "mfli_5654",
            "fridge_parameters": "deviceD.yaml",
            "drop_indices": [0, 1], "max_bias_index": -4, "norm": 1e3
        }
    },
    {
        "name": "deviceE_stage2",
        "type": "experimental",
        "selected_cutter": 1,
        "zbp_cluster_numbers": [],
        "selected_plunger": -1.3893,
        "selected_field": None,
        "pct_boundary_shifts": {1: [0.34, -0.001], 2: [0.34, -0.001]},
        "device_info_key": "deviceE",
        "override_params": {"gap_threshold_high": 70e-3},
        "correct_kwargs": {
            "lockin_left": "mfli_5591", "lockin_right": "mfli_5583",
            "fridge_parameters": "deviceE.yaml",
            "drop_indices": [0], "max_bias_index": -1, "norm": 1e3
        }
    },
    {
        "name": "simulated_DLG_epsilon_R1_stage2",
        "type": "simulated",
        "selected_cutter": 4,
        "zbp_cluster_numbers": [1],
        "selected_plunger": -0.7205,
        "selected_field": None,
        "pct_boundary_shifts": {1: [-0.55, 0.009]},
        "invariant": "SI",
        "device_info_key": None,
        "override_params": {"gap_threshold_high": 70e-3}
    },
    {
        "name": "simulated_SLG_beta_R1_stage2",
        "type": "simulated",
        "selected_cutter": 2,
        "zbp_cluster_numbers": [1],
        "selected_plunger": [-1.37275, -1.36575], # plunger lim?
        "selected_field": None,
        "pct_boundary_shifts": {1: [-0.25, 0.0041]},
        "invariant": "SI",
        "plunger_lim": [-1.385, -1.36],
        "device_info_key": None,
        "override_params": {"gap_threshold_high": 70e-3}
    },
    {
        "name": "simulated_SLG_beta_R2_stage2",
        "type": "simulated",
        "selected_cutter": 2,
        "zbp_cluster_numbers": [],
        "selected_plunger": -1.36,
        "selected_field": None,
        "pct_boundary_shifts": {1: [-0.18, 0.0041]},
        "device_info_key": None,
        "override_params": {"gap_threshold_high": 70e-3}
    }
]

DEVICE_INFO = dict(
    deviceA1=dict(lever_arm=85, g_factor=5.6),
    deviceA2=dict(lever_arm=85, g_factor=6.4),
    deviceA3=dict(lever_arm=85, g_factor=6.4),
    deviceB=dict(lever_arm=78, g_factor=3.7),
    deviceC=dict(lever_arm=79, g_factor=4.4),
    deviceD=dict(lever_arm=83, g_factor=6.8),
    deviceE=dict(lever_arm=86, g_factor=4.4),
)

def run_user_protocol(ds_left, ds_right, ds_meta, config):
    # 1. extract_gap
    # Use paper defaults instead of config.yaml where overridden
    gap_threshold_factor = 0.05
    ds_left, ds_right = tgp.two.extract_gap(
        ds_left, ds_right, 
        gap_threshold_factor=gap_threshold_factor
    )
    
    # 2. zbp_dataset_derivative
    zbp_average_over_cutter = True
    zbp_probability_threshold = 0.6
    zbp_ds = tgp.two.zbp_dataset_derivative(
        ds_left, ds_right, 
        average_over_cutter=zbp_average_over_cutter,
        zbp_probability_threshold=zbp_probability_threshold
    )
    
    # Paper explicitly sets zbp gap on zbp_ds!
    tgp.two.set_zbp_gap(zbp_ds, ds_left, ds_right)
    
    # 3. set_gap_threshold
    gap_threshold_high = ds_meta["override_params"]["gap_threshold_high"]
    tgp.two.set_gap_threshold(zbp_ds, threshold_high=gap_threshold_high)
    
    # 4. cluster_and_score
    zbp_ds = tgp.two.cluster_and_score(
        zbp_ds,
        min_cluster_size=7,
        cluster_gap_threshold=None,
        cluster_volume_threshold=None,
        cluster_percentage_boundary_threshold=None
    )
    
    return zbp_ds

def main():
    base_folder = Path("/home/pseudonym/Documents/Code/azure-quantum-tgp/data")
    out_dir = Path(__file__).parent / "paper_figures"
    out_dir.mkdir(exist_ok=True)
    
    protocol_config_path = Path(__file__).parent.parent.parent / "Inputs/Parameters/agent_protocol_yield_fix.yaml"
    config = ConfigManager.get_protocol_config(str(protocol_config_path))
    
    for i, meta in enumerate(DATASETS):
        name = meta["name"]
        print(f"Processing {name}...")
        
        is_experimental = meta["type"] == "experimental"
        try:
            if is_experimental:
                fname_l = base_folder / "experimental" / f"{name}_left.nc"
                fname_r = base_folder / "experimental" / f"{name}_right.nc"
                ds_left = xr.load_dataset(fname_l)
                ds_right = xr.load_dataset(fname_r)
                ds_left = tgp.prepare.prepare(ds_left)
                ds_right = tgp.prepare.prepare(ds_right)
                
                sys.path.append("/home/pseudonym/Documents/Code/azure-quantum-tgp/notebooks")
                import paper_figures
                c_kwargs = meta["correct_kwargs"].copy()
                c_kwargs["fridge_parameters"] = base_folder / "fridge" / c_kwargs["fridge_parameters"]
                _, (ds_left, ds_right) = paper_figures.correct_two(ds_left, ds_right, **c_kwargs)
            else:
                fname = base_folder / "simulated" / f"{name}.nc"
                sys.path.append("/home/pseudonym/Documents/Code/azure-quantum-tgp/notebooks")
                import paper_figures
                ds = paper_figures.load_cached_broadened(fname, T_mK=30.0 if "beta" in name else 40.0)
                ds_left = ds.rename({"bias": "left_bias"}).copy()
                ds_right = ds.rename({"bias": "right_bias"}).copy()
                
            zbp_ds = run_user_protocol(ds_left, ds_right, meta, config)
            
            # Generate the exact plot
            kwargs = dict(
                cutter_value=meta["selected_cutter"],
                zbp_cluster_numbers=meta["zbp_cluster_numbers"],
                plunger_cut=meta["selected_plunger"],
                pct_boundary_shifts=meta.get("pct_boundary_shifts", None)
            )
            
            if meta.get("selected_field"):
                kwargs["field_cut"] = meta["selected_field"]
            if meta.get("device_info_key"):
                kwargs["device_info"] = DEVICE_INFO[meta["device_info_key"]]
            if meta.get("invariant"):
                kwargs["invariant"] = meta["invariant"]
            if meta.get("plunger_lim"):
                kwargs["plunger_lim"] = meta["plunger_lim"]
                
            fig, axs = tgp.plot.paper.plot_stage2_diagram(zbp_ds, **kwargs)
            out_file = out_dir / f"{name}_replicated.png"
            fig.savefig(out_file, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"  Saved replicated plot to {out_file.name}")
            
        except Exception as e:
            print(f"  Error processing {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
