import json
import os
import sys
from pathlib import Path
import traceback
import numpy as np
import xarray as xr

# Add parent directory to path to import src
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.gpu_broadening import prepare_sim_gpu
from src.parameter_handler import ConfigManager

from scipy.constants import physical_constants

import tgp
import tgp.one
import tgp.two
import tgp.common



def get_roi2_stats(zbp_ds, cutter_threshold):
    results = []
    cnt = {"not passed": 0, "true positive": 0, "false positive": 0}
    if zbp_ds.gapped_zbp_cluster.max() == 0:
        return results, cnt
    roi2s = tgp.common.expand_clusters(zbp_ds["roi2"], dim="roi2")
    for roi2 in roi2s:
        n = zbp_ds.ncutters.where(zbp_ds.cluster_sets == roi2.roi2).max().item()
        if n < cutter_threshold * zbp_ds.dims["cutter_pair_index"]:
            result = "not passed"
        elif (zbp_ds.SI * roi2).any():
            result = "true positive"
        else:
            result = "false positive"
        results.append(result)
        cnt[result] += 1
    return results, cnt

def _soi_stats(zbp_ds, cluster):
    sel = dict(
        zbp_cluster_number=cluster.zbp_cluster_number.item(),
        cutter_pair_index=cluster.cutter_pair_index.item(),
    )
    ds_sel = zbp_ds.sel(sel)
    keys = {
        "top_quintile_gap", "cluster_volume", "median_gap", "cluster_B_center",
        "cluster_V_center", "percentage_boundary", "ncutters", "cluster_B_size", "cluster_V_size"
    }
    r = {k: ds_sel[k].item() for k in keys}
    r["cluster_volume"] *= 1e3
    r.update(sel)
    return r

def get_soi2_stats(zbp_ds):
    stats = [
        _soi_stats(zbp_ds, cluster)
        for cl in zbp_ds.gapped_zbp_cluster.transpose("cutter_pair_index", ...)
        for cluster in tgp.common.expand_clusters(cl, dim="zbp_cluster_number")
    ]
    return {(stat["cutter_pair_index"], stat["zbp_cluster_number"]): stat for stat in stats}




def _roi1(ds, pct_box, min_margin_box):
    cluster_infos = tgp.common.cluster_infos(
        ds.clusters, "B", "V", pct_box=pct_box, min_margin_box=tuple(min_margin_box)
    )
    bboxes = [info.bounding_box for info in cluster_infos]
    if not bboxes:
        return {}
    x1, y1, x2, y2 = np.array(bboxes).T
    gx1, gy1, gx2, gy2 = (
        ds.coords["V"].values[0], ds.coords["B"].values[0],
        ds.coords["V"].values[-1], ds.coords["B"].values[-1],
    )
    return dict(
        V_min=float(max(x1.min(), gx1)), V_max=float(min(x2.max(), gx2)),
        B_min=float(max(y1.min(), gy1)), B_max=float(min(y2.max(), gy2)),
    )

def convert_to_serializable(obj):
    import numpy as np
    import xarray as xr
    if type(obj).__name__ in ('int32', 'int64', 'int16', 'int8') or isinstance(obj, (np.integer, int)):
        return int(obj)
    elif type(obj).__name__ in ('float32', 'float64', 'float16') or isinstance(obj, (np.floating, float)):
        return float(obj)
    elif type(obj).__name__ == 'bool_':
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_to_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, set):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, (xr.Dataset, xr.DataArray)):
        return "<xarray>"
    elif isinstance(obj, tuple):
        return str(obj)
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (str, bool, type(None))):
        return obj
    else:
        return str(obj)

def main():
    base_dir = Path(__file__).parent
    s1_list_path = base_dir / "file_list_stage1.json"
    s2_list_path = base_dir / "file_list_stage2.json"
    out_dir = base_dir / "verification_data" / "user"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    protocol_config_path = base_dir.parent.parent / "Inputs" / "Parameters" / "agent_protocol_yield_fix.yaml"
    config = ConfigManager.get_protocol_config(str(protocol_config_path))
    
    with open(s1_list_path, "r") as f:
        s1_files = json.load(f)
    with open(s2_list_path, "r") as f:
        s2_files = json.load(f)
        
    manifest = {}
    stage1_results = {}
    stage2_results = {}

    thresholds_s1 = dict(
        set_2w_th={"n_tiles": config.n_tiles},
        set_gapped={"th_2w_p": config.th_2w_p},
        set_3w_th={"th_3w": config.th_3w},
        set_3w_tat={"th_3w_tat": config.th_3w_tat},
    )

    for i, (fn1, fn2) in enumerate(zip(s1_files, s2_files)):
        print(f"Processing {i}: {Path(fn1).name}")
        manifest[i] = {
            "stage1_input": fn1,
            "stage2_input": fn2,
            "dataset_output": f"scored_user_{i}.nc"
        }
        
        # ---------------- STAGE 1 ----------------
        try:

            if config.GPU_broadening:
                print(f"  Running Stage 1 Analysis (GPU Broadening {config.T_mK_stage1}mK)...")
                ds_raw1 = xr.load_dataset(fn1, engine="h5netcdf", invalid_netcdf=True)
                ds1 = prepare_sim_gpu(ds_raw1, T_mK=config.T_mK_stage1)

            else:
                print(f"  Running Stage 1 Analysis (CPU Broadening {config.T_mK_stage1}mK)...")
                ds_raw1 = xr.load_dataset(fn1, engine="h5netcdf", invalid_netcdf=True)
                ds1 = tgp.prepare.prepare_sim(ds_raw1, T_mK=config.T_mK_stage1)
            
            tgp.one.analyze(ds1, thresholds_s1)
            roi1 = _roi1(ds1, config.roi1_pct_box, config.roi1_min_margin_box)
            
            # Incorporate original metadata
            r1 = dict(ds1.attrs)
            r1.update(roi1)
            
            # User config parameters for Stage 1
            r1["parameters"] = {
                "GPU_broadening": bool(config.GPU_broadening),
                "T_mK_stage1": float(config.T_mK_stage1),
                "roi1_pct_box": float(config.roi1_pct_box),
                "roi1_min_margin_box": [float(x) for x in config.roi1_min_margin_box],
                "n_tiles": int(config.n_tiles) if config.n_tiles is not None else None,
                "th_2w_p": float(config.th_2w_p) if config.th_2w_p is not None else None,
                "th_3w": float(config.th_3w) if config.th_3w is not None else None,
                "th_3w_tat": float(config.th_3w_tat) if config.th_3w_tat is not None else None
            }
            
            stage1_results[str(i)] = convert_to_serializable(r1)
            print(f"  [Stage 1] V_min: {r1.get('V_min', 'None')}, V_max: {r1.get('V_max', 'None')}")
            
        except tgp.one.ThresholdException as e:
            print(f"  [Stage 1] Exception: {e}. No clusters found.")
            r1 = dict(ds1.attrs)
            stage1_results[str(i)] = convert_to_serializable(r1)
            roi1 = {} # No clusters
        except Exception as e:
            print(f"  [Stage 1] Error: {e}")
            stage1_results[str(i)] = {"error": str(e)}
            roi1 = {}
            
        # ---------------- STAGE 2 ----------------
        try:
            if not roi1:
                raise Exception("No ROI1 found in Stage 1, cannot proceed to Stage 2")
                
            if config.GPU_broadening:
                print(f"  Running Stage 2 Analysis (GPU Broadening {config.T_mK_stage2}mK)...")
                ds_raw2 = xr.load_dataset(fn2, engine="h5netcdf", invalid_netcdf=True)
                ds2 = prepare_sim_gpu(ds_raw2, T_mK=config.T_mK_stage2)
            else:
                print(f"  Running Stage 2 Analysis (CPU Broadening {config.T_mK_stage2}mK)...")
                ds_raw2 = xr.load_dataset(fn2, engine="h5netcdf", invalid_netcdf=True)
                ds2 = tgp.prepare.prepare_sim(ds_raw2, T_mK=config.T_mK_stage2)
            
            # Truncate B max if applicable, then slice by roi1 B_max
            b_max_val = config.B_max_stage2
            if b_max_val is None:
                b_max_val = roi1["B_max"] if roi1 else 3.0
            sel = ds2.B <= b_max_val
            if sel.sum() <= 1:
                raise Exception("No B field left after B_max truncation")
            ds2 = ds2.sel(B=sel)
            
            ds_left = ds2.rename({"bias": "left_bias"}).copy()
            ds_right = ds2.rename({"bias": "right_bias"}).copy()
            
            # Dynamic calculation of parameters matching MS reference
            th_factor = config.gap_threshold_factor
            if isinstance(th_factor, dict):
                sc = getattr(ds2, 'surface_charge', 0.0)
                th_factor = th_factor.get(float(sc), th_factor.get(str(float(sc)), 0.001))
            else:
                th_factor = float(th_factor)
                
            # Dynamic volume threshold calculation matching MS
            lever_arm = 85; g_factor = 5.1
            if ds2.sample_name == "simulated_SLG_beta":
                lever_arm = 77.8; g_factor = 4.11
            elif ds2.sample_name == "simulated_DLG_epsilon":
                lever_arm = 85; g_factor = 5.1
            
            mu_B = physical_constants["Bohr magneton in eV/T"][0]
            E_Z = 0.5 * g_factor * mu_B * 1e3
            volume_threshold = (config.cluster_gap_threshold**2) / E_Z / lever_arm
            
            ds_left, ds_right = tgp.two.extract_gap(
                ds_left, ds_right, 
                gap_threshold_factor=th_factor, noise_threshold=config.noise_threshold,
                upper_conductance_threshold=config.upper_conductance_threshold if config.upper_conductance_threshold is not None else np.inf
            )
            
            zbp_ds = tgp.two.zbp_dataset_derivative(
                ds_left, ds_right, bias_window=config.bias_window,
                derivative_threshold=config.derivative_threshold,
                zbp_probability_threshold=config.zbp_probability_threshold,
                average_over_cutter=config.average_over_cutter
            )
            
            tgp.two.set_gap_threshold(
                zbp_ds, threshold_low=config.threshold_low, threshold_high=config.threshold_high
            )
            
            zbp_ds = tgp.two.cluster_and_score(
                zbp_ds,
                min_cluster_size=config.min_cluster_size,
                cluster_gap_threshold=config.cluster_gap_threshold,
                cluster_percentage_boundary_threshold=config.cluster_percentage_boundary_threshold,
                cluster_ncutter_threshold=config.cluster_ncutter_threshold,
                cluster_volume_threshold=volume_threshold
            )
            
            # Save heavy .nc dataset file
            nc_out_path = out_dir / f"scored_user_{i}.nc"
            print(f"  [Stage 2] Saving dataset to {nc_out_path.name}...")
            zbp_ds.to_netcdf(nc_out_path, engine="h5netcdf", invalid_netcdf=True)
            
            # Use Microsoft's extractors directly for parity
            passing_list, roi2_stats = get_roi2_stats(zbp_ds, config.cluster_ncutter_threshold)
            soi2_stats = get_soi2_stats(zbp_ds)
            overlapping_clusters = tgp.two.cluster_sets_to_cluster_pairs(
                zbp_ds.cluster_sets, mode="dimension"
            )
            
            r2 = {
                "roi2_stats": roi2_stats,
                "soi2_stats": soi2_stats,
                "passing_list": passing_list,
                "overlapping_clusters": overlapping_clusters,
                "extra.device_passed": getattr(zbp_ds, "extra.device_passed", zbp_ds.attrs.get("extra.device_passed", "Unknown")),
                "parameters": {
                    "GPU_broadening": bool(config.GPU_broadening),
                    "T_mK_stage2": float(config.T_mK_stage2),
                    "gap_threshold_factor": float(th_factor),
                    "noise_threshold": float(config.noise_threshold),
                    "upper_conductance_threshold": float(config.upper_conductance_threshold) if config.upper_conductance_threshold is not None else None,
                    "bias_window": float(config.bias_window),
                    "derivative_threshold": float(config.derivative_threshold),
                    "zbp_probability_threshold": float(config.zbp_probability_threshold),
                    "average_over_cutter": bool(config.average_over_cutter),
                    "threshold_low": float(config.threshold_low),
                    "threshold_high": float(config.threshold_high),
                    "min_cluster_size": int(config.min_cluster_size),
                    "cluster_gap_threshold": float(config.cluster_gap_threshold),
                    "cluster_percentage_boundary_threshold": float(config.cluster_percentage_boundary_threshold),
                    "cluster_ncutter_threshold": float(config.cluster_ncutter_threshold),
                    "cluster_volume_threshold": float(volume_threshold)
                },
                **ds2.attrs
            }
            
            stage2_results[str(i)] = convert_to_serializable(r2)
            print(f"  [Stage 2] Passed: {r2.get('extra.device_passed', 'Unknown')}")
            
        except Exception as e:
            print(f"  [Stage 2] Error: {e}")
            stage2_results[str(i)] = {"error": str(e)}

    with open(out_dir / "stage1_results.json", "w") as f:
        json.dump(stage1_results, f, indent=2)
        
    with open(out_dir / "stage2_results.json", "w") as f:
        json.dump(stage2_results, f, indent=2)
        
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("Done generating user outputs.")

if __name__ == "__main__":
    main()
