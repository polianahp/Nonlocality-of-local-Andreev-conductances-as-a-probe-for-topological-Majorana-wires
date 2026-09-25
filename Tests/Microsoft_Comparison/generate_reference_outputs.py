import json
import os
import sys
from pathlib import Path
import traceback
import numpy as np
import xarray as xr
import tgp
import tgp.one
import tgp.two
import tgp.common
import tgp.prepare

from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class Thresholds:
    gap_threshold_factor: Any = 0.001
    noise_threshold: float = 1e-4
    volume_threshold: float = 0
    percentage_boundary_threshold: float = 0.6
    cutter_threshold: float = 0.5
    gap_threshold: float = 0.005
    gap_threshold_high: float = 0.067

    @classmethod
    def from_ds_attrs(cls, ds: xr.Dataset):
        lever_arm = ds.attrs.get("lever_arm", 85)
        g_factor = ds.attrs.get("g_factor", 5.1)
        if ds.sample_name == "simulated_SLG_beta":
            lever_arm = 77.8
            g_factor = 4.11
        from scipy.constants import physical_constants
        mu_B = physical_constants["Bohr magneton in eV/T"][0]
        E_Z = 0.5 * g_factor * mu_B * 1e3
        
        surface_charge = ds.attrs.get("surface_charge", 0.0)
        gap_threshold_factor_map = {0.0: 0.001, 0.1: 0.01, 1.0: 0.05, 2.7: 0.05, 4.0: 0.05}
        gap_threshold_factor = gap_threshold_factor_map.get(float(surface_charge), 0.001)

        return cls(
            volume_threshold=cls.gap_threshold**2 / E_Z / lever_arm,
            gap_threshold_factor=gap_threshold_factor
        )

def _roi1(ds, pct_box=10, min_margin_box=(0.003, 0.2)):
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

def analyze_1(fn, T_mK, *, return_ds: bool = False):
    import xarray as xr
    thresholds = dict(
        set_gapped={"th_2w_p": 0.5},
        set_3w_th={"th_3w": 1e3},
        set_3w_tat={"th_3w_tat": 0.7},
    )
    ds_raw = xr.load_dataset(fn, engine="h5netcdf", invalid_netcdf=True)
    ds = tgp.prepare.prepare_sim(ds_raw, T_mK)
    try:
        tgp.one.analyze(ds, thresholds)
        res = _roi1(ds)
    except tgp.one.ThresholdException:
        res = {}
    r = dict(ds.attrs)
    r.update(res)
    if return_ds:
        return r, ds
    return r

def analyze_2(fn, T_mK, B_max=None, return_datasets=True):
    import xarray as xr
    ds_raw = xr.load_dataset(fn, engine="h5netcdf", invalid_netcdf=True)
    ds = tgp.prepare.prepare_sim(ds_raw, T_mK)
    if B_max is not None:
        sel = ds.B <= B_max
        if sel.sum() <= 1:
            return None
        ds = ds.sel(B=sel)
    th = Thresholds.from_ds_attrs(ds)
    ds_left = ds.rename({"bias": "left_bias"})
    ds_right = ds.rename({"bias": "right_bias"})
    ds_left, ds_right = tgp.two.extract_gap(
        ds_left, ds_right, gap_threshold_factor=th.gap_threshold_factor, noise_threshold=th.noise_threshold,
    )
    zbp_ds = tgp.two.zbp_dataset_derivative(
        ds_left, ds_right, average_over_cutter=False, bias_window=0.007
    )
    tgp.two.set_gap_threshold(zbp_ds, threshold_low=th.gap_threshold, threshold_high=th.gap_threshold_high)
    zbp_ds = tgp.two.cluster_and_score(
        zbp_ds,
        cluster_gap_threshold=th.gap_threshold,
        cluster_percentage_boundary_threshold=th.percentage_boundary_threshold,
        cluster_volume_threshold=th.volume_threshold,
    )
    passing_list, roi2_stats = get_roi2_stats(zbp_ds, th.cutter_threshold)
    soi2_stats = get_soi2_stats(zbp_ds)
    overlapping_clusters = tgp.two.cluster_sets_to_cluster_pairs(
        zbp_ds.cluster_sets, mode="dimension"
    )
    result = {
        "roi2_stats": roi2_stats,
        "soi2_stats": soi2_stats,
        "passing_list": passing_list,
        "overlapping_clusters": overlapping_clusters,
        **asdict(th),
        **ds.attrs,
    }
    if return_datasets:
        result["zbp_ds"] = zbp_ds
        result["ds_left"] = ds_left
        result["ds_right"] = ds_right
    return result

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
    sel = dict(zbp_cluster_number=cluster.zbp_cluster_number.item(), cutter_pair_index=cluster.cutter_pair_index.item())
    ds_sel = zbp_ds.sel(sel)
    keys = {"top_quintile_gap", "cluster_volume", "median_gap", "cluster_B_center", "cluster_V_center", "percentage_boundary", "ncutters", "cluster_B_size", "cluster_V_size"}
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

def main():
    base_dir = Path(__file__).parent
    s1_list_path = base_dir / "file_list_stage1.json"
    s2_list_path = base_dir / "file_list_stage2.json"
    out_dir = base_dir / "verification_data" / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(s1_list_path, "r") as f:
        s1_files = json.load(f)
    with open(s2_list_path, "r") as f:
        s2_files = json.load(f)
        
    manifest = {}
    stage1_results = {}
    stage2_results = {}

    for i, (fn1, fn2) in enumerate(zip(s1_files, s2_files)):
        print(f"Processing {i}: {Path(fn1).name}")
        manifest[i] = {
            "stage1_input": fn1,
            "stage2_input": fn2,
            "dataset_output": f"scored_ref_{i}.nc"
        }
        
        # Stage 1
        try:
            print("  Running Stage 1 Analysis...")
            
            r1 = analyze_1(Path(fn1), T_mK=30, return_ds=False)
            
            # Microsoft explicit & default params for Stage 1
            r1["parameters"] = {
                "T_mK_stage1": 30.0,
                "roi1_pct_box": 10,
                "roi1_min_margin_box": [0.003, 0.2],
                "n_tiles": 100,
                "th_2w_p": 0.5,
                "th_3w": 1000.0,
                "th_3w_tat": 0.7
            }
            
            stage1_results[str(i)] = convert_to_serializable(r1)
            print(f"  [Stage 1] V_min: {r1.get('V_min', 'None')}, V_max: {r1.get('V_max', 'None')}")
        except Exception as e:
            import traceback; traceback.print_exc()
            stage1_results[str(i)] = {"error": str(e)}
            
        # Stage 2
        try:
            print("  Running Stage 2 Analysis...")
            
            
            
            r2 = analyze_2(Path(fn2), T_mK=40, B_max=3.0, return_datasets=True)
            if r2 and 'zbp_ds' in r2:
                r2['extra.device_passed'] = r2['zbp_ds'].attrs.get('extra.device_passed', 'Unknown')
                
            if r2 is None:
                stage2_results[str(i)] = {"error": "analyze_2 returned None"}
            else:
                # Save heavy .nc dataset file before popping
                zbp_ds = r2.pop("zbp_ds", None)
                if zbp_ds is not None:
                    nc_out_path = out_dir / f"scored_ref_{i}.nc"
                    print(f"  [Stage 2] Saving dataset to {nc_out_path.name}...")
                    zbp_ds.to_netcdf(nc_out_path, engine="h5netcdf", invalid_netcdf=True)
                    
                # Remove remaining heavy intermediate datasets
                r2.pop("ds_left", None)
                r2.pop("ds_right", None)
                
                # Microsoft explicit & default params for Stage 2
                r2["parameters"] = {
                    "T_mK_stage2": 40.0,
                    "gap_threshold_factor": r2.get("gap_threshold_factor"),
                    "noise_threshold": r2.get("noise_threshold"),
                    "upper_conductance_threshold": None,
                    "bias_window": 0.007,
                    "derivative_threshold": 100.0,
                    "zbp_probability_threshold": 0.6,
                    "average_over_cutter": False,
                    "threshold_low": 0.005,
                    "threshold_high": r2.get("gap_threshold_high"),
                    "min_cluster_size": 7,
                    "cluster_gap_threshold": r2.get("gap_threshold"),
                    "cluster_percentage_boundary_threshold": r2.get("percentage_boundary_threshold"),
                    "cluster_ncutter_threshold": r2.get("cutter_threshold"),
                    "cluster_volume_threshold": r2.get("volume_threshold")
                }
                
                stage2_results[str(i)] = convert_to_serializable(r2)
                
            print(f"  [Stage 2] Passed: {r2.get('extra.device_passed', 'Unknown') if r2 else 'None'}")
            
        except Exception as e:
            import traceback; traceback.print_exc()
            stage2_results[str(i)] = {"error": str(e)}

    with open(out_dir / "stage1_results.json", "w") as f:
        json.dump(stage1_results, f, indent=2)
        
    with open(out_dir / "stage2_results.json", "w") as f:
        json.dump(stage2_results, f, indent=2)
        
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("Done generating reference outputs.")

if __name__ == "__main__":
    main()
