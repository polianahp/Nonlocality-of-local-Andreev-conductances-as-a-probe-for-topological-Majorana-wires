import io
import json
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")  # Headless execution safety
import matplotlib.pyplot as plt
from PIL import Image
import tgp.plot.paper

def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)

def test_stage1(r1_ref, r1_usr, i, log_file):
    log_file.write(f"  Testing Stage 1 ROI1 Bounds...\n")
    if "error" in r1_ref and "error" in r1_usr:
        log_file.write("    PASS (Both threw exceptions)\n")
        return True
    elif "error" in r1_ref:
        log_file.write(f"    FAIL: Reference threw exception but User did not: {r1_ref['error']}\n")
        return False
    elif "error" in r1_usr:
        log_file.write(f"    FAIL: User threw exception but Reference did not: {r1_usr['error']}\n")
        return False
        
    passed = True
    for k in ["V_min", "V_max", "B_min", "B_max"]:
        v_ref = r1_ref.get(k)
        v_usr = r1_usr.get(k)
        if v_ref is None and v_usr is None:
            continue
        if v_ref is None or v_usr is None:
            log_file.write(f"    FAIL: {k} missing in one output (ref: {v_ref}, usr: {v_usr})\n")
            passed = False
            continue
        
        match = np.isclose(v_ref, v_usr, atol=1e-4)
        log_file.write(f"    [{'PASS' if match else 'FAIL'}] {k}: ref={v_ref} | usr={v_usr}\n")
        if not match: passed = False
            
    # Check Parameters
    p_ref = r1_ref.get("parameters", {})
    p_usr = r1_usr.get("parameters", {})
    
    for k in p_ref.keys():
        v_ref = p_ref.get(k)
        v_usr = p_usr.get(k)
        if v_ref is None and v_usr is None:
            continue
        if v_ref is None or v_usr is None:
            log_file.write(f"    FAIL: Parameter {k} missing in one output (ref: {v_ref}, usr: {v_usr})\n")
            passed = False
            continue
            
        if isinstance(v_ref, float) or isinstance(v_usr, float):
            match = np.isclose(v_ref, v_usr, atol=1e-4)
        else:
            match = (v_ref == v_usr)
            
        log_file.write(f"    [{'PASS' if match else 'FAIL'}] param {k}: ref={v_ref} | usr={v_usr}\n")
        if not match: passed = False

    return passed

def test_stage2(r2_ref, r2_usr, i, log_file):
    log_file.write(f"  Testing Stage 2 Metrics...\n")
    
    if "error" in r2_ref and "error" in r2_usr:
        log_file.write("    PASS (Both threw exceptions or returned None)\n")
        return True
    elif "error" in r2_ref:
        log_file.write(f"    FAIL: Reference threw exception but User did not: {r2_ref['error']}\n")
        return False
    elif "error" in r2_usr:
        log_file.write(f"    FAIL: User threw exception but Reference did not: {r2_usr['error']}\n")
        return False

    passed = True
    
    dp_ref = r2_ref.get("extra.device_passed")
    dp_usr = r2_usr.get("extra.device_passed")
    match = (dp_ref == dp_usr)
    log_file.write(f"    [{'PASS' if match else 'FAIL'}] extra.device_passed: ref={dp_ref} | usr={dp_usr}\n")
    if not match: passed = False
        
    roi2_ref = r2_ref.get("roi2_stats", {})
    roi2_usr = r2_usr.get("roi2_stats", {})
    match = (roi2_ref == roi2_usr)
    log_file.write(f"    [{'PASS' if match else 'FAIL'}] roi2_stats counts: ref={roi2_ref} | usr={roi2_usr}\n")
    if not match: passed = False
        
    pl_ref = r2_ref.get("passing_list", [])
    pl_usr = r2_usr.get("passing_list", [])
    match = (pl_ref == pl_usr)
    log_file.write(f"    [{'PASS' if match else 'FAIL'}] passing_list: ref={pl_ref} | usr={pl_usr}\n")
    if not match: passed = False
        
    soi2_ref = r2_ref.get("soi2_stats", {})
    soi2_usr = r2_usr.get("soi2_stats", {})
    
    ref_keys = set(soi2_ref.keys())
    usr_keys = set(soi2_usr.keys())
    
    match = (ref_keys == usr_keys)
    log_file.write(f"    [{'PASS' if match else 'FAIL'}] soi2_stats cluster keys: ref={sorted(list(ref_keys))} | usr={sorted(list(usr_keys))}\n")
    if not match:
        passed = False
    else:
        for k in sorted(list(ref_keys)):
            c_ref = soi2_ref[k]
            c_usr = soi2_usr[k]
            for metric in ["ncutters", "cluster_V_center", "cluster_B_center", 
                           "median_gap", "cluster_V_size", "top_quintile_gap", 
                           "cluster_volume", "percentage_boundary", "cluster_B_size"]:
                m_ref = c_ref.get(metric, 0.0)
                m_usr = c_usr.get(metric, 0.0)
                match_val = np.isclose(m_ref, m_usr, atol=1e-5, rtol=1e-3)
                log_file.write(f"      [{'PASS' if match_val else 'FAIL'}] cluster {k} {metric}: ref={m_ref} | usr={m_usr}\n")
                if not match_val: passed = False
                    
    oc_ref = r2_ref.get("overlapping_clusters", [])
    oc_usr = r2_usr.get("overlapping_clusters", [])
    
    oc_ref_sorted = sorted([sorted(group) for group in oc_ref])
    oc_usr_sorted = sorted([sorted(group) for group in oc_usr])
    
    match = (oc_ref_sorted == oc_usr_sorted)
    log_file.write(f"    [{'PASS' if match else 'FAIL'}] overlapping_clusters: ref={oc_ref_sorted} | usr={oc_usr_sorted}\n")
    if not match: passed = False

    # Check Parameters
    p_ref = r2_ref.get("parameters", {})
    p_usr = r2_usr.get("parameters", {})
    
    for k in p_ref.keys():
        v_ref = p_ref.get(k)
        v_usr = p_usr.get(k)
        if v_ref is None and v_usr is None:
            continue
        if v_ref is None or v_usr is None:
            log_file.write(f"    FAIL: Parameter {k} missing in one output (ref: {v_ref}, usr: {v_usr})\n")
            passed = False
            continue
            
        if isinstance(v_ref, float) or isinstance(v_usr, float):
            match = np.isclose(v_ref, v_usr, atol=1e-4)
        else:
            match = (v_ref == v_usr)
            
        log_file.write(f"    [{'PASS' if match else 'FAIL'}] param {k}: ref={v_ref} | usr={v_usr}\n")
        if not match: passed = False

    return passed

def generate_comparison_plot(ref_nc_path: Path, usr_nc_path: Path, device_idx: int, out_dir: Path, cutter_idx: int = 0):
    if not ref_nc_path.exists() or not usr_nc_path.exists():
        return None
    try:
        ds_ref = xr.load_dataset(ref_nc_path, engine="h5netcdf", invalid_netcdf=True)
        ds_usr = xr.load_dataset(usr_nc_path, engine="h5netcdf", invalid_netcdf=True)

        fig_ref, _ = tgp.plot.paper.plot_stage2_diagram(ds_ref, cutter_value=cutter_idx)
        fig_ref.suptitle(f"Device {device_idx} — Microsoft Reference Pipeline", fontsize=15, y=1.02)
        buf_ref = io.BytesIO()
        fig_ref.savefig(buf_ref, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig_ref)
        buf_ref.seek(0)
        img_ref = Image.open(buf_ref)

        fig_usr, _ = tgp.plot.paper.plot_stage2_diagram(ds_usr, cutter_value=cutter_idx)
        fig_usr.suptitle(f"Device {device_idx} — User Pipeline", fontsize=15, y=1.02)
        buf_usr = io.BytesIO()
        fig_usr.savefig(buf_usr, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig_usr)
        buf_usr.seek(0)
        img_usr = Image.open(buf_usr)

        total_width = max(img_ref.width, img_usr.width)
        total_height = img_ref.height + img_usr.height
        combined = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))
        combined.paste(img_ref, (0, 0))
        combined.paste(img_usr, (0, img_ref.height))

        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"phase_diagram_comparison_device_{device_idx}.png"
        combined.save(out_file)
        return out_file
    except Exception as e:
        print(f"  [PLOT ERROR] Device {device_idx}: {e}")
        return None

def main():
    base_dir = Path(__file__).parent / "verification_data"
    ref_dir = base_dir / "reference"
    usr_dir = base_dir / "user"
    log_path = Path(__file__).parent / "detailed_comparison_log.txt"
    
    with open(ref_dir / "manifest.json", 'r') as f:
        manifest = json.load(f)
        
    s1_ref = load_json(ref_dir / "stage1_results.json")
    s2_ref = load_json(ref_dir / "stage2_results.json")
    s1_usr = load_json(usr_dir / "stage1_results.json")
    s2_usr = load_json(usr_dir / "stage2_results.json")
    
    with open(log_path, 'w') as log_file:
        log_file.write("="*60 + "\n")
        log_file.write(" TGP VERIFICATION SUITE - DETAILED RESULTS\n")
        log_file.write("="*60 + "\n\n")
        
        total = len(manifest)
        passed_all = 0
        
        plots_dir = base_dir / "plots_paper"
        for i in range(total):
            idx = str(i)
            file_name = Path(manifest[idx]['stage1_input']).name
            log_file.write(f"[Test File {i}] {file_name}\n")
            print(f"Comparing [Test File {i}] {file_name}...")
            
            p1 = test_stage1(s1_ref.get(idx, {}), s1_usr.get(idx, {}), i, log_file)
            p2 = test_stage2(s2_ref.get(idx, {}), s2_usr.get(idx, {}), i, log_file)
            
            if p1 and p2:
                passed_all += 1

            # Generate side-by-side plot comparison in verification_data/plots_paper/
            ref_nc = ref_dir / f"scored_ref_{i}.nc"
            usr_nc = usr_dir / f"scored_user_{i}.nc"
            plot_file = generate_comparison_plot(ref_nc, usr_nc, i, plots_dir)
            if plot_file:
                log_file.write(f"  [PLOT] Generated side-by-side comparison plot: {plot_file.name}\n")
                print(f"  Generated side-by-side comparison plot: {plot_file.name}")
            else:
                log_file.write(f"  [PLOT] Skipped plot generation (dataset missing or error)\n")

            log_file.write("\n")
                
        summary = f"\n" + "="*60 + f"\n SUMMARY: {passed_all} / {total} files passed all equivalence checks.\n" + "="*60 + "\n"
        log_file.write(summary)
        print(summary)
        print(f"Detailed comparison saved to {log_path}")

if __name__ == "__main__":
    main()
