import os
import numpy as np

def inspect_config(dir1, dir2):
    print("=" * 70)
    print("1. CONFIGURATION COMPARISON (all_params.npz)")
    print("=" * 70)
    cfg1 = np.load(os.path.join(dir1, "all_params.npz"), allow_pickle=True)
    cfg2 = np.load(os.path.join(dir2, "all_params.npz"), allow_pickle=True)
    
    all_keys = sorted(list(set(list(cfg1.keys()) + list(cfg2.keys()))))
    for k in all_keys:
        v1 = cfg1[k] if k in cfg1 else "MISSING"
        v2 = cfg2[k] if k in cfg2 else "MISSING"
        
        try:
            if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                if v1.shape != v2.shape:
                    print(f"  [DIFF SHAPE] {k:<25} : shape1={v1.shape} vs shape2={v2.shape}")
                elif v1.dtype == object or v2.dtype == object or v1.dtype.kind in 'SU' or v2.dtype.kind in 'SU':
                    if not np.array_equal(v1, v2):
                        print(f"  [DIFF STR]   {k:<25} : {v1} vs {v2}")
                    else:
                        print(f"  [IDENTICAL]  {k:<25} : {v1}")
                elif not np.allclose(v1, v2, equal_nan=True):
                    max_diff = np.max(np.abs(v1 - v2))
                    print(f"  [DIFF VALUE] {k:<25} : max_abs_diff={max_diff:.6e} (v1={v1}, v2={v2})")
                else:
                    print(f"  [IDENTICAL]  {k:<25} : {v1}")
            else:
                if str(v1) != str(v2):
                    print(f"  [DIFF SCALAR]{k:<25} : '{v1}' vs '{v2}'")
                else:
                    print(f"  [IDENTICAL]  {k:<25} : {v1}")
        except Exception as e:
            print(f"  [ERROR]      {k:<25} : {v1} vs {v2} ({e})")

def inspect_arrays(dir1, dir2):
    print("\n" + "=" * 70)
    print("2. ARRAY SHAPES AND GRID SPACING COMPARISON")
    print("=" * 70)
    files1 = sorted([f for f in os.listdir(dir1) if f.endswith('.npy')])
    files2 = sorted([f for f in os.listdir(dir2) if f.endswith('.npy')])
    
    all_files = sorted(list(set(files1 + files2)))
    for f in all_files:
        p1 = os.path.join(dir1, f)
        p2 = os.path.join(dir2, f)
        if not os.path.exists(p1):
            print(f"  {f:<40} : ONLY in {dir2}")
            continue
        if not os.path.exists(p2):
            print(f"  {f:<40} : ONLY in {dir1}")
            continue
            
        try:
            a1 = np.load(p1, mmap_mode="r", allow_pickle=True)
            a2 = np.load(p2, mmap_mode="r", allow_pickle=True)
            if a1.shape != a2.shape:
                print(f"  [DIFF SHAPE] {f:<35} : {str(a1.shape):<15} vs {str(a2.shape):<15}")
            else:
                # If small or 1D/2D, check value differences
                if a1.dtype != object and a2.dtype != object:
                    if not np.allclose(a1, a2, equal_nan=True):
                        diff = np.abs(a1 - a2)
                        max_diff = np.max(diff)
                        mean_diff = np.mean(diff)
                        print(f"  [DIFF VALUE] {f:<35} : shape={str(a1.shape):<12} max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}")
                    else:
                        print(f"  [IDENTICAL]  {f:<35} : shape={str(a1.shape):<12} exact match")
                else:
                    print(f"  [OBJECT]     {f:<35} : shape={str(a1.shape):<12}")
        except Exception as e:
            print(f"  [ERROR]      {f:<35} : {e}")

def detailed_inspection(dir1, dir2):
    print("\n" + "=" * 70)
    print("3. DETAILED PROTOCOL METRIC ANALYSIS")
    print("=" * 70)
    # Compare barrier_arr
    b1 = np.load(os.path.join(dir1, "barrier_arr.npy"))
    b2 = np.load(os.path.join(dir2, "barrier_arr.npy"))
    print(f"barrier_arr (Tdis_pfaff3): len={len(b1)}, min={b1[0]}, max={b1[-1]}, step={b1[1]-b1[0]:.4f}")
    print(f"barrier_arr (Tdis_pfaff4): len={len(b2)}, min={b2[0]}, max={b2[-1]}, step={b2[1]-b2[0]:.4f}")
    
    # Check symmetric indices
    sym1 = np.argmin(np.abs(b1 - 2.0))
    sym2 = np.argmin(np.abs(b2 - 2.0))
    print(f"U_sym=2.0 index: pfaff3 index={sym1} (U={b1[sym1]:.4f}), pfaff4 index={sym2} (U={b2[sym2]:.4f})")
    
    # Compare correlation metrics
    rc1 = np.load(os.path.join(dir1, "rG_corr.npy"))
    rc2 = np.load(os.path.join(dir2, "rG_corr.npy"))
    lc1 = np.load(os.path.join(dir1, "lG_corr.npy"))
    lc2 = np.load(os.path.join(dir2, "lG_corr.npy"))
    
    print(f"\nRight Sweep Correlation (rG_corr):")
    print(f"  pfaff3: min={np.min(rc1):.4f}, max={np.max(rc1):.4f}, mean={np.mean(rc1):.4f}, >0.7 count={np.sum(rc1 > 0.7)}")
    print(f"  pfaff4: min={np.min(rc2):.4f}, max={np.max(rc2):.4f}, mean={np.mean(rc2):.4f}, >0.7 count={np.sum(rc2 > 0.7)}")
    print(f"  Max difference between rc1 and rc2: {np.max(np.abs(rc1 - rc2)):.4e}")
    
    print(f"\nLeft Sweep Correlation (lG_corr):")
    print(f"  pfaff3: min={np.min(lc1):.4f}, max={np.max(lc1):.4f}, mean={np.mean(lc1):.4f}, >0.7 count={np.sum(lc1 > 0.7)}")
    print(f"  pfaff4: min={np.min(lc2):.4f}, max={np.max(lc2):.4f}, mean={np.mean(lc2):.4f}, >0.7 count={np.sum(lc2 > 0.7)}")
    print(f"  Max difference between lc1 and lc2: {np.max(np.abs(lc1 - lc2)):.4e}")
    
    # Compare peaks
    pl1 = np.load(os.path.join(dir1, "peaks_left.npy"))
    pl2 = np.load(os.path.join(dir2, "peaks_left.npy"))
    print(f"\nPeaks Left differences (shape={pl1.shape}):")
    if not np.allclose(pl1, pl2, equal_nan=True):
        print(f"  Max diff in peaks_left: {np.max(np.abs(pl1 - pl2)):.4e}")
        for col, name in enumerate(["has_peak", "pos", "width", "height", "symmetry", "window/stability"]):
            diff = np.abs(pl1[:, col] - pl2[:, col])
            print(f"    col {col} ({name:<16}): max_diff={np.max(diff):.4e}, mismatch_count={np.sum(diff > 1e-5)}")
    else:
        print("  peaks_left are EXACTLY identical.")

    pr1 = np.load(os.path.join(dir1, "peaks_right.npy"))
    pr2 = np.load(os.path.join(dir2, "peaks_right.npy"))
    print(f"\nPeaks Right differences (shape={pr1.shape}):")
    if not np.allclose(pr1, pr2, equal_nan=True):
        print(f"  Max diff in peaks_right: {np.max(np.abs(pr1 - pr2)):.4e}")
        for col, name in enumerate(["has_peak", "pos", "width", "height", "symmetry", "window/stability"]):
            diff = np.abs(pr1[:, col] - pr2[:, col])
            print(f"    col {col} ({name:<16}): max_diff={np.max(diff):.4e}, mismatch_count={np.sum(diff > 1e-5)}")
    else:
        print("  peaks_right are EXACTLY identical.")

    # Check params_list order / alignment
    p1 = np.load(os.path.join(dir1, "params_list.npy"))
    p2 = np.load(os.path.join(dir2, "params_list.npy"))
    if not np.allclose(p1, p2):
        print(f"\nWARNING: params_list coordinate grids differ between pfaff3 and pfaff4! max_diff={np.max(np.abs(p1 - p2)):.4e}")
    else:
        print(f"\nparams_list coordinate grids are EXACTLY identical.")

    # Check pdi_data
    pdi1 = np.load(os.path.join(dir1, "pdi_data.npy"))
    pdi2 = np.load(os.path.join(dir2, "pdi_data.npy"))
    if not np.allclose(pdi1, pdi2):
        print(f"WARNING: pdi_data (topological invariant) differs! max_diff={np.max(np.abs(pdi1 - pdi2)):.4e}")
    else:
        print(f"pdi_data (topological invariant) is EXACTLY identical.")

def main():
    dir1 = "Data/Tdis_pfaff3"
    dir2 = "Data/Tdis_pfaff4"
    inspect_config(dir1, dir2)
    inspect_arrays(dir1, dir2)
    detailed_inspection(dir1, dir2)

if __name__ == "__main__":
    main()
