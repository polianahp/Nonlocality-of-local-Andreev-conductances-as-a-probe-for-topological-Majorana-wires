import numpy as np
import helpers as hp

from pathlib import Path

def simulate_protocol_cell(brcl, brcr, subdir):
    # Simulates loading barrier_arr inside protocol.ipynb cell and calling hp.calc_correlation without passing barrier_arr
    barrier_arr = hp.np_load_wrapped("barrier_arr", Path(subdir))
    corrs = np.asarray([hp.calc_correlation(brcl[i,:], brcr[i,:]) for i in range(min(5, brcl.shape[0]))])
    return corrs

def main():
    subdir = "Tdis_pfaff4"
    dir4 = f"Data/{subdir}"
    brcl4 = np.load(f"{dir4}/barrier_right_conductance_left_arr.npy", mmap_mode="r")
    brcr4 = np.load(f"{dir4}/barrier_right_conductance_right_arr.npy", mmap_mode="r")
    b4 = np.load(f"{dir4}/barrier_arr.npy")
    
    # 1. Direct explicit call with barrier_arr
    c_explicit = np.array([hp.calc_correlation(brcl4[i], brcr4[i], barrier_arr=b4) for i in range(5)])
    
    # 2. Call via simulated protocol.ipynb (no barrier_arr passed to calc_correlation)
    c_simulated = simulate_protocol_cell(brcl4, brcr4, subdir)
    
    # 3. Check what old correlation would have been (over full [-140, 140])
    c_old = np.zeros(5)
    for i in range(5):
        f1 = brcl4[i] / (brcl4[i][0] if brcl4[i][0] != 0 else 1.0)
        f2 = brcr4[i] / (brcr4[i][0] if brcr4[i][0] != 0 else 1.0)
        c_old[i] = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
        
    print("Verification of positive barrier correlation calculation on Data/Tdis_pfaff4 (first 5 points):")
    print("  Old correlation (full [-140, 140] sweep) :", np.round(c_old, 4))
    print("  New explicit (positive [0, 140] sweep)   :", np.round(c_explicit, 4))
    print("  New via protocol frame inspection        :", np.round(c_simulated, 4))
    assert np.allclose(c_explicit, c_simulated), "Frame inspection did not match explicit call!"
    print("\nSUCCESS: Frame inspection perfectly caught barrier_arr without modifying caller code!")

if __name__ == "__main__":
    main()
