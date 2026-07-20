import numpy as np

def main():
    cfg = np.load('Data/Tdis_pfaff4/all_params.npz', allow_pickle=True)
    print("all_params keys:", list(cfg.keys()))
    for k in cfg.keys():
        print(f"  {k}: {cfg[k]}")
        
    plist = np.load('Data/Tdis_pfaff4/params_list.npy', allow_pickle=True)
    unique_mu = np.unique(plist[:, 1])
    unique_vz = np.unique(plist[:, 2])
    
    d_mu = np.min(np.diff(unique_mu))
    d_vz = np.min(np.diff(unique_vz))
    
    print(f"\nUnique mu count={len(unique_mu)}, min={unique_mu[0]:.6f}, max={unique_mu[-1]:.6f}, step={d_mu:.6f}")
    print(f"Unique Vz count={len(unique_vz)}, min={unique_vz[0]:.6f}, max={unique_vz[-1]:.6f}, step={d_vz:.6f}")

if __name__ == "__main__":
    main()
