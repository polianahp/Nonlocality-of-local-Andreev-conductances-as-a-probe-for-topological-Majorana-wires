#!/usr/bin/env python3
"""Targeted Patch Script: Computes ONLY missing Stage 1 TGP cutter bias sweeps for a realization.

Designed for high-throughput execution on Harpers Ferry (256 cores).
Leaves all existing files (all_params.npz, b_data, conductance_matrix, etc.) completely intact.

Writes strictly:
  - tgp_stage1_dIdVl.npy       shape: (N_points, 5, 97)
  - tgp_stage1_dIdVr.npy       shape: (N_points, 5, 97)
  - tgp_stage1_dIdV_LR.npy     shape: (N_points, 5, 97)
  - tgp_stage1_dIdV_RL.npy     shape: (N_points, 5, 97)
  - tgp_barrier_arr.npy        shape: (5,)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np

# Ensure src is in Python path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import helpers as hp


def _eval_single_point(args: Tuple[int, float, float, Dict[str, Any], np.ndarray, np.ndarray]):
    idx, mu, vz, static_params, tgp_barrier_arr, tgp_energies = args
    t = static_params["t"]
    mu_n = static_params["mu_n"]
    Delta0 = static_params["Delta0"]
    gamma = static_params["gamma"]
    alpha = static_params["alpha"]
    Ln = static_params["Ln"]
    Lb = static_params["Lb"]
    Ls = static_params["Ls"]
    mu_leads = static_params["mu_leads"]
    Vdisx = static_params["Vdisx"]
    solver_type = static_params.get("solver_type", "cpu")

    dL_arr = np.zeros((5, len(tgp_energies)), dtype=np.float64)
    dR_arr = np.zeros((5, len(tgp_energies)), dtype=np.float64)
    dLR_arr = np.zeros((5, len(tgp_energies)), dtype=np.float64)
    dRL_arr = np.zeros((5, len(tgp_energies)), dtype=np.float64)

    for b_idx, b_val in enumerate(tgp_barrier_arr):
        syst_tgp = hp.build_system(
            t=t,
            mu=mu,
            mu_n=mu_n,
            Delta0=Delta0,
            gamma=gamma,
            V_z=vz,
            alpha=alpha,
            Ln=Ln,
            Lb=Lb,
            Ls=Ls,
            mu_leads=mu_leads,
            barrier_l=b_val,
            barrier_r=b_val,
            Vdisx=Vdisx,
        )
        dL, dR, dLR, dRL, _ = hp.calc_dIdV(syst_tgp, tgp_energies, solver_type=solver_type)
        dL_arr[b_idx, :] = dL
        dR_arr[b_idx, :] = dR
        dLR_arr[b_idx, :] = dLR
        dRL_arr[b_idx, :] = dRL

    return idx, dL_arr, dR_arr, dLR_arr, dRL_arr


def resolve_realization_dir(data_root: Path, disorder_folder: str, realization_index: int) -> Path:
    target = data_root / disorder_folder / f"disorder_realization_{realization_index}"
    if not target.exists():
        # Check alternative common locations
        alt_roots = [
            Path("/scratch/nos00002/nonlocalcond/Outputs/Data/Disorder_Realizations"),
            REPO_ROOT / "Outputs" / "Data" / "Disorder_Realizations",
        ]
        for alt in alt_roots:
            cand = alt / disorder_folder / f"disorder_realization_{realization_index}"
            if cand.exists():
                return cand
    return target


def main():
    parser = argparse.ArgumentParser(description="Targeted Stage 1 TGP Data Patch Runner")
    parser.add_argument("--disorder_folder", type=str, required=True, help="e.g. V0_0_378, V0_0_645, V0_0_872, V0_1_2")
    parser.add_argument("--realization_index", type=int, required=True, help="Realization index (e.g. 50)")
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("DATA_ROOT", "/scratch/nos00002/nonlocalcond/Outputs/Data/Disorder_Realizations"),
        help="Root directory for disorder realization outputs",
    )
    parser.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "256")), help="Cores to use")
    parser.add_argument("--force", action="store_true", help="Force recalculation even if tgp_stage1_dIdVl.npy exists")
    args = parser.parse_args()

    realization_dir = resolve_realization_dir(Path(args.data_root), args.disorder_folder, args.realization_index)
    if not realization_dir.exists():
        print(f"[ERROR] Realization directory not found: {realization_dir}", file=sys.stderr)
        sys.exit(1)

    flag_file = realization_dir / "tgp_stage1_dIdVl.npy"
    if flag_file.exists() and not args.force:
        print(f"[SKIP] Realization {args.disorder_folder}/disorder_realization_{args.realization_index} already has Stage 1 data. Use --force to override.")
        sys.exit(0)

    params_npz_path = realization_dir / "all_params.npz"
    if not params_npz_path.exists():
        print(f"[ERROR] all_params.npz not found in {realization_dir}", file=sys.stderr)
        sys.exit(1)

    print("==================================================================")
    print(f" Harpers Ferry Stage 1 TGP Data Patch")
    print(f" Realization:      {args.disorder_folder}/disorder_realization_{args.realization_index}")
    print(f" Target Directory: {realization_dir}")
    print(f" Allocated Cores:  {args.cores}")
    print("==================================================================")

    t_start = time.perf_counter()
    p = np.load(params_npz_path, allow_pickle=True)

    # Reconstruct static physics parameters
    t = float(p["t"])
    Delta0 = float(p["Delta0"])
    gamma = float(p["gamma"])
    alpha = float(p["alpha"])
    Ln = int(p["Ln"])
    Lb = int(p["Lb"])
    Ls = int(p["Ls"])
    mu_leads = float(p["mu_leads"])
    mu_n = float(p["mu_n"])
    V0 = float(p["V0"])
    Vdisx = p["Vdisx"] * V0
    solver_type = str(p.get("solver_type", "cpu"))

    static_params = {
        "t": t,
        "Delta0": Delta0,
        "gamma": gamma,
        "alpha": alpha,
        "Ln": Ln,
        "Lb": Lb,
        "Ls": Ls,
        "mu_leads": mu_leads,
        "mu_n": mu_n,
        "Vdisx": Vdisx,
        "solver_type": solver_type,
    }

    # Parameter grid reconstruction
    params_list_file = realization_dir / "params_list.npy"
    if params_list_file.exists():
        params_list_raw = np.load(params_list_file, allow_pickle=True)
        params_list = [(int(row[0]), float(row[1]), float(row[2])) for row in params_list_raw]
    else:
        mu_var = p["mu_var"]
        Vz_var = p["Vz_var"]
        params_list = []
        idx = 0
        for mu_val in mu_var:
            for vz_val in Vz_var:
                params_list.append((idx, float(mu_val), float(vz_val)))
                idx += 1

    N_points = len(params_list)
    tgp_barrier_arr = np.array([4.3548, 3.6774, 3.0000, 2.3710, 1.6935], dtype=np.float64)
    tgp_energies = hp.make_dynamic_grid()  # 97 points
    N_energies = len(tgp_energies)

    print(f"Total Grid Points: {N_points}")
    print(f"Cutters:           5 barrier configurations {tgp_barrier_arr.tolist()}")
    print(f"Energy Grid:       {N_energies} dynamic points")
    print(f"Dispatching worker pool across {args.cores} processes...")

    tgp_stage1_dIdVl = np.zeros((N_points, 5, N_energies), dtype=np.float64)
    tgp_stage1_dIdVr = np.zeros((N_points, 5, N_energies), dtype=np.float64)
    tgp_stage1_dIdV_LR = np.zeros((N_points, 5, N_energies), dtype=np.float64)
    tgp_stage1_dIdV_RL = np.zeros((N_points, 5, N_energies), dtype=np.float64)

    tasks = [
        (idx, mu, vz, static_params, tgp_barrier_arr, tgp_energies)
        for idx, mu, vz in params_list
    ]

    completed = 0
    t_last_log = time.perf_counter()

    with mp.Pool(processes=args.cores) as pool:
        for idx, dL, dR, dLR, dRL in pool.imap_unordered(_eval_single_point, tasks, chunksize=16):
            tgp_stage1_dIdVl[idx, :, :] = dL
            tgp_stage1_dIdVr[idx, :, :] = dR
            tgp_stage1_dIdV_LR[idx, :, :] = dLR
            tgp_stage1_dIdV_RL[idx, :, :] = dRL

            completed += 1
            if completed % 500 == 0 or completed == N_points:
                t_now = time.perf_counter()
                elapsed = t_now - t_start
                rate = completed / max(elapsed, 0.001)
                eta_s = (N_points - completed) / max(rate, 0.001)
                print(f"[{completed:5d}/{N_points}] Completed ({rate:.1f} pts/s, elapsed {elapsed:.1f}s, ETA {eta_s/60:.1f}m)")

    # Save outputs safely to disk
    print("\nSaving patched Stage 1 arrays to disk...")
    np.save(realization_dir / "tgp_stage1_dIdVl.npy", tgp_stage1_dIdVl)
    np.save(realization_dir / "tgp_stage1_dIdVr.npy", tgp_stage1_dIdVr)
    np.save(realization_dir / "tgp_stage1_dIdV_LR.npy", tgp_stage1_dIdV_LR)
    np.save(realization_dir / "tgp_stage1_dIdV_RL.npy", tgp_stage1_dIdV_RL)
    np.save(realization_dir / "tgp_barrier_arr.npy", tgp_barrier_arr)

    t_total = time.perf_counter() - t_start
    print("==================================================================")
    print(f" [COMPLETE] Successfully patched Stage 1 data in {t_total:.2f} s ({t_total/60:.2f} min).")
    print(f" Destination: {realization_dir}")
    print("==================================================================")


if __name__ == "__main__":
    main()
