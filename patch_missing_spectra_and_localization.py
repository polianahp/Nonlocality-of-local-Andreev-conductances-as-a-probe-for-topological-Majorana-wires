#!/usr/bin/env python3
"""Targeted Patch Script: Computes missing closed-system spectra and localization metrics.

Computes:
  - spectrum_arr.npy               shape: (N_points, 4)
  - topological_gap.npy            shape: (N_points,)
  - weight_localization_arr.npy    shape: (N_points,)
  - mzm_separation_arr.npy         shape: (N_points,)
  - mzm_boundary_confinement_arr.npy shape: (N_points,)
  - mzm_separability_arr.npy       shape: (N_points,)

Designed for high-throughput execution on Harpers Ferry (up to 256 cores).
Checks if the target files already exist on disk, skipping execution unless --force is provided.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# Ensure repository root and src are on sys.path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import helpers as hp


def _eval_single_point(task_args: Tuple[int, float, float, Dict[str, Any]]):
    idx, mu_val, vz_val, static_params = task_args
    t = static_params["t"]
    gamma = static_params["gamma"]
    Delta0 = static_params["Delta0"]
    alpha = static_params["alpha"]
    Ls = static_params["Ls"]
    Vdisx = static_params["Vdisx"]
    solver_type = static_params.get("solver_type", "cpu")
    weight_thresh = static_params.get("weight_threshold", 0.8)

    # 1. Closed-system Hamiltonian diagonalization
    syst_closed = hp.build_system_closed(t, mu_val, gamma, Delta0, vz_val, alpha, Ls, Vdisx)
    evals, evecs = hp.solve_ham(syst_closed, solver_type=solver_type, k=12)

    # 2. Extract 4 closest BdG eigenvalues for spectrum_arr
    spec = hp.sort_spectrum(evals, evecs)
    if len(spec) >= 4:
        idx_closest = np.argsort(np.abs(spec))[:4]
        spec = np.sort(spec[idx_closest])

    # 3. Extract ground-state energy E_0 and closed-system topological gap E_1
    pos_evals = np.sort(evals[evals >= 0])
    e0 = float(pos_evals[0]) if len(pos_evals) > 0 else np.nan
    topogap = float(pos_evals[1]) if len(pos_evals) > 1 else np.nan

    # 4. Compute localization metrics only for near-zero modes (E_0 <= 0.1 meV)
    if not np.isnan(e0) and e0 <= 0.1:
        rho_M1, rho_M2, _ = hp.get_psiM_density(evals, evecs)
        w_loc = float(hp.calc_weight_localization(rho_M1, rho_M2, weight_threshold=weight_thresh))
        m_sep = float(hp.calc_MZM_separation(rho_M1, rho_M2))
        b_conf = float(hp.calc_mzm_boundary_confinement(rho_M1, rho_M2, Ls, pct_thresh=80.0))
        m_separability = float(hp.calc_mzm_separability(rho_M1, rho_M2, continuous=True, auto_orient=True))
    else:
        w_loc = 1.0
        m_sep = 0.0
        b_conf = 0.0
        m_separability = 0.5

    return idx, spec, topogap, w_loc, m_sep, b_conf, m_separability


def resolve_data_root(custom_root: str | None = None) -> Path:
    if custom_root:
        p = Path(custom_root)
        if p.exists():
            return p

    candidates = [
        Path(os.environ.get("SCRATCH", "/scratch/nos00002")) / "nonlocalcond" / "Data" / "Disorder_Realizations",
        Path(os.environ.get("SCRATCH", "/scratch/nos00002")) / "Data" / "Disorder_Realizations",
        REPO_ROOT / "Outputs" / "Data" / "Disorder_Realizations",
        REPO_ROOT / "Data" / "Disorder_Realizations",
    ]
    for c in candidates:
        if c.exists():
            return c
    return REPO_ROOT / "Outputs" / "Data" / "Disorder_Realizations"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch missing closed-system spectra, gap, and localization metrics for a disorder realization."
    )
    parser.add_argument(
        "--disorder_folder",
        type=str,
        required=True,
        help="Disorder folder name, e.g., V0_0_378, V0_0_645, V0_0_872, V0_1_2.",
    )
    parser.add_argument(
        "--realization_index",
        type=int,
        required=True,
        help="Integer index of realization (0..99).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to Disorder_Realizations parent directory.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Number of CPU worker cores for parallel execution (defaults to all available, up to 256).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force recomputation even if files already exist on disk.",
    )

    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    r_name = f"disorder_realization_{args.realization_index}"
    realization_dir = data_root / args.disorder_folder / r_name

    if not realization_dir.exists():
        raise FileNotFoundError(f"Realization directory not found: {realization_dir}")

    required_files = [
        "spectrum_arr.npy",
        "topological_gap.npy",
        "weight_localization_arr.npy",
        "mzm_separation_arr.npy",
    ]

    all_exist = all(
        (realization_dir / f).exists() and (realization_dir / f).stat().st_size > 0
        for f in required_files
    )

    if all_exist and not args.force:
        print(f"[SKIP] All closed-system files ({', '.join(required_files)}) already exist in {realization_dir}.")
        print("Use --force to override and recompute.")
        sys.exit(0)

    t_start = time.perf_counter()
    print("==================================================================")
    print(f" Patching Closed-System Spectra & Localization Metrics")
    print(f" Target Folder:      {realization_dir}")
    print(f" Disorder Strength:  {args.disorder_folder}")
    print(f" Realization Index:  {args.realization_index}")
    print(f"==================================================================")

    # 1. Load parameters
    params_file = realization_dir / "all_params.npz"
    plist_file = realization_dir / "params_list.npy"

    if not params_file.exists():
        raise FileNotFoundError(f"Missing parameter file {params_file}")
    if not plist_file.exists():
        raise FileNotFoundError(f"Missing parameter list file {plist_file}")

    p = np.load(params_file, allow_pickle=True)
    params_list = np.load(plist_file, allow_pickle=True)
    N_points = len(params_list)

    static_params = {
        "t": float(p["t"]),
        "gamma": float(p["gamma"]),
        "Delta0": float(p["Delta0"]),
        "alpha": float(p["alpha"]),
        "Ls": int(p["Ls"]),
        "Vdisx": p["Vdisx"] * float(p["V0"]),
        "weight_threshold": float(p.get("weight_threshold", 0.8)),
        "solver_type": "cpu",
    }

    n_workers = args.cores if args.cores is not None else max(1, min(mp.cpu_count(), 256))
    print(f"Total Grid Points:   {N_points}")
    print(f"Dispatching worker pool across {n_workers} processes...")

    spectrum_arr = np.zeros((N_points, 4), dtype=np.float64)
    topological_gap_arr = np.zeros(N_points, dtype=np.float64)
    weight_localization_arr = np.zeros(N_points, dtype=np.float64)
    mzm_separation_arr = np.zeros(N_points, dtype=np.float64)
    mzm_boundary_confinement_arr = np.zeros(N_points, dtype=np.float64)
    mzm_separability_arr = np.zeros(N_points, dtype=np.float64)

    tasks = [
        (int(params_list[i, 0]), float(params_list[i, 1]), float(params_list[i, 2]), static_params)
        for i in range(N_points)
    ]

    completed = 0
    with mp.Pool(processes=n_workers) as pool:
        for idx, spec, topogap, w_loc, m_sep, b_conf, m_separability in pool.imap_unordered(
            _eval_single_point, tasks, chunksize=16
        ):
            spectrum_arr[idx, :] = spec
            topological_gap_arr[idx] = topogap
            weight_localization_arr[idx] = w_loc
            mzm_separation_arr[idx] = m_sep
            mzm_boundary_confinement_arr[idx] = b_conf
            mzm_separability_arr[idx] = m_separability

            completed += 1
            if completed % 1000 == 0 or completed == N_points:
                t_now = time.perf_counter()
                elapsed = t_now - t_start
                rate = completed / max(elapsed, 0.001)
                eta_s = (N_points - completed) / max(rate, 0.001)
                print(
                    f"[{completed:5d}/{N_points}] Completed ({rate:.1f} pts/s, elapsed {elapsed:.1f}s, ETA {eta_s/60:.1f}m)"
                )

    # 2. Save outputs safely to disk
    print("\nSaving patched arrays to disk...")
    np.save(realization_dir / "spectrum_arr.npy", spectrum_arr)
    np.save(realization_dir / "topological_gap.npy", topological_gap_arr)
    np.save(realization_dir / "weight_localization_arr.npy", weight_localization_arr)
    np.save(realization_dir / "mzm_separation_arr.npy", mzm_separation_arr)
    np.save(realization_dir / "mzm_boundary_confinement_arr.npy", mzm_boundary_confinement_arr)
    np.save(realization_dir / "mzm_separability_arr.npy", mzm_separability_arr)

    # Also save backwards-compatibility aliases matching compute_localization_metrics.py
    np.save(realization_dir / "site_localizations.npy", mzm_boundary_confinement_arr)

    t_total = time.perf_counter() - t_start
    print("==================================================================")
    print(f" [COMPLETE] Successfully patched spectra & localization metrics in {t_total:.2f} s ({t_total/60:.2f} min).")
    print(f" Destination: {realization_dir}")
    print("==================================================================")


if __name__ == "__main__":
    main()
