#!/usr/bin/env python3
"""Run Extra Low-Asymmetry Barrier U-Points for Tdispfaff5.

Computes ONLY the extra barrier conductance evaluations at finite temperature
(e.g., 40 mK and/or 30 mK) across the (mu, Vz) parameter grid, bypassing all
closed BdG Hamiltonian solves, wavefunctions, Pfaffians, and Stage 1 TGP cutter sweeps.

Usage:
  # Dry-run test on first 5 points (completes in ~5-10 seconds):
  python run_extra_barrier_points_tdispfaff5.py --dry-run

  # Full local parallel run (20 workers, 40 mK only):
  python run_extra_barrier_points_tdispfaff5.py --workers 20 --temps 40.0

  # Full local parallel run for both 30 mK and 40 mK:
  python run_extra_barrier_points_tdispfaff5.py --workers 20 --temps 30.0 40.0

  # Merge extra points with the existing base dataset:
  python run_extra_barrier_points_tdispfaff5.py --merge-only
"""

import os
import sys
import time

# 1. THREAD PINNING: Must be set BEFORE importing numpy/scipy/kwant
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import src.helpers as hp

# =============================================================================
# WORKER FUNCTION (Module level for multiprocessing pickling)
# =============================================================================

def worker_barrier_step(
    iter_data: Tuple[int, int, float, float],
    static_params: Dict[str, Any],
    barrier_var_list: np.ndarray,
    target_temps: List[float],
    q_nodes_by_temp: Dict[float, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Computes barrier sweeps exclusively for the specified extra U-points.

    iter_data: (local_idx, global_idx, mu, vz)
    """
    local_idx, global_idx, mu, vz = iter_data

    t = static_params["t"]
    mu_n = static_params["mu_n"]
    Delta0 = static_params["Delta0"]
    gamma = static_params["gamma"]
    alpha = static_params["alpha"]
    Ln = static_params["Ln"]
    Lb = static_params["Lb"]
    Ls = static_params["Ls"]
    mu_leads = static_params["mu_leads"]
    barrier_tot = static_params["barrier0"]
    Vdis_scaled = static_params["Vdis_scaled"]
    solver_type = static_params.get("solver_type", "cpu")

    n_points = len(barrier_var_list)

    # Local pre-allocation per temperature
    res_by_temp: Dict[float, Dict[str, np.ndarray]] = {}
    for T_val in target_temps:
        res_by_temp[T_val] = {
            "b_right_cond_left": np.zeros(n_points, dtype=np.float64),
            "b_right_cond_right": np.zeros(n_points, dtype=np.float64),
            "b_right_GLR": np.zeros(n_points, dtype=np.float64),
            "b_right_GRL": np.zeros(n_points, dtype=np.float64),
            "b_left_cond_left": np.zeros(n_points, dtype=np.float64),
            "b_left_cond_right": np.zeros(n_points, dtype=np.float64),
        }

    for k, b_var in enumerate(barrier_var_list):
        # 1. Vary Right Barrier (UR = b_var, UL = barrier_tot)
        syst_UR = hp.build_system(
            t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz,
            alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
            barrier_l=barrier_tot, barrier_r=b_var, Vdisx=Vdis_scaled
        )
        for T_val in target_temps:
            if T_val > 0.0:
                qe, qw = q_nodes_by_temp[T_val]
                Gmat_UR = hp.calc_conductance_matrix_finite_T(syst_UR, qe, qw, solver_type=solver_type)
            else:
                Gmat_UR = hp.calc_conductance_matrix(syst_UR, eng=0.0, solver_type=solver_type)
            res_by_temp[T_val]["b_right_cond_left"][k] = Gmat_UR[0, 0]
            res_by_temp[T_val]["b_right_cond_right"][k] = Gmat_UR[1, 1]
            res_by_temp[T_val]["b_right_GRL"][k] = Gmat_UR[1, 0]
            res_by_temp[T_val]["b_right_GLR"][k] = Gmat_UR[0, 1]

        # 2. Vary Left Barrier (UL = b_var, UR = barrier_tot)
        syst_UL = hp.build_system(
            t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz,
            alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
            barrier_l=b_var, barrier_r=barrier_tot, Vdisx=Vdis_scaled
        )
        for T_val in target_temps:
            if T_val > 0.0:
                qe, qw = q_nodes_by_temp[T_val]
                Gmat_UL = hp.calc_conductance_matrix_finite_T(syst_UL, qe, qw, solver_type=solver_type)
            else:
                Gmat_UL = hp.calc_conductance_matrix(syst_UL, eng=0.0, solver_type=solver_type)
            res_by_temp[T_val]["b_left_cond_left"][k] = Gmat_UL[0, 0]
            res_by_temp[T_val]["b_left_cond_right"][k] = Gmat_UL[1, 1]

    return {
        "local_idx": local_idx,
        "global_idx": global_idx,
        "results_by_temp": res_by_temp,
    }


# =============================================================================
# DATASET MERGING UTILITY
# =============================================================================

def merge_datasets(base_dir: Path, extra_dir: Path, dest_dir: Path, temps: List[float]):
    """Merges base barrier dataset with extra barrier U-points into a unified, sorted dense dataset."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Merging Datasets ---")
    print(f"Base Directory:  {base_dir}")
    print(f"Extra Directory: {extra_dir}")
    print(f"Destination:     {dest_dir}")

    b_base = np.load(base_dir / "barrier_arr.npy")
    b_extra = np.load(extra_dir / "barrier_arr_extra.npy")

    # Combine barrier arrays and sort
    b_combined = np.concatenate([b_base, b_extra])
    # Find unique elements with tolerance 1e-5
    unique_vals, unique_idx = np.unique(np.round(b_combined, 5), return_index=True)
    sorted_order = np.argsort(b_combined[unique_idx])
    final_barrier_arr = b_combined[unique_idx][sorted_order]

    all_p = np.load(base_dir / "all_params.npz", allow_pickle=True)
    barrier0 = float(all_p["barrier0"]) if "barrier0" in all_p else 2.0
    final_asym_ratios = final_barrier_arr / max(barrier0, 1e-9)

    print(f"Base U-points: {len(b_base)} | Extra U-points: {len(b_extra)} | Combined Unique U-points: {len(final_barrier_arr)}")
    np.save(dest_dir / "barrier_arr.npy", final_barrier_arr)
    np.save(dest_dir / "asym_ratios.npy", final_asym_ratios)

    # Copy all other base metadata and grid files
    for p in base_dir.glob("*.npy"):
        if p.name not in [
            "barrier_arr.npy",
            "asym_ratios.npy",
            "barrier_right_conductance_left_arr.npy",
            "barrier_right_conductance_right_arr.npy",
            "barrier_right_GLR.npy",
            "barrier_right_GRL.npy",
            "barrier_right_GLR_arr.npy",
            "barrier_right_GRL_arr.npy",
            "barrier_left_conductance_left_arr.npy",
            "barrier_left_conductance_right_arr.npy",
            "barrier_left_correlation_arr.npy",
            "barrier_right_correlation_arr.npy",
        ] and not p.name.endswith("mK.npy"):
            if not (dest_dir / p.name).exists():
                (dest_dir / p.name).write_bytes(p.read_bytes())

    for fname in ["all_params.npz", "resolved_params.yaml", "runtime.txt"]:
        if (base_dir / fname).exists():
            (dest_dir / fname).write_bytes((base_dir / fname).read_bytes())

    # Merge arrays per temperature
    for T_val in temps:
        t_tag = f"_{int(T_val)}mK" if T_val == int(T_val) else f"_{T_val:.1f}mK"
        arr_specs = [
            ("barrier_right_conductance_left_arr", ["barrier_right_conductance_left_arr"]),
            ("barrier_right_conductance_right_arr", ["barrier_right_conductance_right_arr"]),
            ("barrier_right_GLR_arr", ["barrier_right_GLR_arr", "barrier_right_GLR"]),
            ("barrier_right_GRL_arr", ["barrier_right_GRL_arr", "barrier_right_GRL"]),
            ("barrier_left_conductance_left_arr", ["barrier_left_conductance_left_arr"]),
            ("barrier_left_conductance_right_arr", ["barrier_left_conductance_right_arr"]),
        ]
        for canonical_name, candidate_stems in arr_specs:
            # Locate base file
            base_file = None
            for stem in candidate_stems:
                for sfx in [f"{t_tag}.npy", ".npy"]:
                    candidate = base_dir / f"{stem}{sfx}"
                    if candidate.exists():
                        base_file = candidate
                        break
                if base_file:
                    break

            # Locate extra file
            extra_file = None
            for stem in candidate_stems:
                for sfx in [f"{t_tag}.npy", ".npy"]:
                    candidate = extra_dir / f"{stem}{sfx}"
                    if candidate.exists():
                        extra_file = candidate
                        break
                if extra_file:
                    break

            if base_file is None or extra_file is None:
                print(f"Warning: {canonical_name} not found in base ({base_file}) or extra ({extra_file}), skipping.")
                continue

            arr_base = np.load(base_file)
            arr_extra = np.load(extra_file)

            if arr_extra.shape[0] != arr_base.shape[0]:
                print(f"  [Notice] Partial run detected ({arr_extra.shape[0]} / {arr_base.shape[0]} points). Aligning by global index...")
                full_extra = np.full((arr_base.shape[0], arr_extra.shape[1]), np.nan, dtype=np.float64)
                p_eval_path = extra_dir / "params_list_evaluated.npy"
                if p_eval_path.exists():
                    p_eval = np.load(p_eval_path)
                    for k_eval, row in enumerate(p_eval):
                        orig_idx = int(row[0])
                        if orig_idx < arr_base.shape[0]:
                            full_extra[orig_idx, :] = arr_extra[k_eval, :]
                arr_extra = full_extra

            arr_combined = np.concatenate([arr_base, arr_extra], axis=1)
            arr_merged = arr_combined[:, unique_idx][:, sorted_order]

            # Save canonical tagged and untagged formats
            out_file_tagged = dest_dir / f"{canonical_name}{t_tag}.npy"
            np.save(out_file_tagged, arr_merged)
            out_file_untagged = dest_dir / f"{canonical_name}.npy"
            np.save(out_file_untagged, arr_merged)

            if "GLR" in canonical_name:
                np.save(dest_dir / f"barrier_right_GLR{t_tag}.npy", arr_merged)
                np.save(dest_dir / "barrier_right_GLR.npy", arr_merged)
            elif "GRL" in canonical_name:
                np.save(dest_dir / f"barrier_right_GRL{t_tag}.npy", arr_merged)
                np.save(dest_dir / "barrier_right_GRL.npy", arr_merged)

            print(f"Saved merged: {out_file_tagged.name} {arr_merged.shape}")

    print(f"Merge Complete! Saved to: {dest_dir}\n")


# =============================================================================
# MAIN DRIVER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Extra Low-Asymmetry Barrier U-Points for Tdispfaff5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Outputs/Data/Tdis_pfaff5_data/Tdispfaff5_40mK",
        help="Path to the existing base realization directory (contains all_params.npz and Vdisx.npy)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="Outputs/Data/Tdis_pfaff5_data/Tdispfaff5_extra_u20",
        help="Path to output directory for the extra U-points data",
    )
    parser.add_argument(
        "--asym-min",
        type=float,
        default=-5.0,
        help="Minimum gate asymmetry ratio (delta_U / barrier0)",
    )
    parser.add_argument(
        "--asym-max",
        type=float,
        default=5.0,
        help="Maximum gate asymmetry ratio (delta_U / barrier0)",
    )
    parser.add_argument(
        "--num-u-points",
        type=int,
        default=20,
        help="Number of extra barrier U-points to evaluate across [asym_min, asym_max]",
    )
    parser.add_argument(
        "--temps",
        type=float,
        nargs="+",
        default=[40.0],
        help="Temperatures in mK to evaluate (e.g. 40.0 or 30.0 40.0)",
    )
    parser.add_argument(
        "--quad-nodes",
        type=int,
        default=8,
        help="Number of Gauss-Legendre quadrature nodes for thermal broadening",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(20, max(1, mp.cpu_count() - 2)),
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test mode: executes only the first 5 (mu, Vz) points to verify timing and output format",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional ceiling on number of (mu, Vz) grid points to evaluate",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="After execution, automatically merge with base dataset into a combined dense folder",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip simulation and only merge existing extra data with base data",
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default="Outputs/Data/Tdis_pfaff5_data/Tdispfaff5_40mK_dense",
        help="Destination directory for merged dataset when --merge or --merge-only is used",
    )

    args = parser.parse_args()

    data_dir = REPO_ROOT / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    out_dir = REPO_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    merged_dir = REPO_ROOT / args.merged_dir if not Path(args.merged_dir).is_absolute() else Path(args.merged_dir)

    if args.merge_only:
        merge_datasets(data_dir, out_dir, merged_dir, args.temps)
        return

    # Verify input data directory
    if not (data_dir / "all_params.npz").exists() or not (data_dir / "Vdisx.npy").exists():
        print(f"[ERROR] Required base data files not found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(" EXTRA LOW-ASYMMETRY BARRIER U-POINTS SIMULATION")
    print(f" Target Realization: {data_dir.name}")
    print(f" Source Directory:   {data_dir}")
    print(f" Output Directory:   {out_dir}")
    print(f" Asymmetry Range:    [{args.asym_min}, {args.asym_max}] (20 points)")
    print(f" Temperatures:       {args.temps} mK (quadrature nodes: {args.quad_nodes})")
    print(f" Parallel Workers:   {args.workers}")
    print(f" Mode:               {'DRY-RUN (5 test points)' if args.dry_run else 'FULL RUN'}")
    print("=" * 70)

    # Load base parameters
    all_p = np.load(data_dir / "all_params.npz", allow_pickle=True)
    vdis_raw = np.load(data_dir / "Vdisx.npy")
    params_list = np.load(data_dir / "params_list.npy")

    v0 = float(all_p["V0"])
    vdis_scaled = vdis_raw * v0
    barrier0 = float(all_p["barrier0"])
    t_hop = float(all_p["t"])
    mu_leads = float(all_p.get("mu_leads", t_hop)) if all_p.get("mu_leads") is not None else t_hop

    # Define extra U-points
    extra_asym_ratios = np.linspace(args.asym_min, args.asym_max, args.num_u_points)
    extra_barrier_arr = extra_asym_ratios * barrier0

    print(f"\nExtra Asymmetry Ratios a = Delta_U / barrier0:\n  {extra_asym_ratios}")
    print(f"Extra Barrier Potentials U_var (meV):\n  {extra_barrier_arr}\n")

    # Static parameters dict
    static_params = {
        "t": t_hop,
        "mu_n": float(all_p.get("mu_n", 0.0)),
        "Delta0": float(all_p["Delta0"]),
        "gamma": float(all_p["gamma"]),
        "alpha": float(all_p["alpha"]),
        "Ln": int(all_p.get("Ln", 0)),
        "Lb": int(all_p.get("Lb", 3)),
        "Ls": int(all_p.get("Ls", 300)),
        "mu_leads": mu_leads,
        "barrier0": barrier0,
        "Vdis_scaled": vdis_scaled,
        "solver_type": "cpu",
    }

    # Precompute quadrature nodes
    q_nodes_by_temp: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for T_val in args.temps:
        if T_val > 0.0:
            qe, qw = hp.get_thermal_quadrature_nodes(T_mK=T_val, n_nodes=args.quad_nodes, bias_meV=0.0)
            q_nodes_by_temp[T_val] = (qe, qw)
        else:
            q_nodes_by_temp[T_val] = (None, None)

    # Restrict points for dry-run
    if args.dry_run:
        eval_params = params_list[:5]
    elif args.max_points is not None:
        eval_params = params_list[:args.max_points]
    else:
        eval_params = params_list

    n_total = len(eval_params)
    print(f"Evaluating {n_total} (mu, Vz) points...")

    # Build tasks with local index to guarantee contiguous 0..n_total-1 indexing
    eval_tasks = [
        (k, int(p[0]), float(p[1]), float(p[2]))
        for k, p in enumerate(eval_params)
    ]

    t_start = time.perf_counter()

    # Worker dispatch
    worker_fn = partial(
        worker_barrier_step,
        static_params=static_params,
        barrier_var_list=extra_barrier_arr,
        target_temps=args.temps,
        q_nodes_by_temp=q_nodes_by_temp,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Allocate result matrices: (N_params, num_u_points)
    num_u = len(extra_barrier_arr)
    conductance_data: Dict[float, Dict[str, np.ndarray]] = {
        T_val: {
            "b_r_l": np.zeros((n_total, num_u), dtype=np.float64),
            "b_r_r": np.zeros((n_total, num_u), dtype=np.float64),
            "b_r_glr": np.zeros((n_total, num_u), dtype=np.float64),
            "b_r_grl": np.zeros((n_total, num_u), dtype=np.float64),
            "b_l_l": np.zeros((n_total, num_u), dtype=np.float64),
            "b_l_r": np.zeros((n_total, num_u), dtype=np.float64),
        }
        for T_val in args.temps
    }

    # Run pool
    with mp.Pool(processes=args.workers) as pool:
        for res in tqdm(
            pool.imap(worker_fn, eval_tasks, chunksize=1),
            total=n_total,
            desc=f"Extra U-Points ({args.num_u_points} pts)",
        ):
            local_idx = int(res["local_idx"])
            for T_val in args.temps:
                t_res = res["results_by_temp"][T_val]
                conductance_data[T_val]["b_r_l"][local_idx, :] = t_res["b_right_cond_left"]
                conductance_data[T_val]["b_r_r"][local_idx, :] = t_res["b_right_cond_right"]
                conductance_data[T_val]["b_r_glr"][local_idx, :] = t_res["b_right_GLR"]
                conductance_data[T_val]["b_r_grl"][local_idx, :] = t_res["b_right_GRL"]
                conductance_data[T_val]["b_l_l"][local_idx, :] = t_res["b_left_cond_left"]
                conductance_data[T_val]["b_l_r"][local_idx, :] = t_res["b_left_cond_right"]

    t_elapsed = time.perf_counter() - t_start
    print(f"\nExecution Complete in {t_elapsed:.2f} s ({t_elapsed/60:.2f} min).")

    # Save extra dataset
    np.save(out_dir / "barrier_arr_extra.npy", extra_barrier_arr)
    np.save(out_dir / "asym_ratios_extra.npy", extra_asym_ratios)
    np.save(out_dir / "params_list_evaluated.npy", eval_params)

    for T_val in args.temps:
        t_tag = f"_{int(T_val)}mK" if T_val == int(T_val) else f"_{T_val:.1f}mK"
        cd = conductance_data[T_val]
        np.save(out_dir / f"barrier_right_conductance_left_arr{t_tag}.npy", cd["b_r_l"])
        np.save(out_dir / f"barrier_right_conductance_right_arr{t_tag}.npy", cd["b_r_r"])
        np.save(out_dir / f"barrier_right_GLR_arr{t_tag}.npy", cd["b_r_glr"])
        np.save(out_dir / f"barrier_right_GRL_arr{t_tag}.npy", cd["b_r_grl"])
        np.save(out_dir / f"barrier_left_conductance_left_arr{t_tag}.npy", cd["b_l_l"])
        np.save(out_dir / f"barrier_left_conductance_right_arr{t_tag}.npy", cd["b_l_r"])

    # Write runtime and parameters log
    meta = {
        "target_realization": data_dir.name,
        "n_evaluated_points": n_total,
        "num_u_points": args.num_u_points,
        "asym_range": [args.asym_min, args.asym_max],
        "barrier_arr_extra": extra_barrier_arr.tolist(),
        "temperatures_mK": args.temps,
        "quadrature_nodes": args.quad_nodes,
        "workers": args.workers,
        "wall_time_seconds": round(t_elapsed, 2),
        "wall_time_minutes": round(t_elapsed / 60.0, 2),
    }
    with open(out_dir / "extra_u_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(out_dir / "runtime.txt", "w") as f:
        f.write(f"Extra U-Points Simulation: {data_dir.name}\n")
        f.write(f"Total Runtime: {t_elapsed:.2f} s ({t_elapsed/60:.2f} min)\n")
        f.write(f"Points Evaluated: {n_total}\n")

    print(f"Results saved to: {out_dir}")

    if args.merge:
        merge_datasets(data_dir, out_dir, merged_dir, args.temps)


if __name__ == "__main__":
    main()
