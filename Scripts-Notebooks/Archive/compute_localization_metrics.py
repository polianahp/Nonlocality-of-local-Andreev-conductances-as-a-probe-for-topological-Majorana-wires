#!/usr/bin/env python3
"""
Compute Localization Metrics (Boundary Confinement, Support Span, and Normalized Overlap)
retroactively for completed simulation datasets.

Does NOT store wave functions; only persists the computed scalar arrays to disk.
"""

import os
# Prevent thread thrashing across parallel workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import argparse
import logging
from pathlib import Path
import multiprocessing as mp
import numpy as np

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from src.config import PathConfigs
import src.helpers as hp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIRS = [
    "Tdis_pfaff5",
    "Tdis_pfaff5_V0_0_0",
    "Tdis_pfaff5_V0_0_1",
    "Tdis_pfaff5_V0_0_91",
    "Tdis_pfaff5_V0_0_378",
    "Tdis_pfaff5_V0_0_645",
    "Tdis_pfaff5_V0_0_872",
]


def _worker_point_task(task_args):
    """
    Worker task: diagonalizes closed system for 1 point and computes the localization metrics.
    Discards wavefunctions immediately to preserve memory.
    """
    idx, mu_val, vz_val, static_params = task_args
    try:
        syst_closed = hp.build_system_closed(
            static_params['t'],
            mu_val,
            static_params['gamma'],
            static_params['Delta0'],
            vz_val,
            static_params['alpha'],
            static_params['Ls'],
            static_params['Vdisx'],
            a=1
        )
        evals, evecs = hp.solve_ham(syst_closed, k=2, solver_type='cpu')
        rho_M1, rho_M2, _ = hp.get_psiM_density(evals, evecs)

        # 1. Normalized Boundary Confinement: 1 - (n / Ls)
        b_conf = hp.calc_mzm_boundary_confinement(rho_M1, rho_M2, static_params['Ls'], pct_thresh=80.0)

        # 2. Density Support Span: (max(idx) - min(idx)) / Ls
        s_span = hp.calc_mzm_density_support_span(rho_M1, rho_M2, weight_threshold=0.9)

        # 3. Normalized MZM Overlap: int(u_l * u_r) / int(u_l + u_r)
        n_overlap = hp.calc_normalized_mzm_overlap(rho_M1, rho_M2)

        # 4. MZM Separability: max_x min(CDF_L(x), CDF_R(x))
        separability = hp.calc_mzm_separability(rho_M1, rho_M2, continuous=True, auto_orient=True)

        return idx, b_conf, s_span, n_overlap, separability
    except Exception as e:
        logger.error(f"Error computing point {idx} (mu={mu_val}, vz={vz_val}): {e}")
        return idx, np.nan, np.nan, np.nan, np.nan


def process_dataset(data_dir: Path, n_workers: int = None, test_limit: int = None, all_points: bool = False):
    """
    Computes localization metrics for a single dataset folder.
    """
    if n_workers is None:
        n_workers = max(1, min(mp.cpu_count() - 2, 20))

    logger.info(f"Processing data directory: {data_dir} using {n_workers} worker processes")
    params_file = data_dir / "all_params.npz"
    params_list_file = data_dir / "params_list.npy"

    if not params_file.exists():
        raise FileNotFoundError(f"Missing parameter file: {params_file}")
    if not params_list_file.exists():
        raise FileNotFoundError(f"Missing parameter list file: {params_list_file}")

    params = np.load(params_file, allow_pickle=True)
    params_list = np.load(params_list_file, allow_pickle=True)
    total_points = len(params_list)

    static_params = {
        't': float(params['t']),
        'gamma': float(params['gamma']),
        'Delta0': float(params['Delta0']),
        'alpha': float(params['alpha']),
        'Ls': int(params['Ls']),
        'Vdisx': params['Vdisx'] * float(params['V0'])
    }

    num_eval_points = total_points if test_limit is None else min(test_limit, total_points)
    logger.info(f"Preparing {num_eval_points} points for calculation...")

    spectrum_file = data_dir / "spectrum_arr.npy"
    spectrum_arr = None
    if spectrum_file.exists() and not all_points:
        spectrum_arr = np.load(spectrum_file, allow_pickle=True)
        logger.info("Found spectrum_arr.npy: applying E_0 <= 0.1 meV condition (matching main_parallel.py)")

    boundary_confinement = np.zeros(num_eval_points)
    density_support_span = np.ones(num_eval_points)  # Default 1.0 (matching main_parallel.py default for non-zero modes)
    normalized_overlap = np.zeros(num_eval_points)
    separability = np.full(num_eval_points, 0.5)     # Default 0.5 (raw unseparated value for bulk/uniform modes)

    tasks = []
    skipped_high_energy = 0
    for i in range(num_eval_points):
        idx = int(params_list[i, 0])
        mu_val = float(params_list[i, 1])
        vz_val = float(params_list[i, 2])
        if spectrum_arr is not None and i < len(spectrum_arr):
            e0 = spectrum_arr[i, 2]
            if e0 > 0.1:
                skipped_high_energy += 1
                continue
        tasks.append((idx, mu_val, vz_val, static_params))

    logger.info(f"Evaluating {len(tasks)} near-zero energy points ({skipped_high_energy} skipped due to E_0 > 0.1 meV)...")

    start_time = time.time()
    completed = 0
    num_tasks = len(tasks)
    log_interval = max(1, num_tasks // 10) if num_tasks > 0 else 1

    # Use multiprocessing pool to parallelize
    if num_tasks > 0:
        with mp.Pool(processes=n_workers) as pool:
            for idx, b_conf, s_span, n_overlap, sep in pool.imap_unordered(_worker_point_task, tasks, chunksize=2):
                boundary_confinement[idx] = b_conf
                density_support_span[idx] = s_span
                normalized_overlap[idx] = n_overlap
                separability[idx] = sep

                completed += 1
                if completed % log_interval == 0 or completed == num_tasks:
                    elapsed = time.time() - start_time
                    pts_per_sec = completed / elapsed if elapsed > 0 else 0
                    logger.info(f"  Progress: {completed}/{num_tasks} points ({100.0*completed/num_tasks:.1f}%) "
                                f"- {pts_per_sec:.1f} pts/sec")

    total_time = time.time() - start_time
    logger.info(f"Finished {num_eval_points} points in {total_time:.2f}s ({num_eval_points/max(0.01, total_time):.2f} pts/sec)")

    # Print summary statistics
    logger.info(f"Summary Statistics for {data_dir.name}:")
    logger.info(f"  Boundary Confinement: mean={np.nanmean(boundary_confinement):.4f}, "
                f"min={np.nanmin(boundary_confinement):.4f}, max={np.nanmax(boundary_confinement):.4f}")
    logger.info(f"  Density Support Span: mean={np.nanmean(density_support_span):.4f}, "
                f"min={np.nanmin(density_support_span):.4f}, max={np.nanmax(density_support_span):.4f}")
    logger.info(f"  Normalized Overlap:   mean={np.nanmean(normalized_overlap):.4f}, "
                f"min={np.nanmin(normalized_overlap):.4f}, max={np.nanmax(normalized_overlap):.4f}")
    logger.info(f"  MZM Separability:     mean={np.nanmean(separability):.4f}, "
                f"min={np.nanmin(separability):.4f}, max={np.nanmax(separability):.4f}")

    if test_limit is None:
        # Save arrays to data directory
        logger.info(f"Saving computed metric arrays to {data_dir}...")
        np.save(data_dir / "mzm_boundary_confinement_arr.npy", boundary_confinement)
        np.save(data_dir / "mzm_density_support_span_arr.npy", density_support_span)
        np.save(data_dir / "mzm_normalized_overlap_arr.npy", normalized_overlap)
        np.save(data_dir / "mzm_separability_arr.npy", separability)

        # Backwards compatibility saves
        np.save(data_dir / "site_localizations.npy", boundary_confinement)
        np.save(data_dir / "weight_localization_arr.npy", density_support_span)
        logger.info(f"All metric arrays successfully saved for {data_dir.name}")
    else:
        logger.info(f"Test batch completed. Arrays NOT saved to avoid overwriting full data with partial batch.")

    return {
        'boundary_confinement': boundary_confinement,
        'density_support_span': density_support_span,
        'normalized_overlap': normalized_overlap,
        'separability': separability
    }


def main():
    parser = argparse.ArgumentParser(description="Retroactively compute MZM localization metrics across parameter grids")
    parser.add_argument("data_dirs", nargs="*", help="List of dataset directory names (relative to Outputs/Data or absolute)")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--test_limit", type=int, default=None, help="If set, only runs the first N points as a test batch")
    parser.add_argument("--all_points", action="store_true", help="Force evaluating all points regardless of E_0 gap")
    args = parser.parse_args()

    dirs_to_process = args.data_dirs if args.data_dirs else DEFAULT_DATA_DIRS

    for d in dirs_to_process:
        p_dir = Path(d)
        if not p_dir.is_absolute():
            p_dir = PathConfigs.DATA / d

        if not p_dir.exists():
            logger.error(f"Directory {p_dir} does not exist. Skipping.")
            continue

        try:
            process_dataset(p_dir, n_workers=args.workers, test_limit=args.test_limit, all_points=args.all_points)
        except Exception as e:
            logger.error(f"Failed processing {p_dir}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
