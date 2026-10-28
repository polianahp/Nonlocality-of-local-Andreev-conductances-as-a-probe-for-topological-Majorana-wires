#!/usr/bin/env python3
"""High-Throughput Parallel Precomputor for Ground-Truth Pfaffian Invariant Masks (`pdi_data.npy`).

Scans `Disorder_Realizations/` across all disorder strengths (V0 = 0.1, 0.378, 0.645, 0.872, 1.2 meV),
identifies realizations missing `pdi_data.npy` (e.g., simulations run with `calc_pfaffian: false`),
and computes the exact topological Pfaffian mask on the fly using `PfaffianSimulator` with decimation
delta_N=0, saving `pdi_data.npy` directly to disk in the canonical 4-column format:
[mu, vz, 0.0, 1.0 if topological else 0.0].

Scales across up to 256 cores on Harpers Ferry.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.Protocol_Elements import DEFAULT_DATA_ROOT


def _realization_sort_key(p: Path) -> int:
    m = re.search(r"(\d+)", p.name)
    return int(m.group(1)) if m else 0


def _compute_single_pfaffian_worker(r_dir_str: str) -> Dict[str, Any]:
    """Worker task: compute and save pdi_data.npy for a single realization."""
    t0 = time.perf_counter()
    r_dir = Path(r_dir_str)
    params_path = r_dir / "all_params.npz"
    if not params_path.exists():
        return {
            "realization": r_dir.name,
            "v0": r_dir.parent.name,
            "status": "error",
            "message": "Missing all_params.npz",
            "elapsed_s": 0.0,
        }

    p = np.load(params_path, allow_pickle=True)
    ts = float(p["t"])
    alphas = float(p["alpha"])
    gamma = float(p["gamma"])
    delta0 = float(p["Delta0"])
    Nx = int(p["Ls"])
    Vdisx = p["Vdisx"] if "Vdisx" in p else None
    V0 = float(p["V0"]) if "V0" in p else 0.0
    delta_N = int(p.get("pfaffian_delta_N", 0))

    mu_vals = np.asarray(p["mu_var"], dtype=np.float64)
    vz_vals = np.asarray(p["Vz_var"], dtype=np.float64)

    from src.helpers import PfaffianSimulator
    from src.pfaffian_invariant import eta_m

    ax = 200.0 / alphas
    ms = 1000.0 * eta_m / (2.0 * ax**2 * ts)
    sim = PfaffianSimulator(
        Nx=Nx,
        Ny=1,
        delta_N=delta_N,
        ax=ax,
        ay=ax,
        ms=ms,
        gamma0=gamma,
        delta0=delta0,
        method="h",
    )
    if Vdisx is not None and np.any(Vdisx != 0):
        sim.load_disorder(Vdisx)

    pdi_rows: List[List[float]] = []
    n_topo = 0
    for mu in mu_vals:
        for vz in vz_vals:
            val = float(np.mean(sim.vPf2Dx(vz, mu, V0, gamma, 0.0)))
            is_topo = (val == 1.0)
            if is_topo:
                n_topo += 1
            pdi_rows.append([float(mu), float(vz), 0.0, 1.0 if is_topo else 0.0])

    pdi_arr = np.array(pdi_rows, dtype=np.float64)
    out_path = r_dir / "pdi_data.npy"
    np.save(out_path, pdi_arr)
    elapsed = time.perf_counter() - t0

    return {
        "realization": r_dir.name,
        "v0": r_dir.parent.name,
        "status": "ok",
        "n_points": len(pdi_rows),
        "n_topo": n_topo,
        "elapsed_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute pdi_data.npy topological ground-truth masks across disorder realizations."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Root directory of Disorder_Realizations (default: auto-detected Data/Disorder_Realizations).",
    )
    parser.add_argument(
        "--v0",
        nargs="*",
        default=None,
        help="Optional V0 folders to target (e.g. V0_0_1 V0_0_378).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 4, 256),
        help="Number of parallel worker processes (default: all available cores, up to 256).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if pdi_data.npy already exists.",
    )
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    if root.name.startswith("V0_"):
        v0_dirs = [root]
    else:
        v0_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("V0_")])

    if args.v0:
        v0_dirs = [d for d in v0_dirs if d.name in args.v0]

    to_process: List[Path] = []
    already_computed = 0

    for v0_dir in v0_dirs:
        r_dirs = sorted(
            [
                d
                for d in v0_dir.iterdir()
                if d.is_dir() and "realization" in d.name and (d / "all_params.npz").exists()
            ],
            key=_realization_sort_key,
        )
        for r_dir in r_dirs:
            pdi_path = r_dir / "pdi_data.npy"
            if pdi_path.exists() and not args.force:
                already_computed += 1
            else:
                to_process.append(r_dir)

    print("==================================================================", flush=True)
    print("Pfaffian Ground-Truth Mask Precomputor", flush=True)
    print(f"Data Root:         {root}", flush=True)
    print(f"Disorder Folders:  {[d.name for d in v0_dirs]}", flush=True)
    print(f"Already Completed: {already_computed} realizations", flush=True)
    print(f"Needs Computation: {len(to_process)} realizations", flush=True)
    print(f"Allocated Workers: {args.workers}", flush=True)
    print("==================================================================", flush=True)

    if not to_process:
        print("[Done] All realizations already contain pdi_data.npy!", flush=True)
        return

    t_start = time.perf_counter()
    num_workers = max(1, min(args.workers, len(to_process)))

    inputs = [str(p) for p in to_process]
    if num_workers == 1:
        results = []
        for i, inp in enumerate(inputs, 1):
            res = _compute_single_pfaffian_worker(inp)
            results.append(res)
            print(
                f"[{i}/{len(inputs)}] {res['v0']}/{res['realization']}: "
                f"{res.get('n_topo', 0)} topo points in {res['elapsed_s']:.2f}s",
                flush=True,
            )
    else:
        with mp.Pool(processes=num_workers) as pool:
            results = []
            for i, res in enumerate(
                pool.imap_unordered(_compute_single_pfaffian_worker, inputs, chunksize=1), 1
            ):
                results.append(res)
                if i % 10 == 0 or i == len(inputs):
                    print(
                        f"[{i}/{len(inputs)}] Completed {res['v0']}/{res['realization']} "
                        f"({res.get('n_topo', 0)} topo pts, {res['elapsed_s']:.1f}s)",
                        flush=True,
                    )

    total_time = time.perf_counter() - t_start
    print("==================================================================", flush=True)
    print(f"[Complete] Finished {len(to_process)} realizations in {total_time:.2f}s "
          f"({total_time/60.0:.2f} min).", flush=True)
    print("==================================================================", flush=True)


if __name__ == "__main__":
    main()
