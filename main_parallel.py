import os
import time

# THREAD CONTROL: Must be set BEFORE importing numpy/scipy/kwant
# Prevents oversubscription where each process tries to use all cores.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import math
import numpy as np
from scipy import linalg as LA
import kwant
import tinyarray
import multiprocessing as mp
from tqdm import tqdm
import src.helpers as hp
import src.legendre_curvature as lc
from src.cpu_broadening import get_broaden_operator
from pathlib import Path
from src.config import PathConfigs
import itertools as itr
from functools import partial
import argparse
import scipy.sparse.linalg as sla
import sys
from typing import Any, Dict, Tuple, List, Optional
import xarray as xr
from omegaconf import OmegaConf, DictConfig
from src.parameter_handler import ConfigManager, SimulationState


# =============================================================================
# MODULAR SLICE SUBROUTINES
# =============================================================================

def compute_spectral_slice(mu: float, vz: float, config: Any, state: SimulationState) -> xr.Dataset:
    """
    Evaluates BdG eigenvalues and optional MZM localization metrics
    for a single (mu, vz) point slice in the closed proximitized nanowire.
    """
    res = config.simulation.resolution
    incl = config.calculation_includes
    
    syst_closed = hp.build_system_closed(
        t=state.t,
        mu=mu,
        gamma=state.gamma,
        Delta0=state.Delta0,
        V_z=vz,
        alpha=state.alpha,
        Ls=state.Ls,
        Vdisx=state.Vdisx
    )
    evals, evecs = hp.solve_ham(syst_closed, k=res.num_eigenvalues)
    spectrum = hp.sort_spectrum(evals, evecs)
    idx_closest = np.argsort(np.abs(spectrum))[:4]
    spectrum_4 = np.sort(spectrum[idx_closest])

    pos_evals = np.sort(evals[evals >= 0])
    energy_0 = float(pos_evals[0]) if len(pos_evals) > 0 else np.nan
    topological_gap = float(pos_evals[1]) if len(pos_evals) > 1 else np.nan

    weight_localization = 0.0
    mzm_separation = 0.0
    if incl.localization:
        rho_M1, rho_M2, _ = hp.get_psiM_density(evals, evecs)
        weight_localization = float(hp.calc_weight_localization(rho_M1, rho_M2, weight_threshold=res.weight_threshold))
        mzm_separation = float(hp.calc_MZM_separation(rho_M1, rho_M2))

    data_vars = {
        "energy_0": (["chemical_potential", "zeeman_field"], [[energy_0]]),
        "topological_gap": (["chemical_potential", "zeeman_field"], [[topological_gap]]),
        "spectrum": (["chemical_potential", "zeeman_field", "eigenvalue_index"], spectrum_4.reshape(1, 1, 4)),
    }
    if incl.localization:
        data_vars["weight_localization"] = (["chemical_potential", "zeeman_field"], [[weight_localization]])
        data_vars["mzm_separation"] = (["chemical_potential", "zeeman_field"], [[mzm_separation]])

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "chemical_potential": [float(mu)],
            "zeeman_field": [float(vz)],
            "eigenvalue_index": np.arange(4, dtype=int),
        }
    )


def compute_barrier_sweep_slice(
    mu: float,
    vz: float,
    config: Any,
    state: SimulationState,
    q_nodes: Dict[float, Tuple[Any, Any]]
) -> xr.Dataset:
    """
    Evaluates zero-bias asymmetric barrier conductance sweeps across temperatures
    and stationary baseline conductances for a single (mu, vz) point slice.
    """
    res = config.simulation.resolution
    target_temps = [float(T) for T in res.temperatures]
    
    Gmat_da = hp.initialize_conductance_dataarray(
        mu_set=[float(mu)],
        Vz_set=[float(vz)],
        target_temps=target_temps,
        G_n_values=state.target_GNs,
        baseline_Gn=state.baseline_GNs,
    )

    t = state.t
    alpha = state.alpha
    Delta0 = state.Delta0
    gamma = state.gamma
    mu_n = state.mu_n
    Ln = state.Ln
    Lb = state.Lb
    Ls = state.Ls
    mu_leads = state.mu_leads
    Vdisx = state.Vdisx

    for u_base in state.baseline_GNs:
        ul_base = float(state.stage1_cutters.sel(barrier_side="left", G_n=u_base).values.item())
        ur_base = float(state.stage1_cutters.sel(barrier_side="right", G_n=u_base).values.item())

        # Sweep Right Barrier (UL = ul_base, UR varies)
        for gn in state.target_GNs:
            ur_var = float(state.stage1_cutters.sel(barrier_side="right", G_n=gn).values.item())
            syst_UR = hp.build_system(
                t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz,
                alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
                barrier_l=ul_base, barrier_r=ur_var, Vdisx=Vdisx
            )
            for T_val in target_temps:
                if T_val > 0.0:
                    qe, qw = q_nodes[T_val]
                    g_mat = hp.calc_conductance_matrix_finite_T(syst_UR, qe, qw)
                else:
                    g_mat = hp.calc_conductance_matrix(syst_UR, eng=0.0)
                Gmat_da.loc[dict(
                    chemical_potential=float(mu),
                    zeeman_field=float(vz),
                    temperature=T_val,
                    baseline_Gn=u_base,
                    swept_barrier="right",
                    G_n=gn
                )] = g_mat

        # Sweep Left Barrier (UL varies, UR = ur_base)
        for gn in state.target_GNs:
            ul_var = float(state.stage1_cutters.sel(barrier_side="left", G_n=gn).values.item())
            syst_UL = hp.build_system(
                t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz,
                alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
                barrier_l=ul_var, barrier_r=ur_base, Vdisx=Vdisx
            )
            for T_val in target_temps:
                if T_val > 0.0:
                    qe, qw = q_nodes[T_val]
                    g_mat = hp.calc_conductance_matrix_finite_T(syst_UL, qe, qw)
                else:
                    g_mat = hp.calc_conductance_matrix(syst_UL, eng=0.0)
                Gmat_da.loc[dict(
                    chemical_potential=float(mu),
                    zeeman_field=float(vz),
                    temperature=T_val,
                    baseline_Gn=u_base,
                    swept_barrier="left",
                    G_n=gn
                )] = g_mat

    return xr.Dataset({"Gmat": Gmat_da})


def compute_tgp_stage1_slice(
    mu: float,
    vz: float,
    config: Any,
    state: SimulationState,
    quad_dict: Dict[float, Any]
) -> xr.Dataset:
    """
    Evaluates the 21 without-replacement cutter gate harmonics (1w, 2w, 3w)
    via Tanh-Legendre / Double-Exponential quadrature for a single (mu, vz) point slice.
    """
    res = config.simulation.resolution
    target_temps = [float(T) for T in res.temperatures]
    num_pairs = len(state.tgp_stage1_pairs.cutter_pair_index)

    l_1w = np.zeros((num_pairs, 1, 1, len(target_temps)), dtype=float)
    r_1w = np.zeros((num_pairs, 1, 1, len(target_temps)), dtype=float)
    l_2w = np.zeros((num_pairs, 1, 1, len(target_temps)), dtype=float)
    r_2w = np.zeros((num_pairs, 1, 1, len(target_temps)), dtype=float)
    l_3w = np.zeros((num_pairs, 1, 1, len(target_temps)), dtype=float)
    r_3w = np.zeros((num_pairs, 1, 1, len(target_temps)), dtype=float)

    t = state.t
    alpha = state.alpha
    Delta0 = state.Delta0
    gamma = state.gamma
    mu_n = state.mu_n
    Ln = state.Ln
    Lb = state.Lb
    Ls = state.Ls
    mu_leads = state.mu_leads
    Vdisx = state.Vdisx
    h_mode = config.hamiltonian_mode

    for k in range(num_pairs):
        b_l = float(state.tgp_stage1_pairs.V_leftcutter.values[k])
        b_r = float(state.tgp_stage1_pairs.V_rightcutter.values[k])

        syst_pair = hp.build_system(
            t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz,
            alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
            barrier_l=b_l, barrier_r=b_r, Vdisx=Vdisx
        )

        for t_idx, T_val in enumerate(target_temps):
            if T_val == 0.0:
                G0 = hp.calc_conductance_matrix(syst_pair, eng=0.0)
                l_1w[k, 0, 0, t_idx] = G0[0, 0]
                r_1w[k, 0, 0, t_idx] = G0[1, 1]
                l_2w[k, 0, 0, t_idx] = G0[1, 0]
                r_2w[k, 0, 0, t_idx] = G0[0, 1]
                l_3w[k, 0, 0, t_idx] = 0.0
                r_3w[k, 0, 0, t_idx] = 0.0
            else:
                quad = quad_dict[T_val]
                harm = lc.compute_harmonics_single_system(syst_pair, quad, hamiltonian_mode=h_mode)
                l_1w[k, 0, 0, t_idx] = harm["L_1w"]
                r_1w[k, 0, 0, t_idx] = harm["R_1w"]
                l_2w[k, 0, 0, t_idx] = harm["RL_2w"]
                r_2w[k, 0, 0, t_idx] = harm["LR_2w"]
                l_3w[k, 0, 0, t_idx] = harm["L_3w"]
                r_3w[k, 0, 0, t_idx] = harm["R_3w"]

    return xr.Dataset(
        data_vars={
            "L_1w": (["cutter_pair_index", "zeeman_field", "chemical_potential", "temperature"], l_1w),
            "R_1w": (["cutter_pair_index", "zeeman_field", "chemical_potential", "temperature"], r_1w),
            "L_2w_nl": (["cutter_pair_index", "zeeman_field", "chemical_potential", "temperature"], l_2w),
            "R_2w_nl": (["cutter_pair_index", "zeeman_field", "chemical_potential", "temperature"], r_2w),
            "L_3w": (["cutter_pair_index", "zeeman_field", "chemical_potential", "temperature"], l_3w),
            "R_3w": (["cutter_pair_index", "zeeman_field", "chemical_potential", "temperature"], r_3w),
        },
        coords={
            "cutter_pair_index": state.tgp_stage1_pairs.cutter_pair_index.values,
            "zeeman_field": [float(vz)],
            "chemical_potential": [float(mu)],
            "temperature": target_temps,
            "G_n_left": ("cutter_pair_index", state.tgp_stage1_pairs.G_n_left.values),
            "G_n_right": ("cutter_pair_index", state.tgp_stage1_pairs.G_n_right.values),
            "V_leftcutter": ("cutter_pair_index", state.tgp_stage1_pairs.V_leftcutter.values),
            "V_rightcutter": ("cutter_pair_index", state.tgp_stage1_pairs.V_rightcutter.values),
        }
    )


def compute_tgp_stage2_slice(
    mu: float,
    vz: float,
    config: Any,
    state: SimulationState,
    Vbias_arr: np.ndarray
) -> xr.Dataset:
    """
    Evaluates symmetric mu-compensated cutter pairs across the bias grid Vbias_arr
    for a single (mu, vz) point slice.
    """
    res = config.simulation.resolution
    target_temps = [float(T) for T in res.temperatures]
    stage2_da = state.stage2_cutters
    if stage2_da is None:
        raise ValueError("state.stage2_cutters must be calibrated before compute_tgp_stage2_slice")

    target_gns = stage2_da.coords["G_n"].values
    num_cutters = len(target_gns)
    n_bias = len(Vbias_arr)

    g_ll = np.full((num_cutters, 1, 1, len(target_temps), n_bias), np.nan, dtype=float)
    g_rr = np.full((num_cutters, 1, 1, len(target_temps), n_bias), np.nan, dtype=float)
    g_lr = np.full((num_cutters, 1, 1, len(target_temps), n_bias), np.nan, dtype=float)
    g_rl = np.full((num_cutters, 1, 1, len(target_temps), n_bias), np.nan, dtype=float)

    t = state.t
    alpha = state.alpha
    Delta0 = state.Delta0
    gamma = state.gamma
    mu_n = state.mu_n
    Ln = state.Ln
    Lb = state.Lb
    Ls = state.Ls
    mu_leads = state.mu_leads
    Vdisx = state.Vdisx
    k_B = state.k_B
    mu_coord = float(np.round(float(mu), 2))

    for c_idx, gn in enumerate(target_gns):
        b_l = float(stage2_da.sel(G_n=float(gn), chemical_potential=mu_coord, barrier_side="left").item())
        b_r = float(stage2_da.sel(G_n=float(gn), chemical_potential=mu_coord, barrier_side="right").item())

        syst_tgp2 = hp.build_system(
            t=t, mu=mu, mu_n=mu_n, Delta0=Delta0, gamma=gamma, V_z=vz,
            alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=mu_leads,
            barrier_l=b_l, barrier_r=b_r, Vdisx=Vdisx
        )
        dL, dR, dLR, dRL, _ = hp.calc_dIdV(syst_tgp2, Vbias_arr, show_progress=False)

        for t_idx, T_val in enumerate(target_temps):
            if T_val == 0.0:
                g_ll[c_idx, 0, 0, t_idx, :] = dL
                g_rr[c_idx, 0, 0, t_idx, :] = dR
                g_lr[c_idx, 0, 0, t_idx, :] = dLR
                g_rl[c_idx, 0, 0, t_idx, :] = dRL
            else:
                M_broaden = get_broaden_operator(Vbias_arr, T_val, k_B=k_B)
                g_ll[c_idx, 0, 0, t_idx, :] = dL @ M_broaden
                g_rr[c_idx, 0, 0, t_idx, :] = dR @ M_broaden
                g_lr[c_idx, 0, 0, t_idx, :] = dLR @ M_broaden
                g_rl[c_idx, 0, 0, t_idx, :] = dRL @ M_broaden

    assert not (
        np.isnan(g_ll).any() or np.isnan(g_rr).any()
        or np.isnan(g_lr).any() or np.isnan(g_rl).any()
    ), "TGP Stage 2 slice produced NaN conductances"

    return xr.Dataset(
        data_vars={
            "g_ll": (["cutter_pair_index", "zeeman_field", "chemical_potential", "temperature", "bias"], g_ll),
            "g_rr": (["cutter_pair_index", "zeeman_field", "chemical_potential", "temperature", "bias"], g_rr),
            "g_lr": (["cutter_pair_index", "zeeman_field", "chemical_potential", "temperature", "bias"], g_lr),
            "g_rl": (["cutter_pair_index", "zeeman_field", "chemical_potential", "temperature", "bias"], g_rl),
        },
        coords={
            "cutter_pair_index": np.arange(num_cutters, dtype=int),
            "zeeman_field": [float(vz)],
            "chemical_potential": [float(mu)],
            "temperature": target_temps,
            "bias": Vbias_arr,
            "target_gn": ("cutter_pair_index", target_gns),
        }
    )


# =============================================================================
# WORKER FUNCTIONS (Must be defined at module level for pickling)
# =============================================================================

def worker_simulation_step(iter_data: Tuple, config: Any, state: SimulationState, worker_context: Dict) -> Dict[str, Any]:
    """
    Worker function for the main transport and spectral parameter loop (Loop 1).
    Delegates to modular subroutines and returns point-slice datasets.

    Parameters
    ----------
    iter_data : Tuple
        Tuple of (index, chemical_potential, zeeman_field).
    config : Any
        Top-level configuration object (EasyDict/dict).
    state : SimulationState
        Realization state containing wire parameters and calibrated cutter arrays.
    worker_context : Dict
        Dictionary of precomputed worker-shared numerical objects to avoid redundant
        re-allocation inside parallel workers:
        - "q_nodes": Dict mapping temperature to thermal quadrature nodes/weights.
        - "quad_dict": Dict mapping temperature to TGP Stage 1 Legendre/DE nodes/weights.
        - "dynamic_energies": Array of finite-bias energy evaluation points.
    """
    i = int(iter_data[0])
    mu = float(iter_data[1])
    vz = float(iter_data[2])
    
    incl = config.calculation_includes
    res_dict: Dict[str, Any] = {"i": i, "mu": mu, "vz": vz}

    # Closed-wire BdG Spectrum & Topology
    if incl.spectra:
        res_dict["spectral"] = compute_spectral_slice(mu, vz, config, state)

    # Calibrated Barrier Sweeps (Conductance Matrix DataArray)
    if incl.barrier_sweep:
        if state.stage1_cutters is None:
            raise ValueError(
                "Calculation includes barrier_sweep, but state.stage1_cutters is None. "
                "Ensure barrier calibration is enabled and successful before running the simulation step."
            )
        q_nodes = worker_context["q_nodes"]
        res_dict["conductance"] = compute_barrier_sweep_slice(mu, vz, config, state, q_nodes)

    # TGP Stage 1: 21 without-replacement cutter harmonics
    if incl.tgp_stage1:
        if state.tgp_stage1_pairs is None:
            raise ValueError(
                "Calculation includes tgp_stage1, but state.tgp_stage1_pairs is None. "
                "Ensure Stage 1 cutter pairing is configured and generated before running the simulation step."
            )
        quad_dict = worker_context["quad_dict"]
        res_dict["tgp_stage1"] = compute_tgp_stage1_slice(mu, vz, config, state, quad_dict)

    # TGP Stage 2: mu-compensated symmetric cutters & bias spectroscopy
    if incl.tgp_stage2 and incl.stage2_fullgrid:
        if state.stage2_cutters is None:
            raise ValueError(
                "Calculation includes tgp_stage2 with stage2_fullgrid, but state.stage2_cutters is None. "
                "Ensure Stage 2 cutter calibration is configured and executed before running the simulation step."
            )
        Vbias_arr = worker_context["Vbias_arr"]
        res_dict["tgp_stage2"] = compute_tgp_stage2_slice(mu, vz, config, state, Vbias_arr)

    return res_dict


def worker_pfaffian_step(iter_data: Tuple, config: Any, state: SimulationState) -> Dict[str, Any]:
    """
    Worker function for the Pfaffian topological invariant calculation loop (Loop 2).
    """
    i = int(iter_data[0])
    mu = float(iter_data[1])
    vz = float(iter_data[2])

    # state.Vdisx is already in meV (scaled by V0), so pass V0=1.0 to avoid double-scaling.
    pfaff_val = hp.cal_pfaffian_invariant(
        ts=state.t,
        alphas=state.alpha,
        gamma=state.gamma,
        delta0=state.Delta0,
        Nx=state.Ls,
        Vdisx=state.Vdisx,
        V0=1.0,
        gm=vz,
        mu=mu,
        delta_N=0
    )
    return {"i": i, "mu": mu, "vz": vz, "pfaffian_invariant": int(pfaff_val)}


def _resolve_gap_threshold_factor(v0: float, cfg_val: Any) -> float:
    """Resolve Stage 2 gap_threshold_factor strictly from disorder amplitude V0."""
    if isinstance(cfg_val, (int, float)):
        return float(cfg_val)
    if isinstance(cfg_val, (dict, DictConfig)):
        for key, factor in cfg_val.items():
            if math.isclose(float(key), float(v0), abs_tol=1e-6):
                return float(factor)
        raise ValueError(
            f"No gap_threshold_factor configured for disorder V0={v0}. "
            f"Configured keys: {list(cfg_val.keys())}"
        )
    raise TypeError(
        f"Expected float or mapping for gap_threshold_factor, got {type(cfg_val).__name__}"
    )


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

if __name__ == "__main__":
    t_start = time.perf_counter()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs="?", default="fixing_code.yaml", help="Path to config YAML file.")
    parser.add_argument("--workers", dest="workers", type=int, default=None, help="Number of workers.")
    cli_args, remaining = parser.parse_known_args()

    config_path = PathConfigs.PARAMETERS / cli_args.config_file
    if not config_path.exists():
        candidate = Path(cli_args.config_file)
        if candidate.exists():
            config_path = candidate
        else:
            raise FileNotFoundError(
                f"Configuration file not found at '{config_path}' or '{candidate}'"
            )

    config = ConfigManager.initialize(str(config_path), cli_args=cli_args)
    config.config_path = config_path 

    state = SimulationState(config)
    
    dirname = config.outputs.dirname
    out_dir = PathConfigs.DATA / dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Simulation for: {dirname} ---")

    # Artifact Log
    ConfigManager.log_artifact(config, state, out_dir)

    # Save Static Real-Space Disorder Profile (vdisx.nc) Once Before Loop
    lambda_dis_val = config.simulation.disorder.lambda_dis
    vdis_ds = xr.Dataset(
        data_vars={
            "Vdisx": (["site"], state.Vdisx, {"units": "meV"}),
        },
        coords={"site": np.arange(state.Ls, dtype=int)},
        attrs={
            "V0_meV": float(state.V0),
            "lambda_dis": float(lambda_dis_val) if lambda_dis_val is not None else np.nan,
            "Ls": int(state.Ls),
            "is_real_space": int(bool(config.simulation.disorder.is_real_space)),
        }
    )
    vdis_ds.to_netcdf(out_dir / "vdisx.nc", engine="h5netcdf")
    print(f"Saved real-space disorder to: {out_dir / 'vdisx.nc'}")

    # Build Worker Context & Precomputed Quadratures
    rs = config.simulation.resolution
    incl = config.calculation_includes
    tgp_cfg = config.protocol.tgp

    target_temps = [float(T) for T in rs.temperatures]

    # Barrier sweep quadrature nodes
    q_nodes = {}
    if incl.barrier_sweep:
        n_bnodes = rs.barrier_quadrature_nodes
        for T_val in target_temps:
            if T_val > 0.0:
                qe, qw = hp.get_thermal_quadrature_nodes(
                    T_mK=T_val, n_nodes=n_bnodes, bias_meV=0.0, k_B=state.k_B
                )
                q_nodes[T_val] = (qe, qw)
            else:
                q_nodes[0.0] = (None, None)

    # TGP Stage 1 quadrature
    quad_dict = {}
    if incl.tgp_stage1:
        if state.tgp_stage1_pairs is None:
            raise ValueError(
                "Calculation includes tgp_stage1, but state.tgp_stage1_pairs is None."
            )
        tgp1 = tgp_cfg.stage1
        q_method = tgp1.quadrature_method
        n_qnodes = tgp1.quadrature_nodes

        for T_val in target_temps:
            if T_val > 0.0:
                if q_method == "double_exponential":
                    quad_dict[T_val] = lc.get_double_exponential_quadrature(T_mK=T_val, n_nodes=n_qnodes)
                else:
                    quad_dict[T_val] = lc.get_tanh_legendre_quadrature(T_mK=T_val, n_nodes=n_qnodes)
            else:
                quad_dict[0.0] = None

    Vbias_arr = hp.make_dynamic_grid()

    worker_context = {
        "q_nodes": q_nodes,
        "quad_dict": quad_dict,
        "Vbias_arr": Vbias_arr,
    }

    params_list = state.params_list
    print(f"Total Parameter Points: {len(params_list)} (mu in [{state.mu_var[0]}, {state.mu_var[-1]}], Vz in [{state.Vz_var[0]}, {state.Vz_var[-1]}])")

    # Main Simulation Loop (Loop 1: Transport & Spectra)
    if config.acceleration_type == "parallel":
        num_workers = hp.get_workers(config.num_workers)
        print(f"Launching Multiprocessing Pool with {num_workers} workers...")
        with mp.Pool(processes=num_workers) as pool:
            func_sim = partial(worker_simulation_step, config=config, state=state, worker_context=worker_context)
            results = list(tqdm(pool.imap(func_sim, params_list, chunksize=1), total=len(params_list), desc="Transport/Spectral Loop"))
    else:
        print("Using Serial Execution (CPU)...")
        results = [worker_simulation_step(pms, config, state, worker_context) for pms in tqdm(params_list, desc="Transport/Spectral Loop")]

    # Assemble Point Slices into Decoupled NetCDF Datasets
    print("\n--- Aggregating Point Slices into NetCDF Datasets ---")

    # Spectra & Topology
    spectral_slices = [r["spectral"] for r in results if "spectral" in r]
    ds_spectral = xr.combine_by_coords(spectral_slices) if spectral_slices else None

    # Conductance Sweeps
    cond_slices = [r["conductance"] for r in results if "conductance" in r]
    if cond_slices:
        ds_cond = xr.combine_by_coords(cond_slices)
        ds_cond.to_netcdf(out_dir / "conductance_sweeps.nc", engine="h5netcdf")
        print(f"Saved Conductance Sweeps to: {out_dir / 'conductance_sweeps.nc'}")

    # TGP Stage 1
    s1_slices = [r["tgp_stage1"] for r in results if "tgp_stage1" in r]
    if s1_slices:
        ds_s1 = xr.combine_by_coords(s1_slices)
        ds_s1.to_netcdf(out_dir / "tgp_stage1.nc", engine="h5netcdf")
        print(f"Saved TGP Stage 1 Harmonics to: {out_dir / 'tgp_stage1.nc'}")

    # TGP Stage 2 (held in memory until Loop 2 assigns Pfaffian topological invariant to L_SI/R_SI)
    s2_slices = [r["tgp_stage2"] for r in results if "tgp_stage2" in r]
    ds_s2: Optional[xr.Dataset] = None
    if s2_slices:
        if not incl.pfaffian:
            raise ValueError(
                "Calculation includes tgp_stage2, which requires incl.pfaffian=True "
                "to populate L_SI/R_SI with the closed-wire topological Pfaffian invariant."
            )
        ds_s2 = xr.combine_by_coords(s2_slices)
        tgp2_cfg = config.protocol.tgp.stage2
        bias_vals = ds_s2.coords["bias"].values
        leading_dims = ("cutter_pair_index", "zeeman_field", "chemical_potential", "temperature")
        leading_shape = tuple(ds_s2.sizes[d] for d in leading_dims)
        g_rl_2d = ds_s2["g_rl"].transpose(*leading_dims, "bias").values.reshape(-1, len(bias_vals))
        g_lr_2d = ds_s2["g_lr"].transpose(*leading_dims, "bias").values.reshape(-1, len(bias_vals))
        upper_th = (
            float("inf")
            if tgp2_cfg.upper_conductance_threshold is None
            else float(tgp2_cfg.upper_conductance_threshold)
        )
        gap_factor = _resolve_gap_threshold_factor(state.V0, tgp2_cfg.gap_threshold_factor)
        noise_th = float(tgp2_cfg.noise_threshold)
        gap_rl_flat, _, _ = hp.extract_nonlocal_gap(
            bias=bias_vals,
            g_nonlocal=g_rl_2d,
            gap_threshold_factor=gap_factor,
            upper_conductance_threshold=upper_th,
            noise_threshold=noise_th,
        )
        gap_lr_flat, _, _ = hp.extract_nonlocal_gap(
            bias=bias_vals,
            g_nonlocal=g_lr_2d,
            gap_threshold_factor=gap_factor,
            upper_conductance_threshold=upper_th,
            noise_threshold=noise_th,
        )
        gap_rl_arr = gap_rl_flat.reshape(leading_shape)
        gap_lr_arr = gap_lr_flat.reshape(leading_shape)
        gap_attrs = {
            "gap_threshold_factor": gap_factor,
            "noise_threshold": noise_th,
            "V0_meV": float(state.V0),
        }
        ds_s2["gap_rl"] = (leading_dims, gap_rl_arr, gap_attrs)
        ds_s2["gap_lr"] = (leading_dims, gap_lr_arr, gap_attrs)
        ds_s2["transport_gap"] = (leading_dims, np.minimum(gap_rl_arr, gap_lr_arr), gap_attrs)

    # Pfaffian Invariant Calculation (Loop 2)
    pfaff_grid: Optional[np.ndarray] = None
    if incl.pfaffian:
        print("\nStarting Pfaffian Topological Invariant Loop (Loop 2)...")
        if config.acceleration_type == "parallel":
            num_workers = hp.get_workers(config.num_workers)
            with mp.Pool(processes=num_workers) as pool:
                func_pfaff = partial(worker_pfaffian_step, config=config, state=state)
                pfaff_results = list(tqdm(pool.imap(func_pfaff, params_list, chunksize=1), total=len(params_list), desc="Pfaffian Sweep"))
        else:
            pfaff_results = [worker_pfaffian_step(pms, config, state) for pms in tqdm(params_list, desc="Pfaffian Sweep")]

        unique_mus = np.unique(state.mu_var)
        unique_vzs = np.unique(state.Vz_var)
        pfaff_grid = np.zeros((len(unique_mus), len(unique_vzs)), dtype=np.int64)
        for r in pfaff_results:
            m_idx = np.where(unique_mus == r["mu"])[0][0]
            v_idx = np.where(unique_vzs == r["vz"])[0][0]
            pfaff_grid[m_idx, v_idx] = r["pfaffian_invariant"]

        if ds_spectral is not None:
            ds_spectral["pfaffian_invariant"] = (["chemical_potential", "zeeman_field"], pfaff_grid)
        else:
            ds_spectral = xr.Dataset(
                data_vars={"pfaffian_invariant": (["chemical_potential", "zeeman_field"], pfaff_grid)},
                coords={"chemical_potential": unique_mus, "zeeman_field": unique_vzs}
            )

    if ds_s2 is not None:
        assert pfaff_grid is not None
        # Convert closed-wire Pfaffian (1=topological, 0=trivial) to TGP SI convention (-1=topological, +1=trivial)
        si_2d = np.where(pfaff_grid > 0, -1, 1).astype(np.int64)
        si_da = xr.DataArray(
            si_2d,
            coords={"chemical_potential": np.unique(state.mu_var), "zeeman_field": np.unique(state.Vz_var)},
            dims=("chemical_potential", "zeeman_field"),
        )
        si_aligned = si_da.reindex(
            chemical_potential=ds_s2.coords["chemical_potential"].values,
            zeeman_field=ds_s2.coords["zeeman_field"].values,
        )
        si_3d = np.broadcast_to(
            si_aligned.transpose("zeeman_field", "chemical_potential").values[None, :, :],
            (ds_s2.sizes["cutter_pair_index"], ds_s2.sizes["zeeman_field"], ds_s2.sizes["chemical_potential"]),
        ).copy()
        si_attrs = {
            "description": (
                "Closed-wire topological Pfaffian invariant (-1 = topological, +1 = trivial), "
                "stored under azure-quantum-tgp L_SI/R_SI variable names for schema compatibility "
                "(not computed via reflection-matrix scattering invariant)."
            )
        }
        ds_s2["L_SI"] = (["cutter_pair_index", "zeeman_field", "chemical_potential"], si_3d, si_attrs)
        ds_s2["R_SI"] = (["cutter_pair_index", "zeeman_field", "chemical_potential"], si_3d.copy(), si_attrs)
        ds_s2.to_netcdf(out_dir / "tgp_stage2.nc", engine="h5netcdf")
        print(f"Saved TGP Stage 2 Spectroscopy to: {out_dir / 'tgp_stage2.nc'}")

    if ds_spectral is not None:
        ds_spectral.to_netcdf(out_dir / "spectra_and_topology.nc", engine="h5netcdf")
        print(f"Saved Spectra and Topology to: {out_dir / 'spectra_and_topology.nc'}")

    # Legacy Export (If configured)
    if incl.save_old_data_format:
        print("\nExporting legacy .npy files from NetCDF datasets...")
        hp.export_legacy_npy_files(out_dir)

    # Runtime Accounting & Metadata Logging
    t_end = time.perf_counter()
    total_runtime_sec = t_end - t_start
    total_runtime_min = total_runtime_sec / 60.0
    total_runtime_hrs = total_runtime_sec / 3600.0

    print("\n" + "=" * 65)
    print(f"Simulation Complete for: {dirname}")
    print(f"Total Runtime: {total_runtime_sec:.2f} s ({total_runtime_min:.2f} min / {total_runtime_hrs:.3f} hrs)")
    print("=" * 65 + "\n")

    if isinstance(config, DictConfig):
        cfg_dict = OmegaConf.to_container(config, resolve=True)
    elif hasattr(config, 'model_dump'):
        cfg_dict = config.model_dump()
    else:
        cfg_dict = dict(config)

    all_params = {
        **cfg_dict,
        'total_runtime_seconds': total_runtime_sec,
        'total_runtime_minutes': total_runtime_min,
        'total_runtime_hours': total_runtime_hrs,
        'mu_var': state.mu_var,
        'Vz_var': state.Vz_var,
    }
    hp.np_savez_wrapped("all_params", dirname, **all_params)

    with open(out_dir / "runtime.txt", "w") as rf:
        rf.write(f"Simulation Directory: {dirname}\n")
        rf.write(f"Total Runtime Seconds: {total_runtime_sec:.4f}\n")
        rf.write(f"Total Runtime Minutes: {total_runtime_min:.4f}\n")
        rf.write(f"Total Runtime Hours: {total_runtime_hrs:.4f}\n")
        rf.write(f"Formatted Runtime: {total_runtime_sec:.2f} s ({total_runtime_min:.2f} min / {total_runtime_hrs:.3f} hrs)\n")

    print("Done.")
