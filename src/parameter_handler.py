import numpy as np
import itertools as itr
import multiprocessing as mp
from typing import List, Any, Tuple
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from src.config import PathConfigs
import src.barrier_calibration as bc
import src.helpers as hp
import xarray as xr


def _merge_arrs(A, B, tol):
    # merges A and B where any elements within tol are replaced with the element of B
    C = []
    A = list(A)
    B = list(B)
    for a in A:
        any_close = np.where(np.isclose(a, B, atol=tol))[0]
        if len(any_close != 0):
            C.append(B[any_close[0]])
        else:
            C.append(a)
    D = np.append(C, B)
    return np.sort(np.asarray(list(set(D))))


def _calibrate_stage2_mu_row(args: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Calls calibrate_wire_cutters for a single chemical potential [mu_pt] with pool=None."""
    mu_pt, stage2_gns, calib, geom, Vdisx_meV, t, alpha, mu_leads = args
    c_pt = bc.calibrate_wire_cutters(
        targets=stage2_gns,
        calibration_cfg=calib,
        geometry_cfg=geom,
        Vdisx=Vdisx_meV,
        t=t,
        alpha=alpha,
        mu_leads=mu_leads,
        mu_points=[float(mu_pt)],
        pool=None,
    )
    return c_pt.sel(side="left").values, c_pt.sel(side="right").values


def _load_and_merge_defaults(node: Any, defaults_dir: Path) -> Any:
    """
    Recursively resolves defaults bottom-up so that sub-section defaults override
    the defaults of the layer above, and parameters explicitly specified in the
    main yaml config override all defaults.
    """
    if not isinstance(node, (dict, DictConfig)):
        return node

    for key in list(node.keys()):
        if key != "default":
            node[key] = _load_and_merge_defaults(node[key], defaults_dir)

    if "default" in node and node["default"]:
        default_file = defaults_dir / str(node["default"])
        if not default_file.exists():
            raise FileNotFoundError(f"Default config '{node['default']}' not found at: {default_file}")

        default_cfg = OmegaConf.load(default_file)
        default_cfg = _load_and_merge_defaults(default_cfg, defaults_dir)
        node = OmegaConf.merge(default_cfg, node)

    return node


class ConfigManager:
    """
    Recursively loads yaml file and its defaults.
    """

    @staticmethod
    def initialize(config_file: str, cli_args=None) -> DictConfig:
        config_path = Path(PathConfigs.PARAMETERS / config_file)
        if not config_path.exists():
            alt_path = Path(config_file)
            if alt_path.exists():
                config_path = alt_path
            else:
                for base in [PathConfigs.ROOT, PathConfigs.ROOT / "Inputs", PathConfigs.PARAMETERS]:
                    candidate = base / config_file
                    if candidate.exists():
                        config_path = candidate
                        break
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        config = OmegaConf.load(config_path)
        defaults_dir = Path(PathConfigs.PARAMETERS / "Defaults")
        return _load_and_merge_defaults(config, defaults_dir)

    @staticmethod
    def log_artifact(config: Any, state: Any, output_dir: Path):
        """
        Dumps the fully resolved configuration to a YAML file in the output directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_file = output_dir / "resolved_params.yaml"
        state_file = output_dir / "state.npz"
        if not isinstance(config, DictConfig):
            conf = OmegaConf.create(config)
        else:
            conf = config
        OmegaConf.save(config=conf, f=artifact_file)
        state_dict = {
            k: v for k, v in state.__dict__.items()
            if isinstance(v, (np.ndarray, list, float, int, str, bool))
        }
        np.savez(state_file, **state_dict)
        print(f"Params Artifact logged to: {artifact_file}")
        print(f"Derived State Artifact logged to: {state_file}")


class SimulationState:
    """
    Holds the derived physical parameters and coordinate arrays.
    Initialized from an OmegaConf DictConfig.
    """
    def __init__(self, config: Any):
        constants_file = PathConfigs.PARAMETERS / "Defaults" / "physical_constants.yaml"
        if not constants_file.exists():
            raise FileNotFoundError(f"Canonical physical constants file not found: {constants_file}")
        constants = OmegaConf.load(constants_file)

        # Free-electron dispersion coefficient eta_m = (hbar^2 / m_0) in eV * Angstrom^2
        eta_m = (constants.hbar**2 * constants.e0 * 1e20) / constants.m0

        geom = config.simulation.geometry
        disorder = config.simulation.disorder
        res = config.simulation.resolution
        calib = config.simulation.barrier_calibration
        calc = config.calculation_includes

        # Derived physical parameters
        a0 = geom.a0
        ms = geom.ms
        Delta_0 = geom.Delta_0
        gamma = geom.gamma
        Ls = geom.Ls
        self.t = 1000 * eta_m / (2 * a0**2 * ms)
        print(f"Derived Hopping Parameter t: {self.t:.4f}")
        self.alpha = 140.0 / a0
        self.Delta = Delta_0 * gamma / (Delta_0 + gamma)

        self.Ln = int(geom.Ln)
        self.Lb = int(geom.Lb)
        self.Ls = int(geom.Ls)
        self.mu_n = float(geom.mu_n)
        self.Delta0 = float(geom.Delta_0)
        self.gamma = float(geom.gamma)
        self.mu_leads = self.t

        # Sweep coordinate arrays
        mu_max = res.mu_max
        mu_min = res.mu_min
        mu_dist = res.mu_dist
        Vz_max = res.Vz_max
        Vz_min = res.Vz_min
        Vz_dist = res.Vz_dist

        Nmu = int(np.round((mu_max - mu_min) / mu_dist))
        Nvz = int(np.round((Vz_max - Vz_min) / Vz_dist))

        self.mu_var = np.round(np.asarray([mu_min + n * mu_dist for n in range(Nmu)]), decimals=2)
        self.Vz_var = np.round(np.asarray([Vz_min + n * Vz_dist for n in range(Nvz)]), decimals=2)

        self.params_list = np.asarray([
            [i, p[0], p[1]]
            for i, p in enumerate(itr.product(self.mu_var, self.Vz_var))
        ])

        # Disorder realization
        disorder_path = PathConfigs.RUN_FILES / disorder.disorder_fpath
        lambda_dis = disorder.lambda_dis
        is_real_space = disorder.is_real_space

        self.Vdisx = hp.initialize_vdis_from_data(
            disorder_path,
            lambda_dis=lambda_dis,
            Ls=Ls,
            is_real_space=is_real_space,
        )
        self.V0 = float(disorder.V0)
        self.Vdisx_meV = self.Vdisx * self.V0

        self.energies = np.linspace(-0.5, 0.5, res.num_engs)

        sweep_gnmin = calib.sweep_gnmin
        sweep_gnmax = calib.sweep_gnmax
        sweep_gnres = calib.sweep_gnres
        num = int(np.round((sweep_gnmax - sweep_gnmin) / sweep_gnres))

        Gn_sweep_points = np.round(np.linspace(sweep_gnmin, sweep_gnmax, num), decimals=2)
        self.target_GNs = _merge_arrs(Gn_sweep_points, calib.symmetric_target_gns, tol=0.01)
        self.baseline_GNs = np.asarray(calib.symmetric_target_gns, dtype=float)

        calib_workers = hp.get_workers(config.num_workers) if config.acceleration_type == "parallel" else 1
        default_mu_pts = np.linspace(float(calib.mu_min), float(calib.mu_max), int(calib.n_mu_points))

        need_calib_pool = (
            calib_workers > 1
            and (calib.calibrate_cutters or calc.tgp_stage1 or calc.tgp_stage2)
        )
        pool = mp.Pool(processes=int(calib_workers)) if need_calib_pool else None

        try:
            # Barrier sweeps calibration (pool parallelizes over default_mu_pts inside calibrate_single_barrier)
            if calib.calibrate_cutters:
                print(f"--- Running Simultaneous Cutter Barrier Calibration for targets: {self.target_GNs} ---")
                self.calibrated_cutters = bc.calibrate_wire_cutters(
                    targets=self.target_GNs,
                    calibration_cfg=calib,
                    geometry_cfg=geom,
                    Vdisx=self.Vdisx_meV,
                    t=self.t,
                    alpha=self.alpha,
                    mu_leads=self.mu_leads,
                    mu_points=default_mu_pts,
                    pool=pool,
                )
                print(f"Calibrated Symmetric Cutter Pairs [G_N, [UL, UR]]:\n{self.calibrated_cutters}")
            else:
                raise ValueError("Cutters have to be calibrated!")

            
            self.calibrated_barriers = self.calibrated_cutters
            self.target_gns = self.target_GNs

            # TGP Stage 1 cutters (injected pool parallelizes over 61-pt mu_calib grid inside calibrate_single_barrier)
            self.tgp_stage1_pairs = None
            if calc.tgp_stage1:
                tgp1 = config.protocol.tgp.stage1
                n_c1 = int(tgp1.num_cutters)
                gn_min = float(tgp1.gn_min)
                gn_max = float(tgp1.gn_max)
                n_mu = int(tgp1.num_mu_calib_points)
                seed = int(tgp1.pairing_seed)

                mu_calib = np.linspace(float(calib.mu_min), float(calib.mu_max), n_mu)
                stage1_gns = np.linspace(gn_min, gn_max, n_c1)
                print(f"--- Calibrating TGP Stage 1 Cutters ({n_c1} points over {n_mu}-pt mu grid, workers={calib_workers}) ---")
                s1_calib = bc.calibrate_wire_cutters(
                    targets=stage1_gns,
                    calibration_cfg=calib,
                    geometry_cfg=geom,
                    Vdisx=self.Vdisx_meV,
                    t=self.t,
                    alpha=self.alpha,
                    mu_leads=self.mu_leads,
                    mu_points=mu_calib,
                    pool=pool,
                )
                rng = np.random.default_rng(seed)
                left_idx = np.arange(n_c1)
                right_idx = rng.permutation(n_c1)

                self.tgp_stage1_pairs = xr.Dataset(
                    coords={
                        "cutter_pair_index": np.arange(n_c1, dtype=int),
                        "G_n_left": ("cutter_pair_index", stage1_gns[left_idx]),
                        "G_n_right": ("cutter_pair_index", stage1_gns[right_idx]),
                        "V_leftcutter": ("cutter_pair_index", s1_calib.sel(side="left").values[left_idx]),
                        "V_rightcutter": ("cutter_pair_index", s1_calib.sel(side="right").values[right_idx]),
                    }
                )

            # TGP Stage 2 cutters (outer pool.map parallelizes across N_mu single-element [mu_pt] calls with pool=None)
            self.tgp_stage2_cutters = None
            if calc.tgp_stage2:
                tgp2 = config.protocol.tgp.stage2
                stage2_gns = np.asarray(tgp2.target_gns, dtype=float)
                if tgp2.compensate_per_mu:
                    unique_mus = np.unique(self.mu_var)
                    print(f"--- Pre-calibrating TGP Stage 2 Cutters ({len(stage2_gns)} targets across {len(unique_mus)} mu rows, workers={calib_workers}) ---")
                    row_tasks = [
                        (float(mu_pt), stage2_gns, calib, geom, self.Vdisx_meV, self.t, self.alpha, self.mu_leads)
                        for mu_pt in unique_mus
                    ]
                    if pool is not None and len(row_tasks) > 1:
                        row_results = pool.map(_calibrate_stage2_mu_row, row_tasks)
                    else:
                        row_results = [_calibrate_stage2_mu_row(task) for task in row_tasks]

                    u_l_comp = np.vstack([r[0] for r in row_results])
                    u_r_comp = np.vstack([r[1] for r in row_results])

                    self.tgp_stage2_cutters = xr.Dataset(
                        data_vars={
                            "U_left_compensated": (["chemical_potential", "cutter_pair_index"], u_l_comp),
                            "U_right_compensated": (["chemical_potential", "cutter_pair_index"], u_r_comp),
                        },
                        coords={
                            "chemical_potential": unique_mus,
                            "cutter_pair_index": np.arange(len(stage2_gns), dtype=int),
                            "target_gn": ("cutter_pair_index", stage2_gns),
                        },
                    )
                else:
                    self.tgp_stage2_cutters = bc.calibrate_wire_cutters(
                        targets=stage2_gns,
                        calibration_cfg=calib,
                        geometry_cfg=geom,
                        Vdisx=self.Vdisx_meV,
                        t=self.t,
                        alpha=self.alpha,
                        mu_leads=self.mu_leads,
                        mu_points=default_mu_pts,
                        pool=pool,
                    )
        finally:
            if pool is not None:
                pool.close()
                pool.join()
