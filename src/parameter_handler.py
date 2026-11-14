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
        self.k_B = float(constants.k_B)

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

        # Disorder realization (Vdisx is stored in meV, already scaled by V0)
        disorder_path = PathConfigs.RUN_FILES / disorder.disorder_fpath
        lambda_dis = disorder.lambda_dis
        is_real_space = disorder.is_real_space

        self.V0 = float(disorder.V0)
        self.Vdisx = hp.initialize_vdis_from_data(
            disorder_path,
            V0=self.V0,
            lambda_dis=lambda_dis,
            Ls=Ls,
            is_real_space=is_real_space,
        )

        self.energies = np.linspace(-0.5, 0.5, res.num_engs)

        sweep_gnmin = calib.sweep_gnmin
        sweep_gnmax = calib.sweep_gnmax
        sweep_gnres = calib.sweep_gnres
        num = int(np.round((sweep_gnmax - sweep_gnmin) / sweep_gnres)) + 1

        nominal_stage1_gns = np.round(np.linspace(sweep_gnmin, sweep_gnmax, num), decimals=2)

        calib_workers = hp.get_workers(config.num_workers) if config.acceleration_type == "parallel" else 1
        default_mu_pts = np.linspace(float(calib.mu_min), float(calib.mu_max), int(calib.n_mu_points))

        need_calib = bool(calib.calibrate_cutters or calc.tgp_stage1 or calc.tgp_stage2)
        need_calib_pool = calib_workers > 1 and need_calib
        pool = mp.Pool(processes=int(calib_workers)) if need_calib_pool else None

        self.stage1_cutters = None
        self.target_GNs = nominal_stage1_gns
        self.baseline_GNs = np.array(
            [nominal_stage1_gns[0], nominal_stage1_gns[len(nominal_stage1_gns) // 2], nominal_stage1_gns[-1]],
            dtype=float,
        )
        self.tgp_stage1_pairs = None
        self.stage2_cutters = None

        try:
            if need_calib:
                print("--- Running Pre-Health Check at Pinch-Off Limit ---")
                g_pinch_l, g_pinch_r = bc.check_wire_health(
                    calibration_cfg=calib,
                    geometry_cfg=geom,
                    Vdisx=self.Vdisx,
                    t=self.t,
                    alpha=self.alpha,
                    mu_leads=self.mu_leads,
                    mu_points=default_mu_pts,
                    pool=pool,
                )
                print(f"Pre-health check passed: G_left={g_pinch_l:.3e}, G_right={g_pinch_r:.3e}")

            # Stage 1 / Barrier Sweep calibration (shared 61-pt mu median)
            if calib.calibrate_cutters or calc.tgp_stage1:
                print(f"--- Running Stage 1 Cutter Calibration (nominal={len(nominal_stage1_gns)} pts) ---")
                self.stage1_cutters = bc.calibrate_stage1_wire_cutters(
                    nominal_targets=nominal_stage1_gns,
                    calibration_cfg=calib,
                    geometry_cfg=geom,
                    Vdisx=self.Vdisx,
                    t=self.t,
                    alpha=self.alpha,
                    mu_leads=self.mu_leads,
                    mu_points=default_mu_pts,
                    pool=pool,
                )
                self.target_GNs = self.stage1_cutters.coords["G_n"].values
                mid_idx = len(self.target_GNs) // 2
                self.baseline_GNs = np.array(
                    [self.target_GNs[0], self.target_GNs[mid_idx], self.target_GNs[-1]],
                    dtype=float,
                )
                print(f"Calibrated Stage 1 Cutters [G_n, barrier_side]:\n{self.stage1_cutters}")

            if calc.tgp_stage1:
                tgp1 = config.protocol.tgp.stage1
                seed = int(tgp1.pairing_seed)
                actual_gns = self.stage1_cutters.coords["G_n"].values
                n_c1 = len(actual_gns)
                rng = np.random.default_rng(seed)
                left_idx = np.arange(n_c1)
                right_idx = rng.permutation(n_c1)

                self.tgp_stage1_pairs = xr.Dataset(
                    coords={
                        "cutter_pair_index": np.arange(n_c1, dtype=int),
                        "G_n_left": ("cutter_pair_index", actual_gns[left_idx]),
                        "G_n_right": ("cutter_pair_index", actual_gns[right_idx]),
                        "V_leftcutter": ("cutter_pair_index", self.stage1_cutters.sel(barrier_side="left").values[left_idx]),
                        "V_rightcutter": ("cutter_pair_index", self.stage1_cutters.sel(barrier_side="right").values[right_idx]),
                    }
                )

            if calc.tgp_stage2:
                unique_mus = np.unique(self.mu_var)
                print(f"--- Calibrating TGP Stage 2 Cutters across {len(unique_mus)} mu rows (workers={calib_workers}) ---")
                self.stage2_cutters = bc.calibrate_stage2_wire_cutters(
                    calibration_cfg=calib,
                    geometry_cfg=geom,
                    Vdisx=self.Vdisx,
                    t=self.t,
                    alpha=self.alpha,
                    mu_leads=self.mu_leads,
                    mu_points=unique_mus,
                    pool=pool,
                )
        finally:
            if pool is not None:
                pool.close()
                pool.join()
