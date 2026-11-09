# parameter_handler.py
import numpy as np
import itertools as itr
from typing import List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from src.config import PathConfigs
import src.barrier_calibration as bc
import src.helpers as hp 


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



class ProtocolConfig(BaseModel):
    """
    Configuration for post-processing using the Azure Quantum TGP pipeline.
    """

    # Hardware / Experimental Corrections
    phase_shift_left: float = 0.0
    phase_shift_right: float = 0.0
    
    # Visualization & Selections
    selected_cutter: int = 0
    selected_clusters: List[int] = Field(default_factory=list)
    B_FIXED: float = 1.18
    V_FIXED: float = 2.85
    
    # Stage 0
    T_mK_stage1: float = 30.0
    T_mK_stage2: float = 40.0
    
    # Stage 1
    roi1_pct_box: int = 10
    roi1_min_margin_box: Tuple[float, float] = (0.003, 0.2)
    th_2w_p: float = 0.5 
    th_2w: Optional[float] = None
    n_tiles: int = 100
    percentile: int = 33
    B_max: Optional[float] = None
    V_max: Optional[float] = None
    B_max_stage2: Optional[float] = None
    th_3w: float = 1000.0
    th_3w_tat: float = 0.7
    
    # Stage 2A: Gap Extraction
    gap_threshold_factor: Any = 0.001
    noise_threshold: float = 1e-4
    upper_conductance_threshold: Optional[float] = None  # None means infinity
    
    # Stage 2B: ZBP Detection
    bias_window: float = 0.003
    threshold_high: float = 0.067
    threshold_low: float = 0.010
    zbp_probability_threshold: float = 0.6
    derivative_threshold: float = 100.0
    average_over_cutter: bool = False  # Official analyze_2 uses False
    
    # Stage 2D: Cluster Scoring
    cluster_gap_threshold: float = 10.0 #mueV
    cluster_ncutter_threshold: float = 0.5
    cluster_percentage_boundary_threshold: float = 0.6
    min_cluster_size: int = 7
    
    class Config:
        arbitrary_types_allowed = True


def _load_and_merge_defaults(node: Any, defaults_dir: Path) -> Any:
    """
    Recursively resolves defaults bottom-up so that:
      1. Sub-section defaults override the defaults of the layer above.
      2. Parameters explicitly specified in the main yaml config override all defaults.
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
        # Recurse on default_cfg in case it has its own nested defaults
        default_cfg = _load_and_merge_defaults(default_cfg, defaults_dir)

        # Merge: default_cfg is base; current node takes precedence
        node = OmegaConf.merge(default_cfg, node)

    return node


class ConfigManager:
    """
    Orchestrates configuration loading, merging CLI overrides,
    and artifact logging using OmegaConf DictConfigs.
    """

    @staticmethod
    def initialize(config_file: str, cli_args = None) -> DictConfig:
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

        # 1. Load the raw user YAML
        config = OmegaConf.load(config_path)

        # 2. Recursively resolve and merge defaults from Inputs/Parameters/Defaults
        defaults_dir = Path(PathConfigs.PARAMETERS / "Defaults")
        return _load_and_merge_defaults(config, defaults_dir)


    @staticmethod
    def get_config(config_path: str = None, cli_args: Optional[List[str]] = None) -> DictConfig:
        """
        Parses CLI arguments if needed, loads YAML base via ConfigManager.initialize,
        merges CLI overrides, and returns a DictConfig.
        """
        if config_path is None:
            parser = argparse.ArgumentParser(description="Input Config YAML file.")
            parser.add_argument("--config_path", "--config", dest="config_path", type=str, default=None, help="Path to a YAML config file.")
            known, _ = parser.parse_known_args(cli_args)
            if not known.config_path:
                raise ValueError("No configuration file specified via config_path or --config_path argument.")
            config_path = known.config_path

        cfg = ConfigManager.initialize(config_path)

        # Merge any key=value CLI arguments if provided
        if cli_args:
            dotlist = [arg.lstrip("-") for arg in cli_args if "=" in arg]
            if dotlist:
                cli_conf = OmegaConf.from_dotlist(dotlist)
                cfg = OmegaConf.merge(cfg, cli_conf)

        return cfg

    @staticmethod
    def log_artifact(config: Any, state:Any, output_dir: Path):
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
        np.savez(state_file, **state.__dict__())
        print(f"Params Artifact logged to: {artifact_file}")
        print(f"Derived State Artifact logged to: {state_file}")


class SimulationState:
    """
    Holds the derived physical parameters and coordinate arrays.
    Initialized from an OmegaConf DictConfig or dictionary.
    """
    def __init__(self, config: Any):
        # Fundamental Physical Constants (loaded from Defaults/physical_constants.yaml)
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


        # 2. Derived Physical Parameters
        a0 = geom.a0
        ms = geom.ms
        Delta_0 = geom.Delta_0
        gamma = geom.gamma
        Ls = geom.Ls
        self.t = 1000 * eta_m / (2 * a0**2 * ms)
        print(f"Derived Hopping Parameter t: {self.t:.4f}")
        self.alpha = 140.0 / a0
        self.Delta = Delta_0 * gamma / (Delta_0 + gamma)
        
        self.mu_leads = self.t

        # 3. Sweep Arrays
        mu_max = res.mu_max
        mu_min = res.mu_min
        mu_dist = res.mu_dist
        Vz_max = res.Vz_max
        Vz_min = res.Vz_min
        Vz_dist = res.Vz_dist

        Nmu = int(np.round((mu_max - mu_min) / mu_dist))
        Nvz = int(np.round((Vz_max - Vz_min) / Vz_dist))

        self.mu_var = np.round(np.asarray([mu_min + n*mu_dist for n in range(Nmu)]), decimals=2)
        self.Vz_var = np.round(np.asarray([Vz_min + n*Vz_dist for n in range(Nvz)]), decimals=2)
        
        # Product space for parallelization: [[index, mu, vz], ...]
        self.params_list = np.asarray([
            [i, p[0], p[1]] 
            for i, p in enumerate(itr.product(self.mu_var, self.Vz_var))
        ])

        # 4. Disorder realization
        disorder_path = PathConfigs.RUN_FILES / disorder.disorder_fpath
        lambda_dis = disorder.lambda_dis
        is_real_space = disorder.is_real_space

        self.Vdisx = hp.initialize_vdis_from_data(
            disorder_path, 
            lambda_dis=lambda_dis, 
            Ls=Ls, 
            is_real_space=is_real_space
        )

        self.energies = np.linspace(-0.5, 0.5, res.num_engs)

        sweep_gnmin=calib.sweep_gnmin
        sweep_gnmax=calib.sweep_gnmax
        sweep_gnres=calib.sweep_gnres
        num = int(np.round((sweep_gnmax - sweep_gnmin)/sweep_gnres))
        
        Gn_sweep_points = np.round(np.linspace(sweep_gnmin, sweep_gnmax, num), decimals=2)
        self.target_GNs = _merge_arrs(Gn_sweep_points, calib.symmetric_target_gns, tol = 0.01)

        if calib.calibrate_cutters:
            print(f"--- Running Simultaneous Cutter Barrier Calibration for targets: {calib.target_gns} ---")
            vdis_for_calib = self.Vdisx * float(disorder.V0) 
            self.calibrated_cutters = bc.calibrate_wire_cutters(
                targets = self.target_GNs,
                calibration_cfg=calib,
                geometry_cfg=geom,
                Vdisx=vdis_for_calib,
                t=self.t,
                alpha=self.alpha,
                mu_leads=self.mu_leads,
            )
            print(f"Calibrated Symmetric Cutter Pairs [G_N, [UL, UR]]:\n{self.calibrated_barriers}")
        else:
            self.calibrated_cutters = None


    def get_static_params(self, Vdisx: np.ndarray, config: Any) -> dict:
        """
        Extracts and returns the parameter dictionary expected by worker functions.
        All fields are read directly from config, geometry, resolution, calculation,
        and simulation state without silent fallback defaults (expert-coding Part B).
        """
        geom = config.simulation.geometry
        disorder = config.simulation.disorder
        res = config.simulation.resolution
        calc = config.calculation_includes

        t_list = [float(x) for x in res.temperatures]
        is_finite_temp = any(t > 0.0 for t in t_list)
        cond_mode = "finite_temperature" if is_finite_temp else "zero_temperature"

        return {
            't': self.t,
            'mu_n': geom.mu_n,
            'Delta0': geom.Delta_0,
            'alpha': self.alpha,
            'gamma': geom.gamma,
            'V0': disorder.V0,
            'qn': geom.qn,
            'Ln': geom.Ln,
            'Lb': geom.Lb,
            'Ls': geom.Ls,
            'mu_leads': self.mu_leads,
            'barrier0': geom.barrier0,
            'Vdisx': Vdisx,
            'energies': self.energies,
            'calibrated_barriers': self.calibrated_barriers,
            'asym_ratios': self.asym_ratios,
            'mu_var': self.mu_var,
            'Vz_var': self.Vz_var,
            'num_eigenvalues': res.num_eigenvalues,
            'weight_threshold': res.weight_threshold,
            'separation_threshold': res.separation_threshold,
            'eng_window_range': res.eng_window_range,
            'conductance_flag': calc.conductance_flag,
            'spectra_flag': calc.spectra_flag,
            'localization_flag': calc.localization_flag,
            'calc_pfaffian': calc.calc_pfaffian,
            'pfaffian_delta_N': calc.pfaffian_delta_N,
            'barrier_conductance_mode': cond_mode,
            'temperatures': t_list[0] if t_list else 0.0,
            'barrier_T_mK_list': t_list,
            'barrier_quadrature_nodes': res.barrier_quadrature_nodes,
            'barrier_sweep_only': calc.barrier_sweep_only,
            'hamiltonian_mode': config.hamiltonian_mode,
            'tgp_barrier_pairs': self.tgp_barrier_pairs,
            'asym_min': res.asym_min,
            'asym_max': res.asym_max,
            'asym_points': res.asym_points,
            'target_gns': self.target_gns,
            'multi_baseline_barriers': calc.multi_baseline_barriers,
            'barrier_sweep_indices': calc.barrier_sweep_indices,
            'calc_barrier_sweeps': calc.calc_barrier_sweeps,
            'calc_25_cutter_harmonics': calc.calc_25_cutter_harmonics,
            'quadrature_nodes_25': res.quadrature_nodes_25,
        }
