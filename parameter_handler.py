# parameter_handler.py
import numpy as np
import itertools as itr
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat, model_validator
from typing import List, Any, Optional, Tuple
import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf
from config import PathConfigs

class SimulationConfig(BaseModel):
    """
    Pure input configuration for the simulation.
    All fields should correspond to what can be specified in YAML or CLI.
    This model enforces type safety and bounds checking.
    """
    # Directory and File Management
    dirname: str = "base_config"
    fname: str = "Tdis.npz"
    
    # Geometry
    Ls: PositiveInt = 300
    Ln: int = 0
    Lb: int = 3
    Lb_pdi: int = 3
    a0: PositiveFloat = 100.0
    
    # Disorder Parameters
    lambda_dis: Optional[PositiveFloat] = None
    realization_index: Optional[int] = None
    
    # Physics Parameters
    ms: PositiveFloat = 0.023
    Delta_0: float = Field(0.3, ge=0)
    gamma: float = Field(0.2, ge=0)
    mu_leads: Optional[float] = None # If None, defaults to hopping 't'
    barrier0: float = 2.0
    mu_n: float = 0.0
    V0: float = 1.2
    qn: int = 20
    
    # Simulation Resolution
    Upoints: PositiveInt = 100
    num_engs: PositiveInt = 101
    num_eigenvalues: PositiveInt = 12
    eng_window_range: PositiveInt = 51
    weight_threshold: float = Field(0.8, ge=0, le=1)
    separation_threshold: float = Field(0.8, ge=0, le=1)
    
    # Sweep Ranges
    mu_max: float = 4.5
    mu_min: float = 0.0
    mu_dist: PositiveFloat = 0.02
    Nmu: Optional[PositiveInt] = None
    
    Vz_max: float = 1.3
    Vz_min: float = 0.0
    Vz_dist: PositiveFloat = 0.02
    Nvz: Optional[PositiveInt] = None

    @model_validator(mode='after')
    def calculate_counts(self) -> 'SimulationConfig':
        """
        Calculates Nmu and Nvz from ranges/dist if not explicitly provided.
        """
        if self.Nmu is None:
            mu_rng = self.mu_max - self.mu_min
            self.Nmu = max(1, int(mu_rng / self.mu_dist))
        
        if self.Nvz is None:
            Vz_rng = self.Vz_max - self.Vz_min
            self.Nvz = max(1, int(Vz_rng / self.Vz_dist))
            
        return self
    
    # Execution Settings
    acceleration_type: str = "parallel" # parallel, gpu, None
    conductance_flag: bool = True
    spectra_flag: bool = True
    localization_flag: bool = True
    pdi_flag: bool = True
    calc_pfaffian: bool = False
    pfaffian_delta_N: int = 0

    class Config:
        arbitrary_types_allowed = True


class ConfigManager:
    """
    Orchestrates configuration loading, merging CLI overrides,
    validation, and artifact logging.
    """
    @staticmethod
    def get_config(config_path: str = None) -> SimulationConfig:
        """
        Parses CLI arguments, optionally loads a YAML base, 
        merges them, and returns a validated SimulationConfig.
        """
        parser = argparse.ArgumentParser(description="Run parallel transport and PDI simulation.")
        parser.add_argument("--config_path", type=str, default=config_path, help="Path to a YAML config file.")
        
        # Dynamically add SimulationConfig fields to argparse for overrides
        # Using model_fields to iterate over pydantic fields (v2)
        for field_name, field in SimulationConfig.model_fields.items():
            # For booleans, we use a slightly different pattern to allow --flag or --no-flag
            if field.annotation == bool:
                parser.add_argument(f"--{field_name}", action="store_true", dest=field_name, default=None)
                parser.add_argument(f"--no-{field_name}", action="store_false", dest=field_name, default=None)
            else:
                # We handle Optionals by getting the inner type if possible
                # Simple version for primitive types:
                parser.add_argument(f"--{field_name}", type=field.annotation, default=None)
        
        args = parser.parse_args()
        
        # 1. Load YAML Base if provided
        yaml_dict = {}
        if args.config_path:
            config_file_path = PathConfigs.ROOT / args.config_path
            if config_file_path.exists():
                yaml_dict = OmegaConf.to_container(OmegaConf.load(str(config_file_path)), resolve=True)
            elif os.path.exists(args.config_path):
                yaml_dict = OmegaConf.to_container(OmegaConf.load(args.config_path), resolve=True)
            else:
                print(f"Warning: Config file {args.config_path} not found. Using defaults/CLI.")

        # 2. Extract CLI Overrides (only those explicitly set by user)
        cli_dict = {k: v for k, v in vars(args).items() if v is not None and k != "config_path"}
        
        # 3. Merge (CLI > YAML > Defaults)
        merged_dict = {**yaml_dict, **cli_dict}
        
        # 4. Validate via Pydantic
        return SimulationConfig(**merged_dict)

    @staticmethod
    def log_artifact(config: SimulationConfig, output_dir: Path):
        """
        Dumps the fully resolved configuration to a YAML file in the output directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_file = output_dir / "resolved_params.yaml"
        
        # Convert to OmegaConf for serialization (using model_dump for v2)
        conf = OmegaConf.create(config.model_dump())
        OmegaConf.save(config=conf, f=artifact_file)
        print(f"Reproducibility Artifact logged to: {artifact_file}")


class SimulationState:
    """
    Holds the derived physical parameters and coordinate arrays.
    Initialized from a SimulationConfig object.
    """
    def __init__(self, config: SimulationConfig):
        # 1. Physical Constants
        hbar = 6.582119569e-16  # eV·s
        m0   = 9.10938356e-31   # kg
        e0   = 1.602176634e-19  # C
        # hbar^2/m0 in eV A^2
        eta_m = (hbar ** 2 * e0) * (1e20) / m0 

        # 2. Derived Physical Parameters
        self.t = 1000 * eta_m / (2 * config.a0**2 * config.ms)
        print(f"Derived Hopping Parameter t: {self.t:.4f}")
        self.alpha = 140.0 / config.a0
        self.Delta = config.Delta_0 * config.gamma / (config.Delta_0 + config.gamma)
        
        # Default mu_leads to 't' if not explicitly provided
        self.mu_leads = config.mu_leads if config.mu_leads is not None else self.t

        # 3. Sweep Arrays (Uses pre-calculated Nmu/Nvz from config validator)
        self.mu_var = np.linspace(config.mu_min, config.mu_max, config.Nmu)
        self.Vz_var = np.linspace(config.Vz_min, config.Vz_max, config.Nvz)
        
        # Product space for parallelization
        # Format: [[index, mu, vz], ...]
        self.params_list = [
            [i, p[0], p[1]] 
            for i, p in enumerate(itr.product(self.mu_var, self.Vz_var))
        ]
        
        # Constant coordinate arrays
        self.barrier_arr = np.linspace(-70*config.barrier0, 70 * config.barrier0, config.Upoints)
        self.energies = np.linspace(-0.5, 0.5, config.num_engs)
        
        # Mapping acceleration type to Kwant solver type
        self.solver_type = 'gpu' if config.acceleration_type == 'gpu' else 'cpu'

    def get_static_params(self, Vdisx: np.ndarray, config: SimulationConfig) -> dict:
        """
        Returns the dictionary expected by worker functions.
        Vdisx must be passed in as it is loaded from an external file.
        """
        return {
            't': self.t,
            'mu_n': config.mu_n,
            'Delta0': config.Delta_0,
            'alpha': self.alpha,
            'gamma': config.gamma,
            'V0': config.V0,
            'qn': config.qn,
            'Ln': config.Ln,
            'Lb': config.Lb,
            'Ls': config.Ls,
            'mu_leads': self.mu_leads,
            'barrier0': config.barrier0,
            'Vdisx': Vdisx,
            'energies': self.energies,
            'barrier_arr': self.barrier_arr,
            'mu_var': self.mu_var,
            'Vz_var': self.Vz_var,
            'num_eigenvalues': config.num_eigenvalues,
            'weight_threshold': config.weight_threshold,
            'separation_threshold': config.separation_threshold,
            'eng_window_range': config.eng_window_range,
            'conductance_flag': config.conductance_flag,
            'spectra_flag': config.spectra_flag,
            'localization_flag': config.localization_flag,
            'solver_type': self.solver_type,
            'calc_pfaffian': config.calc_pfaffian,
            'pfaffian_delta_N': config.pfaffian_delta_N
        }
