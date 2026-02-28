# param_handler.py
import yaml
import numpy as np
import itertools as itr
from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class SimulationConfig:
    # 1. Base Parameters (Loaded directly from YAML)
    dirname: str
    fname: str
    Ls: int
    Ln: int
    Lb: int
    Lb_pdi: int
    a0: float
    ms: float
    Delta_0: float
    gamma: float
    mu_leads: float
    barrier0: float
    mu_n: float
    Upoints: int
    num_engs: int
    mu_max: float
    mu_min: float
    mu_dist: float
    Vz_max: float
    Vz_min: float
    Vz_dist: float

    # 2. Derived Parameters 
    # field(init=False) means the dataclass won't look for these in the YAML file.
    t: float = field(init=False)
    alpha: float = field(init=False)
    Delta: float = field(init=False)
    V0: float = field(init=False)
    barrier_arr: np.ndarray = field(init=False)
    energies: np.ndarray = field(init=False)
    mu_var: np.ndarray = field(init=False)
    Vz_var: np.ndarray = field(init=False)
    params_list: List[Any] = field(init=False)

=