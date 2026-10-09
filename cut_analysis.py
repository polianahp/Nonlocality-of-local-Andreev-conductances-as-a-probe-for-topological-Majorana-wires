#!/usr/bin/env python3
import os
# Prevent thread thrashing/oversubscription across parallel workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import argparse
import logging
import gc
import json
import scipy.ndimage as ndimage

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from src.config import PathConfigs
import src.helpers as hp
from src.parameter_handler import ConfigManager
from scipy.constants import physical_constants
from scipy.ndimage import convolve1d
from src.gpu_broadening import _temp_kernel

# ==========================================
# USER CONFIGURATION
# ==========================================
#DEFAULT_DATA_DIRS = [
#    "Tdis_pfaff5_V0_0_0",
#    "Tdis_pfaff5_V0_0_1",
#    "Tdis_pfaff5_V0_0_378",
#    "Tdis_pfaff5_V0_0_645",
#    "Tdis_pfaff5_V0_0_872",
#    "Tdis_pfaff5_V0_0_91"
#]
DEFAULT_DATA_DIRS = [
    "Tdis_pfaff5",
    "Tdis_pfaff5_V0_0_0",
    "Tdis_pfaff5_V0_0_1",
    "Tdis_pfaff5_V0_0_91",
    "Tdis_pfaff5_V0_0_378",
    "Tdis_pfaff5_V0_0_645",
    "Tdis_pfaff5_V0_0_872",
]

N_CUT_POINTS = 100                           # Number of points to sample along each cut
BARRIER_SWEEP_SCALE = 70                    # Scale multiplier for barrier sweeps

# Load official parameters from custom_protocol.yaml
p = ConfigManager.get_protocol_config(PathConfigs.PARAMETERS / "default_protocol.yaml")
DERIVATIVE_THRESHOLD = p.derivative_threshold

#Cuts and the points that should mathematically "snap" to them
#NOTE: All coordinates MUST be input as (Vz, mu)
CUTS = [
    {
        "start": (0.162, 2.391), "end": (1.4, 3.496), "color": "blue", "label": "Cut_1",
        "snap_points": [
            {"raw_coords": (0.954, 3.054), "color": "red", "label": "Point 1A"},

            {"raw_coords": (1.014, 3.114), "color": "cyan", "label": "Point 1B"},

            {"raw_coords": (0.893, 3.033), "color": "green", "label": "Point 1C"},

            {"raw_coords": (1.136, 3.234), "color": "yellow", "label": "Point 1D"},

            {"raw_coords": (1.38, 3.455), "color": "purple", "label": "Point 1E"},
            

        ]
    },

    {
        "start": (0.0, 2.511), "end": (1.4, 2.511), "color": "red", "label": "Cut_2",
        "snap_points": [
            {"raw_coords": (0.730, 2.511), "color": "red", "label": "Point 2A"},

            {"raw_coords": (0.568, 2.511), "color": "yellow", "label": "Point 2B"},

            {"raw_coords": (0.994, 2.511), "color": "green", "label": "Point 2C"},
            
            {"raw_coords": (0.852, 2.511), "color": "green", "label": "Point 2D"},
        ]
    },

        {
        "start": (0.35, 2.15), "end": (1.4, 3.134), "color": "green", "label": "Cut_3",
        "snap_points": [
            {"raw_coords": (1.319, 3.074), "color": "red", "label": "Point 3A"},

            {"raw_coords": (1.096, 2.813), "color": "yellow", "label": "Point 3B"},

            {"raw_coords": (0.771, 2.511), "color": "green", "label": "Point 3C"},
        ]
    },
    {
        "start": (0.0, 0.0), "end": (1.4, 0.0), "color": "orange", "label": "Cut_4",
        "snap_points": [
            {"raw_coords": (0.2, 0.0), "color": "red", "label": "Point 4A"},
            {"raw_coords": (0.392, 0.0), "color": "orange", "label": "Point 4E"},
            {"raw_coords": (0.4, 0.0), "color": "cyan", "label": "Point 4B"},
            {"raw_coords": (0.5, 0.0), "color": "magenta", "label": "Point 4F"},
            {"raw_coords": (0.6, 0.0), "color": "green", "label": "Point 4C"},
            {"raw_coords": (0.8, 0.0), "color": "yellow", "label": "Point 4D"},
            {"raw_coords": (1.002, 0.0), "color": "lime", "label": "Point 4G"},
            {"raw_coords": (1.01, 0.0), "color": "pink", "label": "Point 4H"},
            {"raw_coords": (1.096, 0.0), "color": "purple", "label": "Point 4I"},
        ]
    }
]


# Points evaluated exactly as given (off-cut)
# NOTE: All coordinates MUST be input as (Vz, mu)
FREESTANDING_POINTS = [
    {"coords": (0.852, 2.511), "color": "purple", "label": "Point 2D"},
]

# Dataset-specific cuts dictionary for specific disorder configurations
# NOTE: All coordinates MUST be input as (Vz, mu)
DATASET_CUTS = {
    "Tdis_pfaff5_V0_0_0": [
        {
            "start": (0.0, 0.0), "end": (1.4, 0.0), "color": "blue", "label": "Cut_1",
            "snap_points": [
                {"raw_coords": (0.2, 0.0), "color": "red", "label": "Point 1A"},
                {"raw_coords": (0.392, 0.0), "color": "orange", "label": "Point 1E"},
                {"raw_coords": (0.4, 0.0), "color": "cyan", "label": "Point 1B"},
                {"raw_coords": (0.5, 0.0), "color": "magenta", "label": "Point 1F"},
                {"raw_coords": (0.6, 0.0), "color": "green", "label": "Point 1C"},
                {"raw_coords": (0.8, 0.0), "color": "yellow", "label": "Point 1D"},
                {"raw_coords": (1.002, 0.0), "color": "lime", "label": "Point 1G"},
                {"raw_coords": (1.01, 0.0), "color": "pink", "label": "Point 1H"},
                {"raw_coords": (1.096, 0.0), "color": "purple", "label": "Point 1I"},
            ]
        },
        {
            "start": (0.0, 0.43), "end": (1.4, 0.43), "color": "red", "label": "Cut_2",
            "snap_points": [
                {"raw_coords": (0.2, 0.43), "color": "red", "label": "Point 2A"},
                {"raw_coords": (0.392, 0.43), "color": "orange", "label": "Point 2E"},
                {"raw_coords": (0.4, 0.43), "color": "cyan", "label": "Point 2B"},
                {"raw_coords": (0.5, 0.43), "color": "magenta", "label": "Point 2F"},
                {"raw_coords": (0.6, 0.43), "color": "green", "label": "Point 2C"},
                {"raw_coords": (0.8, 0.43), "color": "yellow", "label": "Point 2D"},
                {"raw_coords": (1.002, 0.43), "color": "lime", "label": "Point 2G"},
                {"raw_coords": (1.01, 0.43), "color": "pink", "label": "Point 2H"},
                {"raw_coords": (1.096, 0.43), "color": "purple", "label": "Point 2I"},
            ]
        },
    ],
    "Tdis_pfaff5": [
        {
            "start": (0.0, 0.0), "end": (1.4, 0.0), "color": "blue", "label": "Cut_1",
            "snap_points": [
                {"raw_coords": (0.2, 0.0), "color": "red", "label": "Point 1A"},
                {"raw_coords": (0.392, 0.0), "color": "orange", "label": "Point 1E"},
                {"raw_coords": (0.4, 0.0), "color": "cyan", "label": "Point 1B"},
                {"raw_coords": (0.5, 0.0), "color": "magenta", "label": "Point 1F"},
                {"raw_coords": (0.6, 0.0), "color": "green", "label": "Point 1C"},
                {"raw_coords": (0.8, 0.0), "color": "yellow", "label": "Point 1D"},
                {"raw_coords": (1.002, 0.0), "color": "lime", "label": "Point 1G"},
                {"raw_coords": (1.01, 0.0), "color": "pink", "label": "Point 1H"},
                {"raw_coords": (1.096, 0.0), "color": "purple", "label": "Point 1I"},
            ]
        },
        {
            "start": (0.0, 0.43), "end": (1.4, 0.43), "color": "red", "label": "Cut_2",
            "snap_points": [
                {"raw_coords": (0.2, 0.43), "color": "red", "label": "Point 2A"},
                {"raw_coords": (0.392, 0.43), "color": "orange", "label": "Point 2E"},
                {"raw_coords": (0.4, 0.43), "color": "cyan", "label": "Point 2B"},
                {"raw_coords": (0.5, 0.43), "color": "magenta", "label": "Point 2F"},
                {"raw_coords": (0.6, 0.43), "color": "green", "label": "Point 2C"},
                {"raw_coords": (0.8, 0.43), "color": "yellow", "label": "Point 2D"},
                {"raw_coords": (1.002, 0.43), "color": "lime", "label": "Point 2G"},
                {"raw_coords": (1.01, 0.43), "color": "pink", "label": "Point 2H"},
                {"raw_coords": (1.096, 0.43), "color": "purple", "label": "Point 2I"},
            ]
        },
    ],
    "Tdis_pfaff5_V0_0_1": [
        {
            "start": (0.0, 0.0), "end": (1.4, 0.0), "color": "blue", "label": "Cut_1",
            "snap_points": [
                {"raw_coords": (0.2, 0.0), "color": "red", "label": "Point 1A"},
                {"raw_coords": (0.392, 0.0), "color": "orange", "label": "Point 1E"},
                {"raw_coords": (0.4, 0.0), "color": "cyan", "label": "Point 1B"},
                {"raw_coords": (0.5, 0.0), "color": "magenta", "label": "Point 1F"},
                {"raw_coords": (0.6, 0.0), "color": "green", "label": "Point 1C"},
                {"raw_coords": (0.8, 0.0), "color": "yellow", "label": "Point 1D"},
                {"raw_coords": (1.002, 0.0), "color": "lime", "label": "Point 1G"},
                {"raw_coords": (1.01, 0.0), "color": "pink", "label": "Point 1H"},
                {"raw_coords": (1.096, 0.0), "color": "purple", "label": "Point 1I"},
            ]
        },
        {
            "start": (0.0, 0.432), "end": (1.4, 0.432), "color": "red", "label": "Cut_2",
            "snap_points": [
                {"raw_coords": (0.2, 0.432), "color": "red", "label": "Point 2A"},
                {"raw_coords": (0.392, 0.432), "color": "orange", "label": "Point 2E"},
                {"raw_coords": (0.4, 0.432), "color": "cyan", "label": "Point 2B"},
                {"raw_coords": (0.5, 0.432), "color": "magenta", "label": "Point 2F"},
                {"raw_coords": (0.6, 0.432), "color": "green", "label": "Point 2C"},
                {"raw_coords": (0.8, 0.432), "color": "yellow", "label": "Point 2D"},
                {"raw_coords": (1.002, 0.432), "color": "lime", "label": "Point 2G"},
                {"raw_coords": (1.01, 0.432), "color": "pink", "label": "Point 2H"},
                {"raw_coords": (1.096, 0.432), "color": "purple", "label": "Point 2I"},
            ]
        },
        {
            "start": (0.0, -0.05), "end": (1.4, -0.05), "color": "green", "label": "Cut_3",
            "snap_points": [
                {"raw_coords": (0.2, -0.05), "color": "red", "label": "Point 3A"},
                {"raw_coords": (0.392, -0.05), "color": "orange", "label": "Point 3E"},
                {"raw_coords": (0.4, -0.05), "color": "cyan", "label": "Point 3B"},
                {"raw_coords": (0.5, -0.05), "color": "magenta", "label": "Point 3F"},
                {"raw_coords": (0.6, -0.05), "color": "green", "label": "Point 3C"},
                {"raw_coords": (0.8, -0.05), "color": "yellow", "label": "Point 3D"},
                {"raw_coords": (1.002, -0.05), "color": "lime", "label": "Point 3G"},
                {"raw_coords": (1.01, -0.05), "color": "pink", "label": "Point 3H"},
                {"raw_coords": (1.096, -0.05), "color": "purple", "label": "Point 3I"},
            ]
        },
    ],
    "Tdis_pfaff5_V0_0_378": [
        {
            "start": (0.0, 0.312), "end": (1.4, 0.312), "color": "blue", "label": "Cut_1",
            "snap_points": [
                {"raw_coords": (0.2, 0.312), "color": "red", "label": "Point 1A"},
                {"raw_coords": (0.4, 0.312), "color": "cyan", "label": "Point 1B"},
                {"raw_coords": (0.6, 0.312), "color": "green", "label": "Point 1C"},
                {"raw_coords": (0.8, 0.312), "color": "yellow", "label": "Point 1D"},
            ]
        },
        {
            "start": (0.0, -0.26), "end": (1.4, -0.26), "color": "red", "label": "Cut_2",
            "snap_points": [
                {"raw_coords": (0.2, -0.26), "color": "red", "label": "Point 2A"},
                {"raw_coords": (0.4, -0.26), "color": "cyan", "label": "Point 2B"},
                {"raw_coords": (0.6, -0.26), "color": "green", "label": "Point 2C"},
                {"raw_coords": (0.8, -0.26), "color": "yellow", "label": "Point 2D"},
            ]
        },
        {
            "start": (0.0, 0.26), "end": (1.4, 0.26), "color": "green", "label": "Cut_3",
            "snap_points": [
                {"raw_coords": (0.2, 0.26), "color": "red", "label": "Point 3A"},
                {"raw_coords": (0.4, 0.26), "color": "cyan", "label": "Point 3B"},
                {"raw_coords": (0.6, 0.26), "color": "green", "label": "Point 3C"},
                {"raw_coords": (0.8, 0.26), "color": "yellow", "label": "Point 3D"},
            ]
        },
        {
            "start": (0.041, 0.150), "end": (1.339, 1.317), "color": "purple", "label": "Cut_4",
            "snap_points": [
                {"raw_coords": (0.2, 0.150 + ((1.317 - 0.150) / (1.339 - 0.041)) * (0.2 - 0.041)), "color": "red", "label": "Point 4A"},
                {"raw_coords": (0.4, 0.150 + ((1.317 - 0.150) / (1.339 - 0.041)) * (0.4 - 0.041)), "color": "cyan", "label": "Point 4B"},
                {"raw_coords": (0.6, 0.150 + ((1.317 - 0.150) / (1.339 - 0.041)) * (0.6 - 0.041)), "color": "green", "label": "Point 4C"},
                {"raw_coords": (0.8, 0.150 + ((1.317 - 0.150) / (1.339 - 0.041)) * (0.8 - 0.041)), "color": "yellow", "label": "Point 4D"},
            ]
        },
    ],
    "Tdis_pfaff5_V0_0_645": [
        {
            "start": (0.446, 1.868), "end": (1.359, 2.752), "color": "blue", "label": "Cut_1",
            "snap_points": [
                {"raw_coords": (1.177, 2.551), "color": "red", "label": "Point 1A"},
                {"raw_coords": (0.913, 2.290), "color": "cyan", "label": "Point 1B"},
                {"raw_coords": (0.649, 2.069), "color": "green", "label": "Point 1C"},
            ]
        },
        {
            "start": (0.203, 0.864), "end": (1.319, 2.009), "color": "red", "label": "Cut_2",
            "snap_points": [
                {"raw_coords": (0.974, 1.647), "color": "red", "label": "Point 2A"},
                {"raw_coords": (0.670, 1.306), "color": "cyan", "label": "Point 2B"},
                {"raw_coords": (0.406, 1.065), "color": "green", "label": "Point 2C"},
            ]
        },
        {
            "start": (0.0, 1.125), "end": (1.4, 1.25), "color": "green", "label": "Cut_3",
            "snap_points": [
                {"raw_coords": (0.2, 1.125 + ((1.25 - 1.125) / 1.4) * 0.2), "color": "red", "label": "Point 3A"},
                {"raw_coords": (0.364, 1.125 + ((1.25 - 1.125) / 1.4) * 0.364), "color": "cyan", "label": "Point 3B"},
                {"raw_coords": (0.933, 1.125 + ((1.25 - 1.125) / 1.4) * 0.933), "color": "green", "label": "Point 3C"},
                {"raw_coords": (1.238, 1.125 + ((1.25 - 1.125) / 1.4) * 1.238), "color": "yellow", "label": "Point 3D"},
            ]
        },
        {
            "start": (0.4, 0.0), "end": (1.4, 0.984), "color": "purple", "label": "Cut_4",
            "snap_points": [
                {"raw_coords": (1.278, 1.065), "color": "red", "label": "Point 4A"},
                {"raw_coords": (0.893, 0.703), "color": "cyan", "label": "Point 4B"},
                {"raw_coords": (0.528, 0.321), "color": "green", "label": "Point 4C"},
                {"raw_coords": (0.386, 0.0), "color": "yellow", "label": "Point 4D"},
                {"raw_coords": (0.812, ((0.984) / (1.4 - 0.4)) * (0.812 - 0.4)), "color": "orange", "label": "Point 4E"},
                {"raw_coords": (1.207, ((0.984) / (1.4 - 0.4)) * (1.207 - 0.4)), "color": "magenta", "label": "Point 4F"},
            ]
        },
        {
            "start": (0.203, 0.7396), "end": (1.319, 1.8846), "color": "teal", "label": "Cut_5",
            "snap_points": [
                {"raw_coords": (0.872, 1.426), "color": "red", "label": "Point 5A"},
                {"raw_coords": (0.974, 1.426 + ((2.009 - 0.864) / (1.319 - 0.203)) * (0.974 - 0.872)), "color": "cyan", "label": "Point 5B"},
                {"raw_coords": (0.670, 1.426 + ((2.009 - 0.864) / (1.319 - 0.203)) * (0.670 - 0.872)), "color": "green", "label": "Point 5C"},
                {"raw_coords": (0.406, 1.426 + ((2.009 - 0.864) / (1.319 - 0.203)) * (0.406 - 0.872)), "color": "yellow", "label": "Point 5D"},
            ]
        },
    ],
    "Tdis_pfaff5_V0_0_91": [
        {
            "start": (0.345, 1.748), "end": (1.4, 2.712), "color": "blue", "label": "Cut_1",
            "snap_points": [
                {"raw_coords": (0.609, 1.748 + ((2.712 - 1.748) / (1.4 - 0.345)) * (0.609 - 0.345)), "color": "red", "label": "Point 1A"},
                {"raw_coords": (0.852, 1.748 + ((2.712 - 1.748) / (1.4 - 0.345)) * (0.852 - 0.345)), "color": "cyan", "label": "Point 1B"},
                {"raw_coords": (1.075, 1.748 + ((2.712 - 1.748) / (1.4 - 0.345)) * (1.075 - 0.345)), "color": "green", "label": "Point 1C"},
            ]
        },
        {
            "start": (0.0, 1.969), "end": (1.4, 1.969), "color": "red", "label": "Cut_2",
            "snap_points": [
                {"raw_coords": (0.6, 1.969), "color": "red", "label": "Point 2A"},
                {"raw_coords": (1.197, 1.969), "color": "cyan", "label": "Point 2B"},
            ]
        },
        {
            "start": (0.0, 1.346), "end": (1.4, 1.346), "color": "green", "label": "Cut_3",
            "snap_points": [
                {"raw_coords": (0.67, 1.346), "color": "red", "label": "Point 3A"},
                {"raw_coords": (0.832, 1.346), "color": "cyan", "label": "Point 3B"},
                {"raw_coords": (1.055, 1.346), "color": "green", "label": "Point 3C"},
            ]
        },
    ],
    "Tdis_pfaff5_V0_0_872": [
        {
            "start": (0.345, 1.723), "end": (1.4, 2.687), "color": "blue", "label": "Cut_1",
            "snap_points": [
                {"raw_coords": (1.055, 2.3718), "color": "red", "label": "Point 1A"},
            ]
        },
        {
            "start": (0.0, 1.969), "end": (1.4, 1.969), "color": "red", "label": "Cut_2",
            "snap_points": [
                {"raw_coords": (0.6, 1.969), "color": "red", "label": "Point 2A"},
                {"raw_coords": (1.197, 1.969), "color": "cyan", "label": "Point 2B"},
            ]
        },
        {
            "start": (0.0, 1.366), "end": (1.4, 1.366), "color": "green", "label": "Cut_3",
            "snap_points": [
                {"raw_coords": (0.933, 1.366), "color": "red", "label": "Point 3A"},
            ]
        },
    ],
}

# ==========================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# MULTIPROCESSING WORKERS 
# ==========================================
def _eval_cut_point_worker(args):
    """
    Evaluates closed system Hamiltonian for a single cut trajectory point.
    Unifies spectrum and MZM overlap using a single shift-invert solve_ham(k=kvals) call.
    """
    idx, mu_val, vz_val, t_val, gamma, Delta0, alpha, Ls, Vdisx, kvals = args
    scl = hp.build_system_closed(t_val, mu_val, gamma, Delta0, vz_val, alpha, Ls, Vdisx, a=1)
    evals, evecs = hp.solve_ham(scl, k=kvals, solver_type='cpu')
    sorted_evals = np.sort(evals)
    abs_idx = np.argsort(np.abs(evals))[:2]
    try:
        rho_M1, rho_M2, _ = hp.get_psiM_density(evals[abs_idx], evecs[:, abs_idx])
        overlap = float(hp.calc_normalized_mzm_overlap(rho_M1, rho_M2))
    except Exception:
        overlap = 0.0
    return idx, sorted_evals, overlap


def _process_deep_dive_point_worker(args):
    """
    Evaluates differential conductance, Majorana wavefunctions, and barrier sweeps
    for a single deep-dive point in parallel.
    Uses calc_ldos=False and shift-invert sparse eigensolver solve_ham(k=2).
    """
    pt, physics_params, p_T_mK_stage1, pt_2w, pt_3w, barrier_sweep_scale = args
    pt_dir = Path(pt['dir_path'])
    pt_dir.mkdir(parents=True, exist_ok=True)
    mu_val, vz_val = pt['coords']

    t_val = physics_params['t_val']
    mu_n = physics_params['mu_n']
    mu_leads = physics_params['mu_leads']
    gamma = physics_params['gamma']
    Delta0 = physics_params['Delta0']
    alpha = physics_params['alpha']
    Ln = physics_params['Ln']
    Lb = physics_params['Lb']
    Ls = physics_params['Ls']
    barrier_l_base = physics_params['barrier_l_base']
    Vdisx = physics_params['Vdisx']

    # A. Differential Conductance dI/dV (Pruning unneeded LDOS)
    syst = hp.build_system(t_val, mu_val, mu_n, gamma, Delta0, vz_val, alpha, Ln, Lb, Ls, mu_leads, barrier_l_base, barrier_l_base, Vdisx)
    energies = np.linspace(-0.5, 0.5, 101)
    dIdV_left, dIdV_right, _, _, _ = hp.calc_dIdV(syst, energies, calc_ldos=False)

    # Apply Thermal Broadening if T_mK > 0
    k_B = physical_constants["Boltzmann constant in eV/K"][0]
    T_meV = k_B * (p_T_mK_stage1 * 1e-3) * 1e3
    if T_meV > 0.0:
        K = _temp_kernel(energies, T_meV)
        delta_bias = np.diff(energies)[0]
        K_norm = K * delta_bias
        dIdV_left = convolve1d(dIdV_left, K_norm, mode='constant', cval=0.0)
        dIdV_right = convolve1d(dIdV_right, K_norm, mode='constant', cval=0.0)

    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    ax.plot(energies, dIdV_left, label="Left dI/dV", color='royalblue')
    ax.plot(energies, dIdV_right, label="Right dI/dV", color='darkorange')
    ax.plot([], [], color=pt['color'], marker='*', markersize=10, linestyle='None', label=f"Map Marker ({pt['color']})")
    ax.set_title(f"dI/dV for {pt['label']} [Color: {pt['color']}]\n(mu={mu_val:.3f}, Vz={vz_val:.3f}) | 2ω_L={pt_2w:.3f}, 3ω_L={pt_3w:.3f}")
    ax.set_xlabel("Energy (meV)")
    ax.set_ylabel("Conductance")
    ax.legend()
    plt.tight_layout()
    fig.savefig(pt_dir / f"{pt['label']}_{pt['color']}_dIdV.png", dpi=150)
    plt.close(fig)

    # B. Majorana Wave Functions (Sparse Shift-Invert solve_ham k=2)
    scl = hp.build_system_closed(t_val, mu_val, gamma, Delta0, vz_val, alpha, Ls, Vdisx, a=1)
    evals_sub, evecs_sub = hp.solve_ham(scl, k=2, solver_type='cpu')
    rho_M1, rho_M2, _ = hp.get_psiM_density(evals_sub, evecs_sub)

    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    ax.plot(rho_M1, label="Majorana Left (M1)", color='cyan')
    ax.plot(rho_M2, label="Majorana Right (M2)", color='orange')
    ax.set_title(f"Wavefunctions for {pt['label']}")
    ax.set_xlabel("Site Index")
    ax.set_ylabel("Probability Density")
    ax.legend()
    plt.tight_layout()
    fig.savefig(pt_dir / f"{pt['label']}_{pt['color']}_wavefunctions.png", dpi=150)
    plt.close(fig)

    # C. Barrier Sweeps
    barrier_sweep_vals = np.linspace(-barrier_sweep_scale * barrier_l_base, barrier_sweep_scale * barrier_l_base, 50)

    gL_varying_R, gR_varying_R = [], []
    for br in barrier_sweep_vals:
        s = hp.build_system(t_val, mu_val, mu_n, gamma, Delta0, vz_val, alpha, Ln, Lb, Ls, mu_leads, barrier_l_base, br, Vdisx)
        cL, cR = hp.calc_conductance(s, energy=0.0)
        gL_varying_R.append(cL)
        gR_varying_R.append(cR)

    gL_varying_L, gR_varying_L = [], []
    for bl in barrier_sweep_vals:
        s = hp.build_system(t_val, mu_val, mu_n, gamma, Delta0, vz_val, alpha, Ln, Lb, Ls, mu_leads, bl, barrier_l_base, Vdisx)
        cL, cR = hp.calc_conductance(s, energy=0.0)
        gL_varying_L.append(cL)
        gR_varying_L.append(cR)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.8))

    x_data_R = barrier_sweep_vals / barrier_l_base
    axes[0].plot(x_data_R, np.array(gL_varying_R) / gL_varying_R[len(gL_varying_R)//2], label="G_LL", color='green')
    axes[0].plot(x_data_R, np.array(gR_varying_R) / gR_varying_R[len(gR_varying_R)//2], label="G_RR", color='blue')
    axes[0].set_title(f"Varying Right Barrier ({pt['label']})")
    axes[0].set_xlabel("U_R / U_L")
    axes[0].set_ylabel("Normalized Conductance")
    axes[0].legend()

    x_data_L = barrier_sweep_vals / barrier_l_base
    axes[1].plot(x_data_L, np.array(gL_varying_L) / gL_varying_L[len(gL_varying_L)//2], label="G_LL", color='green')
    axes[1].plot(x_data_L, np.array(gR_varying_L) / gR_varying_L[len(gR_varying_L)//2], label="G_RR", color='blue')
    axes[1].set_title(f"Varying Left Barrier ({pt['label']})")
    axes[1].set_xlabel("U_L / U_R")
    axes[1].set_ylabel("Normalized Conductance")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(pt_dir / f"{pt['label']}_{pt['color']}_barrier_asymmetry.png", dpi=150)
    plt.close(fig)
    return pt['label']


def _extract_single_gap_worker(args):
    """
    Worker function to extract transport gap for a single parameter configuration.
    Runs tgp.two.extract_gap on an independent dataset instance.
    """
    tag, tprep_path_str, gap_th_factor, upper_th, noise_th = args
    import tgp
    import xarray as xr
    tprep_local = xr.open_dataset(tprep_path_str)
    tl_local = tprep_local.rename({"bias": "left_bias"})
    tr_local = tprep_local.rename({"bias": "right_bias"})
    tl_res, tr_res = tgp.two.extract_gap(
        tl_local,
        tr_local,
        gap_threshold_factor=gap_th_factor,
        upper_conductance_threshold=upper_th,
        noise_threshold=noise_th
    )
    g_avg = 0.5 * (tl_res.gap.mean(dim='cutter_pair_index') + tr_res.gap.mean(dim='cutter_pair_index'))
    val_2d = g_avg.transpose('V', 'B').values
    tprep_local.close()
    return tag, val_2d

def generate_point_path(pdi_data, N, resl, mu_start, mu_end, Vz_start, Vz_end):
    pdi_params = pdi_data[:, 0:2]
    diff_vec = np.array([mu_end - mu_start, Vz_end - Vz_start])
    total_distance = np.linalg.norm(diff_vec)

    if total_distance == 0:
        return np.array([]), np.array([])

    unit_vec = diff_vec / total_distance
    step_vec = resl * unit_vec 
    num_pts = int(np.floor(total_distance / resl)) + 1

    start_vec = np.array([mu_start, Vz_start])
    pts = np.asarray([start_vec + (n * step_vec) for n in range(num_pts)])

    closest_indices = []
    for i in range(num_pts):
        tst = pts[i, :]
        is_close_mask = np.all(np.isclose(tst, pdi_params, atol=resl), axis=1)
        matched_indices = np.where(is_close_mask)[0]
        if len(matched_indices) > 0:
            matched_pdi_params = pdi_params[matched_indices]
            distances = np.linalg.norm(matched_pdi_params - tst, axis=1)
            closest_indices.append(matched_indices[np.argmin(distances)])
        else:
            closest_indices.append(-1)

    valid_indices = np.array(closest_indices)[np.array(closest_indices) != -1]
    if len(valid_indices) == 0:
        return np.array([]), np.array([])

    unique_indices = np.unique(valid_indices)
    unique_points = pdi_params[unique_indices]
    dist_from_start = np.linalg.norm(unique_points - start_vec, axis=1)
    sort_order = np.argsort(dist_from_start)

    sorted_unique_points = unique_points[sort_order]
    sorted_unique_indices = unique_indices[sort_order]
    
    if N >= len(sorted_unique_points):
        return sorted_unique_points, sorted_unique_indices
    else:
        sample_idx = np.round(np.linspace(0, len(sorted_unique_points) - 1, N)).astype(int)
        return sorted_unique_points[sample_idx], sorted_unique_indices[sample_idx]

def export_phase_map_pair(name_prefix, z_left, z_right, z_inv, B_vals, V_vals, 
                          zmin, zmax, cmap_mpl, cmap_plotly, label_left, label_right, 
                          cbar_label, resolved_cuts, freestanding_points, out_dir,
                          contour_color='black', contour_dash='solid'):
    # Interactive HTML
    fig_html = make_subplots(rows=1, cols=2, subplot_titles=(label_left, label_right), shared_yaxes=True, horizontal_spacing=0.05)
    
    def add_html_overlays(col_idx):
        line_dict = dict(color=contour_color, width=2)
        if contour_dash != 'solid':
            line_dict['dash'] = contour_dash
        fig_html.add_trace(go.Contour(z=z_inv, x=B_vals, y=V_vals, contours=dict(start=0, end=0, size=1), contours_coloring='lines', line=line_dict, showscale=False, hoverinfo='skip'), row=1, col=col_idx)
        for rcut in resolved_cuts:
            cut = rcut['config']
            fig_html.add_trace(go.Scatter(x=[cut['start'][1], cut['end'][1]], y=[cut['start'][0], cut['end'][0]], mode='lines', line=dict(color=cut['color'], width=1.5), name=cut['label'], showlegend=(col_idx==1)), row=1, col=col_idx)
            for sp in rcut['resolved_snaps']:
                fig_html.add_trace(go.Scatter(x=[sp['raw_coords'][1]], y=[sp['raw_coords'][0]], mode='markers', marker=dict(symbol='circle-open', color=sp['color'], size=5, line=dict(width=1)), name=f"{sp['label']} (Raw)", showlegend=False), row=1, col=col_idx)
                fig_html.add_trace(go.Scatter(x=[sp['snapped_coords'][1]], y=[sp['snapped_coords'][0]], mode='markers', marker=dict(symbol='star', color=sp['color'], size=6), name=f"{sp['label']} (Snapped)", showlegend=False), row=1, col=col_idx)
                fig_html.add_trace(go.Scatter(x=[sp['raw_coords'][1], sp['snapped_coords'][1]], y=[sp['raw_coords'][0], sp['snapped_coords'][0]], mode='lines', line=dict(color=sp['color'], width=0.8, dash='dot'), showlegend=False, hoverinfo='skip'), row=1, col=col_idx)
        for fp in freestanding_points:
            fig_html.add_trace(go.Scatter(x=[fp['coords'][1]], y=[fp['coords'][0]], mode='markers', marker=dict(symbol='star', color=fp['color'], size=6), name=fp['label'], showlegend=(col_idx==1)), row=1, col=col_idx)

    hover_temp = "<b>B (Vz)</b>: %{x:.3f} meV<br><b>V (mu)</b>: %{y:.3f} meV<br><b>Value</b>: %{z:.3f}<extra></extra>"
    fig_html.add_trace(go.Heatmap(z=z_left, x=B_vals, y=V_vals, zmin=zmin, zmax=zmax, colorscale=cmap_plotly, showscale=False, hovertemplate=hover_temp), row=1, col=1)
    fig_html.add_trace(go.Heatmap(z=z_right, x=B_vals, y=V_vals, zmin=zmin, zmax=zmax, colorscale=cmap_plotly, colorbar=dict(title=cbar_label), hovertemplate=hover_temp), row=1, col=2)
    add_html_overlays(1)
    add_html_overlays(2)

    fig_html.update_layout(title=f"Interactive Phase Map: {cbar_label}", xaxis_title="Zeeman Field Vz (B) [meV]", yaxis_title="Chemical Potential µ (V) [meV]", xaxis2_title="Zeeman Field Vz (B) [meV]", width=1200, height=700, legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.8)'))
    fig_html.write_html(str(out_dir / f"{name_prefix}_phase_map.html"))

    # Static PNG
    fig_map, axes = plt.subplots(1, 2, figsize=(15.5, 8.5), dpi=300, sharey=True, layout='constrained')
    im1 = axes[0].pcolormesh(B_vals, V_vals, z_left, cmap=cmap_mpl, vmin=zmin, vmax=zmax, shading='nearest')
    im2 = axes[1].pcolormesh(B_vals, V_vals, z_right, cmap=cmap_mpl, vmin=zmin, vmax=zmax, shading='nearest')
    
    for ax in axes:
        ax.contour(B_vals, V_vals, z_inv, levels=[0], colors=contour_color, linewidths=1.5, linestyles='dashed' if contour_dash != 'solid' else 'solid')
        for rcut in resolved_cuts:
            cut = rcut['config']
            ax.plot([cut['start'][1], cut['end'][1]], [cut['start'][0], cut['end'][0]], color=cut['color'], linewidth=1.2, label=cut['label'])
            for sp in rcut['resolved_snaps']:
                ax.plot(sp['raw_coords'][1], sp['raw_coords'][0], marker='o', markerfacecolor='none', markeredgecolor=sp['color'], markersize=4, linestyle='None')
                ax.plot(sp['snapped_coords'][1], sp['snapped_coords'][0], marker='*', color=sp['color'], markersize=6, linestyle='None')
                ax.plot([sp['raw_coords'][1], sp['snapped_coords'][1]], [sp['raw_coords'][0], sp['snapped_coords'][0]], color=sp['color'], linestyle=':', linewidth=0.8)
        for fp in freestanding_points:
            ax.plot(fp['coords'][1], fp['coords'][0], marker='*', color=fp['color'], markersize=6, label=fp['label'], linestyle='None')
        ax.set_xlabel(r"Zeeman Field $V_z$ (meV)")
        
    axes[0].set_ylabel(r"Chemical Potential $\mu$ (meV)")
    axes[0].set_title(label_left)
    axes[1].set_title(label_right)
    axes[0].legend(loc='upper left', framealpha=0.8)
    fig_map.colorbar(im2, ax=axes, label=cbar_label)
    fig_map.suptitle(f"Global Phase Map: {cbar_label}")
    fig_map.savefig(out_dir / f"{name_prefix}_phase_map.png", bbox_inches='tight')
    plt.close(fig_map)



def export_phase_map_single(name_prefix, z_data, z_inv, B_vals, V_vals, 
                            zmin, zmax, cmap_mpl, cmap_plotly, label, 
                            cbar_label, resolved_cuts, freestanding_points, out_dir,
                            contour_color='black', contour_dash='solid'):
    # Interactive HTML
    fig_html = go.Figure()
    
    hover_temp = "<b>B (Vz)</b>: %{x:.3f} meV<br><b>V (mu)</b>: %{y:.3f} meV<br><b>Value</b>: %{z:.3f}<extra></extra>"
    fig_html.add_trace(go.Heatmap(z=z_data, x=B_vals, y=V_vals, zmin=zmin, zmax=zmax, colorscale=cmap_plotly, colorbar=dict(title=cbar_label), hovertemplate=hover_temp))

    line_dict = dict(color=contour_color, width=2)
    if contour_dash != 'solid':
        line_dict['dash'] = contour_dash
    fig_html.add_trace(go.Contour(z=z_inv, x=B_vals, y=V_vals, contours=dict(start=0, end=0, size=1), contours_coloring='lines', line=line_dict, showscale=False, hoverinfo='skip'))
    
    for rcut in resolved_cuts:
        cut = rcut['config']
        fig_html.add_trace(go.Scatter(x=[cut['start'][1], cut['end'][1]], y=[cut['start'][0], cut['end'][0]], mode='lines', line=dict(color=cut['color'], width=1.5), name=cut['label'], showlegend=True))
        for sp in rcut['resolved_snaps']:
            fig_html.add_trace(go.Scatter(x=[sp['raw_coords'][1]], y=[sp['raw_coords'][0]], mode='markers', marker=dict(symbol='circle-open', color=sp['color'], size=5, line=dict(width=1)), name=f"{sp['label']} (Raw)", showlegend=False))
            fig_html.add_trace(go.Scatter(x=[sp['snapped_coords'][1]], y=[sp['snapped_coords'][0]], mode='markers', marker=dict(symbol='star', color=sp['color'], size=6), name=f"{sp['label']} (Snapped)", showlegend=False))
            fig_html.add_trace(go.Scatter(x=[sp['raw_coords'][1], sp['snapped_coords'][1]], y=[sp['raw_coords'][0], sp['snapped_coords'][0]], mode='lines', line=dict(color=sp['color'], width=0.8, dash='dot'), showlegend=False, hoverinfo='skip'))
    for fp in freestanding_points:
        fig_html.add_trace(go.Scatter(x=[fp['coords'][1]], y=[fp['coords'][0]], mode='markers', marker=dict(symbol='star', color=fp['color'], size=6), name=fp['label'], showlegend=True))

    fig_html.update_layout(title=f"Interactive Phase Map: {cbar_label}", xaxis_title="Zeeman Field Vz (B) [meV]", yaxis_title="Chemical Potential µ (V) [meV]", width=800, height=700, legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.8)'))
    fig_html.write_html(str(out_dir / f"{name_prefix}_phase_map.html"))

    # Static PNG
    fig_map, ax = plt.subplots(figsize=(8.5, 8.5), dpi=300, layout='constrained')
    im1 = ax.pcolormesh(B_vals, V_vals, z_data, cmap=cmap_mpl, vmin=zmin, vmax=zmax, shading='nearest')
    
    ax.contour(B_vals, V_vals, z_inv, levels=[0], colors=contour_color, linewidths=1.5, linestyles='dashed' if contour_dash != 'solid' else 'solid')
    for rcut in resolved_cuts:
        cut = rcut['config']
        ax.plot([cut['start'][1], cut['end'][1]], [cut['start'][0], cut['end'][0]], color=cut['color'], linewidth=1.2, label=cut['label'])
        for sp in rcut['resolved_snaps']:
            ax.plot(sp['raw_coords'][1], sp['raw_coords'][0], marker='o', markerfacecolor='none', markeredgecolor=sp['color'], markersize=4, linestyle='None')
            ax.plot(sp['snapped_coords'][1], sp['snapped_coords'][0], marker='*', color=sp['color'], markersize=6, linestyle='None')
            ax.plot([sp['raw_coords'][1], sp['snapped_coords'][1]], [sp['raw_coords'][0], sp['snapped_coords'][0]], color=sp['color'], linestyle=':', linewidth=0.8)
    for fp in freestanding_points:
        ax.plot(fp['coords'][1], fp['coords'][0], marker='*', color=fp['color'], markersize=6, label=fp['label'], linestyle='None')
    
    ax.set_xlabel(r"Zeeman Field $V_z$ (meV)")
    ax.set_ylabel(r"Chemical Potential $\mu$ (meV)")
    ax.set_title(label)
    ax.legend(loc='upper left', framealpha=0.8)
    fig_map.colorbar(im1, ax=ax, label=cbar_label)
    fig_map.suptitle(f"Global Phase Map: {cbar_label}")
    fig_map.savefig(out_dir / f"{name_prefix}_phase_map.png", bbox_inches='tight')
    plt.close(fig_map)


def export_cut_conductance_plots(cut_dir, cut, pts, rcut, tprep, tprep_left, tprep_right, has_tgp_gap, selected_cutter=0, evals=None, pfaffians=None, corr_thresh=None, tgp_roi2=None):
    """
    Exports 4 individual conductance plots along a cut (matching tgp.plot.paper.plot_conductance):
      1. G_LL vs (cut_coord, left_bias) with colormap 'viridis'
      2. G_RR vs (cut_coord, right_bias) with colormap 'viridis'
      3. A(G_RL) = 0.5*(G_RL(Vb) - G_RL(-Vb)) vs (cut_coord, left_bias) with colormap 'PuOr_r' and ±Delta overlay
      4. A(G_LR) = 0.5*(G_LR(Vb) - G_LR(-Vb)) vs (cut_coord, right_bias) with colormap 'PuOr_r' and ±Delta overlay
    Also exports an integrated 5-panel multiplot (multi_panel_conductance.png) showing spectra above these 4 heatmaps.
    """
    actual_N = len(pts)
    if actual_N == 0:
        return

    mu_pts = pts[:, 0]
    vz_pts = pts[:, 1]
    bias = tprep['bias'].values
    bias_uV = 1e3 * bias

    if selected_cutter is not None and 'cutter_pair_index' in tprep.dims:
        tprep_sub = tprep.sel(cutter_pair_index=selected_cutter)
    else:
        tprep_sub = tprep.mean(dim='cutter_pair_index')

    g_ll_cut = np.zeros((actual_N, len(bias)))
    g_rr_cut = np.zeros((actual_N, len(bias)))
    g_rl_cut = np.zeros((actual_N, len(bias)))
    g_lr_cut = np.zeros((actual_N, len(bias)))
    gap_l_cut = np.zeros(actual_N)
    gap_r_cut = np.zeros(actual_N)

    for i in range(actual_N):
        mu_val = mu_pts[i]
        vz_val = vz_pts[i]
        g_ll_cut[i, :] = tprep_sub.g_ll.sel(V=mu_val, B=vz_val, method='nearest').values
        g_rr_cut[i, :] = tprep_sub.g_rr.sel(V=mu_val, B=vz_val, method='nearest').values
        g_rl_cut[i, :] = tprep_sub.g_rl.sel(V=mu_val, B=vz_val, method='nearest').values
        g_lr_cut[i, :] = tprep_sub.g_lr.sel(V=mu_val, B=vz_val, method='nearest').values

        if has_tgp_gap and tprep_left is not None and tprep_right is not None:
            if selected_cutter is not None and 'cutter_pair_index' in tprep_left.gap.dims:
                gap_l_cut[i] = tprep_left.gap.sel(cutter_pair_index=selected_cutter).sel(V=mu_val, B=vz_val, method='nearest').values.item()
                gap_r_cut[i] = tprep_right.gap.sel(cutter_pair_index=selected_cutter).sel(V=mu_val, B=vz_val, method='nearest').values.item()
            else:
                gap_l_cut[i] = tprep_left.gap.mean(dim='cutter_pair_index').sel(V=mu_val, B=vz_val, method='nearest').values.item()
                gap_r_cut[i] = tprep_right.gap.mean(dim='cutter_pair_index').sel(V=mu_val, B=vz_val, method='nearest').values.item()

    # Antisymmetrize non-local conductances
    a_grl_cut = 0.5 * (g_rl_cut - g_rl_cut[:, ::-1])
    a_glr_cut = 0.5 * (g_lr_cut - g_lr_cut[:, ::-1])

    # Coordinate determination
    vz_range = np.max(vz_pts) - np.min(vz_pts)
    mu_range = np.max(mu_pts) - np.min(mu_pts)

    if vz_range > 1e-4 and mu_range < 1e-4:
        x_coords = vz_pts
        is_diagonal = False
        xlabel = r"Zeeman Field $V_z$ (meV)"
        sub_title = f"($\\mu = {mu_pts[0]:.3f}$ meV)"
    elif mu_range > 1e-4 and vz_range < 1e-4:
        x_coords = mu_pts
        is_diagonal = False
        xlabel = r"Chemical Potential $\mu$ (meV)"
        sub_title = f"($V_z = {vz_pts[0]:.3f}$ meV)"
    else:
        x_coords = np.arange(actual_N)
        is_diagonal = True
        xlabel = r"Zeeman Field $V_z$ (meV)"
        sub_title = f"(Path: $\\mu \\in [{mu_pts[0]:.2f}, {mu_pts[-1]:.2f}]$, $V_z \\in [{vz_pts[0]:.2f}, {vz_pts[-1]:.2f}]$)"

    # Colormap normalization
    g_local_max = max(1.0, float(np.nanpercentile([g_ll_cut, g_rr_cut], 99.5)))
    g_nonlocal_max = max(0.04, float(np.nanpercentile(np.abs([a_grl_cut, a_glr_cut]), 99.0)))

    def _plot_single(Z, cmap, vmin, vmax, clabel, title_text, filename_stem, y_bias_label, overlay_gap=None):
        fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)

        pcm = ax.pcolormesh(x_coords, bias_uV, Z.T, shading='auto', vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
        if is_diagonal:
            tick_indices = np.round(np.linspace(0, actual_N - 1, min(8, actual_N))).astype(int)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([f"{vz_pts[idx]:.3f}" for idx in tick_indices], rotation=30, ha='right')
            ax.set_xlabel(r"Zeeman Field $V_z$ (meV)")

            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())
            ax_top.set_xticks(tick_indices)
            ax_top.set_xticklabels([f"{mu_pts[idx]:.3f}" for idx in tick_indices], rotation=30, ha='left')
            ax_top.set_xlabel(r"Chemical Potential $\mu$ (meV)")
        else:
            ax.set_xlabel(xlabel)

        ax.set_ylabel(y_bias_label)
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

        # Overlay gap if provided
        if overlay_gap is not None:
            gap_uV = 1e3 * overlay_gap
            max_b = np.max(np.abs(bias_uV))
            valid = (gap_uV > 0) & (gap_uV <= max_b)
            if np.any(valid):
                gap_plot = np.where(valid, gap_uV, np.nan)
                ax.plot(x_coords, gap_plot, color='black', linewidth=1.5, label=r"$\pm \Delta_{\mathrm{ext}}$")
                ax.plot(x_coords, -gap_plot, color='black', linewidth=1.5)

        # Vertical snapped point markers
        for sp in rcut.get('resolved_snaps', []):
            s_idx = sp['snap_idx']
            x_pt = x_coords[s_idx]
            ax.axvline(x_pt, color=sp['color'], linestyle='--', linewidth=1.2, alpha=0.8, label=sp['label'])

        if (overlay_gap is not None and np.any((1e3 * overlay_gap > 0) & (1e3 * overlay_gap <= np.max(np.abs(bias_uV))))) or len(rcut.get('resolved_snaps', [])) > 0:
            ax.legend(loc='upper right', framealpha=0.8, fontsize=8)

        cb = fig.colorbar(pcm, ax=ax, pad=0.03)
        cb.set_label(clabel)
        ax.set_title(f"{title_text} along {cut['label']}\n{sub_title}", fontsize=11)

        plt.tight_layout()
        fig.savefig(cut_dir / f"{filename_stem}.png", dpi=300)
        fig.savefig(cut_dir / f"{cut['label']}_{filename_stem}.png", dpi=300)
        plt.close(fig)

    # 1. G_LL
    _plot_single(g_ll_cut, 'viridis', 0.0, g_local_max, r"$G_\mathrm{LL}$ [$e^2/h$]", r"$G_\mathrm{LL}$", "G_LL", r"Left bias [$\mu$V]")
    # 2. G_RR
    _plot_single(g_rr_cut, 'viridis', 0.0, g_local_max, r"$G_\mathrm{RR}$ [$e^2/h$]", r"$G_\mathrm{RR}$", "G_RR", r"Right bias [$\mu$V]")
    # 3. A(G_RL)
    _plot_single(a_grl_cut, 'PuOr_r', -g_nonlocal_max, g_nonlocal_max, r"$A(G_\mathrm{RL})$ [$e^2/h$]", r"$A(G_\mathrm{RL})$", "A_GRL", r"Left bias [$\mu$V]", overlay_gap=gap_l_cut if has_tgp_gap else None)
    # 4. A(G_LR)
    _plot_single(a_glr_cut, 'PuOr_r', -g_nonlocal_max, g_nonlocal_max, r"$A(G_\mathrm{LR})$ [$e^2/h$]", r"$A(G_\mathrm{LR})$", "A_GLR", r"Right bias [$\mu$V]", overlay_gap=gap_r_cut if has_tgp_gap else None)

    # 5. Integrated Conductance Multiplot (multi_panel_conductance.png)
    if evals is not None and pfaffians is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig_multi, axes_multi = plt.subplots(5, 1, figsize=(8.0, 13.0), gridspec_kw={'height_ratios': [2.0, 1.0, 1.0, 1.0, 1.0]}, sharex=True)
        dividers = [make_axes_locatable(ax) for ax in axes_multi]
        caxes = [d.append_axes("right", size="2.5%", pad=0.1) for d in dividers]

        # Panel 1: BdG Energy Spectrum
        kvals = evals.shape[1]
        mid_idx = kvals // 2

        for idx in range(actual_N):
            is_pfaff = (pfaffians[idx] > 0) if pfaffians is not None else False
            is_corr = bool(corr_thresh[idx]) if corr_thresh is not None else False

            if not (is_pfaff or is_corr):
                continue

            if is_diagonal:
                x_left = idx - 0.5
                x_right = idx + 0.5
            else:
                dx = (x_coords[-1] - x_coords[0]) / (actual_N - 1) if actual_N > 1 else 1.0
                x_left = x_coords[idx] - 0.5 * dx
                x_right = x_coords[idx] + 0.5 * dx

            if is_pfaff and is_corr:
                # Both topological: mixed colors (slate blue)
                axes_multi[0].axvspan(x_left, x_right, color='#5d768d', alpha=0.55, zorder=0)
            elif is_pfaff and not is_corr:
                # Pfaffian only: dark grey background
                axes_multi[0].axvspan(x_left, x_right, color='#555555', alpha=0.45, zorder=0)
            elif is_corr and not is_pfaff:
                # Correlation only: light blue background
                axes_multi[0].axvspan(x_left, x_right, color='#99ccff', alpha=0.5, zorder=0)

        for sp in rcut.get('resolved_snaps', []):
            s_idx = sp['snap_idx']
            x_pt = x_coords[s_idx]
            for ax in axes_multi:
                ax.axvline(x_pt, color=sp['color'], linestyle='--', linewidth=1.2, alpha=0.8, zorder=3)

        for j in range(kvals):
            color = 'red' if j in [mid_idx-1, mid_idx] else 'royalblue'
            lw = 2.5 if j in [mid_idx-1, mid_idx] else 1.5
            axes_multi[0].plot(x_coords, evals[:, j], color=color, linewidth=lw, alpha=0.85, zorder=2)
        axes_multi[0].axhline(0, color='black', linestyle='--', alpha=0.6, linewidth=1.0)
        axes_multi[0].set_ylabel("Energy (meV)")
        axes_multi[0].set_title(f"Spectra and Conductance along {cut['label']}\n{sub_title}", fontsize=11)
        caxes[0].axis('off')

        # Highlight full TGP Island along bottom runner with red ticks on bottom axis
        if tgp_roi2 is not None:
            tgp_plotted = False
            ymin = np.min(evals) - 0.08 * (np.max(evals) - np.min(evals))
            ymax = np.max(evals) + 0.08 * (np.max(evals) - np.min(evals))
            axes_multi[0].set_ylim(ymin, ymax)
            y_bar = ymin + 0.03 * (ymax - ymin)
            for idx in range(actual_N):
                if bool(tgp_roi2[idx]):
                    if is_diagonal:
                        x_left = idx - 0.5
                        x_right = idx + 0.5
                        x_center = idx
                    else:
                        dx = (x_coords[-1] - x_coords[0]) / (actual_N - 1) if actual_N > 1 else 1.0
                        x_left = x_coords[idx] - 0.5 * dx
                        x_right = x_coords[idx] + 0.5 * dx
                        x_center = x_coords[idx]
                    axes_multi[0].plot([x_left, x_right], [y_bar, y_bar], color='#2ca02c', linewidth=4.5, solid_capstyle='butt', zorder=5, label="TGP Island" if not tgp_plotted else "")
                    axes_multi[0].plot(x_center, ymin, marker='|', color='red', markersize=8, markeredgewidth=1.8, zorder=6)
                    tgp_plotted = True
            if tgp_plotted:
                axes_multi[0].legend(loc='upper right', fontsize=8, framealpha=0.8)

        # Panel 2: G_LL
        pcm2 = axes_multi[1].pcolormesh(x_coords, bias_uV, g_ll_cut.T, shading='auto', vmin=0.0, vmax=g_local_max, cmap='viridis', rasterized=True)
        axes_multi[1].set_ylabel(r"Left bias [$\mu$V]")
        cb2 = fig_multi.colorbar(pcm2, cax=caxes[1])
        cb2.set_label(r"$G_\mathrm{LL}$ [$e^2/h$]", fontsize=9)

        # Panel 3: G_RR
        pcm3 = axes_multi[2].pcolormesh(x_coords, bias_uV, g_rr_cut.T, shading='auto', vmin=0.0, vmax=g_local_max, cmap='viridis', rasterized=True)
        axes_multi[2].set_ylabel(r"Right bias [$\mu$V]")
        cb3 = fig_multi.colorbar(pcm3, cax=caxes[2])
        cb3.set_label(r"$G_\mathrm{RR}$ [$e^2/h$]", fontsize=9)

        # Panel 4: A(G_RL)
        pcm4 = axes_multi[3].pcolormesh(x_coords, bias_uV, a_grl_cut.T, shading='auto', vmin=-g_nonlocal_max, vmax=g_nonlocal_max, cmap='PuOr_r', rasterized=True)
        axes_multi[3].set_ylabel(r"Left bias [$\mu$V]")
        if has_tgp_gap and gap_l_cut is not None:
            gap_uV_l = 1e3 * gap_l_cut
            max_b = np.max(np.abs(bias_uV))
            valid_l = (gap_uV_l > 0) & (gap_uV_l <= max_b)
            if np.any(valid_l):
                gap_plot_l = np.where(valid_l, gap_uV_l, np.nan)
                axes_multi[3].plot(x_coords, gap_plot_l, color='black', linewidth=1.5, label=r"$\pm \Delta_{\mathrm{ext}}$")
                axes_multi[3].plot(x_coords, -gap_plot_l, color='black', linewidth=1.5)
        cb4 = fig_multi.colorbar(pcm4, cax=caxes[3])
        cb4.set_label(r"$A(G_\mathrm{RL})$ [$e^2/h$]", fontsize=9)

        # Panel 5: A(G_LR)
        pcm5 = axes_multi[4].pcolormesh(x_coords, bias_uV, a_glr_cut.T, shading='auto', vmin=-g_nonlocal_max, vmax=g_nonlocal_max, cmap='PuOr_r', rasterized=True)
        axes_multi[4].set_ylabel(r"Right bias [$\mu$V]")
        if has_tgp_gap and gap_r_cut is not None:
            gap_uV_r = 1e3 * gap_r_cut
            max_b = np.max(np.abs(bias_uV))
            valid_r = (gap_uV_r > 0) & (gap_uV_r <= max_b)
            if np.any(valid_r):
                gap_plot_r = np.where(valid_r, gap_uV_r, np.nan)
                axes_multi[4].plot(x_coords, gap_plot_r, color='black', linewidth=1.5, label=r"$\pm \Delta_{\mathrm{ext}}$")
                axes_multi[4].plot(x_coords, -gap_plot_r, color='black', linewidth=1.5)
        cb5 = fig_multi.colorbar(pcm5, cax=caxes[4])
        cb5.set_label(r"$A(G_\mathrm{LR})$ [$e^2/h$]", fontsize=9)

        # Bottom and Top x-axis formatting
        if is_diagonal:
            tick_indices = np.round(np.linspace(0, actual_N - 1, min(8, actual_N))).astype(int)
            axes_multi[4].set_xticks(tick_indices)
            axes_multi[4].set_xticklabels([f"{vz_pts[idx]:.3f}" for idx in tick_indices], rotation=30, ha='right')
            axes_multi[4].set_xlabel(r"Zeeman Field $V_z$ (meV)")

            ax_top = axes_multi[0].twiny()
            ax_top.set_xlim(axes_multi[0].get_xlim())
            ax_top.set_xticks(tick_indices)
            ax_top.set_xticklabels([f"{mu_pts[idx]:.3f}" for idx in tick_indices], rotation=30, ha='left')
            ax_top.set_xlabel(r"Chemical Potential $\mu$ (meV)")
        else:
            axes_multi[4].set_xlabel(xlabel)
            if vz_range > 1e-4 and mu_range < 1e-4:
                ax_top = axes_multi[0].twiny()
                ax_top.set_xlim(axes_multi[0].get_xlim())
                tick_indices = np.round(np.linspace(0, actual_N - 1, min(8, actual_N))).astype(int)
                ax_top.set_xticks(x_coords[tick_indices])
                ax_top.set_xticklabels([f"{mu_pts[0]:.3f}"] * len(tick_indices))
                ax_top.set_xlabel(r"Chemical Potential $\mu$ (meV)")
            elif mu_range > 1e-4 and vz_range < 1e-4:
                ax_top = axes_multi[0].twiny()
                ax_top.set_xlim(axes_multi[0].get_xlim())
                tick_indices = np.round(np.linspace(0, actual_N - 1, min(8, actual_N))).astype(int)
                ax_top.set_xticks(x_coords[tick_indices])
                ax_top.set_xticklabels([f"{vz_pts[0]:.3f}"] * len(tick_indices))
                ax_top.set_xlabel(r"Zeeman Field $V_z$ (meV)")

        fig_multi.savefig(cut_dir / "multi_panel_conductance.png", dpi=300)
        fig_multi.savefig(cut_dir / f"{cut['label']}_multi_panel_conductance.png", dpi=300)
        plt.close(fig_multi)


def plot_stage2_diagram_clean(
    stage2_ds,
    cutter_value=0,
    zbp_cluster_numbers='all',
    invariant='pfaffian',
    gap_lim=0.06,  # in meV
    stars=None,
    draw_cut_lines=False,
    resolved_cuts=None,
    title_suffix="",
    correlation_mask=None
):
    """
    Replicates Microsoft PRL Fig. 29 Stage 2 diagram with clean, de-cluttered formatting:
      - Eliminates all percent boundary text clutter.
      - Expresses transport gap strictly in meV ($q\Delta$ [meV]).
      - Sets axes to $V_z$ [meV] and $\mu$ [meV].
      - Optionally overlays stars at island centers of mass and/or dashed cut trajectories.
      - If correlation_mask is provided, shades regions failing correlation in neutral slate grey.
    """
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True, layout='constrained')
    
    ds = stage2_ds
    ds_sel = ds.sel(cutter_pair_index=cutter_value) if 'cutter_pair_index' in ds.dims else ds
    
    pl_kw = dict(
        x="B",
        y="V",
        add_colorbar=False,
        shading="nearest",
        infer_intervals=False,
        linewidth=0,
        rasterized=True,
    )
    
    # Panel (a): 4-state map
    gap_bool = ds_sel.gap_boolean.squeeze()
    cmap_gap = mcolors.ListedColormap(["white", "#1f77b4"])
    im0 = gap_bool.transpose("V", "B").plot.pcolormesh(ax=axs[0], cmap=cmap_gap, zorder=1, vmin=-0.5, vmax=1.5, **pl_kw)
    
    zbp_bool = ds_sel.zbp.squeeze()
    cmap_zbp = mcolors.ListedColormap([np.array([255, 229, 82]) / 256, "tab:orange"])
    im1 = gap_bool.where(zbp_bool, np.nan).transpose("V", "B").plot.pcolormesh(ax=axs[0], cmap=cmap_zbp, zorder=2, vmin=-0.5, vmax=1.5, **pl_kw)

    # If correlation_mask is provided, shade areas where correlation failed in neutral slate grey
    if correlation_mask is not None:
        if isinstance(correlation_mask, np.ndarray):
            corr_pass_2d = correlation_mask
            if corr_pass_2d.shape == (len(ds["B"]), len(ds["V"])):
                corr_pass_2d = corr_pass_2d.T
            corr_fail_da = xr.DataArray(
                ~corr_pass_2d,
                coords={"V": ds["V"], "B": ds["B"]},
                dims=["V", "B"]
            )
        else:
            corr_fail_da = ~correlation_mask.squeeze()
            if corr_fail_da.dims == ("B", "V"):
                corr_fail_da = corr_fail_da.transpose("V", "B")

        cmap_grey = mcolors.ListedColormap(["#94a3b8"])  # neutral slate grey
        corr_fail_da.where(corr_fail_da, np.nan).plot.pcolormesh(
            ax=axs[0],
            cmap=cmap_grey,
            zorder=3,
            vmin=0.5,
            vmax=1.5,
            **pl_kw
        )
    
    cax1 = axs[0].inset_axes([-0.18, 0.45, 0.03, 0.35], transform=axs[0].transAxes)
    cb1 = fig.colorbar(im0, cax=cax1, orientation="vertical", ticks=[0, 1])
    cb1.ax.set_yticklabels(["Gapless", "Gapped"], fontsize=8)
    
    cax2 = axs[0].inset_axes([-0.18, 0.05, 0.03, 0.35], transform=axs[0].transAxes)
    cb2 = fig.colorbar(im1, cax=cax2, orientation="vertical", ticks=[0, 1])
    cb2.ax.set_yticklabels(["Gapless & ZBP", "Gapped & ZBP"], fontsize=8)
    
    # Grid for smooth contouring (reps=2 for speed and smoothness)
    reps = 2
    B = np.array(ds["B"])
    V = np.array(ds["V"])
    B1 = np.linspace(B.min(), B.max(), B.size * reps)
    V1 = np.linspace(V.min(), V.max(), V.size * reps)
    
    artists = []
    if invariant is not None and invariant in ds:
        cs = (
            ds[invariant]
            .astype(float)
            .interp(B=B1, V=V1, method="nearest")
            .plot.contourf(
                x="B",
                y="V",
                ax=axs[0],
                levels=[-1, 0.5, 2],
                colors=[(1, 1, 1, 0), (1, 1, 1, 0)],
                hatches=[None, r"\\\\"],
                linestyles="-",
                zorder=1001,
                add_colorbar=False,
            )
        )
        artists, _ = cs.legend_elements()

    legend_handles = []
    legend_labels = []
    if invariant is not None and invariant in ds and len(artists) > 1:
        legend_handles.append(artists[1])
        legend_labels.append(r"Topological")
    if correlation_mask is not None:
        import matplotlib.patches as mpatches
        grey_patch = mpatches.Patch(facecolor="#94a3b8", edgecolor="none", label=r"Failed Corr ($C < 0.85$)")
        legend_handles.append(grey_patch)
        legend_labels.append(r"Failed Corr ($C < 0.85$)")
    if legend_handles:
        axs[0].legend(
            legend_handles,
            legend_labels,
            handlelength=1.6,
            handleheight=1.6,
            frameon=True,
            loc="upper left",
            fontsize=8,
        )
            
    # Panel (b): Signed gap in meV
    gp = np.abs(ds_sel.gap)
    clusters = None
    existing_zbp_cluster_numbers = []
    if "gapped_zbp_cluster" in ds_sel:
        try:
            import tgp
            clusters = tgp.common.expand_clusters(ds_sel.gapped_zbp_cluster, dim="zbp_cluster_number")
            existing_zbp_cluster_numbers = np.array(clusters.zbp_cluster_number, dtype=int).tolist()
        except Exception:
            pass
        
    if zbp_cluster_numbers == "all":
        zbp_cluster_numbers = existing_zbp_cluster_numbers
    elif zbp_cluster_numbers is None:
        zbp_cluster_numbers = []
        
    if clusters is not None:
        for i in zbp_cluster_numbers:
            gp = gp * (1.0 - 2.0 * clusters.sel(zbp_cluster_number=i))
            
    pcm = gp.plot.pcolormesh(
        ax=axs[1],
        vmin=-gap_lim,
        vmax=gap_lim,
        cmap="RdBu",
        linewidth=0,
        rasterized=True,
        add_colorbar=False,
        shading="nearest",
        infer_intervals=False,
    )
    cax = axs[1].inset_axes([1.03, 0, 0.03, 1], transform=axs[1].transAxes)
    cb = fig.colorbar(pcm, ax=axs[1], cax=cax, extend="both")
    cb.set_label(r"$q\Delta$ [meV]", fontsize=10)
    
    # Contour outlines for clusters (NO TEXT LABELS)
    if clusters is not None:
        for i in zbp_cluster_numbers:
            cluster = clusters.sel(zbp_cluster_number=i)
            for ax in axs:
                cluster.astype(float).interp(B=B1, V=V1, method="nearest").plot.contour(
                    x="B",
                    y="V",
                    ax=ax,
                    levels=[0.5],
                    colors="k",
                    linewidths=1.3,
                    linestyles="-",
                    zorder=1000,
                )
                
    # Overlay cut lines if requested
    if draw_cut_lines and resolved_cuts is not None:
        for rcut in resolved_cuts:
            cut = rcut['config']
            for ax in axs:
                ax.plot([cut['start'][1], cut['end'][1]], [cut['start'][0], cut['end'][0]],
                        color=cut['color'], linewidth=1.2, linestyle='--', label=cut['label'])
                for sp in rcut.get('resolved_snaps', []):
                    ax.plot(sp['snapped_coords'][1], sp['snapped_coords'][0],
                            marker='*', color=sp['color'], markersize=7, linestyle='None')

    # Overlay Stars if provided
    if stars is not None:
        for star in stars:
            for ax in axs:
                ax.plot(star['vz_com'], star['mu_com'],
                        marker='*', color=star['color'], markersize=14,
                        markeredgecolor='black', markeredgewidth=1.2, zorder=1100)
        star_handles = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=s['color'],
                       markeredgecolor='k', markeredgewidth=1.0, markersize=12, label=s['label'])
            for s in stars
        ]
        axs[1].legend(handles=star_handles, loc='upper left', fontsize=8, framealpha=0.85)

    axs[0].set_xlabel(r"Zeeman Field $V_z$ [meV]", fontsize=10)
    axs[0].set_ylabel(r"Chemical Potential $\mu$ [meV]", fontsize=10)
    axs[0].set_title(f"(a) Phase Map {title_suffix}".strip(), fontsize=11)
    
    axs[1].set_xlabel(r"Zeeman Field $V_z$ [meV]", fontsize=10)
    axs[1].set_title(f"(b) Signed Topological Gap {title_suffix}".strip(), fontsize=11)
    
    return fig, axs


def process_tgp_cuts_and_sweeps(
    data_dir,
    OUT_DIR,
    tprep,
    tprep_left,
    tprep_right,
    zbp_ds,
    stage2_ds,
    B_vals,
    V_vals,
    z_inv,
    resolved_cuts,
    physics_params,
    gap_th_factor,
    p,
    corr_thresh_2d
):
    """
    Executes:
      1. De-cluttered main Stage 2 Paper Diagram with meV units.
      2. Stage 2 Sensitivity Sweeps across gap threshold multipliers and noise floors.
      3. False Discovery Rate (FDR) and topological discovery confusion matrix.
      4. TGP Cuts through the 3 largest orange & topological islands with multiplots.
    """
    import tgp
    logger.info("Executing de-cluttered Stage 2 generation, sensitivity sweeps, and TGP cuts...")
    
    # 1. Main Stage 2 Paper Diagram (Replicating PRL Fig. 29, de-cluttered, meV units)
    try:
        fig_s2, _ = plot_stage2_diagram_clean(
            stage2_ds,
            cutter_value=0,
            zbp_cluster_numbers='all',
            invariant='pfaffian' if 'pfaffian' in stage2_ds else None,
            gap_lim=0.06,
            draw_cut_lines=True,
            resolved_cuts=resolved_cuts
        )
        fig_s2.savefig(OUT_DIR / "stage2_paper_diagram.png", dpi=300, bbox_inches='tight')
        plt.close(fig_s2)
        logger.info(f"Saved de-cluttered stage2_paper_diagram.png to {OUT_DIR}")
    except Exception as e_s2:
        logger.warning(f"Could not generate clean stage2_paper_diagram: {e_s2}")

    # 2. Stage 2 Sensitivity Sweeps across multipliers and noise floor
    sweeps_dir = OUT_DIR / "Stage2_Gap_Sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    multipliers = [0.5, 0.7, 0.75, 0.85, 0.9, 1.0, 1.1, 1.15, 1.25, 1.3, 1.5]
    abs_noise_values = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    upper_th = float('inf') if p.upper_conductance_threshold is None else p.upper_conductance_threshold
    base_noise_th = p.noise_threshold
    pfaff_da = stage2_ds.get('pfaffian', None)

    logger.info("Generating Stage 2 sensitivity diagrams for threshold multipliers...")
    for m in multipliers:
        try:
            fac = gap_th_factor * m
            t_l = tprep.rename({"bias": "left_bias"})
            t_r = tprep.rename({"bias": "right_bias"})
            t_l, t_r = tgp.two.extract_gap(t_l, t_r, gap_threshold_factor=fac, upper_conductance_threshold=upper_th, noise_threshold=base_noise_th)
            z_ds = tgp.two.zbp_dataset_derivative(t_l, t_r, zbp_probability_threshold=0.7, average_over_cutter=False)
            tgp.two.set_gap_threshold(z_ds, threshold_low=10e-3, threshold_high=70e-3)
            s2_m = tgp.two.cluster_and_score(z_ds, min_cluster_size=7, cluster_gap_threshold=10e-3, cluster_percentage_boundary_threshold=0.6, cluster_ncutter_threshold=0.5)
            if pfaff_da is not None:
                s2_m['pfaffian'] = pfaff_da
            fig_m, _ = plot_stage2_diagram_clean(s2_m, draw_cut_lines=False, title_suffix=f"Factor={fac:.4f} ({m:.2f}x)")
            fig_m.savefig(sweeps_dir / f"stage2_multiplier_{m:.2f}.png", dpi=150, bbox_inches='tight')
            plt.close(fig_m)
        except Exception as e_m:
            logger.warning(f"Failed stage2 sweep for multiplier {m}: {e_m}")

    logger.info("Generating Stage 2 sensitivity diagrams for noise floor threshold values...")
    for nth in abs_noise_values:
        try:
            t_l = tprep.rename({"bias": "left_bias"})
            t_r = tprep.rename({"bias": "right_bias"})
            t_l, t_r = tgp.two.extract_gap(t_l, t_r, gap_threshold_factor=0.0, upper_conductance_threshold=upper_th, noise_threshold=nth)
            z_ds = tgp.two.zbp_dataset_derivative(t_l, t_r, zbp_probability_threshold=0.7, average_over_cutter=False)
            tgp.two.set_gap_threshold(z_ds, threshold_low=10e-3, threshold_high=70e-3)
            s2_n = tgp.two.cluster_and_score(z_ds, min_cluster_size=7, cluster_gap_threshold=10e-3, cluster_percentage_boundary_threshold=0.6, cluster_ncutter_threshold=0.5)
            if pfaff_da is not None:
                s2_n['pfaffian'] = pfaff_da
            fig_n, _ = plot_stage2_diagram_clean(s2_n, draw_cut_lines=False, title_suffix=f"Noise Floor={nth:.1e} e^2/h")
            fig_n.savefig(sweeps_dir / f"stage2_abs_noise_{nth:.1e}.png", dpi=150, bbox_inches='tight')
            plt.close(fig_n)
        except Exception as e_n:
            logger.warning(f"Failed stage2 sweep for noise floor {nth}: {e_n}")

    # 3. False Discovery Rate (FDR) & Metrics Extraction
    tgp_cuts_dir = OUT_DIR / "TGP_Cuts"
    tgp_cuts_dir.mkdir(parents=True, exist_ok=True)
    
    pfaff_vb = (tprep["L_SI"].values[0] == 1).T  # (V, B)
    if "roi2" in stage2_ds:
        roi2_vb = stage2_ds.roi2.values.T if stage2_ds.roi2.dims == ("B", "V") else stage2_ds.roi2.values
    else:
        roi2_vb = np.zeros_like(pfaff_vb, dtype=bool)
    roi2_passed = (roi2_vb > 0)

    tp = int(np.sum(roi2_passed & pfaff_vb))
    fp = int(np.sum(roi2_passed & (~pfaff_vb)))
    fn = int(np.sum((~roi2_passed) & pfaff_vb))
    tn = int(np.sum((~roi2_passed) & (~pfaff_vb)))
    total_passed = tp + fp
    total_topo = tp + fn
    ppv = float(tp / total_passed) if total_passed > 0 else 0.0
    fdr = float(fp / total_passed) if total_passed > 0 else 0.0
    tpr = float(tp / total_topo) if total_topo > 0 else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    gzbp_vb = zbp_ds.gapped_zbp.values[0]  # cutter 0
    tp_gz = int(np.sum(gzbp_vb & pfaff_vb))
    fp_gz = int(np.sum(gzbp_vb & (~pfaff_vb)))
    total_gz = tp_gz + fp_gz
    ppv_gz = float(tp_gz / total_gz) if total_gz > 0 else 0.0
    fdr_gz = float(fp_gz / total_gz) if total_gz > 0 else 0.0

    # 4. Top 3 Orange & Topological Islands & Cuts
    orange_topo = gzbp_vb & pfaff_vb
    lbl_gzbp, n_gzbp = ndimage.label(orange_topo)
    sizes = [int((lbl_gzbp == i).sum()) for i in range(1, n_gzbp + 1)]
    s_idx = np.argsort(sizes)[::-1]
    star_colors = ["cyan", "magenta", "gold"]
    star_names = ["Cyan Star", "Magenta Star", "Gold Star"]

    top3_islands = []
    for rank, idx in enumerate(s_idx[:3], 1):
        comp_id = idx + 1
        mask = (lbl_gzbp == comp_id)
        com_v, com_b = ndimage.center_of_mass(mask)
        mu_com = float(np.interp(com_v, np.arange(len(V_vals)), V_vals))
        vz_com = float(np.interp(com_b, np.arange(len(B_vals)), B_vals))
        top3_islands.append({
            "rank": rank,
            "size_pixels": sizes[idx],
            "vz_com": vz_com,
            "mu_com": mu_com,
            "color": star_colors[rank - 1],
            "star_name": star_names[rank - 1]
        })

    fdr_report = {
        "dataset_name": data_dir.name,
        "V0": physics_params.get("Vdisx", np.array([0.0]))[0] if isinstance(physics_params.get("Vdisx"), np.ndarray) else 0.0,
        "total_pixels": int(pfaff_vb.size),
        "ground_truth_topological_pixels": total_topo,
        "stage2_roi2_metrics": {
            "total_passed": total_passed,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "p_topological_given_passed_ppv": ppv,
            "false_discovery_rate_fdr": fdr,
            "sensitivity_tpr": tpr,
            "specificity_tnr": tnr,
            "false_positive_rate_fpr": fpr,
        },
        "candidate_orange_state_metrics": {
            "total_orange_pixels": total_gz,
            "true_positives": tp_gz,
            "false_positives": fp_gz,
            "p_topological_given_orange_ppv": ppv_gz,
            "false_discovery_rate_fdr": fdr_gz,
        },
        "top3_islands": top3_islands
    }
    with open(tgp_cuts_dir / "fdr_metrics.json", "w") as f_json:
        json.dump(fdr_report, f_json, indent=2)
    logger.info(f"Saved fdr_metrics.json to {tgp_cuts_dir}")

    # Export stage2_phase_map.png with stars and NO cut lines
    try:
        stars_meta = [
            {"vz_com": isl["vz_com"], "mu_com": isl["mu_com"], "color": isl["color"], "label": f"Island {isl['rank']} ({isl['star_name']})"}
            for isl in top3_islands
        ]
        fig_stars, _ = plot_stage2_diagram_clean(
            stage2_ds,
            stars=stars_meta,
            draw_cut_lines=False
        )
        fig_stars.savefig(tgp_cuts_dir / "stage2_phase_map.png", dpi=200, bbox_inches='tight')
        plt.close(fig_stars)
        logger.info(f"Saved stage2_phase_map.png to {tgp_cuts_dir}")
    except Exception as e_stars:
        logger.warning(f"Could not generate stage2_phase_map with stars: {e_stars}")

    # Process each of the 3 cuts
    t_val = physics_params['t_val']
    gamma = physics_params['gamma']
    Delta0 = physics_params['Delta0']
    alpha = physics_params['alpha']
    Ls = physics_params['Ls']
    Vdisx = physics_params['Vdisx']
    L_2w_avg = tprep['L_2w_nl'].mean(dim='cutter_pair_index') if 'cutter_pair_index' in tprep['L_2w_nl'].dims else tprep['L_2w_nl']
    L_3w_avg = tprep['L_3w'].mean(dim='cutter_pair_index') if 'cutter_pair_index' in tprep['L_3w'].dims else tprep['L_3w']
    gap_left_avg = tprep_left.gap.mean(dim='cutter_pair_index')
    gap_right_avg = tprep_right.gap.mean(dim='cutter_pair_index')
    invariant_avg = tprep['L_SI'].mean(dim='cutter_pair_index')

    for isl in top3_islands:
        rank = isl["rank"]
        vz_com = isl["vz_com"]
        mu_com = isl["mu_com"]
        star_color = isl["color"]
        star_name = isl["star_name"]
        
        isl_dir = tgp_cuts_dir / f"Island_{rank}_{star_color.capitalize()}_Star"
        isl_dir.mkdir(parents=True, exist_ok=True)
        
        half_w = 0.35
        vz_start = max(0.0, vz_com - half_w)
        vz_end = min(1.4, vz_com + half_w)
        if (vz_end - vz_start) < 0.6:
            if vz_start == 0.0: vz_end = min(1.4, vz_start + 0.7)
            elif vz_end == 1.4: vz_start = max(0.0, vz_end - 0.7)
            
        actual_N = 100
        vz_pts = np.linspace(vz_start, vz_end, actual_N)
        mu_pts = np.full(actual_N, mu_com)
        pts = np.column_stack([mu_pts, vz_pts])
        snap_idx = int(np.argmin(np.abs(vz_pts - vz_com)))
        
        rcut = {
            'config': {
                'start': (mu_com, vz_start),
                'end': (mu_com, vz_end),
                'color': star_color,
                'label': f"TGP_Cut_Island_{rank}_{star_color.capitalize()}"
            },
            'resolved_snaps': [
                {
                    'snap_idx': snap_idx,
                    'snapped_coords': (mu_com, vz_pts[snap_idx]),
                    'color': star_color,
                    'label': f"Island_{rank}_COM"
                }
            ]
        }
        
        kvals = 14
        worker_args = [
            (i, mu_pts[i], vz_pts[i], t_val, gamma, Delta0, alpha, Ls, Vdisx, kvals)
            for i in range(actual_N)
        ]
        with ProcessPoolExecutor(max_workers=12) as executor:
            results = list(executor.map(_eval_cut_point_worker, worker_args))
        results.sort(key=lambda x: x[0])
        
        evals = np.zeros((actual_N, kvals))
        val_overlap = np.zeros(actual_N)
        for i, s_evals, ov in results:
            evals[i, :] = s_evals
            val_overlap[i] = ov
            
        val_2w = np.zeros(actual_N)
        val_3w = np.zeros(actual_N)
        val_gap_left = np.zeros(actual_N)
        val_gap_right = np.zeros(actual_N)
        pfaffians = np.zeros(actual_N)
        val_corr_thresh = np.zeros(actual_N, dtype=bool)
        val_tgp_roi2 = np.zeros(actual_N, dtype=bool)
        
        for i in range(actual_N):
            mv, bv = mu_pts[i], vz_pts[i]
            val_2w[i] = L_2w_avg.sel(V=mv, B=bv, method='nearest').values.item()
            val_3w[i] = L_3w_avg.sel(V=mv, B=bv, method='nearest').values.item()
            pfaffians[i] = invariant_avg.sel(V=mv, B=bv, method='nearest').values.item()
            val_gap_left[i] = gap_left_avg.sel(V=mv, B=bv, method='nearest').values.item()
            val_gap_right[i] = gap_right_avg.sel(V=mv, B=bv, method='nearest').values.item()
            if corr_thresh_2d is not None:
                v_i = np.argmin(np.abs(V_vals - mv))
                b_i = np.argmin(np.abs(B_vals - bv))
                val_corr_thresh[i] = bool(corr_thresh_2d[v_i, b_i] >= 1.0)
            if stage2_ds is not None and 'roi2' in stage2_ds:
                val_tgp_roi2[i] = bool(stage2_ds.roi2.sel(B=bv, V=mv, method='nearest').values > 0)
                
        # Export multi_panel_conductance.png
        export_cut_conductance_plots(
            cut_dir=isl_dir,
            cut=rcut['config'],
            pts=pts,
            rcut=rcut,
            tprep=tprep,
            tprep_left=tprep_left,
            tprep_right=tprep_right,
            has_tgp_gap=True,
            selected_cutter=0,
            evals=evals,
            pfaffians=pfaffians,
            corr_thresh=val_corr_thresh,
            tgp_roi2=val_tgp_roi2
        )
        
        # Cut analytics multi_panel.png (4-panel standardized layout)
        fig_an, axes = plt.subplots(4, 1, figsize=(7.5, 9.2), gridspec_kw={'height_ratios': [2, 1, 1, 0.7]}, sharex=True)
        dx_val = vz_pts[1] - vz_pts[0] if len(vz_pts) > 1 else 1.0
        for idx in range(actual_N):
            is_pfaff = pfaffians[idx] > 0
            is_corr = val_corr_thresh[idx]
            if is_pfaff and is_corr: color = '#5d768d'; alpha = 0.55
            elif is_pfaff and not is_corr: color = '#555555'; alpha = 0.45
            elif is_corr and not is_pfaff: color = '#99ccff'; alpha = 0.5
            else: continue
            for ax in axes:
                ax.axvspan(vz_pts[idx] - 0.5*dx_val, vz_pts[idx] + 0.5*dx_val, color=color, alpha=alpha, zorder=0)

        for ax in axes:
            ax.axvline(vz_pts[snap_idx], color=star_color, linestyle='--', linewidth=1.5, alpha=0.9, zorder=3)
            
        mid_idx = kvals // 2
        for j in range(kvals):
            color = 'red' if j in [mid_idx-1, mid_idx] else 'royalblue'
            lw = 2.5 if j in [mid_idx-1, mid_idx] else 1.2
            axes[0].plot(vz_pts, evals[:, j], color=color, linewidth=lw, alpha=0.85, zorder=2)
        axes[0].plot(vz_pts[snap_idx], 0.0, marker='*', color=star_color, markersize=14, markeredgecolor='black', markeredgewidth=1.2, zorder=10, label=f"COM ({star_name})")
        axes[0].axhline(0, color='black', linestyle='--', alpha=0.6, linewidth=1.0)
        axes[0].set_ylabel("Energy (meV)")
        axes[0].set_title(f"TGP Cut Analytics through Island {rank} [{star_name}]\n($\mu = {mu_com:.3f}$ meV, $V_z \in [{vz_start:.3f}, {vz_end:.3f}]$ meV)", fontsize=11)

        # Highlight full topological island along bottom runner with red ticks on bottom axis
        ymin_an = np.min(evals) - 0.08 * (np.max(evals) - np.min(evals))
        ymax_an = np.max(evals) + 0.08 * (np.max(evals) - np.min(evals))
        axes[0].set_ylim(ymin_an, ymax_an)
        y_bar_an = ymin_an + 0.03 * (ymax_an - ymin_an)
        isl_an_plotted = False
        for idx in range(actual_N):
            if val_tgp_roi2[idx]:
                axes[0].plot([vz_pts[idx] - 0.5*dx_val, vz_pts[idx] + 0.5*dx_val], [y_bar_an, y_bar_an], color='#2ca02c', linewidth=4.5, solid_capstyle='butt', zorder=5, label="TGP Island" if not isl_an_plotted else "")
                axes[0].plot(vz_pts[idx], ymin_an, marker='|', color='red', markersize=8, markeredgewidth=1.8, zorder=6)
                isl_an_plotted = True
        axes[0].legend(loc='upper right', fontsize=8)
        
        # Panel 2: 3w Measurement (Capped)
        cap_val = 1.4 * p.derivative_threshold
        axes[1].plot(vz_pts, np.clip(val_3w, -cap_val, cap_val), color='darkorange', linewidth=1.5, zorder=2)
        axes[1].axhline(-p.derivative_threshold, color='black', linestyle=':', label='ZBP Threshold')
        axes[1].set_ylabel("3ω Curvature")
        axes[1].legend(loc='upper right', fontsize=8)
        
        # Panel 3: Transport Gap and Lowest States
        axes[2].plot(vz_pts, np.minimum(val_gap_left, val_gap_right), color='purple', linewidth=1.5, label=r"Min $\Delta_{ex}$")
        axes[2].plot(vz_pts, evals[:, mid_idx], color='red', linestyle='--', label=r"$E_0$")
        axes[2].plot(vz_pts, evals[:, mid_idx + 1], color='blue', linestyle='--', label=r"$E_1$")
        axes[2].axhline(0.010, color='black', linestyle='--', linewidth=1.2, label=r'Gap Th ($10\ \mu\mathrm{eV}$)', zorder=3)
        axes[2].set_ylabel("Gap / Energy (meV)")
        axes[2].legend(loc='upper right', fontsize=8)
        
        # Panel 4: Thresholded Correlation Condition (±1.5 binarized track)
        corr_bin = np.where(val_corr_thresh, 1.0, -1.0)
        axes[3].step(vz_pts, corr_bin, where='mid', color='#0284c7', linewidth=1.8, zorder=3)
        for idx in range(actual_N):
            c_col = '#10b981' if val_corr_thresh[idx] else '#94a3b8'
            c_alpha = 0.55 if val_corr_thresh[idx] else 0.35
            axes[3].axvspan(vz_pts[idx] - 0.5*dx_val, vz_pts[idx] + 0.5*dx_val,
                            ymin=0.5 if val_corr_thresh[idx] else 0.0,
                            ymax=1.0 if val_corr_thresh[idx] else 0.5,
                            color=c_col, alpha=c_alpha, zorder=2)
        axes[3].axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=4)
        axes[3].axhline(1.0, color='#10b981', linestyle='--', linewidth=0.8, alpha=0.7, zorder=4)
        axes[3].axhline(-1.0, color='#94a3b8', linestyle='--', linewidth=0.8, alpha=0.7, zorder=4)
        axes[3].set_ylim(-1.5, 1.5)
        axes[3].set_yticks([-1.0, 0.0, 1.0])
        axes[3].set_yticklabels(["Fail (-1)", "0", "Pass (+1)"], fontsize=8)
        axes[3].set_ylabel("Barrier Corr")
        axes[3].set_xlabel(r"Zeeman Field $V_z$ (meV)")
        axes[3].set_title(r"Thresholded Barrier Correlation ($C_L \geq 0.85$ & $C_R \geq 0.85$)", fontsize=9)
        
        fig_an.tight_layout()
        fig_an.savefig(isl_dir / "multi_panel.png", dpi=200)
        plt.close(fig_an)
        plt.close(fig_an)
        
        # Standalone spectra plot
        fig_sp, ax_sp = plt.subplots(figsize=(7.5, 4.2))
        for j in range(kvals):
            color = 'red' if j in [mid_idx-1, mid_idx] else 'royalblue'
            lw = 2.5 if j in [mid_idx-1, mid_idx] else 1.2
            ax_sp.plot(vz_pts, evals[:, j], color=color, linewidth=lw, alpha=0.85)
        ax_sp.plot(vz_pts[snap_idx], 0.0, marker='*', color=star_color, markersize=14, markeredgecolor='black', markeredgewidth=1.2, zorder=10, label=f"Center of Mass ({star_name})")
        ax_sp.axvline(vz_pts[snap_idx], color=star_color, linestyle='--', linewidth=1.5, alpha=0.8)
        ax_sp.axhline(0, color='black', linestyle='--', alpha=0.6, linewidth=1.0)
        ax_sp.set_xlabel(r"Zeeman Field $V_z$ [meV]", fontsize=10)
        ax_sp.set_ylabel("Energy [meV]", fontsize=10)
        ax_sp.set_title(f"BdG Energy Spectrum through Island {rank} [{star_name}]\n($\mu = {mu_com:.3f}$ meV, $V_z \in [{vz_start:.3f}, {vz_end:.3f}]$ meV)", fontsize=11)
        
        ymin_sp = np.min(evals) - 0.08 * (np.max(evals) - np.min(evals))
        ymax_sp = np.max(evals) + 0.08 * (np.max(evals) - np.min(evals))
        ax_sp.set_ylim(ymin_sp, ymax_sp)
        y_bar_sp = ymin_sp + 0.03 * (ymax_sp - ymin_sp)
        isl_sp_plotted = False
        for idx in range(actual_N):
            if val_tgp_roi2[idx]:
                ax_sp.plot([vz_pts[idx] - 0.5*dx_val, vz_pts[idx] + 0.5*dx_val], [y_bar_sp, y_bar_sp], color='#2ca02c', linewidth=4.5, solid_capstyle='butt', zorder=5, label="TGP Island" if not isl_sp_plotted else "")
                ax_sp.plot(vz_pts[idx], ymin_sp, marker='|', color='red', markersize=8, markeredgewidth=1.8, zorder=6)
                isl_sp_plotted = True
        ax_sp.legend(loc='upper right', fontsize=9)
        fig_sp.tight_layout()
        fig_sp.savefig(isl_dir / "spectra.png", dpi=200)
        fig_sp.savefig(tgp_cuts_dir / f"tgp_cut_island_{rank}_spectra.png", dpi=200)
        plt.close(fig_sp)
        
        # Point deep dive at COM
        pt_dict = {
            'coords': (mu_com, vz_com),
            'color': star_color,
            'label': f"Island_{rank}_COM",
            'dir_path': str(isl_dir)
        }
        pt_2w = L_2w_avg.sel(V=mu_com, B=vz_com, method='nearest').values.item()
        pt_3w = L_3w_avg.sel(V=mu_com, B=vz_com, method='nearest').values.item()
        _process_deep_dive_point_worker((pt_dict, physics_params, p.T_mK_stage1, pt_2w, pt_3w, 70))
        logger.info(f"Completed TGP Cut & Deep Dive for Island {rank} ({star_name})")

    logger.info("TGP Cuts, Sensitivity Sweeps, and FDR Analysis complete.")
    return fdr_report


def process_data_dir(data_dir):
    OUT_DIR = PathConfigs.PLOTS / f"{data_dir.name}_Plots"
    logger.info(f"Creating output directory: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TPREP_PATH = data_dir / "tprep.nc"

    # Convert all user inputs from (Vz, mu) to internal (mu, Vz) convention
    # Make deep copies so we don't mutate the global config for subsequent runs
    import copy
    import shutil
    dir_name = data_dir.name
    if dir_name in DATASET_CUTS:
        logger.info(f"Using dataset-specific cuts for {dir_name} ({len(DATASET_CUTS[dir_name])} cuts)")
        local_cuts = copy.deepcopy(DATASET_CUTS[dir_name])
        local_freestanding = []
        old_cuts_dir = OUT_DIR / "Cuts"
        old_fp_dir = OUT_DIR / "Freestanding_Points"
        if old_cuts_dir.exists():
            shutil.rmtree(old_cuts_dir)
        if old_fp_dir.exists():
            shutil.rmtree(old_fp_dir)
    else:
        local_cuts = copy.deepcopy(CUTS)
        local_freestanding = copy.deepcopy(FREESTANDING_POINTS)

    for cut in local_cuts:
        cut['start'] = (cut['start'][1], cut['start'][0])
        cut['end'] = (cut['end'][1], cut['end'][0])
        for sp in cut.get('snap_points', []):
            sp['raw_coords'] = (sp['raw_coords'][1], sp['raw_coords'][0])
            
    for fp in local_freestanding:
        fp['coords'] = (fp['coords'][1], fp['coords'][0])

    # 1. Load Data
    params = np.load(data_dir / "all_params.npz", allow_pickle=True)
    pdi_data = np.load(data_dir / "pdi_data.npy", allow_pickle=True)

    if not TPREP_PATH.exists():
        logger.info(f"tprep.nc not found in {data_dir}. Generating it now...")
        from src.tgp_adapter import TGPAdapter
        from src.gpu_broadening import prepare_sim_gpu
        data = TGPAdapter(data_dir).to_xarray()
        tprep = prepare_sim_gpu(data, p.T_mK_stage1)
        
        tmp_path = TPREP_PATH.with_suffix('.nc.tmp')
        tprep.to_netcdf(tmp_path)
        tmp_path.rename(TPREP_PATH)
        logger.info("tprep.nc successfully generated.")
        
    tprep = xr.open_dataset(TPREP_PATH)
    
    # Extract physics params
    t_val = float(params['t'])
    mu_n = float(params['mu_n'])
    mu_leads = float(params['mu_leads'])
    Delta0 = float(params['Delta0'])
    gamma = float(params['gamma'])
    alpha = float(params['alpha'])
    Ln = int(params['Ln'])
    Lb = int(params['Lb'])
    Ls = int(params['Ls'])
    V0 = float(params['V0'])
    barrier_l_base = float(params['barrier0'])
    Vdisx = params['Vdisx'] * V0

    physics_params = {
        't_val': t_val,
        'mu_n': mu_n,
        'mu_leads': mu_leads,
        'gamma': gamma,
        'Delta0': Delta0,
        'alpha': alpha,
        'Ln': Ln,
        'Lb': Lb,
        'Ls': Ls,
        'barrier_l_base': barrier_l_base,
        'Vdisx': Vdisx,
    }

    # Dynamically extract transport gap using tgp with Microsoft threshold factor mapping
    gap_sensitivity_2d = None
    abs_noise_gap_maps = {}
    stage1_ds = None
    stage2_ds = None
    try:
        import tgp
        tprep_left = tprep.rename({"bias": "left_bias"})
        tprep_right = tprep.rename({"bias": "right_bias"})
        upper_th = float('inf') if p.upper_conductance_threshold is None else p.upper_conductance_threshold

        # Map V0 directly to official Microsoft gap_threshold_factor
        threshold_map = {
            0.0: 0.001,
            0.1: 0.01,
            0.378: 0.01,
            0.645: 0.05,
            0.872: 0.05,
            0.91: 0.05,
            1.2: 0.05,
        }
        matched_v0 = min(threshold_map.keys(), key=lambda k: abs(k - V0))
        gap_th_factor = threshold_map[matched_v0] if (abs(matched_v0 - V0) < 0.05 or V0 >= 0.5) else 0.05
        logger.info(f"Using Microsoft gap_threshold_factor = {gap_th_factor} for V0 = {V0} (matched {matched_v0})")

        # 1. Baseline gap extraction
        tprep_left, tprep_right = tgp.two.extract_gap(
            tprep_left, 
            tprep_right,
            gap_threshold_factor=gap_th_factor,
            upper_conductance_threshold=upper_th,
            noise_threshold=p.noise_threshold
        )
        gap_left_avg = tprep_left.gap.mean(dim='cutter_pair_index')
        gap_right_avg = tprep_right.gap.mean(dim='cutter_pair_index')
        has_tgp_gap = True

        # 2. Parallel Sensitivity Sweeps (Sweep 1: relative multipliers; Sweep 2: absolute noise floor)
        logger.info("Executing Gap Sensitivity Sweeps in parallel (12 workers)...")
        multipliers = [0.5, 0.7, 0.75, 0.85, 0.9, 1.0, 1.1, 1.15, 1.25, 1.3, 1.5]
        abs_noise_values = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        sweep_tasks = []
        for m in multipliers:
            sweep_tasks.append((f"rel_{m}", str(TPREP_PATH), gap_th_factor * m, upper_th, p.noise_threshold))
        for nth in abs_noise_values:
            sweep_tasks.append((f"abs_{nth}", str(TPREP_PATH), 0.0, upper_th, nth))

        with ProcessPoolExecutor(max_workers=12) as executor:
            sweep_results = dict(executor.map(_extract_single_gap_worker, sweep_tasks))

        rel_sweep_gaps = [sweep_results[f"rel_{m}"] for m in multipliers]
        rel_stack = np.array(rel_sweep_gaps)
        gap_sensitivity_2d = np.nanmax(rel_stack, axis=0) - np.nanmin(rel_stack, axis=0)

        for nth in abs_noise_values:
            abs_noise_gap_maps[nth] = sweep_results[f"abs_{nth}"]

        # 3. Full Topological Gap Protocol: Stage 1 (ROI 1) -> Stage 2 (ROI 2)
        try:
            logger.info("Executing Stage 1 Protocol (ROI 1 Candidate ZBPs)...")
            th_stage1 = {
                'set_2w_th': {'n_tiles': 25, 'percentile': 50},
                'set_gapped': {'th_2w_p': 0.5, 'method': 'structure'},
                'set_3w_th': {'th_3w': 0.001},
                'set_3w_tat': {'th_3w_tat': 0.5},
                'set_clusters': {'min_samples': 3, 'xi': 0.1, 'min_cluster_size': 0.01, 'max_eps': 10.0}
            }
            stage1_ds = tgp.one.analyze(tprep.copy(), thresholds=th_stage1, force=True)

            logger.info("Executing Stage 2 Protocol (ROI 2 Topological Islands)...")
            zbp_ds = tgp.two.zbp_dataset_derivative(
                tprep_left,
                tprep_right,
                zbp_probability_threshold=0.7,
                average_over_cutter=False
            )
            tgp.two.set_gap_threshold(zbp_ds, threshold_low=10e-3, threshold_high=70e-3)
            stage2_ds = tgp.two.cluster_and_score(
                zbp_ds,
                min_cluster_size=7,
                cluster_gap_threshold=10e-3,
                cluster_percentage_boundary_threshold=0.6,
                cluster_ncutter_threshold=0.5
            )
            if 'L_SI' in tprep:
                inv_da = tprep['L_SI'].mean(dim='cutter_pair_index') if 'cutter_pair_index' in tprep['L_SI'].dims else tprep['L_SI']
                stage2_ds['pfaffian'] = inv_da
        except Exception as e_tgp_roi:
            logger.warning(f"Stage 1/Stage 2 pipeline warning: {e_tgp_roi}")

    except ImportError:
        logger.info("Warning: 'tgp' module not found. Transport gap will be plotted as zeros.")
        has_tgp_gap = False

    # 2. Resolve Cut Paths and Snapped Points
    logger.info("Resolving cuts and snapping points...")
    resolved_cuts = []
    points_to_analyze = []
    
    for cut in local_cuts:
        pts, sampled_indices = generate_point_path(pdi_data, N_CUT_POINTS, 0.02, cut['start'][0], cut['end'][0], cut['start'][1], cut['end'][1])
        actual_N = len(pts)
        
        resolved_snaps = []
        if actual_N > 0:
            for sp in cut.get('snap_points', []):
                raw_coords = np.array(sp['raw_coords'])
                snap_idx = np.argmin(np.linalg.norm(pts - raw_coords, axis=1))
                snapped_coords = pts[snap_idx]
                
                resolved_snaps.append({
                    "raw_coords": raw_coords,
                    "snapped_coords": snapped_coords,
                    "snap_idx": snap_idx,
                    "color": sp['color'],
                    "label": sp['label']
                })
                points_to_analyze.append({
                    "coords": tuple(snapped_coords),
                    "color": sp['color'],
                    "label": sp['label'],
                    "dir_path": OUT_DIR / "Cuts" / cut['label'] / "Points" / sp['label']
                })
                
        resolved_cuts.append({
            "config": cut,
            "pts": pts,
            "sampled_indices": sampled_indices,
            "resolved_snaps": resolved_snaps,
            "actual_N": actual_N
        })

    for fp in local_freestanding:
        fp_dict = fp.copy()
        fp_dict['dir_path'] = OUT_DIR / "Freestanding_Points" / fp['label']
        points_to_analyze.append(fp_dict)

    # 3. Global Phase Maps (Plotly HTML & Matplotlib PNG)
    logger.info("Generating Global Phase Maps (HTML & PNG)...")
    
    B_vals = tprep['B'].values
    V_vals = tprep['V'].values
    z_inv = tprep['L_SI'].mean(dim='cutter_pair_index').transpose('V', 'B').values
    
    # A. 2w Phase Map
    export_phase_map_pair(
        "global_2w",
        tprep['L_2w_nl'].mean(dim='cutter_pair_index').transpose('V', 'B').values,
        tprep['R_2w_nl'].mean(dim='cutter_pair_index').transpose('V', 'B').values,
        z_inv, B_vals, V_vals, -2.0, 2.0, 'RdBu_r', 'RdBu_r',
        "Average Left 2ω", "Average Right 2ω", "2ω Conductance",
        resolved_cuts, local_freestanding, OUT_DIR
    )

    # B. 3w Phase Map
    zrng = DERIVATIVE_THRESHOLD * 1.3
    export_phase_map_pair(
        "global_3w",
        tprep['L_3w'].mean(dim='cutter_pair_index').transpose('V', 'B').values,
        tprep['R_3w'].mean(dim='cutter_pair_index').transpose('V', 'B').values,
        z_inv, B_vals, V_vals, -zrng, zrng, 'RdBu_r', 'RdBu_r',
        "Average Left 3ω", "Average Right 3ω", "3ω Curvature",
        resolved_cuts, local_freestanding, OUT_DIR
    )

    # C. Transport Gap Phase Map
    if has_tgp_gap:
        export_phase_map_pair(
            "global_transport_gap",
            gap_left_avg.transpose('V', 'B').values,
            gap_right_avg.transpose('V', 'B').values,
            z_inv, B_vals, V_vals, 0.0, 0.05, 'gist_heat_r', 'hot_r',
            "Gap from G_RL (Left)", "Gap from G_LR (Right)", "Extracted Gap (meV)",
            resolved_cuts, local_freestanding, OUT_DIR,
            contour_color='dimgray', contour_dash='dash'
        )

        # C1. Relative Threshold Multipliers Sensitivity Phase Map
        if gap_sensitivity_2d is not None:
            sens_max = float(np.nanmax(gap_sensitivity_2d)) if np.nanmax(gap_sensitivity_2d) > 0 else 0.05
            export_phase_map_single(
                "global_gap_threshold_sensitivity",
                gap_sensitivity_2d,
                z_inv, B_vals, V_vals, 0.0, max(0.01, sens_max), 'plasma', 'plasma',
                r"Gap Extraction Sensitivity ($\Delta_{\max} - \Delta_{\min}$ over $\pm 50\%$ Multipliers)", "Gap Sensitivity (meV)",
                resolved_cuts, local_freestanding, OUT_DIR,
                contour_color='cyan'
            )

        # C2. Absolute Noise Floor Gap Sweep Phase Maps
        if abs_noise_gap_maps:
            gap_sweep_dir = OUT_DIR / "Gap_Sweeps"
            gap_sweep_dir.mkdir(parents=True, exist_ok=True)
            for nth, gmap in abs_noise_gap_maps.items():
                nth_str = f"{nth:.0e}".replace("+", "")
                export_phase_map_single(
                    f"global_gap_absolute_noise_{nth_str}",
                    gmap,
                    z_inv, B_vals, V_vals, 0.0, 0.05, 'gist_heat_r', 'hot_r',
                    f"Transport Gap (Absolute Noise Floor = {nth:.1e} $e^2/h$)", "Extracted Gap (meV)",
                    resolved_cuts, local_freestanding, gap_sweep_dir,
                    contour_color='cyan'
                )

    # D. Pfaffian Invariant Phase Map
    export_phase_map_single(
        "global_pfaffian",
        z_inv,
        z_inv, B_vals, V_vals, -1.1, 1.1, 'gray', 'gray',
        "Pfaffian Invariant", "Pfaffian Sign",
        resolved_cuts, local_freestanding, OUT_DIR,
        contour_color='cyan'
    )

    # E. Boundary Confinement Phase Map
    b_conf_path = data_dir / "mzm_boundary_confinement_arr.npy"
    if not b_conf_path.exists():
        b_conf_path = data_dir / "site_localizations.npy"
    if b_conf_path.exists():
        b_conf_raw = np.load(b_conf_path)
        if np.nanmax(b_conf_raw) > 1.0:
            b_conf_raw = 1.0 - (b_conf_raw / Ls)
        b_conf_2d = b_conf_raw.reshape((len(V_vals), len(B_vals)))
        export_phase_map_single(
            "global_boundary_confinement",
            b_conf_2d,
            z_inv, B_vals, V_vals, 0.0, 1.0, 'viridis', 'viridis',
            "Boundary Confinement Ratio", "Boundary Confinement (1 - n/Ls)",
            resolved_cuts, local_freestanding, OUT_DIR,
            contour_color='cyan'
        )

    # F. Density Support Span Phase Map
    s_span_path = data_dir / "mzm_density_support_span_arr.npy"
    if not s_span_path.exists():
        s_span_path = data_dir / "weight_localization_arr.npy"
    if s_span_path.exists():
        s_span_raw = np.load(s_span_path)
        s_span_2d = s_span_raw.reshape((len(V_vals), len(B_vals)))
        export_phase_map_single(
            "global_density_support_span",
            s_span_2d,
            z_inv, B_vals, V_vals, 0.0, 1.0, 'magma_r', 'magma_r',
            "MZM Density Support Span", "Support Span (Wire Fraction)",
            resolved_cuts, local_freestanding, OUT_DIR,
            contour_color='cyan'
        )

    # G. Normalized MZM Overlap Phase Map
    overlap_path = data_dir / "mzm_normalized_overlap_arr.npy"
    if overlap_path.exists():
        overlap_raw = np.load(overlap_path)
        overlap_2d = overlap_raw.reshape((len(V_vals), len(B_vals)))
        max_overlap_val = float(np.nanmax(overlap_2d)) if np.nanmax(overlap_2d) > 0 else 0.05
        export_phase_map_single(
            "global_normalized_overlap",
            overlap_2d,
            z_inv, B_vals, V_vals, 0.0, max(0.05, max_overlap_val), 'plasma', 'plasma',
            "Normalized MZM Overlap", "Wavefunction Overlap I",
            resolved_cuts, local_freestanding, OUT_DIR,
            contour_color='cyan'
        )

    # H. MZM Separability Phase Map
    sep_path = data_dir / "mzm_separability_arr.npy"
    if sep_path.exists():
        sep_raw = np.load(sep_path)
        sep_2d = sep_raw.reshape((len(V_vals), len(B_vals)))
        export_phase_map_single(
            "global_separability",
            sep_2d,
            z_inv, B_vals, V_vals, 0.5, 1.0, 'plasma', 'plasma',
            "MZM Separability", "Separability S",
            resolved_cuts, local_freestanding, OUT_DIR,
            contour_color='cyan'
        )

    # I. Excited State and Excitation Gap Phase Maps
    spec_path = data_dir / "spectrum_arr.npy"
    if spec_path.exists():
        spec_raw = np.load(spec_path)
        if spec_raw.ndim == 2 and spec_raw.shape[1] >= 4:
            E0_raw = spec_raw[:, 2]
            E1_raw = spec_raw[:, 3]
            E1_2d = E1_raw.reshape((len(V_vals), len(B_vals)))
            max_E1 = float(np.nanmax(E1_2d)) if np.nanmax(E1_2d) > 0 else 0.5
            export_phase_map_single(
                "global_excited_state_E1",
                E1_2d,
                z_inv, B_vals, V_vals, 0.0, max_E1, 'magma', 'magma',
                "First Excited State Energy ($E_1$)", "$E_1$ (meV)",
                resolved_cuts, local_freestanding, OUT_DIR,
                contour_color='cyan'
            )

            delta_E_raw = E1_raw - E0_raw
            delta_E_2d = delta_E_raw.reshape((len(V_vals), len(B_vals)))
            max_delta_E = float(np.nanmax(delta_E_2d)) if np.nanmax(delta_E_2d) > 0 else 0.5
            export_phase_map_single(
                "global_excitation_gap_E1_minus_E0",
                delta_E_2d,
                z_inv, B_vals, V_vals, 0.0, max_delta_E, 'plasma', 'plasma',
                "Excitation Gap ($E_1 - E_0$)", r"$\Delta E$ (meV)",
                resolved_cuts, local_freestanding, OUT_DIR,
                contour_color='cyan'
            )

    # J. Barrier Correlation Phase Maps (Monochromatic Blue and White: 'Blues')
    b_arr_path = data_dir / "barrier_arr.npy"
    bl_l_path = data_dir / "barrier_left_conductance_left_arr.npy"
    bl_r_path = data_dir / "barrier_left_conductance_right_arr.npy"
    br_l_path = data_dir / "barrier_right_conductance_left_arr.npy"
    br_r_path = data_dir / "barrier_right_conductance_right_arr.npy"

    corr_thresh_2d = None
    if b_arr_path.exists() and bl_l_path.exists() and bl_r_path.exists() and br_l_path.exists() and br_r_path.exists():
        logger.info("Computing and plotting barrier sweep correlation phase maps...")
        corr_L_file = data_dir / "barrier_left_correlation_arr.npy"
        corr_R_file = data_dir / "barrier_right_correlation_arr.npy"

        if corr_L_file.exists() and corr_R_file.exists():
            corr_L = np.load(corr_L_file)
            corr_R = np.load(corr_R_file)
        else:
            b_arr = np.load(b_arr_path)
            bl_l = np.load(bl_l_path)
            bl_r = np.load(bl_r_path)
            br_l = np.load(br_l_path)
            br_r = np.load(br_r_path)

            N_pts = len(bl_l)
            corr_L = np.array([hp.calc_correlation(bl_l[i], bl_r[i], b_arr) for i in range(N_pts)])
            corr_R = np.array([hp.calc_correlation(br_l[i], br_r[i], b_arr) for i in range(N_pts)])
            np.save(corr_L_file, corr_L)
            np.save(corr_R_file, corr_R)

        corr_L_2d = corr_L.reshape((len(V_vals), len(B_vals)))
        corr_R_2d = corr_R.reshape((len(V_vals), len(B_vals)))

        # 1. Dual side-by-side phase map (Left and Right barrier sweeps)
        export_phase_map_pair(
            "global_barrier_correlation",
            corr_L_2d,
            corr_R_2d,
            z_inv, B_vals, V_vals, 0.0, 1.0, 'Blues', 'Blues',
            "Left Barrier Sweep Correlation", "Right Barrier Sweep Correlation", "Barrier Correlation",
            resolved_cuts, local_freestanding, OUT_DIR,
            contour_color='cyan'
        )

        # 2. Standalone Left Barrier Sweep Correlation Phase Map
        export_phase_map_single(
            "global_barrier_left_correlation",
            corr_L_2d,
            z_inv, B_vals, V_vals, 0.0, 1.0, 'Blues', 'Blues',
            "Left Barrier Sweep Correlation (G_LL vs G_RR)", "Correlation",
            resolved_cuts, local_freestanding, OUT_DIR,
            contour_color='cyan'
        )

        # 3. Standalone Right Barrier Sweep Correlation Phase Map
        export_phase_map_single(
            "global_barrier_right_correlation",
            corr_R_2d,
            z_inv, B_vals, V_vals, 0.0, 1.0, 'Blues', 'Blues',
            "Right Barrier Sweep Correlation (G_LL vs G_RR)", "Correlation",
            resolved_cuts, local_freestanding, OUT_DIR,
            contour_color='cyan'
        )

        # 4. Thresholded / Binarized Dual Barrier Correlation Phase Map (both >= 0.85)
        corr_thresh = ((corr_L >= 0.85) & (corr_R >= 0.85)).astype(float)
        corr_thresh_2d = corr_thresh.reshape((len(V_vals), len(B_vals)))
        bin_cmap = mcolors.ListedColormap(['#f0f0f0', '#08519c'])
        export_phase_map_single(
            "global_barrier_correlation_threshold_85",
            corr_thresh_2d,
            z_inv, B_vals, V_vals, 0.0, 1.0, bin_cmap, 'Blues',
            r"Thresholded Barrier Correlation ($C_L \geq 0.85$ & $C_R \geq 0.85$)", "Correlation (0: <0.85, 1: >=0.85)",
            resolved_cuts, local_freestanding, OUT_DIR,
            contour_color='cyan'
        )

    # K. Clean Microsoft Stage 2 Paper Diagram, Sensitivity Sweeps, and TGP Cuts
    if stage2_ds is not None:
        process_tgp_cuts_and_sweeps(
            data_dir=data_dir,
            OUT_DIR=OUT_DIR,
            tprep=tprep,
            tprep_left=tprep_left,
            tprep_right=tprep_right,
            zbp_ds=zbp_ds,
            stage2_ds=stage2_ds,
            B_vals=B_vals,
            V_vals=V_vals,
            z_inv=z_inv,
            resolved_cuts=resolved_cuts,
            physics_params=physics_params,
            gap_th_factor=gap_th_factor,
            p=p,
            corr_thresh_2d=corr_thresh_2d if 'corr_thresh_2d' in locals() else None
        )

        # Ground-Truth Pfaffian vs Full TGP Island (ROI 2) Comparison Map
        try:
            fig_comp, (ax_comp1, ax_comp2) = plt.subplots(1, 2, figsize=(14.5, 6.0), sharey=True, layout='constrained')
            # Left: Ground truth Pfaffian
            pfaff_cmap = mcolors.ListedColormap(['#e0e0e0', '#2b5c8f'])
            im_c1 = ax_comp1.pcolormesh(B_vals, V_vals, z_inv, cmap=pfaff_cmap, vmin=0, vmax=1, shading='nearest')
            ax_comp1.set_title("Ground-Truth Pfaffian Topological Invariant (Q = -1)")
            ax_comp1.set_xlabel(r"Zeeman Field $V_z$ (meV)")
            ax_comp1.set_ylabel(r"Chemical Potential $\mu$ (meV)")

            # Right: Passing ROI 2 Topological Islands
            roi2_arr = stage2_ds.roi2.transpose('V', 'B').values if 'roi2' in stage2_ds else np.zeros_like(z_inv)
            roi2_bin = (roi2_arr > 0).astype(float)
            roi2_cmap = mcolors.ListedColormap(['#e0e0e0', '#2ca02c'])
            im_c2 = ax_comp2.pcolormesh(B_vals, V_vals, roi2_bin, cmap=roi2_cmap, vmin=0, vmax=1, shading='nearest')
            ax_comp2.set_title("Passing Full Topological Gap Protocol (ROI 2 Islands)")
            ax_comp2.set_xlabel(r"Zeeman Field $V_z$ (meV)")

            # Overlay contours and cuts on both
            for ax in (ax_comp1, ax_comp2):
                ax.contour(B_vals, V_vals, z_inv, levels=[0.5], colors='black', linewidths=1.2, linestyles='--')
                if 'roi2' in stage2_ds and np.nanmax(roi2_bin) > 0:
                    ax.contour(B_vals, V_vals, roi2_bin, levels=[0.5], colors='darkgreen', linewidths=1.5)
                for rcut in resolved_cuts:
                    cut = rcut['config']
                    ax.plot([cut['start'][1], cut['end'][1]], [cut['start'][0], cut['end'][0]],
                            color=cut['color'], linewidth=1.2, label=cut['label'])
                    for sp in rcut['resolved_snaps']:
                        ax.plot(sp['snapped_coords'][1], sp['snapped_coords'][0],
                                marker='*', color=sp['color'], markersize=6, linestyle='None')
                for fp in local_freestanding:
                    ax.plot(fp['coords'][1], fp['coords'][0], marker='*', color=fp['color'],
                            markersize=6, label=fp['label'], linestyle='None')

            ax_comp1.legend(loc='upper left', framealpha=0.8)
            fig_comp.suptitle(f"Topological Verification: Ground-Truth Pfaffian vs. Full TGP ROI2 ({data_dir.name})", fontsize=13)
            fig_comp.savefig(OUT_DIR / "pfaffian_vs_tgp_islands.png", dpi=300, bbox_inches='tight')
            plt.close(fig_comp)
            logger.info(f"Saved pfaffian_vs_tgp_islands.png to {OUT_DIR}")
        except Exception as e_comp:
            logger.warning(f"Could not generate pfaffian_vs_tgp_islands plot: {e_comp}")

    # 4. Cut Analysis (Multi-panel plotting)
    logger.info(f"Analyzing {len(resolved_cuts)} cuts...")
    L_2w_avg = tprep['L_2w_nl'].mean(dim='cutter_pair_index')
    invariant_avg = tprep['L_SI'].mean(dim='cutter_pair_index')
    
    for rcut in resolved_cuts:
        cut = rcut['config']
        actual_N = rcut['actual_N']
        pts = rcut['pts']
        
        if actual_N == 0:
            logger.info(f"  Skipping {cut['label']}: No points found along path.")
            continue
            
        cut_dir = OUT_DIR / "Cuts" / cut['label']
        cut_dir.mkdir(parents=True, exist_ok=True)
        
        kvals = 14
        evals = np.zeros((actual_N, kvals))
        val_2w = np.zeros(actual_N)
        val_3w = np.zeros(actual_N)
        val_overlap = np.zeros(actual_N)
        pfaffians = np.zeros(actual_N)
        val_gap_left = np.zeros(actual_N)
        val_gap_right = np.zeros(actual_N)
        val_corr_thresh = np.zeros(actual_N, dtype=bool)
        
        val_tgp_roi2 = np.zeros(actual_N, dtype=bool)
        cut_point_tasks = [
            (i, pts[i, 0], pts[i, 1], t_val, gamma, Delta0, alpha, Ls, Vdisx, kvals)
            for i in range(actual_N)
        ]
        with ProcessPoolExecutor(max_workers=12) as executor:
            point_results = list(executor.map(_eval_cut_point_worker, cut_point_tasks))

        for i, sorted_evals, overlap in point_results:
            evals[i, :] = sorted_evals
            val_overlap[i] = overlap
            mu_val, vz_val = pts[i, 0], pts[i, 1]

            val_2w[i] = L_2w_avg.sel(V=mu_val, B=vz_val, method='nearest').values.item()
            val_3w[i] = tprep['L_3w'].mean(dim='cutter_pair_index').sel(V=mu_val, B=vz_val, method='nearest').values.item()
            pfaffians[i] = invariant_avg.sel(V=mu_val, B=vz_val, method='nearest').values.item()

            if corr_thresh_2d is not None:
                v_idx = np.argmin(np.abs(V_vals - mu_val))
                b_idx = np.argmin(np.abs(B_vals - vz_val))
                val_corr_thresh[i] = bool(corr_thresh_2d[v_idx, b_idx] >= 1.0)

            if has_tgp_gap:
                val_gap_left[i] = gap_left_avg.sel(V=mu_val, B=vz_val, method='nearest').values.item()
                val_gap_right[i] = gap_right_avg.sel(V=mu_val, B=vz_val, method='nearest').values.item()

            if stage2_ds is not None and 'roi2' in stage2_ds:
                val_tgp_roi2[i] = bool(stage2_ds.roi2.sel(B=vz_val, V=mu_val, method='nearest').values > 0)

        fig, axes = plt.subplots(4, 1, figsize=(7.5, 9.2), gridspec_kw={'height_ratios': [2, 1, 1, 0.7]}, sharex=True)
        
        # Shading
        for idx in range(actual_N):
            is_pfaff = pfaffians[idx] > 0
            is_corr = val_corr_thresh[idx]
            if is_pfaff and is_corr:
                color = '#5d768d'
                alpha = 0.55
            elif is_pfaff and not is_corr:
                color = '#555555'
                alpha = 0.45
            elif is_corr and not is_pfaff:
                color = '#99ccff'
                alpha = 0.5
            else:
                continue
            for ax in axes:
                ax.axvspan(idx - 0.5, idx + 0.5, color=color, alpha=alpha, zorder=0)

        # Draw Snapped Point Vertical Lines
        for sp in rcut['resolved_snaps']:
            for ax in axes:
                ax.axvline(sp['snap_idx'], color=sp['color'], linestyle='--', linewidth=1, alpha=0.7)

        # Panel 1: Spectra
        mid_idx = kvals // 2
        for j in range(kvals):
            color = 'red' if j in [mid_idx-1, mid_idx] else 'royalblue'
            lw = 2.5 if j in [mid_idx-1, mid_idx] else 1.5
            axes[0].plot(range(actual_N), evals[:, j], color=color, linewidth=lw, alpha=0.8, zorder=2)
        axes[0].axhline(0, color='black', linestyle='--', alpha=0.6)
        axes[0].set_ylabel("Energy (meV)")
        axes[0].set_title(f"Spectra along {cut['label']}")

        # Highlight full TGP Island along bottom runner with red ticks on bottom axis
        if stage2_ds is not None and 'roi2' in stage2_ds:
            tgp_plotted_m = False
            ymin = np.min(evals) - 0.08 * (np.max(evals) - np.min(evals))
            ymax = np.max(evals) + 0.08 * (np.max(evals) - np.min(evals))
            axes[0].set_ylim(ymin, ymax)
            y_bar = ymin + 0.03 * (ymax - ymin)
            for idx in range(actual_N):
                if bool(val_tgp_roi2[idx]):
                    axes[0].plot([idx - 0.5, idx + 0.5], [y_bar, y_bar], color='#2ca02c', linewidth=4.5, solid_capstyle='butt', zorder=5, label="TGP Island (ROI 2)" if not tgp_plotted_m else "")
                    axes[0].plot(idx, ymin, marker='|', color='red', markersize=8, markeredgewidth=1.8, zorder=6)
                    tgp_plotted_m = True
            if tgp_plotted_m:
                axes[0].legend(loc='upper right', fontsize=8, framealpha=0.8)

        # Panel 2: 3w Measurement (Capped)
        cap_val = 1.4 * DERIVATIVE_THRESHOLD
        val_3w_capped = np.clip(val_3w, -cap_val, cap_val)
        axes[1].plot(range(actual_N), val_3w_capped, color='darkorange', linewidth=1.5, marker='s', markersize=3, zorder=2)
        axes[1].axhline(-DERIVATIVE_THRESHOLD, color='black', linestyle=':', linewidth=1.5, label='ZBP Threshold', zorder=3)
        axes[1].set_ylim(-cap_val, cap_val)
        axes[1].set_ylabel("3ω Curvature")
        axes[1].set_title(f"3ω Curvature (Strictly Capped to ±{cap_val:.3f})")
        axes[1].legend(loc='upper right')

        # Panel 3: Transport Gap and Lowest States (with threshold line)
        axes[2].plot(range(actual_N), np.minimum(val_gap_left, val_gap_right), color='purple', linewidth=1.5, marker='^', markersize=3, zorder=2, label=r"Min $\Delta_{ex}$")
        axes[2].plot(range(actual_N), evals[:, mid_idx], color='red', linewidth=1.5, linestyle='--', zorder=1, label=r"$E_0$")
        axes[2].plot(range(actual_N), evals[:, mid_idx + 1], color='blue', linewidth=1.5, linestyle='--', zorder=1, label=r"$E_1$")
        axes[2].axhline(0.010, color='black', linestyle='--', linewidth=1.2, label=r'Gap Th ($10\ \mu\mathrm{eV}$)', zorder=3)
        axes[2].set_ylabel(r"Gap / Energy (meV)")
        axes[2].set_title("Transport Gap & Lowest Spectral States")
        axes[2].legend(loc='upper right', fontsize=8)

        # Panel 4: Thresholded Correlation Condition (±1.5 binarized track)
        corr_bin = np.where(val_corr_thresh, 1.0, -1.0)
        axes[3].step(range(actual_N), corr_bin, where='mid', color='#0284c7', linewidth=1.8, zorder=3)
        for idx in range(actual_N):
            c_col = '#10b981' if val_corr_thresh[idx] else '#94a3b8'
            c_alpha = 0.55 if val_corr_thresh[idx] else 0.35
            axes[3].axvspan(idx - 0.5, idx + 0.5,
                            ymin=0.5 if val_corr_thresh[idx] else 0.0,
                            ymax=1.0 if val_corr_thresh[idx] else 0.5,
                            color=c_col, alpha=c_alpha, zorder=2)
        axes[3].axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=4)
        axes[3].axhline(1.0, color='#10b981', linestyle='--', linewidth=0.8, alpha=0.7, zorder=4)
        axes[3].axhline(-1.0, color='#94a3b8', linestyle='--', linewidth=0.8, alpha=0.7, zorder=4)
        axes[3].set_ylim(-1.5, 1.5)
        axes[3].set_yticks([-1.0, 0.0, 1.0])
        axes[3].set_yticklabels(["Fail (-1)", "0", "Pass (+1)"], fontsize=8)
        axes[3].set_ylabel("Barrier Corr")
        axes[3].set_title(r"Thresholded Barrier Correlation ($C_L \geq 0.85$ & $C_R \geq 0.85$)", fontsize=9)

        # X-axes (Dual)
        tick_indices = np.round(np.linspace(0, actual_N - 1, min(10, actual_N))).astype(int)
        axes[3].set_xticks(tick_indices)
        axes[3].set_xticklabels([f"{pts[idx, 0]:.3f}" for idx in tick_indices], rotation=45, ha='right')
        axes[3].set_xlabel(r"Chemical Potential $\mu$ (meV)")
        
        ax_top = axes[0].twiny()
        ax_top.set_xlim(axes[0].get_xlim())
        ax_top.set_xticks(tick_indices)
        ax_top.set_xticklabels([f"{pts[idx, 1]:.3f}" for idx in tick_indices], rotation=45, ha='left')
        ax_top.set_xlabel(r"Zeeman Field $V_z$ (meV)")

        plt.tight_layout()
        fig.savefig(cut_dir / "multi_panel.png", dpi=300)
        plt.close(fig)

        # Export 4 individual conductance plots along this cut (G_LL, G_RR, A_GRL, A_GLR)
        logger.info(f"  Generating 4 individual conductance plots for {cut['label']}...")
        export_cut_conductance_plots(
            cut_dir=cut_dir,
            cut=cut,
            pts=pts,
            rcut=rcut,
            tprep=tprep,
            tprep_left=tprep_left if has_tgp_gap else None,
            tprep_right=tprep_right if has_tgp_gap else None,
            has_tgp_gap=has_tgp_gap,
            selected_cutter=0,
            evals=evals,
            pfaffians=pfaffians,
            corr_thresh=val_corr_thresh,
            tgp_roi2=val_tgp_roi2
        )

    # 5. Point Analysis (Parallel dispatch across 12 workers)
    logger.info(f"Analyzing {len(points_to_analyze)} deep-dive points in parallel (12 workers)...")
    physics_params = {
        't_val': t_val,
        'mu_n': mu_n,
        'mu_leads': mu_leads,
        'gamma': gamma,
        'Delta0': Delta0,
        'alpha': alpha,
        'Ln': Ln,
        'Lb': Lb,
        'Ls': Ls,
        'barrier_l_base': barrier_l_base,
        'Vdisx': Vdisx,
    }
    deep_dive_tasks = []
    for pt in points_to_analyze:
        mu_val, vz_val = pt['coords']
        pt_2w = L_2w_avg.sel(V=mu_val, B=vz_val, method='nearest').values.item()
        pt_3w = tprep['L_3w'].mean(dim='cutter_pair_index').sel(V=mu_val, B=vz_val, method='nearest').values.item()
        deep_dive_tasks.append((pt, physics_params, p.T_mK_stage1, pt_2w, pt_3w, BARRIER_SWEEP_SCALE))

    with ProcessPoolExecutor(max_workers=12) as executor:
        list(executor.map(_process_deep_dive_point_worker, deep_dive_tasks))

    logger.info("Done!")

    # Explicitly free memory for sequential processing
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
    except ImportError:
        pass
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Analyze cuts for a series of data_dir folders")
    parser.add_argument("data_dirs", nargs="*", help="List of active data directory names (relative to PathConfigs.DATA) or absolute paths")
    args = parser.parse_args()

    if not args.data_dirs:
        # Default behavior
        data_dirs_to_process = DEFAULT_DATA_DIRS
    else:
        data_dirs_to_process = args.data_dirs

    for d in data_dirs_to_process:
        p_dir = Path(d)
        if not p_dir.is_absolute():
            p_dir = PathConfigs.DATA / d
        
        if not p_dir.exists():
            logger.error(f"Directory {p_dir} does not exist. Skipping.")
            continue
            
        try:
            logger.info(f"--- Processing Directory: {p_dir} ---")
            process_data_dir(p_dir)
        except Exception as e:
            logger.error(f"Failed processing {p_dir}: {e}", exc_info=True)
            continue

if __name__ == "__main__":

    main()
