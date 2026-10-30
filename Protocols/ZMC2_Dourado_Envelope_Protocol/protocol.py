r"""Option 3 (`ZMC-2`) & Dourado et al. PRB 110, 014504 (2024) Envelope Protocol (`2026-10-21/22`).

Implements:
- Step 1 (`fpe_acd_geometric_mean`): $\sqrt{a_L a_R} \ge \tau$
- Step 2 (`fpe_lead_symmetry_balance`): Dourado Fig. 13 conjugate symmetry balance $S_{\text{sym}} \le s_{\max}$
- Step 3 (`fpe_local_vz_oscillation_correlation`, optional): Dourado Appendix G Figs. 20–21
  1D sliding-window Pearson correlation along $V_z$ ($r_{\text{local}} \ge r_{\text{th}}$)
- Step 4 (`dpe_1d_zeeman_closing`): 1D horizontal morphological closing along $V_z$ (`radius_vz`)
- Step 5 (`dpe_canonical_post_processor` $C_4$): Subtractive island pruning.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

import numpy as np

from src.Protocol_Elements import DPE, FPE, BaseProtocol, RealizationFeatureCache


class Protocol(BaseProtocol):
    name = "ZMC2_Dourado_Envelope_Protocol"
    description = (
        "Option 3 ZMC-2 and Dourado et al. PRB 2024 protocol combining geometric-mean ACD, "
        "conjugate lead symmetry balance S_sym, optional Appendix G local Vz oscillation "
        "correlation, 1D Zeeman closing, and C4 island pruning."
    )

    default_parameters: Dict[str, Any] = {
        "fpe_acd": {
            "tau": 0.08,
            "floor_val": 1.0e-3,
            "sym_asym_ratio": 0.0,
            "pinch_asym_ratio": 60.0,
            "use_signed_diff": False,
        },
        "fpe_symmetry_balance": {
            "enabled": True,
            "s_max": 0.20,
            "eps_sym": 1.0e-4,
            "sym_asym_ratio": 0.0,
        },
        "fpe_local_vz_corr": {
            "enabled": False,
            "r_threshold": 0.50,
            "window_pts": 5,
            "observable": "G_sym",
            "sym_asym_ratio": 0.0,
            "pinch_asym_ratio": 60.0,
            "variance_floor": 1.0e-10,
            "use_cosine": False,
        },
        "dpe_1d_closing": {
            "radius_vz": 1,
        },
        "dpe_c4_post_processor": {
            "enabled": True,
            "use_core_filter": False,
            "core_shape": [2, 2],
            "core_connectivity": 8,
            "core_mode": "component_has_core",
            "use_optics": False,
            "optics_min_samples": 5,
            "optics_max_eps": 3.0,
            "optics_min_cluster_size": 10,
            "optics_xi": 0.05,
            "optics_coordinate_space": "grid_indices",
            "min_island_size": 12,
            "area_connectivity": 8,
            "fallback_min_island_size": 4,
            "fallback_top_n": 6,
        },
    }

    def evaluate_steps(
        self,
        cache: RealizationFeatureCache,
        params: Dict[str, Any],
    ) -> OrderedDict[str, np.ndarray]:
        steps: OrderedDict[str, np.ndarray] = OrderedDict()

        m_acd = FPE.fpe_acd_geometric_mean(cache, **params["fpe_acd"])
        steps["step_1_fpe_acd_geom"] = m_acd
        cand = m_acd.copy()

        sym_cfg = dict(params["fpe_symmetry_balance"])
        if bool(sym_cfg.pop("enabled", True)):
            m_sym = FPE.fpe_lead_symmetry_balance(cache, **sym_cfg)
            steps["step_2_fpe_symmetry_balance"] = m_sym
            cand = DPE.dpe_and(cand, m_sym)

        vzc_cfg = dict(params["fpe_local_vz_corr"])
        if bool(vzc_cfg.pop("enabled", False)):
            m_vzc = FPE.fpe_local_vz_oscillation_correlation(cache, **vzc_cfg)
            steps["step_3_fpe_local_vz_corr"] = m_vzc
            cand = DPE.dpe_and(cand, m_vzc)

        m_closed = DPE.dpe_1d_zeeman_closing(cand, **params["dpe_1d_closing"])
        steps["step_4_dpe_1d_zeeman_closing"] = m_closed

        final_mask = DPE.dpe_canonical_post_processor(
            m_closed,
            mu_vals=cache.mu_vals,
            vz_vals=cache.vz_vals,
            **params["dpe_c4_post_processor"],
        )
        steps["step_5_dpe_C4_final_mask"] = final_mask
        return steps
