r"""Low-Asymmetry Barrier Reflection Correlation + Multi-Cutter 3w Curvature + Transport Gap Protocol.

User Specification (2026-10-28):
1. Low-asymmetry barrier reflection correlation: C_L >= 0.75 and C_R >= 0.75 in [-5, +5] gate window
2. 3w zero-bias curvature: <= -100 (e^2/h)/meV^2 on BOTH sides (Left & Right) for >= 3 out of 5 cutters (leads)
3. Nonlocal transport gap: Delta_transport >= 10 ueV (0.010 meV)
4. Canonical C4 Post-Processing: Minimum island area >= 10 px, optional 2x2 core filter.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

import numpy as np

from src.Protocol_Elements import DPE, FPE, BaseProtocol, RealizationFeatureCache


class Protocol(BaseProtocol):
    name = "Low_Asym_Corr_Curv3w_Transport_Gap_Protocol"
    description = (
        "Low-asymmetry barrier reflection correlation (C >= 0.75 in [-5, +5]) combined with "
        "zero-bias 3w curvature <= -100 on both sides for 3 out of 5 cutters, "
        "transport gap >= 10 ueV, and C4 spatial island filtering."
    )

    default_parameters: Dict[str, Any] = {
        "fpe_corr": {
            "corr_threshold": 0.75,
            "asym_min": -5.0,
            "asym_max": 5.0,
            "mean_subtract": False,
            "normalize_by_first_point": False,
            "norm_floor": 1.0e-12,
        },
        "dpe_corr_lead": {
            "mode": "both",  # C_L >= 0.75 and C_R >= 0.75
        },
        "fpe_3w": {
            "curvature_threshold": 100.0,  # <= -100
        },
        "dpe_cutter_consensus": {
            "k_min": 3,  # 3 of 5 cutters
        },
        "fpe_gap": {
            "min_gap_mev": 0.010,  # >= 10 ueV (0.010 meV)
            "max_gap_mev": None,
            "gap_threshold_factor": 0.05,
            "noise_threshold": 1.0e-4,
            "median_size": 2,
            "use_thermal_broadened": True,
            "global_field_threshold": True,
            "pooling": "cutter_mean_min_LR",
            "k_min_cutters": 3,
        },
        "fpe_lead_symmetry": {
            "enabled": False,
            "s_max": 0.25,
            "eps_sym": 1.0e-4,
        },
        "dpe_c4_post_processor": {
            "enabled": True,
            "use_core_filter": True,
            "core_shape": [2, 2],
            "core_connectivity": 8,
            "core_mode": "component_has_core",
            "use_optics": False,
            "min_island_size": 10,
            "area_connectivity": 8,
            "fallback_min_island_size": None,
            "fallback_top_n": None,
        },
    }

    def evaluate_steps(
        self,
        cache: RealizationFeatureCache,
        params: Dict[str, Any],
    ) -> OrderedDict[str, np.ndarray]:
        steps: OrderedDict[str, np.ndarray] = OrderedDict()

        # 1. Step 1: Low-asymmetry reflection correlation (both leads >= corr_threshold in [-5, +5])
        corr_cfg = dict(params["fpe_corr"])
        asym_min = float(corr_cfg.get("asym_min", -5.0))
        asym_max = float(corr_cfg.get("asym_max", 5.0))
        corr_th = float(corr_cfg.get("corr_threshold", 0.75))

        cL, cR = FPE.barrier_correlation.extract_barrier_reflection_correlations(
            cache,
            asym_min=asym_min,
            asym_max=asym_max,
            mean_subtract=bool(corr_cfg.get("mean_subtract", False)),
            normalize_by_first_point=bool(corr_cfg.get("normalize_by_first_point", False)),
            norm_floor=float(corr_cfg.get("norm_floor", 1.0e-12)),
        )
        steps["step_1a_cL"] = cL
        steps["step_1b_cR"] = cR

        mode = params.get("dpe_corr_lead", {}).get("mode", "both")
        if mode == "both":
            m_corr = (cL >= corr_th) & (cR >= corr_th)
        elif mode == "either":
            m_corr = (cL >= corr_th) | (cR >= corr_th)
        elif mode == "R_only":
            m_corr = cR >= corr_th
        elif mode == "L_only":
            m_corr = cL >= corr_th
        else:
            m_corr = (cL >= corr_th) & (cR >= corr_th)
        steps["step_1_m_corr"] = m_corr

        # 2. Step 2: 3w zero-bias curvature <= -100 on both sides for 3 of 5 cutters
        curv_th = float(params.get("fpe_3w", {}).get("curvature_threshold", 100.0))
        k_min = int(params.get("dpe_cutter_consensus", {}).get("k_min", 3))

        lockin = cache.get_stage1_broadened_and_lockin()
        I_3w_L = lockin["I_3w_L"]  # shape (5, N_mu, N_Vz)
        I_3w_R = lockin["I_3w_R"]  # shape (5, N_mu, N_Vz)

        both_sides_curv = (I_3w_L <= -curv_th) & (I_3w_R <= -curv_th)
        m_curv = np.sum(both_sides_curv, axis=0) >= k_min
        steps["step_2_m_curv_3of5"] = m_curv

        # 3. Step 3: Open-system nonlocal conductance transport gap >= 10 ueV (0.010 meV)
        g_cfg = params.get("fpe_gap", {})
        m_gap = FPE.fpe_nonlocal_conductance_transport_gap(
            cache,
            min_gap_mev=float(g_cfg.get("min_gap_mev", 0.010)),
            max_gap_mev=g_cfg.get("max_gap_mev", None),
            gap_threshold_factor=float(g_cfg.get("gap_threshold_factor", 0.05)),
            noise_threshold=float(g_cfg.get("noise_threshold", 1.0e-4)),
            median_size=int(g_cfg.get("median_size", 2)),
            use_thermal_broadened=bool(g_cfg.get("use_thermal_broadened", True)),
            global_field_threshold=bool(g_cfg.get("global_field_threshold", True)),
            pooling=str(g_cfg.get("pooling", "cutter_mean_min_LR")),
            k_min_cutters=int(g_cfg.get("k_min_cutters", 3)),
        )
        steps["step_3_m_gap"] = m_gap

        # 4. Optional Step 3b: Lead symmetry balance S_sym <= s_max
        sym_cfg = params.get("fpe_lead_symmetry", {})
        if sym_cfg.get("enabled", False):
            s_max = float(sym_cfg.get("s_max", 0.25))
            eps_sym = float(sym_cfg.get("eps_sym", 1.0e-4))
            m_sym = FPE.barrier_asymmetry.fpe_lead_symmetry_balance(cache, s_max=s_max, eps_sym=eps_sym)
            steps["step_3b_m_sym"] = m_sym
            raw_candidate = m_corr & m_curv & m_gap & m_sym
        else:
            raw_candidate = m_corr & m_curv & m_gap

        steps["step_4_raw_candidate"] = raw_candidate

        # 5. Step 5: C4 Post-processing
        c4_cfg = dict(params.get("dpe_c4_post_processor", {}))
        if c4_cfg.get("enabled", True):
            final_mask = DPE.dpe_canonical_post_processor(
                raw_candidate,
                mu_vals=cache.mu_vals,
                vz_vals=cache.vz_vals,
                **c4_cfg,
            )
        else:
            final_mask = raw_candidate

        steps["step_5_final_mask"] = final_mask
        return steps
