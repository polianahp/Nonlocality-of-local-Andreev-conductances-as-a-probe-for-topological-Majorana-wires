r"""Symmetric Asymmetric Conductance Deviation (`S-ACD`) + Multi-Cutter $3\omega$ + Gap Protocol (`2026-10-22`).

Encapsulates the S-ACD Precision-Optimized, Yield-Optimized, and Variations 1–6 protocols
from `Outputs/Postprocessing/Variations_30mK/` and the `2026-10-22` logs:
- Step 1 (`C_2` S-ACD): $\sqrt{a_L a_R} \ge \tau$ (or max-pooling $a_L \ge \tau \lor a_R \ge \tau$ to test Variation 5)
- Step 2 (`C_1` Multi-Cutter Curvature): Per-cutter $\partial^2 G / \partial V^2|_{V=0} \le -t_{3\omega}$
  followed by $k$-of-$N$ cutter consensus (`k_min` in `{1, 2, 3, 5}`)
- Step 3 (`C_gap`): Spectral or nonlocal transport gap $\Delta \ge \Delta_{\min}$
- Step 4 (`C_QD`, optional): Variation 4 Detuned Quantum Dot resonant conductance filter
- Step 5 (Boolean Combinator):
  - `"strict_and"`: $C_{\text{ACD}} \land C_{\text{curv}} \land C_{\text{gap}}$
  - `"complementary_or"` (Variation 1): $C_{\text{ACD}} \land (C_{\text{curv}} \lor C_{\text{gap}})$
  - `"acd_and_curv_only"`: $C_{\text{ACD}} \land C_{\text{curv}}$
  - `"acd_only"`: $C_{\text{ACD}}$
- Step 6 (`C_4`): Subtractive area pruning + optional $2\times 2$ core filter / OPTICS.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

import numpy as np

from src.Protocol_Elements import DPE, FPE, BaseProtocol, RealizationFeatureCache


class Protocol(BaseProtocol):
    name = "SACD_Hybrid_Curvature_Gap_Protocol"
    description = (
        "S-ACD geometric-mean barrier asymmetry combined with k-of-N cutter zero-bias "
        "curvature consensus, spectral/transport gap gate, optional QD sweet-spot proxy, "
        "and C4 island pruning."
    )

    default_parameters: Dict[str, Any] = {
        "fpe_acd": {
            "mode": "geometric_mean",  # "geometric_mean" or "max_LR_or" (Variation 5)
            "tau": 0.05,
            "floor_val": 1.0e-3,
            "sym_asym_ratio": 0.0,
            "pinch_asym_ratio": 60.0,
            "use_signed_diff": False,
        },
        "fpe_curvature": {
            "curvature_threshold": 20.0,
            "lead": "R",
            "curvature_source": "thermal_lockin_3w",
            "scale_mode": "physical_dynamic_grid",
            "savgol_window_length": 7,
            "savgol_polyorder": 2,
        },
        "dpe_cutter_consensus": {
            "k_min": 3,
            "min_fraction": None,
        },
        "fpe_gap": {
            "gap_type": "nonlocal_transport",
            "min_gap_mev": 0.015,
            "max_gap_mev": None,
            "gap_threshold_factor": 0.05,
            "noise_threshold": 1.0e-4,
            "use_thermal_broadened": True,
        },
        "fpe_qd_gamma": {
            "enabled": False,
            "resonant_g_target": 0.50,
            "max_g_dev": 0.25,
            "sym_asym_ratio": 0.0,
        },
        "combinator": {
            "mode": "acd_and_curv_only",
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
            "min_island_size": 10,
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

        # Step 1: S-ACD barrier asymmetry FPE
        acd_cfg = dict(params["fpe_acd"])
        acd_mode = acd_cfg.pop("mode", "geometric_mean")
        if acd_mode == "geometric_mean":
            m_acd = FPE.fpe_acd_geometric_mean(cache, **acd_cfg)
        elif acd_mode == "max_LR_or":
            m_L = FPE.fpe_single_lead_acd(cache, lead="L", **acd_cfg)
            m_R = FPE.fpe_single_lead_acd(cache, lead="R", **acd_cfg)
            m_acd = DPE.dpe_or(m_L, m_R)
        else:
            raise ValueError(f"Unknown fpe_acd.mode '{acd_mode}'.")
        steps["step_1_fpe_sacd"] = m_acd

        # Step 2: Per-cutter zero-bias curvature FPE + k-of-N consensus DPE
        curv_3d = FPE.fpe_zero_bias_curvature_single_cutter(
            cache, **params["fpe_curvature"]
        )
        steps["step_2_fpe_curvature_3d"] = curv_3d
        m_curv = DPE.dpe_k_of_n_cutter_consensus(
            curv_3d, **params["dpe_cutter_consensus"]
        )
        steps["step_3_dpe_curvature_k_of_n"] = m_curv

        # Step 3: Open-system Nonlocal Transport Gap FPE
        g_cfg = params["fpe_gap"]
        if g_cfg["gap_type"] != "nonlocal_transport":
            raise ValueError(
                f"Closed-system gap_type='{g_cfg['gap_type']}' (E1) is theoretically unobservable and "
                "forbidden in experimental protocols; use gap_type='nonlocal_transport'."
            )
        m_gap = FPE.fpe_nonlocal_conductance_transport_gap(
            cache,
            min_gap_mev=g_cfg["min_gap_mev"],
            max_gap_mev=g_cfg["max_gap_mev"],
            gap_threshold_factor=g_cfg["gap_threshold_factor"],
            noise_threshold=g_cfg["noise_threshold"],
            use_thermal_broadened=bool(g_cfg.get("use_thermal_broadened", True)),
        )
        steps["step_4_fpe_gap"] = m_gap

        # Step 4: Logical combination
        c_mode = params["combinator"]["mode"]
        if c_mode == "strict_and":
            candidate = DPE.dpe_and(m_acd, m_curv, m_gap)
        elif c_mode == "complementary_or":
            candidate = DPE.dpe_and(m_acd, DPE.dpe_or(m_curv, m_gap))
        elif c_mode == "acd_and_curv_only":
            candidate = DPE.dpe_and(m_acd, m_curv)
        elif c_mode == "acd_only":
            candidate = m_acd.copy()
        else:
            raise ValueError(f"Unknown combinator.mode '{c_mode}'.")

        # Step 5: Optional QD sweet-spot filter (Variation 4)
        qd_cfg = dict(params["fpe_qd_gamma"])
        if bool(qd_cfg.pop("enabled", False)):
            m_qd = FPE.fpe_quantum_dot_gamma_squared(cache, **qd_cfg)
            steps["step_5_fpe_qd_gamma"] = m_qd
            candidate = DPE.dpe_and(candidate, m_qd)

        steps["step_6_pre_C4_candidate"] = candidate

        # Step 6: Mandatory Terminal DPE C4
        final_mask = DPE.dpe_canonical_post_processor(
            candidate,
            mu_vals=cache.mu_vals,
            vz_vals=cache.vz_vals,
            **params["dpe_c4_post_processor"],
        )
        steps["step_7_dpe_C4_final_mask"] = final_mask
        return steps
