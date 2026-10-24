r"""Option 2: Local Quadrature Modulation Protocol (`LQM-P`) (`2026-10-21`).

Implements Consciousness Council Option 2:
1. `step_1_fpe_pinch_off`: Pinch-off suppression gate $G_{\alpha\alpha}^{\text{pinch}} < G_{\text{thresh}}$
2. `step_2_fpe_detrended_power`: Detrended oscillatory power $P_\alpha(V_z, \mu) \ge \theta_P$
3. `step_3_fpe_quadrature_coherence`: Savitzky-Golay quadrature cross-envelope coherence
   $\mathcal{C}_{LR}^2(V_z, \mu) \ge \theta_C$ (100% invariant to disorder phase shifts $\Delta\Phi$)
4. `step_4_dpe_conjunction`: Logical AND of Steps 1–3
5. `step_5_dpe_C4_final_mask`: Canonical post-processing island pruning $C_4$.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

import numpy as np

from src.Protocol_Elements import DPE, FPE, BaseProtocol, RealizationFeatureCache


class Protocol(BaseProtocol):
    name = "LQMP_Quadrature_Protocol"
    description = (
        "Option 2 Local Quadrature Modulation Protocol combining pinch-off suppression, "
        "detrended oscillatory power, phase-shift-invariant Savitzky-Golay quadrature "
        "coherence, and C4 island pruning."
    )

    default_parameters: Dict[str, Any] = {
        "fpe_pinch_off": {
            "lead": "both",
            "g_thresh": 0.05,
            "pinch_asym_ratio": 60.0,
        },
        "fpe_power": {
            "lead": "min_LR",
            "theta_P": 0.002,
            "win_vz": 5,
            "expected_lambda_mev": 0.18,
            "sym_asym_ratio": 0.0,
            "pinch_asym_ratio": 60.0,
        },
        "fpe_coherence": {
            "theta_C": 0.25,
            "win_vz": 5,
            "expected_lambda_mev": 0.18,
            "sym_asym_ratio": 0.0,
            "pinch_asym_ratio": 60.0,
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

        m_pinch = FPE.fpe_pinch_off_conductance_gate(cache, **params["fpe_pinch_off"])
        steps["step_1_fpe_pinch_off"] = m_pinch

        m_pow = FPE.fpe_lqmp_detrended_power(cache, **params["fpe_power"])
        steps["step_2_fpe_detrended_power"] = m_pow

        m_coh = FPE.fpe_lqmp_quadrature_coherence(cache, **params["fpe_coherence"])
        steps["step_3_fpe_quadrature_coherence"] = m_coh

        m_cand = DPE.dpe_and(m_pinch, m_pow, m_coh)
        steps["step_4_dpe_conjunction"] = m_cand

        final_mask = DPE.dpe_canonical_post_processor(
            m_cand,
            mu_vals=cache.mu_vals,
            vz_vals=cache.vz_vals,
            **params["dpe_c4_post_processor"],
        )
        steps["step_5_dpe_C4_final_mask"] = final_mask
        return steps
