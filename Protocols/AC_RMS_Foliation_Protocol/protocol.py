r"""Option 1: Adaptive Coincident RMS Energy (`AC-RMS`) Foliation Protocol (`2026-10-21`).

Implements the exact algebraic decomposition of Consciousness Council Option 1 (including the
Round 1 Critique 3 high-plateau bypass $1.30\,\theta(T)$):
1. `step_1_fpe_ac_rms_base`: $\mathcal{C}_{\text{asym}} \ge \theta(T)$ (`multiplier=1.0`)
2. `step_2_fpe_ac_rms_high_plateau`: $\mathcal{C}_{\text{asym}} \ge 1.30\,\theta(T)$ (`multiplier=1.30`)
3. `step_3_fpe_inflection`: $\partial^2 \mathcal{C}_{\text{asym}} / \partial V_z^2 \le 0$
4. `step_4_dpe_plateau_or_inflected`: $\mathcal{F}_{\text{AC-RMS}}(1.30\,\theta) \lor (\mathcal{F}_{\text{AC-RMS}}(\theta) \land \mathcal{F}_{\text{inflect}}(0))$
5. `step_5_dpe_1d_zeeman_closing`: 1D morphological closing along $V_z$ (`radius_vz=1`)
6. `step_6_dpe_C4_final_mask`: Canonical post-processing island pruning $C_4$.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

import numpy as np

from src.Protocol_Elements import DPE, FPE, BaseProtocol, RealizationFeatureCache


class Protocol(BaseProtocol):
    name = "AC_RMS_Foliation_Protocol"
    description = (
        "Option 1 Adaptive Coincident RMS Energy along Vz with high-plateau inflection "
        "bypass, 1D Zeeman closing, and C4 island pruning."
    )

    default_parameters: Dict[str, Any] = {
        "fpe_ac_rms": {
            "window_pts": 5,
            "use_adaptive_percentile": True,
            "percentile_rank": 95.0,
            "threshold_factor": 0.35,
            "static_threshold": 0.20,
            "high_plateau_multiplier": 1.30,
            "floor_val": 1.0e-3,
            "sym_asym_ratio": 0.0,
            "pinch_asym_ratio": 60.0,
        },
        "fpe_inflection": {
            "max_d2": 0.0,
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
        rms_cfg = dict(params["fpe_ac_rms"])
        hi_mult = float(rms_cfg.pop("high_plateau_multiplier", 1.30))

        m_base = FPE.fpe_ac_rms_energy(cache, multiplier=1.0, **rms_cfg)
        steps["step_1_fpe_ac_rms_base"] = m_base

        m_high = FPE.fpe_ac_rms_energy(cache, multiplier=hi_mult, **rms_cfg)
        steps["step_2_fpe_ac_rms_high_plateau"] = m_high

        m_inflect = FPE.fpe_ac_rms_inflection(
            cache,
            window_pts=rms_cfg["window_pts"],
            floor_val=rms_cfg["floor_val"],
            max_d2=params["fpe_inflection"]["max_d2"],
            sym_asym_ratio=rms_cfg["sym_asym_ratio"],
            pinch_asym_ratio=rms_cfg["pinch_asym_ratio"],
        )
        steps["step_3_fpe_inflection"] = m_inflect

        m_comb = DPE.dpe_or(m_high, DPE.dpe_and(m_base, m_inflect))
        steps["step_4_dpe_plateau_or_inflected"] = m_comb

        m_closed = DPE.dpe_1d_zeeman_closing(m_comb, **params["dpe_1d_closing"])
        steps["step_5_dpe_1d_zeeman_closing"] = m_closed

        final_mask = DPE.dpe_canonical_post_processor(
            m_closed,
            mu_vals=cache.mu_vals,
            vz_vals=cache.vz_vals,
            **params["dpe_c4_post_processor"],
        )
        steps["step_6_dpe_C4_final_mask"] = final_mask
        return steps
