r"""Low-Asymmetry Barrier Reflection Correlation + $3\omega$ Curvature + $2\omega$ + $C_4$ Protocol (`2026-10-22` to `2026-10-23`).

Encapsulates the entire family of low-asymmetry correlation protocols investigated in
`Outputs/Postprocessing/Low_Asymmetry_Correlation_30mK/` and the `2026-10-22` / `2026-10-23`
project logs:
- Step 1–3 (`C_corr`): Low-asymmetry barrier reflection correlation $C_\alpha(a \in [-5, +5]) \ge \theta_{\text{corr}}$
  evaluated with configurable lead consensus (`"both"`, `"either"`, `"R_only"`, `"L_only"`, or
  `"continuous_pool"`).
- Step 4–6 (`C_3w`): Cutter-averaged or $k$-of-$N$ zero-bias curvature $\langle 3\omega \rangle \le -t_{3\omega}$
  with configurable lead consensus (`"both"`, `"either"`, `"continuous_pool"` for Lead-Average).
- Step 7–9 (`C_2w`, optional): Nonlocal $2\omega$ lock-in bound $|\langle 2\omega \rangle| \le t_{2\omega}$
  (can be toggled via `gate_toggles.use_2w_gate: true/false` to benchmark the `2026-10-23`
  finding that $t_{2\omega}=1.0$ acts as an inverse filter in $3\,\mu\text{m}$ nanowires).
- Step 10 (`C_gap`, optional): Closed-system spectral gap or open-system nonlocal transport gap gate.
- Step 11 (`C_3`, optional): Spatial neighborhood consensus $C_3(C_{\text{corr}}, r, t_N)$.
- Step 12 (`C_4`, mandatory): Canonical post-processing island pruning ($2\times 2$ core radius
  filter + connected-component area pruning).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

import numpy as np

from src.Protocol_Elements import DPE, FPE, BaseProtocol, RealizationFeatureCache


class Protocol(BaseProtocol):
    name = "Low_Asymmetry_Correlation_3w_2w_Protocol"
    description = (
        "Low-asymmetry barrier reflection correlation (a in [-5, +5]) combined with "
        "3w zero-bias curvature, optional 2w bound, optional transport/spectral gap, "
        "spatial neighborhood consensus C3, and 2x2 core / area pruning C4."
    )

    default_parameters: Dict[str, Any] = {
        "fpe_corr": {
            "corr_threshold": 0.85,
            "asym_min": -5.0,
            "asym_max": 5.0,
            "lead": "min_LR",
            "mean_subtract": False,
            "normalize_by_first_point": False,
            "norm_floor": 1.0e-12,
        },
        "dpe_corr_lead": {
            "mode": "both",
        },
        "fpe_3w": {
            "curvature_threshold": 20.0,
            "lead": "mean_LR",
            "curvature_source": "thermal_lockin_3w",
            "scale_mode": "physical_dynamic_grid",
            "savgol_window_length": 7,
            "savgol_polyorder": 2,
            "cap_limit": None,
        },
        "dpe_3w_lead": {
            "mode": "continuous_pool",
        },
        "fpe_2w": {
            "max_abs_2w": 1.0,
            "lead": "signed_mean",
        },
        "dpe_2w_lead": {
            "mode": "continuous_pool",
        },
        "fpe_gap": {
            "gap_type": "nonlocal_transport",
            "min_gap_mev": 0.015,
            "max_gap_mev": 0.067,
            "gap_threshold_factor": 0.05,
            "noise_threshold": 1.0e-4,
        },
        "dpe_c3_neighborhood": {
            "enabled": False,
            "radius": 1.0,
            "t_N": 0.55,
            "metric": "chebyshev_grid",
            "include_center": True,
            "require_center_pass": False,
            "boundary_mode": "valid_support_only",
        },
        "gate_toggles": {
            "use_3w_gate": True,
            "use_2w_gate": False,
            "use_gap_gate": False,
        },
        "dpe_c4_post_processor": {
            "enabled": True,
            "use_core_filter": True,
            "core_shape": [2, 2],
            "core_connectivity": 8,
            "core_mode": "component_has_core",
            "use_optics": False,
            "optics_min_samples": 5,
            "optics_max_eps": 3.0,
            "optics_min_cluster_size": 10,
            "optics_xi": 0.05,
            "optics_coordinate_space": "grid_indices",
            "min_island_size": 1,
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

        # 1. Low-asymmetry reflection correlation FPE + Lead Consensus DPE
        m_corr = DPE.evaluate_bipartite_lead_criterion(
            cache,
            FPE.fpe_barrier_reflection_correlation,
            params["fpe_corr"],
            params["dpe_corr_lead"],
            lead_kwarg="lead",
            prefix="step_1_corr",
            steps_out=steps,
        )
        candidate = m_corr.copy()

        # 2. Optional C3 spatial neighborhood consensus on m_corr
        c3_cfg = dict(params["dpe_c3_neighborhood"])
        c3_enabled = bool(c3_cfg.pop("enabled", False))
        if c3_enabled:
            m_c3 = DPE.dpe_radius_neighborhood_consensus(
                m_corr,
                mu_vals=cache.mu_vals,
                vz_vals=cache.vz_vals,
                **c3_cfg,
            )
            steps["step_2_dpe_C3_neighborhood"] = m_c3
            candidate = DPE.dpe_and(candidate, m_c3)

        # 3. 3w zero-bias curvature FPE + Lead Consensus DPE
        toggles = params["gate_toggles"]
        if toggles.get("use_3w_gate", True):
            m_3w = DPE.evaluate_bipartite_lead_criterion(
                cache,
                FPE.fpe_cutter_averaged_curvature,
                params["fpe_3w"],
                params["dpe_3w_lead"],
                lead_kwarg="lead",
                prefix="step_3_curv3w",
                steps_out=steps,
            )
            candidate = DPE.dpe_and(candidate, m_3w)
            steps["step_4_corr_and_3w"] = candidate

        # 4. Optional 2w nonlocal lock-in bound
        if toggles.get("use_2w_gate", False):
            m_2w = DPE.evaluate_bipartite_lead_criterion(
                cache,
                FPE.fpe_nonlocal_2w_bounded,
                params["fpe_2w"],
                params["dpe_2w_lead"],
                lead_kwarg="lead",
                prefix="step_5_lockin2w",
                steps_out=steps,
            )
            candidate = DPE.dpe_and(candidate, m_2w)
            steps["step_6_after_2w"] = candidate

        # 5. Optional open-system nonlocal transport gap gate
        if toggles.get("use_gap_gate", False):
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
                use_thermal_broadened=True,
            )
            steps["step_7_fpe_gap"] = m_gap
            candidate = DPE.dpe_and(candidate, m_gap)

        steps["step_8_pre_C4_candidate"] = candidate

        # 6. Mandatory Terminal DPE C4
        final_mask = DPE.dpe_canonical_post_processor(
            candidate,
            mu_vals=cache.mu_vals,
            vz_vals=cache.vz_vals,
            **params["dpe_c4_post_processor"],
        )
        steps["step_9_dpe_C4_final_mask"] = final_mask
        return steps
