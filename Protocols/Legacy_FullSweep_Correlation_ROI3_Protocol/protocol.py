r"""Legacy Full-Sweep Barrier Correlation (`[0, 60]`), Protocol v2.0 Peak Gates & ROI 3 Protocol.

Covers the historical baseline and hybrid ROI 3 protocols from `2026-10-08` to `2026-10-20`:
- Full-Sweep $[0, 60]$ ($U \in [0, 120]\,\text{meV}$) Dual-Barrier Reflection Correlation
- Optional Protocol v2.0 5-Gate Resonance Peak Filter (`fpe_legacy_peak_gates`)
- Optional TGP Stage 2 `gapped_zbp` intersection (`ROI 3 = TGP Stage 2 & Dual-Barrier Correlation`)
- Optional spatial neighborhood stability $C_3$ (`check_island_stability_v3`)
- Mandatory $C_4$ island pruning.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

import numpy as np

from src.Protocol_Elements import DPE, FPE, BaseProtocol, RealizationFeatureCache


class Protocol(BaseProtocol):
    name = "Legacy_FullSweep_Correlation_ROI3_Protocol"
    description = (
        "Legacy Full-Sweep [0, 60] Dual-Barrier Reflection Correlation, Protocol v2.0 "
        "5-gate resonance peak filter, and TGP Stage 2 intersection (ROI 3)."
    )

    default_parameters: Dict[str, Any] = {
        "fpe_full_sweep_corr": {
            "corr_threshold": 0.90,
            "asym_min": 0.0,
            "asym_max": 60.0,
            "lead": "min_LR",
            "mean_subtract": False,
            "normalize_by_first_point": False,
            "norm_floor": 1.0e-12,
        },
        "dpe_corr_lead": {
            "mode": "both",
        },
        "fpe_legacy_peaks": {
            "enabled": False,
            "require_pos_resonance": True,
            "height_thresh": 0.02,
            "require_lr_energy_agreement": True,
            "peak_diff_tol": 0.010,
            "require_neg_companion": True,
            "min_energy_mev": 0.0,
            "max_energy_mev": 0.030,
            "require_eh_symmetry": True,
            "sym_tol": 0.008,
            "cutter_idx": 0,
            "use_thermal_broadened": True,
        },
        "roi3_tgp_intersection": {
            "enabled": False,
            "zbp_curvature_threshold": 0.0,
            "min_gap_mev": 0.010,
            "gap_threshold_factor": 0.05,
            "min_cutter_fraction": 0.50,
            "boundary_pct_threshold": 0.60,
            "min_tgp_island_size": 7,
        },
        "dpe_c3_stability": {
            "enabled": False,
            "radius": 1.0,
            "t_N": 0.50,
            "metric": "chebyshev_grid",
            "include_center": True,
            "require_center_pass": True,
            "boundary_mode": "valid_support_only",
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
            "min_island_size": 5,
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

        m_corr = DPE.evaluate_bipartite_lead_criterion(
            cache,
            FPE.fpe_barrier_reflection_correlation,
            params["fpe_full_sweep_corr"],
            params["dpe_corr_lead"],
            lead_kwarg="lead",
            prefix="step_1_fullsweep_corr",
            steps_out=steps,
        )
        cand = m_corr.copy()

        pk_cfg = dict(params["fpe_legacy_peaks"])
        if bool(pk_cfg.pop("enabled", False)):
            m_pk = FPE.fpe_legacy_peak_gates(cache, **pk_cfg)
            steps["step_2_fpe_legacy_peaks"] = m_pk
            cand = DPE.dpe_and(cand, m_pk)

        roi3_cfg = params["roi3_tgp_intersection"]
        if bool(roi3_cfg.get("enabled", False)):
            canon = FPE.compute_tgp_canonical_dataset(
                cache,
                derivative_threshold=float(roi3_cfg.get("zbp_curvature_threshold", 100.0)),
                min_gap_mev=float(roi3_cfg.get("min_gap_mev", 0.010)),
                gap_threshold_factor=float(roi3_cfg.get("gap_threshold_factor", 0.05)),
                min_cluster_size=int(roi3_cfg.get("min_tgp_island_size", 7)),
                boundary_pct_threshold=float(roi3_cfg.get("boundary_pct_threshold", 0.60)),
                min_cutter_fraction=float(roi3_cfg.get("min_cutter_fraction", 0.50)),
            )
            tgp_roi2 = canon["roi2_2d"]
            steps["step_3_dpe_tgp_roi2"] = tgp_roi2
            cand = DPE.dpe_and(cand, tgp_roi2)

        c3_cfg = dict(params["dpe_c3_stability"])
        if bool(c3_cfg.pop("enabled", False)):
            cand = DPE.dpe_radius_neighborhood_consensus(
                cand,
                mu_vals=cache.mu_vals,
                vz_vals=cache.vz_vals,
                **c3_cfg,
            )
            steps["step_4_dpe_c3_stability"] = cand

        steps["step_5_pre_C4_candidate"] = cand

        final_mask = DPE.dpe_canonical_post_processor(
            cand,
            mu_vals=cache.mu_vals,
            vz_vals=cache.vz_vals,
            **params["dpe_c4_post_processor"],
        )
        steps["step_6_dpe_C4_final_mask"] = final_mask
        return steps
