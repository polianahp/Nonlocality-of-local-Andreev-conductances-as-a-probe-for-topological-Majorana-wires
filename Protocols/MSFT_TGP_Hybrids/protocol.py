r"""MSFT_TGP_Hybrids Protocol: Microsoft Azure Quantum TGP Augmented with Non-Local Correlators & Symmetry Principles.

Explores systematic hybridizations of the Microsoft Azure Quantum TGP Stage 2 baseline
with:
1. Gapless boundary condition: standard (boundary_pct=0.60) vs removed entirely (boundary_pct=0.0)
2. Low-asymmetry barrier reflection correlation: threshold in {0.70, 0.80, 0.90} over a in [-5, +5]
3. Lead Conductance Symmetry Balance Ratio (Dourado et al. Fig. 13): S_sym <= s_max (e.g. 0.20, 0.30)
4. Normalized Asymmetric Conductance Deviation (NACD / ACD): sqrt(a_L * a_R) >= tau (e.g. 0.01, 0.05, 0.08)
5. Cross-swept conductance correlation: Corr(G_LL(U_R), G_RR(U_L)) >= theta at low and higher asymmetry.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

import numpy as np

from src.Protocol_Elements import DPE, FPE, BaseProtocol, RealizationFeatureCache


class Protocol(BaseProtocol):
    name = "MSFT_TGP_Hybrids"
    description = (
        "Microsoft Azure Quantum TGP Stage 2 baseline augmented with low-asymmetry "
        "barrier reflection correlation, lead conductance symmetry balance, normalized "
        "asymmetric conductance deviation (ACD), and cross-swept conjugate correlation."
    )

    default_parameters: Dict[str, Any] = {
        "tgp_base": {
            "derivative_threshold": 100.0,
            "zbp_probability_threshold": 0.70,
            "bias_window": 10e-3,
            "polyorder": 2,
            "min_gap_mev": 0.010,
            "max_gap_mev": 0.067,
            "gap_threshold_factor": 0.05,
            "noise_threshold": 1.0e-4,
            "min_cutter_fraction": 0.50,
            "boundary_pct_threshold": 0.60,  # 0.60 = standard, 0.0 = removed entirely
            "variance": 3,
            "min_cluster_size": 7,
        },
        "fpe_low_asym_corr": {
            "enabled": False,
            "corr_threshold": 0.80,
            "asym_min": -5.0,
            "asym_max": 5.0,
            "lead": "R",
            "mean_subtract": False,
            "normalize_by_first_point": False,
        },
        "fpe_lead_symmetry_balance": {
            "enabled": False,
            "s_max": 0.20,
            "eps_sym": 1e-4,
            "sym_asym_ratio": 0.0,
        },
        "fpe_acd": {
            "enabled": False,
            "tau": 0.05,
            "floor_val": 1e-3,
            "sym_asym_ratio": 0.0,
            "pinch_asym_ratio": 60.0,
            "use_signed_diff": False,
        },
        "fpe_cross_swept_corr": {
            "enabled": False,
            "corr_threshold": 0.80,
            "asym_min": -5.0,
            "asym_max": 5.0,
            "mean_subtract": False,
            "normalize_by_first_point": False,
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
            "min_island_size": 7,
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
        tgp_cfg = params["tgp_base"]

        # 1. Base Microsoft TGP Stage 2 evaluation (ROI 2)
        canonical = FPE.compute_tgp_canonical_dataset(
            cache,
            derivative_threshold=float(tgp_cfg.get("derivative_threshold", 100.0)),
            zbp_probability_threshold=float(
                tgp_cfg.get("zbp_probability_threshold", 0.70)
            ),
            bias_window=float(tgp_cfg.get("bias_window", 10e-3)),
            polyorder=int(tgp_cfg.get("polyorder", 2)),
            min_gap_mev=float(tgp_cfg.get("min_gap_mev", 0.010)),
            max_gap_mev=float(tgp_cfg.get("max_gap_mev", 0.067)),
            gap_threshold_factor=float(tgp_cfg.get("gap_threshold_factor", 0.05)),
            noise_threshold=float(tgp_cfg.get("noise_threshold", 1e-4)),
            min_cluster_size=int(tgp_cfg.get("min_cluster_size", 7)),
            variance=int(tgp_cfg.get("variance", 3)),
            boundary_pct_threshold=float(tgp_cfg.get("boundary_pct_threshold", 0.60)),
            min_cutter_fraction=float(tgp_cfg.get("min_cutter_fraction", 0.50)),
        )

        steps["step_1_fpe_tgp_zbp_3d"] = canonical["zbp_3d"]
        steps["step_2_fpe_tgp_gapped_zbp_3d"] = canonical["gapped_zbp_3d"]

        # Base candidate: ROI 2 from cluster_and_score
        current_mask = canonical["roi2_2d"].copy()
        steps["step_3_tgp_stage2_candidate"] = current_mask.copy()

        # 2. Optional Condition: Low-Asymmetry Barrier Reflection Correlation
        corr_cfg = params.get("fpe_low_asym_corr", {})
        if corr_cfg.get("enabled", False):
            kwargs = {k: v for k, v in corr_cfg.items() if k != "enabled"}
            corr_mask = FPE.fpe_barrier_reflection_correlation(cache, **kwargs)
            current_mask = current_mask & corr_mask
            steps["step_4_fpe_low_asym_corr"] = current_mask.copy()

        # 3. Optional Condition: Lead Conductance Symmetry Balance Ratio
        sym_cfg = params.get("fpe_lead_symmetry_balance", {})
        if sym_cfg.get("enabled", False):
            kwargs = {k: v for k, v in sym_cfg.items() if k != "enabled"}
            sym_mask = FPE.fpe_lead_symmetry_balance(cache, **kwargs)
            current_mask = current_mask & sym_mask
            steps["step_5_fpe_lead_symmetry_balance"] = current_mask.copy()

        # 4. Optional Condition: Normalized Asymmetric Conductance Deviation (ACD)
        acd_cfg = params.get("fpe_acd", {})
        if acd_cfg.get("enabled", False):
            kwargs = {k: v for k, v in acd_cfg.items() if k != "enabled"}
            acd_mask = FPE.fpe_acd_geometric_mean(cache, **kwargs)
            current_mask = current_mask & acd_mask
            steps["step_6_fpe_acd_geometric_mean"] = current_mask.copy()

        # 5. Optional Condition: Cross-Swept Conductance Correlation
        cross_cfg = params.get("fpe_cross_swept_corr", {})
        if cross_cfg.get("enabled", False):
            kwargs = {k: v for k, v in cross_cfg.items() if k != "enabled"}
            cross_mask = FPE.fpe_cross_barrier_conjugate_correlation(cache, **kwargs)
            current_mask = current_mask & cross_mask
            steps["step_7_fpe_cross_swept_corr"] = current_mask.copy()

        # 6. Canonical Post-Processor (C4)
        c4_cfg = params.get("dpe_c4_post_processor", {})
        final_mask = DPE.dpe_canonical_post_processor(
            current_mask,
            mu_vals=cache.mu_vals,
            vz_vals=cache.vz_vals,
            **c4_cfg,
        )
        steps["step_8_dpe_C4_final_mask"] = final_mask
        return steps
