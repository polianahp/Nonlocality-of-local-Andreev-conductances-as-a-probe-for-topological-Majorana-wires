r"""Microsoft Azure Quantum TGP Stage 1 & Stage 2 Baseline Protocol in FPE/DPE Algebra.

Expresses Microsoft's Topological Gap Protocol (`azure-quantum-tgp`) as a composition of
canonical FPEs and DPEs:
- Stage 1 (`stage_mode == "stage1_zbp"`):
  1. `fpe_tgp_zbp_per_cutter` (dual-lead zero-bias curvature derivative threshold 100.0, probability 0.70)
  2. `dpe_k_of_n_cutter_consensus` (`min_fraction = 0.70`, i.e., $\ge 4/5$ cutters)
  3. `dpe_canonical_post_processor` ($C_4$, `min_island_size = 7`)
- Stage 2 (`stage_mode == "stage2_gapped_boundary"`):
  1. `fpe_tgp_zbp_per_cutter` (Rank-3 ZBP tensor)
  2. `fpe_tgp_gapped_zbp_per_cutter` (Rank-3 Gapped ZBP tensor with 10 ueV lower / 67 ueV upper bound)
  3. `dpe_k_of_n_cutter_consensus` (`min_fraction = 0.50`)
  4. `dpe_island_boundary_gapless_contiguity_filter` / `tgp.two.cluster_and_score` (ROI 2)
  5. `dpe_canonical_post_processor` ($C_4$, `min_island_size = 7`).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

import numpy as np

from src.Protocol_Elements import DPE, FPE, BaseProtocol, RealizationFeatureCache


class Protocol(BaseProtocol):
    name = "TGP_Baseline_Protocol"
    description = (
        "Microsoft Azure Quantum TGP Stage 1 (ZBP cutter consensus) and Stage 2 "
        "(gapped ZBP + gapless boundary contiguity) expressed in FPE/DPE form."
    )

    default_parameters: Dict[str, Any] = {
        "stage": {
            "mode": "stage2_gapped_boundary",  # "stage1_zbp" or "stage2_gapped_boundary"
        },
        "fpe_tgp": {
            "derivative_threshold": 100.0,
            "zbp_probability_threshold": 0.70,
            "bias_window": 10e-3,
            "polyorder": 2,
            "min_gap_mev": 0.010,
            "max_gap_mev": 0.067,
            "gap_threshold_factor": 0.05,
            "noise_threshold": 1.0e-4,
            "use_thermal_broadened": True,
        },
        "dpe_cutter_consensus": {
            "k_min": None,
            "min_fraction": 0.50,
        },
        "dpe_boundary_scorer": {
            "variance": 3,
            "boundary_pct_threshold": 0.60,
            "min_island_size": 7,
            "connectivity": 8,
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
        tgp_cfg = params["fpe_tgp"]
        bnd_cfg = params.get("dpe_boundary_scorer", {})
        cc_cfg = params.get("dpe_cutter_consensus", {})

        s_mode = params["stage"]["mode"]

        # Run canonical Microsoft TGP Stage 1 / Stage 2 pipeline on cache
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
            min_cluster_size=int(bnd_cfg.get("min_island_size", 7)),
            variance=int(bnd_cfg.get("variance", 3)),
            boundary_pct_threshold=float(bnd_cfg.get("boundary_pct_threshold", 0.60)),
            min_cutter_fraction=float(cc_cfg.get("min_fraction", 0.50)),
        )

        zbp_3d = canonical["zbp_3d"]
        steps["step_1_fpe_tgp_zbp_3d"] = zbp_3d

        if s_mode == "stage1_zbp":
            cand = DPE.dpe_k_of_n_cutter_consensus(zbp_3d, **cc_cfg)
            steps["step_2_dpe_stage1_cutter_consensus"] = cand
        elif s_mode == "stage2_gapped_boundary":
            gapped_zbp_3d = canonical["gapped_zbp_3d"]
            steps["step_2_fpe_tgp_gapped_zbp_3d"] = gapped_zbp_3d
            gapped_2d = DPE.dpe_k_of_n_cutter_consensus(gapped_zbp_3d, **cc_cfg)
            steps["step_3_dpe_gapped_cutter_consensus"] = gapped_2d
            cand = canonical["roi2_2d"]
            steps["step_4_dpe_boundary_contiguity"] = cand
        else:
            raise ValueError(f"Unknown stage.mode '{s_mode}'.")

        final_mask = DPE.dpe_canonical_post_processor(
            cand,
            mu_vals=cache.mu_vals,
            vz_vals=cache.vz_vals,
            **params["dpe_c4_post_processor"],
        )
        steps["step_5_dpe_C4_final_mask"] = final_mask
        return steps
