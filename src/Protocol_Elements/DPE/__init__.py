r"""Derived Protocol Elements (DPEs) Package.

All DPEs map binary inputs $\{0, 1\}^k \times \Theta \to \{0, 1\}^{N_\mu \times N_{V_z}}$.
"""

from .island_pruning import (
    dpe_canonical_post_processor,
    dpe_island_boundary_gapless_contiguity_filter,
    dpe_island_boundary_zbp_contiguity_filter,
    dpe_optics_subtractive_clustering,
    dpe_subtractive_area_pruning,
    dpe_topological_island_radius_core_filter,
)
from .logical_combinators import (
    dpe_and,
    dpe_k_of_n_cutter_consensus,
    dpe_lead_consensus,
    dpe_not,
    dpe_or,
    dpe_xor,
    evaluate_bipartite_lead_criterion,
)
from .spatial_neighborhood import (
    dpe_1d_zeeman_closing,
    dpe_directional_manifold_closing,
    dpe_hysteresis_propagation,
    dpe_radius_neighborhood_consensus,
)

DPE_REGISTRY = {
    "dpe_and": dpe_and,
    "dpe_or": dpe_or,
    "dpe_not": dpe_not,
    "dpe_xor": dpe_xor,
    "dpe_lead_consensus": dpe_lead_consensus,
    "dpe_k_of_n_cutter_consensus": dpe_k_of_n_cutter_consensus,
    "dpe_radius_neighborhood_consensus": dpe_radius_neighborhood_consensus,
    "dpe_1d_zeeman_closing": dpe_1d_zeeman_closing,
    "dpe_hysteresis_propagation": dpe_hysteresis_propagation,
    "dpe_directional_manifold_closing": dpe_directional_manifold_closing,
    "dpe_subtractive_area_pruning": dpe_subtractive_area_pruning,
    "dpe_topological_island_radius_core_filter": dpe_topological_island_radius_core_filter,
    "dpe_optics_subtractive_clustering": dpe_optics_subtractive_clustering,
    "dpe_island_boundary_gapless_contiguity_filter": dpe_island_boundary_gapless_contiguity_filter,
    "dpe_island_boundary_zbp_contiguity_filter": dpe_island_boundary_zbp_contiguity_filter,
    "dpe_canonical_post_processor": dpe_canonical_post_processor,
}

__all__ = list(DPE_REGISTRY.keys()) + [
    "DPE_REGISTRY",
    "evaluate_bipartite_lead_criterion",
]
