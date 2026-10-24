r"""Fundamental Protocol Elements (FPEs) Package.

All FPEs map continuous simulated data (`RealizationFeatureCache`) + free parameters $\Theta$
to a binary phase-space output $\{0, 1\}^{N_\mu \times N_{V_z}}$ (or Rank-3
$\{0, 1\}^{N_{\text{cutters}} \times N_\mu \times N_{V_z}}$ for per-cutter FPEs).
"""

from .barrier_asymmetry import (
    extract_lead_acd_maps,
    fpe_acd_geometric_mean,
    fpe_barrier_max_at_symmetric,
    fpe_barrier_monotonicity,
    fpe_lead_symmetry_balance,
    fpe_pinch_off_conductance_gate,
    fpe_single_lead_acd,
)
from .barrier_correlation import (
    extract_barrier_reflection_correlations,
    extract_cross_barrier_correlations,
    fpe_barrier_reflection_correlation,
    fpe_cross_barrier_conjugate_correlation,
    fpe_local_vz_oscillation_correlation,
)
from .envelope_modulation import (
    extract_ac_rms_fields,
    extract_lqmp_fields,
    fpe_ac_rms_energy,
    fpe_ac_rms_inflection,
    fpe_lqmp_detrended_power,
    fpe_lqmp_quadrature_coherence,
)
from .lockin_curvature_2w_3w import (
    extract_zero_bias_curvature_3d,
    fpe_cutter_averaged_curvature,
    fpe_nonlocal_2w_bounded,
    fpe_zero_bias_curvature_single_cutter,
)
from .tgp_baseline_elements import (
    compute_tgp_canonical_dataset,
    fpe_tgp_gapped_zbp_per_cutter,
    fpe_tgp_zbp_per_cutter,
)
from .transport_and_spectral_gap import (
    extract_nonlocal_transport_gap_maps,
    fpe_legacy_peak_gates,
    fpe_nonlocal_conductance_transport_gap,
    fpe_quantum_dot_gamma_squared,
    fpe_spectral_excitation_gap,
)

FPE_REGISTRY = {
    # 1. Barrier Asymmetry & Symmetry Balance
    "fpe_acd_geometric_mean": fpe_acd_geometric_mean,
    "fpe_single_lead_acd": fpe_single_lead_acd,
    "fpe_lead_symmetry_balance": fpe_lead_symmetry_balance,
    "fpe_pinch_off_conductance_gate": fpe_pinch_off_conductance_gate,
    "fpe_barrier_monotonicity": fpe_barrier_monotonicity,
    "fpe_barrier_max_at_symmetric": fpe_barrier_max_at_symmetric,
    # 2. Barrier Reflection & Local Vz Correlations
    "fpe_barrier_reflection_correlation": fpe_barrier_reflection_correlation,
    "fpe_cross_barrier_conjugate_correlation": fpe_cross_barrier_conjugate_correlation,
    "fpe_local_vz_oscillation_correlation": fpe_local_vz_oscillation_correlation,
    # 3. 1D Zeeman Foliation & Quadrature Envelopes
    "fpe_ac_rms_energy": fpe_ac_rms_energy,
    "fpe_ac_rms_inflection": fpe_ac_rms_inflection,
    "fpe_lqmp_detrended_power": fpe_lqmp_detrended_power,
    "fpe_lqmp_quadrature_coherence": fpe_lqmp_quadrature_coherence,
    # 4. Lock-In 3w Curvature & 2w Observables
    "fpe_zero_bias_curvature_single_cutter": fpe_zero_bias_curvature_single_cutter,
    "fpe_cutter_averaged_curvature": fpe_cutter_averaged_curvature,
    "fpe_nonlocal_2w_bounded": fpe_nonlocal_2w_bounded,
    # 5. Spectral & Nonlocal Transport Gap + Legacy Peak Gates
    "fpe_spectral_excitation_gap": fpe_spectral_excitation_gap,
    "fpe_nonlocal_conductance_transport_gap": fpe_nonlocal_conductance_transport_gap,
    "fpe_legacy_peak_gates": fpe_legacy_peak_gates,
    "fpe_quantum_dot_gamma_squared": fpe_quantum_dot_gamma_squared,
    # 6. TGP Baseline Primitives
    "fpe_tgp_zbp_per_cutter": fpe_tgp_zbp_per_cutter,
    "fpe_tgp_gapped_zbp_per_cutter": fpe_tgp_gapped_zbp_per_cutter,
}

__all__ = list(FPE_REGISTRY.keys()) + [
    "FPE_REGISTRY",
    "extract_lead_acd_maps",
    "extract_barrier_reflection_correlations",
    "extract_ac_rms_fields",
    "extract_lqmp_fields",
    "extract_zero_bias_curvature_3d",
    "extract_nonlocal_transport_gap_maps",
    "compute_tgp_canonical_dataset",
]
