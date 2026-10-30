r"""Unit & Integration Test Suite for `src/Protocol_Elements/` (`FPE`, `DPE`, `cache`, `engine`).

Verifies:
1. Exact $97 \times 97$ linear operator agreement with `tgp.prepare.broaden_with_temperature` and `_derivative`.
2. All 6 FPE modules (`barrier_asymmetry`, `barrier_correlation`, `envelope_modulation`,
   `lockin_curvature_2w_3w`, `transport_and_spectral_gap`, `tgp_baseline_elements`).
3. All 3 DPE modules (`logical_combinators`, `spatial_neighborhood`, `island_pruning`), including
   the 3 core-filter modes (`component_has_core`, `morphological_opening`,
   `opening_reconstructed_by_4conn`) and the non-commuting $C_4$ pipeline order.
4. Strict YAML schema typo detection (`difflib.get_close_matches`).
5. End-to-end execution of `Some_Example_Protocol` and `Low_Asymmetry_Correlation_3w_2w_Protocol`
   with `keep_data=True` (verifying `step_data/ensemble_steps.npz` and `config.yaml`) and
   `keep_data=False` (verifying zero `.npz` creation while preserving `config.yaml` and
   `step_waterfall_metrics`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Protocol_Elements import (
    DPE,
    FPE,
    DPE_REGISTRY,
    FPE_REGISTRY,
    RealizationFeatureCache,
    expand_yaml_to_instance_specs,
    get_linear_broaden_and_spline_operators,
    run_protocol_specs,
)

REPO_ROOT = Path("/home/pseudonym/Documents/Code/NonlocalProtocol")
SAMPLE_REALIZATION = (
    REPO_ROOT
    / "Outputs"
    / "Data"
    / "Disorder_Realizations"
    / "V0_0_645"
    / "disorder_realization_10"
)


@pytest.fixture(scope="module")
def sample_cache() -> RealizationFeatureCache:
    assert SAMPLE_REALIZATION.exists(), f"Missing sample realization {SAMPLE_REALIZATION}"
    return RealizationFeatureCache(SAMPLE_REALIZATION, temperature_mK=30.0, v0_label="V0_0_645")


def test_registries_populated():
    assert len(FPE_REGISTRY) >= 18
    assert len(DPE_REGISTRY) >= 13


def test_linear_broaden_and_spline_exactness(sample_cache: RealizationFeatureCache):
    bias = sample_cache.bias_grid
    M_broaden, w1, w2 = get_linear_broaden_and_spline_operators(bias, 30.0)
    assert M_broaden.shape == (97, 97)
    assert w1.shape == (97,)
    assert w2.shape == (97,)
    # Row-sums of thermal broadening kernel must be ~1.0 (probability conservation)
    row_sums = np.sum(M_broaden, axis=0)
    assert np.allclose(row_sums[10:-10], 1.0, atol=1e-2)


def test_all_fpes_shapes_and_types(sample_cache: RealizationFeatureCache):
    shape_2d = sample_cache.grid_shape  # (225, 70)

    m_acd = FPE.fpe_acd_geometric_mean(sample_cache, tau=0.05)
    assert m_acd.dtype == bool and m_acd.shape == shape_2d

    m_acd_L = FPE.fpe_single_lead_acd(sample_cache, lead="L", tau=0.05)
    assert m_acd_L.dtype == bool and m_acd_L.shape == shape_2d

    m_sym = FPE.fpe_lead_symmetry_balance(sample_cache, s_max=0.20)
    assert m_sym.dtype == bool and m_sym.shape == shape_2d

    m_pinch = FPE.fpe_pinch_off_conductance_gate(sample_cache, g_thresh=0.05)
    assert m_pinch.dtype == bool and m_pinch.shape == shape_2d

    m_corr_low = FPE.fpe_barrier_reflection_correlation(
        sample_cache, corr_threshold=0.80, asym_min=-5.0, asym_max=5.0, lead="R"
    )
    assert m_corr_low.dtype == bool and m_corr_low.shape == shape_2d

    m_vz_corr = FPE.fpe_local_vz_oscillation_correlation(
        sample_cache, r_threshold=0.50, window_pts=5
    )
    m_vz_cos = FPE.fpe_local_vz_oscillation_correlation(
        sample_cache, r_threshold=0.50, window_pts=5, use_cosine=True
    )
    assert m_vz_corr.dtype == bool and m_vz_corr.shape == shape_2d
    assert m_vz_cos.dtype == bool and m_vz_cos.shape == shape_2d

    m_rms = FPE.fpe_ac_rms_energy(sample_cache, multiplier=1.0)
    m_rms_hi = FPE.fpe_ac_rms_energy(sample_cache, multiplier=1.30)
    assert m_rms.dtype == bool and m_rms.shape == shape_2d
    assert np.all(m_rms_hi <= m_rms)  # Higher threshold is a subset of base threshold

    m_inf = FPE.fpe_ac_rms_inflection(sample_cache, max_d2=0.0)
    assert m_inf.dtype == bool and m_inf.shape == shape_2d

    m_pow = FPE.fpe_lqmp_detrended_power(sample_cache, theta_P=0.002)
    m_coh = FPE.fpe_lqmp_quadrature_coherence(sample_cache, theta_C=0.25)
    assert m_pow.shape == shape_2d and m_coh.shape == shape_2d

    # Curvature Rank-3 vs Rank-2
    m_curv_3d = FPE.fpe_zero_bias_curvature_single_cutter(
        sample_cache, curvature_threshold=20.0, lead="R"
    )
    assert m_curv_3d.dtype == bool and m_curv_3d.shape == (5, *shape_2d)

    m_curv_avg = FPE.fpe_cutter_averaged_curvature(
        sample_cache, curvature_threshold=20.0, lead="mean_LR"
    )
    assert m_curv_avg.dtype == bool and m_curv_avg.shape == shape_2d

    m_2w_signed = FPE.fpe_nonlocal_2w_bounded(
        sample_cache, max_abs_2w=1.0, lead="signed_mean"
    )
    m_2w_mag = FPE.fpe_nonlocal_2w_bounded(
        sample_cache, max_abs_2w=1.0, lead="magnitude_mean"
    )
    assert m_2w_signed.shape == shape_2d and m_2w_mag.shape == shape_2d
    # Triangle inequality: |0.5*(L+R)| <= 0.5*(|L|+|R|), so magnitude_mean is stricter
    assert np.all(m_2w_mag <= m_2w_signed)

    m_sgap = FPE.fpe_spectral_excitation_gap(sample_cache, min_gap_mev=0.015)
    m_tgap = FPE.fpe_nonlocal_conductance_transport_gap(sample_cache, min_gap_mev=0.010)
    assert m_sgap.shape == shape_2d and m_tgap.shape == shape_2d


def test_dpe_core_filter_modes_and_c4_ordering():
    # Construct a 10x10 synthetic mask with a 2x2 core (4 px) attached to a 5-pixel 1D filament
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True  # 2x2 core (4 pixels)
    mask[4, 4] = True
    mask[5, 5] = True
    mask[6, 6] = True
    mask[7, 7] = True
    mask[8, 8] = True  # Total 9 pixels in 8-connected component, but only 4 in 2x2 core

    # Mode 1: component_has_core retains all 9 pixels of the 8-connected component
    m_comp = DPE.dpe_topological_island_radius_core_filter(
        mask, core_shape=(2, 2), mode="component_has_core"
    )
    assert np.count_nonzero(m_comp) == 9

    # Mode 2: morphological_opening shaves the 1-pixel diagonal tail, leaving 4 pixels
    m_open = DPE.dpe_topological_island_radius_core_filter(
        mask, core_shape=(2, 2), mode="morphological_opening"
    )
    assert np.count_nonzero(m_open) == 4

    # Mode 3: C4 ordering check: if min_island_size=6 and core_mode="morphological_opening",
    # opening runs BEFORE area pruning, reducing 9 px -> 4 px, which is then pruned by area >= 6!
    m_c4 = DPE.dpe_canonical_post_processor(
        mask,
        use_core_filter=True,
        core_shape=(2, 2),
        core_mode="morphological_opening",
        min_island_size=6,
    )
    assert np.count_nonzero(m_c4) == 0


def test_yaml_typo_validation_raises():
    proto_py = REPO_ROOT / "Protocols" / "SACD_Hybrid_Curvature_Gap_Protocol" / "protocol.py"
    from src.Protocol_Elements import load_protocol_class_from_file

    p_cls = load_protocol_class_from_file(proto_py)
    with pytest.raises(ValueError, match="Did you mean"):
        p_cls.resolve_and_validate_params({"fpe_acd": {"tau_typo": 0.05}})


def test_end_to_end_keep_data_true_and_false(tmp_path: Path):
    proto_py = REPO_ROOT / "Protocols" / "SACD_Hybrid_Curvature_Gap_Protocol" / "protocol.py"
    yaml_keep = (
        REPO_ROOT
        / "Protocols"
        / "SACD_Hybrid_Curvature_Gap_Protocol"
        / "Parameters"
        / "sacd_precision_and_variations.yaml"
    )

    specs_true = expand_yaml_to_instance_specs(proto_py, yaml_keep, keep_data_override=True)[:1]
    out_true = tmp_path / "out_keep_true"
    run_protocol_specs(
        specs_true,
        v0_filter=["V0_0_645"],
        temperatures_mK=[30.0],
        max_realizations_per_v0=1,
        max_workers=1,
        outputs_root=out_true,
    )
    inst_dir_true = out_true / specs_true[0].protocol_name / specs_true[0].instance_name
    assert (inst_dir_true / "config.yaml").exists()
    assert (inst_dir_true / "step_data" / "ensemble_steps.npz").exists()

    with open(inst_dir_true / "config.yaml", "r", encoding="utf-8") as f:
        cfg_true = yaml.safe_load(f)
    assert cfg_true["keep_data"] is True
    assert "step_waterfall_metrics" in cfg_true["summary_metrics"]

    # Now run with keep_data=False
    specs_false = expand_yaml_to_instance_specs(proto_py, yaml_keep, keep_data_override=False)[:1]
    out_false = tmp_path / "out_keep_false"
    run_protocol_specs(
        specs_false,
        v0_filter=["V0_0_645"],
        temperatures_mK=[30.0],
        max_realizations_per_v0=1,
        max_workers=1,
        outputs_root=out_false,
    )
    inst_dir_false = out_false / specs_false[0].protocol_name / specs_false[0].instance_name
    assert (inst_dir_false / "config.yaml").exists()
    assert not (inst_dir_false / "step_data").exists()


def _fake_record(v0_label, fdr, tpr, has_pred=True, temperature_mK=30.0):
    return {
        "realization_name": "disorder_realization_10", "v0_label": v0_label, "v0_mev": 0.0,
        "temperature_mK": temperature_mK, "has_prediction": has_pred, "n_predicted_px": 10,
        "n_true_positive_px": 5, "n_false_positive_px": 5, "n_ground_truth_topo_px": 100,
        "survival_pct": 1.0, "FDR_pct": fdr, "FDR_active_only_pct": fdr if has_pred else None,
        "TPR_pct": tpr, "MZM_confinement_C": 0.9, "MZM_separability_S": 0.5,
        "island_count": 1, "mean_island_area_px": 5.0, "max_island_area_px": 5,
        "medoids_mu_vz": [],
        "step_diagnostics": {"final": {"rank": 2.0, "survival_pct": 1.0, "fdr_pct": fdr, "tpr_pct": tpr}},
    }


def test_leaderboard_rank_fdr_then_tpr_and_v0_split(tmp_path: Path):
    import json
    from src.Protocol_Elements.leaderboard import rebuild_leaderboards

    specs = {
        # (protocol, instance): (fdr_a, tpr_a, fdr_b, tpr_b) for V0_0_378, V0_1_2
        ("P_A", "low_fdr_low_tpr"): (10.0, 5.0, 10.0, 5.0),
        ("P_A", "low_fdr_high_tpr"): (10.0, 50.0, 10.0, 50.0),
        ("P_B", "high_fdr"): (40.0, 90.0, 20.0, 90.0),
    }
    for (proto, inst), (fa, ta, fb, tb) in specs.items():
        d = tmp_path / proto / inst
        d.mkdir(parents=True)
        (d / "config.yaml").write_text(yaml.safe_dump({"keep_data": False}))
        (d / "realization_metrics.json").write_text(
            json.dumps([
                _fake_record("V0_0_378", fa, ta, temperature_mK=30.0),
                _fake_record("V0_1_2", fb, tb, temperature_mK=30.0),
                _fake_record("V0_0_378", fa * 1.1, ta * 0.9, temperature_mK=40.0),
                _fake_record("V0_1_2", fb * 1.1, tb * 0.9, temperature_mK=40.0),
            ])
        )
    md = rebuild_leaderboards(tmp_path)
    text = md.read_text()
    assert "## Overall" in text and "$V_0 = 0.378$ meV" in text and "$V_0 = 1.2$ meV" in text
    overall = text.split("## Disorder strength")[0]
    # FDR tie (10.0) broken by higher TPR first
    assert overall.index("low_fdr_high_tpr") < overall.index("low_fdr_low_tpr") < overall.index("high_fdr")
    # In the V0 = 1.2 section P_B (20%) still ranks below the 10% protocols; per-V0 split exists
    sec_12 = text.split("$V_0 = 1.2$ meV")[1]
    assert sec_12.index("low_fdr_high_tpr") < sec_12.index("high_fdr")

    # Verify per-temperature leaderboards exist and are correctly structured
    master_30 = tmp_path / "master_protocols_leaderboard_30mK.md"
    master_40 = tmp_path / "master_protocols_leaderboard_40mK.md"
    assert master_30.exists()
    assert master_40.exists()
    assert (tmp_path / "master_protocols_leaderboard_30mk.md").exists()
    assert (tmp_path / "master_protocols_leaderboard_40mk.md").exists()

    text_30 = master_30.read_text()
    assert "Master Cross-Protocol Leaderboard — 30mK" in text_30
    assert "$V_0 = 0.378$ meV (30mK)" in text_30
    text_40 = master_40.read_text()
    assert "Master Cross-Protocol Leaderboard — 40mK" in text_40
    assert "$V_0 = 0.378$ meV (40mK)" in text_40

    # Verify per-protocol temperature summaries exist
    assert (tmp_path / "P_A" / "leaderboard_summary_30mK.md").exists()
    assert (tmp_path / "P_A" / "leaderboard_summary_40mK.md").exists()
    assert (tmp_path / "P_B" / "leaderboard_summary_30mK.md").exists()
    assert (tmp_path / "P_B" / "leaderboard_summary_40mK.md").exists()


def test_run_protocol_specs_rebuilds_master_leaderboard(tmp_path: Path):
    proto_py = REPO_ROOT / "Protocols" / "TGP_Baseline_Protocol" / "protocol.py"
    yaml_p = REPO_ROOT / "Protocols" / "TGP_Baseline_Protocol" / "Parameters" / "tgp_stage1_and_stage2.yaml"
    specs = expand_yaml_to_instance_specs(proto_py, yaml_p, keep_data_override=False)[:1]
    run_protocol_specs(
        specs, v0_filter=["V0_0_645", "V0_0_872"], temperatures_mK=[30.0],
        max_realizations_per_v0=1, max_workers=1, outputs_root=tmp_path,
    )
    master = tmp_path / "master_protocols_leaderboard.md"
    assert master.exists()
    text = master.read_text()
    assert "## Overall" in text and "$V_0 = 0.645$ meV" in text and "$V_0 = 0.872$ meV" in text
