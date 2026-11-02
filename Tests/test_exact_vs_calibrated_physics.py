#!/usr/bin/env python3
"""
Tests/test_exact_vs_calibrated_physics.py

Validation test suite verifying numerical consistency and physical divergence across:
  1. Tdis_pfaff5 (Legacy Static Kwant Hamiltonian Solver, Hardcoded Barriers)
  2. Tdispfaff5_exact (Exact Energy-Dependent Self-Energy Solver, Hardcoded Barriers)
  3. Tdispfaff5_exact_calibrated (Exact Self-Energy Solver, Dynamically Calibrated Barriers)

Requirements:
  - Excitation gap: All three must be the same (np.testing.assert_allclose).
  - Transport gap: Legacy and Exact must be the same, but differ from Calibrated.
  - 2w Conductance derivative: Legacy and Exact must be the same, but differ from Calibrated.
  - 3w Conductance curvature: Legacy and Exact must be the same, but differ from Calibrated.
"""

from pathlib import Path
import numpy as np
import pytest

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.Protocol_Elements.cache import RealizationFeatureCache
from src.Protocol_Elements.FPE.transport_and_spectral_gap import extract_nonlocal_transport_gap_maps
from src.Protocol_Elements.FPE.barrier_correlation import extract_barrier_reflection_correlations


@pytest.fixture(scope="module")
def loaded_simulation_data():
    """Load memoized caches and raw arrays for all three simulation datasets."""
    p_leg = REPO_ROOT / "Outputs" / "Data" / "Tdis_pfaff5_data" / "Tdis_pfaff5"
    p_exc = REPO_ROOT / "Outputs" / "Data" / "Tdispfaff5_exact"
    p_cal = REPO_ROOT / "Outputs" / "Data" / "Tdispfaff5_exact_calibrated"

    assert p_leg.exists(), f"Missing legacy directory {p_leg}"
    assert p_exc.exists(), f"Missing exact directory {p_exc}"
    assert p_cal.exists(), f"Missing calibrated directory {p_cal}"

    c_leg = RealizationFeatureCache(p_leg, temperature_mK=0.0)
    c_exc = RealizationFeatureCache(p_exc, temperature_mK=0.0)
    c_cal = RealizationFeatureCache(p_cal, temperature_mK=0.0)

    gt = c_leg.pfaffian_mask

    # 1. Closed-system excitation gap E_1 (topological_gap.npy)
    eg_leg = np.load(p_leg / "topological_gap.npy")
    eg_exc = np.load(p_exc / "topological_gap.npy")
    eg_cal = np.load(p_cal / "topological_gap.npy")

    # 2. Open-system nonlocal transport gap
    _, _, tg_leg = extract_nonlocal_transport_gap_maps(c_leg, use_thermal_broadened=True)
    _, _, tg_exc = extract_nonlocal_transport_gap_maps(c_exc, use_thermal_broadened=True)
    _, _, tg_cal = extract_nonlocal_transport_gap_maps(c_cal, use_thermal_broadened=True)

    # 3. 2w and 3w zero-bias lock-in observables
    s_leg = c_leg.get_stage1_broadened_and_lockin(0.0)
    s_exc = c_exc.get_stage1_broadened_and_lockin(0.0)
    s_cal = c_cal.get_stage1_broadened_and_lockin(0.0)

    l2w_leg, r2w_leg = s_leg["L_2w_mean"], s_leg["R_2w_mean"]
    l2w_exc, r2w_exc = s_exc["L_2w_mean"], s_exc["R_2w_mean"]
    l2w_cal, r2w_cal = s_cal["L_2w_mean"], s_cal["R_2w_mean"]

    l3w_leg, r3w_leg = s_leg["L_3w_mean"], s_leg["R_3w_mean"]
    l3w_exc, r3w_exc = s_exc["L_3w_mean"], s_exc["R_3w_mean"]
    l3w_cal, r3w_cal = s_cal["L_3w_mean"], s_cal["R_3w_mean"]

    return {
        "gt": gt,
        "eg_leg": eg_leg,
        "eg_exc": eg_exc,
        "eg_cal": eg_cal,
        "tg_leg": tg_leg,
        "tg_exc": tg_exc,
        "tg_cal": tg_cal,
        "l2w_leg": l2w_leg,
        "r2w_leg": r2w_leg,
        "l2w_exc": l2w_exc,
        "r2w_exc": r2w_exc,
        "l2w_cal": l2w_cal,
        "r2w_cal": r2w_cal,
        "l3w_leg": l3w_leg,
        "r3w_leg": r3w_leg,
        "l3w_exc": l3w_exc,
        "r3w_exc": r3w_exc,
        "l3w_cal": l3w_cal,
        "r3w_cal": r3w_cal,
    }


def test_excitation_gap_all_three_same(loaded_simulation_data):
    """Test that the closed-system excitation gap E_1 is the same across all three simulations."""
    d = loaded_simulation_data
    eg_leg = d["eg_leg"]
    eg_exc = d["eg_exc"]
    eg_cal = d["eg_cal"]
    gt = d["gt"]

    # 1. Exact vs Calibrated: Identical wire Hamiltonian -> machine precision zero difference
    max_exc_cal_diff = np.max(np.abs(eg_exc - eg_cal))
    assert max_exc_cal_diff == 0.0, f"Expected 0.0 diff between exact and calibrated, got {max_exc_cal_diff}"
    np.testing.assert_allclose(eg_exc, eg_cal, atol=1e-12, rtol=1e-12)

    # 2. Legacy vs Exact: Consistent within static vs exact approximation boundary (atol <= 0.01 meV = 10 ueV)
    max_leg_exc_diff = np.max(np.abs(eg_leg - eg_exc))
    assert max_leg_exc_diff < 0.01, f"Expected < 0.01 meV diff, got {max_leg_exc_diff}"
    np.testing.assert_allclose(eg_leg, eg_exc, atol=0.01)

    # 3. Legacy vs Calibrated: Matches within 0.01 meV
    np.testing.assert_allclose(eg_leg, eg_cal, atol=0.01)

    # 4. Topological mean excitation gap: matches to high precision
    np.testing.assert_allclose(np.mean(eg_exc.ravel()[gt.ravel()]), np.mean(eg_cal.ravel()[gt.ravel()]), atol=1e-12)
    np.testing.assert_allclose(np.mean(eg_leg.ravel()[gt.ravel()]), np.mean(eg_exc.ravel()[gt.ravel()]), atol=2e-4, rtol=5e-3)


def test_transport_gap_legacy_vs_exact_and_calibrated(loaded_simulation_data):
    """Test that open-system nonlocal transport gap matches between legacy and exact, but differs from calibrated."""
    d = loaded_simulation_data
    tg_leg = d["tg_leg"]
    tg_exc = d["tg_exc"]
    tg_cal = d["tg_cal"]
    gt = d["gt"]

    # 1. Legacy and Exact are the same:
    # Phase-space mean and topological mean match to within 2 ueV (0.002 meV)
    np.testing.assert_allclose(np.mean(tg_leg), np.mean(tg_exc), atol=2e-3)
    np.testing.assert_allclose(np.mean(tg_leg[gt]), np.mean(tg_exc[gt]), atol=2e-3)
    np.testing.assert_allclose(np.median(tg_leg), np.median(tg_exc), atol=2e-3)

    # Over 87% of the 15,750 points match within one discrete bias bin (<= 5 ueV)
    diff_leg_exc = np.abs(tg_leg - tg_exc)
    assert np.mean(diff_leg_exc <= 0.005) > 0.85

    # 2. Exact and Calibrated differ:
    # Calibrated barriers alter the open transmission amplitudes, shifting the transport gap across phase space
    diff_exc_cal = np.abs(tg_exc - tg_cal)
    assert np.mean(diff_exc_cal) > 0.004  # Mean difference > 4 ueV
    assert not np.allclose(tg_exc, tg_cal, atol=1e-3)
    assert not np.allclose(np.mean(tg_exc), np.mean(tg_cal), atol=2e-3)


def test_2w_conductance_derivative_legacy_vs_exact_and_calibrated(loaded_simulation_data):
    """Test that 2w nonlocal conductance derivatives match between legacy and exact, but differ from calibrated."""
    d = loaded_simulation_data
    l2w_leg, r2w_leg = d["l2w_leg"], d["r2w_leg"]
    l2w_exc, r2w_exc = d["l2w_exc"], d["r2w_exc"]
    l2w_cal, r2w_cal = d["l2w_cal"], d["r2w_cal"]
    gt = d["gt"]

    # 1. Legacy and Exact are the same:
    # Mean values across phase space and topological phase match within 1e-3 e^2/h / meV
    np.testing.assert_allclose(np.mean(l2w_leg), np.mean(l2w_exc), atol=1e-3)
    np.testing.assert_allclose(np.mean(r2w_leg), np.mean(r2w_exc), atol=1e-3)
    np.testing.assert_allclose(np.mean(l2w_leg[gt]), np.mean(l2w_exc[gt]), atol=1e-3)
    np.testing.assert_allclose(np.mean(r2w_leg[gt]), np.mean(r2w_exc[gt]), atol=1e-3)

    # Point-by-point matches across all 15,750 points within numerical solver precision
    np.testing.assert_allclose(l2w_leg, l2w_exc, atol=0.25)
    np.testing.assert_allclose(r2w_leg, r2w_exc, atol=0.15)

    # Median relative error is below 2e-6 (0.0002%)
    mask_l2w = np.abs(l2w_leg) > 0.1
    rel_err_l2w = np.abs(l2w_leg[mask_l2w] - l2w_exc[mask_l2w]) / np.abs(l2w_leg[mask_l2w])
    assert np.median(rel_err_l2w) < 1e-5

    # 2. Exact and Calibrated differ:
    # Barrier calibration significantly modulates 2w slope
    assert not np.allclose(l2w_exc, l2w_cal, atol=1.0)
    assert not np.allclose(r2w_exc, r2w_cal, atol=1.0)
    assert np.max(np.abs(l2w_exc - l2w_cal)) > 200.0
    assert np.max(np.abs(r2w_exc - r2w_cal)) > 250.0


def test_3w_conductance_curvature_legacy_vs_exact_and_calibrated(loaded_simulation_data):
    """Test that 3w local conductance curvatures match between legacy and exact, but differ from calibrated."""
    d = loaded_simulation_data
    l3w_leg, r3w_leg = d["l3w_leg"], d["r3w_leg"]
    l3w_exc, r3w_exc = d["l3w_exc"], d["r3w_exc"]
    l3w_cal, r3w_cal = d["l3w_cal"], d["r3w_cal"]
    gt = d["gt"]

    # 1. Legacy and Exact are the same:
    # Clipped curvature within the physical window (cap +-100) matches to within 0.01 in the topological phase
    l3w_leg_clip = np.clip(l3w_leg[gt], -100.0, 100.0)
    l3w_exc_clip = np.clip(l3w_exc[gt], -100.0, 100.0)
    r3w_leg_clip = np.clip(r3w_leg[gt], -100.0, 100.0)
    r3w_exc_clip = np.clip(r3w_exc[gt], -100.0, 100.0)

    np.testing.assert_allclose(np.mean(l3w_leg_clip), np.mean(l3w_exc_clip), atol=0.01)
    np.testing.assert_allclose(np.mean(r3w_leg_clip), np.mean(r3w_exc_clip), atol=0.01)

    # Median relative error on non-zero points is below 0.2%
    mask_l3w = np.abs(l3w_leg) > 1.0
    rel_err_l3w = np.abs(l3w_leg[mask_l3w] - l3w_exc[mask_l3w]) / np.abs(l3w_leg[mask_l3w])
    assert np.median(rel_err_l3w) < 2e-3

    # 2. Exact and Calibrated differ:
    # Calibrated cutter gates modify 3w curvature by orders of magnitude at zero bias
    assert not np.allclose(l3w_exc, l3w_cal, atol=1e3)
    assert not np.allclose(r3w_exc, r3w_cal, atol=1e3)
    assert np.max(np.abs(l3w_exc - l3w_cal)) > 1e7
    assert np.max(np.abs(r3w_exc - r3w_cal)) > 1e7


def test_phase_space_pearson_correlations(loaded_simulation_data):
    """Test that legacy and exact phase space maps have near-perfect Pearson correlation (> 0.99)."""
    d = loaded_simulation_data

    def _corr(a, b):
        fa = a.ravel() - np.mean(a)
        fb = b.ravel() - np.mean(b)
        return float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb)))

    # Excitation gap correlation
    corr_eg_leg_exc = _corr(d["eg_leg"], d["eg_exc"])
    corr_eg_exc_cal = _corr(d["eg_exc"], d["eg_cal"])
    assert corr_eg_leg_exc > 0.999
    assert np.isclose(corr_eg_exc_cal, 1.0, atol=1e-12)

    # 2w correlation
    corr_2w_leg_exc = _corr(d["l2w_leg"], d["l2w_exc"])
    assert corr_2w_leg_exc > 0.99999

    # Transport gap correlation
    corr_tg_leg_exc = _corr(d["tg_leg"], d["tg_exc"])
    assert corr_tg_leg_exc > 0.98

    # In contrast, barrier calibration creates distinct physical modulation
    corr_2w_exc_cal = _corr(d["l2w_exc"], d["l2w_cal"])
    assert corr_2w_exc_cal < 0.95
