import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
import helpers as hp


def test_antisymmetric_nonlocal_part_exactness():
    """Verify that antisymmetric_nonlocal_part exactly strips even components and preserves odd components."""
    bias = np.linspace(-1.0, 1.0, 101)
    
    # 1. Even function: should yield 0
    g_even = np.cos(bias) + 5.0
    g_even_antisym = hp.antisymmetric_nonlocal_part(g_even)
    assert np.allclose(g_even_antisym, 0.0, atol=1e-15), "Even function must anti-symmetrize to zero."
    
    # 2. Odd function: should be preserved exactly
    g_odd = bias**3
    g_odd_antisym = hp.antisymmetric_nonlocal_part(g_odd)
    assert np.allclose(g_odd_antisym, g_odd, atol=1e-15), "Odd function must be preserved exactly under anti-symmetrization."
    
    # 3. Mixed function with asymmetric CAR-like baseline
    g_mixed = g_odd + g_even
    g_mixed_antisym = hp.antisymmetric_nonlocal_part(g_mixed)
    assert np.allclose(g_mixed_antisym, g_odd, atol=1e-15), "Mixed function must yield only the odd component."


def test_extract_nonlocal_gap_thresholding():
    """Verify threshold onset gap extraction and linear interpolation on synthetic curves."""
    energies = np.linspace(-1.0, 1.0, 201)  # dE = 0.01
    delta_true = 0.20
    thresh_factor = 0.05
    dE = energies[1] - energies[0]
    
    # Create synthetic step-like onset above delta_true
    g_signal = np.zeros_like(energies)
    g_signal[energies >= delta_true - 1e-9] = 1.0
    g_signal[energies <= -delta_true + 1e-9] = -1.0
    
    # Add symmetric CAR baseline inside gap
    g_car_baseline = 0.05 * np.cos(energies)
    g_total = g_signal + g_car_baseline
    
    gap_val, g_filt, mask = hp.extract_nonlocal_gap(
        energies, g_total, median_size=1, gauss_sigma=0.0, gap_threshold_factor=thresh_factor, noise_threshold=1e-4
    )
    
    # With the Microsoft TGP midpoint formula, the gap is found at the first point
    # crossing the threshold (E=0.20) plus half the grid spacing.
    expected_gap = delta_true + dE / 2.0
    assert np.isclose(gap_val, expected_gap, atol=1e-12), f"Extracted gap {gap_val} does not match expected {expected_gap}."


def test_extract_nonlocal_gap_2d():
    """Verify that extract_nonlocal_gap properly handles 2D arrays across multiple traces."""
    energies = np.linspace(-1.0, 1.0, 101)  # dE = 0.02
    n_params = 5
    g_2d = np.zeros((n_params, len(energies)))
    thresh_factor = 0.05
    dE = energies[1] - energies[0]
    
    delta_expected = [0.1, 0.2, 0.3, 0.4, 0.5]
    for i, d in enumerate(delta_expected):
        g_2d[i, energies >= d - 1e-9] = 2.0
        g_2d[i, energies <= -d + 1e-9] = -2.0
        # Add symmetric baseline
        g_2d[i] += 0.1
        
    gaps, g_filt, mask = hp.extract_nonlocal_gap(
        energies, g_2d, median_size=1, gauss_sigma=0.0, gap_threshold_factor=thresh_factor, noise_threshold=1e-4
    )
    
    assert gaps.shape == (n_params,), f"Expected shape ({n_params},), got {gaps.shape}"
    assert g_filt.shape == g_2d.shape, f"Expected g_filt shape {g_2d.shape}, got {g_filt.shape}"
    assert mask.shape == g_2d.shape, f"Expected mask shape {g_2d.shape}, got {mask.shape}"
    
    for i, d in enumerate(delta_expected):
        expected_gap = d + dE / 2.0
        assert np.isclose(gaps[i], expected_gap, atol=1e-12), f"Trace {i}: expected gap {expected_gap}, got {gaps[i]}"


# Setup cross-validation imports with azure-quantum-tgp if available
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "azure-quantum-tgp"))

try:
    import xarray as xr
    from tgp.two import determine_gap, antisymmetric_conductance_part
    HAS_TGP = True
except ImportError:
    HAS_TGP = False

pytestmark_tgp = pytest.mark.skipif(not HAS_TGP, reason="azure-quantum-tgp or xarray not installed")


@pytestmark_tgp
def test_antisymmetrization_cross_validation():
    """Test 1: Verify exact numerical parity of anti-symmetrization with tgp.two.antisymmetric_conductance_part."""
    bias = np.linspace(-1.5, 1.5, 151)
    g_random = np.sin(2 * bias) + 0.3 * np.cos(3 * bias) + 1.2
    
    # Run helpers.py
    g_ours = hp.antisymmetric_nonlocal_part(g_random)
    
    # Run azure-quantum-tgp
    da = xr.DataArray(g_random, dims=["left_bias"], coords={"left_bias": bias})
    da_tgp = antisymmetric_conductance_part(da, bias_name="left_bias", field_name=None)
    
    assert np.allclose(g_ours, da_tgp.values, atol=1e-14), "Anti-symmetrization does not match TGP exactly."


@pytestmark_tgp
def test_single_trace_cross_validation_no_filters():
    """Test 2: Verify single-trace gap extraction parity against tgp.two.determine_gap (filters disabled)."""
    energies = np.linspace(-1.0, 1.0, 201)
    delta_true = 0.25
    thresh_factor = 0.05
    
    g_signal = np.zeros_like(energies)
    g_signal[energies >= delta_true - 1e-9] = 1.0
    g_signal[energies <= -delta_true + 1e-9] = -1.0
    g_total = g_signal + 0.05 * np.cos(energies)
    
    #helpers.py TGP implementation
    gap_ours, _, _ = hp.extract_nonlocal_gap(
        energies, g_total, median_size=1, gauss_sigma=0.0, gap_threshold_factor=thresh_factor, noise_threshold=1e-4, global_threshold=False
    )
    
    # Microsoft TGP determine_gap
    da = xr.DataArray(g_total, dims=["left_bias"], coords={"left_bias": energies})
    gap_tgp = determine_gap(
        da, bias_name="left_bias", field_name=None,
        median_size=1, gauss_sigma=(0.0,),
        gap_threshold_factor=thresh_factor, noise_threshold=1e-4
    ).item()
    
    assert np.isclose(gap_ours, gap_tgp, atol=1e-12), f"Single trace gap mismatch: {gap_ours} vs {gap_tgp}"


@pytestmark_tgp
def test_multi_trace_cross_validation_no_filters():
    """Test 3: Verify multi-trace global-threshold gap extraction parity against tgp.two.determine_gap."""
    energies = np.linspace(-1.0, 1.0, 101)
    n_params = 5
    g_2d = np.zeros((n_params, len(energies)))
    thresh_factor = 0.05
    
    delta_expected = [0.1, 0.2, 0.3, 0.4, 0.5]
    for i, d in enumerate(delta_expected):
        g_2d[i, energies >= d - 1e-9] = (i + 1) * 0.8
        g_2d[i, energies <= -d + 1e-9] = -(i + 1) * 0.8
        g_2d[i] += 0.1
        
    gaps_ours, _, _ = hp.extract_nonlocal_gap(
        energies, g_2d, median_size=1, gauss_sigma=0.0, gap_threshold_factor=thresh_factor, noise_threshold=1e-4, global_threshold=True
    )
    
    da = xr.DataArray(g_2d, dims=["field", "left_bias"], coords={"field": np.arange(n_params), "left_bias": energies})
    gaps_tgp = determine_gap(
        da, bias_name="left_bias", field_name="field",
        median_size=1, gauss_sigma=(0.0, 0.0),
        gap_threshold_factor=thresh_factor, noise_threshold=1e-4
    ).values
    
    assert np.allclose(gaps_ours, gaps_tgp, atol=1e-12), f"Multi-trace gap mismatch: {gaps_ours} vs {gaps_tgp}"


@pytestmark_tgp
def test_single_trace_cross_validation_with_filters():
    """Test 4: Verify single-trace gap extraction parity with median and Gaussian filters enabled."""
    energies = np.linspace(-1.0, 1.0, 201)
    g_noisy = np.sin(5 * energies) + np.where(np.abs(energies) > 0.3, np.sign(energies) * 2.0, 0.0)
    
    gap_ours, _, _ = hp.extract_nonlocal_gap(
        energies, g_noisy, median_size=3, gauss_sigma=1.0, gap_threshold_factor=0.05, global_threshold=False
    )
    
    da = xr.DataArray(g_noisy, dims=["left_bias"], coords={"left_bias": energies})
    gap_tgp = determine_gap(
        da, bias_name="left_bias", field_name=None,
        median_size=3, gauss_sigma=(1.0,),
        gap_threshold_factor=0.05
    ).item()
    
    assert np.isclose(gap_ours, gap_tgp, atol=1e-12), f"Filtered single trace mismatch: {gap_ours} vs {gap_tgp}"


@pytestmark_tgp
def test_zero_bias_crossing():
    """Test 5: Verify gapless trace (zero-bias crossing) returns V_0 + dV/2 under TGP midpoint formula."""
    energies = np.linspace(-1.0, 1.0, 101)
    g_gapless = 5.0 * energies
    
    gap_ours, _, _ = hp.extract_nonlocal_gap(
        energies, g_gapless, median_size=1, gauss_sigma=0.0, gap_threshold_factor=0.05, global_threshold=False
    )
    
    da = xr.DataArray(g_gapless, dims=["left_bias"], coords={"left_bias": energies})
    gap_tgp = determine_gap(
        da, bias_name="left_bias", field_name=None,
        median_size=1, gauss_sigma=(0.0,),
        gap_threshold_factor=0.05
    ).item()
    
    assert np.isclose(gap_ours, gap_tgp, atol=1e-12), f"Zero-bias crossing mismatch: {gap_ours} vs {gap_tgp}"
    assert gap_ours > 0.0, "TGP midpoint formula should yield positive value for first point exceeding threshold."


@pytestmark_tgp
def test_fully_gapped_trace():
    """Test 6: Verify fully gapped trace returns max bias or NaN depending on max_gap_mode."""
    energies = np.linspace(-1.0, 1.0, 101)
    g_flat = np.zeros_like(energies)
    
    gap_ours_max, _, _ = hp.extract_nonlocal_gap(
        energies, g_flat, median_size=1, gauss_sigma=0.0, max_gap_mode="max_bias", global_threshold=False
    )
    gap_ours_nan, _, _ = hp.extract_nonlocal_gap(
        energies, g_flat, median_size=1, gauss_sigma=0.0, max_gap_mode="nan", global_threshold=False
    )
    
    da = xr.DataArray(g_flat, dims=["left_bias"], coords={"left_bias": energies})
    gap_tgp = determine_gap(
        da, bias_name="left_bias", field_name=None,
        median_size=1, gauss_sigma=(0.0,)
    ).item()
    
    assert np.isclose(gap_ours_max, gap_tgp, atol=1e-12), f"Fully gapped mismatch: {gap_ours_max} vs {gap_tgp}"
    assert np.isnan(gap_ours_nan), "max_gap_mode='nan' must return NaN when fully gapped."


def test_nanmin_gap_combination():
    """Test 7: Verify nanmin_gap_combination handles NaNs correctly as required by TGP."""
    gap_LR = np.array([0.2, np.nan, 0.3, np.nan])
    gap_RL = np.array([0.1, 0.15, np.nan, np.nan])
    
    combined = hp.nanmin_gap_combination(gap_LR, gap_RL)
    expected = np.array([0.1, 0.15, 0.3, np.nan])
    
    assert np.allclose(combined[:3], expected[:3], atol=1e-15)
    assert np.isnan(combined[3]), "Both NaNs must combine to NaN."
