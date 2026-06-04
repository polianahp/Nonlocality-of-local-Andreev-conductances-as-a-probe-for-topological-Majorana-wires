import pytest
import numpy as np
import sys
import os

# Add parent directory to path so that helpers can be imported when running pytest directly from the Tests directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import helpers as hp

def make_didv_with_peaks(energies, peak_positions, peak_heights=1.0, peak_width=0.005, baseline=0.1):
    """Create synthetic dI/dV = baseline + sum of Lorentzians at peak_positions."""
    ys = np.full_like(energies, baseline, dtype=float)
    for pos in peak_positions:
        ys += peak_heights * (peak_width**2) / ((energies - pos)**2 + peak_width**2)
    return ys

def test_detect_peaks_v2_symmetric_peaks():
    xs = np.linspace(-0.15, 0.15, 301)
    ys = make_didv_with_peaks(xs, [-0.05, 0.05])
    res = hp.detect_peaks_v2(ys, xs)
    
    assert res['has_peaks'] == True
    assert res['has_both'] == True
    assert res['pos_energy'] == pytest.approx(0.05, abs=1e-3)
    assert res['neg_energy'] == pytest.approx(-0.05, abs=1e-3)

def test_detect_peaks_v2_single_zbp():
    xs = np.linspace(-0.15, 0.15, 301)
    ys = make_didv_with_peaks(xs, [0.0])
    res = hp.detect_peaks_v2(ys, xs)
    
    assert res['has_peaks'] == True
    assert res['has_both'] == True
    assert res['pos_energy'] == pytest.approx(0.0)
    assert res['neg_energy'] == pytest.approx(0.0)

def test_detect_peaks_v2_no_peaks():
    xs = np.linspace(-0.15, 0.15, 301)
    ys = np.ones_like(xs) * 0.1
    res = hp.detect_peaks_v2(ys, xs)
    
    assert res['has_peaks'] == False
    assert res['has_both'] == False
    assert np.isnan(res['pos_energy'])
    assert np.isnan(res['neg_energy'])

def test_detect_peaks_v2_only_positive():
    xs = np.linspace(-0.15, 0.15, 301)
    ys = make_didv_with_peaks(xs, [0.05])
    res = hp.detect_peaks_v2(ys, xs)
    
    assert res['has_peaks'] == True
    assert res['has_both'] == False
    assert res['pos_energy'] == pytest.approx(0.05, abs=1e-3)
    assert np.isnan(res['neg_energy'])

def test_detect_peaks_v2_multiple_peaks():
    xs = np.linspace(-0.15, 0.15, 301)
    ys = make_didv_with_peaks(xs, [-0.08, -0.02, 0.02, 0.08])
    res = hp.detect_peaks_v2(ys, xs)
    
    assert res['has_peaks'] == True
    assert res['has_both'] == True
    assert res['pos_energy'] == pytest.approx(0.02, abs=1e-3)
    assert res['neg_energy'] == pytest.approx(-0.02, abs=1e-3)

def make_peak_data(n, pos_energy=0.01, neg_energy=-0.01,
                   pos_height=1.0, neg_height=1.0,
                   has_peaks=True, has_both=True):
    """Create a synthetic (n, 6) peak data array with uniform values."""
    data = np.zeros((n, 6))
    data[:, 0] = 1.0 if has_peaks else 0.0
    data[:, 1] = 1.0 if has_both else 0.0
    data[:, 2] = pos_energy
    data[:, 3] = pos_height
    data[:, 4] = neg_energy
    data[:, 5] = neg_height
    return data

def test_symmetry_perfect():
    data = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    res = hp.check_peak_symmetry(data, symmetry_tol=0.005)
    assert np.all(res == 1.0)

def test_symmetry_within_tol():
    data = make_peak_data(10, pos_energy=0.01, neg_energy=-0.012)
    res = hp.check_peak_symmetry(data, symmetry_tol=0.005)
    assert np.all(res == 1.0)

def test_symmetry_outside_tol():
    data = make_peak_data(10, pos_energy=0.01, neg_energy=-0.02)
    res = hp.check_peak_symmetry(data, symmetry_tol=0.005)
    assert np.all(res == 0.0)

def test_symmetry_no_both_peaks():
    data = make_peak_data(10, has_both=False)
    res = hp.check_peak_symmetry(data, symmetry_tol=0.005)
    assert np.all(res == 0.0)

def test_symmetry_zbp():
    data = make_peak_data(10, pos_energy=0.0, neg_energy=0.0)
    res = hp.check_peak_symmetry(data, symmetry_tol=0.005)
    assert np.all(res == 1.0)

def test_symmetry_mixed():
    data = np.zeros((10, 6))
    data[:5] = make_peak_data(5, pos_energy=0.01, neg_energy=-0.01)
    data[5:] = make_peak_data(5, pos_energy=0.01, neg_energy=-0.02)
    res = hp.check_peak_symmetry(data, symmetry_tol=0.005)
    assert np.all(res[:5] == 1.0)
    assert np.all(res[5:] == 0.0)

def test_agreement_perfect():
    left = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    right = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    res = hp.check_peak_position_agreement(left, right, peak_diff_tol=0.005)
    assert np.all(res == 1.0)

def test_agreement_within_tol():
    left = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    right = make_peak_data(10, pos_energy=0.012, neg_energy=-0.012)
    res = hp.check_peak_position_agreement(left, right, peak_diff_tol=0.005)
    assert np.all(res == 1.0)

def test_agreement_pos_fails():
    left = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    right = make_peak_data(10, pos_energy=0.03, neg_energy=-0.01)
    res = hp.check_peak_position_agreement(left, right, peak_diff_tol=0.005)
    assert np.all(res == 0.0)

def test_agreement_neg_fails():
    left = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    right = make_peak_data(10, pos_energy=0.01, neg_energy=-0.03)
    res = hp.check_peak_position_agreement(left, right, peak_diff_tol=0.005)
    assert np.all(res == 0.0)

def test_agreement_one_side_missing():
    left = make_peak_data(10, has_both=True)
    right = make_peak_data(10, has_both=False)
    res = hp.check_peak_position_agreement(left, right, peak_diff_tol=0.005)
    assert np.all(res == 0.0)

def test_agreement_mixed():
    left = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    right = np.zeros((10, 6))
    right[:5] = make_peak_data(5, pos_energy=0.01, neg_energy=-0.01)
    right[5:] = make_peak_data(5, pos_energy=0.03, neg_energy=-0.01)
    res = hp.check_peak_position_agreement(left, right, peak_diff_tol=0.005)
    assert np.all(res[:5] == 1.0)
    assert np.all(res[5:] == 0.0)

def make_params_grid(n_mu, n_vz, mu_range=(0, 1), vz_range=(0, 1)):
    """Create a params_list array matching the simulation grid layout."""
    mu_vals = np.linspace(*mu_range, n_mu)
    vz_vals = np.linspace(*vz_range, n_vz)
    params = []
    idx = 0
    for mu in mu_vals:
        for vz in vz_vals:
            params.append([idx, mu, vz])
            idx += 1
    return np.array(params)

def test_stability_all_positive():
    params = make_params_grid(5, 5)
    pmap = np.ones(25)
    res = hp.check_mode_stability(pmap, params, stability_radius=1, stability_frac=0.5)
    assert np.all(res == 1.0)

def test_stability_all_negative():
    params = make_params_grid(5, 5)
    pmap = np.zeros(25)
    res = hp.check_mode_stability(pmap, params, stability_radius=1, stability_frac=0.5)
    assert np.all(res == 0.0)

def test_stability_isolated_point():
    params = make_params_grid(5, 5)
    pmap = np.zeros((5, 5))
    pmap[2, 2] = 1.0
    res = hp.check_mode_stability(pmap.flatten(), params, stability_radius=1, stability_frac=0.5)
    assert np.all(res == 0.0)

def test_stability_surrounded_point():
    params = make_params_grid(5, 5)
    pmap = np.zeros((5, 5))
    pmap[1:4, 1:4] = 1.0
    res = hp.check_mode_stability(pmap.flatten(), params, stability_radius=1, stability_frac=0.5)
    res = res.reshape(5, 5)
    assert res[2, 2] == 1.0

def test_stability_corner_point():
    params = make_params_grid(5, 5)
    pmap = np.zeros((5, 5))
    pmap[0, 0] = 1.0
    pmap[0, 1] = 1.0
    pmap[1, 0] = 1.0
    res = hp.check_mode_stability(pmap.flatten(), params, stability_radius=1, stability_frac=0.5)
    res = res.reshape(5, 5)
    assert res[0, 0] == 1.0 # 2 out of 3 neighbors are positive >= 0.5

def test_stability_edge_point():
    params = make_params_grid(5, 5)
    pmap = np.zeros((5, 5))
    pmap[0, 2] = 1.0
    pmap[0, 1:4] = 1.0
    pmap[1, 2] = 1.0
    res = hp.check_mode_stability(pmap.flatten(), params, stability_radius=1, stability_frac=0.5)
    res = res.reshape(5, 5)
    assert res[0, 2] == 1.0 # 3 out of 5 neighbors are positive >= 0.5

def test_stability_radius_2():
    params = make_params_grid(7, 7)
    pmap = np.zeros((7, 7))
    pmap[1:6, 1:6] = 1.0
    res = hp.check_mode_stability(pmap.flatten(), params, stability_radius=2, stability_frac=0.5)
    res = res.reshape(7, 7)
    assert res[3, 3] == 1.0

def test_stability_frac_threshold():
    params = make_params_grid(3, 3)
    pmap = np.ones((3, 3))
    pmap[1, 1] = 0.0
    res = hp.check_mode_stability(pmap.flatten(), params, stability_radius=1, stability_frac=1.0)
    res = res.reshape(3, 3)
    assert res[1, 1] == 0.0

def test_protocol_new_all_pass():
    left = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    right = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    corr = np.ones(10)
    res = hp.calc_protocol_new(corr, left, right)
    assert np.all(res == 1.0)

def test_protocol_new_corr_fail():
    left = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    right = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    corr = np.zeros(10) # fails corr filter
    res = hp.calc_protocol_new(corr, left, right)
    assert np.all(res == 0.0)

def test_protocol_new_symmetry_fail():
    left = make_peak_data(10, pos_energy=0.01, neg_energy=-0.03) # asymmetric
    right = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    corr = np.ones(10)
    res = hp.calc_protocol_new(corr, left, right)
    assert np.all(res == 0.0)

def test_protocol_new_agreement_fail():
    left = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    right = make_peak_data(10, pos_energy=0.03, neg_energy=-0.03)
    corr = np.ones(10)
    res = hp.calc_protocol_new(corr, left, right)
    assert np.all(res == 0.0)

def test_protocol_new_no_stability():
    left = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    right = make_peak_data(10, pos_energy=0.01, neg_energy=-0.01)
    corr = np.ones(10)
    # params_list=None by default
    res = hp.calc_protocol_new(corr, left, right)
    assert np.all(res == 1.0)

def test_protocol_new_with_stability():
    # 5x5 grid
    params = make_params_grid(5, 5)
    left = make_peak_data(25, pos_energy=0.01, neg_energy=-0.01)
    right = make_peak_data(25, pos_energy=0.01, neg_energy=-0.01)
    corr = np.zeros(25)
    # Only center is 1.0
    corr[12] = 1.0
    
    # Without stability, center passes
    res_no_stab = hp.calc_protocol_new(corr, left, right)
    assert res_no_stab[12] == 1.0
    
    # With stability, center fails due to isolation
    res_stab = hp.calc_protocol_new(corr, left, right, params_list=params, stability_radius=1, stability_frac=0.5)
    assert res_stab[12] == 0.0

if __name__ == "__main__":
    pytest.main([__file__])
