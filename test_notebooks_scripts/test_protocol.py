import pytest
import numpy as np
import helpers as hp
import test_notebooks_scripts.reference_protocol as ref
from pathlib import Path

# Load data once for all tests
DATA_DIR = Path('/home/pseudonym/code/Nonlocal_Conductance/Nonlocality-of-local-Andreev-conductances-as-a-probe-for-topological-Majorana-wires/Data/Protocol_V2/disorder_realization_0_results')

def get_data():
    brcl = np.load(DATA_DIR / 'barrier_right_conductance_left_arr.npy')
    brcr = np.load(DATA_DIR / 'barrier_right_conductance_right_arr.npy')
    peaks_left = np.load(DATA_DIR / 'peaks_left.npy')
    peaks_right = np.load(DATA_DIR / 'peaks_right.npy')
    pdi_data = np.load(DATA_DIR / 'pdi_data.npy')
    params_list = np.load(DATA_DIR / 'params_list.npy')
    return brcl, brcr, peaks_left, peaks_right, pdi_data, params_list

@pytest.fixture(scope="module")
def data():
    return get_data()

def test_monotonicity(data):
    brcl, brcr, _, _, _, _ = data
    ref_mono = ref.check_arr_monotonic(brcl, brcr)
    
    mono_left = np.all(np.diff(brcl, axis=-1) <= 0, axis=-1)
    mono_right = np.all(np.diff(brcr, axis=-1) <= 0, axis=-1)
    opt_mono = (mono_left & mono_right).astype(int)
    
    np.testing.assert_array_equal(opt_mono, ref_mono)

def test_peak_symmetry(data):
    _, _, peaks_left, peaks_right, _, _ = data
    ref_symm = ref.check_peak_symmetry(peaks_left, peaks_right)
    
    # Vectorized logic
    symm_tol = 1e-6 # Using isclose default for tolerance in ref logic
    opt_sym_left = np.isclose(np.abs(peaks_left[:, 2]), np.abs(peaks_left[:, 4]))
    opt_sym_right = np.isclose(np.abs(peaks_right[:, 2]), np.abs(peaks_right[:, 4]))
    opt_symm = (opt_sym_left & opt_sym_right).astype(int)
    
    np.testing.assert_array_equal(opt_symm, ref_symm)

def test_peak_window(data):
    _, _, peaks_left, peaks_right, _, _ = data
    window = 0.035
    ref_window = ref.check_peak_window(peaks_left, peaks_right, window)
    
    # Vectorized logic
    opt_left = (np.abs(peaks_left[:, 2]) <= window) & (np.abs(peaks_left[:, 4]) <= window)
    opt_right = (np.abs(peaks_right[:, 2]) <= window) & (np.abs(peaks_right[:, 4]) <= window)
    opt_window = (opt_left & opt_right).astype(int)
    
    np.testing.assert_array_equal(opt_window, ref_window)

def test_resonance_peak(data):
    brcl, brcr, peaks_left, peaks_right, _, _ = data
    # ref uses has_peaks which just does sum(peak_dat) > 0
    ref_has_peak = ref.check_resonance_peak(peaks_left, peaks_right)
    
    # Vectorized logic: sum over columns > 0
    opt_left = np.sum(peaks_left, axis=1) > 0
    opt_right = np.sum(peaks_right, axis=1) > 0
    opt_has_peak = (opt_left & opt_right).astype(int)
    
    np.testing.assert_array_equal(opt_has_peak, ref_has_peak)

def test_negative_peaks(data):
    brcl, brcr, peaks_left, peaks_right, _, _ = data
    ref_neg = ref.check_negative_peaks_arr(peaks_left, peaks_right)
    
    # Vectorized logic: neg_peak_pos (col 4) < 0
    opt_left = peaks_left[:, 4] < 0.0
    opt_right = peaks_right[:, 4] < 0.0
    opt_neg = (opt_left | opt_right).astype(int)
    
    np.testing.assert_array_equal(opt_neg, ref_neg)

def test_correlation(data):
    brcl, brcr, _, _, _, _ = data
    thresh = 0.5
    ref_corr = ref.check_correlation(brcl, brcr, thresh)
    
    # Vectorized/precomputed logic
    raw_corrs = np.array([hp.calc_correlation(brcl[i], brcr[i]) for i in range(len(brcl))])
    opt_corr = np.clip(raw_corrs, 0.0, 1.0)
    opt_corr = np.where(opt_corr < thresh, 0.0, opt_corr)
    opt_corr = np.where(opt_corr >= thresh, 1.0, opt_corr)
    
    np.testing.assert_allclose(opt_corr, ref_corr, atol=1e-8)

def test_full_protocol(data):
    brcl, brcr, peaks_left, peaks_right, pdi_data, params_list = data
    params = {
        "check_correlation": True,
        "check_resonance_peak": True,
        "check_negative_peaks": True,
        "check_monotonic": True,
        "check_peak_symmetry": True, 
        "check_peak_window": True,
        "check_island_stability": False, # Exclude spatial stability for now to isolate local filters
        "corr_thresh": 0.5,
        "window": 0.035,
        "stability_radus": 0.027,
        "stability_frac": 0.8
    }
    
    ref_prot = ref.calc_protocol(brcl, brcr, peaks_left, peaks_right, pdi_data, **params)
    
    # Run the equivalent vectorized logic using our new pieces
    corr_thresh = params["corr_thresh"]
    raw_corrs = np.array([hp.calc_correlation(brcl[i], brcr[i]) for i in range(len(brcl))])
    opt_corr = np.clip(raw_corrs, 0.0, 1.0)
    opt_corr = np.where(opt_corr < corr_thresh, 0.0, opt_corr)
    opt_corr = np.where(opt_corr >= corr_thresh, 1.0, opt_corr)
    
    opt_has_peak = (np.sum(peaks_left, axis=1) > 0) & (np.sum(peaks_right, axis=1) > 0)
    opt_neg = (peaks_left[:, 4] < 0.0) | (peaks_right[:, 4] < 0.0)
    
    mono_left = np.all(np.diff(brcl, axis=-1) <= 0, axis=-1)
    mono_right = np.all(np.diff(brcr, axis=-1) <= 0, axis=-1)
    opt_mono = (mono_left & mono_right)
    
    opt_sym_left = np.isclose(np.abs(peaks_left[:, 2]), np.abs(peaks_left[:, 4]))
    opt_sym_right = np.isclose(np.abs(peaks_right[:, 2]), np.abs(peaks_right[:, 4]))
    opt_symm = (opt_sym_left & opt_sym_right)
    
    win = params["window"]
    opt_win_left = (np.abs(peaks_left[:, 2]) <= win) & (np.abs(peaks_left[:, 4]) <= win)
    opt_win_right = (np.abs(peaks_right[:, 2]) <= win) & (np.abs(peaks_right[:, 4]) <= win)
    opt_window = (opt_win_left & opt_win_right)
    
    opt_prot = np.ones_like(opt_corr)
    if params["check_correlation"]: opt_prot *= opt_corr
    if params["check_resonance_peak"]: opt_prot *= (1 - opt_has_peak)
    if params["check_negative_peaks"]: opt_prot *= (1 - opt_neg)
    if params["check_monotonic"]: opt_prot *= opt_mono
    if params["check_peak_symmetry"]: opt_prot *= opt_symm
    if params["check_peak_window"]: opt_prot *= opt_window
    
    np.testing.assert_array_equal(opt_prot, ref_prot)
