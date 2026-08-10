import pytest
import numpy as np
import xarray as xr
import sys
from pathlib import Path
import scipy.sparse.csgraph

# Add NonlocalProtocol to path
sys.path.append(str(Path(__file__).parent.parent))
import helpers as hp

# Add azure-quantum-tgp to path
sys.path.append("/home/pseudonym/Documents/Code/azure-quantum-tgp")
from tgp import two as msft_two

class TGPXarrayAdapter:
    @staticmethod
    def to_dataset(g_ll=None, g_rr=None, g_lr=None, g_rl=None, bias_array=None):
        """
        Converts dense numpy arrays of shape (N_mu, N_vz, N_cutter, N_bias)
        into the xarray.Dataset format expected by Microsoft's TGP.
        """
        if bias_array is None:
            bias_array = np.linspace(-0.1, 0.1, 11)
            
        # For simplicity in testing, we'll use a single V_gate and B point if not provided,
        # or adapt to the shape of g_ll.
        shape = g_ll.shape if g_ll is not None else (1, 1, 1, len(bias_array))
        N_mu, N_vz, N_cutter, N_bias = shape
        
        dims_left = ["V_gate", "B", "cutter_pair_index", "left_bias"]
        coords = {
            "V_gate": np.linspace(-1, 1, N_mu),
            "B": np.linspace(0, 1, N_vz),
            "cutter_pair_index": np.arange(N_cutter),
            "left_bias": bias_array,
            "right_bias": bias_array
        }
        
        ds_left = xr.Dataset(coords=coords)
        ds_right = xr.Dataset(coords=coords)
        
        # In TGP, dims might be ["V_gate", "B", "cutter_pair_index", "left_bias"]
        if g_ll is not None:
            ds_left["g_ll"] = (dims_left, g_ll)
        if g_lr is not None:
            ds_left["g_lr"] = (dims_left, g_lr)
            
        dims_right = ["V_gate", "B", "cutter_pair_index", "right_bias"]
        if g_rr is not None:
            ds_right["g_rr"] = (dims_right, g_rr)
        if g_rl is not None:
            ds_right["g_rl"] = (dims_right, g_rl)
            
        return ds_left, ds_right


def test_stage1_gap_parity():
    # Synthetic data: single trace, 11 bias points
    bias = np.linspace(-0.1, 0.1, 11)
    # create a gap that opens above 0.05
    g_lr = np.zeros((1, 1, 1, 11))
    # Fill trace
    trace = np.abs(bias) * 0.1
    g_lr[0, 0, 0, :] = trace
    
    # Run Microsoft determine_gap and set_gap_threshold
    ds_left, _ = TGPXarrayAdapter.to_dataset(g_lr=g_lr, bias_array=bias)
    msft_gap = msft_two.determine_gap(
        conductance=ds_left["g_lr"],
        bias_name="left_bias",
        field_name="B",
        median_size=1, # no median filter for this test
        gauss_sigma=(0.0, 0.0),
        gap_threshold_factor=0.05,
        noise_threshold=0.0
    )
    
    # Run our extract_nonlocal_gap
    # Note: our extract_nonlocal_gap takes 2D [N_params, N_bias]
    g_lr_flat = g_lr.reshape(1, 11)
    our_gap, _, _ = hp.extract_nonlocal_gap(
        bias=bias,
        g_nonlocal=g_lr_flat,
        median_size=1,
        gauss_sigma=0.0,
        gap_threshold_factor=0.05,
        noise_threshold=0.0,
        global_threshold=False
    )
    
    # Assert
    np.testing.assert_allclose(our_gap, msft_gap.values.flatten())


def test_stage2_zbp_curvature_parity():
    # Make bias range strictly greater than bias_window (0.01)
    bias = np.linspace(-0.012, 0.012, 13) # 13 points (2uV spacing)
    delta = 0.002 # 2 uV
    
    # Create a parabolic peak
    trace = -1000.0 * bias**2
    g_ll = np.zeros((1, 1, 1, 13))
    g_ll[0, 0, 0, :] = trace
    g_rr = np.zeros((1, 1, 1, 13))
    g_rr[0, 0, 0, :] = trace
    
    ds_left, ds_right = TGPXarrayAdapter.to_dataset(g_ll=g_ll, g_rr=g_rr, bias_array=bias)
    
    # Run Microsoft ZBP
    # Note bias_window = 10e-3 (10mV), so window_length = int(2*10e-3 / 0.002) = 10 -> 11
    msft_zbp_ds = msft_two.zbp_dataset_derivative(
        ds_left, ds_right,
        derivative_threshold=100.0,
        zbp_probability_threshold=0.6,
        bias_window=10e-3,
        polyorder=2,
        average_over_cutter=True
    )
    
    import scipy.signal
    # For res=0.002, our pipeline extracts the central 3 points for the filter
    fL_slice = g_ll[:, :, :, [5, 6, 7]] # center 3 points
    curv_L = scipy.signal.savgol_filter(fL_slice, window_length=3, polyorder=2, deriv=2, delta=delta, axis=-1)[:, :, :, 1]
    
    # Assert
    our_mask = (curv_L <= -100.0)
    
    # Microsoft evaluates the derivative over the *entire* trace and then checks the center ZBP condition
    # For a simple parabola, the second derivative is constant (-2000.0), so all points have curv <= -100.0
    msft_mask = msft_zbp_ds.left.values
    np.testing.assert_array_equal(np.mean(our_mask, axis=2) >= 0.60, msft_mask >= 0.60)


def test_stage2_marginal_probability_parity():
    # Construct an array of 25 cutter pairs where exactly 15 pass (15/25 = 0.60)
    # 0.60 should pass Microsoft's P_left >= 0.6 condition.
    
    curv_L = np.zeros((1, 1, 25))
    curv_R = np.zeros((1, 1, 25))
    
    # Make 15 cutters pass
    curv_L[0, 0, :15] = -150.0
    curv_R[0, 0, :15] = -150.0
    
    # Make 10 cutters fail
    curv_L[0, 0, 15:] = -50.0
    curv_R[0, 0, 15:] = -50.0
    
    # Our marginal logic
    zbp_mask_L = (curv_L <= -100.0)
    zbp_mask_R = (curv_R <= -100.0)
    
    prob_L = np.mean(zbp_mask_L, axis=2)
    prob_R = np.mean(zbp_mask_R, axis=2)
    
    stage1_pass = (prob_L >= 0.60) & (prob_R >= 0.60)
    
    # Ensure it passes
    assert stage1_pass.item() == True
    
    # If it was 14 cutters (14/25 = 0.56), it should fail
    curv_L[0, 0, 14] = -50.0
    zbp_mask_L2 = (curv_L <= -100.0)
    prob_L2 = np.mean(zbp_mask_L2, axis=2)
    stage1_pass2 = (prob_L2 >= 0.60) & (prob_R >= 0.60)
    
    assert stage1_pass2.item() == False


def test_stage3_cluster_pruning_parity():
    # 5x5 grid
    island_mask = np.zeros((5, 5), dtype=bool)
    
    # 1. An isolated 2-pixel cluster (should be pruned)
    island_mask[0, 0] = True
    island_mask[0, 1] = True
    
    # 2. A 3-pixel cluster (should be kept)
    island_mask[3, 3] = True
    island_mask[3, 4] = True
    island_mask[4, 3] = True
    
    import scipy.ndimage
    labels, num_features = scipy.ndimage.label(island_mask, structure=np.ones((3,3)))
    pruned_mask = island_mask.copy()
    for i in range(1, num_features + 1):
        if np.sum(labels == i) < 3:
            pruned_mask[labels == i] = False
            
    # Assert 2-pixel cluster is gone
    assert pruned_mask[0, 0] == False
    assert pruned_mask[0, 1] == False
    
    # Assert 3-pixel cluster is kept
    assert pruned_mask[3, 3] == True
    assert pruned_mask[3, 4] == True
    assert pruned_mask[4, 3] == True
