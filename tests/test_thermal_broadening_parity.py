#!/usr/bin/env python3
"""
tests/test_thermal_broadening_parity.py

Validates mathematical and numerical equivalence between:
1. Microsoft TGP official prepare_sim / broaden_with_temperature
2. High-performance memory-efficient chunked CPU implementation (src/cpu_broadening.py)

Uses a slice of real Harpers Ferry simulation data (Outputs/Data/Post_Process_Test)
with the full 97-point dynamic bias grid to guarantee non-linear tail behavior
without exceeding local memory capacity.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tgp_adapter import TGPAdapter
from src.cpu_broadening import broaden_with_temperature_cpu, prepare_sim_cpu
import tgp.prepare
import tgp.two


class TestThermalBroadeningParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate candidate test data
        test_dir = REPO_ROOT / "Outputs" / "Data" / "Post_Process_Test" / "disorder_realization_0"
        if not test_dir.exists():
            raise FileNotFoundError(f"Test data not found at {test_dir}")

        adapter = TGPAdapter(str(test_dir))
        ds_full = adapter.to_xarray()

        # Slice small representative subset: 2 cutters, 3 B points, 4 V points -> 24 traces
        cls.ds_small = ds_full.isel(
            cutter_pair_index=slice(0, 2),
            B=slice(10, 13),
            V=slice(20, 24)
        )
        cls.T_mK = 30.0

    def test_broaden_with_temperature_parity(self):
        """Verify conductances match official TGP broadening to machine precision."""
        ds_off = tgp.prepare.broaden_with_temperature(self.ds_small, T_mK=self.T_mK)
        # Use small chunk_size=5 to ensure chunk boundary stitching is tested
        ds_chk = broaden_with_temperature_cpu(self.ds_small, T_mK=self.T_mK, chunk_size=5)

        for key in ["g_ll", "g_lr", "g_rl", "g_rr"]:
            arr_off = ds_off[key].values
            arr_chk = ds_chk[key].values

            max_abs_diff = float(np.max(np.abs(arr_off - arr_chk)))
            denom = np.maximum(np.abs(arr_off), 1e-12)
            max_rel_diff = float(np.max(np.abs(arr_off - arr_chk) / denom))

            # Machine precision test
            self.assertTrue(
                np.allclose(arr_off, arr_chk, atol=1e-12, rtol=1e-10),
                f"Parity mismatch in {key}: max_abs={max_abs_diff}, max_rel={max_rel_diff}"
            )
            print(f"  [PASS] {key:6s}: max_abs_diff={max_abs_diff:.4e}, max_rel_diff={max_rel_diff:.4e}")

    def test_downstream_stage2_pipeline_parity(self):
        """Verify complete prepare_sim and Stage 2 gap extraction produce identical outputs."""
        ds_prep_off = tgp.prepare.prepare_sim(self.ds_small, T_mK=self.T_mK)
        ds_prep_chk = prepare_sim_cpu(self.ds_small, T_mK=self.T_mK, chunk_size=5)

        def run_stage2(tprep):
            tl = tprep.rename({"bias": "left_bias"})
            tr = tprep.rename({"bias": "right_bias"})
            tl, tr = tgp.two.extract_gap(tl, tr, gap_threshold_factor=0.05, noise_threshold=1e-4)
            z_ds = tgp.two.zbp_dataset_derivative(tl, tr, zbp_probability_threshold=0.7, average_over_cutter=False)
            tgp.two.set_gap_threshold(z_ds, threshold_low=10e-3, threshold_high=0.067)
            return tl, tr, z_ds

        tl_off, tr_off, z_off = run_stage2(ds_prep_off)
        tl_chk, tr_chk, z_chk = run_stage2(ds_prep_chk)

        # Transport gaps extracted must be identical
        gap_l_diff = float(np.max(np.abs(tl_off["gap"].values - tl_chk["gap"].values)))
        gap_r_diff = float(np.max(np.abs(tr_off["gap"].values - tr_chk["gap"].values)))
        self.assertEqual(gap_l_diff, 0.0, f"Transport gap Left difference: {gap_l_diff}")
        self.assertEqual(gap_r_diff, 0.0, f"Transport gap Right difference: {gap_r_diff}")

        # Boolean flags must be exact array match
        np.testing.assert_array_equal(z_off["zbp"].values, z_chk["zbp"].values)
        np.testing.assert_array_equal(z_off["gapped_zbp"].values, z_chk["gapped_zbp"].values)

        print("  [PASS] Transport gaps identical (diff = 0.0)")
        print("  [PASS] Boolean ZBP and Gapped ZBP arrays 100% identical")


if __name__ == "__main__":
    unittest.main()
