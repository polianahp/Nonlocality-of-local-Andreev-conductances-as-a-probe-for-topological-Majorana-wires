#!/usr/bin/env python3
"""
Tests/test_barrier_thermal_conductance.py

Comprehensive verification suite for finite-temperature barrier sweep conductances:
1. test_zero_temperature_rollback_parity: Bitwise identical to legacy E=0 calculation when T=0.
2. test_quadrature_convergence: Verifies bipolar Gauss-Legendre quadrature matches scipy.integrate.quad.
3. test_parity_with_broadened_bias_trace: Validates cross-parity between the newly computed quadrature
   conductance at V=0 and the thermally broadened dI/dV bias curve at V=0.
4. test_physical_plateau_emergence: Verifies the emergence of the finite G_LL plateau at UR >> UL
   under T = 30 mK, reproducing Dourado et al. (PRB 110, 014504) Fig. 5a.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.integrate import quad
import scipy.signal

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.helpers as hp
from src.cpu_broadening import _temp_kernel


class TestBarrierThermalConductance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Physics parameters for test nanowires
        cls.t = 100.0
        cls.mu = 1.0
        cls.mu_n = 0.0
        cls.Delta0 = 0.5
        cls.gamma = 0.2
        cls.alpha = 3.0
        cls.Ln = 0
        cls.Lb = 3
        cls.Ls = 200
        cls.mu_leads = cls.t
        cls.barrier0 = 2.0
        cls.Vdisx = np.zeros(cls.Ls)
        cls.Vc = np.sqrt(cls.mu**2 + cls.Delta0**2)
        cls.Vz = 1.3 * cls.Vc  # Topological regime

        cls.T_mK = 30.0
        cls.k_B = 8.617333262145e-5  # eV/K
        cls.T_meV = (cls.T_mK * 1e-3) * cls.k_B * 1e3

    def test_zero_temperature_rollback_parity(self):
        """Verify that T=0 quadrature reproduces the legacy E=0 calculation exactly."""
        syst = hp.build_system(
            t=self.t, mu=self.mu, mu_n=self.mu_n, Delta0=self.Delta0, gamma=self.gamma,
            V_z=self.Vz, alpha=self.alpha, Ln=self.Ln, Lb=self.Lb, Ls=self.Ls,
            mu_leads=self.mu_leads, barrier_l=self.barrier0, barrier_r=self.barrier0,
            Vdisx=self.Vdisx
        )

        # Legacy zero-temperature calculation
        g_legacy = hp.calc_conductance_matrix(syst, eng=0.0)

        # Finite-temperature quadrature evaluated at T_mK = 0.0
        q_engs, q_wts = hp.get_thermal_quadrature_nodes(T_mK=0.0, n_nodes=10)
        self.assertEqual(len(q_engs), 1)
        self.assertEqual(q_engs[0], 0.0)
        self.assertEqual(q_wts[0], 1.0)

        g_quad_t0 = hp.calc_conductance_matrix_finite_T(syst, q_engs, q_wts)

        max_diff = np.max(np.abs(g_legacy - g_quad_t0))
        self.assertEqual(
            max_diff, 0.0,
            f"Zero-temperature rollback mismatch: max_diff = {max_diff}"
        )
        print(f"\n  [PASS] Zero-T Rollback Parity: max_diff = {max_diff}")

    def test_quadrature_convergence(self):
        """Verify bipolar Gauss-Legendre quadrature achieves < 0.05% error vs scipy.integrate.quad."""
        # Asymmetric transmission function with dip and resonance peaks
        def g_kernel(E):
            return 0.2 + 0.5 * (E**2) / (E**2 + 0.01**2) + 0.1 * np.tanh(E / 0.01)

        beta = 1.0 / self.T_meV
        def integrand(E):
            u = 0.5 * beta * E
            u_clipped = np.clip(u, -50.0, 50.0)
            sech = 2.0 / (np.exp(u_clipped) + np.exp(-u_clipped))
            return g_kernel(E) * 0.25 * beta * (sech**2)

        exact_val, _ = quad(integrand, -0.5, 0.5)

        q_engs, q_wts = hp.get_thermal_quadrature_nodes(T_mK=self.T_mK, n_nodes=10)
        quad_val = np.sum(q_wts * g_kernel(q_engs))

        rel_error = abs(exact_val - quad_val) / exact_val
        self.assertLess(
            rel_error, 5e-4,
            f"Quadrature relative error {rel_error:.2e} exceeds threshold 5e-4 (0.05%)"
        )
        print(f"  [PASS] Quadrature Convergence: exact = {exact_val:.6f}, quad = {quad_val:.6f}, rel_err = {rel_error:.2e}")

    def test_parity_with_broadened_bias_trace(self):
        """
        Cross-Parity Validation:
        Asserts that the zero-bias conductance from the broadened dI/dV bias spectrum
        matches the direct quadrature calculation at the same temperature.
        """
        syst = hp.build_system(
            t=self.t, mu=self.mu, mu_n=self.mu_n, Delta0=self.Delta0, gamma=self.gamma,
            V_z=self.Vz, alpha=self.alpha, Ln=self.Ln, Lb=self.Lb, Ls=self.Ls,
            mu_leads=self.mu_leads, barrier_l=self.barrier0, barrier_r=self.barrier0,
            Vdisx=self.Vdisx
        )

        # Route A: Broadened Bias Trace via fine energy grid
        energies = np.linspace(-0.06, 0.06, 301)
        dIdV_left = np.zeros_like(energies)
        dIdV_right = np.zeros_like(energies)

        for idx, eng in enumerate(energies):
            gmat = hp.calc_conductance_matrix(syst, eng=eng)
            dIdV_left[idx] = gmat[0, 0]
            dIdV_right[idx] = gmat[1, 1]

        K = _temp_kernel(energies, self.T_meV)
        delta_bias = energies[1] - energies[0]
        K_norm = K * delta_bias
        dIdV_left_broadened = scipy.signal.fftconvolve(dIdV_left, K_norm, mode='same')
        dIdV_right_broadened = scipy.signal.fftconvolve(dIdV_right, K_norm, mode='same')

        zero_idx = len(energies) // 2
        g_ll_bias_conv = dIdV_left_broadened[zero_idx]
        g_rr_bias_conv = dIdV_right_broadened[zero_idx]

        # Route B: Direct Quadrature Solver at V=0
        q_engs, q_wts = hp.get_thermal_quadrature_nodes(T_mK=self.T_mK, n_nodes=10)
        gmat_quad = hp.calc_conductance_matrix_finite_T(syst, q_engs, q_wts)
        g_ll_quad = gmat_quad[0, 0]
        g_rr_quad = gmat_quad[1, 1]

        rel_diff_L = abs(g_ll_bias_conv - g_ll_quad) / g_ll_bias_conv
        rel_diff_R = abs(g_rr_bias_conv - g_rr_quad) / g_rr_bias_conv

        # Agreement within 0.5% (difference due solely to discrete grid discretization in bias convolution)
        self.assertLess(
            rel_diff_L, 0.005,
            f"G_LL bias parity mismatch: conv={g_ll_bias_conv:.6f}, quad={g_ll_quad:.6f}, rel_diff={rel_diff_L:.2e}"
        )
        self.assertLess(
            rel_diff_R, 0.005,
            f"G_RR bias parity mismatch: conv={g_rr_bias_conv:.6f}, quad={g_rr_quad:.6f}, rel_diff={rel_diff_R:.2e}"
        )
        print(f"  [PASS] Cross-Parity with Broadened Bias Trace:")
        print(f"         G_LL: bias_conv = {g_ll_bias_conv:.6f}, quad = {g_ll_quad:.6f}, rel_diff = {rel_diff_L:.2e}")
        print(f"         G_RR: bias_conv = {g_rr_bias_conv:.6f}, quad = {g_rr_quad:.6f}, rel_diff = {rel_diff_R:.2e}")

    def test_physical_plateau_emergence(self):
        """
        Verify Dourado et al. Fig. 5a physics:
        In a well-localized wire (Ls = 300), raising UR >> UL at T=0 produces a zero-energy dip (G_LL -> 0).
        At T = 30 mK, G_LL saturates at a finite plateau significantly above the T=0 value.
        """
        Ls_long = 300
        Vdisx_long = np.zeros(Ls_long)
        syst_asym = hp.build_system(
            t=self.t, mu=self.mu, mu_n=self.mu_n, Delta0=self.Delta0, gamma=self.gamma,
            V_z=self.Vz, alpha=self.alpha, Ln=self.Ln, Lb=self.Lb, Ls=Ls_long,
            mu_leads=self.mu_leads, barrier_l=self.barrier0, barrier_r=40.0 * self.barrier0,
            Vdisx=Vdisx_long
        )

        # Zero temperature
        g_asym_t0 = hp.calc_conductance_matrix(syst_asym, eng=0.0)[0, 0]

        # Finite temperature (T = 30 mK)
        q_engs, q_wts = hp.get_thermal_quadrature_nodes(T_mK=self.T_mK, n_nodes=10)
        g_asym_t30 = hp.calc_conductance_matrix_finite_T(syst_asym, q_engs, q_wts)[0, 0]

        # Plateau emergence: finite T conductance is strictly higher than zero-energy dip
        self.assertGreater(
            g_asym_t30, g_asym_t0,
            f"Finite-T plateau expected to exceed T=0 dip value: T=30mK={g_asym_t30:.6f}, T=0={g_asym_t0:.6f}"
        )
        print(f"  [PASS] Physical Plateau Emergence (Dourado Fig 5a):")
        print(f"         G_LL(T=0, dip) = {g_asym_t0:.6f} -> G_LL(T=30mK, plateau) = {g_asym_t30:.6f} (Ratio: {g_asym_t30/g_asym_t0:.1f}x)")


if __name__ == "__main__":
    unittest.main()
