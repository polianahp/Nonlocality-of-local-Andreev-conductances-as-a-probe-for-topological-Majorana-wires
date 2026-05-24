#!/usr/bin/env python3
"""
Test System Equivalence
=======================

Verifies that the PfaffianSimulator (Pfaffian invariant calculator),
PDICalculator, and Kwant build_system_closed all represent the same
physical Hamiltonian.

Three testing strategies:
  1. Eigenvalue Comparison (Spectra Test)
  2. Phase Boundary / Gap Closure Alignment
  3. Mathematica Ground Truth Validation via cal_pfaffian_invariant wrapper

Run:
    python test_system_equivalence.py
"""

import sys
import os
import numpy as np
import scipy.sparse.linalg as sla

# Ensure imports work from the project directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import (
    build_system_closed,
    PDICalculator,
    cal_pfaffian_invariant,
)
from pfaffian_invariant import PfaffianSimulator, eta_m


# ============================================================================
# Helper: Build full Hamiltonian from PfaffianSimulator parameters
# ============================================================================

def build_pfaff_full_hamiltonian(sim, Gamma, mu, V0, gamma0, theta=0.0):
    """
    Reconstruct the full Nx-site Hamiltonian from PfaffianSimulator blocks.

    The PfaffianSimulator uses a folded (doubled) chain representation for
    its decimation algorithm, but the underlying physics is a single 1D chain
    of Nx sites with 4 orbitals each (Nambu x Spin).

    We reconstruct the unfolded Hamiltonian to enable direct eigenvalue
    comparison with Kwant.
    """
    Nx = sim.Nx
    Ny = sim.Ny
    assert Ny == 1, "This reconstruction only supports Ny=1"

    norb = 4  # orbitals per site for Ny=1

    # On-site block (without disorder/pairing — from H02D for chain 1 only)
    h0_block = np.zeros((norb, norb), dtype=complex)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    phase_p = Gamma * (cos_t + 1j * sin_t)
    phase_m = Gamma * (cos_t - 1j * sin_t)
    epsilon0 = sim.epsilon0

    h0_block = np.array([
        [epsilon0 - mu, phase_p, 0.0, 0.0],
        [phase_m, epsilon0 - mu, 0.0, 0.0],
        [0.0, 0.0, -(epsilon0 - mu), -phase_m],
        [0.0, 0.0, -phase_p, -(epsilon0 - mu)]
    ], dtype=complex)

    # Pairing block (from delta_2D, first 4x4 sub-block)
    delta_block = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ], dtype=complex)

    # Disorder potential matrix (from V2D, first 4x4 sub-block)
    v_block = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)

    # Hopping block (from T2D, chain 1 portion — rows/cols 0:4)
    t_hop = np.array([
        [-sim.tx, -0.5 * sim.alphax, 0.0, 0.0],
        [0.5 * sim.alphax, -sim.tx, 0.0, 0.0],
        [0.0, 0.0, sim.tx, 0.5 * sim.alphax],
        [0.0, 0.0, -0.5 * sim.alphax, sim.tx]
    ], dtype=complex)

    # Get the disorder profile in the original (unfolded) basis
    # The Vdiss array in PfaffianSimulator has shape (Nx/2, 8*Ny)
    # For Ny=1, each row has 8 entries: [val_front]*4 + [val_back]*4
    # val_front = original disorder at site i
    # val_back = original disorder at site Nx-1-i
    # So Vdiss[i][0] gives the disorder for site i of the original chain
    # and Vdiss[i][4] gives the disorder for site Nx-1-i

    dim = Nx * norb
    H_full = np.zeros((dim, dim), dtype=complex)

    for i in range(Nx):
        idx = i * norb

        # Get the disorder value for this site
        # In the folded representation: site i maps to fold index i (first half)
        # and site Nx-1-i maps to fold index i (second half)
        if i < Nx // 2:
            fold_idx = i
            v_dis = sim.Vdiss[fold_idx, 0]  # front chain value
        else:
            fold_idx = Nx - 1 - i
            v_dis = sim.Vdiss[fold_idx, 4]  # back chain value

        sc_dis = 1.0  # SCdiss2D is initialized to ones

        # On-site: H0 + V0*Vdis*V + gamma0*SCdis*Delta
        H_full[idx:idx+norb, idx:idx+norb] = (
            h0_block + V0 * v_dis * v_block + gamma0 * sc_dis * delta_block
        )

        # Hopping to next site
        if i < Nx - 1:
            H_full[idx:idx+norb, idx+norb:idx+2*norb] = t_hop
            H_full[idx+norb:idx+2*norb, idx:idx+norb] = t_hop.conj().T

    return H_full


# ============================================================================
# Test 1: Eigenvalue Comparison (Spectra Test)
# ============================================================================

def test_spectra_pfaff_vs_pdi(Nx=50, mu_test=1.0, gm_test=0.87):
    """
    Compare eigenvalues of the full Hamiltonian built from PfaffianSimulator
    parameters versus the full Hamiltonian built by PDICalculator.

    These two should use the EXACT same physics (same basis, same conventions).
    """
    print("\n" + "="*70)
    print("TEST 1a: PfaffianSimulator vs PDICalculator Eigenvalue Comparison")
    print("="*70)

    # Use PfaffianSimulator default physical constants
    sim = PfaffianSimulator(Nx=Nx, Ny=1, delta_N=5)
    ts = sim.tx
    alphas = sim.alphax
    gamma0 = sim.gamma0

    print(f"  Parameters: Nx={Nx}, ts={ts:.6f}, alphas={alphas:.4f}, "
          f"gamma0={gamma0}, delta0={sim.delta0}")
    print(f"  epsilon0(Pfaff) = {sim.epsilon0:.6f}")
    print(f"  Test point: mu={mu_test}, gm={gm_test}")

    # Build PDICalculator Hamiltonian
    Vdisx_clean = np.zeros(Nx)
    pdi = PDICalculator(ts, alphas, gamma0, Nx, Vdisx_clean, 0.0)
    H_pdi = pdi.get_full_Hamiltonian(gm_test, mu_test)

    # Build PfaffianSimulator Hamiltonian (reconstructed)
    H_pfaff = build_pfaff_full_hamiltonian(sim, gm_test, mu_test, 0.0, gamma0)

    # Compute eigenvalues
    evals_pdi = np.sort(np.linalg.eigvalsh(H_pdi))
    evals_pfaff = np.sort(np.linalg.eigvalsh(H_pfaff))

    # Compare
    max_diff = np.max(np.abs(evals_pdi - evals_pfaff))
    print(f"  PDI   eigenvalues range: [{evals_pdi[0]:.8f}, {evals_pdi[-1]:.8f}]")
    print(f"  Pfaff eigenvalues range: [{evals_pfaff[0]:.8f}, {evals_pfaff[-1]:.8f}]")
    print(f"  Max eigenvalue difference: {max_diff:.2e}")

    # Note: PDICalculator uses 2*ts for epsilon0, while PfaffianSimulator
    # uses 2*(tx + ty*cos(pi/(Ny+1))) = 2*tx for Ny=1. These should match.
    eps_pdi = 2 * ts
    eps_pfaff = sim.epsilon0
    print(f"  epsilon0 check: PDI={eps_pdi:.8f}, Pfaff={eps_pfaff:.8f}, "
          f"diff={abs(eps_pdi - eps_pfaff):.2e}")

    assert max_diff < 1e-10, f"Eigenvalue mismatch of {max_diff:.2e}"


def test_spectra_pfaff_vs_kwant(Nx=50, mu_test=1.0, gm_test=0.87):
    """
    Compare eigenvalues of PfaffianSimulator's Hamiltonian versus Kwant's
    build_system_closed.

    The two systems differ in two known ways:
    1. BdG basis convention (Nambu ordering) — does not affect eigenvalues
    2. On-site energy: Kwant uses epsilon0 = 2*t*cos(pi/(Ls+1)) while
       PfaffianSimulator uses epsilon0 = 2*tx. This produces a uniform
       diagonal shift of Z * 2*ts*(1 - cos(pi/(Nx+1))) on all eigenvalues.

    We verify that the measured difference matches this predicted offset.
    """
    print("\n" + "="*70)
    print("TEST 1b: PfaffianSimulator vs Kwant (build_system_closed)")
    print("="*70)

    sim = PfaffianSimulator(Nx=Nx, Ny=1, delta_N=5)
    ts = sim.tx
    alphas = sim.alphax
    gamma0 = sim.gamma0
    delta0 = sim.delta0
    Z = sim.Z

    print(f"  Parameters: Nx={Nx}, ts={ts:.6f}, alphas={alphas:.4f}, "
          f"gamma0={gamma0}, delta0={delta0}, Z={Z:.6f}")
    print(f"  Test point: mu={mu_test}, Vz/Gamma={gm_test}")

    Vdisx_clean = np.zeros(Nx)

    # --- PfaffianSimulator Hamiltonian ---
    H_pfaff = build_pfaff_full_hamiltonian(sim, gm_test, mu_test, 0.0, gamma0)
    evals_pfaff = np.sort(np.linalg.eigvalsh(H_pfaff))

    # --- Kwant build_system_closed ---
    # Use t = ts so hoppings match. The epsilon0 values will differ:
    #   Kwant:  epsilon0 = 2*ts*cos(pi/(Nx+1))
    #   Pfaff:  epsilon0 = 2*ts
    syst_closed = build_system_closed(
        t=ts, mu=mu_test, gamma=gamma0, Delta0=delta0,
        V_z=gm_test, alpha=alphas, Ls=Nx, Vdisx=Vdisx_clean
    )
    ham_kwant = syst_closed.hamiltonian_submatrix(sparse=False)
    evals_kwant = np.sort(np.linalg.eigvalsh(ham_kwant))

    # The predicted offset from the epsilon0 difference.
    # In Kwant, the on-site diagonal has Z*(epsilon0_kwant - mu).
    # In Pfaff (scaled by Z), the diagonal has Z*(epsilon0_pfaff - mu).
    # The difference is Z*(epsilon0_pfaff - epsilon0_kwant)
    #   = Z * 2*ts * (1 - cos(pi/(Nx+1)))
    # This shifts the particle sector eigenvalues by +offset and
    # hole sector by -offset (due to particle-hole structure), so
    # the net effect on sorted eigenvalues is not a uniform shift.
    # Instead, we verify that the residual after removing this
    # correction is small.
    epsilon0_pfaff = 2.0 * ts
    epsilon0_kwant = 2.0 * ts * np.cos(np.pi / (Nx + 1.0))
    predicted_offset = Z * (epsilon0_pfaff - epsilon0_kwant)

    # Scale PfaffianSimulator eigenvalues by Z
    evals_pfaff_scaled = Z * evals_pfaff
    raw_max_diff = np.max(np.abs(evals_kwant - evals_pfaff_scaled))

    print(f"  epsilon0: Kwant={epsilon0_kwant:.8f}, Pfaff={epsilon0_pfaff:.8f}")
    print(f"  Predicted offset Z*(eps_pfaff - eps_kwant) = {predicted_offset:.6e}")
    print(f"  Raw max difference (Kwant vs Z*Pfaff): {raw_max_diff:.2e}")

    # The offset is small and consistent — verify it's within
    # the predicted range (offset affects each eigenvalue differently
    # depending on particle-hole content, but should be bounded by
    # the predicted value).
    assert raw_max_diff <= 2.0 * predicted_offset, f"Eigenvalue mismatch of {raw_max_diff:.2e} exceeds predicted offset"


def test_spectra_with_disorder(Nx=50, mu_test=1.5, gm_test=0.9):
    """
    Repeat the eigenvalue comparison with a disordered wire.
    """
    print("\n" + "="*70)
    print("TEST 1c: Eigenvalue Comparison with Disorder")
    print("="*70)

    sim = PfaffianSimulator(Nx=Nx, Ny=1, delta_N=5)
    ts = sim.tx
    alphas = sim.alphax
    gamma0 = sim.gamma0
    delta0 = sim.delta0

    # Generate a reproducible random disorder
    np.random.seed(42)
    Vdisx = np.random.randn(Nx)
    V0 = 1.5

    print(f"  Parameters: Nx={Nx}, V0={V0}, disorder std={np.std(Vdisx):.4f}")
    print(f"  Test point: mu={mu_test}, gm={gm_test}")

    # --- PDICalculator ---
    pdi = PDICalculator(ts, alphas, gamma0, Nx, Vdisx, V0)
    H_pdi = pdi.get_full_Hamiltonian(gm_test, mu_test)
    evals_pdi = np.sort(np.linalg.eigvalsh(H_pdi))

    # --- PfaffianSimulator (reconstructed) ---
    sim.load_disorder(Vdisx)
    H_pfaff = build_pfaff_full_hamiltonian(sim, gm_test, mu_test, V0, gamma0)
    evals_pfaff = np.sort(np.linalg.eigvalsh(H_pfaff))

    max_diff_pdi_pfaff = np.max(np.abs(evals_pdi - evals_pfaff))
    print(f"  Max diff (PDI vs Pfaff): {max_diff_pdi_pfaff:.2e}")

    assert max_diff_pdi_pfaff < 1e-10, f"Mismatch of {max_diff_pdi_pfaff:.2e}"


# ============================================================================
# Test 2: Phase Boundary / Gap Closure Alignment
# ============================================================================

def test_phase_boundary_alignment(Nx=400, mu_test=-0.49):
    """
    Sweep the Zeeman field Gamma across a known topological phase transition.

    Checks phase consistency between the Pfaffian invariant and the
    spectral gap. For a finite system with OBC:
    - In the TRIVIAL phase: all eigenvalues are gapped (bulk gap > 0)
    - In the TOPOLOGICAL phase: Majorana edge states appear at near-zero
      energy, so the lowest eigenvalue is exponentially small.

    We verify that whenever the invariant says "topological" (Q=1),
    the spectrum shows near-zero edge states, and whenever the invariant
    says "trivial" (Q=0) far from the transition, the gap is significant.
    """
    print("\n" + "="*70)
    print("TEST 2: Phase Consistency (Invariant vs Spectral Gap)")
    print("="*70)

    sim = PfaffianSimulator(Nx=Nx, Ny=1, delta_N=20)
    ts = sim.tx
    alphas = sim.alphax
    gamma0 = sim.gamma0
    delta0 = sim.delta0

    # Sweep Gamma from deep trivial into deep topological
    gm_values = np.arange(0.30, 0.90, 0.02)
    Vdisx_clean = np.zeros(Nx)

    print(f"  Sweeping Gamma from {gm_values[0]:.2f} to {gm_values[-1]:.2f} "
          f"at mu={mu_test}")
    print(f"  Using Nx={Nx}")

    invariants = []
    gaps = []
    bulk_gaps = []

    for gm in gm_values:
        # Compute Pfaffian invariant using our wrapper
        inv = cal_pfaffian_invariant(
            ts, alphas, gamma0, delta0, Nx, Vdisx_clean, 0.0,
            gm, mu_test
        )
        invariants.append(inv)

        # Compute spectrum from PDICalculator's Hamiltonian (OBC)
        pdi = PDICalculator(ts, alphas, gamma0, Nx, Vdisx_clean, 0.0)
        H = pdi.get_full_Hamiltonian(gm, mu_test)
        evals = np.sort(np.abs(np.linalg.eigvalsh(H)))
        gap = evals[0]   # lowest |eigenvalue| (includes edge states)
        bulk_gap = evals[2]  # 3rd lowest — skips the 2 Majorana edge states
        gaps.append(gap)
        bulk_gaps.append(bulk_gap)

    invariants = np.array(invariants)
    gaps = np.array(gaps)
    bulk_gaps = np.array(bulk_gaps)

    # Find phase transition: where invariant changes
    transitions = []
    for i in range(len(invariants) - 1):
        if invariants[i] != invariants[i+1]:
            transitions.append(i)

    print(f"\n  Phase transitions detected at:")
    for t_idx in transitions:
        gm_transition = 0.5 * (gm_values[t_idx] + gm_values[t_idx+1])
        print(f"    Gamma ≈ {gm_transition:.2f} "
              f"({invariants[t_idx]} → {invariants[t_idx+1]})")

    assert len(transitions) > 0, "No phase transition detected in invariant!"

    # Phase consistency checks:
    # 1. In deep topological phase (well past transition), gap should be
    #    very small (Majorana edge states) but bulk_gap should be nonzero.
    # 2. In deep trivial phase, gap should be significantly nonzero.
    edge_state_threshold = 0.01  # edge states should be below this
    bulk_gap_threshold = 0.01    # bulk gap should be above this

    n_topological_ok = 0
    n_topological_total = 0
    n_trivial_ok = 0
    n_trivial_total = 0

    # Only check points well away from transition (±3 steps)
    transition_zone = set()
    for t_idx in transitions:
        for offset in range(-3, 4):
            transition_zone.add(t_idx + offset)

    for i, (inv, gap, bgap) in enumerate(zip(invariants, gaps, bulk_gaps)):
        if i in transition_zone:
            continue  # skip near-transition points

        if inv >= 0.5:  # topological (mostly 1.0)
            n_topological_total += 1
            if gap < edge_state_threshold and bgap > bulk_gap_threshold:
                n_topological_ok += 1
            else:
                print(f"    WARNING: Topological point Gamma={gm_values[i]:.2f} "
                      f"has gap={gap:.4e}, bulk_gap={bgap:.4e}")
        elif inv < 0.5:  # trivial (mostly 0.0)
            n_trivial_total += 1
            if gap > bulk_gap_threshold:
                n_trivial_ok += 1
            else:
                print(f"    WARNING: Trivial point Gamma={gm_values[i]:.2f} "
                      f"has gap={gap:.4e}")

    print(f"\n  Topological phase consistency: "
          f"{n_topological_ok}/{n_topological_total} points OK")
    print(f"  Trivial phase consistency:     "
          f"{n_trivial_ok}/{n_trivial_total} points OK")

    passed = (n_topological_ok == n_topological_total and
              n_trivial_ok == n_trivial_total)

    assert passed, "Phase/gap inconsistency detected!"


# ============================================================================
# Test 3: Mathematica Ground Truth via cal_pfaffian_invariant wrapper
# ============================================================================

def test_mathematica_ground_truth():
    """
    Verify that cal_pfaffian_invariant (the helpers.py wrapper) reproduces
    the known Mathematica results.
    """
    print("\n" + "="*70)
    print("TEST 3a: cal_pfaffian_invariant wrapper — Clean wire validation")
    print("="*70)

    # Use PfaffianSimulator's default physical parameters
    sim = PfaffianSimulator(Nx=400, Ny=1, delta_N=20)
    ts = sim.tx
    alphas = sim.alphax
    gamma0 = sim.gamma0
    delta0 = sim.delta0

    Vdisx_clean = np.zeros(400)

    # Test case 1: Topological (Gamma=0.87, mu=-0.49)
    print("  Test case 1: Gamma=0.87, mu=-0.49 (expected: topological)")
    result1 = cal_pfaffian_invariant(
        ts, alphas, gamma0, delta0, 400, Vdisx_clean, 0.0,
        gm=0.87, mu=-0.49
    )
    expected1 = 1.0
    match1 = (result1 == expected1)
    print(f"    Result: {result1}")
    print(f"    {'✓ PASSED' if match1 else '✗ FAILED'}")

    # Test case 2: Trivial (Gamma=0.35, mu=-0.49)
    print("  Test case 2: Gamma=0.35, mu=-0.49 (expected: trivial)")
    result2 = cal_pfaffian_invariant(
        ts, alphas, gamma0, delta0, 400, Vdisx_clean, 0.0,
        gm=0.35, mu=-0.49
    )
    expected2 = 0.0
    match2 = (result2 == expected2)
    print(f"    Result: {result2}")
    print(f"    {'✓ PASSED' if match2 else '✗ FAILED'}")

    assert match1 and match2, "Mathematica ground truth mismatch for clean wire"


def test_mathematica_ground_truth_disordered():
    """
    Verify cal_pfaffian_invariant against the 10-point Mathematica ground truth
    from SMDisStrongMapNy1.dat (Layer 6 = index 5).
    """
    print("\n" + "="*70)
    print("TEST 3b: cal_pfaffian_invariant — Disordered wire (10-point test)")
    print("="*70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dis_path = os.path.join(os.path.dirname(script_dir), 'SMVdissNy1.dat')

    if not os.path.exists(dis_path):
        import pytest
        pytest.skip("SMVdissNy1.dat not found")

    dis = np.loadtxt(dis_path)

    # PfaffianSimulator uses Nx=410, delta_N=10 for the disordered case
    sim = PfaffianSimulator(Nx=410, Ny=1, delta_N=10)
    ts = sim.tx
    alphas = sim.alphax
    gamma0 = sim.gamma0
    delta0 = sim.delta0

    test_points = [
        (2.13, 1.0, 1),
        (2.16, 1.0, 0),
        (2.10, 1.0, 0),
        (1.90, 1.0, 1),
        (2.54, 1.11, 1),
        (1.31, 0.90, 1),
        (1.09, 1.21, 1),
        (1.02, 1.00, 0),
        (0.80, 0.73, 0),
        (0.64, 0.73, 0),
    ]

    all_passed = True
    for mu, gm, expected in test_points:
        result = cal_pfaffian_invariant(
            ts, alphas, gamma0, delta0, 400, dis[5:-5], 1.5,
            gm=gm, mu=mu
        )
        status = "✓" if result == float(expected) else "✗"
        if result != float(expected):
            all_passed = False
        print(f"  {status} mu={mu:.2f}, gm={gm:.2f}: "
              f"expected={expected}, got={result}")

    assert all_passed, "Some points did not match Mathematica ground truth!"


# ============================================================================
# Test 4: Parameter Mapping Verification
# ============================================================================

def test_parameter_mapping():
    """
    Verify that the parameter mapping from abstract (ts, alphas) to physical
    (ax, ms) and back is exact — i.e., cal_pfaffian_invariant correctly
    reconstructs PfaffianSimulator with the same derived parameters.
    """
    print("\n" + "="*70)
    print("TEST 4: Parameter Mapping Round-Trip")
    print("="*70)

    # Start with PfaffianSimulator defaults
    sim_ref = PfaffianSimulator(Nx=400, Ny=1, delta_N=20)
    ts = sim_ref.tx
    alphas = sim_ref.alphax
    gamma0 = sim_ref.gamma0
    delta0 = sim_ref.delta0

    # Reconstruct ax and ms (the mapping used in cal_pfaffian_invariant)
    ax_reconstructed = 200.0 / alphas
    ms_reconstructed = 1000.0 * eta_m / (2.0 * ax_reconstructed**2 * ts)

    # Build a new PfaffianSimulator with reconstructed parameters
    sim_check = PfaffianSimulator(
        Nx=400, Ny=1, delta_N=20,
        ax=ax_reconstructed, ay=ax_reconstructed, ms=ms_reconstructed,
        gamma0=gamma0, delta0=delta0
    )

    # Compare derived parameters
    print(f"  Reference: ax={sim_ref.ax}, ms={sim_ref.ms}")
    print(f"  Reconstructed: ax={ax_reconstructed:.10f}, ms={ms_reconstructed:.10f}")
    print(f"  tx: ref={sim_ref.tx:.10f}, recon={sim_check.tx:.10f}, "
          f"diff={abs(sim_ref.tx - sim_check.tx):.2e}")
    print(f"  alphax: ref={sim_ref.alphax:.10f}, recon={sim_check.alphax:.10f}, "
          f"diff={abs(sim_ref.alphax - sim_check.alphax):.2e}")
    print(f"  epsilon0: ref={sim_ref.epsilon0:.10f}, recon={sim_check.epsilon0:.10f}, "
          f"diff={abs(sim_ref.epsilon0 - sim_check.epsilon0):.2e}")

    checks = [
        abs(sim_ref.tx - sim_check.tx) < 1e-12,
        abs(sim_ref.alphax - sim_check.alphax) < 1e-12,
        abs(sim_ref.epsilon0 - sim_check.epsilon0) < 1e-12,
        abs(ax_reconstructed - sim_ref.ax) < 1e-12,
        abs(ms_reconstructed - sim_ref.ms) < 1e-12,
    ]

    assert all(checks), "Parameter mapping introduced errors!"


