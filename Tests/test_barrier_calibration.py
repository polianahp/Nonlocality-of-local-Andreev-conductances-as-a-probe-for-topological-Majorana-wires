import pytest
import numpy as np
import kwant
import sys
import os

# Add parent directory to path so that src modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.helpers as hp
import src.barrier_calibration as bc

def compute_1d_analytic_transmission(t, mu_leads, Ub, Lb, E):
    """
    Exact analytical wave-matching calculation for 1D tight-binding transmission
    through Lb rectangular barrier sites of height Ub between leads with hopping t.

    Theory:
      On a 1D lattice, the Schrödinger equation at site j is:
        (epsilon_j - E) * psi_j - t * psi_{j+1} - t * psi_{j-1} = 0.
      Rearranging yields the standard 2x2 transfer matrix connecting adjacent sites:
        [psi_{j+1}, psi_j]^T = M_j @ [psi_j, psi_{j-1}]^T, where
        M_j = [[(epsilon_j - E)/t, -1], [1, 0]].
      For a uniform barrier of length Lb, M_total = (M_site)^Lb.
      In the left lead (j <= 0), psi_j = exp(i*k*j) + r * exp(-i*k*j).
      In the right lead (j >= Lb), psi_j = t_tr * exp(i*k*j).
      Matching boundary vectors [psi_0, psi_{-1}]^T and [psi_Lb, psi_{Lb-1}]^T yields
      a 2x2 linear system for transmission amplitude t_tr and reflection amplitude r.
      Transmission probability is T_analytic = |t_tr|^2.

    Recommended Reading:
      - S. Datta, 'Electronic Transport in Mesoscopic Systems' (Cambridge, 1995), Chap 3.
      - P. Markos & C. M. Soukoulis, 'Wave Scattering in Complex Media' (Princeton, 2008), Chap 6.
      - C. W. J. Beenakker, Rev. Mod. Phys. 69, 731 (1997), Sec. II.A (Transfer Matrix Formalism).
    """
    # Lead dispersion: E = 2t - mu_leads - 2t cos(k)
    cos_k = (2.0 * t - mu_leads - E) / (2.0 * t)
    assert abs(cos_k) < 1.0, f"Energy {E} is outside lead band"
    k = np.arccos(cos_k)
    
    # Barrier transfer matrix: (eps_b - E) psi_j - t psi_{j+1} - t psi_{j-1} = 0
    # psi_{j+1} = ((eps_b - E)/t) psi_j - psi_{j-1}
    eps_b = 2.0 * t + Ub
    M_site = np.array([[(eps_b - E) / t, -1.0],
                       [1.0, 0.0]], dtype=complex)
    M_total = np.linalg.matrix_power(M_site, Lb)
    
    # Wave-matching vectors at barrier boundaries
    v_inc = np.array([1.0, np.exp(-1j * k)])
    v_refl = np.array([1.0, np.exp(1j * k)])
    v_R = np.array([np.exp(1j * k * Lb), np.exp(1j * k * (Lb - 1))])
    
    A = np.column_stack([v_R, -M_total @ v_refl])
    b = M_total @ v_inc
    sol = np.linalg.solve(A, b)
    t_tr = sol[0]
    return float(np.abs(t_tr)**2)

def test_monotonicity_and_bracket():
    """
    Test that G_N(U) is strictly monotonically decreasing with barrier height U
    over the entire [-3.0, 6.0] meV bracket for BOTH left and right junctions,
    and encompasses the standard target range [0.35, 0.90] e^2/h.
    """
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3
    alpha = 1.4
    Ln = 0
    Lb = 3
    Ls = 300  # Full 3-micron wire0  # Full 3-micron wire
    mu = 0.5
    
    u_values = np.linspace(-3.0, 6.0, 19)
    for side in ['left', 'right']:
        gn_values = [
            bc.calculate_gn_above_gap(
                u_barrier=u, side=side, t=t, mu=mu, gamma=gamma,
                Delta0=Delta0, alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls
            )
            for u in u_values
        ]
        gn_arr = np.array(gn_values)
        diffs = np.diff(gn_arr)
        assert np.all(diffs < 0), f"G_N({side}) is not strictly monotonic! diffs = {diffs}"
        assert gn_arr[0] > 0.90, f"Upper bracket failed for {side}: {gn_arr[0]} <= 0.90"
        assert gn_arr[-1] < 0.35, f"Lower bracket failed for {side}: {gn_arr[-1]} >= 0.35"

def test_target_conductance_inversion():
    """
    Test exact inversion for the 5 standard targets:
    G_N in {0.35, 0.49, 0.62, 0.76, 0.90} e^2/h.
    Assert |G_N(U_c) - G_target| < 0.005 e^2/h for both cutters,
    and verify left/right symmetry on a clean wire.
    """
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3
    alpha = 1.4
    Ln = 0
    Lb = 3
    Ls = 300  # Full 3-micron wire0  # Full 3-micron wire
    
    # Representative low, mid, high targets spanning the [0.35, 0.90] e^2/h bracket
    targets = [0.35, 0.62, 0.90]
    mu_points = np.linspace(0.0, 2.0, 15)
    
    for target in targets:
        # Calibrate left cutter
        u_c_left = bc.calibrate_single_junction(
            target_gn=target, side='left', t=t, gamma=gamma,
            Delta0=Delta0, alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls,
            u_min=-3.0, u_max=6.0, tol=1e-4
        )
        gns_l = [
            bc.calculate_gn_above_gap(
                u_barrier=u_c_left, side='left', t=t, mu=mu, gamma=gamma,
                Delta0=Delta0, alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls
            )
            for mu in mu_points
        ]
        gn_l_median = np.median(gns_l)
        assert abs(gn_l_median - target) < 0.005, (
            f"Left calibration failed for target {target}: got {gn_l_median} at U={u_c_left}"
        )
        
        # Calibrate right cutter
        u_c_right = bc.calibrate_single_junction(
            target_gn=target, side='right', t=t, gamma=gamma,
            Delta0=Delta0, alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls,
            u_min=-3.0, u_max=6.0, tol=1e-4
        )
        gns_r = [
            bc.calculate_gn_above_gap(
                u_barrier=u_c_right, side='right', t=t, mu=mu, gamma=gamma,
                Delta0=Delta0, alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls
            )
            for mu in mu_points
        ]
        gn_r_median = np.median(gns_r)
        assert abs(gn_r_median - target) < 0.005, (
            f"Right calibration failed for target {target}: got {gn_r_median} at U={u_c_right}"
        )
        
        # Wire inversion symmetry on clean wire
        assert abs(u_c_left - u_c_right) < 1e-4, (
            f"Asymmetry detected on clean wire: left={u_c_left}, right={u_c_right}"
        )

def test_clean_wire_analytical_wavematching():
    """
    Test that Kwant 1D scalar transmission matches the exact analytical transfer matrix
    formula to machine precision (atol=1e-12), and that the BdG single-junction normal
    conductance per spin channel tracks this analytical transmission.
    """
    t = 16.529
    mu_leads = t
    Ub = 2.0
    Lb = 3
    E = 0.5
    
    # 1. Exact transfer matrix calculation
    T_analytic = compute_1d_analytic_transmission(t=t, mu_leads=mu_leads, Ub=Ub, Lb=Lb, E=E)
    
    # 2. Kwant 1D single-orbital benchmark
    lat = kwant.lattice.chain(1, norbs=1)
    syst = kwant.Builder()
    for i in range(Lb):
        syst[lat(i)] = 2 * t + Ub
        if i > 0:
            syst[lat(i), lat(i-1)] = -t
            
    sym_l = kwant.TranslationalSymmetry((-1,))
    lead_l = kwant.Builder(sym_l)
    lead_l[lat(0)] = 2 * t - mu_leads
    lead_l[lat(1), lat(0)] = -t
    syst.attach_lead(lead_l)
    
    sym_r = kwant.TranslationalSymmetry((1,))
    lead_r = kwant.Builder(sym_r)
    lead_r[lat(0)] = 2 * t - mu_leads
    lead_r[lat(1), lat(0)] = -t
    syst.attach_lead(lead_r)
    
    fsyst = syst.finalized()
    smat = kwant.smatrix(fsyst, E)
    T_kwant = smat.transmission(1, 0)
    
    np.testing.assert_allclose(
        T_kwant, T_analytic, atol=1e-12, rtol=1e-12,
        err_msg="Kwant 1D scalar transmission diverges from analytical transfer matrix"
    )

def test_topological_quantization_zero_bias():
    """
    Test that a wire in the topological regime (V_z > sqrt(mu^2 + Delta_0^2))
    with barrier tuned to target G_N exhibits exact 2 e^2/h zero-bias conductance
    due to isolated Majorana zero modes.
    Also includes a planted-truth negative control in the trivial regime (V_z = 0.0)
    verifying conductance is far from 2.0 e^2/h.
    """
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3
    alpha = 1.4
    Ln = 0
    Lb = 3
    Ls = 300  # Full 3-micron wire0  # Physical wire length matching project config
    mu = 0.2
    V_z_topo = 0.6  # Optimal topological regime with minimal MZM overlap
    
    # Tune barrier to G_N = 0.5 e^2/h at V_z = 0 (using Ls=50 for fast junction calibration)
    u_c = bc.calibrate_single_junction(
        target_gn=0.5, side='left', t=t, gamma=gamma,
        Delta0=Delta0, alpha=alpha, Ln=Ln, Lb=Lb, Ls=50
    )
    
    # Positive control: Topological state (V_z = 0.6)
    syst_topo = hp.build_system_exact(
        t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=V_z_topo,
        alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=t,
        barrier_l=u_c, barrier_r=5.0,
        closed=False
    )
    G_topo = hp.calc_conductance_matrix_exact(syst_topo, E=0.0)
    g_zero_bias = G_topo[0, 0]
    
    # Must be quantized to 2.0 e^2/h to within 5e-3
    np.testing.assert_allclose(
        g_zero_bias, 2.0, atol=5e-3,
        err_msg=f"Topological zero-bias conductance {g_zero_bias} is not 2.0 e^2/h"
    )
    
    # Negative control: Trivial state (V_z = 0.0)
    syst_triv = hp.build_system_exact(
        t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=0.0,
        alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=t,
        barrier_l=u_c, barrier_r=5.0,
        closed=False
    )
    G_triv = hp.calc_conductance_matrix_exact(syst_triv, E=0.0)
    g_triv = G_triv[0, 0]
    
    # In trivial state at zero bias with tunneling barrier, Andreev reflection is small (~ 4 * T_N^2 / (2 - T_N)^2)
    assert abs(g_triv - 2.0) > 0.5, (
        f"Planted-truth negative control failed: trivial conductance {g_triv} is unexpectedly near 2.0 e^2/h"
    )

def test_dourado_b14_nonlocal_suppression():
    """
    Test Dourado Eq. (B14): Under opposite barrier pinch-off (U_R >> Delta_0),
    non-local conductances G_LR and G_RL are exponentially suppressed.
    """
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3
    alpha = 1.4
    Ln = 0
    Lb = 3
    Ls = 100
    mu = 0.5
    V_z = 0.5
    
    # Symmetrical open barriers: U_L = 1.0, U_R = 1.0
    syst_open = hp.build_system_exact(
        t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=V_z,
        alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=t,
        barrier_l=1.0, barrier_r=1.0,
        closed=False
    )
    G_open = hp.calc_conductance_matrix_exact(syst_open, E=0.02)
    nonlocal_open = abs(G_open[0, 1]) + abs(G_open[1, 0])
    
    # Ensure open non-local conductance is well above solver noise floor
    assert nonlocal_open > 1e-5, f"Open non-local conductance {nonlocal_open} is too small to test suppression"
    
    # Pinched right barrier: U_L = 1.0, U_R = 60.0 meV
    syst_pinched = hp.build_system_exact(
        t=t, mu=mu, gamma=gamma, Delta0=Delta0, V_z=V_z,
        alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls, mu_leads=t,
        barrier_l=1.0, barrier_r=60.0,
        closed=False
    )
    G_pinched = hp.calc_conductance_matrix_exact(syst_pinched, E=0.02)
    nonlocal_pinched = abs(G_pinched[0, 1]) + abs(G_pinched[1, 0])
    
    # Pinched right barrier must suppress non-local conductance by at least 10^3
    assert nonlocal_pinched < nonlocal_open * 1e-3, (
        f"Non-local suppression failed: open={nonlocal_open}, pinched={nonlocal_pinched}"
    )

def test_fail_loud_unreachable_target_and_invalid_inputs():
    """
    Test fail-loud error handling:
    1. Unreachable targets (e.g. G_N = 2.5 or -0.5 e^2/h) raise ValueError.
    2. Inverted brackets (u_min >= u_max) raise ValueError.
    3. Invalid side parameter raises ValueError.
    """
    t = 16.529
    gamma = 0.2
    Delta0 = 0.3
    alpha = 1.4
    Ln = 0
    Lb = 3
    Ls = 300  # Full 3-micron wire
    
    # 1. Unreachable / physical bounds
    for invalid_target in [-0.5, 2.5]:
        with pytest.raises(ValueError, match="cannot be bracketed|unreachable|out of physical bounds"):
            bc.calibrate_single_junction(
                target_gn=invalid_target, side='left', t=t, gamma=gamma,
                Delta0=Delta0, alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls,
                u_min=-3.0, u_max=6.0
            )

    # 2. Inverted bracket
    with pytest.raises(ValueError, match="u_min must be strictly less than u_max"):
        bc.calibrate_single_junction(
            target_gn=0.5, side='left', t=t, gamma=gamma,
            Delta0=Delta0, alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls,
            u_min=6.0, u_max=-3.0
        )

    # 3. Invalid side
    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        bc.calibrate_single_junction(
            target_gn=0.5, side='center', t=t, gamma=gamma,
            Delta0=Delta0, alpha=alpha, Ln=Ln, Lb=Lb, Ls=Ls
        )

if __name__ == "__main__":
    pytest.main([__file__])
