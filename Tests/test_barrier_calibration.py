import pytest
import numpy as np
import kwant
import sys
import os
from omegaconf import OmegaConf

# Add parent directory to path so that src modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.helpers as hp
import src.barrier_calibration as bc
from src.config import PathConfigs


def get_default_test_configs():
    """Loads canonical default configurations for barrier calibration tests."""
    defaults_dir = PathConfigs.PARAMETERS / "Defaults"
    calib_cfg = OmegaConf.load(defaults_dir / "barrier_calibration" / "barrier_calibration.yaml")
    geom_cfg = OmegaConf.load(defaults_dir / "geometry" / "InAs-3micron.yaml")
    return calib_cfg, geom_cfg


def compute_1d_analytic_transmission(t, mu_leads, Ub, Lb, E):
    """
    Exact analytical transfer-matrix calculation for 1D tight-binding transmission
    through Lb rectangular barrier sites of height Ub between leads with hopping t.
    """
    cos_k = (2.0 * t - mu_leads - E) / (2.0 * t)
    assert abs(cos_k) < 1.0, f"Energy {E} is outside lead band"
    k = np.arccos(cos_k)
    
    eps_b = 2.0 * t + Ub
    M_site = np.array([[(eps_b - E) / t, -1.0],
                       [1.0, 0.0]], dtype=complex)
    M_total = np.linalg.matrix_power(M_site, Lb)
    
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
    calib_cfg, geom_cfg = get_default_test_configs()
    t = 16.565
    alpha = 1.4
    mu_leads = t
    mu = 0.5
    Vdisx = np.zeros(geom_cfg.Ls)
    
    u_values = np.linspace(-3.0, 6.0, 19)
    
    # Left junction sweep (varying u_left, keeping u_right at u_min)
    gn_l_values = [
        bc.calculate_junction_conductances(
            u_left=u, u_right=calib_cfg.u_min, mu=mu,
            calibration_cfg=calib_cfg, geometry_cfg=geom_cfg,
            Vdisx=Vdisx, t=t, alpha=alpha, mu_leads=mu_leads
        )[0]
        for u in u_values
    ]
    diffs_l = np.diff(gn_l_values)
    assert np.all(diffs_l < 0), f"G_LL is not strictly monotonic! diffs = {diffs_l}"
    assert gn_l_values[0] > 0.90, f"Upper bracket failed for left: {gn_l_values[0]} <= 0.90"
    assert gn_l_values[-1] < 0.35, f"Lower bracket failed for left: {gn_l_values[-1]} >= 0.35"

    # Right junction sweep (varying u_right, keeping u_left at u_min)
    gn_r_values = [
        bc.calculate_junction_conductances(
            u_left=calib_cfg.u_min, u_right=u, mu=mu,
            calibration_cfg=calib_cfg, geometry_cfg=geom_cfg,
            Vdisx=Vdisx, t=t, alpha=alpha, mu_leads=mu_leads
        )[1]
        for u in u_values
    ]
    diffs_r = np.diff(gn_r_values)
    assert np.all(diffs_r < 0), f"G_RR is not strictly monotonic! diffs = {diffs_r}"
    assert gn_r_values[0] > 0.90, f"Upper bracket failed for right: {gn_r_values[0]} <= 0.90"
    assert gn_r_values[-1] < 0.35, f"Lower bracket failed for right: {gn_r_values[-1]} >= 0.35"


def test_target_conductance_inversion():
    """
    Test calibration for the standard targets:
    G_N in {0.35, 0.62, 0.90} e^2/h.
    Assert |G_N(U_c) - G_target| < 0.005 e^2/h for both cutters,
    and verify left/right spatial symmetry on a clean wire.
    """
    calib_cfg, geom_cfg = get_default_test_configs()
    t = 16.565
    alpha = 1.4
    mu_leads = t
    Vdisx = np.zeros(geom_cfg.Ls)
    
    targets = [0.35, 0.62, 0.90]
    mu_points = np.linspace(calib_cfg.mu_min, calib_cfg.mu_max, calib_cfg.n_mu_points)
    
    for target in targets:
        pair = bc.calibrate_symmetric_pair(
            target_gn=target,
            calibration_cfg=calib_cfg,
            geometry_cfg=geom_cfg,
            Vdisx=Vdisx,
            t=t,
            alpha=alpha,
            mu_leads=mu_leads,
            mu_points=mu_points,
        )
        u_l, u_r = pair[0], pair[1]

        # Spatial inversion symmetry on clean wire: UL* == UR*
        assert abs(u_l - u_r) < 1e-4, f"Asymmetry on clean wire: left={u_l}, right={u_r}"

        # Joint evaluation with both calibrated barriers active
        joint_gns = [
            bc.calculate_junction_conductances(
                u_left=u_l, u_right=u_r, mu=mu_val,
                calibration_cfg=calib_cfg, geometry_cfg=geom_cfg,
                Vdisx=Vdisx, t=t, alpha=alpha, mu_leads=mu_leads
            )
            for mu_val in mu_points
        ]
        med_l = np.median([g[0] for g in joint_gns])
        med_r = np.median([g[1] for g in joint_gns])

        assert abs(med_l - target) < 0.005, f"Left calibration failed for target {target}: got {med_l}"
        assert abs(med_r - target) < 0.005, f"Right calibration failed for target {target}: got {med_r}"


def test_calibrate_wire_cutters_table_structure():
    """
    Test calibrate_wire_cutters returns the expected [G_N, [UL, UR]] structure.
    """
    calib_cfg, geom_cfg = get_default_test_configs()
    calib_cfg.target_gns = [0.35, 0.90]  # small subset for quick test
    t = 16.565
    alpha = 1.4
    mu_leads = t
    Vdisx = np.zeros(geom_cfg.Ls)

    mu_points = np.linspace(calib_cfg.mu_min, calib_cfg.mu_max, calib_cfg.n_mu_points)
    table = bc.calibrate_wire_cutters(
        targets=calib_cfg.target_gns,
        calibration_cfg=calib_cfg,
        geometry_cfg=geom_cfg,
        Vdisx=Vdisx,
        t=t,
        alpha=alpha,
        mu_leads=mu_leads,
        mu_points=mu_points,
    )

    assert len(table.G_n) == 2
    assert np.isclose(table.G_n.values[0], 0.35)
    assert np.isclose(table.G_n.values[1], 0.90)
    assert "side" in table.dims
    assert len(table.side) == 2


def test_clean_wire_analytical_transfer_matrix():
    """
    Test that Kwant 1D scalar transmission matches the exact analytical transfer matrix
    formula to machine precision (atol=1e-12).
    """
    t = 16.529
    mu_leads = t
    Ub = 2.0
    Lb = 3
    E = 0.5
    
    T_analytic = compute_1d_analytic_transmission(t=t, mu_leads=mu_leads, Ub=Ub, Lb=Lb, E=E)
    
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
    with barrier tuned to target G_N exhibits exact 2 e^2/h zero-bias conductance.
    Calibrates u_c on a 50-site wire for fast unit test execution, and evaluates
    zero-bias conductance on full Ls=300 to eliminate finite-size Majorana overlap splitting.
    """
    calib_cfg, geom_cfg = get_default_test_configs()
    geom_cfg_fast = OmegaConf.create(dict(geom_cfg))
    geom_cfg_fast.Ls = 50
    t = 16.565
    alpha = 1.4
    mu_leads = t
    mu = 0.2
    V_z_topo = 0.6
    Vdisx_fast = np.zeros(geom_cfg_fast.Ls)
    
    mu_points = [mu]
    u_c = bc.calibrate_single_barrier(
        target_gn=0.5, side='left',
        calibration_cfg=calib_cfg, geometry_cfg=geom_cfg_fast,
        Vdisx=Vdisx_fast, t=t, alpha=alpha, mu_leads=mu_leads,
        mu_points=mu_points,
    )
    
    # Positive control: Topological state on Ls=300 (V_z = 0.6)
    syst_topo = hp.build_system_exact(
        t=t, mu=mu, gamma=geom_cfg.gamma, Delta0=geom_cfg.Delta_0, V_z=V_z_topo,
        alpha=alpha, Ln=geom_cfg.Ln, Lb=geom_cfg.Lb, Ls=geom_cfg.Ls, mu_leads=mu_leads,
        barrier_l=u_c, barrier_r=5.0,
        closed=False
    )
    G_topo = hp.calc_conductance_matrix_exact(syst_topo, E=0.0)
    g_zero_bias = G_topo[0, 0]
    
    np.testing.assert_allclose(
        g_zero_bias, 2.0, atol=5e-3,
        err_msg=f"Topological zero-bias conductance {g_zero_bias} is not 2.0 e^2/h"
    )


def test_error_handling():
    """
    Test error handling:
    1. Unreachable targets (e.g. G_N = 2.5 or -0.5 e^2/h) raise ValueError.
    2. Inverted brackets (u_min >= u_max) raise ValueError.
    3. Invalid side parameter raises ValueError.
    4. Vdisx length mismatch raises ValueError.
    """
    calib_cfg, geom_cfg = get_default_test_configs()
    t = 16.565
    alpha = 1.4
    mu_leads = t
    Vdisx = np.zeros(geom_cfg.Ls)
    
    mu_points = np.linspace(calib_cfg.mu_min, calib_cfg.mu_max, calib_cfg.n_mu_points)

    # 1. Unreachable / physical bounds
    for invalid_target in [-0.5, 2.5]:
        with pytest.raises(ValueError, match="out of physical bounds"):
            bc.calibrate_symmetric_pair(
                target_gn=invalid_target,
                calibration_cfg=calib_cfg, geometry_cfg=geom_cfg,
                Vdisx=Vdisx, t=t, alpha=alpha, mu_leads=mu_leads,
                mu_points=mu_points,
            )

    # 2. Inverted bracket
    calib_inverted = OmegaConf.create(dict(calib_cfg))
    calib_inverted.u_min = 6.0
    calib_inverted.u_max = -3.0
    with pytest.raises(ValueError, match="u_min must be strictly less than u_max"):
        bc.calibrate_symmetric_pair(
            target_gn=0.5,
            calibration_cfg=calib_inverted, geometry_cfg=geom_cfg,
            Vdisx=Vdisx, t=t, alpha=alpha, mu_leads=mu_leads,
            mu_points=mu_points,
        )

    # 3. Invalid side
    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        bc.calibrate_single_barrier(
            target_gn=0.5, side='center',
            calibration_cfg=calib_cfg, geometry_cfg=geom_cfg,
            Vdisx=Vdisx, t=t, alpha=alpha, mu_leads=mu_leads,
            mu_points=mu_points,
        )

    # 4. Vdisx length mismatch
    wrong_vdis = np.zeros(geom_cfg.Ls + 10)
    with pytest.raises(ValueError, match="Vdisx length mismatch"):
        bc.calculate_junction_conductances(
            u_left=0.0, u_right=0.0, mu=0.5,
            calibration_cfg=calib_cfg, geometry_cfg=geom_cfg,
            Vdisx=wrong_vdis, t=t, alpha=alpha, mu_leads=mu_leads
        )


def test_exact_vs_renormalized_conductance_agreement_disordered():
    """
    Verifies that for a disordered nanowire (Vdisx != 0):
    1. calibrate_single_barrier tunes UL and UR independently to match target G_N with custom mu_points.
    2. At above-gap bias (E = 0.5 meV), both calibrated barriers yield G_N near target.
    3. At zero bias (E = 0.0), the exact frequency-dependent Hamiltonian (build_system_exact)
       and the low-energy renormalized Hamiltonian (build_system) yield identical conductance matrices.
    """
    calib_cfg, geom_cfg = get_default_test_configs()
    # Fast 50-site wire for quick unit test execution
    geom_cfg_fast = OmegaConf.create(dict(geom_cfg))
    geom_cfg_fast.Ls = 50
    t = 16.565
    alpha = 1.4
    mu_leads = t
    mu = 2.0

    # Generate reproducible non-zero spatial disorder
    np.random.seed(42)
    vdis = np.random.uniform(-0.5, 0.5, size=geom_cfg_fast.Ls)

    # 1. Calibrate on single mu point
    ul = bc.calibrate_single_barrier(
        target_gn=0.5, side="left", calibration_cfg=calib_cfg, geometry_cfg=geom_cfg_fast,
        Vdisx=vdis, t=t, alpha=alpha, mu_leads=mu_leads, mu_points=[mu]
    )
    ur = bc.calibrate_single_barrier(
        target_gn=0.5, side="right", calibration_cfg=calib_cfg, geometry_cfg=geom_cfg_fast,
        Vdisx=vdis, t=t, alpha=alpha, mu_leads=mu_leads, mu_points=[mu]
    )

    # 2. Above-gap transmission check at E=0.5 meV
    syst_exact = hp.build_system_exact(
        t=t, mu=mu, gamma=geom_cfg_fast.gamma, Delta0=geom_cfg_fast.Delta_0, V_z=0.0,
        alpha=alpha, Ls=geom_cfg_fast.Ls, Vdisx=vdis, barrier_l=ul, barrier_r=ur,
        Ln=geom_cfg_fast.Ln, Lb=geom_cfg_fast.Lb, mu_leads=mu_leads
    )
    G_above_gap = hp.calc_conductance_matrix_exact(syst_exact, E=0.5)
    assert abs(G_above_gap[0, 0] - 0.5) < 0.05, f"Left above-gap conductance {G_above_gap[0, 0]} diverges from 0.5"
    assert abs(G_above_gap[1, 1] - 0.5) < 0.05, f"Right above-gap conductance {G_above_gap[1, 1]} diverges from 0.5"

    # 3. Exact vs Low-Energy Renormalized Zero-Bias Conductance Agreement
    G_exact_0 = hp.calc_conductance_matrix_exact(syst_exact, E=0.0)

    syst_low = hp.build_system(
        t=t, mu=mu, gamma=geom_cfg_fast.gamma, Delta0=geom_cfg_fast.Delta_0, V_z=0.0,
        alpha=alpha, Ls=geom_cfg_fast.Ls, Vdisx=vdis, barrier_l=ul, barrier_r=ur,
        Ln=geom_cfg_fast.Ln, Lb=geom_cfg_fast.Lb, mu_leads=mu_leads
    )
    G_low_0 = hp.calc_conductance_matrix(syst_low, eng=0.0)

    np.testing.assert_allclose(
        G_low_0, G_exact_0, atol=1e-5,
        err_msg="Low-energy renormalized and exact zero-bias conductances diverge!"
    )


if __name__ == "__main__":
    pytest.main([__file__])
