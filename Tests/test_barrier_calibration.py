import pytest
import numpy as np
import xarray as xr
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
    Test Stage 1 calibration for targets G_N in {0.35, 0.62, 0.90} e^2/h.
    Assert |G_N(U_c) - G_target| < 0.005 e^2/h for both cutters,
    and verify left/right spatial symmetry on a clean wire.
    """
    calib_cfg, geom_cfg = get_default_test_configs()
    t = 16.565
    alpha = 1.4
    mu_leads = t
    Vdisx = np.zeros(geom_cfg.Ls)

    targets = [0.35, 0.62, 0.90]
    mu_points = np.linspace(calib_cfg.mu_min, calib_cfg.mu_max, 5)

    da = bc.calibrate_stage1_wire_cutters(
        nominal_targets=targets,
        calibration_cfg=calib_cfg,
        geometry_cfg=geom_cfg,
        Vdisx=Vdisx,
        t=t,
        alpha=alpha,
        mu_leads=mu_leads,
        mu_points=mu_points,
    )

    for target in targets:
        u_l = float(da.sel(G_n=target, barrier_side="left").item())
        u_r = float(da.sel(G_n=target, barrier_side="right").item())

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
    Test calibrate_stage1_wire_cutters returns the expected ('G_n', 'barrier_side') DataArray.
    """
    calib_cfg, geom_cfg = get_default_test_configs()
    t = 16.565
    alpha = 1.4
    mu_leads = t
    Vdisx = np.zeros(geom_cfg.Ls)

    mu_points = np.linspace(calib_cfg.mu_min, calib_cfg.mu_max, 5)
    table = bc.calibrate_stage1_wire_cutters(
        nominal_targets=[0.35, 0.90],
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
    assert table.dims == ("G_n", "barrier_side")
    assert len(table.barrier_side) == 2


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
    u_top_l, g_max_l = bc._find_conductance_top(
        "left", calib_cfg, geom_cfg_fast, Vdisx_fast, t, alpha, mu_leads, mu_points
    )
    u_c = bc._solve_repulsive_barrier(
        target_gn=0.5, side="left", u_top=u_top_l, g_max=g_max_l,
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
    
    mu_points = np.linspace(calib_cfg.mu_min, calib_cfg.mu_max, 5)

    # 1. Unreachable / physical bounds
    for invalid_target in [-0.5, 2.5]:
        with pytest.raises(ValueError, match="outside reachable"):
            bc._solve_repulsive_barrier(
                target_gn=invalid_target, side="left", u_top=0.0, g_max=1.0,
                calibration_cfg=calib_cfg, geometry_cfg=geom_cfg,
                Vdisx=Vdisx, t=t, alpha=alpha, mu_leads=mu_leads,
                mu_points=mu_points,
            )

    # 2. Inverted bracket (u_top >= u_max)
    calib_inverted = OmegaConf.create(dict(calib_cfg))
    calib_inverted.u_max = -5.0
    with pytest.raises(ValueError, match="must be strictly less than u_max"):
        bc._solve_repulsive_barrier(
            target_gn=0.5, side="left", u_top=0.0, g_max=1.0,
            calibration_cfg=calib_inverted, geometry_cfg=geom_cfg,
            Vdisx=Vdisx, t=t, alpha=alpha, mu_leads=mu_leads,
            mu_points=mu_points,
        )

    # 3. Invalid side
    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        bc._find_conductance_top(
            side="center",
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

    # 1. Calibrate on single mu point via _find_conductance_top + _solve_repulsive_barrier
    u_top_l, g_max_l = bc._find_conductance_top(
        "left", calib_cfg, geom_cfg_fast, vdis, t, alpha, mu_leads, [mu]
    )
    u_top_r, g_max_r = bc._find_conductance_top(
        "right", calib_cfg, geom_cfg_fast, vdis, t, alpha, mu_leads, [mu]
    )
    ul = bc._solve_repulsive_barrier(
        target_gn=0.5, side="left", u_top=u_top_l, g_max=g_max_l,
        calibration_cfg=calib_cfg, geometry_cfg=geom_cfg_fast,
        Vdisx=vdis, t=t, alpha=alpha, mu_leads=mu_leads, mu_points=[mu]
    )
    ur = bc._solve_repulsive_barrier(
        target_gn=0.5, side="right", u_top=u_top_r, g_max=g_max_r,
        calibration_cfg=calib_cfg, geometry_cfg=geom_cfg_fast,
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


def test_check_wire_health():
    """
    Test pre-health check at U = u_max (100 meV):
    1. Healthy wire pinches off below pinch_off_tol and passes.
    2. Unhealthy wire (conductance > pinch_off_tol) raises UnhealthyWireError.
    """
    calib_cfg, geom_cfg = get_default_test_configs()
    geom_cfg_fast = OmegaConf.create(dict(geom_cfg))
    geom_cfg_fast.Ls = 50
    t = 16.565
    alpha = 1.4
    mu_leads = t
    Vdisx = np.zeros(geom_cfg_fast.Ls)
    mu_points = np.array([0.0, 1.0, 2.0])

    g_left, g_right = bc.check_wire_health(
        calibration_cfg=calib_cfg,
        geometry_cfg=geom_cfg_fast,
        Vdisx=Vdisx,
        t=t,
        alpha=alpha,
        mu_leads=mu_leads,
        mu_points=mu_points,
    )
    assert g_left <= calib_cfg.pinch_off_tol
    assert g_right <= calib_cfg.pinch_off_tol

    # Force failure with impossibly tight tolerance or open U_max
    calib_bad = OmegaConf.create(dict(calib_cfg))
    calib_bad.u_max = 0.0
    with pytest.raises(bc.UnhealthyWireError):
        bc.check_wire_health(
            calibration_cfg=calib_bad,
            geometry_cfg=geom_cfg_fast,
            Vdisx=Vdisx,
            t=t,
            alpha=alpha,
            mu_leads=mu_leads,
            mu_points=mu_points,
        )


def test_calibrate_stage1_wire_cutters_truncation_and_schema():
    """
    Test calibrate_stage1_wire_cutters:
    1. Returns xr.DataArray named 'stage1_cutters' with dims ('G_n', 'barrier_side').
    2. Truncates nominal targets above G_top and appends round(G_top, 2).
    3. All G_n coordinates are rounded to 2 decimal places.
    """
    calib_cfg, geom_cfg = get_default_test_configs()
    geom_cfg_fast = OmegaConf.create(dict(geom_cfg))
    geom_cfg_fast.Ls = 50
    t = 16.565
    alpha = 1.4
    mu_leads = t
    Vdisx = np.zeros(geom_cfg_fast.Ls)
    mu_points = np.array([0.0, 1.0, 2.0])

    # Include an unreachable target (2.05 > 2.0 e^2/h) to exercise truncation + G_top append
    nominal_targets = np.array([0.10, 0.50, 2.05])
    da = bc.calibrate_stage1_wire_cutters(
        nominal_targets=nominal_targets,
        calibration_cfg=calib_cfg,
        geometry_cfg=geom_cfg_fast,
        Vdisx=Vdisx,
        t=t,
        alpha=alpha,
        mu_leads=mu_leads,
        mu_points=mu_points,
    )

    assert isinstance(da, xr.DataArray)
    assert da.name == "stage1_cutters"
    assert da.dims == ("G_n", "barrier_side")
    assert list(da.coords["barrier_side"].values) == ["left", "right"]
    assert not np.isnan(da.values).any()

    gns = da.coords["G_n"].values
    np.testing.assert_allclose(gns, np.round(gns, 2), atol=1e-12)
    assert 2.05 not in np.round(gns, 2)
    assert np.isclose(gns[0], 0.10)
    assert np.isclose(gns[1], 0.50)
    assert len(gns) >= 3  # [0.10, 0.50, floor_2dec(G_top)]

    # Verify Option B: when g_top is within sweep_gnres (0.042) of last kept target, keep larger g_top
    g_top = float(da.attrs["g_top"])
    near_top = round(g_top - 0.02, 2)
    da_opt_b = bc.calibrate_stage1_wire_cutters(
        nominal_targets=np.array([0.10, near_top, 2.05]),
        calibration_cfg=calib_cfg,
        geometry_cfg=geom_cfg_fast,
        Vdisx=Vdisx,
        t=t,
        alpha=alpha,
        mu_leads=mu_leads,
        mu_points=mu_points,
    )
    np.testing.assert_allclose(da_opt_b.coords["G_n"].values, [0.10, g_top], atol=1e-12)


def test_calibrate_stage2_wire_cutters_spacing_and_schema():
    """
    Test calibrate_stage2_wire_cutters:
    1. Returns xr.DataArray named 'stage2_cutters' with dims ('G_n', 'chemical_potential', 'barrier_side').
    2. Uses exact nominal stage2_target_Gns when reachable across all mu.
    3. When max(stage2_target_Gns) > G_top, reconstructs evenly spaced targets on [stage2_gn_min, G_top]
       and decrements n_c until min(diff(G_n)) >= stage2_min_gn_spacing.
    """
    calib_cfg, geom_cfg = get_default_test_configs()
    geom_cfg_fast = OmegaConf.create(dict(geom_cfg))
    geom_cfg_fast.Ls = 50
    t = 16.565
    alpha = 1.4
    mu_leads = t
    Vdisx = np.zeros(geom_cfg_fast.Ls)
    mu_points = np.array([0.004, 0.504])

    # Case 1: Reachable nominal stage2_target_Gns -> used verbatim
    da_healthy = bc.calibrate_stage2_wire_cutters(
        calibration_cfg=calib_cfg,
        geometry_cfg=geom_cfg_fast,
        Vdisx=Vdisx,
        t=t,
        alpha=alpha,
        mu_leads=mu_leads,
        mu_points=mu_points,
    )
    assert isinstance(da_healthy, xr.DataArray)
    assert da_healthy.name == "stage2_cutters"
    assert da_healthy.dims == ("G_n", "chemical_potential", "barrier_side")
    assert list(da_healthy.coords["barrier_side"].values) == ["left", "right"]
    assert not np.isnan(da_healthy.values).any()
    np.testing.assert_allclose(da_healthy.coords["G_n"].values, calib_cfg.stage2_target_Gns, atol=1e-12)
    np.testing.assert_allclose(da_healthy.coords["chemical_potential"].values, np.round(mu_points, 2), atol=1e-12)

    # Case 2: Conductance-limited wire (max nominal target unreachable -> [0.01, g_top] rescaling)
    calib_limited = OmegaConf.create(dict(calib_cfg))
    calib_limited.stage2_target_Gns = [0.35, 0.49, 0.62, 0.76, 2.05]
    calib_limited.stage2_gn_min = 0.01
    calib_limited.stage2_min_gn_spacing = 0.045

    da_limited = bc.calibrate_stage2_wire_cutters(
        calibration_cfg=calib_limited,
        geometry_cfg=geom_cfg_fast,
        Vdisx=Vdisx,
        t=t,
        alpha=alpha,
        mu_leads=mu_leads,
        mu_points=mu_points,
    )
    gns_limited = da_limited.coords["G_n"].values
    assert np.isclose(gns_limited[0], 0.01)
    assert np.isclose(gns_limited[-1], da_limited.attrs["g_top"])
    assert np.min(np.diff(gns_limited)) >= calib_limited.stage2_min_gn_spacing - 1e-12


if __name__ == "__main__":
    pytest.main([__file__])
