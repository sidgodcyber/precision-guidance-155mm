"""
Unit tests for the 6-DOF simulator.

Everything here has an answer that can be checked against something other
than the code under test: published ISA table values, closed-form vacuum
ballistics, algebraic identities, or the independently measured numbers in
BRL Memorandum Report 1582.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim import aerodata, atmosphere as atm, diagnostics as dg, dynamics as dyn
from sim import frames, integrate as ig, projectile as pr


# ===========================================================================
# frames: quaternion algebra and rotation matrices
# ===========================================================================
def test_quat_from_euler_roundtrip():
    for psi in (-2.0, -0.3, 0.0, 0.7, 2.9):
        for theta in (-1.2, -0.4, 0.0, 0.4, 1.2):
            for phi in (-3.0, -0.5, 0.0, 1.1, 3.0):
                q = frames.quat_from_euler(psi, theta, phi)
                p2, t2, f2 = frames.euler_from_quat(q)
                # Compare through the rotation matrix: Euler triples are not
                # unique but the rotation they name is.
                r1 = frames.dcm_from_quat(q)
                r2 = frames.dcm_from_quat(frames.quat_from_euler(p2, t2, f2))
                assert np.allclose(r1, r2, atol=1e-12)


def test_dcm_is_orthonormal_with_unit_determinant():
    rng = np.random.default_rng(20260903)
    for _ in range(200):
        q = frames.quat_normalize(rng.normal(size=4))
        R = frames.dcm_from_quat(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-13)
        assert abs(np.linalg.det(R) - 1.0) < 1e-13


def test_positive_pitch_is_nose_up():
    """Z is down, so a positive theta must lift the nose (negative Z)."""
    q = frames.quat_from_euler(0.0, math.radians(30.0), 0.0)
    nose = frames.dcm_from_quat(q) @ np.array([1.0, 0.0, 0.0])
    assert nose[0] == pytest.approx(math.cos(math.radians(30.0)))
    assert nose[2] == pytest.approx(-math.sin(math.radians(30.0)))
    assert nose[2] < 0.0


def test_yaw_is_positive_to_the_right():
    """Positive yaw about +Z (down) must swing the nose toward +Y (east/right)."""
    q = frames.quat_from_euler(math.radians(30.0), 0.0, 0.0)
    nose = frames.dcm_from_quat(q) @ np.array([1.0, 0.0, 0.0])
    assert nose[1] > 0.0


def test_quat_multiply_matches_matrix_composition():
    rng = np.random.default_rng(7)
    for _ in range(100):
        a = frames.quat_normalize(rng.normal(size=4))
        b = frames.quat_normalize(rng.normal(size=4))
        assert np.allclose(
            frames.dcm_from_quat(frames.quat_multiply(a, b)),
            frames.dcm_from_quat(a) @ frames.dcm_from_quat(b),
            atol=1e-12,
        )


def test_quat_conjugate_inverts_rotation():
    rng = np.random.default_rng(11)
    q = frames.quat_normalize(rng.normal(size=4))
    v = rng.normal(size=3)
    back = frames.rotate_earth_to_body(q, frames.rotate_body_to_earth(q, v))
    assert np.allclose(back, v, atol=1e-13)
    assert np.allclose(
        frames.dcm_from_quat(frames.quat_conjugate(q)), frames.dcm_from_quat(q).T, atol=1e-13
    )


def test_quat_derivative_gives_the_right_rotation_rate():
    """
    Integrating qdot for a constant body rate about x must precess the body
    frame at exactly that rate.
    """
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([2.0, 0.0, 0.0])
    dt = 1e-5
    for _ in range(100000):
        q = frames.quat_normalize(q + dt * frames.quat_derivative(q, omega))
    _, _, phi = frames.euler_from_quat(q)
    expected = (2.0 * 1.0) % (2.0 * math.pi)
    assert abs(((phi - expected + math.pi) % (2 * math.pi)) - math.pi) < 1e-4


def test_quat_derivative_preserves_norm_to_first_order():
    """qdot must be orthogonal to q, so the norm is conserved."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = frames.quat_normalize(rng.normal(size=4))
        w = rng.normal(size=3)
        assert abs(float(np.dot(q, frames.quat_derivative(q, w)))) < 1e-14


# ===========================================================================
# atmosphere: against published U.S. Standard Atmosphere 1976 values
# ===========================================================================
def _geometric_from_geopotential(h):
    return atm.R_EARTH * h / (atm.R_EARTH - h)


@pytest.mark.parametrize(
    "h_geopot,T,p,rho",
    [
        (0.0, 288.15, 101325.0, 1.225),
        (11000.0, 216.65, 22632.1, 0.363918),
        (15000.0, 216.65, 12044.6, 0.193674),
        (20000.0, 216.65, 5474.89, 0.0880349),
    ],
)
def test_isa_matches_published_tables(h_geopot, T, p, rho):
    st = atm.isa_atmosphere(_geometric_from_geopotential(h_geopot))
    assert st.temperature == pytest.approx(T, rel=1e-5)
    assert st.pressure == pytest.approx(p, rel=1e-5)
    assert st.density == pytest.approx(rho, rel=1e-5)


def test_sea_level_sound_speed():
    assert atm.isa_atmosphere(0.0).sound_speed == pytest.approx(340.294, abs=0.01)


def test_isa_scalars_and_dataclass_agree():
    for h in (-100.0, 0.0, 1234.5, 11019.0, 25000.0):
        t, p, rho, a = atm.isa_scalars(h)
        st = atm.isa_atmosphere(h)
        assert (t, p, rho, a) == (st.temperature, st.pressure, st.density, st.sound_speed)


def test_gravity_falls_off_with_altitude():
    assert atm.gravity(0.0) == pytest.approx(atm.G0)
    # ~0.4 % lighter at a 12 km apogee, which is metres at the target.
    assert atm.gravity(12000.0) / atm.G0 == pytest.approx(0.99624, abs=1e-4)


def test_wind_sign_convention():
    """
    Wind is the VELOCITY OF THE AIR. A wind blowing FROM the north is a
    negative X component, and must reduce the airspeed of a shell flying
    north... no: it is a head wind, so it INCREASES airspeed relative to air.
    """
    w = atm.constant_wind(-10.0, 0.0)
    assert w(0.0)[0] == -10.0
    v_ground = np.array([300.0, 0.0, 0.0])
    v_rel = v_ground - np.array(w(0.0))
    assert v_rel[0] == 310.0  # head wind: airspeed exceeds ground speed


def test_coriolis_is_perpendicular_to_velocity():
    om = atm.earth_rate_ned(math.radians(45.0))
    v = np.array([500.0, 10.0, -100.0])
    a = atm.coriolis_acceleration(v, om)
    assert abs(float(np.dot(a, v))) < 1e-9


def test_earth_rate_reduces_to_spec_form_at_zero_azimuth():
    lat = math.radians(37.0)
    om = atm.earth_rate_ned(lat, 0.0)
    assert np.allclose(
        om, atm.OMEGA_EARTH * np.array([math.cos(lat), 0.0, -math.sin(lat)]), atol=1e-18
    )


# ===========================================================================
# aerodata: interpolation and provenance invariants
# ===========================================================================
def test_interpolation_is_exact_at_table_knots():
    table = aerodata.make_m107_table()
    mach, values = table.as_arrays()
    for i, m in enumerate(mach):
        c = table.coefficients_at(float(m))
        got = np.array(
            [c.C_X0, c.C_X2, c.C_Nalpha, c.C_Ypalpha, c.C_lp, c.C_Malpha, c.C_mq, c.C_Mpalpha]
        )
        assert np.allclose(got, values[i], atol=1e-12)


def test_midpoint_interpolation_is_linear():
    table = aerodata.make_m107_table()
    mach, values = table.as_arrays()
    mid = 0.5 * (mach[2] + mach[3])
    c = table.coefficients_at(float(mid))
    assert c.C_Malpha == pytest.approx(0.5 * (values[2, 5] + values[3, 5]))


def test_lookup_matches_coefficients_at():
    """The hot path and the readable path must not drift apart."""
    table = aerodata.make_m107_table()
    for m in (0.0, 0.3, 0.55, 0.87, 1.0, 1.03, 1.6, 2.0, 3.0):
        c = table.coefficients_at(m)
        t = table.lookup(m)
        assert t == pytest.approx(
            (c.C_X0, c.C_X2, c.C_Nalpha, c.C_Ypalpha, c.C_lp, c.C_Malpha, c.C_mq, c.C_Mpalpha)
        )


def test_out_of_range_is_held_flat_and_flagged():
    table = aerodata.make_m107_table()
    assert not table.extrapolated_above
    hi = table.coefficients_at(9.0)
    assert table.extrapolated_above
    assert hi.C_Malpha == pytest.approx(table.values[-1, 5])
    lo = table.coefficients_at(-1.0)
    assert table.extrapolated_below
    assert lo.C_X0 == pytest.approx(table.values[0, 0])


def test_coefficient_signs_are_physical():
    """
    The sign conventions of SIXDOFSPEC.md sections 5-6. A regression here is
    exactly the class of bug that still produces a plausible trajectory.
    """
    table = aerodata.make_m107_table()
    for m in np.linspace(0.05, 2.0, 40):
        c = table.coefficients_at(float(m))
        assert c.C_X0 > 0.0, "axial drag must oppose motion"
        assert c.C_X2 > 0.0, "yaw drag must add drag"
        assert c.C_Nalpha > 0.0, "normal force must act along the angle of attack"
        assert c.C_Malpha > 0.0, "overturning moment is DESTABILISING for a spun shell"
        assert c.C_lp < 0.0, "spin damping must damp"
        assert c.C_mq < 0.0, "pitch damping must damp"


def test_centre_of_pressure_lies_inside_the_projectile():
    """
    C_Malpha/C_Nalpha is the CP-to-CG distance in calibres, ahead of the CG.
    If the C_Malpha sign convention were inverted, the implied CP would fall
    off the back of a 4.5 calibre shell.
    """
    table = aerodata.make_m107_table()
    proj = pr.M107
    for m in (0.6, 0.8, 1.0, 1.5, 2.0):
        c = table.coefficients_at(m)
        cp_from_nose = proj.x_cg_calibers - c.C_Malpha / c.C_Nalpha
        assert 0.0 < cp_from_nose < proj.length / proj.diameter


# ---------------------------------------------------------------------------
# Sign-convention pinning. These exist so a future edit cannot silently invert
# a convention, and so that a correct result can never be produced by two
# errors cancelling.
# ---------------------------------------------------------------------------
def test_cnalpha_is_stored_positive_and_negated_exactly_once():
    """
    The source table (ASAT/SPINNER-98) lists C_Nalpha NEGATIVE; BRL MR-1582
    Appendix I lists it POSITIVE and states the convention explicitly:
    "A positive C_Nalpha yields a normal force in the direction of the total
    angle of attack." SIXDOFSPEC.md section 5 uses BRL's convention.

    The flip is applied exactly once, where the table is defined. The force
    law then carries the spec's own leading minus sign. If someone ever
    "fixes" the table sign without removing that minus, this test fails.
    """
    table = aerodata.make_m107_table()
    for m in (0.05, 0.6, 0.8, 1.2, 2.0):
        assert table.coefficients_at(m).C_Nalpha > 0.0

    # And the stored magnitude still matches the source deck.
    raw = aerodata.make_m107_table(measured_cnalpha=False)
    assert raw.coefficients_at(1.20).C_Nalpha == pytest.approx(2.325)

    # Now the end-to-end consequence: nose up must give force up.
    model = _model(environment=pr.Environment(include_coriolis=False))
    q = frames.quat_from_euler(0.0, math.radians(2.0), 0.0)
    y = dyn.pack(np.zeros(3), np.array([600.0, 0.0, 0.0]), q, np.array([1000.0, 0.0, 0.0]))
    st = dyn.aero_state(0.0, y, model)
    assert st.v_rel_body[2] > 0.0        # w > 0, nose pitched up into the flow
    assert st.force_body[2] < 0.0        # force up
    assert st.moment_body[1] > 0.0       # moment further nose-up (destabilising)


def test_magnus_force_coefficient_is_stored_negative():
    """
    C_Ypalpha is NOT flipped. A negative Magnus force coefficient is the
    physically expected result for a spin-stabilised shell (the boundary
    layer asymmetry reverses the naive omega x v direction), and BRL
    independently measured C_Npalpha = -0.15 to -0.55 for this shell family.
    Flipping it to match C_Nalpha would be wrong.
    """
    table = aerodata.make_m107_table()
    for m in (0.05, 0.6, 0.95, 1.5, 2.0):
        assert table.coefficients_at(m).C_Ypalpha < 0.0


def test_axial_column_is_C_A_not_BRL_C_D():
    """
    ASAT tabulates C_A, the AXIAL force coefficient along the body axis, which
    is what the spec's force model consumes. BRL tabulates C_D, drag along the
    velocity vector, and BRL's C_D additionally contains yaw drag that would
    have to be removed first. The two must never be interleaved in one table.

    This pins that no BRL C_D value has been pasted into the C_X0 column.
    """
    rows = aerodata._M107_ROWS
    cx0 = rows[:, 1 + aerodata.COEFFICIENT_NAMES.index("C_X0")]
    brl_m107_cd = [0.1575, 0.1477, 0.1413]           # BRL Table III, total drag
    brl_m101_cd = [0.1393, 0.1362, 0.1451, 0.1408]   # BRL Table II samples
    for v in brl_m107_cd + brl_m101_cd:
        assert not np.any(np.isclose(cx0, v, atol=1e-9)), (
            f"BRL C_D value {v} found in the axial-force column"
        )
    # And the column still matches the ASAT C_A deck at its knots.
    assert cx0[2] == pytest.approx(0.146)   # Mach 0.80
    assert cx0[-1] == pytest.approx(0.294)  # Mach 2.00


# ---------------------------------------------------------------------------
# The subsonic C_Nalpha splice
# ---------------------------------------------------------------------------
def test_measured_correction_matches_the_cluster_means():
    """
    The corrected C_Nalpha must reproduce the BRL full-scale cluster means at
    the cluster centre Mach numbers, because that is how it is constructed.
    """
    table = aerodata.make_m107_table()
    for mach, measured in [(0.812, 1.637), (1.186, 2.573), (1.604, 2.615)]:
        assert table.coefficients_at(mach).C_Nalpha == pytest.approx(
            measured, rel=0.02
        ), f"cluster at Mach {mach}"

    # KNOWN LIMITATION, pinned so it cannot be forgotten: the ASAT deck ends
    # at Mach 2.00 and is held flat above, so the Mach 2.265 cluster (measured
    # 2.953) cannot be represented -- the correction is applied at the table
    # knots, and the highest knot is Mach 2.00 where k = 1.0197. The model
    # therefore returns 2.801 above Mach 2.0, about 5 % below the measurement.
    # No trajectory in the validation ladder goes above Mach 2.01 (charge 8
    # launches there and decelerates immediately), so this is recorded rather
    # than papered over with an invented knot.
    assert table.coefficients_at(2.265).C_Nalpha == pytest.approx(2.801, rel=0.01)
    assert table.mach_max == pytest.approx(2.00)


def test_measured_correction_touches_only_C_Nalpha():
    """
    C_Malpha is NOT corrected: BRL measurement confirms the ASAT values
    (ratio 1.019 +- 0.039, no Mach trend). Nothing but C_Nalpha may move.
    """
    raw = aerodata.make_m107_table(measured_cnalpha=False)
    new = aerodata.make_m107_table()
    for m in (0.3, 0.8, 1.2, 1.5, 2.0, 2.5):
        a, b = raw.coefficients_at(m), new.coefficients_at(m)
        for name in aerodata.COEFFICIENT_NAMES:
            if name != "C_Nalpha":
                assert getattr(a, name) == pytest.approx(getattr(b, name)), name
    # and C_Nalpha really does move
    assert new.coefficients_at(0.8).C_Nalpha != pytest.approx(
        raw.coefficients_at(0.8).C_Nalpha
    )


def test_measured_factor_is_continuous_and_physically_bounded():
    f = aerodata.cnalpha_measured_factor
    kx = [p[0] for p in aerodata.CNALPHA_MEASURED_K]
    ky = [p[1] for p in aerodata.CNALPHA_MEASURED_K]
    for x, y in aerodata.CNALPHA_MEASURED_K:
        assert f(x) == pytest.approx(y)
    assert f(0.0) == pytest.approx(ky[0])     # held flat below
    assert f(5.0) == pytest.approx(ky[-1])    # held flat above
    prev = f(0.0)
    for i in range(1, 401):
        m = i * 0.01
        cur = f(m)
        assert abs(cur - prev) < 0.01, "correction factor must be continuous"
        assert 0.80 < cur < 1.25, "correction factor must stay physical"
        prev = cur


def test_measured_correction_improves_centre_of_pressure():
    """
    The correction is justified on the centre of pressure. Measured against
    the CP implied by every usable full-scale BRL row, it must beat both the
    raw computed deck and the previous subsonic-only splice, and it must land
    at the irreducible row-to-row scatter of the measurements themselves.
    """
    from analysis.brl_reference import (
        M101_TABLE_II_PAIRS, M107_TABLE_III, cp_from_nose,
    )

    rows = [(M, y2, cna, cma) for M, y2, _cd, cma, cna, _q, _p in M107_TABLE_III]
    rows += [(M, y2, cna, cma) for M, y2, cma, cna in M101_TABLE_II_PAIRS]
    rows = [r for r in rows if r[1] <= 25.0]

    raw = aerodata.make_m107_table(measured_cnalpha=False)
    new = aerodata.make_m107_table()

    def cp_rms(table):
        rs = []
        for M, _y2, cna, cma in rows:
            c = table.coefficients_at(M)
            rs.append(cp_from_nose(c.C_Malpha, c.C_Nalpha) - cp_from_nose(cma, cna))
        return (sum(v * v for v in rs) / len(rs)) ** 0.5

    assert cp_rms(new) < cp_rms(raw)
    assert cp_rms(new) < 0.13      # previous subsonic-only splice gave 0.1438
    assert cp_rms(raw) > 0.18


def test_moment_reference_transfer_is_identity_at_the_cg():
    proj = pr.M107
    c = aerodata.make_m107_table().coefficients_at(0.8)
    same = aerodata.transfer_moment_reference(
        c.C_Malpha, c.C_Nalpha, proj.x_cg, proj.x_cg, proj.diameter
    )
    assert same == pytest.approx(c.C_Malpha)


def test_moment_reference_transfer_moves_the_right_way():
    """
    Referencing further forward (toward the nose) must reduce the
    destabilising moment about that station.
    """
    proj = pr.M107
    c = aerodata.make_m107_table().coefficients_at(0.8)
    fwd = aerodata.transfer_moment_reference(
        c.C_Malpha, c.C_Nalpha, proj.x_cg - 0.5 * proj.diameter, proj.x_cg, proj.diameter
    )
    assert fwd < c.C_Malpha


# ===========================================================================
# projectile: physical properties against BRL MR-1582 Table I
# ===========================================================================
def test_m107_inertias_match_brl_table_one():
    p = pr.M107
    assert p.mass == pytest.approx(95.8 * 0.45359237)
    assert p.x_cg_calibers == pytest.approx(2.96)
    assert p.length / p.diameter == pytest.approx(4.5)
    # BRL k_axial^-2 = 7.10, k_transverse^-2 = 0.81
    md2 = p.mass * p.diameter**2
    assert md2 / p.I_axial == pytest.approx(7.10)
    assert md2 / p.I_transverse == pytest.approx(0.81)
    # Independent value from Lim NPS 2016 Table 12: Ixx = 0.1461 kg m^2
    assert p.I_axial == pytest.approx(0.1461, rel=0.01)
    # A long shell must have transverse inertia far above axial.
    assert p.I_transverse > 5.0 * p.I_axial


def test_twist_is_a_tube_property_and_the_nominal_tube_is_the_M185():
    """
    Twist belongs to the gun, not the shell. The nominal M107 model is fired
    from an M185/M199 39-calibre tube, which McCoy states is 1 turn in 20
    calibres, and which is the weapon of firing table FT 155-AM-2 -- the data
    this model is validated against. BRL MR-1582's own firings used the older
    1-in-25 tube, which is why its measured stability factors are lower.
    """
    assert pr.M107.twist_calibers == pytest.approx(pr.TUBES["M185"]) == 20.0
    assert pr.TUBES["M1"] == 25.0
    # ASAT-13 states V0 = 684.3 m/s with p0 = 175.48 rps, implying:
    assert pr.TUBES["ASAT"] == pytest.approx(684.3 / 175.48 / 0.155)
    assert pr.TUBES["ASAT"] == pytest.approx(25.16, abs=0.01)
    # and the ASAT projectile reproduces that stated muzzle spin exactly
    assert pr.M107_ASAT.muzzle_spin(684.3) / (2 * math.pi) == pytest.approx(175.48)


def test_asat_projectile_matches_its_published_specification():
    p = pr.M107_ASAT
    assert p.mass == pytest.approx(43.0)
    assert p.I_axial == pytest.approx(0.144)
    assert p.I_transverse == pytest.approx(1.216)
    assert p.x_cg == pytest.approx(0.459)
    assert p.length == pytest.approx(0.698)
    # ASAT and BRL agree on the mass properties of the same shell
    assert p.I_axial == pytest.approx(pr.M107.I_axial, rel=0.03)
    assert p.I_transverse == pytest.approx(pr.M107.I_transverse, rel=0.07)
    assert p.x_cg_calibers == pytest.approx(pr.M107.x_cg_calibers, rel=0.01)


def test_transverse_inertia_is_not_the_commonly_quoted_wrong_value():
    """
    A figure near 1.79 kg m^2 is often quoted for "a 155 mm shell" but is not
    supported by either source used here: BRL Table I gives 1.289 and ASAT
    gives 1.216. Guard against it being reintroduced.
    """
    assert pr.M107.I_transverse < 1.4
    assert abs(pr.M107.I_transverse - 1.79) > 0.4


def test_muzzle_spin_matches_the_spec_worked_example():
    """SIXDOFSPEC.md section 9: 1-in-20 twist at 827 m/s gives ~1676 rad/s."""
    p = pr.M107
    assert p.muzzle_spin(827.0) == pytest.approx(1676.0, rel=1e-3)
    assert p.muzzle_spin(684.0) > 0.0  # right-hand rifling is positive


def test_reference_area():
    assert pr.M107.reference_area == pytest.approx(math.pi * 0.155**2 / 4)


def test_mils_and_degrees_agree():
    a = pr.LaunchConditions.from_mils(684.0, 1600.0)
    assert a.qe_degrees == pytest.approx(90.0)
    b = pr.LaunchConditions.from_degrees(684.0, 45.0)
    assert b.qe_mils == pytest.approx(800.0)


# ===========================================================================
# dynamics: state handling and force/moment signs
# ===========================================================================
def _model(**kw):
    return dyn.FlightModel(projectile=pr.M107, aero=aerodata.make_m107_table(), **kw)


def test_pack_unpack_roundtrip():
    r = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    q = np.array([0.5, 0.5, 0.5, 0.5])
    w = np.array([7.0, 8.0, 9.0])
    y = dyn.pack(r, v, q, w)
    assert y.shape == (dyn.STATE_SIZE,)
    r2, v2, q2, w2 = dyn.unpack(y)
    assert np.allclose(r2, r) and np.allclose(v2, v)
    assert np.allclose(q2, q) and np.allclose(w2, w)


def test_initial_state_is_consistent():
    launch = pr.LaunchConditions.from_degrees(684.0, 45.0)
    y0 = dyn.initial_state(pr.M107, launch)
    r, v, q, w = dyn.unpack(y0)
    assert np.allclose(r, 0.0)
    assert np.linalg.norm(v) == pytest.approx(684.0)
    assert v[0] > 0.0 and v[2] < 0.0  # downrange and climbing
    assert np.linalg.norm(q) == pytest.approx(1.0)
    assert w[0] == pytest.approx(pr.M107.muzzle_spin(684.0))
    assert w[1] == 0.0 and w[2] == 0.0


def test_vacuum_derivative_is_pure_gravity():
    model = _model(
        aero_enabled=False,
        environment=pr.Environment(include_coriolis=False, include_inverse_square_gravity=False),
    )
    y0 = dyn.initial_state(pr.M107, pr.LaunchConditions.from_degrees(684.0, 30.0))
    d = dyn.derivative(0.0, y0, model)
    assert np.allclose(d[3:6], [0.0, 0.0, atm.G0], atol=1e-12)


def test_derivative_does_not_mutate_its_input():
    model = _model()
    y0 = dyn.initial_state(pr.M107, pr.LaunchConditions.from_degrees(684.0, 30.0))
    before = y0.copy()
    dyn.derivative(0.0, y0, model)
    assert np.array_equal(y0, before)


def test_derivative_is_deterministic():
    model = _model()
    y0 = dyn.initial_state(pr.M107, pr.LaunchConditions.from_degrees(684.0, 30.0))
    assert np.array_equal(dyn.derivative(0.0, y0, model), dyn.derivative(0.0, y0, model))


def test_nose_up_angle_of_attack_gives_lift_up_and_nose_up_moment():
    """
    The single most important sign test in the package.

    Body pitched nose-up relative to the flow means w > 0. The normal force
    must then push UP (negative body z) while the overturning moment must be
    POSITIVE about body y, i.e. further nose-up -- destabilising, because the
    centre of pressure of a spun shell is ahead of the CG.
    """
    model = _model(environment=pr.Environment(include_coriolis=False))
    # Level flight, nose pitched up 2 degrees relative to a horizontal velocity.
    q = frames.quat_from_euler(0.0, math.radians(2.0), 0.0)
    y = dyn.pack(np.zeros(3), np.array([600.0, 0.0, 0.0]), q, np.array([1000.0, 0.0, 0.0]))
    st = dyn.aero_state(0.0, y, model)
    u, v, w = st.v_rel_body
    assert w > 0.0, "nose-up attitude with horizontal velocity must give w > 0"
    assert st.force_body[2] < 0.0, "normal force must act upward (negative body z)"
    assert st.moment_body[1] > 0.0, "overturning moment must be nose-up (destabilising)"
    assert st.force_body[0] < 0.0, "axial force must oppose forward motion"


def test_spin_damping_always_opposes_spin():
    model = _model(environment=pr.Environment(include_coriolis=False))
    q = frames.quat_from_euler(0.0, 0.0, 0.0)
    for p_spin in (500.0, 1386.0, -1386.0):
        y = dyn.pack(np.zeros(3), np.array([600.0, 0.0, 0.0]), q, np.array([p_spin, 0.0, 0.0]))
        st = dyn.aero_state(0.0, y, model)
        assert st.moment_body[0] * p_spin < 0.0


def test_pitch_damping_always_opposes_pitch_rate():
    model = _model(environment=pr.Environment(include_coriolis=False))
    q = frames.quat_from_euler(0.0, 0.0, 0.0)
    for qrate in (5.0, -5.0):
        y = dyn.pack(np.zeros(3), np.array([600.0, 0.0, 0.0]), q, np.array([0.0, qrate, 0.0]))
        st = dyn.aero_state(0.0, y, model)
        assert st.moment_body[1] * qrate < 0.0


def test_zero_angle_of_attack_gives_no_transverse_force_or_moment():
    model = _model(environment=pr.Environment(include_coriolis=False))
    q = frames.quat_from_euler(0.0, 0.0, 0.0)
    y = dyn.pack(np.zeros(3), np.array([600.0, 0.0, 0.0]), q, np.array([1386.0, 0.0, 0.0]))
    st = dyn.aero_state(0.0, y, model)
    assert abs(st.force_body[1]) < 1e-9 and abs(st.force_body[2]) < 1e-9
    assert abs(st.moment_body[1]) < 1e-9 and abs(st.moment_body[2]) < 1e-9
    assert st.moment_body[0] < 0.0  # spin damping still acts


def test_gyroscopic_cross_terms_are_present():
    """
    With a transverse rate and high spin, the (It - Ix) coupling must feed
    the q equation from r and vice versa. Dropping those terms is a classic
    error that leaves the trajectory looking plausible.
    """
    model = _model(aero_enabled=False, environment=pr.Environment(include_coriolis=False))
    q = frames.quat_from_euler(0.0, 0.0, 0.0)
    y = dyn.pack(np.zeros(3), np.array([600.0, 0.0, 0.0]), q, np.array([1386.0, 0.0, 3.0]))
    d = dyn.derivative(0.0, y, model)
    p = pr.M107
    expected_qdot = (p.I_transverse - p.I_axial) * 3.0 * 1386.0 / p.I_transverse
    assert d[11] == pytest.approx(expected_qdot)
    assert d[10] == pytest.approx(0.0)


def test_control_callback_adds_force_and_moment():
    """The step-4 canard seam: dynamics must accept and apply the callback."""
    dF = np.array([0.0, 100.0, 0.0])
    dM = np.array([0.0, 0.0, 25.0])

    def control(t, y, st):
        return dF, dM

    y0 = dyn.initial_state(pr.M107, pr.LaunchConditions.from_degrees(684.0, 30.0))
    base = dyn.derivative(0.0, y0, _model())
    with_ctl = dyn.derivative(0.0, y0, _model(control=control))
    assert not np.allclose(base, with_ctl)
    # rdot for the extra yaw moment: N / It
    assert with_ctl[12] - base[12] == pytest.approx(25.0 / pr.M107.I_transverse)


def test_control_callback_defaults_to_none_and_changes_nothing():
    model = _model()
    assert model.control is None


# ===========================================================================
# integrate
# ===========================================================================
def test_vacuum_trajectory_matches_the_analytic_parabola():
    """Rung 1, as a test. Must be far better than 0.1 %."""
    env = pr.Environment(include_coriolis=False, include_inverse_square_gravity=False)
    model = _model(aero_enabled=False, environment=env)
    for qe in (20.0, 45.0, 70.0):
        y0 = dyn.initial_state(pr.M107, pr.LaunchConditions.from_degrees(500.0, qe))
        res = ig.integrate(y0, model, dt=1e-3, log_every=100000, t_max=200.0)
        th = math.radians(qe)
        exact = 500.0**2 * math.sin(2 * th) / atm.G0
        assert res.range_m == pytest.approx(exact, rel=1e-6)
        assert res.drift_m == pytest.approx(0.0, abs=1e-9)


def test_impact_is_interpolated_to_exactly_zero_altitude():
    model = _model(aero_enabled=False, environment=pr.Environment(include_coriolis=False))
    y0 = dyn.initial_state(pr.M107, pr.LaunchConditions.from_degrees(400.0, 40.0))
    res = ig.integrate(y0, model, dt=1e-3, log_every=100000, t_max=200.0)
    assert res.terminated == "impact"
    assert abs(res.impact_state[2]) < 1e-9


def test_quaternion_stays_normalised_through_a_spinning_flight():
    """Rung 6. Every logged quaternion must be unit to near machine precision."""
    model = _model()
    y0 = dyn.initial_state(pr.M107, pr.LaunchConditions.from_mils(684.0, 141.6))
    res = ig.integrate(y0, model, dt=5e-4, log_every=500, t_max=60.0)
    norms = np.linalg.norm(res.trajectory.quaternion, axis=1)
    assert np.max(np.abs(norms - 1.0)) < 1e-12
    assert np.all(np.isfinite(res.trajectory.position))


def test_rk4_is_fourth_order_on_attitude_propagation():
    """
    Halving dt must cut the error by roughly 2^4.

    The test problem is ATTITUDE, not range. In vacuum with constant gravity
    the trajectory is a quadratic in t and RK4 integrates it exactly, so a
    range-based order test measures nothing but the linear impact
    interpolation. Attitude under a constant body rate is a genuine
    exponential, RK4 is not exact on it, and it is the part of the state that
    actually forces the small timestep -- so this is both a real order test
    and the one that matters here.
    """
    env = pr.Environment(include_coriolis=False)
    model = _model(aero_enabled=False, environment=env)  # no moments -> omega constant
    spin = 1386.0
    t_end = 0.05
    errs = []
    for dt in (2e-4, 1e-4):
        y = dyn.pack(
            np.zeros(3), np.array([600.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]), np.array([spin, 0.0, 0.0]),
        )
        n = int(round(t_end / dt))
        t = 0.0
        for _ in range(n):
            y = ig.rk4_step(t, y, dt, model)  # deliberately NOT renormalised
            t += dt
        q = frames.quat_normalize(y[6:10])
        exact = frames.quat_from_euler(0.0, 0.0, spin * t_end)
        # angle between the two rotations
        dot = abs(float(np.dot(q, exact)))
        errs.append(2.0 * math.acos(min(1.0, dot)))
    assert errs[0] > 0.0
    ratio = errs[0] / errs[1]
    assert ratio > 10.0, f"expected ~16x error reduction, got {ratio:.2f}"


# ===========================================================================
# diagnostics
# ===========================================================================
def test_gyroscopic_factor_matches_brl_measured_values():
    """
    BRL MR-1582 Table II tabulates its own measured gyroscopic stability
    factor s = 1.69 to 2.27 for the 155 mm M101 fired from a standard tube
    with a twist of 1 turn in 25 calibres, over M 0.57 to 2.41.

    Reproducing that band from the coefficient table is simultaneously a
    check on C_Malpha, on the inertias, and on the per-radian reading of the
    coefficients: a per-degree misreading would give Sg near 0.03.
    """
    shell = pr.M107.perturbed(twist_calibers=25.0)
    table = aerodata.make_m107_table()
    air = atm.isa_atmosphere(0.0)
    values = []
    for mach in (0.6, 0.8, 1.0, 1.5, 2.0, 2.4):
        V = mach * air.sound_speed
        p = shell.muzzle_spin(V)
        c = table.coefficients_at(mach)
        values.append(
            dg.gyroscopic_stability_factor(shell, air.density, V, p, c.C_Malpha)
        )
    assert min(values) > 1.0
    assert 1.4 < min(values) < 2.6
    assert 1.4 < max(values) < 2.8


def test_gyroscopic_factor_scales_as_expected():
    shell = pr.M107
    table = aerodata.make_m107_table()
    air = atm.isa_atmosphere(0.0)
    c = table.coefficients_at(1.0)
    base = dg.gyroscopic_stability_factor(shell, air.density, 340.0, 1000.0, c.C_Malpha)
    # Sg goes as p^2
    assert dg.gyroscopic_stability_factor(
        shell, air.density, 340.0, 2000.0, c.C_Malpha
    ) == pytest.approx(4.0 * base)
    # and inversely with density
    assert dg.gyroscopic_stability_factor(
        shell, 0.5 * air.density, 340.0, 1000.0, c.C_Malpha
    ) == pytest.approx(2.0 * base)


def test_muzzle_is_gyroscopically_stable_at_every_charge():
    for charge, mv in pr.CHARGE_TABLE.items():
        model = _model()
        y0 = dyn.initial_state(pr.M107, pr.LaunchConditions.from_degrees(mv, 45.0))
        st = dg.muzzle_stability(y0, model)
        assert st.Sg > 1.0, f"charge {charge} is gyroscopically unstable"


def test_dynamic_stability_limit_is_infinite_outside_zero_to_two():
    assert math.isinf(dg.dynamic_stability_limit(-0.1))
    assert math.isinf(dg.dynamic_stability_limit(2.5))
    assert dg.dynamic_stability_limit(1.0) == pytest.approx(1.0)


def test_asat_initial_axial_deceleration_matches_the_published_value():
    """
    Validation rung 5b, as a fast unit test: no integration needed, because
    the initial axial deceleration is an instantaneous quantity.

    ASAT-13 section 4.3 states 4.45 g. Aerodynamic drag alone gives 3.77 g;
    adding the component of gravity along the body axis at the stated 44 deg
    launch angle accounts for the rest. Reproducing it to better than 1 %
    checks C_X0, the reference area, the mass and the force scaling at once,
    and identifies what the published figure includes.
    """
    shell = pr.M107_ASAT
    model = dyn.FlightModel(
        projectile=shell,
        aero=aerodata.make_m107_table(),
        environment=pr.Environment(include_coriolis=False),
    )
    launch = pr.LaunchConditions.from_degrees(684.3, 44.0)
    y0 = dyn.initial_state(shell, launch)
    st = dyn.aero_state(0.0, y0, model)

    drag_g = -st.force_body[0] / shell.mass / atm.G0
    gravity_g = math.sin(math.radians(44.0))
    total_g = drag_g + gravity_g

    assert drag_g == pytest.approx(3.77, abs=0.05)
    assert total_g == pytest.approx(4.45, rel=0.01)


def test_subsonic_dynamic_stability_factor_is_negative():
    """
    Documents a real property of this shell rather than guarding a target.

    The Magnus moment term kx^-2 * C_Mpalpha outweighs C_Lalpha at low
    subsonic Mach, driving Sd out of the (0, 2) interval in which the
    criterion Sg > 1/(Sd(2-Sd)) is defined. BRL MR-1582 reports the same
    from measurement: "The 155-mm M101 shell is also dynamically unstable in
    the subsonic and transonic region", and notes the M107 is worse than the
    M101 because of the C_Mpalpha sign change.

    If a future edit makes this positive, that is a change worth noticing --
    it would mean the Magnus moment or the inertias moved materially.
    """
    table = aerodata.make_m107_table()
    p = pr.M107
    c_low = table.coefficients_at(0.60)
    sd_low = dg.dynamic_stability_factor(
        p, c_low.C_Nalpha, c_low.C_X0, c_low.C_Mpalpha, c_low.C_mq
    )
    assert sd_low < 0.0
    assert math.isinf(dg.dynamic_stability_limit(sd_low))

    # ...and it recovers above the transonic, which is what BRL says.
    c_hi = table.coefficients_at(1.50)
    sd_hi = dg.dynamic_stability_factor(
        p, c_hi.C_Nalpha, c_hi.C_X0, c_hi.C_Mpalpha, c_hi.C_mq
    )
    assert 0.0 < sd_hi < 2.0
    assert dg.dynamic_stability_limit(sd_hi) < 1.5


def test_yaw_of_repose_is_positive_and_small():
    shell = pr.M107
    air = atm.isa_atmosphere(0.0)
    c = aerodata.make_m107_table().coefficients_at(1.5)
    dR = dg.yaw_of_repose_estimate(shell, air.density, 500.0, 1300.0, c.C_Malpha, 0.0)
    assert dR > 0.0
    assert math.degrees(dR) < 2.0


# ===========================================================================
# The headline physics result: a right-hand spun shell drifts RIGHT
# ===========================================================================
def test_right_hand_rifling_drifts_right():
    """
    Rung 4, the sign-error detector. Nothing else in the model reveals an
    inverted Magnus or overturning sign as clearly.
    """
    model = _model(environment=pr.Environment.from_degrees(45.0, include_coriolis=False))
    y0 = dyn.initial_state(pr.M107, pr.LaunchConditions.from_mils(684.0, 141.6))
    res = ig.integrate(y0, model, dt=5e-4, log_every=1000, t_max=60.0)
    assert res.drift_m > 0.0, "a right-hand rifled shell must drift RIGHT"
    assert 10.0 < res.drift_m < 200.0


def test_left_hand_rifling_drifts_left():
    """The mirror image. Reversing the twist must reverse the drift."""
    shell = pr.M107.perturbed(twist_calibers=-20.0)
    model = dyn.FlightModel(
        projectile=shell,
        aero=aerodata.make_m107_table(),
        environment=pr.Environment.from_degrees(45.0, include_coriolis=False),
    )
    y0 = dyn.initial_state(shell, pr.LaunchConditions.from_mils(684.0, 141.6))
    res = ig.integrate(y0, model, dt=5e-4, log_every=1000, t_max=60.0)
    assert res.drift_m < 0.0


def test_angle_of_attack_stays_small_in_nominal_flight():
    """Rung 6: if delta grows without bound the shell is tumbling."""
    model = _model()
    y0 = dyn.initial_state(pr.M107, pr.LaunchConditions.from_mils(684.0, 141.6))
    res = ig.integrate(y0, model, dt=5e-4, log_every=200, t_max=60.0)
    assert math.degrees(res.max_total_aoa) < 5.0
