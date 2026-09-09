"""
Canard set and despun nose assembly for a fuze-well guidance kit.

=============================================================================
READ THIS FIRST: EVERY AERODYNAMIC NUMBER IN THIS MODULE IS ESTIMATED
=============================================================================

`sim/aerodata.py` carries measured, cross-checked coefficients for the M107
body: two independent free-flight determinations, agreement stated per Mach
band, provenance per digit. NOTHING IN THIS FILE HAS THAT STATUS.

There is no Missile DATCOM run behind this module, no wind-tunnel entry and no
free-flight range data for any canard on a 155 mm shell. Every canard
coefficient here is produced by a published engineering correlation applied to
a geometry that this project invented. The correlations are named, their
expected accuracy is stated, and `ESTIMATES` below enumerates every one of
them in a machine-readable form so that a reader cannot mistake them for the
measured deck. `estimate_report()` prints the list;
`tests/test_canards.py::test_every_estimated_quantity_is_declared` fails if a
new estimated constant is added without an entry.

The body aerodynamics that this module drives -- C_Nalpha and C_Malpha, which
turn out to dominate the answer -- ARE measured. That is the reason the
correction authority computed in docs/AUTHORITY-ENVELOPE.md is more
trustworthy than the canard model alone would suggest: the canard estimate
sets the size of a force, and the measured body deck sets the much larger
factor that force is multiplied by.

=============================================================================
GEOMETRY: SIZED FROM THE ENVELOPE
=============================================================================

The kit replaces the fuze. It occupies the fuze cavity and the volume a
standard point-detonating fuze would occupy above the shell nose, and the
canards must stow inside the shell's outer profile until muzzle exit.

    projectile diameter                 155 mm    (body radius 77.5 mm)
    guidance-section radius at the
      canard station, r_local            30 mm    design choice
    canard exposed semi-span, b          30 mm    design choice
    canard tip radius, s = r_local + b   60 mm    -> 120 mm across, 35 mm
                                                     inside the 155 mm profile
    canard chord, c                      45 mm    design choice
    canard station from the nose, x_c    90 mm    = 0.58 calibre

Why these numbers and not others:

  * SPAN. The stowage constraint is that four panels fold flat against a
    60 mm-diameter section and stay inside a 155 mm envelope. Folded aft
    along the section, each panel occupies 45 mm of length and 30 mm of
    circumference against 47 mm of circumference available per panel
    (pi*60/4). It fits with margin. Deployed, the tip sits at 60 mm radius,
    17.5 mm inside the shell's own radius, so the stowed-to-deployed sweep
    never leaves the envelope.

  * CHORD. Bounded by the axial length available on the guidance section
    forward of the shell's ogive shoulder. 45 mm is about a third of the
    exposed length of a standard 155 mm point-detonating fuze.

  * STATION. The canards must sit on the kit, i.e. forward of the shell nose
    thread. 90 mm from the tip is roughly the mid-point of the exposed
    guidance section. THIS IS THE SINGLE MOST CONSEQUENTIAL NUMBER IN THE
    MODULE and docs/AUTHORITY-ENVELOPE.md sweeps it, because the correction
    force is proportional to (x_c - x_cp) and the M107's subsonic centre of
    pressure sits at 0.11-0.13 m from the nose -- a few centimetres from the
    canards themselves.

  * NOT SIZED TO AN ANSWER. None of these dimensions was chosen by running
    the authority sweep and adjusting until the result looked adequate. The
    sweep was run afterwards and its result is reported as it came out. See
    docs/CANARD-MODEL.md section 6.

=============================================================================
ARRANGEMENT AND WHAT THE FOUR PANELS DO
=============================================================================

Four fixed panels at 90 degrees around the nose, in a "+" arrangement. In the
NOSE frame (which shares the projectile's x axis and is free to roll relative
to the body on a bearing):

    panel 0   azimuth   0 deg   steering pair, incidence  +delta_s
    panel 1   azimuth  90 deg   roll pair,     incidence  +delta_cant
    panel 2   azimuth 180 deg   steering pair, incidence  -delta_s
    panel 3   azimuth 270 deg   roll pair,     incidence  +delta_cant

The steering pair carries EQUAL AND OPPOSITE geometric incidence on opposite
sides of the body, which makes their aerodynamic forces ADD and their roll
moments CANCEL. The roll pair carries EQUAL incidence on opposite sides, which
makes their forces cancel and their roll moments add. Both statements are
asserted as tests rather than left as comments, because a sign slip here
produces a kit that rolls when asked to steer.

Positive delta_s produces a net force along the +phi_nose direction defined
below. Positive delta_cant produces a NEGATIVE roll moment on the nose, i.e.
it despins the nose against the right-hand rifling.

=============================================================================
THE NOSE ROLL ANGLE, phi_nose -- THE ONE NUMBER STEERING COMMANDS
=============================================================================

    phi_nose = phi_body + phi_rel

  phi_body  the projectile's 3-2-1 Euler roll angle, atan2(R21, R22).
            Earth-referenced: it is the roll of the body about its own x axis
            measured from the vertical plane containing that axis.
  phi_rel   state y[13], the nose's roll angle RELATIVE to the body.

With that definition the steering force points

    phi_nose =   0 deg   UP     (in the plane containing the shell axis
                                 and the local vertical)
    phi_nose =  90 deg   RIGHT
    phi_nose = 180 deg   DOWN
    phi_nose = 270 deg   LEFT

and the force direction in earth terms is
    cos(phi_nose) * up_hat + sin(phi_nose) * right_hat
where up_hat and right_hat span the plane normal to the shell axis. Proved by
tests/test_canards.py::test_phi_nose_walks_the_canard_force_round_the_compass,
which is parametrised over the four cardinal angles, and by its sibling
test_the_force_direction_does_not_depend_on_the_bodys_own_roll.

RATES: WHICH IS WHICH
---------------------
Getting these confused produces a nose that despins to the wrong reference,
and the error does not show up in a trajectory plot.

    p        y[10]              BODY inertial roll rate, rad/s. ~1100-1400.
    p_rel    y[14]              NOSE rate RELATIVE TO THE BODY, rad/s.
    p_nose   p + p_rel          NOSE INERTIAL roll rate, rad/s. ~0 when
                                despun -- this is the one the canard
                                aerodynamics sees, because the air does not
                                know about the body.

    T_aero   depends on p_nose  (aerodynamic: acts on inertial motion)
    T_fric   depends on p_rel   (bearing: acts on relative motion)
    T_brake  depends on p_rel   (clutch: acts on relative motion)

Despun means p_nose ~ 0, which means p_rel ~ -1100 rad/s. The bearing is
therefore slipping at nearly the full body spin for the whole guided phase.

=============================================================================
WHAT THIS MODULE DOES *NOT* MODEL
=============================================================================

The single largest term in the correction is NOT in this file. The canard
force acts ahead of the centre of gravity, so it trims the body to a small
angle of attack, and the BODY then generates a normal force through the
measured C_Nalpha. That happens inside sim/dynamics.py's existing rigid-body
integration, driven by the measured deck, with no extra code. This module
supplies only the direct canard loads; the induced body force emerges.

That is deliberate. Writing an "induced force" term here would mean modelling
by hand something the 6-DOF already integrates exactly, and the two would
eventually disagree. analysis/authority.py measures the ratio of the two
contributions from the simulation and compares it against the closed-form
prediction in `trim_force_factor()` below.

Also not modelled, and listed so the omissions are countable:
  * deployment transient -- the panels appear at full area instantaneously
  * hinge compliance, panel flexure, panel-to-panel manufacturing scatter
  * canard stall (the model is linear in local incidence at all angles)
  * shock interaction between panels and between panel and body
  * the moment of the panels' own axial drag about the CG (second order:
    it arises only from the incidence-dependent part of the drag)
  * base/wake effects of the guidance section on the shell ogive
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

from .dynamics import FlightModel
from .projectile import Projectile

__all__ = [
    "guided_model",
    "ESTIMATES",
    "Estimate",
    "estimate_report",
    "CanardGeometry",
    "NoseAssembly",
    "CanardModel",
    "canard_lift_curve_slope",
    "canard_zero_lift_drag",
    "body_wing_carryover",
    "trim_force_factor",
    "steering_response",
    "centre_of_pressure",
    "NOMINAL_GEOMETRY",
    "NOMINAL_NOSE",
    "no_brake",
]


# ===========================================================================
# The register of estimated quantities
# ===========================================================================
@dataclass(frozen=True)
class Estimate:
    """One quantity that is estimated rather than measured."""

    name: str
    value: str
    method: str
    accuracy: str


ESTIMATES: tuple[Estimate, ...] = (
    Estimate(
        "canard exposed semi-span, chord, station, local body radius",
        "b = 30 mm, c = 45 mm, x_c = 90 mm, r_local = 30 mm",
        "design choice, sized from the fuze-cavity and stowage envelope "
        "(module docstring). Not measured from any hardware.",
        "not an error bar -- a choice. Authority scales with (x_c - x_cp), "
        "so the station is swept in docs/AUTHORITY-ENVELOPE.md.",
    ),
    Estimate(
        "canard lift-curve slope C_Lalpha_canard(M)",
        "1.88 at M=0, 2.01 at M=0.8, 2.60 at M=1.2, 1.81 at M=2.0 /rad",
        "subsonic: Helmbold/Diederich low-aspect-ratio extension of "
        "lifting-line theory. supersonic: linearised (Ackeret) rectangular-"
        "fin result with the Busemann tip correction. transonic: linear "
        "bridge between M=0.8 and M=1.2.",
        "+-20 % subsonic and supersonic; +-40 % across 0.8 < M < 1.2, where "
        "the bridge omits the transonic lift peak entirely. "
        "CanardGeometry.cla_scale flies that band rather than only stating "
        "it: docs/MONTE-CARLO-DESIGN.md section 3 treats it as EPISTEMIC -- "
        "every round of a lot has the same canards, so it is a band around "
        "the CEP and not a term inside it.",
    ),
    Estimate(
        "body-to-wing carryover factor K_bw",
        "2.25 (= (1 + r_local/s)^2 with r_local/s = 0.5)",
        "slender-body theory (Nielsen, Missile Aerodynamics; "
        "Pitts-Nielsen-Kaattari). K_W(B) + K_B(W) = (1 + r/s)^2 is the "
        "combined panel-plus-carryover lift normalised on exposed-panel-"
        "alone lift.",
        "+-30 %. Slender-body theory is optimistic for r/s as large as 0.5. "
        "Applied to the transverse force and its pitching moment only, NOT "
        "to the roll moment, since the carryover lift acts on the body "
        "surface at a much smaller radius than the panel centroid.",
    ),
    Estimate(
        "canard zero-lift drag C_D0_canard(M)",
        "0.020 subsonic, 0.060 peak near M=1.05, 0.045 at M=2.0, on "
        "exposed panel planform area",
        "turbulent flat-plate skin friction on both faces (~0.008) plus a "
        "form/edge allowance, with a transonic wave-drag rise shaped by "
        "hand.",
        "+-50 %. It contributes a roll-orientation-INDEPENDENT range "
        "decrement, so it biases the authority envelope's centre without "
        "changing its size.",
    ),
    Estimate(
        "canard induced-drag efficiency e",
        "0.70",
        "conventional low-aspect-ratio value.",
        "+-30 %; the induced drag is a small fraction of the panel drag at "
        "the incidences flown here.",
    ),
    Estimate(
        "steering differential incidence delta_s",
        "5 deg (0.0873 rad) nominal",
        "design choice. Chosen so the induced body trim angle of attack "
        "stays near the linear range of the measured M107 deck, and so the "
        "local panel incidence stays far below low-aspect-ratio stall. NOT "
        "chosen to hit an authority target.",
        "a choice. Linearity in delta is measured in "
        "docs/AUTHORITY-ENVELOPE.md task D.4.",
    ),
    Estimate(
        "roll-pair cant angle delta_cant",
        "1.0 deg (0.01745 rad)",
        "design choice, sized so the aerodynamic despin torque exceeds the "
        "estimated bearing drag with margin over the whole guided phase.",
        "a choice. The resulting free-running nose rate is a RESULT, "
        "reported in docs/CANARD-MODEL.md section 5.",
    ),
    Estimate(
        "nose assembly axial inertia I_nose",
        "1.8e-3 kg m^2 (1.4 kg at a 36 mm radius of gyration)",
        "mass and size scaled from a standard 155 mm electronic fuze / "
        "course-correcting fuze envelope.",
        "+-40 %. It sets the despin time constant, not the steady state.",
    ),
    Estimate(
        "bearing viscous coefficient c_v",
        "6.5e-5 N m s/rad (0.072 N m at 1100 rad/s)",
        "SKF-form churning moment M0 = 1e-7 f0 (nu n)^(2/3) d_m^3 with "
        "f0 = 1.7 (greased deep-groove), nu = 30 mm^2/s, d_m = 45 mm, "
        "n = 10500 rpm; then linearised through that point.",
        "+-60 %, and the linearisation is itself wrong in form: the real "
        "churning torque grows as n^(2/3), so a linear coefficient fitted "
        "at 1100 rad/s over-predicts drag above that rate and under-"
        "predicts it below.",
    ),
    Estimate(
        "bearing Coulomb torque T_coulomb",
        "2.0e-3 N m",
        "SKF-form load moment M1 = f1 P d_m with f1 = 9e-4, a 50 N preload "
        "and d_m = 45 mm.",
        "+-70 %. It is 36 times smaller than the viscous term at the "
        "operating relative rate, so it does not matter except at "
        "crossover, which never occurs in flight.",
    ),
    Estimate(
        "brake torque capacity T_brake_max",
        "0.5 N m",
        "design choice for a small electromagnetic friction clutch inside a "
        "60 mm section.",
        "a choice. Step 4 sizes it against the roll-servo bandwidth it "
        "actually needs; this session only needs it to exceed the despin "
        "torque, which it does by a factor of about 5.",
    ),
    Estimate(
        "friction regularisation width p_eps",
        "1.0 rad/s",
        "tanh smoothing of the Coulomb and brake sign functions, so that "
        "fixed-step RK4 does not chatter on the discontinuity.",
        "immaterial: |p_rel| is 1000-1400 rad/s throughout the guided "
        "phase, so tanh(p_rel/p_eps) is 1.0 to machine precision.",
    ),
)


def estimate_report(stream=None) -> list[str]:
    """Print, and return, the register of estimated quantities."""
    lines = ["ESTIMATED QUANTITIES IN sim/canards.py -- none of these is measured data", ""]
    for e in ESTIMATES:
        lines.append(f"  {e.name}")
        lines.append(f"      value    : {e.value}")
        lines.append(f"      method   : {e.method}")
        lines.append(f"      accuracy : {e.accuracy}")
        lines.append("")
    if stream is not None:
        for ln in lines:
            print(ln, file=stream)
    return lines


# ===========================================================================
# Canard aerodynamics -- estimated, per the register above
# ===========================================================================
#: Section efficiency in the Helmbold form, C_l_alpha_section / 2 pi.
_ETA_SECTION = 0.95
#: Mach numbers bounding the transonic bridge.
_M_SUB = 0.80
_M_SUP = 1.20


def canard_lift_curve_slope(mach: float, aspect_ratio: float) -> float:
    """
    Lift-curve slope of one exposed canard panel, per radian, ESTIMATED.

    `aspect_ratio` is the EFFECTIVE aspect ratio of the panel plus its image
    in the body treated as a reflection plane, i.e. 2*b^2/S_panel for a
    rectangular panel of exposed semi-span b and area S_panel.

    Subsonic  -- Helmbold / Diederich extension of lifting-line theory to low
                 aspect ratio:

                     C_La = 2 pi AR / (2 + sqrt(AR^2 beta^2 / eta^2 + 4))

                 which tends to the slender-wing limit pi*AR/2 as AR -> 0 and
                 to 2 pi / beta as AR -> infinity. Expected accuracy +-20 %.

    Supersonic -- linearised two-dimensional theory with the Busemann tip
                 correction for a rectangular planform:

                     C_La = (4/beta) * (1 - 1/(2 AR beta))     AR beta >= 1

                 Below AR*beta = 1 the Mach cones from the two tips overlap
                 and the tip correction above is invalid; the value is
                 interpolated linearly in AR*beta between the slender-wing
                 limit pi*AR/2 at AR*beta = 0 and the formula's own value
                 2*AR at AR*beta = 1.

    Transonic  -- 0.8 < M < 1.2 is a straight line between the two endpoint
                 values. This omits the transonic lift-curve-slope peak and
                 is the weakest part of the model. Expected accuracy +-40 %.
    """
    ar = aspect_ratio

    def _subsonic(m: float) -> float:
        beta2 = max(0.0, 1.0 - m * m)
        return (2.0 * math.pi * ar) / (
            2.0 + math.sqrt(ar * ar * beta2 / (_ETA_SECTION * _ETA_SECTION) + 4.0)
        )

    def _supersonic(m: float) -> float:
        beta = math.sqrt(max(1e-9, m * m - 1.0))
        arb = ar * beta
        slender = 0.5 * math.pi * ar
        at_unity = 2.0 * ar  # (4/beta)(1 - 1/2) with AR*beta = 1
        if arb >= 1.0:
            return (4.0 / beta) * (1.0 - 1.0 / (2.0 * arb))
        return slender + (at_unity - slender) * arb

    if mach <= _M_SUB:
        return _subsonic(mach)
    if mach >= _M_SUP:
        return _supersonic(mach)
    lo = _subsonic(_M_SUB)
    hi = _supersonic(_M_SUP)
    f = (mach - _M_SUB) / (_M_SUP - _M_SUB)
    return lo + f * (hi - lo)


#: Zero-lift drag of one exposed panel on its own planform area, ESTIMATED.
#: Knots: (Mach, C_D0). Linear between, held flat outside.
_CD0_KNOTS = (
    (0.00, 0.020),
    (0.80, 0.020),
    (0.95, 0.040),
    (1.05, 0.060),
    (1.40, 0.052),
    (2.00, 0.045),
)
_CD0_M = tuple(k[0] for k in _CD0_KNOTS)
_CD0_V = tuple(k[1] for k in _CD0_KNOTS)


def canard_zero_lift_drag(mach: float) -> float:
    """Zero-lift drag coefficient of one exposed panel, ESTIMATED, +-50 %."""
    if mach <= _CD0_M[0]:
        return _CD0_V[0]
    if mach >= _CD0_M[-1]:
        return _CD0_V[-1]
    for i in range(len(_CD0_M) - 1):
        if _CD0_M[i] <= mach <= _CD0_M[i + 1]:
            f = (mach - _CD0_M[i]) / (_CD0_M[i + 1] - _CD0_M[i])
            return _CD0_V[i] + f * (_CD0_V[i + 1] - _CD0_V[i])
    return _CD0_V[-1]


def body_wing_carryover(body_radius: float, tip_radius: float) -> float:
    """
    Combined panel-plus-body-carryover lift factor, ESTIMATED.

        K_W(B) + K_B(W) = (1 + r/s)^2

    the slender-body result for the lift of a wing-body combination
    normalised on the lift of the exposed panels alone. r is the body radius
    at the panel station, s the panel tip radius from the body axis.
    """
    if tip_radius <= 0.0:
        return 1.0
    return (1.0 + body_radius / tip_radius) ** 2


# ===========================================================================
# Geometry
# ===========================================================================
@dataclass(frozen=True)
class CanardGeometry:
    """
    Four-panel canard set. Frozen: nothing reconfigures mid-trajectory.

    Lengths in metres, angles in radians. See the module docstring for how
    the nominal dimensions were chosen.
    """

    span_exposed: float = 0.030          # b, exposed semi-span
    chord: float = 0.045                 # c, rectangular planform
    station_from_nose: float = 0.090     # x_c
    body_radius_local: float = 0.030     # r_local at the canard station
    steering_deflection: float = math.radians(5.0)   # delta_s
    cant_angle: float = math.radians(1.0)            # delta_cant
    n_panels: int = 4
    #: Set False to strip the slender-body carryover and use exposed panels
    #: only. Used for the sensitivity case in docs/AUTHORITY-ENVELOPE.md.
    include_carryover: bool = True
    #: Oswald-like efficiency in the panel induced drag. ESTIMATED.
    induced_drag_efficiency: float = 0.70
    #: Step 6. Multiplies the panel lift-curve slope, and therefore the
    #: steering force, the induced pitching moment AND the aerodynamic roll
    #: damping together -- they are all proportional to the same
    #: qbar S_panel C_Lalpha, so this is the correct shape for a C_Lalpha
    #: uncertainty. Scaling only the steering force would flatter the kit by
    #: leaving the damping intact.
    #:
    #: 1.0 is the estimate of record. docs/CANARD-MODEL.md section 8 grades it
    #: at +-30 %, and that uncertainty is EPISTEMIC: every round of a lot
    #: flies with the same canards, so it is a band around the CEP and not a
    #: term inside it. See docs/MONTE-CARLO-DESIGN.md section 3.
    #:
    #: This is the TRUTH-side knob. `gnc.roll_control.NoseRollPlant.aero_scale`
    #: is the CONTROLLER's belief, and moving them together would measure a
    #: kit whose flight computer was told the answer.
    cla_scale: float = 1.0

    # -- derived, all cached in __post_init__ -----------------------------
    @property
    def panel_area(self) -> float:
        """Exposed planform area of ONE panel, m^2."""
        return self.span_exposed * self.chord

    @property
    def tip_radius(self) -> float:
        """Panel tip radius from the projectile axis, m."""
        return self.body_radius_local + self.span_exposed

    @property
    def centroid_radius(self) -> float:
        """Spanwise centroid radius of one panel from the axis, m."""
        return self.body_radius_local + 0.5 * self.span_exposed

    @property
    def aspect_ratio_effective(self) -> float:
        """
        Aspect ratio of the panel plus its image in the body, which is the
        aspect ratio the lift-curve-slope correlations want.
        """
        return 2.0 * self.span_exposed * self.span_exposed / self.panel_area

    @property
    def carryover(self) -> float:
        if not self.include_carryover:
            return 1.0
        return body_wing_carryover(self.body_radius_local, self.tip_radius)

    def moment_arm(self, projectile: Projectile) -> float:
        """
        Distance from the CG FORWARD to the canard station, m. Positive means
        the canards are ahead of the CG, which they always are.
        """
        return projectile.x_cg - self.station_from_nose

    def fits_within_envelope(self, projectile: Projectile) -> bool:
        """Deployed tip radius inside the shell's own radius."""
        return self.tip_radius < 0.5 * projectile.diameter

    def describe(self, projectile: Projectile) -> dict:
        return {
            "span_exposed_mm": 1e3 * self.span_exposed,
            "chord_mm": 1e3 * self.chord,
            "panel_area_mm2": 1e6 * self.panel_area,
            "station_from_nose_mm": 1e3 * self.station_from_nose,
            "station_calibres": self.station_from_nose / projectile.diameter,
            "tip_radius_mm": 1e3 * self.tip_radius,
            "tip_clearance_mm": 1e3 * (0.5 * projectile.diameter - self.tip_radius),
            "centroid_radius_mm": 1e3 * self.centroid_radius,
            "aspect_ratio_effective": self.aspect_ratio_effective,
            "carryover_factor": self.carryover,
            "moment_arm_mm": 1e3 * self.moment_arm(projectile),
            "steering_deflection_deg": math.degrees(self.steering_deflection),
            "cant_angle_deg": math.degrees(self.cant_angle),
            "total_exposed_area_ratio": self.n_panels * self.panel_area / projectile.reference_area,
        }


NOMINAL_GEOMETRY = CanardGeometry()


# ===========================================================================
# The despun nose assembly
# ===========================================================================
def no_brake(t: float) -> float:
    """Zero commanded brake torque. The default open-loop input."""
    return 0.0


#: Panel indices in the nose frame, matching `CanardModel._delta`:
#: the steering pair sits at 0 and 180 degrees with equal and opposite
#: deflection, the cant pair at 90 and 270 degrees with equal cant.
STEERING_PANELS = (0, 2)
CANT_PANELS = (1, 3)
ALL_PANELS = (0, 1, 2, 3)


@dataclass(frozen=True)
class NoseAssembly:
    """
    Mechanical properties of the despun forward section, and the OPEN-LOOP
    inputs that drive it in this step.

    `brake_command` is a function of TIME ALONE. That is a deliberate
    restriction, not an oversight: step 2.5 characterises the plant, and a
    brake torque that could see the state would be a controller. Step 4
    replaces this with a feedback law.

    `hold_angle`, when not None, replaces the two-state nose dynamics with an
    ideal kinematic constraint: phi_nose is pinned at that earth-referenced
    angle from `deploy_time` onward, as if a servo of infinite bandwidth and
    unlimited torque were closing around it. It is the idealisation used to
    measure the correction-authority envelope open-loop -- see
    docs/AUTHORITY-ENVELOPE.md section 2 for why, and for what step 4 must
    add back.
    """

    inertia: float = 1.8e-3              # I_nose, kg m^2
    viscous: float = 6.5e-5              # c_v, N m s/rad
    coulomb: float = 2.0e-3              # T_coulomb, N m
    brake_max: float = 0.5               # N m, capacity
    #: rad/s. tanh regularisation width for the two sign functions.
    p_eps: float = 1.0
    #: s. Canards stow before this; the nose carries no aerodynamic torque.
    #: With a staged deployment this is STAGE 1, the cant pair.
    deploy_time: float = 0.0
    #: rad, or None for the free two-state DOF.
    hold_angle: Optional[float] = None
    #: Open-loop commanded brake torque magnitude, N m, >= 0. f(t).
    brake_command: Callable[[float], float] = no_brake

    # -- staged deployment -------------------------------------------------
    #
    # STAGE 1 releases the CANT pair at `deploy_time`; it produces the despin
    # torque and nothing else. STAGE 2 releases the STEERING pair, which is
    # the only thing that produces a commandable transverse force. Single
    # stage -- both pairs at `deploy_time` -- is the default and is what
    # every number in steps 2.5 and 4 was measured with.
    #
    # A stowed panel carries no load of any kind, so stage 1 is not a
    # modelling switch that scales a coefficient: two of the four panels are
    # simply absent from the sum. The consequences are NOT symmetric --
    # `T_cant` sums over the cant pair alone and is unchanged, while the
    # aerodynamic roll damping sums over all four and is halved. See
    # docs/STAGED-DEPLOYMENT.md section 1.
    #
    #: s, or None for single-stage. A FIXED stage-2 time.
    steering_deploy_time: Optional[float] = None
    #: Predicate of TIME ALONE: has the steering pair been released? Overrides
    #: `steering_deploy_time` when given.
    #:
    #: Time alone, for exactly the reason `brake_command` is a function of
    #: time alone: RK4 evaluates the derivative four times per step at three
    #: different times, and a release condition read from the state inside the
    #: derivative would fire at some stages and not others. A sampled release
    #: law latches the decision at a sample instant and HOLDS it, so within
    #: one integration step the panel really is either out or stowed. See
    #: gnc.roll_control.StagedDeployment.
    steering_gate: Optional[Callable[[float], bool]] = None

    def friction_torque(self, p_rel: float) -> float:
        """
        Bearing torque ON THE NOSE, N m, as a function of the RELATIVE roll
        rate. Always opposes relative motion, so with the nose despun
        (p_rel ~ -1100 rad/s) it is positive: the bearing drags the nose
        around with the body, and the canted canards must overcome it.

        Viscous plus Coulomb. The viscous term dominates by a factor of about
        36 at the operating relative rate, because that rate is nearly the
        full body spin and never approaches zero in flight; the Coulomb term
        would only dominate near p_rel = 0, a condition that occurs only
        before deployment.
        """
        return -(self.viscous * p_rel + self.coulomb * math.tanh(p_rel / self.p_eps))

    def brake_torque(self, t: float, p_rel: float) -> float:
        """
        Clutch torque ON THE NOSE, N m. A friction brake can only oppose
        relative motion and dissipate; it can never drive. Commanded
        magnitude is clipped into [0, brake_max].
        """
        cmd = float(self.brake_command(t))
        if cmd <= 0.0:
            return 0.0
        if cmd > self.brake_max:
            cmd = self.brake_max
        return -cmd * math.tanh(p_rel / self.p_eps)


NOMINAL_NOSE = NoseAssembly()


# ===========================================================================
# Closed-form cross-checks (not used by the simulation)
# ===========================================================================
def centre_of_pressure(C_Nalpha: float, C_Malpha: float, projectile: Projectile) -> float:
    """
    Centre of pressure aft of the nose, m, from the measured deck.

        x_cp = x_cg - d * C_Malpha / C_Nalpha

    with the package's sign convention that POSITIVE C_Malpha is
    destabilising, i.e. the CP lies AHEAD of the CG.
    """
    return projectile.x_cg - projectile.diameter * C_Malpha / C_Nalpha


def trim_force_factor(
    geometry: CanardGeometry,
    projectile: Projectile,
    C_Nalpha: float,
    C_Malpha: float,
    C_Lalpha_canard: float,
) -> dict:
    """
    Closed-form quasi-static prediction of the net steering force, as a
    multiple of the direct canard force. A CROSS-CHECK ONLY: the simulation
    does not call this, and analysis/authority.py compares the two.

    THE MECHANISM
    -------------
    At trim the total moment about the CG is zero. Two transverse forces act,
    both AHEAD of the CG:

        the canard force  F_c  at arm  a_c = x_cg - x_c
        the body normal force N at arm  a_n = x_cg - x_cp

    Moment balance gives  N = -(a_c / a_n) F_c, so

        F_net = F_c (1 - a_c/a_n) = F_c (x_c - x_cp) / (x_cg - x_cp).

    For a fin-stabilised airframe the CP lies BEHIND the CG, a_n is negative,
    and the factor exceeds unity: nose canards are amplified, which is why
    they work well on mortar bombs. For a SPIN-stabilised shell the CP lies
    ahead of the CG, and a nose canard ahead of the CP gets a factor that is
    NEGATIVE and, for the M107 subsonically, very small. The projectile
    accelerates opposite to the canard force, and the net force is a small
    difference of two much larger opposing forces.

    The canards contribute to C_Nalpha and C_Malpha themselves, which moves
    the CP forward toward the canard station and shrinks the factor further.
    That contribution is included here, and it is why the returned factor
    saturates rather than growing with panel area.

    Returns a dict of the intermediate quantities so a caller can report the
    whole chain rather than one opaque number.
    """
    d = projectile.diameter
    S = projectile.reference_area
    a_c = geometry.moment_arm(projectile)

    # A four-panel "+" array responds to body angle of attack like a
    # body-fixed surface of TWO panels' area: summing the four panel forces
    # over equally spaced azimuths gives 2 * S_panel * C_La, independent of
    # roll orientation. Proved in
    # tests/test_canards.py, test_four_panel_array_response_to_body_aoa_is_
    # isotropic_in_roll
    dC_N = 2.0 * geometry.panel_area * C_Lalpha_canard * geometry.carryover / S
    dC_M = dC_N * a_c / d

    C_N_tot = C_Nalpha + dC_N
    C_M_tot = C_Malpha + dC_M

    x_cp_body = centre_of_pressure(C_Nalpha, C_Malpha, projectile)
    x_cp_tot = centre_of_pressure(C_N_tot, C_M_tot, projectile)

    a_n_tot = projectile.x_cg - x_cp_tot
    factor = (geometry.station_from_nose - x_cp_tot) / a_n_tot

    return {
        "x_cp_body_m": x_cp_body,
        "x_cp_body_cal": x_cp_body / d,
        "x_cp_with_canards_m": x_cp_tot,
        "x_cp_with_canards_cal": x_cp_tot / d,
        "canard_station_m": geometry.station_from_nose,
        "moment_arm_canard_m": a_c,
        "moment_arm_body_m": a_n_tot,
        "delta_C_Nalpha": dC_N,
        "delta_C_Malpha": dC_M,
        "C_Nalpha_total": C_N_tot,
        "C_Malpha_total": C_M_tot,
        "trim_force_factor": factor,
        "induced_over_direct": -a_c / a_n_tot,
    }


def steering_response(
    geometry: CanardGeometry,
    projectile: Projectile,
    C_Nalpha: float,
    C_Malpha: float,
    C_Ypalpha: float,
    C_Mpalpha: float,
    C_Lalpha_canard: float,
    spin: float,
    airspeed: float,
    axial_angular_momentum: Optional[float] = None,
    reduced_rate_factor: float = 0.5,
) -> dict:
    """
    The COMPLETE quasi-steady closed form for the net steering acceleration,
    as a complex multiple of the direct canard force. A cross-check only; the
    simulation does not call it.

    WHY THIS EXISTS ALONGSIDE trim_force_factor()
    ---------------------------------------------
    `trim_force_factor()` above carries only the STATIC moment balance. That
    is the dominant term for a control force well away from the centre of
    pressure, and it is what step 2.5 used. It omits two things that are
    negligible in general and are NOT negligible here, because a fuze-well
    canard on a spin-stabilised shell sits essentially AT the subsonic centre
    of pressure, where the static term is the small difference of two much
    larger opposing forces:

      * the GYROSCOPIC term  i H_x / V  in the denominator, which rotates the
        response. It is small -- arctan(H_x / (V m a_n)) is about 2 deg for
        the M107 at apogee -- and it is the quantitative answer to "does a
        spinning shell trim, or precess?". It trims.

      * the MAGNUS FORCE, perpendicular to the angle of attack and therefore
        perpendicular to both the canard force and the induced normal force.
        Its size relative to the normal force is

            mu = |C_Ypalpha| (pd/2V) / C_Nalpha  ~ 0.15  for the M107

        which is negligible against the normal force -- classical
        aeroballistics neglects the Magnus force and keeps only its moment,
        e.g. Ollerenshaw and Costello, J. Guidance 31(5) 2008 -- but is about
        THREE TIMES the ~5 per cent residue that survives the cancellation.

    THE DERIVATION
    --------------
    Complex transverse plane zeta = y + i z of the non-rolling frame. At the
    quasi-steady state the body axis turns with the velocity vector, so the
    transverse angular rate is w = i A / V and the transverse Euler equation
    with dw/dt = 0 gives

        H_x A / V = i k a (a_n - i beta) + i a_c F_c        moment
        m A       = (1 + i mu) k a + F_c                    force

    with k = qbar S C_Nalpha, a the complex angle of attack, a_n the
    centre-of-pressure arm and a_c the canard arm, both ahead of the CG.
    Eliminating a:

        A (H_x/V - gamma m) = F_c (i a_c - gamma),
        gamma = i (a_n - i beta) / (1 + i mu)

    Setting mu = beta = 0 and H_x = 0 recovers trim_force_factor() exactly.

    `axial_angular_momentum` defaults to `projectile.I_axial * spin`; pass the
    despun-nose value `Ix_body * p + I_nose * p_nose` when it matters.

    Returns a dict; `net_over_direct` is complex, with argument measured from
    the canard force direction.
    """
    d = projectile.diameter
    S = projectile.reference_area
    m = projectile.mass
    a_c = geometry.moment_arm(projectile)

    # The canards' own contribution to C_Nalpha and C_Malpha, exactly as in
    # trim_force_factor(): a four-panel "+" array responds to body angle of
    # attack like a body-fixed surface of two panels' area.
    dC_N = 2.0 * geometry.panel_area * C_Lalpha_canard * geometry.carryover / S
    dC_M = dC_N * a_c / d
    C_N_tot = C_Nalpha + dC_N
    C_M_tot = C_Malpha + dC_M
    a_n = d * C_M_tot / C_N_tot

    p_hat = reduced_rate_factor * spin * d / airspeed
    mu = abs(C_Ypalpha) * p_hat / C_N_tot
    beta = d * C_Mpalpha * p_hat / C_N_tot
    # Station of the Magnus force ahead of the CG, for reporting.
    a_M = (-d * C_Mpalpha / abs(C_Ypalpha)) if C_Ypalpha != 0.0 else 0.0

    Hx = (projectile.I_axial * spin if axial_angular_momentum is None
          else float(axial_angular_momentum))

    gamma = 1j * (a_n - 1j * beta) / (1.0 + 1j * mu)
    # A / (F_c / m): the net acceleration per unit direct canard force,
    # normalised so that the real part is the step-2.5 trim force factor.
    net = m * (1j * a_c - gamma) / (Hx / airspeed - gamma * m)

    static = (a_n - a_c) / a_n
    return {
        "x_cp_with_canards_m": projectile.x_cg - a_n,
        "moment_arm_body_m": a_n,
        "moment_arm_canard_m": a_c,
        "magnus_force_ratio_mu": mu,
        "magnus_moment_arm_m": a_M,
        "magnus_moment_beta_m": beta,
        "gyroscopic_over_static": Hx / (airspeed * m * a_n),
        "gyroscopic_rotation_deg": math.degrees(
            math.atan2(Hx / airspeed, m * a_n)),
        "static_trim_factor": static,
        "in_line_term_m": a_c - a_n,
        "magnus_term_m": mu * (a_n - a_M),
        "magnus_over_in_line": (abs(mu * (a_n - a_M) / (a_c - a_n))
                                if a_c != a_n else float("inf")),
        "net_over_direct": net,
        "net_over_direct_magnitude": abs(net),
        "net_over_direct_phase_deg": math.degrees(math.atan2(net.imag, net.real)),
        "magnitude_over_static": (abs(net) / abs(static)
                                  if static != 0.0 else float("inf")),
    }


# ===========================================================================
# The model itself -- the step-4 control seam, filled in
# ===========================================================================
#: Panel azimuths in the NOSE frame, radians. Panel 0 at 0 deg.
_PANEL_AZIMUTH = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)


@dataclass(frozen=True)
class CanardModel:
    """
    The canard force, moment and nose roll torque, as a ControlCallback.

    Call signature is sim.dynamics.ControlCallback extended by one return
    value:

        (force_body, moment_body, nose_roll_torque)

    force_body    N,   canard aerodynamic force in BODY axes, applied at the
                       CG by the caller
    moment_body   N m, about the CG, TRANSVERSE COMPONENTS ONLY. Its x
                       component is identically zero.
    nose_roll_torque
                  N m, the canard roll moment about the projectile axis,
                  which acts on the NOSE and not on the body. dynamics.py
                  routes it into the nose torque balance and lets the
                  bearing and brake pass whatever reaction the physics
                  requires back to the body.

    Splitting the roll moment out is what makes it impossible to accidentally
    apply the despin torque to the shell body, which would be a silent and
    fairly convincing error: the shell would just spin down a little fast.
    """

    geometry: CanardGeometry = NOMINAL_GEOMETRY
    nose: NoseAssembly = NOMINAL_NOSE
    projectile: Optional[Projectile] = None
    #: Set False to remove the canards entirely while keeping the nose DOF.
    enabled: bool = True

    def __post_init__(self):
        g = self.geometry
        object.__setattr__(self, "_S_panel", float(g.panel_area))
        object.__setattr__(self, "_y_c", float(g.centroid_radius))
        object.__setattr__(self, "_ar", float(g.aspect_ratio_effective))
        object.__setattr__(self, "_K", float(g.carryover))
        object.__setattr__(self, "_cla_scale", float(g.cla_scale))
        object.__setattr__(self, "_deploy", float(self.nose.deploy_time))
        object.__setattr__(self, "_hold", self.nose.hold_angle)
        object.__setattr__(
            self, "_pi_ar_e",
            math.pi * float(g.aspect_ratio_effective) * float(g.induced_drag_efficiency),
        )
        # Panel incidences, nose frame: steering pair at 0 and 180 deg with
        # equal and opposite incidence, roll pair at 90 and 270 deg with
        # equal incidence.
        ds = float(g.steering_deflection)
        dc = float(g.cant_angle)
        object.__setattr__(self, "_delta", (ds, dc, -ds, dc))
        # Staged deployment. `_steer_gate` wins if given; otherwise the
        # steering pair is out from `_steer_t`, which equals `_deploy` for the
        # single-stage default and therefore reproduces step 2.5 exactly.
        gate = self.nose.steering_gate
        object.__setattr__(self, "_steer_gate", gate)
        st_t = self.nose.steering_deploy_time
        object.__setattr__(
            self, "_steer_t",
            float(self.nose.deploy_time) if st_t is None else float(st_t))
        object.__setattr__(self, "_staged",
                           gate is not None or st_t is not None)
        if self.projectile is None:
            object.__setattr__(self, "_arm", None)
        else:
            object.__setattr__(self, "_arm", float(g.moment_arm(self.projectile)))

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def body_roll_angle(q) -> float:
        """
        The projectile's earth-referenced roll angle about its own x axis,
        rad, from the body->earth quaternion. This is the 3-2-1 Euler roll,
        atan2(R21, R22), computed directly so the derivative does not have to
        build a matrix.
        """
        qw = float(q[0]); qx = float(q[1]); qy = float(q[2]); qz = float(q[3])
        n2 = qw * qw + qx * qx + qy * qy + qz * qz
        if n2 != 1.0 and n2 > 0.0:
            inv = 1.0 / math.sqrt(n2)
            qw *= inv; qx *= inv; qy *= inv; qz *= inv
        r21 = 2.0 * (qy * qz + qw * qx)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
        return math.atan2(r21, r22)

    def nose_roll_angle(self, y) -> float:
        """
        phi_nose, the earth-referenced roll orientation of the nose, rad.
        In hold mode this is the commanded angle by construction.
        """
        if self._hold is not None:
            return float(self._hold)
        return self.body_roll_angle(y[6:10]) + float(y[13])

    def force_direction_body(self, y) -> np.ndarray:
        """
        Unit vector, BODY axes, along which a positive steering deflection
        pushes. Diagnostic; the derivative computes this inline.
        """
        phi_rel = self.nose_roll_angle(y) - self.body_roll_angle(y[6:10])
        return np.array([0.0, math.sin(phi_rel), -math.cos(phi_rel)])

    def deployed_panels(self, t: float) -> tuple:
        """
        Which panel indices carry load at time `t`. All four unless a staged
        deployment is configured and stage 2 has not fired, in which case the
        cant pair alone.
        """
        if t < self._deploy:
            return ()
        if not self._staged:
            return ALL_PANELS
        if self._steer_gate is not None:
            out = bool(self._steer_gate(t))
        else:
            out = t >= self._steer_t
        return ALL_PANELS if out else CANT_PANELS

    # -- the callback -----------------------------------------------------
    def __call__(self, t: float, y, st) -> tuple:
        """
        Returns (force_body, moment_body, nose_roll_torque).

        `st` is the sim.dynamics.AeroState already computed for this state,
        so nothing here recomputes the atmosphere or the body aerodynamics.
        """
        if (not self.enabled) or t < self._deploy:
            return np.zeros(3), np.zeros(3), 0.0

        # Which panels are out. `ALL_PANELS` for every run in steps 2.5 and 4;
        # `CANT_PANELS` during stage 1 of a staged deployment.
        if self._staged:
            panels = self.deployed_panels(t)
        else:
            panels = ALL_PANELS

        V = st.airspeed
        if V < 1e-6:
            return np.zeros(3), np.zeros(3), 0.0

        qbar = st.dynamic_pressure
        u, v, w = float(st.v_rel_body[0]), float(st.v_rel_body[1]), float(st.v_rel_body[2])
        q_rate = float(y[11])
        r_rate = float(y[12])

        phi_body = self.body_roll_angle(y[6:10])
        if self._hold is not None:
            phi_rel = float(self._hold) - phi_body
            p_nose = 0.0
        else:
            phi_rel = float(y[13])
            p_nose = float(y[10]) + float(y[14])

        arm = self._arm
        if arm is None:
            raise ValueError(
                "CanardModel needs a projectile to know the canard moment arm; "
                "construct it with projectile=..."
            )

        S_p = self._S_panel
        y_c = self._y_c
        C_La = self._cla_scale * canard_lift_curve_slope(st.mach, self._ar)
        C_D0 = canard_zero_lift_drag(st.mach)
        qS = qbar * S_p
        inv_V = 1.0 / V

        # Roll-rate contribution to local incidence is the same for every
        # panel -- it is what makes the sum a pure roll damping.
        rate_term = p_nose * y_c * inv_V

        c0 = math.cos(phi_rel)
        s0 = math.sin(phi_rel)
        # Azimuths phi_rel, phi_rel+90, phi_rel+180, phi_rel+270 in BODY axes.
        cos_psi = (c0, -s0, -c0, s0)
        sin_psi = (s0, c0, -s0, -c0)

        fy = 0.0
        fz = 0.0
        fx = 0.0
        roll = 0.0
        for i in panels:
            cp = cos_psi[i]
            sp = sin_psi[i]
            # Local incidence seen by panel i, positive along its own normal.
            alpha = (
                (-v * sp + w * cp) - arm * (q_rate * cp + r_rate * sp)
            ) * inv_V + rate_term + self._delta[i]
            # Panel normal force, opposing the local transverse motion, along
            # e_n = [0, -sin psi, cos psi].
            f = -qS * C_La * alpha
            fy += f * (-sp)
            fz += f * cp
            roll += y_c * f
            # Panel axial force: profile plus induced drag.
            cl = C_La * alpha
            fx -= qS * (C_D0 + cl * cl / self._pi_ar_e)

        # Slender-body carryover multiplies the transverse force and hence
        # the pitching moment. It does NOT multiply the roll moment: the
        # carryover lift sits on the body surface, not out at the panel
        # centroid radius.
        K = self._K
        fy *= K
        fz *= K

        # Moment about the CG of a transverse force applied at `arm` forward:
        #   M = (arm * x_hat) x F  =>  My = -arm*Fz,  Mz = +arm*Fy
        force = np.array([fx, fy, fz])
        moment = np.array([0.0, -arm * fz, arm * fy])
        return force, moment, roll

    # -- diagnostics ------------------------------------------------------
    def panel_incidences(self, t: float, y, st) -> np.ndarray:
        """
        Local incidence of each panel, rad. For stall and linearity checks.

        All four are returned whatever the deployment stage: a stowed panel's
        entry is the incidence it WOULD see. Use `deployed_panels(t)` to index
        the ones that are actually carrying load.
        """
        V = max(st.airspeed, 1e-6)
        u, v, w = (float(x) for x in st.v_rel_body)
        q_rate = float(y[11]); r_rate = float(y[12])
        phi_body = self.body_roll_angle(y[6:10])
        if self._hold is not None:
            phi_rel = float(self._hold) - phi_body
            p_nose = 0.0
        else:
            phi_rel = float(y[13])
            p_nose = float(y[10]) + float(y[14])
        arm = self._arm
        c0, s0 = math.cos(phi_rel), math.sin(phi_rel)
        cos_psi = (c0, -s0, -c0, s0)
        sin_psi = (s0, c0, -s0, -c0)
        out = np.empty(4)
        for i in range(4):
            cp, sp = cos_psi[i], sin_psi[i]
            out[i] = (
                (-v * sp + w * cp) - arm * (q_rate * cp + r_rate * sp)
            ) / V + p_nose * self._y_c / V + self._delta[i]
        return out

    def with_deflection(self, delta_s: float) -> "CanardModel":
        """A copy with a different steering deflection. For the sweeps."""
        return replace(self, geometry=replace(self.geometry, steering_deflection=delta_s))

    def with_hold(self, phi_nose: Optional[float], deploy_time: Optional[float] = None) -> "CanardModel":
        """A copy with a different commanded hold angle and/or deploy time."""
        nose = replace(
            self.nose,
            hold_angle=phi_nose,
            deploy_time=self.nose.deploy_time if deploy_time is None else deploy_time,
        )
        return replace(self, nose=nose)


# ===========================================================================
# Assembly
# ===========================================================================
def guided_model(
    base: FlightModel,
    geometry: CanardGeometry = NOMINAL_GEOMETRY,
    nose: NoseAssembly = NOMINAL_NOSE,
) -> FlightModel:
    """
    A copy of `base` with the canard kit fitted.

    The SAME NoseAssembly instance goes to both `FlightModel.nose`, which
    integrates the torque balance, and `CanardModel.nose`, which reads the
    deployment time and the hold command. Constructing the two separately is
    the obvious way to end up simulating a nose that deploys at one time
    aerodynamically and another time mechanically, so this function is the
    only supported way to fit the kit.
    """
    kit = CanardModel(geometry=geometry, nose=nose, projectile=base.projectile)
    return replace(base, control=kit, nose=nose)
