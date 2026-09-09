"""
Closed-loop roll-angle control of the despun nose.

Step 4 of the roadmap. The plant is step 2.5's two-state despun forward
section (`sim.canards.NoseAssembly` plus `sim.dynamics`); this module closes a
loop around it so that the earth-referenced nose roll angle `phi_nose` tracks
a commanded angle supplied from outside. There is no guidance here: the
command comes from a test harness.


THE ACTUATOR IS ONE-SIDED
-------------------------
The nose rides on a bearing inside a body spinning at 1100-1400 rad/s. Two
torques act on it about the projectile axis:

    T_aero(p_nose)  from the canted canard pair. At p_nose = 0 it is a
                    DESPIN torque, backwards, proportional to dynamic
                    pressure. It also carries the roll damping.
    bearing + brake both act on the RELATIVE rate and, because the nose runs
                    slower than the body, both drag the nose FORWARD.

The brake is a clutch. It can only couple the nose to the body and dissipate;
it can never drive the nose backwards relative to the body. So the commanded
brake torque `u` is confined to [0, brake_max] and the achievable nose rate at
any flight condition is confined to a one-sided interval

    p_nose in [ p_free , p_full ]        p_free = rate at u = 0
                                         p_full = rate at u = brake_max

with `p_free` ALWAYS negative inside this project's firing-table envelope (the
cant beats the bearing by 2.2-9.5x -- docs/CANARD-MODEL.md section 5) and
`p_full` positive only where the brake can out-torque the cant. Where it
cannot, the nose cannot be held at all: see `ActuatorEnvelope.can_hold`.

Both limits are functions of dynamic pressure and both change by a large
factor between deployment and impact. Nothing here assumes symmetry.


THE LINEARISED PLANT, AND WHY THE SCHEDULING VARIABLE IS DYNAMIC PRESSURE
-------------------------------------------------------------------------
With the nose despun the relative rate never approaches zero, so the two
`tanh(p_rel/p_eps)` sign regularisations in `NoseAssembly` are pinned at -1
and the torque balance is affine in `u`:

    I_n * dp_nose/dt = T_bias(t) + u - c_tot(t) * p_nose
    dphi_nose/dt     = p_nose

    T_bias(t) = c_v * p_body + T_coulomb - T_cant(qbar, V, M)     < 0
    c_tot(t)  = c_aero(qbar, V, M) + c_v                          > 0

`T_cant` and `c_aero` are both directly proportional to `qbar`. Over the
guided phase of the adopted engagement `qbar` falls from 135 kPa to about
40 kPa, so:

    the feed-forward torque the loop must supply just to stand still varies
        by a factor of ~4;
    the plant DC gain 1/c_tot varies by the same factor the other way;
    the plant time constant I_n/c_tot varies by ~2.2x.

A fixed-gain loop tuned at either end is wrong at the other by those factors.
`qbar` is the one variable both coefficients are proportional to, it is
available onboard from the navigation solution and a standard atmosphere, and
scheduling on it removes essentially all of the variation -- what is left is
the Mach dependence of the panel lift-curve slope. Mach is therefore carried
alongside it, and the scheduler evaluates the same `canard_lift_curve_slope()`
the simulator uses rather than a fitted curve.


STRUCTURE
---------
A cascade, not a single PID:

    gate    stay OPEN until the nose has finished despinning from the body
            rate, holding full brake meanwhile
    outer   P on angle error  ->  a RATE command, clipped to the achievable
            one-sided interval AND to what the remaining error can absorb at
            the available deceleration
    inner   PI on rate error, plus feed-forward of the estimated bias torque,
            output clipped to [0, brake_max] with conditional integration

The cascade is what makes the asymmetry expressible. A single PID on angle has
one output limit and cannot say "the fastest you may ask for in this direction
is 32 rad/s and in that direction 201 rad/s"; the outer clip says exactly
that, and it is also the outer loop anti-windup, since the outer loop has no
integrator to wind up.

Three of the four stages exist because the actuator is one-sided, and each was
put there by a measurement rather than by taste:

  the gate            closing the loop while the nose is still sweeping down
                      from 1308 rad/s makes the brake chatter and part-captures
                      it once a revolution. Measured over an ensemble of
                      engagement phases the gate does NOT raise the mean
                      retained authority; it narrows the spread, from 25 points
                      of reachable-set rms to 7, and costs 0.5 s of
                      acquisition. CONTROL-CHARACTERISATION.md section 7.2.
  the stopping limit  the brake accelerates the nose hard and stops it only by
                      being released, at |T_bias| / I_n, which collapses as
                      the bearing drag approaches the cant torque. Without the
                      limit a 90 degree step overshoots by 50 % at the low end
                      of the canard aerodynamic band.
  the rate clip       there is no point asking for a rate on the wrong side of
                      p_free or p_full; nothing can deliver it.

Everything is discrete: the controller is sampled at `sample_rate` and holds
its output between samples, which is what a fuze microcontroller does and what
keeps the RK4 stages of the 6-DOF seeing a constant input inside one step.


WHAT IS MODELLED AND WHAT IS NOT
--------------------------------
Modelled: the one-sided actuator, its capacity limit, a first-order coil lag,
command quantisation, the sample rate, bearing Coulomb friction and the
integrator safeguard it demands, and the exact aerodynamic despin torque.

Not modelled: the roll-angle measurement. `phi_nose` is read from truth. A
sensor bias maps one-for-one into steady-state tracking error -- the loop has
no way to distinguish it from a real angle -- so step 5's roll-angle accuracy
adds directly to the numbers in docs/CONTROL-CHARACTERISATION.md. That is
stated rather than modelled because the sensor is step 5's to specify.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

from sim.canards import (CanardGeometry, CanardModel, NoseAssembly,
                         canard_lift_curve_slope)
from sim.dynamics import aero_state as _aero_state

#: The body 3-2-1 Euler roll angle from the attitude quaternion. Borrowed from
#: the canard model so that the servo and the plant cannot drift apart on the
#: definition of the angle being controlled.
_body_roll_angle = CanardModel.body_roll_angle

__all__ = [
    "wrap_pi",
    "FlightCondition",
    "ConditionSchedule",
    "NoseRollPlant",
    "ActuatorEnvelope",
    "BrakeActuator",
    "ControllerConfig",
    "ControllerState",
    "RollAngleController",
    "DutyCycleCommander",
    "constant_command",
    "step_command",
    "sine_command",
    "StagedDeploymentConfig",
    "StagedDeployment",
    "BrakeLaw",
    "ReducedRun",
    "simulate_reduced",
    "plant_from",
]


def wrap_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi]. The roll error is a circle, not a line."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ===========================================================================
# The reduced-order plant
# ===========================================================================
@dataclass(frozen=True)
class FlightCondition:
    """
    Everything the nose roll dynamics depends on, at one instant.

    That the list is this short is a property of the plant, not a
    simplification: the four-panel roll moment sums the flow-induced panel
    incidences over four equally spaced azimuths, and that sum is identically
    zero. Body angle of attack, transverse rates and roll orientation
    therefore cancel EXACTLY out of the nose roll torque. Asserted against the
    full model in tests/test_roll_control.py.
    """

    time: float           # s
    qbar: float           # Pa
    airspeed: float       # m/s
    mach: float           # -
    body_spin: float      # rad/s, the shell axial rate


@dataclass(frozen=True)
class NoseRollPlant:
    """
    The nose roll degree of freedom, reduced to one scalar second-order
    system. A restatement of the torque balance in `sim.dynamics` and
    `sim.canards.NoseAssembly`, not an independent model: the tests assert the
    two agree to machine precision on the torque and to milliradians on a
    two-second trajectory.
    """

    geometry: CanardGeometry
    nose: NoseAssembly
    #: Body axial inertia EXCLUDING the nose, kg m^2. Needed only for the
    #: reaction of the brake on the body spin.
    body_inertia: float = 0.145240
    #: Multiplies the panel lift-curve slope, and therefore BOTH the cant
    #: torque and the aerodynamic roll damping, which are proportional to the
    #: same qbar * S_panel * C_Lalpha. That is the correct shape for a
    #: C_Lalpha uncertainty: it is one coefficient, not two, and scaling only
    #: the despin torque would flatter the loop by leaving the damping intact.
    #: 1.0 is the estimate of record; docs/CONTROL-ROBUSTNESS.md sweeps it.
    aero_scale: float = 1.0
    #: How many panels are DEPLOYED and therefore contribute roll damping.
    #: 4 for the single-stage kit and for stage 2 of a staged one; 2 during
    #: stage 1, when only the cant pair is out.
    #:
    #: This enters the damping and NOT the cant torque, and that asymmetry is
    #: the whole of the staging question: `T_cant` already sums over the cant
    #: pair alone, so releasing two panels instead of four leaves the despin
    #: torque untouched and halves what resists it. See
    #: docs/STAGED-DEPLOYMENT.md section 1.
    deployed_panels: int = 4

    # -- aerodynamics -----------------------------------------------------
    def _q_s_cla(self, c: FlightCondition) -> float:
        g = self.geometry
        cla = canard_lift_curve_slope(c.mach, g.aspect_ratio_effective)
        return self.aero_scale * c.qbar * g.panel_area * cla

    def cant_torque(self, c: FlightCondition) -> float:
        """
        Magnitude of the aerodynamic despin torque at p_nose = 0, N m,
        positive. The torque ON the nose is minus this.

            T_cant = 2 qbar S_panel C_Lalpha y_c delta_cant

        The factor 2 is the two canted panels; the steering pair carries equal
        and opposite deflections and contributes nothing to roll.
        """
        g = self.geometry
        return 2.0 * self._q_s_cla(c) * g.centroid_radius * g.cant_angle

    def aero_damping(self, c: FlightCondition) -> float:
        """
        Aerodynamic roll damping of the deployed panel array, N m s/rad,
        positive.

            c_aero = n_deployed qbar S_panel C_Lalpha y_c^2 / V

        It acts on the nose INERTIAL rate, because the air does not know about
        the body. `n_deployed` is 4 for the single-stage kit and 2 during
        stage 1 of a staged deployment, where only the cant pair is out.
        """
        g = self.geometry
        y_c = g.centroid_radius
        return (float(self.deployed_panels) * self._q_s_cla(c)
                * y_c * y_c / max(c.airspeed, 1e-6))

    def aero_torque(self, c: FlightCondition, p_nose: float) -> float:
        """Total canard roll torque on the nose, N m."""
        return -(self.cant_torque(c) + self.aero_damping(c) * p_nose)

    # -- internal torques -------------------------------------------------
    def bearing_torque(self, p_rel: float) -> float:
        """Bearing torque on the nose, N m. Delegates to the plant of record."""
        return self.nose.friction_torque(p_rel)

    def brake_torque(self, u: float, p_rel: float) -> float:
        """
        Brake torque on the nose, N m, for a commanded magnitude `u`.

        One-sided by construction: a `u` below zero is not a reverse drive, it
        is simply no brake. The `tanh` is the plant sign regularisation, and
        it is why the brake has NO authority at p_rel = 0 -- the condition
        that holds at the instant of deployment.
        """
        cmd = min(max(u, 0.0), self.nose.brake_max)
        if cmd == 0.0:
            return 0.0
        return -cmd * math.tanh(p_rel / self.nose.p_eps)

    def nose_acceleration(self, c: FlightCondition, p_nose: float,
                          p_body: float, u: float) -> float:
        """dp_nose/dt, rad/s^2. The nose INERTIAL angular acceleration."""
        p_rel = p_nose - p_body
        t_int = self.bearing_torque(p_rel) + self.brake_torque(u, p_rel)
        return (self.aero_torque(c, p_nose) + t_int) / self.nose.inertia

    # -- the affine form the controller is designed against ---------------
    def bias_torque(self, c: FlightCondition) -> float:
        """
        The constant term of the linearised torque balance, N m.

            T_bias = c_v * p_body + T_coulomb - T_cant

        Negative throughout this project's envelope: the cant wins. This is
        the quantity the feed-forward cancels, and minus it is the brake
        torque needed to hold the nose still.
        """
        n = self.nose
        return n.viscous * c.body_spin + n.coulomb - self.cant_torque(c)

    def total_damping(self, c: FlightCondition) -> float:
        """c_aero + c_v, N m s/rad. The coefficient of p_nose."""
        return self.aero_damping(c) + self.nose.viscous

    def hold_command(self, c: FlightCondition) -> float:
        """
        Brake torque required to hold p_nose = 0, N m, UNCLIPPED. Greater than
        `brake_max` means the nose cannot be held at this condition.
        """
        return -self.bias_torque(c)

    def equilibrium_rate(self, c: FlightCondition, u: float) -> float:
        """Steady nose inertial rate for a held brake command, rad/s."""
        return (self.bias_torque(c) + min(max(u, 0.0), self.nose.brake_max)) \
            / self.total_damping(c)

    def time_constant(self, c: FlightCondition) -> float:
        """I_nose / c_tot, s. The open-loop rate time constant."""
        return self.nose.inertia / self.total_damping(c)

    def envelope(self, c: FlightCondition) -> "ActuatorEnvelope":
        """The one-sided rate envelope at this condition. Task A."""
        return ActuatorEnvelope(
            condition=c,
            rate_free=self.equilibrium_rate(c, 0.0),
            rate_full=self.equilibrium_rate(c, self.nose.brake_max),
            hold_command=self.hold_command(c),
            brake_max=self.nose.brake_max,
            time_constant=self.time_constant(c),
            cant_torque=self.cant_torque(c),
            bearing_torque=self.nose.viscous * c.body_spin + self.nose.coulomb,
            bias=self.bias_torque(c),
            inertia=self.nose.inertia,
        )


@dataclass(frozen=True)
class ActuatorEnvelope:
    """
    What the one-sided actuator can do at one flight condition. Task A.

    `rate_free` is the passive return rate, `rate_full` the rate at full
    brake. Both are steady-state; the approach to either is first order with
    `time_constant`.
    """

    condition: FlightCondition
    rate_free: float          # rad/s, negative: the passive return
    rate_full: float          # rad/s, at u = brake_max
    hold_command: float       # N m, unclipped, to hold p_nose = 0
    brake_max: float          # N m
    time_constant: float      # s
    cant_torque: float        # N m
    bearing_torque: float     # N m
    bias: float = 0.0         # N m, T_bias = bearing - cant. Negative.
    inertia: float = 1.8e-3   # kg m^2

    @property
    def can_hold(self) -> bool:
        """
        True when a commanded angle can actually be held, which needs the
        achievable rate interval to BRACKET zero.

        Two ways it fails, and testing only the first is a trap:

        the brake is too small       hold_command > brake_max, so even full
                                     brake leaves the nose despinning;
        the bearing beats the cant   T_bias > 0, so the nose creeps FORWARD
                                     with the brake released and nothing can
                                     slow it, because the brake only pushes
                                     the same way.

        The second inverts the sign of `hold_command`, which then passes a
        naive `hold_command <= brake_max` test while describing a plant that
        cannot hold at all. docs/CANARD-MODEL.md section 5 warned that the
        cant beats the bearing by only 2.2x at the lowest dynamic pressure in
        the envelope and that a servo should not bake the assumption in;
        docs/CONTROL-ROBUSTNESS.md section 4 is where it actually bites.
        """
        return self.rate_free <= 0.0 <= self.rate_full

    @property
    def rate_free_degs(self) -> float:
        return math.degrees(self.rate_free)

    @property
    def rate_full_degs(self) -> float:
        return math.degrees(self.rate_full)

    @property
    def accel_forward(self) -> float:
        """
        Peak nose angular acceleration toward the spin direction, rad/s^2:
        full brake against the cant torque, from rest.

        This, not the steady rate, is what sets how long a small angle change
        takes -- the loop never asks for anything near the steady rates, so a
        90 degree step is governed by how hard the nose can be pushed, not by
        how fast it could eventually go.
        """
        return (self.bias + self.brake_max) / self.inertia

    @property
    def accel_return(self) -> float:
        """Peak angular acceleration backwards, rad/s^2: brake released."""
        return self.bias / self.inertia

    @property
    def accel_asymmetry(self) -> float:
        """|forward| / |return| acceleration. Crosses 1.0 during the flight."""
        a = abs(self.accel_return)
        return abs(self.accel_forward) / a if a > 0 else math.inf

    @property
    def saturation_margin(self) -> float:
        """brake_max / hold_command. Below 1.0 the angle cannot be held."""
        return self.brake_max / self.hold_command if self.hold_command > 0 else math.inf

    @property
    def asymmetry(self) -> float:
        """
        |rate_full| / |rate_free|. Above 1 the forward slew is the faster
        direction, below 1 the passive return is. Where the nose cannot be
        held both limits are negative and the ratio is a slowing factor, not
        a forward capability.
        """
        return abs(self.rate_full) / abs(self.rate_free) if self.rate_free else math.inf

    def as_dict(self) -> dict:
        return {
            "time_s": self.condition.time,
            "qbar_kPa": 1e-3 * self.condition.qbar,
            "mach": self.condition.mach,
            "airspeed_ms": self.condition.airspeed,
            "body_spin_rads": self.condition.body_spin,
            "cant_torque_Nm": self.cant_torque,
            "bearing_torque_Nm": self.bearing_torque,
            "hold_command_Nm": self.hold_command,
            "brake_max_Nm": self.brake_max,
            "can_hold": self.can_hold,
            "saturation_margin": self.saturation_margin,
            "rate_free_rads": self.rate_free,
            "rate_free_degs": math.degrees(self.rate_free),
            "rate_full_rads": self.rate_full,
            "rate_full_degs": math.degrees(self.rate_full),
            "asymmetry": self.asymmetry,
            "time_constant_s": self.time_constant,
            "accel_forward_rads2": self.accel_forward,
            "accel_return_rads2": self.accel_return,
            "accel_asymmetry": self.accel_asymmetry,
        }


# ===========================================================================
# Flight-condition schedule
# ===========================================================================
@dataclass(frozen=True)
class ConditionSchedule:
    """
    Dynamic pressure, airspeed, Mach and body spin as functions of time,
    tabulated from a flown trajectory and linearly interpolated.

    `body_damping` is the shell own roll damping coefficient, N m s/rad,
    recovered from the tabulated spin decay as

        k_body(t) = -Ix * (dp/dt) / p

    so that the reduced model reproduces the ballistic spin history when the
    brake is off, and responds correctly to the brake reaction when it is not.
    Recovering it rather than re-deriving it from `C_lp` keeps this model a
    restatement of the flown one.
    """

    time: np.ndarray
    qbar: np.ndarray
    airspeed: np.ndarray
    mach: np.ndarray
    spin: np.ndarray
    body_damping: np.ndarray

    @staticmethod
    def from_trajectory(traj, axial_inertia: float) -> "ConditionSchedule":
        t = np.asarray(traj.t, dtype=float)
        p = np.asarray(traj.omega[:, 0], dtype=float)
        pdot = np.gradient(p, t)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(np.abs(p) > 1.0, -axial_inertia * pdot / p, 0.0)
        return ConditionSchedule(
            time=t,
            qbar=np.asarray(traj.dynamic_pressure, dtype=float),
            airspeed=np.asarray(traj.airspeed, dtype=float),
            mach=np.asarray(traj.mach, dtype=float),
            spin=p,
            body_damping=k,
        )

    def at(self, t: float, body_spin: Optional[float] = None) -> FlightCondition:
        """
        The flight condition at time `t`. `body_spin` overrides the tabulated
        spin, which the reduced simulation does because it integrates the spin
        itself once the brake starts pushing on it.
        """
        return FlightCondition(
            time=t,
            qbar=float(np.interp(t, self.time, self.qbar)),
            airspeed=float(np.interp(t, self.time, self.airspeed)),
            mach=float(np.interp(t, self.time, self.mach)),
            body_spin=float(np.interp(t, self.time, self.spin))
            if body_spin is None else float(body_spin),
        )

    def damping_at(self, t: float) -> float:
        return float(np.interp(t, self.time, self.body_damping))


# ===========================================================================
# The actuator
# ===========================================================================
@dataclass
class BrakeActuator:
    """
    What sits between the commanded brake torque and the delivered one.

    Three effects, all of them the actuator rather than the control law:

    quantisation  the command is a digital word driving a coil current.
                  `bits` resolves [0, brake_max] into 2**bits levels.
    lag           an electromagnetic brake cannot change torque instantly.
                  Coil L/R plus eddy currents in the rotor; modelled as one
                  pole at `tau`. ESTIMATED at 10 ms -- see
                  docs/CONTROL-CHARACTERISATION.md for the sensitivity.
    capacity      hard clip at `brake_max`.

    Held discretely at the controller sample rate, which is what a
    zero-order-hold digital drive does anyway.
    """

    brake_max: float = 0.5
    tau: float = 0.010          # s
    bits: int = 10
    delivered: float = 0.0      # N m, the state

    def reset(self, value: float = 0.0) -> None:
        self.delivered = float(value)

    def quantise(self, u: float) -> float:
        u = min(max(u, 0.0), self.brake_max)
        if self.bits <= 0:
            return u
        levels = (1 << self.bits) - 1
        return round(u / self.brake_max * levels) / levels * self.brake_max

    def step(self, u_cmd: float, dt: float) -> float:
        """Advance the lag by `dt` under a held command and return the torque."""
        target = self.quantise(u_cmd)
        if self.tau <= 0.0:
            self.delivered = target
        else:
            a = math.exp(-dt / self.tau)
            self.delivered = target + (self.delivered - target) * a
        return self.delivered


# ===========================================================================
# The control law
# ===========================================================================
@dataclass(frozen=True)
class ControllerConfig:
    """
    Everything that sets the loop, and the reason for each number.

    NO NUMBER HERE WAS CHOSEN TO HIT A PERFORMANCE TARGET. The two bandwidths
    are set by the actuator lag and by loop separation, which is how a cascade
    is sized when there is no specification to meet:

      rate_bandwidth   1/3 of the actuator pole 1/tau = 100 rad/s. Any faster
                       and the 10 ms lag eats the phase margin.
      angle_bandwidth  1/5 of the rate bandwidth: standard cascade separation,
                       so the outer loop sees the inner one as a gain of one.
      integral_zero    1/5 of the rate bandwidth, so the PI zero is well
                       inside the loop and adds little phase at crossover.

    `ff_*_scale` are the controller BELIEF about the plant, deliberately
    separable from the truth so that Task E can mis-tell it.
    """

    sample_rate: float = 500.0        # Hz
    rate_bandwidth: float = 30.0      # rad/s, inner loop
    angle_bandwidth: float = 6.0      # rad/s, outer loop
    integral_zero: float = 6.0        # rad/s
    #: Below this |tanh(p_rel/p_eps)| the brake has too little authority to be
    #: worth integrating against. The Coulomb-band safeguard; see section 4 of
    #: docs/CONTROL-CHARACTERISATION.md.
    brake_gain_floor: float = 0.20
    #: Integral term clamp, N m. A hard backstop behind the conditional
    #: integration, sized at the full actuator range.
    integral_limit: float = 0.5
    #: Multiplies the feed-forward estimate of the cant torque. 1.0 is perfect
    #: knowledge; Task E varies it.
    ff_cant_scale: float = 1.0
    #: Multiplies the feed-forward estimate of the bearing torque.
    ff_friction_scale: float = 1.0
    #: Set False to remove the feed-forward entirely and leave the PI alone.
    feed_forward: bool = True
    #: Set False to remove every anti-windup safeguard. The naive integrator,
    #: kept so the tests can show what it costs.
    anti_windup: bool = True
    #: Limit the commanded rate to what the remaining angle error can absorb
    #: at the available DECELERATION, which for a one-sided actuator is not
    #: the same as the available acceleration. Set False to see what it buys.
    stopping_limit: bool = True

    # -- engagement -------------------------------------------------------
    #: Close the angle loop only once the nose rate has fallen INTO the
    #: achievable interval, within this tolerance in rad/s. Before that the
    #: nose is still despinning from the body rate and no brake command can
    #: bring it to the commanded angle, so trying is not merely futile -- it
    #: costs range. See docs/CONTROL-CHARACTERISATION.md section 6.
    engage_gate: bool = True
    engage_tolerance: float = 10.0
    #: Brake command while waiting for the gate, as a fraction of capacity.
    #: Full brake arrests the nose fastest and therefore opens the gate
    #: soonest.
    pre_engage_brake: float = 1.0
    #: SECOND, MODEL-FREE engagement criterion, rad/s^2: the nose has stopped
    #: despinning, whatever rate it stopped at.
    #:
    #: The rate test above is evaluated against the CONTROLLER's model, and
    #: the wait itself holds full brake. Put those together with a true plant
    #: weaker than the model and they deadlock: full brake drives the nose to
    #: the TRUE equilibrium, which is above the model's threshold, so the gate
    #: never opens and the brake is never released. Measured at 15 % below the
    #: nominal canard aerodynamics, which is inside the estimate band, the
    #: loop stuck at +200 rad/s for the whole flight.
    #:
    #: This criterion cannot deadlock, because it asks only whether the
    #: transient is over. See docs/CONTROL-ROBUSTNESS.md section 6.
    engage_settled_accel: float = 50.0
    #: Extra dwell, s, after the gate condition is met and before the loop
    #: closes.
    #:
    #: Zero by design. It exists because the residual acquisition transient is
    #: PHASE-SENSITIVE -- the shell turns 208 times a second, so which body
    #: roll angle the loop happens to engage at is not controllable -- and
    #: sweeping this dwell over a couple of hundred milliseconds samples that
    #: ensemble. docs/CONTROL-CHARACTERISATION.md section 7 reports the
    #: reachable set as a band over the sweep rather than as one number from
    #: one arbitrary phase.
    engage_dwell: float = 0.0

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate


@dataclass
class ControllerState:
    """Mutable loop state. Separated so a run can be restarted or inspected."""

    integral: float = 0.0        # N m
    u_command: float = 0.0       # N m, before the actuator
    rate_command: float = 0.0    # rad/s
    angle_error: float = 0.0     # rad
    saturated_high: bool = False
    saturated_low: bool = False
    integrating: bool = True
    holding: bool = True         # False in the free arc of a duty cycle
    engaged: bool = False        # the angle loop is closed
    engage_time: Optional[float] = None
    #: Previous sample's nose rate, for the model-free engagement test.
    last_rate: Optional[float] = None
    #: When the gate CONDITION was first met, which `engage_dwell` delays.
    gate_open_time: Optional[float] = None


class RollAngleController:
    """
    Gain-scheduled cascade with feed-forward on the estimated despin torque.

    Call `update()` once per sample with the measured nose angle and rate, the
    body spin and the flight condition; it returns the brake torque command in
    [0, brake_max]. `plant` is the controller MODEL of the nose, which need not
    equal the truth -- Task E exploits exactly that.
    """

    def __init__(self, plant: NoseRollPlant,
                 config: ControllerConfig = ControllerConfig(),
                 actuator: Optional[BrakeActuator] = None):
        self.plant = plant
        self.config = config
        self.actuator = actuator or BrakeActuator(brake_max=plant.nose.brake_max)
        self.state = ControllerState()

    # -- scheduling -------------------------------------------------------
    def gains(self, c: FlightCondition) -> tuple:
        """
        (kp_rate, ki_rate, k_angle) at this flight condition.

        The rate loop is placed, not tuned: with

            I_n dp/dt = u - c_tot p + T_bias

        a proportional gain kp makes the closed-loop pole (c_tot + kp)/I_n, so

            kp = I_n * rate_bandwidth - c_tot

        cancels the plant damping and puts the pole exactly at the chosen
        bandwidth whatever the dynamic pressure is doing. That one line is the
        whole gain schedule, and it is why the loop behaves the same at
        135 kPa and at 38 kPa where a fixed gain would not.
        """
        cfg = self.config
        c_tot = self.plant.total_damping(c)
        kp = self.plant.nose.inertia * cfg.rate_bandwidth - c_tot
        kp = max(kp, 1e-6)
        ki = kp * cfg.integral_zero
        return kp, ki, cfg.angle_bandwidth

    def feed_forward(self, c: FlightCondition, rate_command: float) -> float:
        """
        Brake torque that would hold `rate_command` exactly, N m, from the
        controller model of the plant.

            u_ff = -T_bias + c_tot * p_cmd
                 = T_cant - c_v p_body - T_coulomb + c_tot p_cmd

        This is the term that does the work. The despin torque swings by a
        factor of four over the guided phase, and a PI alone would have to
        chase it through its integrator, which costs bandwidth the loop does
        not have to spare.
        """
        cfg = self.config
        if not cfg.feed_forward:
            return 0.0
        n = self.plant.nose
        cant = cfg.ff_cant_scale * self.plant.cant_torque(c)
        friction = cfg.ff_friction_scale * (n.viscous * c.body_spin + n.coulomb)
        return cant - friction + self.plant.total_damping(c) * rate_command

    def rate_limits(self, c: FlightCondition) -> tuple:
        """
        (p_min, p_max) achievable nose rates at this condition, rad/s. The
        outer loop clips its rate command to this interval, which is where the
        one-sidedness of the actuator enters the control law.
        """
        return (self.plant.equilibrium_rate(c, 0.0),
                self.plant.equilibrium_rate(c, self.plant.nose.brake_max))

    # -- the loop ---------------------------------------------------------
    def update(self, phi_command: float, phi_nose: float, p_nose: float,
               p_body: float, c: FlightCondition,
               holding: bool = True) -> float:
        """
        One sample. Returns the DELIVERED brake torque, N m, after
        quantisation and the actuator lag.

        `holding` False is the free arc of a duty cycle: the brake is
        released, the integrator is held, and the loop makes no attempt to
        track. Releasing rather than commanding zero through the loop matters,
        because the nose then runs away backwards at tens of rad/s and any
        integrator left running would wind up against a 180 degree error.
        """
        cfg = self.config
        st = self.state
        dt = cfg.dt
        st.holding = holding

        if not holding:
            # A release is a release. This is tested before the engagement
            # gate on purpose: the gate holds full brake while it waits for
            # the nose to despin, and applying that during a duty-cycle free
            # arc would contradict the command it was given.
            st.angle_error = wrap_pi(phi_command - phi_nose)
            st.rate_command = 0.0
            st.u_command = 0.0
            st.integrating = False
            st.saturated_high = st.saturated_low = False
            return self.actuator.step(0.0, dt)

        # --- the engagement gate ------------------------------------------
        # At deployment the nose is still locked to the body at 1300 rad/s.
        # No brake command brings it to a commanded angle from there: the
        # whole achievable interval is a couple of hundred rad/s wide and the
        # nose is far outside it. Closing the angle loop anyway makes the
        # brake chatter as the error wraps every few hundredths of a second
        # while the nose sweeps down through the epicyclic frequencies, and
        # that pumps the shell's coning motion. What that costs depends on the
        # body roll phase at capture, which nobody controls, so it is a
        # distribution rather than a number -- see
        # docs/CONTROL-CHARACTERISATION.md section 7.
        #
        # So the loop waits, holding full brake to arrest the nose as fast as
        # it can, until the nose rate has fallen into the interval the
        # actuator can actually command. Once engaged it stays engaged.
        if cfg.engage_gate and not st.engaged:
            _, p_max = self.rate_limits(c)
            reachable = p_nose <= p_max + cfg.engage_tolerance
            # ... or the despin transient is simply over. Model-free, and the
            # reason the gate cannot lock itself out when the plant is weaker
            # than the controller believes.
            settled = (st.last_rate is not None
                       and abs(p_nose - st.last_rate) / dt
                       < cfg.engage_settled_accel)
            st.last_rate = p_nose
            if (reachable or settled) and st.gate_open_time is None:
                st.gate_open_time = c.time
            if (st.gate_open_time is not None
                    and c.time >= st.gate_open_time + cfg.engage_dwell):
                st.engaged = True
                st.engage_time = c.time
            else:
                st.rate_command = 0.0
                st.angle_error = wrap_pi(phi_command - phi_nose)
                st.integrating = False
                st.integral = 0.0
                u_pre = cfg.pre_engage_brake * self.plant.nose.brake_max
                st.u_command = u_pre
                return self.actuator.step(u_pre, dt)

        kp, ki, k_angle = self.gains(c)
        p_min, p_max = self.rate_limits(c)

        # --- outer loop: angle error -> rate command, clipped one-sidedly ---
        err = wrap_pi(phi_command - phi_nose)
        st.angle_error = err
        p_cmd = k_angle * err

        # STOPPING DISTANCE. The brake accelerates the nose forward hard and
        # decelerates it only by being released, at |T_bias| / I_n -- and
        # T_bias goes to zero as the bearing drag approaches the cant torque.
        # Arriving at the commanded angle faster than the loop can stop is
        # therefore not a tuning question but a physical one, and the cure is
        # to ask only for a rate the remaining error can absorb:
        #
        #     |p_cmd| <= sqrt( 2 * a_available * |error| )
        #
        # It is inactive at the nominal plant -- a 90 degree error allows
        # 13.4 rad/s against the 9.4 the proportional term asks for -- and
        # binds where the deceleration collapses. At the low end of the canard
        # aerodynamic band a 90 degree step overshoots by 50 % without it and
        # 19 % with it. It uses the CONTROLLER's estimate of T_bias, which is
        # all it has, so it cannot help against a plant the controller has
        # been told wrongly about: docs/CONTROL-ROBUSTNESS.md section 5.
        #
        # `-bias` rather than `abs(bias)`: where the bearing beats the cant,
        # T_bias is positive, releasing the brake accelerates the nose FORWARD
        # and there is no forward deceleration at all. The guard below then
        # skips the limit, which is correct -- there is no stopping distance
        # to respect because there is no stopping.
        if cfg.stopping_limit:
            bias = self.plant.bias_torque(c)
            accel_stop = ((-bias) if p_cmd > 0.0
                          else (bias + self.plant.nose.brake_max)) \
                / self.plant.nose.inertia
            if accel_stop > 0.0:
                v_max = math.sqrt(2.0 * accel_stop * abs(err))
                p_cmd = min(max(p_cmd, -v_max), v_max)

        p_cmd = min(max(p_cmd, p_min), p_max)
        st.rate_command = p_cmd

        # --- inner loop: rate error -> torque -------------------------------
        rate_err = p_cmd - p_nose
        u_lin = self.feed_forward(c, p_cmd) + kp * rate_err + st.integral

        # --- input-nonlinearity inversion -----------------------------------
        # Brake torque on the nose is -u tanh(p_rel/p_eps), so the magnitude
        # delivered per unit command is |tanh|, which collapses to zero as the
        # relative rate does. Divide it out so the loop commands TORQUE and
        # the actuator is left to supply it; floor the inversion so a
        # near-zero relative rate cannot ask for an infinite command.
        p_rel = p_nose - p_body
        gain = math.tanh(p_rel / self.plant.nose.p_eps)
        mag = abs(gain)
        if cfg.anti_windup and mag < cfg.brake_gain_floor:
            u_raw = u_lin / cfg.brake_gain_floor
            authority = False
        else:
            u_raw = u_lin / max(mag, 1e-12)
            authority = True
        # A positive p_rel means the nose LEADS the body, and the brake would
        # then drag it backwards. The loop never commands that: it is outside
        # the one-sided actuator model and outside the flight envelope.
        if gain > 0.0:
            u_raw = 0.0
            authority = False

        u_max = self.plant.nose.brake_max
        u = min(max(u_raw, 0.0), u_max)
        st.saturated_high = u_raw > u_max
        st.saturated_low = u_raw < 0.0
        st.u_command = u

        # --- conditional integration -----------------------------------------
        # Integrate only when the actuator can act on the result. Three stops:
        # saturated high with the error still pushing higher, saturated low
        # with the error still pushing lower, and too little brake authority
        # to resolve at all, which is the Coulomb-band case.
        if not cfg.anti_windup:
            integrate_now = True
        else:
            integrate_now = authority
            if st.saturated_high and rate_err > 0.0:
                integrate_now = False
            if st.saturated_low and rate_err < 0.0:
                integrate_now = False
        st.integrating = integrate_now
        if integrate_now:
            st.integral += ki * rate_err * dt
            if cfg.anti_windup:
                lim = cfg.integral_limit
                st.integral = min(max(st.integral, -lim), lim)

        return self.actuator.step(u, dt)

    def reset(self, integral: float = 0.0) -> None:
        self.state = ControllerState(integral=integral)
        self.actuator.reset(0.0)


# ===========================================================================
# Commands
# ===========================================================================
@dataclass(frozen=True)
class DutyCycleCommander:
    """
    Task D. With a fixed canard deflection the correction MAGNITUDE is not
    commandable -- only its direction. Magnitude comes from alternating
    between holding the commanded angle, which delivers the full correction in
    that direction, and releasing the brake, which lets the nose spin freely
    at tens of rad/s so that the force direction sweeps the circle and the net
    correction over a period averages to nearly nothing.

    period       s, one hold plus one free arc
    duty         hold fraction of the period, in [0, 1]
    phase        s, offset of the first hold
    """

    angle: float                # rad, the commanded roll angle while holding
    period: float = 2.0         # s
    duty: float = 1.0           # -
    phase: float = 0.0          # s

    def holding(self, t: float) -> bool:
        if self.duty >= 1.0:
            return True
        if self.duty <= 0.0:
            return False
        u = (t - self.phase) % self.period
        return u < self.duty * self.period

    def __call__(self, t: float) -> tuple:
        """(phi_command, holding) at time t."""
        return self.angle, self.holding(t)


def constant_command(angle: float) -> Callable[[float], tuple]:
    """A fixed roll-angle command, always holding."""
    def cmd(t: float) -> tuple:
        return angle, True
    return cmd


def step_command(angle_before: float, angle_after: float,
                 step_time: float) -> Callable[[float], tuple]:
    """A single step in commanded roll angle. Task C.1 and C.2."""
    def cmd(t: float) -> tuple:
        return (angle_after if t >= step_time else angle_before), True
    return cmd


def sine_command(centre: float, amplitude: float,
                 frequency_hz: float, start: float = 0.0) -> Callable[[float], tuple]:
    """A sinusoidal roll-angle command. Task C.4, the bandwidth sweep."""
    w = 2.0 * math.pi * frequency_hz
    def cmd(t: float) -> tuple:
        if t < start:
            return centre, True
        return centre + amplitude * math.sin(w * (t - start)), True
    return cmd


# ===========================================================================
# Staged deployment -- the stage-2 release law
# ===========================================================================
@dataclass(frozen=True)
class StagedDeploymentConfig:
    """
    When the STEERING pair is released, given that the CANT pair is already
    out and despinning the nose.

    THE SHAPE OF THIS GATE IS THE ONE docs/CONTROL-ROBUSTNESS.md SECTION 6
    WARNS ABOUT. Its trigger is a condition on the plant, and its action --
    putting two more panels into the flow -- changes the plant. The engagement
    gate had exactly that shape and deadlocked. So the criteria here are
    layered, and the last one consults nothing at all:

      capture    the loop is engaged and holding the commanded angle. The
                 condition staging exists to wait for. Model-based, through
                 the engagement gate's own rate test.
      quiescent  the nose is turning slower than `quiescent_rate` in the earth
                 frame, whoever is or is not controlling it. Model-free: an
                 absolute rate, not a threshold computed from the plant.
      backstop   `max_delay` after stage 1, unconditionally. Model-free and
                 not a condition on anything, so no plant can defeat it.

    `test_the_stage_two_trigger_cannot_deadlock` drives the law with a
    pathological sample stream and asserts the backstop fires anyway.

    WHY NOT THE ENGAGEMENT GATE'S OWN MODEL-FREE TEST. That one asks whether
    the nose rate has stopped CHANGING, which is right for "the despin
    transient is over" and wrong here: the nose stops changing rate when it
    reaches the full-brake equilibrium, which is tens of rad/s and is exactly
    the sweeping condition staging is meant to avoid. Copying it would have
    released the steering pair into the transient it was added to skip.
    """

    #: Master switch. False is the single-stage kit of steps 2.5 and 4.
    enabled: bool = False

    #: Release exactly this long after stage 1, ignoring every condition.
    #: The Task C delay sweep runs on this; it is model-free by construction
    #: and needs no backstop, being its own deterministic release time.
    fixed_delay: Optional[float] = None

    #: No release before this, s after stage 1. Zero by default.
    min_delay: float = 0.0

    # -- primary: the servo has captured -----------------------------------
    capture_tolerance: float = math.radians(5.0)
    capture_dwell: float = 0.10

    # -- secondary, model-free: the nose is nearly stationary --------------
    quiescent_rate: float = 5.0        # rad/s
    quiescent_dwell: float = 0.10

    # -- backstop, model-free and unconditional ----------------------------
    #: s after stage 1. Sized in docs/STAGED-DEPLOYMENT.md section 2: long
    #: enough that the nominal plant never reaches it, short enough that a
    #: round which reaches it still has most of its guided phase left.
    max_delay: float = 2.0


class StagedDeployment:
    """
    The stage-2 release law, and the seam it reaches the 6-DOF through.

    `gate` is passed as `sim.canards.NoseAssembly.steering_gate` and is a
    function of TIME ALONE for the same reason `BrakeLaw.brake_command` is:
    RK4 evaluates the derivative four times per step, and a release condition
    read from the state inside the derivative would fire at some stages and
    not others, putting two panels into the flow for part of a step. `update`
    is called once per controller sample and LATCHES the decision; `gate`
    returns the latched value. Within one integration step the panel is either
    out or stowed, and nothing about the derivative changes.

    The law also owns the controller's BELIEF about how many panels are out.
    During stage 1 the true aerodynamic roll damping is half the four-panel
    value while the cant torque is unchanged, so a controller scheduled
    against the four-panel plant would mis-place its rate pole and, worse,
    compute an engagement threshold the true plant sits above -- which is the
    section 6 deadlock arriving by a second route. `plant` is the stage-2
    (four-panel) model; the law installs a two-panel copy for stage 1 and
    swaps it back on release.
    """

    def __init__(self, config: StagedDeploymentConfig,
                 deploy_time: float,
                 controller: Optional["RollAngleController"] = None,
                 plant: Optional[NoseRollPlant] = None):
        self.config = config
        self.deploy_time = float(deploy_time)
        self.controller = controller
        self.stage2_plant = plant if plant is not None else (
            controller.plant if controller is not None else None)
        self.stage1_plant = (
            replace(self.stage2_plant, deployed_panels=2)
            if self.stage2_plant is not None else None)

        self.released = not config.enabled
        self.release_time = self.deploy_time if self.released else None
        self.release_reason = "single_stage" if self.released else None
        self.samples = 0
        self._capture_since: Optional[float] = None
        self._quiescent_since: Optional[float] = None
        #: (t, reason) of the release, plus the trace the report reads.
        self.log_t: list = []
        self.log_p_nose: list = []
        self.log_err: list = []
        self.log_released: list = []

        if config.enabled and controller is not None and self.stage1_plant is not None:
            controller.plant = self.stage1_plant

    # -- the seam ---------------------------------------------------------
    def gate(self, t: float) -> bool:
        """Held between samples. Deliberately ignores `t`."""
        return self.released

    @property
    def delay(self) -> Optional[float]:
        """Stage 1 to stage 2, s."""
        if self.release_time is None:
            return None
        return self.release_time - self.deploy_time

    # -- the law ----------------------------------------------------------
    def update(self, t: float, angle_error: float, p_nose: float,
               engaged: bool = False) -> None:
        """
        One sample. `angle_error` is the wrapped commanded-minus-measured nose
        angle in rad, `p_nose` the measured nose INERTIAL rate in rad/s, and
        `engaged` whether the angle loop has closed.

        Nothing here consults the controller's plant model except through
        `engaged`, and the backstop consults nothing at all.
        """
        self.samples += 1
        self.log_t.append(t)
        self.log_p_nose.append(p_nose)
        self.log_err.append(angle_error)
        self.log_released.append(self.released)
        if self.released:
            return

        cfg = self.config
        elapsed = t - self.deploy_time
        if elapsed < cfg.min_delay:
            return

        if cfg.fixed_delay is not None:
            if elapsed >= cfg.fixed_delay:
                self._fire(t, "scheduled")
            return

        if engaged and abs(angle_error) <= cfg.capture_tolerance:
            if self._capture_since is None:
                self._capture_since = t
        else:
            self._capture_since = None

        if abs(p_nose) <= cfg.quiescent_rate:
            if self._quiescent_since is None:
                self._quiescent_since = t
        else:
            self._quiescent_since = None

        if (self._capture_since is not None
                and t - self._capture_since >= cfg.capture_dwell):
            self._fire(t, "capture")
        elif (self._quiescent_since is not None
                and t - self._quiescent_since >= cfg.quiescent_dwell):
            self._fire(t, "quiescent")
        elif elapsed >= cfg.max_delay:
            self._fire(t, "backstop")

    def _fire(self, t: float, reason: str) -> None:
        self.released = True
        self.release_time = t
        self.release_reason = reason
        if self.log_released:
            self.log_released[-1] = True
        if self.controller is not None and self.stage2_plant is not None:
            self.controller.plant = self.stage2_plant

    def summary(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "release_time": self.release_time,
            "release_delay_s": self.delay,
            "release_reason": self.release_reason,
            "samples": self.samples,
        }


# ===========================================================================
# The 6-DOF seam
# ===========================================================================
class BrakeLaw:
    """
    Adapter that lets the closed loop drive the full 6-DOF without breaking
    the purity of the derivative.

    `sim.canards.NoseAssembly.brake_command` is a function of TIME ALONE, and
    step 2.5 made that a deliberate restriction so that nothing state-
    dependent could leak into a derivative that RK4 evaluates four times per
    step at three different times. A digital controller does not need it to
    leak: it samples at fixed instants and HOLDS its output in between, so
    inside one integration step the command really is a constant.

    So `command` is passed as `brake_command` and returns the held value
    regardless of the time it is given, while `sample` is called once per
    completed step by `sim.integrate.integrate(step_hook=...)`. The seam stays
    exactly where step 2.5 left it and no equation changes.

    `samples` counts updates; a run that finishes with zero has forgotten to
    wire the hook up, and the harness asserts against that.
    """

    def __init__(self, controller: RollAngleController,
                 command: Callable[[float], tuple],
                 deploy_time: float = 0.0,
                 sample_rate: Optional[float] = None,
                 deployment: Optional["StagedDeployment"] = None,
                 nav=None):
        self.controller = controller
        self.command = command
        self.deploy_time = float(deploy_time)
        #: Step 5. When None the loop reads `phi_nose` from TRUTH, which is
        #: what every number in docs/CONTROL-CHARACTERISATION.md was measured
        #: at and what step 3's CEP excludes. When a `gnc.navigation`
        #: NavigationSystem is supplied the loop reads its ESTIMATE instead --
        #: roll angle, both rates, and the flight condition the gains are
        #: scheduled on. Nothing else changes, so the difference between two
        #: runs that differ only in this argument IS the navigation
        #: contribution, isolated.
        self.nav = nav
        #: Optional stage-2 release law, sampled on the same grid as the loop
        #: and immediately after it, so it sees this sample's error and
        #: engagement state. None is the single-stage kit.
        self.deployment = deployment
        self.period = 1.0 / (sample_rate or controller.config.sample_rate)
        self._u = 0.0
        self._next = -math.inf
        self.samples = 0
        self.log_t: list = []
        self.log_phi: list = []
        self.log_cmd: list = []
        self.log_p_nose: list = []
        self.log_u: list = []
        self.log_err: list = []
        self.log_hold: list = []
        self.log_sat: list = []

    # -- the two halves of the seam ---------------------------------------
    def brake_command(self, t: float) -> float:
        """Held between samples. Deliberately ignores `t`."""
        return self._u

    def sample(self, t: float, y: np.ndarray, model) -> None:
        """
        Called once per completed integration step by
        `sim.integrate.integrate(step_hook=...)`. Updates only on its own
        sample grid, and only then pays for an `AeroState`.
        """
        if t < self.deploy_time:
            self._u = 0.0
            self._next = self.deploy_time
            return
        if t < self._next:
            return
        self._next = (self.deploy_time if self._next == -math.inf else self._next) + self.period
        if self._next <= t:
            self._next = t + self.period

        # phi_nose is the EARTH-referenced nose roll angle, and it is defined
        # here exactly as the authority sweep defined it -- the body 3-2-1
        # Euler roll plus the relative nose angle. Step 3 consumes the
        # envelope in that convention, so the servo has to control that
        # variable and not the integral of the nose inertial rate, which
        # differs from it by the Euler kinematic term.
        if self.nav is None:
            phi_nose = _body_roll_angle(y[6:10]) + float(y[13])
            p_body = float(y[10])
            p_nose = p_body + float(y[14])
            st = _aero_state(t, y, model)
        else:
            # The estimate, and ALL of it. Scheduling the gains on the true
            # dynamic pressure while tracking an estimated angle would price
            # half the navigation error and call it the whole.
            phi_nose, p_nose, p_body = self.nav.roll()
            st = _aero_state(t, self.nav.state(), model)
        cond = FlightCondition(time=t, qbar=st.dynamic_pressure,
                               airspeed=st.airspeed, mach=st.mach,
                               body_spin=p_body)
        phi_cmd, holding = self.command(t)
        self._u = self.controller.update(phi_cmd, phi_nose, p_nose, p_body,
                                         cond, holding=holding)
        self.samples += 1

        cst = self.controller.state
        if self.deployment is not None:
            self.deployment.update(t, cst.angle_error, p_nose,
                                   engaged=cst.engaged)
        self.log_t.append(t)
        self.log_phi.append(phi_nose)
        self.log_cmd.append(phi_cmd)
        self.log_p_nose.append(p_nose)
        self.log_u.append(self._u)
        self.log_err.append(cst.angle_error)
        self.log_hold.append(bool(holding))
        self.log_sat.append(bool(cst.saturated_high))

    def history(self) -> dict:
        return {
            "t": np.array(self.log_t),
            "phi_nose": np.array(self.log_phi),
            "phi_command": np.array(self.log_cmd),
            "p_nose": np.array(self.log_p_nose),
            "brake": np.array(self.log_u),
            "error": np.array(self.log_err),
            "holding": np.array(self.log_hold),
            "saturated": np.array(self.log_sat),
        }


# ===========================================================================
# The reduced-order closed-loop simulation
# ===========================================================================
@dataclass
class ReducedRun:
    """Time histories from `simulate_reduced`. Angles in rad, torques in N m."""

    t: np.ndarray
    phi_nose: np.ndarray
    phi_command: np.ndarray
    p_nose: np.ndarray
    p_body: np.ndarray
    brake: np.ndarray
    error: np.ndarray
    holding: np.ndarray
    saturated: np.ndarray
    hold_command: np.ndarray
    integral: np.ndarray

    @property
    def error_deg(self) -> np.ndarray:
        return np.degrees(self.error)

    def window(self, t0: float, t1: float) -> "ReducedRun":
        m = (self.t >= t0) & (self.t <= t1)
        return ReducedRun(*(getattr(self, f.name)[m] for f in
                            self.__dataclass_fields__.values()))


def simulate_reduced(plant: NoseRollPlant,
                     controller: RollAngleController,
                     schedule: ConditionSchedule,
                     command: Callable[[float], tuple],
                     t0: float, t1: float,
                     phi0: float = 0.0,
                     p_nose0: Optional[float] = None,
                     substeps: int = 4,
                     truth: Optional[NoseRollPlant] = None,
                     deployment: Optional["StagedDeployment"] = None) -> ReducedRun:
    """
    Integrate the reduced nose model in closed loop.

    `plant` is the controller model; `truth` is the plant actually integrated,
    defaulting to `plant`. Splitting them is what Task E varies.

    The controller runs at its own sample rate and the plant is advanced by
    RK4 in `substeps` per sample, which is what keeps the tanh sign
    regularisation resolved through the deployment transient where the
    relative rate sweeps through zero at several hundred rad/s^2.

    Three states: the nose inertial roll angle, the nose inertial rate, and
    the body spin. The body spin is integrated rather than tabulated because
    the brake reaction is not negligible: a held 0.5 N m against the body
    axial inertia is 3.4 rad/s^2, which over a 43 s guided phase is a tenth of
    the muzzle spin.
    """
    true_plant = truth or plant
    # Staged deployment: during stage 1 the true plant has two panels out, so
    # half the aerodynamic roll damping and the same cant torque. Both copies
    # are built once; the law says which is live at each sample.
    truth_stage1 = (replace(true_plant, deployed_panels=2)
                    if deployment is not None else true_plant)
    dt_c = controller.config.dt
    n = max(1, int(round((t1 - t0) / dt_c)))
    h = dt_c / substeps

    phi = float(phi0)
    p_body = schedule.at(t0).body_spin
    c0 = schedule.at(t0, p_body)
    p_nose = float(p_body if p_nose0 is None else p_nose0)

    out_t = np.empty(n + 1)
    out_phi = np.empty(n + 1)
    out_cmd = np.empty(n + 1)
    out_pn = np.empty(n + 1)
    out_pb = np.empty(n + 1)
    out_u = np.empty(n + 1)
    out_e = np.empty(n + 1)
    out_h = np.zeros(n + 1, dtype=bool)
    out_s = np.zeros(n + 1, dtype=bool)
    out_uh = np.empty(n + 1)
    out_ki = np.empty(n + 1)

    t = t0
    for k in range(n + 1):
        cond = schedule.at(t, p_body)
        phi_cmd, holding = command(t)
        u = controller.update(phi_cmd, phi, p_nose, p_body, cond, holding=holding)
        if deployment is not None:
            deployment.update(t, controller.state.angle_error, p_nose,
                              engaged=controller.state.engaged)
        active = (true_plant if deployment is None or deployment.released
                  else truth_stage1)

        out_t[k] = t
        out_phi[k] = phi
        out_cmd[k] = phi_cmd
        out_pn[k] = p_nose
        out_pb[k] = p_body
        out_u[k] = u
        out_e[k] = wrap_pi(phi_cmd - phi)
        out_h[k] = holding
        out_s[k] = controller.state.saturated_high
        out_uh[k] = active.hold_command(cond)
        out_ki[k] = controller.state.integral

        if k == n:
            break

        # Plant advance under the held command. The freestream coefficients
        # are frozen across the sample: dynamic pressure moves by well under a
        # part in a thousand in two milliseconds, and evaluating the schedule
        # and the panel lift-curve slope inside every RK4 stage costs a factor
        # of sixteen for that.
        t_cant = active.cant_torque(cond)
        c_aero = active.aero_damping(cond)
        k_body = schedule.damping_at(t)
        inv_In = 1.0 / active.nose.inertia
        inv_Ib = 1.0 / active.body_inertia

        for _ in range(substeps):
            def deriv(tt, phi_, pn_, pb_):
                p_rel = pn_ - pb_
                t_int = (active.bearing_torque(p_rel)
                         + active.brake_torque(u, p_rel))
                dpn = (-(t_cant + c_aero * pn_) + t_int) * inv_In
                dpb = (-k_body * pb_ - t_int) * inv_Ib
                return pn_, dpn, dpb

            a = deriv(t, phi, p_nose, p_body)
            b = deriv(t + 0.5 * h, phi + 0.5 * h * a[0], p_nose + 0.5 * h * a[1],
                      p_body + 0.5 * h * a[2])
            c_ = deriv(t + 0.5 * h, phi + 0.5 * h * b[0], p_nose + 0.5 * h * b[1],
                       p_body + 0.5 * h * b[2])
            d = deriv(t + h, phi + h * c_[0], p_nose + h * c_[1],
                      p_body + h * c_[2])
            phi += (h / 6.0) * (a[0] + 2 * b[0] + 2 * c_[0] + d[0])
            p_nose += (h / 6.0) * (a[1] + 2 * b[1] + 2 * c_[1] + d[1])
            p_body += (h / 6.0) * (a[2] + 2 * b[2] + 2 * c_[2] + d[2])
            t += h

    return ReducedRun(t=out_t, phi_nose=out_phi, phi_command=out_cmd,
                      p_nose=out_pn, p_body=out_pb, brake=out_u, error=out_e,
                      holding=out_h, saturated=out_s, hold_command=out_uh,
                      integral=out_ki)


# ===========================================================================
# Assembly helpers
# ===========================================================================
def plant_from(geometry: CanardGeometry, nose: NoseAssembly,
               projectile=None, aero_scale: float = 1.0) -> NoseRollPlant:
    """Build the reduced plant, taking the body inertia from the projectile."""
    if projectile is None:
        return NoseRollPlant(geometry=geometry, nose=nose, aero_scale=aero_scale)
    return NoseRollPlant(geometry=geometry, nose=nose, aero_scale=aero_scale,
                         body_inertia=projectile.I_axial - nose.inertia)
