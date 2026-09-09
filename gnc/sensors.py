"""
Step 5 sensor models: what the flight computer actually gets to see.

docs/SENSOR-MODELS.md is this module's account. Nothing here is a nominal
"add some noise" placeholder -- every error term is a named line on a named
part class's datasheet, graded HIGH/MEDIUM/LOW exactly as
`sim.aerodata.COEFFICIENT_CONFIDENCE` grades the aerodynamics, and the LOW
ones are printed at runtime by `warn_low_confidence()`.

THE ONE MEASUREMENT THAT DECIDES THE LAYOUT
-------------------------------------------
The body spins at 1308 rad/s at deployment on the adopted engagement --
74 952 deg/s -- and is still turning 57 360 deg/s at impact. A commodity MEMS
gyro's dynamic range is 2 000 to 4 000 deg/s. A body-mounted roll gyro is
therefore pinned at full scale for the ENTIRE flight, and pinned silently: it
returns its largest representable number, which is a perfectly plausible
reading.

Two further body-mounted numbers, measured on the adopted trajectory by
`analysis/nav_sensors.py`, not assumed:

  * a body-mounted accelerometer 30 mm off the spin axis sees 5 235 g of
    CENTRIPETAL acceleration, against a 2.11 g flight signal. The thing it is
    there to measure is 0.04 % of what it is exposed to.
  * the earth's field sweeps past a body-mounted magnetometer at 208 Hz.

In the despun section all three collapse. The nose settles below 4 000 deg/s
at t_dep + 2.00 s and below 2 000 deg/s at t_dep + 2.10 s, its centripetal
term at 30 mm is 0.00 g, and its transverse rates never exceed 22 deg/s.

AND ONE EXACT IDENTITY THAT MAKES IT MORE THAN CONVENIENT
---------------------------------------------------------
The nose frame is the body frame rotated by `phi_rel` about the common x
axis, so

    C_N^E = C_B^E Rx(phi_rel) = Rz(psi) Ry(theta) Rx(phi_body) Rx(phi_rel)
          = Rz(psi) Ry(theta) Rx(phi_body + phi_rel)

The despun IMU's OWN 3-2-1 Euler roll is therefore identically
`phi_body + phi_rel` = `phi_nose`, the variable `gnc.roll_control.BrakeLaw`
controls, and its pitch and yaw are identically the body's. Verified to
6.9e-12 rad on a flown trajectory by
`test_despun_euler_roll_is_the_controlled_variable`. The despun IMU does not
INFER the servo's controlled variable from something else; it IS that
variable, and it hands guidance the body pitch and yaw for free.

WHAT IT COSTS
-------------
Volume, in the part of the assembly that has none. Section 7 of
docs/SENSOR-MODELS.md prices it.

PURITY
------
Every model is a small object holding a `numpy.random.Generator` and its own
per-flight constant errors, drawn once at construction from the part class's
tolerances. Given the same seed it produces the same flight, and it holds no
module-level mutable state. Step 6 constructs thousands of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

__all__ = [
    "SENSOR_CONFIDENCE", "warn_low_confidence", "low_confidence_keys",
    "GyroSpec", "AccelSpec", "MagSpec", "ResolverSpec", "GnssSpec", "BaroSpec",
    "GYRO_DESPUN", "GYRO_TACTICAL", "ACCEL_DESPUN", "MAG_AMR",
    "RESOLVER_MAGNETIC", "GNSS_L1", "BARO_MEMS",
    "GeomagneticField", "NAGPUR", "MagneticDistortion",
    "Imu", "ImuSample", "Magnetometer", "Resolver", "Gnss", "GnssFix",
    "Barometer", "CarrierRoll", "SensorSuite", "SensorTruth",
    "nose_from_body", "truth_at", "body_mounted_centripetal",
    "DEG", "HOUR", "G0", "GAUSS", "NT",
]

# ---------------------------------------------------------------------------
# Units. Written out rather than assumed, because half the datasheet lines in
# this file are quoted in deg/hr or deg/sqrt(hr) and half in SI.
# ---------------------------------------------------------------------------
def _cross3(a, b) -> np.ndarray:
    """
    Cross product of two 3-vectors.

    `numpy.cross` is general over shapes and axes and spends most of its time
    proving that; on the 3-vectors this module actually uses it is a third of
    the whole navigation runtime. Measured: 17.4 s of a 52 s profile.
    """
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


DEG = math.pi / 180.0
HOUR = 3600.0
G0 = 9.80665
GAUSS = 1.0e-4          # tesla
NT = 1.0e-9             # tesla


# ===========================================================================
# The confidence register
# ===========================================================================
#
# READ THIS BEFORE QUOTING A NUMBER OUT OF THIS MODULE.
#
# The grading rule is `sim.aerodata`'s and is about PROVENANCE, not about
# whether the value looks reasonable:
#
#   HIGH    read from the manufacturer's datasheet during this work, or a
#           physical/geometric consequence of something that was.
#   MEDIUM  a published figure for the part class taken from a secondary
#           source -- distributor summary, product page, review paper --
#           rather than from the datasheet table itself; or a well-established
#           engineering figure that is not part-specific.
#   LOW     an engineering estimate. No source states it for this part in this
#           application. These are printed at runtime by
#           `warn_low_confidence()` and they are where this model is weakest.
#
# The two LOW entries that decide a reported number are
# `mag_calibration_residual`, which sets the roll error floor of Task C, and
# `gnss_reacquire_s`, which decides whether the Task D warm start is a feature.
# Both are called out again in docs/SENSOR-MODELS.md with what would settle
# them.
#
SENSOR_CONFIDENCE = {
    # -- gyroscope, despun IMU -------------------------------------------
    "gyro_full_scale": (
        "HIGH", "ADIS16505-3 datasheet: dynamic range +-2000 deg/s minimum"),
    "gyro_arw": (
        "MEDIUM", "ADIS16505 angular random walk 0.13 deg/sqrt(hr) (x,y axes, "
                  "1 sigma), ADI figure via distributor summary"),
    "gyro_bias_stability": (
        "MEDIUM", "ADIS16505-3 in-run bias stability 7.5 deg/hr typ "
                  "(1 sigma, x axis, 25 C)"),
    "gyro_bias_repeatability": (
        "LOW", "turn-on-to-turn-on bias not separately stated for this part; "
               "taken as 5x the in-run figure, the usual tactical-MEMS ratio"),
    "gyro_scale_factor": (
        "LOW", "ADI quotes sensitivity tolerance and tempco separately; 0.5 % "
               "is an engineering roll-up over the qualification temperature band"),
    "gyro_misalignment": (
        "MEDIUM", "ADIS165xx axis-to-axis misalignment quoted at 0.25 deg for "
                  "the family"),
    "gyro_shock_survival": (
        "MEDIUM", "ADIS16505 mechanical shock survivability 14 700 m/s^2 = 1500 g. "
                  "THAT IS BELOW GUN LAUNCH -- see the note on GYRO_DESPUN"),
    # -- accelerometer ----------------------------------------------------
    "accel_full_scale": (
        "MEDIUM", "ADIS16505 accelerometer range +-392 m/s^2 (40 g); flight-phase "
                  "specific force measured at 2.11 g peak, so range is not binding"),
    "accel_vrw": (
        "LOW", "velocity random walk not found as a table line; 0.023 m/s/sqrt(s) "
               "is the usual figure for this noise-density class"),
    "accel_bias_stability": (
        "MEDIUM", "ADIS16505 accelerometer in-run bias stability 26.5 micro-m/s^2 "
                  "(2.7 micro-g), x and y axes"),
    "accel_bias_repeatability": (
        "LOW", "turn-on repeatability not stated; taken as 1 mg, the tactical-"
               "MEMS norm"),
    "accel_scale_factor": (
        "LOW", "engineering roll-up over temperature, as gyro_scale_factor"),
    "accel_misalignment": (
        "MEDIUM", "as gyro_misalignment; same die pack"),
    # -- gun launch -------------------------------------------------------
    "setback_acceleration": (
        "MEDIUM", "155 mm gun launch 10 000-20 000 g peak; the gun-hard MEMS IMU "
                  "literature and the Colibrys HS8000 part class both size above "
                  "20 000 g"),
    # -- magnetometer -----------------------------------------------------
    "mag_field_range": (
        "HIGH", "HMC1053 datasheet: field range +-6 gauss full scale"),
    "mag_sensitivity_tolerance": (
        "HIGH", "HMC1053 datasheet: sensitivity 0.8 / 1.0 / 1.2 mV/V/gauss "
                "min/typ/max, i.e. +-20 % part-to-part"),
    "mag_linearity": (
        "HIGH", "HMC1053 datasheet: linearity error 0.1 %FS at +-1 gauss "
                "(best fit straight line)"),
    "mag_hysteresis": (
        "HIGH", "HMC1053 datasheet: hysteresis 0.06 %FS, repeatability 0.1 %FS, "
                "3 sweeps across +-3 gauss"),
    "mag_cross_axis": (
        "HIGH", "HMC1053 datasheet: cross-axis effect +-3 %FS at 1 gauss cross "
                "field. This is a SOFT-IRON-shaped error inside the part itself"),
    "mag_noise_density": (
        "HIGH", "HMC1053 datasheet: 50 nV/sqrt(Hz) at 1 kHz, Vbridge 5 V, on a "
                "1.0 mV/V/gauss bridge -> 1.0 nT/sqrt(Hz)"),
    "mag_bandwidth": (
        "HIGH", "HMC1053 datasheet: magnetic signal bandwidth DC to 5 MHz. This is "
                "why an ANALOG AMR bridge and not a digital magnetometer: there is "
                "no internal output data rate for the roll modulation to alias "
                "against, and the ADC sets the sample rate"),
    "mag_tempco": (
        "HIGH", "HMC1053 datasheet: sensitivity tempco -2700 ppm/C typ (Vbridge "
                "drive), bridge offset tempco +-10 ppm/C WITH set/reset"),
    "mag_perming": (
        "HIGH", "HMC1053 datasheet: max exposed field 10 000 gauss with no perming "
                "effect on zero reading; sensitivity degrades above a 20 gauss "
                "disturbing field and a set/reset pulse restores it"),
    "mag_calibration_residual": (
        "LOW", "THE ROLL ERROR FLOOR. No source measures the residual of a hard/"
               "soft-iron calibration against a 43 kg steel shell body with a kit "
               "screwed into the fuze well. 1.5 % of field magnitude and 0.9 deg of "
               "direction is an engineering estimate from published air-cored "
               "ellipsoid-fit results, degraded for the steel. Section 5 of "
               "docs/SENSOR-MODELS.md says what would settle it"),
    "mag_field_model": (
        "MEDIUM", "IGRF/WMM main-field accuracy is conventionally quoted near "
                  "150 nT rms; crustal anomaly rather than model truncation is the "
                  "practical limit and reaches several hundred nT over ordinary "
                  "terrain"),
    "geomagnetic_site": (
        "MEDIUM", "IGRF values for Nagpur (21.15 N, 79.09 E) near epoch 2026 "
                  "recalled, not read from the IGRF calculator. The absolute field "
                  "barely matters -- what the filter sees is the model residual "
                  "above -- but the DIP ANGLE sets roll observability and is worth "
                  "checking against the calculator before flight"),
    # -- bearing resolver -------------------------------------------------
    "resolver_accuracy": (
        "MEDIUM", "magnetic rotary encoder class (AS5047P and similar): 0.35 deg "
                  "typical integral non-linearity with dynamic angle-error "
                  "compensation, 28 000 rpm rating against a required 12 500 rpm"),
    "resolver_quantisation": (
        "MEDIUM", "14-bit single-turn output = 0.022 deg"),
    # -- GNSS -------------------------------------------------------------
    "gnss_position_sigma": (
        "MEDIUM", "single-frequency L1 C/A with SBAS-class corrections, 3 m "
                  "horizontal and 5 m vertical 1 sigma: the standard open-sky "
                  "figure for the class"),
    "gnss_velocity_sigma": (
        "MEDIUM", "Doppler velocity 0.05-0.1 m/s 1 sigma open sky; 0.1 m/s carried, "
                  "degraded for the spin-modulated tracking loop"),
    "gnss_rate_latency": (
        "MEDIUM", "5 Hz solution and 100 ms of transport plus solution latency are "
                  "ordinary for a small OEM receiver. Latency is modelled "
                  "EXPLICITLY because at 500 m/s it is 50 m of position"),
    "gnss_reacquire_s": (
        "LOW", "DECIDES WHETHER THE WARM START IS A FEATURE. Published hot-start "
               "TTFF is 0.5-20 s and the munition literature reports first "
               "acquisition at approximately 20 s, but no source gives a "
               "DISTRIBUTION for a 15 000 g launch into 208 Hz spin with ephemeris "
               "pre-loaded. 3-12 s with a 6 s median is an estimate"),
    "gnss_spin_tracking": (
        "MEDIUM", "roll estimation from single-patch C/N0 modulation demonstrated "
                  "down to 29 dB-Hz, with spin rates to 300 Hz addressed in the "
                  "same literature"),
    "gnss_cn0_roll_sigma": (
        "LOW", "5 deg 1 sigma per fitted revolution is an estimate. The published "
               "work demonstrates the measurement but at spin rates, integration "
               "times and C/N0 that are not this engagement's"),
    # -- barometer --------------------------------------------------------
    "baro_noise": (
        "MEDIUM", "MS5611 class: 10 cm resolution and 1.5 mbar absolute accuracy "
                  "over temperature, which is about 12 m of pressure altitude near "
                  "sea level"),
    "baro_port_error": (
        "LOW", "THE REASON THE BAROMETER IS NOT AN ALTITUDE SOURCE. The static-port "
               "pressure coefficient on a spinning shell at M 1.5 is not published "
               "for this body. -0.15 to +0.05 of q_bar is an estimate from generic "
               "supersonic body pressure distributions; at 137 kPa that is up to "
               "20 kPa, which is over 1 500 m of pressure altitude"),
}

#: Grades that trigger the runtime banner.
_LOW_GRADES = ("LOW",)


def low_confidence_keys() -> list:
    """The register keys graded LOW, in declaration order."""
    return [k for k, (lvl, _) in SENSOR_CONFIDENCE.items() if lvl in _LOW_GRADES]


def _wrap(text: str, width: int) -> list:
    out, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    out.append(cur)
    return out or [""]


def warn_low_confidence(stream=None, full: bool = False) -> list:
    """
    Emit the standing sensor-confidence banner.

    Called by every analysis entry point in this step, exactly as
    `sim.aerodata.warn_unvalidated` is called by step 1's. The LOW entries are
    always printed; `full=True` prints the whole register.
    """
    import sys

    stream = stream if stream is not None else sys.stderr
    lines = [
        "=" * 78,
        "SENSOR MODEL CONFIDENCE -- read before quoting any navigation result",
        "=" * 78,
        "Every error term in gnc/sensors.py is a named line on a named part class.",
        "The ones graded LOW are ENGINEERING ESTIMATES: no source states them for",
        "this part in this application, and they are where the model is weakest.",
        "",
    ]
    keys = list(SENSOR_CONFIDENCE) if full else low_confidence_keys()
    width = max((len(k) for k in keys), default=0)
    for key in keys:
        level, why = SENSOR_CONFIDENCE[key]
        first, *rest = _wrap(why, max(20, 74 - width - 10))
        lines.append(f"  {key:<{width}}  {level:<7} {first}")
        for r in rest:
            lines.append(f"  {'':<{width}}  {'':<7} {r}")
    lines += [
        "",
        "  The two that decide a reported number:",
        "    mag_calibration_residual  sets the roll BIAS, which adds one for one to",
        "                              the servo's 0.58 deg tracking error",
        "                              (docs/CONTROL-CHARACTERISATION.md sec 4.3).",
        "    gnss_reacquire_s          decides whether the Task D warm start moves",
        "                              the solution inside t_dep + 3 s, and so",
        "                              whether it is a feature at all.",
        "=" * 78,
    ]
    for line in lines:
        print(line, file=stream)
    return lines


# ===========================================================================
# Part classes
# ===========================================================================
@dataclass(frozen=True)
class GyroSpec:
    """
    A rate gyro triad. Every field is a datasheet line; see SENSOR_CONFIDENCE
    for which of them a datasheet actually states.

    `full_scale` is not decoration. It is modelled explicitly, it is the whole
    reason this IMU sits in the despun section, and `Imu.sample` latches a
    saturation flag rather than letting a pinned reading pass silently.
    """

    name: str
    full_scale: float               # rad/s, symmetric
    arw: float                      # rad/sqrt(s), angle random walk
    bias_stability: float           # rad/s, 1 sigma in-run (Allan floor)
    bias_tau: float                 # s, Gauss-Markov correlation time
    bias_repeatability: float       # rad/s, 1 sigma turn-on to turn-on
    scale_factor: float             # fractional, 1 sigma
    misalignment: float             # rad, 1 sigma per off-diagonal term
    quantisation: float             # rad/s, LSB
    shock_survival_g: float         # g, unpowered
    gun_hard: bool                  # does it survive 155 mm setback as sold?


@dataclass(frozen=True)
class AccelSpec:
    """An accelerometer triad. SI units throughout."""

    name: str
    full_scale: float               # m/s^2
    vrw: float                      # (m/s)/sqrt(s), velocity random walk
    bias_stability: float           # m/s^2, 1 sigma in-run
    bias_tau: float                 # s
    bias_repeatability: float       # m/s^2, 1 sigma turn-on to turn-on
    scale_factor: float             # fractional, 1 sigma
    misalignment: float             # rad, 1 sigma
    quantisation: float             # m/s^2, LSB


@dataclass(frozen=True)
class MagSpec:
    """
    A three-axis magnetometer. The HMC1053 numbers are datasheet lines, in the
    datasheet's own units, converted to SI here.

    NOTE WHAT THE PERCENTAGES ARE OF. `linearity`, `hysteresis` and
    `cross_axis` are quoted as a fraction of FULL SCALE, and full scale is
    +-6 gauss = 6e-4 T. So 0.1 %FS is 600 nT, which against a 46 000 nT field
    is 1.3 % of signal -- thirteen times what the same number reads as if it
    is mistaken for a fraction of the measurement. Reading these as
    percentages of signal understates the magnetometer error by an order of
    magnitude, and it is the easiest error to make in this file.
    """

    name: str
    full_scale: float               # T
    noise_density: float            # T/sqrt(Hz)
    linearity: float                # fraction OF FULL SCALE
    hysteresis: float               # fraction OF FULL SCALE
    cross_axis: float               # fraction OF FULL SCALE per unit cross field
    sensitivity_tolerance: float    # fractional, part to part
    sensitivity_tempco: float       # 1/K
    offset_tempco: float            # fraction of FS per K, with set/reset
    bandwidth: float                # Hz


@dataclass(frozen=True)
class ResolverSpec:
    """
    The bearing angle pick-off across the despun joint.

    The kit needs this part whatever the navigation architecture: the servo's
    brake torque acts on the RELATIVE rate and its sign is the sign of
    `p_rel`, so `gnc.roll_control.RollAngleController.update` cannot run
    without it. Navigation gets it for free, and it is what turns the despun
    IMU's `phi_nose` into the body's `phi_body = phi_nose - phi_rel` -- a
    quantity no gyro in this kit can measure directly, because the body turns
    at 74 952 deg/s.
    """

    name: str
    accuracy: float                 # rad, 1 sigma integral non-linearity
    quantisation: float             # rad, LSB
    max_rate: float                 # rad/s
    rate_noise: float               # rad/s, 1 sigma on the differentiated rate


@dataclass(frozen=True)
class GnssSpec:
    """
    A single-frequency L1 receiver on a body-mounted patch antenna.

    `reacquire_*` is TIME FROM MUZZLE EXIT to the first valid fix, not
    cold-start TTFF. The receiver is powered and tracking on the gun line
    before firing, loses lock through setback, and has to reacquire under
    208 Hz spin and 500 m/s of Doppler. It is the lowest-confidence number in
    this module and Task D turns on it.
    """

    name: str
    sigma_horizontal: float         # m, 1 sigma per horizontal axis
    sigma_vertical: float           # m, 1 sigma
    sigma_velocity: float           # m/s, 1 sigma per axis
    rate_hz: float
    latency: float                  # s
    reacquire_min: float            # s after muzzle exit
    reacquire_median: float
    reacquire_max: float
    #: Correlation time of the slowly varying part of the position error --
    #: ionosphere, ephemeris, multipath geometry. A GNSS error is NOT white at
    #: 5 Hz, and treating it as white is the classic way to build a filter
    #: that passes its own NIS test and is nonetheless wrong.
    error_tau: float                # s
    #: Fraction of the position error variance that is that correlated part.
    correlated_fraction: float
    #: Pre-detection integration time, s. Sets how much of the spinning
    #: antenna's lever-arm velocity the receiver averages away.
    integration_time: float = 0.020


@dataclass(frozen=True)
class BaroSpec:
    """
    A MEMS absolute pressure sensor plumbed to a static port.

    `port_coefficient_*` is the pressure coefficient at the port,
    Cp = (p_port - p_static) / q_bar. It is the whole reason the barometer is
    an aiding sensor of last resort and not an altitude source: at M 1.5 and
    137 kPa of dynamic pressure a Cp of -0.15 is -20 kPa, which is over
    1 500 m of pressure altitude.
    """

    name: str
    noise: float                    # Pa, 1 sigma white
    bias: float                     # Pa, 1 sigma turn-on
    rate_hz: float
    port_coefficient_mean: float    # -
    port_coefficient_sigma: float   # -


# -- the adopted parts ------------------------------------------------------
#
# GYRO_DESPUN is the baseline: an ADIS16505-3 class industrial MEMS IMU,
# chosen for its +-2000 deg/s range, which the despun section needs and which
# the quieter tactical parts do not have.
#
# NOTE THE gun_hard=False. The ADIS16505's stated shock survivability is
# 14 700 m/s^2, which is 1 500 g, and 155 mm setback is 10 000-20 000 g. The
# part as sold does not survive the launch. What is modelled here is its ERROR
# BEHAVIOUR, which is the right thing to carry into a CEP budget; surviving
# the gun is a hardware programme -- a gun-hard variant, or a potted and
# isolated mount -- and docs/SENSOR-MODELS.md section 2.4 carries it as an
# open item rather than burying it in a margin.
GYRO_DESPUN = GyroSpec(
    name="ADIS16505-3 class, despun section",
    full_scale=2000.0 * DEG,
    arw=0.13 * DEG / math.sqrt(HOUR),
    bias_stability=7.5 * DEG / HOUR,
    bias_tau=200.0,
    bias_repeatability=5.0 * 7.5 * DEG / HOUR,
    scale_factor=0.005,
    misalignment=0.25 * DEG,
    quantisation=2000.0 * DEG / 2 ** 15,
    shock_survival_g=1500.0,
    gun_hard=False,
)

#: The quiet alternative, carried so docs/SENSOR-MODELS.md section 3 can price
#: it rather than assert that it is worse. A STIM300-class part is 25x better
#: in bias stability and has a fifth of the range; the measured servo slews
#: reach 808 deg/s, so it saturates on 4.1 % of post-despin samples.
GYRO_TACTICAL = GyroSpec(
    name="STIM300 class, +-400 deg/s",
    full_scale=400.0 * DEG,
    arw=0.15 * DEG / math.sqrt(HOUR),
    bias_stability=0.3 * DEG / HOUR,
    bias_tau=1000.0,
    bias_repeatability=4.0 * 0.3 * DEG / HOUR,
    scale_factor=0.002,
    misalignment=0.20 * DEG,
    quantisation=400.0 * DEG / 2 ** 15,
    shock_survival_g=1500.0,
    gun_hard=False,
)

ACCEL_DESPUN = AccelSpec(
    name="ADIS16505 class, despun section",
    full_scale=392.0,
    vrw=0.023,
    bias_stability=26.5e-6,
    bias_tau=200.0,
    bias_repeatability=1.0e-3 * G0,
    scale_factor=0.005,
    misalignment=0.25 * DEG,
    quantisation=392.0 / 2 ** 15,
)

MAG_AMR = MagSpec(
    name="HMC1053 class analog AMR bridge",
    full_scale=6.0 * GAUSS,
    noise_density=1.0 * NT,
    linearity=0.001,
    hysteresis=0.0006,
    cross_axis=0.03,
    sensitivity_tolerance=0.20,
    sensitivity_tempco=-2700e-6,
    offset_tempco=10e-6,
    bandwidth=5.0e6,
)

RESOLVER_MAGNETIC = ResolverSpec(
    name="AS5047P class magnetic rotary encoder",
    accuracy=0.35 * DEG,
    quantisation=2.0 * math.pi / 2 ** 14,
    max_rate=28000.0 * 2.0 * math.pi / 60.0,
    rate_noise=2.0 * DEG,
)

GNSS_L1 = GnssSpec(
    name="single-frequency L1 C/A, SBAS-class corrections, body patch antenna",
    sigma_horizontal=3.0,
    sigma_vertical=5.0,
    sigma_velocity=0.10,
    rate_hz=5.0,
    latency=0.10,
    reacquire_min=3.0,
    reacquire_median=6.0,
    reacquire_max=12.0,
    error_tau=100.0,
    correlated_fraction=0.8,
    integration_time=0.020,
)

BARO_MEMS = BaroSpec(
    name="MS5611 class absolute pressure sensor",
    noise=10.0,
    bias=150.0,
    rate_hz=25.0,
    port_coefficient_mean=-0.05,
    port_coefficient_sigma=0.05,
)


# ===========================================================================
# Geometry: the nose frame
# ===========================================================================
def nose_from_body(phi_rel: float) -> np.ndarray:
    """
    Rotation taking NOSE-frame components to BODY-frame components.

    The nose frame is the body frame rotated by `phi_rel` about the shared x
    axis, so C_N^E = C_B^E @ nose_from_body(phi_rel) and the nose's 3-2-1
    Euler roll is identically phi_body + phi_rel. Its transpose takes body
    components to nose components, which is what the sensor models need.
    """
    c, s = math.cos(phi_rel), math.sin(phi_rel)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


def body_mounted_centripetal(spin: float, radius: float) -> float:
    """
    Centripetal specific force at `radius` off the spin axis, m/s^2.

    Exists so docs/SENSOR-MODELS.md section 7 can price the body-mounted
    alternative with the same expression the despun case uses, rather than
    quote a number from prose. At the measured deployment spin of 1308 rad/s
    and 30 mm this is 5 235 g against a 2.11 g flight signal.
    """
    return spin * spin * radius


# ===========================================================================
# The geomagnetic field
# ===========================================================================
@dataclass(frozen=True)
class GeomagneticField:
    """
    The earth's field at a firing site, in the simulator's TRAJECTORY frame.

    The simulator's earth frame is not NED-with-X-north: SIXDOFSPEC.md section 1
    puts X along the AZIMUTH OF FIRE, Y to the right and Z down. So the field
    is built in true NED from (intensity, declination, inclination) and then
    rotated by the firing azimuth. The quantity that matters for roll sensing
    is therefore not the declination but `declination - azimuth`, the angle
    between magnetic north and the gun line, and section 4 of
    docs/SENSOR-MODELS.md sweeps it.

    A first-order dipole gradient is carried so the truth field is not exactly
    constant over the trajectory. Over 16 km of range and 3 km of altitude it
    moves the field by a few tenths of a per cent, which is far below the
    calibration residual -- it is here so that the claim can be a measurement
    rather than an assertion.
    """

    intensity: float = 46_500.0 * NT      # T, total field
    declination: float = 0.0 * DEG        # rad, east of true north
    inclination: float = 33.0 * DEG       # rad, positive downward
    azimuth: float = 0.0 * DEG            # rad, gun line east of true north
    #: Height at which `intensity` applies, m above the muzzle plane.
    reference_altitude: float = 0.0
    #: Fractional change in field magnitude per metre of altitude. The centred
    #: dipole gives -3/R_earth = -4.7e-7 per metre.
    altitude_gradient: float = -3.0 / 6_371_000.0

    def field_ned(self, position: np.ndarray) -> np.ndarray:
        """
        True field vector in the trajectory frame at `position` (x, y, z), T.

        z is DOWN, so altitude is -z.
        """
        c_i, s_i = math.cos(self.inclination), math.sin(self.inclination)
        ang = self.declination - self.azimuth
        f = self.intensity * (1.0 + self.altitude_gradient
                              * (-float(position[2]) - self.reference_altitude))
        return np.array([f * c_i * math.cos(ang),
                         f * c_i * math.sin(ang),
                         f * s_i])

    def axis_angle(self, position: np.ndarray, x_body_earth: np.ndarray) -> float:
        """
        Angle between the field and the projectile's own axis, rad.

        THIS IS THE ROLL OBSERVABILITY GEOMETRY, and it is why the magnetometer
        is a good roll sensor on some azimuths and a poor one on others. Only
        the component of the field TRANSVERSE to the body axis is modulated by
        roll; its magnitude is |B| sin(this angle). As the angle goes to zero
        the roll signal vanishes and any measurement error is divided by
        sin(angle) on its way into the roll estimate.
        """
        b = self.field_ned(position)
        n = float(np.linalg.norm(b))
        if n <= 0.0:
            return 0.0
        c = float(np.dot(b, x_body_earth)) / (n * float(np.linalg.norm(x_body_earth)))
        return math.acos(max(-1.0, min(1.0, c)))


#: The reference site: Yantra India Limited, Ambajhari, Nagpur -- 21.15 N,
#: 79.09 E, the department that published the problem statement. IGRF near
#: epoch 2026 gives roughly 46 500 nT total, declination about 0 deg (India
#: sits close to the agonic line) and inclination about 33 deg.
#:
#: GRADED MEDIUM: these are recalled, not read from the IGRF calculator. The
#: absolute values barely affect any result in this step -- what the filter
#: sees is the MODEL RESIDUAL, not the field -- but the inclination sets roll
#: observability through `axis_angle` and is worth checking before flight.
#:
#: There is no inconsistency with `analysis.roll_servo.LATITUDE_DEG = 45`: the
#: adopted flight model is built with `include_coriolis=False`, so latitude
#: enters no equation of motion, and the field model and the dynamics are
#: decoupled. Section 4 of docs/SENSOR-MODELS.md says so and sweeps both.
NAGPUR = GeomagneticField(
    intensity=46_500.0 * NT,
    declination=0.0 * DEG,
    inclination=33.0 * DEG,
)


@dataclass
class MagneticDistortion:
    """
    Hard and soft iron, and what is left of them after calibration.

    THE MEASUREMENT MODEL. A magnetometer inside a ferrous body reads

        B_meas = A_soft @ B_true + b_hard

    with `A_soft` near identity and `b_hard` a body-fixed offset. On a 43 kg
    steel shell with a kit screwed into the fuze well both terms are LARGE --
    the hard-iron offset is tens of per cent of the earth's field, not a
    trim -- and neither is knowable from the datasheet. They are properties of
    THIS assembly and have to be measured on it.

    THE CALIBRATION, AND WHY ITS RESIDUAL IS THE ERROR FLOOR. The standard
    procedure fits an ellipsoid to the locus swept out by the sensor as the
    body is rotated through all attitudes, and inverts it. What the filter
    actually applies is the fit, so what the filter actually SEES is the
    residual

        B_seen = (A_soft_hat^-1 A_soft) B_true + A_soft_hat^-1 (b_hard - b_hard_hat)
               = (I + E) B_true + e

    and this class models `E` and `e` directly, because they are what matters
    and because modelling the two large terms separately and then cancelling
    them to three digits would be arithmetic theatre.

    `residual_fraction` sets both: `e` is drawn per axis at that fraction of
    the field magnitude, and `E` at the same fraction in scale and skew. At
    1.5 % of a 46 500 nT field that is 700 nT of offset, whose direct
    consequence is atan(0.015) = 0.86 deg of field-direction error -- so the
    0.9 deg quoted in the register is not an independent term, it is this one
    expressed as an angle.

    `full` returns the UNCALIBRATED distortion, which is what Task G's
    magnetometer-disturbance case applies: a calibration invalidated in flight
    is not a slightly worse calibration, it is no calibration.
    """

    residual_offset: np.ndarray            # T, per axis, body/nose frame
    residual_matrix: np.ndarray            # 3x3, near identity
    raw_offset: np.ndarray                 # T, the uncalibrated hard iron
    raw_matrix: np.ndarray                 # 3x3, the uncalibrated soft iron

    def apply(self, b_true: np.ndarray, calibrated: bool = True) -> np.ndarray:
        if calibrated:
            return self.residual_matrix @ b_true + self.residual_offset
        return self.raw_matrix @ b_true + self.raw_offset

    @classmethod
    def draw(cls, rng: np.random.Generator, field_magnitude: float,
             residual_fraction: float = 0.015,
             hard_iron_fraction: float = 0.25,
             soft_iron_fraction: float = 0.12) -> "MagneticDistortion":
        """
        One assembly's distortion and one calibration of it.

        `hard_iron_fraction` and `soft_iron_fraction` size the UNCALIBRATED
        terms -- 25 % and 12 % of the earth's field for a steel shell -- and
        are used only by Task G. Every nominal result depends on
        `residual_fraction` alone.
        """
        f = field_magnitude
        e_off = rng.normal(0.0, residual_fraction * f, 3)
        e_mat = np.eye(3) + rng.normal(0.0, residual_fraction, (3, 3))
        r_off = rng.normal(0.0, hard_iron_fraction * f, 3)
        r_mat = np.eye(3) + rng.normal(0.0, soft_iron_fraction, (3, 3))
        return cls(residual_offset=e_off, residual_matrix=e_mat,
                   raw_offset=r_off, raw_matrix=r_mat)


# ===========================================================================
# Error processes
# ===========================================================================
def _gauss_markov(x: np.ndarray, dt: float, tau: float, sigma: float,
                  rng: np.random.Generator) -> np.ndarray:
    """
    One step of a first-order Gauss-Markov process with steady-state 1 sigma
    `sigma` and correlation time `tau`. Exact discretisation, so the result
    does not depend on the step size.
    """
    if tau <= 0.0:
        return rng.normal(0.0, sigma, x.shape)
    a = math.exp(-dt / tau)
    q = sigma * math.sqrt(max(0.0, 1.0 - a * a))
    return a * x + rng.normal(0.0, q, x.shape)


def _misalignment_matrix(rng: np.random.Generator, sigma_align: float,
                         sigma_scale: float) -> np.ndarray:
    """
    The combined scale-factor and misalignment matrix of a sensor triad.

    Diagonal terms are 1 + scale factor error; off-diagonal terms are the
    misalignment angles. Drawn once per flight: these are manufacturing
    properties, not noise, and a filter that treats them as noise will be
    optimistic.
    """
    m = rng.normal(0.0, sigma_align, (3, 3))
    np.fill_diagonal(m, rng.normal(0.0, sigma_scale, 3))
    return np.eye(3) + m


# ===========================================================================
# Truth extraction: what the sensors are exposed to
# ===========================================================================
@dataclass
class SensorTruth:
    """
    Everything a sensor at the despun station is exposed to at one instant,
    computed from the 6-DOF state and its exact derivative.

    Nothing here is estimated. `truth_at` is the boundary between the
    simulator and the sensor models, and it is the only place they touch.
    """

    t: float
    position: np.ndarray            # (3,) earth, m
    velocity: np.ndarray            # (3,) earth, m/s
    quaternion: np.ndarray          # (4,) body -> earth
    omega_body: np.ndarray          # (3,) body rates, rad/s
    phi_rel: float
    p_rel: float
    #: The despun IMU's exposure, in NOSE axes.
    omega_nose: np.ndarray          # (3,) rad/s
    alpha_nose: np.ndarray          # (3,) rad/s^2, in NOSE axes
    specific_force_nose: np.ndarray  # (3,) m/s^2, at the IMU station
    field_nose: np.ndarray          # (3,) T, the true field at the magnetometer
    #: Book-keeping the analysis needs.
    spin: float                     # body axial rate, rad/s
    p_nose: float                   # nose INERTIAL axial rate, rad/s
    phi_nose: float                 # earth-referenced nose roll, rad
    phi_body: float
    antenna_position: np.ndarray    # (3,) earth, m
    antenna_velocity: np.ndarray    # (3,) earth, m/s
    axis_angle: float               # rad, field to body axis
    dynamic_pressure: float
    mach: float


def _body_roll_angle(q: np.ndarray) -> float:
    """3-2-1 Euler roll of the body. Same expression `gnc.roll_control` uses."""
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def truth_at(t: float, y: np.ndarray, model, field: GeomagneticField,
             imu_lever: np.ndarray, antenna_lever: np.ndarray,
             ydot: Optional[np.ndarray] = None) -> SensorTruth:
    """
    Assemble a `SensorTruth` from the 6-DOF state.

    `ydot` may be supplied when the caller already has it; otherwise the exact
    derivative is evaluated, which costs one aerodynamic call. That cost is
    real -- it is a second derivative evaluation per IMU sample -- and it is
    why the navigation system samples the IMU at 500 Hz rather than at the
    2 kHz integration rate.

    THE SPECIFIC FORCE. An accelerometer measures non-gravitational
    acceleration. The 6-DOF's velocity derivative is

        vdot = (1/m) R F_aero + g_ned + a_coriolis

    so the specific force at the CG is R^T (vdot - g_ned - a_coriolis) -- the
    Coriolis term is a rotating-frame kinematic term and is not sensed. It is
    subtracted explicitly rather than assumed absent, because
    `analysis.roll_servo.base_model` happens to disable Coriolis and step 6
    may not.
    """
    from sim import dynamics as dyn, frames
    from sim import atmosphere as atm

    y = np.asarray(y, dtype=float)
    if ydot is None:
        ydot = dyn.derivative(t, y, model)

    r = y[0:3].copy()
    v = y[3:6].copy()
    q = y[6:10].copy()
    om = y[10:13].copy()
    phi_rel = float(y[13])
    p_rel = float(y[14])

    R = frames.dcm_from_quat(q)                   # body -> earth
    C = nose_from_body(phi_rel)                   # nose -> body

    # -- specific force at the CG, body axes ------------------------------
    g_ned = atm.gravity_ned(float(r[2]) - model.environment.site_altitude) \
        if model.environment.include_inverse_square_gravity else np.array([0.0, 0.0, G0])
    a_cor = np.zeros(3)
    if model.environment.include_coriolis:
        om_e = atm.earth_rate_ned(model.environment.latitude)
        a_cor = atm.coriolis_acceleration(v, om_e)
    f_cg_body = R.T @ (ydot[3:6] - g_ned - a_cor)

    # -- transport to the despun IMU station ------------------------------
    # The IMU is fixed in the NOSE, so the lever arm is constant in NOSE axes
    # and the transport uses the NOSE's angular velocity and acceleration.
    om_nose_body = np.array([om[0] + p_rel, om[1], om[2]])
    al_nose_body = np.array([ydot[10] + ydot[14], ydot[11], ydot[12]])
    om_n = C.T @ om_nose_body
    # THE ANGULAR ACCELERATION IN NOSE AXES IS NOT THE ROTATED BODY-AXES ONE.
    # The nose frame turns relative to the body at `p_rel`, so the component
    # derivatives differ by the transport term
    #
    #     (dw/dt)^N = C^T (dw/dt)^B  -  w_rel^N x w^N ,   w_rel^N = [p_rel,0,0]
    #
    # Omitting it is not a small error here. The body-axes transverse angular
    # accelerations oscillate at the SPIN frequency with amplitude of order
    # p * |w_transverse| = 520 rad/s^2, and almost all of that is the frame
    # rotation rather than any real angular acceleration of the nose. Left in,
    # it inflated the modelled lever-arm term at the IMU from under 1 m/s^2 to
    # 66 m/s^2 rms -- twelve times the flight signal -- and the filter,
    # correctly compensating an acceleration that was not there, accumulated
    # 24 m/s of velocity error in half a second on otherwise perfect sensors.
    al_n = C.T @ al_nose_body - _cross3(np.array([p_rel, 0.0, 0.0]), om_n)
    f_cg_nose = C.T @ f_cg_body
    lever = np.asarray(imu_lever, dtype=float)
    f_imu = f_cg_nose + _cross3(al_n, lever) + _cross3(om_n, _cross3(om_n, lever))

    # -- the field at the magnetometer ------------------------------------
    b_earth = field.field_ned(r)
    b_body = R.T @ b_earth
    b_nose = C.T @ b_body

    # -- the GNSS antenna, body-mounted -----------------------------------
    ant = np.asarray(antenna_lever, dtype=float)
    r_ant = r + R @ ant
    v_ant = v + R @ _cross3(om, ant)

    # -- flight condition -------------------------------------------------
    alt = model.environment.site_altitude - float(r[2])
    atmo = atm.isa_atmosphere(alt)
    wind = model.wind(alt)
    v_rel = v - np.asarray(wind, dtype=float)
    speed = float(np.linalg.norm(v_rel))
    qbar = 0.5 * atmo.density * speed * speed
    mach = speed / atmo.sound_speed

    x_body_earth = R[:, 0]
    return SensorTruth(
        t=float(t), position=r, velocity=v, quaternion=q, omega_body=om,
        phi_rel=phi_rel, p_rel=p_rel,
        omega_nose=om_n, alpha_nose=al_n, specific_force_nose=f_imu,
        field_nose=b_nose,
        spin=float(om[0]), p_nose=float(om[0] + p_rel),
        phi_nose=_body_roll_angle(q) + phi_rel, phi_body=_body_roll_angle(q),
        antenna_position=r_ant, antenna_velocity=v_ant,
        axis_angle=field.axis_angle(r, x_body_earth),
        dynamic_pressure=float(qbar), mach=float(mach),
    )


# ===========================================================================
# The inertial measurement unit
# ===========================================================================
@dataclass
class ImuSample:
    """One IMU sample, in NOSE axes."""

    t: float
    gyro: np.ndarray                # (3,) rad/s
    accel: np.ndarray               # (3,) m/s^2
    gyro_saturated: bool
    accel_saturated: bool


class Imu:
    """
    A strapdown IMU in the despun section.

    Per-flight constants -- turn-on bias, scale factor, misalignment -- are
    drawn once at construction. Per-sample terms are drawn each call. The
    in-run bias is a first-order Gauss-Markov process, which is the standard
    model for the Allan-variance bias-instability floor and is what the filter
    is written to estimate.

    SATURATION IS MODELLED, NOT ASSUMED AWAY. `gyro_saturated` latches when
    any axis reaches full scale, which on the adopted trajectory happens for
    the first 2.10 s after deployment while the nose is still despinning
    through 74 952 deg/s. The reading in that interval is the clipped value,
    not a flag and not a NaN, because that is what a real part returns and the
    navigation filter has to detect it rather than be told.
    """

    def __init__(self, gyro: GyroSpec, accel: AccelSpec,
                 rng: np.random.Generator,
                 lever_arm=(0.399, 0.025, 0.0)):
        self.gyro_spec = gyro
        self.accel_spec = accel
        self.rng = rng
        self.lever_arm = np.asarray(lever_arm, dtype=float)

        self.M_g = _misalignment_matrix(rng, gyro.misalignment, gyro.scale_factor)
        self.M_a = _misalignment_matrix(rng, accel.misalignment, accel.scale_factor)
        self.b_g0 = rng.normal(0.0, gyro.bias_repeatability, 3)
        self.b_a0 = rng.normal(0.0, accel.bias_repeatability, 3)
        self.b_g = rng.normal(0.0, gyro.bias_stability, 3)
        self.b_a = rng.normal(0.0, accel.bias_stability, 3)
        self.gyro_saturated_until: Optional[float] = None
        self.saturated_samples = 0

    #: Total gyro bias at this instant: the turn-on term plus the in-run walk.
    @property
    def bias_gyro(self) -> np.ndarray:
        return self.b_g0 + self.b_g

    @property
    def bias_accel(self) -> np.ndarray:
        return self.b_a0 + self.b_a

    def sample(self, t: float, dt: float, omega_nose: np.ndarray,
               f_nose: np.ndarray) -> ImuSample:
        """
        `omega_nose` is the AVERAGE rate over the sample interval -- an
        angular increment divided by dt -- not the instantaneous rate at the
        sample instant. `SensorSuite.measure` forms it from the exact change
        in nose attitude across the interval, which is what a strapdown gyro
        actually delivers.

        THIS IS NOT A REFINEMENT. Measured with the instantaneous rate
        instead: the strapdown accumulated 5 deg/s of roll error on PERFECT
        sensors, 42 deg by mid-flight, because the nose roll rate changes by
        hundreds of deg/s inside one 2 ms sample while the brake modulates.
        Trapezoidal or coning-compensated integration of the sampled
        instantaneous rate does not fix it, because the information is not in
        the samples; an integrating gyro has it because it never samples.
        """
        g, a = self.gyro_spec, self.accel_spec
        self.b_g = _gauss_markov(self.b_g, dt, g.bias_tau, g.bias_stability, self.rng)
        self.b_a = _gauss_markov(self.b_a, dt, a.bias_tau, a.bias_stability, self.rng)

        n_g = self.rng.normal(0.0, g.arw / math.sqrt(dt), 3)
        n_a = self.rng.normal(0.0, a.vrw / math.sqrt(dt), 3)
        w = self.M_g @ np.asarray(omega_nose, dtype=float) + self.bias_gyro + n_g
        f = self.M_a @ np.asarray(f_nose, dtype=float) + self.bias_accel + n_a

        w_sat = bool(np.any(np.abs(w) >= g.full_scale))
        f_sat = bool(np.any(np.abs(f) >= a.full_scale))
        w = np.clip(w, -g.full_scale, g.full_scale)
        f = np.clip(f, -a.full_scale, a.full_scale)
        if g.quantisation > 0.0:
            w = np.round(w / g.quantisation) * g.quantisation
        if a.quantisation > 0.0:
            f = np.round(f / a.quantisation) * a.quantisation
        if w_sat:
            self.saturated_samples += 1
            self.gyro_saturated_until = float(t)
        return ImuSample(t=float(t), gyro=w, accel=f,
                         gyro_saturated=w_sat, accel_saturated=f_sat)


# ===========================================================================
# The magnetometer
# ===========================================================================
class Magnetometer:
    """
    A three-axis analog AMR bridge in the despun section, with the shell's
    hard and soft iron and one calibration of them.

    The datasheet error terms are fractions of FULL SCALE, and full scale is
    +-6 gauss against a 0.465 gauss field. `MagSpec` says so; this is where it
    bites. Linearity at 0.1 %FS is 600 nT, which is 1.3 % of the field.

    `calibrated=False` applies the RAW distortion, and is Task G's
    magnetometer-disturbance case.
    """

    def __init__(self, spec: MagSpec, field: GeomagneticField,
                 rng: np.random.Generator, sample_rate: float = 500.0,
                 residual_fraction: float = 0.015, calibrated: bool = True,
                 temperature_excursion: float = 20.0):
        self.spec = spec
        self.field = field
        self.rng = rng
        self.sample_rate = float(sample_rate)
        self.calibrated = bool(calibrated)
        self.distortion = MagneticDistortion.draw(rng, field.intensity,
                                                  residual_fraction)
        # Sensitivity tolerance is removed by the same ellipsoid fit that
        # removes the soft iron -- they are the same kind of error and the
        # calibration cannot tell them apart -- so what survives is the
        # TEMPCO, over the excursion between calibration and flight.
        dT = rng.normal(0.0, temperature_excursion)
        self.scale = 1.0 + spec.sensitivity_tempco * dT
        self.offset_drift = rng.normal(0.0, spec.offset_tempco * abs(dT), 3) * spec.full_scale
        self.linearity_gain = rng.normal(0.0, spec.linearity, 3) * spec.full_scale
        self.cross = rng.normal(0.0, spec.cross_axis / 3.0, (3, 3))
        np.fill_diagonal(self.cross, 0.0)
        #: White noise per sample from the datasheet density and the Nyquist
        #: bandwidth of the ADC, not from the part's 5 MHz signal bandwidth --
        #: the anti-alias filter sets it.
        self.sigma = spec.noise_density * math.sqrt(0.5 * self.sample_rate)

    def sample(self, b_nose: np.ndarray) -> np.ndarray:
        b = np.asarray(b_nose, dtype=float)
        m = self.distortion.apply(b, calibrated=self.calibrated)
        m = self.scale * m + self.cross @ m + self.offset_drift
        m = m + self.linearity_gain * np.tanh(m / (0.5 * self.spec.full_scale))
        m = m + self.rng.normal(0.0, self.sigma, 3)
        return np.clip(m, -self.spec.full_scale, self.spec.full_scale)


# ===========================================================================
# The bearing resolver
# ===========================================================================
class Resolver:
    """
    The angle pick-off across the despun bearing: `phi_rel` and `p_rel`.

    The kit carries this part whether or not it navigates, because
    `gnc.roll_control` needs the sign of the relative rate to know which way
    the brake acts. Navigation gets it free, and it is the ONLY route to the
    body roll angle: `phi_body = phi_nose - phi_rel`, and no gyro in this kit
    can measure the body directly at 74 952 deg/s.
    """

    def __init__(self, spec: ResolverSpec, rng: np.random.Generator):
        self.spec = spec
        self.rng = rng
        #: Integral non-linearity is a fixed function of angle, not noise. Two
        #: harmonics is the usual shape for a magnetic encoder and is what
        #: makes this a BIAS on the roll estimate rather than something the
        #: 1.35 Hz loop averages away.
        self.a1 = rng.normal(0.0, spec.accuracy / math.sqrt(2.0))
        self.a2 = rng.normal(0.0, spec.accuracy / math.sqrt(2.0))
        self.phase1 = rng.uniform(0.0, 2.0 * math.pi)
        self.phase2 = rng.uniform(0.0, 2.0 * math.pi)
        self.saturated_samples = 0

    def sample(self, phi_rel: float, p_rel: float) -> tuple:
        e = (self.a1 * math.sin(phi_rel + self.phase1)
             + self.a2 * math.sin(2.0 * phi_rel + self.phase2))
        a = phi_rel + e
        if self.spec.quantisation > 0.0:
            a = round(a / self.spec.quantisation) * self.spec.quantisation
        rate = float(p_rel) + self.rng.normal(0.0, self.spec.rate_noise)
        if abs(p_rel) > self.spec.max_rate:
            self.saturated_samples += 1
            rate = math.copysign(self.spec.max_rate, p_rel)
        return a, rate


# ===========================================================================
# GNSS
# ===========================================================================
@dataclass
class GnssFix:
    """
    One receiver solution, AS THE RECEIVER REPORTS IT.

    `t` is the time the fix is delivered to the flight computer; `t_valid` is
    the time it describes. They differ by the transport and solution latency,
    which at 500 m/s is 50 m of position. A filter that applies a fix at `t`
    is wrong by exactly that, and section 3 of docs/NAV-CONSISTENCY.md shows
    what it does to the NIS.
    """

    t: float
    t_valid: float
    position: np.ndarray            # (3,) earth, m
    velocity: np.ndarray            # (3,) earth, m/s
    sigma_position: np.ndarray      # (3,) m, the receiver's own claim
    sigma_velocity: np.ndarray      # (3,) m/s
    cn0_roll: Optional[float] = None        # rad, body roll from C/N0 modulation
    cn0_roll_sigma: Optional[float] = None  # rad


class Gnss:
    """
    A single-frequency L1 receiver on a body-mounted patch antenna.

    THREE THINGS THAT ARE NOT THE USUAL "add 3 m of noise":

    1. THE ERROR IS NOT WHITE. Ionosphere, ephemeris and multipath geometry
       vary on a hundred-second scale, so most of the position error is a slow
       bias over a 43 s flight, not something 200 fixes average down. Modelled
       as a Gauss-Markov process at `error_tau` carrying
       `correlated_fraction` of the variance. A filter tuned against white
       GNSS error passes its own NIS and is overconfident, which is exactly
       the failure mode Task E exists to catch.

    2. IT DOES NOT EXIST FOR THE FIRST FEW SECONDS. Lock is lost through
       setback and has to be reacquired under spin and Doppler.
       `reacquire_time` is drawn per flight and is the single number Task D
       turns on.

    3. THE ANTENNA IS NOT AT THE CENTRE OF MASS AND THE BODY SPINS. A
       phase centre 10 mm off the axis at 1308 rad/s moves at 13 m/s. That is
       130 times the receiver's velocity noise. It is deterministic given the
       body roll angle, so the filter subtracts it -- and the residual is then
       set by ROLL ERROR, which is how Task C's roll accuracy reaches the
       velocity solution and then the CEP. `lever_arm_compensation` selects
       whether the compensation is applied.
    """

    def __init__(self, spec: GnssSpec, rng: np.random.Generator,
                 muzzle_time: float = 0.0,
                 reacquire_time: Optional[float] = None,
                 outages: tuple = (), denied: bool = False,
                 lever_transverse: float = 0.010):
        self.spec = spec
        self.rng = rng
        #: Transverse offset of the antenna phase centre from the spin axis.
        #: Only the TRANSVERSE part is modulated by the spin.
        self.lever_transverse = float(lever_transverse)
        self.last_spin_residual = 0.0
        self.muzzle_time = float(muzzle_time)
        self.denied = bool(denied)
        #: (start, end) intervals, absolute seconds, during which no fix is
        #: produced. Task G's outage ladder.
        self.outages = tuple((float(a), float(b)) for a, b in outages)
        if reacquire_time is None:
            reacquire_time = self._draw_reacquire()
        self.reacquire_time = float(reacquire_time)

        self.period = 1.0 / spec.rate_hz
        self._next = -math.inf
        sc = math.sqrt(spec.correlated_fraction)
        sw = math.sqrt(1.0 - spec.correlated_fraction)
        self._sigma_c = np.array([spec.sigma_horizontal, spec.sigma_horizontal,
                                  spec.sigma_vertical]) * sc
        self._sigma_w = np.array([spec.sigma_horizontal, spec.sigma_horizontal,
                                  spec.sigma_vertical]) * sw
        #: Unit-variance Gauss-Markov state. Kept dimensionless so the same
        #: process serves all three axes, which have different sigmas.
        self._u = rng.normal(0.0, 1.0, 3)
        self.fixes = 0
        self.first_fix_time: Optional[float] = None

    def _draw_reacquire(self) -> float:
        """
        Triangular on (min, median, max). Triangular rather than normal
        because the quantity is bounded below by the receiver's own
        reacquisition floor and has a long upper tail, and because a
        distribution with three named parameters is honest about being an
        estimate in a way a fitted one would not be.
        """
        s = self.spec
        return float(self.rng.triangular(s.reacquire_min, s.reacquire_median,
                                         s.reacquire_max))

    def available(self, t: float) -> bool:
        if self.denied:
            return False
        if t - self.muzzle_time < self.reacquire_time:
            return False
        return not any(a <= t <= b for a, b in self.outages)

    def sample(self, t: float, dt: float, history) -> Optional[GnssFix]:
        """
        Produce a fix if one is due.

        `history` must answer `history(t_valid)` with the true
        (antenna position, antenna velocity, body roll, body spin) at the
        LATENT time. The receiver reports the past, and pretending otherwise
        is a 50 m error at this speed.
        """
        if t < self._next:
            return None
        if self._next == -math.inf:
            self._next = t
        self._next += self.period
        if self._next <= t:
            self._next = t + self.period

        # The correlated error walks whether or not a fix is produced: the
        # ionosphere does not stop during an outage.
        self._u = _gauss_markov(self._u, self.period, self.spec.error_tau,
                                1.0, self.rng)
        if not self.available(t):
            return None

        t_valid = t - self.spec.latency
        got = history(t_valid)
        if got is None:
            return None
        r_ant, v_cg, phi_body, spin = got
        lever_transverse = self.lever_transverse

        e_pos = self._u * self._sigma_c + self.rng.normal(0.0, 1.0, 3) * self._sigma_w

        # THE SPINNING ANTENNA, AND WHY THE VELOCITY IS NOT 0.1 m/s.
        #
        # A phase centre 10 mm off the axis at 1308 rad/s moves at 13 m/s, and
        # a receiver does not see that as velocity: it forms the Doppler over
        # a pre-detection interval of ~20 ms, which is 4.2 spin revolutions,
        # so most of the sinusoid AVERAGES AWAY. What survives is the part the
        # averaging window does not close, of order A / (pi * T / T_spin).
        #
        # That is why the reported velocity is the CENTRE OF MASS velocity
        # plus a residual, and not the instantaneous antenna velocity. It is
        # also why the filter applies no velocity lever-arm compensation: the
        # term it would correct has already been averaged out, and correcting
        # it again would ADD 13 m/s.
        #
        # Consequence, and it is a real one: the residual is 1.0 m/s against a
        # 0.1 m/s open-sky specification. The spin costs an order of magnitude
        # in velocity accuracy, and no amount of roll knowledge recovers it,
        # because the information was averaged away inside the receiver.
        cycles = abs(spin) * self.spec.integration_time / (2.0 * math.pi)
        amp = abs(spin) * abs(float(lever_transverse))
        resid = amp / (math.pi * max(cycles, 1.0))
        self.last_spin_residual = float(resid)
        e_vel = self.rng.normal(0.0, math.hypot(self.spec.sigma_velocity, resid), 3)

        self.fixes += 1
        if self.first_fix_time is None:
            self.first_fix_time = float(t)
        return GnssFix(
            t=float(t), t_valid=float(t_valid),
            position=np.asarray(r_ant, dtype=float) + e_pos,
            velocity=np.asarray(v_cg, dtype=float) + e_vel,
            sigma_position=np.array([self.spec.sigma_horizontal,
                                     self.spec.sigma_horizontal,
                                     self.spec.sigma_vertical]),
            sigma_velocity=np.full(3, math.hypot(self.spec.sigma_velocity,
                                                 resid)),
        )


class CarrierRoll:
    """
    Roll angle from the modulation of carrier-to-noise ratio as a body-mounted
    antenna's gain pattern sweeps past a satellite.

    Cheap -- the receiver already reports C/N0 per channel -- noisy, and
    INDEPENDENT of both the magnetometer and the gyro, which is the whole
    reason it is worth having. It measures the BODY roll angle, because the
    antenna is on the body; `phi_nose = phi_body + phi_rel` converts it, and
    the resolver supplies `phi_rel`.

    Two properties that shape the fusion:

      * it needs a full revolution to fit, so it produces one estimate per
        spin period, and at 208 Hz that is a 208 Hz measurement of a quantity
        the 1.35 Hz servo loop cares about -- ample.
      * it degrades with C/N0 and with the satellite geometry. The published
        demonstrations reach 29 dB-Hz. When GNSS is unavailable so is this,
        which is why it cannot be the roll estimate's backbone.
    """

    def __init__(self, rng: np.random.Generator, sigma: float = 5.0 * DEG,
                 min_revolutions: float = 1.0):
        self.rng = rng
        self.sigma = float(sigma)
        self.min_revolutions = float(min_revolutions)
        self.samples = 0

    def sample(self, phi_body: float, spin: float, dt: float) -> Optional[float]:
        """One estimate per completed revolution, or None."""
        if abs(spin) < 2.0 * math.pi * self.min_revolutions / max(dt, 1e-9):
            return None
        self.samples += 1
        return float(phi_body + self.rng.normal(0.0, self.sigma))


# ===========================================================================
# Barometer
# ===========================================================================
class Barometer:
    """
    A static-port absolute pressure sensor.

    IT IS NOT AN ALTITUDE SOURCE ON THIS VEHICLE AND THIS CLASS EXISTS TO SHOW
    WHY. The port sees `p_static + Cp * q_bar`, and at the deployment
    condition q_bar is 137 kPa -- comparable with the static pressure itself.
    A Cp of -0.05 +- 0.05, which is a modest estimate for a station on an
    ogive at M 1.5, is -7 +- 7 kPa, and 7 kPa near 1 500 m is about 600 m of
    pressure altitude. The sensor's own 1.5 mbar accuracy is 12 m and is
    irrelevant beside it.

    So the barometer is carried, modelled, and then USED ONLY as a coarse
    validity cross-check on the GNSS vertical channel -- which is what
    `gnc.navigation` does with it, and section 6 of docs/SENSOR-MODELS.md
    measures what including it as an altitude update would cost instead.
    """

    def __init__(self, spec: BaroSpec, rng: np.random.Generator):
        self.spec = spec
        self.rng = rng
        self.bias = float(rng.normal(0.0, spec.bias))
        self.cp = float(rng.normal(spec.port_coefficient_mean,
                                   spec.port_coefficient_sigma))
        self.period = 1.0 / spec.rate_hz
        self._next = -math.inf

    def sample(self, t: float, altitude: float, qbar: float) -> Optional[float]:
        """Indicated PRESSURE, Pa, or None if no sample is due."""
        if t < self._next:
            return None
        self._next = t if self._next == -math.inf else self._next
        self._next += self.period
        if self._next <= t:
            self._next = t + self.period
        from sim import atmosphere as atm
        p = atm.isa_atmosphere(altitude).pressure
        return float(p + self.cp * qbar + self.bias
                     + self.rng.normal(0.0, self.spec.noise))


# ===========================================================================
# The suite
# ===========================================================================
@dataclass
class SuiteConfig:
    """
    The sensor fit. Every default is a part class from the register above or a
    geometric consequence of the fuze-well kit's dimensions.
    """

    imu_rate: float = 500.0
    #: IMU station in NOSE axes, metres from the CG. The M107 CG is 2.96
    #: calibres from the nose (0.459 m); an IMU 60 mm aft of the tip is
    #: therefore 0.399 m FORWARD of it. The 25 mm transverse offset is the
    #: practical limit inside a fuze-well kit's bore, and it is the term the
    #: body-mounted comparison in docs/SENSOR-MODELS.md section 7 turns on.
    imu_lever: tuple = (0.399, 0.025, 0.0)
    #: GNSS antenna, BODY axes, metres from the CG. Aft of the bearing, so it
    #: sweeps -- which the C/N0 roll source needs. 10 mm of transverse phase
    #: centre offset is what a patch inside a fuze-well kit can be held to.
    antenna_lever: tuple = (0.439, 0.010, 0.0)
    gyro: GyroSpec = GYRO_DESPUN
    accel: AccelSpec = ACCEL_DESPUN
    mag: MagSpec = MAG_AMR
    resolver: ResolverSpec = RESOLVER_MAGNETIC
    gnss: GnssSpec = GNSS_L1
    baro: BaroSpec = BARO_MEMS
    field: GeomagneticField = NAGPUR
    #: The number the whole roll error floor rests on. See SENSOR_CONFIDENCE.
    mag_residual_fraction: float = 0.015
    mag_calibrated: bool = True
    cn0_roll_sigma: float = 5.0 * DEG
    #: Task G switches.
    gnss_outages: tuple = ()
    gnss_denied: bool = False
    gnss_reacquire: Optional[float] = None
    #: Set False to place the IMU on the BODY instead, which is the layout the
    #: architecture argument rejects. It is a switch rather than prose so
    #: section 7 can measure what it costs.
    despun_imu: bool = True


class SensorSuite:
    """
    Every sensor on the kit, driven from 6-DOF truth, deterministic given a
    seed.

    `measure(t, y, model)` is called once per IMU period by
    `gnc.navigation.NavigationSystem`, which is itself a `step_hook`. Nothing
    here reads or writes anything outside this object.

    The GNSS latency buffer is the one piece of state that is not a sensor: a
    ring of recent truth, so the receiver can report where the projectile WAS
    `latency` seconds ago rather than where it is.
    """

    def __init__(self, config: SuiteConfig, seed: int, muzzle_time: float = 0.0):
        self.config = config
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        # One child generator per sensor, so that changing one sensor's model
        # does not reshuffle every other sensor's draws and silently move a
        # measured number. Step 6 will vary these one at a time.
        ss = np.random.SeedSequence(seed)
        g_imu, g_mag, g_res, g_gnss, g_cn0, g_baro = [
            np.random.default_rng(s) for s in ss.spawn(6)]

        lever = (config.imu_lever if config.despun_imu
                 else (config.imu_lever[0], config.imu_lever[1], config.imu_lever[2]))
        self.imu = Imu(config.gyro, config.accel, g_imu, lever_arm=lever)
        self.magnetometer = Magnetometer(
            config.mag, config.field, g_mag, sample_rate=config.imu_rate,
            residual_fraction=config.mag_residual_fraction,
            calibrated=config.mag_calibrated)
        self.resolver = Resolver(config.resolver, g_res)
        self.gnss = Gnss(config.gnss, g_gnss, muzzle_time=muzzle_time,
                         reacquire_time=config.gnss_reacquire,
                         outages=config.gnss_outages, denied=config.gnss_denied,
                         lever_transverse=abs(float(config.antenna_lever[1])))
        self.carrier = CarrierRoll(g_cn0, sigma=config.cn0_roll_sigma)
        self.barometer = Barometer(config.baro, g_baro)

        self._hist_t: list = []
        self._hist: list = []
        self._hist_max = 64
        self._last_carrier = -math.inf
        #: Previous sample's true nose attitude, for the angular increment.
        self._q_nose_prev = None
        self._omega_prev_true = None
        #: Diagnostics the analysis reads back.
        self.gyro_saturated_samples = 0
        self.samples = 0

    # -- the integrating gyro --------------------------------------------
    def _increment_rate(self, tr: SensorTruth, dt: float) -> np.ndarray:
        """
        The angular increment across the sample interval, divided by dt.

        Formed from the EXACT change in nose attitude between this sample and
        the last, so it carries whatever the rate did in between. A strapdown
        gyro is an integrating device and this is the property that matters:
        the nose rate swings by hundreds of deg/s inside one 2 ms sample as
        the brake modulates, and a model that reports the instantaneous rate
        instead loses that and drifts 5 deg/s on otherwise perfect sensors.
        """
        from sim import frames
        dq_x = np.array([math.cos(0.5 * tr.phi_rel), math.sin(0.5 * tr.phi_rel),
                         0.0, 0.0])
        q_nose = frames.quat_normalize(
            frames.quat_multiply(tr.quaternion, dq_x))
        prev = self._q_nose_prev
        self._q_nose_prev = q_nose
        if prev is None or dt <= 0.0:
            return tr.omega_nose
        dq = frames.quat_multiply(frames.quat_conjugate(prev), q_nose)
        if dq[0] < 0.0:
            dq = -dq
        v = dq[1:4]
        n = float(np.linalg.norm(v))
        if n < 1e-15:
            return np.zeros(3)
        ang = 2.0 * math.atan2(n, float(dq[0]))
        return (ang / dt) * (v / n)

    def _increment_force(self, tr: SensorTruth, dt: float) -> np.ndarray:
        """
        The velocity increment across the sample interval, divided by dt.

        An accelerometer is an integrating device for the same reason a gyro
        is, and here the integration is not a detail -- it is what makes the
        filter's lever-arm compensation EXACT rather than approximate. Over
        one interval

            integral of (alpha x r) dt  =  (delta omega) x r

        identically, whatever alpha did in between. So an integrating
        accelerometer reports a tangential term that a filter differencing
        consecutive integrating-gyro samples cancels exactly.

        Measured with the instantaneous specific force instead: the
        compensation left about 1.5 m/s^2 of residual -- half the 2.9 m/s^2
        rms tangential term at the 25 mm IMU station -- and the filter's
        velocity error grew to 13 m/s while it claimed 0.1.
        """
        prev = self._omega_prev_true
        self._omega_prev_true = tr.omega_nose.copy()
        f = tr.specific_force_nose
        if prev is None or dt <= 0.0:
            return f
        lever = self.imu.lever_arm
        exact = _cross3(tr.omega_nose - prev, lever) / dt
        return f - _cross3(tr.alpha_nose, lever) + exact

    # -- the latency buffer ----------------------------------------------
    def _push(self, tr: SensorTruth) -> None:
        self._hist_t.append(tr.t)
        # The CENTRE OF MASS velocity, not the antenna's. The antenna's
        # oscillates at 208 Hz and interpolating it inside a 500 Hz ring is
        # meaningless -- and the receiver does not report it anyway.
        self._hist.append((tr.antenna_position, tr.velocity,
                           tr.phi_body, tr.spin))
        if len(self._hist_t) > self._hist_max:
            self._hist_t.pop(0)
            self._hist.pop(0)

    def _at(self, t: float):
        """Linear interpolation into the truth ring. None if not covered."""
        if not self._hist_t or t < self._hist_t[0]:
            return None
        if t >= self._hist_t[-1]:
            return self._hist[-1]
        i = int(np.searchsorted(self._hist_t, t)) - 1
        i = max(0, min(i, len(self._hist_t) - 2))
        t0, t1 = self._hist_t[i], self._hist_t[i + 1]
        w = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        a, b = self._hist[i], self._hist[i + 1]
        return (a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1]),
                a[2] + w * (b[2] - a[2]), a[3] + w * (b[3] - a[3]))

    # -- the call --------------------------------------------------------
    def measure(self, t: float, y: np.ndarray, model, dt: float,
                ydot: Optional[np.ndarray] = None) -> dict:
        """
        One measurement epoch. Returns the sensor outputs, and nothing else --
        no truth leaks out of here except in `truth`, which the analysis reads
        for differencing and the filter never touches.
        """
        cfg = self.config
        tr = truth_at(t, y, model, cfg.field, self.imu.lever_arm,
                      cfg.antenna_lever, ydot=ydot)
        self._push(tr)
        self.samples += 1

        if cfg.despun_imu:
            om = self._increment_rate(tr, dt)
            f = self._increment_force(tr, dt)
        else:
            # The rejected layout, measured rather than asserted: the same
            # part bolted to the BODY sees the body's angular velocity and the
            # centripetal term at the same 25 mm offset.
            om = np.array([tr.spin, tr.omega_body[1], tr.omega_body[2]])
            lever = np.asarray(cfg.imu_lever, dtype=float)
            f = tr.specific_force_nose + _cross3(om, _cross3(om, lever)) \
                - _cross3(tr.omega_nose, _cross3(tr.omega_nose, lever))
        imu = self.imu.sample(t, dt, om, f)
        if imu.gyro_saturated:
            self.gyro_saturated_samples += 1

        mag = self.magnetometer.sample(
            tr.field_nose if cfg.despun_imu
            else nose_from_body(tr.phi_rel) @ tr.field_nose)
        phi_rel_m, p_rel_m = self.resolver.sample(tr.phi_rel, tr.p_rel)

        fix = self.gnss.sample(t, dt, self._at)
        if fix is not None and abs(tr.spin) > 1.0:
            # One C/N0 roll estimate per revolution, reported with the fix so
            # the filter has a single measurement interface.
            # The C/N0 fit needs a whole revolution, and the interval it has
            # is the interval between FIXES, not the IMU period. Passing the
            # IMU period asks for 208 revolutions in 2 ms and never fires.
            r = self.carrier.sample(tr.phi_body, tr.spin, self.gnss.period)
            if r is not None:
                fix.cn0_roll = r
                fix.cn0_roll_sigma = self.carrier.sigma

        baro = self.barometer.sample(
            t, model.environment.site_altitude - float(tr.position[2]),
            tr.dynamic_pressure)

        return {"t": t, "imu": imu, "mag": mag, "phi_rel": phi_rel_m,
                "p_rel": p_rel_m, "gnss": fix, "baro": baro, "truth": tr}
