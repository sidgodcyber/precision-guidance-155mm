# models/ — reduced-order models (step 2, closed)

## `mpmm.py` — modified point-mass model, STANAG 4355 form

The model that would run on the projectile flight computer. Seven states
(position, velocity, axial spin), no attitude, an algebraic yaw of repose in
place of integrated rotational dynamics.

- `derivative(t, y, model) -> list[float]` is **pure**, exactly like
  `sim.dynamics.derivative`, and returns plain floats so the step-7 C port is
  a transcription rather than a redesign.
- `propagate_to_impact(y0, model, dt=0.01, ...)` runs RK4 to ground impact with
  the impact point interpolated to z = 0.
- `state_from_sixdof(y6)` reduces a 13-state 6-DOF vector to the 7-state MPMM
  vector, discarding the quaternion and the two transverse body rates. What
  that discarding costs is measured in `docs/MODEL-ERROR.md`.

### It imports its aerodynamics; it does not own them

Every coefficient comes from `sim.aerodata`, the atmosphere from
`sim.atmosphere`, the projectile from `sim.projectile`. There is no
coefficient literal anywhere in the module, and
`tests/test_mpmm.py::test_no_hardcoded_coefficients_in_the_module` parses the
module and fails if one appears. The comparison against the 6-DOF is only
meaningful while both models are fed identical inputs.

### The fitting factors are unity and stay unity

`FittingFactors` carries the four STANAG ballistic fitting parameters — form
factor `i`, lift factor `fL`, Magnus factor `QM`, yaw-drag factor `QD` — all
defaulting to 1.0. They are present so a future reader can see that they exist
and are *not* being used. `test_all_fitting_factors_are_unity` fails if any
default changes, and also if the field set grows.

Those factors exist to make a model reproduce a firing table for one
projectile lot. Using them here would have made every number in
`docs/MPMM-VALIDATION.md` and `docs/MODEL-ERROR.md` a measurement of the
fitting instead of a measurement of the model.

### Results

| | |
|---|---|
| vs 6-DOF, 15 engagements | range within **0.11 %**, deflection within **1.05 %**, TOF within **0.06 s** |
| model error from apogee | **2.3 m** 1σ range, **0.5 m** 1σ deflection (RMS) |
| the step-1 drift hypothesis | **refuted** — the MPMM tracks the 6-DOF, not the firing table |

`docs/MPMM-VALIDATION.md`, `docs/MODEL-ERROR.md`, `docs/DRIFT-RESOLUTION.md`,
`docs/MPMM-COMPUTE.md`, `docs/STEP2-CLOSEOUT.md`.
