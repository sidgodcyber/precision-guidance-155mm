# gnc/ — guidance, navigation and control (steps 3-5)

Empty at step 1.

- **Step 3, guidance:** impact-point prediction using the step-2 reduced-order
  model, producing a commanded correction direction.
- **Step 4, control:** roll-angle servo for the correction mechanism. It
  attaches to the simulator through `sim.dynamics.FlightModel.control`, an
  optional `control_force_moment(t, state, aero_state) -> (F_body, M_body)`
  callback that already exists and is unit-tested. No restructuring of
  `dynamics.py` is required.
- **Step 5, navigation:** extended Kalman filter fusing satellite navigation
  and inertial measurements. `sim.integrate.Trajectory` already logs true
  position, velocity, attitude quaternion and body rates so the filter can be
  differenced against truth.
