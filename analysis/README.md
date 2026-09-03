# analysis/ — Monte Carlo and statistics (step 6)

At step 1 this holds one file:

- `pointmass3dof.py` — an **independently written** 3-DOF point-mass
  integrator, used only as the reference for validation rung 2. It shares no
  code with `sim/` on purpose: if it imported `sim.integrate` or
  `sim.atmosphere`, a shared bug would cancel on both sides and the rung would
  pass while the model was wrong. That independence is what caught the
  quaternion-normalisation bug recorded in `docs/VALIDATION.md` §9.

Step 6 adds the Monte Carlo dispersion study here. Run it on the step-2
reduced-order model, not on the full 6-DOF: a single 6-DOF trajectory costs
~29 s, so 1000 runs would be ~8 hours per configuration.
