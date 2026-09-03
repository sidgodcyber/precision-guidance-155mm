# Flight Dynamics — 155 mm Spin-Stabilised Projectile

**Steps 1–2 of 7**

Two models of the same shell:

- **`sim/` — a validated six-degree-of-freedom rigid-body simulator** for
  uncorrected ballistic flight. The ground-truth model. *(Step 1, closed.)*
- **`models/mpmm.py` — the reduced-order modified point-mass model**
  (STANAG 4355 form) that would run on the flight computer. *(Step 2, closed.)*

The 6-DOF is what everything else is validated against. The
impact-point-prediction guidance law (step 3), the roll-control loop (step 4),
the navigation filter (step 5), the Monte Carlo dispersion study (step 6) and
the embedded C port (step 7) all build on these two. If the 6-DOF is wrong,
nothing built on it means anything — which is why most of the effort in step 1
went into sourcing the aerodynamic coefficients and into the validation ladder,
not into the rigid-body dynamics, which are textbook.

**Both models are driven by the same coefficient tables, the same atmosphere
and the same projectile definition.** `models/mpmm.py` contains no coefficient
of its own, and a test enforces that by parsing the module. A comparison
between two models that had drifted apart in their inputs would measure
nothing.

**Scope.** Numerical simulation and control algorithms only: flight dynamics,
atmosphere, Kalman filtering, trajectory prediction. Nothing in this
repository relates to explosives, energetics, detonation trains or initiation
circuitry, and no such work is in scope.

---

## Quick start

```bash
pip install numpy scipy matplotlib pytest      # scipy is not used by sim/, only by tooling

python run_ballistic.py                        # one 6-DOF trajectory + plots
python run_validation.py                       # the full six-rung ladder
python -m analysis.mpmm_compare                # MPMM vs 6-DOF over the envelope
python -m analysis.mpmm_compute                # MPMM compute-cost study
python -m pytest tests -q                      # 85 unit tests
```

`run_ballistic.py` writes four figures to `docs/figures/`: trajectory
profile, ground track, angle-of-attack history, and stability diagnostics.

```bash
python run_ballistic.py --charge 8 --qe-mils 248.4    # a firing-table point
python run_ballistic.py --charge 7 --qe-deg 30 --dt 1e-4
python run_ballistic.py --wind-north -10              # 10 m/s head wind
python run_ballistic.py --no-coriolis --no-plots
```

Every run prints an aerodynamic-coefficient confidence banner first. That is
deliberate and is not suppressible: see [Known limitations](#known-limitations).

---

## What it does

| | |
|---|---|
| **State** | 13 elements: position and velocity in earth NED, attitude quaternion (body→earth), body angular rates |
| **Forces** | axial drag with yaw-drag term, normal force, Magnus force |
| **Moments** | overturning, Magnus, spin damping, pitch/yaw damping — about the CG |
| **Environment** | ISA 1976 atmosphere, inverse-square gravity, Coriolis, altitude-dependent wind |
| **Integration** | fixed-step RK4, quaternion renormalised every step, impact interpolated to z = 0 |
| **Diagnostics** | gyroscopic and dynamic stability factors, yaw of repose, full truth logging |

### Conventions

Frame and sign conventions are the single largest source of bugs in
projectile 6-DOF, so they are stated at the top of every module and never
deviated from.

- **Earth frame:** NED, origin at the muzzle. X downrange, Y right, **Z down**.
  Gravity is +Z, altitude is −z, impact is z ≥ 0 on the descending branch.
- **Body frame:** x forward out of the nose, y right, z down.
- **Attitude:** quaternion `q = [w,x,y,z]` mapping **body → earth**. Euler 3-2-1.
  Because Z is down, **positive pitch is nose-up**, and quadrant elevation maps
  directly onto θ at launch.
- **Wind** is the **velocity of the air**. A wind *from* the north is a
  **negative** X component.
- **Positive C_Mα is destabilising** — centre of pressure ahead of the CG.
  Gyroscopic stiffness, not aerodynamics, is what keeps a shell nose-forward.
- **Right-hand rifling gives positive spin p**, and the shell must **drift right**.
- **Rifling twist belongs to the gun, not the shell.** The nominal model is
  1 turn in 20 calibres (M185/M199, the tube of firing table FT 155-AM-2).
  `sim/projectile.py::TUBES` records the alternatives with their sources. At
  fixed twist the gyroscopic stability factor does not depend on muzzle
  velocity at all, so quoting an Sg without naming the tube is meaningless.

---

## Layout

```
sim/
  frames.py        quaternion algebra, DCM, Euler conversions
  atmosphere.py    ISA 1976, gravity, wind
  aerodata.py      Mach-interpolated coefficients + full provenance
  projectile.py    physical properties, launch conditions, environment
  dynamics.py      PURE derivative function: state in, derivative out
  integrate.py     RK4, impact detection, trajectory logging
  diagnostics.py   Sg, Sd, yaw of repose
models/
  mpmm.py          PURE 7-state STANAG 4355 modified point-mass derivative
analysis/
  pointmass3dof.py        independent 3-DOF reference for validation rung 2
  brl_reference.py        verified BRL MR-1582 transcription, per-digit provenance
  brl_figures.py          figure page map, axis calibration, damping-force table
  coefficient_crosscheck.py  source comparison and the centre-of-pressure test
  mpmm_compare.py         MPMM vs 6-DOF; model error; the drift test
  mpmm_compute.py         MPMM compute cost, step size, term ablation
run_ballistic.py   driver, plots, firing-table comparison
run_validation.py  the validation ladder (rungs 1-6 plus 5b)
tests/test_sim.py  6-DOF unit tests   (66)
tests/test_mpmm.py MPMM unit tests    (19)
docs/
  STEP1-CLOSEOUT.md   step 1: final results, the drift residual, fitness for step 2
  STEP2-CLOSEOUT.md   step 2: what the MPMM is worth, fitness for step 3
  VALIDATION.md       6-DOF validation, rung by rung, with numbers
  COEFFICIENTS.md     coefficient provenance and confidence, per Mach band
  DIGITISATION.md     figure digitisation: method and why it was rejected
  REVIEW-RESPONSE.md  response to the external review
  DONE-CHECKLIST.md   SIXDOFSPEC.md section 14, item by item
  MPMM-VALIDATION.md  MPMM vs 6-DOF across the firing-table envelope
  MODEL-ERROR.md      the measured model-error term for the CEP budget
  DRIFT-RESOLUTION.md the step-1 drift hypothesis, tested and refuted
  MPMM-COMPUTE.md     what one onboard propagation costs
```

`gnc/` and `embedded/` are placeholders for steps 3–5 and 7.

### `dynamics.py` is pure

State in, derivative out. No I/O, no globals, no hidden state, no mutation of
inputs. That single discipline is what lets the same function be dropped into
a Monte Carlo harness or a hardware-in-the-loop rig later without being
rewritten. It is enforced by
`test_derivative_does_not_mutate_its_input` and `test_derivative_is_deterministic`.

The force and moment model exists **exactly once**, in `_aero_core()`. Both
the hot path and the diagnostic wrapper call it. A "fast" copy and a
"readable" copy of projectile aerodynamics would eventually disagree, and a
silent sign error in one of them would still produce a plausible trajectory.

### Seams left for later steps

- **Canard force/moment (step 4):** `FlightModel.control` takes an optional
  `control_force_moment(t, state, aero_state) → (F_body, M_body)` callback,
  defaulting to `None`. Tested by `test_control_callback_adds_force_and_moment`.
- **Despun nose as a 14th state (step 4):** state packing goes through
  `pack()`/`unpack()` and `STATE_SIZE`, so appending a state touches one place.
- **Truth logging (step 5):** `Trajectory` logs true position, velocity,
  attitude quaternion, body rates and the aerodynamic state, so the EKF can be
  diffed against truth.

---

## The onboard model — `models/mpmm.py`

Seven states instead of thirteen: position, velocity, axial spin. **No
attitude.** The 6-DOF must resolve a 221 rev/s spin and is therefore stuck at
dt ≈ 2×10⁻⁴ s; the MPMM replaces integrated attitude with STANAG 4355's
**algebraic yaw of repose**

```
alpha_e = -( 8 Ix p (v x dv/dt) ) / ( pi rho d^3 C_Malpha |v|^4 )
```

and takes steps two orders of magnitude larger. `derivative()` is pure, like
`dynamics.derivative`, and returns plain floats for the C port.

### No fitting factors, and a test that says so

Operational STANAG 4355 implementations carry a form factor `i`, a lift factor
`fL`, a Magnus factor `QM` and a yaw-drag factor `QD`, fitted per projectile
lot against firing trials. **All four are present as named constants, all four
are 1.0, and `test_all_fitting_factors_are_unity` fails if any of them moves**
— it also fails if the factor set grows, so a new one cannot be added quietly.

The point is not purity for its own sake. A fitted MPMM would reproduce the
6-DOF because it had been *made* to, and the model-error number in
[docs/MODEL-ERROR.md](docs/MODEL-ERROR.md) would then be a measurement of the
fitting rather than of the model.

---

## Results in one table

155 mm M107, charge 8 (684 m/s), against firing table FT 155-AM-2:

| QE (mils) | Range (m) | vs FT | TOF (s) | vs FT | Drift (m) | vs FT |
|---|---|---|---|---|---|---|
| 141.6 | 7942 | −0.73 % | 16.81 | −1.11 % | +43.3 R | +15.1 % |
| 248.4 | 10940 | −0.55 % | 26.87 | −1.20 % | +106.4 R | +15.1 % |
| 525.3 | 15841 | −0.99 % | 48.49 | −0.84 % | +328.6 R | +12.2 % |

Over all 15 firing-table points (5 charges, 2–16 km): range RMS **0.48 %**,
mean **+0.00 %**; TOF RMS 0.52 %; drift mean +14.4 % (see limitation 3).

Drift is to the **right**, as a right-hand-rifled shell must.

And against the fully specified ASAT-13 §4.3 case, where every input is given
by the source:

| | Model | Published | Error |
|---|---|---|---|
| initial axial deceleration | 4.468 g | 4.45 g | **+0.40 %** |
| total flight time | 66.194 s | 66.67 s | **−0.71 %** |
| summit time | 30.36 s | ~31 s | −2.06 % |
| peak total angle of attack | 1.2975° | ~1.3° | −0.19 % |

### And the reduced-order model against the 6-DOF

Same 15 engagements, same coefficients, no fitting factors. Differences are
MPMM minus 6-DOF, RMS over the envelope:

| Quantity | default | with `iterate_yaw=True` |
|---|---|---|
| Range | 0.079 % (6.60 m) | **0.030 % (1.58 m)** |
| Deflection | 0.715 % (1.18 m) | **0.115 % (0.11 m)** |
| Time of flight | 0.028 s | |
| Impact velocity | 0.058 m/s | |
| Impact angle | 0.006° | |

Initialised from a true 6-DOF state at apogee and propagated to impact — which
is how step 3 will use it — the **model error is 0.65 m RMS in range and
0.06 m RMS in deflection**, against a CEP budget that had been carrying an
estimated 10 m. See [docs/MODEL-ERROR.md](docs/MODEL-ERROR.md).

One full worst-case impact prediction is **about 1050 derivative evaluations**
at the recommended dt = 0.1 s — 13 % of a 100 ms duty cycle even in CPython.
See [docs/MPMM-COMPUTE.md](docs/MPMM-COMPUTE.md).

---

**Steps 1 and 2 are closed.** See
[docs/STEP2-CLOSEOUT.md](docs/STEP2-CLOSEOUT.md) for what the reduced model is
worth and its fitness statement for step 3, and
[docs/STEP1-CLOSEOUT.md](docs/STEP1-CLOSEOUT.md) for
the final ladder, what the drift residual is and everything ruled out, and an
explicit statement of fitness for use as step-2 ground truth.
[docs/VALIDATION.md](docs/VALIDATION.md) has all seven rungs in full;
[docs/REVIEW-RESPONSE.md](docs/REVIEW-RESPONSE.md) records the external review
including the change that was **rejected**;
[docs/DIGITISATION.md](docs/DIGITISATION.md) records a figure digitisation that
was attempted and rejected.

---

## Known limitations

Read these before quoting any number from this model.

1. **Four of eight coefficients rest on a single source.** `C_X0` and `C_Mα`
   are confirmed by a second, independent, *measured* source; `C_Nα` is set by
   that measurement directly. `C_Ypα`, `C_Mpα`, `C_mq` and `C_X2` rest on the
   computed deck alone. Full detail in
   [docs/COEFFICIENTS.md](docs/COEFFICIENTS.md).
2. **One convention is an assumption, and permanently so from this source
   set.** The four rate-dependent coefficients are applied with reduced rates
   pd/(2V) and qd/(2V); the classical aeroballistic literature uses pd/V, whose
   coefficients are half as large for the same physics. The ASAT-13 paper
   cannot settle it: its §3 gives only the rigid-body equations of motion and
   defers the aerodynamic expansion to Etkin and to an unreproduced M.Sc.
   thesis. Worth < 0.25 % in range but 7–11 % in drift.
   `REDUCED_RATE_FACTOR` in `aerodata.py` is the one constant expressing the
   choice.
3. **Drift runs ~14 % high** against the firing table (mean +14.4 % over 15
   points, +5.0 % to +24.8 %, worst where the absolute drift is only a few
   metres). **Closed as an open residual with no coefficient explanation
   left.** Every candidate has been tested against measurement and eliminated:
   the twist (1/25 *inverts* the error and three sources give 1/20), C_Nα
   (replaced by the measurement — drift got *worse*), C_Mα (21 measured rows
   confirm it, ratio 1.019 ± 0.039), the missing pitch-damping force
   (quantified at 0.70 % of the normal force), Coriolis, sign errors, and
   timestep. Range, TOF, summit and impact velocity are unaffected.
   **The last remaining lead — that the FT drift column came from a
   point-mass model of a different class — was tested in step 2 and
   refuted.** A STANAG 4355 MPMM driven by the same coefficients lands at
   +14.38 % against the same column, agreeing with the 6-DOF's +14.37 % to
   0.115 % RMS. Both model classes agree with each other and both disagree
   with the firing table, so the cause is in the inputs or the reference data,
   not in the trajectory integration. See
   [docs/DRIFT-RESOLUTION.md](docs/DRIFT-RESOLUTION.md).
4. **The coefficient table stops at Mach 2.00**, and charge 8 launches at
   M 2.01. End values are held flat and the excursion is reported.
5. **Linear aerodynamics only.** `C_Mpα` at 0° yaw; no nonlinear Magnus, no
   limit-cycle modelling. Valid because nominal flight stays below 0.8° yaw.
6. **Not fast enough for Monte Carlo.** Measured 118 µs per RK4 step, so the
   48.5 s charge-8 flight at dt = 2×10⁻⁴ (242 447 steps) takes **28.7 s** of
   wall clock in pure CPython. That is by design — the spec's own guidance is to use
   full 6-DOF for validation cases and the reduced-order model for the
   1000-run dispersion study — which now exists, runs a full trajectory in
   milliseconds, and agrees with the 6-DOF to 0.03 % in range. `run_validation.py` parallelises across
   cores. A further ~1.5–2× is available and not taken: `integrate.rk4_step`
   still routes through the numpy-facing `dynamics.derivative`, whereas
   `dynamics._deriv_scalars` returns plain floats and avoids the small-array
   allocations. That change was deliberately not made after the validation
   ladder had been run against the current integrator.
7. **The MPMM's default omits the yaw-of-repose iteration**, which costs it a
   factor of 3.5 in range model error and 8 in deflection model error for a
   40 % saving in compute. `iterate_yaw=True` is recommended and is not the
   default only so the simplest closed-form derivative stays the baseline for
   the step-7 C port. It also uses a linear overturning moment: STANAG's
   (C_Mα + C_Mα3·α_e²) cubic term has no published value for the M107 in
   either source used here.
8. **No base bleed, no rocket assist.** Ranges beyond ~24 km are out of family
   for this model and for a standard HE round.

---

## Sources

- B. G. Karpov & L. E. Schmidt (rev. K. Krial, L. C. MacAllister), *The
  Aerodynamic Properties of the 155-mm Shell M101 from Free Flight Range Tests
  of Full Scale and 1/12 Scale Models*, BRL Memorandum Report 1582, Aberdeen
  Proving Ground, June 1964 (DTIC AD0454925).
- M. Khalil, H. Abdalla & O. Kamal, *Dispersion Analysis for Spinning Artillery
  Projectile*, ASAT-13, Cairo, 2009 (DOI 10.21608/asat.2009.23740).
- W. Y. Lim, *Predicting the Accuracy of Unguided Artillery Projectiles*, M.S.
  thesis, Naval Postgraduate School, 2016 (DTIC AD1029824) — source of the
  FT 155-AM-2 firing-table comparison data.
- R. L. McCoy, *Modern Exterior Ballistics: The Launch and Flight Dynamics of
  Symmetric Projectiles*, 2nd ed., Schiffer, 2012.
- R. Balon & J. Komenda, *Analysis of the 155 mm ERFB/BB Projectile
  Trajectory*, Advances in Military Technology 1/2006.
- U.S. Standard Atmosphere, 1976, NOAA/NASA/USAF.
