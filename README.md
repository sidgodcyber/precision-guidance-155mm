# 6-DOF Ballistic Simulator — 155 mm Spin-Stabilised Projectile

**Step 1 of 7**

A validated six-degree-of-freedom rigid-body flight simulator for a
spin-stabilised 155 mm artillery shell in **uncorrected ballistic flight**.

This is the ground-truth model. The reduced-order onboard model (step 2), the
impact-point-prediction guidance law (step 3), the roll-control loop (step 4),
the navigation filter (step 5) and the Monte Carlo dispersion study (step 6)
are all validated against this. If this is wrong, nothing built on it means
anything — which is why most of the effort here went into sourcing the
aerodynamic coefficients and into the validation ladder, not into the
rigid-body dynamics, which are textbook.

**Scope.** Numerical simulation and control algorithms only: flight dynamics,
atmosphere, Kalman filtering, trajectory prediction. Nothing in this
repository relates to explosives, energetics, detonation trains or initiation
circuitry, and no such work is in scope.

---

## Quick start

```bash
pip install numpy scipy matplotlib pytest      # scipy is not used by sim/, only by tooling

python run_ballistic.py                        # one trajectory + plots
python run_validation.py                       # the full six-rung ladder
python -m pytest tests -q                      # unit tests
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
analysis/
  pointmass3dof.py independent 3-DOF reference for validation rung 2
run_ballistic.py   driver, plots, firing-table comparison
run_validation.py  the six-rung validation ladder
tests/test_sim.py  unit tests
docs/
  COEFFICIENTS.md  coefficient provenance and confidence table
  VALIDATION.md    validation results, rung by rung, with numbers
```

`models/`, `gnc/` and `embedded/` are placeholders for steps 2–5 and 7.

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

## Results in one table

155 mm M107, charge 8 (684 m/s), against firing table FT 155-AM-2:

| QE (mils) | Range (m) | vs FT | TOF (s) | vs FT | Drift (m) | vs FT |
|---|---|---|---|---|---|---|
| 141.6 | 7941 | −0.73 % | 16.81 | −1.12 % | +41.7 R | +10.8 % |
| 248.4 | 10939 | −0.55 % | 26.87 | −1.21 % | +102.0 R | +10.4 % |
| 525.3 | 15841 | −1.00 % | 48.49 | −0.85 % | +317.2 R | +8.3 % |

Over all 15 firing-table points (5 charges, 2–16 km): range RMS **0.48 %**,
mean **+0.00 %**; TOF RMS 0.53 %; drift mean +10.2 %.

Drift is to the **right**, as a right-hand-rifled shell must.

And against the fully specified ASAT-13 §4.3 case, where every input is given
by the source:

| | Model | Published | Error |
|---|---|---|---|
| initial axial deceleration | 4.468 g | 4.45 g | **+0.40 %** |
| total flight time | 66.194 s | 66.67 s | **−0.71 %** |
| summit time | 30.36 s | ~31 s | −2.06 % |
| peak total angle of attack | 1.2975° | ~1.3° | −0.19 % |

See [docs/VALIDATION.md](docs/VALIDATION.md) for all seven rungs, all five
charges, the timestep-convergence study and the sensitivity cases, and
[docs/REVIEW-RESPONSE.md](docs/REVIEW-RESPONSE.md) for the response to the
external review — including the one proposed change that was **rejected**.

---

## Known limitations

Read these before quoting any number from this model.

1. **Four of eight coefficients rest on a single source.** `C_X0`, `C_Nα` and
   `C_Mα` — the three that govern range, stability and drift — are each
   confirmed by two independent sources and agree to 3–12 %. `C_Ypα`, `C_Mpα`,
   `C_mq` and `C_X2` are not. Full detail in
   [docs/COEFFICIENTS.md](docs/COEFFICIENTS.md).
2. **One convention is an assumption.** The four rate-dependent coefficients
   are applied with reduced rates pd/(2V) and qd/(2V). The classical
   aeroballistic literature uses pd/V, whose coefficients are half as large
   for the same physics. If the source table is aeroballistic-normalised,
   those four terms are 2× too small. The measured effect on range and drift
   is small; the effect on damping margins is not. `REDUCED_RATE_FACTOR` in
   `aerodata.py` is the one constant that expresses the choice.
3. **Drift runs ~10 % high** against the firing table (mean +10.2 % over 15
   points, worst at the shortest ranges where the drift is only a few metres;
   best case +0.9 %). Diagnosed, not tuned away. The subsonic `C_Nα` splice —
   adopted on the centre-of-pressure evidence, not on drift — brought the mean
   down from +13.3 %. Two further candidates were quantified and **rejected**:
   changing the rifling twist to 1/25 *inverts* the error rather than closing
   it, and a uniform `C_Nα` rescale is contradicted by BRL's own data outside
   the subsonic band. What remains open, and what evidence would settle it, is
   in [docs/VALIDATION.md](docs/VALIDATION.md) rung 4.
4. **The coefficient table stops at Mach 2.00**, and charge 8 launches at
   M 2.01. End values are held flat and the excursion is reported.
5. **Linear aerodynamics only.** `C_Mpα` at 0° yaw; no nonlinear Magnus, no
   limit-cycle modelling. Valid because nominal flight stays below 0.8° yaw.
6. **Not fast enough for Monte Carlo.** Measured 118 µs per RK4 step, so the
   48.5 s charge-8 flight at dt = 2×10⁻⁴ (242 447 steps) takes **28.7 s** of
   wall clock in pure CPython. That is by design — the spec's own guidance is to use
   full 6-DOF for validation cases and the reduced-order model of step 2 for
   the 1000-run dispersion study. `run_validation.py` parallelises across
   cores. A further ~1.5–2× is available and not taken: `integrate.rk4_step`
   still routes through the numpy-facing `dynamics.derivative`, whereas
   `dynamics._deriv_scalars` returns plain floats and avoids the small-array
   allocations. That change was deliberately not made after the validation
   ladder had been run against the current integrator.
7. **No base bleed, no rocket assist.** Ranges beyond ~24 km are out of family
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
