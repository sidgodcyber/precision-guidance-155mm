# 6-DOF Ballistic Simulator — Build Specification

**Step 1 of the software roadmap**

This is the ground-truth model. Everything downstream — the onboard MPMM,
the guidance law, the Monte Carlo that produces your CEP number — is
validated against this. If this is wrong, nothing built on it means anything.

Scope of *this* step: **uncorrected ballistic flight of a spin-stabilised
155 mm shell.** No canards, no guidance, no sensors. Get the dumb shell
flying correctly first, then add control.

---

## 1. Conventions — settle these before writing a line

Most bugs in projectile 6-DOF are frame and sign errors, not physics errors.
Write these at the top of every file and never deviate.

### Earth frame — NED, origin at the muzzle

| Axis | Direction |
|---|---|
| X | North = downrange (azimuth 0 = direction of fire) |
| Y | East = cross-range, positive right |
| Z | **Down**, positive downward |

Consequences: gravity is `+g` in Z. Altitude is `-z`. Impact is `z ≥ 0` on
the descending branch.

### Body frame — standard aerospace

| Axis | Direction |
|---|---|
| x | Forward along the spin axis, out of the nose |
| y | Right |
| z | Down |

### Attitude

Quaternion `q = [w, x, y, z]` maps **body → earth**:

```
v_earth = R(q) @ v_body
v_body  = R(q).T @ v_earth
q̇ = ½ · q ⊗ [0, ω_body]
```

Euler 3-2-1 (yaw ψ → pitch θ → roll φ). Positive pitch is nose-up because Z
is down. Quadrant elevation maps directly onto θ at launch.

**Do not use Euler angles as state.** Spin rate is ~1676 rad/s; you will hit
gimbal lock and you will lose precision. Quaternion only, renormalised every
step.

### Wind

`Wind(alt) → [north, east, 0]` in m/s, describing **the velocity of the air
itself**. A wind *from* the north is a **negative** X component. Get this
backwards and every range correction comes out with the wrong sign.

---

## 2. State vector — 13 states

```
r  = [x, y, z]        earth NED position          (m)
v  = [vx, vy, vz]     earth NED velocity          (m/s)
q  = [qw,qx,qy,qz]    attitude, body → earth      (-)
ω  = [p, q, r]        body angular rates          (rad/s)
```

Pack as a flat length-13 array. Keep a single `unpack(y)` helper so nothing
downstream indexes by hand.

---

## 3. Equations of motion

```
ṙ = v
v̇ = (1/m) · R(q) @ F_body  +  g_ned  +  a_coriolis
q̇ = ½ · q ⊗ [0, ω]
```

Rotational — Euler's equations for an **axisymmetric** body
(`I = diag(Ix, It, It)`, where `Ix` is axial and `It` transverse):

```
ṗ = L / Ix
q̇ = [ M + (It − Ix)·r·p ] / It
ṙ = [ N − (It − Ix)·p·q ] / It
```

Those cross terms are the gyroscopic coupling. They are the entire reason a
spinning shell behaves differently from a missile — do not drop them.

Gravity: use inverse-square, `g = g₀(Rₑ/(Rₑ+h))²`. Worth ~0.4% at a 12 km
apogee, which is metres at the target.

Coriolis (include it; it is tens of metres at 20+ km):

```
a_cor = −2 · Ω_ned × v
Ω_ned = ω_e · [cos(lat), 0, −sin(lat)],   ω_e = 7.292115e−5 rad/s
```

---

## 4. Aerodynamic angles

```
v_rel_body = R(q).T @ (v_ned − wind_ned)
V   = |v_rel_body|
u, v, w = v_rel_body
v_t = √(v² + w²)                    transverse component
sin δ = v_t / V                     total angle of attack
q̄  = ½ ρ V²                         dynamic pressure
S   = π d² / 4                       reference area (d = 0.155 m)
```

Guard `V → 0` and `v_t → 0` with an epsilon or the unit vectors blow up.

---

## 5. Force model — body frame

| Force | Expression |
|---|---|
| **Axial** | `X = −q̄·S·(C_X0 + C_X2·δ²)` |
| **Normal** | `[Y, Z] = −q̄·S·C_Nα · [v, w]/V` |
| **Magnus** | `[Y, Z] += q̄·S·C_Ypα · (p·d/2V) · [−w, v]/V` |

Normal force opposes the transverse velocity — it acts to reduce the angle
of attack in the *force* sense. The destabilising behaviour lives entirely
in the moment, not the force.

Magnus force is small. Its *moment* is not.

---

## 6. Moment model — body frame, about the CG

| Moment | Expression |
|---|---|
| **Overturning** | `[L,M,N] = q̄·S·d·C_Mα · [0, w, −v]/V` |
| **Magnus** | `[L,M,N] += q̄·S·d·C_Mpα · (p·d/2V) · [0, −v, −w]/V` |
| **Spin damping** | `L += q̄·S·d·C_lp · (p·d/2V)` |
| **Pitch/yaw damping** | `M += q̄·S·d·C_mq·(q·d/2V)`, `N += q̄·S·d·C_mq·(r·d/2V)` |

**Sign check for the overturning moment:** nose pitched up relative to the
flow means `w > 0`. With `C_Mα > 0` you get `M > 0`, which is further
nose-up. That is correct — for a spin-stabilised shell the centre of
pressure is *ahead* of the CG and the moment is destabilising. Gyroscopic
stiffness, not aerodynamics, is what keeps it pointed forward.

If your `C_Mα` source defines positive as *stabilising*, flip the sign.
Check this explicitly; it is the single most common error in projectile
6-DOF.

`C_lp` and `C_mq` are **negative** (they damp). If yours are positive,
you have a sign convention mismatch.

---

## 7. Aerodynamic coefficients — the real work

You need each of these as a **function of Mach**, for your specific shell:

`C_X0`, `C_X2`, `C_Nα`, `C_Mα`, `C_Ypα`, `C_Mpα`, `C_lp`, `C_mq`

Sources, in descending order of credibility:

1. **Published aeroballistic data** for an M107 / M795-class projectile —
   check ARL/BRL technical reports on DTIC. Best option if you can find a
   matching geometry.
2. **Missile DATCOM** — semi-empirical, fast, designed for exactly this.
   Gives you the whole set in one run.
3. **CFD** (OpenFOAM / SU2) — highest fidelity, slowest, and you need
   someone who can judge whether the mesh converged.

### Three traps

**Per-degree vs per-radian.** `C_Nα` and `C_Mα` are frequently published
per *degree*. Using a per-degree value in a per-radian equation puts you off
by 57×. Check the units on every table you import.

**Moment reference station.** Moment coefficients are referenced to a
specific axial station — often the nose tip or a fraction of body length,
not your CG. Transfer them:
`C_Mα|CG = C_Mα|ref + (x_ref − x_cg)/d · C_Nα`

**Reference length and area.** Almost always `d` and `πd²/4` for
projectiles, but confirm — some sources use body length.

Until you have real numbers, hard-code a representative table, mark it
`# PLACEHOLDER — NOT VALIDATED` in capitals, and make the code print a
warning on every run. A team that accidentally presents placeholder-derived
CEP figures to a jury has a bad afternoon.

---

## 8. Integration

**RK4, fixed step.** Not `solve_ivp` with loose tolerances — you need
predictable step size because of the spin.

**Step size is set by the spin rate, not the trajectory.** Muzzle spin is
~267 rev/s. You need 20–50 samples per revolution to resolve the Magnus and
gyroscopic terms:

```
dt ≈ 1e-4 s      →  ~37 samples/rev at muzzle
```

An 80-second flight is then ~800,000 steps. In pure Python that is slow. Two
consequences:

- Vectorise the derivative function; avoid object allocation in the loop.
- **Do not run the Monte Carlo on the full 6-DOF.** Use full 6-DOF for a
  handful of validation cases, and a reduced 4-DOF / modified point-mass
  model for the 1000-run dispersion study. That is genuinely how it is done
  in industry, and it is also the honest answer if a judge asks about
  compute cost.

**Renormalise the quaternion every step.** RK4 does not preserve unit norm
and the drift compounds over 800k steps.

**Impact detection:** integrate until `z ≥ 0` with `vz > 0`, then linearly
interpolate the final step back to `z = 0` for a clean impact point. Do not
just take the last step — you will quantise your impact point at ~0.08 m per
step, which is fine, but interpolating is free and cleaner.

---

## 9. Initial conditions

```
r₀ = [0, 0, 0]
θ₀ = QE (quadrant elevation), ψ₀ = azimuth, φ₀ = 0
q₀ = q_from_euler(ψ₀, θ₀, 0)
v₀ = R(q₀) @ [V_muzzle, 0, 0]
p₀ = 2π · V_muzzle / (twist_calibers · d)      right-hand rifling → positive
q₀_rate = r₀_rate = 0
```

For 1-in-20-calibre twist at 827 m/s: `p₀ = 2π·827/(20·0.155) ≈ 1676 rad/s`.

Projectile parameters to make configurable — treat these as inputs, not
constants, because the Monte Carlo will perturb every one of them:

`m`, `d`, `Ix`, `It`, `x_cg`, `V_muzzle`, `twist_calibers`, `QE`, `azimuth`

---

## 10. Validation ladder — do these in order

This is the most important section in the document. Each rung catches a
different class of bug.

**Rung 1 — Vacuum.** Zero all aero. Compare to the analytic parabola:
`R = V²sin(2θ)/g`. Must match to <0.1%. Catches integrator and frame errors.

**Rung 2 — Drag only.** Zero all moments and lift, force `α = 0`. Compare
against an independent 3-DOF point-mass script. Must match closely. Catches
atmosphere and drag-coefficient errors.

**Rung 3 — Gyroscopic stability.** Compute at the muzzle:

```
Sg = (Ix² · p²) / (2 · ρ · S · d · It · V² · C_Mα)
```

`Sg > 1` is required. Typical is 1.3–2.5 at the muzzle, rising through
flight as spin decays more slowly than velocity. If `Sg < 1` the shell
tumbles and either your inertias or your `C_Mα` are wrong.

Also check dynamic stability: `Sg > 1 / (Sd·(2 − Sd))`.

**Rung 4 — Yaw of repose and drift.** With right-hand rifling, a
spin-stabilised shell develops a small persistent yaw to the right and
consequently **drifts right**. At ~20 km expect drift on the order of
hundreds of metres.

This is your sign-error detector. If your shell drifts left, the Magnus or
overturning sign is inverted. Nothing else in the model reveals that error
as clearly.

**Rung 5 — Firing table comparison.** Range and time-of-flight versus
published data for your projectile and charge, across at least three
quadrant elevations. This is the rung that makes the model credible to a
jury.

Sanity anchors, standard 39-calibre tube:
- M107-class: MV ~684 m/s, max range ~18.1 km
- M795-class: MV ~827 m/s, max range ~22.5 km
- Base-bleed / RAP: 30 km+

If someone on your team is quoting 30 km for a standard HE round, that is
where the number came from — it needs base bleed.

**Rung 6 — Energy and norm sanity.** Quaternion norm stays within 1e-9 of
unity. No NaNs. Total angle of attack stays small (a few degrees) in
nominal flight — if δ grows without bound, the shell is tumbling.

---

## 11. Module structure

```
sim/
  frames.py        quaternion algebra, DCM, Euler conversions
  atmosphere.py    ISA 1976, gravity, wind model
  aerodata.py      Mach-interpolated coefficient tables
  projectile.py    physical properties dataclass
  dynamics.py      derivative function — forces, moments, EOM
  integrate.py     RK4, impact detection, trajectory logging
  diagnostics.py   Sg/Sd, validation rungs
run_ballistic.py   driver, plots, firing-table comparison
```

Keep `dynamics.py` **pure** — state in, derivative out, no I/O, no globals.
That single discipline is what lets you swap it into a Monte Carlo, a
hardware-in-the-loop harness, or a batch sweep later without touching it.

---

## 12. Hooks for later — design them in now

You will add these in steps 3–5. Leave the seams:

**Canard force/moment.** `dynamics.py` should accept an optional
`control_force_moment(t, state) → (F_body, M_body)` callback, defaulting to
zero. Step 1 passes nothing; step 4 passes the canard model. No restructuring
required later.

**Despun nose as a 14th state.** The roll-decoupled nose has its own roll
angle and rate relative to the body. When you model the CAA properly, that
is one extra degree of freedom plus a bearing-friction and brake-torque
model. Structure the state packing so appending is trivial.

**Truth logging.** Log everything the navigation filter will later need to
be compared against — true position, velocity, attitude, roll angle. Your
EKF development in step 5 is much easier when you can diff against truth.

---

## 13. Pitfalls, ranked by how much time they will cost you

1. **Per-degree vs per-radian coefficients.** 57× error. Silent.
2. **`C_Mα` sign convention.** Shell tumbles, or is mysteriously too stable.
3. **Wind direction sign.** Corrections go the wrong way.
4. **Quaternion convention** (body→earth vs earth→world). Everything looks
   plausible and the drift is wrong.
5. **Step size too large.** Spin aliases; Magnus terms become noise. If
   results change materially when you halve `dt`, you were under-resolved.
6. **Forgetting to renormalise `q`.** Slow, insidious drift over 800k steps.
7. **Moment reference station** not transferred to the CG.
8. **Running Monte Carlo on full 6-DOF.** Not wrong, just impossibly slow.

---

## 14. Definition of done for step 1

- [ ] All six validation rungs pass
- [ ] Range and TOF within a few percent of firing-table data at 3+ QEs
- [ ] Drift is to the right, magnitude physically plausible
- [ ] `Sg` computed and reported; > 1 throughout flight
- [ ] Results stable under halved timestep
- [ ] Aero coefficient provenance documented — real source or clearly
      marked placeholder
- [ ] `dynamics.py` is pure and accepts the control callback

When that list is green, you have a ground-truth model and you can start
step 2 — the reduced-order onboard MPMM, validated against this.
