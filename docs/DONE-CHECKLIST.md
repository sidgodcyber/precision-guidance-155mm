# Definition of Done — SIXDOFSPEC.md Section 14

Reported item by item, honestly, including what was not achieved.
Updated after the correction pass — see [REVIEW-RESPONSE.md](REVIEW-RESPONSE.md).

| # | Item | Status | Evidence / caveat |
|---|---|---|---|
| 1 | All six validation rungs pass | **Yes** (now seven) | [VALIDATION.md](VALIDATION.md). Rungs 1 and 2 exact to machine precision; 3, 5, 5b and 6 pass with margin; rung 4 passes on **sign** (the thing it exists to test), magnitude still ~10 % high — see caveat below. |
| 2 | Range and TOF within a few percent of firing-table data at 3+ QEs | **Yes, exceeded** | 15 points across 5 charges and 2–16 km. Range RMS **0.48 %**, mean **+0.00 %**; TOF RMS **0.53 %**. Additionally rung 5b matches a fully specified published case: axial deceleration +0.40 %, flight time −0.71 %. |
| 3 | Drift is to the right, magnitude physically plausible | **Yes, with a caveat** | Right in 15/15; reversing the rifling reverses it. Magnitude **+0.9 % to +23 %** (mean +10.2 %, improved from +13.3 %; worst where drift is only ~3 m). Plausible but not tight. |
| 4 | Sg computed and reported; > 1 throughout flight | **Yes** | Sg ≥ 2.58 anywhere in any run, rising to 13.09; `Sg > 1` at every logged sample of every run. The *dynamic* criterion `Sg > 1/(Sd(2−Sd))` holds in 14 of 15 runs; the 15th reproduces a subsonic dynamic instability that BRL MR-1582 reports for this shell from measurement (VALIDATION.md rung 3). Now also stated per tube, since Sg scales as the inverse square of the twist. |
| 5 | Results stable under halved timestep | **Yes** | Halving dt from 2×10⁻⁴ to 1×10⁻⁴ moves range by **0.021 m in 15 841 m** (1.3 ppm) and drift by 0.07 m in 317 m. |
| 6 | Aero coefficient provenance documented — real source or clearly marked placeholder | **Yes** | [COEFFICIENTS.md](COEFFICIENTS.md), now recording per coefficient *which source, at which Mach, and why*, including the splice crossover. **No placeholders.** Confidence graded per coefficient and printed on every run. |
| 7 | `dynamics.py` is pure and accepts the control callback | **Yes** | No I/O, no globals, no input mutation; enforced by tests. `FlightModel.control` hook tested. |

## What is NOT done, stated plainly

1. **Drift magnitude is an open residual (~10 %).** Not tuned away. The one
   correction applied (subsonic C_Nα splice) was adopted on the
   centre-of-pressure evidence and reduced the mean from +13.3 % to +10.2 %.
   Two further candidates were tested and **rejected**: the rifling twist
   (1/25 inverts the error rather than closing it, and three sources give 1/20
   for the tube behind the reference data) and a uniform C_Nα rescale
   (contradicted by BRL's own data outside the subsonic band). What remains
   open and what evidence would settle it is in VALIDATION.md rung 4.
2. **The reduced-rate convention for the four rate-dependent coefficients is
   an assumption**, not a verified fact. Worth <0.25 % in range but 7–11 % in
   drift and a factor 2–4 in the angle-of-attack transient. Direct retrieval
   of the ASAT-13 paper, which would settle it, failed: the publisher's host
   is unreachable.
3. **`C_Mpα` disagrees between sources by 36 %** at the only Mach where both
   measured it (BRL −0.567, ASAT −0.364). Rated LOW. An external review
   claimed 2 % agreement here; that rested on a misread digit and is withdrawn.
4. **The coefficient table ends at Mach 2.00** while charge 8 launches at
   M 2.01. Values held flat; the excursion is reported at run time.
5. **Nonlinear (large-yaw) aerodynamics are not modelled.** The source's
   C_Mpα columns at 2°, 5° and 10° yaw exist and are unused. Valid only
   because nominal flight stays below 0.8° total yaw.
6. **Peak angle of attack is a maximum over logged samples**, not over every
   integration step, because evaluating the aerodynamic state every step would
   roughly double run time. Fine at the default logging cadence; a coarse
   `log_every` under-reports it. Documented on the field itself.
7. **Out of scope for step 1 by instruction, and correctly absent:** canard
   forces, guidance laws, navigation filters, the despun-nose degree of
   freedom, and the Monte Carlo. The seams for all of them are in place
   (control callback, `pack`/`unpack`/`STATE_SIZE`, truth logging).
