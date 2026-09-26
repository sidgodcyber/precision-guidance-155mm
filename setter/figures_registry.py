"""
The figure gallery's declarative manifest: all 38 files in `docs/figures/`,
each with its caption, generating script, and source data -- the figure
half of Part B4/L4 (`setter/registry.py` is the script half; this module
follows the exact same "derive, don't invent" discipline).

HOW THIS WAS BUILT
------------------
Every caption below is the generating function's own docstring, copied
verbatim (or its module docstring's per-figure line, for the few scripts
that document all their figures together) -- not written fresh for this
gallery. Every `module`/`reads` pairing was confirmed by grep against the
actual `savefig(...)` call and its surrounding code, not assumed from the
filename. Two things this turned up that a filename-only guess would have
missed:

  * 8 of the 38 (`trajectory_c*`, `ground_track_c*`, `angle_of_attack_c*`,
    `diagnostics_c*`) come from `run_ballistic.py`, a ROOT-level script
    from Step 1 -- not from anything in `analysis/`, and not covered by
    `setter/registry.py` (which is scoped to `analysis/*.py`). They read
    no stored file at all: each is a live 6-DOF flight from `sim.dynamics`
    at the command line's own charge/QE, computed fresh every run.
  * `design_sweep.png` is ORPHANED: it was added in the same commit as
    `analysis/design_sweep.py` (confirmed via `git log`), but that script
    currently contains no matplotlib code at all -- the plotting logic
    was removed at some point after. No script in the current tree
    produces this file. Marked `orphaned=True` rather than guessing a
    generator; see `setter/pages/archive.py` for how this is surfaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

__all__ = ["FigureEntry", "ENTRIES", "by_id"]


@dataclass(frozen=True)
class FigureEntry:
    id: str
    filename: str  # relative to docs/figures/
    caption: str  # the generating function's own docstring, verbatim
    subsystem: str
    module: Optional[str] = None  # "analysis.x" importable module, or None
    generator_note: str = ""  # for non-module generators (root scripts, orphans)
    reads: Tuple[str, ...] = ()
    orphaned: bool = False
    writes: Tuple[str, ...] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "writes", (f"docs/figures/{self.filename}",))


_BALLISTIC_ARGS = {
    "c4_qe97": "--charge 4 --qe-mils 97",
    "c8_qe525": "--charge 8 --qe-mils 525",
}


def _ballistic(tag: str, kind: str, caption: str) -> FigureEntry:
    return FigureEntry(
        id=f"{kind}_{tag}", filename=f"{kind}_{tag}.png", caption=caption,
        subsystem="ballistics_validation", module=None,
        generator_note=(
            f"python run_ballistic.py {_BALLISTIC_ARGS.get(tag, '')} "
            f"(root-level script, Step 1; not in setter/registry.py; reads "
            f"no stored file -- flies live from sim.dynamics each run)"),
        reads=())


ENTRIES: Tuple[FigureEntry, ...] = (
    # -----------------------------------------------------------------
    # ballistics_validation -- run_ballistic.py, root-level, Step 1
    # -----------------------------------------------------------------
    _ballistic("c4_qe97", "trajectory", "Trajectory profile: downrange vs altitude, impact/TOF/apogee annotated."),
    _ballistic("c8_qe525", "trajectory", "Trajectory profile: downrange vs altitude, impact/TOF/apogee annotated."),
    _ballistic("c4_qe97", "ground_track", "Ground track: downrange vs crossrange, drift direction annotated."),
    _ballistic("c8_qe525", "ground_track", "Ground track: downrange vs crossrange, drift direction annotated."),
    _ballistic("c4_qe97", "angle_of_attack", "Total angle of attack, and its alpha/beta components, over the flight."),
    _ballistic("c8_qe525", "angle_of_attack", "Total angle of attack, and its alpha/beta components, over the flight."),
    _ballistic("c4_qe97", "diagnostics", "Gyroscopic stability factor, Mach, axial spin and yaw of repose, four panels."),
    _ballistic("c8_qe525", "diagnostics", "Gyroscopic stability factor, Mach, axial spin and yaw of repose, four panels."),

    # -----------------------------------------------------------------
    # authority
    # -----------------------------------------------------------------
    FigureEntry(
        id="authority_envelope", filename="authority_envelope.png",
        caption="The envelope in the range/deflection plane, for step 3 to look at.",
        subsystem="authority", module="analysis.authority_report",
        reads=("docs/authority_results.json",)),
    FigureEntry(
        id="design_sweep", filename="design_sweep.png",
        caption="(orphaned -- see this module's docstring)",
        subsystem="authority", module=None, orphaned=True,
        generator_note="No script in the current tree contains the plotting "
                       "code; analysis/design_sweep.py no longer has any "
                       "matplotlib usage, though it wrote this file when "
                       "first added (git log confirms the same commit).",
        reads=()),

    # -----------------------------------------------------------------
    # guidance
    # -----------------------------------------------------------------
    FigureEntry(
        id="guidance_inverse_map", filename="guidance_inverse_map.png",
        caption="The map itself: the rotation between commanded angle and delivered "
               "direction, how it moves over the flight, and how the reach varies "
               "with the direction asked for.",
        subsystem="guidance", module="analysis.guidance_figures",
        reads=("docs/guidance_map.json",)),
    FigureEntry(
        id="guidance_prediction", filename="guidance_prediction.png",
        caption="Impact-point-prediction error against time to go, which is what "
               "sets when the direction can be committed.",
        subsystem="guidance", module="analysis.guidance_figures",
        reads=("docs/guidance_map.json",)),
    FigureEntry(
        id="guidance_schedulers", filename="guidance_schedulers.png",
        caption="The four laws and the isotropic control, as miss distributions "
               "rather than as four CEP numbers.",
        subsystem="guidance", module="analysis.guidance_figures",
        reads=("docs/guidance_cep.json",)),
    FigureEntry(
        id="guidance_aimoff", filename="guidance_aimoff.png",
        caption="CEP against aim-off, with the published 224 m marked.",
        subsystem="guidance", module="analysis.guidance_figures",
        reads=("docs/guidance_cep.json",)),
    FigureEntry(
        id="guidance_ladder", filename="guidance_ladder.png",
        caption="The degradation ladder: CEP against navigation-loss fraction "
               "(Task E rungs).",
        subsystem="guidance", module="analysis.guidance_figures",
        reads=("docs/guidance_cep.json",)),
    FigureEntry(
        id="guidance_envelope", filename="guidance_envelope.png",
        caption="CEP across the firing table against authority and against the "
               "dead time as a fraction of the guided phase -- the short-range finding.",
        subsystem="guidance", module="analysis.guidance_figures",
        reads=("docs/guidance_map.json", "docs/guidance_cep.json")),

    # -----------------------------------------------------------------
    # roll_servo
    # -----------------------------------------------------------------
    FigureEntry(
        id="roll_servo_envelope", filename="roll_servo_envelope.png",
        caption="The actuator is one-sided, and the asymmetry reverses during the flight.",
        subsystem="roll_servo", module="analysis.roll_servo_figures",
        reads=("docs/roll_servo.json",)),
    FigureEntry(
        id="roll_servo_response", filename="roll_servo_response.png",
        caption="What the closed loop achieves, against what the actuator could "
               "do if the loop asked for it.",
        subsystem="roll_servo", module="analysis.roll_servo_figures",
        reads=("docs/roll_servo.json",)),
    FigureEntry(
        id="roll_servo_duty", filename="roll_servo_duty.png",
        caption="Why the duty cycle does not work at short period.",
        subsystem="roll_servo", module="analysis.roll_servo_figures",
        reads=("docs/roll_servo.json",)),
    FigureEntry(
        id="staged_distributions", filename="staged_distributions.png",
        caption="The two configurations as DISTRIBUTIONS over 24 deployment "
               "phases, not as two means.",
        subsystem="roll_servo", module="analysis.staged_deployment_figures",
        reads=("docs/staged_deployment.json",)),
    FigureEntry(
        id="staged_delay", filename="staged_delay.png",
        caption="The trade: cleaner capture bought with a shorter correction "
               "window, and where the optimum is.",
        subsystem="roll_servo", module="analysis.staged_deployment_figures",
        reads=("docs/staged_deployment.json",)),
    FigureEntry(
        id="staged_despin", filename="staged_despin.png",
        caption="Why staging costs time at all -- the two-panel despin, at twice "
               "the time constant to twice the equilibrium rate.",
        subsystem="roll_servo", module="analysis.staged_deployment_figures",
        reads=("docs/staged_deployment.json",)),

    # -----------------------------------------------------------------
    # navigation
    # -----------------------------------------------------------------
    FigureEntry(
        id="nav_saturation", filename="nav_saturation.png",
        caption="The measurement that decides the layout: what a commodity gyro "
               "sees on the body and in the despun section.",
        subsystem="navigation", module="analysis.nav_figures",
        reads=("docs/nav_sensors.json",)),
    FigureEntry(
        id="nav_observability", filename="nav_observability.png",
        caption="Roll observability round the compass, and what it costs.",
        subsystem="navigation", module="analysis.nav_figures",
        reads=("docs/nav_sensors.json",)),
    FigureEntry(
        id="nav_nees", filename="nav_nees.png",
        caption="NEES against its chi-square bounds, over the campaign and per engagement.",
        subsystem="navigation", module="analysis.nav_figures",
        reads=("docs/nav_consistency.json",)),
    FigureEntry(
        id="nav_nis", filename="nav_nis.png",
        caption="NIS per stream, and the whiteness that NIS cannot see.",
        subsystem="navigation", module="analysis.nav_figures",
        reads=("docs/nav_consistency.json",)),
    FigureEntry(
        id="nav_cep", filename="nav_cep.png",
        caption="The navigation contribution, per engagement and per axis.",
        subsystem="navigation", module="analysis.nav_figures",
        reads=("docs/nav_cep.json",)),
    FigureEntry(
        id="nav_degradation", filename="nav_degradation.png",
        caption="The Task G ladder.",
        subsystem="navigation", module="analysis.nav_figures",
        reads=("docs/nav_cep.json",)),
    FigureEntry(
        id="nav_warm_start", filename="nav_warm_start.png",
        caption="Task D: does the warm start move the solution inside the deadline?",
        subsystem="navigation", module="analysis.nav_figures",
        reads=("docs/nav_cep.json",)),

    # -----------------------------------------------------------------
    # monte_carlo
    # -----------------------------------------------------------------
    FigureEntry(
        id="mc_scatter", filename="mc_scatter.png",
        caption="Impact points, unguided and guided, with the CEP and the "
               "requirement. The submission's own artefact figure.",
        subsystem="monte_carlo", module="analysis.monte_carlo_figures",
        reads=("docs/monte_carlo.json",)),
    FigureEntry(
        id="mc_scatter_fresh_met", filename="mc_scatter_fresh_met.png",
        caption="Impact points, unguided and guided, with the CEP and the "
               "requirement (tag: fresh_met -- met uploaded at fuze setting).",
        subsystem="monte_carlo", module="analysis.monte_carlo_figures",
        reads=("docs/monte_carlo.json",)),
    FigureEntry(
        id="mc_scatter_physical", filename="mc_scatter_physical.png",
        caption="Impact points, unguided and guided, with the CEP and the "
               "requirement (tag: physical -- no dispersion top-up).",
        subsystem="monte_carlo", module="analysis.monte_carlo_figures",
        reads=("docs/monte_carlo.json",)),
    FigureEntry(
        id="mc_convergence", filename="mc_convergence.png",
        caption="CEP against sample count, with the bootstrap interval.",
        subsystem="monte_carlo", module="analysis.monte_carlo_figures",
        reads=("docs/monte_carlo.json",)),
    FigureEntry(
        id="mc_budget", filename="mc_budget.png",
        caption="The uncorrected dispersion, decomposed into the causes that make it.",
        subsystem="monte_carlo", module="analysis.monte_carlo_figures",
        reads=("docs/monte_carlo.json",)),
    FigureEntry(
        id="mc_atmosphere", filename="mc_atmosphere.png",
        caption="The atmospheric knowledge term against met message age.",
        subsystem="monte_carlo", module="analysis.monte_carlo_figures",
        reads=("docs/monte_carlo.json",)),
    FigureEntry(
        id="mc_monitor", filename="mc_monitor.png",
        caption="The false-alarm / missed-detection trade, as a curve rather "
               "than a point.",
        subsystem="monte_carlo", module="analysis.monte_carlo_figures",
        reads=("docs/monte_carlo.json",)),
    FigureEntry(
        id="mc_staging", filename="mc_staging.png",
        caption="Staged against single stage over the distributions, not the means.",
        subsystem="monte_carlo", module="analysis.monte_carlo_figures",
        reads=("docs/monte_carlo.json",)),
    FigureEntry(
        id="mc_band", filename="mc_band.png",
        caption="The epistemic band: the CEP at each edge, paired against the nominal.",
        subsystem="monte_carlo", module="analysis.monte_carlo_figures",
        reads=("docs/monte_carlo.json",)),
)

_BY_ID = {e.id: e for e in ENTRIES}


def by_id(entry_id: str) -> Optional[FigureEntry]:
    return _BY_ID.get(entry_id)
