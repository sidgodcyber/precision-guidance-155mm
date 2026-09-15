"""
The script registry: a declarative manifest of every `analysis/*.py` module
with a command-line entry point (Control Room spec Part B4).

This is what makes the repository's 34,000 lines legible from the
dashboard: every figure and table in the Archive page can say which script
produced it, what it read, and whether that input has changed since.

HOW THIS WAS BUILT
------------------
Each entry below was derived from that script's own module docstring and
`argparse` setup -- not invented. Every `reads`/`writes` path was confirmed
by grep against the actual `open(...)`/`json.dump(...)`/`savefig(...)`
calls in the script (not just the docstring prose, which in two cases
turned out to be misleading -- see `authority_report` and `cep_projection`
below). Where a script's true cost could not be pinned down from source
(most of the `campaign` class), the docstring's own stated wall-clock
figure is used, or "minutes to hours" where none is given but the script
is known to fly many 6-DOF rounds.

Nothing here recomputes anything: this module is data.

SAFETY (see also `setter/runner.py`)
-------------------------------------
`runnable=True` is necessary but not sufficient for the UI to offer a run
button -- `setter.runner.can_run` also checks `cost_class != "campaign"`.
A `runnable=False` entry (whatever its cost_class) is NEVER runnable from
the UI, full stop; `notes` says why when the reason isn't just "slow"
(`migrate_c_tag` is fast but mutates `docs/monte_carlo.json` in place, the
one file this app is never allowed to write).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

__all__ = ["ScriptEntry", "ENTRIES", "by_id", "by_subsystem", "SUBSYSTEMS"]


@dataclass(frozen=True)
class ScriptEntry:
    id: str
    title: str
    module: str  # importable as `python -m {module}`
    description: str  # the script's own docstring, first paragraph, verbatim
    subsystem: str
    cost_class: str  # "reader" | "figure" | "analysis" | "campaign"
    cost: str  # free text, from the docstring/argparse where stated
    runnable: bool
    reads: Tuple[str, ...] = ()
    writes: Tuple[str, ...] = ()
    notes: str = ""


#: Subsystem grouping, in the order the Archive page lists them -- roughly
#: the project's own step order (1/2 ballistics -> 2.5 authority -> 3
#: guidance -> 4 roll servo -> 5 navigation -> 6 monte carlo).
SUBSYSTEMS = (
    "ballistics_validation", "authority", "guidance", "roll_servo",
    "navigation", "monte_carlo",
)

ENTRIES: Tuple[ScriptEntry, ...] = (
    # -----------------------------------------------------------------
    # ballistics_validation
    # -----------------------------------------------------------------
    ScriptEntry(
        id="coefficient_crosscheck", title="Coefficient deck cross-check",
        module="analysis.coefficient_crosscheck",
        description=("Cross-check of the ASAT-13 (SPINNER-98) coefficient deck against "
                      "BRL MR-1582 free-flight measurements, including the "
                      "centre-of-pressure consistency test."),
        subsystem="ballistics_validation", cost_class="analysis", cost="seconds",
        runnable=True, reads=(), writes=(),
        notes="Compares against the published BRL tables carried in-module "
              "(analysis.brl_reference); writes nothing.",
    ),
    ScriptEntry(
        id="rate_convention_audit", title="Rate-convention audit",
        module="analysis.rate_convention_audit",
        description="Convention audit for the rate-dependent coefficients.",
        subsystem="ballistics_validation", cost_class="analysis",
        cost="seconds to a minute", runnable=True,
        reads=(), writes=("docs/rate_convention_audit.json",),
    ),
    ScriptEntry(
        id="verify_swerve", title="Swerve-response verification",
        module="analysis.verify_swerve",
        description=("PHASE 2 -- verification of the 6-DOF against a PUBLISHED "
                      "configuration whose control authority is also published."),
        subsystem="ballistics_validation", cost_class="analysis",
        cost="seconds to a minute", runnable=True,
        reads=(), writes=("docs/swerve_verification.json",),
        notes="Ollerenshaw & Costello (2008) / ARL-CR-0604.",
    ),

    # -----------------------------------------------------------------
    # authority
    # -----------------------------------------------------------------
    ScriptEntry(
        id="authority", title="Correction-authority envelope",
        module="analysis.authority",
        description="Open-loop correction-authority envelope for the fuze-well canard kit.",
        subsystem="authority", cost_class="campaign",
        cost="~20 min full sweep on 8 cores; --quick: one engagement, coarse",
        runnable=False, reads=(), writes=("docs/authority_results.json",),
    ),
    ScriptEntry(
        id="authority_report", title="Authority envelope tables",
        module="analysis.authority_report",
        description="Turn docs/authority_results.json into the tables in docs/AUTHORITY-ENVELOPE.md.",
        subsystem="authority", cost_class="reader", cost="seconds", runnable=True,
        reads=("docs/authority_results.json",),
        writes=("docs/figures/authority_envelope.png",),
        notes="Prints its markdown tables to stdout rather than writing a "
              "file (confirmed by source -- the docstring's own prose is "
              "misleading here); only the figure is an actual file write, "
              "and only unless --no-figure.",
    ),
    ScriptEntry(
        id="deployed_stability", title="Deployed-configuration stability",
        module="analysis.deployed_stability",
        description="Gyroscopic and dynamic stability WITH THE CANARDS DEPLOYED.",
        subsystem="authority", cost_class="analysis",
        cost="seconds to a minute", runnable=True,
        reads=(), writes=("docs/deployed_stability.json",),
    ),
    ScriptEntry(
        id="design_sweep", title="Canard design sweep",
        module="analysis.design_sweep",
        description=("PHASE 3 -- the design sweep, and the aerodynamic sensitivity "
                      "that Phase 0 showed has to go with it."),
        subsystem="authority", cost_class="campaign",
        cost="heavy, unspecified duration on 8 cores (per-engagement, per-aero-variant "
             "baselines); --quick: one engagement, station only",
        runnable=False, reads=(), writes=("docs/design_sweep.json",),
        notes="--best/--combos variants (docs/design_sweep_best.json, "
              "docs/design_sweep_combos.json in the repo) need an explicit "
              "--out override not exposed here.",
    ),
    ScriptEntry(
        id="cep_projection", title="Authority-to-CEP projection",
        module="analysis.cep_projection",
        description="PHASE 4 -- what a measured correction authority implies for achievable CEP.",
        subsystem="authority", cost_class="analysis",
        cost="seconds (closed-form quadrature, no simulation)", runnable=True,
        reads=("docs/design_sweep.json", "docs/design_sweep_combos.json"),
        writes=("docs/cep_projection.json",),
        notes="docs/design_sweep_combos.json is a --combos variant of "
              "design_sweep, not that entry's default output.",
    ),
    ScriptEntry(
        id="deploy_mach", title="Deployment-point Mach sweep",
        module="analysis._deploy_mach",
        description="Mach number at the deployment point for every sweep case. Small helper.",
        subsystem="authority", cost_class="campaign",
        cost="a few minutes (flies one uncorrected 6-DOF baseline per firing-table "
             "case via analysis.authority, despite the docstring calling itself small)",
        runnable=False, reads=(), writes=("docs/deploy_mach.json",),
    ),

    # -----------------------------------------------------------------
    # guidance
    # -----------------------------------------------------------------
    ScriptEntry(
        id="guidance_authority", title="Inverse-map calibration",
        module="analysis.guidance_authority",
        description="Calibrate the analytic inverse map, and measure what the onboard predictor costs.",
        subsystem="guidance", cost_class="campaign",
        cost="heavy -- many ScheduledHold flights per engagement", runnable=False,
        reads=(), writes=("docs/guidance_map.json",),
        notes="Produces the map most other guidance/navigation/monte_carlo scripts read.",
    ),
    ScriptEntry(
        id="guidance_cep", title="Closed-loop guidance CEP",
        module="analysis.guidance_cep",
        description="Closed-loop CEP: schedulers, aim-off, the degradation ladder, and the firing table.",
        subsystem="guidance", cost_class="campaign",
        cost="minutes to hours (Tasks c-g, many guided 6-DOF rounds)", runnable=False,
        reads=("docs/guidance_map.json",), writes=("docs/guidance_cep.json",),
    ),
    ScriptEntry(
        id="guidance_figures", title="Guidance figures",
        module="analysis.guidance_figures",
        description=("Figures for step 3. Reads docs/guidance_map.json and "
                      "docs/guidance_cep.json, writes docs/figures/guidance_*.png."),
        subsystem="guidance", cost_class="figure", cost="seconds", runnable=True,
        reads=("docs/guidance_map.json", "docs/guidance_cep.json"),
        writes=("docs/figures/guidance_*.png",),
    ),
    ScriptEntry(
        id="guidance_report", title="Guidance tables",
        module="analysis.guidance_report",
        description="Tables for the step-3 documents, from docs/guidance_map.json and docs/guidance_cep.json.",
        subsystem="guidance", cost_class="reader", cost="seconds", runnable=True,
        reads=("docs/guidance_map.json", "docs/guidance_cep.json"),
        writes=("docs/guidance_tables.md",),
    ),
    ScriptEntry(
        id="mpmm_compare", title="MPMM vs 6-DOF comparison",
        module="analysis.mpmm_compare",
        description="MPMM versus 6-DOF: validation, model error, and the drift-hypothesis test.",
        subsystem="guidance", cost_class="analysis",
        cost="seconds to a minute", runnable=True,
        reads=(), writes=("docs/mpmm_results.json",),
    ),
    ScriptEntry(
        id="mpmm_compute", title="MPMM propagation cost",
        module="analysis.mpmm_compute",
        description="Task E: compute cost of one MPMM propagation to impact.",
        subsystem="guidance", cost_class="analysis",
        cost="seconds to a minute", runnable=True,
        reads=(), writes=("docs/mpmm_compute.json",),
    ),
    ScriptEntry(
        id="migrate_c_tag", title="Task C tag migration (one-off)",
        module="analysis.migrate_c_tag",
        description="One-off migration: move Task C under a tag level, as Task A already is.",
        subsystem="guidance", cost_class="analysis", cost="seconds",
        runnable=False,
        reads=("docs/monte_carlo.json",), writes=("docs/monte_carlo.json",),
        notes="Never runnable from the UI regardless of cost: it mutates "
              "docs/monte_carlo.json IN PLACE (with a backup unless "
              "--no-backup). This app never writes campaign data files.",
    ),

    # -----------------------------------------------------------------
    # roll_servo
    # -----------------------------------------------------------------
    ScriptEntry(
        id="roll_servo", title="Roll servo characterisation",
        module="analysis.roll_servo",
        description="Characterisation of the closed-loop roll-angle servo. Tasks A, C and D.",
        subsystem="roll_servo", cost_class="campaign",
        cost="~30 min full on 8 cores; --quick: Tasks A and C only, no 6-DOF",
        runnable=False, reads=(), writes=("docs/roll_servo.json",),
    ),
    ScriptEntry(
        id="roll_robustness", title="Roll-servo robustness sweep",
        module="analysis.roll_robustness",
        description=("Task E: does the servo still meet its numbers when the plant "
                      "is not what the controller was told it was?"),
        subsystem="roll_servo", cost_class="campaign",
        cost="~3 min, single core", runnable=False,
        reads=("docs/roll_servo.json",), writes=("docs/roll_robustness.json",),
    ),
    ScriptEntry(
        id="roll_servo_figures", title="Roll servo figures",
        module="analysis.roll_servo_figures",
        description="Figures for the roll servo. Reads docs/roll_servo.json, writes docs/figures/roll_servo_*.png.",
        subsystem="roll_servo", cost_class="figure", cost="seconds", runnable=True,
        reads=("docs/roll_servo.json",), writes=("docs/figures/roll_servo_*.png",),
    ),
    ScriptEntry(
        id="roll_servo_report", title="Roll servo tables",
        module="analysis.roll_servo_report",
        description=("Markdown tables for docs/CONTROL-CHARACTERISATION.md and "
                      "docs/CONTROL-ROBUSTNESS.md, generated from docs/roll_servo.json "
                      "and docs/roll_robustness.json."),
        subsystem="roll_servo", cost_class="reader", cost="seconds", runnable=True,
        reads=("docs/roll_servo.json", "docs/roll_robustness.json"),
        writes=("docs/roll_servo_tables.md",),
    ),
    ScriptEntry(
        id="staged_deployment", title="Staged canard deployment",
        module="analysis.staged_deployment",
        description="Staged canard deployment: does releasing the cant pair first buy anything?",
        subsystem="roll_servo", cost_class="campaign",
        cost="full ~50 min on 8 cores; --tasks b: algebra only, seconds",
        runnable=False, reads=(), writes=("docs/staged_deployment.json",),
    ),
    ScriptEntry(
        id="staged_deployment_figures", title="Staged deployment figures",
        module="analysis.staged_deployment_figures",
        description=("Figures for the staged deployment. Reads docs/staged_deployment.json, "
                      "writes docs/figures/staged_*.png."),
        subsystem="roll_servo", cost_class="figure", cost="seconds", runnable=True,
        reads=("docs/staged_deployment.json",), writes=("docs/figures/staged_*.png",),
    ),
    ScriptEntry(
        id="staged_deployment_report", title="Staged deployment tables",
        module="analysis.staged_deployment_report",
        description="Markdown tables for docs/STAGED-DEPLOYMENT.md, generated from docs/staged_deployment.json.",
        subsystem="roll_servo", cost_class="reader", cost="seconds", runnable=True,
        reads=("docs/staged_deployment.json",),
        writes=("docs/staged_deployment_tables.md",),
    ),

    # -----------------------------------------------------------------
    # navigation
    # -----------------------------------------------------------------
    ScriptEntry(
        id="nav_sensors", title="Sensor exposure characterisation",
        module="analysis.nav_sensors",
        description="Step 5, Task A: what the sensors are exposed to, and what that decides.",
        subsystem="navigation", cost_class="analysis",
        cost="seconds to a minute (one flown trajectory)", runnable=True,
        reads=(), writes=("docs/nav_sensors.json",),
    ),
    ScriptEntry(
        id="nav_consistency", title="Filter consistency (NEES/NIS)",
        module="analysis.nav_consistency",
        description="Step 5, Task E: is the filter CONSISTENT, not merely accurate?",
        subsystem="navigation", cost_class="campaign",
        cost="minutes (a navigation campaign; --tune sweeps three inflations)",
        runnable=False, reads=(), writes=("docs/nav_consistency.json",),
    ),
    ScriptEntry(
        id="nav_cep", title="Navigation contribution to CEP",
        module="analysis.nav_cep",
        description=("Step 5, Tasks D, F and G: the warm start, the navigation "
                      "contribution to CEP, and what degradation costs."),
        subsystem="navigation", cost_class="campaign",
        cost="minutes to hours (paired navigation-in-loop rounds over N_SEEDS "
             "sensor realisations)",
        runnable=False, reads=(), writes=("docs/nav_cep.json",),
    ),
    ScriptEntry(
        id="nav_ablation", title="Navigation error ablation",
        module="analysis.nav_ablation",
        description="Step 5.5: WHERE the 30.4 m navigation contribution lives.",
        subsystem="navigation", cost_class="campaign",
        cost="minutes to hours (Tasks a-e fly many navigation-in-loop rounds; "
             "--tasks f alone takes a second)",
        runnable=False, reads=("docs/nav_cep.json",), writes=("docs/nav_ablation.json",),
        notes="docs/nav_cep.json is read only by --tasks f (the authority-monitor "
              "false-alarm count from step 5's saved rows).",
    ),
    ScriptEntry(
        id="nav_antenna", title="Antenna-placement sensitivity",
        module="analysis.nav_antenna",
        description=("Step 5, the sensitivity that follows from Task F: what does "
                      "the SPINNING ANTENNA cost the CEP?"),
        subsystem="navigation", cost_class="campaign",
        cost="minutes (24 draws x 2 seeds of navigation-in-loop rounds by default)",
        runnable=False, reads=(), writes=("docs/nav_antenna.json",),
    ),
    ScriptEntry(
        id="nav_drivers", title="Navigation error drivers (ablation)",
        module="analysis.nav_drivers",
        description="Step 5: WHICH error drives the navigation contribution to CEP?",
        subsystem="navigation", cost_class="campaign",
        cost="minutes (4 variants x 24 draws x 2 seeds of navigation-in-loop rounds by default)",
        runnable=False, reads=(), writes=("docs/nav_drivers.json",),
    ),
    ScriptEntry(
        id="nav_figures", title="Navigation figures",
        module="analysis.nav_figures",
        description="Figures for step 5.",
        subsystem="navigation", cost_class="figure", cost="seconds", runnable=True,
        reads=("docs/nav_sensors.json", "docs/nav_consistency.json", "docs/nav_cep.json"),
        writes=("docs/figures/nav_*.png",),
    ),
    ScriptEntry(
        id="nav_report", title="Navigation tables",
        module="analysis.nav_report",
        description="Generated tables for step 5.",
        subsystem="navigation", cost_class="reader", cost="seconds", runnable=True,
        reads=("docs/nav_cep.json", "docs/nav_consistency.json", "docs/nav_sensors.json"),
        writes=("docs/nav_tables.md",),
    ),

    # -----------------------------------------------------------------
    # monte_carlo
    # -----------------------------------------------------------------
    ScriptEntry(
        id="monte_carlo", title="Dispersion campaign (Monte Carlo)",
        module="analysis.monte_carlo",
        description="Step 6 -- the dispersion campaign and the headline CEP.",
        subsystem="monte_carlo", cost_class="campaign",
        cost="9-12 hours for the full published set (see the module docstring's "
             "exact invocation list -- it runs as many separate --tasks calls)",
        runnable=False, reads=("docs/guidance_map.json",), writes=("docs/monte_carlo.json",),
        notes="The one campaign this whole application is built around; see CLAUDE.md.",
    ),
    ScriptEntry(
        id="monte_carlo_figures", title="Monte Carlo figures",
        module="analysis.monte_carlo_figures",
        description="Figures for step 6.",
        subsystem="monte_carlo", cost_class="figure", cost="seconds", runnable=True,
        reads=("docs/monte_carlo.json",), writes=("docs/figures/mc_*.png",),
    ),
    ScriptEntry(
        id="monte_carlo_report", title="Monte Carlo tables",
        module="analysis.monte_carlo_report",
        description="Generated tables for step 6.",
        subsystem="monte_carlo", cost_class="reader", cost="seconds", runnable=True,
        reads=("docs/monte_carlo.json",), writes=("docs/mc_tables.md",),
    ),
    ScriptEntry(
        id="compare_c_tags", title="Task C tag comparison",
        module="analysis.compare_c_tags",
        description="Compare two Task C campaigns that differ only in the dispersion top-up.",
        subsystem="monte_carlo", cost_class="analysis", cost="seconds", runnable=True,
        reads=("docs/monte_carlo.json",), writes=(),
        notes="Read-only per its own docstring: \"Writes nothing.\"",
    ),
)

_BY_ID = {e.id: e for e in ENTRIES}


def by_id(entry_id: str) -> Optional[ScriptEntry]:
    return _BY_ID.get(entry_id)


def by_subsystem() -> dict:
    """`{subsystem: [entries]}`, in `SUBSYSTEMS` order, each group in the
    order it was declared above."""
    out = {s: [] for s in SUBSYSTEMS}
    for e in ENTRIES:
        out.setdefault(e.subsystem, []).append(e)
    return out
