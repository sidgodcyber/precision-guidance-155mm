"""
Simulation setter / mission-planning dashboard.

Software-only: this package builds a simulation configuration message and a
mission-planning UI around the existing 6-DOF/MPMM/Monte-Carlo simulation
engine. There is no hardware integration, embedded target, or operational
control interface anywhere in it. See `setter/simulation_adapter.py` for the
boundary against the frozen `sim/`, `analysis/`, `models/`, `gnc/` modules.
"""

from __future__ import annotations
