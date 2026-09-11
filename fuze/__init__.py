"""
Generic multi-mode simulation event engine.

Despite the directory name, nothing here is a hardware fuze. It is a
software pipeline -- simulation input -> sensor readings -> state machine ->
abstract event -- built to demonstrate state-machine correctness against
trajectories the existing 6-DOF/MPMM engine already produces. There is no
physical-effect system downstream of the terminal event.
"""

from __future__ import annotations
