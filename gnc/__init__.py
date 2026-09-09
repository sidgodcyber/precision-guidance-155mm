"""
gnc -- guidance, navigation and control.

Step 4 populates `roll_control`: the closed-loop roll-angle servo for the
despun nose. Step 3 adds `inverse_map`, `scheduler` and `guidance` beside it.
Step 5 adds `sensors` and `navigation`.

Nothing here changes the flight dynamics. The servo reaches the 6-DOF through
two seams that already existed or that cost one argument:
`sim.canards.NoseAssembly.brake_command`, still a function of time alone, and
`sim.integrate.integrate(step_hook=...)`, which lets a sampled controller
update at step boundaries instead of inside the derivative. Guidance reaches
it through the SAME two and adds no third: `GuidanceLaw.command` is the
`brake_command`-shaped callable `BrakeLaw` already consumes, and
`GuidanceLaw.sample` runs inside the step hook the servo already uses.

Navigation adds no fourth. `NavigationSystem.sample` is a step hook like the
other two and runs first in the same chain; the servo and the guidance law
then read its ESTIMATE instead of the true state through one optional
argument each. A run that differs from another only in that argument measures
the navigation contribution and nothing else.
"""

from . import roll_control
from . import inverse_map
from . import scheduler
from . import guidance
from . import sensors
from . import navigation

__all__ = ["roll_control", "inverse_map", "scheduler", "guidance",
           "sensors", "navigation"]
