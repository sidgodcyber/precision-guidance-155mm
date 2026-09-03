# embedded/ — C implementation (step 7)

Empty at step 1. Target: a C implementation of the step-2 reduced-order model
and the step-3 to step-5 guidance, navigation and control, suitable for the
fuze-cavity processor.

The Python in `sim/` is deliberately written so the derivative function is a
pure state-in/derivative-out routine with no I/O, no globals and no hidden
state, which is the shape that ports cleanly to C.
