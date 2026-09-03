# models/ — reduced-order models (step 2)

Empty at step 1. This is where the **modified point-mass model (MPMM)** goes:
the reduced-order model that would actually run on the embedded processor, in
the form of STANAG 4355, validated against the full 6-DOF in `sim/`.

Validate it against `sim/`, not against the firing table directly. The 6-DOF
is the ground truth for everything downstream; see `docs/VALIDATION.md` for
what that ground truth is currently worth.
