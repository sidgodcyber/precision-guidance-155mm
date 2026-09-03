"""
Cross-check of the ASAT-13 (SPINNER-98) coefficient deck against BRL MR-1582
free-flight measurements, including the centre-of-pressure consistency test.

The CP test is the important one: BRL plots the centre of pressure of the
normal force independently in its Figure 9, so CP constrains the RATIO
C_Malpha/C_Nalpha using a third measured quantity that no part of this model
was fitted to.

    CP_from_nose [cal] = x_cg/d - C_Malpha / C_Nalpha

Run:  python -m analysis.coefficient_crosscheck
"""

from __future__ import annotations

import statistics as st

from analysis.brl_reference import (
    CG_CALIBERS,
    M101_TABLE_II_PAIRS,
    M107_TABLE_III,
    brl_cnalpha_pairs,
    cp_from_nose,
)
from sim import aerodata

#: BRL Figure 9, centre of pressure of the normal force, calibers from nose.
#: Read off the plotted curve; approximate, +-0.1 cal. Recorded here only as
#: a shape reference -- the tabulated rows above are the quantitative source.
BRL_FIG9_APPROX = [
    (0.6, 0.75), (0.8, 0.7), (0.85, 0.6), (1.0, 1.35),
    (1.2, 1.6), (1.6, 1.65), (2.0, 1.8), (2.4, 1.9),
]


def report():
    table = aerodata.make_m107_table()

    print("=" * 78)
    print("1. C_Nalpha and C_Malpha: ASAT (computed) vs BRL (measured)")
    print("=" * 78)
    print(" Mach  src   BRL_CNa  ASAT_CNa  ratio   BRL_CMa  ASAT_CMa  ratio")
    for mach, cna, cma, src in brl_cnalpha_pairs():
        c = table.coefficients_at(mach)
        print(
            f"{mach:6.3f} {src}  {cna:7.2f} {c.C_Nalpha:9.3f} {c.C_Nalpha/cna:6.3f}  "
            f"{cma:7.2f} {c.C_Malpha:9.3f} {c.C_Malpha/cma:6.3f}"
        )

    m107_mach = st.mean(r[0] for r in M107_TABLE_III)
    m107_cna = st.mean(r[4] for r in M107_TABLE_III)
    m107_cma = st.mean(r[3] for r in M107_TABLE_III)
    c = table.coefficients_at(m107_mach)
    print()
    print(f"  M107 3-round mean at M={m107_mach:.3f}:")
    print(f"    C_Nalpha  BRL {m107_cna:.3f}   ASAT {c.C_Nalpha:.3f}   "
          f"ASAT/BRL {c.C_Nalpha/m107_cna:.4f}")
    print(f"    C_Malpha  BRL {m107_cma:.3f}   ASAT {c.C_Malpha:.3f}   "
          f"ASAT/BRL {c.C_Malpha/m107_cma:.4f}")
    # standard error of the 3-round mean, using BRL's stated per-round error
    print(f"    BRL stated per-round error on C_Nalpha is 0.08 -> "
          f"s.e. of mean = {0.08/3**0.5:.3f} ({100*0.08/3**0.5/m107_cna:.1f}%)")

    print()
    print("=" * 78)
    print("2. CENTRE OF PRESSURE TEST  (CP = 2.96 - C_Malpha/C_Nalpha, calibers)")
    print("=" * 78)
    print(" Mach  src    CP_BRL   CP_ASAT   residual (ASAT - BRL)")
    resid_sub, resid_suptr = [], []
    for mach, cna, cma, src in brl_cnalpha_pairs():
        c = table.coefficients_at(mach)
        cp_b = cp_from_nose(cma, cna)
        cp_a = cp_from_nose(c.C_Malpha, c.C_Nalpha)
        r = cp_a - cp_b
        tag = ""
        if mach < 0.9:
            resid_sub.append(r)
            tag = "  <- subsonic"
        else:
            resid_suptr.append(r)
        print(f"{mach:6.3f} {src}  {cp_b:7.3f}  {cp_a:8.3f}   {r:+8.3f}{tag}")
    print()
    if resid_sub:
        print(f"  mean CP residual, M < 0.9 : {st.mean(resid_sub):+.3f} cal "
              f"(n={len(resid_sub)})")
    if resid_suptr:
        print(f"  mean CP residual, M >= 0.9: {st.mean(resid_suptr):+.3f} cal "
              f"(n={len(resid_suptr)})")

    print()
    print("  ASAT CP against BRL Figure 9 (approximate curve):")
    print("   Mach   Fig9_CP   ASAT_CP   residual")
    for mach, cp9 in BRL_FIG9_APPROX:
        c = table.coefficients_at(mach)
        cp_a = cp_from_nose(c.C_Malpha, c.C_Nalpha)
        print(f"  {mach:5.2f}  {cp9:7.2f}  {cp_a:8.3f}  {cp_a-cp9:+8.3f}")

    print()
    print("=" * 78)
    print("3. Magnus moment at M ~ 0.79, zero yaw")
    print("=" * 78)
    c = table.coefficients_at(m107_mach)
    m107_cmpa = st.mean(r[6] for r in M107_TABLE_III)
    print(f"  BRL M107 measured (3 rounds): "
          f"{[r[6] for r in M107_TABLE_III]}, mean {m107_cmpa:.3f}")
    print(f"  ASAT 0-deg column at M={m107_mach:.3f}: {c.C_Mpalpha:.3f}")
    print(f"  ratio ASAT/BRL = {c.C_Mpalpha/m107_cmpa:.3f}  "
          f"-> they differ by {100*abs(c.C_Mpalpha/m107_cmpa-1):.0f}%")
    print("  (The review claimed 2% agreement using -0.36; the scan reads -0.56.)")


if __name__ == "__main__":
    report()
