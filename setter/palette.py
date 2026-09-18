"""
The app's one colour vocabulary (Control Room spec Part A2 -- ISA-101
colour discipline): a low-saturation neutral base, with colour reserved
for states that need attention and each colour carrying exactly one
meaning.

  ACCENT   the single most important number on a given screen. One meaning
           only -- don't reach for it decoratively.
  AMBER    a value outside its expected band, or an unresolved condition
           (the mid2 non-monotonicity, an unavailable campaign cell, a
           stale artifact).
  RED      a validation failure or an error.

`st.error`/`st.warning`/`st.success` already carry this discipline for
plain Streamlit text; this module exists for matplotlib figures and any
raw HTML, which don't inherit it automatically -- so anything reaching for
a colour outside this module is very likely violating Part A2.

Everything else -- BG/GRID/INK/TEXT -- is the neutral base: dark
background, grid lines, secondary ink, primary text.
"""

ACCENT = "#38bdf8"
AMBER = "#f0b429"
RED = "#ef4444"

BG = "#111827"
GRID = "#374151"
INK = "#9aa5b1"
TEXT = "#e5e7eb"

__all__ = ["ACCENT", "AMBER", "RED", "BG", "GRID", "INK", "TEXT"]
