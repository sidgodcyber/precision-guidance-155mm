"""
Simulation setter / mission-planning dashboard -- navigation shell.

Run with:  streamlit run setter/app.py

This is a software-only mission-planning UI over the existing simulation
engine. See `setter/pages/overview.py`, `setter/pages/mission_control.py`
and `setter/pages/archive.py` for the screens themselves;
`setter/campaign.py` and `setter/simulation_adapter.py` for the boundary
against the frozen simulation engine.

Screen layout follows the Control Room spec's ISA-101 display hierarchy
(see CLAUDE.md): Overview is L1, Mission Control is L2, Archive is (the
registry slice of) L4. `st.navigation` was introduced in Step 4 to let a
second screen exist at all; Step 5 added Overview as a third and made it
the DEFAULT page, per Part A1 ("L1 is where you start, L2 is where you
work") -- before Step 5, this shell defaulted to Mission Control directly.

Each page is registered by FILE PATH, not by importing and passing its
`render` function: `AppTest.switch_page()` and `st.page_link()` both need
a file-based page to resolve (confirmed against the installed Streamlit --
see the note atop `setter/pages/mission_control.py`), and a page file runs
unconditionally top to bottom when Streamlit execs it as the active page,
the same way this shell always runs `pg.run()` unconditionally. Nothing
here needs to `import setter.pages.*` at all.

Flight Deck (L3, Step 6) is added to `st.navigation`'s page list ONLY when
`st.session_state["fired_round"]` already exists: "an empty Flight Deck is
not useful" (Control Room spec). This file reruns on every interaction
(Streamlit always re-executes the main script), so the check is live --
firing a round on Mission Control makes Flight Deck appear in the sidebar
on the very next rerun, without a page reload.
"""


from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run setter/app.py` execs this file directly and only puts its
# OWN directory (`setter/`) on sys.path (see
# `streamlit.web.bootstrap._fix_sys_path`) -- never the repository root. That
# leaves `import setter` and `import fuze` unresolvable even though this file
# lives inside the `setter` package, because the package's PARENT directory
# was never added. pytest/AppTest never hit this: pytest puts the repo root
# on sys.path for the whole test process, which masked it there. Put the
# repo root on sys.path before importing anything from `setter`/`fuze`. This
# runs on EVERY rerun (Streamlit always re-executes the main script), before
# `st.navigation` hands off to whichever page file is active, so every page
# gets it too without needing its own copy.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

st.set_page_config(page_title="Simulation Setter", layout="wide")

_pages = [
    st.Page("pages/overview.py", title="Overview", url_path="overview", default=True),
    st.Page("pages/mission_control.py", title="Mission Control", url_path="mission-control"),
    st.Page("pages/archive.py", title="Archive", url_path="archive"),
]
if "fired_round" in st.session_state:
    _pages.append(st.Page("pages/flight_deck.py", title="Flight Deck", url_path="flight-deck"))

pg = st.navigation(_pages)
pg.run()
