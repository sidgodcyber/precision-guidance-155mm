"""
Simulation setter / mission-planning dashboard -- navigation shell.

Run with:  streamlit run setter/app.py

This is a software-only mission-planning UI over the existing simulation
engine. See `setter/pages/mission_control.py` and `setter/pages/archive.py`
for the screens themselves; `setter/campaign.py` and
`setter/simulation_adapter.py` for the boundary against the frozen
simulation engine.

Screen layout follows the Control Room spec's ISA-101 display hierarchy
(see CLAUDE.md): Mission Control is L2, Archive is (the registry slice of)
L4. `st.navigation` was introduced in Step 4 to let a second screen exist
at all -- before that, this file WAS Mission Control's content directly.
Mission Control stays the default page, so `AppTest.from_file(__file__)`
(the whole `test_setter_app.py` suite) keeps landing on exactly the screen
it did before this file became a shell.
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
# repo root on sys.path before importing anything from `setter`/`fuze`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from setter.pages import archive, mission_control

st.set_page_config(page_title="Simulation Setter", layout="wide")

pg = st.navigation([
    st.Page(mission_control.render, title="Mission Control", url_path="mission-control", default=True),
    st.Page(archive.render, title="Archive", url_path="archive"),
])
pg.run()
