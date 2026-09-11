"""
Static configuration for the setter package: paths into the frozen
repository's data products, and the fixed vocabularies (engagements, met-age
buckets, schema identifiers) the rest of the package validates against.

Nothing in this module reads or writes simulation state -- it only names
where things are.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

GUIDANCE_MAP_PATH = DOCS_DIR / "guidance_map.json"
MONTE_CARLO_PATH = DOCS_DIR / "monte_carlo.json"

#: The five supported named engagements. The campaign/guidance data in
#: docs/guidance_map.json and docs/monte_carlo.json is tied to exactly these;
#: there is no live path from arbitrary target coordinates to a new
#: campaign-quality CEP (see setter/campaign.py, setter/simulation_adapter.py).
SUPPORTED_ENGAGEMENTS = ("short", "short2", "middle", "mid2", "long")

#: Task C's seven met-message-age cases (docs/monte_carlo.json, key "c").
MET_AGE_BUCKETS = ("perfect", "0h", "1h", "2h", "3h", "6h", "none")

#: The three named Task A campaign points (docs/monte_carlo.json, key "a").
TASK_A_TAGS = ("headline", "fresh_met", "physical")

SCHEMA_ID = "SIM-SETTER-V1"
SCHEMA_VERSION = "1.0.0"
