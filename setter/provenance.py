"""
Provenance/staleness checks for script-registry entries (Control Room spec
Part B4): "compare each output's modification time against its inputs...
a genuinely useful engineering signal for one `stat` call."

Pure filesystem inspection -- no script is imported or run. Glob-pattern
outputs (`docs/figures/mc_*.png`) are resolved with `pathlib.Path.glob`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from setter.config import REPO_ROOT
from setter.registry import ScriptEntry

__all__ = ["Staleness", "check"]


@dataclass(frozen=True)
class Staleness:
    reads_found: Tuple[str, ...]
    reads_missing: Tuple[str, ...]
    writes_found: Tuple[str, ...]
    writes_missing: Tuple[str, ...]
    #: True = an input is newer than an output; False = outputs are current;
    #: None = nothing to judge (the entry writes nothing, or nothing it
    #; reads or writes exists on disk yet).
    stale: Optional[bool]
    newest_read_age_s: Optional[float]
    oldest_write_age_s: Optional[float]


def _resolve(pattern: str) -> list:
    if any(ch in pattern for ch in "*?["):
        return sorted(REPO_ROOT.glob(pattern))
    p = REPO_ROOT / pattern
    return [p] if p.exists() else []


def check(entry: ScriptEntry) -> Staleness:
    now = time.time()

    read_matches = {r: _resolve(r) for r in entry.reads}
    write_matches = {w: _resolve(w) for w in entry.writes}

    reads_found = tuple(r for r, m in read_matches.items() if m)
    reads_missing = tuple(r for r, m in read_matches.items() if not m)
    writes_found = tuple(w for w, m in write_matches.items() if m)
    writes_missing = tuple(w for w, m in write_matches.items() if not m)

    read_paths = [p for m in read_matches.values() for p in m]
    write_paths = [p for m in write_matches.values() for p in m]

    if not write_paths:
        stale = None
        oldest_write_age = None
    else:
        oldest_write_mtime = min(p.stat().st_mtime for p in write_paths)
        oldest_write_age = now - oldest_write_mtime
        stale = False if not read_paths else (
            max(p.stat().st_mtime for p in read_paths) > oldest_write_mtime)

    newest_read_age = (now - max(p.stat().st_mtime for p in read_paths)) if read_paths else None

    return Staleness(
        reads_found=reads_found, reads_missing=reads_missing,
        writes_found=writes_found, writes_missing=writes_missing,
        stale=stale, newest_read_age_s=newest_read_age, oldest_write_age_s=oldest_write_age,
    )
