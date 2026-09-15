"""
Safe subprocess execution for script-registry entries (Control Room spec
Part B4).

Never imports or execs a script's code directly -- always a fresh
`python -m <module>` subprocess with a hard timeout, so a hung or runaway
script can never take the Streamlit app down with it. Every registry entry
this module will actually run takes no required arguments (see
`setter/registry.py`): the fixed argument set for every runnable entry is
"none", the safest one available, rather than accepting anything from the
UI.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

from setter.config import REPO_ROOT
from setter.registry import ScriptEntry

__all__ = ["RunResult", "can_run", "run_entry", "DEFAULT_TIMEOUT_S"]

#: A hard ceiling well above anything a "reader"/"figure"/"analysis" entry
#: should cost (seconds to about a minute per the registry's own cost
#: classes) and well below what a "campaign" entry would need -- which is
#: moot, since `can_run` refuses those regardless of timeout.
DEFAULT_TIMEOUT_S = 180


@dataclass(frozen=True)
class RunResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def can_run(entry: ScriptEntry) -> bool:
    """The UI's own gate, independent of `entry.runnable`: a campaign-class
    entry is never runnable regardless of what `runnable` says, so a typo
    in the registry can't accidentally expose a multi-hour job to a button."""
    return entry.runnable and entry.cost_class != "campaign"


def run_entry(entry: ScriptEntry, timeout_s: int = DEFAULT_TIMEOUT_S) -> RunResult:
    """Run `entry` as `python -m {entry.module}` in a subprocess, cwd the
    repository root (so its own relative docs/ paths resolve the way they
    do from a terminal). Raises ValueError rather than running anything if
    `can_run(entry)` is false -- this is the last line of defence, not the
    only one; the UI must not offer a run control for such an entry either.
    """
    if not can_run(entry):
        raise ValueError(
            f"{entry.id!r} is not runnable from the UI "
            f"(cost_class={entry.cost_class!r}, runnable={entry.runnable!r})")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", entry.module],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout_s)
        return RunResult(ok=proc.returncode == 0, returncode=proc.returncode,
                          stdout=proc.stdout, stderr=proc.stderr, timed_out=False)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return RunResult(ok=False, returncode=-1, stdout=stdout, stderr=stderr, timed_out=True)
