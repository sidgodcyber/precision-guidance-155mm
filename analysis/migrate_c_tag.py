"""
One-off migration: move Task C under a tag level, as Task A already is.

WHY THIS EXISTS
---------------
`analysis.monte_carlo.main` dispatched Task A to `out["a"][tag][engagement]`
and Task C to `out["c"][engagement]` -- no tag level. `--tag` was accepted by
the parser and silently ignored by Task C, so a second Task C campaign at the
same engagement REPLACED the first in place, whatever tag it was given, and
the generated tables and figures would have been regenerated from the
replacement with nothing marking the change.

Nothing wrong was ever published as a result: every task ran once per
engagement, and the one repeated Task C invocation was a deliberate restart
whose first attempt was killed at 32/128 of its first arm and never reached
`_write`. The defect was latent. This migration is what makes it stay that
way.

WHAT IT DOES AND DOES NOT DO
----------------------------
It moves `d["c"][engagement]` to `d["c"][tag][engagement]`. It does not
recompute, round or re-serialise any number: the values are moved as loaded
Python objects and dumped with exactly the parameters `monte_carlo._write`
uses, which a round-trip check confirms is byte-identical for everything it
does not move.

It is idempotent. The already-migrated shape is detected structurally -- a
Task C campaign dict carries an `ages` key and a tag level does not -- rather
than by looking for known tag names, so it cannot be confused by a tag called
`long`.

SCOPE
-----
Task C only. `cs`, `d`, `e`, `n`, `g` and `b` share the same flat keying and
are deliberately NOT migrated: re-keying six tasks touches every consumer of
the campaign JSON and puts the published record at risk for a run that needs
none of them. They are recorded as a known latent defect in
docs/STEP6-CLOSEOUT.md.

Run:  python -m analysis.migrate_c_tag            # migrate in place
      python -m analysis.migrate_c_tag --check    # report, change nothing
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

DEFAULT_PATH = "docs/monte_carlo.json"
DEFAULT_TAG = "headline"

#: The number this migration must not move. docs/CEP-FINAL.md section 1.1 and
#: docs/ATMOSPHERIC-ERROR.md section 4 both quote it to one decimal, and it is
#: the basis of step 6's finding that the kit is knowledge-limited rather than
#: authority-limited. If it does not read back, the migration stops.
GUARD_PATH = ("2h", "knowledge_term", "sigma_range_m")
GUARD_VALUE = 148.0
GUARD_DECIMALS = 1


def is_campaign(v) -> bool:
    """A Task C campaign dict, as opposed to a tag level holding several."""
    return isinstance(v, dict) and "ages" in v


def shape_of(c: dict) -> str:
    """`flat`, `tagged`, `mixed` or `empty`, decided structurally."""
    if not c:
        return "empty"
    campaigns = [k for k, v in c.items() if is_campaign(v)]
    if len(campaigns) == len(c):
        return "flat"
    if not campaigns:
        return "tagged"
    return "mixed"


def migrate(path: str = DEFAULT_PATH, tag: str = DEFAULT_TAG,
            check_only: bool = False, backup: bool = True) -> dict:
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    d = json.loads(src)

    c = d.get("c")
    if c is None:
        return {"action": "none", "reason": "no Task C in this file"}

    shape = shape_of(c)
    if shape == "mixed":
        raise SystemExit(
            f"{path}: d['c'] holds both campaigns and tag levels "
            f"({sorted(c)}). Refusing to guess. Inspect it by hand.")
    if shape == "tagged":
        _guard(d["c"], tag, path, required=False)
        return {"action": "none", "reason": "already tagged",
                "tags": sorted(c), "shape": shape}
    if shape == "empty":
        return {"action": "none", "reason": "d['c'] is empty"}

    moved = sorted(c)
    if check_only:
        return {"action": "would-migrate", "tag": tag, "engagements": moved,
                "shape": shape}

    # The move. `c` is the object already loaded from the file; it is placed
    # under the tag key unchanged. Nothing is rebuilt.
    d["c"] = {tag: c}
    _guard(d["c"], tag, path, required=True)

    if backup:
        bak = path + ".pre-tag-migration.bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1, default=float)
    os.replace(tmp, path)
    return {"action": "migrated", "tag": tag, "engagements": moved,
            "backup": (path + ".pre-tag-migration.bak") if backup else None}


def _guard(tagged: dict, tag: str, path: str, required: bool) -> None:
    """
    Assert the one number the record rests on still reads back.

    `required=False` on an already-tagged file: the guard is a check that the
    file is the one we think it is, not a demand that some particular tag
    exists, so a file carrying only a `physical` campaign passes.
    """
    camp = (tagged.get(tag) or {}).get("long")
    if camp is None:
        if required:
            raise SystemExit(f"{path}: no d['c'][{tag!r}]['long'] after the "
                             f"move. Refusing to write.")
        return
    node = camp["ages"]
    for k in GUARD_PATH:
        node = node[k]
    got = round(float(node), GUARD_DECIMALS)
    if got != GUARD_VALUE:
        raise SystemExit(
            f"{path}: the atmospheric knowledge term reads {got} m, not "
            f"{GUARD_VALUE} m. docs/CEP-FINAL.md rests on this number. "
            f"Refusing to write.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--check", action="store_true",
                    help="report what would happen and change nothing")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args(argv)
    r = migrate(args.path, args.tag, check_only=args.check,
                backup=not args.no_backup)
    for k, v in r.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
