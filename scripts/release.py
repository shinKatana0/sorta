#!/usr/bin/env python
"""Release helper — keep the version in sync and cut release notes.

Single source of truth is `pyproject.toml`; `sorta/__init__.py` and the top
`CHANGELOG.md` entry must agree. Run `check` before tagging (also good in CI).

    python scripts/release.py check           # verify the three versions match
    python scripts/release.py notes [X.Y.Z]   # print the CHANGELOG section (for gh release)

Release flow:
    1. bump `version` in pyproject.toml + `__version__` in sorta/__init__.py
    2. add a `## [X.Y.Z] - <date>` section to CHANGELOG.md
    3. python scripts/release.py check
    4. commit, then: git tag -a vX.Y.Z -m ... && git push origin vX.Y.Z
       gh release create vX.Y.Z --notes-file <(python scripts/release.py notes) --latest
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# CHANGELOG (and our own output) may contain non-ASCII (e.g. "×"); make stdout UTF-8
# so this works on a legacy Windows console codepage too.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
_SEMVER = r"[0-9]+\.[0-9]+\.[0-9]+"


def _pyproject() -> str:
    m = re.search(rf'^version\s*=\s*"({_SEMVER})"', (ROOT / "pyproject.toml").read_text("utf-8"), re.M)
    if not m:
        sys.exit("pyproject.toml: no `version = \"X.Y.Z\"`")
    return m.group(1)


def _init() -> str:
    m = re.search(rf'__version__\s*=\s*"({_SEMVER})"', (ROOT / "sorta" / "__init__.py").read_text("utf-8"))
    if not m:
        sys.exit("sorta/__init__.py: no `__version__ = \"X.Y.Z\"`")
    return m.group(1)


def _changelog_top() -> str:
    m = re.search(rf"^##\s*\[({_SEMVER})\]", (ROOT / "CHANGELOG.md").read_text("utf-8"), re.M)
    if not m:
        sys.exit("CHANGELOG.md: no `## [X.Y.Z]` entry")
    return m.group(1)


def _notes(version: str) -> str:
    """The CHANGELOG body for `version` (between its heading and the next `## `)."""
    text = (ROOT / "CHANGELOG.md").read_text("utf-8")
    m = re.search(rf"^##\s*\[{re.escape(version)}\][^\n]*\n(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    if not m:
        sys.exit(f"CHANGELOG.md: no section for {version}")
    return m.group(1).strip()


def cmd_check() -> int:
    py, ini, cl = _pyproject(), _init(), _changelog_top()
    print(f"pyproject.toml      : {py}")
    print(f"sorta/__init__.py   : {ini}")
    print(f"CHANGELOG.md (top)  : {cl}")
    if py == ini == cl:
        print(f"OK: versions agree ({py})")
        return 0
    print("MISMATCH: align all three before tagging.")
    return 1


def cmd_notes(argv: list[str]) -> int:
    version = argv[0] if argv else _pyproject()
    print(_notes(version))
    return 0


def main() -> int:
    args = sys.argv[1:]
    cmd = args[0] if args else "check"
    if cmd == "check":
        return cmd_check()
    if cmd == "notes":
        return cmd_notes(args[1:])
    sys.exit(f"usage: release.py [check|notes [X.Y.Z]]  (got {cmd!r})")


if __name__ == "__main__":
    raise SystemExit(main())
