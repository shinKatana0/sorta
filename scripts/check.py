#!/usr/bin/env python3
"""
check.py — the single quality gate of the sorta project.

Runs in order: ruff (lint) -> mypy (types) -> pytest with coverage
(the threshold is in pyproject.toml, [tool.coverage.report].fail_under).

Returns exit code 0 only if ALL checks passed. Stops at the first failed check and
prints which one failed — enough for an agent (or a human) to know what to fix.

Used:
  - manually: uv run --extra cpu --extra dev python scripts/check.py
  - in CI:    the gate step of the workflow (.github/workflows/check.yml).
"""

import subprocess
import sys

# The Windows console (cp1251) does not encode the emoji in the output below —
# without replace the script crashes with UnicodeEncodeError AFTER all gates have
# passed, and the exit code becomes non-zero on green checks.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(errors="replace")

CHECKS = [
    ("ruff (lint)", [sys.executable, "-m", "ruff", "check", "sorta", "tests"]),
    ("mypy (types)", [sys.executable, "-m", "mypy", "sorta"]),
    (
        "pytest (tests + coverage)",
        [sys.executable, "-m", "pytest", "--cov=sorta", "--cov-report=term-missing"],
    ),
]


def main() -> int:
    for name, cmd in CHECKS:
        print(f"\n=== {name} ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n❌ GATE FAILED: {name} (exit code {result.returncode})")
            print("Committing is blocked until this check is green.")
            return result.returncode
    print("\n✅ All gates passed (lint + types + tests/coverage).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
