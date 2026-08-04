#!/usr/bin/env python3
"""
check.py — the single quality gate of the sorta project.

Runs in order: version sync -> ruff (lint) -> mypy (types) -> pytest with coverage
(the threshold is in pyproject.toml, [tool.coverage.report].fail_under).

Returns exit code 0 only if ALL selected checks passed. Stops at the first failed
check and prints which one failed — enough for an agent (or a human) to know what to
fix.

Two halves, and why (F-tooling, 2026-07-29)
-------------------------------------------
The checks split cleanly by how long they take:

    fast   version sync + ruff + mypy      seconds
    slow   pytest with coverage            ~9 minutes on this collection's suite

That difference stopped being cosmetic. A worker agent's shell moves any command
longer than 600 s into the background, and the full gate sits just under that: under
load it crosses the line, the session ends while the run is still going, and the work
stays uncommitted because committing is only allowed after a green gate. Six sessions
out of thirteen ended that way on 2026-07-28/29 — never losing work, but always
costing the orchestrator a manual gate run and a commit on the worker's behalf.

So the contract is now:

    python scripts/check.py --fast    before committing — seconds, catches the
                                      mistakes that make a diff not worth reading
    python scripts/check.py --slow    the test suite, run it in the background and
                                      wait for it
    python scripts/check.py           everything, unchanged — what the orchestrator
                                      runs before a merge and what CI runs

The safety net does not move: the orchestrator runs the FULL gate on the branch
before merging, and never trusts a self-report. Splitting only changes who waits for
the slow half, not whether it is run.

Used:
  - manually: uv run --extra cpu --extra dev python scripts/check.py
  - in CI:    the gate step of the workflow (.github/workflows/check.yml).
"""

import argparse
import os
import subprocess
import sys

# The Windows console (cp1251) does not encode the emoji in the output below —
# without replace the script crashes with UnicodeEncodeError AFTER all gates have
# passed, and the exit code becomes non-zero on green checks.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))

FAST_CHECKS = [
    ("version sync", [sys.executable, os.path.join(_HERE, "release.py"), "check"]),
    # `scripts` is in the list because the fast gate is now the ONLY check between
    # writing and committing (the rule of 2026-08-04), and it did not look here. Found by
    # walking into it: a multi-line `help=` left an unterminated string literal in
    # measure_deblur.py and --fast said green. Five measurement modules live here and are
    # merged into main with their tests; a file that does not parse must not pass.
    ("ruff (lint)", [sys.executable, "-m", "ruff", "check", "sorta", "tests", "scripts"]),
    ("mypy (types)", [sys.executable, "-m", "mypy", "sorta"]),
]

SLOW_CHECKS = [
    (
        "pytest (tests + coverage)",
        [sys.executable, "-m", "pytest", "--cov=sorta", "--cov-report=term-missing"],
    ),
]


def run(checks: list[tuple[str, list[str]]]) -> int:
    for name, cmd in checks:
        print(f"\n=== {name} ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n❌ GATE FAILED: {name} (exit code {result.returncode})")
            print("Committing is blocked until this check is green.")
            return result.returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fast", action="store_true",
                       help="version sync + ruff + mypy (seconds) — run before committing")
    group.add_argument("--slow", action="store_true",
                       help="pytest with coverage (~9 min) — run it in the background "
                            "and wait for it")
    args = parser.parse_args()

    if args.fast:
        checks, done = FAST_CHECKS, "✅ Fast gate passed (version + lint + types)."
    elif args.slow:
        checks, done = SLOW_CHECKS, "✅ Slow gate passed (tests + coverage)."
    else:
        checks = FAST_CHECKS + SLOW_CHECKS
        done = "✅ All gates passed (lint + types + tests/coverage)."

    code = run(checks)
    if code:
        return code
    print(f"\n{done}")
    if args.fast:
        # Without this line the split quietly becomes a weaker gate: a fast pass reads
        # like a green light, and the suite never runs at all.
        print("The test suite has NOT run. Start `--slow` in the background and wait "
              "for it before reporting the work as done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
