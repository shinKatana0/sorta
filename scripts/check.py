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
    slow   pytest with coverage            see the measurement below

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

The slow half uses the machine (F219, 2026-08-07)
-------------------------------------------------
Measured before this feature: 28 minutes for 5742 tests on a 24-core machine, in ONE
pytest process. The docstring above said "~9 minutes" — the suite had grown by a test
per feature and nobody was watching the number, so it was wrong by a factor of three.
That is why the run now prints its own duration at the end: the next divergence should
be visible on the next run, not half a year later.

The suite is therefore run TWICE, and the split is not cosmetic:

    parallel   pytest -n auto -m "not serial"    the bulk, one worker per core
    serial     pytest -m serial                  the tests that assert about TIME or
                                                 bind a port — `-n auto` is a loaded
                                                 machine, which is exactly the
                                                 condition under which they fail

A naive `-n auto` over everything would make the gate fast and UNRELIABLE, which is
worse than slow: an unreliable gate teaches people to re-run instead of to read. No
test was loosened to survive the parallel half — a test that cannot take it is
`serial`, with the reason written next to the marker.

Coverage is measured over the SUM of the two passes: each one writes with
`--cov-append` and neither judges the threshold (`--cov-fail-under=0`), and the
threshold from pyproject.toml is checked exactly once, at the end, by `coverage
report` over the combined data. Checking it per pass would mean judging 85% against
the dozen tests of the serial half — a red gate — and against the parallel half's
incomplete data — a green one that covers nothing.

Used:
  - manually: uv run --extra cpu --extra dev python scripts/check.py
  - in CI:    the gate step of the workflow (.github/workflows/check.yml).
"""

import argparse
import os
import subprocess
import sys
import time

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

_COVERAGE = ["--cov=sorta", "--cov-append", "--cov-report="]
# Neither pass may judge the threshold: it belongs to the sum of the two, and the check
# for it is the last entry below. `--cov-fail-under=0` is how pytest-cov is told to keep
# `[tool.coverage.report].fail_under` out of ITS exit code without touching that value.
_NO_VERDICT = ["--cov-fail-under=0"]

SLOW_CHECKS = [
    (
        "pytest (parallel half)",
        [sys.executable, "-m", "pytest", "-n", "auto", "-m", "not serial",
         *_COVERAGE, *_NO_VERDICT],
    ),
    (
        "pytest (serial half)",
        [sys.executable, "-m", "pytest", "-m", "serial", *_COVERAGE, *_NO_VERDICT],
    ),
    (
        "coverage threshold",
        [sys.executable, "-m", "coverage", "report", "--show-missing"],
    ),
]

# `--cov-append` on both passes means the previous run's data would be added to this
# one's, and a file deleted since then would keep its coverage forever. The gate erases
# once, before the first pass, instead of letting the first pass erase (it cannot: it
# appends) — one place, and it also covers a run that starts with the serial half.
ERASE_COVERAGE = [sys.executable, "-m", "coverage", "erase"]

# pytest's "no tests were collected". Only the serial half may legitimately hit it —
# if every `serial` marker is ever removed, that is an empty pass, not a failed gate.
_NO_TESTS_COLLECTED = 5


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def run(checks: list[tuple[str, list[str]]]) -> int:
    started = time.monotonic()
    for name, cmd in checks:
        print(f"\n=== {name} ===")
        step = time.monotonic()
        result = subprocess.run(cmd)
        # Printed per check, not only as a total: this is the number that tells whether
        # the parallel half or the serial one is what a slow gate is waiting for.
        print(f"--- {name}: {_duration(time.monotonic() - step)}")
        if result.returncode != 0 and not (
            name == "pytest (serial half)" and result.returncode == _NO_TESTS_COLLECTED
        ):
            print(f"\n❌ GATE FAILED: {name} (exit code {result.returncode})")
            print(f"Total: {_duration(time.monotonic() - started)}")
            print("Committing is blocked until this check is green.")
            return result.returncode
    print(f"\nTotal: {_duration(time.monotonic() - started)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fast", action="store_true",
                       help="version sync + ruff + mypy (seconds) — run before committing")
    group.add_argument("--slow", action="store_true",
                       help="pytest with coverage (minutes) — run it in the background "
                            "and wait for it")
    args = parser.parse_args()

    if args.fast:
        checks, done = FAST_CHECKS, "✅ Fast gate passed (version + lint + types)."
    elif args.slow:
        checks, done = SLOW_CHECKS, "✅ Slow gate passed (tests + coverage)."
    else:
        checks = FAST_CHECKS + SLOW_CHECKS
        done = "✅ All gates passed (lint + types + tests/coverage)."

    if not args.fast:
        subprocess.run(ERASE_COVERAGE)

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
