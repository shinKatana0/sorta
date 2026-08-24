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
Measured on a 24-core machine, same commit, 5766 tests, both install profiles:

                     one process     this gate
    cpu profile      29 min 53 s     6 min 56 s
    gpu profile      29 min 50 s     6 min 45 s

**4.4x, and it was 7.5x before the workers were capped** — say it that way round. The
first version of this ran `-n auto`, one worker per logical core, and did the whole
gate in 4 min 00 s; that was measured on the cpu profile and it died on the gpu one,
where each worker is half a gigabyte of torch (see `_MAX_WORKERS`). The cap costs
about 40% of the win and buys a gate that finishes on both profiles and next to
somebody else's work. A gate that is fast on one machine is not a fast gate.

The docstring above used to say "~9 minutes". The suite had grown by a test per
feature and nobody was watching the number, so it was wrong by a factor of three. That
is why the run now prints its own duration per check and in total: the next divergence
should be visible on the next run rather than half a year later.

The suite is therefore run TWICE, and the split is not cosmetic:

    parallel   pytest -n <chosen, at most 8> -m "not serial"   the bulk
    serial     pytest -m serial                                the tests that assert
                                                 about TIME or bind a port — the
                                                 parallel half is a loaded machine,
                                                 which is exactly the condition under
                                                 which they fail

The cap on the workers is about memory and not about cores; see `_MAX_WORKERS` below.
Since F251 the cap is only the ceiling and the count under it is measured — the section
after this one.

A naive `-n auto` over everything would make the gate fast and UNRELIABLE, which is
worse than slow: an unreliable gate teaches people to re-run instead of to read. No
test was loosened to survive the parallel half — a test that cannot take it is
`serial`, with the reason written next to the marker. The serial half is 27 tests and
1 min 07 s of the 6 min 45 s; the parallel one is the other 5739.

Coverage is measured over the SUM of the two passes: each one writes with
`--cov-append` and neither judges the threshold (`--cov-fail-under=0`), and the
threshold from pyproject.toml is checked exactly once, at the end, by `coverage
report` over the combined data. Checking it per pass would mean judging 85% against
the dozen tests of the serial half — a red gate — and against the parallel half's
incomplete data — a green one that covers nothing.

The worker count is measured, not assumed (F251, 2026-08-24)
------------------------------------------------------------
Eight is now a CEILING and no longer the decision. The decision is

    workers = min(cores, 8, memory headroom / the budget per worker), at least 1

because the machine the gate runs on is not the machine the eight was measured on: over
2026-08-23/24 seven gate runs out of twelve died with `MemoryError`,
`RemoteDisconnected` or `ConnectionAbortedError`, every time on different, unrelated
tests, on a tree that was green before and after. Not one of them was about the branch.
They were about Windows' COMMIT limit — which counts what processes reserved, not what
they touched — standing at 59 of 74.6 GB before the gate started, held by a browser, WSL
and other sessions, while 17.6 GB of physical memory sat free. Eight workers of
CUDA-torch reserve about what was left, so the failure lands wherever the next
allocation happens to be.

Memory may only LOWER the count: on an idle 4-core CI runner the formula returns the
same 4 it always did, and a feature that fixed one laptop by slowing CI down would be a
bad trade. The budget per worker is measured and not estimated — `_COMMIT_BUDGET_MB` has
the table — and there is one per READING, because Windows answers with what is left of
its commit limit and Linux with memory that is actually free, and those are different
quantities however similar the words are. And the run says out loud what it picked and
out of what — always, not only when it lowered — because the alternative is what
actually happened: measuring the machine BY HAND, after the red gate, to find out
whether the machine was the reason.

    workers: 5 of 8 (24 cores) — 18.6 GB of commit headroom, 3.4 GB budgeted per worker

If the headroom does not cover even one worker the run does not start (exit
`_TOO_TIGHT`): eight minutes spent to arrive at a predictable `MemoryError` is not a
check, it is a wait. Where the headroom cannot be measured at all — macOS, a refusing
API — the count falls back to `min(cores, 8)`, exactly as before, and the line says
that this is what happened.

Used:
  - manually: uv run --extra cpu --extra dev python scripts/check.py
  - in CI:    the gate step of the workflow (.github/workflows/check.yml).
"""

import argparse
import os
import subprocess
import sys
import time
from typing import NamedTuple

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
    # The same files read as the OTHER platform. mypy checks for the machine it runs on,
    # so `ctypes.windll` — absent off Windows — passed here and failed both Linux runners
    # on 2026-08-10, after a green local gate. The product supports both, so both are
    # read; `--platform win32` is included so the same hole cannot open the other way on
    # a Linux developer's machine. ~20 s for a class of defect that otherwise only the
    # push finds.
    # Its OWN cache directory, and that is not tidiness: mypy keys one cache per
    # configuration, so two platforms sharing `.mypy_cache` evict each other and both
    # passes go cold every run — measured here as 3 s -> 1 m 24 s for the first pass.
    ("mypy (types, other platforms)",
     [sys.executable, "-m", "mypy", "--platform",
      "win32" if sys.platform != "win32" else "linux",
      "--cache-dir", ".mypy_cache_other", "sorta"]),
]

_COVERAGE = ["--cov=sorta", "--cov-append", "--cov-report="]
# Neither pass may judge the threshold: it belongs to the sum of the two, and the check
# for it is the last entry below. `--cov-fail-under=0` is how pytest-cov is told to keep
# `[tool.coverage.report].fail_under` out of ITS exit code without touching that value.
_NO_VERDICT = ["--cov-fail-under=0"]

# The number of workers is a question about MEMORY PER PROCESS, not about cores, and
# this is the line that says so. `-n auto` (one worker per logical core) was measured on
# the **cpu** install profile, where `import torch` costs a few dozen megabytes. The gate
# that decides a merge runs on the **gpu** profile, where the same import costs 505 MB —
# and 24 of those ran the machine out of memory: `MemoryError` inside a 1 MiB read in
# sorta/hashing.py, on a 63 GB box that had 26 GB free because a product run and another
# worktree's gate were using the rest.
#
# Measured on the gpu profile, peak resident memory of the run's own process tree
# (scripts/measure_gate_workers.py), against the wall clock of the parallel half:
#
#     workers    4      8     12     16     24
#     peak GB  17.3   20.0   23.1   25.5   ~31 (extrapolated; this is what failed)
#     seconds   516    310    240    205     —
#
# The peak is ~15 GB plus ~0.7 GB per worker: a floor set by whichever heavy files
# happen to overlap, and a slope that is the per-process cost. Eight is where the run
# still fits in the memory this machine actually had free with somebody else working on
# it, with about 4 GB to spare; twelve had one, sixteen had none. It is a cap and not a
# fixed count — `min(cores, 8)` — so a 4-core CI runner still gets 4 and needs no branch
# of its own in the configuration.
#
# Since F251 it is only the CEILING of that decision: the measurement above was true of
# one machine on one day and was never re-taken, so the run takes it itself now. See the
# docstring, `_BUDGET_MB` and `plan_workers`.
_MAX_WORKERS = 8

_BYTES_PER_MB = 1024 * 1024

# One budget per READING, because the two readings are not the same quantity — see
# `Headroom`. Measured 2026-08-24 on this machine, 24 cores, by sampling the parallel
# half's own process tree every 2 s for the sum of its PagefileUsage (private commit
# charge; the machine-wide figure would have measured who else was awake):
#
#     profile   workers   peak commit   per worker   parallel half
#     gpu             8      27 944 MB     3 493 MB          394 s
#     gpu             4      20 982 MB     5 246 MB          555 s
#     cpu             8      25 958 MB     3 245 MB          350 s
#     cpu             4      18 884 MB     4 721 MB          555 s
#
# The install profile turned out NOT to be the axis: the marginal cost of a worker is
# 1.74 GB on gpu and 1.77 GB on cpu — the profiles differ in the FLOOR (14.0 against
# 11.8 GB), which the suite's own subprocess-spawning tests pay by importing torch 30-odd
# times whatever `-n` says. So one number, taken at the ceiling: 3493 -> 3500.
#
# The trap in that number: a division cannot express a floor. Charging the amortised cost
# at 8 workers is right when 8 are being considered and optimistic below — on a machine
# with 15 GB of headroom the formula lands on 4 workers where the tree still wants ~19.
# It is the count that gets lowered, not a guarantee that the run fits; the printed line
# is what makes the remaining case readable instead of mysterious.
_COMMIT_BUDGET_MB = 3500

# MemAvailable is about pages actually held, not about reservations, so the commit figure
# above would be the wrong quantity to divide it by. This is F219's measured peak
# RESIDENT memory of the same tree (2026-08-07, gpu profile, 20.0 GB at 8 workers,
# scripts/measure_gate_workers.py). It is a Windows measurement of a Linux number because
# the project has no Linux machine; it is the first thing to re-take when it gets one.
_RESIDENT_BUDGET_MB = 2500

_BUDGET_MB = {"commit headroom": _COMMIT_BUDGET_MB, "available memory": _RESIDENT_BUDGET_MB}


class Headroom(NamedTuple):
    """How much memory the machine will let this run reserve, and what that number is.

    `megabytes` is None where the platform cannot be asked without a new dependency —
    an answer, not a failure. `name` differs per platform on purpose: Windows answers
    with what is left of the COMMIT limit, which counts reservations nobody has touched,
    and Linux with memory that is actually available. Pretending those are one fact is
    how a number stops meaning anything — and it is also why each carries a budget of
    its own, keyed by this name.
    """

    megabytes: int | None
    name: str


class WorkerPlan(NamedTuple):
    """The count the parallel half will use, and the line the run prints about it.

    `workers` is 0 when the headroom does not cover a single worker: the line is then
    the whole answer, and it is the caller that must not start. `headroom` travels with
    the plan so that the refusal quotes the reading the count was made from, and not
    whatever the machine looks like by the time it is printed.
    """

    workers: int
    line: str
    headroom: Headroom


def _windows_commit_headroom_mb() -> int | None:
    """`ullAvailPageFile` of GlobalMemoryStatusEx, in MB. None if the call fails.

    Not `ullAvailPhys`: the failures this feature is about arrived with 17.6 GB of
    physical memory free. Windows refuses an allocation when the COMMIT charge would
    pass the limit, and a worker of CUDA-torch reserves gigabytes whether or not it ever
    writes to them.
    """
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        # Reached through getattr: `windll` does not exist off Windows, where this file
        # is still imported and still read by mypy for the other platform.
        windll = getattr(ctypes, "windll", None)
        if windll is None or not windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPageFile) // _BYTES_PER_MB
    except Exception:
        return None


def _linux_available_mb() -> int | None:
    """`MemAvailable` of /proc/meminfo, in MB. None if it cannot be read.

    MemAvailable and not MemFree: on a box that has been up for a day most of MemFree is
    page cache a run may have back for the asking, and MemFree reads as an emergency on
    every machine.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8", errors="replace") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1]) * 1024 // _BYTES_PER_MB  # the file is in kB
                    return None
    except Exception:
        return None
    return None


def memory_headroom() -> Headroom:
    """What this machine will let the parallel half reserve. Never raises."""
    if sys.platform == "win32":
        return Headroom(_windows_commit_headroom_mb(), "commit headroom")
    if sys.platform.startswith("linux"):
        return Headroom(_linux_available_mb(), "available memory")
    # macOS has no reading here that means the same thing (`vm_stat` counts pages of six
    # kinds and their sum is not MemAvailable), and an invented number would be worse
    # than none: it would lower the count on somebody's machine for no reason.
    return Headroom(None, "available memory")


def _gb(megabytes: float) -> str:
    return f"{megabytes / 1024:.1f} GB"


def budget_for(headroom: Headroom) -> int:
    """What one worker is charged against this reading, in MB. See `_BUDGET_MB`."""
    return _BUDGET_MB[headroom.name]


def plan_workers(cores: int, headroom: Headroom, budget_mb: int | None = None,
                 ceiling: int = _MAX_WORKERS) -> WorkerPlan:
    """How many pytest workers this machine can afford, and the line that says why.

    Memory can only lower the count, never lift it above `min(cores, ceiling)`, and a
    machine that could not be measured gets exactly what it got before this existed.
    `budget_mb` defaults to the one measured for the reading `headroom` carries.
    """
    by_cores = max(1, min(cores, ceiling))
    if headroom.megabytes is None:
        return WorkerPlan(by_cores, f"workers: {by_cores} of {ceiling} ({cores} cores) — "
                                    f"{headroom.name} could not be measured on "
                                    f"{sys.platform}, so memory got no vote here", headroom)
    budget = budget_for(headroom) if budget_mb is None else budget_mb
    workers = min(by_cores, headroom.megabytes // budget)
    return WorkerPlan(workers, f"workers: {workers} of {ceiling} ({cores} cores) — "
                               f"{_gb(headroom.megabytes)} of {headroom.name}, "
                               f"{_gb(budget)} budgeted per worker", headroom)


def too_tight(plan: WorkerPlan) -> str:
    """What is said instead of starting a run that cannot fit. Names both numbers."""
    return (f"NOT STARTING: one worker is budgeted {_gb(budget_for(plan.headroom))} "
            f"and this machine has "
            f"{_gb(plan.headroom.megabytes or 0)} of {plan.headroom.name} left. The run "
            f"would reach a MemoryError in a few minutes instead of a verdict.\n"
            f"That figure counts every process on the machine and not this repository's: "
            f"another worktree's gate, a product run, the browser, WSL — close one of "
            f"them and start the gate again.")


_PLAN = plan_workers(os.cpu_count() or 1, memory_headroom())
# Trap: the command below is built even when the plan refuses (0 workers) — it is `main`
# that stops the run, and `-n 0` would otherwise reach pytest as a valid request.
_WORKERS = str(max(_PLAN.workers, 1))

# `loadfile` and not xdist's default `load` (which hands out one test at a time). The
# suite has modules whose tests share process state on purpose — a module-level
# `default_rng` in tests/test_faces_rescan.py, so what a test draws depends on how many
# draws the tests above it made. Splitting such a file across workers changes the data a
# test sees and flips its verdict; keeping the file whole reproduces exactly the order
# it had in one process. That is what makes `-n 1` and `-n 8` agree, which is the
# property the gate is FOR. Per-file granularity is plenty for eight workers: 225 files.
_DISTRIBUTION = ["-n", _WORKERS, "--dist", "loadfile"]

SLOW_CHECKS = [
    (
        "pytest (parallel half)",
        [sys.executable, "-m", "pytest", *_DISTRIBUTION, "-m", "not serial",
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

# Not 1: "the machine cannot host this run" is not "the branch is broken", and a caller
# that reads exit codes should be able to tell the two apart. Non-zero all the same —
# nothing was checked, so nothing may be reported green.
_TOO_TIGHT = 3


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
        # Printed on every run that will start pytest, not only on a run that lowered
        # the count: a red gate has to be readable as "the machine was tight" WITHOUT
        # measuring the machine by hand afterwards, and that only works if the number is
        # there on the green ones too.
        print(_PLAN.line)
        if _PLAN.workers < 1:
            print(too_tight(_PLAN))
            return _TOO_TIGHT
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
