"""F251: the worker count is measured on the machine, and said out loud.

`_MAX_WORKERS = 8` was measured once, on one machine, on one day, and was then trusted
forever. Over 2026-08-23/24 seven gate runs out of twelve died — `MemoryError`,
`RemoteDisconnected`, `ConnectionAbortedError` — every time on different tests, on a tree
that was green before and after. The machine was at 59 GB of a 74.6 GB COMMIT limit
before the gate started, with 17.6 GB of physical memory free: enough memory, no
allowance left to reserve it with.

What this module holds in place, one class per claim: the count is the largest one whose
estimated cost fits with `_RESERVE_MB` unspent; nothing refuses to run; a platform that
cannot be measured is left exactly as it was before this existed; and the line is printed
on every run that will start pytest, the ones that got the full count included — reading
a red gate should not require measuring the machine by hand afterwards, and that only
works if the green runs carry the number too.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest import mock

from tests.test_gate_parallel import load_check, pytest_args

_ROOT = Path(__file__).resolve().parent.parent

# Enough for the ceiling under any cost model this feature would accept. Not
# `sys.maxsize`: the line prints the number, and a machine with 8 exabytes free is not a
# readable fixture.
_PLENTY_MB = 512 * 1024

# The reading this machine's gate uses, and the one every scenario below is written in
# unless it says otherwise.
_COMMIT = "commit headroom"


class TestTheCountFollowsTheMeasurement(unittest.TestCase):
    """The estimate: floor + workers x marginal, inside the headroom, reserve unspent."""

    def setUp(self) -> None:
        self.check = load_check()

    def plan(self, cores: int, megabytes: int | None, name: str = _COMMIT) -> Any:
        return self.check.plan_workers(cores, self.check.Headroom(megabytes, name))

    def spendable(self, megabytes: int) -> int:
        return megabytes - self.check._RESERVE_MB

    def test_plenty_of_headroom_is_the_count_this_gate_always_had(self):
        for cores, expected in [(24, 8), (16, 8), (8, 8), (4, 4), (2, 2), (1, 1)]:
            with self.subTest(cores=cores):
                self.assertEqual(self.plan(cores, _PLENTY_MB).workers, expected)

    def test_a_small_machine_is_never_given_more_workers_than_it_has_cores(self):
        """Memory has one direction. A 2-core box with a terabyte free still gets 2."""
        self.assertEqual(self.plan(2, _PLENTY_MB).workers, 2)

    def test_the_count_it_lands_on_is_the_largest_one_that_fits(self):
        """Both halves: what was chosen fits, and one more would not have."""
        for megabytes in range(16 * 1024, 40 * 1024, 512):
            plan = self.plan(24, megabytes)
            headroom = self.check.Headroom(megabytes, _COMMIT)
            with self.subTest(megabytes=megabytes, workers=plan.workers):
                if plan.workers > self.check._MIN_WORKERS:
                    self.assertLessEqual(self.check.cost_of(plan.workers, headroom),
                                         self.spendable(megabytes))
                if plan.workers < self.check._MAX_WORKERS:
                    self.assertGreater(self.check.cost_of(plan.workers + 1, headroom),
                                       self.spendable(megabytes))

    def test_the_reserve_is_left_unspent_and_not_handed_to_a_worker(self):
        """The count may not rise when the reserve is the only thing paying for it."""
        headroom = self.check.Headroom(0, _COMMIT)
        for workers in range(self.check._MIN_WORKERS + 1, self.check._MAX_WORKERS + 1):
            exactly_enough = self.check.cost_of(workers, headroom)
            with self.subTest(workers=workers):
                self.assertLess(self.plan(24, exactly_enough).workers, workers)
                self.assertEqual(
                    self.plan(24, exactly_enough + self.check._RESERVE_MB).workers, workers)

    def test_more_headroom_never_means_fewer_workers(self):
        counts = [self.plan(24, megabytes).workers
                  for megabytes in range(2 * 1024, 48 * 1024, 256)]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[0], self.check._MIN_WORKERS)
        self.assertEqual(counts[-1], self.check._MAX_WORKERS)

    def test_a_machine_that_cannot_be_measured_behaves_as_before(self):
        self.assertEqual(self.plan(24, None).workers, 8)
        self.assertEqual(self.plan(4, None).workers, 4)

    def test_zero_cores_still_leaves_a_worker(self):
        """`os.cpu_count()` may answer None; the gate must not then ask for -n 0."""
        self.assertEqual(self.plan(1, _PLENTY_MB).workers, 1)
        self.assertEqual(self.plan(0, _PLENTY_MB).workers, 1)


class TestTheCaseTheQuotientDiedOn(unittest.TestCase):
    """The scenario that replaced the first model, kept as a fixture so it cannot come
    back. On 2026-08-24 the gate had 26.9 GB of headroom; the amortised quotient of
    3500 MB per worker bought 7, and 7 workers want 14.0 + 7 x 1.74 = 26.2 GB — the run
    met a `MemoryError` with 0.7 GB nominally to spare."""

    HEADROOM_MB = 26 * 1024 + 921  # 26.9 GB, the reading taken that day
    AMORTISED_MB = 3500  # what the first model charged per worker

    def setUp(self) -> None:
        self.check = load_check()
        self.headroom = self.check.Headroom(self.HEADROOM_MB, _COMMIT)

    def test_the_old_arithmetic_really_did_say_seven(self):
        """Guard the guard: if this stops being 7, the fixture has stopped modelling the
        day it is named after and the test below proves nothing."""
        self.assertEqual(self.HEADROOM_MB // self.AMORTISED_MB, 7)

    def test_the_model_that_replaced_it_takes_fewer(self):
        plan = self.check.plan_workers(24, self.headroom)
        self.assertLess(plan.workers, 7)
        self.assertLessEqual(self.check.cost_of(plan.workers, self.headroom),
                             self.HEADROOM_MB - self.check._RESERVE_MB)


class TestNothingRefusesToRun(unittest.TestCase):
    """FINDING of 2026-08-24, and the reason there is no refusal path to test: the cost
    model is an intercept fitted through two points on ONE machine, it says a 4-core,
    16 GB runner cannot fit four workers, and CI ran four on ubuntu-latest and
    windows-latest three times that day. A gate only its author can run is not a gate."""

    def setUp(self) -> None:
        self.check = load_check()

    def test_no_reading_however_small_takes_the_count_below_the_floor(self):
        for name in self.check._FLOOR_MB:
            for megabytes in (0, 1, 512, 4 * 1024, 12 * 1024):
                for cores in (1, 2, 4, 24):
                    with self.subTest(name=name, megabytes=megabytes, cores=cores):
                        plan = self.check.plan_workers(
                            cores, self.check.Headroom(megabytes, name))
                        self.assertEqual(plan.workers,
                                         min(cores, self.check._MIN_WORKERS)
                                         if cores > 1 else 1)

    def test_a_machine_below_the_fitted_floor_still_runs_the_suite(self):
        """The CI counter-example itself: less headroom than the model claims one worker
        needs, and the answer is still four workers rather than a refusal."""
        starved = self.check.Headroom(self.check._FLOOR_MB[_COMMIT] // 4, _COMMIT)
        self.assertGreater(self.check.cost_of(1, starved), starved.megabytes)
        self.assertEqual(self.check.plan_workers(4, starved).workers, 4)

    def test_the_gate_starts_the_run_on_a_machine_with_nothing_free(self):
        """End to end: no exit code of its own, no message instead of a run."""
        calls: list[list[str]] = []

        def record(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        plan = self.check.plan_workers(24, self.check.Headroom(0, _COMMIT))
        with mock.patch.object(self.check.subprocess, "run", side_effect=record), \
                mock.patch.object(self.check, "_PLAN", plan), \
                mock.patch.object(sys, "argv", ["check.py", "--slow"]), \
                redirect_stdout(StringIO()):
            self.assertEqual(self.check.main(), 0)
        self.assertEqual(calls[0], self.check.ERASE_COVERAGE)
        self.assertTrue(any("pytest" in cmd for cmd in calls))


class TestCiIsNotMadeSlowerByThis(unittest.TestCase):
    """The trade this feature is not allowed to make: on a 4-core runner with nothing
    else on it the answer stays the cores, byte-for-byte what `min(cores, 8)` gave."""

    # 4 cores, and the free figure differs by platform because the READING does:
    # `MemAvailable` on an idle linux runner is most of the box, while a Windows commit
    # headroom is what the image's page file leaves over the charge. Deliberately taken
    # low — below the fitted floor for either reading — because that is the case the
    # runners actually presented and passed.
    RUNNER_CORES = 4
    RUNNER_HEADROOM_MB = {"available memory": 8 * 1024, "commit headroom": 6 * 1024}

    def setUp(self) -> None:
        self.check = load_check()

    def test_an_idle_runner_gets_every_core_on_both_readings(self):
        for cores in (2, 4):
            for name in self.RUNNER_HEADROOM_MB:
                with self.subTest(cores=cores, name=name):
                    plan = self.check.plan_workers(cores, self.check.Headroom(_PLENTY_MB, name))
                    self.assertEqual(plan.workers, min(cores, self.check._MAX_WORKERS))

    def test_a_github_runner_still_gets_all_four_of_its_cores(self):
        for name, megabytes in self.RUNNER_HEADROOM_MB.items():
            with self.subTest(name=name):
                plan = self.check.plan_workers(
                    self.RUNNER_CORES, self.check.Headroom(megabytes, name))
                self.assertEqual(plan.workers, self.RUNNER_CORES)

    def test_the_floor_under_the_count_is_the_runner_count(self):
        """`_MIN_WORKERS` is not a taste: it is the count CI is observed to run."""
        self.assertGreaterEqual(self.check._MIN_WORKERS, self.RUNNER_CORES)
        self.assertLessEqual(self.check._MIN_WORKERS, self.check._MAX_WORKERS)


class TestTheLineSaysWhatWasChosenAndWhy(unittest.TestCase):
    """Half the value of the feature: the number, on every run, before the wait."""

    def setUp(self) -> None:
        self.check = load_check()

    def line(self, cores: int, megabytes: int | None, name: str = _COMMIT) -> str:
        return self.check.plan_workers(cores, self.check.Headroom(megabytes, name)).line

    def test_the_full_count_is_reported_as_loudly_as_a_lowered_one(self):
        full = self.line(24, _PLENTY_MB)
        self.assertRegex(full, r"^workers: 8 of 8 \(24 cores\)")
        self.assertIn("floor", full)
        self.assertIn("per worker", full)

    def test_the_line_names_every_number_the_decision_was_made_from(self):
        megabytes = 20 * 1024
        line = self.line(24, megabytes)
        for number in (megabytes, self.check._FLOOR_MB[_COMMIT],
                       self.check._MARGINAL_MB[_COMMIT], self.check._RESERVE_MB):
            self.assertIn(self.check._gb(number), line)

    def test_the_line_names_the_reading_it_used_and_platforms_differ(self):
        self.assertIn("commit headroom", self.line(24, 20 * 1024))
        self.assertIn("available memory", self.line(24, 20 * 1024, "available memory"))

    def test_each_reading_is_priced_with_the_numbers_taken_for_it(self):
        """The two readings are different quantities — a reservation Windows counts
        against its limit and a page Linux really holds — so one pair for both would be
        a number that is wrong on at least one platform."""
        for name in self.check._FLOOR_MB:
            with self.subTest(name=name):
                line = self.line(24, 20 * 1024, name)
                self.assertIn(self.check._gb(self.check._FLOOR_MB[name]), line)
                self.assertIn(self.check._gb(self.check._MARGINAL_MB[name]), line)

    def test_a_failed_measurement_is_said_and_not_hidden(self):
        line = self.line(24, None)
        self.assertRegex(line, r"^workers: 8 of 8 \(24 cores\)")
        self.assertIn("could not be measured", line)

    def test_gigabytes_are_printed_and_not_raw_megabytes(self):
        self.assertEqual(self.check._gb(2400), "2.3 GB")
        self.assertEqual(self.check._gb(0), "0.0 GB")


class TestTheProbeNeverBreaksTheGate(unittest.TestCase):
    """A gate that dies while asking how much memory there is has made things worse."""

    def setUp(self) -> None:
        self.check = load_check()

    def test_this_machine_answers_with_an_integer_or_with_nothing(self):
        headroom = self.check.memory_headroom()
        self.assertIsInstance(headroom.name, str)
        if headroom.megabytes is not None:
            self.assertIsInstance(headroom.megabytes, int)
            self.assertGreater(headroom.megabytes, 0)

    @unittest.skipUnless(sys.platform in ("win32",) or sys.platform.startswith("linux"),
                         "only these two platforms have a reading of their own")
    def test_a_supported_platform_really_does_produce_a_number(self):
        """Not `assertIsNotNone` for its own sake: a probe that silently answers None
        everywhere would pass every other test in this module while doing nothing."""
        self.assertIsNotNone(self.check.memory_headroom().megabytes)

    def test_the_windows_probe_answers_none_when_the_api_is_not_there(self):
        import ctypes

        with mock.patch.object(ctypes, "windll", None, create=True):
            self.assertIsNone(self.check._windows_commit_headroom_mb())

    def test_the_windows_probe_answers_none_when_the_call_fails(self):
        import ctypes

        failing = mock.Mock()
        failing.kernel32.GlobalMemoryStatusEx.return_value = 0
        with mock.patch.object(ctypes, "windll", failing, create=True):
            self.assertIsNone(self.check._windows_commit_headroom_mb())

    def test_the_linux_probe_answers_none_on_a_meminfo_it_cannot_read(self):
        with mock.patch("builtins.open", side_effect=OSError("no such file")):
            self.assertIsNone(self.check._linux_available_mb())

    def test_the_linux_probe_reads_kilobytes_and_reports_megabytes(self):
        meminfo = "MemTotal:       16305380 kB\nMemAvailable:    8388608 kB\nSwapFree: 0 kB\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=meminfo)):
            self.assertEqual(self.check._linux_available_mb(), 8192)

    def test_a_meminfo_without_the_field_is_no_answer_rather_than_a_wrong_one(self):
        with mock.patch("builtins.open", mock.mock_open(read_data="MemTotal: 100 kB\n")):
            self.assertIsNone(self.check._linux_available_mb())

    def test_each_platform_is_asked_the_question_it_can_answer(self):
        """macOS included: it has no reading that means the same thing, and inventing
        one would lower somebody's worker count for no reason."""
        with mock.patch.object(self.check.sys, "platform", "darwin"):
            self.assertEqual(self.check.memory_headroom(), (None, "available memory"))
        with mock.patch.object(self.check.sys, "platform", "linux"), \
                mock.patch.object(self.check, "_linux_available_mb", return_value=1234):
            self.assertEqual(self.check.memory_headroom(), (1234, "available memory"))
        with mock.patch.object(self.check.sys, "platform", "win32"), \
                mock.patch.object(self.check, "_windows_commit_headroom_mb",
                                  return_value=4321):
            self.assertEqual(self.check.memory_headroom(), (4321, "commit headroom"))


class TestTheRunActsOnThePlan(unittest.TestCase):
    """What `main` does with the plan: prints it, at the top, on every run with tests."""

    def setUp(self) -> None:
        self.check = load_check()

    def invoke(self, plan: Any, *argv: str) -> tuple[int, str, list[list[str]]]:
        calls: list[list[str]] = []

        def record(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        out = StringIO()
        with mock.patch.object(self.check.subprocess, "run", side_effect=record), \
                mock.patch.object(self.check, "_PLAN", plan), \
                mock.patch.object(sys, "argv", ["check.py", *argv]), \
                redirect_stdout(out):
            code = self.check.main()
        return code, out.getvalue(), calls

    def plan_for(self, megabytes: int | None, cores: int = 24) -> Any:
        return self.check.plan_workers(cores, self.check.Headroom(megabytes, _COMMIT))

    def test_the_slow_half_prints_the_line_before_it_starts(self):
        plan = self.plan_for(_PLENTY_MB)
        code, printed, calls = self.invoke(plan, "--slow")
        self.assertEqual(code, 0)
        self.assertIn(plan.line, printed)
        self.assertLess(printed.index(plan.line), printed.index("pytest"),
                        "the point of the line is to be there BEFORE the wait")
        self.assertTrue(calls)

    def test_the_line_is_flushed_or_a_captured_run_shows_it_last(self):
        """The buffering is the whole difference between the line being read and not.

        Everyone who reads a gate log — the orchestrator, CI, an agent — reads it
        through a pipe, and a piped stdout is block-buffered while each check writes to
        that same pipe from a subprocess directly.
        """
        plan = self.plan_for(_PLENTY_MB)
        with mock.patch.object(self.check.subprocess, "run",
                               return_value=subprocess.CompletedProcess([], 0)), \
                mock.patch.object(self.check, "_PLAN", plan), \
                mock.patch.object(sys, "argv", ["check.py", "--slow"]), \
                mock.patch("builtins.print") as printed:
            self.check.main()
        wrote_the_line = [call for call in printed.call_args_list if plan.line in call.args]
        self.assertTrue(wrote_the_line, printed.call_args_list)
        self.assertTrue(all(call.kwargs.get("flush") for call in wrote_the_line),
                        "the line must not wait in a buffer for the run it precedes")

    def test_the_full_gate_prints_it_too(self):
        plan = self.plan_for(_PLENTY_MB)
        _code, printed, _calls = self.invoke(plan)
        self.assertIn(plan.line, printed)

    def test_a_lowered_count_is_printed_the_same_way_as_a_full_one(self):
        lowered = self.plan_for(20 * 1024)
        self.assertLess(lowered.workers, self.check._MAX_WORKERS)
        _code, printed, _calls = self.invoke(lowered, "--slow")
        self.assertIn(lowered.line, printed)

    def test_the_fast_gate_says_nothing_about_workers_because_it_starts_none(self):
        _code, printed, _calls = self.invoke(self.plan_for(_PLENTY_MB), "--fast")
        self.assertNotIn("workers:", printed)


class TestTheCommandCarriesTheChosenCount(unittest.TestCase):
    """The plan is worthless if the pytest invocation does not follow it."""

    def setUp(self) -> None:
        self.check = load_check()

    def test_the_parallel_half_runs_at_the_planned_count(self):
        args = pytest_args(self.check.SLOW_CHECKS[0][1])
        self.assertEqual(args[args.index("-n") + 1], str(self.check._PLAN.workers))

    def test_the_count_is_a_number_and_never_zero_or_auto(self):
        args = pytest_args(self.check.SLOW_CHECKS[0][1])
        workers = int(args[args.index("-n") + 1])
        self.assertGreaterEqual(workers, 1)
        self.assertLessEqual(workers, self.check._MAX_WORKERS)
        self.assertLessEqual(workers, max(os.cpu_count() or 1, 1))

    def test_the_ceiling_is_still_eight(self):
        """Out of scope for this feature — a change here is somebody else's decision."""
        self.assertEqual(self.check._MAX_WORKERS, 8)


class TestTheNumbersSayWhereTheyCameFrom(unittest.TestCase):
    """A number that is a guess is the same as the old eight, only newer — so the two
    that are measured are findable in the CHANGELOG with their conditions, and the two
    that are reasoned are labelled as reasoned where they are defined."""

    def setUp(self) -> None:
        self.check = load_check()
        self.changelog = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.source = (_ROOT / "scripts" / "check.py").read_text(encoding="utf-8")

    def block_above(self, constant: str) -> str:
        """The comment lines immediately above a constant — where this project keeps the
        reason for a number."""
        lines = self.source.splitlines()
        index = next(number for number, line in enumerate(lines)
                     if line.startswith(constant))
        collected = []
        while index > 0 and lines[index - 1].startswith("#"):
            index -= 1
            collected.insert(0, lines[index])
        return "\n".join(collected)

    def test_the_costs_are_plausible_numbers_of_megabytes(self):
        for name, floor in self.check._FLOOR_MB.items():
            with self.subTest(name=name):
                self.assertGreater(floor, 1024)
                self.assertLess(floor, 32 * 1024)
                self.assertGreater(self.check._MARGINAL_MB[name], 100)
                self.assertLess(self.check._MARGINAL_MB[name], floor)

    def test_every_reading_this_machine_can_produce_has_both_numbers(self):
        """A reading with no price would be a KeyError inside the gate, at the one
        moment the gate is what everything else is waiting for."""
        for platform in ("win32", "linux", "darwin"):
            with self.subTest(platform=platform), \
                    mock.patch.object(self.check.sys, "platform", platform):
                name = self.check.memory_headroom().name
                self.assertIn(name, self.check._FLOOR_MB)
                self.assertIn(name, self.check._MARGINAL_MB)

    def test_the_changelog_carries_the_numbers_the_run_prints(self):
        for number in (self.check._FLOOR_MB[_COMMIT], self.check._MARGINAL_MB[_COMMIT],
                       self.check._RESERVE_MB):
            printed = self.check._gb(number)
            self.assertTrue(printed in self.changelog,
                            f"{printed} is in the line the gate prints and nowhere in "
                            f"the CHANGELOG")

    def test_the_changelog_says_under_what_conditions_they_were_taken(self):
        found = self.changelog.index(self.check._gb(self.check._FLOOR_MB[_COMMIT]))
        entry = self.changelog[max(0, found - 6000):found + 6000].lower()
        for condition in ("2026-08-24", "gpu", "cpu", "cores"):
            self.assertTrue(condition in entry,
                            f"a measurement that does not say {condition!r} is an estimate")

    def test_the_measured_numbers_say_when_they_were_measured(self):
        """A number without a date is a number nobody can decide is stale — which is how
        `_MAX_WORKERS = 8` came to be trusted for a year."""
        self.assertRegex(self.block_above("_FLOOR_MB = "), r"20\d\d-\d\d-\d\d")

    def test_the_reasoned_numbers_admit_that_they_are_reasoned(self):
        """`_RESERVE_MB` and `_MIN_WORKERS` are not measurements, and the day that stops
        being written down is the day they start being quoted as ones."""
        self.assertIn("reasoned, not measured", self.block_above("_RESERVE_MB = "))
        self.assertIn("OBSERVATION", self.block_above("_MIN_WORKERS = "))

    def test_the_docstring_shows_the_line_the_run_prints(self):
        shown = re.search(r"^\s*workers: \d+ of \d+ \(\d+ cores\).*$",
                          self.check.__doc__ or "", re.MULTILINE)
        self.assertIsNotNone(shown, "the docstring should show what the run prints")

    def test_the_docstring_no_longer_promises_a_fixed_number_of_workers(self):
        self.assertNotIn("-n <cores, at most 8>", self.check.__doc__ or "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
