"""F251: the worker count is measured on the machine, and said out loud.

`_MAX_WORKERS = 8` was measured once, on one machine, on one day, and was then trusted
forever. Over 2026-08-23/24 seven gate runs out of twelve died — `MemoryError`,
`RemoteDisconnected`, `ConnectionAbortedError` — every time on different tests, on a tree
that was green before and after. The machine was at 59 GB of a 74.6 GB COMMIT limit
before the gate started, with 17.6 GB of physical memory free: enough memory, no
allowance left to reserve it with.

What this module holds in place:

* memory may only LOWER the count. An idle 4-core runner must still get 4, or the
  feature has paid for one laptop with everybody's CI.
* a machine too tight for even one worker is TOLD SO instead of being made to spend
  eight minutes arriving at a predictable `MemoryError`.
* a platform that cannot be measured behaves exactly as it did before this existed, and
  the line says that this is what happened.
* the line is printed on every run that will start pytest — including the ones that got
  the full count. Reading a red gate should not require measuring the machine by hand
  afterwards, and that only works if the green runs carry the number too.
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

# Enough for the ceiling at any budget this feature would accept. Not `sys.maxsize`: the
# line prints the number, and a machine with 8 exabytes free is not a readable fixture.
_PLENTY_MB = 512 * 1024


class TestTheCountFollowsTheMeasurement(unittest.TestCase):
    """The formula: min(cores, ceiling, headroom / budget), and never more than that."""

    def setUp(self) -> None:
        self.check = load_check()
        self.budget = self.check._BUDGET.megabytes

    def plan(self, cores: int, megabytes: int | None, name: str = "commit headroom") -> Any:
        return self.check.plan_workers(cores, self.check.Headroom(megabytes, name))

    def test_plenty_of_headroom_is_the_count_this_gate_always_had(self):
        for cores, expected in [(24, 8), (16, 8), (8, 8), (4, 4), (2, 2), (1, 1)]:
            with self.subTest(cores=cores):
                self.assertEqual(self.plan(cores, _PLENTY_MB).workers, expected)

    def test_a_small_machine_is_never_given_more_workers_than_it_has_cores(self):
        """Memory has one direction. A 2-core box with a terabyte free still gets 2."""
        self.assertEqual(self.plan(2, _PLENTY_MB).workers, 2)

    def test_a_tight_machine_gets_fewer_workers(self):
        self.assertEqual(self.plan(24, self.budget * 5).workers, 5)
        self.assertEqual(self.plan(24, self.budget * 3 + self.budget // 2).workers, 3)

    def test_the_ceiling_still_caps_a_machine_with_room_to_spare(self):
        self.assertEqual(self.plan(24, self.budget * 40).workers, self.check._MAX_WORKERS)

    def test_no_headroom_at_all_is_zero_workers_and_not_one(self):
        """Clamping to 1 here would be the eight-minute wait this feature removes."""
        self.assertEqual(self.plan(24, self.budget - 1).workers, 0)
        self.assertEqual(self.plan(24, 0).workers, 0)

    def test_a_machine_that_cannot_be_measured_behaves_as_before(self):
        self.assertEqual(self.plan(24, None).workers, 8)
        self.assertEqual(self.plan(4, None).workers, 4)

    def test_zero_cores_still_leaves_a_worker(self):
        """`os.cpu_count()` may answer None; the gate must not then ask for -n 0."""
        self.assertEqual(self.plan(1, _PLENTY_MB).workers, 1)
        self.assertEqual(self.plan(0, _PLENTY_MB).workers, 1)


class TestCiIsNotMadeSlowerByThis(unittest.TestCase):
    """The trade this feature is not allowed to make.

    The runners are 2-4 cores and nothing else runs on them; on such a machine the
    answer has to be the cores, byte-for-byte what `min(cores, 8)` returned before. The
    budget used here is the CPU one on purpose — `.github/workflows/check.yml` installs
    `--extra cpu`, so charging the runner gpu prices would be a bill for a card that is
    not in it.
    """

    def setUp(self) -> None:
        self.check = load_check()
        self.budget = self.check.Budget(self.check._BUDGET_MB["cpu"], "cpu")

    def test_an_idle_runner_gets_every_core_on_both_platforms(self):
        for cores in (2, 4):
            for name in ("commit headroom", "available memory"):
                with self.subTest(cores=cores, name=name):
                    plan = self.check.plan_workers(
                        cores, self.check.Headroom(_PLENTY_MB, name), self.budget)
                    self.assertEqual(plan.workers, min(cores, self.check._MAX_WORKERS))

    # A GitHub-hosted runner — the machine `.github/workflows/check.yml` uses on both
    # platforms — is 4 cores and 16 GB, and a fresh one has ~12 GB of either reading
    # left. Four workers have to be affordable at that figure, which is a constraint on
    # the BUDGET and the reason this test names a number instead of a plenty.
    RUNNER_CORES, RUNNER_HEADROOM_MB = 4, 12 * 1024

    def test_a_github_runner_still_gets_all_four_of_its_cores(self):
        for name in ("commit headroom", "available memory"):
            with self.subTest(name=name):
                plan = self.check.plan_workers(
                    self.RUNNER_CORES, self.check.Headroom(self.RUNNER_HEADROOM_MB, name),
                    self.budget)
                self.assertEqual(plan.workers, self.RUNNER_CORES)


class TestTheLineSaysWhatWasChosenAndWhy(unittest.TestCase):
    """Half the value of the feature: the number, on every run, before the wait."""

    def setUp(self) -> None:
        self.check = load_check()
        self.budget = self.check._BUDGET.megabytes

    def line(self, cores: int, megabytes: int | None, name: str = "commit headroom") -> str:
        return self.check.plan_workers(cores, self.check.Headroom(megabytes, name)).line

    def test_the_full_count_is_reported_as_loudly_as_a_lowered_one(self):
        full = self.line(24, _PLENTY_MB)
        self.assertRegex(full, r"^workers: 8 of 8 \(24 cores\)")
        self.assertIn("budgeted per", full)
        self.assertIn(self.check._BUDGET.profile, full)

    def test_a_lowered_count_names_the_headroom_and_the_budget(self):
        line = self.line(24, self.budget * 5)
        self.assertRegex(line, r"^workers: 5 of 8 \(24 cores\)")
        self.assertIn(self.check._gb(self.budget * 5), line)
        self.assertIn(self.check._gb(self.budget), line)

    def test_the_line_names_the_reading_it_used_and_platforms_differ(self):
        self.assertIn("commit headroom", self.line(24, self.budget * 5))
        self.assertIn("available memory", self.line(24, self.budget * 5, "available memory"))

    def test_a_failed_measurement_is_said_and_not_hidden(self):
        line = self.line(24, None)
        self.assertRegex(line, r"^workers: 8 of 8 \(24 cores\)")
        self.assertIn("could not be measured", line)

    def test_the_refusal_names_both_numbers_and_what_frees_them(self):
        headroom = self.check.Headroom(self.budget // 2, "commit headroom")
        message = self.check.too_tight(self.check.plan_workers(24, headroom))
        self.assertIn(self.check._gb(headroom.megabytes), message)
        self.assertIn(self.check._gb(self.budget), message)
        # Naming a number without naming what moves it is a message nobody can act on.
        self.assertIn("worktree", message)

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
    """What `main` does with the plan: prints it, and refuses on an impossible one."""

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
        return self.check.plan_workers(cores, self.check.Headroom(megabytes,
                                                                 "commit headroom"))

    def test_the_slow_half_prints_the_line_before_it_starts(self):
        plan = self.plan_for(_PLENTY_MB)
        code, printed, calls = self.invoke(plan, "--slow")
        self.assertEqual(code, 0)
        self.assertIn(plan.line, printed)
        self.assertLess(printed.index(plan.line), printed.index("pytest"),
                        "the point of the line is to be there BEFORE the wait")
        self.assertTrue(calls)

    def test_the_full_gate_prints_it_too(self):
        plan = self.plan_for(_PLENTY_MB)
        _code, printed, _calls = self.invoke(plan)
        self.assertIn(plan.line, printed)

    def test_the_fast_gate_says_nothing_about_workers_because_it_starts_none(self):
        _code, printed, _calls = self.invoke(self.plan_for(_PLENTY_MB), "--fast")
        self.assertNotIn("workers:", printed)

    def test_a_machine_with_no_headroom_stops_the_run_before_anything_starts(self):
        code, printed, calls = self.invoke(self.plan_for(0), "--slow")
        self.assertEqual(code, self.check._TOO_TIGHT)
        self.assertNotEqual(code, 0, "nothing was checked, so nothing may read as green")
        self.assertEqual(calls, [], "not even the coverage erase: the run did not start")
        self.assertIn("NOT STARTING", printed)

    def test_the_refusal_reaches_the_full_gate_as_well(self):
        code, _printed, calls = self.invoke(self.plan_for(0))
        self.assertEqual(code, self.check._TOO_TIGHT)
        self.assertEqual(calls, [])

    def test_a_tight_machine_that_can_still_afford_a_worker_runs(self):
        plan = self.plan_for(self.check._BUDGET.megabytes)
        self.assertEqual(plan.workers, 1)
        code, printed, calls = self.invoke(plan, "--slow")
        self.assertEqual(code, 0)
        self.assertIn(plan.line, printed)
        self.assertTrue(calls)

    def test_the_refusal_does_not_reach_the_fast_gate(self):
        """`--fast` is lint and types in one process: it neither needs the headroom nor
        should be blocked by a machine that has none."""
        code, printed, calls = self.invoke(self.plan_for(0), "--fast")
        self.assertEqual(code, 0)
        self.assertNotIn("NOT STARTING", printed)
        self.assertEqual(calls, [cmd for _, cmd in self.check.FAST_CHECKS])


class TestTheCommandCarriesTheChosenCount(unittest.TestCase):
    """The plan is worthless if the pytest invocation does not follow it."""

    def setUp(self) -> None:
        self.check = load_check()

    def test_the_parallel_half_runs_at_the_planned_count(self):
        args = pytest_args(self.check.SLOW_CHECKS[0][1])
        self.assertEqual(args[args.index("-n") + 1], str(max(self.check._PLAN.workers, 1)))

    def test_the_count_is_a_number_and_never_zero_or_auto(self):
        args = pytest_args(self.check.SLOW_CHECKS[0][1])
        workers = int(args[args.index("-n") + 1])
        self.assertGreaterEqual(workers, 1)
        self.assertLessEqual(workers, self.check._MAX_WORKERS)
        self.assertLessEqual(workers, max(os.cpu_count() or 1, 1))

    def test_the_ceiling_is_still_eight(self):
        """Out of scope for this feature — a change here is somebody else's decision."""
        self.assertEqual(self.check._MAX_WORKERS, 8)


class TestTheBudgetIsAMeasurementAndSaysSo(unittest.TestCase):
    """A budget that is a guess is the same as the old eight, only newer.

    So the number in `check.py` has to be findable in the CHANGELOG with the conditions
    it was taken under. This test is what makes re-measuring it a two-file edit.
    """

    def setUp(self) -> None:
        self.check = load_check()
        self.changelog = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    def test_both_budgets_are_plausible_numbers_of_megabytes(self):
        for profile, budget in self.check._BUDGET_MB.items():
            with self.subTest(profile=profile):
                self.assertGreater(budget, 200)
                self.assertLess(budget, 16 * 1024)

    def test_a_gpu_worker_is_charged_more_than_a_cpu_one(self):
        """If these ever converge, the profile split has stopped buying anything and the
        simpler thing to do is delete it rather than keep two numbers in step."""
        self.assertGreater(self.check._BUDGET_MB["gpu"], self.check._BUDGET_MB["cpu"])

    def test_the_profile_is_read_from_the_metadata_and_not_from_an_import(self):
        for reported, expected in [("2.13.0+cu130", "gpu"), ("2.13.0+cpu", "cpu"),
                                   ("2.13.0", "cpu")]:
            with self.subTest(reported=reported):
                with mock.patch("importlib.metadata.version", return_value=reported):
                    self.assertEqual(self.check.install_profile(), expected)

    def test_an_unreadable_profile_is_charged_the_expensive_price(self):
        """Guessing cheap on a machine that pays gpu prices is the MemoryError again."""
        with mock.patch("importlib.metadata.version", side_effect=Exception("no torch")):
            self.assertEqual(self.check.install_profile(), "gpu")

    def test_this_venv_is_recognised_as_one_of_the_two(self):
        self.assertIn(self.check.install_profile(), self.check._BUDGET_MB)

    def budget_mentions(self) -> list[str]:
        """Every `<budget> MB` in the CHANGELOG. `assertIn` is avoided on purpose here:
        it prints the container it searched, and that container is the CHANGELOG."""
        return [f"{budget} MB" for budget in self.check._BUDGET_MB.values()
                if f"{budget} MB" in self.changelog]

    def test_the_changelog_carries_the_numbers_check_py_uses(self):
        self.assertEqual(len(self.budget_mentions()), len(self.check._BUDGET_MB),
                         "a budget in check.py that the CHANGELOG does not report")

    def test_the_changelog_says_under_what_conditions_they_were_taken(self):
        found = self.changelog.index(self.budget_mentions()[0])
        entry = self.changelog[max(0, found - 6000):found + 6000].lower()
        for condition in ("2026-08-24", "gpu", "cpu", "cores"):
            self.assertTrue(condition in entry,
                            f"a measurement that does not say {condition!r} is an estimate")

    def test_the_source_says_when_it_was_measured(self):
        """A number without a date is a number nobody can decide is stale — which is how
        `_MAX_WORKERS = 8` came to be trusted for a year."""
        source = (_ROOT / "scripts" / "check.py").read_text(encoding="utf-8")
        found = source.index("_BUDGET_MB = ")
        self.assertRegex(source[max(0, found - 1200):found], r"20\d\d-\d\d-\d\d")


class TestTheDocstringStillDescribesTheGate(unittest.TestCase):
    """It has been wrong by a factor of three once already, about the duration."""

    def setUp(self) -> None:
        self.check = load_check()
        self.doc = self.check.__doc__ or ""

    def test_it_says_the_count_is_measured_and_eight_is_the_ceiling(self):
        self.assertIn("workers = min(cores, 8, memory headroom", self.doc)
        self.assertRegex(self.doc, r"(?i)ceiling")

    def test_it_no_longer_promises_a_fixed_number_of_workers(self):
        self.assertNotIn("-n <cores, at most 8>", self.doc)

    def test_it_shows_the_line_the_run_prints(self):
        shown = re.search(r"^\s*workers: \d+ of \d+ \(\d+ cores\).*$", self.doc, re.MULTILINE)
        self.assertIsNotNone(shown, "the docstring should show what the run prints")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
