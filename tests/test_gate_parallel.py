"""F219: the gate uses the machine it runs on, and stays a gate while it does.

Making the suite parallel is one line (`-n auto`); making it parallel WITHOUT quietly
becoming a weaker gate is what this module holds in place. Three things can go wrong,
and each has a test here:

* **Coverage measured on half the run.** Two pytest invocations produce two coverage
  reports. If each judged the threshold, the serial half — a dozen tests — would fail
  it, and the parallel half would pass it on incomplete data. The gate appends both
  into one data file and checks the threshold once, at the end; the test below proves
  the combined report is byte-identical to the report of a single one-process run.
* **A marker used as a place to hide a flake.** `serial` is for a test that asserts
  about TIME or binds a port — a loaded machine is exactly the condition under which
  those fail. It is not for "this went red once". So every use carries a reason next
  to it, and a marker without one fails the guard here.
* **A verdict that depends on how many processes ran it.** A test that passes at
  `-n auto` but not at `-n 0` (or the other way round) is a defect, not a setting.

The first two tests build a throwaway package in a temp directory and run the real
`pytest`/`coverage` over it. That is deliberate: the property is about what those two
tools do with `--cov-append` across processes, and no amount of reading `check.py` can
answer it.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

_ROOT = Path(__file__).resolve().parent.parent
_CHECK_PY = _ROOT / "scripts" / "check.py"

# The mini project's floor. Chosen to sit BETWEEN what either half covers on its own
# and what the two cover together — that is the whole trap this feature has to avoid,
# so the fixture has to be able to fall into it.
_FAIL_UNDER = 80

_WIDE = '''\
"""Covered by the parallel half only."""


def classify(number):
    if number < 0:
        return "negative"
    if number == 0:
        return "zero"
    if number % 2:
        return "odd"
    return "even"


def summarize(numbers):
    seen = []
    for number in numbers:
        seen.append(classify(number))
    return ", ".join(seen)


def never_called():
    value = "no test reaches this"
    return value.upper()
'''

_NARROW = '''\
"""Covered by the serial half only."""


def shout(text):
    if not text:
        return ""
    return text.upper()
'''

_PARALLEL_TESTS = '''\
from probe import wide


def test_classify():
    assert wide.classify(-1) == "negative"
    assert wide.classify(0) == "zero"
    assert wide.classify(1) == "odd"
    assert wide.classify(2) == "even"


def test_summarize():
    assert wide.summarize([0, 1]) == "zero, odd"
'''

_SERIAL_TESTS = '''\
import pytest

from probe import narrow


@pytest.mark.serial
def test_shout():
    assert narrow.shout("") == ""
    assert narrow.shout("hi") == "HI"
'''

_MINI_PYPROJECT = f"""\
[tool.pytest.ini_options]
markers = ["serial: the single-process pass"]

[tool.coverage.report]
fail_under = {_FAIL_UNDER}
"""


def load_check() -> Any:
    """`scripts/check.py` as a module — it is a script, not an importable package."""
    spec = importlib.util.spec_from_file_location("sorta_check_gate", _CHECK_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child_env() -> dict[str, str]:
    """The environment for a nested pytest run, with the outer run's coverage removed.

    pytest-cov hands its subprocesses `COV_CORE_*` so that they measure themselves into
    the OUTER data file. Here that would mix the gate's own coverage into the mini
    project's numbers, and the comparison this module makes would be between two
    contaminated reports instead of two clean ones.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("COV_CORE")}
    env.pop("COVERAGE_FILE", None)
    env.pop("PYTEST_ADDOPTS", None)
    return env


def total_percent(report: str) -> int:
    match = re.search(r"^TOTAL.*?(\d+)%", report, re.MULTILINE)
    assert match, f"no TOTAL line in:\n{report}"
    return int(match.group(1))


class MiniProject:
    """A throwaway package plus a suite split the way the real gate splits ours."""

    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "probe").mkdir()
        (root / "probe" / "__init__.py").write_text("", encoding="utf-8")
        (root / "probe" / "wide.py").write_text(_WIDE, encoding="utf-8")
        (root / "probe" / "narrow.py").write_text(_NARROW, encoding="utf-8")
        (root / "test_wide.py").write_text(_PARALLEL_TESTS, encoding="utf-8")
        (root / "test_narrow.py").write_text(_SERIAL_TESTS, encoding="utf-8")
        (root / "pyproject.toml").write_text(_MINI_PYPROJECT, encoding="utf-8")

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", args[0], *args[1:]],
            cwd=self.root, capture_output=True, text=True, env=child_env(),
        )

    def pytest(self, *args: str) -> subprocess.CompletedProcess[str]:
        return self.run("pytest", "-p", "no:cacheprovider", "-q",
                        "--cov=probe", "--cov-append", "--cov-report=", *args)

    def erase(self) -> None:
        self.run("coverage", "erase")

    def report(self, *args: str) -> subprocess.CompletedProcess[str]:
        return self.run("coverage", "report", "--show-missing", *args)


class TestCoverageIsMeasuredOnTheSum(unittest.TestCase):
    """Two passes, one number — and it is the number one pass would have produced."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.project = MiniProject(Path(self._temp.name))

    def one_process(self) -> str:
        """Everything in a single pytest, the way the gate ran before F219."""
        self.project.erase()
        run = self.project.pytest("--cov-fail-under=0")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        return self.project.report("--cov-fail-under=0").stdout

    def two_passes(self) -> subprocess.CompletedProcess[str]:
        """The gate's own sequence: erase, parallel half, serial half, one verdict."""
        self.project.erase()
        parallel = self.project.pytest("-n", "2", "-m", "not serial", "--cov-fail-under=0")
        self.assertEqual(parallel.returncode, 0, parallel.stdout + parallel.stderr)
        serial = self.project.pytest("-m", "serial", "--cov-fail-under=0")
        self.assertEqual(serial.returncode, 0, serial.stdout + serial.stderr)
        return self.project.report()

    def test_the_combined_report_equals_a_single_serial_run(self):
        combined = self.two_passes().stdout
        self.assertEqual(
            [line for line in combined.splitlines() if not line.startswith("Required")],
            self.one_process().splitlines(),
            "coverage over the sum of the two passes must be neither more nor less "
            "than coverage of one process running the same tests",
        )

    def test_the_threshold_is_judged_once_and_it_passes(self):
        verdict = self.two_passes()
        self.assertEqual(verdict.returncode, 0, verdict.stdout + verdict.stderr)
        self.assertGreaterEqual(total_percent(verdict.stdout), _FAIL_UNDER)

    def test_judging_each_pass_on_its_own_would_have_failed(self):
        """The trap, made visible: neither half clears the floor by itself."""
        self.project.erase()
        parallel = self.project.pytest("-n", "2", "-m", "not serial")
        self.assertNotEqual(parallel.returncode, 0,
                            "the parallel half alone must NOT be able to clear the "
                            "threshold — if it can, this fixture no longer models the "
                            "situation the gate has to survive")
        self.project.erase()
        serial = self.project.pytest("-m", "serial")
        self.assertNotEqual(serial.returncode, 0, serial.stdout + serial.stderr)

    def test_the_two_halves_partition_the_suite(self):
        """`not serial` + `serial` = everything; a test in neither half is a test lost."""
        self.project.erase()
        both = self.project.pytest("--cov-fail-under=0", "--collect-only").stdout
        parallel = self.project.pytest("-m", "not serial", "--cov-fail-under=0",
                                       "--collect-only").stdout
        serial = self.project.pytest("-m", "serial", "--cov-fail-under=0",
                                     "--collect-only").stdout
        collected = re.compile(r"^(test_\w+\.py::\S+)$", re.MULTILINE)
        self.assertEqual(
            sorted(collected.findall(parallel) + collected.findall(serial)),
            sorted(collected.findall(both)),
        )


class TestTheGateChecksTheThresholdOnce(unittest.TestCase):
    """Read off `scripts/check.py` itself: the shape the tests above rely on."""

    def setUp(self) -> None:
        self.check = load_check()

    def pytest_steps(self) -> list[list[str]]:
        return [cmd for _, cmd in self.check.SLOW_CHECKS if "pytest" in cmd]

    def test_both_passes_append_instead_of_overwriting(self):
        for cmd in self.pytest_steps():
            self.assertIn("--cov-append", cmd)
            self.assertIn("--cov=sorta", cmd)

    def test_neither_pass_pronounces_a_verdict_on_coverage(self):
        for cmd in self.pytest_steps():
            self.assertIn("--cov-fail-under=0", cmd)

    def test_the_verdict_is_the_last_step_and_there_is_one_of_it(self):
        judges = [name for name, cmd in self.check.SLOW_CHECKS
                  if "coverage" in cmd and "report" in cmd]
        self.assertEqual(len(judges), 1, self.check.SLOW_CHECKS)
        self.assertEqual(judges[0], self.check.SLOW_CHECKS[-1][0])
        # No --cov-fail-under here: the floor comes from pyproject.toml, and this
        # feature is not allowed to move it.
        self.assertNotIn("--cov-fail-under", " ".join(self.check.SLOW_CHECKS[-1][1]))

    def test_the_data_is_erased_before_the_first_appending_pass(self):
        self.assertEqual(self.check.ERASE_COVERAGE[1:], ["-m", "coverage", "erase"])

    def test_one_pass_is_parallel_and_the_other_is_not(self):
        parallel, serial = self.pytest_steps()
        self.assertEqual(parallel[parallel.index("-n") + 1], "auto")
        self.assertEqual(parallel[parallel.index("-m") + 1], "not serial")
        self.assertNotIn("-n", serial)
        self.assertEqual(serial[serial.index("-m") + 1], "serial")

    def test_the_run_reports_how_long_it_took(self):
        """The number in the docstring was wrong by threefold because nobody saw it."""
        self.assertEqual(self.check._duration(0), "0s")
        self.assertEqual(self.check._duration(59.9), "59s")
        self.assertEqual(self.check._duration(28 * 60 + 5), "28m 05s")


class TestTheFastGateStillDoesNotRunTheSuite(unittest.TestCase):
    """The contract of `--fast` is unchanged by F219: seconds, and no tests."""

    def setUp(self) -> None:
        self.check = load_check()

    def invoke(self, *argv: str) -> list[list[str]]:
        calls: list[list[str]] = []

        def record(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        with mock.patch.object(self.check.subprocess, "run", side_effect=record), \
                mock.patch.object(sys, "argv", ["check.py", *argv]):
            self.assertEqual(self.check.main(), 0)
        return calls

    def test_fast_runs_the_fast_checks_and_nothing_else(self):
        calls = self.invoke("--fast")
        self.assertEqual(calls, [cmd for _, cmd in self.check.FAST_CHECKS])

    def test_fast_starts_neither_pytest_nor_coverage(self):
        for cmd in self.invoke("--fast"):
            self.assertNotIn("pytest", cmd)
            self.assertNotIn("coverage", cmd)

    def test_the_full_gate_erases_first_then_runs_both_halves(self):
        calls = self.invoke()
        self.assertEqual(calls[0], self.check.ERASE_COVERAGE)
        self.assertEqual(calls[1:], [cmd for _, cmd in
                                     self.check.FAST_CHECKS + self.check.SLOW_CHECKS])

    def test_an_empty_serial_half_is_not_a_failed_gate(self):
        """Exit code 5 is "nothing collected" — a real answer for a half that may be
        empty, and a confusing red gate if it were treated like a failing test."""
        def refuse(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
            empty = "-m" in cmd and cmd[cmd.index("-m") + 1] == "serial"
            return subprocess.CompletedProcess(cmd, 5 if empty else 0)

        with mock.patch.object(self.check.subprocess, "run", side_effect=refuse):
            self.assertEqual(self.check.run(self.check.SLOW_CHECKS), 0)

    def test_a_failing_parallel_half_still_stops_the_gate(self):
        def refuse(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
            failed = "-m" in cmd and cmd[cmd.index("-m") + 1] == "not serial"
            return subprocess.CompletedProcess(cmd, 1 if failed else 0)

        with mock.patch.object(self.check.subprocess, "run", side_effect=refuse):
            self.assertEqual(self.check.run(self.check.SLOW_CHECKS), 1)


class TestEverySerialMarkerCarriesAReason(unittest.TestCase):
    """`serial` costs the suite wall-clock, so it has to say what it bought.

    Without this, the marker becomes the place flakes go to be forgotten: in a month
    nobody can tell "asserts about elapsed time" from "went red once in July".
    """

    # A comment right above the marker, or trailing on the same line. Long enough that
    # `# serial` does not count as an explanation.
    MINIMUM_REASON = 20

    def reason_for(self, lines: list[str], index: int) -> str:
        trailing = lines[index].split("#", 1)
        if len(trailing) == 2 and "pytest.mark.serial" in trailing[0]:
            return trailing[1].strip()
        collected = []
        cursor = index - 1
        while cursor >= 0 and lines[cursor].lstrip().startswith("#"):
            collected.insert(0, lines[cursor].lstrip().lstrip("#").strip())
            cursor -= 1
        return " ".join(collected).strip()

    def marked_lines(self, source: str) -> list[int]:
        return [number for number, line in enumerate(source.splitlines())
                if "pytest.mark.serial" in line and not line.lstrip().startswith("#")]

    def test_the_guard_catches_a_marker_with_no_reason(self):
        bare = "@pytest.mark.serial\ndef test_thing():\n    pass\n"
        lines = bare.splitlines()
        self.assertEqual(self.marked_lines(bare), [0])
        self.assertEqual(self.reason_for(lines, 0), "")

    def test_the_guard_accepts_a_reason_above_and_beside(self):
        above = "# measures elapsed wall-clock time\n@pytest.mark.serial\ndef t(): pass\n"
        self.assertGreater(len(self.reason_for(above.splitlines(), 1)), self.MINIMUM_REASON)
        beside = "@pytest.mark.serial  # binds a port, two workers collide on it\n"
        self.assertGreater(len(self.reason_for(beside.splitlines(), 0)), self.MINIMUM_REASON)

    def test_every_marker_in_the_suite_says_why(self):
        marked = 0
        for path in sorted((_ROOT / "tests").rglob("test_*.py")):
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()
            for number in self.marked_lines(source):
                marked += 1
                reason = self.reason_for(lines, number)
                self.assertGreater(
                    len(reason), self.MINIMUM_REASON,
                    f"{path.name}:{number + 1} is marked `serial` with no reason next "
                    f"to it. Say what the parallel half does to it — asserts about "
                    f"elapsed time, binds a port — or fix the test instead.",
                )
        self.assertGreater(marked, 0, "the serial half exists; the markers should too")

    def test_the_marker_is_declared_and_typos_are_errors(self):
        config = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("[tool.pytest.ini_options]", config)
        self.assertIn("serial:", config)
        # Without --strict-markers a mistyped `@pytest.mark.seria1` is a warning, and
        # the test silently moves into the parallel half it was taken out of.
        self.assertIn("--strict-markers", config)


class TestTheVerdictDoesNotDependOnTheWorkerCount(unittest.TestCase):
    """A subset of the real suite, run both ways. Not the whole one — that is 28
    minutes twice, and the property is about the mechanism, not about volume."""

    SUBSET = ["tests/test_dates.py", "tests/test_config.py", "tests/test_progress.py"]

    def run_subset(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
             "--no-cov", *self.SUBSET, *args],
            cwd=_ROOT, capture_output=True, text=True, env=child_env(),
        )

    def test_one_process_and_many_agree(self):
        single = self.run_subset()
        many = self.run_subset("-n", "auto")
        self.assertEqual(single.returncode, many.returncode,
                         f"-n 0:\n{single.stdout}\n-n auto:\n{many.stdout}")
        self.assertEqual(single.returncode, 0, single.stdout + single.stderr)
        counts = re.compile(r"(\d+) passed")
        self.assertEqual(counts.findall(single.stdout), counts.findall(many.stdout))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
