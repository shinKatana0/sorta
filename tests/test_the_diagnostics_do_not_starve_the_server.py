"""F246: the start-up diagnostics do not take the process away from the server.

The owner met this in a clean VM on 2026-08-09: `TypeError: Failed to fetch` in the
browser — first on one button, then on everything — with **not one ERROR, Traceback or
Exception in the log**. The server had not died. What the log did carry:

    23:04:17.874  startup ready elapsed=8.314
    23:04:19.670  startup step=environment elapsed=2.008
    23:04:47      the line with the address, THIRTY SECONDS after "ready"
    23:05:40      the last request that was served
                  ...fourteen minutes of nothing, and no `step=gpu` line at all...
    23:19:36      the owner restarted the program
    23:20:53      startup step=gpu elapsed=72.818

F227 moved the probes behind the bind so the page would come up early, and it did. But
the probes stayed in the SAME PROCESS: `warn_if_gpu_mismatch` -> `gpu_health` ->
`import torch`, on a thread of a program that was already serving. An import holds the
interpreter, so the tab that had just been shown the program could not fetch anything
for as long as it took — 72.8 s on the second launch of that machine, more than fifteen
minutes on the first.

What is pinned here, in the order of the acceptance criteria:

1. after a full start-up, neither heavy stack is in `sys.modules` of the server process —
   asked of an OBSERVED import and not of a list of calls, so a probe added by a
   neighbouring feature is caught by the same test (F239);
2. requests are served while the probe is running, on a slow probe too;
3. a probe that does not answer ends, and says so;
4. the log distinguishes a step that is RUNNING from a step that never happened;
5. the GPU warning still arrives, by the new road;
6. `gpu_health()` — what `sorta doctor` asks — is untouched and still answers here;
7. the watchdog goes red when the import comes back (`TestTheWatchdogGoesRed`).
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import json
import subprocess
import sys
import threading
import types
import unittest
from typing import Any, Callable, Iterator
from unittest import mock

import pytest

from sorta import diagnostics, launch, tray, ui
from sorta.ui.common import _StartupState

from tests import waiting
from tests.test_ui import UiServerTestBase

LOGGER_NAME = "sorta.diagnostics"

# The two stacks that may never be imported by a process that has to answer requests.
_WATCHED = ("torch", "onnxruntime")

# What a child on a healthy GPU machine would print back.
_FACTS: dict[str, Any] = {
    "torch_version": "2.13.0+cpu",
    "torch_cuda_available": False,
    "torch_device_name": None,
    "ort_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


def _completed(stdout: str = "", *, returncode: int = 0,
               stderr: str = "") -> "subprocess.CompletedProcess[str]":
    return subprocess.CompletedProcess(["python"], returncode, stdout, stderr)


@contextlib.contextmanager
def children(facts: dict[str, Any] | None = None, *,
             probe: Callable[[float], "subprocess.CompletedProcess[str]"] | None = None,
             card: str = "NVIDIA GeForce RTX 5090, 581.15") -> Iterator[mock.Mock]:
    """Every child process of this feature, answered without starting one.

    Patched at `sorta.launch.run` — the single door F228 built — and not at the two
    callers: a test that patched the callers would stop noticing a third one, which is
    exactly how a probe got back into the server process in the first place.
    """
    def run(command, **kwargs):
        if tuple(command) == tuple(diagnostics._NVIDIA_SMI_CMD):
            return _completed(card + "\n")
        if probe is not None:
            return probe(kwargs.get("timeout"))
        return _completed(json.dumps(facts if facts is not None else _FACTS))

    with mock.patch.object(launch, "run", side_effect=run) as patched:
        yield patched


class _Tripwire:
    """A meta-path finder that answers for the heavy stacks and remembers being asked.

    An import is intercepted rather than counted from the outside: what has to be proven
    is that the start-up does not IMPORT torch, and on a machine without torch installed
    a plain `"torch" not in sys.modules` passes without proving anything. The module it
    hands back is empty, which the diagnostics already treat as a broken install.
    """

    def __init__(self) -> None:
        self.asked: list[str] = []

    def find_spec(self, fullname: str, path: object = None,
                  target: object = None) -> object:
        if fullname.split(".")[0] not in _WATCHED:
            return None
        self.asked.append(fullname)
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec: object) -> types.ModuleType | None:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        return None


@contextlib.contextmanager
def without_torch() -> Iterator[None]:
    """This process as a freshly launched server sees it: neither stack loaded.

    The condition has to be arranged, not assumed: the suite imports torch somewhere in
    every session, and `current_gpu_health` reads `sys.modules` to decide whether asking
    here is free. `patch.dict` puts the real modules back afterwards.
    """
    with mock.patch.dict(sys.modules):
        for name in list(sys.modules):
            if name.split(".")[0] in _WATCHED:
                del sys.modules[name]
        yield


@contextlib.contextmanager
def tripwire() -> Iterator[_Tripwire]:
    """Watch this process for an import of torch or onnxruntime.

    Both are taken out of `sys.modules` first — the import system asks no finder about a
    module that is already there — so the wire sees an import that a `sys.modules` check
    alone would miss.
    """
    wire = _Tripwire()
    with without_torch():
        sys.meta_path.insert(0, wire)
        try:
            yield wire
        finally:
            sys.meta_path.remove(wire)


def fresh_state() -> "_StartupState":
    """A launch state nobody else in this process is writing (see F227's suite)."""
    return _StartupState()


# --- 1 and 7: the start-up is watched, not listed --------------------------------------


class TestNothingHeavyReachesTheServerProcess(unittest.TestCase):
    """Criterion 1, asked of an observed import."""

    def setUp(self):
        self.state = fresh_state()
        patcher = mock.patch.object(ui.common, "_startup_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.state.expect()

    def test_the_whole_start_up_leaves_the_process_without_torch(self):
        with tripwire() as wire, children() as run, \
                self.assertLogs("sorta.tray", level="INFO"):
            tray._finish_startup()
            # Read inside the tripwire: leaving it puts the suite's own torch back.
            loaded = [name for name in _WATCHED if name in sys.modules]
        self.assertEqual(wire.asked, [],
                         "запуск снова импортирует тяжёлый стек в процесс сервера")
        self.assertEqual(loaded, [])
        # And the probe really was made: a start-up that quietly stopped asking would
        # pass the two assertions above by doing nothing.
        self.assertTrue(run.called)

    def test_every_step_of_the_launch_still_ran(self):
        with tripwire(), children(), self.assertLogs("sorta.tray", level="INFO"):
            tray._finish_startup()
        self.assertEqual([done["step"] for done in self.state.snapshot()["done"]],
                         [ui.STARTUP_ENVIRONMENT, ui.STARTUP_GPU, ui.STARTUP_GEO])

    def test_the_guard_on_its_own_asks_a_child_instead(self):
        with tripwire() as wire, children():
            diagnostics.warn_if_gpu_mismatch()
        self.assertEqual(wire.asked, [])


class TestTheWatchdogGoesRed(unittest.TestCase):
    """A check nobody has seen fail is not a check (F182, F216, F228)."""

    def test_the_import_the_feature_removed_is_caught_by_the_tripwire(self):
        """`gpu_health()` in this process is what the launch used to call. If the next
        edit puts it back into the start-up path, this is what finds it."""
        with tripwire() as wire, children():
            diagnostics.gpu_health(gpu_present=False)
        self.assertEqual(wire.asked[:1], ["torch"])
        self.assertIn("onnxruntime", wire.asked)

    def test_the_tripwire_sees_past_a_module_the_suite_already_imported(self):
        """The failure mode of the check itself: torch imported by an earlier test would
        make every later assertion about imports pass without asking anything."""
        with mock.patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
            with tripwire() as wire:
                self.assertFalse("torch" in sys.modules)
                importlib.util.find_spec("torch")
                self.assertEqual(wire.asked, ["torch"])
            self.assertIsNotNone(sys.modules["torch"])


# --- 2: the server answers while the probe runs ----------------------------------------


class TestTheServerAnswersWhileTheProbeRuns(UiServerTestBase):
    """Criterion 2 — the promise F227 made and this feature had to make true."""

    def setUp(self):
        super().setUp()
        self.state = fresh_state()
        patcher = mock.patch.object(ui.common, "_startup_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.start_server()

    def test_a_request_is_served_while_the_gpu_step_is_still_going(self):
        entered = threading.Event()
        release = threading.Event()

        def slow_probe(timeout: float | None) -> "subprocess.CompletedProcess[str]":
            entered.set()
            release.wait(waiting.timeout_s())
            return _completed(json.dumps(_FACTS))

        self.state.expect()
        with without_torch(), children(probe=slow_probe), \
                self.assertLogs("sorta.tray", level="INFO"):
            worker = threading.Thread(target=tray._finish_startup, daemon=True)
            worker.start()
            try:
                self.assertTrue(entered.wait(waiting.timeout_s()), "проба не началась")
                answer = waiting.fetch(f"{self.base_url}/api/startup")
                self.assertEqual(answer.status, 200)
                # Served, and able to say what is holding things up — which is the other
                # half of the same defect: a step in flight used to be invisible.
                self.assertEqual(answer.json()["step"], ui.STARTUP_GPU)
                self.assertTrue(answer.json()["ready"])
            finally:
                release.set()
                worker.join(waiting.timeout_s())
            self.assertFalse(worker.is_alive())


# --- 3: the probe ends, whatever the child does ----------------------------------------


class TestAProbeThatDoesNotAnswerEnds(unittest.TestCase):
    """Criterion 3. Fifteen minutes without a result is not "slow", it is "unknown"."""

    def test_a_child_that_never_comes_back_is_a_timeout_with_a_line(self):
        def hangs(timeout: float) -> "subprocess.CompletedProcess[str]":
            raise subprocess.TimeoutExpired("python", timeout)

        with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            self.assertIsNone(diagnostics.probe_torch_facts(timeout=7.0, run=hangs))
        self.assertIn("7 s", logs.records[0].getMessage())

    def test_the_timeout_is_the_one_the_caller_asked_for(self):
        seen: list[float] = []
        with self.assertNoLogs(LOGGER_NAME, level="WARNING"):
            diagnostics.probe_torch_facts(
                timeout=11.5,
                run=lambda timeout: (seen.append(timeout),
                                     _completed(json.dumps(_FACTS)))[1])
        self.assertEqual(seen, [11.5])

    def test_the_child_is_given_a_timeout_by_default_too(self):
        """The default is a number and not None: `subprocess.run(timeout=None)` waits
        for as long as the child likes, which is the defect itself."""
        with children() as run:
            diagnostics.probe_torch_facts()
        command, kwargs = run.call_args
        self.assertEqual(command[0][:2], [sys.executable, "-c"])
        self.assertEqual(kwargs["timeout"], diagnostics.GPU_PROBE_TIMEOUT_S)
        self.assertGreater(diagnostics.GPU_PROBE_TIMEOUT_S, 0)

    def test_a_child_that_died_is_no_answer_and_a_line(self):
        with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            self.assertIsNone(diagnostics.probe_torch_facts(
                run=lambda timeout: _completed("", returncode=1, stderr="no module")))
        self.assertIn("no module", logs.records[0].getMessage())

    def test_a_child_that_could_not_be_started_is_no_answer_and_a_line(self):
        def refuses(timeout: float) -> "subprocess.CompletedProcess[str]":
            raise OSError("no interpreter")

        with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            self.assertIsNone(diagnostics.probe_torch_facts(run=refuses))
        self.assertIn("no interpreter", logs.records[0].getMessage())

    def test_an_answer_that_cannot_be_read_is_no_answer(self):
        for stdout in ("", "not json at all", '{"torch_version": "2.13.0"}', "[1, 2]"):
            with self.subTest(stdout=stdout):
                with self.assertLogs(LOGGER_NAME, level="WARNING"):
                    self.assertIsNone(diagnostics.probe_torch_facts(
                        run=lambda timeout, out=stdout: _completed(out)))

    def test_the_answer_is_the_last_line_the_child_said(self):
        """Both stacks greet the world on import, and a library that writes to stdout
        must not cost the launch its answer."""
        noisy = f"loading CUDA runtime\n\n{json.dumps(_FACTS)}\n"
        facts = diagnostics.probe_torch_facts(run=lambda timeout: _completed(noisy))
        self.assertEqual(facts, _FACTS)

    def test_a_probe_without_an_answer_warns_about_nothing(self):
        """A launch that could not ask must not invent a diagnosis — and must not raise
        at the thread it runs on either."""
        with without_torch(), \
                mock.patch.object(diagnostics, "probe_torch_facts", return_value=None):
            with self.assertNoLogs(LOGGER_NAME, level="WARNING"):
                self.assertFalse(diagnostics.warn_if_gpu_mismatch())


# --- 5 and 6: the same answer, and `sorta doctor` untouched ----------------------------


class TestTheAnswerArrivesWhole(unittest.TestCase):
    """Criterion 5: moving the question must not change what comes back."""

    def health(self, **kwargs: Any) -> diagnostics.GpuHealth | None:
        with children(**kwargs):
            return diagnostics.gpu_health_out_of_process()

    def test_the_child_answers_exactly_what_this_process_would_have(self):
        here = diagnostics.GpuHealth(
            torch_version=str(_FACTS["torch_version"]),
            torch_cuda_available=bool(_FACTS["torch_cuda_available"]),
            torch_device_name=None,
            ort_providers=tuple(_FACTS["ort_providers"]),
            gpu_present=True,
        )
        self.assertEqual(dataclasses.asdict(self.health()), dataclasses.asdict(here))

    def test_the_hardware_half_is_still_asked_by_this_process(self):
        """`nvidia-smi` needs neither stack and answers in three seconds, and one probe
        is what keeps the wizard and the launch from disagreeing about the card (F230)."""
        health = self.health(card="")
        assert health is not None
        self.assertFalse(health.gpu_present)

    def test_the_device_name_survives_the_journey(self):
        facts = {**_FACTS, "torch_cuda_available": True,
                 "torch_device_name": "NVIDIA GeForce RTX 5090"}
        health = self.health(facts=facts)
        assert health is not None
        self.assertEqual(health.torch_device_name, "NVIDIA GeForce RTX 5090")
        self.assertFalse(health.degraded)

    def test_the_warning_still_reaches_the_log_by_the_new_road(self):
        with without_torch(), children(), \
                self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            self.assertTrue(diagnostics.warn_if_gpu_mismatch())
        message = logs.records[0].getMessage()
        self.assertIn("2.13.0+cpu", message)
        self.assertIn("CUDAExecutionProvider", message)

    def test_a_machine_with_nothing_wrong_is_still_silent(self):
        facts = {**_FACTS, "torch_version": "2.13.0+cu130",
                 "torch_cuda_available": True, "torch_device_name": "RTX 5090"}
        with without_torch(), children(facts=facts), \
                self.assertNoLogs(LOGGER_NAME, level="WARNING"):
            self.assertFalse(diagnostics.warn_if_gpu_mismatch())

    def test_the_child_is_told_where_the_package_lives(self):
        """An installed copy is a `uv pip install --target` tree that a bare `python -c`
        has no reason to look into: the child would answer "torch is not installed" about
        a machine where it is."""
        with children() as run:
            diagnostics.probe_torch_facts()
        path = run.call_args[1]["env"]["PYTHONPATH"].split(";" if sys.platform ==
                                                           "win32" else ":")
        for entry in sys.path:
            if entry:
                self.assertIn(entry, path)


class TestWhoAsksInThisProcessAndWhoDoesNot(unittest.TestCase):
    """Criterion 6: `gpu_health()` is what `sorta doctor` calls, and it is unchanged."""

    def torch_here(self) -> mock._patch_dict:
        module = types.ModuleType("torch")
        module.__version__ = "2.13.0+cu130"
        module.cuda = types.SimpleNamespace(is_available=lambda: True,
                                            get_device_name=lambda index: "RTX 5090")
        return mock.patch.dict(sys.modules, {"torch": module})

    def test_a_process_that_already_has_torch_asks_it_here(self):
        """A run imports torch a line later anyway — starting a second interpreter to
        read a version off the first one's memory would be a cost with no buyer."""
        with self.torch_here(), children() as run:
            health = diagnostics.current_gpu_health()
        assert health is not None
        self.assertEqual(health.torch_version, "2.13.0+cu130")
        self.assertNotIn([sys.executable, "-c", diagnostics._PROBE_SCRIPT],
                         [call.args[0] for call in run.call_args_list])

    def test_a_process_without_torch_asks_a_child(self):
        with tripwire(), children() as run:
            health = diagnostics.current_gpu_health()
        assert health is not None
        self.assertIn([sys.executable, "-c", diagnostics._PROBE_SCRIPT],
                      [call.args[0] for call in run.call_args_list])

    def test_the_command_doctor_runs_still_answers_here(self):
        with self.torch_here(), children():
            health = diagnostics.gpu_health(install_kind="checkout")
        self.assertEqual(health.torch_version, "2.13.0+cu130")
        self.assertTrue(health.torch_cuda_available)


# --- the child, for real ---------------------------------------------------------------


# serial: it starts an interpreter that imports torch — half a gigabyte and several
# seconds — which is exactly what the parallel half of the gate must not run eight of.
@pytest.mark.serial
class TestTheRealChildAnswers(unittest.TestCase):
    """Every other case here fakes the child; one has to not (F228's rule)."""

    def test_a_real_probe_comes_back_with_a_readable_answer(self):
        facts = diagnostics.probe_torch_facts()
        assert facts is not None, "проба в подпроцессе не вернула ответ"
        self.assertEqual(set(facts), set(diagnostics._PROBE_FIELDS))
        self.assertIsInstance(facts["torch_version"], str)
        self.assertIsInstance(facts["ort_providers"], list)

    def test_the_script_the_child_runs_is_a_program_and_not_a_string(self):
        """It runs in another interpreter, so a syntax error in it would first be seen
        on somebody's machine (the rule `_SPLASH_SCRIPT` is held to)."""
        compile(diagnostics._PROBE_SCRIPT, "<gpu-probe>", "exec")
        self.assertIn("torch_facts_json", diagnostics._PROBE_SCRIPT)

    def test_what_the_child_prints_is_one_line_this_can_read(self):
        printed = diagnostics.torch_facts_json()
        self.assertEqual(printed.count("\n"), 0)
        self.assertEqual(set(json.loads(printed)), set(diagnostics._PROBE_FIELDS))


# --- 4: the log says a step is running -------------------------------------------------


class TestTheLogSaysAStepHasBegun(unittest.TestCase):
    """Criterion 4. A step that did not finish left no trace at all, which is why
    fourteen minutes of silence could not be told apart from a dead process."""

    def setUp(self):
        self.state = fresh_state()
        patcher = mock.patch.object(ui.common, "_startup_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_step_is_in_the_log_before_it_is_over(self):
        with self.assertLogs("sorta.tray", level="INFO") as logs:
            with tray._startup_step(ui.STARTUP_GPU):
                during = [record.getMessage() for record in logs.records]
        self.assertEqual(during, ["startup step=gpu started"])

    def test_the_finished_step_still_carries_its_duration(self):
        with self.assertLogs("sorta.tray", level="INFO") as logs:
            with tray._startup_step(ui.STARTUP_GPU):
                pass
        lines = [record.getMessage() for record in logs.records]
        self.assertEqual(len(lines), 2, lines)
        self.assertRegex(lines[1], r"^startup step=gpu elapsed=\d+\.\d{3}$")

    def test_the_beginning_is_not_read_back_as_a_timing(self):
        """`runlog` prices the next run off `stage=<name> elapsed=` lines in this very
        file, and a line without `elapsed=` cannot be mistaken for one."""
        import logging

        from sorta import runlog

        record = logging.LogRecord("sorta.tray", logging.INFO, __file__, 0,
                                   tray._STARTUP_BEGIN_LINE, (ui.STARTUP_GPU,), None)
        line = logging.Formatter(runlog._FORMAT, runlog._DATEFMT).format(record)
        self.assertIn("startup step=gpu started", line)
        self.assertIsNone(runlog._MEASUREMENT_RE.match(line))


if __name__ == "__main__":
    unittest.main()
