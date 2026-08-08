"""F227: the first launch says it is launching — and a second click costs a TCP connect.

Two defects that fed each other, both reported off a clean VM on 2026-08-08. Measured
with the interpreter from the installer payload, on a FAST machine:

    import sorta.tray        1.53 s
    warn_if_gpu_mismatch     3.76 s      the torch import
    config + db connect      0.16 s
    ui.build_server          0.20 s
    total to a bound port    5.65 s

On a VM with a slow disk that is tens of seconds of nothing on screen — no console (the
shortcut runs `pythonw`), no icon yet, no tab — so the person clicks again. And the second
click was the expensive one: the "are we already running" question stood inside `start()`,
after the config had been read, `warn_if_gpu_mismatch()` called and the index opened, so
the surplus instance imported torch in full before finding out it was surplus.

What this module pins, in the order the brief asks for it:

1. **the order.** At the moment the port is bound, `sys.modules` holds neither `torch` nor
   `onnxruntime` — checked in a subprocess, because "was it imported yet" is a question
   about a whole process and the suite has long since imported both;
2. **the second click.** A launch that finds our own server on the port opens a tab and
   leaves without importing either of them, and without opening the index at all;
3. **the window.** Something is on the screen before the index is opened and before the
   server is built, and a machine that cannot draw one starts exactly as it did;
4. **the tab.** It says which step the launch is on, in three languages, and shows the
   program by itself when the last step is done;
5. **the log.** One line per step with its duration — and in a shape `runlog` will not
   mistake for a stage of a run.

The seconds themselves are deliberately NOT asserted anywhere: they are a property of
somebody's disk. What is asserted is the order, which is what the seconds followed from.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import pytest

from sorta import i18n, launcher, runlog, splash as splash_mod, tray, ui
from sorta.ui.common import _StartupState

from tests.test_ui import UiServerTestBase

_ROOT = Path(__file__).resolve().parent.parent
_LANGS: tuple[i18n.Lang, ...] = ("ru", "en", "ja")
# Neither of these may be imported on the way to a bound port, and neither may be
# imported by a launch that turns out to be surplus.
_WATCHED = ("torch", "onnxruntime")


def _write_config(root: Path) -> Path:
    """A minimal project on disk — the config a launch is given on the command line."""
    src = root / "src"
    src.mkdir(exist_ok=True)
    path = root / "config.yaml"
    path.write_text(f"sources:\n  - {src.as_posix()}\n"
                    f"database: {(root / 'test.db').as_posix()}\n"
                    "language: en\n", encoding="utf-8")
    return path


def run_launch(root: Path, name: str, script: str, *args: str) -> dict:
    """Run one launch in a fresh interpreter and read back what it wrote down.

    The report goes into a FILE and not into stdout: a launch prints its own lines
    (`cli.tray.serving`, `cli.tray.already_running`), and mixing them with the answer
    would make the assertions depend on what the program says to a person.
    """
    script_path = root / f"{name}.py"
    script_path.write_text(script, encoding="utf-8")
    result = root / f"{name}.json"
    environment = {
        **os.environ,
        # The child is not started from the checkout, so it is told where the package is.
        "PYTHONPATH": str(_ROOT),
        # It configures logging like any launch, and must not write into the run log of
        # the person running the suite.
        runlog.ENV_LOG_FILE: str(root / f"{name}.log"),
    }
    completed = subprocess.run(
        [sys.executable, str(script_path), str(result), *args],
        capture_output=True, text=True, timeout=600, check=False,
        cwd=str(root), env=environment)
    if completed.returncode != 0 or not result.is_file():
        raise AssertionError(
            f"запуск {name} не отчитался (код {completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return json.loads(result.read_text(encoding="utf-8"))


def _fresh_state() -> _StartupState:
    """A launch state nobody else in this process is writing.

    The real one is a module singleton, and the suite runs `tray.start` in several files:
    a background `_finish_startup` left over from another case would flip `ready` under an
    assertion here. Patching the name in `sorta.ui.common` is what isolates it — both
    `startup_state()` and `_startup_payload()` read that global at call time.
    """
    return _StartupState()


# --- 1 and 2: the order, proven in a process of its own ------------------------------

# One real launch, up to the moment it is serving, reporting what had been imported when.
# `_serve_until_closed` is replaced rather than the server mocked: what has to be true is
# that a REAL bind happened with no torch behind it, and the shortest way to get the
# program to stop after that is the exit it already has (`POST /api/quit`).
_LAUNCH_SCRIPT = '''\
"""Run one launch and write down what had been imported at each point."""
import json
import sys
import time

from sorta import tray, ui

WATCHED = ("torch", "onnxruntime")
result_path = sys.argv[1]
seen = {}


def watched():
    return sorted(name for name in WATCHED if name in sys.modules)


real_build = ui.build_server


def build_and_watch(*args, **kwargs):
    seen["at_bind"] = watched()
    return real_build(*args, **kwargs)


def quit_instead_of_serving(port, lang, url, serving, *, ask, icon_factory):
    """The program is up, which is all this launch had to show. Close it the normal way."""
    seen["at_serving"] = watched()
    tray.request_quit(port, confirm=True)
    serving.join()


ui.build_server = build_and_watch
tray._serve_until_closed = quit_instead_of_serving
seen["exit"] = tray.main(sys.argv[2:])

state = ui.startup_state()
# Waits for the STEPS and not for `ready`: since 2026-08-08 ready is declared as soon as
# the server can serve, so the diagnostics behind it are still running at that moment —
# which is the whole point of the change, and would make this report arrive half empty.
deadline = time.monotonic() + 240
while (len(state.snapshot()["done"]) < len(ui.STARTUP_STEPS)
       and time.monotonic() < deadline):
    time.sleep(0.05)
seen["after"] = watched()
seen["startup"] = state.snapshot()

with open(result_path, "w", encoding="utf-8") as handle:
    json.dump(seen, handle)
'''

# A second click on the shortcut: a Sorta is already on the port, and this process must
# leave without paying for anything it does not need.
_SECOND_LAUNCH_SCRIPT = '''\
"""A launch that finds our own server already on the port."""
import json
import sys
import threading

from sorta import tray, ui
from sorta.config import load_config
from sorta.db import connect

WATCHED = ("torch", "onnxruntime")
result_path, config_path = sys.argv[1], sys.argv[2]


def watched():
    return sorted(name for name in WATCHED if name in sys.modules)


cfg = load_config(config_path)
first = ui.build_server(cfg, connect(cfg.database), port=0)
threading.Thread(target=first.serve_forever, daemon=True).start()

seen = {"before": watched()}


def no_index(*args, **kwargs):
    raise AssertionError("the second launch opened the index")


tray.connect = no_index
seen["exit"] = tray.main(["--config", config_path, "--port", str(first.server_port),
                          "--no-browser", "--no-splash"])
seen["after"] = watched()
first.shutdown()

with open(result_path, "w", encoding="utf-8") as handle:
    json.dump(seen, handle)
'''


# serial: this class starts real launches — a bound port, a server thread and an exit the
# assertions wait for — and each one spawns a fresh interpreter that imports torch. A
# loaded machine is exactly where a free port stops being free between the probe and the
# bind, which is the class of failure the split of the gate exists for.
@pytest.mark.serial
class TestNothingHeavyIsImportedOnTheWayToAnAnsweringPort(unittest.TestCase):
    """Requirements 1 and 2, asked of a whole process because that is what they are about.

    Both launches happen once, in `setUpClass`: each one is a fresh interpreter that ends
    up importing torch, and the cases below read different parts of the same report rather
    than paying for it four times.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.config = _write_config(cls.root)
        cls.first = run_launch(cls.root, "first", _LAUNCH_SCRIPT,
                               "--config", str(cls.config), "--port", "0",
                               "--no-browser", "--no-splash")
        cls.second = run_launch(cls.root, "second", _SECOND_LAUNCH_SCRIPT,
                                str(cls.config))

    def test_the_port_answers_before_torch_is_imported(self):
        """The requirement the brief asked to be able to show as a fact rather than as a
        stopwatch: whatever the machine, the bind does not stand behind the torch import."""
        self.assertEqual(self.first["exit"], 0)
        self.assertEqual(self.first["at_bind"], [],
                         "перед привязкой порта уже импортированы тяжёлые модули")
        self.assertEqual(self.first["at_serving"], [])

    def test_the_checks_still_run_afterwards(self):
        """Moving a check off the critical path is not the same as deleting it. Every step
        of the launch is recorded, in order, and the last of them is what makes it ready."""
        state = self.first["startup"]
        self.assertTrue(state["ready"], state)
        self.assertEqual([done["step"] for done in state["done"]], list(ui.STARTUP_STEPS))
        for done in state["done"]:
            with self.subTest(step=done["step"]):
                self.assertGreaterEqual(done["seconds"], 0.0)

    def test_the_gpu_check_really_did_import_torch_after_the_bind(self):
        """The other half of the same sentence: `warn_if_gpu_mismatch` is still the call
        that imports torch — it just does it with the port already answering."""
        if importlib.util.find_spec("torch") is None:  # pragma: no cover — no torch here
            self.skipTest("torch is not installed in this environment")
        self.assertIn("torch", self.first["after"])

    def test_a_second_launch_imports_nothing_and_opens_nothing(self):
        """Ten clicks used to be ten torch imports. This is what one of the nine costs."""
        self.assertEqual(self.second["exit"], 0)
        self.assertEqual(self.second["before"], [])
        self.assertEqual(self.second["after"], [],
                         "второй запуск всё ещё импортирует тяжёлые модули")


# --- 2, unit-sized: the questions asked and not asked on the surplus path -------------


class TestTheSurplusLaunchLeavesImmediately(unittest.TestCase):
    """The same requirement without a second process — what `main` does and does not do."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = _write_config(self.root)
        self.state = _fresh_state()
        patcher = mock.patch.object(ui.common, "_startup_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)

    def main_against_our_own_server(self, *extra: str) -> tuple:
        with mock.patch.object(tray, "port_holder",
                               return_value=tray.PORT_OURS) as holder, \
             mock.patch.object(splash_mod, "open_splash") as splash, \
             mock.patch.object(tray, "connect") as index, \
             mock.patch.object(tray.webbrowser, "open") as opened:
            code = tray.main(["--config", str(self.config), "--port", "8756", *extra])
        self.assertEqual(holder.call_count, 1)
        return code, splash, index, opened

    def test_it_opens_the_window_of_the_program_that_is_already_running(self):
        code, _splash, _index, opened = self.main_against_our_own_server()
        self.assertEqual(code, 0)
        opened.assert_called_once_with(tray.url_for(8756))

    def test_it_never_opens_the_index(self):
        """The index is behind the port question now, not in front of it: the surplus
        instance has no business opening somebody else's database."""
        _code, _splash, index, _opened = self.main_against_our_own_server()
        index.assert_not_called()

    def test_it_puts_no_window_on_the_screen(self):
        """Nine splashes for nine surplus clicks would be the same noise in a new place —
        and the tab that opens IS the visible answer to a second click."""
        _code, splash, _index, _opened = self.main_against_our_own_server()
        splash.assert_not_called()

    def test_it_leaves_nothing_claiming_to_be_starting(self):
        """A process that is going away must not leave the state saying a launch is
        pending: the launch that matters belongs to the program already on the port."""
        self.main_against_our_own_server()
        self.assertTrue(self.state.snapshot()["ready"])

    def test_a_stranger_on_the_port_is_still_an_error_with_a_code(self):
        with mock.patch.object(tray, "port_holder", return_value=tray.PORT_STRANGER), \
             mock.patch.object(tray, "connect") as index, \
             mock.patch.object(tray.webbrowser, "open") as opened:
            code = tray.main(["--config", str(self.config), "--port", "8756"])
        self.assertEqual(code, 1)
        index.assert_not_called()
        opened.assert_not_called()


# --- 3: the window that goes up before the work ---------------------------------------


class TestTheWindowComesFirst(unittest.TestCase):
    """Requirement 3: the click gets an answer on the screen before anything is measured."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = _write_config(self.root)
        patcher = mock.patch.object(ui.common, "_startup_state", _fresh_state())
        patcher.start()
        self.addCleanup(patcher.stop)

    def launch(self, *extra: str) -> tuple[list[str], mock.Mock]:
        """`launcher.main` with everything it calls replaced by a note of order.

        The window opens in `sorta.launcher` and not in `sorta.tray`: importing the tray
        pulls `sorta.ui`, 3.59 s on the payload interpreter with a warm cache, and a
        window opened after that cannot be on the screen during it.
        """
        order: list[str] = []
        splash = mock.Mock(name="splash")

        def opened(lang=None):
            order.append("splash")
            return splash

        def index(database):
            order.append("database")
            return mock.Mock(name="conn")

        def start(*args, **kwargs):
            order.append("start")
            self.started = kwargs
            return 0

        with mock.patch.object(splash_mod, "open_splash", side_effect=opened), \
             mock.patch.object(tray, "port_holder", return_value=tray.PORT_FREE), \
             mock.patch.object(tray, "connect", side_effect=index), \
             mock.patch.object(tray, "start", side_effect=start):
            self.code = launcher.main(["--config", str(self.config), "--port", "0",
                                       *extra])
        return order, splash

    def test_the_window_is_up_before_the_index_and_before_the_server(self):
        order, _splash = self.launch()
        self.assertEqual(order, ["splash", "database", "start"])
        self.assertEqual(self.code, 0)

    def test_the_window_is_handed_to_the_part_that_closes_it(self):
        _order, splash = self.launch()
        self.assertIs(self.started["splash"], splash)

    def test_no_splash_asks_for_no_window(self):
        order, _splash = self.launch("--no-splash")
        self.assertEqual(order, ["database", "start"])
        self.assertIsNone(self.started["splash"])

    def test_the_launcher_reaches_nothing_heavy_before_the_window(self):
        """Asked of the source rather than of a clock: the module-level imports of
        `sorta/launcher.py` are stdlib, and the program is reached inside `main`."""
        import ast

        tree = ast.parse((_ROOT / "sorta" / "launcher.py").read_text(encoding="utf-8"))
        top_level = [node for node in tree.body
                     if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in top_level:
            names = ([alias.name for alias in node.names]
                     if isinstance(node, ast.Import) else [node.module or ""])
            for name in names:
                with self.subTest(imported=name):
                    self.assertNotIn("tray", name)
                    self.assertNotIn("ui", name.split("."))
                    self.assertFalse(isinstance(node, ast.ImportFrom) and node.level)

    def test_the_window_names_the_product_and_says_it_is_starting(self):
        """One line, in the language the config asked for, and the product name above it.
        A window that said nothing would be indistinguishable from a hung program."""
        for lang in _LANGS:
            with self.subTest(lang=lang), mock.patch.object(splash_mod.subprocess,
                                                            "Popen") as popen:
                handle = splash_mod.open_splash(lang)
            argv = popen.call_args.args[0]
            self.assertEqual(argv[:2], [sys.executable, "-c"])
            self.assertEqual(argv[3], "Sorta")
            self.assertEqual(argv[4], i18n.cli_text("cli.tray.starting", lang))
            self.assertIsNotNone(handle)

    def test_a_machine_that_cannot_draw_a_window_starts_anyway(self):
        """No tkinter, no display, no permission to spawn — the same rule as the tray
        icon: the absence of a screen is a property of somebody's desktop, never a reason
        not to start."""
        with mock.patch.object(splash_mod.subprocess, "Popen",
                               side_effect=OSError("no display")):
            self.assertIsNone(splash_mod.open_splash("en"))

    def test_the_window_script_is_a_program_and_not_a_string(self):
        """It runs in another interpreter, so a syntax error in it would first be seen on
        somebody's desktop. Compiling it here is the cheapest way that cannot happen."""
        compile(splash_mod._SPLASH_SCRIPT, "<splash>", "exec")
        for needle in ("tkinter", "Progressbar", "indeterminate", "stdin", "mainloop"):
            with self.subTest(needle=needle):
                self.assertIn(needle, splash_mod._SPLASH_SCRIPT)

    def test_the_window_is_told_to_close_itself_before_it_is_ended(self):
        """EOF on its stdin first, because that is the only shutdown tkinter likes."""
        process = _FakeSplashProcess()
        splash = splash_mod._Splash(process)
        splash.close()
        self.assertTrue(process.stdin.closed)
        self.assertEqual(process.terminated, 0)
        self.assertEqual(process.waits, 1)

    def test_a_window_that_will_not_go_is_ended(self):
        process = _FakeSplashProcess(wait_times_out=True)
        splash_mod._Splash(process).close()
        self.assertEqual(process.terminated, 1)

    def test_closing_twice_is_closing_once(self):
        """`start` closes it when the tab opens and again in its `finally`; a second close
        must not wait another three seconds for a process that has long gone."""
        process = _FakeSplashProcess()
        splash = splash_mod._Splash(process)
        splash.close()
        splash.close()
        self.assertEqual(process.waits, 1)

    def test_a_window_that_is_already_gone_is_not_an_error(self):
        process = _FakeSplashProcess(stdin_raises=True, wait_raises=True)
        splash_mod._Splash(process).close()  # must not raise
        self.assertEqual(process.terminated, 0)


class _FakeSplashProcess:
    """Just enough of `Popen` to see how the window is asked to leave."""

    class _Stdin:
        def __init__(self, raises: bool) -> None:
            self.closed = False
            self._raises = raises

        def close(self) -> None:
            if self._raises:
                raise OSError("the pipe is gone")
            self.closed = True

    def __init__(self, *, wait_times_out: bool = False, wait_raises: bool = False,
                 stdin_raises: bool = False) -> None:
        self.stdin = self._Stdin(stdin_raises)
        self.waits = 0
        self.terminated = 0
        self._wait_times_out = wait_times_out
        self._wait_raises = wait_raises

    def wait(self, timeout: float | None = None) -> int:
        self.waits += 1
        if self._wait_raises:
            raise OSError("no such process")
        if self._wait_times_out:
            raise subprocess.TimeoutExpired("splash", timeout or 0)
        return 0

    def terminate(self) -> None:
        self.terminated += 1


# --- 4: the tab says what the launch is doing -----------------------------------------


class TestWhatTheLaunchSaysAboutItself(unittest.TestCase):
    """The state the page reads — the whole vocabulary of the waiting screen."""

    def setUp(self):
        self.state = _StartupState()

    def test_a_program_that_never_declared_a_launch_is_ready(self):
        """The default, and the important one: `sorta ui` declares nothing, so a page
        served by it must never show a screen it has no way out of."""
        snapshot = self.state.snapshot()
        self.assertTrue(snapshot["ready"])
        self.assertIsNone(snapshot["step"])
        self.assertEqual(snapshot["steps"], [])

    def test_declaring_a_launch_names_its_steps_and_says_it_is_not_ready(self):
        self.state.expect()
        snapshot = self.state.snapshot()
        self.assertFalse(snapshot["ready"])
        self.assertEqual(snapshot["steps"], list(ui.STARTUP_STEPS))

    def test_the_current_step_is_the_one_that_was_entered(self):
        self.state.expect()
        self.state.enter(ui.STARTUP_GPU)
        self.assertEqual(self.state.snapshot()["step"], ui.STARTUP_GPU)

    def test_a_finished_step_carries_what_it_cost(self):
        self.state.expect()
        self.state.enter(ui.STARTUP_GPU)
        self.state.leave(ui.STARTUP_GPU, 3.76)
        snapshot = self.state.snapshot()
        self.assertEqual(snapshot["done"], [{"step": ui.STARTUP_GPU, "seconds": 3.76}])
        self.assertIsNone(snapshot["step"], "шаг остался текущим после завершения")

    def test_ready_clears_the_step_and_freezes_the_total(self):
        """"How long did the launch take" is a fact about a launch that is over — not a
        clock that keeps running for as long as the program serves."""
        self.state.expect()
        self.state.enter(ui.STARTUP_GEO)
        self.state.ready()
        snapshot = self.state.snapshot()
        self.assertTrue(snapshot["ready"])
        self.assertIsNone(snapshot["step"])
        total = self.state.elapsed()
        time.sleep(0.05)
        self.assertEqual(self.state.elapsed(), total,
                         "общее время продолжает идти после готовности")

    def test_the_clock_only_runs_while_the_launch_does(self):
        self.assertEqual(self.state.elapsed(), 0.0)
        self.state.expect()
        self.assertGreaterEqual(self.state.elapsed(), 0.0)

    def test_a_reset_is_a_process_that_is_not_launching(self):
        self.state.expect()
        self.state.enter(ui.STARTUP_CONFIG)
        self.state.reset()
        self.assertEqual(self.state.snapshot(),
                         {"ready": True, "step": None, "steps": [], "done": [],
                          "elapsed": 0.0})

    def test_the_answer_has_the_five_fields_and_no_sixth(self):
        """The waiting screen is NOT the model download (F222/F225) and must not grow
        into it: this payload says which step, never how many megabytes."""
        self.assertEqual(set(self.state.snapshot()),
                         {"ready", "step", "steps", "done", "elapsed"})

    def test_the_steps_are_the_ones_the_launch_actually_walks(self):
        """The list on screen and the list in the code are one list. A step added to the
        launch without a place here would be numbered "5 of 4" on the page."""
        self.assertEqual(ui.STARTUP_STEPS,
                         (ui.STARTUP_CONFIG, ui.STARTUP_PORT, ui.STARTUP_DATABASE,
                          ui.STARTUP_SERVER, ui.STARTUP_ENVIRONMENT, ui.STARTUP_GPU,
                          ui.STARTUP_GEO))


class TestTheRouteAnswersWhileTheLaunchIsGoing(UiServerTestBase):
    """`GET /api/startup` — the one route that is asked before the program is ready."""

    def setUp(self):
        super().setUp()
        self.state = _fresh_state()
        patcher = mock.patch.object(ui.common, "_startup_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.start_server()

    def get(self) -> dict:
        """The answer, with the CONNECTION retried once.

        The known Windows flake of a threaded server and a client racing on teardown
        (`ConnectionAbortedError: [WinError 10053]`) — retried the way
        `test_ui_review.post_status` retries it, by opening a fresh connection. The answer
        itself is never retried: a wrong body is a failure, not a hiccup.
        """
        last: OSError | None = None
        for _attempt in (1, 2):
            try:
                with urllib.request.urlopen(f"{self.base_url}/api/startup",
                                            timeout=5) as resp:
                    self.assertEqual(resp.status, 200)
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                self.fail(f"/api/startup ответил {exc.code}")
            except OSError as exc:
                last = exc
        raise AssertionError(f"/api/startup недоступен: {last}")

    def test_a_server_that_declared_no_launch_says_it_is_ready(self):
        self.assertTrue(self.get()["ready"])

    def test_it_names_the_step_the_launch_is_on(self):
        self.state.expect()
        self.state.enter(ui.STARTUP_GPU)
        answer = self.get()
        self.assertFalse(answer["ready"])
        self.assertEqual(answer["step"], ui.STARTUP_GPU)
        self.assertEqual(answer["steps"], list(ui.STARTUP_STEPS))

    def test_it_flips_to_ready_without_the_server_being_touched(self):
        """What makes the tab show the program: the same route, a different answer, no
        restart and nothing for the person to press."""
        self.state.expect()
        self.state.enter(ui.STARTUP_GEO)
        self.assertFalse(self.get()["ready"])
        self.state.ready()
        self.assertTrue(self.get()["ready"])


class TestTheWaitingScreenIsOnThePage(unittest.TestCase):
    """The markup and the script — there is no engine here, so what is pinned is the
    shape that IS the behaviour (the way `test_ui_master_switch` pins its half)."""

    @classmethod
    def setUpClass(cls):
        cls.html = ui._render_index_html("ru")

    def test_the_screen_is_in_the_markup_and_starts_hidden(self):
        """Hidden first, asked afterwards: a screen that appeared on every load and then
        found out it was not needed would flash on every page of a working program."""
        self.assertIn('<div id="startup" class="startup" hidden>', self.html)

    def test_it_asks_the_route_and_keeps_asking_until_it_is_ready(self):
        self.assertIn('fetch("/api/startup")', self.html)
        self.assertIn("window.setTimeout(poll, STARTUP_POLL_MS);", self.html)

    def test_ready_shows_the_program_and_a_failure_does_too(self):
        """An overlay nobody can dismiss is worse than a first fetch that went missing."""
        self.assertIn("if (!state || state.ready) { box.hidden = true; return; }",
                      self.html)
        self.assertIn('.catch(function () { box.hidden = true; });', self.html)

    def test_the_step_is_named_in_words_and_never_as_a_percentage(self):
        self.assertIn('I18N["startup_step_" + step]', self.html)
        self.assertIn("I18N.startup_step_other", self.html)
        # The bar says "working" and nothing more — no width, no percentage anywhere near
        # it, which is the whole point of an indeterminate one.
        self.assertIn("startup-bar-run", self.html)
        self.assertIn("animation: startup-sweep", self.html)

    def test_every_language_still_fills_the_screen_in(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                html = ui._render_index_html(lang)
                self.assertIn(ui._UI_STRINGS["startup_title"][lang], html)
                self.assertIn(ui._UI_STRINGS["startup_note"][lang], html)


class TestEveryStepHasWordsInThreeLanguages(unittest.TestCase):
    """F227 in the string catalog: a step the page cannot name is a blank line on screen."""

    CHROME = ("startup_title", "startup_note", "startup_step_counter",
              "startup_step_other")

    def test_every_step_of_the_launch_has_a_caption(self):
        for step in ui.STARTUP_STEPS:
            key = f"startup_step_{step}"
            with self.subTest(step=step):
                self.assertIn(key, ui._UI_STRINGS)
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), set(_LANGS))
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} пуст")

    def test_the_chrome_of_the_screen_exists_in_three_languages(self):
        for key in self.CHROME:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), set(_LANGS))
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} пуст")

    def test_the_counter_carries_both_of_its_numbers(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                template = ui._UI_STRINGS["startup_step_counter"][lang]
                self.assertIn("{step}", template)
                self.assertIn("{total}", template)

    def test_the_line_of_the_window_exists_in_three_languages(self):
        entry = i18n._CLI_STRINGS["cli.tray.starting"]
        self.assertEqual(set(entry), set(_LANGS))
        self.assertEqual(len({value for value in entry.values()}), 3,
                         "строка окна не переведена, а скопирована")

    def test_the_captions_are_translated_and_not_copied(self):
        for step in ui.STARTUP_STEPS:
            with self.subTest(step=step):
                entry = ui._UI_STRINGS[f"startup_step_{step}"]
                self.assertEqual(len({value for value in entry.values()}), 3)


# --- 5: one line per step, with its duration ------------------------------------------


class TestTheLaunchWritesDownWhatItSpent(unittest.TestCase):
    """Requirement 5. «Долго» was a guess about somebody's VM; the next person to ask
    should get the answer out of the file."""

    def setUp(self):
        self.state = _fresh_state()
        patcher = mock.patch.object(ui.common, "_startup_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_step_writes_one_line_with_its_name_and_its_duration(self):
        with self.assertLogs("sorta.tray", level="INFO") as logs:
            with tray._startup_step(ui.STARTUP_GPU):
                pass
        lines = [record.getMessage() for record in logs.records]
        self.assertEqual(len(lines), 1, lines)
        self.assertRegex(lines[0], r"^startup step=gpu elapsed=\d+\.\d{3}$")

    def test_a_step_that_raised_is_still_timed(self):
        """A launch that fell over is exactly the one whose timings are worth having."""
        with self.assertLogs("sorta.tray", level="INFO") as logs:
            with self.assertRaises(ValueError):
                with tray._startup_step(ui.STARTUP_DATABASE):
                    raise ValueError("no index")
        self.assertRegex(logs.records[0].getMessage(),
                         r"^startup step=database elapsed=")

    def test_the_line_is_not_read_back_as_a_stage_of_a_run(self):
        """`runlog` prices the next run off `stage=<name> elapsed=` lines in this very
        file. A launch is not a stage of the pipeline, and a `startup` "stage" appearing
        in the measurements would make an estimate out of a start-up."""
        record = logging.LogRecord("sorta.tray", logging.INFO, __file__, 0,
                                   tray._STARTUP_LINE, (ui.STARTUP_GPU, 3.76), None)
        line = logging.Formatter(runlog._FORMAT, runlog._DATEFMT).format(record)
        self.assertIn("startup step=gpu elapsed=3.760", line)
        self.assertIsNone(runlog._MEASUREMENT_RE.match(line))

    def test_ready_is_declared_before_the_checks_not_after_them(self):
        """The correction of 2026-08-08: ready means the server can serve.

        F227 moved the diagnostics behind the bind and then waited for them anyway, so
        the tab held a page reading "the program already answers" without showing it —
        for several minutes on a cold machine, stopped at the step where
        `log_environment` imports torch. The line that says ready must come FIRST, and
        the three probes report into the log behind it.
        """
        self.state.expect()
        with mock.patch.object(tray, "log_environment"), \
             mock.patch.object(tray, "warn_if_gpu_mismatch"), \
             mock.patch.object(tray, "warn_if_geo_data_missing"), \
             self.assertLogs("sorta.tray", level="INFO") as logs:
            tray._finish_startup()
        lines = [record.getMessage() for record in logs.records]
        self.assertEqual(len(lines), 4, lines)
        self.assertRegex(lines[0], r"^startup ready elapsed=\d+\.\d{3}$")
        for line, step in zip(lines[1:], (ui.STARTUP_ENVIRONMENT, ui.STARTUP_GPU,
                                          ui.STARTUP_GEO)):
            with self.subTest(step=step):
                self.assertRegex(line, rf"^startup step={step} elapsed=\d+\.\d{{3}}$")
        self.assertTrue(self.state.snapshot()["ready"])

    def test_ready_does_not_wait_for_a_slow_check(self):
        """The failure as the owner met it: a probe that takes minutes may not hold the
        page. Ready is already true while the first check is still running."""
        self.state.expect()
        seen: dict = {}

        def slow_environment():
            seen["ready_while_running"] = self.state.snapshot()["ready"]

        with mock.patch.object(tray, "log_environment", side_effect=slow_environment), \
             mock.patch.object(tray, "warn_if_gpu_mismatch"), \
             mock.patch.object(tray, "warn_if_geo_data_missing"), \
             self.assertLogs("sorta.tray", level="INFO"):
            tray._finish_startup()
        self.assertTrue(seen["ready_while_running"])

    def test_a_check_that_fails_does_not_stop_the_launch(self):
        """The program is already serving by the time these run. A failed probe is a
        failed probe — it may not leave the page waiting forever for a step that died."""
        self.state.expect()
        with mock.patch.object(tray, "log_environment",
                               side_effect=RuntimeError("no environment")), \
             mock.patch.object(tray, "warn_if_gpu_mismatch",
                               side_effect=OSError("no driver")), \
             mock.patch.object(tray, "warn_if_geo_data_missing"), \
             self.assertLogs("sorta.tray", level="INFO"):
            tray._finish_startup()
        snapshot = self.state.snapshot()
        self.assertTrue(snapshot["ready"])
        self.assertEqual([done["step"] for done in snapshot["done"]],
                         [ui.STARTUP_ENVIRONMENT, ui.STARTUP_GPU, ui.STARTUP_GEO])


class TestTheServerSideOfTheLaunchIsWiredUp(unittest.TestCase):
    """`start` is where the reorder lands: bind, then serve, then the diagnostics."""

    def setUp(self):
        self.state = _fresh_state()
        patcher = mock.patch.object(ui.common, "_startup_state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_the_diagnostics_run_on_a_thread_of_their_own(self):
        """On the serving thread they would be the silence again, one layer down."""
        finished = threading.Event()
        seen: dict[str, object] = {}

        def finish():
            seen["thread"] = threading.current_thread().name
            finished.set()

        conn = mock.Mock()
        splash = mock.Mock()
        with mock.patch.object(tray, "port_holder", return_value=tray.PORT_FREE), \
             mock.patch.object(tray, "_finish_startup", side_effect=finish), \
             mock.patch.object(tray, "_serve_until_closed"), \
             mock.patch.object(tray.ui, "build_server") as build, \
             mock.patch.object(tray.webbrowser, "open") as opened:
            build.return_value = mock.Mock(server_port=8756)
            code = tray.start(mock.Mock(language="en"), conn, port=8756,
                              open_browser=True, splash=splash)
        self.assertEqual(code, 0)
        self.assertTrue(finished.wait(5), "проверки окружения не запустились")
        self.assertNotEqual(seen["thread"], threading.main_thread().name)
        opened.assert_called_once()
        conn.close.assert_called_once()

    def test_the_window_goes_away_once_the_tab_has_been_asked_for(self):
        """It lives exactly as long as it is the only thing on screen."""
        order: list[str] = []
        splash = mock.Mock()
        splash.close.side_effect = lambda: order.append("splash closed")
        with mock.patch.object(tray, "port_holder", return_value=tray.PORT_FREE), \
             mock.patch.object(tray, "_finish_startup"), \
             mock.patch.object(tray, "_serve_until_closed",
                               side_effect=lambda *a, **k: order.append("serving")), \
             mock.patch.object(tray.ui, "build_server") as build, \
             mock.patch.object(tray.webbrowser, "open",
                               side_effect=lambda url: order.append("browser")):
            build.return_value = mock.Mock(server_port=8756)
            tray.start(mock.Mock(language="en"), mock.Mock(), port=8756,
                       open_browser=True, splash=splash)
        self.assertEqual(order[:3], ["browser", "splash closed", "serving"])

    def test_a_busy_port_found_late_still_takes_the_window_away(self):
        """Every way out of `start` — including the ones that never reach the browser."""
        splash = mock.Mock()
        with mock.patch.object(tray, "port_holder", return_value=tray.PORT_STRANGER), \
             mock.patch("builtins.print"):
            code = tray.start(mock.Mock(language="en"), mock.Mock(), port=8756,
                              open_browser=False, splash=splash)
        self.assertEqual(code, 1)
        splash.close.assert_called_once()

    def test_the_bind_is_a_step_of_the_launch_like_the_rest(self):
        with mock.patch.object(tray, "port_holder", return_value=tray.PORT_FREE), \
             mock.patch.object(tray, "_finish_startup"), \
             mock.patch.object(tray, "_serve_until_closed"), \
             mock.patch.object(tray.ui, "build_server") as build, \
             mock.patch.object(tray.webbrowser, "open"):
            build.return_value = mock.Mock(server_port=8756)
            tray.start(mock.Mock(language="en"), mock.Mock(), port=8756,
                       open_browser=False)
        self.assertEqual([done["step"] for done in self.state.snapshot()["done"]],
                         [ui.STARTUP_SERVER])


class TestTheLaunchIsSpelledTheSameEverywhere(unittest.TestCase):
    """The names the entry point uses and the names the page knows are one list."""

    def test_the_entry_point_walks_exactly_the_declared_steps(self):
        source = Path(tray.__file__).read_text(encoding="utf-8")
        used = set(re.findall(r"ui\.(STARTUP_[A-Z]+)\b", source)) - {"STARTUP_STEPS"}
        declared = {name for name in dir(ui)
                    if name.startswith("STARTUP_") and name != "STARTUP_STEPS"}
        self.assertEqual(used, declared,
                         "шаг объявлен, но запуск его не проходит (или наоборот)")

    def test_the_route_is_served_and_needs_no_body(self):
        source = (Path(ui.__file__).resolve()).read_text(encoding="utf-8")
        self.assertIn('elif path == "/api/startup":', source)
        self.assertIn("_startup_payload()", source)


if __name__ == "__main__":
    unittest.main()
