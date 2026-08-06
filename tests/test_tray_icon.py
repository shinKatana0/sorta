"""F207: the icon in the tray — open the window, close the program, lose no run.

The six things the brief asks to pin, in the order it asks for them:

1. the main one — quitting while a run is going ASKS. Not "warns and quits anyway": a
   no leaves the run counting and the server serving, and only a yes sends the second
   request, the one that carries the confirmation;
2. a correct exit gives the port back — proven the way a person proves it, by starting
   the program again on the very same port;
3. a machine with no tray keeps serving: no icon, no error, and the address on the
   console exactly as `sorta ui` prints it;
4. a second launch against a port held by OUR OWN server opens the browser and leaves;
5. a port held by a stranger still gives a clear error and a non-zero exit code;
6. the menu captions exist in all three languages.

The icon itself is injected (`icon_factory`), and so is the dialog (`ask`). What is
worth pinning is the DECISION — the question asked, the run left alone, the port given
back — and none of that is checkable through somebody else's desktop notification area.
"""
from __future__ import annotations

import http.server
import re
import socket
import sqlite3
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from sorta import i18n, tray, ui
from sorta.config import Config
from sorta.db import connect

_LANGS = ("ru", "en", "ja")


def _free_port() -> int:
    """A port the OS has just confirmed is free — the one the exit has to give back."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _printed(mocked) -> str:
    return " ".join(str(call.args[0]) for call in mocked.call_args_list if call.args)


def _wait_for_print(mocked, needle: str, timeout: float = 10.0) -> str:
    """Everything printed so far, once `needle` is among it (or the wait ran out).

    The lines this waits for are written by the program's own thread AFTER the server
    started answering, so "the server is up" is not yet "the line is out".
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        said = _printed(mocked)
        if needle in said:
            return said
        time.sleep(0.02)
    return _printed(mocked)


class FakeIcon:
    """A stand-in for `pystray.Icon`: it blocks in `run()` and returns from `stop()`.

    That is the whole contract the module leans on — the menu callbacks are held as the
    functions `build_icon` would wire into the two items, so a test can press them.
    """

    def __init__(self, on_open, on_quit) -> None:
        self.on_open = on_open
        self.on_quit = on_quit
        self.running = threading.Event()
        self._stopped = threading.Event()
        self.stops = 0

    def run(self) -> None:
        self.running.set()
        self._stopped.wait()

    def stop(self) -> None:
        self.stops += 1
        self._stopped.set()


class TrayProgramTestBase(unittest.TestCase):
    """A real `tray.start` on a real port, in a thread of its own.

    The server is the real one (`ui.build_server` + `serve_forever`), because what is
    being checked is the EXIT: the loop returning, the socket closing, the connection
    closing. Only the picture and the dialog are stand-ins.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src_dir = self.root / "src"
        self.src_dir.mkdir()
        self.cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                          raw={}, language="en")
        setup_conn = connect(self.cfg.database)
        setup_conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
               VALUES (?, 1, 0, 'jpg', 'photo', '2026-01-01')""",
            (str(self.src_dir / "a.jpg"),))
        setup_conn.commit()
        setup_conn.close()
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.icon: FakeIcon | None = None
        self.conn: sqlite3.Connection | None = None
        self.answers: list[tuple[str, str]] = []
        self.answer = False
        self.thread: threading.Thread | None = None
        self.exit_code: int | None = None
        self.start_error: BaseException | None = None
        self.addCleanup(self._stop_if_still_serving)

    # --- the program under test ---------------------------------------------

    def _ask(self, title: str, question: str) -> bool:
        self.answers.append((title, question))
        return self.answer

    def _make_icon(self, port, lang, *, on_open, on_quit):
        self.icon = FakeIcon(on_open, on_quit)
        return self.icon

    def start_program(self, *, icon_factory=None) -> None:
        """`tray.start` on a thread, with whatever it raises kept for the assertions."""
        factory = self._make_icon if icon_factory is None else icon_factory
        opened = threading.Event()

        def run() -> None:
            conn = connect(self.cfg.database)
            self.conn = conn
            opened.set()
            try:
                self.exit_code = tray.start(
                    self.cfg, conn, port=self.port, open_browser=False,
                    ask=self._ask, icon_factory=factory)
            except BaseException as exc:  # noqa: BLE001 — re-raised by the assertions
                self.start_error = exc

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self.assertTrue(opened.wait(30), "поток программы не стартовал")
        self._wait_until_answering()

    def _wait_until_answering(self) -> None:
        # Generous on purpose: `start` opens with `log_environment`, which probes the
        # GPU, and the first call of that in a test process imports torch (~7 s).
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.start_error is not None:
                raise self.start_error
            try:
                with urllib.request.urlopen(f"{self.base_url}/api/config", timeout=2):
                    return
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        self.fail(f"сервер не отвечает на порту {self.port}")

    def _stop_if_still_serving(self) -> None:
        thread = self.thread
        if thread is None:
            return
        thread.join(timeout=2)
        if thread.is_alive():
            try:
                tray.request_quit(self.port, confirm=True)
            except OSError:
                pass
            thread.join(timeout=10)

    # --- what the tests say --------------------------------------------------

    def press_quit(self) -> None:
        """The «Выйти» menu item, pressed the way pystray presses it."""
        self.icon.on_quit()

    def assert_program_ended(self) -> None:
        self.thread.join(timeout=15)
        self.assertFalse(self.thread.is_alive(), "программа не завершилась")
        self.assertEqual(self.exit_code, 0)

    def assert_still_serving(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/api/config", timeout=5) as resp:
            self.assertEqual(resp.status, 200)


class _CountingRun:
    """A stand-in for the pipeline thread: it counts, through the real state object.

    The progress goes through `_ProcessState.set_progress`, which is where cancellation
    is raised from — so this thread stops exactly the way a real stage stops, and "the
    run carried on" is a number that kept moving rather than a flag nobody read.
    """

    def __init__(self, state) -> None:
        self.state = state
        self.cancelled = threading.Event()
        self._done = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.state.try_start("/somewhere")
        self._thread.start()

    def _loop(self) -> None:
        try:
            while True:
                self._done += 1
                self.state.set_progress(self._done, 1_000_000)
                time.sleep(0.005)
        except ui._PipelineCancelled:
            self.state.finish(None)
            self.cancelled.set()

    def progressed(self, timeout: float = 2.0) -> bool:
        """Did the run advance from where it is now? — the proof it was not lost."""
        was = self.state.snapshot()["done"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.state.snapshot()["done"] > was:
                return True
            time.sleep(0.01)
        return False

    def stop(self) -> None:
        self.state.request_cancel()
        self._thread.join(timeout=5)


class RunningProgramTestBase(TrayProgramTestBase):
    """The same program, with a run genuinely in flight on its `_ProcessState`."""

    def setUp(self):
        super().setUp()
        self.state = ui._ProcessState()
        self.run = _CountingRun(self.state)
        self.run.start()
        self.addCleanup(self.run.stop)
        patcher = mock.patch.object(ui, "_ProcessState", return_value=self.state)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.start_program()


class TestQuittingDuringARunAsks(RunningProgramTestBase):
    """Brief test 1, the main one. A pass over a real collection counts for hours, and
    one click on a menu item may not end it silently."""

    def test_the_question_is_asked(self):
        self.answer = False
        self.press_quit()
        self.assertEqual(len(self.answers), 1, "вопрос не был задан")
        title, question = self.answers[0]
        self.assertEqual(title, i18n.cli_text("cli.tray.quit_title", "en"))
        self.assertEqual(question, i18n.cli_text("cli.tray.quit_process", "en"))

    def test_a_no_leaves_the_run_alone(self):
        self.answer = False
        self.press_quit()
        self.assertFalse(self.state.cancel_requested(),
                         "отказ не должен трогать флаг отмены")
        self.assertTrue(self.run.progressed(), "прогон встал после отказа")
        self.assertTrue(self.state.snapshot()["running"])

    def test_a_no_leaves_the_program_running(self):
        self.answer = False
        self.press_quit()
        self.assert_still_serving()
        self.assertTrue(self.thread.is_alive())
        self.assertEqual(self.icon.stops, 0, "значок пропал после отказа")

    def test_a_yes_interrupts_the_run_through_the_existing_flag(self):
        self.answer = True
        self.press_quit()
        self.assertTrue(self.run.cancelled.wait(5), "прогон не был прерван")
        self.assertFalse(self.state.snapshot()["running"])

    def test_a_yes_closes_the_program_and_takes_the_icon_away(self):
        self.answer = True
        self.press_quit()
        self.assert_program_ended()
        self.assertGreaterEqual(self.icon.stops, 1, "значок остался в трее")

    def test_the_refusal_comes_from_the_server_and_not_from_the_menu(self):
        """The menu asks; the rule lives in the route. `/api/quit` is what says no, so a
        request sent past the tray meets the same 409."""
        status, payload = tray.request_quit(self.port)
        self.assertEqual(status, 409, payload)
        self.assertEqual(payload["reason"], ui.QUIT_RUN_IN_PROGRESS)
        self.assertEqual(payload["running"], "process")
        self.assertFalse(self.state.cancel_requested())


class TestQuittingWithNothingRunning(TrayProgramTestBase):
    """Brief test 2: nothing to ask about, and the port comes back."""

    def setUp(self):
        super().setUp()
        self.start_program()

    def test_nothing_is_asked_when_nothing_is_running(self):
        self.press_quit()
        self.assert_program_ended()
        self.assertEqual(self.answers, [], "спросили там, где нечего терять")

    def test_the_port_is_free_afterwards(self):
        """Proven by starting the program again on the very same port — a socket the
        first one had not given back would come back as a busy-port exit code."""
        self.press_quit()
        self.assert_program_ended()
        self.thread = None
        self.start_program()
        self.assert_still_serving()

    def test_the_index_connection_is_closed_and_the_database_is_whole(self):
        """Requirement 1: the exit is the Ctrl+C one, so the connection the index was
        read through is closed rather than left to a dying process."""
        self.press_quit()
        self.assert_program_ended()
        with self.assertRaises(sqlite3.ProgrammingError):
            self.conn.execute("SELECT 1")
        conn = connect(self.cfg.database)
        try:
            self.assertEqual(len(conn.execute("SELECT path FROM files").fetchall()), 1)
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        finally:
            conn.close()
        for suffix in ("-journal", "-wal"):
            leftover = self.cfg.database.with_name(self.cfg.database.name + suffix)
            with self.subTest(suffix=suffix):
                self.assertFalse(leftover.exists(), f"{leftover.name} остался на диске")

    def test_the_page_can_still_close_the_program_and_the_icon_follows(self):
        """The menu is not the only way out (F209). Whichever way the server goes, the
        icon may not outlive it."""
        status, payload = tray.request_quit(self.port)
        self.assertEqual(status, 200, payload)
        self.assert_program_ended()
        self.assertGreaterEqual(self.icon.stops, 1, "значок остался в трее")


class TestAMachineWithoutATray(TrayProgramTestBase):
    """Brief test 3: no indicator, no pystray, no display — the program still runs."""

    def _no_tray(self, port, lang, *, on_open, on_quit):
        raise tray.TrayUnavailable("pystray is not installed: no module named pystray")

    def test_the_server_works_and_nothing_fails(self):
        # The address is still printed, exactly as `sorta ui` prints it: without an icon
        # the console IS the interface again.
        serving_line = i18n.cli_text("cli.ui.serving", "en", url=f"{self.base_url}/")
        with mock.patch("builtins.print") as printed:
            self.start_program(icon_factory=self._no_tray)
            self.assert_still_serving()
            self.assertIsNone(self.start_error)
            self.assertIsNone(self.icon, "значок был построен там, где трея нет")
            said = _wait_for_print(printed, serving_line)
        self.assertIn(serving_line, said)
        self.assertIn("pystray is not installed", said)

    def test_it_still_closes_cleanly(self):
        self.start_program(icon_factory=self._no_tray)
        status, payload = tray.request_quit(self.port)
        self.assertEqual(status, 200, payload)
        self.assert_program_ended()

    def test_a_missing_library_is_reported_as_no_tray_and_not_as_a_crash(self):
        with mock.patch.dict("sys.modules", {"pystray": None}):
            with self.assertRaises(tray.TrayUnavailable):
                tray.build_icon(8756, "en", on_open=lambda: None, on_quit=lambda: None)

    def test_an_unreadable_icon_file_is_no_tray_either(self):
        with mock.patch.object(tray, "ICON_PATH", self.root / "nope.ico"):
            with self.assertRaises(tray.TrayUnavailable):
                tray.icon_image()


class TestASecondLaunchOnOurOwnPort(TrayProgramTestBase):
    """Brief test 4: a shortcut clicked twice opens the window instead of failing."""

    def setUp(self):
        super().setUp()
        self.start_program()

    def test_the_running_server_is_recognised_as_ours(self):
        self.assertTrue(tray.sorta_is_serving(self.port))
        self.assertEqual(tray.port_holder(self.port), tray.PORT_OURS)

    def test_the_second_launch_opens_the_browser_and_leaves(self):
        with mock.patch.object(tray.webbrowser, "open") as opened:
            code = tray.main(["--port", str(self.port),
                              "--config", str(self._config_file())])
        self.assertEqual(code, 0)
        opened.assert_called_once_with(f"{self.base_url}/")

    def test_the_first_program_is_untouched_by_the_second(self):
        with mock.patch.object(tray.webbrowser, "open"):
            tray.main(["--port", str(self.port), "--config", str(self._config_file())])
        self.assert_still_serving()
        self.assertTrue(self.thread.is_alive())

    def _config_file(self) -> Path:
        path = self.root / "config.yaml"
        path.write_text(f"sources:\n  - {self.src_dir.as_posix()}\n"
                        f"database: {(self.root / 'test.db').as_posix()}\n"
                        "language: en\n", encoding="utf-8")
        return path


class TestAPortHeldByAStranger(unittest.TestCase):
    """Brief test 5: somebody else's server on the port is still a clear error."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = Config(sources=[self.root], database=self.root / "test.db",
                          raw={}, language="en")
        self.stranger = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
        self.addCleanup(self.stranger.server_close)
        self.port = self.stranger.server_port
        threading.Thread(target=self.stranger.serve_forever, daemon=True).start()
        self.addCleanup(self.stranger.shutdown)

    def test_a_stranger_is_not_mistaken_for_us(self):
        self.assertFalse(tray.sorta_is_serving(self.port, timeout=5))
        self.assertEqual(tray.port_holder(self.port, timeout=5), tray.PORT_STRANGER)

    def test_the_launch_fails_with_a_readable_message(self):
        conn = connect(self.cfg.database)
        with mock.patch("builtins.print") as printed:
            code = tray.start(self.cfg, conn, port=self.port, open_browser=False)
        self.assertEqual(code, 1)
        said = _printed(printed)
        self.assertIn(str(self.port), said)
        self.assertIn(i18n.cli_text("cli.tray.port_busy", "en", port=self.port), said)

    def test_nothing_is_opened_for_a_stranger(self):
        conn = connect(self.cfg.database)
        with mock.patch.object(tray.webbrowser, "open") as opened:
            self.assertEqual(
                tray.start(self.cfg, conn, port=self.port, open_browser=True), 1)
        opened.assert_not_called()

    def test_the_stranger_keeps_its_port(self):
        """A busy port is not something to take: `http.server` sets SO_REUSEADDR, and on
        Windows a second bind to a live port SUCCEEDS. The check that stops that is a
        question asked before anything is bound, so the neighbour keeps answering."""
        conn = connect(self.cfg.database)
        tray.start(self.cfg, conn, port=self.port, open_browser=False)
        with socket.create_connection(("127.0.0.1", self.port), timeout=5):
            pass

    def test_a_port_nobody_holds_is_free(self):
        free = _free_port()
        self.assertFalse(tray.sorta_is_serving(free, timeout=2))
        self.assertEqual(tray.port_holder(free, timeout=2), tray.PORT_FREE)

    def test_port_zero_is_nobodys(self):
        """`port=0` is "let the OS choose" — the tests use it, and it must not be read
        as a port somebody is sitting on."""
        self.assertEqual(tray.port_holder(0), tray.PORT_FREE)


class TestTheCaptions(unittest.TestCase):
    """Brief test 6: three languages, from the catalog the rest of the interface uses."""

    KEYS = ("cli.tray.open", "cli.tray.quit", "cli.tray.tooltip", "cli.tray.serving",
            "cli.tray.no_icon", "cli.tray.already_running", "cli.tray.port_busy",
            "cli.tray.quit_title", "cli.tray.quit_running", "cli.tray.quit_failed")

    def test_every_caption_exists_in_all_three_languages(self):
        for key in self.KEYS + tuple(tray.QUIT_QUESTION_KEYS.values()):
            with self.subTest(key=key):
                entry = i18n._CLI_STRINGS[key]
                self.assertEqual(set(entry), set(_LANGS))
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} пуст")

    def test_the_menu_items_are_translated_and_not_copied(self):
        for key in ("cli.tray.open", "cli.tray.quit"):
            with self.subTest(key=key):
                texts = {i18n.cli_text(key, lang) for lang in _LANGS}
                self.assertEqual(len(texts), 3, key)

    def test_every_thing_that_can_be_running_has_a_question_of_its_own(self):
        """The vocabulary of the answer and the questions the tray asks are one list —
        the same rule F209 pinned for the page. A fourth long operation added tomorrow
        would come back in `running` with no question behind it."""
        self.assertEqual(set(tray.QUIT_QUESTION_KEYS), set(ui.QUIT_RUNNING_NAMES))
        for name in ui.QUIT_RUNNING_NAMES:
            with self.subTest(running=name):
                question = tray.quit_question(name, "ru")
                self.assertEqual(question,
                                 i18n.cli_text(tray.QUIT_QUESTION_KEYS[name], "ru"))
                self.assertNotEqual(question, tray.quit_question("whatever", "ru"))

    def test_an_unknown_operation_is_still_asked_about(self):
        """A name this build has no sentence for must not fall through to quitting —
        a generic question is still a question."""
        self.assertEqual(tray.quit_question("something_new", "en"),
                         i18n.cli_text("cli.tray.quit_running", "en"))
        self.assertEqual(tray.quit_question(None, "en"),
                         i18n.cli_text("cli.tray.quit_running", "en"))

    def test_the_tooltip_carries_the_address(self):
        for lang in _LANGS:
            with self.subTest(lang=lang):
                self.assertIn("http://127.0.0.1:8756/",
                              i18n.cli_text("cli.tray.tooltip", lang,
                                            url=tray.url_for(8756)))

    def test_the_catalog_is_asked_for_exactly_the_keys_that_exist(self):
        source = Path(tray.__file__).read_text(encoding="utf-8")
        used = set(re.findall(r'"(cli\.[a-z0-9_.]+)"', source))
        used |= set(tray.QUIT_QUESTION_KEYS.values())
        self.assertGreaterEqual(len(used), 10)  # the catalog is actually wired up
        for key in sorted(used):
            self.assertIn(key, i18n._CLI_STRINGS, key)


class TestTheAnswersToQuitAreReadCorrectly(unittest.TestCase):
    """`quit_program` decides on the ANSWER, not on the request going out — so the two
    unhappy shapes of it are pinned without a server."""

    def test_a_refusal_that_is_not_about_a_run_is_not_a_question(self):
        asked = []
        with mock.patch.object(tray, "request_quit", return_value=(403, {})):
            ok = tray.quit_program(8756, "en",
                                   ask=lambda t, q: asked.append((t, q)) or True)
        self.assertFalse(ok)
        self.assertEqual(asked, [], "спросили про то, что не про прогон")

    def test_a_confirmed_quit_the_server_refuses_anyway_is_a_failure(self):
        answers = [(409, {"reason": ui.QUIT_RUN_IN_PROGRESS, "running": "process"}),
                   (409, {"reason": ui.QUIT_RUN_IN_PROGRESS, "running": "process"})]
        with mock.patch.object(tray, "request_quit", side_effect=answers):
            self.assertFalse(tray.quit_program(8756, "en", ask=lambda t, q: True))

    def test_a_dialog_that_cannot_be_drawn_answers_no(self):
        """Requirement 2 at its strictest: where nobody can be asked, nothing is
        interrupted. A `None` in `sys.modules` is what an absent module looks like to
        the `import` inside the function."""
        with mock.patch.dict("sys.modules", {"tkinter": None}):
            self.assertFalse(tray.ask_yes_no("title", "question"))

    def test_a_dialog_that_blows_up_answers_no_too(self):
        """No display, no window manager, a headless session: every one of them raises
        somewhere inside tkinter, and none of them may read as a yes."""
        import tkinter

        with mock.patch.object(tkinter, "Tk", side_effect=RuntimeError("no display")):
            self.assertFalse(tray.ask_yes_no("title", "question"))


if __name__ == "__main__":
    unittest.main()
