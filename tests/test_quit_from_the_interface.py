"""F209: «Завершить работу» lives in the interface, and a run cannot be lost by pressing it.

The feature exists because the possibility was in the wrong place. A tray icon was the
obvious home for "quit", and on Linux there may be no tray at all — GNOME removed it in
3.26, so on Ubuntu and Fedora it appears only through an extension somebody installs by
hand. An action in the INTERFACE works everywhere the product works, and it is checkable
by an ordinary route test, which the behaviour of an icon in somebody else's desktop is
not.

What is pinned here:

* the main one — while a run is going the route REFUSES (409) and the run carries on.
  Asked past the interface, over a socket, because that is where the rule has to live: a
  dialog the page draws forbids nothing (F133);
* with an explicit confirmation the run is interrupted through the flag
  `/api/process/cancel` already sets, and the server closes;
* with nothing running the answer goes out FIRST and the server stops after it — the
  port is free afterwards, proven by starting a second server on the very same port;
* the connection this server read the index through is closed, and the database it was
  reading is intact and journal-free after the exit;
* the captions exist in all three languages, and the page carries the button.
"""
from __future__ import annotations

import json
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

from sorta import ui
from sorta.config import Config
from sorta.db import connect


def _free_port() -> int:
    """A port the OS has just confirmed is free — the one the exit has to give back."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post(url: str, payload: object) -> tuple[int, dict]:
    """POST as the page sends it (F208: the content type is what gets it served)."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class ServedProgramTestBase(unittest.TestCase):
    """A real `ui.serve` on a real port, in a thread of its own.

    `build_server` is not enough for this feature: what is being checked is the EXIT —
    the serve loop returning, the socket closing, the connection closing — and all three
    of them live in `serve`, not in the handler. The connection is opened on the serving
    thread because sqlite3 hands a connection to one thread only, which is also where
    `serve` closes it.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src_dir = self.root / "src"
        self.src_dir.mkdir()
        self.cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                          raw={})
        # A row written before the server starts: the exit has to leave it readable.
        setup_conn = connect(self.cfg.database)
        setup_conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
               VALUES (?, 1, 0, 'jpg', 'photo', '2026-01-01')""",
            (str(self.src_dir / "a.jpg"),))
        setup_conn.commit()
        setup_conn.close()
        self.port = _free_port()
        self.conn: sqlite3.Connection | None = None
        self.thread: threading.Thread | None = None
        self.serve_error: BaseException | None = None
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.addCleanup(self._stop_if_still_serving)

    # --- the program under test ---------------------------------------------

    def start_program(self) -> None:
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = self._serve_in_a_thread()
        self._wait_until_answering()

    def _serve_in_a_thread(self) -> threading.Thread:
        """`serve` on a thread, with whatever it raises kept for the assertions.

        A start that fails (a port still held by the previous server, say) otherwise
        dies inside the thread and shows up here only as "nothing is answering", which
        names neither the reason nor the line.
        """
        opened = threading.Event()

        def run() -> None:
            try:
                self.conn = connect(self.cfg.database)
                opened.set()
                ui.serve(self.cfg, self.conn, port=self.port, open_browser=False)
            except BaseException as exc:  # noqa: BLE001 — re-raised by the assertions
                self.serve_error = exc
                opened.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.assertTrue(opened.wait(30), "серверный поток не стартовал")
        return thread

    def _wait_until_answering(self) -> None:
        # Generous on purpose: `serve` opens with `log_environment`, which probes the
        # GPU, and the first call of that in a test process imports torch (~7 s).
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.serve_error is not None:
                raise self.serve_error
            try:
                with urllib.request.urlopen(f"{self.base_url}/api/config", timeout=2):
                    return
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        self.fail(f"сервер не отвечает на порту {self.port}")

    def _stop_if_still_serving(self) -> None:
        """Leave no server (and no open database file) behind after a test.

        The join comes first: a test that already asked the program to quit is usually
        a fraction of a second away from it, and a second `/api/quit` sent into a server
        whose loop is winding down is answered by a reset socket rather than by JSON —
        which is a fact about this cleanup, not about the route.
        """
        thread = self.thread
        if thread is None:
            return
        thread.join(timeout=2)
        if thread.is_alive():
            try:
                _post(f"{self.base_url}/api/quit", {"confirm": True})
            except OSError:
                pass
            thread.join(timeout=10)

    def quit(self, **body: object) -> tuple[int, dict]:
        return _post(f"{self.base_url}/api/quit", body)

    def assert_program_ended(self) -> None:
        """The serve loop returned and `serve` ran its `finally` — i.e. it really quit."""
        self.thread.join(timeout=10)
        self.assertFalse(self.thread.is_alive(), "серверный поток не завершился")

    def assert_still_serving(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/api/config", timeout=5) as resp:
            self.assertEqual(resp.status, 200)


class _CountingRun:
    """A stand-in for the pipeline thread: it counts, through the real state object.

    The progress it reports goes through `_ProcessState.set_progress`, which is where
    cancellation is raised from — so this thread stops exactly the way a real stage
    stops, and "the run carried on" is a number that kept moving rather than a flag
    nobody read.
    """

    def __init__(self, state: ui._ProcessState) -> None:
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


class RunningProgramTestBase(ServedProgramTestBase):
    """The same server, with a run genuinely in flight on its `_ProcessState`."""

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


class TestARunIsNotLostByPressingQuit(RunningProgramTestBase):
    """Brief test 1, the main one. Five hours of counting must survive the button."""

    def test_the_route_refuses_while_a_run_is_going(self):
        status, resp = self.quit()
        self.assertEqual(status, 409, resp)
        self.assertEqual(resp["reason"], ui.QUIT_RUN_IN_PROGRESS)
        self.assertEqual(resp["running"], "process")

    def test_the_run_carries_on_after_the_refusal(self):
        self.quit()
        self.assertFalse(self.state.cancel_requested(),
                         "отказ не должен трогать флаг отмены")
        self.assertTrue(self.run.progressed(), "прогон встал после отказа")
        self.assertTrue(self.state.snapshot()["running"])

    def test_the_server_is_still_there_after_the_refusal(self):
        self.quit()
        self.assert_still_serving()
        self.assertTrue(self.thread.is_alive())

    def test_a_body_that_did_not_ask_to_confirm_is_not_a_confirmation(self):
        """Absent, an explicit no, and anything that is merely truthy: the confirmation
        is the literal JSON `true` and nothing else, because every other reading makes
        `{"confirm": "no"}` end a five-hour pass."""
        for body in ({}, {"confirm": False}, {"confirm": "no"}, {"confirm": 1},
                     {"other": 1}):
            with self.subTest(body=body):
                status, resp = _post(f"{self.base_url}/api/quit", body)
                self.assertEqual(status, 409, resp)
        self.assertFalse(self.state.cancel_requested())
        self.assertTrue(self.thread.is_alive())

    def test_the_refusal_is_the_servers_and_not_the_pages(self):
        """F133: the request is sent past the interface — over a socket, with no page
        involved — and meets the same refusal a person would."""
        self.assertIn("/api/quit", ui.BUSY_REFUSED_ROUTES)
        status, _resp = self.quit()
        self.assertEqual(status, 409)


class TestAConfirmedQuitInterruptsTheRun(RunningProgramTestBase):
    """Brief test 2: with the confirmation, the run stops and the program closes."""

    def test_the_run_is_interrupted_through_the_existing_flag(self):
        status, resp = self.quit(confirm=True)
        self.assertEqual(status, 200, resp)
        self.assertEqual(resp["cancelled"], "process")
        self.assertTrue(self.run.cancelled.wait(5), "прогон не был прерван")
        self.assertFalse(self.state.snapshot()["running"])

    def test_the_server_closes(self):
        self.quit(confirm=True)
        self.assert_program_ended()

    def test_the_answer_says_the_program_is_closing(self):
        _status, resp = self.quit(confirm=True)
        self.assertTrue(resp["ok"])
        self.assertTrue(resp["quitting"])


class TestALayoutAndARollbackAreProtectedToo(ServedProgramTestBase):
    """A pass is not the only thing that takes hours: a layout moves 220 GB and a
    rollback puts it back. The refusal names which one it is, because what the person is
    about to lose has a name."""

    def _serve_with(self, attr: str, state) -> None:
        patcher = mock.patch.object(ui, attr, return_value=state)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.start_program()

    def test_a_layout_in_flight_is_named_in_the_refusal(self):
        state = ui._SortState()
        self.assertTrue(state.try_start())
        self._serve_with("_SortState", state)
        status, resp = self.quit()
        self.assertEqual(status, 409, resp)
        self.assertEqual(resp["running"], "sort")
        self.assertFalse(state.cancel_requested())

    def test_a_rollback_in_flight_is_named_in_the_refusal(self):
        state = ui._UndoState()
        self.assertTrue(state.try_start())
        self._serve_with("_UndoState", state)
        status, resp = self.quit()
        self.assertEqual(status, 409, resp)
        self.assertEqual(resp["running"], "undo")

    def test_a_confirmed_quit_cancels_the_layout_and_closes(self):
        state = ui._SortState()
        self.assertTrue(state.try_start())
        self._serve_with("_SortState", state)
        status, resp = self.quit(confirm=True)
        self.assertEqual(status, 200, resp)
        self.assertTrue(state.cancel_requested())
        self.assert_program_ended()


class TestQuittingWithNothingRunning(ServedProgramTestBase):
    """Brief test 3: the answer goes out, and only then does the program stop."""

    def setUp(self):
        super().setUp()
        self.start_program()

    def test_the_answer_arrives_before_the_shutdown(self):
        """The point of the whole ordering: a severed connection reads as a crash, and a
        person who pressed «Завершить работу» has to see that it worked."""
        status, resp = self.quit()
        self.assertEqual(status, 200, resp)
        self.assertTrue(resp["ok"])
        self.assertTrue(resp["quitting"])
        self.assertIsNone(resp["cancelled"])

    def test_the_program_ends(self):
        self.quit()
        self.assert_program_ended()

    def test_the_port_is_free_afterwards(self):
        """Proven the way a person proves it: by starting the program again on the very
        same port. A socket left in the server would make this the RuntimeError
        `serve` raises for a busy port."""
        self.quit()
        self.assert_program_ended()
        # The second program, on the same port: a socket the first one had not given
        # back would come out of `serve` as the RuntimeError it raises for a busy port.
        self.thread = self._serve_in_a_thread()
        self._wait_until_answering()
        self.assert_still_serving()


class TestTheExitLeavesTheIndexAlone(ServedProgramTestBase):
    """Brief test 4: the connection is closed and the database is whole."""

    def setUp(self):
        super().setUp()
        self.start_program()

    def test_the_connection_is_closed(self):
        self.quit()
        self.assert_program_ended()
        with self.assertRaises(sqlite3.ProgrammingError):
            self.conn.execute("SELECT 1")

    def test_the_index_is_readable_and_unchanged(self):
        self.quit()
        self.assert_program_ended()
        conn = connect(self.cfg.database)
        try:
            rows = conn.execute("SELECT path FROM files").fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        finally:
            conn.close()

    def test_no_journal_is_left_behind(self):
        """A rollback journal still on disk is what an interrupted write looks like."""
        self.quit()
        self.assert_program_ended()
        for suffix in ("-journal", "-wal"):
            leftover = self.cfg.database.with_name(self.cfg.database.name + suffix)
            with self.subTest(suffix=suffix):
                self.assertFalse(leftover.exists(), f"{leftover.name} остался на диске")


class TestCtrlCStillEndsTheSameWay(ServedProgramTestBase):
    """The terminal is not changed by this feature — and it gains the same clean exit,
    because both ways out leave through the one `finally`."""

    def test_the_connection_is_closed_on_a_keyboard_interrupt(self):
        conn = connect(self.cfg.database)
        with mock.patch.object(ui, "build_server") as build:
            build.return_value = mock.MagicMock(server_port=self.port)
            build.return_value.serve_forever.side_effect = KeyboardInterrupt
            ui.serve(self.cfg, conn, port=self.port, open_browser=False)
        build.return_value.server_close.assert_called_once()
        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


class TestTheCaptions(unittest.TestCase):
    """Brief test 5: three languages, and the page really carries the button."""

    KEYS = ("quit_button", "quit_title", "quit_running_confirm",
            "quit_sort_running_confirm", "quit_undo_running_confirm",
            "quit_done", "quit_failed")

    def test_every_caption_exists_in_all_three_languages(self):
        for key in self.KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} пуст")

    def test_the_button_is_in_the_header_and_not_among_the_frequent_actions(self):
        """Visible, but not on the way to anything: it is pressed once a day."""
        html = ui._render_index_html("ru")
        header = html[html.index('class="header-bar"'):html.index('class="tabs"')]
        self.assertIn('id="quit-btn"', header)
        self.assertIn(ui._UI_STRINGS["quit_button"]["ru"], header)

    def test_every_thing_that_can_be_running_has_a_question_of_its_own(self):
        """The vocabulary of the answer and the questions the page asks are one list.

        A fourth long operation added tomorrow would come back in `running` with no
        caption behind it, and the page would fall through to "could not quit" — which
        is the one wrong thing to say about a refusal that was protecting the work.
        """
        html = ui._render_index_html("ru")
        block = html[html.index("var QUIT_CONFIRM = {"):]
        block = block[:block.index("};")]
        mapping = dict(re.findall(r'(\w+): "(quit_\w+)"', block))
        self.assertEqual(set(mapping), set(ui.QUIT_RUNNING_NAMES))
        for name, key in sorted(mapping.items()):
            with self.subTest(running=name):
                self.assertEqual(set(ui._UI_STRINGS[key]), {"ru", "en", "ja"})

    def test_the_page_asks_before_it_confirms(self):
        """The client sends `confirm` only after the person answered the question the
        409 named — one refusal, one question, one second request."""
        html = ui._render_index_html("ru")
        self.assertIn('postJson("/api/quit", { confirm: !!confirmed })', html)
        self.assertIn('resp.reason === "run_in_progress"', html)
        self.assertIn("window.confirm(I18N[key])", html)

    def test_the_dead_page_says_the_program_is_gone(self):
        html = ui._render_index_html("ru")
        self.assertIn('id="quit-done"', html)
        self.assertIn(ui._UI_STRINGS["quit_done"]["ru"], html)


if __name__ == "__main__":
    unittest.main()
