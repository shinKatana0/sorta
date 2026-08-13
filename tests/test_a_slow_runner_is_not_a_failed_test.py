"""F240: a test waits through `tests/waiting.py`, and nothing in `tests/` waits its own way.

Two halves, and the first one is the point of the feature.

* **The watchdog.** `tests/` is read with `ast` and every WAIT is listed: a call to
  `urlopen`, `join`, `wait` or `create_connection` carrying a number, and a helper whose
  `timeout` parameter defaults to one. Each of them is a budget typed at a call site, and
  a budget at a call site is what made a busy runner look like a broken function.

  There is no list of excused files, because the three calls that legitimately keep a
  number differ by their FORM and not by where they live: `tray.sorta_is_serving(port,
  timeout=5)`, `tray.port_holder(port, timeout=5)` and `nominatim_timeout=5.0` hand the
  number to somebody else's function, which is the subject of those tests rather than
  the patience of this suite. The same distinction leaves `subprocess.run(...,
  timeout=600)` alone: a budget for a program that starts Python again is not a budget
  for our own threads, and 30 s would be a cut rather than a gift.

  Source-reading and not a grep, for the reason `test_no_console_nobody_asked_for.py`
  gives: `urlopen(req, timeout=5)` written in a comment is not a wait.

* **The helper.** What `waiting.py` promises — one budget, overridable; an HTTP status
  is an answer while a dead port is not; a teardown that really stops the server before
  anything is deleted.
"""
from __future__ import annotations

import ast
import http.server
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from tests import waiting

_SUITE = Path(__file__).resolve().parent

# What a test uses to wait for its OWN machinery: an answer from the server it started,
# a thread it started, an event that thread sets, a socket it is holding open.
_WAITS = frozenset({"urlopen", "join", "wait", "create_connection"})


def _is_number(node: ast.expr) -> bool:
    return (isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool))


class _Waits(ast.NodeVisitor):
    """Every wait in one file whose length is written down where it is used."""

    def __init__(self) -> None:
        self._scope: list[str] = []
        self.found: list[tuple[str, int, str]] = []

    def _where(self) -> str:
        return ".".join(self._scope) or "<module>"

    def _nested(self, node: ast.AST, name: str) -> None:
        self._scope.append(name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._nested(node, node.name)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._timeout_default(node.args, node.lineno, node.name)
        self._nested(node, node.name)

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._timeout_default(node.args, node.lineno, "<lambda>")
        self.generic_visit(node)

    def _timeout_default(self, args: ast.arguments, line: int, name: str) -> None:
        positional = [*args.posonlyargs, *args.args]
        pairs = [*zip(positional[len(positional) - len(args.defaults):], args.defaults),
                 *zip(args.kwonlyargs, args.kw_defaults)]
        for arg, default in pairs:
            if arg.arg == "timeout" and default is not None and _is_number(default):
                self.found.append((self._where(), line, f"{name}(timeout={default.value})"))

    def visit_Call(self, node: ast.Call) -> None:
        called = self._called(node.func)
        if called in _WAITS:
            numbers = [arg for arg in node.args if _is_number(arg)]
            numbers += [kw.value for kw in node.keywords
                        if kw.arg == "timeout" and _is_number(kw.value)]
            for number in numbers:
                self.found.append((self._where(), node.lineno,
                                   f"{called}(... {ast.unparse(number)})"))
        self.generic_visit(node)

    @staticmethod
    def _called(func: ast.expr) -> str | None:
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
        return None


def waits_in(source: str) -> list[tuple[str, int, str]]:
    """Where this source decides for itself how long to wait, and how long that is."""
    visitor = _Waits()
    visitor.visit(ast.parse(source))
    return visitor.found


def suite_sources() -> list[Path]:
    return sorted(_SUITE.rglob("*.py"))


def hardcoded_waits() -> dict[str, list[str]]:
    """File -> the waits in it that go round `waiting.py`."""
    out: dict[str, list[str]] = {}
    for path in suite_sources():
        found = waits_in(path.read_text(encoding="utf-8"))
        if found:
            out[path.name] = [f"{where} line {line}: {what}" for where, line, what in found]
    return out


class TestNothingInTheSuiteTimesItselfOut(unittest.TestCase):
    """The sentinel this whole feature exists for."""

    def test_every_wait_takes_its_budget_from_the_helper(self):
        self.assertEqual(
            hardcoded_waits(), {},
            "a number here is an assertion about the speed of the runner: take the "
            "budget from tests/waiting.py (timeout_s, fetch, post_json, wait_for, "
            "join_thread, stop_server)")

    def test_the_suite_really_goes_through_the_helper(self):
        """The other direction: a green watchdog over a suite that stopped making
        requests at all would look exactly the same."""
        users = [path.name for path in suite_sources()
                 if "from tests import waiting" in path.read_text(encoding="utf-8")]
        self.assertGreater(len(users), 30)
        self.assertIn("test_ui.py", users)


class TestTheWatchdogGoesRed(unittest.TestCase):
    """A watchdog nobody has seen fail is not a watchdog."""

    def test_the_call_the_feature_removed_is_found(self):
        source = ("def get(self):\n"
                  "    with urllib.request.urlopen(req, timeout=5) as resp:\n"
                  "        return resp.read()\n")
        self.assertEqual(waits_in(source), [("get", 2, "urlopen(... 5)")])

    def test_a_teardown_that_gives_up_on_its_thread_is_found(self):
        source = "def tearDown(self):\n    self.thread.join(timeout=5)\n"
        self.assertEqual(waits_in(source), [("tearDown", 2, "join(... 5)")])

    def test_a_positional_number_hides_nothing(self):
        source = "def go():\n    started.wait(30)\n    worker.join(10)\n"
        self.assertEqual([what for _where, _line, what in waits_in(source)],
                         ["wait(... 30)", "join(... 10)"])

    def test_a_helper_with_a_budget_of_its_own_is_found(self):
        source = "def _poll_until(get_status, predicate, timeout=5.0, interval=0.02):\n    pass\n"
        self.assertEqual(waits_in(source),
                         [("<module>", 1, "_poll_until(timeout=5.0)")])

    def test_a_timeout_handed_to_the_product_is_not_ours_to_own(self):
        """The form that distinguishes the three calls left alone — the number is an
        argument of the code under test, not the patience of the test."""
        source = ("def test_a_stranger_is_not_mistaken_for_us(self):\n"
                  "    self.assertFalse(tray.sorta_is_serving(self.port, timeout=5))\n"
                  "    tray.port_holder(self.port, timeout=5)\n"
                  "    GeoConfig(nominatim_timeout=5.0)\n")
        self.assertEqual(waits_in(source), [])

    def test_a_child_process_keeps_its_own_budget(self):
        source = "def run():\n    subprocess.run(cmd, timeout=600, check=False)\n"
        self.assertEqual(waits_in(source), [])

    def test_a_wait_that_asks_the_helper_is_what_the_rule_wants(self):
        source = ("def get(self):\n"
                  "    urllib.request.urlopen(req, timeout=waiting.timeout_s())\n"
                  "    self.thread.join(waiting.timeout_s())\n")
        self.assertEqual(waits_in(source), [])

    def test_a_wait_that_is_only_written_about_is_not_a_wait(self):
        """Why this reads the syntax tree: a number in a comment or a docstring must not
        go red, or the check becomes noise and noise gets switched off."""
        source = ('def helper():\n'
                  '    """It used to be urlopen(req, timeout=5) here."""\n'
                  '    # thread.join(timeout=5) was the other half\n'
                  '    return "join(timeout=5)"\n')
        self.assertEqual(waits_in(source), [])

    def test_the_real_suite_is_what_is_being_read(self):
        """A scanner pointed at nothing finds nothing and looks exactly like a green
        gate — the failure mode of every check that walks a directory."""
        names = [path.name for path in suite_sources()]
        self.assertGreater(len(names), 200)
        self.assertIn("waiting.py", names)
        self.assertIn("test_ui.py", names)


class TestTheBudget(unittest.TestCase):
    def test_it_is_generous_by_default(self):
        """Six times the five seconds it replaces, and free: see the module docstring
        of waiting.py."""
        with mock.patch.dict("os.environ"):
            os.environ.pop(waiting.TIMEOUT_ENV, None)
            self.assertEqual(waiting.timeout_s(), 30.0)
            self.assertEqual(waiting.DEFAULT_TIMEOUT_S, 30.0)

    def test_a_variable_set_to_nothing_is_not_a_budget_of_zero(self):
        with mock.patch.dict("os.environ", {waiting.TIMEOUT_ENV: "  "}):
            self.assertEqual(waiting.timeout_s(), 30.0)

    def test_a_machine_can_say_otherwise(self):
        with mock.patch.dict("os.environ", {waiting.TIMEOUT_ENV: "2.5"}):
            self.assertEqual(waiting.timeout_s(), 2.5)

    def test_it_is_read_again_on_every_wait(self):
        """A value frozen at import would ignore a variable set by the run that needs it."""
        with mock.patch.dict("os.environ", {waiting.TIMEOUT_ENV: "1"}):
            first = waiting.timeout_s()
        with mock.patch.dict("os.environ", {waiting.TIMEOUT_ENV: "2"}):
            self.assertEqual((first, waiting.timeout_s()), (1.0, 2.0))


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # the access log on stderr says nothing a failure here would need

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/missing":
            self._send(404, b'{"error": "no"}', "application/json")
        else:
            self._send(200, b"<html>hello</html>", "text/html; charset=utf-8")

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        answer = {"seen": json.loads(body), "type": self.headers["Content-Type"]}
        self._send(200, json.dumps(answer).encode("utf-8"), "application/json")


class TestTheHelperTalksToARealServer(unittest.TestCase):
    """Every claim below is about a socket, not about a mock of one."""

    def setUp(self):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def test_a_get_brings_back_status_body_and_type(self):
        answer = waiting.fetch(f"{self.base_url}/")
        self.assertEqual(answer.status, 200)
        self.assertEqual(answer.body, b"<html>hello</html>")
        self.assertIn("text/html", answer.content_type)

    def test_an_error_status_is_an_answer(self):
        answer = waiting.fetch(f"{self.base_url}/missing")
        self.assertEqual(answer.status, 404)
        self.assertEqual(answer.json(), {"error": "no"})

    def test_a_post_carries_the_json_and_the_content_type(self):
        answer = waiting.post_json(f"{self.base_url}/api/x", {"file_ids": [1, 2]})
        self.assertEqual(answer.status, 200)
        self.assertEqual(answer.json(),
                         {"seen": {"file_ids": [1, 2]}, "type": "application/json"})

    def test_a_prepared_request_is_sent_as_it_is(self):
        request = urllib.request.Request(f"{self.base_url}/api/x", data=b"{}",
                                         method="POST", headers={"Content-Type": "text/plain"})
        self.assertEqual(waiting.fetch(request).json()["type"], "text/plain")

    def test_a_server_that_is_not_there_still_raises(self):
        """The one thing the helper must NOT swallow: an unanswered connection is the
        server being absent, and no assertion in this suite means to pass over that."""
        self.server.shutdown()
        self.server.server_close()
        with self.assertRaises(urllib.error.URLError):
            waiting.fetch(f"{self.base_url}/")


class TestWaitingForAThread(unittest.TestCase):
    def test_an_event_that_arrives_is_a_yes(self):
        event = threading.Event()
        threading.Thread(target=event.set, daemon=True).start()
        self.assertTrue(waiting.wait_for(event))

    def test_an_event_that_never_comes_is_a_no_and_not_a_raise(self):
        with mock.patch.dict("os.environ", {waiting.TIMEOUT_ENV: "0.01"}):
            self.assertFalse(waiting.wait_for(threading.Event()))

    def test_a_thread_is_waited_for(self):
        done = threading.Event()
        thread = threading.Thread(target=lambda: done.wait(waiting.timeout_s()),
                                  daemon=True)
        thread.start()
        done.set()
        waiting.join_thread(thread)
        self.assertFalse(thread.is_alive())

    def test_a_thread_that_hangs_is_not_a_teardown_that_raises(self):
        """A teardown that raises replaces the failure the test was about."""
        thread = threading.Thread(
            target=lambda: threading.Event().wait(waiting.DEFAULT_TIMEOUT_S), daemon=True)
        thread.start()
        with mock.patch.dict("os.environ", {waiting.TIMEOUT_ENV: "0.01"}):
            waiting.join_thread(thread)
        self.assertTrue(thread.is_alive())


class TestStoppingTheServer(unittest.TestCase):
    """The half that produced `PermissionError [WinError 32]`: the directory goes only
    after the thread that was serving out of it."""

    def test_the_thread_is_gone_before_the_directory_is(self):
        tmp = tempfile.TemporaryDirectory()
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.assertEqual(waiting.fetch(f"http://127.0.0.1:{server.server_port}/").status,
                         200)
        waiting.stop_server(server, thread)
        self.assertFalse(thread.is_alive())
        tmp.cleanup()
        self.assertFalse(Path(tmp.name).exists())

    def test_the_socket_is_closed_too(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_port
        waiting.stop_server(server, thread)
        with mock.patch.dict("os.environ", {waiting.TIMEOUT_ENV: "1"}):
            with self.assertRaises(urllib.error.URLError):
                waiting.fetch(f"http://127.0.0.1:{port}/")


if __name__ == "__main__":
    unittest.main()
