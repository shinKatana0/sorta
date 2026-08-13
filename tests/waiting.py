"""F240: how long a test waits for its own machinery, in one place.

A test that raises a local HTTP server waits for it twice — once for the answer to a
request, once in teardown for the serving thread to end — and both waits used to carry
a number typed at the call site, five seconds in 38 of them. A number there is an
assertion about SPEED inside a test about CORRECTNESS: nothing in
`test_garbage_flag_does_not_clear` is about milliseconds, but at 5 s a runner that was
busy elsewhere turns into a function that answered wrong.

The budget is therefore generous and it lives here. Generosity is free: the timeout is
spent only on a run that is failing anyway, so 30 s costs a green gate exactly what 5 s
cost it. Thirty is the number the one site that had already been forced up arrived at —
`test_ui_process_browse`, where five seconds under the load of the full suite had made
every gate a coin flip. `SORTA_TEST_HTTP_TIMEOUT` overrides it for a machine slower or
faster than the ones this was measured on.

The teardown wait is the same defect and not a second one. A join that gives up leaves
the server thread holding `test.db`, so `TemporaryDirectory.cleanup()` raises
`PermissionError [WinError 32]` and one failure is reported as a failure plus an error,
with the directory left on disk. `stop_server` is the order that avoids it.

What does NOT belong here: a timeout that is an argument of the product
(`tray.sorta_is_serving(port, timeout=...)`), a value under test (`nominatim_timeout`),
and the budget of a child process — `subprocess.run(..., timeout=600)` is about a
program that has to start Python again, not about our own threads.
"""
from __future__ import annotations

import http.client
import http.server
import json
import os
import threading
import urllib.error
import urllib.request
from typing import Any, NamedTuple

TIMEOUT_ENV = "SORTA_TEST_HTTP_TIMEOUT"
DEFAULT_TIMEOUT_S = 30.0


def timeout_s() -> float:
    """The wait budget in seconds, `SORTA_TEST_HTTP_TIMEOUT` first.

    Read on every call, not once at import: the variable is how a run is told to be
    patient, and a value read at collection time would ignore anything set after it.
    """
    override = os.environ.get(TIMEOUT_ENV, "").strip()
    return float(override) if override else DEFAULT_TIMEOUT_S


class Answer(NamedTuple):
    """What the local server said. A 4xx or a 5xx is an answer, not an exception."""

    status: int
    body: bytes
    headers: http.client.HTTPMessage

    @property
    def content_type(self) -> str:
        return self.headers.get("Content-Type", "")

    def json(self) -> Any:
        return json.loads(self.body)


def fetch(target: str | urllib.request.Request) -> Answer:
    """Send a GET to a URL, or a Request the caller has already built, and read it all.

    Only the STATUS is kept from being an exception. A refused, reset or timed-out
    connection still raises: that says the server is not there, which no assertion in
    this suite means to pass over.
    """
    try:
        with urllib.request.urlopen(target, timeout=timeout_s()) as resp:
            return Answer(resp.status, resp.read(), resp.headers)
    except urllib.error.HTTPError as exc:
        return Answer(exc.code, exc.read(), exc.headers)


def post_json(url: str, payload: object) -> Answer:
    """POST a JSON body the way the page sends it — F208: the content type is what
    gets it served at all."""
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    return fetch(request)


def wait_for(event: threading.Event) -> bool:
    """Wait for an event another thread of this process sets; False if it never came."""
    return event.wait(timeout_s())


def join_thread(thread: threading.Thread) -> None:
    """Wait for a thread to end. Says nothing if it does not — the caller asserts."""
    thread.join(timeout_s())


def stop_server(server: http.server.HTTPServer, thread: threading.Thread) -> None:
    """Stop a served-in-a-thread server: loop out, thread ended, socket closed.

    Call it before deleting the temporary directory, never after: until the thread is
    gone the handler may still hold the database open, and on Windows an open file is
    a directory that cannot be removed.
    """
    server.shutdown()
    join_thread(thread)
    server.server_close()
