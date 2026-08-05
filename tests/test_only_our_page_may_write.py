"""F208: only our own page may write into the local server, and only an absolute path
may reach exiftool.

Both halves of the feature are the same sentence: check it at the BOUNDARY, do not lean
on an invariant that holds somewhere else.

The web half. `127.0.0.1` keeps the network out and keeps out nothing else — the user's
browser is exactly the program that visits somebody else's page and this port in one
session. A page in another tab could POST here with `Content-Type: text/plain`, which the
CORS rules call a "simple" request and the browser therefore sends WITHOUT asking
permission first; the answer stayed unreadable to that page, but `/api/sort` had already
moved the files. Requiring `application/json` forces the browser to ask first with an
`OPTIONS` preflight that this server never grants, so the request is not sent at all.

The exiftool half. Paths go to exiftool with no `--` in front of them (it has no such
separator), so a file named `-config` would be read as an OPTION — and `-config` loads a
Perl file. Nothing can be named that today because the indexer resolves its root, but that
invariant was checked nowhere and covered by no test.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import sorta.exif as exif
from sorta import ui

from tests.test_exif_flags import FakeExifTool
from tests.test_ui import UiServerTestBase
from tests.test_ui_master_switch import _BODIES, _post_routes

# A body per POST route, so that a refusal means something: the request refused below is
# one the server would otherwise have carried out. `_BODIES` covers every route the busy
# guard refuses; the four the busy guard lets through on purpose (cancelling, and the
# folder picker) are added here — they act too, and a foreign page must not be able to
# stop somebody's layout either.
_ALL_BODIES: dict[str, object] = {
    **_BODIES,
    "/api/process/cancel": {},
    "/api/sort/cancel": {},
    "/api/undo/cancel": {},
    "/api/browse": {},
}

_FOREIGN_ORIGIN = "http://evil.example"


class TestTheRuleItself(unittest.TestCase):
    """`_post_refusal` on its own — the decision, without a socket around it."""

    def test_plain_text_is_refused(self):
        self.assertEqual(ui._post_refusal("text/plain", None, "127.0.0.1:8756"),
                         ui.REFUSED_CONTENT_TYPE)

    def test_the_other_two_simple_types_are_refused_as_well(self):
        """The three types a browser may send without asking permission. `text/plain` is
        the one an attacking page would pick, the other two are the same door."""
        for raw in ("multipart/form-data", "application/x-www-form-urlencoded"):
            with self.subTest(content_type=raw):
                self.assertEqual(ui._post_refusal(raw, None, "127.0.0.1:8756"),
                                 ui.REFUSED_CONTENT_TYPE)

    def test_a_missing_content_type_is_refused(self):
        self.assertEqual(ui._post_refusal(None, None, "127.0.0.1:8756"),
                         ui.REFUSED_CONTENT_TYPE)

    def test_json_passes_with_and_without_parameters(self):
        # what fetch() sends is `application/json`; a charset parameter and a different
        # case are the same media type and must not be turned into a refusal.
        for raw in ("application/json", "application/json; charset=utf-8",
                    "Application/JSON;charset=UTF-8", "  application/json  "):
            with self.subTest(content_type=raw):
                self.assertIsNone(ui._post_refusal(raw, None, "127.0.0.1:8756"))

    def test_a_type_that_merely_starts_the_same_is_not_json(self):
        self.assertEqual(
            ui._post_refusal("application/json-evil", None, "127.0.0.1:8756"),
            ui.REFUSED_CONTENT_TYPE)

    def test_a_foreign_origin_is_refused(self):
        self.assertEqual(
            ui._post_refusal("application/json", _FOREIGN_ORIGIN, "127.0.0.1:8756"),
            ui.REFUSED_ORIGIN)

    def test_our_own_origin_passes(self):
        self.assertIsNone(ui._post_refusal("application/json",
                                           "http://127.0.0.1:8756", "127.0.0.1:8756"))

    def test_an_absent_origin_passes(self):
        """The second line, not the first: the header is not always sent, so its absence
        cannot be read as an accusation — that is what the content type is for."""
        self.assertIsNone(ui._post_refusal("application/json", None, "127.0.0.1:8756"))

    def test_a_null_origin_is_foreign(self):
        """A sandboxed frame and a `file://` page both say `null` — neither is our page."""
        self.assertEqual(ui._post_refusal("application/json", "null", "127.0.0.1:8756"),
                         ui.REFUSED_ORIGIN)

    def test_the_same_host_on_another_port_is_foreign(self):
        """Another program on another port of this machine is somebody else."""
        self.assertEqual(
            ui._post_refusal("application/json", "http://127.0.0.1:9999", "127.0.0.1:8756"),
            ui.REFUSED_ORIGIN)

    def test_the_content_type_is_decided_before_the_origin(self):
        """Both wrong -> the refusal names the one that closes the class, so a report of
        it says what to fix."""
        self.assertEqual(ui._post_refusal("text/plain", _FOREIGN_ORIGIN, "127.0.0.1:8756"),
                         ui.REFUSED_CONTENT_TYPE)

    def test_every_refusal_code_has_a_sentence(self):
        self.assertEqual(set(ui._POST_REFUSAL_DETAIL),
                         {ui.REFUSED_CONTENT_TYPE, ui.REFUSED_ORIGIN})
        for code, detail in ui._POST_REFUSAL_DETAIL.items():
            with self.subTest(code=code):
                self.assertTrue(detail.strip())


class PostingTestBase(UiServerTestBase):
    """A live server, and a POST whose headers the test chooses."""

    def setUp(self):
        super().setUp()
        # the folder picker is one of the routes walked below and it opens a native
        # dialog — refused or not, no test may make one appear.
        patcher = mock.patch.object(ui, "_browse_for_folder", return_value="")
        patcher.start()
        self.addCleanup(patcher.stop)

    def post(self, path: str, data: object, *, content_type: str = "application/json",
             origin: str | None = None) -> tuple[int, dict]:
        headers = {"Content-Type": content_type}
        if origin is not None:
            headers["Origin"] = origin
        body = json.dumps(data).encode("utf-8")
        # One POST, retried once when the SOCKET fails rather than the server. This suite
        # walks every writing route over a fresh connection each time, and a threading
        # server and its client can disagree about who closes first: the answer is written,
        # the handler returns, the socket goes away, and the client sees
        # `ConnectionAbortedError: [WinError 10053]` with no reply at all. Enough attempts
        # in a row and it happens somewhere -- caught on this file's first red gate, on a
        # route the failure had nothing to do with, which is what a race of the transport
        # looks like from here.
        #
        # The retry does NOT soften what the test claims: it re-opens the CONNECTION, not
        # the assertion. A route that really answered wrong answers wrong the second time
        # too.
        for attempt in (1, 2):
            req = urllib.request.Request(
                f"{self.base_url}{path}", data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.status, json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                return exc.code, json.loads(exc.read())
            except (ConnectionError, TimeoutError):
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")


class TestEveryPostRouteRefusesAForeignContentType(PostingTestBase):
    """Brief test 1: every writing route, walked from the dispatcher rather than named.

    A layout is faked as running throughout, so that a route which somehow got past the
    refusal would still answer 409 instead of doing the thing — the suite must not sort a
    collection to find out that this check is missing.
    """

    def setUp(self):
        super().setUp()
        state = ui._SortState()
        self.assertTrue(state.try_start())
        patcher = mock.patch.object(ui, "_SortState", return_value=state)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.start_server()

    def test_the_body_list_covers_the_dispatcher(self):
        self.assertEqual(set(_ALL_BODIES), _post_routes(),
                         "новый POST-маршрут: добавьте тело в _ALL_BODIES, иначе он не "
                         "проверяется на чужой Content-Type")

    def test_plain_text_is_refused_on_every_route(self):
        for path in sorted(_ALL_BODIES):
            with self.subTest(route=path):
                status, resp = self.post(path, _ALL_BODIES[path],
                                         content_type="text/plain")
                self.assertEqual(status, 403, f"{path} ответил {status}: {resp}")
                self.assertEqual(resp["reason"], ui.REFUSED_CONTENT_TYPE)

    def test_json_reaches_the_route_as_before(self):
        """Brief test 2: the refusal is about the header and nothing else — the same
        bodies with the right type get the answer the route would give (here 409, since
        a layout is running)."""
        for path in sorted(ui.BUSY_REFUSED_ROUTES):
            with self.subTest(route=path):
                status, resp = self.post(path, _ALL_BODIES[path])
                self.assertEqual(status, 409, f"{path} ответил {status}: {resp}")


class TestTheRefusalSaysWhy(PostingTestBase):
    """Brief test 5: a code and a reason, not a bare 400.

    Somebody whose browser extension or own script stops working has to be able to read
    what happened — every other refusal in this server says why.
    """

    def setUp(self):
        super().setUp()
        self.start_server()

    def test_the_content_type_refusal_names_the_type_it_wants(self):
        status, resp = self.post("/api/overrides", _BODIES["/api/overrides"],
                                 content_type="text/plain")
        self.assertEqual(status, 403)
        self.assertEqual(resp["reason"], ui.REFUSED_CONTENT_TYPE)
        self.assertIn("error", resp)
        self.assertIn("application/json", resp["detail"])

    def test_the_origin_refusal_says_it_is_about_the_origin(self):
        status, resp = self.post("/api/overrides", _BODIES["/api/overrides"],
                                 origin=_FOREIGN_ORIGIN)
        self.assertEqual(status, 403)
        self.assertEqual(resp["reason"], ui.REFUSED_ORIGIN)
        self.assertTrue(resp["detail"].strip())


class TestOriginIsTheSecondLine(PostingTestBase):
    """Brief test 3: a foreign origin is refused, our own and an absent one pass."""

    def setUp(self):
        super().setUp()
        self.start_server()

    def test_a_foreign_origin_is_refused(self):
        status, resp = self.post("/api/overrides", _BODIES["/api/overrides"],
                                 origin=_FOREIGN_ORIGIN)
        self.assertEqual(status, 403)
        self.assertEqual(resp["reason"], ui.REFUSED_ORIGIN)

    def test_our_own_origin_is_served(self):
        status, resp = self.post("/api/overrides", _BODIES["/api/overrides"],
                                 origin=self.base_url)
        self.assertEqual(status, 200, resp)

    def test_a_request_without_an_origin_is_served(self):
        status, resp = self.post("/api/overrides", _BODIES["/api/overrides"])
        self.assertEqual(status, 200, resp)


class TestARefusedRequestChangesNothing(PostingTestBase):
    """The acceptance criterion, put as a state check: a page in the same browser can
    move no file and change no setting.

    Read on the server side of the same request: the refusal happens before the body is
    parsed, so the mark is not stored and the trash path is never entered.
    """

    def setUp(self):
        super().setUp()
        self.start_server()

    def test_a_correction_is_not_written(self):
        status, _resp = self.post("/api/overrides", {"file_ids": [1], "action": "exclude"},
                                  content_type="text/plain")
        self.assertEqual(status, 403)
        self.assertIsNone(
            self.conn.execute("SELECT 1 FROM manual_overrides").fetchone())

    def test_nothing_goes_to_the_trash(self):
        with mock.patch("sorta.ui.common.send_to_trash") as mock_trash:
            status, _resp = self.post("/api/photos/trash", {"file_ids": [1]},
                                      content_type="text/plain")
        self.assertEqual(status, 403)
        mock_trash.assert_not_called()

    def test_a_setting_is_not_saved(self):
        with mock.patch("sorta.ui.save_setting") as mock_save:
            status, _resp = self.post("/api/settings", {"vlm.workers": 3},
                                      content_type="text/plain")
        self.assertEqual(status, 403)
        mock_save.assert_not_called()


class TestGetIsUntouched(UiServerTestBase):
    """Brief test 4. Thumbnails and previews are ordinary browser requests: they carry no
    content type and must not start to need one. What keeps a foreign page from reading
    an answer here is the browser's own origin policy, and that has not changed."""

    def get_with_headers(self, path: str, headers: dict[str, str]) -> int:
        req = urllib.request.Request(f"{self.base_url}{path}", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_the_page_and_the_read_routes_answer_without_a_content_type(self):
        self.start_server()
        for path in ("/", "/api/dupes", "/api/overview", "/api/process/status"):
            with self.subTest(path=path):
                status, _body, _ctype = self.get(path)
                self.assertEqual(status, 200)

    def test_a_read_is_not_refused_over_its_headers(self):
        self.start_server()
        self.assertEqual(
            self.get_with_headers("/api/dupes", {"Origin": _FOREIGN_ORIGIN,
                                                 "Content-Type": "text/plain"}),
            200)


class TestTheServerGrantsNoPreflight(UiServerTestBase):
    """Why requiring the content type works at all: the browser has to ask first, and
    this server answers no permission — no `do_OPTIONS`, and no CORS header anywhere."""

    def test_options_is_not_answered_with_permission(self):
        self.start_server()
        req = urllib.request.Request(f"{self.base_url}/api/sort", method="OPTIONS")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status, headers = resp.status, resp.headers
        except urllib.error.HTTPError as exc:
            status, headers = exc.code, exc.headers
        self.assertNotEqual(status, 200)
        self.assertIsNone(headers.get("Access-Control-Allow-Origin"))

    def test_no_route_ever_sends_a_cross_origin_header(self):
        """Read off the sources, because it is a property of the WHOLE server and not of
        the one route the request above happened to name: a single
        `Access-Control-Allow-Origin` anywhere would hand back the permission the
        preflight was refused."""
        package = Path(ui.__file__).parent
        for path in sorted(package.rglob("*.py")) + sorted(package.parent.rglob("*.js")):
            with self.subTest(path=path.name):
                self.assertNotIn("Access-Control-Allow",
                                 path.read_text(encoding="utf-8"))


class TestRelativePathNeverReachesExiftool(unittest.TestCase):
    """The second half of the brief: the check on the way into exiftool.

    The fake exiftool records the arguments it was given and marks every launch, so
    "did not reach it" is read off the binary's own side rather than from the exception.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "a.jpg"
        self.fake = FakeExifTool(self.root, {"a.jpg": {"Make": "samsung"}})

    def tearDown(self):
        self.fake.restore()
        self.tmp.cleanup()

    def test_the_session_refuses_a_relative_path(self):
        with self.assertRaises(exif.UnsafeExifPath):
            exif._pool.sessions(1)[0].read([Path("sub/a.jpg")])
        self.assertEqual(self.fake.last_args(), [])
        self.assertEqual(self.fake.launches(), 0)

    def test_the_one_shot_call_refuses_a_relative_path(self):
        with self.assertRaises(exif.UnsafeExifPath):
            exif.read_batch_exiftool([Path("sub/a.jpg")])
        self.assertEqual(self.fake.last_args(), [])
        self.assertEqual(self.fake.launches(), 0)

    def test_a_name_that_looks_like_an_option_does_not_reach_the_binary(self):
        """`-config` is the argument that made this worth checking: exiftool would read it
        as the option that loads a Perl file, i.e. runs code."""
        for path in (Path("-config"), Path("-config.jpg")):
            with self.subTest(path=str(path)):
                with self.assertRaises(exif.UnsafeExifPath):
                    exif.read_batch([path])
                self.assertEqual(self.fake.last_args(), [])
                self.assertEqual(self.fake.launches(), 0)

    def test_one_relative_path_refuses_the_whole_batch(self):
        """Not filtered out quietly: a caller that got here with a relative path is
        mistaken about something, and dropping one file would hide that."""
        with self.assertRaises(exif.UnsafeExifPath):
            exif.read_batch([self.path, Path("a.jpg")])
        self.assertEqual(self.fake.launches(), 0)

    def test_the_refusal_is_a_value_error_for_an_ordinary_caller(self):
        with self.assertRaises(ValueError):
            exif.read_batch_exiftool([Path("a.jpg")])

    def test_an_absolute_path_still_reads(self):
        out = exif.read_batch([self.path])
        self.assertEqual(out[str(self.path.resolve())].make, "samsung")
        self.assertIn(str(self.path), self.fake.last_args())


if __name__ == "__main__":
    unittest.main()
