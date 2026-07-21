"""F51: a readable toggle layout on the "Process" tab + POST /api/browse
(a native folder-picker dialog via a separate subprocess — tkinter is not
thread-safe, and the ThreadingHTTPServer handler is not on the main thread)."""
from __future__ import annotations

import json
import subprocess
import unittest
from unittest import mock

from sorta import ui

from tests.test_ui import UiServerTestBase


class TestProcessControlsLayout(UiServerTestBase):
    def test_hints_sit_next_to_their_own_checkbox_not_at_the_end(self):
        # .process-toggle-hint (flex-basis:100%) used to slide to the end of
        # .process-controls, after ALL buttons — detached from the checkboxes.
        # Now each hint must go RIGHT after its checkbox/label, before the other
        # buttons ("Process"/"Cancel"/"Start over").
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")

        deep_checkbox = html.index('id="process-deep-checkbox"')
        deep_hint = html.index("Slower; requires")
        geo_checkbox = html.index('id="process-geo-online-checkbox"')
        geo_hint = html.index("More accurate place names")
        start_btn = html.index('id="process-start-btn"')

        self.assertLess(deep_checkbox, deep_hint)
        self.assertLess(deep_hint, geo_checkbox)
        self.assertLess(geo_checkbox, geo_hint)
        # both hints come before the action buttons, not after them
        self.assertLess(deep_hint, start_btn)
        self.assertLess(geo_hint, start_btn)

    def test_toggle_ids_and_action_buttons_still_present(self):
        # F50 ids are not broken by the markup rework.
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        for element_id in (
            "process-source-dir", "process-deep-checkbox",
            "process-geo-online-checkbox", "process-start-btn",
            "process-cancel-btn", "process-reset-btn",
        ):
            self.assertIn(f'id="{element_id}"', html)
        # the floating flex-basis:100% class is no longer used as a standalone
        # "trailing" span — but the class itself (the hint style) remains
        self.assertIn("process-toggle-hint", html)

    def test_no_external_resources_u1(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)


class TestBrowseButtonHtml(UiServerTestBase):
    def test_browse_button_present_with_id(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="process-browse-btn"', html)
        self.assertIn("/api/browse", html)

    def test_browse_button_i18n_ru_en_ja(self):
        for lang, text in (("ru", "Обзор…"), ("en", "Browse…"), ("ja", "参照…")):
            self.cfg.raw = {"language": lang}
            self.start_server()
            _status, body, _ctype = self.get("/")
            html = body.decode("utf-8")
            self.assertIn(text, html, msg=f"lang={lang}")
            self.tearDown()
            self.setUp()


class TestBrowseEndpoint(UiServerTestBase):
    def post_browse(self):
        import urllib.request
        req = urllib.request.Request(
            f"{self.base_url}/api/browse", data=b"{}", method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())

    def test_returns_selected_path(self):
        with mock.patch.object(ui, "_browse_for_folder",
                               return_value="C:\\Users\\me\\Photos"):
            self.start_server()
            status, resp = self.post_browse()
        self.assertEqual(status, 200)
        self.assertEqual(resp, {"path": "C:\\Users\\me\\Photos"})

    def test_cancelled_dialog_returns_empty_path_not_500(self):
        with mock.patch.object(ui, "subprocess") as fake_subprocess:
            fake_subprocess.run.return_value = mock.Mock(
                returncode=0, stdout="", stderr="")
            self.start_server()
            status, resp = self.post_browse()
        self.assertEqual(status, 200)
        self.assertEqual(resp, {"path": ""})

    def test_subprocess_exception_returns_empty_path_not_500(self):
        with mock.patch.object(ui, "subprocess") as fake_subprocess:
            fake_subprocess.run.side_effect = RuntimeError("no display")
            self.start_server()
            status, resp = self.post_browse()
        self.assertEqual(status, 200)
        self.assertEqual(resp, {"path": ""})

    def test_subprocess_timeout_returns_empty_path_not_500(self):
        with mock.patch.object(ui, "subprocess") as fake_subprocess:
            fake_subprocess.TimeoutExpired = subprocess.TimeoutExpired
            fake_subprocess.run.side_effect = subprocess.TimeoutExpired(
                cmd="python", timeout=120)
            self.start_server()
            status, resp = self.post_browse()
        self.assertEqual(status, 200)
        self.assertEqual(resp, {"path": ""})

    def test_nonzero_returncode_returns_empty_path_not_500(self):
        with mock.patch.object(ui, "subprocess") as fake_subprocess:
            fake_subprocess.run.return_value = mock.Mock(
                returncode=1, stdout="", stderr="traceback")
            self.start_server()
            status, resp = self.post_browse()
        self.assertEqual(status, 200)
        self.assertEqual(resp, {"path": ""})

    def test_strips_whitespace_from_selected_path(self):
        with mock.patch.object(ui, "subprocess") as fake_subprocess:
            fake_subprocess.run.return_value = mock.Mock(
                returncode=0, stdout=" C:\\Photos \n", stderr="")
            self.start_server()
            status, resp = self.post_browse()
        self.assertEqual(status, 200)
        self.assertEqual(resp, {"path": "C:\\Photos"})

    def test_uses_fresh_subprocess_not_inline_tkinter(self):
        # tkinter is not thread-safe, and the POST handler is not on the main thread —
        # the dialog must go through subprocess.run, not a direct import of tkinter
        # in the request handler.
        with mock.patch.object(ui, "subprocess") as fake_subprocess:
            fake_subprocess.run.return_value = mock.Mock(
                returncode=0, stdout="C:\\X", stderr="")
            self.start_server()
            self.post_browse()
        self.assertTrue(fake_subprocess.run.called)
        args, kwargs = fake_subprocess.run.call_args
        cmd = args[0]
        self.assertIn(ui.sys.executable, cmd)
        self.assertIn("tkinter", " ".join(cmd))


class TestBrowseJsWiring(UiServerTestBase):
    def test_browse_click_handler_fetches_and_fills_path_field(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('getElementById("process-browse-btn")', html)
        self.assertIn('"/api/browse"', html)
        self.assertIn('getElementById("process-source-dir").value', html)


if __name__ == "__main__":
    unittest.main()
