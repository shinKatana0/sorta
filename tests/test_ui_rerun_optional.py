"""F62/F63: "Re-run selected" — POST /api/process/rerun-optional.

Re-running the SELECTED over an already-built index: faces (with faces), events
(with events), junk with the VLM (with deep) — without index/geo/landmarks/phash.
The same stage-mocking trick as in test_ui_process.py (patch_fast_stages)."""
from __future__ import annotations

import threading
import unittest

from tests.test_ui_process import ProcessTestBase, _poll_until
from tests.test_ui_sort import SortBlockingTestBase


class TestRerunOptionalRunsOnlySelectedStages(ProcessTestBase):
    def test_faces_true_runs_only_faces_stage(self):
        self.patch_fast_stages()
        self.start_server()
        status, resp = self.post("/api/process/rerun-optional", {"faces": True})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))

        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertFalse(final["running"])
        self.assertIsNone(final["error"])
        self.assertEqual(self.calls, ["faces"])
        self.assertEqual(final["stage_total"], 1)

    def test_events_true_runs_only_events_stage(self):
        self.patch_fast_stages()
        self.start_server()
        status, resp = self.post("/api/process/rerun-optional", {"events": True})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))

        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertEqual(self.calls, ["events", "name_events"])
        self.assertEqual(final["stage_total"], 1)

    def test_both_true_runs_both_stages_in_order(self):
        self.patch_fast_stages()
        self.start_server()
        status, resp = self.post(
            "/api/process/rerun-optional", {"faces": True, "events": True})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))

        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertEqual(self.calls, ["faces", "events", "name_events"])
        self.assertEqual(final["stage_total"], 2)

    def test_deep_true_runs_only_junk_stage(self):
        # F63: deep -> re-run junk (reclassification with the VLM), nothing else.
        self.patch_fast_stages()
        self.start_server()
        status, resp = self.post("/api/process/rerun-optional", {"deep": True})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertEqual(self.calls, ["junk"])
        self.assertEqual(final["stage_total"], 1)

    def test_faces_and_deep_runs_faces_then_junk(self):
        # the pipeline order is preserved: faces comes before junk.
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post(
            "/api/process/rerun-optional", {"faces": True, "deep": True})
        self.assertEqual(status, 200)
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertEqual(self.calls, ["faces", "junk"])
        self.assertEqual(final["stage_total"], 2)

    def test_base_stages_never_invoked_without_deep(self):
        # without deep: index/geo/landmarks/junk/phash are not run at all.
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post(
            "/api/process/rerun-optional", {"faces": True, "events": True})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        for name in ("index", "assign_duplicates", "geo", "landmarks", "junk", "phash"):
            self.assertNotIn(name, self.calls)

    def test_does_not_override_cfg_sources(self):
        # source_dir=None -> Path(None) is not called, the run's cfg.sources stays
        # the same as the server's (no reindexing).
        captured: dict = {}

        def fake_faces(cfg, conn, progress=None):
            captured["cfg"] = cfg
            self.calls.append("faces")
            if progress:
                progress(1, 1)

        self.patch_fast_stages()
        self._patch("detect_and_cluster", fake_faces)
        self.start_server()
        status, _resp = self.post("/api/process/rerun-optional", {"faces": True})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertEqual(list(captured["cfg"].sources), list(self.cfg.sources))


class TestRerunOptionalValidation(ProcessTestBase):
    def test_all_false_returns_400(self):
        self.start_server()
        status, resp = self.post(
            "/api/process/rerun-optional", {"faces": False, "events": False, "deep": False})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_deep_non_bool_returns_400(self):
        self.start_server()
        status, resp = self.post("/api/process/rerun-optional", {"deep": "yes"})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_empty_body_returns_400(self):
        self.start_server()
        status, resp = self.post("/api/process/rerun-optional", {})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_faces_non_bool_returns_400(self):
        self.start_server()
        status, resp = self.post("/api/process/rerun-optional", {"faces": "yes"})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_events_non_bool_returns_400(self):
        self.start_server()
        status, resp = self.post("/api/process/rerun-optional", {"events": "yes"})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_non_dict_body_returns_400(self):
        self.start_server()
        status, resp = self.post("/api/process/rerun-optional", "not a dict")
        self.assertEqual(status, 400)
        self.assertIn("error", resp)


class TestRerunOptionalConcurrency(ProcessTestBase):
    def test_second_call_while_running_returns_409(self):
        block = threading.Event()
        self.patch_fast_stages(block_stage="faces", block_event=block)
        self.start_server()
        try:
            status1, resp1 = self.post("/api/process/rerun-optional", {"faces": True})
            self.assertEqual(status1, 200)
            self.assertTrue(resp1.get("ok"))
            self.assertTrue(self.status()["running"])

            status2, resp2 = self.post("/api/process/rerun-optional", {"faces": True})
            self.assertEqual(status2, 409)
            self.assertIn("error", resp2)
        finally:
            block.set()
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertFalse(final["running"])
        self.assertIsNone(final["error"])

    def test_blocked_while_ordinary_process_running(self):
        block = threading.Event()
        self.patch_fast_stages(block_stage="index", block_event=block)
        self.start_server()
        try:
            status1, resp1 = self.post("/api/process", {"source_dir": str(self.src_dir)})
            self.assertEqual(status1, 200)
            self.assertTrue(resp1.get("ok"))
            self.assertTrue(self.status()["running"])

            status2, resp2 = self.post("/api/process/rerun-optional", {"faces": True})
            self.assertEqual(status2, 409)
            self.assertIn("error", resp2)
        finally:
            block.set()
        _poll_until(self.status, lambda d: d["finished"])


class TestRerunOptionalBlockedDuringSort(SortBlockingTestBase):
    def test_blocked_while_sort_running(self):
        block = threading.Event()
        self.patch_blocking_sort(block)
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        try:
            status1, _resp1 = self.post(
                "/api/sort", {"dest": str(self.root / "dest"), "mode": "move"})
            self.assertEqual(status1, 200)
            self.assertTrue(self.sort_status()["running"])

            status2, resp2 = self.post("/api/process/rerun-optional", {"faces": True})
            self.assertEqual(status2, 409)
            self.assertIn("error", resp2)
        finally:
            block.set()
        _poll_until(self.sort_status, lambda d: d["finished"])


class TestRerunOptionalHtml(ProcessTestBase):
    def test_button_in_rerun_block_and_disabled_by_default(self):
        # F63: the button + its hint are moved into a separate column block
        # (.process-rerun-block), not the horizontal .process-actions row.
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="process-rerun-optional-btn"', html)
        self.assertIn('class="process-rerun-block"', html)
        block_pos = html.index('class="process-rerun-block"')
        rerun_pos = html.index('id="process-rerun-optional-btn"')
        self.assertLess(block_pos, rerun_pos)  # the button is inside the re-run block
        # disabled by default (the faces/events/deep checkboxes are cleared at start)
        btn_tag_end = html.index(">", rerun_pos)
        self.assertIn("disabled", html[rerun_pos:btn_tag_end])

    def test_i18n_ru_en_ja(self):
        self.start_server()
        _status, body, _ctype = self.get("/?lang=ru")
        html = body.decode("utf-8")
        self.assertIn("Дозапустить выбранное", html)

        _status, body, _ctype = self.get("/?lang=en")
        html = body.decode("utf-8")
        self.assertIn("Re-run selected", html)

        _status, body, _ctype = self.get("/?lang=ja")
        html = body.decode("utf-8")
        self.assertIn("選択項目を再実行", html)

    def test_no_external_resources(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)


class TestRerunOptionalJs(ProcessTestBase):
    def test_change_listeners_wired_to_all_three_checkboxes(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("updateRerunSelectedDisabled", html)
        # button availability depends on faces/events/deep — deep is in the list too
        self.assertIn('"process-faces-checkbox", "process-events-checkbox", '
                      '"process-deep-checkbox"', html)
        self.assertIn('.addEventListener("change", updateRerunSelectedDisabled)', html)

    def test_click_handler_posts_rerun_optional_endpoint_with_deep(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn(
            'document.getElementById("process-rerun-optional-btn").addEventListener('
            '"click", function () {', html)
        self.assertIn('postJson("/api/process/rerun-optional", '
                      '{ faces: faces, events: events, deep: deep })', html)

    def test_current_process_stages_set_to_selected_on_rerun_click(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("filterRerunStages", html)
        self.assertIn("currentProcessStages = filterRerunStages(faces, events, deep);", html)


class TestRerunOptionalHintPlacement(ProcessTestBase):
    def test_hint_in_rerun_block_below_actions_not_inside_flex_row(self):
        # F63: the hint — in the column block under the re-run button, after the
        # horizontal .process-actions row (not inline between the buttons).
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('class="process-rerun-hint"', html)
        actions_close = html.index("</div>", html.index('class="process-actions"'))
        hint_pos = html.index('class="process-rerun-hint"')
        rerun_pos = html.index('id="process-rerun-optional-btn"')
        self.assertLess(actions_close, hint_pos)   # the hint after the actions row
        self.assertLess(rerun_pos, hint_pos)       # and after the button itself (in its block)


if __name__ == "__main__":
    unittest.main()
