"""F36: the "Process" entry point — POST /api/process (+ status/cancel).

The pipeline is MOCKED by monkeypatching the leaf functions with fast stubs (no ML/
downloads) — see ui._pipeline_steps, which reads these names from the module's
globals() at the moment the background thread starts, so they can be patched right
on the `sorta.ui` object.
"""
from __future__ import annotations

import dataclasses
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

from sorta import ui

from tests.test_ui import UiServerTestBase


class _FakeIndexStats:
    """The `_index` adapter in ui._pipeline_steps wraps progress as
    `lambda s: cb(s.scanned, None)` (the real run_index calls progress(stats) with a
    single object, not (done,total)) — the stub needs the same contract."""

    def __init__(self, scanned: int) -> None:
        self.scanned = scanned


def _poll_until(get_status, predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = get_status()
        if predicate(last):
            return last
        time.sleep(interval)
    raise AssertionError(f"condition not met in {timeout}s; last status: {last}")


class ProcessTestBase(UiServerTestBase):
    """Stage-mock fixtures + JSON POST on top of the base U1 server."""

    def setUp(self):
        super().setUp()
        self.calls: list[str] = []

    def _patch(self, name: str, func) -> None:
        patcher = mock.patch.object(ui, name, func)
        patcher.start()
        self.addCleanup(patcher.stop)

    def patch_fast_stages(self, *, block_stage: str | None = None,
                          block_event: threading.Event | None = None) -> None:
        """Monkeypatch all 7 stages (+assign_duplicates/name_events) with fast stubs.

        block_stage — the name of the stage that waits on block_event.wait() AFTER
        recording the call (for the 409 race / between-stage cancellation tests).
        """
        calls = self.calls

        def maybe_block(name: str) -> None:
            if name == block_stage and block_event is not None:
                block_event.wait(timeout=5)

        def fake_index(cfg, conn, progress=None):
            calls.append("index")
            if progress:
                progress(_FakeIndexStats(1))
            maybe_block("index")

        def fake_assign_duplicates(conn, strategy):
            calls.append("assign_duplicates")
            return 0

        def fake_geo(cfg, conn, progress=None):
            calls.append("geo")
            if progress:
                progress(1, 1)
            maybe_block("geo")

        def fake_landmarks(cfg, conn, classifier=None, progress=None):
            calls.append("landmarks")
            if progress:
                progress(1, 1)
            maybe_block("landmarks")

        def fake_faces(cfg, conn, progress=None):
            calls.append("faces")
            if progress:
                progress(1, 1)
            maybe_block("faces")
            return None

        def fake_events(cfg, conn, progress=None):
            calls.append("events")
            if progress:
                progress(1, 1)
            maybe_block("events")

        def fake_name_events(cfg, conn, namer=None, progress=None):
            calls.append("name_events")

        def fake_junk(cfg, conn, classifier=None, use_clip=True, text_detector=None,
                     progress=None):
            calls.append("junk")
            if progress:
                progress(1, 1)
            maybe_block("junk")

        def fake_phash(cfg, conn, progress=None):
            calls.append("phash")
            if progress:
                progress(1, 1)
            maybe_block("phash")
            return 0

        self._patch("run_index", fake_index)
        self._patch("assign_duplicates", fake_assign_duplicates)
        self._patch("resolve_places", fake_geo)
        self._patch("detect_landmarks", fake_landmarks)
        self._patch("detect_and_cluster", fake_faces)
        self._patch("build_events", fake_events)
        self._patch("name_events", fake_name_events)
        self._patch("classify_junk", fake_junk)
        self._patch("compute_phashes", fake_phash)

    def post(self, path: str, data: dict) -> tuple[int, dict]:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def status(self) -> dict:
        status, body, _ctype = self.get("/api/process/status")
        self.assertEqual(status, 200)
        return json.loads(body)


class TestProcessStartAndProgress(ProcessTestBase):
    def test_idle_status_before_any_run(self):
        self.start_server()
        data = self.status()
        self.assertEqual(
            set(data.keys()),
            {"running", "stage", "stage_index", "stage_total", "done", "total",
             "error", "finished", "cancel_requested", "source_dir"},
        )
        self.assertFalse(data["running"])
        self.assertFalse(data["finished"])
        self.assertIsNone(data["error"])

    def test_start_runs_all_stages_in_order_and_finishes(self):
        # F53/#39: faces/events — opt-in, default off; the full stage set requires
        # explicit flags in the request body.
        self.patch_fast_stages()
        self.start_server()
        status, resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "faces": True, "events": True})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))

        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertFalse(final["running"])
        self.assertIsNone(final["error"])
        self.assertEqual(final["stage_index"], final["stage_total"])
        self.assertEqual(final["source_dir"], str(self.src_dir))
        self.assertEqual(
            self.calls,
            ["index", "assign_duplicates", "geo", "landmarks", "faces",
             "events", "name_events", "junk", "phash"],
        )

    def test_finished_refreshes_plan_cache_for_city_tab(self):
        # PlanCache is built once at startup (on an empty DB -> an empty plan);
        # after a successful /api/process it must be recomputed, not stay frozen at
        # the startup state (see PlanCache.rebuild, F36).
        self.patch_fast_stages()
        self.start_server()
        status_before, body_before, _ = self.get("/api/plan?mode=city")
        self.assertEqual(status_before, 200)
        self.assertEqual(json.loads(body_before), [])

        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])

        status_after, body_after, _ = self.get("/api/plan?mode=city")
        self.assertEqual(status_after, 200)
        items = json.loads(body_after)
        self.assertEqual(len(items), 1)


class TestProcessValidation(ProcessTestBase):
    def test_missing_source_dir_400(self):
        self.start_server()
        status, resp = self.post("/api/process", {})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_blank_source_dir_400(self):
        self.start_server()
        status, resp = self.post("/api/process", {"source_dir": "   "})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_non_dict_body_400(self):
        self.start_server()
        status, resp = self.post("/api/process", {"source_dir": 123})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_nonexistent_dir_400(self):
        self.start_server()
        status, resp = self.post(
            "/api/process", {"source_dir": str(self.root / "does-not-exist")})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)


class TestProcessConcurrency(ProcessTestBase):
    def test_second_post_while_running_returns_409(self):
        block = threading.Event()
        self.patch_fast_stages(block_stage="index", block_event=block)
        self.start_server()
        try:
            status1, resp1 = self.post("/api/process", {"source_dir": str(self.src_dir)})
            self.assertEqual(status1, 200)
            self.assertTrue(resp1.get("ok"))
            # try_start() runs synchronously in the POST handler before the response —
            # by this point running is already guaranteed True (no race).
            self.assertTrue(self.status()["running"])

            status2, resp2 = self.post("/api/process", {"source_dir": str(self.src_dir)})
            self.assertEqual(status2, 409)
            self.assertIn("error", resp2)
        finally:
            block.set()
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertFalse(final["running"])
        self.assertIsNone(final["error"])

    def test_new_run_allowed_after_previous_finished(self):
        self.patch_fast_stages()
        self.start_server()
        status1, _resp1 = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status1, 200)
        _poll_until(self.status, lambda d: d["finished"])

        status2, resp2 = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status2, 200)
        self.assertTrue(resp2.get("ok"))
        _poll_until(self.status, lambda d: d["finished"])
        # index must be called twice (once per run).
        self.assertEqual(self.calls.count("index"), 2)


class TestProcessCancel(ProcessTestBase):
    def test_cancel_stops_between_stages(self):
        block = threading.Event()
        self.patch_fast_stages(block_stage="geo", block_event=block)
        self.start_server()
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status, 200)

        # wait until the pipeline reaches the geo stage (index already ran)
        _poll_until(self.status, lambda d: d["stage"] == "geo")

        cancel_status, cancel_resp = self.post("/api/process/cancel", {})
        self.assertEqual(cancel_status, 200)
        self.assertTrue(cancel_resp.get("ok"))

        # geo already called progress(1,1) BEFORE blocking (before the cancel) — so
        # geo itself is not interrupted mid-stage; we release it, the thread catches
        # the flag BETWEEN stages.
        block.set()

        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertFalse(final["running"])
        self.assertIsNone(final["error"])
        self.assertTrue(final["cancel_requested"])  # the snapshot reflects the cancel
        # landmarks/faces/events/junk/phash must NOT have run.
        self.assertEqual(self.calls, ["index", "assign_duplicates", "geo"])
        self.assertLess(final["stage_index"], final["stage_total"])

    def test_cancel_interrupts_current_stage_mid_run(self):
        # a stage that calls progress AFTER a cancel request is interrupted on that
        # call (mid-stage) — without waiting for the stage boundary.
        block = threading.Event()
        self.patch_fast_stages()

        def blocking_geo(cfg, conn, progress=None):
            self.calls.append("geo")
            block.wait(timeout=5)                    # wait until the test requests a cancel
            if progress:
                progress(1, 1)                       # after the cancel → _PipelineCancelled
            self.calls.append("geo_after_progress")  # must NOT execute

        self._patch("resolve_places", blocking_geo)
        self.start_server()
        self.post("/api/process", {"source_dir": str(self.src_dir)})
        _poll_until(self.status, lambda d: d["stage"] == "geo")
        self.post("/api/process/cancel", {})
        block.set()                                  # geo continues → progress → cancel

        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertTrue(final["cancel_requested"])
        self.assertIn("geo", self.calls)
        self.assertNotIn("geo_after_progress", self.calls)  # interrupted ON progress
        self.assertNotIn("landmarks", self.calls)           # the next stages too

    def test_set_progress_raises_when_cancel_requested(self):
        from sorta.ui import _PipelineCancelled, _ProcessState
        st = _ProcessState()
        self.assertTrue(st.try_start("x"))   # running=True
        st.request_cancel()
        with self.assertRaises(_PipelineCancelled):
            st.set_progress(1, 2)
        self.assertTrue(st.snapshot()["cancel_requested"])

    def test_cancel_when_not_running_is_a_noop(self):
        self.start_server()
        status, resp = self.post("/api/process/cancel", {})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))
        self.assertFalse(self.status()["running"])


class TestProcessPipelineError(ProcessTestBase):
    def test_stage_exception_surfaces_as_error_and_stops(self):
        def boom(cfg, conn, progress=None):
            self.calls.append("index")
            raise RuntimeError("boom")

        self._patch("run_index", boom)
        self.start_server()
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status, 200)

        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertFalse(final["running"])
        self.assertIn("boom", final["error"])
        self.assertEqual(self.calls, ["index"])


class TestEmptyDbNoServerError(ProcessTestBase):
    def test_all_tabs_load_without_500_on_empty_db(self):
        self.start_server()
        for path in ("/", "/api/dupes", "/api/clusters", "/api/events",
                     "/api/moves", "/api/plan?mode=city", "/api/process/status"):
            status, _body, _ctype = self.get(path)
            self.assertLess(status, 500, f"{path} -> {status}")


class TestProcessHtml(ProcessTestBase):
    def test_process_section_present_and_first(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="tab-btn-process"', html)
        self.assertIn('id="tab-process"', html)
        self.assertIn('id="process-source-dir"', html)
        self.assertIn('id="process-start-btn"', html)
        self.assertIn('id="process-cancel-btn"', html)
        self.assertIn('id="process-progress"', html)
        self.assertIn("/api/process", html)
        # "Process" — the first tab button and the default-active one (the landing).
        process_pos = html.index('id="tab-btn-process"')
        city_pos = html.index('id="tab-btn-city"')
        self.assertLess(process_pos, city_pos)
        self.assertIn('class="tab-btn active" id="tab-btn-process"', html)
        self.assertNotIn('class="tab-btn active" id="tab-btn-city"', html)
        # without external resources (see the other tabs/F31-F35)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)

    def test_existing_tabs_still_present(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        for tab_id in ("tab-btn-city", "tab-btn-dupes", "tab-btn-person",
                      "tab-btn-event", "tab-btn-moves"):
            self.assertIn(f'id="{tab_id}"', html)


class TestProcessTogglesHtml(ProcessTestBase):
    """F50/#34: the VLM/online-geo checkboxes on the "Process" tab + i18n×3."""

    def test_toggle_checkboxes_and_warnings_present_default_lang(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="process-deep-checkbox"', html)
        self.assertIn('id="process-geo-online-checkbox"', html)
        self.assertIn("Deep analysis (VLM)", html)
        self.assertIn("uv sync --extra vlm", html)
        self.assertIn("Online geo", html)
        self.assertIn("GPS coordinates", html)

    def test_toggle_labels_and_warnings_en(self):
        self.cfg.raw = {"language": "en"}
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("Deep analysis (VLM)", html)
        self.assertIn("uv sync --extra vlm", html)
        self.assertIn("Online geo (more accurate abroad)", html)
        self.assertIn("GPS coordinates", html)

    def test_toggle_labels_and_warnings_ja(self):
        self.cfg.raw = {"language": "ja"}
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("詳細分析（VLM）", html)
        self.assertIn("オンライン位置情報（海外でより正確）", html)


class TestProcessDeepGeoOverride(ProcessTestBase):
    """F50/#34: the deep/geo_online toggles build a per-run cfg via
    dataclasses.replace, without touching the server cfg/config.yaml."""

    def _capture_cfg_stages(self) -> dict:
        captured: dict = {}

        def fake_index(cfg, conn, progress=None):
            captured["cfg"] = cfg
            self.calls.append("index")
            if progress:
                progress(_FakeIndexStats(0))

        def fake_assign_duplicates(conn, strategy):
            return 0

        def fake_geo(cfg, conn, progress=None):
            self.calls.append("geo")

        def fake_landmarks(cfg, conn, classifier=None, progress=None):
            self.calls.append("landmarks")

        def fake_faces(cfg, conn, progress=None):
            self.calls.append("faces")

        def fake_events(cfg, conn, progress=None):
            self.calls.append("events")

        def fake_name_events(cfg, conn, namer=None, progress=None):
            self.calls.append("name_events")

        def fake_junk(cfg, conn, classifier=None, use_clip=True, text_detector=None,
                     progress=None):
            self.calls.append("junk")

        def fake_phash(cfg, conn, progress=None):
            self.calls.append("phash")
            return 0

        self._patch("run_index", fake_index)
        self._patch("assign_duplicates", fake_assign_duplicates)
        self._patch("resolve_places", fake_geo)
        self._patch("detect_landmarks", fake_landmarks)
        self._patch("detect_and_cluster", fake_faces)
        self._patch("build_events", fake_events)
        self._patch("name_events", fake_name_events)
        self._patch("classify_junk", fake_junk)
        self._patch("compute_phashes", fake_phash)
        return captured

    def test_default_no_toggles_run_cfg_matches_server_cfg(self):
        captured = self._capture_cfg_stages()
        self.start_server()
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertFalse(captured["cfg"].naming.vlm_enabled)
        self.assertEqual(captured["cfg"].geo.provider, "offline")
        # the original server cfg is not mutated
        self.assertFalse(self.cfg.naming.vlm_enabled)
        self.assertEqual(self.cfg.geo.provider, "offline")

    def test_deep_true_enables_vlm_only_on_run_cfg(self):
        captured = self._capture_cfg_stages()
        self.start_server()
        status, _resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "deep": True})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertTrue(captured["cfg"].naming.vlm_enabled)
        self.assertFalse(self.cfg.naming.vlm_enabled)

    def test_geo_online_true_sets_provider_only_on_run_cfg(self):
        captured = self._capture_cfg_stages()
        self.start_server()
        status, _resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "geo_online": True})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertEqual(captured["cfg"].geo.provider, "online")
        self.assertEqual(self.cfg.geo.provider, "offline")

    # F57: a full override — an unchecked box must force OFF, even if cfg
    # (config.yaml) keeps VLM/online-geo enabled. Previously `deep`/`geo_online`
    # were additive (True forced ON, False quietly took cfg as-is) — from the UI
    # you could not disable what is enabled in config.

    def test_deep_false_forces_vlm_off_even_if_cfg_has_it_enabled(self):
        self.cfg.naming = dataclasses.replace(self.cfg.naming, vlm_enabled=True)
        captured = self._capture_cfg_stages()
        self.start_server()
        status, _resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "deep": False})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertFalse(captured["cfg"].naming.vlm_enabled)
        # the original server cfg (config.yaml) is not mutated
        self.assertTrue(self.cfg.naming.vlm_enabled)

    def test_geo_online_false_forces_offline_even_if_cfg_has_online(self):
        self.cfg.geo = dataclasses.replace(self.cfg.geo, provider="online")
        captured = self._capture_cfg_stages()
        self.start_server()
        status, _resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "geo_online": False})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertEqual(captured["cfg"].geo.provider, "offline")
        self.assertEqual(self.cfg.geo.provider, "online")

    def test_omitted_toggles_default_to_off_even_if_cfg_has_them_enabled(self):
        # _validate_process_payload still defaults missing deep/geo_online to False
        # (not touched by this feature) — but now that False is also an authoritative
        # override, not "take cfg as-is".
        self.cfg.naming = dataclasses.replace(self.cfg.naming, vlm_enabled=True)
        self.cfg.geo = dataclasses.replace(self.cfg.geo, provider="online")
        captured = self._capture_cfg_stages()
        self.start_server()
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertFalse(captured["cfg"].naming.vlm_enabled)
        self.assertEqual(captured["cfg"].geo.provider, "offline")

    def test_deep_true_still_enables_vlm_when_cfg_already_enabled(self):
        self.cfg.naming = dataclasses.replace(self.cfg.naming, vlm_enabled=True)
        captured = self._capture_cfg_stages()
        self.start_server()
        status, _resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "deep": True})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertTrue(captured["cfg"].naming.vlm_enabled)


class TestProcessTogglesValidation(ProcessTestBase):
    def test_deep_non_bool_400(self):
        self.start_server()
        status, resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "deep": "yes"})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_geo_online_non_bool_400(self):
        self.start_server()
        status, resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "geo_online": "online"})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_faces_non_bool_400(self):
        self.start_server()
        status, resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "faces": "yes"})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_events_non_bool_400(self):
        self.start_server()
        status, resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "events": "yes"})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)


class TestProcessOptionalStages(ProcessTestBase):
    """F53/#39: faces/events — opt-in steps, default off, independent of each other
    and of deep/geo_online; the filtering is reflected in stage_total."""

    def test_default_skips_faces_and_events(self):
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status, 200)
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertEqual(
            self.calls,
            ["index", "assign_duplicates", "geo", "landmarks", "junk", "phash"])
        self.assertEqual(final["stage_total"], 5)  # index/geo/landmarks/junk/phash

    def test_faces_true_adds_faces_only(self):
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "faces": True})
        self.assertEqual(status, 200)
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertEqual(
            self.calls,
            ["index", "assign_duplicates", "geo", "landmarks", "faces", "junk", "phash"])
        self.assertEqual(final["stage_total"], 6)

    def test_events_true_adds_events_only(self):
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post(
            "/api/process", {"source_dir": str(self.src_dir), "events": True})
        self.assertEqual(status, 200)
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertEqual(
            self.calls,
            ["index", "assign_duplicates", "geo", "landmarks", "events", "name_events",
             "junk", "phash"])
        self.assertEqual(final["stage_total"], 6)

    def test_faces_and_events_true_adds_both(self):
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post(
            "/api/process",
            {"source_dir": str(self.src_dir), "faces": True, "events": True})
        self.assertEqual(status, 200)
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertEqual(
            self.calls,
            ["index", "assign_duplicates", "geo", "landmarks", "faces",
             "events", "name_events", "junk", "phash"])
        self.assertEqual(final["stage_total"], 7)


class TestProcessOptionalStagesHtml(ProcessTestBase):
    """F53/#39: the "Detect faces"/"Detect events" checkboxes + i18n×3."""

    def test_checkboxes_present_default_lang(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="process-faces-checkbox"', html)
        self.assertIn('id="process-events-checkbox"', html)
        self.assertIn("Detect faces", html)
        self.assertIn("Detect events", html)

    def test_labels_en(self):
        self.cfg.raw = {"language": "en"}
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("Detect faces", html)
        self.assertIn("Detect events", html)

    def test_labels_ja(self):
        self.cfg.raw = {"language": "ja"}
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("顔の検出", html)
        self.assertIn("イベントの検出", html)


class TestProcessDefaultsEndpoint(ProcessTestBase):
    """F57: GET /api/process/defaults — the source of truth for initializing the
    "Process" checkboxes on the client (modelled on GET /api/tabs/visibility)."""

    def test_default_cfg_returns_false_false(self):
        self.start_server()
        status, body, ctype = self.get("/api/process/defaults")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        data = json.loads(body)
        self.assertEqual(set(data.keys()), {"deep", "geo_online", "vlm_available"})
        self.assertFalse(data["deep"])
        self.assertFalse(data["geo_online"])

    def test_vlm_enabled_and_online_provider_in_cfg_reflected(self):
        self.cfg.naming = dataclasses.replace(self.cfg.naming, vlm_enabled=True)
        self.cfg.geo = dataclasses.replace(self.cfg.geo, provider="online")
        self.start_server()
        _status, body, _ctype = self.get("/api/process/defaults")
        data = json.loads(body)
        self.assertTrue(data["deep"])
        self.assertTrue(data["geo_online"])

    def test_vlm_available_true_when_transformers_importable(self):
        with mock.patch.object(ui.importlib.util, "find_spec", return_value=object()):
            self.start_server()
            _status, body, _ctype = self.get("/api/process/defaults")
        data = json.loads(body)
        self.assertTrue(data["vlm_available"])

    def test_vlm_available_false_when_transformers_missing(self):
        with mock.patch.object(ui.importlib.util, "find_spec", return_value=None):
            self.start_server()
            _status, body, _ctype = self.get("/api/process/defaults")
        data = json.loads(body)
        self.assertFalse(data["vlm_available"])


class TestProcessDefaultsInitJs(ProcessTestBase):
    """F57: the markup/JS fetches /api/process/defaults and sets .checked on both
    checkboxes on page init (without this they always start empty regardless of
    config.yaml)."""

    def test_fetch_and_checked_assignment_present(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('fetch("/api/process/defaults")', html)
        self.assertIn(
            'document.getElementById("process-deep-checkbox").checked = !!data.deep', html)
        self.assertIn(
            'document.getElementById("process-geo-online-checkbox").checked = '
            '!!data.geo_online', html)


class TestVlmMissingWarningHtml(ProcessTestBase):
    """F57: a muted "VLM not installed" note next to the deep checkbox — shown only
    when the checkbox is checked and vlm_available=false; i18n×3."""

    def test_markup_and_default_text_present(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="process-deep-vlm-missing"', html)
        self.assertIn('style="display:none"', html)
        self.assertIn("VLM is not installed", html)
        self.assertIn("uv sync --extra vlm", html)

    def test_en_text_present(self):
        self.cfg.raw = {"language": "en"}
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("VLM is not installed", html)
        self.assertIn("uv sync --extra vlm", html)

    def test_ja_text_present(self):
        self.cfg.raw = {"language": "ja"}
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("VLM がインストールされていません", html)

    def test_toggle_js_wires_change_listener_to_warning(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("updateVlmMissingWarning", html)
        self.assertIn(
            'document.getElementById("process-deep-checkbox")\n'
            '      .addEventListener("change", updateVlmMissingWarning)', html)


if __name__ == "__main__":
    unittest.main()
