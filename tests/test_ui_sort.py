"""F43: apply the city layout from the UI — POST /api/sort (+ GET /api/sort/status),
a concurrency guard with /api/process (cross-locking), the HTML controls of the
"Cities" tab (dest/move|copy/"Sort"/confirmation texts).

The server runs in a thread (see test_ui.py). Happy-path tests call the real
`sorter.plan_and_sort` without mocks — files are physically moved/copied into
tmp-dest, the journal is written by the engine (the same guarantee as the CLI
`sort --by city --apply`). For the race tests (409 during a sort) `ui.plan_and_sort`
is replaced with a blocking stub — the same trick by which `patch_fast_stages` in
test_ui_process.py blocks a `/api/process` pipeline stage.
"""
from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from sorta import ui
from sorta.config import Config
from sorta.sorter import SortReport

from tests.test_ui import UiServerTestBase


def _poll_until(get_status, predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = get_status()
        if predicate(last):
            return last
        time.sleep(interval)
    raise AssertionError(f"condition not met in {timeout}s; last status: {last}")


class SortTestBase(UiServerTestBase):
    """JSON POST + snapshots of /api/sort/status and /api/process/status on top of U1."""

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

    def sort_status(self) -> dict:
        status, body, _ctype = self.get("/api/sort/status")
        self.assertEqual(status, 200)
        return json.loads(body)

    def process_status(self) -> dict:
        status, body, _ctype = self.get("/api/process/status")
        self.assertEqual(status, 200)
        return json.loads(body)


class TestSortCopy(SortTestBase):
    def test_copy_leaves_originals_and_writes_copy_batch(self):
        _fid1, p1, _c1 = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        _fid2, p2, _c2 = self.add_photo_file("b.jpg", country="ru", city="Moscow")
        self.start_server()
        dest = self.root / "dest"

        status, resp = self.post("/api/sort", {"dest": str(dest), "mode": "copy"})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))

        final = _poll_until(self.sort_status, lambda d: d["finished"])
        self.assertFalse(final["running"])
        self.assertIsNone(final["error"])
        self.assertEqual(final["result"]["moved"], 2)
        self.assertEqual(final["result"]["failed"], 0)

        # originals stay in place
        self.assertTrue(p1.exists())
        self.assertTrue(p2.exists())
        # copies appeared under dest
        copied = list(dest.rglob("*.jpg"))
        self.assertEqual(len(copied), 2)

        batch = self.conn.execute(
            "SELECT operation FROM move_batches ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertEqual(batch["operation"], "copy")


class TestSortMove(SortTestBase):
    def test_move_relocates_files_and_moves_tab_sees_batch(self):
        _fid1, p1, _c1 = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        dest = self.root / "dest"

        status, resp = self.post("/api/sort", {"dest": str(dest), "mode": "move"})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))

        final = _poll_until(self.sort_status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertEqual(final["result"]["moved"], 1)

        self.assertFalse(p1.exists())
        moved = list(dest.rglob("*.jpg"))
        self.assertEqual(len(moved), 1)

        batch = self.conn.execute(
            "SELECT operation FROM move_batches ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertEqual(batch["operation"], "move")

        status, body, _ctype = self.get("/api/moves")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertIsNotNone(payload["batch"])
        self.assertEqual(len(payload["moves"]), 1)


class TestSortValidation(SortTestBase):
    def test_invalid_mode_returns_400(self):
        self.start_server()
        status, resp = self.post(
            "/api/sort", {"dest": str(self.root / "dest"), "mode": "link"})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_missing_mode_returns_400(self):
        self.start_server()
        status, resp = self.post("/api/sort", {"dest": str(self.root / "dest")})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_non_dict_dest_returns_400(self):
        self.start_server()
        status, resp = self.post("/api/sort", {"dest": 123, "mode": "move"})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_non_dict_body_400(self):
        self.start_server()
        status, resp = self.post("/api/sort", "not a dict")
        self.assertEqual(status, 400)
        self.assertIn("error", resp)


class TestSortInPlaceValueError(SortTestBase):
    def test_multiple_sources_in_place_surfaces_error_without_crashing(self):
        # in-place (dest empty) requires exactly one cfg.sources — with two sources
        # plan_and_sort raises ValueError; the thread must catch it and store it in
        # the state, without crashing the server (see brief F43).
        other_src = self.root / "src2"
        other_src.mkdir()
        self.cfg = Config(sources=[self.src_dir, other_src], database=self.cfg.database, raw={})
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()

        status, resp = self.post("/api/sort", {"dest": "", "mode": "move"})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))

        final = _poll_until(self.sort_status, lambda d: d["finished"])
        self.assertFalse(final["running"])
        self.assertIsNotNone(final["error"])
        self.assertIsNone(final["result"])

        # the server is alive after the ValueError in the background thread
        status, _body, _ctype = self.get("/")
        self.assertEqual(status, 200)


class TestSortStatusShape(SortTestBase):
    def test_idle_status_before_any_run(self):
        self.start_server()
        data = self.sort_status()
        self.assertEqual(
            set(data.keys()), {"running", "done", "total", "error", "finished", "result"})
        self.assertFalse(data["running"])
        self.assertFalse(data["finished"])
        self.assertIsNone(data["error"])
        self.assertIsNone(data["result"])

    def test_status_reflects_finished_and_result(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        dest = self.root / "dest"
        status, _resp = self.post("/api/sort", {"dest": str(dest), "mode": "copy"})
        self.assertEqual(status, 200)
        final = _poll_until(self.sort_status, lambda d: d["finished"])
        self.assertEqual(
            set(final["result"].keys()),
            {"moved", "failed", "skipped_in_place", "dirs", "dest", "in_place", "mode"},
        )
        self.assertEqual(final["result"]["mode"], "copy")
        self.assertFalse(final["result"]["in_place"])


class SortBlockingTestBase(SortTestBase):
    """Replaces ui.plan_and_sort with a blocking stub — only for the race tests
    (409 during a sort), the same trick as test_ui_process.py."""

    def patch_blocking_sort(self, block_event: threading.Event) -> list:
        calls: list = []

        def fake_plan_and_sort(cfg, conn, mode, dest, apply=False, copy=False,
                               progress=None, **kwargs):
            calls.append((mode, dest, apply, copy))
            if progress:
                progress(0, 1)
            block_event.wait(timeout=5)
            return SortReport(
                mode=mode, dest=Path(dest) if dest else Path(cfg.sources[0]),
                csv_path=self.root / "plan.csv", html_path=self.root / "plan.html",
                moved=1, failed=0, skipped_in_place=0, dirs=1,
                in_place=dest is None,
            )

        patcher = mock.patch.object(ui, "plan_and_sort", fake_plan_and_sort)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls


class TestSortConcurrency(SortBlockingTestBase):
    def test_second_sort_post_while_running_returns_409(self):
        block = threading.Event()
        self.patch_blocking_sort(block)
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        dest = str(self.root / "dest")
        try:
            status1, resp1 = self.post("/api/sort", {"dest": dest, "mode": "move"})
            self.assertEqual(status1, 200)
            self.assertTrue(resp1.get("ok"))
            self.assertTrue(self.sort_status()["running"])

            status2, resp2 = self.post("/api/sort", {"dest": dest, "mode": "move"})
            self.assertEqual(status2, 409)
            self.assertIn("error", resp2)
        finally:
            block.set()
        final = _poll_until(self.sort_status, lambda d: d["finished"])
        self.assertFalse(final["running"])
        self.assertIsNone(final["error"])

    def test_process_start_blocked_while_sort_running(self):
        block = threading.Event()
        self.patch_blocking_sort(block)
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        try:
            status1, _resp1 = self.post(
                "/api/sort", {"dest": str(self.root / "dest"), "mode": "move"})
            self.assertEqual(status1, 200)
            self.assertTrue(self.sort_status()["running"])

            status2, resp2 = self.post("/api/process", {"source_dir": str(self.src_dir)})
            self.assertEqual(status2, 409)
            self.assertIn("error", resp2)
        finally:
            block.set()
        _poll_until(self.sort_status, lambda d: d["finished"])

    def test_process_reset_blocked_while_sort_running(self):
        block = threading.Event()
        self.patch_blocking_sort(block)
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        try:
            status1, _resp1 = self.post(
                "/api/sort", {"dest": str(self.root / "dest"), "mode": "move"})
            self.assertEqual(status1, 200)
            self.assertTrue(self.sort_status()["running"])

            status2, resp2 = self.post("/api/process/reset", {})
            self.assertEqual(status2, 409)
            self.assertIn("error", resp2)
        finally:
            block.set()
        _poll_until(self.sort_status, lambda d: d["finished"])


class TestSortBlockedDuringProcess(SortTestBase):
    def test_sort_start_blocked_while_process_running(self):
        # All 7 pipeline stages are replaced with fast stubs (see
        # ProcessTestBase.patch_fast_stages in test_ui_process.py) — only "index" is
        # blocked, so running stays True long enough to check the 409, without the
        # test paying seconds for real ML.
        block = threading.Event()

        def fake_index(cfg, conn, progress=None):
            block.wait(timeout=5)

        def fake_noop(*args, **kwargs):
            return None

        def fake_assign_duplicates(conn, strategy):
            return 0

        def fake_phash(cfg, conn, progress=None):
            return 0

        for name, fn in (
            ("run_index", fake_index),
            ("assign_duplicates", fake_assign_duplicates),
            ("resolve_places", fake_noop),
            ("detect_landmarks", fake_noop),
            ("detect_and_cluster", fake_noop),
            ("build_events", fake_noop),
            ("name_events", fake_noop),
            ("classify_junk", fake_noop),
            ("compute_phashes", fake_phash),
        ):
            patcher = mock.patch.object(ui, name, fn)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        try:
            status1, resp1 = self.post("/api/process", {"source_dir": str(self.src_dir)})
            self.assertEqual(status1, 200)
            self.assertTrue(resp1.get("ok"))
            _poll_until(self.process_status,
                       lambda d: d["running"] and d["stage"] == "index")

            status2, resp2 = self.post(
                "/api/sort", {"dest": str(self.root / "dest"), "mode": "move"})
            self.assertEqual(status2, 409)
            self.assertIn("error", resp2)
        finally:
            block.set()
        _poll_until(self.process_status, lambda d: d["finished"])


class TestSortHtml(SortTestBase):
    def test_city_tab_has_sort_controls_and_confirm_texts(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="sort-dest"', html)
        self.assertIn('name="sort-mode"', html)
        self.assertIn('value="move"', html)
        self.assertIn('value="copy"', html)
        self.assertIn('id="sort-apply-btn"', html)
        self.assertIn('id="sort-warning"', html)
        self.assertIn("/api/sort", html)
        # window.I18N contains the apply-warning translations (F43)
        self.assertIn("sort_confirm_move", html)
        self.assertIn("sort_confirm_inplace", html)
        self.assertIn("sort_confirm_copy", html)
        # F45: a warning about a stale preview plan (rebuild ≠ an apply error)
        self.assertIn("sort_preview_stale_warning", html)
        self.assertIn("preview_stale", html)
        # the warning text — en by default: move is explicitly about moving the
        # originals, in-place is explicitly about the source tree (see brief item 2)
        self.assertIn("MOVED", html)
        self.assertIn("SOURCE tree", html)
        # U1 invariant (no external resources)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)

    def test_preview_stale_warning_translated_in_all_three_langs(self):
        entry = ui._UI_STRINGS["sort_preview_stale_warning"]
        self.assertEqual(set(entry.keys()), {"ru", "en", "ja"})
        for lang in ("ru", "en", "ja"):
            self.assertTrue(entry[lang].strip())


class TestSortRebuildFailureDoesNotMaskSuccess(SortTestBase):
    """F45: a PlanCache.rebuild failure after a successful apply — not a layout
    error. The files are already moved/copied (the journal is written by the engine),
    so status must return result (not error) + a soft preview_stale."""

    def test_rebuild_failure_after_apply_keeps_result_and_flags_preview_stale(self):
        _fid, src_path, _content = self.add_photo_file(
            "a.jpg", country="ru", city="Moscow")
        real_rebuild = ui.PlanCache.rebuild
        calls = {"n": 0}

        def flaky_rebuild(cache_self, cfg, conn):
            calls["n"] += 1
            if calls["n"] == 1:
                # the first call — building the cache at server startup (build_server),
                # must go through normally, otherwise the server would not come up at all.
                return real_rebuild(cache_self, cfg, conn)
            raise RuntimeError("rebuild boom")

        with mock.patch.object(ui.PlanCache, "rebuild", flaky_rebuild):
            self.start_server()
            dest = self.root / "dest"

            status, resp = self.post("/api/sort", {"dest": str(dest), "mode": "copy"})
            self.assertEqual(status, 200)
            self.assertTrue(resp.get("ok"))

            final = _poll_until(self.sort_status, lambda d: d["finished"])

        self.assertGreaterEqual(calls["n"], 2)
        # apply went through — the file is physically copied, error is empty, result is present
        self.assertIsNone(final["error"])
        self.assertIsNotNone(final["result"])
        self.assertEqual(final["result"]["moved"], 1)
        self.assertTrue(final["result"]["preview_stale"])
        self.assertTrue(src_path.exists())
        copied = list(dest.rglob("*.jpg"))
        self.assertEqual(len(copied), 1)


if __name__ == "__main__":
    unittest.main()
