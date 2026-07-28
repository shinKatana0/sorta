"""F97: cancelling a layout and rolling the last batch back from the UI.

`POST /api/sort/cancel`, `POST /api/undo` + `GET /api/undo/status` +
`POST /api/undo/cancel`, the cross-locking of all three heavy operations, and the
controls of the "Moves" tab.

Happy-path tests call the real `sorter` engine without mocks — files are physically
copied/moved into tmp-dest and physically rolled back. For the race and cancel tests
`ui.plan_and_sort` / `ui.undo` are replaced with blocking stubs, the same trick
`SortBlockingTestBase` uses in test_ui_sort.py: a real 220 GB copy is what makes the
cancel button necessary, and what makes it untestable at real speed.
"""
from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from unittest import mock

from sorta import ui
from sorta.sorter import SortReport, UndoStats

from tests.test_ui_sort import SortTestBase, _poll_until


class UndoTestBase(SortTestBase):
    def undo_status(self) -> dict:
        status, body, _ctype = self.get("/api/undo/status")
        self.assertEqual(status, 200)
        return json.loads(body)

    def path_of(self, file_id: int) -> str:
        return self.conn.execute(
            "SELECT path FROM files WHERE id = ?", (file_id,)).fetchone()["path"]

    def sort_and_wait(self, dest: Path, mode: str) -> dict:
        status, resp = self.post("/api/sort", {"dest": str(dest), "mode": mode})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))
        final = _poll_until(self.sort_status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        return final["result"]

    def undo_and_wait(self) -> dict:
        status, resp = self.post("/api/undo", {})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))
        return _poll_until(self.undo_status, lambda d: d["finished"])


class TestUndoCopyBatch(UndoTestBase):
    def test_copies_are_deleted_and_the_originals_are_untouched(self):
        fid1, p1, _c1 = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        fid2, p2, _c2 = self.add_photo_file("b.jpg", country="ru", city="Moscow")
        self.start_server()
        dest = self.root / "dest"
        before = (self.path_of(fid1), self.path_of(fid2))

        self.sort_and_wait(dest, "copy")
        self.assertEqual(len(list(dest.rglob("*.jpg"))), 2)

        final = self.undo_and_wait()
        self.assertIsNone(final["error"])
        result = final["result"]
        self.assertEqual(result["undone"], 2)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["stray"], [])
        self.assertFalse(result["cancelled"])

        self.assertEqual(list(dest.rglob("*.jpg")), [])
        self.assertTrue(p1.exists())
        self.assertTrue(p2.exists())
        # copy mode: files.path never pointed at the copies, and undo must not move it
        self.assertEqual((self.path_of(fid1), self.path_of(fid2)), before)

    def test_a_second_rollback_of_the_same_batch_is_a_no_op(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        self.sort_and_wait(self.root / "dest", "copy")
        self.assertEqual(self.undo_and_wait()["result"]["undone"], 1)
        again = self.undo_and_wait()
        self.assertEqual(again["result"]["undone"], 0)
        self.assertIsNone(again["error"])


class TestUndoMoveBatch(UndoTestBase):
    def test_files_come_back_and_files_path_is_updated(self):
        fid, p1, _c1 = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        dest = self.root / "dest"
        original = self.path_of(fid)

        self.sort_and_wait(dest, "move")
        self.assertFalse(p1.exists())
        self.assertNotEqual(self.path_of(fid), original)

        final = self.undo_and_wait()
        self.assertEqual(final["result"]["undone"], 1)
        self.assertTrue(p1.exists())
        self.assertEqual(self.path_of(fid), original)
        self.assertEqual(list(dest.rglob("*.jpg")), [])


class TestUndoWithoutBatches(UndoTestBase):
    def test_no_batches_reports_an_error_without_crashing_the_server(self):
        self.start_server()
        status, resp = self.post("/api/undo", {})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))
        final = _poll_until(self.undo_status, lambda d: d["finished"])
        self.assertIsNotNone(final["error"])
        self.assertIsNone(final["result"])
        status, _body, _ctype = self.get("/")
        self.assertEqual(status, 200)


class TestUndoStatusShape(UndoTestBase):
    def test_idle_status_before_any_run(self):
        self.start_server()
        data = self.undo_status()
        self.assertEqual(
            set(data.keys()),
            {"running", "done", "total", "error", "finished", "result",
             "cancel_requested"})
        self.assertFalse(data["running"])
        self.assertFalse(data["finished"])
        self.assertIsNone(data["result"])

    def test_result_shape_after_a_real_rollback(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        self.sort_and_wait(self.root / "dest", "copy")
        result = self.undo_and_wait()["result"]
        self.assertEqual(
            set(result.keys()),
            {"batch_id", "undone", "missing", "failed", "cancelled", "stray"})


class UndoBlockingTestBase(UndoTestBase):
    """Replaces `ui.undo` with a stub that blocks until released — for the race and
    cancel tests only."""

    def patch_blocking_undo(self, block_event: threading.Event) -> list:
        calls: list = []

        def fake_undo(conn, batch_id=None, progress=None, should_cancel=None):
            calls.append(batch_id)
            if progress:
                progress(0, 1)
            block_event.wait(timeout=5)
            return UndoStats(batch_id=batch_id or 0, undone=1,
                             cancelled=bool(should_cancel and should_cancel()))

        patcher = mock.patch.object(ui, "undo", fake_undo)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def patch_blocking_sort(self, block_event: threading.Event) -> list:
        calls: list = []

        def fake_plan_and_sort(cfg, conn, mode, dest, apply=False, copy=False,
                               progress=None, should_cancel=None, **kwargs):
            calls.append((mode, dest, apply, copy))
            if progress:
                progress(0, 1)
            block_event.wait(timeout=5)
            return SortReport(
                mode=mode, dest=Path(dest) if dest else Path(cfg.sources[0]),
                csv_path=self.root / "plan.csv", html_path=self.root / "plan.html",
                moved=1, failed=0, skipped_in_place=0, dirs=1,
                cancelled=bool(should_cancel and should_cancel()),
            )

        patcher = mock.patch.object(ui, "plan_and_sort", fake_plan_and_sort)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls


class TestUndoConcurrency(UndoBlockingTestBase):
    """A rollback changes paths of files on disk — it may not overlap a layout or a
    pipeline run, in either direction."""

    def _make_batch(self) -> None:
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.conn.execute(
            "INSERT INTO move_batches (mode, dest_root, started_at, operation) "
            "VALUES ('city', ?, '2026-01-01T10:00:00', 'copy')",
            (str(self.root / "dest"),))
        self.conn.commit()

    def test_second_undo_post_while_running_returns_409(self):
        block = threading.Event()
        self.patch_blocking_undo(block)
        self._make_batch()
        self.start_server()
        try:
            status1, resp1 = self.post("/api/undo", {})
            self.assertEqual(status1, 200)
            self.assertTrue(resp1.get("ok"))
            _poll_until(self.undo_status, lambda d: d["running"])

            status2, resp2 = self.post("/api/undo", {})
            self.assertEqual(status2, 409)
            self.assertIn("error", resp2)
        finally:
            block.set()
        _poll_until(self.undo_status, lambda d: d["finished"])

    def test_sort_start_blocked_while_undo_running(self):
        block = threading.Event()
        self.patch_blocking_undo(block)
        self._make_batch()
        self.start_server()
        try:
            self.assertEqual(self.post("/api/undo", {})[0], 200)
            _poll_until(self.undo_status, lambda d: d["running"])

            status, resp = self.post(
                "/api/sort", {"dest": str(self.root / "dest"), "mode": "copy"})
            self.assertEqual(status, 409)
            self.assertIn("error", resp)
        finally:
            block.set()
        _poll_until(self.undo_status, lambda d: d["finished"])

    def test_process_start_blocked_while_undo_running(self):
        block = threading.Event()
        self.patch_blocking_undo(block)
        self._make_batch()
        self.start_server()
        try:
            self.assertEqual(self.post("/api/undo", {})[0], 200)
            _poll_until(self.undo_status, lambda d: d["running"])

            status, resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
            self.assertEqual(status, 409)
            self.assertIn("error", resp)
        finally:
            block.set()
        _poll_until(self.undo_status, lambda d: d["finished"])

    def test_undo_start_blocked_while_sort_running(self):
        block = threading.Event()
        self.patch_blocking_sort(block)
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        try:
            status1, _resp1 = self.post(
                "/api/sort", {"dest": str(self.root / "dest"), "mode": "copy"})
            self.assertEqual(status1, 200)
            self.assertTrue(self.sort_status()["running"])

            status2, resp2 = self.post("/api/undo", {})
            self.assertEqual(status2, 409)
            self.assertIn("error", resp2)
        finally:
            block.set()
        _poll_until(self.sort_status, lambda d: d["finished"])


class TestUndoCancel(UndoBlockingTestBase):
    def test_cancel_flag_reaches_the_engine_and_the_result_says_cancelled(self):
        block = threading.Event()
        self.patch_blocking_undo(block)
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.conn.execute(
            "INSERT INTO move_batches (mode, dest_root, started_at, operation) "
            "VALUES ('city', ?, '2026-01-01T10:00:00', 'copy')",
            (str(self.root / "dest"),))
        self.conn.commit()
        self.start_server()
        try:
            self.assertEqual(self.post("/api/undo", {})[0], 200)
            _poll_until(self.undo_status, lambda d: d["running"])

            status, resp = self.post("/api/undo/cancel", {})
            self.assertEqual(status, 200)
            self.assertTrue(resp.get("ok"))
            self.assertTrue(
                _poll_until(self.undo_status, lambda d: d["cancel_requested"])["running"])
        finally:
            block.set()
        final = _poll_until(self.undo_status, lambda d: d["finished"])
        self.assertTrue(final["result"]["cancelled"])

    def test_cancel_before_any_run_does_not_arm_the_flag(self):
        # request_cancel only bites while running — otherwise the flag would survive
        # into the NEXT rollback and stop it before it started
        self.start_server()
        status, resp = self.post("/api/undo/cancel", {})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))
        self.assertFalse(self.undo_status()["cancel_requested"])


class TestSortCancel(UndoBlockingTestBase):
    def test_cancel_flag_reaches_plan_and_sort_and_the_result_says_cancelled(self):
        block = threading.Event()
        self.patch_blocking_sort(block)
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        try:
            status, _resp = self.post(
                "/api/sort", {"dest": str(self.root / "dest"), "mode": "copy"})
            self.assertEqual(status, 200)
            _poll_until(self.sort_status, lambda d: d["running"])

            status, resp = self.post("/api/sort/cancel", {})
            self.assertEqual(status, 200)
            self.assertTrue(resp.get("ok"))
            _poll_until(self.sort_status, lambda d: d["cancel_requested"])
        finally:
            block.set()
        final = _poll_until(self.sort_status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertTrue(final["result"]["cancelled"])

    def test_real_apply_can_be_cancelled_and_then_resumed_without_duplicates(self):
        # end to end through the real engine: cancel on the first file, then a plain
        # second apply finishes the job and creates no `_1` twins
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            self.add_photo_file(name, country="ru", city="Moscow")
        self.start_server()
        dest = self.root / "dest"

        real_plan_and_sort = ui.plan_and_sort
        seen = {"n": 0}

        def cancel_after_the_first_file() -> bool:
            seen["n"] += 1
            return seen["n"] > 1

        def cancelling(*args, **kwargs):
            kwargs["should_cancel"] = cancel_after_the_first_file
            return real_plan_and_sort(*args, **kwargs)

        with mock.patch.object(ui, "plan_and_sort", cancelling):
            first = self.sort_and_wait(dest, "copy")
        self.assertTrue(first["cancelled"])
        self.assertEqual(first["moved"], 1)

        second = self.sort_and_wait(dest, "copy")
        self.assertFalse(second["cancelled"])
        self.assertEqual(second["moved"], 2)
        self.assertEqual(second["skipped_already_copied"], 1)
        self.assertEqual(sorted(p.name for p in dest.rglob("*.jpg")),
                         ["a.jpg", "b.jpg", "c.jpg"])


class TestUndoHtml(UndoTestBase):
    def test_moves_tab_has_the_rollback_controls_and_the_confirm_dialog(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="undo-btn"', html)
        self.assertIn('id="undo-cancel-btn"', html)
        self.assertIn('id="undo-progress"', html)
        self.assertIn('id="undo-status"', html)
        self.assertIn('id="undo-stray"', html)
        self.assertIn('id="undo-dialog"', html)
        self.assertIn('id="undo-dialog-ok"', html)
        self.assertIn("/api/undo", html)
        self.assertIn("/api/undo/cancel", html)
        self.assertIn("/api/undo/status", html)
        # F104: the "Cities" tab no longer offers a second entry point into the
        # rollback — the manifest that says WHAT is being rolled back lives here, and a
        # rollback from the plan screen is a rollback blind. The hint pointing at this
        # tab after a cancelled layout stays (sort_undo_hint), the button does not.
        self.assertNotIn('id="sort-undo-btn"', html)
        self.assertIn('id="sort-cancel-btn"', html)
        self.assertIn("/api/sort/cancel", html)
        # U1 invariant (no external resources)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)

    def test_the_ui_no_longer_sends_the_user_to_the_terminal_for_a_rollback(self):
        # F97 item 9: where there is a button now, the text must name the button
        for lang in ("ru", "en", "ja"):
            html = ui._render_index_html(lang)
            self.assertNotIn("sorta undo", html)

    def test_every_new_string_exists_in_all_three_languages(self):
        keys = [k for k in ui._UI_STRINGS if k.startswith("undo_")
                or k in ("sort_cancel_button", "sort_cancel_requested",
                         "sort_cancelled_text", "sort_already_copied_note",
                         "sort_undo_hint")]
        self.assertGreater(len(keys), 10)
        for key in keys:
            entry = ui._UI_STRINGS[key]
            self.assertEqual(set(entry.keys()), {"ru", "en", "ja"}, key)
            for lang in ("ru", "en", "ja"):
                self.assertTrue(entry[lang].strip(), f"{key}/{lang}")


if __name__ == "__main__":
    unittest.main()
