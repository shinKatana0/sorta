"""U4: the "Delete" button on a frame -> POST /api/photo/trash (the single trash path)."""
from __future__ import annotations

import unittest
from unittest import mock

from tests.test_ui_dupes import DupesTestBase


class TestApiPhotoTrash(DupesTestBase):
    def test_trash_deletes_row_clears_dedup_choice_and_calls_send2trash(self):
        fid = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=1000)
        path = self.conn.execute(
            "SELECT path FROM files WHERE id = ?", (fid,)).fetchone()["path"]
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at) VALUES (?, 'keep', 'now')",
            (fid,),
        )
        self.conn.commit()
        self.start_server()
        with mock.patch("sorta.ui.send_to_trash") as mock_trash:
            status, payload = self.post("/api/photo/trash", {"file_id": fid})
        self.assertEqual(status, 200)
        mock_trash.assert_called_once_with(path)
        self.assertEqual(payload["trashed"], [{"file_id": fid, "name": "a.jpg"}])

        row = self.conn.execute("SELECT id FROM files WHERE id = ?", (fid,)).fetchone()
        self.assertIsNone(row)
        choice = self.conn.execute(
            "SELECT file_id FROM dedup_choice WHERE file_id = ?", (fid,)).fetchone()
        self.assertIsNone(choice)

    def test_unknown_file_id_returns_empty_trashed_without_error(self):
        self.start_server()
        with mock.patch("sorta.ui.send_to_trash") as mock_trash:
            status, payload = self.post("/api/photo/trash", {"file_id": 999999})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"trashed": []})
        mock_trash.assert_not_called()

    def test_missing_file_id_returns_400(self):
        self.start_server()
        status, payload = self.post("/api/photo/trash", {})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_non_integer_file_id_returns_400(self):
        self.start_server()
        status, payload = self.post("/api/photo/trash", {"file_id": "1"})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_bool_file_id_rejected(self):
        self.start_server()
        status, payload = self.post("/api/photo/trash", {"file_id": True})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_body_with_path_instead_of_file_id_returns_400_and_nothing_is_read(self):
        # path-security: the client cannot slip a path in place of file_id — a body
        # without a valid file_id is always 400, send2trash is never called.
        self.start_server()
        with mock.patch("sorta.ui.send_to_trash") as mock_trash:
            status, payload = self.post(
                "/api/photo/trash", {"path": "../../../etc/passwd"})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        mock_trash.assert_not_called()

    def test_non_dict_body_returns_400(self):
        self.start_server()
        status, payload = self.post("/api/photo/trash", [1, 2, 3])
        self.assertEqual(status, 400)
        self.assertIn("error", payload)


class TestApiPhotosTrash(DupesTestBase):
    """Bulk deletion of the selected: POST /api/photos/trash (the shared _trash_files)."""

    def test_bulk_trash_deletes_all_selected(self):
        a = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=1000)
        b = self.add_dupe("b.jpg", phash="1" * 16, width=100, height=100, size=1000)
        self.start_server()
        with mock.patch("sorta.ui.send_to_trash") as mock_trash:
            status, payload = self.post("/api/photos/trash", {"file_ids": [a, b]})
        self.assertEqual(status, 200)
        self.assertEqual(mock_trash.call_count, 2)
        self.assertEqual({t["file_id"] for t in payload["trashed"]}, {a, b})
        rows = self.conn.execute("SELECT id FROM files WHERE id IN (?, ?)", (a, b)).fetchall()
        self.assertEqual(rows, [])

    def test_bulk_ignores_unknown_ids(self):
        a = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=1000)
        self.start_server()
        with mock.patch("sorta.ui.send_to_trash"):
            status, payload = self.post("/api/photos/trash", {"file_ids": [a, 999999]})
        self.assertEqual(status, 200)
        self.assertEqual([t["file_id"] for t in payload["trashed"]], [a])

    def test_empty_list_returns_400(self):
        self.start_server()
        status, payload = self.post("/api/photos/trash", {"file_ids": []})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_non_list_returns_400(self):
        self.start_server()
        status, payload = self.post("/api/photos/trash", {"file_ids": 5})
        self.assertEqual(status, 400)

    def test_non_int_member_returns_400(self):
        self.start_server()
        with mock.patch("sorta.ui.send_to_trash") as mock_trash:
            status, _payload = self.post("/api/photos/trash", {"file_ids": [1, "2"]})
        self.assertEqual(status, 400)
        mock_trash.assert_not_called()

    def test_bool_member_rejected(self):
        self.start_server()
        status, _payload = self.post("/api/photos/trash", {"file_ids": [True]})
        self.assertEqual(status, 400)


class TestPhotoDeleteButtonInHtml(DupesTestBase):
    def test_delete_button_present_and_shared_session_flag(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="delete-remember"', html)
        # exactly one session "do not ask" checkbox for the whole app
        self.assertEqual(html.count('id="delete-remember"'), 1)
        self.assertIn("deletePhoto(item.file_id", html)
        self.assertIn("deletePhoto(f.file_id", html)
        self.assertIn("/api/photo/trash", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)

    def test_bulk_delete_selected_controls_present(self):
        # per-row selection checkboxes + the "Delete selected" button + bulk endpoint
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="city-delete-selected-btn"', html)
        self.assertIn('"row-select"', html)
        self.assertIn("/api/photos/trash", html)
        self.assertIn("wireBulkDelete(", html)


if __name__ == "__main__":
    unittest.main()
