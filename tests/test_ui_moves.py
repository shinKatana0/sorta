"""U5: the "Moves" tab — /api/moves (the sort --apply manifest) + HTML tab."""
from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path

from tests.test_ui import UiServerTestBase


class MovesTestBase(UiServerTestBase):
    """move_batches/moves fixtures on top of the base U1 server."""

    def add_batch(self, *, mode: str = "city", dest_root: str | None = None,
                 started_at: str = "2026-01-01T10:00:00",
                 finished_at: str | None = "2026-01-01T10:05:00",
                 operation: str = "move") -> int:
        if dest_root is None:
            dest_root = str(self.root / "dest")
        cur = self.conn.execute(
            """INSERT INTO move_batches (mode, dest_root, started_at, finished_at, operation)
               VALUES (?, ?, ?, ?, ?)""",
            (mode, dest_root, started_at, finished_at, operation),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_move(self, batch_id: int, file_id: int, src: str, dst: str,
                *, status: str = "done") -> int:
        cur = self.conn.execute(
            """INSERT INTO moves (batch_id, file_id, src, dst, hash, status)
               VALUES (?, ?, ?, ?, 'deadbeef', ?)""",
            (batch_id, file_id, src, dst, status),
        )
        self.conn.commit()
        return cur.lastrowid


class TestApiMoves(MovesTestBase):
    def test_no_batches_returns_empty(self):
        self.start_server()
        status, body, ctype = self.get("/api/moves")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        payload = json.loads(body)
        self.assertIsNone(payload["batch"])
        self.assertEqual(payload["moves"], [])

    def test_default_returns_last_batch(self):
        fid1, p1, _c1 = self.add_photo_file("a.jpg")
        fid2, p2, _c2 = self.add_photo_file("b.jpg")
        dest_root = str(self.root / "dest")
        b1 = self.add_batch(dest_root=dest_root)
        self.add_move(b1, fid1, str(p1), str(Path(dest_root) / "Moscow" / "a.jpg"))
        b2 = self.add_batch(dest_root=dest_root)
        self.add_move(b2, fid2, str(p2), str(Path(dest_root) / "Paris" / "b.jpg"))
        self.start_server()
        status, body, _ctype = self.get("/api/moves")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["batch"]["id"], b2)
        self.assertEqual([m["file_id"] for m in payload["moves"]], [fid2])

    def test_batch_query_selects_specific_batch(self):
        fid1, p1, _c1 = self.add_photo_file("a.jpg")
        fid2, p2, _c2 = self.add_photo_file("b.jpg")
        dest_root = str(self.root / "dest")
        b1 = self.add_batch(dest_root=dest_root)
        self.add_move(b1, fid1, str(p1), str(Path(dest_root) / "Moscow" / "a.jpg"))
        b2 = self.add_batch(dest_root=dest_root)
        self.add_move(b2, fid2, str(p2), str(Path(dest_root) / "Paris" / "b.jpg"))
        self.start_server()
        status, body, _ctype = self.get(f"/api/moves?batch={b1}")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["batch"]["id"], b1)
        self.assertEqual([m["file_id"] for m in payload["moves"]], [fid1])

    def test_move_fields_and_target_rel(self):
        fid, p, _c = self.add_photo_file("a.jpg")
        dest_root = str(self.root / "dest")
        dst = str(Path(dest_root) / "Moscow" / "2022" / "a.jpg")
        batch_id = self.add_batch(dest_root=dest_root)
        self.add_move(batch_id, fid, str(p), dst, status="done")
        self.start_server()
        status, body, _ctype = self.get("/api/moves")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(len(payload["moves"]), 1)
        item = payload["moves"][0]
        expected_keys = {"file_id", "name", "src", "dst", "target_rel", "status", "thumb_url"}
        self.assertEqual(expected_keys, set(item.keys()))
        self.assertEqual(item["file_id"], fid)
        self.assertEqual(item["name"], "a.jpg")
        self.assertEqual(item["dst"], dst)
        self.assertEqual(item["target_rel"], "Moscow/2022/a.jpg")
        self.assertEqual(item["status"], "done")
        self.assertEqual(item["thumb_url"], f"/thumb/{fid}")

    def test_batch_meta_fields(self):
        dest_root = str(self.root / "dest")
        batch_id = self.add_batch(
            mode="person", dest_root=dest_root, started_at="2026-02-01T09:00:00",
            finished_at="2026-02-01T09:10:00", operation="copy",
        )
        self.start_server()
        _status, body, _ctype = self.get(f"/api/moves?batch={batch_id}")
        payload = json.loads(body)
        batch = payload["batch"]
        self.assertEqual(batch["mode"], "person")
        self.assertEqual(batch["dest_root"], dest_root)
        self.assertEqual(batch["operation"], "copy")
        self.assertEqual(batch["started_at"], "2026-02-01T09:00:00")
        self.assertEqual(batch["finished_at"], "2026-02-01T09:10:00")

    def test_unknown_batch_id_returns_empty(self):
        self.add_batch()
        self.start_server()
        status, body, _ctype = self.get("/api/moves?batch=999999")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertIsNone(payload["batch"])
        self.assertEqual(payload["moves"], [])

    def test_trashed_file_stays_in_manifest_without_crash(self):
        """The files row deleted (trash after move) -> the manifest does not crash, name from dst."""
        fid, p, _c = self.add_photo_file("a.jpg")
        dest_root = str(self.root / "dest")
        dst = str(Path(dest_root) / "Moscow" / "a.jpg")
        batch_id = self.add_batch(dest_root=dest_root)
        self.add_move(batch_id, fid, str(p), dst, status="done")
        # emulates _trash_files (ui.py): a short-lived connection without
        # PRAGMA foreign_keys=ON, as in the real frame-deletion path.
        raw = sqlite3.connect(str(self.cfg.database))
        try:
            raw.execute("DELETE FROM files WHERE id = ?", (fid,))
            raw.commit()
        finally:
            raw.close()
        self.start_server()
        status, body, _ctype = self.get("/api/moves")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(len(payload["moves"]), 1)
        self.assertEqual(payload["moves"][0]["name"], "a.jpg")
        # without a preview (no files row), but not a 500
        thumb_status, _body, _ctype = self.get(f"/thumb/{fid}")
        self.assertEqual(thumb_status, 404)


class TestIndexHtmlMovesTab(MovesTestBase):
    def test_moves_tab_present(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="tab-btn-moves"', html)
        self.assertIn(">Moves<", html)
        self.assertIn('id="tab-moves"', html)
        self.assertIn('id="tree-moves"', html)
        self.assertIn("loadMoves", html)
        self.assertIn("renderMoveFiles", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)


if __name__ == "__main__":
    unittest.main()
