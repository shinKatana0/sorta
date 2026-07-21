"""U3/F32: the Duplicates tab — /api/dupes, choice/choices(batch)/skip/trash, HTML."""
from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request
from unittest import mock

from tests.test_ui import UiServerTestBase


class DupesTestBase(UiServerTestBase):
    """Adds a near-dup file fixture on top of the base U1 server.

    near_duplicate_groups reads only the DB (path/size/phash), so the frames just
    need to exist on disk (for a real send2trash path target) — a valid JPEG is not
    required, since /thumb and /photo are not tested here.
    """

    def add_dupe(self, rel: str, *, phash: str, width: int, height: int, size: int) -> int:
        self._n += 1
        p = self.src_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-image-bytes")
        path = str(p.resolve())
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, hash, hash_algo,
                   phash, taken_at, taken_at_source, taken_at_confidence, width, height,
                   indexed_at)
               VALUES (?, ?, 0, 'jpg', 'photo', ?, 'blake3', ?, '2022-05-01T10:00:00',
                       'exif', 'high', ?, ?, '2026-01-01')""",
            (path, size, f"hash-{self._n}", phash, width, height),
        )
        self.conn.commit()
        return cur.lastrowid

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


class TestApiDupesGet(DupesTestBase):
    def test_groups_recommended_and_fields(self):
        # larger by w*h and by size -> recommended
        best = self.add_dupe("a.jpg", phash="0" * 16, width=4000, height=3000, size=9_000_000)
        worse = self.add_dupe("b.jpg", phash="0" * 16, width=1000, height=750, size=200_000)
        self.start_server()
        status, body, ctype = self.get("/api/dupes")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        groups = json.loads(body)
        self.assertEqual(len(groups), 1)
        frames = groups[0]["frames"]
        self.assertEqual({f["file_id"] for f in frames}, {best, worse})
        expected_keys = {"file_id", "name", "thumb_url", "width", "height",
                         "size", "recommended", "action"}
        for f in frames:
            self.assertEqual(expected_keys, set(f.keys()))
            self.assertEqual(f["thumb_url"], f"/thumb/{f['file_id']}")
            self.assertIsNone(f["action"])
        by_id = {f["file_id"]: f for f in frames}
        self.assertTrue(by_id[best]["recommended"])
        self.assertFalse(by_id[worse]["recommended"])

    def test_no_near_duplicates_returns_empty_list(self):
        self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=1000)
        self.start_server()
        status, body, _ctype = self.get("/api/dupes")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), [])

    def test_action_reflects_dedup_choice(self):
        fid1 = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=2000)
        fid2 = self.add_dupe("b.jpg", phash="0" * 16, width=100, height=100, size=1000)
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at) VALUES (?, 'keep', 'now')",
            (fid1,),
        )
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at) "
            "VALUES (?, 'to_delete', 'now')",
            (fid2,),
        )
        self.conn.commit()
        self.start_server()
        _status, body, _ctype = self.get("/api/dupes")
        by_id = {f["file_id"]: f for f in json.loads(body)[0]["frames"]}
        self.assertEqual(by_id[fid1]["action"], "keep")
        self.assertEqual(by_id[fid2]["action"], "to_delete")


class TestApiDupesChoice(DupesTestBase):
    def test_choice_writes_keep_and_to_delete(self):
        fid1 = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=2000)
        fid2 = self.add_dupe("b.jpg", phash="0" * 16, width=100, height=100, size=1000)
        self.start_server()
        status, payload = self.post("/api/dupes/choice",
                                    {"group": [fid1, fid2], "keep_file_id": fid1})
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"))
        rows = {r["file_id"]: r["action"] for r in
                self.conn.execute("SELECT file_id, action FROM dedup_choice").fetchall()}
        self.assertEqual(rows, {fid1: "keep", fid2: "to_delete"})

    def test_choice_reassigning_keeper_overwrites(self):
        fid1 = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=2000)
        fid2 = self.add_dupe("b.jpg", phash="0" * 16, width=100, height=100, size=1000)
        self.start_server()
        self.post("/api/dupes/choice", {"group": [fid1, fid2], "keep_file_id": fid1})
        status, payload = self.post("/api/dupes/choice",
                                    {"group": [fid1, fid2], "keep_file_id": fid2})
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"))
        rows = {r["file_id"]: r["action"] for r in
                self.conn.execute("SELECT file_id, action FROM dedup_choice").fetchall()}
        self.assertEqual(rows, {fid1: "to_delete", fid2: "keep"})

    def test_choice_idempotent(self):
        fid1 = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=2000)
        fid2 = self.add_dupe("b.jpg", phash="0" * 16, width=100, height=100, size=1000)
        self.start_server()
        self.post("/api/dupes/choice", {"group": [fid1, fid2], "keep_file_id": fid1})
        status, payload = self.post("/api/dupes/choice",
                                    {"group": [fid1, fid2], "keep_file_id": fid1})
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"))
        rows = self.conn.execute("SELECT COUNT(*) c FROM dedup_choice").fetchone()
        self.assertEqual(rows["c"], 2)

    def test_missing_keep_file_id_returns_400(self):
        fid1 = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=2000)
        self.start_server()
        status, payload = self.post("/api/dupes/choice", {"group": [fid1]})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_keep_file_id_not_in_group_returns_400(self):
        fid1 = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=2000)
        fid2 = self.add_dupe("b.jpg", phash="0" * 16, width=100, height=100, size=1000)
        self.start_server()
        status, payload = self.post("/api/dupes/choice",
                                    {"group": [fid1], "keep_file_id": fid2})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_empty_group_returns_400(self):
        self.start_server()
        status, payload = self.post("/api/dupes/choice", {"group": [], "keep_file_id": 1})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)


class TestApiDupesSkip(DupesTestBase):
    def test_skip_clears_group_decisions(self):
        fid1 = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=2000)
        fid2 = self.add_dupe("b.jpg", phash="0" * 16, width=100, height=100, size=1000)
        self.start_server()
        self.post("/api/dupes/choice", {"group": [fid1, fid2], "keep_file_id": fid1})
        status, payload = self.post("/api/dupes/skip", {"group": [fid1, fid2]})
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"))
        rows = self.conn.execute("SELECT COUNT(*) c FROM dedup_choice").fetchone()
        self.assertEqual(rows["c"], 0)


class TestApiDupesTrash(DupesTestBase):
    def test_trash_removes_non_keeper_files_and_sends_to_trash(self):
        fid1 = self.add_dupe("a.jpg", phash="0" * 16, width=4000, height=3000, size=2_000_000)
        fid2 = self.add_dupe("b.jpg", phash="0" * 16, width=100, height=100, size=1000)
        path2 = self.conn.execute(
            "SELECT path FROM files WHERE id = ?", (fid2,)).fetchone()["path"]
        self.start_server()
        with mock.patch("sorta.ui.send_to_trash") as mock_trash:
            status, payload = self.post(
                "/api/dupes/trash", {"group": [fid1, fid2], "keep_file_id": fid1})
        self.assertEqual(status, 200)
        mock_trash.assert_called_once_with(path2)
        self.assertEqual(payload["trashed"], [{"file_id": fid2, "name": "b.jpg"}])

        remaining_ids = {r["id"] for r in self.conn.execute("SELECT id FROM files").fetchall()}
        self.assertEqual(remaining_ids, {fid1})
        choice_ids = {r["file_id"] for r in
                      self.conn.execute("SELECT file_id FROM dedup_choice").fetchall()}
        self.assertNotIn(fid2, choice_ids)

    def test_trash_clears_dedup_choice_of_trashed_files(self):
        fid1 = self.add_dupe("a.jpg", phash="0" * 16, width=4000, height=3000, size=2_000_000)
        fid2 = self.add_dupe("b.jpg", phash="0" * 16, width=100, height=100, size=1000)
        self.start_server()
        self.post("/api/dupes/choice", {"group": [fid1, fid2], "keep_file_id": fid1})
        with mock.patch("sorta.ui.send_to_trash"):
            self.post("/api/dupes/trash", {"group": [fid1, fid2], "keep_file_id": fid1})
        rows = {r["file_id"]: r["action"] for r in
                self.conn.execute("SELECT file_id, action FROM dedup_choice").fetchall()}
        self.assertEqual(rows, {fid1: "keep"})

    def test_keeper_file_row_untouched(self):
        fid1 = self.add_dupe("a.jpg", phash="0" * 16, width=4000, height=3000, size=2_000_000)
        fid2 = self.add_dupe("b.jpg", phash="0" * 16, width=100, height=100, size=1000)
        self.start_server()
        with mock.patch("sorta.ui.send_to_trash") as mock_trash:
            self.post("/api/dupes/trash", {"group": [fid1, fid2], "keep_file_id": fid1})
        mock_trash.assert_called_once()
        row = self.conn.execute("SELECT id FROM files WHERE id = ?", (fid1,)).fetchone()
        self.assertIsNotNone(row)


class TestApiDupesChoicesBatch(DupesTestBase):
    def test_batch_saves_all_groups_in_one_request(self):
        fid1 = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=2000)
        fid2 = self.add_dupe("b.jpg", phash="0" * 16, width=100, height=100, size=1000)
        fid3 = self.add_dupe("c.jpg", phash="1" * 16, width=200, height=200, size=3000)
        fid4 = self.add_dupe("d.jpg", phash="1" * 16, width=100, height=100, size=500)
        self.start_server()
        status, payload = self.post("/api/dupes/choices", {"groups": [
            {"group": [fid1, fid2], "keep_file_id": fid1},
            {"group": [fid3, fid4], "keep_file_id": fid4},
        ]})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"saved": 2})
        rows = {r["file_id"]: r["action"] for r in
                self.conn.execute("SELECT file_id, action FROM dedup_choice").fetchall()}
        self.assertEqual(rows, {fid1: "keep", fid2: "to_delete",
                                fid3: "to_delete", fid4: "keep"})

    def test_batch_atomic_on_invalid_group(self):
        fid1 = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=2000)
        fid2 = self.add_dupe("b.jpg", phash="0" * 16, width=100, height=100, size=1000)
        fid3 = self.add_dupe("c.jpg", phash="1" * 16, width=200, height=200, size=3000)
        self.start_server()
        status, payload = self.post("/api/dupes/choices", {"groups": [
            {"group": [fid1, fid2], "keep_file_id": fid1},
            {"group": [fid3], "keep_file_id": 99999},
        ]})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        rows = self.conn.execute("SELECT COUNT(*) c FROM dedup_choice").fetchone()
        self.assertEqual(rows["c"], 0)

    def test_batch_empty_groups_returns_400(self):
        self.start_server()
        status, payload = self.post("/api/dupes/choices", {"groups": []})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_batch_non_int_keep_returns_400_and_nothing_written(self):
        fid1 = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=2000)
        fid2 = self.add_dupe("b.jpg", phash="0" * 16, width=100, height=100, size=1000)
        self.start_server()
        status, payload = self.post("/api/dupes/choices", {"groups": [
            {"group": [fid1, fid2], "keep_file_id": "not-an-int"},
        ]})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        rows = self.conn.execute("SELECT COUNT(*) c FROM dedup_choice").fetchone()
        self.assertEqual(rows["c"], 0)

    def test_batch_skips_listed_groups(self):
        fid1 = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=2000)
        fid2 = self.add_dupe("b.jpg", phash="0" * 16, width=100, height=100, size=1000)
        fid3 = self.add_dupe("c.jpg", phash="1" * 16, width=200, height=200, size=3000)
        fid4 = self.add_dupe("d.jpg", phash="1" * 16, width=100, height=100, size=500)
        self.start_server()
        self.post("/api/dupes/choice", {"group": [fid3, fid4], "keep_file_id": fid4})
        status, payload = self.post("/api/dupes/choices", {
            "groups": [{"group": [fid1, fid2], "keep_file_id": fid1}],
            "skip": [[fid3, fid4]],
        })
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"saved": 1})
        rows = {r["file_id"]: r["action"] for r in
                self.conn.execute("SELECT file_id, action FROM dedup_choice").fetchall()}
        self.assertEqual(rows, {fid1: "keep", fid2: "to_delete"})
        self.assertNotIn(fid3, rows)
        self.assertNotIn(fid4, rows)


class TestIndexHtmlTabs(DupesTestBase):
    def test_tabs_present_and_no_external_resources(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="tab-btn-city"', html)
        self.assertIn('id="tab-btn-dupes"', html)
        self.assertIn(">Cities<", html)
        self.assertIn(">Duplicates<", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)

    def test_batch_save_button_present_and_per_group_save_removed(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="dupes-save-all-btn"', html)
        self.assertIn("Save all choices", html)
        self.assertIn("/api/dupes/choices", html)
        self.assertNotIn("Save choice", html)
        self.assertIn("don't delete this group", html)


if __name__ == "__main__":
    unittest.main()
