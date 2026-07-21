"""F35: the "Collect into folder" buttons (person/event albums) on top of F34.

POST /api/album (preview apply=false / apply=true), GET /api/events, body
validation, HTML buttons on the People/Events cards. The server runs in a thread
(see test_ui.py).
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from tests.test_ui import UiServerTestBase


class AlbumsTestBase(UiServerTestBase):
    """Face-cluster + event fixtures on top of the base U1 server (F35)."""

    def add_cluster(self, *, label: str | None = None, merged_into: int | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO face_clusters (label, merged_into) VALUES (?, ?)",
            (label, merged_into),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_face(self, file_id: int, cluster_id: int, bbox: str = "[0,0,10,10]") -> int:
        cur = self.conn.execute(
            """INSERT INTO faces (file_id, bbox, embedding, cluster_id)
               VALUES (?, ?, ?, ?)""",
            (file_id, bbox, b"embedding", cluster_id),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_event(self, name: str, started_at: str = "2022-05-01T09:00:00",
                 ended_at: str = "2022-05-01T20:00:00") -> int:
        cur = self.conn.execute(
            """INSERT INTO events (started_at, ended_at, name, name_is_manual, origin)
               VALUES (?, ?, ?, 0, 'auto')""",
            (started_at, ended_at, name),
        )
        event_id = cur.lastrowid
        self.conn.commit()
        return event_id

    def link_event_file(self, event_id: int, file_id: int) -> None:
        self.conn.execute(
            "INSERT INTO event_files (event_id, file_id) VALUES (?, ?)", (event_id, file_id))
        self.conn.commit()

    def post(self, path: str, data: dict) -> tuple[int, dict]:
        import urllib.error
        import urllib.request

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


class TestApiAlbumPersonPreview(AlbumsTestBase):
    def test_preview_counts_without_writing(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        cluster = self.add_cluster(label="Мама")
        self.add_face(fid, cluster)
        self.start_server()
        status, body = self.post(
            "/api/album", {"kind": "person", "selector": "Мама", "mode": "link", "apply": False})
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["applied"], False)
        self.assertEqual(body["transferred"], 0)
        self.assertEqual(body["blocked_multi"], 0)
        self.assertEqual(body["kind"], "person")
        self.assertEqual(body["mode"], "link")
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0], 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)


class TestApiAlbumPersonApply(AlbumsTestBase):
    def test_apply_link_creates_hardlink_and_batch(self):
        fid, p, _content = self.add_photo_file("a.jpg")
        cluster = self.add_cluster(label="Мама")
        self.add_face(fid, cluster)
        self.start_server()
        status, body = self.post(
            "/api/album", {"kind": "person", "selector": "Мама", "mode": "link", "apply": True})
        self.assertEqual(status, 200)
        self.assertEqual(body["transferred"], 1)
        self.assertEqual(body["failed"], 0)
        self.assertTrue(body["applied"])
        dst = Path(body["dest"]) / p.name
        self.assertTrue(dst.exists())
        self.assertGreaterEqual(os.stat(dst).st_nlink, 2)
        batch = self.conn.execute(
            "SELECT mode, operation FROM move_batches ORDER BY id DESC LIMIT 1").fetchone()
        self.assertEqual(batch["mode"], "album_person")
        self.assertEqual(batch["operation"], "link")


class TestApiAlbumMoveBlocked(AlbumsTestBase):
    def test_move_blocks_multi_person_files(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        mama = self.add_cluster(label="Мама")
        papa = self.add_cluster(label="Папа")
        self.add_face(fid, mama)
        self.add_face(fid, papa, bbox="[20,20,30,30]")
        self.start_server()
        status, body = self.post(
            "/api/album", {"kind": "person", "selector": "Мама", "mode": "move", "apply": True})
        self.assertEqual(status, 200)
        self.assertEqual(body["blocked_multi"], 1)
        self.assertEqual(body["transferred"], 0)
        dst = Path(body["dest"]) / "a.jpg"
        self.assertFalse(dst.exists())

    def test_move_preview_shows_blocked_multi_without_writing(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        mama = self.add_cluster(label="Мама")
        papa = self.add_cluster(label="Папа")
        self.add_face(fid, mama)
        self.add_face(fid, papa, bbox="[20,20,30,30]")
        self.start_server()
        status, body = self.post(
            "/api/album", {"kind": "person", "selector": "Мама", "mode": "move", "apply": False})
        self.assertEqual(status, 200)
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["blocked_multi"], 1)
        self.assertEqual(body["transferred"], 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0], 0)


class TestApiAlbumEvent(AlbumsTestBase):
    def test_event_album_with_name_override(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        eid = self.add_event("IEEE Conference on Whatever 2022")
        self.link_event_file(eid, fid)
        self.start_server()
        status, body = self.post(
            "/api/album",
            {"kind": "event", "selector": str(eid), "mode": "copy", "name": "IEEE", "apply": True})
        self.assertEqual(status, 200)
        self.assertEqual(body["album_name"], "IEEE")
        self.assertEqual(Path(body["dest"]).name, "IEEE")
        self.assertTrue((Path(body["dest"]) / "a.jpg").exists())

    def test_event_album_by_id_selector_default_name(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        eid = self.add_event("Свадьба")
        self.link_event_file(eid, fid)
        self.start_server()
        status, body = self.post(
            "/api/album",
            {"kind": "event", "selector": str(eid), "mode": "link", "apply": False})
        self.assertEqual(status, 200)
        self.assertEqual(body["album_name"], "Свадьба")
        self.assertEqual(body["count"], 1)


class TestApiEvents(AlbumsTestBase):
    def test_list_sorted_by_count_descending(self):
        fid1, _p1, _c1 = self.add_photo_file("a.jpg")
        fid2, _p2, _c2 = self.add_photo_file("b.jpg")
        fid3, _p3, _c3 = self.add_photo_file("c.jpg")
        small = self.add_event(
            "Small", started_at="2022-01-01T00:00:00", ended_at="2022-01-01T01:00:00")
        self.link_event_file(small, fid1)
        big = self.add_event(
            "Big", started_at="2022-02-01T00:00:00", ended_at="2022-02-02T00:00:00")
        self.link_event_file(big, fid2)
        self.link_event_file(big, fid3)
        self.start_server()
        status, body, ctype = self.get("/api/events")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        events = json.loads(body)
        self.assertEqual([e["id"] for e in events], [big, small])
        self.assertEqual(events[0]["name"], "Big")
        self.assertEqual(events[0]["count"], 2)
        self.assertEqual(events[1]["count"], 1)
        for e in events:
            self.assertIn("started_at", e)
            self.assertIn("ended_at", e)

    def test_no_events_returns_empty_list(self):
        self.start_server()
        status, body, _ctype = self.get("/api/events")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), [])

    def test_event_payload_includes_sample_file_ids(self):
        # event preview frames (clickable -> lightbox, uniform UI behaviour)
        fid1, _p1, _c1 = self.add_photo_file("a.jpg")
        fid2, _p2, _c2 = self.add_photo_file("b.jpg")
        ev = self.add_event(
            "Trip", started_at="2022-03-01T00:00:00", ended_at="2022-03-02T00:00:00")
        self.link_event_file(ev, fid1)
        self.link_event_file(ev, fid2)
        self.start_server()
        _status, body, _ctype = self.get("/api/events")
        events = json.loads(body)
        self.assertIn("samples", events[0])
        self.assertEqual(set(events[0]["samples"]), {fid1, fid2})


class TestApiAlbumValidation(AlbumsTestBase):
    def test_bad_kind_returns_400(self):
        self.start_server()
        status, body = self.post(
            "/api/album", {"kind": "city", "selector": "x", "mode": "link", "apply": False})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_bad_mode_returns_400(self):
        self.start_server()
        status, body = self.post(
            "/api/album", {"kind": "person", "selector": "x", "mode": "teleport"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_empty_selector_returns_400(self):
        self.start_server()
        status, body = self.post(
            "/api/album", {"kind": "person", "selector": "   ", "mode": "link"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_missing_selector_returns_400(self):
        self.start_server()
        status, body = self.post("/api/album", {"kind": "person", "mode": "link"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_where_not_list_of_strings_returns_400(self):
        self.start_server()
        status, body = self.post(
            "/api/album",
            {"kind": "person", "selector": "Мама", "mode": "link", "where": [1, 2]})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_name_not_string_returns_400(self):
        self.start_server()
        status, body = self.post(
            "/api/album",
            {"kind": "person", "selector": "Мама", "mode": "link", "name": 5})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_apply_not_bool_returns_400(self):
        self.start_server()
        status, body = self.post(
            "/api/album",
            {"kind": "person", "selector": "Мама", "mode": "link", "apply": "yes"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)


class TestIndexHtmlAlbumButtons(AlbumsTestBase):
    def test_named_cluster_has_album_controls(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        cluster = self.add_cluster(label="Мама")
        self.add_face(fid, cluster)
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("album-gather-btn", html)
        self.assertIn("album-controls", html)
        self.assertIn("album_name_first_hint", html)
        self.assertIn("/api/album", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)

    def test_event_tab_has_events_list_not_plan_tree(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="tab-event"', html)
        self.assertIn('id="events-list"', html)
        self.assertNotIn('id="tree-event"', html)
        self.assertIn("/api/events", html)
        self.assertIn("loadEvents", html)
        self.assertIn("gatherAlbum", html)


if __name__ == "__main__":
    unittest.main()
