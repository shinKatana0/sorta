"""F54: conditional visibility of the "People"/"Events" tabs — by data presence in
the DB (variant B, stateless, without a meta table) — GET /api/tabs/visibility + markup."""
from __future__ import annotations

import json
import unittest

from tests.test_ui import UiServerTestBase


class TabsVisibilityTestBase(UiServerTestBase):
    """Face/event fixtures — the same sources as _clusters_payload/_events_payload."""

    def add_clustered_face(self, file_id: int, *, label: str | None = "Alice") -> int:
        cur = self.conn.execute(
            "INSERT INTO face_clusters (label) VALUES (?)", (label,))
        cluster_id = cur.lastrowid
        self.conn.execute(
            """INSERT INTO faces (file_id, bbox, embedding, cluster_id)
               VALUES (?, '[0,0,10,10]', ?, ?)""",
            (file_id, b"embedding", cluster_id),
        )
        self.conn.commit()
        return cluster_id

    def add_noise_face(self, file_id: int) -> None:
        self.conn.execute(
            """INSERT INTO faces (file_id, bbox, embedding, cluster_id)
               VALUES (?, '[0,0,10,10]', ?, NULL)""",
            (file_id, b"embedding"),
        )
        self.conn.commit()

    def add_event(self, file_id: int, *, name: str = "День рождения",
                 started_at: str = "2023-06-01T09:00:00") -> int:
        cur = self.conn.execute(
            """INSERT INTO events (started_at, ended_at, name, name_is_manual, origin)
               VALUES (?, ?, ?, 0, 'auto')""",
            (started_at, started_at, name),
        )
        event_id = cur.lastrowid
        self.conn.execute(
            "INSERT INTO event_files (event_id, file_id) VALUES (?, ?)",
            (event_id, file_id),
        )
        self.conn.commit()
        return event_id


class TestApiTabsVisibility(TabsVisibilityTestBase):
    def test_empty_db_hides_both(self):
        self.start_server()
        status, body, ctype = self.get("/api/tabs/visibility")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        self.assertEqual(json.loads(body), {"person": False, "event": False})

    def test_clustered_faces_show_person_only(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.add_clustered_face(fid)
        self.start_server()
        _status, body, _ctype = self.get("/api/tabs/visibility")
        self.assertEqual(json.loads(body), {"person": True, "event": False})

    def test_noise_only_faces_do_not_show_person(self):
        fid, _p, _c = self.add_photo_file("noise.jpg")
        self.add_noise_face(fid)
        self.start_server()
        _status, body, _ctype = self.get("/api/tabs/visibility")
        self.assertEqual(json.loads(body), {"person": False, "event": False})

    def test_events_show_event_only(self):
        fid, _p, _c = self.add_photo_file("b.jpg")
        self.add_event(fid)
        self.start_server()
        _status, body, _ctype = self.get("/api/tabs/visibility")
        self.assertEqual(json.loads(body), {"person": False, "event": True})

    def test_both_independent_when_both_present(self):
        fid1, _p1, _c1 = self.add_photo_file("a.jpg")
        fid2, _p2, _c2 = self.add_photo_file("b.jpg")
        self.add_clustered_face(fid1)
        self.add_event(fid2)
        self.start_server()
        _status, body, _ctype = self.get("/api/tabs/visibility")
        self.assertEqual(json.loads(body), {"person": True, "event": True})


class TestTabsVisibilityMarkup(TabsVisibilityTestBase):
    def test_person_event_buttons_hidden_by_default(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="tab-btn-person" style="display:none"', html)
        self.assertIn('id="tab-btn-event" style="display:none"', html)

    def test_js_has_apply_tab_visibility_wired_in(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("function applyTabVisibility", html)
        self.assertIn('"/api/tabs/visibility"', html)
        self.assertIn("applyTabVisibility()", html)
        # called both on init and inside refreshTabsAfterProcess
        refresh_idx = html.index("function refreshTabsAfterProcess")
        refresh_body_end = html.index("}", refresh_idx)
        self.assertIn("applyTabVisibility()", html[refresh_idx:refresh_body_end])

    def test_no_external_resources(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)


if __name__ == "__main__":
    unittest.main()
