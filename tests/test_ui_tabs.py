"""U2: the People/Events tabs — /api/plan?mode=person|event + HTML tabs."""
from __future__ import annotations

import json
import unittest

from tests.test_ui import UiServerTestBase


class PersonEventTestBase(UiServerTestBase):
    """Face/event fixtures on top of the base U1 server."""

    def add_face(self, file_id: int, *, label: str | None = "Alice") -> int:
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


class TestApiPlanPersonEvent(PersonEventTestBase):
    def test_person_mode_returns_expected_fields(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.add_face(fid, label="Alice")
        self.start_server()
        status, body, ctype = self.get("/api/plan?mode=person")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        items = json.loads(body)
        self.assertEqual({it["file_id"] for it in items}, {fid})
        expected_keys = {"file_id", "name", "target_rel", "reason", "date",
                         "geo", "category", "thumb_url"}
        for item in items:
            self.assertEqual(expected_keys, set(item.keys()))
        self.assertTrue(any(it["target_rel"].startswith("Alice/") for it in items))

    def test_event_mode_returns_expected_fields(self):
        fid, _p, _c = self.add_photo_file("b.jpg")
        self.add_event(fid, name="День рождения", started_at="2023-06-01T09:00:00")
        self.start_server()
        status, body, ctype = self.get("/api/plan?mode=event")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        items = json.loads(body)
        self.assertEqual({it["file_id"] for it in items}, {fid})
        for item in items:
            self.assertTrue(item["target_rel"].startswith("2023/"))

    def test_unknown_mode_still_returns_400(self):
        self.start_server()
        status, body, _ctype = self.get("/api/plan?mode=not_a_real_mode")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))


class TestIndexHtmlPersonEventTabs(PersonEventTestBase):
    def test_person_and_event_tabs_present(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="tab-btn-person"', html)
        self.assertIn('id="tab-btn-event"', html)
        self.assertIn(">People<", html)
        self.assertIn(">Events<", html)
        self.assertIn('id="tab-person"', html)
        self.assertIn('id="tab-event"', html)
        self.assertIn("renderPlanTab", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<link", html)


if __name__ == "__main__":
    unittest.main()
