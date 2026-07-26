"""(0, 0) is a camera with no satellite fix, not a place.

From the validation run: 35 files carried exactly (0, 0) and were resolved to Ghana —
the nearest land to 0°N 0°E — and 16 more inherited that country through the
time-session rule. The user has never been to Ghana.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sorta.config import Config
from sorta.db import connect
from sorta.geo import _is_null_island, resolve_places


class TestNullIslandPredicate(unittest.TestCase):
    def test_exactly_zero_zero_is_rejected(self):
        self.assertTrue(_is_null_island(0.0, 0.0))

    def test_a_single_zero_is_a_real_coordinate(self):
        """The equator and Greenwich are perfectly real on their own."""
        self.assertFalse(_is_null_island(0.0, 37.6))     # on the equator
        self.assertFalse(_is_null_island(51.5, 0.0))     # on the prime meridian

    def test_near_zero_is_kept(self):
        """Only the exact sentinel is dropped — a real fix near the point stays."""
        self.assertFalse(_is_null_island(0.0001, -0.0001))


class TestResolvePlacesSkipsNullIsland(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "t.db")
        self.conn = connect(self.cfg.database)
        self.addCleanup(self.conn.close)

    def _add(self, name, lat, lon, taken="2023-05-01T10:00:00"):
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, taken_at,
                   taken_at_source, taken_at_confidence, gps_lat, gps_lon, indexed_at)
               VALUES (?, 1, 0, 'jpg', 'photo', ?, 'exif', 'high', ?, ?, 'now')""",
            (f"/p/{name}", taken, lat, lon))
        self.conn.commit()
        return cur.lastrowid

    def _place_of(self, file_id):
        return self.conn.execute(
            "SELECT confidence, country FROM places WHERE file_id = ?", (file_id,)
        ).fetchone()

    def test_zero_zero_photo_gets_no_place(self):
        fid = self._add("null.jpg", 0.0, 0.0)
        resolve_places(self.cfg, self.conn)
        row = self._place_of(fid)
        self.assertEqual(row["confidence"], "unknown")
        self.assertIsNone(row["country"])

    def test_neighbours_do_not_inherit_from_a_zero_zero_photo(self):
        """The session rule must not spread a country that never existed."""
        self._add("null.jpg", 0.0, 0.0, "2023-05-01T10:00:00")
        neighbour = self._add("next.jpg", None, None, "2023-05-01T10:30:00")
        resolve_places(self.cfg, self.conn)
        row = self._place_of(neighbour)
        self.assertEqual(row["confidence"], "unknown")
        self.assertIsNone(row["country"])


if __name__ == "__main__":
    unittest.main()
