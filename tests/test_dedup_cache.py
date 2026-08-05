"""F66: the Duplicates tab caches its payload until the index changes.

`near_duplicate_groups` is patched in `sorta.ui` so the tests count recomputations
instead of measuring time. The DB is a real sqlite file — the cache key is built
from its stat (and that of the `-wal` sidecar), so it must be on disk.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sorta import ui
from sorta.db import connect


class DupesCacheTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.db"
        self.conn = connect(self.db_path)
        self._n = 0
        ui._dupes_cache_clear()

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()
        ui._dupes_cache_clear()

    def add_dupe(self, rel: str, phash: str, size: int = 1000) -> int:
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, hash, hash_algo,
                   phash, width, height, indexed_at)
               VALUES (?, ?, 0, 'jpg', 'photo', ?, 'blake3', ?, 100, 100, '2026-01-01')""",
            (rel, size, f"hash-{self._n}", phash),
        )
        self.conn.commit()
        return int(cur.lastrowid)


class TestDupesPayloadCache(DupesCacheTestBase):
    def test_second_call_without_db_change_is_served_from_cache(self):
        self.add_dupe("/a.jpg", "0" * 16, size=2000)
        self.add_dupe("/b.jpg", "0" * 16, size=1000)
        with mock.patch.object(ui.review, "near_duplicate_groups",
                               wraps=ui.near_duplicate_groups) as spy:
            first = ui._dupes_payload(self.db_path, 5)
            second = ui._dupes_payload(self.db_path, 5)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(len(first["groups"][0]["frames"]), 2)
        self.assertEqual(first, second)

    def test_empty_result_is_cached_too(self):
        self.add_dupe("/lonely.jpg", "0" * 16)
        with mock.patch.object(ui.review, "near_duplicate_groups",
                               wraps=ui.near_duplicate_groups) as spy:
            self.assertEqual(ui._dupes_payload(self.db_path, 5)["groups"], [])
            self.assertEqual(ui._dupes_payload(self.db_path, 5)["groups"], [])
        self.assertEqual(spy.call_count, 1)

    def test_write_to_db_invalidates_the_cache(self):
        self.add_dupe("/a.jpg", "0" * 16, size=2000)
        self.add_dupe("/b.jpg", "0" * 16, size=1000)
        with mock.patch.object(ui.review, "near_duplicate_groups",
                               wraps=ui.near_duplicate_groups) as spy:
            before = ui._dupes_payload(self.db_path, 5)
            self.add_dupe("/c.jpg", "0" * 16, size=3000)
            after = ui._dupes_payload(self.db_path, 5)
        self.assertEqual(spy.call_count, 2)
        self.assertEqual(len(before["groups"][0]["frames"]), 2)
        self.assertEqual(len(after["groups"][0]["frames"]), 3)

    def test_different_max_distance_is_a_different_entry(self):
        self.add_dupe("/a.jpg", "0" * 16)
        self.add_dupe("/b.jpg", "0" * 14 + "ff")  # distance 8
        with mock.patch.object(ui.review, "near_duplicate_groups",
                               wraps=ui.near_duplicate_groups) as spy:
            self.assertEqual(ui._dupes_payload(self.db_path, 5)["groups"], [])
            self.assertEqual(len(ui._dupes_payload(self.db_path, 8)["groups"]), 1)
            self.assertEqual(ui._dupes_payload(self.db_path, 5)["groups"], [])
        self.assertEqual(spy.call_count, 2)  # the third call reuses the max_distance=5 entry

    def test_cache_keeps_at_most_two_entries(self):
        self.add_dupe("/a.jpg", "0" * 16)
        for distance in range(5):
            ui._dupes_payload(self.db_path, distance)
        self.assertLessEqual(len(ui._dupes_cache), ui._DUPES_CACHE_MAX_ITEMS)


class TestDbFingerprint(DupesCacheTestBase):
    def test_missing_files_do_not_raise(self):
        missing = Path(self.tmp.name) / "nope.db"
        self.assertEqual(ui._db_fingerprint(missing), ((-1, -1), (-1, -1)))

    def test_wal_sidecar_is_part_of_the_key(self):
        # In WAL mode a commit can leave the main .db untouched, so the sidecar has
        # to be in the key on its own. Use a path with no real DB behind it: the
        # sqlite handles of a live connection must not be poked at from a test.
        fake = Path(self.tmp.name) / "other.db"
        wal = Path(f"{fake}-wal")
        wal.write_bytes(b"x" * 10)
        fingerprint = ui._db_fingerprint(fake)
        self.assertEqual(fingerprint[0], (-1, -1))  # the .db itself does not exist
        self.assertEqual(fingerprint[1][1], 10)     # the sidecar's size is part of the key


if __name__ == "__main__":
    unittest.main()
