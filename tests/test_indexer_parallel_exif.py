"""F72: the indexer overlaps exiftool with hashing and honours index.exif_workers.

`pool.map` queues the hashing work and returns a lazy iterator, so the exiftool pool
runs alongside blake3 instead of after it. What must NOT change is the result: the same
rows in `files` for the same input, still written by the main thread only.
"""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import sorta.indexer as indexer_mod
from sorta.config import Config, IndexConfig
from sorta.db import connect
from sorta.hashing import file_hash
from sorta.indexer import index, refresh_exif
from tests.test_exif_flags import FakeExifTool
from tests.test_exif_parallel import meta_for
from tests.test_indexer import make_jpeg

_ROWS = 12
_EXIF_WORKERS = 3


class IndexerExifTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "src"
        self.names = [f"img_{i:04d}.jpg" for i in range(_ROWS)]
        for i, name in enumerate(self.names):
            make_jpeg(self.src / name, color=(i, 0, 0))
        self.fake = FakeExifTool(self.root, meta_for(self.names))
        self.cfg = Config(
            sources=[self.src],
            database=self.root / "test.db",
            index=IndexConfig(min_file_size_kb=0, compute_phash=False),
            raw={"index": {"workers": 4, "exif_workers": _EXIF_WORKERS}},
        )
        self.conn = connect(self.cfg.database)

    def tearDown(self):
        self.conn.close()
        self.fake.restore()
        self.tmp.cleanup()


class TestRowsUnchanged(IndexerExifTestCase):
    def test_every_row_matches_the_file_and_its_exif(self):
        stats = index(self.cfg, self.conn)
        self.assertEqual((stats.added, stats.errors), (_ROWS, 0))
        rows = self.conn.execute("SELECT * FROM files ORDER BY path").fetchall()
        self.assertEqual(len(rows), _ROWS)
        for row in rows:
            path = Path(row["path"])
            with self.subTest(name=path.name):
                self.assertEqual(row["hash"], file_hash(path)[0])
                self.assertEqual(row["camera_make"], path.name)  # answers not swapped
                self.assertEqual(row["camera_model"], "cam")
                self.assertEqual((row["width"], row["height"]), (64, 48))
                self.assertEqual((row["gps_lat"], row["gps_lon"]), (55.75, 37.62))
                self.assertEqual(row["taken_at"], "2024-01-02T03:04:05")
                self.assertEqual(row["taken_at_source"], "exif")
                self.assertEqual(row["orientation"], 6)
                self.assertIsNone(row["error"])

    def test_incrementality_still_skips_everything_on_a_rerun(self):
        index(self.cfg, self.conn)
        again = index(self.cfg, self.conn)
        self.assertEqual((again.added, again.updated, again.skipped), (0, 0, _ROWS))


class TestWorkersArePropagated(IndexerExifTestCase):
    """exif.py stays free of Config — the indexer passes the configured number in."""

    def spy(self):
        real = indexer_mod.read_batch
        seen: list[int | None] = []

        def wrapper(paths, workers=None):
            seen.append(workers)
            return real(paths, workers)

        return wrapper, seen

    def test_index_passes_exif_workers(self):
        wrapper, seen = self.spy()
        with mock.patch.object(indexer_mod, "read_batch", wrapper):
            index(self.cfg, self.conn)
        self.assertEqual(seen, [_EXIF_WORKERS])

    def test_refresh_exif_passes_exif_workers(self):
        index(self.cfg, self.conn)
        wrapper, seen = self.spy()
        with mock.patch.object(indexer_mod, "read_batch", wrapper):
            stats = refresh_exif(self.cfg, self.conn, only_missing=False)
        self.assertEqual(seen, [_EXIF_WORKERS])
        self.assertEqual(stats.scanned, _ROWS)


class TestPhasesOverlap(IndexerExifTestCase):
    """Hashing must already be running while exiftool is being asked."""

    def test_hashing_starts_before_exiftool_returns(self):
        started = threading.Event()
        real_hash_one = indexer_mod._hash_one
        real_read_batch = indexer_mod.read_batch
        overlapped: list[bool] = []

        def slow_hash_one(item):
            started.set()
            return real_hash_one(item)

        def watching_read_batch(paths, workers=None):
            # if the phases were serial (exif first), no hash task could have started
            overlapped.append(started.wait(timeout=30))
            return real_read_batch(paths, workers)

        with mock.patch.object(indexer_mod, "_hash_one", slow_hash_one), \
                mock.patch.object(indexer_mod, "read_batch", watching_read_batch):
            stats = index(self.cfg, self.conn)

        self.assertEqual(stats.added, _ROWS)
        self.assertEqual(overlapped, [True])


if __name__ == "__main__":
    unittest.main()
