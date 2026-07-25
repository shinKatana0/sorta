"""F71: refresh_exif — recovering metadata that -fast2 lost, without reindexing.

Every test follows the real scenario: files are indexed while exiftool reports nothing
(what -fast2 did), then the fake starts reporting the metadata that was on disk all
along and refresh_exif picks it up.
"""
import tempfile
import unittest
from pathlib import Path

from sorta.config import Config, DatesConfig, IndexConfig
from sorta.db import connect
from sorta.indexer import index, refresh_exif
from tests.test_exif_flags import FakeExifTool
from tests.test_indexer import make_jpeg

_PHONE = {
    "Make": "samsung", "Model": "Galaxy S23 Ultra",
    "GPSLatitude": 36.5739, "GPSLongitude": 127.0092,
    "DateTimeOriginal": "2023:09:19 19:49:53",
    "ImageWidth": 4032, "ImageHeight": 3024, "Orientation": 6,
}

# columns that depend on the file CONTENT — refresh_exif must not touch any of them
_CONTENT_COLUMNS = ("hash", "hash_algo", "phash", "dup_of", "size", "mtime",
                    "not_personal", "indexed_at", "ext", "media_type", "path")


class RefreshExifTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "src"
        self.cfg = Config(
            sources=[self.src], database=self.root / "test.db",
            index=IndexConfig(min_file_size_kb=0, compute_phash=False),
        )
        self.fake = FakeExifTool(self.root)
        self.conn = connect(self.cfg.database)

    def tearDown(self):
        self.conn.close()
        self.fake.restore()
        self.tmp.cleanup()

    def _row(self, name: str):
        return self.conn.execute(
            "SELECT * FROM files WHERE path LIKE ?", (f"%{name}",)).fetchone()

    def test_recovers_metadata_and_capture_date(self):
        make_jpeg(self.src / "DSC0001.jpg")
        index(self.cfg, self.conn)
        before = self._row("DSC0001.jpg")
        self.assertIsNone(before["camera_make"])          # what -fast2 left behind
        self.assertEqual(before["taken_at_source"], "mtime")

        self.fake.set_meta({"DSC0001.jpg": _PHONE})
        stats = refresh_exif(self.cfg, self.conn)
        self.assertEqual(
            (stats.scanned, stats.updated, stats.recovered_gps,
             stats.recovered_date, stats.still_empty, stats.errors),
            (1, 1, 1, 1, 0, 0))

        row = self._row("DSC0001.jpg")
        self.assertEqual((row["camera_make"], row["camera_model"]),
                         ("samsung", "Galaxy S23 Ultra"))
        self.assertEqual((row["gps_lat"], row["gps_lon"]), (36.5739, 127.0092))
        self.assertEqual((row["width"], row["height"], row["orientation"]),
                         (4032, 3024, 6))
        self.assertEqual(row["taken_at"], "2023-09-19T19:49:53")
        self.assertEqual((row["taken_at_source"], row["taken_at_confidence"]),
                         ("exif", "high"))

    def test_does_not_touch_content_derived_columns(self):
        # the main invariant: the feature must not break dedup
        make_jpeg(self.src / "a.jpg")
        make_jpeg(self.src / "b.jpg", color=(0, 255, 0))
        index(self.cfg, self.conn)
        canonical = self._row("a.jpg")["id"]
        self.conn.execute("UPDATE files SET phash = 'deadbeef'")
        self.conn.execute("UPDATE files SET dup_of = ? WHERE path LIKE '%b.jpg'",
                          (canonical,))
        self.conn.commit()
        before = {r["path"]: {c: r[c] for c in _CONTENT_COLUMNS}
                  for r in self.conn.execute("SELECT * FROM files")}

        self.fake.set_meta({"a.jpg": _PHONE, "b.jpg": _PHONE})
        stats = refresh_exif(self.cfg, self.conn)
        self.assertEqual(stats.updated, 2)

        after = {r["path"]: {c: r[c] for c in _CONTENT_COLUMNS}
                 for r in self.conn.execute("SELECT * FROM files")}
        self.assertEqual(before, after)

    def test_only_missing_skips_rows_that_already_have_exif(self):
        make_jpeg(self.src / "a.jpg")
        self.fake.set_meta({"a.jpg": _PHONE})
        index(self.cfg, self.conn)
        self.assertEqual(self._row("a.jpg")["camera_make"], "samsung")

        self.fake.set_meta({"a.jpg": {**_PHONE, "Make": "Canon"}})
        skipped = refresh_exif(self.cfg, self.conn)
        self.assertEqual((skipped.scanned, skipped.updated), (0, 0))
        self.assertEqual(self._row("a.jpg")["camera_make"], "samsung")

        forced = refresh_exif(self.cfg, self.conn, only_missing=False)
        self.assertEqual((forced.scanned, forced.updated), (1, 1))
        self.assertEqual(self._row("a.jpg")["camera_make"], "Canon")

    def test_file_without_exif_stays_empty_and_is_retried(self):
        make_jpeg(self.src / "plain.png")
        index(self.cfg, self.conn)
        stats = refresh_exif(self.cfg, self.conn)
        # nothing was read -> nothing is written (not even a pointless taken_at rewrite)
        self.assertEqual((stats.scanned, stats.updated, stats.still_empty, stats.errors),
                         (1, 0, 1, 0))
        row = self._row("plain.png")
        self.assertIsNone(row["camera_make"])
        self.assertIsNone(row["width"])
        # the selection deliberately does not remember "already tried"
        again = refresh_exif(self.cfg, self.conn)
        self.assertEqual((again.scanned, again.still_empty), (1, 1))

    def test_rows_with_errors_are_not_selected(self):
        make_jpeg(self.src / "a.jpg")
        index(self.cfg, self.conn)
        self.conn.execute("UPDATE files SET error = 'boom'")
        self.conn.commit()
        self.fake.set_meta({"a.jpg": _PHONE})
        self.assertEqual(refresh_exif(self.cfg, self.conn).scanned, 0)

    def test_out_of_range_date_matches_the_index_path(self):
        # the two paths must not disagree on dates: same file, same cfg.dates bounds
        make_jpeg(self.src / "DSC0002.jpg")
        self.cfg.dates = DatesConfig(min_year=2000, max_year=2035)
        meta = {"DSC0002.jpg": {**_PHONE, "DateTimeOriginal": "1970:01:01 00:00:00"}}

        self.fake.set_meta(meta)
        direct = connect(self.root / "direct.db")
        try:
            index(self.cfg, direct)
            expected = direct.execute(
                "SELECT taken_at, taken_at_source, taken_at_confidence FROM files"
            ).fetchone()
        finally:
            direct.close()

        self.fake.set_meta({})
        index(self.cfg, self.conn)
        self.fake.set_meta(meta)
        stats = refresh_exif(self.cfg, self.conn)
        row = self._row("DSC0002.jpg")
        self.assertEqual(
            (row["taken_at"], row["taken_at_source"], row["taken_at_confidence"]),
            tuple(expected))
        self.assertEqual(row["taken_at_source"], "mtime")  # the 1970 date is rejected
        self.assertEqual(stats.recovered_date, 0)          # ...so no date was recovered
        self.assertEqual(stats.recovered_gps, 1)           # the coordinates still are

    def test_missing_file_does_not_break_the_batch(self):
        for name in ("a1.jpg", "a2.jpg", "a3.jpg"):
            make_jpeg(self.src / name)
        index(self.cfg, self.conn)
        (self.src / "a2.jpg").unlink()

        self.fake.set_meta({"a1.jpg": _PHONE, "a3.jpg": _PHONE})
        stats = refresh_exif(self.cfg, self.conn)
        self.assertEqual((stats.scanned, stats.updated, stats.errors), (3, 2, 1))
        self.assertEqual(stats.recovered_gps, 2)
        self.assertIsNone(self._row("a2.jpg")["camera_make"])

    def test_progress_reports_total(self):
        make_jpeg(self.src / "a.jpg")
        make_jpeg(self.src / "b.jpg", color=(0, 0, 255))
        index(self.cfg, self.conn)
        calls: list[tuple[int, int]] = []
        refresh_exif(self.cfg, self.conn, progress=lambda done, total: calls.append(
            (done, total)))
        self.assertTrue(calls)
        self.assertEqual(calls[-1], (2, 2))

    def test_works_without_orientation_column(self):
        # a pre-v2 DB (no files.orientation) — refresh must not crash on it
        self.conn.execute("ALTER TABLE files DROP COLUMN orientation")
        make_jpeg(self.src / "a.jpg")
        index(self.cfg, self.conn)
        self.fake.set_meta({"a.jpg": _PHONE})
        stats = refresh_exif(self.cfg, self.conn)
        self.assertEqual((stats.updated, stats.errors), (1, 0))
        row = self._row("a.jpg")
        self.assertNotIn("orientation", row.keys())
        self.assertEqual(row["camera_make"], "samsung")


if __name__ == "__main__":
    unittest.main()
