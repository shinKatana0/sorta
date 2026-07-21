"""Indexing: incrementality, errors, dedup — on real temporary files."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sorta.config import Config, IndexConfig
from sorta.db import connect
from sorta.dedup import assign_duplicates, compute_phashes, hamming, near_duplicate_groups
from sorta.indexer import index, is_not_personal_video


def make_jpeg(path: Path, color=(255, 0, 0), size=(64, 64), orientation: int | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if orientation is not None:
        ex = Image.Exif()
        ex[274] = orientation
        kwargs["exif"] = ex
    Image.new("RGB", size, color).save(path, "JPEG", **kwargs)


class TestIndexer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "src"
        self.cfg = Config(
            sources=[self.src],
            database=self.root / "test.db",
            index=IndexConfig(min_file_size_kb=0, compute_phash=False),
        )
        self.conn = connect(self.cfg.database)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_index_and_incremental(self):
        make_jpeg(self.src / "IMG_20190705_123456.jpg")
        make_jpeg(self.src / "a" / "DSC01234.jpg", color=(0, 255, 0))
        (self.src / "notes.txt").write_text("skip me")

        s1 = index(self.cfg, self.conn)
        self.assertEqual((s1.added, s1.skipped, s1.errors), (2, 0, 0))

        row = self.conn.execute(
            "SELECT * FROM files WHERE path LIKE '%IMG_20190705%'").fetchone()
        self.assertEqual(row["taken_at_source"], "filename")
        self.assertEqual(row["taken_at"][:10], "2019-07-05")
        self.assertTrue(os.path.isabs(row["path"]))
        self.assertIsNotNone(row["hash"])

        # repeated run: everything skipped
        s2 = index(self.cfg, self.conn)
        self.assertEqual((s2.added, s2.updated, s2.skipped), (0, 0, 2))

        # a changed file is reindexed
        p = self.src / "a" / "DSC01234.jpg"
        make_jpeg(p, color=(0, 0, 255), size=(128, 128))
        os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 100))
        s3 = index(self.cfg, self.conn)
        self.assertEqual((s3.added, s3.updated), (0, 1))

    def test_broken_file_does_not_crash(self):
        make_jpeg(self.src / "good.jpg")
        (self.src / "broken.jpg").write_bytes(b"\xff\xd8 not really a jpeg")
        s = index(self.cfg, self.conn)
        # a corrupt jpeg is hashed and indexed without EXIF, or an error is written —
        # the point: the process does not crash and good.jpg is in the index
        self.assertEqual(s.added + s.errors, 2)
        self.assertIsNotNone(self.conn.execute(
            "SELECT id FROM files WHERE path LIKE '%good.jpg'").fetchone())

    def test_dedup_prefers_exif_then_largest(self):
        make_jpeg(self.src / "a.jpg")
        data = (self.src / "a.jpg").read_bytes()
        (self.src / "copy1.jpg").write_bytes(data)
        (self.src / "b" / "copy2.jpg").parent.mkdir(parents=True)
        (self.src / "b" / "copy2.jpg").write_bytes(data)
        index(self.cfg, self.conn)
        marked = assign_duplicates(self.conn)
        self.assertEqual(marked, 2)
        canon = self.conn.execute(
            "SELECT COUNT(*) FROM files WHERE dup_of IS NULL AND error IS NULL").fetchone()[0]
        self.assertEqual(canon, 1)


class TestIndexParallel(unittest.TestCase):
    """F11: parallel indexing (ThreadPoolExecutor) gives the same result."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_source(self, name: str, n_files: int) -> Path:
        src = self.root / name
        for i in range(n_files):
            make_jpeg(src / f"IMG_{i:03d}.jpg", color=(i % 256, 0, 0))
        (src / "broken.jpg").write_bytes(b"\xff\xd8 not really a jpeg")
        return src

    def _index_source(self, src: Path, name: str, workers: int):
        cfg = Config(
            sources=[src],
            database=self.root / f"{name}.db",
            index=IndexConfig(min_file_size_kb=0, compute_phash=False),
            raw={"index": {"workers": workers}},
        )
        conn = connect(cfg.database)
        try:
            stats = index(cfg, conn)
            rows = {
                Path(r["path"]).name: (r["hash"], r["size"], r["taken_at"], r["media_type"])
                for r in conn.execute("SELECT * FROM files ORDER BY path")
            }
        finally:
            conn.close()
        return stats, rows

    def test_parallel_matches_serial(self):
        # The same directory is indexed both ways: the files' mtimes are identical,
        # so taken_at (derived from mtime without EXIF) is deterministic. Two
        # independently created file sets gave a one-second divergence at the second
        # boundary (a flaky test, backlog #12).
        src = self._make_source("shared", 8)
        serial_stats, serial_rows = self._index_source(src, "serial", 1)
        parallel_stats, parallel_rows = self._index_source(src, "parallel", 4)
        self.assertEqual((serial_stats.added, serial_stats.errors),
                          (parallel_stats.added, parallel_stats.errors))
        self.assertEqual(serial_rows, parallel_rows)

    def test_default_workers_used_when_unset(self):
        # cfg.raw without an index section -> default resolve_workers, indexing does not crash
        src = self._make_source("defaults", 3)
        cfg = Config(
            sources=[src], database=self.root / "defaults.db",
            index=IndexConfig(min_file_size_kb=0, compute_phash=False),
        )
        conn = connect(cfg.database)
        try:
            stats = index(cfg, conn)
            self.assertEqual(stats.added, 4)  # 3 good + broken.jpg (hashed without error)
            self.assertEqual(stats.errors, 0)
        finally:
            conn.close()

    def test_index_never_writes_phash(self):
        # even with compute_phash=True in config (a deprecated flag) index() writes no phash — moved out
        src = self.root / "src"
        make_jpeg(src / "a.jpg")
        cfg = Config(
            sources=[src], database=self.root / "test.db",
            index=IndexConfig(min_file_size_kb=0, compute_phash=True),
        )
        conn = connect(cfg.database)
        try:
            index(cfg, conn)
            row = conn.execute("SELECT phash FROM files").fetchone()
            self.assertIsNone(row["phash"])
        finally:
            conn.close()

    def test_reindex_does_not_clobber_existing_phash(self):
        src = self.root / "src2"
        p = src / "a.jpg"
        make_jpeg(p)
        cfg = Config(
            sources=[src], database=self.root / "test2.db",
            index=IndexConfig(min_file_size_kb=0, compute_phash=False),
        )
        conn = connect(cfg.database)
        try:
            index(cfg, conn)
            conn.execute("UPDATE files SET phash = 'deadbeef'")
            conn.commit()
            # change mtime so the file is reindexed (action='update')
            os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 100))
            index(cfg, conn)
            row = conn.execute("SELECT phash FROM files").fetchone()
            self.assertEqual(row["phash"], "deadbeef")
        finally:
            conn.close()

    def test_reindex_invalidates_phash_when_content_changes(self):
        # content change (different hash) → phash is reset to NULL so
        # compute_phashes recomputes it (otherwise a stale one would remain)
        src = self.root / "src3"
        p = src / "a.jpg"
        make_jpeg(p, color=(255, 0, 0))
        cfg = Config(
            sources=[src], database=self.root / "test3.db",
            index=IndexConfig(min_file_size_kb=0, compute_phash=False),
        )
        conn = connect(cfg.database)
        try:
            index(cfg, conn)
            conn.execute("UPDATE files SET phash = 'deadbeef'")
            conn.commit()
            make_jpeg(p, color=(0, 0, 255), size=(128, 128))  # different content → different hash
            os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 100))
            index(cfg, conn)
            row = conn.execute("SELECT phash FROM files").fetchone()
            self.assertIsNone(row["phash"])
        finally:
            conn.close()


class TestComputePhashes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "src"
        self.cfg = Config(
            sources=[self.src],
            database=self.root / "test.db",
            index=IndexConfig(min_file_size_kb=0, compute_phash=False),
        )
        self.conn = connect(self.cfg.database)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_computes_incrementally(self):
        make_jpeg(self.src / "a.jpg", color=(255, 0, 0))
        make_jpeg(self.src / "b.jpg", color=(0, 255, 0))
        (self.src / "broken.jpg").write_bytes(b"\xff\xd8 not really a jpeg")
        index(self.cfg, self.conn)

        n = compute_phashes(self.cfg, self.conn)
        self.assertEqual(n, 2)  # broken.jpg does not decode -> phash stays NULL
        rows = {Path(r["path"]).name: r["phash"]
                for r in self.conn.execute("SELECT path, phash FROM files")}
        self.assertIsNotNone(rows["a.jpg"])
        self.assertIsNotNone(rows["b.jpg"])
        self.assertIsNone(rows["broken.jpg"])

        # the second run does not recompute already-computed ones (finds no candidates)
        n2 = compute_phashes(self.cfg, self.conn)
        self.assertEqual(n2, 0)

    def test_progress_reports_total(self):
        make_jpeg(self.src / "a.jpg")
        make_jpeg(self.src / "b.jpg", color=(0, 0, 255))
        index(self.cfg, self.conn)
        calls = []
        compute_phashes(self.cfg, self.conn, progress=lambda done, total: calls.append((done, total)))
        self.assertTrue(calls)
        self.assertEqual(calls[-1], (2, 2))

    def test_no_photos_returns_zero(self):
        self.assertEqual(compute_phashes(self.cfg, self.conn), 0)


class TestExifParseRecords(unittest.TestCase):
    def test_parse_records(self):
        from sorta.exif import _parse_records
        recs = [{
            "SourceFile": "x.jpg", "DateTimeOriginal": "2019:07:05 12:34:56",
            "GPSLatitude": 55.75, "GPSLongitude": 37.61,
            "Make": "Canon", "Model": "EOS", "ImageWidth": 640, "ImageHeight": 480,
            "Orientation": 6,
        }]
        (data,) = _parse_records(recs).values()
        self.assertEqual(data.datetime_original, "2019:07:05 12:34:56")
        self.assertEqual((data.gps_lat, data.gps_lon), (55.75, 37.61))
        self.assertEqual((data.width, data.height, data.orientation), (640, 480, 6))

    def test_parse_records_missing_fields(self):
        from sorta.exif import _parse_records
        (data,) = _parse_records([{"SourceFile": "y.jpg"}]).values()
        self.assertIsNone(data.datetime_original)
        self.assertIsNone(data.orientation)

    def test_parse_records_blank_gps_becomes_none(self):
        # a real case: exiftool returned an empty string for the GPS tag
        from sorta.exif import _parse_records
        recs = [{"SourceFile": "z.jpg", "GPSLatitude": "", "GPSLongitude": ""}]
        (data,) = _parse_records(recs).values()
        self.assertIsNone(data.gps_lat)
        self.assertIsNone(data.gps_lon)


class TestExifToolStayOpen(unittest.TestCase):
    """Integration with the real exiftool; skipped if it is not installed."""

    @classmethod
    def setUpClass(cls):
        from sorta.exif import exiftool_available
        if not exiftool_available():
            raise unittest.SkipTest("exiftool not installed")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_session_reuses_process_and_reads_exif(self):
        from sorta.exif import ExifToolSession
        make_jpeg(self.root / "a.jpg", orientation=6)
        make_jpeg(self.root / "b.jpg", color=(0, 255, 0))
        session = ExifToolSession()
        try:
            out1 = session.read([self.root / "a.jpg"])
            pid = session._proc.pid
            out2 = session.read([self.root / "b.jpg"])
            self.assertEqual(session._proc.pid, pid)  # one process for both requests
            a = out1[str((self.root / "a.jpg").resolve())]
            self.assertEqual((a.width, a.height, a.orientation), (64, 64, 6))
            b = out2[str((self.root / "b.jpg").resolve())]
            self.assertEqual((b.width, b.height), (64, 64))
        finally:
            session.close()
        self.assertIsNone(session._proc)


# A fake exiftool: implements exactly the -stay_open protocol (argfile on stdin,
# response up to {ready}), to test ExifToolSession without a real exiftool.
_FAKE_EXIFTOOL = r'''
import json, sys
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
args = []
for line in sys.stdin:
    line = line.rstrip("\r\n")
    if line == "-execute":
        from pathlib import Path
        files = [a for a in args if Path(a).is_absolute()]
        recs = [{"SourceFile": f, "ImageWidth": 64, "ImageHeight": 64,
                 "Orientation": 6} for f in files]
        if recs:
            sys.stdout.write(json.dumps(recs) + "\n")
        sys.stdout.write("{ready}\n")
        sys.stdout.flush()
        args = []
    elif line == "-stay_open":
        break  # следующая строка False — завершаемся, как настоящий exiftool
    else:
        args.append(line)
'''


class TestExifToolSessionProtocol(unittest.TestCase):
    """The -stay_open protocol against a fake — works even without exiftool installed."""

    def setUp(self):
        import sorta.exif as exif_mod
        self.exif = exif_mod
        self.tmp = tempfile.TemporaryDirectory()
        fake = Path(self.tmp.name) / "fake_exiftool.py"
        fake.write_text(_FAKE_EXIFTOOL, encoding="utf-8")
        self._orig_cmd = exif_mod._EXIFTOOL_CMD
        exif_mod._EXIFTOOL_CMD = [sys.executable, str(fake)]

    def tearDown(self):
        self.exif._EXIFTOOL_CMD = self._orig_cmd
        self.tmp.cleanup()

    def test_read_reuses_process_and_parses(self):
        session = self.exif.ExifToolSession()
        try:
            a = Path(self.tmp.name) / "a.jpg"
            out = session.read([a])
            data = out[str(a.resolve())]
            self.assertEqual((data.width, data.height, data.orientation), (64, 64, 6))
            pid = session._proc.pid
            session.read([Path(self.tmp.name) / "b.jpg"])
            self.assertEqual(session._proc.pid, pid)  # one process for both requests
        finally:
            session.close()
        self.assertIsNone(session._proc)

    def test_recovers_after_process_death(self):
        session = self.exif.ExifToolSession()
        try:
            a = Path(self.tmp.name) / "a.jpg"
            session.read([a])
            session._proc.kill()
            session._proc.wait()
            self.assertEqual(len(session.read([a])), 1)  # transparent restart
        finally:
            session.close()

    def test_empty_paths_no_process(self):
        session = self.exif.ExifToolSession()
        self.assertEqual(session.read([]), {})
        self.assertIsNone(session._proc)
        session.close()


class TestNearDuplicates(unittest.TestCase):
    """near_duplicate_groups on synthetic files rows (without real images)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = connect(Path(self.tmp.name) / "test.db")

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _add(self, path, phash, size=100, dup_of=None, error=None):
        self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, phash,
                                  dup_of, error, indexed_at)
               VALUES (?,?,0,'jpg','photo',?,?,?,'2026-01-01')""",
            (path, size, phash, dup_of, error))
        self.conn.commit()
        return self.conn.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()["id"]

    def test_hamming(self):
        self.assertEqual(hamming("0" * 16, "0" * 16), 0)
        self.assertEqual(hamming("0" * 16, "0" * 15 + "1"), 1)  # 1 bit
        self.assertEqual(hamming("0" * 16, "0" * 15 + "f"), 4)  # 4 bits
        self.assertEqual(hamming("0" * 16, "f" * 16), 64)

    def test_groups_within_threshold(self):
        self._add("/a.jpg", "0" * 16, size=200)
        self._add("/b.jpg", "0" * 15 + "3")   # dist 2 from /a.jpg
        self._add("/far.jpg", "f" * 16)       # far from all
        groups = near_duplicate_groups(self.conn, max_distance=5)
        self.assertEqual(len(groups), 1)
        paths = [r["path"] for r in groups[0]]
        self.assertEqual(paths, ["/a.jpg", "/b.jpg"])  # larger size first

    def test_threshold_respected(self):
        self._add("/a.jpg", "0" * 16)
        self._add("/b.jpg", "0" * 14 + "ff")  # dist 8
        self.assertEqual(near_duplicate_groups(self.conn, max_distance=5), [])
        self.assertEqual(len(near_duplicate_groups(self.conn, max_distance=8)), 1)

    def test_excludes_exact_dups_and_errors(self):
        a = self._add("/a.jpg", "0" * 16)
        self._add("/exact_copy.jpg", "0" * 16, dup_of=a)     # exact duplicate
        self._add("/broken.jpg", "0" * 16, error="Boom")     # error
        self._add("/no_phash.jpg", None)
        self.assertEqual(near_duplicate_groups(self.conn, max_distance=5), [])

    def test_transitive_chain_in_one_group(self):
        # dist(a,b)=2, dist(a,c)=2, dist(b,c)=4 — b and c land in one group via a
        self._add("/a.jpg", "0" * 16)
        self._add("/b.jpg", "0" * 15 + "3")
        self._add("/c.jpg", "0" * 15 + "c")
        groups = near_duplicate_groups(self.conn, max_distance=3)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)


class TestOrientation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "src"
        self.cfg = Config(
            sources=[self.src],
            database=self.root / "test.db",
            index=IndexConfig(min_file_size_kb=0, compute_phash=False),
        )
        self.conn = connect(self.cfg.database)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_pillow_reads_orientation(self):
        from sorta.exif import read_one_pillow
        make_jpeg(self.src / "rot.jpg", orientation=6)
        self.assertEqual(read_one_pillow(self.src / "rot.jpg").orientation, 6)
        make_jpeg(self.src / "plain.jpg")
        self.assertIsNone(read_one_pillow(self.src / "plain.jpg").orientation)

    def test_index_without_column_does_not_crash(self):
        # emulate a pre-v2-migration DB (without files.orientation) — indexing does not crash
        self.conn.execute("ALTER TABLE files DROP COLUMN orientation")
        make_jpeg(self.src / "rot.jpg", orientation=6)
        s = index(self.cfg, self.conn)
        self.assertEqual((s.added, s.errors), (1, 0))
        row = self.conn.execute("SELECT * FROM files").fetchone()
        self.assertNotIn("orientation", row.keys())

    def test_index_writes_orientation_when_column_exists(self):
        # schema v2: the column exists out of the box
        make_jpeg(self.src / "rot.jpg", orientation=6)
        make_jpeg(self.src / "plain.jpg", color=(0, 255, 0))
        index(self.cfg, self.conn)
        rows = {Path(r["path"]).name: r["orientation"]
                for r in self.conn.execute("SELECT path, orientation FROM files")}
        self.assertEqual(rows["rot.jpg"], 6)
        self.assertIsNone(rows["plain.jpg"])


class TestNotPersonalVideoHeuristic(unittest.TestCase):
    """F17: is_not_personal_video as a pure function over a set of names."""

    def test_release_names_match(self):
        release_names = [
            "Show.Name.S01E05.mp4",
            "Show.Name.1x05.mp4",
            "Movie.Name.2021.1080p.WEB-DL.x264-GROUP.mkv",
            "Movie.Name.2020.2160p.BluRay.x265.mkv",
            "Some.Movie.2019.720p.HDTV.mkv",
            "Another.One.2018.DVDRip.XviD.avi",
            "Release.Name.2022.4K.BDRip.HEVC.mkv",
            "[GROUP] Anime Series - 05.mkv",
            "Movie.Name.2021.WEBRip.mp4",
        ]
        for name in release_names:
            with self.subTest(name=name):
                self.assertTrue(is_not_personal_video(name, 0), name)

    def test_personal_names_do_not_match(self):
        personal_names = [
            "VID_20230101_120000.mp4",
            "PXL_20230615_143022000.mp4",
            "MOV_20220101_100000.mp4",
            "IMG_1234.mov",
            "20230615_143022.mp4",
            "WhatsApp Video 2023-06-15 at 14.30.22.mp4",
            "Отпуск в Сочи.mp4",
        ]
        for name in personal_names:
            with self.subTest(name=name):
                self.assertFalse(is_not_personal_video(name, 0), name)

    def test_large_size_alone_is_not_a_signal(self):
        # a very large file without a release pattern in the name is not marked (a 4K family
        # video is also large), size is not used as a standalone signal
        self.assertFalse(is_not_personal_video("VID_20230101_120000.mp4", 20 * 1024**3))


class TestNotPersonalVideoIndexing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "src"
        self.cfg = Config(
            sources=[self.src],
            database=self.root / "test.db",
            index=IndexConfig(min_file_size_kb=0, compute_phash=False),
        )
        self.conn = connect(self.cfg.database)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _make_video(self, name: str, content: bytes = b"fake video bytes"):
        path = self.src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_release_video_marked_not_personal(self):
        self._make_video("Movie.Name.2021.1080p.WEB-DL.x264-GROUP.mkv")
        index(self.cfg, self.conn)
        row = self.conn.execute("SELECT media_type, not_personal FROM files").fetchone()
        self.assertEqual(row["media_type"], "video")
        self.assertEqual(row["not_personal"], 1)

    def test_personal_video_not_marked(self):
        self._make_video("VID_20230101_120000.mp4")
        index(self.cfg, self.conn)
        row = self.conn.execute("SELECT media_type, not_personal FROM files").fetchone()
        self.assertEqual(row["media_type"], "video")
        self.assertEqual(row["not_personal"], 0)

    def test_photo_is_never_marked_even_if_name_looks_like_release(self):
        make_jpeg(self.src / "Movie.Name.2021.1080p.WEB-DL.x264-GROUP.jpg")
        index(self.cfg, self.conn)
        row = self.conn.execute("SELECT media_type, not_personal FROM files").fetchone()
        self.assertEqual(row["media_type"], "photo")
        self.assertEqual(row["not_personal"], 0)

    def test_index_without_not_personal_column_does_not_crash(self):
        self.conn.execute("ALTER TABLE files DROP COLUMN not_personal")
        self._make_video("Movie.Name.2021.1080p.WEB-DL.x264-GROUP.mkv")
        s = index(self.cfg, self.conn)
        self.assertEqual((s.added, s.errors), (1, 0))
        row = self.conn.execute("SELECT * FROM files").fetchone()
        self.assertNotIn("not_personal", row.keys())

    def test_incremental_reindex_keeps_not_personal(self):
        p = self._make_video("Movie.Name.2021.1080p.WEB-DL.x264-GROUP.mkv")
        index(self.cfg, self.conn)
        os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 100))
        p.write_bytes(b"fake video bytes, changed")
        index(self.cfg, self.conn)
        row = self.conn.execute("SELECT not_personal FROM files").fetchone()
        self.assertEqual(row["not_personal"], 1)


if __name__ == "__main__":
    unittest.main()
