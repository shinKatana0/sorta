"""F67: _phash_one through the shared preview cache (drift + reuse + speedup).

The cache is redirected into tmp via SORTA_PREVIEW_DIR in every test.
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

from sorta import dedup, imaging
from sorta.config import Config, IndexConfig
from sorta.db import connect
from sorta.indexer import index
from tests.test_imaging_preview import make_photo, stat_key

# The acceptance benchmark (200 large JPEGs) costs minutes — it is not part of the
# gate; run it explicitly with SORTA_BENCH=1.
_BENCH = os.environ.get("SORTA_BENCH", "").strip() not in {"", "0"}


class PhashPreviewTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = self.root / "previews"
        self._patcher = unittest.mock.patch.dict(
            os.environ,
            {imaging.ENV_PREVIEW_DIR: str(self.cache), imaging.ENV_PREVIEW_CACHE: "1"},
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        imaging.cache_clear()
        self.tmp.cleanup()

    def previews(self) -> list[Path]:
        return sorted(self.cache.rglob("*.jpg")) if self.cache.exists() else []


class TestPhashThroughPreview(PhashPreviewTestCase):
    def test_drift_from_a_full_decode_is_within_two_bits(self):
        import imagehash

        for seed in (1, 2, 3, 4, 5):
            src = self.root / f"p{seed}.jpg"
            make_photo(src, size=(3200, 2400), seed=seed)
            full = imagehash.phash(imaging.decode_rgb(src, 96, grayscale=True))
            through = dedup._phash_one(str(src), *stat_key(src))
            self.assertIsNotNone(through)
            self.assertLessEqual(
                dedup.hamming(str(full), through), 2,
                f"pHash drift over 2 bits on seed={seed} (threshold max_distance=5)")

    def test_disabled_cache_gives_the_previous_hash(self):
        src = self.root / "a.jpg"
        make_photo(src, size=(3200, 2400))
        with unittest.mock.patch.dict(os.environ, {imaging.ENV_PREVIEW_CACHE: "0"}):
            direct = dedup._phash_one(str(src), *stat_key(src))
        self.assertIsNotNone(direct)
        self.assertEqual(self.previews(), [])

    def test_creates_the_preview_once_and_reuses_it(self):
        src = self.root / "a.jpg"
        make_photo(src, size=(3200, 2400))
        mtime, size = stat_key(src)
        first = dedup._phash_one(str(src), mtime, size)
        second = dedup._phash_one(str(src), mtime, size)
        self.assertEqual(first, second)
        self.assertEqual(len(self.previews()), 1)

    def test_undecodable_file_still_yields_none(self):
        src = self.root / "broken.jpg"
        src.write_bytes(b"\xff\xd8 not really a jpeg")
        self.assertIsNone(dedup._phash_one(str(src), *stat_key(src)))
        self.assertIsNone(dedup._phash_one(str(self.root / "missing.jpg"), 1.0, 10))

    def test_compute_phashes_passes_mtime_and_size_through(self):
        src_dir = self.root / "src"
        make_photo(src_dir / "a.jpg", size=(2400, 1600), seed=11)
        make_photo(src_dir / "b.jpg", size=(2400, 1600), seed=12)
        cfg = Config(
            sources=[src_dir], database=self.root / "test.db",
            index=IndexConfig(min_file_size_kb=0, compute_phash=False),
        )
        conn = connect(cfg.database)
        try:
            index(cfg, conn)
            seen: list[tuple[str, float, int]] = []
            real = dedup._phash_one

            def recording(path, mtime, size):
                seen.append((path, mtime, size))
                return real(path, mtime, size)

            with unittest.mock.patch.object(dedup, "_phash_one", recording):
                self.assertEqual(dedup.compute_phashes(cfg, conn), 2)

            rows = {r["path"]: (r["mtime"], r["size"])
                    for r in conn.execute("SELECT path, mtime, size FROM files")}
            self.assertEqual(len(seen), 2)
            for path, mtime, size in seen:
                self.assertEqual(rows[path], (mtime, size))  # straight from the DB row
            self.assertEqual(len(self.previews()), 2)
            self.assertTrue(all(r["phash"] for r in conn.execute("SELECT phash FROM files")))
        finally:
            conn.close()


class TestPreviewSpeedupBench(PhashPreviewTestCase):
    """F67 acceptance: a warm cache makes the pHash pass an order faster."""

    @unittest.skipUnless(_BENCH, "acceptance benchmark — run with SORTA_BENCH=1")
    def test_second_pass_is_at_least_ten_times_faster(self):
        files = []
        for i in range(200):
            src = self.root / "photos" / f"{i:03d}.jpg"
            make_photo(src, size=(4032, 3024), seed=i)
            files.append(src)
        keys = [(str(p), *stat_key(p)) for p in files]

        def phash_pass():
            start = time.perf_counter()
            for path, mtime, size in keys:
                self.assertIsNotNone(dedup._phash_one(path, mtime, size))
            return time.perf_counter() - start

        def baseline_pass():
            """The pre-F67 path: decode each original at 96px, no cache."""
            import imagehash
            start = time.perf_counter()
            for path, _mtime, _size in keys:
                imagehash.phash(imaging.decode_rgb(path, 96, grayscale=True))
            return time.perf_counter() - start

        baseline = baseline_pass()
        cold = phash_pass()
        warm = phash_pass()
        print(f"\nF67 bench (200 x 4032x3024 JPEG): baseline={baseline:.1f}s "
              f"cold={cold:.1f}s warm={warm:.2f}s "
              f"speedup={cold / warm:.1f}x first-pass-cost={cold / baseline:.2f}x")
        self.assertGreaterEqual(cold / warm, 10.0)


if __name__ == "__main__":
    unittest.main()
