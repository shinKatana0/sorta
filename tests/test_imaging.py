"""F18: decode_rgb/decode_rgb_cached — a pure decode layer, without ML/GPU."""
from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from sorta import imaging


def make_jpeg(path: Path, color=(255, 0, 0), size=(64, 64), orientation: int | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if orientation is not None:
        ex = Image.Exif()
        ex[274] = orientation
        kwargs["exif"] = ex
    Image.new("RGB", size, color).save(path, "JPEG", **kwargs)


class TestDecodeRgb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        imaging.cache_clear()

    def tearDown(self):
        imaging.cache_clear()
        self.tmp.cleanup()

    def test_basic_jpeg_decode(self):
        path = self.root / "a.jpg"
        make_jpeg(path, size=(64, 48))
        img = imaging.decode_rgb(path)
        self.assertIsNotNone(img)
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.size, (64, 48))

    def test_max_edge_downscales(self):
        path = self.root / "big.jpg"
        make_jpeg(path, size=(400, 200))
        img = imaging.decode_rgb(path, max_edge=100)
        self.assertIsNotNone(img)
        self.assertLessEqual(max(img.size), 100)

    def test_no_max_edge_keeps_full_size(self):
        path = self.root / "full.jpg"
        make_jpeg(path, size=(300, 150))
        img = imaging.decode_rgb(path)
        self.assertEqual(img.size, (300, 150))

    def test_grayscale_mode(self):
        path = self.root / "gray.jpg"
        make_jpeg(path, size=(64, 64))
        img = imaging.decode_rgb(path, max_edge=32, grayscale=True)
        self.assertIsNotNone(img)
        self.assertEqual(img.mode, "L")

    def test_orientation_applied(self):
        # orientation=6 -> a 90° rotation: physical JPEG 64x48, after exif_transpose
        # the visible dimensions swap -> 48x64.
        path = self.root / "rot.jpg"
        make_jpeg(path, size=(64, 48), orientation=6)
        rotated = imaging.decode_rgb(path, apply_orientation=True)
        not_rotated = imaging.decode_rgb(path, apply_orientation=False)
        self.assertEqual(rotated.size, (48, 64))
        self.assertEqual(not_rotated.size, (64, 48))

    def test_heic_branch(self):
        try:
            import pillow_heif
        except ImportError:
            self.skipTest("pillow_heif not installed")
        path = self.root / "photo.heic"
        im = Image.new("RGB", (48, 32), (10, 20, 30))
        heif = pillow_heif.from_pillow(im)
        try:
            heif.save(path, quality=80)
        except Exception as exc:
            self.skipTest(f"HEIC encoding unavailable in this environment: {exc}")
        img = imaging.decode_rgb(path)
        self.assertIsNotNone(img)
        self.assertEqual(img.mode, "RGB")

    def test_corrupt_file_returns_none(self):
        path = self.root / "corrupt.jpg"
        path.write_bytes(b"\xff\xd8 not jpeg")
        self.assertIsNone(imaging.decode_rgb(path))

    def test_missing_file_returns_none(self):
        self.assertIsNone(imaging.decode_rgb(self.root / "missing.jpg"))

    def test_aggressive_draft_margin_gives_same_final_result(self):
        # F48: margin=2× (default) at this max_edge/source ratio does not pass the
        # first halving threshold of draft() (2016/2=1008 < 1280) -> full-frame
        # decode; margin=1.0 (opt-in, the OCR path) passes (1008 >= 640) -> a cheaper
        # decode. The final RESULT (size, color) must match — the source's solid color
        # guarantees the block DCT downscale of draft() introduces no artifacts the
        # test would notice.
        path = self.root / "big.jpg"
        make_jpeg(path, color=(10, 20, 30), size=(2016, 1512))
        default_img = imaging.decode_rgb(path, max_edge=640)
        aggressive_img = imaging.decode_rgb(path, max_edge=640, draft_margin=1.0)
        self.assertIsNotNone(default_img)
        self.assertIsNotNone(aggressive_img)
        self.assertEqual(default_img.size, aggressive_img.size)
        self.assertLessEqual(max(aggressive_img.size), 640)
        self.assertEqual(aggressive_img.getpixel((0, 0)), default_img.getpixel((0, 0)))

    def test_draft_margin_default_matches_omitted_param(self):
        path = self.root / "same.jpg"
        make_jpeg(path, size=(400, 300))
        explicit = imaging.decode_rgb(path, max_edge=100, draft_margin=imaging._DRAFT_FACTOR)
        omitted = imaging.decode_rgb(path, max_edge=100)
        self.assertEqual(explicit.size, omitted.size)

    def test_aggressive_draft_margin_ignored_for_png(self):
        # draft() is unavailable for PNG — draft_margin must not break anything, the
        # path stays the same (try/except inside decode_rgb).
        path = self.root / "plain.png"
        Image.new("RGB", (300, 200), (5, 6, 7)).save(path, "PNG")
        img = imaging.decode_rgb(path, max_edge=100, draft_margin=1.0)
        self.assertIsNotNone(img)
        self.assertLessEqual(max(img.size), 100)


class TestDecodeRgbCached(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        imaging.cache_clear()

    def tearDown(self):
        imaging.cache_clear()
        self.tmp.cleanup()

    def test_cache_hit_skips_decode(self):
        path = self.root / "a.jpg"
        make_jpeg(path, color=(1, 2, 3))
        mtime = 12345.0

        calls = []
        real_decode = imaging.decode_rgb

        def counting_decode(*args, **kwargs):
            calls.append(1)
            return real_decode(*args, **kwargs)

        with unittest.mock.patch.object(imaging, "decode_rgb", counting_decode):
            first = imaging.decode_rgb_cached(path, mtime, max_edge=32)
            second = imaging.decode_rgb_cached(path, mtime, max_edge=32)

        self.assertEqual(len(calls), 1)
        self.assertIs(first, second)

    def test_mtime_change_invalidates(self):
        path = self.root / "a.jpg"
        make_jpeg(path, color=(1, 2, 3))

        calls = []
        real_decode = imaging.decode_rgb

        def counting_decode(*args, **kwargs):
            calls.append(1)
            return real_decode(*args, **kwargs)

        with unittest.mock.patch.object(imaging, "decode_rgb", counting_decode):
            imaging.decode_rgb_cached(path, 111.0, max_edge=32)
            imaging.decode_rgb_cached(path, 222.0, max_edge=32)

        self.assertEqual(len(calls), 2)

    def test_none_result_not_cached(self):
        path = self.root / "missing.jpg"

        calls = []
        real_decode = imaging.decode_rgb

        def counting_decode(*args, **kwargs):
            calls.append(1)
            return real_decode(*args, **kwargs)

        with unittest.mock.patch.object(imaging, "decode_rgb", counting_decode):
            r1 = imaging.decode_rgb_cached(path, 1.0, max_edge=32)
            r2 = imaging.decode_rgb_cached(path, 1.0, max_edge=32)

        self.assertIsNone(r1)
        self.assertIsNone(r2)
        self.assertEqual(len(calls), 2)

    def test_lru_eviction_bounds_cache_size(self):
        old_max = imaging.CACHE_MAX_ITEMS
        imaging.CACHE_MAX_ITEMS = 5
        try:
            for i in range(20):
                path = self.root / f"f{i}.jpg"
                make_jpeg(path, color=(i % 255, 0, 0))
                imaging.decode_rgb_cached(path, 1.0, max_edge=16)
            self.assertLessEqual(len(imaging._cache), 5)
        finally:
            imaging.CACHE_MAX_ITEMS = old_max
            imaging.cache_clear()

    def test_thread_safety_smoke(self):
        paths = []
        for i in range(10):
            path = self.root / f"t{i}.jpg"
            make_jpeg(path, color=(i * 10 % 255, 0, 0))
            paths.append(path)

        def decode(p):
            return imaging.decode_rgb_cached(p, 1.0, max_edge=16)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(decode, paths * 3))

        self.assertEqual(len(results), 30)
        for r in results:
            self.assertIsNotNone(r)
            self.assertLessEqual(max(r.size), 16)


if __name__ == "__main__":
    unittest.main()
