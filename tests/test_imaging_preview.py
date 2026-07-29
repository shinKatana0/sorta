"""F67: decode_rgb_preview — the lazy disk preview cache.

Every test redirects the cache into tmp via SORTA_PREVIEW_DIR — the suite must
never write into the developer's %LOCALAPPDATA%.
"""
from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from sorta import imaging


def make_photo(
    path: Path, size=(2400, 1600), seed: int = 7, orientation: int | None = None,
    fmt: str = "JPEG",
) -> None:
    """A synthetic frame with real texture (low-frequency blobs + a gradient).

    A flat fill would make the pHash-drift test meaningless (any downscale gives the
    same hash), and pure noise is destroyed differently by different resamplers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    rng = np.random.default_rng(seed)
    blobs = Image.fromarray(
        rng.integers(0, 255, size=(12, 16, 3), dtype=np.uint8), "RGB",
    ).resize((w, h), Image.Resampling.BICUBIC)
    gradient = np.linspace(0, 90, w, dtype=np.float32)[None, :, None]
    arr = np.clip(np.asarray(blobs, dtype=np.float32) + gradient, 0, 255).astype(np.uint8)
    kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        kwargs["exif"] = exif
    Image.fromarray(arr, "RGB").save(path, fmt, **kwargs)


def stat_key(path: Path) -> tuple[float, int]:
    st = path.stat()
    return st.st_mtime, st.st_size


class PreviewCacheTestCase(unittest.TestCase):
    """Base: an isolated tmp source dir + an isolated preview cache dir."""

    env: dict[str, str] = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = self.root / "previews"
        env = {imaging.ENV_PREVIEW_DIR: str(self.cache), imaging.ENV_PREVIEW_CACHE: "1"}
        env.update(self.env)
        self._patcher = unittest.mock.patch.dict(os.environ, env)
        self._patcher.start()
        imaging.cache_clear()

    def tearDown(self):
        self._patcher.stop()
        imaging.cache_clear()
        self.tmp.cleanup()

    def previews(self) -> list[Path]:
        return sorted(self.cache.rglob("*.jpg")) if self.cache.exists() else []


class TestPreviewSettings(unittest.TestCase):
    def test_dir_env_override(self):
        with unittest.mock.patch.dict(os.environ, {imaging.ENV_PREVIEW_DIR: "/tmp/pv"}):
            self.assertEqual(imaging.preview_dir(), Path("/tmp/pv"))

    @unittest.skipUnless(os.name == "nt", "the LOCALAPPDATA branch builds a WindowsPath")
    def test_dir_default_windows(self):
        """The mirror image of test_dir_falls_back_to_home_cache, and the same rule.

        Path() picks its flavour from os.name at instantiation, so the patch makes
        preview_dir() build a WindowsPath, which cannot exist on Linux. And the skip has
        to be the decorator rather than a check inside the test: an exception raised while
        `os.name` is patched takes the whole session down with an INTERNALERROR — pytest's
        own Path(os.getcwd()) fails too, so there is not even a test name in the report.
        """
        env = {"LOCALAPPDATA": r"C:\Users\u\AppData\Local"}
        with unittest.mock.patch.dict(os.environ, env, clear=True), \
                unittest.mock.patch.object(os, "name", "nt"):
            self.assertEqual(
                imaging.preview_dir(), Path(r"C:\Users\u\AppData\Local") / "sorta" / "previews")

    def test_dir_ignores_localappdata_off_windows(self):
        """The other side of the `os.name == "nt"` condition, and it runs everywhere.

        Keeps the skip above from hiding anything: the guard itself — LOCALAPPDATA is
        only honoured on Windows — is checked on every platform, without patching
        os.name in the direction that the running interpreter cannot construct.
        """
        home = Path(tempfile.gettempdir()) / "fake-home"
        env = {"LOCALAPPDATA": str(Path(tempfile.gettempdir()) / "appdata")}
        with unittest.mock.patch.dict(os.environ, env, clear=True), \
                unittest.mock.patch.object(Path, "home", lambda: home):
            expected = (
                Path(env["LOCALAPPDATA"]) / "sorta" / "previews" if os.name == "nt"
                else home / ".cache" / "sorta" / "previews"
            )
            self.assertEqual(imaging.preview_dir(), expected)

    def test_dir_falls_back_to_home_cache(self):
        """Without LOCALAPPDATA the dir comes from the home cache.

        `os.name` is deliberately NOT patched to "posix": Path() picks its flavour
        from os.name at instantiation, so such a patch builds a PosixPath, which is
        unusable on Windows. Clearing the environment reaches the same branch on
        every platform.
        """
        home = Path(tempfile.gettempdir()) / "fake-home"
        with unittest.mock.patch.dict(os.environ, {}, clear=True), \
                unittest.mock.patch.object(Path, "home", lambda: home):
            self.assertEqual(
                imaging.preview_dir(), home / ".cache" / "sorta" / "previews")

    def test_enabled_by_default_and_off_switches(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(imaging.preview_cache_enabled())
        for value in ("0", "false", "NO", "off"):
            with unittest.mock.patch.dict(os.environ, {imaging.ENV_PREVIEW_CACHE: value}):
                self.assertFalse(imaging.preview_cache_enabled())

    def test_int_envs_fall_back_on_garbage(self):
        env = {imaging.ENV_PREVIEW_MAX_EDGE: "nope", imaging.ENV_PREVIEW_QUALITY: "-3"}
        with unittest.mock.patch.dict(os.environ, env):
            self.assertEqual(imaging.preview_max_edge(), imaging.PREVIEW_MAX_EDGE)
            self.assertEqual(imaging.preview_quality(), imaging.PREVIEW_QUALITY)
        with unittest.mock.patch.dict(os.environ, {imaging.ENV_PREVIEW_MAX_EDGE: "512"}):
            self.assertEqual(imaging.preview_max_edge(), 512)

    def test_key_changes_with_mtime_size_and_path(self):
        base = imaging.preview_key("/photos/a.jpg", 1.0, 100)
        self.assertEqual(base, imaging.preview_key("/photos/a.jpg", 1.0, 100))
        self.assertNotEqual(base, imaging.preview_key("/photos/a.jpg", 2.0, 100))
        self.assertNotEqual(base, imaging.preview_key("/photos/a.jpg", 1.0, 200))
        self.assertNotEqual(base, imaging.preview_key("/photos/b.jpg", 1.0, 100))


class TestPreviewDisabled(PreviewCacheTestCase):
    env = {imaging.ENV_PREVIEW_CACHE: "0"}

    def test_matches_decode_rgb_and_writes_nothing(self):
        src = self.root / "a.jpg"
        make_photo(src)
        mtime, size = stat_key(src)
        for max_edge, gray in ((96, True), (448, False), (None, False)):
            direct = imaging.decode_rgb(src, max_edge, grayscale=gray)
            through = imaging.decode_rgb_preview(
                src, mtime, size, max_edge=max_edge, grayscale=gray)
            self.assertIsNotNone(through)
            self.assertEqual(through.size, direct.size)
            self.assertEqual(through.mode, direct.mode)
        self.assertEqual(self.previews(), [])
        self.assertFalse(self.cache.exists())


class TestPreviewCache(PreviewCacheTestCase):
    def test_creates_one_preview_and_reuses_it(self):
        src = self.root / "a.jpg"
        make_photo(src)
        mtime, size = stat_key(src)

        opened: list[str] = []
        real_decode = imaging.decode_rgb

        def counting_decode(path, *args, **kwargs):
            opened.append(str(path))
            return real_decode(path, *args, **kwargs)

        with unittest.mock.patch.object(imaging, "decode_rgb", counting_decode):
            first = imaging.decode_rgb_preview(src, mtime, size, max_edge=96, grayscale=True)
            second = imaging.decode_rgb_preview(src, mtime, size, max_edge=96, grayscale=True)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.size, second.size)
        self.assertEqual(len(self.previews()), 1)
        # the source is opened exactly once — the second call only reads the preview
        self.assertEqual([p for p in opened if p == str(src)], [str(src)])
        self.assertIn(str(self.previews()[0]), opened)

    def test_preview_is_sharded_and_matches_the_key(self):
        src = self.root / "a.jpg"
        make_photo(src)
        mtime, size = stat_key(src)
        imaging.decode_rgb_preview(src, mtime, size, max_edge=96, grayscale=True)
        key = imaging.preview_key(src, mtime, size)
        dest = self.cache / key[:2] / f"{key}.jpg"
        self.assertTrue(dest.is_file())

    def test_stored_preview_is_bounded_by_preview_max_edge(self):
        src = self.root / "a.jpg"
        make_photo(src, size=(2400, 1600))
        mtime, size = stat_key(src)
        imaging.decode_rgb_preview(src, mtime, size, max_edge=96, grayscale=True)
        with Image.open(self.previews()[0]) as stored:
            self.assertEqual(max(stored.size), imaging.PREVIEW_MAX_EDGE)
            self.assertEqual(stored.mode, "RGB")  # stored in RGB, grayscale is a read option

    def test_render_matches_decode_rgb_cold_and_warm(self):
        src = self.root / "a.jpg"
        make_photo(src)
        mtime, size = stat_key(src)
        for max_edge, gray in ((96, True), (448, False), (1280, False)):
            direct = imaging.decode_rgb(src, max_edge, grayscale=gray)
            cold = imaging.decode_rgb_preview(src, mtime, size, max_edge=max_edge, grayscale=gray)
            warm = imaging.decode_rgb_preview(src, mtime, size, max_edge=max_edge, grayscale=gray)
            for got in (cold, warm):
                self.assertIsNotNone(got)
                self.assertEqual(got.size, direct.size)
                self.assertEqual(got.mode, direct.mode)
            # a cold call must not differ from a warm one — otherwise a pHash would
            # depend on whether the cache happened to be warm
            self.assertEqual(list(cold.getdata()), list(warm.getdata()))

    def test_change_of_file_invalidates_and_returns_new_content(self):
        src = self.root / "a.jpg"
        make_photo(src, seed=1)
        mtime, size = stat_key(src)
        before = imaging.decode_rgb_preview(src, mtime, size, max_edge=96)
        self.assertEqual(len(self.previews()), 1)

        make_photo(src, seed=99)  # different content
        st = src.stat()
        os.utime(src, (st.st_atime, st.st_mtime + 100))  # guarantee a different key
        new_mtime, new_size = stat_key(src)
        self.assertNotEqual(
            imaging.preview_key(src, mtime, size), imaging.preview_key(src, new_mtime, new_size))
        after = imaging.decode_rgb_preview(src, new_mtime, new_size, max_edge=96)

        self.assertEqual(len(self.previews()), 2)  # the old entry is simply never read again
        self.assertNotEqual(list(before.getdata()), list(after.getdata()))

    def test_small_source_is_not_cached(self):
        src = self.root / "small.png"
        make_photo(src, size=(800, 600), fmt="PNG")
        mtime, size = stat_key(src)
        img = imaging.decode_rgb_preview(src, mtime, size, max_edge=448)
        direct = imaging.decode_rgb(src, 448)
        self.assertIsNotNone(img)
        self.assertEqual(img.size, direct.size)
        self.assertEqual(self.previews(), [])

    def test_source_exactly_at_the_limit_is_not_cached(self):
        src = self.root / "edge.jpg"
        make_photo(src, size=(imaging.PREVIEW_MAX_EDGE, 900))
        mtime, size = stat_key(src)
        imaging.decode_rgb_preview(src, mtime, size, max_edge=448)
        self.assertEqual(self.previews(), [])

    def test_orientation_is_stored_unrotated_but_applied_on_read(self):
        src = self.root / "rot.jpg"
        make_photo(src, size=(2400, 1600), orientation=6)  # 90° rotation
        mtime, size = stat_key(src)
        direct = imaging.decode_rgb(src, 400, apply_orientation=True)

        cold = imaging.decode_rgb_preview(src, mtime, size, max_edge=400, apply_orientation=True)
        warm = imaging.decode_rgb_preview(src, mtime, size, max_edge=400, apply_orientation=True)
        self.assertEqual(cold.size, direct.size)
        self.assertEqual(warm.size, direct.size)  # portrait: the edges swapped
        self.assertLess(cold.size[0], cold.size[1])

        # the frame ON DISK stays unrotated — consumers that do not ask for the
        # orientation must get exactly what decode_rgb gave them before
        flat = imaging.decode_rgb_preview(src, mtime, size, max_edge=400)
        self.assertEqual(flat.size, imaging.decode_rgb(src, 400).size)

    def test_missing_source_returns_none(self):
        missing = self.root / "nope.jpg"
        self.assertIsNone(imaging.decode_rgb_preview(missing, 1.0, 10, max_edge=96))
        self.assertEqual(self.previews(), [])

    def test_corrupt_source_returns_none(self):
        src = self.root / "broken.jpg"
        src.write_bytes(b"\xff\xd8 not really a jpeg")
        mtime, size = stat_key(src)
        self.assertIsNone(imaging.decode_rgb_preview(src, mtime, size, max_edge=96))
        self.assertEqual(self.previews(), [])

    def test_truncated_source_after_header_returns_none(self):
        # the header parses (_peek succeeds) but the pixel decode fails
        src = self.root / "trunc.jpg"
        make_photo(src)
        data = src.read_bytes()
        src.write_bytes(data[:len(data) // 3])
        mtime, size = stat_key(src)
        self.assertIsNone(imaging.decode_rgb_preview(src, mtime, size, max_edge=96))
        self.assertEqual(self.previews(), [])

    def test_cache_clear_removes_the_directory(self):
        src = self.root / "a.jpg"
        make_photo(src)
        imaging.decode_rgb_preview(src, *stat_key(src), max_edge=96)
        self.assertTrue(self.cache.exists())
        imaging.preview_cache_clear()
        self.assertFalse(self.cache.exists())
        imaging.preview_cache_clear()  # a missing dir is not an error


class TestPreviewResilience(PreviewCacheTestCase):
    def test_corrupt_cache_entry_is_dropped_and_regenerated(self):
        src = self.root / "a.jpg"
        make_photo(src)
        mtime, size = stat_key(src)
        expected = imaging.decode_rgb_preview(src, mtime, size, max_edge=96, grayscale=True)

        entry = self.previews()[0]
        entry.write_bytes(b"garbage, not a jpeg at all")

        again = imaging.decode_rgb_preview(src, mtime, size, max_edge=96, grayscale=True)
        self.assertIsNotNone(again)
        self.assertEqual(again.size, expected.size)
        with Image.open(entry) as stored:  # the bad entry was rewritten, not kept
            stored.load()

    def test_uncreatable_cache_dir_degrades_to_a_plain_decode(self):
        blocker = self.root / "blocker.txt"
        blocker.write_text("not a directory", encoding="utf-8")
        src = self.root / "a.jpg"
        make_photo(src)
        mtime, size = stat_key(src)
        with unittest.mock.patch.dict(
                os.environ, {imaging.ENV_PREVIEW_DIR: str(blocker / "previews")}):
            img = imaging.decode_rgb_preview(src, mtime, size, max_edge=96, grayscale=True)
        self.assertIsNotNone(img)
        self.assertEqual(img.size, imaging.decode_rgb(src, 96, grayscale=True).size)

    def test_write_failure_does_not_break_the_call_or_leave_temp_files(self):
        src = self.root / "a.jpg"
        make_photo(src)
        mtime, size = stat_key(src)

        def failing_save(*args, **kwargs):
            raise OSError("disk full")

        with unittest.mock.patch.object(Image.Image, "save", failing_save):
            img = imaging.decode_rgb_preview(src, mtime, size, max_edge=96, grayscale=True)
        self.assertIsNotNone(img)
        self.assertEqual(self.previews(), [])
        self.assertEqual(list(self.cache.rglob("*.tmp")), [])

    def test_parallel_calls_on_the_same_path_never_see_a_half_file(self):
        src = self.root / "a.jpg"
        make_photo(src)
        mtime, size = stat_key(src)

        def work(_):
            return imaging.decode_rgb_preview(src, mtime, size, max_edge=96, grayscale=True)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(work, range(8)))

        expected = imaging.decode_rgb(src, 96, grayscale=True)
        for img in results:
            self.assertIsNotNone(img)
            self.assertEqual(img.size, expected.size)
            self.assertEqual(img.mode, "L")
        # At most one final file — that is the atomicity claim this test is about.
        # NOT "exactly one": on Windows os.replace fails when the destination is open,
        # and eight threads racing on ONE path can all lose. The write path is
        # best-effort by contract (every failure degrades to a plain decode), and
        # population is covered by test_creates_one_preview_and_reuses_it, which is
        # not racy. Insisting on population here made the suite flaky without
        # asserting anything the design promises.
        self.assertLessEqual(len(self.previews()), 1)
        self.assertEqual(list(self.cache.rglob("*.tmp")), [])
        for stored_path in self.previews():
            with Image.open(stored_path) as stored:
                stored.load()  # a truncated write would raise here
                self.assertEqual(max(stored.size), imaging.PREVIEW_MAX_EDGE)


if __name__ == "__main__":
    unittest.main()
