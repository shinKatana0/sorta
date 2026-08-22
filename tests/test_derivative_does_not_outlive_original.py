"""F210: the derivative does not outlive the original.

Two leaks, one rule. A frame sent to the bin left its 1536px preview in the cache for
months (the key is a hash of path+mtime+size, so after the deletion it can never be
recomputed and the file is simply never read again), and a passport got a preview
BEFORE anything knew it was a passport — the classification is what decodes it.

What is pinned here is the FILESYSTEM, not the code: every case asks whether the JPEG
is still in the cache directory, because "the function was called" is exactly the kind
of evidence that survives a refactor which stops deleting anything.
"""
from __future__ import annotations

import dataclasses
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from sorta import imaging, ui
from sorta.config import Config, VlmConfig
from sorta.db import connect
from sorta.junk import classify, sweep_previews, sweep_previews_for_new_classes

from tests.test_ui_dupes import DupesTestBase
from tests.test_ui_settings import SettingsTestBase


def seed_preview(path: str | Path, mtime: float, size: int, frame: int = 0) -> Path:
    """Put a JPEG exactly where the cache keeps that frame of that file.

    The layout is asked of `imaging` itself rather than spelled out a second time: a
    test that built the path by hand would keep passing after the key or the sharding
    changed, which is the one thing it must not do.
    """
    dest = imaging._preview_path(imaging.preview_key(path, mtime, size, frame))
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), (10, 20, 30)).save(dest, "JPEG")
    return dest


class PreviewCacheMixin:
    """An isolated preview cache under the case's own tmp directory.

    conftest.py already keeps the suite out of the user's %LOCALAPPDATA%; this narrows
    it to one directory per case, so counting the files in it means something.
    """

    def use_preview_cache(self, root: Path) -> Path:
        cache = root / "previews"
        patcher = mock.patch.dict(os.environ, {
            imaging.ENV_PREVIEW_DIR: str(cache),
            imaging.ENV_PREVIEW_CACHE: "1",
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        imaging.cache_clear()
        self.addCleanup(imaging.cache_clear)
        return cache

    def cached_previews(self) -> list[Path]:
        return sorted(self.cache.rglob("*.jpg")) if self.cache.exists() else []


class TestPreviewDelete(PreviewCacheMixin, unittest.TestCase):
    """imaging.preview_delete — the one place that knows where a preview lives."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cache = self.use_preview_cache(self.root)

    def make_photo(self, name: str = "big.jpg") -> tuple[Path, float, int]:
        """A frame large enough that the cache actually stores a preview of it."""
        path = self.root / name
        Image.new("RGB", (2400, 1600), (90, 120, 200)).save(path, "JPEG")
        st = path.stat()
        return path, st.st_mtime, st.st_size

    def test_the_preview_the_cache_wrote_is_the_one_that_is_deleted(self):
        # The main case, end to end through the real write path: whatever key
        # decode_rgb_preview chose, preview_delete has to find the same file.
        path, mtime, size = self.make_photo()
        self.assertIsNotNone(imaging.decode_rgb_preview(path, mtime, size, 512))
        self.assertEqual(len(self.cached_previews()), 1)

        self.assertEqual(imaging.preview_delete(path, mtime, size), 1)
        self.assertEqual(self.cached_previews(), [])

    def test_every_frame_of_a_filmstrip_goes_not_just_the_first(self):
        # F80 stores a clip as six JPEGs under the same key plus an index — deleting
        # frame 0 alone would leave the reel on disk.
        clip = self.root / "clip.mp4"
        for frame in range(imaging.VIDEO_FRAMES):
            seed_preview(clip, 11.0, 4096, frame)
        self.assertEqual(len(self.cached_previews()), imaging.VIDEO_FRAMES)

        removed = imaging.preview_delete(clip, 11.0, 4096)
        self.assertEqual(removed, imaging.VIDEO_FRAMES)
        self.assertEqual(self.cached_previews(), [])

    def test_a_hole_in_the_strip_does_not_stop_the_walk(self):
        # Eviction removes single files by their last use, so a strip with frame 2
        # missing is a real state — and stopping there would leave 3, 4 and 5 behind.
        clip = self.root / "clip.mp4"
        for frame in (0, 1, 3, 4, 5):
            seed_preview(clip, 11.0, 4096, frame)

        self.assertEqual(imaging.preview_delete(clip, 11.0, 4096), 5)
        self.assertEqual(self.cached_previews(), [])

    def test_the_walk_goes_on_while_frames_keep_being_found(self):
        # A strip written when SORTA_VIDEO_FRAMES was higher than it is now: the walk
        # is "while files are found", not "the configured count".
        clip = self.root / "clip.mp4"
        for frame in range(imaging.VIDEO_FRAMES + 3):
            seed_preview(clip, 11.0, 4096, frame)

        removed = imaging.preview_delete(clip, 11.0, 4096)
        self.assertEqual(removed, imaging.VIDEO_FRAMES + 3)
        self.assertEqual(self.cached_previews(), [])

    def test_a_missing_preview_is_a_normal_outcome(self):
        path, mtime, size = self.make_photo()
        self.assertEqual(imaging.preview_delete(path, mtime, size), 0)
        self.assertEqual(self.cached_previews(), [])

    def test_a_preview_that_will_not_unlink_does_not_raise(self):
        path, mtime, size = self.make_photo()
        dest = seed_preview(path, mtime, size)
        with mock.patch.object(Path, "unlink", side_effect=PermissionError("busy")):
            self.assertEqual(imaging.preview_delete(path, mtime, size), 0)
        self.assertTrue(dest.exists())  # still there — and nothing blew up

    def test_the_neighbouring_frame_is_not_touched(self):
        mine, mine_mtime, mine_size = self.make_photo("mine.jpg")
        other, other_mtime, other_size = self.make_photo("other.jpg")
        seed_preview(mine, mine_mtime, mine_size)
        neighbour = seed_preview(other, other_mtime, other_size)

        imaging.preview_delete(mine, mine_mtime, mine_size)
        self.assertEqual(self.cached_previews(), [neighbour])

    @unittest.skipIf(os.name == "nt", "POSIX permission bits; NTFS inherits the ACL")
    def test_the_cache_directory_is_created_private_to_this_user(self):
        path, mtime, size = self.make_photo()
        self.assertIsNotNone(imaging.decode_rgb_preview(path, mtime, size, 512))
        shard = self.cached_previews()[0].parent

        self.assertEqual(stat.S_IMODE(self.cache.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(shard.stat().st_mode), 0o700)


class TestTrashTakesThePreview(PreviewCacheMixin, DupesTestBase):
    """`_trash_files` — the single trash path of the web app."""

    def setUp(self):
        super().setUp()
        self.cache = self.use_preview_cache(self.root)

    def file_row(self, file_id: int) -> tuple[str, float, int]:
        row = self.conn.execute(
            "SELECT path, mtime, size FROM files WHERE id = ?", (file_id,)).fetchone()
        return row["path"], row["mtime"], row["size"]

    def add_with_preview(self, rel: str, *, frames: int = 1) -> tuple[int, list[Path]]:
        file_id = self.add_dupe(rel, phash="0" * 16, width=100, height=100, size=1000)
        path, mtime, size = self.file_row(file_id)
        return file_id, [seed_preview(path, mtime, size, frame)
                         for frame in range(frames)]

    def test_deleting_a_frame_takes_its_preview_with_it(self):
        file_id, previews = self.add_with_preview("a.jpg")
        self.assertTrue(previews[0].exists())
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash"):
            status, _payload = self.post("/api/photo/trash", {"file_id": file_id})

        self.assertEqual(status, 200)
        self.assertEqual(self.cached_previews(), [])

    def test_a_clip_loses_its_whole_filmstrip(self):
        file_id, previews = self.add_with_preview(
            "clip.mp4", frames=imaging.VIDEO_FRAMES)
        self.assertEqual(len(self.cached_previews()), imaging.VIDEO_FRAMES)
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash"):
            status, _payload = self.post("/api/photo/trash", {"file_id": file_id})

        self.assertEqual(status, 200)
        self.assertEqual(self.cached_previews(), [])

    def test_the_neighbours_preview_survives(self):
        doomed, _ = self.add_with_preview("a.jpg")
        _kept, kept_previews = self.add_with_preview("b.jpg")
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash"):
            status, _payload = self.post("/api/photo/trash", {"file_id": doomed})

        self.assertEqual(status, 200)
        self.assertEqual(self.cached_previews(), kept_previews)

    def test_a_frame_with_no_preview_is_deleted_as_before(self):
        file_id = self.add_dupe("a.jpg", phash="0" * 16, width=100, height=100, size=1000)
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash") as trash:
            status, payload = self.post("/api/photo/trash", {"file_id": file_id})

        self.assertEqual(status, 200)
        self.assertEqual(payload["trashed"], [{"file_id": file_id, "name": "a.jpg"}])
        self.assertEqual(len(self.trashed_paths(trash)), 1)
        self.assertIsNone(self.conn.execute(
            "SELECT id FROM files WHERE id = ?", (file_id,)).fetchone())

    def test_a_preview_that_cannot_be_removed_does_not_block_the_deletion(self):
        # The whole contract of the cleanup: tidying a derivative may not become the
        # reason the original stays.
        file_id, previews = self.add_with_preview("a.jpg")
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash") as trash, \
                mock.patch.object(Path, "unlink", side_effect=PermissionError("busy")):
            status, _payload = self.post("/api/photo/trash", {"file_id": file_id})

        self.assertEqual(status, 200)
        self.assertEqual(len(self.trashed_paths(trash)), 1)
        self.assertIsNone(self.conn.execute(
            "SELECT id FROM files WHERE id = ?", (file_id,)).fetchone())
        self.assertTrue(previews[0].exists())

    def test_a_bulk_deletion_takes_every_preview(self):
        first, _ = self.add_with_preview("a.jpg")
        second, _ = self.add_with_preview("b.jpg")
        self.start_server()

        with mock.patch("sorta.ui.common.send_to_trash"):
            status, _payload = self.post(
                "/api/photos/trash", {"file_ids": [first, second]})

        self.assertEqual(status, 200)
        self.assertEqual(self.cached_previews(), [])


class SweepTestCase(PreviewCacheMixin, unittest.TestCase):
    """A database with verdicts and a preview per frame, and nothing else."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cache = self.use_preview_cache(self.root)
        self.cfg = Config(sources=[self.root], database=self.root / "test.db", raw={})
        self.conn = connect(self.cfg.database)
        self.addCleanup(self.conn.close)
        self._n = 0

    def add_frame(self, name: str, verdict: str | None = None,
                  *, frames: int = 1) -> tuple[int, list[Path]]:
        """A `files` row (+ its verdict) whose previews are already on disk."""
        self._n += 1
        path = self.root / name
        path.write_bytes(b"not really a jpeg")
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, width, height,
                   camera_make, camera_model, indexed_at)
               VALUES (?, 1000, 0, ?, ?, 4000, 3000, NULL, NULL, '2026-01-01')""",
            (str(path), path.suffix.lstrip("."),
             "video" if path.suffix == ".mp4" else "photo"))
        file_id = cur.lastrowid
        if verdict is not None:
            self.conn.execute(
                """INSERT INTO media_class (file_id, verdict, source, score,
                       updated_at, tier)
                   VALUES (?, ?, 'clip', 0.9, '2026-01-01', 'clip')""",
                (file_id, verdict))
        self.conn.commit()
        return file_id, [seed_preview(str(path), 0.0, 1000, frame)
                         for frame in range(frames)]


class TestSweepPreviews(SweepTestCase):
    def test_a_sensitive_class_loses_its_preview_and_a_photograph_keeps_its_own(self):
        _doc, doc_previews = self.add_frame("passport.jpg", "document")
        _photo, photo_previews = self.add_frame("holiday.jpg", "photo")

        self.assertEqual(sweep_previews(self.conn, frozenset({"document"})), 1)
        self.assertFalse(doc_previews[0].exists())
        self.assertEqual(self.cached_previews(), photo_previews)

    def test_an_empty_list_removes_nothing(self):
        # Requirement 4, and the reason it is a test rather than a line of prose: an
        # empty `vlm.exclude_classes` is an INSTRUCTION ("no class of mine is private"),
        # so this case has to fail the moment anybody reintroduces a hard-coded list.
        _doc, doc_previews = self.add_frame("passport.jpg", "document")

        self.assertEqual(sweep_previews(self.conn, frozenset()), 0)
        self.assertEqual(self.cached_previews(), doc_previews)

    def test_a_sensitive_clip_loses_every_frame(self):
        _clip, previews = self.add_frame(
            "scan.mp4", "document", frames=imaging.VIDEO_FRAMES)

        self.assertEqual(sweep_previews(self.conn, frozenset({"document"})),
                         imaging.VIDEO_FRAMES)
        self.assertEqual(self.cached_previews(), [])
        self.assertFalse(any(p.exists() for p in previews))

    def test_a_frame_with_no_verdict_is_not_touched(self):
        _unclassified, previews = self.add_frame("unknown.jpg")

        self.assertEqual(sweep_previews(self.conn, frozenset({"document"})), 0)
        self.assertEqual(self.cached_previews(), previews)

    def test_a_missing_preview_does_not_break_the_sweep(self):
        self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
               VALUES ('/gone/passport.jpg', 1000, 0, 'jpg', 'photo', '2026-01-01')""")
        self.conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, updated_at, tier)
               VALUES ((SELECT id FROM files), 'document', 'clip', '2026-01-01', 'clip')""")
        self.conn.commit()

        self.assertEqual(sweep_previews(self.conn, frozenset({"document"})), 0)

    def test_a_class_that_enters_the_list_sweeps_what_is_already_on_disk(self):
        _doc, doc_previews = self.add_frame("passport.jpg", "document")

        removed = sweep_previews_for_new_classes(self.conn, (), ("document",))
        self.assertEqual(removed, 1)
        self.assertFalse(doc_previews[0].exists())

    def test_a_class_that_leaves_the_list_sweeps_nothing(self):
        _doc, doc_previews = self.add_frame("passport.jpg", "document")

        removed = sweep_previews_for_new_classes(self.conn, ("document",), ())
        self.assertEqual(removed, 0)
        self.assertEqual(self.cached_previews(), doc_previews)

    def test_an_unchanged_list_sweeps_nothing(self):
        _doc, doc_previews = self.add_frame("passport.jpg", "document")

        removed = sweep_previews_for_new_classes(
            self.conn, ("document",), ("document",))
        self.assertEqual(removed, 0)
        self.assertEqual(self.cached_previews(), doc_previews)


class TestClassifySweepsAfterTheVerdict(SweepTestCase):
    """The stage that NAMES a frame is the stage that takes its derivative away."""

    def classify_with(self, exclude: tuple[str, ...]):
        self.cfg.vlm = dataclasses.replace(VlmConfig(), exclude_classes=exclude)
        return classify(self.cfg, self.conn, use_clip=False)

    def test_the_run_that_calls_a_frame_a_screenshot_removes_its_preview(self):
        # use_clip=False keeps the case free of any model: the heuristic names
        # `Screenshot_*` a screenshot, and that is the class made sensitive here.
        _shot, shot_previews = self.add_frame("Screenshot_1.png")
        _photo, photo_previews = self.add_frame("IMG_0002.jpg")

        self.classify_with(("screenshot",))

        self.assertEqual(self.conn.execute(
            "SELECT verdict FROM media_class ORDER BY file_id").fetchall()[0][0],
            "screenshot")
        self.assertFalse(shot_previews[0].exists())
        self.assertEqual(self.cached_previews(), photo_previews)

    def test_with_an_empty_exclude_list_the_run_removes_nothing(self):
        _shot, shot_previews = self.add_frame("Screenshot_1.png")
        _photo, photo_previews = self.add_frame("IMG_0002.jpg")

        self.classify_with(())

        self.assertEqual(self.cached_previews(),
                         sorted(shot_previews + photo_previews))


class TestSettingsRouteSweeps(PreviewCacheMixin, SettingsTestBase):
    """The moment the cleanup exists for: the list is edited while the tool runs.

    `vlm.exclude_classes` has no control in the settings column today (the panel offers
    the knobs of `_SETTINGS_SPEC`), so what a route test can pin is the HOOK: when a
    save changes the list, the previews already on disk go. The change itself is made
    by standing in for `_apply_settings` — the one thing the route calls to put a saved
    value into the running config.
    """

    def setUp(self):
        super().setUp()
        self.cache = self.use_preview_cache(self.root)
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, exclude_classes=())

    def add_document(self) -> Path:
        path = self.root / "passport.jpg"
        path.write_bytes(b"not really a jpeg")
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', '2026-01-01')""", (str(path),))
        self.conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, updated_at, tier)
               VALUES (?, 'document', 'clip', '2026-01-01', 'clip')""", (cur.lastrowid,))
        self.conn.commit()
        return seed_preview(str(path), 0.0, 1000)

    def apply_and_exclude(self, classes: tuple[str, ...]):
        """`_apply_settings`, plus the change to the class list the panel cannot make."""
        real = ui._apply_settings

        def fake(cfg, values):
            real(cfg, values)
            cfg.vlm = dataclasses.replace(cfg.vlm, exclude_classes=classes)

        return mock.patch.object(ui, "_apply_settings", fake)

    def test_switching_the_protection_on_sweeps_the_previews_already_on_disk(self):
        preview = self.add_document()
        self.start_server()

        with self.apply_and_exclude(("document",)):
            status, resp = self.post_raw("/api/settings", {"vlm.workers": 3})

        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        self.assertFalse(preview.exists())
        self.assertEqual(self.cached_previews(), [])

    def test_a_save_that_does_not_change_the_list_removes_nothing(self):
        preview = self.add_document()
        self.start_server()

        status, resp = self.post_raw("/api/settings", {"vlm.workers": 3})

        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        self.assertEqual(self.cached_previews(), [preview])

    def test_a_class_leaving_the_list_removes_nothing(self):
        self.cfg.vlm = dataclasses.replace(
            self.cfg.vlm, exclude_classes=("document",))
        preview = self.add_document()
        self.start_server()

        with self.apply_and_exclude(()):
            status, _resp = self.post_raw("/api/settings", {"vlm.workers": 3})

        self.assertEqual(status, 200)
        self.assertEqual(self.cached_previews(), [preview])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
