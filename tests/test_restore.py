"""F149: "try to improve" — one frame, by request, a processed copy beside the original.

The order of the cases below is the order of the risk. The first one is the whole
feature: a model that draws plausible detail is pointed at somebody's archive, and the
one thing that must be impossible is for it to touch the photograph. Everything after
that is about the copy being honest — a different name, never over an existing file, and
a REASON whenever there is no copy, because the weights come off the network and offline
is an ordinary state for this program.

The model itself is never loaded here. The loader is injected (`restore.shared_upscaler`
takes one, exactly like `naming.shared_vlm`), which is also how the "loaded on first use,
not at import" case can be stated as a fact rather than as a hope.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sorta import restore
from sorta.db import connect
from sorta.hashing import file_hash


def make_jpeg(path: Path, size=(64, 48), color=(200, 50, 50)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")
    return path


def doubling_upscaler(_model_name: str) -> restore.UpscaleFn:
    """A stand-in for Swin2SR: the same contract (an image in, a bigger image out).

    Doubling rather than x4 so the fixtures stay small — nothing in the module depends on
    the factor, and a test that measured it would be testing the stub.
    """
    def upscale(image: Image.Image) -> Image.Image:
        return image.resize((image.width * 2, image.height * 2))
    return upscale


class RestoreTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        # The cache is process-wide and lives for the life of the server; between tests it
        # would carry one test's stub into the next one's "nothing is loaded yet".
        restore.reset_upscalers()
        self.addCleanup(restore.reset_upscalers)


class TestTheOriginalIsNeverTouched(RestoreTestBase):
    """The main test of the feature. Principle #5, checked byte for byte."""

    def test_the_source_file_is_identical_afterwards(self):
        src = make_jpeg(self.root / "blurred.jpg")
        before, algo = file_hash(src)
        before_bytes = src.read_bytes()
        before_mtime = src.stat().st_mtime

        result = restore.restore_frame(src, "stub", loader=doubling_upscaler)

        self.assertTrue(result.ok, result)
        self.assertEqual(file_hash(src), (before, algo))
        self.assertEqual(src.read_bytes(), before_bytes)
        self.assertEqual(src.stat().st_mtime, before_mtime)

    def test_the_original_survives_a_failure_too(self):
        """A model that will not load must not leave the frame half-written either."""
        src = make_jpeg(self.root / "blurred.jpg")
        before = src.read_bytes()

        def broken(_name: str) -> restore.UpscaleFn:
            raise ImportError("no transformers")

        result = restore.restore_frame(src, "stub", loader=broken)

        self.assertFalse(result.ok)
        self.assertEqual(src.read_bytes(), before)


class TestTheCopy(RestoreTestBase):
    def test_the_copy_is_a_new_file_with_a_different_name(self):
        src = make_jpeg(self.root / "shot.jpg")
        result = restore.restore_frame(src, "stub", loader=doubling_upscaler)
        self.assertIsNotNone(result.path)
        assert result.path is not None
        self.assertEqual(result.path.name, "shot_restored.jpg")
        self.assertEqual(result.path.parent, src.parent)
        self.assertNotEqual(result.path.read_bytes(), src.read_bytes())

    def test_an_existing_file_is_never_overwritten(self):
        """The sorter's rule (`_1`, `_2`), for the same reason it has one."""
        src = make_jpeg(self.root / "shot.jpg")
        squatter = self.root / "shot_restored.jpg"
        squatter.write_bytes(b"not ours")

        result = restore.restore_frame(src, "stub", loader=doubling_upscaler)

        assert result.path is not None
        self.assertEqual(result.path.name, "shot_restored_1.jpg")
        self.assertEqual(squatter.read_bytes(), b"not ours")

        second = restore.restore_frame(src, "stub", loader=doubling_upscaler)
        assert second.path is not None
        self.assertEqual(second.path.name, "shot_restored_2.jpg")

    def test_the_copy_keeps_the_name_of_a_source_that_is_not_a_jpeg(self):
        """The output is always JPEG (the model returns RGB pixels), and the name says so
        — what a person reads and what is on disk have to agree."""
        src = self.root / "scan.png"
        src.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30), (10, 20, 30)).save(src, "PNG")
        result = restore.restore_frame(src, "stub", loader=doubling_upscaler)
        assert result.path is not None
        self.assertEqual(result.path.name, "scan_restored.jpg")


class TestTheInputIsBounded(RestoreTestBase):
    """x4 over a 4000 px frame is 16000 px and a memory failure — the input is scaled to
    `MAX_INPUT_EDGE` first. A compromise, stated as one in the module."""

    def test_a_large_frame_is_scaled_down_before_the_model_sees_it(self):
        src = make_jpeg(self.root / "big.jpg", size=(2400, 1800))
        seen: list[tuple[int, int]] = []

        def watching(_name: str) -> restore.UpscaleFn:
            def upscale(image: Image.Image) -> Image.Image:
                seen.append(image.size)
                return image
            return upscale

        result = restore.restore_frame(src, "stub", loader=watching)

        self.assertTrue(result.ok, result)
        self.assertEqual(len(seen), 1)
        self.assertLessEqual(max(seen[0]), restore.MAX_INPUT_EDGE)

    def test_a_small_frame_is_not_enlarged_on_the_way_in(self):
        src = make_jpeg(self.root / "small.jpg", size=(64, 48))
        seen: list[tuple[int, int]] = []

        def watching(_name: str) -> restore.UpscaleFn:
            def upscale(image: Image.Image) -> Image.Image:
                seen.append(image.size)
                return image
            return upscale

        restore.restore_frame(src, "stub", loader=watching)
        self.assertEqual(seen, [(64, 48)])


class TestAReasonNotAnEmptyResult(RestoreTestBase):
    def test_a_model_that_does_not_load_names_itself_and_writes_nothing(self):
        src = make_jpeg(self.root / "shot.jpg")

        def broken(_name: str) -> restore.UpscaleFn:
            raise ImportError("transformers is not installed")

        result = restore.restore_frame(src, "stub", loader=broken)

        self.assertFalse(result.ok)
        self.assertEqual(result.error, restore.ERROR_MODEL_UNAVAILABLE)
        self.assertIn("transformers", result.detail or "")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["shot.jpg"])

    def test_a_model_that_fails_on_the_frame_is_the_same_answer(self):
        src = make_jpeg(self.root / "shot.jpg")

        def failing(_name: str) -> restore.UpscaleFn:
            def upscale(_image: Image.Image) -> Image.Image:
                raise RuntimeError("CUDA out of memory")
            return upscale

        result = restore.restore_frame(src, "stub", loader=failing)

        self.assertEqual(result.error, restore.ERROR_MODEL_UNAVAILABLE)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["shot.jpg"])

    def test_a_frame_that_does_not_decode_says_so_and_never_loads_the_model(self):
        """A broken file must not cost a 400 MB load before the same answer comes back."""
        src = self.root / "broken.jpg"
        src.write_bytes(b"this is not an image")
        loaded: list[str] = []

        def counting(name: str) -> restore.UpscaleFn:
            loaded.append(name)
            return doubling_upscaler(name)

        result = restore.restore_frame(src, "stub", loader=counting)

        self.assertFalse(result.ok)
        self.assertEqual(result.error, restore.ERROR_DECODE_FAILED)
        self.assertEqual(loaded, [])
        self.assertEqual(restore.loaded_models(), ())

    def test_a_missing_file_is_a_decode_failure_and_not_a_crash(self):
        result = restore.restore_frame(self.root / "gone.jpg", "stub",
                                       loader=doubling_upscaler)
        self.assertEqual(result.error, restore.ERROR_DECODE_FAILED)


class TestTheModelIsLoadedOnFirstUse(RestoreTestBase):
    """Not at import, not at server start — 400 MB for a button most sessions never press
    is not a trade anybody asked for."""

    def test_importing_the_module_loads_nothing(self):
        self.assertEqual(restore.loaded_models(), ())

    def test_the_first_call_loads_and_the_second_reuses(self):
        src = make_jpeg(self.root / "shot.jpg")
        loads: list[str] = []

        def counting(name: str) -> restore.UpscaleFn:
            loads.append(name)
            return doubling_upscaler(name)

        restore.restore_frame(src, "swin", loader=counting)
        self.assertEqual(loads, ["swin"])
        self.assertEqual(restore.loaded_models(), ("swin",))

        restore.restore_frame(src, "swin", loader=counting)
        self.assertEqual(loads, ["swin"])

    def test_a_failed_load_is_not_cached(self):
        """A machine that has just been given the weights must not need a restart."""
        attempts: list[str] = []

        def flaky(name: str) -> restore.UpscaleFn:
            attempts.append(name)
            if len(attempts) == 1:
                raise OSError("weights are not cached and there is no network")
            return doubling_upscaler(name)

        src = make_jpeg(self.root / "shot.jpg")
        first = restore.restore_frame(src, "swin", loader=flaky)
        second = restore.restore_frame(src, "swin", loader=flaky)

        self.assertEqual(first.error, restore.ERROR_MODEL_UNAVAILABLE)
        self.assertTrue(second.ok, second)
        self.assertEqual(attempts, ["swin", "swin"])


class TestTheCopyIsAMemberOfTheCollection(RestoreTestBase):
    """The 2026-08-02 decision: the copy is indexed like any other file, and the link to
    its original is stored rather than guessed from a name."""

    def setUp(self):
        super().setUp()
        self.conn = connect(self.root / "test.db")
        self.addCleanup(self.conn.close)

    def add_file(self, path: Path, **columns: object) -> int:
        values: dict[str, object] = {
            "path": str(path.resolve()), "size": 10, "mtime": 0.0, "ext": "jpg",
            "media_type": "photo", "taken_at": "2019-07-14T12:00:00",
            "taken_at_source": "exif", "taken_at_confidence": "high",
            "gps_lat": 55.75, "gps_lon": 37.61, "camera_make": "Canon",
            "indexed_at": "2026-01-01",
        }
        values.update(columns)
        columns_sql = ", ".join(values)
        cur = self.conn.execute(
            f"INSERT INTO files ({columns_sql}) VALUES ({','.join('?' * len(values))})",
            tuple(values.values()))
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def row(self, file_id: int) -> sqlite3.Row:
        return self.conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()

    def test_the_copy_inherits_the_facts_of_the_frame_it_came_from(self):
        """Not scanned but DERIVED: metadata read off a re-encoded JPEG would date the
        copy by mtime, i.e. today, and file it under this year instead of 2019."""
        src = make_jpeg(self.root / "shot.jpg")
        source_id = self.add_file(src)
        result = restore.restore_frame(src, "swin", loader=doubling_upscaler)
        assert result.path is not None

        copy_id = restore.record_restored(self.conn, source_id, result.path, model="swin")

        source, copy = self.row(source_id), self.row(copy_id)
        for column in ("taken_at", "taken_at_source", "taken_at_confidence",
                       "gps_lat", "gps_lon", "camera_make"):
            with self.subTest(column=column):
                self.assertEqual(copy[column], source[column])
        self.assertEqual(copy["path"], str(result.path.resolve()))
        self.assertEqual(copy["size"], result.path.stat().st_size)
        self.assertEqual((copy["width"], copy["height"]), (128, 96))
        self.assertIsNone(copy["dup_of"])
        self.assertIsNone(copy["error"])
        self.assertIsNone(copy["phash"])   # the next run computes it
        self.assertIsNotNone(copy["hash"])  # exact-dup and copy verification read it

    def test_the_link_is_stored_and_not_guessed_from_the_name(self):
        src = make_jpeg(self.root / "shot.jpg")
        source_id = self.add_file(src)
        result = restore.restore_frame(src, "swin", loader=doubling_upscaler)
        assert result.path is not None
        copy_id = restore.record_restored(self.conn, source_id, result.path, model="swin")

        stored = self.conn.execute("SELECT * FROM restored_files").fetchall()
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["file_id"], copy_id)
        self.assertEqual(stored[0]["source_file_id"], source_id)
        self.assertEqual(stored[0]["model"], "swin")

        # A rename does not break it — that is what "stored, not guessed" means.
        renamed = result.path.with_name("holiday.jpg")
        result.path.rename(renamed)
        self.conn.execute("UPDATE files SET path = ? WHERE id = ?",
                          (str(renamed.resolve()), copy_id))
        self.conn.commit()
        self.assertEqual(restore.existing_copy(self.conn, source_id, "swin"),
                         (copy_id, str(renamed.resolve())))

    def test_the_same_frame_and_model_return_the_copy_that_exists(self):
        src = make_jpeg(self.root / "shot.jpg")
        source_id = self.add_file(src)
        result = restore.restore_frame(src, "swin", loader=doubling_upscaler)
        assert result.path is not None
        copy_id = restore.record_restored(self.conn, source_id, result.path, model="swin")

        self.assertEqual(restore.existing_copy(self.conn, source_id, "swin"),
                         (copy_id, str(result.path.resolve())))
        # A different model is a different question, not a stale answer.
        self.assertIsNone(restore.existing_copy(self.conn, source_id, "other"))

    def test_forgetting_a_copy_removes_both_of_its_rows(self):
        src = make_jpeg(self.root / "shot.jpg")
        source_id = self.add_file(src)
        result = restore.restore_frame(src, "swin", loader=doubling_upscaler)
        assert result.path is not None
        copy_id = restore.record_restored(self.conn, source_id, result.path, model="swin")

        restore.forget_copy(self.conn, copy_id)

        self.assertIsNone(restore.existing_copy(self.conn, source_id, "swin"))
        self.assertIsNone(self.row(copy_id))
        self.assertIsNotNone(self.row(source_id))


if __name__ == "__main__":
    unittest.main()
