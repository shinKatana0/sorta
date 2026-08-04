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

F169 adds the cases about the CEILING on the way in — the single number that decides
whether the copy is built from the frame or from a quarter of it. Three things are
checked and they are the whole feature: the number comes from the caller
(`features.restore_max_edge`) and not from a constant here, a frame at or under it is
handed to the model untouched, and a frame over it comes back with an answer that SAYS
the copy was rebuilt from a reduced one.
"""
from __future__ import annotations

import builtins
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Callable
from unittest import mock

from PIL import Image

from sorta import restore
from sorta.config import FeaturesConfig
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

    def test_no_path_opens_the_original_for_writing(self):
        """The same invariant one level down: not "the bytes came out equal" but "nothing
        ever asked the file system for a handle to write them with" — and on the frame
        the ceiling fires on, which is the path F169 added."""
        src = make_jpeg(self.root / "big.jpg", size=(2400, 1800))
        writes: list[Path] = []
        real_open = builtins.open

        def watching_open(file, mode="r", *args, **kwargs):
            if isinstance(file, (str, Path)) and any(c in str(mode) for c in "wax+"):
                writes.append(Path(file))
            return real_open(file, mode, *args, **kwargs)

        with mock.patch("builtins.open", watching_open):
            result = restore.restore_frame(src, "stub", max_edge=600,
                                           loader=doubling_upscaler)

        self.assertTrue(result.rebuilt, result)
        self.assertEqual([p.name for p in writes], ["big_restored.jpg"])
        self.assertNotIn(src.resolve(), [p.resolve() for p in writes])

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


def watching_upscaler(seen: list[tuple[int, int]]) -> Callable[[str], restore.UpscaleFn]:
    """A stub that records what the model was actually shown, and hands it straight back."""
    def loader(_name: str) -> restore.UpscaleFn:
        def upscale(image: Image.Image) -> Image.Image:
            seen.append(image.size)
            return image
        return upscale
    return loader


class TestTheInputIsBounded(RestoreTestBase):
    """x4 over a 4000 px frame is 16000 px and a memory failure — the input is scaled to
    the ceiling first. A compromise, stated as one in the module."""

    def test_a_large_frame_is_scaled_down_before_the_model_sees_it(self):
        src = make_jpeg(self.root / "big.jpg", size=(2400, 1800))
        seen: list[tuple[int, int]] = []

        result = restore.restore_frame(src, "stub", loader=watching_upscaler(seen))

        self.assertTrue(result.ok, result)
        self.assertEqual(len(seen), 1)
        self.assertLessEqual(max(seen[0]), restore.DEFAULT_RESTORE_MAX_EDGE)

    def test_a_small_frame_is_not_enlarged_on_the_way_in(self):
        src = make_jpeg(self.root / "small.jpg", size=(64, 48))
        seen: list[tuple[int, int]] = []

        restore.restore_frame(src, "stub", loader=watching_upscaler(seen))
        self.assertEqual(seen, [(64, 48)])


class TestTheCeilingIsASetting(RestoreTestBase):
    """F169. The ceiling decides, alone, whether a person gets their own detail back or a
    plausible redrawing of it — so it is a value the caller passes, not a number in here.
    """

    def test_the_default_is_the_default_of_the_config_key(self):
        """One number in two places is one number that will disagree with itself."""
        self.assertEqual(restore.DEFAULT_RESTORE_MAX_EDGE,
                         FeaturesConfig().restore_max_edge)

    def test_the_ceiling_the_caller_passes_is_what_the_model_is_shown(self):
        src = make_jpeg(self.root / "big.jpg", size=(2400, 1800))
        seen: list[tuple[int, int]] = []

        restore.restore_frame(src, "stub", max_edge=300, loader=watching_upscaler(seen))

        self.assertEqual(seen, [(300, 225)])

    def test_a_frame_under_the_ceiling_is_handed_over_untouched(self):
        """The case the action was built for: a small scan, nothing given up. The pixels
        are compared, not just the size — a re-encode is not "untouched"."""
        src = make_jpeg(self.root / "small.jpg", size=(640, 480))
        seen: list[Image.Image] = []

        def loader(_name: str) -> restore.UpscaleFn:
            def upscale(image: Image.Image) -> Image.Image:
                seen.append(image.copy())
                return image
            return upscale

        result = restore.restore_frame(src, "stub", max_edge=1024, loader=loader)

        with Image.open(src) as source:
            self.assertEqual(seen[0].tobytes(), source.convert("RGB").tobytes())
        self.assertEqual(seen[0].size, (640, 480))
        self.assertFalse(result.rebuilt)
        self.assertEqual((result.source_edge, result.input_edge), (640, 640))

    def test_a_frame_exactly_at_the_ceiling_is_not_reduced_either(self):
        src = make_jpeg(self.root / "edge.jpg", size=(800, 600))
        seen: list[tuple[int, int]] = []

        result = restore.restore_frame(src, "stub", max_edge=800,
                                       loader=watching_upscaler(seen))

        self.assertEqual(seen, [(800, 600)])
        self.assertFalse(result.rebuilt)


class TestTheAnswerSaysWhatTheModelWasShown(RestoreTestBase):
    """F169's other half: a copy rebuilt from a REDUCED frame is not a silent outcome.
    The copy comes back the size of the original and holds less of what was there, which
    is precisely the thing a person cannot see by looking at it."""

    def test_a_frame_over_the_ceiling_says_it_was_rebuilt(self):
        src = make_jpeg(self.root / "big.jpg", size=(2400, 1800))

        result = restore.restore_frame(src, "stub", max_edge=600,
                                       loader=doubling_upscaler)

        self.assertTrue(result.ok, result)
        self.assertTrue(result.rebuilt)
        self.assertEqual(result.source_edge, 2400)
        self.assertEqual(result.input_edge, 600)

    def test_a_frame_under_the_ceiling_claims_nothing_of_the_sort(self):
        src = make_jpeg(self.root / "small.jpg", size=(320, 240))
        result = restore.restore_frame(src, "stub", max_edge=1024,
                                       loader=doubling_upscaler)
        self.assertFalse(result.rebuilt)
        self.assertEqual((result.source_edge, result.input_edge), (320, 320))

    def test_the_original_of_a_rebuilt_copy_is_still_untouched(self):
        """The invariant does not weaken for the frames the ceiling fires on."""
        src = make_jpeg(self.root / "big.jpg", size=(2400, 1800))
        before, algo = file_hash(src)

        result = restore.restore_frame(src, "stub", max_edge=600,
                                       loader=doubling_upscaler)

        self.assertTrue(result.rebuilt)
        self.assertEqual(file_hash(src), (before, algo))

    def test_the_longer_side_of_the_frame_is_read_off_the_header(self):
        src = make_jpeg(self.root / "wide.jpg", size=(1600, 900))
        self.assertEqual(restore.source_edge(src), 1600)

    def test_a_frame_that_will_not_open_has_no_size_and_no_crash(self):
        broken = self.root / "broken.jpg"
        broken.write_bytes(b"not an image")
        self.assertEqual(restore.source_edge(broken), 0)
        self.assertEqual(restore.source_edge(self.root / "gone.jpg"), 0)
        # ...and a result with no numbers claims nothing about a rebuild.
        self.assertFalse(restore.RestoreResult(error=restore.ERROR_DECODE_FAILED).rebuilt)


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
