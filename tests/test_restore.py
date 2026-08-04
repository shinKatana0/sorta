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

F185 adds the two cases the owner found in one click, and the first of them outranks
everything above except the original: a copy that was written but not indexed is a file
the next `index` run reads as a NEW photograph, so the archive grows a near-duplicate
nobody made. The cases are therefore about what is LEFT IN THE FOLDER after a failure,
not about the order of the calls that produced it. The second is the failure itself
arriving as a code (`ERROR_DATABASE_BUSY`) rather than as a stack trace, checked against
a real second writer holding the lock — SQLite's wording is not our contract.
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


def add_source_file(conn: sqlite3.Connection, path: Path, **columns: object) -> int:
    """A `files` row for an original, with the facts the copy is supposed to inherit."""
    values: dict[str, object] = {
        "path": str(path.resolve()), "size": 10, "mtime": 0.0, "ext": "jpg",
        "media_type": "photo", "taken_at": "2019-07-14T12:00:00",
        "taken_at_source": "exif", "taken_at_confidence": "high",
        "gps_lat": 55.75, "gps_lon": 37.61, "camera_make": "Canon",
        "indexed_at": "2026-01-01",
    }
    values.update(columns)
    columns_sql = ", ".join(values)
    cur = conn.execute(
        f"INSERT INTO files ({columns_sql}) VALUES ({','.join('?' * len(values))})",
        tuple(values.values()))
    conn.commit()
    return int(cur.lastrowid or 0)


def watch_writes(case: unittest.TestCase) -> list[Path]:
    """Every path opened for WRITING while the test runs, collected as it happens."""
    writes: list[Path] = []
    real_open = builtins.open

    def watching_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, Path)) and any(c in str(mode) for c in "wax+"):
            writes.append(Path(file))
        return real_open(file, mode, *args, **kwargs)

    patcher = mock.patch("builtins.open", watching_open)
    patcher.start()
    case.addCleanup(patcher.stop)
    return writes


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
        the ceiling fires on, which is the path F169 added.

        F185 moved the write: the bytes now go to a staging neighbour and the final name
        arrives by rename. So the one handle opened for writing is the staging file's, and
        the point of the test is unchanged — the original's is never among them.
        """
        src = make_jpeg(self.root / "big.jpg", size=(2400, 1800))
        writes = watch_writes(self)

        result = restore.restore_frame(src, "stub", max_edge=600,
                                       loader=doubling_upscaler)

        self.assertTrue(result.rebuilt, result)
        self.assertEqual(len(writes), 1)
        self.assertTrue(writes[0].name.endswith(restore.STAGING_SUFFIX), writes[0])
        self.assertEqual(writes[0].parent.resolve(), src.parent.resolve())
        self.assertNotIn(src.resolve(), [p.resolve() for p in writes])

    def test_the_original_is_untouched_on_the_path_that_indexes_the_copy(self):
        """F185 reordered the writes; the invariant is re-checked on the NEW path, where
        a database sits between the model and the file taking its name."""
        conn = connect(self.root / "test.db")
        self.addCleanup(conn.close)
        src = make_jpeg(self.root / "shot.jpg")
        source_id = add_source_file(conn, src)
        before, algo = file_hash(src)
        writes = watch_writes(self)

        result = restore.restore_and_record(conn, source_id, src, "swin",
                                            loader=doubling_upscaler)

        self.assertTrue(result.ok, result)
        self.assertEqual(file_hash(src), (before, algo))
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
        return add_source_file(self.conn, path, **columns)

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


class TestTheFileNeverOutlivesTheRow(RestoreTestBase):
    """F185, and the case is the reason the feature exists.

    The copy used to be written under its final name and the row inserted afterwards, so
    an insert that did not happen left a file in somebody's archive that nothing knew
    about — and the next `index` run reads such a file as a NEW photograph. It was found
    on the live archive as 81 `_restored` files with zero rows behind them. The order is
    now staged file -> row -> rename, and what these check is the leftover, not the order.
    """

    def setUp(self):
        super().setUp()
        self.conn = connect(self.root / "test.db")
        self.addCleanup(self.conn.close)
        # The frames live away from the database file: what these tests assert is the
        # exact content of the folder the copy is written into, and a journal appearing
        # beside an open connection is not something the archive should have a say in.
        self.src = make_jpeg(self.root / "photos" / "shot.jpg")
        self.source_id = add_source_file(self.conn, self.src)

    def names(self) -> list[str]:
        return sorted(p.name for p in self.src.parent.iterdir())

    def test_an_insert_that_raises_leaves_nothing_on_disk(self):
        """THE MAIN TEST. Not "the copy was not indexed" but "the directory is as it was"
        — the whole cost of this defect was the file, not the missing row."""
        def refusing(_dest: Path, _staged: Path) -> int:
            raise RuntimeError("the insert did not happen")

        with self.assertRaises(RuntimeError):
            restore.restore_frame(self.src, "swin", loader=doubling_upscaler,
                                  record=refusing)

        self.assertEqual(self.names(), ["shot.jpg"])

    def test_the_same_holds_when_the_recorder_is_the_real_one(self):
        """Through `restore_and_record`, with `record_restored` itself blowing up: the
        arrangement a caller actually uses, not just the injected stand-in."""
        with mock.patch.object(restore, "record_restored",
                               side_effect=sqlite3.IntegrityError("UNIQUE failed")):
            with self.assertRaises(sqlite3.IntegrityError):
                restore.restore_and_record(self.conn, self.source_id, self.src, "swin",
                                           loader=doubling_upscaler)

        self.assertEqual(self.names(), ["shot.jpg"])
        self.assertIsNone(restore.existing_copy(self.conn, self.source_id, "swin"))

    def test_the_name_is_free_again_after_a_failure(self):
        """A failed press must not push the next one to `_restored_1`: nothing was left
        occupying the name, so the retry gets the name a person would expect."""
        def refusing(_dest: Path, _staged: Path) -> int:
            raise RuntimeError("no")

        with self.assertRaises(RuntimeError):
            restore.restore_frame(self.src, "swin", loader=doubling_upscaler,
                                  record=refusing)
        result = restore.restore_and_record(self.conn, self.source_id, self.src, "swin",
                                            loader=doubling_upscaler)

        assert result.path is not None
        self.assertEqual(result.path.name, "shot_restored.jpg")

    def test_the_row_is_in_before_the_file_takes_its_name(self):
        """The order itself, seen from inside the recorder: at the moment the row is
        written the final name does not exist yet and the bytes are in the staging file.
        """
        seen: dict[str, object] = {}

        def looking(dest: Path, staged: Path) -> int:
            seen["dest_exists"] = dest.exists()
            seen["staged_exists"] = staged.exists()
            seen["staged_dir"] = staged.parent.resolve()
            seen["staged_suffix"] = staged.suffix
            return restore.record_restored(self.conn, self.source_id, dest, model="swin",
                                           measured_from=staged)

        result = restore.restore_frame(self.src, "swin", loader=doubling_upscaler,
                                       record=looking)

        self.assertTrue(result.ok, result)
        self.assertEqual(seen["dest_exists"], False)
        self.assertEqual(seen["staged_exists"], True)
        self.assertEqual(seen["staged_dir"], self.src.parent.resolve())
        self.assertEqual(seen["staged_suffix"], restore.STAGING_SUFFIX)

    def test_a_success_leaves_the_final_file_and_no_staging_debris(self):
        result = restore.restore_and_record(self.conn, self.source_id, self.src, "swin",
                                            loader=doubling_upscaler)

        assert result.path is not None
        self.assertTrue(result.ok, result)
        self.assertEqual(self.names(), ["shot.jpg", "shot_restored.jpg"])
        self.assertEqual(result.path.name, "shot_restored.jpg")

    def test_the_row_records_the_final_path_and_the_bytes_that_landed(self):
        """The row is written while the file is still staged, so the thing that could
        silently go wrong is the row describing the staging name — or a size and a hash
        taken over something other than what ended up on disk."""
        result = restore.restore_and_record(self.conn, self.source_id, self.src, "swin",
                                            loader=doubling_upscaler)
        assert result.path is not None

        row = self.conn.execute("SELECT * FROM files WHERE id = ?",
                                (result.file_id,)).fetchone()
        self.assertEqual(row["path"], str(result.path.resolve()))
        self.assertNotIn(restore.STAGING_SUFFIX, row["path"])
        self.assertEqual(row["size"], result.path.stat().st_size)
        self.assertEqual(row["hash"], file_hash(result.path)[0])
        self.assertEqual((row["width"], row["height"]), (128, 96))
        self.assertEqual(restore.existing_copy(self.conn, self.source_id, "swin"),
                         (result.file_id, str(result.path.resolve())))

    def test_the_new_row_is_reported_back(self):
        """The caller indexes and draws a card in one go; a result that made it look the
        id up again would be a second query against the same answer."""
        result = restore.restore_and_record(self.conn, self.source_id, self.src, "swin",
                                            loader=doubling_upscaler)
        stored = self.conn.execute("SELECT file_id FROM restored_files").fetchone()
        self.assertEqual(result.file_id, stored["file_id"])
        # ...and a plain `restore_frame` claims nothing about the index at all.
        plain = restore.restore_frame(self.src, "swin", loader=doubling_upscaler)
        self.assertEqual(plain.file_id, 0)


class TestABusyIndexIsAReasonAndNotACrash(RestoreTestBase):
    """F185's second half. SQLite allows ONE writer, an index stage can be running from
    the terminal, and the busy-guard (F145) only knows about runs started from the
    interface — so a person pressing the button while `junk` runs got a stack trace out
    of a request handler. Foreseeable states are codes here, like the other three.
    """

    def setUp(self):
        super().setUp()
        self.db_path = self.root / "test.db"
        self.conn = connect(self.db_path)
        self.addCleanup(self.conn.close)
        # The waiting is not what is under test — without this the lock below costs the
        # driver's default five seconds of patience before it says the same thing.
        self.conn.execute("PRAGMA busy_timeout = 0")
        # As above: the frames sit in their own folder so what these assert is the archive
        # and not sqlite's WAL sidecars.
        self.src = make_jpeg(self.root / "photos" / "shot.jpg")
        self.source_id = add_source_file(self.conn, self.src)

    def names(self) -> list[str]:
        return sorted(p.name for p in self.src.parent.iterdir())

    def hold_the_write_lock(self) -> None:
        """A REAL second writer, which is what the terminal is. The message SQLite puts on
        the exception is SQLite's to reword, so nothing here reads it."""
        locker = sqlite3.connect(str(self.db_path), timeout=0)
        self.addCleanup(locker.close)
        locker.execute("BEGIN IMMEDIATE")
        self.addCleanup(locker.rollback)

    def test_a_locked_index_comes_back_as_a_code(self):
        self.hold_the_write_lock()

        result = restore.restore_and_record(self.conn, self.source_id, self.src, "swin",
                                            loader=doubling_upscaler)

        self.assertFalse(result.ok)
        self.assertEqual(result.error, restore.ERROR_DATABASE_BUSY)
        self.assertIn("OperationalError", result.detail or "")

    def test_a_locked_index_leaves_no_file_behind_either(self):
        """The two defects were one click: the busy database is exactly the failure that
        was leaving the orphan."""
        self.hold_the_write_lock()

        restore.restore_and_record(self.conn, self.source_id, self.src, "swin",
                                   loader=doubling_upscaler)

        self.assertEqual(self.names(), ["shot.jpg"])

    def test_busy_is_not_the_same_answer_as_a_write_that_failed(self):
        """One is temporary and the same press works a minute later; the other is not.
        The interface decides whether to offer "try again" off this difference, so the
        two codes must not collapse into one."""
        self.hold_the_write_lock()
        busy = restore.restore_and_record(self.conn, self.source_id, self.src, "swin",
                                          loader=doubling_upscaler)

        with mock.patch.object(Image.Image, "save",
                               side_effect=OSError("no space left on device")):
            written = restore.restore_frame(self.src, "swin", loader=doubling_upscaler)

        self.assertEqual(busy.error, restore.ERROR_DATABASE_BUSY)
        self.assertEqual(written.error, restore.ERROR_WRITE_FAILED)
        self.assertNotEqual(busy.error, written.error)
        self.assertTrue(restore.ERROR_DATABASE_BUSY.endswith("busy"))
        # A write that failed leaves nothing behind either.
        self.assertEqual(self.names(), ["shot.jpg"])

    def test_a_database_error_that_is_not_busy_is_still_a_defect(self):
        """"The index is busy, try again" about a broken query would be a lie that never
        stops being one — only SQLITE_BUSY/SQLITE_LOCKED become the code."""
        def broken(_dest: Path, _staged: Path) -> int:
            self.conn.execute("SELECT * FROM a_table_that_is_not_there")
            return 0

        with self.assertRaises(sqlite3.OperationalError):
            restore.restore_frame(self.src, "swin", loader=doubling_upscaler,
                                  record=broken)

        self.assertEqual(self.names(), ["shot.jpg"])

    def test_the_message_is_read_only_when_there_is_no_result_code(self):
        """A driver that attaches no `sqlite_errorcode` still gets classified — the text
        is a fallback, never the contract the tests above rest on."""
        def busy_without_a_code(_dest: Path, _staged: Path) -> int:
            raise sqlite3.OperationalError("database is locked")

        result = restore.restore_frame(self.src, "swin", loader=doubling_upscaler,
                                       record=busy_without_a_code)

        self.assertEqual(result.error, restore.ERROR_DATABASE_BUSY)

    def test_a_second_press_returns_the_copy_and_computes_nothing(self):
        """The repeat press is `existing_copy` and no model at all — the pairing F185 must
        not have broken while moving the write around it."""
        first = restore.restore_and_record(self.conn, self.source_id, self.src, "swin",
                                           loader=doubling_upscaler)
        assert first.path is not None

        loads: list[str] = []

        def counting(name: str) -> restore.UpscaleFn:
            loads.append(name)
            return doubling_upscaler(name)

        existing = restore.existing_copy(self.conn, self.source_id, "swin")
        self.assertEqual(existing, (first.file_id, str(first.path.resolve())))
        self.assertEqual(loads, [])
        self.assertEqual(self.names(), ["shot.jpg", "shot_restored.jpg"])
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM restored_files").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
