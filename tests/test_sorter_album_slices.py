"""F139: the albums of the remaining slices — classes and quality.

`sorter.plan_album` grew six kinds: `product`/`screenshot`/`meme`, selected on
`media_class.verdict`, and `blurred`/`eyes_closed` (plus `no_subject`, retired by F177),
selected on `frame_quality` by the same rule that draws the "Review" workspace. The engine itself is
F34's and is not re-tested here; what is pinned below is what a new kind can get wrong:

* the slice it selects (and, for `blurred`, that it is a WINDOW and not a threshold —
  `features.blur_review_max`, the same key that bounds the list a person looks at);

F150 adds a seventh, `low_resolution`, and it has a class of its own below: it is the
one kind here whose membership no model produced, so what has to be pinned about it is
different (see `TestLowResolutionSlice`).

* the classes that must NOT be gatherable: a class listed in `vlm.exclude_classes`
  (`document` by default) is refused here, not merely left without a button;
* duplicates and unreadable files stay out of every one of them;
* the inherited properties that are only properties if they are checked — dry-run writes
  nothing, apply journals BEFORE the operation, `move` warns, a repeat gather makes no
  `_1` copies.
"""
from __future__ import annotations

import dataclasses
import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sorta import i18n
from sorta.sorter import (
    ALBUM_FOLDER_KEYS,
    CLASS_ALBUM_KINDS,
    QUALITY_ALBUM_KINDS,
    plan_album,
)

from tests.test_sorter import SorterTestBase

# F179: the eyes slice selects on `eye_openness` — the eyelid geometry — inside the window
# `features.eye_openness_max` (0.18 by default). One value comfortably inside it and one
# comfortably outside, named so that moving the default moves one line and not a dozen.
CLOSED = 0.05
OPEN = 0.30


class SliceAlbumTestBase(SorterTestBase):
    def classify(self, file_id: int, verdict: str) -> None:
        """A `media_class` row as the junk stage writes it — the bucket of a frame."""
        self.conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, updated_at)
               VALUES (?, ?, 'vlm', '2026-01-01')""", (file_id, verdict))
        self.conn.commit()

    def add_classified(self, rel: str, verdict: str, **kwargs) -> int:
        file_id = self.add_file(rel, **kwargs)
        self.classify(file_id, verdict)
        return file_id

    def add_quality(self, rel: str, *, sharpness: float | None = 500.0,
                    eye_openness: float | None = None,
                    verdict: str = "photo", **kwargs) -> int:
        """A photograph with a `frame_quality` row — the population of the flat slices.

        The default sharpness sits far above `features.blur_review_max`, so a frame made
        for the eyes case does not quietly join the blurred one as well. F179: the eyes
        case says how OPEN the eyes are — CLOSED/OPEN below are one value inside the
        default window (`features.eye_openness_max`) and one outside it.
        """
        file_id = self.add_classified(rel, verdict, **kwargs)
        self.conn.execute(
            """INSERT INTO frame_quality (file_id, sharpness, eye_openness,
                   source, updated_at)
               VALUES (?, ?, ?, 'clip', '2026-01-01')""",
            (file_id, sharpness, eye_openness))
        self.conn.commit()
        return file_id

    def add_sized(self, rel: str, width: int, height: int, *,
                  verdict: str = "photo", **kwargs) -> int:
        """A photograph the indexer measured — the population of the F150 slice.

        No `frame_quality` row: this slice does not need one, and building the fixture
        without one is how the test says so.
        """
        file_id = self.add_classified(rel, verdict, **kwargs)
        self.conn.execute("UPDATE files SET width = ?, height = ? WHERE id = ?",
                          (width, height, file_id))
        self.conn.commit()
        return file_id

    def gather(self, kind: str, **kwargs):
        """plan_album for a selectorless slice, with the CLI chatter swallowed."""
        with redirect_stdout(io.StringIO()):
            return plan_album(self.cfg, self.conn, kind, "", self.dest, **kwargs)

    def ids(self, kind: str, **kwargs) -> list[int]:
        return [item.file_id for item in self.gather(kind, **kwargs).plan]


class TestClassSlices(SliceAlbumTestBase):
    def test_each_class_gathers_its_own_verdict_and_nobody_else(self):
        by_verdict = {v: self.add_classified(f"{v}.jpg", v)
                      for v in ("product", "screenshot", "meme", "document", "photo")}
        for kind in CLASS_ALBUM_KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(self.ids(kind), [by_verdict[kind]])

    def test_a_frame_without_a_verdict_is_in_no_class_slice(self):
        # junk has not run on it: not a bucket, simply unclassified.
        self.add_file("unclassified.jpg")
        for kind in CLASS_ALBUM_KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(self.ids(kind), [])

    def test_duplicates_and_unreadable_files_stay_out(self):
        canonical = self.add_classified("a.jpg", "product")
        duplicate = self.add_classified("b.jpg", "product")
        broken = self.add_classified("c.jpg", "product")
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?",
                          (canonical, duplicate))
        self.conn.execute("UPDATE files SET error = 'cannot read' WHERE id = ?", (broken,))
        self.conn.commit()
        self.assertEqual(self.ids("product"), [canonical])

    def test_where_still_narrows_the_slice(self):
        paris = self.add_classified("a.jpg", "product", country="France", city="Paris")
        self.add_classified("b.jpg", "product", country="Russia", city="Moskva")
        self.assertEqual(self.ids("product", where=["city=Paris"]), [paris])

    def test_the_default_name_is_the_folder_of_that_class(self):
        self.add_classified("a.jpg", "product")
        self.add_classified("b.jpg", "screenshot")
        self.assertEqual(self.gather("product").album_name,
                         i18n.folder("products", "en"))
        self.assertEqual(self.gather("screenshot").album_name,
                         i18n.folder("screenshots", "en"))

    def test_the_default_name_follows_the_configured_language(self):
        self.cfg.raw = {"language": "ru"}
        self.add_classified("a.jpg", "product")
        self.assertEqual(self.gather("product").album_name,
                         i18n.folder("products", "ru"))

    def test_an_explicit_name_still_wins(self):
        self.add_classified("a.jpg", "product")
        report = self.gather("product", album_name="Товар")
        self.assertEqual(report.album_name, "Товар")
        self.assertEqual(Path(report.dest).name, "Товар")


class TestSensitiveClassesHaveNoAlbum(SliceAlbumTestBase):
    """The F133 rule, kept whole: a class in `vlm.exclude_classes` has a counter and
    nothing else. `document` is the default member and the reason the key exists —
    gathering a folder of passports in one click is what it is there to prevent."""

    def test_document_is_not_an_album_kind_at_all(self):
        self.add_classified("passport.jpg", "document")
        with self.assertRaises(ValueError):
            self.gather("document")

    def test_a_class_moved_into_the_key_loses_its_album(self):
        # Asserting the default alone cannot tell a config read from a hard-coded
        # "document" — only changing the key can.
        self.add_classified("chair.jpg", "product")
        self.assertEqual(len(self.ids("product")), 1)
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, exclude_classes=("product",))
        with self.assertRaises(ValueError):
            self.gather("product")

    def test_a_class_outside_the_key_keeps_its_album(self):
        self.add_classified("meme.jpg", "meme")
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, exclude_classes=("product",))
        self.assertEqual(len(self.ids("meme")), 1)

    def test_the_refusal_writes_nothing(self):
        self.add_classified("passport.jpg", "document")
        with self.assertRaises(ValueError):
            self.gather("document", apply=True)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)
        self.assertFalse(self.dest.exists())


class TestQualitySlices(SliceAlbumTestBase):
    def test_each_quality_slice_gathers_its_own_frames(self):
        blurred = self.add_quality("blurred.jpg", sharpness=10.0)
        eyes = self.add_quality("eyes.jpg", eye_openness=CLOSED)
        self.add_quality("fine.jpg", eye_openness=OPEN)
        self.assertEqual(self.ids("blurred"), [blurred])
        self.assertEqual(self.ids("eyes_closed"), [eyes])

    def test_a_frame_nobody_asked_about_is_not_an_answer(self):
        # NULL means "not asked" (schema) and must never be shown as "eyes closed".
        self.add_quality("unasked.jpg", eye_openness=None)
        self.assertEqual(self.ids("eyes_closed"), [])

    def test_only_photographs_are_in_a_quality_slice(self):
        # F120: sharpness and open eyes mean nothing on a screenshot or a receipt.
        self.add_quality("shot.jpg", sharpness=10.0, verdict="screenshot")
        photo = self.add_quality("photo.jpg", sharpness=10.0)
        self.assertEqual(self.ids("blurred"), [photo])

    def test_a_photograph_the_quality_stage_never_touched_is_out(self):
        # The two slices that READ `frame_quality`. `low_resolution` is deliberately not
        # in this loop: it reads `files.width/height` instead, so a frame the stage never
        # reached belongs in it like any other — see TestLowResolutionSlice.
        self.add_classified("no_quality_row.jpg", "photo")
        for kind in ("blurred", "eyes_closed"):
            with self.subTest(kind=kind):
                self.assertEqual(self.ids(kind), [])

    def test_duplicates_and_unreadable_files_stay_out(self):
        canonical = self.add_quality("a.jpg", sharpness=10.0)
        duplicate = self.add_quality("b.jpg", sharpness=10.0)
        broken = self.add_quality("c.jpg", sharpness=10.0)
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?",
                          (canonical, duplicate))
        self.conn.execute("UPDATE files SET error = 'cannot read' WHERE id = ?", (broken,))
        self.conn.commit()
        self.assertEqual(self.ids("blurred"), [canonical])

    def test_where_still_narrows_the_slice(self):
        paris = self.add_quality("a.jpg", sharpness=10.0, country="France", city="Paris")
        self.add_quality("b.jpg", sharpness=10.0, country="Russia", city="Moskva")
        self.assertEqual(self.ids("blurred", where=["city=Paris"]), [paris])

    def test_the_default_name_comes_from_the_folder_catalog(self):
        self.add_quality("a.jpg", sharpness=10.0)
        self.assertEqual(self.gather("blurred").album_name,
                         i18n.folder("blurred", "en"))


class TestBlurIsAWindowNotAThreshold(SliceAlbumTestBase):
    """Requirement 3: the button collects what was SHOWN. The blurred list opens to
    `features.blur_review_max`, and an album that ignored it would gather thousands of
    frames nobody has looked at — while the whole point of this slice is that the
    decision is taken by eye. F157 turned that window into the depth of the first page of
    a ranking, which changes what the number means and not what the album does: "the
    first N in order" is an album, "everything below, for ever" is the collection."""

    def setUp(self):
        super().setUp()
        self.inside = self.add_quality("inside.jpg", sharpness=10.0)
        self.edge = self.add_quality("edge.jpg", sharpness=200.0)
        self.sharp = self.add_quality("sharp.jpg", sharpness=900.0)

    def test_the_album_stops_at_the_configured_window(self):
        self.assertEqual(self.cfg.features.blur_review_max, 300.0)
        self.assertEqual(sorted(self.ids("blurred")), sorted([self.inside, self.edge]))

    def test_raising_the_key_grows_the_album(self):
        self.cfg.features = dataclasses.replace(
            self.cfg.features, blur_review_max=1000.0)
        self.assertEqual(sorted(self.ids("blurred")),
                         sorted([self.inside, self.edge, self.sharp]))

    def test_lowering_the_key_empties_it(self):
        self.cfg.features = dataclasses.replace(self.cfg.features, blur_review_max=5.0)
        self.assertEqual(self.ids("blurred"), [])

    def test_the_window_bounds_no_other_slice(self):
        # A frame is in "closed eyes" because its eyes are closed, not because it is
        # also blurred — the ceiling belongs to one slice.
        eyes = self.add_quality("eyes.jpg", sharpness=900.0, eye_openness=CLOSED)
        self.assertEqual(self.ids("eyes_closed"), [eyes])


class TestLowResolutionSlice(SliceAlbumTestBase):
    """F150: the fifth slice of the "Review" workspace, gathered like the rest.

    What separates it from its neighbours is that nothing measured it. `blurred` and
    `eyes_closed` read a number some stage left in `frame_quality`; this one reads
    `files.width * files.height`, which the indexer wrote down while reading the file. So
    the cases below are not about accuracy — there is none to state — but about the two
    ways a fact can still be got wrong: reading it where it was never written (NULL) and
    reading it for a kind of file it does not mean the same thing for (a video).
    """

    def test_the_slice_holds_the_frames_under_the_threshold_and_only_them(self):
        small = self.add_sized("small.jpg", 640, 480)          # 0.31 Mp
        medium = self.add_sized("medium.jpg", 1024, 768)       # 0.79 Mp
        self.add_sized("big.jpg", 4000, 3000)                  # 12 Mp
        self.assertEqual(sorted(self.ids("low_resolution")), sorted([small, medium]))

    def test_the_configured_ceiling_changes_what_is_in_it(self):
        self.assertEqual(self.cfg.features.low_resolution_mp, 1.0)
        small = self.add_sized("small.jpg", 640, 480)
        two_mp = self.add_sized("two_mp.jpg", 1600, 1200)      # 1.92 Mp
        self.assertEqual(self.ids("low_resolution"), [small])
        self.cfg.features = dataclasses.replace(
            self.cfg.features, low_resolution_mp=3.0)
        self.assertEqual(sorted(self.ids("low_resolution")), sorted([small, two_mp]))
        self.cfg.features = dataclasses.replace(
            self.cfg.features, low_resolution_mp=0.1)
        self.assertEqual(self.ids("low_resolution"), [])

    def test_a_frame_exactly_at_the_ceiling_is_not_in_it(self):
        # `<`, not `<=` — the same boundary the blur window has.
        self.add_sized("exactly.jpg", 1000, 1000)              # 1.0 Mp
        self.assertEqual(self.ids("low_resolution"), [])

    def test_a_frame_without_dimensions_is_not_a_frame_of_zero_pixels(self):
        # The trap of this feature: NULL means "never learned", and a slice that read it
        # as 0 would put the least-known frames at the very top of the list.
        self.add_classified("unknown_size.jpg", "photo")
        self.assertEqual(self.ids("low_resolution"), [])

    def test_a_zero_dimension_is_not_a_small_frame_either(self):
        self.add_sized("broken_header.jpg", 0, 0)
        self.assertEqual(self.ids("low_resolution"), [])

    def test_a_photograph_the_quality_stage_never_touched_is_in(self):
        # The point of the slice: it needs no measurement, so it needs no `frame_quality`
        # row. `add_sized` writes none.
        small = self.add_sized("small.jpg", 640, 480)
        self.assertEqual(self.ids("low_resolution"), [small])

    def test_only_photographs_are_in_it(self):
        self.add_sized("icon.jpg", 64, 64, verdict="screenshot")
        self.add_sized("receipt.jpg", 300, 400, verdict="document")
        photo = self.add_sized("small.jpg", 640, 480)
        self.assertEqual(self.ids("low_resolution"), [photo])

    def test_a_video_is_not_in_it(self):
        # Videos have their own logic of resolution and their own meaning for it.
        video = self.add_sized("clip.mp4", 320, 240)
        self.conn.execute("UPDATE files SET media_type = 'video' WHERE id = ?", (video,))
        self.conn.commit()
        self.assertEqual(self.ids("low_resolution"), [])

    def test_duplicates_and_unreadable_files_stay_out(self):
        canonical = self.add_sized("a.jpg", 640, 480)
        duplicate = self.add_sized("b.jpg", 640, 480)
        broken = self.add_sized("c.jpg", 640, 480)
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?",
                          (canonical, duplicate))
        self.conn.execute("UPDATE files SET error = 'cannot read' WHERE id = ?", (broken,))
        self.conn.commit()
        self.assertEqual(self.ids("low_resolution"), [canonical])

    def test_the_default_name_comes_from_the_folder_catalog(self):
        self.add_sized("small.jpg", 640, 480)
        self.assertEqual(self.gather("low_resolution").album_name,
                         i18n.folder("low_resolution", "en"))

    def test_the_album_journals_under_its_own_mode(self):
        # The brief names this string: `move_batches.mode = 'album_low_resolution'`.
        self.add_sized("small.jpg", 640, 480, content=b"small")
        report = self.gather("low_resolution", mode="link", apply=True)
        batch = self.conn.execute(
            "SELECT mode, operation FROM move_batches WHERE id = ?",
            (report.batch_id,)).fetchone()
        self.assertEqual(batch["mode"], "album_low_resolution")
        self.assertEqual(batch["operation"], "link")
        self.assertTrue(
            (self.dest / i18n.folder("low_resolution", "en") / "small.jpg").exists())


class TestApplyAndJournal(SliceAlbumTestBase):
    def test_dry_run_writes_nothing_to_the_db_or_the_filesystem(self):
        self.add_classified("chair.jpg", "product")
        report = self.gather("product", mode="link", apply=False)
        self.assertEqual(len(report.plan), 1)
        self.assertFalse(self.dest.exists())
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0], 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_every_new_kind_journals_under_its_own_album_mode(self):
        self.add_classified("chair.jpg", "product", content=b"chair")
        self.add_classified("shot.jpg", "screenshot", content=b"shot")
        self.add_classified("meme.jpg", "meme", content=b"meme")
        self.add_quality("blurred.jpg", sharpness=10.0, content=b"blur")
        self.add_quality("eyes.jpg", eye_openness=CLOSED, content=b"eyes")
        self.add_sized("small.jpg", 640, 480, content=b"small")
        for kind in CLASS_ALBUM_KINDS + QUALITY_ALBUM_KINDS:
            with self.subTest(kind=kind):
                report = self.gather(kind, mode="link", apply=True)
                self.assertIsNotNone(report.batch_id)
                batch = self.conn.execute(
                    "SELECT mode, operation FROM move_batches WHERE id = ?",
                    (report.batch_id,)).fetchone()
                self.assertEqual(batch["mode"], f"album_{kind}")
                self.assertEqual(batch["operation"], "link")

    def gathered(self, kind: str, filename: str) -> Path:
        return self.dest / i18n.folder(ALBUM_FOLDER_KEYS[kind], "en") / filename

    def test_apply_link_journals_before_the_operation(self):
        file_id = self.add_classified("sub/deep/chair.jpg", "product", content=b"chair")
        report = self.gather("product", mode="link", apply=True)
        self.assertEqual(report.transferred, 1)
        dst = self.gathered("product", "chair.jpg")
        self.assertTrue(dst.exists())
        self.assertGreaterEqual(os.stat(dst).st_nlink, 2)  # a hardlink, not a copy
        move = self.conn.execute(
            "SELECT file_id, dst, status FROM moves WHERE batch_id = ?",
            (report.batch_id,)).fetchone()
        self.assertEqual(move["file_id"], file_id)
        self.assertEqual(move["status"], "done")
        self.assertEqual(Path(move["dst"]), dst)

    def test_apply_copy_leaves_the_canonical_original_in_place(self):
        file_id = self.add_classified("chair.jpg", "product", content=b"chair")
        report = self.gather("product", mode="copy", apply=True)
        self.assertEqual(report.transferred, 1)
        self.assertTrue(self.gathered("product", "chair.jpg").exists())
        self.assertEqual(self.path_of(file_id), str((self.src_dir / "chair.jpg").resolve()))

    def test_apply_move_takes_the_file_out_of_the_canon(self):
        file_id = self.add_classified("chair.jpg", "product", content=b"chair")
        report = self.gather("product", mode="move", apply=True)
        self.assertEqual(report.transferred, 1)
        dst = self.gathered("product", "chair.jpg")
        self.assertTrue(dst.exists())
        self.assertFalse((self.src_dir / "chair.jpg").exists())
        self.assertEqual(Path(self.path_of(file_id)), dst)

    def test_a_quality_slice_moves_too(self):
        self.add_quality("blurred.jpg", sharpness=10.0, content=b"blur")
        report = self.gather("blurred", mode="move", apply=True)
        self.assertEqual(report.transferred, 1)
        self.assertTrue(self.gathered("blurred", "blurred.jpg").exists())

    def test_gathering_the_same_album_twice_makes_no_underscore_one_copies(self):
        # F97, inherited: a file already sitting in the album folder byte-for-byte is
        # left alone instead of being re-materialized under a `_1` name.
        self.add_classified("chair.jpg", "product", content=b"chair")
        self.gather("product", mode="link", apply=True)
        second = self.gather("product", mode="link", apply=True)
        self.assertEqual(second.skipped_already_copied, 1)
        self.assertEqual(second.transferred, 0)
        album_dir = self.dest / i18n.folder("products", "en")
        self.assertEqual(sorted(p.name for p in album_dir.iterdir()), ["chair.jpg"])

    def test_empty_slice_does_not_crash_or_journal(self):
        report = self.gather("meme", apply=True)
        self.assertEqual(report.plan, [])
        self.assertIsNone(report.batch_id)
        self.assertFalse(self.dest.exists())


class TestMoveWarns(SliceAlbumTestBase):
    """Requirement "move only with the warning": these slices overlap each other and the
    duplicates, and "move" is easy to read as "take out of the collection". It is the
    same sentence `plan_album` has printed for `move` since F34, and it is printed on the
    dry run too — before anything has been carried anywhere."""

    def warning_of(self, kind: str, mode: str) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            plan_album(self.cfg, self.conn, kind, "", self.dest, mode=mode)
        return buffer.getvalue()

    def test_move_warns_for_every_new_kind(self):
        self.add_classified("chair.jpg", "product")
        self.add_classified("shot.jpg", "screenshot")
        self.add_classified("meme.jpg", "meme")
        self.add_quality("blurred.jpg", sharpness=10.0)
        self.add_quality("eyes.jpg", eye_openness=CLOSED)
        self.add_sized("small.jpg", 640, 480)
        expected = i18n.cli_text("cli.album.warn_move", "en")
        for kind in CLASS_ALBUM_KINDS + QUALITY_ALBUM_KINDS:
            with self.subTest(kind=kind):
                self.assertIn(expected, self.warning_of(kind, "move"))

    def test_link_and_copy_do_not_warn(self):
        self.add_quality("blurred.jpg", sharpness=10.0)
        expected = i18n.cli_text("cli.album.warn_move", "en")
        for mode in ("link", "copy"):
            with self.subTest(mode=mode):
                self.assertNotIn(expected, self.warning_of("blurred", mode))


if __name__ == "__main__":
    unittest.main()
