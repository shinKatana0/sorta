"""F139: the albums of the remaining slices — classes and quality.

`sorter.plan_album` grew six kinds: `product`/`screenshot`/`meme`, selected on
`media_class.verdict`, and `blurred`/`eyes_closed` (plus `no_subject`, retired by F177),
selected on `frame_quality` by the same rule that draws the "Review" workspace. The engine itself is
F34's and is not re-tested here; what is pinned below is what a new kind can get wrong:

* the slice it selects (and, for `blurred`, that it is a WINDOW and not a threshold —
  `features.blur_review_max`, the same key that bounds the list a person looks at);
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
        self.add_classified("no_quality_row.jpg", "photo")
        for kind in QUALITY_ALBUM_KINDS:
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
    decision is taken by eye."""

    def setUp(self):
        super().setUp()
        self.inside = self.add_quality("inside.jpg", sharpness=10.0)
        self.edge = self.add_quality("edge.jpg", sharpness=200.0)
        self.sharp = self.add_quality("sharp.jpg", sharpness=900.0)

    def test_the_album_stops_at_the_configured_window(self):
        self.assertEqual(self.cfg.features.blur_review_max, 90.0)
        self.assertEqual(self.ids("blurred"), [self.inside])

    def test_raising_the_key_grows_the_album(self):
        self.cfg.features = dataclasses.replace(
            self.cfg.features, blur_review_max=300.0)
        self.assertEqual(sorted(self.ids("blurred")), sorted([self.inside, self.edge]))

    def test_lowering_the_key_empties_it(self):
        self.cfg.features = dataclasses.replace(self.cfg.features, blur_review_max=5.0)
        self.assertEqual(self.ids("blurred"), [])

    def test_the_window_bounds_no_other_slice(self):
        # A frame is in "closed eyes" because its eyes are closed, not because it is
        # also blurred — the ceiling belongs to one slice.
        eyes = self.add_quality("eyes.jpg", sharpness=900.0, eye_openness=CLOSED)
        self.assertEqual(self.ids("eyes_closed"), [eyes])


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
