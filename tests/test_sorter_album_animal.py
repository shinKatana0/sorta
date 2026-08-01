"""F123: the animal album — `sorter.plan_album(kind='animal')`.

The slice is the whole feature: canonical, readable files whose `frame_quality.pet`
carries a verdict. There is nothing to select inside it, so the selector is accepted and
ignored; everything else (dry-run semantics, the journal-before-the-operation invariant,
`_resolve_dst` on a repeat gather) is inherited from F34/F97 and pinned here for the new
kind, because inheritance that is not checked is a plan, not a property.
"""
from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sorta import i18n
from sorta.sorter import plan_album

from tests.test_sorter import SorterTestBase


class AnimalAlbumTestBase(SorterTestBase):
    def mark_animal(self, file_id: int, *, score: float = 0.9,
                    pet: str | None = "animal", source: str = "clip") -> None:
        """A `frame_quality` row as the junk stage writes it (F113/F122).

        `pet=None` with a score is the frame that was ASKED about and did not clear
        `features.pet_threshold` — the case the slice has to leave out.
        """
        self.conn.execute(
            """INSERT INTO frame_quality (file_id, sharpness, pet, pet_score, source,
                   updated_at)
               VALUES (?, 100.0, ?, ?, ?, '2026-01-01')""",
            (file_id, pet, score, source))
        self.conn.commit()

    def gather(self, **kwargs):
        """plan_album for the animal kind, with the CLI chatter swallowed."""
        with redirect_stdout(io.StringIO()):
            return plan_album(self.cfg, self.conn, "animal", "", self.dest, **kwargs)


class TestAnimalAlbumSelection(AnimalAlbumTestBase):
    def test_only_frames_with_a_pet_verdict_are_in_the_slice(self):
        marked = self.add_file("cat.jpg")
        self.mark_animal(marked)
        below = self.add_file("coat.jpg")
        self.mark_animal(below, pet=None, score=0.4)
        self.add_file("never_asked.jpg")
        report = self.gather(apply=False)
        self.assertEqual([it.file_id for it in report.plan], [marked])

    def test_duplicates_and_unreadable_files_stay_out(self):
        canonical = self.add_file("a.jpg")
        duplicate = self.add_file("b.jpg")
        broken = self.add_file("c.jpg")
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?", (canonical, duplicate))
        self.conn.execute("UPDATE files SET error = 'cannot read' WHERE id = ?", (broken,))
        self.conn.commit()
        for fid in (canonical, duplicate, broken):
            self.mark_animal(fid)
        report = self.gather(apply=False)
        self.assertEqual([it.file_id for it in report.plan], [canonical])

    def test_the_selector_is_ignored_rather_than_used(self):
        fid = self.add_file("cat.jpg")
        self.mark_animal(fid)
        with redirect_stdout(io.StringIO()):
            empty = plan_album(self.cfg, self.conn, "animal", "", self.dest)
            noise = plan_album(self.cfg, self.conn, "animal", "whatever", self.dest)
        self.assertEqual([it.file_id for it in empty.plan],
                         [it.file_id for it in noise.plan])

    def test_where_still_narrows_the_slice(self):
        paris = self.add_file("a.jpg", country="France", city="Paris")
        moscow = self.add_file("b.jpg", country="Russia", city="Moskva")
        self.mark_animal(paris)
        self.mark_animal(moscow)
        report = self.gather(where=["city=Paris"], apply=False)
        self.assertEqual([it.file_id for it in report.plan], [paris])

    def test_default_album_name_comes_from_the_folder_catalog(self):
        fid = self.add_file("cat.jpg")
        self.mark_animal(fid)
        report = self.gather(apply=False)
        self.assertEqual(report.album_name, i18n.folder("animals", "en"))

    def test_default_album_name_follows_the_configured_language(self):
        self.cfg.raw = {"language": "ru"}
        fid = self.add_file("cat.jpg")
        self.mark_animal(fid)
        report = self.gather(apply=False)
        self.assertEqual(report.album_name, i18n.folder("animals", "ru"))

    def test_an_explicit_name_still_wins(self):
        fid = self.add_file("cat.jpg")
        self.mark_animal(fid)
        report = self.gather(album_name="Barsik", apply=False)
        self.assertEqual(report.album_name, "Barsik")
        self.assertEqual(Path(report.dest).name, "Barsik")

    def test_empty_slice_does_not_crash_or_journal(self):
        report = self.gather(apply=True)
        self.assertEqual(report.plan, [])
        self.assertIsNone(report.batch_id)
        self.assertFalse(self.dest.exists())


class TestAnimalAlbumApply(AnimalAlbumTestBase):
    def test_dry_run_writes_nothing_to_the_db_or_the_filesystem(self):
        fid = self.add_file("cat.jpg")
        self.mark_animal(fid)
        report = self.gather(mode="link", apply=False)
        self.assertEqual(len(report.plan), 1)
        self.assertFalse(self.dest.exists())
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0], 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_apply_link_journals_before_the_operation_with_the_album_mode(self):
        fid = self.add_file("sub/deep/cat.jpg", content=b"meow")
        self.mark_animal(fid)
        report = self.gather(mode="link", apply=True)
        self.assertEqual(report.transferred, 1)
        dst = self.dest / i18n.folder("animals", "en") / "cat.jpg"
        self.assertTrue(dst.exists())
        self.assertGreaterEqual(os.stat(dst).st_nlink, 2)  # a hardlink, not a copy
        batch = self.conn.execute(
            "SELECT mode, operation, dest_root FROM move_batches WHERE id = ?",
            (report.batch_id,)).fetchone()
        self.assertEqual(batch["mode"], "album_animal")
        self.assertEqual(batch["operation"], "link")
        self.assertEqual(batch["dest_root"], str(self.dest.resolve()))
        move = self.conn.execute(
            "SELECT file_id, dst, status FROM moves WHERE batch_id = ?",
            (report.batch_id,)).fetchone()
        self.assertEqual(move["file_id"], fid)
        self.assertEqual(move["status"], "done")
        self.assertEqual(Path(move["dst"]), dst)

    def test_apply_copy_leaves_the_canonical_original_in_place(self):
        fid = self.add_file("cat.jpg", content=b"meow")
        self.mark_animal(fid)
        report = self.gather(mode="copy", apply=True)
        self.assertEqual(report.transferred, 1)
        self.assertTrue(Path(self.path_of(fid)).exists())
        self.assertEqual(self.path_of(fid), str((self.src_dir / "cat.jpg").resolve()))

    def test_apply_move_takes_the_file_out_of_the_canon_and_updates_the_path(self):
        fid = self.add_file("cat.jpg", content=b"meow")
        self.mark_animal(fid)
        report = self.gather(mode="move", apply=True)
        self.assertEqual(report.transferred, 1)
        dst = self.dest / i18n.folder("animals", "en") / "cat.jpg"
        self.assertTrue(dst.exists())
        self.assertFalse((self.src_dir / "cat.jpg").exists())
        self.assertEqual(Path(self.path_of(fid)), dst)

    def test_move_warns_that_the_file_leaves_the_layout(self):
        fid = self.add_file("cat.jpg")
        self.mark_animal(fid)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            plan_album(self.cfg, self.conn, "animal", "", self.dest, mode="move")
        self.assertIn(i18n.cli_text("cli.album.warn_move", "en"), buffer.getvalue())

    def test_gathering_the_same_album_twice_makes_no_underscore_one_copies(self):
        # F97, inherited: a file already sitting in the album folder byte-for-byte is
        # left alone instead of being re-materialized under a `_1` name.
        fid = self.add_file("cat.jpg", content=b"meow")
        self.mark_animal(fid)
        self.gather(mode="link", apply=True)
        second = self.gather(mode="link", apply=True)
        self.assertEqual(second.skipped_already_copied, 1)
        self.assertEqual(second.transferred, 0)
        album_dir = self.dest / i18n.folder("animals", "en")
        self.assertEqual(sorted(p.name for p in album_dir.iterdir()), ["cat.jpg"])


if __name__ == "__main__":
    unittest.main()
