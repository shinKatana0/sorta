"""F124: the user's own verdict on an animal mark — the table and the read rule.

The point of the feature is WHERE the correction lives and WHEN it is applied. It lives
in `manual_pet` and not in `frame_quality`, because that table has exactly one writer
(`junk`) and every run recomputes it from scratch — F120 even invalidates rows by a
fingerprint of the prompts — so a mark written there would last until the next run. And
it is applied WHEN READ (`sorter.ANIMAL_IDS_SQL`), not when written, which is what makes
it survive a change of model, of prompts or of the threshold.

Hence the shape of this file: the migration, then the slice under every combination of
manual and automatic verdict, then the case the whole design exists for — a full
recompute of `frame_quality` with a different prompt fingerprint, after which the mark
has to be exactly where the user left it.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from sorta.db import connect, reset_index
from sorta.junk import _QUALITY_UPSERT, quality_prompt_fingerprint

from tests.test_sorter_album_animal import AnimalAlbumTestBase


class TestManualPetMigration(unittest.TestCase):
    """Brief test 1: the table appears, the version moves, a repeat run changes nothing.

    Every connection is closed inside its temp directory: on Windows an open sqlite
    handle makes the rmtree fail (the reason test_frame_quality explains at its fixture).
    """

    def test_fresh_db_has_the_table_and_the_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "fresh.db")
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(manual_pet)")}
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            conn.close()
        self.assertEqual(cols, {"file_id", "is_animal", "updated_at"})
        self.assertEqual(version, 20)

    def test_v16_db_gains_the_table_without_touching_its_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "v16.db"
            conn = connect(db)
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
            conn.execute("DROP TABLE manual_pet")
            # A real v16 database predates F130 as well, so it has no `pet_vlm` either.
            # Leaving the column in place would make the v17 migration ADD one that
            # already exists and raise: a simulated old DB has to be old in EVERY
            # respect, not only in the one this feature cares about. (This case is the
            # first to hit it because F130 and F124 were written in parallel against the
            # same v16 main — neither could see the other's column.)
            conn.execute("ALTER TABLE frame_quality DROP COLUMN pet_vlm")
            conn.execute("ALTER TABLE frame_quality DROP COLUMN junk_score")  # F140, v20
            conn.execute("PRAGMA user_version = 16")
            conn.commit()
            conn.close()

            conn = connect(db)
            tables = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            conn.close()
        self.assertIn("manual_pet", tables)
        self.assertEqual(version, 20)
        self.assertEqual(files, 1)

    def test_reopening_is_idempotent_and_keeps_the_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "twice.db"
            conn = connect(db)
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
            conn.execute("INSERT INTO manual_pet (file_id, is_animal, updated_at) "
                         "VALUES (1, 0, 'x')")
            conn.commit()
            conn.close()

            conn = connect(db)  # the migration runs again on the already-migrated DB
            row = conn.execute("SELECT * FROM manual_pet").fetchone()
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            conn.close()
        self.assertEqual((row["file_id"], row["is_animal"]), (1, 0))
        self.assertEqual(version, 20)

    def test_one_row_per_file(self):
        """The PK is what makes "the user's verdict" singular — a second mark on the
        same frame replaces the first instead of adding a second opinion."""
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "pk.db")
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
            conn.execute("INSERT INTO manual_pet (file_id, is_animal, updated_at) "
                         "VALUES (1, 0, 'x')")
            conn.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO manual_pet (file_id, is_animal, updated_at) "
                             "VALUES (1, 1, 'y')")
            conn.close()


class TestManualPetReset(unittest.TestCase):
    """Brief test 7: a from-scratch reindex starts from a clean slate."""

    def test_reset_index_wipes_the_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "r.db")
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
            conn.execute("INSERT INTO manual_pet (file_id, is_animal, updated_at) "
                         "VALUES (1, 0, 'x')")
            conn.commit()
            reset_index(conn)
            left = conn.execute("SELECT COUNT(*) FROM manual_pet").fetchone()[0]
            conn.close()
        self.assertEqual(left, 0)


class ManualPetTestBase(AnimalAlbumTestBase):
    """The F123 album fixture plus the two things this feature adds to it."""

    def mark_by_hand(self, file_id: int, is_animal: bool) -> None:
        """A `manual_pet` row exactly as the web app writes it."""
        self.conn.execute(
            """INSERT INTO manual_pet (file_id, is_animal, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(file_id) DO UPDATE SET
                   is_animal = excluded.is_animal, updated_at = excluded.updated_at""",
            (file_id, 1 if is_animal else 0, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def rerun_junk_quality(self, rows: dict[int, tuple[str | None, float]]) -> None:
        """What a fresh `junk` run does to `frame_quality`, prompt fingerprint included.

        The statement is junk's own upsert and the source marker is built by junk's own
        `quality_prompt_fingerprint`, so this is the real invalidation and not a
        paraphrase of it: after F120 a run whose prompts differ rewrites every row it
        owns. `manual_pet` is not junk's, and that is the whole feature.
        """
        source = f"clip#{quality_prompt_fingerprint(True, with_vlm=True)}"
        now = "2026-08-02"
        with self.conn:
            self.conn.execute("DELETE FROM frame_quality")
            for file_id, (pet, score) in rows.items():
                self.conn.execute(_QUALITY_UPSERT,
                                  (file_id, 100.0, pet, score, source, now))

    def slice_ids(self) -> list[int]:
        return [it.file_id for it in self.gather(apply=False).plan]


class TestManualMarkChangesTheSlice(ManualPetTestBase):
    def test_not_an_animal_takes_the_frame_out(self):
        """Brief test 2 (the album half): the mark outranks the model."""
        cat = self.add_file("cat.jpg")
        coat = self.add_file("fur_coat.jpg")
        self.mark_animal(cat)
        self.mark_animal(coat)
        self.assertEqual(self.slice_ids(), [cat, coat])
        self.mark_by_hand(coat, is_animal=False)
        self.assertEqual(self.slice_ids(), [cat])

    def test_it_is_an_animal_puts_back_a_frame_below_the_threshold(self):
        """Brief test 3: `frame_quality.pet IS NULL` and the frame is in the slice."""
        missed = self.add_file("dark_cat.jpg")
        self.mark_animal(missed, pet=None, score=0.61)   # asked, did not clear 0.70
        self.assertEqual(self.slice_ids(), [])
        self.mark_by_hand(missed, is_animal=True)
        self.assertEqual(self.slice_ids(), [missed])

    def test_a_frame_the_stage_never_touched_can_still_be_added(self):
        """No `frame_quality` row at all — the LEFT JOIN half of the rule. A run with
        `features.pets` off leaves the whole collection like this."""
        never_asked = self.add_file("never_asked.jpg")
        self.mark_by_hand(never_asked, is_animal=True)
        self.assertEqual(self.slice_ids(), [never_asked])

    def test_clearing_the_mark_returns_the_automatic_answer(self):
        """Brief test 5: `clear` is not `not_animal` — it hands the frame back."""
        cat = self.add_file("cat.jpg")
        self.mark_animal(cat)
        self.mark_by_hand(cat, is_animal=False)
        self.assertEqual(self.slice_ids(), [])
        self.conn.execute("DELETE FROM manual_pet WHERE file_id = ?", (cat,))
        self.conn.commit()
        self.assertEqual(self.slice_ids(), [cat])

    def test_a_file_without_a_row_behaves_exactly_as_before(self):
        """Brief test 8: the F123 selection, unchanged, for everybody else."""
        marked = self.add_file("cat.jpg")
        self.mark_animal(marked)
        below = self.add_file("coat.jpg")
        self.mark_animal(below, pet=None, score=0.4)
        self.add_file("never_asked.jpg")
        other = self.add_file("dog.jpg")
        self.mark_animal(other)
        self.mark_by_hand(other, is_animal=False)   # somebody else's frame, not theirs
        self.assertEqual(self.slice_ids(), [marked])

    def test_the_mark_leaves_the_model_table_alone(self):
        """The boundary this feature is built on: `junk` keeps its single writership."""
        cat = self.add_file("cat.jpg")
        self.mark_animal(cat, score=0.88)
        before = dict(self.conn.execute(
            "SELECT pet, pet_score FROM frame_quality WHERE file_id = ?", (cat,)
        ).fetchone())
        self.mark_by_hand(cat, is_animal=False)
        after = dict(self.conn.execute(
            "SELECT pet, pet_score FROM frame_quality WHERE file_id = ?", (cat,)
        ).fetchone())
        self.assertEqual(before, after)
        self.assertEqual(before["pet"], "animal")

    def test_duplicates_and_unreadable_files_stay_out_however_they_are_marked(self):
        """A manual mark selects a frame; it does not promote a duplicate or a file
        that does not open — those rules are older and are not this feature's to bend."""
        canonical = self.add_file("a.jpg")
        duplicate = self.add_file("b.jpg")
        broken = self.add_file("c.jpg")
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?",
                          (canonical, duplicate))
        self.conn.execute("UPDATE files SET error = 'cannot read' WHERE id = ?", (broken,))
        self.conn.commit()
        for fid in (canonical, duplicate, broken):
            self.mark_by_hand(fid, is_animal=True)
        self.assertEqual(self.slice_ids(), [canonical])


class TestTheMarkSurvivesARecompute(ManualPetTestBase):
    """Brief test 4 — the main test of the feature.

    `junk` recomputes `frame_quality` from scratch, and after F120 a prompt edit
    invalidates every row it owns. A correction stored in that table would be gone; a
    correction read on top of it is not, and neither direction of it is.
    """

    def test_an_unmarked_frame_does_not_come_back_marked(self):
        cat = self.add_file("cat.jpg")
        coat = self.add_file("fur_coat.jpg")
        self.mark_animal(cat, score=0.95)
        self.mark_animal(coat, score=0.72)
        self.mark_by_hand(coat, is_animal=False)
        self.assertEqual(self.slice_ids(), [cat])

        # a full re-run with different prompts: every row rewritten, the model still
        # just as sure the fur coat is a cat
        self.rerun_junk_quality({cat: ("animal", 0.96), coat: ("animal", 0.74)})

        self.assertEqual(self.slice_ids(), [cat])
        self.assertEqual(
            self.conn.execute("SELECT is_animal FROM manual_pet WHERE file_id = ?",
                              (coat,)).fetchone()["is_animal"], 0)

    def test_an_added_frame_stays_added(self):
        missed = self.add_file("dark_cat.jpg")
        self.mark_animal(missed, pet=None, score=0.61)
        self.mark_by_hand(missed, is_animal=True)
        # the new prompts are no better at this frame — and score-wise, worse
        self.rerun_junk_quality({missed: (None, 0.55)})
        self.assertEqual(self.slice_ids(), [missed])

    def test_it_survives_the_row_disappearing_entirely(self):
        """The harshest recompute there is: F120 purges rows of frames the classifier
        has stopped calling photographs, so the automatic verdict does not merely change
        — it ceases to exist. The user's answer is still their answer."""
        missed = self.add_file("dark_cat.jpg")
        self.mark_animal(missed, pet=None, score=0.61)
        self.mark_by_hand(missed, is_animal=True)
        with self.conn:
            self.conn.execute("DELETE FROM frame_quality")
        self.assertEqual(self.slice_ids(), [missed])

    def test_a_threshold_change_does_not_move_the_marked_frames(self):
        """Re-running with a lower threshold marks more frames automatically — and
        leaves both hand-made decisions exactly as they were."""
        cat = self.add_file("cat.jpg")
        coat = self.add_file("fur_coat.jpg")
        missed = self.add_file("dark_cat.jpg")
        self.mark_animal(cat, score=0.95)
        self.mark_animal(coat, score=0.72)
        self.mark_animal(missed, pet=None, score=0.61)
        self.mark_by_hand(coat, is_animal=False)
        self.mark_by_hand(missed, is_animal=True)
        # threshold 0.70 -> 0.50: the coat clears it again, the dark cat now clears it too
        self.rerun_junk_quality({cat: ("animal", 0.95), coat: ("animal", 0.72),
                                 missed: ("animal", 0.61)})
        self.assertEqual(self.slice_ids(), [cat, missed])


if __name__ == "__main__":
    unittest.main()
