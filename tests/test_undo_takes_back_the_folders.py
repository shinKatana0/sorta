"""F236: undo removes the directories the sort itself created, and only those.

The rule is a record and never a resemblance: a level the sort made is journaled before
the mkdir, and the rollback removes it only if it is still empty. All FS operations — on
the temporary root of SorterTestBase.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from tests.test_sorter import SorterTestBase

from sorta.sorter import _levels_to_create, plan_and_sort, undo


class DirJournalCase(SorterTestBase):
    @property
    def journal(self) -> Path:
        return self.root / "moves_dirs.jsonl"

    def journaled(self, batch_id: int) -> list[str]:
        if not self.journal.exists():
            return []
        return [json.loads(line)["dir"]
                for line in self.journal.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line)["batch_id"] == batch_id]

    def sort_one(self, rel: str = "img1.jpg", content: bytes = b"data",
                 copy: bool = False) -> tuple[int, int]:
        """One photo laid out into dest/France/Paris/2022 -> (file_id, batch_id)."""
        file_id = self.add_file(rel, content=content, country="France", city="Paris")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True,
                               copy=copy)
        assert report.batch_id is not None
        return file_id, report.batch_id


class TestTheSortWritesDownWhatItCreated(DirJournalCase):
    def test_every_level_of_the_chain_is_journaled(self):
        _fid, batch_id = self.sort_one()
        self.assertEqual(
            self.journaled(batch_id),
            [str(self.dest), str(self.dest / "France"),
             str(self.dest / "France" / "Paris"),
             str(self.dest / "France" / "Paris" / "2022")])

    def test_a_directory_that_already_existed_is_not_journaled(self):
        (self.dest / "France").mkdir(parents=True)
        _fid, batch_id = self.sort_one()
        self.assertNotIn(str(self.dest), self.journaled(batch_id))
        self.assertNotIn(str(self.dest / "France"), self.journaled(batch_id))
        self.assertIn(str(self.dest / "France" / "Paris"), self.journaled(batch_id))

    def test_the_record_is_written_before_the_directory_exists(self):
        seen: dict[str, object] = {}

        def spy(src, dst, src_hash=None, copy=False):
            seen["journaled"] = json.loads(
                self.journal.read_text(encoding="utf-8").splitlines()[-1])["dir"]
            seen["existed"] = dst.parent.exists()
            raise AssertionError("stop before the move")

        with patch("sorta.sorter._transfer", side_effect=spy):
            self.add_file("img1.jpg", country="France", city="Paris")
            with self.assertRaises(AssertionError):
                plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.assertEqual(seen["journaled"], str(self.dest / "France" / "Paris" / "2022"))
        self.assertFalse(seen["existed"])

    def test_a_second_batch_records_only_what_it_had_to_make(self):
        _fid, first = self.sort_one("a.jpg")
        _fid2, second = self.sort_one("b.jpg", content=b"other")
        self.assertNotEqual(first, second)
        self.assertEqual(self.journaled(second), [])
        self.assertEqual(len(self.journaled(first)), 4)


class TestUndoTakesTheFoldersBack(DirJournalCase):
    def test_an_empty_folder_we_created_is_removed(self):
        file_id, _batch = self.sort_one()
        stats = undo(self.conn)
        self.assertFalse((self.dest / "France").exists())
        self.assertFalse(self.dest.exists())
        self.assertEqual(stats.undone, 1)
        self.assertTrue(Path(self.path_of(file_id)).is_file())

    def test_nested_levels_go_bottom_up_and_leave_nothing_empty(self):
        self.add_file("a.jpg", country="France", city="Paris")
        self.add_file("b.jpg", content=b"two", country="Russia", city="Moskva")
        plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        stats = undo(self.conn)
        self.assertEqual(stats.dirs_removed, 7)  # dest + 2 x (country/city/year)
        self.assertFalse(self.dest.exists())

    def test_a_folder_somebody_wrote_into_stays_and_keeps_the_file(self):
        _fid, _batch = self.sort_one()
        target = self.dest / "France" / "Paris" / "2022"
        mine = target / "notes.txt"
        mine.write_text("mine", encoding="utf-8")
        stats = undo(self.conn)
        self.assertTrue(target.is_dir())
        self.assertEqual(mine.read_text(encoding="utf-8"), "mine")
        self.assertEqual(stats.dirs_removed, 0)  # nor its parents, which hold it

    def test_a_folder_that_existed_before_the_sort_is_not_touched(self):
        kept = self.dest / "France"
        kept.mkdir(parents=True)
        _fid, _batch = self.sort_one()
        stats = undo(self.conn)
        self.assertTrue(kept.is_dir())
        self.assertTrue(self.dest.is_dir())
        self.assertFalse((kept / "Paris").exists())
        self.assertEqual(stats.dirs_removed, 2)

    def test_the_photographs_come_back_whole(self):
        file_id, _batch = self.sort_one(content=b"the photo")
        undo(self.conn)
        restored = Path(self.path_of(file_id))
        self.assertEqual(restored.read_bytes(), b"the photo")
        self.assertEqual(restored.parent, self.src_dir)

    def test_a_copy_batch_takes_its_folders_back_too(self):
        file_id, _batch = self.sort_one(copy=True)
        stats = undo(self.conn)
        self.assertEqual(stats.dirs_removed, 4)
        self.assertFalse(self.dest.exists())
        self.assertTrue(Path(self.path_of(file_id)).is_file())

    def test_a_second_undo_of_the_same_batch_removes_nothing_more(self):
        _fid, batch_id = self.sort_one()
        self.assertEqual(undo(self.conn, batch_id).dirs_removed, 4)
        self.assertEqual(undo(self.conn, batch_id).dirs_removed, 0)


class TestAJournalWithoutTheNewRecords(DirJournalCase):
    """Written before F236: the rollback works as it did and deletes no directory."""

    def test_undo_restores_the_files_and_removes_nothing(self):
        file_id, _batch = self.sort_one()
        self.journal.unlink()
        stats = undo(self.conn)
        self.assertEqual((stats.undone, stats.dirs_removed), (1, 0))
        self.assertTrue((self.dest / "France" / "Paris" / "2022").is_dir())
        self.assertTrue(Path(self.path_of(file_id)).is_file())

    def test_a_journal_of_other_batches_only_leaves_this_one_alone(self):
        _fid, batch_id = self.sort_one()
        self.journal.write_text(
            json.dumps({"batch_id": batch_id + 100, "dir": str(self.dest)}) + "\n",
            encoding="utf-8")
        self.assertEqual(undo(self.conn).dirs_removed, 0)
        self.assertTrue(self.dest.is_dir())

    def test_a_half_written_line_does_not_stop_the_rest(self):
        _fid, batch_id = self.sort_one()
        with open(self.journal, "a", encoding="utf-8") as fh:
            fh.write('{"batch_id": ' + str(batch_id))
        self.assertEqual(undo(self.conn).dirs_removed, 4)


class TestTheLevelsToCreate(DirJournalCase):
    def test_an_existing_directory_needs_nothing(self):
        self.assertEqual(_levels_to_create(self.src_dir), [])

    def test_the_chain_comes_back_outermost_first(self):
        self.assertEqual(_levels_to_create(self.root / "a" / "b" / "c"),
                         [self.root / "a", self.root / "a" / "b",
                          self.root / "a" / "b" / "c"])
