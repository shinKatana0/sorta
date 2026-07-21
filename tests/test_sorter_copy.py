"""C16: the --copy mode — copying into the new structure, originals in place.

Inherits the SorterTestBase fixtures from test_sorter.py. All FS operations — on
tmp_path only.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from tests.test_sorter import SorterTestBase

from sorta.hashing import file_hash
from sorta.sorter import TransferError, _transfer, plan_and_sort, undo


class TestTransferCopyMode(SorterTestBase):
    def test_copy_leaves_src_and_verifies_hash(self):
        src = self.write_file("a.jpg", b"hello")
        dst = self.dest / "a.jpg"
        _transfer(src, dst, copy=True)
        self.assertTrue(dst.exists())
        self.assertTrue(src.exists())
        self.assertEqual(dst.read_bytes(), b"hello")
        self.assertEqual(file_hash(dst)[0], file_hash(src)[0])

    def test_copy_never_overwrites_existing_dst(self):
        src = self.write_file("d.jpg", b"new")
        dst = self.dest / "d.jpg"
        dst.parent.mkdir(parents=True)
        dst.write_bytes(b"existing")
        with self.assertRaises(TransferError):
            _transfer(src, dst, copy=True)
        self.assertEqual(dst.read_bytes(), b"existing")
        self.assertTrue(src.exists())

    def test_copy_hash_mismatch_cleans_up_dst_and_keeps_src(self):
        src = self.write_file("c.jpg", b"content")
        dst = self.dest / "c.jpg"
        with patch("sorta.sorter.file_hash", return_value=("deadbeef", "blake3")):
            with self.assertRaises(TransferError):
                _transfer(src, dst, src_hash="cafebabe", copy=True)
        self.assertFalse(dst.exists())
        self.assertTrue(src.exists())


class TestApplyCopyMode(SorterTestBase):
    def test_apply_copy_keeps_src_and_files_path_unchanged(self):
        fid = self.add_file("img1.jpg", country="France", city="Paris")
        orig = Path(self.path_of(fid))
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=True, copy=True)
        new_path = self.dest / "France" / "Paris" / "2022" / "img1.jpg"
        self.assertTrue(new_path.exists())
        self.assertTrue(orig.exists())
        self.assertEqual(file_hash(new_path)[0], file_hash(orig)[0])
        self.assertEqual(self.path_of(fid), str(orig))  # files.path unchanged
        self.assertEqual(report.moved, 1)
        self.assertEqual(self.move_status(report.batch_id, fid), "done")

    def test_apply_copy_journals_operation_copy(self):
        self.add_file("img1.jpg", country="France", city="Paris")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=True, copy=True)
        batch = self.conn.execute(
            "SELECT operation FROM move_batches WHERE id = ?",
            (report.batch_id,)).fetchone()
        self.assertEqual(batch["operation"], "copy")

    def test_apply_move_still_journals_operation_move(self):
        self.add_file("img1.jpg", country="France", city="Paris")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        batch = self.conn.execute(
            "SELECT operation FROM move_batches WHERE id = ?",
            (report.batch_id,)).fetchone()
        self.assertEqual(batch["operation"], "move")

    def test_apply_copy_name_conflict_gets_suffix_and_keeps_both_srcs(self):
        self.add_file("a/img.jpg", content=b"one", country="RU", city="Moskva")
        self.add_file("b/img.jpg", content=b"two", country="RU", city="Moskva")
        plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        target_dir = self.dest / "Russia" / "Moskva" / "2022"
        self.assertTrue((target_dir / "img.jpg").exists())
        self.assertTrue((target_dir / "img_1.jpg").exists())
        contents = {(target_dir / "img.jpg").read_bytes(), (target_dir / "img_1.jpg").read_bytes()}
        self.assertEqual(contents, {b"one", b"two"})
        self.assertTrue((self.src_dir / "a" / "img.jpg").exists())
        self.assertTrue((self.src_dir / "b" / "img.jpg").exists())

    def test_apply_copy_does_not_overwrite_existing_dst_file(self):
        self.add_file("img.jpg", content=b"original", country="RU", city="Moskva")
        target_dir = self.dest / "Russia" / "Moskva" / "2022"
        target_dir.mkdir(parents=True)
        (target_dir / "img.jpg").write_bytes(b"pre-existing")
        plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        # conflict -> suffix, the pre-existing one is untouched
        self.assertEqual((target_dir / "img.jpg").read_bytes(), b"pre-existing")
        self.assertTrue((target_dir / "img_1.jpg").exists())

    def test_journal_row_committed_before_copy(self):
        fid = self.add_file("img1.jpg", country="France", city="Paris")
        seen_status_at_transfer_time = {}
        from sorta.sorter import _transfer as real_transfer

        def spy_transfer(src, dst, src_hash=None, copy=False):
            row = self.conn.execute(
                "SELECT status FROM moves WHERE file_id = ?", (fid,)).fetchone()
            seen_status_at_transfer_time["status"] = row["status"] if row else None
            return real_transfer(src, dst, src_hash, copy=copy)

        with patch("sorta.sorter._transfer", side_effect=spy_transfer):
            plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        self.assertEqual(seen_status_at_transfer_time["status"], "planned")


class TestUndoCopyBatch(SorterTestBase):
    def test_undo_copy_deletes_dst_and_keeps_src_and_path(self):
        fid = self.add_file("img1.jpg", content=b"aaa", country="France", city="Paris")
        orig = self.path_of(fid)
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        new_path = self.dest / "France" / "Paris" / "2022" / "img1.jpg"
        self.assertTrue(new_path.exists())

        stats = undo(self.conn, batch_id=report.batch_id)
        self.assertEqual(stats.undone, 1)
        self.assertFalse(new_path.exists())          # the copy was deleted
        self.assertTrue(Path(orig).exists())          # the original never moved
        self.assertEqual(self.path_of(fid), orig)     # files.path unchanged
        self.assertEqual(
            self.conn.execute(
                "SELECT status FROM moves WHERE batch_id = ? AND file_id = ?",
                (report.batch_id, fid)).fetchone()["status"],
            "undone")

    def test_undo_copy_missing_dst_logs_and_continues(self):
        fid = self.add_file("img1.jpg", country="France", city="Paris")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        new_path = self.dest / "France" / "Paris" / "2022" / "img1.jpg"
        new_path.unlink()

        stats = undo(self.conn, batch_id=report.batch_id)
        self.assertEqual(stats.missing, 1)
        self.assertEqual(stats.undone, 0)
        self.assertEqual(
            self.move_status(report.batch_id, fid), "done")  # status unchanged

    def test_undo_copy_hash_mismatch_keeps_dst(self):
        fid = self.add_file("img1.jpg", content=b"aaa", country="France", city="Paris")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        new_path = self.dest / "France" / "Paris" / "2022" / "img1.jpg"
        new_path.write_bytes(b"tampered-after-copy")

        stats = undo(self.conn, batch_id=report.batch_id)
        self.assertEqual(stats.failed, 1)
        self.assertEqual(stats.undone, 0)
        self.assertTrue(new_path.exists())  # the copy is NOT deleted on a hash mismatch
        self.assertEqual(self.move_status(report.batch_id, fid), "done")

    def test_move_batch_undo_unaffected_by_copy_support(self):
        fid1 = self.add_file("img1.jpg", content=b"aaa", country="France", city="Paris")
        orig1 = self.path_of(fid1)
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)  # move
        self.assertFalse(Path(orig1).exists())

        stats = undo(self.conn, batch_id=report.batch_id)
        self.assertEqual(stats.undone, 1)
        self.assertTrue(Path(orig1).exists())
        self.assertEqual(self.path_of(fid1), orig1)

    def test_undo_picks_last_batch_across_mixed_move_and_copy(self):
        fid_move = self.add_file("move.jpg", country="France", city="Paris")
        move_report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.add_file("copy.jpg", country="RU", city="Moskva")
        copy_report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                                    apply=True, copy=True)

        # the last batch (copy) is undone first when batch_id=None
        stats = undo(self.conn)
        self.assertEqual(stats.batch_id, copy_report.batch_id)
        copy_dst = self.dest / "Russia" / "Moskva" / "2022" / "copy.jpg"
        self.assertFalse(copy_dst.exists())

        stats2 = undo(self.conn)
        self.assertEqual(stats2.batch_id, move_report.batch_id)
        self.assertTrue(Path(self.path_of(fid_move)).exists())
