"""F97: cancelling an apply, resuming it without duplicates, Windows long paths and
the rollback of an interrupted run.

Inherits the SorterTestBase fixtures from test_sorter.py. All FS operations — on
tmp_path only.

The three parts share one story: the first real `sort --apply` on the live collection
(22 364 files, ~220 GB, copy mode) died with the machine, and everything that went
wrong afterwards was operational, not data loss. It could not be stopped from the UI,
the interrupted batch could not be rolled back, and — measured — a second apply into
the same dest re-copied 10 021 files and 140.9 GB instead of skipping them.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from tests.test_sorter import SorterTestBase

from sorta.hashing import file_hash
from sorta.sorter import _fs, plan_and_sort, undo


def _cancel_after(n: int):
    """A should_cancel that lets the first `n` iterations through, then stops.

    Counts its own calls, because the engine polls it exactly once per plan item at
    the start of the iteration — see plan_and_sort/undo.
    """
    state = {"calls": 0}

    def should_cancel() -> bool:
        state["calls"] += 1
        return state["calls"] > n

    return should_cancel


class TestSortCancel(SorterTestBase):
    """Part 1: an apply must be stoppable, and the batch must survive the stop."""

    def _add_three(self) -> None:
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            self.add_file(name, content=name.encode(), country="RU", city="Moskva")

    def batch_finished_at(self, batch_id: int) -> str | None:
        return self.conn.execute(
            "SELECT finished_at FROM move_batches WHERE id = ?",
            (batch_id,)).fetchone()["finished_at"]

    def test_cancel_midway_closes_batch_and_reports_partial_progress(self):
        self._add_three()
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True,
                               copy=True, should_cancel=_cancel_after(2))
        self.assertTrue(report.cancelled)
        self.assertEqual(report.moved, 2)
        # the batch is CLOSED — undo is the tool the user reaches for next, and it
        # must not see a batch that looks like it is still running
        self.assertIsNotNone(self.batch_finished_at(report.batch_id))
        copied = sorted(p.name for p in self.dest.rglob("*.jpg"))
        self.assertEqual(len(copied), 2)
        done = self.conn.execute(
            "SELECT COUNT(*) AS n FROM moves WHERE batch_id = ? AND status = 'done'",
            (report.batch_id,)).fetchone()["n"]
        self.assertEqual(done, 2)

    def test_cancel_before_first_file_closes_batch_without_error(self):
        self._add_three()
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True,
                               copy=True, should_cancel=_cancel_after(0))
        self.assertTrue(report.cancelled)
        self.assertEqual(report.moved, 0)
        self.assertEqual(report.failed, 0)
        self.assertIsNotNone(self.batch_finished_at(report.batch_id))
        self.assertEqual(list(self.dest.rglob("*.jpg")), [])

    def test_undo_after_cancel_rolls_back_exactly_what_was_transferred(self):
        self._add_three()
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True,
                               copy=True, should_cancel=_cancel_after(2))
        stats = undo(self.conn, batch_id=report.batch_id)
        self.assertEqual(stats.undone, 2)
        self.assertEqual(stats.failed, 0)
        self.assertEqual(stats.stray, [])
        self.assertEqual(list(self.dest.rglob("*.jpg")), [])
        # copy mode: the originals never moved
        self.assertEqual(len(list(self.src_dir.rglob("*.jpg"))), 3)

    def test_without_should_cancel_behaviour_is_unchanged(self):
        self._add_three()
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True,
                               copy=True)
        self.assertFalse(report.cancelled)
        self.assertEqual(report.moved, 3)
        self.assertIsNotNone(self.batch_finished_at(report.batch_id))

    def test_move_mode_cancel_leaves_the_rest_of_the_sources_in_place(self):
        self._add_three()
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True,
                               should_cancel=_cancel_after(1))
        self.assertTrue(report.cancelled)
        self.assertEqual(report.moved, 1)
        self.assertEqual(len(list(self.src_dir.rglob("*.jpg"))), 2)


class TestRepeatedApplyDoesNotDuplicate(SorterTestBase):
    """Part 4: the main hole — a second apply into the same dest used to duplicate
    everything the first one had already copied."""

    def _add(self, *names: str) -> None:
        for name in names:
            self.add_file(name, content=name.encode(), country="RU", city="Moskva")

    @property
    def target_dir(self) -> Path:
        return self.dest / "Russia" / "Moskva" / "2022"

    def test_second_apply_into_same_dest_copies_nothing(self):
        self._add("a.jpg", "b.jpg", "c.jpg")
        first = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        self.assertEqual(first.moved, 3)
        before = sorted(p.name for p in self.dest.rglob("*.jpg"))

        second = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        self.assertEqual(second.moved, 0)
        self.assertEqual(second.failed, 0)
        self.assertEqual(second.skipped_already_copied, 3)
        # skipped_in_place is a DIFFERENT event and must not absorb this one
        self.assertEqual(second.skipped_in_place, 0)
        self.assertEqual(sorted(p.name for p in self.dest.rglob("*.jpg")), before)
        self.assertEqual([p for p in self.dest.rglob("*_1.jpg")], [])

    def test_dry_run_plan_predicts_the_same_skip(self):
        # the decision is made while the plan is built, so the dry-run has to show it
        self._add("a.jpg")
        plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        plan = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False).plan
        self.assertEqual(len(plan), 1)
        self.assertTrue(plan[0].already_copied)
        self.assertEqual(plan[0].dst, self.target_dir / "a.jpg")

    def test_resume_after_cancel_copies_only_what_is_missing(self):
        self._add("a.jpg", "b.jpg", "c.jpg")
        first = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True,
                              copy=True, should_cancel=_cancel_after(2))
        self.assertEqual(first.moved, 2)

        second = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        self.assertEqual(second.moved, 1)
        self.assertEqual(second.skipped_already_copied, 2)
        names = sorted(p.name for p in self.dest.rglob("*.jpg"))
        self.assertEqual(names, ["a.jpg", "b.jpg", "c.jpg"])  # no _1 twins

    def test_different_file_with_the_same_name_still_gets_a_suffix(self):
        # the protection against data loss must not be weakened by the skip above
        self._add("a.jpg")
        self.target_dir.mkdir(parents=True)
        (self.target_dir / "a.jpg").write_bytes(b"a completely different picture")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        self.assertEqual(report.moved, 1)
        self.assertEqual(report.skipped_already_copied, 0)
        self.assertEqual((self.target_dir / "a.jpg").read_bytes(),
                         b"a completely different picture")
        self.assertEqual((self.target_dir / "a_1.jpg").read_bytes(), b"a.jpg")

    def test_truncated_copy_of_the_same_size_counts_as_a_different_file(self):
        # size matches, hash does not: a copy interrupted mid-write looks exactly like
        # this, and treating it as "already there" would leave a broken file in place
        # of a good one
        self.add_file("a.jpg", content=b"good-content", country="RU", city="Moskva")
        self.target_dir.mkdir(parents=True)
        (self.target_dir / "a.jpg").write_bytes(b"BAD!-content")  # same length
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        self.assertEqual(report.skipped_already_copied, 0)
        self.assertEqual(report.moved, 1)
        self.assertEqual((self.target_dir / "a.jpg").read_bytes(), b"BAD!-content")
        self.assertEqual((self.target_dir / "a_1.jpg").read_bytes(), b"good-content")

    def test_source_without_an_indexed_hash_falls_back_to_the_suffix(self):
        # no hash in the index -> the question cannot be answered, so the safe
        # (merely wasteful) branch is taken
        fid = self.add_file("a.jpg", content=b"same", country="RU", city="Moskva")
        self.conn.execute("UPDATE files SET hash = NULL WHERE id = ?", (fid,))
        self.conn.commit()
        self.target_dir.mkdir(parents=True)
        (self.target_dir / "a.jpg").write_bytes(b"same")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        self.assertEqual(report.skipped_already_copied, 0)
        self.assertTrue((self.target_dir / "a_1.jpg").exists())

    def test_two_sources_of_the_same_name_still_get_distinct_targets(self):
        # one of them is already in the target; the other must not be handed the same
        # path just because it is now "claimed as already copied"
        self.add_file("x/img.jpg", content=b"one", country="RU", city="Moskva")
        self.add_file("y/img.jpg", content=b"two", country="RU", city="Moskva")
        plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        self.assertEqual(report.skipped_already_copied, 2)
        self.assertEqual(report.moved, 0)
        contents = {(self.target_dir / "img.jpg").read_bytes(),
                    (self.target_dir / "img_1.jpg").read_bytes()}
        self.assertEqual(contents, {b"one", b"two"})
        self.assertFalse((self.target_dir / "img_2.jpg").exists())


class TestUndoInterruptedRun(SorterTestBase):
    """Part 3 (engine): cancelling a rollback, the tail of an interrupted transfer and
    closing a batch that was left open."""

    def _copy_batch(self, *names: str) -> int:
        for name in names:
            self.add_file(name, content=name.encode(), country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        return report.batch_id

    def _interrupt(self, batch_id: int, dst_name: str) -> None:
        """Emulate the crash: the moves row is committed BEFORE the FS operation, so a
        run killed in between leaves a fully written file in status='planned' and a
        batch without finished_at."""
        self.conn.execute(
            "UPDATE moves SET status = 'planned' WHERE batch_id = ? AND dst LIKE ?",
            (batch_id, f"%{dst_name}"))
        self.conn.execute(
            "UPDATE move_batches SET finished_at = NULL WHERE id = ?", (batch_id,))
        self.conn.commit()

    def move_statuses(self, batch_id: int) -> list[str]:
        return [r["status"] for r in self.conn.execute(
            "SELECT status FROM moves WHERE batch_id = ? ORDER BY id", (batch_id,))]

    def test_cancel_midway_leaves_the_rest_untouched_and_a_repeat_finishes_it(self):
        batch_id = self._copy_batch("a.jpg", "b.jpg", "c.jpg")
        stats = undo(self.conn, batch_id=batch_id, should_cancel=_cancel_after(1))
        self.assertTrue(stats.cancelled)
        self.assertEqual(stats.undone, 1)
        self.assertEqual(sorted(self.move_statuses(batch_id)), ["done", "done", "undone"])
        self.assertEqual(len(list(self.dest.rglob("*.jpg"))), 2)

        again = undo(self.conn, batch_id=batch_id)
        self.assertFalse(again.cancelled)
        self.assertEqual(again.undone, 2)
        self.assertEqual(list(self.dest.rglob("*.jpg")), [])

    def test_without_should_cancel_behaviour_is_unchanged(self):
        batch_id = self._copy_batch("a.jpg", "b.jpg")
        stats = undo(self.conn, batch_id=batch_id)
        self.assertFalse(stats.cancelled)
        self.assertEqual(stats.undone, 2)
        self.assertEqual(stats.stray, [])

    def test_tail_row_with_a_matching_hash_is_deleted_and_marked_undone(self):
        batch_id = self._copy_batch("a.jpg", "b.jpg")
        self._interrupt(batch_id, "a.jpg")
        stats = undo(self.conn, batch_id=batch_id)
        self.assertEqual(stats.undone, 2)      # the done row + the tail row
        self.assertEqual(stats.stray, [])
        self.assertEqual(list(self.dest.rglob("*.jpg")), [])
        self.assertEqual(sorted(self.move_statuses(batch_id)), ["undone", "undone"])

    def test_tail_row_with_a_broken_copy_is_kept_and_reported(self):
        batch_id = self._copy_batch("a.jpg")
        dst = self.dest / "Russia" / "Moskva" / "2022" / "a.jpg"
        self._interrupt(batch_id, "a.jpg")
        dst.write_bytes(b"half a picture")  # what an interrupted copy leaves behind
        stats = undo(self.conn, batch_id=batch_id)
        self.assertEqual(stats.undone, 0)
        self.assertEqual(stats.stray, [str(dst)])
        self.assertTrue(dst.exists())  # never deleted on our own initiative
        self.assertEqual(self.move_statuses(batch_id), ["planned"])

    def test_tail_row_without_a_file_on_disk_is_not_reported_at_all(self):
        # the FS operation never started: nothing was written, nothing to undo
        batch_id = self._copy_batch("a.jpg")
        dst = self.dest / "Russia" / "Moskva" / "2022" / "a.jpg"
        self._interrupt(batch_id, "a.jpg")
        dst.unlink()
        stats = undo(self.conn, batch_id=batch_id)
        self.assertEqual((stats.undone, stats.missing, stats.failed), (0, 0, 0))
        self.assertEqual(stats.stray, [])

    def test_move_batch_tail_is_restored_to_src_not_deleted(self):
        fid = self.add_file("a.jpg", content=b"aaa", country="RU", city="Moskva")
        orig = self.path_of(fid)
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.assertFalse(Path(orig).exists())
        # a move batch's tail must never take the copy branch — deleting dst there
        # would destroy the only remaining instance of the file
        self.conn.execute("UPDATE moves SET status = 'planned' WHERE batch_id = ?",
                          (report.batch_id,))
        self.conn.execute("UPDATE files SET path = ? WHERE id = ?",
                          (orig, fid))  # a move whose files.path update never ran
        self.conn.commit()
        stats = undo(self.conn, batch_id=report.batch_id)
        self.assertEqual(stats.undone, 1)
        self.assertTrue(Path(orig).exists())
        self.assertEqual(Path(orig).read_bytes(), b"aaa")

    def test_interrupted_batch_is_closed_by_the_rollback(self):
        batch_id = self._copy_batch("a.jpg")
        self._interrupt(batch_id, "a.jpg")
        undo(self.conn, batch_id=batch_id)
        finished = self.conn.execute(
            "SELECT finished_at FROM move_batches WHERE id = ?",
            (batch_id,)).fetchone()["finished_at"]
        self.assertIsNotNone(finished)

    def test_hash_mismatch_on_a_done_copy_still_keeps_the_file_and_the_status(self):
        batch_id = self._copy_batch("a.jpg")
        dst = self.dest / "Russia" / "Moskva" / "2022" / "a.jpg"
        dst.write_bytes(b"edited by the user after the copy")
        stats = undo(self.conn, batch_id=batch_id)
        self.assertEqual(stats.failed, 1)
        self.assertEqual(stats.undone, 0)
        self.assertEqual(stats.stray, [])  # a 'done' mismatch is `failed`, not stray
        self.assertTrue(dst.exists())
        self.assertEqual(self.move_statuses(batch_id), ["done"])


class TestNonAsciiPaths(SorterTestBase):
    """Part 2 (companion): a Cyrillic/CJK city and file name must keep working — the
    long-path prefix is glued onto exactly these paths."""

    def test_cyrillic_and_cjk_names_round_trip_through_apply_and_undo(self):
        self.add_file("фото.jpg", content=b"ru", country="RU", city="Москва")
        self.add_file("写真.jpg", content=b"jp", country="JP",
                      country_name="日本", city="東京")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        self.assertEqual(report.moved, 2)
        self.assertEqual(report.failed, 0)
        ru = self.dest / "Russia" / "Москва" / "2022" / "фото.jpg"
        jp = self.dest / "日本" / "東京" / "2022" / "写真.jpg"
        self.assertTrue(ru.exists())
        self.assertTrue(jp.exists())

        stats = undo(self.conn, batch_id=report.batch_id)
        self.assertEqual(stats.undone, 2)
        self.assertFalse(ru.exists())
        self.assertFalse(jp.exists())


@unittest.skipUnless(os.name == "nt", "the MAX_PATH limit is a Windows-only class of bug")
class TestWindowsLongPaths(SorterTestBase):
    """Part 2: a destination past 260 characters must be copied, not counted in
    `failed`. `tmp_path` alone never produces this class — the paths are short."""

    # 200 chars: the total goes past MAX_PATH while every component stays inside the
    # 255-character limit the filesystem enforces with the prefix as well as without it
    LONG_DIR = "d" * 200

    def setUp(self):
        super().setUp()
        self.dest = self.root / self.LONG_DIR
        self.target_dir = self.dest / "Russia" / "Moskva" / "2022"

    def test_copy_to_a_path_past_max_path_succeeds(self):
        self.add_file("a.jpg", content=b"long", country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        dst = self.target_dir / "a.jpg"
        self.assertGreater(len(str(dst)), 260)
        self.assertEqual(report.failed, 0)
        self.assertEqual(report.moved, 1)
        self.assertTrue(_fs(dst).exists())
        self.assertEqual(file_hash(_fs(dst))[0], file_hash(self.src_dir / "a.jpg")[0])

    def test_name_conflict_on_a_long_path_picks_the_suffix_correctly(self):
        # the existence check has to see through MAX_PATH too — otherwise exists()
        # lies "no", the name without a suffix is chosen and the write fails anyway
        self.add_file("x/img.jpg", content=b"one", country="RU", city="Moskva")
        self.add_file("y/img.jpg", content=b"two", country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        self.assertEqual(report.failed, 0)
        self.assertEqual(report.moved, 2)
        self.assertTrue(_fs(self.target_dir / "img.jpg").exists())
        self.assertTrue(_fs(self.target_dir / "img_1.jpg").exists())

    def test_undo_deletes_a_copy_at_a_long_path(self):
        self.add_file("a.jpg", content=b"long", country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        dst = self.target_dir / "a.jpg"
        self.assertTrue(_fs(dst).exists())
        stats = undo(self.conn, batch_id=report.batch_id)
        self.assertEqual(stats.undone, 1)
        self.assertEqual(stats.failed, 0)
        self.assertFalse(_fs(dst).exists())

    def test_repeated_apply_to_a_long_path_does_not_duplicate(self):
        self.add_file("a.jpg", content=b"long", country="RU", city="Moskva")
        plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True, copy=True)
        self.assertEqual(report.skipped_already_copied, 1)
        self.assertFalse(_fs(self.target_dir / "a_1.jpg").exists())

    def test_component_over_255_chars_is_still_rejected_by_the_filesystem(self):
        """The prefix lifts MAX_PATH, NOT the per-component limit. Checked rather than
        assumed (brief part 2, item 5): city names are short and file names come from a
        filesystem that enforces the same 255, so nothing here defends against it — but
        the fact has to be on record, and it has to fail as an OSError rather than
        silently produce a truncated name."""
        with self.assertRaises(OSError):
            _fs(self.root / ("x" * 300)).mkdir()


if __name__ == "__main__":
    unittest.main()
