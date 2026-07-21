"""F14: --dedupe / --delete-worse-dupes — near-duplicate resolution during sorting.

All FS operations — on tmp_path only (inheriting the SorterTestBase fixtures from
test_sorter.py). near_duplicate_groups reads files.phash directly — test
near-duplicates get the same phash via a direct UPDATE, without real image decoding.
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tests.test_sorter import SorterTestBase

from sorta.sorter import plan_and_sort


class DedupeTestBase(SorterTestBase):
    def set_phash(self, file_id: int, phash: str) -> None:
        self.conn.execute("UPDATE files SET phash = ? WHERE id = ?", (phash, file_id))
        self.conn.commit()

    def set_dims(self, file_id: int, width: int, height: int) -> None:
        self.conn.execute(
            "UPDATE files SET width = ?, height = ? WHERE id = ?",
            (width, height, file_id))
        self.conn.commit()

    def plan_by_file(self, report, file_id: int):
        return next(it for it in report.plan if it.file_id == file_id)


class TestDedupeGrouping(DedupeTestBase):
    def test_best_by_dimensions_then_size_normal_worse_to_duplicates(self):
        big = self.add_file("big.jpg", content=b"a" * 300, country="France", city="Paris")
        mid = self.add_file("mid.jpg", content=b"b" * 200, country="France", city="Paris")
        small = self.add_file("small.jpg", content=b"c" * 100, country="France", city="Paris")
        for fid in (big, mid, small):
            self.set_phash(fid, "0" * 16)
        self.set_dims(big, 1920, 1080)
        self.set_dims(mid, 800, 600)
        self.set_dims(small, 400, 300)

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=False, dedupe=True)

        best = self.plan_by_file(report, big)
        worse_mid = self.plan_by_file(report, mid)
        worse_small = self.plan_by_file(report, small)

        self.assertEqual(best.reason, "city")
        self.assertEqual(best.near_dup_role, "kept")
        self.assertEqual(best.target_rel, "France/Paris/2022/big.jpg")

        for worse in (worse_mid, worse_small):
            self.assertEqual(worse.reason, "near_dup")
            self.assertEqual(worse.near_dup_role, "moved")
            self.assertTrue(worse.target_rel.startswith("_Duplicates/"))

        self.assertEqual(best.near_dup_group, worse_mid.near_dup_group)
        self.assertEqual(best.near_dup_group, worse_small.near_dup_group)

    def test_equal_quality_tiebreaks_deterministically_by_id(self):
        first = self.add_file("first.jpg", content=b"x" * 100, country="RU", city="Moskva")
        second = self.add_file("second.jpg", content=b"x" * 100, country="RU", city="Moskva")
        for fid in (first, second):
            self.set_phash(fid, "f" * 16)
            self.set_dims(fid, 640, 480)

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=False, dedupe=True)

        self.assertEqual(self.plan_by_file(report, first).near_dup_role, "kept")
        self.assertEqual(self.plan_by_file(report, second).near_dup_role, "moved")

        # The run is deterministic: a repeated call gives the same choice.
        report2 = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                                apply=False, dedupe=True)
        self.assertEqual(self.plan_by_file(report2, first).near_dup_role, "kept")

    def test_where_trims_group_below_two_leaves_normal_sort(self):
        # The second group member is filtered by --where (a different city) — the
        # remaining single file of the selection is not a duplicate: sorted normally.
        paris = self.add_file("paris.jpg", country="France", city="Paris")
        moskva = self.add_file("moskva.jpg", country="RU", city="Moskva")
        for fid in (paris, moskva):
            self.set_phash(fid, "1" * 16)
            self.set_dims(fid, 100, 100)

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False,
                               dedupe=True, where=["city=paris"])
        self.assertEqual(len(report.plan), 1)
        item = report.plan[0]
        self.assertEqual(item.reason, "city")
        self.assertIsNone(item.near_dup_group)
        self.assertIsNone(item.near_dup_role)

    def test_junk_verdict_overrides_near_dup(self):
        junk = self.add_file("junk.jpg", content=b"j" * 50, junk_verdict="screenshot")
        photo = self.add_file("photo.jpg", content=b"p" * 200, country="RU", city="Moskva")
        for fid in (junk, photo):
            self.set_phash(fid, "2" * 16)
        self.set_dims(junk, 2000, 2000)   # larger, but junk still does not participate
        self.set_dims(photo, 100, 100)

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=False, dedupe=True)
        junk_item = self.plan_by_file(report, junk)
        photo_item = self.plan_by_file(report, photo)
        self.assertEqual(junk_item.reason, "junk")
        self.assertIsNone(junk_item.near_dup_group)
        # junk is excluded from the group -> photo remains the only member -> not a dup
        self.assertEqual(photo_item.reason, "city")
        self.assertIsNone(photo_item.near_dup_group)


class TestDedupeWithoutFlag(DedupeTestBase):
    def test_near_duplicates_untouched_without_dedupe_flag(self):
        a = self.add_file("a.jpg", country="RU", city="Moskva")
        b = self.add_file("b.jpg", country="RU", city="Moskva")
        for fid in (a, b):
            self.set_phash(fid, "3" * 16)
            self.set_dims(fid, 100, 100)

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        for item in report.plan:
            self.assertEqual(item.reason, "city")
            self.assertIsNone(item.near_dup_group)
            self.assertIsNone(item.near_dup_role)


class TestDedupeMissingPhash(DedupeTestBase):
    def test_dedupe_without_phash_hints_and_builds_no_plan(self):
        self.add_file("a.jpg", country="RU", city="Moskva")
        buf = io.StringIO()
        with redirect_stdout(buf):
            report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                                   apply=False, dedupe=True)
        self.assertIn("sorta phash", buf.getvalue())
        self.assertEqual(report.plan, [])
        self.assertFalse(report.csv_path.exists())
        self.assertFalse(report.html_path.exists())


class TestDedupeReportSections(DedupeTestBase):
    def test_csv_has_near_dup_columns_only_with_dedupe(self):
        big = self.add_file("big.jpg", content=b"a" * 300, country="France", city="Paris")
        small = self.add_file("small.jpg", content=b"b" * 100, country="France", city="Paris")
        for fid in (big, small):
            self.set_phash(fid, "4" * 16)
        self.set_dims(big, 1000, 1000)
        self.set_dims(small, 10, 10)

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=False, dedupe=True)
        rows = self.read_csv(report.csv_path)
        self.assertIn("near_dup_group", rows[0])
        self.assertIn("near_dup_role", rows[0])
        by_name = {r["path"].split("\\")[-1].split("/")[-1]: r for r in rows}
        self.assertEqual(by_name["big.jpg"]["near_dup_role"], "kept")
        self.assertEqual(by_name["small.jpg"]["near_dup_role"], "moved")

        # Without --dedupe there are no columns at all (the F5 CSV is unchanged).
        report_plain = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        with open(report_plain.csv_path, encoding="utf-8-sig") as fh:
            header = fh.readline().strip().split(";")
        self.assertNotIn("near_dup_group", header)

    def test_html_has_near_dup_section(self):
        big = self.add_file("big.jpg", content=b"a" * 300, country="France", city="Paris")
        small = self.add_file("small.jpg", content=b"b" * 100, country="France", city="Paris")
        for fid in (big, small):
            self.set_phash(fid, "5" * 16)
        self.set_dims(big, 1000, 1000)
        self.set_dims(small, 10, 10)

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=False, dedupe=True)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn("Почти-дубликаты", html)
        self.assertIn("big.jpg", html)
        self.assertIn("small.jpg", html)

        report_plain = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        html_plain = report_plain.html_path.read_text(encoding="utf-8")
        self.assertNotIn("Почти-дубликаты", html_plain)

    def test_leaf_table_category_shows_near_dup_role(self):
        # F23: the leaf's Category column reflects the near-dup role, not only the
        # top "Near-duplicates" section.
        big = self.add_file("big.jpg", content=b"a" * 300, country="France", city="Paris")
        small = self.add_file("small.jpg", content=b"b" * 100, country="France", city="Paris")
        for fid in (big, small):
            self.set_phash(fid, "6" * 16)
        self.set_dims(big, 1000, 1000)
        self.set_dims(small, 10, 10)

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=False, dedupe=True)
        html = report.html_path.read_text(encoding="utf-8")
        self.assertIn("city · оставлен", html)
        self.assertIn("near_dup · в дубли", html)


class TestDedupeApplyMoves(DedupeTestBase):
    def test_apply_moves_worse_to_duplicates_dir_and_journals(self):
        big = self.add_file("big.jpg", content=b"a" * 300, country="France", city="Paris")
        small = self.add_file("small.jpg", content=b"b" * 100, country="France", city="Paris")
        for fid in (big, small):
            self.set_phash(fid, "6" * 16)
        self.set_dims(big, 1000, 1000)
        self.set_dims(small, 10, 10)

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=True, dedupe=True)
        self.assertEqual(report.moved, 2)
        self.assertTrue((self.dest / "France" / "Paris" / "2022" / "big.jpg").exists())
        self.assertTrue((self.dest / "_Duplicates" / "small.jpg").exists())
        self.assertEqual(self.move_status(report.batch_id, small), "done")


class TestDeleteWorseDupes(DedupeTestBase):
    def test_requires_dedupe_flag(self):
        with self.assertRaises(ValueError):
            plan_and_sort(self.cfg, self.conn, "city", self.dest,
                         apply=False, dedupe=False, delete_worse_dupes=True)

    def test_dry_run_does_not_delete(self):
        big = self.add_file("big.jpg", content=b"a" * 300, country="France", city="Paris")
        small = self.add_file("small.jpg", content=b"b" * 100, country="France", city="Paris")
        for fid in (big, small):
            self.set_phash(fid, "7" * 16)
        self.set_dims(big, 1000, 1000)
        self.set_dims(small, 10, 10)
        small_path = Path(self.path_of(small))

        buf = io.StringIO()
        with redirect_stdout(buf):
            report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False,
                                   dedupe=True, delete_worse_dupes=True)
        self.assertIn("БЕЗВОЗВРАТНО", buf.getvalue())
        self.assertTrue(small_path.exists())
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0], 0)
        worse_item = self.plan_by_file(report, small)
        self.assertEqual(worse_item.reason, "near_dup_delete")
        self.assertEqual(worse_item.near_dup_role, "deleted")

    def test_apply_deletes_worse_keeps_best_and_journals(self):
        big = self.add_file("big.jpg", content=b"a" * 300, country="France", city="Paris")
        small = self.add_file("small.jpg", content=b"b" * 100, country="France", city="Paris")
        for fid in (big, small):
            self.set_phash(fid, "8" * 16)
        self.set_dims(big, 1000, 1000)
        self.set_dims(small, 10, 10)
        small_path = Path(self.path_of(small))

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True,
                               dedupe=True, delete_worse_dupes=True)
        self.assertFalse(small_path.exists())
        self.assertTrue((self.dest / "France" / "Paris" / "2022" / "big.jpg").exists())
        self.assertEqual(report.deleted, 1)
        self.assertEqual(self.move_status(report.batch_id, small), "deleted")
        # a deleted file is not undone by undo (undo looks only at status='done')
        from sorta.sorter import undo
        stats = undo(self.conn, batch_id=report.batch_id)
        self.assertEqual(stats.undone, 1)  # only big
        self.assertFalse(small_path.exists())


if __name__ == "__main__":
    unittest.main()
