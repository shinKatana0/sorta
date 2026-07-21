"""F16: --exclude / sort.exclude_dirs — excluding already-sorted directories.

Inherits the SorterTestBase fixtures from test_sorter.py. All FS operations — on
tmp_path only.
"""
from __future__ import annotations

import os
import unittest
from dataclasses import replace

from tests.test_sorter import SorterTestBase

from sorta.sorter import plan_and_sort


class TestExcludeDirs(SorterTestBase):
    def test_file_under_exclude_dir_not_in_plan(self):
        self.add_file("Япония/tokyo.jpg", country="Japan", city="Tokyo")
        self.add_file("moscow.jpg", country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False,
                               exclude=[str(self.src_dir / "Япония")])
        self.assertEqual(len(report.plan), 1)
        self.assertEqual(report.plan[0].city, "Moskva")
        self.assertEqual(report.excluded, 1)

    def test_file_outside_exclude_dir_is_in_plan(self):
        self.add_file("moscow.jpg", country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False,
                               exclude=[str(self.src_dir / "Япония")])
        self.assertEqual(len(report.plan), 1)
        self.assertEqual(report.excluded, 0)

    def test_directory_boundary_not_string_prefix(self):
        # Япония excludes its subfolders, but NOT ЯпонияДругое (a shared string
        # prefix, but not a directory boundary).
        self.add_file("Япония/tokyo.jpg", country="Japan", city="Tokyo")
        self.add_file("ЯпонияДругое/osaka.jpg", country="Japan", city="Osaka")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False,
                               exclude=[str(self.src_dir / "Япония")])
        self.assertEqual(len(report.plan), 1)
        self.assertEqual(report.plan[0].city, "Osaka")
        self.assertEqual(report.excluded, 1)

    def test_nested_subfolders_excluded(self):
        self.add_file("Япония/Токио/Асакуса/temple.jpg", country="Japan", city="Tokyo")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False,
                               exclude=[str(self.src_dir / "Япония")])
        self.assertEqual(len(report.plan), 0)
        self.assertEqual(report.excluded, 1)

    def test_multiple_exclude_dirs(self):
        self.add_file("Япония/tokyo.jpg", country="Japan", city="Tokyo")
        self.add_file("Италия/rome.jpg", country="Italy", city="Rome")
        self.add_file("moscow.jpg", country="RU", city="Moskva")
        report = plan_and_sort(
            self.cfg, self.conn, "city", self.dest, apply=False,
            exclude=[str(self.src_dir / "Япония"), str(self.src_dir / "Италия")])
        self.assertEqual(len(report.plan), 1)
        self.assertEqual(report.plan[0].city, "Moskva")
        self.assertEqual(report.excluded, 2)

    @unittest.skipUnless(
        os.name == "nt",
        "case-insensitive path comparison — Windows only (ntpath casefold); "
        "on a case-sensitive FS ЯПОНИЯ ≠ Япония, which is correct")
    def test_case_insensitive_on_windows(self):
        self.add_file("Япония/tokyo.jpg", country="Japan", city="Tokyo")
        exclude_path = str(self.src_dir / "ЯПОНИЯ")  # different case
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False,
                               exclude=[exclude_path])
        self.assertEqual(len(report.plan), 0)
        self.assertEqual(report.excluded, 1)

    def test_no_exclude_regresses_to_f5_plan(self):
        self.add_file("Япония/tokyo.jpg", country="Japan", city="Tokyo")
        self.add_file("moscow.jpg", country="RU", city="Moskva")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(len(report.plan), 2)
        self.assertEqual(report.excluded, 0)

    def test_config_exclude_dirs_combined_with_param(self):
        cfg = replace(self.cfg,
                      sort=replace(self.cfg.sort, exclude_dirs=[str(self.src_dir / "Италия")]))
        self.add_file("Япония/tokyo.jpg", country="Japan", city="Tokyo")
        self.add_file("Италия/rome.jpg", country="Italy", city="Rome")
        self.add_file("moscow.jpg", country="RU", city="Moskva")
        report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False,
                               exclude=[str(self.src_dir / "Япония")])
        self.assertEqual(len(report.plan), 1)
        self.assertEqual(report.plan[0].city, "Moskva")
        self.assertEqual(report.excluded, 2)

    def test_excluded_not_in_unsorted_or_junk(self):
        # An excluded file without a date/place would go to _Unsorted without exclude —
        # with exclude it must not be in the plan at all.
        self.add_file("Япония/no_place.jpg", taken_at=None)
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False,
                               exclude=[str(self.src_dir / "Япония")])
        self.assertEqual(len(report.plan), 0)
        self.assertEqual(report.excluded, 1)

    def test_exclude_with_where_and_dedupe(self):
        # excluded_fid matches --where (city=paris) and would be a candidate in the
        # same near-dup group (same phash) — exclude must take it out BEFORE both
        # stages, despite the where match and the duplicate.
        excluded_fid = self.add_file("Япония/dup.jpg", content=b"same",
                                     country="France", city="Paris")
        kept_fid = self.add_file("paris1.jpg", content=b"a" * 300,
                                 country="France", city="Paris")
        other_fid = self.add_file("paris2.jpg", content=b"b" * 200,
                                  country="France", city="Paris")
        for fid in (excluded_fid, kept_fid, other_fid):
            self.conn.execute("UPDATE files SET phash = ? WHERE id = ?", ("0" * 16, fid))
        self.conn.execute("UPDATE files SET width = 1920, height = 1080 WHERE id = ?",
                          (kept_fid,))
        self.conn.execute("UPDATE files SET width = 400, height = 300 WHERE id = ?",
                          (other_fid,))
        self.conn.commit()

        report = plan_and_sort(
            self.cfg, self.conn, "city", self.dest, apply=False,
            where=["city=paris"], dedupe=True,
            exclude=[str(self.src_dir / "Япония")])

        self.assertEqual(report.excluded, 1)
        file_ids = {it.file_id for it in report.plan}
        self.assertNotIn(excluded_fid, file_ids)
        self.assertEqual(file_ids, {kept_fid, other_fid})
