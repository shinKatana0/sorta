"""F82: `skip_layout` — "do not lay out", the half of the tri-state tree that `sort` reads.

The distinction the whole feature exists for: a folder marked "do not scan" (F81) has no
rows in the index at all, while a folder marked "do not lay out" is indexed exactly as
before — found by search, counted in statistics, compared for duplicates — and only
left where it lies by the layout. These tests hold that line: the files stay in the
index, the plan does not mention them, and everything outside the exclusions is laid out
byte for byte as it was.

Inherits the SorterTestBase fixtures from test_sorter.py. All FS operations — on
tmp_path only.
"""
from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from tests.test_sorter import SorterTestBase

from sorta.hashing import file_hash
from sorta.indexer import excludes_path
from sorta.sorter import plan_and_sort


class LayoutExcludesTestBase(SorterTestBase):
    def excludes_file(self) -> Path:
        return excludes_path(self.cfg)

    def write_excludes(self, skip_layout: list[str] | None = None,
                       skip_scan: list[str] | None = None,
                       root: Path | None = None) -> None:
        sections: dict[str, list[str]] = {}
        if skip_scan:
            sections["skip_scan"] = skip_scan
        if skip_layout:
            sections["skip_layout"] = skip_layout
        path = self.excludes_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump({(root or self.src_dir).resolve().as_posix(): sections},
                           allow_unicode=True),
            encoding="utf-8")


class TestSkipLayoutLeavesFilesInTheIndex(LayoutExcludesTestBase):
    def test_folder_is_out_of_the_plan_but_its_files_stay_indexed(self):
        greece = self.add_file("Греция/athens.jpg", country="GR", city="Athens")
        moscow = self.add_file("moscow.jpg", country="RU", city="Moskva")
        self.write_excludes(skip_layout=["Греция"])

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)

        self.assertEqual({it.file_id for it in report.plan}, {moscow})
        self.assertEqual(report.excluded, 1)
        # the point of the feature: unlike "do not scan", the row is still there
        rows = self.conn.execute("SELECT id FROM files ORDER BY id").fetchall()
        self.assertEqual({r["id"] for r in rows}, {greece, moscow})

    def test_the_files_still_take_part_in_duplicate_search(self):
        # the same bytes inside and outside the excluded folder: dedup must still see
        # both, because "do not lay out" says nothing about the index
        excluded = self.add_file("Греция/copy.jpg", content=b"identical")
        kept = self.add_file("copy.jpg", content=b"identical")
        self.write_excludes(skip_layout=["Греция"])

        plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)

        digest = self.conn.execute("SELECT hash FROM files WHERE id = ?",
                                   (kept,)).fetchone()["hash"]
        same = self.conn.execute("SELECT id FROM files WHERE hash = ?", (digest,)).fetchall()
        self.assertEqual({r["id"] for r in same}, {excluded, kept})

    def test_nested_subfolders_are_excluded_too(self):
        self.add_file("Греция/Афины/Акрополь/temple.jpg", country="GR", city="Athens")
        self.write_excludes(skip_layout=["Греция"])

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)

        self.assertEqual(report.plan, [])
        self.assertEqual(report.excluded, 1)

    def test_apply_does_not_move_an_excluded_file(self):
        self.add_file("Греция/athens.jpg", country="GR", city="Athens")
        self.write_excludes(skip_layout=["Греция"])
        original = self.src_dir / "Греция" / "athens.jpg"

        plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True,
                      write_reports=False)

        self.assertTrue(original.exists(), "файл «не раскладывать» сдвинулся с места")


class TestSourcesCombineNotReplace(LayoutExcludesTestBase):
    def test_config_exclude_dirs_and_the_file_are_both_applied(self):
        cfg = replace(self.cfg,
                      sort=replace(self.cfg.sort,
                                   exclude_dirs=[str(self.src_dir / "Италия")]))
        self.add_file("Греция/athens.jpg", country="GR", city="Athens")
        self.add_file("Италия/rome.jpg", country="IT", city="Rome")
        moscow = self.add_file("moscow.jpg", country="RU", city="Moskva")
        self.write_excludes(skip_layout=["Греция"])

        report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)

        self.assertEqual({it.file_id for it in report.plan}, {moscow})
        self.assertEqual(report.excluded, 2)

    def test_the_cli_exclude_flag_is_added_to_and_not_replaced_by_the_file(self):
        self.add_file("Греция/athens.jpg", country="GR", city="Athens")
        self.add_file("Франция/paris.jpg", country="FR", city="Paris")
        moscow = self.add_file("moscow.jpg", country="RU", city="Moskva")
        self.write_excludes(skip_layout=["Греция"])

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False,
                               exclude=[str(self.src_dir / "Франция")])

        self.assertEqual({it.file_id for it in report.plan}, {moscow})
        self.assertEqual(report.excluded, 2)

    def test_all_three_sources_at_once(self):
        cfg = replace(self.cfg,
                      sort=replace(self.cfg.sort,
                                   exclude_dirs=[str(self.src_dir / "Италия")]))
        self.add_file("Греция/athens.jpg", country="GR", city="Athens")
        self.add_file("Италия/rome.jpg", country="IT", city="Rome")
        self.add_file("Франция/paris.jpg", country="FR", city="Paris")
        moscow = self.add_file("moscow.jpg", country="RU", city="Moskva")
        self.write_excludes(skip_layout=["Греция"])

        report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False,
                               exclude=[str(self.src_dir / "Франция")])

        self.assertEqual({it.file_id for it in report.plan}, {moscow})
        self.assertEqual(report.excluded, 3)

    def test_skip_scan_alone_does_not_exclude_anything_from_the_layout(self):
        # skip_scan is the indexer's business: a row that somehow IS in the index (an
        # exclusion added after indexing, before the next run) is not the sorter's to
        # drop — index() removes it, and until then the plan stays honest about it.
        athens = self.add_file("Греция/athens.jpg", country="GR", city="Athens")
        moscow = self.add_file("moscow.jpg", country="RU", city="Moskva")
        self.write_excludes(skip_scan=["Греция"])

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)

        self.assertEqual({it.file_id for it in report.plan}, {athens, moscow})
        self.assertEqual(report.excluded, 0)

    def test_an_entry_for_another_root_does_not_touch_this_source(self):
        other = self.root / "elsewhere"
        (other / "Греция").mkdir(parents=True)
        athens = self.add_file("Греция/athens.jpg", country="GR", city="Athens")
        self.write_excludes(skip_layout=["Греция"], root=other)

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)

        self.assertEqual({it.file_id for it in report.plan}, {athens})
        self.assertEqual(report.excluded, 0)

    def test_an_escaping_entry_cannot_widen_the_exclusion(self):
        # the file is outside input (§1): an entry may only narrow the layout, never
        # point somewhere else — a rejected one simply excludes nothing
        moscow = self.add_file("moscow.jpg", country="RU", city="Moskva")
        self.write_excludes(skip_layout=["../..", "/etc", "C:/windows"])

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)

        self.assertEqual({it.file_id for it in report.plan}, {moscow})
        self.assertEqual(report.excluded, 0)


class TestRegressionOutsideTheExclusions(LayoutExcludesTestBase):
    def test_files_outside_are_moved_byte_for_byte(self):
        self.add_file("Греция/athens.jpg", content=b"greece bytes",
                      country="GR", city="Athens")
        self.add_file("moscow.jpg", content=b"moscow bytes" * 40,
                      country="RU", city="Moskva")
        source = self.src_dir / "moscow.jpg"
        before, algo = file_hash(source)
        self.write_excludes(skip_layout=["Греция"])

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True,
                               write_reports=False)

        self.assertEqual(report.moved, 1)
        moved = report.plan[0].dst
        self.assertTrue(moved.exists())
        self.assertFalse(source.exists())
        self.assertEqual(file_hash(moved), (before, algo))
        self.assertEqual(moved.read_bytes(), b"moscow bytes" * 40)

    def test_without_the_file_the_plan_is_exactly_what_it_was(self):
        self.add_file("Греция/athens.jpg", country="GR", city="Athens")
        self.add_file("moscow.jpg", country="RU", city="Moskva")
        self.assertFalse(self.excludes_file().exists())

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)

        self.assertEqual(len(report.plan), 2)
        self.assertEqual(report.excluded, 0)

    def test_an_empty_or_broken_file_does_not_fail_the_run(self):
        self.add_file("moscow.jpg", country="RU", city="Moskva")
        for text in ("", "- not: a mapping\n", "this: [is: not\n  valid yaml\n"):
            with self.subTest(text=text):
                self.excludes_file().write_text(text, encoding="utf-8")
                report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                                       apply=False)
                self.assertEqual(len(report.plan), 1)
                self.assertEqual(report.excluded, 0)


if __name__ == "__main__":
    unittest.main()
