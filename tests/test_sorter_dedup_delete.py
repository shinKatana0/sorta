"""U3b: the sorter routes dedup_choice.action='to_delete' into the _удалить folder.

All FS operations — on tmp_path only (inheriting the SorterTestBase fixtures from
test_sorter.py). dedup_choice — the v7 table (U3), populated by the web app (U3);
here rows are inserted directly via SQL, without running ui.py.
"""
from __future__ import annotations

import unittest

from tests.test_sorter import SorterTestBase

from sorta.config import Config
from sorta.sorter import plan_and_sort


class DedupDeleteTestBase(SorterTestBase):
    def set_dedup_choice(self, file_id: int, action: str) -> None:
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at) VALUES (?, ?, '2026-01-01')",
            (file_id, action))
        self.conn.commit()


class TestToDeleteRouting(DedupDeleteTestBase):
    def test_to_delete_routes_to_delete_folder_en(self):
        fid = self.add_file("dup.jpg", country="France", city="Paris")
        self.set_dedup_choice(fid, "to_delete")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        item = report.plan[0]
        self.assertEqual(item.reason, "dedup_delete")
        self.assertEqual(item.target_rel, "_delete/dup.jpg")

    def test_to_delete_routes_to_delete_folder_ru(self):
        cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                    raw={"language": "ru"})
        fid = self.add_file("dup.jpg", country="France", city="Paris")
        self.set_dedup_choice(fid, "to_delete")
        report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False)
        item = report.plan[0]
        self.assertEqual(item.reason, "dedup_delete")
        self.assertEqual(item.target_rel, "_удалить/dup.jpg")

    def test_to_delete_overrides_city_placement(self):
        fid = self.add_file("dup.jpg", country="France", city="Paris")
        self.set_dedup_choice(fid, "to_delete")
        for mode in ("city", "person", "event"):
            report = plan_and_sort(self.cfg, self.conn, mode, self.dest, apply=False)
            self.assertEqual(report.plan[0].reason, "dedup_delete")
            self.assertEqual(report.plan[0].target_rel, "_delete/dup.jpg")

    def test_to_delete_overrides_junk_verdict(self):
        fid = self.add_file("dup.jpg", junk_verdict="screenshot")
        self.set_dedup_choice(fid, "to_delete")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].reason, "dedup_delete")
        self.assertEqual(report.plan[0].target_rel, "_delete/dup.jpg")

    def test_to_delete_overrides_not_personal(self):
        fid = self.add_file("Movie.mkv", country="France", city="Paris")
        self.conn.execute("UPDATE files SET not_personal = 1 WHERE id = ?", (fid,))
        self.set_dedup_choice(fid, "to_delete")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertEqual(report.plan[0].reason, "dedup_delete")
        self.assertEqual(report.plan[0].target_rel, "_delete/Movie.mkv")

    def test_keep_action_sorts_normally(self):
        fid = self.add_file("keeper.jpg", country="France", city="Paris")
        self.set_dedup_choice(fid, "keep")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        item = report.plan[0]
        self.assertEqual(item.reason, "city")
        self.assertEqual(item.target_rel, "France/Paris/2022/keeper.jpg")

    def test_no_dedup_choice_row_sorts_normally(self):
        self.add_file("plain.jpg", country="France", city="Paris")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        item = report.plan[0]
        self.assertEqual(item.reason, "city")
        self.assertEqual(item.target_rel, "France/Paris/2022/plain.jpg")

    def test_apply_moves_to_delete_folder(self):
        fid = self.add_file("dup.jpg", country="France", city="Paris")
        self.set_dedup_choice(fid, "to_delete")
        report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.assertEqual(report.moved, 1)
        self.assertTrue((self.dest / "_delete" / "dup.jpg").exists())
        self.assertEqual(self.path_of(fid), str(self.dest / "_delete" / "dup.jpg"))


class TestToDeleteDedupeExclusion(DedupDeleteTestBase):
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

    def test_to_delete_excluded_from_near_dup_grouping(self):
        # marked goes to _удалить and must neither "win" the near-duplicate group
        # with its higher resolution nor drag a normal file with it into
        # near_dup/_Duplicates — there are only two in the group, marked is
        # excluded -> the single remaining one is sorted normally.
        marked = self.add_file("marked.jpg", content=b"a" * 300,
                               country="France", city="Paris")
        normal = self.add_file("normal.jpg", content=b"b" * 100,
                               country="France", city="Paris")
        for fid in (marked, normal):
            self.set_phash(fid, "9" * 16)
        self.set_dims(marked, 2000, 2000)
        self.set_dims(normal, 100, 100)
        self.set_dedup_choice(marked, "to_delete")

        report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                               apply=False, dedupe=True)
        marked_item = self.plan_by_file(report, marked)
        normal_item = self.plan_by_file(report, normal)

        self.assertEqual(marked_item.reason, "dedup_delete")
        self.assertIsNone(marked_item.near_dup_group)
        self.assertEqual(normal_item.reason, "city")
        self.assertIsNone(normal_item.near_dup_group)


if __name__ == "__main__":
    unittest.main()
