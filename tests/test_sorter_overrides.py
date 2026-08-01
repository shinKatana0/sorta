"""F77: manual corrections from the web app applied by the layout.

Inherits the SorterTestBase fixtures from test_sorter.py. All FS operations — on
tmp_path only. The corrections themselves are written straight into
`manual_overrides` here: the sorter is the READER of that table, the writer is ui.py
(see test_ui_overrides.py).
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from tests.test_sorter import SorterTestBase

from sorta import i18n
from sorta.sorter import _manual_target_parts, plan_and_sort


class OverridesTestBase(SorterTestBase):
    def override(self, file_id: int, action: str, target: str | None = None) -> None:
        self.conn.execute(
            """INSERT INTO manual_overrides (file_id, action, target, updated_at)
               VALUES (?, ?, ?, '2026-07-26')
               ON CONFLICT(file_id) DO UPDATE SET
                   action = excluded.action, target = excluded.target""",
            (file_id, action, target),
        )
        self.conn.commit()

    def plan(self, mode: str = "city", **kwargs) -> object:
        with redirect_stdout(io.StringIO()):
            return plan_and_sort(self.cfg, self.conn, mode, self.dest, apply=False,
                                 **kwargs)

    def targets(self, report) -> dict[int, str]:
        return {it.file_id: it.target_rel for it in report.plan}


class TestExcludeOverride(OverridesTestBase):
    def test_excluded_file_is_absent_from_the_plan(self):
        left = self.add_file("Париж/a.jpg", country="FR", city="Paris")
        other = self.add_file("Париж/b.jpg", country="FR", city="Paris")
        self.override(left, "exclude")
        report = self.plan()
        self.assertEqual([it.file_id for it in report.plan], [other])
        self.assertEqual(report.manual_excluded, 1)

    def test_other_files_are_laid_out_exactly_as_without_the_override(self):
        kept = self.add_file("b.jpg", country="FR", city="Paris")
        before = self.targets(self.plan())
        dropped = self.add_file("a.jpg", country="FR", city="Paris")
        self.override(dropped, "exclude")
        after = self.targets(self.plan())
        self.assertEqual(after, before)
        self.assertIn(kept, after)

    def test_exclude_is_counted_separately_from_exclude_dirs(self):
        # Two different mechanisms — one report number for both would hide which of
        # them dropped a file.
        marked = self.add_file("a.jpg", country="FR", city="Paris")
        self.add_file("Готовое/b.jpg", country="FR", city="Paris")
        self.add_file("c.jpg", country="FR", city="Paris")
        self.override(marked, "exclude")
        report = self.plan(exclude=[str(self.src_dir / "Готовое")])
        self.assertEqual(report.manual_excluded, 1)
        self.assertEqual(report.excluded, 1)
        self.assertEqual(len(report.plan), 1)

    def test_excluded_in_every_mode(self):
        marked = self.add_file("a.jpg", country="FR", city="Paris")
        self.add_person(marked, "Аня")
        self.override(marked, "exclude")
        for mode in ("city", "person", "event"):
            with self.subTest(mode=mode):
                report = self.plan(mode)
                self.assertEqual(report.plan, [])
                self.assertEqual(report.manual_excluded, 1)

    def test_excluded_file_is_not_moved_by_apply(self):
        marked = self.add_file("a.jpg", country="FR", city="Paris")
        src = self.conn.execute(
            "SELECT path FROM files WHERE id = ?", (marked,)).fetchone()["path"]
        self.override(marked, "exclude")
        with redirect_stdout(io.StringIO()):
            report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True)
        self.assertEqual(report.moved, 0)
        self.assertTrue(self.src_dir.joinpath("a.jpg").is_file())
        self.assertEqual(
            self.conn.execute("SELECT path FROM files WHERE id = ?",
                              (marked,)).fetchone()["path"], src)


class TestExcludePreviewFlag(OverridesTestBase):
    """keep_manual_excluded — the web app's preview keeps the marked frames visible.

    The sorting plan must not contain them (they are not moved); the preview must, or a
    marked frame would vanish from the UI grid with no way to unmark it.
    """

    def test_preview_keeps_the_file_with_its_automatic_target(self):
        marked = self.add_file("a.jpg", country="FR", city="Paris")
        report = self.plan(keep_manual_excluded=True)  # no override yet
        self.assertEqual(report.plan[0].reason, "city")
        self.override(marked, "exclude")
        report = self.plan(keep_manual_excluded=True)
        item = report.plan[0]
        self.assertEqual(item.file_id, marked)
        self.assertEqual(item.reason, "manual_exclude")
        self.assertEqual(item.target_rel, "France/Paris/2022/a.jpg")
        self.assertEqual(report.manual_excluded, 1)

    def test_preview_flag_is_ignored_when_applying(self):
        marked = self.add_file("a.jpg", country="FR", city="Paris")
        self.override(marked, "exclude")
        with redirect_stdout(io.StringIO()):
            report = plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=True,
                                   keep_manual_excluded=True)
        self.assertEqual(report.plan, [])
        self.assertEqual(report.moved, 0)
        self.assertEqual(report.manual_excluded, 1)
        self.assertTrue(self.src_dir.joinpath("a.jpg").is_file())

    def test_preview_kept_file_is_not_pulled_into_a_near_dup_group(self):
        marked = self.add_file("a.jpg", content=b"a" * 400, country="FR", city="Paris")
        other = self.add_file("b.jpg", content=b"b" * 100, country="FR", city="Paris")
        for fid in (marked, other):
            self.conn.execute("UPDATE files SET phash = ? WHERE id = ?", ("0" * 16, fid))
        self.conn.execute("UPDATE files SET width = 4000, height = 3000 WHERE id = ?",
                          (marked,))
        self.conn.execute("UPDATE files SET width = 400, height = 300 WHERE id = ?",
                          (other,))
        self.conn.commit()
        self.override(marked, "exclude")
        report = self.plan(dedupe=True, keep_manual_excluded=True)
        by_id = {it.file_id: it for it in report.plan}
        self.assertIsNone(by_id[marked].near_dup_role)
        # the group is left with a single file, so `other` keeps its normal layout
        # instead of losing to a frame that is not going anywhere
        self.assertEqual(by_id[other].target_rel, "France/Paris/2022/b.jpg")


class TestReassignOverride(OverridesTestBase):
    def test_reassign_puts_the_file_into_the_chosen_folder(self):
        fid = self.add_file("Несортированное/a.jpg")
        self.override(fid, "reassign", "Франция/Париж/2014")
        report = self.plan()
        item = report.plan[0]
        self.assertEqual(item.target_rel, "Франция/Париж/2014/a.jpg")
        self.assertEqual(item.dst, self.dest / "Франция" / "Париж" / "2014" / "a.jpg")
        self.assertEqual(item.reason, "manual_reassign")
        self.assertEqual(report.manual_reassigned, 1)

    def test_name_conflict_inside_the_target_gets_a_suffix(self):
        first = self.add_file("one/a.jpg", content=b"first")
        second = self.add_file("two/a.jpg", content=b"second")
        self.override(first, "reassign", "Франция/Париж")
        self.override(second, "reassign", "Франция/Париж")
        report = self.plan()
        names = sorted(it.dst.name for it in report.plan)
        self.assertEqual(names, ["a.jpg", "a_1.jpg"])
        self.assertEqual({it.file_id for it in report.plan}, {first, second})

    def test_existing_file_on_disk_gets_a_suffix(self):
        occupied = self.dest / "Франция"
        occupied.mkdir(parents=True)
        (occupied / "a.jpg").write_bytes(b"already there")
        fid = self.add_file("one/a.jpg")
        self.override(fid, "reassign", "Франция")
        report = self.plan()
        self.assertEqual(report.plan[0].dst.name, "a_1.jpg")

    def test_reassign_beats_document_verdict(self):
        fid = self.add_file("a.jpg", junk_verdict="document")
        self.override(fid, "reassign", "Франция/Париж/2014")
        report = self.plan()
        self.assertEqual(report.plan[0].target_rel, "Франция/Париж/2014/a.jpg")
        self.assertEqual(report.plan[0].reason, "manual_reassign")

    def test_reassign_beats_screenshot_verdict(self):
        fid = self.add_file("a.jpg", junk_verdict="screenshot")
        self.override(fid, "reassign", "Франция/Париж/2014")
        report = self.plan()
        self.assertEqual(report.plan[0].target_rel, "Франция/Париж/2014/a.jpg")

    def test_reassign_beats_dedup_delete_and_not_personal(self):
        fid = self.add_file("a.jpg")
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at) "
            "VALUES (?, 'to_delete', '2026-07-26')", (fid,))
        self.conn.execute("UPDATE files SET not_personal = 1 WHERE id = ?", (fid,))
        self.conn.commit()
        self.override(fid, "reassign", "Франция/Париж/2014")
        report = self.plan()
        self.assertEqual(report.plan[0].target_rel, "Франция/Париж/2014/a.jpg")

    def test_reassign_beats_near_dup_role(self):
        # Without the correction `worse` would go to the _Duplicates folder as the
        # loser of the group — the human's decision wins.
        best = self.add_file("best.jpg", content=b"a" * 400, country="FR", city="Paris")
        worse = self.add_file("worse.jpg", content=b"b" * 100, country="FR", city="Paris")
        for fid in (best, worse):
            self.conn.execute("UPDATE files SET phash = ? WHERE id = ?", ("0" * 16, fid))
        self.conn.execute("UPDATE files SET width = 4000, height = 3000 WHERE id = ?",
                          (best,))
        self.conn.execute("UPDATE files SET width = 400, height = 300 WHERE id = ?",
                          (worse,))
        self.conn.commit()
        self.override(worse, "reassign", "Франция/Париж/2014")
        report = self.plan(dedupe=True)
        by_id = {it.file_id: it for it in report.plan}
        self.assertEqual(by_id[worse].target_rel, "Франция/Париж/2014/worse.jpg")
        self.assertEqual(by_id[worse].reason, "manual_reassign")
        self.assertIsNone(by_id[worse].near_dup_role)

    def test_reassign_in_person_and_event_modes(self):
        fid = self.add_file("a.jpg")
        self.override(fid, "reassign", "Франция/Париж/2014")
        for mode in ("person", "event"):
            with self.subTest(mode=mode):
                report = self.plan(mode)
                self.assertEqual(report.plan[0].target_rel, "Франция/Париж/2014/a.jpg")


class TestTargetEscapeIsRefused(OverridesTestBase):
    ESCAPES = ["../../evil", "..", "../evil", "C:/windows", "/etc", "..\\..\\x",
               "Франция\\Париж", "//server/share", "  ", "", "./../x"]

    def test_escaping_target_is_ignored_and_the_file_is_laid_out_automatically(self):
        for i, raw in enumerate(self.ESCAPES):
            with self.subTest(target=raw):
                fid = self.add_file(f"esc{i}/a.jpg", country="FR", city="Paris")
                self.override(fid, "reassign", raw)
                with self.assertLogs("sorta.sorter", level="WARNING") as logs:
                    report = self.plan()
                item = {it.file_id: it for it in report.plan}[fid]
                self.assertEqual(item.reason, "city")
                # the automatic city folder, not the target from the DB (the file name
                # may carry a _N suffix — the earlier files of the loop live there too)
                self.assertEqual(item.target_rel.rsplit("/", 1)[0], "France/Paris/2022")
                self.assertEqual(report.manual_reassigned, 0)
                self.assertTrue(any("ручная правка проигнорирована" in m
                                    for m in logs.output))

    def test_no_plan_destination_ever_leaves_the_sort_root(self):
        for i, raw in enumerate(self.ESCAPES):
            self.override(self.add_file(f"esc{i}/a.jpg", country="FR", city="Paris"),
                          "reassign", raw)
        with self.assertLogs("sorta.sorter", level="WARNING"):
            report = self.plan()
        self.assertEqual(len(report.plan), len(self.ESCAPES))
        for item in report.plan:
            with self.subTest(dst=item.dst):
                self.assertTrue(item.dst.resolve().is_relative_to(self.dest.resolve()))

    def test_validator_accepts_a_plain_relative_posix_target(self):
        self.assertEqual(_manual_target_parts("Франция/Париж/2014", "a.jpg"),
                         ["Франция", "Париж", "2014"])
        self.assertEqual(_manual_target_parts(" Франция / Париж ", "a.jpg"),
                         ["Франция", "Париж"])

    def test_validator_sanitizes_segments_like_the_rest_of_the_layout(self):
        self.assertEqual(_manual_target_parts("Па*ри?ж", "a.jpg"), ["Па_ри_ж"])

    def test_validator_refuses_non_string_target(self):
        self.assertIsNone(_manual_target_parts(None, "a.jpg"))


class TestNoOverridesRegression(OverridesTestBase):
    def test_files_without_a_correction_keep_the_f5_layout(self):
        city = self.add_file("a.jpg", country="FR", city="Paris")
        document = self.add_file("b.jpg", junk_verdict="document")
        # camera_make — F78: an undated file without a camera trace lands in
        # `downloaded`; this one is here to cover the low_date branch.
        undated = self.add_file("c.jpg", taken_at=None, camera_make="Apple")
        person_file = self.add_file("d.jpg", country="FR", city="Paris")
        self.add_person(person_file, "Аня")
        expected = {
            "city": {
                city: "France/Paris/2022/a.jpg",
                document: "_Documents/b.jpg",
                undated: "_Unsorted/low_date/c.jpg",
                person_file: "France/Paris/2022/d.jpg",
            },
            "person": {
                city: "_Unsorted/no_faces/a.jpg",
                document: "_Documents/b.jpg",
                undated: "_Unsorted/low_date/c.jpg",
                person_file: "Аня/2022/d.jpg",
            },
            "event": {
                city: "2022/05/a.jpg",
                document: "_Documents/b.jpg",
                undated: "_Unsorted/low_date/c.jpg",
                person_file: "2022/05/d.jpg",
            },
        }
        for mode, targets in expected.items():
            with self.subTest(mode=mode):
                report = self.plan(mode)
                self.assertEqual(self.targets(report), targets)
                self.assertEqual(report.manual_excluded, 0)
                self.assertEqual(report.manual_reassigned, 0)
                self.assertNotIn("manual_reassign", [it.reason for it in report.plan])

    def test_city_layout_of_untouched_files_matches_a_run_before_any_override(self):
        a = self.add_file("a.jpg", country="FR", city="Paris")
        b = self.add_file("b.jpg", country="FR", city="Paris")
        before = self.targets(self.plan())
        marked = self.add_file("c.jpg", country="FR", city="Paris")
        self.override(marked, "reassign", "Италия/Рим/2019")
        after = self.targets(self.plan())
        self.assertEqual(after[a], before[a])
        self.assertEqual(after[b], before[b])
        self.assertEqual(after[marked], "Италия/Рим/2019/c.jpg")

    def test_summary_line_reports_manual_corrections(self):
        excluded = self.add_file("a.jpg", country="FR", city="Paris")
        moved = self.add_file("b.jpg", country="FR", city="Paris")
        self.override(excluded, "exclude")
        self.override(moved, "reassign", "Италия/Рим")
        buf = io.StringIO()
        with redirect_stdout(buf):
            plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        # F118: the note follows `language:` now. Both numbers still matter — that is
        # the point of the case (F77 reports corrections apart from the exclude count).
        self.assertIn(
            i18n.cli_text("cli.sort.plan_manual", "en", reassigned=1, excluded=1),
            buf.getvalue())

    def test_summary_line_stays_silent_without_corrections(self):
        self.add_file("a.jpg", country="FR", city="Paris")
        buf = io.StringIO()
        with redirect_stdout(buf):
            plan_and_sort(self.cfg, self.conn, "city", self.dest, apply=False)
        self.assertNotIn(
            i18n.cli_text("cli.sort.plan_manual", "en", reassigned=0, excluded=0)
            .split("{")[0].split(":")[0],
            buf.getvalue())


class TestClearedOverride(OverridesTestBase):
    def test_deleting_the_row_returns_the_file_to_the_automatic_layout(self):
        fid = self.add_file("a.jpg", country="FR", city="Paris")
        self.override(fid, "reassign", "Италия/Рим/2019")
        self.assertEqual(self.plan().plan[0].target_rel, "Италия/Рим/2019/a.jpg")
        self.conn.execute("DELETE FROM manual_overrides WHERE file_id = ?", (fid,))
        self.conn.commit()
        report = self.plan()
        self.assertEqual(report.plan[0].target_rel, "France/Paris/2022/a.jpg")
        self.assertEqual(report.plan[0].reason, "city")
        self.assertEqual(report.manual_reassigned, 0)

    def test_clearing_an_exclude_brings_the_file_back_into_the_plan(self):
        fid = self.add_file("a.jpg", country="FR", city="Paris")
        self.override(fid, "exclude")
        self.assertEqual(self.plan().plan, [])
        self.conn.execute("DELETE FROM manual_overrides WHERE file_id = ?", (fid,))
        self.conn.commit()
        report = self.plan()
        self.assertEqual([it.file_id for it in report.plan], [fid])
        self.assertEqual(report.manual_excluded, 0)


if __name__ == "__main__":
    unittest.main()
