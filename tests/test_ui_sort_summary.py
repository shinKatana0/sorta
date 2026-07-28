"""F104: the summary before a layout, and the action row it was extracted from.

The dialog in front of "Apply" has one job — to name NUMBERS. "Are you sure?" is a
question nobody can answer: 2026-07-28 one click on this screen wrote 140.9 GB of
duplicate copies, and while the cause of that was elsewhere (F97 fixed it), the reason
it went unnoticed was that nothing on the way in had to state what was about to happen.

So what is pinned here is that the summary counts the SAME plan the tab draws (files,
folders, volume, the review folders) plus what is already lying in the destination —
decided by the rule the apply itself uses — and that an empty plan produces a dead
button with an explanation instead of a dialog full of zeroes.

The second half is the row itself: no roll-back button on this tab (the manifest that
says what would be rolled back lives on "Moves"), a cancel button that exists only
while a layout runs, and deleting moved into the context of a selection.
"""
from __future__ import annotations

import io
import json
import shutil
import unittest
import urllib.parse
from contextlib import redirect_stdout
from pathlib import Path

from sorta import ui
from sorta.sorter import plan_and_sort

from tests.test_ui import UiServerTestBase

_F104_SUMMARY_KEYS = (
    "sort_confirm_title", "sort_confirm_ok", "sort_confirm_cancel",
    "sort_summary_dest", "sort_summary_mode_move", "sort_summary_mode_copy",
    "sort_summary_files", "sort_summary_existing", "sort_summary_existing_none",
    "sort_summary_existing_unknown", "sort_summary_service", "sort_summary_empty",
    "sort_summary_error", "selection_delete_hint",
)


class SummaryTestBase(UiServerTestBase):
    def setUp(self):
        super().setUp()
        self.dest = self.root / "dest"

    def summary(self, dest: Path | str | None = None) -> dict:
        query = "" if dest is None else (
            "?dest=" + urllib.parse.quote(str(dest)))
        status, body, ctype = self.get("/api/sort/summary" + query)
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        return json.loads(body)

    def add_classified(self, rel: str, verdict: str) -> int:
        file_id, _path, _content = self.add_photo_file(rel, country="ru", city="Moscow")
        self.conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, score, updated_at, tier)
               VALUES (?, ?, 'vlm', NULL, '2026-07-28', 'vlm')""",
            (file_id, verdict))
        self.conn.commit()
        return file_id

    def plan(self) -> list:
        """The dry-run plan for the same dest — the yardstick the summary must match."""
        with redirect_stdout(io.StringIO()):
            report = plan_and_sort(self.cfg, self.conn, "city", self.dest,
                                   apply=False, write_reports=False)
        return report.plan

    def place_at_target(self, file_id: int, content: bytes | None = None) -> Path:
        """Put a file where the plan would put `file_id` — its own bytes, or others'."""
        item = next(it for it in self.plan() if it.file_id == file_id)
        target = self.dest.joinpath(*item.target_rel.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            shutil.copyfile(item.src, target)
        else:
            target.write_bytes(content)
        return target


class TestSummaryCountsThePlan(SummaryTestBase):
    def test_files_folders_and_volume_are_the_plan_s_own_numbers(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.add_photo_file("b.jpg", country="ru", city="Moscow")
        self.add_photo_file("c.jpg", country="th", city="Phuket")
        self.start_server()
        data = self.summary(self.dest)
        plan = self.plan()
        self.assertEqual(data["files"], len(plan))
        self.assertEqual(data["dirs"],
                         len({it.target_rel.rpartition("/")[0] for it in plan}))
        sizes = dict(self.conn.execute("SELECT id, size FROM files").fetchall())
        self.assertEqual(data["bytes"], sum(sizes[it.file_id] for it in plan))
        self.assertEqual(data["dest"], str(self.dest))
        self.assertEqual(data["mode"], "city")

    def test_it_agrees_with_the_aggregate_the_tab_renders(self):
        """The tab shows one set of numbers and the dialog another only if somebody
        counted twice; both come off the same built plan on purpose."""
        for name in ("a.jpg", "b.jpg"):
            self.add_photo_file(name, country="ru", city="Moscow")
        self.start_server()
        _s, body, _c = self.get("/api/plan?mode=city")
        aggregate = json.loads(body)
        data = self.summary(self.dest)
        self.assertEqual(data["files"], aggregate["total"] - aggregate["excluded"])
        self.assertEqual(data["dirs"], len(aggregate["categories"]))

    def test_files_left_alone_are_not_counted(self):
        """F77: a frame marked "leave alone" stays VISIBLE in the tree but does not
        move — counting it would promise a transfer that will not happen."""
        keep = self.add_photo_file("a.jpg", country="ru", city="Moscow")[0]
        self.add_photo_file("b.jpg", country="ru", city="Moscow")
        self.start_server()
        before = self.summary(self.dest)
        status, _resp = self._post("/api/overrides",
                                   {"file_ids": [keep], "action": "exclude"})
        self.assertEqual(status, 200)
        after = self.summary(self.dest)
        self.assertEqual(after["files"], before["files"] - 1)
        self.assertLess(after["bytes"], before["bytes"])

    def test_the_review_folders_are_counted_separately(self):
        """"224 GB into Country/City" and "3 000 frames into _Products" are different
        pieces of news, and the second is the one that surprises people."""
        self.add_photo_file("keep.jpg", country="ru", city="Moscow")
        self.add_classified("goods.jpg", "product")
        self.add_classified("passport.jpg", "document")
        self.start_server()
        data = self.summary(self.dest)
        self.assertEqual(data["products"], 1)
        self.assertEqual(data["documents"], 1)
        self.assertEqual(data["files"], 3)

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())


class TestSummaryLooksIntoTheDestination(SummaryTestBase):
    def test_an_empty_destination_reports_nothing_there(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        data = self.summary(self.dest)
        self.assertEqual(data["dest_existing"], 0)
        self.assertEqual(data["dest_same"], 0)

    def test_an_identical_copy_already_there_is_reported_as_skipped(self):
        """F97 knowledge: a second apply into the same destination does not re-copy
        what it copied last time. The dialog has to say so BEFORE the run, otherwise
        "12 000 files" reads as twelve thousand transfers that will not happen."""
        first = self.add_photo_file("a.jpg", country="ru", city="Moscow")[0]
        self.add_photo_file("b.jpg", country="ru", city="Moscow")
        self.place_at_target(first)
        self.start_server()
        data = self.summary(self.dest)
        self.assertEqual(data["dest_existing"], 1)
        self.assertEqual(data["dest_same"], 1)

    def test_a_different_file_under_the_same_name_is_not_a_match(self):
        """The name is taken but the bytes are somebody else's: that file WILL be
        written, next to the other one, and counting it as skipped would be a lie."""
        first = self.add_photo_file("a.jpg", country="ru", city="Moscow")[0]
        self.place_at_target(first, content=b"not our jpeg at all")
        self.start_server()
        data = self.summary(self.dest)
        self.assertEqual(data["dest_existing"], 1)
        self.assertEqual(data["dest_same"], 0)

    def test_the_numbers_follow_the_destination_that_was_asked_about(self):
        first = self.add_photo_file("a.jpg", country="ru", city="Moscow")[0]
        self.place_at_target(first)
        self.start_server()
        self.assertEqual(self.summary(self.dest)["dest_same"], 1)
        other = self.root / "elsewhere"
        self.assertEqual(self.summary(other)["dest_same"], 0)
        self.assertEqual(self.summary(other)["dest"], str(other))

    def test_an_empty_destination_field_means_the_single_source(self):
        """F28: no destination is the in-place layout, whose root the sorter takes
        from the one configured source — the summary resolves it the same way."""
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        self.assertEqual(self.summary("")["dest"], str(self.src_dir))

    def test_with_several_sources_the_destination_side_says_unknown(self):
        """Refusing to answer is the honest option: `plan_and_sort` would refuse to
        guess a common root too, and inventing numbers here is worse than a blank."""
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.cfg.sources = [self.src_dir, self.root / "second"]
        self.start_server()
        data = self.summary("")
        self.assertIsNone(data["dest"])
        self.assertEqual(data["dest_existing"], 0)
        self.assertGreater(data["files"], 0)  # the plan side is still answered


class TestEmptyPlan(SummaryTestBase):
    def test_an_empty_index_summarizes_to_zero_without_an_error(self):
        self.start_server()
        data = self.summary(self.dest)
        self.assertEqual(data["files"], 0)
        self.assertEqual(data["bytes"], 0)

    def test_the_start_button_is_dead_until_the_plan_says_otherwise(self):
        """Requirement: an empty plan disables the start button WITH an explanation.
        The markup starts disabled (before the first plan there is no answer yet) and
        the hint is shown only once a plan has actually come back empty."""
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertRegex(html, r'<button[^>]*id="sort-apply-btn"[^>]*disabled')
        self.assertIn('id="sort-empty-hint"', html)
        self.assertIn("cityPlanLoaded && cityPlanCount === 0", html)
        self.assertIn("applyBtn.disabled = busy || cityPlanCount === 0", html)
        # and the click never opens the dialog for an empty plan
        self.assertIn("if (!cityPlanCount) return;", html)


class TestTheActionRowIsAboutStarting(SummaryTestBase):
    def setUp(self):
        super().setUp()
        self.html = ui._render_index_html("en")

    def _sort_controls(self) -> str:
        return self.html.split('class="sort-controls"', 1)[1].split("</div>", 1)[0]

    def test_the_row_keeps_only_destination_mode_and_start(self):
        row = self._sort_controls()
        for control in ("sort-dest", "sort-browse-btn", "sort-mode", "sort-apply-btn"):
            self.assertIn(control, row)
        for gone in ("folder-lang-select", "sort-undo-btn", "city-delete-selected-btn"):
            self.assertNotIn(gone, row)

    def test_roll_back_is_not_on_this_tab_at_all(self):
        """It lives on "Moves", next to the manifest that says WHAT is rolled back.
        A rollback from the plan screen is a rollback blind."""
        city = self.html.split('id="tab-city"', 1)[1].split("<section", 1)[0]
        self.assertNotIn("sort-undo-btn", city)
        self.assertNotIn("undo-btn", city)
        # the hint pointing at the tab that has it stays (F97)
        self.assertIn("sort_undo_hint", self.html)

    def test_cancel_is_contextual(self):
        self.assertRegex(self.html,
                         r'<button[^>]*id="sort-cancel-btn"[^>]*style="display:none"')
        self.assertIn('cancelBtn.style.display = data.running ? "" : "none";', self.html)

    def test_delete_selected_appears_with_a_selection(self):
        city = self.html.split('id="tab-city"', 1)[1].split("<section", 1)[0]
        bar = city.split('id="city-selection-controls"', 1)[1].split("</div>", 1)[0]
        self.assertIn("city-delete-selected-btn", bar)
        self.assertIn('id="city-selection-controls" style="display:none"', city)
        self.assertIn('barEl.style.display = n === 0 ? "none" : "";', self.html)
        # and it is wired to the tree that owns the checkboxes
        self.assertIn('"city-selection-controls"', self.html)


class TestSummaryDialogMarkup(SummaryTestBase):
    def setUp(self):
        super().setUp()
        self.html = ui._render_index_html("en")

    def test_the_dialog_exists_and_is_filled_from_the_endpoint(self):
        self.assertIn('id="sort-dialog"', self.html)
        self.assertIn('id="sort-dialog-list"', self.html)
        self.assertIn('id="sort-dialog-ok"', self.html)
        self.assertIn('id="sort-dialog-cancel"', self.html)
        self.assertIn("/api/sort/summary?dest=", self.html)

    def test_the_mode_is_stated_in_words_not_as_a_flag(self):
        for key in ("sort_summary_mode_move", "sort_summary_mode_copy"):
            self.assertIn(key, self.html)
        self.assertIn("original", ui._UI_STRINGS["sort_summary_mode_copy"]["en"])

    def test_the_layout_starts_only_from_the_dialog(self):
        """The old path was window.confirm inside the click handler; the new one has
        to go through the dialog's own OK, or the numbers are decoration."""
        self.assertIn('document.getElementById("sort-dialog-ok")', self.html)
        self.assertNotIn("sortConfirmText", self.html)


class TestSummaryStringsAreTranslated(unittest.TestCase):
    def test_every_new_string_exists_in_all_three_languages(self):
        for key in _F104_SUMMARY_KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")

    def test_the_counted_placeholders_survive_translation(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                files = ui._UI_STRINGS["sort_summary_files"][lang]
                for token in ("{n}", "{dirs}", "{size}"):
                    self.assertIn(token, files)
                existing = ui._UI_STRINGS["sort_summary_existing"][lang]
                self.assertIn("{n}", existing)
                self.assertIn("{same}", existing)
                self.assertIn("{dest}", ui._UI_STRINGS["sort_summary_dest"][lang])
                service = ui._UI_STRINGS["sort_summary_service"][lang]
                self.assertIn("{products}", service)
                self.assertIn("{documents}", service)


if __name__ == "__main__":
    unittest.main()
