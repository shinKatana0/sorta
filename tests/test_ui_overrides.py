"""F77: manual corrections in the web app — POST /api/overrides, the mark in the plan
response, the markup that draws it, and the reset that wipes the table.

Body validation is the interesting part: `manual_overrides.target` is the one value of
this feature that later becomes a WRITE path (sorter builds a destination from it), so
the endpoint must never let an invalid body through silently. The rule itself lives in
sorter (`manual_target_parts`; see test_sorter_overrides.TestTargetEscapeIsRefused for
the reader's half) — since F203 the route asks that same function before it stores
anything and answers 400 with a reason, so here we check that ui.py stores a valid target
as it was given and refuses the rest instead of raising.
"""
from __future__ import annotations

import json
import unittest
import urllib.parse

from sorta import db, ui

from tests.test_ui import UiServerTestBase


class OverridesTestBase(UiServerTestBase):
    def post(self, path: str, data: object) -> tuple[int, dict]:
        import urllib.error
        import urllib.request
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def rows(self) -> dict[int, tuple[str, str | None]]:
        return {r["file_id"]: (r["action"], r["target"]) for r in self.conn.execute(
            "SELECT file_id, action, target FROM manual_overrides")}


class TestOverridesPayloadValidation(OverridesTestBase):
    def setUp(self):
        super().setUp()
        self.fid, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()

    def test_unknown_action_returns_400(self):
        status, payload = self.post("/api/overrides",
                                    {"file_ids": [self.fid], "action": "delete"})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        self.assertEqual(self.rows(), {})

    def test_missing_action_returns_400(self):
        status, _payload = self.post("/api/overrides", {"file_ids": [self.fid]})
        self.assertEqual(status, 400)

    def test_empty_file_ids_returns_400(self):
        status, _payload = self.post("/api/overrides",
                                    {"file_ids": [], "action": "exclude"})
        self.assertEqual(status, 400)

    def test_missing_file_ids_returns_400(self):
        status, _payload = self.post("/api/overrides", {"action": "exclude"})
        self.assertEqual(status, 400)

    def test_non_integer_file_id_returns_400(self):
        status, _payload = self.post("/api/overrides",
                                    {"file_ids": [str(self.fid)], "action": "exclude"})
        self.assertEqual(status, 400)
        self.assertEqual(self.rows(), {})

    def test_bool_file_id_returns_400(self):
        status, _payload = self.post("/api/overrides",
                                    {"file_ids": [True], "action": "exclude"})
        self.assertEqual(status, 400)

    def test_reassign_without_target_returns_400(self):
        status, _payload = self.post("/api/overrides",
                                    {"file_ids": [self.fid], "action": "reassign"})
        self.assertEqual(status, 400)
        self.assertEqual(self.rows(), {})

    def test_reassign_with_blank_target_returns_400(self):
        status, _payload = self.post(
            "/api/overrides", {"file_ids": [self.fid], "action": "reassign",
                               "target": "   "})
        self.assertEqual(status, 400)

    def test_reassign_with_non_string_target_returns_400(self):
        status, _payload = self.post(
            "/api/overrides", {"file_ids": [self.fid], "action": "reassign",
                               "target": 5})
        self.assertEqual(status, 400)

    def test_non_dict_body_returns_400(self):
        status, _payload = self.post("/api/overrides", [self.fid])
        self.assertEqual(status, 400)

    def test_empty_body_returns_400(self):
        status, _payload = self.post("/api/overrides", None)
        self.assertEqual(status, 400)

    def test_escaping_target_is_refused_with_a_reason(self):
        # F203: the target is typed rather than picked, so the route answers the person
        # who typed it. It used to be stored here and dropped hours later by the sorter,
        # which still refuses it when it reads the row — see test_reassign_beyond_the_plan
        # for the reasons and for the rule both halves share.
        status, payload = self.post(
            "/api/overrides", {"file_ids": [self.fid], "action": "reassign",
                               "target": "../../evil"})
        self.assertEqual(status, 400)
        self.assertEqual(payload["reason"], "parent")
        self.assertEqual(self.rows(), {})


class TestOverridesWrites(OverridesTestBase):
    def test_exclude_writes_one_row_per_file(self):
        a, _p, _c = self.add_photo_file("a.jpg")
        b, _p2, _c2 = self.add_photo_file("b.jpg")
        self.start_server()
        status, payload = self.post("/api/overrides",
                                    {"file_ids": [a, b], "action": "exclude"})
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(sorted(payload["file_ids"]), sorted([a, b]))
        self.assertEqual(self.rows(), {a: ("exclude", None), b: ("exclude", None)})

    def test_reassign_stores_the_target(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        status, _payload = self.post(
            "/api/overrides", {"file_ids": [fid], "action": "reassign",
                               "target": " Франция/Париж/2014 "})
        self.assertEqual(status, 200)
        self.assertEqual(self.rows(), {fid: ("reassign", "Франция/Париж/2014")})

    def test_repeated_correction_overwrites_the_row_instead_of_adding_one(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        self.post("/api/overrides", {"file_ids": [fid], "action": "exclude"})
        self.post("/api/overrides", {"file_ids": [fid], "action": "reassign",
                                     "target": "Италия/Рим"})
        self.post("/api/overrides", {"file_ids": [fid], "action": "reassign",
                                     "target": "Франция/Париж"})
        count = self.conn.execute(
            "SELECT COUNT(*) FROM manual_overrides").fetchone()[0]
        self.assertEqual(count, 1)
        self.assertEqual(self.rows(), {fid: ("reassign", "Франция/Париж")})

    def test_clear_deletes_the_row(self):
        a, _p, _c = self.add_photo_file("a.jpg")
        b, _p2, _c2 = self.add_photo_file("b.jpg")
        self.start_server()
        self.post("/api/overrides", {"file_ids": [a, b], "action": "exclude"})
        status, payload = self.post("/api/overrides",
                                    {"file_ids": [a], "action": "clear"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["action"], "clear")
        self.assertEqual(self.rows(), {b: ("exclude", None)})

    def test_unknown_file_id_is_ignored_not_inserted(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        status, payload = self.post("/api/overrides",
                                    {"file_ids": [fid, 999999], "action": "exclude"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["file_ids"], [fid])
        self.assertEqual(self.rows(), {fid: ("exclude", None)})

    def test_only_unknown_ids_writes_nothing(self):
        self.start_server()
        status, payload = self.post("/api/overrides",
                                    {"file_ids": [999999], "action": "exclude"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["file_ids"], [])
        self.assertEqual(self.rows(), {})

    def test_duplicate_ids_in_the_body_collapse(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        status, payload = self.post("/api/overrides",
                                    {"file_ids": [fid, fid], "action": "exclude"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["file_ids"], [fid])


class TestPlanCarriesTheMark(OverridesTestBase):
    def _page(self, category: str) -> dict:
        _s, body, _c = self.get(
            "/api/plan?mode=city&category=" + urllib.parse.quote(category))
        return json.loads(body)

    def _aggregate(self) -> dict:
        _s, body, _c = self.get("/api/plan?mode=city")
        return json.loads(body)

    def test_page_marks_corrected_files_only(self):
        marked, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        plain, _p2, _c2 = self.add_photo_file("b.jpg", country="ru", city="Moscow")
        self.start_server()
        self.post("/api/overrides", {"file_ids": [marked], "action": "exclude"})
        category = self._aggregate()["categories"][0]["category"]
        items = {it["file_id"]: it for it in self._page(category)["items"]}
        self.assertEqual(items[marked]["override"], "exclude")
        self.assertIsNone(items[marked]["override_target"])
        self.assertNotIn("override", items[plain])
        self.assertNotIn("override_target", items[plain])

    def test_page_marks_a_reassigned_file_with_its_target(self):
        marked, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        self.post("/api/overrides", {"file_ids": [marked], "action": "reassign",
                                     "target": "Франция/Париж/2014"})
        category = self._aggregate()["categories"][0]["category"]
        item = self._page(category)["items"][0]
        self.assertEqual(item["override"], "reassign")
        self.assertEqual(item["override_target"], "Франция/Париж/2014")

    def test_mark_appears_without_rebuilding_the_plan(self):
        # The plan is built (and cached) BEFORE the correction — the mark still shows
        # up, because it is read live per request rather than baked into the plan.
        marked, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        category = self._aggregate()["categories"][0]["category"]
        self.assertNotIn("override", self._page(category)["items"][0])
        self.post("/api/overrides", {"file_ids": [marked], "action": "exclude"})
        self.assertEqual(self._page(category)["items"][0]["override"], "exclude")

    def test_cleared_mark_disappears_from_the_page(self):
        marked, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        self.post("/api/overrides", {"file_ids": [marked], "action": "exclude"})
        category = self._aggregate()["categories"][0]["category"]
        self.post("/api/overrides", {"file_ids": [marked], "action": "clear"})
        self.assertNotIn("override", self._page(category)["items"][0])

    def test_aggregate_counts_corrections_and_separates_the_excluded(self):
        excluded, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        moved, _p2, _c2 = self.add_photo_file("b.jpg", country="ru", city="Moscow")
        self.add_photo_file("c.jpg", country="fr", city="Paris")
        self.start_server()
        self.post("/api/overrides", {"file_ids": [excluded], "action": "exclude"})
        self.post("/api/overrides", {"file_ids": [moved], "action": "reassign",
                                     "target": "Франция/Париж"})
        data = self._aggregate()
        self.assertEqual(data["total"], 3)
        self.assertEqual(data["overridden"], 2)
        # the "leave alone" frame is listed but will not be moved — the apply
        # confirmation counts total minus excluded
        self.assertEqual(data["excluded"], 1)

    def test_aggregate_reports_zero_without_corrections(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        data = self._aggregate()
        self.assertEqual(data["overridden"], 0)
        self.assertEqual(data["excluded"], 0)

    def test_excluded_file_stays_listed_so_the_mark_can_be_removed(self):
        # It is not moved by the layout (that is the sorter's job), but it must remain
        # visible in the UI — otherwise a marked frame disappears with no way back.
        marked, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        self.post("/api/overrides", {"file_ids": [marked], "action": "exclude"})
        data = self._aggregate()
        self.assertEqual(data["total"], 1)
        item = self._page(data["categories"][0]["category"])["items"][0]
        self.assertEqual(item["file_id"], marked)
        self.assertEqual(item["override"], "exclude")
        self.assertEqual(item["reason"], "manual_exclude")


class TestOverridesMarkup(OverridesTestBase):
    def setUp(self):
        super().setUp()
        self.start_server()
        _status, body, _ctype = self.get("/")
        self.html = body.decode("utf-8")

    def test_excluded_and_reassigned_have_different_classes_and_styles(self):
        self.assertIn("tr.override-exclude", self.html)
        self.assertIn("tr.override-reassign", self.html)
        # two DIFFERENT visual states — the same frame for both would make "left alone"
        # and "moved" indistinguishable
        self.assertIn("outline: 2px solid var(--danger)", self.html)
        self.assertIn("outline: 2px dashed var(--accent)", self.html)

    def test_controls_for_selection_folder_and_single_row_are_present(self):
        self.assertIn('id="city-override-exclude-btn"', self.html)
        self.assertIn('id="city-override-move-btn"', self.html)
        self.assertIn('id="city-override-clear-btn"', self.html)
        self.assertIn('id="city-override-target"', self.html)
        self.assertIn("override-folder-btn", self.html)
        self.assertIn("override-row-btn", self.html)
        self.assertIn("/api/overrides", self.html)
        # the selection is the same one the bulk delete uses
        self.assertIn("wireOverrideControls(", self.html)
        self.assertIn(".row-select:checked", self.html)

    def test_move_targets_come_from_the_loaded_plan_aggregate(self):
        self.assertIn("fillOverrideTargets(categories)", self.html)

    def test_no_external_resources_added(self):
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)
        self.assertNotIn("<link", self.html)


class TestOverrideStringsAreTranslated(unittest.TestCase):
    KEYS = ("override_exclude_button", "override_clear_button", "override_move_button",
            "override_target_placeholder", "override_exclude_folder_button",
            "override_exclude_folder_confirm", "override_excluded_mark",
            "override_reassigned_mark", "override_hint",
            "override_alert_choose_target", "override_error_prefix")

    def test_every_new_string_exists_in_all_three_languages(self):
        for key in self.KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")


class TestResetWipesOverrides(OverridesTestBase):
    def test_reset_index_clears_manual_overrides(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        self.post("/api/overrides", {"file_ids": [fid], "action": "exclude"})
        self.assertEqual(len(self.rows()), 1)
        db.reset_index(self.conn)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM manual_overrides").fetchone()[0], 0)

    def test_start_over_endpoint_clears_manual_overrides(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.start_server()
        self.post("/api/overrides", {"file_ids": [fid], "action": "exclude"})
        status, _payload = self.post("/api/process/reset", {})
        self.assertEqual(status, 200)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM manual_overrides").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
