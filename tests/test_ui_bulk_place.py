"""F85c (part 2): assigning a place to a whole group from the web app.

Six thousand files of the live collection have no place signal at all — no GPS, no
neighbour in time with one, no landmark, nothing readable in a folder name. No model
will place them, because the information is not in them; it is in the person who took
them. What stands between that person and a correct place is the six thousand clicks,
so the feature is a cheap BULK assignment: a group they already think in (a whole event,
a whole source folder), a place from the bundled base, one action.

The properties pinned here are the ones that make it safe to hand a user:
* the place lands in `manual_places`, never in `places` — that table has one writer and
  is recomputed from scratch, so an assignment written there would live until the next
  geo run and no longer (TestAssignmentSurvivesGeo);
* a file the camera placed itself is NOT overwritten silently (TestExactGpsIsProtected);
* the assignment reaches the PLAN and is visibly manual there (TestPlanUsesTheAssignedPlace);
* undoing it restores exactly what the program had worked out (TestClearRestoresThePlace);
* the group is bounded: an event or a source folder, never "everything at once", and a
  sibling folder with a longer name is not swept in (TestSourceFolderSelection).
"""
from __future__ import annotations

import json
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import patch

from sorta import db, geo, ui
from sorta.geodata import GeoResolver

from tests.test_geo_path_hint import _ATHENS, write_geo_fixture
from tests.test_ui import UiServerTestBase


class PlaceTestBase(UiServerTestBase):
    """The UI base plus a mini bundled base (14 rows instead of 12 MB) and helpers."""

    def setUp(self):
        super().setUp()
        write_geo_fixture(self.root / "geo")
        self.resolver = GeoResolver(data_dir=self.root / "geo")
        patcher = patch("sorta.ui._geo_resolver", return_value=self.resolver)
        patcher.start()
        self.addCleanup(patcher.stop)

    def post(self, path: str, data: object) -> tuple[int, dict]:
        import urllib.error
        import urllib.request
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=body, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def get_json(self, path: str) -> dict:
        _status, body, _ctype = self.get(path)
        return json.loads(body)

    def manual_rows(self) -> dict[int, tuple[str, str | None, int | None]]:
        return {r["file_id"]: (r["country"], r["city"], r["city_geonameid"])
                for r in self.conn.execute(
                    "SELECT file_id, country, city, city_geonameid FROM manual_places")}

    def add_event(self, file_ids: list[int], name: str = "Поездка") -> int:
        cur = self.conn.execute(
            """INSERT INTO events (started_at, ended_at, place_city, name)
               VALUES ('2022-05-01T10:00:00', '2022-05-03T10:00:00', NULL, ?)""",
            (name,))
        event_id = cur.lastrowid
        self.conn.executemany(
            "INSERT INTO event_files (event_id, file_id) VALUES (?, ?)",
            [(event_id, fid) for fid in file_ids])
        self.conn.commit()
        return event_id

    def assign(self, kind: str, selector: str, **place) -> tuple[int, dict]:
        body = {"kind": kind, "selector": selector, "action": "assign"}
        body.update(place)
        return self.post("/api/place", body)


class TestPlacesSearch(PlaceTestBase):
    """The picker resolves typed text against the BUNDLED base and nothing else."""

    def setUp(self):
        super().setUp()
        self.start_server()

    def search(self, query: str, lang: str = "ru") -> list[dict]:
        return self.get_json("/api/places/search?lang=" + lang + "&q="
                             + urllib.parse.quote(query))["results"]

    def test_a_country_name_resolves_to_a_country_candidate(self):
        results = self.search("Греция")
        self.assertEqual([r["kind"] for r in results], ["country"])
        self.assertEqual(results[0]["country"], "GR")
        self.assertIsNone(results[0]["city_geonameid"])

    def test_a_city_name_resolves_to_a_city_candidate_with_its_country(self):
        results = self.search("Афины")
        self.assertEqual(results[0]["kind"], "city")
        self.assertEqual(results[0]["city_geonameid"], _ATHENS)
        self.assertEqual(results[0]["country"], "GR")
        # the label has to tell same-named cities apart — region and country
        self.assertIn("Attica", results[0]["label"])

    def test_the_english_name_of_a_city_is_found_too(self):
        # A place is looked up by the name the user knows it under, whichever of the
        # three languages that is — the labels alone follow `lang`.
        results = self.search("Athens", lang="en")
        self.assertEqual([r["city_geonameid"] for r in results], [_ATHENS])

    def test_an_unknown_name_returns_nothing_rather_than_a_guess(self):
        self.assertEqual(self.search("Шмиргород"), [])

    def test_an_empty_query_is_not_a_search(self):
        self.assertEqual(self.search("   "), [])

    def test_the_canonical_english_anchor_travels_with_a_city(self):
        # `places.city` is the en/asciiname anchor everywhere in the program; a
        # hand-picked city has to store the same shape, not a localized name.
        self.assertEqual(self.search("Афины")[0]["city"], "Athens")


class TestPlacePayloadValidation(PlaceTestBase):
    def setUp(self):
        super().setUp()
        self.fid, _p, _c = self.add_photo_file("a.jpg")
        self.event = self.add_event([self.fid])
        self.start_server()

    def test_unknown_kind_returns_400(self):
        status, _payload = self.post("/api/place", {
            "kind": "collection", "selector": "1", "action": "assign", "country": "GR"})
        self.assertEqual(status, 400)
        self.assertEqual(self.manual_rows(), {})

    def test_unknown_action_returns_400(self):
        status, _payload = self.post("/api/place", {
            "kind": "event", "selector": "1", "action": "delete", "country": "GR"})
        self.assertEqual(status, 400)

    def test_assign_without_a_country_returns_400(self):
        # A city alone would leave the layout without its top folder — and the country
        # is the level a place-less file belongs at anyway.
        status, _payload = self.post("/api/place", {
            "kind": "event", "selector": "1", "action": "assign",
            "city_geonameid": _ATHENS})
        self.assertEqual(status, 400)

    def test_empty_selector_returns_400(self):
        status, _payload = self.post("/api/place", {
            "kind": "event", "selector": "  ", "action": "assign", "country": "GR"})
        self.assertEqual(status, 400)

    def test_non_integer_city_returns_400(self):
        status, _payload = self.post("/api/place", {
            "kind": "event", "selector": "1", "action": "assign", "country": "GR",
            "city_geonameid": "264371"})
        self.assertEqual(status, 400)

    def test_non_dict_body_returns_400(self):
        status, _payload = self.post("/api/place", ["event"])
        self.assertEqual(status, 400)

    def test_an_unknown_event_writes_nothing_and_is_not_an_error(self):
        status, payload = self.assign("event", "424242", country="GR")
        self.assertEqual(status, 200)
        self.assertEqual(payload["affected"], 0)
        self.assertEqual(self.manual_rows(), {})

    def test_a_selector_that_is_not_a_number_is_not_an_event(self):
        status, payload = self.assign("event", "../../etc", country="GR")
        self.assertEqual(status, 200)
        self.assertEqual(payload["affected"], 0)


class TestAssignToAnEvent(PlaceTestBase):
    def test_every_file_of_the_event_gets_the_place_in_one_action(self):
        ids = [self.add_photo_file(f"{n}.jpg")[0] for n in range(3)]
        other, _p, _c = self.add_photo_file("outside.jpg")
        event = self.add_event(ids)
        self.start_server()
        status, payload = self.assign("event", str(event), country="GR",
                                      city_geonameid=_ATHENS)
        self.assertEqual(status, 200)
        self.assertEqual(payload["affected"], 3)
        self.assertEqual(self.manual_rows(),
                         {fid: ("GR", "Athens", _ATHENS) for fid in ids})
        self.assertNotIn(other, self.manual_rows())

    def test_a_country_only_assignment_leaves_the_city_empty(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        event = self.add_event([fid])
        self.start_server()
        self.assign("event", str(event), country="gr")
        self.assertEqual(self.manual_rows(), {fid: ("GR", None, None)})

    def test_reassigning_overwrites_the_row_instead_of_adding_one(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        event = self.add_event([fid])
        self.start_server()
        self.assign("event", str(event), country="TH")
        self.assign("event", str(event), country="GR", city_geonameid=_ATHENS)
        self.assertEqual(self.manual_rows(), {fid: ("GR", "Athens", _ATHENS)})


class TestExactGpsIsProtected(PlaceTestBase):
    """A coordinate written by the camera beats a memory of which city it was."""

    def setUp(self):
        super().setUp()
        self.placed, _p, _c = self.add_photo_file("gps.jpg", country="TH", city="Bangkok")
        self.blank, _p2, _c2 = self.add_photo_file("blank.jpg")
        self.event = self.add_event([self.placed, self.blank])
        self.start_server()

    def test_a_file_with_exact_gps_is_skipped_and_reported(self):
        _status, payload = self.assign("event", str(self.event), country="GR")
        self.assertEqual((payload["affected"], payload["skipped_gps"]), (1, 1))
        self.assertEqual(list(self.manual_rows()), [self.blank])

    def test_include_gps_overwrites_it_explicitly(self):
        # The client asks a second time, with the count in the question — an explicit
        # decision, never a silent overwrite.
        _status, payload = self.assign("event", str(self.event), country="GR",
                                       include_gps=True)
        self.assertEqual((payload["affected"], payload["skipped_gps"]), (2, 0))
        self.assertEqual(sorted(self.manual_rows()), sorted([self.placed, self.blank]))

    def test_clear_touches_every_file_of_the_group(self):
        # Undoing is always safe: dropping a manual row can only restore what the
        # program itself worked out, so nothing is protected from it.
        self.assign("event", str(self.event), country="GR", include_gps=True)
        _status, payload = self.post("/api/place", {
            "kind": "event", "selector": str(self.event), "action": "clear"})
        self.assertEqual((payload["affected"], payload["skipped_gps"]), (2, 0))
        self.assertEqual(self.manual_rows(), {})


class TestSourceFolderSelection(PlaceTestBase):
    def test_only_the_files_under_that_folder_are_assigned(self):
        inside, _p, _c = self.add_photo_file("Греция/a.jpg")
        deeper, _p2, _c2 = self.add_photo_file("Греция/2019/b.jpg")
        sibling, _p3, _c3 = self.add_photo_file("Греция2019/c.jpg")
        outside, _p4, _c4 = self.add_photo_file("Тайланд/d.jpg")
        self.start_server()
        folder = str((self.src_dir / "Греция").resolve())
        _status, payload = self.assign("source_dir", folder, country="GR")
        self.assertEqual(payload["affected"], 2)
        self.assertEqual(sorted(self.manual_rows()), sorted([inside, deeper]))
        self.assertNotIn(sibling, self.manual_rows())
        self.assertNotIn(outside, self.manual_rows())

    def test_a_folder_nobody_indexed_assigns_nothing(self):
        self.add_photo_file("a.jpg")
        self.start_server()
        _status, payload = self.assign(
            "source_dir", str((self.root / "nowhere").resolve()), country="GR")
        self.assertEqual(payload["affected"], 0)
        self.assertEqual(self.manual_rows(), {})

    def test_the_folder_is_matched_as_a_string_not_opened(self):
        # The selector never reaches the filesystem: it is compared against
        # `files.path`, which is why a path from the client is acceptable here at all.
        self.add_photo_file("a.jpg")
        self.start_server()
        _status, payload = self.assign("source_dir", "../../", country="GR")
        self.assertEqual(payload["affected"], 0)


class TestPlanUsesTheAssignedPlace(PlaceTestBase):
    """The point of the whole feature: the assignment has to reach the layout."""

    def aggregate(self) -> dict:
        return self.get_json("/api/plan?mode=city")

    def page(self, category: str) -> dict:
        return self.get_json("/api/plan?mode=city&category="
                             + urllib.parse.quote(category))

    def test_a_place_less_file_leaves_no_place_for_the_assigned_country(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        event = self.add_event([fid])
        self.start_server()
        before = self.aggregate()["categories"][0]["category"]
        self.assertIn("no_place", before)
        self.assign("event", str(event), country="GR")
        category = self.aggregate()["categories"][0]["category"]
        self.assertEqual(Path(category).parts, ("Greece", "2022"))
        item = self.page(category)["items"][0]
        self.assertEqual(item["file_id"], fid)
        self.assertEqual(item["category"], "country_only")

    def test_the_plan_marks_the_place_as_manual(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        event = self.add_event([fid])
        self.start_server()
        self.assign("event", str(event), country="GR")
        category = self.aggregate()["categories"][0]["category"]
        item = self.page(category)["items"][0]
        self.assertEqual(item["place_confidence"], "manual")

    def test_an_inferred_place_is_not_marked_manual(self):
        self.add_photo_file("a.jpg", country="TH", city="Bangkok")
        self.start_server()
        category = self.aggregate()["categories"][0]["category"]
        self.assertEqual(self.page(category)["items"][0]["place_confidence"],
                         "exact_gps")

    def test_an_assigned_city_replaces_the_whole_place_not_just_the_country(self):
        fid, _p, _c = self.add_photo_file("a.jpg", country="TH", city="Bangkok")
        event = self.add_event([fid])
        self.start_server()
        with patch("sorta.sorter.GeoResolver", return_value=self.resolver):
            self.assign("event", str(event), country="GR", city_geonameid=_ATHENS,
                        include_gps=True)
            category = self.aggregate()["categories"][0]["category"]
        self.assertEqual(Path(category).parts, ("Greece", "Athens", "2022"))

    def test_a_manual_place_does_not_outrank_a_manual_correction(self):
        # F77 says where a file GOES, F85c says where it was TAKEN. A frame the user
        # dragged into a folder by hand stays there.
        fid, _p, _c = self.add_photo_file("a.jpg")
        event = self.add_event([fid])
        self.start_server()
        self.assign("event", str(event), country="GR")
        self.post("/api/overrides", {"file_ids": [fid], "action": "reassign",
                                     "target": "Франция/Париж"})
        category = self.aggregate()["categories"][0]["category"]
        self.assertEqual(Path(category).parts, ("Франция", "Париж"))


class TestAssignmentSurvivesGeo(PlaceTestBase):
    def test_a_geo_recompute_does_not_lose_the_assignment(self):
        # `geo` wipes `places` on every run, which is exactly why the assignment does
        # not live there. Nothing else in the program protects it.
        fid, _p, _c = self.add_photo_file("a.jpg")
        event = self.add_event([fid])
        self.start_server()
        self.assign("event", str(event), country="GR", city_geonameid=_ATHENS)
        with patch("sorta.geo.GeoResolver", return_value=self.resolver):
            geo.resolve_places(self.cfg, self.conn)
        self.assertEqual(self.manual_rows(), {fid: ("GR", "Athens", _ATHENS)})
        category = self.get_json("/api/plan?mode=city")["categories"][0]["category"]
        self.assertEqual(Path(category).parts[0], "Greece")

    def test_the_recomputed_automatic_place_is_still_the_one_shown_underneath(self):
        # The manual row hides the automatic place, it does not delete it: clearing
        # brings back whatever the last geo run worked out.
        fid, _p, _c = self.add_photo_file("a.jpg")
        event = self.add_event([fid])
        self.start_server()
        self.assign("event", str(event), country="GR")
        with patch("sorta.geo.GeoResolver", return_value=self.resolver):
            geo.resolve_places(self.cfg, self.conn)
        row = self.conn.execute(
            "SELECT confidence FROM places WHERE file_id = ?", (fid,)).fetchone()
        self.assertEqual(row["confidence"], "unknown")


class TestClearRestoresThePlace(PlaceTestBase):
    def test_undoing_the_assignment_returns_the_previous_layout(self):
        fid, _p, _c = self.add_photo_file("a.jpg", country="TH", city="Bangkok")
        event = self.add_event([fid])
        self.start_server()
        before = self.get_json("/api/plan?mode=city")["categories"][0]["category"]
        self.assign("event", str(event), country="GR", include_gps=True)
        moved = self.get_json("/api/plan?mode=city")["categories"][0]["category"]
        self.assertNotEqual(moved, before)
        status, payload = self.post("/api/place", {
            "kind": "event", "selector": str(event), "action": "clear"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["action"], "clear")
        after = self.get_json("/api/plan?mode=city")["categories"][0]["category"]
        self.assertEqual(after, before)

    def test_clearing_a_group_that_was_never_assigned_changes_nothing(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        event = self.add_event([fid])
        self.start_server()
        status, _payload = self.post("/api/place", {
            "kind": "event", "selector": str(event), "action": "clear"})
        self.assertEqual(status, 200)
        self.assertEqual(self.manual_rows(), {})


class TestResetWipesAssignments(PlaceTestBase):
    def test_start_over_clears_manual_places(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        event = self.add_event([fid])
        self.start_server()
        self.assign("event", str(event), country="GR")
        self.assertEqual(len(self.manual_rows()), 1)
        status, _payload = self.post("/api/process/reset", {})
        self.assertEqual(status, 200)
        self.assertEqual(self.manual_rows(), {})

    def test_reset_index_clears_manual_places(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        event = self.add_event([fid])
        self.start_server()
        self.assign("event", str(event), country="GR")
        db.reset_index(self.conn)
        self.assertEqual(self.manual_rows(), {})


class TestPlaceMarkup(PlaceTestBase):
    def setUp(self):
        super().setUp()
        self.start_server()
        _status, body, _ctype = self.get("/")
        self.html = body.decode("utf-8")

    def test_the_picker_and_the_group_buttons_are_present(self):
        self.assertIn('id="city-place-picker"', self.html)
        self.assertIn('id="place-status"', self.html)
        self.assertIn("/api/places/search", self.html)
        self.assertIn("/api/place", self.html)
        self.assertIn("renderPlacePicker(", self.html)
        self.assertIn("place-row-btn", self.html)
        self.assertIn("place-assign-btn", self.html)
        self.assertIn("place-clear-btn", self.html)

    def test_a_hand_assigned_place_is_drawn_differently_from_an_inferred_one(self):
        self.assertIn('item.place_confidence === "manual"', self.html)
        self.assertIn("place_manual_mark", self.html)

    def test_the_gps_overwrite_is_a_second_question(self):
        self.assertIn("place_include_gps_confirm", self.html)

    def test_no_external_resources_added(self):
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)
        self.assertNotIn("<link", self.html)


class TestPlaceStringsAreTranslated(unittest.TestCase):
    KEYS = ("place_search_placeholder", "place_assign_button", "place_clear_button",
            "place_folder_button", "place_not_found", "place_alert_choose",
            "place_assign_confirm", "place_folder_confirm", "place_event_clear_confirm",
            "place_folder_clear_confirm", "place_assigned_status",
            "place_cleared_status", "place_skipped_gps", "place_include_gps_confirm",
            "place_manual_mark", "place_hint", "place_error_prefix")

    def test_every_new_string_exists_in_all_three_languages(self):
        for key in self.KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")


if __name__ == "__main__":
    unittest.main()
