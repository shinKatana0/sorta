"""F202: a place a person assigns can be a REGION, not only a city or a country.

The picker knew two levels and people think in three. Typing «Алтай» used to answer with
two Mongolian towns and nothing else, while «Карелия», «Крым» and «Тоскана» found nothing
at all — and the collection this was measured on holds 7 492 frames with no city, for many
of which the region is precisely the answer their owner remembers.

Nothing had to be downloaded for it: `admin1.tsv` has shipped in the wheel all along, with
3 865 regions, each carrying its own geonameid — so a region localizes through the same
`names.tsv` lookup a city gets, in the same three languages. Everything already SHOWED
regions (a city label reads «Петрозаводск (Карельская республика, Россия)»); only the
search and the layout could not name one.

What the tests pin:

* a region is FOUND, and the list says it is a region — one word is regularly both a city
  and a region, and «Алтай» is exactly that (TestRegionSearch);
* the answer is ordered widest level first, country → region → city, because the bigger
  the miss the more visible it is in the plan (TestRegionSearch);
* the assignment reaches the layout as a level of its own — `<Country>/<Region>/<Year>`
  with reason `region_only` (TestThePlanLaysOutTheRegion);
* a region never mixes with an inferred city: the city column is cleared, and a body
  asking for both levels at once is refused (TestARegionIsOneWholePlace);
* rows written before the column existed keep working, and so does a database that is old
  in every other respect too (TestOlderRowsAndOlderDatabases).
"""
from __future__ import annotations

import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import patch

from sorta import db, ui
from sorta.geodata import GeoResolver

from tests.schema_history import roll_back_before
from tests.test_geo_path_hint import write_geo_fixture
from tests.test_ui_bulk_place import PlaceTestBase

# The regions the owner named, with the ids and the spellings the BUNDLED data really
# carries — including the one that is spelled two ways (552548 is «Republic of Karelia»
# in names.tsv and plainly «Karelia» in admin1.tsv). A fixture that tidied those up would
# be testing a base nobody ships.
_KARELIA, _ALTAI_REPUBLIC, _ALTAI_KRAI = 552548, 1506272, 1511732
_MOSCOW_OBLAST = 524925
# Mongolia: the province of the two towns called «Алтай» that the picker used to answer
# with, and the province next to it. Ids of their own, so the two towns are told apart.
_GOVI_ALTAI, _BAYAN_OLGII = 1515917, 1515918
_ALTAI_GOVI_TOWN, _ALTAI_BAYAN_TOWN = 2032614, 2032615
_MONGOLIA = 1000

# places.tsv: geonameid, lat, lon, fcode, cc, admin1, admin2, name_en, population
_REGION_PLACES = [
    (_ALTAI_GOVI_TOWN, 46.3722, 96.2583, "PPLA", "MN", "10", "", "Altai", "15800"),
    (_ALTAI_BAYAN_TOWN, 48.9667, 89.9667, "PPLA", "MN", "01", "", "Altai", "5000"),
]
_REGION_ADMIN1 = [
    ("RU", "28", _KARELIA, "Karelia"),
    ("RU", "03", _ALTAI_REPUBLIC, "Altai"),
    ("RU", "04", _ALTAI_KRAI, "Altai Krai"),
    # «Мо» is the one prefix of this fixture that finds all three levels at once — the
    # country «Монголия», this region and the town «Море» — which is what the order of
    # the answer is pinned on.
    ("RU", "47", _MOSCOW_OBLAST, "Moscow Oblast"),
    ("MN", "10", _GOVI_ALTAI, "Govi-Altai Province"),
    ("MN", "01", _BAYAN_OLGII, "Bayan-Olgiy Province"),
]
_REGION_COUNTRIES = [("MN", _MONGOLIA, "Mongolia")]
_REGION_NAMES = [
    (_KARELIA, "ru", "Карельская республика"),
    (_KARELIA, "en", "Republic of Karelia"),
    (_KARELIA, "ja", "カレリア共和国"),
    (_ALTAI_REPUBLIC, "ru", "Алтай"), (_ALTAI_REPUBLIC, "en", "Altai Republic"),
    (_ALTAI_REPUBLIC, "ja", "アルタイ共和国"),
    (_ALTAI_KRAI, "ru", "Алтайский Край"), (_ALTAI_KRAI, "en", "Altay Kray"),
    (_MOSCOW_OBLAST, "ru", "Московская область"),
    (_MOSCOW_OBLAST, "en", "Moscow Oblast"),
    (_GOVI_ALTAI, "ru", "Говь-Алтай"), (_GOVI_ALTAI, "en", "Govi-Altai Province"),
    (_BAYAN_OLGII, "ru", "Баян-Улгий"), (_BAYAN_OLGII, "en", "Bayan-Olgiy Province"),
    (_ALTAI_GOVI_TOWN, "ru", "Алтай"), (_ALTAI_GOVI_TOWN, "en", "Altai"),
    (_ALTAI_BAYAN_TOWN, "ru", "Алтай"), (_ALTAI_BAYAN_TOWN, "en", "Altai"),
    (_MONGOLIA, "ru", "Монголия"), (_MONGOLIA, "en", "Mongolia"),
    (_MONGOLIA, "ja", "モンゴル"),
    (800, "ja", "ロシア"),  # the fixture's Russia had no japanese name to answer with
]


def write_region_geo_fixture(data_dir: Path) -> None:
    """The mini bundled base of the suite, plus the regions this feature is about."""
    write_geo_fixture(data_dir)
    _append(data_dir / "places.tsv", _REGION_PLACES)
    _append(data_dir / "admin1.tsv", _REGION_ADMIN1)
    _append(data_dir / "countries.tsv", _REGION_COUNTRIES)
    _append(data_dir / "names.tsv", _REGION_NAMES)


def _append(path: Path, rows: list) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write("\t".join(str(v) for v in row) + "\n")


class RegionPlaceBase(PlaceTestBase):
    """The bulk-place fixtures, over a base that also knows regions.

    The rows are appended AFTER the base class has built its resolver, which is safe
    because a `GeoResolver` reads nothing until it is first asked something — and it is
    first asked inside a request.
    """

    def setUp(self):
        super().setUp()
        write_region_geo_fixture(self.root / "geo")

    def manual_rows(self) -> dict[int, tuple[str, str | None, int | None, int | None]]:
        return {r["file_id"]: (r["country"], r["city"], r["city_geonameid"],
                               r["region_geonameid"])
                for r in self.conn.execute(
                    "SELECT file_id, country, city, city_geonameid, region_geonameid "
                    "FROM manual_places")}

    def search(self, query: str, lang: str = "ru") -> list[dict]:
        return self.get_json("/api/places/search?lang=" + lang + "&q="
                             + urllib.parse.quote(query))["results"]

    def set_language(self, lang: str) -> None:
        """The folder language — the plan is BUILT in it, so this drops the cache."""
        status, _payload = self.post("/api/config/language", {"language": lang})
        self.assertEqual(status, 200)

    def plan_categories(self) -> list[str]:
        with patch("sorta.sorter.GeoResolver", return_value=self.resolver):
            return [c["category"]
                    for c in self.get_json("/api/plan?mode=city")["categories"]]


class TestRegionSearch(RegionPlaceBase):
    def setUp(self):
        super().setUp()
        self.start_server()

    def test_a_region_is_found_and_says_that_it_is_one(self):
        results = self.search("Алтай")
        first = results[0]
        self.assertEqual(first["kind"], "region")
        self.assertEqual(first["region_geonameid"], _ALTAI_REPUBLIC)
        self.assertEqual(first["country"], "RU")
        self.assertEqual(first["label"], "Алтай (область, Россия)")

    def test_one_word_that_is_a_region_and_a_city_answers_with_both(self):
        # What the owner saw: «Алтай» gave two Mongolian towns and no republic. Now the
        # three are in one list and the LABEL is what tells them apart — the towns are
        # separated by their province, the region by the word for its level.
        results = self.search("Алтай")
        regions = [r for r in results if r["kind"] == "region"]
        cities = [r for r in results if r["kind"] == "city"]
        self.assertIn(_ALTAI_REPUBLIC, [r["region_geonameid"] for r in regions])
        self.assertEqual(sorted(c["city_geonameid"] for c in cities),
                         sorted([_ALTAI_GOVI_TOWN, _ALTAI_BAYAN_TOWN]))
        self.assertIn("Алтай (Говь-Алтай, Монголия)", [c["label"] for c in cities])
        self.assertIn("Алтай (Баян-Улгий, Монголия)", [c["label"] for c in cities])
        self.assertEqual(len({r["label"] for r in results}), len(results))

    def test_the_answer_goes_country_then_region_then_city(self):
        # The same reason the country is already first: the bigger the miss, the more
        # visible it is in the plan. A wrong region is nearly as loud as a wrong country;
        # a wrong city disappears among the right ones.
        kinds = [r["kind"] for r in self.search("Мо")]
        self.assertEqual(kinds, sorted(kinds, key=["country", "region", "city"].index))
        self.assertEqual(set(kinds), {"country", "region", "city"})

    def test_an_exact_region_name_comes_before_what_merely_starts_with_it(self):
        labels = [r["label"] for r in self.search("Алта") if r["kind"] == "region"]
        self.assertEqual(labels[0], "Алтай (область, Россия)")
        self.assertIn("Алтайский Край (область, Россия)", labels)

    def test_the_short_administrative_name_finds_the_region_too(self):
        # 552548 is spelled two ways by the base it ships in: «Republic of Karelia» in
        # names.tsv, «Karelia» in admin1.tsv. People type the short one, so both are
        # searchable — while the label keeps showing the localized name.
        results = self.search("Karelia", lang="en")
        self.assertEqual(results[0]["region_geonameid"], _KARELIA)
        self.assertEqual(results[0]["label"], "Republic of Karelia (region, Russia)")

    def test_the_region_is_found_by_the_beginning_of_its_localized_name(self):
        results = self.search("Карел")
        self.assertEqual([r["region_geonameid"] for r in results], [_KARELIA])
        self.assertEqual(results[0]["label"],
                         "Карельская республика (область, Россия)")

    def test_a_region_is_offered_in_each_of_the_three_languages(self):
        expected = {"ru": "Карельская республика (область, Россия)",
                    "en": "Republic of Karelia (region, Russia)",
                    "ja": "カレリア共和国 (地域, ロシア)"}
        for lang, label in expected.items():
            with self.subTest(lang=lang):
                # The name is looked up in ALL three languages whatever the interface
                # language is — a person types the name they know the place under.
                results = self.search("Карел", lang=lang)
                self.assertEqual(results[0]["region_geonameid"], _KARELIA)
                self.assertEqual(results[0]["label"], label)

    def test_a_name_the_base_does_not_hold_stays_unfound(self):
        self.assertEqual(self.search("Шмиргородская область"), [])

    def test_a_city_option_carries_no_region_id(self):
        # One option is one LEVEL: the client passes on whichever id it was given, so a
        # city that carried a region id as well would ask for a place at two levels.
        results = self.search("Афины")
        self.assertIsNone(results[0]["region_geonameid"])


class TestARegionIsOneWholePlace(RegionPlaceBase):
    """F85c's rule inside the manual row: one level, from one source, replaced whole."""

    def setUp(self):
        super().setUp()
        self.fid, _p, _c = self.add_photo_file("a.jpg")
        self.event = self.add_event([self.fid])
        self.start_server()

    def test_the_region_is_stored_and_the_city_stays_empty(self):
        status, payload = self.assign("event", str(self.event), country="RU",
                                      region_geonameid=_KARELIA)
        self.assertEqual(status, 200)
        self.assertEqual(payload["region_geonameid"], _KARELIA)
        self.assertEqual(self.manual_rows(), {self.fid: ("RU", None, None, _KARELIA)})

    def test_a_region_assigned_over_a_city_clears_that_city(self):
        self.assign("event", str(self.event), country="GR", city_geonameid=264371)
        self.assign("event", str(self.event), country="RU", region_geonameid=_KARELIA)
        self.assertEqual(self.manual_rows(), {self.fid: ("RU", None, None, _KARELIA)})

    def test_a_city_assigned_over_a_region_clears_that_region(self):
        self.assign("event", str(self.event), country="RU", region_geonameid=_KARELIA)
        self.assign("event", str(self.event), country="GR", city_geonameid=264371)
        self.assertEqual(self.manual_rows(),
                         {self.fid: ("GR", "Athens", 264371, None)})

    def test_asking_for_a_city_and_a_region_at_once_is_refused(self):
        status, _payload = self.post("/api/place", {
            "kind": "event", "selector": str(self.event), "action": "assign",
            "country": "RU", "city_geonameid": 264371,
            "region_geonameid": _KARELIA})
        self.assertEqual(status, 400)
        self.assertEqual(self.manual_rows(), {})

    def test_an_id_that_is_not_a_region_is_refused(self):
        # `name()` answers anything at all (it ends its chain with the number itself), so
        # an unchecked id would become a folder called `424242`.
        status, _payload = self.post("/api/place", {
            "kind": "event", "selector": str(self.event), "action": "assign",
            "country": "RU", "region_geonameid": 424242})
        self.assertEqual(status, 400)
        self.assertEqual(self.manual_rows(), {})

    def test_a_city_id_is_not_accepted_as_a_region(self):
        status, _payload = self.post("/api/place", {
            "kind": "event", "selector": str(self.event), "action": "assign",
            "country": "GR", "region_geonameid": 264371})
        self.assertEqual(status, 400)

    def test_a_region_id_that_is_not_an_integer_is_refused(self):
        status, _payload = self.post("/api/place", {
            "kind": "event", "selector": str(self.event), "action": "assign",
            "country": "RU", "region_geonameid": str(_KARELIA)})
        self.assertEqual(status, 400)

    def test_clearing_removes_the_region_row_like_any_other(self):
        self.assign("event", str(self.event), country="RU", region_geonameid=_KARELIA)
        status, _payload = self.post("/api/place", {
            "kind": "event", "selector": str(self.event), "action": "clear"})
        self.assertEqual(status, 200)
        self.assertEqual(self.manual_rows(), {})


class TestThePlanLaysOutTheRegion(RegionPlaceBase):
    """The point of the feature: frames with no city land under the region, in one action.

    The layout gains a THIRD branch — `<Country>/<Region>/<Year>` beside
    `<Country>/<City>/<Year>` and `<Country>/<Year>` — and it carries a reason of its own,
    so the plan says at which level the decision was made.
    """

    def setUp(self):
        super().setUp()
        self.fid, _p, _c = self.add_photo_file("a.jpg")
        self.event = self.add_event([self.fid])
        self.start_server()

    def category(self) -> str:
        return self.plan_categories()[0]

    def test_a_city_less_frame_lands_under_the_region_in_one_action(self):
        self.set_language("ru")
        self.assertIn("без_места", self.category())  # nothing places it, before
        self.assign("event", str(self.event), country="RU", region_geonameid=_KARELIA)
        category = self.category()
        self.assertEqual(Path(category).parts,
                         ("Россия", "Карельская республика", "2022"))

    def test_the_plan_says_the_decision_was_made_at_the_region_level(self):
        self.assign("event", str(self.event), country="RU", region_geonameid=_KARELIA)
        category = self.category()
        with patch("sorta.sorter.GeoResolver", return_value=self.resolver):
            item = self.get_json("/api/plan?mode=city&category="
                                 + urllib.parse.quote(category))["items"][0]
        self.assertEqual(item["reason"], "region_only")
        self.assertEqual(item["place_confidence"], "manual")

    def test_the_region_folder_follows_the_folder_language(self):
        self.assign("event", str(self.event), country="RU", region_geonameid=_KARELIA)
        expected = {"ru": ("Россия", "Карельская республика", "2022"),
                    "en": ("Russia", "Republic of Karelia", "2022"),
                    "ja": ("ロシア", "カレリア共和国", "2022")}
        for lang, parts in expected.items():
            with self.subTest(lang=lang):
                self.set_language(lang)
                self.assertEqual(Path(self.category()).parts, parts)

    def test_a_region_assigned_by_hand_never_keeps_the_inferred_city(self):
        # The rule of F85c, and the one this feature could most easily break: a manual
        # row replaces the place WHOLE. The frame was in Bangkok by GPS; it is in Karelia
        # now, and not in «Россия/Карельская республика/Bangkok».
        placed, _p, _c = self.add_photo_file("gps.jpg", country="TH", city="Bangkok")
        event = self.add_event([placed], name="Поездка 2")
        self.set_language("ru")
        self.assign("event", str(event), country="RU", region_geonameid=_KARELIA,
                    include_gps=True)
        self.assertIn(("Россия", "Карельская республика", "2022"),
                      [Path(c).parts for c in self.plan_categories()])

    def test_clearing_the_assignment_puts_the_frame_back_where_it_was(self):
        self.set_language("ru")
        before = self.category()
        self.assign("event", str(self.event), country="RU", region_geonameid=_KARELIA)
        self.assertNotEqual(self.category(), before)
        self.post("/api/place", {"kind": "event", "selector": str(self.event),
                                 "action": "clear"})
        self.assertEqual(self.category(), before)

    def test_a_region_the_bundled_data_cannot_name_falls_back_to_the_country(self):
        # A folder called `552548` explains nothing to anyone. If the id in the row can
        # no longer be named (data rebuilt, a row from another machine), the file goes to
        # the country level — one honest level up, never a number.
        self.set_language("ru")
        self.conn.execute(
            """INSERT INTO manual_places (file_id, country, city, city_geonameid,
                                          region_geonameid, updated_at)
               VALUES (?, 'RU', NULL, NULL, 987654, '2026-01-01')""", (self.fid,))
        self.conn.commit()
        self.assertEqual(Path(self.category()).parts, ("Россия", "2022"))


class TestOlderRowsAndOlderDatabases(RegionPlaceBase):
    """Nothing that was written before the column existed may change meaning."""

    def setUp(self):
        super().setUp()
        self.city_file, _p, _c = self.add_photo_file("city.jpg")
        self.country_file, _p2, _c2 = self.add_photo_file("country.jpg")
        # Exactly the two shapes v14..v27 could hold: country + city, and country alone.
        self.conn.executemany(
            """INSERT INTO manual_places (file_id, country, city, city_geonameid,
                                          updated_at)
               VALUES (?, ?, ?, ?, '2026-01-01')""",
            [(self.city_file, "GR", "Athens", 264371),
             (self.country_file, "GR", None, None)])
        self.conn.commit()
        self.start_server()

    def test_an_old_row_still_lays_out_exactly_as_it_did(self):
        self.assertEqual(
            sorted(Path(c).parts for c in self.plan_categories()),
            sorted([("Greece", "Athens", "2022"), ("Greece", "2022")]))

    def test_an_old_row_reads_as_a_region_less_one(self):
        self.assertEqual(self.manual_rows(),
                         {self.city_file: ("GR", "Athens", 264371, None),
                          self.country_file: ("GR", None, None, None)})


class TestTheMigration(unittest.TestCase):
    """A database that is old in EVERY respect, not only in this column."""

    def setUp(self):
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(self.tmp) / "old.db"

    def _old_database(self):
        """A v27 database — `manual_places` without `region_geonameid`, and nothing that
        arrived after it either (see tests/schema_history.py for why that matters)."""
        conn = db.connect(self.path)
        version = roll_back_before(conn, "manual_places.region_geonameid")
        conn.commit()
        conn.close()
        return version

    def test_the_column_is_added_to_a_database_that_predates_it(self):
        version = self._old_database()
        self.assertEqual(version, db.SCHEMA_VERSION - 1)
        conn = db.connect(self.path)
        try:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(manual_places)")}
            self.assertEqual(cols, {"file_id", "country", "city", "city_geonameid",
                                    "region_geonameid", "updated_at"})
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0],
                             db.SCHEMA_VERSION)
        finally:
            conn.close()

    def test_a_row_written_before_the_migration_survives_it_unchanged(self):
        conn = db.connect(self.path)
        roll_back_before(conn, "manual_places.region_geonameid")
        conn.execute("INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)"
                     " VALUES ('/x.jpg', 1, 0.0, 'jpg', 'photo', 'now')")
        conn.execute("INSERT INTO manual_places (file_id, country, city, city_geonameid,"
                     " updated_at) VALUES (1, 'GR', 'Athens', 264371, 'now')")
        conn.commit()
        conn.close()

        conn = db.connect(self.path)
        try:
            row = conn.execute("SELECT * FROM manual_places").fetchone()
            self.assertEqual((row["country"], row["city"], row["city_geonameid"],
                              row["region_geonameid"]),
                             ("GR", "Athens", 264371, None))
        finally:
            conn.close()


class TestTheCaptionsAreTranslatedThreeWays(unittest.TestCase):
    KEYS = ("place_kind_region", "place_search_placeholder", "dest_why_region_only",
            "dest_group_region")

    def test_every_new_string_exists_in_all_three_languages(self):
        for key in self.KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")

    def test_a_region_destination_is_a_group_of_its_own(self):
        from sorta.sorter import Destination
        told = ui._destination_json(Destination(1, ("Россия", "Крым", "2022"),
                                                "region_only"))
        self.assertEqual(told["dest_group"], "region")

    def test_the_picker_field_offers_the_region_level(self):
        # The placeholder is what tells a person the field takes a region at all.
        html = ui._render_index_html("ru")
        self.assertIn("Город, область или страна", html)


class TestTheResolverAnswersAboutRegions(unittest.TestCase):
    """The lookups underneath, straight on the resolver — the picker only ranks them."""

    def setUp(self):
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        write_region_geo_fixture(Path(self.tmp) / "geo")
        self.resolver = GeoResolver(data_dir=Path(self.tmp) / "geo")

    def test_a_full_name_answers_with_the_region_id(self):
        self.assertEqual(self.resolver.region_ids_by_name("Алтай", "ru"),
                         [_ALTAI_REPUBLIC])

    def test_a_prefix_answers_with_every_region_it_begins(self):
        # A word start, not a substring — «Говь-Алтай» is in the answer because the
        # hyphen separates words, the rule `city_ids_by_prefix` already follows.
        self.assertEqual(sorted(self.resolver.region_ids_by_prefix("Алта", "ru")),
                         sorted([_ALTAI_REPUBLIC, _ALTAI_KRAI, _GOVI_ALTAI]))

    def test_an_empty_prefix_is_not_a_query(self):
        self.assertEqual(self.resolver.region_ids_by_prefix("  ", "ru"), [])

    def test_a_region_id_resolves_back_to_its_country_and_code(self):
        self.assertEqual(self.resolver.region_key_by_id(_KARELIA), ("RU", "28"))

    def test_a_city_id_is_not_a_region(self):
        self.assertIsNone(self.resolver.region_key_by_id(_ALTAI_GOVI_TOWN))

    def test_the_index_is_built_once_per_language(self):
        first = self.resolver._region_name_index("ru")
        self.assertIs(self.resolver._region_name_index("ru"), first)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
