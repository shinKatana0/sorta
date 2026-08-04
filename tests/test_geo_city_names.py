"""F172: one alphabet per language — `places.city` is named by ONE rule.

The live ru collection (2026-08-02, geo.provider=online) held three alphabets at once:
«Санкт-Петербург» and «Москва» next to `Nizhny Novgorod`, `Samara`, `Ryazan` and a Thai
village — and «Сочи» (385 files, the provider named a suburb too) next to `Sochi`
(29 files, no suburb, completed from the bundled base) for one and the same geonameid.
The data was never the problem: `names.tsv` has a Russian name for all four cities. The
two sources of the NAME were: the bundled base was asked for the English anchor while
the provider answered in the language of the request.

Everything here runs against a REAL geodata.GeoResolver over a miniature bundled
fixture, so the chain (lang -> en -> native) is exercised on actual TSV rows, and
against a faked Nominatim, so the tests stay offline.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import patch

from sorta.config import Config, GeoConfig
from sorta.db import connect
from sorta.geo import resolve_places
from sorta.geodata import GeoResolver
from sorta.sorter import _city_display_name

# --- the miniature bundled base ------------------------------------------------------
# geonameids are the real ones of these cities — the defect report names them.
_GID_SOCHI = 491422
_GID_SOCHI_CENTRE = 491423   # a district of Sochi, no localized name
_GID_SAMARA = 499099
_GID_KAPONG = 1150965        # en only: no Russian name for a Thai village exists
_GID_JABAJERO = 1642911      # not in names.tsv at all — asciiname is the only answer

# geonameid, lat, lon, fcode, cc, admin1, admin2, name_en(asciiname), population
_PLACES = [
    (_GID_SOCHI, 43.60, 39.73, "PPLA", "RU", "38", "", "Sochi", "400000"),
    (_GID_SOCHI_CENTRE, 43.61, 39.74, "PPLX", "RU", "38", "", "Tsentralnyy", "0"),
    (_GID_SAMARA, 53.19, 50.10, "PPLA", "RU", "43", "", "Samara", "1100000"),
    (_GID_KAPONG, 8.80, 98.40, "PPLA3", "TH", "62", "", "Kapong", "5000"),
    (_GID_JABAJERO, -8.71, 115.16, "PPLA3", "ID", "02", "", "Jabajero", "1200"),
]
_COUNTRIES = [("RU", 600, "Russia"), ("TH", 601, "Thailand"), ("ID", 602, "Indonesia")]
_NAMES = [
    (_GID_SOCHI, "ru", "Сочи"),
    (_GID_SOCHI, "en", "Sochi"),
    (_GID_SOCHI, "ja", "ソチ"),
    (_GID_SAMARA, "ru", "Самара"),
    (_GID_SAMARA, "en", "Samara"),
    (_GID_SAMARA, "ja", "サマラ"),
    (_GID_KAPONG, "en", "Kapong"),
]

# Coordinates. The two Sochi ones are the point of the fixture: `DISTRICT` sits on top
# of the PPLX, so the local base answers city+district, while `EDGE` is nearer to the
# city itself and answers city only — two different cache keys, therefore two different
# provider answers, exactly as on the live collection.
_SOCHI_DISTRICT = (43.6100, 39.7400)
_SOCHI_EDGE = (43.5800, 39.7100)
_SAMARA = (53.1900, 50.1000)
_KAPONG = (8.8000, 98.4000)
_JABAJERO = (-8.7100, 115.1600)

# What Nominatim says. The pattern of the defect: an answer that names the city (and a
# suburb with it) vs one that stops at the region — the latter is what the bundled base
# then completes (F86).
_ONLINE_ANSWERS = {
    _SOCHI_DISTRICT: {
        "ru": {"city": "Сочи", "suburb": "Центральный район",
               "country": "Россия", "country_code": "ru"},
        "en": {"city": "Sochi", "suburb": "Tsentralny District",
               "country": "Russia", "country_code": "ru"},
        "ja": {"city": "ソチ", "suburb": "ツェントラリヌイ",
               "country": "ロシア", "country_code": "ru"},
    },
    _SOCHI_EDGE: {
        lang: {"state": "Краснодарский край", "country": "Россия", "country_code": "ru"}
        for lang in ("ru", "en", "ja")
    },
    _SAMARA: {
        lang: {"state": "Самарская область", "country": "Россия", "country_code": "ru"}
        for lang in ("ru", "en", "ja")
    },
    # A Thai village OSM knows under one name only — the answer is the same in all
    # three languages, and it is the honest one.
    _KAPONG: {
        lang: {"village": "บ้านบางหลาโอน", "country": "ประเทศไทย", "country_code": "th"}
        for lang in ("ru", "en", "ja")
    },
}


def write_geo_fixture(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "places.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for row in _PLACES:
            f.write("\t".join(str(v) for v in row) + "\n")
    with (data_dir / "countries.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for cc, gid, name_en in _COUNTRIES:
            f.write(f"{cc}\t{gid}\t{name_en}\n")
    with (data_dir / "admin1.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("RU\t38\t491420\tKrasnodarskiy Kray\n")
    with (data_dir / "names.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for gid, lang, name in _NAMES:
            f.write(f"{gid}\t{lang}\t{name}\n")


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class CityNamesTestBase(unittest.TestCase):
    """A DB, a real resolver over the fixture above and a faked reverse endpoint."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_geo_fixture(self.root / "geo")
        self.resolver = GeoResolver(data_dir=self.root / "geo")
        self.conn = connect(self.root / "test.db")
        self.requests: list[tuple[float, float, str]] = []
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def config(self, lang: str, provider: str = "offline") -> Config:
        return Config(sources=[self.root], database=self.root / "test.db",
                      language=lang, raw={"language": lang},
                      geo=GeoConfig(provider=provider))

    def add_file(self, coords, taken_at="2023-03-03T14:28:32") -> int:
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, taken_at,
                   taken_at_source, taken_at_confidence, gps_lat, gps_lon, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', ?, 'exif', 'high', ?, ?, '2026-01-01')""",
            (f"/photos/img_{self._n}.jpg", taken_at, coords[0], coords[1]),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def place_of(self, file_id):
        return self.conn.execute(
            """SELECT country, city, city_geonameid, district_geonameid, district_name
               FROM places WHERE file_id = ?""", (file_id,)).fetchone()

    def city_of(self, file_id) -> str | None:
        row = self.place_of(file_id)
        return row["city"] if row is not None else None

    def _answer(self, req, timeout=None) -> _FakeResponse:
        params = urllib.parse.parse_qs(urllib.parse.urlparse(req.full_url).query)
        lat, lon = float(params["lat"][0]), float(params["lon"][0])
        lang = params["accept-language"][0]
        self.requests.append((lat, lon, lang))
        return _FakeResponse({"address": _ONLINE_ANSWERS[(lat, lon)][lang]})

    def run_geo(self, lang: str, provider: str = "offline"):
        with patch("sorta.geo.urllib.request.urlopen", side_effect=self._answer), \
             patch("sorta.geo.GeoResolver", return_value=self.resolver), \
             patch("sorta.geo.time.sleep"):
            return resolve_places(self.config(lang, provider), self.conn,
                                  progress=lambda done, total: None)


class TestOneCityOneName(CityNamesTestBase):
    """The main symptom: «Сочи» and `Sochi` are one city and must be one name."""

    def test_one_geonameid_gives_one_name_whichever_source_found_the_place(self):
        # Both files are in Sochi. The provider names the city for the one that sits in
        # a district and stops at the region for the other, so the second place comes
        # from the bundled base — the split that produced two folders for one city.
        named = self.add_file(_SOCHI_DISTRICT)
        completed = self.add_file(_SOCHI_EDGE)
        self.run_geo("ru", provider="online")

        self.assertEqual(self.city_of(named), "Сочи")
        self.assertEqual(self.city_of(completed), "Сочи")
        # ...and the two really did travel different roads: only the completed one
        # carries a geonameid (the online provider has none to give).
        self.assertIsNone(self.place_of(named)["city_geonameid"])
        self.assertEqual(self.place_of(completed)["city_geonameid"], _GID_SOCHI)

    def _fill_the_collection(self) -> None:
        """A collection that took both roads at once: two cities named by the provider,
        two completed from the bundled base."""
        for coords in (_SOCHI_DISTRICT, _SOCHI_EDGE, _SAMARA, _KAPONG):
            self.add_file(coords)
            self.add_file(coords, taken_at="2023-07-07T10:00:00")
        self.run_geo("ru", provider="online")

    def test_no_city_of_the_collection_is_spelled_two_ways(self):
        # The acceptance criterion of the brief, as a query over the whole collection.
        self._fill_the_collection()
        names = {r["city"] for r in self.conn.execute(
            "SELECT DISTINCT city FROM places WHERE city IS NOT NULL")}
        self.assertEqual(names, {"Сочи", "Самара", "บ้านบางหลาโอน"})

    def test_the_russian_cities_are_all_cyrillic(self):
        # The second criterion: with `language: ru` no Russian city may be laid out in
        # Latin. A Thai village stays Thai — that one has no Russian name to use.
        self._fill_the_collection()
        russian = [r["city"] for r in self.conn.execute(
            "SELECT city FROM places WHERE country = 'RU' AND city IS NOT NULL")]
        self.assertTrue(russian)
        for name in russian:
            with self.subTest(city=name):
                self.assertTrue(all("А" <= ch <= "я" or ch in " -Ёё" for ch in name),
                                f"{name} is not Cyrillic")


class TestNamedInTheConfigLanguage(CityNamesTestBase):
    """Requirement 1: a name in `language` exists → it is used, whatever the road."""

    def test_the_bundled_base_names_the_city_in_the_config_language(self):
        # The regression itself: 179 files of «Самара» were filed as `Samara` under a
        # Russian country folder because the base was asked for the English anchor.
        photo = self.add_file(_SAMARA)
        self.run_geo("ru")
        self.assertEqual(self.city_of(photo), "Самара")
        self.assertEqual(self.place_of(photo)["city_geonameid"], _GID_SAMARA)

    def test_the_same_holds_when_the_base_completes_an_online_answer(self):
        photo = self.add_file(_SAMARA)
        self.run_geo("ru", provider="online")
        self.assertEqual(self.city_of(photo), "Самара")

    def test_the_provider_answer_is_kept_as_it_came(self):
        # The online provider answers in the language of the request, and that works —
        # it only has to go through the same function, not around it.
        photo = self.add_file(_SOCHI_DISTRICT)
        self.run_geo("ru", provider="online")
        self.assertEqual(self.city_of(photo), "Сочи")
        self.assertEqual(self.place_of(photo)["district_name"], "Центральный район")

    def test_an_inherited_place_carries_the_same_name(self):
        # A file without GPS inherits the whole place from its time session — the name
        # must not be re-derived anywhere on the way.
        donor = self.add_file(_SAMARA)
        heir = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, taken_at,
                   taken_at_source, taken_at_confidence, indexed_at)
               VALUES ('/photos/no_gps.jpg', 1000, 0, 'jpg', 'photo',
                       '2023-03-03T15:00:00', 'exif', 'high', '2026-01-01')""").lastrowid
        self.conn.commit()
        self.run_geo("ru")
        self.assertEqual(self.city_of(donor), "Самара")
        self.assertEqual(self.city_of(heir), "Самара")


class TestTheFallbackChain(CityNamesTestBase):
    """Requirement 2: `language` -> en -> native, in that order and no other."""

    def test_without_a_russian_name_the_english_one_is_used(self):
        # A Thai village has no Russian name in GeoNames — and none exists. An invented
        # Cyrillic transliteration would be worse than the English name.
        photo = self.add_file(_KAPONG)
        self.run_geo("ru")
        self.assertEqual(self.city_of(photo), "Kapong")

    def test_without_any_name_row_the_native_spelling_of_the_base_stays(self):
        photo = self.add_file(_JABAJERO)
        self.run_geo("ru")
        self.assertEqual(self.city_of(photo), "Jabajero")

    def test_a_provider_name_with_no_alternatives_keeps_its_own_script(self):
        # The Thai villages of the live collection: OSM knows one name, so the ru, en
        # and ja answers are all the same string — and it is the one to keep.
        photo = self.add_file(_KAPONG)
        self.run_geo("ru", provider="online")
        self.assertEqual(self.city_of(photo), "บ้านบางหลาโอน")


class TestEveryLanguage(CityNamesTestBase):
    """Requirement: en and ja get their own names on the same collection."""

    def test_each_language_names_the_same_collection_its_own_way(self):
        expected = {
            "ru": {_SOCHI_EDGE: "Сочи", _SAMARA: "Самара", _KAPONG: "Kapong"},
            "en": {_SOCHI_EDGE: "Sochi", _SAMARA: "Samara", _KAPONG: "Kapong"},
            "ja": {_SOCHI_EDGE: "ソチ", _SAMARA: "サマラ", _KAPONG: "Kapong"},
        }
        photos = {c: self.add_file(c) for c in (_SOCHI_EDGE, _SAMARA, _KAPONG)}
        for lang, names in expected.items():
            with self.subTest(lang=lang):
                self.run_geo(lang)
                for coords, name in names.items():
                    self.assertEqual(self.city_of(photos[coords]), name)


class TestSwitchingTheLanguage(CityNamesTestBase):
    """Requirement 5: the name is chosen when READ, so a switch costs no re-run."""

    def test_the_geonameid_lets_a_reader_rename_without_touching_geo(self):
        # What the sorter does with the row geo wrote (sorter._city_display_name): the
        # folder language changes and the place stays where it is, with no geo pass and
        # no write into `places`.
        photo = self.add_file(_SAMARA)
        self.run_geo("ru")
        row = self.place_of(photo)
        self.assertEqual(row["city"], "Самара")
        for lang, name in (("ja", "サマラ"), ("en", "Samara"), ("ru", "Самара")):
            with self.subTest(lang=lang):
                self.assertEqual(
                    _city_display_name(row["city"], row["city_geonameid"], lang,
                                       self.resolver), name)

    def test_a_re_run_in_another_language_asks_the_provider_nothing(self):
        # F93: the cache holds all three languages of an answer, so re-running geo after
        # a language switch renames the online places without a single request.
        photo = self.add_file(_SOCHI_DISTRICT)
        self.run_geo("ru", provider="online")
        self.assertEqual(self.city_of(photo), "Сочи")
        asked = len(self.requests)
        self.run_geo("ja", provider="online")
        self.assertEqual(self.city_of(photo), "ソチ")
        self.assertEqual(len(self.requests), asked)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
