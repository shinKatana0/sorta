"""F85c (part 1): the COUNTRY read off the name of a folder on the file's path.

Every automatic signal has been measured on the live collection and this is what is
left: about 6 300 files with no place signal at all. The folder they lie in is the last
thing that says anything, and it says exactly one thing reliably — a country name in a
folder is right 99.5% of 2 105 hints, while a CITY name in a folder is right 4.3% of
1 152 (the bundled base holds 150 000 settlements, so any ordinary word finds a hamlet).

So the tests below pin the three properties that make the hint safe rather than clever:
the country is read, the city is NEVER read even when a folder is named exactly like a
city, and the hint is last in the queue — it fills what nothing else reached and
overrides neither GPS nor either level of inheritance.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sorta import i18n
from sorta.config import Config
from sorta.db import connect
from sorta.geo import _CountryFromPath, resolve_places
from sorta.geodata import GeoResolver
from sorta.sorter import plan_and_sort

# A tiny world (the fixture shape of test_geodata): three countries and three cities,
# one of which is named like an ordinary Russian word on purpose — «Море» is what a
# folder gets called, and the bundled base really does hold settlements with such names.
_ATHENS, _BANGKOK, _MORE = 264371, 1609350, 500001
_PLACES = [
    (_ATHENS, 37.9838, 23.7275, "PPLC", "GR", "ESYE31", "", "Athens", "664046"),
    (_BANGKOK, 13.7563, 100.5018, "PPLC", "TH", "40", "", "Bangkok", "5104476"),
    (_MORE, 55.0, 38.0, "PPLA", "RU", "", "", "More", "3000"),
]
_ADMIN1 = [("GR", "ESYE31", 400, "Attica"), ("TH", "40", 500, "Bangkok")]
_COUNTRIES = [("GR", 600, "Greece"), ("TH", 700, "Thailand"), ("RU", 800, "Russia"),
              ("HR", 900, "Croatia")]
_NAMES = [
    (_ATHENS, "ru", "Афины"), (_ATHENS, "en", "Athens"),
    (_BANGKOK, "ru", "Бангкок"), (_BANGKOK, "en", "Bangkok"),
    (_MORE, "ru", "Море"), (_MORE, "en", "More"),
    (600, "ru", "Греция"), (600, "en", "Greece"),
    (700, "ru", "Тайланд"), (700, "en", "Thailand"),
    (800, "ru", "Россия"), (800, "en", "Russia"),
    (900, "ru", "Хорватия"), (900, "en", "Croatia"),
]


def write_geo_fixture(data_dir: Path) -> None:
    """The mini bundled base — real GeoResolver, 14 rows instead of 12 MB."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "places.tsv").write_text(
        "".join("\t".join(str(v) for v in row) + "\n" for row in _PLACES),
        encoding="utf-8")
    (data_dir / "names.tsv").write_text(
        "".join(f"{gid}\t{lang}\t{name}\n" for gid, lang, name in _NAMES),
        encoding="utf-8")
    (data_dir / "admin1.tsv").write_text(
        "".join(f"{cc}\t{a1}\t{gid}\t{en}\n" for cc, a1, gid, en in _ADMIN1),
        encoding="utf-8")
    (data_dir / "countries.tsv").write_text(
        "".join(f"{cc}\t{gid}\t{en}\n" for cc, gid, en in _COUNTRIES),
        encoding="utf-8")


class _PathHintBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_geo_fixture(self.root / "geo")
        self.resolver = GeoResolver(data_dir=self.root / "geo")
        self.cfg = Config(sources=[self.root], database=self.root / "test.db")
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file(self, path: str, taken_at="2023-05-01T10:00:00", at=None,
                 confidence="high"):
        self._n += 1
        lat, lon = at if at else (None, None)
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, taken_at,
                   taken_at_source, taken_at_confidence, gps_lat, gps_lon, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', ?, 'exif', ?, ?, ?, '2026-01-01')""",
            (path, taken_at, confidence, lat, lon))
        self.conn.commit()
        return cur.lastrowid

    def place_of(self, file_id):
        return self.conn.execute(
            "SELECT country, city, city_geonameid, confidence FROM places "
            "WHERE file_id = ?", (file_id,)).fetchone()

    def run_geo(self):
        with patch("sorta.geo.GeoResolver", return_value=self.resolver):
            return resolve_places(self.cfg, self.conn, progress=lambda done, total: None)


class TestCountryFromFolderName(_PathHintBase):
    """The reading itself: what a folder name may and may not give away."""

    def hint(self):
        return _CountryFromPath(self.resolver)

    def test_country_with_a_year_after_it(self):
        # «Тайланд 2023» — the shape people actually use. The year is a separator, not
        # part of the name, so the country behind it must still be found.
        self.assertEqual(self.hint().country_of("D:/Фото/Тайланд 2023/IMG_1.jpg"), "TH")

    def test_an_ordinary_word_names_no_country(self):
        self.assertIsNone(self.hint().country_of("D:/Фото/Море/IMG_1.jpg"))

    def test_a_city_name_is_never_read_even_when_it_matches_exactly(self):
        # «Море» is a city of the fixture base, with that exact name in ru — and the
        # hint still returns nothing. This is the whole reason the feature is
        # country-only: a city read from a folder name measured 4.3% precision.
        self.assertEqual(self.resolver.city_ids_by_name("Море", "ru"), [_MORE])
        self.assertIsNone(self.hint().country_of("D:/Фото/Море/IMG_1.jpg"))

    def test_windows_separators_are_understood(self):
        self.assertEqual(self.hint().country_of("D:\\Фото\\Греция\\a.jpg"), "GR")

    def test_the_file_name_itself_is_not_read(self):
        # A camera names files; a person names folders. «Греция.jpg» in a folder that
        # says nothing is not evidence about the place.
        self.assertIsNone(self.hint().country_of("D:/Фото/2019/Греция.jpg"))

    def test_the_deepest_folder_wins(self):
        self.assertEqual(
            self.hint().country_of("D:/Хорватия/Отпуск/Греция 2019/a.jpg"), "GR")

    def test_a_country_the_bundled_base_misses_comes_from_the_curated_dictionary(self):
        # The two name sources are deliberate: the base carries the spellings people
        # type, the curated i18n dictionary carries the ones the program itself prints.
        self.assertIsNone(self.resolver.country_cc_by_name("Франция", "ru"))
        self.assertEqual(i18n.country_cc_by_name("Франция"), "FR")
        self.assertEqual(self.hint().country_of("D:/Фото/Франция 2014/a.jpg"), "FR")

    def test_a_short_word_inside_a_longer_name_is_not_tried(self):
        # Country names of three letters double as ordinary words; one folder filed
        # under the wrong country costs more than the files such a name would add.
        self.assertIsNone(self.hint().country_of("D:/Фото/лето чад 2019/a.jpg"))

    def test_a_resolver_without_the_reverse_lookup_degrades_to_the_dictionary(self):
        # geo runs with whatever resolver the stage built; one without the bundled
        # data must not take the stage down with it.
        hint = _CountryFromPath(object())
        self.assertEqual(hint.country_of("D:/Фото/Греция/a.jpg"), "GR")
        self.assertIsNone(hint.country_of("D:/Фото/Море/a.jpg"))


class TestPathHintInTheGeoStage(_PathHintBase):
    """Where the hint sits in the queue, and what it writes."""

    def test_a_place_less_file_gets_the_country_of_its_folder(self):
        fid = self.add_file("D:/Фото/Тайланд 2023/a.jpg")
        stats = self.run_geo()
        row = self.place_of(fid)
        self.assertEqual((row["country"], row["city"], row["city_geonameid"],
                          row["confidence"]),
                         ("TH", None, None, "path_inferred"))
        self.assertEqual((stats.path_inferred, stats.unknown), (1, 0))

    def test_a_folder_that_names_nothing_stays_unknown(self):
        fid = self.add_file("D:/Фото/Море/a.jpg")
        stats = self.run_geo()
        self.assertEqual(self.place_of(fid)["confidence"], "unknown")
        self.assertEqual((stats.path_inferred, stats.unknown), (0, 1))

    def test_gps_is_not_overridden_by_the_folder_name(self):
        # The folder says Thailand, the camera says Athens. The camera was there.
        fid = self.add_file("D:/Фото/Тайланд 2023/a.jpg", at=(37.9838, 23.7275))
        stats = self.run_geo()
        row = self.place_of(fid)
        self.assertEqual((row["country"], row["confidence"]), ("GR", "exact_gps"))
        self.assertEqual(stats.path_inferred, 0)

    def test_session_inheritance_is_not_overridden_by_the_folder_name(self):
        self.add_file("D:/Фото/Греция 2019/a.jpg", taken_at="2019-05-01T10:00:00",
                      at=(37.9838, 23.7275))
        orphan = self.add_file("D:/Фото/Тайланд 2023/b.jpg",
                               taken_at="2019-05-01T12:00:00")
        stats = self.run_geo()
        row = self.place_of(orphan)
        self.assertEqual((row["country"], row["city_geonameid"], row["confidence"]),
                         ("GR", _ATHENS, "session_inferred"))
        self.assertEqual(stats.path_inferred, 0)

    def test_trip_inheritance_is_not_overridden_by_the_folder_name(self):
        # Three sessions a day apart = one trip; the middle one has no GPS at all and
        # lies between two Athens frames — that is trip_inferred (F85a), and a folder
        # name must not take it away.
        self.add_file("D:/Фото/Греция/a.jpg", taken_at="2019-05-01T10:00:00",
                      at=(37.9838, 23.7275))
        orphan = self.add_file("D:/Фото/Тайланд 2023/b.jpg",
                               taken_at="2019-05-02T10:00:00")
        self.add_file("D:/Фото/Греция/c.jpg", taken_at="2019-05-03T10:00:00",
                      at=(37.9838, 23.7275))
        stats = self.run_geo()
        self.assertEqual(self.place_of(orphan)["confidence"], "trip_inferred")
        self.assertEqual(stats.path_inferred, 0)

    def test_a_file_with_an_unusable_date_still_gets_the_hint(self):
        # Unlike the two inheritance rules, the hint does not consult the date: a
        # folder name says nothing about time, and an undated file is laid out by the
        # sorter's own undated branch anyway.
        fid = self.add_file("D:/Фото/Тайланд 2023/a.jpg", taken_at=None,
                            confidence="low")
        self.run_geo()
        self.assertEqual(self.place_of(fid)["confidence"], "path_inferred")

    def test_the_hint_is_recomputed_from_scratch_like_every_other_level(self):
        fid = self.add_file("D:/Фото/Тайланд 2023/a.jpg")
        self.run_geo()
        self.conn.execute("UPDATE files SET path = ? WHERE id = ?",
                          ("D:/Фото/Море/a.jpg", fid))
        self.conn.commit()
        self.run_geo()
        self.assertEqual(self.place_of(fid)["confidence"], "unknown")


class TestPathHintReachesTheLayout(_PathHintBase):
    """The point of part 1: 520 files leave _Unsorted/no_place for a country folder."""

    def test_a_country_only_place_is_laid_out_under_the_country(self):
        self.add_file("D:/Фото/Греция 2019/a.jpg", taken_at="2019-07-01T10:00:00")
        self.run_geo()
        report = plan_and_sort(self.cfg, self.conn, "city", self.root / "out",
                               apply=False, write_reports=False)
        item = report.plan[0]
        self.assertEqual(item.reason, "country_only")
        self.assertEqual(Path(item.target_rel).parts[:2], ("Greece", "2019"))


if __name__ == "__main__":
    unittest.main()
