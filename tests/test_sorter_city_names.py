"""F99 (G3): the city folder is named in the layout language, not in the DB anchor.

`geo` writes the English/asciiname anchor into `places.city` plus `places.
city_geonameid` and deliberately localizes nothing (see geo._CANONICAL_LANG) — the
translation belongs here. Measured on the live collection (2026-07-28): 18 450 of
26 135 placed files carry a geonameid, 15 284 of them have a Russian name in the
bundled `names.tsv`, and before this feature every one of them was filed under its
English name while the country folder above it was already Russian.

Runs against a REAL geodata.GeoResolver over a miniature bundled fixture, so the whole
fallback chain (lang -> en -> asciiname -> geonameid) is exercised as it is in
production, not a stand-in for it.

Inherits the SorterTestBase fixtures from test_sorter.py; all FS operations — inside
its tmp dir.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_sorter import SorterTestBase

from sorta.config import Config
from sorta.geodata import GeoResolver
from sorta.sorter import _city_display_name, plan_and_sort

# --- the miniature bundled base -----------------------------------------------
# geonameids — arbitrary but stable, like the fixture in tests/test_sorter.py.
_GID_SPB = 498817          # ru/en/ja names — the 10 100-file case of the measurement
_GID_KAPONG = 1150965      # en only: no Russian name for a Thai village exists
_GID_JABAJERO = 1642911    # not in names.tsv at all — asciiname is the only answer
_GID_AKADEM = 1487117      # a district WITH a localized name (F49 keeps it)
_GID_WICHIT = 1609350      # a district without one (F49 drops it)
_GID_STALE = 999999        # in the DB, not in the bundled base (a stale snapshot)

# geonameid, lat, lon, fcode, cc, admin1, admin2, name_en(asciiname), population
_PLACES = [
    (_GID_SPB, 59.9391, 30.3161, "PPLA", "RU", "66", "", "Saint Petersburg", "5028000"),
    (_GID_KAPONG, 8.8027, 98.4023, "PPLA3", "TH", "62", "", "Kapong", "5000"),
    (_GID_JABAJERO, -8.7188, 115.1686, "PPLA3", "ID", "02", "", "Jabajero", "1200"),
    (_GID_AKADEM, 60.0128, 30.3956, "PPLX", "RU", "66", "", "Akademicheskoe", "0"),
    (_GID_WICHIT, 7.8804, 98.3923, "PPLX", "TH", "62", "", "Wichit", "0"),
]
_COUNTRIES = [("RU", 600, "Russia"), ("TH", 601, "Thailand"), ("ID", 602, "Indonesia")]
_NAMES = [
    (_GID_SPB, "ru", "Санкт-Петербург"),
    (_GID_SPB, "en", "Saint Petersburg"),
    (_GID_SPB, "ja", "サンクトペテルブルク"),
    (_GID_KAPONG, "en", "Kapong"),
    (_GID_AKADEM, "ru", "Академическое"),
    (_GID_AKADEM, "en", "Akademicheskoe"),
    (_GID_AKADEM, "ja", "アカデミーチェスコエ"),
    (_GID_WICHIT, "en", "Wichit"),
]


def write_geo_fixture(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "places.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for row in _PLACES:
            f.write("\t".join(str(v) for v in row) + "\n")
    with (data_dir / "countries.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for cc, gid, name_en in _COUNTRIES:
            f.write(f"{cc}\t{gid}\t{name_en}\n")
    with (data_dir / "admin1.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("RU\t66\t536203\tSaint Petersburg City\n")
    with (data_dir / "names.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for gid, lang, name in _NAMES:
            f.write(f"{gid}\t{lang}\t{name}\n")


class CityNamesTestBase(SorterTestBase):
    """SorterTestBase + a real resolver over the fixture above, patched into sorter."""

    def setUp(self):
        super().setUp()
        self.geo_dir = self.root / "geo"
        write_geo_fixture(self.geo_dir)
        self.resolver = GeoResolver(data_dir=self.geo_dir)

    def config(self, lang: str) -> Config:
        return Config(sources=[self.src_dir], database=self.root / "test.db",
                      raw={"language": lang})

    def plan(self, lang: str, mode: str = "city") -> list:
        with patch("sorta.sorter.GeoResolver", return_value=self.resolver):
            report = plan_and_sort(self.config(lang), self.conn, mode, self.dest,
                                   apply=False, write_reports=False)
        return report.plan

    def target_of(self, lang: str, name: str) -> str:
        return next(it.target_rel for it in self.plan(lang) if it.src.name == name)


class TestCityFolderLanguage(CityNamesTestBase):
    """Requirement 1: the name comes from the geonameid, in the layout language."""

    def test_ru_uses_the_russian_name(self):
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB)
        self.assertEqual(self.target_of("ru", "spb.jpg"),
                         "Россия/Санкт-Петербург/2022/spb.jpg")

    def test_ru_without_a_russian_name_keeps_the_english_one(self):
        # A Thai village has no Russian name in GeoNames — and none exists. The English
        # one is the honest answer; what must never appear is a number or an empty
        # segment (the measurement: 21 such cities, 3 166 files).
        self.add_file("kapong.jpg", country="TH", city="Kapong",
                      city_geonameid=_GID_KAPONG)
        self.assertEqual(self.target_of("ru", "kapong.jpg"),
                         "Таиланд/Kapong/2022/kapong.jpg")

    def test_ru_without_any_name_row_falls_back_to_asciiname(self):
        # Not in names.tsv at all: the chain lands on places.tsv's asciiname, which is
        # still a name a person can read.
        self.add_file("bali.jpg", country="ID", city="Jabajero",
                      city_geonameid=_GID_JABAJERO)
        self.assertEqual(self.target_of("ru", "bali.jpg"),
                         "Индонезия/Jabajero/2022/bali.jpg")

    def test_en_uses_the_english_name_even_when_russian_exists(self):
        # Requirement 3 of the brief: the LANGUAGE decides, not what the data happens
        # to hold.
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB)
        self.assertEqual(self.target_of("en", "spb.jpg"),
                         "Russia/Saint Petersburg/2022/spb.jpg")

    def test_ja_uses_the_japanese_name_and_survives_sanitize(self):
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB)
        target = self.target_of("ja", "spb.jpg")
        self.assertEqual(target, "ロシア/サンクトペテルブルク/2022/spb.jpg")
        # _sanitize replaces what NTFS forbids with «_»; Japanese is not among it, so
        # the segment must come through whole and the plan must be usable as a path.
        self.assertNotIn("_", target.split("/")[1])
        self.assertEqual((self.dest / target).name, "spb.jpg")

    def test_row_without_geonameid_keeps_the_db_text(self):
        # A hand-assigned place (manual_places), a `path_inferred` country, a landmark
        # or an online provider's answer: the text is what a person or the provider
        # wrote, and there is no id to translate it by.
        self.add_file("tower.jpg", country="FR", city="Эйфелева башня")
        self.assertEqual(self.target_of("ru", "tower.jpg"),
                         "Франция/Эйфелева башня/2022/tower.jpg")

    def test_manual_place_without_geonameid_keeps_its_text(self):
        fid = self.add_file("manual.jpg")
        self.conn.execute(
            """INSERT INTO manual_places (file_id, country, city, city_geonameid,
                   updated_at) VALUES (?, 'RU', 'Приозерск', NULL, '2026-01-01')""",
            (fid,))
        self.conn.commit()
        self.assertEqual(self.target_of("ru", "manual.jpg"),
                         "Россия/Приозерск/2022/manual.jpg")


class TestNeverABareGeonameid(CityNamesTestBase):
    """Requirement 3: `498817` is not a folder name anyone can explain."""

    def test_stale_geonameid_falls_back_to_the_anchor_text(self):
        self.add_file("mystery.jpg", country="RU", city="Mystery Town",
                      city_geonameid=_GID_STALE)
        self.assertEqual(self.target_of("ru", "mystery.jpg"),
                         "Россия/Mystery Town/2022/mystery.jpg")

    def test_helper_returns_the_id_only_when_there_is_no_anchor_either(self):
        # Belt and braces: `places.city` is NOT NULL for every row that reaches the
        # city branch, so this is unreachable through plan_and_sort — but the helper
        # must still answer something rather than None.
        self.assertEqual(
            _city_display_name(None, _GID_STALE, "ru", self.resolver), str(_GID_STALE))

    def test_stale_district_geonameid_is_dropped_not_shown_as_a_number(self):
        # drop_unlocalized_district=False keeps transliterated districts (F49) — it
        # must not let a bare id through the same door.
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB, district_geonameid=_GID_STALE)
        cfg = self.config("ru")
        cfg.sort.drop_unlocalized_district = False
        with patch("sorta.sorter.GeoResolver", return_value=self.resolver):
            report = plan_and_sort(cfg, self.conn, "city", self.dest, apply=False,
                                   write_reports=False)
        self.assertEqual(report.plan[0].target_rel,
                         "Россия/Санкт-Петербург/2022/spb.jpg")


class TestNoRegressionInTheOtherSegments(CityNamesTestBase):
    """Requirement 5/7: only the city segment changed."""

    def test_country_and_district_and_year_are_untouched(self):
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB, district_geonameid=_GID_AKADEM)
        self.assertEqual(self.target_of("ru", "spb.jpg"),
                         "Россия/Санкт-Петербург/2022/Академическое/spb.jpg")

    def test_online_country_name_still_wins_over_the_iso_code(self):
        self.add_file("spb.jpg", country="RU", country_name="Российская Федерация",
                      city="Saint Petersburg", city_geonameid=_GID_SPB)
        self.assertEqual(self.target_of("ru", "spb.jpg"),
                         "Российская Федерация/Санкт-Петербург/2022/spb.jpg")

    def test_unlocalized_district_is_still_dropped(self):
        self.add_file("phuket.jpg", country="TH", city="Kapong",
                      city_geonameid=_GID_KAPONG, district_geonameid=_GID_WICHIT)
        self.assertEqual(self.target_of("ru", "phuket.jpg"),
                         "Таиланд/Kapong/2022/phuket.jpg")

    def test_service_folders_are_not_affected_by_the_city_name(self):
        # A document/an undated file/a file without a place never reaches the city
        # branch — their folders come from i18n.folder and stay where they were.
        self.add_file("doc.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB, junk_verdict="document")
        self.add_file("noplace.jpg")
        self.add_file("nodate.jpg", taken_at=None, country="RU",
                      city="Saint Petersburg", city_geonameid=_GID_SPB,
                      camera_make="Canon")
        targets = {it.src.name: it.target_rel for it in self.plan("ru")}
        self.assertEqual(targets["doc.jpg"], "_Документы/doc.jpg")
        self.assertEqual(targets["noplace.jpg"], "_Неразобрано/без_места/noplace.jpg")
        self.assertEqual(targets["nodate.jpg"], "_Неразобрано/без_даты/nodate.jpg")

    def test_event_mode_does_not_use_the_city(self):
        fid = self.add_file("ev.jpg", country="RU", city="Saint Petersburg",
                            city_geonameid=_GID_SPB)
        self.add_event(fid, "Поездка")
        target = next(it.target_rel for it in self.plan("ru", mode="event"))
        self.assertEqual(target, "2022/Поездка/ev.jpg")


class TestResolverBuiltOncePerPlan(CityNamesTestBase):
    """Requirement 2: `places.tsv` is 170 472 rows — one load per plan, not per row."""

    def test_constructed_once_for_a_plan_of_many_files_in_many_cities(self):
        cities = [(_GID_SPB, "RU", "Saint Petersburg"),
                  (_GID_KAPONG, "TH", "Kapong"),
                  (_GID_JABAJERO, "ID", "Jabajero"),
                  (_GID_STALE, "RU", "Mystery Town")]
        for i in range(40):
            gid, cc, city = cities[i % len(cities)]
            self.add_file(f"f{i:02d}.jpg", content=f"data{i}".encode(),
                          country=cc, city=city, city_geonameid=gid)
        with patch("sorta.sorter.GeoResolver") as mock_cls:
            mock_cls.return_value = self.resolver
            report = plan_and_sort(self.config("ru"), self.conn, "city", self.dest,
                                   apply=False, write_reports=False)
        self.assertEqual(len(report.plan), 40)
        self.assertEqual(mock_cls.call_count, 1)


class TestLanguageSwitchNeedsNoGeoPass(CityNamesTestBase):
    """Requirement 7: the anchor and the geonameid in the DB do not move — only the
    display does. That is the whole point of G3."""

    def _places_snapshot(self) -> list[tuple]:
        return [tuple(r) for r in self.conn.execute(
            "SELECT file_id, country, city, city_geonameid, confidence, updated_at "
            "FROM places ORDER BY file_id")]

    def test_two_plans_in_two_languages_differ_without_touching_places(self):
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB)
        before = self._places_snapshot()
        ru_target = self.target_of("ru", "spb.jpg")
        ja_target = self.target_of("ja", "spb.jpg")
        en_target = self.target_of("en", "spb.jpg")
        self.assertEqual(ru_target, "Россия/Санкт-Петербург/2022/spb.jpg")
        self.assertEqual(ja_target, "ロシア/サンクトペテルブルク/2022/spb.jpg")
        self.assertEqual(en_target, "Russia/Saint Petersburg/2022/spb.jpg")
        self.assertEqual(len({ru_target, ja_target, en_target}), 3)
        self.assertEqual(self._places_snapshot(), before)


class TestPlanItemAndCsvCarryTheSameName(CityNamesTestBase):
    """Requirement 6: the report says the same city the folder does."""

    def test_plan_item_city_is_localized(self):
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB)
        item = self.plan("ru")[0]
        self.assertEqual(item.city, "Санкт-Петербург")
        self.assertEqual(item.country, "RU")  # the country is localized when formatted

    def test_csv_city_column_matches_the_folder(self):
        self.add_file("spb.jpg", country="RU", city="Saint Petersburg",
                      city_geonameid=_GID_SPB)
        with patch("sorta.sorter.GeoResolver", return_value=self.resolver):
            report = plan_and_sort(self.config("ru"), self.conn, "city", self.dest,
                                   apply=False)
        row = self.read_csv(report.csv_path)[0]
        self.assertEqual(row["city"], "Санкт-Петербург")
        self.assertIn("Санкт-Петербург", row["target"])

    def test_city_of_a_row_without_geonameid_is_reported_as_stored(self):
        self.add_file("tower.jpg", country="FR", city="Эйфелева башня")
        self.assertEqual(self.plan("ru")[0].city, "Эйфелева башня")


if __name__ == "__main__":
    unittest.main()
