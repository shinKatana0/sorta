"""G1 (F26): the offline geo-resolver (a tiny fixture, without the real 12 MB)."""
from __future__ import annotations

from pathlib import Path

import pytest

from sorta.geodata import GeoResolver

# A tiny world: a city (PPLA) in SPb + two districts (PPLX) nearby + a capital
# farther away (PPLC) + a city with no records in names.tsv at all.
SAINT_PETERSBURG = (100, 59.9311, 30.3609, "PPLA", "RU", "66", "", "Saint Petersburg", "5000000")
AKADEMICHESKOE = (101, 59.9350, 30.3700, "PPLX", "RU", "66", "", "Akademicheskoe", "50000")
KRESTOVSKY = (102, 59.9500, 30.2000, "PPLX", "RU", "66", "", "Krestovsky Island", "10000")
MOSCOW = (200, 55.7558, 37.6173, "PPLC", "RU", "48", "", "Moscow", "12000000")
BANGKOK = (300, 13.7563, 100.5018, "PPLC", "TH", "40", "", "Bangkok", "8000000")
# F46: a Moscow namesake (like the real Moscow, Idaho in the bundled data) — en name
# only (no ru), far from the other fixtures, so as not to affect the resolve() tests.
MOSCOW_US = (250, 46.7324, -117.0002, "PPLA2", "US", "16", "", "Moscow", "25000")

PLACES = [SAINT_PETERSBURG, AKADEMICHESKOE, KRESTOVSKY, MOSCOW, BANGKOK, MOSCOW_US]

# admin1 regions: cc, admin1, geonameid, name_en (as from admin1CodesASCII)
ADMIN1 = [
    ("RU", "66", 400, "Sankt-Peterburg"),
    ("TH", "40", 500, "Bangkok"),   # no ru in names -> region_name falls back to en
]
# countries: cc, geonameid, name_en (as from countryInfo)
COUNTRIES = [
    ("RU", 600, "Russia"),
    ("TH", 700, "Thailand"),
]

NAMES = [
    (100, "ru", "Санкт-Петербург"),
    (100, "en", "Saint Petersburg"),
    (101, "en", "Akademicheskoe"),  # no ru -> fallback to en
    (200, "ru", "Москва"),
    (200, "en", "Moscow"),
    (400, "ru", "Санкт-Петербург"),  # region name RU.66
    (600, "ru", "Россия"),            # country name RU
    (700, "ru", "Таиланд"),           # country name TH
    (250, "en", "Moscow"),            # F46: a namesake (en only, no ru) -> homonym
    # 102 (Krestovsky) is not in names.tsv at all -> fallback to asciiname
    # 300 (Bangkok) is not in names.tsv at all -> fallback to asciiname
    # 500 (the TH.40 region) is not in names -> region_name falls back to name_en «Bangkok»
]


def _write_fixture(data_dir: Path, places=PLACES, names=NAMES,
                   admin1=ADMIN1, countries=COUNTRIES) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "places.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for row in places:
            f.write("\t".join(str(v) for v in row) + "\n")
    with (data_dir / "names.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for gid, lang, name in names:
            f.write(f"{gid}\t{lang}\t{name}\n")
    if admin1 is not None:
        with (data_dir / "admin1.tsv").open("w", encoding="utf-8", newline="\n") as f:
            for cc, a1, gid, name_en in admin1:
                f.write(f"{cc}\t{a1}\t{gid}\t{name_en}\n")
    if countries is not None:
        with (data_dir / "countries.tsv").open("w", encoding="utf-8", newline="\n") as f:
            for cc, gid, name_en in countries:
                f.write(f"{cc}\t{gid}\t{name_en}\n")


@pytest.fixture
def resolver(tmp_path: Path) -> GeoResolver:
    data_dir = tmp_path / "geo"
    _write_fixture(data_dir)
    return GeoResolver(data_dir=data_dir)


class TestResolve:
    def test_district_near_city_resolves_both_levels(self, resolver: GeoResolver) -> None:
        res = resolver.resolve(59.9350, 30.3700)  # exactly Akademicheskoe's coordinates
        assert res.city_id == 100  # the nearest PPLA
        assert res.district_id == 101  # a district = its own nearest place

    def test_second_district_near_same_city(self, resolver: GeoResolver) -> None:
        res = resolver.resolve(59.9500, 30.2000)  # Krestovsky's coordinates
        assert res.city_id == 100
        assert res.district_id == 102

    def test_city_center_has_no_separate_district(self, resolver: GeoResolver) -> None:
        res = resolver.resolve(59.9311, 30.3609)  # exactly the city centre
        assert res.city_id == 100
        assert res.district_id is None  # would coincide with city_id -> None

    def test_far_away_point_resolves_remote_city(self, resolver: GeoResolver) -> None:
        res = resolver.resolve(55.7558, 37.6173)  # exactly Moscow
        assert res.city_id == 200
        assert res.district_id is None

    def test_country_cc_from_nearest_place(self, resolver: GeoResolver) -> None:
        res = resolver.resolve(13.7563, 100.5018)  # Bangkok
        assert res.country_cc == "TH"
        assert res.city_id == 300

    def test_empty_data_dir_never_raises(self, tmp_path: Path) -> None:
        empty = GeoResolver(data_dir=tmp_path / "does_not_exist")
        res = empty.resolve(59.9311, 30.3609)
        assert res == (None, None, None) or (res.country_cc is None and res.city_id is None
                                               and res.district_id is None)

    def test_data_dir_exists_but_files_missing(self, tmp_path: Path) -> None:
        d = tmp_path / "geo_empty"
        d.mkdir()
        empty = GeoResolver(data_dir=d)
        res = empty.resolve(0.0, 0.0)
        assert res.country_cc is None
        assert res.city_id is None
        assert res.district_id is None


class TestName:
    def test_requested_lang_present(self, resolver: GeoResolver) -> None:
        assert resolver.name(100, "ru") == "Санкт-Петербург"

    def test_falls_back_to_en_when_lang_missing(self, resolver: GeoResolver) -> None:
        # 101 (Akademicheskoe) has en only, we request ru
        assert resolver.name(101, "ru") == "Akademicheskoe"

    def test_falls_back_to_asciiname_when_no_names_at_all(self, resolver: GeoResolver) -> None:
        # 102 (Krestovsky) is not in names.tsv at all
        assert resolver.name(102, "ru") == "Krestovsky Island"
        assert resolver.name(102, "ja") == "Krestovsky Island"

    def test_unknown_geonameid_falls_back_to_id_string(self, resolver: GeoResolver) -> None:
        assert resolver.name(999999, "ru") == "999999"

    def test_never_returns_empty_string(self, resolver: GeoResolver) -> None:
        for gid in (100, 101, 102, 200, 300, 424242):
            for lang in ("ru", "en", "ja"):
                assert resolver.name(gid, lang) != ""

    def test_lang_case_insensitive(self, resolver: GeoResolver) -> None:
        assert resolver.name(100, "RU") == resolver.name(100, "ru")  # type: ignore[arg-type]

    def test_empty_resolver_falls_back_to_id_string(self, tmp_path: Path) -> None:
        empty = GeoResolver(data_dir=tmp_path / "does_not_exist")
        assert empty.name(100, "ru") == "100"


class TestHasLocalizedName:
    """F49: has_localized_name distinguishes "there is a ru name" from name()'s
    fallback to en/asciiname — needed by the layout to drop transliterated districts."""

    def test_true_when_lang_present(self, resolver: GeoResolver) -> None:
        assert resolver.has_localized_name(100, "ru") is True  # Saint Petersburg

    def test_false_when_only_en_present(self, resolver: GeoResolver) -> None:
        # 101 (Akademicheskoe) en only -> name() would fall back, has_localized_name = False
        assert resolver.has_localized_name(101, "ru") is False

    def test_false_when_no_names_at_all(self, resolver: GeoResolver) -> None:
        # 102 (Krestovsky) is not in names.tsv at all -> name() would give asciiname
        assert resolver.has_localized_name(102, "ru") is False

    def test_false_for_unknown_geonameid(self, resolver: GeoResolver) -> None:
        assert resolver.has_localized_name(999999, "ru") is False

    def test_false_for_empty_resolver(self, tmp_path: Path) -> None:
        empty = GeoResolver(data_dir=tmp_path / "does_not_exist")
        assert empty.has_localized_name(100, "ru") is False

    def test_true_for_en_when_en_present(self, resolver: GeoResolver) -> None:
        assert resolver.has_localized_name(101, "en") is True


class TestRegionAccessors:
    """G-#19: coords/region/country accessors for merging and trip names."""

    def test_coords_of_known_place(self, resolver: GeoResolver) -> None:
        lat, lon = resolver.coords_of(100)
        assert lat == pytest.approx(59.9311)
        assert lon == pytest.approx(30.3609)

    def test_coords_of_unknown_is_none(self, resolver: GeoResolver) -> None:
        assert resolver.coords_of(999999) is None

    def test_region_key_of_city(self, resolver: GeoResolver) -> None:
        assert resolver.region_key_of(100) == ("RU", "66")
        assert resolver.region_key_of(300) == ("TH", "40")

    def test_region_key_of_unknown_is_none(self, resolver: GeoResolver) -> None:
        assert resolver.region_key_of(999999) is None

    def test_region_name_localized(self, resolver: GeoResolver) -> None:
        assert resolver.region_name("RU", "66", "ru") == "Санкт-Петербург"

    def test_region_name_falls_back_to_en_name(self, resolver: GeoResolver) -> None:
        # TH.40 is not in names.tsv -> name_en «Bangkok» from admin1.tsv
        assert resolver.region_name("TH", "40", "ru") == "Bangkok"

    def test_region_name_unknown_is_none(self, resolver: GeoResolver) -> None:
        assert resolver.region_name("XX", "99", "ru") is None

    def test_country_name_localized(self, resolver: GeoResolver) -> None:
        assert resolver.country_name("RU", "ru") == "Россия"
        assert resolver.country_name("TH", "ru") == "Таиланд"

    def test_country_name_unknown_is_none(self, resolver: GeoResolver) -> None:
        assert resolver.country_name("XX", "ru") is None

    def test_accessors_none_without_bundled_files(self, tmp_path: Path) -> None:
        # old bundled data without admin1.tsv/countries.tsv -> accessors None,
        # but coords/region_key from places.tsv still work
        data_dir = tmp_path / "geo_old"
        _write_fixture(data_dir, admin1=None, countries=None)
        r = GeoResolver(data_dir=data_dir)
        assert r.region_name("RU", "66", "ru") is None
        assert r.country_name("RU", "ru") is None
        assert r.region_key_of(100) == ("RU", "66")
        assert r.coords_of(100) is not None


class TestReverseLookups:
    """F46: a name (in the config language) -> ISO cc / geonameids, for a localized --where."""

    def test_country_cc_by_name_localized(self, resolver: GeoResolver) -> None:
        assert resolver.country_cc_by_name("Россия", "ru") == "RU"

    def test_country_cc_by_name_en(self, resolver: GeoResolver) -> None:
        assert resolver.country_cc_by_name("Russia", "en") == "RU"

    def test_country_cc_by_name_case_insensitive(self, resolver: GeoResolver) -> None:
        assert resolver.country_cc_by_name("россия", "ru") == "RU"
        assert resolver.country_cc_by_name("РОССИЯ", "ru") == "RU"

    def test_country_cc_by_name_unknown_is_none(self, resolver: GeoResolver) -> None:
        assert resolver.country_cc_by_name("Wakanda", "ru") is None

    def test_country_cc_by_name_empty_resolver(self, tmp_path: Path) -> None:
        empty = GeoResolver(data_dir=tmp_path / "does_not_exist")
        assert empty.country_cc_by_name("Россия", "ru") is None

    def test_city_ids_by_name_localized(self, resolver: GeoResolver) -> None:
        assert resolver.city_ids_by_name("Москва", "ru") == [200]

    def test_city_ids_by_name_en(self, resolver: GeoResolver) -> None:
        assert resolver.city_ids_by_name("Saint Petersburg", "en") == [100]

    def test_city_ids_by_name_case_insensitive(self, resolver: GeoResolver) -> None:
        assert resolver.city_ids_by_name("москва", "ru") == [200]

    def test_city_ids_by_name_unknown_is_empty(self, resolver: GeoResolver) -> None:
        assert resolver.city_ids_by_name("Atlantis", "ru") == []

    def test_city_ids_by_name_homonyms(self, resolver: GeoResolver) -> None:
        # Moscow (RU, 200) and Moscow (US Idaho, 250) — both "Moscow" in en.
        assert sorted(resolver.city_ids_by_name("Moscow", "en")) == [200, 250]

    def test_city_ids_by_name_district_excluded(self, resolver: GeoResolver) -> None:
        # Krestovsky Island — PPLX (a district), not a city_id -> must not resolve.
        assert resolver.city_ids_by_name("Krestovsky Island", "en") == []

    def test_city_ids_by_name_empty_resolver(self, tmp_path: Path) -> None:
        empty = GeoResolver(data_dir=tmp_path / "does_not_exist")
        assert empty.city_ids_by_name("Москва", "ru") == []
