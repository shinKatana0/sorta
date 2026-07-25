"""G1 (F26): the offline geo-resolver (a tiny fixture, without the real 12 MB)."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from sorta import geodata
from sorta.geodata import GeoDataMissing, GeoResolver

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

    def test_places_file_without_usable_rows_resolves_to_none(self, tmp_path: Path) -> None:
        # The file is there but empty (an interrupted build): no data to resolve
        # against, yet nothing to shout about either — the resolve is simply empty.
        data_dir = tmp_path / "geo_blank"
        data_dir.mkdir()
        (data_dir / "places.tsv").write_text("", encoding="utf-8")
        res = GeoResolver(data_dir=data_dir).resolve(59.9311, 30.3609)
        assert (res.country_cc, res.city_id, res.district_id) == (None, None, None)


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

    def test_missing_data_raises_instead_of_faking_a_name(self, tmp_path: Path) -> None:
        empty = GeoResolver(data_dir=tmp_path / "does_not_exist")
        with pytest.raises(GeoDataMissing):
            empty.name(100, "ru")


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

    def test_missing_data_raises(self, tmp_path: Path) -> None:
        empty = GeoResolver(data_dir=tmp_path / "does_not_exist")
        with pytest.raises(GeoDataMissing):
            empty.has_localized_name(100, "ru")

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

    def test_country_cc_by_name_missing_data_raises(self, tmp_path: Path) -> None:
        empty = GeoResolver(data_dir=tmp_path / "does_not_exist")
        with pytest.raises(GeoDataMissing):
            empty.country_cc_by_name("Россия", "ru")

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

    def test_city_ids_by_name_missing_data_raises(self, tmp_path: Path) -> None:
        empty = GeoResolver(data_dir=tmp_path / "does_not_exist")
        with pytest.raises(GeoDataMissing):
            empty.city_ids_by_name("Москва", "ru")


class TestMissingData:
    """F65 regression: a missing places.tsv used to be a silent `return` — the resolver
    then answered "nowhere" to every coordinate and the whole run lost its geo truth."""

    def test_data_available_false_without_places(self, tmp_path: Path) -> None:
        assert GeoResolver(data_dir=tmp_path / "does_not_exist").data_available() is False

    def test_data_available_true_with_fixture(self, resolver: GeoResolver) -> None:
        assert resolver.data_available() is True

    def test_data_available_loads_nothing(self, tmp_path: Path) -> None:
        # cheap check: it must not touch the data (a resolve after it still works)
        data_dir = tmp_path / "geo"
        _write_fixture(data_dir)
        r = GeoResolver(data_dir=data_dir)
        assert r.data_available() is True
        assert r._loaded is False

    def test_data_dir_is_exposed(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "geo"
        _write_fixture(data_dir)
        assert GeoResolver(data_dir=data_dir).data_dir == data_dir

    def test_resolve_raises_with_actionable_message(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(GeoDataMissing) as exc:
            GeoResolver(data_dir=missing).resolve(50.0875, 14.4213)
        message = str(exc.value)
        assert str(missing / "places.tsv") in message      # where we looked
        assert "scripts/build_geodata.py" in message       # what to do

    def test_error_is_a_filenotfounderror(self, tmp_path: Path) -> None:
        # callers that already catch FileNotFoundError keep working
        with pytest.raises(FileNotFoundError):
            GeoResolver(data_dir=tmp_path / "nope")._ensure_loaded()

    def test_keeps_raising_on_every_call(self, tmp_path: Path) -> None:
        # the failed load must not mark the resolver as "loaded" — otherwise the
        # second call is silent again and we are back to the original bug
        r = GeoResolver(data_dir=tmp_path / "nope")
        for _ in range(2):
            with pytest.raises(GeoDataMissing):
                r.resolve(50.0875, 14.4213)

    def test_optional_admin1_missing_only_warns(self, tmp_path: Path,
                                                caplog: pytest.LogCaptureFixture) -> None:
        data_dir = tmp_path / "geo_no_admin1"
        _write_fixture(data_dir, admin1=None)
        r = GeoResolver(data_dir=data_dir)
        with caplog.at_level(logging.WARNING, logger="sorta.geodata"):
            res = r.resolve(59.9350, 30.3700)
        assert res.city_id == 100                      # loading went through
        assert r.region_name("RU", "66", "ru") is None  # just without region names
        warnings = [rec for rec in caplog.records if "admin1.tsv" in rec.getMessage()]
        assert len(warnings) == 1
        assert str(data_dir / "admin1.tsv") in warnings[0].getMessage()

    def test_optional_files_warn_once_each(self, tmp_path: Path,
                                           caplog: pytest.LogCaptureFixture) -> None:
        data_dir = tmp_path / "geo_places_only"
        _write_fixture(data_dir, names=[], admin1=None, countries=None)
        (data_dir / "names.tsv").unlink()
        r = GeoResolver(data_dir=data_dir)
        with caplog.at_level(logging.WARNING, logger="sorta.geodata"):
            r.resolve(59.9350, 30.3700)
            r.resolve(55.7558, 37.6173)  # data is loaded once -> no second round of warnings
        records = [rec for rec in caplog.records if rec.name == "sorta.geodata"]
        assert len(records) == 3  # admin1 + countries + names


class TestDefaultDataDir:
    """F65: the data must be found from ANY working directory — it lives in the package."""

    def test_default_dir_points_inside_the_package(self) -> None:
        assert geodata._DEFAULT_DATA_DIR.parent.parent.name == "sorta"
        assert geodata._DEFAULT_DATA_DIR.parts[-2:] == ("data", "geo")

    def test_bundled_places_file_exists_in_the_tree(self) -> None:
        assert (geodata._DEFAULT_DATA_DIR / "places.tsv").is_file()

    def test_default_resolver_uses_the_package_dir(self) -> None:
        assert GeoResolver().data_dir == geodata._DEFAULT_DATA_DIR
        assert GeoResolver().data_available() is True

    def test_falls_back_to_the_repo_layout(self, tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
        # a working tree from before the move: no sorta/data/geo, but data/geo is there
        legacy = tmp_path / "repo_data_geo"
        _write_fixture(legacy)
        monkeypatch.setattr(geodata, "_DEFAULT_DATA_DIR", tmp_path / "no_package_data")
        monkeypatch.setattr(geodata, "_LEGACY_DATA_DIR", legacy)
        assert GeoResolver().data_dir == legacy
        assert GeoResolver().resolve(59.9311, 30.3609).city_id == 100

    def test_package_dir_wins_over_the_repo_layout(self, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
        package = tmp_path / "package_data_geo"
        _write_fixture(package)
        legacy = tmp_path / "repo_data_geo"
        _write_fixture(legacy)
        monkeypatch.setattr(geodata, "_DEFAULT_DATA_DIR", package)
        monkeypatch.setattr(geodata, "_LEGACY_DATA_DIR", legacy)
        assert GeoResolver().data_dir == package

    def test_error_points_at_the_package_dir_when_nothing_is_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        package = tmp_path / "package_data_geo"
        monkeypatch.setattr(geodata, "_DEFAULT_DATA_DIR", package)
        monkeypatch.setattr(geodata, "_LEGACY_DATA_DIR", tmp_path / "repo_data_geo")
        assert GeoResolver().data_dir == package
        with pytest.raises(GeoDataMissing, match="places.tsv"):
            GeoResolver().resolve(0.0, 0.0)


class TestBundledData:
    """The acceptance scenario: the real bundled base resolves Prague to CZ.

    Slow-ish (the whole 12 MB is loaded) — deliberately ONE test, this is exactly the
    case that was broken outside the repository root.
    """

    def test_prague_resolves_to_cz(self) -> None:
        res = GeoResolver().resolve(50.0875, 14.4213)
        assert res.country_cc == "CZ"
        assert res.city_id is not None
