"""F93: the online provider's answers live in the DB (geo_cache), not in the process.

`geo` recomputes places from scratch on every run — session inheritance looks at
neighbours in time, so a partial recompute would give a different result — and the
in-memory cache of the resolver died with the process. Adding 200 photos therefore cost
the same ~35 minutes of Nominatim as a full run. On top of that the language was a
property of the RUN (`accept-language`), so switching folder language left the cities in
the old language until the next full pass.

Both halves are pinned here: the cache key (the city+district pair of the bundled base,
which beats any coordinate grid on both speed and accuracy), the three languages fetched
in one go, what must NOT be cached, and the escape hatches — `--clear-geo` and the
checkbox of the reset dialog, without which a wrong cached answer would be unreachable.

No real network and no real bundled data: urllib and geodata.GeoResolver are faked.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from sorta import db, ui
from sorta.config import Config, GeoConfig, SortConfig
from sorta.db import connect
from sorta.geo import clear_geo_cache, geo_cache_size, resolve_places
from sorta.geodata import Resolution
from sorta.sorter import plan_and_sort

from tests.test_ui import UiServerTestBase
from tests.test_ui_process import ProcessTestBase

# fixture geonameids (arbitrary, just stable): ONE city, TWO districts — the pair that
# the cache key is built from
_GID_CITY = 524901
_GID_NORTH = 1111
_GID_SOUTH = 2222

_NORTH = (55.80, 37.60)
_NORTH_NEIGHBOUR = (55.8004, 37.6004)  # another frame of the same district
_SOUTH = (55.70, 37.60)
_OPEN_SEA = (13.5, 92.5)  # the local base places no city here -> the grid fallback key

_OFFLINE_NAMES = {_GID_CITY: "Moscow", _GID_NORTH: "Severny", _GID_SOUTH: "Yuzhny"}

# What Nominatim answers, per district and per language. The ja variants are deliberately
# real translations: the whole point of the second half of F93 is that they are stored
# next to the ru ones instead of being asked for again later.
_ADDRESSES = {
    _GID_NORTH: {
        "ru": {"city": "Москва", "suburb": "Северный", "country": "Россия"},
        "en": {"city": "Moscow", "suburb": "Severny", "country": "Russia"},
        "ja": {"city": "モスクワ", "suburb": "セーヴェルヌイ", "country": "ロシア"},
    },
    _GID_SOUTH: {
        "ru": {"city": "Москва", "suburb": "Южный", "country": "Россия"},
        "en": {"city": "Moscow", "suburb": "Yuzhny", "country": "Russia"},
        "ja": {"city": "モスクワ", "suburb": "ユージヌイ", "country": "ロシア"},
    },
    None: {  # the open sea: a country, no settlement
        "ru": {"country": "Индия"},
        "en": {"country": "India"},
        "ja": {"country": "インド"},
    },
}


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _FakeLocal:
    """geodata.GeoResolver without the bundled data — the source of the cache key."""

    data_dir = Path("/bundled/geo")

    def __init__(self, available: bool = True) -> None:
        self._available = available

    def data_available(self) -> bool:
        return self._available

    @staticmethod
    def district_of(lat: float) -> int | None:
        if lat < 20:  # the open sea — nothing to place it against
            return None
        return _GID_NORTH if lat >= 55.75 else _GID_SOUTH

    def resolve(self, lat: float, lon: float) -> Resolution:
        district = self.district_of(lat)
        if district is None:
            return Resolution(country_cc=None, city_id=None, district_id=None)
        return Resolution(country_cc="RU", city_id=_GID_CITY, district_id=district)

    def name(self, geonameid: int, lang: str) -> str:
        return _OFFLINE_NAMES[geonameid]

    def has_localized_name(self, geonameid: int, lang: str) -> bool:
        return True


class _FakeNominatim:
    """The faked reverse endpoint: an answer per (point, requested language).

    Records every request, so the tests can count how much network a run really cost.
    """

    def __init__(self, *, fail_langs: tuple[str, ...] = (),
                 drop_city_langs: tuple[str, ...] = (), empty: bool = False) -> None:
        self.calls: list[tuple[float, float, str]] = []
        self._fail_langs = fail_langs
        self._drop_city_langs = drop_city_langs
        self._empty = empty

    @property
    def langs(self) -> list[str]:
        return [lang for _lat, _lon, lang in self.calls]

    def __call__(self, req: object, timeout: float | None = None) -> _FakeResponse:
        query = urllib.parse.urlparse(req.full_url).query  # type: ignore[attr-defined]
        params = urllib.parse.parse_qs(query)
        lat, lon = float(params["lat"][0]), float(params["lon"][0])
        lang = params["accept-language"][0]
        self.calls.append((lat, lon, lang))
        if lang in self._fail_langs:
            raise OSError("boom")
        if self._empty:
            return _FakeResponse({})
        address = dict(_ADDRESSES[_FakeLocal.district_of(lat)][lang])
        address["country_code"] = "in" if lat < 20 else "ru"
        if lang in self._drop_city_langs:
            address.pop("city", None)
        return _FakeResponse({"address": address})


class GeoCacheTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src_dir = self.root / "src"
        self.src_dir.mkdir()
        self.cfg = Config(sources=[self.src_dir], database=self.root / "test.db",
                          geo=GeoConfig(provider="online"),
                          sort=SortConfig(report_dir=str(self.root / "reports")),
                          language="ru", raw={"language": "ru"})
        self.conn = connect(self.cfg.database)
        self.local = _FakeLocal()
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    # --- fixtures ------------------------------------------------------

    def add_file(self, coords=None, taken_at="2023-05-01T10:00:00"):
        self._n += 1
        lat, lon = coords if coords is not None else (None, None)
        path = self.src_dir / f"img_{self._n}.jpg"
        path.write_bytes(b"data")
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, taken_at,
                   taken_at_source, taken_at_confidence, gps_lat, gps_lon, indexed_at)
               VALUES (?, 4, 0, 'jpg', 'photo', ?, 'exif', 'high', ?, ?, '2026-01-01')""",
            (str(path.resolve()), taken_at, lat, lon),
        )
        self.conn.commit()
        return cur.lastrowid

    def run_geo(self, net: _FakeNominatim):
        with patch("sorta.geo.urllib.request.urlopen", side_effect=net), \
             patch("sorta.geo.GeoResolver", return_value=self.local), \
             patch("sorta.geo.time.sleep"):
            return resolve_places(self.cfg, self.conn, progress=lambda done, total: None)

    def place_of(self, file_id):
        return self.conn.execute(
            """SELECT country, country_name, city, city_geonameid, district_geonameid,
                      district_name, confidence
               FROM places WHERE file_id = ?""", (file_id,)).fetchone()

    def cache_rows(self):
        return self.conn.execute(
            "SELECT * FROM geo_cache ORDER BY key").fetchall()


class TestNetworkIsAskedOnce(GeoCacheTestBase):
    def test_second_run_with_the_same_coordinates_makes_no_requests(self):
        photo = self.add_file(_NORTH)
        first = _FakeNominatim()
        self.run_geo(first)
        self.assertEqual(len(first.calls), 3)  # one pair × three languages

        second = _FakeNominatim()
        self.run_geo(second)
        self.assertEqual(second.calls, [])
        # and the answer is still the same place, not a hole in the DB
        row = self.place_of(photo)
        self.assertEqual(row["city"], "Москва")
        self.assertEqual(row["district_name"], "Северный")
        self.assertEqual(row["confidence"], "exact_gps")

    def test_two_frames_of_one_district_are_one_request_batch(self):
        # this is what the key buys over a grid: the local base already knows the two
        # frames are the same district, whatever the distance between them
        self.add_file(_NORTH)
        self.add_file(_NORTH_NEIGHBOUR)
        net = _FakeNominatim()
        self.run_geo(net)
        self.assertEqual(len(net.calls), 3)
        self.assertEqual(len(self.cache_rows()), 1)

    def test_neighbouring_districts_are_separate_keys(self):
        north = self.add_file(_NORTH)
        south = self.add_file(_SOUTH)
        net = _FakeNominatim()
        self.run_geo(net)
        self.assertEqual(len(net.calls), 6)  # two pairs × three languages
        self.assertEqual(len(self.cache_rows()), 2)
        # the districts did not get glued into one name (0.9% of them did on a 110 m grid)
        self.assertEqual(self.place_of(north)["district_name"], "Северный")
        self.assertEqual(self.place_of(south)["district_name"], "Южный")

    def test_three_languages_asked_in_one_batch_per_miss(self):
        self.add_file(_NORTH)
        self.add_file(_SOUTH)
        net = _FakeNominatim()
        self.run_geo(net)
        self.assertEqual(sorted(net.langs), ["en", "en", "ja", "ja", "ru", "ru"])
        # every request of a group hit the SAME point (the median of the group)
        points = {(lat, lon) for lat, lon, _lang in net.calls}
        self.assertEqual(points, {_NORTH, _SOUTH})

    def test_the_representative_point_is_the_median_of_the_group(self):
        self.add_file(_NORTH)
        self.add_file(_NORTH_NEIGHBOUR)
        self.add_file((55.9, 37.7))  # still the northern district
        net = _FakeNominatim()
        self.run_geo(net)
        self.assertEqual({(lat, lon) for lat, lon, _lang in net.calls},
                         {(_NORTH_NEIGHBOUR[0], _NORTH_NEIGHBOUR[1])})

    def test_the_key_is_the_city_and_district_pair(self):
        self.add_file(_NORTH)
        self.run_geo(_FakeNominatim())
        rows = self.cache_rows()
        self.assertEqual([r["key"] for r in rows], [f"c:{_GID_CITY}/{_GID_NORTH}"])
        self.assertEqual(rows[0]["provider"], "online")


class TestLanguagesLiveInTheValue(GeoCacheTestBase):
    def test_all_three_languages_are_stored(self):
        self.add_file(_NORTH)
        self.run_geo(_FakeNominatim())
        row = self.cache_rows()[0]
        self.assertEqual((row["city_ru"], row["city_en"], row["city_ja"]),
                         ("Москва", "Moscow", "モスクワ"))
        self.assertEqual((row["district_ru"], row["district_en"], row["district_ja"]),
                         ("Северный", "Severny", "セーヴェルヌイ"))
        self.assertEqual((row["country_name_ru"], row["country_name_en"],
                          row["country_name_ja"]), ("Россия", "Russia", "ロシア"))
        self.assertEqual(row["country"], "RU")

    def test_switching_language_costs_no_network(self):
        photo = self.add_file(_NORTH)
        self.run_geo(_FakeNominatim())

        self.cfg.language = "ja"
        net = _FakeNominatim()
        self.run_geo(net)
        self.assertEqual(net.calls, [])
        row = self.place_of(photo)
        self.assertEqual(row["city"], "モスクワ")
        self.assertEqual(row["district_name"], "セーヴェルヌイ")
        self.assertEqual(row["country_name"], "ロシア")

    def test_the_plan_is_rebuilt_in_the_new_language(self):
        # the point of the whole second half: the folders change language, and the only
        # thing it takes is a geo re-run that never touches the network
        self.add_file(_NORTH)
        self.run_geo(_FakeNominatim())
        report = plan_and_sort(self.cfg, self.conn, "city", self.root / "dest", apply=False)
        self.assertEqual(report.plan[0].target_rel, "Россия/Москва/2023/Северный/img_1.jpg")

        self.cfg.language = "ja"
        net = _FakeNominatim()
        self.run_geo(net)
        report_ja = plan_and_sort(self.cfg, self.conn, "city", self.root / "dest",
                                  apply=False)
        self.assertEqual(net.calls, [])
        self.assertEqual(report_ja.plan[0].target_rel,
                         "ロシア/モスクワ/2023/セーヴェルヌイ/img_1.jpg")

    def test_missing_language_variant_falls_back_to_the_latin_name(self):
        # OSM has no `name:ja` for a Balinese village: the ja answer carries no city at
        # all. An honest fallback to what IS there beats an empty folder name.
        self.cfg.language = "ja"
        photo = self.add_file(_NORTH)
        self.run_geo(_FakeNominatim(drop_city_langs=("ja",)))
        row = self.place_of(photo)
        self.assertEqual(row["city"], "Moscow")  # the en variant, not None
        self.assertEqual(row["district_name"], "セーヴェルヌイ")  # ja where ja exists
        self.assertEqual(row["confidence"], "exact_gps")
        # the answer WAS complete (every language named a country), so it is cached
        self.assertIsNone(self.cache_rows()[0]["city_ja"])


class TestFailedAnswersAreNotCached(GeoCacheTestBase):
    def test_one_failed_language_leaves_the_row_unwritten(self):
        self.add_file(_NORTH)
        first = _FakeNominatim(fail_langs=("ja",))
        self.run_geo(first)
        self.assertEqual(self.cache_rows(), [])

        # the next run tries again — a bad network minute must not become permanent
        second = _FakeNominatim()
        self.run_geo(second)
        self.assertEqual(len(second.calls), 3)
        self.assertEqual(len(self.cache_rows()), 1)

    def test_a_failed_language_does_not_spoil_the_current_run(self):
        # the run still gets the answer it did receive: ru was asked and answered
        photo = self.add_file(_NORTH)
        self.run_geo(_FakeNominatim(fail_langs=("ja",)))
        self.assertEqual(self.place_of(photo)["city"], "Москва")

    def test_empty_address_is_not_cached_and_stays_unknown(self):
        photo = self.add_file(_NORTH)
        stats = self.run_geo(_FakeNominatim(empty=True))
        self.assertEqual(self.cache_rows(), [])
        self.assertEqual(self.place_of(photo)["confidence"], "unknown")
        self.assertEqual(stats.gps_unresolved, 1)


class TestKeyShapes(GeoCacheTestBase):
    def test_provider_is_part_of_the_key(self):
        # offline and online answers must never be mixed: a row of another provider is
        # not an answer to our question, even under the same key
        self.conn.execute(
            """INSERT INTO geo_cache (provider, key, country, country_name_ru, city_ru,
                                      district_ru, updated_at)
               VALUES ('offline', ?, 'XX', 'Нигде', 'Нигдеград', 'Нигдеевский', ?)""",
            (f"c:{_GID_CITY}/{_GID_NORTH}",
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        self.conn.commit()
        photo = self.add_file(_NORTH)
        net = _FakeNominatim()
        self.run_geo(net)
        self.assertEqual(len(net.calls), 3)  # the foreign row was not read
        self.assertEqual(self.place_of(photo)["city"], "Москва")
        providers = {r["provider"] for r in self.cache_rows()}
        self.assertEqual(providers, {"offline", "online"})  # nor was it overwritten

    def test_coordinates_the_local_base_cannot_place_use_the_grid_key(self):
        photo = self.add_file(_OPEN_SEA)
        self.run_geo(_FakeNominatim())
        keys = [r["key"] for r in self.cache_rows()]
        self.assertEqual(keys, ["g:13.5/92.5"])  # cache_coord_digits = 3
        self.assertEqual(self.place_of(photo)["country"], "IN")

    def test_grid_key_is_reused_on_the_second_run(self):
        self.add_file(_OPEN_SEA)
        self.run_geo(_FakeNominatim())
        net = _FakeNominatim()
        self.run_geo(net)
        self.assertEqual(net.calls, [])

    def test_district_of_none_is_a_valid_key(self):
        # a city with no district: the key must stay distinguishable, not collapse into
        # the city-only shape of another row
        with patch.object(_FakeLocal, "resolve",
                          lambda self, lat, lon: Resolution("RU", _GID_CITY, None)):
            self.add_file(_NORTH)
            self.run_geo(_FakeNominatim())
        self.assertEqual([r["key"] for r in self.cache_rows()], [f"c:{_GID_CITY}/-"])


class TestExpiry(GeoCacheTestBase):
    def _backdate(self, days: int) -> None:
        stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
            timespec="seconds")
        self.conn.execute("UPDATE geo_cache SET updated_at = ?", (stamp,))
        self.conn.commit()

    def test_an_expired_row_is_asked_again_and_rewritten(self):
        self.add_file(_NORTH)
        self.run_geo(_FakeNominatim())
        self._backdate(self.cfg.geo.cache_max_age_days + 1)

        net = _FakeNominatim()
        self.run_geo(net)
        self.assertEqual(len(net.calls), 3)
        self.assertEqual(len(self.cache_rows()), 1)  # rewritten, not duplicated
        fresh = _parse_stamp(self.cache_rows()[0]["updated_at"])
        self.assertLess(datetime.now(timezone.utc) - fresh, timedelta(days=1))

    def test_a_row_within_the_term_is_kept(self):
        self.add_file(_NORTH)
        self.run_geo(_FakeNominatim())
        self._backdate(self.cfg.geo.cache_max_age_days - 1)
        net = _FakeNominatim()
        self.run_geo(net)
        self.assertEqual(net.calls, [])

    def test_zero_turns_the_expiry_off(self):
        self.cfg.geo = GeoConfig(provider="online", cache_max_age_days=0)
        self.add_file(_NORTH)
        self.run_geo(_FakeNominatim())
        self._backdate(4000)
        net = _FakeNominatim()
        self.run_geo(net)
        self.assertEqual(net.calls, [])


class TestOfflineProviderIgnoresTheTable(GeoCacheTestBase):
    def test_offline_neither_reads_nor_writes_the_cache(self):
        self.cfg.geo = GeoConfig(provider="offline")
        self.conn.execute(
            """INSERT INTO geo_cache (provider, key, country, country_name_ru, city_ru,
                                      updated_at)
               VALUES ('online', ?, 'XX', 'Нигде', 'Нигдеград', ?)""",
            (f"c:{_GID_CITY}/{_GID_NORTH}",
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        self.conn.commit()
        photo = self.add_file(_NORTH)
        # urlopen is not even patched with an answer: an offline run must not reach it
        with patch("sorta.geo.urllib.request.urlopen",
                   side_effect=AssertionError("offline must not use the network")), \
             patch("sorta.geo.GeoResolver", return_value=self.local):
            resolve_places(self.cfg, self.conn, progress=lambda done, total: None)
        row = self.place_of(photo)
        self.assertEqual(row["city"], "Moscow")           # from the bundled base
        self.assertEqual(row["city_geonameid"], _GID_CITY)
        self.assertIsNone(row["country_name"])            # not the cached "Нигде"
        self.assertEqual(len(self.cache_rows()), 1)       # the table is untouched


def _parse_stamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


class TestResetKeepsTheCache(unittest.TestCase):
    """`reset_index` is about the user's files; the name of a point on the map is not."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = connect(Path(self.tmp.name) / "test.db")
        self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)
               VALUES ('/photos/a.jpg', 1, 0, 'jpg', 'photo', '2026-01-01')""")
        self.conn.execute(
            """INSERT INTO geo_cache (provider, key, country, city_ru, updated_at)
               VALUES ('online', 'c:1/2', 'RU', 'Москва', '2026-07-27T00:00:00+00:00')""")
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_reset_keeps_geo_cache_and_wipes_everything_else(self):
        db.reset_index(self.conn)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"], 0)
        self.assertEqual(geo_cache_size(self.conn), 1)
        # the surviving row is intact, not just its table
        row = self.conn.execute("SELECT city_ru FROM geo_cache").fetchone()
        self.assertEqual(row["city_ru"], "Москва")

    def test_reset_with_clear_geo_wipes_the_cache_too(self):
        db.reset_index(self.conn, clear_geo=True)
        self.assertEqual(geo_cache_size(self.conn), 0)
        # the table itself still exists (the schema is recreated), it is only empty
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"], 0)

    def test_repeated_reset_stays_green(self):
        db.reset_index(self.conn)
        db.reset_index(self.conn)
        self.assertEqual(geo_cache_size(self.conn), 1)

    def test_clear_geo_cache_reports_what_it_removed(self):
        self.assertEqual(clear_geo_cache(self.conn), 1)
        self.assertEqual(geo_cache_size(self.conn), 0)
        self.assertEqual(clear_geo_cache(self.conn), 0)


class TestResetEndpointFlag(ProcessTestBase):
    """The checkbox of the reset dialog has to reach `db.reset_index` — without a way
    out, a WRONG cached answer would survive every "Start over" the user tries."""

    def seed(self):
        fid, _p, _c = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.conn.execute(
            """INSERT INTO geo_cache (provider, key, country, city_ru, updated_at)
               VALUES ('online', 'c:1/2', 'RU', 'Москва', '2026-07-27T00:00:00+00:00')""")
        self.conn.commit()
        return fid

    def test_reset_without_the_flag_keeps_the_geo_cache(self):
        self.seed()
        self.start_server()
        status, payload = self.post("/api/process/reset", {})
        self.assertEqual(status, 200)
        self.assertFalse(payload["clear_geo"])
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"], 0)
        self.assertEqual(geo_cache_size(self.conn), 1)

    def test_reset_with_the_flag_wipes_the_geo_cache(self):
        self.seed()
        self.start_server()
        status, payload = self.post("/api/process/reset", {"clear_geo": True})
        self.assertEqual(status, 200)
        self.assertTrue(payload["clear_geo"])
        self.assertEqual(geo_cache_size(self.conn), 0)

    def test_garbage_flag_does_not_clear(self):
        # the destructive branch is only taken when it was asked for
        self.seed()
        self.start_server()
        status, payload = self.post("/api/process/reset", {"clear_geo": "nope"})
        self.assertEqual(status, 200)
        self.assertTrue(payload["clear_geo"])  # a non-empty string is a truthy request
        status2, _payload2 = self.post("/api/process/reset", {"clear_geo": None})
        self.assertEqual(status2, 200)
        self.assertEqual(geo_cache_size(self.conn), 0)


class TestResetDialogHtml(UiServerTestBase):
    def test_dialog_carries_an_unchecked_geo_checkbox(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="reset-dialog"', html)
        self.assertIn('id="reset-clear-geo-checkbox"', html)
        # unchecked by default in the markup AND reset on every open
        self.assertNotIn('id="reset-clear-geo-checkbox" checked', html)
        self.assertIn("resetClearGeoEl.checked = false;", html)
        # the flag travels to the endpoint
        self.assertIn('postJson("/api/process/reset", { clear_geo: clearGeo })', html)
        # the dialog replaced window.confirm — a confirm() cannot hold a checkbox
        self.assertNotIn("window.confirm(I18N.process_reset_confirm)", html)

    def test_the_checkbox_strings_exist_in_all_three_languages(self):
        for key in ("process_reset_clear_geo_label", "process_reset_clear_geo_hint",
                    "process_reset_confirm_ok", "process_reset_confirm_cancel",
                    "process_reset_done_geo"):
            entry = ui._UI_STRINGS[key]
            for lang in ("ru", "en", "ja"):
                self.assertIn(lang, entry, key)
                self.assertTrue(entry[lang].strip(), key)


class TestCliFlags(unittest.TestCase):
    """The flag reaches the reset/cache functions from the terminal as well."""

    def setUp(self):
        from typer.testing import CliRunner

        self.runner = CliRunner()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "src").mkdir()
        self.db_path = root / "test.db"
        self.cfg_path = root / "config.yaml"
        self.cfg_path.write_text(
            f'sources: ["{(root / "src").as_posix()}"]\n'
            f'database: "{self.db_path.as_posix()}"\n',
            encoding="utf-8")
        conn = connect(self.db_path)
        conn.execute(
            """INSERT INTO geo_cache (provider, key, country, city_ru, updated_at)
               VALUES ('online', 'c:1/2', 'RU', 'Москва', '2026-07-27T00:00:00+00:00')""")
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def _invoke(self, *args):
        from sorta import cli

        result = self.runner.invoke(cli.app, [*args, "--config", str(self.cfg_path)])
        self.assertEqual(result.exit_code, 0, result.output)
        return result

    def _cache_size(self) -> int:
        conn = connect(self.db_path)
        try:
            return geo_cache_size(conn)
        finally:
            conn.close()

    def test_cache_shows_the_geo_cache_size(self):
        result = self._invoke("cache")
        self.assertIn("geo_cache", result.output)
        self.assertIn("1", result.output)
        self.assertEqual(self._cache_size(), 1)

    def test_cache_clear_geo_empties_it(self):
        # F112: the CLI speaks the configured language, and the DEFAULT is `en` — this
        # fixture writes no `language:` key, so the message arrives in English. The
        # assertion is on the effect plus a language-independent number; the wording of
        # each language is covered by the i18n tests, not here.
        result = self._invoke("cache", "--clear-geo")
        self.assertIn("cache cleared", result.output.lower())
        self.assertEqual(self._cache_size(), 0)

    def test_reset_keeps_the_geo_cache_by_default(self):
        self._invoke("reset", "--yes")
        self.assertEqual(self._cache_size(), 1)

    def test_reset_clear_geo_wipes_it(self):
        self._invoke("reset", "--yes", "--clear-geo")
        self.assertEqual(self._cache_size(), 0)
