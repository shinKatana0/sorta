"""F86: an online answer without a city is completed from the bundled offline base.

A run with geo.provider=online (2026-07-26) left 1 471 exact_gps files with a country
and city NULL where the bundled GeoNames data does know a city. The coordinates below
are synthetic fixtures. Both the network and the offline resolver are faked here: the
network so the tests stay offline, the resolver so they do not pull the bundled 12 MB.
"""
import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import patch

from sorta.config import Config, GeoConfig
from sorta.db import connect
from sorta.geo import resolve_places
from sorta.geodata import Resolution

# fixture geonameids (arbitrary, just stable)
_GID_DOMODEDOVO = 567578
_GID_YAKOVLEVSKOE = 468902
_GID_JABAJERO = 1642911
_GID_NUSA_DUA = 8299693
_GID_KANGAR = 1732715  # MY — the country-mismatch guard

_OFFLINE_NAMES = {
    _GID_DOMODEDOVO: "Domodedovo",
    _GID_YAKOVLEVSKOE: "Yakovlevskoye",
    _GID_JABAJERO: "Jabajero",
    _GID_NUSA_DUA: "Nusa Dua",
    _GID_KANGAR: "Kangar",
}

# synthetic fixture coordinates, one per branch of the fallback
_DOMODEDOVO = (55.41, 37.90)
_NUSA_DUA = (-8.80, 115.23)
_BORDER = (6.5417, 100.12)      # online says TH, the offline base answers MY
_OPEN_SEA = (13.5, 92.5)        # no city in the offline base either

_OFFLINE_RESOLUTIONS = {
    _DOMODEDOVO: Resolution(country_cc="RU", city_id=_GID_DOMODEDOVO,
                            district_id=_GID_YAKOVLEVSKOE),
    _NUSA_DUA: Resolution(country_cc="ID", city_id=_GID_JABAJERO,
                          district_id=_GID_NUSA_DUA),
    _BORDER: Resolution(country_cc="MY", city_id=_GID_KANGAR, district_id=None),
    _OPEN_SEA: Resolution(country_cc="TH", city_id=None, district_id=None),
}

# Nominatim answers that name the country but no settlement — the defect itself.
# `county`/`state` are present on purpose: they are NOT read as a city (an oblast is
# not a city, and a folder named after one is worse than the country level).
_ONLINE_ANSWERS = {
    _DOMODEDOVO: {"address": {"county": "городской округ Домодедово",
                              "state": "Московская область",
                              "country": "Россия", "country_code": "ru"}},
    _NUSA_DUA: {"address": {"suburb": "Nusa Dua", "state": "Bali",
                            "country": "Индонезия", "country_code": "id"}},
    _BORDER: {"address": {"state": "Songkhla", "country": "Таиланд",
                          "country_code": "th"}},
    _OPEN_SEA: {"address": {"state": "Andaman", "country": "Таиланд",
                            "country_code": "th"}},
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


class _FakeOffline:
    """geodata.GeoResolver without the bundled data — and a record of what it was asked."""

    data_dir = Path("/bundled/geo")

    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.calls: list[tuple[float, float]] = []

    def data_available(self) -> bool:
        return self._available

    def resolve(self, lat: float, lon: float) -> Resolution:
        self.calls.append((lat, lon))
        return _OFFLINE_RESOLUTIONS[(round(lat, 5), round(lon, 5))]

    def name(self, geonameid: int, lang: str) -> str:
        return _OFFLINE_NAMES[geonameid]


def _online_by_coords(req: object, timeout: float | None = None) -> _FakeResponse:
    """The faked reverse endpoint: the answer is picked by the lat/lon of the request."""
    query = urllib.parse.urlparse(req.full_url).query  # type: ignore[attr-defined]
    params = urllib.parse.parse_qs(query)
    key = (float(params["lat"][0]), float(params["lon"][0]))
    return _FakeResponse(_ONLINE_ANSWERS[key])


class CityFallbackTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          geo=GeoConfig(provider="online"))
        self.conn = connect(self.cfg.database)
        self.offline = _FakeOffline()
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file(self, coords=None, taken_at="2023-03-03T14:28:32"):
        self._n += 1
        lat, lon = coords if coords is not None else (None, None)
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, taken_at,
                   taken_at_source, taken_at_confidence, gps_lat, gps_lon, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', ?, 'exif', 'high', ?, ?, '2026-01-01')""",
            (f"/photos/img_{self._n}.jpg", taken_at, lat, lon),
        )
        self.conn.commit()
        return cur.lastrowid

    def place_of(self, file_id):
        return self.conn.execute(
            """SELECT country, country_name, city, city_geonameid, district_geonameid,
                      district_name, confidence
               FROM places WHERE file_id = ?""", (file_id,)).fetchone()

    def run_geo(self, payload=None):
        opener = _online_by_coords if payload is None else (
            lambda req, timeout=None: _FakeResponse(payload))
        with patch("sorta.geo.urllib.request.urlopen", side_effect=opener), \
             patch("sorta.geo.GeoResolver", return_value=self.offline), \
             patch("sorta.geo.time.sleep"):
            return resolve_places(self.cfg, self.conn, progress=lambda done, total: None)


class TestOfflineFallback(CityFallbackTestBase):
    def test_country_without_city_takes_the_city_from_the_offline_base(self):
        # online returned only the country; the offline base knows both the city and
        # the district it sits in
        photo = self.add_file(_DOMODEDOVO)
        stats = self.run_geo()
        row = self.place_of(photo)
        self.assertEqual(row["city"], "Domodedovo")
        self.assertEqual(row["city_geonameid"], _GID_DOMODEDOVO)
        self.assertEqual(row["district_geonameid"], _GID_YAKOVLEVSKOE)
        self.assertEqual(row["country"], "RU")
        self.assertEqual(row["confidence"], "exact_gps")
        self.assertEqual(stats.exact_gps, 1)

    def test_the_second_coordinate_too(self):
        photo = self.add_file(_NUSA_DUA)
        self.run_geo()
        row = self.place_of(photo)
        self.assertEqual(row["city"], "Jabajero")
        self.assertEqual(row["country"], "ID")
        self.assertEqual(row["district_geonameid"], _GID_NUSA_DUA)

    def test_place_comes_from_one_source_only(self):
        # the whole place is replaced, not merged: the Nominatim country name and its
        # `suburb` are dropped together with the answer they came in, so a GeoNames city
        # never ends up under a country that came from the other provider.
        photo = self.add_file(_NUSA_DUA)
        self.run_geo()
        row = self.place_of(photo)
        self.assertIsNone(row["country_name"])     # was "Индонезия" in the online answer
        self.assertIsNone(row["district_name"])    # was "Nusa Dua" as text
        self.assertEqual(row["district_geonameid"], _GID_NUSA_DUA)

    def test_online_city_is_not_replaced_by_the_offline_base(self):
        # the regression guard: an answer that already names a city must not be
        # completed from the bundled base — the online city wins.
        # F93: the base IS consulted for every coordinate now (that is where the cache
        # key comes from, a free KD-tree lookup), so this is asserted on the RESULT: an
        # offline place would have brought a city_geonameid with it.
        photo = self.add_file(_DOMODEDOVO)
        self.run_geo(payload={"address": {"city": "Moscow", "country_code": "ru",
                                          "country": "Россия"}})
        row = self.place_of(photo)
        self.assertEqual(row["city"], "Moscow")
        self.assertEqual(row["country_name"], "Россия")
        self.assertIsNone(row["city_geonameid"])
        self.assertIsNone(row["district_geonameid"])

    def test_extra_address_keys_count_as_a_city(self):
        # F86 (2): Nominatim names a settlement outside a city with whatever key fits
        for key in ("hamlet", "locality", "isolated_dwelling"):
            with self.subTest(key=key):
                self.offline = _FakeOffline()
                photo = self.add_file(_DOMODEDOVO)
                self.run_geo(payload={"address": {key: "Ostrovtsy",
                                                  "country_code": "ru"}})
                self.assertEqual(self.place_of(photo)["city"], "Ostrovtsy")
                # the offline base did not replace the place (F93: it is still asked
                # for the cache key, but its city must not win here)
                self.assertIsNone(self.place_of(photo)["city_geonameid"])

    def test_country_mismatch_keeps_the_online_answer(self):
        # near a border the nearest-neighbour city of the offline base can sit in the
        # OTHER country (online: TH, offline: MY/Kangar). Following it would move the
        # file to a country the user has never been to — the F75 silent misplacement.
        photo = self.add_file(_BORDER)
        self.run_geo()
        row = self.place_of(photo)
        self.assertIsNone(row["city"])
        self.assertEqual(row["country"], "TH")
        self.assertEqual(row["country_name"], "Таиланд")
        self.assertEqual(row["confidence"], "exact_gps")

    def test_no_city_in_either_source_keeps_the_country(self):
        # nobody knows a settlement here — the city stays NULL (no guessing), the
        # country survives, and the sorter lays the file out at country level
        # (see tests/test_sorter.py::TestCountryWithoutCity).
        photo = self.add_file(_OPEN_SEA)
        stats = self.run_geo()
        row = self.place_of(photo)
        self.assertIsNone(row["city"])
        self.assertIsNone(row["city_geonameid"])
        self.assertEqual(row["country"], "TH")
        self.assertEqual((stats.exact_gps, stats.gps_unresolved), (1, 0))

    def test_session_inheritance_gets_the_completed_place(self):
        # a GPS-less neighbour in the same session must inherit the city recovered
        # offline, not the bare country the online answer carried
        self.add_file(_DOMODEDOVO, taken_at="2023-03-03T14:28:32")
        neighbor = self.add_file(taken_at="2023-03-03T15:00:00")
        stats = self.run_geo()
        row = self.place_of(neighbor)
        self.assertEqual(row["confidence"], "session_inferred")
        self.assertEqual(row["city"], "Domodedovo")
        self.assertEqual(row["city_geonameid"], _GID_DOMODEDOVO)
        self.assertEqual(stats.session_inferred, 1)

    def test_failed_request_does_not_switch_provider(self):
        # F65 keeps its meaning: there is no answer to complete here, so the file stays
        # unknown instead of silently resolving through the other provider.
        photo = self.add_file(_DOMODEDOVO)
        with patch("sorta.geo.urllib.request.urlopen", side_effect=OSError("boom")), \
             patch("sorta.geo.GeoResolver", return_value=self.offline):
            resolve_places(self.cfg, self.conn, progress=lambda done, total: None)
        row = self.place_of(photo)
        self.assertEqual(row["confidence"], "unknown")
        # nothing of the offline base leaked into the row (F93: it is consulted for the
        # cache key, which is not an answer about the place)
        self.assertIsNone(row["country"])
        self.assertIsNone(row["city"])
        self.assertIsNone(row["city_geonameid"])

    def test_missing_offline_data_does_not_break_the_online_run(self):
        # online is usable on an install without the bundled data — a missing base only
        # means the city cannot be recovered, it must not raise mid-run.
        self.offline = _FakeOffline(available=False)
        photo = self.add_file(_DOMODEDOVO)
        with self.assertLogs("sorta.geo", level="WARNING"):
            self.run_geo()
        row = self.place_of(photo)
        self.assertIsNone(row["city"])
        self.assertEqual(row["country"], "RU")
        self.assertEqual(self.offline.calls, [])

    def test_progress_still_ticks_through_the_wrapper(self):
        # the fallback wraps the network resolver, so the per-coordinate progress of the
        # online phase (F52) must survive the wrapping
        self.add_file(_DOMODEDOVO)
        self.add_file(_NUSA_DUA)
        calls: list[tuple[int, int]] = []
        with patch("sorta.geo.urllib.request.urlopen", side_effect=_online_by_coords), \
             patch("sorta.geo.GeoResolver", return_value=self.offline), \
             patch("sorta.geo.time.sleep"):
            resolve_places(self.cfg, self.conn,
                           progress=lambda done, total: calls.append((done, total)))
        self.assertEqual(calls[:2], [(1, 2), (2, 2)])


class TestMissingCityWarning(CityFallbackTestBase):
    """F86 (3): the silence is the reason the defect survived a full production run —
    but a line per file is thousands of lines, so it is written once per N cases."""

    def test_warns_but_not_once_per_file(self):
        for _ in range(5):
            self.add_file(_DOMODEDOVO)
        with self.assertLogs("sorta.geo", level="WARNING") as cm:
            self.run_geo()
        # 5 files without a city -> the first case + the final total, not 5 lines
        self.assertEqual(len(cm.records), 2)
        self.assertIn("without a city", cm.records[0].getMessage())
        self.assertIn("5", cm.records[-1].getMessage())

    def test_the_total_reports_how_many_cities_were_recovered(self):
        self.add_file(_DOMODEDOVO)   # recovered offline
        self.add_file(_OPEN_SEA)     # unknown to both
        with self.assertLogs("sorta.geo", level="WARNING") as cm:
            self.run_geo()
        total = cm.records[-1].getMessage()
        self.assertIn("without a city: 2", total)
        self.assertIn("of which 1 found a city in the offline base", total)

    def test_every_nth_case_is_reported(self):
        for _ in range(4):
            self.add_file(_DOMODEDOVO)
        with patch("sorta.geo._CITY_MISSING_WARN_EVERY", 2), \
             self.assertLogs("sorta.geo", level="WARNING") as cm:
            self.run_geo()
        # cases 1, 2 and 4 (the 1st, then every 2nd) + the final total
        self.assertEqual(len(cm.records), 4)

    def test_no_warning_when_every_answer_has_a_city(self):
        self.add_file(_DOMODEDOVO)
        with self.assertNoLogs("sorta.geo", level="WARNING"):
            self.run_geo(payload={"address": {"city": "Moscow", "country_code": "ru"}})


if __name__ == "__main__":
    unittest.main()
