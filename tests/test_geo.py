"""The geo layer: exact_gps (GeoResolver mocked), session_inferred, idempotency."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sorta.config import Config, GeoConfig
from sorta.db import connect
from sorta.geo import resolve_places
from sorta.geodata import Resolution

# mini-fixture geonameids (numbers are arbitrary, just stable)
_GID_MOSCOW = 524901
_GID_PARIS = 2988507
_GID_SPB = 498817
_GID_AKADEM = 1487117  # a district near SPb — has a separate district_id

# known coordinates → the resolve result (GeoResolver is always mocked in tests, so
# as not to pull the bundled 12 MB of data)
_RESOLUTIONS = {
    (55.75, 37.62): Resolution(country_cc="RU", city_id=_GID_MOSCOW, district_id=None),
    (48.86, 2.35): Resolution(country_cc="FR", city_id=_GID_PARIS, district_id=None),
    (59.87, 30.36): Resolution(country_cc="RU", city_id=_GID_SPB, district_id=_GID_AKADEM),
}
_NAMES = {_GID_MOSCOW: "Moscow", _GID_PARIS: "Paris", _GID_SPB: "Saint Petersburg",
          _GID_AKADEM: "Akademicheskoe"}


class _FakeResolver:
    """A mini resolver instead of geodata.GeoResolver — without the real bundled data."""

    def resolve(self, lat, lon):
        return _RESOLUTIONS[(round(lat, 2), round(lon, 2))]

    def name(self, geonameid, lang):
        return _NAMES[geonameid]


class TestGeo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db")
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file(self, taken_at=None, confidence="high", lat=None, lon=None,
                 dup_of=None, error=None):
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, taken_at,
                   taken_at_source, taken_at_confidence, gps_lat, gps_lon,
                   dup_of, error, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', ?, 'exif', ?, ?, ?, ?, ?, '2026-01-01')""",
            (f"/photos/img_{self._n}.jpg", taken_at, confidence, lat, lon, dup_of, error),
        )
        self.conn.commit()
        return cur.lastrowid

    def place_of(self, file_id):
        return self.conn.execute(
            """SELECT country, region, city, city_geonameid, district_geonameid, confidence
               FROM places WHERE file_id = ?""", (file_id,)).fetchone()

    def run_geo(self):
        with patch("sorta.geo.GeoResolver", return_value=_FakeResolver()):
            return resolve_places(self.cfg, self.conn, progress=lambda done, total: None)

    def test_exact_gps(self):
        moscow = self.add_file("2023-05-01T10:00:00", lat=55.75, lon=37.62)
        paris = self.add_file("2023-06-01T10:00:00", lat=48.86, lon=2.35)
        stats = self.run_geo()
        self.assertEqual(stats.exact_gps, 2)
        row = self.place_of(moscow)
        self.assertEqual((row["country"], row["city"], row["city_geonameid"],
                           row["district_geonameid"], row["confidence"]),
                          ("RU", "Moscow", _GID_MOSCOW, None, "exact_gps"))
        row = self.place_of(paris)
        self.assertEqual((row["country"], row["city"], row["city_geonameid"],
                           row["district_geonameid"], row["confidence"]),
                          ("FR", "Paris", _GID_PARIS, None, "exact_gps"))

    def test_exact_gps_with_district(self):
        # district coordinates -> the city's city_geonameid + a separate district_geonameid
        akadem = self.add_file("2023-05-01T10:00:00", lat=59.87, lon=30.36)
        self.run_geo()
        row = self.place_of(akadem)
        self.assertEqual(row["city"], "Saint Petersburg")
        self.assertEqual(row["city_geonameid"], _GID_SPB)
        self.assertEqual(row["district_geonameid"], _GID_AKADEM)
        self.assertEqual(row["confidence"], "exact_gps")

    def test_region_never_written(self):
        # region — DEPRECATED (G2 no longer writes it), must stay NULL
        moscow = self.add_file("2023-05-01T10:00:00", lat=55.75, lon=37.62)
        self.run_geo()
        self.assertIsNone(self.place_of(moscow)["region"])

    def test_blank_gps_does_not_crash(self):
        # a real Phase-6 case: gps_lat/lon = '' in the index must not crash geo
        good = self.add_file("2023-05-01T10:00:00", lat=55.75, lon=37.62)
        bad = self.add_file("2023-05-01T11:00:00", lat="", lon="")
        stats = self.run_geo()
        self.assertEqual(stats.exact_gps, 1)
        self.assertEqual(self.place_of(good)["city"], "Moscow")
        # bad coordinates are ignored; the file inherits the place of a session neighbour
        self.assertEqual(self.place_of(bad)["confidence"], "session_inferred")

    def test_session_inheritance(self):
        # a session of 5 files (gaps ≤ 6 h), GPS on 2 → the other 3 inherit
        ids = [
            self.add_file("2023-05-01T10:00:00"),
            self.add_file("2023-05-01T12:00:00", lat=55.75, lon=37.62),
            self.add_file("2023-05-01T14:00:00"),
            self.add_file("2023-05-01T16:00:00", lat=55.75, lon=37.62),
            self.add_file("2023-05-01T18:00:00"),
        ]
        stats = self.run_geo()
        self.assertEqual(stats.exact_gps, 2)
        self.assertEqual(stats.session_inferred, 3)
        for fid in (ids[0], ids[2], ids[4]):
            row = self.place_of(fid)
            self.assertEqual((row["country"], row["city"], row["city_geonameid"],
                               row["district_geonameid"], row["confidence"]),
                              ("RU", "Moscow", _GID_MOSCOW, None, "session_inferred"))

    def test_session_inherits_district(self):
        # the inherited set MUST include the neighbour's district_geonameid, not just city
        neighbor = self.add_file("2023-05-01T10:00:00")
        self.add_file("2023-05-01T11:00:00", lat=59.87, lon=30.36)
        self.run_geo()
        row = self.place_of(neighbor)
        self.assertEqual(row["confidence"], "session_inferred")
        self.assertEqual(row["city_geonameid"], _GID_SPB)
        self.assertEqual(row["district_geonameid"], _GID_AKADEM)
        self.assertEqual(row["city"], "Saint Petersburg")

    def test_outside_gap_does_not_inherit(self):
        self.add_file("2023-05-01T10:00:00", lat=55.75, lon=37.62)
        outside = self.add_file("2023-05-01T17:00:00")  # a 7 h gap > 6
        self.run_geo()
        row = self.place_of(outside)
        self.assertEqual((row["city"], row["city_geonameid"], row["confidence"]),
                          (None, None, "unknown"))

    def test_low_confidence_does_not_inherit(self):
        self.add_file("2023-05-01T10:00:00", lat=55.75, lon=37.62)
        low = self.add_file("2023-05-01T11:00:00", confidence="low")
        self.run_geo()
        row = self.place_of(low)
        self.assertEqual((row["city"], row["confidence"]), (None, "unknown"))

    def test_nearest_gps_wins_with_multiple_cities(self):
        self.add_file("2023-05-01T10:00:00", lat=55.75, lon=37.62)   # Moscow
        near_moscow = self.add_file("2023-05-01T11:00:00")
        near_paris = self.add_file("2023-05-01T13:30:00")
        self.add_file("2023-05-01T14:00:00", lat=48.86, lon=2.35)    # Paris
        self.run_geo()
        self.assertEqual(self.place_of(near_moscow)["city"], "Moscow")
        self.assertEqual(self.place_of(near_paris)["city"], "Paris")

    def test_session_gap_from_config(self):
        self.cfg.geo = GeoConfig(session_gap_hours=1)
        self.add_file("2023-05-01T10:00:00", lat=55.75, lon=37.62)
        far = self.add_file("2023-05-01T12:00:00")  # 2 h > 1 h from config
        self.run_geo()
        self.assertEqual(self.place_of(far)["confidence"], "unknown")

    def test_skips_duplicates_and_errors(self):
        canon = self.add_file("2023-05-01T10:00:00", lat=55.75, lon=37.62)
        dup = self.add_file("2023-05-01T10:00:00", lat=55.75, lon=37.62, dup_of=canon)
        broken = self.add_file(error="boom")
        stats = self.run_geo()
        self.assertEqual(stats.total, 1)
        self.assertIsNone(self.place_of(dup))
        self.assertIsNone(self.place_of(broken))

    def test_no_taken_at_is_unknown(self):
        no_date = self.add_file(taken_at=None)
        self.run_geo()
        self.assertEqual(self.place_of(no_date)["confidence"], "unknown")

    def test_idempotent(self):
        self.add_file("2023-05-01T10:00:00", lat=55.75, lon=37.62)
        self.add_file("2023-05-01T11:00:00")
        self.add_file(taken_at=None)
        self.run_geo()
        first = self.conn.execute(
            """SELECT file_id, country, city, city_geonameid, district_geonameid, confidence
               FROM places ORDER BY file_id"""
        ).fetchall()
        self.run_geo()
        second = self.conn.execute(
            """SELECT file_id, country, city, city_geonameid, district_geonameid, confidence
               FROM places ORDER BY file_id"""
        ).fetchall()
        self.assertEqual([tuple(r) for r in first], [tuple(r) for r in second])
        self.assertEqual(len(first), 3)

    def test_unknown_provider_raises(self):
        self.cfg.geo = GeoConfig(provider="bogus")
        self.add_file("2023-05-01T10:00:00", lat=55.75, lon=37.62)
        with self.assertRaises(ValueError):
            resolve_places(self.cfg, self.conn, progress=lambda done, total: None)


class TestGeoProgress(unittest.TestCase):
    """F52 (#37): the first progress call carries the right total, even when the
    stage is small (< _PROGRESS_EVERY) — the "0 of 0" regression."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db")
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file(self, taken_at, lat, lon):
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, taken_at,
                   taken_at_source, taken_at_confidence, gps_lat, gps_lon, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', ?, 'exif', 'high', ?, ?, '2026-01-01')""",
            (f"/photos/img_{self._n}.jpg", taken_at, lat, lon),
        )
        self.conn.commit()
        return cur.lastrowid

    def test_first_call_has_full_total_on_small_stage(self):
        # deliberately smaller than _PROGRESS_EVERY (1000) — the first call used to
        # arrive only at i=1000, here the stage never reaches it.
        self.add_file("2023-05-01T10:00:00", 55.75, 37.62)
        self.add_file("2023-05-01T11:00:00", 48.86, 2.35)
        calls = []
        with patch("sorta.geo.GeoResolver", return_value=_FakeResolver()):
            resolve_places(self.cfg, self.conn,
                           progress=lambda done, total: calls.append((done, total)))
        self.assertTrue(calls)
        self.assertEqual(calls[0], (0, 2))
        self.assertEqual(calls[-1], (2, 2))

    def test_online_provider_progress_ticks_during_network_phase(self):
        # F52 review: the network phase (_NominatimResolver.resolve_places, called
        # BEFORE the write loop) must move progress itself — the write loop for
        # online is instant (the network already ran), so all the useful information
        # about the ~12-minute run comes from here.
        self.cfg.geo = GeoConfig(provider="online")
        self.add_file("2023-05-01T10:00:00", 55.75, 37.62)
        self.add_file("2023-05-01T11:00:00", 48.86, 2.35)
        calls = []
        payload = {"address": {"city": "Moscow", "country_code": "ru", "country": "Россия"}}
        with patch("sorta.geo.urllib.request.urlopen") as mock_urlopen, \
             patch("sorta.geo.time.sleep"):
            import json as _json

            class _Resp:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self):
                    return _json.dumps(payload).encode("utf-8")

            mock_urlopen.return_value = _Resp()
            resolve_places(self.cfg, self.conn,
                           progress=lambda done, total: calls.append((done, total)))
        # the order proves the counter moved DURING the network: (1, 2) and
        # (2, 2) from the resolve come FIRST, before the write loop's initial/final
        # (0, 2)/(2, 2) — not in one burst at the end.
        self.assertEqual(calls, [(1, 2), (2, 2), (0, 2), (2, 2)])


class TestNominatimResolverProgress(unittest.TestCase):
    """F52 review: unit-level on the network resolve itself — previously
    _NominatimResolver.resolve_places got no progress at all, and the counter hung
    at "0 of N" for all ~12 minutes of network, then instantly caught up in the write
    loop. We check that progress(k, len(coords)) is called for EACH coordinate, in
    step with (after) the real network requests, not all at once at the end.
    """

    def _resolver(self):
        from sorta.geo import _NominatimResolver
        return _NominatimResolver(Config(geo=GeoConfig(provider="online")))

    def _fake_response(self):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"address": {"city": "X", "country_code": "ru"}}'
        return _Resp()

    def test_progress_interleaved_with_requests_not_batched_at_end(self):
        resolver = self._resolver()
        coords = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]  # different -> none is cached
        events: list[tuple] = []

        def fake_urlopen(req, timeout=None):
            events.append(("request",))
            return self._fake_response()

        with patch("sorta.geo.urllib.request.urlopen", side_effect=fake_urlopen), \
             patch("sorta.geo.time.sleep"):
            resolver.resolve_places(
                coords, progress=lambda done, total: events.append(("progress", done, total)))

        self.assertEqual(
            events,
            [("request",), ("progress", 1, 3),
             ("request",), ("progress", 2, 3),
             ("request",), ("progress", 3, 3)],
        )

    def test_progress_called_for_cached_coords_too(self):
        # a repeated (rounded) coordinate makes no new request, but the position in
        # coords still advances progress — otherwise cache hits would stick at the
        # previous total.
        resolver = self._resolver()
        coords = [(1.0, 1.0), (1.0, 1.0), (2.0, 2.0)]
        calls = []
        with patch("sorta.geo.urllib.request.urlopen",
                   return_value=self._fake_response()) as mock_urlopen, \
             patch("sorta.geo.time.sleep"):
            resolver.resolve_places(
                coords, progress=lambda done, total: calls.append((done, total)))
        self.assertEqual(calls, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(mock_urlopen.call_count, 2)  # 2 unique coordinates


if __name__ == "__main__":
    unittest.main()
