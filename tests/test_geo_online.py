"""G2b: the online Nominatim/OSM provider — without real network (urllib mocked)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sorta.config import Config, GeoConfig
from sorta.db import connect
from sorta.geo import resolve_places


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


_MOSCOW_ADDRESS = {
    "address": {
        "suburb": "Zamoskvorechye",
        "city": "Moscow",
        "country": "Россия",       # full name in the accept-language language (G6)
        "country_code": "ru",
    }
}


class TestGeoOnline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db",
                          geo=GeoConfig(provider="online"))
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file(self, taken_at="2023-05-01T10:00:00", confidence="high", lat=None, lon=None):
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, taken_at,
                   taken_at_source, taken_at_confidence, gps_lat, gps_lon, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', ?, 'exif', ?, ?, ?, '2026-01-01')""",
            (f"/photos/img_{self._n}.jpg", taken_at, confidence, lat, lon),
        )
        self.conn.commit()
        return cur.lastrowid

    def place_of(self, file_id):
        return self.conn.execute(
            """SELECT country, country_name, city, city_geonameid, district_geonameid,
                      district_name, confidence
               FROM places WHERE file_id = ?""", (file_id,)).fetchone()

    def test_online_resolves_names_without_geonameid(self):
        photo = self.add_file(lat=55.75, lon=37.62)
        with patch("sorta.geo.urllib.request.urlopen",
                   return_value=_FakeResponse(_MOSCOW_ADDRESS)):
            resolve_places(self.cfg, self.conn, progress=lambda done, total: None)
        row = self.place_of(photo)
        self.assertEqual(row["country"], "RU")
        self.assertEqual(row["country_name"], "Россия")  # G6: the full name
        self.assertEqual(row["city"], "Moscow")
        self.assertIsNone(row["city_geonameid"])
        self.assertIsNone(row["district_geonameid"])
        self.assertEqual(row["district_name"], "Zamoskvorechye")
        self.assertEqual(row["confidence"], "exact_gps")

    def test_online_request_uses_config(self):
        self.cfg.geo = GeoConfig(provider="online", nominatim_url="https://geo.example/api",
                                 nominatim_user_agent="my-agent/1.0", nominatim_timeout=5.0)
        self.cfg.language = "ja"
        self.add_file(lat=55.75, lon=37.62)
        mock_urlopen = MagicMock(return_value=_FakeResponse(_MOSCOW_ADDRESS))
        with patch("sorta.geo.urllib.request.urlopen", mock_urlopen):
            resolve_places(self.cfg, self.conn, progress=lambda done, total: None)
        req = mock_urlopen.call_args[0][0]
        self.assertTrue(req.full_url.startswith("https://geo.example/api/reverse?"))
        self.assertIn("accept-language=ja", req.full_url)
        self.assertEqual(req.get_header("User-agent"), "my-agent/1.0")
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 5.0)

    def test_online_caches_same_rounded_coords(self):
        self.add_file(lat=55.75001, lon=37.62001)
        self.add_file(lat=55.75002, lon=37.62002)  # rounds to the same 4 digits
        mock_urlopen = MagicMock(return_value=_FakeResponse(_MOSCOW_ADDRESS))
        with patch("sorta.geo.urllib.request.urlopen", mock_urlopen), \
             patch("sorta.geo.time.sleep") as mock_sleep:
            resolve_places(self.cfg, self.conn, progress=lambda done, total: None)
        self.assertEqual(mock_urlopen.call_count, 1)
        mock_sleep.assert_not_called()  # a single request -> no rate-limit needed

    def test_online_rate_limits_between_distinct_coords(self):
        self.add_file(lat=55.75, lon=37.62)
        self.add_file(lat=48.86, lon=2.35)
        with patch("sorta.geo.urllib.request.urlopen",
                   return_value=_FakeResponse(_MOSCOW_ADDRESS)), \
             patch("sorta.geo.time.sleep") as mock_sleep:
            resolve_places(self.cfg, self.conn, progress=lambda done, total: None)
        mock_sleep.assert_called_once()

    def test_online_network_error_is_unknown(self):
        # GPS was present -> confidence stays exact_gps (like offline with empty data),
        # but the place itself (city/district) does not resolve and geo does not crash
        photo = self.add_file(lat=55.75, lon=37.62)
        with patch("sorta.geo.urllib.request.urlopen", side_effect=OSError("boom")):
            resolve_places(self.cfg, self.conn, progress=lambda done, total: None)
        row = self.place_of(photo)
        self.assertEqual(row["confidence"], "exact_gps")
        self.assertIsNone(row["city"])
        self.assertIsNone(row["country"])
        self.assertIsNone(row["district_name"])

    def test_online_empty_address_is_unknown(self):
        photo = self.add_file(lat=55.75, lon=37.62)
        with patch("sorta.geo.urllib.request.urlopen", return_value=_FakeResponse({})):
            resolve_places(self.cfg, self.conn, progress=lambda done, total: None)
        row = self.place_of(photo)
        self.assertEqual(row["confidence"], "exact_gps")
        self.assertIsNone(row["city"])
        self.assertIsNone(row["district_name"])

    def test_session_inferred_inherits_district_name(self):
        self.add_file("2023-05-01T10:00:00", lat=55.75, lon=37.62)
        neighbor = self.add_file("2023-05-01T11:00:00")
        with patch("sorta.geo.urllib.request.urlopen",
                   return_value=_FakeResponse(_MOSCOW_ADDRESS)):
            resolve_places(self.cfg, self.conn, progress=lambda done, total: None)
        row = self.place_of(neighbor)
        self.assertEqual(row["confidence"], "session_inferred")
        self.assertEqual(row["city"], "Moscow")
        self.assertEqual(row["district_name"], "Zamoskvorechye")
        self.assertEqual(row["country_name"], "Россия")  # G6: fully inherited


if __name__ == "__main__":
    unittest.main()
