"""F92: trip merging by distance uses the coordinates of the FILES, not geonameid.

The proximity branch of _same_trip used to ask geodata for the coordinates of
city_geonameid, so with an online provider (Nominatim returns no geonameid) it never
ran at all and neighbouring villages stayed separate trips. The center of a locality
is now the median GPS of its own files; coords_of remains the fallback for a locality
whose files carry no GPS.
"""
import unittest
from pathlib import Path
from unittest.mock import patch

from sorta.events import build_events
from sorta.geodata import GeoResolver
from tests.test_events import CITY_A, CITY_B, EventsBase, _write_geo_fixture

# Three neighbouring villages of one island: a few kilometres apart, well inside the
# default trip_merge_max_km (120).
VILLAGE_A = (-8.50, 115.20)
VILLAGE_B = (-8.55, 115.30)
VILLAGE_C = (-8.65, 115.22)
# The same island group, but ~720 km north — beyond the threshold.
FAR_ISLAND = (-2.00, 115.20)
# ~1100 km west of VILLAGE_A: an airport stopover frame, the outlier the median must
# survive (its mean with five village frames lands ~185 km away, past the threshold).
AIRPORT = (-8.50, 105.20)

# The fixture geodata coordinates of CITY_A/CITY_B (see tests.test_events) — used
# where a test needs the files' own GPS to agree with what coords_of would answer.
CITY_A_GPS = (0.0, 100.0)
CITY_B_GPS = (0.5, 100.0)


class TripMergeCoordsBase(EventsBase):
    """EventsBase + the fixture GeoResolver (admin1/countries for CITY_A..D)."""

    def setUp(self):
        super().setUp()
        geo_dir = Path(self.tmp.name) / "geo_fixture"
        _write_geo_fixture(geo_dir)
        patcher = patch("sorta.events.GeoResolver",
                        lambda *a, **k: GeoResolver(data_dir=geo_dir))
        patcher.start()
        self.addCleanup(patcher.stop)

    def add_session(self, day, hour, name, gps, count=1, country="ID"):
        """`count` online-like files (no geonameid, a string city) at one place."""
        return [
            self.add_file(f"2023-05-{day:02d}T{hour:02d}:{i:02d}:00",
                          district_name=name, country=country, gps=gps)
            for i in range(count)
        ]


class TestOnlineLikeMergesByFileCoords(TripMergeCoordsBase):
    """city_geonameid IS NULL, different city strings, coordinates next to each other."""

    def test_three_villages_form_one_trip(self):
        a = self.add_session(1, 18, "Ubud", VILLAGE_A, count=3)
        b = self.add_session(2, 9, "Tegallalang", VILLAGE_B, count=2)
        c = self.add_session(3, 0, "Payangan", VILLAGE_C, count=2)
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)
        ev = self.events()[0]
        self.assertEqual(self.files_of(ev["id"]), set(a) | set(b) | set(c))
        # several localities in the group and no geonameid to take a region from →
        # the country (F44/#19-B), unchanged by F92
        self.assertEqual(ev["place_city"], "Indonesia")

    def test_no_merge_beyond_max_km(self):
        # the same shape, but the second session is ~720 km away
        self.add_session(1, 18, "Ubud", VILLAGE_A, count=3)
        self.add_session(2, 9, "Faraway", FAR_ISLAND, count=2)
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)

    def test_threshold_is_respected_not_widened(self):
        # a threshold below the real distance switches the same data back to two trips
        self.cfg.events.trip_merge_max_km = 5  # < ~11 km between A and B
        self.add_session(1, 18, "Ubud", VILLAGE_A, count=3)
        self.add_session(2, 9, "Tegallalang", VILLAGE_B, count=2)
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)

    def test_zero_threshold_disables_distance(self):
        self.cfg.events.trip_merge_max_km = 0
        self.add_session(1, 18, "Ubud", VILLAGE_A, count=3)
        self.add_session(2, 9, "Tegallalang", VILLAGE_B, count=2)
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)

    def test_different_countries_do_not_merge(self):
        # a border: 11 km apart, but the country decides first
        self.add_session(1, 18, "Ubud", VILLAGE_A, count=3, country="ID")
        self.add_session(2, 9, "Border town", VILLAGE_B, count=2, country="TH")
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)

    def test_same_string_still_merges_without_gps(self):
        # F44/#19-A1 regression: string equality does not depend on coordinates
        self.add_file("2023-05-01T18:00:00", district_name="Ubud", country="ID")
        self.add_file("2023-05-02T09:00:00", district_name="UBUD", country="ID")
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)


class TestMedianNotMean(TripMergeCoordsBase):
    def test_single_far_outlier_does_not_move_the_center(self):
        # five frames in the village + one from an airport 1100 km away; the mean of
        # the six lands ~185 km from the village (past the 120 km threshold), the
        # median stays in the village
        a = self.add_session(1, 18, "Ubud", VILLAGE_A, count=5)
        a += self.add_session(1, 19, "Ubud", AIRPORT)
        b = self.add_session(2, 9, "Tegallalang", VILLAGE_B, count=2)
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)
        self.assertEqual(self.files_of(self.events()[0]["id"]), set(a) | set(b))

    def test_majority_of_frames_decides_the_center(self):
        # the mirror case: most of the session really is at the far place, so the
        # center moves there and the neighbouring village no longer merges
        self.add_session(1, 18, "Ubud", FAR_ISLAND, count=5)
        self.add_session(1, 19, "Ubud", VILLAGE_A)
        self.add_session(2, 9, "Tegallalang", VILLAGE_B, count=2)
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)

    def test_center_ignores_files_of_another_locality(self):
        # a session where the dominant locality is the village and two frames belong
        # to a far one: the center is the village's own median, so the merge holds
        a = self.add_session(1, 18, "Ubud", VILLAGE_A, count=3)
        a += self.add_session(1, 19, "Faraway", FAR_ISLAND, count=2)
        b = self.add_session(2, 9, "Tegallalang", VILLAGE_B, count=2)
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)
        self.assertEqual(self.files_of(self.events()[0]["id"]), set(a) | set(b))


class TestOfflineUnchanged(TripMergeCoordsBase):
    """geonameid data must behave exactly as before F92."""

    def _snapshot(self):
        return [(tuple(e)[1:], sorted(self.files_of(e["id"]))) for e in self.events()]

    def test_geonameid_result_identical_with_and_without_gps(self):
        # CITY_A/CITY_B: ~56 km apart in the fixture geodata, different admin1 →
        # they merge through the distance branch, whichever source the coordinates
        # come from
        for i in range(3):
            self.add_file(f"2023-05-01T18:0{i}:00", city_id=CITY_A, country="ID",
                          gps=CITY_A_GPS)
        self.add_file("2023-05-02T09:00:00", city_id=CITY_B, country="ID", gps=CITY_B_GPS)
        build_events(self.cfg, self.conn)
        with_gps = self._snapshot()
        self.assertEqual(len(with_gps), 1)
        self.assertEqual(self.events()[0]["place_city"], "Bali")

        self.conn.execute("UPDATE files SET gps_lat = NULL, gps_lon = NULL")
        self.conn.commit()
        build_events(self.cfg, self.conn)
        self.assertEqual(self._snapshot(), with_gps)

    def test_gps_center_can_overrule_a_stale_geobase_position(self):
        # the files say the two localities are far apart; their own coordinates are
        # what the merge trusts, not the fixture positions of CITY_A/CITY_B
        self.add_file("2023-05-01T18:00:00", city_id=CITY_A, country="ID", gps=VILLAGE_A)
        self.add_file("2023-05-02T09:00:00", city_id=CITY_B, country="ID", gps=FAR_ISLAND)
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)


class TestFallbackToGeodata(TripMergeCoordsBase):
    def test_locality_without_any_gps_falls_back_to_coords_of(self):
        # the city came in by session inheritance — no file of it has GPS; the
        # geodata coordinates of city_geonameid keep the merge working
        a = [self.add_file(f"2023-05-01T18:0{i}:00", city_id=CITY_A, country="ID")
             for i in range(3)]
        b = [self.add_file("2023-05-02T09:00:00", city_id=CITY_B, country="ID")]
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)
        self.assertEqual(self.files_of(self.events()[0]["id"]), set(a) | set(b))

    def test_one_side_gps_other_side_geodata(self):
        # a mixed pair: the anchor has file GPS, the candidate only a geonameid
        a = [self.add_file(f"2023-05-01T18:0{i}:00", city_id=CITY_A, country="ID",
                           gps=CITY_A_GPS) for i in range(3)]
        b = [self.add_file("2023-05-02T09:00:00", city_id=CITY_B, country="ID")]
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)
        self.assertEqual(self.files_of(self.events()[0]["id"]), set(a) | set(b))

    def test_no_gps_and_no_geonameid_does_not_merge(self):
        # online strings without coordinates: nothing to measure — as before F92
        self.add_file("2023-05-01T18:00:00", district_name="Ubud", country="ID")
        self.add_file("2023-05-02T09:00:00", district_name="Tegallalang", country="ID")
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)

    def test_unknown_locality_does_not_merge_on_coordinates_alone(self):
        # country only, no city at all on either side: an unknown locality confirms
        # nothing, even when the frames were shot next to each other
        self.add_file("2023-05-01T18:00:00", country="ID", gps=VILLAGE_A)
        self.add_file("2023-05-02T09:00:00", country="ID", gps=VILLAGE_B)
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)


class TestUnusableCoordinates(TripMergeCoordsBase):
    def test_null_island_is_not_a_position(self):
        # (0, 0) — the "never got a fix" sentinel; treated as no GPS, so the merge
        # falls back to geodata instead of pretending the files are off Ghana
        a = [self.add_file(f"2023-05-01T18:0{i}:00", city_id=CITY_A, country="ID",
                           gps=(0.0, 0.0)) for i in range(3)]
        b = [self.add_file("2023-05-02T09:00:00", city_id=CITY_B, country="ID",
                           gps=(0.0, 0.0))]
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)
        self.assertEqual(self.files_of(self.events()[0]["id"]), set(a) | set(b))

    def test_garbage_coordinates_are_ignored(self):
        # broken EXIF writes '' into the index — it must not crash the stage
        self.add_file("2023-05-01T18:00:00", district_name="Ubud", country="ID",
                      gps=("", ""))
        self.add_file("2023-05-02T09:00:00", district_name="Ubud", country="ID",
                      gps=VILLAGE_A)
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)


if __name__ == "__main__":
    unittest.main()
