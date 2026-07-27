"""F85a: the trip level of place inheritance (confidence='trip_inferred').

The six-hour session is too narrow a unit: on the live collection 1 758 files sat in a
session where nobody had GPS while the trip around them was placed perfectly well. The
tests below pin the properties that make the wider rule safe to ship — a trip is cut by
the SAME thresholds `events` uses (a copy of the rule lives in geo.py, see the block
comment there), the session level keeps its priority, a trip whose GPS frames disagree
about the city lends nothing at all, and a file inherits only from BETWEEN two frames of
that city, never past the last one.

That last rule is not a guess: measured on the validation collection
(scripts/measure_place_inference.py), inheriting anywhere inside the trip gives 94.2%
precision — under the 95% this feature must clear — and 28 of its 32 mistakes are files
past an end of the span. Inside the span precision is 99.0%. TestTripBracketing below is
what keeps that.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sorta.config import Config, EventsConfig, GeoConfig
from sorta.db import connect
from sorta.geo import resolve_places
from sorta.geodata import Resolution

# mini-fixture geonameids (arbitrary but stable)
_GID_MOSCOW = 524901
_GID_ZELENOGRAD = 566199  # ~38 km from Moscow — inside the default trip_merge_max_km
_GID_SPB = 498817         # ~635 km from Moscow — outside it
_GID_AKADEM = 1487117     # a district of SPb: has a district_id of its own
_GID_PARIS = 2988507

_MOSCOW = (55.75, 37.62)
_ZELENOGRAD = (55.99, 37.18)
_SPB = (59.87, 30.36)
_PARIS = (48.86, 2.35)

_RESOLUTIONS = {
    _MOSCOW: Resolution(country_cc="RU", city_id=_GID_MOSCOW, district_id=None),
    _ZELENOGRAD: Resolution(country_cc="RU", city_id=_GID_ZELENOGRAD, district_id=None),
    _SPB: Resolution(country_cc="RU", city_id=_GID_SPB, district_id=_GID_AKADEM),
    _PARIS: Resolution(country_cc="FR", city_id=_GID_PARIS, district_id=None),
}
_NAMES = {_GID_MOSCOW: "Moscow", _GID_ZELENOGRAD: "Zelenograd",
          _GID_SPB: "Saint Petersburg", _GID_AKADEM: "Akademicheskoe",
          _GID_PARIS: "Paris"}


class _FakeResolver:
    """The mini resolver of test_geo.py — no 12 MB of bundled data in the tests."""

    data_dir = Path("/bundled/geo")

    def resolve(self, lat, lon):
        return _RESOLUTIONS[(round(lat, 2), round(lon, 2))]

    def name(self, geonameid, lang):
        return _NAMES[geonameid]


class _TripBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db")
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file(self, taken_at=None, at=None, confidence="high"):
        """A canonical file; `at` — the (lat, lon) pair of a fixture city, or None."""
        self._n += 1
        lat, lon = at if at else (None, None)
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
            """SELECT country, city, city_geonameid, district_geonameid, district_name,
                      confidence
               FROM places WHERE file_id = ?""", (file_id,)).fetchone()

    def run_geo(self):
        with patch("sorta.geo.GeoResolver", return_value=_FakeResolver()):
            return resolve_places(self.cfg, self.conn, progress=lambda done, total: None)


class TestTripInheritance(_TripBase):
    def test_place_less_session_inherits_the_trip(self):
        # a day apart: three sessions (gaps of 24 h > 6), one trip (24 h < 48). The
        # middle session knows nothing — it is exactly what the feature exists for
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        orphan = self.add_file("2023-05-02T10:00:00")
        self.add_file("2023-05-03T10:00:00", at=_MOSCOW)
        stats = self.run_geo()
        row = self.place_of(orphan)
        self.assertEqual((row["country"], row["city"], row["city_geonameid"],
                          row["confidence"]),
                         ("RU", "Moscow", _GID_MOSCOW, "trip_inferred"))
        self.assertEqual((stats.exact_gps, stats.session_inferred, stats.trip_inferred,
                          stats.unknown), (2, 0, 1, 0))

    def test_beyond_the_trip_gap_nothing_is_inherited(self):
        # 72 h > events.trip_merge_gap_hours: three trips, not one wide session. The
        # file has Moscow on both sides of it and still inherits nothing — the trip is
        # the unit, and it ended
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        far = self.add_file("2023-05-04T10:00:00")
        self.add_file("2023-05-07T10:00:00", at=_MOSCOW)
        stats = self.run_geo()
        self.assertEqual(self.place_of(far)["confidence"], "unknown")
        self.assertEqual((stats.trip_inferred, stats.unknown), (0, 1))

    def test_session_inheritance_keeps_priority(self):
        # inside the six hours the nearest-in-time GPS file is the more precise source,
        # so that level must still win — the trip pass only fills what it left
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        near = self.add_file("2023-05-01T11:00:00")
        far = self.add_file("2023-05-02T11:00:00")
        self.add_file("2023-05-03T10:00:00", at=_MOSCOW)
        stats = self.run_geo()
        self.assertEqual(self.place_of(near)["confidence"], "session_inferred")
        self.assertEqual(self.place_of(far)["confidence"], "trip_inferred")
        self.assertEqual((stats.session_inferred, stats.trip_inferred), (1, 1))

    def test_exact_gps_is_never_overwritten(self):
        moscow = self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        self.add_file("2023-05-02T10:00:00", at=_MOSCOW)
        self.run_geo()
        self.assertEqual(self.place_of(moscow)["confidence"], "exact_gps")

    def test_low_date_confidence_does_not_inherit(self):
        # the same guard as at the session level: a date from mtime does not put the
        # file in the trip in the first place
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        low = self.add_file("2023-05-02T10:00:00", confidence="low")
        self.add_file("2023-05-03T10:00:00", at=_MOSCOW)
        self.run_geo()
        self.assertEqual((self.place_of(low)["city"], self.place_of(low)["confidence"]),
                         (None, "unknown"))

    def test_file_without_a_date_is_not_in_any_trip(self):
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        undated = self.add_file(taken_at=None)
        self.add_file("2023-05-03T10:00:00", at=_MOSCOW)
        self.run_geo()
        self.assertEqual(self.place_of(undated)["confidence"], "unknown")

    def test_district_is_not_inherited_from_the_trip(self):
        # the trip agreed on a CITY; a district would be a finer claim than the
        # evidence supports (the session level, being tighter, does inherit it)
        self.add_file("2023-05-01T10:00:00", at=_SPB)
        orphan = self.add_file("2023-05-02T10:00:00")
        self.add_file("2023-05-03T10:00:00", at=_SPB)
        self.run_geo()
        row = self.place_of(orphan)
        self.assertEqual((row["city_geonameid"], row["confidence"]),
                         (_GID_SPB, "trip_inferred"))
        self.assertIsNone(row["district_geonameid"])
        self.assertIsNone(row["district_name"])


class TestTripBracketing(_TripBase):
    """A file inherits only from BETWEEN two frames of the trip's city.

    In the middle of a trip the GPS frames are an alibi: the camera was in this city
    before the file and in the same city after it. Past the last frame there is no alibi
    — nothing says when its owner left, and a day trip out of town lands exactly there.
    Measured: this one rule is the difference between 94.2% precision and 99.0%.
    """

    def test_a_file_after_the_last_gps_frame_is_not_inherited(self):
        # the measured failure mode: the trip is Moscow all the way, the file is a day
        # after its last GPS frame — and was in fact taken 600 km away
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        self.add_file("2023-05-02T10:00:00", at=_MOSCOW)
        tail = self.add_file("2023-05-03T10:00:00")
        stats = self.run_geo()
        self.assertEqual(self.place_of(tail)["confidence"], "unknown")
        self.assertEqual(stats.trip_inferred, 0)

    def test_a_file_before_the_first_gps_frame_is_not_inherited(self):
        # the same on the other end: the first GPS frame does not say when the owner
        # arrived (the collection showed fewer of these, the physics is the same)
        head = self.add_file("2023-05-01T10:00:00")
        self.add_file("2023-05-02T10:00:00", at=_MOSCOW)
        self.add_file("2023-05-03T10:00:00", at=_MOSCOW)
        self.run_geo()
        self.assertEqual(self.place_of(head)["confidence"], "unknown")

    def test_only_frames_of_the_dominant_city_bracket(self):
        # bracketed by SOME GPS (Moscow before, Zelenograd after) but not by the city
        # the trip lends. Zelenograd is 38 km away, so it is the same trip and Moscow
        # still holds 3 of 4 frames — yet the alibi for MOSCOW ends at 05-02
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        self.add_file("2023-05-01T14:00:00", at=_MOSCOW)
        self.add_file("2023-05-02T10:00:00", at=_MOSCOW)
        between = self.add_file("2023-05-03T10:00:00")
        self.add_file("2023-05-04T10:00:00", at=_ZELENOGRAD)
        stats = self.run_geo()
        self.assertEqual(self.place_of(between)["confidence"], "unknown")
        self.assertEqual(stats.trip_inferred, 0)

    def test_one_more_frame_at_the_end_opens_the_tail(self):
        # the contrast with the first test: the same tail file, and a Moscow frame after
        # it. Nothing else changed — the alibi is what the rule is about
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        self.add_file("2023-05-02T10:00:00", at=_MOSCOW)
        tail = self.add_file("2023-05-03T10:00:00")
        self.add_file("2023-05-04T10:00:00", at=_MOSCOW)
        self.run_geo()
        self.assertEqual((self.place_of(tail)["city_geonameid"],
                          self.place_of(tail)["confidence"]),
                         (_GID_MOSCOW, "trip_inferred"))


class TestTripDominantCity(_TripBase):
    """The conservative rule: the dominant city must hold MORE than half of the trip's
    GPS frames. A trip across three cities is left alone — a file in a foreign city is
    worse than an empty folder, because nobody will look for it there (F75/F86)."""

    def test_a_tie_between_two_cities_inherits_nothing(self):
        # Moscow and Zelenograd are 38 km apart: the same country and inside
        # trip_merge_max_km, so they ARE one trip — and exactly 50% is not > 50%.
        # The file sits between the two frames, so only the tie can be what stops it
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        orphan = self.add_file("2023-05-02T10:00:00")
        self.add_file("2023-05-03T10:00:00", at=_ZELENOGRAD)
        stats = self.run_geo()
        self.assertEqual(self.place_of(orphan)["confidence"], "unknown")
        self.assertEqual(stats.trip_inferred, 0)

    def test_a_dominant_city_above_half_is_inherited(self):
        # 3 Moscow frames to 1 Zelenograd, and Moscow on both sides of the file
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        self.add_file("2023-05-01T14:00:00", at=_MOSCOW)
        orphan = self.add_file("2023-05-02T10:00:00")
        self.add_file("2023-05-03T10:00:00", at=_ZELENOGRAD)
        self.add_file("2023-05-04T10:00:00", at=_MOSCOW)
        self.run_geo()
        row = self.place_of(orphan)
        self.assertEqual((row["city_geonameid"], row["confidence"]),
                         (_GID_MOSCOW, "trip_inferred"))

    def test_a_far_city_is_a_different_trip(self):
        # 635 km — beyond trip_merge_max_km, so the place-less session belongs to the
        # SECOND trip and inherits Saint Petersburg, not the dominant Moscow
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        self.add_file("2023-05-01T14:00:00", at=_MOSCOW)
        self.add_file("2023-05-02T10:00:00", at=_SPB)
        orphan = self.add_file("2023-05-03T10:00:00")
        self.add_file("2023-05-04T10:00:00", at=_SPB)
        self.run_geo()
        row = self.place_of(orphan)
        self.assertEqual((row["city_geonameid"], row["confidence"]),
                         (_GID_SPB, "trip_inferred"))

    def test_another_country_starts_a_new_trip(self):
        # the anchor of a trip is its own first placed session; a flight abroad ends it
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        self.add_file("2023-05-02T10:00:00", at=_PARIS)
        orphan = self.add_file("2023-05-03T10:00:00")
        self.add_file("2023-05-04T10:00:00", at=_PARIS)
        self.run_geo()
        row = self.place_of(orphan)
        self.assertEqual((row["country"], row["city"], row["confidence"]),
                         ("FR", "Paris", "trip_inferred"))


class TestTripThresholdsComeFromEvents(_TripBase):
    """The thresholds are the ones `events` cuts trips by — geo owns a copy of the
    rule, not a second set of knobs (F85a: they must not drift apart)."""

    def test_trip_merge_gap_hours_is_read(self):
        self.cfg.events = EventsConfig(trip_merge_gap_hours=12)
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        orphan = self.add_file("2023-05-02T10:00:00")  # 24 h > 12
        self.add_file("2023-05-03T10:00:00", at=_MOSCOW)
        self.run_geo()
        self.assertEqual(self.place_of(orphan)["confidence"], "unknown")

    def test_trip_merge_max_km_is_read(self):
        # a day in Zelenograd, 38 km out, in the middle of a Moscow trip. Under the
        # default 120 km it is one trip: Moscow holds 2 frames of 3 and stands on both
        # sides of the file, so the file gets Moscow
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        self.add_file("2023-05-02T10:00:00", at=_ZELENOGRAD)
        orphan = self.add_file("2023-05-03T10:00:00")
        self.add_file("2023-05-04T10:00:00", at=_MOSCOW)
        self.run_geo()
        self.assertEqual((self.place_of(orphan)["city_geonameid"],
                          self.place_of(orphan)["confidence"]),
                         (_GID_MOSCOW, "trip_inferred"))

        # with the distance branch off only equal cities merge, so Zelenograd cuts the
        # chain in three and the file is left in the middle piece, which knows only a
        # Zelenograd frame from the day before — behind it, vouching for nothing
        self.cfg.events = EventsConfig(trip_merge_max_km=0)
        self.run_geo()
        self.assertEqual(self.place_of(orphan)["confidence"], "unknown")

    def test_session_gap_still_belongs_to_geo(self):
        # the session level keeps its own knob: a wider session must not be needed to
        # get the trip level working (and is not what F85a changes)
        self.cfg.geo = GeoConfig(session_gap_hours=1)
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        orphan = self.add_file("2023-05-01T14:00:00")  # 4 h > 1 h — another session
        self.add_file("2023-05-01T18:00:00", at=_MOSCOW)
        stats = self.run_geo()
        self.assertEqual(self.place_of(orphan)["confidence"], "trip_inferred")
        self.assertEqual((stats.session_inferred, stats.trip_inferred), (0, 1))


class TestTripIdempotency(_TripBase):
    def test_rerun_gives_the_same_places(self):
        self.add_file("2023-05-01T10:00:00", at=_MOSCOW)
        self.add_file("2023-05-01T11:00:00")
        self.add_file("2023-05-02T10:00:00")
        self.add_file("2023-05-03T10:00:00", at=_MOSCOW)
        self.add_file(taken_at=None)
        self.run_geo()
        first = self.conn.execute(
            "SELECT file_id, city_geonameid, confidence FROM places ORDER BY file_id"
        ).fetchall()
        self.run_geo()
        second = self.conn.execute(
            "SELECT file_id, city_geonameid, confidence FROM places ORDER BY file_id"
        ).fetchall()
        self.assertEqual([tuple(r) for r in first], [tuple(r) for r in second])
        self.assertEqual([r["confidence"] for r in first],
                         ["exact_gps", "session_inferred", "trip_inferred", "exact_gps",
                          "unknown"])


if __name__ == "__main__":
    unittest.main()
