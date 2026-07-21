"""Events: sessions by gaps, merging by city_id, the size threshold, localized names,
recomputation, manual events (F4/F30); the online city fallback and trip merging by
region/proximity (F44/#19)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sorta.config import Config, EventsConfig
from sorta.db import connect
from sorta.events import add_manual_event, build_events, rename_event
from sorta.geodata import GeoResolver

# geonameid from bundled data/geo (real records — brief F30 allows testing
# localization either with a geodata.name mock or a real bundled record)
MOSCOW = 524901   # ru: Москва
PARIS = 2988507   # ru: Париж

# F44/#19-B: a fixture set of geonameid for the region/proximity merge tests —
# admin1.tsv/countries.tsv are not in this worktree yet, so the merge tests use their
# own GeoResolver over temporary TSVs, not bundled data/geo.
CITY_A = 9001   # ID, admin1 BA ("Bali")
CITY_B = 9002   # ID, admin1 XX — ~56km from CITY_A, a DIFFERENT region (proximity check)
CITY_C = 9003   # ID, admin1 BA — the same region as CITY_A, but ~556km (region check)
CITY_D = 9004   # TH, admin1 10 — a different country, far (never merges)


def _write_geo_fixture(d: Path) -> None:
    """places/admin1/countries/names.tsv with synthetic geonameids CITY_A..D."""
    d.mkdir(parents=True, exist_ok=True)
    places = [
        (CITY_A, "0.0", "100.0", "PPLA", "ID", "BA", "", "CityA", "10000"),
        (CITY_B, "0.5", "100.0", "PPLA", "ID", "XX", "", "CityB", "10000"),
        (CITY_C, "5.0", "100.0", "PPLA", "ID", "BA", "", "CityC", "10000"),
        (CITY_D, "20.0", "120.0", "PPLA", "TH", "10", "", "CityD", "10000"),
    ]
    (d / "places.tsv").write_text(
        "\n".join("\t".join(map(str, row)) for row in places) + "\n", encoding="utf-8")
    admin1 = [
        ("ID", "BA", 9101, "Bali"),
        ("ID", "XX", 9102, "XX-region"),
        ("TH", "10", 9103, "Bangkok-region"),
    ]
    (d / "admin1.tsv").write_text(
        "\n".join("\t".join(map(str, row)) for row in admin1) + "\n", encoding="utf-8")
    countries = [
        ("ID", 9201, "Indonesia"),
        ("TH", 9202, "Thailand"),
    ]
    (d / "countries.tsv").write_text(
        "\n".join("\t".join(map(str, row)) for row in countries) + "\n", encoding="utf-8")
    names = [
        (9101, "ru", "Бали"),
        (9201, "ru", "Индонезия"),
    ]
    (d / "names.tsv").write_text(
        "\n".join("\t".join(map(str, row)) for row in names) + "\n", encoding="utf-8")


class EventsBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(sources=[Path(self.tmp.name)],
                          database=Path(self.tmp.name) / "test.db")
        # F30: min_event_size/trip_merge_gap_hours are not EventsConfig fields yet
        # (getattr fallback in events.py) — this file's tests check sessions/merging/
        # names/manual events separately from the size threshold, so we keep the
        # threshold=1 (non-blocking) and the trip gap as the old merge_gap_hours
        # (=18) for regression compatibility; the threshold/defaults themselves — TestDefaults.
        self.cfg.events.min_event_size = 1
        self.cfg.events.trip_merge_gap_hours = 18
        self.conn = connect(self.cfg.database)
        self._n = 0

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file(self, taken_at, confidence="high", city_id=None, dup_of=None, error=None,
                 country=None, district_name=None, city=None):
        self._n += 1
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, taken_at,
                   taken_at_source, taken_at_confidence, dup_of, error, indexed_at)
               VALUES (?, 1000, 0, 'jpg', 'photo', ?, 'exif', ?, ?, ?, '2026-01-01')""",
            (f"/photos/img_{self._n}.jpg", taken_at, confidence, dup_of, error),
        )
        fid = cur.lastrowid
        # F44/#19-A1: district_name/city — the string fallback for online places
        # without a geonameid; country — for the "same country" check when merging trips.
        if city_id is not None or country is not None or district_name is not None \
                or city is not None:
            cc = country if country is not None else ("RU" if city_id is not None else None)
            self.conn.execute(
                """INSERT INTO places (file_id, country, region, city, city_geonameid,
                       district_name, confidence, updated_at)
                   VALUES (?, ?, NULL, ?, ?, ?, 'exact_gps', '2026-01-01')""",
                (fid, cc, city, city_id, district_name),
            )
        self.conn.commit()
        return fid

    def events(self):
        return self.conn.execute(
            "SELECT * FROM events ORDER BY started_at").fetchall()

    def files_of(self, event_id):
        return {r["file_id"] for r in self.conn.execute(
            "SELECT file_id FROM event_files WHERE event_id = ?", (event_id,))}

    def event_of(self, file_id):
        rows = self.conn.execute(
            "SELECT event_id FROM event_files WHERE file_id = ?", (file_id,)).fetchall()
        return [r["event_id"] for r in rows]


class TestAutoClustering(EventsBase):
    def test_three_clusters_by_gap(self):
        # 3 clumps without a city, gaps 9+ h (> 6): three events, no merging
        a = [self.add_file("2023-05-01T10:00:00"), self.add_file("2023-05-01T11:00:00")]
        b = [self.add_file("2023-05-01T21:00:00")]
        c = [self.add_file("2023-05-02T08:00:00")]
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 3)
        evs = self.events()
        self.assertEqual(self.files_of(evs[0]["id"]), set(a))
        self.assertEqual(self.files_of(evs[1]["id"]), set(b))
        self.assertEqual(self.files_of(evs[2]["id"]), set(c))

    def test_merge_evening_morning_same_city(self):
        # a wedding: evening + morning, one city (city_id), a 10 h gap < 18 → one event
        evening = self.add_file("2023-05-01T18:00:00", city_id=MOSCOW)
        morning = self.add_file("2023-05-02T09:00:00", city_id=MOSCOW)
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)
        ev = self.events()[0]
        self.assertEqual(self.files_of(ev["id"]), {evening, morning})
        self.assertEqual(ev["name"], "2023-05-01..05-02 Moscow")
        self.assertEqual(ev["place_city"], "Moscow")
        self.assertEqual(ev["origin"], "auto")

    def test_no_merge_different_cities(self):
        # different city_ids (not districts of one city) — two events, even within the gap
        self.add_file("2023-05-01T18:00:00", city_id=MOSCOW)
        self.add_file("2023-05-02T09:00:00", city_id=PARIS)
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)

    def test_no_merge_beyond_trip_gap(self):
        self.add_file("2023-05-01T10:00:00", city_id=MOSCOW)
        self.add_file("2023-05-02T09:00:00", city_id=MOSCOW)  # a 23 h gap > 18
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)

    def test_low_confidence_excluded_from_auto(self):
        kept = self.add_file("2023-05-01T10:00:00")
        low = self.add_file("2023-05-01T10:30:00", confidence="low")
        build_events(self.cfg, self.conn)
        self.assertEqual(self.event_of(low), [])
        self.assertEqual(len(self.event_of(kept)), 1)

    def test_skips_duplicates_errors_and_undated(self):
        canon = self.add_file("2023-05-01T10:00:00")
        dup = self.add_file("2023-05-01T10:00:00", dup_of=canon)
        broken = self.add_file("2023-05-01T10:00:00", error="boom")
        undated = self.add_file(None)
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_files, 1)
        for fid in (dup, broken, undated):
            self.assertEqual(self.event_of(fid), [])

    def test_default_names(self):
        self.add_file("2023-05-01T10:00:00", city_id=MOSCOW)   # a day + a city
        self.add_file("2023-06-10T10:00:00")                   # a day without a city
        self.add_file("2023-12-31T20:00:00", city_id=PARIS)    # a year change
        self.add_file("2024-01-01T02:00:00", city_id=PARIS)
        build_events(self.cfg, self.conn)
        names = [e["name"] for e in self.events()]
        self.assertEqual(names, ["2023-05-01 Moscow", "2023-06-10",
                                 "2023-12-31..2024-01-01 Paris"])

    def test_gap_hours_from_config(self):
        self.cfg.events = EventsConfig(gap_hours=1, merge_gap_hours=2)
        self.cfg.events.min_event_size = 1  # the threshold is not what this test checks
        self.add_file("2023-05-01T10:00:00")
        self.add_file("2023-05-01T13:00:00")  # 3 h > 1 h; no city — no merging
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)

    def test_recompute_idempotent(self):
        self.add_file("2023-05-01T10:00:00", city_id=MOSCOW)
        self.add_file("2023-05-02T09:00:00", city_id=MOSCOW)
        build_events(self.cfg, self.conn)
        first = [tuple(e) for e in self.events()]
        build_events(self.cfg, self.conn)
        second = [tuple(e) for e in self.events()]
        # ids are recreated, the content is stable
        self.assertEqual([e[1:] for e in first], [e[1:] for e in second])


class TestSizeThreshold(EventsBase):
    def test_below_threshold_group_not_created(self):
        self.cfg.events.min_event_size = 3
        below = [self.add_file("2023-05-01T10:00:00"), self.add_file("2023-05-01T10:30:00")]
        # a separate session (far in time) that passes the threshold
        above = [self.add_file("2023-06-01T10:00:00"),
                self.add_file("2023-06-01T10:30:00"),
                self.add_file("2023-06-01T11:00:00")]
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)
        self.assertEqual(stats.auto_files, 3)
        for fid in below:
            self.assertEqual(self.event_of(fid), [])
        ev = self.events()[0]
        self.assertEqual(self.files_of(ev["id"]), set(above))

    def test_manual_events_ignore_threshold(self):
        # a manual event over 1 file — with min_event_size=10 an auto group of this
        # size would not be created, but manual events are not subject to the threshold
        self.cfg.events.min_event_size = 10
        one = self.add_file("2026-01-03T12:00:00")
        eid = add_manual_event(self.conn, "Разовое", "2026-01-01", "2026-01-10")
        self.assertEqual(self.event_of(one), [eid])
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.manual_events, 1)
        self.assertEqual(stats.manual_files, 1)
        self.assertEqual(self.event_of(one), [eid])


class TestDefaults(EventsBase):
    """The real F30 defaults (min_event_size=5, trip_merge_gap_hours=48) via the
    getattr fallback — EventsBase.setUp overrides them for the regression tests above,
    here we use a "clean" EventsConfig without overrides."""

    def setUp(self):
        super().setUp()
        self.cfg.events = EventsConfig()  # without min_event_size/trip_merge_gap_hours overrides

    def test_default_min_event_size_blocks_group_of_four(self):
        for i in range(4):
            self.add_file(f"2023-05-01T1{i}:00:00")
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 0)
        self.assertEqual(stats.auto_files, 0)

    def test_default_min_event_size_allows_group_of_five(self):
        for i in range(5):
            self.add_file(f"2023-05-01T1{i}:00:00")
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)
        self.assertEqual(stats.auto_files, 5)

    def test_default_trip_gap_merges_within_48h(self):
        for i in range(3):
            self.add_file(f"2023-05-01T1{i}:00:00", city_id=MOSCOW)
        for i in range(3):
            self.add_file(f"2023-05-03T0{i}:00:00", city_id=MOSCOW)  # ~38h after the first session, < 48
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)
        self.assertEqual(stats.auto_files, 6)

    def test_default_trip_gap_no_merge_beyond_48h(self):
        # 5 files in each session — both pass min_event_size on their own, we isolate
        # trip_gap as the variable
        for i in range(5):
            self.add_file(f"2023-05-01T{10 + i}:00:00", city_id=MOSCOW)
        for i in range(5):
            self.add_file(f"2023-05-04T0{i}:00:00", city_id=MOSCOW)  # >48h after
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)


class TestRename(EventsBase):
    def test_rename_sets_manual_flag(self):
        self.add_file("2023-05-01T10:00:00")
        build_events(self.cfg, self.conn)
        ev = self.events()[0]
        rename_event(self.conn, ev["id"], "Дача")
        ev = self.events()[0]
        self.assertEqual((ev["name"], ev["name_is_manual"]), ("Дача", 1))

    def test_rename_missing_event_raises(self):
        with self.assertRaises(ValueError):
            rename_event(self.conn, 999, "Нет такого")

    def test_manual_name_survives_recompute(self):
        ids = [self.add_file(f"2023-05-01T1{i}:00:00") for i in range(3)]
        build_events(self.cfg, self.conn)
        rename_event(self.conn, self.events()[0]["id"], "Свадьба")
        # a new file in the same cluster: 3/3 overlap of the old ones > 50%
        ids.append(self.add_file("2023-05-01T14:00:00"))
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.names_preserved, 1)
        ev = self.events()[0]
        self.assertEqual((ev["name"], ev["name_is_manual"]), ("Свадьба", 1))
        self.assertEqual(self.files_of(ev["id"]), set(ids))

    def test_name_not_transferred_below_half_overlap(self):
        a = self.add_file("2023-05-01T10:00:00")
        b = self.add_file("2023-05-01T11:00:00")
        build_events(self.cfg, self.conn)
        rename_event(self.conn, self.events()[0]["id"], "Старое")
        # the old files became duplicates → 0% overlap — the name is not carried over
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?", (a, b))
        self.conn.execute("UPDATE files SET error = 'gone' WHERE id = ?", (a,))
        self.conn.commit()
        self.add_file("2023-05-01T10:30:00")
        build_events(self.cfg, self.conn)
        ev = self.events()[0]
        self.assertEqual((ev["name"], ev["name_is_manual"]), ("2023-05-01", 0))


class TestManualEvents(EventsBase):
    def test_captures_range_including_low(self):
        inside = self.add_file("2026-01-03T12:00:00", city_id=MOSCOW)
        low = self.add_file("2026-01-05T12:00:00", confidence="low")
        outside = self.add_file("2026-02-01T12:00:00")
        dup = self.add_file("2026-01-04T12:00:00", dup_of=inside)
        eid = add_manual_event(self.conn, "Конференция", "2026-01-01", "2026-01-10")
        ev = self.events()[0]
        self.assertEqual((ev["id"], ev["name"], ev["origin"], ev["name_is_manual"]),
                         (eid, "Конференция", "manual", 1))
        # F30: a manual event's place_city is localized too (with the default
        # language — add_manual_event is called from cli.py without cfg)
        self.assertEqual(ev["place_city"], "Moscow")
        self.assertEqual(self.files_of(eid), {inside, low})
        self.assertEqual(self.event_of(outside), [])
        self.assertEqual(self.event_of(dup), [])

    def test_priority_over_existing_auto(self):
        fid = self.add_file("2026-01-03T12:00:00")
        build_events(self.cfg, self.conn)
        eid = add_manual_event(self.conn, "Ёлка", "2026-01-03", "2026-01-03")
        # the file is taken from the auto event, the emptied auto event is deleted
        self.assertEqual(self.event_of(fid), [eid])
        self.assertEqual([e["origin"] for e in self.events()], ["manual"])

    def test_excluded_from_auto_on_recompute(self):
        manual_file = self.add_file("2026-01-03T12:00:00")
        auto_file = self.add_file("2026-02-01T12:00:00")
        eid = add_manual_event(self.conn, "Конференция", "2026-01-01", "2026-01-10")
        stats = build_events(self.cfg, self.conn)
        self.assertEqual((stats.manual_events, stats.auto_events), (1, 1))
        self.assertEqual(self.event_of(manual_file), [eid])
        self.assertEqual(len(self.event_of(auto_file)), 1)
        self.assertNotIn(eid, self.event_of(auto_file))

    def test_survives_recompute_and_picks_new_files(self):
        old = self.add_file("2026-01-03T12:00:00")
        eid = add_manual_event(self.conn, "Конференция", "2026-01-01", "2026-01-10")
        # new range files were indexed after the event was created
        new = self.add_file("2026-01-07T09:00:00")
        new_low = self.add_file("2026-01-08T09:00:00", confidence="low")
        build_events(self.cfg, self.conn)
        ev = self.conn.execute("SELECT * FROM events WHERE id = ?", (eid,)).fetchone()
        self.assertEqual((ev["name"], ev["origin"]), ("Конференция", "manual"))
        self.assertEqual(self.files_of(eid), {old, new, new_low})

    def test_overlapping_ranges_rejected(self):
        add_manual_event(self.conn, "Первое", "2026-01-01", "2026-01-10")
        with self.assertRaisesRegex(ValueError, "пересекается.*Первое"):
            add_manual_event(self.conn, "Второе", "2026-01-10", "2026-01-15")
        # back-to-back without overlap — allowed
        eid = add_manual_event(self.conn, "Третье", "2026-01-11", "2026-01-15")
        self.assertGreater(eid, 0)

    def test_bad_dates_rejected(self):
        with self.assertRaises(ValueError):
            add_manual_event(self.conn, "X", "не дата", "2026-01-10")
        with self.assertRaises(ValueError):
            add_manual_event(self.conn, "X", "2026-01-10", "2026-01-01")


class TestOnlineCityFallback(EventsBase):
    """F44/#19-A1: city_geonameid IS NULL (online, G2b) — fallback to district_name/city."""

    def test_single_file_gets_city_from_district_name(self):
        fid = self.add_file("2023-05-01T10:00:00", district_name="Пхукет")
        build_events(self.cfg, self.conn)
        ev = self.events()[0]
        self.assertEqual(ev["place_city"], "Пхукет")
        self.assertEqual(ev["name"], "2023-05-01 Пхукет")
        self.assertEqual(self.files_of(ev["id"]), {fid})

    def test_falls_back_to_city_when_no_district_name(self):
        self.add_file("2023-05-01T10:00:00", city="Пхукет")
        build_events(self.cfg, self.conn)
        self.assertEqual(self.events()[0]["place_city"], "Пхукет")

    def test_district_name_wins_over_city(self):
        self.add_file("2023-05-01T10:00:00", district_name="Патонг", city="Пхукет")
        build_events(self.cfg, self.conn)
        self.assertEqual(self.events()[0]["place_city"], "Патонг")

    def test_two_online_files_same_string_form_one_event(self):
        # one string (case-insensitive) in one window (session) → one event
        a = self.add_file("2023-05-01T10:00:00", district_name="Пхукет")
        b = self.add_file("2023-05-01T11:00:00", district_name="ПХУКЕТ")
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)
        ev = self.events()[0]
        self.assertEqual(self.files_of(ev["id"]), {a, b})
        self.assertEqual(ev["place_city"], "Пхукет")

    def test_online_sessions_merge_by_matching_string(self):
        # two DIFFERENT sessions (gap > gap_hours), one city string, gap <
        # trip_merge_gap_hours (18, EventsBase) → merge into one trip
        evening = [self.add_file("2023-05-01T18:00:00", district_name="Пхукет", country="TH")]
        morning = [self.add_file("2023-05-02T09:00:00", district_name="Пхукет", country="TH")]
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)
        ev = self.events()[0]
        self.assertEqual(self.files_of(ev["id"]), set(evening) | set(morning))
        self.assertEqual(ev["place_city"], "Пхукет")

    def test_online_sessions_no_merge_different_string(self):
        # online cities without a geonameid: no coordinates/region — merging only on
        # string equality, different strings do not merge even within trip_gap
        self.add_file("2023-05-01T18:00:00", district_name="Пхукет", country="TH")
        self.add_file("2023-05-02T09:00:00", district_name="Патайя", country="TH")
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)


class TestTripMergeRegionProximity(EventsBase):
    """F44/#19-B: merging adjacent sessions into a trip by region/proximity; the name
    is by the region/country of the dominant city. Uses a fixture GeoResolver
    (see _write_geo_fixture) — bundled admin1.tsv/countries.tsv are not in this
    worktree yet."""

    def setUp(self):
        super().setUp()
        geo_dir = Path(self.tmp.name) / "geo_fixture"
        _write_geo_fixture(geo_dir)
        patcher = patch("sorta.events.GeoResolver",
                         lambda *a, **k: GeoResolver(data_dir=geo_dir))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_merge_by_proximity_different_region(self):
        # CITY_A/CITY_B: different admin1 (BA/XX), ~56km apart (< the default 120km)
        a = [self.add_file(f"2023-05-01T18:0{i}:00", city_id=CITY_A, country="ID")
             for i in range(3)]
        b = [self.add_file("2023-05-02T09:00:00", city_id=CITY_B, country="ID")]
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)
        ev = self.events()[0]
        self.assertEqual(self.files_of(ev["id"]), set(a) | set(b))
        # the dominant one (3 files) — CITY_A/admin1 BA → the region name
        self.assertEqual(ev["place_city"], "Bali")

    def test_no_merge_beyond_proximity_threshold(self):
        self.cfg.events.trip_merge_max_km = 10  # < ~56km between CITY_A and CITY_B
        self.add_file("2023-05-01T18:00:00", city_id=CITY_A, country="ID")
        self.add_file("2023-05-02T09:00:00", city_id=CITY_B, country="ID")
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)

    def test_proximity_disabled_by_zero_threshold(self):
        # trip_merge_max_km<=0 disables the proximity branch — back to city/region,
        # CITY_A/CITY_B in different admin1 → do not merge
        self.cfg.events.trip_merge_max_km = 0
        self.add_file("2023-05-01T18:00:00", city_id=CITY_A, country="ID")
        self.add_file("2023-05-02T09:00:00", city_id=CITY_B, country="ID")
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)

    def test_merge_by_same_region_beyond_proximity_threshold(self):
        # CITY_A/CITY_C: same admin1 (BA), but ~556km (> the default 120km) — merge by region
        a = [self.add_file(f"2023-05-01T18:0{i}:00", city_id=CITY_A, country="ID")
             for i in range(3)]
        c = [self.add_file("2023-05-02T09:00:00", city_id=CITY_C, country="ID")]
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 1)
        ev = self.events()[0]
        self.assertEqual(self.files_of(ev["id"]), set(a) | set(c))
        self.assertEqual(ev["place_city"], "Bali")

    def test_no_merge_different_country(self):
        # CITY_D — a different country (TH), close in time, but the country decides
        self.add_file("2023-05-01T18:00:00", city_id=CITY_A, country="ID")
        self.add_file("2023-05-02T09:00:00", city_id=CITY_D, country="TH")
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)

    def test_no_merge_beyond_trip_gap_same_region(self):
        # same region (BA), but a 23h gap > trip_merge_gap_hours (18) — the guard is not weakened
        self.add_file("2023-05-01T10:00:00", city_id=CITY_A, country="ID")
        self.add_file("2023-05-02T09:00:00", city_id=CITY_C, country="ID")
        stats = build_events(self.cfg, self.conn)
        self.assertEqual(stats.auto_events, 2)

    def test_single_city_group_named_by_city_not_region(self):
        # F30 regression: a single-city group is named by the city, not the region,
        # even when the region is known to the fixture resolver
        for i in range(3):
            self.add_file(f"2023-05-01T18:0{i}:00", city_id=CITY_A, country="ID")
        build_events(self.cfg, self.conn)
        ev = self.events()[0]
        self.assertEqual(ev["place_city"], "CityA")
        self.assertEqual(ev["name"], "2023-05-01 CityA")


if __name__ == "__main__":
    unittest.main()
