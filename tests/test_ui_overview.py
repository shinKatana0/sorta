"""F108: the "Overview" tab — the state of the collection in one screen.

Every key number of a collection used to be reachable only through a hand-written SQL
query: how many frames have no place, what sits in the service buckets, whether the deep
tier ran at all, whether a layout ran and was finished. This view answers all of it, and
three properties of it are load-bearing and pinned below:

* the numbers are plain aggregates — opening the tab must NOT build a plan (minutes on a
  live collection), which is exercised by spying on `PlanCache._build`;
* nothing is cached: the tab is opened right AFTER a run to see what changed;
* aggregates only — no file path and no file id leaves the endpoint.

The counters are checked against SQL the test writes itself, so an accidental change of
the population (duplicates, unreadable files) shows up here rather than in a screenshot.
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from sorta import ui

from tests.test_ui import UiServerTestBase


class OverviewTestBase(UiServerTestBase):
    """Index fixtures written straight into the tables the overview reads.

    Rows go in without a file on disk on purpose: nothing in this view opens a file, and
    a test that has to encode a JPEG to count a number would hide exactly that.
    """

    def add_file(self, rel: str, *, media_type: str = "photo",
                 dup_of: int | None = None, error: str | None = None) -> int:
        self._n += 1
        path = str((self.src_dir / rel).resolve())
        cur = self.conn.execute(
            """INSERT INTO files (path, size, mtime, ext, media_type, hash, hash_algo,
                   taken_at, taken_at_source, taken_at_confidence, dup_of, error,
                   indexed_at)
               VALUES (?, 100, 0, 'jpg', ?, ?, 'blake3', '2022-05-01T10:00:00', 'exif',
                       'high', ?, ?, '2026-01-01')""",
            (path, media_type, f"hash{self._n}", dup_of, error))
        self.conn.commit()
        return cur.lastrowid

    def set_place(self, file_id: int, *, confidence: str = "exact_gps",
                  country: str | None = "ru", city: str | None = "Moscow") -> None:
        self.conn.execute(
            """INSERT INTO places (file_id, country, region, city, confidence, updated_at)
               VALUES (?, ?, NULL, ?, ?, '2026-01-01')""",
            (file_id, country, city, confidence))
        self.conn.commit()

    def set_manual_place(self, file_id: int, *, country: str = "fr",
                         city: str | None = "Paris") -> None:
        self.conn.execute(
            """INSERT INTO manual_places (file_id, country, city, city_geonameid,
                   updated_at)
               VALUES (?, ?, ?, NULL, '2026-07-28')""",
            (file_id, country, city))
        self.conn.commit()

    def classify(self, file_id: int, verdict: str, *, source: str = "clip",
                 tier: str | None = "clip", updated_at: str = "2026-07-28") -> None:
        self.conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, score, updated_at, tier)
               VALUES (?, ?, ?, NULL, ?, ?)""",
            (file_id, verdict, source, updated_at, tier))
        self.conn.commit()

    def add_event(self, file_id: int, *, name: str = "Поездка") -> int:
        cur = self.conn.execute(
            """INSERT INTO events (started_at, ended_at, name, name_is_manual, origin)
               VALUES ('2022-05-01T09:00:00', '2022-05-01T20:00:00', ?, 0, 'auto')""",
            (name,))
        event_id = cur.lastrowid
        self.conn.execute(
            "INSERT INTO event_files (event_id, file_id) VALUES (?, ?)",
            (event_id, file_id))
        self.conn.commit()
        return event_id

    def add_batch(self, *, mode: str = "city", dest_root: str | None = None,
                  started_at: str = "2026-07-01T10:00:00",
                  finished_at: str | None = "2026-07-01T10:30:00",
                  operation: str = "move") -> int:
        cur = self.conn.execute(
            """INSERT INTO move_batches (mode, dest_root, started_at, finished_at,
                   operation)
               VALUES (?, ?, ?, ?, ?)""",
            (mode, dest_root or str(self.root / "dest"), started_at, finished_at,
             operation))
        self.conn.commit()
        return cur.lastrowid

    def add_move(self, batch_id: int, file_id: int, *, status: str = "done") -> None:
        self.conn.execute(
            """INSERT INTO moves (batch_id, file_id, src, dst, hash, status)
               VALUES (?, ?, 'src', 'dst', 'deadbeef', ?)""",
            (batch_id, file_id, status))
        self.conn.commit()

    def overview(self) -> dict:
        status, body, ctype = self.get("/api/overview")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        return json.loads(body)

    def scalar(self, sql: str) -> int:
        return int(self.conn.execute(sql).fetchone()[0])


class TestEmptyIndex(OverviewTestBase):
    def test_an_empty_index_answers_with_zeros_and_says_so(self):
        self.start_server()
        data = self.overview()
        self.assertTrue(data["empty"])
        self.assertEqual(data["collection"], {"files": 0, "photos": 0, "videos": 0,
                                              "duplicates": 0, "errors": 0, "events": 0,
                                              "animals": 0, "blurred": 0,
                                              "eyes_closed": 0, "no_subject": 0})
        self.assertEqual(data["place"]["total"], 0)
        self.assertEqual(data["place"]["confidence"], [])
        self.assertEqual(data["place"]["no_place"], 0)
        self.assertEqual(data["classes"]["total"], 0)
        self.assertEqual(data["classes"]["verdicts"], [])
        self.assertFalse(data["classes"]["vlm_ran"])
        self.assertIsNone(data["classes"]["updated_at"])
        self.assertEqual(data["layout"], {"batches": 0, "unfinished": 0, "last": None})

    def test_the_percentage_of_nothing_does_not_divide_by_zero(self):
        self.start_server()
        self.assertEqual(self.overview()["place"]["no_place_percent"], 0.0)

    def test_a_non_empty_index_is_not_flagged_empty(self):
        self.add_file("a.jpg")
        self.start_server()
        self.assertFalse(self.overview()["empty"])


class TestCollectionCounts(OverviewTestBase):
    def test_the_counters_match_direct_sql(self):
        for i in range(3):
            self.add_file(f"p{i}.jpg")
        self.add_file("clip.mp4", media_type="video")
        canonical = self.add_file("orig.jpg")
        self.add_file("copy.jpg", dup_of=canonical)
        self.add_file("broken.jpg", error="cannot read")
        self.add_event(canonical)
        self.start_server()
        collection = self.overview()["collection"]
        self.assertEqual(collection["files"], self.scalar("SELECT COUNT(*) FROM files"))
        self.assertEqual(collection["photos"], self.scalar(
            "SELECT COUNT(*) FROM files WHERE media_type <> 'video'"))
        self.assertEqual(collection["videos"], self.scalar(
            "SELECT COUNT(*) FROM files WHERE media_type = 'video'"))
        self.assertEqual(collection["duplicates"], self.scalar(
            "SELECT COUNT(*) FROM files WHERE dup_of IS NOT NULL"))
        self.assertEqual(collection["errors"], self.scalar(
            "SELECT COUNT(*) FROM files WHERE error IS NOT NULL"))
        self.assertEqual(collection["events"], self.scalar("SELECT COUNT(*) FROM events"))
        self.assertEqual(collection, {"files": 7, "photos": 6, "videos": 1,
                                      "duplicates": 1, "errors": 1, "events": 1,
                                      "animals": 0, "blurred": 0, "eyes_closed": 0,
                                      "no_subject": 0})

    def test_photos_and_videos_add_up_to_the_whole_index(self):
        self.add_file("a.jpg")
        self.add_file("b.raw", media_type="raw")
        self.add_file("c.mp4", media_type="video")
        self.start_server()
        collection = self.overview()["collection"]
        self.assertEqual(collection["photos"] + collection["videos"], collection["files"])


class TestPlaceCounts(OverviewTestBase):
    def test_the_groups_match_direct_sql_and_add_up_to_the_population(self):
        for confidence in ("exact_gps", "exact_gps", "session_inferred",
                           "trip_inferred", "path_inferred", "visual"):
            self.set_place(self.add_file(f"{confidence}{self._n}.jpg"),
                           confidence=confidence)
        self.add_file("nowhere.jpg")  # no places row at all
        self.start_server()
        place = self.overview()["place"]
        expected = {r["confidence"]: r["n"] for r in self.conn.execute(
            "SELECT confidence, COUNT(*) AS n FROM places GROUP BY confidence")}
        got = {row["key"]: row["count"] for row in place["confidence"]}
        self.assertEqual({k: v for k, v in got.items() if k != "unknown"}, expected)
        self.assertEqual(got["unknown"], 1)
        self.assertEqual(sum(got.values()), place["total"])
        self.assertEqual(place["total"], self.scalar(
            "SELECT COUNT(*) FROM files WHERE dup_of IS NULL AND error IS NULL"))

    def test_the_groups_come_in_a_stable_order_from_exact_to_unknown(self):
        for confidence in ("visual", "session_inferred", "exact_gps"):
            self.set_place(self.add_file(f"{confidence}.jpg"), confidence=confidence)
        self.start_server()
        keys = [row["key"] for row in self.overview()["place"]["confidence"]]
        self.assertEqual(keys, ["exact_gps", "session_inferred", "visual"])

    def test_no_place_counts_the_frames_the_layout_sends_to_the_no_place_folder(self):
        self.set_place(self.add_file("placed.jpg"))
        self.add_file("no_row.jpg")
        self.set_place(self.add_file("unknown.jpg"), confidence="unknown",
                       country=None, city=None)
        self.set_place(self.add_file("empty_place.jpg"), confidence="visual",
                       country=None, city=None)
        self.start_server()
        place = self.overview()["place"]
        self.assertEqual(place["no_place"], 3)
        self.assertEqual(place["total"], 4)
        self.assertEqual(place["no_place_percent"], 75.0)

    def test_a_place_the_user_set_by_hand_is_not_placeless(self):
        # The sorter prefers `manual_places` over `places` as a whole (F85c) — a frame
        # placed by hand must not be counted here as one that will land in "no place".
        by_hand = self.add_file("manual.jpg")
        self.set_manual_place(by_hand)
        self.start_server()
        place = self.overview()["place"]
        self.assertEqual(place["no_place"], 0)
        self.assertEqual({row["key"]: row["count"] for row in place["confidence"]},
                         {"manual": 1})

    def test_a_manual_place_outranks_the_automatic_one_in_the_groups(self):
        both = self.add_file("both.jpg")
        self.set_place(both, confidence="visual")
        self.set_manual_place(both)
        self.start_server()
        self.assertEqual(
            {row["key"]: row["count"] for row in self.overview()["place"]["confidence"]},
            {"manual": 1})

    def test_duplicates_and_unreadable_files_are_outside_the_place_population(self):
        canonical = self.add_file("orig.jpg")
        self.set_place(canonical)
        self.add_file("copy.jpg", dup_of=canonical)
        self.add_file("broken.jpg", error="cannot read")
        self.start_server()
        place = self.overview()["place"]
        self.assertEqual(place["total"], 1)
        self.assertEqual(place["no_place"], 0)

    def test_the_percentage_is_rounded_to_one_digit(self):
        for i in range(3):
            self.set_place(self.add_file(f"p{i}.jpg"))
        self.add_file("nowhere.jpg")
        self.start_server()
        self.assertEqual(self.overview()["place"]["no_place_percent"], 25.0)


class TestClassesBreakdown(OverviewTestBase):
    def fill(self) -> None:
        self.classify(self.add_file("a.jpg"), "photo", source="clip", tier="clip")
        self.classify(self.add_file("b.jpg"), "photo", source="heuristic",
                      tier="heuristic")
        self.classify(self.add_file("c.jpg"), "product", source="vlm", tier="vlm")
        self.classify(self.add_file("d.jpg"), "document", source="ocr", tier="vlm")
        self.classify(self.add_file("e.jpg"), "screenshot", source="heuristic",
                      tier=None)
        self.classify(self.add_file("f.jpg"), "meme", source="clip", tier="clip")

    def test_the_verdicts_match_direct_sql(self):
        self.fill()
        self.start_server()
        classes = self.overview()["classes"]
        expected = {r["verdict"]: r["n"] for r in self.conn.execute(
            "SELECT verdict, COUNT(*) AS n FROM media_class GROUP BY verdict")}
        self.assertEqual({row["key"]: row["count"] for row in classes["verdicts"]},
                         expected)

    def test_every_breakdown_sums_to_the_number_of_classified_rows(self):
        self.fill()
        self.start_server()
        classes = self.overview()["classes"]
        total = self.scalar("SELECT COUNT(*) FROM media_class")
        self.assertEqual(classes["total"], total)
        for key in ("verdicts", "sources", "tiers"):
            with self.subTest(breakdown=key):
                self.assertEqual(sum(row["count"] for row in classes[key]), total)

    def test_a_row_without_a_tier_is_its_own_group_not_a_lost_one(self):
        # `tier` is NULL for rows written before schema v11 — dropping them would make
        # the tier breakdown disagree with the verdict one.
        self.fill()
        self.start_server()
        tiers = {row["key"]: row["count"] for row in self.overview()["classes"]["tiers"]}
        self.assertEqual(tiers[None], 1)
        self.assertEqual(tiers["vlm"], 2)
        self.assertEqual(tiers["clip"], 2)

    def test_the_deep_tier_is_reported_as_having_run(self):
        self.fill()
        self.start_server()
        self.assertTrue(self.overview()["classes"]["vlm_ran"])

    def test_a_collection_the_deep_tier_never_touched_says_so(self):
        self.classify(self.add_file("a.jpg"), "photo", source="clip", tier="clip")
        self.start_server()
        self.assertFalse(self.overview()["classes"]["vlm_ran"])

    def test_a_skipped_frame_still_counts_as_handled_by_the_vlm_tier(self):
        # The v11 distinction: the gate judged it a clear personal photo, so `source`
        # stayed 'clip' while the vlm tier did handle the file.
        self.classify(self.add_file("a.jpg"), "photo", source="clip", tier="vlm")
        self.start_server()
        classes = self.overview()["classes"]
        self.assertTrue(classes["vlm_ran"])
        self.assertEqual({row["key"]: row["count"] for row in classes["sources"]},
                         {"clip": 1})

    def test_the_last_change_is_the_newest_updated_at(self):
        self.classify(self.add_file("a.jpg"), "photo", updated_at="2026-07-01")
        self.classify(self.add_file("b.jpg"), "photo", updated_at="2026-07-28")
        self.start_server()
        self.assertEqual(self.overview()["classes"]["updated_at"], "2026-07-28")

    def test_duplicates_and_unreadable_files_are_outside_the_population(self):
        canonical = self.add_file("orig.jpg")
        self.classify(canonical, "photo")
        copy = self.add_file("copy.jpg", dup_of=canonical)
        self.classify(copy, "product")
        broken = self.add_file("broken.jpg", error="cannot read")
        self.classify(broken, "document")
        self.start_server()
        classes = self.overview()["classes"]
        self.assertEqual(classes["total"], 1)
        self.assertEqual({row["key"]: row["count"] for row in classes["verdicts"]},
                         {"photo": 1})
        self.assertEqual(sum(row["count"] for row in classes["tiers"]), classes["total"])

    def test_an_unclassified_index_answers_with_empty_breakdowns(self):
        self.add_file("a.jpg")
        self.start_server()
        classes = self.overview()["classes"]
        self.assertEqual(classes["total"], 0)
        self.assertEqual(classes["sources"], [])
        self.assertEqual(classes["tiers"], [])
        self.assertIsNone(classes["updated_at"])


class TestLayoutGroup(OverviewTestBase):
    def test_without_a_single_batch_there_is_nothing_to_describe(self):
        self.add_file("a.jpg")
        self.start_server()
        self.assertEqual(self.overview()["layout"],
                         {"batches": 0, "unfinished": 0, "last": None})

    def test_the_last_batch_says_when_where_how_and_how_many(self):
        file_id = self.add_file("a.jpg")
        batch = self.add_batch(mode="city", dest_root=str(self.root / "sorted"),
                               started_at="2026-07-20T09:00:00",
                               finished_at="2026-07-20T09:40:00", operation="copy")
        self.add_move(batch, file_id, status="done")
        self.add_move(batch, self.add_file("b.jpg"), status="planned")
        self.start_server()
        last = self.overview()["layout"]["last"]
        self.assertEqual(last["mode"], "city")
        self.assertEqual(last["operation"], "copy")
        self.assertEqual(last["dest_root"], str(self.root / "sorted"))
        self.assertEqual(last["started_at"], "2026-07-20T09:00:00")
        self.assertEqual(last["finished_at"], "2026-07-20T09:40:00")
        self.assertFalse(last["unfinished"])
        self.assertEqual(last["files"], 2)
        self.assertEqual(last["done"], 1)

    def test_the_newest_batch_is_the_one_described(self):
        old = self.add_batch(mode="city", started_at="2026-07-01T10:00:00")
        self.add_move(old, self.add_file("a.jpg"))
        new = self.add_batch(mode="event", started_at="2026-07-20T10:00:00")
        self.add_move(new, self.add_file("b.jpg"))
        self.add_move(new, self.add_file("c.jpg"))
        self.start_server()
        layout = self.overview()["layout"]
        self.assertEqual(layout["batches"], 2)
        self.assertEqual(layout["last"]["mode"], "event")
        self.assertEqual(layout["last"]["files"], 2)

    def test_an_unfinished_batch_is_flagged_on_its_own(self):
        # `finished_at IS NULL` is the trace of an interrupted run — it gets a flag of
        # its own instead of a batch that merely looks like every other one.
        batch = self.add_batch(finished_at=None)
        self.add_move(batch, self.add_file("a.jpg"), status="planned")
        self.start_server()
        layout = self.overview()["layout"]
        self.assertEqual(layout["unfinished"], 1)
        self.assertTrue(layout["last"]["unfinished"])
        self.assertIsNone(layout["last"]["finished_at"])

    def test_an_older_unfinished_batch_is_still_counted(self):
        self.add_batch(started_at="2026-07-01T10:00:00", finished_at=None)
        self.add_batch(started_at="2026-07-20T10:00:00")
        self.start_server()
        layout = self.overview()["layout"]
        self.assertEqual(layout["unfinished"], 1)
        self.assertFalse(layout["last"]["unfinished"])

    def test_a_batch_without_moves_is_zero_files_not_a_crash(self):
        self.add_batch()
        self.start_server()
        last = self.overview()["layout"]["last"]
        self.assertEqual(last["files"], 0)
        self.assertEqual(last["done"], 0)


class TestPrivacy(OverviewTestBase):
    def _keys(self, node: object) -> set[str]:
        if isinstance(node, dict):
            found = set(node)
            for value in node.values():
                found |= self._keys(value)
            return found
        if isinstance(node, list):
            found: set[str] = set()
            for value in node:
                found |= self._keys(value)
            return found
        return set()

    def test_the_answer_carries_no_file_path_and_no_id(self):
        file_id = self.add_file("passport.jpg")
        self.set_place(file_id)
        self.classify(file_id, "document", source="vlm", tier="vlm")
        batch = self.add_batch()
        self.add_move(batch, file_id)
        self.add_event(file_id)
        self.start_server()
        _status, body, _ctype = self.get("/api/overview")
        self.assertNotIn(b"passport.jpg", body)
        self.assertNotIn(str(self.src_dir).encode("utf-8"), body)
        keys = self._keys(json.loads(body))
        self.assertNotIn("id", keys)
        self.assertNotIn("file_id", keys)
        self.assertNotIn("path", keys)

    def test_no_preview_route_is_offered_for_anything(self):
        # There are no thumbnails in the overview at all — documents included.
        file_id = self.add_file("passport.jpg")
        self.classify(file_id, "document", source="vlm", tier="vlm")
        self.start_server()
        _status, body, _ctype = self.get("/api/overview")
        self.assertNotIn(b"/thumb/", body)
        self.assertNotIn(b"/preview/", body)
        self.assertNotIn(b"/photo/", body)


class TestNoPlanIsBuilt(OverviewTestBase):
    def test_opening_the_overview_never_builds_a_plan(self):
        """The acceptance criterion: the tab opens instantly on 24k frames.

        A plan of that collection takes minutes, so the endpoint must not touch the plan
        cache at all. The spy also runs against `/api/plan` afterwards — otherwise a
        probe that silently stopped catching anything would pass forever.
        """
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        built: list[str] = []
        original = ui.PlanCache._build

        def spy(cache, cfg, mode):
            built.append(mode)
            return original(cache, cfg, mode)

        with mock.patch.object(ui.PlanCache, "_build", spy):
            status, _body, _ctype = self.get("/api/overview")
            self.assertEqual(status, 200)
            self.assertEqual(built, [])
            status, _body, _ctype = self.get("/api/plan?mode=city")
            self.assertEqual(status, 200)
            self.assertEqual(built, ["city"])

    def test_the_numbers_are_not_cached_between_requests(self):
        # The tab is opened right AFTER a run to see what changed: a number that is one
        # run out of date is worse than a missing one.
        self.add_file("a.jpg")
        self.start_server()
        self.assertEqual(self.overview()["collection"]["files"], 1)
        self.add_file("b.jpg")
        self.assertEqual(self.overview()["collection"]["files"], 2)


class TestOverviewMarkup(OverviewTestBase):
    def setUp(self):
        super().setUp()
        self.start_server()
        _status, body, _ctype = self.get("/")
        self.html = body.decode("utf-8")

    def test_the_tab_and_its_panel_exist(self):
        self.assertIn('id="tab-btn-overview"', self.html)
        self.assertIn('id="tab-overview"', self.html)
        self.assertIn('id="overview-body"', self.html)

    def test_the_overview_is_the_first_tab(self):
        self.assertLess(self.html.index('id="tab-btn-overview"'),
                        self.html.index('id="tab-btn-review"'))

    def test_it_is_the_tab_the_page_opens_on(self):
        # F133: the run merged into this tab, so there is no longer a second landing
        # place to switch away from — "Overview" is simply the active tab of the markup.
        self.assertIn('class="tab-btn active" id="tab-btn-overview"', self.html)
        self.assertIn('<section id="tab-overview" class="tab-panel active">', self.html)

    def test_the_view_reads_its_own_route(self):
        self.assertIn('fetch("/api/overview")', self.html)

    def test_the_numbers_lead_to_their_tabs(self):
        self.assertIn('overviewCount(c.duplicates, "review", "dupes")', self.html)
        # F133: events, animals and the classifier's classes are slices of one tab now,
        # and a number leads to its own slice rather than to the tab holding it.
        self.assertIn('overviewCount(c.events, "slices", "event")', self.html)
        self.assertIn('overviewCount(c.animals, "slices", "animal")', self.html)
        self.assertIn('row.key === "photo" ? null : "slices"', self.html)
        self.assertIn('"junk:" + row.key', self.html)

    def test_an_empty_index_gets_the_same_rows_with_dashes(self):
        """F145: the block holds its height from the first paint.

        It used to draw an invitation with a button instead, and swap it for the full
        set of counters the moment the index stopped being empty — which is in the
        middle of a run, right after the `index` stage. The block below it, the run
        options among them, moved down the page while a person was reading them.
        """
        self.assertIn("overviewEmpty = !!data.empty", self.html)
        self.assertIn('if (overviewEmpty) return "\\u2014"', self.html)
        self.assertIn("if (overviewEmpty) body.appendChild("
                      "overviewNote(I18N.overview_empty))", self.html)
        # The stub and its button are gone: the run button is on this screen anyway.
        self.assertNotIn("overview-start-btn", self.html)
        self.assertNotIn("overview_empty_button", self.html)

    def test_the_numbers_are_refetched_on_every_open(self):
        self.assertIn('if (name === "overview") loadOverview();', self.html)

    def test_no_external_resources_added(self):
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)
        self.assertNotIn("<link", self.html)


class TestOverviewStringsAreTranslated(unittest.TestCase):
    KEYS = (
        "tab_overview", "overview_empty",
        "overview_group_collection", "overview_group_place", "overview_group_classes",
        "overview_group_layout", "overview_files", "overview_photos", "overview_videos",
        "overview_duplicates", "overview_errors", "overview_events",
        "overview_place_exact_gps", "overview_place_manual",
        "overview_place_session_inferred", "overview_place_trip_inferred",
        "overview_place_path_inferred", "overview_place_visual",
        "overview_no_place", "overview_no_place_hint", "overview_classified",
        "overview_verdict_photo", "overview_by_source", "overview_by_tier",
        "overview_source_heuristic", "overview_source_clip", "overview_source_ocr",
        "overview_source_vlm", "overview_tier_heuristic", "overview_tier_clip",
        "overview_tier_vlm", "overview_tier_none", "overview_vlm_ran",
        "overview_vlm_not_ran", "overview_updated_at", "overview_not_classified",
        "overview_layout_none", "overview_layout_batches", "overview_layout_started",
        "overview_layout_finished", "overview_layout_dest", "overview_layout_mode",
        "overview_layout_files", "overview_layout_done", "overview_layout_unfinished",
        "overview_op_move", "overview_op_copy", "overview_goto_hint",
        "error_loading_overview",
    )

    def test_every_new_string_exists_in_all_three_languages(self):
        for key in self.KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")

    def test_a_label_exists_for_every_place_group_the_view_can_show(self):
        # The label is looked up as `overview_place_<key>` in JS; a group without a key
        # would render as a raw code among translated ones. `unknown` is deliberately
        # absent — those frames are named by the "no place at all" row instead.
        for key in ui._PLACE_CONFIDENCE_ORDER:
            if key == "unknown":
                continue
            with self.subTest(key=key):
                self.assertIn(f"overview_place_{key}", ui._UI_STRINGS)

    def test_a_label_exists_for_every_source_and_tier(self):
        for source in ("heuristic", "clip", "ocr", "vlm"):
            with self.subTest(source=source):
                self.assertIn(f"overview_source_{source}", ui._UI_STRINGS)
        for tier in ("heuristic", "clip", "vlm"):
            with self.subTest(tier=tier):
                self.assertIn(f"overview_tier_{tier}", ui._UI_STRINGS)

    def test_the_placeholders_survive_translation(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertIn("{at}", ui._UI_STRINGS["overview_updated_at"][lang])
                self.assertIn("{tab}", ui._UI_STRINGS["overview_goto_hint"][lang])

    def test_the_tab_title_is_rendered_in_each_language(self):
        for lang, title in (("ru", "Обзор"), ("en", "Overview"), ("ja", "概要")):
            with self.subTest(lang=lang):
                self.assertIn(f">{title}<", ui._render_index_html(lang))


if __name__ == "__main__":
    unittest.main()
