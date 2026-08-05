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
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
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
        # F152: the three face rows are `null` and not 0 — without a faces run nothing
        # was measured, and a zero would read as "no photograph of yours has a person
        # on it". `faces_reason` is what the view shows in their place.
        self.assertEqual(data["collection"], {"files": 0, "photos": 0, "videos": 0,
                                              "duplicates": 0, "errors": 0, "events": 0,
                                              "animals": 0, "with_people": None,
                                              "group_photos": None, "portraits": None,
                                              "faces_reason": "no_faces_run",
                                              # F150 added the low-resolution counter; the
                                              # assertion pins the WHOLE set on purpose, so
                                              # a new counter has to be admitted here — that
                                              # is the test doing its job, not breaking.
                                              "blurred": 0, "eyes_closed": 0,
                                              "low_resolution": 0})
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
                                      "animals": 0, "with_people": None,
                                      "group_photos": None, "portraits": None,
                                      "faces_reason": "no_faces_run",
                                      "blurred": 0, "eyes_closed": 0,
                                      "low_resolution": 0})

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


def _js_function(html: str, name: str) -> str:
    """The source of one JS function of the page, up to its closing brace.

    The view lives in `sorta/web/app/app.js` and is served inside the page; asserting on
    the rendered source is how the other UI tests reach it.
    """
    start = html.index("function " + name + "(")
    depth = 0
    for j in range(html.index("{", start), len(html)):
        if html[j] == "{":
            depth += 1
        elif html[j] == "}":
            depth -= 1
            if depth == 0:
                return html[start:j + 1]
    raise AssertionError(f"the body of {name} does not close")


# --- F190: the two states, rendered and compared node by node --------------------------
#
# "The same containers" is a statement about the tree the browser ends up with, and the
# rest of this file can only reach the source that builds it. So the overview builders are
# run for real — in node, against a stub of the handful of DOM calls they make — and the
# two trees are compared here. Node is not a dependency of the project: where it is
# missing the test says so and skips, and the source-level tests above stay the floor.
_NODE = shutil.which("node")

_PROBE_JS = r"""
'use strict';
// Renders the Overview tab in a stub DOM: the loading skeleton, then two answers from
// the server. Prints the three trees as JSON.
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const start = src.indexOf("var overviewEmpty = false;");
const end = src.indexOf("var vlmAvailable = true;");
if (start < 0 || end < 0) throw new Error("the overview region is not where it was");

function El(tag) {
  this.tagName = tag; this.className = ""; this._text = ""; this.children = [];
}
Object.defineProperty(El.prototype, "textContent", {
  get() { return this._text; },
  // The one call the renderers make to empty the tab before drawing it.
  set(v) { this._text = v; if (v === "") this.children = []; },
});
El.prototype.appendChild = function (c) { this.children.push(c); return c; };
El.prototype.setAttribute = function () {};
El.prototype.addEventListener = function () {};

const body = new El("div");
const document = {
  createElement: (t) => new El(t),
  getElementById: (id) => (id === "overview-body" ? body : null),
};
// Every caption resolves to its own key: this test is about the tree, not the words.
const I18N = new Proxy({}, { get: (_, k) => String(k) });
const fmt = (t, v) => String(t).replace(/\{(\w+)\}/g, (_, k) => v[k]);
const junkBucketLabel = (v) => "junk_bucket_" + v;
function stateEl(kind, text) {
  const d = new El("div"); d.className = "state-msg state-" + kind; d.textContent = text;
  return d;
}
const activateTab = () => {}, gotoSlice = () => {}, selectReviewSlice = () => {};
const fetch = () => { throw new Error("the probe does not run loadOverview"); };

const api = eval(src.slice(start, end) +
                 "\n;({renderOverview, renderOverviewSkeleton});");

function tree(el) {
  return { tag: el.tagName, cls: el.className,
           children: el.children.map(tree) };
}
function snapshot() { return body.children.map(tree); }

const payloads = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
api.renderOverviewSkeleton();
const out = { skeleton: snapshot() };
for (const name of Object.keys(payloads)) {
  api.renderOverview(payloads[name]);
  out[name] = snapshot();
}
console.log(JSON.stringify(out));
"""


def _rows(keys: list[str]) -> list[dict]:
    return [{"key": key, "count": 12} for key in keys]


# What a collection that has been through a full run answers: the lengths the skeleton is
# cut to. The values are irrelevant to a tree of containers, so they are round numbers.
_TYPICAL_PAYLOAD = {
    "empty": False,
    "collection": {"files": 24000, "photos": 23000, "videos": 1000, "duplicates": 40,
                   "errors": 2, "events": 30, "animals": 12, "with_people": 900,
                   "group_photos": 100, "portraits": 80, "faces_reason": None,
                   "blurred": 5, "eyes_closed": 3, "low_resolution": 7},
    "place": {"total": 24000, "no_place": 300, "no_place_percent": 1.2,
              "confidence": _rows(["exact_gps", "session_inferred", "trip_inferred",
                                   "path_inferred"])},
    "classes": {"total": 20000,
                "verdicts": _rows(["photo", "screenshot", "document", "meme", "product"]),
                "sources": _rows(["heuristic", "clip"]),
                "tiers": _rows(["heuristic", "clip"]),
                "vlm_ran": True, "updated_at": "2026-08-05"},
    "layout": {"batches": 1, "unfinished": 0,
               "last": {"mode": "city", "operation": "move", "dest_root": "D:/Sorted",
                        "started_at": "2026-08-04", "finished_at": "2026-08-04",
                        "unfinished": False, "files": 24000, "done": 24000}},
}

# The other extreme: an index that has been through nothing at all. Its variable lists are
# empty and its layout card is a single line — the fixed backbone is what still has to
# match the skeleton row for row.
_BARE_PAYLOAD = {
    "empty": False,
    "collection": dict(_TYPICAL_PAYLOAD["collection"], with_people=None,
                       group_photos=None, portraits=None, faces_reason="no_faces_run"),
    "place": {"total": 3, "no_place": 3, "no_place_percent": 100.0, "confidence": []},
    "classes": {"total": 0, "verdicts": [], "sources": [], "tiers": [], "vlm_ran": False,
                "updated_at": None},
    "layout": {"batches": 0, "unfinished": 0, "last": None},
}


class TestOverviewSkeletonIsTheSameTree(unittest.TestCase):
    """F190, the main test: the loading markup holds the SAME containers as the loaded
    one — checked on the composition of the nodes, not on a picture of them."""

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        if _NODE is None:
            raise unittest.SkipTest("node is not installed — the source-level tests stand")
        app_js = Path(ui.__file__).resolve().parent.parent / "web" / "app" / "app.js"
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.js"
            probe.write_text(_PROBE_JS, encoding="utf-8")
            payloads = Path(tmp) / "payloads.json"
            payloads.write_text(
                json.dumps({"typical": _TYPICAL_PAYLOAD, "bare": _BARE_PAYLOAD}),
                encoding="utf-8")
            done = subprocess.run(
                [_NODE, str(probe), str(app_js), str(payloads)],
                capture_output=True, text=True, encoding="utf-8")
        if done.returncode != 0:
            raise AssertionError("the overview did not render:\n" + done.stderr)
        cls.trees = json.loads(done.stdout)

    def normalised(self, node: dict) -> dict:
        """The tree as the LAYOUT sees it.

        Three differences are not differences of structure: the marker class of the
        skeleton, the indicator that floats over it out of flow, and a value that is a
        link (a number with a tab of its own) rather than a plain cell — the skeleton has
        no number, so it cannot know which of the two a row will get.
        """
        cls = (node["cls"].replace(" overview-skeleton", "")
               .replace(" overview-blank", "").replace("overview-value-link", "overview-value"))
        tag = "span" if cls == "overview-value" else node["tag"]
        return {"tag": tag, "cls": cls,
                "children": [self.normalised(c) for c in node["children"]
                             if "state-msg" not in c["cls"]]}

    def state(self, name: str) -> list[dict]:
        return [self.normalised(node) for node in self.trees[name]]

    def test_a_typical_collection_arrives_into_the_tree_that_was_waiting_for_it(self):
        self.assertEqual(self.state("skeleton"), self.state("typical"))

    def rows_of_the_collection_card(self, name: str) -> list[dict]:
        card = self.state(name)[0]["children"][0]
        return [c for c in card["children"] if "overview-row" in c["cls"]]

    def test_the_fixed_backbone_matches_whatever_arrives(self):
        """The collection card has the same thirteen rows for every possible answer, and
        it is the tallest of the four — the one that sets the height of the whole area on
        a screen wide enough to put the cards side by side. Rows, not the whole card: a
        collection whose faces stage never ran carries one note more, saying so.
        """
        skeleton = self.rows_of_the_collection_card("skeleton")
        self.assertEqual(len(skeleton), 13)
        for name in ("typical", "bare"):
            with self.subTest(payload=name):
                self.assertEqual(skeleton, self.rows_of_the_collection_card(name))

    def test_the_indicator_is_inside_the_area_and_out_of_flow(self):
        """Test 3, from the tree's side: the word "loading" is a child of the grid the
        cards are in — not a block standing where they will be."""
        groups = self.trees["skeleton"][0]
        self.assertIn("overview-skeleton", groups["cls"])
        indicators = [c for c in groups["children"] if "state-msg" in c["cls"]]
        self.assertEqual(len(indicators), 1)
        self.assertIn("state-loading", indicators[0]["cls"])
        self.assertEqual(len(self.trees["skeleton"]), 1,
                         "the loading state is one area and nothing beside it")


class TestOverviewSkeleton(OverviewTestBase):
    """F190: the area takes its final size BEFORE the data arrives.

    F145 stopped the tab from changing height between an empty index and a full one; the
    request itself still changed it. Opening the tab painted a one-line "loading" message
    and replaced it with four cards — a block of an entirely different height — so
    everything below, the run options among them, moved down the page under the cursor of
    somebody who was already aiming at them.

    The fix is structural rather than cosmetic, and so are the tests: the loading state is
    built by the SAME function from a constant stand-in, which is what makes "the same
    containers" a property of the code instead of two lists kept in step by hand.
    """

    def setUp(self):
        super().setUp()
        self.start_server()
        _status, body, _ctype = self.get("/")
        self.html = body.decode("utf-8")

    def test_both_states_are_built_by_the_same_builder(self):
        """Test 1: the loading markup holds the same containers as the loaded one.

        Not by comparing pictures — by the fact that one function builds both. Neither
        renderer may assemble a card of its own, or the two would drift apart the day
        somebody adds a fifth group.
        """
        groups = _js_function(self.html, "overviewGroups")
        for card in ("overviewCollectionCard", "overviewPlaceCard", "overviewClassesCard",
                     "overviewLayoutCard"):
            with self.subTest(card=card):
                self.assertIn(card + "(data)", groups)
        for renderer in ("renderOverview", "renderOverviewSkeleton"):
            with self.subTest(renderer=renderer):
                body = _js_function(self.html, renderer)
                self.assertIn("overviewGroups(", body)
                self.assertNotIn("overviewCard(", body)
                for card in ("overviewCollectionCard", "overviewPlaceCard",
                             "overviewClassesCard", "overviewLayoutCard"):
                    self.assertNotIn(card, body)

    def test_the_skeleton_is_the_same_size_whatever_arrives(self):
        """Test 2: the number of elements does not depend on how much data comes.

        The renderer takes no argument at all, which is the strongest form of that: there
        is nothing for a count to depend on. The row counts of the lists are constants.
        """
        body = _js_function(self.html, "renderOverviewSkeleton")
        self.assertIn("function renderOverviewSkeleton()", body)
        self.assertNotIn("(data", body)
        self.assertNotIn("data.", body)
        self.assertIn("overviewSkeletonData()", body)
        rows = _js_function(self.html, "overviewSkeletonData")
        for group in ("place", "verdicts", "sources", "tiers"):
            with self.subTest(group=group):
                self.assertIn("OVERVIEW_SKELETON_ROWS." + group, rows)

    def test_every_field_a_card_reads_is_in_the_stand_in(self):
        """The stand-in has to choose the same branches the real payload chooses.

        A field it forgets arrives as `undefined` — a face count would dash, a layout
        card would collapse to a single line — and the height stops matching. The check
        is mechanical so that a row added tomorrow cannot quietly skip it.
        """
        stand_in = _js_function(self.html, "overviewSkeletonData")
        for card, var in (("overviewCollectionCard", "c"), ("overviewPlaceCard", "p"),
                          ("overviewClassesCard", "cl"), ("overviewLayoutCard", "lay")):
            body = _js_function(self.html, card)
            for field in sorted(set(re.findall(rf"\b{var}\.([a-z_]+)", body))):
                with self.subTest(card=card, field=field):
                    self.assertIn(field + ":", stand_in)
        # `lay.last` is an object of its own, and the rows are read off it.
        for field in sorted(set(re.findall(r"\blast\.([a-z_]+)",
                                           _js_function(self.html, "overviewLayoutCard")))):
            with self.subTest(field=field):
                self.assertIn(field + ":", stand_in)

    def test_the_word_loading_stands_inside_the_reserved_area(self):
        """Test 3: inside the area, not instead of it.

        The indicator is appended to the grid the cards are in — and it is positioned out
        of flow, so that it can go without taking a line with it. Before, it was the only
        thing in the tab body while the request was in the air.
        """
        body = _js_function(self.html, "renderOverviewSkeleton")
        self.assertIn('groups.appendChild(stateEl("loading", I18N.overview_loading))', body)
        self.assertNotIn("body.appendChild(stateEl", body)
        self.assertIn('groups.setAttribute("aria-busy", "true")', body)
        load = _js_function(self.html, "loadOverview")
        self.assertIn("renderOverviewSkeleton()", load)
        self.assertNotIn('stateEl("loading"', load)
        self.assertIn(".overview-skeleton > .state-msg { position: absolute;", self.html)

    def test_the_skeleton_invents_nothing(self):
        """An empty card is an empty card. A plausible number that changes a second later
        is worse than a dash — it gets read."""
        self.assertIn('el.textContent = overviewLoading ? "" : text;',
                      _js_function(self.html, "overviewValue"))
        for builder in ("overviewCount", "overviewFaceCount"):
            with self.subTest(builder=builder):
                self.assertIn('if (overviewLoading) return overviewValue("");',
                              _js_function(self.html, builder))
        # The labels the DATA names — a place group, a class, a source, a tier — are
        # unknown until it arrives, and a tier row would otherwise read "Tier not
        # recorded" about a collection nobody has looked at yet.
        for label in ("overviewPlaceLabel(row.key)", "overviewVerdictLabel(row.key)",
                      "overviewSourceLabel(row.key)", "overviewTierLabel(row.key)"):
            with self.subTest(label=label):
                self.assertIn("overviewDataText(" + label + ")", self.html)
        self.assertIn('return overviewLoading ? "" : text;',
                      _js_function(self.html, "overviewDataText"))

    def test_a_blank_cell_keeps_the_line_it_is_waiting_for(self):
        """The reserved size is the size of the text, not a guess at it: the blank takes
        its line box from a `::before` non-breaking space in its own font."""
        self.assertIn(".overview-blank {", self.html)
        self.assertIn('.overview-blank::before { content: "\\00a0"; }', self.html)
        self.assertIn('" overview-blank"', _js_function(self.html, "overviewValue"))
        self.assertIn('" overview-blank"', _js_function(self.html, "overviewRow"))
        self.assertIn('" overview-blank"', _js_function(self.html, "overviewNote"))

    def test_the_indicator_is_not_removed(self):
        """It is needed: a grid of empty cells with nothing said over it reads as
        "there is nothing here"."""
        for lang, expected in (("ru", "Загрузка обзора…"), ("en", "Loading the overview…"),
                               ("ja", "概要を読み込み中…")):
            with self.subTest(lang=lang):
                _status, body, _ctype = self.get(f"/?lang={lang}")
                self.assertIn(expected, body.decode("utf-8"))


class TestOverviewStringsAreTranslated(unittest.TestCase):
    KEYS = (
        "tab_overview", "overview_empty", "overview_loading",
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
