"""F70: the UI on large collections — a lazy PlanCache and a paged /api/plan.

Before F70 `sorta ui` built all three plan modes synchronously at startup (~40 s on a
26k collection, and again on every rebuild) and served the whole plan as one JSON
(8.6 MB / 26 445 items). These tests pin the two properties that fixed it: nothing is
built until a mode is actually asked for, and no single request can pull the whole
plan out of the server.
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sorta import ui
from sorta.config import Config
from sorta.db import connect
from sorta.sorter import PlanItem

from tests.test_ui import UiServerTestBase


def _plan_item(file_id: int, target_rel: str) -> PlanItem:
    """A minimal PlanItem — only the fields /api/plan actually reads."""
    return PlanItem(
        file_id=file_id, src=Path(f"/src/{file_id}.jpg"),
        dst=Path("/dst") / target_rel, in_place=False,
        target_rel=target_rel, reason="city",
        taken_at="2022-05-01T10:00:00", taken_at_confidence="high",
        country="ru", city="Moscow", place_confidence="exact_gps",
        gps_lat=None, gps_lon=None, persons=[], event=None,
        junk_verdict=None, junk_source=None, db_hash=None, db_algo=None,
    )


def _synthetic_plan(count: int, folders: int) -> list[PlanItem]:
    """`count` items spread evenly over `folders` target folders."""
    return [_plan_item(i, f"Россия/Москва/2022/{i % folders:04d}/img{i}.jpg")
            for i in range(1, count + 1)]


class PlanCacheTestBase(unittest.TestCase):
    """A PlanCache over a real (tiny) sqlite file, with plan_and_sort mocked out.

    The mock is what makes "was anything built?" observable — the whole point of the
    laziness tests is the CALL COUNT, not the plan contents.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = Config(sources=[self.root / "src"],
                          database=self.root / "test.db", raw={})
        self.conn = connect(self.cfg.database)
        self.dest = self.root / "_sorta_ui_preview"
        self.calls: list[str] = []
        self.plans: dict[str, list[PlanItem]] = {}

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def add_file_row(self, file_id: int, size: int) -> None:
        self.conn.execute(
            """INSERT INTO files (id, path, size, mtime, ext, media_type, indexed_at)
               VALUES (?, ?, ?, 0, 'jpg', 'photo', '2026-01-01')""",
            (file_id, f"/src/{file_id}.jpg", size),
        )
        self.conn.commit()

    def fake_plan_and_sort(self, cfg, conn, mode, dest, **kwargs):
        self.calls.append(mode)
        return SimpleNamespace(plan=list(self.plans.get(mode, [])))

    def patched_cache(self) -> tuple[ui.PlanCache, mock._patch]:
        patcher = mock.patch.object(ui, "plan_and_sort", self.fake_plan_and_sort)
        patcher.start()
        self.addCleanup(patcher.stop)
        return ui.PlanCache(self.cfg, self.conn, self.dest), patcher


class TestPlanCacheLaziness(PlanCacheTestBase):
    def test_construction_builds_nothing(self):
        cache, _p = self.patched_cache()
        self.assertEqual(self.calls, [])
        self.assertIsNotNone(cache)

    def test_first_get_builds_exactly_one_mode(self):
        cache, _p = self.patched_cache()
        cache.get("city")
        self.assertEqual(self.calls, ["city"])

    def test_second_get_of_same_mode_does_not_rebuild(self):
        cache, _p = self.patched_cache()
        cache.get("city")
        cache.get("city")
        cache.aggregate("city")
        cache.page("city", "whatever", 0, 10)
        self.assertEqual(self.calls, ["city"])

    def test_other_mode_builds_separately(self):
        cache, _p = self.patched_cache()
        cache.get("city")
        cache.get("person")
        self.assertEqual(self.calls, ["city", "person"])
        cache.get("person")
        self.assertEqual(self.calls, ["city", "person"])

    def test_unsupported_mode_builds_nothing_and_returns_none(self):
        cache, _p = self.patched_cache()
        self.assertIsNone(cache.get("nonsense"))
        self.assertIsNone(cache.aggregate("nonsense"))
        self.assertIsNone(cache.page("nonsense", "x", 0, 10))
        self.assertEqual(self.calls, [])


class TestPlanCacheRebuildInvalidates(PlanCacheTestBase):
    def test_rebuild_does_not_build_but_next_get_does(self):
        cache, _p = self.patched_cache()
        cache.get("city")
        self.assertEqual(self.calls, ["city"])

        cache.rebuild(self.cfg, self.conn)
        self.assertEqual(self.calls, ["city"])  # invalidation only, no work

        cache.get("city")
        self.assertEqual(self.calls, ["city", "city"])

    def test_rebuild_drops_every_built_mode(self):
        cache, _p = self.patched_cache()
        cache.get("city")
        cache.get("event")
        cache.rebuild(self.cfg, self.conn)
        cache.get("city")
        cache.get("event")
        self.assertEqual(self.calls, ["city", "event", "city", "event"])

    def test_rebuild_picks_up_new_plan_contents(self):
        cache, _p = self.patched_cache()
        self.plans["city"] = _synthetic_plan(3, folders=1)
        self.assertEqual(len(cache.get("city") or []), 3)
        self.plans["city"] = _synthetic_plan(7, folders=1)
        self.assertEqual(len(cache.get("city") or []), 3)  # still cached
        cache.rebuild(self.cfg, self.conn)
        self.assertEqual(len(cache.get("city") or []), 7)


class TestPlanCacheThreadSafety(PlanCacheTestBase):
    def test_eight_threads_get_one_consistent_result(self):
        cache, _p = self.patched_cache()
        self.plans["city"] = _synthetic_plan(50, folders=5)
        results: list[object] = []
        errors: list[BaseException] = []
        start = threading.Event()

        def worker() -> None:
            start.wait(5)
            try:
                results.append(cache.aggregate("city"))
            except BaseException as exc:  # noqa: BLE001 — the test is about "no exception"
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 8)
        first = results[0]
        for other in results[1:]:
            self.assertEqual(first, other)
        self.assertEqual(first["total"], 50)  # type: ignore[index]
        # The per-mode lock must collapse the burst into a single build.
        self.assertEqual(self.calls, ["city"])


class TestPlanAggregate(PlanCacheTestBase):
    def test_aggregate_counts_and_sizes_per_folder(self):
        cache, _p = self.patched_cache()
        self.plans["city"] = [
            _plan_item(1, "Россия/Москва/a.jpg"),
            _plan_item(2, "Россия/Москва/b.jpg"),
            _plan_item(3, "Россия/Питер/c.jpg"),
        ]
        for file_id, size in ((1, 100), (2, 200), (3, 400)):
            self.add_file_row(file_id, size)

        payload = cache.aggregate("city")
        assert payload is not None
        self.assertEqual(payload["mode"], "city")
        self.assertEqual(payload["total"], 3)
        self.assertEqual(payload["categories"], [
            {"category": "Россия/Москва", "count": 2, "size": 300},
            {"category": "Россия/Питер", "count": 1, "size": 400},
        ])

    def test_aggregate_carries_no_file_list(self):
        cache, _p = self.patched_cache()
        self.plans["city"] = _synthetic_plan(200, folders=4)
        payload = cache.aggregate("city")
        assert payload is not None
        text = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("file_id", text)
        self.assertNotIn("thumb_url", text)
        self.assertNotIn("img1.jpg", text)

    def test_size_of_unindexed_file_counts_as_zero(self):
        # A plan item whose files row vanished must not break the aggregate.
        cache, _p = self.patched_cache()
        self.plans["city"] = [_plan_item(42, "Россия/Москва/a.jpg")]
        payload = cache.aggregate("city")
        assert payload is not None
        self.assertEqual(payload["categories"][0]["size"], 0)


class TestPlanPagination(PlanCacheTestBase):
    def _cache_with(self, count: int, folders: int = 1) -> ui.PlanCache:
        cache, _p = self.patched_cache()
        self.plans["city"] = _synthetic_plan(count, folders=folders)
        return cache

    def test_offset_and_limit_slice_the_category(self):
        cache = self._cache_with(10)
        category = "Россия/Москва/2022/0000"
        page = cache.page("city", category, 3, 4)
        assert page is not None
        self.assertEqual(page["total"], 10)
        self.assertEqual(page["offset"], 3)
        self.assertEqual(page["limit"], 4)
        self.assertEqual([it["file_id"] for it in page["items"]], [4, 5, 6, 7])

    def test_offset_past_the_end_is_an_empty_page_with_total(self):
        cache = self._cache_with(10)
        page = cache.page("city", "Россия/Москва/2022/0000", 99, 10)
        assert page is not None
        self.assertEqual(page["items"], [])
        self.assertEqual(page["total"], 10)

    def test_unknown_category_is_an_empty_page_not_an_error(self):
        cache = self._cache_with(10)
        page = cache.page("city", "No/Such/Folder", 0, 10)
        assert page is not None
        self.assertEqual(page["total"], 0)
        self.assertEqual(page["items"], [])


class TestPageWindowParsing(unittest.TestCase):
    def test_defaults_when_absent(self):
        self.assertEqual(ui._parse_page_window({}),
                         (0, ui._PLAN_PAGE_DEFAULT_LIMIT))

    def test_limit_above_maximum_is_clamped(self):
        self.assertEqual(ui._parse_page_window({"limit": ["999999"]}),
                         (0, ui._PLAN_PAGE_MAX_LIMIT))

    def test_garbage_and_negative_values_are_rejected(self):
        for query in ({"limit": ["all"]}, {"limit": ["-1"]}, {"offset": ["-5"]},
                      {"offset": ["1.5"]}, {"offset": [""]}):
            self.assertIsNone(ui._parse_page_window(query), query)

    def test_zero_limit_is_an_empty_page_not_everything(self):
        self.assertEqual(ui._parse_page_window({"limit": ["0"]}), (0, 0))


class TestPlanRouteShape(UiServerTestBase):
    """The same guarantees over HTTP, on a real (small) plan."""

    def plan(self, query: str) -> dict:
        status, body, ctype = self.get("/api/plan?" + query)
        self.assertEqual(status, 200, body)
        self.assertIn("application/json", ctype)
        return json.loads(body)

    def test_aggregate_is_the_default_shape(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.add_photo_file("b.jpg", country="ru", city="Moscow")
        self.start_server()
        data = self.plan("mode=city")
        self.assertEqual(data["total"], 2)
        self.assertEqual(len(data["categories"]), 1)
        row = data["categories"][0]
        self.assertEqual(row["count"], 2)
        self.assertGreater(row["size"], 0)
        self.assertNotIn("items", data)

    def test_page_reports_total_and_respects_limit(self):
        for i in range(5):
            self.add_photo_file(f"f{i}.jpg", country="ru", city="Moscow")
        self.start_server()
        category = self.plan("mode=city")["categories"][0]["category"]
        quoted = urllib.parse.quote(category)
        page = self.plan(f"mode=city&category={quoted}&offset=1&limit=2")
        self.assertEqual(page["total"], 5)
        self.assertEqual(len(page["items"]), 2)
        self.assertEqual(page["category"], category)

    def test_limit_above_maximum_never_serves_everything(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        category = self.plan("mode=city")["categories"][0]["category"]
        quoted = urllib.parse.quote(category)
        page = self.plan(f"mode=city&category={quoted}&limit=100000")
        self.assertEqual(page["limit"], ui._PLAN_PAGE_MAX_LIMIT)

    def test_garbage_limit_is_400_not_a_full_dump(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        category = self.plan("mode=city")["categories"][0]["category"]
        quoted = urllib.parse.quote(category)
        status, body, _ctype = self.get(
            f"/api/plan?mode=city&category={quoted}&limit=everything")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_unknown_category_is_200_with_total_zero(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        page = self.plan("mode=city&category=No%2FSuch%2FFolder")
        self.assertEqual(page["total"], 0)
        self.assertEqual(page["items"], [])

    def test_unsupported_mode_with_category_is_400(self):
        self.start_server()
        status, body, _ctype = self.get("/api/plan?mode=nope&category=x")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))


class TestServerStartBuildsNothing(UiServerTestBase):
    def test_build_server_does_not_plan(self):
        # The acceptance criterion "sorta ui serves the page immediately": starting
        # the server must not call plan_and_sort at all.
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        calls: list[str] = []

        def counting(cfg, conn, mode, dest, **kwargs):
            calls.append(mode)
            return SimpleNamespace(plan=[])

        with mock.patch.object(ui, "plan_and_sort", counting):
            self.start_server()
            status, _body, _ctype = self.get("/")
            self.assertEqual(status, 200)
            self.assertEqual(calls, [])
            # ...and only the requested mode is built afterwards.
            self.get("/api/plan?mode=city")
            self.assertEqual(calls, ["city"])


class TestNoReportSideFiles(PlanCacheTestBase):
    def test_building_a_plan_writes_no_csv_or_html(self):
        # F70 requirement 2: the UI path passes write_reports=False, so opening the
        # web app must not litter report_output/ with a CSV+HTML per mode.
        cache = ui.PlanCache(self.cfg, self.conn, self.dest)
        for mode in ("city", "person", "event"):
            cache.get(mode)
        report_dir = Path(self.cfg.database).resolve().parent / "report_output"
        produced = sorted(p.name for p in report_dir.glob("*")) if report_dir.exists() else []
        self.assertEqual(produced, [])

    def test_write_reports_false_is_passed_through(self):
        seen: list[object] = []

        def spy(cfg, conn, mode, dest, **kwargs):
            seen.append(kwargs.get("write_reports"))
            return SimpleNamespace(plan=[])

        with mock.patch.object(ui, "plan_and_sort", spy):
            ui.PlanCache(self.cfg, self.conn, self.dest).get("city")
        self.assertEqual(seen, [False])


class TestLargeCollectionPayloads(PlanCacheTestBase):
    """The acceptance numbers: the aggregate stays in kilobytes, the page is bounded."""

    def _built(self, count: int, folders: int) -> ui.PlanCache:
        cache, _p = self.patched_cache()
        self.plans["city"] = _synthetic_plan(count, folders=folders)
        cache.get("city")  # warm the cache once, outside the measurements
        return cache

    def test_aggregate_of_5000_files_is_under_100kb(self):
        cache = self._built(5000, folders=120)
        payload = cache.aggregate("city")
        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        self.assertLess(size, 100 * 1024, f"aggregate is {size} bytes")

    def test_aggregate_of_30000_files_is_under_100kb_and_fast(self):
        cache = self._built(30000, folders=400)
        payload = cache.aggregate("city")
        assert payload is not None
        self.assertEqual(payload["total"], 30000)
        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        self.assertLess(size, 100 * 1024, f"aggregate is {size} bytes")

    def test_page_of_200_items_is_under_100kb(self):
        cache = self._built(30000, folders=1)
        page = cache.page("city", "Россия/Москва/2022/0000", 0,
                          ui._PLAN_PAGE_DEFAULT_LIMIT)
        assert page is not None
        self.assertEqual(page["total"], 30000)
        self.assertEqual(len(page["items"]), 200)
        size = len(json.dumps(page, ensure_ascii=False).encode("utf-8"))
        self.assertLess(size, 100 * 1024, f"page is {size} bytes")


class TestPlanTabFrontend(UiServerTestBase):
    """The tab must consume the aggregate and page, not a 26k-element list."""

    def html(self) -> str:
        self.start_server()
        _status, body, _ctype = self.get("/")
        return body.decode("utf-8")

    def test_tree_is_built_from_the_aggregate(self):
        html = self.html()
        self.assertIn("buildCategoryTree", html)
        self.assertIn("data.categories", html)
        self.assertIn("cityPlanDirCount = categories.length", html)
        # the plan tab no longer feeds a full item list into the generic tree
        # builder (which stays for the Moves tab — one bounded batch).
        self.assertIn("buildTree(data.moves)", html)

    def test_category_files_are_fetched_as_pages(self):
        html = self.html()
        self.assertIn("renderCategoryFiles", html)
        self.assertIn("&offset=", html)
        self.assertIn("&limit=", html)
        self.assertIn("PLAN_PAGE_SIZE", html)
        self.assertIn("plan_shown_of", html)
        self.assertIn("plan_load_more", html)

    def test_thumbnail_requests_are_concurrency_limited(self):
        html = self.html()
        self.assertIn("THUMB_CONCURRENCY", html)
        self.assertIn("queueThumb", html)
        self.assertIn("IntersectionObserver", html)


if __name__ == "__main__":
    unittest.main()
