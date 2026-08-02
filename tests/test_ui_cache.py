"""F94: the two caches are reachable from the web app, not only from `sorta cache`.

The program keeps two things that grow and never shrink by themselves: the preview
JPEGs on disk (12 GB on the live collection) and the online geocoder's answers in
`geo_cache`. Both had exactly one way out — a terminal command — while `sorta ui` is
advertised as a full entry point, so for anyone who does not open a terminal the disk
simply filled up with no way to do anything about it.

What is pinned here: the sizes the endpoint reports (an empty cache is zero, not an
error), that both clears actually clear and are idempotent, that neither touches the
other's data (or `places`), that they are refused while a run holds the busy lock and
that the buttons are dead in the interface for the same window, and that every caption
exists in all three languages. The clearing logic itself belongs to
`imaging.preview_cache_clear` / `geo.clear_geo_cache` and is tested with them — this
feature is only the way to reach it.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from sorta import imaging, ui
from sorta.geo import geo_cache_size

from tests.test_ui import UiServerTestBase
from tests.test_ui_process import ProcessTestBase, _poll_until

_F94_STRING_KEYS = (
    "cache_title", "cache_sizes", "cache_hint",
    "cache_clear_preview_button", "cache_clear_geo_button",
    "cache_clear_preview_confirm", "cache_clear_geo_confirm",
    "cache_clear_preview_done", "cache_clear_geo_done",
    "cache_clear_error_prefix",
)


class CacheTestBase(ProcessTestBase):
    """A preview directory of its own per test + fixtures for both caches.

    conftest already points SORTA_PREVIEW_DIR at a shared sandbox; a test that DELETES
    the directory needs its own, or it would wipe the previews other tests are using.
    """

    def setUp(self):
        super().setUp()
        self._previews_tmp = tempfile.TemporaryDirectory()
        self.preview_dir = Path(self._previews_tmp.name) / "previews"
        patcher = mock.patch.dict(
            os.environ, {imaging.ENV_PREVIEW_DIR: str(self.preview_dir)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._previews_tmp.cleanup)

    def add_previews(self, count: int, payload: bytes = b"x" * 1000) -> int:
        """`count` fake preview files, sharded the way imaging writes them."""
        for i in range(count):
            key = f"{i:040x}"
            path = self.preview_dir / key[:2] / f"{key}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        return count * len(payload)

    def add_geo_rows(self, count: int) -> None:
        for i in range(count):
            self.conn.execute(
                """INSERT INTO geo_cache (provider, key, country, city_ru, updated_at)
                   VALUES ('online', ?, 'ru', 'Москва', '2026-07-01T00:00:00')""",
                (f"c:524901/{i}",))
        self.conn.commit()

    def cache(self) -> dict:
        status, body, ctype = self.get("/api/cache")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        return json.loads(body)


class TestCacheSizes(CacheTestBase):
    def test_empty_caches_report_zero_not_an_error(self):
        """A machine that has never run anything has no preview directory at all —
        that is the normal first state of the tab, not a failure to report."""
        self.start_server()
        data = self.cache()
        self.assertFalse(self.preview_dir.exists())
        self.assertEqual(data["preview"]["files"], 0)
        self.assertEqual(data["preview"]["bytes"], 0)
        self.assertEqual(data["geo"]["entries"], 0)

    def test_reports_the_size_of_both_caches(self):
        expected_bytes = self.add_previews(5)
        self.add_geo_rows(3)
        self.start_server()
        data = self.cache()
        self.assertEqual(data["preview"]["files"], 5)
        self.assertEqual(data["preview"]["bytes"], expected_bytes)
        self.assertEqual(data["geo"]["entries"], 3)

    def test_reports_where_the_preview_cache_lives(self):
        self.start_server()
        self.assertEqual(self.cache()["preview"]["dir"], str(self.preview_dir))

    def test_the_numbers_are_the_ones_the_cli_prints(self):
        """`sorta cache` counts the JPEGs and their bytes, and reports the geo cache in
        rows — the interface must not invent a different unit for the same thing."""
        self.add_previews(4)
        self.add_geo_rows(2)
        self.start_server()
        data = self.cache()
        directory = imaging.preview_dir()
        cli_files = sum(1 for _ in directory.rglob("*.jpg"))
        cli_size = sum(f.stat().st_size for f in directory.rglob("*.jpg"))
        self.assertEqual(data["preview"]["files"], cli_files)
        self.assertEqual(data["preview"]["bytes"], cli_size)
        self.assertEqual(data["geo"]["entries"], geo_cache_size(self.conn))


class TestClearPreviewCache(CacheTestBase):
    def test_clear_removes_the_directory_and_reports_what_went(self):
        self.add_previews(6)
        self.start_server()
        status, resp = self.post("/api/cache/clear", {"target": "preview"})
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["target"], "preview")
        self.assertEqual(resp["removed"], 6)
        self.assertFalse(self.preview_dir.exists())

    def test_the_response_carries_the_fresh_sizes(self):
        """The client redraws from this instead of paying for a second walk of a
        directory that holds tens of thousands of files."""
        self.add_previews(6)
        self.add_geo_rows(1)
        self.start_server()
        _status, resp = self.post("/api/cache/clear", {"target": "preview"})
        self.assertEqual(resp["cache"]["preview"]["files"], 0)
        self.assertEqual(resp["cache"]["preview"]["bytes"], 0)
        self.assertEqual(resp["cache"]["geo"]["entries"], 1)

    def test_clearing_twice_is_a_success_not_an_error(self):
        """Idempotent on purpose: the button stays live after the first click, and
        "there was nothing to delete" is not a failure the user has to read about."""
        self.add_previews(2)
        self.start_server()
        first, _resp = self.post("/api/cache/clear", {"target": "preview"})
        self.assertEqual(first, 200)
        second, resp2 = self.post("/api/cache/clear", {"target": "preview"})
        self.assertEqual(second, 200)
        self.assertTrue(resp2["ok"])
        self.assertEqual(resp2["removed"], 0)

    def test_clearing_previews_leaves_the_geo_cache_alone(self):
        self.add_previews(2)
        self.add_geo_rows(3)
        self.start_server()
        self.post("/api/cache/clear", {"target": "preview"})
        self.assertEqual(geo_cache_size(self.conn), 3)


class TestClearGeoCache(CacheTestBase):
    def test_clear_removes_the_rows_and_reports_how_many(self):
        self.add_geo_rows(4)
        self.start_server()
        status, resp = self.post("/api/cache/clear", {"target": "geo"})
        self.assertEqual(status, 200)
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["target"], "geo")
        self.assertEqual(resp["removed"], 4)
        self.assertEqual(geo_cache_size(self.conn), 0)
        self.assertEqual(resp["cache"]["geo"]["entries"], 0)

    def test_clearing_twice_is_a_success_not_an_error(self):
        self.add_geo_rows(1)
        self.start_server()
        self.post("/api/cache/clear", {"target": "geo"})
        status, resp = self.post("/api/cache/clear", {"target": "geo"})
        self.assertEqual(status, 200)
        self.assertEqual(resp["removed"], 0)

    def test_places_survive_the_geo_cache_clear(self):
        """The cache holds what the PROVIDER said; `places` holds what the files got.
        Dropping the first must not un-place a single photo of the collection."""
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.add_geo_rows(2)
        self.start_server()
        self.post("/api/cache/clear", {"target": "geo"})
        rows = self.conn.execute("SELECT country, city FROM places").fetchall()
        self.assertEqual([(r["country"], r["city"]) for r in rows], [("ru", "Moscow")])

    def test_clearing_geo_leaves_the_previews_alone(self):
        self.add_previews(3)
        self.add_geo_rows(1)
        self.start_server()
        self.post("/api/cache/clear", {"target": "geo"})
        self.assertEqual(self.cache()["preview"]["files"], 3)


class TestClearValidation(CacheTestBase):
    def test_an_unknown_target_is_a_400_and_deletes_nothing(self):
        self.add_previews(2)
        self.add_geo_rows(2)
        self.start_server()
        for payload in ({"target": "everything"}, {"target": ""}, {"target": 1},
                        {}, [], "preview"):
            status, _resp = self.post("/api/cache/clear", payload)
            self.assertEqual(status, 400, payload)
        self.assertEqual(self.cache()["preview"]["files"], 2)
        self.assertEqual(geo_cache_size(self.conn), 2)


class TestClearIsRefusedWhileBusy(CacheTestBase):
    def test_409_while_the_pipeline_runs_and_nothing_is_deleted(self):
        """The same busy_lock as "Start over": mid-run the geo clear would send the
        rest of the stage back to the network and the preview clear would delete the
        frames that stage is writing right now."""
        self.add_previews(3)
        self.add_geo_rows(2)
        block = threading.Event()
        self.addCleanup(block.set)
        self.patch_fast_stages(block_stage="geo", block_event=block)
        self.start_server()
        start_status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(start_status, 200)
        _poll_until(self.status, lambda d: d["running"] and "geo" in self.calls)

        for target in ("preview", "geo"):
            status, resp = self.post("/api/cache/clear", {"target": target})
            self.assertEqual(status, 409, target)
            self.assertEqual(resp["error"], "already running")
        self.assertEqual(self.cache()["preview"]["files"], 3)
        self.assertEqual(geo_cache_size(self.conn), 2)

        block.set()
        _poll_until(self.status, lambda d: d["finished"])

    def test_the_clear_works_again_once_the_run_is_over(self):
        self.add_previews(3)
        self.patch_fast_stages()
        self.start_server()
        self.post("/api/process", {"source_dir": str(self.src_dir)})
        _poll_until(self.status, lambda d: d["finished"])
        status, resp = self.post("/api/cache/clear", {"target": "preview"})
        self.assertEqual(status, 200)
        self.assertEqual(resp["removed"], 3)


class TestCacheUiMarkup(UiServerTestBase):
    def setUp(self):
        super().setUp()
        self.html = ui._render_index_html("ru")

    def _body(self, name: str) -> str:
        """Source of a JS function declaration, up to its closing brace."""
        start = self.html.index(f"function {name}(")
        depth = 0
        for j in range(self.html.index("{", start), len(self.html)):
            if self.html[j] == "{":
                depth += 1
            elif self.html[j] == "}":
                depth -= 1
                if depth == 0:
                    return self.html[start:j + 1]
        raise AssertionError(f"не найдено тело {name}")

    def test_the_block_shows_sizes_and_offers_both_clears(self):
        self.assertIn('id="cache-block"', self.html)
        self.assertIn('id="cache-sizes"', self.html)
        self.assertIn('id="cache-clear-preview-btn"', self.html)
        self.assertIn('id="cache-clear-geo-btn"', self.html)
        self.assertIn('fetch("/api/cache")', self.html)
        self.assertIn('postJson("/api/cache/clear", { target: target })', self.html)

    def test_both_clears_ask_before_deleting(self):
        self.assertIn("window.confirm(confirmText)", self._body("clearCache"))
        self.assertIn("I18N.cache_clear_preview_confirm", self.html)
        self.assertIn("I18N.cache_clear_geo_confirm", self.html)

    def test_the_preview_confirmation_states_the_price_of_the_next_run(self):
        """Clearing 12 GB is not free — the next run decodes originals again. The
        numbers are the measured ones (F67), and they belong in the dialog, not in a
        release note nobody reads."""
        for lang in ("ru", "en", "ja"):
            text = ui._t("cache_clear_preview_confirm", lang)
            self.assertIn("336", text, lang)
            self.assertIn("73", text, lang)

    def test_the_buttons_are_dead_while_anything_runs(self):
        body = self._body("updateBusyControlsDisabled")
        self.assertIn("cache-clear-preview-btn", body)
        self.assertIn("cache-clear-geo-btn", body)
        # F145: the three flags moved behind one predicate — a dozen controls on five
        # tabs ask the same question now.
        self.assertIn("var busy = uiBusy();", body)
        self.assertIn("sortRunning || processRunning || undoRunning",
                      self._body("uiBusy"))

    def test_the_size_is_not_asked_for_on_every_status_tick(self):
        """A walk of a directory with tens of thousands of files, once a poll tick,
        would be a directory scan per second for the whole run."""
        self.assertNotIn("/api/cache", self._body("renderProcessStatus"))
        self.assertNotIn("loadCacheSizes", self._body("renderProcessStatus"))
        # ...and it IS refreshed at the moment the number can have changed
        self.assertIn("loadCacheSizes();", self._body("refreshTabsAfterProcess"))

    def test_the_reset_dialog_still_carries_the_f93_checkbox(self):
        """F94 lives next to the reset button; the geo checkbox of the reset dialog is
        a different way out of the same cache and must survive untouched."""
        self.assertIn('id="reset-clear-geo-checkbox"', self.html)
        self.assertIn('postJson("/api/process/reset", { clear_geo: clearGeo })', self.html)


class TestCacheStrings(unittest.TestCase):
    def test_every_caption_exists_in_all_three_languages(self):
        for key in _F94_STRING_KEYS:
            entry = ui._UI_STRINGS[key]
            for lang in ("ru", "en", "ja"):
                self.assertIn(lang, entry, key)
                self.assertTrue(entry[lang].strip(), key)

    def test_the_translations_are_actually_different_texts(self):
        """A copy-pasted English string in the ja column is the usual way a language
        silently goes missing — the labels here are short enough to check outright."""
        for key in ("cache_title", "cache_clear_preview_button", "cache_clear_geo_button"):
            entry = ui._UI_STRINGS[key]
            self.assertEqual(len({entry["ru"], entry["en"], entry["ja"]}), 3, key)

    def test_the_size_line_carries_every_placeholder(self):
        for lang in ("ru", "en", "ja"):
            line = ui._t("cache_sizes", lang)
            for token in ("{preview}", "{files}", "{geo}"):
                self.assertIn(token, line, lang)

    def test_the_buttons_are_rendered_in_the_chosen_language(self):
        for lang in ("ru", "en", "ja"):
            html = ui._render_index_html(lang)
            self.assertIn(ui._t("cache_clear_preview_button", lang), html)
            self.assertIn(ui._t("cache_clear_geo_button", lang), html)
            self.assertNotIn("{{cache_", html)


if __name__ == "__main__":
    unittest.main()
