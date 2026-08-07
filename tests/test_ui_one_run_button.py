"""F135: one run button instead of two.

"Re-run selected" existed to save three stages (index 34 s, geo 3 s, landmarks 138 s
on the live 380 GB run) — about three minutes, paid for with a permanent fork in the
road: which of the two buttons is the right one this time. Everything else was already
skipped by the stages themselves (junk by the prompt fingerprint, faces by the marker,
events by composition), so the fork bought very little.

So the button is gone and "Start" is the whole pipeline again. Three things have to
hold for that to be an improvement rather than a regression:

  * the source comes back by itself — otherwise every repeat run means typing a path;
  * the ROUTE /api/process/rerun-optional keeps working (tests/test_ui_rerun_optional);
  * the run says what it skipped as already done — a silent skip reads as "nothing
    happened", which is exactly the complaint the second button was built around.
"""
from __future__ import annotations

import unittest

from sorta import ui

from tests.test_ui_process import ProcessTestBase, _FakeIndexStats, _poll_until


class _FakeIndexResult:
    """What `indexer.run_index` returns — only the fields the summary reads."""

    def __init__(self, added: int, updated: int, skipped: int) -> None:
        self.added = added
        self.updated = updated
        self.skipped = skipped


class _FakeJunkResult:
    """What `junk.classify_junk` returns; `skipped_incremental` is the F68 counter."""

    def __init__(self, processed: int, skipped_incremental: int) -> None:
        self.processed = processed
        self.skipped_incremental = skipped_incremental


class OneButtonTestBase(ProcessTestBase):
    def patch_stages_with_stats(self, *, added: int = 1, updated: int = 1,
                                skipped: int = 7, processed: int = 2,
                                skipped_incremental: int = 9) -> None:
        """The fast stubs, with `index` and `junk` returning stats like the real ones.

        `patch_fast_stages` returns None from every stage — that is the "a stage tells
        us nothing" case, tested separately below.
        """
        self.patch_fast_stages()
        calls = self.calls

        def fake_index(cfg, conn, progress=None):
            calls.append("index")
            if progress:
                progress(_FakeIndexStats(added + updated + skipped))
            return _FakeIndexResult(added=added, updated=updated, skipped=skipped)

        def fake_junk(cfg, conn, classifier=None, use_clip=True, text_detector=None,
                      verdicts_only=False, progress=None):
            # F165: two stages, one function. Both report the same pair of counters
            # to the status, which is what the cases below read.
            calls.append("classify" if verdicts_only else "junk")
            if progress:
                progress(processed, processed)
            return _FakeJunkResult(processed=processed,
                                   skipped_incremental=skipped_incremental)

        self._patch("run_index", fake_index)
        self._patch("classify_junk", fake_junk)


class TestTheSecondButtonIsGone(OneButtonTestBase):
    def test_markup_has_one_run_button(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="process-start-btn"', html)
        self.assertNotIn("process-rerun-optional-btn", html)
        self.assertNotIn("process-rerun-block", html)
        self.assertNotIn("process-rerun-hint", html)

    def test_its_strings_left_the_catalogue_too(self):
        for key in ("process_rerun_optional_button", "process_rerun_optional_hint"):
            self.assertNotIn(key, ui._UI_STRINGS)

    def test_no_javascript_refers_to_it(self):
        """A leftover getElementById on a removed node throws and kills the rest of
        the handler — the whole client lives in one script."""
        html = ui._render_index_html("ru")
        for name in ("filterRerunStages", "rerunSelectedAllowed",
                     "updateRerunSelectedDisabled", "rerunBtn"):
            self.assertNotIn(name, html)

    def test_the_route_is_still_wired(self):
        """The button goes, the endpoint stays — it is documented and callable from
        outside. Its own behaviour is pinned by tests/test_ui_rerun_optional.py."""
        self.patch_fast_stages()
        self.start_server()
        status, resp = self.post("/api/process/rerun-optional", {"faces": True})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))
        _poll_until(self.status, lambda d: d["finished"])


class TestSourceIsRemembered(OneButtonTestBase):
    def test_status_keeps_the_source_of_the_finished_run(self):
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status, 200)
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        # Not just during the run — this is what the field is refilled from afterwards.
        self.assertFalse(final["running"])
        self.assertEqual(final["source_dir"], str(self.src_dir))

    def test_the_client_fills_an_empty_field_from_it(self):
        html = ui._render_index_html("ru")
        body = _js_body(html, "adoptRememberedSource")
        self.assertIn("data.source_dir", body)
        self.assertIn('document.getElementById("process-source-dir")', body)
        self.assertIn("input.value = data.source_dir", body)
        # and it is fed by the status poll, not by a one-off read at page load
        self.assertIn("adoptRememberedSource(data);",
                      _js_body(html, "renderProcessStatus"))

    def test_a_field_with_something_in_it_is_never_overwritten(self):
        body = _js_body(ui._render_index_html("ru"), "adoptRememberedSource")
        self.assertIn("if (!data.source_dir || input.value.trim()) return;", body)

    def test_a_run_still_writes_the_path_into_the_browser_memory(self):
        """The F81 mechanism (localStorage) is what survives a server restart; the
        status only covers a fresh browser against a live server. Both, or a repeat
        run still means typing the path."""
        html = ui._render_index_html("ru")
        self.assertIn("rememberSourceDir();", html)
        self.assertIn('var SOURCE_DIR_KEY = "sorta.sourceDir";', html)


class TestStartStillNeedsASource(OneButtonTestBase):
    def test_empty_body_is_rejected_by_the_server(self):
        self.patch_fast_stages()
        self.start_server()
        status, resp = self.post("/api/process", {})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)
        self.assertEqual(self.calls, [])

    def test_blank_source_is_rejected_by_the_server(self):
        self.patch_fast_stages()
        self.start_server()
        status, resp = self.post("/api/process", {"source_dir": "   "})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)
        self.assertEqual(self.calls, [])

    def test_the_button_asks_for_a_path_on_an_empty_field(self):
        html = ui._render_index_html("ru")
        self.assertIn("if (!path) { window.alert(I18N.process_enter_path); return; }",
                      html)


class TestCheckboxesStillDecideTheOptionalStages(OneButtonTestBase):
    def test_unchecked_means_the_stage_does_not_run_at_all(self):
        self.patch_fast_stages()
        self.start_server()
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertNotIn("faces", self.calls)
        self.assertNotIn("events", self.calls)

    def test_checked_means_it_runs_inside_the_same_single_run(self):
        self.patch_fast_stages()
        self.start_server()
        # F222: `landmarks` is a third box of exactly this kind — the run walks it only
        # because it was ticked, in the same single run as the other two.
        status, _resp = self.post(
            "/api/process",
            {"source_dir": str(self.src_dir), "faces": True, "events": True,
             "landmarks": True})
        self.assertEqual(status, 200)
        _poll_until(self.status, lambda d: d["finished"])
        self.assertEqual(
            self.calls,
            ["index", "assign_duplicates", "geo", "landmarks", "classify", "faces",
             "events", "name_events", "junk", "phash"],
        )

    def test_the_start_click_still_sends_every_checkbox(self):
        html = ui._render_index_html("ru")
        self.assertIn("source_dir: path, deep: deep, geo_online: geoOnline, "
                      "faces: faces, events: events,", html)
        self.assertIn("landmarks: landmarks, pets: pets,", html)


class TestTheRunSaysWhatItSkipped(OneButtonTestBase):
    def test_status_carries_processed_and_skipped_per_stage(self):
        self.patch_stages_with_stats()
        self.start_server()
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status, 200)
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        # index: added+updated is the work, `skipped` is what path+mtime+size matched
        self.assertEqual(final["stage_stats"]["index"], {"processed": 2, "skipped": 7})
        self.assertEqual(final["stage_stats"]["junk"], {"processed": 2, "skipped": 9})

    def test_a_stage_without_such_a_counter_claims_nothing(self):
        self.patch_stages_with_stats()
        self.start_server()
        self.post("/api/process", {"source_dir": str(self.src_dir)})
        final = _poll_until(self.status, lambda d: d["finished"])
        for stage in ("geo", "landmarks", "phash"):
            self.assertNotIn(stage, final["stage_stats"])

    def test_a_stage_that_returns_nothing_is_not_an_error(self):
        """Stages are replaceable; a missing pair must cost a caption, not the run."""
        self.patch_fast_stages()  # every stub returns None
        self.start_server()
        self.post("/api/process", {"source_dir": str(self.src_dir)})
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertEqual(final["stage_stats"], {})

    def test_an_idle_server_reports_no_numbers(self):
        self.start_server()
        self.assertEqual(self.status()["stage_stats"], {})

    def test_the_next_run_replaces_the_numbers_rather_than_adding_to_them(self):
        self.patch_stages_with_stats(added=1, updated=1, skipped=7)
        self.start_server()
        self.post("/api/process", {"source_dir": str(self.src_dir)})
        _poll_until(self.status, lambda d: d["finished"])
        self.patch_stages_with_stats(added=0, updated=0, skipped=9,
                                     processed=0, skipped_incremental=11)
        self.post("/api/process", {"source_dir": str(self.src_dir)})
        final = _poll_until(
            self.status,
            lambda d: d["finished"] and d["stage_stats"].get("junk", {}).get(
                "skipped") == 11)
        self.assertEqual(final["stage_stats"]["index"], {"processed": 0, "skipped": 9})

    def test_stage_stats_helper_sums_the_work_and_refuses_to_invent_it(self):
        stats = _FakeIndexResult(added=1, updated=2, skipped=5)
        self.assertEqual(ui._stage_stats(stats, ("added", "updated"), "skipped"),
                         {"processed": 3, "skipped": 5})
        # a name the object does not carry -> no line at all, not a zeroed one
        self.assertIsNone(ui._stage_stats(stats, ("added",), "skipped_incremental"))
        self.assertIsNone(ui._stage_stats(None, ("added",), "skipped"))

    def test_the_client_renders_one_line_per_reported_stage(self):
        html = ui._render_index_html("ru")
        self.assertIn('<div id="process-summary" class="process-summary"></div>', html)
        body = _js_body(html, "renderProcessSummary")
        self.assertIn("data.stage_stats", body)
        self.assertIn("I18N.process_summary_title", body)
        self.assertIn("I18N.process_summary_stage", body)
        self.assertIn("processed: stats[name].processed", body)
        self.assertIn("skipped: stats[name].skipped", body)
        # and it is redrawn by the same status tick as everything else on the tab
        self.assertIn("renderProcessSummary(data);",
                      _js_body(html, "renderProcessStatus"))

    def test_start_over_wipes_the_summary_with_the_index(self):
        """The numbers counted files of an index "Start over" has just deleted."""
        html = ui._render_index_html("ru")
        reset = html[html.index('postJson("/api/process/reset"'):]
        self.assertIn("renderProcessSummary({});", reset[:600])

    def test_summary_strings_exist_in_all_three_languages(self):
        for key in ("process_summary_title", "process_summary_stage"):
            entry = ui._UI_STRINGS[key]
            for lang in ("ru", "en", "ja"):
                with self.subTest(key=key, lang=lang):
                    self.assertIn(lang, entry)
                    self.assertTrue(entry[lang].strip())
        for lang in ("ru", "en", "ja"):
            line = ui._UI_STRINGS["process_summary_stage"][lang]
            for placeholder in ("{stage}", "{processed}", "{skipped}"):
                self.assertIn(placeholder, line)

    def test_summary_strings_reach_the_page_in_the_chosen_language(self):
        self.start_server()
        for lang, expected in (("ru", "Что сделал прогон:"),
                               ("en", "What the run did:"),
                               ("ja", "この実行の内容:")):
            with self.subTest(lang=lang):
                _status, body, _ctype = self.get(f"/?lang={lang}")
                self.assertIn(expected, body.decode("utf-8"))


def _js_body(html: str, name: str) -> str:
    """Source of a JS function declaration, up to its closing brace."""
    start = html.index(f"function {name}(")
    depth = 0
    for j in range(html.index("{", start), len(html)):
        if html[j] == "{":
            depth += 1
        elif html[j] == "}":
            depth -= 1
            if depth == 0:
                return html[start:j + 1]
    raise AssertionError(f"no body found for {name}")


if __name__ == "__main__":
    unittest.main()
