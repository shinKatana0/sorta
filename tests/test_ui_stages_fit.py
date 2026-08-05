"""F191: the stage row fits, and keeps fitting when the pipeline changes.

Reported from a live run: while processing, the stages spilled past the width of the
cards above them. Lengthening those cards would have been the wrong fix — the stages
are nine today and were eleven the day before F186 retired two of them, so a width
fitted to the current list comes apart at the next edit of the pipeline, and on a
narrow screen it never fitted at all.

So the row stopped being a row. What is always on screen is the stage that is going,
"N of M" and ONE bar — three scalars out of `/api/process/status`, drawn into nodes
`page.html` already carries. The per-stage chips are behind a disclosure.

What is pinned here:

  * the collapsed row is built from three numbers and creates no nodes, so its shape
    cannot follow the stage count — the main property, checked against runs of three,
    six and nine stages;
  * the name of the current stage and the counter are in it;
  * an error is in it too: a disclosure may hide a list, never a failure;
  * opening it shows what the old row showed — done / now / pending, every stage;
  * the open state survives the 1.5-second status poll;
  * the captions come from the catalogue in all three languages.
"""
from __future__ import annotations

import re
import unittest

from sorta import ui

from tests.test_ui_process import ProcessTestBase, _poll_until

# The ids of the collapsed row. Every one of them exists in the markup — nothing here
# is created while a run goes, which is the whole reason the width holds.
_SUMMARY_IDS = ("process-stages", "process-stages-toggle", "process-stages-current",
                "process-stages-count", "process-stages-bar", "process-stages-caret")


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


def _stage_row_markup(html: str) -> str:
    """The `#process-stages` block of the rendered page, up to its closing div."""
    start = html.index('<div id="process-stages"')
    end = html.index('<div id="process-status"', start)
    return html[start:end]


class StageRowTestBase(ProcessTestBase):
    def patch_stage_count(self, count: int) -> None:
        """Replace the pipeline with `count` stubbed stages.

        The pipeline itself is not touched (F191 is not allowed to touch it) — this is
        test data, and it is the only way to ask the question the feature exists for:
        what does the screen do when the number of stages changes.
        """
        calls = self.calls

        def fake_step(name):
            def step(cfg, conn, cb):
                calls.append(name)
                cb(1, 1)
                return None
            return step

        names = [f"stage{i}" for i in range(1, count + 1)]
        self._patch("_pipeline_steps", lambda: [(n, fake_step(n)) for n in names])

    def run_with_stages(self, count: int) -> dict:
        """Run a whole pipeline of `count` stages and return the final status."""
        self.patch_stage_count(count)
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status, 200)
        return _poll_until(self.status, lambda d: d["finished"])


class TestTheRowDoesNotFollowTheStageCount(StageRowTestBase):
    """The main property: adding a stage changes numbers, never the markup."""

    def test_the_status_describes_the_stages_with_three_scalars(self):
        self.start_server()
        shapes = {}
        for count in (3, 6, 9):
            with self.subTest(stages=count):
                final = self.run_with_stages(count)
                self.assertEqual(final["stage_total"], count)
                self.assertEqual(final["stage_index"], count)
                shapes[count] = set(final)
        # The stage count travels as ONE number in a payload of a fixed shape. There is
        # no per-stage list for the collapsed row to be as long as.
        self.assertEqual(shapes[3], shapes[6])
        self.assertEqual(shapes[6], shapes[9])

    def test_the_markup_of_the_row_holds_no_stage_at_all(self):
        """The row is in the template, so its nodes are the same for any pipeline."""
        html = ui._render_index_html("ru")
        row = _stage_row_markup(html)
        for name in ui._PIPELINE_STAGE_NAMES:
            with self.subTest(stage=name):
                self.assertNotIn(ui._t("process_stage_" + name, "ru"), row)
        # exactly one bar, and it is not one per stage
        self.assertEqual(row.count("<progress"), 1)
        # the fixed set of nodes, and no more of them than that
        for node_id in _SUMMARY_IDS:
            self.assertIn(f'id="{node_id}"', row)
        self.assertEqual(len(re.findall(r'\sid="', row)), len(_SUMMARY_IDS) + 1)

    def test_the_collapsed_row_creates_no_nodes(self):
        """A drawing routine that appends nothing cannot draw a stage's worth of
        anything — this is what makes the property above hold in the browser too."""
        body = _js_body(ui._render_index_html("ru"), "renderStageSummary")
        self.assertNotIn("createElement", body)
        self.assertNotIn("appendChild", body)
        # ...and it is fed by the scalars of the status, not by the list of stages
        self.assertIn("data.stage_index", body)
        self.assertIn("data.stage_total", body)
        self.assertNotIn("currentProcessStages.forEach", body)

    def test_only_the_list_behind_the_disclosure_is_per_stage(self):
        body = _js_body(ui._render_index_html("ru"), "renderStageList")
        self.assertIn("currentProcessStages.forEach", body)
        self.assertIn('getElementById("process-stages-list")', body)


class TestWhatStaysVisibleCollapsed(StageRowTestBase):
    def test_the_row_names_the_stage_that_is_going_and_counts_it(self):
        body = _js_body(ui._render_index_html("ru"), "renderStageSummary")
        self.assertIn("stageStateLabel(data)", body)
        self.assertIn("I18N.process_stage_counter", body)
        self.assertIn("index: index, total: total", body)
        label = _js_body(ui._render_index_html("ru"), "stageStateLabel")
        self.assertIn("I18N.process_stage_current", label)
        self.assertIn("processStageLabel(data.stage)", label)

    def test_the_counter_is_written_into_the_collapsed_node(self):
        body = _js_body(ui._render_index_html("ru"), "renderStageSummary")
        self.assertIn('getElementById("process-stages-current")', body)
        self.assertIn('getElementById("process-stages-count")', body)

    def test_a_failed_stage_is_named_by_the_status(self):
        """The server has to say WHICH stage fell over: `stage` alone cannot, and a
        failure that is only visible after a click is a failure that is hidden."""
        self.start_server()
        self.patch_stage_count(4)

        def boom(cfg, conn, cb):
            self.calls.append("stage3")
            raise RuntimeError("boom")

        original = ui.process._pipeline_steps()
        self._patch("_pipeline_steps",
                    lambda: [(n, boom if n == "stage3" else fn) for n, fn in original])
        status, _resp = self.post("/api/process", {"source_dir": str(self.src_dir)})
        self.assertEqual(status, 200)
        final = _poll_until(self.status, lambda d: d["finished"])
        self.assertIn("boom", final["error"])
        self.assertEqual(final["error_stage"], "stage3")
        self.assertEqual(self.calls, ["stage1", "stage2", "stage3"])

    def test_a_run_that_did_not_fail_names_no_stage(self):
        self.start_server()
        final = self.run_with_stages(3)
        self.assertIsNone(final["error"])
        self.assertIsNone(final["error_stage"])

    def test_the_error_caption_is_the_row_itself_not_a_chip(self):
        html = ui._render_index_html("ru")
        label = _js_body(html, "stageStateLabel")
        self.assertIn("I18N.process_stage_failed", label)
        self.assertIn("data.error_stage || data.stage", label)
        # the failed state is on the collapsed block, so it is visible while it is shut
        self.assertIn('classList.add("failed")',
                      _js_body(html, "renderStageSummary"))

    def test_the_row_is_shown_whenever_there_is_a_run_to_talk_about(self):
        body = _js_body(ui._render_index_html("ru"), "renderStages")
        self.assertIn("data.running || data.finished", body)

    def test_the_bar_is_only_full_for_a_run_that_reached_the_end(self):
        """A stopped or failed run must not be drawn as a completed one — the bar is
        the part of the row a glance reads first."""
        body = _js_body(ui._render_index_html("ru"), "stageBarValue")
        self.assertIn("data.running || data.cancel_requested", body)
        self.assertIn("Math.max(index - 1, 0)", body)


class TestOpeningItShowsTheOldRow(unittest.TestCase):
    def setUp(self):
        self.html = ui._render_index_html("ru")

    def test_every_stage_still_gets_a_chip_with_its_state(self):
        body = _js_body(self.html, "renderStageList")
        for cls in ('"pending"', '"done"', '"now"', '"failed"'):
            with self.subTest(cls=cls):
                self.assertIn(cls, body)
        self.assertIn("processStageLabel(name)", body)
        self.assertIn('icon("check")', body)
        self.assertIn("stepIndex < data.stage_index", body)

    def test_the_phase_and_its_clock_stay_outside_the_disclosure(self):
        """"Nothing is hidden but the list": the sub-phase caption and the elapsed
        seconds are still drawn by their own function, next to the row."""
        status = _js_body(self.html, "renderProcessStatus")
        self.assertIn("renderProcessPhase(data);", status)
        self.assertIn("renderStages(data);", status)
        self.assertIn("I18N.process_phase_elapsed",
                      _js_body(self.html, "renderProcessPhase"))


class TestTheOpenStateSurvivesAnUpdate(unittest.TestCase):
    def setUp(self):
        self.html = ui._render_index_html("ru")

    def test_the_flag_lives_outside_the_drawing(self):
        self.assertIn("var stagesExpanded = false;", self.html)
        for name in ("renderStages", "renderStageSummary", "renderStageList"):
            with self.subTest(name=name):
                self.assertNotIn("stagesExpanded =", _js_body(self.html, name))

    def test_the_poll_redraws_the_list_without_closing_it(self):
        body = _js_body(self.html, "renderStageList")
        self.assertIn('list.style.display = stagesExpanded ? "" : "none";', body)
        self.assertIn('setAttribute("aria-expanded", stagesExpanded ? "true" : "false")',
                      body)

    def test_the_toggle_flips_the_flag_and_redraws_between_polls(self):
        start = self.html.index('getElementById("process-stages-toggle").addEventListener')
        handler = self.html[start:start + 400]
        self.assertIn("stagesExpanded = !stagesExpanded;", handler)
        self.assertIn("renderStageList(lastProcessStatus);", handler)

    def test_start_over_clears_the_row_with_the_index(self):
        reset = self.html[self.html.index('postJson("/api/process/reset"'):]
        self.assertIn("renderStages({});", reset[:700])


class TestTheCaptionsComeFromTheCatalogue(unittest.TestCase):
    KEYS = ("process_stage_current", "process_stage_counter", "process_stage_failed",
            "process_stages_done", "process_stages_stopped", "process_stages_toggle")

    def test_every_caption_exists_in_all_three_languages(self):
        for key in self.KEYS:
            entry = ui._UI_STRINGS[key]
            for lang in ("ru", "en", "ja"):
                with self.subTest(key=key, lang=lang):
                    self.assertIn(lang, entry)
                    self.assertTrue(entry[lang].strip())

    def test_the_placeholders_are_the_same_in_every_language(self):
        for key, placeholders in (("process_stage_current", ("{stage}",)),
                                  ("process_stage_failed", ("{stage}",)),
                                  ("process_stage_counter", ("{index}", "{total}"))):
            for lang in ("ru", "en", "ja"):
                line = ui._UI_STRINGS[key][lang]
                for placeholder in placeholders:
                    with self.subTest(key=key, lang=lang, placeholder=placeholder):
                        self.assertIn(placeholder, line)

    def test_no_caption_names_a_stage_of_its_own(self):
        """`{stage}` is filled from `process_stage_*`, so the row says nothing about
        the pipeline itself — which is what lets the pipeline change under it."""
        for key in self.KEYS:
            for lang in ("ru", "en", "ja"):
                line = ui._UI_STRINGS[key][lang]
                for stage in ui._PIPELINE_STAGE_NAMES:
                    with self.subTest(key=key, lang=lang, stage=stage):
                        self.assertNotIn(ui._t("process_stage_" + stage, lang), line)

    def test_the_toggle_hint_reaches_the_page_in_the_chosen_language(self):
        for lang, expected in (("ru", "Показать все этапы"),
                               ("en", "Show all stages"),
                               ("ja", "全ステージを表示")):
            with self.subTest(lang=lang):
                row = _stage_row_markup(ui._render_index_html(lang))
                self.assertIn(f'title="{expected}"', row)


if __name__ == "__main__":
    unittest.main()
