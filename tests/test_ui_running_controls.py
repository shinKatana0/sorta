"""Controls that must stay dead while the pipeline is running.

Reported from a live run: with a process in flight, ticking the "faces" checkbox
re-enabled "re-run this stage" — the checkbox handler fired immediately and
overwrote what the status poll had disabled, leaving a window in which a second
pipeline could be started on top of the first.
"""
from __future__ import annotations

import re
import unittest

from sorta import ui


class TestRunningStateGuards(unittest.TestCase):
    def setUp(self):
        self.html = ui._render_index_html("ru")

    def _body(self, name: str) -> str:
        """Source of a JS function declaration, up to its closing brace."""
        start = self.html.index(f"function {name}(")
        depth, i = 0, self.html.index("{", start)
        for j in range(i, len(self.html)):
            if self.html[j] == "{":
                depth += 1
            elif self.html[j] == "}":
                depth -= 1
                if depth == 0:
                    return self.html[start:j + 1]
        raise AssertionError(f"не найдено тело {name}")

    def test_checkbox_handler_respects_the_running_state(self):
        """The actual regression: this handler used to ignore `processRunning`."""
        body = self._body("updateRerunSelectedDisabled")
        self.assertIn("processRunning", body)

    def test_status_poll_records_the_running_state(self):
        body = self._body("renderProcessStatus")
        self.assertIn("processRunning = !!data.running", body)
        # and the rerun button is still gated on it, not only on the checkboxes
        self.assertIn("rerunBtn.disabled = processRunning", body)

    def test_source_inputs_are_disabled_while_running(self):
        body = self._body("updateProcessInputsDisabled")
        for control in ("process-browse-btn", "process-source-dir"):
            self.assertIn(control, body)
        self.assertIn("processRunning", body)

    def test_run_option_checkboxes_are_disabled_while_running(self):
        """Reported from a live run: the options stayed clickable mid-flight.

        They are sent to the server once, at start, so a box ticked or cleared
        afterwards changes nothing about the run in progress — it only looks like it
        does, and that is found out an hour later by the stage that still ran.
        """
        body = self._body("updateProcessInputsDisabled")
        for control in ("process-deep-checkbox", "process-geo-online-checkbox",
                        "process-faces-checkbox", "process-events-checkbox"):
            self.assertIn(control, body)

    def test_the_options_come_back_when_the_run_ends(self):
        """`disabled = processRunning`, not a one-way disable: a finished or cancelled
        run must leave every control usable again without reloading the page."""
        body = self._body("updateProcessInputsDisabled")
        self.assertIn("el.disabled = processRunning", body)

    def test_inputs_are_refreshed_on_every_status_tick(self):
        self.assertIn("updateProcessInputsDisabled();",
                      self._body("renderProcessStatus"))

    def test_the_layout_controls_are_dead_while_the_pipeline_runs(self):
        """The server already answers 409 ("process is running") under busy_lock, so
        nothing destructive could start — but the button stayed live, and you found
        that out by clicking. The stronger reason is the plan itself: mid-run `places`
        is wiped and `media_class` is not filled yet, so a layout started now would
        move the collection according to a half-built index."""
        body = self._body("updateBusyControlsDisabled")
        for control in ("sort-apply-btn", "sort-browse-btn", "sort-dest"):
            self.assertIn(control, body)
        self.assertIn("processRunning", body)

    def test_start_over_is_dead_while_anything_runs(self):
        """"Начать заново" wipes the whole index. The server answers 409 under the
        same busy_lock, but the button asked for confirmation FIRST and reported the
        refusal after — a scary dialog for an action that could not happen anyway."""
        body = self._body("updateBusyControlsDisabled")
        self.assertIn("process-reset-btn", body)
        self.assertIn("sortRunning || processRunning", body)

    def test_busy_controls_are_refreshed_by_the_process_poll(self):
        """Sort polling stops when no sort runs, so the process tick has to be the one
        that re-enables them — otherwise they stay dead until the page is reloaded."""
        self.assertIn("updateBusyControlsDisabled();",
                      self._body("renderProcessStatus"))

    def test_rerun_needs_a_non_empty_index(self):
        """Reported from a live session: right after "Start over" the index is empty,
        and ticking "faces" lit up "catch this stage up" — offering to re-run a stage
        over no files at all. The flag is refreshed by the same fetch that shows/hides
        the People and Events tabs, which already runs after a reset."""
        body = self._body("rerunSelectedAllowed")
        self.assertIn("indexHasFiles", body)
        self.assertIn("indexHasFiles = !!data.indexed", self.html)
        self.assertIn("updateRerunSelectedDisabled();", self._body("applyTabVisibility"))

    def test_rerun_button_starts_disabled_in_markup(self):
        """Before the first poll there is no state yet — the safe default is off."""
        markup = re.search(r'<button[^>]*id="process-rerun-optional-btn"[^>]*>', self.html)
        self.assertIsNotNone(markup)
        self.assertIn("disabled", markup.group(0))


if __name__ == "__main__":
    unittest.main()
