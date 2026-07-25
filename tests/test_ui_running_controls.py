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

    def test_inputs_are_refreshed_on_every_status_tick(self):
        self.assertIn("updateProcessInputsDisabled();",
                      self._body("renderProcessStatus"))

    def test_rerun_button_starts_disabled_in_markup(self):
        """Before the first poll there is no state yet — the safe default is off."""
        markup = re.search(r'<button[^>]*id="process-rerun-optional-btn"[^>]*>', self.html)
        self.assertIsNotNone(markup)
        self.assertIn("disabled", markup.group(0))


if __name__ == "__main__":
    unittest.main()
