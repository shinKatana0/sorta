"""Controls that must stay dead while the pipeline is running.

Reported from a live run: with a process in flight, ticking the "faces" checkbox
re-enabled "re-run this stage" — the checkbox handler fired immediately and
overwrote what the status poll had disabled, leaving a window in which a second
pipeline could be started on top of the first.

F135 removed that second button, so the window it opened is gone by construction —
what stays here is the rest of the family: the source inputs, the option checkboxes,
the layout controls and "Start over", all of which are still live buttons next to a
running pipeline.
"""
from __future__ import annotations

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

    def test_status_poll_records_the_running_state(self):
        body = self._body("renderProcessStatus")
        self.assertIn("processRunning = !!data.running", body)
        # F135: the second button is gone, so the flag now guards the start button and
        # the inputs — the same flag, one consumer fewer.
        self.assertIn("startBtn.disabled = processRunning", body)

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
        # F222 moved the ticks into `updateOptionAvailability`, which the function
        # above calls: a missing install tier is a third reason for a control to be
        # dead, and a status tick that only knew about the run would have re-enabled
        # those boxes every 1.5 seconds. The keys are the ones the server prices and
        # probes them under; the ids follow from them.
        self.assertIn("updateOptionAvailability();",
                      self._body("updateProcessInputsDisabled"))
        body = self._body("updateOptionAvailability")
        self.assertIn("processRunning", body)
        keys = self.html.split("var RUN_OPTION_KEYS = ", 1)[1].split("]", 1)[0]
        for key in ("deep", "geo_online", "faces", "events"):
            self.assertIn(f'"{key}"', keys)

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
        # F145: the three flags are read through one predicate now, because the same
        # question is asked by a dozen controls on five tabs.
        self.assertIn("uiBusy()", body)
        self.assertIn("sortRunning || processRunning || undoRunning",
                      self._body("uiBusy"))

    def test_start_over_is_dead_while_anything_runs(self):
        """"Начать заново" wipes the whole index. The server answers 409 under the
        same busy_lock, but the button asked for confirmation FIRST and reported the
        refusal after — a scary dialog for an action that could not happen anyway."""
        body = self._body("updateBusyControlsDisabled")
        self.assertIn("process-reset-btn", body)
        self.assertIn("var busy = uiBusy();", body)

    def test_busy_controls_are_refreshed_by_the_process_poll(self):
        """Sort polling stops when no sort runs, so the process tick has to be the one
        that re-enables them — otherwise they stay dead until the page is reloaded."""
        self.assertIn("updateBusyControlsDisabled();",
                      self._body("renderProcessStatus"))

    def test_the_second_run_button_is_gone_with_its_state(self):
        """F135: one run button. The "re-run selected" button used to need its own
        empty-index guard and its own checkbox handlers — both went with it, and
        nothing may be left referring to a button that no longer exists."""
        for name in ("process-rerun-optional-btn", "rerunSelectedAllowed",
                     "updateRerunSelectedDisabled", "indexHasFiles"):
            self.assertNotIn(name, self.html)


if __name__ == "__main__":
    unittest.main()
