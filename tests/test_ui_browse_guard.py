"""The folder dialog must not open twice.

Reported from a live run: the dialog takes a second or two to appear, the button
stayed active, and every extra click spawned another Explorer window.
"""
from __future__ import annotations

import threading
import time
import unittest
import unittest.mock

from sorta import ui


class TestBrowseDialogGuard(unittest.TestCase):
    def test_concurrent_calls_open_one_dialog(self):
        opened = []
        started = threading.Event()

        def slow_dialog():
            opened.append(1)
            started.set()
            time.sleep(0.3)  # the window is being built — this is the vulnerable gap
            return "C:/chosen"

        with unittest.mock.patch.object(ui.process, "_run_browse_dialog", slow_dialog):
            results: list[str] = []
            first = threading.Thread(target=lambda: results.append(ui._browse_for_folder()))
            first.start()
            self.assertTrue(started.wait(2), "первый диалог не стартовал")
            # a second click while the first dialog is still coming up
            results.append(ui._browse_for_folder())
            first.join(5)

        self.assertEqual(len(opened), 1, "второй клик открыл ещё один диалог")
        self.assertIn("C:/chosen", results)
        self.assertIn("", results)  # the refused call answers like a cancel

    def test_lock_is_released_for_the_next_click(self):
        with unittest.mock.patch.object(ui.process, "_run_browse_dialog", lambda: "C:/a"):
            self.assertEqual(ui._browse_for_folder(), "C:/a")
            self.assertEqual(ui._browse_for_folder(), "C:/a")

    def test_lock_is_released_when_the_dialog_raises(self):
        def boom():
            raise RuntimeError("no display")

        with unittest.mock.patch.object(ui.process, "_run_browse_dialog", boom):
            with self.assertRaises(RuntimeError):
                ui._browse_for_folder()
        # a failure must not wedge the button forever
        with unittest.mock.patch.object(ui.process, "_run_browse_dialog", lambda: "C:/b"):
            self.assertEqual(ui._browse_for_folder(), "C:/b")


class TestBrowseButtonMarkup(unittest.TestCase):
    """The client half of the same fix — the button is disabled for the duration."""

    def test_all_browse_buttons_go_through_the_guard_helper(self):
        html = ui._render_index_html("ru")
        self.assertIn("function browseIntoField(btn, apply)", html)
        # no direct call sites left: every one of the three must use the helper
        self.assertNotIn('postJson("/api/browse"', html.replace(
            'postJson("/api/browse", {})\n      .then', "HELPER"))


if __name__ == "__main__":
    unittest.main()
