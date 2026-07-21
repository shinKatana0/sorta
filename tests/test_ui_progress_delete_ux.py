"""#36/#37: delete-remember contextually (city/dupes) + indeterminate progress
(total unknown -> "processed X" + a running bar, not "X of 0")."""
from __future__ import annotations

import unittest

from tests.test_ui import UiServerTestBase


class TestDeleteRememberContextual(UiServerTestBase):
    def test_delete_remember_row_hidden_by_default_and_toggled_by_tab(self):
        # #36: a wrapper row with style display:none (default — the "Process" tab)
        self.start_server()
        _s, body, _c = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('id="delete-remember-row"', html)
        self.assertIn('id="delete-remember-row" style="display:none"', html)
        # activateTab shows the row only on city/dupes
        self.assertIn('"delete-remember-row").style.display', html)
        self.assertIn('(name === "city" || name === "dupes") ? "" : "none"', html)


class TestIndeterminateProgress(UiServerTestBase):
    def test_indeterminate_i18n_all_langs(self):
        # #37: the indeterminate placeholder string exists in all three languages
        from sorta.ui import _UI_STRINGS
        key = "process_stage_progress_indeterminate"
        self.assertIn(key, _UI_STRINGS)
        for lang in ("ru", "en", "ja"):
            self.assertIn("{done}", _UI_STRINGS[key][lang])
            self.assertIn("{stage}", _UI_STRINGS[key][lang])

    def test_indeterminate_css_and_branch(self):
        # #37: CSS class + i18n key in the markup (the page language is baked in), the total>0 branch
        self.start_server()
        _s, body, _c = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("process_stage_progress_indeterminate", html)
        self.assertIn("{done} processed", html)          # the page language is en
        self.assertIn(".process-progress.indeterminate", html)
        self.assertIn("@keyframes process-indeterminate", html)
        # JS picks the indeterminate string when total<=0
        self.assertIn("data.total > 0 ? I18N.process_stage_progress : "
                      "I18N.process_stage_progress_indeterminate", html)
        self.assertIn('bar.classList.add("indeterminate")', html)

    def test_reduced_motion_respected(self):
        self.start_server()
        _s, body, _c = self.get("/")
        html = body.decode("utf-8")
        # animation only under prefers-reduced-motion: no-preference
        self.assertIn("prefers-reduced-motion: no-preference", html)


if __name__ == "__main__":
    unittest.main()
