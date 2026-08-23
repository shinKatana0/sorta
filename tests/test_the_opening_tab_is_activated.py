"""The tab the markup opens on is activated by the script, not merely shown.

`page.html` marks one tab `active` and puts a "loading" line inside `overview-body`.
Only `activateTab` asks for the numbers that replace it, and nothing called it on boot —
so a freshly opened page sat at "loading" until the person left the tab and came back.
Met in a VM on 2026-08-23; the code had been that way since the tabs existed.

The pair is checked rather than the fix: the markup and the script each name the opening
tab, and a guard that reads one of them alone would pass while they disagreed.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

_WEB = Path(__file__).resolve().parents[1] / "sorta" / "web"
_PAGE = _WEB / "page.html"
_APP = _WEB / "app" / "app.js"


def opening_tabs() -> list[str]:
    """Every tab panel the markup marks `active` — normally exactly one."""
    return re.findall(r'<section id="tab-([a-z]+)" class="tab-panel active"',
                      _PAGE.read_text(encoding="utf-8"))


class TestTheMarkupOpensOnOneTab(unittest.TestCase):
    def test_exactly_one_panel_is_marked_active(self):
        """Two would make "the tab it opens on" meaningless, none would hide the page."""
        self.assertEqual(len(opening_tabs()), 1, opening_tabs())

    def test_the_search_finds_the_markup_it_judges(self):
        """Guards the guard: a pattern matching nothing would pass every case here."""
        self.assertTrue(_PAGE.exists())
        self.assertIn("tab-panel", _PAGE.read_text(encoding="utf-8"))


class TestTheScriptActivatesItOnBoot(unittest.TestCase):
    def test_the_boot_activates_the_tab_rather_than_trusting_the_class(self):
        script = _APP.read_text(encoding="utf-8")
        self.assertIn("activateTheTabTheMarkupOpensOn", script)

    def test_the_opening_tab_is_read_from_the_markup_and_not_hardcoded(self):
        """A name written into the script would rot the day the markup opens elsewhere."""
        script = _APP.read_text(encoding="utf-8")
        boot = script.split("activateTheTabTheMarkupOpensOn", 1)[1][:600]
        self.assertIn('classList.contains("active")', boot)
        for tab in opening_tabs():
            self.assertNotIn(f'activateTab("{tab}")', boot)

    def test_the_opening_tab_has_a_loader_behind_activateTab(self):
        """The whole point: the tab the page opens on must be one `activateTab` fills."""
        script = _APP.read_text(encoding="utf-8")
        body = script.split("function activateTab(", 1)[1].split("\n  }", 1)[0]
        for tab in opening_tabs():
            with self.subTest(tab=tab):
                self.assertIn(f'name === "{tab}"', body)


if __name__ == "__main__":
    unittest.main()
