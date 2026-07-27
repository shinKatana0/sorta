"""The rendered page's JS must at least be parseable — nothing else checks it.

The whole client lives inside a Python string, so a backslash in it needs escaping
twice: `"\\\\"` in the source is what reaches the browser as `\\`. Writing `"\\"`
renders `"\"` — an unterminated string literal, which kills the ENTIRE script: no
tabs, no buttons, no folder picker. It shipped once (d13da77, the Duplicates tab
tooltip) and no test noticed, because the tests around it check server payloads and
the JS is never parsed by anything.

There is no JS engine here to parse with, so this pins the specific shapes that are
always a bug and are invisible in a Python diff.
"""
from __future__ import annotations

import re
import unittest

from sorta import ui


class TestRenderedJsSanity(unittest.TestCase):
    def setUp(self):
        self.html = ui._render_index_html("ru")

    def test_no_string_ends_in_a_lone_backslash(self):
        r'''`"\"` — a quote, one backslash, a quote. In JS the backslash escapes the
        closing quote, so the literal runs on to the next one and the parser dies at
        the end of the line. This is what a mistyped path separator looks like.'''
        self.assertNotIn('"\\"', self.html)
        self.assertNotIn("'\\'", self.html)

    def test_backslash_separators_are_doubled(self):
        """A path separator in JS is written `"\\\\"` (two characters reach the
        browser, one backslash results). Every occurrence must be even-length, an odd
        run means one of them ate the quote."""
        for run in re.findall(r'"(\\+)"', self.html):
            self.assertEqual(len(run) % 2, 0,
                             f'нечётное число обратных слешей в строке JS: "{run}"')

    def test_both_tabs_write_the_separator_the_same_way(self):
        """Cities and Duplicates build the same tooltip; the bug was that one of them
        was written with half the escaping. Pinning them to each other catches the
        next copy of this line before it reaches a browser."""
        lines = [ln.strip() for ln in self.html.splitlines() if "src_path ?" in ln]
        self.assertEqual(len(lines), 2, "ожидались две подсказки с путём (города и дубли)")
        separators = {re.search(r'\+ "(\\+)" \+', ln).group(1) for ln in lines}
        self.assertEqual(len(separators), 1,
                         f"вкладки экранируют разделитель по-разному: {separators}")


if __name__ == "__main__":
    unittest.main()
