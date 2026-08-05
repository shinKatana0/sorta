"""F200: the layout screen offers COPY first, not move.

Everything else about a layout is built so that a careless click costs nothing: the plan
is a dry run, the journal is written before the first file travels, `undo` puts them
back, blake3 says the copy is the original and nothing is ever overwritten. A default of
"move" was the one place left where the first click is irreversible in substance — undo
returns the files, but only while the journal is intact and nobody has tidied the tree by
hand. A collection laid out by copying can always have its originals deleted afterwards;
one laid out by moving has to be reconstructed from a log.

`move` is not removed, it is one click away. What changed is which answer the screen —
and the parser behind it — gives when nobody says.

The two defaults are tested together on purpose. The radio in `page.html`, the fallback
in `_validate_sort_payload` and the fallback in `app.js` are three answers to one
question, and the only failure worth guarding against is them drifting apart: a screen
that shows "copy" over a server that hears "move" is worse than either default alone.
"""
from __future__ import annotations

import unittest
from html.parser import HTMLParser

from sorta import ui

from tests.test_ui_sort import SortTestBase, _poll_until


class _RadioCollector(HTMLParser):
    """Every `input` of one radio group, as (value, is-checked), in document order."""

    def __init__(self, group: str) -> None:
        super().__init__(convert_charrefs=True)
        self._group = group
        self.radios: list[tuple[str, bool]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "input":
            return
        values = {name: value for name, value in attrs}
        if values.get("name") != self._group:
            return
        self.radios.append((values.get("value") or "", "checked" in values))


def radios_of(html: str, group: str) -> list[tuple[str, bool]]:
    parser = _RadioCollector(group)
    parser.feed(html)
    return parser.radios


class TestTheSwitchShowsCopy(unittest.TestCase):
    """The main test: what a person sees selected when the panel opens."""

    def setUp(self):
        self.html = ui._render_index_html("en")

    def test_copy_is_the_checked_option(self):
        checked = [value for value, is_checked in radios_of(self.html, "sort-mode")
                   if is_checked]
        self.assertEqual(checked, ["copy"])

    def test_move_is_still_offered_unchecked(self):
        """Not a capability removed — a default changed. The option has to be there,
        and it has to be there as an option rather than as a preselected one."""
        self.assertEqual(radios_of(self.html, "sort-mode"),
                         [("move", False), ("copy", True)])

    def test_the_switch_looks_the_same_in_every_language(self):
        for lang in ("en", "ru", "ja"):
            with self.subTest(lang=lang):
                self.assertEqual(radios_of(ui._render_index_html(lang), "sort-mode"),
                                 [("move", False), ("copy", True)])

    def test_the_script_falls_back_to_the_same_answer(self):
        """The third default. `app.js` reads the checked radio and needs a value for
        the case where none is checked; "move" there would be a screen and a script
        disagreeing about what silence means."""
        self.assertNotIn('checked ? checked.value : "move"', self.html)
        self.assertIn('checked ? checked.value : "copy"', self.html)


class TestTheParserAgreesWithTheScreen(unittest.TestCase):
    """`mode` on the body of `POST /api/sort` — the same question, asked of the server."""

    def test_an_absent_mode_means_copy(self):
        self.assertEqual(ui._validate_sort_payload({"dest": "/d"}),
                         ("/d", "copy", "city"))

    def test_an_explicit_mode_is_still_obeyed(self):
        for mode in ("move", "copy"):
            with self.subTest(mode=mode):
                self.assertEqual(ui._validate_sort_payload({"dest": "/d", "mode": mode}),
                                 ("/d", mode, "city"))

    def test_a_default_is_not_a_licence_for_nonsense(self):
        """Falling back on an absent field says nothing about a field that is present
        and wrong: those are still 400, not a silent copy."""
        for bad in ("link", "", None, 1, True, "COPY"):
            with self.subTest(mode=bad):
                self.assertIsNone(
                    ui._validate_sort_payload({"dest": "/d", "mode": bad}))


class TestABodyWithoutAModeCopies(SortTestBase):
    """And the whole way through: the default has to reach the files, not just the parse.

    `move` end to end is `TestSortMove` in test_ui_sort.py, unchanged by this feature.
    """

    def test_the_originals_survive_a_layout_nobody_named_a_mode_for(self):
        _fid, path, _content = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        dest = self.root / "dest"

        status, resp = self.post("/api/sort", {"dest": str(dest)})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))

        final = _poll_until(self.sort_status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertEqual(final["result"]["mode"], "copy")
        self.assertEqual(final["result"]["moved"], 1)

        self.assertTrue(path.exists())
        self.assertEqual(len(list(dest.rglob("*.jpg"))), 1)
        batch = self.conn.execute(
            "SELECT operation FROM move_batches ORDER BY id DESC LIMIT 1").fetchone()
        self.assertEqual(batch["operation"], "copy")


if __name__ == "__main__":
    unittest.main()
