"""F195: the frames of every slice stand in a grid, and it is the SAME grid.

The owner's remark of 2026-08-04: «в срезе животные сейчас все фотки растянуты — то есть
одно фото на весь ряд». The slice he was reading is the pinned query «Животные · по
запросу» (F151), and the reason it looked like that is the reason this file enumerates
instead of naming: the layout was written per grid, as an `#id` rule apiece, and the
panel added last was simply never given one. Its cards fell back to block flow — one to a
row, each stretched across the width of the panel — while the four grids that happened to
have a rule looked perfectly right.

So the fix is one class on the container (`.slice-grid`) and one on the tile
(`.slice-card`), and the tests below check the PROPERTY over the whole list of slices:

* every grid the interface keeps a selection over (F193's `makeSelection` — which is what
  makes a grid a grid of frames on this product) is laid out by the shared class, and by
  the same one, so no slice can have more or fewer cards per row than its neighbours;
* no slice keeps a layout of its own for a rule of its own to drift into;
* the tile crops the picture and never scales it along one axis — the other reading of
  "растянуты", and the one a grid rule alone would not have answered;
* an empty slice does not break the grid: "loading" / "nothing found" / "it failed" are
  sentences about the whole slice and span the row instead of standing in one column.

Nothing here starts a server: the page, the stylesheet and the script are assembled at
import (`ui._render_index_html`), which is exactly the artefact the browser is served.
"""
from __future__ import annotations

import re
import unittest

from sorta import ui

# The pair the whole feature consists of.
_GRID_CLASS = "slice-grid"
_CARD_CLASS = "slice-card"

_COMMENTS = re.compile(r"/\*.*?\*/", re.S)


def _rules(css: str) -> list[tuple[str, str]]:
    """[(selector, body)] for every rule of the stylesheet, `@media` blocks unwrapped.

    A block rather than a regex because a declaration outside its rule means nothing:
    the questions below are all of the form "what does THIS selector say", and a media
    query is still that selector saying it — on a narrower screen.
    """
    css = _COMMENTS.sub("", css)
    out: list[tuple[str, str]] = []
    start = 0
    while True:
        opened = css.find("{", start)
        if opened < 0:
            return out
        selector = css[start:opened].strip()
        depth, end = 1, opened + 1
        while depth and end < len(css):
            if css[end] == "{":
                depth += 1
            elif css[end] == "}":
                depth -= 1
            end += 1
        body = css[opened + 1:end - 1]
        if selector.startswith("@"):
            out.extend(_rules(body))
        else:
            out.append((selector, body))
        start = end


def _declarations(body: str) -> dict[str, str]:
    """The property -> value map of one rule body, the last spelling winning."""
    found: dict[str, str] = {}
    for part in body.split(";"):
        if ":" not in part:
            continue
        prop, _sep, value = part.partition(":")
        found[prop.strip()] = value.strip()
    return found


class SliceGridTestBase(unittest.TestCase):
    """The served page, taken apart into the three files it is assembled from."""

    @classmethod
    def setUpClass(cls):
        cls.html = ui._render_index_html("ru")
        cls.css = cls.html.split("<style>", 1)[1].split("</style>", 1)[0]
        cls.script = cls.html.rsplit("<script>", 1)[1].split("</script>", 1)[0]
        cls.rules = _rules(cls.css)

    def slice_grids(self) -> list[str]:
        """The id of every grid of frames the interface draws — enumerated, not listed.

        `makeSelection(<grid>)` is the one thing all of them have and nothing else has: a
        grid whose cards can be ticked and gathered is a grid of frames (F193 made that
        true of every slice), so a panel added tomorrow lands in this list by being a
        slice at all rather than by being named here.
        """
        grids = sorted(set(re.findall(r'makeSelection\("([\w-]+)"\)', self.script)))
        self.assertGreaterEqual(len(grids), 5, "no slice grids found in the script")
        return grids

    def card_classes(self) -> list[str]:
        """The card class of every slice, off the tiles the script actually builds."""
        classes = sorted(set(re.findall(
            r'className = "' + _CARD_CLASS + r' ([\w-]+)"', self.script)))
        self.assertGreaterEqual(len(classes), 5, "no slice cards found in the script")
        return classes

    def rules_for(self, selector_part: str) -> list[dict[str, str]]:
        """Every rule whose selector mentions this word, as declaration maps."""
        return [_declarations(body) for selector, body in self.rules
                if selector_part in selector]

    def tile_rules(self, card: str) -> list[dict[str, str]]:
        """The rules that style the CARD ITSELF — `.animal-card`, `.junk-card.restored`.

        One compound and no combinator: a rule about something inside the card
        (`.animal-card-actions`, `.face-card img`) is a different element and says nothing
        about the box the slice is drawn in.
        """
        whole = re.compile(r"^\." + card + r"(\.[\w-]+)*$")
        return [_declarations(body) for selector, body in self.rules
                for one in selector.split(",")
                if whole.match(one.strip())]


class TestEverySliceIsLaidOutTheSameWay(SliceGridTestBase):
    """The main test, and the one the brief asks to be written by enumeration: the animal
    slice is compared with its neighbours by walking the slices, not by naming two."""

    def grid_classes(self, grid: str) -> list[str]:
        match = re.search(r'<div id="' + grid + r'"([^>]*)>', self.html)
        self.assertIsNotNone(match, f"{grid} is not in the page")
        classes = re.search(r'class="([^"]*)"', match.group(1))
        return sorted((classes.group(1) if classes else "").split())

    def test_every_slice_grid_wears_the_one_layout_class(self):
        for grid in self.slice_grids():
            with self.subTest(grid=grid):
                self.assertIn(_GRID_CLASS, self.grid_classes(grid))

    def test_the_slices_are_laid_out_by_the_same_classes_as_each_other(self):
        # The animal slice against every other one — and the comparison holds whichever
        # of them is added next, because both sides come out of the same enumeration.
        grids = self.slice_grids()
        layouts = {grid: self.grid_classes(grid) for grid in grids}
        for grid in grids:
            with self.subTest(grid=grid):
                self.assertEqual(layouts[grid], layouts[grids[0]])

    def test_no_slice_keeps_a_layout_of_its_own(self):
        """Requirement 2: one layout, never an exception propped under one slice.

        An `#id` rule that sets the tracks is how this defect happened: four grids had
        one, the fifth did not, and the fifth is the one the owner opened.
        """
        for grid in self.slice_grids():
            with self.subTest(grid=grid):
                for found in self.rules_for("#" + grid):
                    self.assertNotIn("grid-template-columns", found)
                    self.assertNotIn("display", found)

    def test_the_number_in_a_row_is_decided_by_the_width(self):
        """Requirement 1: by the width of the panel and not by which slice is open."""
        found = self.rules_for("." + _GRID_CLASS)
        self.assertTrue(found, "the shared grid has no rule at all")
        base = found[0]
        self.assertEqual(base.get("display"), "grid")
        self.assertRegex(base.get("grid-template-columns", ""),
                         r"repeat\(auto-fill,\s*minmax\(")
        # Every restatement of the tracks (the narrow-screen one) fills by width too: a
        # media query that pinned a column count would be the same defect on a phone.
        for rule in found:
            if "grid-template-columns" in rule:
                self.assertRegex(rule["grid-template-columns"],
                                 r"repeat\(auto-fill,\s*minmax\(")

    def test_every_card_of_every_slice_is_the_shared_tile(self):
        for card in self.card_classes():
            with self.subTest(card=card):
                self.assertIn('className = "' + _CARD_CLASS + " " + card + '"',
                              self.script)

    def test_no_card_class_carries_a_tile_of_its_own(self):
        """The other half of "one layout": a slice may say what is different about its
        cards (a struck-through animal, a bucket that must not be deleted) and may not
        restate the box they are drawn in. The modifiers stay — an outline, a colour, an
        opacity are what the card MEANS; the geometry below is what it IS."""
        geometry = {"display", "padding", "border", "border-radius", "flex-direction",
                    "gap", "width", "height"}
        for card in self.card_classes():
            for found in self.tile_rules(card):
                with self.subTest(card=card, rule=sorted(found)):
                    self.assertFalse(set(found) & geometry)

    def test_the_tile_the_slices_share_is_the_one_that_declares_the_box(self):
        # The counterpart of the test above: taking the geometry off the slices is only
        # right because the shared class states it.
        found = self.tile_rules(_CARD_CLASS)
        self.assertTrue(found, "the shared tile has no rule at all")
        self.assertEqual(found[0].get("display"), "flex")
        for prop in ("padding", "border", "border-radius", "flex-direction"):
            with self.subTest(prop=prop):
                self.assertIn(prop, found[0])


class TestTheTileNeverStretchesThePicture(SliceGridTestBase):
    """Requirement 3. «Растянуты» reads two ways — one card to a row, and a picture
    pulled along one axis — and the second one is checked here by the rules rather than
    by the pixels: a fixed width AND a fixed height with no `object-fit` is exactly the
    scaling that distorts a face."""

    def test_the_tile_states_the_fit_beside_the_sizes(self):
        found = self.rules_for("." + _CARD_CLASS + " img")
        self.assertTrue(found, "the shared tile does not style its picture at all")
        for rule in found:
            self.assertEqual(rule.get("object-fit"), "cover")

    def test_no_rule_of_the_page_scales_a_picture_along_one_axis(self):
        """The property over the whole stylesheet: a picture given both sizes is either
        cropped (`cover`) or fitted (`contain`), never stretched — `fill` is the default
        and the one value that distorts, so it must not be spelled anywhere."""
        for selector, body in self.rules:
            if not selector.rstrip().endswith("img"):
                continue
            found = _declarations(body)
            with self.subTest(selector=selector):
                self.assertNotEqual(found.get("object-fit"), "fill")
                if "width" in found and "height" in found:
                    # Its own or the base `img` rule's, but there has to be one.
                    self.assertIn(found.get("object-fit", self.base_img_fit()),
                                  ("cover", "contain"))

    def base_img_fit(self) -> str:
        """What an `<img>` of this page is fitted with unless a rule says otherwise."""
        base = [found for found in self.rules_for("img") if "object-fit" in found]
        self.assertTrue(base, "no rule of the page fits a picture at all")
        return base[0]["object-fit"]


class TestAnEmptySliceDoesNotBreakTheGrid(SliceGridTestBase):
    """A slice with nothing in it says so, and the sentence is about the whole slice —
    so it spans the row instead of being squeezed into one 150px column of tracks it is
    not a tile of."""

    def test_the_state_message_spans_the_row(self):
        found = [rule for selector, body in self.rules
                 if selector.startswith("." + _GRID_CLASS) and "state-msg" in selector
                 for rule in [_declarations(body)]]
        self.assertTrue(found, "nothing tells an empty slice how wide its sentence is")
        for rule in found:
            self.assertEqual(rule.get("grid-column"), "1 / -1")

    def test_the_empty_state_is_put_into_the_grid_itself(self):
        # Which is why the rule above is needed at all: the pager appends the sentence to
        # the grid box rather than beside it, so an empty slice is a grid with one child.
        pager = self.script.split("function makePager", 1)[1].split(
            "function applyTheme", 1)[0]
        self.assertIn("box.appendChild(opts.emptyEl ? opts.emptyEl(data)", pager)

    def test_a_slice_waiting_for_its_first_page_says_so_the_same_way(self):
        # The placeholder in the markup is the same kind of sentence, so the grid does not
        # jump between "loading" and the first page.
        for grid in self.slice_grids():
            placeholder = self.html.split(f'id="{grid}"', 1)[1].split(">", 1)[1]
            if placeholder.startswith("<div"):
                with self.subTest(grid=grid):
                    self.assertTrue(placeholder.startswith('<div class="state-msg'))


if __name__ == "__main__":
    unittest.main()
