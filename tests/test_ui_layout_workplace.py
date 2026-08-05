"""F192: the "Layout" tab is a workplace, not a control panel.

Thirteen controls held the top of the tab. Two of them are needed every single time —
WHERE the collection goes and BY WHAT it is grouped — and the other eleven are answered
once and then never touched again. That is the same split F133 made on the run screen,
and the same remedy: the two questions stay in the open, the rest goes behind the gear.

Nothing is removed. The set of controls before and after this feature is the same set;
what changed is where each one sits. `_F182_CONTROLS` below is that "before" inventory,
taken off the tab as it stood at the end of F182, and the first test in `TestNothingWasLost`
is the one that matters most here — a feature that tidies a screen by quietly dropping
half of it has not tidied anything.

Are the layout and the albums mixed up on this tab? Asked by the brief, and the answer is
no, not on the screen: albums are gathered from `.album-controls` boxes, and every one of
them lives on "Review" or "Slices" (`TestLayoutIsNotAlbums`). They ARE mixed in the module
behind it — `sorta/ui/layout.py` holds `_validate_album_payload`, `_album_dest`,
`_album_report_to_json`, `_clusters_payload` and `_events_payload`, none of which the
canon uses — but that is a question about a file, not about a workplace, and F182 drew
those boundaries a week ago. What the screen owed the reader instead was the difference
said out loud, because the criterion now offers "by person" and an album can be gathered
"by person" too: one moves the originals, the other builds links beside them.
"""
from __future__ import annotations

import json
import unittest
from html.parser import HTMLParser

from sorta import i18n, ui
from sorta.sorter import MODES
from sorta.ui.strings import _UI_STRINGS

from tests.test_ui_sort import SortTestBase, _poll_until
from tests.test_ui_tabs import PersonEventTestBase


class _ChildCollector(HTMLParser):
    """The direct children of the element carrying `wanted_id`, as (tag, id, classes).

    A real parser rather than a `str.split` on "</div>": the whole claim of this feature
    is about the shape of the tree at the top of the tab, and slicing a string cannot
    tell a child from a grandchild — which is exactly the distinction being tested.
    """

    _VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input",
                       "link", "meta", "param", "source", "track", "wbr"})

    def __init__(self, wanted_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self._wanted = wanted_id
        self._depth: int | None = None   # None until the wanted element is open
        self.children: list[tuple[str, str, tuple[str, ...]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: (value or "") for name, value in attrs}
        if self._depth is None:
            if values.get("id") == self._wanted:
                self._depth = 0
            return
        if self._depth == 0:
            self.children.append(
                (tag, values.get("id", ""), tuple(values.get("class", "").split())))
        if tag not in self._VOID:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._depth is None or tag in self._VOID:
            return
        self._depth -= 1
        if self._depth < 0:
            self._depth = None  # the wanted element closed; ignore the rest of the page


def children_of(html: str, element_id: str) -> list[tuple[str, str, tuple[str, ...]]]:
    parser = _ChildCollector(element_id)
    parser.feed(html)
    return parser.children


class LayoutMarkupTestBase(unittest.TestCase):
    def setUp(self):
        self.html = ui._render_index_html("en")
        self.tab = self.html.split('id="tab-layout"', 1)[1].split("</section", 1)[0]

    def block(self, element_id: str) -> str:
        """The markup of one block of the tab, by the id it opens with."""
        return self.tab.split(f'id="{element_id}"', 1)[1]

    def options(self) -> str:
        return self.tab.split('id="layout-options"', 1)[1].split(
            'id="city-selection-controls"', 1)[0]

    def desk(self) -> str:
        return self.tab.split('id="layout-desk"', 1)[1].split('class="layout-run"', 1)[0]


class TestTheTwoThings(LayoutMarkupTestBase):
    """Test 1, the main one: the foreground holds exactly two things."""

    def test_the_desk_holds_exactly_two_fields(self):
        fields = [child for child in children_of(self.html, "layout-desk")
                  if "layout-field" in child[2]]
        self.assertEqual([child[1] for child in fields],
                         ["layout-field-where", "layout-field-by"])
        self.assertEqual(len(children_of(self.html, "layout-desk")), 2)

    def test_where_is_a_path_and_by_is_the_criterion(self):
        where = self.tab.split('id="layout-field-where"', 1)[1].split(
            'id="layout-field-by"', 1)[0]
        self.assertIn('id="sort-dest"', where)
        self.assertIn('id="sort-browse-btn"', where)
        by = self.tab.split('id="layout-field-by"', 1)[1].split("</div>\n<div", 1)[0]
        self.assertIn('id="layout-by"', by)

    def test_the_tab_itself_has_no_other_working_surface(self):
        """The composition of the top-level nodes: the two-field desk, the row that
        starts a layout, the progress of one, the block behind the gear, the selection
        bar that only exists while frames are ticked, and the tree. Anything else
        appearing here is a control that went back to the first plane."""
        found = [(child[1] or "." + " ".join(child[2]))
                 for child in children_of(self.html, "tab-layout")]
        self.assertEqual(found, [
            "layout-review-warning",
            "layout-desk",
            ".layout-run",
            "sort-progress",
            ".process-actions",     # the contextual cancel button
            "sort-status",
            "sort-warning",
            "layout-options",
            "city-selection-controls",
            "tree-city",
        ])

    def test_the_criterion_offers_the_three_the_engine_supports(self):
        by = self.tab.split('id="layout-by"', 1)[1].split("</select>", 1)[0]
        values = [part.split('"', 1)[0] for part in by.split('<option value="')[1:]]
        self.assertEqual(values, list(MODES))


class TestSettingsBehindTheGear(LayoutMarkupTestBase):
    """Test 3: what moved off the desk is one click away, not gone."""

    def test_the_gear_opens_a_block_that_starts_closed(self):
        self.assertIn('id="layout-options-btn"', self.tab)
        self.assertIn('id="layout-options" class="layout-options" hidden', self.tab)
        self.assertIn('id="layout-options-close"', self.tab)
        self.assertIn("function toggleLayoutOptions", self.html)
        # opened by the button that says so, closed by both buttons — a panel that only
        # opens is a panel that stays open for good
        self.assertIn(
            'toggleLayoutOptions(document.getElementById("layout-options").hidden);',
            self.html)
        self.assertIn("toggleLayoutOptions(false);", self.html)

    def test_the_gear_says_what_is_behind_it(self):
        self.assertIn("aria-controls=\"layout-options\"", self.tab)
        self.assertIn('aria-expanded="false"', self.tab)
        self.assertIn('.setAttribute("aria-expanded", open ? "true" : "false")',
                      self.html)

    def test_each_thing_behind_the_gear_carries_its_own_heading(self):
        options = self.options()
        for key in ("layout_transfer_title", "layout_corrections_title",
                    "layout_places_title", "layout_tree_title"):
            with self.subTest(key=key):
                self.assertIn(ui._t(key, "en"), options)


# The controls of the tab as F182 left it — the "before" side of "nothing was lost".
# `sort-mode` is a radio NAME (two inputs share it) and the two tree buttons are reached
# by class, which is why the inventory is written as raw attribute fragments rather than
# as ids.
_F182_CONTROLS = (
    'id="layout-review-warning"',
    'id="layout-review-goto-btn"',
    'id="sort-dest"',
    'id="sort-browse-btn"',
    'name="sort-mode" value="move"',
    'name="sort-mode" value="copy"',
    'id="sort-apply-btn"',
    'id="sort-cancel-btn"',
    'id="sort-progress"',
    'id="sort-status"',
    'id="sort-warning"',
    'id="sort-empty-hint"',
    'class="btn btn-ghost expand-all-btn"',
    'class="btn btn-ghost collapse-all-btn"',
    'id="city-override-exclude-btn"',
    'id="city-override-count"',
    'id="city-override-target"',
    'id="city-override-move-btn"',
    'id="city-override-clear-btn"',
    'id="override-status"',
    'id="city-place-picker"',
    'id="place-status"',
    'id="city-selection-controls"',
    'id="city-delete-selected-btn"',
    'id="city-delete-selected-count"',
    'id="tree-city"',
)

# Which of them stopped being on the first plane. The other half of the same claim: the
# set is unchanged, the placement is not.
_MOVED_BEHIND_THE_GEAR = (
    'name="sort-mode" value="move"',
    'name="sort-mode" value="copy"',
    'class="btn btn-ghost expand-all-btn"',
    'class="btn btn-ghost collapse-all-btn"',
    'id="city-override-exclude-btn"',
    'id="city-override-target"',
    'id="city-override-move-btn"',
    'id="city-override-clear-btn"',
    'id="city-place-picker"',
)


class TestNothingWasLost(LayoutMarkupTestBase):
    """Test 2, and the important one: the same set of controls, in other places."""

    def test_every_control_of_the_old_tab_is_still_on_it(self):
        for control in _F182_CONTROLS:
            with self.subTest(control=control):
                self.assertIn(control, self.tab)

    def test_no_control_was_duplicated_while_being_moved(self):
        """A knob copied instead of moved gives two truths about one value.

        Counted inside the tab: the two tree buttons are reached by class and the "Moves"
        tab has a pair of its own, which is not this tab having two."""
        for control in _F182_CONTROLS:
            with self.subTest(control=control):
                self.assertEqual(self.tab.count(control), 1)

    def test_what_left_the_first_plane_is_behind_the_gear(self):
        options = self.options()
        for control in _MOVED_BEHIND_THE_GEAR:
            with self.subTest(control=control):
                self.assertIn(control, options)
                self.assertNotIn(control, self.desk())

    def test_the_route_the_gear_controls_reach_did_not_change(self):
        """The corrections and the places are the same two endpoints as before — the
        feature moved markup, not behaviour."""
        self.assertIn('postJson("/api/overrides"', self.html)
        self.assertIn('postJson("/api/place"', self.html)


class TestLayoutIsNotAlbums(LayoutMarkupTestBase):
    """The brief's suspicion, checked: is the moving of files mixed with the building
    of a view? On this tab, no — but the difference is now stated, because the criterion
    offers "by person" and so does an album."""

    def test_no_album_control_is_on_the_layout_tab(self):
        """The word "album" is on the tab exactly once — in the sentence that says
        albums are built somewhere else. No box a button could be built into."""
        self.assertNotIn("album-controls", self.tab)
        self.assertNotIn("album-gather", self.tab)
        self.assertEqual(self.tab.lower().count("album"), 1)

    def test_every_album_box_belongs_to_review_or_slices(self):
        boxes = ("review-album", "search-album", "query-album", "face-album",
                 "animals-album", "junk-album")
        review = self.html.split('id="tab-review"', 1)[1].split("</section", 1)[0]
        slices = self.html.split('id="tab-slices"', 1)[1].split("</section", 1)[0]
        for box in boxes:
            with self.subTest(box=box):
                self.assertTrue(f'id="{box}"' in review or f'id="{box}"' in slices)
                self.assertNotIn(box, self.tab)

    def test_the_tab_says_which_action_moves_files_and_which_does_not(self):
        hint = ui._t("layout_moves_hint", "en")
        self.assertIn(hint, self.tab)
        self.assertIn("moves the files themselves", hint)
        self.assertIn("moves nothing", hint)

    def test_the_criterion_says_what_shape_of_folders_it_produces(self):
        """"By person" on this tab is a canon of person folders, not an album of links —
        so each value explains the tree it builds, in the field where it is chosen."""
        self.assertIn('I18N["layout_by_hint_" + this.value]', self.html)
        for mode in MODES:
            with self.subTest(mode=mode):
                self.assertIn(f"layout_by_hint_{mode}", self.html)


class TestTheCaptionsAreInTheCatalogue(unittest.TestCase):
    """Test 5: three languages, through `strings.py`, like every other caption."""

    KEYS = (
        "layout_where_title", "layout_by_title",
        "layout_by_city", "layout_by_person", "layout_by_event",
        "layout_by_hint_city", "layout_by_hint_person", "layout_by_hint_event",
        "layout_moves_hint", "layout_options_button", "layout_options_title",
        "layout_transfer_title", "layout_transfer_hint", "layout_corrections_title",
        "layout_places_title", "layout_tree_title",
    )

    def test_every_new_caption_exists_in_all_three_languages(self):
        for key in self.KEYS:
            with self.subTest(key=key):
                entry = _UI_STRINGS.get(key)
                self.assertIsNotNone(entry, f"нет ключа {key}")
                assert entry is not None
                for lang in i18n._LANGS:
                    self.assertTrue(entry.get(lang, "").strip(),
                                    f"{key}: пустой перевод {lang}")

    def test_the_rendered_page_carries_them_in_the_chosen_language(self):
        for lang in i18n._LANGS:
            html = ui._render_index_html(lang)
            with self.subTest(lang=lang):
                for key in self.KEYS:
                    self.assertIn(_UI_STRINGS[key][lang], html)
                # and no placeholder was left unsubstituted
                self.assertNotIn("{{layout_", html)


class TestTheCriterionIsValidated(unittest.TestCase):
    """`by` on the body of `POST /api/sort` — the field the criterion travels in."""

    def test_an_absent_criterion_still_means_the_city_layout(self):
        """Every client written before F192 sends no `by` and means "city" — the one
        criterion the web app could apply. A default is what keeps that true."""
        self.assertEqual(
            ui._validate_sort_payload({"dest": "/d", "mode": "move"}),
            ("/d", "move", "city"))

    def test_each_criterion_the_engine_supports_is_accepted(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                self.assertEqual(
                    ui._validate_sort_payload({"dest": "", "mode": "copy", "by": mode}),
                    (None, "copy", mode))

    def test_an_unknown_criterion_is_a_bad_request(self):
        for bad in ("cities", "", None, 1, True, "CITY"):
            with self.subTest(by=bad):
                self.assertIsNone(
                    ui._validate_sort_payload({"dest": "/d", "mode": "move", "by": bad}))

    def test_the_transfer_mode_is_still_its_own_question(self):
        """`by` must not be able to answer move-or-copy, and vice versa."""
        self.assertIsNone(ui._validate_sort_payload({"dest": "/d", "mode": "person"}))
        self.assertIsNone(ui._validate_sort_payload({"dest": "/d", "mode": "move",
                                                     "by": "move"}))


class ByCriterionTestBase(SortTestBase, PersonEventTestBase):
    """A server with one photo that has a city AND a named face, so the two criteria
    produce visibly different plans."""

    def summary(self, query: str) -> tuple[int, dict]:
        status, body, _ctype = self.get("/api/sort/summary" + query)
        return status, json.loads(body)


class TestTheCriterionReachesThePlan(ByCriterionTestBase):
    def test_the_summary_answers_about_the_criterion_it_was_asked_for(self):
        fid, _path, _content = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.add_face(fid, label="Alice")
        self.start_server()
        dest = str(self.root / "dest")
        status_city, city = self.summary(f"?dest={dest}&by=city")
        status_person, person = self.summary(f"?dest={dest}&by=person")
        self.assertEqual((status_city, status_person), (200, 200))
        self.assertEqual(city["mode"], "city")
        self.assertEqual(person["mode"], "person")
        self.assertEqual(city["files"], person["files"])   # the same one frame

    def test_an_absent_criterion_summarizes_the_city_plan(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        _status, data = self.summary("?dest=" + str(self.root / "dest"))
        self.assertEqual(data["mode"], "city")

    def test_an_unknown_criterion_is_refused_rather_than_guessed(self):
        self.start_server()
        status, data = self.summary("?dest=x&by=nonsense")
        self.assertEqual(status, 400)
        self.assertIn("error", data)


class TestApplyingAnotherCriterion(ByCriterionTestBase):
    """The whole point of the field: the button lays the collection out by what the tab
    is showing. Before F192 it always called the engine with "city", whatever was on
    screen — there was only ever one thing on screen, and now there are three."""

    def test_a_person_layout_puts_the_frame_in_a_person_folder(self):
        fid, _path, _content = self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.add_face(fid, label="Alice")
        self.start_server()
        dest = self.root / "dest"
        status, resp = self.post("/api/sort",
                                 {"dest": str(dest), "mode": "copy", "by": "person"})
        self.assertEqual(status, 200)
        self.assertTrue(resp.get("ok"))
        final = _poll_until(self.sort_status, lambda d: d["finished"])
        self.assertIsNone(final["error"])
        self.assertEqual(final["result"]["by"], "person")
        self.assertEqual(final["result"]["moved"], 1)
        copied = [p for p in dest.rglob("a.jpg")]
        self.assertEqual(len(copied), 1, list(dest.rglob("*")))
        # the folder is the person, not the city — that is the criterion doing its job
        self.assertIn("Alice", copied[0].parts)
        self.assertNotIn("Moscow", copied[0].parts)

    def test_a_bad_criterion_never_starts_a_layout(self):
        self.add_photo_file("a.jpg", country="ru", city="Moscow")
        self.start_server()
        status, resp = self.post(
            "/api/sort", {"dest": str(self.root / "dest"), "mode": "copy", "by": "city2"})
        self.assertEqual(status, 400)
        self.assertIn("error", resp)
        self.assertFalse(self.sort_status()["running"])


if __name__ == "__main__":
    unittest.main()
