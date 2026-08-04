"""F133: the interface regrouped by the three layers — Overview · Review · Layout ·
Slices.

The tabs used to be grouped by WHICH CODE computed them (geo, faces, duplicates, events,
moves). What a person needs is the other thing: what they are going to do, and there are
exactly three answers with three different relationships to the file system —

    canon   the layout by city         exactly one   a physical move, undone by journal
    slices  people/events/animals/…    any number    hardlinks: made and dropped for free
    junk    what gets thrown out       a subtraction dangerous, needs looking at

This file pins what the regrouping must not lose and the two decisions inside it that
are easy to get wrong: the warning on "Layout" is a HINT and never a lock, and a document
is counted and never rendered.

Nothing here is a new server query. The three new slices — products, screenshots,
documents — are the classifier's own buckets, served by the `/api/junk` route that has
had their counts and their no-preview rule since F103.
"""
from __future__ import annotations

import json
import unittest

from sorta import ui

from tests.test_ui import UiServerTestBase

_F133_KEYS = (
    "tab_overview", "tab_review", "tab_layout", "tab_slices", "tab_moves",
    "tab_person", "tab_event", "tab_animal", "tab_junk",
    # `slices_query_placeholder` and `slices_query_hint` were F133's own wording for a
    # field that did nothing yet; F134 replaced both with the `search_*` catalogue when it
    # made the field work. The keys are gone because the strings are, not because the
    # requirement lapsed — the translated wording is asserted by the F134 tests.
    "slices_intro", "slices_pinned_label", "slices_empty",
    "layout_review_warning", "layout_review_goto",
    "settings_open_button", "settings_close_button",
)


class ThreeLayersTestBase(UiServerTestBase):
    def classify(self, rel: str, verdict: str) -> int:
        """A file plus its `media_class` row — what §6's three slices are read from."""
        file_id, _path, _content = self.add_photo_file(rel, country="ru", city="Moscow")
        self.conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, score, updated_at, tier)
               VALUES (?, ?, 'vlm', NULL, '2026-08-02', 'vlm')""",
            (file_id, verdict))
        self.conn.commit()
        return file_id

    def blurred(self, rel: str, sharpness: float = 12.0) -> int:
        """A photograph in the "blurred" slice of the Review."""
        file_id = self.classify(rel, "photo")
        self.conn.execute(
            """INSERT INTO frame_quality (file_id, sharpness, source, updated_at)
               VALUES (?, ?, 'cheap', '2026-08-02')""", (file_id, sharpness))
        self.conn.commit()
        return file_id

    def decide(self, file_id: int, action: str = "to_delete") -> None:
        self.conn.execute(
            "INSERT INTO dedup_choice (file_id, action, updated_at) VALUES (?, ?, ?)",
            (file_id, action, "2026-08-02"))
        self.conn.commit()

    def review(self, query: str = "?slice=dupes&offset=0&limit=0") -> dict:
        status, body, ctype = self.get("/api/review" + query)
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        return json.loads(body)

    def junk(self, query: str = "") -> dict:
        _status, body, _ctype = self.get("/api/junk" + query)
        return json.loads(body)


class MarkupTestBase(ThreeLayersTestBase):
    def setUp(self):
        super().setUp()
        self.start_server()
        _status, body, _ctype = self.get("/")
        self.html = body.decode("utf-8")

    def section(self, tab_id: str) -> str:
        return self.html.split(f'id="{tab_id}"', 1)[1].split("</section", 1)[0]


class TestTheFourTabs(MarkupTestBase):
    def test_the_strip_holds_exactly_the_new_set(self):
        expected = ["overview", "review", "layout", "slices", "moves"]
        strip = self.html.split('<div class="tabs"', 1)[1].split("</div>", 1)[0]
        found = [line.split('id="tab-btn-', 1)[1].split('"', 1)[0]
                 for line in strip.splitlines() if 'id="tab-btn-' in line]
        self.assertEqual(found, expected)

    def test_the_tabs_that_were_regrouped_are_gone_from_the_strip(self):
        for old in ("process", "city", "person", "event", "animal", "junk"):
            with self.subTest(tab=old):
                self.assertNotIn(f'id="tab-btn-{old}"', self.html)

    def test_every_panel_still_exists(self):
        # "Regrouped, not rewritten": each old panel is still in the page, either as a
        # tab of its own or as a slice panel inside "Slices".
        for panel in ("tab-overview", "tab-review", "tab-layout", "tab-slices",
                      "tab-moves", "tab-person", "tab-event", "tab-animal", "tab-junk"):
            with self.subTest(panel=panel):
                self.assertIn(f'id="{panel}"', self.html)

    def test_the_js_model_of_the_strip_matches_the_markup(self):
        self.assertIn('var TAB_NAMES = ["overview", "review", "layout", "slices", '
                      '"moves"];', self.html)


class TestOverviewHoldsTheRun(MarkupTestBase):
    """§2: one question at two moments in time — "what do I have" and "what do I do
    with it". On an empty collection the tab shows the source picker; after a run, the
    state and the same button."""

    def test_the_run_controls_moved_into_the_overview_panel(self):
        overview = self.section("tab-overview")
        # `process-rerun-optional-btn` is deliberately absent: F135 collapsed the second
        # start button into one, which is the same claim this test makes — the run lives
        # on the overview — expressed with one control fewer.
        for control in ("process-source-dir", "process-browse-btn", "process-start-btn",
                        "process-cancel-btn", "process-reset-btn", "process-progress",
                        "cache-block"):
            with self.subTest(control=control):
                self.assertIn(control, overview)

    def test_the_run_checkboxes_live_here_too(self):
        overview = self.section("tab-overview")
        for box in ("process-faces-checkbox", "process-events-checkbox",
                    "process-pets-checkbox", "process-deep-checkbox",
                    "process-geo-online-checkbox"):
            with self.subTest(box=box):
                self.assertIn(box, overview)

    def test_the_state_comes_before_the_run(self):
        overview = self.section("tab-overview")
        self.assertLess(overview.index('id="overview-body"'),
                        overview.index('id="step-source"'))

    def test_it_is_the_landing_tab(self):
        self.assertIn('<section id="tab-overview" class="tab-panel active">', self.html)
        self.assertIn('class="tab-btn active" id="tab-btn-overview"', self.html)


class TestEmptyCollection(ThreeLayersTestBase):
    """§Tests/6: an empty collection gets the source picker, not a table of zeros."""

    def test_the_overview_route_reports_empty(self):
        self.start_server()
        _status, body, _ctype = self.get("/api/overview")
        self.assertTrue(json.loads(body)["empty"])

    def test_the_picker_is_on_the_same_tab_as_the_invitation(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        overview = html.split('id="tab-overview"', 1)[1].split("</section", 1)[0]
        # The invitation moved from the generic empty-state element to an overview note
        # (F138 gave the panel its own vocabulary); the string it shows is the same one.
        self.assertIn("overviewNote(I18N.overview_empty)", html)
        self.assertIn('id="process-source-dir"', overview)
        # The invitation must not send anyone to another tab — that is the §2 claim, and
        # it still holds.
        self.assertNotIn('activateTab("process")', html)
        # And it moves the caret into the picker. The call was written by F133, was lost
        # somewhere in F135/F138 while the run controls were rebuilt, and F161 put it
        # back — this is the one screen where a first-time reader has nothing else to go
        # on, and the field is right here, so nobody is taken anywhere.
        focus = html.split("if (overviewEmpty) {", 1)[1].split("}", 1)[0]
        self.assertIn('document.getElementById("process-source-dir")', focus)
        self.assertIn(".focus()", focus)


class TestSettingsBehindTheGear(MarkupTestBase):
    """§3: thirteen configuration keys stop holding a third of the screen."""

    def test_the_gear_opens_a_panel_that_starts_closed(self):
        self.assertIn('id="settings-toggle-btn"', self.html)
        self.assertIn('id="settings-panel" class="settings-panel" hidden', self.html)
        self.assertIn('id="settings-close-btn"', self.html)
        self.assertIn("function toggleSettingsPanel", self.html)

    def test_every_knob_lives_in_the_panel_and_nowhere_else(self):
        panel = self.html.split('id="settings-panel"', 1)[1].split("</aside>", 1)[0]
        layout = self.section("tab-layout")
        for key in ui._SETTINGS_SPEC:
            control = "setting-" + key.replace(".", "-").replace("_", "-")
            with self.subTest(key=key):
                self.assertIn(f'id="{control}"', panel)
                self.assertNotIn(control, layout)
                self.assertEqual(self.html.count(f'id="{control}"'), 1)

    def test_the_folder_language_travelled_with_them(self):
        panel = self.html.split('id="settings-panel"', 1)[1].split("</aside>", 1)[0]
        self.assertIn('id="folder-lang-select"', panel)

    def test_the_route_behind_it_did_not_move(self):
        # The behaviour of F104 is pinned by test_ui_settings; what matters here is that
        # the drawer talks to the same endpoint rather than to a new one.
        self.assertIn('"/api/settings"', self.html)


class TestSlicesAreSearchPlusPinned(MarkupTestBase):
    """§5: even without the search of F129 working, the block is drawn as "a query plus
    pinned slices" — otherwise the fixed row of tabs would be redrawn one feature later.
    """

    def test_there_is_a_place_for_the_query(self):
        slices = self.section("tab-slices")
        self.assertIn('id="slice-query"', slices)
        # F134 wired this field for real and moved its wording into the catalogue, so the
        # literal placeholder of the placeholder-era is gone. The PLACE is what §5 claims;
        # which words invite the query is asserted by the F134 tests, in three languages.
        field = slices.split('id="slice-query"', 1)[1].split(">", 1)[0]
        self.assertIn("placeholder=", field)

    def test_the_query_starts_disabled_until_the_index_can_answer_it(self):
        """Was "drawn and not wired" while F129 was still landing. It is wired now — but
        the field still opens disabled, because a query needs embeddings and a collection
        that has not run `junk` has none. The disabled state is now a real answer about
        the index, not a stub.

        F189: the field opens disabled all the same, and is enabled by `usable` —
        `available` OR somebody named, because a person's name is answered out of the face
        clusters and needs no embeddings. What the index can do has not changed."""
        slices = self.section("tab-slices")
        self.assertIn("disabled", slices.split('id="slice-query"', 1)[1].split(">", 1)[0])
        self.assertIn('document.getElementById("slice-query").disabled = !usable;',
                      self.html)
        self.assertIn("var usable = available || !!(state && state.names);", self.html)

    def test_the_pins_are_built_from_data_not_written_out(self):
        slices = self.section("tab-slices")
        self.assertIn('id="slice-pins"', slices)
        # no pin is spelled into the markup — the row is a function of what exists
        for key in ("person", "event", "animal", "junk-product"):
            with self.subTest(key=key):
                self.assertNotIn(f'id="slice-pin-{key}"', self.html)
        self.assertIn("function buildSlicePins", self.html)
        self.assertIn("function renderSlicePins", self.html)

    def test_the_class_slices_have_a_product_order_of_their_own(self):
        # §6: products, screenshots, documents — read in that order, whichever of them
        # happens to be biggest this month.
        self.assertIn('var SLICE_CLASS_ORDER = ["product", "screenshot", "document"];',
                      self.html)

    def test_the_panels_of_the_old_tabs_moved_in_unchanged(self):
        slices = self.section("tab-slices")
        for panel, marker in (("tab-person", "clusters-grid"),
                              ("tab-event", "events-list"),
                              ("tab-animal", "animals-grid"),
                              ("tab-junk", "junk-grid")):
            with self.subTest(panel=panel):
                self.assertIn(f'id="{panel}" class="slice-panel"', slices)
                self.assertIn(marker, slices)


class TestTheThreeNewSlices(ThreeLayersTestBase):
    """§6: products, screenshots and documents were counted since F13/F15 and visible
    nowhere. They are read from `media_class` through the route that already serves the
    classifier's buckets — no new query, no new endpoint."""

    def test_each_class_is_a_bucket_with_a_count_and_a_page(self):
        wanted = {"product": 3, "screenshot": 2, "document": 1}
        ids = {}
        for verdict, n in wanted.items():
            ids[verdict] = [self.classify(f"{verdict}{i}.jpg", verdict)
                            for i in range(n)]
        self.classify("mine.jpg", "photo")
        self.start_server()
        counts = {b["verdict"]: b["count"] for b in self.junk()["buckets"]}
        self.assertEqual(counts, wanted)
        for verdict, n in wanted.items():
            with self.subTest(verdict=verdict):
                page = self.junk("?bucket=" + verdict)
                self.assertEqual(page["total"], n)
                self.assertEqual({it["file_id"] for it in page["items"]},
                                 set(ids[verdict]))

    def test_a_personal_photo_is_in_no_slice_of_this_family(self):
        mine = self.classify("mine.jpg", "photo")
        self.classify("thing.jpg", "product")
        self.start_server()
        ids = {it["file_id"] for it in self.junk()["items"]}
        self.assertNotIn(mine, ids)

    def test_products_and_screenshots_carry_a_preview(self):
        for verdict in ("product", "screenshot"):
            with self.subTest(verdict=verdict):
                self.classify(f"{verdict}.jpg", verdict)
        self.start_server()
        for it in self.junk()["items"]:
            self.assertEqual(it["thumb_url"], f"/thumb/{it['file_id']}")

    def test_a_document_is_counted_and_never_rendered(self):
        """§6.2: among them are passports, medical forms and receipts. The product knows
        about them (`verdict='document'`) and knowing is not the same as laying them out
        on the screen — the privacy gate of F120 exists for this reason and the
        interface must not be weaker than it."""
        doc = self.classify("passport.jpg", "document")
        self.start_server()
        page = self.junk("?bucket=document")
        self.assertEqual(page["total"], 1)
        item = page["items"][0]
        self.assertEqual(item["file_id"], doc)
        self.assertNotIn("thumb_url", item)
        self.assertNotIn("video", item)

    def test_no_document_preview_leaks_through_the_unfiltered_slice(self):
        self.classify("passport.jpg", "document")
        self.classify("thing.jpg", "product")
        self.start_server()
        by_verdict = {it["verdict"]: it for it in self.junk()["items"]}
        self.assertNotIn("thumb_url", by_verdict["document"])
        self.assertIn("thumb_url", by_verdict["product"])

    def test_the_card_of_a_document_offers_no_way_to_open_it(self):
        # A card without `thumb_url` gets a stub instead of a clickable thumbnail, so it
        # never reaches the lightbox; and the grid has no "show in folder"/"download".
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        card = html.split("function renderJunkCard", 1)[1].split("\n  }", 1)[0]
        self.assertIn("if (item.thumb_url) {", card)
        self.assertIn("clickableThumb", card)
        self.assertIn('stub.className = "junk-doc-box";', card)
        self.assertNotIn("download", card)


class TestTheSensitiveSetIsConfigured(ThreeLayersTestBase):
    """Which classes are never rendered is `vlm.exclude_classes` — the list that already
    means "do not show this to the model". One visible list beats two, of which the
    second is the one that gets forgotten."""

    def exclude(self, *classes: str) -> None:
        import dataclasses
        self.cfg.vlm = dataclasses.replace(self.cfg.vlm, exclude_classes=classes)

    def test_the_route_says_which_classes_it_refuses_to_render(self):
        self.classify("passport.jpg", "document")
        self.start_server()
        self.assertEqual(self.junk()["sensitive"], ["document"])

    def test_the_default_protects_documents(self):
        self.classify("passport.jpg", "document")
        self.classify("thing.jpg", "product")
        self.start_server()
        by_verdict = {it["verdict"]: it for it in self.junk()["items"]}
        self.assertNotIn("thumb_url", by_verdict["document"])
        self.assertIn("thumb_url", by_verdict["product"])

    def test_adding_a_class_to_the_list_stops_rendering_it(self):
        self.exclude("document", "product")
        self.classify("thing.jpg", "product")
        self.start_server()
        data = self.junk()
        self.assertEqual(data["sensitive"], ["document", "product"])
        self.assertNotIn("thumb_url", data["items"][0])

    def test_emptying_the_list_lifts_the_guard_as_the_guide_says(self):
        # The cost of reusing one key: clearing it turns both protections off at once.
        # Pinned so that the consequence is a decision somebody made, not a surprise.
        self.exclude()
        self.classify("passport.jpg", "document")
        self.start_server()
        data = self.junk()
        self.assertEqual(data["sensitive"], [])
        self.assertIn("thumb_url", data["items"][0])

    def test_a_protected_class_keeps_its_counter_and_its_cards(self):
        # §6: the documents are opened in the common grid and counted there — only the
        # preview is withheld, so the count stays honest and the way back stays usable.
        doc = self.classify("passport.jpg", "document")
        self.start_server()
        data = self.junk("?bucket=document")
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["items"][0]["file_id"], doc)
        self.assertEqual(data["items"][0]["name"], "passport.jpg")


class TestReviewPendingCounters(ThreeLayersTestBase):
    """What the warning of §4 is about: the part of the Review nobody has answered."""

    def test_an_untouched_slice_is_wholly_pending(self):
        self.blurred("a.jpg")
        self.blurred("b.jpg")
        self.start_server()
        data = self.review()
        pending = {row["slice"]: row["count"] for row in data["pending"]}
        self.assertEqual(pending["blurred"], 2)
        self.assertEqual(data["pending_total"], 2)

    def test_a_decision_takes_a_frame_out_of_the_count(self):
        first = self.blurred("a.jpg")
        self.blurred("b.jpg")
        self.start_server()
        self.assertEqual(self.review()["pending_total"], 2)
        self.decide(first)
        self.assertEqual(self.review()["pending_total"], 1)

    def test_keeping_a_frame_counts_as_deciding_about_it(self):
        # "Reviewed" is a decision, not a deletion — `keep` is one.
        only = self.blurred("a.jpg")
        self.start_server()
        self.decide(only, "keep")
        self.assertEqual(self.review()["pending_total"], 0)

    def test_the_total_is_zero_on_an_empty_collection(self):
        self.start_server()
        data = self.review()
        self.assertEqual(data["pending_total"], 0)
        self.assertEqual([row["slice"] for row in data["pending"]],
                         list(ui._REVIEW_SLICES))

    def test_pending_never_exceeds_the_slice_it_is_counted_over(self):
        self.blurred("a.jpg")
        self.start_server()
        data = self.review()
        counts = {row["slice"]: row["count"] for row in data["counts"]}
        pending = {row["slice"]: row["count"] for row in data["pending"]}
        for name in ui._REVIEW_SLICES:
            with self.subTest(slice=name):
                self.assertLessEqual(pending[name], counts[name])


class TestPendingDupeGroups(unittest.TestCase):
    """The grouped slice is counted from the payload the tab already builds."""

    def test_a_group_with_no_action_is_pending(self):
        groups = [{"frames": [{"action": None}, {"action": None}]}]
        self.assertEqual(ui._pending_dupe_groups(groups), 1)

    def test_one_decided_frame_settles_the_whole_group(self):
        groups = [{"frames": [{"action": "keep"}, {"action": "to_delete"}]}]
        self.assertEqual(ui._pending_dupe_groups(groups), 0)

    def test_groups_are_counted_independently(self):
        groups = [
            {"frames": [{"action": "keep"}]},
            {"frames": [{"action": None}]},
            {"frames": [{"action": None}, {"action": "to_delete"}]},
        ]
        self.assertEqual(ui._pending_dupe_groups(groups), 1)


class TestLayoutWarns(MarkupTestBase):
    """§4: a warning about the order, and never a lock."""

    def test_the_warning_lives_on_the_layout_tab_and_starts_hidden(self):
        layout = self.section("tab-layout")
        self.assertIn('id="layout-review-warning"', layout)
        self.assertIn('id="layout-review-warning" class="layout-warning" '
                      'style="display:none"', layout)
        self.assertIn('id="layout-review-goto-btn"', layout)

    def test_it_is_asked_for_on_every_open_of_the_tab(self):
        self.assertIn('if (name === "layout") loadLayoutWarning();', self.html)
        self.assertIn('fetch("/api/review?slice=dupes&offset=0&limit=0")', self.html)

    def test_it_appears_with_pending_work_and_disappears_without(self):
        render = self.html.split("function renderLayoutWarning", 1)[1] \
                          .split("\n  }", 1)[0]
        self.assertIn('box.style.display = pending ? "" : "none";', render)

    def test_the_warning_disables_nothing(self):
        """§5 of the tests, asked for explicitly: this is the border between a hint and
        a prohibition. The collection is alive, "gather" happens again and again, and a
        person coming back for one album must not be walked through steps."""
        render = self.html.split("function renderLayoutWarning", 1)[1] \
                          .split("\n  }", 1)[0]
        loader = self.html.split("function loadLayoutWarning", 1)[1] \
                          .split("\n  }", 1)[0]
        for body in (render, loader):
            self.assertNotIn("disabled", body)
            self.assertNotIn("hidden", body)

    def test_the_pending_count_reaches_nothing_but_the_warning(self):
        """The other direction of the same border, stated where it cannot rot: the
        number the warning is built from does not appear anywhere else in the client, so
        no control can come to depend on it by accident."""
        render = "function renderLayoutWarning" + \
            self.html.split("function renderLayoutWarning", 1)[1].split("\n  }", 1)[0]
        self.assertIn("data.pending_total", render)
        self.assertEqual(self.html.count("pending_total"), 1)

    def test_the_apply_button_keeps_the_guards_it_already_had(self):
        # A destination and no run in flight — the two conditions from before F133.
        self.assertIn("function updateBusyControlsDisabled", self.html)
        self.assertRegex(
            self.html,
            r'<button[^>]*id="sort-apply-btn"[^>]*\sdisabled>')


class TestStringsAreTranslated(unittest.TestCase):
    def test_every_string_of_the_new_chrome_exists_in_all_three_languages(self):
        for key in _F133_KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} is empty")

    def test_the_warning_carries_its_number_in_every_language(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertIn("{n}", ui._UI_STRINGS["layout_review_warning"][lang])

    def test_no_key_of_a_regrouped_tab_was_dropped(self):
        # The labels of what became slices are still translated — the pins are drawn
        # from the catalogue, not from a string baked into the script.
        for key in ("tab_person", "tab_event", "tab_animal", "tab_junk",
                    "junk_bucket_product", "junk_bucket_screenshot",
                    "junk_bucket_document"):
            with self.subTest(key=key):
                self.assertEqual(set(ui._UI_STRINGS[key]), {"ru", "en", "ja"})


if __name__ == "__main__":
    unittest.main()
