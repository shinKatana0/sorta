"""F145 in the web app: the master switch on screen, and no live button for a dead action.

Three reports off the live run of 2026-08-02, all of them the same mistake — the
interface offering something the server will not do:

* the options that ask the VLM stayed clickable and priced with the "Deep analysis"
  checkbox clear, so the budget on the run screen promised hours the run would not spend;
* every settings field and every data-changing button stayed live DURING a run. The
  server answered 409 and always had, but you found that out by clicking;
* the overview block drew a stub with a button while the index was empty and swapped it
  for the full set of counters the moment it was not — which happens in the middle of a
  run, right after `index`, and everything below it moved down the page.

The client is a string of JS inside this module and there is no engine here to run it, so
what is pinned is the source: the specific shapes that are the behaviour. The server half
— which routes refuse to write while something runs — is exercised for real over HTTP,
because that one is a race between two writers and not a matter of how a button looks.
"""
from __future__ import annotations

import inspect
import re
import unittest
from unittest import mock

from sorta import ui

from tests import waiting
from tests.test_ui import UiServerTestBase


class MarkupCase(unittest.TestCase):
    """The rendered page, read as text."""

    @classmethod
    def setUpClass(cls):
        cls.html = ui._render_index_html("ru")

    def body(self, name: str) -> str:
        """Source of a JS function declaration, up to its closing brace."""
        start = self.html.index(f"function {name}(")
        depth = 0
        for j in range(self.html.index("{", start), len(self.html)):
            if self.html[j] == "{":
                depth += 1
            elif self.html[j] == "}":
                depth -= 1
                if depth == 0:
                    return self.html[start:j + 1]
        raise AssertionError(f"тело {name} не найдено")


class TestSubordinateOptionsFollowTheMaster(MarkupCase):
    """Brief requirement 3: shown, dead, and priced at zero."""

    def test_every_vlm_option_is_named_as_subordinate(self):
        """They are named in one list, so a new one cannot reach the screen without a
        decision about which switch owns it.

        F161 is what that sentence was written for: the deep junk tier was the master
        switch's own effect and therefore in no list at all, and giving it a line of its
        own (`process-products-checkbox`) meant declaring which switch owns it. F186 took
        three entries OUT of the same list — the quality question, the scope that chose
        who it was asked of and the keeper question — so the list is checked exactly
        rather than for what it contains, and a retired control cannot linger in it.

        F204 added the two the list could not have caught: the screenshot rescue and the
        landmark check were subordinate on the server and on no screen at all, so there
        was no control for this list to hold. Now there is one each, and they are in it.
        """
        listed = self.html[self.html.index("var VLM_SUBORDINATE_IDS = ["):]
        listed = listed[:listed.index("];")]
        self.assertEqual(set(re.findall(r'"([\w-]+)"', listed)),
                         {"process-products-checkbox", "process-pets-verify-checkbox",
                          "process-junk-rescue-checkbox",
                          "process-landmarks-verify-checkbox"})
        self.assertIn("VLM_SUBORDINATE_IDS.forEach",
                      self.body("updateVlmSubordinatesDisabled"))

    def test_the_master_is_the_deep_analysis_checkbox(self):
        self.assertIn('document.getElementById("process-deep-checkbox").checked',
                      self.body("vlmMasterOn"))

    def test_they_go_dead_and_are_not_hidden(self):
        """A vanished option reads as "there is no such feature", and there is one."""
        body = self.body("updateVlmSubordinatesDisabled")
        # F222 gave one of the four a second parent — the landmark check goes dead with
        # the stage it checks as well as with the master — so the condition is a function
        # of the control rather than the master alone. The run is still the other reason.
        self.assertIn("el.disabled = dead || processRunning", body)
        self.assertIn("var dead = subordinateOff(id)", body)
        self.assertNotIn("style.display", body.split(".vlm-off-hint")[0])

    def test_the_reason_is_written_next_to_each_of_them(self):
        body = self.body("updateVlmSubordinatesDisabled")
        self.assertIn('document.querySelectorAll(".vlm-off-hint")', body)
        self.assertIn('el.style.display = off ? "" : "none"', body)
        # One caption per subordinate option, and there are four of them: the product
        # line (F161), the animal check, and the screenshot rescue and the landmark check
        # F204 brought onto the screen. Two others went with the questions F186 retired.
        self.assertEqual(self.html.count('class="process-toggle-hint cost-hint '
                                         'vlm-off-hint"'), 4)

    def test_their_price_is_zero_and_not_the_old_number(self):
        """The estimate has to add up to what the run will do — a dash would say
        "unknown" and the old number would say "this evening is gone"."""
        # F222 folded the two ways a line does not run into one predicate — the master
        # is off (including "the tier it needs is not installed"), or the stage it is a
        # question about is not in this run.
        self.assertIn("if (row.vlm && !vlmMasterOn()) return false;",
                      self.body("rowRuns"))
        self.assertIn("if (!rowRuns(row)) return 0;", self.body("costSeconds"))
        # ...and it is spelled out rather than shown as "almost free", which is what a
        # stage that runs and is cheap gets
        self.assertIn(": !rowRuns(row) ? I18N.costs_off : formatCost(seconds)",
                      self.body("renderCosts"))
        self.assertIn("0", ui._UI_STRINGS["costs_off"]["ru"])

    def test_every_vlm_priced_line_is_marked_as_one(self):
        """The priced lines that cost MODEL time carry `vlm: true`; the ones that do not
        — the base pass, faces, events, the CLIP pet group — must not, or the master
        switch would zero out a price it has nothing to do with.

        `deep` is on neither side of that list: it IS the master switch, and F161 marks
        it as such so that it zeroes itself rather than being zeroed.
        """
        rows = self.html[self.html.index("var COST_ROWS = ["):]
        rows = rows[:rows.index("];")]
        entries = {}
        for match in re.finditer(r"\{ key: \"(\w+)\"(.*?)\}", rows, re.S):
            entries[match.group(1)] = "vlm: true" in match.group(2)
        self.assertEqual({key for key, marked in entries.items() if marked},
                         {"products", "pets_verify", "junk_rescue", "landmarks_verify"})
        self.assertNotIn("vlm: true", rows.split('{ key: "deep"', 1)[1].split("}", 1)[0])

    def test_the_sum_is_recomputed_whenever_the_master_moves(self):
        # F222: through the pass that applies every reason a control can be dead, which
        # ends in `updateVlmSubordinatesDisabled` — one place, three reasons.
        self.assertIn("updateOptionAvailability();", self.body("renderCosts"))
        self.assertIn("updateVlmSubordinatesDisabled();",
                      self.body("updateOptionAvailability"))
        self.assertIn('"process-deep-checkbox", "process-products-checkbox"', self.html)


class TestNoSmartSwitchingOn(MarkupCase):
    """Brief requirement 7: one movement, one consequence."""

    def test_nothing_ever_ticks_the_master_for_the_user(self):
        """A subordinate option must not switch deep analysis on by being clicked, and
        the only line in the whole client that sets that checkbox is the one applying
        the config defaults on load."""
        assignments = re.findall(
            r'document\.getElementById\("process-deep-checkbox"\)\.checked\s*=[^;]+;',
            self.html)
        self.assertEqual(
            assignments,
            ['document.getElementById("process-deep-checkbox").checked = !!data.deep;'])

    def test_the_master_does_not_tick_anything_below_it(self):
        """Switching deep analysis ON grants permission, it does not hand out orders.

        Until 2026-08-09 this read "nothing in the disabling path touches `.checked`",
        which is the rule one letter too wide: a subordinate that cannot act was left
        ticked and greyed, and the owner read that as switched on. Unticking it is not
        the failure F145 was written against — that failure is the master turning
        something ON that the person never chose. So what is pinned now is the direction:
        the path may set `checked` to false, and may put back only the value it
        remembered, and there is no literal `checked = true` anywhere in it.
        """
        body = self.body("updateVlmSubordinatesDisabled")
        self.assertNotIn("checked = true", body)
        for assignment in re.findall(r"\.checked\s*=\s*([^;]+);", body):
            with self.subTest(assignment=assignment.strip()):
                self.assertIn(assignment.strip(),
                              ("false", 'el.dataset.wanted === "1"'))


class TestSettingsAreFrozenDuringARun(MarkupCase):
    """Brief requirement 4: the ban the server already enforces, made visible."""

    def test_every_settings_control_is_disabled_while_busy(self):
        body = self.body("updateBusyControlsDisabled")
        self.assertIn("SETTING_CONTROLS.forEach", body)
        self.assertIn("el.disabled = busy", body)
        self.assertIn("folder-lang-select", body)

    def test_the_reason_is_the_string_that_already_existed(self):
        self.assertIn(ui._UI_STRINGS["settings_busy"]["ru"], self.html)

    def test_the_controls_come_back_without_a_reload(self):
        """`= busy`, never a one-way disable: the status poll runs after the run ends
        and puts everything back."""
        self.assertIn("updateBusyControlsDisabled();",
                      self.body("renderProcessStatus"))
        self.assertNotIn("disabled = true", self.body("updateBusyControlsDisabled"))


class TestEverythingThatWritesIsFrozenDuringARun(MarkupCase):
    """Brief requirement 5: wider than the settings — every action that changes data."""

    def test_saving_the_whole_set_of_choices_is_dead(self):
        self.assertIn("dupes-save-all-btn", self.body("updateBusyControlsDisabled"))

    def test_the_review_marks_are_dead(self):
        self.assertIn("uiBusy() || n === 0", self.body("refreshReviewControls"))

    def test_the_trash_and_the_way_back_are_dead(self):
        self.assertIn("uiBusy() || n === 0", self.body("refreshJunkControls"))
        self.assertIn("uiBusy() || n === 0", self.body("wireBulkDelete"))

    def test_gathering_an_album_is_dead(self):
        self.assertIn('document.querySelectorAll(".album-gather-btn")',
                      self.body("updateBusyControlsDisabled"))
        # every one of the album buttons starts from the same state, because the box
        # that holds it is rebuilt on demand and may well be rebuilt mid-run. Seven
        # builders since F156: named people, events, animals, a typed query, the shared
        # row the class buckets and the quality slices are drawn with, `renderFaceAlbum
        # Controls` (which the three face slices have to themselves because the slice they
        # gather is chosen by a parameter rather than by which panel is open) — and the
        # gather row of a PINNED query, which is a slice like any other and so offers the
        # same album as the rest of them.
        # Counting one assignment per button broke the moment F193 unified them, and the
        # thing it guarded got STRONGER: every album button now carries
        # `album-gather-btn` and is swept by class, so a button added tomorrow is
        # frozen without anybody remembering to add a line. Assert the sweep and the
        # class, not how many times a string occurs.
        self.assertIn('document.querySelectorAll(".album-gather-btn")', self.html)
        self.assertIn("btn.disabled = busy;", self.html)
        self.assertGreater(self.html.count("album-gather-btn"), 1)

    def test_applying_the_layout_is_dead(self):
        self.assertIn("applyBtn.disabled = busy || planCount === 0",
                      self.body("updateBusyControlsDisabled"))

    def test_each_group_says_why(self):
        """A caption per block of controls — the run options, the layout, the two sets
        of layout marks, the settings column, the folder language, the duplicate save,
        the review marks and the junk restore, and since F156 the search line's "pin as a
        slice" and the pin/arrows of a pinned slice (three writes of `config.yaml`, which
        the server refuses mid-run) — plus the ones the album boxes build for themselves
        as they are drawn."""
        self.assertEqual(self.html.count('busy-hint" style="display:none"'), 11)
        self.assertIn(ui._UI_STRINGS["actions_busy"]["ru"], self.html)
        self.assertIn('document.querySelectorAll(".busy-hint")',
                      self.body("updateBusyControlsDisabled"))

    def test_a_selection_driven_button_has_both_rules_in_one_place(self):
        """Otherwise the status tick and the selection handler take turns undoing each
        other — which is exactly the bug F135 fixed for the second run button."""
        self.assertIn("busyRefreshers.forEach", self.body("updateBusyControlsDisabled"))
        self.assertIn("busyRefreshers.push(fn)", self.body("registerBusyRefresh"))


class TestTheOverviewBlockHoldsItsHeight(MarkupCase):
    """Brief requirement 6: numbers arriving change the text, not the layout."""

    def test_the_empty_state_draws_the_same_rows_with_dashes(self):
        self.assertIn("overviewEmpty = !!data.empty", self.body("renderOverview"))
        self.assertIn('if (overviewEmpty) return "\\u2014";', self.body("overviewStat"))

    def test_the_four_cards_are_built_either_way(self):
        """No early return before them any more — that early return WAS the jump.

        The four calls used to sit in `renderOverview` and F190 moved them into
        `overviewGroups`, which is why this asserts through it rather than at it. That
        is the STRONGER claim, not a weaker one: the skeleton and the loaded state both
        go through that one builder, so they cannot drift apart by editing one of them.
        """
        body = self.body("renderOverview")
        self.assertIn("overviewGroups(data)", body)
        self.assertNotIn("return;", body)
        groups = self.body("overviewGroups")
        for card in ("overviewCollectionCard", "overviewPlaceCard",
                     "overviewClassesCard", "overviewLayoutCard"):
            self.assertIn(card + "(data)", groups)
        # The loading state builds its cards with the same function and no other.
        self.assertIn("overviewGroups(overviewSkeletonData())",
                      self.body("renderOverviewSkeleton"))

    def test_the_stub_and_its_button_are_gone(self):
        self.assertNotIn("overview-start-btn", self.html)
        self.assertNotIn("overview_empty_button", self.html)

    def test_the_caption_says_the_numbers_are_still_coming(self):
        self.assertIn("if (overviewEmpty) body.appendChild("
                      "overviewNote(I18N.overview_empty));", self.body("renderOverview"))

    def test_the_percentage_is_not_glued_to_a_dash(self):
        """"— (0%)" is a number pretending to be known."""
        self.assertIn("overviewEmpty ? overviewStat(p.no_place)",
                      self.body("overviewPlaceCard"))


class TestTheNewStringsAreTranslated(unittest.TestCase):
    KEYS = ("actions_busy", "process_needs_deep_hint", "overview_empty",
            "settings_busy", "costs_off")

    def test_every_string_exists_in_all_three_languages(self):
        for key in self.KEYS:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang, value in entry.items():
                    self.assertTrue(value.strip(), f"{key}/{lang} пуст")


def _post_routes() -> set[str]:
    """Every path the POST dispatcher answers, read out of its source.

    Read rather than listed by hand on purpose: the point of the case below is that a
    route ADDED to the server cannot quietly skip the busy guard, and a hand-written copy
    of the list would go stale in exactly the same way.
    """
    src = inspect.getsource(ui)
    start = src.index("def _dispatch_post(self, path: str) -> None:")
    end = src.index("def _serve_index(", start)
    return set(re.findall(r'path == "([^"]+)"', src[start:end]))


class TestEveryWriteRouteIsAccountedFor(unittest.TestCase):
    """Brief test 9, the static half: no POST route without a decision about it."""

    def test_every_post_route_is_either_guarded_or_deliberately_exempt(self):
        self.assertEqual(
            _post_routes(),
            ui.BUSY_REFUSED_ROUTES | ui._BUSY_EXEMPT_ROUTES,
            "новый POST-маршрут: внесите его в BUSY_REFUSED_ROUTES или, если он ничего "
            "не пишет, в _BUSY_EXEMPT_ROUTES")

    def test_the_exempt_ones_are_only_the_ones_that_stop_work(self):
        self.assertEqual(
            ui._BUSY_EXEMPT_ROUTES,
            {"/api/process/cancel", "/api/sort/cancel", "/api/undo/cancel",
             "/api/browse"})

    def test_the_two_halves_of_the_guard_do_not_overlap(self):
        self.assertEqual(ui._BUSY_SELF_GUARDED_ROUTES & ui._BUSY_GUARDED_ROUTES, set())


# A body each route accepts, so that a 409 is not confused with a 400 from a validator
# that runs before the guard. The dispatcher-guarded routes never reach their validator
# while busy, but they are given a real body anyway — the case asserts a REFUSAL, and a
# refusal only means something if the request would otherwise have been carried out.
_BODIES: dict[str, object] = {
    "/api/dupes/choice": {"group": [1], "keep_file_id": 1},
    "/api/dupes/choices": {"groups": [{"group": [1], "keep_file_id": 1}]},
    "/api/dupes/skip": {"group": [1]},
    "/api/dupes/trash": {"group": [1], "keep_file_id": 1},
    "/api/review/mark": {"file_ids": [1], "action": "delete"},
    "/api/review/restore": {"file_id": 1},
    "/api/animals/mark": {"file_ids": [1], "action": "add"},
    "/api/photo/trash": {"file_id": 1},
    "/api/photos/trash": {"file_ids": [1]},
    "/api/overrides": {"file_ids": [1], "action": "exclude"},
    "/api/place": {"kind": "city", "selector": "Moscow", "action": "clear"},
    "/api/clusters/label": {"cluster_id": 1, "name": "Ann"},
    "/api/clusters/merge": {"source_id": 1, "target_id": 2},
    "/api/album": {"kind": "animal", "mode": "hardlink"},
    "/api/source-tree/excludes": {"root": ".", "scan": [], "sort": []},
    # F244: a dry run, which is what the button sends first — and still a refusal while
    # busy, because the route that answers it is the one that would rewrite every path.
    "/api/relocate": {"old_prefix": "/old", "new_prefix": "/new"},
    "/api/saved-slices/pin": {"name": "mountains", "query": "mountains"},
    "/api/saved-slices/unpin": {"slice": "children"},
    "/api/saved-slices/move": {"slice": "children", "delta": 1},
    "/api/process": {"source_dir": "."},
    "/api/process/rerun-optional": {"faces": True},
    "/api/process/reset": {},
    "/api/cache/clear": {"target": "geo"},
    "/api/config/language": {"language": "en"},
    "/api/settings": {"vlm.workers": 3},
    "/api/sort": {"dest": None, "mode": "move"},
    "/api/undo": {},
    # F209: without `confirm` — the body the button sends first, and the one this case is
    # about. Confirmed, the same route DOES close the program mid-run, which is the
    # decision it asks the person to make; see tests/test_quit_from_the_interface.py.
    "/api/quit": {},
}


class TestEveryWriteRouteRefusesWhileBusy(UiServerTestBase):
    """Brief test 9, the live half: the race, not the look of a button.

    A layout in flight is used as the "something is running" state — it is the cheapest
    of the three to fake and the guard does not distinguish them. What every route must
    answer is 409, because the pipeline and the layout rewrite `media_class`,
    `frame_quality` and `places` wholesale and a second writer over those tables loses.
    """

    def setUp(self):
        super().setUp()
        state = ui._SortState()
        self.assertTrue(state.try_start())
        patcher = mock.patch.object(ui, "_SortState", return_value=state)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.start_server()

    def post(self, path: str, payload: object) -> tuple[int, dict]:
        """One POST, retried once if the SOCKET fails rather than the server.

        Measured 2026-08-05: this test failed about one run in five, on a different route
        each time, and the reason was never the answer — it was
        `ConnectionAbortedError: [WinError 10053]`, the threaded server and the client
        racing on connection teardown on Windows. Twenty-six requests in a row over
        fresh connections is simply enough tries for that to land somewhere.

        The retry does NOT soften the claim: the assertion still demands 409, and an
        answer of any other status is returned as it came. What is retried is a
        connection that never carried an answer at all.
        """
        for attempt in (1, 2):
            try:
                answer = waiting.post_json(f"{self.base_url}{path}", payload)
            except (ConnectionError, TimeoutError):
                if attempt == 2:
                    raise
            else:
                return answer.status, answer.json()
        raise AssertionError("unreachable")

    def test_every_route_answers_409(self):
        for path in sorted(ui.BUSY_REFUSED_ROUTES):
            with self.subTest(route=path):
                status, resp = self.post(path, _BODIES[path])
                self.assertEqual(status, 409, f"{path} ответил {status}: {resp}")

    def test_nothing_reached_the_database(self):
        """The refusal is a refusal to WRITE and not a refusal to answer: not one of the
        requests above may leave a row behind."""
        for path in sorted(ui.BUSY_REFUSED_ROUTES):
            self.post(path, _BODIES[path])
        for table in ("dedup_choice", "manual_overrides", "manual_places",
                      "manual_pet", "group_keeper"):
            with self.subTest(table=table):
                self.assertIsNone(
                    self.conn.execute(f"SELECT 1 FROM {table}").fetchone())

    def test_cancelling_still_works(self):
        """The exempt routes are the ones a person reaches for while something runs."""
        status, _resp = self.post("/api/sort/cancel", {})
        self.assertEqual(status, 200)


class TestTheRoutesComeBackWhenNothingRuns(UiServerTestBase):
    """The other half of "without reloading the page": the guard is a state, not a latch."""

    def test_a_write_route_answers_normally_once_the_layout_is_over(self):
        state = ui._SortState()
        self.assertTrue(state.try_start())
        with mock.patch.object(ui, "_SortState", return_value=state):
            self.start_server()

            def post():
                return waiting.post_json(f"{self.base_url}/api/overrides",
                                         {"file_ids": [1], "action": "exclude"}).status

            self.assertEqual(post(), 409)
            state.finish(None, None)
            self.assertEqual(post(), 200)


if __name__ == "__main__":
    unittest.main()
