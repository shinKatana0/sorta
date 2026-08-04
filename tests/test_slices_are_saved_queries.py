"""F151: a slice is a saved query — the pinned "Children" / "Products" / "Animals by query".

The measurement this feature exists for (2026-08-02, 200 frames out of 22 096, labelled by
hand, and the first time RECALL was measured rather than the precision of the top): the
hand-written filters find 6% of the blurred frames, 33% of the animals, 0% of the products,
and children have no filter at all — while the SAME vectors, asked in words, give 61% / 65%
/ 60% at the same depth and 89% / 95% / 87% at twice it. So the tests below are about the
properties that make a slice a QUERY rather than a sixth filter:

* the words come from `features.saved_slices` — a config entry, so an edit changes the
  answer and no code moves with it;
* the ensemble is ONE direction. Three phrases average into a vector none of them has, or
  the list of phrases is decoration;
* nothing is ever a threshold. The list is ranked, deterministic, and walked by depth —
  which is the one lever of completeness the measurement confirmed;
* an index that cannot rank gives the REASON, never an empty slice. That rule matters more
  here than in the search line: nobody typed anything, so an empty list would read as a
  fact about the person's own archive;
* the estimate is captioned apart from the exact slice next to it. `pets` (71%, checked by
  a model) and "animals by query" (60%, checked by nobody) are two slices of one archive,
  and with one label a reader takes the second for the first;
* people are NOT a query: `faces` answers that exactly and for free (F152), and blurred is
  deliberately left out too — 100% precision against 36%.

No model is loaded anywhere: the fake text tower of `tests.test_ui_search` and the two-line
encoder of `tests.test_search` are what the injectable encoder of F129 is for.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path

import numpy as np

from sorta import search, ui
from sorta.config import (
    DEFAULT_SAVED_SLICES,
    Config,
    FeaturesConfig,
    SavedSlice,
    load_config,
)

from tests.test_search import SearchTestBase, encoder_for, unit
from tests.test_ui_search import SearchUiTestBase

# The three phrases of a slice, pointing at three different directions — the fixture the
# ensemble tests are built on, because an average is only visible when the parts differ.
_PHRASES = ("a photo of a child", "children playing", "a photo of a kid at a party")
_DIRECTIONS = {
    _PHRASES[0]: unit(1.0, 0.0, 0.0),
    _PHRASES[1]: unit(0.0, 1.0, 0.0),
    _PHRASES[2]: unit(0.0, 0.0, 1.0),
}


class TestTheEnsembleIsOneDirection(SearchTestBase):
    """`search.encode_queries` — several phrases in, one unit vector out."""

    def test_the_ensemble_is_a_unit_vector(self):
        vec = search.encode_queries(_PHRASES, encoder_for(_DIRECTIONS))
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=6)

    def test_three_phrases_are_not_the_first_of_them(self):
        """Requirement three: the average has to be an average.

        If the ensemble came back as one of its phrases (or as the last one to be
        encoded), the list in the config would be decoration and the slice would be
        ranked by whichever line somebody wrote first.
        """
        ensemble = search.encode_queries(_PHRASES, encoder_for(_DIRECTIONS))
        for phrase in _PHRASES:
            with self.subTest(phrase=phrase):
                single = search.encode_query(phrase, encoder_for(_DIRECTIONS))
                self.assertFalse(np.allclose(ensemble, single, atol=1e-3))
        # and it is the direction of all three together
        expected = np.mean([_DIRECTIONS[p] for p in _PHRASES], axis=0)
        expected = expected / float(np.linalg.norm(expected))
        np.testing.assert_allclose(ensemble, expected, atol=1e-6)

    def test_one_phrase_is_the_query_itself(self):
        # A slice may hold a single phrase (one is as good as three on the measurement),
        # and then the ensemble must not bend it anywhere.
        vec = search.encode_queries(["snow"], encoder_for({"snow": unit(0.0, 1.0)}))
        np.testing.assert_allclose(vec, unit(0.0, 1.0), atol=1e-6)

    def test_a_long_answer_from_the_tower_does_not_outweigh_its_neighbours(self):
        """Each phrase is normalized BEFORE the mean, so weight is not length."""
        def encode(texts):
            return np.stack([unit(1.0) * 100.0 if t == "loud" else unit(0.0, 1.0)
                             for t in texts])

        vec = search.encode_queries(["loud", "quiet"], encode)
        np.testing.assert_allclose(vec, search.encode_queries(
            ["quiet", "loud"], encode), atol=1e-6)
        self.assertAlmostEqual(float(vec[0]), float(vec[1]), places=5)

    def test_the_whole_slice_reaches_the_tower_in_one_call(self):
        seen: list[list[str]] = []

        def encode(texts):
            seen.append(list(texts))
            return np.stack([unit(1.0) for _ in texts])

        search.encode_queries(("  a photo of a child  ", "children playing"), encode)
        self.assertEqual(seen, [["a photo of a child", "children playing"]])

    def test_a_slice_with_no_words_is_refused_rather_than_ranked(self):
        # A pinned query with nothing in it would rank the collection by an arbitrary
        # direction and look exactly like an answer.
        for texts in ((), ("", "   ")):
            with self.subTest(texts=texts):
                with self.assertRaises(ValueError):
                    search.encode_queries(texts, encoder_for({}))

    def test_a_zero_answer_is_not_divided_by_zero(self):
        vec = search.encode_queries(
            ["a", "b"], lambda texts: np.zeros((len(texts), 8), np.float32))
        self.assertEqual(float(np.linalg.norm(vec)), 0.0)


class TestThePinnedRankingIsTheSearchRanking(SearchTestBase):
    """`search.rank_queries` — the same engine, asked by the config's words."""

    def cfg_with(self, *slices: SavedSlice, page: int = 200) -> Config:
        return Config(database=self.cfg.database,
                      features=FeaturesConfig(saved_slices=slices, search_page=page))

    def test_the_order_is_the_closeness_to_the_ensemble_and_is_repeatable(self):
        near = self.add_photo(unit(1.0, 1.0, 1.0))     # the ensemble's own direction
        middle = self.add_photo(unit(1.0, 1.0, 0.0))
        far = self.add_photo(unit(0.0, 0.0, -1.0))
        cfg = self.cfg_with(SavedSlice("children", _PHRASES))
        first = search.rank_queries(cfg, self.conn, _PHRASES,
                                    encoder=encoder_for(_DIRECTIONS))
        second = search.rank_queries(cfg, self.conn, _PHRASES,
                                     encoder=encoder_for(_DIRECTIONS))
        self.assertEqual([fid for fid, _s in first.hits], [near, middle, far])
        self.assertEqual(first.hits, second.hits)

    def test_the_page_defaults_to_the_configured_size(self):
        for i in range(5):
            self.add_photo(unit(1.0, 0.1 * i))
        page = search.rank_queries(self.cfg_with(page=2), self.conn, _PHRASES,
                                   encoder=encoder_for(_DIRECTIONS))
        self.assertEqual(len(page.hits), 2)
        self.assertEqual(page.total, 5)
        self.assertTrue(page.has_more)

    def test_the_windows_tile_the_ranking_without_gaps_or_repeats(self):
        for i in range(6):
            self.add_photo(unit(1.0, 0.1 * i))
        cfg = self.cfg_with()
        whole = [fid for fid, _s in search.rank_queries(
            cfg, self.conn, _PHRASES, limit=99,
            encoder=encoder_for(_DIRECTIONS)).hits]
        paged: list[int] = []
        for offset in (0, 2, 4):
            paged.extend(fid for fid, _s in search.rank_queries(
                cfg, self.conn, _PHRASES, limit=2, offset=offset,
                encoder=encoder_for(_DIRECTIONS)).hits)
        self.assertEqual(whole, paged)
        self.assertEqual(len(set(paged)), len(paged))

    def test_an_empty_index_is_a_reason_and_not_an_empty_list(self):
        self.add_photo(None)   # a photograph with no vector stored for it
        with self.assertRaises(search.EmbeddingsMissing) as caught:
            search.rank_queries(self.cfg_with(), self.conn, _PHRASES,
                                encoder=encoder_for(_DIRECTIONS))
        self.assertEqual(caught.exception.reason, search.REASON_EMPTY)

    def test_vectors_of_another_model_never_enter_a_pinned_slice(self):
        self.add_photo(unit(1.0), model="OtherNet-B/laion")
        with self.assertRaises(search.EmbeddingsMissing) as caught:
            search.rank_queries(self.cfg_with(), self.conn, _PHRASES,
                                encoder=encoder_for(_DIRECTIONS))
        self.assertEqual(caught.exception.reason, search.REASON_OTHER_MODEL)


class TestTheSlicesLiveInTheConfig(unittest.TestCase):
    """Requirement one: the phrases are DATA. An edit of `config.yaml` is the whole API."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.yaml"

    def load(self, body: str) -> Config:
        self.path.write_text(body, encoding="utf-8")
        return load_config(str(self.path))

    def names(self, cfg: Config) -> list[str]:
        return [s.name for s in cfg.features.saved_slices]

    def test_the_three_measured_slices_ship_by_default(self):
        self.assertEqual([s.name for s in DEFAULT_SAVED_SLICES],
                         ["children", "products", "animals"])
        for slice_ in DEFAULT_SAVED_SLICES:
            with self.subTest(slice=slice_.name):
                self.assertTrue(slice_.queries)

    def test_the_phrases_that_ship_are_english(self):
        # They go to a CLIP text tower and not to a reader, and the measured numbers were
        # produced by English wording — a translated default would be a slice nobody has
        # measured, silently.
        for slice_ in DEFAULT_SAVED_SLICES:
            for phrase in slice_.queries:
                with self.subTest(phrase=phrase):
                    self.assertTrue(phrase.isascii(), phrase)

    def test_neither_people_nor_blurred_is_a_pinned_query(self):
        """Requirement six and the exclusion of requirement six's neighbour.

        People come off the `faces` table — 7 341 frames against an estimate of 6 080, a
        fact rather than a ranking — and the blurred filter is 100% precise on the sample
        against 36% for the query, so folding it into one would sink the exact signal.
        """
        forbidden = ("people", "person", "faces", "portrait", "group",
                     "blurred", "blur", "sharpness")
        for name in [s.name for s in DEFAULT_SAVED_SLICES]:
            with self.subTest(name=name):
                self.assertNotIn(name, forbidden)

    def test_a_file_of_its_own_replaces_the_defaults(self):
        cfg = self.load("features:\n"
                        "  saved_slices:\n"
                        "    snow:\n"
                        "      - \"a photo of snow\"\n"
                        "      - \"a snowy street\"\n")
        self.assertEqual(self.names(cfg), ["snow"])
        self.assertEqual(cfg.features.saved_slices[0].queries,
                         ("a photo of snow", "a snowy street"))

    def test_the_order_of_the_file_is_the_order_of_the_pins(self):
        cfg = self.load("features:\n"
                        "  saved_slices:\n"
                        "    b: [\"one\"]\n"
                        "    a: [\"two\"]\n")
        self.assertEqual(self.names(cfg), ["b", "a"])

    def test_one_phrase_may_be_written_as_a_plain_string(self):
        cfg = self.load("features:\n  saved_slices:\n    snow: \"a photo of snow\"\n")
        self.assertEqual(cfg.features.saved_slices[0].queries, ("a photo of snow",))

    def test_an_empty_mapping_pins_nothing_and_is_not_the_default(self):
        # "Pin nothing" is a real wish and must survive, the `vlm.exclude_classes` rule:
        # absence and emptiness are different answers.
        self.assertEqual(self.load("features:\n  saved_slices: {}\n")
                         .features.saved_slices, ())

    def test_a_slice_with_no_usable_phrase_is_dropped_and_the_rest_survive(self):
        cfg = self.load("features:\n"
                        "  saved_slices:\n"
                        "    empty: []\n"
                        "    snow: [\"a photo of snow\"]\n")
        self.assertEqual(self.names(cfg), ["snow"])

    def test_garbage_falls_back_to_the_defaults_rather_than_emptying_the_row(self):
        for body in ("features:\n  saved_slices: nope\n",
                     "features:\n  saved_slices: 7\n",
                     "features:\n  saved_slices:\n    broken: []\n"):
            with self.subTest(body=body):
                self.assertEqual(self.load(body).features.saved_slices,
                                 DEFAULT_SAVED_SLICES)

    def test_the_example_config_documents_the_slices_at_their_defaults(self):
        example = (Path(__file__).resolve().parent.parent
                   / "config.example.yaml").read_text(encoding="utf-8")
        self.assertIn("saved_slices:", example)
        cfg = load_config(str(Path(__file__).resolve().parent.parent
                              / "config.example.yaml"))
        self.assertEqual(cfg.features.saved_slices, DEFAULT_SAVED_SLICES)


class SavedSliceUiTestBase(SearchUiTestBase):
    """The UI server of the search tests, asked through `/api/saved-slices`."""

    def slice_(self, name: str = "", extra: str = "") -> dict:
        query = f"?slice={urllib.parse.quote(name)}" if name else "?"
        status, body, ctype = self.get(f"/api/saved-slices{query}{extra}")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        return json.loads(body)

    def pin(self, *slices: SavedSlice) -> None:
        """Re-pin the LIVE config — what editing `features.saved_slices` does."""
        self.cfg.features = dataclasses.replace(self.cfg.features,
                                                saved_slices=slices)

    def ids(self, data: dict) -> list[int]:
        return [it["file_id"] for it in data["items"]]


class TestThePinnedSliceRanks(SavedSliceUiTestBase):
    def test_the_pins_are_listed_without_ranking_anything(self):
        # The call the tab makes on open: a row of pins costs a list and no model.
        self.add_indexed_photo("a.jpg", unit(1.0))
        self.start_server()
        data = self.slice_()
        self.assertEqual([s["slice"] for s in data["slices"]],
                         ["children", "products", "animals"])
        self.assertIsNone(data["slice"])
        self.assertEqual(data["items"], [])
        self.assertEqual(self.encoded, [])

    def test_a_pinned_slice_answers_with_a_ranked_list_in_a_stable_order(self):
        """Requirement one, and the first test of the brief: ranked and deterministic."""
        self.pin(SavedSlice("children", _PHRASES))
        self.vectors.update(_DIRECTIONS)
        near = self.add_indexed_photo("near.jpg", unit(1.0, 1.0, 1.0))
        middle = self.add_indexed_photo("middle.jpg", unit(1.0, 1.0, 0.0))
        far = self.add_indexed_photo("far.jpg", unit(0.0, 0.0, -1.0))
        self.start_server()
        first = self.slice_("children")
        self.assertEqual(self.ids(first), [near, middle, far])
        scores = [it["score"] for it in first["items"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(self.ids(self.slice_("children")), self.ids(first))

    def test_the_phrases_travel_with_the_answer(self):
        self.pin(SavedSlice("children", _PHRASES))
        self.start_server()
        self.assertEqual(self.slice_("children")["queries"], list(_PHRASES))

    def test_editing_the_config_changes_the_list_without_touching_code(self):
        """Requirement two: the phrases are data, and the running server reads them."""
        self.vectors.update({"a photo of snow": unit(0.0, 1.0),
                             "a photo of a cake": unit(1.0, 0.0)})
        snow = self.add_indexed_photo("snow.jpg", unit(0.0, 1.0))
        cake = self.add_indexed_photo("cake.jpg", unit(1.0, 0.0))
        self.pin(SavedSlice("mine", ("a photo of snow",)))
        self.start_server()
        self.assertEqual(self.ids(self.slice_("mine")), [snow, cake])
        # the same slice, one line of the config later
        self.pin(SavedSlice("mine", ("a photo of a cake",)))
        data = self.slice_("mine")
        self.assertEqual(self.ids(data), [cake, snow])
        self.assertEqual(data["queries"], ["a photo of a cake"])

    def test_three_phrases_do_not_answer_as_one_of_them(self):
        """Requirement three, through the route: the ensemble is really averaged."""
        self.vectors.update(_DIRECTIONS)
        first = self.add_indexed_photo("first.jpg", unit(1.0, 0.0, 0.0))
        blend = self.add_indexed_photo("blend.jpg", unit(1.0, 1.0, 1.0))
        self.pin(SavedSlice("one", (_PHRASES[0],)),
                 SavedSlice("three", _PHRASES))
        self.start_server()
        self.assertEqual(self.ids(self.slice_("one")), [first, blend])
        self.assertEqual(self.ids(self.slice_("three")), [blend, first])

    def test_an_unknown_slice_is_refused_rather_than_answered_with_a_void(self):
        self.start_server()
        status, body, _ctype = self.get("/api/saved-slices?slice=nope")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_a_broken_window_is_a_400_like_every_other_paged_route(self):
        self.start_server()
        for extra in ("&offset=nope", "&offset=-1", "&limit=-1", "&limit=nope"):
            with self.subTest(extra=extra):
                status, body, _ctype = self.get(
                    f"/api/saved-slices?slice=children{extra}")
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(body))

    def test_a_sensitive_class_is_ranked_but_never_decoded(self):
        # F133's rule, and a pinned slice must not become the way around it.
        document = self.add_indexed_photo("passport.jpg", unit(1.0))
        photo = self.add_indexed_photo("cake.jpg", unit(1.0, 0.1))
        self.classify(document, "document")
        self.start_server()
        items = {it["file_id"]: it for it in self.slice_("children")["items"]}
        self.assertNotIn("thumb_url", items[document])
        self.assertIn("thumb_url", items[photo])
        self.assertIn("score", items[document])


class TestAnUnfilledIndexIsAReason(SavedSliceUiTestBase):
    """The fourth test of the brief, and the rule this feature could most easily break.

    Nobody typed "children" into a pin, so an empty list there is not a query that missed
    — it reads as "your archive holds no photographs of children", which is a conclusion
    about somebody's own collection drawn from a table nobody filled.
    """

    def test_an_empty_index_gives_the_state_and_not_an_empty_slice(self):
        self.add_photo_file("a.jpg")
        self.start_server()
        data = self.slice_("children")
        self.assertEqual(data["state"], "empty")
        self.assertFalse(data["available"])
        self.assertEqual(data["items"], [])
        self.assertEqual(data["total"], 0)
        self.assertFalse(data["has_more"])
        self.assertEqual(self.encoded, [])   # and no model was loaded to say so

    def test_vectors_of_another_model_are_the_other_reason(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.store_vector(fid, unit(1.0), model="OtherNet-B/laion")
        self.start_server()
        data = self.slice_("children")
        self.assertEqual(data["state"], "other_model")
        self.assertFalse(data["available"])
        self.assertEqual(data["index_model"], "OtherNet-B/laion")
        self.assertEqual(data["items"], [])

    def test_the_pins_are_still_listed_while_nothing_can_be_ranked(self):
        # A pin that hides itself never gets to say why it is empty (the F152 rule).
        self.add_photo_file("a.jpg")
        self.start_server()
        self.assertEqual(len(self.slice_("children")["slices"]), 3)

    def test_the_pin_row_waits_for_a_photograph_and_not_for_a_result(self):
        # The F152 rule: a pin appears as soon as the index holds a frame — its empty
        # state is a sentence and it has to be reachable to say it — while over an index
        # with no photographs at all there is nothing to say and "no slices yet" is true.
        self.start_server()
        _status, body, _ctype = self.get("/")
        self.assertIn("savedSlices = (data.photos ? data.slices : []) || [];",
                      body.decode("utf-8"))
        self.assertEqual(self.slice_()["photos"], 0)

    def test_the_client_shows_the_reason_instead_of_an_empty_grid(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("return data.available ? I18N.search_no_frames "
                      ": searchStateText(data);", html)


class TestDepthIsTheLever(SavedSliceUiTestBase):
    """The seventh test of the brief: the page comes from the config and continues."""

    def add_descending(self, n: int) -> list[int]:
        return [self.add_indexed_photo(f"a{i}.jpg", unit(1.0, 0.1 * i))
                for i in range(n)]

    def test_a_request_without_a_limit_opens_to_the_configured_page(self):
        self.add_descending(5)
        self.cfg.features = dataclasses.replace(self.cfg.features, search_page=2)
        self.start_server()
        data = self.slice_("children")
        self.assertEqual(len(data["items"]), 2)
        self.assertEqual(data["limit"], 2)
        self.assertTrue(data["has_more"])

    def test_show_more_continues_the_same_ranking_without_gaps_or_repeats(self):
        self.add_descending(6)
        self.start_server()
        whole = self.ids(self.slice_("children", extra="&limit=99"))
        paged = (self.ids(self.slice_("children", extra="&limit=2"))
                 + self.ids(self.slice_("children", extra="&limit=2&offset=2"))
                 + self.ids(self.slice_("children", extra="&limit=2&offset=4")))
        self.assertEqual(whole, paged)
        self.assertEqual(len(set(paged)), len(paged))

    def test_the_counter_states_the_length_of_the_ranking(self):
        self.add_descending(5)
        self.start_server()
        data = self.slice_("children", extra="&limit=2")
        self.assertEqual(data["total"], 5)
        self.assertEqual(data["offset"], 0)

    def test_the_end_of_the_list_hides_the_button(self):
        self.add_descending(4)
        self.start_server()
        last = self.slice_("children", extra="&limit=2&offset=2")
        self.assertFalse(last["has_more"])

    def test_the_button_is_the_shared_pager_and_the_prominent_one(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("var queryPager = makePager(", html)
        self.assertEqual(html.count("function makePager("), 1)
        button = [ln for ln in html.splitlines() if 'id="query-more-btn"' in ln]
        self.assertEqual(len(button), 1)
        # Depth is the one measured lever of completeness, so the control that turns it is
        # not the quietest thing on the screen.
        self.assertIn("btn-primary", button[0])
        hint = html.index('id="query-depth-hint"')
        self.assertLess(html.index('id="query-more-btn"'), hint)


class TestTheEstimateIsCaptionedApart(SavedSliceUiTestBase):
    """The fifth test of the brief: two animal slices, both visible, labelled apart."""

    def test_the_pet_label_and_the_query_are_two_different_slices(self):
        self.add_indexed_photo("a.jpg", unit(1.0))
        self.start_server()
        animals = json.loads(self.get("/api/animals")[1])
        query = self.slice_("animals")
        self.assertIn("animals", animals)          # the F123 route, untouched
        self.assertIn("queries", query)            # and the ranking beside it
        self.assertNotIn("queries", animals)
        self.assertTrue(query["approximate"])

    def test_both_pins_are_built_and_their_labels_differ(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('pins.push({ key: "query:" + s.slice', html)
        self.assertIn('pins.push({ key: "animal", label: I18N.tab_animal });', html)
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                exact = ui._t("tab_animal", lang)
                approximate = ui._t("query_slice_pin", lang).replace(
                    "{name}", ui._t("query_slice_animals", lang))
                self.assertNotEqual(exact, approximate)
                self.assertIn(exact, approximate)   # the same subject, marked

    def test_the_panel_says_the_list_is_an_estimate_and_the_face_panel_says_it_is_not(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertNotEqual(ui._t("query_slice_intro", lang),
                                    ui._t("face_slices_intro", lang))
                self.assertNotEqual(ui._t("query_slice_intro", lang),
                                    ui._t("animals_intro", lang))

    def test_a_card_of_a_ranked_slice_carries_its_score(self):
        # The number is the only thing that explains the order, and a slice built by a
        # detector has none to show — which is the difference the captions state.
        fid = self.add_indexed_photo("a.jpg", unit(1.0))
        self.start_server()
        item = self.slice_("children")["items"][0]
        self.assertEqual(item["file_id"], fid)
        self.assertIn("score", item)
        face = json.loads(self.get("/api/face-slices?slice=people")[1])
        self.assertEqual(face["counts"][0]["slice"], "people")
        for card in face["items"]:
            self.assertNotIn("score", card)


class TestPeopleAreNotAQuery(SavedSliceUiTestBase):
    """The sixth test of the brief: the people slice is built on faces, not on words."""

    def test_no_pinned_query_claims_the_people_slice(self):
        self.start_server()
        names = [s["slice"] for s in self.slice_()["slices"]]
        for taken in ui.FACE_SLICES:
            with self.subTest(slice=taken):
                self.assertNotIn(taken, names)

    def test_the_people_slice_still_comes_from_the_faces_table(self):
        # A frame is in it because a detector left a box on it — no vector, no ranking,
        # no phrase in any config.
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding) VALUES (?, '[1,2,3,4]', ?)",
            (fid, b"embedding"))
        self.conn.commit()
        self.start_server()
        data = json.loads(self.get("/api/face-slices?slice=people")[1])
        self.assertEqual([it["file_id"] for it in data["items"]], [fid])
        self.assertIsNone(data["reason"])


class TestTheMarkupAndTheStrings(SavedSliceUiTestBase):
    def test_the_panel_lives_in_the_slices_tab_beside_the_others(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        panel = html.index('<div id="tab-query"')
        pins = html.index('id="slice-pins"')
        tab_end = html.index('<section id="tab-moves"')
        self.assertLess(pins, panel)
        self.assertLess(panel, tab_end)
        self.assertIn('id="query-grid"', html)
        self.assertIn('id="query-phrases"', html)
        self.assertNotIn('id="tab-btn-query"', html)   # a slice, not a tab of its own

    def test_the_client_asks_the_route_for_the_pins_and_for_a_page(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('fetch("/api/saved-slices?offset=0&limit=0")', html)
        self.assertIn('"/api/saved-slices?slice=" + encodeURIComponent(querySlice)', html)
        self.assertNotIn("SAVED_SLICE_PAGE_SIZE", html)   # the page size stays a setting

    def test_every_new_string_is_translated_three_ways(self):
        keys = ("query_slice_children", "query_slice_products", "query_slice_animals",
                "query_slice_pin", "query_slice_intro", "query_slice_phrases",
                "query_slice_shown_label", "error_loading_saved_slices")
        for key in keys:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang in ("ru", "en", "ja"):
                    self.assertTrue(entry[lang].strip())

    def test_the_captions_carry_their_placeholders(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertIn("{name}", ui._UI_STRINGS["query_slice_pin"][lang])
                self.assertIn("{phrases}", ui._UI_STRINGS["query_slice_phrases"][lang])
                shown = ui._UI_STRINGS["query_slice_shown_label"][lang]
                for token in ("{name}", "{shown}", "{total}"):
                    self.assertIn(token, shown)

    def test_the_phrases_hint_names_the_key_a_reader_has_to_edit(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertIn("features.saved_slices",
                              ui._UI_STRINGS["query_slice_phrases"][lang])

    def test_no_caption_promises_that_nothing_was_found(self):
        forbidden = ("не найдено", "ничего не найд", "nothing was found",
                     "no results", "見つかりません")
        for key in ("query_slice_intro", "query_slice_phrases"):
            for lang in ("ru", "en", "ja"):
                text = ui._UI_STRINGS[key][lang].lower()
                for phrase in forbidden:
                    with self.subTest(key=key, lang=lang, phrase=phrase):
                        self.assertNotIn(phrase, text)

    def test_i18n_reaches_the_page_in_three_languages(self):
        self.start_server()
        for lang, expected in (("ru", "Дети"), ("en", "Children"), ("ja", "子ども")):
            with self.subTest(lang=lang):
                _status, body, _ctype = self.get(f"/?lang={lang}")
                self.assertIn(expected, body.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
