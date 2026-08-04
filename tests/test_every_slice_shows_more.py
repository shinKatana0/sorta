"""F173: every ordered slice can be walked past its first page — search most of all.

The measurements of 2026-08-02/03 left exactly one confirmed lever of completeness: the
DEPTH of the list. Doubling it adds ~25 points on average, and on the query «дети» the
recall goes from 61% at top-N to 89% at top-2N — the second half of the ranking holds
nearly a third of what the person is looking for. Four slices had a button for that.
Search, the one slice built by a query rather than by a model's marks, did not: it stopped
at `features.search_limit` frames with a caption that read like an answer ("200 frames"),
and the handle the measurement had just called the main one did not turn.

So the tests here are about four properties, and the last of them is the one that keeps
this from happening again:

* the second page CONTINUES the first — same order, no repeats, nothing skipped. A pager
  that reranks per page is worse than no pager at all, because the reader cannot tell;
* the counter states the length of the LIST. "Showing 200" and "there are 200" read
  identically, and for a query the second is almost never true;
* the button is the server's `has_more` and disappears at the end of the list rather than
  serving an empty page;
* the mechanism is ONE mechanism. `ui._page_payload` on the server and `makePager` in the
  browser, so a slice added tomorrow gets the button by calling them — the fifth copy of
  the same twenty lines is precisely how search shipped without one.

No model is loaded anywhere: the fake text tower of `tests.test_ui_search` and the
two-line encoder of `tests.test_search` are what the injectable encoder of F129 is for.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from sorta import search, ui
from sorta.config import Config, FeaturesConfig, load_config

from tests.test_search import SearchTestBase, encoder_for, unit
from tests.test_ui import UiServerTestBase
from tests.test_ui_search import SearchUiTestBase


class TestTheEngineReturnsAWindowAndTheLength(SearchTestBase):
    """`search.rank` — a page of the ranking plus how long the ranking is."""

    def rank(self, *, limit: int, offset: int = 0) -> search.Page:
        query = search.encode_query("cake", encoder_for({}))
        return search.rank(self.conn, query, self.model, limit=limit, offset=offset)

    def add_descending(self, n: int) -> list[int]:
        """`n` photographs whose closeness to the query strictly descends."""
        return [self.add_photo(unit(1.0, 0.1 * i)) for i in range(n)]

    def test_the_total_is_the_ranking_and_not_the_page(self):
        self.add_descending(5)
        page = self.rank(limit=2)
        self.assertEqual(len(page.hits), 2)
        self.assertEqual(page.total, 5)
        self.assertTrue(page.has_more)

    def test_consecutive_windows_tile_the_ranking_without_gaps_or_repeats(self):
        self.add_descending(7)
        whole = [fid for fid, _s in self.rank(limit=99).hits]
        paged: list[int] = []
        for offset in (0, 3, 6):
            paged.extend(fid for fid, _s in self.rank(limit=3, offset=offset).hits)
        self.assertEqual(whole, paged)
        self.assertEqual(len(set(paged)), len(paged))

    def test_a_window_past_the_end_is_empty_and_says_there_is_no_more(self):
        self.add_descending(3)
        page = self.rank(limit=2, offset=3)
        self.assertEqual(page.hits, [])
        self.assertEqual(page.total, 3)
        self.assertFalse(page.has_more)

    def test_the_last_window_reports_no_more_even_when_it_is_full(self):
        self.add_descending(4)
        page = self.rank(limit=2, offset=2)
        self.assertEqual(len(page.hits), 2)
        self.assertFalse(page.has_more)

    def test_a_negative_offset_reads_from_the_top_rather_than_from_the_end(self):
        # Python would slice a negative index off the tail, which for a ranking means
        # silently answering with the WORST matches. The window starts at zero instead.
        self.add_descending(3)
        page = self.rank(limit=2, offset=-5)
        self.assertEqual([fid for fid, _s in page.hits],
                         [fid for fid, _s in self.rank(limit=2).hits])
        self.assertEqual(page.offset, 0)

    def test_the_scores_keep_descending_across_the_page_boundary(self):
        self.add_descending(6)
        scores = [s for _fid, s in self.rank(limit=3).hits]
        scores += [s for _fid, s in self.rank(limit=3, offset=3).hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_still_answers_with_a_plain_list_for_the_cli(self):
        ids = self.add_descending(3)
        query = search.encode_query("cake", encoder_for({}))
        hits = search.search(self.conn, query, self.model, 2)
        self.assertEqual([fid for fid, _s in hits], ids[:2])

    def test_rank_text_defaults_the_page_to_the_configured_size(self):
        self.add_descending(5)
        cfg = Config(database=self.cfg.database, features=FeaturesConfig(search_page=2))
        page = search.rank_text(cfg, self.conn, "cake", encoder=encoder_for({}))
        self.assertEqual(len(page.hits), 2)
        self.assertEqual(page.limit, 2)
        self.assertEqual(page.total, 5)


class TestSearchPagesPastTheFirstPage(SearchUiTestBase):
    """`GET /api/search?offset=` — the hole this feature was written to close."""

    def add_descending(self, n: int) -> list[int]:
        return [self.add_indexed_photo(f"a{i}.jpg", unit(1.0, 0.1 * i))
                for i in range(n)]

    def ids(self, data: dict) -> list[int]:
        return [it["file_id"] for it in data["items"]]

    def test_the_second_page_holds_no_frame_of_the_first(self):
        """The main test: pressing "show more" has to reach frames nobody has seen."""
        self.add_descending(5)
        self.start_server()
        first = self.ids(self.search("дети", extra="&limit=2"))
        second = self.ids(self.search("дети", extra="&limit=2&offset=2"))
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertEqual(set(first) & set(second), set())

    def test_the_pages_are_the_one_ranking_in_its_own_order(self):
        self.add_descending(6)
        self.start_server()
        whole = self.ids(self.search("дети", extra="&limit=99"))
        paged = (self.ids(self.search("дети", extra="&limit=2"))
                 + self.ids(self.search("дети", extra="&limit=2&offset=2"))
                 + self.ids(self.search("дети", extra="&limit=2&offset=4")))
        self.assertEqual(whole, paged)

    def test_the_counter_states_the_length_of_the_ranking(self):
        # "showing 200" is indistinguishable from "there are exactly 200", and for a query
        # the second is almost never true — so the answer carries both numbers.
        self.add_descending(5)
        self.start_server()
        data = self.search("дети", extra="&limit=2")
        self.assertEqual(len(data["items"]), 2)
        self.assertEqual(data["total"], 5)
        self.assertEqual(data["offset"], 0)
        self.assertTrue(data["has_more"])

    def test_the_end_of_the_list_hides_the_button_instead_of_serving_a_void(self):
        self.add_descending(4)
        self.start_server()
        last = self.search("дети", extra="&limit=2&offset=2")
        self.assertEqual(len(last["items"]), 2)
        self.assertFalse(last["has_more"])
        past = self.search("дети", extra="&limit=2&offset=4")
        self.assertEqual(past["items"], [])
        self.assertFalse(past["has_more"])
        self.assertEqual(past["total"], 4)

    def test_a_broken_window_is_a_400_like_every_other_paged_route(self):
        self.start_server()
        for extra in ("&offset=nope", "&offset=-1", "&limit=-1", "&limit=nope"):
            with self.subTest(extra=extra):
                status, body, _ctype = self.get(f"/api/search?q=x{extra}")
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(body))

    def test_an_unavailable_index_still_answers_in_the_paged_shape(self):
        # The client draws the same controls whatever state it is in; with nothing to rank
        # there is simply nothing below, and no button.
        self.add_photo_file("a.jpg")
        self.start_server()
        data = self.search("дети")
        self.assertFalse(data["available"])
        self.assertEqual(data["items"], [])
        self.assertEqual(data["total"], 0)
        self.assertFalse(data["has_more"])

    def test_the_coverage_line_keeps_its_own_denominator(self):
        # Two different totals live in this answer: the length of the ranking and the size
        # of the collection. Naming them apart is what keeps "showing 2 of 5" from turning
        # into "showing 2 of every photograph you own".
        self.add_descending(3)
        self.add_photo_file("not_indexed.jpg")
        self.start_server()
        data = self.search("дети", extra="&limit=2")
        self.assertEqual(data["photos"], 4)
        self.assertEqual(data["indexed"], 3)
        self.assertEqual(data["total"], 3)


class TestThePageSizeComesFromTheConfig(SearchUiTestBase):
    """`features.search_page` — a page, and the old spelling of it."""

    def test_a_request_without_a_limit_opens_to_the_configured_page(self):
        for i in range(5):
            self.add_indexed_photo(f"a{i}.jpg", unit(1.0, 0.1 * i))
        self.cfg.features = dataclasses.replace(self.cfg.features, search_page=2)
        self.start_server()
        data = self.search("дети")
        self.assertEqual(len(data["items"]), 2)
        self.assertEqual(data["limit"], 2)
        self.assertTrue(data["has_more"])

    def test_the_setting_reaches_the_screen_without_being_copied_into_js(self):
        # The client asks without a `limit` on purpose: a page size repeated in JS is a
        # second copy of the setting, and the copy is the one that goes stale.
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('"/api/search?q=" + encodeURIComponent(searchQuery)', html)
        self.assertNotIn("SEARCH_PAGE_SIZE", html)


class TestTheOldConfigKeyKeepsWorking(unittest.TestCase):
    """A rename must not take somebody's setting with it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.yaml"

    def load(self, body: str):
        self.path.write_text(body, encoding="utf-8")
        return load_config(str(self.path))

    def test_the_new_key_is_read(self):
        self.assertEqual(self.load("features:\n  search_page: 40\n")
                         .features.search_page, 40)

    def test_the_old_key_still_sets_the_page(self):
        self.assertEqual(self.load("features:\n  search_limit: 40\n")
                         .features.search_page, 40)

    def test_the_new_key_wins_when_both_are_given(self):
        self.assertEqual(
            self.load("features:\n  search_limit: 40\n  search_page: 7\n")
            .features.search_page, 7)

    def test_neither_key_is_the_default(self):
        self.assertEqual(self.load("features: {}\n").features.search_page,
                         FeaturesConfig().search_page)

    def test_garbage_under_the_old_key_is_the_default_rather_than_a_crash(self):
        self.assertEqual(self.load("features:\n  search_limit: nope\n")
                         .features.search_page, FeaturesConfig().search_page)

    def test_the_example_config_documents_the_new_spelling(self):
        example = (Path(__file__).resolve().parent.parent
                   / "config.example.yaml").read_text(encoding="utf-8")
        self.assertIn("search_page: 200", example)


class TestTheMechanismIsShared(UiServerTestBase):
    """Requirement four: the rule holds for the slices that do not exist yet.

    F150 (low resolution), F151 (query slices), F156 (pinned queries) and F157 (blurred as
    a list) are all ordered lists, and all of them are ahead of this feature in the queue
    over `ui.py`. They inherit the button by calling the same two things instead of
    copying twenty lines a fifth time — which is what these tests pin.
    """

    def test_a_new_slice_gets_the_whole_paging_contract_from_one_call(self):
        page = ui._page_payload([{"file_id": 1}, {"file_id": 2}],
                                total=5, offset=0, limit=2)
        self.assertEqual(set(page), {"items", "total", "offset", "limit", "has_more"})
        self.assertEqual(page["total"], 5)
        self.assertTrue(page["has_more"])

    def test_the_helper_closes_the_list_at_its_end(self):
        for offset, items, expected in ((3, 2, False), (2, 2, True), (0, 5, False)):
            with self.subTest(offset=offset):
                page = ui._page_payload([{"n": i} for i in range(items)],
                                        total=5, offset=offset, limit=2)
                self.assertEqual(page["has_more"], expected)

    def test_an_empty_slice_offers_no_next_page(self):
        page = ui._page_payload([], total=0, offset=0, limit=200)
        self.assertFalse(page["has_more"])
        self.assertEqual(page["items"], [])

    def test_the_window_parser_is_the_shared_one_with_a_slice_specific_default(self):
        self.assertEqual(ui._parse_page_window({}), (0, ui._PLAN_PAGE_DEFAULT_LIMIT))
        self.assertEqual(ui._parse_page_window({}, 7), (0, 7))
        self.assertEqual(ui._parse_page_window({"limit": [""]}, 7), (0, 7))
        self.assertEqual(ui._parse_page_window({"limit": ["99999"]}, 7),
                         (0, ui._PLAN_PAGE_MAX_LIMIT))
        self.assertIsNone(ui._parse_page_window({"offset": ["-1"]}, 7))

    def test_every_paged_route_answers_in_the_same_shape(self):
        self.add_photo_file("a.jpg")
        self.start_server()
        for path in ("/api/search?q=", "/api/animals", "/api/face-slices?slice=people"):
            with self.subTest(path=path):
                status, body, _ctype = self.get(path)
                self.assertEqual(status, 200)
                data = json.loads(body)
                for key in ("items", "total", "offset", "limit", "has_more"):
                    self.assertIn(key, data)

    def test_the_browser_side_is_one_pager_used_by_every_ordered_slice(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertEqual(html.count("function makePager("), 1)
        for slice_ in ("searchPager", "animalsPager", "facePager"):
            with self.subTest(slice=slice_):
                self.assertIn("var " + slice_ + " = makePager(", html)
        # The per-slice bookkeeping the pager replaced: an offset variable and a click
        # handler each. Their return would mean a slice went back to its own copy.
        for gone in ("animalsOffset", "faceOffset", "fetchAnimals(", "fetchFaceSlice("):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, html)

    def test_the_button_and_the_counter_are_one_string_each(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        for element in ("search-more-btn", "animals-more-btn", "face-more-btn"):
            with self.subTest(element=element):
                self.assertIn('id="' + element + '"', html)
        self.assertIn("I18N.slice_shown_label", html)
        for retired in ("animals_load_more", "face_load_more", "animals_shown_label",
                        "face_shown_label"):
            with self.subTest(retired=retired):
                self.assertNotIn(retired, ui._UI_STRINGS)


class TestTheDepthHint(UiServerTestBase):
    """Requirement five: pressing the button buys coverage with precision, and says so."""

    def test_the_hint_stands_beside_the_button_of_the_ranked_slices(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        for slice_ in ("search", "animals"):
            with self.subTest(slice=slice_):
                hint = html.index('id="' + slice_ + '-depth-hint"')
                button = html.index('id="' + slice_ + '-more-btn"')
                self.assertLess(button, hint)          # the hint follows the button
                self.assertLess(hint - button, 400)    # and stays in the same row

    def test_the_face_slices_carry_no_such_warning(self):
        # Nothing is ranked there — a frame is in the slice because the detector found a
        # face on it — so a line about the model being less sure further down would be a
        # warning about a risk this list does not have.
        self.start_server()
        _status, body, _ctype = self.get("/")
        self.assertNotIn('id="face-depth-hint"', body.decode("utf-8"))

    def test_the_hint_is_one_line_in_three_languages(self):
        entry = ui._UI_STRINGS["slice_depth_hint"]
        self.assertEqual(set(entry), {"ru", "en", "ja"})
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                text = entry[lang]
                self.assertTrue(text.strip())
                self.assertLess(len(text), 220)   # a line, not a lecture

    def test_the_hint_appears_and_disappears_with_the_button(self):
        # One rule in the shared pager: with nothing left to load there is no trade to
        # warn about, and a permanent line about precision is a line nobody reads.
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn("if (hint) hint.style.display = visible ? \"\" : \"none\";", html)


if __name__ == "__main__":
    unittest.main()
