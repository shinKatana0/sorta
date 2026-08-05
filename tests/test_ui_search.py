"""F134: the search line of the "Slices" tab — `GET /api/search` and its four states.

The engine is F129's and is not touched here. What is tested is the thing an interface can
get wrong on its own: answering "nothing was found" when the truth is "nothing was ever
computed". Those two read identically as an empty list, and only one of them is a fact
about the person's photographs — so every test below is, one way or another, about the
index saying which state it is in.

No model is loaded: `ui.text_encoder` is replaced, which is what the injectable encoder of
F129 exists for. The fake also counts its calls, because "an empty query never reaches the
model" is a requirement rather than an optimization.
"""
from __future__ import annotations

import dataclasses
import json
import unittest
import urllib.error
import urllib.parse
import urllib.request
from unittest.mock import patch

import numpy as np

from sorta import ui
from sorta.junk import pack_embedding, search_index_model

from tests.test_search import unit
from tests.test_ui import UiServerTestBase


class SearchUiTestBase(UiServerTestBase):
    """A UI server whose CLIP text tower is a fake with a known direction per query."""

    def setUp(self):
        super().setUp()
        # F141: the model the search side is configured with — `features.search_model`.
        self.model = search_index_model(self.cfg)
        self.vectors: dict[str, np.ndarray] = {}
        self.encoded: list[str] = []

        def fake_tower(_settings):
            def encode(texts):
                self.encoded.extend(texts)
                # Not normalized on purpose (see tests.test_search.encoder_for): the
                # engine has to bring the query back to a unit vector itself.
                return np.stack([self.vectors.get(t, unit(1.0)) * 3.0 for t in texts])
            return encode

        patcher = patch.object(ui.slices, "text_encoder", fake_tower)
        patcher.start()
        self.addCleanup(patcher.stop)

    # --- fixtures ------------------------------------------------------

    def add_indexed_photo(self, rel: str, vec: np.ndarray, *,
                          model: str | None = None) -> int:
        """An indexed photograph with the CLIP vector the junk stage would have left."""
        file_id, _path, _content = self.add_photo_file(rel)
        self.store_vector(file_id, vec, model=model)
        return file_id

    def store_vector(self, file_id: int, vec: np.ndarray,
                     model: str | None = None) -> None:
        """F141: the SEARCH index — the table this route reports on and ranks out of.

        Not `clip_embeddings`: those are the classification model's vectors, they cannot
        answer a query, and a coverage line counting them would say "searching all 19 753
        photographs" about an index the engine refuses to use.
        """
        self.conn.execute(
            """INSERT INTO search_embeddings (file_id, model, dim, vec, updated_at)
               VALUES (?, ?, ?, ?, '2026-01-01')""",
            (file_id, model or self.model, int(vec.size), pack_embedding(vec)))
        self.conn.commit()

    def classify(self, file_id: int, verdict: str) -> None:
        self.conn.execute(
            """INSERT INTO media_class (file_id, verdict, source, score, updated_at, tier)
               VALUES (?, ?, 'vlm', 0.9, '2026-01-01', 'vlm')""",
            (file_id, verdict))
        self.conn.commit()

    # --- requests ------------------------------------------------------

    def search(self, query: str = "", extra: str = "") -> dict:
        status, body, ctype = self.get(
            f"/api/search?q={urllib.parse.quote(query)}{extra}")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        return json.loads(body)

    def post(self, path: str, data: object) -> tuple[int, dict]:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=body, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())


class TestSearchIndexState(SearchUiTestBase):
    """The main test of the feature: an interface that cannot search says WHY."""

    def test_an_empty_index_disables_the_line_and_names_the_reason(self):
        self.add_photo_file("a.jpg")
        self.start_server()
        data = self.search()
        self.assertEqual(data["state"], "empty")
        self.assertFalse(data["available"])
        self.assertEqual(data["indexed"], 0)
        self.assertEqual(data["photos"], 1)
        self.assertIsNone(data["index_model"])
        self.assertEqual(data["items"], [])

    def test_an_empty_index_answers_a_real_query_with_the_reason_not_a_void(self):
        # The whole point: "cake" over an unfilled table must not come back as an empty
        # list, which reads as "you have no photographs of a cake".
        self.add_photo_file("a.jpg")
        self.start_server()
        data = self.search("торт")
        self.assertEqual(data["state"], "empty")
        self.assertFalse(data["available"])
        self.assertEqual(data["items"], [])
        self.assertEqual(self.encoded, [])  # and no model was loaded to say so

    def test_vectors_of_another_model_are_a_different_state_and_name_it(self):
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.store_vector(fid, unit(1.0), model="OtherNet-B/laion")
        self.start_server()
        data = self.search()
        self.assertEqual(data["state"], "other_model")
        self.assertNotEqual(data["state"], "empty")  # the reason is NOT the same one
        self.assertFalse(data["available"])
        self.assertEqual(data["index_model"], "OtherNet-B/laion")
        self.assertEqual(data["model"], self.model)
        self.assertEqual(data["indexed"], 0)

    def test_a_full_classification_index_is_still_nothing_to_search(self):
        """F141: `clip_embeddings` is a different model and cannot answer a query.

        The state line and the ranking have to agree about that, and this is the state a
        real collection is in before `features.search_index` is switched on: every
        classification vector present, not one search vector. A route that counted the
        wrong table would say "searching all 1 photographs" and then refuse to search.
        """
        fid, _p, _c = self.add_photo_file("a.jpg")
        self.conn.execute(
            """INSERT INTO clip_embeddings (file_id, model, dim, vec, updated_at)
               VALUES (?, 'ViT-L-14-quickgelu/openai', ?, ?, '2026-01-01')""",
            (fid, int(unit(1.0).size), pack_embedding(unit(1.0))))
        self.conn.commit()
        self.start_server()
        data = self.search("торт")
        self.assertEqual(data["state"], "empty")
        self.assertFalse(data["available"])
        self.assertEqual(data["indexed"], 0)
        self.assertIsNone(data["index_model"])
        self.assertEqual(data["items"], [])

    def test_the_other_model_is_the_one_with_the_most_rows(self):
        for i in range(2):
            fid, _p, _c = self.add_photo_file(f"a{i}.jpg")
            self.store_vector(fid, unit(1.0), model="Many/laion")
        fid, _p, _c = self.add_photo_file("b.jpg")
        self.store_vector(fid, unit(1.0), model="Few/laion")
        self.start_server()
        self.assertEqual(self.search()["index_model"], "Many/laion")

    def test_partial_coverage_is_searchable_and_states_the_fraction(self):
        self.add_indexed_photo("indexed.jpg", unit(1.0))
        self.add_photo_file("not_indexed.jpg")
        self.start_server()
        data = self.search()
        self.assertEqual(data["state"], "partial")
        self.assertTrue(data["available"])
        self.assertEqual(data["indexed"], 1)
        self.assertEqual(data["photos"], 2)

    def test_full_coverage_carries_no_warning(self):
        self.add_indexed_photo("a.jpg", unit(1.0))
        self.add_indexed_photo("b.jpg", unit(0.0, 1.0))
        self.start_server()
        data = self.search()
        self.assertEqual(data["state"], "ready")
        self.assertTrue(data["available"])
        self.assertEqual(data["indexed"], 2)
        self.assertEqual(data["photos"], 2)
        self.assertEqual(data["index_model"], self.model)

    def test_a_vector_of_a_frame_that_left_the_population_counts_for_neither(self):
        # A file can become a duplicate AFTER its vector was stored; a numerator that
        # counted it would inflate a fraction whose whole job is to be honest.
        keep = self.add_indexed_photo("a.jpg", unit(1.0))
        duplicate = self.add_indexed_photo("b.jpg", unit(1.0))
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?", (keep, duplicate))
        self.conn.commit()
        self.start_server()
        data = self.search()
        self.assertEqual(data["indexed"], 1)
        self.assertEqual(data["photos"], 1)
        self.assertEqual(data["state"], "ready")

    def test_vectors_only_for_frames_outside_the_population_is_the_empty_state(self):
        # Nothing to rank and the fix is another run — the same state `search.py` reports.
        keep, _p, _c = self.add_photo_file("a.jpg")
        duplicate = self.add_indexed_photo("b.jpg", unit(1.0))
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?", (keep, duplicate))
        self.conn.commit()
        self.start_server()
        data = self.search("торт")
        self.assertEqual(data["state"], "empty")
        self.assertFalse(data["available"])
        self.assertEqual(data["items"], [])


class TestSearchRanking(SearchUiTestBase):
    def test_results_are_sorted_by_closeness_descending(self):
        far = self.add_indexed_photo("far.jpg", unit(0.0, 1.0))
        near = self.add_indexed_photo("near.jpg", unit(1.0, 0.1))
        middle = self.add_indexed_photo("middle.jpg", unit(1.0, 1.0))
        self.start_server()
        items = self.search("торт")["items"]
        self.assertEqual([it["file_id"] for it in items], [near, middle, far])
        scores = [it["score"] for it in items]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_equal_scores_keep_a_deterministic_order(self):
        ids = [self.add_indexed_photo(f"a{i}.jpg", unit(1.0)) for i in range(3)]
        self.start_server()
        first = [it["file_id"] for it in self.search("торт")["items"]]
        second = [it["file_id"] for it in self.search("торт")["items"]]
        self.assertEqual(first, sorted(ids))
        self.assertEqual(first, second)

    def test_every_card_carries_the_score(self):
        fid = self.add_indexed_photo("cake.jpg", unit(1.0))
        self.start_server()
        item = self.search("торт")["items"][0]
        self.assertEqual(item["file_id"], fid)
        self.assertEqual(item["name"], "cake.jpg")
        self.assertEqual(item["date"], "2022-05-01T10:00:00")
        self.assertEqual(item["thumb_url"], f"/thumb/{fid}")
        self.assertAlmostEqual(item["score"], 1.0, places=5)

    def test_the_query_direction_decides_the_order(self):
        first = self.add_indexed_photo("snow.jpg", unit(0.0, 1.0))
        second = self.add_indexed_photo("cake.jpg", unit(1.0, 0.0))
        self.vectors["снег"] = unit(0.0, 1.0)
        self.start_server()
        self.assertEqual(
            [it["file_id"] for it in self.search("снег")["items"]], [first, second])
        self.assertEqual(
            [it["file_id"] for it in self.search("торт")["items"]], [second, first])

    def test_limit_bounds_the_sample(self):
        for i in range(4):
            self.add_indexed_photo(f"a{i}.jpg", unit(1.0, 0.1 * i))
        self.start_server()
        data = self.search("торт", extra="&limit=2")
        self.assertEqual(len(data["items"]), 2)
        self.assertEqual(data["limit"], 2)

    def test_an_oversized_limit_is_clamped_rather_than_refused(self):
        self.add_indexed_photo("a.jpg", unit(1.0))
        self.start_server()
        self.assertEqual(self.search("торт", extra="&limit=99999")["limit"],
                         ui._PLAN_PAGE_MAX_LIMIT)

    def test_a_broken_limit_is_a_400(self):
        self.start_server()
        for raw in ("nope", "-1"):
            with self.subTest(limit=raw):
                status, body, _ctype = self.get(f"/api/search?q=x&limit={raw}")
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(body))

    def test_an_empty_query_is_answered_without_the_model(self):
        self.add_indexed_photo("a.jpg", unit(1.0))
        self.start_server()
        for query in ("", "   "):
            with self.subTest(query=query):
                data = self.search(query)
                self.assertTrue(data["available"])
                self.assertEqual(data["items"], [])
        status, _body, _ctype = self.get("/api/search")  # not even the parameter
        self.assertEqual(status, 200)
        self.assertEqual(self.encoded, [])


class TestSearchSensitiveClasses(SearchUiTestBase):
    """F133's rule: a class in `vlm.exclude_classes` is never decoded for display, and a
    search must not become the way around that."""

    def test_a_document_gets_no_thumb_url(self):
        document = self.add_indexed_photo("passport.jpg", unit(1.0))
        photo = self.add_indexed_photo("cake.jpg", unit(1.0, 0.1))
        self.classify(document, "document")
        self.start_server()
        items = {it["file_id"]: it for it in self.search("торт")["items"]}
        self.assertNotIn("thumb_url", items[document])
        self.assertIn("thumb_url", items[photo])
        # it is still IN the ranking, with its score — hiding the row would be a second,
        # silent rule about what a search may return
        self.assertIn("score", items[document])

    def test_the_list_follows_the_running_config(self):
        screenshot = self.add_indexed_photo("shot.jpg", unit(1.0))
        self.classify(screenshot, "screenshot")
        self.start_server()
        self.assertIn("thumb_url", self.search("торт")["items"][0])
        # The settings panel can change `vlm.exclude_classes` without a restart, and a
        # privacy list that needs one is not a privacy list (F133).
        self.cfg.vlm = dataclasses.replace(
            self.cfg.vlm, exclude_classes=("document", "screenshot"))
        self.assertNotIn("thumb_url", self.search("торт")["items"][0])


class TestSearchAlbum(SearchUiTestBase):
    def test_a_query_album_previews_without_writing_anything(self):
        self.add_indexed_photo("cake.jpg", unit(1.0))
        self.start_server()
        status, body = self.post(
            "/api/album",
            {"kind": "query", "selector": "торт", "mode": "link", "apply": False})
        self.assertEqual(status, 200)
        self.assertEqual(body["kind"], "query")
        self.assertEqual(body["album_name"], "торт")
        self.assertEqual(body["count"], 1)
        self.assertFalse(body["applied"])
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM move_batches").fetchone()[0], 0)

    def test_apply_links_the_ranked_frames_into_an_album_query_batch(self):
        self.add_indexed_photo("cake.jpg", unit(1.0))
        self.start_server()
        status, body = self.post(
            "/api/album",
            {"kind": "query", "selector": "торт", "mode": "link", "apply": True})
        self.assertEqual(status, 200)
        self.assertEqual(body["transferred"], 1)
        batch = self.conn.execute(
            "SELECT mode FROM move_batches ORDER BY id DESC LIMIT 1").fetchone()
        self.assertEqual(batch["mode"], "album_query")

    def test_an_empty_selector_is_still_a_400_for_this_kind(self):
        self.start_server()
        status, body = self.post("/api/album", {"kind": "query", "mode": "link"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_an_album_over_an_empty_index_answers_with_the_reason(self):
        # A race (a run emptied the table) — and never an album of zero files, which
        # would read as "your collection holds none of these".
        self.add_photo_file("a.jpg")
        self.start_server()
        status, body = self.post(
            "/api/album", {"kind": "query", "selector": "торт", "mode": "link"})
        self.assertEqual(status, 409)
        self.assertEqual(body["reason"], "empty")

    def test_the_button_sends_the_kind_and_the_query_as_the_selector(self):
        # F193: through the row every slice shares — the kind and the selector are what
        # the search line hands it, and the row is the same one the memes bucket draws.
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        row = html.split("function renderSearchAlbum", 1)[1][:400]
        self.assertIn('box: "search-album"', row)
        self.assertIn('kind: query ? (person ? "person" : "query") : null', row)
        self.assertIn("selector: person || query", row)


class TestSearchMarkup(SearchUiTestBase):
    def test_the_line_lives_in_the_slices_tab_next_to_the_pins(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('<section id="tab-slices"', html)
        query_pos = html.index('id="slice-query"')
        pins_pos = html.index('id="slice-pins"')
        tab_end = html.index('<section id="tab-moves"')
        self.assertLess(query_pos, pins_pos)   # the line stands above the pinned slices
        self.assertLess(pins_pos, tab_end)     # and both are inside the same tab
        self.assertNotIn('id="tab-btn-search"', html)  # not a tab of its own

    def test_the_line_starts_disabled_with_a_way_out_beside_it(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        field = [ln for ln in html.splitlines() if 'id="slice-query"' in ln]
        self.assertEqual(len(field), 1)
        self.assertIn("disabled", field[0])        # until the index says otherwise
        self.assertIn(ui._t("search_placeholder", "en"), field[0])
        self.assertIn('id="slice-query-btn"', html)
        self.assertIn('id="slice-query-goto"', html)
        self.assertIn('id="slice-query-hint"', html)
        self.assertIn('id="tab-search"', html)
        self.assertIn('id="search-grid"', html)
        self.assertIn('id="search-album"', html)

    def test_the_client_asks_for_the_state_and_wires_the_two_buttons(self):
        self.start_server()
        _status, body, _ctype = self.get("/")
        html = body.decode("utf-8")
        self.assertIn('fetch("/api/search?q=")', html)
        self.assertIn('"/api/search?q=" + encodeURIComponent(searchQuery)', html)
        # The reason and the way to fix it are both driven by `available`. F189: the field
        # itself is driven by `usable` — `available` OR somebody named, because a name is
        # answered out of the clusters and needs no index; the sentence about the index
        # does not change, and the case is pinned in test_ui_search_person.py.
        self.assertIn('document.getElementById("slice-query").disabled = !usable;', html)
        self.assertIn("var usable = available || !!(state && state.names);", html)
        self.assertIn('activateTab("overview");', html)
        self.assertIn("I18N.search_state_empty", html)
        self.assertIn("I18N.search_state_other_model", html)
        self.assertIn("I18N.search_score_label", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)


class TestSearchStrings(SearchUiTestBase):
    def test_every_new_string_is_translated_three_ways(self):
        keys = ("search_placeholder", "search_button", "search_state_checking",
                "search_state_empty", "search_state_other_model",
                "search_state_partial", "search_state_ready", "search_goto_overview",
                "search_ranking_hint", "search_score_label", "search_shown_label",
                "search_no_frames", "error_loading_search")
        for key in keys:
            with self.subTest(key=key):
                entry = ui._UI_STRINGS[key]
                self.assertEqual(set(entry), {"ru", "en", "ja"})
                for lang in ("ru", "en", "ja"):
                    self.assertTrue(entry[lang].strip())

    def test_the_two_unavailable_states_never_share_a_sentence(self):
        # The instructions differ ("run it" vs "run it again, the model changed"), so a
        # single sentence for both would teach the reader nothing.
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertNotEqual(ui._UI_STRINGS["search_state_empty"][lang],
                                    ui._UI_STRINGS["search_state_other_model"][lang])
                self.assertIn("{model}",
                              ui._UI_STRINGS["search_state_other_model"][lang])

    def test_the_fraction_and_the_score_carry_their_numbers(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                partial = ui._UI_STRINGS["search_state_partial"][lang]
                self.assertIn("{n}", partial)
                self.assertIn("{all}", partial)
                self.assertIn("{score}", ui._UI_STRINGS["search_score_label"][lang])

    def test_no_state_promises_that_nothing_was_found(self):
        # The forbidden sentence of this feature, in the three languages it could appear
        # in: a data problem is never dressed up as a fact about the collection.
        forbidden = ("не найдено", "ничего не найд", "nothing was found",
                     "no results", "見つかりません")
        for key in ("search_state_empty", "search_state_other_model",
                    "search_state_partial", "search_state_ready", "search_no_frames"):
            for lang in ("ru", "en", "ja"):
                text = ui._UI_STRINGS[key][lang].lower()
                for phrase in forbidden:
                    with self.subTest(key=key, lang=lang, phrase=phrase):
                        self.assertNotIn(phrase, text)

    def test_i18n_reaches_the_page_in_three_languages(self):
        self.start_server()
        for lang, expected in (("ru", "Найти"), ("en", "Search"), ("ja", "検索")):
            with self.subTest(lang=lang):
                _status, body, _ctype = self.get(f"/?lang={lang}")
                self.assertIn(expected, body.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
