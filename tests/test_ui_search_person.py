"""F189: the search line of the "Slices" tab answers a NAME with the person.

The engine half is `tests/test_search_person.py`. What an interface can get wrong on its
own is everything below:

* it can hand back the person's frames and CAPTION them as a ranking, which is how an exact
  answer gets read as the top of a list;
* it can let the name swallow the word search — «Роза» and «Марк» are ordinary words as
  well, and the second answer must not disappear because the first one exists;
* it can answer a pinned name (F156) by a different route than the typed one, and then the
  pin quietly becomes a second engine — the failure F151 exists not to be;
* and it can drift away from `sorta search`, which would make "one query string, one
  behaviour" a claim about two implementations.

The set of frames is checked against `plan_album(kind='person')` here too, not only in the
engine's own tests: the route is the thing a person actually uses, and a bridge tested only
at the far end is not tested.
"""
from __future__ import annotations

import dataclasses
import io
import re
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from sorta import cli, i18n, search, ui
from sorta.sorter import plan_album

from tests.test_pin_a_query import PinTestBase
from tests.test_search import unit

# "  1. …" — a line of the result list, whichever of the two answers printed it.
_HIT_LINE = re.compile(r"^\s*\d+\. ")


class PersonUiTestBase(PinTestBase):
    """A UI server with a config file (the pins live there) and a named face cluster."""

    def add_person(self, file_id: int, label: str | None,
                   merged_into: int | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO face_clusters (label, merged_into) VALUES (?, ?)",
            (label, merged_into))
        cluster_id = int(cur.lastrowid or 0)
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding, cluster_id) VALUES (?, ?, ?, ?)",
            (file_id, "[0,0,10,10]", b"\x00" * 4, cluster_id))
        self.conn.commit()
        return cluster_id

    def add_named_photo(self, rel: str, label: str, **kwargs) -> tuple[int, int]:
        file_id = self.add_indexed_photo(rel, unit(1.0), **kwargs)
        return file_id, self.add_person(file_id, label)

    def ids(self, data: dict) -> list[int]:
        return [it["file_id"] for it in data["items"]]

    def album_ids(self, name: str) -> set[int]:
        with redirect_stdout(io.StringIO()):
            report = plan_album(self.cfg, self.conn, "person", name,
                                self.root / "album", apply=False)
        return {it.file_id for it in report.plan}


class TestANameAnswersWithThePerson(PersonUiTestBase):
    """The acceptance criterion: «Ирина» gives her frames, not frames like the word."""

    def test_the_frames_are_the_person_and_the_answer_says_so(self):
        hers, _cluster = self.add_named_photo("a.jpg", "Ирина")
        self.add_named_photo("b.jpg", "Марк")
        self.start_server()
        data = self.search("Ирина")
        self.assertEqual(self.ids(data), [hers])
        self.assertEqual(data["person"], "Ирина")
        self.assertTrue(data["exact"])
        self.assertEqual(data["total"], 1)

    def test_the_set_is_the_one_the_album_gathers(self):
        # THE property: one source of truth, checked through the route a person uses.
        first, cluster = self.add_named_photo("a.jpg", "Ирина")
        second = self.add_indexed_photo("b.jpg", unit(1.0))
        self.add_person(second, None, merged_into=cluster)
        self.add_named_photo("c.jpg", "Марк")
        self.start_server()
        self.assertEqual(set(self.ids(self.search("Ирина"))), self.album_ids("Ирина"))
        self.assertEqual(set(self.ids(self.search("Ирина"))), {first, second})

    def test_the_frames_of_a_merged_cluster_are_in_the_answer(self):
        first, cluster = self.add_named_photo("a.jpg", "Ирина")
        second = self.add_indexed_photo("b.jpg", unit(1.0))
        self.add_person(second, None, merged_into=cluster)
        self.start_server()
        self.assertEqual(set(self.ids(self.search("Ирина"))), {first, second})

    def test_case_and_blanks_are_not_part_of_a_name(self):
        hers, _cluster = self.add_named_photo("a.jpg", "Ирина")
        self.start_server()
        for typed in ("ирина", "  Ирина ", "ИРИНА"):
            data = self.search(typed)
            with self.subTest(typed=typed):
                self.assertEqual(self.ids(data), [hers])
                # The caption says the name as it was GIVEN, not as it was typed.
                self.assertEqual(data["person"], "Ирина")

    def test_a_card_of_a_selection_carries_no_score(self):
        # There is no order here to explain: every frame is in the list for one reason.
        self.add_named_photo("a.jpg", "Ирина")
        self.start_server()
        item = self.search("Ирина")["items"][0]
        self.assertNotIn("score", item)
        self.assertIn("thumb_url", item)

    def test_an_unnamed_cluster_is_found_by_nothing(self):
        file_id = self.add_indexed_photo("a.jpg", unit(1.0))
        self.add_person(file_id, None)
        self.start_server()
        data = self.search("None")
        self.assertIsNone(data["person"])
        self.assertFalse(data["exact"])

    def test_a_name_answers_even_when_the_index_is_empty(self):
        # A selection needs no vector. The two states of the index still travel with the
        # answer — they are about the search line, not about this list.
        file_id, _cluster = self.add_named_photo("a.jpg", "Ирина")
        self.conn.execute("DELETE FROM search_embeddings")
        self.conn.commit()
        self.start_server()
        data = self.search("Ирина")
        self.assertEqual(self.ids(data), [file_id])
        self.assertTrue(data["exact"])
        self.assertFalse(data["available"])
        self.assertEqual(self.encoded, [])   # and no model was loaded for it

    def test_the_page_walks_the_list_by_count(self):
        first, cluster = self.add_named_photo("a.jpg", "Ирина")
        rest = []
        for i in range(3):
            file_id = self.add_indexed_photo(f"{i}.jpg", unit(1.0))
            self.add_person(file_id, None, merged_into=cluster)
            rest.append(file_id)
        self.cfg.features = dataclasses.replace(self.cfg.features, search_page=2)
        self.start_server()
        head = self.search("Ирина")
        self.assertEqual(head["total"], 4)
        self.assertTrue(head["has_more"])
        tail = self.search("Ирина", extra="&offset=2")
        self.assertEqual(self.ids(head) + self.ids(tail), [first, *rest])
        self.assertFalse(tail["has_more"])


class TestTheWordSearchDoesNotDisappear(PersonUiTestBase):
    """Requirement 4: a name that is also a word keeps both answers."""

    def test_a_name_that_is_a_word_shows_the_person_and_names_the_other_answer(self):
        hers, _cluster = self.add_named_photo("a.jpg", "Роза")
        self.start_server()
        data = self.search("Роза")
        self.assertEqual(self.ids(data), [hers])
        self.assertTrue(data["exact"])
        # The client draws the way back out of `person` — an answer that did not name it
        # could not offer the other one.
        self.assertEqual(data["person"], "Роза")

    def test_words_asks_for_the_ranking_of_the_very_same_string(self):
        hers, _cluster = self.add_named_photo("a.jpg", "Роза")
        other = self.add_indexed_photo("b.jpg", unit(0.0, 1.0))
        self.start_server()
        data = self.search("Роза", extra="&words=1")
        self.assertFalse(data["exact"])
        self.assertEqual(self.ids(data), [hers, other])   # a ranking of everything
        self.assertEqual(data["person"], "Роза")          # and the way back to her
        self.assertEqual(self.encoded, ["Роза"])

    def test_the_ranking_of_a_string_that_names_nobody_is_untouched(self):
        first = self.add_indexed_photo("a.jpg", unit(1.0))
        self.add_named_photo("b.jpg", "Ирина")
        self.start_server()
        data = self.search("торт")
        self.assertIsNone(data["person"])
        self.assertFalse(data["exact"])
        self.assertIn(first, self.ids(data))
        self.assertIn("score", data["items"][0])

    def test_a_name_nobody_gave_is_a_query_and_not_an_empty_person_screen(self):
        first = self.add_indexed_photo("a.jpg", unit(1.0))
        self.add_named_photo("b.jpg", "Ирина")
        self.start_server()
        data = self.search("Пётр")
        self.assertIsNone(data["person"])
        self.assertFalse(data["exact"])
        self.assertTrue(data["items"])
        self.assertIn(first, self.ids(data))

    def test_a_malformed_words_flag_is_not_an_error(self):
        hers, _cluster = self.add_named_photo("a.jpg", "Роза")
        self.start_server()
        data = self.search("Роза", extra="&words=maybe")
        self.assertTrue(data["exact"])
        self.assertEqual(self.ids(data), [hers])


class TestPinningAName(PersonUiTestBase):
    """Requirement 5/7: a named person becomes an ordinary tab, answering the same set."""

    def test_a_pinned_name_answers_exactly_what_the_search_line_answered(self):
        self.pin_nothing()
        first, cluster = self.add_named_photo("a.jpg", "Ирина")
        second = self.add_indexed_photo("b.jpg", unit(1.0))
        self.add_person(second, None, merged_into=cluster)
        self.add_named_photo("c.jpg", "Марк")
        self.start_server()
        typed = self.search("Ирина")
        status, _resp = self.pin("Ирина")
        self.assertEqual(status, 200)
        pinned = self.slice_("Ирина")
        self.assertEqual(self.ids(pinned), self.ids(typed))
        self.assertEqual(set(self.ids(pinned)), {first, second})
        self.assertEqual(set(self.ids(pinned)), self.album_ids("Ирина"))

    def test_the_pin_says_it_is_a_person_and_not_an_estimate(self):
        self.pin_nothing()
        self.add_named_photo("a.jpg", "Ирина")
        self.start_server()
        self.pin("Ирина")
        pinned = self.slice_("Ирина")
        self.assertEqual(pinned["person"], "Ирина")
        self.assertTrue(pinned["exact"])
        # The word every ranking on this tab is captioned with does not apply here.
        self.assertFalse(pinned["approximate"])
        self.assertNotIn("score", pinned["items"][0])

    def test_a_pinned_name_needs_no_index_and_no_model(self):
        self.pin_nothing()
        file_id, _cluster = self.add_named_photo("a.jpg", "Ирина")
        self.conn.execute("DELETE FROM search_embeddings")
        self.conn.commit()
        self.start_server()
        self.pin("Ирина")
        pinned = self.slice_("Ирина")
        self.assertEqual(self.ids(pinned), [file_id])
        self.assertEqual(self.encoded, [])

    def test_a_pin_of_words_is_still_a_ranking(self):
        # Nothing about the pins changed for the queries they were made for.
        self.pin_nothing()
        first = self.add_indexed_photo("a.jpg", unit(1.0))
        self.add_named_photo("b.jpg", "Ирина")
        self.start_server()
        self.pin("cake", "торты")
        pinned = self.slice_("торты")
        self.assertIsNone(pinned["person"])
        self.assertFalse(pinned["exact"])
        self.assertTrue(pinned["approximate"])
        self.assertIn(first, self.ids(pinned))

    def test_a_pin_of_several_phrases_is_a_query_even_when_one_of_them_is_a_name(self):
        # A name averaged with other words is not a name: the vector is a direction none
        # of the phrases has, and the cluster has nothing to do with it.
        self.pin_nothing()
        first = self.add_indexed_photo("a.jpg", unit(1.0))
        self.add_named_photo("b.jpg", "Ирина")
        self.cfg.features = dataclasses.replace(
            self.cfg.features,
            saved_slices=(ui.SavedSlice(name="mix", queries=("Ирина", "cake")),))
        self.start_server()
        pinned = self.slice_("mix")
        self.assertIsNone(pinned["person"])
        self.assertFalse(pinned["exact"])
        self.assertIn(first, self.ids(pinned))


class TestTheCliAndTheInterfaceAgree(PersonUiTestBase):
    """Requirement 6/8: one query string, one behaviour, whichever door it came through."""

    def cli_paths(self, query: str, words: bool = False) -> list[str]:
        buffer = io.StringIO()
        with patch.object(search, "text_encoder",
                          lambda s: (lambda texts: unit(1.0).reshape(1, -1))), \
                redirect_stdout(buffer):
            cli._cmd_search(str(self.config_path), query, words=words)
        # A hit line is "  1. <path>" for a person and "  1. 0.998  <path>" for a
        # ranking — the score column is the visible difference between the two answers,
        # and what is compared here is the frames.
        return [re.sub(r"^\d+\.\d+\s+", "", line.split(". ", 1)[1].strip())
                for line in buffer.getvalue().splitlines()
                if _HIT_LINE.match(line)]

    def paths_of(self, file_ids: list[int]) -> list[str]:
        found = search.file_paths(self.conn, file_ids)
        return [found[file_id] for file_id in file_ids]

    def test_the_same_name_gives_the_same_frames_in_both(self):
        first, cluster = self.add_named_photo("a.jpg", "Ирина")
        second = self.add_indexed_photo("b.jpg", unit(1.0))
        self.add_person(second, None, merged_into=cluster)
        self.add_named_photo("c.jpg", "Марк")
        self.start_server()
        self.assertEqual(self.cli_paths("ирина "),
                         self.paths_of(self.ids(self.search("ирина "))))

    def test_both_treat_a_string_that_names_nobody_as_a_query(self):
        self.add_indexed_photo("a.jpg", unit(1.0))
        self.add_named_photo("b.jpg", "Ирина")
        self.start_server()
        self.assertEqual(self.cli_paths("торт"),
                         self.paths_of(self.ids(self.search("торт"))))

    def test_the_way_back_to_the_words_is_the_same_answer_in_both(self):
        self.add_named_photo("a.jpg", "Роза")
        self.add_indexed_photo("b.jpg", unit(1.0))
        self.start_server()
        self.assertEqual(self.cli_paths("Роза", words=True),
                         self.paths_of(self.ids(self.search("Роза", extra="&words=1"))))


class TestTheCaptions(unittest.TestCase):
    """Requirement 3: the answer says WHAT it is, in all three languages."""

    def test_every_new_string_is_in_the_catalog_three_times(self):
        for key in ("search_person_shown_label", "search_person_hint",
                    "search_person_more_hint", "search_person_words_link",
                    "search_words_person_link", "search_person_no_frames"):
            texts = {lang: ui._t(key, lang) for lang in ("ru", "en", "ja")}
            with self.subTest(key=key):
                self.assertEqual(len(set(texts.values())), 3)
                for text in texts.values():
                    self.assertNotEqual(text, key)

    def test_the_person_caption_is_not_the_ranking_caption(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertNotEqual(ui._t("search_person_shown_label", lang),
                                    ui._t("search_shown_label", lang))
                self.assertNotEqual(ui._t("search_person_hint", lang),
                                    ui._t("search_ranking_hint", lang))

    def test_the_cli_says_a_selection_rather_than_a_ranking(self):
        for lang in ("ru", "en", "ja"):
            with self.subTest(lang=lang):
                self.assertNotEqual(
                    i18n.cli_text("cli.search.person_done", lang, name="X", n=1),
                    i18n.cli_text("cli.search.done", lang, query="X", n=1))


class TestTheScriptWiresTheAnswerUp(unittest.TestCase):
    """The client half, checked the way the rest of the UI's JS is: in the page."""

    def test_the_page_asks_for_the_ranking_with_a_flag(self):
        self.assertIn('(searchWords ? "&words=1" : "")', ui._INDEX_HTML_TEMPLATE)

    def test_the_page_captions_an_exact_answer_apart(self):
        self.assertIn("searchExact ? I18N.search_person_hint : I18N.search_ranking_hint",
                      ui._INDEX_HTML_TEMPLATE)
        self.assertIn("I18N.search_person_shown_label", ui._INDEX_HTML_TEMPLATE)

    def test_the_page_offers_the_other_answer(self):
        self.assertIn("renderSearchOtherAnswer", ui._INDEX_HTML_TEMPLATE)
        self.assertIn("I18N.search_words_person_link", ui._INDEX_HTML_TEMPLATE)

    def test_the_album_of_a_person_gathers_the_person(self):
        self.assertIn('gatherAlbum("person", person', ui._INDEX_HTML_TEMPLATE)
        self.assertIn('gatherAlbum(data.person ? "person" : "query"', ui._INDEX_HTML_TEMPLATE)

    def test_a_card_without_a_score_draws_none(self):
        self.assertIn("item.score !== undefined && item.score !== null", ui._INDEX_HTML_TEMPLATE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
