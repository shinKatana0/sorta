"""F189: a name in the search line finds the person — the engine half.

The main test of the feature is `TestOneSourceOfTruth`: the set of frames a name returns is
the set `sorter.plan_album(kind='person')` gathers for the same name. Everything else here
guards a property that would let those two part company — the roots of the `merged_into`
chains (F31), the case and the blanks people type names with, an unnamed cluster, a name
that is also an ordinary word — and each of them is checked against the album as well
wherever the two can disagree, because a bridge tested only from one bank is not tested.

No CLIP anywhere: a name is not a query and this half of the feature never encodes
anything. That is a property in itself and is checked below.
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from sorta import cli, i18n, search
from sorta.junk import pack_embedding, search_index_model
from sorta.sorter import plan_album

from tests.test_search import encoder_for, unit
from tests.test_sorter import SorterTestBase


class PersonSearchTestBase(SorterTestBase):
    def add_merged_person(self, file_id: int, root_cluster_id: int) -> int:
        """A face on file_id in a NEW cluster merged into root_cluster_id (F31)."""
        cur = self.conn.execute(
            "INSERT INTO face_clusters (label, merged_into) VALUES (NULL, ?)",
            (root_cluster_id,))
        cluster_id = int(cur.lastrowid or 0)
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding, cluster_id) VALUES (?, ?, ?, ?)",
            (file_id, "[0,0,10,10]", b"\x00" * 4, cluster_id))
        self.conn.commit()
        return cluster_id

    def album_ids(self, name: str) -> set[int]:
        """The frames `album person <name>` would gather — the other bank of the bridge."""
        with redirect_stdout(io.StringIO()):
            report = plan_album(self.cfg, self.conn, "person", name, self.dest,
                                apply=False)
        return {it.file_id for it in report.plan}

    def found(self, text: str, limit: int = 100) -> set[int]:
        """What the search line answers for `text`, as a set of file ids."""
        label = search.match_person(self.conn, text)
        self.assertIsNotNone(label)
        page = search.person_page(self.conn, str(label), limit=limit)
        return {file_id for file_id, _score in page.hits}


class TestOneSourceOfTruth(PersonSearchTestBase):
    """THE test: the name and the album select the same frames, or there are two engines."""

    def test_the_name_selects_exactly_what_the_album_gathers(self):
        first = self.add_file("a.jpg")
        root = self.add_person(first, "Ирина")
        second = self.add_file("b.jpg")
        self.add_merged_person(second, root)          # a merged cluster (F31)
        third = self.add_file("c.jpg")
        self.add_person(third, "Марк")                # somebody else
        dup = self.add_file("d.jpg")
        self.add_person(dup, "Ирина")
        self.conn.execute("UPDATE files SET dup_of = ? WHERE id = ?", (first, dup))
        broken = self.add_file("e.jpg")
        self.add_person(broken, "Ирина")
        self.conn.execute("UPDATE files SET error = 'unreadable' WHERE id = ?", (broken,))
        self.conn.commit()

        self.assertEqual(self.found("Ирина"), self.album_ids("Ирина"))
        self.assertEqual(self.found("Ирина"), {first, second})

    def test_the_two_agree_on_a_name_typed_carelessly(self):
        first = self.add_file("a.jpg")
        self.add_person(first, "Ирина")
        # What a careless spelling is resolved to is the LABEL, and the label is what an
        # album of this answer is gathered by — the two are equal there, which is the
        # form the equality has to hold in: the album is handed a name, not a keystroke.
        label = search.match_person(self.conn, "  ирина ")
        self.assertEqual(label, "Ирина")
        self.assertEqual(self.found("  ирина "), self.album_ids(str(label)))
        self.assertEqual(self.found("  ирина "), {first})


class TestMergedClusters(PersonSearchTestBase):
    def test_frames_of_a_merged_cluster_are_in_the_answer(self):
        first = self.add_file("a.jpg")
        root = self.add_person(first, "Ирина")
        second = self.add_file("b.jpg")
        self.add_merged_person(second, root)
        self.assertEqual(self.found("Ирина"), {first, second})

    def test_a_chain_of_merges_is_followed_to_its_root(self):
        first = self.add_file("a.jpg")
        root = self.add_person(first, "Ирина")
        second = self.add_file("b.jpg")
        middle = self.add_merged_person(second, root)
        third = self.add_file("c.jpg")
        self.add_merged_person(third, middle)
        self.assertEqual(self.found("Ирина"), {first, second, third})
        self.assertEqual(self.found("Ирина"), self.album_ids("Ирина"))

    def test_a_label_left_on_a_swallowed_cluster_names_nobody(self):
        # The cluster «Ира» was merged into «Ирина» and kept its old label. The album
        # selects by the ROOT's label, so this name selects nothing — and a name that
        # selects nothing must not be recognized as a name at all, or the reader gets an
        # empty "frames of Ира" screen instead of a search.
        first = self.add_file("a.jpg")
        root = self.add_person(first, "Ирина")
        second = self.add_file("b.jpg")
        swallowed = self.add_person(second, "Ира")
        self.conn.execute("UPDATE face_clusters SET merged_into = ? WHERE id = ?",
                          (root, swallowed))
        self.conn.commit()
        self.assertIsNone(search.match_person(self.conn, "Ира"))
        self.assertEqual(self.album_ids("Ира"), set())


class TestMatchingTheName(PersonSearchTestBase):
    def test_case_and_surrounding_blanks_do_not_matter(self):
        first = self.add_file("a.jpg")
        self.add_person(first, "Ирина")
        for typed in ("Ирина", "ирина", "ИРИНА", "  Ирина  ", "\tирина\n"):
            self.assertEqual(search.match_person(self.conn, typed), "Ирина", typed)

    def test_the_label_comes_back_as_it_is_stored(self):
        # The caption says the name — and it says the name of the person as they were
        # named, not as the search line happened to be typed.
        first = self.add_file("a.jpg")
        self.add_person(first, "Ирина")
        self.assertEqual(search.match_person(self.conn, "ИРИНА"), "Ирина")

    def test_a_name_nobody_gave_is_not_a_name(self):
        first = self.add_file("a.jpg")
        self.add_person(first, "Ирина")
        self.assertIsNone(search.match_person(self.conn, "Марк"))

    def test_nothing_fuzzy(self):
        # «Ира» -> «Ирина» is a separate question with an error cost of its own: somebody
        # else's frames under a name. Here the match is exact or it is not a match.
        first = self.add_file("a.jpg")
        self.add_person(first, "Ирина")
        self.assertIsNone(search.match_person(self.conn, "Ира"))
        self.assertIsNone(search.match_person(self.conn, "Ирина Петрова"))

    def test_an_unnamed_cluster_is_found_by_nothing(self):
        first = self.add_file("a.jpg")
        cur = self.conn.execute(
            "INSERT INTO face_clusters (label, merged_into) VALUES (NULL, NULL)")
        self.conn.execute(
            "INSERT INTO faces (file_id, bbox, embedding, cluster_id) VALUES (?, ?, ?, ?)",
            (first, "[0,0,10,10]", b"\x00" * 4, int(cur.lastrowid or 0)))
        self.conn.commit()
        self.assertIsNone(search.match_person(self.conn, ""))
        self.assertIsNone(search.match_person(self.conn, "   "))
        self.assertIsNone(search.match_person(self.conn, "None"))

    def test_an_empty_collection_answers_none_rather_than_raising(self):
        self.assertIsNone(search.match_person(self.conn, "Ирина"))


class TestThePage(PersonSearchTestBase):
    def setUp(self):
        super().setUp()
        self.ids = []
        root = None
        for i in range(5):
            file_id = self.add_file(f"{i}.jpg", content=f"data{i}".encode())
            if root is None:
                root = self.add_person(file_id, "Ирина")
            else:
                self.add_merged_person(file_id, root)
            self.ids.append(file_id)

    def test_the_total_is_the_length_of_the_list_and_the_window_is_the_window(self):
        page = search.person_page(self.conn, "Ирина", limit=2)
        self.assertEqual([file_id for file_id, _s in page.hits], self.ids[:2])
        self.assertEqual(page.total, 5)
        self.assertTrue(page.has_more)

    def test_paging_walks_the_same_list_without_a_gap_or_a_repeat(self):
        seen: list[int] = []
        offset = 0
        while True:
            page = search.person_page(self.conn, "Ирина", limit=2, offset=offset)
            seen.extend(file_id for file_id, _s in page.hits)
            if not page.has_more:
                break
            offset += len(page.hits)
        self.assertEqual(seen, self.ids)

    def test_the_frames_carry_no_score(self):
        # A selection has no ranking in it, so there is no number to read off a card —
        # every frame is here for the same reason.
        page = search.person_page(self.conn, "Ирина", limit=5)
        self.assertEqual({score for _f, score in page.hits}, {search.PERSON_NO_SCORE})

    def test_a_page_past_the_end_is_empty_and_still_states_the_total(self):
        page = search.person_page(self.conn, "Ирина", limit=2, offset=99)
        self.assertEqual(page.hits, [])
        self.assertEqual(page.total, 5)
        self.assertFalse(page.has_more)

    def test_no_vector_is_read_and_no_encoder_is_needed(self):
        # The collection has no search index at all: a name does not go near one.
        self.assertEqual(
            int(self.conn.execute("SELECT COUNT(*) FROM search_embeddings")
                .fetchone()[0]), 0)
        self.assertEqual(search.person_page(self.conn, "Ирина", limit=5).total, 5)


class TestTheSearchCommand(PersonSearchTestBase):
    """`sorta search Ирина` — the same string, the same behaviour as the interface."""

    def setUp(self):
        super().setUp()
        self.model = search_index_model(self.cfg)
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(
            f'database: "{self.cfg.database.as_posix()}"\nlanguage: en\n',
            encoding="utf-8")

    def add_indexed_file(self, rel: str, vec: np.ndarray, **kwargs) -> int:
        """A file with the search vector the junk stage would have left on it."""
        file_id = self.add_file(rel, **kwargs)
        self.conn.execute(
            """INSERT INTO search_embeddings (file_id, model, dim, vec, updated_at)
               VALUES (?, ?, ?, ?, '2026-01-01')""",
            (file_id, self.model, int(vec.size), pack_embedding(vec)))
        self.conn.commit()
        return file_id

    def run_search(self, query: str, limit: int | None = None,
                   words: bool = False) -> str:
        buffer = io.StringIO()
        with patch.object(search, "text_encoder", lambda s: encoder_for({})), \
                redirect_stdout(buffer):
            cli._cmd_search(str(self.config_path), query, limit=limit, words=words)
        return buffer.getvalue()

    def test_a_name_prints_the_person_and_says_it_is_an_exact_selection(self):
        first = self.add_file("a.jpg")
        self.add_person(first, "Ирина")
        printed = self.run_search("ирина ")
        self.assertIn(self.path_of(first), printed)
        self.assertIn(i18n.cli_text("cli.search.person_done", "en", name="Ирина", n=1),
                      printed)
        # And not the ranking's sentence: the two answers do not read alike.
        self.assertNotIn(i18n.cli_text("cli.search.done", "en", n=1, query="ирина "),
                         printed)

    def test_a_name_answers_without_an_index_and_without_a_model(self):
        # There is nothing in `search_embeddings` at all: a ranking would exit with the
        # "no index" refusal, and a name does not need one.
        first = self.add_file("a.jpg")
        self.add_person(first, "Ирина")
        encoded: list[str] = []

        def tower(_settings):
            def encode(texts):
                encoded.extend(texts)
                return np.stack([unit(1.0) for _t in texts])
            return encode

        with patch.object(search, "text_encoder", tower), \
                redirect_stdout(io.StringIO()):
            cli._cmd_search(str(self.config_path), "Ирина")
        self.assertEqual(encoded, [])

    def test_the_frames_are_the_same_ones_the_engine_selects(self):
        first = self.add_file("a.jpg")
        root = self.add_person(first, "Ирина")
        second = self.add_file("b.jpg", content=b"second")
        self.add_merged_person(second, root)
        printed = self.run_search("Ирина")
        for file_id in (first, second):
            self.assertIn(self.path_of(file_id), printed)

    def test_the_limit_cuts_the_lines_and_the_count_still_states_the_whole_list(self):
        first = self.add_file("a.jpg")
        root = self.add_person(first, "Ирина")
        for i in range(3):
            self.add_merged_person(self.add_file(f"{i}.jpg", content=str(i).encode()),
                                   root)
        printed = self.run_search("Ирина", limit=2)
        # two paths + the sentence + the way back to the ranking
        self.assertEqual(len(printed.strip().splitlines()), 4)
        self.assertIn(i18n.cli_text("cli.search.person_done", "en", name="Ирина", n=4),
                      printed)

    def test_a_name_that_is_also_a_word_still_offers_the_ranking(self):
        first = self.add_file("a.jpg")
        self.add_person(first, "Роза")
        printed = self.run_search("Роза")
        self.assertIn(i18n.cli_text("cli.search.person_words_hint", "en", query="Роза"),
                      printed)

    def test_the_words_flag_asks_the_ranking_for_the_very_same_string(self):
        first = self.add_indexed_file("a.jpg", unit(1.0))
        self.add_person(first, "Роза")
        printed = self.run_search("Роза", words=True)
        self.assertIn(i18n.cli_text("cli.search.done", "en", n=1, query="Роза"), printed)
        self.assertNotIn("cli.search.person_done", printed)

    def test_a_string_that_names_nobody_is_an_ordinary_query(self):
        first = self.add_indexed_file("a.jpg", unit(1.0))
        self.add_person(first, "Ирина")
        printed = self.run_search("торт")
        self.assertIn(i18n.cli_text("cli.search.done", "en", n=1, query="торт"), printed)

    def test_the_help_of_the_new_flag_exists_in_all_three_languages(self):
        texts = {lang: i18n.cli_text("cli.help.search.words", lang)
                 for lang in ("ru", "en", "ja")}
        self.assertEqual(len(set(texts.values())), 3)
        for lang, text in texts.items():
            self.assertNotIn("cli.", text, lang)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
