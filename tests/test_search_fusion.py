"""F153: two indexes answering one query — the merge, and what it is forbidden to do.

The feature exists because of a measurement, not a hunch: over 217 hand-labelled
judgements the classification model and the search model score the same at the top
(88/96/98% at ranks 1/3/5) and return DIFFERENT frames. Two rankings that are wrong in
different places are the one case where merging them can beat either half — so the merge
is offered in two forms, both of them cheap, and the default stays `off` until a
measurement of RECALL says which one to pick.

What is pinned here:

* `off` is today's search EXACTLY — the second index is not read at all, which is checked
  by making a read of it fail the test rather than by inspecting the output;
* `rank` puts agreement first: a frame both models rank first outranks a frame only one of
  them does;
* `union` keeps what only one model found: frames absent from the other model's list are
  in the answer;
* THE MAIN INVARIANT — the scores of two models are never added, averaged or compared.
  `fuse` is handed file ids and no numbers at all, which is what makes that a property of
  the code rather than a promise about it: the same positions produce the same ranking
  whatever the cosines behind them were;
* an index with nothing to rank does not silently halve the answer: the other one ranks,
  the fact is logged, and it is on the result;
* a row of another model takes part in neither index (F128's rule, twice).

No model is loaded anywhere: both text towers are two-line fakes, which is what the
`encoder`/`class_encoder` arguments exist for.
"""
from __future__ import annotations

import inspect
import math
from unittest.mock import patch

import numpy as np

from sorta import search
from sorta.config import (
    SEARCH_FUSION_OFF,
    SEARCH_FUSION_RANK,
    SEARCH_FUSION_UNION,
    Config,
    FeaturesConfig,
)
from sorta.junk import embedding_model, pack_embedding
from tests.test_search import SearchTestBase, encoder_for, unit


def search_at(cosine: float) -> np.ndarray:
    """A stored SEARCH vector whose cosine against the query below is exactly `cosine`.

    The query of the fixture points at the first axis, so the first component IS the
    score — which is what lets a test state the two rankings and the two sets of numbers
    behind them independently of each other.
    """
    return unit(cosine, math.sqrt(1.0 - cosine * cosine))


def class_at(cosine: float) -> np.ndarray:
    """The same for the CLASSIFICATION index, whose query points at the second axis."""
    return unit(math.sqrt(1.0 - cosine * cosine), cosine)


class FusionTestBase(SearchTestBase):
    """A collection with BOTH vectors per photograph — the state F153 needs to exist."""

    def setUp(self):
        super().setUp()
        self.class_model = embedding_model(self.cfg.naming)

    def cfg_with(self, mode: str, limit: int = 10) -> Config:
        return Config(database=self.cfg.database, raw={"language": "en"},
                      features=FeaturesConfig(search_fusion=mode, search_page=limit))

    def add_class_vector(self, file_id: int, vec: np.ndarray,
                         model: str | None = None) -> int:
        """The CLASSIFICATION vector of a frame that already has a search one."""
        self.conn.execute(
            """INSERT INTO clip_embeddings (file_id, model, dim, vec, updated_at)
               VALUES (?, ?, ?, ?, '2026-01-01')""",
            (file_id, model or self.class_model, int(vec.size), pack_embedding(vec)))
        self.conn.commit()
        return file_id

    def add_both(self, search_vec: np.ndarray, class_vec: np.ndarray) -> int:
        """One photograph with a vector in each index."""
        file_id = self.add_photo(search_vec)
        self.add_class_vector(file_id, class_vec)
        return file_id

    def fused(self, mode: str, query: str = "cake", *, limit: int = 10) -> search.Fusion:
        """The whole entry point, with a fake tower per index.

        The two queries point at different axes, which is the situation the feature is
        about: each model has an idea of its own about the word, in a space of its own, and
        neither list is the other's.
        """
        return search.search_fusion(
            self.cfg_with(mode, limit), self.conn, query,
            encoder=encoder_for({query: unit(1.0)}),
            class_encoder=encoder_for({query: unit(0.0, 1.0)}))


class TestOffIsTodaysSearch(FusionTestBase):
    """`off` — one index, and the other one not so much as read."""

    def test_the_classification_index_is_not_read_at_all(self):
        # Checked by making the read itself fail rather than by looking at the output: an
        # index that is read and then ignored still costs a table scan and a second text
        # tower, and `off` promises neither.
        first = self.add_both(unit(1.0), unit(0.0, 1.0))
        self.add_class_vector(self.add_photo(unit(0.0, 1.0)), unit(0.0, 1.0))
        with patch.object(search, "read_clip_embeddings",
                          side_effect=AssertionError("the other index was read")):
            answer = self.fused(SEARCH_FUSION_OFF)
        self.assertEqual([fid for fid, _score in answer.hits][:1], [first])

    def test_the_scores_are_still_the_cosines_of_the_search_model(self):
        self.add_both(unit(1.0, 1.0), unit(0.0, 1.0))
        (_fid, score), = self.fused(SEARCH_FUSION_OFF).hits
        self.assertAlmostEqual(score, float(np.dot(unit(1.0), unit(1.0, 1.0))), places=5)

    def test_the_answer_names_the_search_model_as_the_only_one_that_ranked(self):
        self.add_both(unit(1.0), unit(0.0, 1.0))
        answer = self.fused(SEARCH_FUSION_OFF)
        self.assertEqual(answer.used, (self.model,))
        self.assertEqual(answer.missing, {})
        self.assertEqual(answer.mode, SEARCH_FUSION_OFF)

    def test_off_is_the_default_until_a_measurement_says_otherwise(self):
        # The brief's own condition: the gain expected from a merge is in recall, and
        # recall has not been measured for either model. A default chosen ahead of that
        # number is the assumed accuracy F121/F122 cost 320 labels to unlearn.
        self.assertEqual(FeaturesConfig().search_fusion, SEARCH_FUSION_OFF)
        self.assertEqual(search.fusion_mode(Config(database=self.cfg.database)),
                         SEARCH_FUSION_OFF)


class TestRankPutsAgreementFirst(FusionTestBase):
    def test_a_frame_first_in_both_lists_beats_a_frame_first_in_one(self):
        # `both` is nearest to the query in each space; `search_only` is nearest in the
        # search space alone and has no classification vector to be found by at all.
        both = self.add_both(unit(1.0), unit(0.0, 1.0))
        search_only = self.add_photo(unit(1.0))
        self.add_class_vector(self.add_photo(unit(0.0, 0.0, 1.0)), unit(0.0, 1.0))
        hits = [fid for fid, _score in self.fused(SEARCH_FUSION_RANK).hits]
        self.assertEqual(hits[0], both)
        self.assertIn(search_only, hits)
        self.assertLess(hits.index(both), hits.index(search_only))

    def test_agreement_beats_a_single_models_favourite_even_at_rank_two(self):
        agreed = self.add_both(unit(0.9, 0.1), unit(0.1, 0.9))   # second in both lists
        favourite = self.add_photo(unit(1.0))                    # first in one list only
        self.add_class_vector(self.add_photo(unit(0.0, 0.0, 1.0)), unit(0.0, 1.0))
        hits = [fid for fid, _score in self.fused(SEARCH_FUSION_RANK).hits]
        self.assertLess(hits.index(agreed), hits.index(favourite))

    def test_both_indexes_are_named_as_having_ranked(self):
        self.add_both(unit(1.0), unit(0.0, 1.0))
        answer = self.fused(SEARCH_FUSION_RANK)
        self.assertEqual(set(answer.used), {self.model, self.class_model})
        self.assertEqual(answer.missing, {})


class TestUnionKeepsWhatOnlyOneModelFound(FusionTestBase):
    def test_the_answer_holds_frames_that_are_not_in_the_other_models_top(self):
        # One frame lives in each index and in neither of the other's lists, so a merge
        # that dropped either of them would be a merge in name only.
        search_only = self.add_photo(unit(1.0))
        class_only = self.add_class_vector(self.add_photo(None), unit(0.0, 1.0))
        hits = [fid for fid, _score in self.fused(SEARCH_FUSION_UNION).hits]
        self.assertIn(search_only, hits)
        self.assertIn(class_only, hits)

    def test_a_frame_only_one_model_ranked_first_is_not_pushed_out_by_agreement(self):
        # The difference from `rank`, stated as a test: with a union the two firsts share
        # the head of the list (file_id breaks the tie), because a best place is a best
        # place whichever model gave it.
        agreed = self.add_both(unit(0.9, 0.1), unit(0.1, 0.9))
        favourite = self.add_photo(unit(1.0))
        self.add_class_vector(self.add_photo(unit(0.0, 0.0, 1.0)), unit(0.0, 1.0))
        hits = [fid for fid, _score in self.fused(SEARCH_FUSION_UNION).hits]
        self.assertLess(hits.index(favourite), hits.index(agreed))

    def test_the_limit_cuts_the_merged_list_like_any_other_sample_size(self):
        for i in range(5):
            self.add_both(unit(1.0, 0.1 * i), unit(0.0, 1.0, 0.1 * i))
        self.assertEqual(len(self.fused(SEARCH_FUSION_UNION, limit=3).hits), 3)


class TestScoresOfTwoModelsAreNeverPutTogether(FusionTestBase):
    """The invariant the whole feature is built around, checked from three sides."""

    def test_the_merge_is_declared_over_file_ids_and_nothing_else(self):
        # The invariant made mechanical rather than remembered: a function whose input is
        # a list of ids cannot add, average or compare two cosines, whatever anybody
        # later believes about them.
        self.assertEqual(
            inspect.signature(search.fuse).parameters["rankings"].annotation,
            "Sequence[Sequence[int]]")

    def test_the_weight_is_the_rank_formula_and_holds_no_cosine(self):
        merged = dict(search.fuse([[7, 3], [3, 9]], SEARCH_FUSION_RANK, 10))
        self.assertAlmostEqual(merged[3], 1 / (search.RRF_K + 2) + 1 / (search.RRF_K + 1),
                               places=12)
        self.assertAlmostEqual(merged[7], 1 / (search.RRF_K + 1), places=12)
        self.assertAlmostEqual(merged[9], 1 / (search.RRF_K + 2), places=12)
        self.assertEqual(sorted(merged, key=lambda fid: -merged[fid]), [3, 7, 9])

    def test_the_places_decide_the_order_and_the_size_of_the_numbers_does_not(self):
        # Positions: `top` is 1st and 2nd, `loud` is 3rd and 1st. The cosines behind them
        # are chosen so that ADDING them would answer the other way round — which is the
        # arithmetic this feature is not allowed to do, in two spaces that are not
        # comparable to begin with.
        top = self.add_both(search_at(0.50), class_at(0.20))
        loud = self.add_both(search_at(0.45), class_at(0.99))
        middle = self.add_both(search_at(0.47), class_at(0.10))
        self.assertGreater(0.45 + 0.99, 0.50 + 0.20)  # what a sum would have said
        self.assertEqual([fid for fid, _score in self.fused(SEARCH_FUSION_RANK).hits],
                         [top, loud, middle])

    def test_a_fused_weight_is_never_a_sum_of_the_two_cosines(self):
        file_id = self.add_both(unit(1.0), unit(0.0, 1.0))
        (_fid, weight), = self.fused(SEARCH_FUSION_RANK).hits
        self.assertAlmostEqual(weight, 2.0 / (search.RRF_K + 1), places=9)
        self.assertNotAlmostEqual(weight, 2.0, places=3)   # 1.0 + 1.0, the mistake
        self.assertNotAlmostEqual(weight, 1.0, places=3)   # its average, the same mistake
        self.assertEqual(file_id, _fid)

    def test_union_takes_the_best_place_rather_than_the_better_score(self):
        weights = dict(search.fuse([[1, 2], [2, 1]], SEARCH_FUSION_UNION, 10))
        self.assertAlmostEqual(weights[1], 1.0 / (search.RRF_K + 1), places=9)
        self.assertAlmostEqual(weights[2], 1.0 / (search.RRF_K + 1), places=9)

    def test_an_unknown_mode_raises_instead_of_ranking_by_something_else(self):
        with self.assertRaises(ValueError):
            search.fuse([[1, 2]], "sum", 10)
        with self.assertRaises(ValueError):
            search.fuse([[1, 2]], SEARCH_FUSION_OFF, 10)


class TestFuseIsDeterministic(FusionTestBase):
    def test_ties_are_broken_by_file_id(self):
        self.assertEqual([fid for fid, _w in search.fuse([[9, 4, 7]], SEARCH_FUSION_UNION,
                                                         10)], [9, 4, 7])
        # Three frames each ranked first by nobody but tied on weight: only file_id can
        # pin the order, and a ranking that reshuffles between runs cannot be measured.
        tied = search.fuse([[5], [2], [8]], SEARCH_FUSION_RANK, 10)
        self.assertEqual([fid for fid, _w in tied], [2, 5, 8])

    def test_the_same_query_gives_the_same_merged_list_every_time(self):
        for i in range(4):
            self.add_both(unit(1.0, 0.1 * i), unit(0.0, 1.0, 0.1 * i))
        first = self.fused(SEARCH_FUSION_RANK).hits
        for _attempt in range(3):
            self.assertEqual(self.fused(SEARCH_FUSION_RANK).hits, first)

    def test_an_empty_ranking_list_is_an_empty_answer_and_not_a_crash(self):
        self.assertEqual(search.fuse([], SEARCH_FUSION_RANK, 10), [])
        self.assertEqual(search.fuse([[1, 2]], SEARCH_FUSION_RANK, 0), [])


class TestOneIndexEmpty(FusionTestBase):
    """The other index answers — out loud, because a halved merge looks like a whole one."""

    def test_without_a_classification_vector_the_search_index_still_answers(self):
        file_id = self.add_photo(unit(1.0))
        with self.assertLogs("sorta.search", level="WARNING") as logged:
            answer = self.fused(SEARCH_FUSION_RANK)
        self.assertEqual([fid for fid, _score in answer.hits], [file_id])
        self.assertEqual(answer.used, (self.model,))
        self.assertEqual(answer.missing, {self.class_model: search.REASON_EMPTY})
        self.assertIn(self.class_model, "".join(logged.output))

    def test_without_a_search_vector_the_classification_index_still_answers(self):
        file_id = self.add_class_vector(self.add_photo(None), unit(0.0, 1.0))
        with self.assertLogs("sorta.search", level="WARNING") as logged:
            answer = self.fused(SEARCH_FUSION_RANK)
        self.assertEqual([fid for fid, _score in answer.hits], [file_id])
        self.assertEqual(answer.used, (self.class_model,))
        self.assertEqual(answer.missing, {self.model: search.REASON_EMPTY})
        self.assertIn(self.model, "".join(logged.output))

    def test_neither_index_is_the_refusal_the_interface_already_knows(self):
        self.add_photo(None)
        with self.assertRaises(search.EmbeddingsMissing) as ctx:
            self.fused(SEARCH_FUSION_RANK)
        self.assertEqual(ctx.exception.reason, search.REASON_EMPTY)
        self.assertEqual(ctx.exception.model, self.model)

    def test_a_query_of_the_wrong_width_leaves_that_index_out_rather_than_ranking(self):
        # The last guard against a query encoded by the other tower: 8 numbers against a
        # stored 5 cannot rank, so that index reports nothing instead of a plausible list.
        file_id = self.add_photo(unit(1.0))
        self.add_class_vector(file_id, np.ones(5, dtype=np.float32))
        with self.assertLogs("sorta.search", level="WARNING"):
            answer = self.fused(SEARCH_FUSION_RANK)
        self.assertEqual(answer.used, (self.model,))
        self.assertEqual(answer.missing, {self.class_model: search.REASON_EMPTY})


class TestAnotherModelTakesPartInNeitherIndex(FusionTestBase):
    """F128's rule, applied to both tables — a merge must not be the way around it."""

    def test_a_classification_row_of_another_model_is_not_in_the_output(self):
        ours = self.add_both(unit(0.0, 1.0), unit(0.0, 1.0))
        # A frame the search index never saw, whose only vector was computed by a model
        # nobody asked about: the merge is not a way around the filter that keeps it out.
        theirs = self.add_class_vector(self.add_photo(None), unit(0.0, 1.0),
                                       model="other/model")
        hits = [fid for fid, _score in self.fused(SEARCH_FUSION_UNION).hits]
        self.assertIn(ours, hits)
        self.assertNotIn(theirs, hits)

    def test_a_search_row_of_another_model_is_not_in_the_output(self):
        ours = self.add_both(unit(0.0, 1.0), unit(0.0, 1.0))
        theirs = self.add_photo(unit(1.0), model="other/model")
        hits = [fid for fid, _score in self.fused(SEARCH_FUSION_UNION).hits]
        self.assertIn(ours, hits)
        self.assertNotIn(theirs, hits)

    def test_a_classification_table_of_another_model_only_is_named_as_such(self):
        self.add_class_vector(self.add_photo(unit(1.0)), unit(0.0, 1.0),
                              model="other/model")
        with self.assertLogs("sorta.search", level="WARNING"):
            answer = self.fused(SEARCH_FUSION_RANK)
        self.assertEqual(answer.missing, {self.class_model: search.REASON_OTHER_MODEL})


class TestSearchTextIsTheSameCall(FusionTestBase):
    """The interface does not move: the merge happens under the call that exists."""

    def test_search_text_returns_the_merged_list(self):
        both = self.add_both(unit(1.0), unit(0.0, 1.0))
        self.add_photo(unit(0.9, 0.4))
        hits = search.search_text(
            self.cfg_with(SEARCH_FUSION_RANK), self.conn, "cake",
            encoder=encoder_for({"cake": unit(1.0)}),
            class_encoder=encoder_for({"cake": unit(0.0, 1.0)}))
        self.assertEqual(hits, self.fused(SEARCH_FUSION_RANK).hits)
        self.assertEqual(hits[0][0], both)

    def test_the_configured_sample_size_is_the_depth_of_the_merge(self):
        for i in range(6):
            self.add_both(unit(1.0, 0.1 * i), unit(0.0, 1.0, 0.1 * i))
        hits = search.search_text(
            self.cfg_with(SEARCH_FUSION_UNION, limit=2), self.conn, "cake",
            encoder=encoder_for({}), class_encoder=encoder_for({}))
        self.assertEqual(len(hits), 2)
