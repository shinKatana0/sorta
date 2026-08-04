"""F140: the screenshots and receipts the classifier took for photographs.

The feature is a GATE, not a verdict, and every case below is about one half of that
sentence. What it promises, and therefore what is asserted:

* the score is computed from the STORED vector and written for every photograph that has
  one; a frame without a vector gets NULL and is a candidate at no threshold;
* with `features.junk_rescue` off nothing is computed at all and no verdict moves — the
  run is byte-for-byte the one that existed before this feature;
* with it on and the deep tier OFF no verdict moves either: the score is stored, the
  candidates are counted, and that is the state the feature is meant to be tried in;
* with both on, the frames shown to the model are the ones above
  `features.junk_rescue_threshold` and only those — the number of calls equals the number
  of candidates;
* only the model's answer moves a verdict. A refusal — an unreadable answer, a model that
  raises, a model or an encoder that will not build — leaves the fast verdict, never
  "junk", which is the whole reason the 85%-accurate score is not applied directly;
* a frame that is already junk is never a candidate: the population is what the stage
  called a photograph;
* editing either prompt (the score's or the model's) invalidates the stored scores, so a
  later run recomputes instead of keeping an answer to a question nobody asks any more.

No model is loaded anywhere: the classifier, the text encoder and the asker are injected,
exactly as the rest of the junk suite does it.
"""
from __future__ import annotations

import math
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from sorta import junk
from sorta.config import Config, FeaturesConfig, _naming_from
from sorta.junk import (
    JUNK_RESCUE_DOCUMENT,
    JUNK_RESCUE_PHOTO,
    JUNK_RESCUE_SCREENSHOT,
    classify,
    junk_rescue_prompts,
    junk_rescue_score,
    parse_junk_rescue_answer,
    read_frame_quality,
    unit_rows,
)
from tests.test_clip_embeddings import EmbeddingClassifier
from tests.test_frame_quality import FrameQualityCase
from tests.test_junk import NO_OCR


def vector_for(score: float) -> np.ndarray:
    """A two-dimensional unit vector whose rescue score is exactly `score`.

    With the positive prompts at [1, 0] and the photograph prompt at [0, 1] (see
    `FakeTextEncoder`), the score of a unit vector [a, b] is a - b. Solving a - b = score
    on the unit circle gives the pair below, so a case can name the number it is about
    instead of a direction that happens to produce one.
    """
    b = (-score + math.sqrt(2.0 - score * score)) / 2.0
    return np.array([b + score, b], dtype=np.float32)


class FakeTextEncoder:
    """The text tower, without a model: junk prompts one way, the photograph the other.

    Built off `junk._JUNK_RESCUE_POS_PROMPTS` rather than off a hard-coded count, so a case
    that edits the prompt list (the invalidation cases) still gets a consistent encoder.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, prompts):
        self.calls.append(tuple(prompts))
        positives = len(junk._JUNK_RESCUE_POS_PROMPTS)
        rows = ([[1.0, 0.0]] * positives
                + [[0.0, 1.0]] * (len(prompts) - positives))
        return np.asarray(rows, dtype=np.float32)


class Asker:
    """A rescue asker that answers per frame and remembers what it was shown."""

    def __init__(self, answers: dict[str, str], default: str = "photo",
                 boom: tuple[str, ...] = ()) -> None:
        self.answers = answers
        self.default = default
        self.boom = set(boom)
        self.asked: list[str] = []

    def __call__(self, path: str) -> str:
        name = Path(path).name
        self.asked.append(name)
        if name in self.boom:
            raise RuntimeError("CUDA error: device-side assert triggered")
        return self.answers.get(name, self.default)


class RescueCase(FrameQualityCase):
    """The fixture of the file: the rescue on, the deep tier off, nothing else."""

    def setUp(self):
        super().setUp()
        self.features(junk_rescue=True, junk_rescue_threshold=0.02)
        self.encoder = FakeTextEncoder()

    def deep_tier_on(self):
        """Switch the deep tier on — the condition under which candidates are asked."""
        self.cfg.naming = _naming_from({"vlm_enabled": True})

    def run_stage(self, scores: dict[str, float], asker=None, undecodable=(),
                  products=(), **kwargs):
        """One classify() over one frame per named score, with everything injected.

        `products` names the frames the product-CLIP calls a product — the only way into
        the DEEP tier's own candidate gate, which is otherwise shut for these frames.
        """
        for name in scores:
            self.add_file(name)
        clf = EmbeddingClassifier(
            vectors={name: vector_for(value) for name, value in scores.items()},
            undecodable=undecodable)
        clf.prod_scores = {name: (junk._N_PROD_ANTI, 0.9) for name in products}
        kwargs.setdefault("junk_text_encoder", self.encoder)
        if self.cfg.naming.vlm_enabled and "vlm_classifier" not in kwargs:
            # The DEEP tier's own 3-way classifier: injected so no case here loads a model
            # for it, and answering `personal_photo` so it moves nothing (its candidate
            # gate is closed for these frames anyway — their document and product scores
            # are at the floor).
            kwargs["vlm_classifier"] = lambda _path: "personal_photo"
        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         sharpness_detector=lambda _p, _faces: junk.Sharpness(500.0),
                         junk_rescue_vlm=asker, **kwargs)
        return stats, clf

    def file_id(self, name):
        return self.conn.execute(
            "SELECT id FROM files WHERE path = ?", (f"/photos/{name}",)).fetchone()[0]

    def score_of(self, name):
        row = self.conn.execute(
            "SELECT fq.junk_score FROM frame_quality fq JOIN files f ON f.id = fq.file_id"
            " WHERE f.path = ?", (f"/photos/{name}",)).fetchone()
        return None if row is None else row["junk_score"]

    def verdict_of(self, name):
        return self.media_class(self.file_id(name))["verdict"]


class TestTheScoreItself(unittest.TestCase):
    """The arithmetic, stated once: a margin over the photograph prompt."""

    def encoder_rows(self, positives=None):
        positives = positives if positives is not None else len(
            junk._JUNK_RESCUE_POS_PROMPTS)
        return np.asarray([[1.0, 0.0]] * positives + [[0.0, 1.0]], dtype=np.float32)

    def test_the_score_is_the_difference_of_the_two_maxima(self):
        features = self.encoder_rows()
        for wanted in (0.5, 0.05, 0.0, -0.3):
            with self.subTest(score=wanted):
                self.assertAlmostEqual(
                    junk_rescue_score(vector_for(wanted), features), wanted, places=5)

    def test_a_vector_of_another_width_is_no_score_at_all(self):
        """A number computed across two spaces would look exactly like a real one — which
        is the single thing a selection signal must not do."""
        self.assertIsNone(
            junk_rescue_score(np.ones(8, dtype=np.float32), self.encoder_rows()))

    def test_the_positives_come_first_and_the_split_follows_them(self):
        prompts = junk_rescue_prompts()
        self.assertEqual(prompts[:len(junk._JUNK_RESCUE_POS_PROMPTS)],
                         list(junk._JUNK_RESCUE_POS_PROMPTS))
        self.assertEqual(prompts[len(junk._JUNK_RESCUE_POS_PROMPTS):],
                         list(junk._JUNK_RESCUE_NEG_PROMPTS))

    def test_the_prompts_are_the_stages_own_junk_classes(self):
        """The score is a statement about the question the stage already asks; a second
        wording of "a screenshot" would be a second definition of one."""
        for cls in ("screenshot", "meme"):
            self.assertIn(dict(junk._CLIP_CLASSES)[cls], junk._JUNK_RESCUE_POS_PROMPTS)
        self.assertEqual(junk._JUNK_RESCUE_NEG_PROMPTS,
                         (dict(junk._CLIP_CLASSES)["photo"],))

    def test_rows_come_back_normalized_so_a_dot_product_is_a_cosine(self):
        rows = unit_rows(np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(rows[0])), 1.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(rows[1])), 0.0)  # nothing to preserve


class TestAnswerParsing(unittest.TestCase):
    """Read leniently — and "unreadable" is not "junk"."""

    def test_the_three_expected_answers(self):
        self.assertEqual(parse_junk_rescue_answer("screenshot"), JUNK_RESCUE_SCREENSHOT)
        self.assertEqual(parse_junk_rescue_answer("document"), JUNK_RESCUE_DOCUMENT)
        self.assertEqual(parse_junk_rescue_answer("photo"), JUNK_RESCUE_PHOTO)

    def test_case_and_punctuation_are_not_a_format(self):
        self.assertEqual(parse_junk_rescue_answer("Screenshot."), JUNK_RESCUE_SCREENSHOT)
        self.assertEqual(parse_junk_rescue_answer("  PHOTO!  "), JUNK_RESCUE_PHOTO)

    def test_an_explanation_that_also_says_photo_is_read_as_the_junk_class(self):
        """The reason `_JUNK_RESCUE_KEYWORDS` is ordered: `photo` is the word a model
        reaches for while describing one of the other two."""
        self.assertEqual(parse_junk_rescue_answer("a photo of a receipt"),
                         JUNK_RESCUE_DOCUMENT)
        self.assertEqual(parse_junk_rescue_answer("this is a photo of a screen"),
                         JUNK_RESCUE_SCREENSHOT)

    def test_a_photograph_is_still_a_photo(self):
        self.assertEqual(parse_junk_rescue_answer("an ordinary photograph"),
                         JUNK_RESCUE_PHOTO)

    def test_an_unreadable_answer_is_none_and_none_is_not_junk(self):
        for answer in ("", "I cannot help with that", "42", "да"):
            with self.subTest(answer=answer):
                self.assertIsNone(parse_junk_rescue_answer(answer))


class TestTheScoreIsStored(RescueCase):
    """Brief test 1: written for every photograph with a vector, NULL without one."""

    def test_every_frame_with_an_embedding_gets_its_score(self):
        stats, _clf = self.run_stage({"meme.jpg": 0.4, "family.jpg": -0.2})
        self.assertAlmostEqual(self.score_of("meme.jpg"), 0.4, places=5)
        self.assertAlmostEqual(self.score_of("family.jpg"), -0.2, places=5)
        self.assertEqual(stats.junk_scored, 2)

    def test_a_frame_without_an_embedding_keeps_a_null(self):
        """`store_embeddings: false`, a heuristics-only collection, a frame that would not
        decode — all the same state: no vector, so no score, and no candidacy either."""
        stats, _clf = self.run_stage({"meme.jpg": 0.4, "gone.jpg": 0.9},
                                     undecodable=("gone.jpg",))
        self.assertIsNone(self.score_of("gone.jpg"))
        self.assertEqual(stats.junk_scored, 1)
        self.assertEqual(stats.junk_candidates, 1)  # the vectorless frame is not one

    def test_the_score_is_readable_back_out(self):
        self.run_stage({"meme.jpg": 0.4})
        (row,) = read_frame_quality(self.conn).values()
        self.assertAlmostEqual(row.junk_score, 0.4, places=5)

    def test_the_embeddings_are_not_recomputed_for_it(self):
        """The whole economy of the feature: the vectors are read from the table, and the
        classifier is asked for its cache once per chunk, as it was before F140."""
        _stats, clf = self.run_stage({"a.jpg": 0.4, "b.jpg": -0.1})
        self.assertEqual(len(clf.feature_calls), 1)

    def test_storing_no_embeddings_leaves_the_feature_silent(self):
        self.features(junk_rescue=True, junk_rescue_threshold=0.02,
                      store_embeddings=False)
        stats, _clf = self.run_stage({"meme.jpg": 0.9})
        self.assertIsNone(self.score_of("meme.jpg"))
        self.assertEqual((stats.junk_scored, stats.junk_candidates), (0, 0))
        self.assertEqual(self.verdict_of("meme.jpg"), "photo")


class TestToggleOff(RescueCase):
    """Brief test 2: with the toggle off the run is what it was, down to the encoder."""

    def setUp(self):
        super().setUp()
        self.features(junk_rescue=False)

    def test_no_verdict_moves_and_nothing_is_scored(self):
        stats, _clf = self.run_stage({"meme.jpg": 0.9, "family.jpg": -0.3})
        for name in ("meme.jpg", "family.jpg"):
            self.assertEqual(self.verdict_of(name), "photo")
            self.assertIsNone(self.score_of(name))
        self.assertEqual((stats.junk_scored, stats.junk_candidates, stats.junk_rescued),
                         (0, 0, 0))

    def test_no_encoder_is_built(self):
        def factory(_settings):
            raise AssertionError("no encoder may be built with junk_rescue off")

        self.run_stage({"meme.jpg": 0.9}, junk_text_encoder=None,
                       junk_text_encoder_factory=factory)

    def test_no_model_is_built_either(self):
        self.deep_tier_on()

        def factory(_model):
            raise AssertionError("no model may be built with junk_rescue off")

        self.run_stage({"meme.jpg": 0.9}, junk_rescue_vlm_factory=factory)

    def test_the_row_marker_is_the_one_that_existed_before(self):
        """A collection that never asked this question must not be invalidated by it."""
        self.assertEqual(junk._quality_source(True, False, None), "classic")
        self.assertNotEqual(junk._quality_source(True, False, None, rescue=True),
                            "classic")


class TestWithoutTheDeepTier(RescueCase):
    """Brief test 3: the score is written and the candidates counted; nothing else moves."""

    def test_verdicts_stay_and_candidates_are_only_marked(self):
        stats, _clf = self.run_stage({"meme.jpg": 0.9, "screen.jpg": 0.05,
                                      "family.jpg": -0.3})
        for name in ("meme.jpg", "screen.jpg", "family.jpg"):
            self.assertEqual(self.verdict_of(name), "photo")
        self.assertEqual(stats.junk_candidates, 2)
        self.assertEqual(stats.junk_rescued, 0)
        self.assertEqual(stats.junk_scored, 3)

    def test_no_model_is_built_without_the_tier(self):
        def factory(_model):
            raise AssertionError("the rescue must not build a model without the deep tier")

        self.run_stage({"meme.jpg": 0.9}, junk_rescue_vlm_factory=factory)

    def test_the_candidate_still_keeps_its_quality_row(self):
        """It is still a photograph as far as this run is concerned, and every other
        signal about it stays exactly where it was."""
        self.run_stage({"meme.jpg": 0.9})
        row = self.quality(self.file_id("meme.jpg"))
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row["sharpness"], 500.0)


class TestTheCandidateGate(RescueCase):
    """Brief test 4: the model sees the frames above the threshold, and no more."""

    def setUp(self):
        super().setUp()
        self.deep_tier_on()

    def test_only_the_candidates_are_asked_and_each_exactly_once(self):
        asker = Asker({})
        scores = {"high.jpg": 0.4, "mid.jpg": 0.05, "under.jpg": 0.01,
                  "family.jpg": -0.3}
        stats, _clf = self.run_stage(scores, asker=asker)
        self.assertEqual(sorted(asker.asked), ["high.jpg", "mid.jpg"])
        self.assertEqual(len(asker.asked), stats.junk_candidates)

    def test_the_threshold_comes_from_the_config(self):
        self.features(junk_rescue=True, junk_rescue_threshold=0.3)
        asker = Asker({})
        self.run_stage({"high.jpg": 0.4, "mid.jpg": 0.05}, asker=asker)
        self.assertEqual(asker.asked, ["high.jpg"])

    def test_the_threshold_is_inclusive(self):
        self.features(junk_rescue=True, junk_rescue_threshold=0.25)
        asker = Asker({})
        self.run_stage({"edge.jpg": 0.25}, asker=asker)
        self.assertEqual(asker.asked, ["edge.jpg"])

    def test_nothing_above_the_threshold_asks_nothing(self):
        asker = Asker({})
        stats, _clf = self.run_stage({"family.jpg": -0.3}, asker=asker)
        self.assertEqual(asker.asked, [])
        self.assertEqual(stats.junk_candidates, 0)


class TestTheModelDecides(RescueCase):
    """Brief test 5: the answer moves the verdict, a refusal leaves it alone."""

    def setUp(self):
        super().setUp()
        self.deep_tier_on()

    def test_a_screenshot_answer_moves_the_verdict(self):
        asker = Asker({"meme.jpg": "screenshot"})
        stats, _clf = self.run_stage({"meme.jpg": 0.4}, asker=asker)
        row = self.media_class(self.file_id("meme.jpg"))
        self.assertEqual(row["verdict"], "screenshot")
        self.assertEqual(row["source"], "vlm")
        self.assertEqual(stats.junk_rescued, 1)
        self.assertEqual(stats.by_verdict.get("screenshot"), 1)
        self.assertEqual(stats.by_verdict.get("photo"), 0)

    def test_a_receipt_becomes_a_document(self):
        asker = Asker({"bill.jpg": "document"})
        self.run_stage({"bill.jpg": 0.4}, asker=asker)
        self.assertEqual(self.verdict_of("bill.jpg"), "document")

    def test_a_reclassified_frame_loses_its_quality_row_and_its_vector(self):
        """The F120 population rule, and it has to survive a verdict that moves at the very
        end of the stage: a screenshot has no quality row and no vector in the search."""
        asker = Asker({"meme.jpg": "screenshot"})
        self.run_stage({"meme.jpg": 0.4}, asker=asker)
        file_id = self.file_id("meme.jpg")
        self.assertIsNone(self.quality(file_id))
        self.assertIsNone(self.conn.execute(
            "SELECT 1 FROM clip_embeddings WHERE file_id = ?", (file_id,)).fetchone())

    def test_the_model_calling_it_a_photograph_keeps_it_one(self):
        """~15% of the candidates in the measured band are real photographs, and this is
        the branch that saves them: the score selected the frame, the model overruled it."""
        asker = Asker({"beach.jpg": "photo"})
        stats, _clf = self.run_stage({"beach.jpg": 0.4}, asker=asker)
        self.assertEqual(self.verdict_of("beach.jpg"), "photo")
        self.assertEqual(stats.junk_rescued, 0)
        self.assertIsNotNone(self.quality(self.file_id("beach.jpg")))

    def test_an_unreadable_answer_leaves_the_fast_verdict(self):
        asker = Asker({"meme.jpg": "I'm not sure what this is"})
        stats, _clf = self.run_stage({"meme.jpg": 0.4}, asker=asker)
        self.assertEqual(self.verdict_of("meme.jpg"), "photo")
        self.assertEqual(stats.junk_rescued, 0)

    def test_a_failure_on_one_frame_costs_only_that_frame(self):
        asker = Asker({"fine.jpg": "screenshot"}, boom=("boom.jpg",))
        stats, _clf = self.run_stage({"boom.jpg": 0.4, "fine.jpg": 0.4}, asker=asker)
        self.assertEqual(self.verdict_of("boom.jpg"), "photo")
        self.assertEqual(self.verdict_of("fine.jpg"), "screenshot")
        self.assertEqual(stats.junk_rescued, 1)

    def test_a_model_that_will_not_build_leaves_every_verdict_alone(self):
        def broken(_model):
            raise RuntimeError("transformers not installed")

        stats, _clf = self.run_stage({"meme.jpg": 0.4}, junk_rescue_vlm_factory=broken)
        self.assertEqual(self.verdict_of("meme.jpg"), "photo")
        self.assertEqual(stats.junk_candidates, 1)   # the score still did its half
        self.assertAlmostEqual(self.score_of("meme.jpg"), 0.4, places=5)

    def test_an_encoder_that_will_not_build_leaves_every_verdict_alone(self):
        def broken(_settings):
            raise RuntimeError("open_clip is not installed")

        asker = Asker({"meme.jpg": "screenshot"})
        stats, _clf = self.run_stage({"meme.jpg": 0.4}, asker=asker,
                                     junk_text_encoder=None,
                                     junk_text_encoder_factory=broken)
        self.assertEqual(self.verdict_of("meme.jpg"), "photo")
        self.assertIsNone(self.score_of("meme.jpg"))
        self.assertEqual(asker.asked, [])
        self.assertEqual((stats.junk_scored, stats.junk_candidates), (0, 0))


class TestPopulation(RescueCase):
    """Brief test 7: what is already junk is not a candidate — it is what we look for."""

    def setUp(self):
        super().setUp()
        self.deep_tier_on()

    def test_a_frame_the_fast_tier_already_called_a_screenshot_is_never_asked(self):
        asker = Asker({})
        self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)
        self.add_file("meme.jpg")
        clf = EmbeddingClassifier(vectors={"Screenshot_1.png": vector_for(0.9),
                                           "meme.jpg": vector_for(0.4)})
        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         sharpness_detector=lambda _p, _faces: junk.Sharpness(500.0),
                         junk_text_encoder=self.encoder, junk_rescue_vlm=asker,
                         vlm_classifier=lambda _p: "personal_photo")
        self.assertEqual(self.verdict_of("Screenshot_1.png"), "screenshot")
        self.assertEqual(asker.asked, ["meme.jpg"])
        self.assertEqual(stats.junk_candidates, 1)
        self.assertIsNone(self.score_of("Screenshot_1.png"))

    def test_a_frame_the_deep_tier_reclassified_is_not_asked_either(self):
        """The rescue runs after the deep tier, which is the one thing that can move a
        verdict while its own list is being built — and `document` is precisely the class
        `vlm.exclude_classes` protects by default."""
        asker = Asker({})
        stats, _clf = self.run_stage(
            {"form.jpg": 0.4, "meme.jpg": 0.4}, asker=asker, products=("form.jpg",),
            vlm_classifier=lambda path: (
                "document" if path.endswith("form.jpg") else "personal_photo"))
        self.assertEqual(self.verdict_of("form.jpg"), "document")
        self.assertEqual(asker.asked, ["meme.jpg"])
        self.assertEqual(stats.junk_candidates, 1)

    def test_a_heuristics_only_run_scores_nothing(self):
        def factory(_settings):
            raise AssertionError("no encoder may be built on a heuristics-only run")

        self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)
        classify(self.cfg, self.conn, use_clip=False, junk_text_encoder_factory=factory,
                 sharpness_detector=lambda _p, _faces: junk.Sharpness(1.0))
        self.assertIsNone(self.conn.execute("SELECT 1 FROM frame_quality").fetchone())


class TestIncrementality(RescueCase):
    """Brief test 6: remembered between runs, recomputed when a question moves."""

    def setUp(self):
        super().setUp()
        self.deep_tier_on()

    def rerun(self, scores, asker=None):
        clf = EmbeddingClassifier(
            vectors={name: vector_for(value) for name, value in scores.items()})
        return classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                        sharpness_detector=lambda _p, _faces: junk.Sharpness(500.0),
                        junk_text_encoder=self.encoder, junk_rescue_vlm=asker,
                        vlm_classifier=lambda _p: "personal_photo")

    def test_the_second_run_asks_nothing_again(self):
        asker = Asker({"beach.jpg": "photo"})
        self.run_stage({"beach.jpg": 0.4}, asker=asker)
        self.assertEqual(len(asker.asked), 1)

        stats = self.rerun({"beach.jpg": 0.4}, asker=asker)
        self.assertEqual(len(asker.asked), 1)  # ~12 minutes not paid a second time
        self.assertEqual((stats.junk_scored, stats.junk_candidates), (0, 0))
        self.assertAlmostEqual(self.score_of("beach.jpg"), 0.4, places=5)

    def test_editing_the_score_prompts_recomputes_the_scores(self):
        asker = Asker({"beach.jpg": "photo"})
        self.run_stage({"beach.jpg": 0.4}, asker=asker)
        before = self.conn.execute("SELECT source FROM frame_quality").fetchone()[0]

        with unittest.mock.patch.object(
                junk, "_JUNK_RESCUE_POS_PROMPTS",
                junk._JUNK_RESCUE_POS_PROMPTS + ("a photo of a price tag",)):
            stats = self.rerun({"beach.jpg": 0.4}, asker=asker)
        after = self.conn.execute("SELECT source FROM frame_quality").fetchone()[0]
        self.assertNotEqual(before, after)
        self.assertEqual(stats.junk_scored, 1)
        self.assertEqual(len(asker.asked), 2)

    def test_editing_the_model_question_recomputes_them_too(self):
        asker = Asker({"beach.jpg": "photo"})
        self.run_stage({"beach.jpg": 0.4}, asker=asker)

        with unittest.mock.patch.object(junk, "_JUNK_RESCUE_PROMPT",
                                        "Is this a screenshot? Answer with one word."):
            stats = self.rerun({"beach.jpg": 0.4}, asker=asker)
        self.assertEqual(stats.junk_scored, 1)
        self.assertEqual(len(asker.asked), 2)

    def test_the_fingerprint_moves_only_when_the_question_is_asked(self):
        """A collection scored without the deep tier must not be invalidated by wording
        nobody used on it."""
        plain = junk.quality_prompt_fingerprint(False, rescue=True)
        with unittest.mock.patch.object(junk, "_JUNK_RESCUE_PROMPT", "something else"):
            self.assertEqual(
                junk.quality_prompt_fingerprint(False, rescue=True),
                plain)
            moved = junk.quality_prompt_fingerprint(False, rescue=True,
                                                    rescue_vlm=True)
        self.assertNotEqual(moved, junk.quality_prompt_fingerprint(
            False, rescue=True, rescue_vlm=True))

    def test_switching_the_check_on_marks_the_rows_as_the_model_tier(self):
        source = junk._quality_source(True, False, None, rescue=True,
                                      rescue_ask=lambda _p: "photo")
        self.assertEqual(junk.quality_tier(source), junk.QUALITY_SOURCE_VLM)
        # and it differs from the marker the score alone writes, so switching the deep
        # tier on reprocesses instead of looking done
        self.assertNotEqual(source, junk._quality_source(True, False, None, rescue=True))


class TestSettings(unittest.TestCase):
    """The config reaches the stage, and the defaults are the measured ones."""

    def test_the_config_reaches_the_stage_settings(self):
        cfg = Config(features=FeaturesConfig(junk_rescue=True,
                                             junk_rescue_threshold=0.05))
        q = junk.quality_settings(cfg)
        self.assertTrue(q.junk_rescue)
        self.assertAlmostEqual(q.junk_rescue_threshold, 0.05)

    def test_the_defaults_are_off_and_the_measured_threshold(self):
        d = FeaturesConfig()
        self.assertFalse(d.junk_rescue)
        # +0.02 is the reviewed row: 955 frames, ~12 minutes of the deep tier, with ~17%
        # real photographs still in the band above it — which is why the model decides.
        self.assertAlmostEqual(d.junk_rescue_threshold, 0.02)

    def test_garbage_in_the_config_does_not_switch_the_feature_on(self):
        from sorta.config import _features_from

        parsed = _features_from({"junk_rescue": "false",
                                 "junk_rescue_threshold": "not a number"})
        self.assertFalse(parsed.junk_rescue)
        self.assertAlmostEqual(parsed.junk_rescue_threshold, 0.02)


class TestTheAsker(unittest.TestCase):
    """The real asker over a fake runtime — one frame, one question, one word back."""

    def test_the_asker_decodes_the_frame_and_asks_one_question(self):
        import tempfile

        from PIL import Image

        seen: list[tuple[int, str, int]] = []

        def describe(frames, prompt, max_new_tokens):
            seen.append((len(frames), prompt, max_new_tokens))
            return "screenshot"

        ask = junk.vlm_junk_rescue_asker(describe, max_edge=128)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jpg"
            Image.new("RGB", (256, 192), (30, 60, 90)).save(path, "JPEG")
            answer = ask(str(path))
        self.assertEqual(parse_junk_rescue_answer(answer), JUNK_RESCUE_SCREENSHOT)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], 1)
        for word in ("screenshot", "document", "photo"):
            self.assertIn(word, seen[0][1])

    def test_a_missing_file_is_an_empty_answer_not_a_crash(self):
        def describe(_frames, _prompt, _tokens):
            raise AssertionError("a vanished frame must never reach the model")

        self.assertEqual(junk.vlm_junk_rescue_asker(describe, 128)("/nowhere/x.jpg"), "")


if __name__ == "__main__":
    unittest.main()
