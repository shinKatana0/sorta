"""F130: CLIP selects widely, the VLM checks — the animal label as a cascade.

What the feature promises, and therefore what is asserted below:

* with `features.pets_verify` off the run is EXACTLY today's — the label is
  `pet_score >= features.pet_threshold` and no frame reaches a model. That is an
  acceptance criterion of the brief, not a nicety, so it gets its own cases;
* with it on, the frames shown to the model are the ones between
  `features.pet_candidate_threshold` and the top, and only those — the number of calls
  equals the number of candidates;
* the answer OUTRANKS the score in both directions: `depiction` takes the label off a
  frame scored 0.95, `real` puts one on a frame below the threshold. The second is the
  recall the cascade exists for;
* everything that can go wrong with the expensive tier falls back to the cheap one and
  never to "no" — an answer that does not parse, a model that raises on a frame, a model
  that will not build at all;
* the answer is remembered (`frame_quality.pet_vlm`), so a second run asks nothing again,
  and editing the question invalidates what it produced;
* `pet_score` stays written for every frame, the rejected ones included: a threshold has
  to be re-choosable without a new pass.

No model is loaded anywhere: the classifier and the asker are injected, as everywhere else
in the junk suite.
"""
from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from sorta import junk
from sorta.config import FeaturesConfig, VlmConfig, _naming_from
from sorta.db import connect
from sorta.junk import (
    PET_VLM_DEPICTION,
    PET_VLM_NONE,
    PET_VLM_REAL,
    classify,
    parse_pet_answer,
    pet_label,
    read_frame_quality,
)
from tests.test_frame_quality import FrameQualityCase
from tests.test_junk import NO_OCR

_PET_CLASSES = [cls for cls, _prompt in junk._PET_CLASSES]
_CAT_IDX = len(junk._CLIP_CLASSES) + _PET_CLASSES.index("cat")


class PetClassifier:
    """A CLIP mock that gives each frame the pet score the case is about.

    Probabilities straight out, not logits: the identity that the junk classes survive the
    longer prompt list is tested in test_frame_quality and is not what these cases are
    about — here what matters is that a named frame comes out of the pet group with a
    known score, so the candidate gate can be asserted against an exact number.

    The junk row says "a photograph" confidently and the document pass says "not a
    document", unless the frame is named as one: then the fast verdict is `document` and
    the frame is expected to leave the quality population altogether. A frame named as a
    product is a candidate for the DEEP tier instead — its fast verdict stays `photo`,
    which is how a frame can be a pet candidate and be reclassified afterwards.
    """

    def __init__(self, scores: dict[str, float], documents: tuple[str, ...] = (),
                 products: tuple[str, ...] = ()):
        self.scores = scores
        self.documents = set(documents)
        self.products = set(products)
        self.calls: list[tuple[int, tuple[str, ...]]] = []

    def __call__(self, image_paths, prompts):
        self.calls.append((len(prompts), tuple(image_paths)))
        out = np.zeros((len(image_paths), len(prompts)), dtype=np.float32)
        for i, path in enumerate(image_paths):
            name = Path(path).name
            if len(prompts) == len(junk._DOCUMENT_CLASSES):
                # the anti-classes come first; a positive one only for a named document
                out[i, junk._N_DOC_ANTI if name in self.documents else 0] = 0.99
                continue
            if len(prompts) == len(junk._PRODUCT_CLASSES):
                out[i, junk._N_PROD_ANTI if name in self.products else 0] = 0.9
                continue
            out[i, 0] = 0.9          # "a photograph" — the junk group, confidently
            out[i, 1] = out[i, 2] = 0.05
            if len(prompts) > len(junk._CLIP_CLASSES):
                score = self.scores.get(name, 0.0)
                out[i, _CAT_IDX] = score
                out[i, -1] = 1.0 - score  # the last anti-class holds the rest
        return out


class Asker:
    """A pet asker that answers per frame and remembers what it was shown."""

    def __init__(self, answers: dict[str, str], default: str = "none",
                 boom: tuple[str, ...] = ()):
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


class PetCascadeCase(FrameQualityCase):
    """The fixture of the whole file: animals on, the check on, nothing else."""

    def setUp(self):
        super().setUp()
        self.features(pets=True, pets_verify=True, pet_threshold=0.7,
                      pet_candidate_threshold=0.3)

    def run_stage(self, scores, asker=None, **kwargs):
        """One classify() over one frame per score, with the pet asker injected."""
        for name in scores:
            self.add_file(name)
        clf = PetClassifier(scores, documents=kwargs.pop("documents", ()))
        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         sharpness_detector=lambda _p: 500.0, pet_vlm=asker, **kwargs)
        return stats, clf

    def label(self, name):
        row = self.conn.execute(
            "SELECT fq.pet, fq.pet_vlm, fq.pet_score FROM frame_quality fq"
            " JOIN files f ON f.id = fq.file_id WHERE f.path = ?",
            (f"/photos/{name}",)).fetchone()
        return row


class TestPetLabelRule(unittest.TestCase):
    """The rule itself, stated once: the model outranks the score, NULL falls back."""

    def test_the_model_outranks_a_high_score(self):
        self.assertIsNone(pet_label(PET_VLM_DEPICTION, 0.95, 0.7))
        self.assertIsNone(pet_label(PET_VLM_NONE, 0.99, 0.7))

    def test_the_model_outranks_a_low_score(self):
        self.assertEqual(pet_label(PET_VLM_REAL, 0.31, 0.7), junk.PET_CLASS)

    def test_an_unasked_frame_falls_back_to_the_threshold(self):
        self.assertEqual(pet_label(None, 0.8, 0.7), junk.PET_CLASS)
        self.assertIsNone(pet_label(None, 0.6, 0.7))

    def test_the_threshold_is_inclusive_as_it_always_was(self):
        self.assertEqual(pet_label(None, 0.7, 0.7), junk.PET_CLASS)

    def test_a_frame_without_a_score_has_no_label(self):
        self.assertIsNone(pet_label(None, None, 0.7))


class TestAnswerParsing(unittest.TestCase):
    """Read leniently — and "unreadable" is not "no animal"."""

    def test_the_three_expected_answers(self):
        self.assertEqual(parse_pet_answer("real"), PET_VLM_REAL)
        self.assertEqual(parse_pet_answer("depiction"), PET_VLM_DEPICTION)
        self.assertEqual(parse_pet_answer("none"), PET_VLM_NONE)

    def test_case_and_punctuation_are_not_a_format(self):
        self.assertEqual(parse_pet_answer("Real."), PET_VLM_REAL)
        self.assertEqual(parse_pet_answer("  DEPICTION!  "), PET_VLM_DEPICTION)

    def test_prose_around_the_word_still_parses(self):
        self.assertEqual(
            parse_pet_answer("The image shows a photograph of a real dog."), PET_VLM_REAL)

    def test_an_explanation_that_also_says_real_is_read_as_the_rejection(self):
        """The reason `_PET_VLM_KEYWORDS` is ordered: `real` is the word a model reaches
        for while explaining one of the other two."""
        self.assertEqual(parse_pet_answer("depiction — not a real animal"),
                         PET_VLM_DEPICTION)
        self.assertEqual(parse_pet_answer("none, there is no real animal here"),
                         PET_VLM_NONE)

    def test_an_unreadable_answer_is_none_and_none_is_not_no(self):
        for answer in ("", "I cannot help with that", "42", "да", "maybe a cat"):
            with self.subTest(answer=answer):
                self.assertIsNone(parse_pet_answer(answer))
        # and the distinction survives into the label rule: unreadable keeps the old label
        self.assertEqual(pet_label(parse_pet_answer("???"), 0.9, 0.7), junk.PET_CLASS)


class TestVerifyOff(PetCascadeCase):
    """Brief test 1: with the toggle off the run is what it was — and nothing is asked."""

    def setUp(self):
        super().setUp()
        self.features(pets=True, pets_verify=False, pet_threshold=0.7,
                      pet_candidate_threshold=0.3)

    def test_the_label_is_the_threshold_and_the_model_is_never_called(self):
        def never(_path):
            raise AssertionError("no frame may reach the model with pets_verify off")

        stats, _clf = self.run_stage({"high.jpg": 0.95, "low.jpg": 0.4}, asker=never)
        self.assertEqual(self.label("high.jpg")["pet"], junk.PET_CLASS)
        self.assertIsNone(self.label("low.jpg")["pet"])
        for name in ("high.jpg", "low.jpg"):
            self.assertIsNone(self.label(name)["pet_vlm"])
        self.assertEqual(stats.pet_candidates, 0)
        self.assertEqual(stats.pets_found, 1)

    def test_no_model_is_built_either(self):
        def factory(_model):
            raise AssertionError("no model may be built with pets_verify off")

        self.run_stage({"a.jpg": 0.95}, pet_vlm_factory=factory)
        self.assertEqual(junk.quality_tier(
            self.conn.execute("SELECT source FROM frame_quality").fetchone()[0]),
            junk.QUALITY_SOURCE_CLIP)

    def test_the_check_needs_the_animals_toggle(self):
        """`pets_verify` verifies what the CLIP group found; without the group there is
        nothing to verify, and building a model to ask about nothing is the one outcome
        worth refusing up front."""
        self.features(pets=False, pets_verify=True)

        def factory(_model):
            raise AssertionError("no model may be built with features.pets off")

        self.run_stage({"a.jpg": 0.95}, pet_vlm_factory=factory)
        self.assertIsNone(self.label("a.jpg")["pet_score"])


class TestCandidateGate(PetCascadeCase):
    """Brief test 2: the model sees the frames above the candidate threshold, and no more."""

    def test_only_the_candidates_are_asked_and_each_exactly_once(self):
        asker = Asker({}, default="real")
        scores = {"top.jpg": 0.95, "mid.jpg": 0.5, "edge.jpg": 0.3,
                  "under.jpg": 0.29, "nothing.jpg": 0.0}
        stats, _clf = self.run_stage(scores, asker=asker)
        self.assertEqual(sorted(asker.asked), ["edge.jpg", "mid.jpg", "top.jpg"])
        self.assertEqual(len(asker.asked), stats.pet_candidates)
        self.assertEqual(stats.pet_candidates, 3)
        self.assertEqual(stats.pet_verified, 3)

    def test_the_candidate_threshold_comes_from_the_config(self):
        self.features(pets=True, pets_verify=True, pet_threshold=0.7,
                      pet_candidate_threshold=0.6)
        asker = Asker({}, default="real")
        self.run_stage({"a.jpg": 0.65, "b.jpg": 0.55}, asker=asker)
        self.assertEqual(asker.asked, ["a.jpg"])

    def test_the_stage_asks_the_model_nothing_it_has_no_candidate_for(self):
        asker = Asker({}, default="real")
        self.run_stage({"low.jpg": 0.1}, asker=asker)
        self.assertEqual(asker.asked, [])


class TestTheAnswerDecides(PetCascadeCase):
    """Brief tests 3, 4 and 10: the label moves in both directions, the score stays."""

    def test_depiction_takes_the_label_off_a_confident_frame(self):
        asker = Asker({"toy.jpg": "depiction"})
        stats, _clf = self.run_stage({"toy.jpg": 0.95}, asker=asker)
        row = self.label("toy.jpg")
        self.assertIsNone(row["pet"])
        self.assertEqual(row["pet_vlm"], PET_VLM_DEPICTION)
        self.assertEqual(stats.pets_found, 0)

    def test_none_takes_it_off_too(self):
        asker = Asker({"coat.jpg": "none"})
        self.run_stage({"coat.jpg": 0.9}, asker=asker)
        row = self.label("coat.jpg")
        self.assertIsNone(row["pet"])
        self.assertEqual(row["pet_vlm"], PET_VLM_NONE)

    def test_real_puts_a_label_on_a_frame_below_the_threshold(self):
        """The recall half of the cascade, and the reason the candidate threshold is low:
        the frame is under `pet_threshold`, so nothing before this feature would have
        marked it."""
        asker = Asker({"dim_cat.jpg": "real"})
        stats, _clf = self.run_stage({"dim_cat.jpg": 0.35}, asker=asker)
        row = self.label("dim_cat.jpg")
        self.assertEqual(row["pet"], junk.PET_CLASS)
        self.assertEqual(row["pet_vlm"], PET_VLM_REAL)
        self.assertEqual(stats.pets_found, 1)

    def test_real_leaves_an_already_marked_frame_marked(self):
        asker = Asker({"dog.jpg": "real"})
        stats, _clf = self.run_stage({"dog.jpg": 0.9}, asker=asker)
        self.assertEqual(self.label("dog.jpg")["pet"], junk.PET_CLASS)
        self.assertEqual(stats.pets_found, 1)  # counted once, not twice

    def test_the_score_survives_a_rejection(self):
        """Brief test 10: a threshold has to be re-choosable without another pass, so the
        score is written for every frame — the ones the model turned down included."""
        asker = Asker({"toy.jpg": "depiction", "cat.jpg": "real"})
        self.run_stage({"toy.jpg": 0.95, "cat.jpg": 0.4, "quiet.jpg": 0.05}, asker=asker)
        self.assertAlmostEqual(self.label("toy.jpg")["pet_score"], 0.95, places=5)
        self.assertAlmostEqual(self.label("cat.jpg")["pet_score"], 0.4, places=5)
        self.assertAlmostEqual(self.label("quiet.jpg")["pet_score"], 0.05, places=5)

    def test_the_answer_is_readable_back_out(self):
        asker = Asker({"toy.jpg": "depiction"})
        self.run_stage({"toy.jpg": 0.95}, asker=asker)
        rows = read_frame_quality(self.conn)
        (row,) = rows.values()
        self.assertEqual(row.pet_vlm, PET_VLM_DEPICTION)
        self.assertIsNone(row.pet)


class TestFallbacks(PetCascadeCase):
    """Brief tests 5 and 6: everything the expensive tier can do wrong costs nothing."""

    def test_an_unparsable_answer_falls_back_to_the_rule_not_to_no(self):
        asker = Asker({"high.jpg": "I'm not sure", "low.jpg": "hmm"})
        stats, _clf = self.run_stage({"high.jpg": 0.95, "low.jpg": 0.4}, asker=asker)
        self.assertEqual(self.label("high.jpg")["pet"], junk.PET_CLASS)  # kept, not dropped
        self.assertIsNone(self.label("low.jpg")["pet"])                  # still below 0.7
        for name in ("high.jpg", "low.jpg"):
            self.assertIsNone(self.label(name)["pet_vlm"])  # "not asked" is the truth here
        self.assertEqual(stats.pet_candidates, 2)
        self.assertEqual(stats.pet_verified, 0)

    def test_a_failure_on_one_frame_costs_only_that_frame(self):
        asker = Asker({"fine.jpg": "depiction"}, boom=("boom.jpg",))
        stats, _clf = self.run_stage({"boom.jpg": 0.95, "fine.jpg": 0.95}, asker=asker)
        self.assertEqual(self.label("boom.jpg")["pet"], junk.PET_CLASS)  # fast rule kept
        self.assertIsNone(self.label("boom.jpg")["pet_vlm"])
        self.assertIsNone(self.label("fine.jpg")["pet"])                 # neighbour served
        self.assertEqual(self.label("fine.jpg")["pet_vlm"], PET_VLM_DEPICTION)
        self.assertEqual(stats.pet_verified, 1)
        self.assertEqual(stats.pets_found, 1)

    def test_a_factory_that_raises_leaves_the_cheap_tier_running(self):
        def broken(_model):
            raise RuntimeError("transformers not installed")

        self.run_stage({"cat.jpg": 0.95}, pet_vlm_factory=broken)
        row = self.label("cat.jpg")
        self.assertEqual(row["pet"], junk.PET_CLASS)
        self.assertIsNone(row["pet_vlm"])
        # the row is marked by the tier that actually ran, so a later run with a working
        # model picks it up instead of considering it done
        self.assertEqual(junk.quality_tier(
            self.conn.execute("SELECT source FROM frame_quality").fetchone()[0]),
            junk.QUALITY_SOURCE_CLIP)

    def test_a_heuristics_only_run_asks_nothing(self):
        def factory(_model):
            raise AssertionError("no model may be built on a heuristics-only run")

        self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)
        classify(self.cfg, self.conn, use_clip=False, pet_vlm_factory=factory,
                 sharpness_detector=lambda _p: 1.0)
        self.assertIsNone(self.conn.execute("SELECT 1 FROM frame_quality").fetchone())


class TestPopulation(PetCascadeCase):
    """Brief test 9: only personal photographs are shown to the check."""

    def test_a_screenshot_is_never_asked_about(self):
        asker = Asker({}, default="real")
        self.add_file("Screenshot_1.png", camera_make=None, camera_model=None)
        self.add_file("cat.jpg")
        clf = PetClassifier({"Screenshot_1.png": 0.95, "cat.jpg": 0.95})
        classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                 sharpness_detector=lambda _p: 500.0, pet_vlm=asker)
        self.assertEqual(asker.asked, ["cat.jpg"])
        self.assertIsNone(self.label("Screenshot_1.png"))

    def test_an_excluded_class_is_never_asked_about(self):
        """`vlm.exclude_classes` needs no branch of its own here: the population is
        personal photographs, and every excludable class (document, product, screenshot,
        meme — `photo` is not one) has already left it. The case states that as a
        property rather than trusting the argument."""
        self.vlm(exclude_classes=("document",))
        asker = Asker({}, default="real")
        stats, _clf = self.run_stage({"form.jpg": 0.95, "cat.jpg": 0.95}, asker=asker,
                                     documents=("form.jpg",))
        self.assertEqual(
            self.media_class(self.file_id("form.jpg"))["verdict"], "document")
        self.assertEqual(asker.asked, ["cat.jpg"])
        self.assertEqual(stats.pet_candidates, 1)

    def test_a_frame_the_deep_tier_reclassified_is_not_asked_either(self):
        """The candidate list is built during the fast pass and asked after the deep tier,
        which is the one thing that can move a verdict in between. A frame that has since
        become a `document` must not be shown the question — its row is purged anyway, but
        by then the model would already have seen a passport."""
        self.cfg.naming = _naming_from({"vlm_enabled": True})
        asker = Asker({}, default="real")
        self.add_file("form.jpg")
        self.add_file("cat.jpg")
        clf = PetClassifier({"form.jpg": 0.95, "cat.jpg": 0.95}, products=("form.jpg",))
        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         sharpness_detector=lambda _p: 500.0, pet_vlm=asker,
                         vlm_classifier=lambda path:
                             "document" if path.endswith("form.jpg") else "personal_photo")
        self.assertEqual(
            self.media_class(self.file_id("form.jpg"))["verdict"], "document")
        self.assertEqual(asker.asked, ["cat.jpg"])
        self.assertEqual(stats.pet_candidates, 1)

    def file_id(self, name):
        return self.conn.execute(
            "SELECT id FROM files WHERE path = ?", (f"/photos/{name}",)).fetchone()[0]


class TestIncrementality(PetCascadeCase):
    """Brief tests 7 and 8: remembered between runs, invalidated when the question moves."""

    def test_the_second_run_asks_nothing_again(self):
        asker = Asker({"toy.jpg": "depiction"})
        self.run_stage({"toy.jpg": 0.95}, asker=asker)
        self.assertEqual(len(asker.asked), 1)

        clf = PetClassifier({"toy.jpg": 0.95})
        stats = classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                         sharpness_detector=lambda _p: 500.0, pet_vlm=asker)
        self.assertEqual(len(asker.asked), 1)      # ~500 frames x 0.78 s not paid again
        self.assertEqual(stats.pet_candidates, 0)
        self.assertEqual(self.label("toy.jpg")["pet_vlm"], PET_VLM_DEPICTION)
        self.assertIsNone(self.label("toy.jpg")["pet"])

    def test_editing_the_question_recomputes_the_rows(self):
        asker = Asker({"toy.jpg": "depiction"})
        self.run_stage({"toy.jpg": 0.95}, asker=asker)
        before = self.conn.execute("SELECT source FROM frame_quality").fetchone()[0]

        clf = PetClassifier({"toy.jpg": 0.95})
        with unittest.mock.patch.object(
                junk, "_PET_VLM_PROMPT", "Is this animal alive? real / depiction / none"):
            classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                     sharpness_detector=lambda _p: 500.0, pet_vlm=asker)
        after = self.conn.execute("SELECT source FROM frame_quality").fetchone()[0]
        self.assertEqual(len(asker.asked), 2)   # asked again under the new wording
        self.assertNotEqual(before, after)

    def test_the_fingerprint_moves_only_when_the_check_runs(self):
        """A collection measured without the check must not be invalidated by a prompt
        nobody asked — so the question joins the fingerprint only when it is asked."""
        plain = junk.quality_prompt_fingerprint(True, with_vlm=False)
        with unittest.mock.patch.object(junk, "_PET_VLM_PROMPT", "something else"):
            self.assertEqual(junk.quality_prompt_fingerprint(True, with_vlm=False), plain)
            moved = junk.quality_prompt_fingerprint(True, with_vlm=False,
                                                    verify_pets=True)
        self.assertNotEqual(
            moved, junk.quality_prompt_fingerprint(True, with_vlm=False, verify_pets=True))

    def test_switching_the_check_on_marks_the_rows_as_the_model_tier(self):
        source = junk._quality_source(True, True, None, lambda _p: "real")
        self.assertEqual(junk.quality_tier(source), junk.QUALITY_SOURCE_VLM)
        # and it is a different marker from the one the band alone writes, so switching
        # between the two questions reprocesses instead of looking done
        self.assertNotEqual(source, junk._quality_source(True, True, lambda _p: ""))


class TestSettingsAndAsker(unittest.TestCase):
    """The settings the pipeline reads, and the real asker over a fake runtime."""

    def test_the_config_reaches_the_stage_settings(self):
        from sorta.config import Config

        cfg = Config(features=FeaturesConfig(pets=True, pets_verify=True,
                                             pet_candidate_threshold=0.25),
                     vlm=VlmConfig())
        q = junk.quality_settings(cfg)
        self.assertTrue(q.pets_verify)
        self.assertAlmostEqual(q.pet_candidate_threshold, 0.25)

    def test_the_defaults_are_the_measured_ones(self):
        d = FeaturesConfig()
        self.assertFalse(d.pets_verify)
        # 0.30 was the brief's guess; 0.50 is what the run measured. Replayed against the
        # F122 labels, 0.30 turned out to be the WORST gate of the sweep — 90% precision
        # against a 92% CLIP-only baseline — because the frames it adds down there are
        # only 80% correct and dilute what is already better. The table is in config.py
        # next to the value.
        self.assertAlmostEqual(d.pet_candidate_threshold, 0.5)

    def test_the_asker_decodes_the_frame_and_asks_one_question(self):
        from PIL import Image

        seen: list[tuple[int, str, int]] = []

        def describe(frames, prompt, max_new_tokens):
            seen.append((len(frames), prompt, max_new_tokens))
            return "real"

        ask = junk.vlm_pet_asker(describe, max_edge=128)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jpg"
            Image.new("RGB", (256, 192), (30, 60, 90)).save(path, "JPEG")
            answer = ask(str(path))
        self.assertEqual(parse_pet_answer(answer), PET_VLM_REAL)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], 1)
        self.assertIn("depiction", seen[0][1])
        # the species is deliberately not among the answers (F122 retired those labels)
        for species in ("cat", "dog"):
            self.assertNotIn(species, seen[0][1].split())

    def test_a_missing_file_is_an_empty_answer_not_a_crash(self):
        def describe(_frames, _prompt, _tokens):
            raise AssertionError("a vanished frame must never reach the model")

        self.assertEqual(junk.vlm_pet_asker(describe, 128)("/nowhere/x.jpg"), "")


class TestMigration(unittest.TestCase):
    """The column arrives on an existing index without disturbing what is in it."""

    def test_a_v16_db_gains_the_column_and_keeps_its_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "v16.db"
            conn = connect(db)
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
            conn.execute("ALTER TABLE frame_quality DROP COLUMN pet_vlm")
            # A real v16 DB predates F140's column as well — see the note in
            # test_manual_pet: a simulated old DB has to be old in every respect.
            conn.execute("ALTER TABLE frame_quality DROP COLUMN junk_score")
            conn.execute(
                "INSERT INTO frame_quality (file_id, sharpness, pet, pet_score, source,"
                " updated_at) VALUES (1, 12.5, 'animal', 0.9, 'clip#abcd1234', 'x')")
            conn.execute("PRAGMA user_version = 16")
            conn.commit()
            conn.close()

            conn = connect(db)
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(frame_quality)")}
            row = conn.execute("SELECT * FROM frame_quality").fetchone()
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            conn.close()
        self.assertIn("pet_vlm", cols)
        self.assertEqual(version, 21)
        self.assertEqual(row["pet"], "animal")       # the label survived the migration
        self.assertIsNone(row["pet_vlm"])            # and means "never asked", not "no"
        self.assertAlmostEqual(row["pet_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
