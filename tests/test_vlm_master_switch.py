"""F145: `vlm.enabled` is the precondition for every question the model is asked.

The defect this pins came off a live run on 2026-08-02: a base pass started WITHOUT
deep analysis produced a landmarks stage that called the model. It was not one setting
gone wrong. Four of the five VLM questions of the pipeline gated on their own key alone
and the fifth — landmarks — had no condition at all, so any of them could raise 20 GB of
weights because a key was true in config.yaml from some earlier experiment.

The rule now, and what every case below is a form of:

* the subordinate keys (`vlm.products`, `features.pets_verify`,
  `features.landmarks_verify`, `features.junk_rescue`) decide WHAT is asked;
  `vlm.enabled` decides whether there is anybody to ask;
* with the master off NO FACTORY IS CALLED — asserted by counting calls and not by
  looking at the result, because "the model answered nothing" and "the model was never
  built" differ by five seconds and several gigabytes;
* and the result is not merely close to the one with the subordinate key off, it is the
  same result: each stage falls back onto the path it already had.

The master switch on with every subordinate key off is here too: permission is not an
instruction, and `vlm.enabled` alone must raise nothing either.

F186 retired two of the questions this file was written over — the frame-quality one
(`vlm.quality`) and the comparative keeper one (`dedup.keeper_vlm`) — and their cases went
with them. The invariant does not weaken by losing subjects: every question that is still
asked is still held here, the deep product tier included.
"""
from __future__ import annotations

import dataclasses
import unittest

from sorta import junk, landmarks
from sorta.config import FeaturesConfig, VlmConfig, _naming_from, vlm_allowed
from sorta.junk import classify

from tests.test_frame_quality import FrameQualityCase, QualityClassifier, flat_sharpness
from tests.test_junk import NO_OCR
from tests.test_clip_embeddings import EmbeddingClassifier
from tests.test_junk_rescue import FakeTextEncoder, vector_for
from tests.test_landmark_corroboration import PRAGUE, CorroborationCase, PathClassifier
from tests.test_landmarks_verify import SAYS_PRAGUE
from tests.test_pets_cascade import PetClassifier


class Counter:
    """A factory that counts what it built — and builds something harmless.

    The count is the whole point: every case here asks "were the weights loaded", and
    the only honest answer to that is how many times the thing that loads them was
    called. What it returns afterwards never matters, because in a passing case it is
    never returned.
    """

    def __init__(self, answer: str = ""):
        self.calls: list[str] = []
        self.answer = answer

    def __call__(self, model_name: str):
        self.calls.append(model_name)
        return lambda *_args, **_kwargs: self.answer


class TestTheHelperItself(unittest.TestCase):
    """`vlm_allowed` reads the effective per-run toggle, not the config file's."""

    def test_the_run_toggle_decides(self):
        cfg = dataclasses.replace(_cfg(), naming=_naming_from({"vlm_enabled": True}))
        self.assertTrue(vlm_allowed(cfg))
        self.assertFalse(vlm_allowed(dataclasses.replace(
            cfg, naming=_naming_from({"vlm_enabled": False}))))

    def test_a_config_without_a_naming_section_is_not_a_crash(self):
        """The junk and landmarks stages read it off whatever they are handed — a
        measurement script's object included."""
        self.assertFalse(vlm_allowed(object()))  # type: ignore[arg-type]


def _cfg():
    from sorta.config import Config
    return Config()


class MasterSwitchCase(FrameQualityCase):
    """The junk stage's four questions, each switched on with the master off."""

    def add_photos(self, *names):
        return [self.add_file(name) for name in names]

    def run_junk(self, **kwargs):
        kwargs.setdefault("classifier", QualityClassifier())
        kwargs.setdefault("text_detector", NO_OCR)
        kwargs.setdefault("sharpness_detector", flat_sharpness(100.0))
        return classify(self.cfg, self.conn, **kwargs)


class TestPetVerifyNeedsTheMaster(MasterSwitchCase):
    """`features.pets_verify` — the animal check."""

    def setUp(self):
        super().setUp()
        self.features(pets=True, pets_verify=True, pet_threshold=0.7,
                      pet_candidate_threshold=0.3)

    def test_the_factory_is_never_called(self):
        self.add_photos("cat.jpg")
        factory = Counter("real")
        self.run_junk(classifier=PetClassifier({"cat.jpg": 0.95}),
                      pet_vlm_factory=factory)
        self.assertEqual(factory.calls, [])

    def test_the_label_is_the_one_clip_alone_gives(self):
        """The check exists to OVERRULE the threshold, so "the model was not asked" and
        "the model said nothing" are two different labels on the same frame — which is
        what makes this comparison worth making."""
        self.add_photos("cat.jpg", "kitten.jpg")
        self.run_junk(classifier=PetClassifier({"cat.jpg": 0.95, "kitten.jpg": 0.4}),
                      pet_vlm_factory=Counter("depiction"))
        with_master_off = self._labels()

        self.setUp()
        self.features(pets=True, pets_verify=False, pet_threshold=0.7,
                      pet_candidate_threshold=0.3)
        self.add_photos("cat.jpg", "kitten.jpg")
        self.run_junk(classifier=PetClassifier({"cat.jpg": 0.95, "kitten.jpg": 0.4}),
                      pet_vlm_factory=Counter("depiction"))
        self.assertEqual(with_master_off, self._labels())
        # and it is the CLIP rule that produced it: above the threshold labelled, below
        # it not, with nothing in `pet_vlm`
        self.assertEqual(with_master_off, ((junk.PET_CLASS, None), (None, None)))

    def _labels(self):
        return tuple((r["pet"], r["pet_vlm"]) for r in self.conn.execute(
            "SELECT fq.pet, fq.pet_vlm FROM frame_quality fq"
            " JOIN files f ON f.id = fq.file_id ORDER BY f.path"))


class TestTheProductTierNeedsTheMaster(MasterSwitchCase):
    """`vlm.products` — the deep junk tier, and the fourth question left standing.

    It joined the F145 list at F161, after the two questions retired here had been
    written up: the tier used to BE what the master did by itself, so there was nothing to
    hold it to. It has a key of its own now, and the rule is the one every line above
    follows — the key says what to ask, the master says whether anybody may be asked.
    """

    def setUp(self):
        super().setUp()
        self.vlm(products=True)  # and the master left off

    def test_the_factory_is_never_called(self):
        self.add_photos("shoe.jpg")
        factory = Counter("product")
        self.run_junk(vlm_classifier_factory=factory)
        self.assertEqual(factory.calls, [])

    def test_the_verdict_is_the_one_the_fast_tier_wrote(self):
        """Not merely "no products": the row a run without the master produces."""
        fid = self.add_photos("shoe.jpg")[0]
        self.run_junk(vlm_classifier_factory=Counter("product"))
        row = self.conn.execute(
            "SELECT verdict, tier FROM media_class WHERE file_id = ?", (fid,)).fetchone()
        self.assertEqual((row["verdict"], row["tier"]), ("photo", "clip"))


class TestJunkRescueNeedsTheMaster(MasterSwitchCase):
    """`features.junk_rescue` — the score is still written, nobody is asked.

    This one already required the deep tier before F145, and the point of the case is
    that the promise did not change when the condition moved into the shared helper:
    the score lands in `frame_quality.junk_score` either way, because that is the state
    the feature is meant to be tried in.
    """

    def setUp(self):
        super().setUp()
        self.features(junk_rescue=True, junk_rescue_threshold=0.02)

    def test_the_factory_is_never_called_and_the_score_is_still_written(self):
        self.add_photos("meme.jpg")
        factory = Counter("junk")
        clf = EmbeddingClassifier(vectors={"meme.jpg": vector_for(0.4)})
        stats = self.run_junk(classifier=clf, junk_text_encoder=FakeTextEncoder(),
                              junk_rescue_vlm_factory=factory)
        self.assertEqual(factory.calls, [])
        self.assertEqual(stats.junk_rescued, 0)
        score = self.conn.execute("SELECT junk_score FROM frame_quality").fetchone()[0]
        self.assertIsNotNone(score)


class TestPermissionIsNotAnInstruction(MasterSwitchCase):
    """Brief test 4: the master on with every subordinate key off raises nothing.

    The deep junk tier is the one thing `vlm.enabled` switches on by itself — that is
    its own stage, and it is given a classifier here so the case can say something about
    the other four rather than about it.
    """

    def test_no_subordinate_factory_is_called(self):
        self.cfg.naming = dataclasses.replace(self.cfg.naming, vlm_enabled=True)
        self.vlm(products=False)
        self.features(pets=True, pets_verify=False, junk_rescue=False)
        self.add_photos("IMG_0001.jpg")
        factories = {name: Counter() for name in
                     ("vlm_classifier_factory", "pet_vlm_factory",
                      "junk_rescue_vlm_factory")}
        self.run_junk(classifier=PetClassifier({"IMG_0001.jpg": 0.95}), **factories)
        for name, factory in factories.items():
            with self.subTest(factory=name):
                self.assertEqual(factory.calls, [])


class TestLandmarksNeedTheMaster(CorroborationCase):
    """Brief test 3: the stage the defect was found on.

    It had no condition whatsoever — `features.landmarks_verify` was read straight out
    of the config and the model was asked. The comparison here is the one the criterion
    names: with the master off the stage places exactly the files it placed before F131
    existed, down to the gate the proposals are collected at.
    """

    def setUp(self):
        super().setUp()
        self.cfg = dataclasses.replace(
            self.cfg,
            features=FeaturesConfig(landmarks_verify=True,
                                    landmark_candidate_threshold=0.1),
            naming=_naming_from({"landmarks_file": self.cfg.naming.landmarks_file,
                                 "landmark_threshold": 0.85}),
        )

    def _run(self, factory):
        return landmarks.detect_landmarks(
            self.cfg, self.conn, classifier=PathClassifier(self.scores),
            resolver=self.resolver, asker_factory=factory)

    def test_the_factory_is_never_called(self):
        self.add("/photos/DCIM", PRAGUE, prob=0.95)
        self.add("/photos/DCIM", PRAGUE, prob=0.60)
        factory = Counter("Prague")
        self._run(factory)
        self.assertEqual(factory.calls, [])

    def test_the_places_are_the_ones_the_stage_gave_before_f131(self):
        """The widened candidate gate goes with the check: a proposal at 0.60 is one the
        check would have looked at, and with no check running it must not slip past
        `naming.landmark_threshold` on its own."""
        band = self.add("/photos/DCIM", PRAGUE, prob=0.60)
        strong = self.add("/photos/DCIM", PRAGUE, prob=0.95)
        stats = self._run(Counter(SAYS_PRAGUE))
        self.assertEqual(stats.matched, 1)
        self.assertEqual(self.place_of(band)[3], "unknown")
        self.assertEqual(self.place_of(strong)[3], "visual")
        self.assertEqual(
            (stats.checked, stats.confirmed_by_model, stats.rejected_by_model),
            (0, 0, 0))

    def test_the_scan_marker_matches_a_run_that_never_had_the_check(self):
        """F136 fingerprints the gate, so "the check is off" and "the master is off" have
        to produce the same marker — otherwise the next run rescans the whole collection
        for a setting that changed nothing."""
        self.add("/photos/DCIM", PRAGUE, prob=0.95)
        self._run(Counter())
        with_master_off = self._markers()

        self.cfg = dataclasses.replace(
            self.cfg, features=FeaturesConfig(landmarks_verify=False))
        self._run(Counter())
        self.assertEqual(with_master_off, self._markers())

    def _markers(self):
        return sorted(r["verdict"] for r in self.conn.execute(
            "SELECT verdict FROM landmark_checks"))


class TestLandmarksWithTheMasterOn(TestLandmarksNeedTheMaster):
    """The mirror image: with `vlm.enabled` on the check runs, as F131 wrote it.

    Without this the cases above would pass just as well on a stage that had been
    switched off altogether.
    """

    def setUp(self):
        super().setUp()
        self.cfg = dataclasses.replace(
            self.cfg,
            naming=_naming_from({"landmarks_file": self.cfg.naming.landmarks_file,
                                 "landmark_threshold": 0.85, "vlm_enabled": True}))

    def test_the_factory_is_never_called(self):
        self.add("/photos/DCIM", PRAGUE, prob=0.95)
        factory = Counter(SAYS_PRAGUE)
        self._run(factory)
        self.assertEqual(len(factory.calls), 1)

    def test_the_places_are_the_ones_the_stage_gave_before_f131(self):
        band = self.add("/photos/DCIM", PRAGUE, prob=0.60)
        self._run(Counter(SAYS_PRAGUE))
        # the widened gate is back, so the frame at 0.60 is asked about and placed
        self.assertEqual(self.place_of(band)[3], "visual")

    def test_the_scan_marker_matches_a_run_that_never_had_the_check(self):
        self.add("/photos/DCIM", PRAGUE, prob=0.95)
        self._run(Counter(SAYS_PRAGUE))
        self.assertNotEqual(self._markers(), [])


class TestNothingIsRewrittenInTheConfig(unittest.TestCase):
    """A boundary of the brief: a switched-off model is a state of the run, not a
    reason to erase somebody else's settings."""

    def test_the_subordinate_keys_keep_their_values(self):
        cfg = dataclasses.replace(
            _cfg(),
            vlm=VlmConfig(enabled=False, products=True),
            features=FeaturesConfig(pets=True, pets_verify=True, junk_rescue=True,
                                    landmarks_verify=True),
        )
        self.assertFalse(vlm_allowed(cfg))
        self.assertTrue(cfg.vlm.products)
        self.assertTrue(cfg.features.pets_verify)
        self.assertTrue(cfg.features.landmarks_verify)
        self.assertTrue(cfg.features.junk_rescue)


if __name__ == "__main__":
    unittest.main()
