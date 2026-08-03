"""F154: the object detector as a second tier over a query — the promises, one by one.

The feature is a CASCADE and a BOUNDARY, and every case below is about one half of that:

* with `features.detector` off — or with its master switch `detect.enabled` off — nothing
  is loaded, nothing is stored and every animal label is the one F122/F130 wrote;
* with both on, the detector sees the CANDIDATES OF A QUERY and nothing else: their number
  is `features.detector_candidates` and their identity is the ranking over the stored
  vectors. There is no setting in which the detector runs over the collection;
* the answer overrides the CLIP label in BOTH directions — an animal found where the score
  was too low is labelled, a frame CLIP called an animal with nothing on it loses the label;
* every refusal falls back to the previous rule and never to "no animal": no vectors, an
  encoder that will not build, a detector that will not build, an error on one frame;
* people and food take no part. That is the measurement's boundary (42% precision against
  the faces' ~100%, and 20% for a class COCO does not have), so it is asserted explicitly
  rather than left to the class list to imply;
* a repeated run asks nothing again — the `detections` row is the marker, including on the
  frames the detector found nothing on.

No model is loaded anywhere: the classifier, the text encoder and the detector are all
injected, exactly as the rest of the junk suite does it.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sorta import detect, junk
from sorta.config import (
    Config,
    DetectConfig,
    FeaturesConfig,
    _naming_from,
    detector_allowed,
)
from sorta.db import SCHEMA_VERSION, connect
from sorta.detect import (
    ANIMAL_CLASSES,
    ANIMAL_QUERY_PROMPTS,
    Detection,
    animal_boxes,
    best_animal,
    cascade_label,
    detector_settings,
    pack_boxes,
    query_scores,
    rank_candidates,
    unpack_boxes,
)
from tests.schema_history import roll_back_before
from tests.test_clip_embeddings import EmbeddingClassifier
from tests.test_frame_quality import _CAT_IDX, FrameQualityCase
from tests.test_junk import NO_OCR

# A vector whose animal-query score is exactly `value`: the prompts all point at [1, 0]
# (see FakeAnimalEncoder), so the first component IS the score and the second only makes
# the vector a unit one. A case can then name the ranking it is about.
def vector_for(value: float) -> np.ndarray:
    return np.array([value, math.sqrt(max(0.0, 1.0 - value * value))], dtype=np.float32)


def box(label: str, score: float) -> Detection:
    """One detection, with a box nobody in these cases reads — the class and score do."""
    return Detection(label, score, (10.0, 20.0, 110.0, 220.0))


class FakeAnimalEncoder:
    """The text tower without a model: every animal prompt points the same way.

    Built off `ANIMAL_QUERY_PROMPTS` rather than a hard-coded count, so editing the prompt
    list does not quietly turn these cases into a test of the mock.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, prompts):
        self.calls.append(tuple(prompts))
        return np.asarray([[1.0, 0.0]] * len(prompts), dtype=np.float32)


class FakeDetector:
    """A detector that answers per frame by basename and remembers what it was shown."""

    def __init__(self, boxes=None, boom=()) -> None:
        self.boxes = dict(boxes or {})
        self.boom = set(boom)
        self.seen: list[str] = []

    def __call__(self, path: str):
        name = Path(path).name
        self.seen.append(name)
        if name in self.boom:
            raise RuntimeError("CUDA error: device-side assert triggered")
        return self.boxes.get(name, [])


class CountingFactory:
    """A detector factory that counts how many times a run actually built a model."""

    def __init__(self, detector: FakeDetector | None = None) -> None:
        self.detector = detector or FakeDetector()
        self.builds: list[str] = []

    def __call__(self, model: str):
        self.builds.append(model)
        return self.detector


class DetectorCase(FrameQualityCase):
    """The fixture of the file: both switches on, pets on, a two-frame candidate depth."""

    def setUp(self):
        super().setUp()
        self.features(pets=True, detector=True, detector_candidates=2,
                      detector_threshold=0.5)
        self.cfg.detect = DetectConfig(enabled=True)
        self.encoder = FakeAnimalEncoder()

    def classify_over(self, scores, vectors, detector=None, **kwargs):
        """One classify() over the files already in the index, with every model injected.

        The classifier answers CLIP scores and vectors, the text encoder answers the animal
        prompts, the detector answers boxes — the suite loads none of the three.
        """
        clf = EmbeddingClassifier(scores=scores, vectors=vectors)
        kwargs.setdefault("detector_text_encoder", self.encoder)
        if detector is not None:
            kwargs.setdefault("detector", detector)
        stats = junk.classify(self.cfg, self.conn, classifier=clf, text_detector=NO_OCR,
                              sharpness_detector=lambda _p: 500.0, **kwargs)
        return stats, clf

    def run_stage(self, frames, detector=None, **kwargs):
        """`frames` = {name: (query score, does CLIP call it an animal)} -> one classify()."""
        scores = {}
        vectors = {}
        for name, (query, clip_animal) in frames.items():
            self.add_file(name)
            vectors[name] = vector_for(query)
            if clip_animal:
                scores[name] = (_CAT_IDX, 0.9)
        return self.classify_over(scores, vectors, detector, **kwargs)

    def file_id(self, name):
        return self.conn.execute(
            "SELECT id FROM files WHERE path = ?", (f"/photos/{name}",)).fetchone()[0]

    def pet_of(self, name):
        row = self.quality(self.file_id(name))
        return None if row is None else row["pet"]

    def detection_of(self, name):
        return self.conn.execute(
            "SELECT * FROM detections WHERE file_id = ?", (self.file_id(name),)).fetchone()


class TestTheBoundaryTheMeasurementDrew(unittest.TestCase):
    """Brief test 6: people and food take no part, and that is checked, not implied."""

    def test_the_classes_are_the_coco_animals_and_only_those(self):
        self.assertEqual(sorted(ANIMAL_CLASSES), list(range(16, 26)))
        self.assertEqual(ANIMAL_CLASSES[17], "cat")
        self.assertEqual(ANIMAL_CLASSES[18], "dog")

    def test_no_person_and_no_food_class_exists_at_all(self):
        """42% precision against ~100% from the face boxes, and 20% for a class COCO does
        not have — the two rows of the table this feature said no to."""
        for absent in ("person", "banana", "sandwich", "pizza", "cake", "hot dog"):
            with self.subTest(label=absent):
                self.assertNotIn(absent, ANIMAL_CLASSES.values())

    def test_a_person_box_is_not_an_animal_however_confident(self):
        found = [box("person", 0.99), box("pizza", 0.98), box("cat", 0.6)]
        self.assertEqual([d.label for d in animal_boxes(found, 0.5)], ["cat"])

    def test_a_frame_of_people_and_food_alone_holds_no_animal(self):
        self.assertIsNone(best_animal([box("person", 0.99), box("pizza", 0.9)], 0.5))


class TestTheCascadeRule(unittest.TestCase):
    """`cascade_label` — who outranks whom, and what a refusal falls back to."""

    def test_a_detected_animal_overrides_a_clip_label_that_was_missing(self):
        self.assertEqual(
            cascade_label(box("cat", 0.9), examined=True, verified=False,
                          previous=None, animal=junk.PET_CLASS), junk.PET_CLASS)

    def test_nothing_detected_overrides_a_clip_label_that_was_there(self):
        self.assertIsNone(
            cascade_label(None, examined=True, verified=False,
                          previous=junk.PET_CLASS, animal=junk.PET_CLASS))

    def test_a_frame_the_detector_never_saw_keeps_the_previous_rule(self):
        for previous in (junk.PET_CLASS, None):
            with self.subTest(previous=previous):
                self.assertEqual(
                    cascade_label(None, examined=False, verified=False,
                                  previous=previous, animal=junk.PET_CLASS), previous)

    def test_an_answer_from_the_vlm_check_outranks_the_detector(self):
        """A box detector calls a drawn cat a cat — which is the error F130 exists to
        remove, so a frame that check has answered about keeps its answer."""
        self.assertIsNone(
            cascade_label(box("cat", 0.95), examined=True, verified=True,
                          previous=None, animal=junk.PET_CLASS))


class TestTheQueryThatSelects(unittest.TestCase):
    """The candidates are a RANKING over the stored vectors, not a threshold."""

    def features(self):
        return np.asarray([[1.0, 0.0]] * len(ANIMAL_QUERY_PROMPTS), dtype=np.float32)

    def test_the_score_is_the_best_prompt_of_the_frame(self):
        scored = query_scores({1: vector_for(0.8), 2: vector_for(0.2)}, self.features())
        self.assertAlmostEqual(scored[1], 0.8, places=5)
        self.assertAlmostEqual(scored[2], 0.2, places=5)

    def test_the_depth_is_what_the_feature_costs(self):
        vectors = {i: vector_for(i / 10.0) for i in range(1, 10)}
        self.assertEqual(rank_candidates(vectors, self.features(), 3), [9, 8, 7])
        self.assertEqual(len(rank_candidates(vectors, self.features(), 2)), 2)
        self.assertEqual(rank_candidates(vectors, self.features(), 0), [])

    def test_ties_are_broken_by_file_id_so_a_repeat_run_selects_the_same_frames(self):
        vectors = {5: vector_for(0.4), 2: vector_for(0.4), 9: vector_for(0.4)}
        self.assertEqual(rank_candidates(vectors, self.features(), 2), [2, 5])

    def test_a_vector_of_another_width_is_not_ranked_across_two_spaces(self):
        vectors = {1: np.ones(8, dtype=np.float32), 2: vector_for(0.1)}
        self.assertEqual(rank_candidates(vectors, self.features(), 5), [2])


class TestTheStoredBoxes(unittest.TestCase):
    """The classes and the coordinates survive a round trip; garbage costs one row."""

    def test_boxes_come_back_as_they_went_in(self):
        found = [box("cat", 0.91), box("dog", 0.55)]
        back = unpack_boxes(pack_boxes(found))
        self.assertEqual([(d.label, round(d.score, 2)) for d in back],
                         [("cat", 0.91), ("dog", 0.55)])
        self.assertEqual(back[0].box, (10.0, 20.0, 110.0, 220.0))

    def test_an_unreadable_column_is_no_boxes_rather_than_a_crash(self):
        for text in (None, "", "{", "[[1,2]]", "[42]", '[["cat","x",1,2,3,4]]'):
            with self.subTest(stored=text):
                self.assertEqual(unpack_boxes(text), [])


class TestBothSwitches(unittest.TestCase):
    """F145 applied to a second kind of model: no subordinate key raises one by itself."""

    def config(self, *, master: bool, feature: bool) -> Config:
        cfg = Config(naming=_naming_from({}))
        cfg.detect = DetectConfig(enabled=master)
        cfg.features = FeaturesConfig(detector=feature)
        return cfg

    def test_both_are_needed_for_the_cascade_to_be_enabled(self):
        for master, feature, enabled in ((False, False, False), (True, False, False),
                                         (False, True, False), (True, True, True)):
            with self.subTest(master=master, feature=feature):
                s = detector_settings(self.config(master=master, feature=feature))
                self.assertEqual(s.enabled, enabled)

    def test_the_master_switch_is_not_the_vlm_one(self):
        """A detector is not a VLM: switching the deep tier on must not raise one, and
        clearing the deep tier must not switch this off."""
        cfg = self.config(master=True, feature=True)
        cfg.naming = _naming_from({"vlm_enabled": False})
        self.assertTrue(detector_allowed(cfg))
        self.assertTrue(detector_settings(cfg).enabled)

    def test_the_defaults_are_off_and_the_numbers_are_the_measured_ones(self):
        s = detector_settings(Config(naming=_naming_from({})))
        self.assertFalse(s.enabled)
        self.assertEqual(s.candidates, 2000)
        self.assertEqual(s.threshold, 0.5)


class TestMigration(unittest.TestCase):
    """The table appears, the version moves, and an old database gains it with its rows.

    Every connection is closed inside its temp directory: on Windows an open sqlite handle
    makes the rmtree fail (the reason test_frame_quality states at its own fixture).
    """

    def test_fresh_db_has_the_table_and_the_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "fresh.db")
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(detections)")}
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            conn.close()
        self.assertEqual(cols, {"file_id", "label", "score", "boxes", "model",
                                "updated_at"})
        self.assertEqual(version, SCHEMA_VERSION)

    def test_a_db_from_before_the_table_gains_it_without_touching_its_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "old.db"
            conn = connect(db)
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
            roll_back_before(conn, "detections")
            conn.commit()
            conn.close()

            conn = connect(db)
            tables = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            conn.close()
        self.assertIn("detections", tables)
        self.assertEqual(files, 1)


class TestTheTogglesAreOff(DetectorCase):
    """Brief test 1: nothing is loaded, nothing is stored, no verdict and no label moves."""

    def frames(self):
        return {"cat.jpg": (0.9, True), "beach.jpg": (0.1, False)}

    def test_with_the_feature_off_no_detector_is_ever_built(self):
        self.features(pets=True, detector=False)
        factory = CountingFactory()
        stats, _clf = self.run_stage(self.frames(), detector_factory=factory)
        self.assertEqual(factory.builds, [])
        self.assertEqual(factory.detector.seen, [])
        self.assertEqual(stats.detector_candidates, 0)

    def test_with_the_master_switch_off_no_detector_is_ever_built(self):
        """`features.detector: true` alone must not raise a model — the F145 rule."""
        self.cfg.detect = DetectConfig(enabled=False)
        factory = CountingFactory()
        _stats, _clf = self.run_stage(self.frames(), detector_factory=factory)
        self.assertEqual(factory.builds, [])

    def test_the_labels_are_the_ones_the_clip_threshold_gives(self):
        self.features(pets=True, detector=False)
        self.run_stage(self.frames())
        self.assertEqual(self.pet_of("cat.jpg"), junk.PET_CLASS)
        self.assertIsNone(self.pet_of("beach.jpg"))
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0], 0)

    def test_the_text_encoder_is_not_asked_either(self):
        """The query is the candidate list, and there are no candidates to build."""
        self.features(pets=True, detector=False)
        self.run_stage(self.frames())
        self.assertEqual(self.encoder.calls, [])


class TestOnlyTheCandidatesOfTheQuery(DetectorCase):
    """Brief test 2: the detector sees the query's top N and nothing else, ever."""

    def five_frames(self):
        return {"a.jpg": (0.9, False), "b.jpg": (0.8, False), "c.jpg": (0.7, False),
                "d.jpg": (0.2, False), "e.jpg": (0.1, False)}

    def test_the_number_of_frames_shown_equals_the_setting(self):
        det = FakeDetector()
        stats, _clf = self.run_stage(self.five_frames(), detector=det)
        self.assertEqual(len(det.seen), 2)
        self.assertEqual(stats.detector_candidates, 2)
        self.assertEqual(stats.detector_examined, 2)

    def test_the_frames_shown_are_the_ones_the_query_ranks_highest(self):
        det = FakeDetector()
        self.run_stage(self.five_frames(), detector=det)
        self.assertEqual(sorted(det.seen), ["a.jpg", "b.jpg"])

    def test_a_deeper_list_costs_more_frames_and_nothing_else_changes(self):
        self.features(pets=True, detector=True, detector_candidates=4,
                      detector_threshold=0.5)
        det = FakeDetector()
        self.run_stage(self.five_frames(), detector=det)
        self.assertEqual(sorted(det.seen), ["a.jpg", "b.jpg", "c.jpg", "d.jpg"])

    def test_there_is_no_setting_that_shows_it_the_whole_collection(self):
        """A depth above the collection is still the F120 population and no more — the
        five frames here, never a sixth pass over anything."""
        self.features(pets=True, detector=True, detector_candidates=10_000,
                      detector_threshold=0.5)
        det = FakeDetector()
        self.run_stage(self.five_frames(), detector=det)
        self.assertEqual(len(det.seen), 5)


class TestTheAnswerOverridesTheClipLabel(DetectorCase):
    """Brief test 3: in both directions, and only where the detector actually looked."""

    def test_an_animal_found_below_the_clip_threshold_is_labelled(self):
        det = FakeDetector({"corner.jpg": [box("cat", 0.8)]})
        stats, _clf = self.run_stage({"corner.jpg": (0.9, False)}, detector=det)
        self.assertEqual(self.pet_of("corner.jpg"), junk.PET_CLASS)
        self.assertEqual(stats.detector_found, 1)
        self.assertEqual(stats.pets_found, 1)

    def test_a_clip_animal_with_nothing_detected_loses_the_label(self):
        det = FakeDetector({"fur_coat.jpg": []})
        stats, _clf = self.run_stage({"fur_coat.jpg": (0.9, True)}, detector=det)
        self.assertIsNone(self.pet_of("fur_coat.jpg"))
        self.assertEqual(stats.detector_found, 0)
        self.assertEqual(stats.pets_found, 0)

    def test_a_box_below_the_confidence_threshold_is_not_an_animal(self):
        det = FakeDetector({"maybe.jpg": [box("dog", 0.4)]})
        self.run_stage({"maybe.jpg": (0.9, True)}, detector=det)
        self.assertIsNone(self.pet_of("maybe.jpg"))

    def test_a_frame_outside_the_candidate_list_keeps_the_clip_label(self):
        det = FakeDetector({"cat.jpg": [box("cat", 0.9)]})
        self.run_stage({"cat.jpg": (0.9, True), "x.jpg": (0.8, False),
                        "deep.jpg": (0.1, True)}, detector=det)
        self.assertNotIn("deep.jpg", det.seen)
        self.assertEqual(self.pet_of("deep.jpg"), junk.PET_CLASS)

    def test_the_classes_and_the_scores_are_stored_not_just_the_word_animal(self):
        """Brief requirement 6: the species and the box come from nowhere else, and the
        cat/dog split F122 closed for CLIP is a query over this table."""
        det = FakeDetector({"pets.jpg": [box("dog", 0.93), box("cat", 0.71)]})
        self.run_stage({"pets.jpg": (0.9, False)}, detector=det)
        row = self.detection_of("pets.jpg")
        self.assertEqual(row["label"], "dog")
        self.assertAlmostEqual(row["score"], 0.93, places=3)
        self.assertEqual([d.label for d in unpack_boxes(row["boxes"])], ["dog", "cat"])
        self.assertEqual(row["model"], DetectConfig.model)

    def test_people_and_food_do_not_produce_a_label_or_a_stored_class(self):
        """Brief test 6 again, through the whole pipeline this time."""
        det = FakeDetector({"party.jpg": [box("person", 0.99), box("pizza", 0.97)]})
        stats, _clf = self.run_stage({"party.jpg": (0.9, False)}, detector=det)
        self.assertIsNone(self.pet_of("party.jpg"))
        row = self.detection_of("party.jpg")
        self.assertIsNone(row["label"])
        self.assertEqual(unpack_boxes(row["boxes"]), [])
        self.assertEqual(stats.detector_found, 0)

    def test_without_the_clip_pet_group_the_detector_still_labels(self):
        """`features.pets` is not a dependency: the detector finds animals by itself, it
        does not verify what CLIP found."""
        self.features(pets=False, detector=True, detector_candidates=2,
                      detector_threshold=0.5)
        det = FakeDetector({"cat.jpg": [box("cat", 0.8)]})
        self.run_stage({"cat.jpg": (0.9, False)}, detector=det)
        self.assertEqual(self.pet_of("cat.jpg"), junk.PET_CLASS)


class TestARefusalIsNeverANo(DetectorCase):
    """Brief test 4: one frame's failure costs that frame and nothing else."""

    def test_an_error_on_one_frame_leaves_the_others_labelled(self):
        det = FakeDetector({"cat.jpg": [box("cat", 0.9)]}, boom=("boom.jpg",))
        stats, _clf = self.run_stage({"boom.jpg": (0.95, True), "cat.jpg": (0.9, False)},
                                     detector=det)
        self.assertEqual(self.pet_of("cat.jpg"), junk.PET_CLASS)
        self.assertEqual(stats.detector_examined, 1)

    def test_the_failed_frame_keeps_its_label_and_gets_no_row(self):
        """No row means it is examined again next run — never recorded as "no animal"."""
        det = FakeDetector(boom=("boom.jpg",))
        self.run_stage({"boom.jpg": (0.95, True), "cat.jpg": (0.9, False)}, detector=det)
        self.assertEqual(self.pet_of("boom.jpg"), junk.PET_CLASS)
        self.assertIsNone(self.detection_of("boom.jpg"))

    def test_a_detector_that_will_not_build_leaves_every_label_alone(self):
        def explode(_model):
            raise RuntimeError("torchvision weights are not downloaded")

        with self.assertLogs("sorta.junk", level=logging.WARNING):
            stats, _clf = self.run_stage({"cat.jpg": (0.9, True), "x.jpg": (0.8, False)},
                                         detector_factory=explode)
        self.assertEqual(self.pet_of("cat.jpg"), junk.PET_CLASS)
        self.assertEqual(stats.detector_examined, 0)

    def test_an_encoder_that_will_not_build_leaves_every_label_alone(self):
        def explode(_prompts):
            raise RuntimeError("open_clip is not installed")

        det = FakeDetector({"cat.jpg": [box("cat", 0.9)]})
        with self.assertLogs("sorta.junk", level=logging.WARNING):
            stats, _clf = self.run_stage({"cat.jpg": (0.9, True)}, detector=det,
                                         detector_text_encoder=explode)
        self.assertEqual(det.seen, [])
        self.assertEqual(self.pet_of("cat.jpg"), junk.PET_CLASS)
        self.assertEqual(stats.detector_candidates, 0)


class TestWithoutVectorsTheStageSaysWhy(DetectorCase):
    """Brief test 5: an empty `clip_embeddings` is a reason, not an empty candidate list."""

    def test_nothing_runs_and_the_log_names_the_table(self):
        self.features(pets=True, detector=True, detector_candidates=2,
                      detector_threshold=0.5, store_embeddings=False)
        det = FakeDetector({"cat.jpg": [box("cat", 0.9)]})
        with self.assertLogs("sorta.junk", level=logging.WARNING) as caught:
            stats, _clf = self.run_stage({"cat.jpg": (0.9, True)}, detector=det)
        said = "\n".join(caught.output)
        self.assertIn("clip_embeddings", said)
        self.assertIn("store_embeddings", said)
        self.assertEqual(det.seen, [])
        self.assertEqual((stats.detector_candidates, stats.detector_examined), (0, 0))

    def test_the_label_is_the_one_the_cheap_tier_wrote(self):
        self.features(pets=True, detector=True, detector_candidates=2,
                      detector_threshold=0.5, store_embeddings=False)
        with self.assertLogs("sorta.junk", level=logging.WARNING):
            self.run_stage({"cat.jpg": (0.9, True)}, detector=FakeDetector())
        self.assertEqual(self.pet_of("cat.jpg"), junk.PET_CLASS)


class TestARepeatedRunAsksNothing(DetectorCase):
    """Brief test 7: incrementality on the `detections` row, the marker being the model."""

    def frames(self):
        return {"cat.jpg": (0.9, False), "empty.jpg": (0.8, True)}

    def rerun(self, det):
        """A second classify() over the SAME files — nothing is added to the index."""
        stats, _clf = self.classify_over(
            {"empty.jpg": (_CAT_IDX, 0.9)},
            {"cat.jpg": vector_for(0.9), "empty.jpg": vector_for(0.8)}, det)
        return stats

    def test_the_second_run_shows_the_detector_no_frame_at_all(self):
        det = FakeDetector({"cat.jpg": [box("cat", 0.9)]})
        self.run_stage(self.frames(), detector=det)
        det.seen.clear()
        stats = self.rerun(det)
        self.assertEqual(det.seen, [])
        self.assertEqual(stats.detector_examined, 0)
        self.assertEqual(stats.detector_candidates, 2)

    def test_a_frame_it_found_nothing_on_is_not_asked_about_again(self):
        """The expensive half of incrementality: without a row for the empty answer, the
        frames the detector turned down would be re-examined on every run."""
        det = FakeDetector({"cat.jpg": [box("cat", 0.9)]})
        self.run_stage(self.frames(), detector=det)
        self.assertIsNotNone(self.detection_of("empty.jpg"))
        det.seen.clear()
        self.rerun(det)
        self.assertNotIn("empty.jpg", det.seen)

    def test_the_labels_of_the_second_run_are_the_labels_of_the_first(self):
        det = FakeDetector({"cat.jpg": [box("cat", 0.9)]})
        self.run_stage(self.frames(), detector=det)
        self.rerun(det)
        self.assertEqual(self.pet_of("cat.jpg"), junk.PET_CLASS)
        self.assertIsNone(self.pet_of("empty.jpg"))

    def test_another_detector_is_another_answer_and_is_asked_again(self):
        det = FakeDetector({"cat.jpg": [box("cat", 0.9)]})
        self.run_stage(self.frames(), detector=det)
        det.seen.clear()
        self.cfg.detect = DetectConfig(enabled=True, model="retinanet_resnet50_fpn")
        self.rerun(det)
        self.assertEqual(sorted(det.seen), ["cat.jpg", "empty.jpg"])
        self.assertEqual(self.detection_of("cat.jpg")["model"], "retinanet_resnet50_fpn")


class TestTheBoxesOfANonPhotographAreDropped(DetectorCase):
    """The F120 population, enforced on this table like on every other one of the stage."""

    def test_a_frame_reclassified_away_from_photo_loses_its_boxes(self):
        # No camera EXIF on this one: a photograph with a camera in its metadata vetoes
        # the CLIP junk verdict (F13), and this case is about what happens when the
        # verdict actually moves.
        self.add_file("cat.jpg", camera_make=None, camera_model=None)
        vectors = {"cat.jpg": vector_for(0.9)}
        det = FakeDetector({"cat.jpg": [box("cat", 0.9)]})
        self.classify_over({}, vectors, det)
        self.assertIsNotNone(self.detection_of("cat.jpg"))
        # The next run calls the very same frame a screenshot — the boxes describe
        # something this stage has decided is not a personal photograph.
        self.conn.execute("UPDATE media_class SET tier = 'stale'")
        self.conn.commit()
        self.classify_over({"cat.jpg": (1, 0.99)}, vectors, det)
        self.assertEqual(self.media_class(self.file_id("cat.jpg"))["verdict"],
                         "screenshot")
        self.assertIsNone(self.detection_of("cat.jpg"))


class TestTheStageThatIsOtherwiseUpToDate(DetectorCase):
    """The ordinary way this feature is switched on: over a collection already classified.

    An early return there would leave it silently doing nothing — the mistake F132/F141
    each had to avoid for their own pass.
    """

    def test_the_cascade_runs_when_no_other_half_has_work(self):
        self.features(pets=True, detector=False)
        self.run_stage({"cat.jpg": (0.9, True)})
        self.features(pets=True, detector=True, detector_candidates=2,
                      detector_threshold=0.5)
        det = FakeDetector({"cat.jpg": [box("cat", 0.9)]})
        stats, _clf = self.classify_over({"cat.jpg": (_CAT_IDX, 0.9)},
                                         {"cat.jpg": vector_for(0.9)}, det)
        self.assertEqual(det.seen, ["cat.jpg"])
        self.assertEqual(stats.detector_examined, 1)

    def test_a_label_removed_from_an_untouched_frame_leaves_the_count_at_zero(self):
        """`pets_found` is what THIS run marked, and a run that measured nothing marked
        nothing — a label the detector takes off such a frame cannot make it negative."""
        self.features(pets=True, detector=False)
        self.run_stage({"fur.jpg": (0.9, True)})
        self.features(pets=True, detector=True, detector_candidates=2,
                      detector_threshold=0.5)
        stats, _clf = self.classify_over({"fur.jpg": (_CAT_IDX, 0.9)},
                                         {"fur.jpg": vector_for(0.9)}, FakeDetector())
        self.assertIsNone(self.pet_of("fur.jpg"))
        self.assertEqual(stats.pets_found, 0)

    def test_a_heuristics_only_run_never_reaches_the_detector(self):
        """No CLIP, no vectors, no label to correct — and no model raised for it."""
        factory = CountingFactory()
        self.add_file("cat.jpg")
        junk.classify(self.cfg, self.conn, use_clip=False, detector_factory=factory,
                      detector_text_encoder=self.encoder)
        self.assertEqual(factory.builds, [])
        self.assertEqual(factory.detector.seen, [])


class TestTheDatabaseContract(unittest.TestCase):
    """What the row means, checked against sqlite rather than against the writer."""

    def test_a_row_without_boxes_is_still_a_row(self):
        """"Examined, nothing found" and "never examined" are different facts, and the
        table has to be able to hold both."""
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "x.db")
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at) "
                "VALUES ('/a.jpg', 1, 0.0, 'jpg', 'photo', 'x')")
            conn.execute(
                "INSERT INTO detections (file_id, label, score, boxes, model, updated_at)"
                " VALUES (1, NULL, NULL, '[]', 'm', 'now')")
            conn.commit()
            row = conn.execute("SELECT * FROM detections").fetchone()
            self.assertIsNone(row["label"])
            self.assertEqual(row["boxes"], "[]")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO detections (file_id, label, score, boxes, model,"
                    " updated_at) VALUES (2, NULL, NULL, NULL, 'm', 'now')")
            conn.close()

    def test_the_module_and_the_stage_spell_the_animal_label_the_same_way(self):
        """`detect.cascade_label` takes the label value from the caller precisely so there
        is one spelling of it; this pins the caller's."""
        self.assertEqual(junk.PET_CLASS, "animal")
        self.assertIs(junk.cascade_label, detect.cascade_label)


if __name__ == "__main__":
    unittest.main()
