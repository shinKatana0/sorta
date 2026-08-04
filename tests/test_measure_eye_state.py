"""F178 phase 0: the arithmetic the closed-eyes verdict is made of.

The script decides whether a slice of ~948 frames gets built at all, so what has to be
right here is not "the code runs" but that the measurement cannot flatter itself:

* openness is a ratio of the eye's own geometry — a tilted head and a face twice the size
  must give the SAME number, or a threshold over the collection means nothing;
* the 106-point index map is checked against the detector's own eye points, so a model
  file that renumbers its landmarks fails loudly instead of measuring an eyebrow;
* every share is weighted back to the collection through the strata the sample was drawn
  from — read flat, the same labelling says the VLM has 51% recall instead of 9%;
* a fire on a frame the owner could not read counts against precision, which is the
  convention the 60% baseline was computed under;
* the threshold is picked by a rule fixed before the run (`best_row`), the verdict by
  criteria fixed before the run (`decide`), and the crop probe answers out of fold — a
  probe that memorises its training set must not report a good number.

No model, no GPU, no photograph: the landmarks below are drawn on paper and everything
else is arithmetic over labels.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sorta.db import connect

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_eye_state.py"


def _load_script():
    """Import scripts/measure_eye_state.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_eye_state", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


measure = _load_script()

CLOSED = measure.LABEL_CLOSED
OPEN = measure.LABEL_OPEN
CANNOT = measure.LABEL_CANNOT
SAID_CLOSED = measure.STRATUM_SAID_CLOSED
SAID_OPEN = measure.STRATUM_SAID_OPEN


def ring(width: float, height: float, centre=(0.0, 0.0), angle: float = 0.0):
    """Eight points on an ellipse — an eye of a given opening, tilted by `angle`."""
    points = []
    for i in range(8):
        t = 2 * math.pi * i / 8
        x, y = width / 2 * math.cos(t), height / 2 * math.sin(t)
        points.append((centre[0] + x * math.cos(angle) - y * math.sin(angle),
                       centre[1] + x * math.sin(angle) + y * math.cos(angle)))
    return np.asarray(points, dtype=np.float64)


def frame(label, stratum=SAID_OPEN, **scores):
    return measure.Frame(label=label, stratum=stratum, scores=dict(scores))


class TestOpennessIsAShapeNotASize(unittest.TestCase):
    """The eye's own geometry, and nothing about where the head happens to be."""

    def test_a_closed_eye_scores_below_an_open_one(self):
        self.assertLess(measure.eye_openness(ring(20.0, 3.0)),
                        measure.eye_openness(ring(20.0, 9.0)))

    def test_a_face_twice_the_size_gives_the_same_number(self):
        self.assertAlmostEqual(measure.eye_openness(ring(20.0, 6.0)),
                               measure.eye_openness(ring(40.0, 12.0)))

    def test_a_tilted_head_gives_the_same_number(self):
        upright = measure.eye_openness(ring(20.0, 6.0))
        for degrees in (17, 45, 90, 145):
            with self.subTest(degrees=degrees):
                tilted = measure.eye_openness(
                    ring(20.0, 6.0, centre=(300.0, -40.0), angle=math.radians(degrees)))
                self.assertAlmostEqual(upright, tilted, places=6)

    def test_the_number_is_the_opening_over_the_width(self):
        # An ellipse 20 wide and 6 high: the corners are the ends of the long axis and
        # the spread across it is the short one.
        self.assertAlmostEqual(measure.eye_openness(ring(20.0, 6.0)), 0.3, places=6)

    def test_a_ring_collapsed_to_a_point_is_not_measured(self):
        self.assertIsNone(measure.eye_openness(np.zeros((8, 2))))

    def test_too_few_points_are_not_measured(self):
        self.assertIsNone(measure.eye_openness(np.asarray([[0.0, 0.0], [1.0, 1.0]])))
        self.assertIsNone(measure.eye_openness(np.zeros((8,))))


class TestTheLandmarkMapIsChecked(unittest.TestCase):
    """A model file that renumbers its points must fail loudly, not quietly."""

    def landmarks(self, offset=(0.0, 0.0)):
        points = np.zeros((106, 2), dtype=np.float64)
        for indices, centre in zip(measure.EYE_RINGS, ((30.0, 40.0), (70.0, 40.0))):
            shifted = (centre[0] + offset[0], centre[1] + offset[1])
            points[list(indices)] = ring(16.0, 5.0, centre=shifted)
        return points

    def eyes(self):
        return np.asarray([[30.0, 40.0], [70.0, 40.0]], dtype=np.float64)

    def test_rings_over_the_detectors_eyes_agree(self):
        self.assertTrue(measure.eye_rings_agree(self.landmarks(), self.eyes(), 100.0))

    def test_rings_somewhere_else_on_the_face_do_not(self):
        self.assertFalse(
            measure.eye_rings_agree(self.landmarks(offset=(0.0, 25.0)), self.eyes(), 100.0))

    def test_a_shorter_landmark_set_does_not(self):
        self.assertFalse(measure.eye_rings_agree(np.zeros((68, 2)), self.eyes(), 100.0))

    def test_a_face_of_no_width_does_not(self):
        self.assertFalse(measure.eye_rings_agree(self.landmarks(), self.eyes(), 0.0))

    def test_the_two_rings_are_disjoint_and_exclude_the_centres(self):
        left, right = measure.EYE_RINGS
        self.assertFalse(set(left) & set(right))
        self.assertFalse(set(left + right) & set(measure.EYE_CENTRES))


class TestTheCrop(unittest.TestCase):
    """A square on the eye, clamped to the frame, or nothing at all."""

    def test_it_is_centred_on_the_eye(self):
        self.assertEqual(measure.crop_box((50.0, 60.0), 20.0, (200, 200)),
                         (40, 50, 60, 70))

    def test_a_face_at_the_border_still_gets_the_part_that_is_inside(self):
        self.assertEqual(measure.crop_box((4.0, 60.0), 40.0, (200, 200)),
                         (0, 40, 24, 80))

    def test_an_eye_too_small_to_read_gets_no_crop(self):
        self.assertIsNone(measure.crop_box((50.0, 50.0), measure.MIN_CROP_PX - 3.0,
                                           (200, 200)))

    def test_a_crop_clamped_down_to_nothing_gets_no_crop(self):
        self.assertIsNone(measure.crop_box((1.0, 50.0), 16.0, (200, 200)))


class TestBoxOverlap(unittest.TestCase):
    """How a detection is compared with the box the index already holds."""

    def test_the_same_box_overlaps_itself_completely(self):
        self.assertAlmostEqual(measure.box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_boxes_apart_do_not_overlap(self):
        self.assertEqual(measure.box_iou((0, 0, 10, 10), (50, 50, 60, 60)), 0.0)

    def test_a_rotated_coordinate_space_reads_as_no_overlap(self):
        # The finding of this run: `faces.bbox` on a rotated photograph is written in a
        # space where x and y have swapped, and that has to look like a miss, not a match.
        self.assertLess(measure.box_iou((600, 400, 760, 540), (400, 780, 540, 920)), 0.1)

    def test_a_degenerate_box_overlaps_nothing(self):
        self.assertEqual(measure.box_iou((5, 5, 5, 5), (0, 0, 10, 10)), 0.0)


class TestWhoFires(unittest.TestCase):
    """Two of the variants fire above their threshold and one below it."""

    def test_geometry_fires_on_a_small_opening(self):
        self.assertTrue(measure.fires(measure.VARIANT_EAR, 0.10, 0.20))
        self.assertFalse(measure.fires(measure.VARIANT_EAR, 0.30, 0.20))

    def test_the_probabilities_fire_on_a_high_score(self):
        for variant in (measure.VARIANT_CROP, measure.VARIANT_CLIP):
            with self.subTest(variant=variant):
                self.assertTrue(measure.fires(variant, 0.90, 0.80))
                self.assertFalse(measure.fires(variant, 0.10, 0.80))

    def test_a_variant_that_said_nothing_never_fires(self):
        for variant in measure.VARIANTS:
            with self.subTest(variant=variant):
                self.assertFalse(measure.fires(variant, None, 0.5))
                self.assertFalse(measure.fires(variant, None, 1.0))


class TestTheFramesAnswer(unittest.TestCase):
    """Several faces, one frame, one number — and which faces count."""

    def eyes(self):
        return [measure.Eye(face_area=100.0, openness=0.40, crop_closed=0.10),
                measure.Eye(face_area=10.0, openness=0.05, crop_closed=0.95)]

    def test_any_face_with_a_closed_eye_closes_the_frame(self):
        self.assertAlmostEqual(
            measure.frame_score(self.eyes(), measure.VARIANT_EAR, measure.RULE_ANY), 0.05)
        self.assertAlmostEqual(
            measure.frame_score(self.eyes(), measure.VARIANT_CROP, measure.RULE_ANY), 0.95)

    def test_the_portrait_rule_looks_at_the_largest_face_only(self):
        self.assertAlmostEqual(
            measure.frame_score(self.eyes(), measure.VARIANT_EAR, measure.RULE_LARGEST),
            0.40)
        self.assertAlmostEqual(
            measure.frame_score(self.eyes(), measure.VARIANT_CROP, measure.RULE_LARGEST),
            0.10)

    def test_a_variant_none_of_the_eyes_answered_says_nothing(self):
        self.assertIsNone(measure.frame_score(self.eyes(), measure.VARIANT_CLIP,
                                              measure.RULE_ANY))

    def test_a_frame_without_faces_says_nothing(self):
        self.assertIsNone(measure.frame_score((), measure.VARIANT_EAR, measure.RULE_ANY))


class TestTheSampleIsWeightedBack(unittest.TestCase):
    """The stratification is the difference between 51% recall and 9%."""

    def layers(self):
        return measure.strata({SAID_CLOSED: 50, SAID_OPEN: 200},
                              {SAID_CLOSED: 135, SAID_OPEN: 5948})

    def test_a_layer_stands_for_its_share_of_the_collection(self):
        closed, opened = self.layers()
        self.assertAlmostEqual(closed.weight, 135 / 50)
        self.assertAlmostEqual(opened.weight, 5948 / 200)

    def test_a_layer_nobody_labelled_weighs_nothing(self):
        (empty,) = measure.strata({SAID_CLOSED: 0}, {SAID_CLOSED: 135})[:1]
        self.assertEqual(empty.weight, 0.0)

    def test_the_layers_come_out_in_a_fixed_order(self):
        self.assertEqual([layer.name for layer in self.layers()], list(measure.STRATA))

    def test_the_closed_population_is_the_weighted_count(self):
        frames = [frame(CLOSED, SAID_CLOSED), frame(CLOSED, SAID_OPEN), frame(OPEN)]
        w = measure.weights(self.layers())
        self.assertAlmostEqual(measure.closed_population(frames, w),
                               135 / 50 + 5948 / 200)


class TestTheMetrics(unittest.TestCase):
    """Precision, the lenient reading of it, and recall over the population."""

    def frames(self):
        return [
            frame(CLOSED, SAID_CLOSED, ear=0.05),   # found, and right
            frame(OPEN, SAID_CLOSED, ear=0.05),     # found, and wrong
            frame(CANNOT, SAID_CLOSED, ear=0.05),   # found, and unreadable
            frame(CLOSED, SAID_CLOSED, ear=0.40),   # missed
        ]

    def row(self):
        w = {SAID_CLOSED: 10.0, SAID_OPEN: 1.0}
        (row,) = measure.sweep(self.frames(), measure.VARIANT_EAR, w, grid=(0.20,))
        return row

    def test_a_fire_on_an_unreadable_frame_counts_against_precision(self):
        self.assertAlmostEqual(self.row().precision, 1 / 3)

    def test_the_lenient_column_excuses_it(self):
        self.assertAlmostEqual(self.row().lenient_precision, 1 / 2)

    def test_recall_is_over_every_closed_frame_including_the_missed_one(self):
        self.assertAlmostEqual(self.row().recall, 1 / 2)

    def test_the_counts_are_frames_of_the_collection_not_of_the_sample(self):
        row = self.row()
        self.assertAlmostEqual(row.fired, 30.0)
        self.assertAlmostEqual(row.hits, 10.0)
        self.assertAlmostEqual(row.closed, 20.0)

    def test_a_row_that_fires_on_nothing_claims_no_precision(self):
        w = {SAID_CLOSED: 10.0, SAID_OPEN: 1.0}
        (row,) = measure.sweep(self.frames(), measure.VARIANT_EAR, w, grid=(0.01,))
        self.assertEqual((row.fired, row.precision, row.recall), (0.0, 0.0, 0.0))

    def test_the_sweep_covers_the_whole_grid(self):
        w = {SAID_CLOSED: 1.0, SAID_OPEN: 1.0}
        rows = measure.sweep(self.frames(), measure.VARIANT_EAR, w)
        self.assertEqual([r.threshold for r in rows], list(measure.EAR_GRID))


class TestTheBaselineIsTheVlmsOwnAnswers(unittest.TestCase):
    """It fires on the stratum it created, and its numbers are the bar to beat."""

    def frames(self):
        return ([frame(CLOSED, SAID_CLOSED)] * 30 + [frame(OPEN, SAID_CLOSED)] * 11
                + [frame(CANNOT, SAID_CLOSED)] * 9 + [frame(CLOSED, SAID_OPEN)] * 29
                + [frame(OPEN, SAID_OPEN)] * 136 + [frame(CANNOT, SAID_OPEN)] * 34)

    def test_the_measured_baseline_is_the_60_and_9_of_the_brief(self):
        w = measure.weights(measure.strata({SAID_CLOSED: 50, SAID_OPEN: 199},
                                           {SAID_CLOSED: 135, SAID_OPEN: 5948}))
        row = measure.baseline(self.frames(), w)
        self.assertAlmostEqual(row.precision, 0.60, places=2)
        self.assertAlmostEqual(row.recall, 0.09, places=2)
        self.assertAlmostEqual(row.closed, 948, delta=2)

    def test_read_flat_the_same_answers_look_five_times_better(self):
        flat = {SAID_CLOSED: 1.0, SAID_OPEN: 1.0}
        self.assertAlmostEqual(measure.baseline(self.frames(), flat).recall, 30 / 59,
                               places=2)


class TestTheThresholdIsPickedByARule(unittest.TestCase):
    """A bar chosen after seeing the table is not a bar (F131)."""

    def rows(self):
        return [measure.Row(threshold=0.1, fired=10, hits=9, unsure=0, closed=100),
                measure.Row(threshold=0.2, fired=100, hits=60, unsure=0, closed=100),
                measure.Row(threshold=0.3, fired=400, hits=80, unsure=0, closed=100)]

    def test_it_takes_the_most_recall_the_floor_allows(self):
        row = measure.best_row(self.rows(), precision_floor=0.6)
        self.assertEqual(row.threshold, 0.2)

    def test_a_lower_floor_lets_a_wider_threshold_through(self):
        row = measure.best_row(self.rows(), precision_floor=0.2)
        self.assertEqual(row.threshold, 0.3)

    def test_a_floor_nothing_reaches_leaves_no_candidate(self):
        self.assertIsNone(measure.best_row(self.rows(), precision_floor=0.95))

    def test_a_row_that_fires_on_nothing_is_never_the_answer(self):
        empty = measure.Row(threshold=0.0, fired=0, hits=0, unsure=0, closed=100)
        self.assertIsNone(measure.best_row([empty], precision_floor=0.0))


class TestTheVerdict(unittest.TestCase):
    """Three outcomes, and the criteria are the ones written before the run."""

    def base(self, precision=0.6, recall=0.09):
        closed = 1000.0
        hits = closed * recall
        return measure.Row(threshold=0.0, fired=hits / precision, hits=hits,
                           unsure=0.0, closed=closed)

    def candidate(self, recall, variant=measure.VARIANT_EAR):
        return measure.Candidate(
            variant=variant,
            row=measure.Row(threshold=0.2, fired=1000.0 * recall / 0.6,
                            hits=1000.0 * recall, unsure=0.0, closed=1000.0))

    def test_three_times_the_recall_of_the_vlm_goes_to_phase_one(self):
        verdict, why = measure.decide([self.candidate(0.30)], self.base(), 59)
        self.assertEqual(verdict, measure.VERDICT_GO)
        self.assertIn(measure.VARIANT_EAR, why)

    def test_just_under_the_bar_closes_the_topic(self):
        verdict, _why = measure.decide([self.candidate(0.26)], self.base(), 59)
        self.assertEqual(verdict, measure.VERDICT_CLOSE)

    def test_no_variant_holding_the_precision_closes_the_topic(self):
        verdict, why = measure.decide([], self.base(), 59)
        self.assertEqual(verdict, measure.VERDICT_CLOSE)
        self.assertIn("60%", why)

    def test_too_few_labelled_closed_frames_is_not_a_verdict(self):
        verdict, _why = measure.decide([self.candidate(0.90)], self.base(),
                                       measure.MIN_LABELLED_CLOSED - 1)
        self.assertEqual(verdict, measure.VERDICT_UNCLEAR)

    def test_a_sample_without_closed_eyes_is_not_a_verdict(self):
        empty = measure.Row(threshold=0.0, fired=0.0, hits=0.0, unsure=0.0, closed=0.0)
        verdict, _why = measure.decide([], empty, 60)
        self.assertEqual(verdict, measure.VERDICT_UNCLEAR)

    def test_the_bar_moves_with_the_baseline_rather_than_being_a_constant(self):
        # A re-labelling that makes the VLM look better makes the bar harder, by
        # construction: the criterion is a factor, not a number.
        verdict, _why = measure.decide([self.candidate(0.30)], self.base(recall=0.20), 59)
        self.assertEqual(verdict, measure.VERDICT_CLOSE)


class TestTheProbeAnswersOutOfFold(unittest.TestCase):
    """A classifier scored on its own training set is not a measurement."""

    def dataset(self, separable: bool, count: int = 60):
        rng = np.random.default_rng(4)
        labels = [i % 2 for i in range(count)]
        if separable:
            features = np.asarray([[label + rng.normal(0, 0.05), 0.0]
                                   for label in labels])
        else:
            features = rng.normal(0, 1, size=(count, 12))
        groups = list(range(count))
        fold = measure.folds([CLOSED if label else OPEN for label in labels],
                             measure.PROBE_FOLDS, seed=3)
        return features, labels, groups, fold

    def accuracy(self, separable: bool) -> float:
        features, labels, groups, fold = self.dataset(separable)
        answers = measure.probe_predictions(features, labels, groups, fold, seed=3)
        right = sum(1 for label, answer in zip(labels, answers)
                    if answer is not None and (answer >= 0.5) == bool(label))
        return right / len(labels)

    def test_a_signal_it_can_learn_is_learned(self):
        self.assertGreater(self.accuracy(separable=True), 0.9)

    def test_noise_it_cannot_learn_scores_like_noise(self):
        self.assertLess(self.accuracy(separable=False), 0.75)

    def test_every_eye_gets_an_answer(self):
        features, labels, groups, fold = self.dataset(separable=True)
        answers = measure.probe_predictions(features, labels, groups, fold, seed=3)
        self.assertTrue(all(answer is not None for answer in answers))

    def test_eyes_of_one_frame_stay_on_one_side_of_the_split(self):
        # Both eyes of a face are the same picture twice: split between training and
        # test, they would measure the probe's memory of the frame.
        features = np.asarray([[float(i % 2), 0.0] for i in range(8)])
        labels = [i % 2 for i in range(8)]
        groups = [0, 0, 1, 1, 2, 2, 3, 3]
        fold = [0, 1, 0, 1]
        answers = measure.probe_predictions(features, labels, groups, fold, seed=1)
        self.assertEqual(len(answers), 8)

    def test_a_frame_with_no_per_eye_truth_is_answered_never_trained_on(self):
        features = np.asarray([[float(i % 2), 0.0] for i in range(20)])
        labels = [-1 if i < 4 else i % 2 for i in range(20)]
        groups = list(range(20))
        fold = measure.folds([OPEN] * 20, 2, seed=1)
        answers = measure.probe_predictions(features, labels, groups, fold, seed=1)
        self.assertTrue(all(answer is not None for answer in answers[:4]))


class TestTheFolds(unittest.TestCase):
    """Stratified, reproducible, and covering everything exactly once."""

    def test_every_frame_lands_in_exactly_one_fold(self):
        labels = [CLOSED] * 7 + [OPEN] * 13 + [CANNOT] * 5
        fold = measure.folds(labels, 5, seed=1)
        self.assertEqual(len(fold), len(labels))
        self.assertEqual(sorted(set(fold)), [0, 1, 2, 3, 4])

    def test_each_label_is_spread_over_the_folds(self):
        labels = [CLOSED] * 10 + [OPEN] * 10
        fold = measure.folds(labels, 5, seed=1)
        for offset, label in ((0, CLOSED), (10, OPEN)):
            with self.subTest(label=label):
                self.assertEqual(sorted(fold[offset:offset + 10]),
                                 sorted(list(range(5)) * 2))

    def test_the_same_seed_gives_the_same_split(self):
        labels = [CLOSED] * 9 + [OPEN] * 11
        self.assertEqual(measure.folds(labels, 4, seed=2),
                         measure.folds(labels, 4, seed=2))

    def test_one_fold_is_not_a_split(self):
        with self.assertRaises(SystemExit):
            measure.folds([CLOSED, OPEN], 1, seed=1)


class TestTheWorksheet(unittest.TestCase):
    """What the labelling file may say, and what it may not."""

    def read(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return measure.read_labels(str(path))

    def test_the_ids_come_back_as_numbers(self):
        self.assertEqual(self.read({"7": CLOSED, "9": OPEN}), {7: CLOSED, 9: OPEN})

    def test_a_label_outside_the_three_is_refused(self):
        with self.assertRaises(SystemExit):
            self.read({"7": "maybe"})

    def test_an_empty_worksheet_is_refused(self):
        with self.assertRaises(SystemExit):
            self.read({})

    def test_something_that_is_not_a_worksheet_is_refused(self):
        with self.assertRaises(SystemExit):
            self.read([CLOSED, OPEN])

    def test_a_missing_file_is_refused(self):
        with self.assertRaises(SystemExit):
            measure.read_labels(str(Path(tempfile.gettempdir()) / "no-such-file.json"))


class TestThePrice(unittest.TestCase):
    """Milliseconds per FRAME, and only for the work a run does not already do."""

    def timing(self):
        return measure.Timing(decode_s=10.0, frames=100, detect_s=2.0, landmark_s=1.0,
                              faces=200, crop_s=0.5, clip_s=4.0, eyes=400)

    def rows(self):
        return {row.variant: row for row in measure.prices(self.timing())}

    def test_the_landmark_model_is_priced_per_face_and_summed_per_frame(self):
        # 1 s over 200 faces is 5 ms a face, and a frame holds two of them.
        self.assertAlmostEqual(self.rows()[measure.VARIANT_EAR].extra_ms, 10.0)

    def test_the_crop_work_is_priced_per_eye(self):
        self.assertAlmostEqual(self.rows()[measure.VARIANT_CROP].extra_ms, 5.0)
        self.assertAlmostEqual(self.rows()[measure.VARIANT_CLIP].extra_ms, 40.0)

    def test_the_decode_is_not_charged_to_any_variant(self):
        # The junk stage already decodes this preview (F155) — charging it again would
        # price a phase 1 at several times what it costs.
        self.assertNotIn(self.timing().decode_s * 10, [row.extra_ms
                                                       for row in self.rows().values()])

    def test_no_variant_asks_for_new_weights(self):
        self.assertEqual({row.weights_mb for row in self.rows().values()}, {0.0})

    def test_an_empty_run_prices_nothing(self):
        for row in measure.prices(measure.Timing()):
            with self.subTest(variant=row.variant):
                self.assertEqual(row.extra_ms, 0.0)


class TestItReadsTheCollectionItRunsOn(unittest.TestCase):
    """The strata are the database's, not a constant that goes stale after a re-index."""

    def db(self, tmp, answers):
        conn = connect(Path(tmp) / "x.db")
        for i, eyes_open in enumerate(answers, start=1):
            conn.execute(
                "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)"
                " VALUES (?, 1, 0.0, 'jpg', 'photo', 'x')", (f"/{i}.jpg",))
            conn.execute(
                "INSERT INTO frame_quality (file_id, eyes_open, source, updated_at)"
                " VALUES (?, ?, 'clip', 'now')", (i, eyes_open))
        conn.commit()
        return conn

    def test_the_population_is_the_frames_the_question_was_asked_about(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self.db(tmp, [0, 1, 1, None, None])
            found = measure.population_strata(conn)
            conn.close()
        self.assertEqual(found, {SAID_CLOSED: 1, SAID_OPEN: 2})

    def test_a_frame_never_asked_belongs_to_no_stratum(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self.db(tmp, [0, 1, None])
            found = measure.sample_strata(conn, [1, 2, 3])
            conn.close()
        self.assertEqual(found, {1: SAID_CLOSED, 2: SAID_OPEN})


class TestTheReportTellsNothingAboutAnybody(unittest.TestCase):
    """A table about closed eyes must not become a list of who was photographed."""

    def text(self):
        layers = measure.strata({SAID_CLOSED: 2, SAID_OPEN: 4},
                                {SAID_CLOSED: 135, SAID_OPEN: 5948})
        frames = [frame(CLOSED, SAID_CLOSED, ear=0.05, crop=0.9, clip=0.9),
                  frame(OPEN, SAID_CLOSED, ear=0.40, crop=0.1, clip=0.1),
                  frame(CANNOT, SAID_OPEN, ear=0.20, crop=0.5, clip=0.5),
                  frame(CLOSED, SAID_OPEN, ear=0.06, crop=0.8, clip=0.8),
                  frame(OPEN, SAID_OPEN, ear=0.45, crop=0.2, clip=0.2),
                  frame(OPEN, SAID_OPEN, ear=None, crop=None, clip=None)]
        return measure.report(frames, layers, measure.weights(layers),
                              {CLOSED: 2, OPEN: 3, CANNOT: 1},
                              measure.Timing(frames=6, faces=6, eyes=12), measure.RULE_ANY)

    def test_no_path_and_no_file_id_reaches_the_output(self):
        text = self.text()
        self.assertNotIn("/", text.replace("мс/кадр", "").replace("да/нет", ""))
        self.assertNotIn("\\", text)
        self.assertNotIn("file_id", text)

    def test_every_variant_gets_its_own_table_and_the_baseline_under_each(self):
        text = self.text()
        for variant in measure.VARIANTS:
            with self.subTest(variant=variant):
                self.assertIn(f"«{variant}»", text)
        self.assertEqual(text.count("базовая линия"), len(measure.VARIANTS))

    def test_the_verdict_is_printed_with_the_criteria_that_produced_it(self):
        text = self.text()
        self.assertIn("ВЕРДИКТ ФАЗЫ 0", text)
        self.assertIn("best_row", text)

    def test_the_silence_of_a_variant_is_reported_next_to_its_recall(self):
        self.assertIn("Молчание вариантов", self.text())


if __name__ == "__main__":
    unittest.main()
