"""F154: the tables the threshold and the depth are chosen from — the arithmetic of them.

The script's whole reason for existing is that neither number may be guessed (F130 chose
0.30 in a brief and the measurement made it the worst row of the table), so what has to be
right here is that the numbers mean what the columns say:

* the depth table counts the candidates a depth selects and the share of the KNOWN animals
  that still sit inside it — the ceiling no confidence setting can raise;
* the threshold table replays `detect.best_animal`, so precision and recall are the
  pipeline's own rule and not a paraphrase of it, with the recall denominator being every
  animal in the sample rather than only the ones the detector was shown;
* the baseline row is the label the stage writes today, read off the stored score;
* the worksheet holds file ids and nothing else, stratified over the ranking.

No model, no GPU, no photo: everything below is arithmetic over labels and stored boxes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from sorta.db import connect
from sorta.detect import Detection

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_detector.py"


def _load_script():
    """Import scripts/measure_detector.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_detector", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


measure = _load_script()


def box(label: str, score: float) -> Detection:
    return Detection(label, score, (0.0, 0.0, 10.0, 10.0))


class TestTheDepthTable(unittest.TestCase):
    """What a depth selects, what it costs, and what it can still reach."""

    def rows(self, depths=(2, 4)):
        candidates = [10, 20, 30, 40, 50]
        labels = {10: True, 30: True, 50: True, 20: False}
        return measure.depth_table(candidates, labels, depths)

    def test_the_candidate_count_is_the_head_of_the_ranking(self):
        first, second = self.rows()
        self.assertEqual((first.depth, first.candidates), (2, 2))
        self.assertEqual((second.depth, second.candidates), (4, 4))

    def test_a_depth_past_the_collection_selects_the_collection(self):
        (row,) = self.rows(depths=(500,))
        self.assertEqual(row.candidates, 5)

    def test_the_ceiling_is_the_share_of_known_animals_inside_the_depth(self):
        first, second = self.rows()
        self.assertAlmostEqual(first.ceiling, 1 / 3)   # only file 10 of the three animals
        self.assertAlmostEqual(second.ceiling, 2 / 3)  # 10 and 30

    def test_without_labels_there_is_no_ceiling_to_state(self):
        (row,) = measure.depth_table([1, 2, 3], {}, (2,))
        self.assertIsNone(row.ceiling)

    def test_the_price_is_the_measured_one(self):
        (row,) = measure.depth_table(list(range(2000)), {}, (2000,))
        self.assertAlmostEqual(row.minutes, 2000 * 0.0838 / 60.0, places=6)


class TestTheThresholdTable(unittest.TestCase):
    """Precision and recall, over the stage's own rule rather than a paraphrase."""

    def setUp(self):
        self.boxes = {
            1: [box("cat", 0.9)],     # labelled an animal — right at every threshold here
            2: [box("dog", 0.45)],    # labelled an animal — only found below 0.5
            3: [box("cat", 0.8)],     # NOT an animal — a false positive
            4: [],                    # labelled an animal, nothing detected
        }
        self.labels = {1: True, 2: True, 3: False, 4: True}

    def test_a_higher_threshold_marks_fewer_frames(self):
        low, high = measure.threshold_table(self.boxes, self.labels, (0.3, 0.5))
        self.assertEqual(low.marked, 3)   # 1, 2, 3
        self.assertEqual(high.marked, 2)  # 1, 3

    def test_precision_and_recall_are_counted_the_way_a_human_would(self):
        (row,) = measure.threshold_table(self.boxes, self.labels, (0.5,))
        self.assertAlmostEqual(row.precision, 1 / 2)  # frames 1 and 3 marked, 1 correct
        self.assertAlmostEqual(row.recall, 1 / 3)     # of three animals in the sample

    def test_the_recall_denominator_includes_animals_never_shown_to_the_detector(self):
        """Otherwise the number would flatter a depth that never selected them — which is
        exactly the trade the depth table above is about."""
        (row,) = measure.threshold_table({1: [box("cat", 0.9)]}, self.labels, (0.5,))
        self.assertEqual(row.marked, 1)
        self.assertAlmostEqual(row.recall, 1 / 3)

    def test_an_unlabelled_frame_takes_no_part_in_either_number(self):
        boxes = {**self.boxes, 99: [box("cat", 0.99)]}
        (row,) = measure.threshold_table(boxes, self.labels, (0.5,))
        self.assertEqual(row.marked, 2)

    def test_people_and_food_are_not_animals_here_either(self):
        (row,) = measure.threshold_table({1: [box("person", 0.99)]}, {1: True}, (0.5,))
        self.assertEqual((row.marked, row.correct), (0, 0))

    def test_an_empty_sample_leaves_the_numbers_unstated_rather_than_zero(self):
        (row,) = measure.threshold_table({}, {}, (0.5,))
        self.assertIsNone(row.precision)
        self.assertIsNone(row.recall)


class TestTheBaseline(unittest.TestCase):
    """The label the stage writes today, on the same frames — read off the stored score."""

    def test_the_clip_row_is_the_stage_rule_over_frame_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "x.db")
            for i, (score, label) in enumerate(
                    ((0.9, True), (0.8, False), (0.2, True)), start=1):
                conn.execute(
                    "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)"
                    " VALUES (?, 1, 0.0, 'jpg', 'photo', 'x')", (f"/{i}.jpg",))
                conn.execute(
                    "INSERT INTO frame_quality (file_id, pet_score, source, updated_at)"
                    " VALUES (?, ?, 'clip', 'now')", (i, score))
                del label
            conn.commit()
            row = measure.clip_baseline(conn, {1: True, 2: False, 3: True}, 0.7)
            conn.close()
        self.assertEqual((row.marked, row.correct), (2, 1))  # 0.9 right, 0.8 wrong
        self.assertAlmostEqual(row.precision, 0.5)
        self.assertAlmostEqual(row.recall, 0.5)  # one of the two labelled animals


class TestTheWorksheet(unittest.TestCase):
    """File ids and nothing else, stratified over the ranking — the privacy rule."""

    def test_it_holds_ids_and_nulls_only(self):
        candidates = list(range(1, 3001))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            written = measure.write_sample(path, candidates, per_band=5, seed=1)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(written, len(data))
        self.assertTrue(all(value is None for value in data.values()))
        self.assertTrue(all(key.isdigit() for key in data))

    def test_every_band_of_the_ranking_is_represented(self):
        candidates = list(range(1, 6000))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            measure.write_sample(path, candidates, per_band=4, seed=1)
            picked = [int(k) for k in json.loads(path.read_text(encoding="utf-8"))]
        for low, high in zip(measure.SAMPLE_BANDS, measure.SAMPLE_BANDS[1:]):
            with self.subTest(band=(low, high)):
                self.assertTrue(any(low < file_id <= high for file_id in picked))

    def test_a_worksheet_still_full_of_nulls_measures_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            path.write_text(json.dumps({"1": None, "2": True, "3": False}),
                            encoding="utf-8")
            labels = measure.read_labels(str(path))
        self.assertEqual(labels, {2: True, 3: False})


class TestItPricesTheRealCascade(unittest.TestCase):
    """The script drives the stage's own functions — a private copy would drift."""

    def test_the_population_predicate_is_the_stages_own(self):
        from sorta import junk

        self.assertIn("dup_of IS NULL", junk._DETECTOR_POPULATION_SQL)
        self.assertEqual(measure.detect.best_animal.__module__, "sorta.detect")

    def test_the_default_grid_holds_the_thresholds_the_brief_names(self):
        for threshold in (0.3, 0.5, 0.7):
            self.assertIn(threshold, measure.DEFAULT_THRESHOLDS)

    def test_the_default_depths_bracket_the_configured_one(self):
        from sorta.config import FeaturesConfig

        self.assertIn(FeaturesConfig.detector_candidates, measure.DEFAULT_DEPTHS)
        self.assertLess(min(measure.DEFAULT_DEPTHS), FeaturesConfig.detector_candidates)
        self.assertGreater(max(measure.DEFAULT_DEPTHS),
                           FeaturesConfig.detector_candidates)


if __name__ == "__main__":
    unittest.main()
