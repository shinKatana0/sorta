"""F140: the rescue measurement — the arithmetic behind the table, without a model.

The script exists so that `features.junk_rescue_threshold` is read off a distribution
instead of guessed, and the brief asks for it to print BEFORE the threshold is chosen. So
what is tested is that the tables tell the truth: the sweep counts what somebody counting
by hand would count, the population is the one the stage scores, the price column is the
measured per-frame cost, and a vector of another model never reaches the numbers.

No model, no GPU, no photo — everything below is arithmetic over stored vectors and
aggregates. And, as with every measurement in this project, nothing the script prints may
identify a frame.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sorta import junk
from sorta.db import connect

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_junk_rescue.py"


def _load_script():
    """Import scripts/measure_junk_rescue.py — it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("measure_junk_rescue", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


measure = _load_script()


class TestSweep(unittest.TestCase):
    """The gate replayed: a candidate is a frame whose score REACHES the threshold."""

    def test_counts_what_each_threshold_would_select(self):
        scores = [0.9, 0.05, 0.02, 0.0, -0.4]
        rows = {r.threshold: r for r in measure.sweep(scores, [0.0, 0.02, 0.5])}
        self.assertEqual(rows[0.0].candidates, 4)
        self.assertEqual(rows[0.02].candidates, 3)
        self.assertEqual(rows[0.5].candidates, 1)

    def test_the_threshold_is_inclusive_as_the_stage_has_it(self):
        (row,) = measure.sweep([0.25], [0.25])
        self.assertEqual(row.candidates, 1)

    def test_the_price_is_the_measured_one(self):
        (row,) = measure.sweep([0.9] * 100, [0.5])
        # 100 frames x 0.78 s = 78 s = 1.3 min
        self.assertAlmostEqual(row.minutes, 1.3)
        self.assertAlmostEqual(row.share, 1.0)

    def test_an_empty_population_is_not_a_division_by_zero(self):
        (row,) = measure.sweep([], [0.02])
        self.assertEqual((row.candidates, row.share, row.minutes), (0, 0.0, 0.0))


class TestDistribution(unittest.TestCase):
    def test_nearest_rank_returns_values_that_exist(self):
        values = [v / 100.0 for v in range(1, 101)]
        got = dict(measure.percentiles(values, (50, 100)))
        self.assertIn(got[50], values)
        self.assertAlmostEqual(got[100], 1.0)

    def test_empty_input_is_empty_output(self):
        self.assertEqual(measure.percentiles([]), [])

    def test_the_bands_partition_the_scores(self):
        scores = [-0.4, -0.01, 0.005, 0.015, 0.03, 0.9]
        counts = measure.band_counts(scores)
        self.assertEqual(sum(n for _low, _high, n in counts), len(scores))

    def test_frames_without_a_vector_are_reported_as_null_not_zero(self):
        text = measure.format_distribution([0.1, -0.2], missing=3)
        self.assertIn("без вектора: 3", text)


class TestFormattingIsAnonymous(unittest.TestCase):
    """Privacy: the tables are aggregates. No path, no basename, no file id, ever."""

    def test_no_identifier_reaches_the_output(self):
        rows = measure.sweep([0.4, -0.1], [0.0, 0.02])
        text = "\n".join([measure.format_distribution([0.4, -0.1], missing=1),
                          measure.format_thresholds(rows, 0.02)])
        for forbidden in ("/photos", ".jpg", "file_id"):
            self.assertNotIn(forbidden, text)

    def test_the_configured_threshold_is_marked(self):
        rows = measure.sweep([0.4], [0.0, 0.02])
        self.assertIn("+0.02*", measure.format_thresholds(rows, 0.02))

    def test_the_table_says_it_measures_coverage_and_not_accuracy(self):
        """The one claim this script must never make — the score is a resemblance, and the
        feature exists because applying it as a verdict costs living photographs."""
        text = measure.format_thresholds(measure.sweep([0.4], [0.02]), 0.02)
        self.assertIn("ГЛАЗАМИ", text)


class TestThresholdParsing(unittest.TestCase):
    def test_a_grid_is_sorted_and_deduplicated(self):
        self.assertEqual(measure.parse_thresholds("0.05,0,0.02,0.02"), [0.0, 0.02, 0.05])

    def test_an_empty_grid_is_an_error(self):
        with self.assertRaises(SystemExit):
            measure.parse_thresholds("  ")


class IndexCase(unittest.TestCase):
    """A small index, written by hand: files, verdicts, vectors, stored scores."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "m.db"
        self.conn = connect(self.db)
        self.addCleanup(self.close)
        self.model = "ViT-L-14-quickgelu/openai"

    def close(self):
        self.conn.close()

    def add(self, name, verdict="photo", vec=None, junk_score=None, model=None):
        cur = self.conn.execute(
            "INSERT INTO files (path, size, mtime, ext, media_type, indexed_at)"
            " VALUES (?, 1, 0.0, 'jpg', 'photo', 'x')", (f"/photos/{name}",))
        file_id = cur.lastrowid
        self.conn.execute(
            "INSERT INTO media_class (file_id, verdict, source, updated_at, tier)"
            " VALUES (?, ?, 'clip', 'x', 'clip')", (file_id, verdict))
        if vec is not None:
            packed = junk.pack_embedding(np.asarray(vec, dtype=np.float32))
            self.conn.execute(
                "INSERT INTO clip_embeddings (file_id, model, dim, vec, updated_at)"
                " VALUES (?, ?, ?, ?, 'x')",
                (file_id, model or self.model, len(vec), packed))
        if junk_score is not None:
            self.conn.execute(
                "INSERT INTO frame_quality (file_id, junk_score, source, updated_at)"
                " VALUES (?, ?, 'clip#x', 'x')", (file_id, junk_score))
        self.conn.commit()
        return file_id


class TestPopulation(IndexCase):
    """The frames measured are the ones the stage would score: photographs, canonical."""

    def test_only_photographs_are_measured(self):
        photo = self.add("a.jpg")
        self.add("shot.png", verdict="screenshot")
        conn = measure.open_ro(str(self.db))
        self.addCleanup(conn.close)
        self.assertEqual(measure.photo_ids(conn), [photo])

    def test_the_database_is_opened_read_only(self):
        self.add("a.jpg")
        conn = measure.open_ro(str(self.db))
        self.addCleanup(conn.close)
        with self.assertRaises(Exception):
            conn.execute("DELETE FROM files")


class TestScores(IndexCase):
    """Where the numbers come from — the stored vectors, or a run's stored scores."""

    def features(self):
        positives = len(junk._JUNK_RESCUE_POS_PROMPTS)
        return np.asarray([[1.0, 0.0]] * positives + [[0.0, 1.0]], dtype=np.float32)

    def test_a_score_is_computed_from_the_stored_vector(self):
        junky = self.add("meme.jpg", vec=[1.0, 0.0])
        family = self.add("kids.jpg", vec=[0.0, 1.0])
        conn = measure.open_ro(str(self.db))
        self.addCleanup(conn.close)
        scored = measure.computed_scores(conn, self.model, [junky, family],
                                         self.features())
        self.assertAlmostEqual(scored[junky], 1.0, places=5)
        self.assertAlmostEqual(scored[family], -1.0, places=5)

    def test_a_vector_of_another_model_is_not_measured(self):
        """The filter is `read_clip_embeddings`'s own: two spaces mixed would produce a
        plausible distribution that nothing in the output marks as wrong."""
        other = self.add("old.jpg", vec=[1.0, 0.0], model="ViT-B-32/laion")
        conn = measure.open_ro(str(self.db))
        self.addCleanup(conn.close)
        self.assertEqual(
            measure.computed_scores(conn, self.model, [other], self.features()), {})

    def test_a_frame_without_a_vector_is_simply_absent(self):
        none = self.add("nothing.jpg")
        conn = measure.open_ro(str(self.db))
        self.addCleanup(conn.close)
        self.assertEqual(
            measure.computed_scores(conn, self.model, [none], self.features()), {})

    def test_stored_scores_come_back_for_the_population_only(self):
        photo = self.add("a.jpg", junk_score=0.4)
        self.add("b.jpg", verdict="screenshot", junk_score=0.9)
        conn = measure.open_ro(str(self.db))
        self.addCleanup(conn.close)
        stored = measure.stored_scores(conn, measure.photo_ids(conn))
        self.assertEqual(list(stored), [photo])
        self.assertAlmostEqual(stored[photo], 0.4)


class TestBandWorksheet(unittest.TestCase):
    """The worksheet a review by eye is filled into: stratified, file ids only."""

    def test_it_is_stratified_over_the_score_bands(self):
        scored = {i: 0.9 for i in range(1, 11)}
        scored.update({100 + i: -0.5 for i in range(10)})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bands.json"
            written = measure.write_band_template(path, scored, per_band=3, seed=1)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(written, 6)  # three from each of the two non-empty bands
        self.assertEqual(set(data.values()), {None})

    def test_it_holds_no_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bands.json"
            measure.write_band_template(path, {1: 0.9}, 5, seed=1)
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("photos", text)
        self.assertNotIn(".jpg", text)


class TestItPricesThePipelinesOwnGate(unittest.TestCase):
    """Nothing is reimplemented here: the prompts and the score are the stage's."""

    def test_the_prompts_are_the_stages_own(self):
        self.assertEqual(junk.junk_rescue_prompts(), junk.junk_rescue_prompts())
        self.assertIn(junk._JUNK_RESCUE_NEG_PROMPTS[0], junk.junk_rescue_prompts())

    def test_the_per_frame_cost_is_the_measured_one(self):
        self.assertAlmostEqual(measure.VLM_SECONDS_PER_FRAME, 0.78)


if __name__ == "__main__":
    unittest.main()
